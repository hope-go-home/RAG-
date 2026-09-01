import json
import hashlib
import time
import threading
from fastapi import FastAPI, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import uuid

from SmartQuery.backend.logger import get_logger
from SmartQuery.backend.health import health_router
from SmartQuery.backend.database.milvus import connect_milvus, create_collection
from SmartQuery.backend.database.mysql import init_db, find_file_by_hash, save_file_record
from SmartQuery.rag.agent import app as rag_agent
from SmartQuery.rag.ingest import ingest_file, get_partition

logger = get_logger(__name__)

UPLOAD_DIR = "data/docs"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app = FastAPI(title="Agentic RAG")
app.include_router(health_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def access_log(request: Request, call_next):
    """访问日志：记录每个请求的方法、路径、状态码、耗时"""
    start = time.perf_counter()
    try:
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "request %s %s -> %s (%.1fms)",
            request.method, request.url.path, response.status_code, elapsed,
        )
        return response
    except Exception:
        elapsed = (time.perf_counter() - start) * 1000
        logger.exception("request %s %s failed (%.1fms)", request.method, request.url.path, elapsed)
        raise

session_contexts = {}   # 存储每个 session 的上下文 + 对话历史，用于多轮对话
session_locks = {}      # 每个 session 的锁
session_locks_global = threading.Lock()  # 全局锁（保护 session_locks 字典）


# ------------------ 请求模型 ------------------ #

class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


# ------------------ 启动事件 ------------------ #

@app.on_event("startup")
def startup():
    connect_milvus()
    create_collection()
    init_db()
    os.makedirs(UPLOAD_DIR, exist_ok=True)


# ------------------ 同步 /chat（保留兼容）------------------ #

@app.post("/chat")
def chat(question: str = Query(...), session_id: str = Query(default=None)):
    sid = session_id or uuid.uuid4().hex[:8]
    logger.info("chat start session=%s question=%s", sid, question[:100])
    initial_state = _build_initial_state(sid, question)
    result = rag_agent.invoke(initial_state)
    _persist_session(sid, question, result)
    logger.info("chat done session=%s answer_len=%d", sid, len(result.get("answer", "")))
    return {
        "session_id": sid,
        "question": question,
        "answer": result.get("answer", ""),
    }


# ------------------ SSE 流式 /chat/stream ------------------ #
# 前端通过 SSE 实时接收思考步骤和最终答案，无需等全部完成

def _sse(event: str, data: dict) -> str:
    """格式化 SSE 事件，type 同时写入 data 供前端解析"""
    data["type"] = event  # 前端通过 JSON 里的 type 字段来分发
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    sid = request.session_id or uuid.uuid4().hex[:8]
    initial_state = _build_initial_state(sid, request.question)

    async def event_generator():
        accumulated = {}  # 累积所有节点的输出，避免后一节点覆盖前一节点的字段
        in_answer = False  # 只在 generate/direct_answer 节点中推送 token
        NODES = {"query_analysis", "retrieve", "generate", "direct_answer"}
        try:
            async for event in rag_agent.astream_events(initial_state, version="v2"):
                kind = event["event"]
                name = event.get("name", "")

                # token 级流式：只在 generate / direct_answer 中推送每个 token
                if kind == "on_chat_model_stream" and in_answer:
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        yield _sse("token", {"content": chunk.content})

                # 节点开始：标记是否进入答案生成节点
                elif kind == "on_chain_start" and name in ("generate", "direct_answer"):
                    in_answer = True

                # 节点结束：处理节点输出
                elif kind == "on_chain_end" and name in NODES:
                    if name in ("generate", "direct_answer"):
                        in_answer = False
                    output = event.get("data", {}).get("output", {})
                    if not isinstance(output, dict):
                        continue
                    accumulated.update(output)

                    steps = output.get("thinking_steps", [])
                    if steps:
                        yield _sse("thinking", {"node": name, "info": steps[-1].get("info", "")})

                    sources = output.get("retrieval_sources")
                    if sources:
                        yield _sse("sources", {"sources": sources[:10]})

                    stats = output.get("retrieval_stats")
                    if stats:
                        yield _sse("stats", stats)

            # 从累积状态取答案（generate 节点产出）
            answer = accumulated.get("answer", "")
            if not answer or not answer.strip():
                answer = "抱歉，无法生成回答。请尝试换一种方式提问。"
            yield _sse("answer", {"answer": answer})

            _persist_session(sid, request.question, accumulated)
            yield _sse("done", {"session_id": sid})

        except Exception as e:
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ------------------ /upload 批量上传 + 去重入库 ------------------ #
# 支持一次传多个文件；按内容 SHA-256 哈希查重，重复文件自动跳过
# 请求格式：multipart/form-data，字段名 "files"（兼容单文件字段 "file"）

SUPPORTED_EXTS = {"pdf", "docx", "txt", "md"}


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    results = []
    logger.info("upload start files=%d", len(files))
    for file in files:
        # 检查文件大小
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            results.append({
                "file": file.filename,
                "status": "failed",
                "message": f"文件大小超过限制（{len(content) // 1024 // 1024}MB > 10MB）"
            })
            continue

        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in SUPPORTED_EXTS:
            logger.warning("upload reject unsupported ext=%s file=%s", ext, file.filename)
            results.append({"file": file.filename, "status": "failed", "message": f"不支持的文件类型 .{ext}"})
            continue

        save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.{ext}")
        with open(save_path, "wb") as f:
            f.write(content)  # 使用已读取的内容

        file_hash = _file_sha256(save_path)
        existing = find_file_by_hash(file_hash)
        if existing:
            os.remove(save_path)  # 清理已存在文件的临时副本
            logger.info("upload duplicate file=%s hash=%s", file.filename, file_hash[:12])
            results.append({
                "file": file.filename,
                "status": "duplicate",
                "message": f"已入库过（{existing['chunk_count']} 块，{existing['created_at']}），跳过",
            })
            continue

        try:
            count = ingest_file(save_path)
            partition = get_partition(save_path)
            save_file_record(file_hash, file.filename, partition, count)
            logger.info("upload success file=%s chunks=%d partition=%s", file.filename, count, partition)
            results.append({"file": file.filename, "status": "success", "message": f"成功入库 {count} 个文档块"})
        except Exception as e:
            logger.error("upload failed file=%s error=%s", file.filename, e, exc_info=True)
            results.append({"file": file.filename, "status": "failed", "message": f"入库失败：{e}"})

    success = sum(1 for r in results if r["status"] == "success")
    duplicate = sum(1 for r in results if r["status"] == "duplicate")
    failed = sum(1 for r in results if r["status"] == "failed")
    return {
        "message": f"批量入库完成：成功 {success} 个，重复跳过 {duplicate} 个，失败 {failed} 个",
        "results": results,
    }


# ------------------ /history 不变 ------------------ #

@app.get("/history/{session_id}")
def history(session_id: str):
    from SmartQuery.backend.database.mysql import get_history
    return get_history(session_id)


# ------------------ 内部辅助函数 ------------------ #

def _build_initial_state(sid: str, question: str) -> dict:
    """构建 Agent 初始状态，复用已有 session 的上下文"""
    state: dict = {"question": question}

    # 获取或创建 session 锁
    with session_locks_global:
        if sid not in session_locks:
            session_locks[sid] = threading.Lock()
        lock = session_locks[sid]

    with lock:
        if sid in session_contexts:
            ctx = session_contexts[sid]
            state["context"] = ctx.get("context", [])
            state["chat_history"] = ctx.get("history", [])
    return state


def _persist_session(sid: str, question: str, result: dict):
    """持久化 session 上下文 + 聊天记录"""
    answer = result.get("answer", "")

    # 获取或创建 session 锁
    with session_locks_global:
        if sid not in session_locks:
            session_locks[sid] = threading.Lock()
        lock = session_locks[sid]

    with lock:
        # 更新内存中的 session 上下文
        docs = result.get("context") or []
        history = session_contexts.get(sid, {}).get("history", [])
        history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        session_contexts[sid] = {
            "context": docs,
            "history": history[-20:],  # 最多保留 20 轮
        }
    # 存 MySQL
    from SmartQuery.backend.database.mysql import save_record
    save_record(sid, question, answer)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("SmartQuery.backend.main:app", host="0.0.0.0", port=8000)
