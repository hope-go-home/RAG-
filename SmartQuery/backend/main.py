import json
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import uuid

from SmartQuery.backend.database.milvus import connect_milvus, create_collection
from SmartQuery.backend.database.mysql import init_db
from SmartQuery.rag.agent import app as rag_agent
from SmartQuery.rag.ingest import ingest_file

UPLOAD_DIR = "data/docs"

app = FastAPI(title="Agentic RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_contexts = {}   # 存储每个 session 的上下文 + 对话历史，用于多轮对话


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
    initial_state = _build_initial_state(sid, question)
    result = rag_agent.invoke(initial_state)
    _persist_session(sid, question, result)

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
        try:
            async for chunk in rag_agent.astream(initial_state):
                for node_name, node_output in chunk.items():
                    accumulated.update(node_output)  # 累积，不覆盖

                    # 推送思考步骤
                    steps = node_output.get("thinking_steps", [])
                    if steps:
                        latest = steps[-1]
                        yield _sse("thinking", {
                            "node": node_name,
                            "info": latest.get("info", ""),
                        })

                    # 推送检索来源
                    sources = node_output.get("retrieval_sources")
                    if sources:
                        yield _sse("sources", {"sources": sources[:10]})

                    # 推送检索统计（dense/sparse/RRF 各阶段命中数）
                    stats = node_output.get("retrieval_stats")
                    if stats:
                        yield _sse("stats", stats)

                    # 推送文档评分
                    grades = node_output.get("document_grades")
                    if grades:
                        yield _sse("grades", {
                            "grades": grades,
                            "need_retrieve": node_output.get("need_retrieve", False),
                        })

                    # 推送反思结果
                    reflection = node_output.get("reflection_result")
                    if reflection:
                        yield _sse("reflection", reflection)

            # 从累积状态取答案（generate 节点产出，不会被 reflect 覆盖）
            answer = accumulated.get("answer", "")
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


# ------------------ /upload 不变 ------------------ #

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1]
    save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.{ext}")
    with open(save_path, "wb") as f:
        f.write(file.file.read())
    count = ingest_file(save_path)
    return {"message": f"成功入库 {count} 个文档块", "file": file.filename}


# ------------------ /history 不变 ------------------ #

@app.get("/history/{session_id}")
def history(session_id: str):
    from SmartQuery.backend.database.mysql import get_history
    return get_history(session_id)


# ------------------ 内部辅助函数 ------------------ #

def _build_initial_state(sid: str, question: str) -> dict:
    """构建 Agent 初始状态，复用已有 session 的上下文"""
    state: dict = {"question": question}
    if sid in session_contexts:
        ctx = session_contexts[sid]
        state["context"] = ctx.get("context", [])
        state["chat_history"] = ctx.get("history", [])
    return state


def _persist_session(sid: str, question: str, result: dict):
    """持久化 session 上下文 + 聊天记录"""
    answer = result.get("answer", "")
    # 更新内存中的 session 上下文
    # 新 Agent 用 relevant_docs 作为下次的 context，比原始 context 更精准
    docs = result.get("relevant_docs") or result.get("context") or []
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
