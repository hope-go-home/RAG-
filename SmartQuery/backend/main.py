import json
import hashlib
import time
import threading
from fastapi import FastAPI, UploadFile, File, Query, Request, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import uuid

from SmartQuery.backend.logger import get_logger
from SmartQuery.backend.health import health_router
from SmartQuery.backend.database.milvus import connect_milvus, create_collection, delete_by_source
from SmartQuery.backend.database.mysql import (
    init_db,
    find_file_by_hash,
    save_file_record,
    get_user_by_username,
    create_user,
    list_users,
    write_audit,
    list_audit,
    get_document_by_source,
    get_document_by_id,
    list_documents,
    upsert_document,
    soft_delete_document,
)
from SmartQuery.backend.config import ALLOW_REGISTRATION
from SmartQuery.backend.auth import (
    create_access_token,
    verify_password,
    get_current_user,
    require_admin,
)
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


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    department: str = "公共"


class AdminUserRequest(BaseModel):
    username: str
    password: str
    department: str = "公共"
    role: str = "user"


ALLOWED_DEPARTMENTS = {"HR", "财务", "IT", "技术", "公共", "行政"}


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else ""


# ------------------ 认证接口 ------------------ #

@app.post("/auth/login")
def login(request: Request, body: LoginRequest):
    user = get_user_by_username(body.username)
    if not user or not user["is_active"] or not verify_password(body.password, user["password_hash"]):
        write_audit(None, body.username, "login", detail="failed", ip=_client_ip(request))
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(user)
    write_audit(user["id"], user["username"], "login", detail="success", ip=_client_ip(request))
    logger.info("login ok user=%s dept=%s", user["username"], user["department"])
    return {
        "token": token,
        "username": user["username"],
        "department": user["department"],
        "role": user["role"],
    }


@app.get("/auth/me")
def me(user: dict = Depends(get_current_user)):
    return user


@app.get("/auth/audit")
def audit(limit: int = 100, user: dict = Depends(require_admin)):
    """审计日志（仅管理员）"""
    return list_audit(limit)


@app.post("/auth/register")
def register(request: Request, body: RegisterRequest):
    """自助注册：新用户 role=user，部门限制在白名单内"""
    if not ALLOW_REGISTRATION:
        raise HTTPException(status_code=403, detail="系统未开放注册，请联系管理员")
    username = body.username.strip()
    if not (3 <= len(username) <= 32):
        raise HTTPException(status_code=400, detail="用户名长度需为 3-32 位")
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="密码至少 6 位")
    if get_user_by_username(username):
        raise HTTPException(status_code=409, detail="用户名已存在")
    dept = body.department if body.department in ALLOWED_DEPARTMENTS else "公共"
    user_id = create_user(username, body.password, dept, "user")
    write_audit(user_id, username, "register", resource=dept, ip=_client_ip(request))
    logger.info("register ok user=%s dept=%s", username, dept)
    token = create_access_token({"id": user_id, "username": username,
                                 "department": dept, "role": "user"})
    return {"token": token, "username": username, "department": dept, "role": "user"}


@app.get("/auth/users")
def users(limit: int = 200, user: dict = Depends(require_admin)):
    """用户列表（仅管理员，不含密码）"""
    return list_users(limit)


@app.post("/auth/users")
def create_user_admin(request: Request, body: AdminUserRequest,
                      user: dict = Depends(require_admin)):
    """管理员创建用户（可指定部门与角色）"""
    username = body.username.strip()
    if not (3 <= len(username) <= 32):
        raise HTTPException(status_code=400, detail="用户名长度需为 3-32 位")
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="密码至少 6 位")
    if get_user_by_username(username):
        raise HTTPException(status_code=409, detail="用户名已存在")
    if body.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="角色只能是 admin / user")
    dept = body.department if body.department in ALLOWED_DEPARTMENTS else "公共"
    uid = create_user(username, body.password, dept, body.role)
    write_audit(user["id"], user["username"], "create_user",
                resource=username, detail=f"dept={dept} role={body.role}",
                ip=_client_ip(request))
    return {"id": uid, "username": username, "department": dept, "role": body.role}


# ------------------ 启动事件 ------------------ #

@app.on_event("startup")
def startup():
    connect_milvus()
    create_collection()
    init_db()
    os.makedirs(UPLOAD_DIR, exist_ok=True)


# ------------------ 同步 /chat（保留兼容）------------------ #

@app.post("/chat")
def chat(request: Request, question: str = Query(...), session_id: str = Query(default=None),
         user: dict = Depends(get_current_user)):
    sid = session_id or uuid.uuid4().hex[:8]
    logger.info("chat start session=%s user=%s question=%s", sid, user["username"], question[:100])
    initial_state = _build_initial_state(sid, question, user["department"])
    result = rag_agent.invoke(initial_state)
    _persist_session(sid, question, result)
    write_audit(user["id"], user["username"], "chat", resource=question[:120], ip=_client_ip(request))
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
async def chat_stream(http_request: Request, request: ChatRequest,
                      user: dict = Depends(get_current_user)):
    sid = request.session_id or uuid.uuid4().hex[:8]
    initial_state = _build_initial_state(sid, request.question, user["department"])

    async def event_generator():
        accumulated = {}  # 累积所有节点的输出，避免后一节点覆盖前一节点的字段
        in_answer = False  # 只在 generate/direct_answer 节点中推送 token
        NODES = {"query_analysis", "retrieve", "rewrite", "generate", "direct_answer"}
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
            write_audit(user["id"], user["username"], "chat",
                        resource=request.question[:120], ip=_client_ip(http_request))
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

SUPPORTED_EXTS = {"pdf", "docx", "txt", "md", "xlsx", "png", "jpg", "jpeg"}


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@app.post("/upload")
async def upload(request: Request, files: list[UploadFile] = File(...),
                 doc_type: str = Form(default="员工手册"),
                 department: str = Form(default="公共"),
                 user: dict = Depends(require_admin)):
    results = []
    logger.info("upload start files=%d user=%s", len(files), user["username"])
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

        source = file.filename
        save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.{ext}")
        with open(save_path, "wb") as f:
            f.write(content)
        file_hash = _file_sha256(save_path)

        existing = get_document_by_source(source)
        if existing and existing["file_hash"] == file_hash:
            os.remove(save_path)
            logger.info("upload unchanged file=%s hash=%s", source, file_hash[:12])
            results.append({
                "file": source,
                "status": "duplicate",
                "message": f"内容未变化（v{existing['version']}，{existing['chunk_count']} 块），跳过",
            })
            continue

        try:
            if existing:
                # 增量更新：先删除旧块，再重新入库
                delete_by_source(source)
            count = ingest_file(save_path, doc_type=doc_type, department=department, source=source)
            rec = upsert_document(source, save_path, file_hash, doc_type, department, count)
            logger.info("upload %s file=%s chunks=%d version=%d",
                        "updated" if existing else "success", source, count, rec["version"])
            results.append({
                "file": source,
                "status": "updated" if existing else "success",
                "message": f"{'更新' if existing else '成功入库'} {count} 块（v{rec['version']}）",
            })
        except Exception as e:
            logger.error("upload failed file=%s error=%s", source, e, exc_info=True)
            results.append({"file": source, "status": "failed", "message": f"入库失败：{e}"})

    success = sum(1 for r in results if r["status"] in ("success", "updated"))
    duplicate = sum(1 for r in results if r["status"] == "duplicate")
    failed = sum(1 for r in results if r["status"] == "failed")
    write_audit(user["id"], user["username"], "upload",
                resource=",".join(f.filename for f in files),
                detail=f"success={success} duplicate={duplicate} failed={failed}",
                ip=_client_ip(request))
    return {
        "message": f"批量入库完成：成功 {success} 个，重复跳过 {duplicate} 个，失败 {failed} 个",
        "results": results,
    }


# ------------------ 文档管理（生命周期）------------------ #

@app.get("/documents")
def documents(limit: int = 200, user: dict = Depends(get_current_user)):
    """已入库文档列表"""
    return list_documents(limit)


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, request: Request, user: dict = Depends(require_admin)):
    """删除文档：软删除登记记录 + 从 Milvus 移除全部块"""
    info = get_document_by_id(doc_id)
    if not info or info["status"] != "active":
        raise HTTPException(status_code=404, detail="文档不存在")
    delete_by_source(info["source"])
    soft_delete_document(doc_id)
    write_audit(user["id"], user["username"], "delete", resource=info["source"], ip=_client_ip(request))
    logger.info("document deleted source=%s by=%s", info["source"], user["username"])
    return {"status": "deleted", "source": info["source"]}


@app.post("/documents/{doc_id}/reindex")
def reindex_document(doc_id: int, request: Request, user: dict = Depends(require_admin)):
    """重建索引：删除旧块后按原始文件重新入库"""
    info = get_document_by_id(doc_id)
    if not info or info["status"] != "active":
        raise HTTPException(status_code=404, detail="文档不存在")
    if not info["stored_path"] or not os.path.exists(info["stored_path"]):
        raise HTTPException(status_code=400, detail="原始文件不存在，无法重建索引")
    delete_by_source(info["source"])
    count = ingest_file(info["stored_path"], doc_type=info["doc_type"],
                        department=info["department"], source=info["source"])
    rec = upsert_document(info["source"], info["stored_path"], info["file_hash"] or "",
                          info["doc_type"], info["department"], count)
    write_audit(user["id"], user["username"], "reindex", resource=info["source"], ip=_client_ip(request))
    logger.info("document reindexed source=%s chunks=%d version=%d",
                info["source"], count, rec["version"])
    return {"status": "reindexed", "chunk_count": count, "version": rec["version"]}


# ------------------ /history 不变 ------------------ #

@app.get("/history/{session_id}")
def history(session_id: str, user: dict = Depends(get_current_user)):
    from SmartQuery.backend.database.mysql import get_history
    return get_history(session_id)


@app.get("/sessions")
def sessions(limit: int = 50, user: dict = Depends(get_current_user)):
    """历史会话列表（按末次时间倒序）"""
    from SmartQuery.backend.database.mysql import list_sessions
    return list_sessions(limit)


# ------------------ 内部辅助函数 ------------------ #

def _build_initial_state(sid: str, question: str, department: str | None = None) -> dict:
    """构建 Agent 初始状态，复用已有 session 的上下文"""
    state: dict = {"question": question, "department": department or "公共"}

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
