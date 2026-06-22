from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid

from SmartQuery.backend.database.milvus import connect_milvus, create_collection
from SmartQuery.backend.database.mysql import init_db
from SmartQuery.rag.agent import app as rag_agent
from SmartQuery.rag.ingest import ingest_file

UPLOAD_DIR = "data/docs"

app = FastAPI(title="SmartQuery")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_contexts = {}   # 存储每个 session 的上一次 context + history，用于 follow_up 追问


@app.on_event("startup")
def startup():
    connect_milvus()
    create_collection()
    init_db()
    os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/chat")
def chat(question: str = Query(...), session_id: str = Query(default=None)):
    sid = session_id or uuid.uuid4().hex[:8]
    initial_state = {"question": question}
    if sid in session_contexts:
        initial_state["context"] = session_contexts[sid]["context"]
        initial_state["chat_history"] = session_contexts[sid]["history"]

    result = rag_agent.invoke(initial_state)

    if result.get("context"):
        session_contexts[sid] = {
            "context": result["context"],
            "history": [
                *session_contexts.get(sid, {}).get("history", []),
                {"role": "user", "content": question},
                {"role": "assistant", "content": result.get("answer", "")},
            ],
        }

    return {"session_id": sid, "question": question, "answer": result["answer"]}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1]
    save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.{ext}")
    with open(save_path, "wb") as f:
        f.write(file.file.read())
    count = ingest_file(save_path)
    return {"message": f"成功入库 {count} 个文档块", "file": file.filename}


@app.get("/history/{session_id}")
def history(session_id: str):
    from SmartQuery.backend.database.mysql import get_history
    return get_history(session_id)
