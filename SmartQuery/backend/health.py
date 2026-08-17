"""健康检查模块：/health 存活探针 + /ready 就绪探针（依赖探测）

独立成模块便于维护和扩展，main.py 通过 include_router 挂载。
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from SmartQuery.backend.logger import get_logger
from SmartQuery.backend.database.milvus import COLLECTION_NAME

logger = get_logger(__name__)

health_router = APIRouter(tags=["health"])


@health_router.get("/health")
def health():
    """存活探针：服务进程是否活着（K8s livenessProbe 用）"""
    return {"status": "ok", "service": "Agentic RAG"}


@health_router.get("/ready")
def ready():
    """就绪探针：依赖组件（Milvus / MySQL / 嵌入模型）是否可用（K8s readinessProbe 用）"""
    checks = {"milvus": False, "mysql": False, "embedding": False}

    try:
        from pymilvus import utility
        checks["milvus"] = utility.has_collection(COLLECTION_NAME)
    except Exception as e:
        logger.warning("ready check milvus failed: %s", e)

    try:
        from sqlalchemy import text
        from SmartQuery.backend.database.mysql import SessionLocal
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        checks["mysql"] = True
    except Exception as e:
        logger.warning("ready check mysql failed: %s", e)

    try:
        from SmartQuery.rag.embedding import embed_query
        embed_query("ping")
        checks["embedding"] = True
    except Exception as e:
        logger.warning("ready check embedding failed: %s", e)

    all_ready = all(checks.values())
    logger.info("ready check result=%s %s", "ok" if all_ready else "degraded", checks)
    if all_ready:
        return {"status": "ready", "checks": checks}
    return JSONResponse(status_code=503, content={"status": "not_ready", "checks": checks})