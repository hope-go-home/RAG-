from SmartQuery.backend.config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_EMBEDDING_MODEL
from langchain_openai import OpenAIEmbeddings
from FlagEmbedding import BGEM3FlagModel


# 兼容模式
embeddings = OpenAIEmbeddings(
    model=QWEN_EMBEDDING_MODEL or "qwen3.7-text-embedding",
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    check_embedding_ctx_length=False,
    chunk_size=20,  # qwen3.7-text-embedding 单次最多 20 条
    dimensions=2048,  # 指定输出维度
)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """qwen 稠密向量（2048 维）——批量入库"""
    if not texts:
        return []
    return embeddings.embed_documents(texts)


def embed_query(text: str) -> list[float]:
    """qwen 稠密向量——单条查询"""
    return embeddings.embed_query(text)


# ---------------- BGE-M3：稠密(1024) + 稀疏，一次 forward 同时产出 ---------------- #

bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)


def embed_documents_bge(texts: list[str]) -> tuple[list[list[float]], list[dict[int, float]]]:
    """批量：返回 (bge 稠密 1024 维, bge 稀疏词汇权重)"""
    if not texts:
        return [], []
    out = bge_model.encode(texts, return_dense=True, return_sparse=True)
    dense = [list(map(float, v)) for v in out["dense_vecs"]]
    sparse = [{int(t): float(w) for t, w in d.items()} for d in out["lexical_weights"]]
    return dense, sparse


def embed_query_bge(text: str) -> tuple[list[float], dict[int, float]]:
    """单条查询：返回 (bge 稠密, bge 稀疏)"""
    out = bge_model.encode([text], return_dense=True, return_sparse=True)
    dense = list(map(float, out["dense_vecs"][0]))
    sparse = {int(t): float(w) for t, w in out["lexical_weights"][0].items()}
    return dense, sparse


# 兼容旧接口（只取稀疏）
def embed_query_sparse(text: str) -> dict[int, float]:
    return embed_query_bge(text)[1]


def embed_documents_sparse(texts: list[str]) -> list[dict[int, float]]:
    return embed_documents_bge(texts)[1]
