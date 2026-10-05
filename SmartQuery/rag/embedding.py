"""嵌入层：稠密用 qwen（DashScope，2048 维），稀疏用 BAAI/bge-m3。

实测对比：本语料（英文 WixQA）上 qwen 稠密明显优于 bge-m3 稠密，故稠密走 qwen。
"""
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


# ---------------- BGE-M3：稀疏词汇权重 ---------------- #

sparse_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)


def embed_documents_sparse(texts: list[str]) -> list[dict[int, float]]:
    if not texts:
        return []
    out = sparse_model.encode(texts, return_sparse=True, return_dense=False)
    return [{int(t): float(w) for t, w in d.items()} for d in out["lexical_weights"]]


def embed_query_sparse(text: str) -> dict[int, float]:
    out = sparse_model.encode([text], return_sparse=True, return_dense=False)
    return {int(t): float(w) for t, w in out["lexical_weights"][0].items()}
