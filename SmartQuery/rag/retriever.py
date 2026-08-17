from dataclasses import dataclass, field
import time
from SmartQuery.rag.embedding import embed_query,embed_query_sparse  #只用 embed_query 和 embed_query_sparse，因它只处理用户问题的向量化。
from SmartQuery.backend.database.milvus import search_dense,search_sparse
from SmartQuery.backend.logger import get_logger
from sentence_transformers import CrossEncoder

logger = get_logger(__name__)


# ---------------- 结构化返回 ---------------- #
# 每次检索的每个文档附带元数据，前端可展示来源追溯

@dataclass
class RetrievalSource:
    """单篇检索来源的元数据"""
    text: str                # 子块文本
    parent_text: str         # 父块文本（实际喂给 LLM 的上下文）
    dense_rank: int | None   # 在稠密结果中的排名（1-based），未命中为 None
    sparse_rank: int | None  # 在稀疏结果中的排名（1-based），未命中为 None
    rrf_score: float         # RRF 融合后的分数
    rerank_score: float      # 交叉编码器重排序分数


@dataclass
class RetrievalResult:
    """检索结果的结构化包装，携带元数据供前端展示"""
    documents: list[str]                  # 父块文本列表（最终喂给 LLM）
    sources: list[RetrievalSource] = field(default_factory=list)
    dense_hit_count: int = 0              # 稠密检索命中数
    sparse_hit_count: int = 0             # 稀疏检索命中数
    fused_count: int = 0                  # RRF 融合后的文档数


# ---------------- 核心检索逻辑 ---------------- #

def _rrf_scored(dense_results: list[tuple[str, str, float]], sparse_results: list[tuple[str, str, float]], k: int = 60) -> list[tuple[str, str, float]]:
    """RRF 融合（返回带分数结果，供 rrf_fusion 和 retrieve_with_meta 共用）"""
    doc_scores: dict[str, dict] = {}
    for rank, (text, parent_text, _) in enumerate(dense_results):
        doc_scores[text] = {"parent_text": parent_text, "score": 1 / (k + rank + 1)}
    for rank, (text, parent_text, _) in enumerate(sparse_results):
        if text in doc_scores:
            doc_scores[text]["score"] += 1 / (k + rank + 1)
        else:
            doc_scores[text] = {"parent_text": parent_text, "score": 1 / (k + rank + 1)}
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    return [(text, item["parent_text"], item["score"]) for text, item in sorted_docs]


def rrf_fusion(dense_results: list[tuple[str, str, float]], sparse_results: list[tuple[str, str, float]], k: int = 60) -> list[str]:
    """RRF 融合，返回按融合分数降序的父块文本列表"""
    return [parent_text or text for text, parent_text, _ in _rrf_scored(dense_results, sparse_results, k)]


#实例化交叉编码器，加载预训练模型 BAAI/bge-reranker-v2-m3，用于对查询与文档进行相关性重排序。
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")


# --------------- 带元数据的检索（Agentic RAG 用）--------------- #
# 返回 RetrievalResult，包含每篇文档的 dense/sparse 命中排名、RRF 分数、rerank 分数
# 前端可据此展示"检索链路"：这篇文档来自稠密第 3 名 + 稀疏第 7 名 → RRF 融合 → rerank 排第 1

def retrieve_with_meta(question: str, top_k: int = 10, partition_name: str | None = None) -> RetrievalResult:
    """执行混合检索并返回带元数据的结果"""
    t0 = time.perf_counter()
    query_vector = embed_query(question)
    query_sparse = embed_query_sparse(question)
    dense_results = search_dense(query_vector, top_k=top_k * 2, partition_name=partition_name)
    sparse_results = search_sparse(query_sparse, top_k=top_k * 2, partition_name=partition_name)

    # 构建稠密排名映射：子块文本 → (稠密排名, 父块文本)
    dense_rank_map: dict[str, tuple[int, str]] = {}
    for rank, (text, parent_text, _) in enumerate(dense_results, start=1):
        dense_rank_map[text] = (rank, parent_text)

    # 构建稀疏排名映射
    sparse_rank_map: dict[str, tuple[int, str]] = {}
    for rank, (text, parent_text, _) in enumerate(sparse_results, start=1):
        sparse_rank_map[text] = (rank, parent_text)

    # RRF 融合（复用 _rrf_scored，避免算法重复维护）
    scored = _rrf_scored(dense_results, sparse_results)
    fused_docs = [parent_text or text for text, parent_text, _ in scored]
    fused_texts = [text for text, _, _ in scored]
    doc_rrf_scores = {text: {"parent_text": parent_text, "score": score} for text, parent_text, score in scored}

    # Rerank
    pairs = [[question, doc] for doc in fused_docs]
    rerank_scores = reranker.predict(pairs) if fused_docs else []
    scored = sorted(zip(fused_docs, fused_texts, rerank_scores), key=lambda x: x[2], reverse=True)
    top = scored[:top_k]

    # 组装 RetrievalSource 列表
    sources = []
    for doc, child_text, rerank_score in top:
        _, dense_rank = dense_rank_map.get(child_text, (None, None)) or (None, None)
        _, sparse_rank = sparse_rank_map.get(child_text, (None, None)) or (None, None)
        rrf_score = doc_rrf_scores.get(child_text, {}).get("score", 0.0)
        sources.append(RetrievalSource(
            text=child_text,
            parent_text=doc,
            dense_rank=dense_rank,
            sparse_rank=sparse_rank,
            rrf_score=round(rrf_score, 6),
            rerank_score=round(float(rerank_score), 4),
        ))

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "retrieve_with_meta query=%s dense=%d sparse=%d fused=%d top=%d (%.1fms)",
        question[:60], len(dense_results), len(sparse_results), len(fused_docs), len(top), elapsed,
    )

    return RetrievalResult(
        documents=[s.parent_text for s in sources],
        sources=sources,
        dense_hit_count=len(dense_results),
        sparse_hit_count=len(sparse_results),
        fused_count=len(fused_docs),
    )



