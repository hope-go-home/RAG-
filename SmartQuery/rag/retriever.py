from dataclasses import dataclass, field
from SmartQuery.rag.embedding import embed_query,embed_query_sparse  #只用 embed_query 和 embed_query_sparse，因它只处理用户问题的向量化。
from SmartQuery.backend.database.milvus import search_dense,search_sparse
from sentence_transformers import CrossEncoder


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

def rrf_fusion(dense_results: list[tuple[str, str, float]], sparse_results: list[tuple[str, str, float]], k: int = 60) -> list[str]:
    doc_scores = {}
    for rank, (text, parent_text, _) in enumerate(dense_results):
        doc_scores[text] = {"parent_text": parent_text, "score": 1 / (k + rank + 1)}
    for rank, (text, parent_text, _) in enumerate(sparse_results):
        if text in doc_scores:
            doc_scores[text]["score"] += 1 / (k + rank + 1)
        else:
            doc_scores[text] = {"parent_text": parent_text, "score": 1 / (k + rank + 1)}
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]["score"], reverse=True)  #  d = {"a": 1, "b": 2}，print(d.items())  # dict_items([('a', 1), ('b', 2)])
#对 doc_scores 中的键值对进行排序。doc_scores.items() 返回 (text, value_dict) 元组的可迭代对象。
# 排序依据是每个 value_dict 中的 "score" 字段，reverse=True 表示按分数从高到低降序排列。结果存入 sorted_docs

    return [item[1]["parent_text"] or item[0] for item in sorted_docs]
#item[0]对应子块，item[1]["parent_text"]对应父块


#实例化交叉编码器，加载预训练模型 BAAI/bge-reranker-v2-m3，用于对查询与文档进行相关性重排序。
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

#定义 rerank 函数，接收查询字符串和文档文本列表，返回重排序后的文档列表。
def rerank(query: str, documents: list[str]) -> list[str]:
    pairs = [[query, doc] for doc in documents]    #构造查询与每个文档的配对列表，格式为 [query, doc]
    scores = reranker.predict(pairs)   #调用交叉编码器预测所有配对的相似度得分，返回得分列表（与 pairs 顺序一致
    scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)  # reverse=True从大到小
    #  将文档与对应得分打包，按得分降序排序，得到 [(doc, score), ...]。
    return [doc for doc, _ in scored]  #提取排序后的文档文本（不返回得分）

#定义主检索函数 retrieve  参数：question 查询文本，top_k 最终返回的文档数量（默认5），partition_name Milvus 分区名（可选）
def retrieve(question: str, top_k: int = 5, partition_name: str | None = None) -> list[str]:
    query_vector = embed_query(question)  #调用 embed_query 将问题转为密集向量。
    query_sparse = embed_query_sparse(question)
    dense_results = search_dense(query_vector, top_k=top_k * 2, partition_name=partition_name)
    sparse_results = search_sparse(query_sparse, top_k=top_k * 2, partition_name=partition_name)
    fused = rrf_fusion(dense_results, sparse_results)
    return rerank(question, fused)[:top_k]  #调用 rerank 对融合后的所有文档进行重排序，然后取前 top_k 个作为最终结果返回


# --------------- 带元数据的检索（Agentic RAG 用）--------------- #
# 返回 RetrievalResult，包含每篇文档的 dense/sparse 命中排名、RRF 分数、rerank 分数
# 前端可据此展示"检索链路"：这篇文档来自稠密第 3 名 + 稀疏第 7 名 → RRF 融合 → rerank 排第 1

def retrieve_with_meta(question: str, top_k: int = 10, partition_name: str | None = None) -> RetrievalResult:
    """执行混合检索并返回带元数据的结果"""
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

    # RRF 融合（同时记录每个文档的 RRF 分数）
    doc_rrf_scores: dict[str, dict] = {}  # text → {"parent_text": str, "score": float}
    k = 60
    for rank, (text, parent_text, _) in enumerate(dense_results):
        doc_rrf_scores[text] = {"parent_text": parent_text, "score": 1 / (k + rank + 1)}
    for rank, (text, parent_text, _) in enumerate(sparse_results):
        if text in doc_rrf_scores:
            doc_rrf_scores[text]["score"] += 1 / (k + rank + 1)
        else:
            doc_rrf_scores[text] = {"parent_text": parent_text, "score": 1 / (k + rank + 1)}

    sorted_by_rrf = sorted(doc_rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    fused_docs = [item[1]["parent_text"] or item[0] for item in sorted_by_rrf]
    fused_texts = [item[0] for item in sorted_by_rrf]  # 子块文本列表

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

    return RetrievalResult(
        documents=[s.parent_text for s in sources],
        sources=sources,
        dense_hit_count=len(dense_results),
        sparse_hit_count=len(sparse_results),
        fused_count=len(fused_docs),
    )



