from dataclasses import dataclass, field
import hashlib
import time
import threading
from SmartQuery.rag.embedding import embed_query,embed_query_sparse
from SmartQuery.backend.database.milvus import search_dense,search_sparse
from SmartQuery.backend.logger import get_logger
from sentence_transformers import CrossEncoder

logger = get_logger(__name__)


# ---------------- 数据结构 ---------------- #

@dataclass
class RetrievalSource:
    """单篇检索来源的元数据"""
    text: str
    parent_text: str
    dense_rank: int
    sparse_rank: int
    rrf_score: float
    rerank_score: float
    doc_type: str = ""


@dataclass
class RetrievalResult:
    """检索结果的结构化包装，携带元数据供前端展示"""
    documents: list[str]
    sources: list[RetrievalSource] = field(default_factory=list)
    dense_hit_count: int = 0
    sparse_hit_count: int = 0
    fused_count: int = 0


# ---------------- 检索缓存 ---------------- #
_retrieval_cache: dict[str, tuple[float, RetrievalResult]] = {}
_cache_lock = threading.Lock()
CACHE_TTL = 300

def _get_cache_key(question: str, top_k: int) -> str:
    """生成缓存键"""
    return hashlib.md5(f"{question}:{top_k}".encode()).hexdigest()

def _get_from_cache(key: str) -> RetrievalResult | None:
    """从缓存获取结果"""
    with _cache_lock:
        if key in _retrieval_cache:
            cached_time, cached_result = _retrieval_cache[key]
            if time.time() - cached_time < CACHE_TTL:
                return cached_result
            else:
                del _retrieval_cache[key]
    return None

def _put_to_cache(key: str, result: RetrievalResult) -> None:
    """存入缓存"""
    with _cache_lock:
        _retrieval_cache[key] = (time.time(), result)


# ---------------- 核心检索逻辑 ---------------- #

def _rrf_scored(dense_results: list[tuple[str, str, float]], sparse_results: list[tuple[str, str, float]], k: int = 30) -> list[tuple[str, str, float]]:
    """RRF 融合（返回带分数结果，供 rrf_fusion 和 retrieve_with_meta 共用）
    
    参数 k 控制排名衰减速度：
    - k 越小，排名靠前的文档权重越大（更激进）
    - k 越大，排名差异被平滑（更保守）
    - 默认从 60 调整到 30，提升融合效果
    """
    doc_scores: dict[str, dict] = {}
    for rank, item in enumerate(dense_results):
        text, parent_text = item[0], item[1]
        doc_scores[text] = {"parent_text": parent_text, "score": 1 / (k + rank + 1)}
    for rank, item in enumerate(sparse_results):
        text, parent_text = item[0], item[1]
        if text in doc_scores:
            doc_scores[text]["score"] += 1 / (k + rank + 1)
        else:
            doc_scores[text] = {"parent_text": parent_text, "score": 1 / (k + rank + 1)}
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    return [(text, item["parent_text"], item["score"]) for text, item in sorted_docs]


def rrf_fusion(dense_results: list[tuple[str, str, float]], sparse_results: list[tuple[str, str, float]], k: int = 30) -> list[str]:
    """RRF 融合，返回按融合分数降序的父块文本列表"""
    return [parent_text or text for text, parent_text, _ in _rrf_scored(dense_results, sparse_results, k)]


#实例化交叉编码器，加载预训练模型 BAAI/bge-reranker-v2-m3，用于对查询与文档进行相关性重排序。
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", local_files_only=True)


# --------------- 查询预处理（提升召回）--------------- #

# 中文停用词（高频低语义词）
STOP_WORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "他", "她", "它", "们", "那", "个", "被", "从", "对", "但", "以",
    "而", "与", "或", "等", "之", "把", "被", "让", "给", "用", "按", "通过", "根据",
    "什么", "怎么", "如何", "为什么", "哪些", "哪个", "哪里", "请问", "吗", "呢",
    "啊", "呀", "吧", "嗯", "哦", "哈", "呵", "嘿", "喂", "哎",
}


def preprocess_query(question: str) -> str:
    """查询预处理：去除停用词、标准化格式，提升检索效果"""
    if not question:
        return question

    # 1. 去除特殊字符（保留中文、英文、数字）
    question = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', question)

    # 2. 去除多余空格
    question = re.sub(r'\s+', ' ', question).strip()

    # 3. 分词并去除停用词（简单实现：按空格分词）
    words = question.split()
    filtered_words = [w for w in words if w not in STOP_WORDS and len(w) > 1]

    # 4. 如果过滤后太短，保留原查询
    if len(filtered_words) < 2:
        return question

    return " ".join(filtered_words)


import re


# --------------- 带元数据的检索（Agentic RAG 用）--------------- #
# 返回 RetrievalResult，包含每篇文档的 dense/sparse 命中排名、RRF 分数、rerank 分数
# 前端可据此展示"检索链路"：这篇文档来自稠密第 3 名 + 稀疏第 7 名 → RRF 融合 → rerank 排第 1

def retrieve_with_meta(question: str, top_k: int = 10, partition_name: str | None = None,
                       doc_type: str | None = None, collection_name: str | None = None) -> RetrievalResult:
    """执行混合检索并返回带元数据的结果"""
    # 检查缓存
    cache_key = _get_cache_key(question, top_k)
    cached_result = _get_from_cache(cache_key)
    if cached_result:
        logger.info("retrieve_with_meta cache_hit query=%s", question[:60])
        return cached_result

    t0 = time.perf_counter()

    # 查询预处理：去除停用词、标准化格式，提升检索效果
    processed_query = preprocess_query(question)
    logger.info("retrieve_with_meta original=%s processed=%s", question[:40], processed_query[:40])

    query_vector = embed_query(processed_query)
    query_sparse = embed_query_sparse(processed_query)
    expr = f'doc_type == "{doc_type}"' if doc_type else None
    search_kwargs = {"collection_name": collection_name} if collection_name else {}
    dense_results = search_dense(query_vector, top_k=top_k * 2, partition_name=partition_name,
                                 expr=expr, **search_kwargs)
    sparse_results = search_sparse(query_sparse, top_k=top_k * 2, partition_name=partition_name,
                                   expr=expr, **search_kwargs)

    # 构建稠密排名映射：子块文本 → (稠密排名, 父块文本)
    dense_rank_map: dict[str, tuple[int, str]] = {}
    for rank, item in enumerate(dense_results, start=1):
        text, parent_text = item[0], item[1]
        dense_rank_map[text] = (rank, parent_text)

    # 构建稀疏排名映射
    sparse_rank_map: dict[str, tuple[int, str]] = {}
    for rank, item in enumerate(sparse_results, start=1):
        text, parent_text = item[0], item[1]
        sparse_rank_map[text] = (rank, parent_text)

    # 子块文本 → 文档类型（用于前端按类型着色）
    doc_type_map: dict[str, str] = {}
    for item in list(dense_results) + list(sparse_results):
        if len(item) > 3 and item[3]:
            doc_type_map.setdefault(item[0], item[3])

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
            doc_type=doc_type_map.get(child_text, ""),
        ))

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "retrieve_with_meta query=%s dense=%d sparse=%d fused=%d top=%d (%.1fms)",
        question[:60], len(dense_results), len(sparse_results), len(fused_docs), len(top), elapsed,
    )

    result = RetrievalResult(
        documents=[s.parent_text for s in sources],
        sources=sources,
        dense_hit_count=len(dense_results),
        sparse_hit_count=len(sparse_results),
        fused_count=len(fused_docs),
    )

    # 存入缓存
    _put_to_cache(cache_key, result)

    return result



