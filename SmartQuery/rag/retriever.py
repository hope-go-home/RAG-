from dataclasses import dataclass, field
import hashlib
import time
import threading
from SmartQuery.rag.embedding import embed_query,embed_query_sparse
from SmartQuery.backend.database.milvus import search_dense,search_sparse
from SmartQuery.backend.config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL
from SmartQuery.backend.logger import get_logger
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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
    source: str = ""


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

def _get_cache_key(question: str, top_k: int, doc_type: str | None = None,
                   departments: list[str] | None = None,
                   collection_name: str | None = None) -> str:
    """生成缓存键（必须包含过滤条件，否则跨部门/跨集合会串味，造成权限绕过）"""
    dept_key = ",".join(sorted(departments)) if departments else ""
    raw = f"{question}:{top_k}:{doc_type or ''}:{dept_key}:{collection_name or ''}"
    return hashlib.md5(raw.encode()).hexdigest()

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

# 加权 RRF 参数（稠密权重高于稀疏，避免稀疏噪声稀释稠密强排序）
RRF_K = 30
RRF_DENSE_WEIGHT = 0.7
RRF_SPARSE_WEIGHT = 0.3


# ---------------- Multi-Query：LLM 生成查询变体提升召回 ---------------- #

_mq_llm = ChatOpenAI(model=QWEN_MODEL, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """你是检索查询扩展助手。针对用户问题，生成 {n} 个语义等价但表达不同的检索查询，用于提升召回率。

要求：
- 覆盖同义词、上位词、口语化表达、相关术语
- 每行一个查询，不要编号、不要解释、不要引号

用户问题：{question}

输出（每行一个查询）："""
)


def _rrf_scored(dense_results: list[tuple[str, str, float]], sparse_results: list[tuple[str, str, float]],
                k: int = RRF_K, dense_weight: float = RRF_DENSE_WEIGHT,
                sparse_weight: float = RRF_SPARSE_WEIGHT) -> list[tuple[str, str, float]]:
    """加权 RRF 融合（返回带分数结果，供 rrf_fusion 和 retrieve_with_meta 共用）

    参数 k 控制排名衰减速度：k 越小，靠前排名权重越大（更激进）。
    权重说明：本语料上稠密检索的排序质量高于稀疏，等权融合会让稀疏的噪声
    稀释稠密的强排序（实测 Recall@1 从 0.78 掉到 0.58），因此给稠密更高权重。
    """
    doc_scores: dict[str, dict] = {}
    for rank, item in enumerate(dense_results):
        text, parent_text = item[0], item[1]
        doc_scores[text] = {"parent_text": parent_text,
                            "score": dense_weight / (k + rank + 1)}
    for rank, item in enumerate(sparse_results):
        text, parent_text = item[0], item[1]
        contrib = sparse_weight / (k + rank + 1)
        if text in doc_scores:
            doc_scores[text]["score"] += contrib
        else:
            doc_scores[text] = {"parent_text": parent_text, "score": contrib}
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
                       doc_type: str | None = None, departments: list[str] | None = None,
                       collection_name: str | None = None) -> RetrievalResult:
    """执行混合检索并返回带元数据的结果"""
    # 检查缓存
    cache_key = _get_cache_key(question, top_k, doc_type, departments, collection_name)
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
    # 组合过滤条件：文档类型 + 部门权限（department in [本部门, 公共]）
    exprs = []
    if doc_type:
        exprs.append(f'doc_type == "{doc_type}"')
    if departments:
        quoted = ", ".join(f'"{d}"' for d in departments)
        exprs.append(f"department in [{quoted}]")
    expr = " and ".join(exprs) if exprs else None
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

    # 子块文本 → 来源文件名（用于引用溯源）
    source_map: dict[str, str] = {}
    for item in list(dense_results) + list(sparse_results):
        if len(item) > 4 and item[4]:
            source_map.setdefault(item[0], item[4])

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
            source=source_map.get(child_text, ""),
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


# --------------- Multi-Query 检索（多查询变体提升召回）--------------- #

def _generate_query_variants(question: str, n: int = 3) -> list[str]:
    """用 LLM 生成 n 个查询变体（不含原问题）；失败则返回空列表，退化为单查询"""
    try:
        resp = _mq_llm.invoke(MULTI_QUERY_PROMPT.format_messages(question=question, n=n))
        lines = [ln.strip().strip("-•*\t ").strip() for ln in resp.content.split("\n")]
        return [ln for ln in lines if len(ln) >= 2 and ln != question][:n]
    except Exception as e:
        logger.warning("multi_query 生成失败，退化为单查询：%s", e)
        return []


def retrieve_multi_query(question: str, top_k: int = 10, doc_type: str | None = None,
                         departments: list[str] | None = None, collection_name: str | None = None,
                         n_variants: int = 3) -> RetrievalResult:
    """Multi-Query 检索：原问题 + LLM 生成的多个查询变体，各自混合检索后 RRF 合并，最后统一重排。

    动机：单一查询的表达偏差会造成漏召回；多查询从不同表述切入，合并后提升召回上限。
    成本控制：变体检索不逐次重排，只在合并后重排一次，避免 N 倍重排开销。
    """
    queries = [question] + _generate_query_variants(question, n_variants)

    exprs = []
    if doc_type:
        exprs.append(f'doc_type == "{doc_type}"')
    if departments:
        quoted = ", ".join(f'"{d}"' for d in departments)
        exprs.append(f"department in [{quoted}]")
    expr = " and ".join(exprs) if exprs else None
    search_kwargs = {"collection_name": collection_name} if collection_name else {}

    rank_lists: list[list[str]] = []
    child_to_parent: dict[str, str] = {}
    child_to_source: dict[str, str] = {}
    for q in queries:
        qvec = embed_query(q)
        qsparse = embed_query_sparse(q)
        dense = search_dense(qvec, top_k=top_k * 2, expr=expr, **search_kwargs)
        sparse = search_sparse(qsparse, top_k=top_k * 2, expr=expr, **search_kwargs)
        fused = _rrf_scored(dense, sparse)
        rank_lists.append([text for text, _, _ in fused])
        for text, parent, _ in fused:
            child_to_parent.setdefault(text, parent or text)
        for item in list(dense) + list(sparse):
            if len(item) > 4 and item[4]:
                child_to_source.setdefault(item[0], item[4])

    # 跨查询 RRF 合并
    scores: dict[str, float] = {}
    for rl in rank_lists:
        for rank, text in enumerate(rl):
            scores[text] = scores.get(text, 0.0) + 1.0 / (RRF_K + rank + 1)
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # 去重到父块，保持顺序
    parent_order: list[str] = []
    parent_child: dict[str, str] = {}
    for text, _ in merged:
        parent = child_to_parent.get(text, text)
        if parent not in parent_child:
            parent_child[parent] = text
            parent_order.append(parent)

    # 候选池取足够大（多查询的价值在扩大候选集，不能过早截断）
    candidates = parent_order[: max(top_k * 6, 40)]
    pairs = [[question, doc] for doc in candidates]
    rerank_scores = reranker.predict(pairs) if candidates else []
    ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)[:top_k]

    sources = [
        RetrievalSource(text=parent_child[p], parent_text=p, dense_rank=0, sparse_rank=0,
                        rrf_score=round(scores.get(parent_child[p], 0.0), 6),
                        rerank_score=round(float(s), 4),
                        source=child_to_source.get(parent_child[p], ""))
        for p, s in ranked
    ]
    logger.info("retrieve_multi_query queries=%d candidates=%d top=%d",
                len(queries), len(parent_order), len(sources))
    return RetrievalResult(
        documents=[s.parent_text for s in sources],
        sources=sources,
        fused_count=len(parent_order),
    )



