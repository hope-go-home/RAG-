from dataclasses import dataclass, field
from collections import OrderedDict
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
    title: str = ""
    url: str = ""


@dataclass
class RetrievalResult:
    """检索结果的结构化包装，携带元数据供前端展示"""
    documents: list[str]
    sources: list[RetrievalSource] = field(default_factory=list)
    dense_hit_count: int = 0
    sparse_hit_count: int = 0
    fused_count: int = 0


# ---------------- 检索缓存（LRU + TTL，防止无上限增长） ---------------- #
_retrieval_cache: "OrderedDict[str, tuple[float, RetrievalResult]]" = OrderedDict()
_cache_lock = threading.Lock()
CACHE_TTL = 300
CACHE_MAXSIZE = 512

def _get_cache_key(question: str, top_k: int, doc_type: str | None = None,
                   collection_name: str | None = None, extra: str = "") -> str:
    """生成缓存键（含过滤条件，避免跨类型/跨集合串味）"""
    raw = f"{question}:{top_k}:{doc_type or ''}:{collection_name or ''}:{extra}"
    return hashlib.md5(raw.encode()).hexdigest()

def _get_from_cache(key: str) -> RetrievalResult | None:
    """从缓存获取结果（命中即刷新 LRU 顺序）"""
    with _cache_lock:
        item = _retrieval_cache.get(key)
        if item is None:
            return None
        cached_time, cached_result = item
        if time.time() - cached_time >= CACHE_TTL:
            del _retrieval_cache[key]
            return None
        _retrieval_cache.move_to_end(key)
        return cached_result

def _put_to_cache(key: str, result: RetrievalResult) -> None:
    """存入缓存，超出容量时淘汰最久未使用项"""
    with _cache_lock:
        _retrieval_cache[key] = (time.time(), result)
        _retrieval_cache.move_to_end(key)
        while len(_retrieval_cache) > CACHE_MAXSIZE:
            _retrieval_cache.popitem(last=False)


# ---------------- 核心检索逻辑 ---------------- #

# 加权 RRF 参数：WixQA(英文) 上稠密明显强于稀疏，故给稠密更高权重
RRF_K = 60
RRF_DENSE_WEIGHT = 0.8
RRF_SPARSE_WEIGHT = 0.2

# 重排候选池大小：每个通道取回多少条参与 RRF/重排
CANDIDATE_POOL = 60
# 重排最多处理的候选文章数（CPU 交叉编码器较慢，限制上限以控制延迟）
RERANK_POOL = 30
# 重排输入截断长度（256 在质量与 CPU 延迟间较平衡；CPU 交叉编码器很吃时间）
RERANK_MAX_LENGTH = 256


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
    稠密/稀疏权重默认等权，可按语料在评测中调优。
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


def rrf_fusion(dense_results: list[tuple[str, str, float]], sparse_results: list[tuple[str, str, float]], k: int = 60) -> list[str]:
    """RRF 融合，返回按融合分数降序的父块文本列表"""
    return [parent_text or text for text, parent_text, _ in _rrf_scored(dense_results, sparse_results, k)]


#实例化交叉编码器，加载预训练模型 BAAI/bge-reranker-v2-m3，用于对查询与文档进行相关性重排序。
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", local_files_only=True)
reranker.max_length = RERANK_MAX_LENGTH  # 控制 CPU 推理延迟


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

# 英文停用词（WixQA 等英文语料用；避免 "how / do / i / the" 之类污染稀疏检索）
STOP_WORDS |= {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "am", "do", "does", "did", "doing", "have", "has", "had", "having",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "this", "that", "these", "those",
    "and", "or", "but", "if", "then", "else", "so", "because", "as", "of",
    "to", "in", "on", "at", "by", "for", "with", "about", "into", "from",
    "up", "down", "out", "over", "under", "again", "can", "could", "will",
    "would", "should", "shall", "may", "might", "must", "not", "no", "yes",
    "how", "what", "when", "where", "why", "which", "who", "whom", "whose",
    "there", "here", "please", "my", "any", "some", "all", "very", "just",
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
                       doc_type: str | None = None,
                       collection_name: str | None = None,
                       candidate_pool: int | None = None,
                       rerank_pool: int | None = None) -> RetrievalResult:
    """执行混合检索并返回带元数据的结果。

    candidate_pool：每个通道取回的候选数（默认 CANDIDATE_POOL）
    rerank_pool：送入重排的候选父块数（默认 RERANK_POOL）
    """
    pool_cfg = max(top_k * 2, candidate_pool or CANDIDATE_POOL)
    rp_cfg = rerank_pool or RERANK_POOL
    cache_key = _get_cache_key(question, top_k, doc_type, collection_name, f"{pool_cfg}:{rp_cfg}")
    cached_result = _get_from_cache(cache_key)
    if cached_result:
        logger.info("retrieve_with_meta cache_hit query=%s", question[:60])
        return cached_result

    t0 = time.perf_counter()

    # 稠密用原始问题（语义完整），稀疏用去停用词后的关键词
    processed_query = preprocess_query(question)
    query_vector = embed_query(question)
    query_sparse = embed_query_sparse(processed_query)
    # 过滤条件：可选按文档类型（article_type）限定
    expr = f'doc_type == "{doc_type}"' if doc_type else None
    search_kwargs = {"collection_name": collection_name} if collection_name else {}
    dense_results = search_dense(query_vector, top_k=pool_cfg, partition_name=partition_name,
                                 expr=expr, **search_kwargs)
    sparse_results = search_sparse(query_sparse, top_k=pool_cfg, partition_name=partition_name,
                                   expr=expr, **search_kwargs)

    # 子块文本 → 排名 + 元数据 {doc_type, source, title, url}
    dense_rank_map: dict[str, int] = {}
    sparse_rank_map: dict[str, int] = {}
    meta_map: dict[str, tuple[str, str, str, str]] = {}
    for rank, item in enumerate(dense_results, start=1):
        dense_rank_map[item[0]] = rank
        meta_map.setdefault(item[0], (item[3], item[4], item[5], item[6]))
    for rank, item in enumerate(sparse_results, start=1):
        sparse_rank_map[item[0]] = rank
        meta_map.setdefault(item[0], (item[3], item[4], item[5], item[6]))

    # RRF 融合（复用 _rrf_scored，避免算法重复维护）
    scored = _rrf_scored(dense_results, sparse_results)
    doc_rrf_scores = {text: score for text, _, score in scored}

    # 去重到「文章(source)」再送重排：每篇文章取融合分最高的子块，
    # 保证更多不同文章进入重排（避免同篇多块挤占候选）
    seen_sources: set[str] = set()
    candidates: list[tuple[str, str]] = []   # (parent_text, child_text)
    for text, parent_text, _ in scored:
        src = meta_map.get(text, ("", "", "", ""))[1] or (parent_text or text)
        if src in seen_sources:
            continue
        seen_sources.add(src)
        candidates.append((parent_text or text, text))
        if len(candidates) >= rp_cfg:
            break

    # Rerank
    pairs = [[question, doc] for doc, _ in candidates]
    rerank_scores = reranker.predict(pairs, batch_size=16) if candidates else []
    ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
    top = [(doc, child_text, float(score)) for (doc, child_text), score in ranked][:top_k]

    # 组装 RetrievalSource 列表
    sources = []
    for doc, child_text, rerank_score in top:
        dt, src, title, url = meta_map.get(child_text, ("", "", "", ""))
        sources.append(RetrievalSource(
            text=child_text,
            parent_text=doc,
            dense_rank=dense_rank_map.get(child_text),
            sparse_rank=sparse_rank_map.get(child_text),
            rrf_score=round(doc_rrf_scores.get(child_text, 0.0), 6),
            rerank_score=round(float(rerank_score), 4),
            doc_type=dt,
            source=src,
            title=title,
            url=url,
        ))

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "retrieve_with_meta query=%s dense=%d sparse=%d fused=%d cand=%d top=%d (%.1fms)",
        question[:60], len(dense_results), len(sparse_results), len(scored), len(candidates),
        len(top), elapsed,
    )

    result = RetrievalResult(
        documents=[s.parent_text for s in sources],
        sources=sources,
        dense_hit_count=len(dense_results),
        sparse_hit_count=len(sparse_results),
        fused_count=len(scored),
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


HYDE_PROMPT = ChatPromptTemplate.from_template(
    """你是企业知识库助手。请针对下面的问题，写一段可能是标准答案的短文
（用知识库文档的口吻陈述，即使不确定也要写得具体），用于语义检索。

问题：{question}

只输出这段短文，不要解释、不要加标题。"""
)


def _hyde_text(question: str) -> str:
    """HyDE：生成假设答案文本，用它的向量去检索（提升语义召回）"""
    try:
        resp = _mq_llm.invoke(HYDE_PROMPT.format_messages(question=question))
        return (resp.content or "").strip()[:1600]
    except Exception as e:
        logger.warning("hyde 生成失败：%s", e)
        return ""


def retrieve_multi_query(question: str, top_k: int = 10, doc_type: str | None = None,
                         collection_name: str | None = None,
                         n_variants: int = 3, use_hyde: bool = True,
                         candidate_pool: int | None = None,
                         rerank_pool: int | None = None) -> RetrievalResult:
    """增强检索：原问题 + 查询变体 + HyDE 假设答案，各自混合检索后 RRF 合并，按文章去重后统一重排。

    动机：
    - 单一查询的表达偏差会造成漏召回 → 多查询从不同表述切入
    - 问题与文档措辞不同（"问题像问题、文档像陈述"）→ HyDE 用假设答案靠拢文档分布
    - 同一篇文章的多个块会霸占候选 → 按 source 去重，让更多不同文章进入重排
    """
    pool_cfg = max(top_k * 2, candidate_pool or CANDIDATE_POOL)
    rp_cfg = rerank_pool or RERANK_POOL

    # 检索用的查询集合：原问题 + 变体 + 假设答案
    queries = [question] + _generate_query_variants(question, n_variants)
    hyde = _hyde_text(question) if use_hyde else ""
    retrieval_queries = queries + ([hyde] if hyde else [])

    expr = f'doc_type == "{doc_type}"' if doc_type else None
    search_kwargs = {"collection_name": collection_name} if collection_name else {}

    rank_lists: list[list[str]] = []
    child_to_parent: dict[str, str] = {}
    child_meta: dict[str, tuple[str, str, str, str]] = {}
    for q in retrieval_queries:
        qvec = embed_query(q)
        qsparse = embed_query_sparse(q)
        dense = search_dense(qvec, top_k=pool_cfg, expr=expr, **search_kwargs)
        sparse = search_sparse(qsparse, top_k=pool_cfg, expr=expr, **search_kwargs)
        fused = _rrf_scored(dense, sparse)
        rank_lists.append([text for text, _, _ in fused])
        for text, parent, _ in fused:
            child_to_parent.setdefault(text, parent or text)
        for item in list(dense) + list(sparse):
            child_meta.setdefault(item[0], (item[3], item[4], item[5], item[6]))

    # 跨查询 RRF 合并
    scores: dict[str, float] = {}
    for rl in rank_lists:
        for rank, text in enumerate(rl):
            scores[text] = scores.get(text, 0.0) + 1.0 / (RRF_K + rank + 1)
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # 按 source（文章）去重，每篇取融合分最高的子块
    seen_src: set[str] = set()
    cand_children: list[str] = []
    for text, _ in merged:
        src = child_meta.get(text, ("", "", "", ""))[1] or child_to_parent.get(text, text)
        if src in seen_src:
            continue
        seen_src.add(src)
        cand_children.append(text)
        if len(cand_children) >= rp_cfg:
            break

    # 重排（用原问题，对被选中的子块）
    cand_parents = [child_to_parent.get(c, c) for c in cand_children]
    pairs = [[question, doc] for doc in cand_parents]
    rerank_scores = reranker.predict(pairs, batch_size=16) if cand_parents else []
    ranked = sorted(zip(cand_children, cand_parents, rerank_scores), key=lambda x: x[2], reverse=True)[:top_k]

    sources = []
    for child, parent, score in ranked:
        dt, src, title, url = child_meta.get(child, ("", "", "", ""))
        sources.append(RetrievalSource(
            text=child, parent_text=parent, dense_rank=0, sparse_rank=0,
            rrf_score=round(scores.get(child, 0.0), 6),
            rerank_score=round(float(score), 4),
            doc_type=dt, source=src, title=title, url=url))
    logger.info("retrieve_multi_query queries=%d(+hyde=%s) candidates=%d top=%d",
                len(queries), bool(hyde), len(cand_children), len(sources))
    return RetrievalResult(
        documents=[s.parent_text for s in sources],
        sources=sources,
        fused_count=len(cand_children),
    )



