"""Prometheus 指标
================
集中定义可观测指标，供 /metrics 端点与各模块打点使用。

- HTTP 层：请求数、延迟（按方法/路由模板/状态）
- 检索层：检索耗时、重排耗时、每查询命中数
- LLM 层：token 计数（按 prompt/completion）
"""
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ---------------- HTTP ----------------
HTTP_REQUESTS = Counter(
    "http_requests_total", "HTTP 请求总数", ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP 请求耗时（秒）", ["method", "path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
)

# ---------------- 检索 ----------------
RETRIEVAL_SECONDS = Histogram(
    "rag_retrieval_seconds", "单次检索+重排耗时（秒）",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)
RETRIEVAL_HITS = Histogram(
    "rag_retrieval_hits", "单次返回的来源数",
    buckets=(0, 1, 2, 3, 5, 10, 20),
)

# ---------------- LLM ----------------
LLM_TOKENS = Counter(
    "rag_llm_tokens_total", "LLM token 计数", ["kind"],  # prompt / completion
)


def metrics_response():
    """生成 Prometheus 文本响应内容"""
    return generate_latest(), CONTENT_TYPE_LATEST
