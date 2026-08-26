import json
from typing import TypedDict
from SmartQuery.rag.retriever import retrieve_with_meta, RetrievalResult
from SmartQuery.backend.logger import get_logger
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from SmartQuery.backend.config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL
from langgraph.graph import StateGraph, END
from tenacity import retry, stop_after_attempt, wait_exponential

logger = get_logger(__name__)


# ==================== State 定义 ====================

class GraphState(TypedDict):
    # ------ 核心字段 ------
    question: str               # 用户原始问题
    context: list[str]          # 检索到的父块文档列表
    answer: str                 # LLM 生成的回答
    intent: str                 # 意图分类：rag_query / chit_chat / meta / follow_up
    chat_history: list[dict]    # 多轮对话历史 [{role, content}, ...]

    # ------ 检索控制 ------
    retrieval_count: int            # 已经检索了几次
    max_retrieval_attempts: int     # 最大检索次数，默认 1（简化后）
    need_retrieve: bool             # 文档评分不足时置为 True，触发重检

    # ------ 前端可见 ------
    thinking_steps: list[dict]      # [{node, status, info}, ...] 前端思考面板展示
    retrieval_sources: list[dict]   # 每篇文档的检索元数据（dense/sparse 排名、分数）
    retrieval_stats: dict           # {dense_hits, sparse_hits, fused_count, reranked_count}


# ==================== LLM 实例 ====================

llm = ChatOpenAI(
    model=QWEN_MODEL,
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
)


# ==================== 工具函数 ====================

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=3))
def safe_llm_invoke(messages):
    """带重试的 LLM 调用"""
    return llm.invoke(messages)



def _format_history(history: list[dict], max_turns: int = 6) -> str:
    """将对话历史格式化为文本"""
    if not history:
        return "（无历史）"
    return "\n".join(
        f"用户：{m['content']}" if m["role"] == "user" else f"助手：{m['content']}"
        for m in history[-max_turns:]
    )


def _safe_json_parse(raw: str, default: dict | None = None) -> dict:
    """安全解析 LLM 返回的 JSON，处理 markdown 代码块包裹、多余逗号等常见问题"""
    if default is None:
        default = {}
    text = raw.strip()
    # 去掉 markdown 代码块包裹
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:]) if len(lines) > 1 else text
        if text.endswith("```"):
            text = text[:-3]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


# ==================== 节点 1：query_analysis ====================
# 入口节点，负责意图分类 + 搜索词改写 + 子问题拆解
# 比原来的 classify_node 多了搜索词优化和子问题识别

QUERY_ANALYSIS_PROMPT = ChatPromptTemplate.from_template("""分析用户问题并输出 JSON。

任务：
1. 意图分类：rag_query（需查资料）/ chit_chat（闲聊问候）/ meta（询问系统能力）/ follow_up（追问展开）
2. 如果是 rag_query 或 follow_up：生成一个优化后的搜索词（去除口语化、补充关键词、更适合混合检索）
3. 如果问题包含多个子问题，列出子问题

对话历史：
{chat_history}

用户输入："{question}"

仅输出 JSON（不要 markdown 包裹）：
{{"intent": "...", "search_query": "...", "sub_questions": [...]}}""")


def query_analysis_node(state: GraphState) -> dict:
    history_str = _format_history(state.get("chat_history", []), max_turns=4)
    try:
        result = safe_llm_invoke(QUERY_ANALYSIS_PROMPT.format_messages(
            chat_history=history_str, question=state["question"]
        )).content.strip()
        parsed = _safe_json_parse(result, {"intent": "rag_query"})
        intent = parsed.get("intent", "rag_query")
    except Exception:
        intent = "rag_query"
        logger.warning("query_analysis fallback to default for question=%s", state["question"][:80])

    logger.info("query_analysis intent=%s", intent)

    step = {"node": "query_analysis", "status": "done", "info": f"意图：{intent}"}

    return {
        "intent": intent,
        "retrieval_count": 0,
        "max_retrieval_attempts": 1,
        "need_retrieve": True,
        "thinking_steps": state.get("thinking_steps", []) + [step],
    }


# ==================== 节点 2：retrieve（增强版）====================
# 使用 retrieve_with_meta 获取带元数据的结果，供前端展示和文档评分使用

def retrieve_node(state: GraphState) -> dict:
    search_query = state["question"]
    result: RetrievalResult = retrieve_with_meta(search_query, top_k=10)
    count = state.get("retrieval_count", 0) + 1

    # 组装前端可展示的来源列表
    sources_for_ui = []
    for s in result.sources:
        sources_for_ui.append({
            "text": s.parent_text[:300],
            "dense_rank": s.dense_rank,
            "sparse_rank": s.sparse_rank,
            "rrf_score": s.rrf_score,
            "rerank_score": s.rerank_score,
        })

    # 使用 reranker 分数阈值判断是否需要重检（简化，不再用 LLM 打分）
    max_rerank_score = max([s.rerank_score for s in result.sources]) if result.sources else 0.0
    RERANK_THRESHOLD = 0.5  # reranker 分数阈值
    need_retrieve = max_rerank_score < RERANK_THRESHOLD

    step = {
        "node": "retrieve",
        "status": "done",
        "info": f"第 {count} 次检索：稠密命中 {result.dense_hit_count}，稀疏命中 {result.sparse_hit_count}，融合得 {result.fused_count} 篇，返回 {len(result.documents)} 篇，最高 rerank 分数 {max_rerank_score:.3f}",
    }
    logger.info(
        "retrieve count=%d query=%s dense=%d sparse=%d fused=%d returned=%d max_rerank=%.3f",
        count, search_query[:60], result.dense_hit_count, result.sparse_hit_count,
        result.fused_count, len(result.documents), max_rerank_score,
    )

    return {
        "context": result.documents,
        "retrieval_sources": sources_for_ui,
        "retrieval_count": count,
        "need_retrieve": need_retrieve,
        "retrieval_stats": {
            "dense_hits": result.dense_hit_count,
            "sparse_hits": result.sparse_hit_count,
            "fused_count": result.fused_count,
            "reranked_count": len(result.documents),
        },
        "thinking_steps": state.get("thinking_steps", []) + [step],
    }


# ==================== 节点 3：direct_answer ====================
# 闲聊 / 系统询问时直接回答，不触发检索

DIRECT_PROMPT = ChatPromptTemplate.from_template("""
你是 Agentic RAG 智能问答助手，基于 LangGraph 构建，支持混合检索（稠密+稀疏）。

对话历史：
{chat_history}

用户：{question}
请简洁友好地回答：""")


def direct_answer_node(state: GraphState) -> dict:
    history_str = _format_history(state.get("chat_history", []), max_turns=4)
    answer = llm.invoke(DIRECT_PROMPT.format_messages(
        chat_history=history_str, question=state["question"]
    ))

    if not answer.content or not answer.content.strip():
        answer = "抱歉，我无法回答该问题。请尝试换一种方式提问。"

    step = {"node": "direct_answer", "status": "done", "info": "闲聊/系统询问，直接回答"}
    return {
        "answer": answer.content if hasattr(answer, 'content') else str(answer),
        "thinking_steps": state.get("thinking_steps", []) + [step],
    }


# ==================== 节点 4：generate ====================
# 基于检索到的文档生成答案

GEN_PROMPT = ChatPromptTemplate.from_template("""基于以下资料和对话历史回答问题。

资料：
{context}

对话历史：
{chat_history}

问题：{question}

要求：仅基于资料回答，资料不足时明确说明。回答简洁完整。""")


gen_chain = GEN_PROMPT | llm | StrOutputParser()


def generate_node(state: GraphState) -> dict:
    docs = state.get("context", [])
    history_str = _format_history(state.get("chat_history", []), max_turns=6)

    context_text = "\n\n".join(docs)

    answer = gen_chain.invoke({
        "context": context_text,
        "chat_history": history_str,
        "question": state["question"],
    })

    if not answer or not answer.strip():
        if not docs:
            answer = "抱歉，未检索到相关资料，无法回答该问题。请尝试换一种方式提问。"
        else:
            answer = "抱歉，基于现有资料无法生成完整回答。请尝试换一种方式提问。"

    logger.info("generate docs=%d answer_len=%d", len(docs), len(answer))

    step = {"node": "generate", "status": "done", "info": f"基于 {len(docs)} 篇文档生成回答"}
    return {
        "answer": answer,
        "thinking_steps": state.get("thinking_steps", []) + [step],
    }


# ==================== 路由函数 ====================

def route_after_analysis(state: GraphState) -> str:
    """query_analysis 之后的分流"""
    intent = state.get("intent", "")
    if intent in ("chit_chat", "meta"):
        return "direct_answer"
    return "retrieve"  # rag_query / follow_up 都走检索


# ==================== 构建工作流 ====================

workflow = StateGraph(GraphState)

# 注册节点
workflow.add_node("query_analysis", query_analysis_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("direct_answer", direct_answer_node)

# 边
workflow.set_entry_point("query_analysis")

workflow.add_conditional_edges(
    "query_analysis",
    route_after_analysis,
    {"retrieve": "retrieve", "direct_answer": "direct_answer"},
)

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
workflow.add_edge("direct_answer", END)

app = workflow.compile()  # 编译成可执行应用
