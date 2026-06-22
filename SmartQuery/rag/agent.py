import json
from typing import TypedDict
from SmartQuery.rag.retriever import retrieve, retrieve_with_meta, RetrievalResult
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from SmartQuery.backend.config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL
from langgraph.graph import StateGraph, END
from tenacity import retry, stop_after_attempt, wait_exponential


# ==================== State 定义 ====================

class GraphState(TypedDict):
    # ------ 核心字段（保留原有）------
    question: str               # 用户原始问题
    context: list[str]          # 检索到的父块文档列表
    answer: str                 # LLM 生成的回答
    intent: str                 # 意图分类：rag_query / chit_chat / meta / follow_up
    chat_history: list[dict]    # 多轮对话历史 [{role, content}, ...]

    # ------ Agentic 迭代控制 ------
    rewritten_questions: list[str]  # 改写过的搜索词链，记录每次重检用了什么 query
    retrieval_count: int            # 已经检索了几次
    max_retrieval_attempts: int     # 最大检索次数，默认 3，防止无限循环
    need_retrieve: bool             # 文档评分不足时置为 True，触发重检

    # ------ 文档评判 ------
    document_grades: list[dict]     # [{doc_index, relevance, reason}, ...] LLM 对每篇文档的打分
    relevant_docs: list[str]        # 相关性 >= 阈值的文档文本
    irrelevant_docs: list[str]      # 低于阈值的文档（诊断用）

    # ------ 反思 ------
    reflection_result: dict         # {has_issues, issues, completeness_score}

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

# 轻量模型：文档打分、查询改写、反思检查不需要最强模型，qwen-plus 够用且省钱
SMALL_MODEL = "qwen-plus"
small_llm = ChatOpenAI(
    model=SMALL_MODEL,
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
)


# ==================== 工具函数 ====================

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=3))
def safe_llm_invoke(messages):
    """带重试的 LLM 调用"""
    return llm.invoke(messages)


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=3))
def safe_small_llm_invoke(messages):
    """带重试的小模型调用"""
    return small_llm.invoke(messages)


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
        search_query = parsed.get("search_query", state["question"])
        sub_questions = parsed.get("sub_questions", [])
    except Exception:
        intent = "rag_query"
        search_query = state["question"]
        sub_questions = []

    # 记录思考过程
    step = {"node": "query_analysis", "status": "done", "info": f"意图：{intent}"}
    if search_query != state["question"]:
        step["info"] += f"，搜索词：{search_query}"

    return {
        "intent": intent,
        "rewritten_questions": [search_query],
        "max_retrieval_attempts": 3,
        "retrieval_count": 0,
        "need_retrieve": True,
        "thinking_steps": state.get("thinking_steps", []) + [step],
    }


# ==================== 节点 2：retrieve（增强版）====================
# 使用 retrieve_with_meta 获取带元数据的结果，供前端展示和文档评分使用

def retrieve_node(state: GraphState) -> dict:
    # 取最新的改写搜索词
    search_query = state.get("rewritten_questions", [state["question"]])[-1]
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

    step = {
        "node": "retrieve",
        "status": "done",
        "info": f"第 {count} 次检索：稠密命中 {result.dense_hit_count}，稀疏命中 {result.sparse_hit_count}，融合得 {result.fused_count} 篇，返回 {len(result.documents)} 篇",
    }

    return {
        "context": result.documents,
        "retrieval_sources": sources_for_ui,
        "retrieval_count": count,
        # 检索统计（稠密命中数 / 稀疏命中数 / 融合后数量），前端链路面板展示
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
你是 Agentic RAG 智能问答助手，基于 LangGraph 构建，支持混合检索（稠密+稀疏）和 Agentic 自省。

对话历史：
{chat_history}

用户：{question}
请简洁友好地回答：""")


def direct_answer_node(state: GraphState) -> dict:
    history_str = _format_history(state.get("chat_history", []), max_turns=4)
    answer = llm.invoke(DIRECT_PROMPT.format_messages(
        chat_history=history_str, question=state["question"]
    ))
    step = {"node": "direct_answer", "status": "done", "info": "闲聊/系统询问，直接回答"}
    return {
        "answer": answer.content,
        "thinking_steps": state.get("thinking_steps", []) + [step],
    }


# ==================== 节点 4：grade_documents ====================
# Agentic 核心决策节点：LLM 逐篇评判文档与问题的相关性
# 得分不足 → 触发 rewrite_query 重检
# 得分足够 → 进入 generate 生成答案

GRADE_PROMPT = ChatPromptTemplate.from_template("""评估以下文档与用户问题的相关性，输出 JSON 数组。

用户问题：{question}

文档列表：
{documents}

对每篇文档评分（1-5）并说明理由。仅输出 JSON 数组（不要 markdown 包裹）：
[{{"doc_index": 0, "relevance": 4, "reason": "提到了相关技能"}}, ...]""")


def grade_documents_node(state: GraphState) -> dict:
    context = state.get("context", [])
    if not context:
        return {
            "document_grades": [],
            "relevant_docs": [],
            "irrelevant_docs": [],
            "need_retrieve": state.get("retrieval_count", 0) < state.get("max_retrieval_attempts", 3),
            "thinking_steps": state.get("thinking_steps", []) + [
                {"node": "grade_documents", "status": "done", "info": "无文档可评分"}
            ],
        }

    # 合并文档列表为文本，每篇给编号
    docs_text = "\n\n".join(f"[{i}] {doc[:400]}" for i, doc in enumerate(context))

    try:
        result = safe_small_llm_invoke(GRADE_PROMPT.format_messages(
            question=state["question"], documents=docs_text
        )).content.strip()
        grades = _safe_json_parse(result, [])
        if not isinstance(grades, list):
            grades = []
    except Exception:
        # 打分失败时默认全给 3 分（中性），不阻塞流程
        grades = [{"doc_index": i, "relevance": 3, "reason": "评分异常，默认通过"} for i in range(len(context))]

    # 按相关性分档
    threshold = 3  # 1-5 分制，>=3 为相关
    relevant, irrelevant = [], []
    for g in grades:
        idx = g.get("doc_index", 0)
        if idx < len(context):
            if g.get("relevance", 3) >= threshold:
                relevant.append(context[idx])
            else:
                irrelevant.append(context[idx])

    need = len(relevant) < 2  # 相关文档少于 2 篇则认为检索不充分
    max_attempts = state.get("max_retrieval_attempts", 3)
    retrieval_count = state.get("retrieval_count", 0)
    if need and retrieval_count >= max_attempts:
        need = False  # 已达最大重试次数，强制生成

    step = {
        "node": "grade_documents",
        "status": "done",
        "info": f"相关 {len(relevant)} 篇 / 不相关 {len(irrelevant)} 篇 → {'需要重检' if need else '进入生成'}",
    }

    return {
        "document_grades": grades,
        "relevant_docs": relevant,
        "irrelevant_docs": irrelevant,
        "need_retrieve": need,
        "thinking_steps": state.get("thinking_steps", []) + [step],
    }


# ==================== 节点 5：rewrite_query ====================
# 检索不理想时，LLM 分析失败原因并生成更好的搜索词

REWRITE_PROMPT = ChatPromptTemplate.from_template("""你是一个搜索优化助手。

用户原始问题：{question}
已尝试的搜索词：{previous_queries}
检索结果中不相关文档示例：{irrelevant_samples}

请生成一个不同的、更精准的搜索词。要求：
- 换一个角度或关键词
- 如果之前的词太宽泛就加具体限定，如果太窄就放宽
- 避免之前失败的搜索方向

仅输出新搜索词（纯文本，不要 JSON 包裹）：""")


def rewrite_query_node(state: GraphState) -> dict:
    previous = state.get("rewritten_questions", [])
    irrelevant = state.get("irrelevant_docs", [])
    samples = "\n".join(d[:200] for d in irrelevant[:3]) if irrelevant else "（无）"

    try:
        result = safe_small_llm_invoke(REWRITE_PROMPT.format_messages(
            question=state["question"],
            previous_queries=", ".join(previous),
            irrelevant_samples=samples,
        )).content.strip()
        new_query = result or state["question"]
    except Exception:
        new_query = state["question"]

    previous.append(new_query)
    step = {"node": "rewrite_query", "status": "done", "info": f"新搜索词：{new_query}"}

    return {
        "rewritten_questions": previous,
        "thinking_steps": state.get("thinking_steps", []) + [step],
    }


# ==================== 节点 6：generate ====================
# 基于相关文档生成答案，优先用评分通过的 relevant_docs

GEN_PROMPT = ChatPromptTemplate.from_template("""基于以下资料和对话历史回答问题。

资料：
{context}

对话历史：
{chat_history}

问题：{question}

要求：仅基于资料回答，资料不足时明确说明。回答简洁完整。""")


gen_chain = GEN_PROMPT | llm | StrOutputParser()


def generate_node(state: GraphState) -> dict:
    # 优先使用评分通过的文档，没有则使用全量 context
    docs = state.get("relevant_docs") or state.get("context", [])
    history_str = _format_history(state.get("chat_history", []), max_turns=6)
    retrieval_count = state.get("retrieval_count", 0)
    max_attempts = state.get("max_retrieval_attempts", 3)

    # 多次检索仍不充分时，在 prompt 中加免责提示
    context_text = "\n\n".join(docs)
    if retrieval_count >= max_attempts and len(docs) < 2:
        context_text = f"（注意：多次检索未找到充分资料，以下信息可能不完整）\n\n{context_text}"

    answer = gen_chain.invoke({
        "context": context_text,
        "chat_history": history_str,
        "question": state["question"],
    })

    step = {"node": "generate", "status": "done", "info": f"基于 {len(docs)} 篇文档生成回答"}
    return {
        "answer": answer,
        "thinking_steps": state.get("thinking_steps", []) + [step],
    }


# ==================== 节点 7：reflect ====================
# 幻觉检查：LLM 对比答案与源文档，发现无依据的声明则标记问题

REFLECT_PROMPT = ChatPromptTemplate.from_template("""你是事实核验员。请对比以下回答与源文档。

问题：{question}

回答：
{answer}

源文档：
{context}

检查：
1. 回答中是否有源文档无法支持的声明？（幻觉）
2. 源文档中的重要信息是否被遗漏？
3. 整体完整度评分（1-5）

仅输出 JSON（不要 markdown 包裹）：
{{"has_issues": true/false, "issues": "描述", "completeness_score": 1-5}}""")


def reflect_node(state: GraphState) -> dict:
    docs = state.get("relevant_docs") or state.get("context", [])
    if not docs or not state.get("answer"):
        step = {"node": "reflect", "status": "done", "info": "跳过：无文档或无答案"}
        return {
            "reflection_result": {"has_issues": False, "issues": "", "completeness_score": 3},
            "thinking_steps": state.get("thinking_steps", []) + [step],
        }

    try:
        result = safe_small_llm_invoke(REFLECT_PROMPT.format_messages(
            question=state["question"],
            answer=state["answer"],
            context="\n\n".join(docs),
        )).content.strip()
        reflection = _safe_json_parse(result, {"has_issues": False, "issues": "", "completeness_score": 4})
    except Exception:
        reflection = {"has_issues": False, "issues": "反思检查异常，跳过", "completeness_score": 3}

    has_issues = reflection.get("has_issues", False)
    step = {
        "node": "reflect",
        "status": "done",
        "info": f"{'发现问题' if has_issues else '无问题'}，完整度 {reflection.get('completeness_score', '?')}/5",
    }

    return {
        "reflection_result": reflection,
        "thinking_steps": state.get("thinking_steps", []) + [step],
    }


# ==================== 路由函数 ====================

def route_after_analysis(state: GraphState) -> str:
    """query_analysis 之后的分流"""
    intent = state.get("intent", "")
    if intent in ("chit_chat", "meta"):
        return "direct_answer"
    return "retrieve"  # rag_query / follow_up 都走检索


def route_after_grade(state: GraphState) -> str:
    """grade_documents 之后的决策"""
    if state.get("need_retrieve", False):
        retrieval_count = state.get("retrieval_count", 0)
        max_attempts = state.get("max_retrieval_attempts", 3)
        if retrieval_count < max_attempts:
            return "rewrite_query"
        # 达到最大次数，强制生成
    return "generate"


def route_after_reflect(state: GraphState) -> str:
    """reflect 之后：有严重幻觉且是第一次反思则重新生成"""
    reflection = state.get("reflection_result", {})
    has_issues = reflection.get("has_issues", False)
    completeness = reflection.get("completeness_score", 5)
    if has_issues and completeness <= 2:
        return "generate"  # 带着反思意见重新生成
    return "END"


# ==================== 构建工作流 ====================

workflow = StateGraph(GraphState)

# 注册节点
workflow.add_node("query_analysis", query_analysis_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("generate", generate_node)
workflow.add_node("reflect", reflect_node)
workflow.add_node("direct_answer", direct_answer_node)

# 边
workflow.set_entry_point("query_analysis")

workflow.add_conditional_edges(
    "query_analysis",
    route_after_analysis,
    {"retrieve": "retrieve", "direct_answer": "direct_answer"},
)

workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    route_after_grade,
    {"rewrite_query": "rewrite_query", "generate": "generate"},
)

workflow.add_edge("rewrite_query", "retrieve")  # 循环回检索
workflow.add_edge("generate", "reflect")

workflow.add_conditional_edges(
    "reflect",
    route_after_reflect,
    {"generate": "generate", "END": END},
)

workflow.add_edge("direct_answer", END)

app = workflow.compile()  # 编译成可执行应用
