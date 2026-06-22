import json
from typing import TypedDict
from SmartQuery.rag.retriever import retrieve
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from SmartQuery.backend.config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL
from langgraph.graph import StateGraph, END


class GraphState(TypedDict):
    question: str       #用户问题
    context: list[str]  #检索到的文档列表
    answer: str         #LLM生成的回答
    intent: str         #意图分类
    chat_history: list[dict]  #对话历史


CLASSIFY_PROMPT = ChatPromptTemplate.from_template(
"""判断用户问题属于哪一类，输出 JSON。

分类：
- rag_query：需要查资料回答的知识问题
- chit_chat：闲聊、问候
- meta：询问你是谁、你能做什么
- follow_up：追问、要求展开、说详细点、为什么

对话历史：
{chat_history}
用户的最新输入："{input}"

请严格按照 JSON 格式输出，只包含意图分类：
{{"intent": "..."}}"""
)


def classify_node(state: GraphState) -> dict:
    history = state.get("chat_history", [])
    history_str = "\n".join(
        f"用户：{m['content']}" if m["role"] == "user" else f"助手：{m['content']}"
        for m in history[-4:]
    ) or "（无历史）"
    result = llm.invoke(CLASSIFY_PROMPT.format_messages(
        chat_history=history_str, input=state["question"]
    )).content.strip()
    intent = json.loads(result)["intent"]
    return {"intent": intent}


def retrieve_node(state: GraphState) -> dict: #retrieve_node(state: GraphState) 中的参数 state 并不是一个自定义的 GraphState 对象，而是一个 符合 GraphState 类型定义的字典（因为 GraphState 是 TypedDict，本质就是 dict 类型）
    context = retrieve(state["question"])
    return {"context": context}
# state 参数是 LangGraph 框架自动传递的，不需要你在代码中手动定义或传入。


"""
state["question"] 就是从当前状态字典中取出键为 "question" 的值（即用户问题字符串）。
如果你已经在代码中定义了一个局部变量 question 且赋值为相同的问题字符串，那么 retrieve(question) 与 retrieve(state["question"]) 在效果上完全等价。
state 是 LangGraph 框架自动维护的全局状态字典，用于在工作流的多个节点之间传递和共享数据（question、context、answer）。
直接使用 question 变量（假设已定义）只能在当前函数内使用，无法跨节点共享。
所以简单来说：state 就是一个"有固定格式的字典"，它的 "question" 字段记录着用户问题。retrieve(state["question"]) 就是取出该问题并传给检索函数。
"""

llm = ChatOpenAI(model=QWEN_MODEL, 
                 api_key=QWEN_API_KEY, 
                 base_url=QWEN_BASE_URL
                 )

gen_prompt = ChatPromptTemplate.from_template(
    "基于以下资料和对话历史回答问题。\n\n资料：{context}\n\n对话历史：\n{chat_history}\n\n问题：{question}\n\n回答："
)
gen_chain = gen_prompt | llm | StrOutputParser()


def generate_node(state: GraphState) -> dict:
    history = state.get("chat_history", [])
    history_str = "\n".join(
        f"用户：{m['content']}" if m["role"] == "user" else f"助手：{m['content']}"
        for m in history[-6:]
    ) or "（无历史）"
    return {"answer": gen_chain.invoke({
        "context": "\n\n".join(state["context"]),
        "chat_history": history_str,
        "question": state["question"],
    })}
# "context": "\n\n".join(state["context"])：把状态中存储的检索文档列表（state["context"] 是 list[str]）用两个换行符连接成一个长字符串，作为提示中的资料。
# "chat_history"：最近 6 轮对话，保持上下文连贯
# "question": state["question"]：直接取出状态中的原始用户问题


def direct_answer_node(state: GraphState) -> dict:
    answer = llm.invoke(f"你是智能问答助手。请回答以下问题：\n\n{state['question']}")
    return {"answer": answer.content}


def route_after_classify(state: GraphState) -> str:
    intent = state.get("intent", "")
    if intent == "rag_query":
        return "retrieve"
    if intent == "follow_up" and state.get("context"):
        return "generate"          #有历史上下文，跳过检索直接用旧 context 生成
    if intent == "follow_up" and not state.get("context"):
        return "retrieve"          #没有历史上下文，fallback 走检索
    return "direct_answer"         #chit_chat / meta 直接回答


workflow = StateGraph(GraphState)
workflow.add_node("classify", classify_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("direct_answer", direct_answer_node)

workflow.set_entry_point("classify")    #从 classify 节点开始
workflow.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "retrieve": "retrieve",
        "generate": "generate",
        "direct_answer": "direct_answer",
    },
)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
workflow.add_edge("direct_answer", END)
app = workflow.compile()              #编译成可执行应用
