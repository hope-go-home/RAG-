"""
chat_app.py - 完整 AI 聊天应用（RAG + Agent）
运行: streamlit run chat_app.py
"""

import streamlit as st
import tempfile
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor

st.set_page_config(page_title="AI 助手", page_icon="🤖", layout="wide")

def init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("vectorstore", None)
    st.session_state.setdefault("rag_chain", None)
    st.session_state.setdefault("mode", "chat")

init_state()

@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True)

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

llm = get_llm()
embeddings = get_embeddings()

def process_document(uploaded_file):
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == ".txt":
        loader = TextLoader(tmp_path, encoding="utf-8")
    elif ext == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(tmp_path)
    else:
        st.error(f"不支持的文件格式: {ext}")
        os.unlink(tmp_path)
        return None

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    os.unlink(tmp_path)
    return vectorstore

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate([
        ("system", "基于资料回答问题。不知道就说不知道。\n资料：{context}"),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    return (
        {
            "context": retriever | format_docs,
            "history": lambda x: st.session_state.messages[-10:],
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

@tool
def web_search(query: str) -> str:
    """搜索网络获取最新信息"""
    from langchain_community.tools import DuckDuckGoSearchRun
    search = DuckDuckGoSearchRun()
    try:
        return search.invoke(query)
    except:
        return "搜索失败"

@tool
def calculate(expression: str) -> str:
    """执行数学计算。输入：数学表达式"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"

tools = [web_search, calculate]

def build_agent():
    prompt = ChatPromptTemplate([
        ("system", "你是全能助手。可以用工具搜索和计算。"),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, tools=tools,
        verbose=True, max_iterations=5,
        handle_parsing_errors=True
    )

def main():
    with st.sidebar:
        st.title("🤖 AI 助手")
        st.session_state.mode = st.radio(
            "模式", ["💬 聊天", "📚 文档问答"],
            index=0 if st.session_state.mode == "chat" else 1
        )
        st.divider()

        if "文档问答" in st.session_state.mode:
            st.subheader("上传文档")
            uploaded_file = st.file_uploader("支持 TXT/PDF/DOCX", type=["txt", "pdf", "docx"])
            if uploaded_file:
                with st.spinner("处理中..."):
                    vs = process_document(uploaded_file)
                    if vs:
                        st.session_state.vectorstore = vs
                        st.session_state.rag_chain = build_rag_chain(vs)
                        st.success(f"已处理: {uploaded_file.name}")
            if st.session_state.vectorstore and st.button("清除文档"):
                st.session_state.vectorstore = None
                st.session_state.rag_chain = None
                import shutil
                if os.path.exists("./chroma_db"):
                    shutil.rmtree("./chroma_db")
                st.rerun()

        st.divider()
        if st.button("清除对话"):
            st.session_state.messages = []
            st.rerun()

    if "聊天" in st.session_state.mode:
        st.title("💬 智能聊天")
        st.caption("支持网络搜索 + 数学计算")
        mode = "chat"
    else:
        st.title("📚 文档问答")
        if st.session_state.vectorstore:
            st.caption("已加载知识库")
            mode = "rag"
        else:
            st.info("请在左侧上传文档")
            mode = "rag_no_doc"

    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(msg.content)

    if prompt := st.chat_input("请输入问题..."):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    if mode == "chat":
                        agent = build_agent()
                        result = agent.invoke({
                            "input": prompt,
                            "history": st.session_state.messages[-10:-1]
                        })
                        response = result["output"]
                    elif mode == "rag":
                        response = st.session_state.rag_chain.invoke(prompt)
                    else:
                        response = "请先上传文档"
                    st.write(response)
                    st.session_state.messages.append(AIMessage(content=response))
                except Exception as e:
                    st.error(f"错误: {e}")

if __name__ == "__main__":
    main()
