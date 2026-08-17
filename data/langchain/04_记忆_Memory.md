# 04 - 记忆：Memory

---

## 一、为什么需要 Memory？

```python
# 没有记忆：LLM 记不住之前说过什么
llm.invoke("我叫张三")
llm.invoke("我叫什么名字？")  # ❌ 它不知道！每次调用是独立的
```

**解决方案：** 把历史对话塞进 prompt 里。Memory 组件自动帮你做这件事。

---

## 二、ConversationBufferMemory（基础记忆）

把完整对话历史存入 prompt：

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory, verbose=True)

chain.invoke("你好，我叫张三")
# AI: 你好张三！很高兴认识你。

chain.invoke("我叫什么名字？")
# AI: 你叫张三！
```

### 查看内存内容

```python
print(memory.load_memory_variables({}))
# {'history': 'Human: 你好，我叫张三\nAI: 你好张三！很高兴认识你。\nHuman: 我叫什么名字？\nAI: 你叫张三！'}

# 清空记忆
memory.clear()
```

### 缺点
对话越长，prompt 越大，会超过 token 限制（而且浪费钱）。

---

## 三、ConversationBufferWindowMemory（窗口记忆）

只保留最近 k 轮对话：

```python
from langchain.memory import ConversationBufferWindowMemory

# 只保留最近 2 轮
memory = ConversationBufferWindowMemory(k=2)
chain = ConversationChain(llm=llm, memory=memory)

chain.invoke("我叫张三")     # 第1轮
chain.invoke("我喜欢编程")   # 第2轮
chain.invoke("我喜欢Python") # 第3轮
chain.invoke("我叫什么？")   # 第4轮——第1轮的"张三"已经被遗忘了
```

**适用场景：** 对话轮次多，但只需要最近的上下文。

---

## 四、ConversationSummaryMemory（摘要记忆）

每次对话后，自动把历史**压缩成摘要**，节省 token：

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)
chain = ConversationChain(llm=llm, memory=memory)

chain.invoke("我叫张三，来自北京")
chain.invoke("我是一名程序员")
chain.invoke("我喜欢 Python 和机器学习")

# 内存里不是完整对话，而是摘要：
# "用户叫张三，来自北京，是一名程序员，喜欢Python和机器学习"
```

**适用场景：** 长对话，需要记住关键信息，节省 token。

---

## 五、ConversationSummaryBufferMemory（混合记忆）

结合摘要和最近对话：

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=200   # 最近 200 token 保留原文，之外的压缩成摘要
)
```

---

## 六、VectorStoreMemory（向量记忆）

用向量数据库存储海量历史：

```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)

memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory_key="history"
)

# 适合超长对话或需要搜索特定历史
```

---

## 七、在 LCEL 中使用 Memory

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ChatMessageHistory

# 1. 手动管理历史（推荐：完全可控）
history = ChatMessageHistory()

prompt = ChatPromptTemplate([
    ("system", "你是助手"),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

def chat(input_text):
    result = chain.invoke({
        "history": history.messages,
        "input": input_text
    })
    history.add_user_message(input_text)
    history.add_ai_message(result)
    return result

print(chat("我叫张三"))
print(chat("我叫什么名字？"))
```

### 用 RunnableWithMessageHistory（官方方式）

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

prompt = ChatPromptTemplate([
    ("system", "你是助手"),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

# 存储所有会话的 history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)8

# 使用
result = chain_with_history.invoke(
    {"input": "我叫张三"},
    config={"configurable": {"session_id": "user123"}}
)
# 同一 session_id 会自动累积历史

result = chain_with_history.invoke(
    {"input": "我叫什么？"},
    config={"configurable": {"session_id": "user123"}}
)
# ✅ 回答：你叫张三
```

---

## 八、Memory 类型对比

| 类型 | 保留内容 | token消耗 | 适合场景 |
|------|---------|-----------|---------|
| `ConversationBufferMemory` | 完整历史 | ❌ 高 | 短对话 |
| `ConversationBufferWindowMemory` | 最近k轮 | ✅ 低 | 多轮但不需全部 |
| `ConversationSummaryMemory` | 摘要 | ✅ 低 | 长对话 |
| `ConversationSummaryBufferMemory` | 摘要+最近 | ✅ 中 | 长对话+最近细节 |
| `VectorStoreRetrieverMemory` | 向量检索 | ✅ 中 | 超长对话/搜索 |

---

## 九、完整示例：带记忆的聊天

```python
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

st.title("带记忆的聊天机器人")

# 初始化
if "messages" not in st.session_state:
    st.session_state.messages = []

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

prompt = ChatPromptTemplate([
    ("system", "你是助手，用中文回答。"),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

chain = prompt | llm

# 显示历史
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# 输入
if prompt_input := st.chat_input("请输入..."):
    st.session_state.messages.append(HumanMessage(content=prompt_input))
    with st.chat_message("user"):
        st.write(prompt_input)

    result = chain.invoke({
        "history": st.session_state.messages[:-1],
        "input": prompt_input
    })
    st.session_state.messages.append(AIMessage(content=result.content))
    with st.chat_message("assistant"):
        st.write(result.content)
```

---

**下一篇 → [05_检索_RAG](05_检索_RAG.md)**
