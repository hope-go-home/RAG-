"""
04_memory.py - 记忆功能演示
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

llm = ChatOpenAI(model="gpt-4o-mini")

# 1. ConversationChain + Memory
print("=== 1. ConversationBufferMemory ===")
memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory, verbose=False)

result1 = chain.invoke("你好，我叫张三")
print(f"AI: {result1['response']}")

result2 = chain.invoke("我叫什么名字？")
print(f"AI: {result2['response']}\n")

# 2. 手动管理历史（推荐方式）
print("=== 2. 手动管理历史 ===")
history = []

prompt = ChatPromptTemplate([
    ("system", "你是助手"),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

def chat(text):
    result = chain.invoke({
        "history": history,
        "input": text
    })
    history.append(HumanMessage(content=text))
    history.append(AIMessage(content=result))
    return result

print(f"AI: {chat('我叫李四')}")
print(f"AI: {chat('我来自北京')}")
print(f"AI: {chat('我叫什么？来自哪里？')}")
