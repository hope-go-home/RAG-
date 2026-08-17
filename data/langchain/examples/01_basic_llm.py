"""
01_basic_llm.py - LLM 基础使用
需要设置 OPENAI_API_KEY 环境变量
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 方式1：直接字符串
print("=== 方式1：字符串输入 ===")
result = llm.invoke("用一句话介绍 Python")
print(result.content)

# 方式2：消息列表（推荐）
print("\n=== 方式2：消息列表 ===")
messages = [
    SystemMessage(content="你是一个幽默的助手，回答要简短有趣。"),
    HumanMessage(content="为什么程序员喜欢用 Python？")
]
result = llm.invoke(messages)
print(result.content)

# 方式3：流式输出
print("\n=== 方式3：流式输出 ===")
for chunk in llm.stream("从1数到5，每个数一行"):
    print(chunk.content, end="", flush=True)
