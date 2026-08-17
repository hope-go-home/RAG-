"""
03_chains.py - 链的使用演示
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

llm = ChatOpenAI(model="gpt-4o-mini")

# 1. 基本 LCEL 链
print("=== 1. 基本链 ===")
prompt = ChatPromptTemplate.from_template("用中文回答：{question}")
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"question": "什么是机器学习？"})
print(f"回答: {result}\n")

# 2. 并行链
print("=== 2. 并行链 ===")
summary_prompt = ChatPromptTemplate.from_template("摘要：{text}")
keyword_prompt = ChatPromptTemplate.from_template("提取关键词：{text}")
lang_prompt = ChatPromptTemplate.from_template("判断语言：{text}")

chain_summary = summary_prompt | llm | StrOutputParser()
chain_keywords = keyword_prompt | llm | StrOutputParser()
chain_lang = lang_prompt | llm | StrOutputParser()

parallel_chain = RunnableParallel(
    summary=chain_summary,
    keywords=chain_keywords,
    language=chain_lang
)

result = parallel_chain.invoke({"text": "Python is a powerful programming language used in AI."})
print(f"摘要: {result['summary']}")
print(f"关键词: {result['keywords']}")
print(f"语言: {result['language']}\n")

# 3. RunnablePassthrough
print("=== 3. RunnablePassthrough ===")
prompt = ChatPromptTemplate.from_template(
    "用{language}回答：{question}"
)
chain = {
    "question": RunnablePassthrough(),
    "language": lambda _: "中文"
} | prompt | llm | StrOutputParser()

result = chain.invoke("What is deep learning?")
print(f"回答: {result}")
