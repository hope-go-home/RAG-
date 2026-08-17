"""
02_prompt_template.py - 提示词模板用法
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate
)
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")

# 1. 基本模板
print("=== 1. 基本 PromptTemplate ===")
template = PromptTemplate.from_template(
    "你是一个{role}。请用{language}回答：{question}"
)
prompt = template.invoke({
    "role": "数学老师",
    "language": "中文",
    "question": "什么是勾股定理？"
})
print(prompt.text)
result = llm.invoke(prompt)
print(f"回答: {result.content}\n")

# 2. 对话模板
print("=== 2. ChatPromptTemplate ===")
chat_template = ChatPromptTemplate([
    ("system", "你是一个{role}"),
    ("human", "{question}")
])
chain = chat_template | llm | StrOutputParser()
result = chain.invoke({"role": "历史老师", "question": "唐朝建立于哪一年？"})
print(f"回答: {result}\n")

# 3. FewShot 模板
print("=== 3. FewShotPromptTemplate ===")
examples = [
    {"word": "开心", "sentiment": "正面"},
    {"word": "愤怒", "sentiment": "负面"},
    {"word": "无聊", "sentiment": "负面"},
]

example_template = PromptTemplate.from_template("词语: {word} → 情感: {sentiment}")

fewshot = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="判断以下词语的情感倾向：\n",
    suffix="词语: {word} → 情感:",
    input_variables=["word"]
)

prompt = fewshot.invoke({"word": "兴奋"})
print(prompt.text)
result = llm.invoke(prompt)
print(f"回答: {result.content}")
