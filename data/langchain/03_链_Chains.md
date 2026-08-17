# 03 - 链：Chains

---

## 一、什么是 Chain？

**Chain（链）** 是把多个组件串联成一个 **可执行的流水线**。

```
输入 → [Prompt] → [LLM] → [OutputParser] → 输出
```

### 旧式写法（Legacy Chain）

```python
from langchain.chains import LLMChain

chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    verbose=True
)
result = chain.run(question="什么是机器学习")
```

### ✅ 新式写法：LCEL（推荐）

```python
# LangChain Expression Language (LCEL)
chain = prompt_template | llm | output_parser

result = chain.invoke({"question": "什么是机器学习"})
```

> **核心心法：** LCEL 用 `|` 管道符串联组件，数据从左向右流动。每个组件实现 `Runnable` 接口（都有 `.invoke()` 方法）。

---

## 二、LCEL 基础

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# 构建链
prompt = ChatPromptTemplate.from_template("用中文回答：{question}")
parser = StrOutputParser()

chain = prompt | llm | parser

# 执行
result = chain.invoke({"question": "什么是量子计算？"})
print(result)
# 直接输出字符串（parser 把 AIMessage 转成了字符串）
```

### 管道原理

```
chain.invoke({"question": "..."})

step 1: prompt.invoke({"question": "..."}) → [SystemMessage, HumanMessage]
step 2: llm.invoke([SystemMessage, HumanMessage]) → AIMessage(content="...")
step 3: parser.invoke(AIMessage) → "最终文本"
```

---

## 三、OutputParser（输出解析器）

### 常用解析器

```python
# 1. 字符串输出
StrOutputParser()  # 提取 message.content

# 2. JSON 输出
from langchain_core.output_parsers import JsonOutputParser
parser = JsonOutputParser()
chain = prompt | llm | parser
result = chain.invoke(...)  # → dict

# 3. Pydantic 结构化输出
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Movie(BaseModel):
    title: str = Field(description="电影名")
    year: int = Field(description="上映年份")
    rating: float = Field(description="评分")

parser = PydanticOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_template(
    "从文本中提取电影信息：\n{text}\n{format_instructions}"
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser
movie = chain.invoke({"text": "《肖申克的救赎》1994年上映，评分9.7"})
print(movie.title, movie.year, movie.rating)
```

---

## 四、RunnablePassthrough（传递数据）

在不修改数据的情况下传递变量：

```python
from langchain_core.runnables import RunnablePassthrough

chain = {
    "question": RunnablePassthrough(),      # 原样传递
    "language": lambda _: "中文"             # 固定值
} | prompt | llm | parser

result = chain.invoke("什么是微积分？")
# 实际传入 prompt 的变量为：{"question": "什么是微积分？", "language": "中文"}
```

---

## 五、RunnableParallel（并行执行）

多个任务并行执行：

```python
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel(
    summary=summary_chain,       # 摘要链
    keywords=keywords_chain,     # 关键词链
    sentiment=sentiment_chain    # 情感分析链
)

result = chain.invoke({"text": "今天天气真好，阳光明媚！"})
# result = {"summary": "...", "keywords": "...", "sentiment": "..."}
```

### 实际例子

```python
# 定义三个子任务
prompt_summary = ChatPromptTemplate.from_template("总结：{text}")
prompt_keywords = ChatPromptTemplate.from_template("提取关键词：{text}")
prompt_language = ChatPromptTemplate.from_template("判断语言：{text}")

chain_summary = prompt_summary | llm | StrOutputParser()
chain_keywords = prompt_keywords | llm | StrOutputParser()
chain_language = prompt_language | llm | StrOutputParser()

# 并行执行
parallel_chain = RunnableParallel(
    summary=chain_summary,
    keywords=chain_keywords,
    language=chain_language
)

result = parallel_chain.invoke({"text": "Machine learning is fascinating!"})
print(result)
# {'summary': '...', 'keywords': '...', 'language': '英语'}
```

---

## 六、RunnableBranch（条件分支）

根据条件选择不同路径：

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: len(x["text"]) > 100, long_text_chain),    # 长文本
    (lambda x: len(x["text"]) > 20, medium_text_chain),   # 中等
    short_text_chain                                       # 短文本（默认）
)

result = branch.invoke({"text": "你好"})
```

---

## 七、Legacy Chains（了解即可）

虽然推荐用 LCEL，但有些旧式 Chain 仍然有用：

### LLMChain

```python
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
result = chain.run(question="你好")
```

### SequentialChain（顺序链）

```python
from langchain.chains import SimpleSequentialChain

# 链1：写笑话
chain1 = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template("讲一个关于{topic}的笑话")
)

# 链2：翻译成英文
chain2 = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template("把下面内容翻译成英文：{text}")
)

# 串联：输入 → chain1 → chain2 → 输出
pipeline = SimpleSequentialChain(chains=[chain1, chain2])
result = pipeline.run("程序员")
```

---

## 八、自定义 Runnable

```python
from langchain_core.runnables import RunnableLambda

# 自定义函数
def reverse_text(text: str) -> str:
    return text[::-1]

# 转成 Runnable
reverse = RunnableLambda(reverse_text)

# 插入链中
chain = prompt | llm | reverse
result = chain.invoke({"question": "hello"})
# LLM 输出 "你好" → reverse → "好你"
```

### 更复杂的自定义

```python
from langchain_core.runnables import RunnableGenerator

def process_chunks(chunks):
    for chunk in chunks:
        yield chunk.upper()

chain = prompt | llm | RunnableGenerator(process_chunks)

for chunk in chain.stream({"question": "你好"}):
    print(chunk, end="")
```

---

## 九、链的执行与控制

```python
# 同步执行
chain.invoke(input_data)

# 异步执行
await chain.ainvoke(input_data)

# 流式执行
for chunk in chain.stream(input_data):
    print(chunk)

# 批量执行
results = chain.batch([input1, input2, input3])

# 获取中间步骤
chain = prompt | llm | parser
result = chain.invoke(input_data, config={"callbacks": [callback]})
```

---

## 十、总结

| 概念 | LCEL写法 | 说明 |
|------|---------|------|
| 基本链 | `prompt \| llm \| parser` | 最常用 |
| 传参 | `RunnablePassthrough()` | 原样传参 |
| 并行 | `RunnableParallel(a=chain1, b=chain2)` | 多任务并行 |
| 分支 | `RunnableBranch(...)` | 条件判断 |
| 自定义 | `RunnableLambda(func)` | 自定义处理 |

---

**下一篇 → [04_记忆_Memory](04_记忆_Memory.md)**
