# 02 - 提示词：Prompts

---

## 一、PromptTemplate（提示词模板）

### 为什么需要模板？

```python
# ❌ 不用模板 —— 每次都要手写完整字符串
llm.invoke("请用中文翻译这句话：Hello, world! 只返回翻译结果")
llm.invoke("请用中文翻译这句话：Good morning! 只返回翻译结果")

# ✅ 用模板 —— 变量替换
template = "请用中文翻译这句话：{text} 只返回翻译结果"
prompt = template.format(text="Hello, world!")
llm.invoke(prompt)
```

### 基本用法

```python
from langchain_core.prompts import PromptTemplate

# 定义模板
template = PromptTemplate.from_template(
    "你是一个{role}。请回答下面问题：\n{question}"
)

# 填充变量
prompt = template.invoke({
    "role": "资深 Python 工程师",
    "question": "解释一下装饰器"
})

print(prompt.text)
# 输出：你是一个资深 Python 工程师。请回答下面问题：
# 解释一下装饰器

# 传给 LLM
result = llm.invoke(prompt)
```

### 模板语法

```python
# 基础变量
template = PromptTemplate.from_template("{name}今年{age}岁")

# 多个变量
template = PromptTemplate.from_template(
    "主题：{topic}\n要求：{requirements}\n字数：{word_count}字以内"
)

# 部分变量（先固定一部分）
template = PromptTemplate.from_template(
    "语言：{language}\n问题：{question}"
)
partial_template = template.partial(language="中文")
prompt = partial_template.invoke({"question": "什么是机器学习？"})

# 带默认值的变量
template = PromptTemplate.from_template(
    "使用{language}回答：{question}",
    partial_variables={"language": "中文"}
)
prompt = template.invoke({"question": "你好吗？"})  # language 自动填充
```

---

## 二、ChatPromptTemplate（对话模板）

用于构造多轮对话的消息列表：

```python
from langchain_core.prompts import ChatPromptTemplate

# 方式1：用元组列表
template = ChatPromptTemplate([
    ("system", "你是一个{role}，用{language}回答。"),
    ("human", "{question}")
])

prompt = template.invoke({
    "role": "数学老师",
    "language": "中文",
    "question": "什么是勾股定理？"
})
# prompt 是一个消息列表
# [SystemMessage, HumanMessage]

result = llm.invoke(prompt)

# 方式2：用 MessageTemplate（更灵活）
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

system_template = SystemMessagePromptTemplate.from_template(
    "你是一个{role}，回答要简洁专业。"
)
human_template = HumanMessagePromptTemplate.from_template("{question}")

template = ChatPromptTemplate([system_template, human_template])
prompt = template.invoke({"role": "医生", "question": "感冒了怎么办？"})
```

### 多轮对话模板

```python
template = ChatPromptTemplate([
    ("system", "你是一个导游，用中文回答。"),
    ("human", "介绍一下{place}"),
    ("ai", "{ai_response}"),
    ("human", "还有其他推荐吗？")
])

prompt = template.invoke({
    "place": "北京故宫",
    "ai_response": "故宫位于北京中轴线的中心，是明清两代的皇家宫殿。"
})
# 生成 4 条消息
```

---

## 三、FewShotPromptTemplate（少样本模板）

给 LLM 提供几个示例，让它学会模式：

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# 示例数据
examples = [
    {"input": "苹果", "output": "水果，红色或绿色，脆甜"},
    {"input": "西瓜", "output": "水果，绿色外皮，多汁"},
    {"input": "胡萝卜", "output": "蔬菜，橙色，富含维生素A"},
]

# 单个示例的模板
example_template = PromptTemplate.from_template(
    "输入：{input}\n输出：{output}"
)

# FewShot 模板
fewshot_prompt = FewShotPromptTemplate(
    examples=examples,               # 示例
    example_prompt=example_template,  # 示例格式
    prefix="请对以下物品进行分类和描述：\n",  # 前缀
    suffix="输入：{item}\n输出：",         # 后缀（放主问题）
    input_variables=["item"]
)

prompt = fewshot_prompt.invoke({"item": "西兰花"})
print(prompt.text)
# 请对以下物品进行分类和描述：
# 
# 输入：苹果
# 输出：水果，红色或绿色，脆甜
# 
# 输入：西瓜
# 输出：水果，绿色外皮，多汁
# 
# 输入：胡萝卜
# 输出：蔬菜，橙色，富含维生素A
#
# 输入：西兰花
# 输出：
```

### 动态选择示例（ExampleSelector）

```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 自动选择最相关的示例
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Chroma,
    k=2  # 选 2 个最相关的
)

fewshot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_template,
    prefix="请分类和描述：",
    suffix="输入：{item}\n输出：",
    input_variables=["item"]
)
```

---

## 四、PipelinePromptTemplate（组合模板）

将多个小模板组合成大模板：

```python
from langchain_core.prompts import PipelinePromptTemplate

# 子模板
greeting = PromptTemplate.from_template("你好，{name}！")
context = PromptTemplate.from_template("我们正在讨论{topic}。")
question = PromptTemplate.from_template("请回答：{question}")

# 主模板（引用子模板）
full_prompt = PromptTemplate.from_template(
    "{greeting}\n{context}\n{question}"
)

pipeline = PipelinePromptTemplate(
    final_prompt=full_prompt,
    pipeline_prompts={
        "greeting": greeting,
        "context": context,
        "question": question
    }
)

prompt = pipeline.invoke({
    "name": "张三",
    "topic": "机器学习",
    "question": "什么是监督学习？"
})
```

---

## 五、MessagesPlaceholder（消息占位符）

用于动态插入消息列表（如历史对话）：

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate([
    ("system", "你是一个助手"),
    MessagesPlaceholder(variable_name="history"),  # 历史消息
    ("human", "{question}")
])

# 注入历史
from langchain_core.messages import AIMessage, HumanMessage

prompt = template.invoke({
    "history": [
        HumanMessage(content="我叫张三"),
        AIMessage(content="你好张三！")
    ],
    "question": "我叫什么名字？"
})
# LLM 会看到完整的对话历史
```

---

## 六、Prompt Hub（提示词仓库）

LangChain 提供在线提示词仓库：

```python
from langchain import hub

# 拉取热门模板
prompt = hub.pull("rlm/rag-prompt")        # RAG 模板
prompt = hub.pull("hwchase17/llama-2-prompt")  # Llama 2 模板
prompt = hub.pull("langchain-ai/react-chat-template")  # Agent 模板

# 直接使用
result = llm.invoke(prompt.invoke({"question": "你好"}))
```

---

## 七、总结：模板选择指南

| 模板类型 | 适用场景 | 示例 |
|---------|---------|------|
| `PromptTemplate` | 简单文本模板 | 翻译、摘要 |
| `ChatPromptTemplate` | 对话模型 | 聊天、问答 |
| `FewShotPromptTemplate` | 给示例让模型学习 | 分类、提取 |
| `PipelinePromptTemplate` | 组合多个模板 | 复杂工作流 |
| `MessagesPlaceholder` | 动态插入消息 | 历史对话 |

---

**下一篇 → [03_链_Chains](03_链_Chains.md)**
