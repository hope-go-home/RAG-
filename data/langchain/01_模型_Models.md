# 01 - 模型：Models

---

## 一、模型概述

LangChain 抽象了两种模型接口：

| 类型 | 类名 | 输入 | 输出 | 适用场景 |
|------|------|------|------|---------|
| **LLM** | `BaseLLM` | 字符串 | 字符串 | 文本补全 |
| **ChatModel** | `BaseChatModel` | 消息列表 | 消息对象 | 对话（推荐） |

> **推荐：** 优先使用 `ChatModel`，它的消息结构更灵活。

---

## 二、ChatOpenAI（最常用）

### 安装

```bash
pip install langchain-openai
```

### 基础用法

```python
from langchain_openai import ChatOpenAI

# 创建模型（最简单的形式）
llm = ChatOpenAI(model="gpt-4o-mini")

# 调用
result = llm.invoke("用一句话解释什么是量子计算")
print(result.content)
```

### 完整参数

```python
llm = ChatOpenAI(
    model="gpt-4o-mini",        # 模型名称
    temperature=0.7,            # 随机性：0~2，越高越有创意
    max_tokens=1024,            # 最大输出 token 数
    timeout=30,                 # 请求超时时间（秒）
    max_retries=2,              # 失败重试次数
    api_key="sk-xxx",           # API Key（推荐用环境变量）
    base_url="https://xxx.com/v1",  # 自定义端点（支持代理）
    verbose=True,               # 是否输出调试信息
    model_kwargs={"top_p": 0.9}  # 其他模型参数
)
```

### `invoke` 方法详解

```python
from langchain_core.messages import HumanMessage, SystemMessage

# 基础字符串调用
result = llm.invoke("你好")
print(result.content)      # 输出文本

# 消息列表调用（推荐）
messages = [
    SystemMessage(content="你是一个专业的 Python 导师，回答要简洁。"),
    HumanMessage(content="什么是装饰器？")
]
result = llm.invoke(messages)
print(result.content)

# 获取完整响应信息
print(result.response_metadata)  # 包含 token 用量等
```

### 流式输出（打字机效果）

```python
# 流式输出
for chunk in llm.stream("给我讲一个关于猫的故事"):
    print(chunk.content, end="", flush=True)

# 事件流（更详细信息）
async for event in llm.astream_events("你好", version="v1"):
    print(event)
```

### 批量调用

```python
results = llm.batch([
    "你好",
    "你是谁？",
    "今天天气怎么样？"
])
for r in results:
    print(r.content)
```

---

## 三、消息类型

| 消息类 | 说明 | 角色 |
|--------|------|------|
| `SystemMessage` | 系统指令（设定角色、规则） | system |
| `HumanMessage` | 用户消息 | user |
| `AIMessage` | AI 回复 | assistant |
| `ToolMessage` | 工具返回结果 | tool |

### 典型对话结构

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="你是一个友好的助手，用中文回答。"),
    HumanMessage(content="你好！"),
    AIMessage(content="你好！有什么可以帮助你的吗？"),
    HumanMessage(content="Python 和 JavaScript 哪个好学？")
]

result = llm.invoke(messages)
print(result.content)
```

---

## 四、BaseLLM（传统 LLM）

某些模型不提供 Chat 接口，只能用 LLM 接口：

```python
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7)
result = llm.invoke("量子计算是什么？")
print(result)
```

**区别：**
- `OpenAI` → 输入字符串，输出字符串
- `ChatOpenAI` → 输入消息列表，输出消息对象

---

## 五、本地模型 Ollama

### 安装 Ollama

```bash
# 下载安装：https://ollama.com
# 然后拉取模型
ollama pull qwen2.5:7b     # 通义千问 7B
ollama pull llama3.1:8b    # Meta Llama 3.1
```

### 代码使用

```bash
pip install langchain-ollama
```

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.7,
    num_predict=2048,          # 最大 token 数
    top_k=40,                  # 采样参数
    top_p=0.9                  # 采样参数
)

result = llm.invoke("用中文介绍 LangChain")
print(result.content)
```

---

## 六、多模型切换

```python
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

def get_llm(provider="openai"):
    if provider == "openai":
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    elif provider == "ollama":
        return ChatOllama(model="qwen2.5:7b", temperature=0.7)
    else:
        raise ValueError(f"未知提供商: {provider}")

llm = get_llm("openai")
result = llm.invoke("你好")
print(result.content)
```

---

## 七、绑定运行时参数

```python
# 绑定停止词
llm_with_stop = llm.bind(stop=["\n"])
result = llm_with_stop.invoke("列出 5 种水果名称")

# 绑定格式（JSON 输出）
llm_with_json = llm.bind(response_format={"type": "json_object"})
result = llm_with_json.invoke("列出 3 种编程语言及其特点，返回 JSON")
print(result.content)
```

---

## 八、关键概念总结

```python
# 1. 创建模型
llm = ChatOpenAI(model="gpt-4o-mini")

# 2. 三种调用方式
llm.invoke("你好")          # 普通调用
llm.stream("你好")          # 流式调用
llm.batch(["你好", "嗨"])   # 批量调用

# 3. 消息列表（推荐）
messages = [
    SystemMessage(content="系统指令"),
    HumanMessage(content="用户输入")
]
llm.invoke(messages)

# 4. 获取结构化响应
result = llm.invoke("你好")
result.content              # 文本内容
result.response_metadata    # 元数据
```

---

**下一篇 → [02_提示词_Prompts](02_提示词_Prompts.md)**
