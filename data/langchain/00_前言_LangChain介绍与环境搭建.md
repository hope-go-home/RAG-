# 00 - 前言：LangChain 介绍与环境搭建

---

## 一、LangChain 是什么？

**LangChain** 是一个用于构建 **大语言模型（LLM）应用** 的 Python 框架。

> **一句话：** LangChain = 让你把 LLM（如 GPT-4、文心一言、Claude）和各种工具"链"接起来，构建强大的 AI 应用。

### 它能做什么？

- 🤖 **聊天机器人** —— 带记忆的对话机器人
- 📚 **RAG 问答系统** —— 基于你自己的文档回答问题
- 🔧 **Agent 智能体** —— 让 AI 使用搜索引擎、计算器、API 等工具
- 📊 **数据分析** —— 用自然语言查询数据库
- 🔗 **复杂工作流** —— 多步骤推理、多模型协作

---

## 二、LangChain 的核心架构

LangChain 由 6 大核心模块组成：

```
┌─────────────────────────────────────────────┐
│                 LangChain                    │
├─────────────────────────────────────────────┤
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │  Models  │  │ Prompts  │  │  Chains   │  │
│  │ (模型)   │  │ (提示词) │  │  (链)    │  │
│  └─────────┘  └──────────┘  └───────────┘  │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Memory  │  │  Index   │  │  Agents   │  │
│  │ (记忆)  │  │ (索引)   │  │ (智能体)  │  │
│  └─────────┘  └──────────┘  └───────────┘  │
└─────────────────────────────────────────────┘
```

### 组件速览

| 组件 | 作用 | 类比 |
|------|------|------|
| **Models** | 封装各种 LLM（GPT、Claude、本地模型） | 大脑 |
| **Prompts** | 构建输入给 LLM 的提示词 | 怎么问问题 |
| **Chains** | 把多个步骤串联成工作流 | 流水线 |
| **Memory** | 让 LLM 记住对话历史 | 短期记忆 |
| **Index/RAG** | 从外部文档检索信息 | 查资料 |
| **Agents** | 让 LLM 决定调用什么工具 | 请助手干活 |

---

## 三、底层原理（重要）

### 1. LLM 是无状态的

```python
# 每次调用都是独立的，LLM 不记得之前说过什么
llm.invoke("我叫张三")
llm.invoke("我叫什么？")  # ❌ 它不知道你叫张三
```

**解决方案：** 每次调用时把历史对话也塞进 prompt。

### 2. Chain 的本质

```python
# Chain = 一个函数调用链
def chain(input_text):
    prompt = prompt_template.format(text=input_text)
    result = llm.invoke(prompt)
    return output_parser.parse(result)
```

### 3. LangChain Expression Language (LCEL)

LCEL 是 LangChain 最新推荐的定义 Chain 的方式，用 `|` 管道符串联：

```python
chain = prompt | llm | output_parser
# 等价于: output_parser(llm(prompt(input)))
```

---

## 四、环境搭建

### 步骤 1：安装

```bash
# 基础安装
pip install langchain

# 选择一个 LLM 后端
pip install langchain-openai     # OpenAI（GPT）
pip install langchain-anthropic  # Claude
pip install langchain-community  # 社区模型
pip install langchain-ollama     # 本地模型（Ollama）

# 工具库
pip install langchainhub         # 提示词仓库
pip install langgraph            # 图工作流（高级）
```

### 步骤 2：设置 API Key

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-key-here"

# 或写入 .env 文件
# OPENAI_API_KEY=sk-your-key-here

# 或写在代码里（仅学习用，不要提交到 GitHub）
```

### 步骤 3：验证安装

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
result = llm.invoke("你好，你是谁？")
print(result.content)
```

> 如果没有 OpenAI 的 Key，可以用 Ollama 运行本地模型：`pip install langchain-ollama`

---

## 五、项目目录结构

```
D:\项目\langchain\
├── 00_前言_LangChain介绍与环境搭建.md
├── 01_模型_Models.md
├── 02_提示词_Prompts.md
├── 03_链_Chains.md
├── 04_记忆_Memory.md
├── 05_检索_RAG.md
├── 06_智能体_Agents.md
├── 07_回调_Callbacks.md
├── 08_完整实战项目.md
├── README.md
├── AGENTS.md
├── chat_app.py              ← 完整聊天机器人
├── examples/
│   ├── 01_basic_llm.py      ← LLM 基础使用
│   ├── 02_prompt_template.py ← 提示词模板
│   ├── 03_chains.py          ← 链的用法
│   ├── 04_memory.py          ← 记忆功能
│   ├── 05_rag.py             ← RAG 问答
│   └── 06_agent.py           ← 智能体
```

---

## 六、学习建议

1. **要有 LLM 的 API Key** —— 推荐先搞到 OpenAI 的 key（或本地 Ollama）
2. **边学边跑代码** —— 每个例子都运行一遍
3. **理解 LCEL** —— `|` 管道符是 LangChain 的核心语法
4. **按顺序学** —— 组件之间有依赖关系

---

**下一篇 → [01_模型_Models](01_模型_Models.md)**
