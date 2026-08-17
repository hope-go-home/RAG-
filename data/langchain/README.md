# LangChain 从零到实战 —— 完整学习教程

> 目标人群：Python 开发者 | 框架：LangChain | 主题：LLM 应用开发

---

## 快速开始

```bash
# 1. 安装
pip install langchain langchain-openai langchain-chroma

# 2. 设置 API Key
$env:OPENAI_API_KEY="sk-your-key"

# 3. 运行示例
python examples/01_basic_llm.py

# 4. 运行完整应用
streamlit run chat_app.py
```

---

## 学习路线

| 文件 | 内容 | 难度 |
|------|------|------|
| **[00_前言](00_前言_LangChain介绍与环境搭建.md)** | LangChain 是什么、架构总览、安装 | ⭐ |
| **[01_模型](01_模型_Models.md)** | ChatOpenAI、消息类型、流式/批量调用 | ⭐ |
| **[02_提示词](02_提示词_Prompts.md)** | PromptTemplate、ChatPromptTemplate、FewShot | ⭐⭐ |
| **[03_链](03_链_Chains.md)** | LCEL、RunnableParallel、自定义 Runnable | ⭐⭐⭐ |
| **[04_记忆](04_记忆_Memory.md)** | BufferMemory、WindowMemory、SummaryMemory | ⭐⭐⭐ |
| **[05_检索RAG](05_检索_RAG.md)** | Loader → Splitter → Embeddings → VectorStore | ⭐⭐⭐ |
| **[06_智能体](06_智能体_Agents.md)** | Tool、Agent、AgentExecutor、ReAct | ⭐⭐⭐⭐ |
| **[07_回调](07_回调_Callbacks.md)** | BaseCallbackHandler、事件监听 | ⭐⭐ |
| **[08_完整实战](08_完整实战项目.md)** | RAG + Agent 双模式聊天应用 | ⭐⭐⭐⭐ |

---

## 示例代码

| 文件 | 演示内容 | 运行命令 |
|------|---------|---------|
| `examples/01_basic_llm.py` | LLM 基础调用 | `python examples/01_basic_llm.py` |
| `examples/02_prompt_template.py` | 提示词模板 | `python examples/02_prompt_template.py` |
| `examples/03_chains.py` | 链 | `python examples/03_chains.py` |
| `examples/04_memory.py` | 记忆 | `python examples/04_memory.py` |
| `examples/05_rag.py` | RAG 问答 | `python examples/05_rag.py` |
| `examples/06_agent.py` | 智能体 | `python examples/06_agent.py` |
| `chat_app.py` | 完整聊天应用 | `streamlit run chat_app.py` |

---

## 核心组件关系图

```
                    ┌──────────┐
                    │  Models  │ ← 各种 LLM
                    └────┬─────┘
                         ↓
                    ┌──────────┐
                    │  Prompts │ ← 构造输入
                    └────┬─────┘
                         ↓
             ┌───────────┴───────────┐
             │        Chains         │ ← 串联组件
             └───────────┬───────────┘
                         ↓
        ┌────────────────┴────────────────┐
        │                                  │
   ┌────┴─────┐                     ┌─────┴────┐
   │  Memory  │                     │  Agents  │
   │  (记忆)  │                     │ (智能体)  │
   └──────────┘                     └──────────┘
        │                                  │
   ┌────┴─────┐                     ┌─────┴────┐
   │  RAG     │                     │  Tools   │
   │ (检索)   │                     │  (工具)   │
   └──────────┘                     └──────────┘
```

---

## 环境要求

- Python 3.9+
- 一个 LLM API Key（OpenAI / Anthropic / Ollama 本地模型）
- 推荐：VS Code + Python 插件
