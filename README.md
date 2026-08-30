# 基于LangGraph的RAG智能问答系统

基于 LangGraph 构建的 Agentic RAG（检索增强生成）系统，面向 **专业知识场景**，支持**稠密 + 稀疏混合检索 + 重排序**、**意图分类与自动路由**、**SSE 流式输出**。

## 架构概览

```
用户提问 → query_analysis(意图分类)
                ↓
         retrieve(稠密+稀疏混合检索, top 20+20 → RRF → Rerank)
                ↓
         generate(基于检索文档+对话历史, 流式生成)
                ↓
         SSE 流式推送到前端(Thinking/Stats/Sources/Token/Done)
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 工作流引擎 | **LangGraph** — 4 节点 Agentic 决策流，意图分类 + 自动路由 |
| 稠密向量 | **qwen3.7-text-embedding** (DashScope) — 2048 维语义检索 |
| 稀疏向量 | **BAAI/bge-m3** (FlagEmbedding) — 词汇权重精确匹配 |
| 检索融合 | **RRF** 倒数秩融合 + **BGE-Reranker-v2-M3** 交叉编码器重排序 |
| 向量数据库 | **Milvus** — IVF_FLAT(稠密) + SPARSE_INVERTED_INDEX(稀疏) 混合索引 |
| LLM | **Qwen-Max / Qwen-Plus** (阿里云百炼 DashScope API) |
| 后端 | **FastAPI + Uvicorn** — SSE 流式推送 |
| 前端 | **Streamlit** — 双栏布局，检索链路可视化 |
| 存储 | **MySQL** — 聊天历史持久化 |
| 基础设施 | **Docker Compose** — MySQL / Milvus / etcd / MinIO |

## 核心特性

- **混合检索**：稠密向量（语义）+ 稀疏向量（关键词）+ RRF 融合 + CrossEncoder 重排序
- **Agentic 决策**：LLM 自主判断检索文档质量，不合格则改写搜索词重新检索（最多 3 轮）
- **文档评分**：每篇检索结果 LLM 逐篇打分 1-5，丢弃不相关文档，附带理由
- **幻觉自省**：生成答案后 LLM 对比源文档逐一核验，发现无依据声明则重新生成
- **SSE 流式输出**：Token 级打字机效果 + 检索链路面板（dense#X + sparse#Y → RRF → rerank#Z）
- **父子块策略**：200 token 子块精确定位 + 1000 token 父块给 LLM 提供丰富上下文

## 快速开始

### 1. 环境要求

- Python 3.11+
- Docker Desktop
- 阿里云百炼 DashScope API Key

### 2. 启动基础设施

```bash
docker-compose -f "docker-compose.yml" up -d
docker-compose -f "docker-compose-vis.yml" up -d
```

启动 MySQL、Milvus、etcd、MinIO 四个容器。

### 3. 配置环境变量

编辑 `.env`，填入 API Key：

```env
QWEN_API_KEY=sk-your-key-here
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-max
QWEN_EMBEDDING_MODEL=text-embedding-v4
```

### 4. 安装依赖

```bash
pip install fastapi uvicorn streamlit langchain langchain-openai langchain-community langgraph \
    pymilvus sqlalchemy pymysql sentence-transformers FlagEmbedding \
    pymupdf docx2txt python-dotenv tenacity pydantic
```

### 5. 启动后端

```bash

python app.py
```

### 6. 启动前端

```bash
streamlit run SmartQuery/web/streamlit_app.py
```

浏览器打开 `http://localhost:8501`。

## 使用流程

1. 侧边栏上传 PDF/DOCX/TXT 文档（第一批测试数据在 `data/` 目录）
2. 提问知识问题 → 左栏观察检索链路
3. Stats 面板显示稠密/稀疏命中数 → RRF 融合数 → Rerank 精排数
4. Trace 面板展示每条来源的密集#X + 稀疏#Y → RRF → rerank#Z 链路
5. 文档评分面板显示通过/丢弃的文档及理由
6. 侧边栏历史记录可展开查看过往问答

## 项目结构

```
SmartQuery — 基于 LangGraph 的智能问答系统/
├── app.py                    # 启动入口
├── .env                      # 环境变量
├── docker-compose.yml        # 基础设施容器
├── data/                     # 测试数据（TXT 文档）
├── SmartQuery/
│   ├── backend/
│   │   ├── main.py           # FastAPI 应用（/chat, /chat/stream, /upload, /history）
│   │   ├── config.py         # 配置加载
│   │   └── database/
│   │       ├── milvus.py     # Milvus 向量库操作
│   │       └── mysql.py      # MySQL 聊天记录
│   ├── rag/
│   │   ├── agent.py          # LangGraph 4 节点 Agentic 工作流
│   │   ├── embedding.py      # 稠密 + 稀疏嵌入
│   │   ├── retriever.py      # 混合检索 + RRF + 重排序
│   │   └── ingest.py         # 文档加载与入库
│   └── web/
│       └── streamlit_app.py  # Streamlit 双栏前端
```

## Demo 演示剧本

1. 上传 `data/` 下的 6 份 IT 知识 TXT
2. 问 "B+树索引的原理是什么？"
3. 观察：
   - Thinking 面板：query_analysis → retrieve → generate
   - Stats 面板：稠密命中 / 稀疏命中 / RRF 融合 / Rerank 数量
   - Trace 面板：MySQL 文档 dense#1 + sparse#3 → rerank#1，Python 文档 dense#2 + sparse#1 → rerank 拉低
   - 答案流式逐字输出

## License

MIT
