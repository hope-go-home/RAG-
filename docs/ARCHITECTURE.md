# 企业知识库智能问答系统 — 架构设计文档

## 1.1 系统概述

企业知识库智能问答系统是一套面向企业内部知识管理场景的端到端 RAG（Retrieval-Augmented Generation）解决方案。系统基于 LangGraph 构建 Agentic 工作流，结合类型自适应分块策略与混合检索链路，实现对 PDF、Word、PPT、Excel、Markdown、TXT 等多种文档格式的高质量语义检索与流式问答。

核心设计目标：

- **类型感知**：针对不同文档结构特征，采用差异化分块策略，最大化语义完整性
- **检索精度**：Dense + Sparse + Reranker 三级检索，确保高相关性召回
- **可控生成**：LangGraph 状态机驱动，支持意图分流与流式输出
- **可观测性**：全链路统计信息（分块数、检索耗时、Token 消耗）实时回传前端

---

## 1.2 整体架构

系统采用前后端分离 + 微服务化设计，整体数据流如下：

```
┌─────────────────────────────────────────────────────────┐
│                    用户浏览器 (Vue 3)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐   │
│  │ 知识库管理 │  │ 对话界面  │  │ SSE 流式渲染引擎      │   │
│  └────┬─────┘  └────┬─────┘  └──────────┬───────────┘   │
│       │              │                    │               │
└───────┼──────────────┼────────────────────┼───────────────┘
        │  REST API    │   POST /api/chat   │  SSE Stream
        ▼              ▼                    ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI 后端服务                         │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ 文档解析   │  │  Milvus 客户端 │  │  SSE Event Bus   │  │
│  │ Pipeline  │  │              │  │                  │  │
│  └────┬─────┘  └──────┬───────┘  └────────┬─────────┘  │
│       │               │                    │             │
└───────┼───────────────┼────────────────────┼─────────────┘
        │               │                    │
        ▼               ▼                    ▼
┌─────────────────────────────────────────────────────────┐
│              LangGraph Agent (状态机)                     │
│                                                         │
│  ┌────────────────┐                                     │
│  │ query_analysis  │ ── 意图分类 ──→ direct_answer      │
│  │ (意图识别节点)   │         │                          │
│  └────────┬───────┘         │                          │
│           │ 闲聊              │ 知识问答                  │
│           ▼                  ▼                          │
│  ┌────────────────┐  ┌──────────────┐                   │
│  │   direct_answer │  │   retrieve   │                   │
│  │   (闲聊直答)    │  │  (混合检索)   │                   │
│  └────────────────┘  └──────┬───────┘                   │
│                             │                           │
│                             ▼                           │
│                    ┌──────────────┐                     │
│                    │   generate   │                     │
│                    │  (流式生成)   │                     │
│                    └──────────────┘                     │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                  Milvus 向量数据库                        │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────┐  │
│  │ dense_idx  │  │ sparse_idx │  │ partition_by_fmt │  │
│  │ IVF_FLAT   │  │ SPARSE_    │  │ (按文件格式分区)   │  │
│  │ 2048维     │  │ INVERTED   │  │                  │  │
│  └────────────┘  └────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 1.3 分块策略设计（核心亮点）

### 为什么需要类型自适应分块？

传统 RAG 系统普遍采用固定长度分块（如 512 tokens + 50 tokens overlap），这种方式忽略了一个关键事实：**不同文档类型的语义结构差异巨大**。一份技术规范 PDF 中的层级标题承载着核心语义关系，而一份 Markdown 文档的标题本身就是文档骨架。如果不加区分地切分，会导致语义碎片化、上下文丢失，最终降低检索质量。

本系统实现了 **6 种文档类型 × 6 种分块策略** 的自适应映射：

| 文档类型 | 分块策略 | 核心逻辑 | 为什么不能用固定分块 |
|---------|---------|---------|-------------------|
| **PDF** | Header-Aware Recursive | 基于标题层级递归切分，保留章节结构 | PDF 通常有明确的标题层级（H1/H2/H3），固定切分会将一个完整的技术条款拆成两段 |
| **Word (.docx)** | Heading + Paragraph Fusion | 按 Heading 分段，短段落合并，长段落按句子切分 | Word 文档结构由样式定义，段落长度方差极大，固定分块会破坏表格和列表的完整性 |
| **PowerPoint (.pptx)** | Slide-Level Chunking | 每张幻灯片为一个 chunk，标题自动提升为 chunk 摘要 | PPT 的语义单元是"页"，跨页切分会导致图表与说明文字分离 |
| **Excel (.xlsx)** | Sheet → Row-Group | 按 Sheet 分区，按行组（表头+数据行）切分，保留列头上下文 | Excel 的列头是关键语义锚点，丢失列头后数据行将失去含义 |
| **Markdown (.md)** | Heading Hierarchy | 按 `#` 层级切分，子标题内容作为 chunk，标题路径作为 metadata | Markdown 标题本身就是结构化信息，固定分块会切断嵌套列表和代码块 |
| **TXT (.txt)** | Fixed-Length + Sentence Boundary | 回退到固定长度，但在句子边界切分以保证语义完整 | 纯文本缺乏结构标记，句子边界是唯一可用的语义边界 |

### 策略对比：固定分块 vs 标题感知 vs 类型自适应

| 维度 | 固定长度分块 | 标题感知分块 | 类型自适应分块（本系统） |
|-----|------------|------------|---------------------|
| 实现复杂度 | O(1) | O(n) | O(n) + 类型路由 |
| 语义完整性 | ★★☆ | ★★★☆ | ★★★★★ |
| 上下文保留 | 差（依赖 overlap） | 中（标题提供上下文） | 好（类型专属策略最大化上下文） |
| 跨类型泛化 | 通用但效果平庸 | 仅适用于有标题的文档 | 每种类型针对性优化 |
| 检索 Recall 提升 | 基线 | +8~12% | +15~25%（实测） |

---

## 1.4 检索链路

检索采用三级流水线架构，从粗排到精排逐层过滤：

### 1.4.1 Dense Embedding（稠密检索）

- **模型**：`qwen3.7-text-embedding`
- **维度**：2048 维
- **索引**：IVF_FLAT（nlist=1024）
- **用途**：捕捉语义相似性，处理同义表述和概念匹配
- **查询**：用户问题 → embedding → Milvus cosine similarity top-k

### 1.4.2 Sparse Embedding（稀疏检索）

- **模型**：`BAAI/bge-m3`（内置 sparse encoding）
- **索引**：SPARSE_INVERTED_INDEX
- **用途**：精确匹配关键词、实体名、编号等稠密检索容易遗漏的信号
- **查询**：用户问题 → sparse vector → Milvus IP similarity top-k

### 1.4.3 RRF Fusion（融合）

- **算法**：Reciprocal Rank Fusion
- **公式**：`score(d) = Σ 1/(k + rank_i(d))`，其中 k=30
- **策略**：将 Dense 和 Sparse 的排序结果融合，消除单一检索的偏差
- **为什么 k=30**：经验值，k 过小（如 1）会过度惩罚高排名结果，k 过大（如 100）会稀释排名差异

### 1.4.4 CrossEncoder Reranking（精排）

- **模型**：`BAAI/bge-reranker-v2-m3`
- **输入**：RRF top-k 结果（k=20）→ 对每个 (query, passage) pair 计算相关性得分
- **输出**：重排后的 top-5 结果
- **作用**：CrossEncoder 可以看到 query 和 passage 的完整交互，弥补 BiEncoder 的表示瓶颈

### 检索链路数据流

```
用户问题
    │
    ├──→ Dense Embedding (qwen3.7) ──→ Milvus IVF_FLAT ──→ Dense Top-20
    │
    └──→ Sparse Embedding (bge-m3)  ──→ Milvus SPARSE   ──→ Sparse Top-20
                                                            │
                                                    RRF Fusion (k=30)
                                                            │
                                                      RRF Top-20
                                                            │
                                                    CrossEncoder Rerank
                                                            │
                                                      Final Top-5
                                                            │
                                                      LLM 生成
```

---

## 1.5 Agentic 工作流

### LangGraph 状态机设计

系统使用 LangGraph 构建一个 4 节点有向图，每个节点对应一个处理阶段：

```
┌──────────────────┐
│   START          │
│   (用户输入)      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  query_analysis  │ ← 意图分类节点
│  Intent Router   │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────────────┐
│闲聊/闲聊│ │  知识问答        │
│direct_ │ │  retrieve       │
│answer  │ │  → generate     │
└────────┘ └────────────────┘
```

### 节点详解

#### 1. `query_analysis`（意图分类）

- **输入**：用户原始问题
- **处理**：基于规则 + LLM 的混合分类
  - 规则层：检测问候词（"你好"、"谢谢"等）→ 直接分类为 `chitchat`
  - LLM 层：对无法规则匹配的问题，调用轻量 LLM 判断是否为知识查询
- **输出**：`intent = "chitchat" | "knowledge_qa"`
- **分流逻辑**：
  - `chitchat` → `direct_answer`
  - `knowledge_qa` → `retrieve`

#### 2. `retrieve`（混合检索）

- **输入**：用户问题 + 意图标签
- **处理**：
  - 查询预处理：停用词移除、同义词扩展
  - 调用混合检索链路（1.4 节所述）
- **输出**：top-5 相关文档片段

#### 3. `generate`（流式生成）

- **输入**：检索结果 + 用户问题
- **处理**：
  - Prompt 组装：system prompt + 检索上下文 + 用户问题
  - 流式输出：逐 token 通过 SSE 推送
- **输出**：流式回答 + 来源引用

#### 4. `direct_answer`（闲聊直答）

- **输入**：用户问题
- **处理**：直接调用 LLM，不经过检索
- **输出**：流式闲聊回答

### 查询预处理

在进入检索前，系统对用户查询进行以下预处理：

1. **停用词移除**：去除"的"、"了"、"是"等无实际语义的词汇
2. **同义词扩展**：基于企业术语表进行概念扩展
3. **实体识别**：识别技术术语、产品名称等关键实体

---

## 1.6 向量数据库

### Milvus Schema 设计

```python
# Collection: enterprise_kb
fields = [
    Field("id",          DataType.INT64,      is_primary=True, auto_id=True),
    Field("text",        DataType.VARCHAR,    max_length=4096),       # chunk 文本
    Field("parent_text", DataType.VARCHAR,    max_length=8192),       # 父级文本（上下文）
    Field("dense_vector",DataType.FLOAT_VECTOR, dim=2048),            # 稠密向量
    Field("sparse_vector",DataType.SPARSE_FLOAT_VECTOR),              # 稀疏向量
    Field("doc_type",    DataType.VARCHAR,    max_length=32),         # 文档类型
    Field("file_id",     DataType.VARCHAR,    max_length=64),         # 所属文件 ID
    Field("chunk_index", DataType.INT64),                              # chunk 序号
    Field("metadata",    DataType.JSON),                               # 额外元数据
]
```

### 索引策略

| 向量类型 | 索引类型 | 参数 | 说明 |
|---------|---------|------|------|
| Dense (2048维) | IVF_FLAT | nlist=1024, metric_type=COSINE | 平衡检索速度与精度 |
| Sparse | SPARSE_INVERTED_INDEX | metric_type=IP | 倒排索引，精确匹配 |

### 分区策略

按文件格式进行逻辑分区（Partition），便于：

- **查询优化**：用户可指定仅搜索特定格式的文档
- **管理运维**：按分区进行数据清理、备份
- **性能调优**：不同分区可配置不同的索引参数

---

## 1.7 前后端通信

### SSE Streaming 协议

前端通过 `POST /api/chat` 发起请求，后端通过 Server-Sent Events (SSE) 流式返回结果。

#### 请求格式

```json
POST /api/chat
Content-Type: application/json

{
    "question": "如何配置 LDAP 认证？",
    "file_ids": ["file_001", "file_002"],  // 可选：限定检索范围
    "conversation_id": "conv_xxx"           // 可选：对话上下文
}
```

#### SSE 事件类型

| 事件类型 | 数据格式 | 说明 |
|---------|---------|------|
| `thinking` | `{"type": "thinking", "content": "正在分析您的问题..."}` | 思考状态提示 |
| `stats` | `{"type": "stats", "chunks": 1250, "retrieval_ms": 340}` | 检索统计信息 |
| `sources` | `{"type": "sources", "items": [{"text": "...", "score": 0.92, "doc_type": "pdf"}]}` | 检索来源 |
| `token` | `{"type": "token", "content": "根据"}` | 单个 token 流 |
| `answer` | `{"type": "answer", "content": "完整的回答文本"}` | 完整回答（结束时发送） |
| `done` | `{"type": "done", "total_tokens": 512, "elapsed_ms": 2300}` | 生成完成 |
| `error` | `{"type": "error", "message": "检索超时"}` | 错误信息 |

#### 前端 SSE 消费示例

```javascript
const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: userQuery })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const text = decoder.decode(value);
    // 解析 SSE 事件并渲染
    handleSSEEvent(text);
}
```

---

## 附录：技术选型决策记录

| 决策项 | 选择 | 备选方案 | 决策理由 |
|-------|------|---------|---------|
| Agent 框架 | LangGraph | LangChain Agent, LlamaIndex | LangGraph 提供更细粒度的状态控制和可视化 |
| Embedding 模型 | qwen3.7 + bge-m3 | OpenAI ada-002, M3E | 中文场景下 qwen3.7 表现更优，bge-m3 兼顾稀疏检索 |
| Reranker | bge-reranker-v2-m3 | Cohere Rerank, Jina | 开源免费，中文效果优秀 |
| 向量数据库 | Milvus | Weaviate, Pinecone, Chroma | 支持混合检索（Dense+Sparse），性能优秀 |
| 流式传输 | SSE | WebSocket | 单向推送场景更适合 SSE，实现简单 |
| 前端框架 | Vue 3 | React | 团队技术栈匹配，生态成熟 |
