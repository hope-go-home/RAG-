# Wix 企业知识库智能问答系统 — 架构设计文档

## 1. 系统概述

面向企业知识库场景的端到端 RAG（Retrieval-Augmented Generation）系统。知识库为公开基准 **WixQA**（Wix 帮助中心 6221 篇英文文章），基于 **LangGraph** 构建 Agentic 工作流，采用**稠密 + 稀疏混合检索 + RRF 融合 + 交叉编码器重排**，SSE 流式输出，前端中英双语。

核心目标：检索精度、可控生成、可观测、可评测。

## 2. 整体架构

```
Vue 3 前端(中英双语)
   │  REST / SSE
   ▼
FastAPI 后端  ── /chat/stream(SSE) · /upload · /feedback · /evaluation · /metrics
   │
   ▼
LangGraph Agent：query_analysis → retrieve → (rewrite) → generate
   │
   ▼
Milvus（稠密 dense_vector 2048 + 稀疏 sparse_vector）
   │
MySQL（聊天/用户/文档登记/审计/反馈/token 用量）
```

## 3. 数据与分块

- 语料：`scripts/load_wixqa.py` 把 WixQA `wix_kb_corpus` 转为
  `data/wixqa_corpus/<article_type>/<title>__<id8>.md`，并把 `wixqa_expertwritten` 转为评测标注 `golden_wixqa.json`。
- 文档类型（`article_type`）：`article` / `feature_request` / `known_issue`。
- 分块：父块 ≈1000 字、子块 ≈350 字（`RecursiveCharacterTextSplitter`）；**文档标题前置**到父块（`【title】…`），提升区分度。
- 入库幂等：按 `source` 先删后插；支持**断点续跑**（内容未变则跳过）。

## 4. 检索链路

```
问题
 ├─ 稠密：qwen-embedding(2048)  → Milvus IVF_FLAT  ─┐
 └─ 稀疏：BGE-M3 词汇权重        → Milvus SPARSE    ─┤
                                                     ▼
                              RRF 融合（k=60，稠密:稀疏=0.8:0.2）
                                                     ▼
                              按「文章(source)」去重 → 每篇取最佳块
                                                     ▼
                              BGE-Reranker-v2-M3 重排 → top-k
```

- **稠密用原始问题**嵌入（语义完整），**稀疏用去停用词的关键词**。
- **稠密主导融合**：本语料实测稠密明显强于稀疏，等权 RRF 会稀释排序。
- **按文章去重后再重排**：避免同一篇的多个块挤占候选，提升文章级召回。
- 检索缓存：进程内 LRU + TTL（多副本应换 Redis）。

## 5. Agentic 工作流（LangGraph）

| 节点 | 职责 |
|------|------|
| `query_analysis` | 意图分类（rag_query / chit_chat / meta / follow_up）+ 查询改写 |
| `retrieve` | 混合检索 + 重排；可按 `article_type` 过滤 |
| `rewrite` | 重排分不足时由 LLM 改写查询重检（最多 3 轮） |
| `generate` | 结合检索文档与历史，**按提问语言**流式生成 |
| `direct_answer` | 闲聊/系统询问，不检索直接回答 |

## 6. Milvus Schema

```python
fields = [
    id(int64, auto),
    text(varchar),          # 子块
    parent_text(varchar),   # 父块（含【标题】前缀）
    dense_vector(float[2048]),   # qwen 稠密
    sparse_vector(sparse_float), # bge-m3 稀疏
    doc_type(varchar),      # article_type
    source(varchar),        # 文章文件名（引用溯源 / 去重键）
    title(varchar), url(varchar),
]
# 索引：dense IVF_FLAT(IP, nlist=128) · sparse SPARSE_INVERTED_INDEX(IP) · doc_type INVERTED
# 分区：按文件格式（pdf/docx/txt/md/xlsx）
```

## 7. 接口与 SSE

- `POST /chat/stream`（SSE）、`POST /chat`、`POST /upload`、`GET /documents`、`POST /feedback`、`GET /evaluation`、`GET /metrics`、`/health`、`/ready`
- SSE 事件：`thinking` / `stats` / `sources`（含标题+原文链接）/ `token` / `answer` / `done` / `error`

## 8. 企业级能力

- **认证授权**：JWT + 最小 admin 角色（上传/管理）；审计日志。
- **文档生命周期**：登记表 + 增量 upsert + 软删除 + 重建索引。
- **安全**：提示注入防护（`<资料>` 分隔 + 指令隔离）、引用溯源。
- **可观测**：`/metrics`（请求/延迟/检索耗时/token）+ request-id。
- **成本与限流**：按用户/IP 限流 + 每日 token 预算熔断。
- **交付**：Docker Compose 一键全栈 + GitHub Actions（lint / 测试 / 评测门禁 / 前端构建）。

## 附录：关键决策

| 决策 | 选择 | 理由 |
|------|------|------|
| Agent 框架 | LangGraph | 状态机清晰、支持重检与流式 |
| 稠密模型 | qwen-embedding(2048) | 实测优于本地 bge-m3 稠密（Hit@10 0.89 vs 0.84） |
| 稀疏模型 | BGE-M3 | 与稠密互补，精确匹配关键词 |
| 重排 | BGE-Reranker-v2-M3 | 实测最大增益来源（R@10 +6.4 / Hit@10 +6） |
| 向量库 | Milvus | 原生支持稠密+稀疏混合 |
| 流式 | SSE | 单向推送，实现简单 |
