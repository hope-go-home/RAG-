# Wix 企业知识库智能问答系统

基于 **WixQA**（Wix 帮助中心）企业知识库、**LangGraph** 构建的检索增强问答系统。覆盖**稠密 + 稀疏混合检索 + RRF 融合 + 交叉编码器重排**、**Agentic 决策流**、**SSE 流式输出**，配套**文章级检索评测**与**中英双语前端**。

> 知识库语料：WixQA `wix_kb_corpus`（6221 篇英文帮助文章，3 种 `article_type`），问答标注来自 `wixqa_expertwritten`。

---

## 架构概览

```
用户提问
   │
   ▼
query_analysis  意图分类（闲聊直答 / 知识问答）+ 查询改写
   │
   ▼
retrieve        稠密 + 稀疏混合检索 → RRF 融合 → bge-reranker 重排（可按 article_type 过滤）
   │
   ▼
generate        结合检索文档与对话历史，按提问语言流式生成答案
   │
   ▼
SSE 流式推送    thinking / sources(含标题+原文链接) / stats / token / answer / done
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 工作流引擎 | **LangGraph** — 4 节点 Agentic 决策流，意图分类 + 自动路由 + 重检 |
| 稠密向量 | **qwen3.7-text-embedding** (DashScope) — 2048 维语义检索 |
| 稀疏向量 | **BAAI/bge-m3** (FlagEmbedding) — 词汇权重精确匹配 |
| 检索融合 | **RRF** 倒数秩融合 + **BGE-Reranker-v2-M3** 交叉编码器重排 |
| 向量数据库 | **Milvus** — IVF_FLAT(稠密) + SPARSE_INVERTED_INDEX(稀疏)，字段含 title/url |
| LLM | **Qwen / 兼容 OpenAI 协议模型**（阿里云百炼 DashScope） |
| 后端 | **FastAPI + Uvicorn** — SSE 流式推送 |
| 前端 | **Vue 3 + Vite + Pinia** — 中英双语，检索链路可视化 |
| 存储 | **MySQL** — 聊天历史 / 用户 / 文档登记 / 审计 / 反馈 |
| 认证 | **JWT + bcrypt** — 角色在 Token，服务端解析；管理员可上传/管理 |
| 数据集 | **HuggingFace datasets** — WixQA 语料与问答转换 |

---

## 核心特性

- **混合检索** — 稠密（语义）+ 稀疏（关键词）+ RRF 融合 + CrossEncoder 重排
- **文章级检索评测** — 以 WixQA 的 `article_ids` 为 ground truth，输出 Recall@k / MRR@k / nDCG@k
- **Agentic 决策** — 意图分类自动路由；重排分数不足时由 LLM 改写查询重检（最多 3 轮）
- **SSE 流式输出** — Token 级打字机效果，并实时展示检索链路（dense#X + sparse#Y → RRF → rerank#Z）
- **来源溯源** — 每条来源携带文章标题与 **Wix 原文链接**，可点击跳转
- **按类型过滤** — 支持按 `article_type`（article / feature_request / known_issue）限定检索
- **中英双语前端** — 一键切换；检索控制台式可视化
- **反馈闭环** — 答案 👍/👎 落库，供 badcase 分析
- **企业能力** — JWT 认证 + 管理员 RBAC、审计日志、文档生命周期（增量 upsert / 软删除 / 重建索引）、提示注入防护、扫描件 OCR
- **断点续跑** — 批量入库按 source 幂等，崩溃/重跑不产生重复块

---

## 知识库数据（WixQA）

| config | 内容 | 规模 |
|--------|------|------|
| `wix_kb_corpus` | Wix 帮助中心全文（`id / url / title / contents / article_type`） | 6221 篇 → **63561 块** |
| `wixqa_expertwritten` | 真实客服工单 + 专家答案（`question / answer / article_ids`） | 200 题 |

`scripts/load_wixqa.py` 把语料转成本项目目录结构（`data/wixqa_corpus/<article_type>/<title>__<id8>.md`），并生成评测标注 `SmartQuery/evaluation/golden_wixqa.json`。

---

## 快速开始

### 1. 环境要求
Python 3.11+、Node.js 18+、Docker、DashScope API Key

### 2. 启动基础设施
```bash
docker compose -f docker-compose.yml up -d
docker compose -f docker-compose-vis.yml up -d
```

### 3. 配置环境变量
编辑 `.env`：
```env
QWEN_API_KEY=sk-your-key-here
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-max
QWEN_EMBEDDING_MODEL=qwen3.7-text-embedding
MYSQL_DATABASE=smart_query
# 可选：检索集合与稠密维度（默认 enterprise_kb_docs / 2048）
# RAG_COLLECTION=enterprise_kb_docs
# RAG_DIM=2048
# 可选：每用户每日 LLM token 预算（0 = 不限制）
# DAILY_TOKEN_BUDGET=500000
```

### 4. 安装依赖
```bash
pip install -r requirements.txt
```

### 5. 转换并入库 WixQA（首次约 2–4 小时，瓶颈是 CPU 上的 bge-m3 稀疏编码）
```bash
python scripts/load_wixqa.py                       # 下载语料 + 生成 golden
python scripts/load_corpus.py --corpus-dir data/wixqa_corpus --collection enterprise_kb_docs
```
> 支持断点续跑：崩溃后重跑会跳过已入库且内容未变的文章。

### 6. 启动后端
```bash
python app.py          # http://localhost:8000
```

### 7. 启动前端
```bash
cd frontend
npm install
npm run dev            # http://localhost:5173
```
默认管理员：`admin / admin123`。

---

## 检索评测

文章级评测（ground truth = `article_ids`）：
```bash
python SmartQuery/evaluation/eval_retrieval.py \
  --collection enterprise_kb_docs \
  --golden SmartQuery/evaluation/golden_wixqa.json \
  --limit 100 \
  --strategies dense_only sparse_only hybrid_rrf hybrid_rerank
```

100 题结果（WixQA 文章级，63561 块）：

| 策略 | R@1 | R@10 | Hit@1 | Hit@10 |
|------|-----|------|-------|--------|
| dense_only | 0.287 | 0.743 | 0.350 | 0.820 |
| hybrid_rrf | 0.285 | 0.760 | 0.350 | 0.840 |
| **hybrid_rerank（生产）** | **0.330** | **0.812** | **0.380** | **0.890** |

> **召回诊断**：稠密检索里 gold 有 **92%** 进 top-50 块（平均第 5.7 名）→ **召回已饱和，瓶颈在首位排序**，且语料中大量近似文档（About X / Using X）。
> **已实测无效/更差**：HyDE、多查询、maxP（多段取 max）、标题加权、换 bge-m3 稠密（Hit@10 0.89→0.84）——见 `scripts/exp_rerank_*.py`。

评测报告（`retrieval_eval.json/.md`）由前端「评测结果」页通过 `GET /evaluation` 展示；CI 用 `scripts/eval_gate.py` 对 `SmartQuery/evaluation/baseline.json` 做**回归门禁**。

---

## 可观测性与成本控制

- **指标**：`GET /metrics`（Prometheus）— HTTP 请求数/延迟、检索耗时与命中数、LLM token
- **请求追踪**：每个请求生成 `X-Request-ID` 并贯穿日志
- **限流**：登录按 IP、提问/上传/反馈按用户（进程内固定窗口；多副本应换 Redis）
- **成本熔断**：按用户每日累计 token，超 `DAILY_TOKEN_BUDGET` 拒绝新提问

---

## 项目结构

```
Wix企业知识库智能问答系统/
├── app.py                        # 后端启动入口
├── requirements.txt
├── docker-compose.yml            # 基础设施容器（MySQL / Milvus / etcd / MinIO）
├── data/
│   └── wixqa_corpus/             # WixQA 语料（按 article_type 分目录）+ _meta.json
├── scripts/
│   ├── load_wixqa.py             # WixQA 语料转换 + golden 生成
│   ├── load_corpus.py            # 批量入库（幂等 + 断点续跑）
│   ├── build_ablation.py         # 分块策略消融实验
│   └── create_user.py
├── SmartQuery/
│   ├── backend/
│   │   ├── main.py               # FastAPI（/chat/stream, /upload, /feedback, /evaluation）
│   │   ├── auth.py               # JWT + RBAC
│   │   └── database/
│   │       ├── milvus.py         # schema: text/parent_text/doc_type/source/title/url
│   │       └── mysql.py          # 聊天/用户/文档/审计/反馈
│   ├── rag/
│   │   ├── agent.py              # LangGraph 工作流
│   │   ├── retriever.py          # 混合检索 + RRF + 重排（含 LRU 缓存）
│   │   ├── embedding.py          # 稠密 + 稀疏嵌入
│   │   └── ingest.py             # 分块 + 类型路由 + 加载
│   └── evaluation/
│       ├── eval_retrieval.py     # 检索评测（文章级 / 父块级）
│       ├── eval_answer.py        # 回答评测（LLM-as-Judge）
│       └── golden_wixqa.json     # WixQA 评测标注
├── frontend/                     # Vue 3 + Vite + Pinia（中英双语）
│   └── src/{i18n,api,stores,views,components,utils}
└── tests/                        # 单元测试（pytest）
```

---

## 文档

- [架构设计文档](docs/ARCHITECTURE.md)
- [评测方法文档](docs/EVALUATION.md)

## License

MIT
