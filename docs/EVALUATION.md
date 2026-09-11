# 评测方法说明

本文档说明企业知识库智能问答系统的离线评测框架，包括检索质量评测、回答质量评测和分块策略消融实验。

---

## 一、评测框架总览

| 评测类型 | 脚本 | 评测对象 | 核心指标 |
|---------|------|---------|---------|
| 检索质量 | `SmartQuery/evaluation/eval_retrieval.py` | 5 条检索链路 | Recall@k / MRR@k / nDCG@k |
| 回答质量 | `SmartQuery/evaluation/eval_answer.py` | LLM 生成答案 | 忠实度 / 相关性 / 完整性 |
| 分块消融 | `scripts/build_ablation.py` | 3 种分块策略 | Recall@k 对比 |

统一标注集：`SmartQuery/evaluation/golden_set.json`（90 题）。

---

## 二、标注集设计

### 2.1 难度分层（7 层）

| 层级 | 名称 | 题数 | 考察点 |
|------|------|------|--------|
| A | 事实单跳 | 22 | 从单篇文档检索一个事实 |
| B | 术语理解 | 12 | 解释专业术语/概念 |
| C | 语义改写 | 10 | 同义、口语化表达下的召回能力 |
| D | 多跳推理 | 14 | 跨 2+ 文档组合信息 |
| E | 易混淆 | 10 | 区分相似概念 |
| F | 跨文档 | 14 | 跨文档类型整合信息 |
| G | 结构化查询 | 8 | SOP 步骤、条款号、表格数据 |

### 2.2 文档类型分布

| 文档类型 | 题数 |
|---------|------|
| 规章制度 | 50 |
| FAQ | 23 |
| 操作流程SOP | 12 |
| 员工手册 | 3 |
| 数据报表 | 2 |

### 2.3 标注字段

```json
{
  "level": "A_事实单跳",
  "doc_type": "规章制度",
  "question": "公司的年假天数是如何规定的？",
  "gold_phrases": ["年休假5天", "年休假10天", "年休假15天"],
  "reference": "根据考勤管理制度..."
}
```

- `gold_phrases`：用于检索评测——父块包含任一短语即视为相关
- `reference`：用于回答评测——LLM 评审的参考答案

---

## 三、检索质量评测

### 3.1 对比策略

| 策略 | 说明 |
|------|------|
| `dense_only` | 纯稠密检索（qwen3.7-text-embedding）← 基线 |
| `sparse_only` | 纯稀疏检索（bge-m3 词汇权重） |
| `hybrid_rrf` | 稠密 + 稀疏 + RRF 融合 |
| `hybrid_rerank` | RRF 融合 + BGE-Reranker 重排序 ← 生产链路 |
| `agentic` | 重排概率不足时 LLM 改写查询，多轮重检并合并 |

### 3.2 指标

| 指标 | 定义 |
|------|------|
| Recall@k | 检索到的相关父块数 / 该问题全部相关父块数 |
| Precision@k | 检索到的相关父块数 / k |
| MRR@k | 第一个相关父块排位的倒数 |
| nDCG@k | 归一化折损累计增益 |
| HitRate@k | 前 k 是否至少命中一个相关父块 |

所有指标在 k = 1/3/5/10 上计算，Recall 附带 **Bootstrap 95% 置信区间**（1000 次重采样）。

### 3.3 运行

```bash
python SmartQuery/evaluation/eval_retrieval.py
python SmartQuery/evaluation/eval_retrieval.py --limit 10 --strategies hybrid_rrf hybrid_rerank
```

输出：`evaluation/report/retrieval_eval.md` 和 `retrieval_eval.json`。

### 3.4 结果（示例模板）

| 策略 | Recall@5 (95% CI) | MRR@5 | nDCG@5 |
|------|-------------------|-------|--------|
| dense_only | 0.xxx [0.xxx, 0.xxx] | 0.xxx | 0.xxx |
| sparse_only | 0.xxx | 0.xxx | 0.xxx |
| hybrid_rrf | 0.xxx | 0.xxx | 0.xxx |
| hybrid_rerank | 0.xxx | 0.xxx | 0.xxx |
| agentic | 0.xxx | 0.xxx | 0.xxx |

> 实际数值在语料入库后运行脚本生成。

---

## 四、回答质量评测（LLM-as-Judge）

### 4.1 三维评分

| 维度 | 说明 | 分值 |
|------|------|------|
| 忠实度 Faithfulness | 回答是否基于检索文档，有无幻觉 | 1-5 |
| 相关性 Relevance | 回答是否针对用户问题 | 1-5 |
| 完整性 Completeness | 回答是否覆盖关键信息 | 1-5 |

### 4.2 短语匹配

额外计算 `gold_phrases` 在回答中的命中比例，作为客观辅助指标，避免 LLM 评审的波动。

### 4.3 评审 Prompt

```
你是一个严格的RAG系统评估专家。请对以下问答对进行评分。
用户问题：{question}
参考答案：{reference}
系统回答：{answer}
请从忠实度、相关性、完整性三个维度评分（1-5分），输出JSON。
```

### 4.4 运行

```bash
python SmartQuery/evaluation/eval_answer.py
python SmartQuery/evaluation/eval_answer.py --limit 5 --verbose
```

输出：`evaluation/report/answer_eval.md` 和 `answer_eval.json`。

---

## 五、分块策略消融实验（核心亮点）

### 5.1 实验设计

**控制变量法**：语料、检索链路、标注集全部不变，只改分块策略。

```
同一份企业语料
    ├── kb_fixed     固定长度分块（512 字符一刀切）    ← 基线
    ├── kb_header    标题感知分块（仅按 Markdown 标题）
    └── kb_adaptive  类型自适应分块（按文档类型选策略） ← 本方案

用同一套 golden_set 对 3 个集合各跑一遍评测
对比 Recall@5
```

### 5.2 三种策略对比

| 策略 | 规章制度 | FAQ | SOP | 技术文档 | 员工手册 | 数据报表 |
|------|---------|-----|-----|---------|---------|---------|
| 固定长度 | 一刀切 | 一刀切 | 一刀切 | 一刀切 | 一刀切 | 一刀切 |
| 标题感知 | 按标题 | 按标题 | 按标题 | 按标题 | 按标题 | 按标题 |
| **类型自适应** | 按条款 | 按问答对 | 按步骤 | 按接口+代码块 | 按章节 | 整表保留 |

### 5.3 运行

```bash
# 1. 构建 3 个集合
python scripts/build_ablation.py

# 2. 分别评测
python SmartQuery/evaluation/eval_retrieval.py --collection kb_fixed --strategies hybrid_rerank
python SmartQuery/evaluation/eval_retrieval.py --collection kb_header --strategies hybrid_rerank
python SmartQuery/evaluation/eval_retrieval.py --collection kb_adaptive --strategies hybrid_rerank
```

### 5.4 结果（示例模板）

| 分块策略 | Recall@5 | 相对基线提升 |
|---------|----------|-------------|
| 固定长度（基线） | 0.xxx | — |
| 标题感知 | 0.xxx | +xx.x pp |
| **类型自适应** | **0.xxx** | **+xx.x pp** |

> 该表证明类型自适应分块相比固定分块在检索召回上的提升，是项目核心量化结论。

---

## 六、评测注意事项

1. **评测前需先入库语料**：`python scripts/load_corpus.py --reset`
2. **Bootstrap CI 反映稳定性**：置信区间越窄说明结果越可靠
3. **LLM-as-Judge 需固定 temperature=0**：减少评审波动
4. **分层统计**：按 `level` 和 `doc_type` 分别统计，定位薄弱环节
5. **短语匹配为客观锚点**：与 LLM 评分交叉验证
