# 评测方法说明

本文档说明本系统的离线评测框架：**检索质量评测**（文章级）、**回答质量评测**（LLM-as-Judge）与**回归门禁**。

---

## 一、评测框架总览

| 评测类型 | 脚本 | 数据 | 核心指标 |
|---------|------|------|---------|
| 检索质量 | `SmartQuery/evaluation/eval_retrieval.py` | WixQA `wixqa_expertwritten` | Recall@k / MRR@k / nDCG@k / HitRate@k |
| 回答质量（生成侧） | `SmartQuery/evaluation/eval_answer.py` | 同上 | 答案正确率 / 忠实度 / 幻觉率 |
| 回归门禁 | `scripts/eval_gate.py` | `baseline.json` | 指标跌破阈值即失败 |

---

## 二、数据集与标注

知识库：WixQA `wix_kb_corpus`（**6221 篇**英文帮助文章，`article_type` 三类：article / feature_request / known_issue）。
问答：WixQA `wixqa_expertwritten`（**200 题**真实客服问题 + 专家答案，含 `article_ids`）。

`scripts/load_wixqa.py` 生成标注 `golden_wixqa.json`：

```json
{
  "level": "wixqa_expertwritten",
  "doc_type": "article",
  "question": "Can I start accepting payments while my Wix Payments account is under verification?",
  "gold_sources": ["Wix_Payments_Verification_Process__49d9e88f.md"],
  "reference": "..."
}
```

- `gold_sources`：相关**文章文件名**（= Milvus 的 `source` 字段），用于**文章级**检索评测
- `reference`：参考答案，用于回答评测

> gold 篇数分布：1 篇 148 题、2 篇 46 题、3 篇 6 题（平均 1.29）。

---

## 三、检索质量评测

### 3.1 对比策略

| 策略 | 说明 |
|------|------|
| `dense_only` | 纯稠密（qwen-embedding）← 基线 |
| `sparse_only` | 纯稀疏（BGE-M3 词汇权重） |
| `hybrid_rrf` | 稠密 + 稀疏 + RRF |
| `hybrid_rerank` | RRF + BGE-Reranker-v2-M3 ← 生产链路 |
| `hybrid_multiquery` | 多查询变体 + 重排 |
| `agentic` | 重排分不足时 LLM 改写查询多轮重检 |

### 3.2 指标

| 指标 | 定义 |
|------|------|
| Recall@k | 命中 gold 数 / 该题 gold 总数（文章级） |
| MRR@k | 首个 gold 排位的倒数 |
| nDCG@k | 归一化折损累计增益 |
| HitRate@k | 前 k 是否至少命中一个 gold（**二值**） |

> 说明：**多篇文章作答**会让 Recall@1 有结构性上限 `≈ mean(1/|gold|) = 0.865`；而 **HitRate@1 上限为 1.0**（命中任一 gold 即可）。

### 3.3 运行

```bash
python SmartQuery/evaluation/eval_retrieval.py \
  --collection enterprise_kb_docs \
  --golden SmartQuery/evaluation/golden_wixqa.json \
  --limit 100 --strategies dense_only sparse_only hybrid_rrf hybrid_rerank
```
输出：`evaluation/report/retrieval_eval.md` / `.json`。

### 3.4 结果（100 题，文章级）

| 策略 | R@1 | R@10 | Hit@1 | Hit@10 |
|------|-----|------|-------|--------|
| dense_only | 0.287 | 0.743 | 0.350 | 0.820 |
| hybrid_rrf | 0.285 | 0.760 | 0.350 | 0.840 |
| **hybrid_rerank（生产）** | **0.330** | **0.812** | **0.380** | **0.890** |

**结论与诊断**

- **重排是主要增益**：R@10 0.743 → 0.812，Hit@10 0.82 → 0.89。
- **召回已饱和、瓶颈在排序**：稠密检索里 gold 有 **92%（183/200）进 top-50 块**，平均最佳排名 **5.7**；稀疏 80% / 9.2。→ 提升点不在召回广度，而在首位排序。
- **等权 RRF 会稀释稠密**（稀疏较弱），故采用**稠密主导**融合（0.8:0.2）。
- **已实测无效/更差**：HyDE、多查询（无增益）、maxP 多段取 max（无变化）、标题加权（略降）、换 bge-m3 稠密（Hit@10 0.89→0.84）。见 `scripts/exp_rerank_repr.py`、`scripts/exp_rerank_boost.py`。
- **近似文档**：Wix 帮助中心大量 "About X / Using X" 高度相似，首位命中天然有难度。

---

## 四、回答质量评测（LLM-as-Judge，生成侧）

在检索**之后**评估系统**最终回答**的质量，填补"检索对了 ≠ 答对了"的空白。

| 指标 | 定义 |
|------|------|
| 答案正确率 | 裁判对比系统回答与参考答案，给 correct / partial / incorrect |
| 忠实度 | 裁判**基于检索到的上下文**判断回答是否有据（0–1） |
| 幻觉率 | 含 ≥1 条无上下文支持声明的答案占比 |
| 语义相似度 | 系统回答与参考答案的 embedding 余弦 |

> **关键方法论**：忠实度/幻觉必须用**「检索到的上下文」**判定，而不是参考答案——否则评判依据比生成时实际看到的资料还全，会把有据的陈述误判为幻觉。

裁判模型由 `JUDGE_MODEL` 配置（建议强于生成模型），生成与裁判均 `temperature=0`。

```bash
python SmartQuery/evaluation/eval_answer.py --limit 50 --workers 4   # 抽样 50，4 并发
python SmartQuery/evaluation/eval_answer.py --dry-run                # 只看待跑题数，不调用模型
```

支持**断点续跑**：每题算完即写 `report/answer_eval.checkpoint.jsonl`，中断后重跑自动跳过已完成题（`--no-resume` 可强制全量重跑）。

### 结果（WixQA `wixqa_expertwritten`，抽样 50 题）

| 指标 | 数值 |
|------|------|
| 正确率 (correct) | **80%** |
| correct + partial | **100%** |
| 平均忠实度 | **0.96** |
| 幻觉率（忠实度 < 0.9） | **12%** |
| 平均语义相似度 | 0.87 |

**结论**

- 忠实度 0.96、幻觉率 12%：回答基本有据，编造可控。
- 检索 Hit@10 ≈ 0.89 → 最终答对 0.80，差距与"约 11% 问题 gold 未进 top-10"吻合，说明**丢分主要在检索召回，生成环节未明显拖后腿**。
- 口径说明：该 50 题为分阶段运行（部分题目使用了不同的生成/裁判模型），属**混合口径**；如需对外统一口径，可用 `--no-resume` 以同一模型全量重跑。

---

## 五、回归门禁

```bash
python scripts/eval_gate.py            # 对比 report 与 baseline.json
python scripts/eval_gate.py --tol 0.05 # 自定义容忍度
```
CI 会在有评测报告时执行；指标跌破 `基线 - 容忍度` 则失败，阻断劣化。

---

## 六、注意事项

1. 评测前需先入库语料：`python scripts/load_wixqa.py && python scripts/load_corpus.py --corpus-dir data/wixqa_corpus`
2. 重排在 CPU 上较慢，大量评测耗时较长（可 `--limit` 抽样）
3. LLM-as-Judge 建议固定 `temperature=0`
4. 报告同时给 Recall 与 HitRate，并声明天花板，避免被单指标误导
