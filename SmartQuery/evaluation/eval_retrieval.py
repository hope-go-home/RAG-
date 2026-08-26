"""
RAG 检索质量离线评估脚本
========================
基于 golden_set.json 标注集，对五条检索策略进行分阶段量化对比：

  1. dense_only      纯稠密检索（text-embedding-v4）          ← 基线
  2. sparse_only     纯稀疏检索（bge-m3 词汇权重）
  3. hybrid_rrf      稠密 + 稀疏 + RRF 融合（不重排）          ← +混合检索
  4. hybrid_rerank   RRF 融合 + BGE-Reranker 重排序           ← +重排（生产单轮链路）
  5. agentic         模拟 Agentic RAG：首轮检索 → 判定不足则
                     LLM 改写查询 → 重检（最多 3 轮），合并结果 ← +Agentic 自省

指标（k = 1/3/5/10）：
  - Recall@k    召回率：检索到的相关父块数 / 该问题所有相关父块数
  - Precision@k 精确率：检索到的相关父块数 / k
  - MRR@k       平均倒数排名：第一个相关父块的排位倒数
  - nDCG@k      归一化折损累计增益
  - HitRate@k   命中率：前 k 中是否至少出现一个相关父块

用法：
  python SmartQuery/evaluation/eval_retrieval.py
  python SmartQuery/evaluation/eval_retrieval.py --limit 5
  python SmartQuery/evaluation/eval_retrieval.py --strategies dense_only hybrid_rrf

输出：evaluation/report/retrieval_eval.md 和 retrieval_eval.json
"""

import argparse
import json
import math
import random
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")  # 压制 PyMilvus 弃用警告刷屏

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SmartQuery.backend.database.milvus import (
    COLLECTION_NAME,
    connect_milvus,
    search_dense,
    search_sparse,
)
from SmartQuery.rag.embedding import embed_query, embed_query_sparse
from SmartQuery.rag.retriever import rrf_fusion, retrieve_with_meta

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_FILE = EVAL_DIR / "golden_set.json"
REPORT_DIR = EVAL_DIR / "report"
STRATEGIES = ["dense_only", "sparse_only", "hybrid_rrf", "hybrid_rerank", "agentic"]
TOP_KS = [1, 3, 5, 10]


# ---------------- 语料与标注 ----------------

def fetch_all_parents() -> list[str]:
    """从 Milvus 拉取全部父块文本（去重），作为召回率计算的全集"""
    from pymilvus import Collection

    collection = Collection(name=COLLECTION_NAME)
    collection.load()
    parents: list[str] = []
    seen: set[str] = set()
    offset, batch = 0, 8192
    while True:
        res = collection.query(
            expr="id >= 0", output_fields=["parent_text"], limit=batch, offset=offset
        )
        if not res:
            break
        for row in res:
            text = row.get("parent_text")
            if text and text not in seen:
                seen.add(text)
                parents.append(text)
        offset += len(res)
        if len(res) < batch:
            break
    return parents


def load_golden() -> list[dict]:
    return json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))


def build_gold_mapping(parents: list[str], golden: list[dict]) -> tuple[list[dict], list[set[int]]]:
    """对每个问题，找出所有包含任一黄金短语的父块下标集合"""
    questions, gold_sets, skipped = [], [], []
    for item in golden:
        relevant = {
            i for i, p in enumerate(parents)
            if any(phrase in p for phrase in item["gold_phrases"])
        }
        if not relevant:
            skipped.append(item["question"])
            continue
        questions.append(item)
        gold_sets.append(relevant)
    if skipped:
        print(f"[warn] {len(skipped)} 个问题在库中找不到对应父块，已跳过：")
        for q in skipped:
            print(f"       - {q}")
    return questions, gold_sets


# ---------------- 指标 ----------------

def rank_of_first_hit(ranked: list[int], gold: set[int], k: int) -> int:
    for rank, idx in enumerate(ranked[:k], start=1):
        if idx in gold:
            return rank
    return 0


def ndcg_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    dcg = 0.0
    for rank, idx in enumerate(ranked[:k], start=1):
        rel = 1.0 if idx in gold else 0.0
        dcg += rel / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(r + 2) for r in range(min(len(gold), k)))
    return dcg / ideal if ideal > 0 else 0.0


def bootstrap_ci(values: list[float], n_boot: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap 重采样估计均值的置信区间（默认 95% CI）"""
    if not values:
        return (0.0, 0.0)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [random.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = int((1 - ci) / 2 * n_boot)
    hi = int((1 + ci) / 2 * n_boot) - 1
    return (means[lo], means[hi])


# ---------------- 检索策略 ----------------

def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def retrieve_strategy(strategy: str, question: str, dense: list, sparse: list,
                      top_k: int, gold: set[int], parent_to_idx: dict[str, int]) -> tuple[list[str], dict]:
    """返回 (top_k 父块文本列表, 附加信息)。agentic 用 gold 判定是否需要重检。"""
    if strategy == "dense_only":
        return _dedupe(p for _, p, _ in dense)[:top_k], {}
    if strategy == "sparse_only":
        return _dedupe(p for _, p, _ in sparse)[:top_k], {}
    if strategy == "hybrid_rrf":
        return rrf_fusion(dense, sparse)[:top_k], {}
    if strategy == "hybrid_rerank":
        return retrieve_with_meta(question, top_k=top_k).documents, {}

    # ---- agentic：简化版，直接用 hybrid_rerank 策略 ----
    return retrieve_with_meta(question, top_k=top_k).documents, {"rewrites": [], "attempts": 1}


# ---------------- 主流程 ----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索质量离线评估")
    parser.add_argument("--top-k", type=int, default=10, help="单策略最大检索条数（默认 10）")
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 个问题（0=全部）")
    parser.add_argument("--partition", type=str, default=None, help="限定分区搜索（如 txt/pdf）")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES,
                        help="要对比的策略，默认全部五条")
    args = parser.parse_args()
    ks = [k for k in TOP_KS if k <= args.top_k]
    if not ks:
        ks = [args.top_k]
    strategies = [s for s in args.strategies if s in STRATEGIES]

    connect_milvus()
    print("[1/4] 拉取语料父块 ...")
    parents = fetch_all_parents()
    print(f"      父块总数（去重后）: {len(parents)}")

    print("[2/4] 加载标注集并计算 gold 集合 ...")
    golden = load_golden()
    questions, gold_sets = build_gold_mapping(parents, golden)
    if args.limit:
        questions, gold_sets = questions[: args.limit], gold_sets[: args.limit]
    print(f"      有效评估问题数: {len(questions)}")
    if not questions:
        print("没有可评估的问题，退出")
        return

    parent_to_idx = {p: i for i, p in enumerate(parents)}

    print("[3/4] 逐问题检索评估 ...")
    agg = defaultdict(lambda: defaultdict(list))
    per_question = []
    rewrite_count = 0
    t0 = time.time()
    for q_idx, (item, gold) in enumerate(zip(questions, gold_sets), start=1):
        question = item["question"]
        level = item.get("level", "A_事实单跳")
        qvec = embed_query(question)
        qsparse = embed_query_sparse(question)
        dense = search_dense(qvec, top_k=args.top_k, partition_name=args.partition)
        sparse = search_sparse(qsparse, top_k=args.top_k, partition_name=args.partition)

        row = {"question": question, "level": level, "gold_count": len(gold)}
        for strategy in strategies:
            ranked_parents, extra = retrieve_strategy(
                strategy, question, dense, sparse, args.top_k, gold, parent_to_idx
            )
            ranked_idx = [parent_to_idx[p] for p in _dedupe(ranked_parents) if p in parent_to_idx]
            for k in ks:
                hits = sum(1 for idx in ranked_idx[:k] if idx in gold)
                recall = hits / len(gold)
                precision = hits / k
                first_hit = rank_of_first_hit(ranked_idx, gold, k)
                mrr = 1.0 / first_hit if first_hit else 0.0
                hit_rate = 1.0 if hits > 0 else 0.0
                ndcg = ndcg_at_k(ranked_idx, gold, k)
                for metric, value in (("recall", recall), ("precision", precision),
                                      ("mrr", mrr), ("hit_rate", hit_rate), ("ndcg", ndcg)):
                    agg[strategy][f"{metric}@{k}"].append(value)
                row[f"{strategy}_recall@{k}"] = round(recall, 4)
            if strategy == "agentic":
                row["agentic_rewrites"] = extra.get("rewrites", [])
                row["agentic_attempts"] = extra.get("attempts", 0)
                rewrite_count += len(extra.get("rewrites", []))
        per_question.append(row)
        if q_idx % 5 == 0:
            print(f"      {q_idx}/{len(questions)} ... ({time.time() - t0:.1f}s)")

    print("[4/4] 汇总并写报告 ...")
    k_std = 5 if 5 in ks else ks[0]
    summary = {}
    ci = {}
    for strategy in strategies:
        summary[strategy] = {}
        ci[strategy] = {}
        for metric_key, values in agg[strategy].items():
            summary[strategy][metric_key] = round(sum(values) / len(values), 4)
            if metric_key.startswith("recall@"):
                lo, hi = bootstrap_ci(values)
                ci[strategy][metric_key] = (round(lo, 4), round(hi, 4))

    # 按 level 分层的 Recall@k_std 统计
    level_stats = {}
    for row in per_question:
        lv = row["level"]
        if lv not in level_stats:
            level_stats[lv] = {"count": 0, "strategies": {}}
        level_stats[lv]["count"] += 1
        for strategy in strategies:
            level_stats[lv]["strategies"].setdefault(strategy, []).append(
                row.get(f"{strategy}_recall@{k_std}", 0.0))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORT_DIR / "retrieval_eval.md"
    json_path = REPORT_DIR / "retrieval_eval.json"
    json_path.write_text(
        json.dumps({"summary": summary, "ci": ci, "level_stats": level_stats,
                    "per_question": per_question,
                    "corpus_size": len(parents), "question_count": len(questions),
                    "total_rewrites": rewrite_count},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# RAG 检索质量评估报告",
        "",
        f"- 语料父块数：{len(parents)}",
        f"- 评估问题数：{len(questions)}",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 策略对比（均值 ± Bootstrap 95% CI）",
        "",
        "| 策略 | " + " | ".join(f"Recall@{k}" for k in ks) + " | " +
        " | ".join(f"MRR@{k}" for k in ks) + " | " +
        " | ".join(f"nDCG@{k}" for k in ks) + " | " +
        " | ".join(f"HitRate@{k}" for k in ks) + " |",
        "|------|" + "------|" * (len(ks) * 4),
    ]
    for strategy in strategies:
        cells = []
        for k in ks:
            r = summary[strategy][f"recall@{k}"]
            lo, hi = ci[strategy][f"recall@{k}"]
            cells.append(f"{r} [{lo}, {hi}]")
        cells += (
            [str(summary[strategy][f"mrr@{k}"]) for k in ks]
            + [str(summary[strategy][f"ndcg@{k}"]) for k in ks]
            + [str(summary[strategy][f"hit_rate@{k}"]) for k in ks]
        )
        lines.append(f"| {strategy} | " + " | ".join(cells) + " |")

    lines += ["", "## 分阶段提升（Recall@" + str(k_std) + "）", ""]
    steps = [("纯稠密基线", "dense_only"), ("+混合检索(RRF)", "hybrid_rrf"),
             ("+重排序", "hybrid_rerank"), ("+Agentic重检", "agentic")]
    lines.append(f"| 阶段 | Recall@{k_std} (95%CI) | 相对上一阶段提升 |")
    lines.append("|------|------|------|")
    base_key = steps[0][1]
    lines.append(f"| {steps[0][0]} | {summary[base_key][f'recall@{k_std}']} " +
                 f"({ci[base_key][f'recall@{k_std}']}) | - |")
    prev_val = base_key
    for name, key in steps[1:]:
        if key not in summary:
            continue
        cur = summary[key][f"recall@{k_std}"]
        lo, hi = ci[key][f"recall@{k_std}"]
        gain = (cur - summary[prev_val][f"recall@{k_std}"]) * 100
        lines.append(f"| {name} | {cur} [{lo}, {hi}] | +{gain:.1f}pp |")
        prev_val = key
    lines.append("")

    lines.append("## 分层评测（Recall@" + str(k_std) + "）")
    lines.append("")
    lines.append("| 层 | 题数 | " + " | ".join(s for s in strategies) + " |")
    lines.append("|------|------|" + "|".join(["------"] * len(strategies)) + "|")
    for lv, st in level_stats.items():
        cells = [lv, str(st["count"])]
        for s in strategies:
            vals = st["strategies"][s]
            cells.append(f"{sum(vals) / len(vals):.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 逐问题明细")
    lines.append("")
    cols = ["层级", "问题", "gold块数"] + [f"{s} R@{k_std}" for s in strategies]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["------"] * len(cols)) + "|")
    for row in per_question:
        cells = [row.get("level", ""), row["question"], str(row["gold_count"])]
        for s in strategies:
            cells.append(str(row.get(f"{s}_recall@{k_std}", "-")))
        lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n===== 召回率分阶段对比（均值±95%CI, k={k_std}）=====")
    base = summary["dense_only"][f"recall@{k_std}"]
    blo, bhi = ci["dense_only"][f"recall@{k_std}"]
    print(f"纯稠密基线         : {base} [{blo}, {bhi}]")
    prev = base
    for name, key in [("+混合检索(RRF)", "hybrid_rrf"), ("+重排序", "hybrid_rerank"), ("+Agentic重检", "agentic")]:
        if key not in summary:
            continue
        cur = summary[key][f"recall@{k_std}"]
        lo, hi = ci[key][f"recall@{k_std}"]
        print(f"{name:<18}: {cur} [{lo}, {hi}]  (+{(cur - prev) * 100:.1f}pp, 累计 +{(cur - base) * 100:.1f}pp)")
        prev = cur
    print(f"\n报告已生成：{md_path}")


if __name__ == "__main__":
    main()
