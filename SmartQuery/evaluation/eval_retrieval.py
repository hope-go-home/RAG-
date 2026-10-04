"""
RAG 检索质量离线评估脚本
========================
基于 golden_set.json 标注集，对五条检索策略进行分阶段量化对比：

  1. dense_only      纯稠密检索（qwen3.7-text-embedding）      ← 基线
  2. sparse_only     纯稀疏检索（bge-m3 词汇权重）
  3. hybrid_rrf      稠密 + 稀疏 + RRF 融合（不重排）          ← +混合检索
  4. hybrid_rerank   RRF 融合 + BGE-Reranker 重排序           ← +重排（生产单轮链路）
  5. agentic         真实 Agentic RAG：首轮检索 → 重排概率不足则
                     LLM 改写查询 → 重检（最多 3 轮），合并多轮结果 ← +Agentic 自省

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
from SmartQuery.rag.retriever import rrf_fusion, retrieve_with_meta, retrieve_multi_query
from SmartQuery.rag.agent import (
    safe_llm_invoke,
    REWRITE_PROMPT,
    MAX_RETRIEVAL_ATTEMPTS,
    RERANK_SCORE_THRESHOLD,
)

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_FILE = EVAL_DIR / "golden_wixqa.json"
REPORT_DIR = EVAL_DIR / "report"
STRATEGIES = ["dense_only", "sparse_only", "hybrid_rrf", "hybrid_rerank", "hybrid_multiquery", "agentic"]
TOP_KS = [1, 3, 5, 10]


# ---------------- 语料与标注 ----------------

def fetch_all_parents(collection_name: str | None = None) -> list[tuple[str, str]]:
    """从 Milvus 拉取全部父块文本（去重），返回 (parent_text, source) 列表。

    单次查询结果有大小上限（父块文本可能很大），失败时自动减半批大小重试。
    """
    from pymilvus import Collection
    from pymilvus.exceptions import MilvusException

    name = collection_name or COLLECTION_NAME
    collection = Collection(name=name)
    collection.load()
    parents: list[tuple[str, str]] = []
    seen: set[str] = set()
    offset, batch = 0, 200
    while True:
        try:
            res = collection.query(
                expr="id >= 0", output_fields=["parent_text", "source"],
                limit=batch, offset=offset,
            )
        except MilvusException as e:
            if batch > 5 and "limit size" in str(e):
                batch = max(5, batch // 2)
                continue
            raise
        if not res:
            break
        for row in res:
            text = row.get("parent_text")
            if text and text not in seen:
                seen.add(text)
                parents.append((text, row.get("source", "")))
        offset += len(res)
        if len(res) < batch:
            break
    return parents


def load_golden(golden_file: Path | None = None) -> list[dict]:
    return json.loads((golden_file or GOLDEN_FILE).read_text(encoding="utf-8"))


def build_gold_mapping(parents: list, golden: list[dict]) -> tuple[list[dict], list[set]]:
    """对每个问题，给出相关键集合。

    两种口径：
      - gold_sources：相关键 = 文章文件名（source），与检索结果的 source 直接匹配（WixQA，文章级）
      - gold_phrases：相关键 = 包含黄金短语的父块下标（旧中文语料，父块级）
    """
    questions, gold_sets, skipped = [], [], []
    for item in golden:
        sources = item.get("gold_sources")
        if sources:
            gold = set(sources)
        else:
            phrases = item.get("gold_phrases", [])
            gold = {i for i, (p, _) in enumerate(parents)
                    if any(phrase in p for phrase in phrases)}
        if not gold:
            skipped.append(item["question"])
            continue
        questions.append(item)
        gold_sets.append(gold)
    if skipped:
        print(f"[warn] {len(skipped)} 个问题缺少可判定的 gold，已跳过：")
        for q in skipped[:10]:
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
                      top_k: int, gold: set, parent_to_idx: dict[str, int],
                      collection_name: str | None = None, source_level: bool = False) -> tuple[list, dict]:
    """返回 (top_k 检索键列表, 附加信息)。

    source_level=True 时键为来源文件名（文章级评测），否则为父块文本（父块级评测）。
    """
    key_ix = 4 if source_level else 1  # search 元组中 source[4] / parent_text[1]
    src_of_parent = {}
    for it in list(dense) + list(sparse):
        src_of_parent.setdefault(it[1], it[4])

    if strategy == "dense_only":
        return _dedupe(x[key_ix] for x in dense)[:top_k], {}
    if strategy == "sparse_only":
        return _dedupe(x[key_ix] for x in sparse)[:top_k], {}
    if strategy == "hybrid_rrf":
        ranked = rrf_fusion(dense, sparse)
        if source_level:
            return _dedupe(src_of_parent[p] for p in ranked if src_of_parent.get(p))[:top_k], {}
        return ranked[:top_k], {}
    if strategy == "hybrid_rerank":
        res = retrieve_with_meta(
            question, top_k=top_k, collection_name=collection_name,
            candidate_pool=100 if source_level else None,
            rerank_pool=40 if source_level else None,
        )
        return ([s.source for s in res.sources] if source_level else res.documents), {}
    if strategy == "hybrid_multiquery":
        res = retrieve_multi_query(
            question, top_k=top_k, collection_name=collection_name,
            candidate_pool=100 if source_level else None,
            rerank_pool=40 if source_level else None,
        )
        return ([s.source for s in res.sources] if source_level else res.documents), {}

    # ---- agentic：质量不足时 LLM 改写检索词重检，合并多轮结果 ----
    keys: list = []
    rewrites: list[str] = []
    query = question
    attempts = 0
    while attempts < MAX_RETRIEVAL_ATTEMPTS:
        attempts += 1
        res = retrieve_with_meta(
            query, top_k=top_k, collection_name=collection_name,
            candidate_pool=100 if source_level else None,
            rerank_pool=40 if source_level else None,
        )
        if source_level:
            keys = _dedupe(keys + [s.source for s in res.sources])
        else:
            keys = _dedupe(keys + list(res.documents))
        if not res.sources:
            break
        max_score = max(s.rerank_score for s in res.sources)
        if max_score >= RERANK_SCORE_THRESHOLD or attempts >= MAX_RETRIEVAL_ATTEMPTS:
            break
        try:
            new_q = safe_llm_invoke(REWRITE_PROMPT.format_messages(
                question=question, previous_query=query
            )).content.strip()
        except Exception:
            break
        if not new_q or new_q == query:
            break
        query = new_q
        rewrites.append(new_q)

    return keys[:top_k], {"rewrites": rewrites, "attempts": attempts}


# ---------------- 主流程 ----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索质量离线评估")
    parser.add_argument("--top-k", type=int, default=10, help="单策略最大检索条数（默认 10）")
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 个问题（0=全部）")
    parser.add_argument("--partition", type=str, default=None, help="限定分区搜索（如 txt/pdf）")
    parser.add_argument("--collection", type=str, default=None,
                        help="指定 Milvus 集合（消融实验用，如 kb_adaptive）")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES,
                        help="要对比的策略，默认全部五条")
    parser.add_argument("--golden", type=str, default=None,
                        help="标注集路径（默认 golden_wixqa.json）")
    args = parser.parse_args()
    ks = [k for k in TOP_KS if k <= args.top_k]
    if not ks:
        ks = [args.top_k]
    strategies = [s for s in args.strategies if s in STRATEGIES]

    connect_milvus()

    print("[2/4] 加载标注集并计算 gold 集合 ...")
    golden = load_golden(Path(args.golden) if args.golden else None)
    source_level = any("gold_sources" in it for it in golden)

    if source_level:
        # 文章级评测（WixQA）：gold 为 source，无需加载全部父块
        parents = []
        questions, gold_sets = build_gold_mapping(parents, golden)
        from pymilvus import Collection
        corpus_size = Collection(name=args.collection or COLLECTION_NAME).num_entities
        print(f"[1/4] 文章级评测，语料块数：{corpus_size}")
    else:
        print("[1/4] 拉取语料父块 ...")
        parents = fetch_all_parents(args.collection)
        corpus_size = len(parents)
        questions, gold_sets = build_gold_mapping(parents, golden)
        print(f"      父块总数（去重后）: {corpus_size}")

    if args.limit:
        questions, gold_sets = questions[: args.limit], gold_sets[: args.limit]
    print(f"      有效评估问题数: {len(questions)}")
    if not questions:
        print("没有可评估的问题，退出")
        return

    parent_to_idx = {} if source_level else {p: i for i, (p, _) in enumerate(parents)}
    search_k = max(args.top_k * 5, 50) if source_level else args.top_k

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
        dense = search_dense(qvec, top_k=search_k, partition_name=args.partition,
                             collection_name=args.collection)
        sparse = search_sparse(qsparse, top_k=search_k, partition_name=args.partition,
                               collection_name=args.collection)

        row = {"question": question, "level": level,
               "doc_type": item.get("doc_type", ""), "gold_count": len(gold)}
        for strategy in strategies:
            ranked_keys, extra = retrieve_strategy(
                strategy, question, dense, sparse, args.top_k, gold, parent_to_idx,
                collection_name=args.collection, source_level=source_level,
            )
            if source_level:
                ranked_idx = _dedupe(ranked_keys)
            else:
                ranked_idx = [parent_to_idx[p] for p in _dedupe(ranked_keys) if p in parent_to_idx]
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

    # 按 doc_type 分层的 Recall@k_std 统计
    doc_type_stats = {}
    for row in per_question:
        dt = row.get("doc_type", "未知")
        if dt not in doc_type_stats:
            doc_type_stats[dt] = {"count": 0, "strategies": {}}
        doc_type_stats[dt]["count"] += 1
        for strategy in strategies:
            doc_type_stats[dt]["strategies"].setdefault(strategy, []).append(
                row.get(f"{strategy}_recall@{k_std}", 0.0))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORT_DIR / "retrieval_eval.md"
    json_path = REPORT_DIR / "retrieval_eval.json"
    json_path.write_text(
        json.dumps({"summary": summary, "ci": ci, "level_stats": level_stats,
                    "doc_type_stats": doc_type_stats,
                    "per_question": per_question,
                    "corpus_size": corpus_size, "question_count": len(questions),
                    "level": "source" if source_level else "parent",
                    "total_rewrites": rewrite_count},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# RAG 检索质量评估报告",
        "",
        f"- 语料规模：{corpus_size}",
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

    lines += ["", "## 分阶段提升（多指标）", ""]
    steps = [("纯稠密基线", "dense_only"), ("+混合检索(RRF)", "hybrid_rrf"),
             ("+重排序", "hybrid_rerank"), ("+多查询召回", "hybrid_multiquery"),
             ("+Agentic重检", "agentic")]
    metric_cols = [m for m in ["recall@1", "recall@5", "recall@10", "mrr@5", "ndcg@5", "hit_rate@10"]
                   if m in summary[steps[0][1]]]
    lines.append("| 阶段 | " + " | ".join(metric_cols) + " |")
    lines.append("|------|" + "|".join(["------"] * len(metric_cols)) + "|")
    for name, key in steps:
        if key not in summary:
            continue
        cells = [str(summary[key].get(m, "-")) for m in metric_cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
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

    lines.append("## 按文档类型分层（Recall@" + str(k_std) + "）")
    lines.append("")
    lines.append("| 文档类型 | 题数 | " + " | ".join(s for s in strategies) + " |")
    lines.append("|------|------|" + "|".join(["------"] * len(strategies)) + "|")
    for dt, st in doc_type_stats.items():
        cells = [dt, str(st["count"])]
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

    print("\n===== 检索链路分阶段对比（均值）=====")
    print(f"{'策略':<16}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR@5':>9}{'nDCG@5':>9}{'Hit@10':>8}")
    for name, key in [("纯稠密基线", "dense_only"), ("+混合检索(RRF)", "hybrid_rrf"),
                      ("+重排序", "hybrid_rerank"), ("+多查询召回", "hybrid_multiquery"),
                      ("+Agentic重检", "agentic")]:
        if key not in summary:
            continue
        s = summary[key]
        print(f"{name:<16}{s.get('recall@1', 0):>8}{s.get('recall@5', 0):>8}"
              f"{s.get('recall@10', 0):>8}{s.get('mrr@5', 0):>9}{s.get('ndcg@5', 0):>9}"
              f"{s.get('hit_rate@10', 0):>8}")
    print(f"\n报告已生成：{md_path}")


if __name__ == "__main__":
    main()
