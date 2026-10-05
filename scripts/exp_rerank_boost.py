"""
实验：提升文章级 Hit@1 的两种手段
==================================
对比 4 组配置（同一候选集，重排只跑一次）：
  baseline        : 每篇只取最高 RRF 的 1 段重排（当前做法）
  maxp            : 每篇取 top-2 段重排、取最高分作为文章分
  title           : baseline + 标题词法命中加权
  maxp_title      : maxp + 标题词法命中加权

用法：python scripts/exp_rerank_boost.py --limit 25
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from SmartQuery.backend.database.milvus import connect_milvus, search_dense, search_sparse
from SmartQuery.rag.embedding import embed_query, embed_query_sparse
from SmartQuery.rag.retriever import _rrf_scored, reranker

GOLDEN = ROOT / "SmartQuery" / "evaluation" / "golden_wixqa.json"
CONFIGS = ["baseline", "maxp", "title", "maxp_title"]
KS = [1, 3, 5, 10]
PASSAGES_PER_ARTICLE = 2
TITLE_BOOST_WEIGHT = 0.3

STOP = {"a", "an", "the", "is", "are", "do", "does", "how", "to", "of", "in", "on",
        "for", "my", "your", "you", "i", "can", "with", "and", "or", "it", "its",
        "what", "when", "where", "why", "which", "about", "from", "into", "not"}


def toks(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 1}


def title_overlap(query: str, title: str) -> float:
    qa = toks(query)
    if not qa:
        return 0.0
    return len(qa & toks(title)) / len(qa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--pool", type=int, default=100, help="每通道候选数")
    ap.add_argument("--articles", type=int, default=30, help="候选文章数")
    args = ap.parse_args()

    connect_milvus()
    gold = json.loads(GOLDEN.read_text(encoding="utf-8"))[: args.limit]
    agg = {c: {k: [] for k in KS} for c in CONFIGS}

    for i, g in enumerate(gold, 1):
        q = g["question"]
        gs = set(g["gold_sources"])
        dense = search_dense(embed_query(q), top_k=args.pool)
        sparse = search_sparse(embed_query_sparse(q), top_k=args.pool)
        meta = {}
        for it in list(dense) + list(sparse):
            meta.setdefault(it[0], it[3:7])  # doc_type, source, title, url

        # 按文章聚合子块（RRF 顺序），每篇取前 PASSAGES_PER_ARTICLE 段
        scored = _rrf_scored(dense, sparse)
        articles: list[dict] = []
        index: dict[str, dict] = {}
        for text, parent, _ in scored:
            dt, src, title, url = meta.get(text, ("", "", "", ""))
            key = src or parent or text
            art = index.get(key)
            if art is None:
                if len(articles) >= args.articles:
                    continue
                art = {"key": key, "title": title, "passages": [], "src": src, "parent": parent}
                index[key] = art
                articles.append(art)
            if len(art["passages"]) < PASSAGES_PER_ARTICLE:
                art["passages"].append(parent or text)

        # 重排所有候选段（一次）
        flat = [(art, p) for art in articles for p in art["passages"]]
        pairs = [[q, p] for _, p in flat]
        scores = reranker.predict(pairs, batch_size=16) if pairs else []
        pass_score: dict[int, list[float]] = {}
        for (art, _), sc in zip(flat, scores):
            pass_score.setdefault(id(art), []).append(float(sc))

        for cfg in CONFIGS:
            ranked = []
            for art in articles:
                ps = pass_score.get(id(art), [0.0])
                base = ps[0]
                mx = max(ps)
                if cfg == "baseline":
                    final = base
                elif cfg == "maxp":
                    final = mx
                elif cfg == "title":
                    final = base + TITLE_BOOST_WEIGHT * title_overlap(q, art["title"])
                else:
                    final = mx + TITLE_BOOST_WEIGHT * title_overlap(q, art["title"])
                ranked.append((final, art["key"]))
            ranked.sort(reverse=True)
            order = [k for _, k in ranked]
            for k in KS:
                agg[cfg][k].append(1.0 if any(s in gs for s in order[:k]) else 0.0)
        if i % 5 == 0:
            print(f"  {i}/{len(gold)} ...", flush=True)

    print("\nHit@1 提升对比（文章级 HitRate）")
    print(f"{'config':<14}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'Hit@10':>8}")
    for cfg in CONFIGS:
        vals = [sum(agg[cfg][k]) / len(agg[cfg][k]) for k in KS]
        print(f"{cfg:<14}" + "".join(f"{v:>8.3f}" for v in vals))


if __name__ == "__main__":
    main()
