"""
实验：重排"输入表示"对文章级 Hit@1 的影响
=========================================
对比送进交叉编码器的三种文本：
  parent       : parent_text（当前做法，可能被 max_length 截断）
  child        : 命中的子块文本（短、聚焦、不截断）
  title_child  : 文章标题 + 子块（聚焦且带主题）

用法：python scripts/exp_rerank_repr.py --limit 25
"""

import argparse
import json
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
MODES = ["parent", "child", "title_child"]
KS = [1, 3, 5, 10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--pool", type=int, default=100, help="每通道候选数")
    ap.add_argument("--rerank", type=int, default=30, help="送重排的文章数")
    args = ap.parse_args()

    connect_milvus()
    gold = json.loads(GOLDEN.read_text(encoding="utf-8"))[: args.limit]
    agg = {m: {k: [] for k in KS} for m in MODES}

    for i, g in enumerate(gold, 1):
        q = g["question"]
        gs = set(g["gold_sources"])
        dense = search_dense(embed_query(q), top_k=args.pool)
        sparse = search_sparse(embed_query_sparse(q), top_k=args.pool)
        meta = {}
        for it in list(dense) + list(sparse):
            meta.setdefault(it[0], it[3:7])  # doc_type, source, title, url

        scored = _rrf_scored(dense, sparse)
        seen, cand = set(), []
        for text, parent, _ in scored:
            dt, src, title, url = meta.get(text, ("", "", "", ""))
            key = src or parent or text
            if key in seen:
                continue
            seen.add(key)
            cand.append((text, parent, title, src))
            if len(cand) >= args.rerank:
                break

        for mode in MODES:
            pairs = []
            for text, parent, title, _ in cand:
                if mode == "parent":
                    rt = parent or text
                elif mode == "child":
                    rt = text
                else:
                    rt = f"{title}\n{text}" if title else text
                pairs.append([q, rt])
            scores = reranker.predict(pairs, batch_size=16) if pairs else []
            ranked = sorted(zip(cand, scores), key=lambda x: x[1], reverse=True)
            srcs = []
            for (text, parent, title, src), _ in ranked:
                key = src or parent or text
                if key not in srcs:
                    srcs.append(key)
            for k in KS:
                agg[mode][k].append(1.0 if any(s in gs for s in srcs[:k]) else 0.0)
        if i % 5 == 0:
            print(f"  {i}/{len(gold)} ...", flush=True)

    print("\n重排输入表示对比（文章级 HitRate）")
    print(f"{'mode':<14}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'Hit@10':>8}")
    for mode in MODES:
        vals = [sum(agg[mode][k]) / len(agg[mode][k]) for k in KS]
        print(f"{mode:<14}" + "".join(f"{v:>8.3f}" for v in vals))


if __name__ == "__main__":
    main()
