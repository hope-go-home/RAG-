"""
分块策略消融实验脚本
====================
用同一份企业语料，分别以 3 种分块策略入库到 3 个 Milvus 集合：

  kb_fixed     固定长度分块（512 字符一刀切）        ← 基线
  kb_header    标题感知分块（仅按 Markdown 标题切）
  kb_adaptive  类型自适应分块（按文档类型选策略）     ← 本方案

然后可用 eval_retrieval.py --collection <集合名> 分别评测，对比 Recall@k。

用法：
  python scripts/build_ablation.py                 # 构建全部 3 个集合
  python scripts/build_ablation.py --strategies adaptive
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SmartQuery.backend.database.milvus import (
    connect_milvus,
    create_collection,
    drop_collection,
)
from SmartQuery.rag.ingest import ingest_file

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"
SUPPORTED_EXTS = {".md", ".txt", ".pdf", ".docx", ".xlsx"}

STRATEGIES = {
    "fixed": "kb_fixed",
    "header": "kb_header",
    "adaptive": "kb_adaptive",
}


def iter_corpus():
    for type_dir in sorted(CORPUS_DIR.iterdir()):
        if not type_dir.is_dir():
            continue
        for file in sorted(type_dir.iterdir()):
            if file.suffix.lower() in SUPPORTED_EXTS:
                yield file, type_dir.name


def build(strategy: str, collection: str) -> None:
    print(f"\n===== 策略 {strategy} -> 集合 {collection} =====")
    drop_collection(collection)
    create_collection(collection)

    files = list(iter_corpus())
    total = 0
    t0 = time.time()
    for i, (file, doc_type) in enumerate(files, start=1):
        try:
            count = ingest_file(str(file), doc_type=doc_type, strategy=strategy,
                                collection_name=collection)
            total += count
            print(f"[{i}/{len(files)}] {doc_type}/{file.name} -> {count} 块")
        except Exception as e:
            print(f"[{i}/{len(files)}] {doc_type}/{file.name} 失败：{e}")
    print(f"集合 {collection} 完成：{total} 块，{time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="分块策略消融实验")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES),
                        choices=list(STRATEGIES))
    args = parser.parse_args()

    connect_milvus()
    for strategy in args.strategies:
        build(strategy, STRATEGIES[strategy])

    print("\n全部完成。评测命令：")
    for strategy in args.strategies:
        print(f"  python SmartQuery/evaluation/eval_retrieval.py "
              f"--collection {STRATEGIES[strategy]} --strategies hybrid_rerank")


if __name__ == "__main__":
    main()
