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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from SmartQuery.backend.database.milvus import (
    connect_milvus,
    create_collection,
    drop_collection,
    insert_documents,
)
from SmartQuery.rag.ingest import (
    clean_documents,
    get_partition,
    load_file,
    split_documents,
)
from SmartQuery.rag.embedding import embed_documents, embed_documents_sparse
from corpus_meta import department_of

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"
SUPPORTED_EXTS = {".md", ".txt", ".pdf", ".docx", ".xlsx", ".png", ".jpg", ".jpeg"}

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
                yield file, type_dir.name, department_of(file.name)


def build(strategy: str, collection: str) -> None:
    from pymilvus import Collection

    print(f"\n===== 策略 {strategy} -> 集合 {collection} =====")
    drop_collection(collection)
    create_collection(collection)

    # 1) 解析 + 切分
    jobs = []
    for file, doc_type, dept in iter_corpus():
        docs = clean_documents(load_file(str(file), doc_type=doc_type))
        child, parent = split_documents(docs, doc_type=doc_type, strategy=strategy)
        if not child:
            continue
        jobs.append((file, doc_type, dept, child, parent, get_partition(str(file))))

    total = sum(len(j[3]) for j in jobs)

    # 2) 一次性批量嵌入
    all_children = [c for j in jobs for c in j[3]]
    dense = embed_documents(all_children)
    sparse = embed_documents_sparse(all_children)

    # 3) 写入（最后统一 flush）
    offset = 0
    for file, doc_type, dept, child, parent, part in jobs:
        n = len(child)
        insert_documents(child, parent, dense[offset:offset + n], sparse[offset:offset + n],
                         part, doc_type=doc_type, department=dept, source=file.name,
                         collection_name=collection, flush=False)
        offset += n
    Collection(name=collection).flush()
    print(f"集合 {collection} 完成：{total} 块")


def main() -> None:
    parser = argparse.ArgumentParser(description="分块策略消融实验")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES),
                        choices=list(STRATEGIES))
    args = parser.parse_args()

    connect_milvus()
    t0 = time.time()
    for strategy in args.strategies:
        build(strategy, STRATEGIES[strategy])
    print(f"\n全部完成（{time.time() - t0:.1f}s）。评测命令：")
    for strategy in args.strategies:
        print(f"  python SmartQuery/evaluation/eval_retrieval.py "
              f"--collection {STRATEGIES[strategy]} --strategies dense_only hybrid_rerank")


if __name__ == "__main__":
    main()
