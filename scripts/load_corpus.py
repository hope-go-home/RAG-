"""
批量语料入库脚本
================
遍历 data/corpus/<文档类型>/ 下的所有文件，按文件夹名作为 doc_type 入库。

用法：
  python scripts/load_corpus.py                 # 全量入库
  python scripts/load_corpus.py --doc-type FAQ  # 只入库某一类型
  python scripts/load_corpus.py --reset         # 先删集合再入库
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SmartQuery.backend.database.milvus import (
    COLLECTION_NAME,
    connect_milvus,
    create_collection,
    drop_collection,
)
from SmartQuery.rag.ingest import ingest_file

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"
SUPPORTED_EXTS = {".md", ".txt", ".pdf", ".docx", ".xlsx"}


def iter_corpus(doc_type: str | None = None):
    """遍历语料目录，产出 (文件路径, 文档类型)"""
    if not CORPUS_DIR.exists():
        raise SystemExit(f"语料目录不存在：{CORPUS_DIR}")
    for type_dir in sorted(CORPUS_DIR.iterdir()):
        if not type_dir.is_dir():
            continue
        if doc_type and type_dir.name != doc_type:
            continue
        for file in sorted(type_dir.iterdir()):
            if file.suffix.lower() in SUPPORTED_EXTS:
                yield file, type_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(description="企业语料批量入库")
    parser.add_argument("--doc-type", default=None, help="只入库指定文档类型")
    parser.add_argument("--reset", action="store_true", help="入库前删除集合")
    args = parser.parse_args()

    connect_milvus()
    if args.reset:
        drop_collection()
    create_collection()

    files = list(iter_corpus(args.doc_type))
    print(f"待入库文件数：{len(files)}")

    total_chunks = 0
    failed = []
    t0 = time.time()
    for i, (file, doc_type) in enumerate(files, start=1):
        try:
            count = ingest_file(str(file), doc_type=doc_type)
            total_chunks += count
            print(f"[{i}/{len(files)}] {doc_type}/{file.name} -> {count} 块")
        except Exception as e:
            failed.append((file.name, str(e)))
            print(f"[{i}/{len(files)}] {doc_type}/{file.name} 失败：{e}")

    elapsed = time.time() - t0
    print(f"\n入库完成：集合={COLLECTION_NAME} 文件={len(files) - len(failed)} "
          f"块={total_chunks} 耗时={elapsed:.1f}s")
    if failed:
        print(f"失败 {len(failed)} 个：")
        for name, err in failed:
            print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()
