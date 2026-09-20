"""
批量语料入库脚本（批量嵌入版）
==============================
遍历 data/corpus/<文档类型>/ 下的所有文件，按文件夹名作为 doc_type 入库。

关键优化：先把所有文件解析、切分，收集全部子块，再**一次性批量嵌入**，
最后分文件写入 Milvus。这样摊薄了稀疏模型（BGE-M3）每次调用的固定开销，
比逐文件嵌入快数倍。

用法：
  python scripts/load_corpus.py                 # 全量入库
  python scripts/load_corpus.py --doc-type FAQ  # 只入库某一类型
  python scripts/load_corpus.py --reset         # 先删集合再入库
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from SmartQuery.backend.database.milvus import (
    COLLECTION_NAME,
    connect_milvus,
    create_collection,
    drop_collection,
    delete_by_sources,
    insert_documents,
)
from SmartQuery.backend.database.mysql import (
    init_db,
    save_file_record,
    upsert_document,
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


def file_sha256(path: str) -> str:
    """文件内容 SHA-256，与上传接口的去重口径一致"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_corpus(doc_type: str | None = None):
    """遍历语料目录，产出 (文件路径, 文档类型, 所属部门)"""
    if not CORPUS_DIR.exists():
        raise SystemExit(f"语料目录不存在：{CORPUS_DIR}")
    for type_dir in sorted(CORPUS_DIR.iterdir()):
        if not type_dir.is_dir():
            continue
        if doc_type and type_dir.name != doc_type:
            continue
        for file in sorted(type_dir.iterdir()):
            if file.suffix.lower() in SUPPORTED_EXTS:
                yield file, type_dir.name, department_of(file.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="企业语料批量入库")
    parser.add_argument("--doc-type", default=None, help="只入库指定文档类型")
    parser.add_argument("--strategy", default="adaptive", help="分块策略")
    parser.add_argument("--reset", action="store_true", help="入库前删除集合")
    args = parser.parse_args()

    connect_milvus()
    init_db()  # 确保 documents / uploaded_files 登记表存在（供前端 /upload 去重）
    if args.reset:
        drop_collection()
    create_collection()

    files = list(iter_corpus(args.doc_type))
    print(f"待入库文件数：{len(files)}")

    # 幂等清理：先按 source 删除同名旧块，再重新插入。
    # 这样重复运行本脚本不会产生重复块（不加 --reset 也安全）。
    removed = delete_by_sources([f.name for f, _, _ in files])
    if removed:
        print(f"幂等清理：删除同源旧块 {removed} 个")

    # 1) 解析 + 切分，收集所有子块
    jobs = []
    for file, doc_type, dept in files:
        try:
            docs = clean_documents(load_file(str(file), doc_type=doc_type))
            child, parent = split_documents(docs, doc_type=doc_type, strategy=args.strategy)
            if not child:
                print(f"  跳过（无块）：{file.name}")
                continue
            jobs.append((file, doc_type, dept, child, parent, get_partition(str(file))))
        except Exception as e:
            print(f"  切分失败 {file.name}：{e}")

    total_chunks = sum(len(j[3]) for j in jobs)
    print(f"待嵌入块数：{total_chunks}")

    # 2) 一次性批量嵌入（关键：摊薄固定开销）
    t_embed = time.time()
    all_children = [c for j in jobs for c in j[3]]
    dense_vectors = embed_documents(all_children)
    sparse_vectors = embed_documents_sparse(all_children)
    print(f"批量嵌入完成：{time.time() - t_embed:.1f}s")

    # 3) 分文件写入 Milvus（先不 flush，最后统一 flush 一次）
    t_insert = time.time()
    offset = 0
    for file, doc_type, dept, child, parent, part in jobs:
        n = len(child)
        insert_documents(
            child, parent,
            dense_vectors[offset:offset + n],
            sparse_vectors[offset:offset + n],
            part, doc_type=doc_type, department=dept, source=file.name, flush=False,
        )
        # 同步登记到 MySQL，使前端 /upload 的重复检查能识别批量入库的语料
        file_hash = file_sha256(str(file))
        upsert_document(file.name, str(file), file_hash, doc_type, dept, n)
        save_file_record(file_hash, file.name, part, n)
        offset += n
        print(f"  {doc_type}/{dept}/{file.name} -> {n} 块")

    from pymilvus import Collection
    Collection(name=COLLECTION_NAME).flush()
    print(f"  flush 完成：{time.time() - t_insert:.1f}s")

    print(f"\n入库完成：集合={COLLECTION_NAME} 文件={len(jobs)} 块={total_chunks} "
          f"写入耗时={time.time() - t_insert:.1f}s")


if __name__ == "__main__":
    main()
