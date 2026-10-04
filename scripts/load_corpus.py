"""
批量语料入库脚本（批量嵌入版）
==============================
遍历 <corpus-dir>/<文档类型>/ 下的所有文件，按文件夹名作为 doc_type 入库；
若目录下存在 _meta.json（WixQA 等），则从中读取每篇的 title / url / article_type。

关键优化：先把所有文件解析、切分，收集全部子块，再**一次性批量嵌入**，
最后分文件写入 Milvus。这样摊薄了稀疏模型（BGE-M3）每次调用的固定开销。

用法：
  python scripts/load_corpus.py                                  # 默认 data/corpus
  python scripts/load_corpus.py --corpus-dir data/wixqa_corpus    # 指定语料目录
  python scripts/load_corpus.py --corpus-dir data/wixqa_corpus --collection enterprise_kb_docs
  python scripts/load_corpus.py --doc-type FAQ                    # 只入库某一类型
  python scripts/load_corpus.py --reset                           # 先删集合再入库
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# Windows 控制台默认 GBK，打印含特殊字符的文件名会崩溃；统一切到 UTF-8
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
    get_document_by_source,
)
from SmartQuery.rag.ingest import (
    clean_documents,
    get_partition,
    load_file,
    split_documents,
)
from SmartQuery.rag.embedding import embed_documents, embed_documents_sparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = ROOT / "data" / "corpus"
SUPPORTED_EXTS = {".md", ".txt", ".pdf", ".docx", ".xlsx", ".png", ".jpg", ".jpeg"}


def file_sha256(path: str) -> str:
    """文件内容 SHA-256，与上传接口的去重口径一致"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_meta(corpus_dir: Path) -> dict[str, dict]:
    """读取 _meta.json（可选）：文件名 -> {title,url,article_type}"""
    meta_file = corpus_dir / "_meta.json"
    if not meta_file.exists():
        return {}
    try:
        return json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  读取 _meta.json 失败：{e}")
        return {}


def iter_corpus(corpus_dir: Path, doc_type: str | None = None):
    """遍历语料目录，产出 (文件路径, 文档类型)"""
    if not corpus_dir.exists():
        raise SystemExit(f"语料目录不存在：{corpus_dir}")
    for type_dir in sorted(corpus_dir.iterdir()):
        if not type_dir.is_dir():
            continue
        if doc_type and type_dir.name != doc_type:
            continue
        for file in sorted(type_dir.iterdir()):
            if file.suffix.lower() in SUPPORTED_EXTS:
                yield file, type_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(description="企业语料批量入库")
    parser.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR), help="语料根目录")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Milvus 集合名")
    parser.add_argument("--doc-type", default=None, help="只入库指定文档类型")
    parser.add_argument("--strategy", default="adaptive", help="分块策略")
    parser.add_argument("--reset", action="store_true", help="入库前删除集合")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.is_absolute():
        corpus_dir = (ROOT / corpus_dir).resolve()
    collection = args.collection

    connect_milvus()
    init_db()  # 确保 documents / uploaded_files 登记表存在（供前端 /upload 去重）
    if args.reset:
        drop_collection(collection)
    create_collection(collection)

    meta = load_meta(corpus_dir)
    files = list(iter_corpus(corpus_dir, args.doc_type))
    print(f"待入库文件数：{len(files)}（集合={collection}）")

    # 1) 解析 + 切分；断点续跑：已入库且内容未变的文件直接跳过，避免重复嵌入
    jobs = []
    stale_sources = []
    skipped = 0
    for file, folder_type in files:
        info = meta.get(file.name, {})
        doc_type = info.get("article_type") or folder_type
        title = info.get("title") or ""
        url = info.get("url") or ""
        fhash = file_sha256(str(file))
        rec = get_document_by_source(file.name)
        if (rec and rec.get("status") == "active" and rec.get("file_hash") == fhash
                and rec.get("chunk_count", 0) > 0):
            skipped += 1
            continue
        stale_sources.append(file.name)
        try:
            docs = clean_documents(load_file(str(file), doc_type=doc_type))
            child, parent = split_documents(docs, doc_type=doc_type, strategy=args.strategy)
            if not child:
                print(f"  跳过（无块）：{file.name}")
                continue
            jobs.append((file, doc_type, title, url, fhash, child, parent, get_partition(str(file))))
        except Exception as e:
            print(f"  切分失败 {file.name}：{e}")

    if skipped:
        print(f"断点续跑：跳过已入库 {skipped} 篇")

    # 幂等清理：仅删除将要重插的旧块（不误删跳过的）
    removed = delete_by_sources(stale_sources, collection_name=collection)
    if removed:
        print(f"幂等清理：删除同源旧块 {removed} 个")

    total_chunks = sum(len(j[5]) for j in jobs)
    print(f"待嵌入块数：{total_chunks}")

    # 2) 一次性批量嵌入（关键：摊薄固定开销）
    t_embed = time.time()
    all_children = [c for j in jobs for c in j[5]]
    dense_vectors = embed_documents(all_children)
    sparse_vectors = embed_documents_sparse(all_children)
    print(f"批量嵌入完成：{time.time() - t_embed:.1f}s")

    # 3) 分文件写入 Milvus（先不 flush，最后统一 flush 一次）
    t_insert = time.time()
    offset = 0
    for file, doc_type, title, url, fhash, child, parent, part in jobs:
        n = len(child)
        insert_documents(
            child, parent,
            dense_vectors[offset:offset + n],
            sparse_vectors[offset:offset + n],
            part, doc_type=doc_type, source=file.name,
            titles=[title] * n, urls=[url] * n,
            collection_name=collection, flush=False,
        )
        upsert_document(file.name, str(file), fhash, doc_type, n, title=title, url=url)
        save_file_record(fhash, file.name, part, n)
        offset += n
        print(f"  {doc_type}/{file.name} -> {n} 块")

    from pymilvus import Collection
    Collection(name=collection).flush()
    print(f"  flush 完成：{time.time() - t_insert:.1f}s")

    print(f"\n入库完成：集合={collection} 本次文件={len(jobs)} 跳过={skipped} "
          f"块={total_chunks} 写入耗时={time.time() - t_insert:.1f}s")


if __name__ == "__main__":
    main()
