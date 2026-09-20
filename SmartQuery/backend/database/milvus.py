from SmartQuery.backend.config import MILVUS_HOST, MILVUS_PORT
from SmartQuery.backend.logger import get_logger
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

logger = get_logger(__name__)

COLLECTION_NAME = "enterprise_kb_docs"
DIMENSION = 2048
PARTITIONS = ["pdf", "docx", "txt", "md", "xlsx"]


def connect_milvus():
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
    )
    logger.info("milvus connected %s:%s", MILVUS_HOST, MILVUS_PORT)


def create_collection(collection_name: str | None = None):
    collection_name = collection_name or COLLECTION_NAME
    if utility.has_collection(collection_name):
        collection = Collection(name=collection_name)
        existing = {p.name for p in collection.partitions}
        for partition in PARTITIONS:
            if partition not in existing:
                collection.create_partition(partition)
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="parent_text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="department", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
    ]

    schema = CollectionSchema(fields, description="Enterprise KB document embeddings")
    collection = Collection(name=collection_name, schema=schema)

    for partition in PARTITIONS:
        collection.create_partition(partition)

    index_params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
    collection.create_index(field_name="dense_vector", index_params=index_params)
    collection.create_index(field_name="sparse_vector",
                            index_params={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
    # 标量字段倒排索引：加速按文档类型 / 部门过滤（否则为暴力扫描）
    collection.create_index(field_name="doc_type", index_params={"index_type": "INVERTED"})
    collection.create_index(field_name="department", index_params={"index_type": "INVERTED"})
    collection.load()
    logger.info("created collection %s", collection_name)


def drop_collection(collection_name: str | None = None):
    """删除集合（用于 schema 变更或消融实验重建）"""
    collection_name = collection_name or COLLECTION_NAME
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        logger.info("dropped collection %s", collection_name)


def delete_by_source(source: str, collection_name: str | None = None) -> int:
    """按来源文件名删除该文档的所有块（用于增量更新 / 删除文档）"""
    collection = Collection(name=collection_name or COLLECTION_NAME)
    collection.load()
    expr = f'source == "{source}"'
    try:
        rows = collection.query(expr=expr, output_fields=["id"], limit=16384)
        count = len(rows)
    except Exception as e:
        logger.warning("delete_by_source query failed: %s", e)
        count = 0
    if count:
        collection.delete(expr=expr)
        collection.flush()
    logger.info("delete_by_source source=%s deleted=%d", source, count)
    return count


def delete_by_sources(sources: list[str], collection_name: str | None = None) -> int:
    """按来源文件名批量删除（一次 delete 完成，避免逐文件 flush）。

    用于批量入库前按 source 幂等清理：先删同源旧块，再重新插入，
    从而保证同一份语料重复入库时不会产生重复块。
    """
    sources = [s for s in sources if s]
    if not sources:
        return 0
    collection = Collection(name=collection_name or COLLECTION_NAME)
    collection.load()
    escaped = ", ".join('"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"' for s in sources)
    expr = f"source in [{escaped}]"
    try:
        rows = collection.query(expr=expr, output_fields=["id"], limit=16384)
        count = len(rows)
    except Exception as e:
        logger.warning("delete_by_sources query failed: %s", e)
        count = 0
    if count:
        collection.delete(expr=expr)
        collection.flush()
    logger.info("delete_by_sources sources=%d deleted=%d", len(sources), count)
    return count


def insert_documents(
    texts: list[str],
    parent_texts: list[str],
    dense_vectors: list[list[float]],
    sparse_vectors: list[dict[int, float]],
    partition_name: str = "txt",
    doc_type: str = "员工手册",
    department: str = "公共",
    source: str = "",
    collection_name: str | None = None,
    flush: bool = True,
):
    collection = Collection(name=collection_name or COLLECTION_NAME)
    entities = [texts, parent_texts, dense_vectors, sparse_vectors,
                [doc_type] * len(texts), [department] * len(texts), [source] * len(texts)]
    collection.insert(entities, partition_name=partition_name)
    # 批量入库时逐文件 flush 极慢，改为全部插入后统一 flush 一次
    if flush:
        collection.flush()


def search_dense(
    query_vector: list[float],
    top_k: int = 5,
    partition_name: str | None = None,
    expr: str | None = None,
    collection_name: str | None = None,
) -> list[tuple[str, str, float, str, str]]:
    collection = Collection(name=collection_name or COLLECTION_NAME)
    collection.load()

    kwargs = {
        "data": [query_vector],
        "anns_field": "dense_vector",
        "param": {"metric_type": "IP", "params": {"nprobe": 10}},
        "limit": top_k,
        "output_fields": ["text", "parent_text", "doc_type", "source"],
    }
    if partition_name:
        kwargs["partition_names"] = [partition_name]
    if expr:
        kwargs["expr"] = expr

    results = collection.search(**kwargs)
    return [(hit.entity.get("text"), hit.entity.get("parent_text"), hit.score,
             hit.entity.get("doc_type"), hit.entity.get("source")) for hit in results[0]]


def search_sparse(
    query_vector: dict[int, float],
    top_k: int = 5,
    partition_name: str | None = None,
    expr: str | None = None,
    collection_name: str | None = None,
) -> list[tuple[str, str, float, str, str]]:
    collection = Collection(name=collection_name or COLLECTION_NAME)
    collection.load()

    kwargs = {
        "data": [query_vector],
        "anns_field": "sparse_vector",
        "param": {"metric_type": "IP"},
        "limit": top_k,
        "output_fields": ["text", "parent_text", "doc_type", "source"],
    }
    if partition_name:
        kwargs["partition_names"] = [partition_name]
    if expr:
        kwargs["expr"] = expr

    results = collection.search(**kwargs)
    return [(hit.entity.get("text"), hit.entity.get("parent_text"), hit.score,
             hit.entity.get("doc_type"), hit.entity.get("source")) for hit in results[0]]
