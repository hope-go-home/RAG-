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
