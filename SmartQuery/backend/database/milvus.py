from SmartQuery.backend.config import MILVUS_HOST,MILVUS_PORT
from SmartQuery.backend.logger import get_logger
from pymilvus import connections,Collection,CollectionSchema,FieldSchema,DataType,utility
#从 pymilvus 导入所需组件：connections：管理连接。Collection：管理集合的类。CollectionSchema：定义集合结构。
# FieldSchema：定义每个字段。DataType：字段类型枚举。utility：工具函数，如 has_collection。

logger = get_logger(__name__)


COLLECTION_NAME = "smart_query_docs"  #集合名
DIMENSION = 2048   #向量维度
PARTITIONS = ["pdf", "docx", "txt", "md"]  # 支持的文档分区

def connect_milvus():
    connections.connect(
        alias = "default",    #为这个连接指定一个别名（默认为 "default"）。后续通过 connections[别名] 或操作集合时，可以隐式使用该连接。显式命名有利于多连接场景。
        host = MILVUS_HOST,  #主机地址
        port = MILVUS_PORT  #端口号
    )
    logger.info("milvus connected %s:%s", MILVUS_HOST, MILVUS_PORT)

def create_collection():
    if utility.has_collection(COLLECTION_NAME):
        # 集合已存在：补建缺失的分区（如旧库没有 md 分区）
        collection = Collection(name=COLLECTION_NAME)
        existing = {p.name for p in collection.partitions}
        for partition in PARTITIONS:
            if partition not in existing:
                collection.create_partition(partition)
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="parent_text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),  #稠密向量
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),   #稀疏向量
    ]

    schema = CollectionSchema(fields, description="RAG document embeddings")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    #创建分区。如果分区已存在会抛出异常，但此处刚创建集合，不会有重复。

    for partition in PARTITIONS:
        collection.create_partition(partition)

    index_params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
    #度量类型：IP（内积） 引类型：IVF_FLAT（基于聚类的倒排索引），适合百万级数据。参数：nlist=128 聚类中心数
    collection.create_index(field_name="dense_vector", index_params=index_params)
    #Milvus 集合中的 dense_vector 字段创建索引，从而加速相似度检索
    collection.create_index(field_name="sparse_vector", index_params={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
    ## 为 sparse_vector 字段创建稀疏向量专用索引（倒排索引），度量类型仍为 IP
    collection.load()
    #将集合加载到内存，后续才能进行插入和搜索

#定义 insert_documents 函数，向集合中插入文档记录
def insert_documents(
    texts: list[str],
    parent_texts: list[str],
    dense_vectors: list[list[float]],
    sparse_vectors: list[dict[int, float]],
    partition_name: str = "pdf",  # 默认为pdf
):

    collection = Collection(name=COLLECTION_NAME)
    entities = [texts, parent_texts, dense_vectors, sparse_vectors]
    collection.insert(entities, partition_name=partition_name)
    # 调用 insert 方法将数据插入到指定分区
    collection.flush()
    # 刷新缓冲区，确保数据持久化到磁盘（通常插入后会自动 flush，此处显式调用更保险）


def search_dense(
    query_vector: list[float],
    top_k: int = 5,
    partition_name: str | None = None,
) -> list[tuple[str, str, float]]:
    collection = Collection(name=COLLECTION_NAME)  #获取集合对象
    collection.load()  #将集合加载到内存（如果已经在内存中，此操作几乎无开销）

    kwargs = {"data": [query_vector],                                     # 查询向量需包装在列表中
              "anns_field": "dense_vector",                               # 指定在稠密向量字段上搜索
              "param": {"metric_type": "IP", "params": {"nprobe": 10}},   # 搜索参数：度量 IP，nprobe=10 表示搜索 10 个聚类单元
              "limit": top_k,                                             # 返回 top_k 条结果
              "output_fields": ["text", "parent_text"]                    # 返回结果中附带这两个标量字段的值
    }
    # # 如果指定了分区名，则添加到参数中，限定搜索范围
    if partition_name:
        kwargs["partition_names"] = [partition_name]

    results = collection.search(**kwargs)  #执行搜索，返回 SearchResult 对象列表
    return [(hit.entity.get("text"), hit.entity.get("parent_text"), hit.score) for hit in results[0]]
#  解析结果：遍历第一个查询的结果（此处只有一个查询向量），提取 text、parent_text 和 score（内积值）

def search_sparse(
    query_vector: dict[int, float],
    top_k: int = 5,
    partition_name: str | None = None,
) -> list[tuple[str, str, float]]:
    collection = Collection(name=COLLECTION_NAME)
    collection.load()

    kwargs = {"data": [query_vector],
               "anns_field": "sparse_vector",
              "param": {"metric_type": "IP"},
              "limit": top_k,
              "output_fields": ["text", "parent_text"]
    }
    if partition_name:
        kwargs["partition_names"] = [partition_name]

    results = collection.search(**kwargs)
    return [(hit.entity.get("text"), hit.entity.get("parent_text"), hit.score) for hit in results[0]]
