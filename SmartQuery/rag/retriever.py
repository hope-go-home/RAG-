from SmartQuery.rag.embedding import embed_query,embed_query_sparse  #只用 embed_query 和 embed_query_sparse，因它只处理用户问题的向量化。
from SmartQuery.backend.database.milvus import search_dense,search_sparse
from sentence_transformers import CrossEncoder

def rrf_fusion(dense_results: list[tuple[str, str, float]], sparse_results: list[tuple[str, str, float]], k: int = 60) -> list[str]:
    doc_scores = {}
    for rank, (text, parent_text, _) in enumerate(dense_results):
        doc_scores[text] = {"parent_text": parent_text, "score": 1 / (k + rank + 1)}
    for rank, (text, parent_text, _) in enumerate(sparse_results):
        if text in doc_scores:
            doc_scores[text]["score"] += 1 / (k + rank + 1)
        else:
            doc_scores[text] = {"parent_text": parent_text, "score": 1 / (k + rank + 1)}
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]["score"], reverse=True)  #  d = {"a": 1, "b": 2}，print(d.items())  # dict_items([('a', 1), ('b', 2)])
#对 doc_scores 中的键值对进行排序。doc_scores.items() 返回 (text, value_dict) 元组的可迭代对象。
# 排序依据是每个 value_dict 中的 "score" 字段，reverse=True 表示按分数从高到低降序排列。结果存入 sorted_docs

    return [item[1]["parent_text"] or item[0] for item in sorted_docs]
#item[0]对应子块，item[1]["parent_text"]对应父块


#实例化交叉编码器，加载预训练模型 BAAI/bge-reranker-v2-m3，用于对查询与文档进行相关性重排序。
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

#定义 rerank 函数，接收查询字符串和文档文本列表，返回重排序后的文档列表。
def rerank(query: str, documents: list[str]) -> list[str]:
    pairs = [[query, doc] for doc in documents]    #构造查询与每个文档的配对列表，格式为 [query, doc]
    scores = reranker.predict(pairs)   #调用交叉编码器预测所有配对的相似度得分，返回得分列表（与 pairs 顺序一致
    scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)  # reverse=True从大到小
    #  将文档与对应得分打包，按得分降序排序，得到 [(doc, score), ...]。
    return [doc for doc, _ in scored]  #提取排序后的文档文本（不返回得分）

#定义主检索函数 retrieve  参数：question 查询文本，top_k 最终返回的文档数量（默认5），partition_name Milvus 分区名（可选）
def retrieve(question: str, top_k: int = 5, partition_name: str | None = None) -> list[str]:
    query_vector = embed_query(question)  #调用 embed_query 将问题转为密集向量。
    query_sparse = embed_query_sparse(question)
    dense_results = search_dense(query_vector, top_k=top_k * 2, partition_name=partition_name)
    sparse_results = search_sparse(query_sparse, top_k=top_k * 2, partition_name=partition_name)
    fused = rrf_fusion(dense_results, sparse_results)
    return rerank(question, fused)[:top_k]  #调用 rerank 对融合后的所有文档进行重排序，然后取前 top_k 个作为最终结果返回




"""
用户问题
   ↓
embed_query()      → 稠密向量（OpenAI）
embed_query_sparse() → 稀疏向量（bge-m3）
   ↓      ↓
search_dense()    search_sparse()
   ↓      ↓
   rrf_fusion()   ← 融合排序
       ↓
   rerank()       ← 交叉编码器重排序
       ↓
   返回 TopK 文档
"""