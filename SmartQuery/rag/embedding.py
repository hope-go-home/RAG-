from SmartQuery.backend.config import QWEN_EMBEDDING_MODEL, QWEN_API_KEY, QWEN_BASE_URL  #注意导config的方式
# from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from FlagEmbedding import BGEM3FlagModel


# 兼容模式
embeddings = OpenAIEmbeddings(
    model=QWEN_EMBEDDING_MODEL,
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    check_embedding_ctx_length=False,  # 禁用 LangChain 文本预处理，否则 DashScope 兼容模式报 input.contents 错误
    chunk_size=10,  # text-embedding-v4 单次最多 10 条，超过会报 batch size invalid
)

# # 现在（原生 API）
# from langchain_community.embeddings import DashScopeEmbeddings
# embeddings = DashScopeEmbeddings(
#     model=QWEN_EMBEDDING_MODEL,
#     dashscope_api_key=QWEN_API_KEY,
# )


"""
这两套走的是不同的 API 通道：

             OpenAIEmbeddings	                  DashScopeEmbeddings
端点	.../compatible-mode/v1/embeddings	      DashScope 原生 API
用 .env 的 QWEN_BASE_URL  用了	                 没用到（原生 SDK 内置 URL）
text-embedding-v4  支持	 	                    取决于 langchain_community 版本
DashScopeEmbeddings 底层调的是 dashscope SDK 的原生 API。大部分情况能工作，
但如果版本旧可能不认识 text-embedding-v4。如果运行时报模型名错误，换回 OpenAIEmbeddings 加 dimensions=1024 就行：
"""


def embed_documents(texts: list[str]) -> list[list[float]]:
    return embeddings.embed_documents(texts)
#定义一个函数，接收一个字符串列表 texts，返回类型为 list[list[float]]，即每个输入文本对应一个浮点数向量（列表形式） 批量编码多个文档（如知识库段落）
"""
embeddings 是通过 OpenAIEmbeddings(...) 创建的一个对象实例，而 OpenAIEmbeddings 这个类正是从 langchain_openai 导入的（来自 LangChain 库）。
因此，embeddings.embed_documents(texts) 调用的就是 LangChain 框架中 OpenAIEmbeddings 类提供的 embed_documents 方法
这样方便，别的文件可以方便点用
"""
 
def embed_query(text: str) -> list[float]:    #   （单个文本）向量化  编码单个查询（如用户问题）
    return embeddings.embed_query(text)


sparse_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
tokenizer = sparse_model.tokenizer  # 用来把 token 字符串转成 int ID（Milvus 需要）

def embed_query_sparse(text: str) -> dict[int, float]:
    output = sparse_model.encode([text], return_sparse=True, return_dense=False)
    raw = output['lexical_weights'][0]
    return {int(t): float(w) for t, w in raw.items()}

def embed_documents_sparse(texts: list[str]) -> list[dict[int, float]]:
    output = sparse_model.encode(texts, return_sparse=True, return_dense=False)
    results = []
    for raw in output['lexical_weights']:
        results.append({int(t): float(w) for t, w in raw.items()})
    return results



"""
def embed_query_sparse(text: str) -> dict[int, float]:
    vec = sparse_model.encode(text, return_dense=False, return_sparse=True)
    return vec.legacy_dict

 调用 sparse_model.encode。return_dense=False：不计算稠密向量，节省计算资源。return_sparse=True：要求模型返回稀疏表示（对 BGE-M3 有效）。
返回对象 vec 的类型通常是 SparseEmbedding，包含稀疏索引和权重的数据结构。

访问 vec 的 legacy_dict 属性（不是方法调用），返回一个 dict[int, float]，例如 {101: 0.5, 202: 0.8, ...}。
这是 sentence-transformers 提供的兼容格式，可直接用于 search_sparse 等函数。

def embed_documents_sparse(texts: list[str]) -> list[dict[int, float]]:
    vecs = sparse_model.encode(texts, return_dense=False, return_sparse=True)
    return [v.legacy_dict for v in vecs]
embedding.py:48 — 就这一个文件，embed_query_sparse 函数里：
vec = sparse_model.encode(text, return_dense=False, return_sparse=True)
以及 embedding.py:60 的 embed_documents_sparse 也有同样问题：
vecs = sparse_model.encode(texts, return_dense=False, return_sparse=True)
两处都用到了旧版的 return_sparse=True 参数。
新版 sentence-transformers 把 return_sparse 删了。稀疏向量得改用 FlagEmbedding 库。
"""


# 函数	稠密/稀疏	                       用途
# embed_documents	        稠密	     入库批量
# embed_query	            稠密	     问题单条
# embed_documents_sparse	稀疏	     入库批量
# embed_query_sparse	    稀疏	     问题单条
#稠密向量是text-embedding-v1模型     稀疏向量是bge-m3模型