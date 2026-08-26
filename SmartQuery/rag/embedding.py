from SmartQuery.backend.config import QWEN_API_KEY, QWEN_BASE_URL
from langchain_openai import OpenAIEmbeddings
from FlagEmbedding import BGEM3FlagModel


# 兼容模式
embeddings = OpenAIEmbeddings(
    model="qwen3.7-text-embedding",
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    check_embedding_ctx_length=False,
    chunk_size=20,  # qwen3.7-text-embedding 单次最多 20 条
    dimensions=2048,  # 指定输出维度
)

"""
说明：本项目走 OpenAIEmbeddings 兼容通道（端点 .../compatible-mode/v1/embeddings，支持 text-embedding-v4）。
另一条备选是 DashScope 原生 API（DashScopeEmbeddings），但依赖 langchain_community 版本，
旧版本不认识 text-embedding-v4，故未采用。
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


# 函数	稠密/稀疏	                       用途
# embed_documents	        稠密	     入库批量
# embed_query	            稠密	     问题单条
# embed_documents_sparse	稀疏	     入库批量
# embed_query_sparse	    稀疏	     问题单条
#稠密向量是text-embedding-v1模型     稀疏向量是bge-m3模型