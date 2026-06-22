from langchain_community.document_loaders import PyMuPDFLoader,Docx2txtLoader,TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from SmartQuery.rag.embedding import embed_documents, embed_documents_sparse
from SmartQuery.backend.database.milvus import insert_documents



def load_file(file_path:str) -> list[Document]:
    ext = file_path.rsplit(".", 1)[-1].lower() #rsplit 是 Python 字符串方法，从右侧开始分割字符串。第一个参数 "." 表示以点号作为分隔符。第二个参数 1 表示最多分割1 次
    if ext == "pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext == "docx":
        loader = Docx2txtLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件类型: {ext}")
    return loader.load()


parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)


def split_documents(docs: list[Document]) -> tuple[list[str], list[str]]:
    parent_chunks = parent_splitter.split_documents(docs)
    child_texts = []
    parent_texts = []
    for parent in parent_chunks:
        children = child_splitter.split_documents([parent])
        for child in children:
            child_texts.append(child.page_content)
            parent_texts.append(parent.page_content)
    return child_texts, parent_texts


def get_partition(file_path: str) -> str:
    ext = file_path.rsplit(".", 1)[-1].lower()
    return ext if ext in ("pdf", "docx", "txt") else "pdf"


def ingest_file(file_path: str) -> int:
    docs = load_file(file_path)
    child_texts, parent_texts = split_documents(docs)

    dense_vectors = embed_documents(child_texts)  # 生成子块稠密向量
    sparse_vectors = embed_documents_sparse(child_texts)  #  生成子块稀疏向量

    partition = get_partition(file_path)    # 分区
    insert_documents(child_texts, parent_texts, dense_vectors, sparse_vectors, partition)
    # 将子块文本、父块文本、稠密向量、稀疏向量以及分区信息写入 Milvus 数据库。

    return len(child_texts) #返回成功插入的子块数量，便于调用方了解本次摄入的数据规模
