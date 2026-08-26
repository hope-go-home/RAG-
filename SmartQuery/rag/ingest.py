from langchain_community.document_loaders import PyMuPDFLoader,Docx2txtLoader,TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from SmartQuery.rag.embedding import embed_documents, embed_documents_sparse
from SmartQuery.backend.database.milvus import insert_documents
from SmartQuery.backend.logger import get_logger
import re

logger = get_logger(__name__)


# ==================== 文档清洗 ====================

def clean_text(text: str) -> str:
    """清洗单个文本块，去除杂质和噪声"""
    if not text:
        return ""

    # 1. 去除控制字符（\x00-\x1f，除了换行和制表符）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 2. 去除特殊字符（保留中文、英文、数字、常用标点）
    allowed_chars = r'\u4e00-\u9fa5a-zA-Z0-9\s.,;:!?\u3002\uff0c\uff1b\uff1a\uff01\uff1f\u3001\uff08\uff09\u3010\u3011\u300a\u300b\u201c\u201d\u2018\u2019+/=%#@&*~^-'
    text = re.sub(f'[^{allowed_chars}]', '', text)

    # 3. 去除多余空行（连续空行合并为一个）
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # 4. 去除行首行尾空格
    text = '\n'.join(line.strip() for line in text.split('\n'))

    # 5. 去除页眉页脚（如"第X页"、"共X页"、"Page X"）
    text = re.sub(r'^第\s*\d+\s*页', '', text, flags=re.MULTILINE)
    text = re.sub(r'^共\s*\d+\s*页', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Page\s+\d+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\s*/\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # 6. 去除多余空格（连续空格合并为一个）
    text = re.sub(r' +', ' ', text)

    # 7. 标准化换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    return text.strip()


def clean_documents(docs: list[Document]) -> list[Document]:
    """清洗文档列表，去除无效内容"""
    cleaned = []
    for doc in docs:
        cleaned_content = clean_text(doc.page_content)
        # 过滤过短内容（少于 50 字符的块可能是噪声）
        if len(cleaned_content) >= 50:
            cleaned.append(Document(page_content=cleaned_content, metadata=doc.metadata))
        else:
            logger.debug("clean_documents: skip short chunk (%d chars)", len(cleaned_content))
    logger.info("clean_documents: %d -> %d chunks", len(docs), len(cleaned))
    return cleaned



def load_file(file_path:str) -> list[Document]:
    ext = file_path.rsplit(".", 1)[-1].lower() #rsplit 是 Python 字符串方法，从右侧开始分割字符串。第一个参数 "." 表示以点号作为分隔符。第二个参数 1 表示最多分割1 次
    if ext == "pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext == "docx":
        loader = Docx2txtLoader(file_path)
    elif ext in ("txt", "md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件类型: {ext}")
    return loader.load()


parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)


def _split_by_headers(text: str) -> list[str]:
    """按 Markdown 标题切分，保留标题作为上下文"""
    sections = re.split(r'^(#{1,4}\s+.+)$', text, flags=re.MULTILINE)
    result = []
    current_header = ""
    current_body = ""
    for part in sections:
        if re.match(r'^#{1,4}\s+', part):
            if current_body.strip():
                result.append(f"{current_header}\n{current_body.strip()}")
            current_header = part
            current_body = ""
        else:
            current_body += part
    if current_body.strip():
        result.append(f"{current_header}\n{current_body.strip()}" if current_header else current_body.strip())
    return result if result else [text]


def _split_by_sentences(text: str) -> list[str]:
    """中文按句号、问号、感叹号切分"""
    sentences = re.split(r'(?<=[。！？])\s*', text)
    return [s.strip() for s in sentences if s.strip()]


def split_documents(docs: list[Document]) -> tuple[list[str], list[str]]:
    child_texts = []
    parent_texts = []
    for doc in docs:
        text = doc.page_content
        sections = _split_by_headers(text)
        for section in sections:
            parent_chunks = parent_splitter.split_text(section)
            for parent in parent_chunks:
                sentences = _split_by_sentences(parent)
                child_chunks = child_splitter.split_text("\n".join(sentences))
                for child in child_chunks:
                    child_texts.append(child)
                    parent_texts.append(parent)
    return child_texts, parent_texts


def get_partition(file_path: str) -> str:
    ext = file_path.rsplit(".", 1)[-1].lower()
    return ext if ext in ("pdf", "docx", "txt", "md") else "pdf"


def ingest_file(file_path: str) -> int:
    docs = load_file(file_path)
    # 文档清洗：去除杂质、噪声、无效内容
    docs = clean_documents(docs)
    child_texts, parent_texts = split_documents(docs)

    dense_vectors = embed_documents(child_texts)  # 生成子块稠密向量
    sparse_vectors = embed_documents_sparse(child_texts)  #  生成子块稀疏向量

    partition = get_partition(file_path)    # 分区
    insert_documents(child_texts, parent_texts, dense_vectors, sparse_vectors, partition)
    # 将子块文本、父块文本、稠密向量、稀疏向量以及分区信息写入 Milvus 数据库。

    logger.info("ingest file=%s partition=%s chunks=%d", file_path, partition, len(child_texts))
    return len(child_texts) #返回成功插入的子块数量，便于调用方了解本次摄入的数据规模
