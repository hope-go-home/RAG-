from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from SmartQuery.rag.embedding import embed_documents, embed_documents_sparse
from SmartQuery.backend.database.milvus import insert_documents
from SmartQuery.backend.logger import get_logger
import re

logger = get_logger(__name__)


# ==================== 文档清洗 ====================

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    allowed_chars = r'\u4e00-\u9fa5a-zA-Z0-9\s.,;:!?\u3002\uff0c\uff1b\uff1a\uff01\uff1f\u3001\uff08\uff09\u3010\u3011\u300a\u300b\u201c\u201d\u2018\u2019+/=%#@&*~^-'
    text = re.sub(f'[^{allowed_chars}]', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))
    text = re.sub(r'^第\s*\d+\s*页', '', text, flags=re.MULTILINE)
    text = re.sub(r'^共\s*\d+\s*页', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Page\s+\d+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\s*/\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r' +', ' ', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text.strip()


def clean_documents(docs: list[Document]) -> list[Document]:
    cleaned = []
    for doc in docs:
        cleaned_content = clean_text(doc.page_content)
        if len(cleaned_content) >= 50:
            cleaned.append(Document(page_content=cleaned_content, metadata=doc.metadata))
        else:
            logger.debug("clean_documents: skip short chunk (%d chars)", len(cleaned_content))
    logger.info("clean_documents: %d -> %d chunks", len(docs), len(cleaned))
    return cleaned


# ==================== 文件加载器 ====================

def load_file(file_path: str, doc_type: str = "员工手册") -> list[Document]:
    ext = file_path.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext == "docx":
        loader = Docx2txtLoader(file_path)
    elif ext in ("txt", "md"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == "xlsx":
        loader = _load_xlsx(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {ext}")
    docs = loader.load()
    for doc in docs:
        doc.metadata["doc_type"] = doc_type
        doc.metadata["file_format"] = ext
    return docs


def _load_xlsx(file_path: str):
    """兼容 xlsx 加载器，返回 loader 对象"""
    try:
        from langchain_community.document_loaders import UnstructuredExcelLoader
        return UnstructuredExcelLoader(file_path, mode="elements")
    except ImportError:
        from langchain_community.document_loaders import CSVLoader
        import pandas as pd
        df = pd.read_excel(file_path)
        csv_path = file_path.rsplit(".", 1)[0] + "_temp.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return CSVLoader(csv_path, encoding="utf-8-sig")


# ==================== 分块策略 ====================

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


def _split_by_articles(text: str) -> list[str]:
    """按「第X条」条款切分，保持条款完整性"""
    parts = re.split(r'^(第[一二三四五六七八九十百千\d]+条.+)$', text, flags=re.MULTILINE)
    result = []
    preamble = ""
    for part in parts:
        if re.match(r'^第[一二三四五六七八九十百千\d]+条', part):
            if preamble.strip():
                result.append(preamble.strip())
                preamble = ""
            result.append(part.strip())
        else:
            preamble += part
    if preamble.strip():
        result.append(preamble.strip())
    return result if result else [text]


def _split_faq(text: str) -> list[str]:
    """按 Q/A 问答对切分，问答对整体不切断"""
    pattern = r'^(Q[：:]\s*.+?)(?=Q[：:]\s*|$)'
    matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
    if matches:
        return [m.strip() for m in matches if m.strip()]
    return [text]


def _split_by_steps(text: str) -> list[str]:
    """按「步骤X」/「Step X」/「1. 2. 3.」编号切分"""
    pattern = r'^(?:步骤\s*\d+[.、：:]|Step\s*\d+[.、：:]|\d+[.、）)]\s*\S).*$'
    lines = text.split('\n')
    result = []
    current_block = []
    for line in lines:
        if re.match(pattern, line, re.IGNORECASE) and current_block:
            result.append('\n'.join(current_block).strip())
            current_block = [line]
        else:
            current_block.append(line)
    if current_block:
        result.append('\n'.join(current_block).strip())
    return result if result else [text]


def _split_by_sections(text: str) -> list[str]:
    """按二级/三级标题切分，代码块（```）整体保留不切"""
    code_block = False
    sections = []
    current_header = ""
    current_body = ""

    for line in text.split('\n'):
        if line.strip().startswith('```'):
            code_block = not code_block
            current_body += line + '\n'
            continue
        if not code_block and re.match(r'^#{2,4}\s+', line):
            if current_body.strip():
                sections.append(f"{current_header}\n{current_body.strip()}" if current_header else current_body.strip())
            current_header = line
            current_body = ""
        else:
            current_body += line + '\n'

    if current_body.strip():
        sections.append(f"{current_header}\n{current_body.strip()}" if current_header else current_body.strip())

    if not sections:
        return [text]

    final = []
    for s in sections:
        if len(s) > 1200:
            final.extend(parent_splitter.split_text(s))
        else:
            final.append(s)
    return final


def _split_by_table(text: str) -> list[str]:
    """表格整体保留不切断，非表格部分按段落切"""
    lines = text.split('\n')
    result = []
    current_block = []
    in_table = False

    for line in lines:
        is_table_line = bool(re.match(r'^\s*\|', line) or re.match(r'^\s*[\-\|]{3,}', line))
        if is_table_line and not in_table:
            if current_block:
                result.append('\n'.join(current_block).strip())
                current_block = []
            in_table = True
            current_block.append(line)
        elif is_table_line and in_table:
            current_block.append(line)
        elif not is_table_line and in_table:
            result.append('\n'.join(current_block).strip())
            current_block = [line]
            in_table = False
        else:
            current_block.append(line)

    if current_block:
        result.append('\n'.join(current_block).strip())

    final = []
    for block in result:
        if len(block) > 1000 and not re.match(r'^\s*\|', block):
            final.extend(parent_splitter.split_text(block))
        else:
            final.append(block)
    return final if final else [text]


def _whole_document(text: str) -> list[str]:
    """短文档整体作为一块"""
    return [text]


# ==================== 分块器路由 ====================

SPLITTER_MAP = {
    "规章制度": _split_by_articles,
    "FAQ": _split_faq,
    "操作流程SOP": _split_by_steps,
    "技术文档": _split_by_sections,
    "数据报表": _split_by_table,
    "员工手册": _split_by_headers,
}


def split_documents(docs: list[Document], doc_type: str = "员工手册",
                    strategy: str = "adaptive") -> tuple[list[str], list[str]]:
    """按 doc_type 选择分块策略，生成子块和父块"""
    if strategy == "fixed":
        return _split_fixed(docs)
    if strategy == "header":
        return _split_header_only(docs)

    splitter_fn = SPLITTER_MAP.get(doc_type, _split_by_headers)
    child_texts = []
    parent_texts = []

    for doc in docs:
        text = doc.page_content
        sections = splitter_fn(text)
        for section in sections:
            parent_chunks = parent_splitter.split_text(section)
            for parent in parent_chunks:
                child_chunks = child_splitter.split_text(parent)
                for child in child_chunks:
                    child_texts.append(child)
                    parent_texts.append(parent)

    logger.info("split_documents type=%s strategy=%s children=%d", doc_type, strategy, len(child_texts))
    return child_texts, parent_texts


def _split_fixed(docs: list[Document]) -> tuple[list[str], list[str]]:
    """基线：固定长度分块（无任何结构感知）"""
    fixed_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    child_texts = []
    parent_texts = []
    for doc in docs:
        chunks = fixed_splitter.split_text(doc.page_content)
        for chunk in chunks:
            child_texts.append(chunk)
            parent_texts.append(chunk)
    return child_texts, parent_texts


def _split_header_only(docs: list[Document]) -> tuple[list[str], list[str]]:
    """标题感知分块（仅按标题切，不区分文档类型）"""
    child_texts = []
    parent_texts = []
    for doc in docs:
        sections = _split_by_headers(doc.page_content)
        for section in sections:
            parent_chunks = parent_splitter.split_text(section)
            for parent in parent_chunks:
                child_chunks = child_splitter.split_text(parent)
                for child in child_chunks:
                    child_texts.append(child)
                    parent_texts.append(parent)
    return child_texts, parent_texts


# ==================== 分区与入库 ====================

PARTITION_MAP = {
    "pdf": "pdf", "docx": "docx", "txt": "txt", "md": "md",
    "xlsx": "xlsx", "csv": "csv", "pptx": "pptx", "html": "html", "json": "json",
}


def get_partition(file_path: str) -> str:
    ext = file_path.rsplit(".", 1)[-1].lower()
    return PARTITION_MAP.get(ext, "txt")


def ingest_file(file_path: str, doc_type: str = "员工手册",
                strategy: str = "adaptive", collection_name: str | None = None) -> int:
    docs = load_file(file_path, doc_type=doc_type)
    docs = clean_documents(docs)
    child_texts, parent_texts = split_documents(docs, doc_type=doc_type, strategy=strategy)

    dense_vectors = embed_documents(child_texts)
    sparse_vectors = embed_documents_sparse(child_texts)

    partition = get_partition(file_path)
    insert_kwargs = {"collection_name": collection_name} if collection_name else {}
    insert_documents(child_texts, parent_texts, dense_vectors, sparse_vectors, partition,
                     doc_type=doc_type, **insert_kwargs)

    logger.info("ingest file=%s type=%s strategy=%s partition=%s chunks=%d",
                file_path, doc_type, strategy, partition, len(child_texts))
    return len(child_texts)
