from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from SmartQuery.rag.embedding import embed_documents, embed_documents_sparse
from SmartQuery.backend.database.milvus import insert_documents
from SmartQuery.backend.logger import get_logger
import os
import re

logger = get_logger(__name__)


# ==================== 文档清洗 ====================

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    allowed_chars = r'\u4e00-\u9fa5a-zA-Z0-9\s.,;:!?\u3002\uff0c\uff1b\uff1a\uff01\uff1f\u3001\uff08\uff09\u3010\u3011\u300a\u300b\u201c\u201d\u2018\u2019+/=%#@&*~^\|_-'
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
        docs = PyMuPDFLoader(file_path).load()
        text_len = sum(len(d.page_content.strip()) for d in docs)
        # 文本层过少 → 判定为扫描件，回退 OCR
        if text_len < 50:
            logger.info("PDF 文本层过少（%d 字），回退 OCR：%s", text_len, file_path)
            docs = _load_scanned_pdf(file_path).load()
    elif ext == "docx":
        docs = Docx2txtLoader(file_path).load()
    elif ext in ("txt", "md"):
        docs = TextLoader(file_path, encoding="utf-8").load()
    elif ext == "xlsx":
        docs = _load_xlsx(file_path).load()
    elif ext in ("png", "jpg", "jpeg", "bmp", "tiff"):
        docs = _load_image(file_path).load()
    else:
        raise ValueError(f"不支持的文件类型: {ext}")
    for doc in docs:
        doc.metadata["doc_type"] = doc_type
        doc.metadata["file_format"] = ext
    return docs


def is_scanned(file_path: str) -> bool:
    """判断是否需要 OCR：图片文件，或文本层过短的 PDF（扫描件）"""
    ext = file_path.rsplit(".", 1)[-1].lower()
    if ext in ("png", "jpg", "jpeg", "bmp", "tiff"):
        return True
    if ext == "pdf":
        try:
            docs = PyMuPDFLoader(file_path).load()
            total = sum(len(d.page_content.strip()) for d in docs)
            return total < 50
        except Exception:
            return False
    return False


def _load_xlsx(file_path: str):
    """用 openpyxl 读取，转成 Markdown 表格文本（表格整体保留，便于结构化检索）"""
    from openpyxl import load_workbook
    wb = load_workbook(file_path, read_only=True, data_only=True)
    lines: list[str] = []
    for ws in wb.worksheets:
        lines.append(f"# {ws.title}")
        for row in ws.iter_rows(values_only=True):
            cells = ["" if c is None else str(c) for c in row]
            if any(c.strip() for c in cells):
                lines.append("| " + " | ".join(cells) + " |")
    return _StaticLoader("\n".join(lines))


class _StaticLoader:
    """把纯文本包装成 LangChain loader 接口（xlsx / OCR 转换用）"""

    def __init__(self, text: str):
        self._text = text

    def load(self) -> list[Document]:
        return [Document(page_content=self._text)]


# ==================== OCR（扫描件 / 图片） ==================== #

_ocr_client = None


def _get_ocr_client():
    global _ocr_client
    if _ocr_client is None:
        from openai import OpenAI
        from SmartQuery.backend.config import QWEN_API_KEY, QWEN_BASE_URL
        _ocr_client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
    return _ocr_client


def _ocr_model() -> str:
    from SmartQuery.backend.config import QWEN_VL_OCR_MODEL
    return QWEN_VL_OCR_MODEL or "qwen-vl-max"


OCR_PROMPT = "请识别图片中的全部文字，按原始阅读顺序输出纯文本；不要翻译、不要解释、不要总结。"


def _ocr_image_b64(b64_png: str) -> str:
    """调用视觉模型识别单张图片（base64 PNG）中的文字"""
    resp = _get_ocr_client().chat.completions.create(
        model=_ocr_model(),
        messages=[{"role": "user", "content": [
            {"type": "text", "text": OCR_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_png}"}},
        ]}],
    )
    return (resp.choices[0].message.content or "").strip()


def _load_scanned_pdf(file_path: str, max_pages: int = 20):
    """扫描件 PDF：逐页渲染为图片后 OCR"""
    import base64
    import fitz  # PyMuPDF

    doc = fitz.open(file_path)
    texts = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            logger.warning("扫描件页数超过上限 %d，仅处理前 %d 页", max_pages, max_pages)
            break
        pix = page.get_pixmap(dpi=150)
        b64 = base64.b64encode(pix.tobytes("png")).decode()
        try:
            text = _ocr_image_b64(b64)
            if text:
                texts.append(f"第{i + 1}页\n{text}")
        except Exception as e:
            logger.warning("OCR 第 %d 页失败：%s", i + 1, e)
    doc.close()
    return _StaticLoader("\n\n".join(texts))


def _load_image(file_path: str):
    """图片文件：直接 OCR"""
    import base64
    import io
    from PIL import Image

    img = Image.open(file_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return _StaticLoader(_ocr_image_b64(b64))


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


def _doc_title(text: str) -> str:
    """取文档首个一级标题作为标题"""
    m = re.search(r'^#\s+(.+?)\s*$', text, flags=re.MULTILINE)
    return m.group(1).strip() if m else ""


def _titled(title: str, text: str) -> str:
    """给父块加上文档标题，避免多版本文档的条款无法区分（如 2022/2023/2024 版）"""
    return f"【{title}】{text}" if title else text


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
        title = _doc_title(text)
        sections = splitter_fn(text)
        for section in sections:
            parent_chunks = parent_splitter.split_text(section)
            for parent in parent_chunks:
                titled_parent = _titled(title, parent)
                child_chunks = child_splitter.split_text(parent)
                for child in child_chunks:
                    child_texts.append(child)
                    parent_texts.append(titled_parent)

    logger.info("split_documents type=%s strategy=%s children=%d", doc_type, strategy, len(child_texts))
    return child_texts, parent_texts


def _split_fixed(docs: list[Document]) -> tuple[list[str], list[str]]:
    """基线：固定长度分块（无任何结构感知）"""
    fixed_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    child_texts = []
    parent_texts = []
    for doc in docs:
        title = _doc_title(doc.page_content)
        chunks = fixed_splitter.split_text(doc.page_content)
        for chunk in chunks:
            child_texts.append(chunk)
            parent_texts.append(_titled(title, chunk))
    return child_texts, parent_texts


def _split_header_only(docs: list[Document]) -> tuple[list[str], list[str]]:
    """标题感知分块（仅按标题切，不区分文档类型）"""
    child_texts = []
    parent_texts = []
    for doc in docs:
        title = _doc_title(doc.page_content)
        sections = _split_by_headers(doc.page_content)
        for section in sections:
            parent_chunks = parent_splitter.split_text(section)
            for parent in parent_chunks:
                titled_parent = _titled(title, parent)
                child_chunks = child_splitter.split_text(parent)
                for child in child_chunks:
                    child_texts.append(child)
                    parent_texts.append(titled_parent)
    return child_texts, parent_texts


# ==================== 分区与入库 ====================

PARTITION_MAP = {
    "pdf": "pdf", "docx": "docx", "txt": "txt", "md": "md",
    "xlsx": "xlsx", "csv": "csv", "pptx": "pptx", "html": "html", "json": "json",
}


def get_partition(file_path: str) -> str:
    ext = file_path.rsplit(".", 1)[-1].lower()
    return PARTITION_MAP.get(ext, "txt")


def ingest_file(file_path: str, doc_type: str = "员工手册", department: str = "公共",
                source: str | None = None, strategy: str = "adaptive",
                collection_name: str | None = None) -> int:
    docs = load_file(file_path, doc_type=doc_type)
    docs = clean_documents(docs)
    child_texts, parent_texts = split_documents(docs, doc_type=doc_type, strategy=strategy)

    if not child_texts:
        logger.warning("ingest file=%s 未产出任何块，跳过（内容可能过短）", file_path)
        return 0

    dense_vectors = embed_documents(child_texts)
    sparse_vectors = embed_documents_sparse(child_texts)

    partition = get_partition(file_path)
    insert_kwargs = {"collection_name": collection_name} if collection_name else {}
    source_name = source or os.path.basename(file_path)
    insert_documents(child_texts, parent_texts, dense_vectors, sparse_vectors, partition,
                     doc_type=doc_type, department=department, source=source_name, **insert_kwargs)

    logger.info("ingest file=%s type=%s dept=%s source=%s strategy=%s partition=%s chunks=%d",
                file_path, doc_type, department, source_name, strategy, partition, len(child_texts))
    return len(child_texts)
