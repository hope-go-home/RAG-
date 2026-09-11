"""ingest 纯函数测试（mock 掉 embedding 和 milvus）"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

# mock 重依赖
sys.modules["SmartQuery.rag.embedding"] = MagicMock()
sys.modules["SmartQuery.backend.database.milvus"] = MagicMock()
sys.modules["SmartQuery.backend.config"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["FlagEmbedding"] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SmartQuery.rag.ingest import (
    clean_text,
    clean_documents,
    _split_by_headers,
    _split_by_articles,
    _split_faq,
    _split_by_steps,
    _split_by_sections,
    _split_by_table,
    _whole_document,
    get_partition,
)
from langchain_core.documents import Document


class TestCleanText:
    def test_removes_control_chars(self):
        assert clean_text("hello\x00world") == "helloworld"

    def test_removes_page_header(self):
        assert "第 1 页" not in clean_text("第 1 页\n正文")

    def test_removes_page_footer(self):
        assert "共 5 页" not in clean_text("正文\n共 5 页")

    def test_removes_special_chars(self):
        r = clean_text("@#$%^ 中文")
        assert "@#$%^" not in r and "中文" in r

    def test_collapses_blank_lines(self):
        assert "\n\n\n" not in clean_text("a\n\n\n\nb")

    def test_empty(self):
        assert clean_text("") == ""


class TestSplitByHeaders:
    def test_no_headers(self):
        assert len(_split_by_headers("纯文本")) == 1

    def test_two_headers(self):
        r = _split_by_headers("# H1\n内容1\n## H2\n内容2")
        assert len(r) == 2
        assert "H1" in r[0] and "H2" in r[1]

    def test_header_preserved(self):
        r = _split_by_headers("# Title\nBody")
        assert r[0].startswith("#")


class TestSplitByArticles:
    def test_two_articles(self):
        text = "第一条 内容一。\n第二条 内容二。"
        r = _split_by_articles(text)
        assert len(r) == 2
        assert "第一条" in r[0] and "第二条" in r[1]

    def test_no_articles(self):
        assert _split_by_articles("纯文本") == ["纯文本"]

    def test_article_not_broken(self):
        text = "第一条 第一条的内容包含很多信息。\n第二条 第二条内容。"
        r = _split_by_articles(text)
        assert "第一条的内容" in r[0]


class TestSplitFaq:
    def test_qa_pairs(self):
        text = "Q: 问题一？\nA: 答案一。\nQ: 问题二？\nA: 答案二。"
        r = _split_faq(text)
        assert len(r) == 2
        assert "问题一" in r[0] and "问题二" in r[1]

    def test_no_qa(self):
        assert _split_faq("纯文本") == ["纯文本"]


class TestSplitBySteps:
    def test_numbered_steps(self):
        text = "步骤1：第一步\n内容一\n步骤2：第二步\n内容二"
        r = _split_by_steps(text)
        assert len(r) == 2

    def test_arabic_numbering(self):
        text = "1. 第一项\n内容一\n2. 第二项\n内容二"
        r = _split_by_steps(text)
        assert len(r) == 2

    def test_no_steps(self):
        assert _split_by_steps("纯文本") == ["纯文本"]


class TestSplitBySections:
    def test_sections(self):
        text = "## 第一节\n内容一\n## 第二节\n内容二"
        r = _split_by_sections(text)
        assert len(r) == 2

    def test_code_block_not_broken(self):
        text = "## 代码\n说明\n```python\nprint(1)\nprint(2)\n```\n结束"
        r = _split_by_sections(text)
        joined = "\n".join(r)
        assert "print(1)" in joined and "print(2)" in joined


class TestSplitByTable:
    def test_table_preserved(self):
        text = "| 列1 | 列2 |\n|-----|-----|\n| a | b |\n| c | d |"
        r = _split_by_table(text)
        joined = "\n".join(r)
        assert "| a | b |" in joined and "| c | d |" in joined


class TestWholeDocument:
    def test_single_block(self):
        assert _whole_document("短文") == ["短文"]


class TestGetPartition:
    def test_known_ext(self):
        assert get_partition("a.md") == "md"
        assert get_partition("a.xlsx") == "xlsx"

    def test_unknown_ext_fallback(self):
        assert get_partition("a.xyz") == "txt"


class TestCleanDocuments:
    def test_filters_short(self):
        short = Document(page_content="短")
        long = Document(page_content="这是一个非常长的文本内容用来测试文档清洗过滤功能是否正常工作超过五十个字符以上的要求确保不会被过滤掉")
        assert len(clean_documents([short, long])) == 1

    def test_keeps_long(self):
        docs = [Document(page_content="这是一个足够长的文本内容，用于测试文档清洗功能是否正常工作，确保长内容被保留下来超过五十个字符以上要求。")]
        assert len(clean_documents(docs)) == 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
