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

from SmartQuery.rag.ingest import clean_text, _split_by_headers, _split_by_sentences, clean_documents
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


class TestSplitBySentences:
    def test_chinese_periods(self):
        assert len(_split_by_sentences("句一。句二。句三。")) == 3

    def test_mixed(self):
        assert len(_split_by_sentences("问？叹！句。")) == 3

    def test_empty(self):
        assert _split_by_sentences("") == []


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
