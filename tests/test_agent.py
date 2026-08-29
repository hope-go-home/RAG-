"""agent 辅助函数测试（mock 掉重依赖）"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

# mock 掉所有重依赖，避免加载模型和初始化 LLM
for mod in [
    "SmartQuery.rag.embedding", "SmartQuery.backend.database.milvus",
    "SmartQuery.backend.config", "sentence_transformers", "FlagEmbedding",
    "langchain_openai",
]:
    sys.modules[mod] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SmartQuery.rag.agent import _safe_json_parse, _format_history


class TestSafeJsonParse:
    def test_valid_json(self):
        assert _safe_json_parse('{"intent": "rag_query"}') == {"intent": "rag_query"}

    def test_markdown_wrapped(self):
        assert _safe_json_parse('```json\n{"intent": "chit_chat"}\n```') == {"intent": "chit_chat"}

    def test_invalid_json(self):
        assert _safe_json_parse("not json") == {}

    def test_custom_default(self):
        assert _safe_json_parse("bad", default={"x": 1}) == {"x": 1}

    def test_empty(self):
        assert _safe_json_parse("") == {}


class TestFormatHistory:
    def test_empty(self):
        assert "无历史" in _format_history([])

    def test_single_turn(self):
        h = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "嗨！"}]
        r = _format_history(h)
        assert "用户：你好" in r and "助手：嗨！" in r

    def test_max_turns(self):
        h = [{"role": "user", "content": f"q{i}"} for i in range(10)]
        r = _format_history(h, max_turns=3)
        assert "q9" in r and "q0" not in r


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
