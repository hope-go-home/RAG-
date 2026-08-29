"""retriever 纯函数测试（mock 掉 embedding 和 milvus）"""
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

# mock 重依赖
sys.modules["SmartQuery.rag.embedding"] = MagicMock()
sys.modules["SmartQuery.backend.database.milvus"] = MagicMock()
sys.modules["SmartQuery.backend.config"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["FlagEmbedding"] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SmartQuery.rag.retriever import _rrf_scored, _get_cache_key, _retrieval_cache, _cache_lock, CACHE_TTL, RetrievalResult


class TestRRF:
    def test_dual_source_higher_score(self):
        dense = [("d1", "p1", 0.9), ("d2", "p2", 0.8)]
        sparse = [("d1", "p1", 0.7)]
        r = _rrf_scored(dense, sparse, k=30)
        scores = {x[0]: x[2] for x in r}
        assert scores["d1"] > scores["d2"]

    def test_empty(self):
        assert _rrf_scored([], []) == []

    def test_ranking_order(self):
        dense = [("d1", "p1", 0.9), ("d2", "p2", 0.8), ("d3", "p3", 0.7)]
        r = _rrf_scored(dense, [], k=30)
        assert r[0][2] >= r[1][2] >= r[2][2]


class TestCache:
    def test_key_deterministic(self):
        assert _get_cache_key("q", 10) == _get_cache_key("q", 10)
        assert _get_cache_key("a", 10) != _get_cache_key("b", 10)

    def test_put_get(self):
        with _cache_lock:
            _retrieval_cache.clear()
        key = _get_cache_key("test", 10)
        mock_r = RetrievalResult(documents=["d1"], dense_hit_count=1)
        with _cache_lock:
            _retrieval_cache[key] = (time.time(), mock_r)
        _, cached = _retrieval_cache[key]
        assert cached.documents == ["d1"]
        with _cache_lock:
            _retrieval_cache.clear()

    def test_expiry_detection(self):
        with _cache_lock:
            _retrieval_cache.clear()
        key = _get_cache_key("old", 10)
        mock_r = RetrievalResult(documents=["d1"])
        with _cache_lock:
            _retrieval_cache[key] = (time.time() - CACHE_TTL - 1, mock_r)
        t, _ = _retrieval_cache[key]
        assert time.time() - t >= CACHE_TTL
        with _cache_lock:
            _retrieval_cache.clear()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
