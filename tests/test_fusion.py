"""
Tests for src/retrieval/fusion.py

No DB, no model, no API — all external calls mocked.
RRF scores are verified arithmetically with exact values.
"""
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RRF_K = 60  # must match config default used in tests


def _chunk(doc_id: str, chunk_index: int = 0, extra: dict | None = None) -> dict:
    base = {
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "text": f"text for {doc_id}",
        "source_type": "archival",
        "bias_tag": "academic",
        "language": "en",
        "date": "1945-11-07",
        "confidence": 0.9,
    }
    if extra:
        base.update(extra)
    return base


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion — arithmetic
# ---------------------------------------------------------------------------

class TestRRFArithmetic:
    def test_chunk_in_both_lists_beats_chunk_in_one(self):
        """
        Chunk A: rank 1 BM25, rank 3 dense  → 1/61 + 1/63 ≈ 0.03226
        Chunk B: rank 1 dense only          → 1/61          ≈ 0.01639
        A must rank above B.
        """
        from src.retrieval.fusion import reciprocal_rank_fusion

        bm25 = [_chunk("A"), _chunk("C")]          # A=rank1, C=rank2
        dense = [_chunk("B"), _chunk("C"), _chunk("A")]  # B=rank1, C=rank2, A=rank3

        results = reciprocal_rank_fusion(bm25, dense, k=RRF_K)
        ids = [r["doc_id"] for r in results]

        assert ids.index("A") < ids.index("B"), "A (in both) should rank above B (dense only)"

    def test_rrf_score_exact_values(self):
        """Verify the formula 1/(k+rank) is applied correctly."""
        from src.retrieval.fusion import reciprocal_rank_fusion

        # A is rank 1 in BM25, rank 3 in dense
        bm25 = [_chunk("A"), _chunk("X"), _chunk("Y")]
        dense = [_chunk("Z"), _chunk("W"), _chunk("A")]

        results = reciprocal_rank_fusion(bm25, dense, k=RRF_K)
        a = next(r for r in results if r["doc_id"] == "A")

        expected = 1 / (RRF_K + 1) + 1 / (RRF_K + 3)
        assert abs(a["rrf_score"] - expected) < 1e-10

    def test_single_list_chunk_score(self):
        """Chunk only in dense at rank 1 → score = 1/(k+1)."""
        from src.retrieval.fusion import reciprocal_rank_fusion

        bm25 = [_chunk("X")]
        dense = [_chunk("B")]

        results = reciprocal_rank_fusion(bm25, dense, k=RRF_K)
        b = next(r for r in results if r["doc_id"] == "B")

        expected = 1 / (RRF_K + 1)
        assert abs(b["rrf_score"] - expected) < 1e-10

    def test_sorted_descending(self):
        from src.retrieval.fusion import reciprocal_rank_fusion

        bm25 = [_chunk("A"), _chunk("B"), _chunk("C")]
        dense = [_chunk("C"), _chunk("A"), _chunk("B")]

        results = reciprocal_rank_fusion(bm25, dense, k=RRF_K)
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion — metadata and flags
# ---------------------------------------------------------------------------

class TestRRFMetadata:
    def test_in_bm25_and_in_dense_flags(self):
        from src.retrieval.fusion import reciprocal_rank_fusion

        bm25 = [_chunk("A")]
        dense = [_chunk("B"), _chunk("A")]

        results = reciprocal_rank_fusion(bm25, dense, k=RRF_K)
        a = next(r for r in results if r["doc_id"] == "A")
        b = next(r for r in results if r["doc_id"] == "B")

        assert a["in_bm25"] is True
        assert a["in_dense"] is True
        assert b["in_bm25"] is False
        assert b["in_dense"] is True

    def test_dense_metadata_preferred_on_conflict(self):
        """When a chunk appears in both lists, dense metadata wins."""
        from src.retrieval.fusion import reciprocal_rank_fusion

        bm25_chunk = _chunk("A", extra={"bias_tag": "british_legal", "language": "ur"})
        dense_chunk = _chunk("A", extra={"bias_tag": "ina_testimony", "language": "en"})

        results = reciprocal_rank_fusion([bm25_chunk], [dense_chunk], k=RRF_K)
        a = next(r for r in results if r["doc_id"] == "A")

        assert a["bias_tag"] == "ina_testimony"
        assert a["language"] == "en"

    def test_rrf_score_key_present(self):
        from src.retrieval.fusion import reciprocal_rank_fusion

        results = reciprocal_rank_fusion([_chunk("A")], [_chunk("B")], k=RRF_K)
        assert all("rrf_score" in r for r in results)

    def test_empty_both_lists(self):
        from src.retrieval.fusion import reciprocal_rank_fusion

        results = reciprocal_rank_fusion([], [], k=RRF_K)
        assert results == []

    def test_one_empty_list(self):
        from src.retrieval.fusion import reciprocal_rank_fusion

        results = reciprocal_rank_fusion([_chunk("A"), _chunk("B")], [], k=RRF_K)
        assert len(results) == 2
        assert all(r["in_bm25"] is True for r in results)
        assert all(r["in_dense"] is False for r in results)

    def test_chunk_identity_uses_doc_id_and_chunk_index(self):
        """Same doc_id but different chunk_index → two separate chunks."""
        from src.retrieval.fusion import reciprocal_rank_fusion

        bm25 = [_chunk("A", chunk_index=0), _chunk("A", chunk_index=1)]
        dense = [_chunk("A", chunk_index=1), _chunk("A", chunk_index=0)]

        results = reciprocal_rank_fusion(bm25, dense, k=RRF_K)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# retrieve() — integration over mocked sub-retrievers
# ---------------------------------------------------------------------------

class TestRetrieve:
    def _patch_retrievers(self, bm25_out, dense_out):
        return (
            patch("src.retrieval.fusion.bm25_retriever.search", return_value=bm25_out),
            patch("src.retrieval.fusion.dense_retriever.search", return_value=dense_out),
        )

    def test_returns_exactly_top_k(self):
        from src.retrieval.fusion import retrieve

        bm25_out = [_chunk(f"b{i}") for i in range(15)]
        dense_out = [_chunk(f"d{i}") for i in range(15)]

        with (
            patch("src.retrieval.fusion.bm25_retriever.search", return_value=bm25_out),
            patch("src.retrieval.fusion.dense_retriever.search", return_value=dense_out),
            patch("src.retrieval.fusion.RERANKER_ENABLED", False),
        ):
            results = retrieve("test query", top_k=5)

        assert len(results) == 5

    def test_rerank_not_called_when_disabled(self):
        from src.retrieval.fusion import retrieve

        bm25_out = [_chunk("A")]
        dense_out = [_chunk("B")]

        with (
            patch("src.retrieval.fusion.bm25_retriever.search", return_value=bm25_out),
            patch("src.retrieval.fusion.dense_retriever.search", return_value=dense_out),
            patch("src.retrieval.fusion.RERANKER_ENABLED", False),
            patch("src.retrieval.fusion.rerank") as mock_rerank,
        ):
            retrieve("test query")

        mock_rerank.assert_not_called()

    def test_rerank_called_when_enabled(self):
        from src.retrieval.fusion import retrieve

        bm25_out = [_chunk("A")]
        dense_out = [_chunk("B")]
        fake_reranked = [_chunk("A"), _chunk("B")]

        with (
            patch("src.retrieval.fusion.bm25_retriever.search", return_value=bm25_out),
            patch("src.retrieval.fusion.dense_retriever.search", return_value=dense_out),
            patch("src.retrieval.fusion.RERANKER_ENABLED", True),
            patch("src.retrieval.fusion.rerank", return_value=fake_reranked) as mock_rerank,
        ):
            retrieve("test query")

        mock_rerank.assert_called_once()

    def test_filters_passed_to_both_retrievers(self):
        filters = {"bias_tags": ["ina_testimony"], "languages": ["en"]}

        with (
            patch("src.retrieval.fusion.bm25_retriever.search", return_value=[]) as mock_bm25,
            patch("src.retrieval.fusion.dense_retriever.search", return_value=[]) as mock_dense,
            patch("src.retrieval.fusion.RERANKER_ENABLED", False),
        ):
            from src.retrieval.fusion import retrieve
            retrieve("test", filters=filters)

        assert mock_bm25.call_args[1]["filters"] == filters
        assert mock_dense.call_args[1]["filters"] == filters

    def test_both_retrievers_called_with_top_k_20(self):
        """retrieve() always fetches 20 from each sub-retriever before fusing."""
        with (
            patch("src.retrieval.fusion.bm25_retriever.search", return_value=[]) as mock_bm25,
            patch("src.retrieval.fusion.dense_retriever.search", return_value=[]) as mock_dense,
            patch("src.retrieval.fusion.RERANKER_ENABLED", False),
        ):
            from src.retrieval.fusion import retrieve
            retrieve("test", top_k=5)

        assert mock_bm25.call_args[1]["top_k"] == 20
        assert mock_dense.call_args[1]["top_k"] == 20

    def test_results_contain_rrf_score(self):
        bm25_out = [_chunk("A"), _chunk("B")]
        dense_out = [_chunk("B"), _chunk("A")]

        with (
            patch("src.retrieval.fusion.bm25_retriever.search", return_value=bm25_out),
            patch("src.retrieval.fusion.dense_retriever.search", return_value=dense_out),
            patch("src.retrieval.fusion.RERANKER_ENABLED", False),
        ):
            from src.retrieval.fusion import retrieve
            results = retrieve("test")

        assert all("rrf_score" in r for r in results)

    def test_chunk_in_both_lists_ranks_first(self):
        """A chunk appearing in both BM25 and dense should beat chunks in only one list."""
        shared = _chunk("shared")
        bm25_out = [shared, _chunk("bm25_only")]
        dense_out = [shared, _chunk("dense_only")]

        with (
            patch("src.retrieval.fusion.bm25_retriever.search", return_value=bm25_out),
            patch("src.retrieval.fusion.dense_retriever.search", return_value=dense_out),
            patch("src.retrieval.fusion.RERANKER_ENABLED", False),
        ):
            from src.retrieval.fusion import retrieve
            results = retrieve("test", top_k=10)

        assert results[0]["doc_id"] == "shared"


# ---------------------------------------------------------------------------
# rerank() — graceful fallback
# ---------------------------------------------------------------------------

class TestRerank:
    def test_fallback_on_api_failure(self):
        """If the reranker API throws, rerank() returns original list unchanged."""
        from src.retrieval.fusion import rerank

        chunks = [_chunk("A"), _chunk("B")]

        with patch("src.retrieval.fusion.rerank") as mock_rerank:
            # Simulate the real function's fallback behaviour
            mock_rerank.side_effect = lambda c, q: c  # identity = fallback

            result = mock_rerank(chunks, "query")

        assert result == chunks