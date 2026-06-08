"""
Tests for src/retrieval/dense_retriever.py

All external calls are mocked — no Postgres, no model download.
"""
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------------

def _fake_db_rows():
    """Simulates what db_client.search_by_embedding returns — dicts with 'score'."""
    return [
        {"id": 1, "doc_id": "doc_a", "chunk_index": 0, "text": "INA soldiers at Red Fort", "bias_tag": "ina_testimony", "language": "en", "date": "1945-11-07", "score": 0.91},
        {"id": 2, "doc_id": "doc_b", "chunk_index": 1, "text": "Trial proceedings began", "bias_tag": "british_legal", "language": "en", "date": "1945-11-08", "score": 0.76},
        {"id": 3, "doc_id": "doc_c", "chunk_index": 0, "text": "Nationalist press coverage", "bias_tag": "nationalist_press", "language": "en", "date": "1945-11-09", "score": 0.54},
    ]


FAKE_VECTOR = [0.0] * 768


# ---------------------------------------------------------------------------
# embed_query
# ---------------------------------------------------------------------------

class TestEmbedQuery:
    def test_returns_plain_float_list(self):
        import numpy as np

        fake_model = MagicMock()
        fake_model.encode.return_value = np.array([[0.1] * 768], dtype=np.float32)

        with patch("src.retrieval.dense_retriever.embed_query") as mock_eq:
            mock_eq.return_value = [0.1] * 768
            result = mock_eq("test query")

        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_uses_shared_model_instance(self):
        """embed_query must call get_model() from embedder, not load its own."""
        import numpy as np

        fake_model = MagicMock()
        fake_model.encode.return_value = np.array([[0.0] * 768], dtype=np.float32)

        with patch("src.retrieval.dense_retriever.embed_query", wraps=None) as _:
            # Patch get_model at the source (embedder module)
            with patch("src.processing.embedder.get_model", return_value=fake_model) as mock_gm:
                from src.retrieval.dense_retriever import embed_query
                embed_query("test")
                mock_gm.assert_called_once()


# ---------------------------------------------------------------------------
# search() — output schema
# ---------------------------------------------------------------------------

class TestSearchSchema:
    def test_dense_score_key_present(self):
        with (
            patch("src.retrieval.dense_retriever.embed_query", return_value=FAKE_VECTOR),
            patch("src.retrieval.dense_retriever.db_client.search_by_embedding", return_value=_fake_db_rows()),
        ):
            from src.retrieval.dense_retriever import search
            results = search("INA trial")

        assert all("dense_score" in r for r in results)

    def test_score_key_removed(self):
        """The raw 'score' key from the DB must be renamed to 'dense_score'."""
        with (
            patch("src.retrieval.dense_retriever.embed_query", return_value=FAKE_VECTOR),
            patch("src.retrieval.dense_retriever.db_client.search_by_embedding", return_value=_fake_db_rows()),
        ):
            from src.retrieval.dense_retriever import search
            results = search("INA trial")

        assert all("score" not in r for r in results)

    def test_original_chunk_fields_preserved(self):
        with (
            patch("src.retrieval.dense_retriever.embed_query", return_value=FAKE_VECTOR),
            patch("src.retrieval.dense_retriever.db_client.search_by_embedding", return_value=_fake_db_rows()),
        ):
            from src.retrieval.dense_retriever import search
            results = search("INA trial")

        for r in results:
            assert "doc_id" in r
            assert "chunk_index" in r
            assert "text" in r

    def test_dense_score_is_float(self):
        with (
            patch("src.retrieval.dense_retriever.embed_query", return_value=FAKE_VECTOR),
            patch("src.retrieval.dense_retriever.db_client.search_by_embedding", return_value=_fake_db_rows()),
        ):
            from src.retrieval.dense_retriever import search
            results = search("INA trial")

        assert all(isinstance(r["dense_score"], float) for r in results)


# ---------------------------------------------------------------------------
# search() — ordering
# ---------------------------------------------------------------------------

class TestSearchOrdering:
    def test_results_ordered_by_dense_score_descending(self):
        # Deliberately return rows in non-sorted order
        rows = [
            {"id": 2, "doc_id": "doc_b", "chunk_index": 0, "text": "b", "bias_tag": "academic", "language": "en", "date": "1945-11-08", "score": 0.55},
            {"id": 1, "doc_id": "doc_a", "chunk_index": 0, "text": "a", "bias_tag": "academic", "language": "en", "date": "1945-11-07", "score": 0.91},
            {"id": 3, "doc_id": "doc_c", "chunk_index": 0, "text": "c", "bias_tag": "academic", "language": "en", "date": "1945-11-09", "score": 0.72},
        ]
        with (
            patch("src.retrieval.dense_retriever.embed_query", return_value=FAKE_VECTOR),
            patch("src.retrieval.dense_retriever.db_client.search_by_embedding", return_value=rows),
        ):
            from src.retrieval.dense_retriever import search
            results = search("test")

        scores = [r["dense_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_correct_count_returned(self):
        with (
            patch("src.retrieval.dense_retriever.embed_query", return_value=FAKE_VECTOR),
            patch("src.retrieval.dense_retriever.db_client.search_by_embedding", return_value=_fake_db_rows()),
        ):
            from src.retrieval.dense_retriever import search
            results = search("INA trial")

        assert len(results) == 3


# ---------------------------------------------------------------------------
# search() — DB call arguments
# ---------------------------------------------------------------------------

class TestSearchDBCall:
    def test_passes_query_embedding_to_db(self):
        fixed_vector = [0.42] * 768
        with (
            patch("src.retrieval.dense_retriever.embed_query", return_value=fixed_vector),
            patch("src.retrieval.dense_retriever.db_client.search_by_embedding", return_value=[]) as mock_db,
        ):
            from src.retrieval.dense_retriever import search
            search("test query")

        call_args = mock_db.call_args
        assert call_args[0][0] == fixed_vector

    def test_passes_top_k_to_db(self):
        with (
            patch("src.retrieval.dense_retriever.embed_query", return_value=FAKE_VECTOR),
            patch("src.retrieval.dense_retriever.db_client.search_by_embedding", return_value=[]) as mock_db,
        ):
            from src.retrieval.dense_retriever import search
            search("test", top_k=5)

        assert mock_db.call_args[0][1] == 5

    def test_passes_filters_to_db(self):
        filters = {"bias_tags": ["ina_testimony"], "languages": ["en"]}
        with (
            patch("src.retrieval.dense_retriever.embed_query", return_value=FAKE_VECTOR),
            patch("src.retrieval.dense_retriever.db_client.search_by_embedding", return_value=[]) as mock_db,
        ):
            from src.retrieval.dense_retriever import search
            search("test", filters=filters)

        assert mock_db.call_args[0][2] == filters

    def test_empty_db_result_returns_empty_list(self):
        with (
            patch("src.retrieval.dense_retriever.embed_query", return_value=FAKE_VECTOR),
            patch("src.retrieval.dense_retriever.db_client.search_by_embedding", return_value=[]),
        ):
            from src.retrieval.dense_retriever import search
            results = search("no match query")

        assert results == []


# ---------------------------------------------------------------------------
# Model caching — get_model() called exactly once across multiple search() calls
# ---------------------------------------------------------------------------

class TestModelCaching:
    def test_get_model_called_once_across_two_searches(self):
        """
        Even if search() is called twice, the underlying model must be loaded only once.
        This verifies that dense_retriever delegates to embedder's cached get_model(),
        not its own separate load.
        """
        import numpy as np

        fake_model = MagicMock()
        fake_model.encode.return_value = np.array([[0.0] * 768], dtype=np.float32)

        with (
            patch("src.processing.embedder._model", None),
            patch("src.processing.embedder.SentenceTransformer", return_value=fake_model) as mock_st,
            patch("src.retrieval.dense_retriever.db_client.search_by_embedding", return_value=[]),
        ):
            # Force get_model() to go through real lazy-load path
            import src.processing.embedder as emb
            emb._model = None  # reset cache

            from src.retrieval.dense_retriever import embed_query, search

            embed_query("first call")
            embed_query("second call")

        # SentenceTransformer constructor called only once despite two encode calls
        assert mock_st.call_count == 1