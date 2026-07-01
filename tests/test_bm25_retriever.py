"""
Tests for src/retrieval/bm25_retriever.py

All DB calls are mocked — no Postgres connection required.
"""
from unittest.mock import patch

import pytest


SEED_CHUNKS = [
    {"id": 1, "doc_id": "doc_a", "chunk_index": 0, "text": "British officers arrested the soldiers", "source_type": "archival", "bias_tag": "british_legal", "language": "en", "confidence": 0.9},
    {"id": 2, "doc_id": "doc_b", "chunk_index": 0, "text": "Shah Nawaz Khan led the INA regiment bravely", "source_type": "archival", "bias_tag": "ina_testimony", "language": "en", "confidence": 0.9},
    {"id": 3, "doc_id": "doc_c", "chunk_index": 0, "text": "The weather in Delhi was hot that summer", "source_type": "archival", "bias_tag": "academic", "language": "en", "confidence": 0.9},
    {"id": 4, "doc_id": "doc_d", "chunk_index": 0, "text": "INA regiment marched under Shah Nawaz Khan command", "source_type": "archival", "bias_tag": "nationalist_press", "language": "ur", "confidence": 0.9},
    {"id": 5, "doc_id": "doc_e", "chunk_index": 0, "text": "Colonial administration maintained detailed records", "source_type": "archival", "bias_tag": "british_military", "language": "en", "confidence": 0.9},
]


def _make_retriever(chunks=None):
    """Returns a freshly built BM25Retriever with mocked DB data."""
    from src.retrieval.bm25_retriever import BM25Retriever

    data = chunks if chunks is not None else SEED_CHUNKS
    with patch("src.retrieval.bm25_retriever.db_client.get_all_texts_for_bm25", return_value=data):
        r = BM25Retriever()
        r.build()
    return r


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------

class TestBuild:
    def test_built_flag_set_after_build(self):
        r = _make_retriever()
        assert r._built is True

    def test_chunks_stored(self):
        r = _make_retriever()
        assert len(r._chunks) == len(SEED_CHUNKS)

    def test_bm25_object_created(self):
        from rank_bm25 import BM25Okapi
        r = _make_retriever()
        assert isinstance(r._bm25, BM25Okapi)

    def test_build_called_lazily_on_first_search(self):
        from src.retrieval.bm25_retriever import BM25Retriever

        r = BM25Retriever()
        assert r._built is False

        with patch(
            "src.retrieval.bm25_retriever.db_client.get_all_texts_for_bm25",
            return_value=SEED_CHUNKS,
        ):
            r.search("INA")

        assert r._built is True


# ---------------------------------------------------------------------------
# search() — basic ranking
# ---------------------------------------------------------------------------

class TestSearch:
    def test_matching_chunks_returned(self):
        """Query 'Shah Nawaz Khan' should return chunks 2 and 4 (both contain those terms)."""
        r = _make_retriever()
        results = r.search("Shah Nawaz Khan")
        doc_ids = [res["doc_id"] for res in results]
        assert "doc_b" in doc_ids
        assert "doc_d" in doc_ids

    def test_results_ordered_by_score_descending(self):
        r = _make_retriever()
        results = r.search("Shah Nawaz Khan")
        scores = [res["bm25_score"] for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_zero_score_chunks_excluded(self):
        """Chunks with no query term overlap must not appear in results."""
        r = _make_retriever()
        # "weather Delhi summer" shares no terms with chunks 2 or 4
        results = r.search("Shah Nawaz Khan")
        for res in results:
            assert res["bm25_score"] > 0.0

    def test_bm25_score_key_present(self):
        r = _make_retriever()
        results = r.search("INA regiment")
        assert all("bm25_score" in res for res in results)

    def test_bm25_score_is_float(self):
        r = _make_retriever()
        results = r.search("INA regiment")
        assert all(isinstance(res["bm25_score"], float) for res in results)

    def test_top_k_limits_results(self):
        r = _make_retriever()
        results = r.search("the", top_k=2)
        assert len(results) <= 2

    def test_no_match_returns_empty_list(self):
        r = _make_retriever()
        results = r.search("xyzzy nonexistent term qqq")
        assert results == []

    def test_original_chunk_fields_preserved(self):
        """Result dicts must contain original chunk fields alongside bm25_score."""
        r = _make_retriever()
        results = r.search("INA regiment")
        for res in results:
            assert "doc_id" in res
            assert "chunk_index" in res
            assert "text" in res


# ---------------------------------------------------------------------------
# search() — filters
# ---------------------------------------------------------------------------

class TestFilters:
    def test_bias_tag_filter_excludes_non_matching(self):
        r = _make_retriever()
        results = r.search("INA regiment", filters={"bias_tags": ["ina_testimony"]})
        for res in results:
            assert res["bias_tag"] == "ina_testimony"

    def test_language_filter_excludes_non_matching(self):
        r = _make_retriever()
        # chunk 4 is Urdu; filter to English only → chunk 4 must not appear
        results = r.search("Shah Nawaz Khan", filters={"languages": ["en"]})
        for res in results:
            assert res["language"] == "en"
        doc_ids = [res["doc_id"] for res in results]
        assert "doc_d" not in doc_ids

    def test_combined_filters(self):
        r = _make_retriever()
        results = r.search(
            "INA Shah Nawaz",
            filters={"bias_tags": ["nationalist_press"], "languages": ["ur"]},
        )
        for res in results:
            assert res["bias_tag"] == "nationalist_press"
            assert res["language"] == "ur"

    def test_filter_that_matches_nothing_returns_empty(self):
        r = _make_retriever()
        results = r.search("INA", filters={"bias_tags": ["nonexistent_tag"]})
        assert results == []

    def test_none_filters_does_not_filter(self):
        """filters=None must behave identically to no filters."""
        r = _make_retriever()
        results_no_filter = r.search("INA regiment")
        results_none = r.search("INA regiment", filters=None)
        assert [res["doc_id"] for res in results_no_filter] == [
            res["doc_id"] for res in results_none
        ]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_retriever_returns_same_instance(self):
        import src.retrieval.bm25_retriever as mod

        # Reset singleton for test isolation
        mod._retriever = None

        with patch(
            "src.retrieval.bm25_retriever.db_client.get_all_texts_for_bm25",
            return_value=SEED_CHUNKS,
        ):
            r1 = mod.get_retriever()
            r2 = mod.get_retriever()

        assert r1 is r2
        mod._retriever = None  # clean up

    def test_module_search_delegates_to_singleton(self):
        import src.retrieval.bm25_retriever as mod

        mod._retriever = None

        with patch(
            "src.retrieval.bm25_retriever.db_client.get_all_texts_for_bm25",
            return_value=SEED_CHUNKS,
        ):
            results = mod.search("Shah Nawaz Khan")

        assert isinstance(results, list)
        mod._retriever = None  # clean up