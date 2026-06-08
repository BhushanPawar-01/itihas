"""
Tests for src/processing/embedder.py

Rules:
- get_model() is mocked to return a deterministic fake encoder (returns zeros).
- db_client functions are mocked — no real Postgres connection.
- Importing embedder must not trigger a model download.
"""
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_model(dim: int = 768):
    """Returns a mock SentenceTransformer that always produces zero vectors."""
    fake = MagicMock()
    fake.encode = MagicMock(
        side_effect=lambda texts, **kwargs: np.zeros((len(texts), dim), dtype=np.float32)
    )
    return fake


def _make_chunks(n: int, has_embedding: bool = False) -> list[dict]:
    return [
        {
            "id": i + 1,
            "doc_id": f"doc_{i}",
            "chunk_index": 0,
            "text": f"chunk text {i}",
            "embedding": [0.0] * 768 if has_embedding else None,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# embed_texts
# ---------------------------------------------------------------------------

class TestEmbedTexts:
    def test_returns_empty_list_for_empty_input(self):
        """embed_texts([]) must return [] without loading the model."""
        # If get_model() were called it would need sentence_transformers installed;
        # the empty-input guard must short-circuit before that.
        from src.processing.embedder import embed_texts

        result = embed_texts([])
        assert result == []

    def test_returns_list_of_plain_float_lists(self):
        """Each element must be a plain Python list of floats, not a numpy array."""
        fake_model = _make_fake_model(dim=768)
        with patch("src.processing.embedder.get_model", return_value=fake_model):
            from src.processing.embedder import embed_texts

            result = embed_texts(["hello", "world"])

        assert isinstance(result, list)
        assert len(result) == 2
        for vec in result:
            assert isinstance(vec, list), "Expected plain list, got " + type(vec).__name__
            assert all(isinstance(x, float) for x in vec)

    def test_vector_dimension_matches_model_output(self):
        fake_model = _make_fake_model(dim=768)
        with patch("src.processing.embedder.get_model", return_value=fake_model):
            from src.processing.embedder import embed_texts

            result = embed_texts(["test"])

        assert len(result[0]) == 768

    def test_passes_batch_size_to_encode(self):
        """embed_texts must forward EMBEDDING_BATCH_SIZE to model.encode."""
        fake_model = _make_fake_model()
        with (
            patch("src.processing.embedder.get_model", return_value=fake_model),
            patch("src.processing.embedder.EMBEDDING_BATCH_SIZE", 32),
        ):
            from src.processing.embedder import embed_texts

            embed_texts(["a", "b"])

        call_kwargs = fake_model.encode.call_args[1]
        assert call_kwargs["batch_size"] == 32


# ---------------------------------------------------------------------------
# embed_all
# ---------------------------------------------------------------------------

class TestEmbedAll:
    def test_calls_update_embedding_once_per_chunk(self):
        """embed_all must call db_client.update_embedding exactly once per chunk."""
        chunks = _make_chunks(5)
        fake_model = _make_fake_model()

        with (
            patch("src.processing.embedder.get_model", return_value=fake_model),
            patch(
                "src.processing.embedder.db_client.get_chunks_without_embeddings",
                return_value=chunks,
            ),
            patch(
                "src.processing.embedder.db_client.update_embedding"
            ) as mock_update,
        ):
            from src.processing.embedder import embed_all

            result = embed_all(force=False)

        assert mock_update.call_count == 5
        assert result["embedded"] == 5

    def test_force_false_uses_null_embedding_query(self):
        """force=False must call get_chunks_without_embeddings, not get_all_texts_for_bm25."""
        chunks = _make_chunks(3)
        fake_model = _make_fake_model()

        with (
            patch("src.processing.embedder.get_model", return_value=fake_model),
            patch(
                "src.processing.embedder.db_client.get_chunks_without_embeddings",
                return_value=chunks,
            ) as mock_null,
            patch(
                "src.processing.embedder.db_client.get_all_texts_for_bm25"
            ) as mock_all,
            patch("src.processing.embedder.db_client.update_embedding"),
        ):
            from src.processing.embedder import embed_all

            embed_all(force=False)

        mock_null.assert_called_once()
        mock_all.assert_not_called()

    def test_force_true_uses_all_chunks_query(self):
        """force=True must call get_all_texts_for_bm25 to re-embed everything."""
        chunks = _make_chunks(3, has_embedding=True)
        fake_model = _make_fake_model()

        with (
            patch("src.processing.embedder.get_model", return_value=fake_model),
            patch(
                "src.processing.embedder.db_client.get_all_texts_for_bm25",
                return_value=chunks,
            ) as mock_all,
            patch(
                "src.processing.embedder.db_client.get_chunks_without_embeddings"
            ) as mock_null,
            patch("src.processing.embedder.db_client.update_embedding"),
        ):
            from src.processing.embedder import embed_all

            embed_all(force=True)

        mock_all.assert_called_once()
        mock_null.assert_not_called()

    def test_no_chunks_returns_early_without_calling_model(self):
        """When there are no chunks to embed, get_model must never be called."""
        with (
            patch(
                "src.processing.embedder.db_client.get_chunks_without_embeddings",
                return_value=[],
            ),
            patch("src.processing.embedder.get_model") as mock_get_model,
        ):
            from src.processing.embedder import embed_all

            result = embed_all(force=False)

        mock_get_model.assert_not_called()
        assert result["embedded"] == 0

    def test_update_embedding_receives_correct_chunk_id(self):
        """Each update_embedding call must receive the id from the corresponding chunk."""
        chunks = _make_chunks(3)  # ids are 1, 2, 3
        fake_model = _make_fake_model(dim=768)

        captured_ids = []

        def capture_update(chunk_id, embedding):
            captured_ids.append(chunk_id)

        with (
            patch("src.processing.embedder.get_model", return_value=fake_model),
            patch(
                "src.processing.embedder.db_client.get_chunks_without_embeddings",
                return_value=chunks,
            ),
            patch(
                "src.processing.embedder.db_client.update_embedding",
                side_effect=capture_update,
            ),
        ):
            from src.processing.embedder import embed_all

            embed_all(force=False)

        assert captured_ids == [1, 2, 3]

    def test_result_dict_has_required_keys(self):
        """Return value must contain 'embedded', 'model', and 'device'."""
        chunks = _make_chunks(2)
        fake_model = _make_fake_model()

        with (
            patch("src.processing.embedder.get_model", return_value=fake_model),
            patch(
                "src.processing.embedder.db_client.get_chunks_without_embeddings",
                return_value=chunks,
            ),
            patch("src.processing.embedder.db_client.update_embedding"),
        ):
            from src.processing.embedder import embed_all

            result = embed_all(force=False)

        assert "embedded" in result
        assert "model" in result
        assert "device" in result
        assert result["device"] in ("cpu", "cuda")