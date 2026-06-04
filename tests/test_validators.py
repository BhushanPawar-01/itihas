from __future__ import annotations

from src.utils.validators import validate_metadata, validate_registry_row


def test_validate_metadata_valid_input_returns_empty_list() -> None:
    meta = {
        "doc_id": "ia_trial_19451107_001",
        "url": "https://archive.org/details/example",
        "bias_type": "british_legal",
        "language_original": "en",
        "format": "pdf",
        "processed_at": "2026-05-27T10:32:00Z",
    }

    assert validate_metadata(meta) == []


def test_validate_metadata_missing_doc_id_returns_doc_id_error() -> None:
    meta = {
        "url": "https://archive.org/details/example",
        "bias_type": "british_legal",
        "language_original": "en",
        "format": "pdf",
        "processed_at": None,
    }

    errors = validate_metadata(meta)

    assert len(errors) == 1
    assert "doc_id" in errors[0]


def test_validate_registry_row_valid_input_returns_empty_list() -> None:
    row = {
        "doc_id": "ia_trial_19451107_001",
        "source": "internet_archive",
        "url": "https://archive.org/details/example",
        "format": "pdf",
        "bias_type": "british_legal",
        "downloaded": "true",
    }

    assert validate_registry_row(row) == []


def test_validate_registry_row_missing_doc_id_returns_doc_id_error() -> None:
    row = {
        "source": "internet_archive",
        "url": "https://archive.org/details/example",
        "format": "pdf",
        "bias_type": "british_legal",
        "downloaded": True,
    }

    errors = validate_registry_row(row)

    assert len(errors) == 1
    assert "doc_id" in errors[0]
