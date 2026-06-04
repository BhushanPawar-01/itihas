from __future__ import annotations

import csv

import pytest

from src.storage import file_store


@pytest.fixture()
def file_store_paths(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    text_dir = tmp_path / "processed" / "text"
    chunks_dir = tmp_path / "processed" / "chunks"
    metadata_dir = tmp_path / "processed" / "metadata"
    registry_path = tmp_path / "registry.csv"

    monkeypatch.setattr(file_store, "RAW_DIR", raw_dir)
    monkeypatch.setattr(file_store, "PROCESSED_TEXT_DIR", text_dir)
    monkeypatch.setattr(file_store, "PROCESSED_CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(file_store, "METADATA_DIR", metadata_dir)
    monkeypatch.setattr(file_store, "REGISTRY_PATH", registry_path)

    return {
        "raw": raw_dir,
        "text": text_dir,
        "chunks": chunks_dir,
        "metadata": metadata_dir,
        "registry": registry_path,
    }


def test_read_raw_finds_txt_file(file_store_paths) -> None:
    doc_id = "ia_trial_19451107_001"
    path = file_store_paths["raw"] / "internet_archive" / f"{doc_id}.txt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"trial text")

    data, extension = file_store.read_raw(doc_id)

    assert data == b"trial text"
    assert extension == "txt"


def test_read_raw_finds_pdf_file(file_store_paths) -> None:
    doc_id = "ia_trial_19451107_002"
    path = file_store_paths["raw"] / "internet_archive" / f"{doc_id}.pdf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"%PDF")

    data, extension = file_store.read_raw(doc_id)

    assert data == b"%PDF"
    assert extension == "pdf"


def test_read_raw_prefers_txt_over_pdf(file_store_paths) -> None:
    doc_id = "ia_trial_19451107_003"
    raw_dir = file_store_paths["raw"] / "internet_archive"
    raw_dir.mkdir(parents=True)
    (raw_dir / f"{doc_id}.pdf").write_bytes(b"%PDF")
    (raw_dir / f"{doc_id}.txt").write_bytes(b"text wins")

    data, extension = file_store.read_raw(doc_id)

    assert data == b"text wins"
    assert extension == "txt"


def test_read_raw_raises_file_not_found_for_unknown_doc_id(file_store_paths) -> None:
    file_store_paths["raw"].mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="unknown_doc"):
        file_store.read_raw("unknown_doc")


def test_processed_text_round_trip(file_store_paths) -> None:
    doc_id = "ia_trial_19451107_004"
    text = "Proceedings day one\nEvidence recorded."

    path = file_store.write_processed_text(doc_id, text)

    assert path == file_store_paths["text"] / f"{doc_id}.txt"
    assert file_store.read_processed_text(doc_id) == text


def test_metadata_round_trip(file_store_paths) -> None:
    doc_id = "ia_trial_19451107_005"
    meta = {
        "doc_id": doc_id,
        "url": "https://archive.org/details/example",
        "format": "pdf",
        "bias_type": "british_legal",
    }

    path = file_store.write_metadata(doc_id, meta)

    assert path == file_store_paths["metadata"] / f"{doc_id}.json"
    assert file_store.read_metadata(doc_id) == meta


def test_write_metadata_cleans_tmp_file_after_success(file_store_paths) -> None:
    doc_id = "ia_trial_19451107_006"
    old_meta = {"doc_id": doc_id, "title": "old"}
    new_meta = {"doc_id": doc_id, "title": "new"}

    path = file_store.write_metadata(doc_id, old_meta)
    file_store.write_metadata(doc_id, new_meta)

    assert file_store.read_metadata(doc_id) == new_meta
    assert not path.with_suffix(".json.tmp").exists()


def test_update_registry_updates_only_specified_fields(file_store_paths) -> None:
    registry_path = file_store_paths["registry"]
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["doc_id", "source", "url", "format", "bias_type", "downloaded", "notes"]
    rows = [
        {
            "doc_id": "ia_trial_19451107_007",
            "source": "internet_archive",
            "url": "https://archive.org/details/one",
            "format": "pdf",
            "bias_type": "unknown",
            "downloaded": "false",
            "notes": "keep",
        },
        {
            "doc_id": "ia_trial_19451107_008",
            "source": "internet_archive",
            "url": "https://archive.org/details/two",
            "format": "txt",
            "bias_type": "academic",
            "downloaded": "false",
            "notes": "untouched",
        },
    ]
    with open(registry_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    file_store.update_registry("ia_trial_19451107_007", {"downloaded": "true", "bias_type": "british_legal"})

    with open(registry_path, newline="", encoding="utf-8") as fh:
        updated_rows = list(csv.DictReader(fh))

    assert updated_rows[0] == {
        "doc_id": "ia_trial_19451107_007",
        "source": "internet_archive",
        "url": "https://archive.org/details/one",
        "format": "pdf",
        "bias_type": "british_legal",
        "downloaded": "true",
        "notes": "keep",
    }
    assert updated_rows[1] == rows[1]


def test_update_registry_raises_key_error_for_unknown_doc_id(file_store_paths) -> None:
    registry_path = file_store_paths["registry"]
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["doc_id", "source"])
        writer.writeheader()
        writer.writerow({"doc_id": "known_doc", "source": "internet_archive"})

    with pytest.raises(KeyError, match="unknown_doc"):
        file_store.update_registry("unknown_doc", {"source": "manual"})


def test_get_raw_format_returns_correct_extension(file_store_paths) -> None:
    doc_id = "ia_trial_19451107_009"
    path = file_store_paths["raw"] / "internet_archive" / f"{doc_id}.jpg"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"image")

    assert file_store.get_raw_format(doc_id) == "jpg"
