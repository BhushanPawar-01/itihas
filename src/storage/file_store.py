from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from config.settings import METADATA_DIR, PROCESSED_CHUNKS_DIR, PROCESSED_TEXT_DIR, RAW_DIR, REGISTRY_PATH
from src.utils.logger import get_logger


log = get_logger(__name__)

RAW_EXTENSION_PRIORITY = [".txt", ".pdf", ".jpg", ".jpeg", ".png", ".md"]


def _find_raw_path(doc_id: str) -> Path:
    try:
        matches = sorted(RAW_DIR.rglob(f"{doc_id}.*"))
        if not matches:
            raise FileNotFoundError(f"Raw file not found for doc_id={doc_id}")

        for extension in RAW_EXTENSION_PRIORITY:
            for path in matches:
                if path.suffix.lower() == extension:
                    return path
        return matches[0]
    except Exception as exc:
        log.error("Failed to find raw file: doc_id=%s error=%s", doc_id, exc)
        raise


def read_raw(doc_id: str) -> tuple[bytes, str]:
    log.info("Reading raw file: doc_id=%s", doc_id)
    path = _find_raw_path(doc_id)
    try:
        data = path.read_bytes()
        extension = path.suffix.lower().lstrip(".")
        log.info("Read raw file: doc_id=%s path=%s size=%d", doc_id, path, len(data))
        return data, extension
    except Exception as exc:
        log.error("Failed to read raw file: doc_id=%s path=%s error=%s", doc_id, path, exc)
        raise


def get_raw_format(doc_id: str) -> str:
    log.info("Getting raw format: doc_id=%s", doc_id)
    path = _find_raw_path(doc_id)
    try:
        extension = path.suffix.lower().lstrip(".")
        log.info("Got raw format: doc_id=%s path=%s extension=%s", doc_id, path, extension)
        return extension
    except Exception as exc:
        log.error("Failed to get raw format: doc_id=%s path=%s error=%s", doc_id, path, exc)
        raise


def read_processed_text(doc_id: str) -> str:
    log.info("Reading processed text: doc_id=%s", doc_id)
    path = PROCESSED_TEXT_DIR / f"{doc_id}.txt"
    try:
        text = path.read_text(encoding="utf-8")
        log.info("Read processed text: doc_id=%s path=%s size=%d", doc_id, path, len(text))
        return text
    except Exception as exc:
        log.error("Failed to read processed text: doc_id=%s path=%s error=%s", doc_id, path, exc)
        raise


def write_processed_text(doc_id: str, text: str) -> Path:
    log.info("Writing processed text: doc_id=%s size=%d", doc_id, len(text))
    path = PROCESSED_TEXT_DIR / f"{doc_id}.txt"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        log.info("Wrote processed text: doc_id=%s path=%s size=%d", doc_id, path, len(text))
        return path
    except Exception as exc:
        log.error("Failed to write processed text: doc_id=%s path=%s error=%s", doc_id, path, exc)
        raise


def write_processed_chunks(doc_id: str, chunks: list[dict]) -> Path:
    log.info("Writing processed chunks: doc_id=%s count=%d", doc_id, len(chunks))
    path = PROCESSED_CHUNKS_DIR / f"{doc_id}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Wrote processed chunks: doc_id=%s path=%s count=%d", doc_id, path, len(chunks))
        return path
    except Exception as exc:
        log.error("Failed to write processed chunks: doc_id=%s path=%s error=%s", doc_id, path, exc)
        raise


def read_processed_chunks(doc_id: str) -> list[dict]:
    log.info("Reading processed chunks: doc_id=%s", doc_id)
    path = PROCESSED_CHUNKS_DIR / f"{doc_id}.json"
    try:
        chunks = json.loads(path.read_text(encoding="utf-8"))
        log.info("Read processed chunks: doc_id=%s path=%s count=%d", doc_id, path, len(chunks))
        return chunks
    except Exception as exc:
        log.error("Failed to read processed chunks: doc_id=%s path=%s error=%s", doc_id, path, exc)
        raise


def read_metadata(doc_id: str) -> dict:
    log.info("Reading metadata: doc_id=%s", doc_id)
    path = METADATA_DIR / f"{doc_id}.json"
    try:
        if not path.exists():
            log.warning("Metadata file not found, returning empty dictionary: path=%s", path)
            return {}
        meta = json.loads(path.read_text(encoding="utf-8"))
        log.info("Read metadata: doc_id=%s path=%s keys=%d", doc_id, path, len(meta))
        return meta
    except Exception as exc:
        log.error("Failed to read metadata: doc_id=%s path=%s error=%s", doc_id, path, exc)
        raise


def write_metadata(doc_id: str, meta: dict) -> Path:
    log.info("Writing metadata: doc_id=%s keys=%d", doc_id, len(meta))
    path = METADATA_DIR / f"{doc_id}.json"
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
        log.info("Wrote metadata: doc_id=%s path=%s keys=%d", doc_id, path, len(meta))
        return path
    except Exception as exc:
        log.error("Failed to write metadata: doc_id=%s path=%s tmp_path=%s error=%s", doc_id, path, tmp_path, exc)
        raise


def update_registry(doc_id: str, updates: dict) -> None:
    log.info("Updating registry: doc_id=%s update_keys=%s", doc_id, sorted(updates))
    try:
        with open(REGISTRY_PATH, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise KeyError(f"Registry has no header: {REGISTRY_PATH}")
            rows: list[dict[str, Any]] = list(reader)

        found = False
        for row in rows:
            if row.get("doc_id") == doc_id:
                row.update(updates)
                found = True
                break

        if not found:
            raise KeyError(f"Registry row not found for doc_id={doc_id}")

        with open(REGISTRY_PATH, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        log.info("Updated registry: doc_id=%s path=%s rows=%d", doc_id, REGISTRY_PATH, len(rows))
    except Exception as exc:
        log.error("Failed to update registry: doc_id=%s path=%s error=%s", doc_id, REGISTRY_PATH, exc)
        raise
