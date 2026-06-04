from __future__ import annotations

from datetime import datetime
from typing import Any


VALID_BIAS_TYPES = {
    "british_legal",
    "british_military",
    "ina_testimony",
    "nationalist_press",
    "academic",
    "urdu_press",
    "regional_press",
    "unknown",
}


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_non_empty_format(value: Any) -> bool:
    if _is_non_empty_string(value):
        return True
    if isinstance(value, list):
        return any(_is_non_empty_string(item) for item in value)
    return False


def _is_iso_datetime(value: str) -> bool:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return True


def validate_metadata(meta: dict) -> list[str]:
    errors: list[str] = []

    if not _is_non_empty_string(meta.get("doc_id")):
        errors.append("doc_id must be a non-empty string")
    if not _is_non_empty_string(meta.get("url")):
        errors.append("url must be a non-empty string")
    if meta.get("bias_type") not in VALID_BIAS_TYPES:
        errors.append("bias_type must be a known value")
    if not _is_non_empty_string(meta.get("language_original")):
        errors.append("language_original must be a non-empty string")
    if not _is_non_empty_format(meta.get("format")):
        errors.append("format must be a non-empty string or list")

    processed_at = meta.get("processed_at")
    if processed_at is not None:
        if not _is_non_empty_string(processed_at) or not _is_iso_datetime(processed_at):
            errors.append("processed_at must be None or a valid ISO datetime string")

    return errors


def validate_registry_row(row: dict) -> list[str]:
    errors: list[str] = []

    for field in ("doc_id", "source", "url", "format", "bias_type"):
        if not _is_non_empty_string(row.get(field)):
            errors.append(f"{field} must be a non-empty string")

    downloaded = row.get("downloaded")
    if isinstance(downloaded, bool):
        return errors
    if not isinstance(downloaded, str) or downloaded.lower() not in {"true", "false"}:
        errors.append("downloaded must be a boolean-like value")

    return errors
