from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.settings import METADATA_DIR
from src.storage.file_store import read_metadata, write_metadata
from src.utils.logger import get_logger


log = get_logger(__name__)

URDU_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")
INA_TESTIMONY_KEYWORDS = ["testimony", "statement", "deposition", "account"]
BRITISH_LEGAL_KEYWORDS = ["trial", "proceedings", "court", "charge"]
BRITISH_MILITARY_KEYWORDS = ["dispatch", "memo", "report", "intelligence"]


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _combined_text(value: Any) -> str:
    return " ".join(_string_values(value))


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def infer_bias_type(meta: dict) -> str:
    collection_values = _string_values(meta.get("collection"))
    subject_text = _combined_text(meta.get("subject"))
    title = str(meta.get("title", "")).lower()
    source = meta.get("source")
    topic = meta.get("topic")

    if any("booksbylanguage" in value.lower() for value in collection_values) and URDU_SCRIPT_RE.search(subject_text):
        return "urdu_press"

    if source == "internet_archive" and topic == "ina" and _contains_any(title, BRITISH_LEGAL_KEYWORDS):
        return "british_legal"

    if _contains_any(title, INA_TESTIMONY_KEYWORDS):
        return "ina_testimony"

    if _contains_any(title, BRITISH_MILITARY_KEYWORDS):
        return "british_military"

    if source == "semantic_scholar":
        return "academic"

    if source == "internet_archive" and topic == "press":
        return "nationalist_press"

    return "unknown"


def run_on_doc(doc_id: str) -> dict:
    log.info("Running bias tagging: doc_id=%s", doc_id)
    try:
        meta = read_metadata(doc_id)
        bias_type = infer_bias_type(meta)
        meta["bias_type"] = bias_type
        write_metadata(doc_id, meta)
        log.info("Tagged bias type: doc_id=%s bias_type=%s", doc_id, bias_type)
        return meta
    except Exception as exc:
        log.error("Failed bias tagging for document: doc_id=%s error=%s", doc_id, exc)
        raise


def _metadata_doc_id(path: Path) -> str:
    return path.stem


def _should_process(path: Path, overwrite: bool) -> bool:
    if overwrite:
        return True

    try:
        meta = read_metadata(_metadata_doc_id(path))
        return meta.get("bias_type") == "unknown"
    except Exception as exc:
        log.error("Failed to inspect metadata: path=%s error=%s", path, exc)
        raise


def run_all(overwrite: bool = False) -> list[dict]:
    log.info("Running bias tagging for all metadata: overwrite=%s", overwrite)
    try:
        metadata_paths = sorted(METADATA_DIR.glob("*.json"))
        docs_to_process = [path for path in metadata_paths if _should_process(path, overwrite)]
        updated_metas = [run_on_doc(_metadata_doc_id(path)) for path in docs_to_process]
        log.info("Completed bias tagging run: overwrite=%s count=%d", overwrite, len(updated_metas))
        return updated_metas
    except Exception as exc:
        log.error("Failed bias tagging run: overwrite=%s error=%s", overwrite, exc)
        raise


def _bias_counts(metas: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for meta in metas:
        counts[str(meta.get("bias_type", "unknown"))] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    metas = run_all(overwrite=False)
    counts = _bias_counts(metas)
    summary = ", ".join(f"{bias_type}: {count}" for bias_type, count in counts.items())
    print(summary or "unknown: 0")


if __name__ == "__main__":
    main()
