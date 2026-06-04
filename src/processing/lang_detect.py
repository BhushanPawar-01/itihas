from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.settings import METADATA_DIR, PROCESSED_TEXT_DIR
from src.storage.file_store import read_metadata, read_processed_text, read_raw, write_metadata
from src.utils.logger import get_logger


log = get_logger(__name__)

CONFIDENCE_THRESHOLD = 0.7


try:
    from lingua import Language, LanguageDetectorBuilder
except ImportError as exc:
    Language = None
    LanguageDetectorBuilder = None
    _LINGUA_IMPORT_ERROR = exc
else:
    _LINGUA_IMPORT_ERROR = None


INDIAN_SCHEDULED_LANGUAGE_CODES = {
    "ASSAMESE": "as",
    "BENGALI": "bn",
    "BODO": "brx",
    "DOGRI": "doi",
    "GUJARATI": "gu",
    "HINDI": "hi",
    "KANNADA": "kn",
    "KASHMIRI": "ks",
    "KONKANI": "kok",
    "MAITHILI": "mai",
    "MALAYALAM": "ml",
    "MANIPURI": "mni",
    "MARATHI": "mr",
    "NEPALI": "ne",
    "ODIA": "or",
    "PUNJABI": "pa",
    "SANSKRIT": "sa",
    "SANTALI": "sat",
    "SINDHI": "sd",
    "TAMIL": "ta",
    "TELUGU": "te",
    "URDU": "ur",
}

HISTORICAL_CONTEXT_LANGUAGE_CODES = {
    "ENGLISH": "en",
    "ARABIC": "ar",
    "PERSIAN": "fa",
}

LANGUAGE_CODES = {
    **INDIAN_SCHEDULED_LANGUAGE_CODES,
    **HISTORICAL_CONTEXT_LANGUAGE_CODES,
}

LINGUA_LANGUAGE_NAMES = [
    *INDIAN_SCHEDULED_LANGUAGE_CODES,
    *HISTORICAL_CONTEXT_LANGUAGE_CODES,
]

DETECTOR = None


def _get_detector() -> Any:
    global DETECTOR

    if _LINGUA_IMPORT_ERROR is not None:
        raise ImportError(
            "lingua-language-detector is required. Install it with: pip install lingua-language-detector"
        ) from _LINGUA_IMPORT_ERROR

    if DETECTOR is None:
        languages = []
        unsupported = []
        for language_name in LINGUA_LANGUAGE_NAMES:
            language = getattr(Language, language_name, None)
            if language is None:
                unsupported.append(language_name)
                continue
            languages.append(language)

        if unsupported:
            log.info("Lingua unsupported languages skipped: languages=%s", sorted(unsupported))

        DETECTOR = LanguageDetectorBuilder.from_languages(*languages).build()

    return DETECTOR


def _split_paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]


def _detect_paragraph_language(paragraph: str) -> str | None:
    detector = _get_detector()
    confidence_values = detector.compute_language_confidence_values(paragraph)
    if not confidence_values:
        return None

    top_confidence = confidence_values[0]
    if top_confidence.value < CONFIDENCE_THRESHOLD:
        return None

    return LANGUAGE_CODES.get(top_confidence.language.name)


def detect_language(text: str) -> str:
    try:
        counts: Counter[str] = Counter()
        for paragraph in _split_paragraphs(text):
            language = _detect_paragraph_language(paragraph)
            if language is not None:
                counts[language] += 1

        if not counts:
            return "unknown"
        return counts.most_common(1)[0][0]
    except Exception as exc:
        log.error("Failed to detect language: size=%d error=%s", len(text), exc)
        raise


def _read_doc_text(doc_id: str) -> str:
    processed_path = PROCESSED_TEXT_DIR / f"{doc_id}.txt"
    try:
        if processed_path.exists():
            return read_processed_text(doc_id)

        raw_data, _ = read_raw(doc_id)
        return raw_data.decode("utf-8", errors="replace")
    except Exception as exc:
        log.error("Failed to read document text: doc_id=%s error=%s", doc_id, exc)
        raise


def _process_doc(doc_id: str, dry_run: bool) -> dict:
    log.info("Running language detection: doc_id=%s dry_run=%s", doc_id, dry_run)
    try:
        meta = read_metadata(doc_id)
        text = _read_doc_text(doc_id)
        language = detect_language(text)
        meta["language_detected"] = language

        if dry_run:
            print(f"{doc_id}: {language}")
        else:
            write_metadata(doc_id, meta)

        log.info("Detected language: doc_id=%s language=%s dry_run=%s", doc_id, language, dry_run)
        return meta
    except Exception as exc:
        log.error("Failed language detection for document: doc_id=%s dry_run=%s error=%s", doc_id, dry_run, exc)
        raise


def run_on_doc(doc_id: str) -> dict:
    return _process_doc(doc_id, dry_run=False)


def _metadata_doc_id(path: Path) -> str:
    return path.stem


def _needs_language_detection(path: Path) -> bool:
    try:
        meta = read_metadata(_metadata_doc_id(path))
        return meta.get("language_detected") in {None, "unknown"}
    except Exception as exc:
        log.error("Failed to inspect metadata: path=%s error=%s", path, exc)
        raise


def run_all(dry_run: bool = False) -> list[dict]:
    log.info("Running language detection for all metadata: dry_run=%s", dry_run)
    try:
        metadata_paths = sorted(METADATA_DIR.glob("*.json"))
        docs_to_process = [path for path in metadata_paths if _needs_language_detection(path)]
        updated_metas = [_process_doc(_metadata_doc_id(path), dry_run=dry_run) for path in docs_to_process]
        log.info("Completed language detection run: dry_run=%s count=%d", dry_run, len(updated_metas))
        return updated_metas
    except Exception as exc:
        log.error("Failed language detection run: dry_run=%s error=%s", dry_run, exc)
        raise


def _language_counts(metas: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for meta in metas:
        counts[str(meta.get("language_detected", "unknown"))] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect document languages.")
    parser.add_argument("--dry-run", action="store_true", help="Print detections without writing metadata.")
    args = parser.parse_args()

    metas = run_all(dry_run=args.dry_run)
    print(f"Processed {len(metas)} docs. Languages found: {_language_counts(metas)}")


if __name__ == "__main__":
    main()
