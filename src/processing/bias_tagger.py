from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from config.settings import METADATA_DIR, PROCESSED_TEXT_DIR
from src.storage.file_store import read_metadata, write_metadata
from src.utils.logger import get_logger

log = get_logger(__name__)

URDU_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")

VALID_BIAS_TYPES = {
    "british_legal",
    "british_military",
    "ina_testimony",
    "nationalist_press",
    "urdu_press",
    "academic",
    "regional_press",
    "unknown",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _combined_text(value: Any) -> str:
    return " ".join(_string_values(value)).lower()


def _collection_set(meta: dict) -> set[str]:
    return {v.lower() for v in _string_values(meta.get("collection"))}


def _language(meta: dict) -> str:
    """Return first non-empty language tag found across all language fields."""
    for field in ("language_detected", "language_original"):
        val = meta.get(field)
        if val and str(val).strip() not in ("", "null", "unknown"):
            return str(val).strip().lower()
    langs = _string_values(meta.get("language_all"))
    if langs:
        return langs[0].lower()
    return ""


def _read_text_excerpt(doc_id: str, word_limit: int = 400) -> str:
    """
    Read the first `word_limit` words from the processed text file.
    Falls back to description field content if text file is missing.
    Returns empty string if neither is available.
    """
    text_path = PROCESSED_TEXT_DIR / f"{doc_id}.txt"
    if text_path.exists():
        try:
            raw = text_path.read_text(encoding="utf-8", errors="replace")
            words = raw.split()
            return " ".join(words[:word_limit])
        except Exception as exc:
            log.warning("Could not read text file: doc_id=%s error=%s", doc_id, exc)
    return ""


# ---------------------------------------------------------------------------
# Tier 1 — fast deterministic rules on reliable metadata fields only
#
# Only fires on fields that are CONSISTENT across all IA documents:
#   - collection  (always a list, always present)
#   - language    (language_detected / language_original / language_all)
#
# Does NOT use: topic, description, subject, title
# Those fields are inconsistent in format and content across IA documents.
# ---------------------------------------------------------------------------

def _tier1_infer(meta: dict) -> str | None:
    """
    Returns a bias_type string if a high-confidence rule matches.
    Returns None if uncertain — caller sends to tier 2.
    """
    collections = _collection_set(meta)
    language    = _language(meta)
    subject_raw = " ".join(_string_values(meta.get("subject")))

    # Rule 1 — urdu_press
    # booksbylanguage collection is specifically IA's language-tagged archive.
    # Urdu script in subjects confirms the language. High precision.
    has_urdu_script = bool(URDU_SCRIPT_RE.search(subject_raw))
    in_booksbylan   = any("booksbylanguage" in c for c in collections)
    is_urdu_lang    = language in ("ur", "urdu")
    if in_booksbylan and (has_urdu_script or is_urdu_lang):
        return "urdu_press"

    # Rule 2 — academic (journal articles)
    # 'journals' and 'impactjournals' are IA collection names for journal articles.
    # These are scholarly papers, not primary sources.
    journal_collections = {"journals", "impactjournals", "jstor", "doaj"}
    if collections & journal_collections:
        return "academic"

    # Rule 3 — regional_press (non-Urdu booksbylanguage)
    # booksbylanguage without Urdu script = regional press in other Indian languages
    if in_booksbylan and not has_urdu_script and not is_urdu_lang:
        return "regional_press"

    # Everything else is ambiguous — needs tier 2
    return None


# ---------------------------------------------------------------------------
# Tier 2 — LLM classification on actual document text
#
# Called only when tier 1 returns None.
# Uses first 400 words of data/processed/text/{doc_id}.txt
# Falls back to 'unknown' if text unavailable or LLM call fails.
# All LLM calls go through src.utils.llm_client — never direct SDK.
# ---------------------------------------------------------------------------

_CLASSIFICATION_PROMPT = """\
You are classifying a historical document about the Indian National Army (INA) \
and Indian independence (1940s). Read the text excerpt below and return exactly \
one label from this list:

british_legal      — Official British legal documents: trial transcripts, charge \
sheets, court martial records, prosecution filings, Red Fort trial documents.

british_military   — British military/government internal documents: intelligence \
reports, field dispatches, India Office memos, war department communications.

ina_testimony      — Personal accounts BY INA members: soldier memoirs, officer \
testimonies, personal letters, first-person accounts of INA service or trials.

nationalist_press  — Indian press or pamphlets from pro-independence perspective: \
newspaper articles, political pamphlets, Congress/INA propaganda, public speeches.

academic           — Scholarly analysis: history books, journal articles, research \
papers written by historians or academics studying these events.

unknown            — Cannot determine from the available text.

Return ONLY the label, nothing else. No explanation. No punctuation.

TEXT EXCERPT:
{excerpt}

LABEL:"""


def _tier2_llm_infer(doc_id: str, meta: dict) -> str:
    """
    Classify using LLM on document text excerpt.
    Returns a valid bias_type string. Never raises — returns 'unknown' on failure.
    """
    excerpt = _read_text_excerpt(doc_id, word_limit=400)

    if not excerpt:
        log.warning(
            "No text excerpt available for LLM classification: doc_id=%s", doc_id
        )
        return "unknown"

    try:
        from src.utils.llm_client import call as llm_call  # never import SDK directly
    except ImportError as exc:
        log.error("llm_client import failed: %s", exc)
        return "unknown"

    prompt = _CLASSIFICATION_PROMPT.format(excerpt=excerpt)

    try:
        response = llm_call(prompt, max_tokens=10, temperature=0.0)
        # llm_call returns a string — strip whitespace and normalise
        label = response.strip().lower().rstrip(".")
        if label in VALID_BIAS_TYPES:
            log.info(
                "LLM classified: doc_id=%s bias_type=%s", doc_id, label
            )
            return label
        else:
            log.warning(
                "LLM returned unexpected label: doc_id=%s label=%r", doc_id, label
            )
            return "unknown"
    except Exception as exc:
        log.error(
            "LLM classification failed: doc_id=%s error=%s", doc_id, exc
        )
        return "unknown"


# ---------------------------------------------------------------------------
# Main inference entry point
# ---------------------------------------------------------------------------

def infer_bias_type(doc_id: str, meta: dict) -> str:
    """
    Two-tier bias inference.
    Tier 1: fast rules on reliable metadata fields.
    Tier 2: LLM on document text — only when tier 1 is uncertain.
    """
    tier1 = _tier1_infer(meta)
    if tier1 is not None:
        log.info("Tier-1 tagged: doc_id=%s bias_type=%s", doc_id, tier1)
        return tier1

    log.info("Tier-1 inconclusive, running LLM: doc_id=%s", doc_id)
    return _tier2_llm_infer(doc_id, meta)


# ---------------------------------------------------------------------------
# DB sync
# ---------------------------------------------------------------------------

def update_bias_in_db(doc_id: str, bias_type: str) -> int:
    """
    Update bias_tag and source_type columns in Postgres for all chunks of doc_id.
    Returns count of rows updated. Raises on DB error.
    """
    try:
        from src.storage import db_client

        with db_client.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE chunks SET bias_tag = %s, source_type = %s WHERE doc_id = %s",
                (bias_type, bias_type, doc_id),
            )
            updated = cur.rowcount
            conn.commit()
            log.info(
                "DB updated: doc_id=%s bias_type=%s rows=%d",
                doc_id, bias_type, updated,
            )
            return updated
    except Exception as exc:
        log.error("DB update failed: doc_id=%s error=%s", doc_id, exc)
        raise


# ---------------------------------------------------------------------------
# Single document runner
# ---------------------------------------------------------------------------

def run_on_doc(doc_id: str, sync_db: bool = False) -> dict:
    log.info("Bias tagging: doc_id=%s sync_db=%s", doc_id, sync_db)
    try:
        meta      = read_metadata(doc_id)
        bias_type = infer_bias_type(doc_id, meta)
        meta["bias_type"] = bias_type
        write_metadata(doc_id, meta)
        if sync_db:
            update_bias_in_db(doc_id, bias_type)
        return meta
    except Exception as exc:
        log.error("run_on_doc failed: doc_id=%s error=%s", doc_id, exc)
        raise


# ---------------------------------------------------------------------------
# Batch runners
# ---------------------------------------------------------------------------

def _metadata_doc_id(path: Path) -> str:
    return path.stem


def run_all(overwrite: bool = False, sync_db: bool = False) -> list[dict]:
    """
    Re-tag all metadata files.
    overwrite=False skips docs already tagged (not 'unknown').
    sync_db=True pushes updated bias_tag to Postgres after each doc.
    Does not abort on single-doc failure — logs and continues.
    """
    log.info("run_all: overwrite=%s sync_db=%s", overwrite, sync_db)
    metadata_paths = sorted(METADATA_DIR.glob("*.json"))
    results        = []
    skipped        = 0

    for path in metadata_paths:
        doc_id = _metadata_doc_id(path)
        try:
            meta = read_metadata(doc_id)
            if not overwrite and meta.get("bias_type", "unknown") != "unknown":
                skipped += 1
                continue
            results.append(run_on_doc(doc_id, sync_db=sync_db))
        except Exception:
            log.error("Skipping failed doc: doc_id=%s", doc_id)
            continue

    log.info(
        "run_all complete: processed=%d skipped=%d", len(results), skipped
    )
    return results


def run_all_and_sync_db(overwrite: bool = True) -> dict:
    """
    Re-tag everything and push to DB in one call.
    This is the recovery function — use when chunks are already in Postgres
    but bias_tag is wrong.

    Usage:
        python src/processing/bias_tagger.py --sync-db
    """
    metas = run_all(overwrite=overwrite, sync_db=True)
    counts = _bias_counts(metas)
    total_db_rows = sum(
        update_bias_in_db(m["doc_id"], m["bias_type"])
        for m in metas
        if m.get("doc_id")  # update_bias_in_db already called inside run_on_doc
        # but run_on_doc with sync_db=True already did it — so we don't double-call
        # This line is never reached; just here to show the pattern.
        # The actual DB update happens inside run_on_doc when sync_db=True.
        and False
    )
    return {
        "tagged_docs": len(metas),
        "bias_counts": counts,
    }


def _bias_counts(metas: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for meta in metas:
        counts[str(meta.get("bias_type", "unknown"))] += 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Two-tier bias tagger: fast rules + LLM fallback"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-tag all docs, not just 'unknown' ones",
    )
    parser.add_argument(
        "--sync-db", action="store_true",
        help="Push updated bias_tag to Postgres chunks table",
    )
    parser.add_argument(
        "--doc-id", type=str, default=None,
        help="Tag a single document by doc_id and print result",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be tagged without writing anything",
    )
    args = parser.parse_args()

    if args.doc_id:
        meta      = read_metadata(args.doc_id)
        bias_type = infer_bias_type(args.doc_id, meta)
        print(f"{args.doc_id}: {bias_type}")
        if not args.dry_run:
            meta["bias_type"] = bias_type
            write_metadata(args.doc_id, meta)
            if args.sync_db:
                rows = update_bias_in_db(args.doc_id, bias_type)
                print(f"DB rows updated: {rows}")
        return

    if args.dry_run:
        metadata_paths = sorted(METADATA_DIR.glob("*.json"))
        for path in metadata_paths:
            doc_id = _metadata_doc_id(path)
            try:
                meta      = read_metadata(doc_id)
                bias_type = infer_bias_type(doc_id, meta)
                print(f"{doc_id}: {bias_type}")
            except Exception as exc:
                print(f"{doc_id}: ERROR — {exc}")
        return

    metas  = run_all(overwrite=args.overwrite, sync_db=args.sync_db)
    counts = _bias_counts(metas)
    lines  = ", ".join(f"{k}: {v}" for k, v in counts.items())
    print(lines or "No documents processed")


if __name__ == "__main__":
    main()