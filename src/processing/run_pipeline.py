from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.settings import REGISTRY_PATH
from src.processing.bias_tagger import run_on_doc as run_bias_tagger
from src.processing.chunker import chunk_document
from src.processing.lang_detect import run_on_doc as run_lang_detect
from src.storage.file_store import read_metadata
from src.utils.logger import get_logger
from src.utils.report_writer import DocResult, PipelineReport, save_report


log = get_logger(__name__)

PipelineStep = Callable[[str], Any]


@dataclass
class PipelineSummary:
    success: int = 0
    skipped: int = 0
    failed: int = 0
    failed_doc_ids: list[str] = field(default_factory=list)
    report_path: Path | None = None


PIPELINE_STEPS: list[tuple[str, PipelineStep]] = [
    ("lang_detect", run_lang_detect),
    ("bias_tagger", run_bias_tagger),
    ("chunker", chunk_document),
]


def _read_registry_doc_ids() -> list[str]:
    log.info("Reading registry doc IDs: path=%s", REGISTRY_PATH)
    try:
        with open(REGISTRY_PATH, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            doc_ids = [str(row["doc_id"]) for row in reader if row.get("doc_id")]
        log.info("Read registry doc IDs: path=%s count=%d", REGISTRY_PATH, len(doc_ids))
        return doc_ids
    except Exception as exc:
        log.error("Failed to read registry doc IDs: path=%s error=%s", REGISTRY_PATH, exc)
        raise


def _is_fully_processed(doc_id: str) -> bool:
    meta = read_metadata(doc_id)
    return int(meta.get("chunk_count", 0) or 0) > 0


def _run_step(doc_id: str, step_name: str, step: PipelineStep) -> None:
    start = perf_counter()
    log.info("Starting pipeline step: doc_id=%s step=%s", doc_id, step_name)
    try:
        step(doc_id)
    except Exception as exc:
        duration = perf_counter() - start
        log.error(
            "Failed pipeline step: doc_id=%s step=%s duration=%.3fs error=%s",
            doc_id,
            step_name,
            duration,
            exc,
        )
        raise

    duration = perf_counter() - start
    log.info("Completed pipeline step: doc_id=%s step=%s duration=%.3fs", doc_id, step_name, duration)


def _select_doc_ids(registry_doc_ids: list[str], doc_id: str | None) -> list[str]:
    if doc_id is None:
        return registry_doc_ids
    if doc_id not in registry_doc_ids:
        log.error("Requested doc_id not found in registry: doc_id=%s", doc_id)
        return []
    return [doc_id]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metadata_fields(doc_id: str) -> dict:
    try:
        meta = read_metadata(doc_id)
    except Exception as exc:
        log.error("Failed to read metadata for report: doc_id=%s error=%s", doc_id, exc)
        return {}

    return {
        "chunk_count": int(meta.get("chunk_count", 0) or 0),
        "language_detected": meta.get("language_detected"),
        "bias_type": meta.get("bias_type"),
    }


def _finalize_report(report: PipelineReport) -> Path:
    report.completed_at = _utcnow()
    report.total_docs = len(report.results)
    report.success_count = sum(1 for result in report.results if result.status == "success")
    report.skipped_count = sum(1 for result in report.results if result.status == "skipped")
    report.failed_count = sum(1 for result in report.results if result.status == "failed")
    return save_report(report)


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def run_pipeline(force: bool = False, doc_id: str | None = None) -> PipelineSummary:
    summary = PipelineSummary()
    report = PipelineReport(run_id=str(uuid4()), started_at=_utcnow(), completed_at="")
    registry_doc_ids = _read_registry_doc_ids()
    doc_ids = _select_doc_ids(registry_doc_ids, doc_id)

    if doc_id is not None and not doc_ids:
        summary.failed = 1
        summary.failed_doc_ids.append(doc_id)
        report.results.append(
            DocResult(
                doc_id=doc_id,
                status="failed",
                error="doc_id not found in registry",
            )
        )
        summary.report_path = _finalize_report(report)
        return summary

    for current_doc_id in doc_ids:
        doc_start = perf_counter()
        failed_step = None
        try:
            if not force and _is_fully_processed(current_doc_id):
                log.info("Skipping fully processed document: doc_id=%s", current_doc_id)
                summary.skipped += 1
                report.results.append(
                    DocResult(
                        doc_id=current_doc_id,
                        status="skipped",
                        skipped_reason="already_processed",
                        duration_seconds=perf_counter() - doc_start,
                        **_metadata_fields(current_doc_id),
                    )
                )
                continue

            for step_name, step in PIPELINE_STEPS:
                failed_step = step_name
                _run_step(current_doc_id, step_name, step)
                failed_step = None

            summary.success += 1
            report.results.append(
                DocResult(
                    doc_id=current_doc_id,
                    status="success",
                    duration_seconds=perf_counter() - doc_start,
                    **_metadata_fields(current_doc_id),
                )
            )
            log.info("Completed document pipeline: doc_id=%s", current_doc_id)
        except Exception as exc:
            summary.failed += 1
            summary.failed_doc_ids.append(current_doc_id)
            report.results.append(
                DocResult(
                    doc_id=current_doc_id,
                    status="failed",
                    failed_step=failed_step,
                    error=str(exc),
                    duration_seconds=perf_counter() - doc_start,
                    **_metadata_fields(current_doc_id),
                )
            )
            log.error("Failed document pipeline: doc_id=%s error=%s", current_doc_id, exc)

    summary.report_path = _finalize_report(report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run processing pipeline for registered documents.")
    parser.add_argument("--force", action="store_true", help="Reprocess documents even when chunk_count is set.")
    parser.add_argument("--doc-id", help="Process one registered document ID.")
    args = parser.parse_args()

    summary = run_pipeline(force=args.force, doc_id=args.doc_id)
    print(f"Pipeline complete. Success: {summary.success} | Skipped: {summary.skipped} | Failed: {summary.failed}")
    if summary.failed_doc_ids:
        print("Failed doc_ids:")
        for failed_doc_id in summary.failed_doc_ids:
            print(failed_doc_id)
    if summary.report_path is not None:
        print(f"Report saved: {_display_path(summary.report_path)}")


if __name__ == "__main__":
    main()
