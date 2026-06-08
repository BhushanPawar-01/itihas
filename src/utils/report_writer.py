from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from config.settings import REPORTS_DIR
from src.utils.logger import get_logger


log = get_logger(__name__)


@dataclass
class DocResult:
    doc_id: str
    status: str
    skipped_reason: Optional[str] = None
    failed_step: Optional[str] = None
    error: Optional[str] = None
    chunk_count: int = 0
    language_detected: Optional[str] = None
    bias_type: Optional[str] = None
    needs_ocr: bool = False
    duration_seconds: float = 0.0


@dataclass
class PipelineReport:
    run_id: str
    started_at: str
    completed_at: str
    total_docs: int = 0
    success_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    results: list[DocResult] = field(default_factory=list)


def _report_filename(report: PipelineReport) -> str:
    started_at_date = report.started_at[:10].replace("-", "")
    return f"pipeline_run_{started_at_date}_{report.run_id[:8]}.json"


def save_report(report: PipelineReport) -> Path:
    log.info("Saving pipeline report: run_id=%s", report.run_id)
    path = REPORTS_DIR / _report_filename(report)
    latest_path = REPORTS_DIR / "latest.json"
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(report), ensure_ascii=False, indent=2)
        path.write_text(payload, encoding="utf-8")
        latest_path.write_text(payload, encoding="utf-8")
        log.info("Saved pipeline report: run_id=%s path=%s latest_path=%s", report.run_id, path, latest_path)
        return path
    except Exception as exc:
        log.error("Failed to save pipeline report: run_id=%s path=%s error=%s", report.run_id, path, exc)
        raise


def load_report(path: Path) -> PipelineReport:
    log.info("Loading pipeline report: path=%s", path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        results = [DocResult(**result) for result in data.get("results", [])]
        report = PipelineReport(
            run_id=data["run_id"],
            started_at=data["started_at"],
            completed_at=data["completed_at"],
            total_docs=int(data.get("total_docs", 0)),
            success_count=int(data.get("success_count", 0)),
            skipped_count=int(data.get("skipped_count", 0)),
            failed_count=int(data.get("failed_count", 0)),
            results=results,
        )
        log.info("Loaded pipeline report: path=%s run_id=%s results=%d", path, report.run_id, len(results))
        return report
    except Exception as exc:
        log.error("Failed to load pipeline report: path=%s error=%s", path, exc)
        raise
