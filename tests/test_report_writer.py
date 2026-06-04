from __future__ import annotations

import json

from src.utils import report_writer
from src.utils.report_writer import DocResult, PipelineReport


def _report() -> PipelineReport:
    return PipelineReport(
        run_id="12345678-1234-5678-1234-567812345678",
        started_at="2026-06-02T07:30:00+00:00",
        completed_at="2026-06-02T07:31:00+00:00",
        total_docs=1,
        failed_count=1,
        results=[
            DocResult(
                doc_id="ia_ina_00000000_052",
                status="failed",
                failed_step="lang_detect",
                error="metadata missing",
            )
        ],
    )


def test_save_report_creates_timestamped_file_and_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(report_writer, "REPORTS_DIR", tmp_path)

    path = report_writer.save_report(_report())

    assert path == tmp_path / "pipeline_run_20260602_12345678.json"
    assert path.exists()
    assert (tmp_path / "latest.json").exists()
    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(
        (tmp_path / "latest.json").read_text(encoding="utf-8")
    )


def test_load_report_round_trips_correctly(tmp_path, monkeypatch):
    monkeypatch.setattr(report_writer, "REPORTS_DIR", tmp_path)
    original = _report()

    path = report_writer.save_report(original)
    loaded = report_writer.load_report(path)

    assert loaded == original


def test_failed_doc_id_appears_with_error(tmp_path, monkeypatch):
    monkeypatch.setattr(report_writer, "REPORTS_DIR", tmp_path)

    path = report_writer.save_report(_report())
    loaded = report_writer.load_report(path)

    assert loaded.results[0].doc_id == "ia_ina_00000000_052"
    assert loaded.results[0].status == "failed"
    assert loaded.results[0].error is not None
