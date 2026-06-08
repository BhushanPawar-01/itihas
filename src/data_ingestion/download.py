from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

from config.settings import RAW_DIR, REGISTRY_PATH
from src.utils.scraper import download_file, filename_from_url, slugify
from src.utils.logger import get_logger


log = get_logger("download")


def read_registry(registry_path: Path) -> tuple[list[str], list[dict]]:
    with open(registry_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames or [], list(reader)


def write_registry(registry_path: Path, columns: list[str], rows: list[dict]) -> None:
    with open(registry_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def registry_downloads(rows: list[dict], topic: str | None) -> list[dict]:
    rows = [row for row in rows if row.get("download_url") and row.get("downloaded") not in ("yes", "failed")]
    if topic:
        rows = [row for row in rows if row.get("topic") == topic]
    return rows


def target_path(row: dict, out_dir: Path) -> Path:
    topic = slugify(row.get("topic", "archive"))
    doc_id = row.get("doc_id", "record")
    # Use `doc_id` as the filename with the same extension as the original file
    orig_name = filename_from_url(row["download_url"], "")
    ext = Path(orig_name).suffix or ".bin"
    name = f"{doc_id}{ext}"
    return out_dir / "raw" / row.get("source", "source") / name


def download_registry(registry_path: Path, out_dir: Path, topic: str | None, limit: int | None, dry_run: bool) -> None:
    columns, all_rows = read_registry(registry_path)
    rows = registry_downloads(all_rows, topic)
    selected = rows[:limit] if limit else rows
    downloaded = 0
    skipped = 0
    failed = 0

    for row in selected:
        path = target_path(row, out_dir)
        if dry_run:
            log.info("[DRY-RUN] Would download %s -> %s", row["download_url"], path)
            continue
        if path.exists() and path.stat().st_size > 0:
            row["downloaded"] = "yes"
            row["notes"] = row.get("notes", "")
            skipped += 1
            write_registry(registry_path, columns, all_rows)
            log.info("Already downloaded %s", path)
            continue
        try:
            download_file(row["download_url"], path)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            failed += 1
            row["downloaded"] = "failed"
            row["notes"] = f"download failed: {exc}"
            write_registry(registry_path, columns, all_rows)
            log.warning("Download failed: url=%s error=%s", row["download_url"], exc)
            continue
        row["downloaded"] = "yes"
        row["notes"] = row.get("notes", "")
        downloaded += 1
        write_registry(registry_path, columns, all_rows)
        log.info("Downloaded %s", path)

    log.info(
        "Download summary: selected=%d downloaded=%d already_present=%d failed=%d dry_run=%s",
        len(selected),
        downloaded,
        skipped,
        failed,
        dry_run,
    )


def download_one(url: str, out_dir: Path, topic: str, dry_run: bool) -> None:
    path = out_dir / "raw" / "manual" / slugify(topic) / filename_from_url(url, "download.bin")
    if dry_run:
        log.info("[DRY-RUN] Would download %s -> %s", url, path)
        return
    try:
        download_file(url, path)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        log.warning("Download failed: url=%s error=%s", url, exc)
        return
    log.info("Downloaded %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw files from registry or a direct URL.")
    parser.add_argument("--url")
    parser.add_argument("--topic")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR.parent)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.url:
        download_one(args.url, args.out_dir, args.topic or "manual", args.dry_run)
        return
    download_registry(args.registry, args.out_dir, args.topic, args.limit, args.dry_run)


if __name__ == "__main__":
    main()
