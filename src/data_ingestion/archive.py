from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config.settings import REGISTRY_PATH
from src.utils.scraper import append_csv, json_get, make_doc_id, normalise_list, slugify
from src.utils.logger import get_logger


IA_SEARCH_URL = "https://archive.org/advancedsearch.php"
IA_META_URL = "https://archive.org/metadata"
IA_DETAILS_URL = "https://archive.org/details"
IA_DOWNLOAD_URL = "https://archive.org/download"

PIPELINE_VERSION = "0.3"
TEXT_QUERY = "mediatype:texts"
TEXT_EXTENSIONS = {".txt", ".md"}
ALLOWED_EXTENSIONS = TEXT_EXTENSIONS | {".pdf", ".djvu", ".epub", ".doc", ".docx", ".rtf", ".chm"}

SEARCH_FIELDS = [
    "identifier",
    "title",
    "creator",
    "date",
    "description",
    "subject",
    "language",
    "mediatype",
    "format",
    "collection",
    "licenseurl",
]

REGISTRY_COLUMNS = [
    "doc_id",
    "identifier",
    "source",
    "topic",
    "query",
    "url",
    "download_url",
    "title",
    "creator",
    "date",
    "language",
    "format",
    "mediatype",
    "subject",
    "bias_type",
    "access_method",
    "downloaded",
    "ocr_applied",
    "translated",
    "notes",
    "scraped_at",
]

log = get_logger("archive")


def archive_query(query: str) -> str:
    if any(token in query for token in ['"', " AND ", " OR ", "mediatype:", "title:", "subject:"]):
        return f"({query}) AND {TEXT_QUERY}"
    return f'("{query}") AND {TEXT_QUERY}'


def search_archive(query: str, rows: int, page: int) -> tuple[list[dict], int]:
    q = archive_query(query)
    params = {
        "q": q,
        "fl[]": SEARCH_FIELDS,
        "rows": rows,
        "page": page,
        "output": "json",
        "sort[]": "date desc",
    }
    log.info("Searching IA: q=%r rows=%d page=%d", q, rows, page)
    response = json_get(IA_SEARCH_URL, params).get("response", {})
    docs = response.get("docs", [])
    total = int(response.get("numFound", 0))
    log.info("Results found: total=%d page_count=%d", total, len(docs))
    return docs, total


def fetch_files(identifier: str) -> list[dict]:
    try:
        files = json_get(f"{IA_META_URL}/{identifier}").get("files", [])
    except Exception as exc:
        log.warning("Could not fetch item metadata: identifier=%s error=%s", identifier, exc)
        return []

    kept = []
    for file_item in files:
        name = file_item.get("name", "")
        if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        kept.append(
            {
                "name": name,
                "format": file_item.get("format", ""),
                "size": file_item.get("size"),
                "md5": file_item.get("md5"),
                "sha1": file_item.get("sha1"),
                "mtime": file_item.get("mtime"),
                "download_url": f"{IA_DOWNLOAD_URL}/{identifier}/{name}",
            }
        )
    text_files = [file_item for file_item in kept if Path(file_item["name"]).suffix.lower() in TEXT_EXTENSIONS]
    return text_files or kept


def load_existing(registry_path: Path, metadata_dir: Path) -> tuple[set[str], set[str], set[str], set[str]]:
    identifiers: set[str] = set()
    urls: set[str] = set()
    titles: set[str] = set()
    doc_ids: set[str] = set()

    if registry_path.exists():
        with open(registry_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                identifiers.add(row.get("identifier", ""))
                urls.add(row.get("url", ""))
                titles.add(row.get("title", "").strip().lower())
                doc_ids.add(row.get("doc_id", ""))

    if metadata_dir.exists():
        for path in metadata_dir.rglob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("Could not read metadata: path=%s error=%s", path, exc)
                continue
            identifiers.add(record.get("ia_identifier", ""))
            urls.add(record.get("url", ""))
            titles.add(record.get("title", "").strip().lower())
            doc_ids.add(record.get("doc_id", ""))

    return identifiers - {""}, urls - {""}, titles - {""}, doc_ids - {""}


def already_saved(item: dict, identifiers: set[str], urls: set[str], titles: set[str]) -> bool:
    identifier = item.get("identifier", "")
    url = f"{IA_DETAILS_URL}/{identifier}" if identifier else ""
    title = item.get("title", "").strip().lower()
    return identifier in identifiers or url in urls or title in titles


def next_sequence(topic: str, doc_ids: set[str]) -> int:
    prefix = f"ia_{topic}_"
    numbers = []
    for doc_id in doc_ids:
        if doc_id.startswith(prefix):
            match = re.search(r"_(\d{3})$", doc_id)
            if match:
                numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def build_record(item: dict, files: list[dict], doc_id: str, topic: str, query: str) -> dict:
    identifier = item.get("identifier", "")
    details_url = f"{IA_DETAILS_URL}/{identifier}"
    languages = normalise_list(item.get("language"))

    return {
        "doc_id": doc_id,
        "source": "internet_archive",
        "topic": topic,
        "query": query,
        "url": details_url,
        "title": item.get("title", ""),
        "date": item.get("date", ""),
        "language_original": languages[0] if languages else "unknown",
        "language_detected": None,
        "format": normalise_list(item.get("format")),
        "mediatype": item.get("mediatype", ""),
        "bias_type": "unknown",
        "access_method": "pipeline",
        "ocr_applied": False,
        "translated": False,
        "chunk_count": 0,
        "pipeline_version": PIPELINE_VERSION,
        "processed_at": None,
        "ia_identifier": identifier,
        "creator": normalise_list(item.get("creator")),
        "subject": normalise_list(item.get("subject")),
        "language_all": languages,
        "collection": normalise_list(item.get("collection")),
        "description": item.get("description", ""),
        "licenseurl": item.get("licenseurl", ""),
        "source_url": details_url,
        "metadata_url": f"{IA_META_URL}/{identifier}",
        "download_base_url": f"{IA_DOWNLOAD_URL}/{identifier}",
        "file_urls": [file_item["download_url"] for file_item in files],
        "files": files,
        "file_count": len(files),
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "notes": "",
    }


def registry_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        files = record.get("files") or [{"download_url": "", "format": ""}]
        for file_item in files:
            rows.append(
                {
                    "doc_id": record["doc_id"],
                    "identifier": record["ia_identifier"],
                    "source": record["source"],
                    "topic": record["topic"],
                    "query": record["query"],
                    "url": record["url"],
                    "download_url": file_item.get("download_url", ""),
                    "title": record["title"],
                    "creator": "; ".join(record.get("creator", [])),
                    "date": record.get("date", ""),
                    "language": record.get("language_original", ""),
                    "format": file_item.get("format", ""),
                    "mediatype": record.get("mediatype", ""),
                    "subject": "; ".join(record.get("subject", [])),
                    "bias_type": record.get("bias_type", ""),
                    "access_method": "pipeline",
                    "downloaded": "no",
                    "ocr_applied": "no",
                    "translated": "no",
                    "notes": "" if record.get("files") else "no downloadable text files found",
                    "scraped_at": record["scraped_at"],
                }
            )
    return rows


def run(query: str, topic: str | None, rows: int, max_pages: int, out_dir: Path, dry_run: bool, delay: float) -> None:
    topic_slug = slugify(topic or query)
    metadata_root = out_dir / "metadata" / "internet_archive"
    metadata_dir = metadata_root / topic_slug
    registry_path = out_dir / "registry.csv"
    identifiers, urls, titles, doc_ids = load_existing(registry_path, metadata_root)
    records: list[dict] = []
    seq = next_sequence(topic_slug, doc_ids)
    skipped = 0

    if not dry_run:
        metadata_dir.mkdir(parents=True, exist_ok=True)

    for page in range(1, max_pages + 1):
        docs, total = search_archive(query, rows, page)
        if not docs:
            break

        for item in docs:
            identifier = item.get("identifier", "")
            if not identifier:
                continue
            if already_saved(item, identifiers, urls, titles):
                skipped += 1
                log.info("Skipping existing item: %s", identifier)
                continue

            log.info("%s file list: %s", "[DRY-RUN] Would fetch" if dry_run else "Fetching", identifier)
            files = [] if dry_run else fetch_files(identifier)
            doc_id = make_doc_id("ia", topic_slug, seq, item.get("date", ""))
            while doc_id in doc_ids:
                seq += 1
                doc_id = make_doc_id("ia", topic_slug, seq, item.get("date", ""))
            record = build_record(item, files, doc_id, topic_slug, query)
            records.append(record)
            identifiers.add(identifier)
            urls.add(record["url"])
            titles.add(record["title"].strip().lower())
            doc_ids.add(doc_id)

            path = metadata_dir / f"{doc_id}.json"
            if dry_run:
                log.info("[DRY-RUN] Would write %s", path)
            else:
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(record, fh, ensure_ascii=False, indent=2)
                log.info("Wrote %s", path)
                time.sleep(delay)
            seq += 1

        if page * rows >= total:
            break
        time.sleep(delay)

    if records and not dry_run:
        append_csv(registry_path, REGISTRY_COLUMNS, registry_rows(records))
        log.info("Registry updated: %s", registry_path)

    log.info(
        "Run summary: query=%r topic=%s items=%d skipped=%d files=%d metadata_dir=%s registry=%s dry_run=%s",
        query,
        topic_slug,
        len(records),
        skipped,
        sum(record["file_count"] for record in records),
        metadata_dir,
        registry_path,
        dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Internet Archive metadata by topic.")
    parser.add_argument("--query", "-q", required=True)
    parser.add_argument("--topic", "-t")
    parser.add_argument("--rows", "-r", type=int, default=20)
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=REGISTRY_PATH.parent)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    run(args.query, args.topic, args.rows, args.max_pages, args.out_dir, args.dry_run, args.delay)


if __name__ == "__main__":
    main()
