from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, unquote, quote, urlunparse
from urllib.request import Request, urlopen


USER_AGENT = "Itihas-research-bot/0.3 (historical data pipeline; non-commercial)"


def json_get(url: str, params: dict | None = None, retries: int = 3) -> Any:
    if params:
        url = f"{url}?{urlencode(params, doseq=True)}"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except HTTPError as exc:
            if exc.code not in (429, 503) or attempt == retries:
                raise
            time.sleep(10 * attempt)
        except (URLError, TimeoutError):
            if attempt == retries:
                raise
            time.sleep(5 * attempt)
    return {}


def download_file(url: str, path: Path, retries: int = 3) -> None:
    headers = {"User-Agent": USER_AGENT}
    path.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https"):
        path_enc = quote(parsed.path, safe="/")
        query_enc = quote(parsed.query, safe="=&/")
        url = urlunparse((parsed.scheme, parsed.netloc, path_enc, parsed.params, query_enc, parsed.fragment))

    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=60) as resp:
                with open(path, "wb") as fh:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
            return
        except HTTPError as exc:
            if exc.code in (500, 502, 503, 504) and attempt < retries:
                time.sleep(5 * attempt)
                continue
            raise
        except (URLError, TimeoutError):
            if attempt < retries:
                time.sleep(5 * attempt)
                continue
            raise


def slugify(value: str, fallback: str = "archive") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug or fallback


def normalise_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def make_doc_id(source: str, record_type: str, seq: int, date_str: str) -> str:
    date_part = "00000000"
    match = re.match(r"^(\d{1,4})(?:[-/](\d{1,2}))?(?:[-/](\d{1,2}))?", date_str.strip())
    if match:
        year = match.group(1).zfill(4)
        month = (match.group(2) or "00").zfill(2)
        day = (match.group(3) or "00").zfill(2)
        date_part = f"{year}{month}{day}"
    return f"{source}_{slugify(record_type)}_{date_part}_{seq:03d}"


def append_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    is_new = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    if not is_new:
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        if first_line != ",".join(columns):
            existing = path.read_text(encoding="utf-8")
            path.write_text(",".join(columns) + "\n" + existing, encoding="utf-8")
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def filename_from_url(url: str, fallback: str) -> str:
    name = Path(unquote(urlparse(url).path)).name
    return name or fallback
