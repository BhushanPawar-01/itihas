"""
Ingest registry.csv into the documents table on Neon Postgres.
Run once: python src/storage/ingest_registry.py

Safe to re-run — uses INSERT ... ON CONFLICT DO UPDATE (upsert).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import DB_URL, DB_SSLMODE
from src.utils.logger import get_logger
import psycopg2

log = get_logger(__name__)

REGISTRY_PATH = project_root / "data" / "registry.csv"

def ingest_registry() -> None:
    log.info("Connecting to Neon Postgres...")
    conn = psycopg2.connect(DB_URL, sslmode=DB_SSLMODE)

    try:
        with open(REGISTRY_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        log.info("Found %d rows in registry.csv", len(rows))

        with conn.cursor() as cur:
            inserted = 0
            for row in rows:
                cur.execute("""
                    INSERT INTO documents (doc_id, title, url, source, bias_type, language, date, creator)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        title     = EXCLUDED.title,
                        url       = EXCLUDED.url,
                        source    = EXCLUDED.source,
                        bias_type = EXCLUDED.bias_type,
                        language  = EXCLUDED.language,
                        date      = EXCLUDED.date,
                        creator   = EXCLUDED.creator
                """, (
                    row["doc_id"],
                    row.get("title")    or None,
                    row.get("url")      or None,
                    row.get("source")   or None,
                    row.get("bias_type") or None,
                    row.get("language") or None,
                    row.get("date")     or None,
                    row.get("creator")  or None,
                ))
                inserted += 1

        conn.commit()
        log.info("Upserted %d documents into documents table", inserted)

    finally:
        conn.close()


if __name__ == "__main__":
    ingest_registry()