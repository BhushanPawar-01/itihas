import argparse
import csv

from config.settings import REGISTRY_PATH
from src.storage import db_client, file_store
from src.utils.logger import get_logger


logger = get_logger(__name__)


def ingest_doc(doc_id: str) -> int:
    """
    Reads chunk file via file_store.read_processed_chunks(doc_id).
    Calls db_client.upsert_chunks_batch(chunks).
    Returns count of chunks ingested.
    Raises FileNotFoundError if chunk file missing — caller handles it.
    """
    try:
        chunks = file_store.read_processed_chunks(doc_id)
    except FileNotFoundError:
        raise FileNotFoundError(f"Chunk file missing for doc_id: {doc_id}")
    except Exception as e:
        # Catch other exceptions like JSONDecodeError but preserve FileNotFoundError if any.
        # file_store.read_processed_chunks raises the original exception.
        if isinstance(e, FileNotFoundError):
            raise
        raise e

    if not chunks:
        return 0

    db_client.upsert_chunks_batch(chunks)
    return len(chunks)


def ingest_all(force: bool = False) -> dict:
    """
    Reads all doc_ids from REGISTRY_PATH.
    For each doc_id:
      If not force: check if chunks already in DB via db_client.get_chunks_by_doc(doc_id).
      If results exist and not force: skip.
      Otherwise call ingest_doc(doc_id).
      Track: success list, skipped list, failed list with error strings.
    Returns summary dict:
      {"success": [...], "skipped": [...], "failed": [...], "total_chunks": int}
    """
    summary = {
        "success": [],
        "skipped": [],
        "failed": [],
        "total_chunks": 0
    }

    try:
        with open(REGISTRY_PATH, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    except Exception as e:
        logger.error(f"Failed to read registry: {e}")
        return summary

    for row in rows:
        doc_id = row.get("doc_id")
        if not doc_id:
            continue

        if not force:
            try:
                existing_chunks = db_client.get_chunks_by_doc(doc_id)
                if existing_chunks:
                    summary["skipped"].append(doc_id)
                    continue
            except Exception as e:
                summary["failed"].append((doc_id, f"DB check failed: {e}"))
                continue

        try:
            count = ingest_doc(doc_id)
            summary["success"].append(doc_id)
            summary["total_chunks"] += count
        except FileNotFoundError as e:
            summary["failed"].append((doc_id, str(e)))
        except Exception as e:
            summary["failed"].append((doc_id, str(e)))

    return summary


def main():
    parser = argparse.ArgumentParser(description="Ingest chunks into PostgreSQL.")
    parser.add_argument("--force", action="store_true", help="Re-ingest all chunks even if they exist in DB.")
    parser.add_argument("--doc-id", type=str, help="Ingest a single document by ID.")

    args = parser.parse_args()

    if args.doc_id:
        doc_id = args.doc_id
        try:
            if not args.force:
                existing_chunks = db_client.get_chunks_by_doc(doc_id)
                if existing_chunks:
                    print(f"Skipped {doc_id}: already exists in DB. Use --force to overwrite.")
                    return

            count = ingest_doc(doc_id)
            print(f"Ingestion complete. Success: 1 docs ({count} chunks) | Skipped: 0 | Failed: 0")
        except Exception as e:
            print(f"Failed to ingest {doc_id}: {e}")
    else:
        summary = ingest_all(force=args.force)

        print(f"Ingestion complete. Success: {len(summary['success'])} docs ({summary['total_chunks']} chunks) | Skipped: {len(summary['skipped'])} | Failed: {len(summary['failed'])}")
        if summary["failed"]:
            print("Failed documents:")
            for doc_id, err in summary["failed"]:
                print(f"  - {doc_id}: {err}")


if __name__ == "__main__":
    main()
