import argparse
import csv
from typing import Dict, Any

from config.settings import REGISTRY_PATH
from src.storage import file_store
from src.storage import db_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


def ingest_doc(doc_id: str) -> int:
    """
    Reads chunk file via file_store.read_processed_chunks(doc_id).
    Calls db_client.upsert_chunks_batch(chunks).
    Returns count of chunks ingested.
    Raises FileNotFoundError if chunk file missing - caller handles it.
    """
    try:
        chunks = file_store.read_processed_chunks(doc_id)
    except FileNotFoundError:
        logger.warning(f"Chunk file missing for doc_id={doc_id}")
        raise

    if not chunks:
        logger.warning(f"No chunks found for doc_id={doc_id}")
        return 0

    db_client.upsert_chunks_batch(chunks)
    logger.info(f"Ingested {len(chunks)} chunks for doc_id={doc_id}")
    return len(chunks)


def ingest_all(force: bool = False) -> Dict[str, Any]:
    """
    Reads all doc_ids from REGISTRY_PATH.
    For each doc_id:
      If not force: check if chunks already in DB via db_client.get_chunks_by_doc(doc_id).
      If results exist and not force: skip.
      Otherwise call ingest_doc(doc_id).
      Track: success list, skipped list, failed list with error strings.
    Returns summary dict.
    """
    success = []
    skipped = []
    failed = []
    total_chunks = 0

    try:
        with open(REGISTRY_PATH, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    except Exception as e:
        logger.error(f"Failed to read registry: {e}")
        return {"success": success, "skipped": skipped, "failed": failed, "total_chunks": total_chunks}

    for row in rows:
        doc_id = row.get("doc_id")
        if not doc_id:
            continue

        if not force:
            try:
                existing = db_client.get_chunks_by_doc(doc_id)
                if existing:
                    logger.info(f"Skipping doc_id={doc_id}, already exists in DB.")
                    skipped.append(doc_id)
                    continue
            except Exception as e:
                logger.error(f"Error checking DB for doc_id={doc_id}: {e}")
                failed.append({"doc_id": doc_id, "error": str(e)})
                continue

        try:
            count = ingest_doc(doc_id)
            success.append(doc_id)
            total_chunks += count
        except FileNotFoundError as e:
            logger.warning(f"File not found for doc_id={doc_id}: {e}")
            failed.append({"doc_id": doc_id, "error": "FileNotFoundError"})
        except Exception as e:
            logger.error(f"Failed to ingest doc_id={doc_id}: {e}")
            failed.append({"doc_id": doc_id, "error": str(e)})

    return {
        "success": success,
        "skipped": skipped,
        "failed": failed,
        "total_chunks": total_chunks
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest processed chunks into PostgreSQL.")
    parser.add_argument("--force", action="store_true", help="Re-ingest everything, overwriting existing records")
    parser.add_argument("--doc-id", type=str, help="Ingest a specific doc_id")
    args = parser.parse_args()

    if args.doc_id:
        doc_id = args.doc_id
        if not args.force:
            existing = db_client.get_chunks_by_doc(doc_id)
            if existing:
                print(f"Ingestion complete. Success: 0 docs (0 chunks) | Skipped: 1 | Failed: 0")
                return

        try:
            count = ingest_doc(doc_id)
            print(f"Ingestion complete. Success: 1 docs ({count} chunks) | Skipped: 0 | Failed: 0")
        except Exception as e:
            print(f"Ingestion complete. Success: 0 docs (0 chunks) | Skipped: 0 | Failed: 1")
            print(f"Failed doc_ids: {doc_id} - Error: {e}")
    else:
        results = ingest_all(force=args.force)

        success_count = len(results["success"])
        skipped_count = len(results["skipped"])
        failed_count = len(results["failed"])
        total_chunks = results["total_chunks"]

        print(f"Ingestion complete. Success: {success_count} docs ({total_chunks} chunks) | Skipped: {skipped_count} | Failed: {failed_count}")

        if results["failed"]:
            print("Failed doc_ids:")
            for fail in results["failed"]:
                print(f"  {fail['doc_id']}: {fail['error']}")


if __name__ == "__main__":
    main()
