import argparse
import time

from config.settings import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL
from src.storage import db_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

_model = None


def get_model():
    """
    Lazily loads the SentenceTransformer model on first call.
    Caches in module-level _model. Safe to import this module in tests
    without triggering a download.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        import torch

        logger.info("Loading embedding model", extra={"model": EMBEDDING_MODEL})
        _model = SentenceTransformer(EMBEDDING_MODEL)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device)

        logger.info(
            "Embedding model loaded",
            extra={"model": EMBEDDING_MODEL, "device": device},
        )
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Encodes a list of strings into embedding vectors.
    Returns a list of plain Python float lists (not numpy arrays).
    Returns [] for empty input without loading the model.
    """
    if not texts:
        return []

    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


def embed_all(force: bool = False) -> dict:
    """
    Fetches chunks that need embeddings, encodes them in batches, and
    writes vectors back to Postgres via db_client.update_embedding().

    force=False: only chunks with embedding IS NULL
    force=True:  all chunks (re-embed everything)

    Returns {"embedded": int, "model": str, "device": str}
    """
    import torch

    if force:
        chunks = db_client.get_all_texts_for_bm25()
        logger.info("Force mode: fetched all chunks", extra={"count": len(chunks)})
    else:
        chunks = db_client.get_chunks_without_embeddings(limit=10_000)
        logger.info(
            "Fetched chunks without embeddings", extra={"count": len(chunks)}
        )

    if not chunks:
        logger.info("No chunks to embed")
        return {
            "embedded": 0,
            "model": EMBEDDING_MODEL,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

    total = len(chunks)
    logger.info(
        "Embedding chunks",
        extra={"count": total, "model": EMBEDDING_MODEL},
    )

    embedded_count = 0
    batch_num = 0
    start_time = time.time()

    for batch_start in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
        texts = [c["text"] for c in batch]

        vectors = embed_texts(texts)

        for chunk, vector in zip(batch, vectors):
            db_client.update_embedding(chunk["id"], vector)

        embedded_count += len(batch)
        batch_num += 1

        if batch_num % 10 == 0:
            logger.info(
                "Embedding progress",
                extra={"embedded": embedded_count, "total": total},
            )

    elapsed = time.time() - start_time
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        "Embedding complete",
        extra={
            "embedded": embedded_count,
            "total": total,
            "model": EMBEDDING_MODEL,
            "device": device,
            "duration_sec": round(elapsed, 2),
        },
    )

    return {"embedded": embedded_count, "model": EMBEDDING_MODEL, "device": device}


def main():
    parser = argparse.ArgumentParser(
        description="Embed chunk text into vectors and write to Postgres."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed all chunks, not just those with null embeddings.",
    )
    args = parser.parse_args()

    result = embed_all(force=args.force)

    model_short = result["model"].split("/")[-1]
    print(
        f"Embedding complete. {result['embedded']} chunks embedded with "
        f"{model_short} on {result['device']}"
    )


if __name__ == "__main__":
    main()