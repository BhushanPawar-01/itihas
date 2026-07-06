import sys
import time

from config.settings import DENSE_TOP_K
from src.storage import db_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


def embed_query(query: str) -> list[float]:
    """
    Embeds a single query string using the shared model instance from embedder.py.
    Importing get_model here — never loads a second copy of the model.
    """
    from src.processing.embedder import get_model

    model = get_model()
    return model.encode([query], convert_to_numpy=True)[0].tolist()


def search(
    query: str,
    top_k: int = DENSE_TOP_K,
    filters: dict | None = None,
) -> list[dict]:
    """
    Embeds the query and runs cosine similarity search via pgvector.
    Returns results in the same schema shape as bm25_retriever so RRF can fuse them.
    Each result has "dense_score" (not "score") and all original chunk fields.
    Results are ordered by dense_score descending (db_client already does this,
    but we re-sort to be explicit and safe).
    """
    start = time.time()

    try:
        query_embedding = embed_query(query)
    except Exception as e:
        logger.error(
            "Dense retriever: failed to embed query — returning empty results",
            extra={"error": str(e), "query": query[:80]},
        )
        return []

    try:
        raw_results = db_client.search_by_embedding(query_embedding, top_k, filters)
    except Exception as e:
        logger.error(
            "Dense retriever: embedding DB search failed — returning empty results",
            extra={"error": str(e)},
        )
        return []

    results = []
    for row in raw_results:
        result = dict(row)
        result["dense_score"] = float(result.pop("score"))
        results.append(result)

    results.sort(key=lambda x: x["dense_score"], reverse=True)

    filter_summary = None
    if filters:
        filter_summary = {k: v for k, v in filters.items() if v}

    logger.info(
        "Dense search complete",
        extra={
            "query": query[:80],
            "top_k": top_k,
            "filters": filter_summary,
            "result_count": len(results),
            "duration_sec": round(time.time() - start, 3),
        },
    )

    return results


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "INA trial Red Fort"
    results = search(query)
    for r in results:
        print(
            f"[{r['dense_score']:.4f}] {r['doc_id']} chunk {r['chunk_index']}: {r['text'][:120]}"
        )