import sys
import time

from config.settings import (
    RERANKER_ENABLED,
    RERANKER_MODEL,
    RETRIEVAL_TOP_K,
    RRF_K,
)
from src.retrieval import bm25_retriever, dense_retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    bm25_results: list[dict],
    dense_results: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """
    Fuses two ranked lists using Reciprocal Rank Fusion.
    RRF score per chunk = sum of 1/(k + rank) across all lists it appears in.
    rank is 1-based. Chunks appearing in only one list still receive a score.
    Unique chunk identity: (doc_id, chunk_index).
    """
    scores: dict[tuple, float] = {}
    in_bm25: set[tuple] = set()
    in_dense: set[tuple] = set()
    # Prefer dense metadata when a chunk appears in both (more fields: bias_tag, language, date)
    metadata: dict[tuple, dict] = {}

    for rank, chunk in enumerate(bm25_results, start=1):
        key = (chunk["doc_id"], chunk["chunk_index"])
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        in_bm25.add(key)
        if key not in metadata:
            metadata[key] = chunk

    for rank, chunk in enumerate(dense_results, start=1):
        key = (chunk["doc_id"], chunk["chunk_index"])
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        in_dense.add(key)
        metadata[key] = chunk  # dense wins on conflict

    results = []
    for key, rrf_score in scores.items():
        result = dict(metadata[key])
        result["rrf_score"] = rrf_score
        result["in_bm25"] = key in in_bm25
        result["in_dense"] = key in in_dense
        results.append(result)

    results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return results


def rerank(chunks: list[dict], query: str) -> list[dict]:
    """
    Re-sorts chunks using a cross-encoder via HuggingFace Inference API.
    Only called when RERANKER_ENABLED=True. Gracefully falls back to original
    order on any API failure.
    """
    logger.warning(
        "Reranker enabled — this incurs API cost and latency.",
        extra={"model": RERANKER_MODEL, "chunk_count": len(chunks)},
    )

    try:
        from src.utils.llm_client import rerank as llm_rerank

        pairs = [(query, chunk["text"]) for chunk in chunks]
        reranker_scores = llm_rerank(pairs, model=RERANKER_MODEL)

        for chunk, score in zip(chunks, reranker_scores):
            chunk["reranker_score"] = float(score)

        chunks.sort(key=lambda x: x["reranker_score"], reverse=True)
        return chunks

    except Exception as e:
        logger.error(
            "Reranker API call failed — returning original RRF order.",
            extra={"error": str(e)},
        )
        return chunks


def retrieve(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    filters: dict | None = None,
) -> list[dict]:
    """
    Single entry point for agents. Never call bm25_retriever or dense_retriever directly.
    Pipeline: BM25 + Dense → RRF fusion → top_k → optional rerank.
    """
    start = time.time()

    bm25_results = bm25_retriever.search(query, top_k=20, filters=filters)
    dense_results = dense_retriever.search(query, top_k=20, filters=filters)

    fused = reciprocal_rank_fusion(bm25_results, dense_results)
    top_chunks = fused[:top_k]

    if RERANKER_ENABLED:
        top_chunks = rerank(top_chunks, query)

    logger.info(
        "Retrieval complete",
        extra={
            "query": query[:80],
            "bm25_count": len(bm25_results),
            "dense_count": len(dense_results),
            "fused_count": len(fused),
            "top_k": top_k,
            "reranker_enabled": RERANKER_ENABLED,
            "duration_sec": round(time.time() - start, 3),
        },
    )

    return top_chunks


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "INA trial Red Fort"
    results = retrieve(query)
    for r in results:
        in_both = (
            "BM25+Dense"
            if r["in_bm25"] and r["in_dense"]
            else ("BM25 only" if r["in_bm25"] else "Dense only")
        )
        print(
            f"[{r['rrf_score']:.4f}] ({in_both}) {r['doc_id']} c{r['chunk_index']}: {r['text'][:100]}"
        )