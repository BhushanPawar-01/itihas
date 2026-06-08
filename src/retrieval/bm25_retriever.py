import sys

from rank_bm25 import BM25Okapi

from config.settings import BM25_TOP_K
from src.storage import db_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase + whitespace split. Used at index time and query time — never diverge."""
    return text.lower().split()


class BM25Retriever:
    def __init__(self):
        self._chunks: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._built: bool = False

    def build(self) -> None:
        self._chunks = db_client.get_all_texts_for_bm25()
        tokenized_corpus = [_tokenize(c["text"]) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._built = True
        logger.info("BM25 index built", extra={"corpus_size": len(self._chunks)})

    def search(
        self,
        query: str,
        top_k: int = BM25_TOP_K,
        filters: dict | None = None,
    ) -> list[dict]:
        if not self._built:
            self.build()

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        pairs = list(zip(scores, self._chunks))

        if filters:
            bias_tags = filters.get("bias_tags")
            languages = filters.get("languages")
            if bias_tags:
                pairs = [(s, c) for s, c in pairs if c.get("bias_tag") in bias_tags]
            if languages:
                pairs = [(s, c) for s, c in pairs if c.get("language") in languages]

        # Exclude zero-score chunks, sort descending, take top_k
        pairs = [(s, c) for s, c in pairs if s > 0.0]
        pairs.sort(key=lambda x: x[0], reverse=True)
        pairs = pairs[:top_k]

        results = []
        for score, chunk in pairs:
            result = dict(chunk)
            result["bm25_score"] = float(score)
            results.append(result)

        return results


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_retriever: BM25Retriever | None = None


def get_retriever() -> BM25Retriever:
    global _retriever
    if _retriever is None:
        _retriever = BM25Retriever()
    return _retriever


def search(
    query: str,
    top_k: int = BM25_TOP_K,
    filters: dict | None = None,
) -> list[dict]:
    return get_retriever().search(query, top_k, filters)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "INA trial Red Fort"
    results = search(query)
    for r in results:
        print(
            f"[{r['bm25_score']:.3f}] {r['doc_id']} chunk {r['chunk_index']}: {r['text'][:120]}"
        )