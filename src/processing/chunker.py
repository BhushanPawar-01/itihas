from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.settings import METADATA_DIR, PROCESSED_TEXT_DIR, PROCESSED_TRANSLATED_DIR
from src.storage.file_store import read_metadata, write_metadata, write_processed_chunks
from src.utils.logger import get_logger


log = get_logger(__name__)

LAST_RUN_DOC_COUNT = 0
LAST_RUN_CHUNK_COUNT = 0


def count_tokens(text: str) -> int:
    return len(text.split())


def split_into_sentences(text: str) -> list[str]:
    try:
        import nltk

        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            return nltk.sent_tokenize(text)
    except Exception as exc:
        log.error("Failed to split text into sentences: size=%d error=%s", len(text), exc)
        raise


def _overlap_text(chunk: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    return " ".join(chunk.split()[-overlap:])


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    try:
        sentences = split_into_sentences(text)
        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_tokens = count_tokens(sentence)
            if current_parts and current_tokens + sentence_tokens > chunk_size:
                chunk = " ".join(current_parts).strip()
                chunks.append(chunk)

                overlap_part = _overlap_text(chunk, overlap)
                overlap_tokens = count_tokens(overlap_part)
                if overlap_part and overlap_tokens + sentence_tokens <= chunk_size:
                    current_parts = [overlap_part]
                    current_tokens = overlap_tokens
                else:
                    current_parts = []
                    current_tokens = 0

            current_parts.append(sentence)
            current_tokens += sentence_tokens

        if current_parts:
            chunks.append(" ".join(current_parts).strip())

        return chunks
    except Exception as exc:
        log.error(
            "Failed to chunk text: size=%d chunk_size=%d overlap=%d error=%s",
            len(text),
            chunk_size,
            overlap,
            exc,
        )
        raise


def _read_source_text(doc_id: str) -> str:
    translated_path = PROCESSED_TRANSLATED_DIR / f"{doc_id}.txt"
    text_path = PROCESSED_TEXT_DIR / f"{doc_id}.txt"

    try:
        if translated_path.exists():
            return translated_path.read_text(encoding="utf-8")
        if text_path.exists():
            return text_path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"No processed text found for doc_id={doc_id}")
    except Exception as exc:
        log.error("Failed to read chunk source text: doc_id=%s error=%s", doc_id, exc)
        raise


def _build_chunk(doc_id: str, chunk_index: int, text: str, meta: dict) -> dict:
    source_type = str(meta.get("bias_type", ""))
    return {
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "text": text,
        "embedding": None,
        "source_type": source_type,
        "bias_tag": source_type,
        "language": str(meta.get("language_detected", "")),
        "date": str(meta.get("date") or ""),
        "page": None,
        "confidence": None,
    }


def chunk_document(doc_id: str) -> list[dict]:
    log.info("Chunking document: doc_id=%s", doc_id)
    try:
        meta = read_metadata(doc_id)
        text = _read_source_text(doc_id)
        chunk_strings = chunk_text(text)
        chunks = [_build_chunk(doc_id, index, chunk, meta) for index, chunk in enumerate(chunk_strings)]

        write_processed_chunks(doc_id, chunks)
        meta["chunk_count"] = len(chunks)
        meta["processed_at"] = datetime.now(timezone.utc).isoformat()
        write_metadata(doc_id, meta)

        log.info("Chunked document: doc_id=%s chunk_count=%d", doc_id, len(chunks))
        return chunks
    except Exception as exc:
        log.error("Failed to chunk document: doc_id=%s error=%s", doc_id, exc)
        raise


def _metadata_doc_id(path: Path) -> str:
    return path.stem


def _should_process(path: Path, overwrite: bool) -> bool:
    if overwrite:
        return True

    try:
        meta = read_metadata(_metadata_doc_id(path))
        return int(meta.get("chunk_count", 0) or 0) == 0
    except Exception as exc:
        log.error("Failed to inspect metadata for chunking: path=%s error=%s", path, exc)
        raise


def run_all(overwrite: bool = False) -> None:
    global LAST_RUN_CHUNK_COUNT, LAST_RUN_DOC_COUNT

    log.info("Running chunking for all metadata: overwrite=%s", overwrite)
    LAST_RUN_DOC_COUNT = 0
    LAST_RUN_CHUNK_COUNT = 0

    metadata_paths = sorted(METADATA_DIR.glob("*.json"))
    docs_to_process = [path for path in metadata_paths if _should_process(path, overwrite)]

    for path in docs_to_process:
        doc_id = _metadata_doc_id(path)
        try:
            chunks = chunk_document(doc_id)
            LAST_RUN_DOC_COUNT += 1
            LAST_RUN_CHUNK_COUNT += len(chunks)
        except Exception as exc:
            log.error("Skipping failed chunking document: doc_id=%s error=%s", doc_id, exc)

    log.info(
        "Completed chunking run: overwrite=%s docs=%d chunks=%d",
        overwrite,
        LAST_RUN_DOC_COUNT,
        LAST_RUN_CHUNK_COUNT,
    )


def main() -> None:
    run_all()
    print(f"Chunked {LAST_RUN_DOC_COUNT} docs, total {LAST_RUN_CHUNK_COUNT} chunks")


if __name__ == "__main__":
    main()
