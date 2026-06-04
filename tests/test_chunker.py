from __future__ import annotations

from src.processing import chunker


def _sentences(count: int, words_per_sentence: int) -> list[str]:
    return [
        " ".join([f"word{sentence_index}_{word_index}" for word_index in range(words_per_sentence)]) + "."
        for sentence_index in range(count)
    ]


def test_chunk_text_produces_no_chunk_over_512_tokens(monkeypatch):
    sentences = _sentences(count=6, words_per_sentence=100)
    text = " ".join(sentences)
    monkeypatch.setattr(chunker, "split_into_sentences", lambda _text: sentences)

    chunks = chunker.chunk_text(text)

    assert chunks
    assert all(chunker.count_tokens(chunk) <= 512 for chunk in chunks)


def test_chunk_text_preserves_overlap_tokens(monkeypatch):
    sentences = _sentences(count=6, words_per_sentence=100)
    text = " ".join(sentences)
    monkeypatch.setattr(chunker, "split_into_sentences", lambda _text: sentences)

    chunks = chunker.chunk_text(text)

    assert len(chunks) > 1
    first_overlap = chunks[0].split()[-64:]
    second_start = chunks[1].split()[:64]
    assert second_start == first_overlap


def test_chunk_text_respects_sentence_boundaries(monkeypatch):
    sentences = _sentences(count=6, words_per_sentence=100)
    text = " ".join(sentences)
    monkeypatch.setattr(chunker, "split_into_sentences", lambda _text: sentences)

    chunks = chunker.chunk_text(text)

    assert chunks
    assert all(chunk.endswith(".") for chunk in chunks)
