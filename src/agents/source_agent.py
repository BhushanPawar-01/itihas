"""
LangGraph node — Source Agent.

Responsibilities:
  1. Run hybrid retrieval (BM25 + dense + RRF) via fusion.retrieve().
  2. Triage each chunk with the default llm_client backend: KEEP or DISCARD.
     If fewer than 3 survive triage, bypass and keep all retrieved chunks.
  3. Return a partial state dict — never the full AgentState.

LLM calls: call() only; llm_client defaults agents to OpenAI.
"""

from __future__ import annotations

import json
import time
import traceback

from .state import AgentOutput, AgentState, RetrievedChunk
from src.retrieval.fusion import retrieve
from src.utils.llm_client import call
from src.utils.logger import get_logger

log = get_logger(__name__)

# Triage prompt — exact wording kept here so tests can assert on it if needed
_TRIAGE_PROMPT = (
    "Query: {query}\n"
    "Chunk: {chunk_text}\n"
    "Is this chunk relevant to the query? Reply with KEEP or DISCARD only."
)


def _triage_chunk(query: str, chunk: RetrievedChunk) -> bool:
    """
    Call the default llm_client backend to decide KEEP/DISCARD for a single chunk.
    Returns True if the reply contains "KEEP" (case-insensitive).
    Any LLMCallError is treated as DISCARD — logged but not raised.
    """
    prompt = _TRIAGE_PROMPT.format(
        query=query,
        chunk_text=chunk["text"][:400],
    )
    try:
        reply = call(
            prompt,
            max_tokens=40,
            temperature=0.01,
            stop_sequences=["KEEP", "DISCARD"],
        )
        return "keep" in reply.lower()
    except Exception as exc:
        log.warning(
            "source_agent: triage call failed for doc_id=%s chunk=%d — treating as DISCARD: %s",
            chunk["doc_id"], chunk["chunk_index"], exc,
        )
        return False


def source_node(state: AgentState) -> dict:
    """
    LangGraph node — returns partial state update dict.

    Return keys:
        retrieved_chunks : list[RetrievedChunk]  — kept chunks after triage
        source_output    : AgentOutput
        debug_log        : list[str]             — single entry, appended by LangGraph reducer

    On unrecoverable error returns:
        {"error": str, "debug_log": [str]}
    """
    t0 = time.monotonic()
    query = state["query"]

    try:
        # ── 1. Retrieve ───────────────────────────────────────────────────────
        raw: list[dict] = retrieve(query=query, top_k=20, filters={})

        # Cast to RetrievedChunk — retrieve() guarantees these fields exist
        retrieved: list[RetrievedChunk] = [
            RetrievedChunk(
                doc_id=r["doc_id"],
                chunk_index=r["chunk_index"],
                text=r["text"],
                source_type=r["source_type"],
                bias_tag=r["bias_tag"],
                language=r["language"],
                date=r["date"],
                score=r.get("reranker_score", r.get("rrf_score", 0.0)),
            )
            for r in raw
        ]

        # ── 2. Triage ─────────────────────────────────────────────────────────
        kept: list[RetrievedChunk] = [
            chunk for chunk in retrieved if _triage_chunk(query, chunk)
        ]

        bypass_used = False
        if len(kept) < 3:
            bypass_used = True
            kept = retrieved
            log.warning(
                "source_agent: triage kept fewer than 3 chunks (%d) — bypassing, keeping all %d",
                len(kept), len(retrieved),
            )

        # ── 3. Build AgentOutput ──────────────────────────────────────────────
        content_list = [
            {
                "doc_id":      c["doc_id"],
                "chunk_index": c["chunk_index"],
                "text":        c["text"],
                "bias_tag":    c["bias_tag"],
                "score":       c["score"],
            }
            for c in kept
        ]

        confidence = round(len(kept) / len(retrieved), 2) if retrieved else 0.0

        output: AgentOutput = {
            "agent_name": "source",
            "content":    json.dumps(content_list),
            "confidence": confidence,
            "citations":  [c["doc_id"] for c in kept],
        }

        # ── 4. Build debug entry ──────────────────────────────────────────────
        duration_ms = round((time.monotonic() - t0) * 1000)
        bypass_note = " [triage bypassed]" if bypass_used else ""
        debug_entry = (
            f"source_agent: retrieved={len(retrieved)}, kept={len(kept)}, "
            f"query_chars={len(query)}, duration_ms={duration_ms}{bypass_note}"
        )

        return {
            "retrieved_chunks": kept,
            "source_output":    output,
            "debug_log":        [debug_entry],
        }

    except Exception as exc:
        tb = traceback.format_exc()
        log.error("source_agent: FAILED — %s\n%s", exc, tb)
        return {
            "error":     str(exc),
            "debug_log": [f"source_agent: FAILED — {exc}"],
        }