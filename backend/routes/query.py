"""
FastAPI routes — Itihas agent graph endpoints.

POST /query         — blocking, returns full QueryResponse JSON.
                      Accepts optional conversation_history to enable follow-up
                      queries within a session. The history is passed to run_query()
                      which builds a conversation context string via memory.py and
                      seeds it into AgentState before the graph runs.

POST /query/stream  — streaming SSE, emits agent events as they complete;
                      used by the frontend DebateFeed component. Does NOT carry
                      conversation history — the stream is live visual feedback only;
                      the final contextual answer comes from POST /query.

Neither endpoint imports LangGraph, llm_client, or agent files directly.
All graph execution goes through src.agents.graph.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agents.graph import run_query, run_query_streaming

router = APIRouter()


# ── Request / Response models ──────────────────────────────────────────────────

class ConversationTurn(BaseModel):
    """
    One completed prior turn. Mirrors what the frontend accumulates in
    App.jsx's conversationHistory state.

    source_chunks may be None if the frontend did not receive them (e.g. on
    an earlier error turn that still produced a narrative). memory.py handles
    None gracefully.
    """
    query:         str
    narrative:     str
    source_chunks: list[dict] | None = None


class QueryRequest(BaseModel):
    query:                str
    include_debug_log:    bool                    = False
    # Optional list of prior completed turns, oldest first.
    # Omit or pass [] for the first query in a session.
    conversation_history: list[ConversationTurn] = []


class CitationItem(BaseModel):
    doc_id: str
    title:  str
    url:    str | None


class QueryResponse(BaseModel):
    query_id:           str
    query:              str
    narrative:          str
    confidence:         float
    citations:          list[CitationItem]
    political_analysis: str
    military_analysis:  str
    critique_loops:     int
    critique_output:    str | None
    source_chunks:      list[dict] | None
    debug_log:          list[str] | None
    error:              str | None


# ── Citation resolver ──────────────────────────────────────────────────────────

def resolve_citations(doc_ids: list[str]) -> list[dict]:
    """
    Look up title and URL for each doc_id from the documents table.
    Falls back to doc_id as title if not found. Never raises.
    """
    if not doc_ids:
        return []
    try:
        import psycopg2
        from config.settings import DB_URL, DB_SSLMODE
        conn = psycopg2.connect(DB_URL, sslmode=DB_SSLMODE)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT doc_id, title, url
                    FROM documents
                    WHERE doc_id = ANY(%s)
                """, (doc_ids,))
                rows = {r[0]: {"doc_id": r[0], "title": r[1], "url": r[2]}
                        for r in cur.fetchall()}
        finally:
            conn.close()
        return [
            rows.get(doc_id, {"doc_id": doc_id, "title": doc_id, "url": None})
            for doc_id in dict.fromkeys(doc_ids)
        ]
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("resolve_citations failed: %s", exc)
        return [{"doc_id": d, "title": d, "url": None} for d in dict.fromkeys(doc_ids)]


# ── POST /query — blocking ─────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_history(request: QueryRequest) -> QueryResponse:
    import traceback

    # Convert Pydantic ConversationTurn objects to plain dicts for memory.py.
    # memory.py expects: {"query": str, "narrative": str, "source_chunks": list|None}
    history_dicts = [
        {
            "query":         turn.query,
            "narrative":     turn.narrative,
            "source_chunks": turn.source_chunks,
        }
        for turn in request.conversation_history
    ]

    loop = asyncio.get_event_loop()
    try:
        state = await loop.run_in_executor(
            None, run_query, request.query, history_dicts
        )
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"EXECUTOR CRASH:\n{tb}")
        raise HTTPException(status_code=500, detail=str(exc))

    if state.get("error"):
        raise HTTPException(status_code=500, detail=state["error"])
    if state.get("narrative_output") is None:
        raise HTTPException(
            status_code=500,
            detail="narrative_output missing from final state — check debug_log",
        )

    source_chunks = None
    try:
        source_chunks = json.loads(state["source_output"]["content"])
    except Exception:
        pass

    return QueryResponse(
        query_id=state["query_id"],
        query=state["query"],
        narrative=state["narrative_output"]["content"],
        confidence=state["narrative_output"]["confidence"],
        citations=resolve_citations(state["narrative_output"]["citations"]),
        political_analysis=state["political_output"]["content"],
        military_analysis=state["military_output"]["content"],
        critique_loops=state["critique_loop_count"],
        critique_output=(
            state.get("critique_output", {}).get("content")
            if state.get("critique_output") else None
        ),
        source_chunks=source_chunks,
        debug_log=state["debug_log"] if request.include_debug_log else None,
        error=None,
    )


# ── POST /query/stream — SSE streaming ────────────────────────────────────────

class StreamQueryRequest(BaseModel):
    query: str


@router.post("/query/stream")
async def stream_query(request: StreamQueryRequest) -> StreamingResponse:
    """
    Streams agent events as Server-Sent Events while the graph runs.
    Does NOT accept conversation_history — the stream feeds DebateFeed
    (live visual feedback only). The final answer with full context is
    delivered by the concurrent POST /query call from the frontend.

    Each event is a JSON object:
      { type, agent, label, loop, content, error }   — node_complete / rebuttal
      { type: "done" }                               — graph finished
      { type: "error", content, traceback }          — unhandled exception
    """
    loop = asyncio.get_event_loop()

    async def async_generator():
        import queue
        import threading

        q: queue.Queue = queue.Queue()
        SENTINEL = object()

        def producer():
            try:
                for event in run_query_streaming(request.query):
                    q.put(event)
            except Exception as exc:
                q.put(f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n")
            finally:
                q.put(SENTINEL)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue

            if item is SENTINEL:
                break
            yield item

    return StreamingResponse(
        async_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )