"""
FastAPI routes — Itihas agent graph endpoints.

POST /query         — blocking, returns full QueryResponse JSON (unchanged)
POST /query/stream  — streaming SSE, emits agent events as they complete;
                      used by the frontend DebateFeed component

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

class QueryRequest(BaseModel):
    query: str
    include_debug_log: bool = False

class CitationItem(BaseModel):
    doc_id: str
    title: str
    url: str | None

class QueryResponse(BaseModel):
    query_id: str
    query: str
    narrative: str
    confidence: float
    citations: list[CitationItem]
    political_analysis: str
    military_analysis: str
    critique_loops: int
    critique_output: str | None
    source_chunks: list[dict] | None
    debug_log: list[str] | None
    error: str | None


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


# ── POST /query — blocking (unchanged) ────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_history(request: QueryRequest) -> QueryResponse:
    import traceback
    loop = asyncio.get_event_loop()
    try:
        state = await loop.run_in_executor(None, run_query, request.query)
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

    Each event is a JSON object:
      { type, agent, label, loop, content, error }   — node_complete / rebuttal
      { type: "done" }                               — graph finished
      { type: "error", content, traceback }          — unhandled exception

    The frontend runs this alongside POST /query:
      - /query/stream feeds DebateFeed (live agent steps)
      - /query         delivers the final QueryResponse when complete

    Uses StreamingResponse with text/event-stream — no extra dependencies.
    The generator runs in a thread pool (run_in_executor) so it doesn't block
    the event loop; FastAPI iterates the async wrapper.
    """
    loop = asyncio.get_event_loop()

    async def async_generator():
        # run_query_streaming is a sync generator — wrap in a thread
        # We collect events via a queue to bridge sync generator → async generator
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
            # Poll queue without blocking the event loop
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
            "X-Accel-Buffering":           "no",   # disable nginx buffering if present
            "Access-Control-Allow-Origin": "*",
        },
    )