"""
FastAPI route — exposes the Itihas agent graph as a single POST endpoint.

Does not import LangGraph, llm_client, or any agent file directly.
Only calls run_query() from src.agents.graph.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.agents.graph import run_query

router = APIRouter()


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


def resolve_citations(doc_ids: list[str]) -> list[dict]:
    """
    Look up title and URL for each doc_id from the documents table.
    Falls back to doc_id as title if not found.
    Never raises — returns safe fallback on any DB error.
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
        # Preserve order, fill gaps for any doc_id not in documents table
        return [
            rows.get(doc_id, {"doc_id": doc_id, "title": doc_id, "url": None})
            for doc_id in dict.fromkeys(doc_ids)  # deduplicate preserving order
        ]
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("resolve_citations failed: %s", exc)
        return [{"doc_id": d, "title": d, "url": None} for d in dict.fromkeys(doc_ids)]


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
        critique_output=state.get("critique_output", {}).get("content") if state.get("critique_output") else None,
        source_chunks=source_chunks,
        debug_log=state["debug_log"] if request.include_debug_log else None,
        error=None,
    )