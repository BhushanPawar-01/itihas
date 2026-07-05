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


class QueryResponse(BaseModel):
    query_id: str
    query: str
    narrative: str
    confidence: float
    citations: list[str]
    political_analysis: str
    military_analysis: str
    critique_loops: int
    critique_output: str | None
    source_chunks: list[dict] | None  # parsed from source_output["content"], None on error
    debug_log: list[str] | None  # None unless include_debug_log=True
    error: str | None


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
        citations=state["narrative_output"]["citations"],
        political_analysis=state["political_output"]["content"],
        military_analysis=state["military_output"]["content"],
        critique_loops=state["critique_loop_count"],
        critique_output=state.get("critique_output", {}).get("content") if state.get("critique_output") else None,
        source_chunks=source_chunks,
        debug_log=state["debug_log"] if request.include_debug_log else None,
        error=None,
    )