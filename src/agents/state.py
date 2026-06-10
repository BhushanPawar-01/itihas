"""
agents/state.py — Shared state schema for all LangGraph agents.

All agents read from and write to AgentState. No field is added anywhere
else — update this file first, then the agent that needs it.

No LangGraph imports. No SDK imports. Pure Python typing only.
"""

import operator
from typing import Annotated, Optional, TypedDict


class RetrievedChunk(TypedDict):
    doc_id: str
    chunk_index: int
    text: str
    source_type: str
    bias_tag: str
    language: str
    date: str
    score: float  # RRF score from fusion.py


class AgentOutput(TypedDict):
    agent_name: str    # "source" | "political" | "military" | "critique" | "narrative"
    content: str       # agent's prose or structured output
    confidence: float  # 0.0–1.0
    citations: list[str]  # list of doc_id strings this output draws from


class AgentState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────────
    query: str       # original user query, never mutated
    query_id: str    # uuid4 string, set by orchestrator on entry

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieved_chunks: list[RetrievedChunk]  # set by source agent, read by all others

    # ── Agent outputs — each agent writes exactly one AgentOutput ─────────────
    source_output: Optional[AgentOutput]
    political_output: Optional[AgentOutput]
    military_output: Optional[AgentOutput]
    critique_output: Optional[AgentOutput]
    narrative_output: Optional[AgentOutput]

    # ── Orchestration ─────────────────────────────────────────────────────────
    critique_loop_count: int   # incremented by critique agent; hard max 3
    route_to: Optional[str]    # set by orchestrator/critique to direct next node
    critique_passed: bool      # set True by critique agent when output is acceptable

    # ── Debug ─────────────────────────────────────────────────────────────────
    debug_log: Annotated[list[str], operator.add]  # append-only; all agents write here
    error: Optional[str]                           # set by any agent on unrecoverable failure