"""
LangGraph StateGraph — wires all five nodes and exposes run_query() as the
public entry point.

Topology:
  START → source → political ──┐
                 → military  ──┴→ critique → route_after_critique
                                               ↓ "narrative"      → narrative → END
                                               ↓ "political"      → political (loop, max 3x)
                                               ↓ "end_with_error" → END

The graph topology IS the orchestrator — no separate orchestrator agent file.
Political and military run in parallel via LangGraph's fan-out (two add_edge
calls from source). LangGraph joins them automatically before critique.
"""

from __future__ import annotations

import uuid

from langgraph.graph import END, StateGraph

from src.agents.state import AgentState
from src.agents.source_agent    import source_node
from src.agents.political_agent import political_node
from src.agents.military_agent  import military_node
from src.agents.critique_agent  import critique_node
from src.agents.narrative_agent import narrative_node


def route_after_source(state: AgentState) -> str | list[str]:
    """
    Conditional edge function after retrieval.

    Returns:
      "end_with_error" - source_node failed and set state["error"]
      ["political", "military"] - source_node succeeded; run both analyses in parallel
    """
    if state.get("error"):
        return "end_with_error"
    return ["political", "military"]


def route_after_critique(state: AgentState) -> str:
    """
    Conditional edge function — called by LangGraph after critique_node completes.
    Reads state only. No LLM calls. No side effects.

    Returns:
      "end_with_error" — state["error"] is set (any non-None, non-empty value)
      "narrative"      — state["critique_passed"] is True
      "political"      — state["critique_passed"] is False
                         (loops back to political+military; source not re-run
                          because retrieved_chunks are already in state)
    """
    if state.get("error"):
        return "end_with_error"
    if state.get("critique_passed"):
        return "narrative"
    return "political"


def build_graph():
    """
    Build and compile the Itihas LangGraph.
    Returns the compiled graph app.
    Caller: app.invoke(initial_state) -> final AgentState.
    """
    graph = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    graph.add_node("source",    source_node)
    graph.add_node("political", political_node)
    graph.add_node("military",  military_node)
    graph.add_node("critique",  critique_node)
    graph.add_node("narrative", narrative_node)

    # ── Entry ─────────────────────────────────────────────────────────────────
    graph.set_entry_point("source")

    # ── Fan-out: source → political and military in parallel ──────────────────
    graph.add_conditional_edges(
        "source",
        route_after_source,
        {
            "political":      "political",
            "military":       "military",
            "end_with_error": END,
        },
    )

    # ── Fan-in: both converge on critique ────────────────────────────────────
    graph.add_edge("political", "critique")
    graph.add_edge("military",  "critique")

    # ── Conditional: critique loops back or proceeds to narrative ─────────────
    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {
            "narrative":      "narrative",
            "political":      "political",   # re-runs political+military, not source
            "end_with_error": END,
        },
    )

    # ── Terminal ──────────────────────────────────────────────────────────────
    graph.add_edge("narrative", END)

    return graph.compile()


def run_query(query: str) -> AgentState:
    """
    Public entry point. Initialises state and runs the full graph.

    Args:
        query: The user's historical question, passed unchanged into state.

    Returns:
        Final AgentState after graph execution completes.
        Always check state["error"] before reading state["narrative_output"].
    """
    app = build_graph()

    # All keys must be explicitly set — LangGraph requires every key present at invoke().
    initial_state: AgentState = {
        "query":               query,
        "query_id":            str(uuid.uuid4()),
        "retrieved_chunks":    [],
        "source_output":       None,
        "political_output":    None,
        "military_output":     None,
        "critique_output":     None,
        "narrative_output":    None,
        "critique_loop_count": 0,
        "route_to":            None,
        "critique_passed":     False,
        "debug_log":           [],
        "error":               None,
    }

    return app.invoke(initial_state)
