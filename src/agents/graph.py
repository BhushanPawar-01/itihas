"""
LangGraph StateGraph — wires all five nodes and exposes two public entry points:

  run_query(query)            — blocking, returns final AgentState
  run_query_streaming(query)  — generator, yields SSE-formatted strings as each
                                node completes; used by the streaming endpoint in
                                backend/routes/query.py

Topology (FIXED):
  START → source → political ──┐
                 → military  ──┴→ critique → route_after_critique
                                               ↓ "narrative"        → narrative → END
                                               ↓ "rebuttal_fan_out" → political ──┐
                                                                     → military  ──┴→ critique
                                               ↓ "end_with_error"   → END

The rebuttal_fan_out node is a no-op junction. A conditional edge cannot
return a list directly (that syntax only works for the initial source fan-out).
We need this intermediate node so LangGraph can fan out to BOTH political AND
military on every rebuttal loop — not just political as before.
"""

from __future__ import annotations

import json
import uuid

from langgraph.graph import END, StateGraph

from src.agents.state import AgentState
from src.agents.source_agent    import source_node
from src.agents.political_agent import political_node
from src.agents.military_agent  import military_node
from src.agents.critique_agent  import critique_node
from src.agents.narrative_agent import narrative_node


# ── Agent display metadata (for the frontend DebateFeed) ──────────────────────

_AGENT_LABELS = {
    "source":           "Source Agent — retrieving evidence",
    "political":        "Political Agent — analysing power dynamics",
    "military":         "Military Agent — checking physical plausibility",
    "critique":         "Critique Agent — detecting contradictions",
    "narrative":        "Narrative Agent — synthesising final answer",
    "rebuttal_fan_out": "Rebuttal — both agents re-running",
}


# ── No-op fan-out node ─────────────────────────────────────────────────────────

def rebuttal_fan_out_node(state: AgentState) -> dict:
    """
    No-op node — returns no state changes.
    Exists solely as a junction point so the conditional edge from critique
    can route here, and two unconditional edges then fan out to both
    political and military in parallel.
    """
    return {}


# ── Routing functions ──────────────────────────────────────────────────────────

def route_after_source(state: AgentState) -> str | list[str]:
    if state.get("error"):
        return "end_with_error"
    return ["political", "military"]


def route_after_critique(state: AgentState) -> str:
    if state.get("error"):
        return "end_with_error"
    if state.get("critique_passed"):
        return "narrative"
    return "rebuttal_fan_out"   # ← was "political"; now routes to fan-out junction


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    graph.add_node("source",           source_node)
    graph.add_node("political",        political_node)
    graph.add_node("military",         military_node)
    graph.add_node("critique",         critique_node)
    graph.add_node("narrative",        narrative_node)
    graph.add_node("rebuttal_fan_out", rebuttal_fan_out_node)   # ← new

    # ── Entry ──────────────────────────────────────────────────────────────────
    graph.set_entry_point("source")

    # ── Source → political + military in parallel ──────────────────────────────
    graph.add_conditional_edges(
        "source",
        route_after_source,
        {
            "political":      "political",
            "military":       "military",
            "end_with_error": END,
        },
    )

    # ── Fan-in: both converge on critique ─────────────────────────────────────
    graph.add_edge("political", "critique")
    graph.add_edge("military",  "critique")

    # ── Critique: pass → narrative, loop → rebuttal_fan_out ───────────────────
    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {
            "narrative":        "narrative",
            "rebuttal_fan_out": "rebuttal_fan_out",   # ← replaces direct "political"
            "end_with_error":   END,
        },
    )

    # ── Rebuttal fan-out → political + military in parallel ────────────────────
    graph.add_edge("rebuttal_fan_out", "political")
    graph.add_edge("rebuttal_fan_out", "military")

    # ── Terminal ───────────────────────────────────────────────────────────────
    graph.add_edge("narrative", END)

    return graph.compile()


# ── Shared initial state builder ───────────────────────────────────────────────

def _initial_state(query: str) -> AgentState:
    return {
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


# ── Public entry points ────────────────────────────────────────────────────────

def run_query(query: str) -> AgentState:
    """
    Blocking entry point — runs the full graph and returns final AgentState.
    query.py POST endpoint uses this.
    """
    app = build_graph()
    return app.invoke(_initial_state(query))


def run_query_streaming(query: str):
    """
    Generator — yields SSE-formatted strings as each node completes.
    Used by POST /query/stream in backend/routes/query.py.

    Each yielded value is a complete SSE message:
        data: <json>\n\n

    Event payload shape:
        {
          "type":    "node_complete" | "rebuttal" | "done" | "error",
          "agent":   "political_agent",
          "label":   "Political Agent — analysing power dynamics",
          "loop":    0,
          "content": "...first 400 chars of agent output...",
          "error":   null | "error string"
        }
    """
    app   = build_graph()
    state = _initial_state(query)

    try:
        for chunk in app.stream(state):
            for node_name, update in chunk.items():
                if node_name in ("__end__", "rebuttal_fan_out"):
                    continue

                error   = update.get("error")
                loop    = update.get("critique_loop_count", 0)

                content = ""
                for key in ("source_output", "political_output", "military_output",
                            "critique_output", "narrative_output"):
                    val = update.get(key)
                    if val and isinstance(val, dict) and val.get("content"):
                        raw     = val["content"]
                        content = raw[:400] + ("…" if len(raw) > 400 else "")
                        break

                # Rebuttal rounds: political or military firing after loop > 0
                event_type = "node_complete"
                if node_name in ("political", "military") and loop > 0:
                    event_type = "rebuttal"

                payload = {
                    "type":    event_type,
                    "agent":   f"{node_name}_agent",
                    "label":   _AGENT_LABELS.get(node_name, node_name),
                    "loop":    loop,
                    "content": content,
                    "error":   error,
                }
                yield f"data: {json.dumps(payload)}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as exc:
        import traceback
        tb      = traceback.format_exc()
        payload = {"type": "error", "content": str(exc), "traceback": tb[:500]}
        yield f"data: {json.dumps(payload)}\n\n"