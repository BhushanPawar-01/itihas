"""
LangGraph node — Critique Agent.

Reads source_output, political_output, military_output; detects contradictions;
assigns a confidence score; decides PASS (proceed to narrative) or LOOP (re-run
political + military).

LLM calls: call() only; llm_client defaults agents to OpenAI.
Meta-reasoning uses the general model, not the domain fine-tune.
"""

from __future__ import annotations

import json
import time
import traceback

from src.agents.state import AgentOutput, AgentState
from src.utils.llm_client import call
from src.utils.logger import get_logger

log = get_logger(__name__)

MAX_CRITIQUE_LOOPS = 3

SYSTEM = (
    "You are a critical reasoning engine. Three specialist agents have analysed "
    "the same historical query. Identify contradictions between their outputs. "
    "Rate overall confidence from 0.0 to 1.0. "
    "Decide: PASS if outputs are consistent enough to synthesise, "
    "or LOOP if outputs contradict on a material fact that must be resolved. "
    "Respond with ONLY valid JSON, no preamble, no markdown fences:\n"
    '{"contradictions": ["..."], "confidence": 0.0, '
    '"decision": "PASS", "critique_notes": "..."}'
)


def critique_node(state: AgentState) -> dict:
    """
    LangGraph node — returns partial state update dict.

    Return keys (success):
        critique_output      : AgentOutput
        critique_passed      : bool
        critique_loop_count  : int
        debug_log            : list[str]

    Return keys (loop-break):
        critique_loop_count  : int
        critique_passed      : bool  (True)
        debug_log            : list[str]

    Return keys (failure):
        error     : str
        debug_log : list[str]
    """
    t0 = time.monotonic()

    # ── 1. Increment loop count — always, before any other check ─────────────
    new_count = state["critique_loop_count"] + 1

    if new_count >= MAX_CRITIQUE_LOOPS:
        log.warning("critique_agent: loop limit reached (%d) — forcing PASS", new_count)
        return {
            "critique_loop_count": new_count,
            "critique_passed":     True,
            "debug_log":           ["critique_agent: forced PASS at loop limit"],
        }

    # ── 2. Assert upstream outputs are all present ────────────────────────────
    missing = [
        name for name, val in (
            ("source_output",   state.get("source_output")),
            ("political_output", state.get("political_output")),
            ("military_output",  state.get("military_output")),
        )
        if val is None
    ]
    if missing:
        log.error("critique_agent: missing upstream output(s): %s", missing)
        return {
            "error":     "critique_node: upstream output missing",
            "debug_log": ["critique_agent: FAILED — missing upstream output"],
        }

    try:
        # ── 3. Build prompt ───────────────────────────────────────────────────
        user_content = (
            f"Query: {state['query']}\n\n"
            f"SOURCE OUTPUT (first 500 chars): {state['source_output']['content'][:500]}\n\n"
            f"POLITICAL OUTPUT: {state['political_output']['content']}\n\n"
            f"MILITARY OUTPUT: {state['military_output']['content']}"
        )
        prompt = f"{SYSTEM}\n\n{user_content}"

        # ── 4. LLM call — llm_client default backend for meta-reasoning ───────
        raw_response = call(prompt, max_tokens=512, temperature=0.1)

        # ── 5. Parse JSON — degrade gracefully on malformed output ────────────
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            log.warning(
                "critique_agent: JSON parse failed — defaulting to PASS. Raw response: %r",
                raw_response[:200],
            )
            parsed = {
                "contradictions": [],
                "confidence":     0.5,
                "decision":       "PASS",
                "critique_notes": "JSON parse failed",
            }

        # ── 6. Resolve decision ───────────────────────────────────────────────
        critique_passed = parsed.get("decision", "PASS").upper() == "PASS"

        # ── 7. Build output ───────────────────────────────────────────────────
        output: AgentOutput = {
            "agent_name": "critique",
            "content":    raw_response,
            "confidence": float(parsed.get("confidence", 0.5)),
            "citations":  [],
        }

        duration_ms = round((time.monotonic() - t0) * 1000)
        contradictions = parsed.get("contradictions", [])
        debug_entry = (
            f"critique_agent: loop={new_count}, "
            f"decision={parsed.get('decision', 'PASS')}, "
            f"confidence={parsed.get('confidence', 0.5)}, "
            f"contradictions={len(contradictions)}, "
            f"duration_ms={duration_ms}"
        )

        return {
            "critique_output":     output,
            "critique_passed":     critique_passed,
            "critique_loop_count": new_count,
            "debug_log":           [debug_entry],
        }

    except Exception as exc:
        tb = traceback.format_exc()
        log.error("critique_agent: FAILED — %s\n%s", exc, tb)
        return {
            "error":     str(exc),
            "debug_log": [f"critique_agent: FAILED — {exc}"],
        }
