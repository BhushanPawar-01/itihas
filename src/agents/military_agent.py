"""
LangGraph node — Military Agent.

Reads source_output["content"] (JSON written by source_node) and checks physical
plausibility of claims: terrain, logistics, order of battle, period doctrine.

On rebuttal rounds (critique_loop_count > 0): injects the specific contradictions
identified by critique_agent and the current political_output, instructing the model
to address the named disagreement — not rerun blind.

If conversation_context is present in state, it is prepended to the prompt so the
agent is aware of what was established in prior turns.

LLM calls: call(). No other LLM import in this file.
Independent of political_agent — do not import from it.
"""

from __future__ import annotations

import json
import time
import traceback

from src.agents.state import AgentOutput, AgentState
from src.utils.llm_client import call
from src.utils.logger import get_logger

log = get_logger(__name__)

SYSTEM = (
    "You are a military historian specialising in South Asian warfare 1600-1947. "
    "Analyse the following source chunks for military accuracy. Check: "
    "(1) physical possibility given terrain and period logistics, "
    "(2) order-of-battle consistency — numbers, equipment, command structure, "
    "(3) tactical claims versus documented doctrine of the period. "
    "Flag claims that are militarily implausible. "
    "Output exactly three labelled sections: PLAUSIBLE, IMPLAUSIBLE, UNCERTAIN."
)

REBUTTAL_SYSTEM = (
    "You are a military historian specialising in South Asian warfare 1600-1947. "
    "You have already produced a military analysis. A critique agent has identified "
    "specific contradictions between your analysis and the political analysis. "
    "You must now produce a revised military analysis that directly addresses each "
    "named contradiction. "
    "Either revise your prior position with stated justification from the evidence, "
    "or explain precisely why the apparent conflict is not a real contradiction. "
    "Do not simply restate your original analysis unchanged — that is not acceptable. "
    "Output exactly three labelled sections: PLAUSIBLE, IMPLAUSIBLE, UNCERTAIN."
)


def _parse_contradictions(critique_output: dict | None) -> list[str]:
    """Extract the contradictions list from critique_output. Never raises."""
    if not critique_output:
        return []
    try:
        parsed = json.loads(critique_output["content"])
        return parsed.get("contradictions", [])
    except Exception:
        return []


def _context_prefix(state: AgentState) -> str:
    """
    Return the conversation context block if present, else empty string.
    Prepended to user_content so the model sees prior turns as input to
    reason about, not as a system instruction.
    """
    ctx = state.get("conversation_context", "")
    return f"{ctx}\n\n" if ctx else ""


def military_node(state: AgentState) -> dict:
    t0 = time.monotonic()
    loop_count  = state.get("critique_loop_count", 0)
    is_rebuttal = loop_count > 0 and state.get("critique_output") is not None

    if state.get("source_output") is None:
        msg = "military_node: source_output is None"
        log.error(msg)
        return {
            "error":     msg,
            "debug_log": ["military_agent: FAILED — source_output missing"],
        }

    try:
        try:
            chunks: list[dict] = json.loads(state["source_output"]["content"])
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(
                f"military_node: failed to parse source_output content — {exc}"
            ) from exc

        if not chunks:
            raise ValueError("military_node: source_output content is an empty list")

        chunk_lines = "".join(
            f"[{c['doc_id']}] {c['text'][:300]}\n" for c in chunks
        )
        ctx_prefix = _context_prefix(state)

        if is_rebuttal:
            contradictions   = _parse_contradictions(state.get("critique_output"))
            political_output = state.get("political_output")
            political_text   = political_output["content"] if political_output else "(not available)"

            contradiction_lines = "\n".join(
                f"  {i+1}. {c}" for i, c in enumerate(contradictions)
            ) if contradictions else "  (no specific contradictions listed — revise for clarity)"

            user_content = (
                f"{ctx_prefix}"
                f"Query: {state['query']}\n\n"
                f"--- EVIDENCE ---\n{chunk_lines}\n"
                f"--- YOUR PRIOR MILITARY ANALYSIS ---\n"
                f"{state['military_output']['content']}\n\n"
                f"--- POLITICAL AGENT'S CURRENT ANALYSIS ---\n{political_text}\n\n"
                f"--- CONTRADICTIONS IDENTIFIED BY CRITIQUE (round {loop_count}) ---\n"
                f"{contradiction_lines}\n\n"
                f"Now produce your revised military analysis addressing these contradictions."
            )
            prompt = f"{REBUTTAL_SYSTEM}\n\n{user_content}"
            log.info(
                "military_agent: rebuttal round %d — %d contradiction(s) to address",
                loop_count, len(contradictions),
            )
        else:
            user_content = f"{ctx_prefix}{chunk_lines}\nQuery: {state['query']}"
            prompt = f"{SYSTEM}\n\n{user_content}"

        response = call(prompt, max_tokens=800, temperature=0.2)

        output: AgentOutput = {
            "agent_name": "military",
            "content":    response,
            "confidence": 0.8,
            "citations":  [c["doc_id"] for c in chunks],
        }

        duration_ms  = round((time.monotonic() - t0) * 1000)
        round_label  = f"rebuttal_round={loop_count}" if is_rebuttal else "first_pass"
        context_note = f" [ctx={len(state.get('conversation_context',''))}chars]" \
                       if state.get("conversation_context") else ""
        debug_entry  = (
            f"military_agent: {round_label}, "
            f"chunks={len(chunks)}, "
            f"duration_ms={duration_ms}{context_note}"
        )

        return {
            "military_output": output,
            "debug_log":       [debug_entry],
        }

    except Exception as exc:
        tb = traceback.format_exc()
        log.error("military_agent: FAILED — %s\n%s", exc, tb)
        return {
            "error":     str(exc),
            "debug_log": [f"military_agent: FAILED — {exc}"],
        }