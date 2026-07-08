"""
LangGraph node — Political Agent.

Reads source_output["content"] (JSON written by source_node), groups chunks by
bias_tag, and produces a three-section political analysis: BENEFICIARY, OMISSIONS,
INTERPRETATION.

On rebuttal rounds (critique_loop_count > 0): injects the specific contradictions
identified by critique_agent and the current military_output, instructing the model
to address the named disagreement — not rerun blind.

If conversation_context is present in state, it is prepended to the prompt so the
agent is aware of what was established in prior turns.

LLM calls: call(). No other LLM import in this file.
"""

from __future__ import annotations

import json
import time
import traceback
from collections import defaultdict

from src.agents.state import AgentOutput, AgentState
from src.utils.llm_client import call
from src.utils.logger import get_logger

log = get_logger(__name__)

SYSTEM = (
    "You are a political historian specialising in colonial India 1600-1947. "
    "Analyse the following evidence chunks. Identify: "
    "(1) who benefits from each source's framing, "
    "(2) what is omitted from each perspective, "
    "(3) the most defensible political interpretation given all sources combined. "
    "Do not narrate. Output exactly three labelled sections: "
    "BENEFICIARY, OMISSIONS, INTERPRETATION."
)

REBUTTAL_SYSTEM = (
    "You are a political historian specialising in colonial India 1600-1947. "
    "You have already produced a political analysis. A critique agent has identified "
    "specific contradictions between your analysis and the military analysis. "
    "You must now produce a revised political analysis that directly addresses each "
    "named contradiction. "
    "Either revise your prior position with stated justification from the evidence, "
    "or explain precisely why the apparent conflict is not a real contradiction. "
    "Do not simply restate your original analysis unchanged — that is not acceptable. "
    "Output exactly three labelled sections: BENEFICIARY, OMISSIONS, INTERPRETATION."
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


def political_node(state: AgentState) -> dict:
    t0 = time.monotonic()
    loop_count  = state.get("critique_loop_count", 0)
    is_rebuttal = loop_count > 0 and state.get("critique_output") is not None

    if state.get("source_output") is None:
        msg = "political_node: source_output is None"
        log.error(msg)
        return {
            "error":     msg,
            "debug_log": ["political_agent: FAILED — source_output missing"],
        }

    try:
        try:
            chunks: list[dict] = json.loads(state["source_output"]["content"])
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(
                f"political_node: failed to parse source_output content — {exc}"
            ) from exc

        if not chunks:
            raise ValueError("political_node: source_output content is an empty list")

        groups: dict[str, list[dict]] = defaultdict(list)
        for chunk in chunks:
            groups[chunk["bias_tag"]].append(chunk)

        user_parts: list[str] = []
        for tag in sorted(groups.keys()):
            group_text = "\n".join(c["text"][:300] for c in groups[tag])
            user_parts.append(f"## {tag}\n{group_text}")

        evidence_section = "\n\n".join(user_parts)
        ctx_prefix       = _context_prefix(state)

        if is_rebuttal:
            contradictions  = _parse_contradictions(state.get("critique_output"))
            military_output = state.get("military_output")
            military_text   = military_output["content"] if military_output else "(not available)"

            contradiction_lines = "\n".join(
                f"  {i+1}. {c}" for i, c in enumerate(contradictions)
            ) if contradictions else "  (no specific contradictions listed — revise for clarity)"

            user_content = (
                f"{ctx_prefix}"
                f"Query: {state['query']}\n\n"
                f"--- EVIDENCE ---\n{evidence_section}\n\n"
                f"--- YOUR PRIOR POLITICAL ANALYSIS ---\n"
                f"{state['political_output']['content']}\n\n"
                f"--- MILITARY AGENT'S CURRENT ANALYSIS ---\n{military_text}\n\n"
                f"--- CONTRADICTIONS IDENTIFIED BY CRITIQUE (round {loop_count}) ---\n"
                f"{contradiction_lines}\n\n"
                f"Now produce your revised political analysis addressing these contradictions."
            )
            prompt = f"{REBUTTAL_SYSTEM}\n\n{user_content}"
            log.info(
                "political_agent: rebuttal round %d — %d contradiction(s) to address",
                loop_count, len(contradictions),
            )
        else:
            user_content = f"{ctx_prefix}{evidence_section}\n\nQuery: {state['query']}"
            prompt = f"{SYSTEM}\n\n{user_content}"

        response = call(prompt, max_tokens=800, temperature=0.3)

        output: AgentOutput = {
            "agent_name": "political",
            "content":    response,
            "confidence": 0.8,
            "citations":  [c["doc_id"] for c in chunks],
        }

        duration_ms  = round((time.monotonic() - t0) * 1000)
        round_label  = f"rebuttal_round={loop_count}" if is_rebuttal else "first_pass"
        context_note = f" [ctx={len(state.get('conversation_context',''))}chars]" \
                       if state.get("conversation_context") else ""
        debug_entry  = (
            f"political_agent: {round_label}, "
            f"chunks={len(chunks)}, "
            f"bias_tags={sorted(groups.keys())}, "
            f"duration_ms={duration_ms}{context_note}"
        )

        return {
            "political_output": output,
            "debug_log":        [debug_entry],
        }

    except Exception as exc:
        tb = traceback.format_exc()
        log.error("political_agent: FAILED — %s\n%s", exc, tb)
        return {
            "error":     str(exc),
            "debug_log": [f"political_agent: FAILED — {exc}"],
        }