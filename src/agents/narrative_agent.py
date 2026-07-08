"""
LangGraph node — Narrative Agent.

Synthesises validated outputs from political, military, and critique agents into
a four-section human-readable response with citations and confidence scores.

If conversation_context is present in state, it is prepended to the prompt so
the synthesis reflects what was already established in prior turns — the agent
can avoid re-explaining settled facts and instead focus on what the current
query adds.

LLM calls: call() only; llm_client defaults agents to OpenAI.
Runs only after critique_passed = True (enforced by graph edge; asserted here).
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
    "You are a historian synthesising a multi-perspective analysis of Indian "
    "military and political history 1600-1947. "
    "Write a structured response with exactly these four section headers "
    "(use them verbatim as markdown headings):\n"
    "## Political Reality\n"
    "## Military Reality\n"
    "## Ground Truth vs Propaganda\n"
    "## Confidence Assessment\n"
    "Cite sources by doc_id in square brackets, e.g. [ia_trial_19451107_001]. "
    "Be specific. State what the evidence shows. "
    "Reserve hedging for the Confidence Assessment section only.\n"
    "If a conversation context is provided, do not re-explain facts already "
    "established in prior turns -- build on them and focus on what the current "
    "query adds."
)


def narrative_node(state: AgentState) -> dict:
    t0 = time.monotonic()

    if not state.get("critique_passed"):
        msg = "narrative_node called before critique passed"
        log.error(msg)
        return {
            "error":     msg,
            "debug_log": ["narrative_agent: FAILED — critique_passed is False"],
        }

    try:
        confidence = 0.7
        try:
            critique_content = state["critique_output"]["content"]
            confidence = float(json.loads(critique_content)["confidence"])
        except Exception:
            log.warning(
                "narrative_agent: could not parse confidence from critique_output — using %.1f",
                confidence,
            )

        all_ids = sorted(set(
            state["source_output"]["citations"] +
            state["political_output"]["citations"] +
            state["military_output"]["citations"]
        ))

        conversation_context = state.get("conversation_context", "")
        ctx_section = ""
        if conversation_context:
            ctx_section = f"{conversation_context}\n\n"
            log.info(
                "narrative_agent: injecting conversation_context (%d chars)",
                len(conversation_context),
            )

        user_content = (
            f"{ctx_section}"
            f"Query: {state['query']}\n\n"
            f"Political Analysis:\n{state['political_output']['content']}\n\n"
            f"Military Analysis:\n{state['military_output']['content']}\n\n"
            f"Critique Notes:\n{state['critique_output']['content']}\n\n"
            f"Available doc_ids for citation: {', '.join(all_ids)}"
        )
        prompt = f"{SYSTEM}\n\n{user_content}"

        response = call(prompt, max_tokens=1500, temperature=0.2)

        output: AgentOutput = {
            "agent_name": "narrative",
            "content":    response,
            "confidence": confidence,
            "citations":  all_ids,
        }

        duration_ms  = round((time.monotonic() - t0) * 1000)
        context_note = f" [ctx={len(conversation_context)}chars]" if conversation_context else ""
        return {
            "narrative_output": output,
            "debug_log": [
                f"narrative_agent: response_chars={len(response)}, "
                f"citations={len(all_ids)}, duration_ms={duration_ms}{context_note}"
            ],
        }

    except Exception as exc:
        tb = traceback.format_exc()
        log.error("narrative_agent: FAILED — %s\n%s", exc, tb)
        return {
            "error":     str(exc),
            "debug_log": [f"narrative_agent: FAILED — {exc}"],
        }