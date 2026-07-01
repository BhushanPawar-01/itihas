"""
LangGraph node — Narrative Agent.

Synthesises validated outputs from political, military, and critique agents into
a four-section human-readable response with citations and confidence scores.

LLM calls: call(backend="hf") — Mistral via HF API, same as critique.
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
    "military and political history 1600–1947. "
    "Write a structured response with exactly these four section headers "
    "(use them verbatim as markdown headings):\n"
    "## Political Reality\n"
    "## Military Reality\n"
    "## Ground Truth vs Propaganda\n"
    "## Confidence Assessment\n"
    "Cite sources by doc_id in square brackets, e.g. [ia_trial_19451107_001]. "
    "Be specific. State what the evidence shows. "
    "Reserve hedging for the Confidence Assessment section only."
)


def narrative_node(state: AgentState) -> dict:
    """
    LangGraph node — returns partial state update dict.

    Return keys (success):
        narrative_output : AgentOutput
        debug_log        : list[str]

    Return keys (failure):
        error     : str
        debug_log : list[str]
    """
    t0 = time.monotonic()

    # ── 1. Guard: critique must have passed ───────────────────────────────────
    if not state.get("critique_passed"):
        msg = "narrative_node called before critique passed"
        log.error(msg)
        return {
            "error":     msg,
            "debug_log": ["narrative_agent: FAILED — critique_passed is False"],
        }

    try:
        # ── 2. Parse confidence from critique_output ──────────────────────────
        confidence = 0.7  # fallback
        try:
            critique_content = state["critique_output"]["content"]
            confidence = float(json.loads(critique_content)["confidence"])
        except Exception:
            log.warning(
                "narrative_agent: could not parse confidence from critique_output — using %.1f",
                confidence,
            )

        # ── 3. Collect citations — union, deduplicated, sorted ────────────────
        all_ids = sorted(set(
            state["source_output"]["citations"] +
            state["political_output"]["citations"] +
            state["military_output"]["citations"]
        ))

        # ── 4. Build prompt ───────────────────────────────────────────────────
        user_content = (
            f"Query: {state['query']}\n\n"
            f"Political Analysis:\n{state['political_output']['content']}\n\n"
            f"Military Analysis:\n{state['military_output']['content']}\n\n"
            f"Critique Notes:\n{state['critique_output']['content']}\n\n"
            f"Available doc_ids for citation: {', '.join(all_ids)}"
        )
        prompt = f"{SYSTEM}\n\n{user_content}"

        # ── 5. LLM call — HF/Mistral for synthesis ────────────────────────────
        response = call(prompt, backend="hf", max_tokens=1500, temperature=0.4)

        # ── 6. Build output ───────────────────────────────────────────────────
        output: AgentOutput = {
            "agent_name": "narrative",
            "content":    response,
            "confidence": confidence,
            "citations":  all_ids,
        }

        duration_ms = round((time.monotonic() - t0) * 1000)
        return {
            "narrative_output": output,
            "debug_log": [
                f"narrative_agent: response_chars={len(response)}, "
                f"citations={len(all_ids)}, duration_ms={duration_ms}"
            ],
        }

    except Exception as exc:
        tb = traceback.format_exc()
        log.error("narrative_agent: FAILED — %s\n%s", exc, tb)
        return {
            "error":     str(exc),
            "debug_log": [f"narrative_agent: FAILED — {exc}"],
        }