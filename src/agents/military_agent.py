"""
LangGraph node — Military Agent.

Reads source_output["content"] (JSON written by source_node) and checks physical
plausibility of claims: terrain, logistics, order of battle, period doctrine.

LLM calls: call() only. No other LLM import in this file.
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
    "You are a military historian specialising in Worldly warfare 1600-1947. "
    "Analyse the following source chunks for military accuracy. Check: "
    "(1) physical possibility given terrain and period logistics, "
    "(2) order-of-battle consistency — numbers, equipment, command structure, "
    "(3) tactical claims versus documented doctrine of the period. "
    "Flag claims that are militarily implausible. "
    "Output exactly three labelled sections: PLAUSIBLE, IMPLAUSIBLE, UNCERTAIN."
)


def military_node(state: AgentState) -> dict:
    """
    LangGraph node — returns partial state update dict.

    Return keys (success):
        military_output : AgentOutput
        debug_log       : list[str]  — single entry, appended by LangGraph reducer

    Return keys (failure):
        error     : str
        debug_log : list[str]
    """
    t0 = time.monotonic()

    # ── 1. Guard: source_output must exist ───────────────────────────────────
    if state.get("source_output") is None:
        msg = "military_node: source_output is None"
        log.error(msg)
        return {
            "error":     msg,
            "debug_log": ["military_agent: FAILED — source_output missing"],
        }

    try:
        # ── 2. Parse source evidence ──────────────────────────────────────────
        try:
            chunks: list[dict] = json.loads(state["source_output"]["content"])
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(
                f"military_node: failed to parse source_output content — {exc}"
            ) from exc

        if not chunks:
            raise ValueError("military_node: source_output content is an empty list")

        # ── 3. Build user_content — per-claim, no bias_tag grouping ──────────
        chunk_lines = "".join(
            f"[{c['doc_id']}] {c['text'][:300]}\n" for c in chunks
        )
        user_content = chunk_lines + f"\nQuery: {state['query']}"
        prompt = f"{SYSTEM}\n\n{user_content}"

        # ── 4. LLM call ───────────────────────────────────────────────────────
        response = call(prompt, max_tokens=800, temperature=0.2)

        # ── 5. Build output ───────────────────────────────────────────────────
        output: AgentOutput = {
            "agent_name": "military",
            "content":    response,
            "confidence": 0.8,  # fixed; critique_node adjusts
            "citations":  [c["doc_id"] for c in chunks],
        }

        duration_ms = round((time.monotonic() - t0) * 1000)
        return {
            "military_output": output,
            "debug_log":       [f"military_agent: chunks={len(chunks)}, duration_ms={duration_ms}"],
        }

    except Exception as exc:
        tb = traceback.format_exc()
        log.error("military_agent: FAILED — %s\n%s", exc, tb)
        return {
            "error":     str(exc),
            "debug_log": [f"military_agent: FAILED — {exc}"],
        }