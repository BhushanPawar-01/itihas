"""
LangGraph node — Political Agent.

Reads source_output["content"] (JSON written by source_node), groups chunks by
bias_tag, and produces a three-section political analysis: BENEFICIARY, OMISSIONS,
INTERPRETATION.

LLM calls: call. No other LLM import in this file.
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


def political_node(state: AgentState) -> dict:
    """
    LangGraph node — returns partial state update dict.

    Return keys (success):
        political_output : AgentOutput
        debug_log        : list[str]  — single entry, appended by LangGraph reducer

    Return keys (failure):
        error     : str
        debug_log : list[str]
    """
    t0 = time.monotonic()

    # ── 1. Guard: source_output must exist ───────────────────────────────────
    if state.get("source_output") is None:
        msg = "political_node: source_output is None"
        log.error(msg)
        return {
            "error":     msg,
            "debug_log": ["political_agent: FAILED — source_output missing"],
        }

    try:
        # ── 2. Parse source evidence ─────────────────────────────────────────
        try:
            chunks: list[dict] = json.loads(state["source_output"]["content"])
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(
                f"political_node: failed to parse source_output content — {exc}"
            ) from exc

        if not chunks:
            raise ValueError("political_node: source_output content is an empty list")

        # ── 3. Group by bias_tag, build user_content ─────────────────────────
        groups: dict[str, list[dict]] = defaultdict(list)
        for chunk in chunks:
            groups[chunk["bias_tag"]].append(chunk)

        user_parts: list[str] = []
        for tag in sorted(groups.keys()):
            group_text = "\n".join(c["text"][:300] for c in groups[tag])
            user_parts.append(f"## {tag}\n{group_text}")

        user_content = "\n\n".join(user_parts) + f"\n\nQuery: {state['query']}"
        prompt = f"{SYSTEM}\n\n{user_content}"

        # ── 4. LLM call ───────────────────────────────────────────────────────
        response = call(prompt, max_tokens=800, temperature=0.3)

        # ── 5. Build output ───────────────────────────────────────────────────
        output: AgentOutput = {
            "agent_name": "political",
            "content":    response,
            "confidence": 0.8,  # fixed; critique_node adjusts
            "citations":  [c["doc_id"] for c in chunks],
        }

        duration_ms = round((time.monotonic() - t0) * 1000)
        sorted_tags = sorted(groups.keys())
        debug_entry = (
            f"political_agent: chunks={len(chunks)}, "
            f"bias_tags={sorted_tags}, "
            f"duration_ms={duration_ms}"
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