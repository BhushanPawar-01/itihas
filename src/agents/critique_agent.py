"""
LangGraph node — Critique Agent.

Reads source_output, political_output, military_output; detects contradictions;
assigns a confidence score; decides PASS or LOOP.

LOOP decision logic (tightened):
  - PASS if confidence >= 0.55  (was: only on explicit "PASS" decision)
  - PASS if no material contradictions found
  - PASS if this is loop >= 2 (critique has already given agents two chances;
    forcing another loop rarely resolves anything and wastes tokens)
  - LOOP only if confidence < 0.55 AND contradictions are flagged as material
    AND loop count < 2

This prevents the common failure mode where critique finds a minor or
irresolvable contradiction and loops all three rounds before forcing PASS.
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
# Only loop if confidence is below this AND material contradictions exist
LOOP_CONFIDENCE_THRESHOLD = 0.55

SYSTEM = (
    "You are a critical reasoning engine. Three specialist agents have analysed "
    "the same historical query. Your job:\n"
    "1. Identify MATERIAL contradictions — factual claims that directly conflict "
    "   and would change the historical conclusion if resolved differently. "
    "   Minor differences in emphasis or framing are NOT material.\n"
    "2. Rate overall confidence from 0.0 to 1.0 based on source quality and "
    "   agreement between agents.\n"
    "3. Decide PASS or LOOP:\n"
    "   - PASS: outputs are consistent enough to synthesise, OR contradictions "
    "     are minor/irresolvable from available evidence.\n"
    "   - LOOP: a specific material factual contradiction exists that the agents "
    "     could plausibly resolve if they address each other's reasoning directly. "
    "     Only choose LOOP if you can name the exact claim in dispute.\n"
    "Respond with ONLY valid JSON, no preamble, no markdown fences:\n"
    '{"contradictions": ["..."], "material": true, "confidence": 0.0, '
    '"decision": "PASS", "critique_notes": "..."}'
)


def critique_node(state: AgentState) -> dict:
    t0        = time.monotonic()
    new_count = state["critique_loop_count"] + 1

    # ── Hard loop limit ───────────────────────────────────────────────────────
    if new_count >= MAX_CRITIQUE_LOOPS:
        log.warning(
            "critique_agent: loop limit reached (%d) — forcing PASS", new_count
        )
        return {
            "critique_loop_count": new_count,
            "critique_passed":     True,
            "debug_log": [f"critique_agent: forced PASS at loop limit {new_count}"],
        }

    # ── Assert upstream outputs present ───────────────────────────────────────
    missing = [
        name for name, val in (
            ("source_output",    state.get("source_output")),
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
        # ── Build prompt ──────────────────────────────────────────────────────
        user_content = (
            f"Query: {state['query']}\n\n"
            f"SOURCE OUTPUT (first 500 chars): {state['source_output']['content'][:500]}\n\n"
            f"POLITICAL OUTPUT:\n{state['political_output']['content']}\n\n"
            f"MILITARY OUTPUT:\n{state['military_output']['content']}"
        )
        prompt = f"{SYSTEM}\n\n{user_content}"

        raw_response = call(prompt, max_tokens=512, temperature=0.1)

        # ── Parse JSON ────────────────────────────────────────────────────────
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            log.warning(
                "critique_agent: JSON parse failed — defaulting to PASS. Raw: %r",
                raw_response[:200],
            )
            parsed = {
                "contradictions": [],
                "material":       False,
                "confidence":     0.6,
                "decision":       "PASS",
                "critique_notes": "JSON parse failed — defaulting to PASS",
            }

        confidence    = float(parsed.get("confidence", 0.6))
        contradictions = parsed.get("contradictions", [])
        is_material   = bool(parsed.get("material", False))
        llm_decision  = parsed.get("decision", "PASS").upper()

        # ── Resolve PASS/LOOP with tightened logic ────────────────────────────
        # LLM says LOOP, but we override to PASS if:
        #   (a) confidence is already acceptable, or
        #   (b) contradictions are flagged non-material, or
        #   (c) we've already looped once (given agents one chance to rebuttal)
        if llm_decision == "LOOP":
            if confidence >= LOOP_CONFIDENCE_THRESHOLD:
                critique_passed = True
                override_reason = f"confidence {confidence:.2f} >= threshold — overriding LOOP to PASS"
            elif not is_material:
                critique_passed = True
                override_reason = "contradictions flagged non-material — overriding LOOP to PASS"
            elif new_count >= 2:
                critique_passed = True
                override_reason = f"already looped {new_count-1}x — overriding LOOP to PASS"
            else:
                critique_passed = False
                override_reason = None
        else:
            critique_passed = True
            override_reason = None

        # ── Build output ──────────────────────────────────────────────────────
        output: AgentOutput = {
            "agent_name": "critique",
            "content":    raw_response,
            "confidence": confidence,
            "citations":  [],
        }

        duration_ms = round((time.monotonic() - t0) * 1000)
        debug_entry = (
            f"critique_agent: loop={new_count}, "
            f"llm_decision={llm_decision}, "
            f"final_passed={critique_passed}, "
            f"confidence={confidence:.2f}, "
            f"material={is_material}, "
            f"contradictions={len(contradictions)}"
            + (f", override='{override_reason}'" if override_reason else "")
            + f", duration_ms={duration_ms}"
        )
        log.info(debug_entry)

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