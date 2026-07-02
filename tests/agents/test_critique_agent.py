"""
tests/agents/test_critique_agent.py

Tests for agents/critique_agent.py — critique_node().

All tests are fully mocked — no real API calls.
"""

import json
from unittest.mock import patch

import pytest

from src.agents.critique_agent import MAX_CRITIQUE_LOOPS, critique_node

_PASS_RESPONSE = json.dumps({
    "contradictions": [],
    "confidence":     0.85,
    "decision":       "PASS",
    "critique_notes": "Outputs are consistent.",
})

_LOOP_RESPONSE = json.dumps({
    "contradictions": ["Political agent claims British won; military agent says inconclusive."],
    "confidence":     0.4,
    "decision":       "LOOP",
    "critique_notes": "Material contradiction on battle outcome.",
})

_MALFORMED_RESPONSE = "Here is my analysis: the outputs seem somewhat consistent overall."

_FAKE_SOURCE_OUTPUT = {
    "agent_name": "source",
    "content":    json.dumps([{
        "doc_id": "ia_trial_19451107_001", "chunk_index": 0,
        "text": "Shah Nawaz Khan commanded.", "bias_tag": "british_legal", "score": 0.9,
    }]),
    "confidence": 1.0,
    "citations":  ["ia_trial_19451107_001"],
}

_FAKE_POLITICAL_OUTPUT = {
    "agent_name": "political",
    "content":    "BENEFICIARY: British\nOMISSIONS: INA voices\nINTERPRETATION: Colonial framing.",
    "confidence": 0.8,
    "citations":  ["ia_trial_19451107_001"],
}

_FAKE_MILITARY_OUTPUT = {
    "agent_name": "military",
    "content":    "PLAUSIBLE: March distances\nIMPLAUSIBLE: None\nUNCERTAIN: Supply lines.",
    "confidence": 0.8,
    "citations":  ["ia_trial_19451107_001"],
}

_BASE_STATE = {
    "query":               "Who commanded the INA at the Battle of Imphal",
    "query_id":            "test-critique-001",
    "retrieved_chunks":    [],
    "source_output":       _FAKE_SOURCE_OUTPUT,
    "political_output":    _FAKE_POLITICAL_OUTPUT,
    "military_output":     _FAKE_MILITARY_OUTPUT,
    "critique_output":     None,
    "narrative_output":    None,
    "critique_loop_count": 0,
    "route_to":            None,
    "critique_passed":     False,
    "debug_log":           [],
    "error":               None,
}


# ── Test 1 ────────────────────────────────────────────────────────────────────

def test_return_keys_present():
    """critique_node must return dict with 'critique_output' and 'critique_passed'."""
    with patch("src.agents.critique_agent.call", return_value=_PASS_RESPONSE) as mock_call:
        result = critique_node(_BASE_STATE)

    assert isinstance(result, dict), "must return dict"
    assert "critique_output" in result, "missing key: critique_output"
    assert "critique_passed" in result, "missing key: critique_passed"
    assert "backend" not in mock_call.call_args.kwargs, "must use llm_client default backend"


# ── Test 2 ────────────────────────────────────────────────────────────────────

def test_loop_limit_forces_pass_without_llm_call():
    """
    When critique_loop_count is already MAX_CRITIQUE_LOOPS - 1 (so new_count hits limit),
    node must return critique_passed=True without calling call().
    """
    at_limit_state = {**_BASE_STATE, "critique_loop_count": MAX_CRITIQUE_LOOPS - 1}

    with patch("src.agents.critique_agent.call") as mock_call:
        result = critique_node(at_limit_state)
        mock_call.assert_not_called()

    assert result.get("critique_passed") is True, "must force PASS at loop limit"
    assert result.get("critique_loop_count") == MAX_CRITIQUE_LOOPS
    assert "forced PASS" in result["debug_log"][0]


# ── Test 3 ────────────────────────────────────────────────────────────────────

def test_missing_upstream_output_returns_error_no_raise():
    """
    If any upstream output is None, node must return dict with 'error' key.
    Must not raise. critique_output must not be present.
    """
    for missing_key in ("source_output", "political_output", "military_output"):
        bad_state = {**_BASE_STATE, missing_key: None}

        with patch("src.agents.critique_agent.call", return_value=_PASS_RESPONSE):
            result = critique_node(bad_state)

        assert isinstance(result, dict), f"must return dict when {missing_key} is None"
        assert "error" in result, f"must set 'error' when {missing_key} is None"
        assert "critique_output" not in result, (
            f"'critique_output' must not be present when {missing_key} is None"
        )


# ── Test 4 ────────────────────────────────────────────────────────────────────

def test_critique_passed_true_on_pass_decision():
    """critique_passed must be True when LLM returns decision='PASS'."""
    with patch("src.agents.critique_agent.call", return_value=_PASS_RESPONSE):
        result = critique_node(_BASE_STATE)

    if result.get("error"):
        pytest.fail(f"critique_node returned error: {result['error']}")

    assert result["critique_passed"] is True, (
        f"expected critique_passed=True for PASS decision, got {result['critique_passed']}"
    )


# ── Test 5 ────────────────────────────────────────────────────────────────────

def test_critique_passed_false_on_loop_decision():
    """critique_passed must be False when LLM returns decision='LOOP'."""
    with patch("src.agents.critique_agent.call", return_value=_LOOP_RESPONSE):
        result = critique_node(_BASE_STATE)

    if result.get("error"):
        pytest.fail(f"critique_node returned error: {result['error']}")

    assert result["critique_passed"] is False, (
        f"expected critique_passed=False for LOOP decision, got {result['critique_passed']}"
    )


# ── Test 6 ────────────────────────────────────────────────────────────────────

def test_malformed_json_defaults_to_pass_confidence_half():
    """
    When call() returns non-JSON text, node must not raise.
    critique_passed must default to True; confidence must be 0.5.
    """
    with patch("src.agents.critique_agent.call", return_value=_MALFORMED_RESPONSE):
        result = critique_node(_BASE_STATE)

    assert isinstance(result, dict), "must return dict on malformed JSON"
    assert "error" not in result, (
        f"must not set error key on JSON parse failure, got: {result.get('error')}"
    )
    assert result.get("critique_passed") is True, (
        "must default critique_passed=True on malformed JSON"
    )
    assert result["critique_output"]["confidence"] == 0.5, (
        f"expected confidence=0.5 on parse failure, got {result['critique_output']['confidence']}"
    )
