"""
Tests for agents/narrative_agent.py — narrative_node().

All tests are fully mocked — no real API calls.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from src.agents.narrative_agent import narrative_node

_MOCK_LLM_RESPONSE = (
    "## Political Reality\n"
    "The INA trials exposed the tension between British legal authority and Indian nationalist sentiment. "
    "[ia_trial_19451107_001]\n\n"
    "## Military Reality\n"
    "The INA's advance to Imphal was logistically constrained by monsoon terrain. "
    "[ia_testimony_19451108_001]\n\n"
    "## Ground Truth vs Propaganda\n"
    "British sources overstate INA disorder; nationalist press overstates their operational success.\n\n"
    "## Confidence Assessment\n"
    "Overall confidence: 0.85. Primary sources corroborate on command structure."
)

_CRITIQUE_CONTENT = json.dumps({
    "contradictions": [],
    "confidence":     0.85,
    "decision":       "PASS",
    "critique_notes": "Outputs are consistent.",
})

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
    "citations":  ["ia_trial_19451107_001", "ia_press_19451110_001"],
}

_FAKE_MILITARY_OUTPUT = {
    "agent_name": "military",
    "content":    "PLAUSIBLE: March distances\nIMPLAUSIBLE: None\nUNCERTAIN: Supply lines.",
    "confidence": 0.8,
    "citations":  ["ia_testimony_19451108_001"],
}

_FAKE_CRITIQUE_OUTPUT = {
    "agent_name": "critique",
    "content":    _CRITIQUE_CONTENT,
    "confidence": 0.85,
    "citations":  [],
}

_BASE_STATE = {
    "query":               "Who commanded the INA at the Battle of Imphal",
    "query_id":            "test-narrative-001",
    "retrieved_chunks":    [],
    "source_output":       _FAKE_SOURCE_OUTPUT,
    "political_output":    _FAKE_POLITICAL_OUTPUT,
    "military_output":     _FAKE_MILITARY_OUTPUT,
    "critique_output":     _FAKE_CRITIQUE_OUTPUT,
    "narrative_output":    None,
    "critique_loop_count": 1,
    "route_to":            None,
    "critique_passed":     True,
    "debug_log":           [],
    "error":               None,
}


@pytest.fixture(scope="module")
def node_result():
    """Run narrative_node once with mocked call(). No real LLM or DB."""
    with patch("src.agents.narrative_agent.call", return_value=_MOCK_LLM_RESPONSE):
        result = narrative_node(_BASE_STATE)
    return result


# ── Test 1 ────────────────────────────────────────────────────────────────────

def test_return_key_present(node_result):
    """narrative_node must return a dict containing 'narrative_output'."""
    assert isinstance(node_result, dict), "must return dict"
    assert "narrative_output" in node_result, "missing key: narrative_output"


# ── Test 2 ────────────────────────────────────────────────────────────────────

def test_content_contains_political_reality(node_result):
    """narrative_output['content'] must contain '## Political Reality'."""
    if node_result.get("error"):
        pytest.fail(f"narrative_node returned error: {node_result['error']}")

    content = node_result["narrative_output"]["content"]
    assert "## Political Reality" in content, (
        f"'## Political Reality' not found in content:\n{content[:300]}"
    )


# ── Test 3 ────────────────────────────────────────────────────────────────────

def test_content_contains_military_reality(node_result):
    """narrative_output['content'] must contain '## Military Reality'."""
    if node_result.get("error"):
        pytest.fail(f"narrative_node returned error: {node_result['error']}")

    content = node_result["narrative_output"]["content"]
    assert "## Military Reality" in content, (
        f"'## Military Reality' not found in content:\n{content[:300]}"
    )


# ── Test 4 ────────────────────────────────────────────────────────────────────

def test_citations_nonempty_list_of_strings(node_result):
    """narrative_output['citations'] must be a non-empty list of strings."""
    if node_result.get("error"):
        pytest.fail(f"narrative_node returned error: {node_result['error']}")

    citations = node_result["narrative_output"]["citations"]
    assert isinstance(citations, list), "citations must be a list"
    assert len(citations) > 0, "citations must be non-empty"
    for cit in citations:
        assert isinstance(cit, str), f"each citation must be str, got {type(cit).__name__}"


# ── Test 5 ────────────────────────────────────────────────────────────────────

def test_critique_not_passed_returns_error_no_llm_call():
    """
    When critique_passed is False, narrative_node must return dict with 'error'
    and must NOT call the LLM. Must not raise.
    """
    bad_state = {**_BASE_STATE, "critique_passed": False}

    with patch("src.agents.narrative_agent.call") as mock_call:
        result = narrative_node(bad_state)
        mock_call.assert_not_called()

    assert isinstance(result, dict), "must return dict even on failure"
    assert "error" in result, "must set 'error' when critique_passed is False"
    assert "narrative_output" not in result, (
        "'narrative_output' must not be present when critique has not passed"
    )


def test_uses_llm_client_default_backend():
    """narrative_node must not override llm_client's default backend."""
    with patch("src.agents.narrative_agent.call", return_value=_MOCK_LLM_RESPONSE) as mock_call:
        result = narrative_node(_BASE_STATE)

    if result.get("error"):
        pytest.fail(f"narrative_node returned error: {result['error']}")

    assert "backend" not in mock_call.call_args.kwargs, "must use llm_client default backend"
