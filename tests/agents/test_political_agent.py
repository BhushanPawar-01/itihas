"""
Tests for agents/political_agent.py — political_node().

All tests are fully mocked — no real Ollama or Postgres calls.
"""

import json
from unittest.mock import patch

import pytest

from src.agents.political_agent import political_node

_MOCK_LLM_RESPONSE = (
    "BENEFICIARY: British\nOMISSIONS: INA voices\n"
    "INTERPRETATION: Colonial framing dominates"
)

_FAKE_CHUNKS = [
    {
        "doc_id":      "ia_trial_19451107_001",
        "chunk_index": 0,
        "text":        "The accused Shah Nawaz Khan was charged with waging war against the King.",
        "bias_tag":    "british_legal",
        "score":       0.91,
    },
    {
        "doc_id":      "ia_testimony_19451108_001",
        "chunk_index": 0,
        "text":        "We fought for Azad Hind. Our cause was just and our commander inspired us.",
        "bias_tag":    "ina_testimony",
        "score":       0.87,
    },
    {
        "doc_id":      "ia_press_19451110_001",
        "chunk_index": 0,
        "text":        "The INA heroes are being persecuted by a dying empire.",
        "bias_tag":    "nationalist_press",
        "score":       0.82,
    },
]

_MOCK_SOURCE_OUTPUT = {
    "agent_name": "source",
    "content":    json.dumps(_FAKE_CHUNKS),
    "confidence": 1.0,
    "citations":  [c["doc_id"] for c in _FAKE_CHUNKS],
}

_BASE_STATE = {
    "query":               "Who commanded the INA at the Battle of Imphal",
    "query_id":            "test-political-001",
    "retrieved_chunks":    [],
    "source_output":       _MOCK_SOURCE_OUTPUT,
    "political_output":    None,
    "military_output":     None,
    "critique_output":     None,
    "narrative_output":    None,
    "critique_loop_count": 0,
    "route_to":            None,
    "critique_passed":     False,
    "debug_log":           [],
    "error":               None,
}


@pytest.fixture(scope="module")
def node_result():
    """Run political_node once with mocked. No real LLM or DB."""
    with patch("src.agents.political_agent.call", return_value=_MOCK_LLM_RESPONSE):
        result = political_node(_BASE_STATE)
    return result


# ── Test 1 ────────────────────────────────────────────────────────────────────

def test_return_key_present(node_result):
    """political_node must return a dict containing 'political_output'."""
    assert isinstance(node_result, dict), "must return dict"
    assert "political_output" in node_result, "missing key: political_output"


# ── Test 2 ────────────────────────────────────────────────────────────────────

def test_agent_name(node_result):
    """political_output['agent_name'] must be 'political'."""
    if node_result.get("error"):
        pytest.fail(f"political_node returned error: {node_result['error']}")

    assert node_result["political_output"]["agent_name"] == "political", (
        f"expected 'political', got {node_result['political_output']['agent_name']!r}"
    )


# ── Test 3 ────────────────────────────────────────────────────────────────────

def test_content_nonempty_string(node_result):
    """political_output['content'] must be a non-empty string."""
    if node_result.get("error"):
        pytest.fail(f"political_node returned error: {node_result['error']}")

    content = node_result["political_output"]["content"]
    assert isinstance(content, str), f"content must be str, got {type(content).__name__}"
    assert len(content) > 0, "content must not be empty"


# ── Test 4 ────────────────────────────────────────────────────────────────────

def test_citations_list_of_strings(node_result):
    """political_output['citations'] must be a list of strings, one per chunk."""
    if node_result.get("error"):
        pytest.fail(f"political_node returned error: {node_result['error']}")

    citations = node_result["political_output"]["citations"]
    assert isinstance(citations, list), "citations must be a list"
    assert len(citations) == len(_FAKE_CHUNKS), (
        f"expected {len(_FAKE_CHUNKS)} citations, got {len(citations)}"
    )
    for cit in citations:
        assert isinstance(cit, str), f"each citation must be str, got {type(cit).__name__}"


# ── Test 5 ────────────────────────────────────────────────────────────────────

def test_none_source_output_returns_error_no_raise():
    """
    When source_output is None, political_node must return a dict with 'error'
    and must NOT include 'political_output'. Must not raise.
    """
    bad_state = {**_BASE_STATE, "source_output": None}

    with patch("src.agents.political_agent.call", return_value=_MOCK_LLM_RESPONSE):
        result = political_node(bad_state)

    assert isinstance(result, dict), "must return dict even on failure"
    assert "error" in result, "must set 'error' key when source_output is None"
    assert "political_output" not in result, (
        "'political_output' must not be present when source_output is None"
    )