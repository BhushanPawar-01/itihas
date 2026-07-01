"""
tests/agents/test_military_agent.py

Tests for agents/military_agent.py — military_node().

All tests are fully mocked — no real LLM or DB calls.
"""

import json
from unittest.mock import patch

import pytest

from src.agents.military_agent import military_node

_MOCK_LLM_RESPONSE = (
    "PLAUSIBLE: March distances consistent\n"
    "IMPLAUSIBLE: None\n"
    "UNCERTAIN: Supply line claims"
)

_FAKE_CHUNKS = [
    {
        "doc_id":      "ia_trial_19451107_001",
        "chunk_index": 0,
        "text":        "The INA advanced through the Imphal plain with two divisions.",
        "bias_tag":    "british_legal",
        "score":       0.91,
    },
    {
        "doc_id":      "ia_testimony_19451108_001",
        "chunk_index": 0,
        "text":        "We marched forty miles in two days through jungle terrain.",
        "bias_tag":    "ina_testimony",
        "score":       0.87,
    },
    {
        "doc_id":      "ia_press_19451110_001",
        "chunk_index": 0,
        "text":        "Supply lines stretched from Rangoon to Kohima without interruption.",
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
    "query_id":            "test-military-001",
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
    """Run military_node once with mocked call(). No real LLM or DB."""
    with patch("src.agents.military_agent.call", return_value=_MOCK_LLM_RESPONSE):
        result = military_node(_BASE_STATE)
    return result


# ── Test 1 ────────────────────────────────────────────────────────────────────

def test_return_key_present(node_result):
    """military_node must return a dict containing 'military_output'."""
    assert isinstance(node_result, dict), "must return dict"
    assert "military_output" in node_result, "missing key: military_output"


# ── Test 2 ────────────────────────────────────────────────────────────────────

def test_agent_name(node_result):
    """military_output['agent_name'] must be 'military'."""
    if node_result.get("error"):
        pytest.fail(f"military_node returned error: {node_result['error']}")

    assert node_result["military_output"]["agent_name"] == "military", (
        f"expected 'military', got {node_result['military_output']['agent_name']!r}"
    )


# ── Test 3 ────────────────────────────────────────────────────────────────────

def test_content_contains_section_labels(node_result):
    """military_output['content'] must contain at least one of the expected section labels."""
    if node_result.get("error"):
        pytest.fail(f"military_node returned error: {node_result['error']}")

    content = node_result["military_output"]["content"]
    assert isinstance(content, str) and len(content) > 0, "content must be a non-empty string"

    labels = {"PLAUSIBLE", "IMPLAUSIBLE", "UNCERTAIN"}
    found = [label for label in labels if label in content]
    assert found, (
        f"content must contain at least one of {labels}. Got:\n{content}"
    )


# ── Test 4 ────────────────────────────────────────────────────────────────────

def test_none_source_output_returns_error_no_raise():
    """
    When source_output is None, military_node must return a dict with 'error'
    and must NOT include 'military_output'. Must not raise.
    """
    bad_state = {**_BASE_STATE, "source_output": None}

    with patch("src.agents.military_agent.call", return_value=_MOCK_LLM_RESPONSE):
        result = military_node(bad_state)

    assert isinstance(result, dict), "must return dict even on failure"
    assert "error" in result, "must set 'error' key when source_output is None"
    assert "military_output" not in result, (
        "'military_output' must not be present when source_output is None"
    )


# ── Test 5 ────────────────────────────────────────────────────────────────────

def test_citations_list_of_strings(node_result):
    """military_output['citations'] must be a list of strings, one per chunk."""
    if node_result.get("error"):
        pytest.fail(f"military_node returned error: {node_result['error']}")

    citations = node_result["military_output"]["citations"]
    assert isinstance(citations, list), "citations must be a list"
    assert len(citations) == len(_FAKE_CHUNKS), (
        f"expected {len(_FAKE_CHUNKS)} citations, got {len(citations)}"
    )
    for cit in citations:
        assert isinstance(cit, str), f"each citation must be str, got {type(cit).__name__}"