"""
tests/agents/test_source_agent.py

Tests for agents/source_agent.py — source_node().

Rules (from spec):
  - Real query: "Who commanded the INA at the Battle of Imphal"
  - retrieve() runs against real Neon Postgres — do NOT mock it.
  - call is mocked to return "KEEP" for all chunks.
"""

import json
from unittest.mock import patch

import pytest

from src.agents.source_agent import source_node

REAL_QUERY = "Who commanded the INA at the Battle of Imphal"

_BASE_STATE = {
    "query":              REAL_QUERY,
    "query_id":           "test-source-001",
    "retrieved_chunks":   [],
    "source_output":      None,
    "political_output":   None,
    "military_output":    None,
    "critique_output":    None,
    "narrative_output":   None,
    "critique_loop_count": 0,
    "route_to":           None,
    "critique_passed":    False,
    "debug_log":          [],
    "error":              None,
}


@pytest.fixture(scope="module")
def node_result():
    """
    Run source_node once against real Neon Postgres.
    call is patched to return "KEEP" for all chunks so triage
    never discards anything — isolates retrieval from LLM behaviour.
    """
    with patch("src.agents.source_agent.call", return_value="KEEP"):
        result = source_node(_BASE_STATE)
    return result


# ── Test 1 ────────────────────────────────────────────────────────────────────

def test_return_keys_present(node_result):
    """source_node must return a dict with source_output, retrieved_chunks, debug_log."""
    assert isinstance(node_result, dict), "source_node must return a dict"
    assert "source_output"    in node_result, "missing key: source_output"
    assert "retrieved_chunks" in node_result, "missing key: retrieved_chunks"
    assert "debug_log"        in node_result, "missing key: debug_log"


# ── Test 2 ────────────────────────────────────────────────────────────────────

def test_agent_name(node_result):
    """source_output['agent_name'] must be 'source'."""
    # Guard: if error key is set, surface it clearly
    if "error" in node_result and node_result["error"]:
        pytest.fail(f"source_node returned error: {node_result['error']}")

    output = node_result["source_output"]
    assert output is not None, "source_output is None"
    assert output["agent_name"] == "source", (
        f"expected agent_name='source', got {output['agent_name']!r}"
    )


# ── Test 3 ────────────────────────────────────────────────────────────────────

def test_retrieved_chunks_nonempty(node_result):
    """
    retrieved_chunks must be a non-empty list.
    Failure here means Neon Postgres has no indexed chunks — fix ingestion.
    """
    chunks = node_result.get("retrieved_chunks", [])
    assert isinstance(chunks, list), "retrieved_chunks must be a list"
    assert len(chunks) > 0, (
        "retrieved_chunks is empty — Neon Postgres appears to have no indexed chunks. "
        "Run the ingestion pipeline first."
    )


# ── Test 4 ────────────────────────────────────────────────────────────────────

def test_content_is_parseable_json_list(node_result):
    """source_output['content'] must be a JSON string that parses to a non-empty list."""
    if "error" in node_result and node_result["error"]:
        pytest.fail(f"source_node returned error: {node_result['error']}")

    content = node_result["source_output"]["content"]
    assert isinstance(content, str), "content must be a string"

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        pytest.fail(f"source_output['content'] is not valid JSON: {exc}")

    assert isinstance(parsed, list), "content must deserialise to a list"
    assert len(parsed) > 0, "content list must be non-empty"


# ── Test 5 ────────────────────────────────────────────────────────────────────

def test_confidence_is_valid_float(node_result):
    """source_output['confidence'] must be a float in [0.0, 1.0]."""
    if "error" in node_result and node_result["error"]:
        pytest.fail(f"source_node returned error: {node_result['error']}")

    confidence = node_result["source_output"]["confidence"]
    assert isinstance(confidence, float), (
        f"confidence must be float, got {type(confidence).__name__}"
    )
    assert 0.0 <= confidence <= 1.0, (
        f"confidence {confidence} is outside [0.0, 1.0]"
    )