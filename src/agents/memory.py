"""
Conversational memory for Itihas — ephemeral, per-request only.

Public API:
    build_conversation_context(turns) -> str

Tiering rule:
    len(turns) == 0  ->  ""                          (first query, no history)
    len(turns) == 1  ->  full detail of turns[0]     (turn 2: previous turn verbatim)
    len(turns) >= 2  ->  LLM summary of turns[0:-1]  (everything before the last)
                         + full detail of turns[-1]  (previous turn verbatim)

"Full detail" = the query, narrative answer, and a compact evidence listing.
"Summary" = a single LLM call that compresses older turns into a paragraph,
re-generated fresh on every request (never cached, never persisted).

This costs one extra LLM call per follow-up once the conversation is 3+ turns
deep. That is intentional and bounded — acceptable for demo scale.

Each turn dict shape (mirrors what the frontend already holds):
    {
        "query":         str,
        "narrative":     str,          # QueryResponse.narrative
        "source_chunks": list[dict],   # QueryResponse.source_chunks (may be None)
    }
"""

from __future__ import annotations

from src.utils.llm_client import call
from src.utils.logger import get_logger

log = get_logger(__name__)

# How many source chunk titles to include in full-detail serialisation.
# Enough to anchor context without ballooning the prompt.
_MAX_CHUNKS_IN_DETAIL = 5

_SUMMARY_SYSTEM = (
    "You are a concise summariser. The user is having a multi-turn conversation "
    "about Indian military and political history with an AI research system. "
    "Summarise the conversation turns below into a single dense paragraph (≤120 words). "
    "Preserve: the main questions asked, the key historical conclusions reached, "
    "and any named people, events, or dates that were established as facts. "
    "Do not editorialize. Do not add information not present in the turns. "
    "Output only the summary paragraph — no preamble, no headings."
)


def _serialise_turn_full(turn: dict) -> str:
    """
    Render a single turn as a compact text block for full-detail inclusion.
    Never raises — missing fields are handled gracefully.
    """
    query     = turn.get("query", "").strip()
    narrative = turn.get("narrative", "").strip()

    # Condense the narrative: first 600 chars is enough to carry the substance
    # without duplicating the entire synthesis in the prompt.
    narrative_excerpt = narrative[:600] + ("…" if len(narrative) > 600 else "")

    # Optionally list a few source doc_ids so agents know what evidence was used
    chunks      = turn.get("source_chunks") or []
    chunk_ids   = [c.get("doc_id", "") for c in chunks[:_MAX_CHUNKS_IN_DETAIL] if c.get("doc_id")]
    source_line = f"Sources used: {', '.join(chunk_ids)}" if chunk_ids else ""

    parts = [f"Q: {query}", f"A: {narrative_excerpt}"]
    if source_line:
        parts.append(source_line)

    return "\n".join(parts)


def _summarise_turns(turns: list[dict]) -> str:
    """
    Call the LLM to compress a list of turns into a single paragraph.
    Falls back to a plain concatenation if the LLM call fails — so a
    network error here never aborts the main query.
    """
    if not turns:
        return ""

    turn_text = "\n\n---\n\n".join(
        f"Turn {i + 1}:\n{_serialise_turn_full(t)}"
        for i, t in enumerate(turns)
    )
    prompt = f"{_SUMMARY_SYSTEM}\n\nTurns to summarise:\n{turn_text}"

    try:
        summary = call(prompt, max_tokens=200, temperature=0.1)
        log.info(
            "memory: summarised %d turn(s) → %d chars", len(turns), len(summary)
        )
        return summary.strip()
    except Exception as exc:
        log.warning(
            "memory: LLM summarisation failed (%s) — falling back to plain concat", exc
        )
        # Plain fallback: concatenate serialised turns, truncated to 800 chars
        fallback = "\n\n".join(_serialise_turn_full(t) for t in turns)
        return fallback[:800] + ("…" if len(fallback) > 800 else "")


def build_conversation_context(turns: list[dict]) -> str:
    """
    Build a context string to inject into the current request's agent prompts.

    Args:
        turns: ordered list of prior completed turns, each shaped as:
               {"query": str, "narrative": str, "source_chunks": list[dict] | None}
               The list must NOT include the current in-flight query — only
               turns that have already received a full response.

    Returns:
        A formatted string ready to prepend to agent prompts, or "" if no history.

    Tiering:
        0 turns  ->  ""
        1 turn   ->  full detail of turns[0]
        2+ turns ->  LLM summary of turns[0:-1] + full detail of turns[-1]
    """
    if not turns:
        return ""

    if len(turns) == 1:
        detail = _serialise_turn_full(turns[0])
        return _wrap(detail)

    # 2+ turns: summarise everything except the last, keep last in full
    older_summary = _summarise_turns(turns[:-1])
    last_detail   = _serialise_turn_full(turns[-1])

    parts: list[str] = []
    if older_summary:
        parts.append(f"Earlier conversation summary:\n{older_summary}")
    parts.append(f"Most recent exchange:\n{last_detail}")

    return _wrap("\n\n".join(parts))


def _wrap(context_body: str) -> str:
    """
    Wrap the context body in a clearly delimited block so agent prompt
    templates can treat it as an opaque prefix without reformatting it.
    """
    return (
        "=== CONVERSATION SO FAR ===\n"
        f"{context_body}\n"
        "=== END CONVERSATION CONTEXT ===\n"
    )