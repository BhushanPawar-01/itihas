# Itihas — Full Project Context
> Load this at the start of every new chat session. This is the single source of truth.

---

## What This Is

**Itihas** (Sanskrit: *thus it was*) is a multi-agent adversarial AI system for Indian
military and political history, 1600–1947. It ingests multilingual primary source
documents, processes them through a data pipeline, and answers historical queries
by having specialized agents argue with each other before synthesizing a response.

This is not a history chatbot. It is an adversarial reasoning system with hybrid
retrieval at its core.

**Builder:** Bhushan Pawar.

---

## Current Status

**Weeks 1–4 are complete and deployed.** Data pipeline, retrieval stack, agent
graph, FastAPI backend, React frontend, and HuggingFace Spaces deployment (Docker,
CPU free tier) are all live. GitHub Actions auto-syncs `main` to the HF Space on
every push.

**This document defines Phase 5–7** — a quality and depth pass on the working
MVP. Nothing here is a rewrite. The retrieval stack, agent graph topology, and
data pipeline are solid and unchanged. What's being upgraded: frontend polish,
a real debate mechanism (the current critique "loop" reruns agents blind — it
doesn't actually feed the contradiction back to them), and follow-up question
support with tiered conversational memory.

**Scope: 4–5 focused days.** This is not a new build. Existing Week 1–4 systems
are the foundation; everything below extends them.

---

## Repo Layout (as built)

```
itihas/
├── backend/
│   ├── main.py                  # FastAPI app, CORS, static frontend serving
│   └── routes/query.py          # /api/v1/query endpoint, citation resolution
├── frontend/
│   ├── src/
│   │   ├── api/client.js        # single fetch wrapper — no other file calls fetch()
│   │   ├── components/          # UI panels
│   │   └── constants/biasTags.js
│   └── dist/                    # production build, served by FastAPI
├── src/
│   ├── agents/
│   │   ├── graph.py              # LangGraph orchestrator, run_query() entrypoint
│   │   ├── state.py              # AgentState TypedDict, shared by all nodes
│   │   ├── source_agent.py
│   │   ├── political_agent.py
│   │   ├── military_agent.py
│   │   ├── critique_agent.py
│   │   └── narrative_agent.py
│   ├── retrieval/
│   │   ├── bm25_retriever.py
│   │   ├── dense_retriever.py
│   │   └── fusion.py              # RRF fusion, optional reranker
│   ├── storage/
│   │   ├── db_client.py           # psycopg2 pool, pgvector search
│   │   ├── file_store.py
│   │   ├── ingest_chunks.py
│   │   └── ingest_registry.py     # populates documents table from registry.csv
│   ├── processing/                # OCR, lang detect, bias tagging, chunking pipeline
│   └── utils/
│       ├── llm_client.py          # single LLM gateway — call() defaults to backend="openai";
│       │                          # call_hf() / call_openai() / call_ollama() convenience wrappers
│       └── logger.py
├── data/
│   ├── raw/                       # read-only, never modified
│   ├── processed/
│   └── registry.csv
├── config/settings.py             # all env-driven config, single source
├── AGENTS.md
└── ITIHAS_CONTEXT.md              # this file
```

---

## Storage — Postgres (Neon)

**`chunks` table** — unchanged since Week 2. doc_id, chunk_index, text, embedding
(vector 768), source_type, bias_tag, language, date, confidence.

**`documents` table** — added in Week 4. Populated from `registry.csv` via
`src/storage/ingest_registry.py`. Columns: doc_id (PK), title, url, source,
bias_type, language, date, creator. Used by `backend/routes/query.py` to resolve
citations to human-readable titles and source URLs instead of raw doc_ids.

No new tables are needed for Phase 5–7. Conversation memory is ephemeral
(browser-session only, not persisted) — see Phase 7 below.

---

## LLM Configuration — Current State

`src/utils/llm_client.py` is the single LLM gateway. No other file imports an
LLM SDK directly. Three backends, selected via `call(..., backend=...)`:

- **`hf`** — HuggingFace Inference Providers router, via the OpenAI-compatible
  chat completions API. Rejects `temperature=0.0` exactly — floors to `0.01`.
- **`openai`** — OpenAI's own API, same SDK, different `base_url`/credentials.
  Accepts `temperature=0.0` exactly. **This is the default backend** —
  `call()`'s `backend` parameter defaults to `"openai"`, and every existing
  agent file that calls `call()` without an explicit `backend=` argument is
  now hitting OpenAI, not HuggingFace or Ollama.
- **`ollama`** — local Ollama server, different protocol (`/api/generate`,
  not the OpenAI SDK), used for the fine-tuned local model when it's running.

Convenience wrappers: `call_hf()`, `call_openai()`, `call_ollama()` — each
just calls `call()` with the backend pre-set. `hf` and `openai` share one
internal implementation (`_openai_sdk_call`) since both speak the OpenAI chat
completions API; only credentials, base URL, retryable status codes, and the
minimum temperature floor differ between them.

**Practical consequence for Phase 6 and 7 work:** any new prompt added to
`political_agent.py`, `military_agent.py` (Phase 6 rebuttal prompts), or the
new `src/agents/memory.py` summarizer (Phase 7) should call `call()` with no
`backend` argument unless there's a specific reason to pin it to `hf` or
`ollama` — the default is `openai`, and that default should be treated as
intentional going forward rather than re-litigated per call site.

**Still open:** the strategic reason for choosing OpenAI as the default
(cost, latency, output quality, or something else) was not recorded during
the planning session that produced this document. Doesn't block Phase 5–7
work — the technical facts above are all that's needed — but worth writing
down here once decided, since it'll matter again if API costs become a
concern at higher query volume.

---

## Phase 5 — Frontend Overhaul

**Problem being fixed:** the current UI is functionally complete but reads as an
unstyled MVP — raw markdown text dumps, doc_ids shown instead of titles, a
knowledge graph panel that adds demo flourish without analytical value, and no
visual hierarchy between the main answer and supporting evidence.

### 5.1 — Remove the knowledge graph entirely
Delete `frontend/src/components/GraphPanel.jsx`, `frontend/src/utils/buildGraphData.js`,
and the `d3` npm dependency. It added visual interest but no analytical value —
the heuristic entity extraction (regex-based person/event detection) was never
reliable enough to trust, and it wasn't answering a real question the user had.

### 5.2 — Layout inversion
**Left sidebar** (fixed width, each section independently collapsible, default
**closed**):
- **References** — replaces the old "Retrieved Evidence (N chunks)" panel entirely.
  Deduplicated by title (a doc_id may back multiple chunks, but each title appears
  once). Fixed-height scrollable window. Each entry is the title, hyperlinked to
  its source URL. No chunk text, no bias tag cards, no raw doc_id shown.
- **Military Analysis** — the raw agent output, now rendered through `ReactMarkdown`
  (not a `<pre>` dump of literal `###` text) so `PLAUSIBLE` / `IMPLAUSIBLE` /
  `UNCERTAIN` render as actual structured headings.
- **Political Analysis** — same treatment as Military.

**Right main column** (full width, always visible, not collapsible):
- **Narrative** — the four-section synthesis (`Political Reality`, `Military
  Reality`, `Ground Truth vs Propaganda`, `Confidence Assessment`). Each section
  parsed out of the `##`-delimited markdown string and rendered as its own
  titled card, not one continuous markdown blob.
- Inline citations within the narrative text (currently literal `[doc_id]`
  markers) are rewritten before rendering: each `[doc_id]` is replaced with a
  short hyperlinked title fragment pointing at the source URL, using the
  `citations` array already returned by the API (`{doc_id, title, url}`). This
  is a pure frontend text transform — no backend change required, since citation
  resolution already happens server-side in `backend/routes/query.py`.

### 5.3 — Branding

**Wordmark font: Samarkan, self-hosted.** Confirmed — the font file will be
supplied. Exact wiring, so there's no ambiguity when this gets built:

1. Place the font file at exactly this path:
   `frontend/public/fonts/samarkan.woff2`
   (if the sourced file is `.ttf` only, convert to `.woff2` first — smaller,
   and every modern browser Itihas needs to support takes it directly)

2. Add this `@font-face` block to `frontend/src/index.css`, **above** the
   existing `@tailwind` directives (an `@import`/`@font-face` block placed
   after `@tailwind` lines has already caused a build error once in this
   project — keep it first):

   ```css
   @font-face {
     font-family: 'Samarkan';
     src: url('/fonts/samarkan.woff2') format('woff2');
     font-weight: normal;
     font-style: normal;
     font-display: swap;
   }
   ```

3. Register it in `frontend/tailwind.config.js`, inside `theme.extend.fontFamily`,
   alongside the existing `sans`/`serif` entries:

   ```javascript
   fontFamily: {
     sans:     ['Inter', 'system-ui', 'sans-serif'],
     serif:    ['Merriweather', 'Georgia', 'serif'],
     samarkan: ['Samarkan', 'serif'],   // serif fallback if the font file fails to load
   },
   ```

4. Apply it to the wordmark only (`<h1>Itihas</h1>` in the header) via
   `className="font-samarkan"` — not globally. Samarkan is a display/stylized
   font; using it for body text or UI labels would hurt readability. It's a
   wordmark treatment, not a typeface swap for the whole app.

5. Verify: after wiring, open DevTools → Network tab, confirm `samarkan.woff2`
   loads with a 200 status, not a 404. A silent fallback to the `serif` stack
   with no console error is the most likely failure mode if the path or
   filename doesn't match exactly what's in step 1 — check Network, not just
   visual appearance, since a fallback serif font can look "close enough" at
   a glance and mask the bug.

**Top bar color:** current navy-on-parchment pairing feels disconnected from
the rest of the page — rework so the header shares the page's palette rather
than reading as a separate block bolted on top.

### 5.4 — What does NOT change
- `QueryPanel.jsx` (input box) — unchanged, still the entry point.
- `backend/routes/query.py`'s citation resolution logic — unchanged, frontend
  just consumes it differently.
- `DebugPanel.jsx` — unchanged, still gated behind the debug checkbox.

---

## Phase 6 — Agent Debate Mechanism

**Problem being fixed:** the critique loop exists structurally (`critique_loop_count`,
max 3, `route_after_critique` sends control back to `political_node` on LOOP) but
it currently does nothing useful. On a loop-back, `political_node` and
`military_node` rerun their original prompts against the same evidence with no
awareness that critique found a contradiction — they're not reconsidering
anything, just rerolling the dice on the same question and hoping for a
different, converging answer by round 3.

**The fix is prompt-level, not topology-level.** No new graph nodes, no new
state fields beyond what's already there, no change to the loop cap (confirmed:
reuse `critique_loop_count`, cap stays at 3).

### What changes
`political_node` and `military_node` both gain a check at the top of their
prompt-building step:

```
if state.get("critique_loop_count", 0) > 0 and state.get("critique_output") is not None:
    # This is a rebuttal round, not a first pass.
    # Build an additional prompt section containing:
    #   - the specific contradictions from critique_output["content"] (parsed JSON)
    #   - the OTHER agent's current output — already sitting in state, e.g.
    #     political_node reads state["military_output"], military_node reads
    #     state["political_output"] — no graph change needed, this data is
    #     already present after the first round completes
    # Instruct the model explicitly: address the named contradiction — either
    # revise the prior position with stated justification, or explain why the
    # apparent conflict is not actually a conflict. Do not just restate the
    # original analysis unchanged.
```

This is the actual debate: round 2 and 3 are not blind reruns, they are direct
responses to a named disagreement, with visibility into what the other side
argued. `critique_agent.py` itself does not need to change — it already produces
a `contradictions` list and a `PASS`/`LOOP` decision; it just wasn't being used
downstream until now.

---

## Phase 7 — Follow-Up Questions & Conversational Memory

**Confirmed scope: ephemeral only.** No `conversation_id`, no new Postgres table,
no persistence across a page refresh. All conversation state lives in React
state in the browser tab. This significantly simplifies the implementation —
the backend stays stateless per-request, exactly as it is today.

### Memory tiering rule (as specified)
For a conversation at turn N:
- **Turn 1** — no prior context. Behaves exactly as today.
- **Turn 2** — context passed alongside the new query is the **full detail** of
  turn 1: its complete narrative answer plus its retrieval evidence (the chunks
  BM25/dense retrieval surfaced for that turn).
- **Turn N ≥ 3** — context is: **a rolling summary** covering everything before
  turn N−1 (i.e. a summary of the summary-so-far, folding in each older turn as
  it ages out of "full detail" status) **plus the full detail of turn N−1**
  (the immediately preceding turn, not compressed) **plus fresh retrieval** for
  the current query N.

In effect: the most recent prior turn always gets full fidelity; everything
older than that gets progressively compressed into one rolling summary that's
re-summarized each time a turn ages out of the "immediately preceding" slot.

### Implementation shape
- New module `src/agents/memory.py`:
  ```python
  def build_conversation_context(turns: list[dict]) -> str:
      """
      turns: ordered list of prior turns, each shaped like
             {"query": str, "narrative": str, "source_chunks": list[dict]}
             (this is exactly the shape the frontend already holds, since
             QueryResponse already returns narrative + source_chunks per turn —
             no new data needs to be captured anywhere)

      Returns a single context string to inject into the current request's
      agent prompts and retrieval query. Empty string if turns is empty.

      Behaviour:
        len(turns) == 0  -> ""
        len(turns) == 1  -> full detail of turns[0]
        len(turns) >= 2  -> LLM-generated rolling summary of turns[0:-1]
                            + full detail of turns[-1]
      """
  ```
  The rolling summary is generated by a single LLM call each time a follow-up
  request comes in — it is not cached or persisted anywhere, consistent with
  the ephemeral-only decision. This costs one extra LLM call per follow-up turn
  once the conversation is 3+ turns deep, which is an acceptable and bounded
  cost for a demo-scale system.

- `run_query()` in `src/agents/graph.py` gains an optional `history` parameter.
  When present, it calls `build_conversation_context(history)` and seeds
  `initial_state["conversation_context"]` with the result.

- `AgentState` (in `src/agents/state.py`) gains one new field:
  `conversation_context: str` (default `""`).

- `source_agent.py` folds `conversation_context` into the query string passed to
  `retrieve()`, so follow-up retrieval is biased by prior conversation, not just
  the bare new question.

- `political_agent.py`, `military_agent.py`, `narrative_agent.py` prepend
  `conversation_context` (if non-empty) to their existing prompts as a
  "Conversation so far:" section.

- `backend/routes/query.py`'s `QueryRequest` model gains an optional field:
  `conversation_history: list[dict] | None = None`, forwarded straight into
  `run_query(query, history=conversation_history)`.

- **Frontend responsibility:** the growing turn array already exists implicitly
  since every API response already contains `narrative` and `source_chunks` —
  the frontend just needs to accumulate these into a list as the conversation
  progresses and resend the whole list as `conversation_history` on each new
  submission. No new frontend data-fetching, just state accumulation and
  resending what's already been received.

### What does NOT change
- No database schema change.
- No new backend persistence layer.
- `critique_agent.py`, `graph.py` topology — unchanged. Conversation context is
  purely an input to the existing graph, not a new node or edge.

---

## Hard Rules (Never Violate) — unchanged from Week 1–4

1. `data/raw/` is read-only.
2. Every document gets a `registry.csv` entry before processing.
3. Chunking happens after translation, never before.
4. LLM calls only through `src/utils/llm_client.py` — no SDK imports elsewhere.
5. LangGraph only for agents. No LangChain agents, no CrewAI.
6. No raw `fetch()` calls in frontend components — everything through
   `frontend/src/api/client.js`.
7. Conversation memory is ephemeral only (Phase 7 decision, confirmed) — do not
   introduce a persistence layer for it without a fresh explicit decision.

---

## Decision Log

- **2026-05-27:** MVP event chosen as INA Trials 1945–46. Repo structure
  finalized. Retrieval design: BM25 + dense + RRF. Neo4j deferred (never
  ended up needed).
- **2026-06-09:** Weeks 1–2 complete. Retrieval stack verified via smoke test.
- **Week 3:** Agent graph (source → political/military → critique → narrative)
  built and verified end-to-end via FastAPI.
- **Week 4:** Frontend (React + Tailwind), Docker deployment to HuggingFace
  Spaces (CPU free tier, Python 3.11 — deliberately not matching the
  developer's local Python 3.14, since 3.14 has a known Pydantic V1
  compatibility issue with LangChain/LangGraph internals that was the suspected
  cause of a silent request-crash bug during local dev). `documents` table
  added for citation title/URL resolution. GitHub Action added to auto-sync
  `main` to the HF Space on push.
- **Phase 5–7 planning session:** Frontend overhaul scoped (remove knowledge
  graph, layout inversion — evidence panels left/collapsed, narrative right/
  prominent, Samarkan wordmark with exact `@font-face`/Tailwind wiring, structured
  section rendering). Debate mechanism scoped as a prompt-level fix to the
  existing critique loop, reusing the existing loop cap of 3 — no graph
  topology change. Follow-up memory scoped as ephemeral-only (no DB
  persistence), tiered rolling-summary approach, no new storage layer.
  `llm_client.py` confirmed to have three backends (`hf`, `openai`, `ollama`);
  `call()` defaults to `backend="openai"` — new Phase 6/7 prompts should call
  `call()` with no explicit `backend=` unless there's a specific reason to pin
  to `hf` or `ollama`. **Still open: the strategic reason for choosing OpenAI
  as default (cost/latency/quality) was never recorded — not a blocker, just
  worth writing down when decided.**