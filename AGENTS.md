# AGENTS.md
> Read before every task. Working code only. Plausibility is not correctness.

Symlink for tools that look elsewhere:
```bash
ln -s AGENTS.md CLAUDE.md && ln -s AGENTS.md GEMINI.md
```

---

## 0. Non-negotiables

1. No flattery, no filler. Start with the answer or the action.
2. Disagree when you disagree. Say so before doing the work.
3. Never fabricate — paths, function names, test results, API shapes. Read the file or run the command.
4. Stop when confused. Ask; do not pick silently and proceed.
5. Touch only what the task requires. No drive-by refactors.

---

## 1. Before writing code

- State your plan in one or two sentences. For non-trivial tasks, list steps with a verification check for each.
- Read the files you will touch and the files that call them.
- Match existing patterns. If the project uses pattern X, use X.
- Surface assumptions out loud before burying them in code.

---

## 2. Code quality

- Minimum code that solves the stated problem. Nothing speculative.
- No abstractions for single-use code. No hooks or extension points not requested.
- Handle failures that can actually happen. Ignore impossible scenarios.
- If a solution runs 200 lines and could be 50, rewrite it first.

---

## 3. Surgical changes

- Do not improve adjacent code not part of the task.
- Do not delete pre-existing dead code unless asked. Mention it in the summary instead.
- Clean up orphans your own edit creates (unused imports, variables, functions).
- Match project style exactly: indentation, quotes, naming.

---

## 4. Verification

- Run the code. Do not report done based on a plausible diff.
- If a test suite exists, run it. If a linter exists, run it.
- Fix root causes, not symptoms. Suppressing an error is not fixing it.
- After two failed corrections on the same issue, stop and ask for a session reset.

---

## 5. Stack

- **Language:** Python 3.11+
- **Agent framework:** LangGraph only — not LangChain agents, not CrewAI
- **LLM calls:** only through `src/utils/llm_client.py` — no SDK imports in agent files
- **LLM API:** HuggingFace Inference API (`mistralai/Mistral-7B-Instruct-v0.3` or `meta-llama/Meta-Llama-3-8B-Instruct`)
- **Local inference:** Ollama (fine-tuned Llama 3 8B for domain agents)
- **Retrieval:** BM25 (`rank_bm25`) + pgvector dense search, fused with RRF. HF reranker API optional fallback only.
- **Vector storage:** PostgreSQL + pgvector — no external vector DB services
- **Graph DB:** Neo4j — Week 3 only, do not build earlier
- **Backend:** FastAPI + Pydantic — no raw dicts in route handlers
- **Frontend:** React + Tailwind + D3 + vis.js

---

## 6. Commands

```bash
# Install
pip install -r requirements.txt --break-system-packages

# Run any pipeline script independently
python src/ingestion/archive_scraper.py
python src/processing/ocr_pipeline.py

# Backend
uvicorn backend.main:app --reload

# Lint
black . && isort .

# Test
pytest tests/
```

---

## 7. Layout

- Source: `src/` (pipeline), `agents/` (LangGraph), `backend/` (FastAPI), `training/` (QLoRA)
- Tests: `tests/` — mirror `src/` structure
- Config: `config/settings.py` — all paths, URLs, model names imported from here
- Secrets: `config/.env` — gitignored, loaded once in settings.py
- **Do not modify:** `data/raw/` — read-only, original files live here forever
- **Do not touch yet:** `agents/`, `training/`, `backend/`, `frontend/` — Week 1 is data pipeline only

---

## 8. Conventions

**Naming:**
```
Files:    snake_case.py
Classes:  PascalCase
Funcs:    snake_case
Doc IDs:  {source}_{type}_{YYYYMMDD}_{seq}  e.g. ia_trial_19451107_001
```

**Imports:** Absolute only — `from src.utils.logger import get_logger`

**Error handling:** Wrap every external call (API, DB, file read) in try/except. Log with context, then raise or return safe fallback. Never swallow silently.

**Logging:** Use `src/utils/logger.py` (structured JSON). Every pipeline step logs: input, output, duration, errors. No `print()` in pipeline code.

**Registry:** Update `data/registry.csv` on every document ingest. No exceptions.

**Chunking:** 512 tokens max, 64-token overlap, sentence-boundary aware, after translation, never before.

**Metadata:** Every processed document gets a `.json` sidecar in `data/processed/metadata/` before any downstream step runs on it.

---

## 9. Forbidden

- Calling any LLM SDK directly in `agents/` files
- Modifying anything in `data/raw/`
- Building Neo4j, embeddings, or agent code before 100+ validated documents exist in `data/processed/`
- LangChain agents, CrewAI, AutoGen
- Hardcoding secrets, API keys, or file paths outside `config/`
- Skipping registry.csv updates

---

## 10. Project Learnings

*Agent maintains this section. Append one concrete line per session correction.*

- (empty — fill as mistakes are caught)
