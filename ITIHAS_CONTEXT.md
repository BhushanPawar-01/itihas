# Itihas — Full Project Context
> Load this at the start of every new chat session. This is the single source of truth.

---

## What This Is

**Itihas** (Sanskrit: *thus it was*) is a multi-agent adversarial AI system for Indian military and political history 1600–1947. It ingests multilingual primary source documents, processes them through a data pipeline, and answers historical queries by having specialized agents argue with each other before synthesizing a response.

This is not a history chatbot. It is an adversarial reasoning system with a fine-tuned domain model at its core.

**Builder:** Bhushan Pawar — AI/ML engineer, ~2 years production experience (RAG systems, data pipelines, embeddings at RediMinds). Python/PyTorch, FastAPI, React, PostgreSQL, GCS. ACL + ICML workshop papers pending publication (1–2 months out).

**Timeline:** 1 month full-time. Budget: under $50 total.

---

## Current Status

**Active sprint: Week 1 — Data Pipeline (INA Trials MVP)**

MVP scope is the INA Trials 1945–46 before expanding to the full 1600–1947 timeline. Rationale: English/Urdu only, three naturally conflicting source types (British legal, INA testimony, nationalist press), best demo potential of any single event.

Planned event expansion order after INA Trials:
1. 1857 Delhi Uprising
2. Battle of Palkhed (1728)
3. Fall of Maratha Confederacy (1803–1818)
4. Siege of Srirangapatna / Tipu Sultan (1799)
5. Full 1600–1947 timeline

**Session memory:** At the end of every chat session, summarize decisions made and append to the `## Decision Log` section at the bottom of this file. Paste that updated log into the next session. This is the memory system.

---

## Repo Structure

```
itihas/
├── data/                        # gitignored
│   ├── raw/                     # original files, never modified
│   │   ├── internet_archive/ina_trials/
│   │   ├── internet_archive/press_1945_46/
│   │   ├── british_library/india_office_records/
│   │   ├── semantic_scholar/
│   │   └── manual_downloads/
│   ├── processed/
│   │   ├── text/                # cleaned UTF-8 .txt
│   │   ├── translated/          # IndicTrans2 English output
│   │   ├── chunks/              # chunked text ready for embedding
│   │   └── metadata/            # one .json per document
│   └── registry.csv
├── src/
│   ├── ingestion/               # scrapers and downloaders
│   ├── processing/              # OCR, translation, chunking, tagging
│   ├── retrieval/               # BM25 + dense retrieval, reranking
│   ├── storage/                 # PostgreSQL client, file store
│   └── utils/                   # logger, validators, llm_client
├── training/                    # QLoRA fine-tuning on Kaggle
├── agents/                      # LangGraph graph + 5 agent files
├── backend/                     # FastAPI
├── frontend/                    # React + Tailwind + D3
├── notebooks/
├── config/
│   ├── settings.py              # all config imported from here
│   └── .env                     # gitignored
├── AGENTS.md                    # coding agent instructions
├── ITIHAS_CONTEXT.md            # this file
└── requirements.txt
```

**File naming convention:**
```
{source}_{type}_{YYYYMMDD}_{seq}.pdf
ia_trial_19451107_001.pdf
bl_dispatch_19460215_003.pdf
```

---

## Architecture

### Data Pipeline (Week 1–2)

```
raw files (PDF / image / HTML)
    │
    ▼
ocr_pipeline.py          TrOCR (printed) → Tesseract fallback → IndicOCR (Devanagari)
    │
    ▼
lang_detect.py           lingua or langdetect, per paragraph not per document
    │
    ▼
translator.py            IndicTrans2 — Urdu/Marathi/Persian → English, original preserved
    │
    ▼
chunking                 512 tokens, 64-token overlap, sentence boundary aware
                         chunk gets: doc_id, chunk_index, page, language, bias_tag
    │
    ▼
metadata_tagger.py       produces .json sidecar — see schema below
    │
    ▼
storage                  PostgreSQL (chunks + embeddings via pgvector)
                         registry.csv updated on every ingest
```

**Chunking is a first-class concern.** Bad chunks = bad retrieval = bad agent answers. Rules:
- 512 tokens max, never split mid-sentence
- 64-token overlap between adjacent chunks
- Each chunk tagged with: doc_id, chunk_index, source_type, bias_tag, language, date
- Mixed-language documents: detect and tag per-chunk, not per-document

### Retrieval (Week 2–3)

Hybrid retrieval — BM25 + dense semantic search, results fused before agent use. No paid vector DB APIs.

```
query
  │
  ├──▶ BM25 (rank_bm25)           keyword match over chunk text
  │
  └──▶ Dense retrieval             sentence-transformers: paraphrase-multilingual-mpnet-base-v2
       (pgvector cosine)           embeddings stored in PostgreSQL
  │
  ▼
Reciprocal Rank Fusion (RRF)      merge BM25 + dense rankings — no ML needed, deterministic
  │
  ▼ (optional, for final polish)
HuggingFace Reranker API          cross-encoder/ms-marco-MiniLM-L-6-v2 — only if RRF result quality is low
  │
  ▼
top-k chunks with metadata → Source Agent
```

Why RRF before reranker: RRF is free, deterministic, and usually good enough. Reranker API call costs money and latency — use it as a fallback or for the upload feature demo only.

### Intelligence Layer (Week 2)

Fine-tuned Llama 3 8B via QLoRA (4-bit, LoRA adapters).
Platform: Kaggle 2x T4, 30hr/week free. Training time: 3–6 hours per run. Cost: $0.

Domain agents (Source, Political, Military) run on the fine-tuned model locally via Ollama.
Meta-reasoning agents (Orchestrator, Critique, Narrative) use HuggingFace Inference API.

**LLM API:** All calls go through `src/utils/llm_client.py`. No SDK called directly in agent files. Swap the model by changing `config/settings.py` only.

Current API target: `HuggingFace Inference API` — `mistralai/Mistral-7B-Instruct-v0.3` or `meta-llama/Meta-Llama-3-8B-Instruct` (free tier, rate-limited but sufficient for dev).

### Agent System (Week 3)

Framework: LangGraph. Five agents communicate through shared `AgentState` TypedDict. No agent calls another directly.

```
User query
    │
    ▼
ORCHESTRATOR        breaks query into sub-tasks, routes agents, assembles response
    │
    ▼
SOURCE AGENT        hybrid retrieval → tagged evidence structs (never prose)
    │
    ├──▶ POLITICAL AGENT    power dynamics, who benefits, propaganda detection
    │
    └──▶ MILITARY AGENT     terrain, logistics, physical possibility check
              │
              ▼
        CRITIQUE AGENT      reads all outputs, detects contradictions,
                            assigns confidence scores, loops back if needed (max 3x)
              │
              ▼
        NARRATIVE AGENT     final human-readable response:
                            1. political reality
                            2. military/physical reality
                            3. ground truth vs propaganda
                            + citations + confidence scores
```

Internal debate written to `debug_log` in state — not shown to user but shown in demo.

### Memory System (Current)

Simple summarization-based session memory:
- Each conversation ends with a summary of decisions, code written, and problems found
- Summary appended to `## Decision Log` at bottom of this file
- Pasted into the next session as context
- No vector DB, no external memory service — just this file

This is sufficient for a solo 1-month build. Revisit if the project grows.

### Storage

```
PostgreSQL
  └── chunks table
        doc_id, chunk_index, text, embedding (pgvector), source_type,
        bias_tag, language, date, confidence

registry.csv
  └── doc_id, source, url, language, format, bias_type,
      access_method, downloaded, ocr_applied, translated, notes
```

Neo4j (Week 3, not before): nodes for people/events/places/documents, edges for causal relationships. Not needed until agents are running. Don't build it in Week 1.

---

## Document Metadata Schema

Every processed document gets a `.json` sidecar in `data/processed/metadata/`:

```json
{
  "doc_id": "ia_trial_19451107_001",
  "source": "internet_archive",
  "url": "https://archive.org/details/...",
  "title": "INA Trial Proceedings Day 1 Red Fort",
  "date": "1945-11-07",
  "language_original": "en",
  "language_detected": "en",
  "format": "pdf",
  "pages": 34,
  "bias_type": "british_legal",
  "access_method": "pipeline",
  "ocr_applied": false,
  "translated": false,
  "chunk_count": 0,
  "pipeline_version": "0.1",
  "processed_at": "2026-05-27T10:32:00Z",
  "notes": "Day 1 Red Fort trial. Defendants: Shah Nawaz Khan, Sahgal, Dhillon."
}
```

`bias_type` vocabulary: `british_legal` | `british_military` | `ina_testimony` | `nationalist_press` | `academic` | `urdu_press` | `regional_press`

---

## Tech Stack

| Layer | Technology |
|---|---|
| OCR | TrOCR → Tesseract → IndicOCR (waterfall) |
| Translation | IndicTrans2 |
| Embeddings | sentence-transformers paraphrase-multilingual-mpnet-base-v2 |
| BM25 | rank_bm25 |
| Retrieval fusion | Reciprocal Rank Fusion (custom, no library needed) |
| Reranker (optional) | HF API: cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector storage | PostgreSQL + pgvector |
| Graph DB | Neo4j (week 3+) |
| Fine-tuning | QLoRA, Kaggle 2x T4 |
| Domain agents | Fine-tuned Llama 3 8B via Ollama |
| Meta agents | HF Inference API (Mistral-7B or Llama-3-8B-Instruct) |
| Agent framework | LangGraph |
| Backend | FastAPI |
| Frontend | React + Tailwind + D3 + vis.js |
| Deployment | HuggingFace Spaces + Railway |

---

## Data Sources (INA Trials MVP)

| Source | Method | Content |
|---|---|---|
| Internet Archive | Pipeline (IA API) | Trial transcripts, INA documents |
| Internet Archive | Pipeline | Indian press 1945–46 |
| British Library IOR | Semi-manual batch | Dispatches, internal memos |
| Semantic Scholar | Pipeline (API) | Academic papers on INA/Bose |
| NMML / Netaji Bureau | Email request | Primary Bose correspondence |
| Manual downloads | Manual | Key books (Fay, Sugata Bose — personal copies only) |

Books: do not pipeline copyrighted text. Personal reading notes and summaries only as training data.

---

## Hard Rules (Never Violate)

1. `data/raw/` is read-only. Transformations write to `data/processed/` only.
2. Every document gets a registry.csv entry before processing begins.
3. Chunking happens after translation, never before.
4. Embeddings are Step 2. Do not start embedding until 100+ validated documents exist in `processed/`.
5. LLM calls only through `src/utils/llm_client.py`. No SDK imports in agent files.
6. LangGraph only for agents. No LangChain agents, no CrewAI.
7. Neo4j is Week 3+. Do not design for it in Week 1 or 2.
8. Do not conflate Itihas with the Indus Valley decipherment project.

---

## Week 1 Goal (Current)

300–500 documents in `data/raw/`, every one validated, registry.csv complete, metadata JSONs written.

No agents, no embeddings, no fine-tuning. Just clean data.

---

## Decision Log

*Append session summaries here. Paste the updated log into every new session.*

- 2026-05-27: MVP event chosen as INA Trials 1945–46. Expansion order set. Repo structure finalized. Retrieval design: BM25 + dense + RRF, HF reranker as optional fallback. LLM API: HuggingFace Inference API (no Anthropic/OpenAI dependency). Memory: summarization-based via this file. Neo4j deferred to Week 3.
