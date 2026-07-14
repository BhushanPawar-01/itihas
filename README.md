---
title: Itihas
emoji: ⚔️
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Itihas

Itihas is a multi-agent reasoning platform for analyzing Indian military and political history from 1600–1947. It combines retrieval, adversarial critique, and narrative synthesis to generate evidence-backed historical reports from source documents.

## Overview

Itihas is designed for research-grade historical analysis with a modular architecture that separates:

- retrieval and evidence gathering
- political and military critique
- narrative synthesis
- storage, processing, and API serving

The system is built to support both local development and cloud deployment, including Hugging Face Spaces.

## Key capabilities

- Multi-agent analysis workflow for source evaluation and critique
- Hybrid retrieval using BM25, dense embeddings, and fusion
- FastAPI-based backend for query execution and API access
- React + Vite frontend for interactive exploration
- PostgreSQL-backed storage for documents, chunks, and metadata
- Docker-ready deployment configuration

## Architecture at a glance

- Backend: FastAPI application in [backend](backend)
- Frontend: React/Vite application in [frontend](frontend)
- Agent orchestration: LangGraph-based workflow in [src/agents](src/agents)
- Retrieval layer: [src/retrieval](src/retrieval)
- Processing pipeline: [src/processing](src/processing)
- Storage layer: [src/storage](src/storage)

## Local development

### 1. Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL database
- Optional: OpenAI or Hugging Face compatible inference access

### 2. Setup

```bash
git clone https://github.com/BhushanPawar-01/itihas.git
cd itihas
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the backend

```bash
uvicorn backend.main:app --reload
```

The API will be available at http://localhost:8000.

### 4. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend typically runs at http://localhost:5173.

## Configuration

The application reads configuration from environment variables and the settings module in [config/settings.py](config/settings.py).

### Required runtime variables

| Variable | Required | Secret | Purpose |
|---|---:|---:|---|
| `HF_API_TOKEN` | Yes | Yes | Hugging Face inference authentication |
| `DB_URL` | Yes | Yes | PostgreSQL connection string |

### Optional variables

| Variable | Required | Secret | Purpose |
|---|---:|---:|---|
| `OPENAI_API_KEY` | No | Yes | OpenAI-compatible model access |
| `HF_MODEL` | No | No | Override the default Hugging Face model |
| `OLLAMA_BASE_URL` | No | No | Local Ollama endpoint |
| `OLLAMA_MODEL` | No | No | Local model name |

> Never commit secrets to the repository. Store them in your deployment platform or local environment.

## Deployment

This repository is configured for deployment to Hugging Face Spaces using GitHub Actions.

### GitHub → Hugging Face Spaces

The workflow in [.github/workflows/sync_to_hf.yml](.github/workflows/sync_to_hf.yml) automatically syncs the `main` branch to your Hugging Face Space on every push.

### Hugging Face Space setup

In your Hugging Face Space settings, configure: 

- runtime environment variables for the application
- the required secrets such as `HF_API_TOKEN` and `DB_URL`
- the container port as `7860`

### GitHub Actions secrets

For deployment sync, define the following secret in GitHub:

- `HF_TOKEN` — Hugging Face access token used by the workflow

## API endpoints

The backend exposes:

- `GET /health` for liveness checks
- query endpoints under `/api/v1` for historical analysis requests

## Testing and validation

Run the test suite and frontend build before shipping changes:

```bash
pytest
cd frontend && npm run build
```

## Notes

This project is intended for research, historical analysis, and experimentation. It should be configured with appropriate environment variables and access controls before being used in production environments.