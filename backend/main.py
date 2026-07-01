"""
Itihas FastAPI application entrypoint.

Run with:
    uvicorn backend.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.query import router as query_router
from src.utils.logger import get_logger

log = get_logger(__name__)

app = FastAPI(
    title="Itihas API",
    description="Multi-agent adversarial reasoning system for Indian military and political history, 1600-1947.",
    version="0.1.0",
)

# ── CORS — permissive for local dev frontend (React, Week 4) ──────────────
# Restrict allow_origins to actual deployed frontend origin(s) before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────
app.include_router(query_router, prefix="/api/v1")


@app.get("/health")
async def health() -> dict[str, str]:
    """
    Basic liveness check. Returns 200 if the process is up.
    Does not check downstream dependencies (Postgres, Ollama, HF API) —
    those are checked by src/utils/llm_client.py's check_hf_connection()
    and check_ollama_connection() if a deeper check is ever needed here.
    """
    return {"status": "ok"}


@app.on_event("startup")
async def on_startup() -> None:
    log.info("Itihas API starting up")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    log.info("Itihas API shutting down")