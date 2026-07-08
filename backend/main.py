"""
Itihas FastAPI application entrypoint.

Run with:
    uvicorn backend.main:app --reload
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.routes.query import (
    QueryRequest,
    QueryResponse,
    StreamQueryRequest,
    query_history,
    router as query_router,
    stream_query,
)
from src.utils.logger import get_logger

# Absolute path to the built frontend
FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
FRONTEND_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        for key, value in FRONTEND_CACHE_HEADERS.items():
            response.headers[key] = value
        return response


log = get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Itihas API",
        description="Multi-agent adversarial reasoning system for Indian military and political history, 1600-1947.",
        version="0.1.0",
    )

    # CORS — permissive for local dev frontend.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(query_router, prefix="/api/v1")

    if FRONTEND_DIST.exists():
        app.mount(
            "/assets",
            NoCacheStaticFiles(directory=FRONTEND_DIST / "assets"),
            name="assets",
        )

        def _frontend_index_response() -> FileResponse:
            return FileResponse(
                str(FRONTEND_DIST / "index.html"),
                headers=FRONTEND_CACHE_HEADERS,
            )

        @app.get("/", include_in_schema=False)
        async def serve_frontend_root() -> FileResponse:
            return _frontend_index_response()

        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_frontend(full_path: str) -> FileResponse:
            """
            Catch-all for React Router — returns index.html for any non-/api path.
            FastAPI matches routes in registration order, so this only fires if no
            API route matched first. /api/* is registered before this catch-all.
            """
            return _frontend_index_response()

    @app.post("/query", include_in_schema=False, response_model=QueryResponse)
    async def legacy_query(request: QueryRequest) -> QueryResponse:
        return await query_history(request)

    @app.post("/query/stream", include_in_schema=False)
    async def legacy_stream_query(request: StreamQueryRequest):
        return await stream_query(request)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """
        Basic liveness check. Returns 200 if the process is up.
        """
        return {"status": "ok"}

    @app.on_event("startup")
    async def on_startup() -> None:
        log.info("Itihas API starting up")

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        log.info("Itihas API shutting down")

    return app


app = create_app()
