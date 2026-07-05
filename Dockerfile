# ── Stage 1: Build React frontend ────────────────────────────────────────────
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build
# Produces /app/frontend/dist/


# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM python:3.11-slim

# HuggingFace Spaces runs containers as a non-root user (UID 1000).
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python deps first (layer caches until requirements.txt changes)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# Copy source — matches actual repo layout: agents live under src/agents/
COPY src/       ./src/
COPY backend/   ./backend/
COPY config/    ./config/

# Copy built frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# HF Spaces free tier expects port 7860
EXPOSE 7860

USER appuser

# Workers=1: sentence-transformers loads one embedding model into memory.
# Multiple workers would each load it separately — wasted RAM on free tier for no gain at demo scale.
CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1"]