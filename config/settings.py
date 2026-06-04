from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / "config" / ".env")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_TEXT_DIR = PROJECT_ROOT / "data" / "processed" / "text"
PROCESSED_TRANSLATED_DIR = PROJECT_ROOT / "data" / "processed" / "translated"
PROCESSED_CHUNKS_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"
METADATA_DIR = PROJECT_ROOT / "data" / "processed" / "metadata"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
REGISTRY_PATH = PROJECT_ROOT / "data" / "registry.csv"

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
DB_URL = os.getenv("DATABASE_URL")
