import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Resolve paths relative to the project root
_base_dir = Path(__file__).resolve().parents[1]

RAW_DIR = _base_dir / Path("data/raw")
PROCESSED_TEXT_DIR = _base_dir / Path("data/processed/text")
PROCESSED_TRANSLATED_DIR = _base_dir / Path("data/processed/translated")
PROCESSED_CHUNKS_DIR = _base_dir / Path("data/processed/chunks")
METADATA_DIR = _base_dir / Path("data/processed/metadata")
REGISTRY_PATH = _base_dir / Path("data/registry.csv")
REPORTS_DIR = _base_dir / Path("data/reports")

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
DB_URL = os.getenv("DB_URL")
DB_SSLMODE = os.getenv("DB_SSLMODE", "require")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "itihas")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_POOL_MIN = 1
DB_POOL_MAX = 5
EMBEDDING_DIM = 768

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_BATCH_SIZE = 64

BM25_TOP_K = 20
DENSE_TOP_K = 20
RRF_K = 60
RETRIEVAL_TOP_K = 10
RERANKER_ENABLED = False
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"