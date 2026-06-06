import sys
from pathlib import Path

import psycopg2

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import DB_URL, DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_SSLMODE

def create_schema():
    print("Connecting to database to create schema...")
    if DB_URL:
        print("Using DATABASE_URL connection string")
        conn = psycopg2.connect(DB_URL, sslmode=DB_SSLMODE)
    else:
        print(f"Using DB_HOST={DB_HOST}, DB_PORT={DB_PORT}, DB_NAME={DB_NAME}")
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            sslmode=DB_SSLMODE
        )
        
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                  id SERIAL PRIMARY KEY,
                  doc_id TEXT NOT NULL,
                  chunk_index INTEGER NOT NULL,
                  text TEXT NOT NULL,
                  embedding vector(768),
                  source_type TEXT,
                  bias_tag TEXT,
                  language TEXT,
                  date TEXT,
                  confidence REAL,
                  created_at TIMESTAMPTZ DEFAULT NOW(),
                  UNIQUE(doc_id, chunk_index)
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_bias_tag ON chunks(bias_tag);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_language ON chunks(language);")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                  ON chunks USING ivfflat (embedding vector_cosine_ops)
                  WITH (lists = 100);
            """)
        conn.commit()
        print("Schema created successfully")
    finally:
        conn.close()

if __name__ == "__main__":
    create_schema()
