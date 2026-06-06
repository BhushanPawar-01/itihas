import time
from contextlib import contextmanager
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import execute_values
from config.settings import DB_URL, DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_SSLMODE, DB_POOL_MIN, DB_POOL_MAX
from src.utils.logger import get_logger

logger = get_logger(__name__)

_pool = None

def get_pool():
    global _pool
    if _pool is None:
        # Priority to connection string for Neon, fallback to individual parameters
        if DB_URL:
            _pool = ThreadedConnectionPool(
                minconn=DB_POOL_MIN,
                maxconn=DB_POOL_MAX,
                dsn=DB_URL,
                sslmode=DB_SSLMODE
            )
        else:
            _pool = ThreadedConnectionPool(
                minconn=DB_POOL_MIN,
                maxconn=DB_POOL_MAX,
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                sslmode=DB_SSLMODE
            )
    return _pool

@contextmanager
def get_connection():
    pool = get_pool()
    conn = pool.getconn()
    
    # Neon Tech Serverless optimization: Pre-ping the connection. 
    # If Neon scaled to zero and dropped the idle connection, we throw it away and get a fresh one.
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
    except psycopg2.OperationalError:
        logger.warning("Stale database connection detected (likely Neon scale-to-zero). Reconnecting...")
        pool.putconn(conn, close=True)
        conn = pool.getconn()

    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)

def _log_query_success(query_name: str, duration: float):
    logger.info("Query executed successfully", extra={"query_name": query_name, "duration_sec": duration})

def _log_query_failure(query_name: str, error: Exception):
    logger.error(f"Query failed: {error}", extra={"query_name": query_name})

def upsert_chunk(chunk: dict) -> None:
    query_name = "upsert_chunk"
    start_time = time.time()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chunks (doc_id, chunk_index, text, embedding, source_type,
                        bias_tag, language, date, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id, chunk_index) DO UPDATE SET
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        source_type = EXCLUDED.source_type,
                        bias_tag = EXCLUDED.bias_tag,
                        language = EXCLUDED.language,
                        date = EXCLUDED.date,
                        confidence = EXCLUDED.confidence;
                """, (
                    chunk.get("doc_id"),
                    chunk.get("chunk_index"),
                    chunk.get("text"),
                    chunk.get("embedding"),
                    chunk.get("source_type"),
                    chunk.get("bias_tag"),
                    chunk.get("language"),
                    chunk.get("date"),
                    chunk.get("confidence")
                ))
        _log_query_success(query_name, time.time() - start_time)
    except Exception as e:
        _log_query_failure(query_name, e)
        raise

def upsert_chunks_batch(chunks: list[dict]) -> None:
    query_name = "upsert_chunks_batch"
    start_time = time.time()
    query = """
        INSERT INTO chunks (doc_id, chunk_index, text, embedding, source_type,
            bias_tag, language, date, confidence)
        VALUES %s
        ON CONFLICT (doc_id, chunk_index) DO UPDATE SET
            text = EXCLUDED.text,
            embedding = EXCLUDED.embedding,
            source_type = EXCLUDED.source_type,
            bias_tag = EXCLUDED.bias_tag,
            language = EXCLUDED.language,
            date = EXCLUDED.date,
            confidence = EXCLUDED.confidence;
    """
    values = [
        (
            c.get("doc_id"),
            c.get("chunk_index"),
            c.get("text"),
            c.get("embedding"),
            c.get("source_type"),
            c.get("bias_tag"),
            c.get("language"),
            c.get("date"),
            c.get("confidence")
        )
        for c in chunks
    ]
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                execute_values(cur, query, values)
        _log_query_success(query_name, time.time() - start_time)
    except Exception as e:
        _log_query_failure(query_name, e)
        raise

def get_chunks_by_doc(doc_id: str) -> list[dict]:
    query_name = "get_chunks_by_doc"
    start_time = time.time()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM chunks WHERE doc_id = %s ORDER BY chunk_index", (doc_id,))
                columns = [desc[0] for desc in cur.description]
                results = [dict(zip(columns, row)) for row in cur.fetchall()]
        _log_query_success(query_name, time.time() - start_time)
        return results
    except Exception as e:
        _log_query_failure(query_name, e)
        raise

def get_chunks_without_embeddings(limit: int = 1000) -> list[dict]:
    query_name = "get_chunks_without_embeddings"
    start_time = time.time()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, doc_id, chunk_index, text 
                    FROM chunks 
                    WHERE embedding IS NULL 
                    LIMIT %s
                """, (limit,))
                columns = [desc[0] for desc in cur.description]
                results = [dict(zip(columns, row)) for row in cur.fetchall()]
        _log_query_success(query_name, time.time() - start_time)
        return results
    except Exception as e:
        _log_query_failure(query_name, e)
        raise

def update_embedding(chunk_id: int, embedding: list[float]) -> None:
    query_name = "update_embedding"
    start_time = time.time()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE chunks SET embedding = %s WHERE id = %s", (embedding, chunk_id))
        _log_query_success(query_name, time.time() - start_time)
    except Exception as e:
        _log_query_failure(query_name, e)
        raise

def search_by_embedding(query_embedding: list[float], top_k: int = 20, filters: dict | None = None) -> list[dict]:
    query_name = "search_by_embedding"
    start_time = time.time()
    filters = filters or {}
    
    query = """
        SELECT id, doc_id, chunk_index, text, bias_tag, language, date,
               1 - (embedding <=> %s::vector) AS score
        FROM chunks
        WHERE embedding IS NOT NULL
    """
    params = [query_embedding]
    
    if filters.get("bias_tags"):
        query += " AND bias_tag = ANY(%s)"
        params.append(filters["bias_tags"])
        
    if filters.get("languages"):
        query += " AND language = ANY(%s)"
        params.append(filters["languages"])
        
    query += " ORDER BY embedding <=> %s::vector LIMIT %s;"
    params.extend([query_embedding, top_k])
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
                columns = [desc[0] for desc in cur.description]
                results = [dict(zip(columns, row)) for row in cur.fetchall()]
        _log_query_success(query_name, time.time() - start_time)
        return results
    except Exception as e:
        _log_query_failure(query_name, e)
        raise

def get_all_texts_for_bm25() -> list[dict]:
    query_name = "get_all_texts_for_bm25"
    start_time = time.time()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, doc_id, chunk_index, text FROM chunks ORDER BY id")
                columns = [desc[0] for desc in cur.description]
                results = [dict(zip(columns, row)) for row in cur.fetchall()]
                
        if len(results) > 100000:
            logger.warning(f"Row count exceeds 100000: {len(results)} rows returned.")
            
        _log_query_success(query_name, time.time() - start_time)
        return results
    except Exception as e:
        _log_query_failure(query_name, e)
        raise
