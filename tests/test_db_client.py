import os
import pytest
from src.storage import db_client
from src.storage.create_schema import create_schema
from config import settings
from dotenv import load_dotenv

@pytest.fixture(scope="module", autouse=True)
def setup_database():
    load_dotenv(".env.test", override=True)
    if os.getenv("DATABASE_URL"):
        settings.DB_URL = os.getenv("DATABASE_URL")
        # Update connection pool since settings changed
        db_client._pool = None
    
    # Optional: run schema creation in test DB if needed
    try:
        create_schema()
    except Exception as e:
        print("Schema creation skipped or failed:", e)

    with db_client.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE chunks RESTART IDENTITY;")

    yield

    with db_client.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE chunks RESTART IDENTITY;")


def test_upsert_and_get_chunks():
    chunk = {
        "doc_id": "test_doc_1",
        "chunk_index": 0,
        "text": "Hello world",
        "embedding": [0.1] * 768,
        "source_type": "test",
        "bias_tag": "neutral",
        "language": "en",
        "date": "2023-01-01",
        "confidence": 0.99
    }
    db_client.upsert_chunk(chunk)
    
    results = db_client.get_chunks_by_doc("test_doc_1")
    assert len(results) == 1
    assert results[0]["doc_id"] == "test_doc_1"
    assert results[0]["text"] == "Hello world"
    assert results[0]["bias_tag"] == "neutral"
    
    # Upsert again to update
    chunk["text"] = "Hello updated"
    db_client.upsert_chunk(chunk)
    
    results = db_client.get_chunks_by_doc("test_doc_1")
    assert len(results) == 1
    assert results[0]["text"] == "Hello updated"

def test_upsert_chunks_batch():
    chunks = [
        {"doc_id": "test_doc_2", "chunk_index": 0, "text": "A", "embedding": [0.2]*768},
        {"doc_id": "test_doc_2", "chunk_index": 1, "text": "B", "embedding": [0.3]*768},
    ]
    db_client.upsert_chunks_batch(chunks)
    
    results = db_client.get_chunks_by_doc("test_doc_2")
    assert len(results) == 2
    assert results[0]["text"] == "A"
    assert results[1]["text"] == "B"

def test_get_chunks_without_embeddings():
    chunks = [
        {"doc_id": "test_doc_3", "chunk_index": 0, "text": "No emb 1"},
        {"doc_id": "test_doc_3", "chunk_index": 1, "text": "No emb 2"}
    ]
    db_client.upsert_chunks_batch(chunks)
    
    results = db_client.get_chunks_without_embeddings(limit=10)
    # Check that they are returned
    doc3_results = [r for r in results if r["doc_id"] == "test_doc_3"]
    assert len(doc3_results) == 2
    assert "embedding" not in doc3_results[0]  # Check that only selected columns are returned

def test_search_by_embedding():
    # Insert some dummy chunks with embeddings
    chunks = [
        {"doc_id": "test_doc_4", "chunk_index": 0, "text": "Match 1", "embedding": [1.0] + [0.0]*767, "bias_tag": "A", "language": "en"},
        {"doc_id": "test_doc_4", "chunk_index": 1, "text": "Match 2", "embedding": [0.0, 1.0] + [0.0]*766, "bias_tag": "B", "language": "en"},
        {"doc_id": "test_doc_4", "chunk_index": 2, "text": "Match 3", "embedding": [0.0, 0.0, 1.0] + [0.0]*765, "bias_tag": "A", "language": "fr"},
    ]
    db_client.upsert_chunks_batch(chunks)
    
    query_emb = [1.0] + [0.0]*767
    results = db_client.search_by_embedding(query_emb, top_k=3)
    
    # We might have other documents, filter to test_doc_4
    filtered_results = [r for r in results if r["doc_id"] == "test_doc_4"]
    assert len(filtered_results) >= 1
    assert filtered_results[0]["text"] == "Match 1"  # Best match should be first
    
    # Test filters
    results_filtered = db_client.search_by_embedding(query_emb, top_k=3, filters={"bias_tags": ["B"]})
    filtered_docs = [r for r in results_filtered if r["doc_id"] == "test_doc_4"]
    assert len(filtered_docs) == 1
    assert filtered_docs[0]["text"] == "Match 2"
