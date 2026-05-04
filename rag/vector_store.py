"""
rag/vector_store.py

SQLite + sqlite-vec vector store for TravelDude RAG.
Stores document chunks + embeddings in the same TravelDude.db,
keeping the entire system in a single file.

Schema added to existing DB:
  - rag_documents   : raw chunks with metadata
  - rag_embeddings  : sqlite-vec virtual table for ANN search

Embedding model: Anthropic's text-embedding via the API
(falls back to a lightweight sentence-transformers model if offline)
"""

import os
import json
import sqlite3
import hashlib
import struct
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "database", "TravelDude.db")

# Embedding dimension for claude / sentence-transformers fallback
EMBED_DIM = 384   # all-MiniLM-L6-v2 fallback dim
ANTHROPIC_EMBED_DIM = 1024  # voyage-lite-02-instruct via Anthropic


# ── Helpers ──────────────────────────────────────────────────────────────────

def _doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _pack_vector(vec: List[float]) -> bytes:
    """Pack float list → little-endian binary blob for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_vector(blob: bytes) -> List[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


# ── Schema ────────────────────────────────────────────────────────────────────

def init_rag_schema(conn: sqlite3.Connection, embed_dim: int = EMBED_DIM) -> None:
    """
    Add RAG tables to the existing TravelDude.db.
    Safe to call multiple times (uses IF NOT EXISTS).
    """
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS rag_documents (
            doc_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_hash    TEXT UNIQUE,
            source_type TEXT,          -- 'review' | 'note' | 'guide'
            destination TEXT,
            title       TEXT,
            content     TEXT,
            chunk_index INTEGER DEFAULT 0,
            metadata    TEXT,          -- JSON blob
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- sqlite-vec virtual table (requires sqlite-vec extension loaded)
        -- Falls back to plain cosine search on blob column if extension missing.
        CREATE TABLE IF NOT EXISTS rag_embeddings (
            doc_id    INTEGER PRIMARY KEY REFERENCES rag_documents(doc_id),
            embedding BLOB NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_rag_dest ON rag_documents(destination);
        CREATE INDEX IF NOT EXISTS idx_rag_type ON rag_documents(source_type);
    """)
    conn.commit()


# ── Embedding ─────────────────────────────────────────────────────────────────

def get_embedding(text: str, use_anthropic: bool = False) -> List[float]:
    """
    Generate embedding vector for a text chunk.

    Priority:
      1. sentence-transformers (local, free, no API key needed) — default
      2. Anthropic Voyage via API (higher quality, costs tokens) — opt-in

    sentence-transformers installs as: pip install sentence-transformers
    """
    if use_anthropic:
        return _anthropic_embed(text)
    return _local_embed(text)


def _local_embed(text: str) -> List[float]:
    """
    Local embedding with sentence-transformers all-MiniLM-L6-v2.
    384-dim, runs fully offline, no API key.
    """
    try:
        from sentence_transformers import SentenceTransformer
        _model_cache = getattr(_local_embed, "_model", None)
        if _model_cache is None:
            _local_embed._model = SentenceTransformer("all-MiniLM-L6-v2")
        vec = _local_embed._model.encode(text, normalize_embeddings=True)
        return vec.tolist()
    except ImportError:
        raise RuntimeError(
            "sentence-transformers not installed. Run:\n"
            "  pip install sentence-transformers\n"
            "Or set use_anthropic=True to use the Anthropic embedding API."
        )


def _anthropic_embed(text: str) -> List[float]:
    """
    Embedding via Anthropic's voyage-lite-02-instruct model.
    1024-dim, requires ANTHROPIC_API_KEY.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    # Anthropic exposes Voyage embeddings via their API
    resp = client.embeddings.create(
        model="voyage-lite-02-instruct",
        input=[text],
    )
    return resp.embeddings[0].embedding


# ── Cosine similarity (pure Python fallback) ──────────────────────────────────

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_document(
    content: str,
    destination: str,
    source_type: str,           # 'review' | 'note' | 'guide'
    title: str = "",
    metadata: dict = None,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    use_anthropic_embed: bool = False,
    db_path: str = DB_PATH,
) -> int:
    """
    Chunk a document and index all chunks into the vector store.
    Returns number of chunks indexed.
    """
    chunks = _chunk_text(content, chunk_size, chunk_overlap)
    conn = sqlite3.connect(db_path)
    init_rag_schema(conn)
    indexed = 0

    for i, chunk in enumerate(chunks):
        h = _doc_hash(chunk)
        # Skip duplicate chunks
        existing = conn.execute(
            "SELECT doc_id FROM rag_documents WHERE doc_hash=?", (h,)
        ).fetchone()
        if existing:
            continue

        vec = get_embedding(chunk, use_anthropic=use_anthropic_embed)
        blob = _pack_vector(vec)

        cur = conn.execute(
            """INSERT INTO rag_documents
               (doc_hash, source_type, destination, title, content, chunk_index, metadata)
               VALUES (?,?,?,?,?,?,?)""",
            (h, source_type, destination, title, chunk, i, json.dumps(metadata or {})),
        )
        doc_id = cur.lastrowid
        conn.execute(
            "INSERT INTO rag_embeddings (doc_id, embedding) VALUES (?,?)",
            (doc_id, blob),
        )
        indexed += 1

    conn.commit()
    conn.close()
    return indexed


def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Split text into overlapping word-count chunks."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    return [c for c in chunks if len(c.strip()) > 30]


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = 5,
    destination_filter: Optional[str] = None,
    source_type_filter: Optional[str] = None,
    use_anthropic_embed: bool = False,
    db_path: str = DB_PATH,
) -> List[Dict]:
    """
    Retrieve top-K most relevant chunks for a query.
    Uses pure-Python cosine similarity over stored embedding blobs.
    (sqlite-vec extension not required — works out of the box.)

    Returns list of dicts with keys:
      doc_id, destination, source_type, title, content, score, metadata
    """
    query_vec = get_embedding(query, use_anthropic=use_anthropic_embed)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Build SQL filter
    filters, params = [], []
    if destination_filter:
        filters.append("d.destination LIKE ?")
        params.append(f"%{destination_filter}%")
    if source_type_filter:
        filters.append("d.source_type = ?")
        params.append(source_type_filter)

    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    rows = conn.execute(
        f"""SELECT d.doc_id, d.destination, d.source_type, d.title,
                   d.content, d.metadata, e.embedding
            FROM rag_documents d
            JOIN rag_embeddings e ON d.doc_id = e.doc_id
            {where}""",
        params,
    ).fetchall()
    conn.close()

    if not rows:
        return []

    scored = []
    for row in rows:
        stored_vec = _unpack_vector(row["embedding"])
        score = _cosine_similarity(query_vec, stored_vec)
        scored.append({
            "doc_id":      row["doc_id"],
            "destination": row["destination"],
            "source_type": row["source_type"],
            "title":       row["title"],
            "content":     row["content"],
            "metadata":    json.loads(row["metadata"] or "{}"),
            "score":       score,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_index_stats(db_path: str = DB_PATH) -> Dict:
    """Return a summary of what's in the vector store."""
    try:
        conn = sqlite3.connect(db_path)
        total = conn.execute("SELECT COUNT(*) FROM rag_documents").fetchone()[0]
        by_type = conn.execute(
            "SELECT source_type, COUNT(*) FROM rag_documents GROUP BY source_type"
        ).fetchall()
        by_dest = conn.execute(
            "SELECT destination, COUNT(*) FROM rag_documents GROUP BY destination ORDER BY 2 DESC LIMIT 10"
        ).fetchall()
        conn.close()
        return {
            "total_chunks": total,
            "by_type":  {r[0]: r[1] for r in by_type},
            "top_destinations": {r[0]: r[1] for r in by_dest},
        }
    except Exception:
        return {"total_chunks": 0, "by_type": {}, "top_destinations": {}}
