# src/rag/vector_store.py
"""
pgvector-backed vector store with hybrid search (dense + BM25),
similarity search, re-ranking, and retrieval quality evaluation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import openai
import psycopg2
import psycopg2.extras
from openai import OpenAI

from src.rag.chunker import Chunk

logger = logging.getLogger(__name__)

EMBED_MODEL   = "text-embedding-3-small"
EMBED_DIM     = 1536
TABLE_NAME    = "rag_chunks"


@dataclass
class SearchResult:
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    source_id: str
    score: float
    rank: int


@dataclass
class RetrievalEval:
    query: str
    results: list[SearchResult]
    latency_ms: float
    precision_at_k: float | None = None
    recall_at_k: float | None = None
    mrr: float | None = None


class VectorStore:
    """
    pgvector-backed vector store.
    Handles embedding generation, indexing, and hybrid retrieval.
    """

    def __init__(self, connection_string: str, openai_api_key: str):
        self.conn_str = connection_string
        self.client = OpenAI(api_key=openai_api_key)
        self._conn = None

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.conn_str)
        return self._conn

    # ── SCHEMA SETUP ────────────────────────────────────────────────────────
    def setup(self):
        """Create pgvector extension and chunks table."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                    id            SERIAL PRIMARY KEY,
                    chunk_id      TEXT UNIQUE NOT NULL,
                    source_id     TEXT NOT NULL,
                    content       TEXT NOT NULL,
                    metadata      JSONB DEFAULT '{{}}',
                    embedding     vector({EMBED_DIM}),
                    chunk_index   INT,
                    total_chunks  INT,
                    strategy      TEXT,
                    indexed_at    TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            # HNSW index for fast ANN search
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {TABLE_NAME}_embedding_idx
                ON {TABLE_NAME} USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
            # Full-text search index for BM25 hybrid
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {TABLE_NAME}_content_fts_idx
                ON {TABLE_NAME} USING gin(to_tsvector('english', content));
            """)
            conn.commit()
        logger.info("Vector store schema ready")

    # ── EMBEDDING ───────────────────────────────────────────────────────────
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch embed texts using OpenAI API."""
        results = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(model=EMBED_MODEL, input=batch)
            results.extend([e.embedding for e in response.data])
        return results

    # ── INDEXING ─────────────────────────────────────────────────────────────
    def index_chunks(self, chunks: list[Chunk], batch_size: int = 50) -> int:
        """Embed and index a list of Chunk objects. Returns count indexed."""
        if not chunks:
            return 0

        indexed = 0
        conn = self._get_conn()

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.content for c in batch]
            embeddings = self.embed_texts(texts)

            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    f"""
                    INSERT INTO {TABLE_NAME}
                        (chunk_id, source_id, content, metadata, embedding, chunk_index, total_chunks, strategy)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        content     = EXCLUDED.content,
                        embedding   = EXCLUDED.embedding,
                        metadata    = EXCLUDED.metadata,
                        indexed_at  = NOW()
                    """,
                    [
                        (
                            c.chunk_id,
                            c.source_id,
                            c.content,
                            json.dumps(c.metadata),
                            str(emb),
                            c.chunk_index,
                            c.total_chunks,
                            c.strategy,
                        )
                        for c, emb in zip(batch, embeddings)
                    ],
                )
                conn.commit()
            indexed += len(batch)
            logger.info(f"Indexed {indexed}/{len(chunks)} chunks")

        return indexed

    # ── RETRIEVAL ────────────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str | None = None,
        hybrid: bool = True,
    ) -> list[SearchResult]:
        """
        Retrieve relevant chunks for a query.
        Uses hybrid search (dense vector + BM25) when hybrid=True.
        """
        t0 = time.perf_counter()
        [query_embedding] = self.embed_texts([query])

        if hybrid:
            results = self._hybrid_search(query, query_embedding, top_k, source_filter)
        else:
            results = self._dense_search(query_embedding, top_k, source_filter)

        latency = (time.perf_counter() - t0) * 1000
        logger.debug(f"Search completed in {latency:.1f}ms, {len(results)} results")
        return results

    def _dense_search(self, embedding: list[float], top_k: int, source_filter: str | None) -> list[SearchResult]:
        conn = self._get_conn()
        filter_clause = "AND source_id = %s" if source_filter else ""
        params = [str(embedding), top_k] if not source_filter else [str(embedding), source_filter, top_k]

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT chunk_id, content, metadata, source_id,
                       1 - (embedding <=> %s::vector) AS score
                FROM {TABLE_NAME}
                {filter_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, [str(embedding)] + ([source_filter] if source_filter else []) + [top_k])
            rows = cur.fetchall()

        return [
            SearchResult(
                chunk_id=r["chunk_id"], content=r["content"],
                metadata=r["metadata"] or {}, source_id=r["source_id"],
                score=float(r["score"]), rank=i + 1,
            )
            for i, r in enumerate(rows)
        ]

    def _hybrid_search(self, query: str, embedding: list[float],
                       top_k: int, source_filter: str | None) -> list[SearchResult]:
        """Reciprocal Rank Fusion of dense vector + BM25 results."""
        dense = self._dense_search(embedding, top_k * 2, source_filter)

        conn = self._get_conn()
        filter_clause = "AND source_id = %s" if source_filter else ""
        params = [query, query] + ([source_filter] if source_filter else []) + [top_k * 2]

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT chunk_id, content, metadata, source_id,
                       ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) AS score
                FROM {TABLE_NAME}
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                {filter_clause}
                ORDER BY score DESC
                LIMIT %s
            """, params)
            bm25_rows = cur.fetchall()

        bm25 = [
            SearchResult(
                chunk_id=r["chunk_id"], content=r["content"],
                metadata=r["metadata"] or {}, source_id=r["source_id"],
                score=float(r["score"]), rank=i + 1,
            )
            for i, r in enumerate(bm25_rows)
        ]

        return self._rrf_fusion(dense, bm25, top_k)

    def _rrf_fusion(self, dense: list[SearchResult], bm25: list[SearchResult],
                    top_k: int, k: int = 60) -> list[SearchResult]:
        """Reciprocal Rank Fusion scoring."""
        scores: dict[str, float] = {}
        chunk_map: dict[str, SearchResult] = {}

        for rank, result in enumerate(dense):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0) + 1 / (k + rank + 1)
            chunk_map[result.chunk_id] = result

        for rank, result in enumerate(bm25):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0) + 1 / (k + rank + 1)
            chunk_map[result.chunk_id] = result

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            SearchResult(
                chunk_id=cid, content=chunk_map[cid].content,
                metadata=chunk_map[cid].metadata, source_id=chunk_map[cid].source_id,
                score=score, rank=i + 1,
            )
            for i, (cid, score) in enumerate(ranked)
        ]

    # ── RETRIEVAL EVALUATION ─────────────────────────────────────────────────
    def evaluate_retrieval(
        self,
        test_queries: list[dict],  # [{"query": str, "relevant_ids": list[str]}]
        top_k: int = 5,
    ) -> dict:
        """Evaluate retrieval quality on a labeled test set."""
        precisions, recalls, reciprocal_ranks = [], [], []

        for item in test_queries:
            query = item["query"]
            relevant = set(item["relevant_ids"])
            results = self.search(query, top_k=top_k)
            retrieved_ids = [r.source_id for r in results]

            hits = sum(1 for rid in retrieved_ids if rid in relevant)
            precisions.append(hits / top_k)
            recalls.append(hits / max(len(relevant), 1))

            rr = 0.0
            for rank, rid in enumerate(retrieved_ids, 1):
                if rid in relevant:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

        return {
            f"precision@{top_k}": round(np.mean(precisions), 4),
            f"recall@{top_k}":    round(np.mean(recalls), 4),
            "mrr":                round(np.mean(reciprocal_ranks), 4),
            "num_queries":        len(test_queries),
        }
