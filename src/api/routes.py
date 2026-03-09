# src/api/routes.py
"""FastAPI route definitions for search, ingest, profile, and pipeline endpoints."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field

router = APIRouter()


# ── REQUEST / RESPONSE MODELS ─────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(5, ge=1, le=50)
    source_filter: str | None = None
    hybrid: bool = True


class SearchResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]
    total: int
    latency_ms: float


class IngestRequest(BaseModel):
    source_type: str = Field(..., description="csv | json | postgres | rest_api")
    source_path: str = Field(..., description="File path, DB query, or URL")
    source_id: str
    column_map: dict[str, str] | None = None
    text_columns: list[str] | None = None
    id_column: str | None = None
    chunk_strategy: str = "sentence"
    chunk_size: int = 512


class IngestResponse(BaseModel):
    source_id: str
    rows_ingested: int
    chunks_indexed: int
    quality_score: float
    issues: list[str]


class ProfileRequest(BaseModel):
    source_id: str


# ── ROUTES ────────────────────────────────────────────────────────────────────
@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, request: Request):
    """Semantic + hybrid search over indexed data."""
    import time
    vs = request.app.state.vector_store
    t0 = time.perf_counter()

    try:
        results = vs.search(
            query=req.query,
            top_k=req.top_k,
            source_filter=req.source_filter,
            hybrid=req.hybrid,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = round((time.perf_counter() - t0) * 1000, 1)

    return SearchResponse(
        query=req.query,
        results=[
            {
                "chunk_id": r.chunk_id,
                "content": r.content,
                "source_id": r.source_id,
                "score": round(r.score, 4),
                "rank": r.rank,
                "metadata": r.metadata,
            }
            for r in results
        ],
        total=len(results),
        latency_ms=latency,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, request: Request, background_tasks: BackgroundTasks):
    """Ingest, clean, chunk, embed, and index a data source."""
    from src.ingestion.connectors import CSVConnector, JSONConnector
    from src.cleaning.normalizer import SchemaNormalizer
    from src.cleaning.profiler import DataProfiler
    from src.rag.chunker import DocumentChunker, ChunkStrategy

    # Load data
    if req.source_type == "csv":
        connector = CSVConnector(req.source_path, source_id=req.source_id, column_map=req.column_map or {})
    elif req.source_type == "json":
        connector = JSONConnector(req.source_path, source_id=req.source_id)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported source_type: {req.source_type}")

    result = connector.read()
    if not result.success:
        raise HTTPException(status_code=422, detail={"errors": result.errors})

    # Clean & normalize
    normalizer = SchemaNormalizer(column_map=req.column_map or {})
    norm_result = normalizer.normalize(result.df)

    # Profile
    profiler = DataProfiler()
    profile = profiler.profile(norm_result.df, source=req.source_id)

    # Chunk
    strategy = ChunkStrategy(req.chunk_strategy)
    chunker = DocumentChunker(strategy=strategy, chunk_size=req.chunk_size)
    text_cols = req.text_columns or [c for c in norm_result.df.select_dtypes(include="object").columns
                                      if not c.startswith("_")][:3]
    chunks = chunker.chunk_dataframe(
        norm_result.df,
        text_cols=text_cols,
        id_col=req.id_column,
    )

    # Index
    vs = request.app.state.vector_store
    indexed = vs.index_chunks(chunks)

    return IngestResponse(
        source_id=req.source_id,
        rows_ingested=norm_result.final_rows,
        chunks_indexed=indexed,
        quality_score=profile.quality_score(),
        issues=norm_result.issues + profile.anomalies,
    )


@router.post("/profile")
async def profile_dataset(req: ProfileRequest, request: Request):
    """Run a data quality profile on an indexed source."""
    import psycopg2
    import pandas as pd
    from src.cleaning.profiler import DataProfiler

    try:
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        with conn.cursor() as cur:
            cur.execute("SELECT metadata FROM rag_chunks WHERE source_id = %s LIMIT 1000", [req.source_id])
            rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not rows:
        raise HTTPException(status_code=404, detail=f"Source not found: {req.source_id}")

    import json
    records = [json.loads(r[0]) for r in rows if r[0]]
    df = pd.json_normalize(records)

    profiler = DataProfiler()
    profile = profiler.profile(df, source=req.source_id)
    result = profile.to_dict()
    result["quality_score"] = profile.quality_score()
    return result


@router.get("/sources")
async def list_sources(request: Request):
    """List all indexed data sources."""
    import psycopg2
    import psycopg2.extras

    try:
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT source_id, COUNT(*) as chunk_count,
                       MIN(indexed_at) as first_indexed,
                       MAX(indexed_at) as last_indexed
                FROM rag_chunks
                GROUP BY source_id
                ORDER BY last_indexed DESC
            """)
            return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate")
async def evaluate_retrieval(request: Request, body: dict):
    """Evaluate retrieval quality on a labeled test set."""
    vs = request.app.state.vector_store
    test_queries = body.get("test_queries", [])
    top_k = body.get("top_k", 5)

    if not test_queries:
        raise HTTPException(status_code=400, detail="test_queries required")

    metrics = vs.evaluate_retrieval(test_queries, top_k=top_k)
    return metrics
