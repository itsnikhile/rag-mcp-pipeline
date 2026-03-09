# src/api/main.py
"""
FastAPI developer API — exposes RAG search, data ingestion,
profiling, and pipeline management endpoints.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ── APP LIFECYCLE ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG API server...")
    vs = VectorStore(
        connection_string=os.environ["DATABASE_URL"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )
    vs.setup()
    app.state.vector_store = vs
    logger.info("Vector store ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="RAG Infrastructure API",
    description="Data ingestion, cleaning, embedding, and retrieval API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_timing_header(request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(round((time.perf_counter() - t0) * 1000, 1))
    return response

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "rag-mcp-pipeline"}
