# 🔍 RAG Infrastructure & MCP Data Pipeline

> Production-grade RAG (Retrieval-Augmented Generation) infrastructure with MCP (Model Context Protocol) servers, data cleaning pipelines, and Claude-powered context retrieval.

## 🎯 What This Project Does

This system ingests **messy, multi-source real-world data**, cleans and normalizes it, chunks and embeds it into a vector store, and exposes it through:
- A **FastAPI REST layer** for developer queries
- An **MCP server** for Claude Code / AI agent context retrieval
- A **RAG pipeline** with retrieval quality evaluation

```
Raw Data Sources (CSV/JSON/DB/APIs)
         ↓
  Schema Remapping & Normalization
         ↓
  Validation & Quality Profiling
         ↓
  Chunking → Embedding → pgvector
         ↓
  ┌──────────────┬──────────────┐
  │  FastAPI     │  MCP Server  │
  │  REST API    │  for Claude  │
  └──────────────┴──────────────┘
```

---

## 🗂️ Project Structure

```
rag-mcp-pipeline/
├── src/
│   ├── ingestion/          # Multi-source data ingestion
│   │   ├── connectors.py   # CSV, JSON, Postgres, REST API connectors
│   │   └── loader.py       # Unified data loader with schema detection
│   ├── cleaning/           # Data cleaning & normalization
│   │   ├── normalizer.py   # Schema remapping, type casting, dedup
│   │   ├── validator.py    # Row-level validation with error reporting
│   │   └── profiler.py     # Data quality profiling & anomaly detection
│   ├── rag/                # RAG infrastructure
│   │   ├── chunker.py      # Semantic & fixed-size chunking strategies
│   │   ├── embedder.py     # Embedding generation (OpenAI / local)
│   │   ├── vector_store.py # pgvector indexing & similarity search
│   │   └── retriever.py    # Retrieval pipeline with re-ranking & eval
│   ├── mcp_server/         # MCP server for Claude Code integration
│   │   ├── server.py       # MCP server with registered tools
│   │   └── tools.py        # Data query, search, and context tools
│   ├── api/                # FastAPI developer API
│   │   ├── main.py         # App entrypoint & router registration
│   │   ├── routes.py       # Query, ingest, and health endpoints
│   │   └── schemas.py      # Pydantic request/response models
│   └── monitoring/         # Pipeline monitoring & alerting
│       ├── quality.py      # Anomaly detection & data quality checks
│       └── metrics.py      # Pipeline metrics & Prometheus export
├── tests/
│   ├── unit/               # Unit tests for each module
│   └── integration/        # End-to-end pipeline tests
├── infra/                  # Docker + deployment config
│   ├── docker-compose.yml
│   └── Dockerfile
├── .github/workflows/      # CI/CD
└── docs/                   # Architecture & runbooks
```

---

## 🚀 Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/itsnikhile/rag-mcp-pipeline.git
cd rag-mcp-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Start infrastructure
docker-compose up -d  # starts pgvector + Redis

# 3. Run the ingestion + cleaning pipeline
python -m src.ingestion.loader --source data/sample/

# 4. Build the vector index
python -m src.rag.embedder --index

# 5. Start the FastAPI server
uvicorn src.api.main:app --reload

# 6. Start the MCP server (for Claude Code)
python -m src.mcp_server.server
```

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Data Ingestion | Python, Pandas, SQLAlchemy, httpx |
| Cleaning & Validation | Pandas, Pydantic, Great Expectations |
| Chunking | LangChain text splitters |
| Embeddings | OpenAI `text-embedding-3-small` / sentence-transformers |
| Vector Store | pgvector (PostgreSQL extension) |
| RAG Retrieval | LangChain + custom re-ranking |
| MCP Server | Anthropic MCP SDK (Python) |
| API | FastAPI + Uvicorn |
| Monitoring | Prometheus + custom anomaly detection |
| CI/CD | GitHub Actions |
| Infrastructure | Docker Compose, PostgreSQL + pgvector |

---

## 📋 Key Features

- **Multi-source ingestion** — CSV, JSON, REST APIs, PostgreSQL with unified schema detection
- **Messy data handling** — automatic schema remapping, type coercion, deduplication
- **Validation pipeline** — row-level checks with detailed error reporting and quarantine
- **Semantic chunking** — context-aware document splitting with overlap management
- **Hybrid search** — dense vector + sparse BM25 retrieval with re-ranking
- **MCP tools** — `search_data`, `get_context`, `query_schema`, `profile_dataset` tools for Claude
- **Retrieval evaluation** — precision@k, recall@k, MRR scoring on test query sets
- **Data quality monitoring** — anomaly detection, null rate tracking, schema drift alerts

---

## License

MIT
