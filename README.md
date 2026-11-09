# 🔍 RAG Infrastructure & MCP Data Pipeline

[![CI](https://github.com/itsnikhile/rag-mcp-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/itsnikhile/rag-mcp-pipeline/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://github.com/itsnikhile/rag-mcp-pipeline/blob/master/.github/workflows/ci.yml)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](#-tech-stack)
[![pgvector](https://img.shields.io/badge/Vector%20Store-pgvector-336791?logo=postgresql&logoColor=white)](#-tech-stack)
[![GitHub stars](https://img.shields.io/github/stars/itsnikhile/rag-mcp-pipeline)](https://github.com/itsnikhile/rag-mcp-pipeline/stargazers)

> Production-grade RAG (Retrieval-Augmented Generation) infrastructure with MCP (Model Context Protocol) servers, data cleaning pipelines, and Claude-powered context retrieval.

Built by [Nikhil E](https://github.com/itsnikhile) — AI & Data Engineer | Cloud Architect.

## 🎯 What This Project Does

Most RAG demos start with clean PDFs. Real enterprise data doesn't look like that — it arrives as messy CSVs, inconsistent JSON, database tables, and API responses, with mismatched column names, mixed types, duplicates, and missing values.

This system handles that reality end to end. It ingests **messy, multi-source real-world data**, cleans and normalizes it, chunks and embeds it into a vector store, and exposes it through:

- A **FastAPI REST layer** for developer queries — search, ingest, profile, and evaluation endpoints
- An **MCP server** for Claude Code / AI agent context retrieval
- A **RAG pipeline** with retrieval quality evaluation

## ✨ Highlights

- **Messy data in, grounded context out** — automatic column-alias remapping (`email_address`, `e_mail`, `mail` → `email`), type coercion, deduplication, and text / date standardization before anything is embedded
- **Data quality is measured, not assumed** — every source is profiled for null rates, duplicate rate, cardinality and outlier anomalies, and schema drift, with a 0.0–1.0 quality score returned on ingest
- **Hybrid retrieval, not just vector search** — dense embeddings fused with Postgres full-text search using Reciprocal Rank Fusion (RRF), stored in pgvector with an HNSW index
- **Built for agents, not just humans** — 5 MCP tools let Claude Code search, inspect, profile, and query the indexed data directly
- **Evaluation is a first-class endpoint** — precision@k, recall@k, and MRR on labeled test queries, via the API or the vector store directly

## 🛠️ MCP Tools

The MCP server (`rag-data-server`, stdio transport) registers 5 tools for Claude Code and other MCP clients:

| Tool | What it does |
|---|---|
| `search_data` | Semantic / hybrid search over indexed chunks |
| `get_context` | Retrieve grounded context chunks for a question |
| `query_schema` | Inspect the schema of an indexed source |
| `profile_dataset` | Run a data quality profile on a source |
| `run_query` | Run a query against the underlying PostgreSQL data |

## ▶️ Demo

Once the API is running (see Quick Start below):

**1. Ingest a CSV — it is cleaned, profiled, chunked, embedded, and indexed in one call:**

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "csv",
    "source_path": "path/to/your.csv",
    "source_id": "my_source",
    "text_columns": ["description"]
  }'
```

The response reports `rows_ingested`, `chunks_indexed`, a data `quality_score`, and any `issues` found during cleaning and profiling.

**2. Run a hybrid search:**

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "your natural language question",
    "top_k": 5
  }'
```

Each result returns the chunk content, `source_id`, relevance `score`, `rank`, and metadata — plus the query `latency_ms`.

**3. List what's indexed, or evaluate retrieval quality:**

```bash
curl http://localhost:8000/sources

curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "top_k": 5,
    "test_queries": [
      { "query": "your test question", "relevant_ids": ["chunk_id_1"] }
    ]
  }'
```

**4. Or skip the API and let Claude use the data directly** — point an MCP client at the server:

```json
{
  "mcpServers": {
    "rag-data-server": {
      "command": "python",
      "args": ["-m", "src.mcp_server.server"],
      "env": {
        "DATABASE_URL": "postgresql://raguser:ragpass@localhost:5432/ragdb",
        "OPENAI_API_KEY": "your-openai-api-key"
      }
    }
  }
}
```

## 🏗️ Architecture

```mermaid
flowchart TD
    A["Raw sources<br/>CSV · JSON · PostgreSQL · REST APIs"] --> B["Connectors<br/>src/ingestion/connectors.py"]
    B --> C["SchemaNormalizer<br/>alias remapping · type coercion · dedup"]
    C --> D["DataProfiler<br/>null / duplicate / anomaly checks<br/>quality score + schema drift"]
    D --> E["DocumentChunker<br/>fixed · sentence · paragraph<br/>semantic · structured"]
    E --> F["VectorStore<br/>OpenAI text-embedding-3-small · 1536 dims"]
    F --> G[("PostgreSQL + pgvector<br/>rag_chunks table<br/>HNSW + full-text GIN indexes")]
    G --> H["Hybrid search<br/>dense + full-text, fused with RRF"]
    H --> I["FastAPI REST API<br/>/search · /ingest · /profile<br/>/sources · /evaluate"]
    H --> J["MCP Server<br/>search_data · get_context · query_schema<br/>profile_dataset · run_query"]
    I --> K["Developers / applications"]
    J --> L["Claude Code / AI agents"]
```

How a query flows: the question is embedded with the same model used at indexing time, dense (cosine, HNSW) and full-text results are retrieved in parallel, and the two ranked lists are fused with Reciprocal Rank Fusion (`k = 60`) into a single ranked result set with source attribution on every chunk.

## 🚀 Quick Start

### Prerequisites

- Python 3.11
- Docker (for the pgvector database)
- An OpenAI API key (used for embeddings)

### 1. Clone & install

```bash
git clone https://github.com/itsnikhile/rag-mcp-pipeline.git
cd rag-mcp-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the vector database

```bash
docker compose -f infra/docker-compose.yml up -d db
```

This starts PostgreSQL 16 with the pgvector extension (`pgvector/pgvector:pg16`) on port 5432.

### 3. Configure environment

```bash
# Local development defaults — matches infra/docker-compose.yml and CI.
# Use a different password anywhere beyond your own machine.
export DATABASE_URL="postgresql://raguser:ragpass@localhost:5432/ragdb"
export OPENAI_API_KEY="your-openai-api-key"
```

### 4. Start the FastAPI server

```bash
uvicorn src.api.main:app --reload --port 8000
```

### 5. Start the MCP server (for Claude Code)

In a second terminal, with the same environment variables set:

```bash
python -m src.mcp_server.server
```

Then ingest and search using the examples in the Demo section above. Ingestion runs through the API — there is no separate loader CLI to run first.

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/search` | Semantic + hybrid search over indexed data (`query`, `top_k` 1–50, optional `source_filter`, `hybrid` on by default) |
| `POST` | `/ingest` | Ingest, clean, chunk, embed, and index a source (CSV / JSON via the API; returns rows, chunks, quality score, and issues) |
| `POST` | `/profile` | Data quality profile for an indexed source, including its quality score |
| `GET` | `/sources` | List indexed sources with chunk counts and first / last indexed times |
| `POST` | `/evaluate` | Retrieval evaluation on labeled test queries (precision@k, recall@k, MRR) |

## 📊 Results & Evaluation

Retrieval quality is evaluated with the same code path used in production — not a separate offline script:

- **Metrics:** precision@k, recall@k, and MRR, computed by `VectorStore.evaluate_retrieval()` and exposed through `POST /evaluate`
- **Inputs:** labeled test queries in the form `{ "query": "...", "relevant_ids": ["chunk_id", ...] }`
- **Search latency** is returned on every `/search` response (`latency_ms`), so retrieval speed can be tracked alongside quality
- **Data quality** is scored on every ingest (0.0–1.0, penalizing null rate, duplicate rate, and detected anomalies), so a bad source is visible before it degrades answers

Scores depend on your data and your labeled test set — run `/evaluate` against your own sources to get numbers that mean something for your use case.

## 🗂️ Project Structure

```
rag-mcp-pipeline/
├── src/
│   ├── ingestion/
│   │   └── connectors.py   # CSV, JSON, PostgreSQL, REST API connectors
│   ├── cleaning/
│   │   ├── normalizer.py   # Schema remapping, type casting, dedup
│   │   └── profiler.py     # Data quality profiling, anomaly & drift detection
│   ├── rag/
│   │   ├── chunker.py      # Fixed, sentence, paragraph, semantic & structured chunking
│   │   └── vector_store.py # pgvector indexing, hybrid search, re-ranking & evaluation
│   ├── mcp_server/
│   │   └── server.py       # MCP server (rag-data-server) with 5 registered tools
│   └── api/
│       ├── main.py         # FastAPI app entrypoint & lifecycle
│       └── routes.py       # Search, ingest, profile, sources & evaluate endpoints
├── tests/
│   └── unit/
│       └── test_pipeline.py
├── infra/
│   └── docker-compose.yml  # pgvector database, API, and MCP services
├── docs/
│   └── rag_mcp_pipeline_presentation.pptx
├── .github/workflows/
│   └── ci.yml              # Lint + unit tests on GitHub Actions
└── requirements.txt
```

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data Ingestion | Pandas, SQLAlchemy, httpx |
| Cleaning & Profiling | Pandas, NumPy, Pydantic — custom normalizer & profiler |
| Chunking | Custom `DocumentChunker` — fixed, sentence, paragraph, semantic, and structured strategies (default: sentence, 512 tokens, 64 overlap) |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dimensions, batched) |
| Vector Store | PostgreSQL 16 + pgvector — HNSW index for vector search, GIN index for full-text search |
| Retrieval | Hybrid dense + full-text search with Reciprocal Rank Fusion |
| MCP Server | MCP Python SDK, stdio transport |
| API | FastAPI + Uvicorn, Pydantic request / response models |
| Testing & CI | pytest, Ruff, Black — GitHub Actions with a pgvector service container |
| Infrastructure | Docker Compose |

## 📋 Key Features

- **Multi-source ingestion** — CSV, JSON, PostgreSQL, and REST API connectors returning normalized DataFrames with provenance (`_source_id`, `_ingested_at`, `_row_hash`) on every row
- **Messy CSV handling** — automatic delimiter detection and multi-encoding fallback (UTF-8 → latin-1 → cp1252 → UTF-8-SIG)
- **Schema normalization** — built-in alias map for common fields (id, name, email, phone, address, dates, amounts, and more), fuzzy column matching, type coercion, and deduplication
- **Validation & profiling** — per-column null / unique rates, outlier detection, dataset anomalies, schema-drift detection against a reference profile, and a single quality score per source
- **Chunking strategies** — fixed-size with overlap, sentence, paragraph, semantic, and structured (field-aware for tabular data), with source attribution preserved on every chunk
- **Hybrid search** — dense vector + full-text retrieval fused with RRF, with optional per-source filtering
- **MCP tools** — `search_data`, `get_context`, `query_schema`, `profile_dataset`, and `run_query` for Claude Code and other MCP clients
- **Retrieval evaluation** — precision@k, recall@k, and MRR on labeled test query sets

## 🧪 Testing & CI

```bash
pytest tests/unit
ruff check src/ tests/
black --check src/ tests/
```

GitHub Actions (`.github/workflows/ci.yml`) runs the same lint and unit tests on Python 3.11, against a live pgvector service container.

## 🗺️ Roadmap

- Add a Dockerfile so the `api` and `mcp` services in `infra/docker-compose.yml` build and run fully containerized
- Expose the PostgreSQL and REST API connectors through `POST /ingest` (today the API route handles CSV / JSON; all four connectors are available in code)
- Publish evaluation results on a sample dataset as a worked example

## License

MIT
