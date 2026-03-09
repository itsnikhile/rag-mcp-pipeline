# src/mcp_server/server.py
"""
MCP (Model Context Protocol) server exposing data tools for Claude Code.
Tools: search_data, get_context, query_schema, profile_dataset, run_query
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.rag.vector_store import VectorStore
from src.cleaning.profiler import DataProfiler
from src.ingestion.connectors import PostgresConnector

logger = logging.getLogger(__name__)

app = Server("rag-data-server")

# ── DEPENDENCY INITIALIZATION ────────────────────────────────────────────────
def get_vector_store() -> VectorStore:
    return VectorStore(
        connection_string=os.environ["DATABASE_URL"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

def get_profiler() -> DataProfiler:
    return DataProfiler()


# ── TOOL DEFINITIONS ─────────────────────────────────────────────────────────
@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_data",
            description=(
                "Search the indexed data using semantic similarity and keyword matching. "
                "Returns the most relevant data chunks for a given query. "
                "Use this to find relevant records, documents, or context."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    },
                    "source_filter": {
                        "type": "string",
                        "description": "Filter results to a specific data source ID (optional)"
                    },
                    "hybrid": {
                        "type": "boolean",
                        "description": "Use hybrid search (dense + BM25). Default: true",
                        "default": True
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_context",
            description=(
                "Get rich context for a specific entity or topic. "
                "Retrieves all relevant chunks and assembles them into a coherent context block. "
                "Best for: understanding a specific record, entity, or concept in depth."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The entity, topic, or concept to get context for"
                    },
                    "source_id": {
                        "type": "string",
                        "description": "Specific source document/record ID to retrieve context from"
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens in the assembled context (default: 2000)",
                        "default": 2000
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="query_schema",
            description=(
                "Inspect the schema of available datasets. "
                "Returns column names, data types, null rates, and sample values. "
                "Use this to understand what data is available before querying."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_id": {
                        "type": "string",
                        "description": "Source dataset ID to inspect. Leave empty to list all sources."
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="profile_dataset",
            description=(
                "Run a full data quality profile on a dataset. "
                "Returns null rates, cardinality, anomalies, duplicates, and quality score. "
                "Use this to assess data quality before using it in analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_id": {
                        "type": "string",
                        "description": "The source dataset ID to profile"
                    }
                },
                "required": ["source_id"]
            }
        ),
        Tool(
            name="run_sql",
            description=(
                "Run a read-only SQL query against the data warehouse. "
                "Only SELECT statements are permitted. "
                "Use this for precise analytical queries or aggregations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SELECT SQL query to execute"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum rows to return (default: 100, max: 1000)",
                        "default": 100
                    }
                },
                "required": ["sql"]
            }
        ),
    ]


# ── TOOL HANDLERS ─────────────────────────────────────────────────────────────
@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:

    if name == "search_data":
        return await _handle_search(arguments)
    elif name == "get_context":
        return await _handle_get_context(arguments)
    elif name == "query_schema":
        return await _handle_query_schema(arguments)
    elif name == "profile_dataset":
        return await _handle_profile_dataset(arguments)
    elif name == "run_sql":
        return await _handle_run_sql(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _handle_search(args: dict) -> list[TextContent]:
    vs = get_vector_store()
    results = vs.search(
        query=args["query"],
        top_k=args.get("top_k", 5),
        source_filter=args.get("source_filter"),
        hybrid=args.get("hybrid", True),
    )

    if not results:
        return [TextContent(type="text", text="No results found for this query.")]

    output = [f"Found {len(results)} results for: '{args['query']}'\n"]
    for r in results:
        output.append(
            f"--- Result {r.rank} (score: {r.score:.4f}) ---\n"
            f"Source: {r.source_id}\n"
            f"Content: {r.content}\n"
            f"Metadata: {json.dumps(r.metadata, indent=2)}\n"
        )

    return [TextContent(type="text", text="\n".join(output))]


async def _handle_get_context(args: dict) -> list[TextContent]:
    vs = get_vector_store()
    topic = args["topic"]
    source_id = args.get("source_id")
    max_tokens = args.get("max_tokens", 2000)

    results = vs.search(query=topic, top_k=10, source_filter=source_id)

    # Assemble context within token budget
    context_parts = []
    token_count = 0
    for r in results:
        chunk_tokens = len(r.content.split())
        if token_count + chunk_tokens > max_tokens:
            break
        context_parts.append(r.content)
        token_count += chunk_tokens

    if not context_parts:
        return [TextContent(type="text", text=f"No context found for: {topic}")]

    context = f"Context for '{topic}' ({token_count} tokens from {len(context_parts)} chunks):\n\n"
    context += "\n\n---\n\n".join(context_parts)

    return [TextContent(type="text", text=context)]


async def _handle_query_schema(args: dict) -> list[TextContent]:
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    source_id = args.get("source_id")

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if source_id:
            cur.execute("""
                SELECT source_id, COUNT(*) as chunk_count,
                       MIN(indexed_at) as first_indexed, MAX(indexed_at) as last_indexed,
                       jsonb_object_keys(metadata) as metadata_key
                FROM rag_chunks WHERE source_id = %s
                GROUP BY source_id, metadata_key
            """, [source_id])
        else:
            cur.execute("""
                SELECT source_id, COUNT(*) as chunk_count,
                       MIN(indexed_at) as first_indexed, MAX(indexed_at) as last_indexed
                FROM rag_chunks GROUP BY source_id ORDER BY chunk_count DESC
            """)
        rows = cur.fetchall()

    output = json.dumps([dict(r) for r in rows], indent=2, default=str)
    return [TextContent(type="text", text=f"Schema info:\n{output}")]


async def _handle_profile_dataset(args: dict) -> list[TextContent]:
    import psycopg2
    import pandas as pd

    source_id = args["source_id"]
    conn = psycopg2.connect(os.environ["DATABASE_URL"])

    with conn.cursor() as cur:
        cur.execute("SELECT content, metadata FROM rag_chunks WHERE source_id = %s", [source_id])
        rows = cur.fetchall()

    if not rows:
        return [TextContent(type="text", text=f"No data found for source: {source_id}")]

    records = [r[1] for r in rows if r[1]]
    df = pd.json_normalize(records)

    profiler = get_profiler()
    profile = profiler.profile(df, source=source_id)

    return [TextContent(type="text", text=json.dumps(profile.to_dict(), indent=2, default=str))]


async def _handle_run_sql(args: dict) -> list[TextContent]:
    sql = args["sql"].strip()
    limit = min(args.get("limit", 100), 1000)

    # Safety: only allow SELECT
    if not sql.upper().startswith("SELECT"):
        return [TextContent(type="text", text="Error: only SELECT statements are permitted.")]

    # Inject LIMIT if not present
    if "LIMIT" not in sql.upper():
        sql = f"{sql} LIMIT {limit}"

    import psycopg2
    import psycopg2.extras

    try:
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        output = json.dumps([dict(r) for r in rows], indent=2, default=str)
        return [TextContent(type="text", text=f"{len(rows)} rows returned:\n{output}")]
    except Exception as e:
        return [TextContent(type="text", text=f"SQL error: {e}")]


# ── ENTRYPOINT ───────────────────────────────────────────────────────────────
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
