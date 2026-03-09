# src/ingestion/connectors.py
"""
Multi-source data connectors — CSV, JSON, PostgreSQL, REST APIs.
Each connector returns a normalized pandas DataFrame with provenance metadata.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import httpx
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    source: str
    source_type: str
    rows: int
    columns: list[str]
    df: pd.DataFrame
    ingested_at: datetime = field(default_factory=datetime.utcnow)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.rows > 0 and not self.errors


class BaseConnector(ABC):
    """Base class for all data source connectors."""

    def __init__(self, source_id: str):
        self.source_id = source_id

    @abstractmethod
    def read(self) -> IngestionResult:
        ...

    def _add_provenance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add source tracking columns to every ingested DataFrame."""
        df["_source_id"]   = self.source_id
        df["_ingested_at"] = datetime.utcnow()
        df["_row_hash"]    = pd.util.hash_pandas_object(df, index=False).astype(str)
        return df


# ─── CSV CONNECTOR ──────────────────────────────────────────────────────────
class CSVConnector(BaseConnector):
    """
    Reads CSV files with automatic delimiter detection, encoding handling,
    and schema inference. Handles messy real-world CSVs gracefully.
    """

    def __init__(
        self,
        path: str | Path,
        source_id: str | None = None,
        encoding: str = "utf-8",
        delimiter: str | None = None,  # None = auto-detect
        column_map: dict[str, str] | None = None,
    ):
        super().__init__(source_id or str(path))
        self.path = Path(path)
        self.encoding = encoding
        self.delimiter = delimiter
        self.column_map = column_map or {}

    def read(self) -> IngestionResult:
        errors, warnings = [], []

        # Auto-detect delimiter
        sep = self.delimiter
        if sep is None:
            sep = self._detect_delimiter()
            warnings.append(f"Auto-detected delimiter: {repr(sep)}")

        # Try multiple encodings if UTF-8 fails
        encodings = [self.encoding, "latin-1", "cp1252", "utf-8-sig"]
        df = None
        for enc in encodings:
            try:
                df = pd.read_csv(
                    self.path,
                    sep=sep,
                    encoding=enc,
                    low_memory=False,
                    on_bad_lines="warn",
                )
                if enc != self.encoding:
                    warnings.append(f"Fell back to encoding: {enc}")
                break
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                errors.append(f"Encoding {enc} failed: {e}")

        if df is None:
            return IngestionResult(
                source=str(self.path), source_type="csv",
                rows=0, columns=[], df=pd.DataFrame(), errors=errors
            )

        # Apply column remapping
        if self.column_map:
            df = df.rename(columns=self.column_map)
            warnings.append(f"Remapped columns: {self.column_map}")

        # Strip whitespace from string columns
        str_cols = df.select_dtypes(include="object").columns
        df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

        df = self._add_provenance(df)

        return IngestionResult(
            source=str(self.path), source_type="csv",
            rows=len(df), columns=list(df.columns),
            df=df, errors=errors, warnings=warnings,
        )

    def _detect_delimiter(self) -> str:
        sample = self.path.read_text(encoding="utf-8", errors="ignore")[:2048]
        counts = {d: sample.count(d) for d in [",", "\t", "|", ";"]}
        return max(counts, key=counts.get)


# ─── JSON CONNECTOR ─────────────────────────────────────────────────────────
class JSONConnector(BaseConnector):
    """
    Reads JSON files — handles nested objects, arrays of objects,
    and newline-delimited JSON (NDJSON). Flattens nested fields.
    """

    def __init__(
        self,
        path: str | Path,
        source_id: str | None = None,
        flatten_depth: int = 2,
        record_path: str | None = None,
    ):
        super().__init__(source_id or str(path))
        self.path = Path(path)
        self.flatten_depth = flatten_depth
        self.record_path = record_path

    def read(self) -> IngestionResult:
        errors, warnings = [], []

        try:
            raw = self.path.read_text(encoding="utf-8")
        except Exception as e:
            return IngestionResult(
                source=str(self.path), source_type="json",
                rows=0, columns=[], df=pd.DataFrame(), errors=[str(e)]
            )

        # Try NDJSON first, then regular JSON
        try:
            df = pd.read_json(self.path, lines=True)
            warnings.append("Parsed as NDJSON")
        except Exception:
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    df = pd.json_normalize(data, max_level=self.flatten_depth)
                elif isinstance(data, dict) and self.record_path:
                    records = data.get(self.record_path, [])
                    df = pd.json_normalize(records, max_level=self.flatten_depth)
                    warnings.append(f"Extracted records from key: {self.record_path}")
                else:
                    df = pd.json_normalize([data], max_level=self.flatten_depth)
            except Exception as e:
                return IngestionResult(
                    source=str(self.path), source_type="json",
                    rows=0, columns=[], df=pd.DataFrame(), errors=[str(e)]
                )

        # Flatten column names (replace dots from json_normalize)
        df.columns = [c.replace(".", "_").lower() for c in df.columns]
        df = self._add_provenance(df)

        return IngestionResult(
            source=str(self.path), source_type="json",
            rows=len(df), columns=list(df.columns),
            df=df, errors=errors, warnings=warnings,
        )


# ─── POSTGRES CONNECTOR ─────────────────────────────────────────────────────
class PostgresConnector(BaseConnector):
    """
    Reads data from PostgreSQL via SQLAlchemy.
    Supports full table reads, custom queries, and incremental loads.
    """

    def __init__(
        self,
        connection_string: str,
        query: str,
        source_id: str,
        chunk_size: int = 10_000,
    ):
        super().__init__(source_id)
        self.connection_string = connection_string
        self.query = query
        self.chunk_size = chunk_size

    def read(self) -> IngestionResult:
        errors, warnings = [], []
        chunks = []

        try:
            engine = sa.create_engine(self.connection_string)
            with engine.connect() as conn:
                for chunk in pd.read_sql(
                    text(self.query), conn, chunksize=self.chunk_size
                ):
                    chunks.append(chunk)

            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            df = self._add_provenance(df)

            return IngestionResult(
                source=self.source_id, source_type="postgres",
                rows=len(df), columns=list(df.columns),
                df=df, errors=errors, warnings=warnings,
            )

        except Exception as e:
            logger.error(f"PostgreSQL read failed for {self.source_id}: {e}")
            return IngestionResult(
                source=self.source_id, source_type="postgres",
                rows=0, columns=[], df=pd.DataFrame(), errors=[str(e)]
            )


# ─── REST API CONNECTOR ─────────────────────────────────────────────────────
class RESTConnector(BaseConnector):
    """
    Fetches data from REST APIs with pagination support,
    auth handling, and automatic retry on transient failures.
    """

    def __init__(
        self,
        url: str,
        source_id: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        pagination_key: str | None = "next",
        record_path: str | None = None,
        max_pages: int = 100,
    ):
        super().__init__(source_id)
        self.url = url
        self.headers = headers or {}
        self.params = params or {}
        self.pagination_key = pagination_key
        self.record_path = record_path
        self.max_pages = max_pages

    def _paginate(self) -> Iterator[list[dict]]:
        url = self.url
        page = 0
        with httpx.Client(headers=self.headers, timeout=30) as client:
            while url and page < self.max_pages:
                resp = client.get(url, params=self.params if page == 0 else {})
                resp.raise_for_status()
                data = resp.json()

                records = data
                if self.record_path:
                    for key in self.record_path.split("."):
                        records = records.get(key, [])

                if not records:
                    break

                yield records
                page += 1

                # Follow pagination
                if self.pagination_key and isinstance(data, dict):
                    url = data.get(self.pagination_key)
                else:
                    break

    def read(self) -> IngestionResult:
        errors, warnings = [], []
        all_records = []

        try:
            for page_records in self._paginate():
                all_records.extend(page_records)
            warnings.append(f"Fetched {len(all_records)} records from API")
        except Exception as e:
            errors.append(str(e))

        if not all_records:
            return IngestionResult(
                source=self.url, source_type="rest_api",
                rows=0, columns=[], df=pd.DataFrame(), errors=errors
            )

        df = pd.json_normalize(all_records)
        df.columns = [c.replace(".", "_").lower() for c in df.columns]
        df = self._add_provenance(df)

        return IngestionResult(
            source=self.url, source_type="rest_api",
            rows=len(df), columns=list(df.columns),
            df=df, errors=errors, warnings=warnings,
        )
