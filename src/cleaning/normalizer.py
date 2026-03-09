# src/cleaning/normalizer.py
"""
Schema remapping, type casting, deduplication, and normalization.
Handles the messiest real-world data scenarios.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NormalizationResult:
    df: pd.DataFrame
    original_rows: int
    final_rows: int
    dropped_rows: int
    remapped_columns: dict[str, str]
    type_coercions: dict[str, str]
    issues: list[str] = field(default_factory=list)

    @property
    def drop_rate(self) -> float:
        return self.dropped_rows / max(self.original_rows, 1)


class SchemaNormalizer:
    """
    Normalizes messy DataFrames:
    - Remaps inconsistent column names to a canonical schema
    - Coerces types with fallback strategies
    - Deduplicates records
    - Standardizes text, dates, phone numbers, emails
    """

    # Common column name aliases → canonical name
    DEFAULT_COLUMN_ALIASES: dict[str, list[str]] = {
        "id":           ["record_id", "uid", "uuid", "key", "pk", "row_id"],
        "name":         ["full_name", "fullname", "display_name", "person_name", "username"],
        "email":        ["email_address", "e_mail", "mail", "contact_email", "user_email"],
        "phone":        ["phone_number", "mobile", "cell", "telephone", "tel", "contact_phone"],
        "address":      ["street_address", "addr", "street", "location", "address_line1"],
        "city":         ["town", "city_name", "municipality"],
        "state":        ["province", "region", "state_code"],
        "zip":          ["zip_code", "postal_code", "postcode"],
        "country":      ["country_code", "country_name", "nation"],
        "created_at":   ["created", "create_date", "creation_date", "date_created", "insert_date"],
        "updated_at":   ["updated", "update_date", "modification_date", "date_modified", "last_modified"],
        "status":       ["state", "record_status", "active", "is_active"],
        "amount":       ["value", "price", "cost", "total", "sum", "revenue"],
        "description":  ["desc", "details", "notes", "comment", "text", "body", "content"],
        "category":     ["type", "kind", "group", "segment", "class"],
        "user_id":      ["userid", "user", "owner_id", "customer_id", "member_id"],
    }

    def __init__(
        self,
        column_map: dict[str, str] | None = None,
        type_map: dict[str, str] | None = None,
        dedup_keys: list[str] | None = None,
        drop_null_threshold: float = 0.95,
        fuzzy_column_matching: bool = True,
    ):
        self.column_map = column_map or {}
        self.type_map = type_map or {}
        self.dedup_keys = dedup_keys
        self.drop_null_threshold = drop_null_threshold
        self.fuzzy_column_matching = fuzzy_column_matching

    def normalize(self, df: pd.DataFrame) -> NormalizationResult:
        original_rows = len(df)
        issues = []

        # Step 1: Normalize column names
        df, remapped = self._normalize_columns(df)

        # Step 2: Drop columns that are almost entirely null
        df, dropped_cols = self._drop_empty_columns(df)
        if dropped_cols:
            issues.append(f"Dropped {len(dropped_cols)} near-empty columns: {dropped_cols}")

        # Step 3: Type coercion
        df, coercions = self._coerce_types(df)

        # Step 4: Standardize common fields
        df = self._standardize_text(df)
        df = self._standardize_dates(df)
        df = self._standardize_phones(df)
        df = self._standardize_emails(df)

        # Step 5: Deduplication
        df = self._deduplicate(df, issues)

        final_rows = len(df)

        return NormalizationResult(
            df=df,
            original_rows=original_rows,
            final_rows=final_rows,
            dropped_rows=original_rows - final_rows,
            remapped_columns=remapped,
            type_coercions=coercions,
            issues=issues,
        )

    def _normalize_columns(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
        # Normalize column name format first
        df.columns = [
            re.sub(r"[^a-z0-9_]", "_", c.lower().strip()).strip("_")
            for c in df.columns
        ]

        # Apply explicit column map
        remapped = {}
        rename_map = {}

        for old, new in self.column_map.items():
            if old in df.columns:
                rename_map[old] = new
                remapped[old] = new

        # Fuzzy alias matching for unmapped columns
        if self.fuzzy_column_matching:
            for col in df.columns:
                if col in rename_map:
                    continue
                canonical = self._find_alias(col)
                if canonical and canonical != col:
                    rename_map[col] = canonical
                    remapped[col] = canonical

        df = df.rename(columns=rename_map)
        return df, remapped

    def _find_alias(self, col: str) -> str | None:
        for canonical, aliases in self.DEFAULT_COLUMN_ALIASES.items():
            if col in aliases:
                return canonical
        return None

    def _drop_empty_columns(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        null_rates = df.isnull().mean()
        drop = null_rates[null_rates >= self.drop_null_threshold].index.tolist()
        return df.drop(columns=drop), drop

    def _coerce_types(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
        coercions = {}

        # Apply explicit type map
        for col, dtype in self.type_map.items():
            if col not in df.columns:
                continue
            try:
                if dtype in ("int", "int64"):
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                elif dtype in ("float", "float64"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif dtype == "datetime":
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                elif dtype == "bool":
                    df[col] = df[col].map({"true": True, "false": False, "1": True, "0": False, 1: True, 0: False})
                coercions[col] = dtype
            except Exception as e:
                logger.warning(f"Type coercion failed for {col} → {dtype}: {e}")

        # Auto-detect numeric strings
        for col in df.select_dtypes(include="object").columns:
            sample = df[col].dropna().head(100)
            numeric_ratio = sample.apply(lambda x: _is_numeric(str(x))).mean()
            if numeric_ratio > 0.9:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                coercions[col] = "float (auto-detected)"

        return df, coercions

    def _standardize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].str.strip()
            # Replace empty strings with NaN
            df[col] = df[col].replace({"": np.nan, "N/A": np.nan, "NULL": np.nan, "null": np.nan, "None": np.nan})
        return df

    def _standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        date_pattern = re.compile(r"date|timestamp|created|updated|modified|at$", re.IGNORECASE)
        for col in df.columns:
            if date_pattern.search(col) and df[col].dtype == "object":
                converted = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                if converted.notna().sum() / max(df[col].notna().sum(), 1) > 0.7:
                    df[col] = converted
        return df

    def _standardize_phones(self, df: pd.DataFrame) -> pd.DataFrame:
        phone_cols = [c for c in df.columns if "phone" in c or "mobile" in c or "tel" in c]
        for col in phone_cols:
            if df[col].dtype == "object":
                df[col] = df[col].str.replace(r"[^\d+]", "", regex=True)
        return df

    def _standardize_emails(self, df: pd.DataFrame) -> pd.DataFrame:
        email_cols = [c for c in df.columns if "email" in c or "mail" in c]
        for col in email_cols:
            if df[col].dtype == "object":
                df[col] = df[col].str.lower().str.strip()
        return df

    def _deduplicate(self, df: pd.DataFrame, issues: list[str]) -> pd.DataFrame:
        before = len(df)
        if self.dedup_keys:
            available_keys = [k for k in self.dedup_keys if k in df.columns]
            if available_keys:
                df = df.drop_duplicates(subset=available_keys, keep="last")
        else:
            # Hash-based dedup on all non-provenance columns
            data_cols = [c for c in df.columns if not c.startswith("_")]
            df = df.drop_duplicates(subset=data_cols)

        dupes = before - len(df)
        if dupes > 0:
            issues.append(f"Removed {dupes} duplicate rows")
        return df.reset_index(drop=True)


def _is_numeric(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False
