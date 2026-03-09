# src/cleaning/profiler.py
"""
Data quality profiling and anomaly detection.
Produces a full profile report for any DataFrame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    count: int
    null_count: int
    null_rate: float
    unique_count: int
    unique_rate: float
    sample_values: list[Any]
    min: Any = None
    max: Any = None
    mean: float | None = None
    std: float | None = None
    anomalies: list[str] = field(default_factory=list)


@dataclass
class DataProfile:
    source: str
    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    overall_null_rate: float
    duplicate_rate: float
    schema_hash: str
    anomalies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "overall_null_rate": round(self.overall_null_rate, 4),
            "duplicate_rate": round(self.duplicate_rate, 4),
            "schema_hash": self.schema_hash,
            "anomalies": self.anomalies,
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "null_rate": round(c.null_rate, 4),
                    "unique_rate": round(c.unique_rate, 4),
                    "anomalies": c.anomalies,
                }
                for c in self.columns
            ],
        }

    def quality_score(self) -> float:
        """0.0 to 1.0 quality score. Higher is better."""
        score = 1.0
        score -= self.overall_null_rate * 0.4
        score -= self.duplicate_rate * 0.3
        score -= min(len(self.anomalies) * 0.05, 0.3)
        return max(0.0, round(score, 3))


class DataProfiler:
    """
    Profiles a DataFrame and detects anomalies:
    - Null rate spikes
    - Cardinality anomalies
    - Outliers in numeric columns
    - Schema drift vs a reference profile
    """

    NULL_RATE_THRESHOLD  = 0.20   # warn if >20% null
    UNIQUE_RATE_HIGH     = 0.99   # warn if almost all values unique (possible ID column not deduped)
    UNIQUE_RATE_LOW      = 0.005  # warn if almost no variety (possible constant column)
    OUTLIER_STD_FACTOR   = 4.0    # flag values > 4 stddevs from mean

    def profile(self, df: pd.DataFrame, source: str = "unknown") -> DataProfile:
        col_profiles = [self._profile_column(df[col]) for col in df.columns]

        overall_null_rate = df.isnull().mean().mean()
        data_cols = [c for c in df.columns if not c.startswith("_")]
        dup_count = df.duplicated(subset=data_cols).sum()
        duplicate_rate = dup_count / max(len(df), 1)

        schema_hash = self._schema_hash(df)
        anomalies = self._detect_dataset_anomalies(df, col_profiles)

        return DataProfile(
            source=source,
            row_count=len(df),
            column_count=len(df.columns),
            columns=col_profiles,
            overall_null_rate=overall_null_rate,
            duplicate_rate=duplicate_rate,
            schema_hash=schema_hash,
            anomalies=anomalies,
        )

    def detect_drift(self, current: DataProfile, reference: DataProfile) -> list[str]:
        """Compare current profile vs reference. Returns list of drift alerts."""
        alerts = []

        if current.schema_hash != reference.schema_hash:
            alerts.append(f"SCHEMA DRIFT: hash changed {reference.schema_hash[:8]} → {current.schema_hash[:8]}")

        # Check for new/missing columns
        current_cols  = {c.name for c in current.columns}
        reference_cols = {c.name for c in reference.columns}
        for col in current_cols - reference_cols:
            alerts.append(f"NEW COLUMN: {col}")
        for col in reference_cols - current_cols:
            alerts.append(f"MISSING COLUMN: {col}")

        # Check null rate drift per column
        ref_map = {c.name: c for c in reference.columns}
        for col in current.columns:
            if col.name not in ref_map:
                continue
            ref_col = ref_map[col.name]
            null_delta = col.null_rate - ref_col.null_rate
            if null_delta > 0.10:
                alerts.append(
                    f"NULL RATE SPIKE in {col.name}: "
                    f"{ref_col.null_rate:.2%} → {col.null_rate:.2%} (+{null_delta:.2%})"
                )

        if current.row_count < reference.row_count * 0.5:
            alerts.append(
                f"ROW COUNT DROP: {reference.row_count:,} → {current.row_count:,} "
                f"({(1 - current.row_count/reference.row_count):.0%} decrease)"
            )

        return alerts

    def _profile_column(self, series: pd.Series) -> ColumnProfile:
        n = len(series)
        null_count = series.isnull().sum()
        null_rate = null_count / max(n, 1)
        non_null = series.dropna()
        unique_count = non_null.nunique()
        unique_rate = unique_count / max(len(non_null), 1)

        profile = ColumnProfile(
            name=series.name,
            dtype=str(series.dtype),
            count=n - null_count,
            null_count=null_count,
            null_rate=null_rate,
            unique_count=unique_count,
            unique_rate=unique_rate,
            sample_values=non_null.head(3).tolist(),
        )

        if pd.api.types.is_numeric_dtype(series):
            profile.min  = float(non_null.min()) if len(non_null) else None
            profile.max  = float(non_null.max()) if len(non_null) else None
            profile.mean = float(non_null.mean()) if len(non_null) else None
            profile.std  = float(non_null.std())  if len(non_null) > 1 else None

        profile.anomalies = self._detect_column_anomalies(series, profile)
        return profile

    def _detect_column_anomalies(self, series: pd.Series, profile: ColumnProfile) -> list[str]:
        anomalies = []

        if profile.null_rate > self.NULL_RATE_THRESHOLD:
            anomalies.append(f"High null rate: {profile.null_rate:.1%}")

        if profile.unique_rate > self.UNIQUE_RATE_HIGH and not series.name.endswith("_id"):
            anomalies.append(f"Near-unique values ({profile.unique_rate:.1%}) — possible un-deduped ID")

        if profile.unique_rate < self.UNIQUE_RATE_LOW and profile.count > 100:
            anomalies.append(f"Near-constant column ({profile.unique_count} unique values)")

        if pd.api.types.is_numeric_dtype(series) and profile.std and profile.std > 0:
            non_null = series.dropna()
            z_scores = (non_null - profile.mean) / profile.std
            outlier_count = (z_scores.abs() > self.OUTLIER_STD_FACTOR).sum()
            if outlier_count > 0:
                anomalies.append(f"{outlier_count} statistical outliers (>{self.OUTLIER_STD_FACTOR}σ)")

        return anomalies

    def _detect_dataset_anomalies(self, df: pd.DataFrame, col_profiles: list[ColumnProfile]) -> list[str]:
        anomalies = []
        high_null_cols = [c.name for c in col_profiles if c.null_rate > 0.5]
        if high_null_cols:
            anomalies.append(f"Columns with >50% nulls: {high_null_cols}")
        if df.duplicated().sum() / max(len(df), 1) > 0.05:
            anomalies.append(f"High duplicate rate: {df.duplicated().sum() / len(df):.1%}")
        return anomalies

    def _schema_hash(self, df: pd.DataFrame) -> str:
        import hashlib
        schema_str = "|".join(f"{c}:{t}" for c, t in zip(df.columns, df.dtypes))
        return hashlib.md5(schema_str.encode()).hexdigest()
