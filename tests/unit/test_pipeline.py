# tests/unit/test_normalizer.py
"""Unit tests for SchemaNormalizer — messy data cleaning logic."""

import pytest
import pandas as pd
import numpy as np
from src.cleaning.normalizer import SchemaNormalizer


@pytest.fixture
def messy_df():
    return pd.DataFrame({
        "Record ID":     ["1", "2", "3", "3", None],
        "Full Name ":    ["Alice Smith", " Bob Jones", "Charlie", "Charlie", "Dana"],
        "E_Mail":        ["ALICE@EXAMPLE.COM", "bob@test.com", "NULL", "NULL", "dana@co.com"],
        "phone_number":  ["(555) 123-4567", "555.987.6543", "N/A", "N/A", "+1-800-555-0000"],
        "Amount":        ["1,250.50", "875", "N/A", "N/A", "3000"],
        "created":       ["2024-01-15", "2024-02-20", "2024-03-01", "2024-03-01", "2024-03-10"],
        "EmptyCol":      [None, None, None, None, None],
    })


class TestSchemaNormalizer:

    def test_column_name_normalization(self, messy_df):
        norm = SchemaNormalizer()
        result = norm.normalize(messy_df)
        cols = list(result.df.columns)
        # Should be lowercase, no spaces, no trailing underscores
        for col in cols:
            assert col == col.lower(), f"Column {col} not lowercased"
            assert " " not in col, f"Column {col} has spaces"

    def test_alias_remapping_email(self, messy_df):
        norm = SchemaNormalizer(fuzzy_column_matching=True)
        result = norm.normalize(messy_df)
        assert "email" in result.df.columns, "e_mail should be remapped to email"

    def test_alias_remapping_created_at(self, messy_df):
        norm = SchemaNormalizer(fuzzy_column_matching=True)
        result = norm.normalize(messy_df)
        assert "created_at" in result.df.columns, "created should be remapped to created_at"

    def test_drops_empty_columns(self, messy_df):
        norm = SchemaNormalizer(drop_null_threshold=0.95)
        result = norm.normalize(messy_df)
        assert "emptycol" not in result.df.columns

    def test_deduplication(self, messy_df):
        norm = SchemaNormalizer()
        result = norm.normalize(messy_df)
        # Row 2 and 3 are duplicates — should be removed
        assert result.dropped_rows >= 1
        assert result.final_rows < result.original_rows

    def test_null_standardization(self, messy_df):
        norm = SchemaNormalizer()
        result = norm.normalize(messy_df)
        # "NULL" and "N/A" strings should become NaN
        if "email" in result.df.columns:
            assert result.df["email"].isna().sum() > 0

    def test_email_lowercased(self, messy_df):
        norm = SchemaNormalizer()
        result = norm.normalize(messy_df)
        email_col = "email" if "email" in result.df.columns else "e_mail"
        if email_col in result.df.columns:
            valid_emails = result.df[email_col].dropna()
            assert all(e == e.lower() for e in valid_emails), "Emails should be lowercased"

    def test_phone_digits_only(self, messy_df):
        norm = SchemaNormalizer()
        result = norm.normalize(messy_df)
        phone_col = "phone" if "phone" in result.df.columns else "phone_number"
        if phone_col in result.df.columns:
            valid_phones = result.df[phone_col].dropna()
            for phone in valid_phones:
                assert all(c.isdigit() or c == "+" for c in str(phone)), \
                    f"Phone {phone} contains non-digit characters"

    def test_explicit_column_map(self):
        df = pd.DataFrame({"cust_id": [1, 2], "cust_name": ["Alice", "Bob"]})
        norm = SchemaNormalizer(column_map={"cust_id": "id", "cust_name": "name"})
        result = norm.normalize(df)
        assert "id" in result.df.columns
        assert "name" in result.df.columns
        assert result.remapped_columns == {"cust_id": "id", "cust_name": "name"}

    def test_normalization_result_metadata(self, messy_df):
        norm = SchemaNormalizer()
        result = norm.normalize(messy_df)
        assert result.original_rows == 5
        assert result.final_rows <= result.original_rows
        assert isinstance(result.remapped_columns, dict)
        assert isinstance(result.issues, list)


class TestDataProfiler:

    def test_profile_null_rates(self):
        from src.cleaning.profiler import DataProfiler
        df = pd.DataFrame({
            "a": [1, 2, None, None, None],
            "b": [1, 2, 3, 4, 5],
        })
        profiler = DataProfiler()
        profile = profiler.profile(df, source="test")
        col_a = next(c for c in profile.columns if c.name == "a")
        assert col_a.null_rate == pytest.approx(0.6)

    def test_quality_score_clean_data(self):
        from src.cleaning.profiler import DataProfiler
        df = pd.DataFrame({"a": range(100), "b": [f"val_{i}" for i in range(100)]})
        profiler = DataProfiler()
        profile = profiler.profile(df)
        assert profile.quality_score() > 0.8

    def test_quality_score_dirty_data(self):
        from src.cleaning.profiler import DataProfiler
        df = pd.DataFrame({"a": [None] * 90 + [1] * 10, "b": [None] * 80 + ["x"] * 20})
        profiler = DataProfiler()
        profile = profiler.profile(df)
        assert profile.quality_score() < 0.6

    def test_detects_outliers(self):
        from src.cleaning.profiler import DataProfiler
        values = [10.0] * 98 + [10000.0, -10000.0]  # 2 obvious outliers
        df = pd.DataFrame({"value": values})
        profiler = DataProfiler()
        profile = profiler.profile(df)
        col = profile.columns[0]
        assert any("outlier" in a.lower() for a in col.anomalies)


class TestChunker:

    def test_sentence_chunking_basic(self):
        from src.rag.chunker import DocumentChunker, ChunkStrategy
        chunker = DocumentChunker(strategy=ChunkStrategy.SENTENCE, chunk_size=20)
        text = "This is sentence one. This is sentence two. This is sentence three. And four."
        chunks = chunker.chunk_text(text, source_id="test_doc")
        assert len(chunks) >= 1
        assert all(c.source_id == "test_doc" for c in chunks)
        assert all(c.chunk_id.startswith("test_doc__chunk_") for c in chunks)

    def test_chunk_metadata_preserved(self):
        from src.rag.chunker import DocumentChunker, ChunkStrategy
        chunker = DocumentChunker(strategy=ChunkStrategy.FIXED, chunk_size=10)
        metadata = {"category": "finance", "year": 2024}
        chunks = chunker.chunk_text("word " * 50, source_id="src", metadata=metadata)
        for chunk in chunks:
            assert chunk.metadata["category"] == "finance"
            assert chunk.metadata["year"] == 2024

    def test_dataframe_chunking(self):
        from src.rag.chunker import DocumentChunker, ChunkStrategy
        import pandas as pd
        df = pd.DataFrame({
            "id":          ["doc1", "doc2", "doc3"],
            "title":       ["Alpha report", "Beta study", "Gamma analysis"],
            "description": ["Detailed alpha content here.", "Beta content details.", "Gamma findings."],
        })
        chunker = DocumentChunker(strategy=ChunkStrategy.SENTENCE)
        chunks = chunker.chunk_dataframe(df, text_cols=["title", "description"], id_col="id")
        assert len(chunks) >= 3
        source_ids = {c.source_id for c in chunks}
        assert "doc1" in source_ids
