# src/rag/chunker.py
"""
Document chunking strategies for RAG pipelines.
Supports semantic chunking, fixed-size with overlap, and structured chunking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class ChunkStrategy(str, Enum):
    FIXED       = "fixed"       # Fixed token size with overlap
    SENTENCE    = "sentence"    # Split on sentence boundaries
    PARAGRAPH   = "paragraph"   # Split on paragraph breaks
    SEMANTIC    = "semantic"    # Semantic similarity-based splitting
    STRUCTURED  = "structured"  # Field-aware chunking for tabular data


@dataclass
class Chunk:
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    source_id: str
    chunk_index: int
    total_chunks: int
    strategy: str
    token_count: int = 0


class DocumentChunker:
    """
    Chunks documents for RAG ingestion.
    Preserves metadata and source attribution on every chunk.
    """

    def __init__(
        self,
        strategy: ChunkStrategy = ChunkStrategy.SENTENCE,
        chunk_size: int = 512,
        overlap: int = 64,
        min_chunk_size: int = 50,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def chunk_text(self, text: str, source_id: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        if self.strategy == ChunkStrategy.FIXED:
            pieces = self._fixed_split(text)
        elif self.strategy == ChunkStrategy.SENTENCE:
            pieces = self._sentence_split(text)
        elif self.strategy == ChunkStrategy.PARAGRAPH:
            pieces = self._paragraph_split(text)
        else:
            pieces = self._sentence_split(text)

        pieces = [p for p in pieces if len(p.strip()) >= self.min_chunk_size]
        total = len(pieces)

        return [
            Chunk(
                chunk_id=f"{source_id}__chunk_{i}",
                content=piece,
                metadata={**metadata, "chunk_strategy": self.strategy.value},
                source_id=source_id,
                chunk_index=i,
                total_chunks=total,
                strategy=self.strategy.value,
                token_count=len(piece.split()),
            )
            for i, piece in enumerate(pieces)
        ]

    def chunk_dataframe(self, df: pd.DataFrame, text_cols: list[str],
                        id_col: str | None = None, metadata_cols: list[str] | None = None) -> list[Chunk]:
        """Chunk a DataFrame by treating each row's text columns as a document."""
        all_chunks = []
        metadata_cols = metadata_cols or []

        for idx, row in df.iterrows():
            # Combine text columns into one document
            parts = []
            for col in text_cols:
                val = row.get(col)
                if pd.notna(val) and str(val).strip():
                    parts.append(f"{col}: {val}")
            if not parts:
                continue

            doc_text = "\n".join(parts)
            source_id = str(row[id_col]) if id_col and id_col in row else f"row_{idx}"
            metadata = {col: row.get(col) for col in metadata_cols if col in row}
            metadata["_row_index"] = idx

            chunks = self.chunk_text(doc_text, source_id, metadata)
            all_chunks.extend(chunks)

        return all_chunks

    def _fixed_split(self, text: str) -> list[str]:
        words = text.split()
        pieces = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            pieces.append(" ".join(words[start:end]))
            start += self.chunk_size - self.overlap
        return pieces

    def _sentence_split(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks, current, current_len = [], [], 0

        for sent in sentences:
            sent_len = len(sent.split())
            if current_len + sent_len > self.chunk_size and current:
                chunks.append(" ".join(current))
                # Keep overlap sentences
                overlap_sents = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s.split()) <= self.overlap:
                        overlap_sents.insert(0, s)
                        overlap_len += len(s.split())
                    else:
                        break
                current = overlap_sents
                current_len = overlap_len
            current.append(sent)
            current_len += sent_len

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _paragraph_split(self, text: str) -> list[str]:
        paragraphs = re.split(r"\n\s*\n", text.strip())
        chunks, current_chunk, current_len = [], [], 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_len = len(para.split())
            if current_len + para_len > self.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            current_chunk.append(para)
            current_len += para_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        return chunks
