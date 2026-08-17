"""
ResearchIQ Compression Package
===============================
Section-aware extractive summarization using TextRank.

NO RAG. NO embedding-based retrieval. NO sentence-transformers.
Every selected sentence is extracted verbatim from the original paper.

Public API
----------
    compress_paper(paper_dict, query, config) -> PaperFactSheet
    format_fact_sheet(fact_sheet) -> str
"""

from .fact_sheet import compress_paper, format_fact_sheet
from .models import (
    PaperFactSheet,
    SectionFactSheet,
    ExtractedSentence,
    PaperSection,
    CompressionConfig,
    CompressionMetrics,
    DEFAULT_CONFIG,
)

__all__ = [
    "compress_paper",
    "format_fact_sheet",
    "PaperFactSheet",
    "SectionFactSheet",
    "ExtractedSentence",
    "PaperSection",
    "CompressionConfig",
    "CompressionMetrics",
    "DEFAULT_CONFIG",
]
