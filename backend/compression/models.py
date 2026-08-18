"""
Pydantic models for the ResearchIQ compression pipeline.

All models extend the existing project schema conventions.
No embedding, no RAG, no retrieval — purely extractive.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configurable sentence budgets per section
# These are the DEFAULTS — they can be overridden at runtime.
# ---------------------------------------------------------------------------

class SectionBudget(BaseModel):
    """Min and max sentence count for a given section."""
    min_sentences: int = Field(default=1, ge=1)
    max_sentences: int = Field(default=5, ge=1)


class CompressionConfig(BaseModel):
    """
    Configurable sentence budgets per normalized section name.
    Override at runtime for experiment A/B/C configurations.
    """
    # Per-section budgets
    budgets: Dict[str, SectionBudget] = Field(
        default_factory=lambda: {
            "abstract":          SectionBudget(min_sentences=1, max_sentences=3),
            "introduction":      SectionBudget(min_sentences=2, max_sentences=5),
            "related_work":      SectionBudget(min_sentences=2, max_sentences=5),
            "problem_definition": SectionBudget(min_sentences=1, max_sentences=4),
            "methodology":       SectionBudget(min_sentences=3, max_sentences=8),
            "dataset":           SectionBudget(min_sentences=1, max_sentences=4),
            "experiments":       SectionBudget(min_sentences=2, max_sentences=6),
            "results":           SectionBudget(min_sentences=2, max_sentences=8),
            "discussion":        SectionBudget(min_sentences=2, max_sentences=5),
            "limitations":       SectionBudget(min_sentences=1, max_sentences=5),
            "future_work":       SectionBudget(min_sentences=1, max_sentences=4),
            "conclusion":        SectionBudget(min_sentences=1, max_sentences=4),
            "unknown":           SectionBudget(min_sentences=2, max_sentences=6),
        }
    )

    # Diversity: suppress a sentence if its Jaccard overlap with an already-selected
    # sentence in the same section exceeds this threshold.
    diversity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # Technical weight: boost for sentences matching technical patterns.
    technical_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    # Memory safety: warn (do not truncate) if section text exceeds this many characters.
    section_size_warn_chars: int = Field(default=100_000)

    # Global safety: if the full paper text exceeds this, process section-by-section
    # and log a warning. Set to 0 to disable.
    paper_size_warn_chars: int = Field(default=500_000)


# Singleton default config — imported by the orchestration layer
DEFAULT_CONFIG = CompressionConfig()


# ---------------------------------------------------------------------------
# Fine-grained sentence model with full provenance
# ---------------------------------------------------------------------------

class ExtractedSentence(BaseModel):
    """
    A single sentence selected from a paper, with complete provenance.
    Every field must be populated; no sentence can be added without traceability.
    """
    sentence_id: str = Field(
        description="Unique ID: {paper_id}-{SECTION_ABBREV}-{zero_padded_index}. E.g. P001-METH-04"
    )
    paper_id: str = Field(description="Identifier of the source paper (e.g. P001)")
    section: str = Field(description="Normalized section key (e.g. 'methodology')")
    text: str = Field(description="Verbatim sentence text extracted from the original paper")
    original_index: int = Field(description="0-based position of this sentence within its section")
    textrank_score: float = Field(description="Raw TextRank centrality score for this sentence")
    technical_score: float = Field(description="Score from technical-information pattern matching (0.0–1.0)")
    selected: bool = Field(default=True)
    selection_reason: str = Field(
        description="Why this sentence was selected: 'textrank', 'technical', 'textrank+technical', 'min_coverage'"
    )
    # Source validation
    source_verified: bool = Field(
        default=False,
        description="True if the sentence was found verbatim in the original paper text"
    )


# ---------------------------------------------------------------------------
# Section-level models
# ---------------------------------------------------------------------------

class PaperSection(BaseModel):
    """
    Stores the original text and extracted sentences for one paper section.
    The original_text is always preserved for debugging/validation.
    """
    section_name: str
    original_text: str = Field(description="Complete original section text before any compression")
    sentences: List[ExtractedSentence] = Field(default_factory=list)


class SectionFactSheet(BaseModel):
    """Compressed representation of a single section, ready for Gemini consumption."""
    section_name: str
    original_word_count: int
    selected_sentence_count: int
    technical_sentence_count: int
    selected_sentences: List[ExtractedSentence] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Paper-level models
# ---------------------------------------------------------------------------

class CompressionMetrics(BaseModel):
    """Quantitative compression measurements. Measured, never assumed."""
    original_word_count: int = 0
    compressed_word_count: int = 0
    original_token_count: int = 0
    compressed_token_count: int = 0
    compression_ratio: float = Field(
        default=0.0,
        description="1 - (compressed_tokens / original_tokens). Measured value."
    )
    section_coverage: float = Field(
        default=0.0,
        description="Fraction of detected sections that have at least 1 selected sentence"
    )
    detected_sections: int = 0
    covered_sections: int = 0
    selected_sentence_count: int = 0
    technical_sentence_count: int = 0
    total_validated_sentences: int = 0
    faithfulness_ratio: float = Field(
        default=1.0,
        description="Fraction of selected sentences verified verbatim in original text"
    )
    token_count_method: str = Field(
        default="estimated_word_split",
        description="Token counting method: 'estimated_word_split' = heuristic (words × 1.15). NOT an exact Gemini token count."
    )


class PaperFactSheet(BaseModel):
    """
    Complete compressed representation of one research paper.
    Sections contain only verbatim-extracted sentences.
    """
    paper_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    year: int = 2024
    doi: str = ""
    source_url: str = ""

    # Coverage type — set by compress_paper() based on available input
    coverage_type: str = Field(
        default="unavailable",
        description="full_paper | partial_paper | abstract_only | unavailable"
    )

    # Section data
    sections: Dict[str, SectionFactSheet] = Field(default_factory=dict)

    # Metrics
    metrics: CompressionMetrics = Field(default_factory=CompressionMetrics)

    def to_text(self) -> str:
        """
        Render the fact sheet as a structured plain-text block for inclusion
        in a Gemini prompt. Only verbatim-extracted sentences are included.
        """
        from .fact_sheet import format_fact_sheet
        return format_fact_sheet(self)
