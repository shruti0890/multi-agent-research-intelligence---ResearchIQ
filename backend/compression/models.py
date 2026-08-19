"""
Pydantic models for the ResearchIQ compression pipeline.

All models extend the existing project schema conventions.
No embedding, no RAG, no retrieval — purely extractive.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Target compression parameters (65–75% RETENTION, 25–35% REDUCTION, preferred 70% retention)
# ---------------------------------------------------------------------------

TARGET_RETENTION_MIN: float = 0.65      #: 65% retained (35% reduction)
TARGET_RETENTION_MAX: float = 0.75      #: 75% retained (25% reduction)
TARGET_RETENTION_DEFAULT: float = 0.70  #: 70% retained (30% reduction)

# Legacy aliases for reduction ratio
TARGET_COMPRESSION_MIN: float = 0.25    #: 25% reduction (75% retained)
TARGET_COMPRESSION_MAX: float = 0.35    #: 35% reduction (65% retained)


class SectionBudget(BaseModel):
    """Min and max sentence count for a given section."""
    min_sentences: int = Field(default=1, ge=1)
    max_sentences: int = Field(default=5, ge=1)


class CompressionConfig(BaseModel):
    """
    Configurable sentence budgets and targets per normalized section name.
    Override at runtime for experiment A/B/C configurations.
    """
    # Target retention range (0.65–0.75 = 65–75% RETENTION, 25–35% REDUCTION)
    target_retention_min: float = Field(
        default=0.65,
        description="Minimum target content retention fraction (0.65 = 65% retained, 35% reduction)"
    )
    target_retention_max: float = Field(
        default=0.75,
        description="Maximum target content retention fraction (0.75 = 75% retained, 25% reduction)"
    )
    target_retention_default: float = Field(
        default=0.70,
        description="Preferred default content retention fraction (0.70 = 70% retained, 30% reduction)"
    )
    target_retention_ratio: float = Field(
        default=0.70,
        description="Target content retention fraction (0.70 = 70% retained, 30% reduction)"
    )

    # Legacy reduction ratio aliases
    target_compression_min: float = Field(
        default=0.25,
        description="Minimum target compression reduction ratio (0.25 = 25% reduction, 75% retained)"
    )
    target_compression_max: float = Field(
        default=0.35,
        description="Maximum target compression reduction ratio (0.35 = 35% reduction, 65% retained)"
    )

    # Minimum retention safeguards for large papers (always maintain at least 65% retention)
    min_retention_5k: float = Field(
        default=0.65,
        description="Minimum retained fraction for papers > 5,000 tokens (e.g. 0.65 = 65% retained)"
    )
    min_retention_20k: float = Field(
        default=0.65,
        description="Minimum retained fraction for papers > 20,000 tokens (e.g. 0.65 = 65% retained)"
    )

    # Section priority weights for dynamic budget allocation
    section_priority_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "abstract": 1.0,
            "introduction": 1.0,
            "background": 0.9,
            "related_work": 0.8,
            "problem_definition": 0.9,
            "methodology": 1.3,
            "dataset": 1.1,
            "experiments": 1.2,
            "results": 1.3,
            "discussion": 1.0,
            "limitations": 1.1,
            "future_work": 0.8,
            "conclusion": 0.9,
            "unknown": 0.9,
        }
    )

    # Per-section baseline budgets (minimums and fallback maximums)
    budgets: Dict[str, SectionBudget] = Field(
        default_factory=lambda: {
            "abstract":          SectionBudget(min_sentences=1, max_sentences=50),
            "introduction":      SectionBudget(min_sentences=2, max_sentences=1000),
            "background":        SectionBudget(min_sentences=2, max_sentences=1000),
            "related_work":      SectionBudget(min_sentences=2, max_sentences=1000),
            "problem_definition": SectionBudget(min_sentences=1, max_sentences=1000),
            "methodology":       SectionBudget(min_sentences=3, max_sentences=2000),
            "dataset":           SectionBudget(min_sentences=1, max_sentences=1000),
            "experiments":       SectionBudget(min_sentences=2, max_sentences=2000),
            "results":           SectionBudget(min_sentences=2, max_sentences=2000),
            "discussion":        SectionBudget(min_sentences=2, max_sentences=1000),
            "limitations":       SectionBudget(min_sentences=1, max_sentences=1000),
            "future_work":       SectionBudget(min_sentences=1, max_sentences=1000),
            "conclusion":        SectionBudget(min_sentences=1, max_sentences=1000),
            "unknown":           SectionBudget(min_sentences=2, max_sentences=2000),
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
    # Legacy: total detected sections (includes non-research sections in denominator).
    # Kept for backward compatibility. Use research_section_coverage instead.
    section_coverage: float = Field(
        default=0.0,
        description="Fraction of all detected sections that have at least 1 selected sentence (legacy)"
    )
    detected_sections: int = 0
    covered_sections: int = 0

    # Research-section-aware coverage (correct metric — excludes skipped non-research sections)
    research_sections: int = Field(
        default=0,
        description="Count of sections containing research content (excludes references/acks/supplementary)"
    )
    research_sections_covered: int = Field(
        default=0,
        description="Count of research sections that have at least 1 selected sentence"
    )
    research_section_coverage: float = Field(
        default=0.0,
        description="research_sections_covered / research_sections — the correct coverage metric"
    )
    non_research_sections_skipped: int = Field(
        default=0,
        description="Count of sections intentionally skipped (references, acknowledgments, supplementary)"
    )

    # Retained and Reduction ratios (Target: 65–75% retention, 25–35% reduction, preferred 70% retention)
    retained_ratio: float = Field(
        default=0.0,
        description="compressed_token_count / original_token_count. Target: 0.65 - 0.75 (65% - 75% retained, preferred ~70%)."
    )
    reduction_ratio: float = Field(
        default=0.0,
        description="1.0 - retained_ratio. Target: 0.25 - 0.35 (25% - 35% reduction, preferred ~30%)."
    )

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
    # Compression evaluation status — abstract-only papers cannot be meaningfully evaluated
    compression_status: str = Field(
        default="evaluated",
        description="'evaluated' | 'not_evaluated' | 'abstract_only_not_evaluated' — abstract-only papers produce unreliable compression ratios"
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

    def all_sentences(self) -> List[ExtractedSentence]:
        """Returns all selected sentences across all sections."""
        res: List[ExtractedSentence] = []
        for sec in self.sections.values():
            res.extend(sec.selected_sentences)
        return res

    def to_text(self) -> str:
        """
        Render the fact sheet as a structured plain-text block for inclusion
        in a Gemini prompt. Only verbatim-extracted sentences are included.
        """
        from .fact_sheet import format_fact_sheet
        return format_fact_sheet(self)
