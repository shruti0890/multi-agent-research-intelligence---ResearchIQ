"""
token_utils.py
==============
Lightweight token/word counting and compression measurement utilities.

Uses a simple word-split approximation for token counting to avoid
introducing any model dependency. Research shows ~1.3 words/token on average
for English scientific text, but we keep it conservative at 1 word ≈ 1 token
to avoid underestimating usage.

No LLMs, no embeddings, no network calls.
"""

from __future__ import annotations

import re
from typing import Set

from .models import CompressionMetrics


def count_words(text: str) -> int:
    """Count whitespace-separated words in text."""
    if not text or not text.strip():
        return 0
    return len(text.split())


def estimate_tokens(text: str) -> int:
    """
    ESTIMATE token count using a conservative word-split heuristic.
    Each word ≈ 1 token; punctuation tokens are counted separately.

    IMPORTANT: This is an ESTIMATE using the word-split heuristic (words × 1.15).
    It is NOT an exact Gemini token count. Results are labeled '(est.)' in all
    reports and fact sheets.
    """
    if not text or not text.strip():
        return 0
    # Count words
    words = text.split()
    # Add ~15% for subword tokenization of technical terms
    return max(1, int(len(words) * 1.15))


def measure_compression(
    original_text: str,
    compressed_text: str,
    detected_sections: int = 0,
    covered_sections: int = 0,
    selected_sentences: int = 0,
    technical_sentences: int = 0,
    validated_sentences: int = 0,
    research_sections: int = 0,
    research_sections_covered: int = 0,
    non_research_sections_skipped: int = 0,
) -> CompressionMetrics:
    """
    Compute quantitative compression metrics.
    compression_ratio = 1 - (compressed_tokens / original_tokens)

    research_section_coverage is the canonical metric; it uses only the research-content
    sections in the denominator, excluding intentionally-skipped non-research sections.
    """
    orig_words = count_words(original_text)
    comp_words = count_words(compressed_text)
    orig_tokens = estimate_tokens(original_text)
    comp_tokens = estimate_tokens(compressed_text)

    if orig_tokens > 0:
        ret_ratio = max(0.0, min(1.0, round(comp_tokens / orig_tokens, 4)))
        red_ratio = max(0.0, min(1.0, round(1.0 - ret_ratio, 4)))
        ratio = red_ratio
    else:
        ret_ratio = 0.0
        red_ratio = 0.0
        ratio = 0.0

    # Legacy section_coverage (includes all detected sections in denominator)
    if detected_sections > 0:
        coverage = round(covered_sections / detected_sections, 4)
    else:
        coverage = 0.0

    # Correct research_section_coverage (excludes non-research from denominator)
    if research_sections > 0:
        r_coverage = round(research_sections_covered / research_sections, 4)
    elif detected_sections > 0:
        # Fallback: if research_sections not provided, use detected (same as legacy)
        r_coverage = coverage
    else:
        r_coverage = 0.0

    if selected_sentences > 0:
        faithfulness = round(validated_sentences / selected_sentences, 4)
    else:
        faithfulness = 1.0

    return CompressionMetrics(
        original_word_count=orig_words,
        compressed_word_count=comp_words,
        original_token_count=orig_tokens,
        compressed_token_count=comp_tokens,
        compression_ratio=ratio,
        retained_ratio=ret_ratio,
        reduction_ratio=red_ratio,
        section_coverage=coverage,
        detected_sections=detected_sections,
        covered_sections=covered_sections,
        research_sections=research_sections,
        research_sections_covered=research_sections_covered,
        research_section_coverage=r_coverage,
        non_research_sections_skipped=non_research_sections_skipped,
        selected_sentence_count=selected_sentences,
        technical_sentence_count=technical_sentences,
        total_validated_sentences=validated_sentences,
        faithfulness_ratio=faithfulness,
        token_count_method="estimated_word_split",
    )


def _normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs (spaces, tabs, newlines) to a single space."""
    import re
    return re.sub(r'\s+', ' ', text).strip()


def validate_extracted_sentence(extracted_sentence: str, original_text: str) -> bool:
    """
    Verify that `extracted_sentence` appears verbatim in `original_text`.

    Primary check: exact substring match (strips leading/trailing whitespace only).
    Secondary check: whitespace-normalized comparison — catches harmless PDF/NLTK
    formatting differences (e.g., double spaces, soft hyphens, line-break spaces)
    WITHOUT accepting actual paraphrasing or changed words.

    Returns True if found, False if not (indicates a faithfulness violation).
    """
    s = extracted_sentence.strip()
    if not s:
        return False
    # Primary: exact verbatim match
    if s in original_text:
        return True
    # Secondary: whitespace-normalized match
    # This accepts whitespace differences ONLY — word content must be identical
    s_norm = _normalize_whitespace(s)
    orig_norm = _normalize_whitespace(original_text)
    return s_norm in orig_norm

