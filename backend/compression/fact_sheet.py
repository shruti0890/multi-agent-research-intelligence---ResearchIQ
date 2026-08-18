"""
fact_sheet.py
=============
Orchestrates the full compression pipeline and produces PaperFactSheet objects.

Pipeline per paper:
    raw_text
    → section_parser        (detect & normalize sections)
    → segment_sentences     (NLTK sentence tokenizer)
    → textrank              (score sentences per section)
    → technical_preservation (score + force-include technical sentences)
    → diversity filter      (suppress near-duplicate sentences via Jaccard)
    → sentence selection    (TextRank + technical + min-coverage)
    → provenance attachment (sentence_id, scores, reason)
    → source validation     (verify verbatim membership in original)
    → CompressionMetrics    (measure actual token savings)
    → PaperFactSheet

No RAG. No embeddings. No LLM calls.
Every sentence in the output is verbatim from the source paper.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .models import (
    CompressionConfig,
    DEFAULT_CONFIG,
    ExtractedSentence,
    PaperFactSheet,
    SectionFactSheet,
)
from .section_parser import parse_paper_sections
from .technical_preservation import (
    filter_technical_sentences,
    score_technical_importance,
)
from .textrank import run_textrank_for_section, segment_sentences
from .token_utils import measure_compression, validate_extracted_sentence

# ---------------------------------------------------------------------------
# Section abbreviation map for sentence IDs
# ---------------------------------------------------------------------------
_SECTION_ABBREV: Dict[str, str] = {
    "abstract": "ABS",
    "introduction": "INTRO",
    "related_work": "RW",
    "problem_definition": "PROB",
    "methodology": "METH",
    "dataset": "DATA",
    "experiments": "EXP",
    "results": "RES",
    "discussion": "DISC",
    "limitations": "LIM",
    "future_work": "FW",
    "conclusion": "CONC",
    "unknown": "UNK",
}


def _section_abbrev(section_name: str) -> str:
    return _SECTION_ABBREV.get(section_name, section_name[:4].upper())


# ---------------------------------------------------------------------------
# Jaccard similarity for diversity filtering
# ---------------------------------------------------------------------------

def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity between two sentences based on word sets."""
    words_a = set(re.findall(r'\b\w+\b', a.lower()))
    words_b = set(re.findall(r'\b\w+\b', b.lower()))
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _apply_diversity_filter(
    candidates: List[Tuple[int, str, float, float]],
    threshold: float,
) -> List[Tuple[int, str, float, float]]:
    """
    Suppress near-duplicate candidates by Jaccard similarity.
    Prefers the candidate with the higher combined score.
    Returns the filtered list in original order.

    candidates: [(original_index, text, textrank_score, technical_score), ...]
    """
    if not candidates or threshold >= 1.0:
        return candidates

    accepted: List[Tuple[int, str, float, float]] = []
    for cand in candidates:
        idx, text, tr, tech = cand
        combined = tr + tech
        is_duplicate = False
        for _, acc_text, acc_tr, acc_tech in accepted:
            if _jaccard(text, acc_text) >= threshold:
                is_duplicate = True
                # If current candidate is stronger, replace
                if combined > (acc_tr + acc_tech):
                    accepted = [(i, t, r, tc) for i, t, r, tc in accepted if t != acc_text]
                    accepted.append(cand)
                break
        if not is_duplicate:
            accepted.append(cand)

    # Restore original index order
    accepted.sort(key=lambda x: x[0])
    return accepted


# ---------------------------------------------------------------------------
# Per-section compression
# ---------------------------------------------------------------------------

def _compress_section(
    paper_id: str,
    section_name: str,
    section_text: str,
    config: CompressionConfig,
) -> SectionFactSheet:
    """
    Compress a single section using TextRank + technical preservation + diversity.
    Returns a SectionFactSheet with fully provenance-annotated sentences.
    """
    budget = config.budgets.get(section_name, config.budgets.get("unknown"))
    max_s = budget.max_sentences
    min_s = budget.min_sentences

    abbrev = _section_abbrev(section_name)
    original_word_count = len(section_text.split())

    # --- Warn on unusually large sections ---
    if len(section_text) > config.section_size_warn_chars:
        print(
            f"[COMPRESSION] WARNING: Section '{section_name}' in {paper_id} "
            f"is {len(section_text):,} chars. Processing normally (no truncation)."
        )

    # --- TextRank: get (original_index, text, tr_score) ---
    textrank_candidates = run_textrank_for_section(
        section_text, max_sentences=max_s, min_sentences=min_s
    )

    if not textrank_candidates:
        # Section has no parseable sentences — return empty fact sheet
        return SectionFactSheet(
            section_name=section_name,
            original_word_count=original_word_count,
            selected_sentence_count=0,
            technical_sentence_count=0,
            selected_sentences=[],
        )

    # --- Annotate all sentences with technical scores ---
    annotated = filter_technical_sentences(textrank_candidates, threshold=0.0)
    # annotated: [(idx, text, tr_score, tech_score)]

    # --- Force-add technical sentences not already selected ---
    # Get all sentences in section for force-include check
    all_sentences = segment_sentences(section_text)
    all_annotated = []
    selected_indices = {idx for idx, _, _, _ in annotated}

    for i, sent in enumerate(all_sentences):
        if i not in selected_indices:
            tech_score = score_technical_importance(sent)
            if tech_score >= 0.25:
                # Force-include important technical sentence not in TextRank top-k
                # Only if we haven't already hit max budget
                if len(annotated) < max_s:
                    all_annotated.append((i, sent, 0.0, tech_score))

    # Merge TextRank picks with forced technical sentences
    combined = list(annotated) + all_annotated
    # Sort by combined score descending for diversity filtering
    combined.sort(key=lambda x: x[2] + x[3] * config.technical_weight, reverse=True)

    # --- Diversity filter ---
    filtered = _apply_diversity_filter(combined, threshold=config.diversity_threshold)

    # --- Enforce budget constraints ---
    # Keep best up to max_s, ensuring at least min_s
    if len(filtered) > max_s:
        # Sort by combined score to keep best ones, then restore order
        filtered_scored = sorted(
            filtered, key=lambda x: x[2] + x[3] * config.technical_weight, reverse=True
        )[:max_s]
        filtered = sorted(filtered_scored, key=lambda x: x[0])
    elif len(filtered) < min_s and all_sentences:
        # Add more sentences if below minimum
        already_have = {idx for idx, _, _, _ in filtered}
        for i, sent in enumerate(all_sentences):
            if i not in already_have:
                tech_score = score_technical_importance(sent)
                filtered.append((i, sent, 0.0, tech_score))
                if len(filtered) >= min_s:
                    break
        filtered.sort(key=lambda x: x[0])

    # --- Build ExtractedSentence objects with provenance ---
    extracted: List[ExtractedSentence] = []
    textrank_indices = {idx for idx, _, _, _ in annotated}
    technical_count = 0

    for pos, (orig_idx, text, tr_score, tech_score) in enumerate(filtered):
        # Determine selection reason
        in_textrank = orig_idx in textrank_indices
        is_technical = tech_score >= 0.25

        if in_textrank and is_technical:
            reason = "textrank+technical"
        elif in_textrank:
            reason = "textrank"
        elif is_technical:
            reason = "technical"
        else:
            reason = "min_coverage"

        if is_technical:
            technical_count += 1

        sentence_id = f"{paper_id}-{abbrev}-{orig_idx:02d}"

        # Source validation
        verified = validate_extracted_sentence(text, section_text)

        extracted.append(ExtractedSentence(
            sentence_id=sentence_id,
            paper_id=paper_id,
            section=section_name,
            text=text,
            original_index=orig_idx,
            textrank_score=round(tr_score, 6),
            technical_score=round(tech_score, 4),
            selected=True,
            selection_reason=reason,
            source_verified=verified,
        ))

    return SectionFactSheet(
        section_name=section_name,
        original_word_count=original_word_count,
        selected_sentence_count=len(extracted),
        technical_sentence_count=technical_count,
        selected_sentences=extracted,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compress_paper(
    paper_dict: dict,
    query: str = "",
    paper_id: Optional[str] = None,
    config: CompressionConfig = DEFAULT_CONFIG,
) -> PaperFactSheet:
    """
    Compress a single paper dict into a PaperFactSheet.

    Parameters
    ----------
    paper_dict:
        Dict with at least: 'title', 'authors', 'year', 'url', 'full_text'.
        'full_text' should be the complete acquired paper text (no pre-truncation).
    query:
        The user's research query (used only for logging, not for retrieval).
    paper_id:
        Optional short ID for this paper (e.g. "P001"). Auto-generated if not provided.
    config:
        Compression configuration. Uses DEFAULT_CONFIG if not provided.

    Returns
    -------
    PaperFactSheet with all sections compressed and metrics measured.
    """
    title = paper_dict.get("title", "Unknown")
    authors = paper_dict.get("authors", [])
    year = paper_dict.get("year", 2024)
    url = paper_dict.get("url", "")
    raw_text = paper_dict.get("full_text", "") or ""
    abstract_text = paper_dict.get("abstract", "") or ""

    # Auto-generate paper_id if not provided
    if not paper_id:
        safe_title = re.sub(r'[^a-z0-9]', '', title.lower())[:8]
        paper_id = f"P-{safe_title}"

    print(f"[COMPRESSION] {paper_id} — '{title[:55]}...' ({len(raw_text):,} chars)")

    # Log large papers
    if config.paper_size_warn_chars > 0 and len(raw_text) > config.paper_size_warn_chars:
        print(
            f"[COMPRESSION] WARNING: {paper_id} full text is {len(raw_text):,} chars. "
            f"Processing section-by-section (no truncation)."
        )

    # If no full text, use abstract as the only available content
    # Track coverage_type: caller may pass 'partial_paper' or 'full_paper' in paper_dict.
    # If we fall back to abstract here, override to 'abstract_only'.
    input_coverage_type = paper_dict.get("coverage_type", "unavailable")

    if not raw_text.strip():
        if abstract_text.strip():
            print(f"[COMPRESSION] {paper_id}: No full text. Using abstract only.")
            raw_text = abstract_text
            resolved_coverage = "abstract_only"
        else:
            print(f"[COMPRESSION] {paper_id}: No text available. Empty fact sheet.")
            return PaperFactSheet(
                paper_id=paper_id,
                title=title,
                authors=authors if isinstance(authors, list) else [str(authors)],
                year=year,
                source_url=url,
                coverage_type="unavailable",
                sections={},
            )
    else:
        # Full text was provided — respect the caller's coverage_type
        # (full_paper or partial_paper, set by the full-text resolver)
        resolved_coverage = input_coverage_type if input_coverage_type in (
            "full_paper", "partial_paper"
        ) else "full_paper"

    # ── Section parsing ──────────────────────────────────────────────────────
    is_html = bool(re.search(r'<[a-zA-Z][^>]{0,50}>', raw_text))
    sections_raw = parse_paper_sections(
        raw_text,
        is_html=is_html,
        paper_size_warn_chars=config.paper_size_warn_chars,
    )

    # If section parser found nothing useful, treat as flat "unknown"
    if not sections_raw:
        sections_raw = {"unknown": raw_text}

    # Ensure the abstract from metadata is always present
    if "abstract" not in sections_raw and abstract_text.strip():
        sections_raw["abstract"] = abstract_text

    print(f"[COMPRESSION] {paper_id}: Detected sections: {list(sections_raw.keys())}")

    # ── Per-section compression ──────────────────────────────────────────────
    section_sheets: Dict[str, SectionFactSheet] = {}
    all_selected_text_parts: List[str] = []

    for section_name, section_text in sections_raw.items():
        if not section_text.strip():
            continue
        sheet = _compress_section(paper_id, section_name, section_text, config)
        section_sheets[section_name] = sheet
        for sent in sheet.selected_sentences:
            all_selected_text_parts.append(sent.text)

    compressed_text = " ".join(all_selected_text_parts)

    # ── Compute metrics ──────────────────────────────────────────────────────
    detected_sections = len(sections_raw)
    covered_sections = sum(
        1 for s in section_sheets.values() if s.selected_sentence_count > 0
    )
    total_selected = sum(s.selected_sentence_count for s in section_sheets.values())
    total_technical = sum(s.technical_sentence_count for s in section_sheets.values())
    total_validated = sum(
        1
        for s in section_sheets.values()
        for sent in s.selected_sentences
        if sent.source_verified
    )

    metrics = measure_compression(
        original_text=raw_text,
        compressed_text=compressed_text,
        detected_sections=detected_sections,
        covered_sections=covered_sections,
        selected_sentences=total_selected,
        technical_sentences=total_technical,
        validated_sentences=total_validated,
    )

    # ── Log results ──────────────────────────────────────────────────────────
    print(
        f"[COMPRESSION] {paper_id} results:\n"
        f"  Original sections : {detected_sections}\n"
        f"  Covered sections  : {covered_sections}\n"
        f"  Original tokens   : {metrics.original_token_count:,}\n"
        f"  Compressed tokens : {metrics.compressed_token_count:,}\n"
        f"  Compression ratio : {metrics.compression_ratio * 100:.1f}%\n"
        f"  Selected sentences: {total_selected}\n"
        f"  Technical sentences: {total_technical}\n"
        f"  Faithfulness      : {metrics.faithfulness_ratio * 100:.1f}%"
    )

    if metrics.faithfulness_ratio < 1.0:
        n_violated = total_selected - total_validated
        print(
            f"[COMPRESSION] WARNING: {paper_id} has {n_violated} sentence(s) "
            f"that could not be verified verbatim. Check source_verified flag."
        )

    return PaperFactSheet(
        paper_id=paper_id,
        title=title,
        authors=authors if isinstance(authors, list) else [str(authors)],
        year=year,
        source_url=url,
        coverage_type=resolved_coverage,
        sections=section_sheets,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Fact sheet formatter (for Gemini prompt inclusion)
# ---------------------------------------------------------------------------

_SECTION_DISPLAY_ORDER = [
    "abstract", "introduction", "related_work", "problem_definition",
    "methodology", "dataset", "experiments", "results",
    "discussion", "limitations", "future_work", "conclusion", "unknown",
]

_SECTION_DISPLAY_LABELS = {
    "abstract": "ABSTRACT",
    "introduction": "INTRODUCTION",
    "related_work": "RELATED WORK",
    "problem_definition": "PROBLEM DEFINITION",
    "methodology": "METHODOLOGY",
    "dataset": "DATASET",
    "experiments": "EXPERIMENTS",
    "results": "RESULTS",
    "discussion": "DISCUSSION",
    "limitations": "LIMITATIONS",
    "future_work": "FUTURE WORK",
    "conclusion": "CONCLUSION",
    "unknown": "FULL TEXT (SECTION HEADERS NOT DETECTED)",
}


def format_fact_sheet(fact_sheet: PaperFactSheet) -> str:
    """
    Render a PaperFactSheet as a structured plain-text block for Gemini.

    Each sentence is verbatim from the original paper.
    Sections appear in canonical reading order.
    Compression statistics are included at the top.
    """
    lines = []
    sep = "=" * 60

    lines.append(sep)
    lines.append("PAPER FACT SHEET")
    lines.append("=" * 16)
    lines.append(f"Paper ID    : {fact_sheet.paper_id}")
    lines.append(f"Title       : {fact_sheet.title}")
    if fact_sheet.authors:
        authors_str = ", ".join(fact_sheet.authors[:5])
        if len(fact_sheet.authors) > 5:
            authors_str += f" et al. (+{len(fact_sheet.authors) - 5} more)"
        lines.append(f"Authors     : {authors_str}")
    lines.append(f"Year        : {fact_sheet.year}")
    if fact_sheet.source_url:
        lines.append(f"URL         : {fact_sheet.source_url}")

    m = fact_sheet.metrics
    lines.append("")
    lines.append("COMPRESSION METRICS")
    lines.append(f"  Original tokens (est.)   : {m.original_token_count:,}")
    lines.append(f"  Compressed tokens (est.) : {m.compressed_token_count:,}")
    lines.append(f"  Compression ratio        : {m.compression_ratio * 100:.1f}%")
    lines.append(f"  Section coverage         : {m.covered_sections}/{m.detected_sections} sections")
    lines.append(f"  Selected sentences       : {m.selected_sentence_count}")
    lines.append(f"  Technical sentences      : {m.technical_sentence_count}")
    lines.append(f"  Faithfulness             : {m.faithfulness_ratio * 100:.1f}%")
    lines.append(f"  Token count method       : {m.token_count_method}")
    lines.append("")
    # Coverage type badge
    coverage = getattr(fact_sheet, 'coverage_type', 'unavailable')
    if coverage == "full_paper":
        lines.append("COVERAGE: Full paper (complete available text processed)")
    elif coverage == "partial_paper":
        lines.append("COVERAGE: Partial paper (some content could not be extracted)")
    elif coverage == "abstract_only":
        lines.append("WARNING: ABSTRACT ONLY — full text was unavailable.")
        lines.append("         Gap analysis based on abstract evidence only.")
    else:
        lines.append("COVERAGE: Unavailable")
    lines.append("")
    lines.append("NOTE: Every sentence below is extracted verbatim from the original paper.")
    lines.append(sep)

    # Sections in canonical order, then any remaining
    ordered_keys = [k for k in _SECTION_DISPLAY_ORDER if k in fact_sheet.sections]
    remaining_keys = [k for k in fact_sheet.sections if k not in _SECTION_DISPLAY_ORDER]

    for section_name in ordered_keys + remaining_keys:
        sheet = fact_sheet.sections[section_name]
        if not sheet.selected_sentences:
            continue
        label = _SECTION_DISPLAY_LABELS.get(section_name, section_name.upper().replace("_", " "))
        lines.append(f"\n[{label}]")
        for sent in sheet.selected_sentences:
            lines.append(f"  [{sent.sentence_id}] {sent.text}")

    lines.append(f"\n{sep}")
    return "\n".join(lines)
