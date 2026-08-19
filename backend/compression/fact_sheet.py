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
from .section_parser import NON_RESEARCH_SECTIONS, parse_paper_sections
from .technical_preservation import (
    filter_technical_sentences,
    score_technical_importance,
)
from .textrank import (
    _textrank_scores,
    run_textrank_for_section,
    segment_sentences,
)
from .token_utils import (
    estimate_tokens,
    measure_compression,
    validate_extracted_sentence,
)

# ---------------------------------------------------------------------------
# Section abbreviation map for sentence IDs
# ---------------------------------------------------------------------------
_SECTION_ABBREV: Dict[str, str] = {
    "abstract": "ABS",
    "introduction": "INTRO",
    "background": "BG",
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
# Sentence Candidate Extraction & Scoring
# ---------------------------------------------------------------------------

def _extract_and_score_section_sentences(
    paper_id: str,
    section_name: str,
    section_text: str,
    config: CompressionConfig,
) -> List[dict]:
    """
    Segments a section into sentences, computes TextRank PageRank scores,
    evaluates technical feature density, and computes a composite priority score.

    Returns a list of candidate dictionaries:
        {
            "paper_id": str,
            "section": str,
            "index": int,
            "text": str,
            "tokens": int,
            "textrank_score": float,
            "technical_score": float,
            "composite_score": float,
        }
    """
    if not section_text or not section_text.strip():
        return []

    # Warn on unusually large sections
    if len(section_text) > config.section_size_warn_chars:
        print(
            f"[COMPRESSION] WARNING: Section '{section_name}' in {paper_id} "
            f"is {len(section_text):,} chars. Processing normally (no truncation)."
        )

    sentences = segment_sentences(section_text)
    if not sentences:
        return []

    tr_scores = _textrank_scores(sentences)
    sec_weight = config.section_priority_weights.get(section_name, 1.0)

    candidates = []
    for i, sent in enumerate(sentences):
        tech_score = score_technical_importance(sent)
        tr_score = tr_scores[i] if i < len(tr_scores) else 0.0
        # Priority order: Section weight * (TextRank + technical boost)
        composite = (tr_score + (tech_score * config.technical_weight)) * sec_weight
        tokens = estimate_tokens(sent)

        candidates.append({
            "paper_id": paper_id,
            "section": section_name,
            "index": i,
            "text": sent,
            "tokens": tokens,
            "textrank_score": tr_score,
            "technical_score": tech_score,
            "composite_score": composite,
        })

    return candidates


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
    Compress a single paper dict into a PaperFactSheet targeting 65–75% RETENTION
    (25–35% reduction, preferred ~70% retention).

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
    input_coverage_type = paper_dict.get("coverage_type", "unavailable")

    def _has_real_abstract(text: str) -> bool:
        if not text or not text.strip():
            return False
        _placeholder_re = re.compile(
            r'^(no abstract available\.?|abstract not available\.?|not available\.?|n/a\.?|abstract metadata fetched from)\s*$',
            re.IGNORECASE
        )
        if _placeholder_re.match(text.strip()):
            return False
        return len(text.strip().split()) >= 5

    if not raw_text.strip():
        if _has_real_abstract(abstract_text):
            print(f"[COMPRESSION] {paper_id}: No full text. Using abstract only.")
            raw_text = abstract_text
            resolved_coverage = "abstract_only"
        else:
            if abstract_text.strip():
                print(f"[COMPRESSION] {paper_id}: No full text and no real abstract (placeholder). Empty fact sheet.")
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

    if not sections_raw:
        sections_raw = {"unknown": raw_text}

    if "abstract" not in sections_raw and abstract_text.strip():
        sections_raw["abstract"] = abstract_text

    print(f"[COMPRESSION] {paper_id}: Detected sections: {list(sections_raw.keys())}")

    # ── Token Targets: 65–75% RETENTION (Target ~70%) ────────────────────────
    orig_tokens = estimate_tokens(raw_text)
    min_target_tokens = int(round(orig_tokens * config.target_retention_min))      # 65%
    max_target_tokens = int(round(orig_tokens * config.target_retention_max))      # 75%
    preferred_target_tokens = int(round(orig_tokens * config.target_retention_default))  # 70%

    # ── Segment & Score Research Sections ────────────────────────────────────
    non_research_count = 0
    research_section_names: List[str] = []
    section_candidates: Dict[str, List[dict]] = {}

    for section_name, section_text in sections_raw.items():
        if not section_text.strip():
            continue
        if section_name in NON_RESEARCH_SECTIONS:
            print(f"[COMPRESSION] {paper_id}: Skipping non-research section '{section_name}'")
            non_research_count += 1
            continue

        research_section_names.append(section_name)
        cands = _extract_and_score_section_sentences(
            paper_id=paper_id,
            section_name=section_name,
            section_text=section_text,
            config=config,
        )
        if cands:
            section_candidates[section_name] = cands

    # Calculate total available candidate tokens across all research sections
    available_candidate_tokens = sum(
        sum(c["tokens"] for c in cands) for cands in section_candidates.values()
    )

    # ── Candidate Pool Diagnostics ───────────────────────────────────────────
    research_text_tokens = sum(estimate_tokens(sections_raw.get(s, "")) for s in research_section_names)
    excluded_non_research_tokens = sum(estimate_tokens(sections_raw.get(s, "")) for s in sections_raw if s in NON_RESEARCH_SECTIONS)
    sent_extracted_tokens = available_candidate_tokens
    filtered_sent_tokens = sent_extracted_tokens
    candidate_pool_tokens = available_candidate_tokens

    print(
        f"[COMPRESSION] {paper_id} Diagnostics:\n"
        f"  Original raw tokens        : {orig_tokens:,}\n"
        f"  Research text tokens       : {research_text_tokens:,}\n"
        f"  Excluded non-research tokens: {excluded_non_research_tokens:,}\n"
        f"  Sentence extraction tokens : {sent_extracted_tokens:,}\n"
        f"  Filtered sentence tokens   : {filtered_sent_tokens:,}\n"
        f"  Candidate pool tokens      : {candidate_pool_tokens:,}"
    )

    # ── Sentence Selection Algorithm ─────────────────────────────────────────
    # Phase 1: Guaranteed Research Section Coverage (Minimum allocation)
    selected_by_section: Dict[str, List[dict]] = {sec: [] for sec in research_section_names}
    selected_indices_by_section: Dict[str, set] = {sec: set() for sec in research_section_names}
    current_tokens = 0

    for sec_name, cands in section_candidates.items():
        base_budget = config.budgets.get(sec_name, config.budgets.get("unknown"))
        min_s = min(len(cands), max(1, base_budget.min_sentences if base_budget else 1))

        # Sort candidates in this section by composite score descending
        sorted_cands = sorted(cands, key=lambda c: c["composite_score"], reverse=True)
        for c in sorted_cands[:min_s]:
            c["selection_reason"] = "min_coverage"
            selected_by_section[sec_name].append(c)
            selected_indices_by_section[sec_name].add(c["index"])
            current_tokens += c["tokens"]

    # Phase 2: Priority Global Token Accumulation up to Preferred Target (~70%)
    remaining_candidates: List[dict] = []
    for sec_name, cands in section_candidates.items():
        for c in cands:
            if c["index"] not in selected_indices_by_section[sec_name]:
                remaining_candidates.append(c)

    # Sort global candidate pool by composite score descending
    remaining_candidates.sort(key=lambda c: c["composite_score"], reverse=True)

    for c in remaining_candidates:
        # Check if preferred retention target is reached
        if current_tokens >= preferred_target_tokens:
            if current_tokens >= min_target_tokens:
                break

        sec_name = c["section"]
        c_tokens = c["tokens"]

        # Check section max_sentences limit if configured
        sec_budget = config.budgets.get(sec_name, config.budgets.get("unknown"))
        if sec_budget and len(selected_by_section[sec_name]) >= sec_budget.max_sentences:
            continue

        # Diversity check within section: skip near-duplicate unless needed for minimum retention
        is_duplicate = False
        if config.diversity_threshold < 1.0:
            for sel in selected_by_section[sec_name]:
                if _jaccard(c["text"], sel["text"]) >= config.diversity_threshold:
                    is_duplicate = True
                    break

        if is_duplicate and (current_tokens + c_tokens >= min_target_tokens):
            continue

        # Prevent exceeding maximum target tokens (75%) if already at or above minimum (65%)
        if (current_tokens + c_tokens > max_target_tokens) and (current_tokens >= min_target_tokens):
            continue

        # Determine selection reason
        tr_score = c["textrank_score"]
        tech_score = c["technical_score"]
        if tr_score > 0 and tech_score >= 0.25:
            reason = "textrank+technical"
        elif tr_score > 0:
            reason = "textrank"
        elif tech_score >= 0.25:
            reason = "technical"
        else:
            reason = "budget_fill"

        c["selection_reason"] = reason
        selected_by_section[sec_name].append(c)
        selected_indices_by_section[sec_name].add(c["index"])
        current_tokens += c_tokens

    # Phase 3: Insufficient Candidate Detection & Warning
    if current_tokens < min_target_tokens and orig_tokens > 500:
        shortfall = max(0, min_target_tokens - current_tokens)
        print(
            f"[COMPRESSION] WARNING: Insufficient eligible source sentences to reach retention target.\n"
            f"  Original tokens           : {orig_tokens:,}\n"
            f"  Available candidate tokens: {available_candidate_tokens:,}\n"
            f"  Target tokens             : {min_target_tokens:,}–{max_target_tokens:,}\n"
            f"  Actual tokens             : {current_tokens:,}\n"
            f"  Shortfall                 : {shortfall:,}"
        )

    # ── Build Section Fact Sheets with Provenance & Verbatim Validation ──────
    section_sheets: Dict[str, SectionFactSheet] = {}
    all_selected_text_parts: List[str] = []

    for section_name in research_section_names:
        section_text = sections_raw.get(section_name, "")
        original_word_count = len(section_text.split())
        abbrev = _section_abbrev(section_name)

        selected_cands = selected_by_section.get(section_name, [])
        # Restore original reading order
        selected_cands.sort(key=lambda c: c["index"])

        extracted: List[ExtractedSentence] = []
        technical_count = 0

        for c in selected_cands:
            orig_idx = c["index"]
            sent_text = c["text"]
            tr_score = c["textrank_score"]
            tech_score = c["technical_score"]
            reason = c.get("selection_reason", "textrank")

            if tech_score >= 0.25:
                technical_count += 1

            sentence_id = f"{paper_id}-{abbrev}-{orig_idx:02d}"
            verified = validate_extracted_sentence(sent_text, section_text)

            extracted.append(ExtractedSentence(
                sentence_id=sentence_id,
                paper_id=paper_id,
                section=section_name,
                text=sent_text,
                original_index=orig_idx,
                textrank_score=round(tr_score, 6),
                technical_score=round(tech_score, 4),
                selected=True,
                selection_reason=reason,
                source_verified=verified,
            ))
            all_selected_text_parts.append(sent_text)

        section_sheets[section_name] = SectionFactSheet(
            section_name=section_name,
            original_word_count=original_word_count,
            selected_sentence_count=len(extracted),
            technical_sentence_count=technical_count,
            selected_sentences=extracted,
        )

    compressed_text = " ".join(all_selected_text_parts)

    # ── Compute metrics ──────────────────────────────────────────────────────
    detected_sections = len(sections_raw)
    research_sec_count = len(research_section_names)
    covered_sections = sum(
        1 for s in section_sheets.values() if s.selected_sentence_count > 0
    )
    research_sec_covered = covered_sections

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
        research_sections=research_sec_count,
        research_sections_covered=research_sec_covered,
        non_research_sections_skipped=non_research_count,
    )

    if resolved_coverage == "abstract_only":
        metrics.compression_status = "not_evaluated"

    # ── Log results ──────────────────────────────────────────────────────────
    if research_sec_count > 0:
        rs_cov_str = f"{research_sec_covered}/{research_sec_count} = {metrics.research_section_coverage * 100:.0f}%"
    else:
        rs_cov_str = "N/A"

    print(
        f"[COMPRESSION] {paper_id} results:\n"
        f"  Original tokens         : {orig_tokens:,}\n"
        f"  Target retention        : {config.target_retention_min * 100:.0f}–{config.target_retention_max * 100:.0f}%\n"
        f"  Preferred retention     : {config.target_retention_default * 100:.0f}%\n"
        f"  Target tokens           : {min_target_tokens:,}–{max_target_tokens:,}\n"
        f"  Compressed tokens (est.): {metrics.compressed_token_count:,}\n"
        f"  Actual retention        : {metrics.retained_ratio * 100:.1f}%\n"
        f"  Reduction ratio         : {metrics.reduction_ratio * 100:.1f}%\n"
        f"  Research sec. coverage  : {rs_cov_str}\n"
        f"  Selected sentences      : {total_selected}\n"
        f"  Technical sentences     : {total_technical}\n"
        f"  Faithfulness            : {metrics.faithfulness_ratio * 100:.1f}%"
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
    "abstract", "introduction", "background", "related_work", "problem_definition",
    "methodology", "dataset", "experiments", "results",
    "discussion", "limitations", "future_work", "conclusion", "unknown",
]

_SECTION_DISPLAY_LABELS = {
    "abstract": "ABSTRACT",
    "introduction": "INTRODUCTION",
    "background": "BACKGROUND",
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
    lines.append(f"  Original tokens (est.)          : {m.original_token_count:,}")
    lines.append(f"  Compressed tokens (est.)        : {m.compressed_token_count:,}")
    # Display N/A for abstract-only papers where the compression ratio is not meaningful
    if m.compression_status == "abstract_only_not_evaluated":
        lines.append(f"  Compression ratio               : N/A (abstract only — not evaluated)")
    else:
        lines.append(f"  Compression ratio               : {m.compression_ratio * 100:.1f}%")
    # Show research-section-aware coverage (correct metric)
    if m.research_sections > 0:
        lines.append(
            f"  Detected sections               : {m.detected_sections} "
            f"(research: {m.research_sections}, skipped non-research: {m.non_research_sections_skipped})"
        )
        lines.append(
            f"  Research section coverage       : "
            f"{m.research_sections_covered}/{m.research_sections} sections "
            f"= {m.research_section_coverage * 100:.0f}%"
        )
    else:
        lines.append(f"  Section coverage                : {m.covered_sections}/{m.detected_sections} sections")
    lines.append(f"  Selected sentences              : {m.selected_sentence_count}")
    lines.append(f"  Technical sentences             : {m.technical_sentence_count}")
    lines.append(f"  Faithfulness                    : {m.faithfulness_ratio * 100:.1f}%")
    lines.append(f"  Token count method              : {m.token_count_method} (NOT exact Gemini token count)")
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
