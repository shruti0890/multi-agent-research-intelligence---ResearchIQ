"""
section_parser.py
=================
Parses raw paper text or HTML into labelled, normalized sections.

Design principles:
- No fixed character truncation; processes the full available text.
- If section headings are found, returns each section mapped to a normalized key.
- If no headings are found, returns the full text under "unknown" as a controlled fallback.
- Missing sections are simply absent from the returned dict (not filled with placeholders).
- Gracefully handles malformed HTML, unusual headings, duplicate sections.

Returns: Dict[str, str]  — { normalized_section_key: raw_section_text }
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Section name normalization map
# Keys are lowercase substrings that appear in raw headings.
# The parser uses substring matching, not exact matching.
# ---------------------------------------------------------------------------

# Ordered list of (pattern, normalized_key) — earlier entries take priority.
_NORMALIZATION_RULES: List[Tuple[str, str]] = [
    # Abstract variants
    ("abstract", "abstract"),
    ("summary", "abstract"),

    # Introduction variants
    ("introduction", "introduction"),
    ("background and motivation", "introduction"),
    ("motivation", "introduction"),
    ("overview", "introduction"),

    # Related Work variants
    ("related work", "related_work"),
    ("prior work", "related_work"),
    ("literature review", "related_work"),
    ("previous work", "related_work"),
    ("state of the art", "related_work"),
    ("related", "related_work"),

    # Problem definition variants
    ("problem formulation", "problem_definition"),
    ("problem definition", "problem_definition"),
    ("problem statement", "problem_definition"),
    ("task definition", "problem_definition"),
    ("formulation", "problem_definition"),

    # Methodology variants
    ("materials and methods", "methodology"),
    ("methods and materials", "methodology"),
    ("proposed method", "methodology"),
    ("proposed approach", "methodology"),
    ("our approach", "methodology"),
    ("our method", "methodology"),
    ("approach", "methodology"),
    ("methodology", "methodology"),
    ("method", "methodology"),
    ("model", "methodology"),
    ("architecture", "methodology"),
    ("framework", "methodology"),
    ("system design", "methodology"),
    ("technical approach", "methodology"),
    ("algorithm", "methodology"),

    # Dataset variants
    ("dataset", "dataset"),
    ("data collection", "dataset"),
    ("data", "dataset"),
    ("corpus", "dataset"),
    ("benchmark dataset", "dataset"),

    # Results variants — MORE SPECIFIC patterns must appear BEFORE generic 'experiment'
    # because _normalize_section_name uses substring matching (first match wins).
    # "experimental results" contains "experiment" so it must appear first.
    ("experimental results", "results"),
    ("quantitative result", "results"),
    ("qualitative result", "results"),
    ("performance analysis", "results"),
    ("result", "results"),
    ("finding", "results"),
    ("ablation study", "results"),
    ("ablation", "results"),

    # Experiments variants — placed AFTER the more-specific result patterns
    ("experimental setup", "experiments"),
    ("experimental settings", "experiments"),
    ("experimental evaluation", "experiments"),
    ("experiments and results", "experiments"),
    ("experiment", "experiments"),
    ("evaluation setup", "experiments"),
    ("evaluation", "experiments"),
    ("implementation detail", "experiments"),
    ("setup", "experiments"),

    # Generic result patterns — after experiment to avoid premature matches
    ("performance", "results"),
    ("analysis", "results"),
    ("comparison", "results"),

    # Discussion variants
    ("discussion and conclusion", "discussion"),
    ("discussion", "discussion"),
    ("interpretation", "discussion"),

    # Limitations — check before future_work combos
    ("limitations and future work", "limitations"),
    ("limitation", "limitations"),
    ("shortcoming", "limitations"),
    ("weakness", "limitations"),
    ("failure case", "limitations"),

    # Future work variants
    ("future work", "future_work"),
    ("future direction", "future_work"),
    ("open problem", "future_work"),
    ("future research", "future_work"),

    # Conclusion variants
    ("conclusion and future", "conclusion"),
    ("concluding remark", "conclusion"),
    ("conclusion", "conclusion"),
    ("summary and conclusion", "conclusion"),
    ("closing remark", "conclusion"),
]

# Section keys to check after "limitations" match on combined headings so that
# "Limitations and Future Work" splits correctly.
_COMBINED_SPLITS: Dict[str, Tuple[str, str]] = {
    "limitations and future work": ("limitations", "future_work"),
    "conclusion and future work":  ("conclusion", "future_work"),
    "discussion and conclusion":   ("discussion", "conclusion"),
    "experiments and results":     ("experiments", "results"),
}

# Heading detection: matches lines/spans that look like section headings.
# A heading is typically:
#   - All-caps or Title Case
#   - Optionally prefixed with a section number (1., 2.1, I., A.)
#   - Short (< 100 chars)
_HEADING_PATTERN = re.compile(
    r'^'
    r'(?:\d+[\.\d]*\s+|[IVX]+\.\s+|[A-Z]\.\s+)?'   # optional numbering
    r'([A-Z][^\n]{0,90})'                              # heading text
    r'$',
    re.MULTILINE
)

# HTML heading tag extractor
_HTML_HEADING_RE = re.compile(
    r'<(h[1-6]|b|strong)[^>]*>(.*?)</\1>',
    re.IGNORECASE | re.DOTALL
)


def _clean_html(html: str) -> str:
    """Strip HTML tags and decode common entities. Preserves whitespace structure."""
    # Remove script/style blocks
    html = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', html, flags=re.IGNORECASE | re.DOTALL)
    # Remove HTML comments
    html = re.sub(r'<!--.*?-->', ' ', html, flags=re.DOTALL)
    # Replace block-level tags with newlines
    html = re.sub(r'<(p|div|br|li|h[1-6]|section|article|header|footer)[^>]*>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'</(p|div|li|h[1-6]|section|article|header|footer)>', '\n', html, flags=re.IGNORECASE)
    # Remove remaining tags
    html = re.sub(r'<[^>]+>', ' ', html)
    # Decode common HTML entities
    html = html.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    html = html.replace('&nbsp;', ' ').replace('&quot;', '"').replace('&#39;', "'")
    html = html.replace('&mdash;', '—').replace('&ndash;', '–').replace('&hellip;', '...')
    # Collapse excessive whitespace while preserving paragraph breaks
    html = re.sub(r'[ \t]+', ' ', html)
    html = re.sub(r'\n{3,}', '\n\n', html)
    return html.strip()


def _normalize_section_name(raw_heading: str) -> Optional[str]:
    """
    Map a raw heading string to a normalized section key.
    Returns None if the heading does not match any known section.
    Handles combined headings (e.g., "Limitations and Future Work") by
    returning the primary key; the caller handles splitting.
    """
    lower = raw_heading.lower().strip()

    # Check combined headings first
    for combo_text, (primary, _secondary) in _COMBINED_SPLITS.items():
        if combo_text in lower:
            return primary  # Caller can request secondary separately

    # Standard rules (first match wins — order matters in _NORMALIZATION_RULES)
    for pattern, key in _NORMALIZATION_RULES:
        if pattern in lower:
            return key

    return None


def _get_combined_secondary(raw_heading: str) -> Optional[str]:
    """
    If the heading is a known combined heading, return the secondary section key.
    E.g., "Limitations and Future Work" → "future_work"
    """
    lower = raw_heading.lower().strip()
    for combo_text, (_primary, secondary) in _COMBINED_SPLITS.items():
        if combo_text in lower:
            return secondary
    return None


def _split_into_raw_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split plain text into (raw_heading, section_text) tuples using heading detection.
    If no headings are found, returns [("unknown", full_text)].
    """
    lines = text.split('\n')
    sections: List[Tuple[str, str]] = []
    current_heading = "preamble"
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_lines.append(line)
            continue

        # A heading candidate: short, mostly title-case or all-caps, no trailing period
        is_heading = (
            len(stripped) < 100
            and not stripped.endswith('.')
            and not stripped.endswith(',')
            and re.match(r'^(?:\d+[\.\d]*\s+)?[A-Z]', stripped)
            and len(stripped.split()) <= 12
            # Must contain at least one alphabetic character
            and re.search(r'[a-zA-Z]{3,}', stripped)
            # Should not look like a sentence (no verb-like patterns in the middle)
            and _normalize_section_name(stripped) is not None
        )

        if is_heading:
            # Save the previous section
            section_text = '\n'.join(current_lines).strip()
            if section_text:
                sections.append((current_heading, section_text))
            current_heading = stripped
            current_lines = []
        else:
            current_lines.append(line)

    # Save the last section
    section_text = '\n'.join(current_lines).strip()
    if section_text:
        sections.append((current_heading, section_text))

    # If no real headings were found, all text is under "preamble" or "unknown"
    if len(sections) <= 1:
        full_text = '\n'.join(
            text for _, text in sections
        ).strip() or text.strip()
        return [("unknown", full_text)]

    return sections


def parse_paper_sections(
    raw_text: str,
    is_html: bool = False,
    paper_size_warn_chars: int = 500_000,
) -> Dict[str, str]:
    """
    Main entry point for section parsing.

    Parameters
    ----------
    raw_text:
        The full raw paper text or HTML. Not truncated before calling this.
    is_html:
        If True, strips HTML tags before parsing.
    paper_size_warn_chars:
        If the paper exceeds this size, log a warning but do NOT truncate.

    Returns
    -------
    Dict[str, str]
        Mapping from normalized section key to section text.
        Missing sections are absent (not represented with empty strings).
    """
    if not raw_text or not raw_text.strip():
        return {}

    # Log large papers
    if paper_size_warn_chars > 0 and len(raw_text) > paper_size_warn_chars:
        print(
            f"[PARSER] WARNING: Paper text is {len(raw_text):,} chars "
            f"(>{paper_size_warn_chars:,}). Processing section-by-section."
        )

    # Clean HTML if needed
    if is_html or bool(re.search(r'<[a-zA-Z][^>]{0,50}>', raw_text)):
        text = _clean_html(raw_text)
    else:
        text = raw_text

    # Normalize whitespace but preserve paragraph structure
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Split into raw (heading, text) pairs
    raw_sections = _split_into_raw_sections(text)

    # Map to normalized keys — handle duplicates by concatenating
    result: Dict[str, str] = {}
    for raw_heading, section_text in raw_sections:
        if not section_text.strip():
            continue

        norm_key = _normalize_section_name(raw_heading)
        if norm_key is None:
            # Heading not recognized — group under 'unknown' only if it has content
            norm_key = "unknown"

        if norm_key in result:
            # Duplicate section: concatenate (handles papers where abstract is split)
            result[norm_key] = result[norm_key] + "\n\n" + section_text
        else:
            result[norm_key] = section_text

        # Handle combined headings — add secondary key if content is long enough
        secondary = _get_combined_secondary(raw_heading)
        if secondary and secondary not in result and len(section_text) > 100:
            # Split roughly at the midpoint for combined headings
            mid = len(section_text) // 2
            # Find nearest paragraph break
            split_pos = section_text.find('\n\n', mid - 100)
            if split_pos == -1 or split_pos > mid + 200:
                split_pos = mid
            primary_text = section_text[:split_pos].strip()
            secondary_text = section_text[split_pos:].strip()
            if primary_text:
                result[norm_key] = primary_text
            if secondary_text:
                result[secondary] = secondary_text

    return result
