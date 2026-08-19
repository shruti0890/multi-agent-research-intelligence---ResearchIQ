"""
textrank.py
===========
TextRank extractive summarization for individual paper sections.

Design principles:
- EXTRACTIVE ONLY: every returned sentence is verbatim from the source.
- No LLM, no embeddings, no sentence-transformers.
- Sentence similarity via TF-IDF cosine (pure Python + networkx).
- NLTK punkt tokenizer for robust sentence segmentation.
- Restores original sentence order after TextRank selection.
- Adaptive sentence budget controlled by CompressionConfig.
- Minimum 1 sentence enforced for meaningful sections.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# NLTK punkt tokenizer with graceful fallback
# ---------------------------------------------------------------------------

def _load_sent_tokenizer():
    """Load NLTK punkt tokenizer, downloading if needed."""
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.download('punkt_tab', quiet=True)
            except Exception:
                pass
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except Exception:
                pass
        from nltk.tokenize import sent_tokenize
        return sent_tokenize
    except ImportError:
        return None


_sent_tokenize = _load_sent_tokenizer()

# Stop words for TF-IDF
_STOP_WORDS = frozenset([
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
    'she', 'her', 'they', 'them', 'what', 'which', 'who', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'a', 'an', 'the',
    'and', 'but', 'if', 'or', 'as', 'at', 'by', 'for', 'with', 'about',
    'of', 'to', 'from', 'in', 'out', 'on', 'up', 'it', 'its', 'so',
    'than', 'too', 'very', 'can', 'will', 'just', 'not', 'also', 'into',
    'through', 'during', 'such', 'then', 'when', 'where', 'how', 'all',
    'each', 'both', 'more', 'most', 'other', 'same', 'no', 'nor', 'only',
    'own', 'here', 'there', 'over', 'under', 'again',
])


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------

def segment_sentences(text: str) -> List[str]:
    """
    Segment text into sentences using NLTK punkt tokenizer.
    Falls back to a regex-based heuristic if NLTK is unavailable.

    The tokenizer handles:
    - Decimal values (3.14, 0.95)
    - Common abbreviations (et al., vs., etc.)
    - Citations ([1], (Smith et al., 2021))
    - URLs, mathematical formulas, and statistical reports
    - Preserves short legitimate research sentences
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    if _sent_tokenize is not None:
        try:
            sentences = _sent_tokenize(text)
            # Post-process: merge tiny non-sentence fragments (< 3 words without uppercase/math) back into previous sentence
            merged = []
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if merged and len(s.split()) < 3 and not s[0].isupper() and not re.search(r'[0-9=><\+\-±%]', s):
                    merged[-1] = merged[-1] + ' ' + s
                else:
                    merged.append(s)
            # Retain all sentences with at least 2 words or containing alphanumeric/mathematical content
            return [s for s in merged if len(s.split()) >= 2 or re.search(r'[a-zA-Z0-9]', s)]
        except Exception:
            pass

    # Fallback: regex sentence splitter
    # Avoids splitting on: decimal numbers, abbreviations (2-3 char), citations
    sentence_endings = re.compile(
        r'(?<!\w\.\w)'           # not "e.g." or "U.S."
        r'(?<![A-Z][a-z]\.)'    # not "Mr." / "Dr."
        r'(?<!\d)'               # not after digit
        r'(?<!\.\d)'             # not mid-decimal
        r'(?:[.!?])'             # sentence end punctuation
        r'(?:\s+|$)'             # followed by whitespace or end
        r'(?=[A-Z"\'\[]|$)',     # followed by capital or quote
        re.MULTILINE
    )
    parts = sentence_endings.split(text)
    sentences = [p.strip() for p in parts if p and (len(p.strip().split()) >= 2 or re.search(r'[a-zA-Z0-9]', p))]
    return sentences


# ---------------------------------------------------------------------------
# TF-IDF representation
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens, removing stop words and short tokens."""
    words = re.findall(r'\b[a-zA-Z0-9][a-zA-Z0-9_\-]{1,}\b', text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) >= 2]


def _build_tfidf_vectors(sentences: List[str]) -> List[Dict[str, float]]:
    """
    Build TF-IDF vectors for a list of sentences.
    Returns a list of {term: tfidf_weight} dicts, one per sentence.
    """
    n = len(sentences)
    if n == 0:
        return []

    # Tokenize
    tokenized = [_tokenize(s) for s in sentences]

    # Term frequency per sentence
    tf_docs = []
    for tokens in tokenized:
        tf: Dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        # Normalize by sentence length to avoid bias toward longer sentences
        length = max(len(tokens), 1)
        tf = {t: v / length for t, v in tf.items()}
        tf_docs.append(tf)

    # Document frequency
    df: Dict[str, int] = {}
    for tokens in tokenized:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    # IDF
    idf: Dict[str, float] = {
        t: math.log((1 + n) / (1 + count)) + 1.0
        for t, count in df.items()
    }

    # TF-IDF
    tfidf_vectors = []
    for tf in tf_docs:
        vec = {t: tf[t] * idf.get(t, 1.0) for t in tf}
        tfidf_vectors.append(vec)

    return tfidf_vectors


def _cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Cosine similarity between two TF-IDF vectors."""
    if not vec1 or not vec2:
        return 0.0
    intersection = set(vec1) & set(vec2)
    if not intersection:
        return 0.0
    numerator = sum(vec1[t] * vec2[t] for t in intersection)
    denom = math.sqrt(sum(v ** 2 for v in vec1.values())) * math.sqrt(
        sum(v ** 2 for v in vec2.values())
    )
    if denom == 0.0:
        return 0.0
    return numerator / denom


# ---------------------------------------------------------------------------
# TextRank graph scoring
# ---------------------------------------------------------------------------

def _textrank_scores(sentences: List[str], damping: float = 0.85, iterations: int = 25) -> List[float]:
    """
    Compute TextRank scores for a list of sentences.

    Builds a weighted directed graph where edge weight = sentence similarity,
    then applies the PageRank iteration formula.

    Returns a list of floats (one score per sentence), in original order.
    """
    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    # For very large sections (e.g. synthetic benchmarks with 1000+ sentences),
    # evaluate representative candidate sentences for the graph to maintain fast performance.
    MAX_GRAPH_NODES = 150
    if n > MAX_GRAPH_NODES:
        step = n / MAX_GRAPH_NODES
        sample_indices = [min(n - 1, int(i * step)) for i in range(MAX_GRAPH_NODES)]
        sampled_sentences = [sentences[i] for i in sample_indices]
        sampled_scores = _textrank_scores(sampled_sentences, damping=damping, iterations=iterations)

        full_scores = [1.0 / n] * n
        for idx, score in zip(sample_indices, sampled_scores):
            full_scores[idx] = score
        return full_scores

    vectors = _build_tfidf_vectors(sentences)

    # Build similarity matrix
    sim_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim_matrix[i][j] = _cosine_similarity(vectors[i], vectors[j])

    # Normalize rows
    for i in range(n):
        row_sum = sum(sim_matrix[i])
        if row_sum > 0:
            sim_matrix[i] = [v / row_sum for v in sim_matrix[i]]

    # Power iteration
    scores = [1.0 / n] * n
    for _ in range(iterations):
        new_scores = [0.0] * n
        for i in range(n):
            s = sum(sim_matrix[j][i] * scores[j] for j in range(n))
            new_scores[i] = (1 - damping) / n + damping * s
        scores = new_scores

    return scores


# ---------------------------------------------------------------------------
# Sentence selection with order restoration
# ---------------------------------------------------------------------------

def run_textrank_for_section(
    section_text: str,
    max_sentences: int = 5,
    min_sentences: int = 1,
) -> List[Tuple[int, str, float]]:
    """
    Run TextRank on a single section and return selected (index, sentence, score) tuples.

    Parameters
    ----------
    section_text:
        The full text of one paper section.
    max_sentences:
        Maximum number of sentences to select.
    min_sentences:
        Minimum sentences to select (enforced even if TextRank scores are low).

    Returns
    -------
    List of (original_index, sentence_text, textrank_score) tuples,
    sorted by original_index (i.e., restored to document order).
    """
    sentences = segment_sentences(section_text)
    if not sentences:
        return []

    n = len(sentences)

    # Cap at the number of available sentences
    max_sentences = min(max_sentences, n)
    min_sentences = min(min_sentences, n)

    if n == 1:
        return [(0, sentences[0], 1.0)]

    scores = _textrank_scores(sentences)

    # Pair sentences with their original index and score
    ranked = sorted(
        enumerate(zip(sentences, scores)),
        key=lambda x: x[1][1],  # sort by score desc
        reverse=True,
    )

    # Select top-max_sentences
    selected_indices = set()
    for orig_idx, (_, score) in ranked[:max_sentences]:
        selected_indices.add(orig_idx)

    # Ensure minimum coverage
    if len(selected_indices) < min_sentences:
        for orig_idx, (_, _score) in ranked:
            if orig_idx not in selected_indices:
                selected_indices.add(orig_idx)
            if len(selected_indices) >= min_sentences:
                break

    # Restore to original document order
    result = []
    score_map = {idx: sc for idx, (_, sc) in enumerate(zip(sentences, scores))}
    for idx in sorted(selected_indices):
        result.append((idx, sentences[idx], score_map.get(idx, 0.0)))

    return result
