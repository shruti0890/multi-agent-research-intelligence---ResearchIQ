"""
technical_preservation.py
==========================
Detects and scores sentences containing critical technical information
that must be preserved regardless of TextRank score.

Design principles:
- Pattern-based only. No LLM, no embeddings.
- Returns a float score in [0.0, 1.0] per sentence.
- Higher score = more technical importance.
- Supplements TextRank; does not replace it.

Categories detected:
  - Performance metrics (accuracy, F1, AUROC, BLEU, ROUGE, etc.)
  - Numerical results (percentages, comparisons, reductions)
  - Experimental settings (batch size, epochs, learning rate)
  - Dataset / benchmark references
  - Model / algorithm names
  - Compute characteristics (parameters, FLOPs, latency, memory)
"""

from __future__ import annotations

import re
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Pattern definitions — compiled once at module load
# ---------------------------------------------------------------------------

# Each entry: (regex_pattern, weight)
# Weight contributes to the final score proportionally.
_PATTERNS: List[Tuple[re.Pattern, float]] = []

def _p(pattern: str, weight: float, flags: int = re.IGNORECASE) -> None:
    _PATTERNS.append((re.compile(pattern, flags), weight))


# ── Performance metric names ────────────────────────────────────────────────
_p(r'\b(accuracy|acc)\b', 0.5)
_p(r'\b(f1[_\s-]?score|f1)\b', 0.6)
_p(r'\b(precision|recall)\b', 0.5)
_p(r'\b(auroc|auc|roc)\b', 0.6)
_p(r'\b(bleu|rouge[_\s-]?\d?|meteor|cider)\b', 0.6)
_p(r'\b(perplexity|ppl)\b', 0.6)
_p(r'\b(rmse|mae|mse|mape)\b', 0.6)
_p(r'\b(map|ndcg|mrr|hit@\d)\b', 0.6)
_p(r'\b(em|exact\s+match)\b', 0.5)

# ── Quantitative comparison words ───────────────────────────────────────────
_p(r'\b(outperform|surpass|exceed|improve|achieve|obtain|attain)\b', 0.3)
_p(r'\b(state[_\s-]of[_\s-]the[_\s-]art|sota)\b', 0.5)
_p(r'\b(baseline|benchmark)\b', 0.4)
_p(r'\b(reduction|decrease|increase|improvement|gain|drop)\b', 0.3)
_p(r'\b(significant|substantially|notably|considerably)\b', 0.2)
_p(r'\b(best|top|highest|lowest|optimal|superior)\b', 0.2)
_p(r'\b(comparison|compared|vs\.?|versus|outperf)\b', 0.3)
_p(r'\b(ablation)\b', 0.5)

# ── Numerical values (percentages, decimals, ratios) ────────────────────────
_p(r'\d+\.?\d*\s*%', 0.5)                          # percentages
_p(r'\b\d+\.\d+\b', 0.3)                            # decimal numbers
_p(r'\b\d{2,}\s*(million|billion|thousand|k|M|B)\b', 0.4)  # large numbers
_p(r'\b\d+\s*(ms|seconds|minutes|hours)\b', 0.4)   # time measurements
_p(r'\b\d+\s*(GB|MB|KB|GiB|MiB)\b', 0.4)          # memory measurements
_p(r'\b\d+x\b', 0.3)                                # speedup factors (e.g., 3x)

# ── Experimental settings ────────────────────────────────────────────────────
_p(r'\b(learning\s+rate|lr)\b', 0.4)
_p(r'\b(batch\s+size|mini[_\s-]batch)\b', 0.4)
_p(r'\b(epoch[s]?|iteration[s]?)\b', 0.3)
_p(r'\b(dropout|weight\s+decay|regularization)\b', 0.3)
_p(r'\b(optimizer|adam|sgd|adagrad|rmsprop)\b', 0.4)
_p(r'\b(training\s+time|inference\s+time|latency)\b', 0.5)
_p(r'\b(gpu|cpu|tpu|cuda|hardware)\b', 0.3)
_p(r'\b(parameter[s]?|weights?)\b', 0.3)
_p(r'\b(flop[s]?|gflop[s]?|tflop[s]?)\b', 0.5)

# ── Dataset and corpus references ────────────────────────────────────────────
_p(r'\b(dataset|corpus|benchmark|collection)\b', 0.4)
_p(r'\b(imagenet|cifar[_-]?\d+|coco|voc|mnist|squad|glue|superglue)\b', 0.6)
_p(r'\b(pubmed|mimic|chexpert|nihcc|nlm)\b', 0.5)
_p(r'\b(wikipedia|bookcorpus|openwebtext|pile)\b', 0.5)
_p(r'\b(train(ing)?|val(idation)?|test)\s+set\b', 0.3)
_p(r'\b\d+[,\d]*\s+(samples?|examples?|instances?|records?|images?|tokens?)\b', 0.4)

# ── Model / architecture names ───────────────────────────────────────────────
_p(r'\b(bert|gpt|t5|roberta|xlnet|albert|deberta|electra)\b', 0.5)
_p(r'\b(transformer|attention|self[_\s-]attention|multi[_\s-]head)\b', 0.4)
_p(r'\b(lstm|gru|rnn|cnn|resnet|vgg|vit|densenet)\b', 0.5)
_p(r'\b(gan|vae|diffusion|stable\s+diffusion|ddpm)\b', 0.5)
_p(r'\b(llm|large\s+language\s+model|foundation\s+model)\b', 0.5)
_p(r'\b(reinforcement\s+learning|rl|ppo|dqn|a3c)\b', 0.5)
_p(r'\b(contrastive|clip|simclr|byol|moco)\b', 0.5)

# ── Limitations / Future work signals ───────────────────────────────────────
_p(r'\b(limitation|limit|constrain|restrict|bottleneck)\b', 0.4)
_p(r'\b(future\s+work|future\s+direction|future\s+research)\b', 0.5)
_p(r'\b(open\s+problem|open\s+question|unresolved|unsolved)\b', 0.4)
_p(r'\b(scalab(le|ility)|generaliz(e|ation|ability)|transfer)\b', 0.4)
_p(r'\b(failure\s+(case|mode)|weakness|shortcoming)\b', 0.4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_technical_importance(sentence: str) -> float:
    """
    Return a technical importance score in [0.0, 1.0] for a sentence.

    The score is proportional to the number and weight of matching patterns.
    Capped at 1.0. A score >= 0.3 generally indicates a sentence worth preserving.
    """
    if not sentence or not sentence.strip():
        return 0.0

    total_weight = 0.0
    for pattern, weight in _PATTERNS:
        if pattern.search(sentence):
            total_weight += weight

    # Normalise: a single strong signal gives ~0.5, multiple give ~1.0
    return min(1.0, total_weight / 2.0)


def is_technical_sentence(sentence: str, threshold: float = 0.25) -> bool:
    """Return True if the sentence is considered technically important."""
    return score_technical_importance(sentence) >= threshold


def filter_technical_sentences(
    sentences: List[Tuple[int, str, float]],
    threshold: float = 0.25,
) -> List[Tuple[int, str, float, float]]:
    """
    Annotate a list of (original_index, text, textrank_score) tuples
    with their technical_score, returning (idx, text, textrank_score, technical_score).
    """
    result = []
    for idx, text, tr_score in sentences:
        tech_score = score_technical_importance(text)
        result.append((idx, text, tr_score, tech_score))
    return result
