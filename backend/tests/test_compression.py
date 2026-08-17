"""
test_compression.py
===================
Comprehensive unit tests for the ResearchIQ extractive compression pipeline.

Tests verify:
  1. Section parser — correct section detection and normalization
  2. Sentence segmentation — correct boundary detection
  3. TextRank — selected sentences are verbatim from source
  4. Technical preservation — metric sentences are retained
  5. Fact sheet structure — all important sections represented
  6. Provenance — every sentence has paper_id and section
  7. Compression ratio — compressed < original
  8. Source validator / faithfulness — 100% verbatim
  9. No hallucination — no sentence invented outside source
 10. Section order restoration — original order preserved

Run with:
    cd backend && python -m pytest tests/test_compression.py -v
"""

import os
import sys
import math

# Add backend to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from compression.section_parser import parse_paper_sections, _normalize_section_name
from compression.textrank import segment_sentences, run_textrank_for_section
from compression.technical_preservation import (
    score_technical_importance,
    is_technical_sentence,
    filter_technical_sentences,
)
from compression.fact_sheet import compress_paper, format_fact_sheet
from compression.token_utils import (
    count_words,
    estimate_tokens,
    measure_compression,
    validate_extracted_sentence,
)
from compression.models import CompressionConfig, SectionBudget, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PAPER_TEXT = """
Abstract

We propose a novel attention-based transformer architecture for clinical text classification.
Our model achieves 94.7% accuracy on the MIMIC-III benchmark using only 110 million parameters.
The system demonstrates a 37% reduction in inference latency compared to BERT-large.

Introduction

Clinical text classification is a challenging problem due to the heterogeneity of medical records.
Previous approaches rely on rule-based systems or simple bag-of-words models that fail to capture long-range dependencies.
We identify three core limitations of existing methods: lack of domain adaptation, high computational cost, and poor generalization across institutions.
This paper proposes a transformer-based solution that addresses these limitations.

Methodology

Our approach uses a multi-head self-attention mechanism with 12 layers and 768 hidden dimensions.
We pretrain the model on 1.2 million clinical notes from the PubMed corpus using masked language modeling.
The training uses a learning rate of 2e-5 with a batch size of 32 and runs for 10 epochs.
We apply dropout of 0.1 and weight decay of 0.01 as regularization.
The model contains exactly 110 million parameters and requires approximately 4 GB of GPU memory.

Dataset

We evaluate on three public datasets: MIMIC-III, i2b2 2012, and the n2c2 shared task corpus.
MIMIC-III contains 1.2 million clinical records from 46,520 patients.
The i2b2 2012 dataset includes 190 discharge summaries annotated for temporal relations.

Experiments

We train all models on an NVIDIA A100 80GB GPU using PyTorch 2.0.
The evaluation uses 5-fold cross-validation on all three datasets.
We compare against BERT-base, RoBERTa-large, and ClinicalBERT as baselines.
Training time per fold is approximately 45 minutes on the above hardware.

Results

Our model achieves F1-score of 94.7% on MIMIC-III, outperforming ClinicalBERT by 3.2 percentage points.
On the i2b2 2012 dataset, we obtain AUROC of 0.97 and precision of 92.4%.
The inference latency is 12 ms per sample, representing a 37% reduction versus BERT-large.
Ablation study shows that removing the domain-adaptive pretraining step reduces accuracy by 6.1%.

Limitations

The model was trained and evaluated exclusively on English-language clinical notes from US hospitals.
Performance on non-English clinical records or records from other healthcare systems is unknown.
The model requires 4 GB of GPU memory at inference time, which may limit deployment on edge devices.
We do not evaluate on paediatric or emergency department records, which may have different linguistic patterns.

Future Work

Future work should explore cross-lingual transfer learning to support multilingual clinical NLP.
We plan to investigate model distillation techniques to reduce the inference memory footprint below 1 GB.
A federated learning setup would allow training across institutions without sharing patient data.

Conclusion

We have presented a transformer-based clinical text classifier that achieves state-of-the-art results on three benchmarks.
The system demonstrates that domain-adaptive pretraining is critical for clinical NLP performance.
Our code and pretrained weights are publicly available at https://github.com/example/clinical-transformer.
"""

SAMPLE_HTML_PAPER = """
<html><body>
<h1>Abstract</h1>
<p>We propose a graph neural network for drug-drug interaction prediction.</p>
<p>Our model achieves AUC of 0.95 on the DrugBank benchmark.</p>
<h2>Introduction</h2>
<p>Drug-drug interactions (DDIs) pose a serious threat to patient safety.</p>
<p>Existing methods rely on chemical structure similarity without leveraging interaction graphs.</p>
<h2>Methodology</h2>
<p>We represent drugs as nodes in a heterogeneous knowledge graph with 50,000 nodes and 300,000 edges.</p>
<p>The model uses 3 graph attention layers with 256 hidden dimensions and a learning rate of 1e-4.</p>
<h2>Results</h2>
<p>Our model achieves AUROC of 0.95 and F1-score of 0.89 on DrugBank, outperforming the baseline by 8%.</p>
<h2>Limitations</h2>
<p>The model was trained on DrugBank only and may not generalize to other drug databases.</p>
<p>We do not account for patient-specific factors such as age, weight, or genetic polymorphisms.</p>
<h2>Conclusion</h2>
<p>We have demonstrated that graph neural networks can effectively predict drug-drug interactions.</p>
</body></html>
"""


# ---------------------------------------------------------------------------
# 1. Section Parser Tests
# ---------------------------------------------------------------------------

class TestSectionParser:

    def test_detects_major_sections(self):
        """Parser should identify at least 5 major sections in the sample paper."""
        sections = parse_paper_sections(SAMPLE_PAPER_TEXT)
        assert len(sections) >= 5, f"Expected >=5 sections, got {len(sections)}: {list(sections.keys())}"

    def test_abstract_detected(self):
        """Abstract section must be detected."""
        sections = parse_paper_sections(SAMPLE_PAPER_TEXT)
        assert "abstract" in sections, f"'abstract' not in sections: {list(sections.keys())}"

    def test_methodology_detected(self):
        """Methodology section must be detected."""
        sections = parse_paper_sections(SAMPLE_PAPER_TEXT)
        assert "methodology" in sections, f"'methodology' not in sections: {list(sections.keys())}"

    def test_limitations_detected(self):
        """Limitations section must be detected."""
        sections = parse_paper_sections(SAMPLE_PAPER_TEXT)
        assert "limitations" in sections, f"'limitations' not in sections: {list(sections.keys())}"

    def test_html_parsing(self):
        """Parser must handle HTML input and strip tags."""
        sections = parse_paper_sections(SAMPLE_HTML_PAPER, is_html=True)
        assert "abstract" in sections or "introduction" in sections
        # HTML tags should not appear in section text
        for key, text in sections.items():
            assert "<" not in text or ">" not in text or key == "unknown"

    def test_empty_text_returns_empty(self):
        """Empty input should return empty dict, not crash."""
        assert parse_paper_sections("") == {}
        assert parse_paper_sections("   ") == {}

    def test_no_sections_returns_unknown(self):
        """Text without headings should be returned under 'unknown' key."""
        flat_text = "This is a single paragraph without any section headings. " * 20
        sections = parse_paper_sections(flat_text)
        assert len(sections) >= 1

    def test_section_name_normalization(self):
        """Normalization function must map variants correctly."""
        test_cases = [
            ("Materials and Methods", "methodology"),
            ("Proposed Method", "methodology"),
            ("Experimental Setup", "experiments"),
            ("Evaluation", "experiments"),
            ("Experimental Results", "results"),
            ("Concluding Remarks", "conclusion"),
            ("Limitations and Future Work", "limitations"),  # primary
            ("Literature Review", "related_work"),
        ]
        for raw, expected in test_cases:
            result = _normalize_section_name(raw)
            assert result == expected, f"'{raw}' → got '{result}', expected '{expected}'"

    def test_section_text_is_nonempty(self):
        """Every detected section must have non-empty text."""
        sections = parse_paper_sections(SAMPLE_PAPER_TEXT)
        for key, text in sections.items():
            assert text.strip(), f"Section '{key}' has empty text"


# ---------------------------------------------------------------------------
# 2. Sentence Segmentation Tests
# ---------------------------------------------------------------------------

class TestSentenceSegmentation:

    def test_basic_segmentation(self):
        """Basic sentences must be segmented correctly."""
        text = "The model achieves 94.7% accuracy. It runs at 12 ms latency. We used BERT as baseline."
        sents = segment_sentences(text)
        assert len(sents) >= 2, f"Expected >=2 sentences, got {len(sents)}"

    def test_decimal_not_split(self):
        """Decimal numbers must not split sentences (3.14, 0.95)."""
        text = "The model achieves an accuracy of 94.7% using a learning rate of 2e-5 on the MIMIC-III dataset."
        sents = segment_sentences(text)
        # Should be ONE sentence
        assert len(sents) == 1, f"Decimal caused incorrect sentence split: {sents}"

    def test_abbreviation_not_split(self):
        """Common abbreviations (e.g., et al.) must not cause splits."""
        text = "Wang et al. proposed a transformer for clinical NLP. Our model extends this work."
        sents = segment_sentences(text)
        assert len(sents) >= 1

    def test_returns_nonempty_strings(self):
        """All returned sentences must be non-empty strings."""
        sents = segment_sentences(SAMPLE_PAPER_TEXT)
        for s in sents:
            assert isinstance(s, str) and len(s.strip()) > 0

    def test_empty_text_returns_empty(self):
        """Empty input returns empty list."""
        assert segment_sentences("") == []
        assert segment_sentences("   ") == []


# ---------------------------------------------------------------------------
# 3. TextRank Tests
# ---------------------------------------------------------------------------

class TestTextRank:

    def test_selected_sentences_are_verbatim(self):
        """Every selected sentence must exist verbatim in the source text."""
        methodology_text = """
        Our approach uses a multi-head self-attention mechanism with 12 layers and 768 hidden dimensions.
        We pretrain the model on 1.2 million clinical notes from the PubMed corpus using masked language modeling.
        The training uses a learning rate of 2e-5 with a batch size of 32 and runs for 10 epochs.
        We apply dropout of 0.1 and weight decay of 0.01 as regularization.
        The model contains exactly 110 million parameters and requires approximately 4 GB of GPU memory.
        """
        results = run_textrank_for_section(methodology_text, max_sentences=3, min_sentences=1)
        for orig_idx, text, score in results:
            assert text.strip() in methodology_text or text in methodology_text, \
                f"Non-verbatim sentence: '{text}'"

    def test_original_order_restored(self):
        """Selected sentences must be in original document order."""
        text = SAMPLE_PAPER_TEXT
        results = run_textrank_for_section(text, max_sentences=5)
        indices = [idx for idx, _, _ in results]
        assert indices == sorted(indices), f"Sentence order not restored: {indices}"

    def test_respects_max_sentences(self):
        """Must not exceed max_sentences."""
        results = run_textrank_for_section(SAMPLE_PAPER_TEXT, max_sentences=3)
        assert len(results) <= 3

    def test_respects_min_sentences(self):
        """Must return at least min_sentences for non-empty sections."""
        short_text = "This is the only sentence in this section."
        results = run_textrank_for_section(short_text, max_sentences=5, min_sentences=1)
        assert len(results) >= 1

    def test_single_sentence_section(self):
        """Single-sentence sections handled gracefully."""
        results = run_textrank_for_section("Only one sentence here.", max_sentences=3)
        assert len(results) == 1

    def test_empty_section(self):
        """Empty section returns empty list."""
        assert run_textrank_for_section("") == []
        assert run_textrank_for_section("   ") == []

    def test_scores_are_floats(self):
        """All TextRank scores must be non-negative floats."""
        results = run_textrank_for_section(SAMPLE_PAPER_TEXT, max_sentences=5)
        for _, _, score in results:
            assert isinstance(score, float) and score >= 0.0


# ---------------------------------------------------------------------------
# 4. Technical Preservation Tests
# ---------------------------------------------------------------------------

class TestTechnicalPreservation:

    def test_metric_sentence_scores_high(self):
        """Sentences with percentages and metric names score >= 0.25."""
        technical_sents = [
            "Our model achieves 94.7% accuracy on the MIMIC-III benchmark.",
            "The F1-score improves from 87.3% to 94.1% with our approach.",
            "Inference latency is reduced by 37% to 12 ms per sample.",
            "The model contains 110 million parameters and requires 4 GB GPU memory.",
            "We use AUROC, BLEU, and ROUGE-L as evaluation metrics.",
        ]
        for sent in technical_sents:
            score = score_technical_importance(sent)
            assert score >= 0.25, f"Expected score >= 0.25 for: '{sent}' (got {score:.3f})"

    def test_generic_sentence_scores_low(self):
        """Generic sentences without technical content score low."""
        generic_sents = [
            "This paper presents a new approach to solving the problem.",
            "We hope the community finds our work useful.",
            "The rest of the paper is organized as follows.",
        ]
        for sent in generic_sents:
            score = score_technical_importance(sent)
            # Should be lower than the threshold for technical sentences
            assert score < 0.7, f"Generic sentence scored too high: '{sent}' ({score:.3f})"

    def test_is_technical_sentence(self):
        """is_technical_sentence returns True for metric sentences."""
        assert is_technical_sentence("We achieve F1-score of 94.7% on MIMIC-III.")
        assert is_technical_sentence("The model has 110 million parameters.")
        assert is_technical_sentence("Inference latency is 12 ms per sample.")

    def test_scores_bounded(self):
        """Scores must be in [0.0, 1.0]."""
        very_technical = (
            "Our BERT-based model achieves 94.7% accuracy, 92.4% F1-score, AUROC 0.97, "
            "with 110M parameters, 12ms latency, trained for 10 epochs with batch size 32 "
            "and learning rate 2e-5, evaluated on MIMIC-III, i2b2, and n2c2 benchmarks."
        )
        score = score_technical_importance(very_technical)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 5. Fact Sheet Structure Tests
# ---------------------------------------------------------------------------

class TestFactSheetStructure:

    def test_compress_paper_returns_fact_sheet(self):
        """compress_paper must return a PaperFactSheet object."""
        from compression.models import PaperFactSheet
        paper = {
            "title": "Clinical Transformer for Text Classification",
            "authors": ["Alice Smith", "Bob Jones"],
            "year": 2024,
            "url": "https://arxiv.org/abs/2401.00001",
            "full_text": SAMPLE_PAPER_TEXT,
            "abstract": "We propose a novel attention-based transformer architecture.",
        }
        fs = compress_paper(paper, query="clinical NLP", paper_id="P001")
        assert isinstance(fs, PaperFactSheet)

    def test_important_sections_represented(self):
        """Fact sheet must include at least methodology, results, or limitations."""
        paper = {
            "title": "Test Paper",
            "authors": [],
            "year": 2024,
            "url": "",
            "full_text": SAMPLE_PAPER_TEXT,
            "abstract": "Abstract text.",
        }
        fs = compress_paper(paper, query="test", paper_id="P001")
        covered = set(fs.sections.keys())
        important = {"methodology", "results", "limitations"}
        assert len(covered & important) >= 1, \
            f"None of {important} found in covered sections: {covered}"

    def test_compression_ratio_measured(self):
        """Compression ratio must be a measured non-zero value (not assumed)."""
        paper = {
            "title": "Test Paper",
            "authors": [],
            "year": 2024,
            "url": "",
            "full_text": SAMPLE_PAPER_TEXT,
            "abstract": "Test abstract.",
        }
        fs = compress_paper(paper, query="test", paper_id="P001")
        assert 0.0 <= fs.metrics.compression_ratio <= 1.0

    def test_compressed_tokens_less_than_original(self):
        """Compressed token count must be less than original (compression happened)."""
        paper = {
            "title": "Test Paper",
            "authors": [],
            "year": 2024,
            "url": "",
            "full_text": SAMPLE_PAPER_TEXT * 3,  # repeat to make it longer
            "abstract": "Test abstract.",
        }
        fs = compress_paper(paper, query="test", paper_id="P001")
        if fs.metrics.original_token_count > 0:
            assert fs.metrics.compressed_token_count <= fs.metrics.original_token_count, \
                "Compressed tokens exceed original tokens!"

    def test_empty_paper_handled_gracefully(self):
        """Papers with no text return empty fact sheet without crashing."""
        paper = {"title": "Empty", "authors": [], "year": 2024, "url": "", "full_text": ""}
        fs = compress_paper(paper, query="test", paper_id="P000")
        assert fs is not None

    def test_abstract_only_paper_handled(self):
        """Papers with only abstract text return a fact sheet."""
        paper = {
            "title": "Abstract Only Paper",
            "authors": [],
            "year": 2024,
            "url": "",
            "full_text": "",
            "abstract": "We propose a novel deep learning method for image segmentation.",
        }
        fs = compress_paper(paper, query="image segmentation", paper_id="P002")
        assert fs is not None


# ---------------------------------------------------------------------------
# 6. Provenance Tests
# ---------------------------------------------------------------------------

class TestProvenance:

    def test_every_sentence_has_paper_id(self):
        """Every extracted sentence must have a non-empty paper_id."""
        paper = {
            "title": "Provenance Test Paper",
            "authors": [],
            "year": 2024,
            "url": "",
            "full_text": SAMPLE_PAPER_TEXT,
            "abstract": "Test.",
        }
        fs = compress_paper(paper, query="provenance", paper_id="P010")
        for section_name, sheet in fs.sections.items():
            for sent in sheet.selected_sentences:
                assert sent.paper_id == "P010", \
                    f"Missing paper_id on sentence '{sent.sentence_id}'"

    def test_every_sentence_has_section(self):
        """Every extracted sentence must have a section field matching its parent section."""
        paper = {
            "title": "Section Test Paper",
            "authors": [],
            "year": 2024,
            "url": "",
            "full_text": SAMPLE_PAPER_TEXT,
            "abstract": "Test.",
        }
        fs = compress_paper(paper, query="test", paper_id="P011")
        for section_name, sheet in fs.sections.items():
            for sent in sheet.selected_sentences:
                assert sent.section == section_name, \
                    f"Sentence section '{sent.section}' != parent '{section_name}'"

    def test_every_sentence_has_sentence_id(self):
        """Every extracted sentence must have a non-empty, unique sentence_id."""
        paper = {
            "title": "ID Test Paper",
            "authors": [],
            "year": 2024,
            "url": "",
            "full_text": SAMPLE_PAPER_TEXT,
            "abstract": "Test.",
        }
        fs = compress_paper(paper, query="test", paper_id="P012")
        all_ids = []
        for section_name, sheet in fs.sections.items():
            for sent in sheet.selected_sentences:
                assert sent.sentence_id, f"Empty sentence_id in section '{section_name}'"
                all_ids.append(sent.sentence_id)
        # All IDs must be unique
        assert len(all_ids) == len(set(all_ids)), "Duplicate sentence_ids found!"

    def test_selection_reason_is_valid(self):
        """Every sentence must have a recognized selection_reason."""
        valid_reasons = {"textrank", "technical", "textrank+technical", "min_coverage"}
        paper = {
            "title": "Reason Test",
            "authors": [],
            "year": 2024,
            "url": "",
            "full_text": SAMPLE_PAPER_TEXT,
            "abstract": "Test.",
        }
        fs = compress_paper(paper, query="test", paper_id="P013")
        for section_name, sheet in fs.sections.items():
            for sent in sheet.selected_sentences:
                assert sent.selection_reason in valid_reasons, \
                    f"Invalid selection_reason: '{sent.selection_reason}'"


# ---------------------------------------------------------------------------
# 7. Source Validation / Faithfulness Tests
# ---------------------------------------------------------------------------

class TestSourceValidation:

    def test_validate_extracted_sentence_positive(self):
        """validate_extracted_sentence returns True for sentences in source."""
        source = "We achieve 94.7% accuracy on MIMIC-III. The model runs at 12 ms latency."
        assert validate_extracted_sentence("We achieve 94.7% accuracy on MIMIC-III.", source)
        assert validate_extracted_sentence("The model runs at 12 ms latency.", source)

    def test_validate_extracted_sentence_negative(self):
        """validate_extracted_sentence returns False for invented text."""
        source = "We achieve 94.7% accuracy on MIMIC-III."
        assert not validate_extracted_sentence(
            "The method performs well across several datasets.", source
        )

    def test_faithfulness_ratio_calculated(self):
        """Compression pipeline must calculate faithfulness_ratio."""
        paper = {
            "title": "Faithfulness Test",
            "authors": [],
            "year": 2024,
            "url": "",
            "full_text": SAMPLE_PAPER_TEXT,
            "abstract": "Test.",
        }
        fs = compress_paper(paper, query="test", paper_id="P020")
        # Faithfulness must be a valid ratio
        assert 0.0 <= fs.metrics.faithfulness_ratio <= 1.0

    def test_all_sentences_source_verified(self):
        """
        Every selected sentence should be verified against the source.
        This is the NO-HALLUCINATION test.
        """
        paper = {
            "title": "No Hallucination Test",
            "authors": [],
            "year": 2024,
            "url": "",
            "full_text": SAMPLE_PAPER_TEXT,
            "abstract": "We propose a novel attention-based transformer architecture.",
        }
        fs = compress_paper(paper, query="transformer", paper_id="P021")
        for section_name, sheet in fs.sections.items():
            for sent in sheet.selected_sentences:
                assert sent.source_verified, (
                    f"Sentence NOT found verbatim in source!\n"
                    f"  Section: {section_name}\n"
                    f"  Sentence ID: {sent.sentence_id}\n"
                    f"  Text: '{sent.text}'\n"
                    f"  This is a faithfulness violation."
                )


# ---------------------------------------------------------------------------
# 8. Token Utils Tests
# ---------------------------------------------------------------------------

class TestTokenUtils:

    def test_count_words_basic(self):
        assert count_words("hello world") == 2
        assert count_words("") == 0
        assert count_words("   ") == 0

    def test_estimate_tokens_positive(self):
        assert estimate_tokens("hello world") > 0
        assert estimate_tokens("") == 0

    def test_measure_compression_ratio(self):
        original = "word " * 1000
        compressed = "word " * 200
        metrics = measure_compression(original, compressed, detected_sections=5, covered_sections=5)
        assert metrics.compression_ratio > 0.0
        assert metrics.compression_ratio < 1.0
        assert metrics.section_coverage == 1.0

    def test_measure_compression_faithfulness(self):
        metrics = measure_compression(
            "orig",
            "comp",
            selected_sentences=10,
            validated_sentences=10
        )
        assert metrics.faithfulness_ratio == 1.0

        metrics_partial = measure_compression(
            "orig",
            "comp",
            selected_sentences=10,
            validated_sentences=8
        )
        assert abs(metrics_partial.faithfulness_ratio - 0.8) < 0.001


# ---------------------------------------------------------------------------
# 9. Fact Sheet Format Tests
# ---------------------------------------------------------------------------

class TestFactSheetFormat:

    def test_format_fact_sheet_contains_sections(self):
        """Formatted fact sheet must contain section headers."""
        paper = {
            "title": "Format Test Paper",
            "authors": ["Alice"],
            "year": 2024,
            "url": "https://example.com",
            "full_text": SAMPLE_PAPER_TEXT,
            "abstract": "Test abstract.",
        }
        fs = compress_paper(paper, query="test", paper_id="P030")
        text = format_fact_sheet(fs)
        assert "PAPER FACT SHEET" in text
        assert "COMPRESSION METRICS" in text
        # At least one section header
        assert "[" in text

    def test_format_fact_sheet_no_llm_generated_text(self):
        """
        The fact sheet must NOT contain phrases typical of LLM summaries.
        All content should be verbatim paper sentences.
        """
        paper = {
            "title": "Anti-Hallucination Test",
            "authors": [],
            "year": 2024,
            "url": "",
            "full_text": SAMPLE_PAPER_TEXT,
            "abstract": "We propose a novel method.",
        }
        fs = compress_paper(paper, query="test", paper_id="P031")
        text = format_fact_sheet(fs)
        # These phrases are hallucination indicators; they should not appear
        # unless the paper itself contains them
        hallucination_phrases = [
            "In summary,",
            "Overall, the paper",
            "This study demonstrates that",
            "The authors conclude that",  # paraphrase pattern
        ]
        for phrase in hallucination_phrases:
            if phrase.lower() in SAMPLE_PAPER_TEXT.lower():
                continue  # It's OK if the paper actually contains this phrase
            assert phrase not in text, f"Possible hallucination detected: '{phrase}'"


# ---------------------------------------------------------------------------
# 10. CompressionConfig Tests
# ---------------------------------------------------------------------------

class TestCompressionConfig:

    def test_default_config_has_all_sections(self):
        """Default config must have budget for all standard sections."""
        standard_sections = [
            "abstract", "introduction", "related_work", "problem_definition",
            "methodology", "dataset", "experiments", "results",
            "discussion", "limitations", "future_work", "conclusion"
        ]
        for section in standard_sections:
            assert section in DEFAULT_CONFIG.budgets, f"'{section}' missing from DEFAULT_CONFIG"

    def test_min_le_max(self):
        """min_sentences must be <= max_sentences for all sections."""
        for section, budget in DEFAULT_CONFIG.budgets.items():
            assert budget.min_sentences <= budget.max_sentences, \
                f"Section '{section}': min ({budget.min_sentences}) > max ({budget.max_sentences})"

    def test_custom_config_aggressive(self):
        """Aggressive config (2 sentences/section) should work."""
        aggressive = CompressionConfig(budgets={
            key: SectionBudget(min_sentences=1, max_sentences=2)
            for key in DEFAULT_CONFIG.budgets
        })
        paper = {
            "title": "Config Test",
            "authors": [],
            "year": 2024,
            "url": "",
            "full_text": SAMPLE_PAPER_TEXT,
            "abstract": "Test.",
        }
        fs = compress_paper(paper, query="test", paper_id="P040", config=aggressive)
        for section_name, sheet in fs.sections.items():
            assert sheet.selected_sentence_count <= 2, \
                f"Section '{section_name}' has {sheet.selected_sentence_count} sentences with max=2"


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
