"""
test_evidence_preservation_gaps.py

Phase 5 Test Suite:
- Evidence Provenance: Every selected sentence retains paper_id, section, sentence_id, source_verified.
- Sentence IDs: Traceable sentence IDs (e.g. P001-METH-01, P001-LIM-01) in fact sheets and gap objects.
- Full-Paper Eligibility: Only FULL_TEXT_AVAILABLE / full_paper / partial_paper papers contribute evidence to gaps.
- Unsupported Gap Prevention: Unavailable papers receive static non-evidence markers and empty supporting_paper_ids/sentence_ids.
- Cross-Paper Evidence: Cross-paper gaps only reference eligible paper IDs.
- Faithfulness: TextRank produces ~100% extractive sentences verifiable in source.
- Technical Evidence Preservation: Methodology, datasets, models, experiments, results, limitations preserved in fact sheets.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import PaperMetadata, Agent1ResearchOutput, ResearchGap, Agent2GapOutput
from compression.models import (
    PaperFactSheet,
    SectionFactSheet,
    ExtractedSentence,
    CompressionConfig,
)
from compression.fact_sheet import compress_paper, format_fact_sheet
from agents.agent_gap import (
    is_evidence_eligible,
    cluster_and_analyze_gaps,
)


class TestEvidenceProvenanceAndSentenceIDs:
    """Every extracted sentence retains paper_id, section, sentence_id, and source_verified."""

    def test_sentence_provenance_metadata(self):
        text = (
            "1. Introduction\n\n"
            "Deep learning neural networks have transformed automated voice synthesis and audio generation.\n\n"
            "2. Methodology\n\n"
            "We deploy a dual-branch convolutional architecture with residual connections for multi-spectral analysis.\n\n"
            "3. Results\n\n"
            "The model achieves an equal error rate (EER) of 0.82% on the ASVspoof 2021 evaluation partition.\n\n"
            "4. Limitations\n\n"
            "However, computational complexity is high and the model suffers performance degradation under low-bitrate codecs."
        )
        paper_dict = {
            "title": "Dual-Branch Audio Architecture",
            "authors": ["Test Researcher"],
            "year": 2024,
            "url": "https://example.com/audio",
            "full_text": text,
            "coverage_type": "full_paper",
        }
        fs = compress_paper(paper_dict, paper_id="P001")
        assert isinstance(fs, PaperFactSheet)
        assert fs.metrics.faithfulness_ratio == 1.0

        for sec_name, sec_sheet in fs.sections.items():
            assert isinstance(sec_sheet, SectionFactSheet)
            for sent in sec_sheet.selected_sentences:
                assert isinstance(sent, ExtractedSentence)
                assert sent.paper_id == "P001"
                assert sent.section == sec_name
                assert sent.sentence_id.startswith("P001-")
                assert sent.source_verified is True
                assert len(sent.text.strip()) > 10

    def test_formatted_fact_sheet_displays_sentence_ids(self):
        text = (
            "1. Methodology\n\n"
            "We construct a Transformer-based encoder with positional embeddings to capture temporal cues.\n\n"
            "2. Limitations\n\n"
            "The primary drawback is high memory footprint during multi-channel audio processing."
        )
        paper_dict = {
            "title": "Transformer Audio",
            "authors": ["A. Author"],
            "year": 2024,
            "url": "https://example.com/tf",
            "full_text": text,
            "coverage_type": "full_paper",
        }
        fs = compress_paper(paper_dict, paper_id="P002")
        formatted = format_fact_sheet(fs)
        assert "P002-METH-" in formatted or "P002-LIM-" in formatted
        assert "Transformer-based encoder" in formatted
        assert "primary drawback is high memory footprint" in formatted


class TestFullPaperEligibilityAndUnsupportedGapPrevention:
    """Ensure unavailable papers never support gaps or have fabricated claims."""

    def test_is_evidence_eligible_rules(self):
        # Eligible cases
        p_full = PaperMetadata(
            title="Full Paper",
            authors=["A"],
            year=2024,
            abstract="Abs",
            relevance_rank="High",
            notebook_summary="Sum",
            technical_execution="Tech",
            datasets=[],
            url="http://a",
            relevance_score=0.9,
            access_status="FULL_TEXT_AVAILABLE",
            coverage_type="full_paper",
        )
        assert is_evidence_eligible(p_full) is True

        p_partial = PaperMetadata(
            title="Partial Paper",
            authors=["B"],
            year=2024,
            abstract="Abs",
            relevance_rank="High",
            notebook_summary="Sum",
            technical_execution="Tech",
            datasets=[],
            url="http://b",
            relevance_score=0.8,
            access_status="FULL_TEXT_AVAILABLE",
            coverage_type="partial_paper",
        )
        assert is_evidence_eligible(p_partial) is True

        # Ineligible cases
        p_abstract_only = PaperMetadata(
            title="Abstract Only",
            authors=["C"],
            year=2024,
            abstract="Abs",
            relevance_rank="Medium",
            notebook_summary="Sum",
            technical_execution="Tech",
            datasets=[],
            url="http://c",
            relevance_score=0.7,
            access_status="ABSTRACT_ONLY",
            coverage_type="abstract_only",
        )
        assert is_evidence_eligible(p_abstract_only) is False

        p_unavailable = PaperMetadata(
            title="Unavailable Paper",
            authors=["D"],
            year=2024,
            abstract="",
            relevance_rank="Low",
            notebook_summary="",
            technical_execution="",
            datasets=[],
            url="http://d",
            relevance_score=0.0,
            access_status="UNAVAILABLE",
            coverage_type="unavailable",
        )
        assert is_evidence_eligible(p_unavailable) is False

    def test_unavailable_papers_produce_static_gaps_without_supporting_ids(self):
        """When unavailable papers are evaluated in gap analysis, they must not cite evidence."""
        p_unavail = PaperMetadata(
            title="Unavailable Paywalled Citation",
            authors=["E"],
            year=2023,
            abstract="",
            relevance_rank="Low",
            notebook_summary="",
            technical_execution="",
            datasets=[],
            url="http://e",
            relevance_score=0.1,
            access_status="UNAVAILABLE",
            coverage_type="unavailable",
        )
        research_out = Agent1ResearchOutput(
            query="Deepfake audio detection",
            papers=[p_unavail],
        )
        gap_out = cluster_and_analyze_gaps(research_out)
        assert isinstance(gap_out, Agent2GapOutput)
        assert len(gap_out.gaps) >= 1
        gap = gap_out.gaps[0]
        assert "Evidence unavailable" in gap.description or "Evidence unavailable" in gap.gap_statement
        # Unavailable papers must have NO supporting IDs or fabricated claims
        assert gap.supporting_paper_ids == []
        assert gap.supporting_sentence_ids == []


class TestCrossPaperEvidenceAndTraceability:
    """Gaps generated from eligible papers contain traceable sentence IDs and valid schema fields."""

    def test_gap_provenance_schema_fields(self):
        text_p1 = (
            "1. Methodology\n\n"
            "We extract Mel-frequency cepstral coefficients (MFCC) and train a ResNet-34 classifier.\n\n"
            "2. Limitations\n\n"
            "The model suffers substantial performance drop when tested on unseen acoustic environments."
        )
        text_p2 = (
            "1. Methodology\n\n"
            "We utilize wav2vec 2.0 self-supervised representations with linear projection heads.\n\n"
            "2. Limitations\n\n"
            "Memory constraints restrict fine-tuning to small batch sizes of 8 samples."
        )

        p1_fs = compress_paper({"title": "MFCC ResNet", "authors": ["A"], "year": 2024, "url": "http://1", "full_text": text_p1, "coverage_type": "full_paper"}, paper_id="P001")
        p2_fs = compress_paper({"title": "Wav2Vec Model", "authors": ["B"], "year": 2024, "url": "http://2", "full_text": text_p2, "coverage_type": "full_paper"}, paper_id="P002")

        p1 = PaperMetadata(
            paper_id="P001",
            title="MFCC ResNet Detection",
            authors=["A"],
            year=2024,
            abstract="Abstract 1",
            relevance_rank="High",
            notebook_summary="Summary 1",
            technical_execution="Execution 1",
            datasets=["ASVspoof"],
            url="http://1",
            relevance_score=0.95,
            access_status="FULL_TEXT_AVAILABLE",
            coverage_type="full_paper",
            fact_sheet=p1_fs,
            fact_sheet_text=format_fact_sheet(p1_fs),
        )
        p2 = PaperMetadata(
            paper_id="P002",
            title="Wav2Vec Audio Detection",
            authors=["B"],
            year=2024,
            abstract="Abstract 2",
            relevance_rank="High",
            notebook_summary="Summary 2",
            technical_execution="Execution 2",
            datasets=["In-the-Wild"],
            url="http://2",
            relevance_score=0.92,
            access_status="FULL_TEXT_AVAILABLE",
            coverage_type="full_paper",
            fact_sheet=p2_fs,
            fact_sheet_text=format_fact_sheet(p2_fs),
        )

        research_out = Agent1ResearchOutput(
            query="Deepfake audio detection",
            papers=[p1, p2],
        )

        gap_out = cluster_and_analyze_gaps(research_out)
        assert isinstance(gap_out, Agent2GapOutput)
        assert len(gap_out.gaps) >= 2

        for gap in gap_out.gaps:
            assert isinstance(gap, ResearchGap)
            assert gap.gap_id.startswith("GAP-")
            assert len(gap.gap_statement) > 0
            assert gap.severity in ("Critical", "Moderate", "Minor", "Low")
            assert len(gap.supporting_paper_ids) > 0
            # All supporting paper IDs must be eligible
            for pid in gap.supporting_paper_ids:
                assert pid in ("P001", "P002")


class TestExtractiveFaithfulnessAndTechnicalPreservation:
    """Verify 100% extractive faithfulness and technical sections presence."""

    def test_technical_sections_retained_in_fact_sheet(self):
        text = (
            "1. Abstract\n\n"
            "This study evaluates synthetic audio detection using contrastive learning representations.\n\n"
            "2. Methodology\n\n"
            "Our model integrates a Conformer backbone with dynamic multi-scale convolutional kernels.\n\n"
            "3. Dataset\n\n"
            "We train on 50,000 synthetic samples from the WaveFake benchmark and LibriSeVoc corpus.\n\n"
            "4. Experiments\n\n"
            "We run 100 epochs using AdamW optimizer with initial learning rate of 1e-4 and cosine warmup.\n\n"
            "5. Results\n\n"
            "The proposed method achieves 99.4% accuracy with an F1 score of 0.992 across test sets.\n\n"
            "6. Limitations\n\n"
            "The computational footprint limits real-time inference on mobile DSP chips."
        )
        paper_dict = {
            "title": "Conformer Contrastive Audio",
            "authors": ["C. Scientist"],
            "year": 2024,
            "url": "https://example.com/conformer",
            "full_text": text,
            "coverage_type": "full_paper",
        }
        fs = compress_paper(paper_dict, paper_id="P003")
        assert fs.metrics.faithfulness_ratio == 1.0
        assert fs.metrics.technical_sentence_count >= 3
        # Check that core technical sections were detected and retained
        for sec in ["methodology", "dataset", "results", "limitations"]:
            assert sec in fs.sections
            assert len(fs.sections[sec].selected_sentences) >= 1
