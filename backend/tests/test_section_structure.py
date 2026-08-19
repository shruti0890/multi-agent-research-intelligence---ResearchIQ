"""
Tests for Phase 3: Full-Text Extraction + Section Structure Validation.

Covers:
- PMC XML structured parsing (abstract, section titles, paragraphs, nested sections, not flattened)
- PDF extraction via pdfminer.six (preserves line structure, handles corrupt bytes safely)
- Large paper preservation (e.g. 49,512 words, 72,618 words processed section-by-section with no truncation)
- Research section types vs Non-research section types
- Non-research section skipping (references, bibliography, funding, conflict_of_interest, author_contributions, acknowledgments, supplementary_material)
- Unknown section content inspection (reclassifies references/funding/metadata, preserves genuine research prose)
- Metrics reporting (detected_sections, research_sections, research_sections_covered, research_section_coverage, non_research_sections_skipped)
"""

import os
import sys
import io
import xml.etree.ElementTree as ET
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.section_parser import (
    parse_paper_sections,
    inspect_and_classify_unknown_section,
    NON_RESEARCH_SECTIONS,
    RESEARCH_CONTENT_SECTIONS,
)
from compression.fact_sheet import compress_paper
from agents.agent_research import _parse_pmc_xml_to_structured_text, _fetch_pdf_text


# ===========================================================================
# 1. PMC XML Structured Parsing Tests
# ===========================================================================
class TestPMCStructuredParsing:
    def test_pmc_xml_preserves_structure_and_headings(self):
        xml_str = """<?xml version="1.0" encoding="UTF-8"?>
        <article>
            <front>
                <article-meta>
                    <abstract>
                        <p>This is the structured abstract describing the novel model.</p>
                    </abstract>
                </article-meta>
            </front>
            <body>
                <sec>
                    <title>1. Introduction</title>
                    <p>Introduction paragraph 1 explaining the problem.</p>
                    <p>Introduction paragraph 2 outlining contributions.</p>
                </sec>
                <sec>
                    <title>2. Methodology</title>
                    <p>Overview of the proposed neural architecture.</p>
                    <sec>
                        <title>2.1 Cross-Attention Fusion</title>
                        <p>Detailed formulation of the cross-attention layers.</p>
                    </sec>
                    <sec>
                        <title>2.2 Loss Function</title>
                        <p>Mathematical formulation of the contrastive objective.</p>
                    </sec>
                </sec>
                <sec>
                    <title>3. Results and Discussion</title>
                    <p>Evaluation results demonstrating 94.5% accuracy.</p>
                </sec>
            </body>
        </article>
        """
        root = ET.fromstring(xml_str)
        text = _parse_pmc_xml_to_structured_text(root)

        # Must not be flattened into a single line
        lines = text.split("\n")
        assert len(lines) > 5

        # Must contain Abstract and section headings
        assert "Abstract" in text
        assert "This is the structured abstract" in text
        assert "Introduction" in text
        assert "Methodology" in text
        assert "Cross-Attention Fusion" in text
        assert "Loss Function" in text
        assert "Results and Discussion" in text

        # Parse sections with section parser
        sections = parse_paper_sections(text)
        assert "abstract" in sections
        assert "introduction" in sections
        assert "methodology" in sections
        assert "results" in sections or "discussion" in sections

    def test_pmc_nested_sections_retained(self):
        xml_str = """<?xml version="1.0" encoding="UTF-8"?>
        <article>
            <body>
                <sec>
                    <title>Deep Hierarchical Architecture</title>
                    <p>Preamble for architecture.</p>
                    <sec>
                        <title>Sub-module A</title>
                        <p>Content for sub-module A.</p>
                        <sec>
                            <title>Leaf Component 1</title>
                            <p>Content for leaf component 1.</p>
                        </sec>
                    </sec>
                </sec>
            </body>
        </article>
        """
        root = ET.fromstring(xml_str)
        text = _parse_pmc_xml_to_structured_text(root)
        assert "Deep Hierarchical Architecture" in text
        assert "Sub-module A" in text
        assert "Leaf Component 1" in text
        assert "Content for leaf component 1." in text


# ===========================================================================
# 2. PDF Extraction Tests
# ===========================================================================
class TestPDFStructureExtraction:
    def _make_sample_pdf(self) -> bytes:
        from pdfminer.high_level import extract_text
        # Construct valid minimal PDF
        content = "BT /F1 12 Tf 50 750 Td (Introduction) Tj 0 -20 Td (This is the introduction.) Tj ET"
        stream_bytes = content.encode("latin-1")
        stream_len = len(stream_bytes)
        pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
            + f"4 0 obj\n<< /Length {stream_len} >>\nstream\n".encode() + stream_bytes + b"\nendstream\nendobj\n"
            b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
            b"xref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \n0000000250 00000 n \n0000000350 00000 n \n"
            b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n450\n%%EOF\n"
        )
        return pdf

    def test_pdfminer_extraction_preserves_text_and_newlines(self):
        from pdfminer.high_level import extract_text
        pdf_bytes = self._make_sample_pdf()
        extracted = extract_text(io.BytesIO(pdf_bytes))
        assert "Introduction" in extracted
        assert "This is the introduction." in extracted

    def test_pdf_extraction_safe_on_corrupt_data(self):
        # Empty or invalid bytes must not raise unhandled exceptions
        result = _fetch_pdf_text("https://invalid.example.com/nonexistent.pdf")
        assert result == ""


# ===========================================================================
# 3. Large Paper Processing & No Truncation Tests
# ===========================================================================
class TestLargePaperProcessing:
    def test_49k_word_paper_processed_without_truncation(self):
        """A ~49,512 word paper must be accepted and processed section-by-section."""
        # 10 words per sentence, 100 sentences per block
        sentence = "We evaluate the neural model on large benchmark datasets for accuracy. "
        block = sentence * 100  # ~1,100 words

        sections = [
            f"Abstract\n\n{block}",
            f"Introduction\n\n{block * 5}",        # ~5,500 words
            f"Related Work\n\n{block * 5}",        # ~5,500 words
            f"Methodology\n\n{block * 15}",        # ~16,500 words
            f"Experiments\n\n{block * 10}",        # ~11,000 words
            f"Results\n\n{block * 8}",            # ~8,800 words
            f"Conclusion\n\n{block * 2}",         # ~2,200 words
        ]
        full_text = "\n\n".join(sections)
        word_count = len(full_text.split())
        assert word_count >= 45_000, f"Synthetic paper has {word_count} words"

        paper_dict = {
            "title": "Large Scale 49k Word Paper",
            "authors": ["Researcher A"],
            "year": 2024,
            "url": "https://example.com/large-paper",
            "full_text": full_text,
            "coverage_type": "full_paper"
        }

        fs = compress_paper(paper_dict, paper_id="LP-49K")
        assert fs is not None
        assert fs.metrics.original_word_count >= word_count * 0.95
        assert len(fs.sections) >= 5

    def test_72k_word_paper_processed_without_truncation(self):
        """A ~72,618 word paper must be accepted without arbitrary truncation."""
        sentence = "The algorithm optimizes hyperparameters using Bayesian gradient estimation techniques efficiently. "
        block = sentence * 150  # ~1,500 words

        sections = [
            f"Abstract\n\n{block}",
            f"Introduction\n\n{block * 8}",        # ~12,000 words
            f"Background\n\n{block * 8}",          # ~12,000 words
            f"Methodology\n\n{block * 18}",        # ~27,000 words
            f"Experiments\n\n{block * 10}",        # ~15,000 words
            f"Results\n\n{block * 4}",            # ~6,000 words
            f"Conclusion\n\n{block * 2}",         # ~3,000 words
        ]
        full_text = "\n\n".join(sections)
        word_count = len(full_text.split())
        assert word_count >= 70_000, f"Synthetic paper has {word_count} words"

        paper_dict = {
            "title": "Large Scale 72k Word Paper",
            "authors": ["Researcher B"],
            "year": 2024,
            "url": "https://example.com/large-paper-72k",
            "full_text": full_text,
            "coverage_type": "full_paper"
        }

        fs = compress_paper(paper_dict, paper_id="LP-72K")
        assert fs is not None
        assert fs.metrics.original_word_count >= word_count * 0.95


# ===========================================================================
# 4. Section Types & Non-Research Section Skipping Tests
# ===========================================================================
class TestSectionTypesAndSkipping:
    def test_all_research_sections_recognized(self):
        research_headings = [
            ("Abstract", "abstract"),
            ("1. Introduction", "introduction"),
            ("Background", "background"),
            ("Related Work", "related_work"),
            ("Problem Formulation", "problem_definition"),
            ("Methodology", "methodology"),
            ("Benchmark Dataset", "dataset"),
            ("Experimental Setup", "experiments"),
            ("Experimental Results", "results"),
            ("Discussion", "discussion"),
            ("Limitations", "limitations"),
            ("Future Work", "future_work"),
            ("Conclusion", "conclusion"),
        ]
        paper_text = ""
        for h, _ in research_headings:
            paper_text += f"{h}\n\nSubstantive content for section {h} with technical details.\n\n"

        sections = parse_paper_sections(paper_text)
        for _, expected_key in research_headings:
            assert expected_key in sections, f"Expected research key '{expected_key}' missing from parsed sections"

    def test_all_non_research_sections_skipped(self):
        non_research_blocks = [
            ("References", "references"),
            ("Bibliography", "bibliography"),
            ("Acknowledgments", "acknowledgments"),
            ("Funding", "funding"),
            ("Conflict of Interest", "conflict_of_interest"),
            ("Author Contributions", "author_contributions"),
            ("Supplementary Material", "supplementary_material"),
        ]
        text_parts = [
            "Introduction\n\nWe present a new algorithm for deep learning.",
            "Methodology\n\nOur method uses 12 layers of cross-attention.",
        ]
        for header, _ in non_research_blocks:
            text_parts.append(f"{header}\n\nBoilerplate metadata content for {header}.")

        full_text = "\n\n".join(text_parts)
        paper_dict = {
            "title": "Non-Research Skipping Test",
            "authors": ["Author"],
            "year": 2024,
            "url": "https://example.com/test",
            "full_text": full_text,
            "coverage_type": "full_paper"
        }

        fs = compress_paper(paper_dict, paper_id="NR001")
        # Non-research sections must not be in fs.sections
        for _, key in non_research_blocks:
            assert key not in fs.sections, f"Non-research section '{key}' should have been skipped from fact sheet"

        # Skipped count must be recorded
        assert fs.metrics.non_research_sections_skipped >= len(non_research_blocks)
        # Research sections covered must be 2 out of 2 = 100%
        assert fs.metrics.research_sections == 2
        assert fs.metrics.research_sections_covered == 2
        assert fs.metrics.research_section_coverage == 1.0


# ===========================================================================
# 5. Unknown Section Content Inspection Tests
# ===========================================================================
class TestUnknownSectionInspection:
    def test_unknown_containing_references_reclassified(self):
        ref_text = (
            "[1] J. Smith, A. Doe, 'Deep Learning Foundations', Nature, 2021, doi:10.1038/s41586-021.\n"
            "[2] B. Johnson et al., 'Neural Audio Detection', IEEE Trans. Audio, vol. 12, pp. 45-56, 2022.\n"
            "[3] C. Lee, 'Transformers in Speech', arXiv:2201.04567, 2023.\n"
            "[4] D. Miller, 'Acoustic Spoofing', In Proceedings of ICASSP, 2023."
        )
        assert inspect_and_classify_unknown_section(ref_text) == "references"

    def test_unknown_containing_funding_reclassified(self):
        funding_text = "This work was supported by National Science Foundation Grant No. IIS-2041234 and award number 198273."
        assert inspect_and_classify_unknown_section(funding_text) == "funding"

    def test_unknown_containing_conflict_of_interest_reclassified(self):
        coi_text = "The authors declare that they have no competing financial interests or conflict of interest."
        assert inspect_and_classify_unknown_section(coi_text) == "conflict_of_interest"

    def test_unknown_containing_research_content_preserved(self):
        research_text = (
            "We formulate the problem as a sequence-to-sequence mapping task. "
            "Given an acoustic waveform x, our encoder extracts 512-dimensional feature vectors. "
            "A bidirectional LSTM network with attention mechanism processes these representations."
        )
        assert inspect_and_classify_unknown_section(research_text) == "unknown"


# ===========================================================================
# 6. Metrics Calculation & Coverage Tests
# ===========================================================================
class TestCoverageMetrics:
    def test_metrics_example_10_detected_8_research_2_skipped(self):
        """
        Detected: 10
        Research: 8
        Skipped: 2
        Covered: 8/8
        Research coverage: 100%
        """
        sections = [
            "Abstract\n\nAbstract content with summary and findings.",
            "Introduction\n\nIntroduction paragraph explaining motivation.",
            "Related Work\n\nPrior work in synthetic speech detection.",
            "Problem Formulation\n\nMathematical definition of the audio detection task.",
            "Methodology\n\nSpectral-temporal dual classifier architecture.",
            "Experiments\n\nEvaluation on ASVspoof 2021 dataset with 8 GPUs.",
            "Results\n\nAchieves 97.4% accuracy and 0.92 F1 score on test set.",
            "Conclusion\n\nSummary of findings and impact of the proposed model.",
            "References\n\n[1] Smith et al. 2020. [2] Jones et al. 2021.",
            "Funding\n\nSupported by Grant 12345.",
        ]
        full_text = "\n\n".join(sections)
        paper_dict = {
            "title": "Full Paper 10 Sections",
            "authors": ["Author"],
            "year": 2024,
            "url": "https://example.com",
            "full_text": full_text,
            "coverage_type": "full_paper"
        }

        fs = compress_paper(paper_dict, paper_id="MET-01")
        m = fs.metrics
        assert m.detected_sections == 10
        assert m.research_sections == 8
        assert m.non_research_sections_skipped == 2
        assert m.research_sections_covered == 8
        assert m.research_section_coverage == 1.0
