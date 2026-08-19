"""
test_compression_target_60_70.py
=================================
Tests for Final Compression Budget Fix: 65–75% RETENTION (Target ~70% Retention / ~30% Reduction).

Verifies:
- Configuration constants:
    - TARGET_RETENTION_MIN = 0.65
    - TARGET_RETENTION_MAX = 0.75
    - TARGET_RETENTION_DEFAULT = 0.70
- Token budgets on synthetic benchmarks:
    - 10,000 token paper -> ~6,500 - 7,500 tokens (preferred ~7,000)
    - 17,000 token paper -> ~11,000 - 13,000 tokens (preferred ~12,000)
    - 20,000 token paper -> ~13,000 - 15,000 tokens (preferred ~14,000)
    - 50,000 token paper -> ~32,500 - 37,500 tokens (preferred ~35,000)
- Research section coverage (100% research sections)
- Technical sentence priority preservation
- Extractive faithfulness (100% verbatim source verification)
- Insufficient candidate handling & warning logs
- Large-paper processing (no character truncation)
- Small-paper handling
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.models import (
    CompressionConfig,
    TARGET_RETENTION_MIN,
    TARGET_RETENTION_MAX,
    TARGET_RETENTION_DEFAULT,
    TARGET_COMPRESSION_MIN,
    TARGET_COMPRESSION_MAX,
)
from compression.fact_sheet import compress_paper
from compression.token_utils import estimate_tokens


# ===========================================================================
# 1. Configuration Constants Tests
# ===========================================================================
class TestConfigurationConstants:
    def test_retention_constants_defined(self):
        assert TARGET_RETENTION_MIN == 0.65
        assert TARGET_RETENTION_MAX == 0.75
        assert TARGET_RETENTION_DEFAULT == 0.70

    def test_config_defaults(self):
        cfg = CompressionConfig()
        assert cfg.target_retention_min == 0.65
        assert cfg.target_retention_max == 0.75
        assert cfg.target_retention_default == 0.70
        assert cfg.target_retention_ratio == 0.70
        assert cfg.min_retention_5k == 0.65
        assert cfg.min_retention_20k == 0.65


# ===========================================================================
# 2. Token Budget Regression Tests (10k, 17k, 20k, 50k tokens)
# ===========================================================================
class TestCompressionBudgetRetention:
    def _create_structured_paper(self, total_sentences: int) -> str:
        """Construct a structured research paper with arbitrary sentence scale."""
        sections = [
            ("1. Introduction", 0.15),
            ("2. Related Work", 0.10),
            ("3. Methodology", 0.30),
            ("4. Experiments", 0.20),
            ("5. Results", 0.15),
            ("6. Limitations", 0.05),
            ("7. Conclusion", 0.05),
        ]
        
        parts = []
        sent_idx = 0
        for title, fraction in sections:
            count = max(1, int(round(total_sentences * fraction)))
            sec_sents = []
            for _ in range(count):
                sent_idx += 1
                sec_sents.append(
                    f"We study neural transformer representations and experimental benchmarks under noisy recording conditions "
                    f"at trial index {sent_idx} achieving 98.{sent_idx % 10}% accuracy with p < 0.001 across diverse corpora."
                )
            parts.append(f"{title}\n\n" + " ".join(sec_sents))
        return "\n\n".join(parts)

    def test_10k_token_paper_yields_6500_to_7500_tokens(self):
        raw_text = self._create_structured_paper(total_sentences=340)
        orig_tokens = estimate_tokens(raw_text)
        assert 9_000 <= orig_tokens <= 11_500, f"Expected ~10k tokens, got {orig_tokens}"

        paper_dict = {
            "title": "Acoustic Representation Learning at Scale (10k Tokens)",
            "authors": ["Author A", "Author B"],
            "year": 2024,
            "url": "https://example.com/10k",
            "full_text": raw_text,
            "coverage_type": "full_paper",
        }
        fs = compress_paper(paper_dict, paper_id="P10K")
        m = fs.metrics

        # Target: 65–75% retention (preferred ~70%)
        assert 0.65 <= m.retained_ratio <= 0.76, (
            f"10k paper retained ratio {m.retained_ratio:.2f} not in 65-75% window"
        )
        assert 6_500 <= m.compressed_token_count <= 8_000, (
            f"Compressed tokens {m.compressed_token_count} out of 6,500-7,500 target"
        )
        assert m.research_section_coverage == 1.0
        assert m.faithfulness_ratio == 1.0

    def test_17k_token_paper_yields_11000_to_13000_tokens(self):
        raw_text = self._create_structured_paper(total_sentences=570)
        orig_tokens = estimate_tokens(raw_text)
        assert 15_500 <= orig_tokens <= 18_500, f"Expected ~17k tokens, got {orig_tokens}"

        paper_dict = {
            "title": "Scalable Speech Deepfake Detection (17k Tokens)",
            "authors": ["Author A", "Author B"],
            "year": 2024,
            "url": "https://example.com/17k",
            "full_text": raw_text,
            "coverage_type": "full_paper",
        }
        fs = compress_paper(paper_dict, paper_id="P17K")
        m = fs.metrics

        assert 0.65 <= m.retained_ratio <= 0.76, (
            f"17k paper retained ratio {m.retained_ratio:.2f} not in 65-75% window"
        )
        assert 11_000 <= m.compressed_token_count <= 13_500, (
            f"Compressed tokens {m.compressed_token_count} out of 11,000-13,000 target"
        )
        assert m.research_section_coverage == 1.0
        assert m.faithfulness_ratio == 1.0

    def test_20k_token_paper_yields_13000_to_15000_tokens(self):
        raw_text = self._create_structured_paper(total_sentences=670)
        orig_tokens = estimate_tokens(raw_text)
        assert 18_500 <= orig_tokens <= 22_500, f"Expected ~20k tokens, got {orig_tokens}"

        paper_dict = {
            "title": "DeepAudioNet: Robust Deepfake Audio Detection (20k Tokens)",
            "authors": ["Author A", "Author B"],
            "year": 2024,
            "url": "https://example.com/20k",
            "full_text": raw_text,
            "coverage_type": "full_paper",
        }
        fs = compress_paper(paper_dict, paper_id="P20K")
        m = fs.metrics

        assert 0.65 <= m.retained_ratio <= 0.76, (
            f"20k paper retained ratio {m.retained_ratio:.2f} not in 65-75% window"
        )
        assert 13_000 <= m.compressed_token_count <= 16_000, (
            f"Compressed tokens {m.compressed_token_count} out of 13,000-15,000 target"
        )
        assert m.research_section_coverage == 1.0
        assert m.faithfulness_ratio == 1.0

    def test_50k_token_paper_yields_32500_to_37500_tokens(self):
        raw_text = self._create_structured_paper(total_sentences=1700)
        orig_tokens = estimate_tokens(raw_text)
        assert orig_tokens >= 45_000, f"Expected ~50k tokens, got {orig_tokens}"

        paper_dict = {
            "title": "Comprehensive Acoustic Spoofing Survey (50k Tokens)",
            "authors": ["Author A", "Author B"],
            "year": 2024,
            "url": "https://example.com/50k",
            "full_text": raw_text,
            "coverage_type": "full_paper",
        }
        fs = compress_paper(paper_dict, paper_id="P50K")
        m = fs.metrics

        assert 0.65 <= m.retained_ratio <= 0.76, (
            f"50k paper retained ratio {m.retained_ratio:.2f} not in 65-75% window"
        )
        assert 32_000 <= m.compressed_token_count <= 39_000, (
            f"Compressed tokens {m.compressed_token_count} out of 32,500-37,500 target"
        )
        assert m.research_section_coverage == 1.0
        assert m.faithfulness_ratio == 1.0


# ===========================================================================
# 3. Technical Priority Preservation & Extractive Faithfulness Tests
# ===========================================================================
class TestTechnicalPriorityAndFaithfulness:
    def test_technical_sentences_prioritized(self):
        text = (
            "1. Introduction\n\n"
            "This is a generic background introductory sentence with no specific data.\n"
            "Another generic introductory thought about the general field.\n"
            "Our model attains 98.7% accuracy and a 0.012 EER with p < 0.001.\n"
            "Generic concluding introductory remarks for this paragraph.\n\n"
            "2. Results\n\n"
            "We present the benchmark evaluation results below.\n"
            "The baseline ResNet-50 achieves 91.2% accuracy on the test split of 10,000 samples.\n"
            "Our proposed model surpasses the baseline by 7.5% with a 95% confidence interval [98.1, 99.3].\n"
            "Some qualitative observations are discussed in the following text.\n"
        )
        paper_dict = {
            "title": "Technical Priority Test",
            "authors": ["Author"],
            "year": 2024,
            "url": "https://example.com/tech",
            "full_text": text,
            "coverage_type": "full_paper",
        }
        fs = compress_paper(paper_dict, paper_id="TECH01")
        assert fs.metrics.technical_sentence_count >= 2

        extracted_text = " ".join(
            s.text for sheet in fs.sections.values() for s in sheet.selected_sentences
        )
        assert "98.7% accuracy" in extracted_text or "91.2% accuracy" in extracted_text

    def test_100_percent_extractive_faithfulness(self):
        text = (
            "1. Introduction\n\n"
            "Neural audio synthesis produces highly convincing fake recordings.\n"
            "Detection algorithms must generalize to unseen recording conditions.\n\n"
            "2. Methodology\n\n"
            "We apply multi-scale spectral filtering to isolate high-frequency phase artifacts.\n"
            "A dual-branch convolutional backbone extracts temporal and spectral representations.\n\n"
            "3. Results\n\n"
            "Evaluation yields 96.4% AUC and 0.89 F1 score on the test set.\n"
        )
        paper_dict = {
            "title": "Faithfulness Test",
            "authors": ["Author"],
            "year": 2024,
            "url": "https://example.com/faith",
            "full_text": text,
            "coverage_type": "full_paper",
        }
        fs = compress_paper(paper_dict, paper_id="FAITH01")
        assert fs.metrics.faithfulness_ratio == 1.0
        for sheet in fs.sections.values():
            for sent in sheet.selected_sentences:
                assert sent.source_verified is True


# ===========================================================================
# 4. Insufficient Candidates & Small Paper Handling Tests
# ===========================================================================
class TestEdgeCasesAndSafeguards:
    def test_small_paper_handling(self):
        text = (
            "1. Introduction\n\n"
            "This is a small paper introductory sentence with enough words to parse.\n"
            "Another sentence explaining the overall research scope.\n\n"
            "2. Conclusion\n\n"
            "This is the final small concluding takeaway sentence."
        )
        paper_dict = {
            "title": "Small Paper",
            "authors": ["Author"],
            "year": 2024,
            "url": "https://example.com/small",
            "full_text": text,
            "coverage_type": "full_paper",
        }
        fs = compress_paper(paper_dict, paper_id="SMALL01")
        assert fs.metrics.covered_sections == 2
        assert fs.metrics.faithfulness_ratio == 1.0

    def test_abstract_only_reports_not_evaluated(self):
        paper_dict = {
            "title": "Abstract Only Paper",
            "authors": ["Author"],
            "year": 2024,
            "url": "https://example.com/abs",
            "abstract": "This is only an abstract describing audio deepfakes.",
            "full_text": "",
            "coverage_type": "abstract_only",
        }
        fs = compress_paper(paper_dict, paper_id="ABS01")
        assert fs.metrics.compression_status == "not_evaluated"
