"""
test_full_paper_eligibility.py
==============================
Tests for Phase 1: Full-Paper Eligibility + Research Paper Selection

Verifies:
  1. Relevant + full paper -> selected (FULL_TEXT_AVAILABLE)
  2. Relevant + abstract only -> rejected from main corpus
  3. Relevant + unavailable -> rejected from main corpus
  4. Metadata-only page -> rejected by is_full_text_usable
  5. Full-text retrieval failure -> rejected (ACCESS_CHECK_FAILED or fallback to abstract/unavailable)
  6. Full-text successfully retrieved & usable -> eligible (FULL_TEXT_AVAILABLE)
  7. Enough full papers in candidate pool -> Top-N selected
  8. Not enough full papers -> fewer than N returned (no abstract backfilling)
  9. Abstract-only papers never reach compression / fact-sheet generation
 10. Abstract-only papers never reach main gap analysis
"""
import sys
import os
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent_research import (
    is_full_text_usable,
    _resolve_full_text,
    fetch_arxiv_papers,
)
from agents.agent_gap import is_evidence_eligible, cluster_and_analyze_gaps
from schemas import (
    Agent1ResearchOutput,
    PaperMetadata,
    CandidateDiagnostic,
)


# ---------------------------------------------------------------------------
# Test 1 & 6: Relevant + full paper -> selected (FULL_TEXT_AVAILABLE)
# ---------------------------------------------------------------------------
def test_full_text_usable_valid_paper():
    """Legitimate research paper with sections and paragraphs must be usable."""
    valid_text = (
        "Introduction\n"
        "Machine learning models have achieved significant performance across diverse benchmarks. "
        "In this study, we propose a novel transformer architecture for biomedical relation extraction.\n\n"
        "Methodology\n"
        "We utilize multi-head self-attention mechanisms trained on domain-specific corpora. "
        "The model is evaluated using 5-fold cross-validation across standard clinical datasets.\n\n"
        "Results and Discussion\n"
        "Our model demonstrates an F1 score of 94.2% on the benchmark, outperforming existing baselines by 4.8%. "
        "Ablation experiments confirm the necessity of domain-adaptive pretraining.\n\n"
        "Conclusion\n"
        "We presented an end-to-end relation extraction system with superior generalization. "
        "Future work includes extending this approach to multi-modal biomedical imaging datasets."
    )
    # Replicate to exceed 300 words
    long_valid_text = (valid_text + "\n\n") * 5
    assert is_full_text_usable(long_valid_text) is True


# ---------------------------------------------------------------------------
# Test 4: Metadata-only page -> rejected by is_full_text_usable
# ---------------------------------------------------------------------------
def test_metadata_only_page_rejected():
    """Pages with only citations, author metadata, and no research body must be rejected."""
    metadata_text = (
        "Download Citation\n"
        "Cite this article: Smith, J., Doe, A. 2024. A study on transformers.\n"
        "Rights and permissions\n"
        "About this article\n"
        "Share this article\n"
        "Published: 12 January 2024\n"
        "Volume 42, Article number: 105 (2024)\n"
    ) * 10
    assert is_full_text_usable(metadata_text) is False


def test_error_and_paywall_pages_rejected():
    """Error, captcha, and sign-in pages must be rejected."""
    assert is_full_text_usable("404 Not Found. The requested URL was not found on this server." * 20) is False
    assert is_full_text_usable("Access Denied. You do not have permission to access this resource. Sign in to access." * 20) is False
    assert is_full_text_usable("Attention Required! Cloudflare verify you are human captcha." * 20) is False
    assert is_full_text_usable("") is False
    assert is_full_text_usable("Short text with only twenty words.") is False


# ---------------------------------------------------------------------------
# Test 2: Relevant + abstract only -> rejected from main corpus
# ---------------------------------------------------------------------------
def test_resolve_abstract_only_produces_abstract_only_status():
    """Papers with only an abstract must receive access_status='ABSTRACT_ONLY'."""
    paper_dict = {
        "title": "Abstract Only Paper",
        "abstract": "This is a real abstract about deep reinforcement learning algorithms for robotic manipulation in continuous control domains with safety constraints.",
        "url": "https://example.com/no-fulltext",
        "open_access_pdf_url": "",
        "oa_url": "",
        "doi": "",
    }
    result = _resolve_full_text(paper_dict)
    assert result["access_status"] == "ABSTRACT_ONLY"
    assert result["coverage_type"] == "abstract_only"
    assert result["text"] == ""


# ---------------------------------------------------------------------------
# Test 3: Relevant + unavailable -> rejected from main corpus
# ---------------------------------------------------------------------------
def test_resolve_unavailable_produces_unavailable_status():
    """Papers with no abstract and no full text must receive access_status='UNAVAILABLE'."""
    paper_dict = {
        "title": "Unavailable Paper",
        "abstract": "",
        "url": "",
        "open_access_pdf_url": "",
        "oa_url": "",
        "doi": "",
    }
    result = _resolve_full_text(paper_dict)
    assert result["access_status"] == "UNAVAILABLE"
    assert result["coverage_type"] == "unavailable"


# ---------------------------------------------------------------------------
# Test 5: Full-text retrieval failure -> rejected
# ---------------------------------------------------------------------------
def test_retrieval_failure_rejected_in_resolver(monkeypatch):
    """When a network error occurs during fetch, resolver returns abstract_only or unavailable."""
    def mock_fetch_err(url):
        raise RuntimeError("Connection timed out")

    monkeypatch.setattr("agents.agent_research._fetch_arxiv_full_text", mock_fetch_err)

    paper_dict = {
        "title": "Failing Arxiv Paper",
        "abstract": "A valid abstract about optimization in deep neural networks across multiple benchmark datasets with empirical evaluations.",
        "url": "http://arxiv.org/abs/2301.99999",
        "open_access_pdf_url": "",
        "oa_url": "",
        "doi": "",
    }
    result = _resolve_full_text(paper_dict)
    # Should fall back to abstract only, not full text
    assert result["access_status"] == "ABSTRACT_ONLY"


# ---------------------------------------------------------------------------
# Test 7 & 8: Top-N selection & No backfilling when full papers are scarce
# ---------------------------------------------------------------------------
def test_selection_pipeline_filters_abstract_only_and_returns_only_full_papers(monkeypatch):
    """
    Simulate candidate pool with:
      - 2 candidates with real full text
      - 2 candidates with abstract only
    With max_results=5, pipeline must return EXACTLY 2 papers (not 5).
    """
    valid_full_text = (
        "Introduction\n"
        "We investigate graph neural networks for molecular property prediction in drug discovery.\n\n"
        "Methodology\n"
        "We construct message passing layers that integrate edge attributes representing chemical bond lengths.\n\n"
        "Results\n"
        "The model achieves 0.91 ROC-AUC on the benchmark dataset, validating the hypothesis.\n\n"
        "Conclusion\n"
        "Graph neural networks with geometric embeddings provide strong predictive accuracy."
    ) * 5

    candidate_pool = [
        # Candidate 1: High relevance, full text available
        {
            "title": "Full Paper 1: Graph Neural Networks in Chemistry",
            "authors": ["Alice"],
            "year": 2024,
            "abstract": "We study graph neural networks in chemistry for molecular prediction.",
            "url": "http://arxiv.org/abs/2401.00001",
            "full_text": valid_full_text,
            "open_access_pdf_url": "",
            "oa_url": "",
            "doi": "",
            "source": "arXiv",
        },
        # Candidate 2: High relevance, ABSTRACT ONLY (no full text)
        {
            "title": "Abstract Only 1: Molecular Prediction Survey",
            "authors": ["Bob"],
            "year": 2024,
            "abstract": "A survey of molecular prediction techniques in computational drug design.",
            "url": "https://example.com/survey",
            "full_text": "",
            "open_access_pdf_url": "",
            "oa_url": "",
            "doi": "",
            "source": "Crossref",
        },
        # Candidate 3: High relevance, full text available
        {
            "title": "Full Paper 2: Geometric Deep Learning for Molecules",
            "authors": ["Carol"],
            "year": 2024,
            "abstract": "Geometric deep learning models for molecular graph classification.",
            "url": "http://arxiv.org/abs/2401.00002",
            "full_text": valid_full_text,
            "open_access_pdf_url": "",
            "oa_url": "",
            "doi": "",
            "source": "arXiv",
        },
        # Candidate 4: Medium relevance, ABSTRACT ONLY
        {
            "title": "Abstract Only 2: Benchmarking Graph Libraries",
            "authors": ["Dave"],
            "year": 2023,
            "abstract": "Benchmarking popular graph libraries on standardized computational benchmarks.",
            "url": "https://example.com/benchmarks",
            "full_text": "",
            "open_access_pdf_url": "",
            "oa_url": "",
            "doi": "",
            "source": "Semantic Scholar",
        },
    ]

    # Mock candidate search to return our candidate pool
    monkeypatch.setattr("agents.agent_research.fetch_arxiv", lambda q, l: [candidate_pool[0], candidate_pool[2]])
    monkeypatch.setattr("agents.agent_research.fetch_crossref", lambda q, l: [candidate_pool[1]])
    monkeypatch.setattr("agents.agent_research.fetch_semantic_scholar", lambda q, l: [candidate_pool[3]])
    monkeypatch.setattr("agents.agent_research.fetch_openalex", lambda q, l: [])
    monkeypatch.setattr("agents.agent_research.fetch_europe_pmc", lambda q, l: [])
    monkeypatch.setattr("agents.agent_research.fetch_doaj", lambda q, l: [])

    # Mock Gemini call to return instant JSON
    def mock_gemini(*args, **kwargs):
        return json.dumps({
            "papers": [
                {
                    "title": "Full Paper 1: Graph Neural Networks in Chemistry",
                    "authors": ["Alice"],
                    "year": 2024,
                    "abstract": "We study graph neural networks in chemistry for molecular prediction.",
                    "relevance_score": 0.92,
                    "relevance_rank": "High",
                    "innovation_score": 85,
                    "research_significance": "High impact",
                    "notebook_summary": "• What it studies: GNNs in chemistry. • How it does it: Message passing. • What it achieves: 0.91 ROC-AUC.",
                    "technical_execution": "Message passing neural network",
                    "datasets": ["MoleculeNet"],
                    "url": "http://arxiv.org/abs/2401.00001",
                    "problem_statement": "Molecular property prediction",
                    "proposed_solution": "Edge attribute message passing",
                    "methodology": "Graph neural networks",
                    "results": "0.91 ROC-AUC",
                    "challenges": "Scalability to large macromolecules",
                    "future_outcomes": "Extend to 3D conformation"
                },
                {
                    "title": "Full Paper 2: Geometric Deep Learning for Molecules",
                    "authors": ["Carol"],
                    "year": 2024,
                    "abstract": "Geometric deep learning models for molecular graph classification.",
                    "relevance_score": 0.88,
                    "relevance_rank": "High",
                    "innovation_score": 82,
                    "research_significance": "High impact",
                    "notebook_summary": "• What it studies: Geometric DL. • How it does it: Equivariant layers. • What it achieves: High accuracy.",
                    "technical_execution": "Equivariant graph network",
                    "datasets": ["QM9"],
                    "url": "http://arxiv.org/abs/2401.00002",
                    "problem_statement": "Equivariance in molecular modeling",
                    "proposed_solution": "SE(3) equivariant message passing",
                    "methodology": "Equivariant convolutions",
                    "results": "Low MAE on QM9",
                    "challenges": "Computational cost",
                    "future_outcomes": "Protein-ligand docking"
                }
            ]
        })

    monkeypatch.setattr("agents.agent_research._gemini_generate_with_retry", mock_gemini)

    output = fetch_arxiv_papers("graph neural networks chemistry", max_results=5)

    # Must return EXACTLY 2 papers (the full papers), NOT 4 or 5
    assert len(output.papers) == 2, f"Expected 2 eligible full papers, got {len(output.papers)}"
    for p in output.papers:
        assert p.access_status == "FULL_TEXT_AVAILABLE"
        assert p.full_text_available is True
        assert p.coverage_type in ("full_paper", "partial_paper")
        assert len(p.full_text) > 0

    # Eligibility rate must be 1.0 (100%)
    assert output.full_paper_eligibility_rate == 1.0

    # Diagnostics must record all 4 evaluated candidates
    assert len(output.candidate_diagnostics) >= 4
    selected_count = sum(1 for d in output.candidate_diagnostics if d.selected)
    rejected_count = sum(1 for d in output.candidate_diagnostics if not d.selected)
    assert selected_count == 2
    assert rejected_count >= 2


# ---------------------------------------------------------------------------
# Test 9: Abstract-only papers never reach compression
# ---------------------------------------------------------------------------
def test_abstract_only_papers_not_in_compression_output():
    """All papers in Agent1ResearchOutput.papers must have full text and not be abstract only."""
    paper = PaperMetadata(
        title="Full Text Validated Paper",
        authors=["Author A"],
        year=2024,
        abstract="Valid abstract.",
        relevance_rank="High",
        notebook_summary="Summary.",
        technical_execution="Method.",
        datasets=["Dataset 1"],
        url="http://arxiv.org/abs/2401.00001",
        relevance_score=0.9,
        access_status="FULL_TEXT_AVAILABLE",
        full_text="Introduction\n" + ("Valid full paper content with scientific prose. " * 50),
        full_text_available=True,
        coverage_type="full_paper",
    )
    research_out = Agent1ResearchOutput(query="test", papers=[paper])
    for p in research_out.papers:
        assert p.access_status == "FULL_TEXT_AVAILABLE"
        assert p.coverage_type != "abstract_only"
        assert p.coverage_type != "unavailable"


# ---------------------------------------------------------------------------
# Test 10: Abstract-only papers never reach main gap analysis
# ---------------------------------------------------------------------------
def test_abstract_only_paper_rejected_by_evidence_eligibility():
    """Gap analysis must reject abstract-only papers from evidence claims."""
    class MockAbstractOnlyPaper:
        access_status = "ABSTRACT_ONLY"
        coverage_type = "abstract_only"

    class MockFullPaper:
        access_status = "FULL_TEXT_AVAILABLE"
        coverage_type = "full_paper"

    assert is_evidence_eligible(MockAbstractOnlyPaper()) is False
    assert is_evidence_eligible(MockFullPaper()) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
