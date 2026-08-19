"""
Tests for Agent 3 patent fallback behavior when Gemini synthesis fails.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import Agent3PatentOutput, PatentInfo


def _make_minimal_patent_output(status: str) -> Agent3PatentOutput:
    """Helper: build a minimal Agent3PatentOutput with given status."""
    return Agent3PatentOutput(
        patents=[
            PatentInfo(
                patent_id="US11789567B2",
                title="Test Patent",
                assignee="Test Corp",
                relevance="Overlap",
                summary="Tests patent fallback behavior.",
                fto_rating="Caution",
                design_around_strategy="Use alternative approach.",
                url="https://patents.google.com/patent/US11789567B2/en",
                source_links={"Google Patents": "https://patents.google.com/patent/US11789567B2/en"},
                relevance_score=72,
                match_explanation="Keyword overlap on test terms.",
            )
        ],
        white_space_opportunities=["Opportunity A", "Opportunity B", "Opportunity C"],
        patent_analysis_status=status,
        retrieved_patents_raw=[{"patent_id": "US11789567B2", "title": "Test Patent"}],
    )


def test_patent_output_has_status_field():
    out = _make_minimal_patent_output("success")
    assert hasattr(out, "patent_analysis_status")
    assert out.patent_analysis_status == "success"


def test_patent_output_retrieval_success_synthesis_failed():
    out = _make_minimal_patent_output("retrieval_success_synthesis_failed")
    assert out.patent_analysis_status == "retrieval_success_synthesis_failed"
    # Must still have patents (built from raw data)
    assert len(out.patents) >= 1


def test_patent_output_preserves_raw_data():
    out = _make_minimal_patent_output("retrieval_success_synthesis_failed")
    assert len(out.retrieved_patents_raw) >= 1
    assert out.retrieved_patents_raw[0]["patent_id"] == "US11789567B2"


def test_patent_output_default_status_is_success():
    """Default status should be 'success' when not explicitly set."""
    out = Agent3PatentOutput(
        patents=[],
        white_space_opportunities=[],
    )
    assert out.patent_analysis_status == "success"
    assert out.retrieved_patents_raw == []


def test_patent_output_success_path_has_patents():
    out = _make_minimal_patent_output("success")
    assert len(out.patents) > 0
    assert all(p.patent_id for p in out.patents)
    assert all(p.summary for p in out.patents)
