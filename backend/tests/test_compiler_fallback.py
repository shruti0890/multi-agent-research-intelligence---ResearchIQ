"""
Tests for Agent 4 compiler fallback behavior and ProjectReportState status fields.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import (
    ProjectReportState,
    Agent1ResearchOutput,
    Agent2GapOutput,
    Agent3PatentOutput,
    ResearchGap,
    NovelMethodProposal,
    PaperMetadata,
)


def _make_minimal_state() -> ProjectReportState:
    research = Agent1ResearchOutput(
        query="test topic",
        papers=[
            PaperMetadata(
                title="Test Paper",
                authors=["A. Author"],
                year=2024,
                abstract="A test abstract.",
                relevance_rank="High",
                notebook_summary="Summary.",
                technical_execution="Method.",
                datasets=["Dataset A"],
                url="https://example.com",
                relevance_score=0.85,
                problem_statement="Problem.",
                proposed_solution="Solution.",
                methodology="Method.",
                results="Results.",
                challenges="Challenges.",
                future_outcomes="Future.",
                innovation_score=80,
                research_significance="Significant.",
            )
        ],
    )
    gaps = Agent2GapOutput(
        gaps=[
            ResearchGap(
                description="A test gap",
                severity="High",
                why_it_matters="It matters because of X.",
                evidence_papers=["Test Paper"],
            )
        ],
        proposed_method=NovelMethodProposal(
            title="A Novel Method",
            approach="An approach.",
            novelty_score=80,
            rationale="Sound rationale.",
        ),
    )
    patents = Agent3PatentOutput(
        patents=[],
        white_space_opportunities=["Opportunity A"],
        patent_analysis_status="success",
    )
    return ProjectReportState(
        topic="test topic",
        research=research,
        gaps=gaps,
        patents=patents,
    )


def test_project_report_state_has_compiler_status():
    state = _make_minimal_state()
    assert hasattr(state, "compiler_status")
    assert state.compiler_status == "success"


def test_project_report_state_has_gemini_quota_exhausted():
    state = _make_minimal_state()
    assert hasattr(state, "gemini_quota_exhausted")
    assert state.gemini_quota_exhausted is False


def test_project_report_state_compiler_fallback():
    state = _make_minimal_state()
    state.compiler_status = "fallback_due_to_gemini"
    state.gemini_quota_exhausted = True
    assert state.compiler_status == "fallback_due_to_gemini"
    assert state.gemini_quota_exhausted is True


def test_project_report_state_patent_quota_propagation():
    """If Agent 3 failed due to quota, state.gemini_quota_exhausted should be settable."""
    state = _make_minimal_state()
    state.patents.patent_analysis_status = "retrieval_success_synthesis_failed"
    # Simulate what main.py does
    if state.patents.patent_analysis_status in ("retrieval_success_synthesis_failed",):
        state.gemini_quota_exhausted = True
    assert state.gemini_quota_exhausted is True
