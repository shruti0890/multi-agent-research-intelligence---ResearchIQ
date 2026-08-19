"""
test_runtime_stability.py
==========================
Tests for Phase 6: Gemini Runtime Stability + API Failure Handling.

Verifies:
- 503 and transient 429 retry logic with exponential backoff (max 3 retries)
- Quota exhaustion (RESOURCE_EXHAUSTED, daily quota, free-tier limit, GenerateRequestsPerDay, quotaValue) raises immediately with no retry
- Agent 1 preserves paper data on Gemini failure
- Agent 2 preserves evidence fact sheets on Gemini failure
- Agent 3 preserves real patent records and sets retrieval_success_synthesis_failed
- Agent 4 compiles PDF with compiler_status = fallback_due_to_gemini and warning banner
- Semantic Scholar handles 429 rate limit with Retry-After and backoff
- Frontend single-click request lock (pipelineRunning) prevents duplicate requests
"""

import os
import sys
import time
import json
import tempfile
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import (
    PaperMetadata,
    Agent1ResearchOutput,
    Agent2GapOutput,
    ResearchGap,
    NovelMethodProposal,
    Agent3PatentOutput,
    PatentInfo,
    ProjectReportState,
)
from agents.agent_research import (
    _gemini_generate_with_retry as a1_gemini_retry,
    GeminiQuotaExhaustedError,
    _is_quota_exhausted,
    fetch_semantic_scholar,
)
from agents.agent_gap import (
    cluster_and_analyze_gaps,
    _gemini_generate_with_retry as a2_gemini_retry,
)
from agents.agent_patent import (
    search_and_classify_patents,
    _gemini_generate_with_retry as a3_gemini_retry,
)
from agents.agent_compile import (
    compile_final_report,
    _gemini_generate_with_retry as a4_gemini_retry,
)


class TestGeminiRetriesAndQuotaExhaustion:
    """Validate 503, transient 429 retry and immediate quota stop."""

    def test_503_retry_succeeds_after_transient_failures(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda x: None)
        call_count = {"n": 0}

        class TransientClient:
            class models:
                @staticmethod
                def generate_content(model, contents, config):
                    call_count["n"] += 1
                    if call_count["n"] < 3:
                        raise Exception("503 Service Unavailable")
                    mock_resp = MagicMock()
                    mock_resp.text = '{"success": true}'
                    return mock_resp

        res = a1_gemini_retry(
            client=TransientClient(),
            model="gemini-2.5-flash",
            prompt="test",
            config=None,
            max_retries=3,
            agent_label="TestA1",
        )
        assert call_count["n"] == 3
        assert json.loads(res) == {"success": True}

    def test_transient_429_retried_up_to_max_retries(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda x: None)
        call_count = {"n": 0}

        class RateLimitedClient:
            class models:
                @staticmethod
                def generate_content(model, contents, config):
                    call_count["n"] += 1
                    raise Exception("429 Too Many Requests: transient rate limit exceeded")

        with pytest.raises(Exception, match="429"):
            a2_gemini_retry(
                client=RateLimitedClient(),
                model="gemini-2.5-flash",
                prompt="test",
                config=None,
                max_retries=3,
                agent_label="TestA2",
            )
        assert call_count["n"] == 3

    @pytest.mark.parametrize("error_sig", [
        "429 RESOURCE_EXHAUSTED",
        "GenerateRequestsPerDay limit exceeded",
        "quotaValue reached maximum allowed for daily quota",
        "free-tier limit exhausted for today",
        "daily quota exceeded",
    ])
    def test_quota_signals_fail_immediately_without_retrying(self, monkeypatch, error_sig):
        monkeypatch.setattr(time, "sleep", lambda x: None)
        call_count = {"n": 0}

        class QuotaClient:
            class models:
                @staticmethod
                def generate_content(model, contents, config):
                    call_count["n"] += 1
                    raise Exception(error_sig)

        with pytest.raises(GeminiQuotaExhaustedError):
            a3_gemini_retry(
                client=QuotaClient(),
                model="gemini-2.5-flash",
                prompt="test",
                config=None,
                max_retries=3,
                agent_label="TestA3",
            )
        assert call_count["n"] == 1, f"Quota signal '{error_sig}' was retried {call_count['n']} times!"


class TestAgentFallbacksOnGeminiFailure:
    """Validate data preservation and status flags when Gemini fails."""

    def test_agent_1_preserves_paper_data_on_gemini_failure(self):
        """When Gemini fails in Agent 1, research papers and full text are preserved."""
        raw_candidates = [
            {
                "title": "Acoustic Representation Analysis",
                "authors": ["Alice"],
                "year": 2024,
                "abstract": "We analyze acoustic representations.",
                "url": "https://example.com/p1",
                "source": "arXiv",
                "citations": 10,
                "open_access_pdf_url": "",
                "oa_url": "",
                "doi": "10.1234/5678",
            }
        ]

        with patch("agents.agent_research.fetch_arxiv", return_value=raw_candidates), \
             patch("agents.agent_research.fetch_semantic_scholar", return_value=[]), \
             patch("agents.agent_research.fetch_crossref", return_value=[]), \
             patch("agents.agent_research.fetch_openalex", return_value=[]), \
             patch("agents.agent_research.fetch_europe_pmc", return_value=[]), \
             patch("agents.agent_research.fetch_doaj", return_value=[]), \
             patch("agents.agent_research._resolve_full_text") as mock_resolve, \
             patch("agents.agent_research._get_client") as mock_client:
            long_text = (
                "1. Introduction\n\n" +
                "Deep learning neural networks have transformed audio representation analysis across multi-modal benchmarks. " * 20 +
                "\n\n2. Methodology\n\n" +
                "We construct an end-to-end convolutional neural network with residual dense connections to process spectrogram features. " * 20 +
                "\n\n3. Results\n\n" +
                "The proposed framework achieves superior classification performance on public benchmarks with statistical significance. " * 20
            )
            mock_resolve.return_value = {
                "text": long_text,
                "coverage_type": "full_paper",
                "source": "arxiv",
                "url": "https://example.com/pdf",
                "access_status": "FULL_TEXT_AVAILABLE",
            }
            mock_client.return_value.models.generate_content.side_effect = Exception("RESOURCE_EXHAUSTED: daily limit")

            from agents.agent_research import fetch_arxiv_papers
            out = fetch_arxiv_papers("audio analysis", max_results=1)

        assert isinstance(out, Agent1ResearchOutput)
        assert len(out.papers) == 1
        assert out.papers[0].title == "Acoustic Representation Analysis"
        assert out.papers[0].access_status == "FULL_TEXT_AVAILABLE"
        assert len(out.papers[0].full_text) > 50

    def test_agent_2_preserves_evidence_fact_sheets_on_gemini_failure(self, monkeypatch):
        """When Gemini fails, Agent 2 uses deterministic fact sheet limitation extraction."""
        p = PaperMetadata(
            paper_id="P001",
            title="Audio Spoofing Defense",
            authors=["Alice"],
            year=2024,
            abstract="We evaluate audio spoofing defenses.",
            relevance_rank="High",
            notebook_summary="Summary",
            technical_execution="Execution",
            datasets=["ASVspoof"],
            url="http://example.com/p1",
            relevance_score=0.9,
            access_status="FULL_TEXT_AVAILABLE",
            coverage_type="full_paper",
            full_text="1. Limitations\nThe model suffers from latency overhead under constrained hardware.",
        )
        research_out = Agent1ResearchOutput(
            query="Audio deepfakes",
            papers=[p],
        )

        with patch("agents.agent_gap._get_client") as mock_client:
            mock_client.return_value.models.generate_content.side_effect = Exception("RESOURCE_EXHAUSTED: daily limit")
            gap_out = cluster_and_analyze_gaps(research_out)

        assert isinstance(gap_out, Agent2GapOutput)
        assert len(gap_out.gaps) == 1
        gap = gap_out.gaps[0]
        assert gap.gap_id.startswith("GAP-")
        assert gap.supporting_paper_ids == ["P001"]
        assert len(gap.gap_statement) > 0

    def test_agent_3_fallback_sets_retrieval_success_synthesis_failed(self):
        """When patent retrieval succeeds but Gemini fails, status is retrieval_success_synthesis_failed."""
        gap_data = Agent2GapOutput(
            gaps=[ResearchGap(description="Gap 1", severity="Critical", why_it_matters="Impact", evidence_papers=["Paper 1"])],
            proposed_method=NovelMethodProposal(title="Method", approach="Approach", novelty_score=80, rationale="Rationale")
        )
        fake_raw_patent = {
            "patent_id": "US10999999B2",
            "title": "Audio Deepfake Verification System",
            "abstract": "A deep learning neural network for acoustic verification.",
            "claims": "1. A method for verifying acoustic input...",
            "applicant": "Security Corp",
            "jurisdiction": "US",
            "publication_date": "2023-01-01",
            "source": "EPMC",
            "relevance_score": 85,
        }

        with patch("agents.agent_patent.retrieve_multi_source_patents") as mock_retrieval, \
             patch("agents.agent_patent._classify_patents_with_gemini") as mock_classify:
            mock_retrieval.return_value = ([fake_raw_patent], {"EPMC": "SUCCESS"}, "SUCCESS")
            mock_classify.side_effect = Exception("Gemini daily quota exhausted")

            out = search_and_classify_patents(gap_data, "Deepfake audio")

        assert isinstance(out, Agent3PatentOutput)
        assert out.patent_analysis_status == "retrieval_success_synthesis_failed"
        assert len(out.patents) == 1
        assert out.patents[0].patent_id == "US10999999B2"
        assert out.patents[0].title == "Audio Deepfake Verification System"

    def test_agent_4_pdf_fallback_sets_fallback_due_to_gemini(self):
        """When Gemini fails in Agent 4, PDF is generated with compiler_status = fallback_due_to_gemini."""
        p = PaperMetadata(
            paper_id="P001",
            title="Acoustic Detection",
            authors=["Bob"],
            year=2024,
            abstract="Abstract",
            relevance_rank="High",
            notebook_summary="Summary",
            technical_execution="Execution",
            datasets=[],
            url="http://example.com/p",
            relevance_score=0.9,
            access_status="FULL_TEXT_AVAILABLE",
            coverage_type="full_paper",
        )
        g = Agent2GapOutput(
            gaps=[ResearchGap(gap_id="GAP-01", gap_statement="Gap stmt", description="Gap desc", severity="Critical", why_it_matters="Impact", evidence_papers=["Acoustic Detection"])],
            proposed_method=NovelMethodProposal(title="Proposed", approach="Step 1...", novelty_score=85, rationale="Rationale")
        )
        pat = Agent3PatentOutput(
            patents=[PatentInfo(patent_id="US10000001B2", title="Patent 1", assignee="Corp", relevance="Overlap", summary="Sum", fto_rating="Safe", design_around_strategy="Strat", relevance_score=80)],
            white_space_opportunities=["Opportunity 1"],
            patent_analysis_status="retrieval_success_synthesis_failed",
            patent_retrieval_status="SUCCESS",
        )
        state = ProjectReportState(
            topic="Acoustic Detection",
            research=Agent1ResearchOutput(query="Acoustic Detection", papers=[p]),
            gaps=g,
            patents=pat,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("agents.agent_compile._get_client") as mock_client:
                mock_client.return_value.models.generate_content.side_effect = Exception("RESOURCE_EXHAUSTED: daily limit")
                pdf_path, status = compile_final_report(state, output_dir=tmpdir)

            assert os.path.exists(pdf_path)
            assert status == "fallback_due_to_gemini"
            assert state.compiler_status == "fallback_due_to_gemini"
            assert state.gemini_quota_exhausted is True
            assert os.path.getsize(pdf_path) > 1000  # Non-empty valid PDF generated


class TestSemanticScholarAndFrontendStability:
    """Test API resilience and UI request lock."""

    def test_semantic_scholar_retry_on_429(self, monkeypatch):
        """Semantic scholar retries on 429 with backoff."""
        monkeypatch.setattr(time, "sleep", lambda x: None)
        call_count = {"n": 0}

        import urllib.error

        class MockResponse:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return b'{"data": [{"title": "SS Paper", "year": 2024, "authors": [{"name": "Auth"}]}]}'

        def mock_urlopen(req, timeout=10):
            call_count["n"] += 1
            if call_count["n"] < 2:
                headers = MagicMock()
                headers.get.return_value = "2"
                raise urllib.error.HTTPError(
                    url=req.full_url, code=429, msg="Too Many Requests", hdrs=headers, fp=None
                )
            return MockResponse()

        monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
        papers = fetch_semantic_scholar("test query", limit=5)
        assert call_count["n"] == 2
        assert len(papers) == 1
        assert papers[0]["title"] == "SS Paper"

    def test_frontend_duplicate_protection_present_in_index_html(self):
        """Verify pipelineRunning lock is present and prevents duplicate submissions."""
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        html_path = os.path.join(repo_root, "frontend", "index.html")
        assert os.path.exists(html_path)
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "pipelineRunning" in content
        assert "if (pipelineRunning)" in content
        assert "startBtn.disabled = true" in content
