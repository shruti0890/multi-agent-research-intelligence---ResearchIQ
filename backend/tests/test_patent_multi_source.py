"""
Tests for Phase 2: Robust Multi-Source Patent Retrieval in Agent 3.

Covers:
- Missing credentials (skipped sources)
- HTTP 401 and 403 authentication failures
- DNS failure and network connection errors
- API timeouts
- Zero results handling
- Successful retrieval and common schema normalization
- Duplicate patent deduplication
- Multi-source record merging with source tracking (sources = [EPO, Lens])
- Deterministic query expansion (domain-aware 3-6 queries, no Gemini)
- Deterministic relevance scoring (40% Title, 35% Abstract, 25% Claims, configurable)
- Gemini failure handling (patent_analysis_status = retrieval_success_synthesis_failed)
- Real patent preservation across Gemini failures
- Partial success status
- Overall RETRIEVAL_FAILED status when all sources fail or error
"""
import sys
import os
import json
import urllib.error
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import Agent2GapOutput, NovelMethodProposal, Agent3PatentOutput, PatentInfo
from agents.agent_patent import (
    expand_patent_queries,
    compute_patent_relevance,
    deduplicate_and_merge_patents,
    normalize_patent_id,
    retrieve_multi_source_patents,
    search_and_classify_patents,
    _fetch_google_patents,
    _fetch_epo_ops_patents,
    _fetch_lens_patents,
    _fetch_patentsview_patents,
    _fetch_epmc_patents,
)


# ===========================================================================
# 1. Deterministic Query Expansion Tests
# ===========================================================================
class TestQueryExpansion:
    def test_query_expansion_deepfake_audio(self):
        topic = "Deepfake audio detection"
        queries = expand_patent_queries(topic)
        assert 5 <= len(queries) <= 8
        assert "Deepfake audio detection" in queries or "deepfake audio detection" in [q.lower() for q in queries]
        queries_str = " ".join(queries).lower()
        assert any(term in queries_str for term in ["synthetic speech", "voice spoofing", "audio forgery", "synthetic voice", "speech manipulation"])

    def test_query_expansion_agentic_ai_multiagent(self):
        topic = "Agentic AI in Multiagent Systems"
        queries = expand_patent_queries(topic)
        assert 5 <= len(queries) <= 8
        queries_str = " ".join(queries).lower()
        assert any(term in queries_str for term in ["agentic ai", "ai agents", "autonomous agents", "multi-agent systems", "agent orchestration"])

    def test_query_expansion_deterministic(self):
        topic = "Deepfake audio detection"
        q1 = expand_patent_queries(topic)
        q2 = expand_patent_queries(topic)
        assert q1 == q2


# ===========================================================================
# 2. Google Patents Fetcher Tests
# ===========================================================================
class TestGooglePatentsFetcher:
    def test_google_patents_successful_parsing(self):
        mock_gp_data = {
            "results": {
                "total_num_results": 150,
                "cluster": [
                    {
                        "result": [
                            {
                                "id": "patent/US11694694B2/en",
                                "patent": {
                                    "publication_number": "US11694694B2",
                                    "title": "Detecting <b>deep-fake</b> audio through vocal tract reconstruction",
                                    "snippet": "A method is provided for identifying synthetic &quot;deep-fake&quot; audio samples.",
                                    "assignee": "University Of Florida Research Foundation, Incorporated",
                                    "inventor": "Patrick G. Traynor",
                                    "filing_date": "2021-07-27",
                                    "priority_date": "2020-07-30"
                                }
                            }
                        ]
                    }
                ]
            }
        }
        with patch("urllib.request.urlopen") as mock_url:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(mock_gp_data).encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp
            mock_url.return_value = mock_resp

            patents, status = _fetch_google_patents(["deepfake audio"])
            assert status == "SUCCESS"
            assert len(patents) == 1
            p = patents[0]
            assert p["patent_id"] == "US11694694B2"
            assert p["publication_number"] == "US11694694B2"
            assert p["title"] == "Detecting deep-fake audio through vocal tract reconstruction"
            assert "synthetic \"deep-fake\" audio samples" in p["abstract"]
            assert p["assignee"] == "University Of Florida Research Foundation, Incorporated"
            assert p["inventors"] == ["Patrick G. Traynor"]
            assert p["source"] == "Google Patents"
            assert p["sources"] == ["Google Patents"]
            assert p["source_url"] == "https://patents.google.com/patent/US11694694B2/en"
            assert p["priority_date"] == "2020-07-30"
            assert p["publication_date"] == "2021-07-27"
            assert p["jurisdiction"] == "US"

    def test_google_patents_zero_results(self):
        mock_gp_data = {"results": {"cluster": []}}
        with patch("urllib.request.urlopen") as mock_url:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(mock_gp_data).encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp
            mock_url.return_value = mock_resp

            patents, status = _fetch_google_patents(["xyznonexistentquery123"])
            assert patents == []
            assert status == "NO_RESULTS"

    def test_google_patents_403_auth_or_blocked(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError("url", 403, "Forbidden", {}, None)):
            patents, status = _fetch_google_patents(["deepfake audio"])
            assert patents == []
            assert status in ("AUTHENTICATION_FAILED", "SOURCE_UNAVAILABLE")

    def test_google_patents_429_rate_limited(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError("url", 429, "Too Many Requests", {}, None)):
            patents, status = _fetch_google_patents(["deepfake audio"])
            assert patents == []
            assert status == "RATE_LIMITED"

    def test_google_patents_timeout(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("Timed out")):
            patents, status = _fetch_google_patents(["deepfake audio"])
            assert patents == []
            assert status == "SOURCE_UNAVAILABLE"


# ===========================================================================
# 3. Missing Credentials Handling Tests
# ===========================================================================
class TestMissingCredentials:
    def test_epo_missing_credentials_skipped(self, monkeypatch):
        monkeypatch.delenv("EPO_OPS_CONSUMER_KEY", raising=False)
        monkeypatch.delenv("EPO_OPS_CONSUMER_SECRET", raising=False)
        patents, status = _fetch_epo_ops_patents(["deepfake audio"])
        assert patents == []
        assert status == "SKIPPED_NO_CREDENTIALS"

    def test_lens_missing_token_skipped(self, monkeypatch):
        monkeypatch.delenv("LENS_API_TOKEN", raising=False)
        patents, status = _fetch_lens_patents(["deepfake audio"])
        assert patents == []
        assert status == "SKIPPED_NO_CREDENTIALS"

    def test_pipeline_runs_with_all_credentials_missing(self, monkeypatch):
        monkeypatch.delenv("EPO_OPS_CONSUMER_KEY", raising=False)
        monkeypatch.delenv("EPO_OPS_CONSUMER_SECRET", raising=False)
        monkeypatch.delenv("LENS_API_TOKEN", raising=False)
        monkeypatch.delenv("PATENTSVIEW_API_KEY", raising=False)

        mock_gp_data = {
            "results": {
                "cluster": [
                    {
                        "result": [
                            {
                                "id": "patent/US11694694B2/en",
                                "patent": {
                                    "publication_number": "US11694694B2",
                                    "title": "Detecting deep-fake audio",
                                    "snippet": "Synthetic voice detection methods.",
                                    "assignee": "Research Foundation"
                                }
                            }
                        ]
                    }
                ]
            }
        }
        with patch("urllib.request.urlopen") as mock_url:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(mock_gp_data).encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp
            mock_url.return_value = mock_resp

            patents, statuses, overall_status = retrieve_multi_source_patents("Deepfake audio detection")
            assert statuses["EPO"] == "SKIPPED_NO_CREDENTIALS"
            assert statuses["Lens"] == "SKIPPED_NO_CREDENTIALS"
            assert statuses["Google Patents"] == "SUCCESS"
            assert len(patents) >= 1
            assert overall_status in ("SUCCESS", "PARTIAL_SUCCESS")


# ===========================================================================
# 4. HTTP 401 & 403 Authentication Failure Tests
# ===========================================================================
class TestAuthFailures:
    def test_epo_auth_401(self, monkeypatch):
        monkeypatch.setenv("EPO_OPS_CONSUMER_KEY", "invalid_key")
        monkeypatch.setenv("EPO_OPS_CONSUMER_SECRET", "invalid_secret")
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError("url", 401, "Unauthorized", {}, None)):
            patents, status = _fetch_epo_ops_patents(["deepfake audio"])
            assert patents == []
            assert status == "AUTHENTICATION_FAILED"

    def test_lens_auth_403(self, monkeypatch):
        monkeypatch.setenv("LENS_API_TOKEN", "bad_token")
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError("url", 403, "Forbidden", {}, None)):
            patents, status = _fetch_lens_patents(["deepfake audio"])
            assert patents == []
            assert status == "AUTHENTICATION_FAILED"

    def test_patentsview_auth_401(self, monkeypatch):
        monkeypatch.setenv("PATENTSVIEW_API_KEY", "bad_key")
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError("url", 401, "Unauthorized", {}, None)):
            patents, status = _fetch_patentsview_patents(["deepfake audio"])
            assert patents == []
            assert status == "AUTHENTICATION_FAILED"


# ===========================================================================
# 5. DNS Failure & Timeout Tests
# ===========================================================================
class TestNetworkFailures:
    def test_patentsview_dns_failure(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Name or service not known")):
            patents, status = _fetch_patentsview_patents(["deepfake audio"])
            assert patents == []
            assert status == "SOURCE_UNAVAILABLE"

    def test_epmc_timeout(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("Request timed out")):
            patents, status = _fetch_epmc_patents(["deepfake audio"])
            assert patents == []
            assert status == "SOURCE_UNAVAILABLE"


# ===========================================================================
# 6. Zero Results Handling Tests
# ===========================================================================
class TestZeroResults:
    def test_epmc_zero_results(self):
        mock_data = {"resultList": {"result": []}}
        with patch("urllib.request.urlopen") as mock_url:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(mock_data).encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp
            mock_url.return_value = mock_resp

            patents, status = _fetch_epmc_patents(["xyz123nonexistenttopic"])
            assert patents == []
            assert status == "NO_RESULTS"


# ===========================================================================
# 7. Deduplication & Multi-Source Merging Tests
# ===========================================================================
class TestDeduplicationAndMerging:
    def test_id_normalization(self):
        assert normalize_patent_id("US-11223344-B2") == "US11223344B2"
        assert normalize_patent_id("EP 3456789 A1") == "EP3456789A1"

    def test_merge_same_patent_from_google_and_epo(self):
        raw_patents = [
            {
                "patent_id": "US11223344B2",
                "publication_number": "US-11223344-B2",
                "title": "Deepfake audio detection system",
                "assignee": "Tech Corp",
                "applicant": "Tech Corp",
                "abstract": "Abstract from Google Patents",
                "claims": "",
                "source": "Google Patents",
                "sources": ["Google Patents"],
                "jurisdiction": "US",
            },
            {
                "patent_id": "US11223344B2",
                "publication_number": "US11223344B2",
                "title": "Deepfake audio detection system",
                "assignee": "Tech Corp",
                "applicant": "Tech Corp",
                "abstract": "Abstract from EPO",
                "claims": "1. An acoustic verification claim from EPO.",
                "source": "EPO",
                "sources": ["EPO"],
                "jurisdiction": "US",
            }
        ]
        merged = deduplicate_and_merge_patents(raw_patents)
        assert len(merged) == 1
        p = merged[0]
        assert p["patent_id"] == "US11223344B2"
        assert sorted(p["sources"]) == ["EPO", "Google Patents"]
        assert "Google Patents" in p["abstract"] or "EPO" in p["abstract"]
        assert "acoustic verification claim" in p["claims"]


# ===========================================================================
# 8. Deterministic Relevance Scoring Tests
# ===========================================================================
class TestRelevanceScoring:
    def test_scoring_weights_title_abstract_claims(self):
        topic = "Deepfake audio detection"
        relevant_patent = {
            "title": "System for deepfake audio detection using neural networks",
            "abstract": "We detect synthetic speech and audio spoofing using deep convolutional networks.",
            "claims": "1. A method for deepfake audio detection."
        }
        score = compute_patent_relevance(relevant_patent, topic)
        assert 70 <= score <= 98

    def test_irrelevant_patent_scores_lower(self):
        topic = "Deepfake audio detection"
        relevant = {
            "title": "Deepfake audio detection system",
            "abstract": "Detecting synthetic voice anomalies in acoustic audio.",
            "claims": "1. Method for deepfake detection."
        }
        irrelevant = {
            "title": "Semiconductor manufacturing apparatus and cooling chamber",
            "abstract": "A wafer cooling chamber for high temperature plasma etching processes.",
            "claims": "1. An apparatus for etching silicon wafers."
        }
        score_rel = compute_patent_relevance(relevant, topic)
        score_irrel = compute_patent_relevance(irrelevant, topic)
        assert score_rel > score_irrel


# ===========================================================================
# 9. Failure Differentiation & Status Tests
# ===========================================================================
class TestFailureDifferentiation:
    def test_all_apis_fail_reports_retrieval_failed(self):
        with patch("agents.agent_patent._fetch_google_patents", return_value=([], "SOURCE_UNAVAILABLE")), \
             patch("agents.agent_patent._fetch_epmc_patents", return_value=([], "NO_RESULTS")), \
             patch("agents.agent_patent._fetch_patentsview_patents", return_value=([], "SOURCE_UNAVAILABLE")), \
             patch("agents.agent_patent._fetch_lens_patents", return_value=([], "AUTHENTICATION_FAILED")), \
             patch("agents.agent_patent._fetch_epo_ops_patents", return_value=([], "SOURCE_UNAVAILABLE")):

            patents, statuses, overall_status = retrieve_multi_source_patents("Deepfake audio detection")
            assert patents == []
            assert overall_status in ("RETRIEVAL_FAILED", "SOURCE_UNAVAILABLE")

    def test_partial_success_when_some_succeed_and_some_fail(self):
        success_patent = {
            "patent_id": "EP1234567A1",
            "publication_number": "EP1234567A1",
            "title": "Synthetic speech detection",
            "assignee": "Fraunhofer",
            "abstract": "Detecting synthetic speech.",
            "source": "EuropePMC",
            "sources": ["EuropePMC"],
            "jurisdiction": "EP"
        }
        with patch("agents.agent_patent._fetch_epmc_patents", return_value=([success_patent], "SUCCESS")), \
             patch("agents.agent_patent._fetch_patentsview_patents", return_value=([], "SOURCE_UNAVAILABLE")), \
             patch("agents.agent_patent._fetch_lens_patents", return_value=([], "AUTHENTICATION_FAILED")), \
             patch("agents.agent_patent._fetch_epo_ops_patents", return_value=([], "SOURCE_UNAVAILABLE")), \
             patch("agents.agent_patent._fetch_google_patents", return_value=([], "SOURCE_UNAVAILABLE")):

            patents, statuses, overall_status = retrieve_multi_source_patents("Deepfake audio detection")
            assert len(patents) == 1
            assert overall_status == "PARTIAL_SUCCESS"


# ===========================================================================
# 10. Gemini Failure Fallback & Real Patent Preservation Tests
# ===========================================================================
class TestGeminiFailureFallback:
    def test_gemini_failure_preserves_real_patents(self):
        gap_data = Agent2GapOutput(
            gaps=[],
            proposed_method=NovelMethodProposal(
                title="Spectral-Temporal Dual Classifier",
                approach="Combine spectrogram analysis with temporal attention",
                novelty_score=88,
                rationale="Novel cross-domain fusion"
            )
        )
        mock_patent = {
            "patent_id": "US11223344B2",
            "publication_number": "US-11223344-B2",
            "title": "Audio Deepfake Verification System",
            "assignee": "Cyber Security Corp",
            "applicant": "Cyber Security Corp",
            "inventors": ["Dr. Smith"],
            "abstract": "A deep learning neural network system for acoustic deepfake audio verification.",
            "claims": "1. A system comprising an audio feature extractor.",
            "publication_date": "2023-01-15",
            "jurisdiction": "US",
            "source": "EuropePMC",
            "sources": ["EuropePMC", "Lens"],
            "source_url": "https://patents.google.com/patent/US11223344B2/en",
            "url": "https://patents.google.com/patent/US11223344B2/en",
            "relevance_score": 85,
        }

        # Mock retrieval to return the real patent and mock Gemini to raise an exception
        with patch("agents.agent_patent.retrieve_multi_source_patents", return_value=([mock_patent], {"EuropePMC": "SUCCESS"}, "SUCCESS")), \
             patch("agents.agent_patent._classify_patents_with_gemini", side_effect=RuntimeError("Gemini Quota Exceeded")):

            output = search_and_classify_patents(gap_data, "Deepfake audio detection")

            assert output.patent_analysis_status == "retrieval_success_synthesis_failed"
            assert output.patent_retrieval_status == "SUCCESS"
            assert len(output.patents) == 1
            p = output.patents[0]
            assert p.patent_id == "US11223344B2"
            assert p.title == "Audio Deepfake Verification System"
            assert p.assignee == "Cyber Security Corp"
            assert p.abstract == mock_patent["abstract"]
            assert sorted(p.sources) == ["EuropePMC", "Lens"]
            assert len(output.retrieved_patents_raw) == 1
            assert len(output.white_space_opportunities) == 3

    def test_retrieval_failed_pipeline_output(self):
        gap_data = Agent2GapOutput(
            gaps=[],
            proposed_method=NovelMethodProposal(title="T", approach="A", novelty_score=10, rationale="R")
        )
        with patch("agents.agent_patent.retrieve_multi_source_patents", return_value=([], {"EuropePMC": "NO_RESULTS", "Lens": "SOURCE_UNAVAILABLE"}, "RETRIEVAL_FAILED")):
            output = search_and_classify_patents(gap_data, "Deepfake audio detection")
            assert output.patent_analysis_status == "retrieval_failed"
            assert output.patent_retrieval_status == "RETRIEVAL_FAILED"
            assert output.patents == []
            assert len(output.white_space_opportunities) == 3
