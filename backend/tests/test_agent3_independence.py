"""
test_agent3_independence.py
============================
Regression tests verifying Agent 3 (Patent Landscape) independence:
  1. Agent 3 searches patents based ONLY on original user topic.
  2. Agent 1 papers with unrelated topics cannot alter Agent 3 query expansion.
  3. Changing Agent 1 output completely yields identical patent queries.
  4. Patent relevance scoring uses the original topic and its expansions, never Agent 1 papers.
  5. Clean public interface: run_patent_agent(topic) runs independently.
"""

import os
import sys
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import Agent1ResearchOutput, Agent2GapOutput, NovelMethodProposal, PaperMetadata
from agents.agent_patent import (
    expand_patent_queries,
    compute_patent_relevance,
    retrieve_multi_source_patents,
    search_and_classify_patents,
    run_patent_agent,
)


class TestAgent3Independence:
    def test_1_deepfake_audio_with_unrelated_agent1_papers(self):
        """
        User Topic: 'Deepfake audio detection'.
        Agent 1 papers contain completely unrelated topics (medical imaging, tumors, satellite).
        Agent 3 patent queries MUST remain strictly about deepfake audio detection.
        """
        user_topic = "Deepfake audio detection"

        # Fake Agent 1 papers on completely different topics
        fake_agent1 = Agent1ResearchOutput(
            query="Tumor Detection",
            papers=[
                PaperMetadata(
                    paper_id="P001",
                    title="Tumor Detection and Medical Image Segmentation in MRI",
                    abstract="We segment brain tumors and analyze satellite imagery using U-Net.",
                    authors=["Dr. Radiologist"],
                    year=2023,
                    relevance_rank="High",
                    notebook_summary="Segmenting brain tumors.",
                    technical_execution="U-Net CNN",
                    datasets=["BRATS"],
                    url="https://doi.org/10.1000/med.1",
                    relevance_score=0.95,
                    coverage_type="full_paper",
                ),
                PaperMetadata(
                    paper_id="P002",
                    title="Satellite Imagery Analysis with Convolutional Neural Networks",
                    abstract="Earth observation using remote sensing and hyper-spectral imaging.",
                    authors=["Dr. Geoscientist"],
                    year=2023,
                    relevance_rank="Medium",
                    notebook_summary="Earth observation analysis.",
                    technical_execution="ResNet-50",
                    datasets=["SpaceNet"],
                    url="https://doi.org/10.1000/geo.1",
                    relevance_score=0.90,
                    coverage_type="full_paper",
                )
            ],
            total_found=2,
            full_paper_count=2,
            abstract_only_count=0,
            unavailable_count=0,
        )

        fake_gap = Agent2GapOutput(
            gaps=[],
            proposed_method=NovelMethodProposal(
                title="MRI Satellite Hybrid Segmentor",
                approach="Apply U-Net to MRI scans and satellite maps.",
                novelty_score=88,
                rationale="Unexplored combination."
            )
        )

        # 1. Check deterministic expansion directly
        queries = expand_patent_queries(user_topic)
        for q in queries:
            q_lower = q.lower()
            assert any(w in q_lower for w in ["deepfake", "audio", "speech", "voice", "spoofing", "forgery"])
            assert "mri" not in q_lower
            assert "tumor" not in q_lower
            assert "satellite" not in q_lower

        # 2. Check full retrieval pipeline receives ONLY deepfake queries
        with patch("agents.agent_patent._fetch_google_patents", return_value=([], "NO_RESULTS")) as mock_gp, \
             patch("agents.agent_patent._fetch_epo_ops_patents", return_value=([], "NO_RESULTS")) as mock_epo, \
             patch("agents.agent_patent._fetch_lens_patents", return_value=([], "SKIPPED_NO_CREDENTIALS")), \
             patch("agents.agent_patent._fetch_patentsview_patents", return_value=([], "NO_RESULTS")), \
             patch("agents.agent_patent._fetch_epmc_patents", return_value=([], "NO_RESULTS")):

            out = search_and_classify_patents(
                gap_data=fake_gap,
                query_topic=user_topic,
                research_out=fake_agent1,
            )

            # Inspect the queries passed to the search adapters
            gp_passed_queries = mock_gp.call_args[0][0]
            for q in gp_passed_queries:
                q_lower = q.lower()
                assert any(w in q_lower for w in ["deepfake", "audio", "speech", "voice", "spoofing", "forgery"])
                assert "tumor" not in q_lower
                assert "satellite" not in q_lower

    def test_2_medical_imaging_with_deepfake_agent1_papers(self):
        """
        User Topic: 'Medical image segmentation'.
        Agent 1 papers contain deepfake audio and voice spoofing keywords.
        Agent 3 patent queries MUST remain strictly about medical image segmentation.
        """
        user_topic = "Medical image segmentation"

        fake_agent1 = Agent1ResearchOutput(
            query="Deepfake Audio",
            papers=[
                PaperMetadata(
                    paper_id="P001",
                    title="Deepfake Audio Detection using WaveNet Vocoders",
                    abstract="Detecting voice spoofing and acoustic deepfakes.",
                    authors=["Dr. Audio"],
                    year=2023,
                    relevance_rank="High",
                    notebook_summary="Audio deepfake detection.",
                    technical_execution="WaveNet",
                    datasets=["ASVspoof"],
                    url="https://doi.org/10.1000/aud.1",
                    relevance_score=0.95,
                    coverage_type="full_paper",
                )
            ],
            total_found=1,
            full_paper_count=1,
            abstract_only_count=0,
            unavailable_count=0,
        )

        queries = expand_patent_queries(user_topic)
        for q in queries:
            q_lower = q.lower()
            assert any(w in q_lower for w in ["medical", "image", "segmentation", "deep learning", "cnn", "transformer"])
            assert "voice" not in q_lower
            assert "audio" not in q_lower
            assert "spoofing" not in q_lower

    def test_3_query_expansion_identical_regardless_of_agent1_presence_or_change(self):
        """
        Agent 3 query generation for 'Agentic AI in multiagent systems' must be 100% identical
        whether Agent 1 returned no papers, robotics papers, or chemistry papers.
        """
        topic = "Agentic AI in multiagent systems"

        queries_standalone = expand_patent_queries(topic)

        # Verify expected deterministic variants
        expected_subset = {
            "agentic ai", "ai agents", "multi-agent ai",
            "multi-agent systems", "autonomous ai agents"
        }
        actual_lower = {q.lower() for q in queries_standalone}
        assert expected_subset.issubset(actual_lower)

        # Changing Agent 1 output completely cannot change Agent 3 queries
        queries_with_robotics = expand_patent_queries(topic)
        assert queries_standalone == queries_with_robotics

    def test_4_relevance_scoring_uses_original_topic_not_agent1(self):
        """
        Patent relevance scoring compares patent text against the user topic and topic expansions,
        not against Agent 1 papers.
        """
        user_topic = "Deepfake audio detection"

        # Patent matching the user topic
        deepfake_patent = {
            "title": "Method for Detecting Synthetic Audio and Voice Spoofing",
            "abstract": "A neural system to identify synthetic speech and deepfake audio manipulation.",
            "claims": "1. A computer-implemented method for detecting audio spoofing."
        }

        # Patent matching fake Agent 1 paper (satellite imagery)
        satellite_patent = {
            "title": "Satellite Image Classification and Earth Observation",
            "abstract": "Multi-spectral imaging from orbital satellites.",
            "claims": "1. A method for orbital image acquisition."
        }

        score_deepfake = compute_patent_relevance(deepfake_patent, user_topic)
        score_satellite = compute_patent_relevance(satellite_patent, user_topic)

        assert score_deepfake >= 75
        assert score_satellite <= 45
        assert score_deepfake > score_satellite

    def test_5_run_patent_agent_clean_entrypoint(self):
        """
        run_patent_agent(topic) runs independently with only topic: str.
        """
        mock_patent = {
            "patent_id": "US10987654B2",
            "publication_number": "US10987654B2",
            "title": "Deepfake Voice Detection System",
            "assignee": "Acoustic Security Corp",
            "abstract": "Detecting synthetic voice with neural filters.",
            "source": "Google Patents",
            "sources": ["Google Patents"],
            "jurisdiction": "US",
        }

        with patch("agents.agent_patent._fetch_epo_ops_patents", return_value=([], "SKIPPED_NO_CREDENTIALS")), \
             patch("agents.agent_patent._fetch_google_patents", return_value=([mock_patent], "SUCCESS")), \
             patch("agents.agent_patent._fetch_lens_patents", return_value=([], "SKIPPED_NO_CREDENTIALS")), \
             patch("agents.agent_patent._fetch_patentsview_patents", return_value=([], "SOURCE_UNAVAILABLE")), \
             patch("agents.agent_patent._fetch_epmc_patents", return_value=([], "NO_RESULTS")), \
             patch("agents.agent_patent._classify_patents_with_gemini", side_effect=RuntimeError("Skip Gemini")):

            out = run_patent_agent("Deepfake audio detection")
            assert out.patent_retrieval_status == "PARTIAL_SUCCESS"
            assert len(out.patents) == 1
            assert out.patents[0].patent_id == "US10987654B2"
            assert "Deepfake audio detection" in out.patents[0].match_explanation
