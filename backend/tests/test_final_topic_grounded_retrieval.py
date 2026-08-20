"""
backend/tests/test_final_topic_grounded_retrieval.py — Final Topic-Grounded Retrieval Tests.

Validates the full set of requirements:
Part A & B: Shared Topic Decomposition
Part C: Agent 1 Research Paper Retrieval, Query Generation, Full-Paper Filtering, and Relevance Gates
Part D: Agent 3 Patent Landscape Discovery, Multi-Source Aggregation, and 3-Stage Relevance Scoring
Part K: Full 16 Regression Test Suite
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topic_decomposition import (
    decompose_research_topic,
    generate_targeted_research_queries,
    generate_targeted_patent_queries,
)
from agents.agent_research import (
    evaluate_paper_hard_gate,
    compute_paper_4factor_relevance,
)
from agents.agent_patent import (
    evaluate_hard_relevance_gates,
    compute_patent_relevance,
)


class TestTopicDecompositionAndQueries:
    def test_multi_agent_decomposition(self):
        decomp = decompose_research_topic("Multi agent in Artificial Intelligence")
        assert decomp["topic_type"] == "MULTI_AGENT_AI"
        assert decomp["requires_multiagent"] is True
        assert any("multi-agent" in a for a in decomp["core_concept_anchors"])
        assert any("reinforcement learning" in m or "coordination" in m for m in decomp["mechanism_anchors"])

    def test_deepfake_audio_decomposition(self):
        decomp = decompose_research_topic("Deepfake audio detection")
        assert decomp["topic_type"] == "DEEPFAKE_AUDIO"
        assert "audio" in decomp["domain_anchors"]
        assert any("deepfake" in c or "synthetic" in c for c in decomp["core_concept_anchors"])

    def test_medical_image_segmentation_decomposition(self):
        decomp = decompose_research_topic("Medical image segmentation using deep learning")
        assert decomp["topic_type"] == "MEDICAL_IMAGE_SEGMENTATION"
        assert any("mri" in d or "medical" in d for d in decomp["domain_anchors"])
        assert any("segmentation" in m or "contouring" in m for m in decomp["mechanism_anchors"])

    def test_targeted_queries_multi_word_not_isolated(self):
        queries = generate_targeted_research_queries("Multi agent in Artificial Intelligence")
        assert len(queries) >= 5
        # Ensure no broad single-word query
        assert "ai" not in [q.lower() for q in queries]
        assert "agent" not in [q.lower() for q in queries]
        assert any("multi-agent" in q.lower() for q in queries)


class TestAgent1PaperRelevanceGates:
    @pytest.fixture
    def multi_agent_decomp(self):
        return decompose_research_topic("Multi agent in Artificial Intelligence")

    def test_generic_ai_fermi_paradox_rejected(self, multi_agent_decomp):
        paper = {
            "title": "Compression, The Fermi Paradox and Artificial Super-Intelligence",
            "abstract": "We analyze computational limits and artificial super-intelligence in the context of the Fermi paradox and SETI observations."
        }
        res = evaluate_paper_hard_gate(paper, "Multi agent in Artificial Intelligence", multi_agent_decomp)
        assert res["passed"] is False
        assert "fermi paradox" in res["rejection_reason"].lower() or "multi-agent" in res["rejection_reason"].lower()

    def test_clinical_ai_without_multiagent_rejected(self, multi_agent_decomp):
        paper = {
            "title": "Artificial Intelligence Framework for Simulating Clinical Decision-Making",
            "abstract": "A supervised deep learning model assists hospital clinicians in diagnosing diagnostic triage pathways."
        }
        res = evaluate_paper_hard_gate(paper, "Multi agent in Artificial Intelligence", multi_agent_decomp)
        assert res["passed"] is False
        assert "multi-agent" in res["rejection_reason"].lower()

    def test_explainable_ai_review_without_multiagent_rejected(self, multi_agent_decomp):
        paper = {
            "title": "Explainable Artificial Intelligence: A Review of Methods and Applications",
            "abstract": "We provide a comprehensive review of SHAP, LIME, and saliency maps for interpretability in neural networks."
        }
        res = evaluate_paper_hard_gate(paper, "Multi agent in Artificial Intelligence", multi_agent_decomp)
        assert res["passed"] is False
        assert "multi-agent" in res["rejection_reason"].lower()

    def test_multi_agent_rl_paper_accepted(self, multi_agent_decomp):
        paper = {
            "title": "Multi-Agent Reinforcement Learning for Decentralized Cooperative Decision Making",
            "abstract": "We propose a multi-agent actor-critic architecture enabling autonomous AI agents to collaborate and communicate under partial observability.",
            "full_text": "Introduction\n\nCooperative multi-agent reinforcement learning (MARL) poses unique challenges..."
        }
        res = evaluate_paper_hard_gate(paper, "Multi agent in Artificial Intelligence", multi_agent_decomp)
        assert res["passed"] is True
        scoring = compute_paper_4factor_relevance(paper, "Multi agent in Artificial Intelligence", multi_agent_decomp)
        assert scoring["score"] >= 0.75
        assert scoring["rank"] in ("High", "Medium")


class Test16PartKPatentRegressionSuite:
    # 1. Autonomous robot -> REJECTED for Multi-Agent AI
    def test_case_1_autonomous_robot_lawn_mowing_rejected(self):
        pat = {
            "patent_id": "US10112233",
            "title": "Autonomous Robot for Lawn Mowing",
            "abstract": "A physical wheeled autonomous robot machine navigating lawns with boundary wire sensors.",
            "claims": "1. An autonomous robotic mowing apparatus."
        }
        gate = evaluate_hard_relevance_gates(pat, "Multi agent in Artificial Intelligence")
        score = compute_patent_relevance(pat, "Multi agent in Artificial Intelligence")
        assert gate["passed"] is False
        assert score == 0

    # 2. Travel agency booking -> REJECTED
    def test_case_2_travel_agency_booking_rejected(self):
        pat = {
            "patent_id": "US9988776",
            "title": "Travel Agency Multi-Agent Booking System",
            "abstract": "A travel booking portal allowing multiple travel agents to book airline tickets and hotel reservations.",
            "claims": "1. A booking system for travel agent personnel."
        }
        gate = evaluate_hard_relevance_gates(pat, "Multi agent in Artificial Intelligence")
        score = compute_patent_relevance(pat, "Multi agent in Artificial Intelligence")
        assert gate["passed"] is False
        assert score == 0

    # 3. Chemical synthesis reactor -> REJECTED
    def test_case_3_chemical_synthesis_reactor_rejected(self):
        pat = {
            "patent_id": "EP3344556",
            "title": "Multi-agent Chemical Synthesis Reactor",
            "abstract": "A chemical synthesis reactor introducing multiple reactant chemical agents to catalyze polymerization.",
            "claims": "1. A chemical vessel comprising multiple catalytic agents."
        }
        gate = evaluate_hard_relevance_gates(pat, "Multi agent in Artificial Intelligence")
        score = compute_patent_relevance(pat, "Multi agent in Artificial Intelligence")
        assert gate["passed"] is False
        assert score == 0

    # 4. Telecom packet routing -> REJECTED / ADJACENT
    def test_case_4_telecom_packet_routing_rejected(self):
        pat = {
            "patent_id": "WO2007113521",
            "title": "Autonomous Packet Routing in Telecommunications Network",
            "abstract": "A telecommunications network with autonomous switching nodes managing TCP packet routing queues.",
            "claims": "1. A telecommunication network routing data packets."
        }
        gate = evaluate_hard_relevance_gates(pat, "Multi agent in Artificial Intelligence")
        score = compute_patent_relevance(pat, "Multi agent in Artificial Intelligence")
        assert gate["passed"] is False
        assert score == 0

    # 5. Multi-agent strategy prediction -> DIRECT_MATCH / STRONG_RELATED (Score >= 70)
    def test_case_5_multi_agent_strategy_prediction_accepted(self):
        pat = {
            "patent_id": "CN114565432A",
            "title": "Multi-agent strategy prediction method and device",
            "abstract": "A multi-agent artificial intelligence system using machine learning and neural networks to predict strategies and actions of multiple intelligent agents in cooperative environments.",
            "claims": "1. A multi-agent strategy prediction method comprising training multiple neural network agents."
        }
        gate = evaluate_hard_relevance_gates(pat, "Multi agent in Artificial Intelligence")
        score = compute_patent_relevance(pat, "Multi agent in Artificial Intelligence")
        assert gate["passed"] is True
        assert gate["relevance_level"] in ("DIRECT_MATCH", "STRONG_RELATED")
        assert score >= 70

    # 6. Multi-agent RL traffic control -> DIRECT_MATCH / STRONG_RELATED
    def test_case_6_multi_agent_rl_traffic_control_accepted(self):
        pat = {
            "patent_id": "US11887766B2",
            "title": "Multi-agent reinforcement learning for cooperative traffic control",
            "abstract": "A distributed artificial intelligence system where multiple software agents coordinate traffic signal phases using multi-agent reinforcement learning.",
            "claims": "1. A traffic control system comprising multiple AI agents communicating over a network."
        }
        gate = evaluate_hard_relevance_gates(pat, "Multi agent in Artificial Intelligence")
        score = compute_patent_relevance(pat, "Multi agent in Artificial Intelligence")
        assert gate["passed"] is True
        assert score >= 70

    # 7. Intelligent multi-agent coordination system -> DIRECT_MATCH / STRONG_RELATED
    def test_case_7_intelligent_multi_agent_coordination_accepted(self):
        pat = {
            "patent_id": "US10998877B2",
            "title": "Intelligent multi-agent coordination system",
            "abstract": "An artificial intelligence architecture for multi-agent coordination, negotiation, and collaborative task planning among autonomous software agents.",
            "claims": "1. An AI system managing inter-agent negotiation protocols."
        }
        gate = evaluate_hard_relevance_gates(pat, "Multi agent in Artificial Intelligence")
        score = compute_patent_relevance(pat, "Multi agent in Artificial Intelligence")
        assert gate["passed"] is True
        assert score >= 70

    # 8. Audio breathing detection -> ADJACENT (excluded, score 0)
    def test_case_8_audio_breathing_sound_detection_adjacent(self):
        pat = {
            "patent_id": "WO2011155048",
            "title": "Audio Breathing Sound Detection Apparatus",
            "abstract": "A microphone apparatus detecting human breathing frequency sounds during sleep.",
            "claims": "1. An audio apparatus detecting acoustic breathing cycles."
        }
        gate = evaluate_hard_relevance_gates(pat, "Deepfake audio detection")
        score = compute_patent_relevance(pat, "Deepfake audio detection")
        assert gate["passed"] is False
        assert gate["relevance_level"] == "ADJACENT"
        assert score == 0

    # 9. Speech intelligibility assessment -> ADJACENT (excluded, score 0)
    def test_case_9_speech_sound_intelligibility_adjacent(self):
        pat = {
            "patent_id": "US2011152708",
            "title": "Speech Sound Intelligibility Assessment Method",
            "abstract": "A method for calculating speech intelligibility metrics from acoustic test signals for hearing aid calibration.",
            "claims": "1. A speech signal intelligibility evaluation method."
        }
        gate = evaluate_hard_relevance_gates(pat, "Deepfake audio detection")
        score = compute_patent_relevance(pat, "Deepfake audio detection")
        assert gate["passed"] is False
        assert gate["relevance_level"] == "ADJACENT"
        assert score == 0

    # 10. Synthetic voice spoofing detection -> DIRECT_MATCH (Score >= 70)
    def test_case_10_synthetic_voice_spoofing_detection_accepted(self):
        pat = {
            "patent_id": "US11223344B2",
            "title": "Synthetic Voice Spoofing Detection using Deep Learning",
            "abstract": "A voice biometric verification apparatus detecting synthetic speech, text-to-speech audio spoofing, and voice conversion deepfakes using neural networks.",
            "claims": "1. An audio processing apparatus classifying authentic vs synthetic speech signals."
        }
        gate = evaluate_hard_relevance_gates(pat, "Deepfake audio detection")
        score = compute_patent_relevance(pat, "Deepfake audio detection")
        assert gate["passed"] is True
        assert gate["relevance_level"] in ("DIRECT_MATCH", "STRONG_RELATED")
        assert score >= 70

    # 11. Audio deepfake verification system -> DIRECT_MATCH (Score >= 70)
    def test_case_11_audio_deepfake_verification_accepted(self):
        pat = {
            "patent_id": "US12334455B2",
            "title": "Audio Deepfake Verification System",
            "abstract": "A system for detecting deepfake audio recordings and identifying acoustic artifacts in synthesized speech.",
            "claims": "1. A system detecting deepfake audio tampering."
        }
        gate = evaluate_hard_relevance_gates(pat, "Deepfake audio detection")
        score = compute_patent_relevance(pat, "Deepfake audio detection")
        assert gate["passed"] is True
        assert gate["relevance_level"] == "DIRECT_MATCH"
        assert score >= 70

    # 12. Facial image deepfake detection -> REJECTED / ADJACENT for audio topic
    def test_case_12_facial_image_deepfake_rejected_for_audio(self):
        pat = {
            "patent_id": "US9988776B2",
            "title": "Facial Image Deepfake Detection",
            "abstract": "A convolutional neural network analyzes visual facial frames to detect face swapping in video recordings.",
            "claims": "1. A video frame analysis method for facial forgery detection."
        }
        gate = evaluate_hard_relevance_gates(pat, "Deepfake audio detection")
        score = compute_patent_relevance(pat, "Deepfake audio detection")
        assert gate["passed"] is False
        assert score == 0

    # 13. Chemical forgery marker -> REJECTED
    def test_case_13_chemical_forgery_marker_rejected(self):
        pat = {
            "patent_id": "WO02053767",
            "title": "Chemical Forgery Marker Detection",
            "abstract": "A chemical substrate used as a marker for document authenticity discrimination.",
            "claims": "1. A luminescent marker for anti-forgery."
        }
        gate = evaluate_hard_relevance_gates(pat, "Deepfake audio detection")
        score = compute_patent_relevance(pat, "Deepfake audio detection")
        assert gate["passed"] is False
        assert score == 0

    # 14. Industrial welding robot -> REJECTED for Medical Image Segmentation
    def test_case_14_industrial_welding_robot_rejected(self):
        pat = {
            "patent_id": "US7766554B2",
            "title": "Industrial Welding Robot Arm",
            "abstract": "An industrial multi-axis robotic arm with optical sensors for automated metal welding.",
            "claims": "1. An industrial welding manipulator."
        }
        gate = evaluate_hard_relevance_gates(pat, "Medical image segmentation using deep learning")
        score = compute_patent_relevance(pat, "Medical image segmentation using deep learning")
        assert gate["passed"] is False
        assert score == 0

    # 15. Medical image viewer without segmentation -> ADJACENT (excluded, score 0)
    def test_case_15_medical_image_viewer_adjacent(self):
        pat = {
            "patent_id": "US8899001B2",
            "title": "Medical Image Viewer with Database Indexing",
            "abstract": "A PACS workstation displaying DICOM radiological images and storing patient diagnostic metadata in an electronic medical record database.",
            "claims": "1. A workstation for displaying medical images."
        }
        gate = evaluate_hard_relevance_gates(pat, "Medical image segmentation using deep learning")
        score = compute_patent_relevance(pat, "Medical image segmentation using deep learning")
        assert gate["passed"] is False
        assert gate["relevance_level"] == "ADJACENT"
        assert score == 0

    # 16. Brain tumor segmentation in MRI -> DIRECT_MATCH (Score >= 80)
    def test_case_16_brain_tumor_mri_segmentation_accepted(self):
        pat = {
            "patent_id": "US9988112B2",
            "title": "Brain Tumor Segmentation in MRI Using Convolutional Neural Networks",
            "abstract": "A deep learning neural network for automatic segmentation and anatomical contouring of brain tumor lesions from volumetric MRI scans.",
            "claims": "1. A medical image processing method for segmenting tumor tissue regions from MRI scans."
        }
        gate = evaluate_hard_relevance_gates(pat, "Medical image segmentation using deep learning")
        score = compute_patent_relevance(pat, "Medical image segmentation using deep learning")
        assert gate["passed"] is True
        assert gate["relevance_level"] == "DIRECT_MATCH"
        assert score >= 80
