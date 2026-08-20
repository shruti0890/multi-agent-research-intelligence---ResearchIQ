"""
backend/tests/test_agent1_strict_relevance.py — Unit tests for Agent 1 Gate-First Relevance & Full-Text Validation.

Covers all 15 required test cases from Section 18:
1. Multi-agent RL survey -> ACCEPT
2. Multi-agent LLM paper -> ACCEPT
3. Multi-agent distributed AI paper -> ACCEPT
4. Single-agent RL -> REJECT
5. Generic AI paper -> REJECT
6. Explainable AI without agents -> REJECT
7. Clinical AI without agents -> REJECT
8. Fermi paradox AI paper without agents -> REJECT
9. Political economy of sanctions -> REJECT (final score 0, status REJECTED_OFF_TOPIC)
10. Autonomous robot -> REJECT
11. Autonomous network routing -> REJECT
12. Abstract-only paper -> EXCLUDED
13. unknown + abstract only -> EXCLUDED
14. Full paper with unusual section names -> ACCEPTABLE
15. Topic spelling normalization -> PASS
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topic_decomposition import (
    normalize_research_topic,
    decompose_research_topic,
    generate_targeted_research_queries
)
from agents.agent_research import (
    validate_full_paper_content,
    evaluate_paper_hard_gate,
    compute_paper_4factor_relevance
)


class TestAgent1StrictTopicRelevance:
    @pytest.fixture
    def multi_agent_decomp(self):
        topic = "Multi-agent artificial intelligence"
        return decompose_research_topic(topic)

    # 1. Multi-agent RL survey -> ACCEPT
    def test_1_multi_agent_rl_survey_accepted(self, multi_agent_decomp):
        paper = {
            "title": "A Survey of Multi-Agent Deep Reinforcement Learning",
            "abstract": "We provide a comprehensive survey of multi-agent deep reinforcement learning (MADRL) algorithms, cooperative and competitive agent environments, and multi-agent coordination frameworks.",
            "full_text": "Introduction\n\nMulti-agent reinforcement learning has gained significant traction..."
        }
        gate = evaluate_paper_hard_gate(paper, "Multi-agent artificial intelligence", multi_agent_decomp)
        scoring = compute_paper_4factor_relevance(paper, "Multi-agent artificial intelligence", multi_agent_decomp, gate_res=gate)

        assert gate["passed"] is True
        assert gate["relevance_level"] in ("DIRECT_MATCH", "STRONG_RELATED")
        assert scoring["score"] >= 0.85
        assert scoring["relevance_status"] == "ACCEPTED"

    # 2. Multi-agent LLM paper -> ACCEPT
    def test_2_triz_agents_multi_agent_llm_accepted(self, multi_agent_decomp):
        paper = {
            "title": "TRIZ Agents: A Multi-Agent LLM Approach for Systematic Innovation and Problem Solving",
            "abstract": "We present TRIZ Agents, an autonomous multi-agent framework where multiple large language model agents collaborate to perform problem decomposition, conflict analysis, and creative engineering reasoning.",
            "full_text": "1. Introduction\n\nRecent advances in LLM agents have enabled multi-agent architectures..."
        }
        gate = evaluate_paper_hard_gate(paper, "Multi-agent artificial intelligence", multi_agent_decomp)
        scoring = compute_paper_4factor_relevance(paper, "Multi-agent artificial intelligence", multi_agent_decomp, gate_res=gate)

        assert gate["passed"] is True
        assert gate["relevance_level"] == "DIRECT_MATCH"
        assert scoring["score"] >= 0.85
        assert scoring["relevance_status"] == "ACCEPTED"

    # 3. Multi-agent distributed AI paper -> ACCEPT
    def test_3_multi_agent_systems_distributed_ai_accepted(self, multi_agent_decomp):
        paper = {
            "title": "Multi-Agent Systems: An Introduction to Distributed Artificial Intelligence",
            "abstract": "This work explores multi-agent systems within distributed artificial intelligence, detailing inter-agent communication, negotiation protocols, agent coordination, and collaborative decision making.",
            "full_text": "Chapter 1: Multi-Agent Architectures\n\nDistributed artificial intelligence focuses on..."
        }
        gate = evaluate_paper_hard_gate(paper, "Multi-agent artificial intelligence", multi_agent_decomp)
        scoring = compute_paper_4factor_relevance(paper, "Multi-agent artificial intelligence", multi_agent_decomp, gate_res=gate)

        assert gate["passed"] is True
        assert gate["relevance_level"] == "DIRECT_MATCH"
        assert scoring["score"] >= 0.85

    # 4. Single-agent RL -> REJECT
    def test_4_single_agent_rl_rejected(self, multi_agent_decomp):
        paper = {
            "title": "Deep Q-Learning for Single Robotic Arm Trajectory Optimization",
            "abstract": "We train a single deep Q-network to optimize trajectory and torque control for a single industrial manipulator without any multi-agent interactions.",
            "full_text": "Methodology\n\nThe Q-learning agent observes the single arm joint angles..."
        }
        gate = evaluate_paper_hard_gate(paper, "Multi-agent artificial intelligence", multi_agent_decomp)
        scoring = compute_paper_4factor_relevance(paper, "Multi-agent artificial intelligence", multi_agent_decomp, gate_res=gate)

        assert gate["passed"] is False
        assert scoring["score"] == 0.0
        assert scoring["relevance_status"] == "REJECTED_OFF_TOPIC"

    # 5. Generic AI paper -> REJECT
    def test_5_generic_ai_paper_rejected(self, multi_agent_decomp):
        paper = {
            "title": "Advances in Artificial Intelligence: Survey of Deep Learning Architectures",
            "abstract": "A broad survey of convolutional networks, transformers, and recurrent architectures in modern artificial intelligence systems.",
            "full_text": "Overview\n\nDeep learning has revolutionized pattern recognition..."
        }
        gate = evaluate_paper_hard_gate(paper, "Multi-agent artificial intelligence", multi_agent_decomp)
        scoring = compute_paper_4factor_relevance(paper, "Multi-agent artificial intelligence", multi_agent_decomp, gate_res=gate)

        assert gate["passed"] is False
        assert scoring["score"] == 0.0

    # 6. Explainable AI without agents -> REJECT
    def test_6_explainable_ai_without_agents_rejected(self, multi_agent_decomp):
        paper = {
            "title": "Explainable Artificial Intelligence: A Review of Methods and Applications",
            "abstract": "We evaluate SHAP, LIME, and gradient-based attribution maps for interpretable machine learning across tabular and computer vision benchmarks.",
            "full_text": "1. Introduction\n\nInterpretability in black-box models is essential..."
        }
        gate = evaluate_paper_hard_gate(paper, "Multi-agent artificial intelligence", multi_agent_decomp)
        scoring = compute_paper_4factor_relevance(paper, "Multi-agent artificial intelligence", multi_agent_decomp, gate_res=gate)

        assert gate["passed"] is False
        assert scoring["score"] == 0.0

    # 7. Clinical AI without agents -> REJECT
    def test_7_clinical_ai_without_agents_rejected(self, multi_agent_decomp):
        paper = {
            "title": "Artificial Intelligence Framework for Clinical Decision Making in Hospital Triage",
            "abstract": "A supervised neural network classifies patient vital signs to assist clinicians in emergency room triage pathways.",
            "full_text": "Methods\n\nPatient records were preprocessed and fed into a dense network..."
        }
        gate = evaluate_paper_hard_gate(paper, "Multi-agent artificial intelligence", multi_agent_decomp)
        scoring = compute_paper_4factor_relevance(paper, "Multi-agent artificial intelligence", multi_agent_decomp, gate_res=gate)

        assert gate["passed"] is False
        assert scoring["score"] == 0.0

    # 8. Fermi paradox AI paper without agents -> REJECT
    def test_8_fermi_paradox_rejected(self, multi_agent_decomp):
        paper = {
            "title": "Artificial Super-Intelligence and the Fermi Paradox",
            "abstract": "We discuss universal computational limits, super-intelligence development timescales, and the Fermi paradox in SETI research.",
            "full_text": "Section 1\n\nConsidering the vast age of the galaxy..."
        }
        gate = evaluate_paper_hard_gate(paper, "Multi-agent artificial intelligence", multi_agent_decomp)
        scoring = compute_paper_4factor_relevance(paper, "Multi-agent artificial intelligence", multi_agent_decomp, gate_res=gate)

        assert gate["passed"] is False
        assert scoring["score"] == 0.0

    # 9. Explicit Negative Example: Political economy of sanctions on Iran -> REJECT
    def test_9_political_economy_of_sanctions_rejected(self, multi_agent_decomp):
        paper = {
            "title": "The Political Economy of Sanctions on Iran: Mechanism D...",
            "abstract": "We analyze macroeconomic impacts, petroleum trade flows, and geopolitical policy mechanisms surrounding international economic sanctions on Iran.",
            "full_text": "1. Introduction\n\nEconomic sanctions have significantly altered the macroeconomic equilibrium..."
        }
        gate = evaluate_paper_hard_gate(paper, "Multi-agent artificial intelligence", multi_agent_decomp)
        scoring = compute_paper_4factor_relevance(paper, "Multi-agent artificial intelligence", multi_agent_decomp, gate_res=gate)

        assert gate["passed"] is False
        assert scoring["score"] == 0.0
        assert scoring["relevance_status"] == "REJECTED_OFF_TOPIC"
        assert "no meaningful relationship" in gate["rejection_reason"].lower() or "excluded" in gate["rejection_reason"].lower()

    # 10. Autonomous robot without multi-agent coordination -> REJECT
    def test_10_autonomous_robot_without_multiagent_rejected(self, multi_agent_decomp):
        paper = {
            "title": "Autonomous Robot for Plant Care and Soil Moisture Monitoring",
            "abstract": "An autonomous physical ground robot navigates greenhouse rows using lidar to measure soil moisture and water individual potted plants.",
            "full_text": "System Architecture\n\nThe wheeled robotic chassis carries a water tank..."
        }
        gate = evaluate_paper_hard_gate(paper, "Multi-agent artificial intelligence", multi_agent_decomp)
        scoring = compute_paper_4factor_relevance(paper, "Multi-agent artificial intelligence", multi_agent_decomp, gate_res=gate)

        assert gate["passed"] is False
        assert scoring["score"] == 0.0

    # 11. Autonomous network routing without multi-agent AI -> REJECT
    def test_11_autonomous_network_routing_rejected(self, multi_agent_decomp):
        paper = {
            "title": "Autonomous Packet Routing in Telecommunications Network Queues",
            "abstract": "An autonomous switching protocol minimizes latency across cellular telecommunications network nodes using static routing tables.",
            "full_text": "Protocol Specification\n\nData packets are inspected at ingress routers..."
        }
        gate = evaluate_paper_hard_gate(paper, "Multi-agent artificial intelligence", multi_agent_decomp)
        scoring = compute_paper_4factor_relevance(paper, "Multi-agent artificial intelligence", multi_agent_decomp, gate_res=gate)

        assert gate["passed"] is False
        assert scoring["score"] == 0.0


class TestAgent1FullTextValidation:
    # 12. Abstract-only paper -> EXCLUDED
    def test_12_abstract_only_paper_excluded(self):
        text = "Abstract: This is an abstract only text with no research body. " * 5
        is_valid, reason, stats = validate_full_paper_content(text)
        assert is_valid is False
        assert "abstract" in reason.lower() or "below minimum" in reason.lower() or "insufficient research" in reason.lower()

    # 13. unknown + abstract only (OpenAlex short wrapper) -> EXCLUDED
    def test_13_unknown_and_abstract_only_excluded(self):
        text = "Abstract\n\nThis paper presents a theoretical review of artificial intelligence...\n\nPublisher: Elsevier Volume 12 Issue 3 Terms and conditions apply."
        is_valid, reason, stats = validate_full_paper_content(text)
        assert is_valid is False
        assert "only abstract" in reason or "research body tokens" in reason or "below minimum" in reason

    # 14. Full paper with unusual section names but substantial research body -> ACCEPTABLE
    def test_14_full_paper_unusual_sections_acceptable(self):
        # 1200+ words of real research content in structured sections
        body = (
            "System Architecture and Multi-Agent Design\n\n"
            + ("The multi-agent coordination protocol enables distributed AI agents to communicate via message passing. " * 50)
            + "\n\nExperimental Evaluation and Benchmark Results\n\n"
            + ("We evaluated the multi-agent reinforcement learning algorithm across cooperative games with high reward. " * 50)
        )
        is_valid, reason, stats = validate_full_paper_content(body)
        assert is_valid is True
        assert stats["research_tokens"] >= 1000

    # 15. Topic spelling normalization -> PASS
    def test_15_topic_spelling_normalization(self):
        t1 = "Multi agents in Artificial Intelligience"
        norm1 = normalize_research_topic(t1)
        assert norm1 == "Multi-agent artificial intelligence"

        t2 = "Multi agents in Intelligience Artificial"
        norm2 = normalize_research_topic(t2)
        assert norm2 == "Multi-agent artificial intelligence"

        t3 = "Deeepfake audio detection"
        norm3 = normalize_research_topic(t3)
        assert norm3 == "Deepfake audio detection"
