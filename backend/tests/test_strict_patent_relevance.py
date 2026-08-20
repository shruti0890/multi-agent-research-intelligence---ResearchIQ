"""
Unit tests for Strict Patent Relevance & Hard Relevance Gate Engine in Agent 3.

Covers all mandatory calibration test cases:
1. Autonomous robot -> reject for Agentic AI (e.g. Robot for Caring for Plants)
2. Autonomous network routing -> reject for Agentic AI
3. AI-controlled robot without agent -> reject for multi-agent (ADJACENT/REJECTED)
4. AI software agent -> pass Agentic AI gate
5. Multi-agent AI coordination -> pass multi-agent gate (DIRECT_MATCH)
6. Travel agent -> reject
7. Chemical agent -> reject
8. Forgery agent -> reject for deepfake audio
9. Deepfake image -> reject for deepfake audio
10. Synthetic speech detection -> pass for deepfake audio (DIRECT_MATCH)
11. Medical image analysis without segmentation -> reject / ADJACENT (excluded from final)
12. Medical MRI segmentation -> pass for medical image segmentation (DIRECT_MATCH)
13. Regression: Speech Sound Intelligibility Assessment -> ADJACENT for deepfake audio (excluded)
14. Regression: Audio Processing and Breathing Detection Device -> ADJACENT for deepfake audio (excluded)
15. Regression: Catheter tip audio locator and cochlear implant -> ADJACENT for deepfake audio (excluded)
16. Speaker Verification Against Synthetic Speech -> pass for deepfake audio (DIRECT_MATCH / STRONG_RELATED)
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent_patent import (
    decompose_topic_concepts,
    evaluate_hard_relevance_gates,
    compute_patent_relevance,
    retrieve_multi_source_patents,
)


class TestStrictRelevanceGates:
    def test_1_autonomous_robot_rejected_for_agentic_ai(self):
        """
        TEST 1: Autonomous physical robot ('Robot for Caring for Plants')
        MUST be REJECTED for 'Agentic AI in multiagent systems' (Score: 0).
        """
        topic = "Agentic AI in multiagent systems"
        robot_patent = {
            "patent_id": "US2008148633",
            "title": "Robot for Caring for Plants",
            "abstract": "An autonomous mobile robot system for inspecting, watering, and caring for household and greenhouse plants with environmental sensors.",
            "claims": "1. An autonomous robot device comprising a water reservoir and motor wheels."
        }

        gate_res = evaluate_hard_relevance_gates(robot_patent, topic)
        score = compute_patent_relevance(robot_patent, topic)

        assert gate_res["passed"] is False
        assert gate_res["relevance_level"] == "REJECTED"
        assert score == 0
        assert "physical robot" in gate_res["rejection_reason"].lower() or "missing ai" in gate_res["rejection_reason"].lower()

    def test_2_autonomous_network_routing_rejected(self):
        """
        TEST 2: Autonomous network routing MUST be REJECTED for 'Agentic AI in multiagent systems'.
        """
        topic = "Agentic AI in multiagent systems"
        telecom_patent = {
            "patent_id": "WO2007113521",
            "title": "AUTONOMOUS SYSTEMS FOR A ROUTING DATA VIA A COMMUNICATIONS NETWORK",
            "abstract": "A communications system is provided where interconnected communications networks are each associated with a control computer arranged to set tariff data for network services.",
            "claims": "1. A communications system for routing data across nodes."
        }

        gate_res = evaluate_hard_relevance_gates(telecom_patent, topic)
        score = compute_patent_relevance(telecom_patent, topic)

        assert gate_res["passed"] is False
        assert gate_res["relevance_level"] == "REJECTED"
        assert score == 0

    def test_3_ai_controlled_robot_without_multiagent_rejected_for_multiagent(self):
        """
        TEST 3: AI-controlled physical robot without multi-agent coordination
        MUST be rejected / marked ADJACENT for 'Agentic AI in multiagent systems'.
        """
        topic = "Agentic AI in multiagent systems"
        ai_robot_patent = {
            "patent_id": "US9988112B2",
            "title": "AI-controlled robotic arm for factory part sorting",
            "abstract": "A convolutional neural network model guides a single robotic manipulator to identify and sort manufactured parts.",
            "claims": "1. A machine learning system for robotic arm trajectory planning."
        }

        gate_res = evaluate_hard_relevance_gates(ai_robot_patent, topic)
        score = compute_patent_relevance(ai_robot_patent, topic)

        assert gate_res["passed"] is False
        assert gate_res["relevance_level"] in ("ADJACENT", "REJECTED")
        assert score == 0

    def test_4_ai_software_agent_passes_agentic_ai_gate(self):
        """
        TEST 4: AI Software Agent passes Agentic AI gate.
        """
        topic = "Agentic AI"
        agent_patent = {
            "patent_id": "US10123456B2",
            "title": "AI Agent for Autonomous Task Planning",
            "abstract": "An autonomous AI software agent powered by a large language model performs multi-step task planning and reasoning.",
            "claims": "1. An intelligent software agent system executing automated task workflows."
        }

        gate_res = evaluate_hard_relevance_gates(agent_patent, topic)
        score = compute_patent_relevance(agent_patent, topic)

        assert gate_res["passed"] is True
        assert gate_res["relevance_level"] in ("DIRECT_MATCH", "STRONG_RELATED")
        assert score >= 70

    def test_5_multi_agent_ai_coordination_passes_direct_match(self):
        """
        TEST 5: Multi-agent AI framework passes as DIRECT_MATCH.
        """
        topic = "Agentic AI in multiagent systems"
        multi_patent = {
            "patent_id": "US12345678B2",
            "title": "Multi-Agent AI Framework for Collaborative Task Planning",
            "abstract": "A multi-agent artificial intelligence framework where autonomous software agents collaborate to plan and execute distributed reasoning tasks using large language models.",
            "claims": "1. A computer-implemented system comprising multiple AI agents executing agent orchestration and communication."
        }

        gate_res = evaluate_hard_relevance_gates(multi_patent, topic)
        score = compute_patent_relevance(multi_patent, topic)

        assert gate_res["passed"] is True
        assert gate_res["relevance_level"] == "DIRECT_MATCH"
        assert score >= 85

    def test_6_travel_agent_rejected(self):
        """
        TEST 6: False positive 'travel agent' MUST be REJECTED.
        """
        topic = "Agentic AI in multiagent systems"
        travel_patent = {
            "patent_id": "US6677889B2",
            "title": "Online reservation and booking system for travel agent",
            "abstract": "A web interface for a travel agent to book flights and hotels for human customers.",
            "claims": "1. A computer system for a travel agent client."
        }

        gate_res = evaluate_hard_relevance_gates(travel_patent, topic)
        score = compute_patent_relevance(travel_patent, topic)

        assert gate_res["passed"] is False
        assert gate_res["relevance_level"] == "REJECTED"
        assert score == 0

    def test_7_chemical_agent_rejected(self):
        """
        TEST 7: False positive 'chemical agent' MUST be REJECTED.
        """
        topic = "Agentic AI in multiagent systems"
        chem_patent = {
            "patent_id": "EP1122334A1",
            "title": "Surfactant cleaning agent composition and method of use",
            "abstract": "A chemical cleaning agent comprising a bleaching agent and a solidifying agent for surface washing.",
            "claims": "1. A chemical agent mixture comprising surfactant compounds."
        }

        gate_res = evaluate_hard_relevance_gates(chem_patent, topic)
        score = compute_patent_relevance(chem_patent, topic)

        assert gate_res["passed"] is False
        assert gate_res["relevance_level"] == "REJECTED"
        assert score == 0

    def test_8_forgery_agent_rejected_for_deepfake_audio(self):
        """
        TEST 8: Chemical / marker forgery agent MUST be REJECTED for deepfake audio.
        """
        topic = "Deepfake audio detection"
        forgery_patent = {
            "patent_id": "WO02053767",
            "title": "MARKER, FORGERY DETECTION AGENT, METHOD OF REAL/FORGERY DISCRIMINATION",
            "abstract": "A marker comprising at least one active ingredient selected among chemiluminescent substrates and bioluminescent substrates as a forgery detection agent.",
            "claims": "1. A forgery detection agent for chemical security marking."
        }

        gate_res = evaluate_hard_relevance_gates(forgery_patent, topic)
        score = compute_patent_relevance(forgery_patent, topic)

        assert gate_res["passed"] is False
        assert gate_res["relevance_level"] == "REJECTED"
        assert score == 0

    def test_9_deepfake_image_rejected_for_deepfake_audio(self):
        """
        TEST 9: Deepfake image / video detection MUST be REJECTED for deepfake audio.
        """
        topic = "Deepfake audio detection"
        img_patent = {
            "patent_id": "US9988776B2",
            "title": "AI-based deepfake image detection and face swap classification",
            "abstract": "A neural network analyzes pixel artifacts in video frames and facial images to detect deepfake images and face swap manipulations.",
            "claims": "1. A method for analyzing optical video frames to identify facial manipulation."
        }

        gate_res = evaluate_hard_relevance_gates(img_patent, topic)
        score = compute_patent_relevance(img_patent, topic)

        assert gate_res["passed"] is False
        assert gate_res["relevance_level"] == "REJECTED"
        assert score == 0

    def test_10_synthetic_speech_detection_passes(self):
        """
        TEST 10: Synthetic speech / voice spoofing detection MUST PASS.
        """
        topic = "Deepfake audio detection"
        audio_patent = {
            "patent_id": "US11223344B2",
            "title": "Deepfake speech detection using neural networks",
            "abstract": "An acoustic verification system analyzes audio waveforms and spectrograms to identify synthetic speech and deepfake voice spoofing.",
            "claims": "1. An audio processing apparatus configured to detect synthetic voice anomalies."
        }

        gate_res = evaluate_hard_relevance_gates(audio_patent, topic)
        score = compute_patent_relevance(audio_patent, topic)

        assert gate_res["passed"] is True
        assert gate_res["relevance_level"] in ("DIRECT_MATCH", "STRONG_RELATED")
        assert score >= 70

    def test_11_medical_image_analysis_without_segmentation_rejected_or_adjacent(self):
        """
        TEST 11: Medical image indexing/viewing without segmentation MUST be marked ADJACENT and excluded.
        """
        topic = "Medical image segmentation using deep learning"
        viewer_patent = {
            "patent_id": "US8899001B2",
            "title": "Medical image viewer and database indexing workstation",
            "abstract": "A clinical workstation for displaying radiological DICOM scans and indexing patient metadata in an electronic medical records database.",
            "claims": "1. A system for retrieving and rendering radiological image files."
        }

        gate_res = evaluate_hard_relevance_gates(viewer_patent, topic)
        score = compute_patent_relevance(viewer_patent, topic)

        assert gate_res["passed"] is False
        assert gate_res["relevance_level"] in ("ADJACENT", "REJECTED")
        assert score == 0

    def test_12_medical_mri_segmentation_passes(self):
        """
        TEST 12: Deep learning MRI segmentation MUST PASS as DIRECT_MATCH.
        """
        topic = "Medical image segmentation using deep learning"
        mri_patent = {
            "patent_id": "US7788990B2",
            "title": "Deep learning system for segmentation of MRI images",
            "abstract": "A deep convolutional neural network automatically performs anatomical organ segmentation and brain tumor delineation on radiological MRI scans.",
            "claims": "1. A medical image processing method for segmenting anatomical regions of interest."
        }

        gate_res = evaluate_hard_relevance_gates(mri_patent, topic)
        score = compute_patent_relevance(mri_patent, topic)

        assert gate_res["passed"] is True
        assert gate_res["relevance_level"] == "DIRECT_MATCH"
        assert score >= 85

    def test_13_speech_sound_intelligibility_is_adjacent_not_strong_related(self):
        """
        TEST 13: 'SYSTEM AND METHOD OF SPEECH SOUND INTELLIGIBILITY ASSESSMENT'
        MUST be ADJACENT (excluded, score 0) for 'Deepfake audio detection'.
        """
        topic = "Deepfake audio detection"
        intelligibility_patent = {
            "patent_id": "US2011152708",
            "title": "SYSTEM AND METHOD OF SPEECH SOUND INTELLIGIBILITY ASSESSMENT, AND PROGRAM THEREOF",
            "abstract": "An assessment apparatus calculates a speech sound intelligibility score from an audio test signal for hearing evaluation.",
            "claims": "1. An assessment method for evaluating speech sound intelligibility."
        }

        gate_res = evaluate_hard_relevance_gates(intelligibility_patent, topic)
        score = compute_patent_relevance(intelligibility_patent, topic)

        assert gate_res["passed"] is False
        assert gate_res["relevance_level"] == "ADJACENT"
        assert score == 0

    def test_14_breathing_detection_device_is_adjacent_not_strong_related(self):
        """
        TEST 14: 'AUDIO PROCESSING DEVICE, AND BREATHING DETECTION DEVICE'
        MUST be ADJACENT (excluded, score 0) for 'Deepfake audio detection'.
        """
        topic = "Deepfake audio detection"
        breathing_patent = {
            "patent_id": "WO2011155048",
            "title": "AUDIO PROCESSING DEVICE, AND BREATHING DETECTION DEVICE",
            "abstract": "An audio processing device detects human breathing acoustic patterns using microphone frequency analysis.",
            "claims": "1. An audio processing device for biological sound detection."
        }

        gate_res = evaluate_hard_relevance_gates(breathing_patent, topic)
        score = compute_patent_relevance(breathing_patent, topic)

        assert gate_res["passed"] is False
        assert gate_res["relevance_level"] == "ADJACENT"
        assert score == 0

    def test_15_catheter_and_cochlear_generic_audio_are_adjacent(self):
        """
        TEST 15: Catheter locating and cochlear implants MUST be ADJACENT for deepfake audio.
        """
        topic = "Deepfake audio detection"
        catheter_patent = {
            "patent_id": "US2012083702",
            "title": "Method for Locating a Catheter Tip Using Audio Detection",
            "abstract": "A medical instrument uses acoustic audio detection to locate catheter position in blood vessels.",
            "claims": "1. An audio detection method for medical catheter tracking."
        }
        cochlear_patent = {
            "patent_id": "US2011106211",
            "title": "Methods and Systems for Presenting an Audio Signal to a Cochlear Implant Patient",
            "abstract": "An audio signal processor shapes auditory stimuli for presentation to a cochlear implant patient.",
            "claims": "1. A system for processing an audio signal for an implant."
        }

        gate_1 = evaluate_hard_relevance_gates(catheter_patent, topic)
        gate_2 = evaluate_hard_relevance_gates(cochlear_patent, topic)

        assert gate_1["passed"] is False
        assert gate_1["relevance_level"] in ("ADJACENT", "REJECTED")
        assert compute_patent_relevance(catheter_patent, topic) == 0

        assert gate_2["passed"] is False
        assert gate_2["relevance_level"] in ("ADJACENT", "REJECTED")
        assert compute_patent_relevance(cochlear_patent, topic) == 0

    def test_16_speaker_verification_against_synthetic_speech_passes(self):
        """
        TEST 16: Speaker Verification Against Synthetic Speech MUST PASS.
        """
        topic = "Deepfake audio detection"
        auth_patent = {
            "patent_id": "US10998877B2",
            "title": "Speaker Verification and Voice Biometric Anti-Spoofing System",
            "abstract": "A voice biometric verification apparatus detects audio replay attacks and authenticates speaker identity against synthetic voice spoofing.",
            "claims": "1. A voice authentication system comprising anti-spoofing classifiers."
        }

        gate_res = evaluate_hard_relevance_gates(auth_patent, topic)
        score = compute_patent_relevance(auth_patent, topic)

        assert gate_res["passed"] is True
        assert gate_res["relevance_level"] in ("DIRECT_MATCH", "STRONG_RELATED")
        assert score >= 70
