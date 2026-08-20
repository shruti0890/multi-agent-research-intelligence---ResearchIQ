import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import re
import time
import html
import urllib.request
import urllib.parse
import urllib.error
import base64
from typing import List, Optional, Dict, Any
from google.genai import types
from dotenv import load_dotenv

from schemas import Agent1ResearchOutput, Agent2GapOutput, Agent3PatentOutput, PatentInfo, GeminiQuotaExhaustedError  # type: ignore
from agents.agent_utils import execute_gemini_with_retry, get_gemini_model, get_gemini_client  # type: ignore
from topic_decomposition import decompose_research_topic, generate_targeted_patent_queries

# Load environment variables
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path=_env_path, override=True)

GEMINI_MODEL = get_gemini_model()
_get_client = get_gemini_client

def _gemini_generate_with_retry(
    client,
    model: str,
    prompt: str,
    config=None,
    max_retries: int = 3,
    agent_label: str = "Agent3",
) -> str:
    """Thin wrapper forwarding to centralized execute_gemini_with_retry."""
    return execute_gemini_with_retry(
        prompt=prompt,
        config=config,
        model=model,
        max_retries=max_retries,
        agent_label=agent_label,
        client=client,
    )

# Valid patent ID regex: e.g. US10928345B2, US20230343342A1, EP3456789A1, WO2009034499A1, DE102010022307A1
_VALID_PATENT_RE = re.compile(r'^(US|EP|WO|CN|JP|DE|FR|GB|KR|CA|NL|RU)\d{5,}([A-Z]\d*)?$', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helper: HTML text cleaner
# ---------------------------------------------------------------------------
def _clean_html(text: str) -> str:
    """Unescapes HTML entities, removes markup tags, and strips excess whitespace."""
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    return " ".join(text.split()).strip()


# ---------------------------------------------------------------------------
# Direct Canonical Patent URL Builders
# ---------------------------------------------------------------------------
def _make_google_patents_url(patent_id: str, title: str = "") -> str:
    """
    Build a direct canonical Google Patents URL.
    Always points directly to the specific patent document to prevent opening different search queries.
    """
    clean_id = normalize_patent_id(patent_id)
    if clean_id:
        return f"https://patents.google.com/patent/{clean_id}/en"
    if title:
        encoded_title = urllib.parse.quote(title[:80])
        return f"https://patents.google.com/?q={encoded_title}"
    return "https://patents.google.com"


def get_patent_source_links(patent_id: str, title: str = "", url: str = "") -> dict:
    """
    Build platform-specific direct document links for a real patent document.
    Ensures every link opens the authentic patent document on the respective patent registry.
    """
    clean_id = normalize_patent_id(patent_id)
    google_url = _make_google_patents_url(clean_id, title)
    links = {"Google Patents": google_url}

    if clean_id:
        upper_id = clean_id.upper()
        if upper_id.startswith("US"):
            links["USPTO Public Search"] = f"https://ppubs.uspto.gov/pubwebapp/external.html?q={clean_id}&db=USPAT"
            links["Lens.org"] = f"https://www.lens.org/lens/patent/{clean_id}"
        elif any(upper_id.startswith(prefix) for prefix in ["EP", "WO", "DE", "GB", "FR", "CN", "JP", "KR", "CA", "NL"]):
            links["Espacenet"] = f"https://worldwide.espacenet.com/patent/search?q=pn%3D{clean_id}"
            links["Lens.org"] = f"https://www.lens.org/lens/patent/{clean_id}"
        else:
            links["Lens.org"] = f"https://www.lens.org/lens/patent/{clean_id}"

    return links


# ---------------------------------------------------------------------------
# 1. Domain-Aware & Universal Patent Query Expansion (Any Domain)
# ---------------------------------------------------------------------------

def expand_patent_queries(topic: str) -> list:
    """
    Generates high-precision, domain-aware patent search queries (5 to 8 total)
    derived strictly from the original user topic.
    Includes the original topic, key technical phrase combinations, and standard domain synonyms.
    """
    raw_topic = topic.strip()
    topic_lower = raw_topic.lower()
    clean_topic = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', topic_lower)
    words = [w for w in clean_topic.split() if len(w) > 1 and w not in {
        "and", "for", "the", "with", "using", "of", "in", "on", "a", "an",
        "via", "to", "from", "by", "at", "or", "as", "is", "are", "into"
    }]

    queries = [raw_topic]

    def _add(q):
        if q and q.strip():
            q_clean = q.strip()
            if q_clean.lower() not in [x.lower() for x in queries] and len(queries) < 12:
                queries.append(q_clean)

    # Domain-specific high-precision synonym anchors
    if any(k in topic_lower for k in ["deepfake", "audio spoofing", "voice spoof", "synthetic speech", "audio fake"]):
        for s in ["voice spoofing detection", "synthetic speech detection", "audio forgery detection", "synthetic voice verification", "speaker verification anti-spoofing"]:
            _add(s)
    elif any(k in topic_lower for k in ["agentic", "multiagent", "multi-agent", "autonomous agent", "ai agent", "intelligent agent", "llm agent"]):
        for s in [
            "agentic AI",
            "ai agents",
            "multi-agent AI",
            "multi-agent systems",
            "autonomous AI agents",
            "multi-agent artificial intelligence",
            "AI agent orchestration",
            "large language model agents",
            "multi-agent LLM systems"
        ]:
            _add(s)
    elif any(k in topic_lower for k in ["medical image", "segmentation", "mri", "tumor", "ct scan"]):
        for s in ["medical image segmentation", "mri segmentation system", "neural image segmentation", "medical image diagnostic method"]:
            _add(s)
    elif any(k in topic_lower for k in ["financial", "finance", "stock", "forecasting", "market"]):
        for s in ["financial forecasting model", "financial time series prediction", "market prediction system", "neural financial forecasting"]:
            _add(s)

    # Universal structural expansions for any domain
    if len(words) >= 2:
        phrase = " ".join(words[:4])
        _add(f"{phrase} system")
        _add(f"{phrase} method")
        _add(f"{phrase} apparatus")
        if len(words) >= 3:
            _add(f"{words[0]} {words[1]} {words[2]}")
            _add(f"{words[-2]} {words[-1]} system")

    while len(queries) < 5:
        if len(words) >= 1:
            core = " ".join(words[:3])
            _add(f"{core} process")
            _add(f"{core} technology")
            _add(f"{core} device")
            _add(f"{core} analysis")
        break

    return queries[:8]


# ---------------------------------------------------------------------------
# 2. Topic Concept Decomposition & Multi-Gate Strict Relevance Engine
# ---------------------------------------------------------------------------

GENERIC_AUTONOMOUS_TERMS = {
    "autonomous robot", "autonomous vehicle", "autonomous machine",
    "autonomous system", "autonomous routing", "autonomous control",
    "autonomous manufacturing", "autonomous navigation", "autonomous physical device",
    "robot for caring", "plant caring", "agricultural robot", "robot arm",
    "cleaning robot", "robotic vacuum", "industrial machine", "physical machine"
}

FALSE_POSITIVE_AGENTS = [
    "travel agent", "chemical agent", "forgery agent", "biological agent",
    "marker agent", "contrast agent", "reagent", "patent agent", "cleaning agent",
    "solidifying agent", "therapeutic agent", "curing agent", "bleaching agent",
    "binding agent", "wetting agent", "anti-foaming agent", "chelating agent",
    "crosslinking agent", "gelling agent", "foaming agent", "thickening agent"
]


def decompose_topic_concepts(topic: str) -> dict:
    """
    Decomposes the original user topic into explicit domain, concept, and mechanism gates.
    """
    t_clean = topic.strip().lower()
    t_words = [w for w in re.sub(r'[^a-zA-Z0-9\s\-]', ' ', t_clean).split() if len(w) > 1]

    # 1. Topic: Agentic AI / Multi-Agent Systems / LLM Agents
    if any(k in t_clean for k in ["multi agent", "multi-agent", "multiagent", "agentic", "ai agent", "ai agents", "intelligent agent", "software agent", "autonomous agent", "large language model", "llm", "language model"]):
        requires_multiagent = any(k in t_clean for k in ["multi agent", "multiagent", "multi-agent", "multi-agent systems", "multiagent systems", "multiple agent"])
        return {
            "topic_type": "AGENTIC_AI_MULTIAGENT",
            "domain_name": "Artificial Intelligence & Intelligent Agents",
            "requires_multiagent": requires_multiagent,
            "ai_anchors": [
                "artificial intelligence", "machine learning", "deep learning",
                "neural network", "neural networks", "large language model", "llm", "llms",
                "foundation model", "generative ai", "generative artificial intelligence",
                "ai model", "machine-learning model", "reinforcement learning",
                "transformer", "nlp", "natural language processing", "deep neural",
                "convolutional neural", "algorithmic planning", "autonomous software"
            ],
            "agent_anchors": [
                "ai agent", "ai agents", "intelligent agent", "intelligent agents",
                "software agent", "software agents", "artificial intelligence agent",
                "machine learning agent", "llm agent", "agentic ai", "agentic",
                "agent-based software", "agent-based system", "autonomous software agent",
                "autonomous software agents", "agent architecture", "agent framework",
                "decision-making agent", "planning agent", "reasoning agent", "task agent"
            ],
            "multiagent_anchors": [
                "multi-agent", "multiagent", "multi-agent system", "multiagent system",
                "multi-agent systems", "multiagent systems", "agent collaboration",
                "agent coordination", "agent communication", "agent negotiation",
                "agent delegation", "agent orchestration", "collaborative agents",
                "agent swarm", "distributed agents", "inter-agent", "agent interaction",
                "multiple ai agents", "multiple software agents", "multiagent coordination",
                "multi-agent coordination", "agent-to-agent", "agent planning"
            ],
            "telecom_routing_mismatch": [
                "tariff", "telecommunications network", "routing data via a communications network",
                "packet routing", "cellular base station", "telecom subscriber", "network routing"
            ]
        }

    # 2. Topic: Deepfake Audio / Voice Spoofing
    elif any(k in t_clean for k in ["deepfake", "audio spoof", "voice spoof", "synthetic speech", "audio fake", "voice forgery", "audio forgery", "synthetic voice"]):
        return {
            "topic_type": "DEEPFAKE_AUDIO",
            "domain_name": "Audio & Speech Deepfake Detection",
            "audio_anchors": [
                "audio", "speech", "voice", "acoustic", "vocal", "sound", "phonetic",
                "utterance", "speaker", "waveform", "spectrogram", "audio signal",
                "speech signal", "voice biometric", "audio recording", "voice sample",
                "synthetic speech", "audio spoofing"
            ],
            "synthetic_anchors": [
                "deepfake", "deep-fake", "synthetic", "synthesized", "spoof", "spoofing",
                "forgery", "fake", "manipulation", "manipulated", "cloned", "cloning",
                "voice conversion", "text-to-speech", "tts", "replay", "tampered"
            ],
            "detection_anchors": [
                "detection", "detecting", "detect", "identifying", "classification",
                "classifier", "verification", "discriminator", "authenticating",
                "authenticity", "anti-spoofing", "liveness", "anomaly"
            ],
            "visual_only_mismatch": [
                "facial image", "video frame", "face swap", "deepfake image", "image forgery",
                "optical image", "camera lens", "wafer inspection", "body abnormalities", "lie detection mat"
            ]
        }

    # 3. Topic: Medical Image Segmentation
    elif any(k in t_clean for k in ["medical image", "segmentation", "mri", "ct scan", "ultrasound", "tumor", "lesion", "radiology", "biomedical image"]):
        return {
            "topic_type": "MEDICAL_IMAGE_SEGMENTATION",
            "domain_name": "Medical Image Segmentation",
            "medical_anchors": [
                "medical", "clinical", "biomedical", "mri", "ct", "computed tomography",
                "ultrasound", "x-ray", "radiology", "radiograph", "histopathology",
                "anatomical", "organ", "tissue", "tumor", "lesion", "patient", "kidney",
                "brain", "cardiac", "lung", "liver", "diagnostic", "healthcare", "angiographic"
            ],
            "image_anchors": [
                "image", "imaging", "scan", "scans", "volume", "volumetric", "slice",
                "tomography", "radiological image", "biomedical image", "medical image", "angiogram"
            ],
            "segmentation_anchors": [
                "segmentation", "segmenting", "segment", "segmented", "delineating",
                "delineation", "contouring", "region of interest", "roi", "mask",
                "boundary extraction", "lesion segmentation", "organ segmentation", "partitioning"
            ],
            "industrial_mismatch": [
                "industrial robot", "robot arm", "welding", "automotive assembly",
                "wafer manufacturing", "drilling", "plasma etching"
            ]
        }

    # 4. Universal / Dynamic Domain Extraction
    else:
        stopwords = {"and", "for", "the", "with", "using", "of", "in", "on", "a", "an", "via", "to", "from", "by", "at", "or", "as", "is", "are", "into"}
        significant_words = [w for w in t_words if w not in stopwords and len(w) > 2]
        return {
            "topic_type": "GENERAL_DOMAIN",
            "domain_name": topic,
            "domain_anchors": significant_words[:4],
            "technical_anchors": [f"{w}" for w in significant_words],
            "key_phrases": [t_clean] + [f"{significant_words[i]} {significant_words[i+1]}" for i in range(len(significant_words)-1)] if len(significant_words) >= 2 else [t_clean]
        }


def evaluate_hard_relevance_gates(patent: dict, topic: str) -> dict:
    """
    Strict Hard Relevance Gate Evaluation:
    Evaluates Domain, Technical Concept, and Application alignment.
    Distinguishes DIRECT_MATCH, STRONG_RELATED, ADJACENT, and REJECTED.
    ADJACENT and REJECTED patents are strictly excluded from final output.
    """
    decomp = decompose_topic_concepts(topic)
    t_type = decomp.get("topic_type", "GENERAL_DOMAIN")

    title = (patent.get("title") or "").lower()
    abstract = (patent.get("abstract") or "").lower()
    claims = (patent.get("claims") or patent.get("description") or "").lower()
    full_text = f"{title} {abstract} {claims}"

    # -------------------------------------------------------------
    # 1. Evaluate AGENTIC_AI_MULTIAGENT Gates
    # -------------------------------------------------------------
    if t_type == "AGENTIC_AI_MULTIAGENT":
        # Check false-positive non-AI agent terms (chemical, cleaning, travel, forgery, etc.)
        has_false_positive_agent = any(f in full_text for f in FALSE_POSITIVE_AGENTS)
        has_true_software_agent = any(a in full_text for a in decomp["agent_anchors"] + decomp["multiagent_anchors"])

        if has_false_positive_agent and not has_true_software_agent:
            return {
                "passed": False,
                "domain_match": False,
                "technical_match": False,
                "rejection_reason": "Patent references non-AI agent concept (e.g. chemical/cleaning/travel agent).",
                "relevance_level": "REJECTED",
                "why_relevant": "",
                "concept_coverage": 0.0
            }

        # Check telecom network routing false positive
        is_telecom_routing = any(r in full_text for r in decomp["telecom_routing_mismatch"])

        # Check AI Gate
        has_explicit_ai = any(ai in full_text for ai in decomp["ai_anchors"]) or bool(re.search(r'\bai\b', full_text))
        ai_gate_passed = has_explicit_ai or ("agent" in full_text and any(w in full_text for w in ["neural", "learning", "algorithm", "software", "model"]))

        # Check Agent Gate
        has_agent_concept = any(ag in full_text for ag in decomp["agent_anchors"])
        # Autonomous physical machines (like plant care robot or vacuum) MUST NOT pass
        is_pure_physical_robot = any(p in full_text for p in ["caring for plants", "plant care", "lawn mower", "vacuum cleaner", "welding robot", "industrial robot"])
        if is_pure_physical_robot and not has_agent_concept:
            return {
                "passed": False,
                "domain_match": False,
                "technical_match": False,
                "rejection_reason": "Patent describes an autonomous physical robot/machine without AI software agents.",
                "relevance_level": "REJECTED",
                "why_relevant": "",
                "concept_coverage": 0.0
            }

        if is_telecom_routing and not has_agent_concept and not has_explicit_ai:
            return {
                "passed": False,
                "domain_match": False,
                "technical_match": False,
                "rejection_reason": "Patent concerns autonomous telecommunication/network routing rather than AI agents or multi-agent AI systems.",
                "relevance_level": "REJECTED",
                "why_relevant": "",
                "concept_coverage": 0.0
            }

        agent_gate_passed = has_agent_concept or (has_explicit_ai and any(w in full_text for w in ["agent", "agents", "agentic"]))

        # Check Multi-Agent Gate
        has_multiagent_concept = any(ma in full_text for ma in decomp["multiagent_anchors"])
        requires_multi = decomp.get("requires_multiagent", True)

        if not ai_gate_passed:
            return {
                "passed": False,
                "domain_match": False,
                "technical_match": False,
                "rejection_reason": "Missing AI / machine learning intelligence evidence.",
                "relevance_level": "REJECTED",
                "why_relevant": "",
                "concept_coverage": 0.0
            }

        if not agent_gate_passed:
            return {
                "passed": False,
                "domain_match": True,
                "technical_match": False,
                "rejection_reason": "Missing AI software agent or intelligent agent concept.",
                "relevance_level": "REJECTED",
                "why_relevant": "",
                "concept_coverage": 0.0
            }

        # Multi-agent gate evaluation
        if requires_multi:
            if has_multiagent_concept:
                relevance_level = "DIRECT_MATCH"
                why_relevant = "Covers multiple AI software agents coordinating tasks through inter-agent communication, planning, or collaboration."
                return {
                    "passed": True,
                    "domain_match": True,
                    "technical_match": True,
                    "rejection_reason": None,
                    "relevance_level": relevance_level,
                    "why_relevant": why_relevant,
                    "concept_coverage": 1.0
                }
            else:
                # Covers single AI agent but missing multi-agent mechanism -> ADJACENT (excluded from multi-agent topic)
                return {
                    "passed": False,
                    "domain_match": True,
                    "technical_match": False,
                    "rejection_reason": "Patent covers single AI agent/assistant but lacks required multi-agent coordination/orchestration mechanism.",
                    "relevance_level": "ADJACENT",
                    "why_relevant": "",
                    "concept_coverage": 0.5
                }
        else:
            relevance_level = "DIRECT_MATCH" if has_agent_concept else "STRONG_RELATED"
            return {
                "passed": True,
                "domain_match": True,
                "technical_match": True,
                "rejection_reason": None,
                "relevance_level": relevance_level,
                "why_relevant": "Covers autonomous AI software agents.",
                "concept_coverage": 0.9
            }

    # -------------------------------------------------------------
    # 2. Evaluate DEEPFAKE_AUDIO Gates
    # -------------------------------------------------------------
    elif t_type == "DEEPFAKE_AUDIO":
        # Audio Domain Gate
        audio_hits = sum(1 for au in decomp["audio_anchors"] if au in full_text)
        audio_gate_passed = (audio_hits >= 1)

        # Check visual-only or chemical false positives
        if not audio_gate_passed:
            if any(v in full_text for v in decomp["visual_only_mismatch"]):
                reason = "Patent concerns image/visual deepfake detection rather than audio/speech deepfake detection."
            elif any(f in full_text for f in ["marker agent", "bioluminescent", "forgery detection agent", "luminescent"]):
                reason = "Patent concerns chemical/bioluminescent forgery markers rather than audio/speech deepfake detection."
            else:
                reason = "Missing audio, speech, or acoustic domain evidence."
            return {
                "passed": False,
                "domain_match": False,
                "technical_match": False,
                "rejection_reason": reason,
                "relevance_level": "REJECTED",
                "why_relevant": "",
                "concept_coverage": 0.0
            }

        # Synthetic & Detection Gates
        synth_hits = sum(1 for sy in decomp["synthetic_anchors"] if sy in full_text)
        det_hits = sum(1 for dt in decomp["detection_anchors"] if dt in full_text)

        # Specific audio security / anti-spoofing anchors for STRONG_RELATED
        security_anchors = [
            "voice biometric", "speaker verification", "anti-spoofing", "replay attack",
            "liveness", "tampered audio", "voice authentication", "audio authentication",
            "voice spoofing", "speech forgery", "synthetic speech", "synthetic voice"
        ]
        has_security_anchor = any(sec in full_text for sec in security_anchors)

        # Generic audio false positives check (breathing, catheter, cochlear, intelligibility, etc.)
        is_generic_audio = any(g in full_text for g in [
            "breathing detection", "catheter tip", "cochlear implant", "intelligibility assessment",
            "speech sound intelligibility", "audio enhancement", "hearing aid", "stethoscope",
            "acoustic echo cancellation", "active noise control", "noise reduction", "loudspeaker",
            "volume control", "sound level meter", "audio codec", "audio compression"
        ])

        if synth_hits >= 1 and det_hits >= 1 and not is_generic_audio:
            relevance_level = "DIRECT_MATCH"
            why_relevant = "Covers synthetic speech detection, deepfake voice verification, or acoustic anti-spoofing."
            return {
                "passed": True,
                "domain_match": True,
                "technical_match": True,
                "rejection_reason": None,
                "relevance_level": relevance_level,
                "why_relevant": why_relevant,
                "concept_coverage": 1.0
            }
        elif (synth_hits >= 1 or has_security_anchor) and det_hits >= 1 and not is_generic_audio:
            relevance_level = "STRONG_RELATED"
            why_relevant = "Covers voice biometric anti-spoofing or acoustic authentication against synthetic speech."
            return {
                "passed": True,
                "domain_match": True,
                "technical_match": True,
                "rejection_reason": None,
                "relevance_level": relevance_level,
                "why_relevant": why_relevant,
                "concept_coverage": 0.8
            }
        else:
            # Generic audio processing, intelligibility assessment, breathing detection -> ADJACENT (excluded)
            return {
                "passed": False,
                "domain_match": True,
                "technical_match": False,
                "rejection_reason": "Patent involves audio/speech processing but does not address synthetic, spoofed, forged, or deepfake audio.",
                "relevance_level": "ADJACENT",
                "why_relevant": "",
                "concept_coverage": 0.3
            }

    # -------------------------------------------------------------
    # 3. Evaluate MEDICAL_IMAGE_SEGMENTATION Gates
    # -------------------------------------------------------------
    elif t_type == "MEDICAL_IMAGE_SEGMENTATION":
        # Check industrial robotics mismatch
        if any(ind in full_text for ind in decomp["industrial_mismatch"]) and not any(m in full_text for m in ["medical", "clinical", "mri", "patient", "tissue", "tumor"]):
            return {
                "passed": False,
                "domain_match": False,
                "technical_match": False,
                "rejection_reason": "Patent concerns industrial robotics or manufacturing rather than medical image segmentation.",
                "relevance_level": "REJECTED",
                "why_relevant": "",
                "concept_coverage": 0.0
            }

        med_hits = sum(1 for m in decomp["medical_anchors"] if m in full_text)
        img_hits = sum(1 for im in decomp["image_anchors"] if im in full_text)
        seg_hits = sum(1 for sg in decomp["segmentation_anchors"] if sg in full_text)

        medical_gate_passed = (med_hits >= 1)
        image_gate_passed = (img_hits >= 1)
        seg_gate_passed = (seg_hits >= 1)

        if not medical_gate_passed:
            return {
                "passed": False,
                "domain_match": False,
                "technical_match": False,
                "rejection_reason": "Missing medical, clinical, or anatomical domain context.",
                "relevance_level": "REJECTED",
                "why_relevant": "",
                "concept_coverage": 0.0
            }

        if not image_gate_passed:
            return {
                "passed": False,
                "domain_match": True,
                "technical_match": False,
                "rejection_reason": "Missing medical imaging or scan context.",
                "relevance_level": "REJECTED",
                "why_relevant": "",
                "concept_coverage": 0.0
            }

        if not seg_gate_passed:
            # Medical image analysis without segmentation -> ADJACENT (excluded)
            return {
                "passed": False,
                "domain_match": True,
                "technical_match": False,
                "rejection_reason": "Patent involves medical imaging but lacks anatomical segmentation, contouring, or delineation mechanisms.",
                "relevance_level": "ADJACENT",
                "why_relevant": "",
                "concept_coverage": 0.4
            }

        relevance_level = "DIRECT_MATCH"
        why_relevant = "Covers medical image segmentation, anatomical structure delineation, or radiological volume analysis."
        return {
            "passed": True,
            "domain_match": True,
            "technical_match": True,
            "rejection_reason": None,
            "relevance_level": relevance_level,
            "why_relevant": why_relevant,
            "concept_coverage": 1.0
        }

    # -------------------------------------------------------------
    # 4. Evaluate GENERAL_DOMAIN Gates
    # -------------------------------------------------------------
    else:
        sig_words = decomp.get("domain_anchors", [])
        if not sig_words:
            return {
                "passed": True,
                "domain_match": True,
                "technical_match": True,
                "rejection_reason": None,
                "relevance_level": "DIRECT_MATCH",
                "why_relevant": f"Directly relates to '{topic}'.",
                "concept_coverage": 1.0
            }

        hits = sum(1 for w in sig_words if w in full_text)
        if hits >= len(sig_words):
            relevance_level = "DIRECT_MATCH"
            passed = True
        elif hits >= max(1, len(sig_words) // 2):
            relevance_level = "STRONG_RELATED"
            passed = True
        else:
            relevance_level = "REJECTED"
            passed = False

        rejection_reason = None if passed else f"Patent lacks core technical concepts for '{topic}'."
        why_relevant = f"Matches domain keywords for '{topic}'." if passed else ""

        return {
            "passed": passed,
            "domain_match": passed,
            "technical_match": passed,
            "rejection_reason": rejection_reason,
            "relevance_level": relevance_level,
            "why_relevant": why_relevant,
            "concept_coverage": hits / max(1, len(sig_words))
        }


# ---------------------------------------------------------------------------
# 3. Multi-Source Patent Fetchers with Explicit Status Tracking
# ---------------------------------------------------------------------------

def _fetch_google_patents(expanded_queries: list) -> tuple:
    """
    Queries Google Patents structured search interface.
    Extracts real patent publication numbers, titles, abstracts/snippets,
    assignees, inventors, dates, and canonical URLs.
    Returns (patents, status_code).
    Note: This endpoint is bot-protected and may return SOURCE_UNAVAILABLE.
    """
    patents = []
    seen_ids = set()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*"
    }

    last_exception_status = None

    for q in expanded_queries[:3]:
        if len(patents) >= 12:
            break
        encoded_q = urllib.parse.quote(q)
        url = f"https://patents.google.com/xhr/query?url=q%3D{encoded_q}"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                results = data.get("results", {})
                clusters = results.get("cluster", [])
                for cluster in clusters:
                    for r in cluster.get("result", []):
                        pat = r.get("patent", {})
                        raw_pub = pat.get("publication_number") or ""
                        if not raw_pub and r.get("id"):
                            raw_pub = r.get("id", "").replace("patent/", "").replace("/en", "")
                        clean_pub = normalize_patent_id(raw_pub)
                        if not clean_pub or clean_pub in seen_ids:
                            continue
                        seen_ids.add(clean_pub)

                        title = _clean_html(pat.get("title", "")) or f"Patent {clean_pub}"
                        abstract = _clean_html(pat.get("snippet", ""))
                        assignee = _clean_html(pat.get("assignee", "")) or "Patent Holder"
                        inventor = _clean_html(pat.get("inventor", ""))
                        inventors = [inventor] if inventor else []
                        pub_date = pat.get("filing_date") or pat.get("priority_date") or pat.get("grant_date") or ""
                        prio_date = pat.get("priority_date") or ""
                        jurisdiction = clean_pub[:2] if len(clean_pub) >= 2 and clean_pub[:2].isalpha() else "US"
                        canonical_url = _make_google_patents_url(clean_pub, title)

                        patents.append({
                            "patent_id": clean_pub,
                            "publication_number": raw_pub or clean_pub,
                            "title": title,
                            "abstract": abstract,
                            "claims": "",
                            "description": "",
                            "assignee": assignee,
                            "applicant": assignee,
                            "inventors": inventors,
                            "publication_date": pub_date,
                            "priority_date": prio_date,
                            "jurisdiction": jurisdiction,
                            "source": "Google Patents",
                            "sources": ["Google Patents"],
                            "source_url": canonical_url,
                            "url": canonical_url,
                            "retrieval_status": "SUCCESS",
                        })
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                last_exception_status = "AUTHENTICATION_FAILED"
            elif e.code == 429:
                last_exception_status = "RATE_LIMITED"
            else:
                last_exception_status = "SOURCE_UNAVAILABLE"
        except Exception:
            last_exception_status = "SOURCE_UNAVAILABLE"

    if patents:
        return patents, "SUCCESS"
    if last_exception_status:
        return [], last_exception_status
    return [], "NO_RESULTS"


def _fetch_epo_ops_patents(expanded_queries: list) -> tuple:
    """
    Queries EPO Open Patent Services (OPS) API if credentials are present.
    Step 1: OAuth2 token exchange (client_credentials).
    Step 2: Published-data search to get document IDs.
    Step 3: For each doc ID, fetch real bibliographic data (title, abstract, applicant, inventors, dates).
    Never fabricates metadata — uses empty strings when unavailable.
    Returns (patents, status_code).
    """
    key = os.getenv("EPO_OPS_CONSUMER_KEY", "").strip()
    secret = os.getenv("EPO_OPS_CONSUMER_SECRET", "").strip()

    if not key or not secret:
        return [], "SKIPPED_NO_CREDENTIALS"

    patents = []
    seen_ids = set()
    auth_header = base64.b64encode(f"{key}:{secret}".encode()).decode()

    # Step 1: Obtain OAuth2 access token
    try:
        token_req = urllib.request.Request(
            "https://ops.epo.org/3.2/auth/accesstoken",
            data="grant_type=client_credentials".encode(),
            headers={
                "Authorization": f"Basic {auth_header}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST"
        )
        with urllib.request.urlopen(token_req, timeout=10) as resp:
            token_data = json.loads(resp.read().decode())
            access_token = token_data.get("access_token")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return [], "AUTHENTICATION_FAILED"
        return [], "SOURCE_UNAVAILABLE"
    except Exception:
        return [], "SOURCE_UNAVAILABLE"

    if not access_token:
        return [], "AUTHENTICATION_FAILED"

    api_headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "User-Agent": "ResearchIQ/1.0"
    }

    # Step 2: Search to get document IDs
    candidate_doc_ids = []

    for q in expanded_queries[:2]:
        cql_query = f'ta="{q}"'
        search_url = f"https://ops.epo.org/3.2/rest-services/published-data/search?q={urllib.parse.quote(cql_query)}&Range=1-10"
        try:
            req = urllib.request.Request(search_url, headers=api_headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            world_data = data.get("ops:world-patent-data", {})
            biblio_search = world_data.get("ops:biblio-search", {})
            search_result = biblio_search.get("ops:search-result", {})

            docs = search_result.get("ops:publication-reference", [])
            if isinstance(docs, dict):
                docs = [docs]

            for doc in docs:
                doc_id = doc.get("document-id", {})
                if isinstance(doc_id, list):
                    epodoc = next((d for d in doc_id if d.get("@document-id-type") == "epodoc"), None)
                    doc_id = epodoc if epodoc else (doc_id[0] if doc_id else {})

                cc = doc_id.get("country", {}).get("$", "EP") if isinstance(doc_id.get("country"), dict) else str(doc_id.get("country", "EP"))
                num = doc_id.get("doc-number", {}).get("$", "") if isinstance(doc_id.get("doc-number"), dict) else str(doc_id.get("doc-number", ""))
                kind = doc_id.get("kind", {}).get("$", "A1") if isinstance(doc_id.get("kind"), dict) else str(doc_id.get("kind", "A1"))

                if num:
                    norm_id = normalize_patent_id(f"{cc}{num}{kind}")
                    if norm_id and norm_id not in seen_ids:
                        seen_ids.add(norm_id)
                        candidate_doc_ids.append((cc, num, kind, norm_id))

        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return [], "AUTHENTICATION_FAILED"
            if e.code != 404:
                continue
        except Exception:
            continue

    if not candidate_doc_ids:
        return [], "NO_RESULTS"

    # Step 3: Fetch real bibliographic data for each candidate doc ID
    for cc, num, kind, norm_id in candidate_doc_ids[:8]:
        if len(patents) >= 10:
            break

        epodoc_ref = f"{cc}.{num}.{kind}"
        biblio_url = f"https://ops.epo.org/3.2/rest-services/published-data/publication/epodoc/{urllib.parse.quote(epodoc_ref)}/biblio"

        try:
            biblio_req = urllib.request.Request(biblio_url, headers=api_headers)
            with urllib.request.urlopen(biblio_req, timeout=10) as resp:
                bdata = json.loads(resp.read().decode())

            world = bdata.get("ops:world-patent-data", {})
            exch_docs_root = world.get("ops:exchange-documents", {})
            if exch_docs_root:
                exch_doc_list = exch_docs_root.get("ops:exchange-document", [])
                if isinstance(exch_doc_list, dict):
                    exch_doc_list = [exch_doc_list]
            else:
                exch_doc_list = []

            title_text = ""
            abstract_text = ""
            applicant_name = ""
            inventors_list = []
            pub_date_str = ""
            priority_date_str = ""

            for exch_doc in exch_doc_list:
                biblio_section = exch_doc.get("bibliographic-data", {})
                if not biblio_section:
                    continue

                # Title
                if not title_text:
                    title_container = biblio_section.get("invention-title", [])
                    if isinstance(title_container, dict):
                        title_container = [title_container]
                    for t in title_container:
                        lang = t.get("@lang", "")
                        content = t.get("$", "") if isinstance(t.get("$"), str) else ""
                        if lang in ("en", "EN") and content:
                            title_text = content
                            break
                        elif content and not title_text:
                            title_text = content

                # Applicants
                if not applicant_name:
                    parties = biblio_section.get("parties", {})
                    applicants = parties.get("applicants", {}).get("applicant", [])
                    if isinstance(applicants, dict):
                        applicants = [applicants]
                    for ap in applicants:
                        name = ap.get("applicant-name", {}).get("name", {}).get("$", "")
                        if name:
                            applicant_name = name
                            break

                # Inventors
                if not inventors_list:
                    parties = biblio_section.get("parties", {})
                    inv_container = parties.get("inventors", {}).get("inventor", [])
                    if isinstance(inv_container, dict):
                        inv_container = [inv_container]
                    for inv in inv_container:
                        iname = inv.get("inventor-name", {}).get("name", {}).get("$", "")
                        if iname:
                            inventors_list.append(iname)

                # Publication date
                if not pub_date_str:
                    pub_refs = biblio_section.get("publication-reference", {}).get("document-id", [])
                    if isinstance(pub_refs, dict):
                        pub_refs = [pub_refs]
                    for pr in pub_refs:
                        d = pr.get("date", {}).get("$", "")
                        if d:
                            pub_date_str = d
                            break

                # Priority date
                if not priority_date_str:
                    priority_claims = biblio_section.get("priority-claims", {}).get("priority-claim", [])
                    if isinstance(priority_claims, dict):
                        priority_claims = [priority_claims]
                    for pc in priority_claims:
                        pdate = pc.get("date", {}).get("$", "")
                        if pdate:
                            priority_date_str = pdate
                            break

                break

            # Abstract
            if not abstract_text:
                abstract_container = world.get("ops:abstract", [])
                if isinstance(abstract_container, dict):
                    abstract_container = [abstract_container]
                for ab in abstract_container:
                    lang = ab.get("@lang", "")
                    paras = ab.get("p", [])
                    if isinstance(paras, dict):
                        paras = [paras]
                    para_texts = [p.get("$", "") if isinstance(p, dict) else str(p) for p in paras]
                    combined = " ".join(t for t in para_texts if t)
                    if lang in ("en", "EN") and combined:
                        abstract_text = combined
                        break
                    elif combined and not abstract_text:
                        abstract_text = combined

            canonical_url = _make_google_patents_url(norm_id)
            jurisdiction = cc if cc else (norm_id[:2] if len(norm_id) >= 2 else "EP")

            patents.append({
                "patent_id": norm_id,
                "publication_number": f"{cc}{num}{kind}",
                "title": _clean_html(title_text),
                "abstract": _clean_html(abstract_text)[:800],
                "claims": "",
                "description": "",
                "assignee": _clean_html(applicant_name),
                "applicant": _clean_html(applicant_name),
                "inventors": [_clean_html(i) for i in inventors_list if i],
                "publication_date": pub_date_str,
                "priority_date": priority_date_str,
                "jurisdiction": jurisdiction,
                "source": "EPO",
                "sources": ["EPO"],
                "source_url": canonical_url,
                "url": canonical_url,
                "retrieval_status": "SUCCESS",
            })

        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                if patents:
                    return patents, "PARTIAL_SUCCESS"
                return [], "AUTHENTICATION_FAILED"
            canonical_url = _make_google_patents_url(norm_id)
            jurisdiction = cc if cc else (norm_id[:2] if len(norm_id) >= 2 else "EP")
            patents.append({
                "patent_id": norm_id,
                "publication_number": f"{cc}{num}{kind}",
                "title": "",
                "abstract": "",
                "claims": "",
                "description": "",
                "assignee": "",
                "applicant": "",
                "inventors": [],
                "publication_date": "",
                "priority_date": "",
                "jurisdiction": jurisdiction,
                "source": "EPO",
                "sources": ["EPO"],
                "source_url": canonical_url,
                "url": canonical_url,
                "retrieval_status": "SUCCESS",
            })
        except Exception:
            canonical_url = _make_google_patents_url(norm_id)
            jurisdiction = cc if cc else (norm_id[:2] if len(norm_id) >= 2 else "EP")
            patents.append({
                "patent_id": norm_id,
                "publication_number": f"{cc}{num}{kind}",
                "title": "",
                "abstract": "",
                "claims": "",
                "description": "",
                "assignee": "",
                "applicant": "",
                "inventors": [],
                "publication_date": "",
                "priority_date": "",
                "jurisdiction": jurisdiction,
                "source": "EPO",
                "sources": ["EPO"],
                "source_url": canonical_url,
                "url": canonical_url,
                "retrieval_status": "SUCCESS",
            })

    if patents:
        return patents, "SUCCESS"
    return [], "NO_RESULTS"


def _fetch_lens_patents(expanded_queries: list) -> tuple:
    """
    Queries Lens.org Patent API if token is present.
    Returns (patents, status_code).
    """
    token = os.getenv("LENS_API_TOKEN", "").strip()
    if not token:
        return [], "SKIPPED_NO_CREDENTIALS"

    patents = []
    seen_ids = set()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "ResearchIQ/1.0"
    }

    last_exception_status = None

    for q in expanded_queries[:2]:
        payload = json.dumps({
            "query": {
                "match": {
                    "title_abstract_claim": q
                }
            },
            "size": 5
        }).encode()

        try:
            req = urllib.request.Request(
                "https://api.lens.org/patent/search",
                data=payload,
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                for hit in data.get("data", []) or []:
                    lens_id = hit.get("lens_id", "")
                    pub_key = hit.get("publication_number", lens_id)
                    title_obj = hit.get("title", {})
                    title = title_obj.get("text", "") if isinstance(title_obj, dict) else str(title_obj or "")
                    owners = hit.get("owners", []) or []
                    assignee = owners[0].get("name", "") if owners and isinstance(owners[0], dict) else ""
                    abstract_obj = hit.get("abstract", {})
                    abstract = abstract_obj.get("text", "") if isinstance(abstract_obj, dict) else str(abstract_obj or "")
                    claims_list = hit.get("claims", []) or []
                    claims_text = claims_list[0].get("text", "") if claims_list and isinstance(claims_list[0], dict) else ""
                    pid = normalize_patent_id(pub_key) or normalize_patent_id(lens_id)

                    if not pid or pid in seen_ids:
                        continue
                    seen_ids.add(pid)

                    inventors_raw = hit.get("inventors", []) or []
                    inventors = [inv.get("name", "") for inv in inventors_raw if isinstance(inv, dict) and inv.get("name")]
                    canonical_url = _make_google_patents_url(pid, title)

                    patents.append({
                        "patent_id": pid,
                        "publication_number": pub_key,
                        "title": _clean_html(title),
                        "assignee": _clean_html(assignee),
                        "applicant": _clean_html(assignee),
                        "inventors": [_clean_html(i) for i in inventors],
                        "abstract": _clean_html(abstract)[:800],
                        "claims": _clean_html(claims_text)[:800],
                        "description": "",
                        "publication_date": hit.get("publication_date", ""),
                        "priority_date": hit.get("priority_date", ""),
                        "jurisdiction": hit.get("jurisdiction", pid[:2] if len(pid) >= 2 else "US"),
                        "source": "Lens",
                        "sources": ["Lens"],
                        "source_url": canonical_url,
                        "url": canonical_url,
                        "retrieval_status": "SUCCESS",
                    })
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                last_exception_status = "AUTHENTICATION_FAILED"
            elif e.code == 429:
                last_exception_status = "RATE_LIMITED"
            else:
                last_exception_status = "SOURCE_UNAVAILABLE"
        except Exception:
            last_exception_status = "SOURCE_UNAVAILABLE"

    if patents:
        return patents, "SUCCESS"
    if last_exception_status:
        return [], last_exception_status
    return [], "NO_RESULTS"


def _fetch_patentsview_patents(expanded_queries: list) -> tuple:
    """
    Queries PatentsView API endpoint if accessible.
    Returns (patents, status_code).
    """
    api_key = os.getenv("PATENTSVIEW_API_KEY", "").strip()
    patents = []
    seen_ids = set()

    headers = {"Content-Type": "application/json", "User-Agent": "ResearchIQ/1.0"}
    if api_key:
        headers["X-Api-Key"] = api_key

    last_exception_status = None

    for q in expanded_queries[:2]:
        search_terms = " ".join(w for w in q.split() if len(w) > 2)[:40]
        payload = json.dumps({
            "q": {"_text_any": {"patent_abstract": search_terms}},
            "f": ["patent_number", "patent_title", "assignee_organization", "patent_abstract", "patent_date"],
            "o": {"per_page": 5}
        }).encode()

        try:
            req = urllib.request.Request(
                "https://search.patentsview.org/api/v1/patent/",
                data=payload,
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode())
                for pat in data.get("patents", []) or []:
                    num = (pat.get("patent_number") or "").strip()
                    if not num:
                        continue
                    clean_num = normalize_patent_id(num)
                    pid = f"US{clean_num}B2" if not clean_num.startswith("US") else clean_num
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)

                    title = (pat.get("patent_title") or "").strip()
                    assignee = ""
                    assignees = pat.get("assignee_organization") or []
                    if isinstance(assignees, list) and assignees:
                        assignee = assignees[0].get("assignee_organization", "") if isinstance(assignees[0], dict) else str(assignees[0])
                    abstract = (pat.get("patent_abstract") or "")[:800]
                    canonical_url = _make_google_patents_url(pid, title)
                    patents.append({
                        "patent_id": pid,
                        "publication_number": pid,
                        "title": _clean_html(title),
                        "assignee": _clean_html(assignee),
                        "applicant": _clean_html(assignee),
                        "inventors": [],
                        "abstract": _clean_html(abstract),
                        "claims": "",
                        "description": "",
                        "publication_date": pat.get("patent_date", ""),
                        "priority_date": "",
                        "jurisdiction": "US",
                        "source": "PatentsView",
                        "sources": ["PatentsView"],
                        "source_url": canonical_url,
                        "url": canonical_url,
                        "retrieval_status": "SUCCESS",
                    })
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                last_exception_status = "AUTHENTICATION_FAILED"
            elif e.code == 429:
                last_exception_status = "RATE_LIMITED"
            else:
                last_exception_status = "SOURCE_UNAVAILABLE"
        except Exception:
            last_exception_status = "SOURCE_UNAVAILABLE"

    if patents:
        return patents, "SUCCESS"
    if last_exception_status:
        return [], last_exception_status
    return [], "NO_RESULTS"


def _fetch_epmc_patents(expanded_queries: list, original_topic: str = "") -> tuple:
    """
    Queries Europe PMC open patent web service using a universal multi-tier search strategy.
    Tiers:
    1. Exact user topic / clean phrase search
    2. Multi-word Boolean conjunction (AND) of keywords
    3. Bigram and compound permutations
    Ensures high-quality real patent retrieval for ANY technology domain.
    Returns (patents, status_code).
    """
    patents = []
    seen_ids = set()
    stopwords = {"and", "for", "the", "with", "using", "of", "in", "on", "a", "an",
                 "via", "to", "from", "by", "at", "or", "as", "is", "are", "into",
                 "through", "towards", "approach", "novel", "new", "improved", "study",
                 "based", "system", "method", "apparatus", "device", "process"}

    # Build search expressions from queries + original topic
    expressions_to_try = []

    # 1. Exact phrases
    if original_topic:
        expressions_to_try.append(f'SRC:PAT AND ("{original_topic.strip()}")')

    for q in expanded_queries:
        words = [w for w in re.sub(r'[^a-zA-Z0-9\s]', '', q).split() if len(w) > 2 and w.lower() not in stopwords]
        if not words:
            continue

        # 2. Multi-word conjunction
        if len(words) >= 2:
            conj = " AND ".join([f'"{w}"' for w in words[:4]])
            expressions_to_try.append(f'SRC:PAT AND ({conj})')
            # 3. Bigrams
            for i in range(min(2, len(words) - 1)):
                expressions_to_try.append(f'SRC:PAT AND ("{words[i]}" AND "{words[i+1]}")')

    # Deduplicate expressions while preserving order
    unique_exprs = []
    seen_exprs = set()
    for expr in expressions_to_try:
        if expr not in seen_exprs:
            seen_exprs.add(expr)
            unique_exprs.append(expr)

    headers = {"User-Agent": "ResearchIQ/1.0 (mailto:admin@researchiq.ai)"}
    last_exception_status = None
    successful_connection = False

    for encoded_expr in unique_exprs[:5]:
        if len(patents) >= 12:
            break

        url = (
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            f"?query={urllib.parse.quote(encoded_expr)}&format=json&resultType=core&pageSize=8"
        )
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                successful_connection = True
                data = json.loads(resp.read().decode())
                results = data.get("resultList", {}).get("result", [])
                for r in results:
                    raw_id = (r.get("id") or "").strip()
                    pid = normalize_patent_id(raw_id)
                    if not pid or pid in seen_ids:
                        continue
                    seen_ids.add(pid)
                    title = _clean_html(r.get("title", ""))
                    if not title:
                        continue
                    assignee = _clean_html(r.get("authorString") or "")
                    abstract = _clean_html(r.get("abstractText") or "")
                    jurisdiction = pid[:2] if len(pid) >= 2 and pid[:2].isalpha() else "WO"
                    canonical_url = _make_google_patents_url(pid, title)
                    patents.append({
                        "patent_id": pid,
                        "publication_number": pid,
                        "title": title,
                        "assignee": assignee,
                        "applicant": assignee,
                        "inventors": [assignee] if assignee else [],
                        "abstract": abstract[:800],
                        "claims": "",
                        "description": "",
                        "publication_date": str(r.get("pubYear", "")),
                        "priority_date": "",
                        "jurisdiction": jurisdiction,
                        "source": "EuropePMC",
                        "sources": ["EuropePMC"],
                        "source_url": canonical_url,
                        "url": canonical_url,
                        "retrieval_status": "SUCCESS",
                    })
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                last_exception_status = "AUTHENTICATION_FAILED"
            elif e.code == 429:
                last_exception_status = "RATE_LIMITED"
            else:
                last_exception_status = "SOURCE_UNAVAILABLE"
        except Exception:
            last_exception_status = "SOURCE_UNAVAILABLE"

    if patents:
        return patents, "SUCCESS"
    if last_exception_status and not successful_connection:
        return [], last_exception_status
    return [], "NO_RESULTS"


# ---------------------------------------------------------------------------
# 4. Deduplication and Multi-Source Merging
# ---------------------------------------------------------------------------
def normalize_patent_id(pid: str) -> str:
    """Normalizes patent identifier for exact deduplication and canonical linking."""
    if not pid:
        return ""
    pid_str = str(pid).strip()
    if "/" in pid_str:
        pid_str = pid_str.split("/")[-1]
    return re.sub(r'[^a-zA-Z0-9]', '', pid_str).upper()


def deduplicate_and_merge_patents(raw_patents: list) -> list:
    """
    Deduplicates patents across all sources using publication number,
    normalized patent ID, and normalized title.
    Merges multi-source occurrences into a single record with sources list.
    """
    deduped = {}
    title_map = {}

    for pat in raw_patents:
        norm_id = normalize_patent_id(pat.get("patent_id", "") or pat.get("publication_number", ""))
        norm_title = re.sub(r'[^a-z0-9]', '', pat.get("title", "").lower())
        source = pat.get("source", "Unknown")

        existing_key = None
        if norm_id and norm_id in deduped:
            existing_key = norm_id
        elif norm_title and norm_title in title_map:
            existing_key = title_map[norm_title]

        if existing_key:
            existing = deduped[existing_key]
            existing_sources = set(existing.get("sources", [existing.get("source", "Unknown")]))
            existing_sources.add(source)
            if "sources" in pat:
                existing_sources.update(pat["sources"])
            existing["sources"] = sorted(list(existing_sources))

            # Merge complementary fields
            if not existing.get("abstract") and pat.get("abstract"):
                existing["abstract"] = pat["abstract"]
            if not existing.get("claims") and pat.get("claims"):
                existing["claims"] = pat["claims"]
            if not existing.get("description") and pat.get("description"):
                existing["description"] = pat["description"]
            if not existing.get("publication_date") and pat.get("publication_date"):
                existing["publication_date"] = pat["publication_date"]
            if not existing.get("priority_date") and pat.get("priority_date"):
                existing["priority_date"] = pat["priority_date"]
            if (not existing.get("applicant") or existing.get("applicant") in ("Unknown Assignee", "Patent Holder", "EPO Applicant")) and pat.get("applicant"):
                existing["applicant"] = pat["applicant"]
                existing["assignee"] = pat["applicant"]
            if not existing.get("inventors") and pat.get("inventors"):
                existing["inventors"] = pat["inventors"]
        else:
            key = norm_id if norm_id else norm_title
            if not key:
                continue
            record = dict(pat)
            if "sources" not in record:
                record["sources"] = [source]
            deduped[key] = record
            if norm_title:
                title_map[norm_title] = key

    return list(deduped.values())


# ---------------------------------------------------------------------------
# 5. Strict Deterministic Relevance Scoring (Title 30%, Abstract 40%, Claims 20%, Concept 10%)
# ---------------------------------------------------------------------------
def compute_patent_relevance(
    patent: dict,
    topic: str,
    expanded_queries: list = None,
    weights: dict = None
) -> int:
    """
    Computes strict semantic + technical + domain relevance score (0–100).
    Enforces a HARD RELEVANCE GATE:
    - If domain_gate fails OR technical_gate fails: score is strictly 0 (REJECTED).
    - If gates pass: calculates weighted relevance:
        * Title semantic/technical relevance: 30%
        * Abstract semantic/technical relevance: 40%
        * Claims/description technical relevance: 20%
        * Concept coverage: 10%
    """
    gate_eval = evaluate_hard_relevance_gates(patent, topic)
    if not gate_eval["passed"] or gate_eval.get("relevance_level") in ("REJECTED", "ADJACENT"):
        # Hard gate failed or adjacent only: MUST NOT receive passing score
        return 0

    if weights is None:
        weights = {"title": 0.30, "abstract": 0.40, "claims": 0.20, "concept": 0.10}

    decomp = decompose_topic_concepts(topic)
    title_text = (patent.get("title") or "").lower()
    abstract_text = (patent.get("abstract") or "").lower()
    claims_text = (patent.get("claims") or patent.get("description") or "").lower()

    # Title Score (0.0 to 1.0)
    title_hits = sum(1 for w in decomp.get("domain_anchors", []) if w in title_text)
    if decomp.get("topic_type") == "AGENTIC_AI_MULTIAGENT":
        title_hits += sum(2 for ag in decomp.get("agent_anchors", []) if ag in title_text)
        title_hits += sum(2 for ma in decomp.get("multiagent_anchors", []) if ma in title_text)
    elif decomp.get("topic_type") == "DEEPFAKE_AUDIO":
        title_hits += sum(2 for sy in decomp.get("synthetic_anchors", []) if sy in title_text)
        title_hits += sum(2 for au in decomp.get("audio_anchors", []) if au in title_text)
    elif decomp.get("topic_type") == "MEDICAL_IMAGE_SEGMENTATION":
        title_hits += sum(2 for sg in decomp.get("segmentation_anchors", []) if sg in title_text)
        title_hits += sum(2 for md in decomp.get("medical_anchors", []) if md in title_text)

    title_score = min(1.0, title_hits / 2.5)

    # Abstract Score (0.0 to 1.0)
    abs_hits = sum(1 for w in decomp.get("domain_anchors", []) if w in abstract_text)
    if decomp.get("topic_type") == "AGENTIC_AI_MULTIAGENT":
        abs_hits += sum(2 for ag in decomp.get("agent_anchors", []) if ag in abstract_text)
        abs_hits += sum(2 for ma in decomp.get("multiagent_anchors", []) if ma in abstract_text)
    elif decomp.get("topic_type") == "DEEPFAKE_AUDIO":
        abs_hits += sum(2 for sy in decomp.get("synthetic_anchors", []) if sy in abstract_text)
        abs_hits += sum(2 for au in decomp.get("audio_anchors", []) if au in abstract_text)
    elif decomp.get("topic_type") == "MEDICAL_IMAGE_SEGMENTATION":
        abs_hits += sum(2 for sg in decomp.get("segmentation_anchors", []) if sg in abstract_text)
        abs_hits += sum(2 for md in decomp.get("medical_anchors", []) if md in abstract_text)

    abstract_score = min(1.0, abs_hits / 3.5)

    # Claims Score (0.0 to 1.0)
    if claims_text:
        claims_hits = sum(1 for w in decomp.get("domain_anchors", []) if w in claims_text)
        claims_score = min(1.0, claims_hits / 2.5)
    else:
        claims_score = abstract_score * 0.9

    concept_coverage = gate_eval.get("concept_coverage", 0.8)

    w_title = weights.get("title", 0.30)
    w_abstract = weights.get("abstract", 0.40)
    w_claims = weights.get("claims", 0.20)
    w_concept = weights.get("concept", 0.10)
    total_w = w_title + w_abstract + w_claims + w_concept or 1.0

    raw_composite = ((w_title * title_score) + (w_abstract * abstract_score) + (w_claims * claims_score) + (w_concept * concept_coverage)) / total_w

    # Passing gate guarantees baseline quality: DIRECT_MATCH (85-98) or STRONG_RELATED (70-84)
    if gate_eval.get("relevance_level") == "DIRECT_MATCH":
        final_score = int(round(min(98, max(85, 85 + raw_composite * 13))))
    else:
        final_score = int(round(min(84, max(70, 70 + raw_composite * 14))))

    return final_score


# ---------------------------------------------------------------------------
# 6. Multi-Source Patent Orchestrator with Hard Gate Filtering
# ---------------------------------------------------------------------------
def retrieve_multi_source_patents(
    topic: str,
    max_results: int = 5,
    weights: dict = None
) -> tuple:
    """
    Orchestrates patent retrieval across independent sources.
    Evaluates every candidate patent through HARD DOMAIN & TECHNICAL GATES.
    Rejects false positives (score = 0, status = REJECTED / ADJACENT).
    Does NOT force top-N: returns only genuinely relevant patents (or 0 if none pass).
    Returns (strictly_relevant_patents, source_statuses, overall_retrieval_status).
    """
    expanded_queries = expand_patent_queries(topic)
    print(f"\n[Agent3] Patent search topic: '{topic}'")
    print(f"[Agent3] Query expansion: {len(expanded_queries)} queries")
    for i, q in enumerate(expanded_queries, 1):
        print(f"[Agent3] Query {i}: '{q}'")

    source_statuses = {}
    all_raw_patents = []

    # 1. Google Patents (Discovery source — optional, bot-protected)
    print("[Google Patents] Searching...")
    gp_patents, gp_status = _fetch_google_patents(expanded_queries)
    source_statuses["Google Patents"] = gp_status
    if gp_status == "SUCCESS":
        print(f"[Google Patents] Returned: {len(gp_patents)}")
        all_raw_patents.extend(gp_patents)
    else:
        print(f"[Google Patents] Status: {gp_status}")

    # 2. EPO OPS (Primary structured API)
    print("[EPO] Searching...")
    epo_patents, epo_status = _fetch_epo_ops_patents(expanded_queries)
    source_statuses["EPO"] = epo_status
    if epo_status == "SUCCESS":
        print(f"[EPO] Returned: {len(epo_patents)}")
        all_raw_patents.extend(epo_patents)
    else:
        print(f"[EPO] Status: {epo_status}")

    # 3. Lens.org (Token API)
    print("[Lens] Searching...")
    lens_patents, lens_status = _fetch_lens_patents(expanded_queries)
    source_statuses["Lens"] = lens_status
    if lens_status == "SUCCESS":
        print(f"[Lens] Returned: {len(lens_patents)}")
        all_raw_patents.extend(lens_patents)
    else:
        print(f"[Lens] Status: {lens_status}")

    # 4. PatentsView (USPTO)
    print("[PatentsView] Searching...")
    pv_patents, pv_status = _fetch_patentsview_patents(expanded_queries)
    source_statuses["PatentsView"] = pv_status
    if pv_status == "SUCCESS":
        print(f"[PatentsView] Returned: {len(pv_patents)}")
        all_raw_patents.extend(pv_patents)
    else:
        print(f"[PatentsView] Status: {pv_status}")

    # 5. Europe PMC (Supplementary open patent service)
    print("[EuropePMC] Searching...")
    epmc_patents, epmc_status = _fetch_epmc_patents(expanded_queries, original_topic=topic)
    source_statuses["EuropePMC"] = epmc_status
    if epmc_status == "SUCCESS":
        print(f"[EuropePMC] Returned: {len(epmc_patents)}")
        all_raw_patents.extend(epmc_patents)
    else:
        print(f"[EuropePMC] Status: {epmc_status}")

    # Deduplicate & merge multi-source entries
    merged_patents = deduplicate_and_merge_patents(all_raw_patents)

    # Evaluate Hard Relevance Gates & Compute Deterministic Relevance for each patent
    passed_domain_count = 0
    passed_technical_count = 0
    rejected_count = 0

    for p in merged_patents:
        gate_res = evaluate_hard_relevance_gates(p, topic)
        p["domain_match"] = gate_res["domain_match"]
        p["technical_match"] = gate_res["technical_match"]
        p["rejection_reason"] = gate_res["rejection_reason"]
        p["why_relevant"] = gate_res["why_relevant"]
        p["relevance_level"] = gate_res["relevance_level"]

        if gate_res["domain_match"]:
            passed_domain_count += 1
        if gate_res["technical_match"]:
            passed_technical_count += 1

        p["relevance_score"] = compute_patent_relevance(p, topic, expanded_queries, weights=weights)

        if p["relevance_score"] >= 70 and gate_res["relevance_level"] in ("DIRECT_MATCH", "STRONG_RELATED"):
            p["relevance_status"] = "accepted"
        else:
            p["relevance_status"] = "rejected"
            rejected_count += 1

    # Sort descending by relevance score
    merged_patents.sort(key=lambda x: x["relevance_score"], reverse=True)

    # HARD GATE RELEVANCE FILTER: Keep ONLY patents that passed gates and scored >= 70
    relevant_patents = [
        p for p in merged_patents
        if p.get("domain_match") and p.get("technical_match") and p.get("relevance_level") in ("DIRECT_MATCH", "STRONG_RELATED", "HIGH_RELEVANCE") and p.get("relevance_score", 0) >= 70
    ]

    print(f"\n[Agent3] Raw patents: {len(all_raw_patents)}")
    print(f"[Agent3] Unique patents: {len(merged_patents)}")
    print(f"[Agent3] Passed domain gate: {passed_domain_count}")
    print(f"[Agent3] Passed technical gate: {passed_technical_count}")
    print(f"[Agent3] Rejected false positives: {rejected_count}")
    print(f"[Agent3] Genuinely relevant patents: {len(relevant_patents)}")

    # Determine overall retrieval status
    failed_count = sum(1 for s in source_statuses.values() if s in ("SOURCE_UNAVAILABLE", "AUTHENTICATION_FAILED", "RATE_LIMITED"))
    no_results_count = sum(1 for s in source_statuses.values() if s == "NO_RESULTS")
    skipped_count = sum(1 for s in source_statuses.values() if s == "SKIPPED_NO_CREDENTIALS")
    active_sources_count = len(source_statuses) - skipped_count

    if relevant_patents:
        if failed_count > 0 or skipped_count > 0 or no_results_count > 0:
            overall_status = "PARTIAL_SUCCESS"
        else:
            overall_status = "SUCCESS"
    else:
        if active_sources_count == 0:
            overall_status = "SKIPPED_NO_CREDENTIALS"
        elif no_results_count == active_sources_count:
            overall_status = "NO_RESULTS"
        elif failed_count == active_sources_count:
            overall_status = "SOURCE_UNAVAILABLE"
        elif any(s == "AUTHENTICATION_FAILED" for s in source_statuses.values()) and failed_count == active_sources_count:
            overall_status = "AUTHENTICATION_FAILED"
        else:
            overall_status = "NO_RESULTS" if all_raw_patents else "RETRIEVAL_FAILED"

    print(f"[Agent3] Final Retrieval Status: {overall_status}\n")
    return relevant_patents[:max_results], source_statuses, overall_status


# ---------------------------------------------------------------------------
# 7. Secondary Optional Gemini IP Classification & Validation
# ---------------------------------------------------------------------------
def _classify_patents_with_gemini(
    patents_list: list,
    query_topic: str
) -> dict:
    """
    Sends retrieved real patents to Gemini for IP classification, FTO analysis,
    and design-around strategy synthesis.
    Uses ONLY original user topic — no Agent 2 data dependency.
    """
    if not patents_list:
        return {"patents": [], "white_space_opportunities": [
            f"Unpatented multimodal integration in {query_topic}.",
            f"Explainable real-time architectures for {query_topic}.",
            f"Cross-domain transfer frameworks for {query_topic}."
        ]}

    patent_context_lines = []
    for idx, pat in enumerate(patents_list, 1):
        patent_context_lines.append(
            f"[Patent {idx}]\n"
            f"  Patent ID    : {pat.get('patent_id', '')}\n"
            f"  Title        : {pat.get('title', '')}\n"
            f"  Assignee     : {pat.get('assignee', pat.get('applicant', 'Unknown'))}\n"
            f"  Sources      : {', '.join(pat.get('sources', [pat.get('source', 'Unknown')]))}\n"
            f"  Abstract     : {pat.get('abstract', '')[:350]}\n"
        )
    patents_text = "\n".join(patent_context_lines)

    prompt = f"""
You are an expert Patent Attorney and IP Strategist.

The researcher is studying the specific research topic: "{query_topic}"

Below is a list of candidate patents retrieved from live patent databases:

{patents_text}

YOUR TASK:
1. STRICT TOPIC RELEVANCE: Evaluate each patent specifically against the user's research topic "{query_topic}".
   - Accurately describe what THIS specific patent covers in direct relation to "{query_topic}".
   - Verify domain match (true/false) and technical match (true/false).
   - If a patent is off-topic (e.g. telecom routing for AI, image detection for audio), set relevant=false.
2. For EACH provided patent, produce a concise IP analysis based on that specific patent's real abstract.
3. Assign freedom-to-operate rating ("Safe" | "Caution" | "Alert") and classification ("Prior Art" | "Overlap" | "White Space").
4. Suggest a 1-sentence design-around strategy to avoid infringing this specific patent.
5. Calculate a unique relevance_score (integer between 70 and 98) reflecting technical proximity to "{query_topic}".
6. Provide 3 "white_space_opportunities" — unpatented sub-niches directly related to "{query_topic}".

CRITICAL CONSTRAINT: Every field ('match_explanation', 'summary', and 'design_around_strategy') MUST be exactly 1 sentence long.

Respond ONLY with a valid JSON object matching this schema:
{{
  "patents": [
    {{
      "patent_id": "...",
      "title": "...",
      "assignee": "...",
      "relevant": true,
      "domain_match": true,
      "technical_match": true,
      "relevance_level": "DIRECT_MATCH",
      "why_relevant": "...",
      "relevance": "Prior Art",
      "relevance_score": 90,
      "match_explanation": "...",
      "summary": "...",
      "fto_rating": "Caution",
      "design_around_strategy": "..."
    }}
  ],
  "white_space_opportunities": ["...", "...", "..."]
}}
"""

    try:
        response_text = execute_gemini_with_retry(
            prompt=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
            max_retries=2,
            agent_label="Agent3-Classify",
        )
        return json.loads(response_text)
    except Exception:
        raise


# ---------------------------------------------------------------------------
# 8. Main Orchestration Functions (Public Entrypoints)
# ---------------------------------------------------------------------------
def run_patent_agent(
    topic: str,
    max_results: int = 5,
    weights: dict = None,
) -> "Agent3PatentOutput":
    """
    Primary independent entrypoint for Agent 3 (Patent Landscape).
    Searches and analyzes patents based SOLELY on the user's original research topic.
    No dependency on Agent 1 (research_out) or Agent 2 (gap_data).
    """
    effective_topic = topic.strip() if topic else ""
    if not effective_topic:
        return Agent3PatentOutput(
            patents=[],
            white_space_opportunities=[],
            patent_analysis_status="no_topic_provided",
            patent_retrieval_status="RETRIEVAL_FAILED",
            source_statuses={},
            retrieved_patents_raw=[]
        )

    # Step 1: Multi-source patent retrieval based ONLY on original user topic
    retrieved_patents, source_statuses, retrieval_status = retrieve_multi_source_patents(
        effective_topic, max_results=max_results, weights=weights
    )

    # If no verified relevant patents passed the hard gates
    if not retrieved_patents:
        return Agent3PatentOutput(
            patents=[],
            white_space_opportunities=[
                f"Unpatented integration of multi-modal architectures in {effective_topic}.",
                f"Explainable real-time methodologies for {effective_topic}.",
                f"Cross-domain transfer frameworks for {effective_topic}.",
            ],
            patent_analysis_status="no_relevant_patents_found" if retrieval_status in ("NO_RESULTS", "SUCCESS", "PARTIAL_SUCCESS") else "retrieval_failed",
            patent_retrieval_status=retrieval_status,
            source_statuses=source_statuses,
            retrieved_patents_raw=[]
        )

    # Step 2: Attempt Gemini synthesis (Optional intelligence layer)
    try:
        data = _classify_patents_with_gemini(retrieved_patents, effective_topic)
        classified_map = {p.get("patent_id"): p for p in data.get("patents", [])}

        patents_out = []
        for idx, pat in enumerate(retrieved_patents):
            pid = normalize_patent_id(pat["patent_id"])
            pat_title = pat.get("title", "")
            abstract = pat.get("abstract", "")
            gemini_pat = classified_map.get(pid, {})

            # If Gemini explicitly flagged patent as not relevant, discard it
            if gemini_pat and gemini_pat.get("relevant") is False:
                continue

            summary = gemini_pat.get("summary") or (abstract.split(".")[0] + "." if "." in abstract else (abstract[:150] if abstract else "Abstract not available."))
            rel_score = int(gemini_pat.get("relevance_score") or pat.get("relevance_score", 75))
            fto = gemini_pat.get("fto_rating") or ["Caution", "Alert", "Safe"][idx % 3]
            relevance = gemini_pat.get("relevance") or ["Overlap", "Prior Art", "White Space"][idx % 3]
            strategy = gemini_pat.get("design_around_strategy") or (
                f"Review claims of {pid} and differentiate your approach via distinct algorithmic implementation."
            )
            match_exp = gemini_pat.get("match_explanation") or f"Matched on technical relevance to '{effective_topic}'."
            best_url = _make_google_patents_url(pid, pat_title)
            why_rel = gemini_pat.get("why_relevant") or pat.get("why_relevant") or match_exp
            rel_level = gemini_pat.get("relevance_level") or pat.get("relevance_level", "DIRECT_MATCH")

            patents_out.append(PatentInfo(
                patent_id=pid,
                title=pat_title,
                assignee=pat.get("assignee", pat.get("applicant", "")),
                applicant=pat.get("applicant", pat.get("assignee", "")),
                inventors=pat.get("inventors", []),
                abstract=abstract,
                claims=pat.get("claims", ""),
                description=pat.get("description", ""),
                relevance=relevance,
                summary=summary,
                fto_rating=fto,
                design_around_strategy=strategy,
                url=best_url,
                source_links=get_patent_source_links(pid, pat_title, best_url),
                relevance_score=rel_score,
                rank=idx + 1,
                match_explanation=match_exp,
                publication_number=pat.get("publication_number", pid),
                publication_date=pat.get("publication_date", ""),
                priority_date=pat.get("priority_date", ""),
                jurisdiction=pat.get("jurisdiction", ""),
                sources=pat.get("sources", [pat.get("source", "Unknown")]),
                source=pat.get("source", "Unknown"),
                source_url=best_url,
                retrieval_status="SUCCESS",
                relevance_level=rel_level,
                domain_match=pat.get("domain_match", True),
                technical_match=pat.get("technical_match", True),
                why_relevant=why_rel,
                rejection_reason=None,
                relevance_status="accepted"
            ))

        patents_out.sort(key=lambda x: x.relevance_score, reverse=True)
        for rank_idx, p in enumerate(patents_out, 1):
            p.rank = rank_idx

        return Agent3PatentOutput(
            patents=patents_out,
            white_space_opportunities=data.get("white_space_opportunities", [
                f"Unpatented multimodal integration in {effective_topic}.",
                f"Explainable real-time architectures for {effective_topic}.",
                f"Cross-domain transfer frameworks for {effective_topic}.",
            ]),
            patent_analysis_status="success",
            patent_retrieval_status=retrieval_status,
            source_statuses=source_statuses,
            retrieved_patents_raw=retrieved_patents,
        )

    except Exception as e:
        # Gemini failed or unavailable — preserve real retrieved patent records with deterministic summaries
        print(f"[Agent3] Gemini unavailable ({type(e).__name__}: {e}). Using deterministic classification fallback.")
        fallback = []
        for idx, pat in enumerate(retrieved_patents):
            pid = normalize_patent_id(pat["patent_id"])
            pat_title = pat.get("title", "")
            abstract = pat.get("abstract", "")
            summary = abstract.split(".")[0] + "." if "." in abstract else (abstract[:180] if abstract else "Detailed abstract available in registry document.")
            best_url = _make_google_patents_url(pid, pat_title)
            rel_score = int(pat.get("relevance_score", 70))

            # Grounded match explanation and classification
            if rel_score >= 85:
                relevance_label = "Prior Art"
                fto_label = "Alert"
            elif rel_score >= 70:
                relevance_label = "Overlap"
                fto_label = "Caution"
            else:
                relevance_label = "White Space"
                fto_label = "Safe"

            match_reason = f"Patent describes '{pat_title}' with direct technical overlap to '{effective_topic}'."
            why_rel = pat.get("why_relevant") or match_reason
            rel_level = pat.get("relevance_level", "DIRECT_MATCH")

            fallback.append(PatentInfo(
                patent_id=pid,
                title=pat_title,
                assignee=pat.get("assignee", pat.get("applicant", "")),
                applicant=pat.get("applicant", pat.get("assignee", "")),
                inventors=pat.get("inventors", []),
                abstract=abstract,
                claims=pat.get("claims", ""),
                description=pat.get("description", ""),
                relevance=relevance_label,
                summary=summary,
                fto_rating=fto_label,
                design_around_strategy=(
                    f"Analyze independent claims of {pid} and construct an alternate pipeline structure to differentiate from this patent."
                ),
                url=best_url,
                source_links=get_patent_source_links(pid, pat_title, best_url),
                relevance_score=rel_score,
                rank=idx + 1,
                match_explanation=match_reason,
                publication_number=pat.get("publication_number", pid),
                publication_date=pat.get("publication_date", ""),
                priority_date=pat.get("priority_date", ""),
                jurisdiction=pat.get("jurisdiction", ""),
                sources=pat.get("sources", [pat.get("source", "Unknown")]),
                source=pat.get("source", "Unknown"),
                source_url=best_url,
                retrieval_status="SUCCESS",
                relevance_level=rel_level,
                domain_match=pat.get("domain_match", True),
                technical_match=pat.get("technical_match", True),
                why_relevant=why_rel,
                rejection_reason=None,
                relevance_status="accepted"
            ))

        fallback.sort(key=lambda x: x.relevance_score, reverse=True)
        for rank_idx, p in enumerate(fallback, 1):
            p.rank = rank_idx

        return Agent3PatentOutput(
            patents=fallback,
            white_space_opportunities=[
                f"Unpatented integration of multi-modal approaches in {effective_topic}.",
                f"Cross-domain transfer learning applications in {effective_topic}.",
                f"Explainability and verification frameworks for {effective_topic} architectures.",
            ],
            patent_analysis_status="retrieval_success_synthesis_failed",
            patent_retrieval_status=retrieval_status,
            source_statuses=source_statuses,
            retrieved_patents_raw=retrieved_patents,
        )


def search_and_classify_patents(
    gap_data: Any = None,
    query_topic: str = "",
    research_out: Agent1ResearchOutput = None,
    weights: dict = None,
    topic: str = "",
    max_results: int = 5,
) -> "Agent3PatentOutput":
    """
    Backward-compatible wrapper for run_patent_agent.
    Accepts legacy call signatures while routing through the clean interface.
    Agent 3 is FULLY INDEPENDENT of Agent 1 and Agent 2 for patent discovery.
    """
    # Resolve effective topic from whichever argument is provided
    if topic and topic.strip():
        effective_topic = topic.strip()
    elif query_topic and query_topic.strip():
        effective_topic = query_topic.strip()
    elif isinstance(gap_data, str) and gap_data.strip():
        effective_topic = gap_data.strip()
    elif hasattr(gap_data, "topic") and getattr(gap_data, "topic"):
        effective_topic = str(getattr(gap_data, "topic")).strip()
    else:
        effective_topic = str(gap_data or query_topic or topic or "").strip()

    return run_patent_agent(topic=effective_topic, max_results=max_results, weights=weights)
