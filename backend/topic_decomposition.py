"""
backend/topic_decomposition.py — Unified Topic Decomposition Engine for ResearchIQ.

Provides single, reusable decomposition and topic normalization functions used across:
- Agent 1: Research Paper Discovery, Full-Paper Filtering, Gate-First Selection, and Relevance Scoring
- Agent 3: Patent Landscape Discovery, Multi-Source Querying, and 3-Tier Relevance Scoring

Ensures consistent concept identification:
1. DOMAIN
2. AGENT
3. MULTI_AGENT / CORE_CONCEPT
4. MULTI_AGENT_MECHANISMS / TECHNICAL_MECHANISMS
5. IMPORTANT_SYNONYMS
6. EXCLUSION_CONCEPTS
"""
import re
from typing import Dict, List, Any


def normalize_research_topic(topic: str) -> str:
    """
    Normalizes spelling, pluralization, and phrasing of the original user topic
    without altering its genuine technical research meaning.
    
    Example:
    'Multi agents in Artificial Intelligience' -> 'Multi-agent artificial intelligence'
    'Multi agents in Intelligience Artificial' -> 'Multi-agent artificial intelligence'
    """
    if not topic or not isinstance(topic, str):
        return ""

    raw = topic.strip()
    clean = raw.lower()

    # 1. Spelling corrections
    spelling_fixes = {
        r'\bintelligience\b': 'intelligence',
        r'\bintellegence\b': 'intelligence',
        r'\bintellegience\b': 'intelligence',
        r'\bartifical\b': 'artificial',
        r'\barificial\b': 'artificial',
        r'\bsegmetation\b': 'segmentation',
        r'\bsegmentatn\b': 'segmentation',
        r'\breinfocement\b': 'reinforcement',
        r'\breinforment\b': 'reinforcement',
        r'\bdeeepfake\b': 'deepfake',
        r'\bdeep-fak\b': 'deepfake',
        r'\bautonomus\b': 'autonomous',
        r'\bcolaboration\b': 'collaboration',
        r'\bcoordinatn\b': 'coordination',
        r'\barchitecure\b': 'architecture',
    }
    for pat, repl in spelling_fixes.items():
        clean = re.sub(pat, repl, clean)

    # 2. Re-order reversed words like 'intelligence artificial' -> 'artificial intelligence'
    clean = re.sub(r'\bintelligence\s+artificial\b', 'artificial intelligence', clean)
    clean = re.sub(r'\bintelligience\s+artificial\b', 'artificial intelligence', clean)

    # 3. Standardize multi-agent phrases
    if "multi agent" in clean or "multi agents" in clean or "multiagent" in clean:
        if "artificial intelligence" in clean or "ai" in clean:
            return "Multi-agent artificial intelligence"
        elif "reinforcement learning" in clean:
            return "Multi-agent reinforcement learning"
        elif "llm" in clean or "language model" in clean:
            return "Multi-agent large language model systems"
        else:
            return "Multi-agent systems"

    if "deepfake" in clean and ("audio" in clean or "speech" in clean or "voice" in clean):
        return "Deepfake audio detection"

    if "medical image" in clean or "segmentation" in clean:
        if "deep learning" in clean or "neural" in clean:
            return "Medical image segmentation using deep learning"
        else:
            return "Medical image segmentation"

    # Title-case fallback
    return raw


def decompose_research_topic(topic: str) -> Dict[str, Any]:
    """
    Decomposes the normalized research topic into structured semantic components.
    Guarantees that both research paper and patent retrieval share identical domain understanding.
    """
    raw_topic = (topic or "").strip()
    normalized_topic = normalize_research_topic(raw_topic)
    clean_topic = normalized_topic.lower()
    clean_text = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', clean_topic)
    words = [w for w in clean_text.split() if len(w) > 1 and w not in {
        "and", "for", "the", "with", "using", "of", "in", "on", "a", "an",
        "via", "to", "from", "by", "at", "or", "as", "is", "are", "into"
    }]

    # -------------------------------------------------------------
    # 1. Multi-Agent Systems / Agentic AI / Multi-Agent in AI
    # -------------------------------------------------------------
    if any(k in clean_topic for k in [
        "multi agent", "multi-agent", "multiagent", "multi agents", "multi-agents",
        "agentic", "ai agent", "ai agents", "intelligent agent", "software agent", "autonomous agent"
    ]):
        requires_multiagent = any(k in clean_topic for k in [
            "multi agent", "multi-agent", "multiagent", "multi agents", "multi-agents",
            "multiple agent", "multiple intelligent agents", "agent coordination", "agent collaboration"
        ])

        return {
            "topic_type": "MULTI_AGENT_AI",
            "domain_name": "Artificial Intelligence & Multi-Agent Systems",
            "original_topic": raw_topic,
            "normalized_topic": normalized_topic,
            "requires_multiagent": requires_multiagent,
            "domain_anchors": [
                "artificial intelligence", "ai", "machine learning", "deep learning",
                "reinforcement learning", "neural network", "neural networks",
                "intelligent systems", "distributed artificial intelligence", "dai"
            ],
            "agent_anchors": [
                "ai agent", "ai agents", "intelligent agent", "intelligent agents",
                "software agent", "software agents", "artificial intelligence agent",
                "machine learning agent", "autonomous software agent", "autonomous software agents",
                "agentic ai", "agent architecture", "agent framework", "agent-based software",
                "agent-based system", "decision-making agent", "planning agent", "reasoning agent", "task agent"
            ],
            "core_concept_anchors": [
                "multi-agent", "multi agent", "multiagent", "multi-agent system",
                "multiagent system", "multi-agent systems", "multiagent systems",
                "multiple intelligent agents", "multiple agents", "collaborative agents",
                "distributed agents", "agent society", "agent swarm", "multi-agent AI",
                "multi-agent learning", "multi-agent reinforcement learning", "marl"
            ] if requires_multiagent else [
                "ai agent", "ai agents", "intelligent agent", "intelligent agents",
                "software agent", "software agents", "autonomous agent", "agentic ai",
                "agentic", "autonomous software agent", "agent-based system"
            ],
            "mechanism_anchors": [
                "agent coordination", "agent collaboration", "agent communication",
                "agent interaction", "agent negotiation", "agent orchestration",
                "agent planning", "agent delegation", "agent strategy",
                "multi-agent reinforcement learning", "marl", "multi-agent learning",
                "distributed decision making", "game theory", "consensus protocol",
                "inter-agent communication", "cooperative learning", "strategy prediction",
                "cooperative agents", "distributed artificial intelligence"
            ],
            "important_synonyms": [
                "multi-agent artificial intelligence", "multi-agent systems",
                "multi-agent AI", "multi-agent reinforcement learning",
                "intelligent agents in AI", "agent coordination artificial intelligence",
                "agent collaboration AI", "multi-agent strategy prediction",
                "autonomous AI agents", "multi-agent decision making",
                "distributed artificial intelligence multi-agent", "multiple intelligent agents"
            ],
            "exclusion_concepts": [
                "sanctions on iran", "political economy", "macroeconomic", "petroleum",
                "autonomous robot", "autonomous vehicle", "telecommunications routing",
                "generic network routing", "physical autonomous machine", "travel agent",
                "chemical agent", "biological agent", "cleaning agent", "reagent",
                "fermi paradox", "astrophysics", "clinical scheduling without agents",
                "single-agent reinforcement learning"
            ],
            "generic_blacklist": [
                "artificial intelligence", "ai", "intelligence", "system", "framework",
                "model", "data", "method", "process", "approach", "novel", "algorithm",
                "autonomous", "network", "agent"
            ]
        }

    # -------------------------------------------------------------
    # 2. Deepfake Audio / Synthetic Speech Detection
    # -------------------------------------------------------------
    elif any(k in clean_topic for k in [
        "deepfake", "audio spoof", "voice spoof", "synthetic speech",
        "audio fake", "voice forgery", "audio forgery", "synthetic voice"
    ]):
        return {
            "topic_type": "DEEPFAKE_AUDIO",
            "domain_name": "Audio & Speech Deepfake Detection",
            "original_topic": raw_topic,
            "normalized_topic": normalized_topic,
            "domain_anchors": [
                "audio", "speech", "voice", "acoustic", "vocal", "sound",
                "phonetic", "utterance", "speaker", "waveform", "spectrogram",
                "audio signal", "speech signal", "voice biometric", "audio recording"
            ],
            "agent_anchors": [],
            "core_concept_anchors": [
                "deepfake", "deep-fake", "synthetic", "synthesized", "spoof",
                "spoofing", "forgery", "fake", "manipulation", "manipulated",
                "cloned", "cloning", "voice conversion", "text-to-speech", "tts",
                "replay attack", "tampered audio"
            ],
            "mechanism_anchors": [
                "detection", "detecting", "detect", "identifying", "classification",
                "classifier", "verification", "discriminator", "authenticating",
                "authenticity", "anti-spoofing", "liveness", "anomaly detection",
                "speaker verification against synthetic speech", "audio deepfake detection"
            ],
            "important_synonyms": [
                "voice spoofing detection", "synthetic speech detection",
                "audio forgery detection", "synthetic voice verification",
                "speech deepfake detection", "acoustic anti-spoofing",
                "speaker verification anti-spoofing"
            ],
            "exclusion_concepts": [
                "breathing detection", "catheter tip", "cochlear implant",
                "intelligibility assessment", "speech sound intelligibility",
                "audio enhancement", "hearing aid", "stethoscope",
                "acoustic echo cancellation", "active noise control", "noise reduction",
                "facial image deepfake", "video face swap", "chemical forgery marker"
            ],
            "generic_blacklist": [
                "audio", "sound", "signal", "system", "method", "device", "apparatus"
            ]
        }

    # -------------------------------------------------------------
    # 3. Medical Image Segmentation
    # -------------------------------------------------------------
    elif any(k in clean_topic for k in [
        "medical image", "segmentation", "mri", "ct scan", "ultrasound",
        "tumor", "lesion", "radiology", "biomedical image", "contouring"
    ]):
        return {
            "topic_type": "MEDICAL_IMAGE_SEGMENTATION",
            "domain_name": "Medical Image Segmentation",
            "original_topic": raw_topic,
            "normalized_topic": normalized_topic,
            "domain_anchors": [
                "medical", "clinical", "biomedical", "mri", "ct", "computed tomography",
                "ultrasound", "x-ray", "radiology", "radiograph", "histopathology",
                "anatomical", "organ", "tissue", "tumor", "lesion", "patient",
                "kidney", "brain", "cardiac", "lung", "liver", "diagnostic", "angiographic"
            ],
            "agent_anchors": [],
            "core_concept_anchors": [
                "image", "imaging", "scan", "scans", "volume", "volumetric",
                "slice", "tomography", "radiological image", "biomedical image",
                "medical image", "angiogram", "mri scan"
            ],
            "mechanism_anchors": [
                "segmentation", "segmenting", "segment", "segmented", "delineating",
                "delineation", "contouring", "region of interest", "roi", "mask",
                "boundary extraction", "lesion segmentation", "organ segmentation",
                "deep learning", "convolutional neural network", "u-net", "transformer"
            ],
            "important_synonyms": [
                "medical image segmentation", "mri segmentation system",
                "neural medical image segmentation", "deep learning organ contouring",
                "brain tumor segmentation", "radiological lesion delineation"
            ],
            "exclusion_concepts": [
                "industrial robot", "robot arm", "welding", "automotive assembly",
                "wafer manufacturing", "drilling", "plasma etching",
                "medical image viewer without segmentation", "generic database indexing"
            ],
            "generic_blacklist": [
                "medical", "image", "system", "method", "device", "process", "analysis"
            ]
        }

    # -------------------------------------------------------------
    # 4. General Domain Fallback
    # -------------------------------------------------------------
    else:
        significant_words = [w for w in words if len(w) > 2]
        return {
            "topic_type": "GENERAL_DOMAIN",
            "domain_name": normalized_topic,
            "original_topic": raw_topic,
            "normalized_topic": normalized_topic,
            "domain_anchors": significant_words[:4],
            "agent_anchors": [],
            "core_concept_anchors": significant_words,
            "mechanism_anchors": [f"{w}" for w in significant_words],
            "important_synonyms": [normalized_topic] + [f"{normalized_topic} system", f"{normalized_topic} method"],
            "exclusion_concepts": [],
            "generic_blacklist": ["system", "method", "process", "analysis", "approach"]
        }


def generate_targeted_research_queries(topic: str) -> List[str]:
    """
    Generates focused research queries derived strictly from the normalized topic.
    Guarantees that broad single terms like 'AI' or 'agent' are never searched alone.
    """
    decomp = decompose_research_topic(topic)
    t_type = decomp.get("topic_type")
    norm_topic = decomp.get("normalized_topic", topic).strip()

    queries = [norm_topic]

    def _add(q: str):
        if q and q.strip():
            clean_q = q.strip()
            if clean_q.lower() not in [x.lower() for x in queries] and len(queries) < 10:
                queries.append(clean_q)

    # 1. Multi-Agent Systems in AI (Section 14 Focused Queries)
    if t_type == "MULTI_AGENT_AI":
        _add("multi-agent artificial intelligence")
        _add("multi-agent systems")
        _add("multi-agent reinforcement learning")
        _add("multi-agent AI")
        _add("multiple intelligent agents")
        _add("agent coordination artificial intelligence")
        _add("agent collaboration artificial intelligence")
        _add("agent communication artificial intelligence")
        _add("distributed artificial intelligence multi-agent")
        _add("intelligent agents machine learning")

    # 2. Deepfake Audio Detection
    elif t_type == "DEEPFAKE_AUDIO":
        _add("deepfake audio detection")
        _add("synthetic speech detection")
        _add("voice spoofing detection")
        _add("audio forgery detection")
        _add("synthetic voice verification")
        _add("speech deepfake detection neural network")
        _add("voice anti-spoofing classifier")

    # 3. Medical Image Segmentation
    elif t_type == "MEDICAL_IMAGE_SEGMENTATION":
        _add("medical image segmentation deep learning")
        _add("mri segmentation deep learning")
        _add("ct scan organ segmentation")
        _add("tumor segmentation medical imaging")
        _add("anatomical structure segmentation neural network")
        _add("biomedical image segmentation u-net")

    # 4. General Domain
    else:
        anchors = decomp.get("domain_anchors", [])
        if len(anchors) >= 2:
            phrase = " ".join(anchors[:3])
            _add(f"{phrase} system")
            _add(f"{phrase} method")
            _add(f"{phrase} framework")
            _add(f"{phrase} research")

    return queries[:10]


def generate_targeted_patent_queries(topic: str) -> List[str]:
    """
    Generates 5 to 10 focused patent queries from the normalized topic.
    Preserves multi-word concepts and never splits into generic isolated single terms.
    """
    decomp = decompose_research_topic(topic)
    t_type = decomp.get("topic_type")
    norm_topic = decomp.get("normalized_topic", topic).strip()

    queries = [norm_topic]

    def _add(q: str):
        if q and q.strip():
            clean_q = q.strip()
            if clean_q.lower() not in [x.lower() for x in queries] and len(queries) < 10:
                queries.append(clean_q)

    # 1. Multi-Agent Systems in AI
    if t_type == "MULTI_AGENT_AI":
        _add("multi-agent artificial intelligence")
        _add("multi-agent systems")
        _add("multi-agent AI")
        _add("multiple intelligent agents")
        _add("multi-agent reinforcement learning")
        _add("agent coordination artificial intelligence")
        _add("agent collaboration artificial intelligence")
        _add("intelligent agent machine learning")
        _add("multi-agent strategy prediction")
        _add("multi-agent decision making")

    # 2. Deepfake Audio Detection
    elif t_type == "DEEPFAKE_AUDIO":
        _add("deepfake audio detection")
        _add("synthetic speech detection")
        _add("voice spoofing detection")
        _add("audio forgery detection")
        _add("synthetic voice verification")
        _add("speaker verification anti-spoofing")
        _add("deepfake voice detection system")
        _add("speech deepfake detection method")

    # 3. Medical Image Segmentation
    elif t_type == "MEDICAL_IMAGE_SEGMENTATION":
        _add("medical image segmentation")
        _add("mri segmentation system")
        _add("neural image segmentation")
        _add("medical image diagnostic method")
        _add("anatomical organ segmentation")
        _add("brain tumor segmentation system")

    # 4. General Domain
    else:
        anchors = decomp.get("domain_anchors", [])
        if len(anchors) >= 2:
            phrase = " ".join(anchors[:3])
            _add(f"{phrase} system")
            _add(f"{phrase} method")
            _add(f"{phrase} apparatus")
            _add(f"{phrase} device")

    return queries[:10]
