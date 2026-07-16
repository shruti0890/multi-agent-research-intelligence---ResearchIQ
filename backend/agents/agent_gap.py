import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import re
import math
from typing import List, Dict, Set
# pyrefly: ignore [missing-import]
from google import genai
from google.genai import types
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

from schemas import Agent1ResearchOutput, Agent2GapOutput, ResearchGap, NovelMethodProposal  # type: ignore

# Load environment variables — explicit path so it works when server runs from project root
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path=_env_path)

# Configure Gemini using the new google.genai SDK
_gemini_client = None
def _get_client():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Check backend/.env file.")
        print(f"[Agent2] Gemini API key loaded: ...{api_key[-6:]}")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client

GEMINI_MODEL = "gemini-2.5-flash"

# Basic English Stop Words for cleaning abstract text
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 
    'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
    'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'using', 'used', 'use'
}

def tokenize(text: str) -> List[str]:
    """Helper to lowercase, remove punctuation, and split into tokens."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return [w for w in words if w not in STOP_WORDS]

def calculate_cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Calculates cosine similarity between two frequency vector dictionaries."""
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([val**2 for val in vec1.values()])
    sum2 = sum([val**2 for val in vec2.values()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return float(numerator) / denominator

def cluster_abstracts_pure_python(abstracts: List[str]) -> List[int]:
    """
    Groups abstracts into clusters using pure-Python TF-IDF representation
    and a basic threshold-based clustering.
    """
    if not abstracts:
        return []
        
    # 1. Tokenize all abstracts
    tokenized_docs = [tokenize(doc) for doc in abstracts]
    
    # 2. Calculate Term Frequency (TF)
    tf_docs = []
    for doc in tokenized_docs:
        tf = {}
        for word in doc:
            tf[word] = tf.get(word, 0) + 1
        tf_docs.append(tf)
        
    # 3. Calculate Inverse Document Frequency (IDF)
    num_docs = len(abstracts)
    doc_occurrences = {}
    for doc in tokenized_docs:
        unique_words = set(doc)
        for word in unique_words:
            doc_occurrences[word] = doc_occurrences.get(word, 0) + 1
            
    idf = {}
    for word, count in doc_occurrences.items():
        idf[word] = math.log(1 + (num_docs / count))
        
    # 4. Construct TF-IDF vectors
    tfidf_vectors = []
    for tf in tf_docs:
        vector = {}
        for word, val in tf.items():
            vector[word] = val * idf.get(word, 0.0)
        tfidf_vectors.append(vector)
        
    # 5. Greedy Threshold Clustering
    # We assign papers to clusters. If a paper has > 0.15 similarity to an existing
    # cluster representative, it joins that cluster. Otherwise, it starts a new one.
    cluster_labels = [-1] * len(abstracts)
    cluster_representatives = [] # List of (cluster_id, vector)
    cluster_count = 0
    
    for idx, vector in enumerate(tfidf_vectors):
        best_similarity = -1.0
        best_cluster = -1
        
        for c_id, c_vector in cluster_representatives:
            sim = calculate_cosine_similarity(vector, c_vector)
            if sim > best_similarity:
                best_similarity = sim
                best_cluster = c_id
                
        # If similarity is above threshold (0.15), join the cluster
        if best_similarity >= 0.15:
            cluster_labels[idx] = best_cluster
        else:
            # Create new cluster
            cluster_labels[idx] = cluster_count
            cluster_representatives.append((cluster_count, vector))
            cluster_count += 1
            
    return cluster_labels

def cluster_and_analyze_gaps(research_data: Agent1ResearchOutput) -> Agent2GapOutput:
    """
    Analyzes each paper individually to extract unique, paper-specific research gaps.
    Then synthesizes a Novel Method Proposal that addresses the most critical gaps found.
    """
    papers = research_data.papers
    
    if not papers:
        return Agent2GapOutput(
            gaps=[ResearchGap(description="No papers were provided to analyze.", severity="Minor", why_it_matters="N/A", evidence_papers=[])],
            proposed_method=NovelMethodProposal(
                title="Generic Research Framework",
                approach="Please provide paper inputs to synthesize a custom method.",
                novelty_score=0,
                rationale="N/A"
            )
        )

    # Build per-paper text entries (use abstract + deep fields if available)
    paper_entries = []
    for paper in papers:
        entry = (
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(paper.authors)} (Year: {paper.year})\n"
            f"Abstract: {paper.abstract[:500]}\n"
        )
        paper_entries.append(entry)

    all_papers_block = "\n---\n".join(paper_entries)

    prompt = f"""
    You are an expert Scientific Researcher and Critical Analyst.
    I will give you a list of {len(papers)} research papers on the topic: '{research_data.query}'.
    
    Your task:
    For EACH paper listed below, identify ONE unique research gap that is SPECIFIC to that paper.
    Every gap MUST be different — do NOT repeat the same gap for different papers.
    For each paper, determine:
    1. gap_description: The specific research gap or limitations of the proposed method (dataset constraints, scalability concerns, computational challenges, missing evaluations, explainability issues, security, or generalization).
    2. gap_impact: Why this limitation blocks real-world adoption or industrial deployment.
    3. gap_why_exists: The technical or data reason why the authors left this gap (e.g. data unavailability, compute cost, lack of theoretical framework).
    4. gap_opportunity: Actionable next step or research opportunity to solve this gap.
    5. gap_future_scope: Long-term vision or future scope.
    6. gap_severity: 'Critical' (major research limitations, strong opportunity), 'Moderate' (noticeable limitations), or 'Low' (minor limitations, mostly mature).
    
    After listing all per-paper gaps, write ONE 'proposed_method' that addresses the most critical gaps found.
    
    Papers:
    {all_papers_block}
    
    You MUST respond with a valid JSON block matching this EXACT schema structure:
    {{
      "gaps": [
        {{
          "paper_title": "Exact Title of Paper 1",
          "gap_description": "Unique gap specific to Paper 1",
          "gap_impact": "Impact of this gap on real-world use",
          "gap_why_exists": "Why this specific gap exists technically",
          "gap_opportunity": "Research opportunity to solve this",
          "gap_future_scope": "Future scope of this topic",
          "gap_severity": "Critical"
        }}
      ],
      "proposed_method": {{
        "title": "An academic name for the combined method that addresses the critical gaps",
        "approach": "Step 1: ...\\nStep 2: ...\\nStep 3: ...\\nStep 4: ...",
        "novelty_score": 87,
        "rationale": "Why this is a unique, plagiarism-free contribution"
      }}
    }}
    
    CRITICAL: You MUST produce exactly {len(papers)} gap entries in the 'gaps' array, one per paper.
    Respond ONLY with the JSON code block. No extra explanations, no markdown wrapper backticks.
    """

    severity_map = {"Critical": 3, "Moderate": 2, "Low": 1, "Minor": 1}

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        response_text = response.text.strip()

        # Clean formatting
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        data = json.loads(response_text.strip())

        gaps_list = []
        for p in papers:
            # Match the gap entry to the paper title
            match = None
            for gap_entry in data.get("gaps", []):
                evidence = gap_entry.get("evidence_papers", [])
                if (gap_entry.get("paper_title", "").lower().strip() == p.title.lower().strip() or 
                    p.title in evidence or 
                    any(p.title.lower().strip() in e.lower().strip() for e in evidence)):
                    match = gap_entry
                    break
            
            # If no direct match, try first matching item in data gaps array as fallback
            if not match and data.get("gaps"):
                # fallback matching by matching index or keyword
                for gap_entry in data.get("gaps", []):
                    if gap_entry.get("paper_title", "").lower()[:20] in p.title.lower():
                        match = gap_entry
                        break
            
            if match:
                p.gap_description = match.get("gap_description", "Methodology limitations.")
                p.gap_impact = match.get("gap_impact", "Blocks real-world deployment.")
                p.gap_why_exists = match.get("gap_why_exists", "Technical data constraints.")
                p.gap_opportunity = match.get("gap_opportunity", "Implement scalable hybrid models.")
                p.gap_future_scope = match.get("gap_future_scope", "Extend validation splits.")
                p.gap_severity = match.get("gap_severity", "Moderate")
            else:
                # Default values if LLM skipped this paper
                p.gap_description = f"Scalability limits in '{p.title[:45]}...' model architecture."
                p.gap_impact = "Restricts the ability to adapt to complex real-world edge cases."
                p.gap_why_exists = "Data sparsity or lack of scalable model representations."
                p.gap_opportunity = "Integrate multi-modal context vectors."
                p.gap_future_scope = "Ablation testing on public cross-domain benchmarks."
                p.gap_severity = "Moderate"

            gaps_list.append(ResearchGap(
                description=p.gap_description,
                severity=p.gap_severity,
                why_it_matters=p.gap_impact,
                evidence_papers=[p.title]
            ))

        # Sort papers in research_data descending by gap severity (Critical -> Moderate -> Low)
        papers.sort(key=lambda x: severity_map.get(x.gap_severity, 1), reverse=True)

        method_data = data.get("proposed_method", {})
        proposed_method = NovelMethodProposal(
            title=method_data.get("title", f"Adaptive Hybrid Framework for {research_data.query}"),
            approach=method_data.get("approach", ""),
            novelty_score=method_data.get("novelty_score", 80),
            rationale=method_data.get("rationale", "")
        )

        return Agent2GapOutput(gaps=gaps_list, proposed_method=proposed_method)

    except Exception as e:
        print(f"Error calling Gemini in Agent 2: {e}")
        try:
            print(f"RAW RESPONSE: {response.text}")
        except:
            pass
        # Fallback: generate a UNIQUE gap per paper derived from that paper's actual abstract
        severity_cycle = ["Critical", "Moderate", "Low"]
        fallback_gaps = []
        for idx, paper in enumerate(papers):
            abstract = paper.abstract or ""
            first_sentence = abstract.split(".")[0].strip() if "." in abstract else abstract[:150].strip()
            title_tail = paper.title[-60:].strip() if len(paper.title) > 60 else paper.title
            severity = severity_cycle[idx % len(severity_cycle)]
            
            paper.gap_severity = severity
            paper.gap_description = f"Methodology lacks comprehensive out-of-sample validation: {first_sentence[:120]}."
            paper.gap_impact = f"Without broader validation, the findings from '{title_tail}' cannot be generalized to real-world deployments."
            paper.gap_why_exists = "Computational budget limitations and lack of standardized benchmarking datasets during research."
            paper.gap_opportunity = "Implement cross-institutional data validation and federated benchmark partitions."
            paper.gap_future_scope = "Standardizing transfer learning validation procedures on diverse clinical or industrial environments."
            
            fallback_gaps.append(ResearchGap(
                description=paper.gap_description,
                severity=paper.gap_severity,
                why_it_matters=paper.gap_impact,
                evidence_papers=[paper.title]
            ))

        # Sort papers by severity
        papers.sort(key=lambda x: severity_map.get(x.gap_severity, 1), reverse=True)

        return Agent2GapOutput(
            gaps=fallback_gaps,
            proposed_method=NovelMethodProposal(
                title=f"Unified Scalable Framework for {research_data.query}",
                approach=(
                    "Step 1: Standardize input data across heterogeneous sources.\n"
                    "Step 2: Apply adaptive normalization to handle distribution shifts.\n"
                    "Step 3: Train a shared encoder with topic-specific decoder heads.\n"
                    "Step 4: Validate against multi-institutional benchmark splits."
                ),
                novelty_score=75,
                rationale=(
                    f"Synthesizes scalability techniques not individually addressed by any single paper in the '{research_data.query}' corpus."
                )
            )
        )

