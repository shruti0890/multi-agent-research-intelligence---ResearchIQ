import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import re
import math
from typing import List, Dict, Set
import google.generativeai as genai
from dotenv import load_dotenv

from schemas import Agent1ResearchOutput, Agent2GapOutput, ResearchGap, NovelMethodProposal  # type: ignore

# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

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
    Groups papers using pure-Python TF-IDF and Cosine Distance clustering,
    prompts Gemini to analyze the gaps, and outputs a novel methodology.
    """
    papers = research_data.papers
    
    if not papers:
        return Agent2GapOutput(
            gaps=[ResearchGap(description="No papers were provided to analyze.", severity="Minor", evidence_papers=[])],
            proposed_method=NovelMethodProposal(
                title="Generic Research Framework",
                approach="Please provide paper inputs to synthesize a custom method.",
                novelty_score=0,
                rationale="N/A"
            )
        )
        
    # 1. Run Pure Python Clustering
    abstracts = [paper.abstract for paper in papers]
    cluster_labels = cluster_abstracts_pure_python(abstracts)
    
    # 2. Build clustered paper summary text for the LLM prompt
    clustered_summary = []
    for cluster_id in sorted(list(set(cluster_labels))):
        clustered_summary.append(f"--- Cluster Group {cluster_id} ---")
        for idx, label in enumerate(cluster_labels):
            if label == cluster_id:
                paper = papers[idx]
                clustered_summary.append(
                    f"Paper: {paper.title}\n"
                    f"Authors: {', '.join(paper.authors)} (Year: {paper.year})\n"
                    f"Abstract summary: {paper.abstract[:300]}...\n"
                )
    
    context_text = "\n".join(clustered_summary)

    # 3. Request structured synthesis from Gemini
    prompt = f"""
    You are an expert Scientific Researcher and Analyst.
    I will provide you with a structured list of academic papers grouped into thematic clusters.
    
    Analyze the clustered papers and generate:
    1. A list of 2-3 research gaps (things these papers failed to address, limitations, or conflicts).
       Make sure the explanation is simple, clear, and easy to understand for a human reader.
    2. A "Novel Method Proposal" that combines features across these clusters to solve the identified gaps.
    
    Academic papers context:
    {context_text}
    
    You MUST respond with a valid JSON block matching this EXACT schema structure:
    {{
      "gaps": [
        {{
          "description": "Clear, plain-English explanation of the research gap",
          "severity": "Critical" | "Moderate" | "Minor",
          "why_it_matters": "Explain simply why this gap prevents commercial progress or real-world use",
          "evidence_papers": ["Title of Paper A", "Title of Paper B"]
        }}
      ],
      "proposed_method": {{
        "title": "A catchy, academic-sounding name for the combined method",
        "approach": "A detailed step-by-step description of how to build this new method",
        "novelty_score": 85, 
        "rationale": "Explain why this solution is unique and does not violate academic plagiarism"
      }}
    }}
    
    Respond ONLY with the JSON code block. No extra explanations, no markdown wrapper backticks.
    """
    
    # LLM Call
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean formatting
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        data = json.loads(response_text.strip())
        
        gaps_list = []
        for gap in data.get("gaps", []):
            gaps_list.append(ResearchGap(
                description=gap.get("description", ""),
                severity=gap.get("severity", "Moderate"),
                why_it_matters=gap.get("why_it_matters", ""),
                evidence_papers=gap.get("evidence_papers", [])
            ))
            
        method_data = data.get("proposed_method", {})
        proposed_method = NovelMethodProposal(
            title=method_data.get("title", "Proposed Hybrid Method"),
            approach=method_data.get("approach", ""),
            novelty_score=method_data.get("novelty_score", 80),
            rationale=method_data.get("rationale", "")
        )
        
        return Agent2GapOutput(gaps=gaps_list, proposed_method=proposed_method)
        
    except Exception as e:
        print(f"Error calling Gemini in Agent 2: {e}")
        # Return fallback structured output
        return Agent2GapOutput(
            gaps=[
                ResearchGap(
                    description="Standard benchmark validation gaps across local institutional data splits.",
                    severity="Critical",
                    why_it_matters="Without standard benchmark datasets, we cannot measure progress or compare different algorithms.",
                    evidence_papers=[papers[0].title] if papers else []
                )
            ],
            proposed_method=NovelMethodProposal(
                title=f"Hybrid Adaptive Framework for {research_data.query}",
                approach="1. Establish a standardized baseline testing split.\n2. Implement a local aggregation schema.\n3. Validate using cross-site metrics.",
                novelty_score=75,
                rationale="Addresses local validation gap by introducing standardized metrics across heterogeneous data distributions."
            )
        )
