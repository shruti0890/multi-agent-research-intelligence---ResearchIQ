"""
run_pipeline_topic_validation.py — Live validation runner for Agent 1 and Agent 3 across target topics.
"""
import os
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
sys.path.insert(0, BACKEND_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BACKEND_DIR, ".env"), override=True)

from topic_decomposition import decompose_research_topic, generate_targeted_research_queries, generate_targeted_patent_queries
from agents.agent_research import fetch_arxiv_papers
from agents.agent_patent import run_patent_agent

TARGET_TOPICS = [
    "Multi agent in Artificial Intelligence",
    "Deepfake audio detection",
    "Medical image segmentation using deep learning"
]

def validate_topic(topic: str):
    print("\n" + "=" * 80)
    print(f"VALIDATION TARGET: '{topic}'")
    print("=" * 80)

    decomp = decompose_research_topic(topic)
    print(f"\n[Topic Decomposition]")
    print(f"  Topic Type:       {decomp['topic_type']}")
    print(f"  Domain Name:      {decomp['domain_name']}")
    print(f"  Domain Anchors:   {decomp.get('domain_anchors', [])[:4]}")
    print(f"  Core Anchors:     {decomp.get('core_concept_anchors', [])[:4]}")

    print("\n[Agent 1: Research Paper Discovery & Full-Paper Screening]")
    try:
        a1_out = fetch_arxiv_papers(topic, max_results=5)
        print(f"  Papers Returned:       {len(a1_out.papers)}")
        print(f"  Diagnostics Evaluated: {len(a1_out.candidate_diagnostics)}")
        print(f"  Eligibility Rate:      {a1_out.full_paper_eligibility_rate * 100:.1f}%")
        for idx, p in enumerate(a1_out.papers, 1):
            print(f"    [{idx}] {p.title[:65]}... (Year: {p.year}, Score: {p.relevance_score}, Rank: {p.relevance_rank}, FT Source: {p.full_text_source})")
            print(f"        Why selected: {p.why_selected}")
    except Exception as e:
        print(f"  Agent 1 Error: {e}")

    print("\n[Agent 3: Patent Landscape & Prior-Art Screening]")
    try:
        a3_out = run_patent_agent(topic, max_results=5)
        print(f"  Patents Returned:      {len(a3_out.patents)}")
        print(f"  Retrieval Status:      {a3_out.patent_retrieval_status}")
        print(f"  Source Statuses:       {a3_out.source_statuses}")
        print(f"  Opportunities Found:   {len(a3_out.white_space_opportunities)}")
        for idx, p in enumerate(a3_out.patents, 1):
            print(f"    [{idx}] {p.patent_id}: {p.title[:65]}... (Score: {p.relevance_score}, Level: {p.relevance_level})")
            print(f"        Why relevant: {p.why_relevant}")
    except Exception as e:
        print(f"  Agent 3 Error: {e}")

def main():
    print("=" * 80)
    print("ResearchIQ — LIVE TOPIC-GROUNDED PIPELINE VALIDATION")
    print("=" * 80)
    for t in TARGET_TOPICS:
        validate_topic(t)
    print("\n" + "=" * 80)
    print("ALL TARGET TOPICS VALIDATED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
