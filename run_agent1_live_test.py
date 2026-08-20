"""
run_agent1_live_test.py — Live execution script for Section 19 test topic.
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

from topic_decomposition import normalize_research_topic, decompose_research_topic, generate_targeted_research_queries
from agents.agent_research import fetch_arxiv_papers

def main():
    test_topic = "Multi agents in Intelligience Artificial"
    print("=" * 80)
    print(f"LIVE AGENT 1 TEST: '{test_topic}'")
    print("=" * 80)

    norm_topic = normalize_research_topic(test_topic)
    print(f"\n[Spelling & Phrasing Normalization]")
    print(f"  Raw Input:       '{test_topic}'")
    print(f"  Normalized:      '{norm_topic}'")

    decomp = decompose_research_topic(test_topic)
    queries = generate_targeted_research_queries(test_topic)
    print(f"\n[Generated Focused Queries]")
    for idx, q in enumerate(queries, 1):
        print(f"  {idx}. '{q}'")

    print(f"\n[Executing Live Agent 1 Discovery & Gate-First Screening]")
    output = fetch_arxiv_papers(test_topic, max_results=6)

    print("\n" + "=" * 80)
    print("LIVE AGENT 1 SELECTION SUMMARY")
    print("=" * 80)
    print(f"Normalized Topic:                {norm_topic}")
    print(f"Total Diagnostics Evaluated:     {len(output.candidate_diagnostics)}")
    print(f"Final Research Papers Selected:  {len(output.papers)}")
    print(f"Full-Paper Eligibility Rate:     {output.full_paper_eligibility_rate * 100:.1f}%\n")

    for idx, p in enumerate(output.papers, 1):
        title_safe = str(p.title).encode('ascii', 'replace').decode('ascii')
        print(f"  [{idx}] {title_safe}")
        print(f"      Score:          {p.relevance_score} ({p.relevance_rank})")
        print(f"      Classification: {p.relevance_level}")
        print(f"      FT Source:      {p.full_text_source} ({p.full_text_word_count:,} words)")
        print(f"      Why relevant:   {p.why_selected}")
        print()

if __name__ == "__main__":
    main()
