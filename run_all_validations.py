import os
import sys

# Line buffering
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
sys.path.insert(0, BACKEND_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BACKEND_DIR, ".env"), override=True)

from agents.agent_patent import (
    expand_patent_queries,
    retrieve_multi_source_patents,
    evaluate_hard_relevance_gates,
    decompose_topic_concepts,
)

TOPICS = [
    "Agentic AI in multiagent systems",
    "multi-agent artificial intelligence",
    "AI agent orchestration",
    "large language model agents",
    "multi-agent LLM systems",
    "agentic AI",
    "Deepfake audio detection",
    "Medical image segmentation using deep learning"
]

def main():
    print("=" * 70)
    print("AGENT 3 MULTI-TOPIC CALIBRATION & LIVE VALIDATION")
    print("=" * 70)
    
    results = []
    
    for topic in TOPICS:
        print(f"\n>>> Running Topic: '{topic}'...")
        queries = expand_patent_queries(topic)
        relevant_patents, source_statuses, overall_status = retrieve_multi_source_patents(topic, max_results=5)
        
        direct_count = sum(1 for p in relevant_patents if p.get("relevance_level") == "DIRECT_MATCH")
        strong_count = sum(1 for p in relevant_patents if p.get("relevance_level") == "STRONG_RELATED")
        
        entry = {
            "topic": topic,
            "queries": queries,
            "source_statuses": source_statuses,
            "relevant_patents": relevant_patents,
            "direct_count": direct_count,
            "strong_count": strong_count,
            "final_count": len(relevant_patents),
            "overall_status": overall_status,
        }
        results.append(entry)
        
        print(f"    Done: {len(relevant_patents)} genuinely relevant patents (Direct: {direct_count}, Strong: {strong_count}, Status: {overall_status})")
        for idx, pat in enumerate(relevant_patents, 1):
            print(f"      [{idx}] {pat.get('patent_id')}: {pat.get('title')} ({pat.get('relevance_level')}, Score: {pat.get('relevance_score')})")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY REPORT")
    print("=" * 70)
    for r in results:
        print(f"\nTopic:                     {r['topic']}")
        print(f"Source Statuses:           {r['source_statuses']}")
        print(f"Direct Matches:            {r['direct_count']}")
        print(f"Strong Related:            {r['strong_count']}")
        print(f"Final Relevant Patents:    {r['final_count']}")
        print(f"Final Retrieval Status:    {r['overall_status']}")
        if r['relevant_patents']:
            for idx, pat in enumerate(r['relevant_patents'], 1):
                print(f"  {idx}. [{pat.get('patent_id')}] {pat.get('title')}")
                print(f"     Source:          {', '.join(pat.get('sources', [pat.get('source', 'Unknown')]))}")
                print(f"     Score:           {pat.get('relevance_score')}/100 ({pat.get('relevance_level')})")
                print(f"     Why Relevant:    {pat.get('why_relevant')}")

if __name__ == "__main__":
    main()
