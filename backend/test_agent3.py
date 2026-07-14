from agents.agent_research import fetch_arxiv_papers
from agents.agent_gap import cluster_and_analyze_gaps
from agents.agent_patent import search_and_classify_patents

if __name__ == "__main__":
    topic = "Federated Learning Medical Imaging"
    print(f"=== Running Agent 1 for: '{topic}' ===")
    research_output = fetch_arxiv_papers(topic, max_results=2)
    
    print("\n=== Running Agent 2 Gap Analysis ===")
    gap_output = cluster_and_analyze_gaps(research_output)
    
    print("\n=== Running Agent 3 Patent Evaluation ===")
    patent_output = search_and_classify_patents(gap_output, topic)
    
    print("\n--- Classified Patents ---")
    for pat in patent_output.patents:
        print(f"\n[{pat.relevance}] Patent ID: {pat.patent_id}")
        print(f"    Title: {pat.title}")
        print(f"    Assignee: {pat.assignee}")
        print(f"    Summary: {pat.summary}")
        if pat.url:
            print(f"    Link: {pat.url}")
            
    print("\n--- Unexplored White Space Opportunities ---")
    for idx, opp in enumerate(patent_output.white_space_opportunities, 1):
        print(f"[{idx}] {opp}")
