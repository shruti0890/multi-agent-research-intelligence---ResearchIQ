from agents.agent_research import fetch_arxiv_papers
from agents.agent_gap import cluster_and_analyze_gaps

if __name__ == "__main__":
    topic = "Federated Learning Medical Imaging"
    print(f"=== Fetching papers for: '{topic}' ===")
    research_output = fetch_arxiv_papers(topic, max_results=3)
    
    print("\n=== Running Agent 2 Gap Analysis ===")
    gap_output = cluster_and_analyze_gaps(research_output)
    
    print("\n--- Identified Gaps ---")
    for idx, gap in enumerate(gap_output.gaps, 1):
        print(f"[{gap.severity}] Gap {idx}: {gap.description}")
        print(f"    Evidence Papers: {', '.join(gap.evidence_papers)}")
        
    print("\n--- Proposed Novel Method ---")
    print(f"Title: {gap_output.proposed_method.title}")
    print(f"Novelty Score: {gap_output.proposed_method.novelty_score}/100")
    print(f"Approach:\n{gap_output.proposed_method.approach}")
    print(f"Rationale: {gap_output.proposed_method.rationale}")
