from agents.agent_research import fetch_arxiv_papers

if __name__ == "__main__":
    test_query = "Federated Learning Medical Imaging"
    print(f"=== Testing Agent 1 for query: '{test_query}' ===")
    
    result = fetch_arxiv_papers(test_query, max_results=3)
    
    print(f"Results Count: {len(result.papers)}")
    for idx, paper in enumerate(result.papers, 1):
        print(f"\n[{idx}] {paper.title}")
        print(f"    Authors: {', '.join(paper.authors)}")
        print(f"    Year: {paper.year}")
        print(f"    URL: {paper.url}")
        print(f"    Abstract Snippet: {paper.abstract[:150]}...")
