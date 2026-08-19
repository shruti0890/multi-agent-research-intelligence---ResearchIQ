import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.agent_research import fetch_arxiv_papers
from agents.agent_gap import cluster_and_analyze_gaps, is_evidence_eligible

def run_verification():
    query = 'Ecology and ecosystem'
    print(f'=== RUNNING VERIFICATION FOR: {query} ===\n')
    
    # 1. Run Agent 1
    research_output = fetch_arxiv_papers(query, max_results=8)
    papers = research_output.papers
    print(f'Top {len(papers)} papers selected:')
    for p in papers:
        eligible = is_evidence_eligible(p)
        print(f'  [{p.paper_id}] {p.title[:65]}...')
        print(f'       Coverage: {p.coverage_type} | Evidence-eligible: {eligible} | Rank: {p.relevance_rank} (score: {p.relevance_score})')
        if hasattr(p, 'fact_sheet') and p.fact_sheet:
            fs = p.fact_sheet
            detected_sec = list(fs.sections.keys())
            tech_sents = sum(s.technical_sentence_count for s in fs.sections.values())
            print(f'       Sections detected: {detected_sec}')
            print(f'       Comp ratio: {p.compression_ratio*100:.1f}% | Tech sentences: {tech_sents} | Tokens: {p.original_tokens} -> {p.compressed_tokens}')
        print()

    # 2. Run Agent 2
    gap_output = cluster_and_analyze_gaps(research_output)
    print('\n=== GAP ANALYSIS RESULTS ===')
    for g in gap_output.gaps:
        print(f'Gap: {g.description[:80]}...')
        print(f'  Severity: {g.severity}')
        print(f'  Evidence papers: {g.evidence_papers}')
        if g.evidence_references:
            print(f'  Evidence refs: {g.evidence_references}')
        print()

    print('=== NOVEL METHOD PROPOSAL ===')
    print(f'Title: {gap_output.proposed_method.title}')
    print(f'Novelty: {gap_output.proposed_method.novelty_score}/100')
    print(f'Rationale: {gap_output.proposed_method.rationale[:100]}...')
    print('\nVERIFICATION COMPLETE')

if __name__ == '__main__':
    run_verification()
