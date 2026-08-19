import os
from schemas import ProjectReportState
from agents.agent_research import fetch_arxiv_papers
from agents.agent_gap import cluster_and_analyze_gaps
from agents.agent_patent import search_and_classify_patents
from agents.agent_compile import compile_final_report

if __name__ == "__main__":
    topic = "Federated Learning Medical Imaging"
    print(f"=== [Pipeline Run] Testing All Agents and PDF Compilation ===")
    
    # 1. Agent 1
    print("\n[1/4] Running Agent 1: Research Discovery...")
    research_out = fetch_arxiv_papers(topic, max_results=3)
    print(f"    Fetched {len(research_out.papers)} papers successfully.")
    
    # 2. Agent 2
    print("\n[2/4] Running Agent 2: Gap Analysis...")
    gap_out = cluster_and_analyze_gaps(research_out)
    print(f"    Novel Method Title: {gap_out.proposed_method.title}")
    
    # 3. Agent 3
    print("\n[3/4] Running Agent 3: Patent Discovery...")
    patent_out = search_and_classify_patents(gap_out, topic)
    print(f"    Classified {len(patent_out.patents)} patents.")
    
    # 4. Bind into Unified State
    state = ProjectReportState(
        topic=topic,
        research=research_out,
        gaps=gap_out,
        patents=patent_out
    )
    
    # 5. Agent 4 Compile PDF
    print("\n[4/4] Running Agent 4: PDF Report Compilation...")
    pdf_path, compiler_status = compile_final_report(state, output_dir=".")

    
    print(f"\n=== SUCCESS ===")
    print(f"Final PDF generated at: {pdf_path}")
    print(f"Does file exist? {os.path.exists(pdf_path)}")
