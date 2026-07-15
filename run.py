import os
import sys
import time

# Ensure backend folder is in path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

# pyrefly: ignore [missing-import]
from schemas import ProjectReportState
# pyrefly: ignore [missing-import]
from agents.agent_research import fetch_arxiv_papers
# pyrefly: ignore [missing-import]
from agents.agent_gap import cluster_and_analyze_gaps
# pyrefly: ignore [missing-import]
from agents.agent_patent import search_and_classify_patents
# pyrefly: ignore [missing-import]
from agents.agent_compile import compile_final_report

def run_interactive_pipeline():
    print("=" * 60)
    print("🔬 ResearchIQ: Multi-Agent Research & Patent Discovery Engine")
    print("=" * 60)
    
    # 1. Check for API key
    # pyrefly: ignore [missing-import]
    from dotenv import load_dotenv
    load_dotenv(os.path.join("backend", ".env"))
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key or api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE" or api_key.strip() == "":
        print("\n[ERROR] Gemini API Key is missing!")
        print("Please open backend/.env and paste your Google AI Studio API key first.")
        return

    # 2. Get User Input
    topic = input("\nEnter your research topic: ").strip()
    if not topic:
        print("[Error] Topic cannot be empty.")
        return
        
    print(f"\n🚀 Starting pipeline for: '{topic}'\n")
    time.sleep(1)

    # 3. Agent 1: Research Paper Discovery (arXiv + Semantic Scholar + Crossref)
    print("🔍 [Agent 1] Searching literature databases (arXiv, Semantic Scholar, Crossref)...")
    research_out = fetch_arxiv_papers(topic, max_results=4)
    print(f"    -> Done. Discovered and ranked {len(research_out.papers)} unique papers.")
    for idx, paper in enumerate(research_out.papers, 1):
        print(f"       [{idx}] {paper.title} ({paper.year})")
    print("-" * 50)
    time.sleep(1)

    # 4. Agent 2: Gap Analysis & Novel Synthesis
    print("📊 [Agent 2] Clustering papers and analyzing research gaps...")
    gap_out = cluster_and_analyze_gaps(research_out)
    print(f"    -> Done. Found {len(gap_out.gaps)} critical research gaps.")
    print(f"    -> Proposed Novel Method: '{gap_out.proposed_method.title}'")
    print("-" * 50)
    time.sleep(1)

    # 5. Agent 3: Patent Landscape Classification
    print("📜 [Agent 3] Querying USPTO database and evaluating IP risk...")
    patent_out = search_and_classify_patents(gap_out, topic)
    print(f"    -> Done. Evaluated {len(patent_out.patents)} patents matching proposed method.")
    for pat in patent_out.patents:
        print(f"       * [{pat.relevance}] Patent {pat.patent_id} - Assignee: {pat.assignee}")
    print("-" * 50)
    time.sleep(1)

    # 6. Agent 4: Compile State & PDF Generation
    print("🤝 [Agent 4] Synthesizing final report and compiling PDF layout...")
    state = ProjectReportState(
        topic=topic,
        research=research_out,
        gaps=gap_out,
        patents=patent_out
    )
    
    pdf_path = compile_final_report(state, output_dir=".")
    print("=" * 60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Final Report PDF saved at:\n{pdf_path}")
    print("=" * 60)
    
    # 7. Auto-open PDF on Windows
    try:
        print("\nOpening the PDF report for you...")
        os.startfile(pdf_path)
    except Exception as e:
        print(f"Could not auto-open PDF, please locate the file manually at: {pdf_path}")

if __name__ == "__main__":
    run_interactive_pipeline()
