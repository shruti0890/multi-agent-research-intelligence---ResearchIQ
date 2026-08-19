import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from schemas import ProjectReportState
from agents.agent_research import fetch_arxiv_papers
from agents.agent_gap import cluster_and_analyze_gaps
from agents.agent_patent import search_and_classify_patents
from agents.agent_compile import compile_final_report
from evaluator import run_evaluation

def test_full_pipeline():
    topic = "Attention Mechanism in Vision Transformers"
    print(f"=== [E2E Verification Run] Topic: {topic} ===")

    # 1. Agent 1
    print("\n[1/4] Running Agent 1: Research Discovery + Extractive Compression...")
    research_out = fetch_arxiv_papers(topic, max_results=3)
    print(f"  Fetched {len(research_out.papers)} papers.")
    for idx, p in enumerate(research_out.papers):
        print(f"  [{idx+1}] {p.title[:45]} | Coverage: {p.coverage_type} | Source: {p.full_text_source} | CompRatio: {p.compression_ratio*100:.1f}%")

    # 2. Agent 2
    print("\n[2/4] Running Agent 2: Gap Analysis & Synthesis...")
    gap_out = cluster_and_analyze_gaps(research_out)
    print(f"  Novel Method: {gap_out.proposed_method.title}")

    # 3. Agent 3
    print("\n[3/4] Running Agent 3: Patent Landscape...")
    patent_out = search_and_classify_patents(gap_out, topic, research_out)
    print(f"  Patents analyzed: {len(patent_out.patents)}")

    # 4. Construct Unified State
    state = ProjectReportState(
        topic=topic,
        research=research_out,
        gaps=gap_out,
        patents=patent_out
    )

    # 5. Agent 4: Compile PDF
    print("\n[4/4] Running Agent 4: PDF Report Compilation...")
    pdf_path, compiler_status = compile_final_report(state, output_dir=os.path.dirname(os.path.abspath(__file__)))
    print(f"  PDF generated at: {pdf_path} (compiler_status={compiler_status})")
    print(f"  PDF exists? {os.path.exists(pdf_path)} (size: {os.path.getsize(pdf_path):,} bytes)")

    # 6. Run Evaluator
    print("\n[5/5] Running Evaluator...")
    eval_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation_results.json")
    results = run_evaluation(state, output_path=eval_path)
    print(f"  Evaluation saved to: {eval_path}")

    print("\n=== E2E VERIFICATION COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_full_pipeline()
