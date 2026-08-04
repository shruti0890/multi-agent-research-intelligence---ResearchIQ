import sys
import os
import json
sys.path.insert(0, 'backend')

from agents.agent_research import fetch_arxiv_papers
from agents.agent_gap import cluster_and_analyze_gaps
from agents.agent_patent import search_and_classify_patents
from agents.agent_compile import compile_final_report
from schemas import ProjectReportState
from evaluator import run_evaluation

topic = "Neural Networks for Financial Forecasting"

print("Step 1: Running Agent 1 (Research Paper Relevance Scoring)...")
research_out = fetch_arxiv_papers(topic, max_results=4)
print(f"Found {len(research_out.papers)} papers.")
print("Relevance scores and ranks:")
for idx, p in enumerate(research_out.papers, 1):
    print(f"  [{idx}] Title: {p.title[:50]}...")
    print(f"      Relevance Score: {p.relevance_score * 100:.1f}%, Rank: {p.relevance_rank}, Innovation: {p.innovation_score}/100")
    print(f"      Significance: {p.research_significance[:100]}...")

print("\nStep 2: Running Agent 2 (Gap Analysis)...")
gap_out = cluster_and_analyze_gaps(research_out)
print("Updated paper gap severities and sorting:")
for idx, p in enumerate(research_out.papers, 1):
    print(f"  [{idx}] Title: {p.title[:50]}...")
    print(f"      Gap Severity: {p.gap_severity}")
    print(f"      Gap Description: {p.gap_description[:80]}...")
    print(f"      Gap Impact: {p.gap_impact[:80]}...")

print("\nStep 3: Running Agent 3 (Patent Discovery)...")
patent_out = search_and_classify_patents(gap_out, topic, research_out)
print(f"Found {len(patent_out.patents)} patents.")
print("Patent ranks and relevance scores:")
for pat in patent_out.patents:
    print(f"  [#{pat.rank}] ID: {pat.patent_id} (Score: {pat.relevance_score}/100, FTO: {pat.fto_rating})")
    print(f"      Title: {pat.title}")
    print(f"      Explanation: {pat.match_explanation[:100]}...")

print("\nStep 4: Running Agent 4 (Compile McKinsey Style PDF Report)...")
state = ProjectReportState(
    topic=topic,
    research=research_out,
    gaps=gap_out,
    patents=patent_out
)
pdf_path = compile_final_report(state, output_dir=".")
print(f"PDF compiled successfully at: {pdf_path}")

print("\nStep 5: Running Quantitative Evaluation Layer...")
run_evaluation(state)
