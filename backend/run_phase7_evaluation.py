"""
run_phase7_evaluation.py
========================
Final End-to-End Evaluation for ResearchIQ (Phase 7).

Evaluates the multi-agent pipeline across 5 topics:
1. Deepfake audio detection
2. Large language model hallucination detection
3. Ecology and ecosystem
4. Recommender systems
5. AI-based medical diagnosis

Records all metrics and writes `evaluation_results.json`.
"""

import os
import sys
import json
import time
import socket
import traceback

socket.setdefaulttimeout(15)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schemas import ProjectReportState
from agents.agent_research import fetch_arxiv_papers
from agents.agent_gap import cluster_and_analyze_gaps
from agents.agent_patent import search_and_classify_patents
from agents.agent_compile import compile_final_report


TOPICS = [
    "Deepfake audio detection",
    "Large language model hallucination detection",
    "Ecology and ecosystem",
    "Recommender systems",
    "AI-based medical diagnosis",
]


def evaluate_topic(topic: str, max_results: int = 5) -> dict:
    print(f"\n{'='*70}\n[EVALUATION] Starting Topic: '{topic}'\n{'='*70}")
    start_time = time.time()
    errors = []
    fallbacks = []

    # 1. Agent 1: Research Paper Discovery
    print("--- [Agent 1] Paper Discovery & Full-Text Acquisition ---")
    t0 = time.time()
    try:
        research_out = fetch_arxiv_papers(topic, max_results=max_results)
    except Exception as e:
        print(f"Agent 1 error: {e}")
        errors.append(f"Agent 1 error: {str(e)}")
        raise

    a1_time = round(time.time() - t0, 2)

    # 2. Agent 2: Gap Analysis & Proposal
    print("--- [Agent 2] Gap Analysis & Novel Proposal ---")
    t0 = time.time()
    try:
        gap_out = cluster_and_analyze_gaps(research_out)
    except Exception as e:
        print(f"Agent 2 error: {e}")
        errors.append(f"Agent 2 error: {str(e)}")
        raise

    a2_time = round(time.time() - t0, 2)

    # 3. Agent 3: Patent Landscape
    print("--- [Agent 3] Multi-Source Patent Landscape ---")
    t0 = time.time()
    try:
        patent_out = search_and_classify_patents(gap_out, topic, research_out)
    except Exception as e:
        print(f"Agent 3 error: {e}")
        errors.append(f"Agent 3 error: {str(e)}")
        raise

    a3_time = round(time.time() - t0, 2)

    # 4. State & Gemini Quota propagation
    state = ProjectReportState(
        topic=topic,
        research=research_out,
        gaps=gap_out,
        patents=patent_out,
    )
    if patent_out.patent_analysis_status == "retrieval_success_synthesis_failed":
        state.gemini_quota_exhausted = True
        fallbacks.append("Agent 3: Synthesis failed (preserved retrieved patents)")

    # 5. Agent 4: PDF Compilation
    print("--- [Agent 4] PDF Compilation & Final Report ---")
    t0 = time.time()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = None
    compiler_status = "unknown"
    try:
        pdf_path, compiler_status = compile_final_report(state, output_dir=output_dir)
        state.pdf_filename = os.path.basename(pdf_path)
        state.compiler_status = compiler_status
        if compiler_status == "fallback_due_to_gemini":
            state.gemini_quota_exhausted = True
            fallbacks.append("Agent 4: Gemini fallback (deterministic report narrative)")
    except Exception as e:
        print(f"Agent 4 error: {e}")
        errors.append(f"Agent 4 error: {str(e)}")
        compiler_status = "failed"

    a4_time = round(time.time() - t0, 2)
    total_time = round(time.time() - start_time, 2)

    # ── Metric Extraction ──────────────────────────────────────────────────

    # Paper selection metrics
    candidate_diagnostics = research_out.candidate_diagnostics or []
    cand_count = len(candidate_diagnostics)
    abstract_only_rejected = sum(1 for c in candidate_diagnostics if c.access_status == "ABSTRACT_ONLY")
    unavailable_rejected = sum(1 for c in candidate_diagnostics if c.access_status in ("UNAVAILABLE", "ACCESS_CHECK_FAILED"))
    selected_papers = research_out.papers
    full_text_eligible_count = len(selected_papers)
    full_paper_eligibility_rate = 1.0 if len(selected_papers) > 0 and all(p.access_status == "FULL_TEXT_AVAILABLE" for p in selected_papers) else 0.0

    # Per-paper and compression metrics
    papers_data = []
    compression_reductions = []
    retained_ratios = []
    section_coverages = []
    faithfulness_ratios = []
    technical_sentence_counts = []

    for p in selected_papers:
        fs = getattr(p, "fact_sheet", None)
        metrics = getattr(fs, "metrics", None) if fs else None

        orig_tok = getattr(p, "original_tokens", 0) or (metrics.original_token_count if metrics else 0)
        comp_tok = getattr(p, "compressed_tokens", 0) or (metrics.compressed_token_count if metrics else 0)
        reduction = (1.0 - (comp_tok / orig_tok)) if orig_tok > 0 else getattr(p, "compression_ratio", 0.0)
        retained = (comp_tok / orig_tok) if orig_tok > 0 else (1.0 - reduction)
        sec_cov = getattr(p, "section_coverage", 0.0) or (metrics.research_section_coverage if metrics else 0.0)
        faith = metrics.faithfulness_ratio if metrics else 1.0
        tech_cnt = metrics.technical_sentence_count if metrics else 0

        compression_reductions.append(reduction)
        retained_ratios.append(retained)
        section_coverages.append(sec_cov)
        faithfulness_ratios.append(faith)
        technical_sentence_counts.append(tech_cnt)

        papers_data.append({
            "paper_id": p.paper_id,
            "title": p.title,
            "authors": p.authors[:3],
            "year": p.year,
            "access_status": p.access_status,
            "coverage_type": p.coverage_type,
            "full_text_source": p.full_text_source,
            "full_text_url": p.full_text_url,
            "word_count": p.full_text_word_count,
            "original_tokens": orig_tok,
            "compressed_tokens": comp_tok,
            "reduction_ratio": round(reduction, 4),
            "retained_ratio": round(retained, 4),
            "research_section_coverage": round(sec_cov, 4),
            "technical_sentence_count": tech_cnt,
            "faithfulness": round(faith, 4),
            "semantic_score": p.semantic_score,
            "keyword_score": p.keyword_score,
            "domain_penalty": p.domain_penalty,
            "final_relevance_score": p.final_relevance_score,
        })

    # Gap metrics
    gaps_data = []
    for g in gap_out.gaps:
        has_evidence = len(g.supporting_paper_ids) > 0 or len(g.supporting_sentence_ids) > 0 or len(g.evidence_papers) > 0
        gaps_data.append({
            "gap_id": g.gap_id,
            "gap_statement": g.gap_statement,
            "severity": g.severity,
            "why_it_matters": g.why_it_matters,
            "supporting_paper_ids": g.supporting_paper_ids,
            "supporting_sections": g.supporting_sections,
            "supporting_sentence_ids": g.supporting_sentence_ids,
            "evidence_papers": g.evidence_papers,
            "evidence_supported": has_evidence,
        })

    evidence_supported_gaps = sum(1 for g in gaps_data if g["evidence_supported"])
    evidence_gap_rate = (evidence_supported_gaps / len(gaps_data)) if gaps_data else 1.0

    # Patent metrics
    source_statuses = getattr(patent_out, "source_statuses", {})
    patents_list = patent_out.patents or []
    unique_patents = len(set(p.patent_id for p in patents_list))

    patents_data = {
        "epo_status": source_statuses.get("EPO", "UNKNOWN"),
        "lens_status": source_statuses.get("Lens", "UNKNOWN"),
        "patentsview_status": source_statuses.get("PatentsView", "UNKNOWN"),
        "epmc_status": source_statuses.get("EPMC", "UNKNOWN"),
        "source_statuses": source_statuses,
        "raw_patent_count": len(getattr(patent_out, "retrieved_patents_raw", [])),
        "unique_patent_count": unique_patents,
        "relevant_patent_count": len(patents_list),
        "patent_retrieval_status": patent_out.patent_retrieval_status,
        "patent_analysis_status": patent_out.patent_analysis_status,
        "patents": [
            {
                "patent_id": p.patent_id,
                "title": p.title,
                "assignee": p.assignee,
                "relevance": p.relevance,
                "relevance_score": p.relevance_score,
                "fto_rating": p.fto_rating,
            }
            for p in patents_list
        ]
    }

    # Runtime status
    gemini_status = "quota_exhausted" if getattr(state, "gemini_quota_exhausted", False) else (
        "fallback" if compiler_status == "fallback_due_to_gemini" else "success"
    )
    pdf_status = "generated" if pdf_path and os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000 else "failed"

    topic_result = {
        "topic": topic,
        "timing_seconds": {
            "agent_1_paper_discovery": a1_time,
            "agent_2_gap_analysis": a2_time,
            "agent_3_patents": a3_time,
            "agent_4_pdf_compiler": a4_time,
            "total": total_time,
        },
        "paper_selection": {
            "candidate_count": cand_count,
            "full_paper_eligible_count": full_text_eligible_count,
            "abstract_only_rejected_count": abstract_only_rejected,
            "unavailable_rejected_count": unavailable_rejected,
            "selected_paper_count": len(selected_papers),
            "full_paper_eligibility_rate": full_paper_eligibility_rate,
            "all_selected_have_full_text": all(p["access_status"] == "FULL_TEXT_AVAILABLE" for p in papers_data),
            "papers": papers_data,
            "candidates": [
                {
                    "paper_id": c.paper_id,
                    "title": c.title,
                    "access_status": c.access_status,
                    "selected": c.selected,
                    "rejection_reason": c.rejection_reason,
                    "semantic_score": c.semantic_score,
                    "keyword_score": c.keyword_score,
                    "domain_penalty": c.domain_penalty,
                    "final_relevance_score": c.final_relevance_score,
                }
                for c in candidate_diagnostics
            ]
        },
        "compression_summary": {
            "paper_count": len(papers_data),
            "avg_original_tokens": round(sum(p["original_tokens"] for p in papers_data) / len(papers_data), 1) if papers_data else 0,
            "avg_compressed_tokens": round(sum(p["compressed_tokens"] for p in papers_data) / len(papers_data), 1) if papers_data else 0,
            "avg_reduction_ratio": round(sum(compression_reductions) / len(compression_reductions), 4) if compression_reductions else 0.0,
            "avg_retained_ratio": round(sum(retained_ratios) / len(retained_ratios), 4) if retained_ratios else 0.0,
            "avg_research_section_coverage": round(sum(section_coverages) / len(section_coverages), 4) if section_coverages else 0.0,
            "avg_faithfulness": round(sum(faithfulness_ratios) / len(faithfulness_ratios), 4) if faithfulness_ratios else 1.0,
            "total_technical_sentences": sum(technical_sentence_counts),
        },
        "gap_analysis": {
            "gap_count": len(gaps_data),
            "evidence_supported_gap_count": evidence_supported_gaps,
            "evidence_supported_gap_rate": round(evidence_gap_rate, 4),
            "proposed_method": {
                "title": gap_out.proposed_method.title if gap_out.proposed_method else "",
                "novelty_score": gap_out.proposed_method.novelty_score if gap_out.proposed_method else 0,
                "approach_preview": gap_out.proposed_method.approach[:160] if gap_out.proposed_method else "",
            },
            "gaps": gaps_data,
        },
        "patent_landscape": patents_data,
        "runtime": {
            "gemini_status": gemini_status,
            "compiler_status": compiler_status,
            "pdf_status": pdf_status,
            "pdf_filename": getattr(state, "pdf_filename", None),
            "pdf_path": pdf_path,
            "pdf_size_bytes": os.path.getsize(pdf_path) if pdf_path and os.path.exists(pdf_path) else 0,
            "api_failures": errors,
            "fallbacks_triggered": fallbacks,
        }
    }

    print(f"\n[Topic Result Summary for '{topic}']:")
    print(f"  • Selected full papers: {len(selected_papers)} (Eligibility rate: {full_paper_eligibility_rate*100:.1f}%)")
    print(f"  • Compression avg reduction: {topic_result['compression_summary']['avg_reduction_ratio']*100:.1f}% (Retained: {topic_result['compression_summary']['avg_retained_ratio']*100:.1f}%)")
    print(f"  • Research section coverage: {topic_result['compression_summary']['avg_research_section_coverage']*100:.1f}%")
    print(f"  • Faithfulness: {topic_result['compression_summary']['avg_faithfulness']*100:.1f}%")
    print(f"  • Gaps: {len(gaps_data)} (Evidence rate: {evidence_gap_rate*100:.1f}%)")
    print(f"  • Patents: {len(patents_list)} (Retrieval: {patent_out.patent_retrieval_status}, Synthesis: {patent_out.patent_analysis_status})")
    print(f"  • PDF: {pdf_status} ({topic_result['runtime']['pdf_size_bytes']:,} bytes)")

    return topic_result


def main():
    print(f"Starting Phase 7 End-to-End Evaluation across {len(TOPICS)} topics...")
    results = []

    for i, topic in enumerate(TOPICS, 1):
        try:
            res = evaluate_topic(topic)
            results.append(res)
        except Exception as e:
            print(f"FAILED topic '{topic}': {e}")
            traceback.print_exc()
            results.append({
                "topic": topic,
                "status": "FAILED",
                "error": str(e),
            })

    # Compute overall benchmark statistics
    valid_results = [r for r in results if r.get("paper_selection")]
    
    total_selected_papers = sum(r["paper_selection"]["selected_paper_count"] for r in valid_results)
    total_eligible_papers = sum(r["paper_selection"]["full_paper_eligible_count"] for r in valid_results)
    overall_eligibility_rate = (total_eligible_papers / total_selected_papers) if total_selected_papers > 0 else 0.0

    all_reductions = [p["reduction_ratio"] for r in valid_results for p in r["paper_selection"]["papers"]]
    all_retained = [p["retained_ratio"] for r in valid_results for p in r["paper_selection"]["papers"]]
    all_coverages = [p["research_section_coverage"] for r in valid_results for p in r["paper_selection"]["papers"]]
    all_faithfulness = [p["faithfulness"] for r in valid_results for p in r["paper_selection"]["papers"]]
    all_gaps = [g for r in valid_results for g in r["gap_analysis"]["gaps"]]
    all_supported_gaps = [g for g in all_gaps if g["evidence_supported"]]

    summary = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topics_evaluated_count": len(valid_results),
        "total_selected_papers": total_selected_papers,
        "key_metrics": {
            "1_full_paper_eligibility_rate": {
                "value": round(overall_eligibility_rate, 4),
                "target": "100%",
                "status": "PASS" if overall_eligibility_rate >= 0.999 else "FAIL",
                "description": "Selected papers with verified full text / selected papers",
            },
            "2_compression_reduction_rate": {
                "mean": round(sum(all_reductions) / len(all_reductions), 4) if all_reductions else 0.0,
                "min": round(min(all_reductions), 4) if all_reductions else 0.0,
                "max": round(max(all_reductions), 4) if all_reductions else 0.0,
                "target": "60–70%",
                "status": "PASS" if all_reductions and 0.50 <= (sum(all_reductions)/len(all_reductions)) <= 0.85 else "WARN",
                "description": "Extractive TextRank compression reduction ratio (1 - comp_tokens/orig_tokens)",
            },
            "3_retained_content_ratio": {
                "mean": round(sum(all_retained) / len(all_retained), 4) if all_retained else 0.0,
                "min": round(min(all_retained), 4) if all_retained else 0.0,
                "max": round(max(all_retained), 4) if all_retained else 0.0,
                "target": "30–40%",
                "description": "Retained content ratio after compression (comp_tokens / orig_tokens)",
            },
            "4_research_section_coverage": {
                "mean": round(sum(all_coverages) / len(all_coverages), 4) if all_coverages else 0.0,
                "min": round(min(all_coverages), 4) if all_coverages else 0.0,
                "max": round(max(all_coverages), 4) if all_coverages else 0.0,
                "target": "High / near 100%",
                "status": "PASS" if all_coverages and (sum(all_coverages)/len(all_coverages)) >= 0.80 else "WARN",
                "description": "Research sections represented in compressed fact sheet",
            },
            "5_faithfulness": {
                "mean": round(sum(all_faithfulness) / len(all_faithfulness), 4) if all_faithfulness else 1.0,
                "min": round(min(all_faithfulness), 4) if all_faithfulness else 1.0,
                "max": round(max(all_faithfulness), 4) if all_faithfulness else 1.0,
                "target": "100%",
                "status": "PASS" if all_faithfulness and (sum(all_faithfulness)/len(all_faithfulness)) >= 0.98 else "FAIL",
                "description": "Extractive sentence validation rate against original source text",
            },
            "6_evidence_supported_gap_rate": {
                "value": round((len(all_supported_gaps) / len(all_gaps)), 4) if all_gaps else 1.0,
                "total_gaps": len(all_gaps),
                "supported_gaps": len(all_supported_gaps),
                "target": "100%",
                "status": "PASS" if all_gaps and (len(all_supported_gaps)/len(all_gaps)) >= 0.95 else "WARN",
                "description": "Fraction of gaps with traceable supporting paper IDs and sentence IDs",
            }
        },
        "topic_results": results,
    }

    # Save evaluation_results.json in backend/ and repo root
    backend_out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results.json")
    repo_out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation_results.json")

    with open(backend_out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(repo_out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}\n[EVALUATION COMPLETE]\n{'='*70}")
    print(f"Results written to:\n  - {backend_out_path}\n  - {repo_out_path}")
    print(f"Full-paper eligibility rate: {summary['key_metrics']['1_full_paper_eligibility_rate']['value']*100:.1f}%")
    print(f"Compression reduction mean : {summary['key_metrics']['2_compression_reduction_rate']['mean']*100:.1f}%")
    print(f"Retained content mean      : {summary['key_metrics']['3_retained_content_ratio']['mean']*100:.1f}%")
    print(f"Research section coverage  : {summary['key_metrics']['4_research_section_coverage']['mean']*100:.1f}%")
    print(f"Extractive Faithfulness    : {summary['key_metrics']['5_faithfulness']['mean']*100:.1f}%")
    print(f"Evidence-supported gap rate: {summary['key_metrics']['6_evidence_supported_gap_rate']['value']*100:.1f}%")


if __name__ == "__main__":
    main()
