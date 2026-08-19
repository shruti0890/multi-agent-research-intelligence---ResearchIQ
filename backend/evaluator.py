import json
from rouge_score import rouge_scorer
from typing import List, Dict, Any
from schemas import ProjectReportState

def evaluate_summaries(state: ProjectReportState) -> Dict[str, Any]:
    """
    Computes ROUGE scores comparing the generated notebook_summary against the original paper abstract.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    r1_scores = []
    r2_scores = []
    rl_scores = []

    for paper in state.research.papers:
        ref = paper.abstract or ""
        cand = paper.notebook_summary or ""

        if not ref or not cand:
            continue

        scores = scorer.score(ref, cand)
        r1_scores.append(scores['rouge1'].fmeasure)
        r2_scores.append(scores['rouge2'].fmeasure)
        rl_scores.append(scores['rougeL'].fmeasure)

    avg_r1 = sum(r1_scores) / len(r1_scores) if r1_scores else 0.0
    avg_r2 = sum(r2_scores) / len(r2_scores) if r2_scores else 0.0
    avg_rl = sum(rl_scores) / len(rl_scores) if rl_scores else 0.0

    return {
        "rouge1_f1": avg_r1,
        "rouge2_f1": avg_r2,
        "rougeL_f1": avg_rl
    }


def evaluate_relevance_accuracy(state: ProjectReportState) -> Dict[str, Any]:
    """
    Evaluates the python-computed relevance scores.
    Since we compute relevance score deterministically based on embeddings and citations,
    we check the distribution of High, Medium, Low ranks.
    """
    ranks = [p.relevance_rank for p in state.research.papers]
    total = len(ranks)

    high_count = ranks.count("High")
    med_count = ranks.count("Medium")
    low_count = ranks.count("Low")

    return {
        "total_papers": total,
        "high_relevance_pct": (high_count / total * 100) if total else 0.0,
        "medium_relevance_pct": (med_count / total * 100) if total else 0.0,
        "low_relevance_pct": (low_count / total * 100) if total else 0.0
    }


def evaluate_compression_pipeline(state: ProjectReportState) -> List[Dict[str, Any]]:
    """
    Per-paper compression evaluation records.

    For each paper, returns a record with measured (not assumed) values
    from the TextRank compression pipeline.

    Fields:
        paper_id                  - unique paper identifier
        title                     - paper title
        source                    - API source (arXiv, PMC, etc.)
        full_text_available       - True if real full text was acquired
        full_text_source          - which source provided the full text
        coverage_type             - full_paper | partial_paper | abstract_only | unavailable
        full_text_word_count      - words in acquired full text (pre-compression)
        full_text_character_count - chars in acquired full text (pre-compression)
        original_words            - word count fed into compression
        compressed_words          - word count of fact sheet output
        original_tokens_estimated - estimated token count of original text
        compressed_tokens_estimated - estimated token count of fact sheet
        token_count_method        - always "estimated_word_split" unless changed
        compression_ratio         - measured: 1 - (compressed / original)
        sections_detected         - number of sections found by section parser
        sections_covered          - sections with at least 1 selected sentence
        section_coverage          - covered / detected
        selected_sentences        - total sentences in fact sheet
        technical_sentences       - sentences with technical content
        faithfulness_ratio        - fraction of sentences verified verbatim in source
    """
    records = []

    for paper in state.research.papers:
        paper_id = getattr(paper, 'paper_id', None) or ""
        if not paper_id:
            # Derive a fallback ID from title
            import re
            paper_id = "P-" + re.sub(r'[^a-z0-9]', '', paper.title.lower())[:8]

        # Source API
        source = getattr(paper, 'full_text_source', 'none') or 'none'
        if source == 'none':
            # Try to infer from URL
            url = paper.url or ""
            if "arxiv" in url:
                source = "arxiv"
            elif "pmc" in url.lower() or "europepmc" in url.lower():
                source = "pmc"
            elif "semanticscholar" in url.lower():
                source = "semantic_scholar"

        record = {
            "paper_id": paper_id,
            "title": paper.title,
            "source": source,
            "full_text_available": getattr(paper, 'full_text_available', False),
            "full_text_source": getattr(paper, 'full_text_source', 'none'),
            "coverage_type": getattr(paper, 'coverage_type', 'unavailable'),
            "full_text_word_count": getattr(paper, 'full_text_word_count', 0),
            "full_text_character_count": getattr(paper, 'full_text_character_count', 0),
            # Compression pipeline metrics (estimated)
            "original_words": 0,
            "compressed_words": 0,
            "original_tokens_estimated": getattr(paper, 'original_tokens', 0),
            "compressed_tokens_estimated": getattr(paper, 'compressed_tokens', 0),
            "token_count_method": "estimated_word_split",
            "compression_ratio": getattr(paper, 'compression_ratio', 0.0),
            "sections_detected": 0,
            "sections_covered": 0,
            "section_coverage": getattr(paper, 'section_coverage', 0.0),
            "selected_sentences": 0,
            "technical_sentences": 0,
            "faithfulness_ratio": 1.0,
        }

        # Enrich from PaperFactSheet if it was stored on the paper object
        fact_sheet = getattr(paper, 'fact_sheet', None)
        if fact_sheet is not None and hasattr(fact_sheet, 'metrics'):
            m = fact_sheet.metrics
            record["original_words"] = m.original_word_count
            record["compressed_words"] = m.compressed_word_count
            record["original_tokens_estimated"] = m.original_token_count
            record["compressed_tokens_estimated"] = m.compressed_token_count
            record["token_count_method"] = getattr(m, 'token_count_method', 'estimated_word_split')
            record["compression_ratio"] = round(m.compression_ratio, 4)
            record["compression_status"] = getattr(m, 'compression_status', 'evaluated')
            record["sections_detected"] = m.detected_sections
            record["sections_covered"] = m.covered_sections
            record["section_coverage"] = round(m.section_coverage, 4)  # legacy
            # Correct research-section-aware coverage
            record["research_sections"] = getattr(m, 'research_sections', 0)
            record["research_sections_covered"] = getattr(m, 'research_sections_covered', 0)
            record["research_section_coverage"] = round(getattr(m, 'research_section_coverage', 0.0), 4)
            record["non_research_sections_skipped"] = getattr(m, 'non_research_sections_skipped', 0)
            record["selected_sentences"] = m.selected_sentence_count
            record["technical_sentences"] = m.technical_sentence_count
            record["faithfulness_ratio"] = round(m.faithfulness_ratio, 4)
            record["coverage_type"] = getattr(fact_sheet, 'coverage_type', record["coverage_type"])

        # Relevance diagnostic fields (populated from Agent 1 scoring)
        record["access_status"] = getattr(paper, 'access_status', 'UNAVAILABLE')
        record["semantic_score"] = getattr(paper, 'semantic_score', 0.0)
        record["keyword_score"] = getattr(paper, 'keyword_score', 0.0)
        record["domain_penalty"] = getattr(paper, 'domain_penalty', 1.0)
        record["domain_mismatch"] = getattr(paper, 'domain_mismatch', False)
        record["final_relevance_score"] = getattr(paper, 'final_relevance_score', 0.0)

        records.append(record)

    return records


def run_evaluation(state: ProjectReportState, output_path: str = "evaluation_results.json") -> Dict[str, Any]:
    """
    Runs the entire evaluation layer and saves the output to a JSON file.
    Includes:
      - ROUGE summary scores
      - Relevance rank distribution
      - Per-paper compression pipeline metrics
    """
    summary_metrics = evaluate_summaries(state)
    relevance_metrics = evaluate_relevance_accuracy(state)
    compression_records = evaluate_compression_pipeline(state)

    # Pipeline-level status metadata
    eligibility_rate = getattr(state.research, 'full_paper_eligibility_rate', 1.0)
    pipeline_status = {
        "compiler_status": getattr(state, 'compiler_status', 'success'),
        "gemini_quota_exhausted": getattr(state, 'gemini_quota_exhausted', False),
        "patent_analysis_status": getattr(state.patents, 'patent_analysis_status', 'success'),
        "full_paper_eligibility_rate": eligibility_rate,
        "total_papers": len(state.research.papers),
        "total_gaps": len(state.gaps.gaps),
        "total_patents": len(state.patents.patents),
        "candidates_evaluated": len(getattr(state.research, 'candidate_diagnostics', [])),
    }

    results = {
        "pipeline_status": pipeline_status,
        "summary_evaluation": summary_metrics,
        "relevance_evaluation": relevance_metrics,
        "compression_pipeline": compression_records,
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*65)
    print("          QUANTITATIVE EVALUATION RESULTS")
    print("="*65)
    print(f"ROUGE-1 F1 Score : {summary_metrics['rouge1_f1']*100:.2f}%")
    print(f"ROUGE-2 F1 Score : {summary_metrics['rouge2_f1']*100:.2f}%")
    print(f"ROUGE-L F1 Score : {summary_metrics['rougeL_f1']*100:.2f}%")
    print("-"*65)
    print(f"Pipeline Status  : compiler={pipeline_status['compiler_status']} | "
          f"quota_exhausted={pipeline_status['gemini_quota_exhausted']} | "
          f"patents={pipeline_status['patent_analysis_status']}")
    print(f"Full-Paper Eligibility Rate : {eligibility_rate*100:.1f}% (Main Corpus)")
    print("-"*65)
    print(f"Total Papers Processed: {relevance_metrics['total_papers']}")
    print(f"  High Relevance  : {relevance_metrics['high_relevance_pct']:.1f}%")
    print(f"  Medium Relevance: {relevance_metrics['medium_relevance_pct']:.1f}%")
    print(f"  Low Relevance   : {relevance_metrics['low_relevance_pct']:.1f}%")
    print("-"*65)
    print(f"Per-Paper Compression Records: {len(compression_records)}")
    for rec in compression_records:
        ctype = rec.get("coverage_type", "unavailable")
        cstatus = rec.get("compression_status", "evaluated")
        rs_cov = rec.get("research_section_coverage", None)
        faithful = rec.get("faithfulness_ratio", 1.0) * 100
        if cstatus == "abstract_only_not_evaluated":
            ratio_str = "N/A (abstract only)"
        else:
            ratio_str = f"{rec.get('compression_ratio', 0.0) * 100:.1f}%"
        rs_cov_str = f"{rs_cov*100:.0f}%" if rs_cov is not None else "N/A"
        print(
            f"  [{rec['paper_id']}] {rec['title'][:36]:36s} | "
            f"{ctype:14s} | Compression: {ratio_str:18s} | "
            f"ResearchCoverage: {rs_cov_str:6s} | Faithfulness: {faithful:.1f}%"
        )
    print("="*65 + "\n")

    return results
