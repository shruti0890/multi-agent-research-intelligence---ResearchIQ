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
        # Compare notebook_summary (candidate) with abstract (reference)
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

def run_evaluation(state: ProjectReportState, output_path: str = "evaluation_results.json") -> Dict[str, Any]:
    """
    Runs the entire evaluation layer and saves the output to a JSON file.
    """
    summary_metrics = evaluate_summaries(state)
    relevance_metrics = evaluate_relevance_accuracy(state)
    
    results = {
        "summary_evaluation": summary_metrics,
        "relevance_evaluation": relevance_metrics
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*50)
    print("           QUANTITATIVE EVALUATION RESULTS")
    print("="*50)
    print(f"ROUGE-1 F1 Score: {summary_metrics['rouge1_f1']*100:.2f}%")
    print(f"ROUGE-2 F1 Score: {summary_metrics['rouge2_f1']*100:.2f}%")
    print(f"ROUGE-L F1 Score: {summary_metrics['rougeL_f1']*100:.2f}%")
    print("-"*50)
    print(f"Total Papers Processed: {relevance_metrics['total_papers']}")
    print(f"  High Relevance: {relevance_metrics['high_relevance_pct']:.1f}%")
    print(f"  Medium Relevance: {relevance_metrics['medium_relevance_pct']:.1f}%")
    print(f"  Low Relevance: {relevance_metrics['low_relevance_pct']:.1f}%")
    print("="*50 + "\n")
    
    return results
