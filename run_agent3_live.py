#!/usr/bin/env python
"""
run_agent3_live.py — Standalone Agent 3 live patent search & strict relevance diagnostic tool.

Usage:
    python run_agent3_live.py "Agentic AI in multiagent systems"
    python run_agent3_live.py "multi-agent artificial intelligence"
    python run_agent3_live.py "Deepfake audio detection"
"""
import os
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Add backend to path
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
sys.path.insert(0, BACKEND_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BACKEND_DIR, ".env"), override=True)

from agents.agent_patent import (
    expand_patent_queries,
    retrieve_multi_source_patents,
    evaluate_hard_relevance_gates,
    decompose_topic_concepts,
)


def check_credentials() -> dict:
    """Check which optional credentials are configured."""
    return {
        "EPO_OPS_CONSUMER_KEY": bool(os.getenv("EPO_OPS_CONSUMER_KEY", "").strip()),
        "EPO_OPS_CONSUMER_SECRET": bool(os.getenv("EPO_OPS_CONSUMER_SECRET", "").strip()),
        "LENS_API_TOKEN": bool(os.getenv("LENS_API_TOKEN", "").strip()),
        "PATENTSVIEW_API_KEY": bool(os.getenv("PATENTSVIEW_API_KEY", "").strip()),
        "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY", "").strip()),
    }


def run_live_test(topic: str, max_results: int = 5):
    print("=" * 60)
    print("AGENT 3 LIVE PATENT TEST (STRICT RELEVANCE & HARD GATES)")
    print("=" * 60)
    print()
    print(f"Topic: {topic}")
    print()

    # Credentials check
    creds = check_credentials()
    print("Credentials:")
    for name, present in creds.items():
        status = "YES" if present else "NO (MISSING)"
        print(f"  {name}: {status}")

    if not creds["EPO_OPS_CONSUMER_KEY"] or not creds["EPO_OPS_CONSUMER_SECRET"]:
        print("  [WARN] EPO_OPS credentials missing (EPO will be skipped).")
    if not creds["LENS_API_TOKEN"]:
        print("  [WARN] LENS_API_TOKEN missing (Lens will be skipped).")

    print()

    # Query expansion
    queries = expand_patent_queries(topic)
    print(f"Generated Queries ({len(queries)}):")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    print()

    # Live patent retrieval
    relevant_patents, source_statuses, overall_status = retrieve_multi_source_patents(
        topic, max_results=max_results
    )

    direct_count = sum(1 for p in relevant_patents if p.get("relevance_level") == "DIRECT_MATCH")
    strong_count = sum(1 for p in relevant_patents if p.get("relevance_level") == "STRONG_RELATED")

    print("-" * 60)
    print(f"Summary for: '{topic}'")
    print(f"  Source Statuses: {source_statuses}")
    print(f"  Direct Matches:  {direct_count}")
    print(f"  Strong Related:  {strong_count}")
    print(f"  Final Patents:   {len(relevant_patents)}")
    print(f"  Overall Status:  {overall_status}")
    print("-" * 60)
    print()

    if relevant_patents:
        print(f"Top {len(relevant_patents)} Genuinely Relevant Patent(s):")
        print()
        for i, pat in enumerate(relevant_patents, 1):
            pid = pat.get("patent_id") or pat.get("publication_number") or "N/A"
            title = pat.get("title") or "(no title)"
            sources_str = ", ".join(pat.get("sources", [pat.get("source", "Unknown")]))
            score = pat.get("relevance_score", "?")
            level = pat.get("relevance_level", "HIGH_RELEVANCE")
            why_rel = pat.get("why_relevant") or ""
            assignee = pat.get("assignee") or pat.get("applicant") or ""
            pub_date = pat.get("publication_date") or ""

            # Extract key concepts matched
            text_corpus = f"{title} {pat.get('abstract', '')} {pat.get('claims', '')}".lower()
            decomp = decompose_topic_concepts(topic)
            matched_concepts = []
            for k, anchors in decomp.items():
                if isinstance(anchors, list) and k not in ("telecom_routing_mismatch", "visual_only_mismatch", "industrial_mismatch"):
                    for a in anchors:
                        if a.lower() in text_corpus and a.lower() not in [m.lower() for m in matched_concepts]:
                            matched_concepts.append(a)

            print(f"  {i}. Patent ID:       {pid}")
            print(f"     Title:           {title}")
            print(f"     Source:          {sources_str}")
            print(f"     Score:           {score}/100")
            print(f"     Relevance Level: {level}")
            print(f"     Why it matches:  {why_rel}")
            if matched_concepts:
                print(f"     Key Concepts:    {', '.join(matched_concepts[:6])}")
            if assignee:
                print(f"     Assignee:        {assignee}")
            if pub_date:
                print(f"     Date:            {pub_date}")
            print()
    else:
        print("No genuinely relevant patents passed the hard gates.")
        print("  (Generic or off-topic patents were strictly rejected).")
        print()

    print("=" * 60)
    print()
    return relevant_patents, source_statuses, overall_status


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_agent3_live.py \"<research topic>\"")
        sys.exit(1)

    topic = sys.argv[1].strip()
    max_results = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run_live_test(topic, max_results=max_results)


if __name__ == "__main__":
    main()
