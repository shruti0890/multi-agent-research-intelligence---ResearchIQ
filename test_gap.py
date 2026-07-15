import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend.agents.agent_gap import cluster_and_analyze_gaps
from backend.schemas import Agent1ResearchOutput, PaperMetadata

mock_data = Agent1ResearchOutput(
    query="Enterprise Agentic AI",
    papers=[
        PaperMetadata(
            title="Vector Databases in AI Applications in Enterprise Agentic AI",
            authors=["Alice"],
            year=2024,
            abstract="",
            relevance_rank="High",
            notebook_summary="Summary",
            technical_execution="Tech",
            datasets=[],
            url="",
            relevance_score=0.9
        ),
        PaperMetadata(
            title="Introduction to Enterprise Agentic AI",
            authors=["Bob"],
            year=2024,
            abstract="No abstract available",
            relevance_rank="High",
            notebook_summary="Summary",
            technical_execution="Tech",
            datasets=[],
            url="",
            relevance_score=0.9
        )
    ]
)

try:
    res = cluster_and_analyze_gaps(mock_data)
    print("SUCCESS:")
    print(res.model_dump_json(indent=2))
except Exception as e:
    print("FAILED EXCEPTION:")
    print(repr(e))
