import sys
sys.path.append("backend")
from schemas import Agent2GapOutput, NovelMethodProposal
from agents.agent_patent import search_and_classify_patents

gap_data = Agent2GapOutput(
    gaps=[],
    proposed_method=NovelMethodProposal(title="test", approach="test", novelty_score=10, rationale="test")
)
try:
    res = search_and_classify_patents(gap_data, "machine learning")
    print("SUCCESS", res)
except Exception as e:
    import traceback
    traceback.print_exc()
