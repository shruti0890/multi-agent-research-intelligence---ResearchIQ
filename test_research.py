import sys
import os
sys.path.insert(0, 'backend')
from agents.agent_research import fetch_arxiv_papers

res = fetch_arxiv_papers("Neural Networks for Financial Forecasting")
print(f"Query: {res.query}")
print(f"Found {len(res.papers)} papers:")
for p in res.papers:
    print(f"  - {p.title}")
