import sys
import os
sys.path.insert(0, 'backend')
from agents.agent_patent import _filter_candidates_by_topic

# Candidate patents from previous EPMC/Crossref fetch
patents_list = [
    {
        "patent_id": "WO2009034499",
        "title": "Sensitivity Enhanced Arbitrary Polynomial Chaos",
        "abstract": "Sensitivity Enhanced Arbitrary Polynomial Chaos applied to medical image analysis."
    },
    {
        "patent_id": "WO03043490",
        "title": "Learning Active Multimodal Subspaces in the Brain",
        "abstract": "Method for mapping active multimodal subspaces in the human brain for neuroscience applications."
    },
    {
        "patent_id": "US77671010B2",
        "title": "Global energy forecasting competition 2017: Hierarchical probabilistic load forecasting",
        "abstract": "Energy load forecasting methods utilizing hierarchical probabilistic modeling for utility grid optimization."
    },
    {
        "patent_id": "US46070293B2",
        "title": "Forecasting Probability Distributions of Financial Returns with Deep Neural Networks",
        "abstract": "A machine learning system that uses deep neural networks to forecast probability distributions of stock and asset returns in financial markets."
    },
    {
        "patent_id": "US44247366B2",
        "title": "Modeling company's financial sustainability with artificial neural networks",
        "abstract": "Neural network architectures trained on balance sheet data to forecast financial sustainability and bankruptcy risk of corporations."
    }
]

filtered = _filter_candidates_by_topic(patents_list, "Neural Networks for Financial Forecasting")
print(f"Filtered down to {len(filtered)} patents:")
for p in filtered:
    print(f"  - {p['patent_id']}: {p['title']}")
