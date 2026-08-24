# 🔬 ResearchIQ: Multi-Agent Academic & Technological Research Intelligence Platform

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)
![Gemini AI](https://img.shields.io/badge/LLM-Google%20Gemini%202.5--Flash-4285F4.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**ResearchIQ** is an autonomous multi-agent intelligence platform designed to revolutionize deep scientific literature review, patent analysis, gap discovery, and executive report synthesis. By coupling Google's **Gemini 2.5 Flash** models with deterministic text-compression pipelines and multi-source API retrievers, ResearchIQ produces publication-ready research reports and visual analytics in seconds.

---

## 🌟 Key Highlights

- 📚 **Multi-Source Academic Retrieval**: Performs topic decomposition and federated queries across **arXiv**, **PubMed**, **Europe PMC**, **OpenAlex**, and **bioRxiv**.
- 📜 **Global Patent Intelligence**: Queries **USPTO**, **Espacenet**, **Lens.org**, **WIPO**, and **Google Patents** for freedom-to-operate (FTO) assessment, innovation scoring, and Technology Readiness Levels (TRL).
- 🔍 **Strict Grounding & Quality Gatekeeping**: Implements deterministic full-text validation (`FULL_TEXT_MIN_WORDS >= 300`) and relevance gating to eliminate off-topic retrievals and hallucinated citations.
- ⚡ **Deterministic Fact-Sheet Compression**: Uses targeted section parsing (`section_parser.py`) to compress full-text papers into dense, structured fact sheets without vector database or RAG overhead.
- 💡 **Autonomous Research Gap Matrix**: Agent 2 identifies unaddressed technical bottlenecks, methodological gaps, and cross-disciplinary opportunities categorized by feasibility and potential impact.
- 📊 **Publication-Grade Visuals & PDF Generation**: Automatically renders high-resolution charts (`matplotlib`/`seaborn`) and compiles comprehensive executive PDFs.

---

## 🏗️ Multi-Agent System Architecture

ResearchIQ operates on a 4-Agent pipeline designed for strict data validation, schema enforcement, and non-hallucinatory output.

```mermaid
flowchart TD
    User([User Query / Research Topic]) --> Decomp[Topic Decomposition Engine]
    
    subgraph Agent1 ["Agent 1: Primary Literature Researcher"]
        Decomp --> Retr[Federated Literature Retrieval\narXiv | PubMed | EuropePMC | OpenAlex]
        Retr --> Gate[Relevance & Quality Gate\nWords >= 300 & Domain Match]
        Gate --> Comp[Section Parser & Fact Sheet Generator]
    end

    subgraph Agent2 ["Agent 2: Research Gap Analyst"]
        Comp --> GapAnalysis[Methodological & Technical Gap Matrix\nImpact vs. Feasibility Assessment]
    end

    subgraph Agent3 ["Agent 3: Patent & IP Intelligence"]
        Decomp --> PatentSearch[Patent Landscape Query\nUSPTO | Espacenet | Lens.org | WIPO]
        PatentSearch --> FTO[FTO & Technology Readiness Assessment]
    end

    subgraph Agent4 ["Agent 4: Executive Intelligence Compiler"]
        Comp --> Compiler[Synthesizer & Chart Renderer]
        GapAnalysis --> Compiler
        FTO --> Compiler
        Compiler --> PDF[PDF Report & Visual Analytics Output]
        Compiler --> UI[Interactive Frontend Dashboard]
    end
```

---

## 🧩 Agent Breakdown

| Agent | Module | Primary Responsibility | Key Outputs |
| :--- | :--- | :--- | :--- |
| **Agent 1** | [`agent_research.py`](file:///c:/Users/Hp/.gemini/antigravity/scratch/research-iq/backend/agents/agent_research.py) | Literature retrieval, relevance gating, and text compression. | Grounded paper summaries, fact sheets, diagnostics. |
| **Agent 2** | [`agent_gap.py`](file:///c:/Users/Hp/.gemini/antigravity/scratch/research-iq/backend/agents/agent_gap.py) | Literature gap discovery & innovation opportunity analysis. | Categorized Research Gap Matrix, feasibility scores. |
| **Agent 3** | [`agent_patent.py`](file:///c:/Users/Hp/.gemini/antigravity/scratch/research-iq/backend/agents/agent_patent.py) | Patent searching, IP overlap, and commercial readiness. | Patent landscape metrics, TRL, FTO analysis, direct registry links. |
| **Compiler** | [`agent_compile.py`](file:///c:/Users/Hp/.gemini/antigravity/scratch/research-iq/backend/agents/agent_compile.py) | Data consolidation, chart rendering, and PDF compilation. | Structured PDF report, interactive comparison charts. |

---

## 📁 Repository Structure

```text
research-iq/
├── backend/
│   ├── agents/
│   │   ├── agent_research.py      # Agent 1: Literature Retriever & Compressor
│   │   ├── agent_gap.py           # Agent 2: Research Gap Analyst
│   │   ├── agent_patent.py        # Agent 3: Patent & IP Intelligence
│   │   ├── agent_compile.py       # Agent 4: Executive Report Compiler
│   │   └── agent_utils.py         # Quota-aware Gemini API execution & retry engine
│   ├── compression/
│   │   └── section_parser.py      # Structured paper parser & fact sheet generator
│   ├── tests/                     # Comprehensive Pytest suite
│   ├── evaluator.py               # Automated pipeline evaluator
│   ├── main.py                    # FastAPI Web Server
│   ├── schemas.py                 # Pydantic data schemas
│   └── topic_decomposition.py     # Sub-query decomposition logic
├── frontend/
│   └── index.html                 # Interactive dashboard UI
├── run.py                         # Single-command pipeline executor
├── setup.ps1                      # Powershell environment bootstrapper
└── requirements.txt               # Backend dependencies
```

---

## ⚡ Quick Start Guide

### Prerequisites

- **Python 3.10+**
- **Google Gemini API Key** (Get one at [Google AI Studio](https://aistudio.google.com/))

### 1. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/shruti0890/multi-agent-research-intelligence---ResearchIQ.git
cd multi-agent-research-intelligence---ResearchIQ

# Create & activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Environment Configuration

Create a `.env` file inside the `backend/` directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

---

## 🚀 Running ResearchIQ

### Option A: Interactive Web Dashboard (FastAPI + UI)

Launch the backend server:

```bash
python backend/main.py
```

Then open `frontend/index.html` in your web browser or navigate to `http://localhost:8000`.

### Option B: Command Line Pipeline Execution

Run a complete multi-agent research analysis directly from the terminal:

```bash
python run.py --topic "Neural Networks for Financial Forecasting"
```

The pipeline will execute all 4 agents in sequence and output:
- `report_<topic_name>.pdf` — High-resolution executive PDF report.
- `chart_paper_comparison.png` — Literature metric comparison chart.
- `chart_gap_distribution.png` — Research gap feasibility vs impact distribution.
- `chart_patent_scores.png` — Patent innovation & TRL scores.

---

## 🧪 Testing & Evaluation

Execute the complete automated test suite:

```bash
pytest backend/tests/
```

To run the pipeline evaluation benchmark:

```bash
python backend/run_phase7_evaluation.py
```

---

## 🛡️ Fault Tolerance & API Quota Management

ResearchIQ includes built-in safeguards in [`agent_utils.py`](file:///c:/Users/Hp/.gemini/antigravity/scratch/research-iq/backend/agents/agent_utils.py):
- **Quota Exhaustion Detection**: Detects daily free-tier or API quota limits (`RESOURCE_EXHAUSTED`, `requests_per_day`) and halts gracefully without wasting retry budgets.
- **Transient Backoff**: Automatically handles transient network issues (`429`, `500`, `503`) with exponential backoff and dynamic delay parsing.
- **Keyword Signature Enforcement**: Fully updated for the official `google-genai` SDK (`genai.Client`).

---

## 📜 License

This project is licensed under the **MIT License**.
