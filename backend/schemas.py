from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# --- Agent 1: Research Paper Output Model (NotebookLM Style) ---
class PaperMetadata(BaseModel):
    title: str = Field(description="Title of the paper")
    authors: List[str] = Field(description="Authors of the paper")
    year: int = Field(description="Publication year")
    abstract: str = Field(description="Summary/abstract of the paper")
    relevance_rank: str = Field(description="Relevance classification: High, Medium, or Low")
    notebook_summary: str = Field(description="NotebookLM-style plain-English card summary of what the paper does")
    technical_execution: str = Field(description="How the paper does it (core algorithmic flow)")
    datasets: List[str] = Field(description="Datasets utilized, if any")
    url: str = Field(description="Link to the original paper")
    relevance_score: float = Field(description="Numerical similarity score (0.0 to 1.0)")
    
    # Deep analysis fields for the PDF report
    problem_statement: str = Field(default="Not extracted", description="The core research problem this paper addresses")
    proposed_solution: str = Field(default="Not extracted", description="The solution or approach the paper proposes")
    methodology: str = Field(default="Not extracted", description="Methods, algorithms, and techniques used")
    results: str = Field(default="Not extracted", description="Key results, metrics, and performance numbers")
    challenges: str = Field(default="Not extracted", description="Limitations and challenges acknowledged by the authors")
    future_outcomes: str = Field(default="Not extracted", description="Future work directions mentioned in the paper")
    
    # Quality Enhancements
    innovation_score: int = Field(default=0, description="Innovation level score out of 100")
    research_significance: str = Field(default="Not analyzed", description="Detailed contribution and significance analysis")
    
    # Paper-specific Gap Analysis
    gap_description: str = Field(default="Not analyzed", description="Unsolved research limitation or gap specific to this paper")
    gap_impact: str = Field(default="Not analyzed", description="Blocker/impact of this gap on real-world use cases")
    gap_why_exists: str = Field(default="Not analyzed", description="Technical or data reasons why this gap exists")
    gap_opportunity: str = Field(default="Not analyzed", description="Actionable next step/research opportunity")
    gap_future_scope: str = Field(default="Not analyzed", description="Long-term vision/future scope")
    gap_severity: str = Field(default="Low", description="Severity classification: Critical, Moderate, or Low")


class Agent1ResearchOutput(BaseModel):
    query: str
    papers: List[PaperMetadata]

# --- Agent 2: Gap Analysis Output Model ---
class ResearchGap(BaseModel):
    description: str = Field(description="Clear explanation of the research gap or unsolved limitation")
    severity: str = Field(description="Severity: Critical, Moderate, or Minor")
    why_it_matters: str = Field(description="Explanation of why this gap prevents commercial progress")
    evidence_papers: List[str] = Field(description="Paper titles that fail to address this gap")

class NovelMethodProposal(BaseModel):
    title: str = Field(description="Academic name for the proposed combined method")
    approach: str = Field(description="Step-by-step technical implementation path")
    novelty_score: int = Field(description="Originality index out of 100")
    rationale: str = Field(description="Why this solution is unique and plagiarism-free")

class Agent2GapOutput(BaseModel):
    gaps: List[ResearchGap]
    proposed_method: NovelMethodProposal

# --- Agent 3: Patent Landscape Output Model ---
class PatentInfo(BaseModel):
    patent_id: str = Field(description="Patent registration number")
    title: str = Field(description="Title of the patent")
    assignee: str = Field(description="Company or university owning the patent")
    relevance: str = Field(description="Prior Art, Overlap, or White Space")
    summary: str = Field(description="Short description of the patented technology")
    fto_rating: str = Field(description="Freedom to Operate: Safe, Caution, or Alert")
    design_around_strategy: str = Field(description="Detailed suggestion on how to build your code differently to avoid infringing this patent")
    url: Optional[str] = Field(None, description="URL link to Google Patents")
    source_links: Optional[Dict[str, str]] = Field(None, description="URLs to various patent platforms")
    
    # Patent relevance and ranking enhancements
    relevance_score: int = Field(default=0, description="Patent relevance score (0-100)")
    rank: int = Field(default=1, description="Rank from highest to lowest relevance")
    match_explanation: str = Field(default="", description="Explanations for why each patent was matched")

class Agent3PatentOutput(BaseModel):
    patents: List[PatentInfo]
    white_space_opportunities: List[str] = Field(description="Unpatented opportunity segments")

# --- Agent 4: Final State Compilation ---
class ProjectReportState(BaseModel):
    topic: str
    research: Agent1ResearchOutput
    gaps: Agent2GapOutput
    patents: Agent3PatentOutput
    pdf_filename: Optional[str] = None
