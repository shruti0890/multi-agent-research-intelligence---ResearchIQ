from pydantic import BaseModel, Field
from typing import List, Optional

# --- Agent 1: Research Paper Output Model ---
class PaperMetadata(BaseModel):
    title: str = Field(description="Title of the paper")
    authors: List[str] = Field(description="Authors of the paper")
    year: int = Field(description="Publication year")
    abstract: str = Field(description="Summary/abstract of the paper")
    key_findings: List[str] = Field(description="Top 3 key discoveries or methods used")
    datasets: List[str] = Field(description="Datasets utilized, if any")
    url: str = Field(description="Link to the original paper (arXiv, Semantic Scholar)")
    relevance_score: float = Field(description="Similarity score to user query (0.0 to 1.0)")

class Agent1ResearchOutput(BaseModel):
    query: str
    papers: List[PaperMetadata]

# --- Agent 2: Gap Analysis Output Model ---
class ResearchGap(BaseModel):
    description: str = Field(description="Detailed explanation of the unsolved problem or gap")
    severity: str = Field(description="Severity: Critical, Moderate, or Minor")
    evidence_papers: List[str] = Field(description="Paper titles that highlight or omit this gap")

class NovelMethodProposal(BaseModel):
    title: str = Field(description="Catchy name for the combined method suggestion")
    approach: str = Field(description="Step-by-step description of how to combine technologies to resolve the gaps")
    novelty_score: int = Field(description="Estimated originality index out of 100")
    rationale: str = Field(description="Explanation of why this approach is novel and safe from plagiarism")

class Agent2GapOutput(BaseModel):
    gaps: List[ResearchGap]
    proposed_method: NovelMethodProposal

# --- Agent 3: Patent Landscape Output Model ---
class PatentInfo(BaseModel):
    patent_id: str = Field(description="Patent number/ID")
    title: str = Field(description="Title of the patent")
    assignee: str = Field(description="Company or institution holding the patent")
    relevance: str = Field(description="Relevance classification: Prior Art, Overlap, or White Space")
    summary: str = Field(description="Short description of the patented technology")
    url: Optional[str] = Field(None, description="URL link to the patent page")

class Agent3PatentOutput(BaseModel):
    patents: List[PatentInfo]
    white_space_opportunities: List[str] = Field(description="List of areas in this domain that have no patents yet")

# --- Agent 4: Final State Compilation ---
class ProjectReportState(BaseModel):
    topic: str
    research: Agent1ResearchOutput
    gaps: Agent2GapOutput
    patents: Agent3PatentOutput
    pdf_filename: Optional[str] = None
