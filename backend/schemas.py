from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class GeminiError(Exception):
    """Base exception for Gemini API errors."""
    pass

class GeminiModelUnavailableError(GeminiError):
    """Raised when Gemini model is deprecated, not found, or unavailable (404)."""
    pass

class GeminiAuthenticationError(GeminiError):
    """Raised when API key is missing, invalid, or permission is denied (401/403)."""
    pass

class GeminiQuotaExceededError(GeminiError):
    """Raised when Gemini daily quota or request budget is exhausted (429/RESOURCE_EXHAUSTED)."""
    pass

# Backward compatibility alias
GeminiQuotaExhaustedError = GeminiQuotaExceededError

class GeminiServiceError(GeminiError):
    """Raised when Gemini server returns 5xx or connection failures."""
    pass

class GeminiTimeoutError(GeminiError):
    """Raised when Gemini request times out."""
    pass


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
    # Full text content for deep extraction (kept for section parsing and validation)
    full_text: str = Field(default="", description="Clean full text fetched from PMC/ArXiv (no character truncation)")

    paper_id: str = Field(default="", description="Unique paper identifier assigned by Agent 1 (e.g. P001)")
    # Compression pipeline outputs — populated after TextRank fact sheet generation
    fact_sheet: Any = Field(default=None, description="PaperFactSheet instance produced by extractive compression")
    fact_sheet_text: str = Field(default="", description="Formatted fact sheet produced by the extractive compression pipeline")
    compression_ratio: float = Field(default=0.0, description="Measured compression ratio: 1 - (compressed_tokens / original_tokens)")
    section_coverage: float = Field(default=0.0, description="Fraction of detected sections represented in the fact sheet")
    original_tokens: int = Field(default=0, description="Estimated token count of the original full paper text")
    compressed_tokens: int = Field(default=0, description="Estimated token count of the compressed fact sheet")

    # Full-text acquisition status — populated by the full-text resolver
    full_text_available: bool = Field(default=False, description="True if real full text (not just abstract) was successfully acquired")
    full_text_source: str = Field(default="none", description="Source used for full text: 'arxiv', 'pmc', 'semantic_scholar', 'openalex', 'doaj', 'unpaywall', 'none'")
    full_text_url: str = Field(default="", description="URL from which full text was fetched")
    compression_source: str = Field(default="none", description="What was fed to the compression pipeline: 'full_text', 'abstract', 'none'")
    coverage_type: str = Field(default="unavailable", description="Coverage level: 'full_paper' | 'partial_paper' | 'abstract_only' | 'unavailable'")
    full_text_word_count: int = Field(default=0, description="Word count of the acquired full text before compression")
    full_text_character_count: int = Field(default=0, description="Character count of the acquired full text before compression")

    # Access status — standardized access state
    access_status: str = Field(
        default="UNAVAILABLE",
        description="Standardized access state: 'FULL_TEXT_AVAILABLE' | 'ABSTRACT_ONLY' | 'UNAVAILABLE' | 'ACCESS_CHECK_FAILED'"
    )

    # Relevance diagnostic fields — populated by Agent 1 scoring pipeline
    semantic_score: float = Field(default=0.0, description="Raw semantic similarity score from sentence-transformer (0.0–1.0)")
    keyword_score: float = Field(default=0.0, description="Keyword overlap score (0.0–1.0)")
    domain_penalty: float = Field(default=1.0, description="Domain mismatch multiplier (1.0 = no penalty, <1.0 = penalized)")
    domain_mismatch: bool = Field(default=False, description="True if domain mismatch penalty was applied to this paper")
    final_relevance_score: float = Field(default=0.0, description="Final composite relevance score after all penalties")
    domain_match: bool = Field(default=True, description="True if paper passed the domain gate")
    agent_match: bool = Field(default=True, description="True if paper passed the agent concept gate")
    multiagent_match: bool = Field(default=True, description="True if paper passed the multi-agent concept gate")
    core_concept_match: bool = Field(default=True, description="True if paper passed the core concept gate")
    technical_concept_match: bool = Field(default=True, description="True if paper passed the technical mechanism gate")
    relevance_status: str = Field(default="ACCEPTED", description="'ACCEPTED' | 'REJECTED_OFF_TOPIC' | 'REJECTED'")
    relevance_level: str = Field(default="DIRECT_MATCH", description="'DIRECT_MATCH' | 'STRONG_RELATED' | 'ADJACENT' | 'REJECTED'")
    why_selected: str = Field(default="", description="Explanation of why this paper was selected for the topic")


class CandidateDiagnostic(BaseModel):
    """Discovery diagnostic record for every paper candidate evaluated."""
    paper_id: str = Field(description="Identifier (e.g. P001 or cand-001)")
    title: str = Field(description="Title of the paper candidate")
    semantic_score: float = Field(default=0.0, description="Semantic similarity score (0.0 to 1.0)")
    keyword_score: float = Field(default=0.0, description="Keyword overlap score (0.0 to 1.0)")
    domain_penalty: float = Field(default=1.0, description="Domain mismatch multiplier")
    domain_mismatch: bool = Field(default=False, description="Whether domain mismatch was detected")
    final_relevance_score: float = Field(default=0.0, description="Final composite relevance score")
    domain_match: bool = Field(default=True, description="True if candidate passed domain gate")
    agent_match: bool = Field(default=True, description="True if candidate passed agent gate")
    multiagent_match: bool = Field(default=True, description="True if candidate passed multiagent gate")
    core_concept_match: bool = Field(default=True, description="True if candidate passed core concept gate")
    technical_concept_match: bool = Field(default=True, description="True if candidate passed technical gate")
    relevance_status: str = Field(default="ACCEPTED", description="'ACCEPTED' | 'REJECTED_OFF_TOPIC' | 'REJECTED'")
    relevance_level: str = Field(default="DIRECT_MATCH", description="'DIRECT_MATCH' | 'STRONG_RELATED' | 'ADJACENT' | 'REJECTED'")
    access_status: str = Field(
        default="UNAVAILABLE",
        description="'FULL_TEXT_AVAILABLE' | 'ABSTRACT_ONLY' | 'UNAVAILABLE' | 'ACCESS_CHECK_FAILED'"
    )
    full_text_source: str = Field(default="none", description="Source of full text if resolved")
    selected: bool = Field(default=False, description="True if accepted into main research corpus")
    rejection_reason: Optional[str] = Field(default=None, description="Reason if rejected from main corpus")


class Agent1ResearchOutput(BaseModel):
    query: str
    papers: List[PaperMetadata] = Field(description="Main research corpus (strictly FULL_TEXT_AVAILABLE papers)")
    candidate_diagnostics: List[CandidateDiagnostic] = Field(
        default_factory=list,
        description="Diagnostics for all candidates evaluated during discovery"
    )
    full_paper_eligibility_rate: float = Field(
        default=1.0,
        description="Fraction of main corpus papers with verified full text (1.0 for main corpus)"
    )

# --- Agent 2: Gap Analysis Output Model ---
class ResearchGap(BaseModel):
    gap_id: str = Field(default="", description="Unique gap identifier (e.g. GAP-01, GAP-02)")
    gap_statement: str = Field(default="", description="Core gap statement grounded in empirical evidence")
    description: str = Field(default="", description="Detailed explanation of the research gap or unsolved limitation")
    severity: str = Field(default="Moderate", description="Severity: Critical, Moderate, or Minor")
    why_it_matters: str = Field(default="", description="Explanation of why this gap prevents commercial progress")
    supporting_paper_ids: List[str] = Field(default_factory=list, description="IDs of supporting papers (e.g. ['P001', 'P002'])")
    supporting_sections: List[str] = Field(default_factory=list, description="Sections containing supporting evidence (e.g. ['methodology', 'limitations'])")
    supporting_sentence_ids: List[str] = Field(default_factory=list, description="Verbatim supporting sentence IDs (e.g. ['P001-METH-02', 'P002-LIM-01'])")
    evidence_papers: List[str] = Field(default_factory=list, description="Paper titles that fail to address or exhibit this gap")
    evidence_references: Optional[List[str]] = Field(
        default=None,
        description="Traceability references in format: 'paper_id | section | sentence_id | verbatim sentence'"
    )

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

    # Common multi-source metadata fields
    publication_number: str = Field(default="", description="Standard publication number")
    abstract: str = Field(default="", description="Patent abstract text")
    claims: str = Field(default="", description="Patent claims text or primary claim summary")
    description: str = Field(default="", description="Patent description text or technical details")
    publication_date: str = Field(default="", description="Date of patent publication")
    priority_date: str = Field(default="", description="Priority date of patent filing")
    jurisdiction: str = Field(default="", description="Jurisdiction code (e.g. US, EP, WO)")
    applicant: str = Field(default="", description="Applicant / assignee entity")
    inventors: List[str] = Field(default_factory=list, description="List of inventor names")
    source: str = Field(default="", description="Primary source of retrieval")
    sources: List[str] = Field(default_factory=list, description="All sources providing this patent (merged)")
    source_url: str = Field(default="", description="Direct URL to source patent record")
    retrieval_status: str = Field(default="SUCCESS", description="Record retrieval status")

    # Strict hard gate & semantic relevance diagnostic fields
    relevance_level: str = Field(default="HIGH_RELEVANCE", description="'HIGH_RELEVANCE' | 'MEDIUM_RELEVANCE' | 'REJECTED'")
    domain_match: bool = Field(default=True, description="True if patent passed the domain relevance gate")
    technical_match: bool = Field(default=True, description="True if patent passed the technical concept gate")
    why_relevant: str = Field(default="", description="Concise explanation of conceptual and technical relevance")
    rejection_reason: Optional[str] = Field(default=None, description="Detailed reason if patent failed gates or threshold")
    relevance_status: str = Field(default="accepted", description="'accepted' | 'rejected'")

class Agent3PatentOutput(BaseModel):
    patents: List[PatentInfo]
    white_space_opportunities: List[str] = Field(description="Unpatented opportunity segments")
    # Status of the patent analysis pipeline
    patent_analysis_status: str = Field(
        default="success",
        description="'success' | 'retrieval_success_synthesis_failed' | 'retrieval_failed'"
    )
    # Overall multi-source retrieval status
    patent_retrieval_status: str = Field(
        default="SUCCESS",
        description="'SUCCESS' | 'PARTIAL_SUCCESS' | 'NO_RESULTS' | 'RETRIEVAL_FAILED'"
    )
    # Granular status per patent source adapter
    source_statuses: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-source statuses: 'SUCCESS' | 'NO_RESULTS' | 'SOURCE_UNAVAILABLE' | 'AUTHENTICATION_FAILED' | 'SKIPPED_NO_CREDENTIALS'"
    )
    # Raw retrieved patent data preserved even if Gemini synthesis fails
    retrieved_patents_raw: List[dict] = Field(
        default_factory=list,
        description="Raw patent metadata dict list from fetchers, preserved across Gemini failures"
    )

# --- Agent 4: Final State Compilation ---
class ProjectReportState(BaseModel):
    topic: str
    research: Agent1ResearchOutput
    gaps: Agent2GapOutput
    patents: Agent3PatentOutput
    pdf_filename: Optional[str] = None
    # Pipeline status tracking
    compiler_status: str = Field(
        default="success",
        description="'success' | 'fallback_due_to_gemini' — set by Agent 4 if Gemini synthesis failed"
    )
    gemini_quota_exhausted: bool = Field(
        default=False,
        description="True if any agent encountered Gemini daily quota exhaustion during this run"
    )
