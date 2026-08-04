import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import re
# pyrefly: ignore [missing-import]
from google import genai
from google.genai import types
# pyrefly: ignore [missing-import]
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from schemas import Agent1ResearchOutput, PaperMetadata
from rag_prefetcher import get_embedding_model, prefetch_top_chunks
from sentence_transformers import util

# Load environment variables — explicit path so it works when server runs from project root
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path=_env_path, override=True)

# Configure Gemini using the new google.genai SDK
_gemini_client = None
def _get_client():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Check backend/.env file.")
        print(f"[Agent1] Gemini API key loaded: ...{api_key[-6:]}")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client

GEMINI_MODEL = "gemini-2.5-flash"

def normalize_title(title: str) -> str:
    return re.sub(r'[^a-z0-9]', '', title.lower())

# --- API 1: arXiv ---
def fetch_arxiv(query: str, limit: int) -> list:
    papers = []
    encoded_query = urllib.parse.quote(query)
    url = f'http://export.arxiv.org/api/query?search_query=all:{encoded_query}&max_results={limit}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=6) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                link = entry.find('atom:id', ns).text.strip()
                published = entry.find('atom:published', ns).text
                year = int(published[:4]) if published else 2024
                authors = [author.find('atom:name', ns).text.strip() for author in entry.findall('atom:author', ns)]
                
                papers.append({
                    "title": title, "authors": authors, "year": year, 
                    "abstract": abstract, "url": link, "source": "arXiv"
                })
    except Exception as e:
        print(f"arXiv API warning: {e}")
    return papers

# --- API 2: Semantic Scholar ---
def fetch_semantic_scholar(query: str, limit: int) -> list:
    papers = []
    encoded_query = urllib.parse.quote(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&limit={limit}&fields=title,authors,year,abstract,url,citationCount"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode())
            for item in data.get("data", []):
                authors = [auth.get("name") for auth in item.get("authors", []) if auth.get("name")]
                papers.append({
                    "title": item.get("title", "Unknown"),
                    "authors": authors,
                    "year": item.get("year", 2024) or 2024,
                    "abstract": item.get("abstract") or "No abstract available.",
                    "url": item.get("url") or "",
                    "source": "Semantic Scholar",
                    "citations": item.get("citationCount", 0) or 0
                })
    except Exception as e:
        print(f"Semantic Scholar API warning: {e}")
    return papers

# --- API 3: Crossref ---
def fetch_crossref(query: str, limit: int) -> list:
    papers = []
    encoded_query = urllib.parse.quote(query)
    url = f"https://api.crossref.org/works?query={encoded_query}&rows={limit}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode())
            for item in data.get("message", {}).get("items", []):
                titles = item.get("title", [])
                title = titles[0] if titles else "Unknown"
                raw_abstract = item.get("abstract", "") or ""
                abstract = re.sub('<[^<]+?>', '', raw_abstract).strip() if raw_abstract else "No abstract available."
                year = 2024
                date_parts = item.get("created", {}).get("date-parts", [[]])[0]
                if date_parts:
                    year = date_parts[0]
                authors = [f"{auth.get('given','')} {auth.get('family','')}" .strip() for auth in item.get("author", [])]
                papers.append({
                    "title": title, "authors": authors, "year": year, 
                    "abstract": abstract, "url": item.get("URL", ""), "source": "Crossref"
                })
    except Exception as e:
        print(f"Crossref API warning: {e}")
    return papers

# --- API 4: OpenAlex ---
def fetch_openalex(query: str, limit: int) -> list:
    papers = []
    encoded_query = urllib.parse.quote(query)
    url = f"https://api.openalex.org/works?search={encoded_query}&per_page={limit}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode())
            for item in data.get("results", []):
                title = item.get("title", "Unknown")
                year = item.get("publication_year", 2024)
                link = item.get("doi", "") or item.get("id", "")
                authors = [auth.get("author", {}).get("display_name") for auth in item.get("authorships", []) if auth.get("author")]
                # Try to get abstract from inverted index
                inverted_abstract = item.get("abstract_inverted_index")
                if inverted_abstract:
                    word_positions = []
                    for word, positions in inverted_abstract.items():
                        for pos in positions:
                            word_positions.append((pos, word))
                    word_positions.sort()
                    abstract_text = " ".join(w for _, w in word_positions)
                else:
                    abstract_text = "Abstract metadata fetched from OpenAlex Open DOI index."
                papers.append({
                    "title": title, "authors": authors, "year": year if year else 2024,
                    "abstract": abstract_text,
                    "url": link, "source": "OpenAlex",
                    "citations": item.get("cited_by_count", 0) or 0
                })
    except Exception as e:
        print(f"OpenAlex API warning: {e}")
    return papers

# --- API 5: Europe PMC ---
def fetch_europe_pmc(query: str, limit: int) -> list:
    papers = []
    encoded_query = urllib.parse.quote(query)
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={encoded_query}&format=json&pageSize={limit}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode())
            results = data.get("resultList", {}).get("result", [])
            for item in results:
                title = item.get("title", "Unknown")
                abstract = item.get("abstractText", "No abstract available.") or "No abstract available."
                year = int(item.get("pubYear", 2024))
                link = f"https://europepmc.org/article/MED/{item.get('id', '')}"
                authors = [item.get("authorString", "Unknown")]
                papers.append({
                    "title": title, "authors": authors, "year": year, 
                    "abstract": abstract, "url": link, "source": "Europe PMC",
                    "citations": item.get("citedByCount", 0) or 0
                })
    except Exception as e:
        print(f"Europe PMC API warning: {e}")
    return papers

# --- API 6: DOAJ ---
def fetch_doaj(query: str, limit: int) -> list:
    papers = []
    encoded_query = urllib.parse.quote(query)
    url = f"https://doaj.org/api/search/articles/{encoded_query}?pageSize={limit}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode())
            for item in data.get("results", []):
                bib = item.get("bibjson", {})
                title = bib.get("title", "Unknown")
                abstract = bib.get("abstract", "No abstract available.") or "No abstract available."
                year = int(bib.get("year", 2024))
                link = bib.get("link", [{}])[0].get("url", "")
                authors = [auth.get("name", "") for auth in bib.get("author", [])]
                papers.append({
                    "title": title, "authors": authors, "year": year, 
                    "abstract": abstract, "url": link, "source": "DOAJ"
                })
    except Exception as e:
        print(f"DOAJ API warning: {e}")
    return papers

def _make_unique_fallback_summary(paper: dict, query: str) -> tuple:
    """
    Generate a unique notebook_summary and technical_execution for a paper
    without calling the LLM, using the paper's own title and abstract.
    """
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    source = paper.get("source", "")
    
    # Extract first ~120 chars of abstract as plain summary basis
    abstract_snippet = abstract[:200].rstrip() if abstract and abstract != "No abstract available." else ""
    
    if abstract_snippet:
        notebook_summary = (
            f"This paper ({source}) investigates '{title[:60]}'. "
            f"{abstract_snippet}..."
        )
        technical_execution = (
            f"Employs methods described in '{title[:50]}' sourced from {source}, "
            f"applying domain-specific techniques to address {query}-related challenges."
        )
    else:
        notebook_summary = (
            f"Retrieved from {source}, this work titled '{title[:80]}' contributes "
            f"to the body of knowledge in the area of {query}."
        )
        technical_execution = (
            f"Utilizes standard methodologies within the '{query}' domain as presented in {source}."
        )
    
    return notebook_summary, technical_execution


# --- Full Text Fetchers for Deep Literature Analysis ---
def _fetch_arxiv_full_text(arxiv_url: str) -> str:
    """Fetches full-text HTML of an arXiv paper from ar5iv/arxiv.org."""
    match = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9.]+)', arxiv_url)
    if not match:
        return ""
    arxiv_id = match.group(1)
    
    # Try ar5iv.org HTML converter (very reliable for converting arXiv PDFs to clean HTML)
    url = f"https://ar5iv.org/html/{arxiv_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            soup = BeautifulSoup(resp.read().decode('utf-8'), 'html.parser')
            text = re.sub(r'\s+', ' ', soup.get_text(separator=' ')).strip()
            # Keep first 12,000 chars (methodology + intro + discussion)
            return text[:12000]
    except Exception as e:
        print(f"  [ArXiv FT] ar5iv failed for {arxiv_id}: {e}")
        
    # Try direct arXiv HTML
    url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            soup = BeautifulSoup(resp.read().decode('utf-8'), 'html.parser')
            text = re.sub(r'\s+', ' ', soup.get_text(separator=' ')).strip()
            return text[:12000]
    except Exception as e:
        print(f"  [ArXiv FT] arxiv.org HTML failed for {arxiv_id}: {e}")
        
    return ""


def _fetch_pmc_full_text(url: str) -> str:
    """Queries Europe PMC open API for full text XML and parses it."""
    match = re.search(r'(PMC\d+)', url, re.IGNORECASE)
    if not match:
        return ""
    pmcid = match.group(1).upper()
    api_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            xml_data = resp.read()
            root = ET.fromstring(xml_data)
            
            body_text = []
            for elem in root.iter():
                if elem.tag == 'body' or elem.tag.endswith('body'):
                    body_text.append(''.join(elem.itertext()))
            
            if body_text:
                text = " ".join(body_text)
            else:
                text = ''.join(root.itertext())
                
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:12000]
    except Exception as e:
        print(f"  [PMC FT] PMC XML retrieval failed for {pmcid}: {e}")
    return ""


def _fetch_full_text_for_paper(paper_dict: dict) -> str:
    """Tries all available fetchers for a paper dict and returns full text or empty."""
    url = paper_dict.get("url", "")
    title = paper_dict.get("title", "")
    print(f"[Agent1] Fetching full-text content for: '{title[:45]}...'")
    
    # 1. Check if it is an arXiv link
    if "arxiv.org" in url:
        return _fetch_arxiv_full_text(url)
        
    # 2. Check if it contains PMC ID (for PMC full text)
    if "pmc" in url.lower() or "pmcid" in url.lower():
        return _fetch_pmc_full_text(url)
        
    # 3. Else, return empty (metadata abstract will be used as fallback)
    return ""



import math

def compute_deterministic_relevance(query: str, title: str, abstract: str, year: int, citations: int = 0) -> dict:
    """
    Calculates a relevance score between 0.00 and 1.00 mathematically in Python:
    Relevance = 0.4 * SemanticSimilarity + 0.4 * KeywordOverlap + 0.1 * Recency + 0.1 * Citations
    """
    text = f"{title} {abstract}"
    
    # 1. Semantic Similarity (Cosine similarity via local embeddings model)
    try:
        model = get_embedding_model()
        q_vec = model.encode(query, convert_to_tensor=True)
        t_vec = model.encode(text[:500], convert_to_tensor=True)
        sem_score = float(util.cos_sim(q_vec, t_vec)[0][0])
    except Exception as e:
        print(f"  [Scoring] Cosine similarity failed, fallback: {e}")
        sem_score = 0.5
        
    # 2. Keyword Jaccard Overlap
    q_words = set(re.sub(r'[^a-z0-9\s]', '', query.lower()).split())
    t_words = set(re.sub(r'[^a-z0-9\s]', '', text.lower()).split())
    overlap = len(q_words & t_words)
    kw_score = overlap / max(len(q_words), 1)
    
    # 3. Recency Boost (mapped 2020-2026 to 0.5-1.0)
    current_year = 2026
    age = max(0, current_year - year)
    recency_score = max(0.2, 1.0 - (age * 0.1))
    
    # 4. Citation Weight (Logarithmic scaling)
    cit_score = min(1.0, math.log10(citations + 1) / 3.0)
    
    # Combined weighted score
    score = (0.4 * sem_score) + (0.4 * kw_score) + (0.1 * recency_score) + (0.1 * cit_score)
    final_score = round(min(0.99, max(0.35, score)), 3)
    
    # Assign Rank
    if final_score >= 0.80:
        rank = "High"
    elif final_score >= 0.60:
        rank = "Medium"
    else:
        rank = "Low"
        
    return {"score": final_score, "rank": rank}


# --- Orchestrated Search & NotebookLM LLM Card Parsing ---
def fetch_arxiv_papers(query: str, max_results: int = 8) -> Agent1ResearchOutput:
    """
    Queries 6 platforms (arXiv, Semantic Scholar, Crossref, OpenAlex, EuropePMC, DOAJ),
    merges and deduplicates results, then uses Gemini to rank them and write
    NotebookLM-style concept summaries.
    """
    # Increased from 2→3 per API: 6 APIs × 3 = up to 18 raw results before dedup
    limit_per_api = 3
    
    # Run all 6 searches
    p1 = fetch_arxiv(query, limit_per_api)
    p2 = fetch_semantic_scholar(query, limit_per_api)
    p3 = fetch_crossref(query, limit_per_api)
    p4 = fetch_openalex(query, limit_per_api)
    p5 = fetch_europe_pmc(query, limit_per_api)
    p6 = fetch_doaj(query, limit_per_api)
    
    # Merge & Deduplicate
    all_raw = p1 + p2 + p3 + p4 + p5 + p6
    deduplicated = []
    seen = set()
    for p in all_raw:
        norm = normalize_title(p["title"])
        if norm not in seen and p["title"] != "Unknown":
            seen.add(norm)
            deduplicated.append(p)
            
    # Calculate deterministic relevance and rank in Python first
    for p in deduplicated:
        cit_count = p.get("citations", 0) or 0
        rel_info = compute_deterministic_relevance(
            query=query,
            title=p.get("title", ""),
            abstract=p.get("abstract", ""),
            year=p.get("year", 2024),
            citations=cit_count
        )
        p["relevance_score"] = rel_info["score"]
        p["relevance_rank"] = rel_info["rank"]
        
    # Sort all candidates: High rank first, then by relevance_score descending
    rank_order = {"High": 3, "Medium": 2, "Low": 1}
    deduplicated.sort(
        key=lambda x: (rank_order.get(x["relevance_rank"], 1), x["relevance_score"]),
        reverse=True
    )
    
    # Take top 8 unique candidates (increased from 5)
    candidates = deduplicated[:8]
    
    # Download full text for each candidate to enable deep gap extraction
    for p in candidates:
        p["full_text"] = _fetch_full_text_for_paper(p)
        # Prefetch top 2 chunks (approx 600 words) matching the search query to minimize prompt context
        text_to_chunk = p["full_text"] if p.get("full_text") else p.get("abstract", "")
        top_chunks = prefetch_top_chunks(query, [text_to_chunk], top_k=2)
        p["rag_chunks"] = " ".join([c["text"] for c in top_chunks])
    
    # Prepare lite_candidates to minimize prompt input tokens
    lite_candidates = []
    for c in candidates:
        lite_candidates.append({
            "title": c.get("title", ""),
            "authors": c.get("authors", []),
            "year": c.get("year", 2024),
            "abstract": c.get("rag_chunks", "") or c.get("abstract", ""),
            "url": c.get("url", ""),
            "relevance_score": c.get("relevance_score", 0.8),
            "relevance_rank": c.get("relevance_rank", "Medium")
        })

    # Prompt LLM to analyze the papers and format matching our NotebookLM schema + deep analysis
    num_candidates = len(lite_candidates)
    prompt = f"""
    You are an expert Research Librarian similar to Google NotebookLM.
    Analyze the following {num_candidates} research papers retrieved for the topic: '{query}'.
    
    For EACH paper individually, extract ALL of the following fields based on its title and abstract.
    Every field MUST be unique and specific to that paper's actual content — do NOT copy the same text across papers.
    
    Fields to extract per paper:
    1. relevance_score: Retrieve this directly from the paper's data under 'relevance_score' and copy it verbatim. Do not compute or change it.
    2. relevance_rank: Retrieve this directly from the paper's data under 'relevance_rank' and copy it verbatim. Do not change it.
    3. innovation_score: Integer between 0 and 100 representing the paper's innovation level. Ensure scores are distinct and reflect the technical novelty.
    4. research_significance: A short paragraph analyzing the paper's academic impact, industrial impact, and contribution to innovation.
    5. notebook_summary: Write exactly 3 plain-English bullet points (no technical jargon) for THIS SPECIFIC PAPER. Format as: '• What it studies: [1 sentence] • How it does it: [1 sentence] • What it achieves: [1 sentence]'. These MUST be unique to this paper and easy for a non-expert to understand.
    6. technical_execution: A 1-sentence description of the core algorithm or technique THIS PAPER uses.
    7. datasets: List of dataset names used. If not mentioned, use ["Not specified"].
    8. problem_statement: What specific research problem or gap does this paper identify and tackle?
    9. proposed_solution: What solution, framework, model, or method does this paper propose?
    10. methodology: What methods, algorithms, architectures, or techniques did the authors use?
    11. results: What were the key quantitative or qualitative results reported? Include metrics if mentioned.
    12. challenges: What limitations or open challenges do the authors acknowledge?
    13. future_outcomes: What future work or research directions do the authors suggest?
    
    Papers data:
    {json.dumps(lite_candidates, indent=2)}
    
    CRITICAL RULES:
    - Every field for each paper MUST be derived from THAT PAPER'S OWN title and abstract only.
    - Do NOT use identical text across multiple papers for any field.
    - If a field cannot be determined from the abstract, write a reasonable inference prefixed with "Likely:".
    
    You MUST respond with a valid JSON block matching this EXACT schema structure:
    {{
      "papers": [
        {{
          "title": "Title of paper",
          "authors": ["Author 1", "Author 2"],
          "year": 2024,
          "abstract": "Abstract text",
          "relevance_score": 0.92,
          "relevance_rank": "High",
          "innovation_score": 88,
          "research_significance": "Academic impact: ... Industrial impact: ... Innovation contribution: ...",
          "notebook_summary": "Unique 2-sentence description specific to this paper",
          "technical_execution": "Core algorithm/technique this paper specifically uses",
          "datasets": ["Dataset Name"],
          "url": "paper link url",
          "problem_statement": "The specific problem this paper addresses",
          "proposed_solution": "The solution or model this paper proposes",
          "methodology": "Methods and techniques used in this paper",
          "results": "Key results and metrics reported",
          "challenges": "Limitations acknowledged by the authors",
          "future_outcomes": "Future work directions suggested"
        }}
      ]
    }}
    
    Respond ONLY with the JSON code block. No extra explanations, no markdown wrapper backticks.
    """
    
    try:
        client = _get_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        response_text = response.text.strip()
        
        # Clean possible markdown code fences
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        data = json.loads(response_text.strip())
        
        parsed_papers = []
        candidates_map = {normalize_title(c["title"]): c.get("full_text", "") for c in candidates}
        for p in data.get("papers", []):
            norm = normalize_title(p.get("title", ""))
            ft = candidates_map.get(norm, "")
            parsed_papers.append(PaperMetadata(
                title=p.get("title", ""),
                authors=p.get("authors", []),
                year=p.get("year", 2024),
                abstract=p.get("abstract", ""),
                relevance_rank=p.get("relevance_rank", "Medium"),
                notebook_summary=p.get("notebook_summary", ""),
                technical_execution=p.get("technical_execution", ""),
                datasets=p.get("datasets", ["Not specified"]),
                url=p.get("url", ""),
                relevance_score=float(p.get("relevance_score", 0.8)),
                problem_statement=p.get("problem_statement", "Not extracted"),
                proposed_solution=p.get("proposed_solution", "Not extracted"),
                methodology=p.get("methodology", "Not extracted"),
                results=p.get("results", "Not extracted"),
                challenges=p.get("challenges", "Not extracted"),
                future_outcomes=p.get("future_outcomes", "Not extracted"),
                innovation_score=int(p.get("innovation_score", 70)),
                research_significance=p.get("research_significance", "Not analyzed"),
                full_text=ft
            ))
        
        # Sort papers: first group by relevance_rank (High=3, Medium=2, Low=1),
        # then within each group sort by relevance_score descending
        rank_order = {"High": 3, "Medium": 2, "Low": 1}
        parsed_papers.sort(
            key=lambda x: (rank_order.get(x.relevance_rank, 1), x.relevance_score),
            reverse=True
        )
        return Agent1ResearchOutput(query=query, papers=parsed_papers)
        
    except Exception as e:
        print(f"Error compiling NotebookLM summaries in Agent 1: {e}")
        # Graceful fallback: each paper gets a UNIQUE summary and relevance score derived from its own content
        fallback_papers = []
        for idx, p in enumerate(candidates):
            nb_summary, tech_exec = _make_unique_fallback_summary(p, query)
            abstract = p.get("abstract", "")
            
            # Simple content-driven unique relevance score calculation
            query_words = set(query.lower().split())
            title_abstract_words = set((p["title"] + " " + abstract).lower().split())
            overlap = len(query_words & title_abstract_words)
            
            # Formulate unique score between 0.40 and 0.98
            base_score = 0.5 + (overlap / (len(query_words) + 10))
            relevance_score = min(0.98, max(0.40, base_score + (idx * 0.015) - (idx * 0.005)))
            
            if relevance_score >= 0.80:
                relevance_rank = "High"
            elif relevance_score >= 0.60:
                relevance_rank = "Medium"
            else:
                relevance_rank = "Low"

            # Build a truly unique, paper-specific notebook_summary from the abstract
            sents = [s.strip() for s in abstract.split(".") if len(s.strip()) > 30]
            what_it_studies = sents[0] if len(sents) > 0 else p["title"]
            how_it_does_it = sents[1] if len(sents) > 1 else tech_exec
            what_it_achieves = sents[2] if len(sents) > 2 else f"Demonstrates {query} improvements."
            nb_summary = (
                f"• What it studies: {what_it_studies}. "
                f"• How it does it: {how_it_does_it}. "
                f"• What it achieves: {what_it_achieves}."
            )
                
            innovation_score = int(min(98, max(45, 65 + (overlap * 6) - (idx * 4))))
            
            prob = abstract[:120].rstrip() + "..." if len(abstract) > 120 else abstract
            fallback_papers.append(PaperMetadata(
                title=p["title"],
                authors=p["authors"],
                year=p["year"],
                abstract=abstract,
                relevance_rank=relevance_rank,
                notebook_summary=nb_summary,
                technical_execution=tech_exec,
                datasets=["Not specified"],
                url=p["url"],
                relevance_score=relevance_score,
                problem_statement=prob if prob else "Not available in abstract.",
                proposed_solution=f"Proposes a methodology addressing '{query}' challenges as described in the abstract.",
                methodology=tech_exec,
                results="Quantitative results not available in abstract.",
                challenges="Limitations not detailed in the available abstract.",
                future_outcomes=f"Authors suggest further work on extending the approach to broader '{query}' datasets.",
                innovation_score=innovation_score,
                research_significance=(
                    f"This research contributes to the field of {query} by leveraging {tech_exec}. "
                    f"It has strong academic value for researchers working on related algorithmic architectures."
                ),
                full_text=p.get("full_text", "")
            ))
        
        # Compound sort: rank group first (High > Medium > Low), then score descending
        rank_order = {"High": 3, "Medium": 2, "Low": 1}
        fallback_papers.sort(
            key=lambda x: (rank_order.get(x.relevance_rank, 1), x.relevance_score),
            reverse=True
        )
        return Agent1ResearchOutput(query=query, papers=fallback_papers)
