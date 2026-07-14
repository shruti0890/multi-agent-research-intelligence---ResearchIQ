import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import re
from schemas import Agent1ResearchOutput, PaperMetadata  # type: ignore

def normalize_title(title: str) -> str:
    """Helper to clean titles for deduplication checks."""
    return re.sub(r'[^a-z0-9]', '', title.lower())

# --- API 1: arXiv Search ---
def fetch_arxiv(query: str, limit: int) -> list:
    papers = []
    encoded_query = urllib.parse.quote(query)
    url = f'http://export.arxiv.org/api/query?search_query=all:{encoded_query}&max_results={limit}'
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            for entry in root.findall('atom:entry', ns):
                title_elem = entry.find('atom:title', ns)
                summary_elem = entry.find('atom:summary', ns)
                id_elem = entry.find('atom:id', ns)
                published_elem = entry.find('atom:published', ns)
                
                title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else "Unknown"
                abstract = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else "No abstract."
                link = id_elem.text.strip() if id_elem is not None else ""
                
                year = 2024
                if published_elem is not None and len(published_elem.text) >= 4:
                    try:
                        year = int(published_elem.text[:4])
                    except ValueError:
                        pass
                
                authors = [author.find('atom:name', ns).text.strip() for author in entry.findall('atom:author', ns) if author.find('atom:name', ns) is not None]
                
                papers.append(PaperMetadata(
                    title=title,
                    authors=authors,
                    year=year,
                    abstract=abstract,
                    key_findings=["Method found in preprint source."],
                    datasets=["Unknown"],
                    url=link,
                    relevance_score=0.9
                ))
    except Exception as e:
        print(f"arXiv API warning: {e}")
    return papers

# --- API 2: Semantic Scholar Search ---
def fetch_semantic_scholar(query: str, limit: int) -> list:
    papers = []
    encoded_query = urllib.parse.quote(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&limit={limit}&fields=title,authors,year,abstract,citationCount,url"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode())
            for item in data.get("data", []):
                title = item.get("title", "Unknown")
                abstract = item.get("abstract", "No abstract available.") or "No abstract available."
                year = item.get("year", 2024)
                link = item.get("url", "")
                
                authors = [auth.get("name") for auth in item.get("authors", []) if auth.get("name")]
                
                # Check metrics
                citations = item.get("citationCount", 0)
                relevance = 0.95 if citations > 50 else 0.88
                
                papers.append(PaperMetadata(
                    title=title,
                    authors=authors,
                    year=year if year else 2024,
                    abstract=abstract,
                    key_findings=[f"Highly cited paper ({citations} citations)."],
                    datasets=["See paper text"],
                    url=link if link else "",
                    relevance_score=relevance
                ))
    except Exception as e:
        print(f"Semantic Scholar API warning: {e}")
    return papers

# --- API 3: Crossref Search ---
def fetch_crossref(query: str, limit: int) -> list:
    papers = []
    encoded_query = urllib.parse.quote(query)
    url = f"https://api.crossref.org/works?query={encoded_query}&rows={limit}"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (mailto:admin@researchiq.io)'})
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode())
            items = data.get("message", {}).get("items", [])
            for item in items:
                titles = item.get("title", [])
                title = titles[0] if titles else "Unknown"
                
                # Crossref abstracts are stored as XML strings, we clean them if present
                raw_abstract = item.get("abstract", "No abstract available.") or "No abstract available."
                abstract = re.sub('<[^<]+?>', '', raw_abstract).strip() # strip XML tags
                
                # Extract year
                year = 2024
                date_parts = item.get("created", {}).get("date-parts", [[]])[0]
                if date_parts:
                    year = date_parts[0]
                    
                link = item.get("URL", "")
                
                authors = []
                for auth in item.get("author", []):
                    given = auth.get("given", "")
                    family = auth.get("family", "")
                    if given or family:
                        authors.append(f"{given} {family}".strip())
                        
                papers.append(PaperMetadata(
                    title=title,
                    authors=authors,
                    year=year,
                    abstract=abstract,
                    key_findings=["Crossref registered metadata."],
                    datasets=["Check DOI references"],
                    url=link,
                    relevance_score=0.85
                ))
    except Exception as e:
        print(f"Crossref API warning: {e}")
    return papers

# --- Main Multi-API Orchestration ---
def fetch_arxiv_papers(query: str, max_results: int = 6) -> Agent1ResearchOutput:
    """
    Fetches, normalizes, and deduplicates papers across arXiv, Semantic Scholar, and Crossref.
    """
    # 1. Fetch from all three APIs concurrently (2 papers per API)
    limit_per_api = max(2, max_results // 2)
    
    arxiv_list = fetch_arxiv(query, limit_per_api)
    scholar_list = fetch_semantic_scholar(query, limit_per_api)
    crossref_list = fetch_crossref(query, limit_per_api)
    
    # 2. Merge and Deduplicate
    all_papers = arxiv_list + scholar_list + crossref_list
    deduplicated_papers = []
    seen_titles = set()
    
    for paper in all_papers:
        norm = normalize_title(paper.title)
        if norm not in seen_titles and paper.title != "Unknown":
            seen_titles.add(norm)
            deduplicated_papers.append(paper)
            
    # 3. Sort by relevance score (highest first)
    deduplicated_papers.sort(key=lambda x: x.relevance_score, reverse=True)
    
    # Slice to final desired results count
    final_list = deduplicated_papers[:max_results]
    
    return Agent1ResearchOutput(query=query, papers=final_list)
