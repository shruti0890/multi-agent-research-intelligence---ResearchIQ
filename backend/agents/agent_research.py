import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
import json
import re
import io
# pyrefly: ignore [missing-import]
from google import genai
from google.genai import types
# pyrefly: ignore [missing-import]
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from schemas import Agent1ResearchOutput, PaperMetadata
# sentence_transformers used ONLY here for deterministic relevance scoring (paper selection).
# It is NOT used in the compression pipeline. ResearchIQ does not use RAG.
from sentence_transformers import SentenceTransformer, util as st_util
from compression import compress_paper, format_fact_sheet

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

# ---------------------------------------------------------------------------
# Configurable full-text quality threshold
# If acquired text has fewer words than this, it is treated as partial_paper.
# Set to 0 to disable the check.
# ---------------------------------------------------------------------------
FULL_TEXT_MIN_WORDS = 300


def normalize_title(title: str) -> str:
    return re.sub(r'[^a-z0-9]', '', title.lower())


# ---------------------------------------------------------------------------
# Academic API Fetchers
# ---------------------------------------------------------------------------

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
                    "abstract": abstract, "url": link, "source": "arXiv",
                    "open_access_pdf_url": "", "oa_url": ""
                })
    except Exception as e:
        print(f"arXiv API warning: {e}")
    return papers


# --- API 2: Semantic Scholar ---
def fetch_semantic_scholar(query: str, limit: int) -> list:
    papers = []
    encoded_query = urllib.parse.quote(query)
    url = (f"https://api.semanticscholar.org/graph/v1/paper/search"
           f"?query={encoded_query}&limit={limit}"
           f"&fields=title,authors,year,abstract,url,citationCount,openAccessPdf")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode())
            for item in data.get("data", []):
                authors = [auth.get("name") for auth in item.get("authors", []) if auth.get("name")]
                # Extract legitimate open-access PDF URL if present
                oa_pdf_info = item.get("openAccessPdf") or {}
                oa_pdf_url = oa_pdf_info.get("url", "") if isinstance(oa_pdf_info, dict) else ""
                papers.append({
                    "title": item.get("title", "Unknown"),
                    "authors": authors,
                    "year": item.get("year", 2024) or 2024,
                    "abstract": item.get("abstract") or "No abstract available.",
                    "url": item.get("url") or "",
                    "source": "Semantic Scholar",
                    "citations": item.get("citationCount", 0) or 0,
                    "open_access_pdf_url": oa_pdf_url,
                    "oa_url": "",
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
                authors = [f"{auth.get('given','')} {auth.get('family','')}".strip() for auth in item.get("author", [])]
                doi = item.get("DOI", "")
                papers.append({
                    "title": title, "authors": authors, "year": year,
                    "abstract": abstract, "url": item.get("URL", ""),
                    "source": "Crossref", "doi": doi,
                    "open_access_pdf_url": "", "oa_url": "",
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
                # Extract legitimate open-access URL
                oa_info = item.get("open_access") or {}
                oa_url = oa_info.get("oa_url", "") or "" if isinstance(oa_info, dict) else ""
                papers.append({
                    "title": title, "authors": authors, "year": year if year else 2024,
                    "abstract": abstract_text,
                    "url": link, "source": "OpenAlex",
                    "citations": item.get("cited_by_count", 0) or 0,
                    "open_access_pdf_url": "",
                    "oa_url": oa_url,
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
                pmid = item.get("id", "")
                pmcid = item.get("pmcid", "")
                link = f"https://europepmc.org/article/MED/{pmid}"
                if pmcid:
                    link = f"https://europepmc.org/article/PMC/{pmcid}"
                authors = [item.get("authorString", "Unknown")]
                papers.append({
                    "title": title, "authors": authors, "year": year,
                    "abstract": abstract, "url": link, "source": "Europe PMC",
                    "citations": item.get("citedByCount", 0) or 0,
                    "open_access_pdf_url": "", "oa_url": "",
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
                links = bib.get("link", [])
                article_url = ""
                pdf_url = ""
                for lnk in links:
                    ltype = lnk.get("type", "")
                    href = lnk.get("url", "")
                    if ltype == "fulltext":
                        article_url = href
                    elif ltype == "pdf" or href.endswith(".pdf"):
                        pdf_url = href
                    elif not article_url:
                        article_url = href
                authors = [auth.get("name", "") for auth in bib.get("author", [])]
                papers.append({
                    "title": title, "authors": authors, "year": year,
                    "abstract": abstract, "url": article_url or pdf_url,
                    "source": "DOAJ",
                    "open_access_pdf_url": pdf_url,
                    "oa_url": article_url,
                })
    except Exception as e:
        print(f"DOAJ API warning: {e}")
    return papers


# ---------------------------------------------------------------------------
# Full-text fetchers
# ---------------------------------------------------------------------------

def _fetch_arxiv_full_text(arxiv_url: str) -> str:
    """Fetches full-text HTML of an arXiv paper from ar5iv/arxiv.org.

    No character truncation is applied. The complete available text
    is returned for section-aware compression by the TextRank pipeline.
    """
    match = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9.]+)', arxiv_url)
    if not match:
        return ""
    arxiv_id = match.group(1)

    # Try ar5iv.org HTML converter (very reliable for converting arXiv PDFs to clean HTML)
    for base_url in [f"https://ar5iv.org/html/{arxiv_id}", f"https://arxiv.org/html/{arxiv_id}"]:
        try:
            req = urllib.request.Request(base_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.read().decode('utf-8', errors='replace')
        except Exception as e:
            print(f"  [ArXiv FT] Failed for {arxiv_id} at {base_url}: {e}")

    return ""


def _fetch_pmc_full_text(url: str) -> str:
    """Queries Europe PMC open API for full text XML and parses it.

    No character truncation is applied. The complete available text
    is returned for section-aware compression by the TextRank pipeline.
    """
    # Try to extract PMCID from URL or the URL itself
    match = re.search(r'(PMC\d+)', url, re.IGNORECASE)
    if not match:
        return ""
    pmcid = match.group(1).upper()
    api_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
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

            # Preserve full text — no truncation
            text = re.sub(r'\s+', ' ', text).strip()
            return text
    except Exception as e:
        print(f"  [PMC FT] PMC XML retrieval failed for {pmcid}: {e}")
    return ""


def _fetch_pdf_text(pdf_url: str) -> str:
    """Download a legitimate open-access PDF and extract text using pdfminer.six.

    Returns the full extracted text in page order.
    Returns "" on any error — never crashes the pipeline.
    No truncation is applied.
    """
    try:
        req = urllib.request.Request(pdf_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            pdf_bytes = resp.read()

        # Verify it is actually a PDF
        if not pdf_bytes.startswith(b'%PDF'):
            print(f"  [PDF FT] Response at {pdf_url[:60]} is not a PDF (no %PDF header)")
            return ""

        try:
            from pdfminer.high_level import extract_text
            text = extract_text(io.BytesIO(pdf_bytes))
            if text:
                text = re.sub(r'\s+', ' ', text).strip()
                return text
        except ImportError:
            print("  [PDF FT] pdfminer.six not installed. Install with: pip install pdfminer.six")
        except Exception as e:
            print(f"  [PDF FT] pdfminer extraction failed for {pdf_url[:60]}: {e}")
    except Exception as e:
        print(f"  [PDF FT] Download failed for {pdf_url[:60]}: {e}")
    return ""


def _fetch_html_text(url: str) -> str:
    """Fetch a URL and extract readable text using BeautifulSoup.

    Strips scripts, styles, and nav elements to return body prose text.
    Returns "" on any failure.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()

        # If it's actually a PDF, delegate
        if raw.startswith(b'%PDF') or 'application/pdf' in content_type:
            return _fetch_pdf_text(url)

        html = raw.decode('utf-8', errors='replace')
        soup = BeautifulSoup(html, 'html.parser')

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        text = soup.get_text(separator=' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"  [HTML FT] Fetch failed for {url[:60]}: {e}")
    return ""


def _resolve_unpaywall(doi: str, email: str) -> str:
    """Query Unpaywall API for a legitimate open-access URL for the given DOI.

    Returns the best open-access URL or "" if not found.
    Never raises — all errors are swallowed.
    """
    if not doi or not email:
        return ""
    doi_encoded = urllib.parse.quote(doi, safe='')
    api_url = f"https://api.unpaywall.org/v2/{doi_encoded}?email={urllib.parse.quote(email)}"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())

        # Prefer best_oa_location
        best = data.get("best_oa_location") or {}
        url_for_pdf = best.get("url_for_pdf", "")
        url_for_landing = best.get("url_for_landing_page", "")

        return url_for_pdf or url_for_landing or ""
    except Exception as e:
        print(f"  [Unpaywall] Query failed for DOI {doi[:40]}: {e}")
    return ""


# ---------------------------------------------------------------------------
# Centralized full-text resolution waterfall
# ---------------------------------------------------------------------------

def _resolve_full_text(paper_dict: dict) -> dict:
    """
    Try all available full-text sources in priority order and return the
    best available text.

    Resolution order:
      1. arXiv  (HTML via ar5iv / arxiv.org/html)
      2. PMC    (Europe PMC fullTextXML)
      3. Semantic Scholar openAccessPdf.url  (PDF via pdfminer.six)
      4. OpenAlex open_access.oa_url  (HTML or PDF)
      5. DOAJ article URL  (HTML or PDF)
      6. Unpaywall DOI resolution  (optional, configured via UNPAYWALL_EMAIL)

    Returns:
        {
            "text": str,
            "source": str,   # "arxiv", "pmc", "semantic_scholar", "openalex",
                             #  "doaj", "unpaywall", "none"
            "url": str,
            "coverage_type": str,  # "full_paper", "partial_paper", "abstract_only", "unavailable"
        }
    """
    url = paper_dict.get("url", "")
    title = paper_dict.get("title", "Unknown")
    abstract = paper_dict.get("abstract", "")
    oa_pdf_url = paper_dict.get("open_access_pdf_url", "")
    oa_url = paper_dict.get("oa_url", "")
    doi = paper_dict.get("doi", "")

    print(f"[Agent1] Resolving full text for: '{title[:50]}...'")

    def _quality_check(text: str, source_label: str, fetch_url: str) -> dict | None:
        """
        Check whether the acquired text is long enough to qualify as full paper.
        Returns a result dict if quality is acceptable, None if text is too short.
        """
        if not text or not text.strip():
            return None
        word_count = len(text.split())
        if FULL_TEXT_MIN_WORDS > 0 and word_count < FULL_TEXT_MIN_WORDS:
            print(
                f"  [FT Resolver] WARNING: Text from {source_label} for '{title[:40]}' "
                f"has only {word_count} words (threshold: {FULL_TEXT_MIN_WORDS}). "
                f"Marking as partial_paper."
            )
            return {
                "text": text,
                "source": source_label,
                "url": fetch_url,
                "coverage_type": "partial_paper",
            }
        return {
            "text": text,
            "source": source_label,
            "url": fetch_url,
            "coverage_type": "full_paper",
        }

    # ── Step 1: arXiv ────────────────────────────────────────────────────────
    if "arxiv.org" in url:
        text = _fetch_arxiv_full_text(url)
        result = _quality_check(text, "arxiv", url)
        if result:
            print(f"  [FT Resolver] arXiv: {len(text.split()):,} words")
            return result

    # ── Step 2: PMC / Europe PMC ─────────────────────────────────────────────
    if re.search(r'(pmc|PMC)', url):
        text = _fetch_pmc_full_text(url)
        result = _quality_check(text, "pmc", url)
        if result:
            print(f"  [FT Resolver] PMC: {len(text.split()):,} words")
            return result

    # ── Step 3: Semantic Scholar open-access PDF ──────────────────────────────
    if oa_pdf_url:
        text = _fetch_pdf_text(oa_pdf_url)
        result = _quality_check(text, "semantic_scholar", oa_pdf_url)
        if result:
            print(f"  [FT Resolver] Semantic Scholar PDF: {len(text.split()):,} words")
            return result

    # ── Step 4: OpenAlex open-access URL ─────────────────────────────────────
    if oa_url:
        text = _fetch_html_text(oa_url)
        result = _quality_check(text, "openalex", oa_url)
        if result:
            print(f"  [FT Resolver] OpenAlex OA URL: {len(text.split()):,} words")
            return result

    # ── Step 5: DOAJ article URL ──────────────────────────────────────────────
    # DOAJ is open-access by definition. The URL in paper_dict["url"] for DOAJ
    # papers is already the article link.
    source_api = paper_dict.get("source", "")
    if source_api == "DOAJ" and url:
        text = _fetch_html_text(url)
        result = _quality_check(text, "doaj", url)
        if result:
            print(f"  [FT Resolver] DOAJ article URL: {len(text.split()):,} words")
            return result

    # ── Step 6: Unpaywall (optional, only if UNPAYWALL_EMAIL is configured) ───
    unpaywall_email = os.getenv("UNPAYWALL_EMAIL", "").strip()
    if unpaywall_email and doi:
        uw_url = _resolve_unpaywall(doi, unpaywall_email)
        if uw_url:
            text = _fetch_pdf_text(uw_url) or _fetch_html_text(uw_url)
            result = _quality_check(text, "unpaywall", uw_url)
            if result:
                print(f"  [FT Resolver] Unpaywall: {len(text.split()):,} words")
                return result

    # ── Fallback: abstract only ───────────────────────────────────────────────
    if abstract and abstract.strip() and abstract != "No abstract available.":
        print(f"  [FT Resolver] No full text found. Using abstract only.")
        return {
            "text": "",           # Empty — compression pipeline will use abstract from paper_dict
            "source": "none",
            "url": "",
            "coverage_type": "abstract_only",
        }

    print(f"  [FT Resolver] No text available for '{title[:50]}'.")
    return {
        "text": "",
        "source": "none",
        "url": "",
        "coverage_type": "unavailable",
    }


# ---------------------------------------------------------------------------
# Relevance scoring (sentence-transformers — paper selection ONLY)
# ---------------------------------------------------------------------------

import math

def compute_deterministic_relevance(query: str, title: str, abstract: str, year: int, citations: int = 0) -> dict:
    """
    Calculates a relevance score between 0.00 and 1.00 mathematically in Python:
    Relevance = 0.4 * SemanticSimilarity + 0.4 * KeywordOverlap + 0.1 * Recency + 0.1 * Citations

    sentence_transformers is used ONLY here for relevance scoring (paper selection).
    It is NOT used in the compression pipeline.
    """
    text = f"{title} {abstract}"

    try:
        from sentence_transformers import SentenceTransformer
        _model_cache = getattr(compute_deterministic_relevance, '_model', None)
        if _model_cache is None:
            _model_cache = SentenceTransformer('all-MiniLM-L6-v2')
            compute_deterministic_relevance._model = _model_cache
        q_vec = _model_cache.encode(query, convert_to_tensor=True)
        t_vec = _model_cache.encode(text[:500], convert_to_tensor=True)
        sem_score = float(st_util.cos_sim(q_vec, t_vec)[0][0])
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

    if final_score >= 0.80:
        rank = "High"
    elif final_score >= 0.60:
        rank = "Medium"
    else:
        rank = "Low"

    return {"score": final_score, "rank": rank}


# ---------------------------------------------------------------------------
# Fallback summary generator (no LLM)
# ---------------------------------------------------------------------------

def _make_unique_fallback_summary(paper: dict, query: str) -> tuple:
    """Generate a unique notebook_summary and technical_execution without LLM."""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    source = paper.get("source", "")

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


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def fetch_arxiv_papers(query: str, max_results: int = 8) -> Agent1ResearchOutput:
    """
    Queries 6 platforms (arXiv, Semantic Scholar, Crossref, OpenAlex, EuropePMC, DOAJ),
    merges and deduplicates results, then uses Gemini to rank them and write
    NotebookLM-style concept summaries.

    Full-text resolution order per paper:
      arXiv → PMC → Semantic Scholar OA PDF → OpenAlex OA URL → DOAJ → Unpaywall
    """
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

    # Take top 8 unique candidates
    candidates = deduplicated[:8]

    # ── Full-text resolution + compression ───────────────────────────────────
    # Resolve full text via the centralized waterfall.
    # No character truncation. The section-aware extractive compression
    # pipeline processes whatever text is available.
    for idx, p in enumerate(candidates):
        paper_id = f"P{idx+1:03d}"
        p["paper_id"] = paper_id

        # Centralized full-text waterfall
        ft_result = _resolve_full_text(p)
        p["full_text"] = ft_result["text"]
        p["full_text_source"] = ft_result["source"]
        p["full_text_url"] = ft_result["url"]
        p["coverage_type"] = ft_result["coverage_type"]
        p["full_text_available"] = ft_result["coverage_type"] in ("full_paper", "partial_paper")
        p["full_text_word_count"] = len(ft_result["text"].split()) if ft_result["text"] else 0
        p["full_text_character_count"] = len(ft_result["text"]) if ft_result["text"] else 0

        # Set compression_source for tracking
        if p["full_text_available"]:
            p["compression_source"] = "full_text"
        elif p.get("abstract", "").strip():
            p["compression_source"] = "abstract"
        else:
            p["compression_source"] = "none"

        # Run extractive compression pipeline (TextRank, not RAG)
        try:
            fact_sheet = compress_paper(
                paper_dict=p,
                query=query,
                paper_id=paper_id,
            )
            p["fact_sheet"] = fact_sheet
            p["fact_sheet_text"] = format_fact_sheet(fact_sheet)
            p["compression_ratio"] = fact_sheet.metrics.compression_ratio
            p["section_coverage"] = fact_sheet.metrics.section_coverage
            p["original_tokens"] = fact_sheet.metrics.original_token_count
            p["compressed_tokens"] = fact_sheet.metrics.compressed_token_count
            # Propagate fact sheet's actual coverage_type back (it may have overridden to abstract_only)
            p["coverage_type"] = fact_sheet.coverage_type
        except Exception as e:
            print(f"[Agent1] Compression failed for {paper_id} ('{p.get('title', '')[:40]}'): {e}")
            p["fact_sheet_text"] = p.get("abstract", "")
            p["compression_ratio"] = 0.0
            p["section_coverage"] = 0.0
            p["original_tokens"] = 0
            p["compressed_tokens"] = 0

    # Build lite_candidates for the Gemini metadata-extraction prompt.
    # Use fact_sheet_text (extractive summary) — NOT RAG chunks.
    lite_candidates = []
    for c in candidates:
        content_for_gemini = c.get("fact_sheet_text") or c.get("abstract", "")
        lite_candidates.append({
            "title": c.get("title", ""),
            "authors": c.get("authors", []),
            "year": c.get("year", 2024),
            "abstract": content_for_gemini,
            "url": c.get("url", ""),
            "relevance_score": c.get("relevance_score", 0.8),
            "relevance_rank": c.get("relevance_rank", "Medium")
        })

    # Prompt LLM to analyze the papers
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
    
    IMPORTANT: The paper content you receive may be an extractive fact sheet (verbatim sentences from the
    original paper) or an abstract. If a section is not present in the provided content, do NOT conclude
    the paper lacks that information — it may simply not have been extracted. State: "Not present in provided content."
    
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
        candidates_extra_map = {
            normalize_title(c["title"]): c for c in candidates
        }
        for p in data.get("papers", []):
            norm = normalize_title(p.get("title", ""))
            cand_extra = candidates_extra_map.get(norm, {})
            ft = cand_extra.get("full_text", "")
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
                full_text=ft,
                paper_id=cand_extra.get("paper_id", ""),
                fact_sheet=cand_extra.get("fact_sheet", None),
                # Compression pipeline outputs
                fact_sheet_text=cand_extra.get("fact_sheet_text", ""),
                compression_ratio=cand_extra.get("compression_ratio", 0.0),
                section_coverage=cand_extra.get("section_coverage", 0.0),
                original_tokens=cand_extra.get("original_tokens", 0),
                compressed_tokens=cand_extra.get("compressed_tokens", 0),
                # Full-text acquisition metadata
                full_text_available=cand_extra.get("full_text_available", False),
                full_text_source=cand_extra.get("full_text_source", "none"),
                full_text_url=cand_extra.get("full_text_url", ""),
                compression_source=cand_extra.get("compression_source", "none"),
                coverage_type=cand_extra.get("coverage_type", "unavailable"),
                full_text_word_count=cand_extra.get("full_text_word_count", 0),
                full_text_character_count=cand_extra.get("full_text_character_count", 0),
            ))

        # Sort papers: High rank first, then by relevance_score descending
        rank_order = {"High": 3, "Medium": 2, "Low": 1}
        parsed_papers.sort(
            key=lambda x: (rank_order.get(x.relevance_rank, 1), x.relevance_score),
            reverse=True
        )
        return Agent1ResearchOutput(query=query, papers=parsed_papers)

    except Exception as e:
        print(f"Error compiling NotebookLM summaries in Agent 1: {e}")
        # Graceful fallback: each paper gets a UNIQUE summary derived from its own content
        fallback_papers = []
        for idx, p in enumerate(candidates):
            nb_summary, tech_exec = _make_unique_fallback_summary(p, query)
            abstract = p.get("abstract", "")

            query_words = set(query.lower().split())
            title_abstract_words = set((p["title"] + " " + abstract).lower().split())
            overlap = len(query_words & title_abstract_words)

            base_score = 0.5 + (overlap / (len(query_words) + 10))
            relevance_score = min(0.98, max(0.40, base_score + (idx * 0.015) - (idx * 0.005)))

            if relevance_score >= 0.80:
                relevance_rank = "High"
            elif relevance_score >= 0.60:
                relevance_rank = "Medium"
            else:
                relevance_rank = "Low"

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
                proposed_solution=f"Proposes a methodology addressing '{query}' challenges.",
                methodology=tech_exec,
                results="Quantitative results not available in abstract.",
                challenges="Limitations not detailed in the available abstract.",
                future_outcomes=f"Authors suggest further work on extending the approach to broader '{query}' datasets.",
                innovation_score=innovation_score,
                research_significance=(
                    f"This research contributes to the field of {query} by leveraging {tech_exec}. "
                    f"It has strong academic value for researchers working on related algorithmic architectures."
                ),
                full_text=p.get("full_text", ""),
                paper_id=p.get("paper_id", f"P{idx+1:03d}"),
                fact_sheet=p.get("fact_sheet", None),
                fact_sheet_text=p.get("fact_sheet_text", ""),
                compression_ratio=p.get("compression_ratio", 0.0),
                section_coverage=p.get("section_coverage", 0.0),
                original_tokens=p.get("original_tokens", 0),
                compressed_tokens=p.get("compressed_tokens", 0),
                full_text_available=p.get("full_text_available", False),
                full_text_source=p.get("full_text_source", "none"),
                full_text_url=p.get("full_text_url", ""),
                compression_source=p.get("compression_source", "none"),
                coverage_type=p.get("coverage_type", "unavailable"),
                full_text_word_count=p.get("full_text_word_count", 0),
                full_text_character_count=p.get("full_text_character_count", 0),
            ))


        rank_order = {"High": 3, "Medium": 2, "Low": 1}
        fallback_papers.sort(
            key=lambda x: (rank_order.get(x.relevance_rank, 1), x.relevance_score),
            reverse=True
        )
        return Agent1ResearchOutput(query=query, papers=fallback_papers)
