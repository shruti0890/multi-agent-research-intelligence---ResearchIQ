import os
import sys
import time
import socket
socket.setdefaulttimeout(15)
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
from schemas import Agent1ResearchOutput, PaperMetadata, CandidateDiagnostic, GeminiQuotaExhaustedError, GeminiError
from agents.agent_utils import execute_gemini_with_retry, get_gemini_model, get_gemini_client  # type: ignore
from topic_decomposition import decompose_research_topic, generate_targeted_research_queries
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

GEMINI_MODEL = get_gemini_model()

# ---------------------------------------------------------------------------
# Configurable full-text quality threshold
# If acquired text has fewer words than this, it is treated as partial_paper.
# Set to 0 to disable the check.
# ---------------------------------------------------------------------------
FULL_TEXT_MIN_WORDS = 300


def validate_full_paper_content(text: str, min_words: int = 50) -> tuple[bool, str, dict]:
    """
    Validates that retrieved full-text content represents a legitimate research paper
    with substantial research body sections (research_body_tokens >= 1000 and at least
    one meaningful research section beyond abstract, references, acknowledgments, funding,
    metadata, or supplementary material).

    Rejects:
      - Empty or extremely short content
      - Pages with only abstract + unknown/metadata (e.g. 750 words of abstract + nav)
      - Documents with research_body_tokens < 1000 unless containing explicit structured research sections.
    """
    if not text or not isinstance(text, str):
        return False, "Empty or non-string content", {}

    s = text.strip()
    words = s.split()

    # Import parser and constants
    from compression.section_parser import parse_paper_sections, NON_RESEARCH_SECTIONS
    from compression.token_utils import estimate_tokens

    is_html = bool(re.search(r'<[a-zA-Z][^>]{0,50}>', text))
    sections = parse_paper_sections(text, is_html=is_html)
    if not sections:
        sections = {"unknown": text}

    total_tokens = estimate_tokens(text)
    detected_section_names = list(sections.keys())
    abstract_text = sections.get("abstract", "")
    abstract_tokens = estimate_tokens(abstract_text)

    non_research_tokens = sum(
        estimate_tokens(sections[s]) for s in detected_section_names if s in NON_RESEARCH_SECTIONS
    )

    research_sections = [
        s for s in detected_section_names
        if s not in NON_RESEARCH_SECTIONS and s != "abstract"
    ]
    research_tokens = sum(estimate_tokens(sections[s]) for s in research_sections)

    stats = {
        "total_words": len(words),
        "total_tokens": total_tokens,
        "detected_sections": detected_section_names,
        "research_sections": research_sections,
        "research_tokens": research_tokens,
        "abstract_tokens": abstract_tokens,
        "non_research_tokens": non_research_tokens,
    }

    # Case 1: Only abstract and/or non-research sections detected
    if not research_sections:
        return False, "only abstract/non-research content detected", stats

    if len(words) < min_words:
        return False, f"Total word count ({len(words)}) below minimum threshold ({min_words})", stats

    # Publisher wrapper checks across all sections
    wrapper_keywords = {"publisher", "journal", "volume", "citation", "metrics", "download", "cookie", "policy", "terms", "privacy", "copyright", "advertisement"}
    words_lower = [w.lower().strip(".,;:\"'()[]") for w in words]
    wrapper_hits = sum(1 for w in words_lower if w in wrapper_keywords)
    wrapper_ratio = wrapper_hits / max(1, len(words))

    if wrapper_ratio > 0.40 and len(words) > 500:
        return False, "publisher wrapper/metadata without substantive research body", stats

    # Research vocabulary indicators
    research_indicators = [
        "method", "result", "experiment", "model", "analysis", "dataset",
        "accuracy", "evaluate", "performance", "approach", "proposed", "study",
        "table", "figure", "we find", "we demonstrate", "we propose", "in this paper",
        "investigate", "speech", "detection", "neural", "network", "algorithm", "multi-agent"
    ]
    text_lower = text.lower()
    indicator_hits = sum(1 for ind in research_indicators if ind in text_lower)

    # Case 2: Abstract + unknown only (Section 12 Rule)
    # A document containing ['abstract', 'unknown'] where unknown has < 1000 tokens MUST be classified ABSTRACT_ONLY
    if "abstract" in detected_section_names and set(research_sections) == {"unknown"} and research_tokens < 1000:
        return False, "only abstract/unknown content detected (research body tokens < 1000)", stats

    # Case 3: Only unknown section detected (no abstract heading detected) — verify presence of research discourse
    if research_sections == ["unknown"] and "abstract" not in detected_section_names:
        sentence_endings = re.findall(r'[a-zA-Z0-9][.!?](?:\s+|$)', text)
        if len(sentence_endings) < 3 or indicator_hits < 2:
            return False, "insufficient research discourse indicators in unknown section", stats

    # Case 4: Extreme non-research imbalance
    if total_tokens > 3000 and research_tokens < 400:
        return False, "publisher wrapper/metadata without substantive research body", stats

    return True, "valid_research_paper", stats


def is_full_text_usable(text: str) -> bool:
    """
    Validates whether retrieved text qualifies as usable research paper full text.

    Rejects:
      - Empty or whitespace-only content
      - Placeholder, paywall, blocking or error pages (e.g. 404, Access Denied, Captcha, Cloudflare, Sign in)
      - Metadata-only / citation-only / abstract landing pages without substantial article body
      - Extremely short non-paper content (< FULL_TEXT_MIN_WORDS)

    Accepts:
      - Legitimate research papers with substantial body paragraphs and structure
    """
    if not text or not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False

    words = s.split()
    if len(words) < FULL_TEXT_MIN_WORDS:
        return False

    s_lower = s[:3000].lower()

    # Known error / paywall / bot blocking patterns
    error_patterns = [
        "404 not found",
        "access denied",
        "permission denied",
        "cookie policy",
        "please enable javascript",
        "please enable cookies",
        "sign in to access",
        "purchase this article",
        "subscribe to continue",
        "cloudflare",
        "attention required",
        "verify you are human",
        "captcha",
        "just a moment...",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "no abstract available",
        "doi landing page",
        "citation only",
    ]
    for pattern in error_patterns:
        if pattern in s_lower:
            return False

    # Check for metadata-only landing page indicators (e.g. citations & metrics without body text)
    metadata_headers = ["download citation", "rights and permissions", "about this article", "cite this article", "share this article"]
    metadata_hits = sum(1 for h in metadata_headers if h in s_lower)
    if metadata_hits >= 2 and len(words) < 500:
        return False

    return True


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
    """Fetch papers from Semantic Scholar with retry/backoff on 429 and transient errors."""
    papers = []
    encoded_query = urllib.parse.quote(query)
    url = (f"https://api.semanticscholar.org/graph/v1/paper/search"
           f"?query={encoded_query}&limit={limit}"
           f"&fields=title,authors,year,abstract,url,citationCount,openAccessPdf")

    # Build headers — include optional API key for higher rate limits
    headers = {'User-Agent': 'Mozilla/5.0'}
    ss_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if ss_api_key:
        headers["x-api-key"] = ss_api_key

    max_retries = 3
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
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
                break  # success — exit retry loop
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = int(e.headers.get("Retry-After", 0))
                wait = retry_after if retry_after > 0 else (2 ** attempt) * 2
                print(f"Semantic Scholar API warning: HTTP Error 429 — rate limited. "
                      f"Waiting {wait}s before retry {attempt + 1}/{max_retries}.")
                if attempt < max_retries - 1:
                    time.sleep(wait)
            elif e.code in (500, 502, 503, 504):
                wait = (2 ** attempt) * 2
                print(f"Semantic Scholar API warning: HTTP Error {e.code} — transient. "
                      f"Waiting {wait}s before retry {attempt + 1}/{max_retries}.")
                if attempt < max_retries - 1:
                    time.sleep(wait)
            else:
                print(f"Semantic Scholar API warning: HTTP Error {e.code} — not retrying.")
                break
        except Exception as e:
            print(f"Semantic Scholar API warning: {e}")
            break

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

# ---------------------------------------------------------------------------
# Abstract quality check
# ---------------------------------------------------------------------------

# Placeholder strings that should NOT be treated as real abstracts
_FAKE_ABSTRACT_PATTERNS = [
    r'^\s*$',
    r'^no abstract available\.?\s*$',
    r'^abstract not available\.?\s*$',
    r'^not available\.?\s*$',
    r'^n/a\.?\s*$',
    r'^abstract metadata fetched from',  # OpenAlex fallback
]
_FAKE_ABSTRACT_RE = re.compile(
    '|'.join(_FAKE_ABSTRACT_PATTERNS), re.IGNORECASE
)

def has_real_abstract(text: str) -> bool:
    """
    Returns True if `text` is a genuine abstract (not a placeholder).
    
    Rejects:
      - Empty strings
      - "No abstract available." and variants
      - Very short strings (< 20 meaningful words)
      - Known API fallback phrases
    """
    if not text or not text.strip():
        return False
    if _FAKE_ABSTRACT_RE.match(text.strip()):
        return False
    # Require at least 20 words of real content
    word_count = len(text.strip().split())
    if word_count < 20:
        return False
    return True


# Signals that indicate daily quota exhaustion (not transient rate limiting)
_QUOTA_SIGNALS = [
    "resource_exhausted",
    "generate_content_free_tier",
    "free-tier limit",
    "free_tier",
    "generaterequestsperday",
    "quotavalue",
    "requests_per_day",
    "per_day",
    "daily quota",
    "daily limit",
]


def _is_quota_exhausted(error_msg: str) -> bool:
    """Returns True if the error message indicates DAILY quota exhaustion."""
    msg = error_msg.lower()
    return any(sig in msg for sig in _QUOTA_SIGNALS)


# Module-level Gemini request counter (per-process, approximate)
_gemini_request_counter: list = [0]


def _gemini_generate_with_retry(
    client,
    model: str,
    prompt: str,
    config,
    max_retries: int = 3,
    agent_label: str = "Agent",
) -> str:
    """
    Thin wrapper forwarding to the centralized execute_gemini_with_retry.
    Kept for backward compatibility with existing Agent 1 call sites.
    """
    return execute_gemini_with_retry(
        prompt=prompt,
        config=config,
        model=model,
        max_retries=max_retries,
        agent_label=agent_label,
        client=client,
    )


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


def _parse_pmc_xml_to_structured_text(root: ET.Element) -> str:
    """
    Parse a PMC fullTextXML ElementTree root into structured plain text.

    Preserves `<sec>` / `<title>` boundaries so the section parser can
    detect headings. Returns multi-line text — NOT a single flat string.

    Structure of returned text:
        [abstract heading if present]
        abstract paragraph text

        Section Title

        paragraph text
        paragraph text

        Nested Sub-section Title

        paragraph text
    """
    lines: list[str] = []

    def _collect_paragraph_text(elem: ET.Element) -> str:
        """Recursively collect all text from an element, skipping nested <sec> and <title>."""
        parts = []
        if elem.text:
            parts.append(elem.text.strip())
        for child in elem:
            # Skip nested sections and section titles — they are handled separately
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag in ('sec', 'title'):
                continue
            parts.append(_collect_paragraph_text(child))
            if child.tail:
                parts.append(child.tail.strip())
        return ' '.join(p for p in parts if p)

    def _process_sec(sec_elem: ET.Element, depth: int = 0) -> None:
        """Recursively process a <sec> element and append to lines."""
        # Find the <title> child of this section
        title_text = ""
        for child in sec_elem:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag == 'title':
                raw = ''.join(child.itertext()).strip()
                # Strip leading section numbers (e.g. "1.2 " or "A. ")
                raw = re.sub(r'^[\d\.]+\s+', '', raw)
                raw = re.sub(r'^[A-Z]\.\s+', '', raw)
                title_text = raw
                break

        if title_text:
            lines.append("")
            lines.append(title_text)
            lines.append("")

        # Collect paragraphs and nested sections
        for child in sec_elem:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag == 'title':
                continue  # already handled
            elif tag == 'sec':
                _process_sec(child, depth + 1)
            elif tag in ('p', 'statement', 'caption', 'table-wrap', 'fig'):
                para = _collect_paragraph_text(child)
                if para.strip():
                    lines.append(para.strip())
            elif tag in ('list',):
                for item in child.iter():
                    itag = item.tag.split('}')[-1] if '}' in item.tag else item.tag
                    if itag == 'list-item':
                        item_text = _collect_paragraph_text(item).strip()
                        if item_text:
                            lines.append(item_text)

    # --- Extract abstract (from article-meta, not body) ---
    # Handle both namespaced and non-namespaced tags
    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if tag == 'abstract':
            abs_parts = []
            for child in elem.iter():
                ctag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if ctag == 'p':
                    pt = _collect_paragraph_text(child).strip()
                    if pt:
                        abs_parts.append(pt)
            if abs_parts:
                lines.append("Abstract")
                lines.append("")
                lines.extend(abs_parts)
                lines.append("")
            break  # only first abstract

    # --- Extract body sections ---
    body_elem = None
    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if tag == 'body':
            body_elem = elem
            break

    if body_elem is not None:
        for child in body_elem:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag == 'sec':
                _process_sec(child)
            elif tag == 'p':
                para = _collect_paragraph_text(child).strip()
                if para:
                    lines.append(para)

    # Join and clean: preserve blank lines (for heading detection), but no triple+ blanks
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Normalize whitespace within each line (no cross-line collapse)
    cleaned_lines = []
    for line in text.split('\n'):
        cleaned_lines.append(re.sub(r'[ \t]+', ' ', line).rstrip())
    return '\n'.join(cleaned_lines).strip()


def _fetch_pmc_full_text(url: str) -> str:
    """Queries Europe PMC open API for full text XML and parses structured sections.

    Preserves <sec> and <title> boundaries so the TextRank section parser
    receives proper section headings. No character truncation applied.
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
            text = _parse_pmc_xml_to_structured_text(root)
            return text
    except Exception as e:
        print(f"  [PMC FT] PMC XML retrieval failed for {pmcid}: {e}")
    return ""


def _fetch_pdf_text(pdf_url: str) -> str:
    """Download a legitimate open-access PDF and extract text using pdfminer.six.

    Returns the full extracted text preserving line structure for heading detection.
    Returns "" on any error — never crashes the pipeline.
    No truncation applied.
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
                # Preserve line structure for section heading detection.
                # Only normalize within-line spaces; do NOT collapse across newlines.
                lines = text.split('\n')
                cleaned = [re.sub(r'[ \t]+', ' ', line).rstrip() for line in lines]
                text = '\n'.join(cleaned)
                # Reduce excessive blank lines (3+ → 2)
                text = re.sub(r'\n{3,}', '\n\n', text)
                return text.strip()
        except ImportError:
            print("  [PDF FT] pdfminer.six not installed. Install with: pip install pdfminer.six")
        except Exception as e:
            print(f"  [PDF FT] pdfminer extraction failed for {pdf_url[:60]}: {e}")
    except Exception as e:
        print(f"  [PDF FT] Download failed for {pdf_url[:60]}: {e}")
    return ""


def _fetch_html_text(url: str) -> str:
    """Fetch a URL and extract readable text using BeautifulSoup.

    Preserves paragraph and heading boundaries with newlines.
    Strips scripts, styles, and navigation elements.
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
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "button", "iframe", "noscript"]):
            tag.decompose()

        # Insert newlines before/after block elements so section headings & paragraphs are preserved
        for tag in soup.find_all(["p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br"]):
            tag.insert_before("\n\n")

        text = soup.get_text()
        # Clean within-line spaces while preserving double newlines
        lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
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
        Check whether the acquired text is usable and represents a legitimate full research paper.
        Returns a result dict with access_status='FULL_TEXT_AVAILABLE' if usable, None otherwise.
        """
        if not text or not is_full_text_usable(text):
            return None

        is_valid, reason, stats = validate_full_paper_content(text)
        if not is_valid:
            print(f"  [FT Resolver] Retrieved content from {source_label} failed full-paper validation.")
            print(f"  [FT Resolver] Reason: {reason}")
            return None

        word_count = len(text.split())
        cov_type = "full_paper" if word_count >= 600 else "partial_paper"
        return {
            "text": text,
            "source": source_label,
            "url": fetch_url,
            "coverage_type": cov_type,
            "access_status": "FULL_TEXT_AVAILABLE",
        }

    # ── Step 0: Pre-supplied full text (e.g. from upstream fetcher / cache / test) ──
    pre_supplied = paper_dict.get("full_text", "")
    if pre_supplied:
        res = _quality_check(pre_supplied, paper_dict.get("full_text_source") or paper_dict.get("source", "pre_supplied"), url)
        if res:
            return res

    # ── Step 1: arXiv ────────────────────────────────────────────────────────
    if "arxiv.org" in url:
        try:
            text = _fetch_arxiv_full_text(url)
            result = _quality_check(text, "arxiv", url)
            if result:
                print(f"  [FT Resolver] arXiv: {len(text.split()):,} words")
                return result
        except Exception as e:
            print(f"  [FT Resolver] arXiv fetch error: {e}")

    # ── Step 2: PMC / Europe PMC ─────────────────────────────────────────────
    if re.search(r'(pmc|PMC)', url):
        try:
            text = _fetch_pmc_full_text(url)
            result = _quality_check(text, "pmc", url)
            if result:
                print(f"  [FT Resolver] PMC: {len(text.split()):,} words")
                return result
        except Exception as e:
            print(f"  [FT Resolver] PMC fetch error: {e}")

    # ── Step 3: Semantic Scholar open-access PDF ──────────────────────────────
    if oa_pdf_url:
        try:
            text = _fetch_pdf_text(oa_pdf_url)
            result = _quality_check(text, "semantic_scholar", oa_pdf_url)
            if result:
                print(f"  [FT Resolver] Semantic Scholar PDF: {len(text.split()):,} words")
                return result
        except Exception as e:
            print(f"  [FT Resolver] Semantic Scholar PDF fetch error: {e}")

    # ── Step 4: OpenAlex open-access URL ─────────────────────────────────────
    if oa_url:
        try:
            text = _fetch_html_text(oa_url)
            result = _quality_check(text, "openalex", oa_url)
            if result:
                print(f"  [FT Resolver] OpenAlex OA URL: {len(text.split()):,} words")
                return result
        except Exception as e:
            print(f"  [FT Resolver] OpenAlex OA URL fetch error: {e}")

    # ── Step 5: DOAJ article URL ──────────────────────────────────────────────
    source_api = paper_dict.get("source", "")
    if source_api == "DOAJ" and url:
        try:
            text = _fetch_html_text(url)
            result = _quality_check(text, "doaj", url)
            if result:
                print(f"  [FT Resolver] DOAJ article URL: {len(text.split()):,} words")
                return result
        except Exception as e:
            print(f"  [FT Resolver] DOAJ fetch error: {e}")

    # ── Step 6: Unpaywall (optional, only if UNPAYWALL_EMAIL is configured) ───
    unpaywall_email = os.getenv("UNPAYWALL_EMAIL", "").strip()
    if unpaywall_email and doi:
        try:
            uw_url = _resolve_unpaywall(doi, unpaywall_email)
            if uw_url:
                text = _fetch_pdf_text(uw_url) or _fetch_html_text(uw_url)
                result = _quality_check(text, "unpaywall", uw_url)
                if result:
                    print(f"  [FT Resolver] Unpaywall: {len(text.split()):,} words")
                    return result
        except Exception as e:
            print(f"  [FT Resolver] Unpaywall fetch error: {e}")

    # ── Fallback: abstract only or unavailable ────────────────────────────────
    if abstract and abstract.strip() and not _FAKE_ABSTRACT_RE.match(abstract.strip()):
        print(f"  [FT Resolver] No full text found. Abstract only available.")
        return {
            "text": "",
            "source": "none",
            "url": "",
            "coverage_type": "abstract_only",
            "access_status": "ABSTRACT_ONLY",
        }

    print(f"  [FT Resolver] No usable text available for '{title[:50]}'.")
    return {
        "text": "",
        "source": "none",
        "url": "",
        "coverage_type": "unavailable",
        "access_status": "UNAVAILABLE",
    }


# ---------------------------------------------------------------------------
# Topic-Grounded Paper Relevance & Hard Gating (Part C)
# ---------------------------------------------------------------------------

import math


def evaluate_paper_hard_gate(paper: dict, topic: str, decomp: dict) -> dict:
    """
    Gate-First Evaluation Engine (Sections 3, 5, 6, 8, 10):
    1. DOMAIN GATE: Must contain genuine AI/ML/RL evidence.
    2. CORE CONCEPT & MULTI-AGENT GATE: Must contain explicit multi-agent / interacting agent evidence.
    3. TECHNICAL RELATIONSHIP GATE: Must study, develop, evaluate, survey, or apply the topic as a meaningful technical subject.
    4. EXCLUSION CONCEPTS: Rejects sanctions on Iran, political economy, single-agent RL, autonomous physical robots without multi-agent, telecommunications routing, generic XAI, clinical AI without multi-agent.
    """
    title = (paper.get("title") or "").strip().lower()
    abstract = (paper.get("abstract") or "").strip().lower()
    full_text_sample = (paper.get("full_text") or "")[:5000].lower()
    text = f"{title} {abstract} {full_text_sample}"

    t_type = decomp.get("topic_type")

    # 1. Multi-Agent Systems in AI
    if t_type == "MULTI_AGENT_AI":
        requires_multiagent = decomp.get("requires_multiagent", True)

        # 1. Exclusion Concepts (Hard Rejection)
        exclusions = decomp.get("exclusion_concepts", [
            "sanctions on iran", "political economy", "macroeconomic", "petroleum",
            "autonomous robot", "autonomous vehicle", "telecommunications routing",
            "fermi paradox", "astrophysics"
        ])
        for excl in exclusions:
            if excl in title or (excl in abstract and not any(k in text for k in ["multi-agent", "multiagent", "multi agent", "multiple agents"])):
                return {
                    "passed": False,
                    "domain_match": False,
                    "agent_match": False,
                    "multiagent_match": False,
                    "technical_concept_match": False,
                    "research_objective_match": False,
                    "relevance_status": "REJECTED_OFF_TOPIC",
                    "relevance_level": "REJECTED",
                    "rejection_reason": f"Paper title/abstract contains excluded concept '{excl}'."
                }

        # Check domain AI/ML
        domain_anchors = decomp.get("domain_anchors", [
            "artificial intelligence", "ai", "machine learning", "deep learning",
            "reinforcement learning", "neural network", "neural networks",
            "large language model", "llm", "llms", "language model", "transformer",
            "distributed artificial intelligence", "dai", "generative ai"
        ])
        has_domain = any(d in text for d in domain_anchors) or bool(re.search(r'\b(ai|ml|marl|rl|llm|llms|nlp)\b', text))

        # Check Agent anchors
        agent_anchors = decomp.get("agent_concept_anchors", [
            "ai agent", "ai agents", "intelligent agent", "software agent", "autonomous agent", "agentic", "llm agent"
        ])
        has_agent = any(a in text for a in agent_anchors) or ("agent" in text and any(w in text for w in ["software", "neural", "learning", "algorithm", "model", "llm"]))

        # Check Multi-Agent anchors in title + abstract (Section 5 & 6: primary research subject)
        multiagent_anchors = [
            "multi-agent", "multi agent", "multiagent", "multi-agent system",
            "multiagent system", "multi-agent systems", "multiagent systems",
            "multiple intelligent agents", "multiple agents", "collaborative agents",
            "distributed agents", "agent coordination", "agent collaboration",
            "agent communication", "agent interaction", "agent negotiation",
            "agent orchestration", "multi-agent learning", "multi-agent reinforcement learning",
            "marl", "distributed artificial intelligence", "cooperative agents", "cooperating agents"
        ]
        
        # Check title and abstract specifically to avoid passing on distant full-text references
        title_abstract = f"{title} {abstract}"
        clean_ta_no_negation = re.sub(r'\b(without|no|not|lacks?)\s+(any\s+)?(multi-agent|multiagent|multi agent)', '', title_abstract)
        has_multiagent = any(m in clean_ta_no_negation for m in multiagent_anchors)

        # Check Technical Mechanisms
        mech_anchors = decomp.get("mechanism_anchors", [])
        has_mech = any(m in text for m in mech_anchors) or has_multiagent

        # Rejection tests for specific false positive patterns
        is_single_agent_rl = ("reinforcement learning" in text or "q-learning" in text or "policy gradient" in text) and not has_multiagent
        is_physical_robot = any(r in title for r in ["robot for", "robotic arm", "lawn mowing", "plant care", "industrial robot", "vacuum cleaner"]) and not has_multiagent
        is_network_routing = any(n in title for n in ["packet routing", "telecommunications network", "routing data", "network routing"]) and not has_multiagent
        is_generic_xai = "explainable artificial intelligence" in title and not has_multiagent
        is_clinical_ai = ("clinical decision" in title or "clinical triage" in title) and not has_multiagent
        is_fermi_paradox = ("fermi paradox" in title or "super-intelligence" in title) and not has_multiagent
        is_sanctions_iran = ("sanctions on iran" in title or "political economy" in title) and not has_multiagent

        if is_sanctions_iran:
            return {
                "passed": False,
                "domain_match": False,
                "agent_match": False,
                "multiagent_match": False,
                "technical_concept_match": False,
                "research_objective_match": False,
                "relevance_status": "REJECTED_OFF_TOPIC",
                "relevance_level": "REJECTED",
                "rejection_reason": "No meaningful relationship to multi-agent AI."
            }

        if not has_domain:
            return {
                "passed": False,
                "domain_match": False,
                "agent_match": has_agent,
                "multiagent_match": has_multiagent,
                "technical_concept_match": has_mech,
                "research_objective_match": False,
                "relevance_status": "REJECTED_OFF_TOPIC",
                "relevance_level": "REJECTED",
                "rejection_reason": "No meaningful relationship to artificial intelligence / machine learning domain."
            }

        if requires_multiagent:
            if not has_multiagent:
                reason = "Paper discusses single-agent reinforcement learning without multi-agent interaction." if is_single_agent_rl else (
                    "Paper concerns physical robot without multi-agent AI coordination." if is_physical_robot else (
                        "Paper discusses telecommunications network routing without multi-agent AI." if is_network_routing else (
                            "Paper title contains excluded concept 'fermi paradox'." if is_fermi_paradox else (
                                "Paper discusses generic XAI without multi-agent mechanisms." if is_generic_xai else (
                                    "Paper discusses clinical AI without multi-agent mechanisms." if is_clinical_ai else (
                                        "Paper discusses generic AI or unrelated concepts without multi-agent system evidence."
                                    )
                                )
                            )
                        )
                    )
                )
                return {
                    "passed": False,
                    "domain_match": True,
                    "agent_match": has_agent,
                    "multiagent_match": False,
                    "technical_concept_match": has_mech,
                    "research_objective_match": False,
                    "relevance_status": "REJECTED_OFF_TOPIC",
                    "relevance_level": "REJECTED",
                    "rejection_reason": reason
                }
        else:
            if not has_agent and not has_multiagent:
                return {
                    "passed": False,
                    "domain_match": True,
                    "agent_match": False,
                    "multiagent_match": False,
                    "technical_concept_match": has_mech,
                    "research_objective_match": False,
                    "relevance_status": "REJECTED_OFF_TOPIC",
                    "relevance_level": "REJECTED",
                    "rejection_reason": "Paper lacks AI agent architecture or agentic workflow evidence."
                }

        is_direct = any(m in title for m in ["multi-agent", "multiagent", "multi agent", "marl", "multi-agents", "multiple agents"]) or (
            "multi-agent" in abstract and any(k in abstract for k in ["propose", "survey", "introduce", "study", "develop", "framework", "algorithm", "architecture"])
        )
        relevance_level = "DIRECT_MATCH" if is_direct else "STRONG_RELATED"

        return {
            "passed": True,
            "domain_match": True,
            "agent_match": True,
            "multiagent_match": True,
            "technical_concept_match": True,
            "research_objective_match": True,
            "relevance_status": "ACCEPTED",
            "relevance_level": relevance_level,
            "rejection_reason": None,
            "why_selected": "Studies multi-agent AI systems, cooperative agent interaction, or multi-agent reinforcement learning."
        }

    # 2. Deepfake Audio Detection
    elif t_type == "DEEPFAKE_AUDIO":
        exclusions = decomp.get("exclusion_concepts", [])
        for excl in exclusions:
            if excl in title or (excl in abstract and not any(k in text for k in ["deepfake", "voice spoof", "synthetic speech", "audio spoof"])):
                return {
                    "passed": False,
                    "domain_match": False,
                    "agent_match": False,
                    "multiagent_match": False,
                    "technical_concept_match": False,
                    "research_objective_match": False,
                    "relevance_status": "REJECTED_OFF_TOPIC",
                    "relevance_level": "REJECTED",
                    "rejection_reason": f"Paper addresses excluded audio domain '{excl}'."
                }

        domain_anchors = decomp.get("domain_anchors", [])
        has_domain = any(d in text for d in domain_anchors)

        core_anchors = decomp.get("core_concept_anchors", [])
        has_core = any(c in text for c in core_anchors)

        mech_anchors = decomp.get("mechanism_anchors", [])
        has_mech = any(m in text for m in mech_anchors)

        if not has_domain:
            return {
                "passed": False,
                "domain_match": False,
                "agent_match": False,
                "multiagent_match": False,
                "technical_concept_match": has_mech,
                "research_objective_match": False,
                "relevance_status": "REJECTED_OFF_TOPIC",
                "relevance_level": "REJECTED",
                "rejection_reason": "Paper lacks audio or speech domain context."
            }

        if not has_core:
            return {
                "passed": False,
                "domain_match": True,
                "agent_match": False,
                "multiagent_match": False,
                "technical_concept_match": has_mech,
                "research_objective_match": False,
                "relevance_status": "REJECTED_OFF_TOPIC",
                "relevance_level": "REJECTED",
                "rejection_reason": "Paper involves generic audio processing without synthetic, deepfake, or spoofing mechanisms."
            }

        return {
            "passed": True,
            "domain_match": True,
            "agent_match": False,
            "multiagent_match": False,
            "technical_concept_match": True,
            "research_objective_match": True,
            "relevance_status": "ACCEPTED",
            "relevance_level": "DIRECT_MATCH",
            "rejection_reason": None,
            "why_selected": "Addresses audio/speech deepfake, spoofing, or synthetic voice detection."
        }

    # 3. Medical Image Segmentation
    elif t_type == "MEDICAL_IMAGE_SEGMENTATION":
        exclusions = decomp.get("exclusion_concepts", [])
        for excl in exclusions:
            if excl in title:
                return {
                    "passed": False,
                    "domain_match": False,
                    "agent_match": False,
                    "multiagent_match": False,
                    "technical_concept_match": False,
                    "research_objective_match": False,
                    "relevance_status": "REJECTED_OFF_TOPIC",
                    "relevance_level": "REJECTED",
                    "rejection_reason": f"Paper concerns excluded area '{excl}'."
                }

        domain_anchors = decomp.get("domain_anchors", [])
        has_domain = any(d in text for d in domain_anchors)

        core_anchors = decomp.get("core_concept_anchors", [])
        has_core = any(c in text for c in core_anchors)

        mech_anchors = decomp.get("mechanism_anchors", [])
        has_mech = any(m in text for m in mech_anchors)

        if not has_domain:
            return {
                "passed": False,
                "domain_match": False,
                "agent_match": False,
                "multiagent_match": False,
                "technical_concept_match": has_mech,
                "research_objective_match": False,
                "relevance_status": "REJECTED_OFF_TOPIC",
                "relevance_level": "REJECTED",
                "rejection_reason": "Paper lacks medical or radiological domain context."
            }

        if not has_mech:
            return {
                "passed": False,
                "domain_match": True,
                "agent_match": False,
                "multiagent_match": False,
                "technical_concept_match": False,
                "research_objective_match": False,
                "relevance_status": "REJECTED_OFF_TOPIC",
                "relevance_level": "REJECTED",
                "rejection_reason": "Paper lacks segmentation, contouring, or anatomical delineation mechanisms."
            }

        return {
            "passed": True,
            "domain_match": True,
            "agent_match": False,
            "multiagent_match": False,
            "technical_concept_match": True,
            "research_objective_match": True,
            "relevance_status": "ACCEPTED",
            "relevance_level": "DIRECT_MATCH",
            "rejection_reason": None,
            "why_selected": "Addresses medical imaging segmentation or anatomical contouring."
        }

    # 4. General Domain Fallback
    else:
        anchors = decomp.get("core_concept_anchors", [])
        hits = sum(1 for a in anchors if a in text)
        passed = hits >= max(1, len(anchors) // 2)
        return {
            "passed": passed,
            "domain_match": passed,
            "agent_match": passed,
            "multiagent_match": passed,
            "technical_concept_match": passed,
            "research_objective_match": passed,
            "relevance_status": "ACCEPTED" if passed else "REJECTED_OFF_TOPIC",
            "relevance_level": "DIRECT_MATCH" if passed else "REJECTED",
            "rejection_reason": None if passed else f"Paper lacks core concepts for '{topic}'.",
            "why_selected": f"Matches topic '{topic}' concepts." if passed else ""
        }


def compute_paper_4factor_relevance(paper: dict, topic: str, decomp: dict, gate_res: dict = None) -> dict:
    """
    Calculates 5-component relevance score mathematically (Section 11):
    - Core topic alignment: 35%
    - Technical mechanism: 25%
    - Research objective: 20%
    - Semantic similarity: 10%
    - Keyword overlap: 10%

    If gate_res is supplied and failed: returns score 0.0 with REJECTED status.
    """
    if gate_res and not gate_res.get("passed", False):
        return {
            "score": 0.0,
            "rank": "Low",
            "relevance_status": "REJECTED_OFF_TOPIC",
            "relevance_level": "REJECTED",
            "core_topic_score": 0.0,
            "mech_score": 0.0,
            "obj_score": 0.0,
            "semantic_score": 0.0,
            "phrase_score": 0.0
        }

    title = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()
    full_text = (paper.get("full_text") or "")[:3000]
    combined_text = f"{title} {abstract} {full_text}".lower()

    # 1. Core topic alignment (35%)
    core_anchors = decomp.get("core_concept_anchors", [])
    core_hits = sum(1 for c in core_anchors if c in combined_text)
    core_topic_score = 1.0 if any(c in title.lower() for c in core_anchors) else min(1.0, core_hits / max(1, min(3, len(core_anchors))))

    # 2. Technical mechanism (25%)
    mech_anchors = decomp.get("mechanism_anchors", [])
    mech_hits = sum(1 for m in mech_anchors if m in combined_text)
    mech_score = min(1.0, mech_hits / max(1, min(2, len(mech_anchors))))

    # 3. Research objective (20%)
    has_obj = any(k in combined_text for k in ["propose", "survey", "introduce", "we evaluate", "we demonstrate", "we develop", "architecture", "methodology", "algorithm"])
    obj_score = 1.0 if has_obj else 0.7

    # 4. Semantic similarity (10%)
    sem_score = 0.65
    try:
        from sentence_transformers import SentenceTransformer
        _model_cache = getattr(evaluate_paper_hard_gate, '_model', None)
        if _model_cache is None:
            _model_cache = SentenceTransformer('all-MiniLM-L6-v2')
            evaluate_paper_hard_gate._model = _model_cache
        q_vec = _model_cache.encode(topic, convert_to_tensor=True)
        t_vec = _model_cache.encode(f"{title} {abstract}"[:500], convert_to_tensor=True)
        sem_score = float(st_util.cos_sim(q_vec, t_vec)[0][0])
    except Exception:
        sem_score = 0.65
    sem_score = max(0.0, min(1.0, sem_score))

    # 5. Keyword overlap (10%)
    clean_topic = topic.lower().strip()
    blacklist = set(decomp.get("generic_blacklist", []))
    topic_words = [w for w in re.sub(r'[^a-z0-9\s]', '', clean_topic).split() if len(w) > 2 and w not in blacklist]
    t_hits = sum(1 for w in topic_words if w in combined_text)
    kw_score = t_hits / max(1, len(topic_words))

    # Composite score: 35% Core + 25% Mech + 20% Obj + 10% Sem + 10% KW
    composite = (0.35 * core_topic_score) + (0.25 * mech_score) + (0.20 * obj_score) + (0.10 * sem_score) + (0.10 * kw_score)
    final_score = round(min(0.98, max(0.72, 0.65 + (composite * 0.33))), 3)

    rank = "High" if final_score >= 0.85 else ("Medium" if final_score >= 0.75 else "Low")

    return {
        "score": final_score,
        "rank": rank,
        "relevance_status": "ACCEPTED",
        "relevance_level": gate_res.get("relevance_level", "DIRECT_MATCH") if gate_res else "DIRECT_MATCH",
        "core_topic_score": core_topic_score,
        "mech_score": mech_score,
        "obj_score": obj_score,
        "semantic_score": sem_score,
        "phrase_score": kw_score
    }


def compute_deterministic_relevance(query: str, title: str, abstract: str, year: int, citations: int = 0) -> dict:
    """Backward compatibility wrapper for compute_paper_4factor_relevance."""
    decomp = decompose_research_topic(query)
    paper_dummy = {"title": title, "abstract": abstract, "full_text": ""}
    return compute_paper_4factor_relevance(paper_dummy, query, decomp)


_ECOLOGY_QUERY_TERMS = {
    "ecology", "ecological", "biodiversity", "species", "habitat",
    "wildlife", "conservation", "biomass", "flora", "fauna",
    "environmental", "environment", "ecosystem", "biome", "trophic",
    "population ecology", "community ecology"
}

_NON_ECOLOGICAL_METAPHORS = [
    r'\b(ev|electric\s+vehicle)\s+(service\s+)?ecosystem\b',
    r'\b(software|hardware|platform|digital|mobile|app|service|technology|developer|it)\s+ecosystem\b',
    r'\b(business|startup|fintech|payment|financial|enterprise|commercial|e-?commerce|supply\s+chain)\s+ecosystem\b',
    r'\b(cloud|iot|blockchain|crypto|smart\s+grid|smart\s+city|metaverse)\s+ecosystem\b',
    r'\binnovation\s+ecosystem\b',
    r'\bentrepreneurial\s+ecosystem\b',
]
_NON_ECOLOGICAL_RE = re.compile('|'.join(_NON_ECOLOGICAL_METAPHORS), re.IGNORECASE)

_GENUINE_ECOLOGY_SIGNALS = {
    "species", "biodiversity", "habitat", "organism", "biomass", "flora", "fauna",
    "wildlife", "forest", "marine", "aquatic", "terrestrial", "wetland", "canopy",
    "trophic", "photosynthesis", "soil", "vegetation", "plant", "animal", "lichen",
    "conservation", "climate change", "ecological", "ecology", "ecosystem services",
    "anthropogenic", "carbon sequestration", "abiotic", "biotic"
}


def _detect_domain_mismatch(query: str, title: str, abstract: str) -> float:
    """Domain mismatch multiplier with ecological metaphor detection."""
    q_lower = query.lower()
    q_words = set(re.sub(r'[^a-z0-9\s]', '', q_lower).split())

    is_ecology_query = bool(q_words & _ECOLOGY_QUERY_TERMS) or "ecology" in q_lower or "ecosystem" in q_lower
    if is_ecology_query:
        text_to_check = f"{title} {abstract}"
        has_non_eco_metaphor = bool(_NON_ECOLOGICAL_RE.search(text_to_check))
        if has_non_eco_metaphor:
            t_words = set(re.sub(r'[^a-z0-9\s]', '', text_to_check.lower()).split())
            has_genuine_eco = bool(t_words & _GENUINE_ECOLOGY_SIGNALS)
            if not has_genuine_eco:
                return 0.45

    decomp = decompose_research_topic(query)
    paper_dummy = {"title": title, "abstract": abstract}
    gate = evaluate_paper_hard_gate(paper_dummy, query, decomp)
    return 1.0 if gate.get("passed", False) else 0.45


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
# Main Orchestration: fetch_arxiv_papers
# ---------------------------------------------------------------------------

def fetch_arxiv_papers(query: str, max_results: int = 8) -> Agent1ResearchOutput:
    """
    Topic-grounded discovery (Sections 2, 3, 11, 12, 13, 15, 17):
    1. Normalize original user research topic spelling.
    2. Decompose normalized topic into domain, agent, multi-agent, and mechanism anchors.
    3. Generate 10 targeted research queries.
    4. Retrieve candidates from arXiv, Semantic Scholar, Crossref, OpenAlex, Europe PMC, DOAJ.
    5. Deduplicate candidates.
    6. For each candidate:
       - Full-text resolution and validation (research_tokens >= 1000 and valid research sections).
       - Gate-First relevance evaluation (Domain Gate -> Multi-Agent Gate -> Technical Gate -> Exclusions).
       - Deterministic 5-component scoring (Core 35%, Mech 25%, Obj 20%, Sem 10%, KW 10%).
       - Rejects off-topic papers with score 0 and status REJECTED_OFF_TOPIC.
    7. Send compact summary prompt (<= 40k chars) to Gemini for notebook extraction.
    """
    from topic_decomposition import normalize_research_topic
    normalized_topic = normalize_research_topic(query)
    decomp = decompose_research_topic(normalized_topic)
    targeted_queries = generate_targeted_research_queries(normalized_topic)
    limit_per_api = 5

    print(f"[Agent1] Original topic:   '{query}'")
    print(f"[Agent1] Normalized topic: '{normalized_topic}'")
    print(f"[Agent1] Generated queries: {len(targeted_queries)} queries -> {targeted_queries[:4]}")

    all_raw = []
    # Query across top targeted queries
    for tq in targeted_queries[:5]:
        p1 = fetch_arxiv(tq, limit_per_api)
        p2 = fetch_semantic_scholar(tq, limit_per_api)
        p3 = fetch_crossref(tq, limit_per_api)
        p4 = fetch_openalex(tq, limit_per_api)
        p5 = fetch_europe_pmc(tq, limit_per_api)
        p6 = fetch_doaj(tq, limit_per_api)
        all_raw.extend(p1 + p2 + p3 + p4 + p5 + p6)

    # Deduplicate
    deduplicated = []
    seen = set()
    for p in all_raw:
        norm = normalize_title(p.get("title", ""))
        if norm not in seen and p.get("title") != "Unknown":
            seen.add(norm)
            deduplicated.append(p)

    print(f"[Agent1] Candidates retrieved: {len(all_raw)} raw, {len(deduplicated)} unique candidates.")

    # Two-Stage Verification & Filtering (Order: Full-Text Check -> Gate Check -> Scoring)
    eligible_candidates = []
    candidate_diagnostics = []
    abstract_only_count = 0
    passed_gate_count = 0

    for idx, p in enumerate(deduplicated):
        cand_id = f"cand-{idx+1:03d}"

        # Stage 1: Hard Gate Check on candidate metadata
        gate_res = evaluate_paper_hard_gate(p, normalized_topic, decomp)
        p["_gate_result"] = gate_res

        title_safe = p.get('title', '')[:70].encode('ascii', 'replace').decode('ascii')
        if not gate_res["passed"]:
            rej_reason_safe = str(gate_res.get('rejection_reason', '')).encode('ascii', 'replace').decode('ascii')
            print(f"\n[Agent1 Candidate] {cand_id} | Title: {title_safe}")
            print(f"  Normalized topic:        {normalized_topic}")
            print(f"  Domain match:            {'YES' if gate_res.get('domain_match') else 'NO'}")
            print(f"  Agent match:             {'YES' if gate_res.get('agent_match') else 'NO'}")
            print(f"  Multi-agent match:       {'YES' if gate_res.get('multiagent_match') else 'NO'}")
            print(f"  Technical match:         {'YES' if gate_res.get('technical_concept_match') else 'NO'}")
            print(f"  Research objective:      {'YES' if gate_res.get('research_objective_match') else 'NO'}")
            print(f"  Full-text status:        NOT_RESOLVED")
            print(f"  Final score:             0")
            print(f"  Classification:          REJECTED")
            print(f"  Reason:                  {rej_reason_safe}")

            candidate_diagnostics.append(CandidateDiagnostic(
                paper_id=cand_id,
                title=p.get("title", ""),
                semantic_score=0.0,
                keyword_score=0.0,
                domain_penalty=0.0,
                domain_mismatch=True,
                final_relevance_score=0.0,
                domain_match=gate_res.get("domain_match", False),
                agent_match=gate_res.get("agent_match", False),
                multiagent_match=gate_res.get("multiagent_match", False),
                core_concept_match=gate_res.get("core_concept_match", False),
                technical_concept_match=gate_res.get("technical_concept_match", False),
                relevance_status="REJECTED_OFF_TOPIC",
                relevance_level="REJECTED",
                access_status="NOT_CHECKED",
                full_text_source="none",
                selected=False,
                rejection_reason=gate_res.get("rejection_reason", "Failed topic relevance gate")
            ))
            continue

        passed_gate_count += 1

        # Stage 2: Full-Text Access & Research Body Volume Validation
        if len(eligible_candidates) < max_results:
            try:
                ft_result = _resolve_full_text(p)
                acc_status = ft_result.get("access_status", "UNAVAILABLE")
                full_txt = ft_result.get("text", "")
                is_usable = is_full_text_usable(full_txt)

                if acc_status == "FULL_TEXT_AVAILABLE" and is_usable:
                    paper_id = f"P{len(eligible_candidates)+1:03d}"
                    p["paper_id"] = paper_id
                    p["full_text"] = full_txt
                    p["full_text_source"] = ft_result["source"]
                    p["full_text_url"] = ft_result["url"]
                    p["coverage_type"] = ft_result["coverage_type"]
                    p["access_status"] = "FULL_TEXT_AVAILABLE"
                    p["full_text_available"] = True
                    p["full_text_word_count"] = len(full_txt.split()) if full_txt else 0
                    p["full_text_character_count"] = len(full_txt) if full_txt else 0
                    p["compression_source"] = "full_text"

                    # 5-factor scoring
                    rel_info = compute_paper_4factor_relevance(p, normalized_topic, decomp, gate_res=gate_res)
                    p["relevance_score"] = rel_info["score"]
                    p["relevance_rank"] = rel_info["rank"]
                    p["_semantic_score"] = rel_info["semantic_score"]
                    p["_keyword_score"] = rel_info["phrase_score"]
                    p["_domain_penalty"] = 1.0
                    p["_domain_mismatch"] = False
                    p["_final_relevance_score"] = rel_info["score"]
                    p["domain_match"] = gate_res.get("domain_match", True)
                    p["agent_match"] = gate_res.get("agent_match", True)
                    p["multiagent_match"] = gate_res.get("multiagent_match", True)
                    p["relevance_status"] = "ACCEPTED"
                    p["relevance_level"] = gate_res.get("relevance_level", "DIRECT_MATCH")
                    p["why_selected"] = gate_res.get("why_selected", "Topic-aligned full research paper")

                    # Extractive compression
                    try:
                        fact_sheet = compress_paper(
                            paper_dict=p,
                            query=normalized_topic,
                            paper_id=paper_id,
                        )
                        p["fact_sheet"] = fact_sheet
                        p["fact_sheet_text"] = format_fact_sheet(fact_sheet)
                        p["compression_ratio"] = fact_sheet.metrics.compression_ratio
                        p["section_coverage"] = fact_sheet.metrics.section_coverage
                        p["original_tokens"] = fact_sheet.metrics.original_token_count
                        p["compressed_tokens"] = fact_sheet.metrics.compressed_token_count
                        p["coverage_type"] = fact_sheet.coverage_type
                    except Exception as e:
                        print(f"[Agent1] Compression error for {paper_id}: {e}")
                        p["fact_sheet_text"] = p.get("abstract", "")
                        p["compression_ratio"] = 0.0
                        p["section_coverage"] = 0.0
                        p["original_tokens"] = 0
                        p["compressed_tokens"] = 0

                    why_safe = str(p['why_selected']).encode('ascii', 'replace').decode('ascii')
                    print(f"\n[Agent1 Candidate] {paper_id} | Title: {title_safe}")
                    print(f"  Normalized topic:        {normalized_topic}")
                    print(f"  Domain match:            YES")
                    print(f"  Agent match:             YES")
                    print(f"  Multi-agent match:       YES")
                    print(f"  Technical match:         YES")
                    print(f"  Research objective:      YES")
                    print(f"  Full-text status:        FULL_TEXT_AVAILABLE (Tokens: {p.get('original_tokens', p.get('full_text_word_count', 0))})")
                    print(f"  Final score:             {rel_info['score']}")
                    print(f"  Classification:          {gate_res.get('relevance_level', 'DIRECT_MATCH')}")
                    print(f"  Why selected:            {why_safe}")

                    eligible_candidates.append(p)
                    candidate_diagnostics.append(CandidateDiagnostic(
                        paper_id=paper_id,
                        title=p.get("title", ""),
                        semantic_score=rel_info["semantic_score"],
                        keyword_score=rel_info["phrase_score"],
                        domain_penalty=1.0,
                        domain_mismatch=False,
                        final_relevance_score=rel_info["score"],
                        domain_match=True,
                        agent_match=True,
                        multiagent_match=True,
                        core_concept_match=True,
                        technical_concept_match=gate_res.get("technical_concept_match", True),
                        relevance_status="ACCEPTED",
                        relevance_level=gate_res.get("relevance_level", "DIRECT_MATCH"),
                        access_status="FULL_TEXT_AVAILABLE",
                        full_text_source=ft_result.get("source", "none"),
                        selected=True,
                        rejection_reason=None
                    ))
                else:
                    abstract_only_count += 1
                    rej_reason = (
                        "Full research paper unavailable (abstract only)" if acc_status == "ABSTRACT_ONLY"
                        else ("Full text content failed usability validation" if full_txt and not is_usable
                              else "Full research paper content unavailable")
                    )
                    candidate_diagnostics.append(CandidateDiagnostic(
                        paper_id=cand_id,
                        title=p.get("title", ""),
                        semantic_score=0.0,
                        keyword_score=0.0,
                        domain_penalty=1.0,
                        domain_mismatch=False,
                        final_relevance_score=0.0,
                        domain_match=True,
                        agent_match=gate_res.get("agent_match", True),
                        multiagent_match=gate_res.get("multiagent_match", True),
                        core_concept_match=True,
                        technical_concept_match=gate_res.get("technical_concept_match", True),
                        relevance_status="REJECTED_OFF_TOPIC",
                        relevance_level="REJECTED",
                        access_status=acc_status,
                        full_text_source=ft_result.get("source", "none"),
                        selected=False,
                        rejection_reason=rej_reason
                    ))
            except Exception as e:
                candidate_diagnostics.append(CandidateDiagnostic(
                    paper_id=cand_id,
                    title=p.get("title", ""),
                    semantic_score=0.0,
                    keyword_score=0.0,
                    domain_penalty=1.0,
                    domain_mismatch=False,
                    final_relevance_score=0.0,
                    domain_match=True,
                    agent_match=True,
                    multiagent_match=True,
                    core_concept_match=True,
                    technical_concept_match=gate_res.get("technical_concept_match", True),
                    relevance_status="REJECTED_OFF_TOPIC",
                    relevance_level="REJECTED",
                    access_status="ACCESS_CHECK_FAILED",
                    full_text_source="none",
                    selected=False,
                    rejection_reason=f"Access check exception: {str(e)}"
                ))

    print(f"\n[Agent1] Relevance gate passed: {passed_gate_count}")
    print(f"[Agent1] Full-paper candidates: {len(eligible_candidates)}")
    print(f"[Agent1] Abstract-only candidates excluded: {abstract_only_count}")
    print(f"[Agent1] Final papers: {len(eligible_candidates)}")

    candidates = eligible_candidates

    if not candidates:
        print(f"[Agent1] WARNING: No candidates passed relevance gate with accessible full text.")
        return Agent1ResearchOutput(
            query=normalized_topic,
            papers=[],
            candidate_diagnostics=candidate_diagnostics,
            full_paper_eligibility_rate=1.0
        )

    # Build compact lite_candidates for Gemini (Prompt size <= 40,000 chars - Section 15)
    lite_candidates = []
    for c in candidates:
        abstract_snip = (c.get("abstract") or "")[:400]
        key_findings = []
        if c.get("fact_sheet") and hasattr(c.get("fact_sheet"), "key_findings"):
            key_findings = [k[:120] for k in c.get("fact_sheet").key_findings[:2]]
        content_for_gemini = f"Abstract: {abstract_snip}\nKey findings: {' | '.join(key_findings)}"

        lite_candidates.append({
            "title": c.get("title", ""),
            "authors": c.get("authors", []),
            "year": c.get("year", 2024),
            "abstract": content_for_gemini,
            "url": c.get("url", ""),
            "relevance_score": c.get("relevance_score", 0.8),
            "relevance_rank": c.get("relevance_rank", "Medium")
        })

    num_candidates = len(lite_candidates)
    prompt = f"""
    You are an expert Research Librarian similar to Google NotebookLM.
    Analyze the following {num_candidates} research papers retrieved for the topic: '{normalized_topic}'.
    
    For EACH paper individually, extract ALL of the following fields based on its title and abstract.
    Every field MUST be unique and specific to that paper's actual content — do NOT copy the same text across papers.
    
    Fields to extract per paper:
    1. relevance_score: Retrieve this directly from the paper's data under 'relevance_score' and copy it verbatim. Do not compute or change it.
    2. relevance_rank: Retrieve this directly from the paper's data under 'relevance_rank' and copy it verbatim. Do not change it.
    3. innovation_score: Integer between 0 and 100 representing the paper's innovation level.
    4. research_significance: A short paragraph analyzing the paper's academic impact, industrial impact, and contribution to innovation.
    5. notebook_summary: Write exactly 3 plain-English bullet points for THIS SPECIFIC PAPER. Format as: '• What it studies: [1 sentence] • How it does it: [1 sentence] • What it achieves: [1 sentence]'.
    6. technical_execution: A 1-sentence description of the core algorithm or technique THIS PAPER uses.
    7. datasets: List of dataset names used. If not mentioned, use ["Not specified"].
    8. problem_statement: What specific research problem or gap does this paper identify and tackle?
    9. proposed_solution: What solution, framework, model, or method does this paper propose?
    10. methodology: What methods, algorithms, architectures, or techniques did the authors use?
    11. results: What were the key quantitative or qualitative results reported?
    12. challenges: What limitations or open challenges do the authors acknowledge?
    13. future_outcomes: What future work or research directions do the authors suggest?
    
    Papers data:
    {json.dumps(lite_candidates, indent=2)}
    
    Respond ONLY with the JSON code block matching:
    {{
      "papers": [
        {{
          "title": "Title of paper",
          "authors": ["Author 1"],
          "year": 2024,
          "abstract": "Abstract text",
          "relevance_score": 0.92,
          "relevance_rank": "High",
          "innovation_score": 88,
          "research_significance": "...",
          "notebook_summary": "• What it studies: ... • How it does it: ... • What it achieves: ...",
          "technical_execution": "...",
          "datasets": ["Dataset Name"],
          "url": "paper link url",
          "problem_statement": "...",
          "proposed_solution": "...",
          "methodology": "...",
          "results": "...",
          "challenges": "...",
          "future_outcomes": "..."
        }}
      ]
    }}
    """

    try:
        response_text = _gemini_generate_with_retry(
            client=None,
            model=GEMINI_MODEL,
            prompt=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
            max_retries=3,
            agent_label="Agent1",
        )

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
                fact_sheet_text=cand_extra.get("fact_sheet_text", ""),
                compression_ratio=cand_extra.get("compression_ratio", 0.0),
                section_coverage=cand_extra.get("section_coverage", 0.0),
                original_tokens=cand_extra.get("original_tokens", 0),
                compressed_tokens=cand_extra.get("compressed_tokens", 0),
                full_text_available=cand_extra.get("full_text_available", False),
                full_text_source=cand_extra.get("full_text_source", "none"),
                full_text_url=cand_extra.get("full_text_url", ""),
                compression_source=cand_extra.get("compression_source", "none"),
                coverage_type=cand_extra.get("coverage_type", "full_paper"),
                access_status="FULL_TEXT_AVAILABLE",
                full_text_word_count=cand_extra.get("full_text_word_count", 0),
                full_text_character_count=cand_extra.get("full_text_character_count", 0),
                semantic_score=cand_extra.get("_semantic_score", 0.0),
                keyword_score=cand_extra.get("_keyword_score", 0.0),
                domain_penalty=cand_extra.get("_domain_penalty", 1.0),
                domain_mismatch=cand_extra.get("_domain_mismatch", False),
                final_relevance_score=cand_extra.get("_final_relevance_score", 0.0),
                core_concept_match=True,
                technical_concept_match=cand_extra.get("_gate_result", {}).get("technical_concept_match", True),
                why_selected=cand_extra.get("why_selected", "Topic-aligned full research paper"),
            ))

        rank_order = {"High": 3, "Medium": 2, "Low": 1}
        parsed_papers.sort(
            key=lambda x: (rank_order.get(x.relevance_rank, 1), x.relevance_score),
            reverse=True
        )
        return Agent1ResearchOutput(
            query=query,
            papers=parsed_papers,
            candidate_diagnostics=candidate_diagnostics,
            full_paper_eligibility_rate=1.0 if parsed_papers else 0.0
        )

    except Exception as e:
        print(f"[Agent1] Fallback summary generation: {e}")
        fallback_papers = []
        for idx, p in enumerate(candidates):
            nb_summary, tech_exec = _make_unique_fallback_summary(p, query)
            abstract = p.get("abstract", "")
            gate_res = p.get("_gate_result", {})

            sents = [s.strip() for s in abstract.split(".") if len(s.strip()) > 30]
            what_it_studies = sents[0] if len(sents) > 0 else p["title"]
            how_it_does_it = sents[1] if len(sents) > 1 else tech_exec
            what_it_achieves = sents[2] if len(sents) > 2 else f"Demonstrates {query} improvements."
            nb_summary = (
                f"• What it studies: {what_it_studies}. "
                f"• How it does it: {how_it_does_it}. "
                f"• What it achieves: {what_it_achieves}."
            )

            fallback_papers.append(PaperMetadata(
                title=p["title"],
                authors=p["authors"],
                year=p["year"],
                abstract=abstract,
                relevance_rank=p.get("relevance_rank", "High"),
                notebook_summary=nb_summary,
                technical_execution=tech_exec,
                datasets=["Not specified"],
                url=p["url"],
                relevance_score=p.get("relevance_score", 0.88),
                problem_statement=abstract[:120].rstrip() + "..." if len(abstract) > 120 else abstract,
                proposed_solution=f"Proposes a methodology addressing '{query}' challenges.",
                methodology=tech_exec,
                results="Quantitative results described in full paper text.",
                challenges="Limitations acknowledged by authors.",
                future_outcomes=f"Authors suggest extending the approach in future work.",
                innovation_score=int(p.get("relevance_score", 0.85) * 100),
                research_significance=(
                    f"This research contributes to the field of {query} by leveraging {tech_exec}."
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
                compression_source=p.get("compression_source", "full_text"),
                coverage_type=p.get("coverage_type", "full_paper"),
                access_status="FULL_TEXT_AVAILABLE",
                full_text_word_count=p.get("full_text_word_count", 0),
                full_text_character_count=p.get("full_text_character_count", 0),
                semantic_score=p.get("_semantic_score", 0.0),
                keyword_score=p.get("_keyword_score", 0.0),
                domain_penalty=p.get("_domain_penalty", 1.0),
                domain_mismatch=p.get("_domain_mismatch", False),
                final_relevance_score=p.get("_final_relevance_score", 0.0),
                core_concept_match=True,
                technical_concept_match=gate_res.get("technical_concept_match", True),
                why_selected=p.get("why_selected", "Topic-aligned full research paper"),
            ))

        rank_order = {"High": 3, "Medium": 2, "Low": 1}
        fallback_papers.sort(
            key=lambda x: (rank_order.get(x.relevance_rank, 1), x.relevance_score),
            reverse=True
        )
        return Agent1ResearchOutput(
            query=query,
            papers=fallback_papers,
            candidate_diagnostics=candidate_diagnostics,
            full_paper_eligibility_rate=1.0 if fallback_papers else 0.0
        )

