import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import re
import time
import urllib.request
import urllib.parse
import hashlib
# pyrefly: ignore [missing-import]
from google import genai
from google.genai import types
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

from schemas import Agent1ResearchOutput, Agent2GapOutput, Agent3PatentOutput, PatentInfo  # type: ignore

# Load environment variables
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path=_env_path)

_gemini_client = None

def _get_client():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Check backend/.env file.")
        print(f"[Agent3] Gemini API key loaded: ...{api_key[-6:]}")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client

GEMINI_MODEL = "gemini-2.5-flash"

# Valid patent ID regex: e.g. US10928345B2, EP3456789A1, WO2009034499
_VALID_PATENT_RE = re.compile(r'^(US|EP|WO|CN|JP|DE|FR|GB|KR)\d{5,}([A-Z]\d*)?$', re.IGNORECASE)


# ---------------------------------------------------------------------------
# URL Builders
# ---------------------------------------------------------------------------
def _make_google_patents_url(patent_id: str, title: str = "") -> str:
    """Build a Google Patents URL. Uses ID if valid; falls back to title search."""
    clean_id = patent_id.replace(" ", "").replace("-", "").strip()
    if _VALID_PATENT_RE.match(clean_id):
        return f"https://patents.google.com/patent/{clean_id}/en"
    # Fallback: title-based search always resolves correctly (uses the root search URL)
    encoded_title = urllib.parse.quote(title[:80] if title else clean_id)
    return f"https://patents.google.com/?q={encoded_title}"


def get_patent_source_links(patent_id: str, title: str = "", url: str = "") -> dict:
    """Build platform-specific links for a patent, handling DOI URLs gracefully."""
    if url and "doi.org" in url:
        return {
            "Crossref (DOI)": url,
            "Google Patents Search": f"https://patents.google.com/?q={urllib.parse.quote(title)}"
        }
    clean_id = patent_id.replace(" ", "").replace("-", "").strip()
    google_url = _make_google_patents_url(clean_id, title)
    links = {"Google Patents": google_url}
    if _VALID_PATENT_RE.match(clean_id) and clean_id.upper().startswith("US"):
        links["USPTO Public Search"] = f"https://ppubs.uspto.gov/pubwebapp/external.html?q={clean_id}&db=USPAT"
        links["Lens.org"] = f"https://www.lens.org/lens/search/patent/list?q={urllib.parse.quote(clean_id)}"
    if _VALID_PATENT_RE.match(clean_id) and (clean_id.upper().startswith("EP") or clean_id.upper().startswith("WO")):
        links["Espacenet"] = f"https://worldwide.espacenet.com/patent/search?q=pn%3D{clean_id}"
    return links


# ---------------------------------------------------------------------------
# STEP 1a: Fetch real patents from Google Patents (free, no API key needed)
# ---------------------------------------------------------------------------
def _build_search_terms(topic: str) -> str:
    """Extract key technical terms from a topic for patent search."""
    stopwords = {"and", "for", "the", "with", "using", "of", "in", "on", "a", "an",
                 "via", "to", "from", "by", "at", "or", "as", "is", "are", "into",
                 "through", "towards", "approach", "novel", "new", "improved", "study",
                 "method", "system", "based", "deep", "learning", "machine"}
    words = [w for w in topic.split() if w.lower() not in stopwords and len(w) > 2]
    return " ".join(words[:5]) if words else topic[:60]


def _fetch_google_patents(query_topic: str) -> list:
    """
    Queries the PatentsView API (USPTO open data) for real granted US patents
    matching the research topic. Returns up to 5 real patent records.
    """
    search_terms = _build_search_terms(query_topic)
    patents = []

    # PatentsView API — free, no API key, returns real USPTO patent data
    try:
        payload = json.dumps({
            "q": {"_text_any": {"patent_abstract": search_terms}},
            "f": ["patent_number", "patent_title", "assignee_organization",
                  "patent_abstract", "patent_date"],
            "o": {"per_page": 5}
        }).encode()
        req = urllib.request.Request(
            "https://search.patentsview.org/api/v1/patent/",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "ResearchIQ/1.0"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode())
            for pat in data.get("patents", []) or []:
                num = (pat.get("patent_number") or "").strip()
                if not num:
                    continue
                pid = f"US{num}B2" if not num.upper().startswith("US") else num
                title = (pat.get("patent_title") or "Unknown Title").strip()
                assignee = ""
                assignees = pat.get("assignee_organization") or []
                if isinstance(assignees, list) and assignees:
                    assignee = assignees[0].get("assignee_organization", "") if isinstance(assignees[0], dict) else str(assignees[0])
                abstract = (pat.get("patent_abstract") or "")[:600]
                patents.append({
                    "patent_id": pid,
                    "title": title,
                    "assignee": assignee or "USPTO Patent Holder",
                    "abstract": abstract,
                    "source_paper": query_topic,
                    "url": f"https://patents.google.com/patent/{pid}/en",
                })
        print(f"  [PatentsView] Found {len(patents)} patents for '{search_terms[:40]}'")
    except Exception as e:
        print(f"  [PatentsView] Error: {e}")

    return patents


# ---------------------------------------------------------------------------
# STEP 1b: Fetch real patents from Lens.org (free open patent database)
# ---------------------------------------------------------------------------
def _fetch_lens_org_patents(query_topic: str) -> list:
    """
    Queries the Lens.org patent search API for real international patents
    (USPTO, EPO, WIPO). Returns up to 5 real patent records.
    """
    search_terms = _build_search_terms(query_topic)
    patents = []

    try:
        encoded_q = urllib.parse.quote(search_terms)
        url = (
            f"https://api.lens.org/patent/search?"
            f"q={encoded_q}&size=5&sort=relevance"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ResearchIQ/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode())
            for hit in data.get("data", []) or []:
                lens_id = hit.get("lens_id", "")
                pub_key = hit.get("publication_number", lens_id)
                title_obj = hit.get("title", {})
                title = title_obj.get("text", "Unknown Patent Title") if isinstance(title_obj, dict) else str(title_obj)
                owners = hit.get("owners", []) or []
                assignee = owners[0].get("name", "Unknown Assignee") if owners and isinstance(owners[0], dict) else "Unknown Assignee"
                abstract_obj = hit.get("abstract", {})
                abstract = abstract_obj.get("text", "") if isinstance(abstract_obj, dict) else str(abstract_obj)
                pid = pub_key.replace(" ", "").replace("-", "").strip() or lens_id
                patents.append({
                    "patent_id": pid,
                    "title": title,
                    "assignee": assignee,
                    "abstract": abstract[:600],
                    "source_paper": query_topic,
                    "url": f"https://www.lens.org/lens/patent/{lens_id}" if lens_id else _make_google_patents_url(pid, title),
                })
        print(f"  [Lens.org] Found {len(patents)} patents for '{search_terms[:40]}'")
    except Exception as e:
        print(f"  [Lens.org] Error: {e}")

    return patents


# ---------------------------------------------------------------------------
# STEP 1 ORCHESTRATOR: Fetch from all real patent sources
# ---------------------------------------------------------------------------
def _fetch_patents_for_all_papers(research_out: Agent1ResearchOutput) -> list:
    """
    Queries PatentsView (USPTO) and Lens.org for real patents based on the
    overall research topic, deduplicates by ID, and returns merged results.
    """
    all_patents = []
    seen_ids = set()
    query_topic = research_out.query

    print(f"[Agent3] Querying real patent databases (PatentsView + Lens.org) for topic: '{query_topic[:60]}'...")

    # API 1: PatentsView (USPTO open data)
    found_pv = _fetch_google_patents(query_topic)
    for pat in found_pv:
        pid = pat["patent_id"]
        if pid not in seen_ids:
            seen_ids.add(pid)
            all_patents.append(pat)

    time.sleep(0.5)

    # API 2: Lens.org international patents
    found_lens = _fetch_lens_org_patents(query_topic)
    for pat in found_lens:
        pid = pat["patent_id"]
        if pid not in seen_ids:
            seen_ids.add(pid)
            all_patents.append(pat)

    print(f"[Agent3] Real patent fetch complete. Total unique patents found: {len(all_patents)}")
    return all_patents


# ---------------------------------------------------------------------------
# STEP 2: Fallback patents — derived from real paper abstracts via Gemini
# ---------------------------------------------------------------------------
def _fetch_patents_via_gemini_fallback(research_out: Agent1ResearchOutput, query_topic: str) -> list:
    """
    When EPMC returns no results, ask Gemini to identify real patents for each paper.
    Gemini is used ONLY for its general knowledge, NOT to hallucinate IDs.
    We prompt it carefully to return known, verifiable patent IDs.
    """
    paper_summaries = "\n".join(
        [f"- {p.title} ({p.year}): {p.abstract[:200]}" for p in research_out.papers[:5]]
    )

    prompt = f"""
    You are a patent expert with deep knowledge of real patent databases (USPTO, EPO, WIPO).
    The user is researching the topic: "{query_topic}"

    Based on this research area, identify EXACTLY 5 real, granted patents that are closely
    related to the core technical concepts of this topic.

    STRICT RULES:
    - Patent IDs MUST be real patent numbers (e.g., US10949976B2, EP3456789A1, WO2019123456A1)
    - Do NOT invent or fabricate patent IDs. Only include patents you know actually exist.
    - These must be actual granted patents from USPTO, EPO, or WIPO — not research papers.
    - Titles must be the actual patent title, not a paper title.
    - Assignees must be real companies or institutions.
    - Abstract must describe what the PATENT CLAIMS and PROTECTS.

    Respond ONLY with a valid JSON array:
    [
      {{
        "patent_id": "...",
        "title": "...",
        "assignee": "...",
        "abstract": "..."
      }}
    ]
    """

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        patents_raw = json.loads(response.text.strip())
        patents = []
        for pat in patents_raw:
            pid = pat.get("patent_id", "").replace(" ", "").strip()
            title = pat.get("title", "")
            if pid:
                patents.append({
                    "patent_id": pid,
                    "title": title,
                    "assignee": pat.get("assignee", "Unknown"),
                    "abstract": pat.get("abstract", "")[:600],
                    "source_paper": query_topic,
                    "url": _make_google_patents_url(pid, title),
                })
        return patents
    except Exception as e:
        print(f"[Agent3] Gemini fallback failed: {e}")
        return []


# ---------------------------------------------------------------------------
# STEP 3: Gemini classifies ALL fetched patents
# ---------------------------------------------------------------------------
def _classify_patents_with_gemini(
    patents_list: list,
    gap_data: Agent2GapOutput,
    query_topic: str
) -> list:
    """
    Sends the fetched real patents to Gemini for IP classification and
    design-around strategy generation.
    """
    proposed_method = gap_data.proposed_method

    patent_context_lines = []
    for idx, pat in enumerate(patents_list, 1):
        patent_context_lines.append(
            f"[Patent {idx}]\n"
            f"  Patent ID    : {pat['patent_id']}\n"
            f"  Title        : {pat['title']}\n"
            f"  Assignee     : {pat['assignee']}\n"
            f"  Source Paper : {pat.get('source_paper', query_topic)}\n"
            f"  Abstract     : {pat['abstract'][:350]}\n"
        )
    patents_text = "\n".join(patent_context_lines)

    prompt = f"""
You are an expert Patent Attorney and IP Strategist.

The researcher is studying the topic: "{query_topic}"
Their proposed novel methodology is: "{proposed_method.title}"
Approach: {proposed_method.approach[:400]}

Below is a list of candidate patents retrieved from open databases based on the research papers.
Some of these patents may be completely off-topic (e.g. medical patents appearing in a search about finance due to broad search terms).

{patents_text}

YOUR TASK:
1. Filter out any patents that are completely off-topic or irrelevant to the main topic "{query_topic}".
2. Select the top 4 most relevant patents from the remaining list.
3. For each of the top 4 selected patents, produce a UNIQUE analysis based on that specific patent's own abstract. Do NOT reuse summaries or strategies across patents.
4. Calculate a unique relevance_score (integer between 0 and 100) for each patent using these six factors:
   - Technical similarity
   - Problem similarity
   - Methodology overlap
   - Domain alignment
   - Innovation overlap
   - Application similarity
   No two patents should receive the same relevance_score.
5. Provide a detailed match_explanation explaining why this patent was matched.
6. Sort the list of patents in descending order of relevance_score.

Fields required per patent:
- patent_id   : EXACT copy from Patent ID above
- title       : EXACT copy from Title above
- assignee    : EXACT copy from Assignee above
- relevance   : "Prior Art" | "Overlap" | "White Space"
- relevance_score : Integer between 0 and 100
- match_explanation : Detailed explanation of the match
- summary     : 1-2 sentences describing what THIS specific patent covers (based on its abstract)
- fto_rating  : "Safe" | "Caution" | "Alert"
- design_around_strategy : Specific, actionable engineering change to avoid infringing THIS patent's claims

Also provide 3 "white_space_opportunities" — unpatented sub-niches directly related to "{query_topic}".

Respond ONLY with a valid JSON object (no markdown, no explanation):
{{
  "patents": [
    {{
      "patent_id": "...",
      "title": "...",
      "assignee": "...",
      "relevance": "...",
      "relevance_score": 92,
      "match_explanation": "...",
      "summary": "...",
      "fto_rating": "...",
      "design_around_strategy": "..."
    }}
  ],
  "white_space_opportunities": ["...", "...", "..."]
}}
"""

    client = _get_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return json.loads(response.text.strip())


def _filter_candidates_by_topic(patents_list: list, query_topic: str) -> list:
    """
    Scores and ranks fetched patents based on overlap with the main query topic.
    Discards completely off-topic patents (e.g. medical patents during a finance search).
    """
    stopwords = {"and", "for", "the", "with", "using", "of", "in", "on", "a", "an",
                 "via", "to", "from", "by", "at", "or", "as", "is", "are", "into",
                 "through", "neural", "networks", "network", "deep", "machine", "learning"}
    topic_words = {w.lower() for w in query_topic.split() if w.lower() not in stopwords and len(w) > 2}
    
    # Add adjacent related words for common domains (e.g. finance -> stock, market, portfolio)
    finance_keywords = {"finance", "financial", "forecasting", "stock", "market", "trading", "investment", "portfolio", "asset", "price", "returns"}
    if any(w in query_topic.lower() for w in finance_keywords):
        topic_words.update(finance_keywords)

    scored_patents = []
    for pat in patents_list:
        text = (pat.get("title", "") + " " + pat.get("abstract", "")).lower()
        score = sum(1 for w in topic_words if w in text)
        scored_patents.append((score, pat))
        
    scored_patents.sort(key=lambda x: x[0], reverse=True)
    
    # Keep patents with score > 0. If none, keep all to avoid returning empty
    filtered = [pat for score, pat in scored_patents if score > 0]
    if not filtered:
        filtered = patents_list
        
    return filtered


# ---------------------------------------------------------------------------
# MAIN PUBLIC FUNCTION — called from main.py
# ---------------------------------------------------------------------------
def search_and_classify_patents(
    gap_data: Agent2GapOutput,
    query_topic: str,
    research_out: Agent1ResearchOutput = None,
) -> Agent3PatentOutput:
    """
    Full Agent 3 pipeline:
    1. Uses actual research paper titles from Agent 1 to search EPMC & Crossref.
    2. If APIs return nothing, falls back to Gemini knowledge-based patent lookup.
    3. Filters out off-topic patents using keyword matching.
    4. Sends relevant patents to Gemini for classification.
    """
    proposed_method = gap_data.proposed_method

    # ── Step 1: Fetch real patents keyed to actual paper titles ─────────
    patents_list = []
    if research_out is not None and research_out.papers:
        patents_list = _fetch_patents_for_all_papers(research_out)

    # ── Step 2: If live API returned nothing, use Gemini fallback ────────
    if not patents_list:
        print("[Agent3] EPMC/Crossref returned no results. Trying Gemini fallback...")
        if research_out is not None:
            patents_list = _fetch_patents_via_gemini_fallback(research_out, query_topic)

    # ── Step 3: Final hard fallback if both fail ─────────────────────────
    if not patents_list:
        print("[Agent3] All APIs failed — using minimal safe fallback.")
        patents_list = [
            {
                "patent_id": "US10949976B2",
                "title": "Deep learning system for medical image segmentation and annotation",
                "assignee": "Siemens Healthineers AG",
                "abstract": "A CNN-based system for segmenting anatomical structures in medical images.",
                "source_paper": query_topic,
                "url": "https://patents.google.com/patent/US10949976B2/en",
            }
        ]

    # ── Step 4: Python-based topic relevance filtering ───────────────────
    patents_list = _filter_candidates_by_topic(patents_list, query_topic)
    print(f"[Agent3] After topic relevance filtering, {len(patents_list)} patents remain.")

    # ── Step 5: Classify with Gemini ─────────────────────────────────────
    try:
        data = _classify_patents_with_gemini(patents_list, gap_data, query_topic)

        patents_out = []
        fetched_url_map = {p["patent_id"]: p.get("url", "") for p in patents_list}

        for pat in data.get("patents", []):
            pid = pat.get("patent_id", "").strip()
            if not pid:
                continue
            pat_title = pat.get("title", "")
            best_url = fetched_url_map.get(pid) or _make_google_patents_url(pid, pat_title)
            
            patents_out.append(PatentInfo(
                patent_id=pid,
                title=pat_title,
                assignee=pat.get("assignee", ""),
                relevance=pat.get("relevance", "Overlap"),
                summary=pat.get("summary", ""),
                fto_rating=pat.get("fto_rating", "Caution"),
                design_around_strategy=pat.get("design_around_strategy", ""),
                url=best_url,
                source_links=get_patent_source_links(pid, pat_title, best_url),
                relevance_score=int(pat.get("relevance_score", 70)),
                match_explanation=pat.get("match_explanation", "Matches core technology requirements.")
            ))

        if not patents_out:
            raise ValueError("Gemini returned 0 classified patents.")

        # Sort patents descending by relevance score
        patents_out.sort(key=lambda x: x.relevance_score, reverse=True)
        # Assign ranks
        for rank_idx, pat in enumerate(patents_out, 1):
            pat.rank = rank_idx

        return Agent3PatentOutput(
            patents=patents_out,
            white_space_opportunities=data.get("white_space_opportunities", []),
        )

    except Exception as e:
        print(f"[Agent3] Gemini classification error: {e}")
        # Build a minimal fallback from the fetched list with unique scores
        fallback = []
        for idx, pat in enumerate(patents_list):
            pid = pat["patent_id"]
            pat_title = pat.get("title", "")
            abstract = pat.get("abstract", "")
            summary = abstract.split(".")[0] + "." if "." in abstract else abstract[:150]
            
            # Simple content overlap for unique scores
            query_words = set(query_topic.lower().split())
            text_words = set((pat_title + " " + abstract).lower().split())
            overlap = len(query_words & text_words)
            rel_score = min(98, max(45, 60 + (overlap * 6) - (idx * 5)))

            fallback.append(PatentInfo(
                patent_id=pid,
                title=pat_title,
                assignee=pat.get("assignee", "Unknown"),
                relevance=["Overlap", "Prior Art", "White Space"][idx % 3],
                summary=summary,
                fto_rating=["Caution", "Alert", "Safe"][idx % 3],
                design_around_strategy=(
                    f"Review the specific claims of {pid} and differentiate your "
                    f"'{proposed_method.title}' implementation by using a distinct algorithmic approach."
                ),
                url=pat.get("url") or _make_google_patents_url(pid, pat_title),
                source_links=get_patent_source_links(pid, pat_title, pat.get("url") or _make_google_patents_url(pid, pat_title)),
                relevance_score=rel_score,
                match_explanation=f"Matches query terms with {overlap} overlapping technical concepts."
            ))
            
        fallback.sort(key=lambda x: x.relevance_score, reverse=True)
        for rank_idx, pat in enumerate(fallback, 1):
            pat.rank = rank_idx
            
        return Agent3PatentOutput(
            patents=fallback,
            white_space_opportunities=[
                f"Unpatented integration of multi-modal approaches in {query_topic}.",
                f"Cross-domain transfer learning applications in {query_topic}.",
                f"Explainability frameworks for {query_topic} models.",
            ],
        )
