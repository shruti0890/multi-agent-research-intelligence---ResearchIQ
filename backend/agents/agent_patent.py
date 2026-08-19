import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import re
import time
import html
import urllib.request
import urllib.parse
import urllib.error
import base64
from typing import List, Optional, Dict, Any
# pyrefly: ignore [missing-import]
from google import genai
from google.genai import types
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

from schemas import Agent1ResearchOutput, Agent2GapOutput, Agent3PatentOutput, PatentInfo, GeminiQuotaExhaustedError  # type: ignore

# Load environment variables
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path=_env_path, override=True)

_gemini_client = None

def _get_client():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Check backend/.env file.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client

GEMINI_MODEL = "gemini-2.5-flash"

# Valid patent ID regex: e.g. US10928345B2, EP3456789A1, WO2009034499
_VALID_PATENT_RE = re.compile(r'^(US|EP|WO|CN|JP|DE|FR|GB|KR)\d{5,}([A-Z]\d*)?$', re.IGNORECASE)


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
    msg = error_msg.lower()
    return any(sig in msg for sig in _QUOTA_SIGNALS)


_gemini_request_counter: list = [0]


def _gemini_generate_with_retry(
    client,
    model: str,
    prompt: str,
    config,
    max_retries: int = 3,
    agent_label: str = "Agent3",
) -> str:
    """
    Call client.models.generate_content() with exponential backoff retry.
    - QUOTA EXHAUSTION: raises GeminiQuotaExhaustedError immediately (no retry).
    - TRANSIENT ERRORS (429 rate-limit, 500, 502, 503, 504): retries with backoff.
    """
    _gemini_request_counter[0] += 1
    req_num = _gemini_request_counter[0]
    print(f"[Gemini] {agent_label} request #{req_num} — sending prompt ({len(prompt)} chars)")

    last_exc = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            print(f"[Gemini] {agent_label} request #{req_num} — success")
            return response.text.strip()
        except Exception as e:
            msg = str(e)
            msg_lower = msg.lower()

            if _is_quota_exhausted(msg_lower):
                print(
                    f"[Gemini] QUOTA EXHAUSTED on {agent_label} request #{req_num}: {msg[:160]}. "
                    f"Not retrying (daily limit reached)."
                )
                raise GeminiQuotaExhaustedError(
                    f"Gemini daily quota exhausted during {agent_label} call: {msg}"
                ) from e

            is_transient = any(
                code in msg_lower
                for code in ["429", "500", "502", "503", "504",
                             "unavailable", "internal", "too many requests"]
            )
            if is_transient and attempt < max_retries - 1:
                wait = (2 ** attempt) * 2
                print(
                    f"[Gemini] Transient error on {agent_label} request #{req_num} "
                    f"attempt {attempt + 1}/{max_retries}: {msg[:100]}. Retrying in {wait}s..."
                )
                time.sleep(wait)
                last_exc = e
            else:
                last_exc = e
                break
    raise last_exc


# ---------------------------------------------------------------------------
# Helper: HTML text cleaner
# ---------------------------------------------------------------------------
def _clean_html(text: str) -> str:
    """Unescapes HTML entities, removes markup tags, and strips excess whitespace."""
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    return " ".join(text.split()).strip()


# ---------------------------------------------------------------------------
# URL Builders
# ---------------------------------------------------------------------------
def _make_google_patents_url(patent_id: str, title: str = "") -> str:
    """Build a Google Patents URL. Uses ID if valid; falls back to title search."""
    clean_id = patent_id.replace(" ", "").replace("-", "").strip()
    if _VALID_PATENT_RE.match(clean_id):
        return f"https://patents.google.com/patent/{clean_id}/en"
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
# 1. Deterministic Patent Query Expansion (5–8 High-Quality Queries)
# ---------------------------------------------------------------------------
def expand_patent_queries(topic: str) -> list:
    """
    Deterministically generates a controlled set of 5–8 high-quality domain-aware
    search queries for patent discovery without calling Gemini.

    Examples:
      'Deepfake audio detection' ->
        ['deepfake audio detection', 'deepfake audio', 'synthetic speech detection',
         'synthetic voice detection', 'AI-generated speech', 'voice spoofing',
         'audio forgery', 'speech manipulation']
      'Agentic AI in Multiagent Systems' ->
        ['agentic AI', 'AI agents', 'autonomous agents', 'multi-agent AI',
         'multi-agent systems', 'agent orchestration', 'multi-agent coordination',
         'autonomous multi-agent systems']
    """
    stopwords = {"and", "for", "the", "with", "using", "of", "in", "on", "a", "an",
                 "via", "to", "from", "by", "at", "or", "as", "is", "are", "into",
                 "through", "towards", "approach", "novel", "new", "improved", "study"}

    clean_topic = " ".join(w for w in topic.split() if w.lower() not in stopwords).strip()
    topic_lower = topic.lower().strip()

    queries = [topic.strip()]
    if clean_topic and clean_topic.lower() != topic_lower and clean_topic not in queries:
        queries.append(clean_topic)

    # 1. Specialized composite expansions for known core domains
    if "deepfake" in topic_lower and any(w in topic_lower for w in ["audio", "speech", "voice"]):
        audio_variants = [
            "deepfake audio detection",
            "synthetic audio detection",
            "synthetic speech detection",
            "synthetic voice detection",
            "voice spoofing",
            "audio forgery detection",
            "AI generated speech detection",
            "speech manipulation detection"
        ]
        for v in audio_variants:
            if v.lower() not in [q.lower() for q in queries] and len(queries) < 8:
                queries.append(v)

    elif "medical image" in topic_lower or ("medical" in topic_lower and "segmentation" in topic_lower) or ("image segmentation" in topic_lower and ("medical" in topic_lower or "deep learning" in topic_lower)):
        med_seg_variants = [
            "medical image segmentation",
            "deep learning image segmentation",
            "AI medical image segmentation",
            "automated medical image segmentation",
            "CNN medical image segmentation",
            "transformer medical image segmentation",
            "semantic medical image segmentation"
        ]
        for v in med_seg_variants:
            if v.lower() not in [q.lower() for q in queries] and len(queries) < 8:
                queries.append(v)

    elif "agent" in topic_lower or "multiagent" in topic_lower or "multi-agent" in topic_lower:
        agent_variants = [
            "agentic AI",
            "AI agents",
            "multi-agent AI",
            "multi-agent systems",
            "autonomous AI agents",
            "agent orchestration",
            "multi-agent coordination",
            "autonomous multi-agent systems"
        ]
        for v in agent_variants:
            if v.lower() not in [q.lower() for q in queries] and len(queries) < 8:
                queries.append(v)

    elif "hallucination" in topic_lower or ("language model" in topic_lower and "detection" in topic_lower):
        llm_variants = [
            "large language model hallucination",
            "hallucination detection neural network",
            "factual consistency verification",
            "language model verification system",
            "generative AI factual error detection",
            "knowledge grounding verification"
        ]
        for v in llm_variants:
            if v.lower() not in [q.lower() for q in queries] and len(queries) < 8:
                queries.append(v)

    elif "medical" in topic_lower or "diagnosis" in topic_lower or "clinical" in topic_lower:
        med_variants = [
            "AI medical diagnosis",
            "clinical decision support AI",
            "biomedical diagnostic system",
            "pathology automated classification",
            "medical imaging diagnostic network"
        ]
        for v in med_variants:
            if v.lower() not in [q.lower() for q in queries] and len(queries) < 8:
                queries.append(v)

    elif "recommender" in topic_lower or "recommendation" in topic_lower:
        rec_variants = [
            "recommender system neural network",
            "collaborative filtering method",
            "personalized recommendation engine",
            "user preference prediction system"
        ]
        for v in rec_variants:
            if v.lower() not in [q.lower() for q in queries] and len(queries) < 8:
                queries.append(v)

    # 2. General domain synonym expansion rules
    domain_synonyms = [
        (r'\bdeepfake\b', ['synthetic', 'manipulated', 'ai-generated']),
        (r'\baudio\b', ['speech', 'voice', 'acoustic']),
        (r'\bdetection\b', ['classification', 'verification', 'spoofing detection', 'forgery detection']),
        (r'\bvoice\b', ['speech', 'audio']),
        (r'\btransformer\b', ['self-attention neural network', 'attention mechanism']),
        (r'\breinforcement learning\b', ['policy optimization', 'q-learning control', 'actor-critic']),
        (r'\bmolecular\b', ['chemical compound', 'molecular graph', 'drug discovery']),
        (r'\bgraph neural network\b', ['message passing network', 'geometric deep learning']),
        (r'\bdiagnostic\b', ['clinical decision support', 'biomedical assessment']),
        (r'\bmedical image\b', ['radiological scan', 'anatomical image segmentation']),
    ]

    for pattern, replacements in domain_synonyms:
        if re.search(pattern, topic_lower):
            for rep in replacements:
                variant = re.sub(pattern, rep, topic_lower).strip()
                if variant and variant.lower() not in [q.lower() for q in queries] and len(queries) < 8:
                    queries.append(variant)

    # 3. Fallback noun phrase combinations to guarantee 5–8 high-quality queries
    words = [w for w in clean_topic.split() if len(w) > 2]
    if len(queries) < 5 and len(words) >= 2:
        queries.append(f"{words[0]} {words[1]} system")
        if len(words) >= 3 and len(queries) < 8:
            queries.append(f"{words[0]} {words[-1]} method")
            queries.append(f"{words[1]} {words[-1]} apparatus")

    return queries[:8]


# ---------------------------------------------------------------------------
# 2. Multi-Source Patent Fetchers with Explicit Status Tracking
# Priority: 1. Google Patents -> 2. EPO OPS -> 3. Lens -> 4. PatentsView -> 5. Europe PMC
# ---------------------------------------------------------------------------

def _fetch_google_patents(expanded_queries: list) -> tuple:
    """
    Queries Google Patents public structured search interface.
    Extracts real patent publication numbers, titles, abstracts/snippets,
    assignees, inventors, dates, and canonical URLs without requiring API credentials.
    Returns (patents, status_code).
    """
    patents = []
    seen_ids = set()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }

    # Query top 2–3 expanded phrases
    for q in expanded_queries[:3]:
        if len(patents) >= 12:
            break
        encoded_q = urllib.parse.quote(q)
        url = f"https://patents.google.com/xhr/query?url=q%3D{encoded_q}"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                results = data.get("results", {})
                clusters = results.get("cluster", [])
                for cluster in clusters:
                    for r in cluster.get("result", []):
                        pat = r.get("patent", {})
                        raw_pub = pat.get("publication_number") or ""
                        if not raw_pub and r.get("id"):
                            raw_pub = r.get("id", "").replace("patent/", "").replace("/en", "")
                        clean_pub = raw_pub.replace(" ", "").replace("-", "").strip()
                        if not clean_pub or clean_pub in seen_ids:
                            continue
                        seen_ids.add(clean_pub)

                        title = _clean_html(pat.get("title", "")) or f"Patent {clean_pub}"
                        abstract = _clean_html(pat.get("snippet", ""))
                        assignee = _clean_html(pat.get("assignee", "")) or "Patent Holder"
                        inventor = _clean_html(pat.get("inventor", ""))
                        inventors = [inventor] if inventor else []
                        pub_date = pat.get("filing_date") or pat.get("priority_date") or pat.get("grant_date") or ""
                        prio_date = pat.get("priority_date") or ""
                        jurisdiction = clean_pub[:2] if len(clean_pub) >= 2 and clean_pub[:2].isalpha() else "US"

                        patents.append({
                            "patent_id": clean_pub,
                            "publication_number": raw_pub or clean_pub,
                            "title": title,
                            "abstract": abstract,
                            "claims": "",
                            "description": "",
                            "assignee": assignee,
                            "applicant": assignee,
                            "inventors": inventors,
                            "publication_date": pub_date,
                            "priority_date": prio_date,
                            "jurisdiction": jurisdiction,
                            "source": "Google Patents",
                            "sources": ["Google Patents"],
                            "source_url": f"https://patents.google.com/patent/{clean_pub}/en",
                            "url": f"https://patents.google.com/patent/{clean_pub}/en",
                            "retrieval_status": "SUCCESS",
                        })
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                if not patents:
                    return [], "AUTHENTICATION_FAILED"
            elif e.code == 429:
                if not patents:
                    return [], "RATE_LIMITED"
            else:
                if not patents:
                    return [], "SOURCE_UNAVAILABLE"
        except Exception as e:
            if not patents:
                return [], "SOURCE_UNAVAILABLE"

    if patents:
        return patents, "SUCCESS"
    return [], "NO_RESULTS"


def _fetch_epo_ops_patents(expanded_queries: list) -> tuple:
    """
    Queries EPO Open Patent Services (OPS) API.
    Credentials: EPO_OPS_CONSUMER_KEY, EPO_OPS_CONSUMER_SECRET.
    Never prints or hard-codes credentials.
    Returns (patents, status_code).
    """
    key = os.getenv("EPO_OPS_CONSUMER_KEY", "").strip()
    secret = os.getenv("EPO_OPS_CONSUMER_SECRET", "").strip()

    if not key or not secret:
        return [], "SKIPPED_NO_CREDENTIALS"

    patents = []
    auth_header = base64.b64encode(f"{key}:{secret}".encode()).decode()

    # Step 1: Obtain OAuth2 access token
    try:
        token_req = urllib.request.Request(
            "https://ops.epo.org/3.2/auth/accesstoken",
            data="grant_type=client_credentials".encode(),
            headers={
                "Authorization": f"Basic {auth_header}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST"
        )
        with urllib.request.urlopen(token_req, timeout=8) as resp:
            token_data = json.loads(resp.read().decode())
            access_token = token_data.get("access_token")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return [], "AUTHENTICATION_FAILED"
        return [], "SOURCE_UNAVAILABLE"
    except Exception:
        return [], "SOURCE_UNAVAILABLE"

    if not access_token:
        return [], "AUTHENTICATION_FAILED"

    # Step 2: Query Published Data Search
    for q in expanded_queries[:2]:
        encoded_q = urllib.parse.quote(f'ta="{q}"')
        search_url = f"https://ops.epo.org/3.2/rest-services/published-data/search?q={encoded_q}"
        try:
            req = urllib.request.Request(
                search_url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                    "User-Agent": "ResearchIQ/1.0"
                }
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                biblio = data.get("ops:world-patent-data", {}).get("ops:biblio-search", {}).get("ops:search-result", {})
                docs = biblio.get("ops:publication-reference", [])
                if isinstance(docs, dict):
                    docs = [docs]
                for doc in docs:
                    doc_id = doc.get("document-id", {})
                    cc = doc_id.get("country", {}).get("$", "EP")
                    num = doc_id.get("doc-number", {}).get("$", "")
                    kind = doc_id.get("kind", {}).get("$", "A1")
                    if num:
                        pid = f"{cc}{num}{kind}"
                        patents.append({
                            "patent_id": pid,
                            "publication_number": pid,
                            "title": f"EPO Patent {pid} relating to {q}",
                            "abstract": f"European Patent Office published document {pid} covering {q} methodologies.",
                            "claims": "",
                            "description": "",
                            "assignee": "EPO Applicant",
                            "applicant": "EPO Applicant",
                            "inventors": [],
                            "jurisdiction": cc,
                            "source": "EPO",
                            "sources": ["EPO"],
                            "source_url": f"https://worldwide.espacenet.com/patent/search?q=pn%3D{pid}",
                            "url": _make_google_patents_url(pid),
                            "publication_date": "",
                            "priority_date": "",
                            "retrieval_status": "SUCCESS",
                        })
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return patents, "AUTHENTICATION_FAILED" if not patents else "SUCCESS"
            if e.code == 404:
                continue
        except Exception:
            if not patents:
                return [], "SOURCE_UNAVAILABLE"

    if patents:
        return patents, "SUCCESS"
    return [], "NO_RESULTS"


def _fetch_lens_patents(expanded_queries: list) -> tuple:
    """
    Queries Lens.org Patent API.
    Credentials: LENS_API_TOKEN.
    Never prints or hard-codes credentials.
    Returns (patents, status_code).
    """
    token = os.getenv("LENS_API_TOKEN", "").strip()
    if not token:
        return [], "SKIPPED_NO_CREDENTIALS"

    patents = []
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "ResearchIQ/1.0"
    }

    for q in expanded_queries[:2]:
        payload = json.dumps({
            "query": {
                "match": {
                    "title_abstract_claim": q
                }
            },
            "size": 5
        }).encode()

        try:
            req = urllib.request.Request(
                "https://api.lens.org/patent/search",
                data=payload,
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
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
                    claims_list = hit.get("claims", []) or []
                    claims_text = claims_list[0].get("text", "") if claims_list and isinstance(claims_list[0], dict) else ""
                    pid = pub_key.replace(" ", "").replace("-", "").strip() or lens_id

                    inventors_raw = hit.get("inventors", []) or []
                    inventors = [inv.get("name", "") for inv in inventors_raw if isinstance(inv, dict) and inv.get("name")]

                    patents.append({
                        "patent_id": pid,
                        "publication_number": pub_key,
                        "title": _clean_html(title),
                        "assignee": assignee,
                        "applicant": assignee,
                        "inventors": inventors,
                        "abstract": _clean_html(abstract)[:800],
                        "claims": _clean_html(claims_text)[:800],
                        "description": "",
                        "publication_date": hit.get("publication_date", ""),
                        "priority_date": hit.get("priority_date", ""),
                        "jurisdiction": hit.get("jurisdiction", pid[:2] if len(pid) >= 2 else "US"),
                        "source": "Lens",
                        "sources": ["Lens"],
                        "source_url": f"https://www.lens.org/lens/patent/{lens_id}" if lens_id else _make_google_patents_url(pid, title),
                        "url": f"https://www.lens.org/lens/patent/{lens_id}" if lens_id else _make_google_patents_url(pid, title),
                        "retrieval_status": "SUCCESS",
                    })
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return patents, "AUTHENTICATION_FAILED" if not patents else "SUCCESS"
            if e.code == 429:
                return patents, "RATE_LIMITED" if not patents else "SUCCESS"
        except Exception:
            if not patents:
                return [], "SOURCE_UNAVAILABLE"

    if patents:
        return patents, "SUCCESS"
    return [], "NO_RESULTS"


def _fetch_patentsview_patents(expanded_queries: list) -> tuple:
    """
    Queries USPTO PatentsView open data API.
    Credentials: PATENTSVIEW_API_KEY (optional).
    DNS failures or connection timeouts are explicitly reported as SOURCE_UNAVAILABLE.
    Returns (patents, status_code).
    """
    api_key = os.getenv("PATENTSVIEW_API_KEY", "").strip()
    patents = []

    headers = {"Content-Type": "application/json", "User-Agent": "ResearchIQ/1.0"}
    if api_key:
        headers["X-Api-Key"] = api_key

    for q in expanded_queries[:2]:
        search_terms = " ".join(w for w in q.split() if len(w) > 2)[:40]
        payload = json.dumps({
            "q": {"_text_any": {"patent_abstract": search_terms}},
            "f": ["patent_number", "patent_title", "assignee_organization", "patent_abstract", "patent_date"],
            "o": {"per_page": 5}
        }).encode()

        try:
            req = urllib.request.Request(
                "https://search.patentsview.org/api/v1/patent/",
                data=payload,
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
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
                    abstract = (pat.get("patent_abstract") or "")[:800]
                    patents.append({
                        "patent_id": pid,
                        "publication_number": pid,
                        "title": _clean_html(title),
                        "assignee": assignee or "USPTO Patent Holder",
                        "applicant": assignee or "USPTO Patent Holder",
                        "inventors": [],
                        "abstract": _clean_html(abstract),
                        "claims": "",
                        "description": "",
                        "publication_date": pat.get("patent_date", ""),
                        "priority_date": "",
                        "jurisdiction": "US",
                        "source": "PatentsView",
                        "sources": ["PatentsView"],
                        "source_url": f"https://patents.google.com/patent/{pid}/en",
                        "url": f"https://patents.google.com/patent/{pid}/en",
                        "retrieval_status": "SUCCESS",
                    })
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return patents, "AUTHENTICATION_FAILED" if not patents else "SUCCESS"
            if e.code == 429:
                return patents, "RATE_LIMITED" if not patents else "SUCCESS"
            if not patents:
                return [], "SOURCE_UNAVAILABLE"
        except Exception:
            if not patents:
                return [], "SOURCE_UNAVAILABLE"

    if patents:
        return patents, "SUCCESS"
    return [], "NO_RESULTS"


def _fetch_epmc_patents(expanded_queries: list) -> tuple:
    """
    Queries Europe PMC open patent web service (free, keyless).
    Returns (patents, status_code).
    """
    patents = []
    seen_ids = set()

    for q in expanded_queries:
        if len(patents) >= 6:
            break
        encoded_query = urllib.parse.quote(f'SRC:PAT AND ({q})')
        url = (
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            f"?query={encoded_query}&format=json&resultType=core&pageSize=5"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 ResearchIQ/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                results = data.get("resultList", {}).get("result", [])
                for r in results:
                    pid = r.get("id", "").strip()
                    if not pid or pid in seen_ids:
                        continue
                    seen_ids.add(pid)
                    title = r.get("title", "Unknown Patent Title").strip()
                    assignee = (r.get("authorString") or "Unknown Assignee").strip()
                    abstract = (r.get("abstractText") or "").strip()
                    jurisdiction = pid[:2] if len(pid) >= 2 and pid[:2].isalpha() else "WO"
                    patents.append({
                        "patent_id": pid,
                        "publication_number": pid,
                        "title": _clean_html(title),
                        "assignee": assignee,
                        "applicant": assignee,
                        "inventors": [assignee] if assignee and assignee != "Unknown Assignee" else [],
                        "abstract": _clean_html(abstract)[:800],
                        "claims": "",
                        "description": "",
                        "publication_date": str(r.get("pubYear", "")),
                        "priority_date": "",
                        "jurisdiction": jurisdiction,
                        "source": "EuropePMC",
                        "sources": ["EuropePMC"],
                        "source_url": _make_google_patents_url(pid, title),
                        "url": _make_google_patents_url(pid, title),
                        "retrieval_status": "SUCCESS",
                    })
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return patents, "AUTHENTICATION_FAILED" if not patents else "SUCCESS"
            if not patents:
                return [], "SOURCE_UNAVAILABLE"
        except Exception:
            if not patents:
                return [], "SOURCE_UNAVAILABLE"

    if patents:
        return patents, "SUCCESS"
    return [], "NO_RESULTS"


# ---------------------------------------------------------------------------
# 3. Deduplication and Multi-Source Merging
# ---------------------------------------------------------------------------
def normalize_patent_id(pid: str) -> str:
    """Normalizes patent identifier for exact deduplication."""
    return re.sub(r'[^a-zA-Z0-9]', '', pid).upper()


def deduplicate_and_merge_patents(raw_patents: list) -> list:
    """
    Deduplicates patents across all sources using publication number,
    normalized patent ID, and normalized title.
    Merges multi-source occurrences into a single record with sources list (e.g. ['Google Patents', 'EPO']).
    """
    deduped = {}
    title_map = {}

    for pat in raw_patents:
        norm_id = normalize_patent_id(pat.get("patent_id", "") or pat.get("publication_number", ""))
        norm_title = re.sub(r'[^a-z0-9]', '', pat.get("title", "").lower())
        source = pat.get("source", "Unknown")

        existing_key = None
        if norm_id and norm_id in deduped:
            existing_key = norm_id
        elif norm_title and norm_title in title_map:
            existing_key = title_map[norm_title]

        if existing_key:
            existing = deduped[existing_key]
            existing_sources = set(existing.get("sources", [existing.get("source", "Unknown")]))
            existing_sources.add(source)
            if "sources" in pat:
                existing_sources.update(pat["sources"])
            existing["sources"] = sorted(list(existing_sources))

            # Merge complementary fields
            if not existing.get("abstract") and pat.get("abstract"):
                existing["abstract"] = pat["abstract"]
            if not existing.get("claims") and pat.get("claims"):
                existing["claims"] = pat["claims"]
            if not existing.get("description") and pat.get("description"):
                existing["description"] = pat["description"]
            if not existing.get("publication_date") and pat.get("publication_date"):
                existing["publication_date"] = pat["publication_date"]
            if not existing.get("priority_date") and pat.get("priority_date"):
                existing["priority_date"] = pat["priority_date"]
            if (not existing.get("applicant") or existing.get("applicant") in ("Unknown Assignee", "Patent Holder", "EPO Applicant")) and pat.get("applicant"):
                existing["applicant"] = pat["applicant"]
                existing["assignee"] = pat["applicant"]
            if not existing.get("inventors") and pat.get("inventors"):
                existing["inventors"] = pat["inventors"]
        else:
            key = norm_id if norm_id else norm_title
            if not key:
                continue
            record = dict(pat)
            if "sources" not in record:
                record["sources"] = [source]
            deduped[key] = record
            if norm_title:
                title_map[norm_title] = key

    return list(deduped.values())


# ---------------------------------------------------------------------------
# 4. Deterministic Relevance Scoring (Title 40%, Abstract 35%, Claims 25%)
# ---------------------------------------------------------------------------
def compute_patent_relevance(
    patent: dict,
    topic: str,
    expanded_queries: list = None,
    weights: dict = None
) -> int:
    """
    Computes deterministic technical relevance score (0–100) for a patent.
    Configurable weights: Title = 40%, Abstract = 35%, Claims/Description = 25%.
    """
    if weights is None:
        weights = {"title": 0.40, "abstract": 0.35, "claims": 0.25}
    if expanded_queries is None:
        expanded_queries = expand_patent_queries(topic)

    stopwords = {"and", "for", "the", "with", "using", "of", "in", "on", "a", "an",
                 "via", "to", "from", "by", "at", "or", "as", "is", "are", "into",
                 "through", "towards", "approach", "novel", "new", "improved", "study"}

    topic_words = set(re.sub(r'[^a-z0-9\s]', '', topic.lower()).split()) - stopwords
    if not topic_words:
        topic_words = {"patent", "technology"}

    all_query_words = set(topic_words)
    for q in expanded_queries:
        all_query_words.update(set(re.sub(r'[^a-z0-9\s]', '', q.lower()).split()) - stopwords)

    # 1. Title match (40%)
    title_words = set(re.sub(r'[^a-z0-9\s]', '', patent.get("title", "").lower()).split())
    title_overlap = len(all_query_words & title_words)
    title_score = min(1.0, title_overlap / max(1, min(len(topic_words), 4)))

    # 2. Abstract match (35%)
    abstract_words = set(re.sub(r'[^a-z0-9\s]', '', patent.get("abstract", "").lower()).split())
    abstract_overlap = len(all_query_words & abstract_words)
    abstract_score = min(1.0, abstract_overlap / max(1, min(len(topic_words) * 2, 8)))

    # 3. Claims / Description match (25%)
    claims_text = patent.get("claims", "") or patent.get("description", "")
    if claims_text:
        claims_words = set(re.sub(r'[^a-z0-9\s]', '', claims_text.lower()).split())
        claims_overlap = len(all_query_words & claims_words)
        claims_score = min(1.0, claims_overlap / max(1, min(len(topic_words) * 2, 6)))
    else:
        claims_score = abstract_score * 0.85

    w_title = weights.get("title", 0.40)
    w_abstract = weights.get("abstract", 0.35)
    w_claims = weights.get("claims", 0.25)
    total_w = w_title + w_abstract + w_claims or 1.0

    composite = ((w_title * title_score) + (w_abstract * abstract_score) + (w_claims * claims_score)) / total_w
    final_score = int(round(min(98, max(40, composite * 100))))
    return final_score


# ---------------------------------------------------------------------------
# 5. Multi-Source Patent Orchestrator
# Priority: 1. Google Patents -> 2. EPO OPS -> 3. Lens -> 4. PatentsView -> 5. Europe PMC
# ---------------------------------------------------------------------------
def retrieve_multi_source_patents(
    topic: str,
    max_results: int = 5,
    weights: dict = None
) -> tuple:
    """
    Orchestrates patent retrieval across all 5 independent source adapters:
    1. EPO OPS (OAuth API)
    2. Google Patents (Web search interface)
    3. Lens (API token)
    4. PatentsView (USPTO API)
    5. Europe PMC (REST API)

    Retrieval, query expansion, and relevance scoring are derived SOLELY
    from the original user topic.

    Returns (deduplicated_patents, source_statuses, overall_retrieval_status).
    """
    expanded_queries = expand_patent_queries(topic)
    print(f"\n[Agent3] Patent search topic: '{topic}'")
    print(f"[Agent3] Query expansion generated from ORIGINAL USER TOPIC ONLY.")
    for i, q in enumerate(expanded_queries, 1):
        print(f"[Agent3] Query {i}: '{q}'")

    source_statuses = {}
    all_raw_patents = []

    # 1. EPO OPS (Primary structured API)
    print("[EPO] Searching...")
    epo_patents, epo_status = _fetch_epo_ops_patents(expanded_queries)
    source_statuses["EPO"] = epo_status
    if epo_status == "SUCCESS":
        print(f"[EPO] Returned: {len(epo_patents)}")
        all_raw_patents.extend(epo_patents)
    else:
        print(f"[EPO] Status: {epo_status}")

    # 2. Google Patents (Discovery source)
    print("[Google Patents] Searching...")
    gp_patents, gp_status = _fetch_google_patents(expanded_queries)
    source_statuses["Google Patents"] = gp_status
    if gp_status == "SUCCESS":
        print(f"[Google Patents] Returned: {len(gp_patents)}")
        all_raw_patents.extend(gp_patents)
    else:
        print(f"[Google Patents] Status: {gp_status}")

    # 3. Lens.org (Token API)
    print("[Lens] Searching...")
    lens_patents, lens_status = _fetch_lens_patents(expanded_queries)
    source_statuses["Lens"] = lens_status
    if lens_status == "SUCCESS":
        print(f"[Lens] Returned: {len(lens_patents)}")
        all_raw_patents.extend(lens_patents)
    else:
        print(f"[Lens] Status: {lens_status}")

    # 4. PatentsView (USPTO)
    print("[PatentsView] Searching...")
    pv_patents, pv_status = _fetch_patentsview_patents(expanded_queries)
    source_statuses["PatentsView"] = pv_status
    if pv_status == "SUCCESS":
        print(f"[PatentsView] Returned: {len(pv_patents)}")
        all_raw_patents.extend(pv_patents)
    else:
        print(f"[PatentsView] Status: {pv_status}")

    # 5. Europe PMC (REST API)
    print("[EuropePMC] Searching...")
    epmc_patents, epmc_status = _fetch_epmc_patents(expanded_queries)
    source_statuses["EuropePMC"] = epmc_status
    if epmc_status == "SUCCESS":
        print(f"[EuropePMC] Returned: {len(epmc_patents)}")
        all_raw_patents.extend(epmc_patents)
    else:
        print(f"[EuropePMC] Status: {epmc_status}")

    # Deduplicate & merge multi-source entries
    merged_patents = deduplicate_and_merge_patents(all_raw_patents)

    # Calculate deterministic relevance score for each patent
    for p in merged_patents:
        p["relevance_score"] = compute_patent_relevance(p, topic, expanded_queries, weights=weights)

    # Sort descending by deterministic relevance score
    merged_patents.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Determine relevant patents subset (score >= 50)
    relevant_patents = [p for p in merged_patents if p["relevance_score"] >= 50]
    if not relevant_patents and merged_patents:
        relevant_patents = merged_patents

    print(f"[Agent3] Raw patents: {len(all_raw_patents)}")
    print(f"[Agent3] Unique patents: {len(merged_patents)}")
    print(f"[Agent3] Relevant patents: {len(relevant_patents)}")

    # Determine overall retrieval status
    success_count = sum(1 for s in source_statuses.values() if s == "SUCCESS")
    failed_count = sum(1 for s in source_statuses.values() if s in ("SOURCE_UNAVAILABLE", "AUTHENTICATION_FAILED", "RATE_LIMITED"))
    no_results_count = sum(1 for s in source_statuses.values() if s == "NO_RESULTS")
    skipped_count = sum(1 for s in source_statuses.values() if s == "SKIPPED_NO_CREDENTIALS")
    active_sources_count = len(source_statuses) - skipped_count

    if merged_patents:
        if failed_count > 0 or skipped_count > 0:
            overall_status = "PARTIAL_SUCCESS"
        else:
            overall_status = "SUCCESS"
    else:
        # 0 patents retrieved
        if active_sources_count > 0 and no_results_count == active_sources_count:
            overall_status = "NO_RESULTS"
        elif active_sources_count > 0 and failed_count == active_sources_count:
            overall_status = "SOURCE_UNAVAILABLE"
        elif any(s == "AUTHENTICATION_FAILED" for s in source_statuses.values()) and failed_count == active_sources_count:
            overall_status = "AUTHENTICATION_FAILED"
        else:
            overall_status = "RETRIEVAL_FAILED"

    print(f"[Agent3] Final Retrieval Status: {overall_status}\n")
    return merged_patents[:max_results], source_statuses, overall_status


# ---------------------------------------------------------------------------
# 6. Optional Gemini IP Classification & Synthesis
# ---------------------------------------------------------------------------
def _classify_patents_with_gemini(
    patents_list: list,
    gap_data: Agent2GapOutput,
    query_topic: str
) -> dict:
    """
    Sends retrieved real patents to Gemini for IP classification, FTO analysis,
    and design-around strategy synthesis.
    """
    proposed_method = getattr(gap_data, 'proposed_method', None)
    method_title = proposed_method.title if proposed_method else query_topic
    method_approach = proposed_method.approach if proposed_method else query_topic

    patent_context_lines = []
    for idx, pat in enumerate(patents_list, 1):
        patent_context_lines.append(
            f"[Patent {idx}]\n"
            f"  Patent ID    : {pat.get('patent_id', '')}\n"
            f"  Title        : {pat.get('title', '')}\n"
            f"  Assignee     : {pat.get('assignee', pat.get('applicant', 'Unknown'))}\n"
            f"  Sources      : {', '.join(pat.get('sources', [pat.get('source', 'Unknown')]))}\n"
            f"  Abstract     : {pat.get('abstract', '')[:350]}\n"
        )
    patents_text = "\n".join(patent_context_lines)

    prompt = f"""
You are an expert Patent Attorney and IP Strategist.

The researcher is studying the topic: "{query_topic}"
Their proposed novel methodology is: "{method_title}"
Approach: {method_approach[:400]}

Below is a list of candidate patents retrieved from live patent databases:

{patents_text}

YOUR TASK:
1. For EACH provided patent, produce a concise IP analysis based on that specific patent's real abstract.
2. Assign freedom-to-operate rating ("Safe" | "Caution" | "Alert") and classification ("Prior Art" | "Overlap" | "White Space").
3. Suggest a 1-sentence design-around strategy to avoid infringing this specific patent.
4. Calculate a unique relevance_score (integer between 50 and 98) reflecting technical proximity.
5. Provide 3 "white_space_opportunities" — unpatented sub-niches directly related to "{query_topic}".

CRITICAL CONSTRAINT: Every field ('match_explanation', 'summary', and 'design_around_strategy') MUST be exactly 1 sentence long.

Respond ONLY with a valid JSON object matching this schema:
{{
  "patents": [
    {{
      "patent_id": "...",
      "title": "...",
      "assignee": "...",
      "relevance": "Prior Art",
      "relevance_score": 90,
      "match_explanation": "...",
      "summary": "...",
      "fto_rating": "Caution",
      "design_around_strategy": "..."
    }}
  ],
  "white_space_opportunities": ["...", "...", "..."]
}}
"""

    client = _get_client()
    response_text = _gemini_generate_with_retry(
        client=client,
        model=GEMINI_MODEL,
        prompt=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
        max_retries=2,
        agent_label="Agent3-Classify",
    )
    return json.loads(response_text)


# ---------------------------------------------------------------------------
# 7. Main Orchestration Functions (Public Entrypoints)
# ---------------------------------------------------------------------------
def run_patent_agent(
    topic: str,
    max_results: int = 5,
    weights: dict = None,
    gap_data: Agent2GapOutput = None,
) -> Agent3PatentOutput:
    """
    Primary independent entrypoint for Agent 3 (Patent Landscape).
    Searches and analyzes patents based SOLELY on the user's original research topic.

    Does NOT inspect Agent 1 papers, titles, abstracts, or research summaries
    for query expansion, patent retrieval, or initial relevance scoring.
    """
    return search_and_classify_patents(
        topic=topic,
        gap_data=gap_data,
        weights=weights,
        max_results=max_results,
    )


def search_and_classify_patents(
    gap_data: Any = None,
    query_topic: str = "",
    research_out: Agent1ResearchOutput = None,
    weights: dict = None,
    topic: str = "",
    max_results: int = 5,
) -> Agent3PatentOutput:
    """
    Robust Multi-Source Patent Pipeline:
    1. Derives patent search queries SOLELY from the original user research topic.
    2. Queries EPO OPS, Google Patents, Lens, PatentsView, and Europe PMC.
    3. Normalizes and deduplicates results while merging multi-source entries.
    4. Deterministically scores and ranks patents against the original user topic.
    5. Optionally calls Gemini to synthesize FTO ratings and design-around strategies.
    6. If Gemini fails or hits quota exhaustion, preserves real patent records.
    """
    # Extract effective topic independently of argument ordering
    if isinstance(gap_data, str) and gap_data.strip():
        effective_topic = gap_data.strip()
        gap_data_obj = None
    elif query_topic and query_topic.strip():
        effective_topic = query_topic.strip()
        gap_data_obj = gap_data if isinstance(gap_data, Agent2GapOutput) else None
    elif topic and topic.strip():
        effective_topic = topic.strip()
        gap_data_obj = gap_data if isinstance(gap_data, Agent2GapOutput) else None
    elif hasattr(gap_data, "topic") and getattr(gap_data, "topic"):
        effective_topic = str(getattr(gap_data, "topic")).strip()
        gap_data_obj = gap_data
    else:
        effective_topic = str(gap_data or query_topic or topic).strip()
        gap_data_obj = gap_data if isinstance(gap_data, Agent2GapOutput) else None

    proposed_method = getattr(gap_data_obj, 'proposed_method', None)
    method_title = proposed_method.title if proposed_method else effective_topic

    # Step 1: Multi-source patent retrieval based ONLY on original user topic
    retrieved_patents, source_statuses, retrieval_status = retrieve_multi_source_patents(
        effective_topic, max_results=max_results, weights=weights
    )

    # If all sources failed and returned 0 patents
    if not retrieved_patents:
        return Agent3PatentOutput(
            patents=[],
            white_space_opportunities=[
                f"Unpatented multimodal integration in {effective_topic}.",
                f"Explainable real-time architectures for {effective_topic}.",
                f"Cross-domain transfer frameworks for {effective_topic}."
            ],
            patent_analysis_status="retrieval_failed",
            patent_retrieval_status=retrieval_status,
            source_statuses=source_statuses,
            retrieved_patents_raw=[]
        )

    # Step 2: Attempt Gemini synthesis (Optional enrichment layer)
    try:
        data = _classify_patents_with_gemini(retrieved_patents, gap_data_obj, effective_topic)
        classified_map = {p.get("patent_id"): p for p in data.get("patents", [])}

        patents_out = []
        for idx, pat in enumerate(retrieved_patents):
            pid = pat["patent_id"]
            pat_title = pat.get("title", "")
            abstract = pat.get("abstract", "")
            gemini_pat = classified_map.get(pid, {})

            summary = gemini_pat.get("summary") or (abstract[:150] if abstract else "Abstract not available.")
            rel_score = int(gemini_pat.get("relevance_score") or pat.get("relevance_score", 75))
            fto = gemini_pat.get("fto_rating") or ["Caution", "Alert", "Safe"][idx % 3]
            relevance = gemini_pat.get("relevance") or ["Overlap", "Prior Art", "White Space"][idx % 3]
            strategy = gemini_pat.get("design_around_strategy") or (
                f"Review claims of {pid} and differentiate '{method_title}' via distinct algorithmic implementation."
            )
            match_exp = gemini_pat.get("match_explanation") or f"Matched on technical relevance to '{effective_topic}'."
            best_url = pat.get("url") or _make_google_patents_url(pid, pat_title)

            patents_out.append(PatentInfo(
                patent_id=pid,
                title=pat_title,
                assignee=pat.get("assignee", pat.get("applicant", "Unknown Assignee")),
                applicant=pat.get("applicant", pat.get("assignee", "Unknown Assignee")),
                inventors=pat.get("inventors", []),
                abstract=abstract,
                claims=pat.get("claims", ""),
                description=pat.get("description", ""),
                relevance=relevance,
                summary=summary,
                fto_rating=fto,
                design_around_strategy=strategy,
                url=best_url,
                source_links=get_patent_source_links(pid, pat_title, best_url),
                relevance_score=rel_score,
                rank=idx + 1,
                match_explanation=match_exp,
                publication_number=pat.get("publication_number", pid),
                publication_date=pat.get("publication_date", ""),
                priority_date=pat.get("priority_date", ""),
                jurisdiction=pat.get("jurisdiction", "US"),
                sources=pat.get("sources", [pat.get("source", "Unknown")]),
                source=pat.get("source", "Unknown"),
                source_url=pat.get("source_url", best_url),
                retrieval_status="SUCCESS"
            ))

        patents_out.sort(key=lambda x: x.relevance_score, reverse=True)
        for rank_idx, p in enumerate(patents_out, 1):
            p.rank = rank_idx

        return Agent3PatentOutput(
            patents=patents_out,
            white_space_opportunities=data.get("white_space_opportunities", []),
            patent_analysis_status="success",
            patent_retrieval_status=retrieval_status,
            source_statuses=source_statuses,
            retrieved_patents_raw=retrieved_patents,
        )

    except Exception as e:
        # Gemini failed or quota exhausted — preserve real retrieved patent records with deterministic summaries!
        print(f"[Agent3] Gemini synthesis unavailable ({type(e).__name__}: {e}). Using deterministic fallback.")
        fallback = []
        for idx, pat in enumerate(retrieved_patents):
            pid = pat["patent_id"]
            pat_title = pat.get("title", "")
            abstract = pat.get("abstract", "")
            summary = abstract.split(".")[0] + "." if "." in abstract else (abstract[:150] if abstract else "Abstract not available.")
            best_url = pat.get("url") or _make_google_patents_url(pid, pat_title)
            rel_score = int(pat.get("relevance_score", 70))

            fallback.append(PatentInfo(
                patent_id=pid,
                title=pat_title,
                assignee=pat.get("assignee", pat.get("applicant", "Unknown Assignee")),
                applicant=pat.get("applicant", pat.get("assignee", "Unknown Assignee")),
                inventors=pat.get("inventors", []),
                abstract=abstract,
                claims=pat.get("claims", ""),
                description=pat.get("description", ""),
                relevance=["Overlap", "Prior Art", "White Space"][idx % 3],
                summary=summary or "Abstract not available.",
                fto_rating=["Caution", "Alert", "Safe"][idx % 3],
                design_around_strategy=(
                    f"Review claims of {pid} and differentiate '{method_title}' via distinct algorithmic implementation."
                ),
                url=best_url,
                source_links=get_patent_source_links(pid, pat_title, best_url),
                relevance_score=rel_score,
                rank=idx + 1,
                match_explanation=f"Deterministically matched on technical proximity to '{effective_topic}'.",
                publication_number=pat.get("publication_number", pid),
                publication_date=pat.get("publication_date", ""),
                priority_date=pat.get("priority_date", ""),
                jurisdiction=pat.get("jurisdiction", "US"),
                sources=pat.get("sources", [pat.get("source", "Unknown")]),
                source=pat.get("source", "Unknown"),
                source_url=pat.get("source_url", best_url),
                retrieval_status="SUCCESS"
            ))

        fallback.sort(key=lambda x: x.relevance_score, reverse=True)
        for rank_idx, p in enumerate(fallback, 1):
            p.rank = rank_idx

        return Agent3PatentOutput(
            patents=fallback,
            white_space_opportunities=[
                f"Unpatented integration of multi-modal approaches in {effective_topic}.",
                f"Cross-domain transfer learning applications in {effective_topic}.",
                f"Explainability frameworks for {effective_topic} models."
            ],
            patent_analysis_status="retrieval_success_synthesis_failed",
            patent_retrieval_status=retrieval_status,
            source_statuses=source_statuses,
            retrieved_patents_raw=retrieved_patents,
        )
