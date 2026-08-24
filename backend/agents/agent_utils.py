import time
import re
import math
import os
from typing import Optional, Any
from google import genai
from google.genai import errors, types
from dotenv import load_dotenv

# Ensure environment variables are loaded
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path=_env_path, override=True)

from schemas import GeminiQuotaExhaustedError, GeminiError

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

_gemini_request_counter: list = [0]
_gemini_client: Optional[genai.Client] = None


def get_gemini_model() -> str:
    """Returns the configured Gemini model name from environment or fallback default."""
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def get_gemini_client(api_key: Optional[str] = None) -> Optional[genai.Client]:
    """Returns a singleton or newly initialized google.genai.Client instance."""
    global _gemini_client
    if api_key:
        return genai.Client(api_key=api_key)
    if _gemini_client is None:
        key = os.getenv("GEMINI_API_KEY")
        if key and key != "PASTE_YOUR_GEMINI_API_KEY_HERE" and key.strip() != "":
            _gemini_client = genai.Client(api_key=key)
    return _gemini_client


def _is_quota_exhausted(error_msg: str) -> bool:
    """Returns True if the error message indicates DAILY quota exhaustion."""
    msg = str(error_msg).lower()
    return any(sig in msg for sig in _QUOTA_SIGNALS)


is_quota_exhausted = _is_quota_exhausted


def execute_gemini_with_retry(
    prompt: str,
    config: Optional[Any] = None,
    model: Optional[str] = None,
    max_retries: int = 3,
    agent_label: str = "Agent",
    client: Optional[Any] = None,
) -> str:
    """
    Centralized execution helper for Gemini API calls with rate-limit and transient retry logic.
    - Quota exhaustion (daily limits): raises GeminiQuotaExhaustedError immediately without retrying.
    - Transient errors (429, 500, 502, 503, 504): retries with exponential backoff.
    """
    if client is None:
        client = get_gemini_client()
        if client is None:
            raise GeminiError("GEMINI_API_KEY is not configured or invalid in backend/.env")

    if model is None:
        model = get_gemini_model()

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
            return response.text.strip() if hasattr(response, "text") and response.text else ""
        except Exception as e:
            msg = str(e)
            msg_lower = msg.lower()

            # Quota exhaustion: do not waste budget retrying
            if _is_quota_exhausted(msg_lower):
                print(
                    f"[Gemini] QUOTA EXHAUSTED on {agent_label} request #{req_num}: {msg[:160]}. "
                    f"Not retrying (daily limit reached)."
                )
                raise GeminiQuotaExhaustedError(
                    f"Gemini daily quota exhausted during {agent_label} call: {msg}"
                ) from e

            # Transient errors: retry with backoff
            is_transient = any(
                code in msg_lower
                for code in [
                    "429",
                    "500",
                    "502",
                    "503",
                    "504",
                    "unavailable",
                    "internal",
                    "too many requests",
                ]
            )
            if is_transient and attempt < max_retries - 1:
                # If error specifies retry seconds, honor it
                match = re.search(r"Please retry in (\d+(\.\d+)?)s", msg)
                if match:
                    wait = math.ceil(float(match.group(1))) + 2
                else:
                    wait = (2 ** attempt) * 2  # 2s, 4s, 8s

                print(
                    f"[Gemini] Transient error on {agent_label} request #{req_num} "
                    f"attempt {attempt + 1}/{max_retries}: {msg[:100]}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
                last_exc = e
            else:
                last_exc = e
                break

    if last_exc:
        raise last_exc
    raise GeminiError("Gemini API call failed. All retries exhausted.")


def generate_content_with_retry(client, contents, config=None, retries=4, backoff=8):
    """
    Legacy wrapper for backwards compatibility with generate_content calls.
    """
    return execute_gemini_with_retry(
        prompt=contents,
        config=config,
        model=get_gemini_model(),
        max_retries=retries,
        agent_label="Legacy",
        client=client,
    )
