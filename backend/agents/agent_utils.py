import time
import re
import math
import os
from google.genai import errors

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

def generate_content_with_retry(client, contents, config=None, retries=4, backoff=8):
    """
    Wraps the generate_content API call with automatic retry on 429 rate limit errors.
    Dynamically parses the required retry delay from the Google API error message.
    """
    last_exception = None
    for attempt in range(retries):
        try:
            if config:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=config
                )
            else:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents
                )
            return response
        except errors.APIError as e:
            last_exception = e
            if e.code == 429 or any(w in str(e).lower() for w in ("exhausted", "quota", "rate limit")):
                # Parse custom retry delay from error message
                msg = str(e)
                match = re.search(r"Please retry in (\d+(\.\d+)?)s", msg)
                if match:
                    wait_time = math.ceil(float(match.group(1))) + 2
                else:
                    wait_time = backoff * (attempt + 1)
                
                print(f"  [API Rate Limit] 429/Quota error: {e.message if hasattr(e, 'message') else str(e)}. Sleeping for {wait_time}s before retrying (Attempt {attempt+1}/{retries})...")
                time.sleep(wait_time)
            else:
                raise e
        except Exception as e:
            last_exception = e
            if "429" in str(e) or any(w in str(e).lower() for w in ("exhausted", "quota", "rate limit")):
                msg = str(e)
                match = re.search(r"Please retry in (\d+(\.\d+)?)s", msg)
                if match:
                    wait_time = math.ceil(float(match.group(1))) + 2
                else:
                    wait_time = backoff * (attempt + 1)
                
                print(f"  [API Rate Limit] 429 exception: {e}. Sleeping for {wait_time}s before retrying (Attempt {attempt+1}/{retries})...")
                time.sleep(wait_time)
            else:
                raise e
    
    if last_exception:
        raise last_exception
    raise ValueError("Gemini API Rate limits exceeded. All retry attempts failed.")
