import os
import sys
from dotenv import load_dotenv
from google import genai

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', '.env')
load_dotenv(dotenv_path=_env_path)

api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key loaded: {'YES' if api_key else 'NO'}")
if api_key:
    print(f"Key ends with: {api_key[-5:]}")

try:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents='Write a 1-sentence hello world.'
    )
    print("API CALL SUCCESS:")
    print(response.text)
except Exception as e:
    print("API CALL FAILED:")
    print(repr(e))
