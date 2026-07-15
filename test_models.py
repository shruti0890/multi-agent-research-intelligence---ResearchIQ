import os
from dotenv import load_dotenv
from google import genai

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', '.env')
load_dotenv(dotenv_path=_env_path)
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
for model in client.models.list():
    print(model.name)
