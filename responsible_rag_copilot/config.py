import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Read OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

# Default model to use
CHAT_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"