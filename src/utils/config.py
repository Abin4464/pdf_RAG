import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Free local model
    LLM_MODEL = "llama-3.1-8b-instant"  # Free Groq model
    TOP_K_RESULTS = 3
