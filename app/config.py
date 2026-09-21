import os 
from dotenv import load_dotenv

# load environment variables from dot env file
load_dotenv()

class Settings: 
    # EMBEDDINGS
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # VECTOR DB
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "entriprise_rag"

    # REASONING ENGINE
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL =  "llama-3.3-70b-versatile"
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")

settings = Settings()