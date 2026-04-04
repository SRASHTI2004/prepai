import os
from dotenv import load_dotenv

load_dotenv()

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
RAG_MODEL    = "llama-3.1-8b-instant"
EVAL_MODEL   = "llama-3.3-70b-versatile"

# Qdrant
QDRANT_URL        = os.getenv("QDRANT_URL")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME   = "techprep_ai"

# Cohere
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Langfuse
LANGFUSE_PUBLIC_KEY  = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY  = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST        = "https://cloud.langfuse.com"

# Chunking
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

# Retrieval
TOP_K        = 10
RERANK_TOP_N = 5