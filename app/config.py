import os
from dotenv import load_dotenv

load_dotenv()

# Groq API
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
RAG_MODEL        = "llama-3.1-8b-instant"
EVAL_MODEL       = "llama-3.3-70b-versatile"

# Qdrant Cloud
QDRANT_URL       = os.getenv("QDRANT_URL")
QDRANT_API_KEY   = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME  = "rag_docs"

# Chunking
CHUNK_SIZE       = 400
CHUNK_OVERLAP    = 50

# Retrieval
TOP_K            = 5
RERANK_TOP_N = 5
