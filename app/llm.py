from langchain_groq import ChatGroq
from app.config import GROQ_API_KEY, RAG_MODEL, EVAL_MODEL


def get_rag_llm():
    return ChatGroq(
        model=RAG_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.1,
        max_tokens=1024,
    )


def get_eval_llm():
    return ChatGroq(
        model=EVAL_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=512,
    )