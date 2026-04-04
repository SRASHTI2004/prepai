from langfuse import Langfuse
from app.config import (
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_HOST
)

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST
)


def create_trace(name: str, user_id: str = "default"):
    return langfuse.trace(
        name=name,
        user_id=user_id
    )


def log_retrieval(trace, query: str, chunks: list, latency_ms: float):
    trace.span(
        name="retrieval",
        input={"query": query},
        output={"num_chunks": len(chunks)},
        metadata={"latency_ms": latency_ms}
    )


def log_generation(trace, prompt: str, answer: str, latency_ms: float):
    trace.generation(
        name="generation",
        input=prompt,
        output=answer,
        metadata={"latency_ms": latency_ms}
    )


def log_score(trace_id: str, score: float, comment: str = ""):
    langfuse.score(
        trace_id=trace_id,
        name="user_feedback",
        value=score,
        comment=comment
    )