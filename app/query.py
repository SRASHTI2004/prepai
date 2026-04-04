import time
from app.retrieval import hybrid_search
from app.llm import get_rag_llm
from app.tracer import create_trace, log_retrieval, log_generation


def query(question: str, user_id: str = "default") -> dict:
    print(f"\nQuery: {question}")

    # Start trace
    trace = create_trace(name="techprep_query", user_id=user_id)
    trace_id = trace.id

    # 1. Retrieval
    t0 = time.time()
    chunks, retrieval_timings = hybrid_search(question)
    retrieval_latency = round((time.time() - t0) * 1000)

    log_retrieval(trace, question, chunks, retrieval_latency)

    if not chunks:
        return {
            "answer": "I don't have enough information to answer this.",
            "sources": [],
            "retrieval_latency_ms": retrieval_latency,
            "generation_latency_ms": 0,
            "total_latency_ms": retrieval_latency,
            "trace_id": trace_id
        }

    # 2. Build context
    context_parts = []
    sources = []
    for i, chunk in enumerate(chunks):
        context_parts.append(f"[{i+1}] {chunk.payload['content']}")
        sources.append(chunk.payload.get("filename", "unknown"))

    context = "\n\n".join(context_parts)

    # 3. Prompt
    prompt = f"""You are TechPrep AI — a technical interview preparation assistant.

Answer the question using ONLY the information in the context below.
Do NOT hallucinate or add information not present in the context.
Be concise, clear, and technically accurate.

Answer format:
- Direct and confident tone
- Use bullet points for lists
- Keep it under 150 words unless the question requires more detail
- If context does not have enough info say "I don't have enough data on this"

Context:
{context}

Question: {question}

Answer:"""

    # 4. Generate
    llm = get_rag_llm()
    t0 = time.time()
    response = llm.invoke(prompt)
    generation_latency = round((time.time() - t0) * 1000)
    answer = response.content

    log_generation(trace, prompt, answer, generation_latency)


    total_latency = retrieval_latency + generation_latency
    print(f"  Total latency: {total_latency}ms")

    return {
        "answer": answer,
        "sources": list(set(sources)),
        "retrieval_latency_ms": retrieval_latency,
        "generation_latency_ms": generation_latency,
        "total_latency_ms": total_latency,
        "retrieval_timings": retrieval_timings,
        "trace_id": trace_id
    }