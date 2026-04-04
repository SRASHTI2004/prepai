import time
import cohere
from qdrant_client import QdrantClient
from langchain_community.embeddings import FastEmbedEmbeddings
from rank_bm25 import BM25Okapi
from app.config import (
    QDRANT_URL, QDRANT_API_KEY,
    COLLECTION_NAME, TOP_K,
    RERANK_TOP_N, COHERE_API_KEY
)


def get_all_chunks() -> list:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        with_payload=True,
        with_vectors=False
    )
    return results[0]


def dense_search(query: str, embedder) -> list:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    query_vector = embedder.embed_query(query)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K
    ).points
    return results


def bm25_search(query: str, all_chunks: list) -> list:
    texts = [c.payload["content"] for c in all_chunks]
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:TOP_K]
    return [all_chunks[i] for i in top_indices]


def reciprocal_rank_fusion(dense_results, bm25_results) -> list:
    k = 60
    scores = {}
    doc_map = {}

    for rank, doc in enumerate(dense_results):
        doc_id = str(doc.id)
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        doc_map[doc_id] = doc

    for rank, doc in enumerate(bm25_results):
        doc_id = str(doc.id)
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        doc_map[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    # Return more candidates for reranker to work with
    return [doc_map[id] for id in sorted_ids[:TOP_K]]


def cohere_rerank(query: str, chunks: list) -> list:
    """
    NEW — Cross encoder reranking using Cohere.
    This is the step that makes retrieval much more precise.
    """
    co = cohere.Client(COHERE_API_KEY)

    documents = [c.payload["content"] for c in chunks]

    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=RERANK_TOP_N
    )

    # Return reranked chunks in order
    reranked = [chunks[r.index] for r in response.results]
    return reranked


def hybrid_search(query: str) -> tuple[list, dict]:
    """
    Full pipeline:
    1. Dense search (vector)
    2. BM25 search (keyword)
    3. RRF fusion
    4. Cohere reranking
    Returns chunks + timing metadata
    """
    embedder = FastEmbedEmbeddings()
    timings = {}

    # Dense search
    t0 = time.time()
    all_chunks = get_all_chunks()
    dense_results = dense_search(query, embedder)
    timings["dense_ms"] = round((time.time() - t0) * 1000)

    # BM25 search
    t0 = time.time()
    bm25_results = bm25_search(query, all_chunks)
    timings["bm25_ms"] = round((time.time() - t0) * 1000)

    # RRF Fusion
    t0 = time.time()
    fused = reciprocal_rank_fusion(dense_results, bm25_results)
    timings["fusion_ms"] = round((time.time() - t0) * 1000)

    # Cohere Reranking
    t0 = time.time()
    final_results = cohere_rerank(query, fused)
    timings["rerank_ms"] = round((time.time() - t0) * 1000)

    timings["total_ms"] = sum(timings.values())

    print(f"  Retrieval timings: {timings}")
    return final_results, timings