from qdrant_client import QdrantClient
from langchain_community.embeddings import FastEmbedEmbeddings
from rank_bm25 import BM25Okapi
from app.config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, TOP_K, RERANK_TOP_N


def get_all_chunks() -> list[dict]:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        with_payload=True,
        with_vectors=False
    )
    return results[0]


def dense_search(query: str, embedder) -> list[dict]:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    query_vector = embedder.embed_query(query)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K
    ).points
    return results


def bm25_search(query: str, all_chunks: list) -> list[dict]:
    texts = [c.payload["content"] for c in all_chunks]
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
    return [all_chunks[i] for i in top_indices]


def reciprocal_rank_fusion(dense_results, bm25_results) -> list[dict]:
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
    return [doc_map[id] for id in sorted_ids[:RERANK_TOP_N]]


def hybrid_search(query: str) -> list[dict]:
    embedder = FastEmbedEmbeddings()
    all_chunks = get_all_chunks()

    print("  Running dense search...")
    dense_results = dense_search(query, embedder)

    print("  Running BM25 search...")
    bm25_results = bm25_search(query, all_chunks)

    print("  Fusing results...")
    final_results = reciprocal_rank_fusion(dense_results, bm25_results)

    return final_results