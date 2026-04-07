from app.loaders import load_directory
from app.chunkers import chunk_documents
from app.config import (
    QDRANT_URL, QDRANT_API_KEY,
    COLLECTION_NAME
)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_community.embeddings import FastEmbedEmbeddings
import hashlib


def ingest(data_dir: str = "./data/raw"):
    # 1. Load documents
    docs = load_directory(data_dir)
    if not docs:
        print("No documents found. Add .txt or .md files to data/raw/")
        return

    # 2. Chunk
    chunks = chunk_documents(docs)

    # 3. Connect to Qdrant
    print("\nConnecting to Qdrant...")
    embedder = FastEmbedEmbeddings()

    # 4. Embed all chunks
    print("Embedding chunks...")
    texts = [c["content"] for c in chunks]
    embeddings = embedder.embed_documents(texts)

    # 5. Store in Qdrant
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    # Create collection if not exists
    try:
        client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists")
    except Exception:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=len(embeddings[0]),
                distance=Distance.COSINE
            )
        )
        print(f"Created collection '{COLLECTION_NAME}'")

    # Upsert in batches
    points = [
        PointStruct(
            id=abs(int(hashlib.md5(
                c["id"].encode()).hexdigest()[:8], 16)),
            vector=embeddings[i],
            payload={
                "content": c["content"],
                **c["metadata"]
            }
        )
        for i, c in enumerate(chunks)
    ]

    print(f"\nUploading {len(points)} chunks in batches...")
    batch_size = 50
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
        print(f"  Uploaded batch {i//batch_size + 1}")

    print(f"\nDone! {len(points)} chunks stored in Qdrant Cloud.")


if __name__ == "__main__":
    ingest()