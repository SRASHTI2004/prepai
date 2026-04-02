import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def make_chunk_id(content: str, source: str) -> str:
    key = f"{source}::{content[:80]}"
    return hashlib.md5(key.encode()).hexdigest()


def chunk_documents(docs: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["content"])
        for i, split in enumerate(splits):
            chunk = {
                "id": make_chunk_id(split, doc["metadata"]["source"]),
                "content": split,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": i,
                    "total_chunks": len(splits),
                }
            }
            chunks.append(chunk)

    print(f"\nChunking report:")
    print(f"  Total chunks: {len(chunks)}")
    if chunks:
        sizes = [len(c["content"]) for c in chunks]
        print(f"  Avg size: {sum(sizes)//len(sizes)} chars")
        print(f"  Min: {min(sizes)}  Max: {max(sizes)}")
    return chunks