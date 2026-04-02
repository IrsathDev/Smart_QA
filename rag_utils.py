# rag_utils.py
# Step 2 Upgrade: Hybrid Search — BM25 + Vector + RRF Fusion

import os
import requests
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"

HEADERS = {
    "Authorization": f"Bearer {NVIDIA_API_KEY}",
    "Content-Type": "application/json"
}

# ── ChromaDB client (in-memory, Railway-friendly) ────────────────────────────
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))


# ── NVIDIA NIM Embeddings API ────────────────────────────────────────────────
def get_nvidia_embeddings(texts: list[str]) -> list[list[float]]:
    """Calls NVIDIA NIM API to embed texts. Batches 50 at a time."""
    all_embeddings = []
    batch_size = 50

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        payload = {
            "model": EMBED_MODEL,
            "input": batch,
            "input_type": "passage",
            "encoding_format": "float",
            "truncate": "END"
        }
        response = requests.post(NVIDIA_EMBED_URL, headers=HEADERS, json=payload)
        response.raise_for_status()

        data = response.json()["data"]
        data_sorted = sorted(data, key=lambda x: x["index"])
        all_embeddings.extend([item["embedding"] for item in data_sorted])

    return all_embeddings


# ── Build index: ChromaDB (vector) + BM25 (keyword) ─────────────────────────
def build_hybrid_index(chunks: list[str], collection_name: str = "documents"):
    """
    Builds TWO indexes from the same chunks:
      1. ChromaDB collection  → for semantic/vector search
      2. BM25Okapi object     → for keyword search

    Returns: (chroma_collection, bm25_index)
    Both are kept in memory — no disk needed.
    """
    # ── 1. ChromaDB (vector) ──────────────────────────────────────────────────
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"📡 Embedding {len(chunks)} chunks via NVIDIA API...")
    embeddings = get_nvidia_embeddings(chunks)

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"✅ ChromaDB vector index built — {len(chunks)} chunks.")

    # ── 2. BM25 (keyword) ────────────────────────────────────────────────────
    # Tokenise by splitting on whitespace (simple, fast, no extra deps)
    tokenised = [chunk.lower().split() for chunk in chunks]
    bm25_index = BM25Okapi(tokenised)
    print(f"✅ BM25 keyword index built — {len(chunks)} chunks.")

    return collection, bm25_index


# ── Reciprocal Rank Fusion ───────────────────────────────────────────────────
def reciprocal_rank_fusion(
    bm25_hits: list[str],
    vector_hits: list[str],
    all_chunks: list[str],
    k: int = 60
) -> list[str]:
    """
    Merges two ranked lists using the RRF formula:
        score(chunk) = 1/(k + rank_bm25) + 1/(k + rank_vector)

    k=60 is the standard default from the original RRF paper.
    Higher k → reduces the impact of very high ranks.

    Returns chunks sorted by descending RRF score (best first).
    """
    scores: dict[str, float] = {}

    for rank, chunk in enumerate(bm25_hits):
        scores[chunk] = scores.get(chunk, 0.0) + 1.0 / (k + rank + 1)

    for rank, chunk in enumerate(vector_hits):
        scores[chunk] = scores.get(chunk, 0.0) + 1.0 / (k + rank + 1)

    # Sort by score descending
    sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in sorted_chunks]


# ── Hybrid Search (main entry point) ─────────────────────────────────────────
def hybrid_search(
    query: str,
    all_chunks: list[str],
    chroma_collection,
    bm25_index,
    top_k: int = 5
) -> list[str]:
    """
    Step 1 — BM25 keyword search:
        Score every chunk by exact token overlap with the query.

    Step 2 — Vector semantic search:
        Embed the query via NVIDIA API, search ChromaDB by cosine similarity.

    Step 3 — RRF fusion:
        Combine both ranked lists into one final ranking.

    Returns the top_k best chunks to use as LLM context.
    """
    fetch_k = top_k * 3   # fetch 3x more from each, then fuse & trim

    # ── BM25 ──────────────────────────────────────────────────────────────────
    query_tokens = query.lower().split()
    bm25_scores = bm25_index.get_scores(query_tokens)

    # Pair each chunk with its BM25 score, sort descending
    bm25_ranked = sorted(
        zip(all_chunks, bm25_scores),
        key=lambda x: x[1],
        reverse=True
    )
    bm25_hits = [chunk for chunk, score in bm25_ranked[:fetch_k] if score > 0]

    # ── Vector ────────────────────────────────────────────────────────────────
    query_embedding = get_nvidia_embeddings([query])[0]
    vector_results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(fetch_k, len(all_chunks)),
        include=["documents"]
    )
    vector_hits = vector_results["documents"][0] if vector_results["documents"] else []

    # ── RRF fusion ────────────────────────────────────────────────────────────
    fused = reciprocal_rank_fusion(bm25_hits, vector_hits, all_chunks)

    print(f"🔍 BM25 hits: {len(bm25_hits)} | Vector hits: {len(vector_hits)} | After RRF: {len(fused)}")

    return fused[:top_k]