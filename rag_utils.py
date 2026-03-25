# rag_utils.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------- Create Embeddings --------
def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return np.array(embeddings).astype("float32")

# -------- Build FAISS Index --------
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# -------- Search --------
def search_faiss(query, chunks, index, top_k=3):
    """
    Returns top_k relevant chunks with page info
    """
    query_vector = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, top_k)
    results = []
    for i in indices[0]:
        results.append(chunks[i])  # each chunk is {"chunk":..., "page":...}
    return results