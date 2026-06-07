"""
src/embeddings.py — Embeddings + FAISS vector store (Step 3)
"""
import os
import faiss
import numpy as np
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL, VECTORSTORE_DIR

INDEX_PATH  = os.path.join(VECTORSTORE_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(VECTORSTORE_DIR, "chunks.pkl")

_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Embeddings] Loading model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_chunks(chunks: list) -> np.ndarray:
    """Embed a list of text chunks. Returns float32 numpy array (N, 384)."""
    model = get_model()
    print(f"[Embeddings] Embedding {len(chunks)} chunks...")
    vectors = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return vectors.astype("float32")


def build_index(chunks: list) -> faiss.IndexFlatL2:
    """
    Build a FAISS L2 index from chunks, save to disk.
    IndexFlatL2 = exact nearest-neighbor search (no approximation).
    Fine for PDFs (<100k chunks). Switch to IndexIVFFlat for millions of chunks.
    """
    vectors = embed_chunks(chunks)
    dim = vectors.shape[1]  # 384 for MiniLM
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"[Embeddings] Index saved: {index.ntotal} vectors, dim={dim}")
    return index, chunks


def load_index():
    """Load a previously saved FAISS index + chunks from disk."""
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("No index found. Upload a PDF first.")
    index  = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    print(f"[Embeddings] Index loaded: {index.ntotal} vectors")
    return index, chunks


def index_exists() -> bool:
    return os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)
