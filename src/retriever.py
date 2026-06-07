"""
src/retriever.py — Top-k similarity search (Step 4)
"""
import numpy as np
import time
from src.embeddings import get_model, load_index, build_index, index_exists
from src.config import TOP_K


class Retriever:
    def __init__(self):
        self.index  = None
        self.chunks = None

    def load_or_build(self, chunks=None):
        """Load existing index or build from new chunks."""
        if chunks:
            self.index, self.chunks = build_index(chunks)
        elif index_exists():
            self.index, self.chunks = load_index()
        else:
            raise RuntimeError("No index and no chunks provided.")

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        """
        Embed query, search FAISS, return top-k chunks with scores.

        Why L2 distance?
          Lower = more similar. We negate it to get a similarity score.
          MiniLM embeddings are not normalized by default; L2 is appropriate.
          (If you normalize embeddings first, cosine ≈ L2 — they're equivalent.)

        k tuning guide:
          k=2 → precise, risks missing context
          k=4 → good default for single-topic questions  ← default
          k=6 → better for multi-part / comparative questions
          k=8 → rarely better; adds noise and confuses the LLM
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load_or_build() first.")

        model = get_model()
        t0 = time.time()
        query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(query_vec, k)
        latency_ms = (time.time() - t0) * 1000

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "chunk": self.chunks[idx],
                "score": float(-dist),   # negate: higher = more similar
                "index": int(idx),
                "latency_ms": round(latency_ms, 1),
            })
        return results
