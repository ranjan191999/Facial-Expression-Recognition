# stage2/embeddings_store.py
"""
ReviewVectorStore: sentence-transformers + FAISS persistent store.

Usage:
    from stage2.embeddings_store import ReviewVectorStore
    store = ReviewVectorStore(persist_dir="stage2")
    store.add_texts(["a", "b"])
    results = store.search("query", k=5)
"""

import os
import json
from typing import List, Tuple, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError("Install sentence-transformers: pip install sentence-transformers") from e

try:
    import faiss
except Exception as e:
    raise ImportError("Install faiss-cpu: pip install faiss-cpu") from e

class ReviewVectorStore:
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 persist_dir: str = "stage2"):
        """
        persist_dir will contain:
          - faiss.index  (binary)
          - texts.json   (list of texts)
        """
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index_path = os.path.join(self.persist_dir, "faiss.index")
        self.texts_path = os.path.join(self.persist_dir, "texts.json")

        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.dim = None
        self.texts: List[str] = []

        # load existing texts
        if os.path.exists(self.texts_path):
            try:
                with open(self.texts_path, "r", encoding="utf-8") as f:
                    self.texts = json.load(f)
            except Exception:
                self.texts = []

        # if FAISS exists, load it
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                self.dim = self.index.d
            except Exception:
                # corrupted or incompatible index: start fresh
                self.index = None
                self.dim = None

    def _init_index(self, dim: int):
        if self.index is None:
            # use inner-product on normalized embeddings (cosine similarity)
            self.dim = dim
            self.index = faiss.IndexFlatIP(dim)

    def add_texts(self, texts: List[str]) -> None:
        """
        Add new texts to index and persist.
        """
        if not texts:
            return
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        embs = np.asarray(embeddings).astype("float32")
        self._init_index(embs.shape[1])
        self.index.add(embs)
        self.texts.extend(texts)
        # persist
        faiss.write_index(self.index, self.index_path)
        with open(self.texts_path, "w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False, indent=2)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Return list of (text, score) sorted by score desc.
        Score is cosine-like inner product in [-1,1].
        """
        if self.index is None or len(self.texts) == 0:
            return []
        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        D, I = self.index.search(q_emb, k)
        results: List[Tuple[str, float]] = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            results.append((self.texts[idx], float(score)))
        return results

    def get_all_texts(self) -> List[str]:
        return list(self.texts)

    def persist_paths(self) -> Tuple[str, str]:
        return self.index_path, self.texts_path

if __name__ == "__main__":
    # quick demo
    store = ReviewVectorStore()
    store.add_texts(["I love this.", "I hate this.", "This is okay."])
    print(store.search("love", k=3))