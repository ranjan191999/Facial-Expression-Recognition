# stage2/rag_langchain.py
"""
Simple RAG helper:
 - retrieves top-k reviews from FAISS (ReviewVectorStore)
 - returns snippets, scores, sentiments, and an optional summary

Usage:
    from stage2.embeddings_store import ReviewVectorStore
    from stage2.rag_langchain import rag_query
    store = ReviewVectorStore()
    rag_query(store, "what are positive reviews about happiness?", k=5)
"""
from typing import Dict, Any, List
from transformers import pipeline

try:
    from .embeddings_store import ReviewVectorStore
    from .sentiment import get_sentiment
except Exception:
    # relative import fallback if used as script
    from stage2.embeddings_store import ReviewVectorStore
    from stage2.sentiment import get_sentiment

# small summarizer model (local fallback)
_SUMMARIZER = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text: str, max_length: int = 120) -> str:
    if not text.strip():
        return ""
    out = _SUMMARIZER(text, max_length=max_length, min_length=20, do_sample=False)
    return out[0]["summary_text"]

def rag_query(store: ReviewVectorStore, query: str, k: int = 5, summarize: bool = True) -> Dict[str, Any]:
    """
    Returns:
      {
        "query": query,
        "retrieved": [ {"text":..., "score":..., "sentiment": {...}}, ... ],
        "summary": "..." or ""
      }
    """
    results = store.search(query, k=k)
    retrieved: List[Dict[str, Any]] = []
    texts = []
    for text, score in results:
        sent = get_sentiment(text)
        retrieved.append({"text": text, "score": score, "sentiment": sent})
        texts.append(text)
    summary = ""
    if summarize and texts:
        big = "\n\n".join(texts)
        # summarizer has token limit; trim if necessary
        if len(big) > 2000:
            big = big[:2000]
        try:
            summary = summarize_text(big)
        except Exception:
            summary = ""
    return {"query": query, "retrieved": retrieved, "summary": summary}

if __name__ == "__main__":
    # demo flow
    store = ReviewVectorStore()
    # add some example texts if empty
    if len(store.get_all_texts()) == 0:
        store.add_texts([
            "Absolutely loved it! The experience made my day.",
            "Felt a bit let down; expected more.",
            "Extremely frustrating — this needs to be fixed.",
            "Unexpectedly great — exceeded my expectations!"
        ])
    res = rag_query(store, "positive experiences", k=3)
    import pprint; pprint.pprint(res)