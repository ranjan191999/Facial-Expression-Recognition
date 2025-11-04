# api/main.py
from fastapi import FastAPI, Form
from stage2.embeddings_store import ReviewVectorStore
from stage2.generate_reviews import generate_review
from stage2.rag_langchain import rag_query

app = FastAPI(title="Emotion-Based Review API", description="Stage 2 - FAISS + RAG demo", version="1.0")

store = ReviewVectorStore(persist_dir="stage2")

@app.get("/")
def home():
    return {"message": "Emotion Review API is running!"}


@app.post("/add_reviews_from_emotion")
def add_reviews(emotion: str, k: int = 5):
    """
    Generate k synthetic reviews for the given emotion and store in FAISS.
    """
    reviews = generate_review(emotion, k=k)
    store.add_texts(reviews)
    return {"emotion": emotion, "added_reviews": len(reviews)}


@app.post("/query_reviews")
def query_reviews(q: str = Form(...), k: int = 5):
    """
    Query the stored reviews and return summarized results.
    """
    result = rag_query(store, q, k=k)
    return {
        "query": q,
        "summary": result["summary"],
        "retrieved_reviews": result["retrieved"]
    }