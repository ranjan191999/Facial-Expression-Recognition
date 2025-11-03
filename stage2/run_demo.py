import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stage2.generate_reviews import generate_review
from stage2.embeddings_store import ReviewVectorStore
from stage2.rag_langchain import rag_query

# 1️⃣ Create or load your FAISS vector store
store = ReviewVectorStore(persist_dir="stage2")

# 2️⃣ Generate some synthetic reviews (like from your model’s emotion output)
emotion = "happy"
reviews = generate_review(emotion, k=5)
store.add_texts(reviews)
print(f"✅ Added {len(reviews)} reviews for emotion: {emotion}\n")

# 3️⃣ Query RAG
query = "What do happy users say?"
out = rag_query(store, query, k=5)

print(f"🔍 Query: {query}\n")
print(f"📊 Summary:\n{out['summary']}\n")
print("💬 Retrieved Reviews:")
for r in out["retrieved"]:
    s = r["sentiment"]
    print(f"  - [{s['label']} {s['score']:.2f}] {r['text']}")