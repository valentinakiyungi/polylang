import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

app = FastAPI(title="Cross-Lingual Semantic Cache & Hybrid RAG Engine")

# Load a lightweight, state-of-the-art multilingual embedding model
print("Loading multilingual embedding model (paraphrase-multilingual-MiniLM-L12-v2)...")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Simulated Document Knowledge Base (Multilingual support: English, Swahili, Russian)
DOCUMENTS = [
    {"id": 1, "text": "High-throughput AI systems utilize Rust and PyTorch for low-latency inference."},
    {"id": 2, "text": "Mfumo wa akili bandia wenye kasi ya juu hutumia Rust na PyTorch kwa usindikaji."},
    {"id": 3, "text": "Высокопроизводительные ИИ-системы используют Rust и PyTorch для быстрой инференции."},
    {"id": 4, "text": "Retrieval-Augmented Generation (RAG) combines vector search with LLM generation."},
    {"id": 5, "text": "Semantic caching reduces API latency by matching conceptually similar user queries."}
]

# Initialize BM25 Sparse Index
tokenized_corpus = [doc["text"].lower().split() for doc in DOCUMENTS]
bm25 = BM25Okapi(tokenized_corpus)

# Precompute document embeddings for dense vector search
doc_texts = [doc["text"] for doc in DOCUMENTS]
doc_embeddings = embedder.encode(doc_texts, convert_to_tensor=True)

# In-Memory Cross-Lingual Semantic Cache
# Structure: { query_text: {"embedding": tensor, "response": str, "hit_count": int} }
SEMANTIC_CACHE = {}
SIMILARITY_THRESHOLD = 0.85

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def semantic_cache_rag(req: QueryRequest):
    query_embedding = embedder.encode(req.query, convert_to_tensor=True)
    
    # 1. Check Semantic Cache
    cache_hit = None
    best_score = 0.0
    
    for cached_q, cache_data in SEMANTIC_CACHE.items():
        score = util.cos_sim(query_embedding, cache_data["embedding"]).item()
        if score > best_score:
            best_score = score
            cache_hit = cached_q

    if cache_hit and best_score >= SIMILARITY_THRESHOLD:
        SEMANTIC_CACHE[cache_hit]["hit_count"] += 1
        return {
            "status": "cache_hit",
            "similarity_score": round(best_score, 4),
            "matched_cache_query": cache_hit,
            "response": SEMANTIC_CACHE[cache_hit]["response"]
        }

    # 2. Cache Miss: Execute Hybrid Retrieval (Dense Vector + BM25 Keyword)
    # Dense Search Scores
    dense_scores = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()
    
    # Sparse BM25 Scores
    tokenized_query = req.query.lower().split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    # Normalize BM25 scores between 0 and 1
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()

    # Hybrid Combination (Alpha blending: 0.6 dense, 0.4 sparse)
    hybrid_scores = (0.6 * dense_scores) + (0.4 * bm25_scores)
    best_doc_idx = np.argmax(hybrid_scores)
    
    retrieved_doc = DOCUMENTS[best_doc_idx]["text"]
    generated_response = f"RAG Generated Answer based on retrieved context: '{retrieved_doc}'"

    # Store in Semantic Cache
    SEMANTIC_CACHE[req.query] = {
        "embedding": query_embedding,
        "response": generated_response,
        "hit_count": 1
    }

    return {
        "status": "cache_miss",
        "hybrid_score": round(float(hybrid_scores[best_doc_idx]), 4),
        "retrieved_document": retrieved_doc,
        "response": generated_response
    }

@get("/")
def root():
    return {"message": "Cross-Lingual Semantic Cache & Hybrid RAG Engine is running."}
