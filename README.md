# 🌐 Cross-Lingual Semantic Cache & Hybrid RAG Engine

A high-performance RAG middleware designed to slash LLM latency and compute costs by caching semantically equivalent queries across multiple languages (**English, Swahili, Russian, etc.**).

---

## 🏗️ Architecture
* **Semantic Cache:** Uses multilingual sentence embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) and cosine similarity to serve instant cache hits for paraphrased or cross-lingual queries.
* **Hybrid Retrieval:** Falls back to a combined dense vector and sparse keyword (BM25) search pipeline on cache misses.

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install sentence-transformers rank_bm25 numpy fastapi uvicorn pydantic
```

### 2. Run Server
```Bash
uvicorn main:app --reload --port 8001
```

### 3. Test API
```Bash
curl -X POST [http://127.0.0.1:8001/query](http://127.0.0.1:8001/query) \
  -H "Content-Type: application/json" \
  -d '{"query": "How do high-throughput AI systems use Rust and PyTorch?"}'
```
