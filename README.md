# 🌐 Cross-Lingual Semantic Cache & Hybrid RAG Engine

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance Retrieval-Augmented Generation (RAG) middleware designed to drastically slash LLM latency and compute costs by caching semantically equivalent queries across multiple languages (English, Swahili, Russian, and more).

---

## 🏗️ Architecture

* **Semantic Cache:** Uses multilingual sentence embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) and cosine similarity to serve instant cache hits for paraphrased or cross-lingual queries.
* **Hybrid Retrieval:** Falls back to a combined dense vector and sparse keyword (`BM25`) search pipeline on cache misses.

---

## 🚀 Quick Start

### 1. Setup Environment

Clone the repository and set up a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install sentence-transformers rank_bm25 numpy fastapi uvicorn pydantic
```

### 2. Run Server

Start the FastAPI development server:

```Bash
uvicorn main:app --reload --port 8001
```

### 3. Test API

Send a test query to the endpoint using curl:

```Bash
curl -X POST [http://127.0.0.1:8001/query](http://127.0.0.1:8001/query) \
  -H "Content-Type: application/json" \
  -d '{"query": "How do high-throughput AI systems use Rust and PyTorch?"}'
```

## 🛠️ Tech Stack

* **Web Framework:** FastAPI & Uvicorn

* **Embeddings:** Hugging Face sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)

* **Keyword Search:** rank_bm25

* **Data Validation:** Pydantic & NumPy
