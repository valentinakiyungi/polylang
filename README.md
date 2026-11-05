# 🌐 Polylang

**Ask in any language. Answer from every language.**

[![CI](https://github.com/valentinakiyungi/polylang/actions/workflows/ci.yml/badge.svg)](https://github.com/valentinakiyungi/polylang/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A self-hosted Retrieval-Augmented Generation service for multilingual knowledge bases.
Ask in English, Kiswahili, Russian, French or 50+ other languages and get answers grounded in
your own documents — whatever language those documents are written in.

A **cross-lingual semantic cache** sits in front of the pipeline: if someone already asked
the same thing (in any language, in any wording), the stored answer comes back in
milliseconds and no retrieval or LLM call is made.

```
                       ┌──────────────────────────┐
  question (any lang) ─▶ multilingual embedding   │
                       └────────────┬─────────────┘
                                    ▼
                       ┌──────────────────────────┐  hit (cosine ≥ threshold)
                       │   semantic cache (LRU,   ├──────────────────────────▶ cached answer
                       │   TTL, cross-lingual)    │
                       └────────────┬─────────────┘
                                    │ miss
                    ┌───────────────┴────────────────┐
                    ▼                                ▼
          dense vector search                 BM25 keyword search
          (meaning, any language)             (exact names, codes)
                    └───────────────┬────────────────┘
                                    ▼
                    reciprocal-rank fusion  →  top-k passages
                                    ▼
             generator: extractive │ OpenAI │ Anthropic │ Ollama
                                    ▼
                answer in the question's language + cited sources ──▶ stored in cache
```

## Features

- **Document ingestion** – upload `.txt .md .pdf .docx .html .csv .json .jsonl`, paste text,
  or ingest whole folders from the CLI. Documents are split into overlapping, sentence-aligned
  chunks; language is detected per document and per chunk; duplicates are skipped.
- **Hybrid retrieval** – dense multilingual embeddings + an incremental BM25 index, merged with
  reciprocal rank fusion (or a weighted min-max blend). Filter by collection.
- **Cross-lingual semantic cache** – vectorised lookup, configurable threshold, LRU eviction,
  TTL, per-collection scoping, hit statistics. It is invalidated automatically whenever the
  knowledge base changes, so it never serves stale answers.
- **Pluggable answer generation**
  - `extractive` (default) – no LLM, no API key: picks the most relevant sentences and cites them.
  - `openai` – OpenAI or any OpenAI-compatible endpoint (vLLM, LM Studio, OpenRouter, …).
  - `anthropic` – Claude via the Messages API.
  - `ollama` – local models.
  LLM answers are written in the question's language and cite passages as `[1]`, `[2]`.
  If the LLM call fails, the engine falls back to an extractive answer and reports a warning.
- **Persistence** – documents, vectors and cache are saved to disk and reloaded on start. Switching
  embedding models triggers an automatic re-embed.
- **Web UI, REST API, CLI** – a built-in, mobile-friendly web app at `/` (light/dark theme, a settings
  panel, cited answers that show which languages they came from), interactive API docs at `/docs`,
  and a `polylang` command.
- **Production bits** – optional API-key auth, CORS, upload limits, Docker image with the model
  baked in, health check, GitHub Actions CI.

## Quick start

```bash
git clone https://github.com/valentinakiyungi/polylang.git
cd polylang
python3 -m venv venv && source venv/bin/activate
pip install -e ".[files]"

polylang ingest data/samples/knowledge_base.jsonl   # optional sample data
polylang serve                                      # http://127.0.0.1:8001
```

Open **http://127.0.0.1:8001** for the web UI, or **/docs** for the API explorer.
The first start downloads the embedding model (~470 MB) from Hugging Face.

The old entry point still works: `uvicorn main:app --port 8001`.

### Docker

```bash
cp .env.example .env        # optional
docker compose up --build   # http://localhost:8001
```

Data lives in the `polylang-storage` volume. For a fully local LLM:
`docker compose --profile ollama up`, then `docker compose exec ollama ollama pull llama3.1`
and set `POLYLANG_LLM_PROVIDER=ollama`, `POLYLANG_LLM_BASE_URL=http://ollama:11434` in `.env`.

## Using it

### CLI

```bash
polylang ingest ./docs ./notes.pdf -c handbook     # files or folders, into a collection
polylang query "Je, semantic caching hufanyaje kazi?"
polylang query "How does semantic caching work?"   # → cache hit from the Swahili question
polylang query "What is BM25?" -k 6 --no-cache --json
polylang list
polylang delete <doc-id>
polylang stats
polylang clear-cache
```

Add `--offline` to any command to use the dependency-free hashing embedder (see below).

### REST API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/query` | Ask a question (cache → hybrid retrieval → generation) |
| `POST` | `/retrieve` | Retrieval only, no generation |
| `POST` | `/documents` | Add documents as JSON |
| `POST` | `/documents/upload` | Upload files (multipart, field `files`, optional `collection`) |
| `GET` | `/documents` | List documents (`?collection=`) |
| `GET` | `/documents/{id}` | One document with its chunks |
| `DELETE` | `/documents/{id}` | Delete a document |
| `DELETE` | `/documents?confirm=true` | Delete everything |
| `GET` | `/cache` | Cached answers |
| `DELETE` | `/cache`, `/cache/{id}` | Clear the cache / one entry |
| `GET` | `/stats` | Index and cache statistics |
| `GET` / `PATCH` | `/config` | Read or change answer and cache settings while running |
| `GET` | `/health` | Liveness (never needs a key) |

```bash
curl -X POST http://127.0.0.1:8001/documents -H "Content-Type: application/json" -d '{
  "collection": "faq",
  "documents": [{"id": "refunds", "title": "Refund policy",
                 "text": "Refunds are issued within 14 days of purchase."}]
}'

curl -X POST http://127.0.0.1:8001/documents/upload -F "files=@handbook.pdf" -F "collection=faq"

curl -X POST http://127.0.0.1:8001/query -H "Content-Type: application/json" \
  -d '{"query": "Ninaweza kurudishiwa pesa baada ya siku ngapi?", "collection": "faq"}'
```

A query response looks like:

```json
{
  "query": "How does semantic caching work?",
  "language": "en",
  "status": "cache_hit",
  "answer": "Semantic caching reduces LLM latency and cost by reusing answers ... [1]",
  "cache": {"similarity": 0.9312, "matched_query": "Je, semantic caching hufanyaje kazi?",
            "matched_language": "sw", "hits": 1, "entry_id": "3f0c9a1b2d4e"},
  "sources": [{"rank": 1, "title": "Semantic caching", "language": "en", "score": 0.0328,
               "dense_score": 0.71, "sparse_score": 6.2, "text": "...", "doc_id": "cache-en",
               "chunk_id": "cache-en:0"}],
  "generator": "extractive",
  "warnings": [],
  "timings_ms": {"embed": 11.4, "cache_lookup": 0.08, "total": 11.6}
}
```

`status` is `cache_hit`, `cache_miss` (answered through retrieval) or `no_context`.

### Python

```python
from polylang import RAGEngine, Settings

engine = RAGEngine(Settings.from_env(llm_provider="ollama", llm_model="llama3.1"))
engine.ingest_path("docs/")
result = engine.query("Как работает гибридный поиск?")
print(result.answer, [s.title for s in result.sources])
```

## Configuration

All settings come from environment variables (or a `.env` file); see
[`.env.example`](.env.example) for the full list.

| Variable | Default | Notes |
|---|---|---|
| `POLYLANG_DATA_DIR` | `./storage` | Where documents, vectors and cache are saved |
| `POLYLANG_EMBEDDING_BACKEND` | `sentence-transformers` | `hashing` = offline fallback |
| `POLYLANG_EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Any sentence-transformers model |
| `POLYLANG_CHUNK_SIZE` / `POLYLANG_CHUNK_OVERLAP` | `800` / `120` | Characters |
| `POLYLANG_TOP_K` | `4` | Passages per answer |
| `POLYLANG_FUSION` | `rrf` | `rrf` or `weighted` |
| `POLYLANG_DENSE_WEIGHT` | `0.6` | Dense vs BM25 balance |
| `POLYLANG_CACHE_THRESHOLD` | `0.88` | Cosine similarity needed for a hit |
| `POLYLANG_CACHE_CROSS_LINGUAL` | `true` | `false` = only reuse answers within the same language |
| `POLYLANG_CACHE_TTL_SECONDS` | `86400` | `0` = never expire |
| `POLYLANG_LLM_PROVIDER` | `extractive` | `openai`, `anthropic`, `ollama` |
| `POLYLANG_LLM_MODEL` | provider default | e.g. `gpt-4o-mini`, `claude-haiku-5-5`, `llama3.1` |
| `POLYLANG_LLM_BASE_URL` | provider default | For OpenAI-compatible servers / Ollama |
| `POLYLANG_LLM_API_KEY` | – | Falls back to `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` |
| `POLYLANG_API_KEY` | – | If set, requests need header `X-API-Key` |

Passages per answer, search method, the meaning/keyword balance and all saved-answer settings can
also be changed while the server runs, from **Settings** in the web app or `PATCH /config`. Those
changes are saved to `settings.json` in the data directory and take precedence over environment
variables.

**Choosing the cache threshold.** Higher is safer (fewer wrong reuses), lower saves more calls.
With the default model, 0.85–0.90 works well for paraphrases; cross-lingual pairs of the same
question typically score 0.85–0.95. Watch `hit_rate` in `/stats` and adjust.

**Cross-lingual cache hits return the cached answer as written.** If an English question hits an
answer that was generated for a Swahili question, you get the Swahili answer (the response says
so in `cache.matched_language`). Set `POLYLANG_CACHE_CROSS_LINGUAL=false` if answers must always be in
the asker's language.

### Offline / lightweight mode

`POLYLANG_EMBEDDING_BACKEND=hashing` (or `--offline`) swaps the neural model for a character n-gram
hashing embedder that needs only NumPy. Retrieval, caching, the API and the UI all work, but
matching is lexical, so it is **not** cross-lingual. It is used by the test suite and is handy
on machines without PyTorch.

## Development

```bash
pip install -e ".[files,dev]"
pytest          # runs offline with the hashing embedder
ruff check polylang tests
```

Project layout:

```
polylang/
  engine.py        orchestration: ingest, retrieve, cache, generate, persist
  cache.py         cross-lingual semantic cache
  bm25.py          incremental BM25 index
  vector_store.py  dense index
  embeddings.py    sentence-transformers + hashing embedders
  llm.py           extractive / OpenAI / Anthropic / Ollama generators
  chunking.py      sentence-aligned chunking
  loaders.py       file parsers
  text.py          tokenisation and language detection
  api.py           FastAPI routes      server.py  ASGI entry point
  cli.py           polylang command  static/    web UI
tests/             pytest suite
data/samples/      multilingual sample knowledge base
```

## License

[MIT](LICENSE)
