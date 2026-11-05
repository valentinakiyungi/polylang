# Sample knowledge base

`knowledge_base.jsonl` holds short articles in English, Kiswahili, Russian and French about
inference, RAG, semantic caching, hybrid retrieval and multilingual embeddings.

Load it with:

```bash
rag-engine ingest data/samples/knowledge_base.jsonl
```

Each line is a JSON object with `text` and optional `id`, `title` and `metadata`.
