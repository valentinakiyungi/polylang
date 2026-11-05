"""Command-line interface: ``rag-engine <command>`` (or ``python -m rag_engine``)."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from rag_engine.config import Settings


def _engine(args):
    from rag_engine.engine import RAGEngine
    overrides = {}
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    if args.offline:
        overrides["embedding_backend"] = "hashing"
    return RAGEngine(Settings.from_env(**overrides))


def cmd_ingest(args) -> int:
    eng = _engine(args)
    total = 0
    for path in args.paths:
        for r in eng.ingest_path(path, args.collection):
            total += 1
            state = "added" if r.created else "unchanged"
            print(f"  {state:9} {r.document.title}  [{r.document.language}, {len(r.document.chunk_ids)} chunks]")
    print(f"{total} document(s) processed. Knowledge base: {len(eng.documents)} documents, {len(eng.chunks)} chunks.")
    eng.close()
    return 0


def cmd_query(args) -> int:
    eng = _engine(args)
    r = eng.query(" ".join(args.question), args.top_k, args.collection, use_cache=not args.no_cache)
    eng.close()
    if args.json:
        print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
        return 0
    status = r.status.replace("_", " ")
    if r.cache:
        status += f" (similarity {r.cache['similarity']} to “{r.cache['matched_query']}”)"
    print(f"[{status} · {r.language} · {r.timings_ms.get('total')} ms]\n")
    print(r.answer)
    for w in r.warnings:
        print(f"\n! {w}", file=sys.stderr)
    if r.sources:
        print("\nSources:")
        for s in r.sources:
            print(f"  [{s.rank}] {s.title} ({s.language}, score {s.score:.4f})")
    return 0


def cmd_list(args) -> int:
    eng = _engine(args)
    docs = eng.list_documents(args.collection)
    for d in docs:
        print(f"{d['id']}  {d['language']:3}  {d['collection']:12}  {d['chunk_count']:4} chunks  {d['title']}")
    print(f"{len(docs)} document(s)")
    return 0


def cmd_delete(args) -> int:
    eng = _engine(args)
    ok = eng.delete_document(args.doc_id)
    print("deleted" if ok else "not found")
    return 0 if ok else 1


def cmd_stats(args) -> int:
    print(json.dumps(_engine(args).stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_clear_cache(args) -> int:
    eng = _engine(args)
    n = eng.cache.clear()
    eng._cache_dirty += 1
    eng.close()
    print(f"cleared {n} cache entries")
    return 0


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed: pip install 'uvicorn[standard]'", file=sys.stderr)
        return 1
    if args.data_dir:
        import os
        os.environ["RAG_DATA_DIR"] = args.data_dir
    if args.offline:
        import os
        os.environ["RAG_EMBEDDING_BACKEND"] = "hashing"
    uvicorn.run("rag_engine.server:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag-engine", description="Cross-lingual semantic cache & hybrid RAG engine")
    p.add_argument("--data-dir", help="storage directory (default: $RAG_DATA_DIR or ./storage)")
    p.add_argument("--offline", action="store_true", help="use the dependency-free hashing embedder")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest", help="add files or folders to the knowledge base")
    s.add_argument("paths", nargs="+")
    s.add_argument("-c", "--collection", default="default")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("query", help="ask a question")
    s.add_argument("question", nargs="+")
    s.add_argument("-k", "--top-k", type=int)
    s.add_argument("-c", "--collection")
    s.add_argument("--no-cache", action="store_true")
    s.add_argument("--json", action="store_true", help="print the full JSON result")
    s.set_defaults(func=cmd_query)

    s = sub.add_parser("list", help="list documents")
    s.add_argument("-c", "--collection")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("delete", help="delete a document by id")
    s.add_argument("doc_id")
    s.set_defaults(func=cmd_delete)

    sub.add_parser("stats", help="show index and cache statistics").set_defaults(func=cmd_stats)
    sub.add_parser("clear-cache", help="empty the semantic cache").set_defaults(func=cmd_clear_cache)

    s = sub.add_parser("serve", help="run the HTTP API and web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8001)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
