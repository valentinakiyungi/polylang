.PHONY: install dev test lint run seed docker

install:
	pip install -e ".[files]"

dev:
	pip install -e ".[files,dev]"

test:
	pytest

lint:
	ruff check rag_engine tests

run:
	rag-engine serve --reload

seed:
	rag-engine ingest data/samples/knowledge_base.jsonl

docker:
	docker compose up --build
