.PHONY: install dev test lint run seed docker

install:
	pip install -e ".[files]"

dev:
	pip install -e ".[files,dev]"

test:
	pytest

lint:
	ruff check polylang tests

run:
	polylang serve --reload

seed:
	polylang ingest data/samples/knowledge_base.jsonl

docker:
	docker compose up --build
