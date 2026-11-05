FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    POLYLANG_DATA_DIR=/app/storage

WORKDIR /app

# CPU-only PyTorch keeps the image small (~1.5 GB smaller than the CUDA build).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md ./
COPY polylang ./polylang
RUN pip install ".[files]"

# Bake the embedding model into the image so containers start offline.
ARG EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
ENV POLYLANG_EMBEDDING_MODEL=${EMBEDDING_MODEL}
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

COPY data ./data
RUN useradd --create-home app && mkdir -p /app/storage && chown -R app /app
USER app

EXPOSE 8001
VOLUME ["/app/storage"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health')" || exit 1

CMD ["uvicorn", "polylang.server:app", "--host", "0.0.0.0", "--port", "8001"]
