# Production image for the AI Code Review Agent on Railway.
#
# Single uvicorn worker is intentional and load-bearing: queue_manager.py's
# per-repo serial guarantee is per-process. Do not raise --workers above 1
# without first making the queue shared (e.g. Redis-backed).

FROM python:3.11-slim

# git    — clones target repos for RAG indexing
# nodejs/npm — required to run ESLint on JS/TS files in reviewed PRs
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g eslint@9

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake the CodeBERT embedding model (~500 MB) into the image so the
# first webhook after a deploy doesn't pay a ~30s model-load penalty.
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('microsoft/codebert-base')"

COPY . .

# ChromaDB lives inside the container's ephemeral filesystem. The index
# is wiped on every restart/deploy, so the first PR per repo after a
# restart pays a re-index cost. To make it persistent later, mount a
# Railway volume at /app/chroma_db (no code changes needed).
ENV CHROMA_PATH=/app/chroma_db
RUN mkdir -p /app/chroma_db

EXPOSE 8000

# Shell form so ${PORT} expands at runtime — Railway sets PORT dynamically;
# 8000 is the fallback for local `docker run`.
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --log-level info
