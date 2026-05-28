#!/bin/sh
set -eu

HOST="${RAG_CORE_HOST:-0.0.0.0}"
PORT="${RAG_CORE_PORT:-8010}"
WEB_CONCURRENCY="${RAG_CORE_WEB_CONCURRENCY:-1}"

echo "=== Starting rag-core service ==="
echo "HOST=$HOST"
echo "PORT=$PORT"
echo "WEB_CONCURRENCY=$WEB_CONCURRENCY"

exec gunicorn rag_core.api.main:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind "$HOST:$PORT" \
  --timeout 120 \
  --workers "$WEB_CONCURRENCY" \
  --access-logfile - \
  --error-logfile - \
  --log-level info

