#!/bin/sh
set -e

PORT=${PORT:-9000}
LOG_LEVEL=${LOG_LEVEL:-info}

exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level "$LOG_LEVEL"
