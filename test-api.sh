#!/bin/bash
set -e

BASE_URL="${ASR_BASE_URL:-http://localhost:9000}"

echo "=== Health check ==="
curl -s "$BASE_URL/health" | python3 -m json.tool

echo ""
echo "=== POST /asr ==="
curl -X POST "$BASE_URL/asr" \
  -F "audio_file=@${1:-test.mp3}" \
  -F "language=${LANGUAGE:-en}" \
  -F "diarize=${DIARIZE:-true}" \
  -F "output_format=${OUTPUT_FORMAT:-json}" \
  | python3 -m json.tool
