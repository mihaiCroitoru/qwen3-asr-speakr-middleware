ARG BUILD_WITH_CUDA=false
ARG HF_TOKEN=""

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .

ARG BUILD_WITH_CUDA=false

RUN if [ "$BUILD_WITH_CUDA" = "true" ]; then \
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128; \
    else \
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# pre-download ForcedAligner if token provided at build time
ARG HF_TOKEN=""
RUN if [ -n "$HF_TOKEN" ]; then \
        python -c "from transformers import AutoProcessor, AutoModelForCTC; \
        AutoProcessor.from_pretrained('Qwen/Qwen3-ForcedAligner-0.6B', token='${HF_TOKEN}'); \
        AutoModelForCTC.from_pretrained('Qwen/Qwen3-ForcedAligner-0.6B', token='${HF_TOKEN}')"; \
    fi

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 9000

ENTRYPOINT ["./entrypoint.sh"]
