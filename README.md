# qwen3-asr-service

Drop-in replacement for [murtaza-nasir/whisperx-asr-service](https://github.com/murtaza-nasir/whisperx-asr-service) that offloads ASR to llama-swap.

## Architecture

```
Client (Speakr)
  └─► POST /asr
        └─► [this service]
              ├─► llama-swap /v1/audio/transcriptions  (Qwen3-ASR-1.7B, remote GPU)
              ├─► Qwen3-ForcedAligner-0.6B             (local CPU/GPU)
              └─► Pyannote speaker-diarization         (local CPU/GPU)
            ◄─── unified JSON (whisperx-compatible)
```

## Prerequisites

### Accept pyannote model terms on HuggingFace

Visit and accept terms for all three:
- https://huggingface.co/pyannote/speaker-diarization-community-1
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM

### llama-swap config.yaml

Add to your llama-swap `config.yaml`:

```yaml
models:
  qwen3-asr-1.7b:
    cmd: >
      llama-server
        --model /models/Qwen3-ASR-1.7B-Q8_0.gguf
        --port 8081
        --host 0.0.0.0
    proxy: http://localhost:8081
```

## Speakr Integration

Point Speakr at this service by changing only `ASR_BASE_URL`:

```env
ASR_BASE_URL=http://qwen3-asr-service:9000
```

No other Speakr config changes needed.

## Quick Start

```bash
cp .env.example .env
# edit .env — set LLAMA_SWAP_URL, HF_TOKEN

docker compose up -d

curl http://localhost:9000/health

bash test-api.sh test.mp3
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLAMA_SWAP_URL` | `http://llama-swap:8080` | llama-swap base URL |
| `ASR_MODEL_NAME` | `qwen3-asr-1.7b` | model name in llama-swap config |
| `HF_TOKEN` | — | HuggingFace token (required for pyannote) |
| `DEVICE` | `cpu` | device for pyannote (`cpu` or `cuda`) |
| `ALIGNER_DEVICE` | `cpu` | device for ForcedAligner |
| `DISABLE_DIARIZATION` | `false` | skip diarization entirely |
| `MAX_FILE_SIZE_MB` | `1000` | max upload size |
| `REQUEST_TIMEOUT` | `300` | llama-swap timeout seconds |
| `PORT` | `9000` | service port |

## Differences from whisperx-asr-service

- **`model` parameter ignored** — ASR model always comes from `ASR_MODEL_NAME` env var. The model is fixed to whatever is configured in llama-swap.
- No local GPU required for ASR — llama-swap handles it.
- No whisperx, faster-whisper, or ctranslate2 dependencies.

## API

### POST /asr
Same as whisperx-asr-service. Fields: `audio_file`, `language`, `diarize`, `output_format` (json/text/srt/vtt/tsv), `num_speakers`, `min_speakers`, `max_speakers`, `initial_prompt`, `hotwords`.

### POST /v1/audio/transcriptions
OpenAI-compatible. Fields: `file`, `model`, `language`, `prompt`, `response_format`.

### GET /health
```json
{"status": "healthy", "device": "cpu", "asr_backend": "llama-swap", "asr_url": "..."}
```

## OpenPlaude Compatibility

Placeholder — details TBD.

## CUDA Build

```bash
BUILD_WITH_CUDA=true docker compose build
```

Uncomment the `deploy.resources.reservations` block in `docker-compose.yml` and set `DEVICE=cuda`.
