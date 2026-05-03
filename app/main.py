import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from app.config import get_settings
from app import pipeline
from app.formatters import to_srt, to_vtt, to_tsv, to_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()
_gpu_semaphore: Optional[asyncio.Semaphore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _gpu_semaphore
    if settings.device == "cuda":
        _gpu_semaphore = asyncio.Semaphore(1)

    # preload models
    if not settings.disable_diarization and settings.hf_token:
        try:
            from app.diarizer import _load_pipeline
            await asyncio.get_event_loop().run_in_executor(None, _load_pipeline)
        except Exception as e:
            logger.warning(f"Diarization model preload failed: {e}")

    try:
        from app.aligner import _load_model
        await asyncio.get_event_loop().run_in_executor(None, _load_model)
    except Exception as e:
        logger.warning(f"Aligner model preload failed: {e}")

    yield


app = FastAPI(title="qwen3-asr-service", lifespan=lifespan)

MAX_BYTES = settings.max_file_size_mb * 1024 * 1024

SUPPORTED_FORMATS = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/wave", "audio/x-wav",
    "audio/ogg", "audio/flac", "audio/m4a", "audio/mp4", "video/mp4",
    "audio/webm", "application/octet-stream",
}


async def _process(
    audio_bytes: bytes,
    content_type: str,
    filename: str,
    language: Optional[str],
    initial_prompt: Optional[str],
    do_diarize: bool,
    output_format: str,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
):
    if len(audio_bytes) > MAX_BYTES:
        raise HTTPException(413, detail={"error": "File too large", "detail": f"Max {settings.max_file_size_mb}MB"})

    try:
        result = await pipeline.run(
            audio_bytes=audio_bytes,
            content_type=content_type,
            filename=filename,
            language=language,
            initial_prompt=initial_prompt,
            do_diarize=do_diarize,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    except RuntimeError as e:
        msg = str(e)
        if "unreachable" in msg or "backend" in msg:
            raise HTTPException(503, detail={"error": "ASR backend unavailable", "detail": msg})
        raise HTTPException(500, detail={"error": "Processing failed", "detail": msg})

    segments = result["segments"]

    if output_format == "srt":
        return PlainTextResponse(to_srt(segments), media_type="text/plain")
    if output_format == "vtt":
        return PlainTextResponse(to_vtt(segments), media_type="text/vtt")
    if output_format == "tsv":
        return PlainTextResponse(to_tsv(segments), media_type="text/tab-separated-values")
    if output_format == "text":
        return PlainTextResponse(to_text(segments), media_type="text/plain")

    return JSONResponse(result)


@app.post("/asr")
async def asr(
    audio_file: UploadFile = File(...),
    task: str = Form("transcribe"),
    language: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    initial_prompt: Optional[str] = Form(None),
    hotwords: Optional[str] = Form(None),
    output_format: str = Form("json"),
    word_timestamps: bool = Form(True),
    diarize: bool = Form(False),
    num_speakers: Optional[int] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
):
    audio_bytes = await audio_file.read()
    content_type = audio_file.content_type or "application/octet-stream"
    prompt = initial_prompt or hotwords

    return await _process(
        audio_bytes=audio_bytes,
        content_type=content_type,
        filename=audio_file.filename or "audio.wav",
        language=language,
        initial_prompt=prompt,
        do_diarize=diarize,
        output_format=output_format,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
):
    audio_bytes = await file.read()
    content_type = file.content_type or "application/octet-stream"

    result = await pipeline.run(
        audio_bytes=audio_bytes,
        content_type=content_type,
        filename=file.filename or "audio.wav",
        language=language,
        initial_prompt=prompt,
        do_diarize=False,
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
    )

    if response_format == "verbose_json":
        return JSONResponse(result)

    full_text = " ".join(seg["text"].strip() for seg in result["segments"])
    return JSONResponse({"text": full_text})


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": settings.device,
        "asr_backend": "llama-swap",
        "asr_url": settings.llama_swap_url,
    }


@app.get("/metrics")
async def metrics():
    return {"requests_total": 0, "requests_active": 0, "queue_size": 0}


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": settings.asr_model_name,
                "object": "model",
                "owned_by": "llama-swap",
            }
        ],
    }
