import asyncio
import io
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from app import asr_client, aligner, diarizer, reconciler
from app.config import get_settings

logger = logging.getLogger(__name__)


def _audio_to_array(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    buf = io.BytesIO(audio_bytes)
    array, sample_rate = sf.read(buf, dtype="float32", always_2d=False)
    if array.ndim == 2:
        array = array.mean(axis=1)
    return array, sample_rate


def _get_audio_duration(audio_bytes: bytes) -> float:
    array, sr = _audio_to_array(audio_bytes)
    return len(array) / sr


async def _convert_to_wav(audio_bytes: bytes, content_type: str) -> bytes:
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix=_ext(content_type)) as f:
        f.write(audio_bytes)
        in_path = f.name

    out_path = in_path + ".wav"
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", in_path, "-ar", "16000", "-ac", "1", "-f", "wav", out_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


def _ext(content_type: str) -> str:
    mapping = {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/wave": ".wav",
        "audio/x-wav": ".wav",
        "audio/ogg": ".ogg",
        "audio/flac": ".flac",
        "audio/m4a": ".m4a",
        "audio/mp4": ".mp4",
        "video/mp4": ".mp4",
        "audio/webm": ".webm",
    }
    return mapping.get(content_type, ".audio")


async def _chunk_audio(audio_bytes: bytes, tmp_dir: Path, chunk_duration: int) -> list[Path]:
    import os
    input_path = tmp_dir / "input.wav"
    input_path.write_bytes(audio_bytes)
    chunks = []
    i = 0
    while True:
        chunk_path = tmp_dir / f"chunk_{i:04d}.wav"
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-ss", str(i * chunk_duration),
            "-t", str(chunk_duration),
            "-ar", "16000", "-ac", "1", "-f", "wav",
            str(chunk_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        if chunk_path.exists() and chunk_path.stat().st_size > 44:
            chunks.append(chunk_path)
            i += 1
        else:
            if chunk_path.exists():
                chunk_path.unlink()
            break
    return chunks


async def _transcribe_chunked(
    audio_bytes: bytes,
    filename: str,
    language: Optional[str],
    initial_prompt: Optional[str],
    chunk_duration: int,
) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        chunks = await _chunk_audio(audio_bytes, tmp_dir, chunk_duration)
        logger.info(f"Chunked audio into {len(chunks)} x {chunk_duration}s chunks")

        all_segments: list[dict] = []
        detected_language = language or "en"

        for i, chunk_path in enumerate(chunks):
            chunk_bytes = chunk_path.read_bytes()
            time_offset = i * chunk_duration
            result = await asr_client.transcribe(
                chunk_bytes, filename, language=language, initial_prompt=initial_prompt
            )
            detected_language = result["language"]
            for seg in result["segments"]:
                all_segments.append({
                    "text": seg["text"],
                    "start": seg["start"] + time_offset,
                    "end": seg["end"] + time_offset,
                })

        return {"language": detected_language, "segments": all_segments}


async def run(
    audio_bytes: bytes,
    content_type: str,
    filename: str,
    language: Optional[str],
    initial_prompt: Optional[str],
    do_diarize: bool,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> dict:
    settings = get_settings()

    wav_bytes = await _convert_to_wav(audio_bytes, content_type)
    audio_array, sample_rate = _audio_to_array(wav_bytes)
    audio_duration = len(audio_array) / sample_rate

    # 1. ASR — chunk if needed
    strategy = settings.asr_chunking_strategy.lower()
    use_chunking = (
        strategy == "always"
        or (strategy == "auto" and audio_duration > settings.asr_chunk_threshold_seconds)
    )

    if use_chunking:
        logger.info(f"Audio {audio_duration:.1f}s — using chunked ASR ({settings.asr_chunk_duration}s chunks)")
        asr_result = await _transcribe_chunked(
            wav_bytes, filename, language, initial_prompt, settings.asr_chunk_duration
        )
    else:
        asr_result = await asr_client.transcribe(
            wav_bytes, filename, language=language, initial_prompt=initial_prompt
        )

    asr_segments = asr_result["segments"]
    detected_language = asr_result["language"]

    # fill real duration on segments that have placeholder end=0.0
    for seg in asr_segments:
        if seg["end"] == 0.0:
            seg["end"] = round(audio_duration, 3)

    # 2. Align + diarize in parallel
    async def _align():
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, aligner.align, audio_array, sample_rate, asr_segments, detected_language
        )

    async def _diarize():
        if not do_diarize or settings.disable_diarization:
            return []
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, diarizer.diarize, audio_array, sample_rate,
                num_speakers, min_speakers, max_speakers
            )
        except Exception as e:
            logger.warning(f"Diarization failed: {e}, returning no speaker labels")
            return []

    word_timestamps, diar_segments = await asyncio.gather(_align(), _diarize())

    # 3. Reconcile
    final_segments = reconciler.reconcile(
        asr_segments, word_timestamps, diar_segments,
        do_diarize and not settings.disable_diarization
    )
    word_segments = [w for seg in final_segments for w in seg.get("words", [])]

    return {
        "text": final_segments,
        "language": detected_language,
        "segments": final_segments,
        "word_segments": word_segments,
    }
