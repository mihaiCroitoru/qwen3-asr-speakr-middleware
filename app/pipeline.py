import asyncio
import io
import logging
import numpy as np
import soundfile as sf
from typing import Optional

from app import asr_client, aligner, diarizer, reconciler
from app.config import get_settings

logger = logging.getLogger(__name__)


def _audio_to_array(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    buf = io.BytesIO(audio_bytes)
    array, sample_rate = sf.read(buf, dtype="float32", always_2d=False)
    if array.ndim == 2:
        array = array.mean(axis=1)
    return array, sample_rate


async def _convert_to_wav(audio_bytes: bytes, content_type: str) -> bytes:
    import subprocess
    import tempfile
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

    # convert to wav for local processing
    wav_bytes = await _convert_to_wav(audio_bytes, content_type)
    audio_array, sample_rate = _audio_to_array(wav_bytes)

    # 1. ASR (remote)
    asr_result = await asr_client.transcribe(
        wav_bytes, filename, language=language, initial_prompt=initial_prompt
    )
    asr_segments = asr_result["segments"]
    detected_language = asr_result["language"]
    audio_duration = len(audio_array) / sample_rate
    for seg in asr_segments:
        if seg["end"] == 0.0:
            seg["end"] = round(audio_duration, 3)

    # 2. Align + diarize in parallel
    async def _align():
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            aligner.align,
            audio_array,
            sample_rate,
            asr_segments,
            detected_language,
        )

    async def _diarize():
        if not do_diarize or settings.disable_diarization:
            return []
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None,
                diarizer.diarize,
                audio_array,
                sample_rate,
                num_speakers,
                min_speakers,
                max_speakers,
            )
        except Exception as e:
            logger.warning(f"Diarization failed: {e}, returning no speaker labels")
            return []

    word_timestamps, diar_segments = await asyncio.gather(_align(), _diarize())

    # 3. Reconcile
    final_segments = reconciler.reconcile(
        asr_segments, word_timestamps, diar_segments, do_diarize and not settings.disable_diarization
    )

    word_segments = [w for seg in final_segments for w in seg.get("words", [])]

    return {
        "text": final_segments,
        "language": detected_language,
        "segments": final_segments,
        "word_segments": word_segments,
    }
