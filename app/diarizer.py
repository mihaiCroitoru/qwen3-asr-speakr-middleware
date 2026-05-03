import logging
import tempfile
import os
import numpy as np
import soundfile as sf
from typing import Optional
from app.config import get_settings

logger = logging.getLogger(__name__)

_pipeline = None


def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import torch
    from pyannote.audio import Pipeline

    settings = get_settings()
    if not settings.hf_token:
        raise RuntimeError("HF_TOKEN required for pyannote diarization")

    logger.info(f"Loading diarization pipeline: {settings.pyannote_model}")
    _pipeline = Pipeline.from_pretrained(
        settings.pyannote_model,
        token=settings.hf_token,
    )
    device = settings.device
    if device == "cuda":
        import torch
        _pipeline = _pipeline.to(torch.device("cuda"))

    logger.info("Diarization pipeline loaded")
    return _pipeline


def diarize(
    audio_array: np.ndarray,
    sample_rate: int,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> list[dict]:
    pipeline = _load_pipeline()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    try:
        sf.write(tmp_path, audio_array, sample_rate)

        kwargs: dict = {}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers
        elif min_speakers or max_speakers:
            if min_speakers:
                kwargs["min_speakers"] = min_speakers
            if max_speakers:
                kwargs["max_speakers"] = max_speakers

        diarization = pipeline(tmp_path, **kwargs)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker,
            })
        return segments

    finally:
        os.unlink(tmp_path)
