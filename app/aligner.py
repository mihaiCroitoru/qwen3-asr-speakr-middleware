import logging
import numpy as np
from app.config import get_settings

logger = logging.getLogger(__name__)

LANGUAGE_MAP = {
    "en": "English",
    "zh": "Chinese",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
}

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model

    from qwen_asr import Qwen3ForcedAligner
    import torch

    settings = get_settings()
    model_id = settings.forced_aligner_model
    device = settings.aligner_device

    logger.info(f"Loading ForcedAligner: {model_id} on {device}")
    _model = Qwen3ForcedAligner.from_pretrained(
        model_id,
        dtype=torch.float32,
        device_map=device,
    )
    logger.info("ForcedAligner loaded")
    return _model


def _fallback_words(segments: list[dict]) -> list[dict]:
    words = []
    for seg in segments:
        tokens = seg["text"].strip().split()
        if not tokens:
            continue
        duration = (seg["end"] - seg["start"]) / max(len(tokens), 1)
        for i, token in enumerate(tokens):
            words.append({
                "word": token,
                "start": round(seg["start"] + i * duration, 3),
                "end": round(seg["start"] + (i + 1) * duration, 3),
                "score": 0.0,
            })
    return words


def align(
    audio_array: np.ndarray,
    sample_rate: int,
    segments: list[dict],
    language: str,
) -> list[dict]:
    lang_name = LANGUAGE_MAP.get(language)
    if lang_name is None:
        logger.warning(f"Language '{language}' not supported by ForcedAligner, using fallback")
        return _fallback_words(segments)

    try:
        model = _load_model()
        full_text = " ".join(seg["text"].strip() for seg in segments)

        results = model.align(
            audio=(audio_array, sample_rate),
            text=full_text,
            language=lang_name,
        )

        words = []
        for token in results[0]:
            words.append({
                "word": token.text,
                "start": round(token.start_time, 3),
                "end": round(token.end_time, 3),
                "score": 0.9,
            })
        return words

    except Exception as e:
        logger.warning(f"ForcedAligner failed: {e}, using fallback")
        return _fallback_words(segments)
