import logging
import numpy as np
from typing import Optional
from app.config import get_settings

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {"en", "zh", "fr", "de", "ja", "ko", "es", "pt", "it", "ar", "ru"}

_model = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    import torch
    from transformers import AutoProcessor, AutoModelForCTC

    settings = get_settings()
    model_id = settings.forced_aligner_model
    device = settings.aligner_device

    logger.info(f"Loading ForcedAligner: {model_id} on {device}")
    _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    _model = AutoModelForCTC.from_pretrained(model_id, trust_remote_code=True).to(device)
    _model.eval()
    logger.info("ForcedAligner loaded")
    return _model, _processor


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
    if language not in SUPPORTED_LANGUAGES:
        logger.warning(f"Language '{language}' not supported by ForcedAligner, using fallback")
        return _fallback_words(segments)

    try:
        import torch
        model, processor = _load_model()
        settings = get_settings()
        device = settings.aligner_device

        # resample to 16kHz if needed
        if sample_rate != 16000:
            import torchaudio
            waveform = torch.tensor(audio_array).unsqueeze(0)
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            audio_array = waveform.squeeze(0).numpy()
            sample_rate = 16000

        full_text = " ".join(seg["text"].strip() for seg in segments)

        inputs = processor(
            audio_array,
            sampling_rate=sample_rate,
            text=full_text,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # word timestamps from logits
        word_offsets = processor.decode(
            outputs.logits.argmax(dim=-1)[0],
            output_word_offsets=True,
        ).word_offsets

        time_per_frame = audio_array.shape[-1] / sample_rate / outputs.logits.shape[1]

        words = []
        for wo in word_offsets:
            words.append({
                "word": wo.word,
                "start": round(wo.start_offset * time_per_frame, 3),
                "end": round(wo.end_offset * time_per_frame, 3),
                "score": 0.9,
            })
        return words

    except Exception as e:
        logger.warning(f"ForcedAligner failed: {e}, using fallback")
        return _fallback_words(segments)
