import re
import httpx
import logging
from typing import Optional
from app.config import get_settings

logger = logging.getLogger(__name__)

_LANG_NAME_TO_CODE = {
    "english": "en",
    "chinese": "zh",
    "mandarin": "zh",
    "cantonese": "yue",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
    "arabic": "ar",
}


def parse_qwen3_asr_output(raw_text: str, fallback_language: str = "en") -> tuple[str, str]:
    lang_match = re.match(r"language\s+(\w+)", raw_text, re.IGNORECASE)
    if lang_match:
        name = lang_match.group(1).lower()
        language = _LANG_NAME_TO_CODE.get(name, name[:2])
    else:
        language = fallback_language

    asr_match = re.search(r"<asr_text>(.*?)</asr_text>", raw_text, re.DOTALL)
    text = asr_match.group(1).strip() if asr_match else raw_text.strip()

    return language, text


async def transcribe(
    audio_bytes: bytes,
    filename: str,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
) -> dict:
    settings = get_settings()
    url = f"{settings.llama_swap_url}/v1/audio/transcriptions"

    form_data: dict = {
        "model": settings.asr_model_name,
        "response_format": "json",
    }
    if language:
        form_data["language"] = language
    if initial_prompt:
        form_data["prompt"] = initial_prompt

    files = {"file": (filename, audio_bytes, "audio/wav")}

    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        try:
            resp = await client.post(
                url,
                data=form_data,
                files=files,
                headers={"Authorization": f"Bearer {settings.llama_swap_api_key}"},
            )
        except httpx.ConnectError as e:
            raise RuntimeError(f"ASR backend unreachable: {e}") from e
        except httpx.TimeoutException as e:
            raise RuntimeError(f"ASR backend timeout: {e}") from e

    if resp.status_code != 200:
        raise RuntimeError(f"ASR backend error {resp.status_code}: {resp.text}")

    data = resp.json()
    raw_text = data.get("text", "")
    detected_language, clean_text = parse_qwen3_asr_output(raw_text, fallback_language=language or "en")

    segments = [{"text": clean_text, "start": 0.0, "end": 0.0}]
    return {"language": detected_language, "segments": segments}
