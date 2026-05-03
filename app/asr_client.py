import httpx
import logging
from typing import Optional
from app.config import get_settings

logger = logging.getLogger(__name__)


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
    detected_language = data.get("language", language or "en")
    full_text = data.get("text", "")

    # llama.cpp json format returns flat text only — one segment, real timestamps
    # come from ForcedAligner; duration placeholder filled by pipeline
    segments = [{"text": full_text, "start": 0.0, "end": 0.0}]

    return {"language": detected_language, "segments": segments}
