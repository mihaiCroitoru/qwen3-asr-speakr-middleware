from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    llama_swap_url: str = "http://llama-swap:8080"
    llama_swap_api_key: str = "placeholder"
    asr_model_name: str = "qwen3-asr-1.7b"

    hf_token: str = ""
    pyannote_model: str = "pyannote/speaker-diarization-3.1"

    forced_aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    aligner_device: str = "cpu"

    device: str = "cpu"
    max_file_size_mb: int = 1000
    request_timeout: int = 300

    port: int = 9000
    log_level: str = "info"

    disable_diarization: bool = False

    asr_chunking_strategy: str = "auto"  # auto | always | never
    asr_chunk_duration: int = 25
    asr_chunk_threshold_seconds: int = 28

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
