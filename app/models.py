from pydantic import BaseModel
from typing import Optional


class WordSegment(BaseModel):
    word: str
    start: float
    end: float
    score: float = 0.0


class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: list[WordSegment] = []


class ASRResponse(BaseModel):
    text: list[Segment]
    language: str
    segments: list[Segment]
    word_segments: list[WordSegment]


class HealthResponse(BaseModel):
    status: str
    device: str
    asr_backend: str
    asr_url: str


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
