def _ts_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_vtt(seconds: float) -> str:
    return _ts_srt(seconds).replace(",", ".")


def to_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_ts_srt(seg['start'])} --> {_ts_srt(seg['end'])}")
        text = seg["text"].strip()
        if seg.get("speaker"):
            text = f"[{seg['speaker']}] {text}"
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def to_vtt(segments: list[dict]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_ts_vtt(seg['start'])} --> {_ts_vtt(seg['end'])}")
        text = seg["text"].strip()
        if seg.get("speaker"):
            text = f"<v {seg['speaker']}>{text}"
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def to_tsv(segments: list[dict]) -> str:
    lines = ["start\tend\ttext"]
    for seg in segments:
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        text = seg["text"].strip()
        lines.append(f"{start_ms}\t{end_ms}\t{text}")
    return "\n".join(lines)


def to_text(segments: list[dict]) -> str:
    return "\n".join(seg["text"].strip() for seg in segments)
