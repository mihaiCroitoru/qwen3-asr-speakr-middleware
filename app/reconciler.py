from typing import Optional


def _find_speaker(midpoint: float, diar_segments: list[dict]) -> Optional[str]:
    for seg in diar_segments:
        if seg["start"] <= midpoint <= seg["end"]:
            return seg["speaker"]
    # nearest fallback
    if not diar_segments:
        return None
    nearest = min(diar_segments, key=lambda s: abs((s["start"] + s["end"]) / 2 - midpoint))
    return nearest["speaker"]


def _normalize_speaker(label: str) -> str:
    # pyannote returns SPEAKER_00, SPEAKER_01 etc — already correct format
    parts = label.split("_")
    if len(parts) == 2 and parts[1].isdigit():
        return f"SPEAKER_{int(parts[1]):02d}"
    return label


def reconcile(
    asr_segments: list[dict],
    word_timestamps: list[dict],
    diar_segments: list[dict],
    do_diarize: bool,
) -> list[dict]:
    # assign speaker to each word
    for word in word_timestamps:
        if do_diarize and diar_segments:
            midpoint = (word["start"] + word["end"]) / 2
            raw_speaker = _find_speaker(midpoint, diar_segments)
            word["speaker"] = _normalize_speaker(raw_speaker) if raw_speaker else None
        else:
            word["speaker"] = None

    # group words into utterance segments by speaker continuity
    if not word_timestamps:
        # no words — return asr segments as-is
        return [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "speaker": None,
                "words": [],
            }
            for seg in asr_segments
        ]

    utterances = []
    current: Optional[dict] = None

    for word in word_timestamps:
        speaker = word.get("speaker")
        if current is None or current["speaker"] != speaker:
            if current is not None:
                utterances.append(current)
            current = {
                "start": word["start"],
                "end": word["end"],
                "speaker": speaker,
                "words": [word],
            }
        else:
            current["end"] = word["end"]
            current["words"].append(word)

    if current is not None:
        utterances.append(current)

    # build text from words
    for utt in utterances:
        utt["text"] = " ".join(w["word"] for w in utt["words"])
        # strip speaker from word dicts in output
        for w in utt["words"]:
            w.pop("speaker", None)

    return utterances
