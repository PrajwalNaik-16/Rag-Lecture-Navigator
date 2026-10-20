"""Audio -> Segment[], cached to JSON.

The cache check at the top of transcribe() is the single most important line
in this codebase. Re-transcribing a 40-video playlist by accident costs hours.
"""

import json
import os
from pathlib import Path
import sysconfig
from types import SimpleNamespace

from ytrag.config import (
    TRANSCRIPT_DIR,
    WHISPER_BATCH,
    WHISPER_BEAM,
    WHISPER_COMPUTE,
    WHISPER_DEVICE,
    WHISPER_LANG,
    WHISPER_MODEL,
    WHISPER_WINDOW_SECONDS,
)
from ytrag.models import Segment, Video
from ytrag.playlist import delete_audio, download_audio

# Loading large-v3 takes a while, so keep one instance per (model, device).
_MODEL_CACHE: dict[tuple, object] = {}
_CUDA_DLL_HANDLES = []
_CUDA_DLLS_ADDED = False


def _add_cuda_dll_dirs() -> None:
    """Expose pip-installed NVIDIA runtime DLLs to CTranslate2 on Windows."""
    global _CUDA_DLLS_ADDED
    if os.name != "nt":
        return
    if _CUDA_DLLS_ADDED:
        return

    site_packages = Path(sysconfig.get_paths()["purelib"])
    for package in ("cublas", "cudnn", "cuda_nvrtc"):
        dll_dir = site_packages / "nvidia" / package / "bin"
        if not dll_dir.is_dir():
            continue
        try:
            # Keep handles alive: Windows removes a DLL directory when its
            # returned handle is closed or garbage-collected.
            _CUDA_DLL_HANDLES.append(os.add_dll_directory(str(dll_dir)))
        except (AttributeError, OSError):
            pass
        os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
    _CUDA_DLLS_ADDED = True


def _resolve_device() -> tuple[str, str]:
    """Pick device and compute type, preferring CUDA when it actually works."""
    device = WHISPER_DEVICE
    if device == "auto":
        try:
            _add_cuda_dll_dirs()
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"

    compute = WHISPER_COMPUTE or ("float16" if device == "cuda" else "int8")
    return device, compute


def get_model(model_name: str = WHISPER_MODEL):
    """Load (and cache) a faster-whisper model, falling back to CPU if CUDA fails.

    A GPU newer than the shipped ctranslate2 build will load and then fail at
    the first inference, so the fallback here is deliberately broad.
    """
    from faster_whisper import WhisperModel

    device, compute = _resolve_device()
    key = (model_name, device, compute)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    try:
        model = WhisperModel(model_name, device=device, compute_type=compute)
    except Exception as exc:
        if device == "cpu":
            raise
        print(f"[transcribe] CUDA unavailable ({exc}); falling back to CPU int8.")
        device, compute = "cpu", "int8"
        key = (model_name, device, compute)
        model = WhisperModel(model_name, device=device, compute_type=compute)

    _MODEL_CACHE[key] = model
    return model


def get_batched_model(model_name: str = WHISPER_MODEL):
    """Wrap the model in faster-whisper's batched pipeline.

    Same weights, same output quality — it just feeds several VAD-detected
    speech regions through the encoder at once instead of one at a time.
    """
    from faster_whisper import BatchedInferencePipeline

    model = get_model(model_name)
    key = ("batched", id(model))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = BatchedInferencePipeline(model=model)
    return _MODEL_CACHE[key]


def run_whisper(audio_path: str, language: str, model_name: str = WHISPER_MODEL):
    """Transcribe one audio file. Returns (segment_iterator, info).

    vad_filter drops silence outright, and condition_on_previous_text=False
    stops Whisper's loop hallucination — on a long pause or a music sting it
    otherwise repeats the previous phrase over and over. The batched pipeline
    treats each speech region independently, so it never conditions on
    previous text in the first place.
    """
    # faster-whisper extracts features for the full input before decoding.
    # For a long lecture this can require hundreds of MB in one allocation
    # (and fail on ordinary 8 GB machines). Decode once, then process bounded
    # overlapping windows and restore each segment's original timestamp.
    from faster_whisper.audio import decode_audio

    import numpy as np

    sample_rate = 16_000
    audio = decode_audio(audio_path, sampling_rate=sample_rate)
    duration = len(audio) / sample_rate
    window_seconds = max(60, WHISPER_WINDOW_SECONDS)
    window_samples = window_seconds * sample_rate

    def _transcribe(source):
        if WHISPER_BATCH > 0:
            try:
                return get_batched_model(model_name).transcribe(
                    source,
                    language=language,
                    beam_size=WHISPER_BEAM,
                    batch_size=WHISPER_BATCH,
                    vad_filter=True,
                )
            except (ImportError, AttributeError, TypeError) as exc:
                # Only fall back when this build genuinely cannot batch.
                print(f"[transcribe] batched pipeline unavailable ({exc}); using sequential.")

        return get_model(model_name).transcribe(
            source,
            language=language,
            beam_size=WHISPER_BEAM,
            condition_on_previous_text=False,
            vad_filter=True,
        )

    if duration <= window_seconds:
        return _transcribe(audio_path)

    print(
        f"[transcribe] long audio ({duration / 60:.1f} min); "
        f"processing in {window_seconds}s windows."
    )
    overlap_seconds = 5
    step_samples = window_samples - overlap_seconds * sample_rate
    all_segments = []
    info = None

    for start_sample in range(0, len(audio), step_samples):
        end_sample = min(start_sample + window_samples, len(audio))
        window = np.asarray(audio[start_sample:end_sample], dtype=np.float32)
        segments, info = _transcribe(window)
        offset = start_sample / sample_rate
        overlap = overlap_seconds if start_sample else 0
        for segment in segments:
            # Drop segments fully inside the duplicated overlap while keeping
            # the original timeline for every retained result.
            if overlap and segment.end <= overlap:
                continue
            all_segments.append(
                SimpleNamespace(
                    start=offset + segment.start,
                    end=offset + segment.end,
                    text=segment.text,
                )
            )
        if end_sample == len(audio):
            break

    return iter(all_segments), SimpleNamespace(duration=duration, language=language)


def transcript_path(video_id: str, directory: Path | None = None) -> Path:
    return (directory or TRANSCRIPT_DIR) / f"{video_id}.json"


def load_transcript(video_id: str, directory: Path | None = None) -> dict | None:
    """Return the cached transcript dict, or None if absent/corrupt."""
    path = transcript_path(video_id, directory)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or "segments" not in data:
        return None
    return data


def cached_video_ids(directory: Path | None = None) -> list[str]:
    return sorted(p.stem for p in (directory or TRANSCRIPT_DIR).glob("*.json"))


def segments_from_transcript(data: dict) -> list[Segment]:
    return [
        Segment(start=float(s["start"]), end=float(s["end"]), text=s["text"])
        for s in data.get("segments", [])
    ]


def _write_transcript(video: Video, language: str, model_name: str, segments: list[Segment]) -> None:
    """Write atomically.

    A Ctrl-C part way through a plain write leaves a truncated JSON file that
    the cache check then trusts forever. Temp file + os.replace makes the
    transcript either fully there or not there at all.
    """
    payload = {
        "video_id": video.video_id,
        "title": video.title,
        "language": language,
        "model": model_name,
        "duration": video.duration,
        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
    }
    final = transcript_path(video.video_id)
    tmp = final.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, final)


def transcribe(
    video: Video,
    force: bool = False,
    language: str | None = None,
    model_name: str = WHISPER_MODEL,
    keep_audio: bool = False,
) -> list[Segment]:
    """Transcribe a video, using the cached transcript when one exists."""
    language = language or WHISPER_LANG

    if not force:
        cached = load_transcript(video.video_id)
        if cached is not None:
            return segments_from_transcript(cached)

    audio_path = download_audio(video, force=force)
    raw_segments, info = run_whisper(str(audio_path), language, model_name)

    segments = [
        Segment(start=float(s.start), end=float(s.end), text=s.text.strip())
        for s in raw_segments
        if s.text and s.text.strip()
    ]

    if not video.duration and getattr(info, "duration", None):
        video.duration = int(info.duration)

    _write_transcript(video, language, model_name, segments)

    if not keep_audio:
        delete_audio(video.video_id)

    return segments
