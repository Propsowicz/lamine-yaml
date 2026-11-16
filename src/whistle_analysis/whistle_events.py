"""Find potential whistle events within an inclusive video-frame interval."""

import csv
import json
import math
import os
import soundfile as sf
import shutil
import subprocess
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from data_prepare.games import GAME, GAMES, video_path
from utils.time_format import format_elapsed_time

SAMPLE_RATE = 16000
HOP = 7680  # 0.48 seconds: YAMNet patch hop.
WINDOW = 15600  # 0.975 seconds including the final STFT window.
MODEL_URL = "https://tfhub.dev/google/yamnet/1"


@dataclass(frozen=True)
class WhistleEvent:
    """A merged run of above-threshold YAMNet whistle windows."""

    start_frame_id: int
    start_timestamp: timedelta
    end_frame_id: int
    end_timestamp: timedelta
    peak_frame_id: int
    peak_timestamp: timedelta
    peak_score: float
    threshold: float
    above_threshold_windows: int

    @property
    def start_time(self) -> str:
        return format_elapsed_time(self.start_timestamp)

    @property
    def end_time(self) -> str:
        return format_elapsed_time(self.end_timestamp)


def load_yamnet(hub):
    """Validate the cached download before asking TensorFlow to load it."""
    model_dir = Path(hub.resolve(MODEL_URL))
    if not any((model_dir / name).is_file()
               for name in ("saved_model.pb", "saved_model.pbtxt")):
        raise RuntimeError(
            f"Incomplete YAMNet download at {model_dir}: missing saved_model.pb. "
            "Use a new empty model_cache directory to download again."
        )
    return hub.load(str(model_dir))


def merge_candidates(scores, first_window, threshold, duration, gap_seconds):
    """Merge contiguous above-threshold windows into acoustic intervals."""
    candidates = []
    for local_index, score in enumerate(scores):
        if score < threshold:
            continue
        index = first_window + local_index
        start = index * HOP / SAMPLE_RATE
        end = min(duration, start + WINDOW / SAMPLE_RATE)
        peak_time = min(duration, start + WINDOW / SAMPLE_RATE / 2)
        if candidates and start <= candidates[-1]["end_seconds"] + gap_seconds:
            current = candidates[-1]
            current["end_seconds"] = end
            current["above_threshold_windows"] += 1
            if score > current["peak_score"]:
                current.update(peak_score=float(score), peak_seconds=peak_time)
        else:
            candidates.append(dict(
                start_seconds=start,
                end_seconds=end,
                peak_seconds=peak_time, 
                peak_score=float(score),
                above_threshold_windows=1)
            )
    return candidates


def _extract_audio(video: Path, wav: Path, ffmpeg: str, sample_rate: int,
                   start_seconds: float, end_seconds: float):
    """Extract and validate only the requested audio interval."""
    wav.parent.mkdir(parents=True, exist_ok=True)
    ffprobe = str(Path(ffmpeg).with_name(
        "ffprobe.exe" if str(ffmpeg).lower().endswith(".exe") else "ffprobe"))
    probe = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-show_entries",
        "format=duration,start_time:stream=index,codec_type,sample_rate,channels,start_time,duration",
        "-of", "json", str(video)], text=True))
    audio_stream = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    if abs(float(audio_stream.get("start_time", 0)) -
           float(video_stream.get("start_time", 0))) > 1 / sample_rate:
        raise ValueError("Audio/video start timestamps differ; timeline alignment needs explicit handling.")

    source_duration = float(audio_stream["duration"])
    if start_seconds >= source_duration:
        raise ValueError("start_frame is beyond the end of the source video.")
    segment_end = min(end_seconds, source_duration)
    if not wav.exists():
        subprocess.run([
            str(ffmpeg), "-hide_banner", "-loglevel", "warning", "-nostdin", "-n",
            "-i", str(video), "-ss", f"{start_seconds:.9f}",
            "-t", f"{segment_end - start_seconds:.9f}",
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-c:a", "pcm_s16le", str(wav),
        ], check=True)

    info = sf.info(wav)
    if info.samplerate != sample_rate or info.channels != 1 or info.subtype != "PCM_16":
        raise ValueError(f"Expected mono PCM16 WAV at {sample_rate} Hz")
    expected_duration = segment_end - start_seconds
    if abs(info.frames / sample_rate - expected_duration) > 0.1:
        raise ValueError("WAV duration does not match the requested audio segment")
    return info, segment_end


def detect_whistle_events(
    start_frame: int,
    end_frame: int,
    game: str,
    video: str | Path,
    *,
    data_dir: str | Path | None = None,
    threshold: float = 0.01,
    merge_gap: float = 0.0,
    chunk_hops: int = 120,
    model_cache: str | Path | None = None,
    ffmpeg: str | Path | None = None,
) -> list[WhistleEvent]:
    """Return potential whistle events in an inclusive, zero-based video-frame range.

    By default the game's tactical video, FPS, and YAMNet cache are used. Audio is
    extracted only from the requested interval; its WAV cache is keyed by that range.
    """
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("Use 0 <= start_frame <= end_frame (both inclusive).")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1].")
    if merge_gap < 0 or chunk_hops < 4:
        raise ValueError("Use merge_gap >= 0 and chunk_hops >= 4.")

    fps = GAMES[game]["fps"]
    video = Path(video)
    wav = Path(__file__).resolve().parents[2] / "output" / "whistle_full" / f"{video.stem}_{start_frame}_{end_frame}_audio_16k_mono.wav"
    model_cache = (Path(model_cache) if model_cache is not None
                   else Path(__file__).resolve().parents[2] / "output" / "tfhub_cache")
    ffmpeg = str(ffmpeg or shutil.which("ffmpeg") or "")
    if not ffmpeg:
        raise ValueError("FFmpeg not found: install it or provide its executable path with ffmpeg.")

    model_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TFHUB_CACHE_DIR"] = str(model_cache.resolve())

    segment_start = start_frame / fps
    requested_end = (end_frame + 1) / fps
    info, segment_end = _extract_audio(
        video, wav, ffmpeg, SAMPLE_RATE, segment_start, requested_end)
    duration = info.frames / SAMPLE_RATE
    first_time = 0.0
    last_time = min(duration, segment_end - segment_start)
    # Center-based selection gives each video frame interval a clear and stable meaning.
    first_window = max(0, math.ceil((first_time - WINDOW / SAMPLE_RATE / 2) /
                                    (HOP / SAMPLE_RATE)))
    after_last_window = min(
        1 + max(0, math.ceil((info.frames - WINDOW) / HOP)),
        math.ceil((last_time - WINDOW / SAMPLE_RATE / 2) / (HOP / SAMPLE_RATE)),
    )
    # Very short requested segments still get one patch; YAMNet pads its tail.
    after_last_window = max(first_window + 1, after_last_window)
    if after_last_window <= first_window:
        return []

    model = load_yamnet(hub)
    with tf.io.gfile.GFile(model.class_map_path().numpy().decode("utf-8")) as handle:
        names = [row["display_name"] for row in csv.DictReader(handle)]
    whistle_index = names.index("Whistle")
    count_windows = after_last_window - first_window
    scores = np.empty(count_windows, dtype=np.float32)
    with sf.SoundFile(wav) as handle:
        for local_first in range(0, count_windows, chunk_hops):
            count = min(chunk_hops, count_windows - local_first)
            global_first = first_window + local_first
            handle.seek(global_first * HOP)
            waveform = handle.read((count - 1) * HOP + WINDOW, dtype="float32")
            result = model(waveform)[0].numpy()[:, whistle_index]
            if len(result) < count:
                raise AssertionError(f"Expected at least {count} patches, received {len(result)}")
            scores[local_first:local_first + count] = result[:count]

    if not np.all(np.isfinite(scores)):
        raise AssertionError("Model returned non-finite whistle scores")
    candidates = merge_candidates(scores, first_window, threshold, duration, merge_gap)
    events = []
    for candidate in candidates:
        local_start = max(first_time, candidate["start_seconds"])
        local_end = min(last_time, candidate["end_seconds"])
        local_peak = candidate["peak_seconds"]
        if local_end < local_start:
            continue
        start_s = segment_start + local_start
        end_s = segment_start + local_end
        peak_s = segment_start + local_peak
        start_id = max(start_frame, min(end_frame, round(start_s * fps)))
        end_id = max(start_frame, min(end_frame, round(end_s * fps)))
        peak_id = max(start_frame, min(end_frame, round(peak_s * fps)))
        events.append(WhistleEvent(
            start_frame_id=start_id,
            start_timestamp=timedelta(seconds=start_s),
            end_frame_id=end_id,
            end_timestamp=timedelta(seconds=end_s),
            peak_frame_id=peak_id,
            peak_timestamp=timedelta(seconds=peak_s),
            peak_score=candidate["peak_score"],
            threshold=threshold,
            above_threshold_windows=candidate["above_threshold_windows"],
        ))
    return events