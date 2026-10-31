"""Extract the full tactical audio and count exploratory YAMNet whistle candidates.

Run from any directory with the project's .venv. Outputs use the full video clock.
Existing clip-based check_whistle.py is intentionally independent of this runner.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess

SAMPLE_RATE = 16000
HOP = 7680  # 0.48 seconds: YAMNet patch hop.
WINDOW = 15600  # 0.975 seconds including the final STFT window.
MODEL_URL = "https://tfhub.dev/google/yamnet/1"


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def merge_candidates(scores, threshold, duration, gap_seconds):
    """Merge overlapping above-threshold acoustic windows, plus the requested gap."""
    candidates = []
    for index, score in enumerate(scores):
        if score < threshold:
            continue
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
            candidates.append(dict(threshold=threshold, start_seconds=start,
                                   end_seconds=end, peak_seconds=peak_time,
                                   peak_score=float(score), above_threshold_windows=1))
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--video", type=Path,
                        default=root / "data/tactical/BAR-ATH_tactical.mp4")
    parser.add_argument("--wav", type=Path,
                        default=root / "data/tactical/full_audio_16k_mono.wav")
    parser.add_argument("--output-dir", type=Path, default=root / "output/whistle_full")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument("--chunk-hops", type=int, default=120)
    parser.add_argument("--merge-gap", type=float, default=0.0,
                        help="Extra seconds allowed between acoustic windows (default: 0).")
    args = parser.parse_args()
    if not args.ffmpeg:
        parser.error("FFmpeg not found: pass --ffmpeg with its executable path")
    if args.chunk_hops < 4 or args.merge_gap < 0 or any(t <= 0 or t > 1 for t in args.thresholds):
        parser.error("Use chunk-hops >= 4, merge-gap >= 0, and thresholds in (0, 1].")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf
    import tensorflow as tf
    import tensorflow_hub as hub

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.wav.parent.mkdir(parents=True, exist_ok=True)
    ffprobe = str(Path(args.ffmpeg).with_name("ffprobe.exe" if str(args.ffmpeg).endswith(".exe") else "ffprobe"))
    probe = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-show_entries",
        "format=duration,start_time:stream=index,codec_type,sample_rate,channels,start_time,duration",
        "-of", "json", str(args.video)], text=True))
    audio_stream = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    # This POC explicitly verifies the clocks rather than silently dropping a stream offset.
    if abs(float(audio_stream.get("start_time", 0)) - float(video_stream.get("start_time", 0))) > 1 / SAMPLE_RATE:
        raise ValueError("Audio/video start timestamps differ; timeline alignment needs explicit handling.")
    command = [str(args.ffmpeg), "-hide_banner", "-loglevel", "warning", "-nostdin", "-n",
               "-i", str(args.video), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
               "-c:a", "pcm_s16le", str(args.wav)]
    # -n protects an existing extraction. Reuse only if its format and duration agree.
    if not args.wav.exists():
        print("Extracting entire audio stream (no start offset or duration limit)...", flush=True)
        subprocess.run(command, check=True)
    info = sf.info(args.wav)
    if info.samplerate != SAMPLE_RATE or info.channels != 1 or info.subtype != "PCM_16":
        raise ValueError("Expected mono PCM16 WAV at 16 kHz")
    duration = info.frames / SAMPLE_RATE
    if abs(duration - float(audio_stream["duration"])) > 0.1:
        raise ValueError("WAV duration does not match the complete source audio")
    print(f"Audio: {duration:.6f}s; {info.frames} samples; mono 16 kHz", flush=True)

    model = hub.load(MODEL_URL)
    with tf.io.gfile.GFile(model.class_map_path().numpy().decode("utf-8")) as handle:
        names = [row["display_name"] for row in csv.DictReader(handle)]
    whistle_index = names.index("Whistle")
    # Match YAMNet's final padding: minimum 975 ms, then round up to a 480 ms hop.
    n_windows = 1 + max(0, math.ceil((info.frames - WINDOW) / HOP))
    all_scores = np.empty(n_windows, dtype=np.float32)
    with sf.SoundFile(args.wav) as handle:
        for first in range(0, n_windows, args.chunk_hops):
            count = min(args.chunk_hops, n_windows - first)
            handle.seek(first * HOP)
            # Every retained patch has all 975 ms of context; chunks overlap by 495 ms.
            waveform = handle.read((count - 1) * HOP + WINDOW, dtype="float32")
            result = model(waveform)[0].numpy()[:, whistle_index]
            if len(result) < count:
                raise AssertionError(f"Expected at least {count} patches, received {len(result)}")
            # YAMNet can add a trailing padded patch due to float32 padding arithmetic.
            # Keep only the requested global windows; the next chunk owns later ones.
            all_scores[first:first + count] = result[:count]
            print(f"Scored {min(n_windows, first + count)}/{n_windows} windows "
                  f"({min(duration, (first + count) * HOP / SAMPLE_RATE):.1f}s)", flush=True)

        # Independently re-score four windows straddling the first chunk boundary.
        first = args.chunk_hops - 2
        if n_windows >= first + 4:
            handle.seek(first * HOP)
            reference = model(handle.read(3 * HOP + WINDOW, dtype="float32"))[0].numpy()[:4, whistle_index]
            boundary_error = float(np.max(np.abs(reference - all_scores[first:first + 4])))
            if not np.allclose(reference, all_scores[first:first + 4], atol=1e-5, rtol=1e-4):
                raise AssertionError(f"Chunk boundary mismatch: {boundary_error}")
        else:
            boundary_error = None

    starts = np.arange(n_windows) * HOP / SAMPLE_RATE
    ends = np.minimum(starts + WINDOW / SAMPLE_RATE, duration)
    centers = np.minimum(starts + WINDOW / SAMPLE_RATE / 2, duration)
    if not np.all(np.isfinite(all_scores)) or starts[-1] >= duration:
        raise AssertionError("Invalid model scores or timestamps")
    write_csv(args.output_dir / "scores.csv",
              ["window_index", "start_seconds", "end_seconds", "video_seconds", "whistle_score"],
              (dict(window_index=i, start_seconds=float(starts[i]), end_seconds=float(ends[i]),
                    video_seconds=float(centers[i]), whistle_score=float(all_scores[i]))
               for i in range(n_windows)))
    candidates = []
    counts = {}
    for threshold in sorted(set(args.thresholds)):
        rows = merge_candidates(all_scores, threshold, duration, args.merge_gap)
        counts[str(threshold)] = len(rows)
        candidates.extend(dict(candidate_id=i + 1, **row) for i, row in enumerate(rows))
    write_csv(args.output_dir / "candidates.csv",
              ["threshold", "candidate_id", "start_seconds", "end_seconds", "peak_seconds",
               "peak_score", "above_threshold_windows"], candidates)

    fig, (overview, detail) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    for axis in (overview, detail):
        axis.plot(centers / 60, all_scores, linewidth=0.6, color="steelblue")
        for threshold, count in counts.items():
            axis.axhline(float(threshold), linestyle="--", linewidth=0.8,
                         label=f"{threshold}: {count} candidates")
        axis.set_ylabel("YAMNet Whistle score")
        axis.grid(alpha=0.2)
        axis.set_xlim(0, duration / 60)
    overview.set_title("Full tactical video: exploratory whistle candidates (not verified referee whistles)")
    overview.legend(loc="upper right")
    detail.set_ylim(0, max(args.thresholds) * 2.5)
    detail.set_title("Low-score detail; peaks above the displayed range are clipped")
    detail.set_xlabel("Minutes from tactical video start (includes pre-match and halftime)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "scores.png", dpi=160)
    plt.close(fig)
    summary = dict(video=str(args.video.resolve()), wav=str(args.wav.resolve()), model=MODEL_URL,
                   extraction_command=command, source_probe=probe, audio_duration_seconds=duration,
                   audio_samples=info.frames, video_duration_seconds=float(video_stream["duration"]),
                   video_audio_duration_difference_seconds=float(video_stream["duration"]) - duration,
                   sample_rate=SAMPLE_RATE, channels=1, score_windows=n_windows,
                   hop_seconds=HOP / SAMPLE_RATE, acoustic_window_seconds=WINDOW / SAMPLE_RATE,
                   timestamp_convention="video_seconds is acoustic window center; final window clipped to audio end",
                   chunk_hops=args.chunk_hops, overlap_seconds=(WINDOW - HOP) / SAMPLE_RATE,
                   boundary_max_absolute_error=boundary_error, merge_gap_seconds=args.merge_gap,
                   candidate_counts=counts, maximum_score=float(all_scores.max()),
                   maximum_score_video_seconds=float(centers[all_scores.argmax()]),
                   limitations="Exploratory thresholds, not calibrated; crowd whistles/music can trigger scores. "
                   "Merged candidate counts are not verified referee whistle counts. Timing resolution is 0.48 s; "
                   "candidate bounds are model acoustic windows, not exact whistle onsets/offsets.")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
