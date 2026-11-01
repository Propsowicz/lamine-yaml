"""Find broadcast camera-cut candidates and optionally map them to tactical time."""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def load_segments(path):
    """Each segment is a continuous, bounded correspondence between recordings."""
    if path is None:
        return []
    segments = json.loads(path.read_text(encoding="utf-8"))["segments"]
    previous_b = previous_t = -1.0
    for segment in segments:
        anchors = segment["anchors"]
        if len(anchors) < 2:
            raise ValueError("Each segment needs at least two anchors")
        for anchor in anchors:
            b, t = anchor["broadcast_seconds"], anchor["tactical_seconds"]
            if not (math.isfinite(b) and math.isfinite(t) and b >= 0 and t >= 0):
                raise ValueError("Anchor times must be finite and nonnegative")
            if b <= previous_b or t <= previous_t:
                raise ValueError("Anchors and segments must strictly increase in both timelines")
            previous_b, previous_t = b, t
    return segments


def map_time(seconds, segments):
    for index, segment in enumerate(segments):
        anchors = segment["anchors"]
        for left, right in zip(anchors, anchors[1:]):
            b0, b1 = left["broadcast_seconds"], right["broadcast_seconds"]
            if b0 <= seconds <= b1:
                fraction = (seconds - b0) / (b1 - b0)
                return (left["tactical_seconds"] + fraction *
                        (right["tactical_seconds"] - left["tactical_seconds"]), index)
    return None, None


def features(frame):
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # A broad green mask is descriptive only: close-ups can also have green backgrounds.
    grass = float(np.mean((hsv[:, :, 0] >= 30) & (hsv[:, :, 0] <= 90) &
                          (hsv[:, :, 1] >= 50) & (hsv[:, :, 2] >= 35)))
    histogram = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, norm_type=cv2.NORM_L1)
    return small.astype(np.float32), histogram, grass


def change_score(previous, current):
    # Spatial appearance catches different green-field views; histogram distance
    # contributes evidence for a change in shot composition. This is not a probability.
    pixel = float(np.mean(np.abs(previous[0] - current[0])) / 255)
    histogram = cv2.compareHist(previous[1], current[1], cv2.HISTCMP_BHATTACHARYYA)
    return 0.6 * pixel + 0.4 * histogram


def metadata(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError(f"Invalid video metadata: {path}")
        return {"path": str(path.resolve()), "fps": fps, "frames": count,
                "duration_seconds": count / fps}
    finally:
        capture.release()


def detect(video, info, threshold, min_gap, start, duration):
    capture = cv2.VideoCapture(str(video))
    first = math.ceil(start * info["fps"])
    stop = min(info["frames"], math.ceil((start + duration) * info["fps"])) if duration else info["frames"]
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    previous = None
    pending = None
    decoded = 0
    cuts = []
    try:
        for frame_index in range(first, stop):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Decode stopped unexpectedly at frame {frame_index}")
            decoded += 1
            current = features(frame)
            seconds = frame_index / info["fps"]
            if pending and seconds - pending["broadcast_seconds"] >= min_gap:
                cuts.append(pending)
                pending = None
            if previous is not None:
                score = change_score(previous, current)
                if score >= threshold:
                    row = dict(broadcast_frame=frame_index, broadcast_seconds=seconds,
                               change_score=score, grass_before=previous[2], grass_after=current[2])
                    if pending is None or score > pending["change_score"]:
                        pending = row
            previous = current
            if decoded % 15000 == 0:
                print(f"Analyzed through {seconds:.1f}s ({len(cuts)} cuts)", flush=True)
    finally:
        capture.release()
    if pending:
        cuts.append(pending)
    return cuts, decoded


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broadcast", type=Path, default=root / "data/broadcast/BAR-ATH_broadcast.mp4")
    parser.add_argument("--tactical", type=Path, default=root / "data/tactical/BAR-ATH_tactical.mp4")
    parser.add_argument("--alignment", type=Path, help="JSON containing continuous segments of synchronization anchors")
    parser.add_argument("--output-dir", type=Path, default=root / "output/broadcast_cuts")
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--min-gap", type=float, default=0.5, help="Suppress nearby detections, retaining strongest score")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, help="Optional analysis length in seconds")
    args = parser.parse_args()
    values = [args.threshold, args.min_gap, args.start] + ([] if args.duration is None else [args.duration])
    if (not all(math.isfinite(v) for v in values) or not 0 < args.threshold <= 1 or
            args.min_gap < 0 or args.start < 0 or (args.duration is not None and args.duration <= 0)):
        parser.error("Use finite values: threshold in (0,1], nonnegative start/min-gap, positive duration")
    segments = load_segments(args.alignment)
    broadcast = metadata(args.broadcast)
    tactical = metadata(args.tactical) if segments else None
    if args.start >= broadcast["duration_seconds"]:
        parser.error("Start is past broadcast end")
    for segment in segments:
        for anchor in segment["anchors"]:
            if (anchor["broadcast_seconds"] >= broadcast["duration_seconds"] or
                    anchor["tactical_seconds"] >= tactical["duration_seconds"]):
                parser.error("Alignment anchor lies outside its recording")
    cuts, decoded = detect(args.broadcast, broadcast, args.threshold, args.min_gap, args.start, args.duration)
    for row in cuts:
        seconds, segment = map_time(row["broadcast_seconds"], segments)
        row.update(tactical_seconds=seconds, tactical_frame=None, alignment_segment=segment,
                   mapping_status="outside_alignment" if segments else "no_alignment")
        if seconds is not None:
            frame = int(math.floor(seconds * tactical["fps"] + 0.5))
            if 0 <= frame < tactical["frames"]:
                row.update(tactical_frame=frame, mapping_status="mapped")
            else:
                row.update(tactical_seconds=None, alignment_segment=None, mapping_status="outside_tactical")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["broadcast_frame", "broadcast_seconds", "change_score", "grass_before", "grass_after",
              "tactical_seconds", "tactical_frame", "alignment_segment", "mapping_status"]
    with (args.output_dir / "cuts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cuts)
    summary = dict(broadcast=broadcast, tactical=tactical, alignment=segments,
                   threshold=args.threshold, min_gap=args.min_gap, start=args.start, duration=args.duration,
                   decoded_frames=decoded, cut_candidates=len(cuts),
                   mapped_candidates=sum(row["mapping_status"] == "mapped" for row in cuts),
                   timing="Zero-based frames; seconds = frame / nominal FPS. CFR inputs required.",
                   interpretation="Uncalibrated shot-change scores, not stoppage probabilities. "
                   "Mapped times describe when the broadcast cut occurred, not the earlier stoppage onset. "
                   "Replay contents do not determine the synchronization anchors.")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Found {len(cuts)} candidates; {summary['mapped_candidates']} mapped. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
