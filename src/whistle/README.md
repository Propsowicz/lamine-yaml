# Whistle event detection

Call the event detector with an inclusive, zero-based frame range from the tactical video:

```python
from whistle.full_video_whistle import detect_whistle_events

events = detect_whistle_events(start_frame=10_000, end_frame=12_500)
for event in events:
    print(event.start_frame_id, event.end_frame_id, event.peak_score)
```

The detector returns `list[WhistleEvent]`. Each event provides start/end frame IDs
and elapsed timestamps, peak frame/time/score, the threshold used, and the number
of above-threshold YAMNet windows merged into it. The default threshold is `0.01`;
pass `threshold=...` or `merge_gap=...` to adjust candidate grouping. These are
potential acoustic whistle events, not verified referee whistles.

Defaults use the game's tactical video and FPS. `video`, `wav`, `game`, `data_dir`,
`fps`, `ffmpeg`, and `model_cache` can be supplied as keyword arguments when using
other assets or cache locations. FFmpeg and FFprobe must be available (or FFmpeg's
executable can be passed as `ffmpeg`). The extracted segment WAV is cached under
`output/whistle_full/` with the video name and frame bounds; the YAMNet model is
cached under `output/tfhub_cache/`.

Only audio from the requested frame interval is extracted and scored. YAMNet uses a
0.48-second hop and 0.975-second acoustic windows, so it may pad the segment edges
when a full window of samples is unavailable. Event bounds are clipped to the
requested interval; timing is approximate at the model's 0.48-second resolution.
