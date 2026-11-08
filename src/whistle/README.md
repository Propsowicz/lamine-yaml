# Full tactical-video whistle POC

Run with the actual project's virtual environment from the project root:

```powershell
.venv/Scripts/python.exe src/whistle/full_video_whistle.py --ffmpeg 'PATH/TO/ffmpeg.exe'
```

FFprobe must be alongside FFmpeg. The defaults use only
`data/tactical/BAR-ATH_tactical.mp4`, never the unsynchronized broadcast video.
The original `check_whistle.py` remains a separate two-minute experiment.

YAMNet downloads are cached persistently in `output/tfhub_cache/`, avoiding
incomplete downloads in the shared Windows temporary TF Hub cache. Override
this location with `--model-cache PATH` (takes precedence over `TFHUB_CACHE_DIR`).
If validation reports a missing `saved_model.pb`, pass a new empty cache
directory to download again. The existing extracted WAV can still be reused.

The runner extracts the entire first audio stream into
`data/tactical/full_audio_16k_mono.wav` (mono PCM16, 16 kHz). There is no seek or
duration limit. An existing WAV is reused after checking its format and duration;
remove or rename it explicitly if the source changes.

Outputs in `output/whistle_full/`:

- `scores.csv`: every Whistle score, window bounds, and center time in seconds
  from tactical-video start, including pre-match and halftime.
- `candidates.csv`: intervals merged separately at each threshold, with peak
  score/time and number of above-threshold windows. IDs restart per threshold.
- `summary.json`: counts, source metadata, extraction command, duration checks,
  chunk-boundary verification, and timing conventions.
- `scores.png`: full timeline and a low-score detail view.

YAMNet uses a 0.48-second hop. Its nominal 0.96-second spectrogram patch needs
0.975 seconds of waveform because of the final STFT window. Timestamps here use
the center of that acoustic support (start + 0.4875 seconds), clipped at audio
end. They are approximate event times, not exact whistle onsets.

Default chunks retain 120 windows (57.6 seconds of starts), with 0.495 seconds
of waveform overlap so every retained patch has full context. Extra padded
patches are discarded. Only the actual file end needs zero padding. A separate
four-window inference spanning the first chunk boundary checks score agreement.
Candidates are merged globally, so a whistle crossing a chunk boundary is not
counted twice. Above-threshold acoustic windows are merged when they overlap;
`--merge-gap` optionally permits an extra gap (default zero).

The default thresholds 0.005, 0.01, and 0.02 are exploratory. Counts are candidate
intervals, not validated referee whistles: crowd whistles and music may trigger
the generic AudioSet Whistle class, and real whistles may be missed. Calibration
requires manually reviewed positive and negative examples.

Model timing reference: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet

## Completed run

The delivered tactical audio contains 127,563,093 mono samples at 16 kHz
(7,972.6933125 seconds, approximately 2:12:52.693). Audio and video both start
at zero. The video lasts 7,972.76 seconds; the 66.7 ms difference is in the source.

All 16,609 score windows were processed. Candidate interval counts were:

| Threshold | Candidates |
| --- | ---: |
| 0.005 | 261 |
| 0.01 | 140 |
| 0.02 | 56 |

The maximum Whistle score was 0.130830 at video 2:04:03.3675. The independent
chunk-boundary comparison had maximum absolute error 0.0. Synthetic checks
also covered threshold filtering, overlapping-window merging, merging across
the chunk boundary, and clipping to the file end. Candidate CSV row counts
were checked independently against the summary.
