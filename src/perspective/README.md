# Broadcast perspective-change candidates

Run from the repository root (uses existing OpenCV and NumPy dependencies):

```powershell
.venv/Scripts/python.exe src/perspective/broadcast_cuts.py --start 600 --duration 60
```

Omit `--start` and `--duration` to analyze the whole recording. `--broadcast`,
`--tactical`, and `--output-dir` override the default local paths. Each run
replaces `cuts.csv` and `summary.json` in its output directory.

The script compares adjacent decoded frames at reduced resolution using spatial
color differences and HSV histogram distance. Nearby detections are suppressed,
retaining the strongest change. Every frame is analyzed; it does not require a
model download. The default `--threshold 0.22` is exploratory: lower it for more
sensitivity, raise it for fewer candidates. Fast pans, graphics, lighting changes,
and flashes can trigger it. Subtle cuts and gradual transitions can be missed.
Green-pixel fractions before and after each cut are exported as review features;
they do not classify wide shots, players, or replays.

## Mapping to the tactical recording

The videos are not synchronized. Without `--alignment`, tactical columns remain
empty; broadcast cuts are still exported. Do not use the tracking frame offsets
in `games.py` as video-to-video offsets.

Create a JSON file with corresponding **recording times in seconds**, measured
using the same live event visible in both recordings. This example is illustrative,
not an alignment for the supplied match:

```json
{
  "segments": [
    {
      "anchors": [
        {"broadcast_seconds": 100, "tactical_seconds": 450},
        {"broadcast_seconds": 1000, "tactical_seconds": 1350}
      ]
    },
    {
      "anchors": [
        {"broadcast_seconds": 1200, "tactical_seconds": 4550},
        {"broadcast_seconds": 2000, "tactical_seconds": 5350}
      ]
    }
  ]
}
```

```powershell
.venv/Scripts/python.exe src/perspective/broadcast_cuts.py --alignment alignment.json
```

Mapping interpolates between anchors within each segment, never extrapolating
or crossing gaps between segments. Use separate segments across recording edits
or missing footage, such as a shortened halftime. Add anchors to measure drift.
Verify alignment on additional events, and avoid using replay content as an
anchor: it depicts an earlier moment. Synchronization is manual in this version.

`cuts.csv` contains broadcast seconds/frame, appearance-change score (not a
probability), green fractions, mapped tactical seconds/frame, and mapping status.
Frames are zero-based; the cut frame is the first frame of the detected new view.
Tactical frames are rounded to the nearest frame. Both sources must have constant
frame rate: times use frame index divided by metadata FPS. The summary records
settings, source metadata, anchors, and counts for reproducibility.

## Using this as a stoppage feature

A camera cut is a candidate signal, not a stoppage label. Cuts also happen during
live play, while stoppages can remain in the main camera view. A replay or close-up
may start seconds after the actual stoppage, and the return cut may occur before
the restart. This detector does not identify replay intervals or infer ball-in-play.

Review candidates in both videos, measure their delay from actual stoppages, and
combine them with tactical whistle scores and ball/player movement. Assess missed
stoppages as well as false alarms before choosing a threshold or a look-back window.

Tests:

```powershell
.venv/Scripts/python.exe -m unittest discover -s src/perspective -p 'test_*.py'
```
