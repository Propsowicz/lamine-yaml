# Possesion

## Plan
1. Detect players and their teams
2. Normalize with given data
3. Analyze events data
4. ....


## Strcuture
```
src
    data_prepare
        detection_models (here is model - ignored)
    models
    utils
data*
    tactical
        frame_pics
output*
```
* - git ignored

## Smart Tagging events

From the repository root, load the tactical XML with:

```python
from src.data_prepare.smart_tagging import load_smart_tagging

events = load_smart_tagging()
clip_events = load_smart_tagging(start_frame=11185, end_frame=18684)
kickoffs = load_smart_tagging(event_types="Kick Off (Start)")
```

The loader reads `data/tactical/BAR-ATH_tactical_smart_tagging.xml` by default.
Set `MTAH_DATA_DIR` or pass `data_dir` to use a different data directory.
Each event has a `code`, `tags`, and start and end times in seconds and tactical
video frames. Frame intervals include `start_frame` and exclude `end_frame`.
Events are sorted by `start_frame`.
`start_time` and `end_time` show elapsed tactical video time as `HH:MM:SS`,
with fractional seconds dropped.
The optional filter includes both requested frames and returns any event that
overlaps that range with its original boundaries. Either filter bound can be omitted.
Pass a string or a list of strings as `event_types` to match the XML's `Type`
tag case-insensitively. You can combine this with the frame-range filter.
