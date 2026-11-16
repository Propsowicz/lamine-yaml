# CK - Posession detection system

## How to run it

1. Clone this repository.
2. Create `data/tactical` directory (as shown in data structure below) and place match's data bundle there.
3. Create `detection_models directory` in `src/data_prepare` and place football-player-detection.pt there (https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q - rememmber to trim version of model to match name given above).  
4. Run `python main.py` in src/ directory. Since the whole detection of match is being performed it can take a while to complete process.
5. Output is placed in `src/CK.csv`.

### Repository structure: 
```
src
    data_prepare
        detection_models
    models
    utils
    ....
data
    tactical
        frame_pics
```

## The Approach

Our approach was based on few simple rules:
- Possession is the period during which a team controls the ball. This includes situations where a player has direct control of the ball as well as sequences where teammates pass the ball between each other. To detect possession, we created a chain of ball-touch events that tracks the positions of the ball and players, calculates distances between them, and determines the type of ball-touch event that occurred.
- Dead ball occurs whenever the ball goes out of the pitch or the game is stopped.
- Unknown possession is assigned to every remaining frame that cannot be classified as either possession or dead ball.

## Problems

During implementation, we encountered several challenges and had to find ways to solve or work around them.

1. Missing ball detections

The ball was missing from some frames. For short gaps, simple interpolation was sufficient to fill in the missing positions. Longer gaps were usually associated with specific events occurring on the pitch.

2. Missing player detections

To recover missing players, we performed our own player detection on the tactical dataset. Since missing players were usually in secondary positions, we ran detection once per second and propagated the detected positions to the following 25 frames.

We then combined the players detected by our model with those already available in the ATD data, avoiding duplicates. Only players that were not already present in the ATD data were added to the final player collection.

Newly detected players were assigned to the correct team using a combination of shirt colors, our own team detection, and existing team assignments from the ATD data. For example, if we detected five players belonging to Team 1, and four of them were already present in the ATD data, we assigned the newly detected player to the same team.

3. Player overlaps and incorrect tracking assignments

We discovered that player tracking could fail when two players came very close to each other, for example, when fighting for position or challenging for the ball. In these situations, one player's detection would temporarily disappear. When tracking resumed, that player could be assigned an incorrect team.

To address this problem, we introduced an additional player-tracking algorithm. Whenever a player disappeared and we identified the overlap with another player as the likely cause, we created a virtual representation of the missing player. This virtual player used the visible player's position while preserving the missing player's original ID and team assignment.

Once the missing player was detected again, we adjusted the reappeared player's ID and team assignment to match the virtual representation.

4. Half-time detection

tbd

## Rules

As mentioned above, two simple rules formed the foundation of our approach. Based on them, we established the following labeling procedure:


1. First labeling layer – team possession from ball-touch events

We defined three possible states for ball-touch events:

`POSSESSION` – when a player maintains control of the ball for at least 3 consecutive frames (i.e., stays close to the ball).

`PASS` – when a player touches the ball briefly and passes it to a teammate.

`UNCONFIRMED` – when neither of the above conditions is met.

Based on the detected ball-touch events, we assigned team labels to frames.

Frames belonging to POSSESSION or PASS events were assigned to the corresponding team. For UNCONFIRMED events, we assigned the label of the team that gained possession in the next event.

2. Second labeling layer – ball out of play

We checked whether the ball was outside the pitch. Frames in which the ball was detected outside the pitch were assigned the `D` label.

3. Third labeling layer – set-piece events

Finally, we combined the smart-tagging events with the detected ball-touch events.

Whenever a SET-PIECE event was detected by smart-tagging, we labeled the interval between the event's starting frame and the first subsequent ball-touch event as `D`.


## Limitations

- known limitation is our players detection - it is not perfect. It tends to mess up team assignement,
- there are some match breaks that were not properly recognized,
- missing ball is just primitevly fixed - but it only works for short ranges of ball miss,
- ball touches events detection is not perfect - it tends to miss some touches (especially headers)

## Possible fixes

- the plan were to detect refree's whistle to recognize match breaks better - we could not manage to make it work good enough to use it in our system, 
- the thing we did not touch was broadcast data - maybe we could took some more data from there.

## Internal assets

1. tactical_atd - used as a main source of players/ball positions
2. tactical_homography - used in our detection algorithm
3. tactical_smart_tagging - used to get "SET_PIECE" events
4. mp4 video clip - used to detect missing players position using free model and our algorithm

## External assets

1. football-player-detection.pt - used in players detection, source: https://github.com/roboflow/sports, license: https://github.com/roboflow/sports/blob/main/LICENSE

## Smart Tagging events - to remove?

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
