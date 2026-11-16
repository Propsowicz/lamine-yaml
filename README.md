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

## "the approach"

Our approach was based on few simple rules:
- the possesion is a time that ball spend under controll of a team - either when one player is controlling it or pass between each other. To detect that we created a chain of ball touches that track ball's and players positions, calculate distances and choose what type of ball touch had place,
- every time ball is out of the pitch or game is stopped - there is a dead ball,
- unknown posession for every other frame.

## "problems"

During implementing we faced a few problems and had to decide how to solve them/ work around them.

1. Ball is missing in some frames - simple interpolation done the trick for quick misses. The longer ones were connected with some events from the pitch.
2. Some players were missing - to fix that we performed a whole players detection on tactical dataset. Since, for most of the times, missing players were in secondary positions we performed detection every second and just copied from one detection to next 25 frames. Then the combination of all players (the ones from atd data and from our detection) had place. Only players that were not a part of atd data were added to final players collection. Detected players were assigned to correct team based on color of their shirt, our own detection of teams and team assigns from atd data, i.e. when we have detected that 5 players were part of team#1, 4 of them were already part of atd data - we just added detected player to team where these 4 players belong.
3. We discovered that during the match, whenever "collision" of players appeared (two players were figthing for position/ball and were really close to each other) one of them would dissapair, and then after he is tracked once again he would be assigned to a wrong team. To solve this problem we introduced an algorith to track players, and whenever one of them is missing (and we clearly see that it is because of distance to other player) we add him virtually (by copying the visible player position and missing player team and id). After he is tracked again, we adjust "appeared" player's id and team assignment to hte virtual one.
4. We had to detect when there is a half-time .... tbd

## "rules"

As mentioned above, 2 simple rules were our foundation. Based on that, we created simple instruction for us:

1. There are 3 states of ball touch events: `POSSESION` - when player has sustained possesion (close to ball for minimum 3 frames), `PASS` - when player is touching ball for just a moment and it passes to his teammate and `UNCONFIRMED` - when any of above requirements is not met.
2. Based on these events the first layer of frames labeling were done. Each team that were in `POSSESION` or `PASS` get label. For every `UNCONFIRMED`, the team that had ball in next event get label.
3. The second layer was just to simply check if ball is out of the pitch. These frames get label "D".
4. The third layer were combining `smart-tagging` events with ball touch events. Every time the `smart-tagging`'s `SET-PIECE` event took place, we labeled range between start frame and closes ball touch after start time as "D".

## "limitations"

- known limitation is our players detection - it is not perfect. It tends to mess up team assignement,
- there are some match breaks that were not properly recognized,
- missing ball is just primitevly fixed - but it only works for short ranges of ball miss,
- ball touches events detection is not perfect - it tends to miss some touches (especially headers)

## "possible fixes"

- the plan were to detect refree's whistle to recognize match breaks better - we could not manage to make it work good enough to use it in our system, 
- the thing we did not touch was broadcast data - maybe we could took some more data from there.

## Internal assets

1. tactical_atd - used as a main source of players/ball positions
2. tactical_homography - used in our detection algorithm
3. tactical_smart_tagging - used to get "SET_PIECE" events
4. mp4 video clip - used to detect missing players position using free model and our algorithm

## External assets

1. football-player-detection.pt - used in players detection, source: https://github.com/roboflow/sports, license: https://github.com/roboflow/sports/blob/main/LICENSE



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
