from collections.abc import Sequence

from data_prepare.smart_tagging import SmartTaggingEvent
from data_prepare.touch_ball_tracker import BallTouchEvent
from models.frame import Frame
from whistle_analysis.whistle_events import WhistleEvent


def detect_dead_ball_ranges(
    frames: Sequence[Frame],
    st_events: Sequence[SmartTaggingEvent],
    touches: Sequence[BallTouchEvent],
    whistle_events: Sequence[WhistleEvent],
) -> list[tuple[int, int]]:
    """Return inclusive dead-ball frame ranges for set-piece events."""
    if not frames:
        return []


    analyzed_start_frame = min(frame.id for frame in frames)
    analyzed_end_frame = max(frame.id for frame in frames)
    touch_frames = [
        touch.touch_frame_id
        if touch.touch_frame_id is not None
        else touch.start_frame_id
        for touch in touches
    ]

    dead_ball_ranges: list[tuple[int, int]] = []
    dead_ball_range_before_first_half = find_first_half_dead_ball_range(st_events)
    dead_ball_range_before_second_half = find_second_half_dead_ball_range(st_events, whistle_events)
    dead_ball_range_after_game_finished = find_dead_ball_range_after_game_finished(
        whistle_events, analyzed_end_frame
    )
    for event in st_events:
        is_kickoff = (
            event.code == "SET PIECES"
            and event.tags.get("Type") == "Kick Off (Start)"
        )

        next_touch_frame = min(
            (
                frame_id
                for frame_id in touch_frames
                if frame_id > event.start_frame
            ),
            default=None,
        )
        dead_start = max(event.start_frame, analyzed_start_frame)
        dead_end = min(
            next_touch_frame - 1
            if next_touch_frame is not None
            else analyzed_end_frame,
            analyzed_end_frame,
        )
        if dead_start <= dead_end:
            dead_ball_ranges.append((dead_start, dead_end))

    for dead_ball_range in (
        dead_ball_range_before_first_half,
        dead_ball_range_before_second_half,
        dead_ball_range_after_game_finished,
    ):
        if dead_ball_range is None:
            continue

        dead_start = max(dead_ball_range[0], analyzed_start_frame)
        dead_end = min(dead_ball_range[1], analyzed_end_frame)
        if dead_start <= dead_end:
            dead_ball_ranges.append((dead_start, dead_end))

    return sorted(dead_ball_ranges)

    
def find_first_half_dead_ball_range(st_events: Sequence[SmartTaggingEvent]):
    kickoff = next((st_ev for st_ev in st_events if st_ev.tags.get("Type") == "Kick Off (Start)" and st_ev.tags.get("Half") == '1st Half'), None)
    
    if kickoff is None:
        return None
    return (0, kickoff.start_frame)

def find_second_half_dead_ball_range(st_events: Sequence[SmartTaggingEvent], whistle_events: Sequence[WhistleEvent]):
    kickoff = next((st_ev for st_ev in st_events if st_ev.tags.get("Type") == "Kick Off (Start)" and st_ev.tags.get("Half") == '2nd Half'), None)
    
    if kickoff is None:
        return None
    
    last_whistle = max(
        (
            whistle
            for whistle in whistle_events
            if whistle.peak_frame_id < kickoff.start_frame
        ),
        key=lambda whistle: whistle.peak_frame_id,
        default=None,
    )
    return (last_whistle.peak_frame_id if last_whistle is not None else 0, kickoff.start_frame)

def find_dead_ball_range_after_game_finished(
    whistle_events: Sequence[WhistleEvent],
    lastFrame: int
):
    last_whistle = max(
        whistle_events,
        key=lambda whistle: whistle.peak_frame_id,
        default=None,
    )
    if last_whistle is None:
        return None
    return (last_whistle.peak_frame_id, lastFrame)
