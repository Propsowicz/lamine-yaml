from collections.abc import Sequence

from data_prepare.smart_tagging import SmartTaggingEvent
from data_prepare.touch_ball_tracker import BallTouchEvent
from models.frame import Frame

def detect_dead_ball_ranges(
    frames: Sequence[Frame],
    st_events: Sequence[SmartTaggingEvent],
    touches: Sequence[BallTouchEvent]
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
    for event in st_events:
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

    return sorted(dead_ball_ranges)

    
