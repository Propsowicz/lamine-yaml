from dataclasses import dataclass
from typing import Optional
from models.ball import Ball
from models.frame import Frame

@dataclass
class BallOutEvent:
    start_frame_id: int
    end_frame_id: int
    start_timestamp: float
    end_timestamp: float
    start_frame_index: int
    end_frame_index: int


def detect_ball_out_events(
    frames: list[Frame],
    pitch_length: float,
    pitch_width: float,
) -> list[BallOutEvent]:
    events: list[BallOutEvent] = []

    start_frame: Optional[Frame] = None
    start_index: Optional[int] = None
    end_frame: Optional[Frame] = None
    end_index: Optional[int] = None

    for index, frame in enumerate(frames):
        ball = frame.ball

        is_out = (
            ball.x is not None
            and ball.y is not None
            and (
                ball.x < 0
                or ball.x > pitch_length
                or ball.y < 0
                or ball.y > pitch_width
            )
        )

        if is_out:
            if start_frame is None:
                start_frame = frame
                start_index = index

            end_frame = frame
            end_index = index

        elif start_frame is not None:
            events.append(
                BallOutEvent(
                    start_frame_id=start_frame.id,
                    end_frame_id=end_frame.id,
                    start_timestamp=start_frame.timestamp,
                    end_timestamp=end_frame.timestamp,
                    start_frame_index=start_index,
                    end_frame_index=end_index,
                )
            )

            start_frame = None
            start_index = None
            end_frame = None
            end_index = None

    # Handle an event that continues until the last frame.
    if start_frame is not None:
        events.append(
            BallOutEvent(
                start_frame_id=start_frame.id,
                end_frame_id=end_frame.id,
                start_timestamp=start_frame.timestamp,
                end_timestamp=end_frame.timestamp,
                start_frame_index=start_index,
                end_frame_index=end_index,
            )
        )

    return events