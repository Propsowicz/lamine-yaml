from math import acos, degrees, hypot, isfinite
from typing import Sequence
from models.player import Player
from models.frame import Frame
from models.ball import Ball
from utils.time_format import format_elapsed_time
from math import ceil, hypot
from typing import Sequence
from datetime import timedelta

Point = tuple[float, float]

from dataclasses import dataclass
from enum import Enum


class TouchType(str, Enum):
    POSSESSION = "possession"
    PASS = "pass"
    UNCONFIRMED = "unconfirmed"


@dataclass(frozen=True)
class TouchingPlayer:
    player_id: int
    team: str
    distance_m: float


@dataclass(frozen=True)
class BallPosition:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class PlayerTouchSummary:
    player_id: str
    team: str

    # Minimum distance observed during the event.
    min_distance_m: float

    # First and last frames where the player was within the radius.
    first_frame_id: int
    last_frame_id: int


@dataclass(frozen=True)
class BallTouchEvent:
    touch_type: TouchType

    # Event boundaries.
    start_frame_id: int
    start_timestamp: timedelta
    end_frame_id: int
    end_timestamp: timedelta

    # Relevant for PASS. None for POSSESSION.
    touch_frame_id: int | None
    touch_timestamp: timedelta | None

    # Ball coordinates at event boundaries.
    start_ball: BallPosition
    end_ball: BallPosition

    # Players involved, ordered by minimum distance.
    players: tuple[PlayerTouchSummary, ...]

    @property
    def start_time(self) -> str:
        """Start time as elapsed video time in HH:MM:SS format."""
        return format_elapsed_time(self.start_timestamp)

    @property
    def end_time(self) -> str:
        """End time as elapsed video time in HH:MM:SS format."""
        return format_elapsed_time(self.end_timestamp)

from math import ceil, hypot
from typing import Sequence

def _has_sustained_possession(
    start_index: int,
    end_index: int,
    min_possession_frames: int = 3,
) -> bool:
    return (
        end_index - start_index + 1
        >= min_possession_frames
    )

def _summarize_players(
    frames: Sequence[Frame],
    players_by_frame: Sequence[tuple[TouchingPlayer, ...]],
    start_index: int,
    end_index: int,
) -> tuple[PlayerTouchSummary, ...]:

    summaries: dict[str, dict] = {}

    for index in range(start_index, end_index + 1):
        for player in players_by_frame[index]:
            existing = summaries.get(player.player_id)

            if existing is None:
                summaries[player.player_id] = {
                    "player_id": player.player_id,
                    "team": player.team,
                    "min_distance_m": player.distance_m,
                    "first_frame_id": frames[index].id,
                    "last_frame_id": frames[index].id,
                }
                continue

            existing["min_distance_m"] = min(
                existing["min_distance_m"],
                player.distance_m,
            )

            existing["last_frame_id"] = frames[index].id

    return tuple(
        PlayerTouchSummary(**summary)
        for summary in sorted(
            summaries.values(),
            key=lambda item: item["min_distance_m"],
        )
    )

def _get_player_distance(
    frame: Frame,
    player_id: str,
    ball_position: Point | None,
) -> float | None:
    if ball_position is None:
        return None

    player = next(
        (p for p in frame.players if p.player_id == player_id),
        None,
    )

    if player is None:
        return None

    return hypot(
        player.x - ball_position[0],
        player.y - ball_position[1],
    )


def _find_pass_end(
    frames: Sequence[Frame],
    positions: Sequence[Point | None],
    start_index: int,
    player_id: str,
    radius_m: float,
    lookahead_frames: int = 8,
    min_distance_increase_m: float = 0.75,
    min_ball_displacement_m: float = 0.75,
) -> int | None:
    """
    Returns the frame index confirming a pass,
    or None if no pass is detected.
    """

    distances: list[float] = []

    end_limit = min(
        start_index + lookahead_frames,
        len(frames) - 1,
    )

    for index in range(start_index, end_limit + 1):
        distance = _get_player_distance(
            frames[index],
            player_id,
            positions[index],
        )

        if distance is None:
            break

        distances.append(distance)

        if len(distances) < 3:
            continue

        # Is the ball moving sufficiently far from its initial position?
        start_ball = positions[start_index]
        current_ball = positions[index]

        if start_ball is None or current_ball is None:
            continue

        ball_displacement = hypot(
            current_ball[0] - start_ball[0],
            current_ball[1] - start_ball[1],
        )

        if ball_displacement < min_ball_displacement_m:
            continue

        # Has the ball moved sufficiently far away from the player?
        if distances[-1] - distances[0] < min_distance_increase_m:
            continue

        # Check that the distance increases consistently.
        increments = [
            distances[i + 1] - distances[i]
            for i in range(len(distances) - 1)
        ]

        increasing_steps = sum(
            increment >= -0.05
            for increment in increments
        )

        required_steps = ceil(len(increments) * 0.7)

        if increasing_steps < required_steps:
            continue

        # Ball must have left the touch radius for at least
        # two consecutive observations (not necessarily consecutive).
        outside_radius = sum(
            distance > radius_m
            for distance in distances[1:]
        )

        if distances[-1] > radius_m and outside_radius >= 2:
            return index

    return None

def _get_ball_position(frame: Frame) -> Point | None:
    x = frame.ball.x
    y = frame.ball.y

    if x is None or y is None:
        return None

    if not isfinite(x) or not isfinite(y):
        return None

    return float(x), float(y)


def _get_players_near_ball(
    frame: Frame,
    ball_position: Point,
    radius_m: float,
) -> tuple[TouchingPlayer, ...]:
    ball_x, ball_y = ball_position

    touching_players = []

    for player in frame.players:
        if not isfinite(player.x) or not isfinite(player.y):
            continue

        distance = hypot(
            player.x - ball_x,
            player.y - ball_y,
        )

        if distance <= radius_m:
            touching_players.append(
                TouchingPlayer(
                    player_id=player.player_id,
                    team=player.team,
                    distance_m=distance,
                )
            )

    return tuple(
        sorted(touching_players, key=lambda player: player.distance_m)
    )


def _get_displacement(
    positions: Sequence[Point | None],
    start_index: int,
    end_index: int,
) -> tuple[float, float] | None:
    start = positions[start_index]
    end = positions[end_index]

    if start is None or end is None:
        return None

    return (
        end[0] - start[0],
        end[1] - start[1],
    )
def _is_ball_moving_away_from_player(
    frames: Sequence[Frame],
    positions: Sequence[Point | None],
    frame_index: int,
    touching_players: Sequence[TouchingPlayer],
    radius_m: float = 1.5,
    lookahead_frames: int = 3,
    min_distance_increase_m: float = 0.75,
    min_ball_displacement_m: float = 0.75,
) -> bool:
    """
    Detects whether the ball is moving away from the closest player.

    A pass is suspected when:
    - The ball travels a sufficient distance.
    - The distance between the ball and the player increases.
    - The player-ball distance increases consistently.
    - The ball leaves the player's touch radius.
    """

    if not touching_players:
        return False

    end_index = min(
        frame_index + lookahead_frames,
        len(frames) - 1,
    )

    if end_index - frame_index < 2:
        return False

    # Closest player is the most likely source of the ball movement.
    player_id = touching_players[0].player_id

    start_ball = positions[frame_index]
    end_ball = positions[end_index]

    if start_ball is None or end_ball is None:
        return False

    ball_displacement = hypot(
        end_ball[0] - start_ball[0],
        end_ball[1] - start_ball[1],
    )

    if ball_displacement < min_ball_displacement_m:
        return False

    distances: list[float] = []

    # Track the same player throughout the lookahead window.
    for index in range(frame_index, end_index + 1):
        ball_position = positions[index]

        if ball_position is None:
            return False

        player = next(
            (
                p for p in frames[index].players
                if p.player_id == player_id
            ),
            None,
        )

        if player is None:
            return False

        distance = hypot(
            player.x - ball_position[0],
            player.y - ball_position[1],
        )

        distances.append(distance)

    # Has the ball moved sufficiently far away from the player?
    distance_increase = distances[-1] - distances[0]

    if distance_increase < min_distance_increase_m:
        return False

    # Check whether distance increases consistently.
    increments = [
        distances[i + 1] - distances[i]
        for i in range(len(distances) - 1)
    ]

    increasing_steps = sum(
        increment >= -0.05
        for increment in increments
    )

    required_increasing_steps = ceil(len(increments) * 0.7)

    if increasing_steps < required_increasing_steps:
        return False

    # The ball should leave the touch radius.
    if distances[-1] <= radius_m:
        return False

    outside_radius_frames = sum(
        distance > radius_m
        for distance in distances[1:]
    )

    return outside_radius_frames >= 2

def _has_direction_change(
    positions: Sequence[Point | None],
    frame_index: int,
    window: int,
    direction_threshold_degrees: float,
    min_motion_displacement_m: float,
    min_pass_displacement_m: float,
) -> bool:
    """
    Compares the ball's movement before and after a potential touch.

    Also detects a stationary ball becoming a moving ball,
    which can indicate a pass or kick.
    """

    if frame_index - window < 0:
        return False

    if frame_index + window >= len(positions):
        return False

    # Movement before the potential touch.
    before = _get_displacement(
        positions,
        frame_index - window,
        frame_index - 1,
    )

    # Movement after the potential touch.
    after = _get_displacement(
        positions,
        frame_index + 1,
        frame_index + window,
    )

    if before is None or after is None:
        return False

    before_distance = hypot(*before)
    after_distance = hypot(*after)

    # Ball was stationary and starts moving significantly.
    if before_distance < min_motion_displacement_m:
        return after_distance >= min_pass_displacement_m

    # Not enough movement to determine a reliable direction.
    if after_distance < min_motion_displacement_m:
        return False

    dot_product = (
        before[0] * after[0] +
        before[1] * after[1]
    )

    cosine = dot_product / (before_distance * after_distance)

    # Floating-point protection.
    cosine = max(-1.0, min(1.0, cosine))

    angle = degrees(acos(cosine))

    return angle >= direction_threshold_degrees

def _find_closest_touch_frame(
    players_by_frame: Sequence[tuple[TouchingPlayer, ...]],
    start_index: int,
    end_index: int,
) -> tuple[int, TouchingPlayer]:
    """
    Finds the frame and player with the smallest ball-player
    distance within a consecutive proximity sequence.
    """

    candidates = [
        (player.distance_m, index, player)
        for index in range(start_index, end_index + 1)
        for player in players_by_frame[index]
    ]

    _, closest_frame_index, closest_player = min(
        candidates,
        key=lambda candidate: candidate[0],
    )

    return closest_frame_index, closest_player

def _ball_separates_from_players(
    positions: Sequence[Point | None],
    players_by_frame: Sequence[tuple[TouchingPlayer, ...]],
    frame_index: int,
    lookahead_frames: int,
    required_unaccompanied_frames: int,
) -> bool:
    """
    Returns True when the ball remains outside every player's
    touch radius for a specified number of consecutive frames.
    """

    consecutive_frames = 0

    end_index = min(
        frame_index + lookahead_frames + 1,
        len(positions),
    )

    for index in range(frame_index + 1, end_index):
        # Missing ball coordinates must not count as separation.
        if positions[index] is None:
            consecutive_frames = 0
            continue

        if not players_by_frame[index]:
            consecutive_frames += 1

            if consecutive_frames >= required_unaccompanied_frames:
                return True
        else:
            consecutive_frames = 0

    return False


def detect_ball_touches(
    frames: Sequence[Frame],
    radius_m: float = 1.5,
    lookahead_frames: int = 8,
) -> list[BallTouchEvent]:

    if radius_m <= 0:
        raise ValueError("radius_m must be positive")

    positions = [
        _get_ball_position(frame)
        for frame in frames
    ]

    players_by_frame = [
        (
            _get_players_near_ball(frame, position, radius_m)
            if position is not None
            else ()
        )
        for frame, position in zip(frames, positions)
    ]

    events: list[BallTouchEvent] = []
    index = 0

    while index < len(frames):

        if not players_by_frame[index]:
            index += 1
            continue

        # 1. Find the complete proximity sequence.
        sequence_start = index
        sequence_end = index

        while (
            sequence_end + 1 < len(frames)
            and players_by_frame[sequence_end + 1]
        ):
            sequence_end += 1

        # 2. Find the closest frame and player.
        touch_index, closest_player = _find_closest_touch_frame(
            players_by_frame,
            sequence_start,
            sequence_end,
        )

        # 3. Check whether the ball moves away from that player.
        pass_end_index = _find_pass_end(
            frames=frames,
            positions=positions,
            start_index=touch_index,
            player_id=closest_player.player_id,
            radius_m=radius_m,
            lookahead_frames=lookahead_frames,
        )

        # 4. Choose event boundaries based on event type.
        if pass_end_index is not None:
            touch_type = TouchType.PASS

            event_start = touch_index
            event_end = pass_end_index

            # Players present at the actual touch moment.
            event_players = _summarize_players(
                frames,
                players_by_frame,
                touch_index,
                touch_index,
            )

            representative_frame = frames[touch_index]

        elif _has_sustained_possession(
            sequence_start,
            sequence_end,
            min_possession_frames=3,
        ):
            touch_type = TouchType.POSSESSION

            # Possession covers the entire proximity sequence.
            event_start = sequence_start
            event_end = sequence_end

            # All players involved throughout possession.
            event_players = _summarize_players(
                frames,
                players_by_frame,
                sequence_start,
                sequence_end,
            )

            representative_frame = None
        else:
            touch_type = TouchType.UNCONFIRMED
            event_start = sequence_start
            event_end = sequence_end
            event_players = _summarize_players(
                            frames,
                            players_by_frame,
                            sequence_start,
                            sequence_end,
                        )
            representative_frame = None


        # 5. Create the event.
        start_ball = positions[event_start]
        end_ball = positions[event_end]

        if start_ball is not None and end_ball is not None:
            events.append(
                BallTouchEvent(
                    touch_type=touch_type,

                    start_frame_id=frames[event_start].id,
                    start_timestamp=frames[event_start].timestamp,

                    end_frame_id=frames[event_end].id,
                    end_timestamp=frames[event_end].timestamp,

                    touch_frame_id=(
                        representative_frame.id
                        if representative_frame else None
                    ),
                    touch_timestamp=(
                        representative_frame.timestamp
                        if representative_frame else None
                    ),

                    start_ball=BallPosition(
                        *start_ball,
                        frames[event_start].ball.z,
                    ),
                    end_ball=BallPosition(
                        *end_ball,
                        frames[event_end].ball.z,
                    ),

                    players=event_players,
                )
            )

        # 6. Skip all frames belonging to this event.
        index = max(sequence_end, event_end) + 1

    return events
