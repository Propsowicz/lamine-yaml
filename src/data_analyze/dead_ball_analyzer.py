"""Estimate dead-ball intervals from set-piece context and corroborating motion."""

from collections.abc import Sequence
from math import hypot, isfinite
from statistics import mean, median

from data_prepare.games import GAME, GAMES
from data_prepare.smart_tagging import SmartTaggingEvent
from data_prepare.touch_ball_tracker import BallTouchEvent
from models.frame import Frame
from whistle.full_video_whistle import WhistleEvent


def _video_frame_id(frame_id: int, game: str) -> int | None:
    """Map a tracking counter to the tactical video frame timeline."""
    game_data = GAMES[game]
    for half, (first_tracking_id, last_tracking_id) in game_data["halves"].items():
        if first_tracking_id <= frame_id <= last_tracking_id:
            return frame_id - game_data["offsets"][half]
    if 0 <= frame_id < game_data["n_video_frames"]:
        # Also accept frames already expressed on the video timeline.
        return frame_id
    return None


def _frame_motion(frame: Frame) -> tuple[float | None, tuple[float, float] | None]:
    speeds = [
        float(player.speed)
        for player in frame.players
        if player.speed is not None and isfinite(float(player.speed)) and float(player.speed) >= 0
    ]
    average_speed = mean(speeds) if speeds else None
    x, y = frame.ball.x, frame.ball.y
    ball_position = (
        (float(x), float(y))
        if x is not None and y is not None and isfinite(float(x)) and isfinite(float(y))
        else None
    )
    return average_speed, ball_position


def _window_metrics(
    motion: dict[int, tuple[float | None, tuple[float, float] | None]],
    start: int,
    end: int,
) -> tuple[float | None, float | None, int, int]:
    rows = [motion[frame_id] for frame_id in range(start, end + 1) if frame_id in motion]
    if not rows:
        return None, None, 0, 0
    speeds = [speed for speed, _ in rows if speed is not None]
    positions = [position for _, position in rows if position is not None]
    average_speed = mean(speeds) if speeds else None
    spread = None
    if len(positions) >= 2:
        center_x = median(position[0] for position in positions)
        center_y = median(position[1] for position in positions)
        spread = max(hypot(x - center_x, y - center_y) for x, y in positions)
    return average_speed, spread, len(speeds), len(positions)


def _candidate_support(
    whistle: WhistleEvent,
    motion: dict[int, tuple[float | None, tuple[float, float] | None]],
    slowdown_window_frames: int,
    settle_frames: int,
    slowdown_ratio: float,
    minimum_speed_drop_mps: float,
    ball_stationary_radius_m: float,
) -> tuple[float, float] | None:
    candidate_frame = whistle.peak_frame_id
    before_start = candidate_frame - slowdown_window_frames
    before_end = candidate_frame - 1
    after_start = candidate_frame + settle_frames
    after_end = after_start + slowdown_window_frames - 1
    before_speed, _, before_speed_count, _ = _window_metrics(motion, before_start, before_end)
    after_speed, ball_spread, after_speed_count, after_ball_count = _window_metrics(
        motion, after_start, after_end)
    required = max(2, int(slowdown_window_frames * 0.65))
    if (before_speed_count < required or after_speed_count < required
            or after_ball_count < required or before_speed is None
            or after_speed is None or ball_spread is None):
        return None

    speed_drop = before_speed - after_speed
    drop_ratio = speed_drop / max(before_speed, 0.1)
    if speed_drop < minimum_speed_drop_mps or drop_ratio < slowdown_ratio:
        return None
    if ball_spread > ball_stationary_radius_m:
        return None

    # Corroboration decides the ranking; model confidence only breaks a tie.
    support = 0.7 * min(drop_ratio, 1.0) + 0.3 * max(
        0.0, 1.0 - ball_spread / ball_stationary_radius_m)
    return support, speed_drop


def _find_resumption(
    motion: dict[int, tuple[float | None, tuple[float, float] | None]],
    start: int,
    end: int,
    window_frames: int,
    resume_player_speed_mps: float,
    resume_ball_displacement_m: float,
) -> int | None:
    """Return the first sustained movement window, if either tracked signal resumes."""
    for window_start in range(start, end - window_frames + 2):
        window_end = window_start + window_frames - 1
        average_speed, ball_spread, speed_count, ball_count = _window_metrics(
            motion, window_start, window_end)
        required = max(2, int(window_frames * 0.65))
        if ((speed_count >= required and average_speed is not None
             and average_speed >= resume_player_speed_mps)
                or (ball_count >= required and ball_spread is not None
                    and ball_spread * 2 >= resume_ball_displacement_m)):
            return window_start
    return None


def _contiguous_tracking_end(motion: dict[int, object], frame_id: int, max_gap: int) -> int:
    """Do not let a search jump across halftime or a long tracking-data gap."""
    frame_ids = sorted(motion)
    segment_start = previous = frame_ids[0]
    for current in frame_ids[1:]:
        if current - previous > max_gap:
            if segment_start <= frame_id <= previous:
                return previous
            segment_start = current
        previous = current
    if segment_start <= frame_id <= previous:
        return previous
    return frame_id


def detect_dead_ball_ranges(
    frames: Sequence[Frame],
    st_events: Sequence[SmartTaggingEvent],
    touches: Sequence[BallTouchEvent],
    whistle_events: Sequence[WhistleEvent] = (),
    *,
    game: str = GAME,
    fps: float | None = None,
    whistle_search_before_seconds: float = 2.0,
    whistle_search_after_seconds: float = 0.5,
    slowdown_window_seconds: float = 1.0,
    whistle_settle_seconds: float = 0.4,
    player_slowdown_ratio: float = 0.3,
    minimum_player_speed_drop_mps: float = 0.25,
    ball_stationary_radius_m: float = 0.45,
    resume_window_seconds: float = 0.75,
    resume_player_speed_mps: float = 1.0,
    resume_ball_displacement_m: float = 0.8,
) -> list[tuple[int, int]]:
    """Return inclusive dead-ball ranges on the tactical video frame timeline.

    Set-piece annotations provide broad context. A whistle is accepted only when
    the following motion window shows a meaningful drop in mean player speed and
    the ball remains within ``ball_stationary_radius_m`` of its median location.
    The best corroborated whistle is selected (model score only breaks ties).
    A range ends at the first ``resume_window_seconds`` window with sustained
    player speed or ball movement. Searches stop at the next set-piece and across
    long tracking gaps. If evidence is unavailable, the legacy start-to-next-touch
    fallback is retained within the annotation's bounds.

    Player speed is read from tracking records (in m/s); ball positions are in
    metres. Defaults assume the configured game's FPS. The thresholds are
    intentionally keyword-configurable because tracking noise varies by source.
    """
    if not frames:
        return []
    fps = GAMES[game]["fps"] if fps is None else fps
    if fps <= 0:
        raise ValueError("fps must be positive")
    if any(value < 0 for value in (
        whistle_search_before_seconds, whistle_search_after_seconds,
        slowdown_window_seconds, whistle_settle_seconds,
        minimum_player_speed_drop_mps, ball_stationary_radius_m,
        resume_window_seconds, resume_player_speed_mps,
        resume_ball_displacement_m,
    )):
        raise ValueError("Motion thresholds and windows must be nonnegative")
    if (ball_stationary_radius_m <= 0 or not 0 <= player_slowdown_ratio <= 1):
        raise ValueError("ball_stationary_radius_m must be positive and player_slowdown_ratio <= 1")

    motion: dict[int, tuple[float | None, tuple[float, float] | None]] = {}
    for frame in frames:
        video_id = _video_frame_id(int(frame.id), game)
        if video_id is not None:
            motion[video_id] = _frame_motion(frame)
    if not motion:
        return []
    analyzed_start = min(motion)
    analyzed_end = max(motion)
    slowdown_frames = max(2, round(slowdown_window_seconds * fps))
    settle_frames = max(0, round(whistle_settle_seconds * fps))
    resume_frames = max(2, round(resume_window_seconds * fps))

    touch_video_frames = sorted(
        mapped for touch in touches
        if (mapped := _video_frame_id(
            touch.touch_frame_id if touch.touch_frame_id is not None else touch.start_frame_id,
            game,
        )) is not None
    )
    whistles_by_frame = sorted(whistle_events, key=lambda event: event.peak_frame_id)
    ordered_sets = sorted(st_events, key=lambda event: event.start_frame)
    ranges: list[tuple[int, int]] = []

    for set_index, event in enumerate(ordered_sets):
        context_start = max(event.start_frame, analyzed_start)
        context_end = min(event.end_frame, analyzed_end)
        if context_start > context_end:
            continue
        next_context_start = (
            ordered_sets[set_index + 1].start_frame
            if set_index + 1 < len(ordered_sets)
            else analyzed_end + 1
        )
        context_end = min(context_end, next_context_start - 1)
        if context_start > context_end:
            continue
        next_touch = next((frame_id for frame_id in touch_video_frames
                           if frame_id > event.start_frame), None)

        candidate_start = max(
            analyzed_start,
            event.start_frame - round(whistle_search_before_seconds * fps),
        )
        candidate_end = min(
            analyzed_end,
            event.end_frame + round(whistle_search_after_seconds * fps),
        )
        supported = []
        for whistle in whistles_by_frame:
            if candidate_start <= whistle.peak_frame_id <= candidate_end:
                support = _candidate_support(
                    whistle, motion, slowdown_frames, settle_frames,
                    player_slowdown_ratio, minimum_player_speed_drop_mps,
                    ball_stationary_radius_m,
                )
                if support is not None:
                    supported.append((support[0], support[1], whistle.peak_score, whistle))

        if supported:
            _, _, _, selected = max(supported, key=lambda item: (item[0], item[1], item[2]))
            dead_start = max(analyzed_start, selected.peak_frame_id)
            # Do not mistake the immediate post-whistle settling for a restart.
            resume_search_start = max(
                dead_start + slowdown_frames + settle_frames,
                selected.end_frame_id + 1,
            )
            search_end = min(analyzed_end, next_context_start - 1)
            search_end = min(
                search_end,
                _contiguous_tracking_end(motion, dead_start, max(2, round(fps))),
            )
            resume = _find_resumption(
                motion, resume_search_start, search_end, resume_frames,
                resume_player_speed_mps, resume_ball_displacement_m,
            )
            # The annotation bounds an evidence-backed estimate when resumption
            # is not observed in the available tracking frames.
            dead_end = (resume - 1 if resume is not None else
                        min(context_end, search_end))
        else:
            # Keep the previous annotation/touch fallback when motion or whistle
            # evidence is missing or inconclusive.
            dead_start = context_start
            dead_end = min(
                next_touch - 1 if next_touch is not None else analyzed_end,
                context_end,
            )

        dead_end = min(dead_end, analyzed_end)
        if dead_start <= dead_end:
            ranges.append((dead_start, dead_end))

    # Overlapping set-piece annotations should not duplicate conflicting ranges.
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
