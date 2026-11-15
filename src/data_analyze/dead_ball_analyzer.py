"""Estimate dead-ball intervals from whistle, motion, and player visibility cues."""

from collections.abc import Sequence
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import timedelta
from math import ceil, hypot, isfinite
from statistics import mean, median
from typing import Literal

from data_prepare.games import GAME, GAMES
from data_prepare.smart_tagging import SmartTaggingEvent
from data_prepare.touch_ball_tracker import BallTouchEvent
from models.frame import Frame
from whistle.full_video_whistle import WhistleEvent
from utils.constants import PITCH_LENGTH, PITCH_WIDTH
from utils.time_format import format_elapsed_time

Point = tuple[float, float]


@dataclass(frozen=True)
class Observation:
    average_speed: float | None
    ball: Point | None
    player_positions: dict[str, Point]
    identified_player_positions: dict[str, Point]
    visible_count: int
    identified_count: int
    unassigned_count: int
    near_ball_count: int | None
    approaching_count: int | None
    leaving_count: int | None


@dataclass(frozen=True)
class DeadBallRange:
    """Inclusive video-frame range with the method that produced it."""

    start_frame_id: int
    end_frame_id: int
    source: Literal["evidence", "fallback"]
    fps: float = field(default=GAMES[GAME]["fps"], repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.start_frame_id > self.end_frame_id:
            raise ValueError("start_frame_id must be <= end_frame_id")
        if self.source not in ("evidence", "fallback"):
            raise ValueError("source must be 'evidence' or 'fallback'")
        if not isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be a finite positive number")

    @property
    def start_time(self) -> str:
        """Start time as elapsed video time in HH:MM:SS format."""
        return format_elapsed_time(timedelta(seconds=self.start_frame_id / self.fps))

    @property
    def end_time(self) -> str:
        """End time as elapsed video time in HH:MM:SS format."""
        return format_elapsed_time(timedelta(seconds=self.end_frame_id / self.fps))


def _video_frame_id(frame_id: int, game: str) -> int | None:
    """Map the tracking counter onto the tactical video's frame timeline."""
    game_data = GAMES[game]
    for half, (first_tracking_id, last_tracking_id) in game_data["halves"].items():
        if first_tracking_id <= frame_id <= last_tracking_id:
            return frame_id - game_data["offsets"][half]
    if 0 <= frame_id < game_data["n_video_frames"]:
        # Also accept frames that already use the video timeline.
        return frame_id
    return None


def _tracking_segments(frame_ids: Sequence[int], max_gap: int) -> list[tuple[int, int]]:
    """Split the tracked timeline at halftime and long gaps in the input data."""
    if not frame_ids:
        return []
    segments = []
    start = previous = frame_ids[0]
    for frame_id in frame_ids[1:]:
        if frame_id - previous > max_gap:
            segments.append((start, previous))
            start = frame_id
        previous = frame_id
    segments.append((start, previous))
    return segments


def _ball_position(frame: Frame) -> Point | None:
    x, y = frame.ball.x, frame.ball.y
    if x is None or y is None:
        return None
    x, y = float(x), float(y)
    return (x, y) if isfinite(x) and isfinite(y) else None


def _build_observations(
    frames: Sequence[Frame], game: str, approach_lookback_frames: int,
    exit_lookback_frames: int, near_ball_radius_m: float, pitch_margin_m: float,
    exit_outside_margin_m: float, exit_edge_radius_m: float,
    minimum_exit_progress_m: float,
) -> dict[int, Observation]:
    raw: dict[int, tuple[
        Frame, Point | None, dict[str, Point], dict[str, Point], dict[str, Point],
        int, int, int, float | None,
    ]] = {}
    known_teams = {
        str(team).strip().casefold()
        for frame in frames
        for team in frame.players_by_team
    }
    unknown_team_names = {"", "tbd", "unknown", "none", "unassigned", "unidentified"}

    for frame in frames:
        video_id = _video_frame_id(int(frame.id), game)
        if video_id is None:
            continue
        positions: dict[str, Point] = {}
        identified_positions: dict[str, Point] = {}
        tracked_positions: dict[str, Point] = {}
        speeds = []
        unassigned_count = 0
        for player in frame.players:
            x, y = player.x, player.y
            if x is None or y is None:
                continue
            x, y = float(x), float(y)
            if not isfinite(x) or not isfinite(y):
                continue
            player_id = str(player.player_id)
            team = str(player.team or "").strip().casefold()
            assigned = team not in unknown_team_names and (not known_teams or team in known_teams)
            is_detection = getattr(player, "from_detection", False)
            is_virtual = getattr(player, "validator_anchor_id", None) is not None
            # Retain real tracked positions just beyond the lines to detect a
            # player moving off the pitch; large homography outliers are ignored.
            if (assigned and not is_detection and not is_virtual
                    and -exit_outside_margin_m <= x <= PITCH_LENGTH + exit_outside_margin_m
                    and -exit_outside_margin_m <= y <= PITCH_WIDTH + exit_outside_margin_m):
                tracked_positions[player_id] = (x, y)
            # Homography/detection outliers can land well beyond the pitch.
            # They should not inflate visible-player or ball-proximity cues.
            if not (-pitch_margin_m <= x <= PITCH_LENGTH + pitch_margin_m
                    and -pitch_margin_m <= y <= PITCH_WIDTH + pitch_margin_m):
                continue
            positions[player_id] = (x, y)
            if assigned and not is_detection and not is_virtual:
                identified_positions[player_id] = (x, y)
            if team in unknown_team_names or (known_teams and team not in known_teams):
                unassigned_count += 1
            speed = player.speed
            if (not is_detection
                    and speed is not None and isfinite(float(speed)) and float(speed) >= 0):
                speeds.append(float(speed))
        ball = _ball_position(frame)
        raw[video_id] = (
            frame, ball, positions, tracked_positions, identified_positions, len(positions),
            len(identified_positions), unassigned_count,
            mean(speeds) if speeds else None,
        )

    observations: dict[int, Observation] = {}
    for video_id, (frame, ball, positions, tracked_positions, identified_positions,
                   visible, identified, unassigned, speed) in raw.items():
        if ball is None:
            near_count = None
        else:
            near_count = sum(
                hypot(x - ball[0], y - ball[1]) <= near_ball_radius_m
                for x, y in identified_positions.values()
            )

        previous = raw.get(video_id - approach_lookback_frames)
        approaching_count = None
        if previous is not None and ball is not None:
            old_positions = previous[2]
            changes = []
            for player_id, current in positions.items():
                old = old_positions.get(player_id)
                if old is None:
                    continue
                old_distance = hypot(old[0] - ball[0], old[1] - ball[1])
                current_distance = hypot(current[0] - ball[0], current[1] - ball[1])
                changes.append(old_distance - current_distance)
            if changes:
                approaching_count = sum(change >= 0.25 for change in changes)

        exit_previous = raw.get(video_id - exit_lookback_frames)
        leaving_count = None
        if exit_previous is not None:
            old_tracked_positions = exit_previous[4]
            exit_progress = []
            for player_id, current in tracked_positions.items():
                old = old_tracked_positions.get(player_id)
                if old is None:
                    continue
                old_edge = min(old[0], PITCH_LENGTH - old[0], old[1], PITCH_WIDTH - old[1])
                current_edge = min(
                    current[0], PITCH_LENGTH - current[0],
                    current[1], PITCH_WIDTH - current[1],
                )
                exit_progress.append((old_edge, current_edge))
            if exit_progress:
                leaving_count = sum(
                    old_edge - current_edge >= minimum_exit_progress_m
                    and current_edge <= exit_edge_radius_m
                    for old_edge, current_edge in exit_progress
                )

        observations[video_id] = Observation(
            average_speed=speed,
            ball=ball,
            player_positions=positions,
            identified_player_positions=identified_positions,
            visible_count=visible,
            identified_count=identified,
            unassigned_count=unassigned,
            near_ball_count=near_count,
            approaching_count=approaching_count,
            leaving_count=leaving_count,
        )
    return observations


def _window_values(
    observations: dict[int, Observation], start: int, end: int
) -> list[Observation]:
    return [observations[frame_id] for frame_id in range(start, end + 1)
            if frame_id in observations]


def _ball_spread(values: Sequence[Observation]) -> tuple[float | None, int]:
    positions = [item.ball for item in values if item.ball is not None]
    if len(positions) < 2:
        return None, len(positions)
    center_x = median(point[0] for point in positions)
    center_y = median(point[1] for point in positions)
    return max(hypot(x - center_x, y - center_y) for x, y in positions), len(positions)


def _window_ball_speed(
    observations: dict[int, Observation], start: int, end: int, fps: float,
) -> float | None:
    positions = [
        (frame_id, observations[frame_id].ball)
        for frame_id in range(start, end + 1)
        if frame_id in observations and observations[frame_id].ball is not None
    ]
    if len(positions) < 2:
        return None
    distance = sum(
        hypot(current[1][0] - previous[1][0], current[1][1] - previous[1][1])
        for previous, current in zip(positions, positions[1:])
    )
    duration = max(1 / fps, (positions[-1][0] - positions[0][0]) / fps)
    return distance / duration


def _window_mean(values: Sequence[Observation], attribute: str) -> float | None:
    found = [getattr(item, attribute) for item in values]
    valid = [value for value in found if value is not None]
    return mean(valid) if valid else None


def _window_cue_rate(values: Sequence[Observation], attribute: str, predicate) -> float | None:
    found = [getattr(item, attribute) for item in values]
    valid = [value for value in found if value is not None]
    return sum(predicate(value) for value in valid) / len(valid) if valid else None


def _onset_evidence(
    frame_id: int,
    observations: dict[int, Observation],
    segment_start: int,
    window_frames: int,
    visible_baseline: float,
    unassigned_baseline: float,
    whistle_frames: Sequence[int],
    set_piece_ranges: Sequence[tuple[int, int]],
    set_piece_starts: Sequence[int],
    *,
    slowdown_ratio: float,
    minimum_speed_drop_mps: float,
    ball_stationary_radius_m: float,
    unassigned_player_minimum: int,
    fewer_visible_margin: int,
    leaving_players_minimum: int,
    no_near_ball_fraction: float,
    no_approach_fraction: float,
    whistle_followup_frames: int,
    slowdown_reference_frame: int | None = None,
) -> tuple[float, bool, dict[str, float | bool]]:
    reference_frame = frame_id if slowdown_reference_frame is None else slowdown_reference_frame
    pre = _window_values(
        observations, max(segment_start, reference_frame - window_frames), reference_frame - 1)
    post = _window_values(observations, frame_id, frame_id + window_frames - 1)
    required_samples = max(2, int(window_frames * 0.65))
    pre_speed = _window_mean(pre, "average_speed")
    post_speed = _window_mean(post, "average_speed")
    pre_speed_count = sum(item.average_speed is not None for item in pre)
    post_speed_count = sum(item.average_speed is not None for item in post)
    slowdown = False
    slowdown_strength = 0.0
    if (pre_speed_count >= required_samples and post_speed_count >= required_samples
            and pre_speed is not None and post_speed is not None):
        drop = pre_speed - post_speed
        slowdown_strength = min(max(0.0, drop / max(pre_speed, 0.1)), 1.0)
        slowdown = (drop >= minimum_speed_drop_mps
                    and drop / max(pre_speed, 0.1) >= slowdown_ratio)

    spread, ball_samples = _ball_spread(post)
    ball_stable = (
        spread is not None
        and ball_samples >= max(2, int(window_frames * 0.65))
        and spread <= ball_stationary_radius_m
    )
    ball_stability_strength = (
        max(0.0, 1.0 - spread / ball_stationary_radius_m)
        if spread is not None and ball_samples >= required_samples
        and spread <= ball_stationary_radius_m else 0.0
    )
    near_rate = _window_cue_rate(post, "near_ball_count", lambda count: count == 0)
    near_samples = sum(item.near_ball_count is not None for item in post)
    no_near_ball = (near_samples >= required_samples and near_rate is not None
                    and near_rate >= no_near_ball_fraction)
    approach_rate = _window_cue_rate(post, "approaching_count", lambda count: count == 0)
    approach_samples = sum(item.approaching_count is not None for item in post)
    no_approach = (approach_samples >= required_samples and approach_rate is not None
                   and approach_rate >= no_approach_fraction)

    visible_average = _window_mean(post, "visible_count")
    fewer_visible = (
        visible_average is not None
        and visible_average <= visible_baseline - fewer_visible_margin
    )
    unassigned_average = _window_mean(post, "unassigned_count")
    unassigned_anomaly = (
        unassigned_average is not None
        and unassigned_average >= max(unassigned_player_minimum, unassigned_baseline + 1)
    )
    leaving_rate = _window_cue_rate(
        post, "leaving_count", lambda count: count >= leaving_players_minimum)
    leaving_samples = sum(item.leaving_count is not None for item in post)
    leaving_pitch = (leaving_samples >= required_samples
                     and leaving_rate is not None and leaving_rate >= 0.5)

    whistle_index = bisect_right(whistle_frames, frame_id) - 1
    whistle = (whistle_index >= 0
               and frame_id - whistle_frames[whistle_index] <= whistle_followup_frames)
    set_piece_index = bisect_right(set_piece_starts, frame_id) - 1
    in_set_piece = (set_piece_index >= 0
                    and frame_id <= set_piece_ranges[set_piece_index][1])

    features: dict[str, float | bool] = {
        "whistle": whistle,
        "slowdown": slowdown,
        "slowdown_strength": slowdown_strength,
        "ball_stable": ball_stable,
        "ball_stability_strength": ball_stability_strength,
        "no_near_ball": no_near_ball,
        "no_approach": no_approach,
        "fewer_visible": fewer_visible,
        "unassigned_anomaly": unassigned_anomaly,
        "leaving_pitch": leaving_pitch,
        "in_set_piece": in_set_piece,
    }
    score = _weighted_score(features)
    threshold = 0.55 if in_set_piece else 0.70
    accepted = score >= threshold and ball_stable and (
        slowdown or unassigned_anomaly or no_near_ball or no_approach
    ) or (whistle and leaving_pitch)
    return score, accepted, features


def _weighted_score(features: dict[str, float | bool]) -> float:
    return (
        0.28 * bool(features.get("whistle"))
        + 0.25 * bool(features.get("slowdown"))
        + 0.18 * bool(features.get("ball_stable"))
        + 0.08 * bool(features.get("no_near_ball"))
        + 0.08 * bool(features.get("no_approach"))
        + 0.10 * bool(features.get("fewer_visible"))
        + 0.22 * bool(features.get("unassigned_anomaly"))
        + 0.25 * bool(features.get("leaving_pitch"))
    )


def _confirm_whistle_forward(
    whistle_frame: int,
    lookahead_frames: int,
    observations: dict[int, Observation],
    segment_start: int,
    segment_end: int,
    window_frames: int,
    scan_step_frames: int,
    visible_baseline: float,
    unassigned_baseline: float,
    set_piece_ranges: Sequence[tuple[int, int]],
    set_piece_starts: Sequence[int],
    *,
    slowdown_ratio: float,
    minimum_speed_drop_mps: float,
    ball_stationary_radius_m: float,
    unassigned_player_minimum: int,
    fewer_visible_margin: int,
    leaving_players_minimum: int,
    no_near_ball_fraction: float,
    no_approach_fraction: float,
) -> tuple[int, float, float] | None:
    """Accumulate evidence after a whistle until it confirms or times out."""
    scan_start = max(whistle_frame, segment_start + window_frames)
    scan_end = min(
        segment_end - window_frames + 1,
        whistle_frame + lookahead_frames - window_frames + 1,
    )
    accumulated: dict[str, float | bool] = {
        "whistle": True,
        "slowdown": False,
        "slowdown_strength": 0.0,
        "ball_stable": False,
        "ball_stability_strength": 0.0,
        "no_near_ball": False,
        "no_approach": False,
        "fewer_visible": False,
        "unassigned_anomaly": False,
        "leaving_pitch": False,
        "in_set_piece": False,
    }
    for confirmation_frame in range(scan_start, scan_end + 1, scan_step_frames):
        _, _, cues = _onset_evidence(
            confirmation_frame, observations, segment_start, window_frames,
            visible_baseline, unassigned_baseline, [whistle_frame],
            set_piece_ranges, set_piece_starts,
            slowdown_ratio=slowdown_ratio,
            minimum_speed_drop_mps=minimum_speed_drop_mps,
            ball_stationary_radius_m=ball_stationary_radius_m,
            unassigned_player_minimum=unassigned_player_minimum,
            fewer_visible_margin=fewer_visible_margin,
            leaving_players_minimum=leaving_players_minimum,
            no_near_ball_fraction=no_near_ball_fraction,
            no_approach_fraction=no_approach_fraction,
            whistle_followup_frames=lookahead_frames,
            slowdown_reference_frame=whistle_frame,
        )
        for name in ("slowdown", "ball_stable", "no_near_ball", "no_approach",
                     "fewer_visible", "unassigned_anomaly", "leaving_pitch"):
            accumulated[name] = bool(accumulated[name]) or bool(cues[name])
        for name in ("slowdown_strength", "ball_stability_strength"):
            accumulated[name] = max(float(accumulated[name]), float(cues[name]))
        corroborating_cue_here = any(bool(cues[name]) for name in (
            "slowdown", "ball_stable", "no_near_ball", "no_approach",
            "fewer_visible", "unassigned_anomaly", "leaving_pitch",
        ))
        accumulated["in_set_piece"] = (
            bool(accumulated["in_set_piece"])
            or (bool(cues["in_set_piece"]) and corroborating_cue_here)
        )

        score = _weighted_score(accumulated)
        threshold = 0.55 if accumulated["in_set_piece"] else 0.70
        has_other_cue = any(bool(accumulated[name]) for name in (
            "slowdown", "no_near_ball", "no_approach", "fewer_visible",
            "unassigned_anomaly", "leaving_pitch",
        ))
        departure_confirm = bool(accumulated["whistle"]) and bool(accumulated["leaving_pitch"])
        if departure_confirm or (score >= threshold and accumulated["ball_stable"] and has_other_cue):
            support = (0.5 * float(accumulated["slowdown_strength"])
                       + 0.3 * float(accumulated["ball_stability_strength"])
                       + 0.2 * bool(accumulated["leaving_pitch"]))
            return confirmation_frame, score, support
    return None


def _find_resumption(
    observations: dict[int, Observation],
    start: int,
    end: int,
    window_frames: int,
    resume_player_speed_mps: float,
    resume_ball_displacement_m: float,
    *,
    fps: float,
    minimum_identified_players: int,
    baseline_player_speed: float | None,
    allow_weak_candidate: bool = False,
) -> tuple[int, float] | None:
    """Choose the strongest restart window, optionally allowing a set-piece fallback."""
    best: tuple[int, float] | None = None
    for window_start in range(start, end - window_frames + 2):
        window_end = window_start + window_frames - 1
        values = _window_values(observations, window_start, window_end)
        required = max(2, int(window_frames * 0.65))
        if len(values) < required:
            continue

        identified = _window_mean(values, "identified_count") or 0.0
        average_speed = _window_mean(values, "average_speed")
        near_ball_count = _window_mean(values, "near_ball_count") or 0.0
        speed_increase = max(
            0.0,
            (average_speed or 0.0) - (baseline_player_speed or 0.0),
        )

        ball_positions = [
            (frame_id, observations[frame_id].ball)
            for frame_id in range(window_start, window_end + 1)
            if frame_id in observations and observations[frame_id].ball is not None
        ]
        enough_ball_samples = len(ball_positions) >= required
        displacement = (
            hypot(
                ball_positions[-1][1][0] - ball_positions[0][1][0],
                ball_positions[-1][1][1] - ball_positions[0][1][1],
            )
            if enough_ball_samples else 0.0
        )
        ball_speed = (
            _window_ball_speed(observations, window_start, window_end, fps) or 0.0
            if enough_ball_samples else 0.0
        )
        baseline_ball_speed = _window_ball_speed(
            observations, window_start - window_frames, window_start - 1, fps) or 0.0

        visibility_strength = min(1.0, identified / 22.0)
        ball_motion_strength = min(1.0, displacement / max(resume_ball_displacement_m, 0.01))
        player_speed_strength = min(1.0, speed_increase / max(resume_player_speed_mps * 0.5, 0.1))
        ball_speed_strength = min(
            1.0,
            max(0.0, ball_speed - baseline_ball_speed)
            / max(resume_player_speed_mps * 0.5, 0.1),
        )
        proximity_strength = min(1.0, near_ball_count / 3.0)
        score = (
            0.35 * visibility_strength
            + 0.25 * ball_motion_strength
            + 0.15 * player_speed_strength
            + 0.15 * ball_speed_strength
            + 0.10 * proximity_strength
        )

        strong_candidate = (
            enough_ball_samples
            and
            identified >= minimum_identified_players
            and displacement >= resume_ball_displacement_m
        )
        if not strong_candidate and not allow_weak_candidate:
            continue
        if best is None or score > best[1]:
            best = (window_start, score)
    return best


def _next_set_piece_range(
    frame_id: int, ranges: Sequence[tuple[int, int]], segment_end: int,
) -> tuple[int, int] | None:
    for start, end in ranges:
        if end >= frame_id and start <= segment_end:
            return max(start, frame_id), min(end, segment_end)
    return None


def detect_dead_ball_ranges(
    frames: Sequence[Frame],
    st_events: Sequence[SmartTaggingEvent],
    touches: Sequence[BallTouchEvent],
    whistle_events: Sequence[WhistleEvent] = (),
    *,
    game: str = GAME,
    fps: float | None = None,
    whistle_followup_seconds: float = 15.0,
    slowdown_window_seconds: float = 1.0,
    whistle_settle_seconds: float = 0.4,
    player_slowdown_ratio: float = 0.3,
    minimum_player_speed_drop_mps: float = 0.25,
    ball_stationary_radius_m: float = 0.45,
    near_ball_radius_m: float = 8.0,
    pitch_margin_m: float = 1.5,
    exit_lookback_seconds: float = 1.0,
    exit_outside_margin_m: float = 3.0,
    exit_edge_radius_m: float = 6.0,
    minimum_exit_progress_m: float = 0.6,
    leaving_players_minimum: int = 3,
    unassigned_player_minimum: int = 3,
    fewer_visible_margin: int = 3,
    onset_confirm_seconds: float = 0.4,
    resume_window_seconds: float = 0.75,
    resume_player_speed_mps: float = 1.0,
    resume_ball_displacement_m: float = 0.8,
    resume_minimum_identified_players: int = 19,
    scan_step_frames: int = 5,
) -> list[DeadBallRange]:
    """Return inclusive dead-ball ranges on the tactical video's frame timeline.

    Stop evidence is scored per frame with weights: whistle 0.28, player
    slowdown 0.25, stable ball 0.18, no nearby players 0.08, no one approaching
    the ball 0.08, fewer visible players than usual 0.10, unassigned players
    0.22, and players moving toward/off a pitch boundary 0.25. A set-piece window
    lowers the score threshold from 0.70 to 0.55; it never gates detection. The
    ball must be stable for the general evidence path. A whistle remains pending for
    ``whistle_followup_seconds`` (15 seconds by default), allowing later motion
    and stability cues to confirm the stoppage. Quiet whistle candidates are not
    filtered here; the caller controls the upstream detector threshold.

    Player speeds and positions are taken from tracking frames. Positions beyond
    ``pitch_margin_m`` are excluded from normal visibility and proximity cues.
    For the departure cue, assigned real tracks are retained up to
    ``exit_outside_margin_m`` beyond the pitch. At least
    ``leaving_players_minimum`` tracks must move at least
    ``minimum_exit_progress_m`` closer to a boundary within
    ``exit_lookback_seconds`` while within ``exit_edge_radius_m`` of it. This
    cue can confirm a pending whistle without ball stability. Detection-only
    players (whose placeholder speed is zero) count for visibility but do not
    contribute to tracked-player average speed or boundary departures. A player
    is counted as approaching when its distance to the current ball position falls
    by at least 0.25 m over a 0.4-second lookback. Resumption prefers a window
    with at least ``resume_minimum_identified_players`` assigned players on the
    pitch and sustained ball movement. It ranks qualifying windows by visibility,
    ball movement, rising player and ball speeds, and players near the ball. If no
    strong window is found, the next set-piece interval is searched and its best
    candidate is used as a fallback. A supported stoppage with no observed
    resumption or subsequent set piece extends to the end of its contiguous
    tracking segment. Set-piece annotations retain a bounded start-to-next-touch
    fallback only when no inferred range overlaps them.
    """
    if not frames:
        return []
    fps = GAMES[game]["fps"] if fps is None else fps
    if fps <= 0 or scan_step_frames <= 0:
        raise ValueError("fps and scan_step_frames must be positive")
    if (ball_stationary_radius_m <= 0 or near_ball_radius_m <= 0 or pitch_margin_m < 0
            or exit_lookback_seconds <= 0 or exit_outside_margin_m < 0
            or exit_edge_radius_m <= 0 or minimum_exit_progress_m <= 0):
        raise ValueError("Distance thresholds must be positive; margins must be nonnegative")
    if not 0 <= player_slowdown_ratio <= 1:
        raise ValueError("player_slowdown_ratio must be between 0 and 1")
    if any(value < 0 for value in (
        slowdown_window_seconds, whistle_settle_seconds, minimum_player_speed_drop_mps,
        exit_lookback_seconds,
        whistle_followup_seconds, onset_confirm_seconds, resume_window_seconds, resume_player_speed_mps,
        resume_ball_displacement_m,
    )) or min(unassigned_player_minimum, fewer_visible_margin) < 0 or leaving_players_minimum < 1:
        raise ValueError("Motion thresholds and time windows must be nonnegative")
    if resume_minimum_identified_players < 1:
        raise ValueError("resume_minimum_identified_players must be positive")

    window_frames = max(2, round(slowdown_window_seconds * fps))
    settle_frames = max(0, round(whistle_settle_seconds * fps))
    resume_frames = max(2, round(resume_window_seconds * fps))
    confirm_samples = max(1, ceil(onset_confirm_seconds * fps / scan_step_frames))
    whistle_followup_frames = round(whistle_followup_seconds * fps)
    approach_lookback = max(1, round(0.4 * fps))
    exit_lookback = max(1, round(exit_lookback_seconds * fps))
    observations = _build_observations(
        frames, game, approach_lookback, exit_lookback, near_ball_radius_m,
        pitch_margin_m, exit_outside_margin_m, exit_edge_radius_m,
        minimum_exit_progress_m)
    if not observations:
        return []

    frame_ids = sorted(observations)
    segments = _tracking_segments(frame_ids, max_gap=max(2, round(fps)))
    set_piece_ranges = []
    for event in sorted(st_events, key=lambda item: item.start_frame):
        if set_piece_ranges and event.start_frame <= set_piece_ranges[-1][1] + 1:
            set_piece_ranges[-1] = (set_piece_ranges[-1][0],
                                    max(set_piece_ranges[-1][1], event.end_frame))
        else:
            set_piece_ranges.append((event.start_frame, event.end_frame))
    set_piece_starts = [start for start, _ in set_piece_ranges]
    touch_video_frames = sorted(
        mapped for touch in touches
        if (mapped := _video_frame_id(
            touch.touch_frame_id if touch.touch_frame_id is not None else touch.start_frame_id,
            game,
        )) is not None
    )
    inferred_candidates: list[tuple[DeadBallRange, float, float, float]] = []

    for segment_start, segment_end in segments:
        segment_values = [observations[frame_id]
                          for frame_id in range(segment_start, segment_end + 1)
                          if frame_id in observations]
        visible_baseline = median(item.visible_count for item in segment_values)
        unassigned_baseline = median(item.unassigned_count for item in segment_values)

        # A whistle opens a pending candidate. Accumulate each cue independently
        # throughout the forward horizon so slowdown and ball stability need not
        # appear in the same one-second window.
        for whistle in whistle_events:
            whistle_frame = whistle.peak_frame_id
            if not segment_start <= whistle_frame <= segment_end:
                continue
            confirmation = _confirm_whistle_forward(
                whistle_frame, whistle_followup_frames, observations,
                segment_start, segment_end, window_frames, scan_step_frames,
                visible_baseline, unassigned_baseline,
                set_piece_ranges, set_piece_starts,
                slowdown_ratio=player_slowdown_ratio,
                minimum_speed_drop_mps=minimum_player_speed_drop_mps,
                ball_stationary_radius_m=ball_stationary_radius_m,
                unassigned_player_minimum=unassigned_player_minimum,
                fewer_visible_margin=fewer_visible_margin,
                leaving_players_minimum=leaving_players_minimum,
                no_near_ball_fraction=0.7,
                no_approach_fraction=0.7,
            )
            if confirmation is None:
                continue
            confirmation_frame, evidence_score, motion_support = confirmation
            resume_search_start = max(
                confirmation_frame + window_frames,
                whistle.end_frame_id + 1,
                whistle_frame + settle_frames,
            )
            next_set_piece = _next_set_piece_range(
                resume_search_start, set_piece_ranges, segment_end)
            search_end = next_set_piece[1] if next_set_piece else segment_end
            baseline_speed = _window_mean(
                _window_values(observations, confirmation_frame,
                               confirmation_frame + window_frames - 1),
                "average_speed",
            )
            resume_result = _find_resumption(
                observations, resume_search_start, search_end, resume_frames,
                resume_player_speed_mps, resume_ball_displacement_m,
                fps=fps,
                minimum_identified_players=resume_minimum_identified_players,
                baseline_player_speed=baseline_speed,
            )
            if resume_result is None and next_set_piece is not None:
                fallback_start, fallback_end = next_set_piece
                fallback_window = min(resume_frames, fallback_end - fallback_start + 1)
                if fallback_window >= 2:
                    resume_result = _find_resumption(
                        observations, fallback_start, fallback_end, fallback_window,
                        resume_player_speed_mps, resume_ball_displacement_m,
                        fps=fps,
                        minimum_identified_players=resume_minimum_identified_players,
                        baseline_player_speed=baseline_speed,
                        allow_weak_candidate=True,
                    )
            resume = resume_result[0] if resume_result is not None else None
            dead_end = resume - 1 if resume is not None else segment_end
            if whistle_frame <= dead_end:
                inferred_candidates.append((
                    DeadBallRange(whistle_frame, dead_end, "evidence", fps),
                    motion_support, evidence_score, whistle.peak_score,
                ))

        # Continue to find stoppages with strong non-whistle evidence as well.
        scan_start = segment_start + window_frames
        run_start = None
        run_samples = 0
        frame_id = scan_start
        while frame_id + window_frames <= segment_end + 1:
            score, accepted, features = _onset_evidence(
                frame_id, observations, segment_start, window_frames,
                visible_baseline, unassigned_baseline, (),
                set_piece_ranges, set_piece_starts,
                slowdown_ratio=player_slowdown_ratio,
                minimum_speed_drop_mps=minimum_player_speed_drop_mps,
                ball_stationary_radius_m=ball_stationary_radius_m,
                unassigned_player_minimum=unassigned_player_minimum,
                fewer_visible_margin=fewer_visible_margin,
                leaving_players_minimum=leaving_players_minimum,
                no_near_ball_fraction=0.7,
                no_approach_fraction=0.7,
                whistle_followup_frames=0,
            )
            if accepted:
                if run_start is None:
                    run_start = frame_id
                    run_samples = 0
                run_samples += 1
            else:
                run_start = None
                run_samples = 0

            if run_start is not None and run_samples >= confirm_samples:
                dead_start = run_start
                resume_search_start = max(
                    run_start + window_frames,
                    run_start + settle_frames,
                )
                next_set_piece = _next_set_piece_range(
                    resume_search_start, set_piece_ranges, segment_end)
                search_end = next_set_piece[1] if next_set_piece else segment_end
                baseline_speed = _window_mean(
                    _window_values(observations, run_start, run_start + window_frames - 1),
                    "average_speed",
                )
                resume_result = _find_resumption(
                    observations, resume_search_start, search_end, resume_frames,
                    resume_player_speed_mps, resume_ball_displacement_m,
                    fps=fps,
                    minimum_identified_players=resume_minimum_identified_players,
                    baseline_player_speed=baseline_speed,
                )
                if resume_result is None and next_set_piece is not None:
                    fallback_start, fallback_end = next_set_piece
                    fallback_window = min(resume_frames, fallback_end - fallback_start + 1)
                    if fallback_window >= 2:
                        resume_result = _find_resumption(
                            observations, fallback_start, fallback_end, fallback_window,
                            resume_player_speed_mps, resume_ball_displacement_m,
                            fps=fps,
                            minimum_identified_players=resume_minimum_identified_players,
                            baseline_player_speed=baseline_speed,
                            allow_weak_candidate=True,
                        )
                resume = resume_result[0] if resume_result is not None else None
                dead_end = resume - 1 if resume is not None else segment_end
                if dead_start <= dead_end:
                    motion_support = (
                        0.7 * float(features["slowdown_strength"])
                        + 0.3 * float(features["ball_stability_strength"])
                    )
                    inferred_candidates.append((
                        DeadBallRange(dead_start, dead_end, "evidence", fps),
                        motion_support, score, 0.0,
                    ))
                # Resume scanning after confirmed movement, avoiding duplicate
                # onset runs during the same stoppage.
                frame_id = max(frame_id + scan_step_frames, (resume or segment_end + 1))
                run_start = None
                run_samples = 0
                continue
            frame_id += scan_step_frames

    # Merge overlapping supported intervals, while preserving gaps that contain
    # sustained movement and therefore represent a real restart.
    ranges: list[DeadBallRange] = []
    ranked_ranges: list[tuple[DeadBallRange, float, float, float]] = []
    for candidate in sorted(inferred_candidates,
                            key=lambda item: item[0].start_frame_id):
        dead_range = candidate[0]
        if ranked_ranges and dead_range.start_frame_id <= ranked_ranges[-1][0].end_frame_id:
            previous, previous_support, previous_score, previous_peak = ranked_ranges[-1]
            merged_end = max(previous.end_frame_id, dead_range.end_frame_id)
            if candidate[1:] > (previous_support, previous_score, previous_peak):
                winner = candidate
            else:
                winner = ranked_ranges[-1]
            winner_range = DeadBallRange(
                winner[0].start_frame_id, merged_end, "evidence", fps)
            ranked_ranges[-1] = (winner_range, winner[1], winner[2], winner[3])
        else:
            ranked_ranges.append(candidate)
    ranges = [item[0] for item in ranked_ranges]

    # SmartTagging remains a bounded fallback only for annotations not already
    # covered by any multi-signal inferred interval.
    ordered_sets = sorted(st_events, key=lambda event: event.start_frame)
    for set_index, event in enumerate(ordered_sets):
        next_context_start = (
            ordered_sets[set_index + 1].start_frame
            if set_index + 1 < len(ordered_sets)
            else max(frame_ids) + 1
        )
        for segment_start, segment_end in segments:
            context_start = max(event.start_frame, segment_start)
            context_end = min(event.end_frame, segment_end, next_context_start - 1)
            if context_start > context_end:
                continue
            if any(item.start_frame_id <= context_end
                   and item.end_frame_id >= context_start for item in ranges):
                continue
            next_touch = next((touch_frame for touch_frame in touch_video_frames
                               if touch_frame > event.start_frame), None)
            dead_end = min(next_touch - 1 if next_touch is not None else context_end,
                           context_end)
            if context_start <= dead_end:
                ranges.append(DeadBallRange(context_start, dead_end, "fallback", fps))

    merged: list[DeadBallRange] = []
    for dead_range in sorted(ranges, key=lambda item: item.start_frame_id):
        if merged and dead_range.start_frame_id <= merged[-1].end_frame_id:
            previous = merged[-1]
            source = ("evidence" if "evidence" in (previous.source, dead_range.source)
                      else "fallback")
            merged[-1] = DeadBallRange(
                previous.start_frame_id,
                max(previous.end_frame_id, dead_range.end_frame_id),
                source,
                fps,
            )
        else:
            merged.append(dead_range)
    return merged
