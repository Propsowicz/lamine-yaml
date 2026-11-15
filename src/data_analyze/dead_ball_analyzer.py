"""Estimate dead-ball intervals from whistle, ball motion, and player engagement."""

from collections.abc import Sequence
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import timedelta
from math import ceil, hypot, isfinite
from statistics import median
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
    ball: Point | None
    near_ball_by_team: dict[str, int] | None
    approaching_by_team: dict[str, int] | None


@dataclass(frozen=True)
class DeadBallRange:
    """Inclusive video-frame range with the method that produced it."""

    start_frame_id: int
    end_frame_id: int
    source: Literal["evidence", "fallback", "assumption"]
    fps: float = field(default=GAMES[GAME]["fps"], repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.start_frame_id > self.end_frame_id:
            raise ValueError("start_frame_id must be <= end_frame_id")
        if self.source not in ("evidence", "fallback", "assumption"):
            raise ValueError("source must be 'evidence', 'fallback', or 'assumption'")
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


def _ball_position(frame: Frame, pitch_margin_m: float) -> Point | None:
    x, y = frame.ball.x, frame.ball.y
    if x is None or y is None:
        return None
    x, y = float(x), float(y)
    if not isfinite(x) or not isfinite(y):
        return None
    if not (-pitch_margin_m <= x <= PITCH_LENGTH + pitch_margin_m
            and -pitch_margin_m <= y <= PITCH_WIDTH + pitch_margin_m):
        return None
    return x, y


def _build_observations(
    frames: Sequence[Frame], game: str, approach_lookback_frames: int,
    near_ball_radius_m: float, approach_radius_m: float, pitch_margin_m: float,
) -> dict[int, Observation]:
    raw: dict[int, tuple[Point | None, dict[str, Point], dict[str, str]]] = {}
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
        identified_positions: dict[str, Point] = {}
        player_teams: dict[str, str] = {}
        ball = _ball_position(frame, pitch_margin_m)
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
            # Homography/detection outliers can land well beyond the pitch.
            # They should not affect ball-proximity or approach evidence.
            if not (-pitch_margin_m <= x <= PITCH_LENGTH + pitch_margin_m
                    and -pitch_margin_m <= y <= PITCH_WIDTH + pitch_margin_m):
                continue
            is_detection = getattr(player, "from_detection", False)
            is_virtual = getattr(player, "validator_anchor_id", None) is not None
            if assigned and not is_detection and not is_virtual:
                identified_positions[player_id] = (x, y)
                player_teams[player_id] = team
        raw[video_id] = (ball, identified_positions, player_teams)

    observations: dict[int, Observation] = {}
    for video_id, (ball, identified_positions, player_teams) in raw.items():
        near_by_team: dict[str, int] | None = None
        if ball is None or not identified_positions:
            near_by_team = None
        else:
            near_by_team = {}
            for player_id, (x, y) in identified_positions.items():
                if hypot(x - ball[0], y - ball[1]) <= near_ball_radius_m:
                    team = player_teams[player_id]
                    near_by_team[team] = near_by_team.get(team, 0) + 1

        previous = raw.get(video_id - approach_lookback_frames)
        approaching_by_team: dict[str, int] | None = None
        if (previous is not None and ball is not None and identified_positions
                and previous[1]):
            old_positions = previous[1]
            approaching_by_team = {}
            matched_players = 0
            for player_id, current in identified_positions.items():
                old = old_positions.get(player_id)
                if old is None:
                    continue
                matched_players += 1
                old_distance = hypot(old[0] - ball[0], old[1] - ball[1])
                current_distance = hypot(current[0] - ball[0], current[1] - ball[1])
                if (current_distance <= approach_radius_m
                        and old_distance - current_distance >= 0.25):
                    team = player_teams[player_id]
                    approaching_by_team[team] = approaching_by_team.get(team, 0) + 1
            if matched_players == 0:
                approaching_by_team = None

        observations[video_id] = Observation(
            ball=ball,
            near_ball_by_team=near_by_team,
            approaching_by_team=approaching_by_team,
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


def _window_ball_motion(
    observations: dict[int, Observation], start: int, end: int, fps: float,
) -> tuple[float, float] | None:
    positions = [
        (frame_id, observations[frame_id].ball)
        for frame_id in range(start, end + 1)
        if frame_id in observations and observations[frame_id].ball is not None
    ]
    if len(positions) < 2:
        return None
    edge_samples = max(2, ceil(len(positions) * 0.2))
    first = positions[:edge_samples]
    last = positions[-edge_samples:]
    first_point = (
        median(point[0] for _, point in first),
        median(point[1] for _, point in first),
    )
    last_point = (
        median(point[0] for _, point in last),
        median(point[1] for _, point in last),
    )
    distance = hypot(last_point[0] - first_point[0], last_point[1] - first_point[1])
    duration = max(1 / fps, (positions[-1][0] - positions[0][0]) / fps)
    return distance, distance / duration


def _onset_evidence(
    frame_id: int,
    observations: dict[int, Observation],
    ball_stationary_window_frames: int,
    strong_stationary_window_frames: int,
    whistle_frames: Sequence[int],
    *,
    ball_stationary_radius_m: float,
    whistle_followup_frames: int,
    few_nearby_players_max: int,
) -> tuple[bool, float]:
    whistle_index = bisect_right(whistle_frames, frame_id) - 1
    whistle = (whistle_index >= 0
               and frame_id - whistle_frames[whistle_index] <= whistle_followup_frames)
    required_stationary = (ball_stationary_window_frames if whistle else
                           strong_stationary_window_frames)
    ball_post = _window_values(observations, frame_id,
                               frame_id + required_stationary - 1)
    spread, ball_samples = _ball_spread(ball_post)
    ball_required_samples = max(2, ceil(required_stationary * 0.9))
    ball_stable = (spread is not None and ball_samples >= ball_required_samples
                   and spread <= ball_stationary_radius_m)
    post = _window_values(observations, frame_id,
                          frame_id + required_stationary - 1)
    required_evidence_samples = max(2, ceil(required_stationary * 0.9))
    valid_approach = [item.approaching_by_team for item in post
                      if item.approaching_by_team is not None]
    valid_near = [item.near_ball_by_team for item in post
                  if item.near_ball_by_team is not None]
    approach_evidence_valid = len(valid_approach) >= required_evidence_samples
    proximity_evidence_valid = len(valid_near) >= required_evidence_samples
    no_approach = (approach_evidence_valid
                   and all(not teams for teams in valid_approach))
    approach_teams = {
        team for teams in valid_approach for team, count in teams.items()
        if count > 0
    }
    team_near_totals: dict[str, int] = {}
    for teams in valid_near:
        for team, count in teams.items():
            team_near_totals[team] = team_near_totals.get(team, 0) + count
    near_total = sum(team_near_totals.values())
    team_dominated = (near_total > 0 and max(team_near_totals.values())
                      >= near_total * 0.7)
    few_nearby = (proximity_evidence_valid and
                  sum(sum(teams.values()) for teams in valid_near) / max(len(valid_near), 1)
                  <= few_nearby_players_max)
    engagement_evidence_valid = approach_evidence_valid and proximity_evidence_valid
    low_engagement = engagement_evidence_valid and len(approach_teams) < 2 and (
        (no_approach or team_dominated or few_nearby) if whistle
        else (no_approach and (team_dominated or few_nearby))
    )
    accepted = ball_stable and low_engagement
    stability_support = (max(0.0, 1.0 - spread / ball_stationary_radius_m)
                         if ball_stable and spread is not None else 0.0)
    return accepted, stability_support


def _confirm_whistle_forward(
    whistle_frame: int,
    lookahead_frames: int,
    observations: dict[int, Observation],
    segment_start: int,
    segment_end: int,
    ball_stationary_window_frames: int,
    strong_stationary_window_frames: int,
    scan_step_frames: int,
    *,
    ball_stationary_radius_m: float,
    few_nearby_players_max: int,
) -> tuple[int, float] | None:
    """Accumulate evidence after a whistle until it confirms or times out."""
    analysis_window_frames = ball_stationary_window_frames
    scan_start = max(whistle_frame, segment_start + 1)
    scan_end = min(
        segment_end - analysis_window_frames + 1,
        whistle_frame + lookahead_frames - analysis_window_frames + 1,
    )
    for confirmation_frame in range(scan_start, scan_end + 1, scan_step_frames):
        accepted, stability_support = _onset_evidence(
            confirmation_frame, observations,
            ball_stationary_window_frames, strong_stationary_window_frames,
            [whistle_frame],
            ball_stationary_radius_m=ball_stationary_radius_m,
            whistle_followup_frames=lookahead_frames,
            few_nearby_players_max=few_nearby_players_max,
        )
        if accepted:
            return confirmation_frame, stability_support
    return None


def _find_resumption(
    observations: dict[int, Observation],
    start: int,
    end: int,
    window_frames: int,
    resume_ball_displacement_m: float,
    *,
    fps: float,
    minimum_ball_speed_mps: float = 0.8,
    no_whistle_ball_displacement_m: float = 1.5,
    allow_weak_candidate: bool = False,
    whistle_precedes_ball_change: bool = False,
    return_first_strong_candidate: bool = False,
) -> int | None:
    """Find the first strongly supported restart, or the best fallback window."""
    best: tuple[int, float] | None = None
    for window_start in range(start, end - window_frames + 2):
        window_end = window_start + window_frames - 1
        values = _window_values(observations, window_start, window_end)
        required = max(2, int(window_frames * 0.65))
        if len(values) < required:
            continue
        valid_approach = [item.approaching_by_team for item in values
                          if item.approaching_by_team is not None]
        valid_near = [item.near_ball_by_team for item in values
                      if item.near_ball_by_team is not None]
        if len(valid_approach) < required or len(valid_near) < required:
            continue

        ball_positions = [
            (frame_id, observations[frame_id].ball)
            for frame_id in range(window_start, window_end + 1)
            if frame_id in observations and observations[frame_id].ball is not None
        ]
        enough_ball_samples = len(ball_positions) >= required
        ball_motion = (_window_ball_motion(
            observations, window_start, window_end, fps) if enough_ball_samples else None)
        displacement, ball_speed = ball_motion or (0.0, 0.0)
        approached_teams = {
            team for teams in valid_approach for team, count in teams.items()
            if count > 0
        }
        engaged_near_ball = any(
            any(teams.values()) for teams in valid_near
        ) or any(
            any(teams.values()) for teams in valid_approach
        )
        required_displacement = (
            resume_ball_displacement_m if whistle_precedes_ball_change or allow_weak_candidate
            else no_whistle_ball_displacement_m
        )
        movement_confirmed = (
            enough_ball_samples
            and displacement >= required_displacement
            and ball_speed >= minimum_ball_speed_mps
        )
        engagement_confirmed = (engaged_near_ball if whistle_precedes_ball_change
                                else len(approached_teams) >= 2)
        strong_candidate = movement_confirmed and engagement_confirmed
        if not movement_confirmed:
            continue
        if not strong_candidate and not allow_weak_candidate:
            continue
        if strong_candidate and return_first_strong_candidate:
            return window_start
        if best is None or displacement > best[1]:
            best = (window_start, displacement)
    return best[0] if best is not None else None


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
    whistle_followup_seconds: float = 10.0,
    whistle_settle_seconds: float = 0.4,
    ball_stationary_radius_m: float = 0.45,
    ball_stationary_min_duration_seconds: float = 3.0,
    no_whistle_stationary_min_duration_seconds: float = 5.0,
    few_nearby_players_max: int = 2,
    near_ball_radius_m: float = 8.0,
    approach_radius_m: float = 15.0,
    pitch_margin_m: float = 1.5,
    onset_confirm_seconds: float = 0.4,
    resume_window_seconds: float = 0.75,
    resume_ball_displacement_m: float = 0.8,
    resume_no_whistle_displacement_m: float = 1.5,
    resume_ball_speed_mps: float = 0.8,
    scan_step_frames: int = 5,
) -> list[DeadBallRange]:
    """Return inclusive dead-ball ranges on the tactical video's frame timeline.

    A whistle-supported stoppage needs a stationary ball for
    ``ball_stationary_min_duration_seconds`` (3 seconds by default), plus
    either no approaches or a near-ball group dominated by one team / containing
    few players. Without a recent whistle, the ball must remain stationary
    longer (5 seconds by default), with no approaches and a one-team-dominated
    or small near-ball group. Approach and proximity evidence each need at least
    90% of expected samples. The whistle evidence horizon is 10 seconds by
    default.
    Unless the first input frame is inside a whistle event, the segment starts
    with an assumed dead-ball range; the detector then searches for its first
    resumption and uses the next set-piece interval if normal evidence is weak.
    A whistle remains pending for ``whistle_followup_seconds`` (10 seconds by
    default), allowing ball-stationarity and engagement evidence to confirm the
    stoppage. Quiet whistle candidates are not
    filtered here; the caller controls the upstream detector threshold.

    Ball and player positions beyond ``pitch_margin_m`` outside the pitch are
    excluded. A player is counted as approaching when its distance to the current
    ball position falls by at least 0.25 m over a 0.4-second lookback, and only
    when the player is within ``approach_radius_m`` (15 m by default). Resumption
    requires ball movement plus engagement: after a whistle nearby or
    approaching players suffice; without a whistle, both teams must approach.
    Ball movement uses the displacement between median positions at the start
    and end of each window, which dampens frame-to-frame tracking wobble.
    Defaults require at least 0.8 m displacement and 0.8 m/s after a whistle, or
    1.5 m displacement and approaches from both teams without a whistle.
    If no strong window is found, the next set-piece interval is searched and its best
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
    if (ball_stationary_radius_m <= 0 or near_ball_radius_m <= 0
            or approach_radius_m <= 0 or pitch_margin_m < 0
            or ball_stationary_min_duration_seconds <= 0
            or no_whistle_stationary_min_duration_seconds <= 0
            or few_nearby_players_max < 0):
        raise ValueError("Ball thresholds must be positive and player-count threshold nonnegative")
    if any(value < 0 for value in (
        whistle_settle_seconds,
        ball_stationary_min_duration_seconds, no_whistle_stationary_min_duration_seconds,
        whistle_followup_seconds, onset_confirm_seconds, resume_window_seconds,
        resume_ball_displacement_m, resume_no_whistle_displacement_m, resume_ball_speed_mps,
    )):
        raise ValueError("Motion thresholds and time windows must be nonnegative")

    ball_stationary_window_frames = max(
        2, int(ball_stationary_min_duration_seconds * fps) + 2)
    strong_stationary_window_frames = max(
        ball_stationary_window_frames,
        int(no_whistle_stationary_min_duration_seconds * fps) + 2)
    settle_frames = max(0, round(whistle_settle_seconds * fps))
    resume_frames = max(2, round(resume_window_seconds * fps))
    confirm_samples = max(1, ceil(onset_confirm_seconds * fps / scan_step_frames))
    whistle_followup_frames = round(whistle_followup_seconds * fps)
    approach_lookback = max(1, round(0.4 * fps))
    observations = _build_observations(
        frames, game, approach_lookback, near_ball_radius_m,
        approach_radius_m, pitch_margin_m)
    if not observations:
        return []

    frame_ids = sorted(observations)
    set_piece_ranges = []
    for event in sorted(st_events, key=lambda item: item.start_frame):
        if set_piece_ranges and event.start_frame <= set_piece_ranges[-1][1] + 1:
            set_piece_ranges[-1] = (set_piece_ranges[-1][0],
                                    max(set_piece_ranges[-1][1], event.end_frame))
        else:
            set_piece_ranges.append((event.start_frame, event.end_frame))
    touch_video_frames = sorted(
        mapped for touch in touches
        if (mapped := _video_frame_id(
            touch.touch_frame_id if touch.touch_frame_id is not None else touch.start_frame_id,
            game,
        )) is not None
    )
    inferred_candidates: list[tuple[DeadBallRange, float, float]] = []

    segment_start, segment_end = frame_ids[0], frame_ids[-1]
    assumed_initial_range = None
    first_frame_is_whistle = any(
        whistle.start_frame_id <= segment_start <= whistle.end_frame_id
        for whistle in whistle_events
    )
    if not first_frame_is_whistle:
        next_set_piece = _next_set_piece_range(
            segment_start, set_piece_ranges, segment_end)
        search_end = next_set_piece[1] if next_set_piece else segment_end
        resume_result = _find_resumption(
            observations, segment_start, search_end, resume_frames,
            resume_ball_displacement_m,
            fps=fps,
            minimum_ball_speed_mps=resume_ball_speed_mps,
            no_whistle_ball_displacement_m=resume_no_whistle_displacement_m,
            return_first_strong_candidate=True,
        )
        if resume_result is None and next_set_piece is not None:
            fallback_start, fallback_end = next_set_piece
            fallback_window = min(resume_frames, fallback_end - fallback_start + 1)
            if fallback_window >= 2:
                resume_result = _find_resumption(
                    observations, fallback_start, fallback_end, fallback_window,
                    resume_ball_displacement_m,
                    fps=fps,
                    minimum_ball_speed_mps=resume_ball_speed_mps,
                    no_whistle_ball_displacement_m=resume_no_whistle_displacement_m,
                    allow_weak_candidate=True,
                )
        resume = resume_result
        if resume is None or resume > segment_start:
            assumed_initial_range = DeadBallRange(
                segment_start, resume - 1 if resume is not None else segment_end,
                "assumption", fps)

    # A whistle opens a candidate that needs ball and engagement evidence.
    for whistle in whistle_events:
        whistle_frame = whistle.peak_frame_id
        if not segment_start <= whistle_frame <= segment_end:
            continue
        confirmation = _confirm_whistle_forward(
            whistle_frame, whistle_followup_frames, observations,
            segment_start, segment_end, ball_stationary_window_frames,
            strong_stationary_window_frames, scan_step_frames,
            ball_stationary_radius_m=ball_stationary_radius_m,
            few_nearby_players_max=few_nearby_players_max,
        )
        if confirmation is None:
            continue
        confirmation_frame, motion_support = confirmation
        resume_search_start = max(
            whistle.end_frame_id + 1,
            whistle_frame + settle_frames,
        )
        next_set_piece = _next_set_piece_range(
            resume_search_start, set_piece_ranges, segment_end)
        search_end = next_set_piece[1] if next_set_piece else segment_end
        resume_result = _find_resumption(
            observations, resume_search_start, search_end, resume_frames,
            resume_ball_displacement_m,
            fps=fps,
            minimum_ball_speed_mps=resume_ball_speed_mps,
            no_whistle_ball_displacement_m=resume_no_whistle_displacement_m,
            whistle_precedes_ball_change=True,
            return_first_strong_candidate=True,
        )
        if resume_result is None and next_set_piece is not None:
            fallback_start, fallback_end = next_set_piece
            fallback_window = min(resume_frames, fallback_end - fallback_start + 1)
            if fallback_window >= 2:
                resume_result = _find_resumption(
                    observations, fallback_start, fallback_end, fallback_window,
                    resume_ball_displacement_m,
                    fps=fps,
                    minimum_ball_speed_mps=resume_ball_speed_mps,
                    no_whistle_ball_displacement_m=resume_no_whistle_displacement_m,
                    allow_weak_candidate=True,
                )
        resume = resume_result
        dead_end = resume - 1 if resume is not None else segment_end
        if whistle_frame <= dead_end:
            inferred_candidates.append((
                DeadBallRange(whistle_frame, dead_end, "evidence", fps),
                motion_support, whistle.peak_score,
            ))

    # Continue to find stoppages with strong non-whistle evidence as well.
    scan_start = segment_start + 1
    run_start = None
    run_samples = 0
    frame_id = scan_start
    analysis_window_frames = strong_stationary_window_frames
    while frame_id + analysis_window_frames <= segment_end + 1:
        accepted, motion_support = _onset_evidence(
            frame_id, observations, ball_stationary_window_frames,
            strong_stationary_window_frames, (),
            ball_stationary_radius_m=ball_stationary_radius_m,
            whistle_followup_frames=0,
            few_nearby_players_max=few_nearby_players_max,
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
            resume_search_start = run_start + settle_frames
            next_set_piece = _next_set_piece_range(
                resume_search_start, set_piece_ranges, segment_end)
            search_end = next_set_piece[1] if next_set_piece else segment_end
            resume_result = _find_resumption(
                observations, resume_search_start, search_end, resume_frames,
                resume_ball_displacement_m,
                fps=fps,
                minimum_ball_speed_mps=resume_ball_speed_mps,
                no_whistle_ball_displacement_m=resume_no_whistle_displacement_m,
            )
            if resume_result is None and next_set_piece is not None:
                fallback_start, fallback_end = next_set_piece
                fallback_window = min(resume_frames, fallback_end - fallback_start + 1)
                if fallback_window >= 2:
                    resume_result = _find_resumption(
                        observations, fallback_start, fallback_end, fallback_window,
                        resume_ball_displacement_m,
                        fps=fps,
                        minimum_ball_speed_mps=resume_ball_speed_mps,
                        no_whistle_ball_displacement_m=resume_no_whistle_displacement_m,
                        allow_weak_candidate=True,
                    )
            resume = resume_result
            dead_end = resume - 1 if resume is not None else segment_end
            if dead_start <= dead_end:
                inferred_candidates.append((
                    DeadBallRange(dead_start, dead_end, "evidence", fps),
                    motion_support, 0.0,
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
    ranked_ranges: list[tuple[DeadBallRange, float, float]] = []
    for candidate in sorted(inferred_candidates,
                            key=lambda item: item[0].start_frame_id):
        dead_range = candidate[0]
        if ranked_ranges and dead_range.start_frame_id <= ranked_ranges[-1][0].end_frame_id:
            previous, previous_support, previous_peak = ranked_ranges[-1]
            merged_end = max(previous.end_frame_id, dead_range.end_frame_id)
            if candidate[1:] > (previous_support, previous_peak):
                winner = candidate
            else:
                winner = ranked_ranges[-1]
            winner_range = DeadBallRange(
                winner[0].start_frame_id, merged_end, "evidence", fps)
            ranked_ranges[-1] = (winner_range, winner[1], winner[2])
        else:
            ranked_ranges.append(candidate)
    ranges = [item[0] for item in ranked_ranges]
    if assumed_initial_range is not None:
        ranges.append(assumed_initial_range)

    # SmartTagging remains a bounded fallback only for annotations not already
    # covered by any multi-signal inferred interval.
    ordered_sets = sorted(st_events, key=lambda event: event.start_frame)
    for set_index, event in enumerate(ordered_sets):
        next_context_start = (
            ordered_sets[set_index + 1].start_frame
            if set_index + 1 < len(ordered_sets)
            else max(frame_ids) + 1
        )
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
            sources = (previous.source, dead_range.source)
            source = (
                "evidence" if "evidence" in sources
                else "fallback" if "fallback" in sources
                else "assumption"
            )
            merged[-1] = DeadBallRange(
                previous.start_frame_id,
                max(previous.end_frame_id, dead_range.end_frame_id),
                source,
                fps,
            )
        else:
            merged.append(dead_range)
    return merged
