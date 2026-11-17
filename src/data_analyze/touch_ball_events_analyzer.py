from collections.abc import Sequence
from typing import Literal
from data_prepare.touch_ball_tracker import BallTouchEvent, TouchType
from data_analyze.ball_out_analyzer import BallOutEvent
FrameLabel = Literal["X", "D", "H", "A"]


def create_frame_labels(
    number_of_frames: int,
    home_team: str,
    away_team: str,
    events: Sequence[BallTouchEvent],
    dead_ball_ranges: Sequence[tuple[int, int]] = (),
    ball_out_events: Sequence[BallOutEvent] = (),
    outside_game_ranges: Sequence[tuple[int, int]] = (),
) -> list[FrameLabel]:
    if number_of_frames < 0:
        raise ValueError("number_of_frames cannot be negative")

    if home_team == away_team:
        raise ValueError("Teams must be different")

    labels: list[FrameLabel] = ["X"] * number_of_frames

    ordered_events = sorted(
        events,
        key=lambda event: event.start_frame_id,
    )

    def get_player_team_label(event: BallTouchEvent) -> FrameLabel | None:
        if not event.players:
            return None

        player = min(
            event.players,
            key=lambda player: player.min_distance_m,
        )

        if player.team == home_team:
            return "H"

        if player.team == away_team:
            return "A"

        return None

    def get_team_label(event: BallTouchEvent) -> FrameLabel | None:
        if event.touch_type not in (
            TouchType.POSSESSION,
            TouchType.PASS,
        ):
            return None

        return get_player_team_label(event)

    def fill_range(
        start_frame: int,
        end_frame: int,
        label: FrameLabel,
    ) -> None:
        start = max(0, start_frame)
        end = min(number_of_frames - 1, end_frame)

        if start > end:
            return

        labels[start:end + 1] = [label] * (end - start + 1)

    # Determine team for each event.
    resolved_labels = [
        get_team_label(event)
        for event in ordered_events
    ]

    # Resolve UNCONFIRMED events using the next known team.
    next_team: FrameLabel | None = None

    for i in range(len(ordered_events) - 1, -1, -1):
        event = ordered_events[i]

        if event.touch_type == TouchType.UNCONFIRMED:
            resolved_labels[i] = next_team
        elif resolved_labels[i] is not None:
            next_team = resolved_labels[i]

    # Build possession chains.
    i = 0

    while i < len(ordered_events):
        label = resolved_labels[i]

        if label is None:
            i += 1
            continue

        chain_start = i
        chain_end = i

        # Group consecutive events belonging to the same team.
        while (
            chain_end + 1 < len(ordered_events)
            and resolved_labels[chain_end + 1] == label
        ):
            chain_end += 1

        start_frame = ordered_events[chain_start].start_frame_id
        end_frame = ordered_events[chain_end].end_frame_id

        # When the next event is UNCONFIRMED, the current
        # team's possession continues until that event starts.
        next_index = chain_end + 1

        if (
            next_index < len(ordered_events)
            and ordered_events[next_index].touch_type
            == TouchType.UNCONFIRMED
        ):
            end_frame = (
                ordered_events[next_index].start_frame_id - 1
            )

        fill_range(start_frame, end_frame, label)

        i = chain_end + 1

    # Apply ball-out labels last, overriding any previous labels.
    for event in ball_out_events:
        fill_range(
            event.start_frame_id,
            event.end_frame_id,
            "D",
        )

    # Apply dead-ball labels.
    for start_frame, end_frame in dead_ball_ranges:
        fill_range(start_frame, end_frame, "D")

    # A restart touch can be UNCONFIRMED, but its nearest player still
    # identifies the team that took the first touch after a dead-ball range.
    restart_frames = {end_frame + 1 for _, end_frame in dead_ball_ranges}
    for restart_frame in restart_frames:
        if not 0 <= restart_frame < number_of_frames:
            continue

        restart_event = next(
            (
                event
                for event in ordered_events
                if event.touch_type == TouchType.UNCONFIRMED
                and (
                    event.touch_frame_id
                    if event.touch_frame_id is not None
                    else event.start_frame_id
                ) == restart_frame
            ),
            None,
        )
        if restart_event is None:
            continue

        restart_label = get_player_team_label(restart_event)
        if restart_label is not None:
            labels[restart_frame] = restart_label

    for start_frame, end_frame in outside_game_ranges:
        fill_range(start_frame, end_frame, "D")
    
    return labels
