from collections.abc import Sequence
from data_prepare.smart_tagging import SmartTaggingEvent
from whistle_analysis.whistle_events import WhistleEvent
from models.frame import Frame


def detect_outside_game_ranges(
    st_events: Sequence[SmartTaggingEvent],
    whistle_events: Sequence[WhistleEvent],
    end_frame: int,
) -> list[tuple[int, int]]:
    first_half_outside_game_range = find_first_half_outside_game_range(st_events)
    second_half_outside_game_range = find_second_half_outside_game_range(st_events, whistle_events)
    outside_game_range_after_finished = find_outside_game_range_after_finished(
        whistle_events, end_frame
    )
    outside_ranges: list[tuple[int, int]] = []

    for outside_range in (
        first_half_outside_game_range,
        second_half_outside_game_range,
        outside_game_range_after_finished,
    ):
        if outside_range is None:
            continue

        start = max(outside_range[0], 0)
        end = min(outside_range[1], end_frame)
        if start <= end:
            outside_ranges.append((start, end))
    return sorted(outside_ranges)


def find_first_half_outside_game_range(st_events: Sequence[SmartTaggingEvent]):
    kickoff = next((st_ev for st_ev in st_events if st_ev.tags.get("Type") == "Kick Off (Start)" and st_ev.tags.get("Half") == '1st Half'), None)
    
    if kickoff is None:
        return None
    return (0, kickoff.start_frame)

def find_second_half_outside_game_range(st_events: Sequence[SmartTaggingEvent], whistle_events: Sequence[WhistleEvent]):
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

def find_outside_game_range_after_finished(
    whistle_events: Sequence[WhistleEvent],
    end_frame: int
):
    last_whistle = max(
        whistle_events,
        key=lambda whistle: whistle.peak_frame_id,
        default=None,
    )
    if last_whistle is None:
        return None
    return (last_whistle.peak_frame_id, end_frame)
