"""Load Smart Tagging events on the tactical video's frame timeline."""

import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path
import xml.etree.ElementTree as ET

from .games import GAME, GAMES, require
from utils.time_format import format_elapsed_time


class SmartTaggingCode(str, Enum):
    PLAYERS_IN_THE_BOX = "PLAYERS IN THE BOX"
    BALL_IN_FINAL_THIRD = "BALL IN FINAL THIRD"
    BALL_IN_THE_BOX = "BALL IN THE BOX"
    PROGRESSION = "PROGRESSION"
    DEFENDING_IN_MIDDLE_THIRD = "DEFENDING IN MIDDLE THIRD"
    SET_PIECES = "SET PIECES"
    CREATING_CHANCES = "CREATING CHANCES"
    DEFENDING_IN_DEFENSIVE_THIRD = "DEFENDING IN DEFENSIVE THIRD"
    LONG_BALL = "LONG BALL"
    ATTACKING_TRANSITION = "ATTACKING TRANSITION"
    DEFENSIVE_TRANSITION = "DEFENSIVE TRANSITION"
    BUILD_UP = "BUILD UP"
    DEFENDING_IN_ATTACKING_THIRD = "DEFENDING IN ATTACKING THIRD"
    GOALS = "GOALS"


@dataclass
class SmartTaggingEvent:
    code: str | None
    start_timestamp: timedelta
    end_timestamp: timedelta
    start_frame: int
    end_frame: int
    tags: dict[str | None, str | None]

    @property
    def start_time(self) -> str:
        return format_elapsed_time(self.start_timestamp)

    @property
    def end_time(self) -> str:
        return format_elapsed_time(self.end_timestamp)


def load_smart_tagging(
    data_dir: str | Path | None = None,
    *,
    game: str = GAME,
    fps: float | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    event_types: str | Iterable[str] | None = None,
    code: SmartTaggingCode | None = None,
) -> list[SmartTaggingEvent]:
    """Read events in start-frame order with half-open tactical video intervals.

    XML times are seconds from the start of the video. Frame boundaries are
    rounded to the nearest frame using the game's FPS unless overridden.
    The optional frame range is inclusive at both ends. Overlapping events
    retain their original boundaries.
    Event types match the XML's Type tag case-insensitively. Pass one type
    or an iterable of types to match any of them.
    The optional code matches the XML's code element exactly.
    """
    if start_frame is not None and end_frame is not None and start_frame > end_frame:
        raise ValueError("start_frame must be less than or equal to end_frame")
    if data_dir is None:
        data_dir = os.environ.get("MTAH_DATA_DIR") or Path(__file__).resolve().parents[2] / "data"
    xml_path, = require(["smart_tagging"], game=game, data_dir=data_dir)
    fps = GAMES[game]["fps"] if fps is None else fps
    if isinstance(event_types, str):
        event_types = (event_types,)
    selected_types = (
        None if event_types is None else {value.strip().casefold() for value in event_types}
    )

    events: list[SmartTaggingEvent] = []
    for instance in ET.parse(xml_path).iter("instance"):
        event_code = instance.findtext("code")
        if code is not None and event_code != code.value:
            continue
        start_s = float(instance.findtext("start", ""))
        end_s = float(instance.findtext("end", ""))
        event_start = round(start_s * fps)
        event_end = round(end_s * fps)
        if start_frame is not None and event_end <= start_frame:
            continue
        if end_frame is not None and event_start > end_frame:
            continue
        tags = {
            label.findtext("group"): label.findtext("text")
            for label in instance.findall("label")
        }
        event_type = tags.get("Type")
        if selected_types is not None and (
            event_type is None or event_type.strip().casefold() not in selected_types
        ):
            continue
        events.append(SmartTaggingEvent(
            code=event_code,
            start_timestamp=timedelta(seconds=start_s),
            end_timestamp=timedelta(seconds=end_s),
            start_frame=event_start,
            end_frame=event_end,
            tags=tags,
        ))
    events.sort(key=lambda event: event.start_frame)
    return events
