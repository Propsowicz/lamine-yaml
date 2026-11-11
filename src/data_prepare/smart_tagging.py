"""Load Smart Tagging events on the tactical video's frame timeline."""

import os
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict
import xml.etree.ElementTree as ET

from .games import GAME, GAMES, require


class SmartTaggingEvent(TypedDict):
    code: str | None
    start_s: float
    end_s: float
    start_time: str
    end_time: str
    start_frame: int
    end_frame: int
    tags: dict[str | None, str | None]


def _format_video_time(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds_in_minute = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_in_minute:02d}"


def load_smart_tagging(
    data_dir: str | Path | None = None,
    *,
    game: str = GAME,
    fps: float | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    event_types: str | Iterable[str] | None = None,
) -> list[SmartTaggingEvent]:
    """Read events in start-frame order with half-open tactical video intervals.

    XML times are seconds from the start of the video. Frame boundaries are
    rounded to the nearest frame using the game's FPS unless overridden.
    The optional frame range is inclusive at both ends. Overlapping events
    retain their original boundaries.
    Event types match the XML's Type tag case-insensitively. Pass one type
    or an iterable of types to match any of them.
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
        events.append({
            "code": instance.findtext("code"),
            "start_s": start_s,
            "end_s": end_s,
            "start_time": _format_video_time(start_s),
            "end_time": _format_video_time(end_s),
            "start_frame": event_start,
            "end_frame": event_end,
            "tags": tags,
        })
    events.sort(key=lambda event: event["start_frame"])
    return events
