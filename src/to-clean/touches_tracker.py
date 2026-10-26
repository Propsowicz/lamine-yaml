
TOUCH_DISTANCE_LIMIT_IN_METERS = 0.8
MIN_TOUCH_FRAMES = 5



import pandas as pd
from pathlib import Path
from models import Ball, Player, Frame
from utils import read_frames
from enum import Enum
from typing import Any

import math
import pandas as pd


class TouchType(str, Enum):
    CONTROLLED = "Controlled"
    CONTEST = "Contest"
    TOUCH = "Touch"





def collect_touch_periods(
    frames: list[Frame],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    touches: list[dict[str, Any]] = []
    untouched_periods: list[dict[str, Any]] = []

    period_frames: list[Frame] = []
    period_players: dict[str, Player] = {}
    period_kind: str | None = None

    def close_period() -> None:
        nonlocal period_frames, period_players, period_kind

        if not period_frames:
            return

        first = period_frames[0]
        last = period_frames[-1]

        result: dict[str, Any] = {
            "start_frame_id": first.id,
            "end_frame_id": last.id,
            "start_timestamp": first.timestamp,
            "end_timestamp": last.timestamp,
        }

        if period_kind == "untouched":
            untouched_periods.append(result)
        else:
            player_list = [
                {"player_id": player.player_id, "team": player.team}
                for player in sorted(
                    period_players.values(),
                    key=lambda player: player.dist_to_ball,
                )
            ]
            if len(player_list) >= 2 and len(period_frames):
                touch_type = TouchType.CONTEST
            elif len(player_list) == 1 and len(period_frames) >= MIN_TOUCH_FRAMES:
                touch_type = TouchType.CONTROLLED
            else:
                touch_type = TouchType.TOUCH

            result["touch_type"] = touch_type.value
            result["list_of_players"] = player_list
            touches.append(result)

        period_frames = []
        period_players = {}
        period_kind = None

    for frame in frames[0:1000]:
        nearby_players = [
            player
            for player in frame.players
            if player.dist_to_ball < TOUCH_DISTANCE_LIMIT_IN_METERS
        ]

        kind = "touch" if nearby_players else "untouched"

        # Start a new period when touch status changes.
        if kind != period_kind:
            close_period()
            period_kind = kind

        period_frames.append(frame)

        for player in nearby_players:
            period_players[player.player_id] = player

    close_period()

    return touches, untouched_periods
