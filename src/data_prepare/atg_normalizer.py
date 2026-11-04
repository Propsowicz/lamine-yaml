
import os
import sys
import math
from data_prepare.games import require, asset_paths

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FIELD_SEP = ":"

GAME = "BAR-ATH"
DEFAULT_SOURCE = "tactical"

def atd_paths(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """(meta_data, raw_data) paths, or a readable error naming what is missing."""
    return tuple(require(["atd_meta", "atd_raw"], game, data_dir, source))


def load(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """The ATD as a kloppy TrackingDataset.

    Coordinates come back in kloppy's Metrica convention. The real pitch is
    GAMES[game]["pitch"] metres, which is what you want for anything measured in metres rather
    than in fractions of a pitch.
    """
    try:
        from kloppy import metrica
    except ImportError:
        raise ImportError(
            "kloppy is not installed. It is optional -- the evaluation needs only numpy and "
            "pandas -- but this loader needs it:\n    pip install -r requirements-starter.txt")

    meta, raw = atd_paths(game, data_dir, source)
    return metrica.load_tracking_epts(meta_data=meta, raw_data=raw)


def video_frame_of(frame_id):
    """Video frame -- and therefore submission row -- of an ATD record."""
    return int(frame_id) - 1


def frame_id_of(video_frame):
    """ATD frame_id of a video frame. The inverse of video_frame_of."""
    return int(video_frame) + 1

def check_alignment_raw(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """The same check as check_alignment, straight off the raw file. No kloppy, no full load.

    Loading 199319 frames x 46 slots through kloppy to count them costs minutes and gigabytes;
    the raw file answers the same question in one streaming pass, because every line begins with
    its own frame id. Use this one unless you already have a dataset in hand.
    """
    raw_path = require(["atd_raw"], game, data_dir, source)[0]
    expected = source_frames(source, game)

    first = last = None
    count = 0
    with open(raw_path) as handle:
        for line in handle:
            frame_id = int(line[:line.index(FIELD_SEP)])
            if first is None:
                first = frame_id
            last = frame_id
            count += 1

    contiguous = count > 0 and last - first + 1 == count
    return {
        "n_records": count,
        "n_video_frames": expected,
        "frame_id_range": (first, last),
        "contiguous": contiguous,
        "first_video_frame": video_frame_of(first) if first is not None else None,
        "agrees": count == expected and contiguous and first == 1,
    }


def check_alignment(dataset, game=GAME, source=DEFAULT_SOURCE):
    """Confirm the ATD really is one record per video frame, numbered from 1.

    Cheap, and worth running once against any new export rather than trusting the note above.
    Two claims, both falsifiable: the record count equals the video length, and the ids are
    exactly 1..n with nothing missing.
    """
    expected = source_frames(source, game)
    ids = sorted(int(record.frame_id) for record in dataset.records)
    contiguous = bool(ids) and ids == list(range(ids[0], ids[0] + len(ids)))
    return {
        "n_records": len(ids),
        "n_video_frames": expected,
        "frame_id_range": (ids[0], ids[-1]) if ids else (None, None),
        "contiguous": contiguous,
        "first_video_frame": video_frame_of(ids[0]) if ids else None,
        "agrees": len(ids) == expected and contiguous and bool(ids) and ids[0] == 1,
    }

import csv
from pathlib import Path

from models.ball import Ball
from models.player import Player
from models.frame import Frame

def normalize_atg_file(data_path: str, start_frame: int = 1, end_frame: int = 199319) -> list[Frame]:
    dataset = load(data_dir=data_path)
    players_by_team = {
            team.name: [player.player_id for player in team.players]
            for team in dataset.metadata.teams
        }
    frames: list[Frame] = []

    for frame in dataset.frames[start_frame:end_frame]:
        ball = Ball(frame.ball_coordinates.x, frame.ball_coordinates.y, frame.ball_coordinates.z)

        players = [
            Player(
                player_id=player.player_id,
                x=player_data.coordinates.x,
                y=player_data.coordinates.y,
                speed=player_data.speed,
                players_by_team=players_by_team,
                ball = ball
            )
            for player, player_data in frame.players_data.items()
        ]

        players.sort(
            key=lambda p: (
                p.dist_to_ball is None or math.isnan(p.dist_to_ball),
                p.dist_to_ball if p.dist_to_ball is not None and not math.isnan(p.dist_to_ball) else 0,
            )
        )
        frames.append(Frame(frame.frame_id, frame.timestamp, players, ball, players_by_team=players_by_team))
    return frames
