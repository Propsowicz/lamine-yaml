


import pandas as pd
from pathlib import Path
from models import Ball, Player, Frame
from utils import read_frames
from touches_tracker import collect_touch_periods
from enum import Enum
from typing import Any
from drawer import draw_frame, detect_player_coordinate_anomalies, plot_player_track
import math
import pandas as pd

MIN_STATIONARY_SECONDS = 0.1
BALL_POSITION_TOLERANCE = 0.01
MIN_STATIONARY_FRAMES = 15

PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0

from typing import TypedDict


class StationaryPeriod(TypedDict):
    start_frame_id: int
    end_frame_id: int
    start_timestamp: float
    end_timestamp: float
    ball_x: float
    ball_y: float

def positions_differ(
    current: float | None,
    previous: float | None,
    tolerance: float,
) -> bool:
    if current is None or previous is None:
        return current != previous

    return abs(current - previous) > tolerance

def collect_stationary_ball_periods(
    frames: list[Frame],
    min_stationary_frames: int = MIN_STATIONARY_FRAMES,
    ball_position_tolerance: float = BALL_POSITION_TOLERANCE,
) -> list[StationaryPeriod]:
    if len(frames) < min_stationary_frames:
        return []

    periods: list[StationaryPeriod] = []
    start_index = 0

    for index in range(1, len(frames)):
        previous = frames[index - 1]
        current = frames[index]

        moved = (
            positions_differ(
                current.ball.x,
                previous.ball.x,
                ball_position_tolerance,
            )
            or positions_differ(
                current.ball.y,
                previous.ball.y,
                ball_position_tolerance,
            )
        )

        if not moved:
            continue

        end_index = index - 1
        frame_count = end_index - start_index + 1

        if frame_count >= min_stationary_frames:
            start_frame = frames[start_index]
            end_frame = frames[end_index]

            periods.append(
                {
                    "start_frame_id": start_frame.id,
                    "end_frame_id": end_frame.id,
                    "start_timestamp": start_frame.timestamp,
                    "end_timestamp": end_frame.timestamp,
                    "ball_x": start_frame.ball.x,
                    "ball_y": start_frame.ball.y,
                }
            )

        start_index = index

    # Handle the final stationary period.
    end_index = len(frames) - 1
    frame_count = end_index - start_index + 1

    if frame_count >= min_stationary_frames:
        start_frame = frames[start_index]
        end_frame = frames[end_index]

        periods.append(
            {
                "start_frame_id": start_frame.id,
                "end_frame_id": end_frame.id,
                "start_timestamp": start_frame.timestamp,
                "end_timestamp": end_frame.timestamp,
                "ball_x": start_frame.ball.x,
                "ball_y": start_frame.ball.y,
            }
        )

    return periods

def read_frames1(path: str | Path) -> list[Frame]:
    path = Path(path)

    frames_df = pd.read_csv(path / "frames.csv")
    players_df = pd.read_csv(path / "players.csv")
    frames_df["timestamp"] = pd.to_timedelta(frames_df["timestamp"])    

    players_by_frame: dict[int, list[Player]] = {}

    # Reconstruct players without recalculating their saved values.
    for row in players_df.to_dict(orient="records"):
        player = Player.__new__(Player)
        player.player_id = row["player_id"]
        player.team = row["team"]
        player.x = row["x"]
        player.y = row["y"]
        player.speed = row["speed"]
        player.dist_to_ball = row["dist_to_ball"]

        players_by_frame.setdefault(row["frame_id"], []).append(player)

    frames: list[Frame] = []

    for row in frames_df.to_dict(orient="records"):
        ball = Ball.__new__(Ball)
        ball.x = row["ball_x"]
        ball.y = row["ball_y"]
        ball.z = row["ball_z"]

        frame = Frame.__new__(Frame)
        frame.id = row["frame_id"]
        frame.timestamp = row["timestamp"]
        frame.ball = ball
        frame.players = players_by_frame.get(row["frame_id"], [])

        frames.append(frame)

    return frames

def check_ball_out_of_pitch(frames: list[Frame]):
    for f in frames:
            if not math.isnan(f.ball.z):
                f.repr()
    
            if f.ball.x > PITCH_LENGTH or f.ball.x < 0:
                f.repr()
            
            if f.ball.y > PITCH_WIDTH or f.ball.y < 0:
                f.repr()



def mock_frame(frame: Frame) -> Frame:
    mock_players = {
    1: ("Team 2", 51.19747241546176, 50.997117440825086),
    2: ("Team 1", 50.10454683314889, 40.41612604583447),
    3: ("Team 1", 52.66115850690693, 39.854280940505845),
    4: ("Team 2", 64.99700797243648, 53.3643042435743),
    5: ("Team 1", 48.67109080132336, 48.40648686871555),
    6: ("Team 2", 67.685374888684, 36.021529599065616),
    7: ("Team 2", 63.55694875712307, 13.40940132280987),
    8: ("Team 2", 68.80199790482646, 25.428031168029012),
    9: ("Team 2", 50.463039983629834, 28.32364839047699),
    10: ("Team 2", 56.996612308824204, 39.5718174274465),
    11: ("Team 1", 37.294906672122174, 46.86380697830209),
    12: ("Team 1", 35.456425692607816, 35.64495197953493),
    13: ("Team 2", 50.129971894093664, 30.548962091725677),
    14: ("Team 1", 42.226349794578695, 26.54937140402781),
    15: ("Team 2", 59.79838615264252, 26.40753689518521),
    16: ("Team 1", 33.10259078724857, 20.192015537859177),
    17: ("Team 1", 37.54382438809661, 30.625551737148864),
    18: ("Team 1", 37.7582963875256, 13.265832723140644),
    19: ("Team 2", 50.189888237531015, 13.257027708504989),
    20: ("Team 1", 49.8189545177838, 19.776040855382764),
    21: ("Team 1", 49.79573239978892, 15.519584979435962),
}
    
    players_by_id = {player.player_id: player for player in frame.players}

    mocked_players = []

    for player_id, (team, x, y) in mock_players.items():
        original = players_by_id.get(player_id)

        if original is None:
            continue

        player = Player(
            player_id=player_id,
            x=x / PITCH_LENGTH,
            y=y / PITCH_WIDTH,
            speed=original.speed,
            players_by_team={},
            ball=frame.ball,
            player_team=team
        )

        # Override the team because the constructor normally assigns it.
        player.team = team
        mocked_players.append(player)

    return Frame(
        id=frame.id,
        timestamp=frame.timestamp,
        players=mocked_players,
        ball=frame.ball,
    )


if __name__ == '__main__':
    frames = read_frames1(".")
    # frames_df["timestamp"] = pd.to_timedelta(frames_df["timestamp"])    


    # plot_player_track(
    #     frames=frames,
    #     player_id=45,
    #     max_step_distance=10.0,
    # )

    # anomalies = detect_player_coordinate_anomalies(
    #     frames=frames,
    #     player_id=18,
    # )
    # print(anomalies)

    add = 0
    for f in frames:
        # if add == 25:
        #     add = 0
        # if add == 0:
        #     draw_frame(f, show_labels=True, save=False)
        # add += 1
        if f.id == 11186:
            draw_frame(mock_frame(f), show_labels=True)
    
    stationary_ball_periods = collect_stationary_ball_periods(frames)

    # for b in stationary_ball_periods:
    #     print(dict(b))

    touches, untouched_periods = collect_touch_periods(frames)

    for touch in touches:
        print(touch)

    # for period in untouched_periods:
    #     print(period)
        