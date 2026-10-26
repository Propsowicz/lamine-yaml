import pandas as pd
from pathlib import Path
from models import Ball, Player, Frame


def save_frames(frames: list[Frame], path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    frame_rows = []
    player_rows = []

    for frame in frames:
        # Save every frame, even when it has no players.
        frame_rows.append({
            "frame_id": frame.id,
            "timestamp": frame.timestamp,
            "ball_x": frame.ball.x,
            "ball_y": frame.ball.y,
            "ball_z": frame.ball.z,
        })

        # Save every player belonging to this frame.
        for player in frame.players:
            player_rows.append({
                "frame_id": frame.id,
                "player_id": player.player_id,
                "team": player.team,
                "x": player.x,
                "y": player.y,
                "speed": player.speed,
                "dist_to_ball": player.dist_to_ball,
            })

    pd.DataFrame(
        frame_rows,
        columns=[
            "frame_id",
            "timestamp",
            "ball_x",
            "ball_y",
            "ball_z",
        ],
    ).to_csv(path / "frames.csv", index=False)

    pd.DataFrame(
        player_rows,
        columns=[
            "frame_id",
            "player_id",
            "team",
            "x",
            "y",
            "speed",
            "dist_to_ball",
        ],
    ).to_csv(path / "players.csv", index=False)


def read_frames(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(path)

    frames_df = pd.read_csv(path / "frames.csv")
    players_df = pd.read_csv(path / "players.csv")

    return frames_df, players_df
