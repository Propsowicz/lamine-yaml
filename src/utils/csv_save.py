import csv
from models.player import DetectedPlayer
from models.frame import DetectedFrame 
from data_analyze.touch_ball_events_analyzer import FrameLabel
from utils.constants import PITCH_WIDTH, PITCH_LENGTH
from pathlib import Path

def save_frames(frames: list[DetectedFrame], file_path: str) -> None:
    with open(file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "frame_id",
            "x",
            "y",
            "team",
            "box",
            "team_confidence",
        ])

        for frame in frames:
            for player in frame.players:
                writer.writerow([
                    frame.id,
                    player.x,
                    player.y,
                    player.team,
                    player.box,
                    player.team_confidence,
                ])


def read_frames(file_path: str) -> list[DetectedFrame]:
    frames_by_id: dict[int, DetectedFrame] = {}

    with open(file_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            frame_id = int(row["frame_id"])

            if frame_id not in frames_by_id:
                frames_by_id[frame_id] = DetectedFrame(frame_id, [])

            player = DetectedPlayer(
                x=float(row["x"]) / PITCH_LENGTH,
                y=float(row["y"]) / PITCH_WIDTH,
                player_team=row["team"],
                box=tuple(map(float, row["box"].strip("()").split(", "))),
            )

            player.team_confidence = float(row["team_confidence"])

            frames_by_id[frame_id].players.append(player)

    return list(frames_by_id.values())

def csv_exists(file_path: str) -> bool:
    return Path(file_path).is_file()

from collections.abc import Sequence
from pathlib import Path


def save_frame_labels_to_csv(
    labels: Sequence[FrameLabel],
    output_path: str | Path,
) -> None:
    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(["frame", "label"])

        writer.writerows(
            (frame_id, label)
            for frame_id, label in enumerate(labels)
        )