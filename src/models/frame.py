from math import isfinite

from utils.constants import PITCH_LENGTH, PITCH_WIDTH
from models.player import Player, DetectedPlayer
from models.ball import Ball

class Frame:
    def __init__(self, id, timestamp, players: list[Player], ball: Ball, players_by_team: dict) -> None:
        self.id = id
        self.timestamp = timestamp
        self.players = players
        self.ball = ball
        self.players_by_team = players_by_team

    @property
    def average_speed(self) -> float:
        speeds = [
            player.speed for player in self.players
            if player.speed is not None and isfinite(player.speed)
        ]
        return sum(speeds) / len(speeds) if speeds else 0.0

    def repr(self) -> None:
        print(f"{self.id} | {self.timestamp} | BALL: {self.ball.x}:{self.ball.y}:{self.ball.z}")

class DetectedFrame:
    def __init__(self, frame_id, players: list[DetectedPlayer]):
        self.id = frame_id
        self.players = players
