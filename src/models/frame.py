from utils.constants import PITCH_LENGTH, PITCH_WIDTH
from models.player import Player, DetectedPlayer
from models.ball import Ball

class Frame:
    def __init__(self, id, timestamp, players: list[Player], ball: Ball) -> None:
        self.id = id
        self.timestamp = timestamp
        self.players = players
        self.ball = ball

    def repr(self) -> None:
        print(f"{self.id} | {self.timestamp} | BALL: {self.ball.x}:{self.ball.y}:{self.ball.z}")

class DetectedFrame:
    def __init__(self, frame_id, players: list[DetectedPlayer]):
        self.id = frame_id
        self.players = players