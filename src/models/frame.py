from utils.constants import PITCH_LENGTH, PITCH_WIDTH
from player import Player
from ball import Ball

class Frame:
    def __init__(self, id, timestamp, players: list[Player], ball: Ball) -> None:
        self.id = id
        self.timestamp = timestamp
        self.players = players
        self.ball = ball

    def repr(self) -> None:
        print(f"{self.id} | {self.timestamp} | BALL: {self.ball.x}:{self.ball.y}:{self.ball.z}")