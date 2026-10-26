from utils.constants import PITCH_LENGTH, PITCH_WIDTH
import math
from .ball import Ball

class Player:
    def __init__(self, player_id, x, y, speed, players_by_team, ball: Ball, player_team: str = None, from_detection: bool = False, box: tuple = ()) -> None:
        self.player_id = player_id
        self.team: str = player_team if player_team else self._assign_team(player_id, players_by_team)
        self.x: float = x*PITCH_LENGTH
        self.y: float = y*PITCH_WIDTH
        self.speed: float = speed
        self.dist_to_ball: float = 0 if from_detection else self._calculate_distance_to_ball(x, y, ball.x, ball.y)
        self.box: tuple = box if from_detection else ()

    # TODO: needs to be calcualted differently
    def _assign_team(self, player_id: str | int, players_by_team: dict) -> str | None:
        for team_name, player_ids in players_by_team.items():
            if str(player_id) in map(str, player_ids):
                return team_name

        raise "no team for player"
        
    def _calculate_distance_to_ball(self, x, y, ball_x: float, ball_y: float) -> float:
        result = math.hypot(ball_x - x*PITCH_LENGTH, ball_y - y*PITCH_WIDTH)
        return result