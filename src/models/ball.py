from utils.constants import PITCH_LENGTH, PITCH_WIDTH

class Ball:
    def __init__(self, x, y, z):
        self.x: float = x*PITCH_LENGTH
        self.y: float = y*PITCH_WIDTH
        self.z: float = z