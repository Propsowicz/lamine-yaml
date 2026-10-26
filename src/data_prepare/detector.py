
from pathlib import Path
import supervision as sv
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from models.player import Player
from utils.constants import PITCH_LENGTH, PITCH_WIDTH

GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
BASE_DIR = Path(__file__).resolve().parent

# TODO: should be read - mock for now
HOMOGRAPHY = np.array(
    [
        [1800, -500, 45],
        [1.7, 170, 255],
        [-0.02, -0.5, 1.0],
    ],
    dtype=np.float64,
)

class PlayersDetector:
    def __init__(self, image_path: str) -> None:
        self.player_model = YOLO(
            str(BASE_DIR / "detection_models" / "football-player-detection.pt")
        )

        self.image_path = image_path
        self.image = cv2.imread(str(self.image_path))


    def detect(self) -> list[Player]:
        IMAGE_TO_PITCH = np.linalg.inv(HOMOGRAPHY)
        return self.assign_players_to_teams(
                        self.detect_players(IMAGE_TO_PITCH))


    def detect_players(self, homography: np.ndarray) -> list[Player]:
        if self.image is None:
            raise FileNotFoundError(f"Cannot read {self.image_path}")

        # TODO: experiment maybe with this?
        player_result = self.player_model.predict(
            self.image,
            imgsz=1568,
            conf=0.1,
            iou=0.7,
            max_det=300,
            verbose=False,
        )[0]

        players: list[Player] = []

        for detection in player_result.boxes:
            class_id = int(detection.cls.item())

            if class_id not in {PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID}:
                continue

            x1, y1, x2, y2 = map(
                int,
                detection.xyxy[0].cpu().numpy().round(),
            )

            box = (x1, y1, x2, y2)

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            pitch_x, pitch_y = self._image_to_pitch_meters(
                (center_x, center_y),
                homography,
            )

            players.append(
                Player(
                    player_id=999,
                    x=pitch_x,
                    y=pitch_y,
                    speed=999,
                    players_by_team={},
                    ball={},
                    player_team="tbd",
                    from_detection=True,
                    box=box
                )
            )

        if not players:
            raise RuntimeError(
                "No players detected. Try lowering the confidence threshold."
            )

        return players
        
    # TODO: this method is crucial to recognize teams BUT we need more than two teams
    def assign_players_to_teams(self, players: list[Player]) -> list[Player]:
        outfield = [
            p for p in players #if p.kind == "player"
        ]

        if len(outfield) >= 2:
            features = np.stack([
                self._get_shirt_color(self.image, p.box)
                for p in outfield
            ])

            labels = KMeans(
                n_clusters=2,
                random_state=0,
                n_init=10,
            ).fit_predict(features)

            for player, label in zip(outfield, labels):
                player.team = f"Team {label + 1}"
        return outfield

    def _get_shirt_color(self, image: np.ndarray, box: tuple) -> tuple:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        left = max(0, x1 + int(width * 0.25))
        right = min(image.shape[1], x2 - int(width * 0.25))
        top = max(0, y1 + int(height * 0.20))
        bottom = min(image.shape[0], y1 + int(height * 0.55))

        crop = image[top:bottom, left:right]

        if crop.size == 0:
            return np.zeros(3, dtype=np.float32)

        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        return np.median(lab.reshape(-1, 3), axis=0).astype(
            np.float32
        )

    def _image_to_pitch_meters(
        self,
        point: tuple[float, float],
        homography: np.ndarray,
    ) -> tuple[float, float]:
        image_point = np.asarray(
            [[point]],
            dtype=np.float64,
        )

        normalized_pitch_point = cv2.perspectiveTransform(
            image_point,
            homography,
        )[0, 0]

        x = float(normalized_pitch_point[0])
        y = float(normalized_pitch_point[1])

        return x, y
