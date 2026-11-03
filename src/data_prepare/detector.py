
from pathlib import Path
import supervision as sv
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from models.player import DetectedPlayer
from models.frame import DetectedFrame
import cv2
import pandas as pd
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from collections import Counter

GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
BASE_DIR = Path(__file__).resolve().parent

@dataclass(frozen=True)
class YOLOConfig:
    imgsz: int = 1568
    conf: float = 0.1
    iou: float = 0.7
    max_det: int = 300

class PlayersDetector:
    def __init__(self, video_path: str, homography_path: str, config: YOLOConfig = YOLOConfig()) -> None:
        self.player_model = YOLO(
            str(BASE_DIR / "detection_models" / "football-player-detection.pt")
        )
        self.homography_df = pd.read_csv(homography_path)
        self.video_path = video_path
        self.config = config

    def load_homography(self, frame_id) -> np.array:
        row = self.homography_df.iloc[frame_id]  

        if not row["valid"]:
            return None

        homography_per_frame = row[
            ["h00", "h01", "h02",
            "h10", "h11", "h12",
            "h20", "h21", "h22"]
        ].to_numpy(dtype=np.float64).reshape(3, 3)

        return np.linalg.inv(homography_per_frame)

    def detect(self, start_frame: int = 11185, end_frame: int = 11189) -> list[DetectedFrame]:
        cap = cv2.VideoCapture(self.video_path)
        if cap is None:
            raise FileNotFoundError(f"Cannot read {self.image_path}")

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            result: list[DetectedFrame] = []

            for frame_index in range(start_frame, end_frame):
                success, frame = cap.read()
                if not success:
                    print(f"cannot read data for frame: {frame_index}")
                    break

                player_result = self.player_model.predict(
                    frame,
                    imgsz=self.config.imgsz,
                    conf=self.config.conf,
                    iou=self.config.iou,
                    max_det=self.config.max_det,
                    verbose=False,
                )[0]
                homography = self.load_homography(frame_index)

                players: list[DetectedPlayer] = []
                
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
                        DetectedPlayer(
                            x=pitch_x,
                            y=pitch_y,                            
                            player_team="tbd",
                            box=box
                        )
                    )
        
                if not players:
                    raise RuntimeError(
                        "No players detected. Try lowering the confidence threshold."
                    )

                result.append(DetectedFrame(
                    frame_id=frame_index,
                    players=self.assign_players_to_teams(frame, players)
                ))
        except:
            raise "cannot detect players"
        finally:
            cap.release()      
        return result
        
    # TODO: this method is crucial to recognize teams BUT we need more than two teams
    def assign_players_to_teams(
        self,
        frame: np.ndarray,
        players: list[DetectedPlayer],
    ) -> list[DetectedPlayer]:
        if not players:
            return players

        features = np.stack([
            self._get_shirt_color(frame, player.box)
            for player in players
        ])

        if len(players) == 1:
            players[0].team = "Single 1"
            players[0].team_confidence = 0.5
            return players

        labels = DBSCAN(
            #TODO: to determine
            eps=26,
            min_samples=1,
            metric="euclidean",
        ).fit_predict(features)

        cluster_counts = Counter(labels)

        ordered_clusters = [
            cluster
            for cluster, _ in cluster_counts.most_common()
        ]

        cluster_names = {}

        for index, cluster in enumerate(ordered_clusters):
            if index < 2:
                cluster_names[cluster] = f"Team {index + 1}"
            elif index < 5:
                cluster_names[cluster] = f"Single {index - 1}"
            else:
                cluster_names[cluster] = None

        # Calculate representative color for every cluster.
        cluster_centers = {
            cluster: np.median(
                features[labels == cluster],
                axis=0,
            )
            for cluster in ordered_clusters
        }

        for player, color, label in zip(players, features, labels):
            player.team = cluster_names[label]

            center = cluster_centers[label]
            distance = float(np.linalg.norm(color - center))

            # Distance-based confidence.
            player.team_confidence = float(
                np.exp(-distance / 15.0)
            )

            # A single observation provides weaker evidence.
            if cluster_counts[label] == 1:
                player.team_confidence *= 0.5

        return players

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
