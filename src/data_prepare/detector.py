
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


    def detect(
        self,
        start_frame: int = 11185,
        end_frame: int = 11189,
        frame_interval: int = 25,
    ) -> list[DetectedFrame]:
        if frame_interval <= 0:
            raise ValueError("frame_interval must be greater than 0")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot read {self.video_path}")

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            result: list[DetectedFrame] = []
            previous_players: list[DetectedPlayer] | None = None

            for frame_index in range(start_frame, end_frame):
                success, frame = cap.read()
                if not success:
                    print(f"Cannot read data for frame: {frame_index}")
                    break

                # Run detection on every Nth frame.
                should_detect = (
                    previous_players is None
                    or (frame_index - start_frame) % frame_interval == 0
                )

                if should_detect:
                    print(f"Detecting players in frame {frame_index}...")

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

                        if class_id not in {
                            PLAYER_CLASS_ID,
                            GOALKEEPER_CLASS_ID,
                        }:
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
                                box=box,
                            )
                        )

                    if not players:
                        raise RuntimeError(
                            f"No players detected in frame {frame_index}. "
                            "Try lowering the confidence threshold."
                        )

                    previous_players = self.assign_players_to_teams(
                        frame,
                        players,
                    )

                # Reuse players from the last detected frame.
                result.append(
                    DetectedFrame(
                        frame_id=frame_index,
                        players=previous_players.copy(),
                    )
                )

            return result

        except Exception as exc:
            raise RuntimeError("Cannot detect players") from exc

        finally:
            cap.release()
        
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

        valid = ~np.isnan(features).any(axis=1)
    
        if valid.sum() < 4:
            for player in players:
                player.team, player.team_confidence = None, 0.0
            return players
    
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(features[valid])
        centers = km.cluster_centers_

        dists = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
        nearest = dists.argmin(axis=1)
        d_near = dists.min(axis=1)
        d_other = dists.max(axis=1)
    
        med = np.median(d_near[valid])
        mad = np.median(np.abs(d_near[valid] - med)) + 1e-6
        outlier_thr = med + 3.5 * 1.4826 * mad
    
        for index, player in enumerate(players):
            if not valid[index] or d_near[index] > outlier_thr:
                player.team = "Other"       
                player.team_confidence = 0.0
                continue
            
            player.team = f"Team {nearest[index] + 1}"
            player.team_confidence = float(1 - d_near[index] / (d_other[index] + 1e-6))
        return players

    def _get_shirt_color(self, image: np.ndarray, box: tuple) -> np.ndarray:
        x1, y1, x2, y2 = map(int, box)
        width, height = x2 - x1, y2 - y1
    
        left   = max(0, x1 + int(width * 0.25))
        right  = min(image.shape[1], x2 - int(width * 0.25))
        top    = max(0, y1 + int(height * 0.15))
        bottom = min(image.shape[0], y1 + int(height * 0.50))
    
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            return np.full(3, np.nan, dtype=np.float32)
    
        # Grass mask
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        grass = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        keep = grass == 0
    
        if keep.sum() < 0.15 * keep.size:
            keep = np.ones_like(keep, dtype=bool)
    
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        pixels = lab[keep].astype(np.float32)
        return np.median(pixels, axis=0)

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
