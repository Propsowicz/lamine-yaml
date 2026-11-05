from pathlib import Path
from data_prepare.detector import PlayersDetector, YOLOConfig
from data_prepare.atg_normalizer import normalize_atg_file
from models.frame import Frame, DetectedFrame
from utils.chart_helper import plot_coordinates, draw_frame
from data_prepare.data_sets_combiner import Combiner
from models.player import Player
from data_prepare.player_tracker_validator import validate_player_tracking

ROOT_PATH = Path(__file__).resolve().parent.parent
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = str(ROOT_PATH / "data")
HOMOGRAPHY_PATH = str(BASE_DIR / ".." / "data" / "tactical" / "BAR-ATH_tactical_homography.csv")
VIDEO_PATH = str(BASE_DIR / ".." / "data" / "tactical" / "BAR-ATH_tactical-001.mp4")


class PosessionCalc:
    def __init__(self, start_frame: int, end_frame: int):
        self.start_frame = start_frame
        self.end_frame = end_frame        

    def calculate(self):
        print("start calculations")
        frames = self._normalize_atg_data()
        print("frames loaded")
        plot_coordinates([f.ball.x for f in frames], [f.ball.y for f in frames], [f.id for f in frames])

    def _detect_players_position(self) -> list[DetectedFrame]:
        detection_config = YOLOConfig(
            imgsz=1568,
            conf=0.1,
            iou=0.7,
            max_det=300
        )

        detector = PlayersDetector(VIDEO_PATH, HOMOGRAPHY_PATH, detection_config)
        frames = detector.detect(self.start_frame, self.end_frame)
        return frames

    def _normalize_atg_data(self) -> list[Frame]:
        return normalize_atg_file(data_path=DATA_PATH, start_frame=self.start_frame, end_frame=self.end_frame)
        
if __name__ == '__main__':
    calc = PosessionCalc(start_frame=11400, end_frame=11405)
    # calc = PosessionCalc(start_frame=11287, end_frame=11410)

    detected_frames = calc._detect_players_position()
    frames = calc._normalize_atg_data()
    print(len(detected_frames))
    print(len(frames))
    indeksik = 0
    for f, df in zip(frames, detected_frames):
        if indeksik > 0:
            frames[indeksik].players = validate_player_tracking(
                current_players=frames[indeksik].players,
                previous_players=frames[indeksik-1].players,
                merge_distance=2.0,
                reappearance_distance=2.0,
            )

        draw_frame(frames[indeksik].players, f.id, timestamp=None, save=True, show_labels=True, save_name_prefix="from_atg")


        if f.id == 11186 or f.id == 11425:
            i = f.id
            for l in df.players:
                print(f"{l.team} - {l.team_confidence}")
            draw_frame(f.players, i, timestamp=None, save=True, save_name_prefix="from_atg")
            draw_frame(df.players, i, timestamp=None, save=True, save_name_prefix="from_detection")

            combiner = Combiner(
                max_match_distance=3.0,
                team_confidence_threshold=0.90,
            )

            combined_players: list[Player] = combiner.combine(
                players=f.players,
                detected_players=df.players,
                ball=f.ball,
                players_by_team=f.players_by_team,
            )
            draw_frame(combined_players, i, timestamp=None, save=True, save_name_prefix="from_combination")
        indeksik += 1
