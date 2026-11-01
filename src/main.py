from pathlib import Path
from data_prepare.detector import PlayersDetector, YOLOConfig
from data_prepare.atg_normalizer import normalize_atg_file
from models.frame import Frame
from utils.chart_helper import plot_coordinates

ROOT_PATH = Path(__file__).resolve().parent.parent
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = str(ROOT_PATH / "data")
HOMOGRAPHY_PATH = str(BASE_DIR / ".." / "data" / "tactical" / "BAR-ATH_tactical_homography.csv")
VIDEO_PATH = str(BASE_DIR / ".." / "data" / "tactical" / "BAR-ATH_tactical-001.mp4")


class PosessionCalc:
    def __init__(self):
        pass

    def calculate(self):
        print("start calculations")
        frames = self._normalize_atg_data()
        print("frames loaded")
        plot_coordinates([f.ball.x for f in frames], [f.ball.y for f in frames], [f.id for f in frames])

    def _detect_players_position(self):
        detection_config = YOLOConfig(
            imgsz=1568,
            conf=0.1,
            iou=0.7,
            max_det=300
        )
        detector = PlayersDetector(VIDEO_PATH, HOMOGRAPHY_PATH, detection_config)
        result = detector.detect()
        for p in result:
            print(f"{p.frame_id}")
            for pl in p.players:
                print(f"{pl.x}, {pl.y}, {pl.player_team}")

    def _normalize_atg_data(self) -> list[Frame]:
        start_frame = 11185
        trial_end = 18684
        return normalize_atg_file(data_path=DATA_PATH, start_frame=start_frame, end_frame=trial_end)
        
if __name__ == '__main__':
    calc = PosessionCalc()
    calc._detect_players_position()