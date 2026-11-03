from pathlib import Path
from data_prepare.detector import PlayersDetector, YOLOConfig
from data_prepare.atg_normalizer import normalize_atg_file
from models.frame import Frame, DetectedFrame
from utils.chart_helper import plot_coordinates, draw_frame

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

    def _detect_players_position(self) -> list[DetectedFrame]:
        detection_config = YOLOConfig(
            imgsz=1568,
            conf=0.1,
            iou=0.7,
            max_det=300
        )
        start_frame = 11185
        trial_end = 11190
        detector = PlayersDetector(VIDEO_PATH, HOMOGRAPHY_PATH, detection_config)
        frames = detector.detect(start_frame, trial_end)
        for f in frames:
            print(f"{f.id}")
            for p in f.players:
                print(f"{p.x}, {p.y}, {p.team}")
        return frames

    def _normalize_atg_data(self) -> list[Frame]:
        start_frame = 11185
        # trial_end = 18684
        trial_end = 11190
        return normalize_atg_file(data_path=DATA_PATH, start_frame=start_frame, end_frame=trial_end)
        
if __name__ == '__main__':
    calc = PosessionCalc()
    detected_frames = calc._detect_players_position()
    frames = calc._normalize_atg_data()
    print(len(detected_frames))
    print(len(frames))
    for f, df in zip(frames, detected_frames):
        if f.id == 11186:
            i = f.id
            for l in df.players:
                print(f"{l.team} - {l.team_confidence}")
            draw_frame(f.players, i, timestamp=None, save=True, save_name_prefix="from_atg")
            draw_frame(df.players, i, timestamp=None, save=True, save_name_prefix="from_detection")