from pathlib import Path
from data_prepare.detector import PlayersDetector, YOLOConfig
from data_prepare.atg_normalizer import normalize_atg_file
from models.frame import Frame, DetectedFrame
from utils.chart_helper import plot_coordinates, draw_frame
from data_prepare.data_sets_combiner import Combiner
from models.player import Player
from data_prepare.player_tracker_validator import validate_player_tracking
from utils.csv_save import save_frames, read_frames, csv_exists

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

    def load_frames(self) -> list[Frame]:
        print("load frames -- start")
        normalized_frames_from_atg: list[Frame] = normalize_atg_file(data_path=DATA_PATH, start_frame=self.start_frame, end_frame=self.end_frame)
        print("atg data loaded")
        
        frames_from_detection_file_name = f"./frames_from_detection_{self.start_frame}_{self.end_frame}.csv"
        if csv_exists(frames_from_detection_file_name):
            frames_from_detection = read_frames(frames_from_detection_file_name)
        else:
            detection_config = YOLOConfig(
                        imgsz=1568,
                        conf=0.1,
                        iou=0.7,
                        max_det=300
                    )
            
            detector = PlayersDetector(VIDEO_PATH, HOMOGRAPHY_PATH, detection_config)
            frames_from_detection = detector.detect(self.start_frame, self.end_frame)
            save_frames(frames_from_detection, frames_from_detection_file_name)
        print("pitch data detected")

        combiner = Combiner(
                    max_match_distance=3.0,
                    team_confidence_threshold=0.90,
                )
        result: list[Frame] = []
        for index, _ in enumerate(normalized_frames_from_atg):
            current_frame: Frame = normalized_frames_from_atg[index]
            previous_frame: Frame = normalized_frames_from_atg[index-1]

            if index > 0:
                normalized_frames_from_atg[index].players = validate_player_tracking(
                    current_players=current_frame.players,
                    previous_players=previous_frame.players,
                    max_tracking_distance=3.0,
                    merge_distance=2.0,
                    reappearance_distance=2.0,
                )
            #TODO need to fix - merging players is broken
            combined_players: list[Player] = combiner.combine(
                    players=normalized_frames_from_atg[index].players,
                    detected_players=frames_from_detection[index].players,
                    ball=current_frame.ball,
                    players_by_team=current_frame.players_by_team,
                )
            for p in combined_players:
                p.calculate_distance_to_ball(current_frame.ball)

            # TODO distance to ball needs to recalculated (remove it from init)
            result.append(Frame(
                current_frame.id,
                current_frame.timestamp,
                combined_players,
                current_frame.ball,
                current_frame.players_by_team
            ))
        return result

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
    # calc = PosessionCalc(start_frame=11400, end_frame=11405)
    calc = PosessionCalc(start_frame=11285, end_frame=11287)
    frames = calc.load_frames()
    for f in frames:
        draw_frame(f.players, f.id, timestamp=None, save=True, show_labels=True, save_name_prefix="frame_result")

   