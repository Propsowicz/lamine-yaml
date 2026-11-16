from data_prepare.ball_gaps_interpolation import interpolate_short_gaps
import os
import argparse
from data_prepare.games import GAME, GAMES
from pathlib import Path
from data_prepare.detector import PlayersDetector, YOLOConfig
from data_prepare.atd_normalizer import normalize_atd_file, load_atd_metadata
from models.frame import Frame
from data_prepare.data_sets_combiner import Combiner
from models.player import Player
from data_prepare.player_tracker_validator import validate_player_tracking
from utils.csv_save import save_frames, read_frames, csv_exists, save_frame_labels_to_csv
from data_prepare.touch_ball_tracker import detect_ball_touches, TouchType
from data_analyze.dead_ball_analyzer import detect_dead_ball_ranges
from data_analyze.touch_ball_events_analyzer import create_frame_labels
import logging
from data_analyze.ball_out_analyzer import detect_ball_out_events
from utils.constants import PITCH_LENGTH, PITCH_WIDTH
from data_prepare.smart_tagging import SmartTaggingCode, load_smart_tagging

from utils.setup_logging import setup_logging

from whistle_analysis.whistle_events import detect_whistle_events
from data_analyze.outside_game_time_analyzer import detect_outside_game_ranges

ROOT_PATH = Path(__file__).resolve().parent.parent
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = str(ROOT_PATH / "data")
HOMOGRAPHY_PATH = str(BASE_DIR / ".." / "data" / "tactical" / "BAR-ATH_tactical_homography.csv")
TACTICAL_FILE_NAME = os.environ.get("TACTICAL_FILE_NAME") or "BAR-ATH_tactical-001.mp4"
VIDEO_PATH = str(BASE_DIR / ".." / "data" / "tactical" / TACTICAL_FILE_NAME)


class PosessionCalc:
    def __init__(self, start_frame: int, end_frame: int):
        self.start_frame = start_frame
        self.end_frame = end_frame        
        setup_logging()
        self.logger = logging.getLogger(__name__)


    def load_frames(self) -> list[Frame]:
        self.logger.info("Load frames started.")
        normalized_frames_from_atd: list[Frame] = normalize_atd_file(data_path=DATA_PATH, start_frame=self.start_frame, end_frame=self.end_frame)
        interpolate_short_gaps(normalized_frames_from_atd)

        self.logger.info("Atd data loaded.")

        frames_from_detection_file_name = f"./frames_from_detection_{self.start_frame}_{self.end_frame}.csv"
        if csv_exists(frames_from_detection_file_name):
            self.logger.info("Loaded players detection results from pre-calculated file.")

            frames_from_detection = read_frames(frames_from_detection_file_name)
        else:
            self.logger.info("Started players detection.")
            detection_config = YOLOConfig(
                        imgsz=800,
                        conf=0.1,
                        iou=0.7,
                        max_det=300
                    )
            
            detector = PlayersDetector(VIDEO_PATH, HOMOGRAPHY_PATH, detection_config)
            frames_from_detection = detector.detect(self.start_frame, self.end_frame)
            save_frames(frames_from_detection, frames_from_detection_file_name)
            self.logger.info("Players detection finished.")
        

        combiner = Combiner(
                    max_match_distance=3.0,
                    team_confidence_threshold=0.90,
                )
        result: list[Frame] = []
        for index, _ in enumerate(normalized_frames_from_atd):
            current_frame: Frame = normalized_frames_from_atd[index]
            previous_frame: Frame = normalized_frames_from_atd[index-1]

            if index > 0:
                normalized_frames_from_atd[index].players = validate_player_tracking(
                    current_players=current_frame.players,
                    previous_players=previous_frame.players,
                    max_tracking_distance=3.0,
                    merge_distance=2.0,
                    reappearance_distance=2.0,
                )

            combined_players: list[Player] = combiner.combine(
                    players=normalized_frames_from_atd[index].players,
                    detected_players=frames_from_detection[index].players,
                    ball=current_frame.ball,
                    players_by_team=current_frame.players_by_team,
                )
            for p in combined_players:
                p.calculate_distance_to_ball(current_frame.ball)
            
            result.append(Frame(
                current_frame.id,
                current_frame.timestamp,
                combined_players,
                current_frame.ball,
                current_frame.players_by_team
            ))
        self.logger.info("Atd data combined with players detection and players tracking.")
        return result
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-name", type=str, default="CK", help="Name for the result file")
    args = parser.parse_args()

    atd_metadata = load_atd_metadata(data_dir=DATA_PATH)
    teams = atd_metadata['teams']

    START_FRAME = 0
    END_FRAME = GAMES[GAME]["n_video_frames"]
    
    calc = PosessionCalc(start_frame=START_FRAME, end_frame=END_FRAME)
    frames = calc.load_frames()
    
    st_events = load_smart_tagging(
        start_frame=START_FRAME,
        end_frame=END_FRAME,
        code=SmartTaggingCode.SET_PIECES,
    )   

    whistle_events = detect_whistle_events(
        start_frame=START_FRAME,
        end_frame=END_FRAME,
        video=VIDEO_PATH,
        game=GAME,
        threshold=0.01,
    )
    outside_game_ranges = detect_outside_game_ranges(st_events, whistle_events, end_frame=END_FRAME)
    calc = PosessionCalc(start_frame=START_FRAME, end_frame=END_FRAME)
    frames = calc.load_frames()

    touches = detect_ball_touches(frames, radius_m=1.5)

    ball_out_events = detect_ball_out_events(frames, PITCH_LENGTH, PITCH_WIDTH)
    dead_ball_ranges = detect_dead_ball_ranges(frames, st_events, touches)

    labels = create_frame_labels(
        END_FRAME,
        teams['home'],
        teams['away'],
        touches,
        dead_ball_ranges,
        ball_out_events,
        outside_game_ranges
    )
    save_frame_labels_to_csv(labels, f"{args.result_name}.csv")
   
