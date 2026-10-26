from pathlib import Path
from data_prepare.detector import PlayersDetector

BASE_DIR = Path(__file__).resolve().parent
FRAME_PICS = str(BASE_DIR / ".." / "data" / "tactical" / "frame_pics" / "frame_1.jpg")

detector = PlayersDetector(FRAME_PICS)
result = detector.detect()
for p in result:
    print(f"{p.team}, {p.x}, {p.y}")