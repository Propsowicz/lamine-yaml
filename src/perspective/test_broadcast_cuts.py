import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from broadcast_cuts import detect, load_segments, map_time, metadata


class PerspectiveTests(unittest.TestCase):
    def test_bounded_alignment_and_recording_gap(self):
        segments = [{"anchors": [
            {"broadcast_seconds": 10, "tactical_seconds": 100},
            {"broadcast_seconds": 20, "tactical_seconds": 120}]},
            {"anchors": [
                {"broadcast_seconds": 30, "tactical_seconds": 200},
                {"broadcast_seconds": 40, "tactical_seconds": 210}]}]
        self.assertEqual(map_time(15, segments), (110, 0))
        self.assertEqual(map_time(35, segments), (205, 1))
        for seconds in (0, 25, 41):
            self.assertEqual(map_time(seconds, segments), (None, None))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchors.json"
            path.write_text(json.dumps({"segments": segments}))
            self.assertEqual(load_segments(path), segments)
            segments[1]["anchors"][0]["tactical_seconds"] = 110
            path.write_text(json.dumps({"segments": segments}))
            with self.assertRaises(ValueError):
                load_segments(path)

    def test_known_cut_and_static_shots(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cuts.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (160, 90))
            self.assertTrue(writer.isOpened())
            try:
                for color in [(0, 120, 0)] * 25 + [(0, 0, 220)] * 25:
                    writer.write(np.full((90, 160, 3), color, dtype=np.uint8))
            finally:
                writer.release()
            cuts, decoded = detect(path, metadata(path), 0.22, 0.5, 0, None)
            self.assertEqual(decoded, 50)
            self.assertEqual([row["broadcast_frame"] for row in cuts], [25])
            self.assertEqual(cuts[0]["broadcast_seconds"], 1)
            cuts, decoded = detect(path, metadata(path), 0.22, 0.5, 1.2, 0.4)
            self.assertEqual(cuts, [])
            self.assertEqual(decoded, 10)


if __name__ == "__main__":
    unittest.main()
