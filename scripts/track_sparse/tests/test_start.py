from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import yaml

from track_sparse.config import load_config
from track_sparse.start import PROJECT_ROOT, build_command


class StartConfigTests(unittest.TestCase):
    def test_builds_run_command_and_keeps_run_out_of_algorithm_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "scene.yaml"
            config_path.write_text(
                yaml.safe_dump({
                    "run": {
                        "python": "/opt/4dgs-player/env/bin/python3.11",
                        "input": "data/pubulicdata/ccoffee",
                        "output": "outputs/ccoffee_tracks",
                        "start_frame": 1,
                        "end_frame": 300,
                        "debug_frames": [1, 150, 300],
                        "gpu": 0,
                        "resume": True,
                    },
                    "features": {"max_keypoints": 2048},
                }),
                encoding="utf-8",
            )
            command = build_command(config_path)
            self.assertEqual(command[0], "/opt/4dgs-player/env/bin/python3.11")
            self.assertIn(str(PROJECT_ROOT / "data/pubulicdata/ccoffee"), command)
            self.assertIn(str(PROJECT_ROOT / "outputs/ccoffee_tracks"), command)
            self.assertIn("1,150,300", command)
            self.assertIn("--resume", command)
            algorithm = load_config(config_path)
            self.assertNotIn("run", algorithm)
            self.assertEqual(algorithm["features"]["max_keypoints"], 2048)

    def test_rejects_conflicting_output_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "scene.yaml"
            config_path.write_text(
                yaml.safe_dump({
                    "run": {
                        "python": "/opt/4dgs-player/env/bin/python3.11",
                        "input": "data/pubulicdata/ccoffee",
                        "output": "outputs/ccoffee_tracks",
                        "resume": True,
                        "overwrite": True,
                    }
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "不能同时"):
                build_command(config_path)


if __name__ == "__main__":
    unittest.main()
