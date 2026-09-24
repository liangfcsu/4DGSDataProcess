from __future__ import annotations

from contextlib import redirect_stderr
import io
import tempfile
from pathlib import Path
import unittest

import yaml

from track_sparse.core.config import load_config
from track_sparse.run import DEFAULT_CONFIG_PATH, PROJECT_ROOT, parse_args


class RunConfigTests(unittest.TestCase):
    def test_default_config_uses_nested_video_directory(self):
        args = parse_args([])
        self.assertEqual(args.config, DEFAULT_CONFIG_PATH.resolve())
        self.assertEqual(args.input, PROJECT_ROOT / "data/coffee_martini/coffee_martini")
        self.assertEqual(args.output, PROJECT_ROOT / "outputs/coffee_martini_tracks")
        self.assertEqual(args.input_type, "videos")
        self.assertEqual(args.end_frame, 300)
        self.assertEqual(args.max_frames, 300)
        self.assertTrue(args.resume)
        self.assertNotIn("run", load_config())

    def test_cli_overrides_yaml_execution_settings(self):
        args = parse_args(["--input", "data/other", "--output", "outputs/other", "--overwrite"])
        self.assertEqual(args.input, "data/other")
        self.assertEqual(args.output, Path("outputs/other"))
        self.assertTrue(args.overwrite)
        self.assertFalse(args.resume)

    def test_custom_yaml_merges_algorithm_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "scene.yaml"
            config_path.write_text(yaml.safe_dump({
                "run": {
                    "input": "data/coffee_martini/coffee_martini",
                    "output": "outputs/example",
                    "debug_frames": [1, 150, 300],
                },
                "features": {"max_keypoints": 2048},
            }), encoding="utf-8")
            args = parse_args(["--config", str(config_path)])
            self.assertEqual(args.debug_frames, "1,150,300")
            algorithm = load_config(config_path)
            self.assertNotIn("run", algorithm)
            self.assertEqual(algorithm["features"]["max_keypoints"], 2048)

    def test_rejects_conflicting_output_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "scene.yaml"
            config_path.write_text(yaml.safe_dump({
                "run": {"input": "data/example", "output": "outputs/example",
                        "resume": True, "overwrite": True}
            }), encoding="utf-8")
            with redirect_stderr(io.StringIO()), self.assertRaisesRegex(SystemExit, "2"):
                parse_args(["--config", str(config_path)])


if __name__ == "__main__":
    unittest.main()
