from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from track_sparse.input.preprocess import PreprocessOptions, detect_input_type, prepare_input


class PreprocessTests(unittest.TestCase):
    def _make_raw_images(self, root: Path) -> None:
        for cam_id in (0, 1):
            cam_dir = root / f"cam{cam_id:02d}"
            cam_dir.mkdir(parents=True)
            for frame_id in (1, 2):
                image = np.full((24, 32, 3), 30 + cam_id * 50 + frame_id, np.uint8)
                self.assertTrue(cv2.imwrite(str(cam_dir / f"frame{frame_id:03d}.png"), image))

    def _make_calibration(self, path: Path) -> None:
        frames = []
        for cam_id in (0, 1):
            w2c = np.eye(4)
            w2c[0, 3] = -float(cam_id)
            frames.append({
                "filename": f"cam{cam_id:03d}.png",
                "camera_id": cam_id,
                "camera_model": "OPENCV",
                "image_size": [32, 24],
                "intrinsic": [30.0, 0.0, 16.0, 0.0, 30.0, 12.0, 0.0, 0.0, 1.0],
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
                "w2c": w2c.reshape(-1).tolist(),
            })
        path.write_text(json.dumps({"frames": frames}), encoding="utf-8")

    def test_known_calibration_preprocess_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            self._make_raw_images(source)
            calibration = root / "calibration.json"
            self._make_calibration(calibration)
            work = root / "output" / "preprocess"
            base = dict(
                input_dir=source,
                work_dir=work,
                input_type="auto",
                python=Path(sys.executable),
                gpu_ids=[0],
                max_frames=0,
                frames_per_second=None,
                rig_json=None,
                calib_json=calibration,
                undistort_workers=2,
                overwrite=False,
                dry_run=False,
            )
            first = prepare_input(PreprocessOptions(resume=False, **base))
            self.assertEqual(detect_input_type(source), "images")
            self.assertFalse(first.reused)
            self.assertTrue((first.images_dir / "cam000" / "cam000frame001.png").is_file())
            self.assertTrue((first.sparse_dir / "cameras.txt").is_file())
            self.assertEqual(first.metadata["camera_count"], 2)
            self.assertEqual(first.metadata["frame_count"], 2)

            second = prepare_input(PreprocessOptions(resume=True, **base))
            self.assertTrue(second.reused)
            self.assertEqual(first.tracking_source(), second.tracking_source())

            shutil.rmtree(second.images_dir / "cam001")
            repaired = prepare_input(PreprocessOptions(resume=True, **base))
            self.assertFalse(repaired.reused)
            self.assertEqual(repaired.metadata["camera_count"], 2)
            self.assertTrue((repaired.images_dir / "cam001" / "cam001frame002.png").is_file())

    def test_prepared_dataset_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            sparse = root / "sparse" / "0"
            sparse.mkdir(parents=True)
            (sparse / "cameras.txt").write_text("", encoding="utf-8")
            (sparse / "images.txt").write_text("", encoding="utf-8")
            self.assertEqual(detect_input_type(root), "prepared")


if __name__ == "__main__":
    unittest.main()
