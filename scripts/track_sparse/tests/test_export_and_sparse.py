from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from track_sparse.export import write_frame_exports, write_tracks_h5
from track_sparse.schema import Camera
from track_sparse.sparse_verify import SparseCloudVerifier


def _camera() -> Camera:
    K = np.eye(3)
    R = np.eye(3)
    t = np.zeros(3)
    return Camera(0, 7, 9, 10, 10, K, R, t, K @ np.column_stack((R, t)), np.zeros(3), "cam000frame001.png")


class ExportAndSparseTests(unittest.TestCase):
    def test_empty_frame_exports_are_still_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            count = write_frame_exports({}, root / "frames", True, True, [1, 2])
            self.assertEqual(count, 2)
            with np.load(root / "frames" / "frame001_track_points.npz") as data:
                self.assertEqual(data["xyz"].shape, (0, 3))
            write_tracks_h5({}, {0: _camera()}, root / "tracks.h5")
            with h5py.File(root / "tracks.h5", "r") as handle:
                self.assertEqual(handle.attrs["schema_version"], "2.0.0")
                self.assertEqual(handle["observations/uv"].shape, (0, 2))
                self.assertEqual(handle["samples3d/xyz"].shape, (0, 3))
                self.assertEqual(handle["samples3d/raw_xyz"].shape, (0, 3))
                self.assertEqual(handle["samples3d/covariance"].shape, (0, 3, 3))
                self.assertEqual(handle["camera_corrections/R_w2c"].shape, (0, 3, 3))

    def test_sparse_cloud_support_distance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame001_points3D.txt"
            path.write_text(
                "1 0 0 5 255 0 0 0\n2 1 0 5 0 255 0 0\n3 0 1 5 0 0 255 0\n",
                encoding="utf-8",
            )
            verifier = SparseCloudVerifier({1: path}, nn_scale=3.0)
            distance, confidence = verifier.score(1, np.array([0.0, 0.0, 5.0]))
            self.assertAlmostEqual(distance, 0.0)
            self.assertAlmostEqual(confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
