from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from track_sparse.io_colmap import load_rig, qvec_to_rotmat, read_images_text


class ColmapIoTests(unittest.TestCase):
    def test_qvec_identity(self):
        np.testing.assert_allclose(qvec_to_rotmat(np.array([2.0, 0.0, 0.0, 0.0])), np.eye(3))

    def test_empty_and_missing_points_lines(self):
        text = """# header
7 1 0 0 0 0 0 0 3 cam009frame001.png

2 1 0 0 0 1 0 0 8 cam003frame001.png
5 6 -1
4 1 0 0 0 0 1 0 2 cam001frame001.png
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.txt"
            path.write_text(text, encoding="utf-8")
            images = read_images_text(path)
        self.assertEqual(set(images), {2, 4, 7})
        self.assertEqual(images[7].points2d.shape, (0, 3))
        self.assertEqual(images[2].points2d.shape, (1, 3))
        self.assertEqual(images[4].name, "cam001frame001.png")

    def test_mapping_does_not_assume_equal_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cameras.txt").write_text(
                "8 PINHOLE 100 80 50 51 49 39\n", encoding="utf-8"
            )
            (root / "images.txt").write_text(
                "2 1 0 0 0 1 0 0 8 cam003frame001.png\n\n", encoding="utf-8"
            )
            rig = load_rig(root)
        self.assertEqual(rig[3].image_id, 2)
        self.assertEqual(rig[3].camera_id, 8)
        np.testing.assert_allclose(rig[3].center_world, [-1, 0, 0])


if __name__ == "__main__":
    unittest.main()

