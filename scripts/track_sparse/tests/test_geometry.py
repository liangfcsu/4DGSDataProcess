from __future__ import annotations

import unittest

import numpy as np

from track_sparse.geometry.geometry import project, robust_triangulate
from track_sparse.core.schema import Camera


def camera(cam_id: int, center: np.ndarray) -> Camera:
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    R = np.eye(3)
    t = -center
    return Camera(cam_id, cam_id + 1, cam_id + 1, 640, 480, K, R, t, K @ np.column_stack((R, t)), center, f"cam{cam_id:03d}frame001.png")


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.cameras = [camera(0, np.array([-1.0, 0.0, 0.0])), camera(1, np.array([0.0, 0.0, 0.0])), camera(2, np.array([1.0, 0.0, 0.0]))]
        self.xyz = np.array([0.2, -0.1, 8.0])
        self.uvs = np.asarray([project(cam, self.xyz)[0][0] for cam in self.cameras])

    def test_exact_triangulation(self):
        result = robust_triangulate(self.cameras, self.uvs, min_angle_deg=0.1)
        self.assertTrue(result.valid)
        np.testing.assert_allclose(result.xyz, self.xyz, atol=1e-8)
        self.assertEqual(int(result.inliers.sum()), 3)

    def test_outlier_is_removed(self):
        cameras = self.cameras + [camera(3, np.array([2.0, 0.0, 0.0]))]
        uvs = np.vstack((self.uvs, project(cameras[-1], self.xyz)[0][0] + np.array([20.0, -10.0])))
        result = robust_triangulate(cameras, uvs, min_angle_deg=0.1, max_reprojection_error_px=2.0)
        self.assertTrue(result.valid)
        self.assertFalse(result.inliers[-1])
        np.testing.assert_allclose(result.xyz, self.xyz, atol=1e-6)

    def test_small_gaussian_pixel_noise_is_stable(self):
        noisy = self.uvs + np.array([[0.2, -0.1], [-0.3, 0.2], [0.1, 0.25]])
        result = robust_triangulate(self.cameras, noisy, min_angle_deg=0.1, max_reprojection_error_px=2.0)
        self.assertTrue(result.valid)
        np.testing.assert_allclose(result.xyz, self.xyz, atol=0.02)

    def test_one_view_and_negative_depth_rejected(self):
        one = robust_triangulate(self.cameras[:1], self.uvs[:1])
        self.assertFalse(one.valid)
        behind = np.array([0.0, 0.0, -5.0])
        behind_uvs = np.asarray([project(cam, behind)[0][0] for cam in self.cameras])
        result = robust_triangulate(self.cameras, behind_uvs, min_angle_deg=0.1)
        self.assertFalse(result.valid)

    def test_small_angle_rejected(self):
        close = [camera(0, np.array([0.0, 0.0, 0.0])), camera(1, np.array([1e-5, 0.0, 0.0]))]
        uvs = np.asarray([project(cam, self.xyz)[0][0] for cam in close])
        result = robust_triangulate(close, uvs, min_angle_deg=1.0)
        self.assertFalse(result.valid)


if __name__ == "__main__":
    unittest.main()
