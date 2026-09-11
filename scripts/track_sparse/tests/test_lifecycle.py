from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from track_sparse.config import apply_overrides, load_config
from track_sparse.geometry import project
from track_sparse.lifecycle import TrackManager
from track_sparse.schema import Camera, FeatureObservation, Observation, SpatialGroup


def _camera(cam_id: int, center_x: float) -> Camera:
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    R = np.eye(3)
    center = np.array([center_x, 0.0, 0.0])
    t = -center
    return Camera(cam_id, cam_id + 1, cam_id + 1, 640, 480, K, R, t, K @ np.column_stack((R, t)), center, f"cam{cam_id:03d}frame001.png")


class _FakeTemporal:
    def propagate(self, track_id: int, source: Observation, target_frame: int) -> Observation:
        return Observation(
            track_id, target_frame, source.cam_id, source.feature_id,
            source.u, source.v, tracker_confidence=0.9, source="temporal",
        )


class LifecycleTests(unittest.TestCase):
    def test_periodic_birth_suppresses_existing_track(self):
        cameras = {0: _camera(0, -1.0), 1: _camera(1, 1.0)}
        xyz = np.array([0.0, 0.0, 8.0])
        observations = {}
        for cam_id, camera in cameras.items():
            uv = project(camera, xyz)[0][0]
            observations[cam_id] = FeatureObservation(1, cam_id, 10 + cam_id, uv, 0.9, 0.9)
        first = SpatialGroup(1, observations, xyz, 0.0, 10.0, 1.0)
        second_observations = {
            cam_id: FeatureObservation(2, cam_id, item.feature_id, item.uv.copy(), 0.9, 0.9)
            for cam_id, item in observations.items()
        }
        second = SpatialGroup(2, second_observations, xyz, 0.0, 10.0, 1.0)
        config = apply_overrides(load_config(), {
            "spawn.min_seed_views": 2,
            "spawn.min_duplicate_views": 2,
            "triangulation.min_angle_deg": 0.1,
            "performance.triangulation_backend": "cpu",
            "performance.cpu_workers": 2,
        })
        manager = TrackManager(cameras, [1, 2], config, _FakeTemporal())
        tracks = manager.run({1: [first], 2: [second]})
        self.assertEqual(len(tracks), 1)
        self.assertEqual(manager.stats["duplicate_groups_suppressed"], 1)
        self.assertTrue(tracks[1].samples[1].valid_3d)
        self.assertTrue(tracks[1].samples[2].valid_3d)

    def test_frame_checkpoint_resumes_after_last_completed_frame(self):
        cameras = {0: _camera(0, -1.0), 1: _camera(1, 1.0)}
        xyz = np.array([0.0, 0.0, 8.0])
        observations = {}
        for cam_id, camera in cameras.items():
            uv = project(camera, xyz)[0][0]
            observations[cam_id] = FeatureObservation(1, cam_id, 10 + cam_id, uv, 0.9, 0.9)
        group = SpatialGroup(1, observations, xyz, 0.0, 10.0, 1.0)
        config = apply_overrides(load_config(), {
            "spawn.min_seed_views": 2,
            "triangulation.min_angle_deg": 0.1,
            "performance.triangulation_backend": "cpu",
            "performance.cpu_workers": 2,
            "performance.checkpoint_interval_frames": 1,
        })
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "tracks.pkl"
            first = TrackManager(cameras, [1, 2], config, _FakeTemporal())
            first.run({1: [group]}, checkpoint, "same-input", resume=False)
            second = TrackManager(cameras, [1, 2, 3], config, _FakeTemporal())
            tracks = second.run({1: [group]}, checkpoint, "same-input", resume=True)
            self.assertIn(3, tracks[1].samples)
            self.assertTrue(tracks[1].samples[3].valid_3d)
            self.assertEqual(second.stats["new_tracks"], 1)


if __name__ == "__main__":
    unittest.main()
