from __future__ import annotations

import unittest

import numpy as np

from track_sparse.core.config import apply_overrides, load_config
from track_sparse.geometry.geometry import project
from track_sparse.tracking.identity_validation import split_identity_jumps
from track_sparse.optimization.motion_groups import group_dynamic_tracks
from track_sparse.optimization.motion_models import fit_motion_models
from track_sparse.matching.observation_graph import ObservationGraphAssociator
from track_sparse.optimization.pose_refinement import refine_camera_poses
from track_sparse.core.schema import Camera, MotionClass, Observation, Track, TrackSample, TrackState
from track_sparse.geometry.triangulation_batch import (
    TriangulationInput, cuda_is_available, triangulate_cuda_batch,
)
from track_sparse.geometry.uncertainty import estimate_point_covariance


def _camera(cam_id: int, center: np.ndarray) -> Camera:
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    R = np.eye(3)
    center = np.asarray(center, dtype=np.float64)
    t = -center
    return Camera(
        cam_id, cam_id + 1, cam_id + 1, 640, 480, K, R, t,
        K @ np.column_stack((R, t)), center, f"cam{cam_id:03d}frame001.png",
    )


def _sample(track_id: int, frame_id: int, xyz: np.ndarray, rmse: float = 0.2) -> TrackSample:
    return TrackSample(
        track_id, frame_id, np.asarray(xyz, dtype=np.float64), True, 2, 2,
        rmse, 10.0, 0.9, state=TrackState.ACTIVE,
        covariance=np.eye(3) * 1.0e-6, position_std=1.0e-3,
    )


class _ConflictTemporal:
    def __init__(self):
        self.links = {
            (0, 10): (100, 0.9), (0, 20): (100, 0.8),
            (1, 11): (101, 0.2), (1, 21): (101, 0.8),
        }

    def propagate(self, track_id: int, source: Observation, target_frame: int):
        target, score = self.links[(source.cam_id, source.feature_id)]
        return Observation(
            track_id, target_frame, source.cam_id, target, float(target), 20.0,
            tracker_confidence=score, association_score=score, source="temporal",
        )


class TrajectoryOptimizationTests(unittest.TestCase):
    @unittest.skipUnless(cuda_is_available(), "CUDA is not available")
    def test_cuda_batch_triangulation_matches_geometry(self):
        cameras = [
            _camera(0, [-1, 0, 0]),
            _camera(1, [1, 0, 0]),
            _camera(2, [0, 0.5, 0]),
        ]
        inputs = []
        expected = []
        for track_id in range(1, 7):
            xyz = np.array([0.03 * track_id, -0.02 * track_id, 5.0 + 0.1 * track_id])
            uvs = np.asarray([project(camera, xyz)[0][0] for camera in cameras])
            inputs.append(TriangulationInput(track_id, cameras, uvs))
            expected.append(xyz)
        config = apply_overrides(load_config(), {
            "performance.cpu_workers": 2,
            "performance.gpu_hypothesis_batch_size": 8,
            "triangulation.min_angle_deg": 0.1,
        })
        results, stats = triangulate_cuda_batch(inputs, config)
        self.assertGreater(stats["cuda_hypotheses"], 0)
        for item, xyz in zip(inputs, expected):
            self.assertTrue(results[item.track_id].valid)
            self.assertTrue(np.allclose(results[item.track_id].xyz, xyz, atol=1e-4))

    def test_uncertainty_reflects_camera_baseline(self):
        xyz = np.array([0.0, 0.0, 8.0])
        narrow = [_camera(0, [-0.05, 0, 0]), _camera(1, [0.05, 0, 0])]
        wide = [_camera(0, [-1.0, 0, 0]), _camera(1, [1.0, 0, 0])]
        narrow_cov, narrow_std = estimate_point_covariance(narrow, xyz, 0.5)
        wide_cov, wide_std = estimate_point_covariance(wide, xyz, 0.5)
        self.assertTrue(np.all(np.linalg.eigvalsh(wide_cov) >= 0))
        self.assertTrue(np.all(np.isfinite(narrow_cov)))
        self.assertLess(wide_std, narrow_std)

    def test_joint_assignment_respects_two_camera_quota(self):
        config = apply_overrides(load_config(), {
            "temporal.offsets": [1],
            "temporal.min_camera_votes": 2,
            "association.reject_cycle_inconsistent": False,
        })
        tracks = {}
        for track_id, features in ((1, (10, 11)), (2, (20, 21))):
            track = Track(track_id, 1, state=TrackState.ACTIVE)
            for cam_id, feature_id in enumerate(features):
                track.observations[(1, cam_id)] = Observation(
                    track_id, 1, cam_id, feature_id, 10.0, 20.0, source="seed"
                )
            tracks[track_id] = track
        observations, stats = ObservationGraphAssociator(
            _ConflictTemporal(), [0, 1], config
        ).associate(tracks, 2)
        self.assertEqual({item.track_id for item in observations}, {2})
        self.assertEqual({item.cam_id for item in observations}, {0, 1})
        self.assertEqual(stats["association_accepted"], 2)

    def test_identity_jump_is_split_and_lineage_is_kept(self):
        config = apply_overrides(load_config(), {
            "identity_validation.min_segment_frames": 3,
        })
        track = Track(1, 1, state=TrackState.ACTIVE)
        positions = [0.00, 0.01, 0.02, 0.03, 1.00, 1.01, 1.02, 1.03]
        for frame_id, x in enumerate(positions, 1):
            track.samples[frame_id] = _sample(1, frame_id, [x, 0.0, 5.0])
            for cam_id in (0, 1):
                score = 0.1 if frame_id == 5 else 0.9
                track.observations[(frame_id, cam_id)] = Observation(
                    1, frame_id, cam_id, frame_id * 10 + cam_id, 0.0, 0.0,
                    is_inlier=True, source="temporal", association_score=score,
                )
        output, events, stats = split_identity_jumps({1: track}, 1.0, config)
        self.assertEqual(stats["track_split_count"], 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(len(output), 2)
        child = output[max(output)]
        self.assertEqual(child.identity_parent_id, 1)
        self.assertEqual(child.split_frame, 5)

    def test_competing_models_anchor_static_and_preserve_dynamic_motion(self):
        cameras = {0: _camera(0, [-1, 0, 0]), 1: _camera(1, [1, 0, 0])}
        tracks = {}
        for track_id, moving in ((1, False), (2, True)):
            track = Track(track_id, 1, state=TrackState.ACTIVE)
            for frame_id in range(1, 9):
                xyz = np.array([0.02 * (frame_id - 1) if moving else 0.0, 0.0, 6.0])
                track.samples[frame_id] = _sample(track_id, frame_id, xyz)
                for cam_id, camera in cameras.items():
                    uv = project(camera, xyz)[0][0]
                    track.observations[(frame_id, cam_id)] = Observation(
                        track_id, frame_id, cam_id, 100 * frame_id + cam_id,
                        float(uv[0]), float(uv[1]), tracker_confidence=0.9,
                        association_score=0.9, reprojection_error=0.2,
                        is_inlier=True, source="temporal",
                    )
            tracks[track_id] = track
        config = apply_overrides(load_config(), {
            "motion_models.selection_margin": 1.0,
            "dynamic_optimization.acceleration_weight": 0.05,
        })
        fit_motion_models(tracks, cameras, 2.0, config)
        self.assertEqual(tracks[1].motion_class, MotionClass.STATIC)
        self.assertEqual(tracks[2].motion_class, MotionClass.DYNAMIC)
        static_values = np.asarray([sample.output_xyz for sample in tracks[1].samples.values()])
        self.assertTrue(np.allclose(static_values, static_values[0], atol=1e-12))
        dynamic_values = np.asarray([sample.output_xyz for sample in tracks[2].samples.values()])
        self.assertGreater(float(np.ptp(dynamic_values[:, 0])), 0.08)

    def test_pose_refinement_uses_only_static_canonical_points(self):
        camera = _camera(0, [0.0, 0.0, 0.0])
        true_camera = _camera(0, [0.02, 0.0, 0.0])
        tracks = {}
        for index in range(30):
            xyz = np.array([
                (index % 6 - 2.5) * 0.12,
                (index // 6 - 2.0) * 0.10,
                4.5 + 0.08 * (index % 5),
            ])
            uv = project(true_camera, xyz)[0][0]
            track = Track(index + 1, 1, state=TrackState.ACTIVE)
            track.motion_class = MotionClass.STATIC
            track.static_confidence = 0.99
            track.valid_frame_count = 20
            track.canonical_xyz = xyz
            track.observations[(2, 0)] = Observation(
                index + 1, 2, 0, index, float(uv[0]), float(uv[1]), is_inlier=True
            )
            tracks[index + 1] = track
        config = apply_overrides(load_config(), {
            "pose_refinement.min_points": 8,
            "pose_refinement.min_track_frames": 5,
            "pose_refinement.max_translation_ratio": 0.1,
            "pose_refinement.max_rotation_deg": 2.0,
            "pose_refinement.blend": 1.0,
        })
        corrected, _, stats = refine_camera_poses(tracks, {0: camera}, 1.0, config)
        self.assertEqual(stats["pose_corrections"], 1)
        self.assertTrue(np.allclose(corrected[(2, 0)].center_world, true_camera.center_world, atol=2e-3))

    def test_dynamic_motion_groups_are_geometry_based(self):
        config = apply_overrides(load_config(), {
            "motion_groups.min_track_frames": 5,
            "motion_groups.search_radius_ratio": 0.5,
            "motion_groups.min_overlap_frames": 5,
            "motion_groups.motion_residual_ratio": 0.01,
            "motion_groups.max_rigid_residual_ratio": 0.01,
        })
        tracks = {}
        offsets = ([0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0.1, 0.1, 0])
        for track_id, offset in enumerate(offsets, 1):
            track = Track(track_id, 1, motion_class=MotionClass.DYNAMIC)
            for frame_id in range(1, 9):
                angle = 0.03 * frame_id
                rotation = np.array([
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ])
                xyz = rotation @ np.asarray(offset, dtype=float) + np.array([0.02 * frame_id, 0, 5])
                sample = _sample(track_id, frame_id, xyz)
                sample.optimized_xyz = xyz.copy()
                track.samples[frame_id] = sample
            tracks[track_id] = track
        records, stats = group_dynamic_tracks(tracks, 1.0, config)
        self.assertEqual(stats["motion_group_count"], 1)
        self.assertEqual(stats["grouped_dynamic_tracks"], 4)
        self.assertEqual(records[0]["track_ids"], [1, 2, 3, 4])
        self.assertEqual({track.motion_group_id for track in tracks.values()}, {0})


if __name__ == "__main__":
    unittest.main()
