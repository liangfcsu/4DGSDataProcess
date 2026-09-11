"""Small per-frame camera corrections estimated from high-confidence static tracks."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping

import cv2
import numpy as np

from .geometry import project, reprojection_errors, triangulation_angle_deg
from .schema import Camera, MotionClass, Track
from .uncertainty import camera_at, projection_jacobian


def _camera_with_pose(camera: Camera, R_w2c: np.ndarray, t_w2c: np.ndarray) -> Camera:
    R_w2c = np.asarray(R_w2c, dtype=np.float64)
    t_w2c = np.asarray(t_w2c, dtype=np.float64)
    center = -R_w2c.T @ t_w2c
    return Camera(
        camera.cam_id,
        camera.image_id,
        camera.camera_id,
        camera.width,
        camera.height,
        camera.K.copy(),
        R_w2c,
        t_w2c,
        camera.K @ np.column_stack((R_w2c, t_w2c)),
        center,
        camera.reference_image_name,
    )


def refine_camera_poses(
    tracks: dict[int, Track],
    cameras: Mapping[int, Camera],
    scene_scale: float,
    config: dict,
) -> tuple[dict[tuple[int, int], Camera], list[dict], dict[str, float | int]]:
    """Solve robust PnP corrections without changing the reference calibration."""
    section = config["pose_refinement"]
    if not bool(section["enabled"]):
        return {}, [], {"pose_corrections": 0, "pose_candidates": 0}
    correspondences: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for track in tracks.values():
        if (
            track.motion_class != MotionClass.STATIC
            or track.static_confidence < float(section["min_static_confidence"])
            or track.valid_frame_count < int(section["min_track_frames"])
            or not np.all(np.isfinite(track.canonical_xyz))
        ):
            continue
        for observation in track.observations.values():
            if observation.visible and observation.is_inlier and observation.cam_id in cameras:
                correspondences[(observation.frame_id, observation.cam_id)].append(
                    (track.canonical_xyz, np.array([observation.u, observation.v]))
                )

    minimum = int(section["min_points"])

    def solve_pose(item):
        (frame_id, cam_id), values = item
        if len(values) < minimum:
            return None
        camera = cameras[cam_id]
        object_points = np.asarray([value[0] for value in values], dtype=np.float64)
        image_points = np.asarray([value[1] for value in values], dtype=np.float64)
        initial_rvec = cv2.Rodrigues(camera.R_w2c)[0]
        initial_tvec = camera.t_w2c.reshape(3, 1).copy()
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera.K,
            np.zeros(4),
            initial_rvec,
            initial_tvec,
            useExtrinsicGuess=True,
            iterationsCount=int(section["ransac_iterations"]),
            reprojectionError=float(section["max_reprojection_error_px"]),
            confidence=float(section["ransac_confidence"]),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        inlier_count = 0 if inliers is None else len(inliers)
        if not success or inlier_count < minimum:
            return None
        ids = inliers.reshape(-1)
        if hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points[ids], image_points[ids], camera.K, np.zeros(4), rvec, tvec
            )
        new_R = cv2.Rodrigues(rvec)[0]
        new_t = tvec.reshape(3)
        delta_R = new_R @ camera.R_w2c.T
        delta_rvec = cv2.Rodrigues(delta_R)[0].reshape(3)
        rotation_deg = math.degrees(float(np.linalg.norm(delta_rvec)))
        candidate = _camera_with_pose(camera, new_R, new_t)
        translation = float(np.linalg.norm(candidate.center_world - camera.center_world))
        if (
            rotation_deg > float(section["max_rotation_deg"])
            or translation > float(scene_scale) * float(section["max_translation_ratio"])
        ):
            return None
        blend = float(section["blend"])
        blended_delta = cv2.Rodrigues(delta_rvec * blend)[0]
        blended_R = blended_delta @ camera.R_w2c
        blended_t = camera.t_w2c + blend * (new_t - camera.t_w2c)
        refined = _camera_with_pose(camera, blended_R, blended_t)
        projected, _ = project(refined, object_points[ids])
        rmse = float(np.sqrt(np.mean(np.sum((projected - image_points[ids]) ** 2, axis=1))))
        return (frame_id, cam_id), refined, rotation_deg * blend, translation * blend, {
            "frame_id": frame_id,
            "cam_id": cam_id,
            "R_w2c": refined.R_w2c.tolist(),
            "t_w2c": refined.t_w2c.tolist(),
            "rotation_delta_deg": rotation_deg * blend,
            "translation_delta": translation * blend,
            "inlier_count": inlier_count,
            "reprojection_rmse": rmse,
        }

    corrected: dict[tuple[int, int], Camera] = {}
    records: list[dict] = []
    rotation_values = []
    translation_values = []
    solutions = [solve_pose(item) for item in sorted(correspondences.items())]
    for solution in solutions:
        if solution is None:
            continue
        key, refined, rotation, translation, record = solution
        corrected[key] = refined
        rotation_values.append(rotation)
        translation_values.append(translation)
        records.append(record)
    return corrected, records, {
        "pose_candidates": len(correspondences),
        "pose_corrections": len(corrected),
        "median_pose_rotation_deg": float(np.median(rotation_values)) if rotation_values else 0.0,
        "median_pose_translation": float(np.median(translation_values)) if translation_values else 0.0,
    }


def _refine_point_gauss_newton(
    xyz: np.ndarray,
    cameras: list[Camera],
    uvs: np.ndarray,
    iterations: int,
    huber_px: float,
) -> np.ndarray:
    value = np.asarray(xyz, dtype=np.float64).copy()
    for _ in range(iterations):
        normal = np.zeros((3, 3), dtype=np.float64)
        gradient = np.zeros(3, dtype=np.float64)
        for camera, uv in zip(cameras, uvs):
            predicted, depth = project(camera, value)
            if depth[0] <= 0:
                continue
            residual = predicted[0] - uv
            norm = float(np.linalg.norm(residual))
            weight = 1.0 if norm <= huber_px else huber_px / max(norm, 1e-12)
            jacobian = projection_jacobian(camera, value)
            normal += weight * jacobian.T @ jacobian
            gradient += weight * jacobian.T @ residual
        regularizer = max(float(np.trace(normal)) / 3.0, 1.0) * 1e-10
        try:
            delta = -np.linalg.solve(normal + regularizer * np.eye(3), gradient)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(delta)):
            break
        value += delta
        if float(np.linalg.norm(delta)) < 1e-9:
            break
    return value


def refine_track_measurements(
    tracks: dict[int, Track],
    cameras: Mapping[int, Camera],
    corrected_cameras: Mapping[tuple[int, int], Camera],
    config: dict,
) -> dict[str, int]:
    """Fast local re-triangulation after pose correction, preserving raw xyz."""
    if not corrected_cameras:
        return {"pose_refined_samples": 0, "pose_refinement_rejections": 0}
    tri = config["triangulation"]
    section = config["pose_refinement"]
    def refine_track(track: Track) -> tuple[int, int]:
        refined_count = 0
        rejected = 0
        for frame_id, sample in track.samples.items():
            if not sample.valid_3d or not np.all(np.isfinite(sample.xyz)):
                continue
            observations = [
                observation for (frame, _), observation in track.observations.items()
                if frame == frame_id and observation.visible and observation.is_inlier
            ]
            if len(observations) < int(tri["min_views"]):
                continue
            if not any((frame_id, observation.cam_id) in corrected_cameras for observation in observations):
                continue
            sample_cameras = [
                camera_at(cameras, corrected_cameras, frame_id, observation.cam_id)
                for observation in observations
            ]
            uvs = np.asarray([[observation.u, observation.v] for observation in observations])
            xyz = _refine_point_gauss_newton(
                sample.xyz,
                sample_cameras,
                uvs,
                int(section["point_refine_iterations"]),
                float(tri["max_reprojection_error_px"]),
            )
            errors, depths = reprojection_errors(sample_cameras, uvs, xyz)
            inliers = (errors <= float(tri["max_reprojection_error_px"])) & (depths > 0)
            accepted_cameras = [sample_cameras[index] for index in np.flatnonzero(inliers)]
            angle = triangulation_angle_deg(accepted_cameras, xyz)
            if len(accepted_cameras) < int(tri["min_views"]) or angle < float(tri["min_angle_deg"]):
                rejected += 1
                continue
            sample.pose_refined_xyz = xyz
            sample.pose_refined = True
            sample.num_inlier_views = int(inliers.sum())
            sample.reprojection_rmse = float(np.sqrt(np.mean(errors[inliers] ** 2)))
            sample.min_triangulation_angle_deg = angle
            for observation, error, is_inlier in zip(observations, errors, inliers):
                observation.reprojection_error = float(error)
                observation.is_inlier = bool(is_inlier)
            refined_count += 1
        return refined_count, rejected

    refined_count = rejected = 0
    for track_refined, track_rejected in (refine_track(track) for track in tracks.values()):
        refined_count += track_refined
        rejected += track_rejected
    return {"pose_refined_samples": refined_count, "pose_refinement_rejections": rejected}
