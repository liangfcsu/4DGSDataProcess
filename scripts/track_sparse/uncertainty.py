"""Per-sample triangulation uncertainty from multi-view projection geometry."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .schema import Camera, Track


CameraKey = tuple[int, int]  # (frame_id, cam_id)


def camera_at(
    cameras: Mapping[int, Camera],
    corrected: Mapping[CameraKey, Camera] | None,
    frame_id: int,
    cam_id: int,
) -> Camera:
    if corrected is not None and (frame_id, cam_id) in corrected:
        return corrected[(frame_id, cam_id)]
    return cameras[cam_id]


def projection_jacobian(camera: Camera, xyz: np.ndarray) -> np.ndarray:
    """Analytic d(pixel_xy)/d(world_xyz) for a pinhole K and w2c pose."""
    xyz = np.asarray(xyz, dtype=np.float64)
    x_cam = camera.R_w2c @ xyz + camera.t_w2c
    homogeneous = camera.K @ x_cam
    z = float(homogeneous[2])
    if not math.isfinite(z) or abs(z) < 1e-12:
        return np.full((2, 3), np.nan, dtype=np.float64)
    first = (camera.K[0] * z - homogeneous[0] * camera.K[2]) / (z * z)
    second = (camera.K[1] * z - homogeneous[1] * camera.K[2]) / (z * z)
    return np.vstack((first, second)) @ camera.R_w2c


def estimate_point_covariance(
    cameras: list[Camera],
    xyz: np.ndarray,
    reprojection_rmse: float,
    *,
    min_pixel_sigma: float = 0.5,
    damping: float = 1e-9,
    max_condition_number: float = 1e12,
) -> tuple[np.ndarray, float]:
    """Return a 3x3 covariance and conservative one-sigma position radius."""
    jacobians = [projection_jacobian(camera, xyz) for camera in cameras]
    jacobians = [value for value in jacobians if np.all(np.isfinite(value))]
    if len(jacobians) < 2:
        return np.full((3, 3), np.nan), float("inf")
    J = np.vstack(jacobians)
    normal = J.T @ J
    scale = max(float(reprojection_rmse), float(min_pixel_sigma))
    regularizer = max(float(np.trace(normal)) / 3.0, 1.0) * float(damping)
    normal = normal + regularizer * np.eye(3)
    try:
        condition = float(np.linalg.cond(normal))
        if not math.isfinite(condition) or condition > max_condition_number:
            inverse = np.linalg.pinv(normal, rcond=1.0 / max_condition_number)
        else:
            inverse = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return np.full((3, 3), np.nan), float("inf")
    covariance = 0.5 * (inverse + inverse.T) * scale * scale
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.all(np.isfinite(eigenvalues)):
        return np.full((3, 3), np.nan), float("inf")
    position_std = float(math.sqrt(max(float(eigenvalues[-1]), 0.0)))
    return covariance, position_std


def annotate_track_uncertainty(
    tracks: dict[int, Track],
    cameras: Mapping[int, Camera],
    config: dict,
    corrected_cameras: Mapping[CameraKey, Camera] | None = None,
) -> dict[str, float | int]:
    """Attach covariance to every valid sample using its accepted observations."""
    section = config["uncertainty"]
    finite_std: list[float] = []
    invalid = 0
    for track in tracks.values():
        for frame_id, sample in track.samples.items():
            if not sample.valid_3d or not np.all(np.isfinite(sample.measurement_xyz)):
                invalid += 1
                continue
            observations = [
                observation
                for (frame, _), observation in track.observations.items()
                if frame == frame_id and observation.visible and observation.is_inlier
            ]
            if len(observations) < 2:
                observations = [
                    observation
                    for (frame, _), observation in track.observations.items()
                    if frame == frame_id and observation.visible
                ]
            sample_cameras = [
                camera_at(cameras, corrected_cameras, frame_id, observation.cam_id)
                for observation in observations
                if observation.cam_id in cameras
            ]
            covariance, position_std = estimate_point_covariance(
                sample_cameras,
                sample.measurement_xyz,
                sample.reprojection_rmse,
                min_pixel_sigma=float(section["min_pixel_sigma"]),
                damping=float(section["damping"]),
                max_condition_number=float(section["max_condition_number"]),
            )
            sample.covariance = covariance
            sample.position_std = position_std
            if math.isfinite(position_std):
                finite_std.append(position_std)
            else:
                invalid += 1
    return {
        "uncertainty_sample_count": len(finite_std),
        "uncertainty_invalid_count": invalid,
        "median_position_std": float(np.median(finite_std)) if finite_std else float("nan"),
        "p95_position_std": float(np.percentile(finite_std, 95)) if finite_std else float("nan"),
    }


def mahalanobis_distance(delta: np.ndarray, covariance: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(covariance)):
        return float("nan")
    try:
        value = float(delta @ np.linalg.pinv(covariance, rcond=1e-10) @ delta)
    except np.linalg.LinAlgError:
        return float("nan")
    return math.sqrt(max(value, 0.0))
