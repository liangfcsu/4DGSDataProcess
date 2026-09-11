"""Geometry-only static/dynamic model selection and trajectory optimization."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .geometry import project
from .identity_validation import recompute_track_metadata
from .schema import Camera, MotionClass, Track
from .uncertainty import camera_at, mahalanobis_distance, projection_jacobian

try:
    from scipy.sparse import csr_matrix, diags, eye
    from scipy.sparse.linalg import spsolve
except ImportError:  # pragma: no cover
    csr_matrix = diags = eye = spsolve = None


def _track_observations(track: Track):
    observations = [value for value in track.observations.values() if value.visible and value.is_inlier]
    return observations or [value for value in track.observations.values() if value.visible]


def optimize_static_point(
    track: Track,
    cameras: Mapping[int, Camera],
    corrected_cameras: Mapping[tuple[int, int], Camera] | None,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Jointly fit one world point to all frames and cameras with Huber IRLS."""
    samples = [sample for sample in track.samples.values() if sample.valid_3d]
    if not samples:
        return np.full(3, np.nan), np.full((3, 3), np.nan), float("inf"), 0
    xyz_values = np.asarray([sample.measurement_xyz for sample in samples], dtype=np.float64)
    finite = np.all(np.isfinite(xyz_values), axis=1)
    if not finite.any():
        return np.full(3, np.nan), np.full((3, 3), np.nan), float("inf"), 0
    xyz = np.median(xyz_values[finite], axis=0)
    observations = _track_observations(track)
    maximum = int(config["motion_models"]["max_static_observations"])
    if len(observations) > maximum:
        indices = np.linspace(0, len(observations) - 1, maximum, dtype=np.int64)
        observations = [observations[int(index)] for index in indices]
    huber = float(config["motion_models"]["static_huber_px"])
    damping = float(config["uncertainty"]["damping"])
    normal = np.eye(3)
    residual_values: list[float] = []
    for _ in range(int(config["motion_models"]["static_refine_iterations"])):
        normal = np.zeros((3, 3), dtype=np.float64)
        gradient = np.zeros(3, dtype=np.float64)
        residual_values = []
        for observation in observations:
            if observation.cam_id not in cameras:
                continue
            camera = camera_at(
                cameras, corrected_cameras, observation.frame_id, observation.cam_id
            )
            predicted, depth = project(camera, xyz)
            if depth[0] <= 0:
                continue
            residual = predicted[0] - np.array([observation.u, observation.v])
            norm = float(np.linalg.norm(residual))
            confidence = max(
                0.05, observation.association_score,
                observation.tracker_confidence, observation.spatial_confidence,
            )
            robust_weight = 1.0 if norm <= huber else huber / max(norm, 1e-12)
            weight = confidence * robust_weight
            jacobian = projection_jacobian(camera, xyz)
            if not np.all(np.isfinite(jacobian)):
                continue
            normal += weight * (jacobian.T @ jacobian)
            gradient += weight * (jacobian.T @ residual)
            residual_values.append(norm)
        regularizer = max(float(np.trace(normal)) / 3.0, 1.0) * damping
        try:
            delta = -np.linalg.solve(normal + regularizer * np.eye(3), gradient)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(delta)):
            break
        xyz += delta
        if float(np.linalg.norm(delta)) < 1e-9:
            break

    # Recompute unbiased residuals and covariance at the solution.
    normal = np.zeros((3, 3), dtype=np.float64)
    squared_errors = []
    for observation in observations:
        if observation.cam_id not in cameras:
            continue
        camera = camera_at(cameras, corrected_cameras, observation.frame_id, observation.cam_id)
        predicted, depth = project(camera, xyz)
        if depth[0] <= 0:
            continue
        residual = predicted[0] - np.array([observation.u, observation.v])
        jacobian = projection_jacobian(camera, xyz)
        if not np.all(np.isfinite(jacobian)):
            continue
        normal += jacobian.T @ jacobian
        squared_errors.append(float(residual @ residual))
    rmse = math.sqrt(float(np.mean(squared_errors))) if squared_errors else float("inf")
    regularizer = max(float(np.trace(normal)) / 3.0, 1.0) * damping
    try:
        covariance = np.linalg.pinv(normal + regularizer * np.eye(3), rcond=1e-10)
        covariance *= max(rmse, float(config["uncertainty"]["min_pixel_sigma"])) ** 2
    except np.linalg.LinAlgError:
        covariance = np.full((3, 3), np.nan)
    return xyz, covariance, rmse, len(squared_errors)


def _dynamic_reprojection_sse(track: Track) -> tuple[float, int, float]:
    sse = 0.0
    count = 0
    values = []
    for sample in track.samples.values():
        if not sample.valid_3d or not math.isfinite(sample.reprojection_rmse):
            continue
        observations = max(sample.num_inlier_views, 1)
        sse += sample.reprojection_rmse**2 * observations
        count += observations
        values.append(sample.reprojection_rmse)
    rmse = math.sqrt(sse / count) if count else float("inf")
    return sse, count, rmse


def _static_reprojection_sse(
    track: Track,
    xyz: np.ndarray,
    cameras: Mapping[int, Camera],
    corrected_cameras: Mapping[tuple[int, int], Camera] | None,
) -> tuple[float, int]:
    sse = 0.0
    count = 0
    for observation in _track_observations(track):
        if observation.cam_id not in cameras:
            continue
        camera = camera_at(cameras, corrected_cameras, observation.frame_id, observation.cam_id)
        predicted, depth = project(camera, xyz)
        if depth[0] <= 0:
            continue
        residual = predicted[0] - np.array([observation.u, observation.v])
        sse += float(residual @ residual)
        count += 1
    return sse, count


def _smooth_dynamic(track: Track, scene_scale: float, config: dict) -> None:
    section = config["dynamic_optimization"]
    entries = [
        (frame, sample) for frame, sample in sorted(track.samples.items())
        if sample.valid_3d and np.all(np.isfinite(sample.measurement_xyz))
    ]
    if not entries:
        return
    raw = np.asarray([sample.measurement_xyz for _, sample in entries], dtype=np.float64)
    frames = np.asarray([frame for frame, _ in entries], dtype=np.int32)
    std = np.asarray([sample.position_std for _, sample in entries], dtype=np.float64)
    minimum_std = max(float(scene_scale) * float(section["min_position_std_ratio"]), 1e-9)
    std = np.where(np.isfinite(std), np.maximum(std, minimum_std), minimum_std * 5.0)
    data_weights = 1.0 / np.square(std)
    count = len(entries)
    if count < 3 or spsolve is None:
        for (_, sample), value in zip(entries, raw):
            sample.optimized_xyz = value.copy()
        return

    rows_first = []
    rows_second = []
    maximum_gap = int(section["max_smooth_gap"])
    for index in range(count - 1):
        gap = int(frames[index + 1] - frames[index])
        if 0 < gap <= maximum_gap:
            row = np.zeros(count)
            row[index], row[index + 1] = -1.0 / gap, 1.0 / gap
            rows_first.append(row)
    for index in range(1, count - 1):
        left = int(frames[index] - frames[index - 1])
        right = int(frames[index + 1] - frames[index])
        if left == right == 1:
            row = np.zeros(count)
            row[index - 1:index + 2] = (1.0, -2.0, 1.0)
            rows_second.append(row)
    temporal_sigma = max(float(scene_scale) * float(section["temporal_sigma_ratio"]), 1e-9)
    matrix = diags(data_weights, format="csr")
    if rows_first and float(section["velocity_weight"]) > 0:
        D1 = csr_matrix(np.asarray(rows_first))
        matrix = matrix + float(section["velocity_weight"]) / temporal_sigma**2 * (D1.T @ D1)
    if rows_second and float(section["acceleration_weight"]) > 0:
        D2 = csr_matrix(np.asarray(rows_second))
        matrix = matrix + float(section["acceleration_weight"]) / temporal_sigma**2 * (D2.T @ D2)
    matrix = matrix + 1e-12 * eye(count, format="csr")
    optimized = np.column_stack([
        spsolve(matrix, data_weights * raw[:, dimension]) for dimension in range(3)
    ])
    maximum_sigma = float(section["max_correction_sigma"])
    for index, ((_, sample), value) in enumerate(zip(entries, optimized)):
        correction = value - raw[index]
        limit = maximum_sigma * std[index]
        norm = float(np.linalg.norm(correction))
        if norm > limit > 0:
            value = raw[index] + correction * (limit / norm)
        sample.optimized_xyz = np.asarray(value, dtype=np.float64)


def _set_quality(track: Track, config: dict) -> None:
    valid = [sample for sample in track.samples.values() if sample.valid_3d]
    confidences = [sample.geometry_confidence for sample in valid]
    reprojections = [sample.reprojection_rmse for sample in valid if math.isfinite(sample.reprojection_rmse)]
    track.mean_confidence = float(np.mean(confidences)) if confidences else 0.0
    track.median_reprojection_error = float(np.median(reprojections)) if reprojections else float("nan")
    reproj_limit = float(config["triangulation"]["max_reprojection_error_px"])
    length_score = min(track.valid_frame_count / 50.0, 1.0)
    reproj_score = (
        max(0.0, 1.0 - track.median_reprojection_error / reproj_limit)
        if math.isfinite(track.median_reprojection_error) else 0.0
    )
    uncertainty_values = [sample.position_std for sample in valid if math.isfinite(sample.position_std)]
    uncertainty_score = 1.0
    if uncertainty_values and np.all(np.isfinite(track.canonical_xyz)):
        depth_scale = max(float(np.linalg.norm(track.canonical_xyz)), 1e-9)
        uncertainty_score = math.exp(-float(np.median(uncertainty_values)) / (0.01 * depth_scale))
    quality_score = (
        0.35 * length_score + 0.30 * track.mean_confidence
        + 0.20 * reproj_score + 0.15 * uncertainty_score
    )
    track.quality = "high" if quality_score >= 0.75 else "medium" if quality_score >= 0.45 else "low"


def fit_motion_models(
    tracks: dict[int, Track],
    cameras: Mapping[int, Camera],
    scene_scale: float,
    config: dict,
    corrected_cameras: Mapping[tuple[int, int], Camera] | None = None,
    *,
    optimize_coordinates: bool = True,
) -> dict[str, int | float]:
    """Fit competing static/dynamic models and write final sample coordinates."""
    section = config["motion_models"]
    counts = {"static": 0, "dynamic": 0, "unknown": 0}
    static_rmse_values = []
    for track in tracks.values():
        recompute_track_metadata(track)
        canonical, covariance, static_rmse, static_count = optimize_static_point(
            track, cameras, corrected_cameras, config
        )
        track.canonical_xyz = canonical
        track.canonical_covariance = covariance
        static_sse, static_count_full = _static_reprojection_sse(
            track, canonical, cameras, corrected_cameras
        ) if np.all(np.isfinite(canonical)) else (float("inf"), 0)
        dynamic_sse, dynamic_count, dynamic_rmse = _dynamic_reprojection_sse(track)
        count = max(2, min(static_count_full, dynamic_count))
        epsilon = float(section["score_epsilon"])
        static_mean = max(static_sse / max(static_count_full, 1), epsilon)
        dynamic_mean = max(dynamic_sse / max(dynamic_count, 1), epsilon)
        # BIC-like competition; dynamic_dof_per_frame models the effective
        # degrees of freedom left after temporal regularization.
        frames = max(track.valid_frame_count, 1)
        static_k = 3.0
        dynamic_k = 3.0 + float(section["dynamic_dof_per_frame"]) * 3.0 * max(frames - 1, 0)
        track.static_model_score = count * math.log(static_mean) + static_k * math.log(count)
        track.dynamic_model_score = count * math.log(dynamic_mean) + dynamic_k * math.log(count)
        track.model_score_margin = track.dynamic_model_score - track.static_model_score

        significances = []
        for sample in track.samples.values():
            if not sample.valid_3d or not np.all(np.isfinite(canonical)):
                continue
            combined = sample.covariance + covariance
            significance = mahalanobis_distance(sample.measurement_xyz - canonical, combined)
            sample.motion_significance = significance
            if math.isfinite(significance):
                significances.append(significance)
        p90_significance = float(np.percentile(significances, 90)) if significances else float("inf")
        normalized_margin = track.model_score_margin / max(count, 1)
        track.static_confidence = float(1.0 / (1.0 + math.exp(-np.clip(normalized_margin, -30, 30))))

        if track.valid_frame_count < int(section["min_valid_frames"]):
            track.motion_class = MotionClass.UNKNOWN
        elif (
            track.model_score_margin >= float(section["selection_margin"])
            and static_rmse <= float(section["static_max_reprojection_px"])
            and p90_significance <= float(section["static_max_motion_sigma"])
        ):
            track.motion_class = MotionClass.STATIC
        elif (
            -track.model_score_margin >= float(section["selection_margin"])
            and p90_significance >= float(section["dynamic_min_motion_sigma"])
        ):
            track.motion_class = MotionClass.DYNAMIC
        else:
            track.motion_class = MotionClass.UNKNOWN
        counts[track.motion_class.value] += 1
        if math.isfinite(static_rmse):
            static_rmse_values.append(static_rmse)

        if optimize_coordinates:
            if track.motion_class == MotionClass.STATIC and np.all(np.isfinite(canonical)):
                for sample in track.samples.values():
                    if sample.valid_3d:
                        # A static track owns one canonical point by definition.
                        sample.optimized_xyz = canonical.copy()
            elif track.motion_class == MotionClass.DYNAMIC:
                _smooth_dynamic(track, scene_scale, config)
            else:
                for sample in track.samples.values():
                    if sample.valid_3d:
                        sample.optimized_xyz = sample.measurement_xyz.copy()
        _set_quality(track, config)
    return {
        "static_tracks": counts["static"],
        "dynamic_tracks": counts["dynamic"],
        "unknown_tracks": counts["unknown"],
        "median_static_model_reprojection": (
            float(np.median(static_rmse_values)) if static_rmse_values else float("nan")
        ),
    }
