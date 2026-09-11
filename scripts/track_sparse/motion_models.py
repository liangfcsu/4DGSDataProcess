"""Geometry-only static/dynamic model selection and trajectory optimization."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .identity_validation import recompute_track_metadata
from .schema import Camera, MotionClass, Track
from .uncertainty import camera_at, mahalanobis_distance

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
    observations = [item for item in observations if item.cam_id in cameras]
    if not observations:
        return np.full(3, np.nan), np.full((3, 3), np.nan), float("inf"), 0
    sample_cameras = [
        camera_at(cameras, corrected_cameras, item.frame_id, item.cam_id)
        for item in observations
    ]
    rotations = np.asarray([camera.R_w2c for camera in sample_cameras], dtype=np.float64)
    translations = np.asarray([camera.t_w2c for camera in sample_cameras], dtype=np.float64)
    intrinsics = np.asarray([camera.K for camera in sample_cameras], dtype=np.float64)
    target_uvs = np.asarray([[item.u, item.v] for item in observations], dtype=np.float64)
    observation_confidence = np.asarray([
        max(0.05, item.association_score, item.tracker_confidence, item.spatial_confidence)
        for item in observations
    ], dtype=np.float64)

    def residuals_and_jacobians(value: np.ndarray):
        camera_xyz = np.einsum("nij,j->ni", rotations, value) + translations
        homogeneous = np.einsum("nij,nj->ni", intrinsics, camera_xyz)
        z = homogeneous[:, 2]
        valid = np.isfinite(z) & (z > 1.0e-12)
        predicted = homogeneous[:, :2] / np.where(valid, z, 1.0)[:, None]
        residuals = predicted - target_uvs
        denominator = np.where(valid, z * z, 1.0)[:, None]
        first = (
            intrinsics[:, 0] * z[:, None]
            - homogeneous[:, 0, None] * intrinsics[:, 2]
        ) / denominator
        second = (
            intrinsics[:, 1] * z[:, None]
            - homogeneous[:, 1, None] * intrinsics[:, 2]
        ) / denominator
        jacobian_camera = np.stack((first, second), axis=1)
        jacobians = np.einsum("nkq,nqj->nkj", jacobian_camera, rotations)
        valid &= np.all(np.isfinite(residuals), axis=1)
        valid &= np.all(np.isfinite(jacobians), axis=(1, 2))
        residuals = np.where(valid[:, None], residuals, 0.0)
        jacobians = np.where(valid[:, None, None], jacobians, 0.0)
        return residuals, jacobians, valid

    huber = float(config["motion_models"]["static_huber_px"])
    damping = float(config["uncertainty"]["damping"])
    normal = np.eye(3)
    for _ in range(int(config["motion_models"]["static_refine_iterations"])):
        residuals, jacobians, valid = residuals_and_jacobians(xyz)
        norms = np.linalg.norm(residuals, axis=1)
        robust = np.minimum(1.0, huber / np.maximum(norms, 1.0e-12))
        weights = observation_confidence * robust * valid
        normal = np.einsum("nki,nkj,n->ij", jacobians, jacobians, weights)
        gradient = np.einsum("nki,nk,n->i", jacobians, residuals, weights)
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
    residuals, jacobians, valid = residuals_and_jacobians(xyz)
    normal = np.einsum(
        "nki,nkj,n->ij", jacobians, jacobians, valid.astype(np.float64)
    )
    squared_errors = np.sum(residuals[valid] ** 2, axis=1)
    rmse = math.sqrt(float(np.mean(squared_errors))) if len(squared_errors) else float("inf")
    regularizer = max(float(np.trace(normal)) / 3.0, 1.0) * damping
    try:
        covariance = np.linalg.pinv(normal + regularizer * np.eye(3), rcond=1e-10)
        covariance *= max(rmse, float(config["uncertainty"]["min_pixel_sigma"])) ** 2
    except np.linalg.LinAlgError:
        covariance = np.full((3, 3), np.nan)
    return xyz, covariance, rmse, int(len(squared_errors))


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
        canonical, covariance, static_rmse, static_count_full = optimize_static_point(
            track, cameras, corrected_cameras, config
        )
        track.canonical_xyz = canonical
        track.canonical_covariance = covariance
        static_sse = (
            static_rmse * static_rmse * static_count_full
            if math.isfinite(static_rmse) else float("inf")
        )
        dynamic_sse, dynamic_count, _ = _dynamic_reprojection_sse(track)
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
