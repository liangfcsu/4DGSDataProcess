"""Epipolar geometry and robust multi-view triangulation."""

from __future__ import annotations

import itertools
import math

import numpy as np

from .schema import Camera, TriangulationResult

try:
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover - linear fallback is still usable
    least_squares = None


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def fundamental_matrix(camera_a: Camera, camera_b: Camera) -> np.ndarray:
    """Return F such that x_b.T @ F @ x_a == 0."""
    R_ba = camera_b.R_w2c @ camera_a.R_w2c.T
    t_ba = camera_b.t_w2c - R_ba @ camera_a.t_w2c
    essential = skew(t_ba) @ R_ba
    return np.linalg.inv(camera_b.K).T @ essential @ np.linalg.inv(camera_a.K)


def sampson_errors(F: np.ndarray, uv_a: np.ndarray, uv_b: np.ndarray) -> np.ndarray:
    uv_a = np.atleast_2d(np.asarray(uv_a, dtype=np.float64))
    uv_b = np.atleast_2d(np.asarray(uv_b, dtype=np.float64))
    x_a = np.column_stack((uv_a, np.ones(len(uv_a))))
    x_b = np.column_stack((uv_b, np.ones(len(uv_b))))
    Fx = (F @ x_a.T).T
    Ftx = (F.T @ x_b.T).T
    numerator = np.sum(x_b * Fx, axis=1) ** 2
    denominator = Fx[:, 0] ** 2 + Fx[:, 1] ** 2 + Ftx[:, 0] ** 2 + Ftx[:, 1] ** 2
    return np.sqrt(numerator / np.maximum(denominator, 1e-15))


def project(camera: Camera, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
    x_cam = (camera.R_w2c @ xyz.T).T + camera.t_w2c
    pixels_h = (camera.K @ x_cam.T).T
    denominator = pixels_h[:, 2:3].copy()
    near_zero = np.abs(denominator) < 1e-15
    denominator[near_zero] = np.where(denominator[near_zero] < 0, -1e-15, 1e-15)
    uv = pixels_h[:, :2] / denominator
    return uv, x_cam[:, 2]


def triangulate_dlt(cameras: list[Camera], uvs: np.ndarray) -> np.ndarray:
    if len(cameras) < 2:
        raise ValueError("三角化至少需要两个视角")
    rows = []
    for camera, uv in zip(cameras, np.asarray(uvs, dtype=np.float64)):
        x = np.linalg.inv(camera.K) @ np.array([uv[0], uv[1], 1.0], dtype=np.float64)
        extrinsic = np.column_stack((camera.R_w2c, camera.t_w2c))
        rows.extend((x[0] * extrinsic[2] - extrinsic[0], x[1] * extrinsic[2] - extrinsic[1]))
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    homogeneous = vh[-1]
    if abs(homogeneous[3]) < 1e-12:
        return np.full(3, np.nan, dtype=np.float64)
    return homogeneous[:3] / homogeneous[3]


def reprojection_errors(cameras: list[Camera], uvs: np.ndarray, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    errors, depths = [], []
    for camera, uv in zip(cameras, np.asarray(uvs, dtype=np.float64)):
        projected, depth = project(camera, xyz)
        errors.append(float(np.linalg.norm(projected[0] - uv)))
        depths.append(float(depth[0]))
    return np.asarray(errors), np.asarray(depths)


def triangulation_angle_deg(cameras: list[Camera], xyz: np.ndarray) -> float:
    if len(cameras) < 2:
        return 0.0
    rays = []
    for camera in cameras:
        ray = np.asarray(xyz) - camera.center_world
        norm = np.linalg.norm(ray)
        if norm < 1e-12:
            return 0.0
        rays.append(ray / norm)
    angles = []
    for first, second in itertools.combinations(rays, 2):
        angles.append(math.degrees(math.acos(float(np.clip(first @ second, -1.0, 1.0)))))
    # The best-supported baseline determines whether the point is well conditioned.
    return max(angles, default=0.0)


def _refine(cameras: list[Camera], uvs: np.ndarray, initial: np.ndarray, loss: str, max_nfev: int) -> np.ndarray:
    if least_squares is None:
        return initial

    def residual(xyz: np.ndarray) -> np.ndarray:
        values = []
        for camera, uv in zip(cameras, uvs):
            projected, _ = project(camera, xyz)
            values.extend(projected[0] - uv)
        return np.asarray(values)

    result = least_squares(residual, initial, method="trf", loss=loss, max_nfev=max_nfev)
    return result.x if result.success and np.all(np.isfinite(result.x)) else initial


def robust_triangulate(
    cameras: list[Camera],
    uvs: np.ndarray,
    *,
    min_views: int = 2,
    max_reprojection_error_px: float = 3.0,
    min_angle_deg: float = 1.0,
    robust_loss: str = "huber",
    max_refinement_iterations: int = 50,
) -> TriangulationResult:
    count = len(cameras)
    empty = TriangulationResult(
        valid=False,
        xyz=np.full(3, np.nan),
        inliers=np.zeros(count, dtype=bool),
        errors=np.full(count, np.nan),
        rmse=float("nan"),
        angle_deg=0.0,
        confidence=0.0,
    )
    if count < min_views:
        empty.reason = "insufficient_views"
        return empty
    uvs = np.asarray(uvs, dtype=np.float64)
    best: tuple[int, float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for pair in itertools.combinations(range(count), 2):
        xyz = triangulate_dlt([cameras[i] for i in pair], uvs[list(pair)])
        if not np.all(np.isfinite(xyz)):
            continue
        errors, depths = reprojection_errors(cameras, uvs, xyz)
        inliers = (errors <= max_reprojection_error_px) & (depths > 0)
        support = int(inliers.sum())
        if support < min_views:
            continue
        angle = triangulation_angle_deg([cameras[i] for i in np.flatnonzero(inliers)], xyz)
        if angle < min_angle_deg:
            continue
        robust_cost = float(np.median(errors[inliers]))
        candidate = (support, -robust_cost, xyz, inliers, errors)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        empty.reason = "no_valid_ransac_candidate"
        return empty

    inliers = best[3].copy()
    xyz = best[2].copy()
    while int(inliers.sum()) >= min_views:
        ids = np.flatnonzero(inliers)
        subset_cameras = [cameras[i] for i in ids]
        subset_uvs = uvs[ids]
        xyz = triangulate_dlt(subset_cameras, subset_uvs)
        xyz = _refine(subset_cameras, subset_uvs, xyz, robust_loss, max_refinement_iterations)
        errors, depths = reprojection_errors(cameras, uvs, xyz)
        updated = (errors <= max_reprojection_error_px) & (depths > 0)
        if np.array_equal(updated, inliers):
            break
        if int(updated.sum()) < min_views:
            break
        inliers = updated

    errors, depths = reprojection_errors(cameras, uvs, xyz)
    inliers = inliers & (errors <= max_reprojection_error_px) & (depths > 0)
    ids = np.flatnonzero(inliers)
    if len(ids) < min_views:
        empty.errors = errors
        empty.reason = "insufficient_inliers_after_refinement"
        return empty
    angle = triangulation_angle_deg([cameras[i] for i in ids], xyz)
    if angle < min_angle_deg:
        empty.xyz, empty.inliers, empty.errors = xyz, inliers, errors
        empty.angle_deg, empty.reason = angle, "small_triangulation_angle"
        return empty
    rmse = float(np.sqrt(np.mean(errors[inliers] ** 2)))
    reproj_score = math.exp(-(rmse**2) / (2 * max_reprojection_error_px**2))
    view_score = min(len(ids) / 4.0, 1.0)
    angle_score = min(angle / max(4.0 * min_angle_deg, 1e-6), 1.0)
    confidence = float(np.clip(0.5 * reproj_score + 0.3 * view_score + 0.2 * angle_score, 0.0, 1.0))
    return TriangulationResult(True, xyz, inliers, errors, rmse, angle, confidence, "ok")
