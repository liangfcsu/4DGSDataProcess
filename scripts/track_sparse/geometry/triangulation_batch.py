"""CUDA-batched RANSAC hypotheses with parallel CPU point refinement."""

from __future__ import annotations

import itertools
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from track_sparse.geometry.geometry import robust_triangulate, triangulation_angle_deg
from track_sparse.core.schema import Camera, TriangulationResult


@dataclass(slots=True)
class TriangulationInput:
    track_id: int
    cameras: list[Camera]
    uvs: np.ndarray


def cuda_is_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, RuntimeError):
        return False


def _dlt_hypotheses(
    inputs: list[TriangulationInput], min_views: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    matrices: list[np.ndarray] = []
    owners: list[int] = []
    ranges: list[tuple[int, int]] = []
    for input_index, item in enumerate(inputs):
        start = len(matrices)
        if len(item.cameras) >= min_views:
            for first, second in itertools.combinations(range(len(item.cameras)), 2):
                rows = []
                for camera_index in (first, second):
                    camera = item.cameras[camera_index]
                    uv = item.uvs[camera_index]
                    normalized = np.linalg.inv(camera.K) @ np.array([uv[0], uv[1], 1.0])
                    extrinsic = np.column_stack((camera.R_w2c, camera.t_w2c))
                    rows.extend((
                        normalized[0] * extrinsic[2] - extrinsic[0],
                        normalized[1] * extrinsic[2] - extrinsic[1],
                    ))
                matrix = np.asarray(rows, dtype=np.float32)
                if np.all(np.isfinite(matrix)):
                    matrices.append(matrix)
                    owners.append(input_index)
        ranges.append((start, len(matrices)))
    return (
        np.asarray(matrices, dtype=np.float32).reshape(-1, 4, 4),
        np.asarray(owners, dtype=np.int64),
        ranges,
    )


def _padded_observations(
    inputs: list[TriangulationInput], maximum_views: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projections = np.zeros((len(inputs), maximum_views, 3, 4), dtype=np.float32)
    uvs = np.zeros((len(inputs), maximum_views, 2), dtype=np.float32)
    mask = np.zeros((len(inputs), maximum_views), dtype=bool)
    for index, item in enumerate(inputs):
        count = len(item.cameras)
        if count:
            projections[index, :count] = np.asarray([camera.P for camera in item.cameras])
            uvs[index, :count] = item.uvs
            mask[index, :count] = True
    return projections, uvs, mask


def _cuda_refine(
    inputs: list[TriangulationInput],
    initials: list[np.ndarray | None],
    projections: np.ndarray,
    target_uvs: np.ndarray,
    observation_mask: np.ndarray,
    config: dict,
) -> dict[int, TriangulationResult]:
    """Batched robust Gauss-Newton over padded per-track observations."""
    import torch

    tri = config["triangulation"]
    device = torch.device("cuda")
    count = len(inputs)
    initial_values = np.zeros((count, 3), dtype=np.float32)
    active_values = np.zeros(count, dtype=bool)
    for index, initial in enumerate(initials):
        if initial is not None and np.all(np.isfinite(initial)):
            initial_values[index] = initial
            active_values[index] = True
    with torch.inference_mode():
        xyz = torch.as_tensor(initial_values, device=device)
        active = torch.as_tensor(active_values, device=device)
        P = torch.as_tensor(projections, device=device)
        uv = torch.as_tensor(target_uvs, device=device)
        mask = torch.as_tensor(observation_mask, device=device)
        identity = torch.eye(3, device=device)[None]
        minimum = int(tri["min_views"])
        threshold = float(tri["max_reprojection_error_px"])
        loss = str(tri["robust_loss"])
        iterations = int(tri["max_refinement_iterations"])
        for _ in range(iterations):
            xyz_h = torch.cat((xyz, torch.ones((count, 1), device=device)), dim=1)
            projected_h = torch.einsum("tmij,tj->tmi", P, xyz_h)
            depth = projected_h[:, :, 2]
            safe_depth = torch.where(
                depth.abs() > 1.0e-8, depth, torch.full_like(depth, 1.0e-8)
            )
            predicted = projected_h[:, :, :2] / safe_depth[:, :, None]
            residual = predicted - uv
            errors = torch.linalg.vector_norm(residual, dim=2)
            accepted = (
                mask & active[:, None] & (depth > 0)
                & torch.isfinite(errors) & (errors <= threshold)
            )
            normalized = errors / max(threshold, 1.0e-8)
            if loss == "huber":
                robust = torch.where(
                    normalized <= 1.0, torch.ones_like(normalized),
                    1.0 / normalized.clamp(min=1.0e-8),
                )
            elif loss == "cauchy":
                robust = 1.0 / (1.0 + normalized.square())
            elif loss == "soft_l1":
                robust = 1.0 / torch.sqrt(1.0 + normalized.square())
            else:
                robust = torch.ones_like(normalized)
            weights = torch.where(accepted, robust, torch.zeros_like(robust))
            p0, p1, p2 = P[:, :, 0, :3], P[:, :, 1, :3], P[:, :, 2, :3]
            h0, h1 = projected_h[:, :, 0], projected_h[:, :, 1]
            denominator = safe_depth.square()[:, :, None]
            jacobian0 = (p0 * safe_depth[:, :, None] - h0[:, :, None] * p2) / denominator
            jacobian1 = (p1 * safe_depth[:, :, None] - h1[:, :, None] * p2) / denominator
            jacobian = torch.nan_to_num(torch.stack((jacobian0, jacobian1), dim=2))
            safe_residual = torch.nan_to_num(residual)
            normal = torch.einsum("tmki,tmkj,tm->tij", jacobian, jacobian, weights)
            gradient = torch.einsum("tmki,tmk,tm->ti", jacobian, safe_residual, weights)
            trace = normal.diagonal(dim1=1, dim2=2).sum(dim=1) / 3.0
            damping = trace.clamp(min=1.0) * 1.0e-8
            normal = normal + damping[:, None, None] * identity
            delta = torch.linalg.solve(normal, -gradient[:, :, None])[:, :, 0]
            supported = accepted.sum(dim=1) >= minimum
            update = active & supported & torch.isfinite(delta).all(dim=1)
            xyz = xyz + torch.where(update[:, None], delta, torch.zeros_like(delta))
            if not bool(update.any().item()):
                break
            if float(torch.linalg.vector_norm(delta[update], dim=1).max().item()) < 1.0e-7:
                break

        xyz_h = torch.cat((xyz, torch.ones((count, 1), device=device)), dim=1)
        projected_h = torch.einsum("tmij,tj->tmi", P, xyz_h)
        depth = projected_h[:, :, 2]
        safe_depth = torch.where(
            depth.abs() > 1.0e-8, depth, torch.full_like(depth, 1.0e-8)
        )
        predicted = projected_h[:, :, :2] / safe_depth[:, :, None]
        errors = torch.linalg.vector_norm(predicted - uv, dim=2)
        inliers = (
            mask & active[:, None] & (depth > 0)
            & torch.isfinite(errors) & (errors <= threshold)
        )
        xyz_values = xyz.cpu().numpy().astype(np.float64)
        error_values = errors.cpu().numpy().astype(np.float64)
        inlier_values = inliers.cpu().numpy()

    output: dict[int, TriangulationResult] = {}
    for index, item in enumerate(inputs):
        if not active_values[index]:
            continue
        view_count = len(item.cameras)
        point_errors = error_values[index, :view_count]
        point_inliers = inlier_values[index, :view_count]
        accepted_cameras = [
            item.cameras[camera_index]
            for camera_index in np.flatnonzero(point_inliers)
        ]
        angle = triangulation_angle_deg(accepted_cameras, xyz_values[index])
        valid = len(accepted_cameras) >= int(tri["min_views"]) and angle >= float(tri["min_angle_deg"])
        if valid:
            rmse = float(np.sqrt(np.mean(point_errors[point_inliers] ** 2)))
            reproj_score = math.exp(-(rmse**2) / (2 * threshold**2))
            view_score = min(len(accepted_cameras) / 4.0, 1.0)
            angle_score = min(angle / max(4.0 * float(tri["min_angle_deg"]), 1e-6), 1.0)
            confidence = float(np.clip(
                0.5 * reproj_score + 0.3 * view_score + 0.2 * angle_score, 0.0, 1.0
            ))
            reason = "ok"
        else:
            rmse, confidence = float("nan"), 0.0
            reason = "small_triangulation_angle" if len(accepted_cameras) >= int(tri["min_views"]) else "insufficient_inliers_after_refinement"
        output[item.track_id] = TriangulationResult(
            valid, xyz_values[index], point_inliers, point_errors,
            rmse, angle, confidence, reason,
        )
    return output


def triangulate_cuda_batch(
    inputs: list[TriangulationInput], config: dict
) -> tuple[dict[int, TriangulationResult], dict[str, int | float]]:
    """Triangulate one frame; all pair hypotheses are evaluated on CUDA."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA triangulation requested but torch.cuda.is_available() is false")
    tri = config["triangulation"]
    performance = config["performance"]
    min_views = int(tri["min_views"])
    threshold = float(tri["max_reprojection_error_px"])
    started = time.perf_counter()
    matrices, owners, ranges = _dlt_hypotheses(inputs, min_views)
    if not len(matrices):
        return {}, {
            "cuda_triangulation_batches": 0,
            "cuda_hypotheses": 0,
            "cuda_triangulation_seconds": time.perf_counter() - started,
        }
    maximum_views = max((len(item.cameras) for item in inputs), default=0)
    projections, target_uvs, observation_mask = _padded_observations(inputs, maximum_views)
    device = torch.device("cuda")
    batch_size = int(performance["gpu_hypothesis_batch_size"])
    hypothesis_xyz = np.full((len(matrices), 3), np.nan, dtype=np.float64)
    hypothesis_support = np.zeros(len(matrices), dtype=np.int16)
    hypothesis_cost = np.full(len(matrices), np.inf, dtype=np.float32)
    batch_count = 0
    with torch.inference_mode():
        all_projections = torch.as_tensor(projections, device=device)
        all_target_uvs = torch.as_tensor(target_uvs, device=device)
        all_observation_masks = torch.as_tensor(observation_mask, device=device)
        for start in range(0, len(matrices), batch_size):
            end = min(start + batch_size, len(matrices))
            owner = torch.as_tensor(owners[start:end], device=device, dtype=torch.long)
            A = torch.as_tensor(matrices[start:end], device=device)
            _, _, Vh = torch.linalg.svd(A, full_matrices=False)
            homogeneous = Vh[:, -1]
            denominator = homogeneous[:, 3]
            finite = torch.isfinite(homogeneous).all(dim=1) & (denominator.abs() > 1.0e-8)
            xyz = homogeneous[:, :3] / torch.where(
                denominator.abs() > 1.0e-8, denominator, torch.ones_like(denominator)
            )[:, None]
            xyz_h = torch.cat((xyz, torch.ones((len(xyz), 1), device=device)), dim=1)
            P = all_projections[owner]
            uv = all_target_uvs[owner]
            mask = all_observation_masks[owner]
            projected_h = torch.einsum("hmij,hj->hmi", P, xyz_h)
            depth = projected_h[:, :, 2]
            safe_depth = torch.where(
                depth.abs() > 1.0e-8, depth,
                torch.full_like(depth, 1.0e-8),
            )
            projected = projected_h[:, :, :2] / safe_depth[:, :, None]
            errors = torch.linalg.vector_norm(projected - uv, dim=2)
            valid_projection = mask & (depth > 0) & torch.isfinite(errors)
            inliers = valid_projection & (errors <= threshold) & finite[:, None]
            support = inliers.sum(dim=1)
            penalty = torch.full_like(errors, threshold * 4.0)
            capped = torch.where(valid_projection, errors.clamp(max=threshold * 4.0), penalty)
            cost = torch.where(mask, capped, torch.zeros_like(capped)).sum(dim=1)
            hypothesis_xyz[start:end] = xyz.cpu().numpy().astype(np.float64)
            hypothesis_support[start:end] = support.cpu().numpy().astype(np.int16)
            hypothesis_cost[start:end] = cost.cpu().numpy()
            batch_count += 1

    initials: list[np.ndarray | None] = []
    for start, end in ranges:
        if start == end:
            initials.append(None)
            continue
        indices = list(range(start, end))
        best = max(indices, key=lambda index: (
            int(hypothesis_support[index]), -float(hypothesis_cost[index])
        ))
        initials.append(
            hypothesis_xyz[best]
            if int(hypothesis_support[best]) >= min_views else None
        )

    results = _cuda_refine(
        inputs, initials, projections, target_uvs, observation_mask, config
    )
    fallback_inputs = [
        item for item, initial in zip(inputs, initials) if initial is None
    ]

    def finish(item: TriangulationInput) -> TriangulationResult:
        keywords = {
            "min_views": min_views,
            "max_reprojection_error_px": threshold,
            "min_angle_deg": float(tri["min_angle_deg"]),
            "robust_loss": tri["robust_loss"],
            "max_refinement_iterations": int(tri["max_refinement_iterations"]),
        }
        return robust_triangulate(item.cameras, item.uvs, **keywords)

    workers = max(1, int(performance["cpu_workers"]))
    if fallback_inputs:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tri-fallback") as executor:
            fallback_results = list(executor.map(finish, fallback_inputs))
        results.update({
            item.track_id: result for item, result in zip(fallback_inputs, fallback_results)
        })
    return results, {
        "cuda_triangulation_batches": batch_count,
        "cuda_hypotheses": len(matrices),
        "cuda_triangulation_seconds": time.perf_counter() - started,
    }
