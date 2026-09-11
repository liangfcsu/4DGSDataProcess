"""Build a fixed camera-neighbour graph from calibration and scene points."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

from .geometry import project
from .schema import Camera, Point3D


def _scene_scale(cameras: dict[int, Camera]) -> float:
    distances = [
        float(np.linalg.norm(a.center_world - b.center_world))
        for a, b in itertools.combinations(cameras.values(), 2)
    ]
    return float(np.median(distances)) if distances else 1.0


def build_camera_graph(
    cameras: dict[int, Camera],
    config: dict,
    reference_points: dict[int, Point3D] | None = None,
) -> dict:
    cam_ids = sorted(cameras)
    top_k = min(int(config["top_k"]), max(len(cam_ids) - 1, 0))
    scene_scale = _scene_scale(cameras)
    points = np.asarray([point.xyz for point in (reference_points or {}).values()], dtype=np.float64)
    limit = int(config.get("overlap_sample_points", 20000))
    if len(points) > limit:
        indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
        points = points[indices]

    visibility: dict[int, np.ndarray] = {}
    if len(points):
        for cam_id in cam_ids:
            camera = cameras[cam_id]
            uv, depth = project(camera, points)
            visibility[cam_id] = (
                (depth > 0)
                & (uv[:, 0] >= 0)
                & (uv[:, 0] < camera.width)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < camera.height)
            )

    pair_records: dict[tuple[int, int], dict] = {}
    for cam_a, cam_b in itertools.combinations(cam_ids, 2):
        first, second = cameras[cam_a], cameras[cam_b]
        baseline = float(np.linalg.norm(first.center_world - second.center_world))
        axis_a = first.R_w2c.T @ np.array([0.0, 0.0, 1.0])
        axis_b = second.R_w2c.T @ np.array([0.0, 0.0, 1.0])
        view_angle = math.degrees(math.acos(float(np.clip(axis_a @ axis_b, -1.0, 1.0))))
        if visibility:
            visible_a, visible_b = visibility[cam_a], visibility[cam_b]
            denominator = max(1, min(int(visible_a.sum()), int(visible_b.sum())))
            overlap = float(np.count_nonzero(visible_a & visible_b) / denominator)
        else:
            # With no reference cloud, similar optical axes are the least surprising proxy.
            overlap = float(max(0.0, math.cos(math.radians(view_angle))))
        baseline_ratio = baseline / max(scene_scale, 1e-12)
        baseline_score = min(baseline_ratio / 0.25, 1.0) * math.exp(-max(0.0, baseline_ratio - 1.5))
        angle_score = max(0.0, math.cos(math.radians(view_angle / 2.0)))
        score = 0.60 * overlap + 0.20 * angle_score + 0.20 * baseline_score
        eligible = (
            view_angle <= float(config["max_view_angle_deg"])
            and baseline_ratio >= float(config["min_baseline_ratio"])
            and (not visibility or overlap >= float(config["min_overlap"]))
        )
        pair_records[(cam_a, cam_b)] = {
            "cam_a": cam_a,
            "cam_b": cam_b,
            "score": score,
            "overlap": overlap,
            "baseline": baseline,
            "baseline_ratio": baseline_ratio,
            "view_angle_deg": view_angle,
            "eligible": eligible,
            "selected": False,
            "reason": "filtered",
        }

    # Per-camera top-K followed by undirected symmetrisation.
    selected: set[tuple[int, int]] = set()
    for cam_id in cam_ids:
        candidates = [
            record for pair, record in pair_records.items()
            if cam_id in pair and record["eligible"]
        ]
        candidates.sort(key=lambda item: (-item["score"], item["cam_a"], item["cam_b"]))
        for record in candidates[:top_k]:
            selected.add((record["cam_a"], record["cam_b"]))

    # If strict overlap filters isolate a camera, retain its best geometrically plausible edge.
    for cam_id in cam_ids:
        if any(cam_id in pair for pair in selected):
            continue
        candidates = [record for pair, record in pair_records.items() if cam_id in pair]
        candidates.sort(key=lambda item: (-item["score"], item["cam_a"], item["cam_b"]))
        if candidates:
            record = candidates[0]
            selected.add((record["cam_a"], record["cam_b"]))
            record["reason"] = "connectivity_fallback"

    for pair in selected:
        record = pair_records[pair]
        record["selected"] = True
        if record["reason"] != "connectivity_fallback":
            record["reason"] = "top_k"
    adjacency = {str(cam): [] for cam in cam_ids}
    for cam_a, cam_b in sorted(selected):
        adjacency[str(cam_a)].append(cam_b)
        adjacency[str(cam_b)].append(cam_a)
    for neighbours in adjacency.values():
        neighbours.sort()
    return {
        "scene_scale": scene_scale,
        "reference_point_count": int(len(points)),
        "adjacency": adjacency,
        "edges": [pair_records[pair] for pair in sorted(pair_records) if pair in selected],
        "filtered_edges": int(len(pair_records) - len(selected)),
    }


def save_camera_graph(graph: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_camera_graph(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

