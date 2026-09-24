"""Unsupervised local motion grouping for dynamic sparse trajectories."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from track_sparse.core.schema import MotionClass, Track

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None


class _DisjointSet:
    def __init__(self, values: list[int]):
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            next_value = self.parent[value]
            self.parent[value] = root
            value = next_value
        return root

    def union(self, first: int, second: int) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[max(first_root, second_root)] = min(first_root, second_root)


def _positions(track: Track) -> dict[int, np.ndarray]:
    return {
        frame_id: sample.output_xyz
        for frame_id, sample in track.samples.items()
        if sample.valid_3d and np.all(np.isfinite(sample.output_xyz))
    }


def _motion_residual(
    first_positions: dict[int, np.ndarray],
    second_positions: dict[int, np.ndarray],
    minimum_overlap: int,
) -> tuple[float, int]:
    """Compare pairwise-distance changes, invariant to shared rigid motion."""
    common = sorted(set(first_positions) & set(second_positions))
    if len(common) < minimum_overlap:
        return float("inf"), len(common)
    relative = np.asarray(
        [first_positions[frame] - second_positions[frame] for frame in common],
        dtype=np.float64,
    )
    # Pairwise distance is invariant to a shared rigid rotation and translation.
    # Its robust temporal spread therefore measures violation of local rigidity.
    distances = np.linalg.norm(relative, axis=1)
    residuals = np.abs(distances - np.median(distances))
    return float(np.median(residuals)), len(common)


def _fit_rigid(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    U, _, Vt = np.linalg.svd(covariance)
    rotation = Vt.T @ U.T
    if np.linalg.det(rotation) < 0:
        Vt[-1] *= -1
        rotation = Vt.T @ U.T
    translation = target_center - rotation @ source_center
    predicted = (rotation @ source.T).T + translation
    residual = float(np.sqrt(np.mean(np.sum((predicted - target) ** 2, axis=1))))
    return rotation, translation, residual


def group_dynamic_tracks(
    tracks: dict[int, Track], scene_scale: float, config: dict
) -> tuple[list[dict], dict[str, int | float]]:
    """Cluster nearby dynamic tracks by common 3D motion and fit group SE(3).

    This is intentionally geometry-only. It does not assume that the scene
    contains a person, nor does it require masks or semantic labels. Articulated
    parts are allowed to form separate local groups.
    """
    section = config["motion_groups"]
    for track in tracks.values():
        track.motion_group_id = -1
    if not bool(section["enabled"]):
        return [], {"motion_group_count": 0, "grouped_dynamic_tracks": 0}

    minimum_frames = int(section["min_track_frames"])
    positions_by_track = {track.track_id: _positions(track) for track in tracks.values()}
    dynamic = [
        track for track in tracks.values()
        if track.motion_class == MotionClass.DYNAMIC
        and len(positions_by_track[track.track_id]) >= minimum_frames
    ]
    if not dynamic:
        return [], {"motion_group_count": 0, "grouped_dynamic_tracks": 0}
    centers = np.asarray([
        np.median(np.asarray(list(positions_by_track[track.track_id].values())), axis=0)
        for track in dynamic
    ])
    radius = max(float(scene_scale) * float(section["search_radius_ratio"]), 1e-9)
    maximum_neighbors = int(section["max_neighbors"])
    pairs: set[tuple[int, int]] = set()
    if cKDTree is not None:
        tree = cKDTree(centers)
        for index, center in enumerate(centers):
            distances, neighbours = tree.query(
                center, k=min(maximum_neighbors + 1, len(dynamic)), distance_upper_bound=radius
            )
            for distance, neighbour in zip(np.atleast_1d(distances), np.atleast_1d(neighbours)):
                neighbour = int(neighbour)
                if math.isfinite(float(distance)) and neighbour < len(dynamic) and neighbour != index:
                    pairs.add(tuple(sorted((index, neighbour))))
    else:  # pragma: no cover
        for first in range(len(dynamic)):
            distances = np.linalg.norm(centers - centers[first], axis=1)
            neighbours = np.argsort(distances)[1:maximum_neighbors + 1]
            for second in neighbours:
                if distances[second] <= radius:
                    pairs.add(tuple(sorted((first, int(second)))))

    disjoint = _DisjointSet([track.track_id for track in dynamic])
    residual_limit = float(scene_scale) * float(section["motion_residual_ratio"])
    tested = accepted = 0
    for first_index, second_index in sorted(pairs):
        residual, overlap = _motion_residual(
            positions_by_track[dynamic[first_index].track_id],
            positions_by_track[dynamic[second_index].track_id],
            int(section["min_overlap_frames"]),
        )
        tested += 1
        if overlap >= int(section["min_overlap_frames"]) and residual <= residual_limit:
            disjoint.union(dynamic[first_index].track_id, dynamic[second_index].track_id)
            accepted += 1

    components: dict[int, list[Track]] = defaultdict(list)
    for track in dynamic:
        components[disjoint.find(track.track_id)].append(track)
    components = {
        root: members for root, members in components.items()
        if len(members) >= int(section["min_group_size"])
    }

    records: list[dict] = []
    rigid_frames = 0
    rejected_frames = 0
    grouped_tracks = 0
    for group_id, members in enumerate(
        sorted(components.values(), key=lambda values: min(item.track_id for item in values))
    ):
        for track in members:
            track.motion_group_id = group_id
        grouped_tracks += len(members)
        frame_counts: dict[int, int] = defaultdict(int)
        member_positions = {
            track.track_id: positions_by_track[track.track_id] for track in members
        }
        for positions in member_positions.values():
            for frame_id in positions:
                frame_counts[frame_id] += 1
        reference_frame = max(frame_counts, key=lambda frame: (frame_counts[frame], -frame))
        reference = {
            track_id: positions[reference_frame]
            for track_id, positions in member_positions.items() if reference_frame in positions
        }
        transforms = []
        for frame_id in sorted(frame_counts):
            common_ids = sorted(
                track_id for track_id in reference
                if frame_id in member_positions[track_id]
            )
            if len(common_ids) < int(section["rigid_fit_min_points"]):
                continue
            source = np.asarray([reference[track_id] for track_id in common_ids])
            target = np.asarray([member_positions[track_id][frame_id] for track_id in common_ids])
            try:
                rotation, translation, residual = _fit_rigid(source, target)
            except np.linalg.LinAlgError:
                rejected_frames += 1
                continue
            accepted_fit = residual <= float(scene_scale) * float(section["max_rigid_residual_ratio"])
            if accepted_fit:
                rigid_frames += 1
                blend = float(section["rigid_blend"])
                if blend > 0:
                    for track_id in common_ids:
                        sample = tracks[track_id].samples[frame_id]
                        predicted = rotation @ reference[track_id] + translation
                        sample.optimized_xyz = (
                            (1.0 - blend) * sample.output_xyz + blend * predicted
                        )
            else:
                rejected_frames += 1
            transforms.append({
                "frame_id": frame_id,
                "accepted": accepted_fit,
                "R": rotation.tolist(),
                "t": translation.tolist(),
                "rmse": residual,
                "point_count": len(common_ids),
            })
        records.append({
            "motion_group_id": group_id,
            "reference_frame": reference_frame,
            "track_ids": sorted(track.track_id for track in members),
            "transforms": transforms,
        })
    return records, {
        "motion_group_count": len(records),
        "grouped_dynamic_tracks": grouped_tracks,
        "motion_pair_candidates": tested,
        "motion_pair_edges": accepted,
        "rigid_group_frames": rigid_frames,
        "rigid_group_frame_rejections": rejected_frames,
    }
