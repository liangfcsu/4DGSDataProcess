"""Constrained same-frame multi-view grouping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import fundamental_matrix, robust_triangulate, sampson_errors
from .schema import Camera, FeatureObservation, PairMatches, SpatialGroup


Node = tuple[int, int]  # (cam_id, feature_id)


@dataclass(slots=True)
class _Component:
    nodes: set[Node]
    cameras: set[int]


class ConstrainedComponents:
    def __init__(self):
        self.parent: dict[Node, Node] = {}
        self.components: dict[Node, _Component] = {}

    def _add(self, node: Node) -> None:
        if node not in self.parent:
            self.parent[node] = node
            self.components[node] = _Component({node}, {node[0]})

    def find(self, node: Node) -> Node:
        self._add(node)
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            parent = self.parent[node]
            self.parent[node] = root
            node = parent
        return root

    def merge(self, first: Node, second: Node) -> bool:
        root_a, root_b = self.find(first), self.find(second)
        if root_a == root_b:
            return True
        component_a, component_b = self.components[root_a], self.components[root_b]
        if component_a.cameras & component_b.cameras:
            return False
        if len(component_a.nodes) < len(component_b.nodes):
            root_a, root_b = root_b, root_a
            component_a, component_b = component_b, component_a
        self.parent[root_b] = root_a
        component_a.nodes.update(component_b.nodes)
        component_a.cameras.update(component_b.cameras)
        del self.components[root_b]
        return True

    def groups(self) -> list[set[Node]]:
        return [component.nodes for component in self.components.values()]


def filter_spatial_matches(
    matches: PairMatches,
    camera_a: Camera,
    camera_b: Camera,
    min_score: float,
    epipolar_threshold_px: float,
) -> PairMatches:
    if len(matches.scores) == 0:
        return matches
    epi = sampson_errors(fundamental_matrix(camera_a, camera_b), matches.uv_a, matches.uv_b)
    keep = (matches.scores >= min_score) & np.isfinite(epi) & (epi <= epipolar_threshold_px)
    return PairMatches(
        matches.feature_ids_a[keep], matches.feature_ids_b[keep],
        matches.uv_a[keep], matches.uv_b[keep], matches.scores[keep],
    )


def build_spatial_groups(
    frame_id: int,
    pair_matches: list[tuple[int, int, PairMatches]],
    feature_data: dict[int, tuple[np.ndarray, np.ndarray]],
    cameras: dict[int, Camera],
    config: dict,
) -> tuple[list[SpatialGroup], dict[str, int]]:
    edges: list[tuple[float, Node, Node]] = []
    for cam_a, cam_b, matches in pair_matches:
        for feature_a, feature_b, score in zip(matches.feature_ids_a, matches.feature_ids_b, matches.scores):
            edges.append((float(score), (cam_a, int(feature_a)), (cam_b, int(feature_b))))
    edges.sort(key=lambda item: (-item[0], item[1], item[2]))
    components = ConstrainedComponents()
    conflicts = 0
    edge_scores: dict[frozenset[Node], float] = {}
    for score, first, second in edges:
        edge_scores[frozenset((first, second))] = score
        if not components.merge(first, second):
            conflicts += 1

    groups: list[SpatialGroup] = []
    rejected_geometry = 0
    rejected_views = 0
    for nodes in components.groups():
        if len(nodes) < int(config["spawn"]["min_seed_views"]):
            rejected_views += 1
            continue
        ordered = sorted(nodes)
        group_cameras = [cameras[cam_id] for cam_id, _ in ordered]
        uvs = np.asarray([feature_data[cam_id][0][feature_id] for cam_id, feature_id in ordered])
        result = robust_triangulate(
            group_cameras,
            uvs,
            min_views=int(config["triangulation"]["min_views"]),
            max_reprojection_error_px=float(config["triangulation"]["max_reprojection_error_px"]),
            min_angle_deg=float(config["triangulation"]["min_angle_deg"]),
            robust_loss=config["triangulation"]["robust_loss"],
            max_refinement_iterations=int(config["triangulation"]["max_refinement_iterations"]),
        )
        if not result.valid or int(result.inliers.sum()) < int(config["spawn"]["min_seed_views"]):
            rejected_geometry += 1
            continue
        observations: dict[int, FeatureObservation] = {}
        for index in np.flatnonzero(result.inliers):
            cam_id, feature_id = ordered[int(index)]
            keypoint_score = float(feature_data[cam_id][1][feature_id])
            incident = [
                value for key, value in edge_scores.items()
                if (cam_id, feature_id) in key
            ]
            spatial_score = float(np.mean(incident)) if incident else keypoint_score
            observations[cam_id] = FeatureObservation(
                frame_id, cam_id, feature_id, uvs[index].astype(np.float32), keypoint_score, spatial_score
            )
        groups.append(SpatialGroup(
            frame_id=frame_id,
            observations=observations,
            xyz=result.xyz,
            reprojection_rmse=result.rmse,
            triangulation_angle_deg=result.angle_deg,
            geometry_confidence=result.confidence,
        ))
    groups.sort(key=lambda group: tuple(group.xyz.tolist()))
    return groups, {
        "candidate_edges": len(edges),
        "merge_conflicts": conflicts,
        "components": len(components.groups()),
        "rejected_views": rejected_views,
        "rejected_geometry": rejected_geometry,
        "accepted_groups": len(groups),
    }

