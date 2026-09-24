"""Constrained global assignment of temporal observation-graph edges."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from track_sparse.core.schema import Observation, Track, TrackState

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
except ImportError:  # pragma: no cover
    Bounds = LinearConstraint = milp = None


@dataclass(slots=True)
class AssociationEdge:
    track_id: int
    observation: Observation
    score: float
    source_frame: int


def _conflict_components(edges: list[AssociationEdge]) -> list[list[AssociationEdge]]:
    """Connected components in a sparse track-to-feature bipartite graph."""
    by_track: dict[int, list[int]] = defaultdict(list)
    by_feature: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, edge in enumerate(edges):
        by_track[edge.track_id].append(index)
        by_feature[(edge.observation.cam_id, edge.observation.feature_id)].append(index)
    unseen = set(range(len(edges)))
    result: list[list[AssociationEdge]] = []
    while unseen:
        start = unseen.pop()
        queue = deque([start])
        indices = [start]
        while queue:
            current = queue.popleft()
            edge = edges[current]
            feature = (edge.observation.cam_id, edge.observation.feature_id)
            neighbours = by_track[edge.track_id] + by_feature[feature]
            for neighbour in neighbours:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
                    indices.append(neighbour)
        result.append([edges[index] for index in indices])
    return result


def _solve_quota_component(
    edges: list[AssociationEdge], minimum_votes: int
) -> list[AssociationEdge]:
    """Maximum-weight assignment with an all-or-at-least-N camera quota."""
    tracks = sorted({edge.track_id for edge in edges})
    # A single track component needs no integer solver; just choose its best
    # candidate in each camera and apply the quota.
    if len(tracks) == 1:
        best: dict[int, AssociationEdge] = {}
        for edge in edges:
            cam_id = edge.observation.cam_id
            if cam_id not in best or edge.score > best[cam_id].score:
                best[cam_id] = edge
        return list(best.values()) if len(best) >= minimum_votes else []

    if milp is None:
        selected = []
        used_features: set[tuple[int, int]] = set()
        used_track_cameras: set[tuple[int, int]] = set()
        for edge in sorted(edges, key=lambda item: -item.score):
            feature = (edge.observation.cam_id, edge.observation.feature_id)
            track_camera = (edge.track_id, edge.observation.cam_id)
            if feature not in used_features and track_camera not in used_track_cameras:
                selected.append(edge)
                used_features.add(feature)
                used_track_cameras.add(track_camera)
        counts: dict[int, int] = defaultdict(int)
        for edge in selected:
            counts[edge.track_id] += 1
        return [edge for edge in selected if counts[edge.track_id] >= minimum_votes]

    edge_count = len(edges)
    track_index = {track_id: edge_count + index for index, track_id in enumerate(tracks)}
    variable_count = edge_count + len(tracks)
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    by_feature: dict[tuple[int, int], list[int]] = defaultdict(list)
    by_track_camera: dict[tuple[int, int], list[int]] = defaultdict(list)
    by_track: dict[int, list[int]] = defaultdict(list)
    for edge_index, edge in enumerate(edges):
        by_feature[(edge.observation.cam_id, edge.observation.feature_id)].append(edge_index)
        by_track_camera[(edge.track_id, edge.observation.cam_id)].append(edge_index)
        by_track[edge.track_id].append(edge_index)
    for indices in list(by_feature.values()) + list(by_track_camera.values()):
        row = np.zeros(variable_count, dtype=np.float64)
        row[indices] = 1.0
        rows.append(row)
        lower.append(-np.inf)
        upper.append(1.0)
    for track_id, indices in by_track.items():
        # sum(edges) >= minimum_votes * active_track
        lower_quota = np.zeros(variable_count, dtype=np.float64)
        lower_quota[indices] = 1.0
        lower_quota[track_index[track_id]] = -float(minimum_votes)
        rows.append(lower_quota)
        lower.append(0.0)
        upper.append(np.inf)
        # No edge may survive if the track activation variable is zero.
        activation = np.zeros(variable_count, dtype=np.float64)
        activation[indices] = 1.0
        activation[track_index[track_id]] = -float(len(indices))
        rows.append(activation)
        lower.append(-np.inf)
        upper.append(0.0)

    objective = np.zeros(variable_count, dtype=np.float64)
    objective[:edge_count] = -np.asarray([edge.score for edge in edges])
    try:
        result = milp(
            objective,
            integrality=np.ones(variable_count, dtype=np.int8),
            bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
            constraints=LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper)),
            options={"time_limit": 2.0, "mip_rel_gap": 0.0},
        )
    except (TypeError, ValueError):  # pragma: no cover - old SciPy fallback
        result = None
    if result is None or not result.success or result.x is None:
        # Keep a deterministic safe fallback if HiGHS is unavailable.
        selected = []
        used_features: set[tuple[int, int]] = set()
        used_track_cameras: set[tuple[int, int]] = set()
        for edge in sorted(edges, key=lambda item: (-item.score, item.track_id)):
            feature = (edge.observation.cam_id, edge.observation.feature_id)
            track_camera = (edge.track_id, edge.observation.cam_id)
            if feature not in used_features and track_camera not in used_track_cameras:
                selected.append(edge)
                used_features.add(feature)
                used_track_cameras.add(track_camera)
        counts: dict[int, int] = defaultdict(int)
        for edge in selected:
            counts[edge.track_id] += 1
        return [edge for edge in selected if counts[edge.track_id] >= minimum_votes]
    return [edge for index, edge in enumerate(edges) if result.x[index] >= 0.5]


class ObservationGraphAssociator:
    """Build and optimize all temporal candidates that enter one target frame.

    Unlike the previous score-sorted greedy pass, every collision component is
    solved as a maximum-weight bipartite assignment. Direct two-frame links are
    checked against composed one-frame links when both are available.
    """

    def __init__(self, temporal, cameras: list[int], config: dict):
        self.temporal = temporal
        self.cameras = list(cameras)
        self.config = config

    def _cycle_consistency(
        self,
        track: Track,
        source: Observation,
        target_frame: int,
        target_feature: int,
    ) -> int:
        if not hasattr(self.temporal, "link_feature"):
            return -1
        cam_id = source.cam_id
        gap = target_frame - source.frame_id
        evidence: list[bool] = []

        # SuperGlue matches are mutual, but explicitly verify the reverse edge
        # so alternative temporal backends are held to the same contract.
        reverse = self.temporal.link_feature(cam_id, target_frame, source.frame_id, target_feature)
        if reverse is not None:
            evidence.append(int(reverse[0]) == source.feature_id)

        if gap == 1:
            older = track.observations.get((target_frame - 2, cam_id))
            if older is not None:
                direct = self.temporal.link_feature(
                    cam_id, target_frame - 2, target_frame, older.feature_id
                )
                if direct is not None:
                    evidence.append(int(direct[0]) == target_feature)
        elif gap == 2:
            middle = track.observations.get((target_frame - 1, cam_id))
            if middle is not None:
                composed = self.temporal.link_feature(
                    cam_id, target_frame - 1, target_frame, middle.feature_id
                )
                if composed is not None:
                    evidence.append(int(composed[0]) == target_feature)
        if not evidence:
            return -1
        return int(all(evidence))

    def _candidate_edges(
        self, tracks: dict[int, Track], frame_id: int
    ) -> tuple[list[AssociationEdge], int]:
        section = self.config["association"]
        offsets = sorted({int(value) for value in self.config["temporal"]["offsets"]})
        candidates: dict[tuple[int, int, int], AssociationEdge] = {}
        cycle_rejections = 0
        for track in tracks.values():
            if track.state == TrackState.ENDED or frame_id <= track.birth_frame:
                continue
            for cam_id in self.cameras:
                if (frame_id, cam_id) in track.observations:
                    continue
                for offset in offsets:
                    source = track.observations.get((frame_id - offset, cam_id))
                    if source is None or not source.visible:
                        continue
                    observation = self.temporal.propagate(track.track_id, source, frame_id)
                    if observation is None:
                        continue
                    observation.cycle_consistency = self._cycle_consistency(
                        track, source, frame_id, observation.feature_id
                    )
                    base = observation.association_score or observation.tracker_confidence
                    score = float(base) / (1.0 + float(section["offset_penalty"]) * (offset - 1))
                    if observation.cycle_consistency == 0:
                        if bool(section["reject_cycle_inconsistent"]):
                            cycle_rejections += 1
                            continue
                        score *= float(section["cycle_inconsistent_weight"])
                    elif observation.cycle_consistency == 1:
                        score *= float(section["cycle_consistent_bonus"])
                    observation.association_score = score
                    key = (track.track_id, cam_id, observation.feature_id)
                    edge = AssociationEdge(track.track_id, observation, score, source.frame_id)
                    if key not in candidates or edge.score > candidates[key].score:
                        candidates[key] = edge
        return list(candidates.values()), cycle_rejections

    def associate(self, tracks: dict[int, Track], frame_id: int) -> tuple[list[Observation], dict[str, int]]:
        edges, cycle_rejections = self._candidate_edges(tracks, frame_id)
        selected: list[AssociationEdge] = []
        minimum_votes = int(self.config["temporal"]["min_camera_votes"])
        allow_bridge = bool(self.config["association"]["allow_single_camera_bridge"])
        effective_votes = 1 if allow_bridge else minimum_votes
        components = _conflict_components(edges)
        for component in components:
            selected.extend(_solve_quota_component(component, effective_votes))
        selected_tracks = {edge.track_id for edge in selected}
        candidate_tracks = {edge.track_id for edge in edges}
        candidate_edges_per_track: dict[int, int] = defaultdict(int)
        candidate_cameras_per_track: dict[int, set[int]] = defaultdict(set)
        for edge in edges:
            candidate_edges_per_track[edge.track_id] += 1
            candidate_cameras_per_track[edge.track_id].add(edge.observation.cam_id)
        rejected_votes = sum(
            candidate_edges_per_track[track_id]
            for track_id in candidate_tracks - selected_tracks
            if len(candidate_cameras_per_track[track_id]) < effective_votes
        )
        return [edge.observation for edge in selected], {
            "association_candidates": len(edges),
            "association_components": len(components),
            "association_conflicts": max(0, len(edges) - len(selected)),
            "association_vote_rejections": rejected_votes,
            "association_rejected_tracks": len(candidate_tracks - selected_tracks),
            "association_accepted": len(selected),
            "cycle_inconsistent_rejections": cycle_rejections,
        }
