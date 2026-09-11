"""Sequential global Track-ID construction and lifecycle management."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .geometry import robust_triangulate
from .schema import Observation, SpatialGroup, Track, TrackSample, TrackState
from .temporal_tracker import TemporalTracker


class TrackManager:
    def __init__(self, cameras: dict, frames: list[int], config: dict, temporal: TemporalTracker, sparse_verifier=None):
        self.cameras = cameras
        self.frames = frames
        self.config = config
        self.temporal = temporal
        self.sparse_verifier = sparse_verifier
        self.tracks: dict[int, Track] = {}
        self.next_track_id = 1
        self.stats = defaultdict(int)

    def _propagate_to_frame(self, frame_id: int) -> None:
        offsets = sorted({int(value) for value in self.config["temporal"]["offsets"]})
        proposed: list[tuple[float, Track, Observation]] = []
        for track in list(self.tracks.values()):
            if track.state == TrackState.ENDED or frame_id <= track.birth_frame:
                continue
            for cam_id in self.cameras:
                if (frame_id, cam_id) in track.observations:
                    continue
                candidates: list[Observation] = []
                for offset in offsets:
                    source = track.observations.get((frame_id - offset, cam_id))
                    if source is None or not source.visible:
                        continue
                    candidate = self.temporal.propagate(track.track_id, source, frame_id)
                    if candidate is not None:
                        # Prefer high confidence and then the shorter temporal edge.
                        candidate.tracker_confidence *= 1.0 / (1.0 + 0.1 * (offset - 1))
                        candidates.append(candidate)
                if candidates:
                    best = max(candidates, key=lambda item: (item.tracker_confidence, item.source == "temporal"))
                    proposed.append((best.tracker_confidence, track, best))
        # Resolve offset-based collisions globally: a target feature belongs to one Track ID.
        reserved_features: set[tuple[int, int]] = set()
        reserved_track_cameras: set[tuple[int, int]] = set()
        for _, track, observation in sorted(proposed, key=lambda item: -item[0]):
            feature_key = (observation.cam_id, observation.feature_id)
            track_camera_key = (track.track_id, observation.cam_id)
            if feature_key in reserved_features or track_camera_key in reserved_track_cameras:
                self.stats["temporal_conflicts"] += 1
                continue
            reserved_features.add(feature_key)
            reserved_track_cameras.add(track_camera_key)
            track.observations[(frame_id, observation.cam_id)] = observation

    def _duplicate_candidate(self, group: SpatialGroup, consumed: set[int]) -> Track | None:
        radius = float(self.config["spawn"]["duplicate_radius_px"])
        min_views = int(self.config["spawn"]["min_duplicate_views"])
        candidates: list[tuple[int, float, int, Track]] = []
        for track in self.tracks.values():
            if track.track_id in consumed or track.state == TrackState.ENDED:
                continue
            agreements, exact, distances = 0, 0, []
            for cam_id, seed in group.observations.items():
                existing = track.observations.get((group.frame_id, cam_id))
                if existing is None:
                    continue
                distance = float(np.linalg.norm(np.array([existing.u, existing.v]) - seed.uv))
                if existing.feature_id == seed.feature_id or distance <= radius:
                    agreements += 1
                    exact += int(existing.feature_id == seed.feature_id)
                    distances.append(distance)
            if agreements >= min_views:
                candidates.append((agreements, exact, -float(np.mean(distances)), track))
        return max(candidates, key=lambda item: item[:3])[3] if candidates else None

    @staticmethod
    def _seed_observation(track_id: int, seed, source: str = "seed") -> Observation:
        return Observation(
            track_id=track_id,
            frame_id=seed.frame_id,
            cam_id=seed.cam_id,
            feature_id=seed.feature_id,
            u=float(seed.uv[0]),
            v=float(seed.uv[1]),
            visible=True,
            tracker_confidence=float(seed.feature_score),
            spatial_confidence=float(seed.spatial_confidence),
            source=source,
        )

    def _attach_group(self, track: Track, group: SpatialGroup, source: str, seeds=None) -> None:
        for cam_id, seed in (seeds if seeds is not None else group.observations).items():
            key = (group.frame_id, cam_id)
            candidate = self._seed_observation(track.track_id, seed, source)
            existing = track.observations.get(key)
            if existing is None or candidate.spatial_confidence > existing.spatial_confidence:
                track.observations[key] = candidate

    def _spawn_and_merge(self, groups: list[SpatialGroup]) -> None:
        consumed: set[int] = set()
        for group in sorted(groups, key=lambda item: -item.geometry_confidence):
            track = self._duplicate_candidate(group, consumed)
            if track is not None:
                self._attach_group(track, group, "reconnect" if track.state in {TrackState.LOST, TrackState.OCCLUDED} else "seed")
                consumed.add(track.track_id)
                self.stats["duplicate_groups_suppressed"] += 1
                continue
            occupied = {
                (observation.cam_id, observation.feature_id)
                for existing_track in self.tracks.values()
                for (existing_frame, _), observation in existing_track.observations.items()
                if existing_frame == group.frame_id
            }
            available = {
                cam_id: seed for cam_id, seed in group.observations.items()
                if (cam_id, seed.feature_id) not in occupied
            }
            if len(available) < int(self.config["spawn"]["min_seed_views"]):
                self.stats["uncertain_spawn_conflicts"] += 1
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            track = Track(track_id=track_id, birth_frame=group.frame_id, color_rgb=group.color_rgb.copy())
            self._attach_group(track, group, "seed", available)
            self.tracks[track_id] = track
            consumed.add(track_id)
            self.stats["new_tracks"] += 1

    def _triangulate_track(self, track: Track, frame_id: int) -> TrackSample:
        observations = sorted(
            (obs for (frame, _), obs in track.observations.items() if frame == frame_id and obs.visible),
            key=lambda item: item.cam_id,
        )
        cameras = [self.cameras[obs.cam_id] for obs in observations]
        uvs = np.asarray([[obs.u, obs.v] for obs in observations], dtype=np.float64)
        tri = self.config["triangulation"]
        result = robust_triangulate(
            cameras,
            uvs,
            min_views=int(tri["min_views"]),
            max_reprojection_error_px=float(tri["max_reprojection_error_px"]),
            min_angle_deg=float(tri["min_angle_deg"]),
            robust_loss=tri["robust_loss"],
            max_refinement_iterations=int(tri["max_refinement_iterations"]),
        )
        for index, observation in enumerate(observations):
            if index < len(result.errors):
                observation.reprojection_error = float(result.errors[index])
                observation.is_inlier = bool(result.valid and result.inliers[index])
        sample = TrackSample(
            track_id=track.track_id,
            frame_id=frame_id,
            xyz=result.xyz if result.valid else np.full(3, np.nan),
            valid_3d=result.valid,
            num_visible_views=len(observations),
            num_inlier_views=int(result.inliers.sum()) if result.valid else 0,
            reprojection_rmse=result.rmse,
            min_triangulation_angle_deg=result.angle_deg,
            geometry_confidence=result.confidence,
        )
        if result.valid and self.sparse_verifier is not None:
            support = self.sparse_verifier.score(frame_id, result.xyz)
            if support is not None:
                distance, confidence = support
                weight = float(self.config["sparse_verify"]["confidence_weight"])
                sample.sparse_support_distance = distance
                sample.geometry_confidence = float((1.0 - weight) * sample.geometry_confidence + weight * confidence)
        return sample

    def _update_state(self, track: Track, sample: TrackSample, frame_id: int) -> None:
        previous = track.state
        if sample.valid_3d:
            track.last_valid_frame = frame_id
            track.valid_frame_count += 1
            track.current_valid_run += 1
            track.longest_valid_run = max(track.longest_valid_run, track.current_valid_run)
            track.missing_count = 0
            if previous in {TrackState.OCCLUDED, TrackState.LOST}:
                track.recovery_count += 1
                self.stats["recoveries"] += 1
            if track.longest_valid_run >= int(self.config["track"]["tentative_frames"]):
                track.state = TrackState.ACTIVE
            else:
                track.state = TrackState.TENTATIVE
        else:
            track.current_valid_run = 0
            reference = track.last_valid_frame if track.last_valid_frame >= 0 else track.birth_frame
            gap = frame_id - reference
            track.missing_count = max(0, gap)
            if gap <= int(self.config["track"]["max_occlusion_gap"]):
                track.state = TrackState.OCCLUDED
            elif gap <= int(self.config["track"]["max_lost_gap"]):
                track.state = TrackState.LOST
            else:
                track.state = TrackState.ENDED
                self.stats["ended_tracks"] += 1
        sample.state = track.state
        track.samples[frame_id] = sample

    def run(self, groups_by_frame: dict[int, list[SpatialGroup]]) -> dict[int, Track]:
        for frame_id in self.frames:
            self._propagate_to_frame(frame_id)
            self._spawn_and_merge(groups_by_frame.get(frame_id, []))
            for track in list(self.tracks.values()):
                if frame_id < track.birth_frame or track.state == TrackState.ENDED:
                    continue
                sample = self._triangulate_track(track, frame_id)
                self._update_state(track, sample, frame_id)
        return self.tracks
