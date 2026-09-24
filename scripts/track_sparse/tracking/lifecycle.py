"""Sequential global Track-ID construction and lifecycle management."""

from __future__ import annotations

import pickle
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path

import numpy as np

from track_sparse.geometry.geometry import robust_triangulate
from track_sparse.matching.observation_graph import ObservationGraphAssociator
from track_sparse.core.schema import Observation, SpatialGroup, Track, TrackSample, TrackState, TriangulationResult
from track_sparse.matching.temporal_tracker import TemporalTracker
from track_sparse.geometry.triangulation_batch import TriangulationInput, cuda_is_available, triangulate_cuda_batch


LOGGER = logging.getLogger("track_sparse")


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
        self.associator = ObservationGraphAssociator(temporal, sorted(cameras), config)
        requested = str(config["performance"]["triangulation_backend"]).lower()
        if requested not in {"auto", "cpu", "cuda"}:
            raise ValueError("performance.triangulation_backend 必须是 auto/cpu/cuda")
        available = cuda_is_available() if requested != "cpu" else False
        if requested == "cuda" and not available:
            LOGGER.warning("请求 CUDA 三角化，但 CUDA 不可用；回退到多线程 CPU")
        self.triangulation_backend = (
            "cuda" if available and requested in {"auto", "cuda"} else "cpu"
        )
        self.stats["triangulation_backend"] = self.triangulation_backend

    def _propagate_to_frame(self, frame_id: int) -> None:
        observations, stats = self.associator.associate(self.tracks, frame_id)
        for key, value in stats.items():
            self.stats[key] += value
        self.stats["temporal_conflicts"] += stats["association_conflicts"]
        for observation in observations:
            track = self.tracks.get(observation.track_id)
            if track is None:
                continue
            track.observations[(frame_id, observation.cam_id)] = observation

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
        if not groups:
            return
        frame_id = groups[0].frame_id
        feature_owner: dict[tuple[int, int], int] = {}
        radius = float(self.config["spawn"]["duplicate_radius_px"])
        cell_size = max(radius, 1.0e-6)
        spatial_grid: dict[tuple[int, int, int], list[tuple[float, float, int]]] = defaultdict(list)

        def insert(cam_id: int, u: float, v: float, track_id: int) -> None:
            cell = (cam_id, int(np.floor(u / cell_size)), int(np.floor(v / cell_size)))
            spatial_grid[cell].append((u, v, track_id))

        def nearest(cam_id: int, uv: np.ndarray) -> tuple[int, float] | None:
            center_x = int(np.floor(float(uv[0]) / cell_size))
            center_y = int(np.floor(float(uv[1]) / cell_size))
            best: tuple[int, float] | None = None
            for offset_x in (-1, 0, 1):
                for offset_y in (-1, 0, 1):
                    for u, v, track_id in spatial_grid.get(
                        (cam_id, center_x + offset_x, center_y + offset_y), []
                    ):
                        distance = float(np.hypot(float(uv[0]) - u, float(uv[1]) - v))
                        if distance <= radius and (best is None or distance < best[1]):
                            best = (track_id, distance)
            return best

        for track in self.tracks.values():
            if track.state == TrackState.ENDED:
                continue
            for cam_id in self.cameras:
                observation = track.observations.get((frame_id, cam_id))
                if observation is None:
                    continue
                feature_owner[(cam_id, observation.feature_id)] = track.track_id
                insert(cam_id, observation.u, observation.v, track.track_id)
        min_views = int(self.config["spawn"]["min_duplicate_views"])
        for group in sorted(groups, key=lambda item: -item.geometry_confidence):
            votes: dict[int, list[tuple[bool, float]]] = defaultdict(list)
            for cam_id, seed in group.observations.items():
                exact_owner = feature_owner.get((cam_id, seed.feature_id))
                if exact_owner is not None:
                    votes[exact_owner].append((True, 0.0))
                    continue
                proximity = nearest(cam_id, seed.uv)
                if proximity is not None:
                    track_id, distance = proximity
                    votes[track_id].append((False, distance))
            candidates = []
            for track_id, agreements in votes.items():
                track = self.tracks.get(track_id)
                if (
                    track is None or track_id in consumed or track.state == TrackState.ENDED
                    or len(agreements) < min_views
                ):
                    continue
                exact = sum(value[0] for value in agreements)
                mean_distance = float(np.mean([value[1] for value in agreements]))
                candidates.append((len(agreements), exact, -mean_distance, track))
            track = max(candidates, key=lambda item: item[:3])[3] if candidates else None
            if track is not None:
                self._attach_group(track, group, "reconnect" if track.state in {TrackState.LOST, TrackState.OCCLUDED} else "seed")
                consumed.add(track.track_id)
                self.stats["duplicate_groups_suppressed"] += 1
                for cam_id, seed in group.observations.items():
                    feature_owner[(cam_id, seed.feature_id)] = track.track_id
                    insert(cam_id, float(seed.uv[0]), float(seed.uv[1]), track.track_id)
                continue
            occupied = {
                key for key in ((cam_id, seed.feature_id) for cam_id, seed in group.observations.items())
                if key in feature_owner
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
            for cam_id, seed in available.items():
                feature_owner[(cam_id, seed.feature_id)] = track_id
                insert(cam_id, float(seed.uv[0]), float(seed.uv[1]), track_id)

    @staticmethod
    def _frame_observations(track: Track, frame_id: int) -> list[Observation]:
        return sorted(
            (obs for (frame, _), obs in track.observations.items() if frame == frame_id and obs.visible),
            key=lambda item: item.cam_id,
        )

    def _sample_from_result(
        self,
        track: Track,
        frame_id: int,
        observations: list[Observation],
        result: TriangulationResult,
    ) -> TrackSample:
        cameras = [self.cameras[obs.cam_id] for obs in observations]
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

    def _triangulate_track(self, track: Track, frame_id: int) -> TrackSample:
        observations = self._frame_observations(track, frame_id)
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
        return self._sample_from_result(track, frame_id, observations, result)

    def _triangulate_frame(
        self,
        tracks: list[Track],
        frame_id: int,
        executor: ThreadPoolExecutor | None,
    ) -> list[TrackSample]:
        started = time.perf_counter()
        if self.triangulation_backend == "cuda" and tracks:
            inputs = []
            observations_by_track = {}
            for track in tracks:
                observations = self._frame_observations(track, frame_id)
                observations_by_track[track.track_id] = observations
                inputs.append(TriangulationInput(
                    track.track_id,
                    [self.cameras[item.cam_id] for item in observations],
                    np.asarray([[item.u, item.v] for item in observations], dtype=np.float64).reshape(-1, 2),
                ))
            try:
                results, stats = triangulate_cuda_batch(inputs, self.config)
            except RuntimeError as exc:
                LOGGER.warning("CUDA 批量三角化失败，本次运行回退 CPU: %s", exc)
                self.triangulation_backend = "cpu"
                self.stats["triangulation_backend"] = "cpu_fallback"
            else:
                for key, value in stats.items():
                    self.stats[key] += value
                samples = []
                for track in tracks:
                    result = results.get(track.track_id)
                    if result is None:
                        samples.append(self._triangulate_track(track, frame_id))
                    else:
                        samples.append(self._sample_from_result(
                            track, frame_id, observations_by_track[track.track_id], result
                        ))
                self.stats["triangulation_seconds"] += time.perf_counter() - started
                return samples
        if executor is None:
            samples = [self._triangulate_track(track, frame_id) for track in tracks]
        else:
            samples = list(executor.map(
                lambda track: self._triangulate_track(track, frame_id), tracks
            ))
        self.stats["triangulation_seconds"] += time.perf_counter() - started
        return samples

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

    def _load_checkpoint(self, path: Path, fingerprint: str) -> int | None:
        if not path.is_file():
            return None
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError) as exc:
            LOGGER.warning("忽略损坏的轨迹 checkpoint %s: %s", path, exc)
            return None
        if payload.get("fingerprint") != fingerprint:
            LOGGER.warning("轨迹 checkpoint 指纹不一致，将从 frame 1 重建")
            return None
        self.tracks = payload["tracks"]
        self.next_track_id = int(payload["next_track_id"])
        self.stats = defaultdict(int, payload.get("stats", {}))
        self.stats["triangulation_backend"] = self.triangulation_backend
        return int(payload["last_completed_frame"])

    def _write_checkpoint(self, path: Path, fingerprint: str, frame_id: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump({
                "version": 1,
                "fingerprint": fingerprint,
                "last_completed_frame": frame_id,
                "next_track_id": self.next_track_id,
                "tracks": self.tracks,
                "stats": dict(self.stats),
            }, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)

    def run(
        self,
        groups_by_frame: dict[int, list[SpatialGroup]],
        checkpoint_path: Path | None = None,
        checkpoint_fingerprint: str = "",
        resume: bool = False,
    ) -> dict[int, Track]:
        interval = max(1, int(self.config["track"].get("log_interval", 10)))
        checkpoint_interval = max(1, int(self.config["performance"]["checkpoint_interval_frames"]))
        completed_frame = None
        if resume and checkpoint_path is not None:
            completed_frame = self._load_checkpoint(checkpoint_path, checkpoint_fingerprint)
            if completed_frame is not None:
                LOGGER.info(
                    "恢复轨迹 checkpoint: frame=%d, tracks=%d",
                    completed_frame, len(self.tracks),
                )
        workers = max(1, int(self.config["performance"]["cpu_workers"]))
        executor = (
            ThreadPoolExecutor(max_workers=workers, thread_name_prefix="triangulate")
            if workers > 1 else None
        )
        LOGGER.info(
            "三角化后端: %s, CPU workers=%d, checkpoint=%d 帧",
            self.triangulation_backend, workers, checkpoint_interval,
        )
        try:
            for frame_index, frame_id in enumerate(self.frames):
                if completed_frame is not None and frame_id <= completed_frame:
                    continue
                self._propagate_to_frame(frame_id)
                self._spawn_and_merge(groups_by_frame.get(frame_id, []))
                frame_tracks = [
                    track for track in self.tracks.values()
                    if frame_id >= track.birth_frame and track.state != TrackState.ENDED
                ]
                samples = self._triangulate_frame(frame_tracks, frame_id, executor)
                for track, sample in zip(frame_tracks, samples):
                    self._update_state(track, sample, frame_id)
                if (
                    checkpoint_path is not None
                    and ((frame_index + 1) % checkpoint_interval == 0 or frame_index + 1 == len(self.frames))
                ):
                    self._write_checkpoint(
                        checkpoint_path, checkpoint_fingerprint, frame_id
                    )
                    LOGGER.info("已保存轨迹 checkpoint: frame=%d", frame_id)
                if frame_index == 0 or (frame_index + 1) % interval == 0 or frame_index + 1 == len(self.frames):
                    active = sum(track.state != TrackState.ENDED for track in self.tracks.values())
                    LOGGER.info(
                        "轨迹构建 %d/%d (frame=%d): total=%d, live=%d",
                        frame_index + 1, len(self.frames), frame_id, len(self.tracks), active,
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        return self.tracks
