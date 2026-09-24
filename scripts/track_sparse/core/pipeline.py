"""End-to-end orchestration for multi-view sparse 3D tracks."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import SCHEMA_VERSION
from track_sparse.geometry.camera_graph import build_camera_graph, load_camera_graph, save_camera_graph
from track_sparse.input.dataset import DatasetIndex, discover_dataset, validate_dataset
from track_sparse.exports.export import (
    calculate_metrics,
    write_frame_exports,
    write_freetimegs_export,
    write_summaries,
    write_tracks_h5,
)
from track_sparse.input.io_colmap import load_rig, read_points3d_text
from track_sparse.tracking.lifecycle import TrackManager
from track_sparse.matching.spatial_matcher import HlocFeatureMatcher
from track_sparse.tracking.sparse_verify import SparseCloudVerifier, discover_per_frame_clouds
from track_sparse.matching.temporal_tracker import TemporalTracker
from track_sparse.matching.track_graph import build_spatial_groups, filter_spatial_matches
from track_sparse.optimization.trajectory_optimizer import optimize_trajectories, write_optimization_diagnostics
from track_sparse.exports.visualize import assign_group_colors, write_track_overlays, write_trajectory_ply


LOGGER = logging.getLogger("track_sparse")
STAGES = ["prepare", "camera_graph", "features", "spawn", "temporal", "triangulate", "classify", "export"]


@dataclass(slots=True)
class PipelineOptions:
    dataset: str | None
    images_dir: str | None
    sparse_dir: str | None
    persparse_dir: str | None
    output: Path
    start_frame: int | None
    end_frame: int | None
    cameras: list[int] | None
    stages: list[str]
    resume: bool
    overwrite: bool
    dry_run: bool
    debug_frames: list[int]
    source_metadata: dict | None = None


def _atomic_json(data: dict, path: Path) -> None:
    def json_safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(data), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_pickle(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _config_slice(config: dict, keys: tuple[str, ...]) -> dict:
    return {key: config[key] for key in keys if key in config}


def _fingerprint(
    index: DatasetIndex,
    config: dict,
    per_frame_clouds: dict[int, Path],
    source_metadata: dict | None = None,
) -> str:
    digest = hashlib.sha256()
    for name in ("cameras.txt", "images.txt"):
        path = index.sparse_dir / name
        stat = path.stat()
        digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    for key in sorted(index.images):
        record = index.images[key]
        stat = record.path.stat()
        digest.update(f"{record.relative_name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    for frame_id, path in sorted(per_frame_clouds.items()):
        stat = path.stat()
        digest.update(f"persparse:{frame_id}:{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    digest.update(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())
    if source_metadata:
        digest.update(json.dumps(source_metadata, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _manifest(
    index: DatasetIndex,
    config: dict,
    fingerprint: str,
    warnings: list[str],
    per_frame_clouds: dict[int, Path],
    source_metadata: dict | None,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "coordinate_system": "COLMAP world; x_cam = R_w2c @ X_world + t_w2c",
        "pixel_coordinates": "original undistorted image pixels",
        "frame_id_origin": min(index.frame_ids),
        "images_dir": str(index.images_dir),
        "sparse_dir": str(index.sparse_dir),
        "camera_ids": index.cam_ids,
        "frame_ids": index.frame_ids,
        "image_count": len(index.images),
        "missing_count": len(index.missing),
        "missing_examples": [{"cam_id": c, "frame_id": f} for c, f in index.missing[:100]],
        "persparse_frame_count": len(per_frame_clouds),
        "source": source_metadata or {"input_type": "prepared"},
        "warnings": warnings,
        "config": config,
        "stages": {},
    }


def _mark(manifest: dict, output: Path, stage: str, status: str = "completed", **details) -> None:
    manifest["stages"][stage] = {"status": status, **details}
    _atomic_json(manifest, output / "manifest.json")


def _stage_limit(requested: list[str]) -> int:
    if not requested or requested == ["all"]:
        return len(STAGES) - 1
    aliases = {"matching": "temporal", "track": "triangulate"}
    normalized = [aliases.get(value, value) for value in requested]
    unknown = sorted(set(normalized) - set(STAGES))
    if unknown:
        raise ValueError(f"未知阶段 {unknown}；可选: {', '.join(STAGES)}")
    return max(STAGES.index(value) for value in normalized)


def _spawn_frames(frame_ids: list[int], interval: int) -> list[int]:
    first = min(frame_ids)
    return [frame for frame in frame_ids if (frame - first) % interval == 0]


def _spatial_pairs(index: DatasetIndex, graph: dict, spawn_frames: list[int]) -> list[tuple[str, str]]:
    pairs = []
    for frame_id in spawn_frames:
        for edge in graph["edges"]:
            first = index.get(int(edge["cam_a"]), frame_id)
            second = index.get(int(edge["cam_b"]), frame_id)
            if first is not None and second is not None:
                pairs.append((first.relative_name, second.relative_name))
    return pairs


def _temporal_pairs(index: DatasetIndex, offsets: list[int]) -> list[tuple[str, str]]:
    pairs = []
    available = set(index.frame_ids)
    for cam_id in index.cam_ids:
        for frame_id in index.frame_ids:
            first = index.get(cam_id, frame_id)
            if first is None:
                continue
            for offset in offsets:
                target_frame = frame_id + int(offset)
                if target_frame not in available:
                    continue
                second = index.get(cam_id, target_frame)
                if second is not None:
                    pairs.append((first.relative_name, second.relative_name))
    return pairs


def _clear_generated(output: Path) -> None:
    for name in (
        "manifest.json", "camera_graph.json", "tracks.h5", "tracks_summary.json",
        "features.h5", "matches.h5", "pairs.txt", "freetimegs_tracks.npz",
        "spatial_groups.pkl", "track_checkpoint.pkl",
    ):
        path = output / name
        if path.is_file():
            path.unlink()
    for name in ("exports", "debug"):
        path = output / name
        if path.is_dir():
            shutil.rmtree(path)


def run_pipeline(options: PipelineOptions, config: dict) -> dict:
    output = options.output.expanduser().resolve()
    manifest_path = output / "manifest.json"
    old_manifest = None
    if manifest_path.is_file():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if old_manifest is not None and not options.resume and not options.overwrite and not options.dry_run:
        raise FileExistsError(f"输出目录已有 manifest；请使用 --resume 或 --overwrite: {output}")

    index = discover_dataset(
        options.dataset, options.images_dir, options.sparse_dir,
        options.start_frame, options.end_frame, options.cameras,
    )
    rig_all = load_rig(index.sparse_dir)
    rig = {cam_id: rig_all[cam_id] for cam_id in index.cam_ids if cam_id in rig_all}
    warnings = validate_dataset(index, rig, int(config["data"]["image_size_tolerance_px"]))
    if index.missing and not bool(config["data"]["allow_missing_frames"]):
        raise ValueError(f"数据存在 {len(index.missing)} 个缺帧，但 data.allow_missing_frames=false")
    per_frame_clouds = discover_per_frame_clouds(options.dataset, options.persparse_dir)
    per_frame_clouds = {frame: path for frame, path in per_frame_clouds.items() if frame in set(index.frame_ids)}
    fingerprint = _fingerprint(index, config, per_frame_clouds, options.source_metadata)
    cache_compatible = bool(old_manifest and old_manifest.get("fingerprint") == fingerprint)
    old_inputs_unchanged = False
    matcher_cache_compatible = False
    if old_manifest is not None:
        old_config = old_manifest.get("config")
        if isinstance(old_config, dict):
            old_fingerprint = _fingerprint(
                index, old_config, per_frame_clouds, options.source_metadata
            )
            old_inputs_unchanged = old_manifest.get("fingerprint") == old_fingerprint
            matcher_cache_compatible = bool(
                old_inputs_unchanged
                and old_config.get("features") == config.get("features")
                and old_config.get("matching") == config.get("matching")
            )
    if options.resume and old_manifest is not None and not cache_compatible:
        if matcher_cache_compatible:
            LOGGER.warning(
                "配置已变化：保留兼容的 SuperPoint/SuperGlue 缓存，重新构建轨迹"
            )
        else:
            raise ValueError(
                "--resume 检测到输入或特征/匹配配置已经变化，拒绝复用旧缓存；"
                "请改用新的 --output，或确认后使用 --overwrite"
            )
    highest_stage = _stage_limit(options.stages)
    spawn_frames = _spawn_frames(index.frame_ids, int(config["spawn"]["interval"]))

    reference_points = read_points3d_text(
        index.sparse_dir / "points3D.txt",
        limit=int(config["camera_graph"]["overlap_sample_points"]),
    )
    graph = build_camera_graph(rig, config["camera_graph"], reference_points)
    spatial_pairs = _spatial_pairs(index, graph, spawn_frames)
    temporal_pairs = _temporal_pairs(index, config["temporal"]["offsets"])
    workload = {
        "images": len(index.images),
        "cameras": len(index.cam_ids),
        "frames": len(index.frame_ids),
        "missing": len(index.missing),
        "spawn_frames": spawn_frames,
        "camera_edges": len(graph["edges"]),
        "spatial_pairs": len(spatial_pairs),
        "temporal_pairs": len(temporal_pairs),
        "persparse_frames": len(per_frame_clouds),
        "output": str(output),
        "warnings": warnings,
    }
    if options.dry_run:
        return workload

    # Validate and fingerprint every input before an explicit overwrite removes old outputs.
    if options.overwrite:
        _clear_generated(output)
        old_manifest = None

    if (
        options.resume
        and cache_compatible
        and highest_stage == STAGES.index("export")
        and (output / "tracks.h5").is_file()
        and old_manifest.get("stages", {}).get("export", {}).get("status") == "completed"
    ):
        LOGGER.info("输入与配置未变化，复用已完成的输出: %s", output)
        return old_manifest["stages"]["export"].get("metrics", workload)

    manifest = _manifest(
        index, config, fingerprint, warnings, per_frame_clouds, options.source_metadata
    )
    output.mkdir(parents=True, exist_ok=True)
    _mark(manifest, output, "prepare", image_count=len(index.images), missing_count=len(index.missing))
    for warning in warnings:
        LOGGER.warning(warning)
    LOGGER.info("输入: %d 相机, %d 帧, %d 图像", len(index.cam_ids), len(index.frame_ids), len(index.images))
    if highest_stage == STAGES.index("prepare"):
        return workload

    graph_path = output / "camera_graph.json"
    graph_cache_compatible = bool(
        old_inputs_unchanged
        and old_manifest
        and old_manifest.get("config", {}).get("camera_graph") == config.get("camera_graph")
    )
    if options.resume and graph_cache_compatible and graph_path.is_file():
        graph = load_camera_graph(graph_path)
        LOGGER.info("复用 camera_graph.json")
    else:
        save_camera_graph(graph, graph_path)
    _mark(manifest, output, "camera_graph", edge_count=len(graph["edges"]))
    if highest_stage == STAGES.index("camera_graph"):
        return workload

    matcher = HlocFeatureMatcher(index.images_dir, output, config)
    needs_temporal = highest_stage >= STAGES.index("temporal")
    required_names = index.image_names() if needs_temporal else sorted({name for pair in spatial_pairs for name in pair})
    matcher.extract(required_names, overwrite=options.overwrite)
    _mark(manifest, output, "features", image_count=len(required_names))
    if highest_stage == STAGES.index("features"):
        return workload

    matcher.match_pairs(spatial_pairs, overwrite=options.overwrite)
    spatial_cache_path = output / "spatial_groups.pkl"
    spatial_config = _config_slice(
        config,
        ("camera_graph", "features", "matching", "spatial", "spawn", "triangulation"),
    )
    spatial_fingerprint = _fingerprint(
        index, spatial_config, per_frame_clouds, options.source_metadata
    )
    spatial_cache = None
    if options.resume and spatial_cache_path.is_file():
        try:
            with spatial_cache_path.open("rb") as handle:
                candidate = pickle.load(handle)
            if candidate.get("fingerprint") == spatial_fingerprint:
                spatial_cache = candidate
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError):
            spatial_cache = None
    if spatial_cache is not None:
        groups_by_frame = spatial_cache["groups_by_frame"]
        spatial_stats = spatial_cache["spatial_stats"]
        graph = spatial_cache.get("graph", graph)
        LOGGER.info("复用出生帧空间组缓存: %s", spatial_cache_path)
    else:
        groups_by_frame = {}
        spatial_stats = {}
        first_spawn = spawn_frames[0] if spawn_frames else None
        first_edge_counts: dict[tuple[int, int], int] = {}
        for frame_id in spawn_frames:
            feature_data = {}
            for cam_id in index.cam_ids:
                record = index.get(cam_id, frame_id)
                if record is not None:
                    feature_data[cam_id] = matcher.features(record.relative_name)
            frame_matches = []
            for edge in graph["edges"]:
                cam_a, cam_b = int(edge["cam_a"]), int(edge["cam_b"])
                first, second = index.get(cam_a, frame_id), index.get(cam_b, frame_id)
                if first is None or second is None:
                    continue
                pair = matcher.matches(first.relative_name, second.relative_name)
                pair = filter_spatial_matches(
                    pair, rig[cam_a], rig[cam_b], float(config["matching"]["min_score"]),
                    float(config["spatial"]["epipolar_threshold_px"]),
                )
                frame_matches.append((cam_a, cam_b, pair))
                if frame_id == first_spawn:
                    first_edge_counts[(cam_a, cam_b)] = len(pair.scores)
            groups, stats = build_spatial_groups(frame_id, frame_matches, feature_data, rig, config)
            groups_by_frame[frame_id] = groups
            spatial_stats[str(frame_id)] = stats
            LOGGER.info("出生帧 %d: %d 个几何有效多视角种子", frame_id, len(groups))
        assign_group_colors(groups_by_frame, index)
        for edge in graph["edges"]:
            edge["validated_match_count"] = first_edge_counts.get(
                (int(edge["cam_a"]), int(edge["cam_b"])), 0
            )
        save_camera_graph(graph, graph_path)
        if bool(config["performance"]["cache_spatial_groups"]):
            _atomic_pickle({
                "version": 1,
                "fingerprint": spatial_fingerprint,
                "groups_by_frame": groups_by_frame,
                "spatial_stats": spatial_stats,
                "graph": graph,
            }, spatial_cache_path)
    _mark(
        manifest, output, "spawn",
        spawn_frame_count=len(spawn_frames),
        seed_count=sum(len(groups) for groups in groups_by_frame.values()),
        per_frame=spatial_stats,
    )
    if highest_stage == STAGES.index("spawn"):
        return workload

    matcher.match_pairs(temporal_pairs, overwrite=options.overwrite)
    _mark(manifest, output, "temporal", pair_count=len(temporal_pairs))
    if highest_stage == STAGES.index("temporal"):
        return workload

    temporal = TemporalTracker(matcher, index, float(config["temporal"]["min_score"]))
    sparse_verifier = None
    if bool(config["sparse_verify"]["enabled"]) and per_frame_clouds:
        sparse_verifier = SparseCloudVerifier(per_frame_clouds, float(config["sparse_verify"]["nn_scale"]))
    manager = TrackManager(rig, index.frame_ids, config, temporal, sparse_verifier)
    checkpoint_config = dict(config)
    checkpoint_config.pop("performance", None)
    checkpoint_config.pop("export", None)
    checkpoint_fingerprint = _fingerprint(
        index, checkpoint_config, per_frame_clouds, options.source_metadata
    )
    tracks = manager.run(
        groups_by_frame,
        checkpoint_path=output / "track_checkpoint.pkl",
        checkpoint_fingerprint=checkpoint_fingerprint,
        resume=options.resume,
    )
    _mark(manifest, output, "triangulate", track_count=len(tracks), lifecycle_stats=dict(manager.stats))
    if highest_stage == STAGES.index("triangulate"):
        write_tracks_h5(tracks, rig, output / "tracks.h5")
        return workload

    optimization = optimize_trajectories(tracks, rig, float(graph["scene_scale"]), config)
    tracks = optimization.tracks
    write_optimization_diagnostics(optimization, output)
    _mark(manifest, output, "classify", **optimization.metrics)
    if highest_stage == STAGES.index("classify"):
        write_tracks_h5(
            tracks, rig, output / "tracks.h5", optimization.corrected_cameras
        )
        return workload

    write_tracks_h5(tracks, rig, output / "tracks.h5", optimization.corrected_cameras)
    exports = output / "exports"
    exported_frames = write_frame_exports(
        tracks, exports, bool(config["export"]["npz"]), bool(config["export"]["ply"]), index.frame_ids
    )
    write_freetimegs_export(tracks, output / "freetimegs_tracks.npz")
    write_trajectory_ply(tracks, output / "debug" / "trajectories.ply")
    overlay_frames = options.debug_frames or ([spawn_frames[0]] if spawn_frames else [])
    overlay_count = 0
    if bool(config["export"]["overlays"]) and overlay_frames:
        overlay_count = write_track_overlays(tracks, index, overlay_frames, output / "debug" / "overlays")
    metrics = calculate_metrics(tracks)
    metrics.update(manager.stats)
    metrics.update(optimization.metrics)
    write_summaries(tracks, metrics, output)
    _mark(
        manifest, output, "export", track_count=len(tracks),
        exported_frame_count=exported_frames, overlay_count=overlay_count, metrics=metrics,
    )
    LOGGER.info("完成: %d 条轨迹, %d 个有效 3D 样本", len(tracks), metrics["valid_3d_samples"])
    return metrics
