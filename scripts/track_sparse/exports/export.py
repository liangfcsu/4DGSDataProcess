"""HDF5, NPZ, PLY and statistics exporters."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from track_sparse.core import SCHEMA_VERSION
from track_sparse.core.schema import Camera, MotionClass, Track


def _string_array(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=h5py.string_dtype(encoding="utf-8"))


def _json_safe(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_tracks_h5(
    tracks: dict[int, Track],
    cameras: dict[int, Camera],
    path: Path,
    corrected_cameras: dict[tuple[int, int], Camera] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    observations = sorted(
        (observation for track in tracks.values() for observation in track.observations.values()),
        key=lambda item: (item.track_id, item.frame_id, item.cam_id),
    )
    samples = sorted(
        (sample for track in tracks.values() for sample in track.samples.values()),
        key=lambda item: (item.track_id, item.frame_id),
    )
    ordered_tracks = sorted(tracks.values(), key=lambda item: item.track_id)
    with h5py.File(temporary, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["coordinate_system"] = "COLMAP world; world_to_camera extrinsics"
        handle.attrs["pixel_coordinates"] = "original undistorted image pixels"

        group = handle.create_group("cameras")
        ordered_cameras = [cameras[key] for key in sorted(cameras)]
        group.create_dataset("cam_id", data=np.asarray([cam.cam_id for cam in ordered_cameras], dtype=np.int16))
        group.create_dataset("image_id", data=np.asarray([cam.image_id for cam in ordered_cameras], dtype=np.int32))
        group.create_dataset("camera_id", data=np.asarray([cam.camera_id for cam in ordered_cameras], dtype=np.int32))
        group.create_dataset("width", data=np.asarray([cam.width for cam in ordered_cameras], dtype=np.int32))
        group.create_dataset("height", data=np.asarray([cam.height for cam in ordered_cameras], dtype=np.int32))
        group.create_dataset("K", data=np.asarray([cam.K for cam in ordered_cameras], dtype=np.float64))
        group.create_dataset("R_w2c", data=np.asarray([cam.R_w2c for cam in ordered_cameras], dtype=np.float64))
        group.create_dataset("t_w2c", data=np.asarray([cam.t_w2c for cam in ordered_cameras], dtype=np.float64))
        group.create_dataset("P", data=np.asarray([cam.P for cam in ordered_cameras], dtype=np.float64))
        group.create_dataset("center_world", data=np.asarray([cam.center_world for cam in ordered_cameras], dtype=np.float64))
        group.create_dataset("reference_image_name", data=_string_array([cam.reference_image_name for cam in ordered_cameras]))

        group = handle.create_group("camera_corrections")
        ordered_corrections = sorted((corrected_cameras or {}).items())
        group.create_dataset("frame_id", data=np.asarray([key[0] for key, _ in ordered_corrections], dtype=np.int32))
        group.create_dataset("cam_id", data=np.asarray([key[1] for key, _ in ordered_corrections], dtype=np.int16))
        group.create_dataset("R_w2c", data=np.asarray([cam.R_w2c for _, cam in ordered_corrections], dtype=np.float64).reshape(-1, 3, 3))
        group.create_dataset("t_w2c", data=np.asarray([cam.t_w2c for _, cam in ordered_corrections], dtype=np.float64).reshape(-1, 3))

        group = handle.create_group("observations")
        group.create_dataset("track_id", data=np.asarray([item.track_id for item in observations], dtype=np.int64))
        group.create_dataset("frame_id", data=np.asarray([item.frame_id for item in observations], dtype=np.int32))
        group.create_dataset("cam_id", data=np.asarray([item.cam_id for item in observations], dtype=np.int16))
        group.create_dataset("feature_id", data=np.asarray([item.feature_id for item in observations], dtype=np.int32))
        group.create_dataset("uv", data=np.asarray([[item.u, item.v] for item in observations], dtype=np.float32).reshape(-1, 2))
        group.create_dataset("visible", data=np.asarray([item.visible for item in observations], dtype=np.bool_))
        group.create_dataset("tracker_confidence", data=np.asarray([item.tracker_confidence for item in observations], dtype=np.float32))
        group.create_dataset("spatial_confidence", data=np.asarray([item.spatial_confidence for item in observations], dtype=np.float32))
        group.create_dataset("reprojection_error", data=np.asarray([item.reprojection_error for item in observations], dtype=np.float32))
        group.create_dataset("is_inlier", data=np.asarray([item.is_inlier for item in observations], dtype=np.bool_))
        group.create_dataset("source", data=_string_array([item.source for item in observations]))
        group.create_dataset("association_score", data=np.asarray([item.association_score for item in observations], dtype=np.float32))
        group.create_dataset("cycle_consistency", data=np.asarray([item.cycle_consistency for item in observations], dtype=np.int8))
        group.create_dataset("switch_score", data=np.asarray([item.switch_score for item in observations], dtype=np.float32))

        group = handle.create_group("samples3d")
        group.create_dataset("track_id", data=np.asarray([item.track_id for item in samples], dtype=np.int64))
        group.create_dataset("frame_id", data=np.asarray([item.frame_id for item in samples], dtype=np.int32))
        group.create_dataset("xyz", data=np.asarray([item.output_xyz for item in samples], dtype=np.float32).reshape(-1, 3))
        group.create_dataset("raw_xyz", data=np.asarray([item.xyz for item in samples], dtype=np.float32).reshape(-1, 3))
        group.create_dataset("pose_refined_xyz", data=np.asarray([item.pose_refined_xyz for item in samples], dtype=np.float32).reshape(-1, 3))
        group.create_dataset("optimized_xyz", data=np.asarray([item.optimized_xyz for item in samples], dtype=np.float32).reshape(-1, 3))
        group.create_dataset("covariance", data=np.asarray([item.covariance for item in samples], dtype=np.float32).reshape(-1, 3, 3))
        group.create_dataset("position_std", data=np.asarray([item.position_std for item in samples], dtype=np.float32))
        group.create_dataset("motion_significance", data=np.asarray([item.motion_significance for item in samples], dtype=np.float32))
        group.create_dataset("pose_refined", data=np.asarray([item.pose_refined for item in samples], dtype=np.bool_))
        group.create_dataset("valid_3d", data=np.asarray([item.valid_3d for item in samples], dtype=np.bool_))
        group.create_dataset("num_visible_views", data=np.asarray([item.num_visible_views for item in samples], dtype=np.int16))
        group.create_dataset("num_inlier_views", data=np.asarray([item.num_inlier_views for item in samples], dtype=np.int16))
        group.create_dataset("reprojection_rmse", data=np.asarray([item.reprojection_rmse for item in samples], dtype=np.float32))
        group.create_dataset("min_triangulation_angle_deg", data=np.asarray([item.min_triangulation_angle_deg for item in samples], dtype=np.float32))
        group.create_dataset("geometry_confidence", data=np.asarray([item.geometry_confidence for item in samples], dtype=np.float32))
        group.create_dataset("sparse_support_distance", data=np.asarray([item.sparse_support_distance for item in samples], dtype=np.float32))
        group.create_dataset("state", data=_string_array([item.state.value for item in samples]))

        group = handle.create_group("tracks")
        group.create_dataset("track_id", data=np.asarray([item.track_id for item in ordered_tracks], dtype=np.int64))
        group.create_dataset("birth_frame", data=np.asarray([item.birth_frame for item in ordered_tracks], dtype=np.int32))
        group.create_dataset("last_valid_frame", data=np.asarray([item.last_valid_frame for item in ordered_tracks], dtype=np.int32))
        group.create_dataset("state", data=_string_array([item.state.value for item in ordered_tracks]))
        group.create_dataset("motion_class", data=_string_array([item.motion_class.value for item in ordered_tracks]))
        group.create_dataset("quality", data=_string_array([item.quality for item in ordered_tracks]))
        group.create_dataset("valid_frame_count", data=np.asarray([item.valid_frame_count for item in ordered_tracks], dtype=np.int32))
        group.create_dataset("longest_valid_run", data=np.asarray([item.longest_valid_run for item in ordered_tracks], dtype=np.int32))
        group.create_dataset("mean_confidence", data=np.asarray([item.mean_confidence for item in ordered_tracks], dtype=np.float32))
        group.create_dataset("median_reprojection_error", data=np.asarray([item.median_reprojection_error for item in ordered_tracks], dtype=np.float32))
        group.create_dataset("recovery_count", data=np.asarray([item.recovery_count for item in ordered_tracks], dtype=np.int32))
        group.create_dataset("color_rgb", data=np.asarray([item.color_rgb for item in ordered_tracks], dtype=np.uint8).reshape(-1, 3))
        group.create_dataset("canonical_xyz", data=np.asarray([item.canonical_xyz for item in ordered_tracks], dtype=np.float32).reshape(-1, 3))
        group.create_dataset("canonical_covariance", data=np.asarray([item.canonical_covariance for item in ordered_tracks], dtype=np.float32).reshape(-1, 3, 3))
        group.create_dataset("static_model_score", data=np.asarray([item.static_model_score for item in ordered_tracks], dtype=np.float64))
        group.create_dataset("dynamic_model_score", data=np.asarray([item.dynamic_model_score for item in ordered_tracks], dtype=np.float64))
        group.create_dataset("model_score_margin", data=np.asarray([item.model_score_margin for item in ordered_tracks], dtype=np.float64))
        group.create_dataset("static_confidence", data=np.asarray([item.static_confidence for item in ordered_tracks], dtype=np.float32))
        group.create_dataset("motion_group_id", data=np.asarray([item.motion_group_id for item in ordered_tracks], dtype=np.int32))
        group.create_dataset("identity_parent_id", data=np.asarray([item.identity_parent_id for item in ordered_tracks], dtype=np.int64))
        group.create_dataset("split_frame", data=np.asarray([item.split_frame for item in ordered_tracks], dtype=np.int32))
        group.create_dataset("identity_switch_count", data=np.asarray([item.identity_switch_count for item in ordered_tracks], dtype=np.int32))
    temporary.replace(path)


def _valid_by_frame(tracks: dict[int, Track]) -> dict[int, list[tuple[Track, object]]]:
    result: dict[int, list[tuple[Track, object]]] = {}
    for track in tracks.values():
        for frame_id, sample in track.samples.items():
            if sample.valid_3d:
                result.setdefault(frame_id, []).append((track, sample))
    return result


def write_frame_exports(
    tracks: dict[int, Track],
    output_dir: Path,
    write_npz: bool,
    write_ply: bool,
    frame_ids: list[int] | None = None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_frame = _valid_by_frame(tracks)
    for frame_id in frame_ids or []:
        by_frame.setdefault(frame_id, [])
    for frame_id, entries in sorted(by_frame.items()):
        entries.sort(key=lambda item: item[0].track_id)
        ids = np.asarray([track.track_id for track, _ in entries], dtype=np.int64)
        xyz = np.asarray([sample.output_xyz for _, sample in entries], dtype=np.float32).reshape(-1, 3)
        raw_xyz = np.asarray([sample.xyz for _, sample in entries], dtype=np.float32).reshape(-1, 3)
        rgb = np.asarray([track.color_rgb for track, _ in entries], dtype=np.uint8).reshape(-1, 3)
        confidence = np.asarray([sample.geometry_confidence for _, sample in entries], dtype=np.float32)
        views = np.asarray([sample.num_inlier_views for _, sample in entries], dtype=np.int16)
        motion = np.asarray([track.motion_class.value for track, _ in entries])
        states = np.asarray([sample.state.value for _, sample in entries])
        state_codes = np.asarray([
            {"new": 0, "tentative": 1, "active": 2, "occluded": 3, "lost": 4, "ended": 5}[sample.state.value]
            for _, sample in entries
        ], dtype=np.uint8)
        sparse_distance = np.asarray([sample.sparse_support_distance for _, sample in entries], dtype=np.float32)
        position_std = np.asarray([sample.position_std for _, sample in entries], dtype=np.float32)
        motion_significance = np.asarray([sample.motion_significance for _, sample in entries], dtype=np.float32)
        static_confidence = np.asarray([track.static_confidence for track, _ in entries], dtype=np.float32)
        motion_group_id = np.asarray([track.motion_group_id for track, _ in entries], dtype=np.int32)
        motion_codes = np.asarray([
            {MotionClass.UNKNOWN: 0, MotionClass.STATIC: 1, MotionClass.DYNAMIC: 2}[track.motion_class]
            for track, _ in entries
        ], dtype=np.uint8)
        if write_npz:
            np.savez_compressed(
                output_dir / f"frame{frame_id:03d}_track_points.npz",
                track_id=ids, xyz=xyz, raw_xyz=raw_xyz, rgb=rgb, confidence=confidence,
                num_views=views, valid=np.ones(len(ids), dtype=np.bool_), motion_class=motion,
                motion_class_code=motion_codes, motion_group_id=motion_group_id,
                static_confidence=static_confidence, position_std=position_std,
                motion_significance=motion_significance,
                state=states, sparse_support_distance=sparse_distance,
            )
        if write_ply:
            path = output_dir / f"frame{frame_id:03d}_track_points.ply"
            with path.open("w", encoding="utf-8") as handle:
                handle.write("ply\nformat ascii 1.0\n")
                handle.write(f"element vertex {len(entries)}\n")
                handle.write("property float x\nproperty float y\nproperty float z\n")
                handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                handle.write("property uint track_id\nproperty float confidence\nproperty ushort num_views\n")
                handle.write("property uchar state\nproperty uchar motion_class\n")
                handle.write("property int motion_group_id\nproperty float static_confidence\n")
                handle.write("property float position_std\nproperty float motion_significance\n")
                handle.write("property float sparse_support_distance\nend_header\n")
                for index in range(len(entries)):
                    values = [
                        *xyz[index], *rgb[index], int(ids[index]), float(confidence[index]),
                        int(views[index]), int(state_codes[index]), int(motion_codes[index]),
                        int(motion_group_id[index]), float(static_confidence[index]),
                        float(position_std[index]), float(motion_significance[index]),
                        float(sparse_distance[index]),
                    ]
                    handle.write(" ".join(map(str, values)) + "\n")
    return len(by_frame)


def write_freetimegs_export(tracks: dict[int, Track], path: Path) -> None:
    entries = []
    for track in tracks.values():
        for sample in track.samples.values():
            if sample.valid_3d:
                entries.append((track, sample))
    entries.sort(key=lambda item: (item[0].track_id, item[1].frame_id))
    np.savez_compressed(
        path,
        track_id=np.asarray([track.track_id for track, _ in entries], dtype=np.int64),
        frame_id=np.asarray([sample.frame_id for _, sample in entries], dtype=np.int32),
        xyz=np.asarray([sample.output_xyz for _, sample in entries], dtype=np.float32).reshape(-1, 3),
        raw_xyz=np.asarray([sample.xyz for _, sample in entries], dtype=np.float32).reshape(-1, 3),
        confidence=np.asarray([sample.geometry_confidence for _, sample in entries], dtype=np.float32),
        visibility=np.ones(len(entries), dtype=np.bool_),
        num_views=np.asarray([sample.num_inlier_views for _, sample in entries], dtype=np.int16),
        motion_type=np.asarray([track.motion_class.value for track, _ in entries]),
        motion_group_id=np.asarray([track.motion_group_id for track, _ in entries], dtype=np.int32),
        static_confidence=np.asarray([track.static_confidence for track, _ in entries], dtype=np.float32),
        position_std=np.asarray([sample.position_std for _, sample in entries], dtype=np.float32),
        birth_frame=np.asarray([track.birth_frame for track, _ in entries], dtype=np.int32),
    )


def calculate_metrics(tracks: dict[int, Track]) -> dict:
    ordered = list(tracks.values())
    lengths = [track.valid_frame_count for track in ordered]
    valid_samples = [sample for track in ordered for sample in track.samples.values() if sample.valid_3d]
    reprojections = [sample.reprojection_rmse for sample in valid_samples if math.isfinite(sample.reprojection_rmse)]
    confidences = [sample.geometry_confidence for sample in valid_samples]
    views = [sample.num_inlier_views for sample in valid_samples]
    state_counts = Counter(track.state.value for track in ordered)
    motion_counts = Counter(track.motion_class.value for track in ordered)
    quality_counts = Counter(track.quality for track in ordered)
    position_std = [sample.position_std for sample in valid_samples if math.isfinite(sample.position_std)]
    static_drift = [
        float(np.linalg.norm(sample.output_xyz - track.canonical_xyz))
        for track in ordered if track.motion_class == MotionClass.STATIC
        and np.all(np.isfinite(track.canonical_xyz))
        for sample in track.samples.values() if sample.valid_3d
    ]
    dynamic_path_lengths = []
    for track in ordered:
        if track.motion_class != MotionClass.DYNAMIC:
            continue
        values = [
            sample.output_xyz for _, sample in sorted(track.samples.items())
            if sample.valid_3d and np.all(np.isfinite(sample.output_xyz))
        ]
        if len(values) >= 2:
            dynamic_path_lengths.append(float(np.linalg.norm(np.diff(np.asarray(values), axis=0), axis=1).sum()))
    return {
        "total_tracks": len(ordered),
        "state_counts": dict(state_counts),
        "motion_class_counts": dict(motion_counts),
        "quality_counts": dict(quality_counts),
        "mean_track_length": float(np.mean(lengths)) if lengths else 0.0,
        "median_track_length": float(np.median(lengths)) if lengths else 0.0,
        "tracks_1_frame": sum(value == 1 for value in lengths),
        "tracks_2_frames": sum(value == 2 for value in lengths),
        "tracks_ge_10_frames": sum(value >= 10 for value in lengths),
        "tracks_ge_50_frames": sum(value >= 50 for value in lengths),
        "tracks_ge_100_frames": sum(value >= 100 for value in lengths),
        "valid_3d_samples": len(valid_samples),
        "mean_views_per_point": float(np.mean(views)) if views else 0.0,
        "mean_reprojection_error": float(np.mean(reprojections)) if reprojections else None,
        "median_reprojection_error": float(np.median(reprojections)) if reprojections else None,
        "p95_reprojection_error": float(np.percentile(reprojections, 95)) if reprojections else None,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "median_position_std": float(np.median(position_std)) if position_std else None,
        "p95_position_std": float(np.percentile(position_std, 95)) if position_std else None,
        "static_output_drift_max": max(static_drift, default=0.0),
        "median_dynamic_path_length": float(np.median(dynamic_path_lengths)) if dynamic_path_lengths else None,
        "motion_group_count": len({track.motion_group_id for track in ordered if track.motion_group_id >= 0}),
        "occlusion_recovery_count": sum(track.recovery_count for track in ordered),
        "track_survival_rate": (
            sum(bool(track.samples) and track.samples[max(track.samples)].valid_3d for track in ordered) / len(ordered)
            if ordered else 0.0
        ),
        "track_merge_count": 0,
        "track_split_count": sum(track.split_frame >= 0 for track in ordered),
    }


def write_summaries(tracks: dict[int, Track], metrics: dict, output_dir: Path) -> None:
    summaries = []
    for track in sorted(tracks.values(), key=lambda item: item.track_id):
        observed_frames = sorted({frame for frame, _ in track.observations})
        summaries.append({
            "track_id": track.track_id,
            "birth_frame": track.birth_frame,
            "last_valid_frame": track.last_valid_frame,
            "num_observed_frames": len(observed_frames),
            "valid_frame_count": track.valid_frame_count,
            "longest_valid_run": track.longest_valid_run,
            "mean_confidence": track.mean_confidence,
            "median_reprojection_error": track.median_reprojection_error if math.isfinite(track.median_reprojection_error) else None,
            "state": track.state.value,
            "motion_class": track.motion_class.value,
            "quality": track.quality,
            "recovery_count": track.recovery_count,
            "canonical_xyz": track.canonical_xyz.tolist() if np.all(np.isfinite(track.canonical_xyz)) else None,
            "static_model_score": track.static_model_score if math.isfinite(track.static_model_score) else None,
            "dynamic_model_score": track.dynamic_model_score if math.isfinite(track.dynamic_model_score) else None,
            "model_score_margin": track.model_score_margin if math.isfinite(track.model_score_margin) else None,
            "static_confidence": track.static_confidence,
            "motion_group_id": track.motion_group_id,
            "identity_parent_id": track.identity_parent_id,
            "split_frame": track.split_frame,
            "identity_switch_count": track.identity_switch_count,
        })
    temporary = output_dir / "tracks_summary.json.tmp"
    temporary.write_text(
        json.dumps(_json_safe({"metrics": metrics, "tracks": summaries}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_dir / "tracks_summary.json")
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    metrics_tmp = debug_dir / "metrics.json.tmp"
    metrics_tmp.write_text(json.dumps(_json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_tmp.replace(debug_dir / "metrics.json")
