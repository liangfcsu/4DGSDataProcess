"""Ordered post-triangulation optimization of sparse 3D trajectories."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from track_sparse.tracking.identity_validation import SwitchEvent, split_identity_jumps
from track_sparse.optimization.motion_groups import group_dynamic_tracks
from track_sparse.optimization.motion_models import fit_motion_models
from track_sparse.optimization.pose_refinement import refine_camera_poses, refine_track_measurements
from track_sparse.core.schema import Camera, Track
from track_sparse.geometry.uncertainty import annotate_track_uncertainty


@dataclass(slots=True)
class OptimizationResult:
    tracks: dict[int, Track]
    corrected_cameras: dict[tuple[int, int], Camera]
    pose_corrections: list[dict]
    switch_events: list[SwitchEvent]
    motion_groups: list[dict]
    metrics: dict[str, int | float]


def optimize_trajectories(
    tracks: dict[int, Track], cameras: dict[int, Camera], scene_scale: float, config: dict
) -> OptimizationResult:
    """Run the six geometry stages in their dependency-safe order."""
    metrics: dict[str, int | float] = {}

    # Uncertainty is needed before a large 3D step can be called significant.
    metrics.update(annotate_track_uncertainty(tracks, cameras, config))
    tracks, switch_events, split_stats = split_identity_jumps(tracks, scene_scale, config)
    metrics.update(split_stats)

    # Bootstrap a conservative static set using the immutable calibration.
    bootstrap = fit_motion_models(
        tracks, cameras, scene_scale, config, optimize_coordinates=False
    )
    metrics.update({f"bootstrap_{key}": value for key, value in bootstrap.items()})

    corrected, pose_records, pose_stats = refine_camera_poses(
        tracks, cameras, scene_scale, config
    )
    metrics.update(pose_stats)
    metrics.update(refine_track_measurements(tracks, cameras, corrected, config))

    # Recompute covariance and the competing models in the refined geometry.
    metrics.update(annotate_track_uncertainty(tracks, cameras, config, corrected))
    metrics.update(fit_motion_models(
        tracks, cameras, scene_scale, config, corrected, optimize_coordinates=True
    ))
    motion_groups, group_stats = group_dynamic_tracks(tracks, scene_scale, config)
    metrics.update(group_stats)
    return OptimizationResult(
        tracks, corrected, pose_records, switch_events, motion_groups, metrics
    )


def write_optimization_diagnostics(result: OptimizationResult, output_dir: Path) -> None:
    """Write human-readable evidence for pose, identity and motion decisions."""
    debug = output_dir / "debug"
    debug.mkdir(parents=True, exist_ok=True)
    payloads = {
        "pose_corrections.json": result.pose_corrections,
        "identity_switches.json": [
            {
                "track_id": event.track_id,
                "split_frame": event.split_frame,
                "score": event.score,
                "geometric_jump": event.geometric_jump,
                "acceleration_jump": event.acceleration_jump,
                "cycle_inconsistent": event.cycle_inconsistent,
                "weak_association": event.weak_association,
            }
            for event in result.switch_events
        ],
        "motion_groups.json": result.motion_groups,
        "optimization_metrics.json": result.metrics,
    }

    def json_safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value

    for name, payload in payloads.items():
        destination = debug / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(destination)
