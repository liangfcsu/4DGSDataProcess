"""Conservative static/dynamic classification and track quality grading."""

from __future__ import annotations

import numpy as np

from .schema import MotionClass, Track


def classify_tracks(tracks: dict[int, Track], scene_scale: float, config: dict) -> None:
    minimum = int(config["classification"]["min_valid_frames"])
    threshold = max(float(scene_scale) * float(config["classification"]["static_baseline_ratio"]), 1e-9)
    dynamic_multiplier = float(config["classification"]["dynamic_threshold_multiplier"])
    reproj_limit = float(config["triangulation"]["max_reprojection_error_px"])
    for track in tracks.values():
        valid_samples = [sample for sample in track.samples.values() if sample.valid_3d]
        confidences = [sample.geometry_confidence for sample in valid_samples]
        reprojections = [sample.reprojection_rmse for sample in valid_samples if np.isfinite(sample.reprojection_rmse)]
        track.mean_confidence = float(np.mean(confidences)) if confidences else 0.0
        track.median_reprojection_error = float(np.median(reprojections)) if reprojections else float("nan")
        if len(valid_samples) < minimum:
            track.motion_class = MotionClass.UNKNOWN
        else:
            xyz = np.asarray([sample.xyz for sample in valid_samples])
            center = np.median(xyz, axis=0)
            residual = np.linalg.norm(xyz - center, axis=1)
            p90 = float(np.percentile(residual, 90))
            if p90 <= threshold:
                track.motion_class = MotionClass.STATIC
            elif p90 >= dynamic_multiplier * threshold:
                track.motion_class = MotionClass.DYNAMIC
            else:
                track.motion_class = MotionClass.UNKNOWN

        length_score = min(track.valid_frame_count / 50.0, 1.0)
        confidence_score = track.mean_confidence
        reproj_score = (
            max(0.0, 1.0 - track.median_reprojection_error / reproj_limit)
            if np.isfinite(track.median_reprojection_error) else 0.0
        )
        quality_score = 0.4 * length_score + 0.35 * confidence_score + 0.25 * reproj_score
        track.quality = "high" if quality_score >= 0.75 else "medium" if quality_score >= 0.45 else "low"

