"""Detect abrupt identity changes and split contaminated trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .schema import MotionClass, Track, TrackState
from .uncertainty import mahalanobis_distance


@dataclass(slots=True)
class SwitchEvent:
    track_id: int
    split_frame: int
    score: float
    geometric_jump: bool
    acceleration_jump: bool
    cycle_inconsistent: bool
    weak_association: bool


def _robust_limit(values: np.ndarray, floor: float, multiplier: float) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return floor
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return max(float(floor), median + multiplier * max(1.4826 * mad, floor * 0.05))


def detect_identity_switches(track: Track, scene_scale: float, config: dict) -> list[SwitchEvent]:
    section = config["identity_validation"]
    valid = [
        sample for _, sample in sorted(track.samples.items())
        if sample.valid_3d and np.all(np.isfinite(sample.measurement_xyz))
    ]
    if len(valid) < int(section["min_track_frames"]):
        return []
    steps = []
    boundaries = []
    for first, second in zip(valid[:-1], valid[1:]):
        gap = max(1, second.frame_id - first.frame_id)
        steps.append(float(np.linalg.norm(second.measurement_xyz - first.measurement_xyz) / gap))
        boundaries.append((first, second))
    steps_array = np.asarray(steps, dtype=np.float64)
    step_limit = _robust_limit(
        steps_array,
        float(scene_scale) * float(section["max_step_ratio"]),
        float(section["mad_multiplier"]),
    )
    accelerations = np.full(len(steps), np.nan, dtype=np.float64)
    for index in range(1, len(steps)):
        accelerations[index] = abs(steps[index] - steps[index - 1])
    acceleration_limit = _robust_limit(
        accelerations,
        float(scene_scale) * float(section["max_acceleration_ratio"]),
        float(section["mad_multiplier"]),
    )

    events: list[SwitchEvent] = []
    for index, (first, second) in enumerate(boundaries):
        if second.frame_id - first.frame_id > int(section["max_checked_gap"]):
            continue
        covariance = first.covariance + second.covariance
        sigma_jump = mahalanobis_distance(
            second.measurement_xyz - first.measurement_xyz, covariance
        )
        geometric = bool(
            steps[index] > step_limit
            and (not math.isfinite(sigma_jump) or sigma_jump >= float(section["min_jump_sigma"]))
        )
        acceleration = bool(
            index > 0 and math.isfinite(accelerations[index])
            and accelerations[index] > acceleration_limit
        )
        frame_observations = [
            observation for (frame, _), observation in track.observations.items()
            if frame == second.frame_id
        ]
        known_cycles = [value.cycle_consistency for value in frame_observations if value.cycle_consistency >= 0]
        cycle_bad = bool(
            known_cycles
            and np.mean(np.asarray(known_cycles) == 0) >= float(section["cycle_bad_fraction"])
        )
        association_scores = [
            value.association_score for value in frame_observations
            if value.source in {"temporal", "reconnect"}
        ]
        weak = bool(
            association_scores
            and float(np.median(association_scores)) < float(section["min_association_score"])
        )
        score = (
            float(geometric)
            + float(acceleration)
            + float(section["cycle_evidence_weight"]) * float(cycle_bad)
            + float(section["weak_association_weight"]) * float(weak)
        )
        # A jump alone can be legitimate fast motion. Require a second signal.
        if geometric and score >= float(section["split_score"]):
            for observation in frame_observations:
                observation.switch_score = score
            events.append(SwitchEvent(
                track.track_id, second.frame_id, score, geometric, acceleration, cycle_bad, weak
            ))
    return events


def recompute_track_metadata(track: Track) -> None:
    valid_frames = sorted(
        frame for frame, sample in track.samples.items() if sample.valid_3d
    )
    track.birth_frame = min(
        [frame for frame, _ in track.observations] + list(track.samples) + [track.birth_frame]
    )
    track.valid_frame_count = len(valid_frames)
    track.last_valid_frame = max(valid_frames, default=-1)
    longest = current = 0
    previous = None
    for frame in valid_frames:
        current = current + 1 if previous is not None and frame == previous + 1 else 1
        longest = max(longest, current)
        previous = frame
    track.longest_valid_run = longest
    track.current_valid_run = current
    if track.samples:
        track.state = track.samples[max(track.samples)].state


def _segment_track(
    source: Track,
    track_id: int,
    start: int,
    end: int | None,
    parent_id: int,
) -> Track:
    result = Track(
        track_id=track_id,
        birth_frame=start,
        color_rgb=source.color_rgb.copy(),
        identity_parent_id=parent_id,
        split_frame=start if track_id != source.track_id else source.split_frame,
    )
    result.observations = {
        key: value for key, value in source.observations.items()
        if key[0] >= start and (end is None or key[0] < end)
    }
    result.samples = {
        frame: value for frame, value in source.samples.items()
        if frame >= start and (end is None or frame < end)
    }
    for observation in result.observations.values():
        observation.track_id = track_id
    for sample in result.samples.values():
        sample.track_id = track_id
    result.motion_class = MotionClass.UNKNOWN
    recompute_track_metadata(result)
    return result


def split_identity_jumps(
    tracks: dict[int, Track], scene_scale: float, config: dict
) -> tuple[dict[int, Track], list[SwitchEvent], dict[str, int]]:
    """Split tracks at high-confidence switch boundaries, preserving lineage."""
    if not bool(config["identity_validation"]["enabled"]):
        return tracks, [], {"track_split_count": 0, "identity_switch_events": 0}
    next_id = max(tracks, default=0) + 1
    output: dict[int, Track] = {}
    all_events: list[SwitchEvent] = []
    minimum_segment = int(config["identity_validation"]["min_segment_frames"])
    for track in sorted(tracks.values(), key=lambda item: item.track_id):
        events = detect_identity_switches(track, scene_scale, config)
        possible_frames = sorted({event.split_frame for event in events})
        starts = [min(track.samples or {track.birth_frame: None})]
        accepted_frames = []
        final_frame = max(track.samples or {track.birth_frame: None}) + 1
        for frame in possible_frames:
            if frame - starts[-1] >= minimum_segment and final_frame - frame >= minimum_segment:
                accepted_frames.append(frame)
                starts.append(frame)
        boundaries = accepted_frames + [None]
        source_parent = track.identity_parent_id if track.identity_parent_id >= 0 else track.track_id
        for segment_index, (start, end) in enumerate(zip(starts, boundaries)):
            segment_id = track.track_id if segment_index == 0 else next_id
            if segment_index > 0:
                next_id += 1
            parent_id = (
                track.identity_parent_id
                if segment_index == 0
                else source_parent
            )
            segment = _segment_track(track, segment_id, start, end, parent_id)
            segment.identity_switch_count = len(accepted_frames)
            output[segment_id] = segment
        all_events.extend(event for event in events if event.split_frame in accepted_frames)
    return output, all_events, {
        "track_split_count": len(output) - len(tracks),
        "identity_switch_events": len(all_events),
    }
