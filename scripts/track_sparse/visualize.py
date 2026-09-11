"""Auditable 2D overlays and 3D trajectory PLY visualisations."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .dataset import DatasetIndex
from .schema import SpatialGroup, Track


def assign_group_colors(groups_by_frame: dict[int, list[SpatialGroup]], dataset: DatasetIndex) -> None:
    """Use the robust median of seed-view pixels as each track candidate's RGB."""
    for frame_id, groups in groups_by_frame.items():
        images: dict[int, np.ndarray] = {}
        for group in groups:
            colors = []
            for cam_id, observation in group.observations.items():
                record = dataset.get(cam_id, frame_id)
                if record is None:
                    continue
                if cam_id not in images:
                    images[cam_id] = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
                image = images[cam_id]
                if image is None:
                    continue
                u, v = int(round(float(observation.uv[0]))), int(round(float(observation.uv[1])))
                if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                    colors.append(image[v, u, ::-1])
            if colors:
                group.color_rgb = np.clip(np.median(np.asarray(colors), axis=0), 0, 255).astype(np.uint8)


def _id_color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    rgb = rng.integers(64, 256, size=3, dtype=np.uint8)
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def write_track_overlays(
    tracks: dict[int, Track],
    dataset: DatasetIndex,
    frames: list[int],
    output_dir: Path,
    max_cameras: int | None = None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cam_ids = dataset.cam_ids[:max_cameras] if max_cameras else dataset.cam_ids
    written = 0
    for frame_id in frames:
        for cam_id in cam_ids:
            record = dataset.get(cam_id, frame_id)
            if record is None:
                continue
            image = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            scale = min(1.0, 1600.0 / max(image.shape[:2]))
            for track in tracks.values():
                observation = track.observations.get((frame_id, cam_id))
                if observation is None:
                    continue
                color = _id_color(track.track_id)
                position = (int(round(observation.u)), int(round(observation.v)))
                cv2.circle(image, position, 4, color, -1, lineType=cv2.LINE_AA)
                if observation.is_inlier:
                    cv2.circle(image, position, 7, color, 1, lineType=cv2.LINE_AA)
                cv2.putText(
                    image, str(track.track_id), (position[0] + 5, position[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
                )
                history = []
                for previous_frame in sorted(dataset.frame_ids):
                    if previous_frame > frame_id or previous_frame < frame_id - 10:
                        continue
                    previous = track.observations.get((previous_frame, cam_id))
                    if previous is not None:
                        history.append((int(round(previous.u)), int(round(previous.v))))
                if len(history) >= 2:
                    cv2.polylines(image, [np.asarray(history, dtype=np.int32)], False, color, 1, cv2.LINE_AA)
            if scale < 1.0:
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            destination = output_dir / f"cam{cam_id:03d}_frame{frame_id:03d}.jpg"
            cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            written += 1
    return written


def write_trajectory_ply(tracks: dict[int, Track], path: Path) -> None:
    vertices: list[tuple] = []
    edges: list[tuple[int, int]] = []
    for track in sorted(tracks.values(), key=lambda item: item.track_id):
        previous_index: int | None = None
        previous_frame: int | None = None
        for frame_id, sample in sorted(track.samples.items()):
            if not sample.valid_3d:
                previous_index = previous_frame = None
                continue
            current = len(vertices)
            rgb = track.color_rgb
            xyz = sample.output_xyz
            motion_code = {"unknown": 0, "static": 1, "dynamic": 2}[track.motion_class.value]
            vertices.append((
                xyz[0], xyz[1], xyz[2], rgb[0], rgb[1], rgb[2],
                track.track_id, frame_id, motion_code, track.motion_group_id,
                sample.position_std,
            ))
            if previous_index is not None and previous_frame is not None and frame_id == previous_frame + 1:
                edges.append((previous_index, current))
            previous_index, previous_frame = current, frame_id
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("property uint track_id\nproperty int frame_id\n")
        handle.write("property uchar motion_class\nproperty int motion_group_id\n")
        handle.write("property float position_std\n")
        handle.write(f"element edge {len(edges)}\n")
        handle.write("property int vertex1\nproperty int vertex2\nend_header\n")
        for vertex in vertices:
            handle.write(" ".join(map(str, vertex)) + "\n")
        for edge in edges:
            handle.write(f"{edge[0]} {edge[1]}\n")
