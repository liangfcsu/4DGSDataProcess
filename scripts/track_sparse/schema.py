"""Typed records shared by the sparse tracking pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np


class TrackState(str, Enum):
    NEW = "new"
    TENTATIVE = "tentative"
    ACTIVE = "active"
    OCCLUDED = "occluded"
    LOST = "lost"
    ENDED = "ended"


class MotionClass(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray
    K: np.ndarray


@dataclass(slots=True)
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    points2d: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))


@dataclass(slots=True)
class Camera:
    cam_id: int
    image_id: int
    camera_id: int
    width: int
    height: int
    K: np.ndarray
    R_w2c: np.ndarray
    t_w2c: np.ndarray
    P: np.ndarray
    center_world: np.ndarray
    reference_image_name: str


@dataclass(slots=True)
class Point3D:
    point3d_id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    track: list[tuple[int, int]]


@dataclass(slots=True)
class ImageRecord:
    cam_id: int
    frame_id: int
    path: Path
    relative_name: str
    width: int = 0
    height: int = 0


@dataclass(slots=True)
class FeatureObservation:
    frame_id: int
    cam_id: int
    feature_id: int
    uv: np.ndarray
    feature_score: float
    spatial_confidence: float = 0.0


@dataclass(slots=True)
class PairMatches:
    feature_ids_a: np.ndarray
    feature_ids_b: np.ndarray
    uv_a: np.ndarray
    uv_b: np.ndarray
    scores: np.ndarray


@dataclass(slots=True)
class SpatialGroup:
    frame_id: int
    observations: dict[int, FeatureObservation]
    xyz: np.ndarray
    reprojection_rmse: float
    triangulation_angle_deg: float
    geometry_confidence: float
    color_rgb: np.ndarray = field(default_factory=lambda: np.array([255, 255, 255], dtype=np.uint8))


@dataclass(slots=True)
class Observation:
    track_id: int
    frame_id: int
    cam_id: int
    feature_id: int
    u: float
    v: float
    visible: bool = True
    tracker_confidence: float = 0.0
    spatial_confidence: float = 0.0
    reprojection_error: float = float("nan")
    is_inlier: bool = False
    source: str = "temporal"


@dataclass(slots=True)
class TrackSample:
    track_id: int
    frame_id: int
    xyz: np.ndarray
    valid_3d: bool
    num_visible_views: int
    num_inlier_views: int
    reprojection_rmse: float
    min_triangulation_angle_deg: float
    geometry_confidence: float
    sparse_support_distance: float = float("nan")
    state: TrackState = TrackState.NEW


@dataclass(slots=True)
class Track:
    track_id: int
    birth_frame: int
    state: TrackState = TrackState.NEW
    motion_class: MotionClass = MotionClass.UNKNOWN
    observations: dict[tuple[int, int], Observation] = field(default_factory=dict)
    samples: dict[int, TrackSample] = field(default_factory=dict)
    last_valid_frame: int = -1
    valid_frame_count: int = 0
    longest_valid_run: int = 0
    current_valid_run: int = 0
    missing_count: int = 0
    recovery_count: int = 0
    color_rgb: np.ndarray = field(default_factory=lambda: np.array([255, 255, 255], dtype=np.uint8))
    mean_confidence: float = 0.0
    median_reprojection_error: float = float("nan")
    quality: str = "low"


@dataclass(slots=True)
class TriangulationResult:
    valid: bool
    xyz: np.ndarray
    inliers: np.ndarray
    errors: np.ndarray
    rmse: float
    angle_deg: float
    confidence: float
    reason: str = ""
