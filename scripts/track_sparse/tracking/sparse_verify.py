"""Optional per-frame sparse-cloud support scoring (never used for identity)."""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np

from track_sparse.input.io_colmap import read_points3d_text

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None


FRAME_RE = re.compile(r"frame[_-]?0*(\d+)_points3D\.txt$", re.IGNORECASE)


def discover_per_frame_clouds(dataset: str | Path | None, explicit: str | Path | None) -> dict[int, Path]:
    roots: list[Path] = []
    if explicit:
        roots.append(Path(explicit).expanduser().resolve())
    elif dataset:
        dataset_path = Path(dataset).expanduser().resolve()
        roots.extend((dataset_path / "persparse", dataset_path / "persparse_shards"))
    files: dict[int, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("frame*_points3D.txt"), key=lambda item: (len(item.parts), str(item))):
            match = FRAME_RE.search(path.name)
            if match:
                files.setdefault(int(match.group(1)), path.resolve())
    return files


class SparseCloudVerifier:
    def __init__(self, files: dict[int, Path], nn_scale: float):
        self.files = files
        self.nn_scale = float(nn_scale)
        self._cache: OrderedDict[int, tuple[object, float] | None] = OrderedDict()

    def _load(self, frame_id: int):
        if frame_id in self._cache:
            self._cache.move_to_end(frame_id)
            return self._cache[frame_id]
        path = self.files.get(frame_id)
        if path is None or cKDTree is None:
            self._cache[frame_id] = None
            return None
        points = read_points3d_text(path)
        xyz = np.asarray([point.xyz for point in points.values()], dtype=np.float64)
        if len(xyz) < 2:
            self._cache[frame_id] = None
            return None
        tree = cKDTree(xyz)
        sample = xyz if len(xyz) <= 5000 else xyz[np.linspace(0, len(xyz) - 1, 5000, dtype=np.int64)]
        distances, _ = tree.query(sample, k=2)
        spacing = float(np.median(distances[:, 1]))
        self._cache[frame_id] = (tree, max(spacing * self.nn_scale, 1e-12))
        self._cache.move_to_end(frame_id)
        while len(self._cache) > 4:
            self._cache.popitem(last=False)
        return self._cache[frame_id]

    def score(self, frame_id: int, xyz: np.ndarray) -> tuple[float, float] | None:
        loaded = self._load(frame_id)
        if loaded is None:
            return None
        tree, threshold = loaded
        distance = float(tree.query(np.asarray(xyz, dtype=np.float64), k=1)[0])
        confidence = math.exp(-(distance**2) / (2.0 * threshold**2))
        return distance, confidence
