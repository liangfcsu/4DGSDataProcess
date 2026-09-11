"""Adapter around the repository's vendored HLOC SuperPoint/SuperGlue stack."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

from .schema import PairMatches


def vendored_hloc_root() -> Path:
    return Path(__file__).resolve().parents[1] / "gs_pipeline" / "engine" / "Hierarchical-Localization"


def _ensure_hloc_importable() -> None:
    root = vendored_hloc_root()
    if not (root / "hloc" / "extract_features.py").is_file():
        raise FileNotFoundError(f"未找到仓库内置 HLOC: {root}")
    value = str(root)
    if value not in sys.path:
        sys.path.insert(0, value)


def pair_key(name_a: str, name_b: str) -> str:
    return "/".join((name_a.replace("/", "-"), name_b.replace("/", "-")))


class HlocFeatureMatcher:
    """Caches original-image pixel keypoints and pairwise matches in HDF5."""

    def __init__(self, images_dir: Path, work_dir: Path, config: dict):
        self.images_dir = Path(images_dir)
        self.work_dir = Path(work_dir)
        self.config = config
        self.features_path = self.work_dir / "features.h5"
        self.matches_path = self.work_dir / "matches.h5"
        self.pairs_path = self.work_dir / "pairs.txt"
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, image_names: list[str], overwrite: bool = False) -> Path:
        _ensure_hloc_importable()
        from hloc import extract_features

        features = self.config["features"]
        if features["extractor"] != "superpoint":
            raise ValueError("当前内置后端仅支持 features.extractor=superpoint")
        conf = {
            "model": {
                "name": "superpoint",
                "nms_radius": int(features["nms_radius"]),
                "keypoint_threshold": float(features["keypoint_threshold"]),
                "max_keypoints": int(features["max_keypoints"]),
            },
            "preprocessing": {
                "grayscale": True,
                "resize_max": int(features["resize_max"]),
            },
        }
        extract_features.main(
            conf,
            self.images_dir,
            image_list=image_names,
            feature_path=self.features_path,
            as_half=bool(features.get("half_precision_cache", True)),
            overwrite=overwrite,
        )
        return self.features_path

    def match_pairs(self, pairs: list[tuple[str, str]], overwrite: bool = False) -> Path:
        _ensure_hloc_importable()
        from hloc import match_features

        unique: list[tuple[str, str]] = []
        seen: set[frozenset[str]] = set()
        for first, second in pairs:
            if first == second:
                continue
            key = frozenset((first, second))
            if key not in seen:
                seen.add(key)
                unique.append((first, second))
        self.pairs_path.write_text("".join(f"{a} {b}\n" for a, b in unique), encoding="utf-8")
        matching = self.config["matching"]
        if matching["matcher"] != "superglue":
            raise ValueError("当前内置后端仅支持 matching.matcher=superglue")
        conf = {
            "model": {
                "name": "superglue",
                "weights": matching["weights"],
                "sinkhorn_iterations": int(matching["sinkhorn_iterations"]),
                "match_threshold": float(matching["min_score"]),
            }
        }
        match_features.main(
            conf,
            self.pairs_path,
            features=self.features_path,
            matches=self.matches_path,
            overwrite=overwrite,
        )
        return self.matches_path

    def features(self, image_name: str) -> tuple[np.ndarray, np.ndarray]:
        with h5py.File(self.features_path, "r", libver="latest") as handle:
            if image_name not in handle:
                raise KeyError(f"特征缓存中没有 {image_name}")
            group = handle[image_name]
            keypoints = np.asarray(group["keypoints"], dtype=np.float32)
            scores = np.asarray(group["scores"], dtype=np.float32)
        return keypoints, scores

    def match_links(self, name_a: str, name_b: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read only feature IDs and scores, avoiding keypoint I/O for graph building."""
        reverse = False
        direct, flipped = pair_key(name_a, name_b), pair_key(name_b, name_a)
        with h5py.File(self.matches_path, "r", libver="latest") as handle:
            if direct in handle:
                group = handle[direct]
            elif flipped in handle:
                group = handle[flipped]
                reverse = True
            else:
                empty_i = np.empty(0, dtype=np.int32)
                return empty_i, empty_i.copy(), np.empty(0, dtype=np.float32)
            matches0 = np.asarray(group["matches0"], dtype=np.int32)
            scores0 = np.asarray(
                group.get("matching_scores0", np.ones_like(matches0)), dtype=np.float32
            )
        source_ids = np.flatnonzero(matches0 >= 0).astype(np.int32)
        target_ids = matches0[source_ids].astype(np.int32)
        scores = scores0[source_ids]
        return (target_ids, source_ids, scores) if reverse else (source_ids, target_ids, scores)

    def matches(self, name_a: str, name_b: str) -> PairMatches:
        ids_a, ids_b, scores = self.match_links(name_a, name_b)
        keypoints_a, _ = self.features(name_a)
        keypoints_b, _ = self.features(name_b)
        return PairMatches(ids_a, ids_b, keypoints_a[ids_a], keypoints_b[ids_b], scores)
