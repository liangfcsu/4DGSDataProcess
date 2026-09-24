"""Feature-based temporal tracking adapter with short-gap reconnect support."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from track_sparse.input.dataset import DatasetIndex
from track_sparse.core.schema import Observation
from track_sparse.matching.spatial_matcher import HlocFeatureMatcher


class TemporalTracker:
    """Propagate feature identities through cached same-camera SuperGlue matches.

    The adapter intentionally exposes observations rather than HLOC tensor layouts.
    It keeps a bounded pair cache so long sequences do not retain every match table.
    """

    def __init__(
        self,
        matcher: HlocFeatureMatcher,
        dataset: DatasetIndex,
        min_score: float,
        cache_size: int = 256,
    ):
        self.matcher = matcher
        self.dataset = dataset
        self.min_score = float(min_score)
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[tuple[int, int, int], dict[int, tuple[int, float]]] = OrderedDict()

    def _links(self, cam_id: int, source_frame: int, target_frame: int) -> dict[int, tuple[int, float]]:
        key = (cam_id, source_frame, target_frame)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        source = self.dataset.get(cam_id, source_frame)
        target = self.dataset.get(cam_id, target_frame)
        if source is None or target is None:
            links: dict[int, tuple[int, float]] = {}
        else:
            feature_ids_a, feature_ids_b, scores = self.matcher.match_links(
                source.relative_name, target.relative_name
            )
            links = {
                int(first): (int(second), float(score))
                for first, second, score in zip(feature_ids_a, feature_ids_b, scores)
                if float(score) >= self.min_score
            }
        self._cache[key] = links
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return links

    def link_feature(
        self, cam_id: int, source_frame: int, target_frame: int, feature_id: int
    ) -> tuple[int, float] | None:
        """Expose one cached temporal graph edge for cycle-consistency checks."""
        return self._links(cam_id, source_frame, target_frame).get(int(feature_id))

    def propagate(
        self,
        track_id: int,
        source: Observation,
        target_frame: int,
    ) -> Observation | None:
        link = self._links(source.cam_id, source.frame_id, target_frame).get(source.feature_id)
        if link is None:
            return None
        target_feature, score = link
        target_record = self.dataset.get(source.cam_id, target_frame)
        if target_record is None:
            return None
        keypoints, feature_scores = self.matcher.features(target_record.relative_name)
        if not (0 <= target_feature < len(keypoints)):
            return None
        uv = keypoints[target_feature]
        visible = bool(
            np.all(np.isfinite(uv))
            and 0 <= uv[0] < target_record.width
            and 0 <= uv[1] < target_record.height
        )
        if not visible:
            return None
        return Observation(
            track_id=track_id,
            frame_id=target_frame,
            cam_id=source.cam_id,
            feature_id=target_feature,
            u=float(uv[0]),
            v=float(uv[1]),
            visible=True,
            tracker_confidence=float(score * float(feature_scores[target_feature])),
            spatial_confidence=0.0,
            source="temporal" if target_frame - source.frame_id == 1 else "reconnect",
            association_score=float(score),
        )
