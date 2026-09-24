"""Configuration loading, merging and validation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


class ConfigError(ValueError):
    pass


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ConfigError("默认配置文件根节点必须是映射")
    config.pop("run", None)
    if path:
        with Path(path).expanduser().open("r", encoding="utf-8") as handle:
            user = yaml.safe_load(handle) or {}
        if not isinstance(user, dict):
            raise ConfigError("配置文件根节点必须是映射")
        # ``run`` belongs to run.py (paths, GPU and resume mode), not to the
        # algorithm configuration or its cache fingerprint.
        user = copy.deepcopy(user)
        user.pop("run", None)
        config = _deep_merge(config, user)
    validate_config(config)
    return config


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        node = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    validate_config(result)
    return result


def validate_config(config: dict[str, Any]) -> None:
    positive = {
        "camera_graph.top_k": config["camera_graph"]["top_k"],
        "features.max_keypoints": config["features"]["max_keypoints"],
        "spawn.interval": config["spawn"]["interval"],
        "spawn.min_seed_views": config["spawn"]["min_seed_views"],
        "triangulation.min_views": config["triangulation"]["min_views"],
        "triangulation.max_reprojection_error_px": config["triangulation"]["max_reprojection_error_px"],
        "track.max_lost_gap": config["track"]["max_lost_gap"],
        "temporal.min_camera_votes": config["temporal"]["min_camera_votes"],
        "uncertainty.min_pixel_sigma": config["uncertainty"]["min_pixel_sigma"],
        "identity_validation.min_track_frames": config["identity_validation"]["min_track_frames"],
        "identity_validation.min_segment_frames": config["identity_validation"]["min_segment_frames"],
        "motion_models.min_valid_frames": config["motion_models"]["min_valid_frames"],
        "motion_models.max_static_observations": config["motion_models"]["max_static_observations"],
        "pose_refinement.min_points": config["pose_refinement"]["min_points"],
        "motion_groups.min_group_size": config["motion_groups"]["min_group_size"],
        "motion_groups.rigid_fit_min_points": config["motion_groups"]["rigid_fit_min_points"],
        "performance.cpu_workers": config["performance"]["cpu_workers"],
        "performance.gpu_hypothesis_batch_size": config["performance"]["gpu_hypothesis_batch_size"],
        "performance.checkpoint_interval_frames": config["performance"]["checkpoint_interval_frames"],
    }
    invalid = [name for name, value in positive.items() if float(value) <= 0]
    if invalid:
        raise ConfigError(f"以下配置必须大于 0: {', '.join(invalid)}")
    if config["spawn"]["min_seed_views"] < config["triangulation"]["min_views"]:
        raise ConfigError("spawn.min_seed_views 不能小于 triangulation.min_views")
    if config["track"]["max_occlusion_gap"] > config["track"]["max_lost_gap"]:
        raise ConfigError("track.max_occlusion_gap 不能大于 track.max_lost_gap")
    offsets = config["temporal"]["offsets"]
    if not offsets or any(int(v) <= 0 for v in offsets):
        raise ConfigError("temporal.offsets 必须包含正整数")
    if config["temporal"].get("tracker") != "superglue":
        raise ConfigError("当前内置时间跟踪后端仅支持 temporal.tracker=superglue")
    if config["performance"]["triangulation_backend"] not in {"auto", "cpu", "cuda"}:
        raise ConfigError("performance.triangulation_backend 必须是 auto/cpu/cuda")
    if int(config["performance"]["feature_cache_images"]) < 0:
        raise ConfigError("performance.feature_cache_images 不能小于 0")
    fractions = {
        "association.cycle_inconsistent_weight": config["association"]["cycle_inconsistent_weight"],
        "identity_validation.cycle_bad_fraction": config["identity_validation"]["cycle_bad_fraction"],
        "pose_refinement.min_static_confidence": config["pose_refinement"]["min_static_confidence"],
        "pose_refinement.blend": config["pose_refinement"]["blend"],
        "motion_groups.rigid_blend": config["motion_groups"]["rigid_blend"],
    }
    invalid_fractions = [name for name, value in fractions.items() if not 0.0 <= float(value) <= 1.0]
    if invalid_fractions:
        raise ConfigError(f"以下配置必须在 [0, 1] 内: {', '.join(invalid_fractions)}")
    if int(config["motion_groups"]["min_group_size"]) < int(config["motion_groups"]["rigid_fit_min_points"]):
        raise ConfigError("motion_groups.min_group_size 不能小于 rigid_fit_min_points")
