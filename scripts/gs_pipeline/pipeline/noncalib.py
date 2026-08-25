"""无标定分支：用首帧跑 COLMAP SfM 自动估计内外参，再对全部帧去畸变。

复用 multicam_4dgs_pipeline 里两个成熟脚本：
- complete_3dgs_pipeline.py：首帧 SfM → estimated_calib.json + 参考 sparse/0
- undistort_all_frames.py：用 estimated_calib.json 对所有帧去畸变
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .common import (
    PipelineError,
    cam_dirname,
    frame_filename,
    discover_cam_sequences,
    run_checked,
)

ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
COMPLETE_PIPELINE = ENGINE_DIR / "complete_3dgs_pipeline.py"
UNDISTORT_ALL = ENGINE_DIR / "undistort_all_frames.py"


def _build_first_frames(ims_dir: Path, first_frames: Path) -> None:
    """从 ims/camXXX 提取每台相机第 1 帧，铺成扁平 SfM 种子目录。"""
    if first_frames.exists():
        shutil.rmtree(first_frames)
    first_frames.mkdir(parents=True)
    sequences = discover_cam_sequences(ims_dir)
    for cam_id, frames in sequences.items():
        src = frames[0]
        dst = first_frames / frame_filename(cam_id, 1)
        shutil.copy2(src, dst)


def calibrate_and_undistort_sfm(work: Path, ims_dir: Path, python: Path, gpu_ids: list[int],
                                *, undistort_workers: int, dry_run: bool) -> tuple[Path, Path]:
    """返回 (参考 sparse/0 目录, 去畸变全帧目录)。"""
    for script in (COMPLETE_PIPELINE, UNDISTORT_ALL):
        if not script.is_file():
            raise PipelineError(f"缺少脚本: {script}")

    first_frames = work / "first_frames"
    calibration = work / "calibration"
    undistorted = work / "undistorted"
    estimated_calib = calibration / "colmap_sfm" / "estimated_calib.json"
    reference_sparse = calibration / "3dgs_training_data" / "sparse" / "0"

    if not dry_run:
        _build_first_frames(ims_dir, first_frames)
        if calibration.exists():
            shutil.rmtree(calibration)
    else:
        print(f"  [dry-run] 将从 {ims_dir} 提取首帧到 {first_frames}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
    run_checked([
        str(python), str(COMPLETE_PIPELINE),
        "--stage", "all",
        "--feature-method", "superpoint",
        "--matcher-method", "superglue",
        "--undistort-method", "custom",
        "--non-interactive",
        "--images-dir", str(first_frames),
        "--output-dir", str(calibration),
    ], env=env, dry_run=dry_run)

    if not dry_run and not estimated_calib.is_file():
        raise PipelineError(f"SfM 未产出标定文件: {estimated_calib}")

    run_checked([
        str(python), str(UNDISTORT_ALL),
        "--input-dir", str(ims_dir),
        "--calib-file", str(estimated_calib),
        "--output-dir", str(undistorted),
        "--workers", str(undistort_workers),
        "--overwrite",
    ], dry_run=dry_run)

    return reference_sparse, undistorted
