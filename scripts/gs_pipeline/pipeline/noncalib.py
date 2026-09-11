"""无标定分支：用首帧跑 COLMAP SfM 自动估计内外参，再对全部帧去畸变。

复用 multicam_4dgs_pipeline 里两个成熟脚本：
- complete_3dgs_pipeline.py 的第 1 阶段：首帧 SfM → estimated_calib.json + 参考位姿/点云
- undistort_all_frames.py：用 estimated_calib.json 对所有帧去畸变
去畸变后仅替换参考模型的内参，位姿和点云直接复用已成功的首帧 SfM。
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .common import (
    PipelineError,
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


def _image_camera_pairs(images_txt: Path) -> list[tuple[int, str]]:
    """Read ``(camera_id, image_name)`` pairs from a COLMAP text model."""
    pairs: list[tuple[int, str]] = []
    expect_image_line = True
    for raw_line in images_txt.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            continue
        if expect_image_line:
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 10:
                raise PipelineError(f"COLMAP images.txt 图像行格式错误: {stripped[:120]}")
            pairs.append((int(parts[8]), " ".join(parts[9:])))
            expect_image_line = False
        else:
            # Every image metadata line is followed by exactly one POINTS2D line,
            # which is allowed to be empty.
            expect_image_line = True
    return pairs


def _build_undistorted_reference_sparse(source_sparse: Path, cameras_json: Path,
                                        out_sparse: Path) -> None:
    """Reuse the successful SfM poses/points with the undistorted image intrinsics.

    The legacy no-calibration pipeline discarded the first (usually complete) SfM
    model and ran a second unconstrained reconstruction after undistortion.  Apart
    from being redundant, that reconstruction can fragment into multiple models.
    The camera poses and 3D points do not change under image undistortion, so only
    the camera intrinsics need to be replaced.
    """
    required = [source_sparse / name for name in ("cameras.txt", "images.txt", "points3D.txt")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise PipelineError(f"SfM 参考模型不完整，缺少: {', '.join(missing)}")
    if not cameras_json.is_file():
        raise PipelineError(f"缺少去畸变相机参数: {cameras_json}")

    try:
        entries = json.loads(cameras_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"无法读取去畸变相机参数 {cameras_json}: {exc}") from exc
    if not isinstance(entries, list):
        raise PipelineError(f"去畸变相机参数应为列表: {cameras_json}")

    by_name = {entry.get("img_name"): entry for entry in entries
               if isinstance(entry, dict) and entry.get("img_name")}
    image_camera_pairs = _image_camera_pairs(source_sparse / "images.txt")
    if not image_camera_pairs:
        raise PipelineError(f"SfM 参考模型没有注册图像: {source_sparse / 'images.txt'}")

    by_camera_id: dict[int, dict] = {}
    for camera_id, image_name in image_camera_pairs:
        entry = by_name.get(image_name)
        if entry is None:
            raise PipelineError(f"去畸变参数中找不到 SfM 图像: {image_name}")
        previous = by_camera_id.setdefault(camera_id, entry)
        if previous is not entry:
            raise PipelineError(f"相机 ID {camera_id} 对应多个去畸变参数，无法构建参考模型")

    if len(by_camera_id) != len(entries):
        raise PipelineError(
            f"SfM/去畸变相机数不一致: SfM={len(by_camera_id)}，去畸变={len(entries)}"
        )

    if out_sparse.exists():
        shutil.rmtree(out_sparse)
    out_sparse.mkdir(parents=True)
    with (out_sparse / "cameras.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        handle.write(f"# Number of cameras: {len(by_camera_id)}\n")
        for camera_id, entry in sorted(by_camera_id.items()):
            try:
                width, height = int(entry["width"]), int(entry["height"])
                fx, fy = float(entry["fx"]), float(entry["fy"])
                cx, cy = float(entry["cx"]), float(entry["cy"])
            except (KeyError, TypeError, ValueError) as exc:
                raise PipelineError(
                    f"相机 {camera_id} 的去畸变内参不完整: {entry}"
                ) from exc
            handle.write(
                f"{camera_id} PINHOLE {width} {height} {fx:.17g} {fy:.17g} "
                f"{cx:.17g} {cy:.17g}\n"
            )

    # Poses and points remain in the exact coordinate system estimated by the
    # successful first SfM.  Per-frame triangulation strips the old observations
    # before loading this model, while the final 3DGS dataset reuses its point cloud.
    shutil.copy2(source_sparse / "images.txt", out_sparse / "images.txt")
    shutil.copy2(source_sparse / "points3D.txt", out_sparse / "points3D.txt")


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
    source_sparse = calibration / "colmap_sfm" / "sparse" / "0"
    reference_sparse = calibration / "reference_sparse" / "0"

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
        # Stage 1 already produces the complete SfM poses and point cloud.  Do
        # not discard it and launch the fragile second unconstrained HLOC SfM.
        "--stage", "1",
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

    if not dry_run:
        cameras_json = undistorted / "tool" / "cameras_undistorted.json"
        _build_undistorted_reference_sparse(source_sparse, cameras_json, reference_sparse)
        print(f"  复用首帧 SfM 参考模型: {reference_sparse}")

    return reference_sparse, undistorted
