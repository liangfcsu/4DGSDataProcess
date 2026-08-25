#!/usr/bin/env python3
"""统一多相机 → 4DGS 数据处理流水线（单一入口）。

一条命令覆盖输入 × 标定的全部组合：

  输入类型（--input-type，默认 auto 自动判别）
    videos  多相机视频           camXX.mp4
    images  多相机图像           camXXX/ 每台一段序列，或扁平目录每张一台相机

  标定来源（互斥；都不给则走无标定自动 SfM）
    --rig-json   FILE   refined_rig_group.json（有理径向畸变，rig 已知内外参）
    --calib-json FILE   libCalib calib*.json（多项式畸变，已知内外参）
    （缺省）            无标定：首帧 COLMAP SfM 自动估计内外参

输出（--output）：images/ ims/ persparse/ sparse/0/，可直接用于 4DGS 训练。

示例：
  # 多相机视频 + 无标定
  python run.py --input data/cook_spinach --output data/cook_spinach_4dgs --max-frames 30

  # 多相机多帧图像 + rig 标定
  python run.py --input data/rig_seq --output data/rig_4dgs --rig-json data/rig_seq/refined_rig_group.json

  # 多相机单帧图像 + libCalib 标定
  python scripts/gs_pipeline/run.py --input data/testiamges/liang --output data/testiamges/liangoutputcalib --calib-json data/testiamges/calib0.3343.json

  # 先看清每一步会跑什么命令，不实际执行
  python run.py --input data/x --output data/x_4dgs --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import acquire, assemble as assemble_stage, calibrated, noncalib
from pipeline import sparse as sparse_stage
from pipeline.common import (
    PipelineError,
    VIDEO_SUFFIXES,
    choose_python,
    detect_gpu_ids,
    discover_cam_sequences,
    frame_count_of,
    list_images,
    run_checked,
)

CONVERT_CALIB = Path(__file__).resolve().parent / "stages" / "convert_calib_to_cameras_json.py"


# ---------------------------------------------------------------------------
# 输入类型判别
# ---------------------------------------------------------------------------

def detect_input_type(input_dir: Path) -> str:
    if any(p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES for p in input_dir.iterdir()):
        return "videos"
    return "images"


# ---------------------------------------------------------------------------
# libCalib：calib*.json → 扁平 cameras.json（复用 1.convert_calib_to_cameras_json.py）
# ---------------------------------------------------------------------------

def libcalib_to_cameras_json(calib_json: Path, seed_images_dir: Path, out_json: Path,
                             python: Path, *, dry_run: bool) -> None:
    if not CONVERT_CALIB.is_file():
        raise PipelineError(f"缺少标定转换脚本: {CONVERT_CALIB}")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    run_checked([
        str(python), str(CONVERT_CALIB),
        "--calib", str(calib_json),
        "--images-dir", str(seed_images_dir),
        "--output", str(out_json),
        "--verify-images",
    ], dry_run=dry_run)


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="多相机（视频/图像，标定/无标定）到 4DGS 初始数据集的统一流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="输入目录（视频目录或图像目录）")
    parser.add_argument("--output", required=True, help="最终 4DGS 数据集目录")
    parser.add_argument("--input-type", choices=["auto", "videos", "images"], default="auto")

    calib = parser.add_mutually_exclusive_group()
    calib.add_argument("--rig-json", default=None, help="refined_rig_group.json（rig 标定）")
    calib.add_argument("--calib-json", default=None, help="libCalib calib*.json（多项式标定）")

    parser.add_argument("--work-dir", default=None, help="中间目录（默认 <output>.work）")
    parser.add_argument("--gpus", default="auto", help="auto 或逗号分隔的 GPU 编号，如 0,1,2,3")
    parser.add_argument("--python", default=None, help="指定已装依赖的 Python 解释器")

    # 提帧（仅视频输入）
    parser.add_argument("--max-frames", type=int, default=0,
                        help="每台相机最多提取前 N 帧；0=全部（默认 0）")
    parser.add_argument("--frames-per-second", type=int, default=None,
                        help="按每秒均匀采样 N 帧（默认按原始帧顺序）")
    parser.add_argument("--frame", type=int, default=None,
                        help="只处理指定的单帧（1 起）：视频取第 N 帧、多帧序列取第 N 张，"
                             "产出单帧数据集；设置后忽略 --max-frames/--frames-per-second")

    # rig 去畸变高级项
    parser.add_argument("--rig-pose-mode", default="wtc_center")
    parser.add_argument("--rig-radial-mode", default="direct",
                        choices=["direct", "swap", "negate", "swap_negate"])
    parser.add_argument("--rig-small-terms", default="tangential_d8d9",
                        choices=["none", "tangential_d8d9", "thin_prism_d8d9"])
    parser.add_argument("--rig-direction", default="ideal_to_observed",
                        choices=["ideal_to_observed", "observed_to_ideal"])
    parser.add_argument("--rig-scale", type=float, default=1.0)
    parser.add_argument("--rig-iterations", type=int, default=8)

    # SuperGlue / 去畸变通用项
    parser.add_argument("--max-keypoints", type=int, default=4096)
    parser.add_argument("--resize-max", type=int, default=4000)
    parser.add_argument("--superglue-weights", choices=["indoor", "outdoor"], default="indoor")
    parser.add_argument("--undistort-workers", type=int, default=8)

    parser.add_argument("--overwrite", action="store_true", help="清空并覆盖已有输出/中间结果")
    parser.add_argument("--keep-intermediate", action="store_true", help="成功后保留中间目录")
    parser.add_argument("--dry-run", action="store_true", help="只打印每一步计划与命令，不实际执行")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    dry_run = args.dry_run

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    work = Path(args.work_dir).expanduser().resolve() if args.work_dir \
        else output_dir.parent / f".{output_dir.name}.work"

    if not input_dir.is_dir():
        raise PipelineError(f"输入目录不存在: {input_dir}")
    if output_dir == input_dir or output_dir in input_dir.parents:
        raise PipelineError("输出目录不能是输入目录或其父目录")
    if args.frame is not None and args.frame < 1:
        raise PipelineError("--frame 必须是不小于 1 的帧号")

    calib_mode = "rig" if args.rig_json else "libcalib" if args.calib_json else "none"
    input_type = args.input_type if args.input_type != "auto" else detect_input_type(input_dir)

    gpu_ids = detect_gpu_ids(args.gpus)
    python = Path(sys.executable) if dry_run else choose_python(args.python)
    if not dry_run and shutil.which("colmap") is None:
        raise PipelineError("未找到 colmap 可执行文件（首帧 SfM 与逐帧三角化均依赖 COLMAP）")

    print("=" * 72)
    print("统一 4DGS 数据处理流水线")
    print(f"输入: {input_dir}（类型: {input_type}）")
    print(f"标定: {calib_mode}" + (
        f"  <- {args.rig_json}" if calib_mode == "rig"
        else f"  <- {args.calib_json}" if calib_mode == "libcalib" else "（自动 SfM）"))
    print(f"输出: {output_dir}")
    print(f"GPU: {gpu_ids}   Python: {python}")
    if dry_run:
        print("模式: DRY-RUN（仅打印计划）")
    print("=" * 72)

    if not dry_run:
        if output_dir.exists():
            if not args.overwrite:
                raise PipelineError(f"输出已存在：{output_dir}；请换目录或加 --overwrite")
            shutil.rmtree(output_dir)
        if work.exists():
            if not args.overwrite:
                raise PipelineError(f"中间目录已存在：{work}；请加 --overwrite")
            shutil.rmtree(work)
        work.mkdir(parents=True)

    ims_dir = work / "ims"

    # ---- 阶段 1：采集为 camXXX/camXXXframeYYY.png ----
    print("\n[1/4] 采集输入 → 规范帧布局" + (f"（只处理第 {args.frame} 帧）" if args.frame else ""))
    if input_type == "videos":
        if args.frame is not None:
            acquire.acquire_single_frame_from_videos(input_dir, ims_dir, args.frame, dry_run=dry_run)
        else:
            acquire.acquire_from_videos(
                input_dir, ims_dir, python, gpu_ids,
                max_frames=args.max_frames, frames_per_second=args.frames_per_second,
                resume=False, dry_run=dry_run,
            )
    else:
        acquire.acquire_from_images(input_dir, ims_dir, select_frame=args.frame, dry_run=dry_run)

    if dry_run:
        return _dry_run_finish(calib_mode, input_dir, input_type, python, gpu_ids, args, work, output_dir)

    sequences = discover_cam_sequences(ims_dir)
    cam_ids = sorted(sequences.keys())
    frame_count = frame_count_of(sequences)
    print(f"  相机数: {len(cam_ids)}，每相机帧数: {frame_count}")

    # ---- 阶段 2：标定 + 全帧去畸变 → 参考 sparse/0 + undistorted ----
    print(f"\n[2/4] 标定与去畸变（模式: {calib_mode}）")
    if calib_mode == "none":
        reference_sparse, undistorted = noncalib.calibrate_and_undistort_sfm(
            work, ims_dir, python, gpu_ids,
            undistort_workers=args.undistort_workers, dry_run=dry_run,
        )
    else:
        if calib_mode == "rig":
            cameras = calibrated.load_rig_calibration(Path(args.rig_json).resolve(), cam_ids, {
                "pose_mode": args.rig_pose_mode, "radial_mode": args.rig_radial_mode,
                "small_terms": args.rig_small_terms, "direction": args.rig_direction,
                "scale": args.rig_scale, "iterations": args.rig_iterations,
            })
        else:  # libcalib
            seed = work / "calibration" / "seed_first_frames"
            noncalib._build_first_frames(ims_dir, seed)
            cameras_json = work / "calibration" / "cameras.json"
            libcalib_to_cameras_json(Path(args.calib_json).resolve(), seed, cameras_json,
                                     python, dry_run=False)
            cameras = calibrated.load_libcalib_calibration(cameras_json, cam_ids)

        undistorted = work / "undistorted"
        reference_sparse = work / "calibration" / "sparse" / "0"
        print(f"  去畸变 {len(cameras)} 台相机 × {frame_count} 帧 …")
        new_intrinsics = calibrated.undistort_all(cameras, sequences, undistorted,
                                                   args.undistort_workers)
        calibrated.build_reference_sparse(cameras, new_intrinsics, reference_sparse)
        print(f"  参考模型: {reference_sparse}")

    # ---- 阶段 3：逐帧稀疏点云 ----
    print("\n[3/4] 逐帧稀疏点云（多 GPU）")
    persparse = work / "persparse"
    sparse_stage.generate_per_frame_sparse(
        undistorted, reference_sparse, persparse, frame_count, python, gpu_ids,
        sg_opts={"max_keypoints": args.max_keypoints, "resize_max": args.resize_max,
                 "superglue_weights": args.superglue_weights},
        dry_run=dry_run,
    )

    # ---- 阶段 4：组装 ----
    print("\n[4/4] 组装 4DGS 数据集")
    assemble_stage.assemble(output_dir, reference_sparse, undistorted, persparse, dry_run=dry_run)

    if not args.keep_intermediate:
        shutil.rmtree(work, ignore_errors=True)

    print("\n" + "=" * 72)
    print("🎉 4DGS 初始数据集已生成")
    print(f"输出: {output_dir}")
    for name in ("images", "ims", "persparse", "sparse"):
        print(f"  - {output_dir / name}")
    print("=" * 72)
    return 0


def _dry_run_finish(calib_mode, input_dir, input_type, python, gpu_ids, args, work, output_dir) -> int:
    """dry-run 下无中间文件，逐阶段打印后续将执行的命令。"""
    print("\n[2/4] 标定与去畸变（模式: %s）" % calib_mode)
    if calib_mode == "none":
        noncalib.calibrate_and_undistort_sfm(
            work, work / "ims", python, gpu_ids,
            undistort_workers=args.undistort_workers, dry_run=True)
    elif calib_mode == "rig":
        print("  将用有理径向模型对每台相机全部帧去畸变，并写出 PINHOLE 参考 sparse/0（纯 numpy/cv2，无子进程）")
    else:
        print("  将用 1.convert_calib_to_cameras_json.py 生成 cameras.json，再多项式去畸变并写出参考 sparse/0")
    print("\n[3/4] 逐帧稀疏点云（多 GPU）")
    ref_display = (work / "calibration" / "3dgs_training_data" / "sparse" / "0") \
        if calib_mode == "none" else (work / "calibration" / "sparse" / "0")
    sparse_stage.generate_per_frame_sparse(
        work / "undistorted", ref_display, work / "persparse",
        frame_count=1, python=python, gpu_ids=gpu_ids,
        sg_opts={"max_keypoints": args.max_keypoints, "resize_max": args.resize_max,
                 "superglue_weights": args.superglue_weights}, dry_run=True)
    print("\n[4/4] 组装 4DGS 数据集")
    assemble_stage.assemble(output_dir, work / "calibration" / "sparse" / "0",
                            work / "undistorted", work / "persparse", dry_run=True)
    print("\n[dry-run] 计划打印完毕，未执行任何实际处理。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⏹️  已中断。", file=sys.stderr)
        sys.exit(130)
    except PipelineError as exc:
        print(f"\n❌ 流水线失败: {exc}", file=sys.stderr)
        sys.exit(1)
