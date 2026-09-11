#!/usr/bin/env python3
"""Single CLI entry point for sparse multi-view trajectory construction."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


REQUIRED_MODULES = (
    "numpy", "cv2", "scipy", "h5py", "yaml", "torch", "pycolmap", "PIL", "tqdm", "packaging"
)


def _runtime_ready() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES)


def _explicit_python(argv: list[str]) -> str | None:
    for index, value in enumerate(argv):
        if value == "--python" and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith("--python="):
            return value.split("=", 1)[1]
    return None


def _python_works(path: Path) -> bool:
    probe = ";".join(f"import {name}" for name in REQUIRED_MODULES)
    try:
        return path.is_file() and os.access(path, os.X_OK) and subprocess.run(
            [str(path), "-c", probe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _bootstrap_runtime() -> None:
    explicit = _explicit_python(sys.argv[1:])
    current = Path(sys.executable).resolve()
    if _runtime_ready() and (not explicit or Path(explicit).expanduser().resolve() == current):
        return
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "bin" / "python")
    candidates.extend([
        Path("/opt/4dgs-player/env/bin/python3.11"),
        Path.home() / "miniconda3" / "envs" / "4dgsplayer" / "bin" / "python",
        Path.home() / "miniconda3" / "envs" / "3dgslf" / "bin" / "python",
        Path.home() / "miniconda3" / "envs" / "4dgs" / "bin" / "python",
    ])
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate == current:
            continue
        if _python_works(candidate):
            os.execv(str(candidate), [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]])
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    raise SystemExit(
        "找不到稀疏轨迹所需的 Python 环境。缺少: " + ", ".join(missing)
        + "。请激活项目环境或使用 --python /path/to/python。"
    )


_bootstrap_runtime()

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from track_sparse.config import apply_overrides, load_config  # noqa: E402
from track_sparse.pipeline import PipelineOptions, STAGES, run_pipeline  # noqa: E402
from track_sparse.preprocess import PreprocessOptions, PreparedInput, prepare_input  # noqa: E402


def _csv_ints(value: str | None) -> list[int]:
    if not value:
        return []
    try:
        return sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"应为逗号分隔整数: {value}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="固定多相机二维时空匹配 + 每帧重三角化的稀疏 3D 轨迹构建"
    )
    parser.add_argument(
        "--input", "--dataset", dest="input",
        help="原始多相机视频/图像目录；也兼容已有 gs_pipeline 数据集（--dataset 是别名）",
    )
    parser.add_argument(
        "--input-type", choices=["auto", "videos", "images", "prepared"], default="auto",
        help="默认自动判断原始视频、原始图像或已预处理数据集",
    )
    parser.add_argument("--images-dir", help="高级兼容入口：直接指定去畸变图像根目录")
    parser.add_argument("--sparse-dir", help="高级兼容入口：直接指定固定 COLMAP sparse/0")
    parser.add_argument("--persparse-dir", help="可逐帧 frameXXX_points3D.txt 根目录")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--preprocess-dir", type=Path,
        help="抽帧/标定/去畸变缓存目录（默认 <output>/preprocess）",
    )
    parser.add_argument("--config", help="覆盖默认 YAML 配置")
    parser.add_argument("--python", help="具备 torch/OpenCV/HDF5 的 Python 解释器")
    parser.add_argument("--gpu", default="0", help="GPU 编号，或 cpu")
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="原始视频每相机最多抽取 N 帧；缺省时沿用 --end-frame，均未给则处理全部",
    )
    parser.add_argument(
        "--frames-per-second", type=int,
        help="原始视频按每秒均匀抽取 N 帧；缺省为逐帧抽取",
    )
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument("--rig-json", type=Path, help="可选的 refined_rig_group.json")
    calibration.add_argument(
        "--calib-json", type=Path,
        help="可选已知内外参 JSON；不提供时从多相机首帧自动 SfM 标定",
    )
    parser.add_argument("--undistort-workers", type=int, default=8)
    parser.add_argument("--rig-pose-mode", default="wtc_center")
    parser.add_argument(
        "--rig-radial-mode", default="direct",
        choices=["direct", "swap", "negate", "swap_negate"],
    )
    parser.add_argument(
        "--rig-small-terms", default="tangential_d8d9",
        choices=["none", "tangential_d8d9", "thin_prism_d8d9"],
    )
    parser.add_argument(
        "--rig-direction", default="ideal_to_observed",
        choices=["ideal_to_observed", "observed_to_ideal"],
    )
    parser.add_argument("--rig-scale", type=float, default=1.0)
    parser.add_argument("--rig-iterations", type=int, default=8)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--cameras", help="仅处理这些 cam_id，例如 0,1,2,3")
    parser.add_argument("--spawn-interval", type=int)
    parser.add_argument("--camera-neighbors", type=int)
    parser.add_argument("--min-seed-views", type=int)
    parser.add_argument("--min-triangulation-views", type=int)
    parser.add_argument("--max-reprojection-error", type=float)
    parser.add_argument("--min-triangulation-angle", type=float)
    parser.add_argument("--max-keypoints", type=int)
    parser.add_argument("--resize-max", type=int)
    parser.add_argument("--spatial-matcher", choices=["superglue"], default=None)
    parser.add_argument("--temporal-tracker", choices=["superglue"], default=None)
    parser.add_argument("--stages", default="all", help=f"逗号分隔；可选 all,{','.join(STAGES)}")
    parser.add_argument("--debug-frames", help="输出这些帧的 2D overlay，例如 1,10,20")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preprocess-only", action="store_true", help="只完成抽帧、标定和去畸变")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input and not (args.images_dir and args.sparse_dir):
        raise SystemExit("必须提供 --input/--dataset，或同时提供 --images-dir 与 --sparse-dir")
    if bool(args.images_dir) != bool(args.sparse_dir):
        raise SystemExit("--images-dir 与 --sparse-dir 必须同时提供")
    if args.input and (args.images_dir or args.sparse_dir):
        raise SystemExit("--input/--dataset 不能与 --images-dir/--sparse-dir 混用")
    if args.preprocess_only and not args.input:
        raise SystemExit("--preprocess-only 需要 --input/--dataset 原始输入")
    if args.start_frame is not None and args.end_frame is not None and args.start_frame > args.end_frame:
        raise SystemExit("--start-frame 不能大于 --end-frame")
    if args.max_frames is not None and args.max_frames < 0:
        raise SystemExit("--max-frames 不能小于 0")
    if args.max_frames and args.end_frame and args.max_frames < args.end_frame:
        raise SystemExit("--max-frames 不能小于 --end-frame")
    if args.gpu.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        try:
            int(args.gpu)
        except ValueError as exc:
            raise SystemExit("--gpu 必须是整数编号或 cpu") from exc
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    prepared: PreparedInput | None = None
    if args.input:
        input_dir = Path(args.input).expanduser().resolve()
        output_dir = args.output.expanduser().resolve()
        if output_dir == input_dir or output_dir in input_dir.parents or input_dir in output_dir.parents:
            raise SystemExit("--output 与原始输入目录不能相同或互相包含")
        preprocess_dir = (
            args.preprocess_dir.expanduser().resolve()
            if args.preprocess_dir else output_dir / "preprocess"
        )
        max_frames = args.max_frames if args.max_frames is not None else (args.end_frame or 0)
        gpu_ids: list[int | str] = [""] if args.gpu.lower() == "cpu" else [int(args.gpu)]
        prepared = prepare_input(PreprocessOptions(
            input_dir=input_dir,
            work_dir=preprocess_dir,
            input_type=args.input_type,
            python=Path(sys.executable).resolve(),
            gpu_ids=gpu_ids,
            max_frames=max_frames,
            frames_per_second=args.frames_per_second,
            rig_json=args.rig_json,
            calib_json=args.calib_json,
            undistort_workers=args.undistort_workers,
            resume=args.resume,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            rig_pose_mode=args.rig_pose_mode,
            rig_radial_mode=args.rig_radial_mode,
            rig_small_terms=args.rig_small_terms,
            rig_direction=args.rig_direction,
            rig_scale=args.rig_scale,
            rig_iterations=args.rig_iterations,
        ))
        if (prepared.dry_run and prepared.input_type != "prepared") or args.preprocess_only:
            print(json.dumps({"preprocess": prepared.summary()}, ensure_ascii=False, indent=2))
            return 0

    config = load_config(args.config)
    config = apply_overrides(config, {
        "spawn.interval": args.spawn_interval,
        "camera_graph.top_k": args.camera_neighbors,
        "spawn.min_seed_views": args.min_seed_views,
        "triangulation.min_views": args.min_triangulation_views,
        "triangulation.max_reprojection_error_px": args.max_reprojection_error,
        "triangulation.min_angle_deg": args.min_triangulation_angle,
        "features.max_keypoints": args.max_keypoints,
        "features.resize_max": args.resize_max,
        "matching.matcher": args.spatial_matcher,
        "temporal.tracker": args.temporal_tracker,
    })
    options = PipelineOptions(
        dataset=str(prepared.dataset_dir) if prepared else None,
        images_dir=str(prepared.images_dir) if prepared and prepared.images_dir else args.images_dir,
        sparse_dir=str(prepared.sparse_dir) if prepared and prepared.sparse_dir else args.sparse_dir,
        persparse_dir=args.persparse_dir,
        output=args.output,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        cameras=_csv_ints(args.cameras) or None,
        stages=[value.strip() for value in args.stages.split(",") if value.strip()],
        resume=args.resume,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        debug_frames=_csv_ints(args.debug_frames),
        source_metadata=prepared.tracking_source() if prepared else {
            "input_type": "explicit_prepared",
            "images_dir": str(Path(args.images_dir).expanduser().resolve()),
            "sparse_dir": str(Path(args.sparse_dir).expanduser().resolve()),
        },
    )
    result = run_pipeline(options, config)
    payload = {"tracking": result}
    if prepared:
        payload["preprocess"] = prepared.summary()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        logging.getLogger("track_sparse").exception("处理失败: %s", exc)
        raise SystemExit(1)
