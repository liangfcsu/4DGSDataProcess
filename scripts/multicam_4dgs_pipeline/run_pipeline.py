#!/usr/bin/env python3
"""
将同步多相机视频一键转换为4DGS初始数据集。

成功后输出目录只包含 images、ims、persparse 和 sparse。
中间结果在成功后删除；失败时保留以便排查和 --resume。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
REQUIRED_WEIGHTS = (
    "superpoint_v1.pth",
    "superglue_indoor.pth",
    "superglue_outdoor.pth",
)


class PipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoEntry:
    camera_id: int
    path: Path


def parse_camera_id(stem: str) -> int | None:
    match = re.fullmatch(r"cam0*(\d+)", stem, re.IGNORECASE)
    if not match:
        match = re.fullmatch(r"0*(\d+)", stem)
    return int(match.group(1)) if match else None


def discover_videos(video_dir: Path) -> list[VideoEntry]:
    entries = []
    for path in sorted(video_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
            continue
        camera_id = parse_camera_id(path.stem)
        if camera_id is not None:
            entries.append(VideoEntry(camera_id, path.resolve()))
    entries.sort(key=lambda item: item.camera_id)
    ids = [item.camera_id for item in entries]
    duplicates = sorted({camera_id for camera_id in ids if ids.count(camera_id) > 1})
    if duplicates:
        raise PipelineError(f"视频文件映射到了重复相机编号: {duplicates}")
    if not entries:
        raise PipelineError(f"未找到可处理的视频: {video_dir}")
    return entries


def balanced_chunks(items: list, count: int) -> list[list]:
    count = max(1, min(count, len(items)))
    base, remainder = divmod(len(items), count)
    chunks = []
    start = 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        chunks.append(items[start:start + size])
        start += size
    return [chunk for chunk in chunks if chunk]


def detect_gpu_ids(spec: str) -> list[int]:
    if spec.strip().lower() != "auto":
        try:
            ids = [int(value.strip()) for value in spec.split(",") if value.strip()]
        except ValueError as exc:
            raise PipelineError(f"无效 --gpus: {spec}") from exc
        if not ids or len(ids) != len(set(ids)):
            raise PipelineError("--gpus 必须是不重复的GPU编号，例如 0,1,2,3")
        return ids

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        ids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        if ids:
            return ids
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        pass
    raise PipelineError("未检测到NVIDIA GPU；可用 --gpus 显式指定")


def python_has_dependencies(executable: Path) -> bool:
    probe = "import cv2,numpy,torch,h5py,pycolmap; assert torch.cuda.is_available()"
    try:
        return subprocess.run(
            [str(executable), "-c", probe],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        ).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def choose_python(explicit: str | None) -> Path:
    if explicit:
        executable = Path(explicit).expanduser().resolve()
        if not python_has_dependencies(executable):
            raise PipelineError(f"指定的Python缺少cv2/torch/h5py/pycolmap或CUDA不可用: {executable}")
        return executable

    candidates = [Path(sys.executable)]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "bin" / "python")
    candidates.extend([
        Path.home() / "miniconda3" / "envs" / "3dgslf" / "bin" / "python",
        Path.home() / "miniconda3" / "envs" / "4dgs" / "bin" / "python",
    ])
    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if python_has_dependencies(candidate):
            return candidate
    raise PipelineError("找不到具备cv2/torch/h5py/pycolmap和CUDA的Python；请用 --python 指定")


def run_checked(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"\n$ {shlex.join(command)}", flush=True)
    result = subprocess.run(command, env=env)
    if result.returncode != 0:
        raise PipelineError(f"命令失败（退出码 {result.returncode}）: {shlex.join(command)}")


def run_parallel(commands: list[tuple[list[str], dict[str, str] | None]]) -> None:
    processes = []
    for command, env in commands:
        print(f"\n$ {shlex.join(command)}", flush=True)
        processes.append((command, subprocess.Popen(command, env=env)))
    failures = []
    for command, process in processes:
        returncode = process.wait()
        if returncode != 0:
            failures.append((returncode, command))
    if failures:
        details = "; ".join(f"rc={code}: {shlex.join(command)}" for code, command in failures)
        raise PipelineError(f"并行任务失败: {details}")


def image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def hardlink_or_copy(source: str, destination: str) -> str:
    """最终组装优先使用硬链接，避免全量PNG在中间目录中再占一份空间。"""
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def discover_sequence(sequence_dir: Path, camera_ids: Iterable[int]) -> dict[int, list[Path]]:
    result = {}
    for camera_id in camera_ids:
        camera_dir = sequence_dir / f"cam{camera_id:03d}"
        if not camera_dir.is_dir():
            raise PipelineError(f"缺少相机目录: {camera_dir}")
        files = image_files(camera_dir)
        if not files:
            raise PipelineError(f"相机目录中没有图像: {camera_dir}")
        result[camera_id] = files
    return result


def validate_equal_frame_counts(sequence_dir: Path, camera_ids: list[int], expected: int | None) -> int:
    sequence = discover_sequence(sequence_dir, camera_ids)
    counts = {camera_id: len(files) for camera_id, files in sequence.items()}
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        raise PipelineError(f"各相机帧数不一致: {counts}")
    frame_count = unique_counts.pop()
    if expected is not None and frame_count != expected:
        raise PipelineError(f"帧数不符：期望每相机 {expected}，实际 {frame_count}")
    return frame_count


def validate_first_frames(directory: Path, camera_ids: list[int]) -> None:
    expected = {f"cam{camera_id:03d}frame001.png" for camera_id in camera_ids}
    actual = {path.name for path in image_files(directory)}
    if actual != expected:
        raise PipelineError(
            f"首帧集不完整：缺少={sorted(expected - actual)}, 多出={sorted(actual - expected)}"
        )


def validate_calibration(calibration_dir: Path, camera_ids: list[int]) -> None:
    calib_file = calibration_dir / "colmap_sfm" / "estimated_calib.json"
    sparse_dir = calibration_dir / "3dgs_training_data" / "sparse" / "0"
    images_dir = calibration_dir / "3dgs_training_data" / "images"
    required = [
        calib_file,
        sparse_dir / "cameras.txt",
        sparse_dir / "images.txt",
        sparse_dir / "points3D.txt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise PipelineError(f"标定/首帧重建输出不完整: {missing}")
    try:
        calibration = json.loads(calib_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"无法读取标定文件: {calib_file}") from exc
    calib_ids = sorted(int(camera["id"]) for camera in calibration)
    if calib_ids != sorted(camera_ids):
        raise PipelineError(f"标定相机不完整：期望 {camera_ids}，实际 {calib_ids}")
    image_names = {path.name for path in image_files(images_dir)}
    expected_names = {f"cam{camera_id:03d}frame001.png" for camera_id in camera_ids}
    if image_names != expected_names:
        raise PipelineError("标定阶段的去畸变首帧不完整")


def validate_persparse(directory: Path, frame_count: int) -> None:
    expected = {f"frame{frame_id:03d}_points3D.txt" for frame_id in range(1, frame_count + 1)}
    actual = {path.name for path in directory.glob("frame*_points3D.txt") if path.is_file()}
    if actual != expected:
        raise PipelineError(
            f"逐帧点云不完整：缺少={sorted(expected - actual)}, 多出={sorted(actual - expected)}"
        )
    empty = []
    for path in directory.glob("frame*_points3D.txt"):
        has_point = any(line.strip() and not line.startswith("#") for line in path.open())
        if not has_point:
            empty.append(path.name)
    if empty:
        raise PipelineError(f"以下帧的稀疏点云为空: {sorted(empty)}")


def save_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    temporary.replace(path)


def mark_stage(state_path: Path, state: dict, stage: str, **values) -> None:
    state.setdefault("completed_stages", []).append(stage)
    state.update(values)
    save_state(state_path, state)


def print_summary(output_dir: Path, camera_count: int, frame_count: int, gpu_ids: list[int]) -> None:
    print("\n" + "=" * 72)
    print("🎉 4DGS初始数据集已生成")
    print(f"输出: {output_dir}")
    print(f"相机: {camera_count}，每相机帧数: {frame_count}，GPU: {gpu_ids}")
    for name in ("images", "ims", "persparse", "sparse"):
        print(f"  - {output_dir / name}")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多相机视频到4DGS初始数据集的一键处理工具")
    parser.add_argument('--video-dir', required=True, help='多相机视频目录')
    parser.add_argument('--output-dir', required=True, help='最终4DGS数据集目录')
    parser.add_argument('--max-frames', type=int, default=30,
                        help='每路最多提取前N帧；0表示全部帧（默认30）')
    parser.add_argument('--frames-per-second', type=int, default=None,
                        help='可选：每秒均匀采样N帧；默认按原始帧顺序提取')
    parser.add_argument('--gpus', default='auto', help='auto或逗号分隔的GPU编号，例如0,1,2,3')
    parser.add_argument('--extract-workers', type=int, default=0,
                        help='提帧并行进程数；0表示与GPU数一致')
    parser.add_argument('--undistort-workers', type=int, default=min(32, os.cpu_count() or 8),
                        help='去畸变CPU线程数')
    parser.add_argument('--python', default=None, help='显式指定已安装依赖的Python解释器')
    parser.add_argument('--resume', action='store_true', help='从保留的中间目录续跑')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的输出/中间结果')
    parser.add_argument('--keep-intermediate', action='store_true', help='成功后仍保留中间目录（调试用）')
    parser.add_argument('--max-keypoints', type=int, default=4096)
    parser.add_argument('--resize-max', type=int, default=4000)
    parser.add_argument('--superglue-weights', choices=['indoor', 'outdoor'], default='indoor')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_dir = Path(args.video_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    work_dir = output_dir.parent / f".{output_dir.name}.work"
    state_path = work_dir / "state.json"
    backup_dir = work_dir / "previous_output"

    try:
        if args.max_frames < 0:
            raise PipelineError("--max-frames 不能小于0")
        if args.frames_per_second is not None and args.frames_per_second <= 0:
            raise PipelineError("--frames-per-second 必须大于0")
        if args.extract_workers < 0 or args.undistort_workers <= 0:
            raise PipelineError("并行进程/线程数参数无效")
        if not video_dir.is_dir():
            raise PipelineError(f"视频目录不存在: {video_dir}")
        if output_dir == video_dir or output_dir in video_dir.parents:
            raise PipelineError("输出目录不能是视频目录或其父目录")

        videos = discover_videos(video_dir)
        camera_ids = [entry.camera_id for entry in videos]
        gpu_ids = detect_gpu_ids(args.gpus)
        python = choose_python(args.python)

        if shutil.which("colmap") is None:
            raise PipelineError("未找到colmap可执行文件")
        weights_dir = (SCRIPT_DIR / "Hierarchical-Localization" / "third_party" /
                       "SuperGluePretrainedNetwork" / "models" / "weights")
        missing_weights = [name for name in REQUIRED_WEIGHTS
                           if not (weights_dir / name).is_file() or (weights_dir / name).stat().st_size < 1_000_000]
        if missing_weights:
            raise PipelineError(f"缺少或损坏的SuperPoint/SuperGlue权重: {missing_weights}")

        config = {
            "video_dir": str(video_dir),
            "output_dir": str(output_dir),
            "camera_ids": camera_ids,
            "max_frames": args.max_frames,
            "frames_per_second": args.frames_per_second,
            "gpus": gpu_ids,
            "max_keypoints": args.max_keypoints,
            "resize_max": args.resize_max,
            "superglue_weights": args.superglue_weights,
        }

        if output_dir.exists():
            if args.resume:
                raise PipelineError(f"--resume 时最终输出不应已存在: {output_dir}")
            if not args.overwrite:
                raise PipelineError(f"输出目录已存在：{output_dir}；请更换目录或使用 --overwrite")

        if work_dir.exists() and not args.resume:
            if not args.overwrite:
                raise PipelineError(f"中间目录已存在：{work_dir}；请用 --resume 或 --overwrite")
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        if output_dir.exists():
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.move(str(output_dir), str(backup_dir))

        if args.resume:
            if not state_path.is_file():
                raise PipelineError(f"无法续跑，缺少状态文件: {state_path}")
            state = json.loads(state_path.read_text())
            if state.get("config") != config:
                raise PipelineError("续跑参数与原任务不一致；请恢复原参数或使用 --overwrite")
        else:
            state = {"config": config, "completed_stages": []}
            save_state(state_path, state)

        completed = set(state.get("completed_stages", []))
        raw_ims = work_dir / "raw_ims"
        first_frames = work_dir / "first_frames"
        calibration_dir = work_dir / "calibration"
        undistorted_all = work_dir / "undistorted_all"
        persparse_dir = work_dir / "persparse"
        shard_root = work_dir / "persparse_shards"

        print("=" * 72)
        print("多相机视频 → 4DGS初始数据集")
        print(f"视频: {video_dir}")
        print(f"相机: {camera_ids}（共{len(camera_ids)}路）")
        print(f"GPU: {gpu_ids}")
        print(f"Python: {python}")
        print(f"模式: {'全量' if args.max_frames == 0 else f'前{args.max_frames}帧验证'}")
        print("=" * 72)

        # 1. 提取所有帧（按相机分片并行）
        if "extract_frames" not in completed:
            shutil.rmtree(raw_ims, ignore_errors=True)
            worker_count = args.extract_workers or len(gpu_ids)
            chunks = balanced_chunks(videos, worker_count)
            commands = []
            for index, chunk in enumerate(chunks):
                command = [
                    str(python), str(SCRIPT_DIR / "extract_frames_from_videos.py"),
                    "--video-dir", str(video_dir),
                    "--output-dir", str(raw_ims),
                    "--start-cam", str(chunk[0].camera_id),
                    "--end-cam", str(chunk[-1].camera_id),
                ]
                if args.frames_per_second is None:
                    command.append("--all-frames")
                else:
                    command.extend(["--frames-per-second", str(args.frames_per_second)])
                if args.max_frames:
                    command.extend(["--max-frames", str(args.max_frames)])
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[index % len(gpu_ids)])
                commands.append((command, env))
            run_parallel(commands)
            expected = args.max_frames if args.max_frames and args.frames_per_second is None else None
            frame_count = validate_equal_frame_counts(raw_ims, camera_ids, expected)
            mark_stage(state_path, state, "extract_frames", frame_count=frame_count)
            completed.add("extract_frames")
        else:
            frame_count = validate_equal_frame_counts(raw_ims, camera_ids, state.get("frame_count"))
            print(f"⏭️  跳过已完成阶段：提帧（{frame_count}帧/相机）")

        # 2. 从视频独立提取首帧，命名与序列第1帧一致
        if "first_frames" not in completed:
            shutil.rmtree(first_frames, ignore_errors=True)
            run_checked([
                str(python), str(SCRIPT_DIR / "extract_first_frames.py"),
                "--video-dir", str(video_dir),
                "--output-dir", str(first_frames),
                "--frame-style-names",
            ])
            validate_first_frames(first_frames, camera_ids)
            mark_stage(state_path, state, "first_frames")
            completed.add("first_frames")
        else:
            validate_first_frames(first_frames, camera_ids)
            print("⏭️  跳过已完成阶段：首帧提取")

        # 3. 首帧标定 + HLoc/SuperGlue首帧重建
        if "calibration" not in completed:
            shutil.rmtree(calibration_dir, ignore_errors=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
            run_checked([
                str(python), str(SCRIPT_DIR / "complete_3dgs_pipeline.py"),
                "--stage", "all",
                "--feature-method", "superpoint",
                "--matcher-method", "superglue",
                "--undistort-method", "custom",
                "--non-interactive",
                "--images-dir", str(first_frames),
                "--output-dir", str(calibration_dir),
            ], env=env)
            validate_calibration(calibration_dir, camera_ids)
            mark_stage(state_path, state, "calibration")
            completed.add("calibration")
        else:
            validate_calibration(calibration_dir, camera_ids)
            print("⏭️  跳过已完成阶段：内外参估计")

        # 4. 所有相机和帧去畸变
        if "undistort" not in completed:
            shutil.rmtree(undistorted_all, ignore_errors=True)
            run_checked([
                str(python), str(SCRIPT_DIR / "undistort_all_frames.py"),
                "--input-dir", str(raw_ims),
                "--calib-file", str(calibration_dir / "colmap_sfm" / "estimated_calib.json"),
                "--output-dir", str(undistorted_all),
                "--workers", str(args.undistort_workers),
            ])
            validate_equal_frame_counts(undistorted_all, camera_ids, frame_count)
            mark_stage(state_path, state, "undistort")
            completed.add("undistort")
        else:
            validate_equal_frame_counts(undistorted_all, camera_ids, frame_count)
            print("⏭️  跳过已完成阶段：全帧去畸变")

        # 5. 帧区间均分到所有GPU，每张卡一个独立进程
        if "per_frame_sparse" not in completed:
            shutil.rmtree(shard_root, ignore_errors=True)
            shutil.rmtree(persparse_dir, ignore_errors=True)
            shard_root.mkdir(parents=True)
            persparse_dir.mkdir(parents=True)
            frame_ids = list(range(1, frame_count + 1))
            chunks = balanced_chunks(frame_ids, len(gpu_ids))
            commands = []
            for index, chunk in enumerate(chunks):
                gpu_id = gpu_ids[index]
                shard_dir = shard_root / f"gpu{gpu_id}"
                command = [
                    str(python), str(SCRIPT_DIR / "generate_per_frame_sparse.py"),
                    "--images-dir", str(undistorted_all),
                    "--sparse-dir", str(calibration_dir / "3dgs_training_data" / "sparse" / "0"),
                    "--output-dir", str(shard_dir),
                    "--start-frame", str(chunk[0]),
                    "--end-frame", str(chunk[-1]),
                    "--gpu", str(gpu_id),
                    "--max-keypoints", str(args.max_keypoints),
                    "--resize-max", str(args.resize_max),
                    "--superglue-weights", args.superglue_weights,
                ]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                commands.append((command, env))
                print(f"🎯 GPU {gpu_id}: frame{chunk[0]:03d} ~ frame{chunk[-1]:03d}（{len(chunk)}帧）")
            run_parallel(commands)
            for shard_dir in sorted(shard_root.iterdir()):
                for source in sorted(shard_dir.glob("frame*_points3D.txt")):
                    destination = persparse_dir / source.name
                    if destination.exists():
                        raise PipelineError(f"逐帧点云合并时发现重复文件: {source.name}")
                    shutil.copy2(source, destination)
            validate_persparse(persparse_dir, frame_count)
            mark_stage(state_path, state, "per_frame_sparse")
            completed.add("per_frame_sparse")
        else:
            validate_persparse(persparse_dir, frame_count)
            print("⏭️  跳过已完成阶段：逐帧稀疏点云")

        # 6. 组装最终四个目录并做最后一次完整性检查
        final_staging = work_dir / "final"
        shutil.rmtree(final_staging, ignore_errors=True)
        final_staging.mkdir()
        shutil.copytree(calibration_dir / "3dgs_training_data" / "images", final_staging / "images",
                        copy_function=hardlink_or_copy)
        shutil.copytree(undistorted_all, final_staging / "ims", copy_function=hardlink_or_copy)
        shutil.copytree(persparse_dir, final_staging / "persparse", copy_function=hardlink_or_copy)
        shutil.copytree(calibration_dir / "3dgs_training_data" / "sparse", final_staging / "sparse",
                        copy_function=hardlink_or_copy)
        if {path.name for path in final_staging.iterdir()} != {"images", "ims", "persparse", "sparse"}:
            raise PipelineError("最终目录组装异常")
        validate_first_frames(final_staging / "images", camera_ids)
        validate_equal_frame_counts(final_staging / "ims", camera_ids, frame_count)
        validate_persparse(final_staging / "persparse", frame_count)

        if output_dir.exists():
            raise PipelineError(f"组装前发现输出目录被重新创建: {output_dir}")
        shutil.move(str(final_staging), str(output_dir))
        if args.keep_intermediate:
            print(f"🛠️  已保留中间目录: {work_dir}")
        else:
            shutil.rmtree(work_dir)
        print_summary(output_dir, len(camera_ids), frame_count, gpu_ids)
        return 0

    except (PipelineError, OSError, json.JSONDecodeError) as exc:
        print(f"\n❌ 流水线失败: {exc}", file=sys.stderr)
        if backup_dir.exists() and not output_dir.exists():
            try:
                shutil.move(str(backup_dir), str(output_dir))
                print(f"↩️  已恢复原输出目录: {output_dir}", file=sys.stderr)
            except OSError as restore_exc:
                print(f"⚠️  恢复原输出失败: {restore_exc}", file=sys.stderr)
        if work_dir.exists():
            print(f"🧩 中间结果已保留: {work_dir}", file=sys.stderr)
            print("修复问题后可使用相同参数加 --resume 续跑。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
