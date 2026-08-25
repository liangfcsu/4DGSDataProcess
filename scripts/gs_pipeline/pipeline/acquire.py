"""输入采集：把三类输入统一成规范布局 ims/camXXX/camXXXframe{n:03d}.png。

- 多相机视频：调用 engine/extract_frames_from_videos.py 按相机分片并行提帧。
- 多相机多帧图像（camXXX/ 子目录，每台一段序列）：按序重命名为 camXXXframe{n:03d}.png。
- 多相机单帧图像（扁平目录，每张一台相机，如 data/7.7/Photo）：按排序索引铺成 camXXX/camXXXframe001.png。

规范化后，标定/去畸变/逐帧点云三个阶段对三类输入完全一致。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .common import (
    IMAGE_SUFFIXES,
    PipelineError,
    balanced_chunks,
    cam_dirname,
    discover_videos,
    frame_filename,
    list_images,
    parse_cam_id,
    run_parallel,
)

ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
EXTRACT_FRAMES = ENGINE_DIR / "extract_frames_from_videos.py"


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# 视频输入
# ---------------------------------------------------------------------------

def acquire_single_frame_from_videos(video_dir: Path, out_ims: Path, frame_no: int,
                                     *, dry_run: bool) -> None:
    """从每路视频抽取第 frame_no 帧（1 起），铺成单帧 camXXX/camXXXframe001.png。

    用直接的 cv2 定位读取（不经提帧脚本），保证取到的正是指定帧。
    """
    videos = discover_videos(video_dir)
    if not videos:
        raise PipelineError(f"视频目录中没有可识别的相机视频: {video_dir}")
    if frame_no < 1:
        raise PipelineError("--frame 必须是不小于 1 的帧号")
    print(f"  抽取每路视频第 {frame_no} 帧 → 单帧数据集（{len(videos)} 台相机）")
    if dry_run:
        for cam_id, path in videos[:3]:
            print(f"    cam{cam_id:03d}: {path.name} 取第 {frame_no} 帧 → {cam_dirname(cam_id)}/{frame_filename(cam_id, 1)}")
        if len(videos) > 3:
            print(f"    … 其余 {len(videos) - 3} 台相机同理")
        return

    import cv2
    for cam_id, path in videos:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise PipelineError(f"无法打开视频: {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0 and frame_no > total:
            cap.release()
            raise PipelineError(f"{path.name} 只有 {total} 帧，无法取第 {frame_no} 帧")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise PipelineError(f"读取 {path.name} 第 {frame_no} 帧失败")
        dst = out_ims / cam_dirname(cam_id) / frame_filename(cam_id, 1)
        dst.parent.mkdir(parents=True, exist_ok=True)
        encoded, buffer = cv2.imencode(".png", frame)
        if not encoded:
            raise PipelineError(f"编码 {path.name} 第 {frame_no} 帧失败")
        buffer.tofile(str(dst))


def acquire_from_videos(video_dir: Path, out_ims: Path, python: Path, gpu_ids: list[int],
                        *, max_frames: int, frames_per_second: int | None,
                        resume: bool, dry_run: bool) -> None:
    if not EXTRACT_FRAMES.is_file():
        raise PipelineError(f"缺少提帧脚本: {EXTRACT_FRAMES}")
    videos = discover_videos(video_dir)
    if not videos:
        raise PipelineError(f"视频目录中没有可识别的相机视频: {video_dir}")
    chunks = balanced_chunks(videos, len(gpu_ids))
    commands = []
    for index, chunk in enumerate(chunks):
        command = [
            str(python), str(EXTRACT_FRAMES),
            "--video-dir", str(video_dir),
            "--output-dir", str(out_ims),
            "--start-cam", str(chunk[0][0]),
            "--end-cam", str(chunk[-1][0]),
        ]
        if frames_per_second is None:
            command.append("--all-frames")
        else:
            command.extend(["--frames-per-second", str(frames_per_second)])
        if max_frames:
            command.extend(["--max-frames", str(max_frames)])
        if resume:
            command.append("--resume")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[index % len(gpu_ids)])
        commands.append((command, env))
    run_parallel(commands, dry_run=dry_run)


# ---------------------------------------------------------------------------
# 图像输入
# ---------------------------------------------------------------------------

def _has_cam_subdirs(input_dir: Path) -> bool:
    return any(
        child.is_dir() and parse_cam_id(child.name) is not None
        for child in input_dir.iterdir()
    )


def acquire_from_images(input_dir: Path, out_ims: Path, *, select_frame: int | None = None,
                        dry_run: bool) -> None:
    """规范化图像输入到 out_ims/camXXX/camXXXframe{n:03d}.png。

    select_frame 非空时只取每台相机序列里的第 N 张（1 起），产出单帧数据集。
    """
    if not input_dir.is_dir():
        raise PipelineError(f"图像输入目录不存在: {input_dir}")

    if _has_cam_subdirs(input_dir):
        _normalize_cam_subdirs(input_dir, out_ims, select_frame=select_frame, dry_run=dry_run)
    else:
        if select_frame not in (None, 1):
            raise PipelineError(f"扁平单帧输入只有 1 帧，无法取第 {select_frame} 帧")
        _normalize_flat_single_frame(input_dir, out_ims, dry_run=dry_run)


def _normalize_cam_subdirs(input_dir: Path, out_ims: Path, *, select_frame: int | None = None,
                           dry_run: bool) -> None:
    """camXXX/ 每台一段序列：按文件名排序重命名为 camXXXframe{n:03d}.png。

    select_frame 非空时每台相机只取第 N 张，输出为单帧 camXXXframe001.png。
    """
    if select_frame is not None and select_frame < 1:
        raise PipelineError("--frame 必须是不小于 1 的帧号")
    cam_dirs = sorted(
        (child for child in input_dir.iterdir()
         if child.is_dir() and parse_cam_id(child.name) is not None),
        key=lambda d: parse_cam_id(d.name),
    )
    frame_counts = set()
    for cam_dir in cam_dirs:
        cam_id = parse_cam_id(cam_dir.name)
        frames = list_images(cam_dir)
        if not frames:
            raise PipelineError(f"相机目录为空: {cam_dir}")
        if select_frame is not None:
            if select_frame > len(frames):
                raise PipelineError(f"{cam_dir.name} 只有 {len(frames)} 帧，无法取第 {select_frame} 帧")
            frames = [frames[select_frame - 1]]   # 只保留选中的一帧，输出为 frame001
        frame_counts.add(len(frames))
        suffix = f"（取第 {select_frame} 帧）" if select_frame is not None else ""
        print(f"  {cam_dir.name}: {len(frames)} 帧 → {cam_dirname(cam_id)}/{suffix}")
        if not dry_run:
            for index, frame in enumerate(frames, start=1):
                _link_or_copy(frame, out_ims / cam_dirname(cam_id) / frame_filename(cam_id, index))
    if len(frame_counts) > 1:
        raise PipelineError(f"各相机帧数不一致: {sorted(frame_counts)}")


def _normalize_flat_single_frame(input_dir: Path, out_ims: Path, *, dry_run: bool) -> None:
    """扁平目录：每张图一台相机，按排序索引铺成 camXXX/camXXXframe001.png。"""
    images = list_images(input_dir, IMAGE_SUFFIXES | {".bmp", ".tif", ".tiff", ".webp"})
    if not images:
        raise PipelineError(f"扁平图像目录里没有可用图像: {input_dir}")
    print(f"  扁平单帧输入: {len(images)} 台相机（按文件名排序编号 cam000..cam{len(images) - 1:03d}）")
    if dry_run:
        return
    for cam_id, image in enumerate(images):
        _link_or_copy(image, out_ims / cam_dirname(cam_id) / frame_filename(cam_id, 1))
