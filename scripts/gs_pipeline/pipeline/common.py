"""统一流水线的共享基础设施：路径、命名、发现、校验、子进程与环境探测。

命名规范（全流程唯一标准，务必保持一致）：
- 相机 3 位补零：cam{id:03d}
- 多帧图像：cam{id:03d}/cam{id:03d}frame{n:03d}.png（n 为 1 起的连续保存序号）
- 逐帧点云：frame{n:03d}_points3D.txt
- 参考稀疏模型：sparse/0/{cameras.txt,images.txt,points3D.txt}
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Iterable

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
READABLE_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


class PipelineError(RuntimeError):
    """流水线内部可预期的失败，main 捕获后打印并返回非零。"""


# ---------------------------------------------------------------------------
# 命名与解析
# ---------------------------------------------------------------------------

def cam_dirname(cam_id: int) -> str:
    return f"cam{cam_id:03d}"


def frame_filename(cam_id: int, frame_id: int) -> str:
    return f"cam{cam_id:03d}frame{frame_id:03d}.png"


def reference_image_name(cam_id: int) -> str:
    """参考稀疏模型里每台相机对应的图像名。

    用 camXXXframe001.png（而非 camXXX.png）：既能被 cam(\\d+) 解析供逐帧引擎按
    相机对齐，又与 images/ 首帧文件名、4DGS 训练器读取 sparse/0/images.txt 的约定一致。
    """
    return f"cam{cam_id:03d}frame001.png"


def parse_cam_id(name: str) -> int | None:
    """从文件/目录名解析相机编号：cam001frame020.png→1，cam007→7，001.png→1。"""
    match = re.search(r"cam0*(\d+)", name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    stem = Path(name).stem
    if stem.isdigit():
        return int(stem)
    return None


def parse_frame_id(name: str) -> int | None:
    match = re.search(r"frame0*(\d+)", name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_video_cam_id(stem: str) -> int | None:
    """视频文件名解析相机号：cam00→0，cam001→1，001→1，20→20。"""
    match = re.fullmatch(r"cam0*(\d+)", stem, re.IGNORECASE)
    if not match:
        match = re.fullmatch(r"0*(\d+)", stem)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# 目录发现
# ---------------------------------------------------------------------------

def list_images(directory: Path, suffixes: Iterable[str] = IMAGE_SUFFIXES) -> list[Path]:
    if not directory.is_dir():
        return []
    allowed = set(suffixes)
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in allowed
    )


def discover_videos(video_dir: Path) -> list[tuple[int, Path]]:
    """返回按相机号排序的 [(cam_id, video_path)]；相机号重复即报错。"""
    entries: list[tuple[int, Path]] = []
    for path in sorted(video_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        cam_id = parse_video_cam_id(path.stem)
        if cam_id is not None:
            entries.append((cam_id, path.resolve()))
    entries.sort(key=lambda item: item[0])
    ids = [cam_id for cam_id, _ in entries]
    duplicates = sorted({cam_id for cam_id in ids if ids.count(cam_id) > 1})
    if duplicates:
        raise PipelineError(f"视频文件映射到重复相机编号: {duplicates}")
    return entries


def discover_cam_sequences(ims_dir: Path) -> dict[int, list[Path]]:
    """从 camXXX/ 布局收集 {cam_id: [帧路径(按名排序)]}。"""
    result: dict[int, list[Path]] = {}
    if not ims_dir.is_dir():
        return result
    for cam_dir in sorted(ims_dir.iterdir()):
        if not cam_dir.is_dir() or not cam_dir.name.lower().startswith("cam"):
            continue
        cam_id = parse_cam_id(cam_dir.name)
        if cam_id is None:
            continue
        frames = list_images(cam_dir)
        if frames:
            result[cam_id] = frames
    return result


def frame_count_of(sequences: dict[int, list[Path]]) -> int:
    """校验各相机帧数一致，返回统一帧数。"""
    if not sequences:
        raise PipelineError("未发现任何 camXXX 相机序列")
    counts = {cam_id: len(frames) for cam_id, frames in sequences.items()}
    unique = set(counts.values())
    if len(unique) != 1:
        raise PipelineError(f"各相机帧数不一致: {counts}")
    return unique.pop()


# ---------------------------------------------------------------------------
# 子进程
# ---------------------------------------------------------------------------

def _terminate(processes: list[subprocess.Popen]) -> None:
    running = [p for p in processes if p.poll() is None]
    for p in running:
        try:
            os.killpg(p.pid, signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            pass
    for p in running:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            p.wait()


def run_checked(command: list[str], *, env: dict | None = None, cwd: str | None = None,
                dry_run: bool = False) -> None:
    printable = shlex.join(str(c) for c in command)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return
    process = subprocess.Popen([str(c) for c in command], env=env, cwd=cwd,
                               start_new_session=True)
    try:
        code = process.wait()
    except KeyboardInterrupt:
        _terminate([process])
        raise
    if code != 0:
        raise PipelineError(f"命令失败(退出码 {code}): {printable}")


def run_parallel(commands: list[tuple[list[str], dict | None]], *, dry_run: bool = False) -> None:
    """并行执行若干命令，任一失败即抛错。"""
    if dry_run:
        for command, _ in commands:
            print(f"\n$ {shlex.join(str(c) for c in command)}", flush=True)
        return
    processes: list[tuple[list[str], subprocess.Popen]] = []
    for command, env in commands:
        print(f"\n$ {shlex.join(str(c) for c in command)}", flush=True)
        processes.append((command, subprocess.Popen([str(c) for c in command], env=env,
                                                     start_new_session=True)))
    failures = []
    try:
        for command, process in processes:
            if process.wait() != 0:
                failures.append(command)
    except KeyboardInterrupt:
        _terminate([p for _, p in processes])
        raise
    if failures:
        details = "; ".join(shlex.join(str(c) for c in cmd) for cmd in failures)
        raise PipelineError(f"并行任务失败: {details}")


# ---------------------------------------------------------------------------
# 环境探测
# ---------------------------------------------------------------------------

def balanced_chunks(items: list, count: int) -> list[list]:
    count = max(1, min(count, len(items))) if items else 1
    base, remainder = divmod(len(items), count)
    chunks, start = [], 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        chunks.append(items[start:start + size])
        start += size
    return [chunk for chunk in chunks if chunk]


def detect_gpu_ids(spec: str) -> list[int]:
    if spec.strip().lower() != "auto":
        try:
            ids = [int(v.strip()) for v in spec.split(",") if v.strip()]
        except ValueError as exc:
            raise PipelineError(f"无效 --gpus: {spec}") from exc
        if not ids or len(ids) != len(set(ids)):
            raise PipelineError("--gpus 必须是不重复的 GPU 编号，例如 0,1,2,3")
        return ids
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        ids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        if ids:
            return ids
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        pass
    # 没有 GPU 时退化为单“逻辑设备”，让 hloc 走 CPU。
    return [0]


def python_has_dependencies(executable: Path) -> bool:
    probe = "import cv2,numpy,torch,h5py,pycolmap,tqdm,PIL,packaging"
    try:
        return subprocess.run(
            [str(executable), "-c", probe],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
        ).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def choose_python(explicit: str | None) -> Path:
    if explicit:
        executable = Path(explicit).expanduser().resolve()
        if not python_has_dependencies(executable):
            raise PipelineError(f"指定的 Python 缺少 cv2/torch/h5py/pycolmap 等依赖: {executable}")
        return executable
    candidates = [Path(sys.executable)]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "bin" / "python")
    candidates.extend([
        Path.home() / "miniconda3" / "envs" / "4dgsplayer" / "bin" / "python",
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
    raise PipelineError(
        "找不到具备 cv2/torch/h5py/pycolmap/tqdm/Pillow/packaging 的 Python；"
        "请激活相应 conda 环境或用 --python 指定解释器"
    )
