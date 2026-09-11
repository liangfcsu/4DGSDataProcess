"""逐帧稀疏点云：帧区间均分到各 GPU，复用 multicam 的 generate_per_frame_sparse.py。

输入去畸变的 camXXX/camXXXframeYYY.png + 参考 sparse/0，输出 frame{n:03d}_points3D.txt。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .common import (
    PipelineError,
    balanced_chunks,
    run_parallel,
)

ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
PER_FRAME_SPARSE = ENGINE_DIR / "generate_per_frame_sparse.py"


def generate_per_frame_sparse(undistorted_dir: Path, reference_sparse: Path, out_persparse: Path,
                              frame_count: int, python: Path, gpu_ids: list[int],
                              *, sg_opts: dict, dry_run: bool) -> None:
    if not PER_FRAME_SPARSE.is_file():
        raise PipelineError(f"缺少逐帧稀疏脚本: {PER_FRAME_SPARSE}")

    shard_root = out_persparse.parent / "persparse_shards"
    if not dry_run:
        if shard_root.exists():
            shutil.rmtree(shard_root)
        if out_persparse.exists():
            shutil.rmtree(out_persparse)
        shard_root.mkdir(parents=True)
        out_persparse.mkdir(parents=True)

    frame_ids = list(range(1, frame_count + 1))
    chunks = balanced_chunks(frame_ids, len(gpu_ids))
    commands = []
    for index, chunk in enumerate(chunks):
        gpu_id = gpu_ids[index]
        shard_dir = shard_root / f"gpu{gpu_id}"
        command = [
            str(python), str(PER_FRAME_SPARSE),
            "--images-dir", str(undistorted_dir),
            "--sparse-dir", str(reference_sparse),
            "--output-dir", str(shard_dir),
            "--start-frame", str(chunk[0]),
            "--end-frame", str(chunk[-1]),
            "--gpu", str(gpu_id),
            "--max-keypoints", str(sg_opts["max_keypoints"]),
            "--resize-max", str(sg_opts["resize_max"]),
            "--superglue-weights", sg_opts["superglue_weights"],
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        commands.append((command, env))
        print(f"  GPU {gpu_id}: frame{chunk[0]:03d} ~ frame{chunk[-1]:03d}（{len(chunk)} 帧）")
    run_parallel(commands, dry_run=dry_run)

    if dry_run:
        return

    for shard_dir in sorted(shard_root.iterdir()):
        for source in sorted(shard_dir.glob("frame*_points3D.txt")):
            destination = out_persparse / source.name
            if destination.exists():
                raise PipelineError(f"逐帧点云合并时发现重复文件: {source.name}")
            shutil.copy2(source, destination)

    expected = {f"frame{fid:03d}_points3D.txt" for fid in frame_ids}
    actual = {p.name for p in out_persparse.glob("frame*_points3D.txt")}
    if actual != expected:
        raise PipelineError(
            f"逐帧点云不完整：缺少={sorted(expected - actual)}, 多出={sorted(actual - expected)}"
        )

    empty_frames = []
    for name in sorted(expected):
        points_file = out_persparse / name
        has_points = any(
            line.strip() and not line.lstrip().startswith("#")
            for line in points_file.read_text(encoding="utf-8").splitlines()
        )
        if not has_points:
            empty_frames.append(name.removesuffix("_points3D.txt"))
    if empty_frames:
        preview = ", ".join(empty_frames[:12])
        suffix = " …" if len(empty_frames) > 12 else ""
        print(
            f"  ⚠️ {len(empty_frames)} 帧未三角化出 3D 点，保留空点云并继续: "
            f"{preview}{suffix}"
        )
