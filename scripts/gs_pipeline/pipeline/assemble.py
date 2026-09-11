"""组装最终 4DGS 初始数据集。

output/
├── images/       # 每台相机第 1 帧（去畸变，扁平）
├── ims/          # 全部去畸变帧 camXXX/camXXXframeYYY.png
├── persparse/    # frame{n:03d}_points3D.txt
└── sparse/0/     # 参考模型 cameras.txt / images.txt / points3D.txt
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .common import PipelineError, cam_dirname, discover_cam_sequences, parse_cam_id


def _hardlink_or_copy(src: str, dst: str) -> str:
    try:
        os.link(src, dst)
        return dst
    except OSError:
        return shutil.copy2(src, dst)


def _count_points(points_txt: Path) -> int:
    if not points_txt.is_file():
        return 0
    return sum(1 for line in points_txt.read_text().splitlines()
               if line.strip() and not line.startswith("#"))


def _reference_cam_ids(reference_sparse: Path) -> set[int]:
    """从参考模型 images.txt 的图像名解析出注册到的相机编号集合。"""
    images_txt = reference_sparse / "images.txt"
    if not images_txt.is_file():
        return set()
    cam_ids: set[int] = set()
    expect_image_line = True
    for line in images_txt.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if not stripped:
            if not expect_image_line:
                expect_image_line = True
            continue
        if expect_image_line:
            parts = stripped.split()
            if len(parts) >= 10:
                cam_id = parse_cam_id(parts[9])
                if cam_id is not None:
                    cam_ids.add(cam_id)
            expect_image_line = False
        else:
            expect_image_line = True
    return cam_ids


def assemble(output_dir: Path, reference_sparse: Path, undistorted_dir: Path,
             persparse_dir: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f"  [dry-run] 将组装 {output_dir}/{{images,ims,persparse,sparse}}")
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # 只保留参考模型里注册到的相机（无标定 SfM 可能只注册子集），
    # 使 ims/ images/ sparse/0/ persparse 的相机集一致。
    sequences = discover_cam_sequences(undistorted_dir)
    if not sequences:
        raise PipelineError(f"去畸变目录中没有相机序列: {undistorted_dir}")
    ref_cam_ids = _reference_cam_ids(reference_sparse)
    if ref_cam_ids:
        dropped = sorted(set(sequences) - ref_cam_ids)
        if dropped:
            print(f"  ⚠️ {len(dropped)} 台相机未注册进参考模型，已从输出剔除: "
                  f"{', '.join('cam%03d' % c for c in dropped[:8])}{' …' if len(dropped) > 8 else ''}")
        sequences = {c: f for c, f in sequences.items() if c in ref_cam_ids}
        if not sequences:
            raise PipelineError("参考模型的相机与去畸变目录无交集，无法组装")

    # ims：参考模型相机的全部去畸变帧
    ims_dir = staging / "ims"
    for cam_id, frames in sequences.items():
        cam_out = ims_dir / cam_dirname(cam_id)
        cam_out.mkdir(parents=True)
        for frame in frames:
            _hardlink_or_copy(str(frame), str(cam_out / frame.name))

    # images：每台相机第 1 帧（扁平）
    images_dir = staging / "images"
    images_dir.mkdir()
    for cam_id, frames in sequences.items():
        first = frames[0]
        _hardlink_or_copy(str(first), str(images_dir / first.name))

    # persparse：逐帧点云
    shutil.copytree(persparse_dir, staging / "persparse", copy_function=_hardlink_or_copy)

    # sparse/0：相机与位姿来自参考模型；points3D.txt 必须是非空的静态初始化点云，
    # 否则标准 3DGS 训练器（读 sparse/0/points3D.txt）会「初始化 0 点」而崩溃。
    #   - 无标定：参考模型自带 hloc 重建点云，直接用；
    #   - 标定：参考模型点为空，用第 1 帧的逐帧三角化点云（persparse/frame001）作初始化。
    # 二者都与 sparse/0 的位姿同坐标系，且 4DGS 训练另读 persparse/ 不受影响。
    sparse0 = staging / "sparse" / "0"
    sparse0.mkdir(parents=True)
    for name in ("cameras.txt", "images.txt"):
        src = reference_sparse / name
        if not src.is_file():
            raise PipelineError(f"参考模型缺少 {name}: {src}")
        _hardlink_or_copy(str(src), str(sparse0 / name))

    ref_points = reference_sparse / "points3D.txt"
    frame1_points = persparse_dir / "frame001_points3D.txt"
    if _count_points(ref_points) > 0:
        init_points = ref_points
    elif frame1_points.is_file():
        init_points = frame1_points
    else:
        raise PipelineError(f"找不到可用于 sparse/0 初始化的 points3D（{ref_points} 为空且缺少 {frame1_points}）")
    _hardlink_or_copy(str(init_points), str(sparse0 / "points3D.txt"))
    if _count_points(sparse0 / "points3D.txt") == 0:
        raise PipelineError("sparse/0/points3D.txt 初始化点云为空，训练器将无法初始化")

    produced = {p.name for p in staging.iterdir()}
    if produced != {"images", "ims", "persparse", "sparse"}:
        raise PipelineError(f"最终目录组装异常: {sorted(produced)}")
    shutil.move(str(staging), str(output_dir))
