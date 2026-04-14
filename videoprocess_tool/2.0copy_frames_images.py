#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从每台相机文件夹中复制指定帧图像到目标目录

示例：
python data/2.0copy_first_images.py --data-dir data/huabantraindata/ims --output-dir data/huabantraindata/imas
python data/2.0copy_first_images.py --frame 30
python data/2.0copy_first_images.py --frames 30 60 90
python data/2.0copy_first_images.py --frame-range 30-60
python data/2.0copy_first_images.py --frame-range 30:90:10
"""

import shutil
from pathlib import Path
import argparse
import re
from typing import Dict, List, Optional
from tqdm import tqdm

# 定义路径（默认使用当前项目结构）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "huabantraindata" / "ims"
OUTPUT_DIR = PROJECT_ROOT / "data" / "huabantraindata" / "firstimages"

def parse_frame_range(frame_range: str) -> List[int]:
    """
    解析帧范围字符串，支持两种格式：
    1) start-end      例如 30-60
    2) start:end[:step] 例如 30:90:10
    """
    frame_range = frame_range.strip()

    if ":" in frame_range:
        parts = [p.strip() for p in frame_range.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError("--frame-range 格式错误，应为 start:end 或 start:end:step")
        start = int(parts[0])
        end = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
    elif "-" in frame_range:
        parts = [p.strip() for p in frame_range.split("-", 1)]
        if len(parts) != 2:
            raise ValueError("--frame-range 格式错误，应为 start-end")
        start = int(parts[0])
        end = int(parts[1])
        step = 1
    else:
        raise ValueError("--frame-range 必须包含 '-' 或 ':'")

    if start <= 0 or end <= 0:
        raise ValueError("帧编号必须从 1 开始")
    if end < start:
        raise ValueError("帧范围结束值必须大于等于起始值")
    if step <= 0:
        raise ValueError("步长必须为正整数")

    return list(range(start, end + 1, step))


def resolve_target_frames(frame: int, frames: Optional[List[int]], frame_range: Optional[str]) -> List[int]:
    """解析命令行参数，得到目标帧编号列表（1-based）。"""
    if frames is not None:
        selected = frames
    elif frame_range is not None:
        selected = parse_frame_range(frame_range)
    else:
        selected = [frame]

    deduped: List[int] = list(dict.fromkeys(selected))
    if any(f <= 0 for f in deduped):
        raise ValueError("帧编号必须为正整数（从 1 开始）")

    return deduped


def build_frame_map(cam_dir: Path, png_files: List[Path]) -> Dict[int, Path]:
    """
    从文件名中提取 frame 编号，构建 frame_num -> 文件路径映射。
    对不符合命名模式的文件自动忽略。
    """
    frame_map: Dict[int, Path] = {}
    pattern = re.compile(r"frame0*(\d+)\.png$", re.IGNORECASE)

    for file_path in png_files:
        match = pattern.search(file_path.name)
        if not match:
            continue

        frame_num = int(match.group(1))
        if frame_num not in frame_map:
            frame_map[frame_num] = file_path

    return frame_map


def copy_selected_images(
    data_dir: Path,
    output_dir: Path,
    target_frames: List[int],
    start_cam: Optional[int] = None,
    end_cam: Optional[int] = None,
):
    """
    从每台相机文件夹中复制指定帧图像到目标目录。
    帧编号从 1 开始。
    """
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    invalid_cam_count = 0
    empty_cam_count = 0
    missing_frame_count = 0
    copy_error_count = 0

    candidate_cam_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("cam")])
    filtered_cam_dirs = []

    for cam_dir in candidate_cam_dirs:
        match = re.match(r"cam0*(\d+)$", cam_dir.name)
        if not match:
            invalid_cam_count += 1
            continue

        cam_num = int(match.group(1))
        if start_cam is not None and cam_num < start_cam:
            continue
        if end_cam is not None and cam_num > end_cam:
            continue

        filtered_cam_dirs.append(cam_dir)

    for cam_dir in tqdm(filtered_cam_dirs, desc="Copying selected frames"):

        # 获取该相机文件夹中的所有png文件并排序
        png_files = sorted(cam_dir.glob("*.png"))

        if not png_files:
            empty_cam_count += 1
            continue

        frame_map = build_frame_map(cam_dir, png_files)
        use_index_fallback = len(frame_map) == 0

        for frame_num in target_frames:
            source_file: Optional[Path] = frame_map.get(frame_num)

            # 如果文件名不含 frame 编号，则回退到按排序位置取第 N 帧
            if source_file is None and use_index_fallback and frame_num <= len(png_files):
                source_file = png_files[frame_num - 1]

            if source_file is None:
                missing_frame_count += 1
                continue

            target_file = output_dir / source_file.name

            try:
                shutil.copy2(source_file, target_file)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {source_file}: {e}")
                copy_error_count += 1
    
    print(f"\n✓ 复制完成！")
    print(f"  成功复制: {copied_count} 张图像")
    print(f"  缺失帧: {missing_frame_count} 张图像（对应某些相机缺少该帧）")
    print(f"  空相机目录: {empty_cam_count} 个")
    print(f"  非法相机目录: {invalid_cam_count} 个")
    print(f"  复制失败: {copy_error_count} 张图像")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从每台相机目录复制指定帧图像到目标目录")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="源目录，默认: data/huabantraindata/ims")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="输出目录，默认: data/huabantraindata/firstimages")
    parser.add_argument("--start-cam", type=int, default=None, help="起始相机编号（可选）")
    parser.add_argument("--end-cam", type=int, default=None, help="结束相机编号（可选）")
    parser.add_argument("--frame", type=int, default=1, help="单个帧编号（从1开始），默认: 1")
    parser.add_argument("--frames", type=int, nargs="+", default=None, help="多个帧编号（空格分隔），例如: --frames 1 30 60")
    parser.add_argument("--frame-range", type=str, default=None, help="帧范围，支持 start-end 或 start:end[:step]，例如: 30-60 或 30:90:10")
    args = parser.parse_args()

    if args.frames is not None and args.frame_range is not None:
        parser.error("--frames 和 --frame-range 不能同时使用")

    try:
        target_frames = resolve_target_frames(args.frame, args.frames, args.frame_range)
    except ValueError as e:
        parser.error(str(e))

    print("开始从每台相机复制指定帧图像...")
    print(f"源目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"目标帧: {target_frames}")
    if args.start_cam is None and args.end_cam is None:
        print("相机范围: 自动检测全部 cam* 文件夹")
    else:
        print(f"相机范围: {args.start_cam if args.start_cam is not None else '-inf'} - {args.end_cam if args.end_cam is not None else 'inf'}")
    print()

    copy_selected_images(
        args.data_dir,
        args.output_dir,
        target_frames=target_frames,
        start_cam=args.start_cam,
        end_cam=args.end_cam,
    )
    
    # 显示复制后目录中的文件
    print(f"\n{args.output_dir} 中的文件:")
    files = sorted(args.output_dir.glob("*.png"))
    for f in files[:5]:
        print(f"  {f.name}")
    if len(files) > 5:
        print(f"  ... 共 {len(files)} 个文件")
