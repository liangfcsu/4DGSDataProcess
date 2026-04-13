#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从每台相机文件夹中复制第一张图像到 firstimages 目录
python videoprocess/2.0copy_first_images.py --data-dir publicdata/coffee_martini_files/coffee_martini_images --output-dir publicdata/coffee_martini_files/1
"""

import shutil
from pathlib import Path
import argparse
import re
from tqdm import tqdm

# 定义路径（默认使用当前项目结构）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "100camdata" / "ims200-400"
OUTPUT_DIR = PROJECT_ROOT / "100camersdata" / "firstimages2"

def copy_first_images(data_dir: Path, output_dir: Path, start_cam=None, end_cam=None):
    """
    从每台相机文件夹中复制第一张图像到 firstimages 目录
    """
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    skipped_count = 0
    
    cam_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("cam")])

    for cam_dir in tqdm(cam_dirs, desc="Copying first images"):
        match = re.match(r"cam0*(\d+)$", cam_dir.name)
        if not match:
            skipped_count += 1
            continue

        cam_num = int(match.group(1))

        if start_cam is not None and cam_num < start_cam:
            continue
        if end_cam is not None and cam_num > end_cam:
            continue
        
        # 获取该相机文件夹中的所有png文件并排序
        png_files = sorted(cam_dir.glob("*.png"))
        
        if not png_files:
            skipped_count += 1
            continue
        
        # 获取第一张图像（排序后的第一个）
        first_image = png_files[0]
        
        # 复制到 firstimages 目录，保持原文件名
        target_file = output_dir / first_image.name
        
        try:
            shutil.copy2(first_image, target_file)
            copied_count += 1
        except Exception as e:
            print(f"Error copying {first_image}: {e}")
            skipped_count += 1
    
    print(f"\n✓ 复制完成！")
    print(f"  成功复制: {copied_count} 张图像")
    print(f"  跳过: {skipped_count} 个相机")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从每台相机目录复制第一张图到firstimages")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="源目录，默认: data/100camdata/ims")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="输出目录，默认: 100camersdata/firstimages")
    parser.add_argument("--start-cam", type=int, default=None, help="起始相机编号（可选）")
    parser.add_argument("--end-cam", type=int, default=None, help="结束相机编号（可选）")
    args = parser.parse_args()

    print("开始从每台相机复制第一张图像...")
    print(f"源目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    if args.start_cam is None and args.end_cam is None:
        print("相机范围: 自动检测全部 cam* 文件夹")
    else:
        print(f"相机范围: {args.start_cam if args.start_cam is not None else '-inf'} - {args.end_cam if args.end_cam is not None else 'inf'}")
    print()

    copy_first_images(args.data_dir, args.output_dir, start_cam=args.start_cam, end_cam=args.end_cam)
    
    # 显示复制后目录中的文件
    print(f"\n{args.output_dir} 中的文件:")
    files = sorted(args.output_dir.glob("*.png"))
    for f in files[:5]:
        print(f"  {f.name}")
    if len(files) > 5:
        print(f"  ... 共 {len(files)} 个文件")
