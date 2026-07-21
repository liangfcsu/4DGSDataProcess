#!/usr/bin/env python3
"""
从100台相机的视频中提取第一帧，保存到同一个文件夹
输入：100camersdata/100台 8秒/*.mp4
输出：data/100camdata/first_frames/cam001.png, cam002.png, ...
"""

import cv2
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import re
import sys


def extract_first_frame(video_path, output_path, cam_num):
    """
    从视频中提取第一帧
    
    Args:
        video_path: 视频文件路径
        output_path: 输出文件路径
        cam_num: 相机编号
    
    Returns:
        bool: 是否成功
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return False
    
    # 读取第一帧
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ 无法读取第一帧: {video_path}")
        return False
    
    # 保存图片
    return bool(cv2.imwrite(str(output_path), frame))


def camera_id_from_video_name(path):
    """支持 cam00.mp4、cam001.mp4 和 001.mp4。"""
    match = re.match(r'^cam0*(\d+)$', path.stem, re.IGNORECASE)
    if not match:
        match = re.match(r'^0*(\d+)$', path.stem)
    return int(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser(description="从100台相机视频中提取第一帧")
    parser.add_argument('--video-dir', type=str, 
                       default='100camersdata/100台 8秒',
                       help='视频文件目录')
    parser.add_argument('--output-dir', type=str,
                       default='data/100camdata/first_frames',
                       help='输出目录（所有第一帧保存在同一个文件夹）')
    parser.add_argument('--start-cam', type=int, default=None,
                       help='起始相机编号（默认自动检测）')
    parser.add_argument('--end-cam', type=int, default=None,
                       help='结束相机编号（默认自动检测）')
    parser.add_argument('--frame-style-names', action='store_true',
                       help='输出camXXXframe001.png，与多帧序列和4DGS images命名保持一致')
    
    args = parser.parse_args()
    
    # 路径配置
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    
    if not video_dir.exists():
        print(f"❌ 视频目录不存在: {video_dir}")
        return 1

    video_entries = []
    for video_path in sorted(video_dir.iterdir()):
        if not video_path.is_file() or video_path.suffix.lower() not in {'.mp4', '.mov', '.avi', '.mkv'}:
            continue
        cam_num = camera_id_from_video_name(video_path)
        if cam_num is None:
            print(f"⚠️  跳过无法解析相机编号的视频: {video_path.name}")
            continue
        if args.start_cam is not None and cam_num < args.start_cam:
            continue
        if args.end_cam is not None and cam_num > args.end_cam:
            continue
        video_entries.append((cam_num, video_path))

    video_entries.sort(key=lambda item: item[0])

    if not video_entries:
        print(f"❌ 未找到可处理的视频: {video_dir}")
        return 1

    duplicate_ids = sorted({cam_id for cam_id, _ in video_entries
                            if sum(item_id == cam_id for item_id, _ in video_entries) > 1})
    if duplicate_ids:
        print(f"❌ 检测到重复相机编号: {duplicate_ids}")
        return 1
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("📸 提取视频第一帧")
    print("=" * 60)
    print(f"视频目录: {video_dir}")
    print(f"输出目录: {output_dir}")
    print(f"相机范围: {video_entries[0][0]:03d} - {video_entries[-1][0]:03d}")
    print("=" * 60)
    print()
    
    # 统计信息
    success_count = 0
    failed_videos = []
    
    # 遍历自动发现的相机视频
    for cam_num, video_path in tqdm(video_entries,
                        desc="提取第一帧", 
                        unit="video"):
        output_filename = (f"cam{cam_num:03d}frame001.png"
                           if args.frame_style_names else f"cam{cam_num:03d}.png")
        output_path = output_dir / output_filename
        
        # 提取第一帧
        try:
            if extract_first_frame(video_path, output_path, cam_num):
                success_count += 1
            else:
                failed_videos.append(cam_num)
        except Exception as e:
            print(f"\n❌ 处理视频出错: {video_path.name}")
            print(f"   错误信息: {e}")
            failed_videos.append(cam_num)
    
    # 总结
    print()
    print("=" * 60)
    print("✅ 提取完成！")
    print("=" * 60)
    print(f"成功提取: {success_count} 个视频的第一帧")
    print(f"输出目录: {output_dir.absolute()}")
    
    if failed_videos:
        print(f"\n⚠️  失败/跳过的相机编号: {failed_videos}")
    
    print("\n📁 输出文件:")
    example_suffix = "frame001.png" if args.frame_style_names else ".png"
    first_id = video_entries[0][0]
    second_id = video_entries[min(1, len(video_entries) - 1)][0]
    print(f"   {output_dir}/cam{first_id:03d}{example_suffix}")
    print(f"   {output_dir}/cam{second_id:03d}{example_suffix}")
    print(f"   ...")
    print(f"   {output_dir}/cam{video_entries[-1][0]:03d}{example_suffix}")
    
    # 显示一些提取的文件
    extracted_files = sorted(output_dir.glob("cam*.png"))
    if extracted_files:
        print(f"\n✅ 实际生成了 {len(extracted_files)} 个文件")
        print("前5个文件:")
        for f in extracted_files[:5]:
            file_size = f.stat().st_size / 1024  # KB
            print(f"   {f.name} ({file_size:.1f} KB)")

    return 0 if success_count == len(video_entries) and not failed_videos else 1


if __name__ == '__main__':
    sys.exit(main())
