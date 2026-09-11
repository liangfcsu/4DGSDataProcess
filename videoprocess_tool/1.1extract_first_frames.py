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
import re
import sys
from tqdm import tqdm


VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv'}


def parse_video_cam_id(stem):
    """兼容 cam_1、cam001、cam-001 和纯数字视频名。"""
    match = re.fullmatch(r'cam[_-]?0*(\d+)', stem, re.IGNORECASE)
    if not match:
        match = re.fullmatch(r'0*(\d+)', stem)
    return int(match.group(1)) if match else None


def discover_videos(video_dir):
    """返回 {camera_id: video_path}，并拒绝重复相机编号。"""
    videos = {}
    for path in sorted(video_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        cam_id = parse_video_cam_id(path.stem)
        if cam_id is None:
            print(f"⚠️  跳过无法识别相机编号的视频: {path.name}")
            continue
        if cam_id in videos:
            raise ValueError(
                f"相机 {cam_id} 存在重复视频: {videos[cam_id].name}, {path.name}"
            )
        videos[cam_id] = path
    return videos


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
    
    # imencode + tofile 同时兼容中文输出路径，并检查实际写入结果。
    encoded, buffer = cv2.imencode('.png', frame)
    if not encoded:
        print(f"❌ 第一帧 PNG 编码失败: {video_path}")
        return False
    try:
        buffer.tofile(str(output_path))
    except OSError as exc:
        print(f"❌ 无法保存图片 {output_path}: {exc}")
        return False
    return output_path.is_file() and output_path.stat().st_size > 0


def main():
    parser = argparse.ArgumentParser(description="从100台相机视频中提取第一帧")
    parser.add_argument('--video-dir', type=str, 
                       default='data/2026-08-10-192847zhishangyue/video',
                       help='视频文件目录')
    parser.add_argument('--output-dir', type=str,
                       default='data/2026-08-10-192847zhishangyue/first_frames',
                       help='输出目录（所有第一帧保存在同一个文件夹）')
    parser.add_argument('--start-cam', type=int, default=1,
                       help='起始相机编号（默认：1）')
    parser.add_argument('--end-cam', type=int, default=100,
                       help='结束相机编号（默认：100）')
    
    args = parser.parse_args()
    
    # 路径配置
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    
    if not video_dir.is_dir():
        print(f"❌ 视频目录不存在: {video_dir}")
        return 1
    if args.start_cam < 0 or args.end_cam < args.start_cam:
        print("❌ 相机范围无效：要求 0 <= start-cam <= end-cam")
        return 1

    try:
        videos = discover_videos(video_dir)
    except ValueError as exc:
        print(f"❌ {exc}")
        return 1
    selected_videos = {
        cam_id: path for cam_id, path in videos.items()
        if args.start_cam <= cam_id <= args.end_cam
    }
    if not selected_videos:
        print(f"❌ 目录中没有相机范围 {args.start_cam}-{args.end_cam} 的可识别视频: {video_dir}")
        return 1
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("📸 提取视频第一帧")
    print("=" * 60)
    print(f"视频目录: {video_dir}")
    print(f"输出目录: {output_dir}")
    print(f"相机范围: {args.start_cam:03d} - {args.end_cam:03d}")
    print(f"识别视频: {len(selected_videos)} 个")
    print("=" * 60)
    print()
    
    # 统计信息
    success_count = 0
    failed_videos = []
    
    # 遍历所有相机视频
    for cam_num in tqdm(range(args.start_cam, args.end_cam + 1), 
                        desc="提取第一帧", 
                        unit="video"):
        video_path = selected_videos.get(cam_num)
        
        if video_path is None:
            print(f"\n⚠️  相机{cam_num:03d}: 没有对应视频")
            failed_videos.append(cam_num)
            continue

        video_filename = video_path.name
        
        # 输出文件名：cam001.png
        output_filename = f"cam{cam_num:03d}.png"
        output_path = output_dir / output_filename
        
        # 提取第一帧
        try:
            if extract_first_frame(video_path, output_path, cam_num):
                success_count += 1
            else:
                failed_videos.append(cam_num)
        except Exception as e:
            print(f"\n❌ 处理视频出错: {video_filename}")
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
    print(f"   {output_dir}/cam001.png")
    print(f"   {output_dir}/cam002.png")
    print(f"   ...")
    print(f"   {output_dir}/cam{args.end_cam:03d}.png")
    
    # 显示一些提取的文件
    extracted_files = sorted(output_dir.glob("cam*.png"))
    if extracted_files:
        print(f"\n✅ 实际生成了 {len(extracted_files)} 个文件")
        print("前5个文件:")
        for f in extracted_files[:5]:
            file_size = f.stat().st_size / 1024  # KB
            print(f"   {f.name} ({file_size:.1f} KB)")

    return 0 if not failed_videos and success_count > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
