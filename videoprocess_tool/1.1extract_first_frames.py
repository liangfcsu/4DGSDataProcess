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
    cv2.imwrite(str(output_path), frame)
    return True


def main():
    parser = argparse.ArgumentParser(description="从100台相机视频中提取第一帧")
    parser.add_argument('--video-dir', type=str, 
                       default='100camersdata/100台 8秒',
                       help='视频文件目录')
    parser.add_argument('--output-dir', type=str,
                       default='data/100camdata/first_frames',
                       help='输出目录（所有第一帧保存在同一个文件夹）')
    parser.add_argument('--start-cam', type=int, default=1,
                       help='起始相机编号（默认：1）')
    parser.add_argument('--end-cam', type=int, default=100,
                       help='结束相机编号（默认：100）')
    
    args = parser.parse_args()
    
    # 路径配置
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    
    if not video_dir.exists():
        print(f"❌ 视频目录不存在: {video_dir}")
        return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("📸 提取视频第一帧")
    print("=" * 60)
    print(f"视频目录: {video_dir}")
    print(f"输出目录: {output_dir}")
    print(f"相机范围: {args.start_cam:03d} - {args.end_cam:03d}")
    print("=" * 60)
    print()
    
    # 统计信息
    success_count = 0
    failed_videos = []
    
    # 遍历所有相机视频
    for cam_num in tqdm(range(args.start_cam, args.end_cam + 1), 
                        desc="提取第一帧", 
                        unit="video"):
        video_filename = f"{cam_num:03d}.mp4"
        video_path = video_dir / video_filename
        
        if not video_path.exists():
            print(f"\n⚠️  相机{cam_num:03d}: 视频文件不存在 - {video_filename}")
            failed_videos.append(cam_num)
            continue
        
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


if __name__ == '__main__':
    main()
