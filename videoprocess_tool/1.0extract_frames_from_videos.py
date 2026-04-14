#!/usr/bin/env python3
"""
批量从100台相机的视频中提取帧
输入：100camersdata/100台 8秒/*.mp4
输出：data/100camdata/ims/cam{001-100}/cam{001-100}frame{001-n}.png
"""

import cv2
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import re


def extract_frames_from_video(video_path, output_dir, cam_num, fps=None, max_frames=None,
                              start_second=None, end_second=None):
    """
    从视频中提取帧
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        cam_num: 相机编号 (1-100)
        fps: 提取帧率（None表示提取所有帧）
        max_frames: 最大提取帧数（None表示不限制）
        start_second: 起始秒数（None表示从0秒开始）
        end_second: 结束秒数（None表示到视频结尾）
    """
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return 0
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 相机{cam_num:03d}: {video_path.name}")
    print(f"   视频FPS: {video_fps:.2f}, 总帧数: {total_frames}")

    # 根据秒数计算提取帧范围 [start_frame, end_frame)
    start_frame = 0
    end_frame = total_frames

    if start_second is not None:
        start_frame = max(0, min(int(start_second * video_fps), total_frames))
    if end_second is not None:
        end_frame = max(0, min(int(end_second * video_fps), total_frames))

    if end_frame <= start_frame:
        print(f"   ⚠️ 时间范围无效，已跳过（start={start_second}, end={end_second}）")
        cap.release()
        return 0

    # 定位到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    range_frames = end_frame - start_frame
    
    # 计算帧间隔
    if fps is not None and fps < video_fps:
        frame_interval = int(video_fps / fps)
        print(f"   提取帧率: {fps} FPS (每隔{frame_interval}帧提取一次)")
    else:
        frame_interval = 1
        print(f"   提取所有帧")
    
    # 提取帧
    frame_count = start_frame
    local_frame_count = 0
    saved_count = 0

    expected_saved = (range_frames + frame_interval - 1) // frame_interval
    pbar_total = expected_saved if max_frames is None else min(expected_saved, max_frames)

    pbar = tqdm(total=pbar_total,
                desc=f"Cam{cam_num:03d}", 
                unit="frame")

    if start_second is not None or end_second is not None:
        start_show = 0.0 if start_second is None else start_second
        end_show = "视频结尾" if end_second is None else end_second
        print(f"   时间范围: {start_show}s - {end_show}s")

    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 根据帧间隔决定是否保存
        if local_frame_count % frame_interval == 0:
            # 生成输出文件名：cam001frame001.png
            frame_filename = f"cam{cam_num:03d}frame{saved_count+1:03d}.png"
            output_path = output_dir / frame_filename
            
            # 保存帧
            cv2.imwrite(str(output_path), frame)
            saved_count += 1
            pbar.update(1)
            
            # 检查是否达到最大帧数
            if max_frames is not None and saved_count >= max_frames:
                break
        
        frame_count += 1
        local_frame_count += 1
    
    pbar.close()
    cap.release()
    
    print(f"   ✅ 保存了 {saved_count} 帧到 {output_dir}")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="从100台相机视频中批量提取帧")
    parser.add_argument('--video-dir', type=str, 
                       default='data/publicdata/coffee_martini_files/coffee_martini_video_origin',
                       help='视频文件目录')
    parser.add_argument('--output-dir', type=str,
                       default='data/publicdata/coffee_martini_files/coffee_martini_images',
                       help='输出根目录')
    parser.add_argument('--fps', type=float, default=None,
                       help='提取帧率（留空表示提取所有帧）')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='每个视频最大提取帧数（留空表示不限制）')
    parser.add_argument('--start-second', type=float, default=None,
                       help='起始秒数（默认从0秒开始）')
    parser.add_argument('--end-second', type=float, default=None,
                       help='结束秒数（默认到视频结尾）')
    parser.add_argument('--start-cam', type=int, default=None,
                       help='起始相机编号（默认：不限制）')
    parser.add_argument('--end-cam', type=int, default=None,
                       help='结束相机编号（默认：不限制）')
    
    args = parser.parse_args()
    
    # 路径配置
    video_dir = Path(args.video_dir)
    output_base_dir = Path(args.output_dir)
    
    if not video_dir.exists():
        print(f"❌ 视频目录不存在: {video_dir}")
        return

    if args.start_second is not None and args.start_second < 0:
        print("❌ --start-second 不能小于0")
        return

    if args.end_second is not None and args.end_second < 0:
        print("❌ --end-second 不能小于0")
        return

    if args.start_second is not None and args.end_second is not None and args.end_second <= args.start_second:
        print("❌ --end-second 必须大于 --start-second")
        return
    
    print("=" * 60)
    print("📹 批量视频提帧工具")
    print("=" * 60)
    print(f"视频目录: {video_dir}")
    print(f"输出目录: {output_base_dir}")
    if args.start_cam is None and args.end_cam is None:
        print("相机范围: 自动检测全部视频")
    else:
        start_show = "-inf" if args.start_cam is None else f"{args.start_cam:03d}"
        end_show = "inf" if args.end_cam is None else f"{args.end_cam:03d}"
        print(f"相机范围: {start_show} - {end_show}")
    print(f"提取帧率: {'全部帧' if args.fps is None else f'{args.fps} FPS'}")
    print(f"最大帧数: {'不限制' if args.max_frames is None else args.max_frames}")
    if args.start_second is None and args.end_second is None:
        print("时间范围: 全时段")
    else:
        start_show = 0.0 if args.start_second is None else args.start_second
        end_show = "视频结尾" if args.end_second is None else args.end_second
        print(f"时间范围: {start_show}s - {end_show}s")
    print("=" * 60)
    
    # 统计信息
    total_frames = 0
    processed_videos = 0
    failed_videos = []
    
    # 自动扫描视频目录中的所有.mp4文件
    video_files = sorted(video_dir.glob('*.mp4'))
    
    if not video_files:
        print(f"❌ 视频目录中找不到任何 .mp4 文件: {video_dir}")
        return
    
    print(f"📂 在目录中找到 {len(video_files)} 个视频文件")
    print()
    
    # 遍历所有相机视频
    for video_path in video_files:
        video_filename = video_path.name
        
        # 尝试从文件名中提取相机编号 (支持 cam00.mp4, cam001.mp4, 001.mp4 等格式)
        # 优先匹配 camXX 格式，其次匹配纯数字格式
        match = re.match(r'cam0*(\d+)', video_filename)
        if not match:
            match = re.match(r'0*(\d+)', video_filename)
        
        if not match:
            print(f"⚠️  跳过: {video_filename} (无法提取相机编号)")
            continue
        
        cam_num = int(match.group(1))
        
        # 根据 --start-cam 和 --end-cam 筛选
        if args.start_cam is not None and cam_num < args.start_cam:
            continue
        if args.end_cam is not None and cam_num > args.end_cam:
            continue
        
        # 创建输出目录：output_base_dir/cam{cam_num:03d}/
        cam_output_dir = output_base_dir / f"cam{cam_num:03d}"
        
        # 提取帧
        try:
            saved_count = extract_frames_from_video(
                video_path, 
                cam_output_dir, 
                cam_num,
                fps=args.fps,
                max_frames=args.max_frames,
                start_second=args.start_second,
                end_second=args.end_second
            )
            total_frames += saved_count
            processed_videos += 1
            print()
            
        except Exception as e:
            print(f"❌ 处理视频出错: {video_filename}")
            print(f"   错误信息: {e}")
            failed_videos.append(cam_num)
            print()
    
    # 总结
    print("=" * 60)
    print("✅ 提帧完成！")
    print("=" * 60)
    print(f"成功处理: {processed_videos} 个视频")
    print(f"总计提取: {total_frames} 帧")
    print(f"输出目录: {output_base_dir.absolute()}")
    
    if failed_videos:
        print(f"\n⚠️  失败/跳过的相机编号: {failed_videos}")
    
    print("\n📁 输出结构示例:")
    print(f"   {output_base_dir}/cam001/cam001frame001.png")
    print(f"   {output_base_dir}/cam001/cam001frame002.png")
    print(f"   ...")
    print(f"   {output_base_dir}/cam100/cam100frame001.png")


if __name__ == '__main__':
    main()
