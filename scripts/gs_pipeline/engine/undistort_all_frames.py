#!/usr/bin/env python3
"""
undistort_all_frames_batch.py  使用supergule估计的参数批量对图像进行去畸变

使用已估计的相机参数对 usdata/2_Seconds 中的所有帧进行去畸变处理
"""

import json
import cv2
import numpy as np
from pathlib import Path
import shutil
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import argparse
import sys

def load_camera_params_from_json(json_file):
    """从标定文件加载相机参数"""
    with open(json_file, 'r') as f:
        cameras = json.load(f)
    
    camera_params = {}
    
    for cam in cameras:
        img_name = cam['img_name']

        # 兼容 cam001frame001.png / 001.png 等命名
        cam_id = None
        if img_name.startswith('cam') and len(img_name) >= 6 and img_name[3:6].isdigit():
            cam_id = int(img_name[3:6])
        else:
            m = re.match(r'^(\d+)', img_name)
            if m:
                # 001.png 按相机 0 处理，便于与 cam000 对齐
                cam_id = int(m.group(1)) - 1

        if cam_id is None:
            continue
        
        fx = cam['fx']
        fy = cam['fy']
        cx = cam['cx']
        cy = cam['cy']
        
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float64)
        
        # 读取 COLMAP 估计畸变参数（格式 [k1, k2, p1, p2]），并补齐 k3
        raw_dist = cam.get('distortion', [0.0, 0.0, 0.0, 0.0])
        k1 = float(raw_dist[0]) if len(raw_dist) > 0 else 0.0
        k2 = float(raw_dist[1]) if len(raw_dist) > 1 else 0.0
        p1 = float(raw_dist[2]) if len(raw_dist) > 2 else 0.0
        p2 = float(raw_dist[3]) if len(raw_dist) > 3 else 0.0
        k3 = float(raw_dist[4]) if len(raw_dist) > 4 else 0.0
        dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (cam['width'], cam['height']), 0.0)
        if roi[2] > 0 and roi[3] > 0:
            x, y, w_roi, h_roi = roi
            K_final = new_K.copy()
            K_final[0, 2] -= x
            K_final[1, 2] -= y
            final_width, final_height = w_roi, h_roi
        else:
            K_final = new_K
            final_width, final_height = cam['width'], cam['height']
        
        camera_params[cam_id] = {
            'cam_id': cam_id,
            'output_id': int(cam.get('id', cam_id)),
            'output_img_name': cam.get('img_name', f'cam{cam_id:03d}frame001.png'),
            'K': K,
            'dist_coeffs': dist_coeffs,
            'width': int(cam['width']),
            'height': int(cam['height']),
            'fx': float(K_final[0, 0]),
            'fy': float(K_final[1, 1]),
            'cx': float(K_final[0, 2]),
            'cy': float(K_final[1, 2]),
            'final_width': int(final_width),
            'final_height': int(final_height),
            'position': cam.get('position'),
            'rotation': cam.get('rotation')
        }
    
    return camera_params

def undistort_image(input_path, output_path, K, dist_coeffs):
    """去畸变单张图像"""
    try:
        # 读取图像
        img = cv2.imread(str(input_path))
        if img is None:
            return False

        h, w = img.shape[:2]
        
        # 与 undistort_for_hloc.py 对齐：alpha=0 保守裁剪
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), alpha=0.0)
        undistorted_img = cv2.undistort(img, K, dist_coeffs, None, new_K)
        
        # 裁剪ROI区域
        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted_img = undistorted_img[y:y+h, x:x+w]
        
        # 保存结果
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(output_path), undistorted_img))
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="使用SuperGlue/COLMAP估计的相机参数批量去畸变")
    parser.add_argument('--input-dir', required=True, help='原始多帧目录（camXXX/图像）')
    parser.add_argument('--calib-file', required=True, help='estimated_calib.json')
    parser.add_argument('--output-dir', required=True, help='去畸变多帧输出目录')
    parser.add_argument('--workers', type=int, default=8, help='CPU去畸变线程数')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的输出目录')
    args = parser.parse_args()

    print("🚀 使用估计的相机参数对所有帧进行去畸变处理")
    print("=" * 60)

    input_dir = Path(args.input_dir).resolve()
    calib_file = Path(args.calib_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    print(f"📂 输入目录: {input_dir}")
    print(f"📄 标定文件: {calib_file}")
    print(f"📂 输出目录: {output_dir}")
    
    # 检查输入
    if not input_dir.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return 1
    
    if not calib_file.exists():
        print(f"❌ 标定文件不存在: {calib_file}")
        return 1

    if args.workers <= 0:
        print("❌ --workers 必须大于0")
        return 1
    
    # 创建输出目录
    if output_dir.exists():
        if not args.overwrite:
            print(f"❌ 输出目录已存在，请使用 --overwrite: {output_dir}")
            return 1
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载相机参数
    print("\n🔧 加载相机参数...")
    camera_params = load_camera_params_from_json(calib_file)
    print(f"✅ 加载了 {len(camera_params)} 个相机的参数")
    if not camera_params:
        print("❌ 标定文件中没有可用相机")
        return 1
    
    # 处理所有帧
    print(f"\n🚀 开始批量去畸变处理...")
    
    total_count = 0
    success_count = 0
    lock = threading.Lock()
    
    def process_frame(args):
        nonlocal success_count, total_count
        cam_id, img_file = args
        
        if cam_id not in camera_params:
            return False
        
        params = camera_params[cam_id]
        K = params['K']
        dist_coeffs = params['dist_coeffs']
        
        output_cam_dir = output_dir / f'cam{cam_id:03d}'
        output_path = output_cam_dir / img_file.name
        
        if undistort_image(img_file, output_path, K, dist_coeffs):
            with lock:
                success_count += 1
            return True
        return False
    
    # 自动扫描输入目录中所有 cam* 子目录。无标定 SfM 可能只注册
    # 其中一部分相机；未出现在 estimated_calib.json 中的相机应该被剔除，
    # 而不是计为去畸变失败并中止整条流水线。
    tasks = []
    skipped_cam_ids = []
    for cam_dir in sorted(input_dir.iterdir()):
        if not cam_dir.is_dir() or not cam_dir.name.startswith('cam'):
            continue

        try:
            cam_id = int(cam_dir.name[3:])  # cam000 -> 0, cam020 -> 20
        except ValueError:
            continue

        if cam_id not in camera_params:
            skipped_cam_ids.append(cam_id)
            continue

        # 获取所有图像文件
        image_files = sorted([f for f in cam_dir.iterdir()
                             if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])

        for img_file in image_files:
            tasks.append((cam_id, img_file))
            total_count += 1

    if skipped_cam_ids:
        preview = ', '.join(f'cam{cam_id:03d}' for cam_id in skipped_cam_ids[:12])
        suffix = ' …' if len(skipped_cam_ids) > 12 else ''
        print(
            f"⚠️  SfM 未注册 {len(skipped_cam_ids)} 台相机，已跳过: "
            f"{preview}{suffix}"
        )
    
    print(f"📊 共需处理 {total_count} 张图像")
    if total_count == 0:
        print("❌ 输入目录中没有找到多相机图像")
        return 1
    
    # 多线程处理
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_frame, task) for task in tasks]
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 100 == 0:
                print(f"  进度: {completed}/{len(futures)} (已成功: {success_count})")
    
    # 保存相机参数
    print("\n💾 保存相机参数...")
    undistorted_cameras = []
    ordered_params = sorted(camera_params.values(), key=lambda x: x['output_id'])
    for params in ordered_params:
        output_id = params['output_id']
        undistorted_cameras.append({
            'id': output_id,
            'img_name': params['output_img_name'],
            'width': params['final_width'],
            'height': params['final_height'],
            'fx': params['fx'],
            'fy': params['fy'],
            'cx': params['cx'],
            'cy': params['cy'],
            'position': params['position'],
            'rotation': params['rotation']
        })
    
    tool_dir = output_dir / 'tool'
    tool_dir.mkdir(parents=True, exist_ok=True)
    
    cameras_file = tool_dir / 'cameras_undistorted.json'
    with open(cameras_file, 'w') as f:
        json.dump(undistorted_cameras, f, indent=2)
    
    print(f"\n📊 处理完成！")
    print(f"✅ 成功处理: {success_count}/{total_count} 张图像")
    print(f"📂 输出目录: {output_dir}")
    print(f"📄 相机参数: {cameras_file}")
    
    if success_count == total_count:
        print("\n🎉 已注册相机的所有帧去畸变处理完成！")
        return 0
    else:
        print(f"\n⚠️  部分帧处理失败: {total_count - success_count} 张")
        return 1

if __name__ == "__main__":
    sys.exit(main())
