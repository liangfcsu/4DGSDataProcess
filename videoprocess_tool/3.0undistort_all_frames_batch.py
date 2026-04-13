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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def load_camera_params_from_json(json_file):
    """从标定文件加载相机参数"""
    with open(json_file, 'r') as f:
        cameras = json.load(f)
    
    camera_params = {}
    
    for cam in cameras:
        img_name = cam['img_name']  # 例如 "cam001frame001.png"
        
        # 从img_name提取相机ID
        # cam001frame001.png -> 001
        cam_id = int(img_name[3:6])
        
        fx = cam['fx']
        fy = cam['fy']
        cx = cam['cx']
        cy = cam['cy']
        
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float64)
        
        # 无畸变系数（已在COLMAP中处理）
        dist_coeffs = np.zeros(5, dtype=np.float64)
        
        camera_params[cam_id] = {
            'K': K,
            'dist_coeffs': dist_coeffs,
            'width': cam['width'],
            'height': cam['height'],
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy
        }
    
    return camera_params

def undistort_image(input_path, output_path, K, dist_coeffs, width, height):
    """去畸变单张图像"""
    try:
        # 读取图像
        img = cv2.imread(str(input_path))
        if img is None:
            return False
        
        # 获取新的相机矩阵
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (width, height), 1, (width, height))
        
        # 创建去畸变映射
        map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeffs, None, new_K, (width, height), cv2.CV_32F)
        
        # 应用去畸变
        undistorted_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        
        # 裁剪ROI区域
        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted_img = undistorted_img[y:y+h, x:x+w]
        
        # 保存结果
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), undistorted_img)
        
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def main():
    print("🚀 使用估计的相机参数对所有帧进行去畸变处理")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    
    # 路径配置
    input_dir = base_dir / 'publicdata/coffee_martini_files/coffee_martini_images_origin'
    calib_file = base_dir / 'publicdata/coffee_martini_files/superglue/colmap_sfm/estimated_calib.json'
    output_dir = base_dir / 'publicdata/coffee_martini_files/undistorted_all_frames_complete_superglue'
    
    print(f"📂 输入目录: {input_dir}")
    print(f"📄 标定文件: {calib_file}")
    print(f"📂 输出目录: {output_dir}")
    
    # 检查输入
    if not input_dir.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    if not calib_file.exists():
        print(f"❌ 标定文件不存在: {calib_file}")
        return
    
    # 创建输出目录
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载相机参数
    print("\n🔧 加载相机参数...")
    camera_params = load_camera_params_from_json(calib_file)
    print(f"✅ 加载了 {len(camera_params)} 个相机的参数")
    
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
        width = params['width']
        height = params['height']
        
        output_cam_dir = output_dir / f'cam{cam_id:03d}'
        output_path = output_cam_dir / img_file.name
        
        if undistort_image(img_file, output_path, K, dist_coeffs, width, height):
            with lock:
                success_count += 1
            return True
        return False
    
    # 自动扫描输入目录中所有 cam* 子目录
    tasks = []
    for cam_dir in sorted(input_dir.iterdir()):
        if not cam_dir.is_dir() or not cam_dir.name.startswith('cam'):
            continue

        try:
            cam_id = int(cam_dir.name[3:])  # cam000 -> 0, cam020 -> 20
        except ValueError:
            continue

        # 获取所有图像文件
        image_files = sorted([f for f in cam_dir.iterdir()
                             if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])

        for img_file in image_files:
            tasks.append((cam_id, img_file))
            total_count += 1
    
    print(f"📊 共需处理 {total_count} 张图像")
    
    # 多线程处理
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_frame, task) for task in tasks]
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 100 == 0:
                print(f"  进度: {completed}/{len(futures)} (已成功: {success_count})")
    
    # 保存相机参数
    print("\n💾 保存相机参数...")
    undistorted_cameras = []
    for cam_id in sorted(camera_params.keys()):
        params = camera_params[cam_id]
        undistorted_cameras.append({
            'id': cam_id,
            'img_name': f'cam{cam_id:03d}frame001.png',
            'width': params['width'],
            'height': params['height'],
            'fx': params['fx'],
            'fy': params['fy'],
            'cx': params['cx'],
            'cy': params['cy']
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
        print("\n🎉 所有帧去畸变处理完成！")
    else:
        print(f"\n⚠️  部分帧处理失败: {total_count - success_count} 张")

if __name__ == "__main__":
    main()
