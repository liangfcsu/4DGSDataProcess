#!/usr/bin/env python3
"""
convert_colmap_to_calib.py

将COLMAP SfM重建结果转换为标定文件格式
支持从COLMAP的cameras.txt、images.txt转换为JSON标定文件

输入: COLMAP sparse重建结果目录
输出: JSON格式的相机标定文件
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import os
import re


def parse_real_cam_id(image_name, fallback_id):
    """优先从 camXXX... 文件名提取真实相机ID，失败时回退。"""
    m = re.match(r'^cam(\d+)', image_name)
    if m:
        return int(m.group(1))
    return fallback_id

def parse_colmap_cameras(cameras_file):
    """解析COLMAP cameras.txt文件"""
    cameras = {}
    
    with open(cameras_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
                
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            
            # 根据相机模型提取参数
            if model == "SIMPLE_PINHOLE":
                fx = fy = params[0]
                cx = params[1] 
                cy = params[2]
                distortion = [0.0, 0.0, 0.0, 0.0]  # 无畸变
            elif model == "PINHOLE":
                fx = params[0]
                fy = params[1]
                cx = params[2]
                cy = params[3]
                distortion = [0.0, 0.0, 0.0, 0.0]  # 无畸变
            elif model == "SIMPLE_RADIAL":
                fx = fy = params[0]
                cx = params[1]
                cy = params[2]
                k1 = params[3]
                distortion = [k1, 0.0, 0.0, 0.0]
            elif model == "RADIAL":
                fx = fy = params[0]
                cx = params[1]
                cy = params[2]
                k1 = params[3]
                k2 = params[4] if len(params) > 4 else 0.0
                distortion = [k1, k2, 0.0, 0.0]
            elif model == "OPENCV":
                fx = params[0]
                fy = params[1]
                cx = params[2]
                cy = params[3]
                k1 = params[4]
                k2 = params[5]
                p1 = params[6]
                p2 = params[7]
                distortion = [k1, k2, p1, p2]
            else:
                print(f"⚠️ 不支持的相机模型: {model}")
                continue
            
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'distortion': distortion
            }
    
    return cameras

def parse_colmap_images(images_file):
    """解析COLMAP images.txt文件"""
    images = {}
    
    with open(images_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue
        
        parts = line.split()
        image_id = int(parts[0])
        
        # 四元数 (qw, qx, qy, qz)
        qw = float(parts[1])
        qx = float(parts[2])
        qy = float(parts[3])
        qz = float(parts[4])
        
        # 平移向量 (tx, ty, tz)
        tx = float(parts[5])
        ty = float(parts[6])
        tz = float(parts[7])
        
        camera_id = int(parts[8])
        # 文件名可能包含空格，需要合并剩余部分
        image_name = ' '.join(parts[9:])
        
        # 将四元数转换为旋转矩阵，再转换为欧拉角
        rotation_matrix = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        rx, ry, rz = rotation_matrix_to_euler(rotation_matrix)
        
        # COLMAP使用的是从世界坐标到相机坐标的变换
        # 我们需要相机在世界坐标系中的位置和姿态
        # 相机位置 = -R^T * t
        R_inv = rotation_matrix.T
        camera_position = -R_inv @ np.array([tx, ty, tz])
        
        # 相机姿态是R^T的欧拉角
        camera_rx, camera_ry, camera_rz = rotation_matrix_to_euler(R_inv)
        
        images[image_id] = {
            'name': image_name,
            'camera_id': camera_id,
            'position': camera_position.tolist(),
            'rotation': {
                'rx': camera_rx,
                'ry': camera_ry, 
                'rz': camera_rz
            }
        }
        
        # 跳过下一行（特征点数据）
        i += 2
    
    return images

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """四元数转旋转矩阵"""
    # 归一化四元数
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    
    # 计算旋转矩阵
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]
    ])
    return R

def rotation_matrix_to_euler(R):
    """旋转矩阵转欧拉角 (ZYX顺序)"""
    sy = np.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
        
    return x, y, z

def convert_colmap_to_calib(colmap_dir, output_file):
    """将COLMAP结果转换为标定文件格式"""
    
    colmap_path = Path(colmap_dir)
    cameras_file = colmap_path / 'cameras.txt'
    images_file = colmap_path / 'images.txt'
    
    if not cameras_file.exists():
        print(f"❌ 未找到cameras.txt文件: {cameras_file}")
        return False
        
    if not images_file.exists():
        print(f"❌ 未找到images.txt文件: {images_file}")
        return False
    
    print(f"📖 解析COLMAP结果...")
    print(f"  - cameras.txt: {cameras_file}")
    print(f"  - images.txt: {images_file}")
    
    # 解析COLMAP文件
    cameras = parse_colmap_cameras(cameras_file)
    images = parse_colmap_images(images_file)
    
    print(f"📊 COLMAP解析结果:")
    print(f"  - 相机模型数: {len(cameras)}")
    print(f"  - 重建图像数: {len(images)}")
    
    # 转换为标定文件格式（与undistort_for_hloc.py兼容）
    calib_data = []
    
    for image_id, image_info in images.items():
        camera_id = image_info['camera_id']
        
        if camera_id not in cameras:
            print(f"⚠️ 图像 {image_info['name']} 的相机ID {camera_id} 未找到")
            continue
        
        camera = cameras[camera_id]
        
        # 构建标定数据项（与generate_pointcloud_multicam.py兼容的格式）
        real_cam_id = parse_real_cam_id(image_info['name'], image_id - 1)
        calib_item = {
            'id': real_cam_id,
            'reconstruction_id': image_id - 1,  # 保留重建序号，便于排查
            'img_name': image_info['name'],
            'width': camera['width'],
            'height': camera['height'],
            'fx': camera['fx'],
            'fy': camera['fy'],
            'cx': camera['cx'],
            'cy': camera['cy'],
            'distortion': camera['distortion'],  # k1, k2, p1, p2
            'position': image_info['position'],
            'rotation': image_info['rotation']
        }
        
        calib_data.append(calib_item)
    
    # 按id排序
    calib_data.sort(key=lambda x: x['id'])
    
    # 保存标定文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(calib_data, f, indent=2)
    
    print(f"✅ 标定文件已保存: {output_path}")
    print(f"📋 包含 {len(calib_data)} 台相机的标定参数")
    
    # 显示统计信息
    if calib_data:
        fx_values = [cam['fx'] for cam in calib_data]
        fy_values = [cam['fy'] for cam in calib_data]
        print(f"📊 相机参数统计:")
        print(f"  - 焦距范围: fx [{min(fx_values):.1f}, {max(fx_values):.1f}]")
        print(f"  - 焦距范围: fy [{min(fy_values):.1f}, {max(fy_values):.1f}]")
        print(f"  - 图像尺寸: {calib_data[0]['width']}x{calib_data[0]['height']}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="将COLMAP SfM结果转换为标定文件格式")
    parser.add_argument('--colmap_dir', required=True, 
                       help="COLMAP sparse重建结果目录")
    parser.add_argument('--output', required=True,
                       help="输出标定文件路径")
    
    args = parser.parse_args()
    
    print("🔄 COLMAP结果转换为标定文件")
    print(f"输入: {args.colmap_dir}")
    print(f"输出: {args.output}")
    
    success = convert_colmap_to_calib(args.colmap_dir, args.output)
    
    if success:
        print("✅ 转换完成!")
    else:
        print("❌ 转换失败!")
        sys.exit(1)

if __name__ == '__main__':
    main()