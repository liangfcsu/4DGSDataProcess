#!/usr/bin/env python3
"""
undistort_for_hloc.py

专门为hloc重建优化的去畸变脚本。
确保去畸变后的图像和相机参数与hloc重建完全兼容，避免重投影误差。

特点：
1. 使用保守的alpha=0参数最小化内参变化
2. 验证去畸变前后的几何一致性
3. 生成与hloc兼容的相机格式
4. 提供详细的质量检查报告
"""

import json
import cv2
import numpy as np
from pathlib import Path
import shutil
import re
import math


def load_calib(path):
    """Load calibration data and preserve original text for precision extraction"""
    with open(path, 'r') as f:
        calib_text = f.read()
    calib_data = json.loads(calib_text)
    return calib_data, calib_text


def extract_params(camera, calib_text, camera_index):
    """从camera条目提取内参和畸变参数，兼容COLMAP转换的calibration格式"""
    
    # 检查是否是简单格式（直接有fx,fy,cx,cy字段）
    if 'fx' in camera and 'fy' in camera:
        # 简单格式：直接字段
        fx = float(camera['fx'])
        fy = float(camera['fy'])
        cx = float(camera['cx'])
        cy = float(camera['cy'])
        width = int(camera['width'])
        height = int(camera['height'])
        
        # 读取畸变参数（由 convert_colmap_to_calib.py 写入，格式 [k1, k2, p1, p2]）
        raw_dist = camera.get('distortion', [0.0, 0.0, 0.0, 0.0])
        # OpenCV 畸变系数顺序: [k1, k2, p1, p2, k3]
        k1 = float(raw_dist[0]) if len(raw_dist) > 0 else 0.0
        k2 = float(raw_dist[1]) if len(raw_dist) > 1 else 0.0
        p1 = float(raw_dist[2]) if len(raw_dist) > 2 else 0.0
        p2 = float(raw_dist[3]) if len(raw_dist) > 3 else 0.0
        k3 = float(raw_dist[4]) if len(raw_dist) > 4 else 0.0
        dist = [k1, k2, p1, p2, k3]
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        
        # 获取位姿（如果有的话）
        rotation = camera.get('rotation')
        translation = camera.get('position')
        
        return K, np.array(dist, dtype=np.float64), (width, height), rotation, translation
        
    # 检查是否是COLMAP转换的格式
    elif 'camera_parameters' in camera:
        # 新格式：COLMAP转换的格式
        params = camera['camera_parameters']
        image_size = camera['image_size']
        
        # 提取内参
        fx = float(params['f']['val'])
        ar = float(params['ar']['val'])
        fy = fx * ar
        cx = float(params['cx']['val'])
        cy = float(params['cy']['val'])
        
        # 提取畸变参数
        k1 = float(params['k1']['val'])
        k2 = float(params['k2']['val'])
        k3 = float(params['k3']['val'])
        p1 = float(params['p1']['val'])
        p2 = float(params['p2']['val'])
        
        dist = [k1, k2, p1, p2, k3]
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        
        # 获取位姿
        rotation = camera.get('transform', {}).get('rotation')
        translation = camera.get('transform', {}).get('translation')
        
        return K, np.array(dist, dtype=np.float64), (image_size['width'], image_size['height']), rotation, translation
        
    else:
        # 原格式：使用正则表达式提取
        model = camera['model']['ptr_wrapper']['data']
        params = model['parameters']
        image_size = model['CameraModelCRT']['CameraModelBase']['imageSize']

        # 使用正则表达式从原始文本中提取高精度数值
        f_values = re.findall(r'"f":\s*{\s*"val":\s*([0-9.-]+)', calib_text)
        cx_values = re.findall(r'"cx":\s*{\s*"val":\s*([0-9.-]+)', calib_text)  
        cy_values = re.findall(r'"cy":\s*{\s*"val":\s*([0-9.-]+)', calib_text)
        ar_values = re.findall(r'"ar":\s*{\s*"val":\s*([0-9.-]+)', calib_text)
        
        # 畸变参数提取
        k1_values = re.findall(r'"k1":\s*{\s*"val":\s*([0-9.-]+)', calib_text)
        k2_values = re.findall(r'"k2":\s*{\s*"val":\s*([0-9.-]+)', calib_text)
        k3_values = re.findall(r'"k3":\s*{\s*"val":\s*([0-9.-]+)', calib_text)
        p1_values = re.findall(r'"p1":\s*{\s*"val":\s*([0-9.-]+)', calib_text)
        p2_values = re.findall(r'"p2":\s*{\s*"val":\s*([0-9.-]+)', calib_text)
        # 使用高精度字符串值
        f_str = f_values[camera_index] if camera_index < len(f_values) else '7000.0'
        ar_str = ar_values[camera_index] if camera_index < len(ar_values) else '1.0'
        cx_str = cx_values[camera_index] if camera_index < len(cx_values) else '2880.0'
        cy_str = cy_values[camera_index] if camera_index < len(cy_values) else '1620.0'
        
        # 转换为浮点数
        fx = float(f_str)
        ar = float(ar_str)
        fy = fx * ar
        cx = float(cx_str)
        cy = float(cy_str)

        # 畸变参数处理
        k1 = float(k1_values[camera_index]) if camera_index < len(k1_values) else 0.0
        k2 = float(k2_values[camera_index]) if camera_index < len(k2_values) else 0.0
        k3 = float(k3_values[camera_index]) if camera_index < len(k3_values) else 0.0
        p1 = float(p1_values[camera_index]) if camera_index < len(p1_values) else 0.0
        p2 = float(p2_values[camera_index]) if camera_index < len(p2_values) else 0.0
        
        dist = [k1, k2, p1, p2, k3]

        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

        # rotation & translation from camera['transform']
        transform = camera.get('transform', {})
        rotation = transform.get('rotation')
        translation = transform.get('translation')

        return K, np.array(dist, dtype=np.float64), (image_size['width'], image_size['height']), rotation, translation


def validate_undistortion_quality(original_K, undistorted_K, dist_coeffs):
    """验证去畸变质量"""
    
    # 计算内参变化百分比
    fx_change = abs(undistorted_K[0,0] - original_K[0,0]) / original_K[0,0] * 100
    fy_change = abs(undistorted_K[1,1] - original_K[1,1]) / original_K[1,1] * 100
    cx_change = abs(undistorted_K[0,2] - original_K[0,2]) / original_K[0,2] * 100
    cy_change = abs(undistorted_K[1,2] - original_K[1,2]) / original_K[1,2] * 100
    
    # 评估畸变严重程度
    k1, k2 = dist_coeffs[0], dist_coeffs[1]
    distortion_severity = abs(k1) + abs(k2)
    
    quality_report = {
        'fx_change_percent': fx_change,
        'fy_change_percent': fy_change,
        'cx_change_percent': cx_change,
        'cy_change_percent': cy_change,
        'distortion_severity': distortion_severity,
        'k1': k1,
        'k2': k2
    }
    
    # 判断质量等级
    max_change = max(fx_change, fy_change)
    if max_change < 1.0:
        quality_report['quality'] = 'EXCELLENT'
    elif max_change < 3.0:
        quality_report['quality'] = 'GOOD'
    elif max_change < 8.0:
        quality_report['quality'] = 'ACCEPTABLE'
    else:
        quality_report['quality'] = 'POOR'
    
    return quality_report


def undistort_for_hloc(calib_path, images_dir, out_dir, alpha=0.0, validation_mode=True):
    """
    专门为hloc优化的去畸变函数
    
    Args:
        calib_path: 原始标定文件路径
        images_dir: 原始图像目录 
        out_dir: 输出目录
        alpha: 去畸变参数 (0.0=最小内参变化, 1.0=保留所有像素)
        validation_mode: 是否启用验证模式
    """
    
    print("=== Hloc兼容去畸变处理 ===")
    
    calib, calib_text = load_calib(calib_path)
    
    # 处理不同格式的标定文件
    if isinstance(calib, dict) and 'Calibration' in calib:
        # 旧格式：嵌套字典
        cameras = calib['Calibration']['cameras']
    elif isinstance(calib, list):
        # 新格式：直接列表
        cameras = calib
    else:
        raise ValueError(f"不支持的标定文件格式: {type(calib)}")
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 输出目录
    images_out = out_path / 'images_undistorted'
    images_out.mkdir(parents=True, exist_ok=True)
    
    tool_out = out_path / 'tool'
    tool_out.mkdir(parents=True, exist_ok=True)
    
    updated_cameras = []
    quality_reports = []
    processed = 0
    
    for i, cam in enumerate(cameras):
        # 从calibration文件获取实际的图像文件名
        actual_img_name = cam.get('img_name', f"{i+1:03d}.png")
        src = Path(images_dir) / actual_img_name
        
        # 如果指定的图像不存在，尝试其他格式
        if not src.exists():
            # 尝试不同的命名格式
            for pattern in [f"{i+1:03d}.png", f"{i+1:03d}.jpg", f"{i+1:03d}.jpeg"]:
                alt_src = Path(images_dir) / pattern
                if alt_src.exists():
                    src = alt_src
                    actual_img_name = pattern
                    break
        
        if not src.exists():
            print(f"跳过：未找到图片 {actual_img_name} (索引 {i+1})")
            continue

        img = cv2.imread(str(src))
        if img is None:
            print(f"无法读取图片: {src}")
            continue

        # 提取相机参数
        K, dist, calib_size, rotation, translation = extract_params(cam, calib_text, i)
        
        h, w = img.shape[:2]
        
        # 计算去畸变后的相机矩阵
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=alpha)
        
        # 验证去畸变质量
        if validation_mode:
            quality = validate_undistortion_quality(K, new_K, dist)
            quality['camera_id'] = i + 1
            quality_reports.append(quality)
            
            print(f"相机{i+1:2d}: 质量={quality['quality']:10s} "
                  f"fx变化={quality['fx_change_percent']:4.1f}% "
                  f"畸变={quality['distortion_severity']:.3f}")
        
        # 执行去畸变
        if alpha == 0.0 and roi[2] > 0 and roi[3] > 0:
            # 裁剪到有效区域
            undist_full = cv2.undistort(img, K, dist, None, new_K)
            x, y, w_roi, h_roi = roi
            final_img = undist_full[y:y+h_roi, x:x+w_roi]
            final_w, final_h = w_roi, h_roi
            
            # 调整主点
            K_final = new_K.copy()
            K_final[0,2] -= x  # 调整cx
            K_final[1,2] -= y  # 调整cy
        else:
            # 保持原始尺寸
            final_img = cv2.undistort(img, K, dist, None, new_K)
            final_w, final_h = w, h
            K_final = new_K
        
        # 保存去畸变图像 - 使用标准化名称格式
        output_name = f"{i+1:03d}.png"
        out_file = images_out / output_name
        cv2.imencode('.png', final_img)[1].tofile(str(out_file))
        
        # 构建相机条目
        fx = float(K_final[0,0])
        fy = float(K_final[1,1])
        cx = float(K_final[0,2])
        cy = float(K_final[1,2])
        
        # 处理位姿信息
        pos = None
        if translation:
            if isinstance(translation, dict):
                pos = [translation.get('x', 0), translation.get('y', 0), translation.get('z', 0)]
            else:
                pos = list(translation)
        
        rot = None
        if rotation:
            if isinstance(rotation, dict):
                rot = {
                    'rx': rotation.get('rx', 0),
                    'ry': rotation.get('ry', 0),
                    'rz': rotation.get('rz', 0)
                }
            else:
                rot = rotation
        
        camera_entry = {
            'id': i,
            'img_name': output_name,  # 使用输出的标准化名称
            'width': final_w,
            'height': final_h,
            'position': pos,
            'rotation': rot,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy
        }
        updated_cameras.append(camera_entry)
        processed += 1
    
    # 保存相机文件
    cameras_file = tool_out / 'cameras_undistorted.json'
    with open(cameras_file, 'w') as f:
        json.dump(updated_cameras, f, indent=2)
    
    print(f"\n已处理 {processed} 张图像")
    print(f"去畸变图像保存至: {images_out}")
    print(f"相机参数保存至: {cameras_file}")
    
    # 生成质量报告
    if validation_mode and quality_reports:
        print(f"\n=== 去畸变质量报告 ===")
        
        quality_counts = {}
        for report in quality_reports:
            quality = report['quality']
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        print(f"质量分布:")
        for quality, count in quality_counts.items():
            print(f"  {quality}: {count} 个相机")
        
        # 检查是否有质量差的相机
        poor_cameras = [r for r in quality_reports if r['quality'] == 'POOR']
        if poor_cameras:
            print(f"\n⚠️ 警告：{len(poor_cameras)} 个相机去畸变质量较差：")
            for cam in poor_cameras:
                print(f"  相机{cam['camera_id']}: fx变化={cam['fx_change_percent']:.1f}%")
        
        # 保存详细质量报告
        quality_file = tool_out / 'undistortion_quality_report.json'
        with open(quality_file, 'w') as f:
            json.dump(quality_reports, f, indent=2)
        print(f"详细质量报告: {quality_file}")
    
    return updated_cameras, quality_reports


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='为hloc优化的去畸变处理')
    parser.add_argument('--calib', default='3dgs_dataprocess/calib/calib12_1.json',
                        help='原始标定文件路径')
    parser.add_argument('--images', default='3dgs_dataprocess/origin_seg_images',
                        help='原始图像目录')
    parser.add_argument('--output', default='3dgs_dataprocess_seg_undistorted',
                        help='输出目录')
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='去畸变参数 (0.0=最小内参变化, 1.0=保留所有像素)')
    parser.add_argument('--no-validation', action='store_true',
                        help='禁用质量验证')
    
    args = parser.parse_args()
    
    validation_mode = not args.no_validation
    
    cameras, reports = undistort_for_hloc(
        args.calib,
        args.images, 
        args.output,
        alpha=args.alpha,
        validation_mode=validation_mode
    )
    
    print(f"\n✅ 去畸变完成！")
    print(f"建议下一步：使用去畸变后的图像进行hloc重建")


if __name__ == '__main__':
    main()