#!/usr/bin/env python3
"""
转换标定文件 calib12_1.json 为 cameras.json 格式，保留高精度相机内参和畸变参数。
支持自动检测 JSON 结构（简单或复杂），并从原始文本中提取高精度数值。
可以选择验证图像尺寸是否与标定文件匹配。
"""
import argparse
import json
import os
import sys
from pathlib import Path
from PIL import Image
import math


def extract_cameras_from_calib(calib_data, calib_text):
    """从 calib12_1.json 中提取相机内参、位姿和畸变参数，并保持原始文本的高精度值"""
    cameras = []
    
    calibration = calib_data.get('Calibration', calib_data)
    cam_list = calibration.get('cameras', [])
    
    # 提取原始文本中的高精度数值字符串，以便最终写出时保持精度
    import re

    f_values = re.findall(r'"f":\s*{\s*"val":\s*([0-9.-]+)', calib_text)
    cx_values = re.findall(r'"cx":\s*{\s*"val":\s*([0-9.-]+)', calib_text)  
    cy_values = re.findall(r'"cy":\s*{\s*"val":\s*([0-9.-]+)', calib_text)
    ar_values = re.findall(r'"ar":\s*{\s*"val":\s*([0-9.-]+)', calib_text)
    
    # 提取畸变参数的原始字符串值
    distortion_patterns = {
        'k1': r'"k1":\s*{\s*"val":\s*([0-9.-]+)',
        'k2': r'"k2":\s*{\s*"val":\s*([0-9.-]+)',
        'k3': r'"k3":\s*{\s*"val":\s*([0-9.-]+)',
        'k4': r'"k4":\s*{\s*"val":\s*([0-9.-]+)',
        'k5': r'"k5":\s*{\s*"val":\s*([0-9.-]+)',
        'k6': r'"k6":\s*{\s*"val":\s*([0-9.-]+)',
        'p1': r'"p1":\s*{\s*"val":\s*([0-9.-]+)',
        'p2': r'"p2":\s*{\s*"val":\s*([0-9.-]+)',
        's1': r'"s1":\s*{\s*"val":\s*([0-9.-]+)',
        's2': r'"s2":\s*{\s*"val":\s*([0-9.-]+)',
        's3': r'"s3":\s*{\s*"val":\s*([0-9.-]+)',
        's4': r'"s4":\s*{\s*"val":\s*([0-9.-]+)',
        'tauX': r'"tauX":\s*{\s*"val":\s*([0-9.-]+)',
        'tauY': r'"tauY":\s*{\s*"val":\s*([0-9.-]+)'
    }
    
    # 为每个畸变参数提取所有匹配的值
    distortion_values = {}
    for param_name, pattern in distortion_patterns.items():
        distortion_values[param_name] = re.findall(pattern, calib_text)
    
    for i, cam_entry in enumerate(cam_list):
        try:
            # Extract intrinsics from nested structure
            ptr_wrapper = cam_entry['model']['ptr_wrapper']
            data = ptr_wrapper['data']
            
            # Get image size
            image_size = data['CameraModelCRT']['CameraModelBase']['imageSize']
            width = image_size['width']
            height = image_size['height']
            
            # 使用原始字符串精度的值，若不足则回退到默认值
            f_str = f_values[i] if i < len(f_values) else '7000.0'
            ar_str = ar_values[i] if i < len(ar_values) else '1.0'
            cx_str = cx_values[i] if i < len(cx_values) else '2880.0'
            cy_str = cy_values[i] if i < len(cy_values) else '1620.0'
            
            # 转换为浮点数进行计算，但记录原始字符串
            f = float(f_str)
            ar = float(ar_str)
            cx = float(cx_str)
            cy = float(cy_str)
            
            # Calculate fx, fy from f and aspect ratio
            fx = f
            fy = f * ar
            
            # Extract distortion parameters with original precision
            distortion = {}
            original_distortion_strings = {}
            for param_name in distortion_patterns.keys():
                if i < len(distortion_values[param_name]):
                    param_str = distortion_values[param_name][i]
                    original_distortion_strings[param_name] = param_str
                    distortion[param_name] = float(param_str)
                else:
                    # Default to 0.0 if parameter not found
                    original_distortion_strings[param_name] = '0.0'
                    distortion[param_name] = 0.0
            
            # Extract pose (CTW format - Camera to World transform)
            transform = cam_entry['transform']
            rotation = transform['rotation']
            translation = transform['translation']
            
            # 旋转以欧拉角形式表示（单位：弧度）
            rx = rotation['rx']
            ry = rotation['ry'] 
            rz = rotation['rz']
            
            # Translation is camera position in world coordinates
            position = [
                translation['x'],
                translation['y'],
                translation['z']
            ]
            
            cameras.append({
                'width': width,
                'height': height,
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'distortion': distortion,  # 添加畸变参数
                'position': position,
                'rotation': {'rx': rx, 'ry': ry, 'rz': rz},
                # 记录原始字符串，确保最终 JSON 保持与 calib 文件一致的精度
                '_original_strings': {
                    'f': f_str,
                    'cx': cx_str,
                    'cy': cy_str,
                    'distortion': original_distortion_strings  # 保存畸变参数的原始字符串
                }
            })
            
        except (KeyError, TypeError) as e:
            print(f"Error processing camera {i}: {e}")
            # 出现异常时用默认值代替，避免整体失败
            default_distortion = {param: 0.0 for param in distortion_patterns.keys()}
            cameras.append({
                'width': 5760,
                'height': 3240,
                'fx': 7000.0,
                'fy': 7000.0,
                'cx': 2880.0,
                'cy': 1620.0,
                'distortion': default_distortion,
                'position': [0.0, 0.0, 0.0],
                'rotation': {'rx': 0.0, 'ry': 0.0, 'rz': 0.0}
            })
    
    return cameras


def get_image_files(images_dir):
    """返回按文件名排序的图像文件路径列表"""
    images_path = Path(images_dir)
    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Get all image files and sort them
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg']:
        image_files.extend(images_path.glob(f'*{ext}'))
        image_files.extend(images_path.glob(f'*{ext.upper()}'))
    
    # Sort by filename
    image_files.sort(key=lambda x: x.name)
    
    return image_files


def verify_image_dimensions(image_path, expected_width, expected_height):
    """可选：校验图像尺寸是否与标定文件中的值一致"""
    try:
        with Image.open(image_path) as img:
            actual_width, actual_height = img.size
            if actual_width != expected_width or actual_height != expected_height:
                print(f"Warning: {image_path.name} dimensions {actual_width}x{actual_height} "
                      f"don't match expected {expected_width}x{expected_height}")
                return actual_width, actual_height
        return expected_width, expected_height
    except Exception as e:
        print(f"Warning: Could not verify dimensions for {image_path.name}: {e}")
        return expected_width, expected_height


def main():
    parser = argparse.ArgumentParser(description='Convert calib.json to cameras.json format')
    parser.add_argument('--images-dir', default='data1.15/origin/origin_images',
                        help='Directory containing original images')
    parser.add_argument('--calib', default='data1.15/origin/calib/calib0.4723.json',
                        help='Path to calibration file')
    parser.add_argument('--output', default='data1.15/process/selfdataprocess/bkprocess/calib/cameras.json',
                        help='Output cameras.json file')
    parser.add_argument('--verify-images', action='store_true',
                        help='Verify actual image dimensions')
    
    args = parser.parse_args()
    
    # 解析命令行参数并读取标定 JSON
    calib_path = Path(args.calib)
    if not calib_path.exists():
        print(f"Error: Calibration file not found: {args.calib}")
        sys.exit(1)
    
    with open(calib_path, 'r') as f:
        calib_text = f.read()
        calib_data = json.loads(calib_text)
    
    # Extract camera data from calibration, preserving precision
    cameras_data = extract_cameras_from_calib(calib_data, calib_text)
    
    # Get image files
    try:
        image_files = get_image_files(args.images_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if len(image_files) != len(cameras_data):
        print(f"Warning: Found {len(image_files)} images but {len(cameras_data)} cameras in calib")
    
    # Build final cameras.json structure
    output_cameras = []
    
    # Process each camera
    num_cameras = min(len(image_files), len(cameras_data))
    
    for i in range(num_cameras):
        img_file = image_files[i] if i < len(image_files) else None
        cam_data = cameras_data[i] if i < len(cameras_data) else cameras_data[0]
        
        # Get image name
        img_name = img_file.name if img_file else f"{i+1:03d}.png"
        
        # 如果已指定 --verify-images，则逐张校验实际尺寸
        width, height = cam_data['width'], cam_data['height']
        if args.verify_images and img_file and img_file.exists():
            width, height = verify_image_dimensions(img_file, width, height)
        
        camera_entry = {
            "id": i,
            "img_name": img_name,
            "width": width,
            "height": height,
            "position": cam_data['position'],
            "rotation": cam_data['rotation'],
            "fx": float(cam_data['_original_strings']['f']) if '_original_strings' in cam_data else cam_data['fx'],
            "fy": float(cam_data['_original_strings']['f']) if '_original_strings' in cam_data else cam_data['fy'],
            "cx": float(cam_data['_original_strings']['cx']) if '_original_strings' in cam_data else cam_data['cx'],
            "cy": float(cam_data['_original_strings']['cy']) if '_original_strings' in cam_data else cam_data['cy'],
            "distortion": cam_data['distortion']  # 添加畸变参数到输出
        }
        
        output_cameras.append(camera_entry)
    
    # Create output structure
    output_data = {
        "cameras": output_cameras
    }
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write output file with preserved precision
    with open(output_path, 'w') as f:
        # 使用原始字符串替换来保持精度
        output_str = json.dumps(output_data, indent=2, ensure_ascii=False)
        
        # 如果有原始字符串，替换回去以保持精度
        for i, cam_data in enumerate(cameras_data):
            if '_original_strings' in cam_data:
                # 替换fx和fy的值
                import re
                # 注意这里要匹配JSON中的格式
                fx_pattern = f'"fx": {re.escape(str(cam_data["fx"]))}'
                fy_pattern = f'"fy": {re.escape(str(cam_data["fy"]))}'
                cx_pattern = f'"cx": {re.escape(str(cam_data["cx"]))}'
                cy_pattern = f'"cy": {re.escape(str(cam_data["cy"]))}'
                
                # 替换为原始精度的字符串
                output_str = output_str.replace(f'"fx": {cam_data["fx"]}', f'"fx": {cam_data["_original_strings"]["f"]}')
                output_str = output_str.replace(f'"fy": {cam_data["fy"]}', f'"fy": {cam_data["_original_strings"]["f"]}')
                output_str = output_str.replace(f'"cx": {cam_data["cx"]}', f'"cx": {cam_data["_original_strings"]["cx"]}')  
                output_str = output_str.replace(f'"cy": {cam_data["cy"]}', f'"cy": {cam_data["_original_strings"]["cy"]}')
        
        f.write(output_str)
    
    print(f"Successfully converted {len(output_cameras)} cameras to {args.output}")
    print(f"Summary:")
    print(f"  - Images processed: {len(image_files)}")
    print(f"  - Cameras in calib: {len(cameras_data)}")
    print(f"  - Cameras in output: {len(output_cameras)}")
    
    if args.verify_images:
        print(f"  - Image dimensions verified against actual files")


if __name__ == '__main__':
    main()