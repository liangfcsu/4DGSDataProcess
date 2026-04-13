#!/usr/bin/env python3
"""
convert_to_colmap.py
这个脚本将校正后的相机参数和图像转换为COLMAP格式，以用于3D高斯训练。不带k1/k2畸变参数。 
前提就是自己要先去畸变化图像和校正相机参数。 再用这个脚本
"""

import argparse
import json
import os
import sys
import math
import numpy as np
from pathlib import Path
from PIL import Image


def detect_pose_format(cameras):
    """Automatically detect the pose format from camera data
    
    Args:
        cameras: list of camera dictionaries
        
    Returns:
        str: detected format ('ctw', 'wtc', or 'tcw')
    """
    if not cameras or len(cameras) == 0:
        return 'ctw'  # default
        
    # Check the first few cameras for format indicators
    sample_size = min(3, len(cameras))
    
    for cam in cameras[:sample_size]:
        if 'position' in cam and 'rotation' in cam:
            # CTW format: camera position/rotation in world coordinates
            return 'ctw'
        elif 'extrinsic_matrix' in cam:
            # Matrix format, need to determine if it's WTC or TCW
            matrix = np.array(cam['extrinsic_matrix'])
            if matrix.shape == (4, 4):
                # Check translation component to determine format
                # In WTC, translation is usually negative camera position
                # In TCW, translation is camera position in world
                translation = matrix[:3, 3]
                if np.linalg.norm(translation) < 1e-3:
                    # Likely identity or near-identity, assume WTC
                    return 'wtc'
                else:
                    # Non-zero translation, check rotation determinant
                    rotation = matrix[:3, :3]
                    det = np.linalg.det(rotation)
                    if abs(det - 1.0) < 1e-3:
                        return 'wtc'  # Proper rotation matrix
                    else:
                        return 'tcw'
            else:
                return 'wtc'  # default for matrix format
        elif 'quaternion' in cam or 'qw' in cam:
            # Quaternion format, assume WTC (COLMAP style)
            return 'wtc'
        elif 'R' in cam and 'T' in cam:
            # R,T format - need to check dimensions and values
            return 'wtc'  # most common format
            
    # Default fallback
    print("Warning: Could not detect pose format, assuming CTW")
    return 'ctw'


def convert_pose_to_ctw(cam, input_format='auto'):
    """Convert camera pose to CTW format
    
    Args:
        cam: camera dictionary
        input_format: 'ctw', 'wtc', 'tcw', or 'auto' for auto-detection
        
    Returns:
        dict: camera data with 'position' and 'rotation' in CTW format
    """
    if input_format == 'auto':
        input_format = detect_pose_format([cam])
        
    result = cam.copy()
    
    if input_format == 'ctw':
        # Already in CTW format
        if 'position' in cam and 'rotation' in cam:
            return result
        else:
            raise ValueError("CTW format requires 'position' and 'rotation' fields")
            
    elif input_format == 'wtc':
        # Convert from WTC (World-to-Camera) to CTW
        if 'extrinsic_matrix' in cam:
            # 4x4 extrinsic matrix format
            matrix = np.array(cam['extrinsic_matrix'])
            R_wtc = matrix[:3, :3]  # World-to-Camera rotation
            t_wtc = matrix[:3, 3]   # World-to-Camera translation
            
            # Convert to CTW: R_ctw = R_wtc^T, t_ctw = -R_ctw @ t_wtc
            R_ctw = R_wtc.T
            t_ctw = -R_ctw @ t_wtc
            
            # Convert rotation matrix to Euler angles (Z-Y-X order)
            result['position'] = t_ctw.tolist()
            result['rotation'] = rotation_matrix_to_euler_zyx(R_ctw)
            
        elif 'quaternion' in cam or ('qw' in cam and 'qx' in cam):
            # Quaternion + translation format (COLMAP style)
            if 'quaternion' in cam:
                qw, qx, qy, qz = cam['quaternion']
            else:
                qw, qx, qy, qz = cam['qw'], cam['qx'], cam['qy'], cam['qz']
                
            if 'translation' in cam:
                tx, ty, tz = cam['translation']
            else:
                tx, ty, tz = cam['tx'], cam['ty'], cam['tz']
                
            # Convert quaternion to rotation matrix
            R_wtc = quaternion_to_rotation_matrix(qw, qx, qy, qz)
            t_wtc = np.array([tx, ty, tz])
            
            # Convert to CTW
            R_ctw = R_wtc.T
            t_ctw = -R_ctw @ t_wtc
            
            result['position'] = t_ctw.tolist()
            result['rotation'] = rotation_matrix_to_euler_zyx(R_ctw)
            
        elif 'R' in cam and 'T' in cam:
            # Separate R and T matrices
            R_wtc = np.array(cam['R'])
            t_wtc = np.array(cam['T'])
            
            # Convert to CTW
            R_ctw = R_wtc.T
            t_ctw = -R_ctw @ t_wtc
            
            result['position'] = t_ctw.tolist()
            result['rotation'] = rotation_matrix_to_euler_zyx(R_ctw)
            
        else:
            raise ValueError("WTC format not recognized in camera data")
            
    elif input_format == 'tcw':
        # Convert from TCW (similar to CTW but different convention)
        # TCW usually means T is camera position, R is camera orientation
        if 'extrinsic_matrix' in cam:
            matrix = np.array(cam['extrinsic_matrix'])
            # In TCW format, the matrix might be inverted
            # Try to detect and convert appropriately
            R = matrix[:3, :3]
            t = matrix[:3, 3]
            
            # Assume TCW means camera-to-world transform
            result['position'] = t.tolist()
            result['rotation'] = rotation_matrix_to_euler_zyx(R)
            
        else:
            # Assume position and rotation are already in world coordinates
            if 'position' in cam and 'rotation' in cam:
                result['position'] = cam['position']
                result['rotation'] = cam['rotation']
            else:
                raise ValueError("TCW format not recognized in camera data")
                
    else:
        raise ValueError(f"Unknown input format: {input_format}")
        
    return result


def rotation_matrix_to_euler_zyx(R):
    """Convert rotation matrix to Euler angles (Z-Y-X order)
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        dict: {'rx': roll, 'ry': pitch, 'rz': yaw} in radians
    """
    # Extract Euler angles from rotation matrix
    # For Z-Y-X order: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        rx = math.atan2(R[2, 1], R[2, 2])  # roll
        ry = math.atan2(-R[2, 0], sy)      # pitch  
        rz = math.atan2(R[1, 0], R[0, 0])  # yaw
    else:
        rx = math.atan2(-R[1, 2], R[1, 1]) # roll
        ry = math.atan2(-R[2, 0], sy)      # pitch
        rz = 0                             # yaw
        
    return {'rx': rx, 'ry': ry, 'rz': rz}


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix
    
    Args:
        qw, qx, qy, qz: quaternion components
        
    Returns:
        np.array: 3x3 rotation matrix
    """
    # Normalize quaternion
    norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ])
    
    return R


def load_cameras_json(path):
    """Load cameras from JSON file"""
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Handle both formats: {"cameras": [...]} or directly [...]
    if isinstance(data, dict) and 'cameras' in data:
        return data['cameras']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Unsupported cameras.json format")


def euler_to_quaternion_zyx(rx, ry, rz):
    """Convert Euler angles to quaternion using Z-Y-X rotation order (Yaw-Pitch-Roll)
    
    Based on mathematical derivation:
    - rx: Roll (φ) - rotation around X-axis
    - ry: Pitch (θ) - rotation around Y-axis  
    - rz: Yaw (ψ) - rotation around Z-axis
    
    Rotation order: Z(ψ) * Y(θ) * X(φ)
    
    Final quaternion formula:
    w = cos(ψ/2)cos(θ/2)cos(φ/2) + sin(ψ/2)sin(θ/2)sin(φ/2)
    x = cos(ψ/2)cos(θ/2)sin(φ/2) - sin(ψ/2)sin(θ/2)cos(φ/2)
    y = cos(ψ/2)sin(θ/2)cos(φ/2) + sin(ψ/2)cos(θ/2)sin(φ/2)
    z = sin(ψ/2)cos(θ/2)cos(φ/2) - cos(ψ/2)sin(θ/2)sin(φ/2)
    
    Args:
        rx: Roll angle in radians (rotation around X-axis)
        ry: Pitch angle in radians (rotation around Y-axis)
        rz: Yaw angle in radians (rotation around Z-axis)
        
    Returns:
        tuple: (w, x, y, z) quaternion components
    """
    # Half angles for trigonometric efficiency
    phi_half = rx * 0.5    # Roll/2
    theta_half = ry * 0.5  # Pitch/2
    psi_half = rz * 0.5    # Yaw/2
    
    # Precompute trigonometric values
    c_phi = math.cos(phi_half)
    s_phi = math.sin(phi_half)
    c_theta = math.cos(theta_half)
    s_theta = math.sin(theta_half)
    c_psi = math.cos(psi_half)
    s_psi = math.sin(psi_half)
    
    # Apply derived quaternion formula for Z-Y-X rotation order
    w = c_psi * c_theta * c_phi + s_psi * s_theta * s_phi
    x = c_psi * c_theta * s_phi - s_psi * s_theta * c_phi
    y = c_psi * s_theta * c_phi + s_psi * c_theta * s_phi
    z = s_psi * c_theta * c_phi - c_psi * s_theta * s_phi
    
    return w, x, y, z


def rodrigues_to_quaternion(rx, ry, rz):
    """
    将Rodrigues旋转向量转换为四元数
    输入：Rodrigues向量 [rx, ry, rz] （弧度）
    输出：四元数 [qw, qx, qy, qz]
    
    Rodrigues向量是轴角表示法的紧凑形式：
    - 向量方向表示旋转轴
    - 向量长度表示旋转角度
    """
    # 构建Rodrigues向量
    rvec = np.array([rx, ry, rz])
    
    # 计算旋转角度（向量的模长）
    angle = np.linalg.norm(rvec)
    
    if angle < 1e-10:
        # 角度很小，返回单位四元数
        return 1.0, 0.0, 0.0, 0.0
    
    # 归一化得到旋转轴
    axis = rvec / angle
    
    # 轴角转四元数
    half_angle = angle * 0.5
    c = math.cos(half_angle)
    s = math.sin(half_angle)
    
    # COLMAP格式需要 [w, x, y, z] 顺序
    qw = c
    qx = s * axis[0]
    qy = s * axis[1]
    qz = s * axis[2]
    
    return qw, qx, qy, qz


def single_axis_quaternion(angle, axis):
    """Generate quaternion for single axis rotation
    
    For rotation θ around unit vector u=(ux,uy,uz):
    q = cos(θ/2) + sin(θ/2)(ux*i + uy*j + uz*k)
    
    Args:
        angle: rotation angle in radians
        axis: 'x', 'y', or 'z' for rotation axis
        
    Returns:
        tuple: (w, x, y, z) quaternion components
    """
    half_angle = angle * 0.5
    c = math.cos(half_angle)
    s = math.sin(half_angle)
    
    if axis.lower() == 'x':
        return c, s, 0.0, 0.0  # [cos(φ/2), sin(φ/2), 0, 0]
    elif axis.lower() == 'y':
        return c, 0.0, s, 0.0  # [cos(θ/2), 0, sin(θ/2), 0]
    elif axis.lower() == 'z':
        return c, 0.0, 0.0, s  # [cos(ψ/2), 0, 0, sin(ψ/2)]
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")


def quaternion_multiply(q1, q2):
    """Multiply two quaternions: q1 ⊗ q2
    
    Quaternion multiplication formula:
    q1 ⊗ q2 = (w1*w2 - x1*x2 - y1*y2 - z1*z2) +
               (w1*x2 + x1*w2 + y1*z2 - z1*y2)i +
               (w1*y2 - x1*z2 + y1*w2 + z1*x2)j +
               (w1*z2 + x1*y2 - y1*x2 + z1*w2)k
    
    Args:
        q1: tuple (w1, x1, y1, z1)
        q2: tuple (w2, x2, y2, z2)
        
    Returns:
        tuple: (w, x, y, z) resulting quaternion
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return w, x, y, z


def normalize_quaternion(q):
    """Normalize quaternion to unit length
    
    Args:
        q: tuple (w, x, y, z)
        
    Returns:
        tuple: normalized (w, x, y, z)
    """
    w, x, y, z = q
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    
    if norm < 1e-12:
        # Return identity quaternion for zero-length input
        return 1.0, 0.0, 0.0, 0.0
        
    return w/norm, x/norm, y/norm, z/norm


def verify_quaternion_properties(q):
    """Verify quaternion properties for debugging
    
    Args:
        q: tuple (w, x, y, z)
        
    Returns:
        dict: verification results
    """
    w, x, y, z = q
    norm_squared = w*w + x*x + y*y + z*z
    norm = math.sqrt(norm_squared)
    
    return {
        'norm': norm,
        'is_unit': abs(norm - 1.0) < 1e-6,
        'norm_deviation': abs(norm - 1.0),
        'components': {'w': w, 'x': x, 'y': y, 'z': z}
    }


def convert_to_wtc_transform(position, rotation_euler, method='direct'):
    """Convert camera pose to WTC (World-to-Camera) transform for COLMAP
    
    This function now assumes input is always in CTW format and converts to WTC.
    Use convert_pose_to_ctw() first if your input is in different format.
    
    Args:
        position: camera position in world coordinates [x, y, z] (CTW format)
        rotation_euler: dict with 'rx', 'ry', 'rz' Euler angles in radians (CTW format)
        method: 'direct' for mathematical formula, 'matrix' for rotation matrix method
        
    Returns:
        tuple: (qw, qx, qy, qz, tx, ty, tz) - WTC quaternion and translation
    """
    # CTW: Camera center in world coordinates
    # WTC: World point in camera coordinates
    
    rx, ry, rz = rotation_euler['rx'], rotation_euler['ry'], rotation_euler['rz']
    
    if method == 'direct':
        # Method 1: Direct mathematical formula (recommended)
        # Convert CTW Euler angles to CTW quaternion
        qw_ctw, qx_ctw, qy_ctw, qz_ctw = euler_to_quaternion_zyx(rx, ry, rz)
        
        # Verify quaternion properties
        ctw_verification = verify_quaternion_properties((qw_ctw, qx_ctw, qy_ctw, qz_ctw))
        if not ctw_verification['is_unit']:
            print(f"Warning: CTW quaternion norm deviation: {ctw_verification['norm_deviation']}")
            qw_ctw, qx_ctw, qy_ctw, qz_ctw = normalize_quaternion((qw_ctw, qx_ctw, qy_ctw, qz_ctw))
        
        # Convert CTW to WTC: q_wtc = conjugate(q_ctw)
        qw_wtc = qw_ctw
        qx_wtc = -qx_ctw
        qy_wtc = -qy_ctw
        qz_wtc = -qz_ctw
        
        # Convert position using quaternion rotation
        # t_wtc = -R_wtc @ t_ctw where R_wtc is derived from q_wtc
        # For efficiency, use the formula: t_wtc = -conjugate(q_ctw) * t_ctw * q_ctw
        px, py, pz = position
        
        # Manual quaternion-vector rotation: q_conj * [0,px,py,pz] * q
        # First: q_conj * [0,px,py,pz]
        temp_w = -(qx_ctw*px + qy_ctw*py + qz_ctw*pz)
        temp_x = qw_ctw*px - qz_ctw*py + qy_ctw*pz
        temp_y = qw_ctw*py + qz_ctw*px - qx_ctw*pz
        temp_z = qw_ctw*pz - qy_ctw*px + qx_ctw*py
        
        # Second: [temp] * q
        tx = -(temp_w*qx_ctw + temp_x*qw_ctw + temp_y*qz_ctw - temp_z*qy_ctw)
        ty = -(temp_w*qy_ctw - temp_x*qz_ctw + temp_y*qw_ctw + temp_z*qx_ctw)
        tz = -(temp_w*qz_ctw + temp_x*qy_ctw - temp_y*qx_ctw + temp_z*qw_ctw)
        
        return qw_wtc, qx_wtc, qy_wtc, qz_wtc, tx, ty, tz
        
    else:
        # Method 2: Traditional rotation matrix approach (for verification)
        # Build CTW rotation matrix using Z-Y-X order
        Rz = np.array([[math.cos(rz), -math.sin(rz), 0], 
                       [math.sin(rz), math.cos(rz), 0], 
                       [0, 0, 1]])
        Ry = np.array([[math.cos(ry), 0, math.sin(ry)], 
                       [0, 1, 0], 
                       [-math.sin(ry), 0, math.cos(ry)]])
        Rx = np.array([[1, 0, 0], 
                       [0, math.cos(rx), -math.sin(rx)], 
                       [0, math.sin(rx), math.cos(rx)]])
        
        R_ctw = Rz @ Ry @ Rx  # Camera-to-World rotation
        
        # Convert CTW to WTC
        R_wtc = R_ctw.T  # World-to-Camera rotation
        t_wtc = -R_wtc @ np.array(position)  # World-to-Camera translation
        
        # Extract quaternion from rotation matrix (Shepperd's method)
        trace = R_wtc[0, 0] + R_wtc[1, 1] + R_wtc[2, 2]
        
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R_wtc[2, 1] - R_wtc[1, 2]) / s
            qy = (R_wtc[0, 2] - R_wtc[2, 0]) / s
            qz = (R_wtc[1, 0] - R_wtc[0, 1]) / s
        elif R_wtc[0, 0] > R_wtc[1, 1] and R_wtc[0, 0] > R_wtc[2, 2]:
            s = math.sqrt(1.0 + R_wtc[0, 0] - R_wtc[1, 1] - R_wtc[2, 2]) * 2  # s = 4 * qx
            qw = (R_wtc[2, 1] - R_wtc[1, 2]) / s
            qx = 0.25 * s
            qy = (R_wtc[0, 1] + R_wtc[1, 0]) / s
            qz = (R_wtc[0, 2] + R_wtc[2, 0]) / s
        elif R_wtc[1, 1] > R_wtc[2, 2]:
            s = math.sqrt(1.0 + R_wtc[1, 1] - R_wtc[0, 0] - R_wtc[2, 2]) * 2  # s = 4 * qy
            qw = (R_wtc[0, 2] - R_wtc[2, 0]) / s
            qx = (R_wtc[0, 1] + R_wtc[1, 0]) / s
            qy = 0.25 * s
            qz = (R_wtc[1, 2] + R_wtc[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R_wtc[2, 2] - R_wtc[0, 0] - R_wtc[1, 1]) * 2  # s = 4 * qz
            qw = (R_wtc[1, 0] - R_wtc[0, 1]) / s
            qx = (R_wtc[0, 2] + R_wtc[2, 0]) / s
            qy = (R_wtc[1, 2] + R_wtc[2, 1]) / s
            qz = 0.25 * s
        
        return qw, qx, qy, qz, t_wtc[0], t_wtc[1], t_wtc[2]
    
    return qw, qx, qy, qz, t_wtc[0], t_wtc[1], t_wtc[2]


def write_cameras_txt(cameras, output_path):
    """Write cameras.txt in COLMAP format"""
    with open(output_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: {}\n".format(len(cameras)))
        
        for i, cam in enumerate(cameras):
            camera_id = i + 1  # COLMAP uses 1-based indexing
            model = "PINHOLE"
            width = cam['width']
            height = cam['height']
            fx = cam['fx']
            fy = cam['fy']
            cx = cam['cx']
            cy = cam['cy']
            
            f.write(f"{camera_id} {model} {width} {height} {fx} {fy} {cx} {cy}\n")
    
    print(f"Written {len(cameras)} cameras to {output_path}")


def write_images_txt(cameras, images_dir, output_path, method='rodrigues', debug=False, input_format='wtc'):
    """Write images.txt in COLMAP format using Rodrigues vector conversion
    
    Args:
        cameras: list of camera dictionaries
        images_dir: directory containing images
        output_path: output file path
        method: quaternion conversion method ('rodrigues' or 'legacy')
        debug: enable debug output
        input_format: camera pose format (默认 'wtc' - World-to-Camera)
    """
    images_path = Path(images_dir)
    
    print(f"使用 {method} 方法和 {input_format} 格式进行四元数转换...")
    
    with open(output_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write("# Number of images: {}, mean observations per image: 0\n".format(len(cameras)))
        
        for i, cam in enumerate(cameras):
            image_id = i + 1  # COLMAP uses 1-based indexing
            camera_id = i + 1  # Each camera corresponds to one image
            
            # Get image name
            if 'img_name' in cam:
                image_name = cam['img_name']
            elif 'name' in cam:
                image_name = cam['name']
            else:
                image_name = f"{i+1:03d}.png"
            
            # Check if image exists
            img_path = images_path / image_name
            if not img_path.exists():
                # Try different extensions
                for ext in ['.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                    alt_path = images_path / (img_path.stem + ext)
                    if alt_path.exists():
                        image_name = alt_path.name
                        break
                else:
                    print(f"Warning: Image not found: {image_name}")
            
            if method == 'rodrigues':
                # 使用 Rodrigues 向量转换方法
                if 'rotation' in cam and isinstance(cam['rotation'], dict):
                    # 从相机数据中提取 Rodrigues 向量（假设已经是 wtc 格式）
                    rx = cam['rotation']['rx']
                    ry = cam['rotation']['ry'] 
                    rz = cam['rotation']['rz']
                    
                    # 平移向量
                    if 'position' in cam:
                        tx, ty, tz = cam['position']
                    elif 'translation' in cam:
                        if isinstance(cam['translation'], dict):
                            tx = cam['translation']['x']
                            ty = cam['translation']['y']
                            tz = cam['translation']['z']
                        else:
                            tx, ty, tz = cam['translation']
                    else:
                        tx, ty, tz = 0.0, 0.0, 0.0
                    
                    # 使用 Rodrigues 向量转换为四元数
                    qw, qx, qy, qz = rodrigues_to_quaternion(rx, ry, rz)
                    
                    if debug and i < 3:
                        print(f"Camera {image_id} Rodrigues 转换:")
                        print(f"  输入 rx={rx:.6f}, ry={ry:.6f}, rz={rz:.6f}")
                        print(f"  四元数: qw={qw:.6f}, qx={qx:.6f}, qy={qy:.6f}, qz={qz:.6f}")
                        print(f"  平移: tx={tx:.6f}, ty={ty:.6f}, tz={tz:.6f}")
                        
                        # 验证四元数归一化
                        norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
                        print(f"  四元数模长: {norm:.6f} (应该接近 1.0)")
                
                else:
                    print(f"Error: Camera {image_id} missing rotation data for Rodrigues conversion")
                    continue
                    
            else:
                # 使用原有的转换方法（向后兼容）
                try:
                    cam_ctw = convert_pose_to_ctw(cam, input_format)
                    position = cam_ctw['position']
                    rotation = cam_ctw['rotation']
                    qw, qx, qy, qz, tx, ty, tz = convert_to_wtc_transform(position, rotation, method='direct')
                except Exception as e:
                    print(f"Error converting camera {image_id} pose: {e}")
                    print(f"Skipping camera {image_id}")
                    continue
            
            # Verify quaternion properties
            norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
            if abs(norm - 1.0) > 1e-6:
                print(f"Warning: Quaternion norm deviation for image {image_id}: {abs(norm - 1.0):.8f}")
                qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
            
            # Write image line
            f.write(f"{image_id} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} {tx:.10f} {ty:.10f} {tz:.10f} {camera_id} {image_name}\n")
            # Write empty points line (no feature points yet)
            f.write("\n")
    
    print(f"Written {len(cameras)} images to {output_path}")


def write_points3d_txt(output_path):
    """Write empty points3D.txt in COLMAP format"""
    with open(output_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0, mean track length: 0\n")
    
    print(f"Written empty points3D.txt to {output_path}")


def verify_images_exist(cameras, images_dir):
    """Verify that all referenced images exist"""
    images_path = Path(images_dir)
    missing_images = []
    found_images = []
    
    for i, cam in enumerate(cameras):
        if 'img_name' in cam:
            image_name = cam['img_name']
        else:
            image_name = f"{i+1:03d}.png"
        
        img_path = images_path / image_name
        if img_path.exists():
            found_images.append(image_name)
        else:
            # Try different extensions
            found = False
            for ext in ['.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                alt_path = images_path / (img_path.stem + ext)
                if alt_path.exists():
                    found_images.append(alt_path.name)
                    found = True
                    break
            if not found:
                missing_images.append(image_name)
    
    return found_images, missing_images


def main():
    parser = argparse.ArgumentParser(description='Convert cameras.json and images to COLMAP format')
    parser.add_argument('--cameras-json', default='data1.15/process/agisoft/undiscalib/cameras.json',
                        help='Path to cameras.json file with corrected parameters')
    parser.add_argument('--images-dir', default='data1.15/process/agisoft/undistorted',
                        help='Directory containing corrected images')
    parser.add_argument('--output-dir', default='data1.15/process/agisoft/sparse/0',
                        help='Output directory for COLMAP files')
    parser.add_argument('--verify-images', action='store_true',
                        help='Verify that all images exist before processing')
    parser.add_argument('--quaternion-method', choices=['rodrigues', 'direct', 'matrix'], default='rodrigues',
                        help='Method for quaternion conversion: rodrigues (Rodrigues vector), direct (mathematical) or matrix (traditional)')
    parser.add_argument('--debug-quaternion', action='store_true',
                        help='Enable detailed quaternion conversion debugging output')
    parser.add_argument('--pose-format', choices=['auto', 'ctw', 'wtc', 'tcw'], default='wtc',
                        help='Input pose format: auto (detect), ctw (camera-to-world), wtc (world-to-camera), tcw (camera-to-world variant)')
    
    args = parser.parse_args()
    
    # Load cameras data
    cameras_path = Path(args.cameras_json)
    if not cameras_path.exists():
        print(f"Error: Cameras file not found: {args.cameras_json}")
        sys.exit(1)
    
    try:
        cameras = load_cameras_json(cameras_path)
        print(f"Loaded {len(cameras)} cameras from {args.cameras_json}")
    except Exception as e:
        print(f"Error loading cameras: {e}")
        sys.exit(1)
    
    # Verify images directory exists
    images_path = Path(args.images_dir)
    if not images_path.exists():
        print(f"Error: Images directory not found: {args.images_dir}")
        sys.exit(1)
    
    # Verify images if requested
    if args.verify_images:
        found_images, missing_images = verify_images_exist(cameras, args.images_dir)
        print(f"Found {len(found_images)} images, {len(missing_images)} missing")
        if missing_images:
            print("Missing images:")
            for img in missing_images[:5]:  # Show first 5
                print(f"  - {img}")
            if len(missing_images) > 5:
                print(f"  ... and {len(missing_images) - 5} more")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write COLMAP files
    cameras_txt = output_path / 'cameras.txt'
    images_txt = output_path / 'images.txt'
    points3d_txt = output_path / 'points3D.txt'
    
    write_cameras_txt(cameras, cameras_txt)
    write_images_txt(cameras, args.images_dir, images_txt, 
                     method=args.quaternion_method, debug=args.debug_quaternion,
                     input_format=args.pose_format)
    write_points3d_txt(points3d_txt)
    
    print(f"\nCOLMAP sparse reconstruction files written to {output_path}")
    print(f"Files created:")
    print(f"  - cameras.txt ({len(cameras)} cameras)")
    print(f"  - images.txt ({len(cameras)} images)")
    print(f"  - points3D.txt (empty, ready for reconstruction)")
    
    print(f"\nNext steps:")
    print(f"1. Use these files with 3D Gaussian Splatting training")
    print(f"2. Or run COLMAP feature extraction and matching to populate points3D.txt")
    print(f"3. Directory structure ready for: colmap feature_extractor --database_path database.db --image_path {args.images_dir}")


def test_quaternion_conversion():
    """Test and verify the mathematical quaternion conversion"""
    import random
    
    print("Testing Euler to Quaternion conversion...")
    print("=" * 60)
    
    # Test cases: [rx, ry, rz] in radians
    test_cases = [
        [0.0, 0.0, 0.0],  # Identity
        [math.pi/4, 0.0, 0.0],  # 45° around X
        [0.0, math.pi/4, 0.0],  # 45° around Y  
        [0.0, 0.0, math.pi/4],  # 45° around Z
        [0.0777, -0.4426, 0.0147],  # Your example values
        [math.pi/6, math.pi/3, math.pi/4],  # Mixed angles
    ]
    
    for i, (rx, ry, rz) in enumerate(test_cases):
        print(f"Test case {i+1}: rx={rx:.4f}, ry={ry:.4f}, rz={rz:.4f}")
        print(f"  Degrees: rx={math.degrees(rx):.2f}°, ry={math.degrees(ry):.2f}°, rz={math.degrees(rz):.2f}°")
        
        # Test mathematical formula
        qw, qx, qy, qz = euler_to_quaternion_zyx(rx, ry, rz)
        norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        
        print(f"  Mathematical: w={qw:.6f}, x={qx:.6f}, y={qy:.6f}, z={qz:.6f}")
        print(f"  Norm: {norm:.6f} (should be 1.0)")
        
        # Test step-by-step multiplication
        qx_single = single_axis_quaternion(rx, 'x')
        qy_single = single_axis_quaternion(ry, 'y')
        qz_single = single_axis_quaternion(rz, 'z')
        
        # Z * Y * X order
        qyz = quaternion_multiply(qy_single, qx_single)
        q_step = quaternion_multiply(qz_single, qyz)
        norm_step = math.sqrt(q_step[0]**2 + q_step[1]**2 + q_step[2]**2 + q_step[3]**2)
        
        print(f"  Step-by-step: w={q_step[0]:.6f}, x={q_step[1]:.6f}, y={q_step[2]:.6f}, z={q_step[3]:.6f}")
        print(f"  Norm: {norm_step:.6f}")
        
        # Compare methods
        diff = math.sqrt((qw-q_step[0])**2 + (qx-q_step[1])**2 + (qy-q_step[2])**2 + (qz-q_step[3])**2)
        print(f"  Difference: {diff:.8f} (should be ~0)")
        print()
        
        if diff > 1e-6:
            print("  WARNING: Large difference detected!")
    
    print("Testing complete.")


if __name__ == '__main__':
    # Check if running tests
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_quaternion_conversion()
        sys.exit(0)
        
    main()