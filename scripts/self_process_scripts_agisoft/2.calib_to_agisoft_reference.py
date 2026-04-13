#!/usr/bin/env python3
"""
将 calib.json (libCalib 格式) 转换为 Agisoft Metashape XML 格式。
"""
import argparse
import json
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

def extract_camera_params(camera):
    """从 camera 对象中提取内参和外参"""
    model_data = camera["model"]["ptr_wrapper"]["data"]
    params = model_data["parameters"]
    
    # 内参
    f = params["f"]["val"]
    cx = params["cx"]["val"]
    cy = params["cy"]["val"]
    k1 = params.get("k1", {}).get("val", 0.0)
    k2 = params.get("k2", {}).get("val", 0.0)
    k3 = params.get("k3", {}).get("val", 0.0)
    p1 = params.get("p1", {}).get("val", 0.0)
    p2 = params.get("p2", {}).get("val", 0.0)
    
    # 图像尺寸
    image_size = model_data["CameraModelCRT"]["CameraModelBase"]["imageSize"]
    width = image_size["width"]
    height = image_size["height"]
    
    # 外参（旋转和平移）
    transform = camera["transform"]
    rx = transform["rotation"]["rx"]
    ry = transform["rotation"]["ry"] 
    rz = transform["rotation"]["rz"]
    tx = transform["translation"]["x"]
    ty = transform["translation"]["y"]
    tz = transform["translation"]["z"]
    
    return {
        "f": f, "cx": cx, "cy": cy, "k1": k1, "k2": k2, "k3": k3, "p1": p1, "p2": p2,
        "width": width, "height": height,
        "rx": rx, "ry": ry, "rz": rz, "tx": tx, "ty": ty, "tz": tz
    }

def rodrigues_to_rotation_matrix(rvec):
    """将 Rodrigues 向量转换为旋转矩阵"""
    theta = np.linalg.norm(rvec)
    
    if theta < 1e-8:
        # 角度很小，使用一阶近似
        return np.eye(3) + skew_symmetric(rvec)
    
    # 归一化轴
    k = rvec / theta
    
    # Rodrigues 公式
    K = skew_symmetric(k)
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    
    return R

def skew_symmetric(v):
    """构建反对称矩阵"""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def rodrigues_to_transform_matrix(rx, ry, rz, tx, ty, tz):
    """从 Rodrigues 向量和平移向量构建4x4变换矩阵 (c2w格式)"""
    # Rodrigues 向量
    rvec = np.array([rx, ry, rz])
    
    # 转换为旋转矩阵 (w2c)
    rotation_matrix_w2c = rodrigues_to_rotation_matrix(rvec)
    
    # 转换为 c2w
    rotation_matrix_c2w = rotation_matrix_w2c.T
    
    # 构建4x4变换矩阵 (camera-to-world)
    tw2c = np.array([tx, ty, tz])
    tc2w = -rotation_matrix_c2w @ tw2c
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix_c2w
    transform_matrix[:3, 3] = tc2w
    
    return transform_matrix

def main():
    parser = argparse.ArgumentParser(description="将 calib.json 转换为 Agisoft camera.xml 格式")
    parser.add_argument("--input", type=Path, default="data1.15/process/agisoft/undistorted/calib_undistorted.json", help="输入的 calib.json 文件路径")
    parser.add_argument("--output", type=Path, default="data1.15/process/agisoft/camera_agisoft_generated.xml", help="输出的 XML 文件路径")
    args = parser.parse_args()

    # 加载 calib.json
    try:
        calib_data = json.loads(args.input.read_text())
        cameras = calib_data["Calibration"]["cameras"]
    except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法读取或解析 {args.input}: {e}")
        return

    if not cameras:
        print("错误: 未找到相机数据")
        return

    # 构建XML结构
    doc = ET.Element("document", version="2.0.0")
    chunk = ET.SubElement(doc, "chunk", label="Chunk 1", enabled="true")
    
    # 传感器配置 - 为每个相机创建独立的sensor
    sensors = ET.SubElement(chunk, "sensors", next_id=str(len(cameras)))
    
    for i, camera in enumerate(cameras):
        params = extract_camera_params(camera)
        
        # 创建独立的sensor
        sensor = ET.SubElement(sensors, "sensor", id=str(i), label=f"sensor_{i+1:03d}", type="frame")
        
        w, h = params["width"], params["height"]
        ET.SubElement(sensor, "resolution", width=str(w), height=str(h))
        
        # 相机标定参数
        calib = ET.SubElement(sensor, "calibration", type="frame", class_="adjusted")
        ET.SubElement(calib, "resolution", width=str(w), height=str(h))
        
        # Agisoft中cx、cy是相对于图像中心的偏移
        cx_agisoft = params["cx"] - w / 2
        cy_agisoft = params["cy"] - h / 2
        
        ET.SubElement(calib, "f").text = str(params["f"])
        ET.SubElement(calib, "cx").text = str(cx_agisoft)
        ET.SubElement(calib, "cy").text = str(cy_agisoft)
        ET.SubElement(calib, "k1").text = str(params["k1"])
        ET.SubElement(calib, "k2").text = str(params["k2"])
        ET.SubElement(calib, "k3").text = str(params["k3"])
        ET.SubElement(calib, "p1").text = str(params["p1"])
        ET.SubElement(calib, "p2").text = str(params["p2"])

    # 相机列表
    cameras_node = ET.SubElement(chunk, "cameras", next_id=str(len(cameras)))
    
    for i, camera in enumerate(cameras):
        params = extract_camera_params(camera)
        
        # 构建变换矩阵 - 使用 Rodrigues 向量
        transform_matrix = rodrigues_to_transform_matrix(
            params["rx"], params["ry"], params["rz"],
            params["tx"], params["ty"], params["tz"]
        )
        
        # 添加相机节点，使用对应的sensor_id
        camera_node = ET.SubElement(cameras_node, "camera", 
                                   id=str(i), sensor_id=str(i), label=f"{i+1:03d}")
        ET.SubElement(camera_node, "transform").text = " ".join(map(str, transform_matrix.flatten()))

    # 写入文件
    xml_str = minidom.parseString(ET.tostring(doc)).toprettyxml(indent="  ")
    args.output.write_text(xml_str)
    print(f"成功将 {len(cameras)} 个相机写入到 {args.output}")

if __name__ == "__main__":
    main()
