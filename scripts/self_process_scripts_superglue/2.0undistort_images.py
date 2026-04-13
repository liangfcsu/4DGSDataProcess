"""
用 ChatGPT 优化后的去畸变方法处理图像，输出 images_undistorted/（保持原始尺寸），并生成匹配的 cameras.json（更新内参 fx/fy/cx/cy）。

去畸变方法：使用 undistort_colmap_equiv 函数，通过计算新的内参保证覆盖四角射线，然后构建映射表进行去畸变。

用途：消除图像畸变，生成与新内参一致的图像。
对图像/标定的作用：修正图像的像素坐标；更新相机内参以匹配去畸变后的像素系。
点云作用：关键——三角化必须用去畸变图像 + 对应的 cameras.json，否则投影错位。
"""

import json
import cv2
import numpy as np
from pathlib import Path
import shutil
import re
import math


def distort_points(x, y, k1, k2, k3, p1, p2):
    """畸变点计算"""
    r2 = x**2 + y**2
    radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
    x_dist = x*radial + 2*p1*x*y + p2*(r2 + 2*x**2)
    y_dist = y*radial + p1*(r2 + 2*y**2) + 2*p2*x*y
    return x_dist, y_dist


def undistort_colmap_equiv(img, cam_params, iter_count=5):
    """使用 ChatGPT 优化后的去畸变方法"""
    H, W = cam_params['height'], cam_params['width']
    fx, fy, cx, cy = cam_params['fx'], cam_params['fy'], cam_params['cx'], cam_params['cy']
    k1, k2, k3, p1, p2 = cam_params['k1'], cam_params['k2'], cam_params['k3'], cam_params['p1'], cam_params['p2']

    # Step1: 计算 newK 保证覆盖四角射线
    corners = np.array([[0,0],[W-1,0],[0,H-1],[W-1,H-1]])
    max_x, max_y = 0,0
    for u,v in corners:
        x = (u - cx)/fx
        y = (v - cy)/fy
        x_d, y_d = distort_points(x,y,k1,k2,k3,p1,p2)
        max_x = max(max_x, abs(x_d))
        max_y = max(max_y, abs(y_d))
    fx_new = (W/2)/max_x
    fy_new = (H/2)/max_y
    cx_new = W/2
    cy_new = H/2
    newK = np.array([[fx_new,0,cx_new],[0,fy_new,cy_new],[0,0,1]])

    # Step2: 构建映射表 (向量化)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x_n = (u - cx_new)/fx_new
    y_n = (v - cy_new)/fy_new

    x, y = x_n.copy(), y_n.copy()
    for _ in range(iter_count):
        x_d, y_d = distort_points(x, y, k1, k2, k3, p1, p2)
        x = x_n - (x_d - x)
        y = y_n - (y_d - y)

    map_x = x*fx + cx
    map_y = y*fy + cy

    undistorted = cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    return undistorted, newK


def load_calib(path):
    """Load calibration data and preserve original text for precision extraction"""
    with open(path, 'r') as f:
        calib_text = f.read()
    calib_data = json.loads(calib_text)
    return calib_data, calib_text


def extract_params(camera, calib_text, camera_index):
    """从camera条目提取内参和畸变参数，使用高精度字符串匹配，返回K, distCoeffs, image_size, rotation, translation, original_strings"""
    
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
    fy = fx * ar  # Calculate fy from f and aspect ratio
    cx = float(cx_str)
    cy = float(cy_str)

    # 畸变参数处理
    k1 = float(k1_values[camera_index]) if camera_index < len(k1_values) else 0.0
    k2 = float(k2_values[camera_index]) if camera_index < len(k2_values) else 0.0
    k3 = float(k3_values[camera_index]) if camera_index < len(k3_values) else 0.0
    p1 = float(p1_values[camera_index]) if camera_index < len(p1_values) else 0.0
    p2 = float(p2_values[camera_index]) if camera_index < len(p2_values) else 0.0
    
    # 尝试提取其他畸变参数，如果不存在则为0
    def v(name):
        param_values = re.findall(f'"{name}":\\s*{{\\s*"val":\\s*([0-9.-]+)', calib_text)
        return float(param_values[camera_index]) if camera_index < len(param_values) else 0.0
    
    k4 = v('k4')
    k5 = v('k5')
    k6 = v('k6')

    dist = [k1, k2, p1, p2, k3]
    if any([k4, k5, k6]):
        dist += [k4, k5, k6]

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    # rotation & translation from camera['transform']
    transform = camera.get('transform', {})
    rotation = transform.get('rotation')
    translation = transform.get('translation')

    # 保存原始字符串以便后续使用
    original_strings = {
        'f': f_str,
        'cx': cx_str,
        'cy': cy_str,
        'ar': ar_str
    }

    return K, np.array(dist, dtype=np.float64), (image_size['width'], image_size['height']), rotation, translation, original_strings


def undistort_images(calib_path='my_dataset/calib.json', images_dir='my_dataset/origin_images', out_dir='my_dataset/images', write_cameras=True, alpha=0.0, tool_dir_override=None, compat_name='cameras.json'):
    calib, calib_text = load_calib(calib_path)
    cameras = calib['Calibration']['cameras']
    

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    updated_cameras = []
    # 同步记录原始（标定）相机条目，便于输出 calib 格式的 cameras.json
    original_cameras = []
    processed = 0
    for i, cam in enumerate(cameras):
        name = f"{i+1:03d}.png"
        src = Path(images_dir) / name
        if not src.exists():
            src = Path(images_dir) / f"{i+1:03d}.jpg"
            if not src.exists():
                print(f"跳过：未找到图片 {name}")
                continue

        img = cv2.imread(str(src))
        if img is None:
            print(f"无法读取图片: {src}")
            continue

        K, dist, calib_size, rotation, translation, original_strings = extract_params(cam, calib_text, i)

        # 记录原始相机内参（使用高精度原始字符串）
        try:
            fx0 = float(original_strings['f'])
            fy0 = float(original_strings['f'])  # fy = f * ar, 但这里记录原始f值
            cx0 = float(original_strings['cx'])
            cy0 = float(original_strings['cy'])
        except Exception:
            fx0 = fy0 = cx0 = cy0 = None

        pos0 = None
        if translation:
            if isinstance(translation, dict):
                pos0 = [translation.get('x', 0), translation.get('y', 0), translation.get('z', 0)]
            else:
                try:
                    pos0 = list(translation)
                except Exception:
                    pos0 = None

        rot0 = None
        if rotation:
            if isinstance(rotation, dict):
                rx = rotation.get('rx'); ry = rotation.get('ry'); rz = rotation.get('rz')
                rot0 = {'rx': rx, 'ry': ry, 'rz': rz}
            else:
                rot0 = rotation

        orig_entry = {
            'id': i,
            'img_name': name,
            'width': calib_size[0],
            'height': calib_size[1],
            'position': pos0,
            'rotation': rot0,
            'fx': fx0,
            'fy': fy0,
            'cx': cx0,
            'cy': cy0,
            # 保存原始字符串以便后续精度保持
            '_original_strings': original_strings
        }
        original_cameras.append(orig_entry)

        h, w = img.shape[:2]
        if (w, h) != calib_size:
            # 使用实际读取的尺寸作为输入尺寸
            calib_size = (w, h)

        # 构建相机参数字典供 undistort_colmap_equiv 使用
        cam_params = {
            'width': w,
            'height': h,
            'fx': K[0,0],
            'fy': K[1,1],
            'cx': K[0,2],
            'cy': K[1,2],
            'k1': dist[0] if len(dist) > 0 else 0.0,
            'k2': dist[1] if len(dist) > 1 else 0.0,
            'k3': dist[4] if len(dist) > 4 else 0.0,
            'p1': dist[2] if len(dist) > 2 else 0.0,
            'p2': dist[3] if len(dist) > 3 else 0.0
        }

        # 使用新的去畸变方法
        final_img, newK = undistort_colmap_equiv(img, cam_params)

        # 新的方法总是保持原始尺寸
        final_w, final_h = w, h
        K_for_json = newK

        # 检查内参变化程度
        fx_change = abs(newK[0,0] - K[0,0]) / K[0,0] if K[0,0] != 0 else 0
        fy_change = abs(newK[1,1] - K[1,1]) / K[1,1] if K[1,1] != 0 else 0

        if fx_change > 0.05 or fy_change > 0.05:  # 如果内参变化超过5%
            print(f"警告：相机{i+1} 内参变化较大 - fx: {fx_change:.1%}, fy: {fy_change:.1%}")
            print(f"   原始: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
            print(f"   去畸变后: fx={newK[0,0]:.1f}, fy={newK[1,1]:.1f}")

        out_file = out_path / name
        # 保存为 PNG（保持无损），使用 imencode->tofile to support Unicode paths
        cv2.imencode('.png', final_img)[1].tofile(str(out_file))

        # 构建相机条目（假定 rotation/translation 已在原 calib 中）
        fx = float(K_for_json[0,0]); fy = float(K_for_json[1,1])
        cx = float(K_for_json[0,2]); cy = float(K_for_json[1,2])

        pos = None
        if translation:
            # translation 字段可能是 dict x,y,z
            if isinstance(translation, dict):
                pos = [translation.get('x',0), translation.get('y',0), translation.get('z',0)]
            else:
                try:
                    pos = list(translation)
                except Exception:
                    pos = None

        rot_mat = None
        if rotation:
            # rotation 可能是 dict rx,ry,rz；这里尽可能保留原始结构
            if isinstance(rotation, dict):
                rx = rotation.get('rx'); ry = rotation.get('ry'); rz = rotation.get('rz')
                # 不强行转换为矩阵，保留原始 rotation 字段也可
                rot_mat = {'rx': rx, 'ry': ry, 'rz': rz}
            else:
                rot_mat = rotation

        camera_entry = {
            'id': i,
            'img_name': name,
            'width': final_w,
            'height': final_h,
            'position': pos,
            'rotation': rot_mat,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
        }
        updated_cameras.append(camera_entry)

        processed += 1

    print(f"已处理并保存 {processed} 张去畸变图片 到 {out_path}")

    # 将生成的内参写入 my_dataset/cameras.json（备份原文件），并同步写入 my_dataset/tool/cameras.json（备份）
    if write_cameras:
        # 顶层 cameras.json（保留原来的字典包装）
        cameras_out_path = Path('my_dataset/cameras.json')
        if cameras_out_path.exists():
            bak = cameras_out_path.with_suffix('.json.bak')
            try:
                shutil.copy2(str(cameras_out_path), str(bak))
                print(f"已备份原 cameras.json 到 {bak}")
            except Exception as e:
                print(f"备份原 cameras.json 失败: {e}")

        out = {'cameras': updated_cameras}
        try:
            with open(cameras_out_path, 'w') as f:
                json.dump(out, f, indent=2)
            print(f"已写入去畸变匹配的相机文件：{cameras_out_path}")
        except Exception as e:
            print(f"写入 cameras.json 失败: {e}")

        # 在 tool 目录下写入两个文件：一个保存原始标定相机（cameras_calib.json），一个保存去畸变后的相机（cameras_undistorted.json）
        tool_dir = Path(tool_dir_override) if tool_dir_override else Path('my_dataset/tool')
        tool_dir.mkdir(parents=True, exist_ok=True)

        calib_path = tool_dir / 'cameras_calib.json'
        undist_path = tool_dir / 'cameras_undistorted.json'
        compat_path = tool_dir / compat_name  # 仍写入以保持兼容

        # 备份并写入 calib（原始标定内参）
        if calib_path.exists():
            try:
                shutil.copy2(str(calib_path), str(calib_path.with_suffix('.json.bak')))
                print(f"已备份原 {calib_path} 到 {calib_path.with_suffix('.json.bak')}")
            except Exception as e:
                print(f"备份原 {calib_path} 失败: {e}")
        try:
            # 使用精度保持的方式写入原始标定相机数据
            output_str = json.dumps(original_cameras, indent=2, ensure_ascii=False)
            
            # 替换为原始精度的字符串
            for i, cam_data in enumerate(original_cameras):
                if '_original_strings' in cam_data and cam_data['_original_strings']:
                    orig_strs = cam_data['_original_strings']
                    # 替换fx和fy的值（fx = f, fy = f）
                    output_str = output_str.replace(f'"fx": {cam_data["fx"]}', f'"fx": {orig_strs["f"]}')
                    output_str = output_str.replace(f'"fy": {cam_data["fy"]}', f'"fy": {orig_strs["f"]}')
                    output_str = output_str.replace(f'"cx": {cam_data["cx"]}', f'"cx": {orig_strs["cx"]}')  
                    output_str = output_str.replace(f'"cy": {cam_data["cy"]}', f'"cy": {orig_strs["cy"]}')
            
            with open(calib_path, 'w') as f:
                f.write(output_str)
            print(f"已写入标定相机文件（高精度）：{calib_path}")
        except Exception as e:
            print(f"写入 {calib_path} 失败: {e}")

        # 备份并写入 undistorted（去畸变内参）
        if undist_path.exists():
            try:
                shutil.copy2(str(undist_path), str(undist_path.with_suffix('.json.bak')))
                print(f"已备份原 {undist_path} 到 {undist_path.with_suffix('.json.bak')}")
            except Exception as e:
                print(f"备份原 {undist_path} 失败: {e}")
        try:
            with open(undist_path, 'w') as f:
                json.dump(updated_cameras, f, indent=2)
            print(f"已写入去畸变相机文件：{undist_path}")
        except Exception as e:
            print(f"写入 {undist_path} 失败: {e}")

        # 兼容：仍然写入 tool/cameras.json（旧脚本可能依赖）
        try:
            if compat_path.exists():
                shutil.copy2(str(compat_path), str(compat_path.with_suffix('.json.bak')))
            with open(compat_path, 'w') as f:
                json.dump(updated_cameras, f, indent=2)
            print(f"已写入兼容相机文件：{compat_path}")
        except Exception as e:
            print(f"写入兼容 {compat_path} 失败: {e}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Undistort images and write camera intrinsics to JSON files.')
    parser.add_argument('--calib', type=str, default='data1.15/origin/calib/calib0.4723.json', help='Path to calibration JSON,原始标定文件')
    parser.add_argument('--images', type=str, default='data1.15/origin/lf', help='Input images directory')
    parser.add_argument('--out', type=str, default='data1.15/process/selfdataprocess/origin_images/lf', help='Output images directory for undistorted images')
    parser.add_argument('--no-write-cameras', dest='write_cameras', action='store_false', help='Do not write cameras JSON files')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha parameter (deprecated - new method always keeps original size)')
    parser.add_argument('--tool-dir', type=str, default='data1.15/process/selfdataprocess/origin_images/lf/tool', help='Tool directory to write cameras files')
    parser.add_argument('--compat-name', type=str, default='cameras.json', help='Compatibility filename to write in tool dir')

    args = parser.parse_args()

    undistort_images(calib_path=args.calib, images_dir=args.images, out_dir=args.out, write_cameras=args.write_cameras, alpha=args.alpha, tool_dir_override=args.tool_dir, compat_name=args.compat_name)
