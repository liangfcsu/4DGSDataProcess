#!/usr/bin/env python3
"""
使用 COLMAP image_undistorter 对 ims200-400 里的所有图像去畸变
- 相机畸变参数来自 data/100camdata/data/spare
- 输入图像：data/100camdata/ims200-400/camXXX/camXXXframeYYY.png
- 输出目录：data/100camdata/ims200-400_undistorted/
"""

import subprocess
import shutil
import os
import re
from pathlib import Path
from tqdm import tqdm


# ========== 路径配置 ==========
BASE_DIR      = Path("data/100camdata")
SPARE_DIR     = BASE_DIR / "data/spare"          # COLMAP稀疏模型（相机参数来源）
IMS_DIR       = BASE_DIR / "ims200-400"           # 待去畸变的图像目录
OUTPUT_DIR    = BASE_DIR / "ims200-400_undistorted"  # 去畸变输出目录
TEMP_DIR      = BASE_DIR / "_undistort_temp"      # 临时工作目录

# ==============================


def parse_images_txt(images_txt_path):
    """
    解析 images.txt，返回 cam_num -> (pose_str, camera_id) 的映射
    pose_str 格式: "qw qx qy qz tx ty tz"
    """
    cam_data = {}
    with open(images_txt_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue
        parts = line.split()
        if len(parts) >= 10:
            pose_str = ' '.join(parts[1:8])   # qw qx qy qz tx ty tz
            cam_id   = int(parts[8])
            img_name = parts[9]
            m = re.match(r'cam(\d+)frame\d+', img_name)
            if m:
                cam_num = int(m.group(1))
                cam_data[cam_num] = (pose_str, cam_id)
        i += 2  # 跳过特征点行
    return cam_data


def build_temp_model(cam_data, spare_txt_dir, temp_model_dir, temp_images_dir, ims_dir):
    """
    构建临时 COLMAP 文本模型，包含 ims200-400 全部帧
    同时将所有图像软链接到 temp_images_dir（flat 结构）
    """
    temp_model_dir.mkdir(parents=True, exist_ok=True)
    temp_images_dir.mkdir(parents=True, exist_ok=True)

    # 复制 cameras.txt（不变）
    shutil.copy2(spare_txt_dir / "cameras.txt", temp_model_dir / "cameras.txt")

    # 空 points3D.txt
    with open(temp_model_dir / "points3D.txt", 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: 0, mean track length: 0\n")

    # 收集所有图像
    all_image_entries = []
    img_id = 1

    cam_dirs = sorted([d for d in ims_dir.iterdir() if d.is_dir() and d.name.startswith("cam")])
    print(f"发现 {len(cam_dirs)} 个相机目录，开始建立图像链接...")

    for cam_dir in tqdm(cam_dirs, desc="处理相机目录"):
        m = re.match(r'cam(\d+)', cam_dir.name)
        if not m:
            continue
        cam_num = int(m.group(1))

        if cam_num not in cam_data:
            tqdm.write(f"  ⚠️  cam{cam_num:03d} 不在 COLMAP 模型中，跳过")
            continue

        pose_str, camera_id = cam_data[cam_num]

        for img_file in sorted(cam_dir.glob("*.png")):
            # 建立软链接（flat 结构，文件名已包含相机信息）
            link = temp_images_dir / img_file.name
            if not link.exists():
                link.symlink_to(img_file.resolve())

            all_image_entries.append((img_id, pose_str, camera_id, img_file.name))
            img_id += 1

    # 写 images.txt
    with open(temp_model_dir / "images.txt", 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(all_image_entries)}\n")
        for img_id, pose_str, cam_id, img_name in all_image_entries:
            f.write(f"{img_id} {pose_str} {cam_id} {img_name}\n")
            f.write("\n")  # 空特征点行

    print(f"✅ 临时模型构建完成：{len(all_image_entries)} 张图像")
    return len(all_image_entries)


def run_undistorter(temp_images_dir, temp_model_dir, output_dir):
    """运行 colmap image_undistorter"""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        'colmap', 'image_undistorter',
        '--image_path',  str(temp_images_dir),
        '--input_path',  str(temp_model_dir),
        '--output_path', str(output_dir),
        '--output_type', 'COLMAP',
    ]

    print("\n⚙️  运行 COLMAP image_undistorter ...")
    print("命令:", ' '.join(cmd))
    print()

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print("❌ COLMAP image_undistorter 失败")
        return False

    print("✅ image_undistorter 完成")
    return True


def reorganize_output(output_dir):
    """
    将 output_dir/images/camXXXframeYYY.png 整理为
    output_dir/images/camXXX/camXXXframeYYY.png
    """
    flat_images_dir = output_dir / "images"
    if not flat_images_dir.exists():
        print("⚠️  未找到输出 images 目录，跳过整理")
        return

    print("\n📁 整理输出目录结构...")
    moved = 0
    for img_file in tqdm(list(flat_images_dir.glob("cam*.png")), desc="整理文件"):
        m = re.match(r'(cam\d+)frame\d+\.png', img_file.name)
        if not m:
            continue
        cam_folder = flat_images_dir / m.group(1)
        cam_folder.mkdir(exist_ok=True)
        img_file.rename(cam_folder / img_file.name)
        moved += 1

    print(f"✅ 整理完成：{moved} 张图像按相机目录归类")


def main():
    print("=" * 60)
    print("📸 COLMAP 批量去畸变")
    print("=" * 60)
    print(f"相机参数来源: {SPARE_DIR}")
    print(f"输入图像:     {IMS_DIR}")
    print(f"输出目录:     {OUTPUT_DIR}")
    print(f"临时目录:     {TEMP_DIR}")
    print("=" * 60)

    # 检查依赖
    if not SPARE_DIR.exists():
        print(f"❌ COLMAP 模型目录不存在: {SPARE_DIR}")
        return

    if not IMS_DIR.exists():
        print(f"❌ 输入图像目录不存在: {IMS_DIR}")
        return

    # 1. 将 spare 转为文本格式
    spare_txt_dir = TEMP_DIR / "spare_txt"
    spare_txt_dir.mkdir(parents=True, exist_ok=True)
    print("\n🔄 转换 COLMAP 模型为文本格式...")
    subprocess.run([
        'colmap', 'model_converter',
        '--input_path',  str(SPARE_DIR),
        '--output_path', str(spare_txt_dir),
        '--output_type', 'TXT'
    ], check=True)
    print("✅ 模型转换完成")

    # 2. 解析相机-图像映射
    print("\n🔍 解析相机参数映射...")
    cam_data = parse_images_txt(spare_txt_dir / "images.txt")
    print(f"✅ 解析到 {len(cam_data)} 台相机的映射")

    # 3. 构建临时模型和图像目录
    temp_model_dir  = TEMP_DIR / "model"
    temp_images_dir = TEMP_DIR / "images"
    print("\n🔗 构建临时模型...")
    total = build_temp_model(cam_data, spare_txt_dir, temp_model_dir, temp_images_dir, IMS_DIR)

    if total == 0:
        print("❌ 没有可处理的图像，退出")
        return

    # 4. 运行去畸变
    if not run_undistorter(temp_images_dir, temp_model_dir, OUTPUT_DIR):
        return

    # 5. 将输出整理为 cam 子目录结构
    reorganize_output(OUTPUT_DIR)

    # 6. 清理临时目录
    print(f"\n🗑️  清理临时目录: {TEMP_DIR}")
    shutil.rmtree(TEMP_DIR)
    print("✅ 临时目录已清理")

    # 统计结果
    out_images_dir = OUTPUT_DIR / "images"
    if out_images_dir.exists():
        total_out = sum(1 for _ in out_images_dir.rglob("*.png"))
        print(f"\n{'=' * 60}")
        print("🎉 去畸变完成！")
        print(f"{'=' * 60}")
        print(f"输出图像总数: {total_out}")
        print(f"输出目录:     {OUTPUT_DIR.absolute()}")
        print(f"\n📁 输出结构:")
        print(f"  {OUTPUT_DIR}/images/cam001/cam001frame200.png")
        print(f"  {OUTPUT_DIR}/images/cam001/cam001frame201.png")
        print(f"  ...")
        print(f"  {OUTPUT_DIR}/sparse/  (去畸变后的 COLMAP 模型)")


if __name__ == '__main__':
    main()
