#!/usr/bin/env python3
"""
生成colmap格式后的点云 使用Hierarchical-Localization中的SuperGlue进行特征匹配和三角测量重建，输入校正后的图像
SuperGlue点云生成器 - 简单配置版本
直接修改下面的参数，然后运行脚本即可生成点云
"""

import json
import numpy as np
from pathlib import Path
import cv2
import shutil
import sys
import os
import time
from tqdm import tqdm
import argparse

# 添加hloc路径
hloc_path = Path(__file__).parent / "Hierarchical-Localization"
sys.path.insert(0, str(hloc_path.absolute()))

# 添加SuperGluePretrainedNetwork路径
superglue_path = hloc_path / "third_party" / "SuperGluePretrainedNetwork"
sys.path.insert(0, str(superglue_path.absolute()))

from hloc import extract_features, match_features, triangulation

# ===============================
# 🚀 配置参数 - 直接修改这里的值
# ===============================

# 📁 路径配置
INPUT_IMAGES_DIR = "data1.15/process/selfdataprocess/origin_images/lf/undistortimages"      # 输入图像文件夹路径
INPUT_SPARSE_DIR = "data1.15/process/selfdataprocess/origin_images/lf/sparse/0"    # 输入稀疏重建文件夹路径 (可选，设为None则不使用)
OUTPUT_DIR = "data1.15/process/selfdataprocess/origin_images/lf/superglue_output"          # 输出文件夹路径

# 🔍 SuperPoint特征提取参数
SUPERPOINT_MAX_KEYPOINTS = 4096          # 最大关键点数 (256-4096)
SUPERPOINT_KEYPOINT_THRESHOLD = 0.005    # 关键点置信度阈值 (0.001-0.1)
SUPERPOINT_NMS_RADIUS = 4                # 非极大值抑制半径 (1-10)
SUPERPOINT_RESIZE_MAX = 4000             # 图像最大尺寸 (800-4000)

# 🎯 SuperGlue匹配参数
SUPERGLUE_WEIGHTS = "indoor"            # 预训练权重 ("indoor" 或 "outdoor")
SUPERGLUE_SINKHORN_ITERATIONS = 100       # Sinkhorn算法迭代次数 (10-100)
SUPERGLUE_MATCH_THRESHOLD = 0.2          # 匹配置信度阈值 (0.1-0.5)

# 📐 三角测量参数
TRIANGULATION_MIN_ANGLE = 1.0            # 最小三角化角度（度）(0.5-5.0)
TRIANGULATION_MAX_ERROR = 4.0            # 最大重投影误差（像素）(1.0-10.0)

# ===============================
# 🔧 系统参数 (通常不需要修改)
# ===============================
GPU_ENABLED = True                       # 是否启用GPU
VERBOSE = True                           # 是否显示详细信息

def check_gpu_availability():
    """检查GPU可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"🖥️  GPU可用: {device_name} ({device_count} 个设备)")
            return True
        else:
            print("💻 使用CPU模式")
            return False
    except ImportError:
        print("⚠️  PyTorch未安装，使用CPU模式")
        return False

def validate_configuration():
    """验证配置参数"""
    try:
        # 测试hloc的SuperPoint和SuperGlue导入
        from hloc.extractors.superpoint import SuperPoint
        from hloc.matchers.superglue import SuperGlue
        print("✅ SuperGluePretrainedNetwork通过hloc可用")
        return True
    except ImportError as e:
        print(f"❌ SuperGluePretrainedNetwork不可用: {e}")
        return False

def main():
    """主函数"""
    # 检查是否提供了命令行参数（除了脚本名以外的参数）
    import sys
    provided_args = [arg for arg in sys.argv[1:] if not arg.startswith('--test')]
    has_args = len(provided_args) > 0
    
    if has_args:
        # 使用命令行参数
        parser = argparse.ArgumentParser(description='SuperGlue点云生成器 - 支持命令行参数')
        parser.add_argument('--images-dir', type=str, 
                           default="3dgs_dataprocess_usecalib/12.19test/test4dgs/images",
                           help='输入图像文件夹路径')
        parser.add_argument('--sparse-dir', type=str, 
                           default="3dgs_dataprocess_usecalib/12.19test/test4dgs/sparse/0",
                           help='输入稀疏重建文件夹路径 (可选)')
        parser.add_argument('--output-dir', type=str, 
                           default="3dgs_dataprocess_usecalib/12.19test/test4dgs/superglue_output",
                           help='输出文件夹路径')
        parser.add_argument('--max-keypoints', type=int, default=4096,
                           help='SuperPoint最大关键点数 (256-4096)')
        parser.add_argument('--keypoint-threshold', type=float, default=0.005,
                           help='SuperPoint关键点置信度阈值 (0.001-0.1)')
        parser.add_argument('--nms-radius', type=int, default=4,
                           help='SuperPoint非极大值抑制半径 (1-10)')
        parser.add_argument('--resize-max', type=int, default=4000,
                           help='SuperPoint图像最大尺寸 (800-4000)')
        parser.add_argument('--superglue-weights', type=str, default="indoor",
                           choices=['indoor', 'outdoor'],
                           help='SuperGlue预训练权重')
        parser.add_argument('--sinkhorn-iterations', type=int, default=100,
                           help='SuperGlue Sinkhorn算法迭代次数 (10-100)')
        parser.add_argument('--match-threshold', type=float, default=0.2,
                           help='SuperGlue匹配置信度阈值 (0.1-0.5)')
        parser.add_argument('--min-angle', type=float, default=1.0,
                           help='三角测量最小角度（度）(0.5-5.0)')
        parser.add_argument('--max-error', type=float, default=4.0,
                           help='三角测量最大重投影误差（像素）(1.0-10.0)')
        
        args = parser.parse_args()
        
        # 使用命令行参数
        images_dir_str = args.images_dir
        sparse_dir_str = args.sparse_dir
        output_dir_str = args.output_dir
        max_keypoints = args.max_keypoints
        keypoint_threshold = args.keypoint_threshold
        nms_radius = args.nms_radius
        resize_max = args.resize_max
        superglue_weights = args.superglue_weights
        sinkhorn_iterations = args.sinkhorn_iterations
        match_threshold = args.match_threshold
    else:
        # 使用硬编码配置
        images_dir_str = INPUT_IMAGES_DIR
        sparse_dir_str = INPUT_SPARSE_DIR
        output_dir_str = OUTPUT_DIR
        max_keypoints = SUPERPOINT_MAX_KEYPOINTS
        keypoint_threshold = SUPERPOINT_KEYPOINT_THRESHOLD
        nms_radius = SUPERPOINT_NMS_RADIUS
        resize_max = SUPERPOINT_RESIZE_MAX
        superglue_weights = SUPERGLUE_WEIGHTS
        sinkhorn_iterations = SUPERGLUE_SINKHORN_ITERATIONS
        match_threshold = SUPERGLUE_MATCH_THRESHOLD
    
    start_time = time.time()
    print("🚀 SuperGlue点云生成器 (简单配置版)")
    print("=" * 50)

    # 显示当前配置
    print("📋 当前配置参数:")
    print(f"  输入图像: {images_dir_str}")
    print(f"  稀疏重建: {sparse_dir_str or '无'}")
    print(f"  输出目录: {output_dir_str}")
    print(f"  SuperPoint关键点数: {max_keypoints}")
    print(f"  SuperGlue权重: {superglue_weights}")
    print()

    # 系统检查
    print("🔧 系统检查...")
    if not validate_configuration():
        return

    check_gpu_availability()

    # 设置路径 - 智能处理绝对路径和相对路径
    script_dir = Path(__file__).parent
    script_dir_name = script_dir.name  # 'self_process_scripts'
    cwd = Path.cwd()  # 当前工作目录
    
    def resolve_path(path_str):
        """解析路径：绝对路径直接使用，相对路径相对于当前工作目录"""
        if not path_str:
            return None
        path = Path(path_str)
        if path.is_absolute():
            return path
        else:
            # 相对路径，相对于当前工作目录（运行脚本的位置）
            return cwd / path
    
    images_dir = resolve_path(images_dir_str)
    sparse_dir = resolve_path(sparse_dir_str) if sparse_dir_str else None
    output_dir = resolve_path(output_dir_str)

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "temp"
    work_dir.mkdir(exist_ok=True)

    print(f"📂 图像目录: {images_dir}")
    print(f"📁 输出目录: {output_dir}")

    # 检查输入
    if not images_dir.exists():
        print(f"❌ 图像目录不存在: {images_dir}")
        return

    image_list = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    print(f"🖼️  找到 {len(image_list)} 张图像")

    if len(image_list) < 2:
        print("❌ 至少需要2张图像")
        return

    # 1. 特征提取 (SuperPoint)
    print("\n🔍 SuperPoint特征提取...")
    step_start = time.time()
    features_path = work_dir / "features.h5"

    if features_path.exists():
        print("⏭️  特征文件已存在，跳过特征提取")
    else:
        try:
            feature_conf = {
                'model': {
                    'name': 'superpoint',
                    'nms_radius': nms_radius,
                    'keypoint_threshold': keypoint_threshold,
                    'max_keypoints': max_keypoints
                },
                'preprocessing': {
                    'grayscale': True,
                    'resize_max': resize_max
                }
            }

            extract_features.main(feature_conf, images_dir, feature_path=features_path)
            print(f"✅ 特征提取完成 (耗时: {time.time() - step_start:.1f}s)")
        except Exception as e:
            print(f"❌ 特征提取失败: {e}")
            return

    # 2. 生成图像对
    print("🔗 生成图像对...")
    step_start = time.time()
    pairs_path = work_dir / "pairs.txt"

    if pairs_path.exists():
        print("⏭️  图像对文件已存在，跳过生成")
        with open(pairs_path, 'r') as f:
            pairs = [line.strip() for line in f if line.strip()]
    else:
        image_names = sorted([img.name for img in image_list])

        pairs = []
        for i, img1 in enumerate(image_names):
            for j, img2 in enumerate(image_names[i+1:], i+1):
                pairs.append(f"{img1} {img2}")

        with open(pairs_path, 'w') as f:
            f.write('\n'.join(pairs))

        print(f"✅ 生成 {len(pairs)} 个图像对 (耗时: {time.time() - step_start:.1f}s)")

    # 3. SuperGlue匹配
    print("🎯 SuperGlue匹配...")
    step_start = time.time()
    matches_path = work_dir / "matches.h5"

    if matches_path.exists():
        print("⏭️  匹配文件已存在，跳过SuperGlue匹配")
    else:
        try:
            matcher_conf = {
                'model': {
                    'name': 'superglue',
                    'weights': superglue_weights,
                    'sinkhorn_iterations': sinkhorn_iterations,
                    'match_threshold': match_threshold
                }
            }

            match_features.main(matcher_conf, pairs_path, features=features_path, matches=matches_path)
            print(f"✅ SuperGlue匹配完成 (耗时: {time.time() - step_start:.1f}s)")
        except Exception as e:
            print(f"❌ SuperGlue匹配失败: {e}")
            return

    # 4. 三角测量重建
    print("📐 三角测量重建...")
    step_start = time.time()
    sfm_dir = work_dir / "sfm"
    sfm_dir.mkdir(exist_ok=True)

    # 检查是否已有重建结果
    result_exists = False
    for candidate in [sfm_dir / "0", sfm_dir]:
        if (candidate / "points3D.txt").exists() or (candidate / "points3D.bin").exists():
            result_exists = True
            break

    if result_exists:
        print("⏭️  重建结果已存在，跳过三角测量")
    else:
        try:
            # 使用现有稀疏重建作为参考（如果存在）
            if sparse_dir and sparse_dir.exists():
                triangulation.main(sfm_dir, sparse_dir, images_dir, pairs_path, features_path, matches_path)
            else:
                # 从零开始重建
                from hloc.reconstruction import main as reconstruction_main
                reconstruction_main(sfm_dir, images_dir, pairs_path, features_path, matches_path)

            print(f"✅ 重建完成 (耗时: {time.time() - step_start:.1f}s)")
        except Exception as e:
            print(f"❌ 重建失败: {e}")
            return

    # 5. 导出结果并生成点云
    print("💾 导出结果...")
    step_start = time.time()

    # 找到重建结果 - 检查二进制和文本格式
    result_dir = None
    for candidate in [sfm_dir / "0", sfm_dir]:
        if (candidate / "points3D.txt").exists() or (candidate / "points3D.bin").exists():
            result_dir = candidate
            break

    if result_dir is None:
        print("❌ 未找到重建结果")
        return

    # 复制稀疏重建
    sparse_output = output_dir / "sparse" / "0"
    sparse_output.mkdir(parents=True, exist_ok=True)

    # 如果是二进制格式，转换为文本格式
    if (result_dir / "points3D.bin").exists():
        print("🔄 转换二进制格式为文本格式...")
        import subprocess

        # 使用COLMAP转换二进制到文本
        cmd = [
            "colmap", "model_converter",
            "--input_path", str(result_dir),
            "--output_path", str(sparse_output),
            "--output_type", "TXT"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
            print("✅ 转换完成")
        except subprocess.TimeoutExpired:
            print("❌ 转换超时")
            return
        except subprocess.CalledProcessError as e:
            print(f"❌ 转换失败: {e}")
            if e.stderr:
                print(f"错误详情: {e.stderr}")
            return
    else:
        # 复制文本文件
        for file in result_dir.iterdir():
            if file.suffix == '.txt':
                shutil.copy2(file, sparse_output)

    # 读取并导出点云
    points_file = sparse_output / "points3D.txt"
    if not points_file.exists():
        print("❌ 未找到points3D.txt")
        return

    try:
        points_3d = []
        with open(points_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 7:
                    x, y, z = map(float, parts[1:4])
                    r, g, b = map(int, parts[4:7])
                    points_3d.append([x, y, z, r, g, b])

        if not points_3d:
            print("❌ 点云为空")
            return

        print(f"📊 重建3D点数量: {len(points_3d)}")

        # 导出PLY格式
        ply_file = output_dir / "pointcloud.ply"
        with open(ply_file, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points_3d)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            for point in points_3d:
                f.write(f"{point[0]} {point[1]} {point[2]} {point[3]} {point[4]} {point[5]}\n")

        print(f"✅ 点云已导出: {ply_file}")

        # 详细统计信息
        points = np.array(points_3d)
        xyz = points[:, :3]
        colors = points[:, 3:]

        print(f"\n📈 点云统计信息:")
        print(f"   总点数: {len(points_3d)}")
        print(f"   X范围: {xyz[:, 0].min():.3f} ~ {xyz[:, 0].max():.3f} (范围: {xyz[:, 0].max() - xyz[:, 0].min():.3f})")
        print(f"   Y范围: {xyz[:, 1].min():.3f} ~ {xyz[:, 1].max():.3f} (范围: {xyz[:, 1].max() - xyz[:, 1].min():.3f})")
        print(f"   Z范围: {xyz[:, 2].min():.3f} ~ {xyz[:, 2].max():.3f} (范围: {xyz[:, 2].max() - xyz[:, 2].min():.3f})")

        # 颜色统计
        unique_colors = np.unique(colors, axis=0)
        print(f"   唯一颜色数: {len(unique_colors)}")

        # 计算点云密度（近似）
        volume = ((xyz[:, 0].max() - xyz[:, 0].min()) *
                 (xyz[:, 1].max() - xyz[:, 1].min()) *
                 (xyz[:, 2].max() - xyz[:, 2].min()))
        if volume > 0:
            density = len(points_3d) / volume
            print(f"   点云密度: {density:.2e} 点/单位³")

        print(f"\n✅ 导出完成 (耗时: {time.time() - step_start:.1f}s)")

    except Exception as e:
        print(f"❌ 点云导出失败: {e}")
        return

    # 总时间统计
    total_time = time.time() - start_time
    print(f"\n🎉 任务完成！总耗时: {total_time:.1f}s")
    print(f"📁 稀疏重建: {sparse_output}")
    print(f"🎯 点云文件: {ply_file}")
    print(f"📊 最终点数: {len(points_3d)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()