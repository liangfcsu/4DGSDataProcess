#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
# 使用方法: 
# python scripts/colmap_process_scripts/convert.py -s usdata/images
# 或指定输出路径: 
# python scripts/colmap_process_scripts/convert.py -s usdata/seg1 -o usdata/dataprocess/colmap/nonbgdata

import os
import logging
from argparse import ArgumentParser
import shutil

# 这个 Python 脚本基于 MipNerF 360 仓库中提供的 shell 转换脚本。
# 脚本使用 COLMAP 对图像数据集进行处理，包括特征提取、匹配、重建和图像 undistortion。

# 创建参数解析器，用于处理命令行参数
parser = ArgumentParser("Colmap converter")
# 添加参数：是否禁用 GPU
parser.add_argument("--no_gpu", action='store_true')
# 添加参数：是否跳过特征匹配步骤
parser.add_argument("--skip_matching", action='store_true')
# 添加参数：源路径，必填，包含图像数据的目录
parser.add_argument("--source_path", "-s", required=True, type=str)
# 添加参数：输出路径，可选，用于存放处理结果（如不指定则使用 source_path）
parser.add_argument("--output_path", "-o", default="", type=str)
# 添加参数：相机模型，默认使用 OPENCV
parser.add_argument("--camera", default="OPENCV", type=str)
# 添加参数：COLMAP 可执行文件路径，默认使用系统路径的 colmap
parser.add_argument("--colmap_executable", default="", type=str)
# 添加参数：是否进行图像缩放
parser.add_argument("--resize", action="store_true")
# 添加参数：ImageMagick 可执行文件路径，默认使用系统路径的 magick
parser.add_argument("--magick_executable", default="", type=str)
# 解析命令行参数
args = parser.parse_args()

# 设置输出路径：如果未指定 output_path，则使用 source_path
output_path = args.output_path if args.output_path else args.source_path

# 设置 COLMAP 命令，如果指定了可执行文件路径，则使用它，否则使用默认的 colmap
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
# 设置 ImageMagick 命令，如果指定了可执行文件路径，则使用它，否则使用默认的 magick
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
# 根据是否禁用 GPU 设置 use_gpu 变量，1 表示使用 GPU，0 表示不使用
use_gpu = 1 if not args.no_gpu else 0
# 设置 GPU 索引，如果使用 GPU 则为 0，否则为 -1（表示 CPU）
gpu_index = 0 if use_gpu else -1

if not args.skip_matching:
    # 创建 distorted/sparse 目录，用于存储重建结果
    os.makedirs(output_path + "/distorted/sparse", exist_ok=True)

    # 特征提取步骤：从图像中提取特征点和描述符
    # 使用 COLMAP 的 feature_extractor 命令
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + output_path + "/distorted/database.db \
        --image_path " + args.source_path + " \
        --ImageReader.single_camera 0 \
        --ImageReader.camera_model " + args.camera + " \
        --FeatureExtraction.use_gpu " + str(use_gpu)
    # 执行特征提取命令
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # 特征匹配步骤：匹配不同图像之间的特征点
    # 使用 COLMAP 的 exhaustive_matcher 命令，进行穷举匹配
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + output_path + "/distorted/database.db \
        --FeatureMatching.use_gpu " + str(use_gpu)
    # 执行特征匹配命令
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # 束调整步骤：优化相机参数和3D点位置
    # 使用 COLMAP 的 mapper 命令
    # 默认的 Mapper 容差过大，这里减小它以加速束调整步骤
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + output_path + "/distorted/database.db \
        --image_path "  + args.source_path + " \
        --output_path "  + output_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    # 执行 mapper 命令
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

# 图像 undistortion 步骤：将图像 undistort 到理想的针孔内参
# 使用 COLMAP 的 image_undistorter 命令
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + " \
    --input_path " + output_path + "/distorted/sparse/1 \
    --output_path " + output_path + "\
    --output_type COLMAP")
# 执行图像 undistortion 命令
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

# 整理 sparse 目录：将 undistortion 后的 sparse 文件移动到 sparse/0 子目录
# 列出 sparse 目录中的文件
files = os.listdir(output_path + "/sparse")
# 创建 sparse/0 目录
os.makedirs(output_path + "/sparse/0", exist_ok=True)
# 遍历文件，将除了 '0' 目录外的文件移动到 sparse/0 中
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(output_path, "sparse", file)
    destination_file = os.path.join(output_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

# 如果指定了 resize 参数，则创建不同分辨率的图像副本
if(args.resize):
    print("Copying and resizing...")

    # 创建用于存储缩放图像的目录
    os.makedirs(output_path + "/images_2", exist_ok=True)
    os.makedirs(output_path + "/images_4", exist_ok=True)
    os.makedirs(output_path + "/images_8", exist_ok=True)
    # 获取 images 目录中的文件列表
    files = os.listdir(output_path + "/images")
    # 遍历每个文件，进行缩放
    for file in files:
        source_file = os.path.join(output_path, "images", file)

        # 缩放到 50%（images_2）
        destination_file = os.path.join(output_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        # 缩放到 25%（images_4）
        destination_file = os.path.join(output_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        # 缩放到 12.5%（images_8）
        destination_file = os.path.join(output_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

# 处理完成
print("Done.")
