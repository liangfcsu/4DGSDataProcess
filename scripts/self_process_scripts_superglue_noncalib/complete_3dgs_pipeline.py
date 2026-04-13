#!/usr/bin/env python3
"""
complete_3dgs_pipeline.py - 无标定版本（两阶段处理）
python scripts/self_process_scripts_superglue_noncalib/complete_3dgs_pipeline.py --stage all --feature-method superpoint --matcher-method superglue --non-interactive
python scripts/self_process_scripts_superglue_noncalib/complete_3dgs_pipeline.py --undistort-method colmap --non-interactive
python scripts/self_process_scripts_superglue_noncalib/complete_3dgs_pipeline.py --undistort-method colmap --non-interactive
完整的3DGS训练数据预处理管线（无需预先标定文件）。

处理流程：
阶段1: COLMAP SfM自动标定
  - 使用100台相机的图像进行特征提取和匹配
  - 运行COLMAP SfM估计相机内外参数和畸变系数
  - 将COLMAP结果转换为标定文件格式

阶段2: 基于标定文件的3DGS数据准备
  - 使用估计的标定文件进行去畸变
  - 重新运行hloc获得更精确的3D重建
  - 转换为3DGS训练格式

适用于：
- 100台相机视频提取的同一帧图像
- 无相机内外参标定文件
- 依赖COLMAP自动估计相机参数

这是为3DGS训练优化的无标定工作流程。
"""
import json
import cv2
import numpy as np
import subprocess
import shutil
import argparse
from pathlib import Path
import sys
import os


def ensure_correct_working_directory():
    """确保脚本从正确的工作目录运行"""
    script_dir = Path(__file__).parent.absolute()
    current_dir = Path.cwd().absolute()
    
    if current_dir != script_dir:
        print(f"⚠️  工作目录不正确")
        print(f"当前目录: {current_dir}")
        print(f"脚本目录: {script_dir}")
        print(f"🔄 切换到正确目录...")
        os.chdir(script_dir)
        print(f"✅ 已切换到: {Path.cwd()}")
    
    return Path.cwd()


# ========================================
# 📁 路径配置 - 统一管理所有输入输出路径
# ========================================

class PathConfig:
    """路径配置类 - 统一管理所有文件路径（两阶段处理版本）"""
    
    # ========================================
    # 🔧 脚本路径 - 统一管理所有处理脚本
    # ========================================
    CONVERT_COLMAP_SCRIPT = 'tool/convert_colmap_to_calib.py' # COLMAP结果转标定文件
    UNDISTORT_SCRIPT = 'tool/undistort_for_hloc.py'         # 去畸变脚本
    HLOC_SCRIPT = 'tool/generate_pointcloud_multicam.py'    # hloc重建脚本
    
    # ========================================
    # 📁 用户配置路径（只需要修改这里）
    # ========================================
    ORIGIN_IMAGES = '../../huaban/59'               # 原始图像目录（100台相机）
    OUTPUT_BASE_DIR = '../../huaban/supergule'  # 输出根目录
    
    # ========================================
    # 🎯 阶段1: COLMAP SfM自动标定路径（自动生成）
    # ========================================
    COLMAP_SFM_DIR = f'{OUTPUT_BASE_DIR}/colmap_sfm'                     # COLMAP SfM输出目录
    COLMAP_DATABASE = f'{OUTPUT_BASE_DIR}/colmap_sfm/database.db'        # COLMAP数据库
    COLMAP_SPARSE = f'{OUTPUT_BASE_DIR}/colmap_sfm/sparse'               # COLMAP稀疏重建结果
    ESTIMATED_CALIB = f'{OUTPUT_BASE_DIR}/colmap_sfm/estimated_calib.json' # 估计的标定文件
    
    # ========================================
    # 📂 阶段2: 去畸变路径（自动生成）
    # ========================================
    UNDISTORT_OUTPUT = f'{OUTPUT_BASE_DIR}/undistorted/'                 # 去畸变输出根目录
    UNDISTORTED_IMAGES = f'{OUTPUT_BASE_DIR}/undistorted/images_undistorted' # 去畸变图像
    UNDISTORTED_CAMERAS = f'{OUTPUT_BASE_DIR}/undistorted/tool/cameras_undistorted.json' # 更新相机参数
    
    # ========================================
    # 🔺 hloc重建路径（自动生成）
    # ========================================
    HLOC_INPUT_DIR = f'{OUTPUT_BASE_DIR}/colmap'                           # hloc输入目录
    HLOC_OUTPUT_DIR = f'{OUTPUT_BASE_DIR}/hloc_outputs'           # hloc输出目录
    HLOC_SPARSE_DIR = f'{OUTPUT_BASE_DIR}/hloc_outputs/sparse_multicam_text'  # hloc重建结果
    
    # ========================================
    # 🎯 最终3DGS训练数据路径（自动生成）
    # ========================================
    TRAINING_DATA_DIR = f'{OUTPUT_BASE_DIR}/3dgs_training_data'            # 3DGS训练数据根目录
    TRAINING_IMAGES = f'{OUTPUT_BASE_DIR}/3dgs_training_data/images'       # 训练图像
    TRAINING_SPARSE = f'{OUTPUT_BASE_DIR}/3dgs_training_data/sparse/0'     # COLMAP格式数据
    TRAINING_TRANSFORMS = f'{OUTPUT_BASE_DIR}/3dgs_training_data/transforms.json'  # transforms.json
    
    @classmethod
    def print_config(cls):
        """打印当前路径配置"""
        print("📁 当前路径配置（两阶段处理版本）:")
        print(f"  输入图像: {cls.ORIGIN_IMAGES}")
        print(f"  输出根目录: {cls.OUTPUT_BASE_DIR}")
        print(f"  COLMAP SfM: {cls.COLMAP_SFM_DIR}")
        print(f"  估计标定: {cls.ESTIMATED_CALIB}")
        print(f"  去畸变输出: {cls.UNDISTORTED_IMAGES}")
        print(f"  最终训练数据: {cls.TRAINING_DATA_DIR}")
        print(f"  预期图像数量: 100张（100台相机）")


def stage1_colmap_sfm_calibration(feature_method='superpoint', matcher_method='superglue'):
    """阶段1: 使用COLMAP SfM进行自动标定"""
    
    print("\n=== 阶段1: COLMAP SfM自动标定 ===")
    
    # 检查输入
    images_dir = Path(PathConfig.ORIGIN_IMAGES)
    if not images_dir.exists():
        print("❌ 输入图像目录不存在")
        return False
    
    # 统计图像数量
    image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))
    print(f"📷 检测到图像: {len(image_files)} 张")
    
    # 创建COLMAP SfM目录
    sfm_dir = Path(PathConfig.COLMAP_SFM_DIR)
    sfm_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行COLMAP SfM自动标定
    try:
        print("🚀 运行COLMAP SfM...")
        
        # 特征提取
        feature_extract_cmd = [
            'colmap', 'feature_extractor',
            '--database_path', str(Path(PathConfig.COLMAP_DATABASE)),
            '--image_path', str(images_dir),
            '--ImageReader.single_camera', '0'
        ]
        
        result = subprocess.run(feature_extract_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ 特征提取失败: {result.stderr}")
            return False
        
        print("✅ 特征提取完成")
        
        # 特征匹配
        feature_match_cmd = [
            'colmap', 'exhaustive_matcher',
            '--database_path', str(Path(PathConfig.COLMAP_DATABASE))
        ]
        
        result = subprocess.run(feature_match_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ 特征匹配失败: {result.stderr}")
            return False
        
        print("✅ 特征匹配完成")
        
        # 稀疏重建 (SfM)
        sparse_dir = Path(PathConfig.COLMAP_SPARSE)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        
        mapper_cmd = [
            'colmap', 'mapper',
            '--database_path', str(Path(PathConfig.COLMAP_DATABASE)),
            '--image_path', str(images_dir),
            '--output_path', str(sparse_dir),
            '--Mapper.multiple_models', '0',
            '--Mapper.ba_refine_focal_length', '1',
            '--Mapper.ba_refine_extra_params', '1',
            '--Mapper.min_num_matches', '15',
            '--Mapper.num_threads', '-1'
        ]
        
        result = subprocess.run(mapper_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ SfM重建失败: {result.stderr}")
            return False
        
        print("✅ SfM重建完成")
        
        # 将COLMAP二进制结果转换为文本格式
        convert_binary_cmd = [
            'colmap', 'model_converter',
            '--input_path', str(sparse_dir / '0'),
            '--output_path', str(sparse_dir / '0'),
            '--output_type', 'TXT'
        ]
        
        result = subprocess.run(convert_binary_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ 格式转换失败: {result.stderr}")
            return False
        
        print("✅ 格式转换完成")
        
        # 转换COLMAP结果为标定文件格式
        convert_cmd = [
            sys.executable, PathConfig.CONVERT_COLMAP_SCRIPT,
            '--colmap_dir', str(sparse_dir / '0'),
            '--output', PathConfig.ESTIMATED_CALIB
        ]
        
        result = subprocess.run(convert_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ 标定转换失败: {result.stderr}")
            return False
        
        print(f"✅ 生成估计标定文件: {PathConfig.ESTIMATED_CALIB}")
        
        # 验证结果
        if not Path(PathConfig.ESTIMATED_CALIB).exists():
            print("❌ 估计标定文件未生成")
            return False
        
        # 显示简要统计
        with open(PathConfig.ESTIMATED_CALIB, 'r') as f:
            calib_data = json.load(f)
        
        # 检查数据格式：新格式是数组，旧格式是嵌套字典
        if isinstance(calib_data, list):
            cameras_count = len(calib_data)
        else:
            cameras_count = len(calib_data['Calibration']['cameras'])
        
        print(f"📊 SfM统计:")
        print(f"  - 估计相机数: {cameras_count} 台")
        print(f"  - 原始图像: {len(image_files)} 张")
        print(f"  - 重建成功率: {cameras_count/len(image_files)*100:.1f}%")
        
        return True
        
    except FileNotFoundError:
        print("❌ COLMAP未安装，请先安装COLMAP")
        print("安装指南: https://colmap.github.io/install.html")
        return False
    except Exception as e:
        print(f"❌ COLMAP SfM处理出错: {e}")
        return False


def stage2_undistort_with_calibration(undistort_method='custom'):
    """阶段2: 使用估计的标定文件进行去畸变"""
    
    print("\n=== 阶段2: 基于估计标定的去畸变 ===")
    
    # 检查估计标定文件
    if not Path(PathConfig.ESTIMATED_CALIB).exists():
        print("❌ 估计标定文件不存在，请先运行阶段1")
        return False
    
    if undistort_method == 'custom':
        # 使用自定义去畸变
        cmd = [
            sys.executable, PathConfig.UNDISTORT_SCRIPT,
            '--calib', PathConfig.ESTIMATED_CALIB,
            '--images', PathConfig.ORIGIN_IMAGES, 
            '--output', PathConfig.UNDISTORT_OUTPUT,
            '--alpha', '0.0'  # 最小化内参变化
        ]
        
        print(f"运行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ 去畸变失败: {result.stderr}")
            return False
        
        print("✅ 自定义去畸变完成")
        
        # 验证结果
        undist_dir = Path(PathConfig.UNDISTORTED_IMAGES)
        cameras_file = Path(PathConfig.UNDISTORTED_CAMERAS)
        
        if not undist_dir.exists() or not cameras_file.exists():
            print("❌ 去畸变输出文件不完整")
            return False
        
        # 统计图像数量
        image_count = len(list(undist_dir.glob('*.png')))
        print(f"✅ 生成去畸变图像: {image_count} 张")
        
    elif undistort_method == 'colmap':
        # 使用COLMAP image_undistorter
        print("🚀 使用COLMAP image_undistorter...")
        
        # COLMAP undistorter需要sparse目录和图像目录
        sparse_dir = Path(PathConfig.COLMAP_SPARSE) / '0'
        images_dir = Path(PathConfig.ORIGIN_IMAGES)
        output_dir = Path(PathConfig.UNDISTORT_OUTPUT)
        
        if not sparse_dir.exists():
            print(f"❌ COLMAP sparse目录不存在: {sparse_dir}")
            return False
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 运行COLMAP image_undistorter
        undistorter_cmd = [
            'colmap', 'image_undistorter',
            '--image_path', str(images_dir),
            '--input_path', str(sparse_dir),
            '--output_path', str(output_dir),
            '--output_type', 'COLMAP'
        ]
        
        print(f"⚙️ COLMAP去畸变: {' '.join(undistorter_cmd)}")
        result = subprocess.run(undistorter_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ COLMAP去畸变失败: {result.stderr}")
            return False
        
        print("✅ COLMAP去畸变完成")
        
        # COLMAP输出直接在output_path下
        colmap_images = output_dir / 'images'
        colmap_sparse = output_dir / 'sparse'
        
        if colmap_images.exists() and colmap_sparse.exists():
            # 移动内容到我们的标准位置
            target_images = Path(PathConfig.UNDISTORTED_IMAGES)
            if target_images.exists():
                shutil.rmtree(target_images)
            shutil.move(str(colmap_images), str(target_images))
            
            # 复制sparse到我们的位置
            target_sparse = Path(PathConfig.TRAINING_SPARSE)
            if target_sparse.exists():
                shutil.rmtree(target_sparse.parent)
            target_sparse.parent.mkdir(parents=True, exist_ok=True)
            target_sparse.mkdir(exist_ok=True)
            
            # 移动sparse目录的内容到sparse/0
            for item in colmap_sparse.iterdir():
                shutil.move(str(item), str(target_sparse))
            
            # 转换为文本格式
            convert_cmd = [
                'colmap', 'model_converter',
                '--input_path', str(target_sparse),
                '--output_path', str(target_sparse),
                '--output_type', 'TXT'
            ]
            subprocess.run(convert_cmd, capture_output=True, text=True)
            
            # 转换为文本格式
            convert_cmd = [
                'colmap', 'model_converter',
                '--input_path', str(target_sparse),
                '--output_path', str(target_sparse),
                '--output_type', 'TXT'
            ]
            subprocess.run(convert_cmd, capture_output=True, text=True)
        else:
            print(f"❌ COLMAP输出不完整: images={colmap_images.exists()}, sparse={colmap_sparse.exists()}")
            return False
        
        # 验证结果
        undist_dir = Path(PathConfig.UNDISTORTED_IMAGES)
        sparse_dir_final = Path(PathConfig.TRAINING_SPARSE)
        
        if not undist_dir.exists():
            print("❌ 未找到去畸变图像目录")
            return False
        
        image_count = len(list(undist_dir.glob('*.png')))
        print(f"✅ 生成去畸变图像: {image_count} 张")
        
        if sparse_dir_final.exists():
            print("✅ 生成COLMAP sparse数据")
            
            # 生成cameras_undistorted.json供hloc使用
            cameras_json_path = Path(PathConfig.UNDISTORTED_CAMERAS)
            cameras_json_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 从COLMAP sparse数据生成相机JSON
            cameras_txt = sparse_dir_final / 'cameras.txt'
            images_txt = sparse_dir_final / 'images.txt'
            
            if cameras_txt.exists() and images_txt.exists():
                # 使用现有的转换脚本
                convert_cmd = [
                    sys.executable, PathConfig.CONVERT_COLMAP_SCRIPT,
                    '--colmap_dir', str(sparse_dir_final),
                    '--output', str(cameras_json_path)
                ]
                result = subprocess.run(convert_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ 生成相机参数JSON文件")
                else:
                    print(f"⚠️ 生成相机JSON失败: {result.stderr}")
            else:
                print("⚠️ 无法生成相机JSON文件，缺少COLMAP文本文件")
        
    else:
        print(f"❌ 不支持的去畸变方法: {undistort_method}")
        return False
    
    return True


def step2_hloc_reconstruction(feature_method='superpoint', matcher_method='superglue'):
    """步骤2: 使用去畸变图像进行hloc重建"""
    
    print("\n=== 步骤2: hloc点云重建 ===")
    
    # 检查输入
    undist_dir = Path(PathConfig.UNDISTORTED_IMAGES)
    cameras_file = Path(PathConfig.UNDISTORTED_CAMERAS)
    
    if not undist_dir.exists() or not cameras_file.exists():
        print("❌ 缺少去畸变输入文件")
        return False
    
    # hloc脚本期望的目录结构 - 使用绝对路径
    script_dir = Path(__file__).parent.absolute()
    data_dir = (script_dir / PathConfig.OUTPUT_BASE_DIR).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    colmap_input = data_dir / 'colmap'
    colmap_input.mkdir(exist_ok=True)
    
    # 复制去畸变图像到colmap/images
    colmap_images = colmap_input / 'images'
    if colmap_images.exists():
        shutil.rmtree(colmap_images)
    shutil.copytree(undist_dir, colmap_images)
    
    # 复制去畸变相机参数到colmap/cameras.json
    colmap_cameras = colmap_input / 'cameras.json'
    shutil.copy2(cameras_file, colmap_cameras)
    
    print(f"✅ 准备hloc输入: {colmap_input}")
    
    # 运行hloc重建 - 使用绝对路径避免相对路径问题
    original_cwd = os.getcwd()
    hloc_script_path = script_dir / PathConfig.HLOC_SCRIPT
    
    try:
        # 切换到data目录
        os.chdir(data_dir)
        
        cmd = [sys.executable, str(hloc_script_path), 
               "--feature-method", feature_method,
               "--matcher-method", matcher_method,
               "--non-interactive"]
        print(f"在 {data_dir} 目录运行命令: {' '.join(cmd)}")
        
        # 非交互式模式，不需要自动输入
        result = subprocess.run(cmd, text=True, capture_output=True)
        
    finally:
        # 恢复原目录
        os.chdir(original_cwd)
    
    if result.returncode != 0:
        print(f"❌ hloc重建失败")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    
    print(f"✅ hloc重建完成")
    
    # hloc脚本输出到data_dir下的hloc_outputs_multicam，需要重命名为目标目录
    hloc_default_output = data_dir / 'hloc_outputs_multicam'
    hloc_target_output = Path(PathConfig.HLOC_OUTPUT_DIR).resolve()
    
    if hloc_default_output.exists():
        if hloc_target_output.exists():
            shutil.rmtree(hloc_target_output)
        shutil.move(str(hloc_default_output), str(hloc_target_output))
        print(f"✅ 移动hloc输出到: {hloc_target_output}")
    else:
        print("⚠️ 未找到hloc输出目录")
        print(f"❌ 期望目录: {hloc_default_output}")
        if result.stdout:
            print(f"STDOUT: {result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
    
    # 检查重建结果
    points_file = hloc_target_output / 'sparse_multicam_text' / 'points3D.txt'
    if not points_file.exists():
        print("❌ 未生成点云文件")
        print(f"❌ 期望文件: {points_file}")
        
        # 详细诊断输出目录
        if hloc_target_output.exists():
            print(f"📁 输出目录内容: {list(hloc_target_output.iterdir())}")
            # 检查所有可能的sparse目录
            for item in hloc_target_output.rglob('*'):
                if item.is_dir() and 'sparse' in item.name.lower():
                    print(f"📁 找到sparse目录: {item}")
                    if item.is_dir():
                        print(f"   内容: {list(item.iterdir())}")
        else:
            print(f"❌ 输出目录不存在: {hloc_target_output}")
        
        print("⚠️ 可能的原因：图像匹配不足或重建失败")
        print("⚠️ 继续处理，但可能影响3DGS训练质量")
        return True
    
    # 统计点云数量
    with open(points_file, 'r') as f:
        point_count = len([line for line in f if not line.startswith('#') and line.strip()])
    
    print(f"✅ 生成3D点数量: {point_count}")
    
    if point_count < 1000:
        print("⚠️ 点云数量较少，可能影响3DGS训练效果")
    elif point_count > 10000:
        print("✅ 点云密度充足，适合3DGS训练")
    
    return True


def step3_prepare_3dgs_data(use_colmap_sparse=False):
    """步骤3: 准备3DGS训练数据"""
    
    print("\n=== 步骤3: 准备3DGS训练数据 ===")
    
    # 根据模式选择sparse源
    if use_colmap_sparse:
        source_sparse = Path(PathConfig.TRAINING_SPARSE)
        print("📊 使用COLMAP sparse数据")
    else:
        source_sparse = Path(PathConfig.HLOC_SPARSE_DIR)
        print("📊 使用hloc sparse数据")
    
    if not source_sparse.exists():
        print(f"❌ 重建结果不存在: {source_sparse}")
        return False
    
    # 创建3DGS训练目录
    training_dir = Path(PathConfig.TRAINING_DATA_DIR)
    training_dir.mkdir(exist_ok=True)
    
    # 复制去畸变图像
    images_dir = Path(PathConfig.TRAINING_IMAGES)
    if images_dir.exists():
        shutil.rmtree(images_dir)
    
    shutil.copytree(PathConfig.UNDISTORTED_IMAGES, images_dir)
    print(f"✅ 复制训练图像到: {images_dir}")
    
    # 创建sparse目录结构
    sparse_dir = Path(PathConfig.TRAINING_SPARSE)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    # 如果不是colmap sparse，复制重建结果
    if not use_colmap_sparse:
        for file_name in ['cameras.txt', 'images.txt', 'points3D.txt']:
            src = source_sparse / file_name
            dst = sparse_dir / file_name
            if src.exists():
                shutil.copy2(src, dst)
                print(f"✅ 复制: {file_name}")
            else:
                print(f"❌ 缺失: {file_name}")
                return False
    else:
        print("✅ COLMAP sparse数据已在目标位置")
    
    # 转换相机格式为PINHOLE
    if not convert_cameras_to_pinhole_format(sparse_dir):
        print("❌ 相机格式转换失败")
        return False
    
    # 生成transforms.json (某些3DGS实现需要)
    create_transforms_json(training_dir)
    
    print(f"✅ 3DGS训练数据准备完成: {PathConfig.TRAINING_DATA_DIR}")
    return True


def convert_cameras_to_pinhole_format(sparse_dir):
    """将SIMPLE_RADIAL相机转换为PINHOLE格式 (3DGS要求)"""
    
    print("\n📸 转换相机格式: SIMPLE_RADIAL → PINHOLE")
    
    cameras_file = sparse_dir / 'cameras.txt'
    if not cameras_file.exists():
        print(f"❌ 相机文件不存在: {cameras_file}")
        return False
    
    # 读取相机文件
    with open(cameras_file, 'r') as f:
        lines = f.readlines()
    
    converted_lines = []
    conversion_count = 0
    
    for line in lines:
        # 保留注释行和空行
        if line.startswith('#') or not line.strip():
            converted_lines.append(line)
            continue
        
        parts = line.strip().split()
        if len(parts) < 4:
            converted_lines.append(line)
            continue
        
        cam_id = parts[0]
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        
        if model == 'SIMPLE_RADIAL':
            # SIMPLE_RADIAL格式: CAM_ID SIMPLE_RADIAL WIDTH HEIGHT f cx cy k
            if len(parts) >= 8:
                f = float(parts[4])     # 焦距
                cx = float(parts[5])    # 主点x
                cy = float(parts[6])    # 主点y
                # k = float(parts[7])   # 径向畸变(PINHOLE不需要)
                
                # 转换为PINHOLE格式: CAM_ID PINHOLE WIDTH HEIGHT fx fy cx cy
                pinhole_line = f"{cam_id} PINHOLE {width} {height} {f} {f} {cx} {cy}\n"
                converted_lines.append(pinhole_line)
                conversion_count += 1
                print(f"✅ 转换相机 {cam_id}: SIMPLE_RADIAL → PINHOLE")
            else:
                print(f"⚠️ 相机 {cam_id}: SIMPLE_RADIAL参数不足")
                converted_lines.append(line)
                
        elif model == 'PINHOLE':
            print(f"✅ 相机 {cam_id}: 已经是PINHOLE格式")
            converted_lines.append(line)
            
        else:
            print(f"⚠️ 相机 {cam_id}: 未知格式 {model}")
            converted_lines.append(line)
    
    # 写入转换后的文件
    with open(cameras_file, 'w') as f:
        f.writelines(converted_lines)
    
    if conversion_count > 0:
        print(f"🎉 成功转换 {conversion_count} 个相机为PINHOLE格式")
    
    # 验证转换结果
    return verify_pinhole_conversion(sparse_dir)


def verify_pinhole_conversion(sparse_dir):
    """验证所有相机都是PINHOLE格式"""
    
    cameras_file = sparse_dir / 'cameras.txt'
    
    with open(cameras_file, 'r') as f:
        lines = f.readlines()
    
    pinhole_count = 0
    other_count = 0
    
    print("\n📋 相机模型验证:")
    
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue
            
        parts = line.strip().split()
        if len(parts) >= 4:
            cam_id = parts[0]
            model = parts[1]
            
            if model == 'PINHOLE':
                pinhole_count += 1
            else:
                other_count += 1
                print(f"⚠️ 相机 {cam_id}: {model} (非PINHOLE)")
    
    print(f"✅ PINHOLE相机: {pinhole_count}")
    if other_count > 0:
        print(f"⚠️ 其他格式: {other_count}")
        return False
    
    print("🎉 所有相机都是PINHOLE格式，满足3DGS训练要求！")
    return True


def create_transforms_json(training_dir):
    """创建transforms.json文件 (兼容某些3DGS实现)"""
    
    # 读取相机和图像信息
    sparse_dir = Path(PathConfig.TRAINING_SPARSE)
    cameras_file = sparse_dir / 'cameras.txt'
    images_file = sparse_dir / 'images.txt'
    
    if not cameras_file.exists() or not images_file.exists():
        return
    
    # 解析相机参数
    cameras = {}
    with open(cameras_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                cam_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                fx, fy, cx, cy = map(float, parts[4:8])
                cameras[cam_id] = {
                    'w': width, 'h': height,
                    'fl_x': fx, 'fl_y': fy,
                    'cx': cx, 'cy': cy
                }
    
    # 解析图像位姿
    frames = []
    with open(images_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue
        
        parts = line.split()
        if len(parts) >= 10:
            try:
                img_id = int(parts[0])
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                cam_id = int(parts[8])
                file_path = parts[9]
                
                # 四元数转旋转矩阵
                def quat_to_rotation_matrix(qw, qx, qy, qz):
                    return np.array([
                        [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
                        [2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
                        [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx*qx-2*qy*qy]
                    ])
                
                R = quat_to_rotation_matrix(qw, qx, qy, qz)
                t = np.array([tx, ty, tz])
                
                # WTC到CTW变换 (3DGS需要camera-to-world)
                R_ctw = R.T
                t_ctw = -R.T @ t
                
                # 构建4x4变换矩阵
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = R_ctw
                transform_matrix[:3, 3] = t_ctw
                
                frame = {
                    "file_path": f"./images/{file_path}",
                    "transform_matrix": transform_matrix.tolist()
                }
                
                if cam_id in cameras:
                    frame.update(cameras[cam_id])
                
                frames.append(frame)
            except ValueError as e:
                print(f"⚠️ 跳过无效行: {line[:50]}... (错误: {e})")
        
        # 跳过下一行（特征点数据）
        i += 2
    
    # 创建transforms.json
    if frames:
        # 获取第一个相机的参数作为默认值
        first_cam = cameras.get(1, {})
        
        transforms = {
            "camera_model": "OPENCV",
            "fl_x": first_cam.get('fl_x', 1000),
            "fl_y": first_cam.get('fl_y', 1000),
            "cx": first_cam.get('cx', 500),
            "cy": first_cam.get('cy', 500),
            "w": first_cam.get('w', 1000),
            "h": first_cam.get('h', 1000),
            "frames": frames
        }
        
        transforms_file = Path(PathConfig.TRAINING_TRANSFORMS)
        with open(transforms_file, 'w') as f:
            json.dump(transforms, f, indent=2)
        
        print(f"✅ 创建: transforms.json ({len(frames)} 帧)")


def verify_training_data():
    """验证3DGS训练数据的完整性"""
    
    print("\n=== 验证训练数据 ===")
    
    training_dir = Path(PathConfig.TRAINING_DATA_DIR)
    issues = []
    
    # 检查必要文件
    required_files = [
        PathConfig.TRAINING_IMAGES,
        str(Path(PathConfig.TRAINING_SPARSE) / 'cameras.txt'),
        str(Path(PathConfig.TRAINING_SPARSE) / 'images.txt'), 
        str(Path(PathConfig.TRAINING_SPARSE) / 'points3D.txt')
    ]
    
    for file_path in required_files:
        full_path = Path(file_path)
        if not full_path.exists():
            issues.append(f"缺失: {file_path}")
    
    # 检查图像数量
    images_dir = Path(PathConfig.TRAINING_IMAGES)
    if images_dir.exists():
        image_count = len(list(images_dir.glob('*.png')))
        if image_count == 0:
            issues.append("图像目录为空")
        else:
            print(f"✅ 训练图像: {image_count} 张")
    
    # 检查点云数量
    points_file = Path(PathConfig.TRAINING_SPARSE) / 'points3D.txt'
    if points_file.exists():
        with open(points_file, 'r') as f:
            point_count = len([line for line in f if not line.startswith('#') and line.strip()])
        
        if point_count < 1000:
            issues.append(f"点云数量过少: {point_count} (建议 >5000)")
        else:
            print(f"✅ 3D点云: {point_count} 个点")
    
    if issues:
        print("\n❌ 发现问题:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✅ 训练数据验证通过!")
        return True


def main():
    """完整的3DGS预处理管线（两阶段处理）"""
    
    # 确保从正确的工作目录运行
    ensure_correct_working_directory()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="3DGS训练数据预处理管线（两阶段处理）")
    parser.add_argument('--feature-method', type=str, default='superpoint',
                       choices=['sift', 'superpoint'],
                       help="特征提取方法 (默认: superpoint)")
    parser.add_argument('--matcher-method', type=str, default='superglue',
                       choices=['nn-mutual', 'nn-ratio', 'superglue'],
                       help="特征匹配方法 (默认: superglue)")
    parser.add_argument('--undistort-method', type=str, default='custom',
                       choices=['custom', 'colmap'],
                       help="去畸变方法: custom=自定义(alpha=0), colmap=COLMAP官方 (默认: custom)")
    parser.add_argument('--stage', type=str, choices=['1', '2', 'all'], default='all',
                       help="运行阶段: 1=COLMAP SfM标定, 2=基于标定文件的3DGS数据准备, all=完整流程 (默认: all)")
    parser.add_argument('--non-interactive', action='store_true',
                       help='非交互模式（当前脚本默认非交互，此参数保留以兼容子脚本调用）')
    
    args = parser.parse_args()
    
    print("=== 3DGS训练数据完整预处理管线（两阶段处理）===")
    print("阶段1: COLMAP SfM自动标定 → 阶段2: 基于标定文件的3DGS数据准备")
    
    # 打印路径配置
    PathConfig.print_config()
    print()
    
    # 检查输入图像目录
    if not Path(PathConfig.ORIGIN_IMAGES).exists():
        print("❌ 输入图像目录不存在")
        return
    
    # 使用命令行参数或默认值
    feature_method = args.feature_method
    matcher_method = args.matcher_method
    undistort_method = args.undistort_method
    
    print(f"🔧 特征提取: {feature_method.upper()}")
    print(f"🔗 特征匹配: {matcher_method.upper()}")
    print(f"🔧 去畸变方法: {undistort_method}")
    
    success = True
    
    # 阶段1: COLMAP SfM自动标定
    if args.stage in ['1', 'all'] and success:
        success = stage1_colmap_sfm_calibration(feature_method, matcher_method)
    
    # 阶段2: 基于标定文件的处理
    if args.stage in ['2', 'all'] and success:
        # 2.1 去畸变
        success = stage2_undistort_with_calibration(undistort_method)
        
        # 2.2 hloc重建 (无论哪种去畸变方法都进行hloc重建)
        if success:
            success = step2_hloc_reconstruction(feature_method, matcher_method)
        
        # 2.3 准备训练数据
        if success:
            success = step3_prepare_3dgs_data(use_colmap_sparse=False)
    
    
    # 验证结果
    if success:
        verify_training_data()
        
        print("\n" + "="*50)
        print("🎉 3DGS训练数据预处理完成!")
        print(f"\n📁 训练数据位置: {PathConfig.TRAINING_DATA_DIR}")
        print("📋 包含文件:")
        print("  - images/          # 去畸变训练图像")
        print("  - sparse/0/        # COLMAP重建结果 (PINHOLE格式)")
        print("  - transforms.json  # 可选：某些3DGS实现需要")
        print("\n🔄 处理流程包含:")
        print("  ✅ alpha=0去畸变 (最小化内参变化)")
        print("  ✅ hloc特征提取与匹配")
        print("  ✅ SIMPLE_RADIAL → PINHOLE转换")
        print("  ✅ 3DGS训练数据格式化")
        print("\n🚀 下一步: 使用3DGS进行训练")
        print(f"例如: python train.py -s {PathConfig.TRAINING_DATA_DIR}")
    else:
        print("\n❌ 预处理失败，请检查错误信息")


if __name__ == '__main__':
    main()