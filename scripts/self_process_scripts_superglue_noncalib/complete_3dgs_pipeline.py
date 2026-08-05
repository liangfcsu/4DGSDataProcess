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
import re


def runtime_script(rel_path: str) -> str:
    source = Path(__file__).parent / rel_path
    bytecode = source.with_suffix(source.suffix + "c")
    return rel_path if source.exists() else str(Path(rel_path).with_suffix(Path(rel_path).suffix + "c"))


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
    CONVERT_COLMAP_SCRIPT = runtime_script('tool/convert_colmap_to_calib.py') # COLMAP结果转标定文件
    UNDISTORT_SCRIPT = runtime_script('tool/undistort_for_hloc.py')         # 去畸变脚本
    HLOC_SCRIPT = runtime_script('tool/generate_pointcloud_multicam.py')    # hloc重建脚本
    
    # ========================================
    # 📁 用户配置路径（只需要修改这里）
    # ========================================
    ORIGIN_IMAGES = '../../data/footage/1'               # 原始图像目录（100台相机）
    OUTPUT_BASE_DIR = '../../data/footage/1/superglue'  # 输出根目录
    
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
    def configure(cls, input_images=None, output_dir=None):
        """运行时覆盖输入图像目录和输出根目录，并重算所有派生路径。

        供桌面 App / 命令行传入用户选择的路径使用。
        """
        if input_images:
            cls.ORIGIN_IMAGES = str(input_images)
        if output_dir:
            cls.OUTPUT_BASE_DIR = str(output_dir)

        base = cls.OUTPUT_BASE_DIR
        # 阶段1: COLMAP SfM
        cls.COLMAP_SFM_DIR = f'{base}/colmap_sfm'
        cls.COLMAP_DATABASE = f'{base}/colmap_sfm/database.db'
        cls.COLMAP_SPARSE = f'{base}/colmap_sfm/sparse'
        cls.ESTIMATED_CALIB = f'{base}/colmap_sfm/estimated_calib.json'
        # 阶段2: 去畸变
        cls.UNDISTORT_OUTPUT = f'{base}/undistorted/'
        cls.UNDISTORTED_IMAGES = f'{base}/undistorted/images_undistorted'
        cls.UNDISTORTED_CAMERAS = f'{base}/undistorted/tool/cameras_undistorted.json'
        # hloc 重建
        cls.HLOC_INPUT_DIR = f'{base}/colmap'
        cls.HLOC_OUTPUT_DIR = f'{base}/hloc_outputs'
        cls.HLOC_SPARSE_DIR = f'{base}/hloc_outputs/sparse_multicam_text'
        # 最终 3DGS 训练数据
        cls.TRAINING_DATA_DIR = f'{base}/3dgs_training_data'
        cls.TRAINING_IMAGES = f'{base}/3dgs_training_data/images'
        cls.TRAINING_SPARSE = f'{base}/3dgs_training_data/sparse/0'
        cls.TRAINING_TRANSFORMS = f'{base}/3dgs_training_data/transforms.json'

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


# 可读取的输入图像扩展名 / 已是标准(COLMAP+下游都能直接用)的扩展名
READABLE_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}
STD_IMAGE_EXTS = {'.png', '.jpg', '.jpeg'}
CACHE_VERSION = 2


def list_readable_images(image_dir):
    """列出原始输入中可参与预处理的图像文件。"""
    path = Path(image_dir)
    if not path.exists():
        return []
    return sorted(
        f for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in READABLE_IMAGE_EXTS
    )


def list_standard_images(image_dir):
    """列出 COLMAP / 训练可直接读取的标准图像文件。"""
    path = Path(image_dir)
    if not path.exists():
        return []
    return sorted(
        f for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in STD_IMAGE_EXTS
    )


def input_signature(image_dir):
    """生成输入目录指纹；图片没变时可安全复用缓存。"""
    root = Path(image_dir).resolve()
    files = []
    for f in list_readable_images(root):
        st = f.stat()
        files.append({
            'name': f.name,
            'size': st.st_size,
            'mtime_ns': st.st_mtime_ns,
        })
    return {
        'version': CACHE_VERSION,
        'root': str(root),
        'files': files,
    }


def expected_normalized_names(src_dir):
    """按规范化规则推导输出 PNG 文件名，用于判断旧转换结果能否复用。"""
    used = set()
    names = []
    for f in list_readable_images(src_dir):
        base = re.sub(r'[^A-Za-z0-9_-]', '_', f.stem)
        out_name = base + '.png'
        k = 1
        while out_name in used:
            out_name = f"{base}_{k}.png"
            k += 1
        used.add(out_name)
        names.append(out_name)
    return names


def cache_path():
    return Path(PathConfig.OUTPUT_BASE_DIR) / 'preprocess_cache.json'


def load_cache():
    path = cache_path()
    if not path.exists():
        return {'version': CACHE_VERSION}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data.get('version') == CACHE_VERSION:
            return data
    except Exception:
        pass
    return {'version': CACHE_VERSION}


def save_cache(data):
    data['version'] = CACHE_VERSION
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def pipeline_key(input_sig, feature_method, matcher_method, undistort_method):
    return {
        'input': input_sig,
        'feature_method': feature_method,
        'matcher_method': matcher_method,
        'undistort_method': undistort_method,
    }


def update_cache_section(section, key, **payload):
    data = load_cache()
    data[section] = {'key': key, **payload}
    save_cache(data)


def cache_key_matches(section, key):
    return load_cache().get(section, {}).get('key') == key


def remove_path(path):
    """删除一次运行生成的旧产物，避免 stale COLMAP 模型污染新输入。"""
    p = Path(path)
    if p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()


def count_colmap_images(sparse_dir):
    images_txt = Path(sparse_dir) / 'images.txt'
    if not images_txt.exists():
        return 0
    count = 0
    with open(images_txt, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 10:
                count += 1
    return count


def count_points(points_file):
    path = Path(points_file)
    if not path.exists():
        return 0
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return len([line for line in f if not line.startswith('#') and line.strip()])


def sparse_complete(sparse_dir):
    path = Path(sparse_dir)
    return all((path / name).exists() for name in ('cameras.txt', 'images.txt', 'points3D.txt'))


def training_data_complete():
    images_dir = Path(PathConfig.TRAINING_IMAGES)
    sparse_dir = Path(PathConfig.TRAINING_SPARSE)
    if not images_dir.exists() or not sparse_complete(sparse_dir):
        return False
    return len(list_standard_images(images_dir)) > 0


def normalized_cache_valid(src_dir, dst_dir, input_sig):
    dst = Path(dst_dir)
    names = expected_normalized_names(src_dir)
    if not names or not dst.exists():
        return False

    entry = load_cache().get('normalized', {})
    if entry.get('key') == {'input': input_sig} and entry.get('files') == names:
        return all((dst / name).exists() for name in names)

    # 兼容旧版本已生成但还没有 manifest 的工作区：文件名一致且输出不旧于原图即可接管缓存。
    src_files = list_readable_images(src_dir)
    for src, name in zip(src_files, names):
        out = dst / name
        if not out.exists() or out.stat().st_mtime_ns < src.stat().st_mtime_ns:
            return False
    if len(list_standard_images(dst)) != len(names):
        return False
    update_cache_section('normalized', {'input': input_sig}, count=len(names), files=names)
    return True


def stage1_cache_valid(key, expected_count):
    sparse_dir = Path(PathConfig.COLMAP_SPARSE) / '0'
    if not sparse_complete(sparse_dir) or not Path(PathConfig.ESTIMATED_CALIB).exists():
        return False
    image_count = count_colmap_images(sparse_dir)
    if image_count != expected_count:
        return False
    if cache_key_matches('stage1', key):
        return True
    update_cache_section('stage1', key, image_count=image_count)
    return True


def undistort_cache_valid(key, expected_count):
    image_count = len(list_standard_images(PathConfig.UNDISTORTED_IMAGES))
    if image_count != expected_count or not Path(PathConfig.UNDISTORTED_CAMERAS).exists():
        return False
    if cache_key_matches('undistort', key):
        return True
    update_cache_section('undistort', key, image_count=image_count)
    return True


def hloc_cache_valid(key):
    sparse_dir = Path(PathConfig.HLOC_SPARSE_DIR)
    points_file = sparse_dir / 'points3D.txt'
    if not sparse_complete(sparse_dir) or count_points(points_file) == 0:
        return False
    point_count = count_points(points_file)
    if cache_key_matches('hloc', key):
        return True
    update_cache_section('hloc', key, point_count=point_count)
    return True


def training_cache_valid(key, expected_count):
    if not training_data_complete():
        return False
    image_count = len(list_standard_images(PathConfig.TRAINING_IMAGES))
    point_count = count_points(Path(PathConfig.TRAINING_SPARSE) / 'points3D.txt')
    if image_count != expected_count or point_count == 0:
        return False
    if cache_key_matches('training', key):
        return True
    update_cache_section('training', key, image_count=image_count, point_count=point_count)
    return True


def clean_generated_outputs(stage):
    """按运行阶段清理派生产物。

    input_normalized 会由 normalize_input_images 自己刷新；这里不删除它。
    """
    if stage in ('1', 'all'):
        remove_path(PathConfig.COLMAP_SFM_DIR)

    if stage in ('2', 'all'):
        for path in (
            PathConfig.UNDISTORT_OUTPUT,
            PathConfig.HLOC_INPUT_DIR,
            PathConfig.HLOC_OUTPUT_DIR,
            Path(PathConfig.OUTPUT_BASE_DIR) / 'hloc_outputs_multicam',
            PathConfig.TRAINING_DATA_DIR,
        ):
            remove_path(path)


def cuda_unavailable(stderr):
    return 'no CUDA-capable device is detected' in (stderr or '')


def _load_image_rgb_uint8(path):
    """稳健读取一张图（含 16bit TIFF / 非标准格式），返回 HxWx3 uint8 RGB。失败返回 None。"""
    # 优先 PIL（对位深/模式处理更可控）
    try:
        from PIL import Image
        im = Image.open(str(path))
        if im.mode in ('I', 'I;16', 'I;16B', 'I;16L', 'F'):
            arr = np.asarray(im).astype(np.float32)
            if arr.max() > 255:                 # 16bit → 8bit（整体缩放，保持各图一致性）
                arr = arr / 256.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            return arr
        return np.asarray(im.convert('RGB'))
    except Exception:
        pass
    # 回退 OpenCV（libtiff 等）
    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.dtype != np.uint8:               # 16bit 等 → 8bit
            img = np.clip(img.astype(np.float32) / 256.0, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception:
        return None


def normalize_input_images(src_dir, dst_dir, input_sig=None):
    """把输入图像统一规范化为 PNG（处理 tif/16bit/非标准命名）。

    - 若目录里全是 png/jpg，则原样返回，不做多余转换。
    - 否则把所有可读图像转成 PNG 写入 dst_dir，文件名去掉内部的点。
    返回 (可用图像目录, 图像数量)。
    """
    from PIL import Image  # noqa: F401  仅确保 Pillow 可用
    src = Path(src_dir)
    if not src.exists():
        return src_dir, 0
    files = sorted(f for f in src.iterdir()
                   if f.is_file() and f.suffix.lower() in READABLE_IMAGE_EXTS)
    if not files:
        return src_dir, 0
    if all(f.suffix.lower() in STD_IMAGE_EXTS for f in files):
        return src_dir, len(files)   # 已是标准格式，无需转换

    if input_sig and normalized_cache_valid(src_dir, dst_dir, input_sig):
        print(f"♻️ 复用已规范化图像: {dst_dir}（{len(files)} 张）")
        return str(Path(dst_dir)), len(files)

    dst = Path(dst_dir)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"🖼️ 规范化 {len(files)} 张输入图像为 PNG（含 tif/16bit）…")
    n = 0
    used = set()
    out_files = []
    from PIL import Image
    for f in files:
        arr = _load_image_rgb_uint8(f)
        if arr is None:
            print(f"  ⚠️ 无法读取，已跳过: {f.name}")
            continue
        # 清理文件名：点/空格/特殊字符 → 下划线（COLMAP images.txt 以空格分隔，名字不能含空格）
        base = re.sub(r'[^A-Za-z0-9_-]', '_', f.stem)
        out_name = base + '.png'
        k = 1
        while out_name in used:                 # 去重，避免清理后重名互相覆盖
            out_name = f"{base}_{k}.png"; k += 1
        used.add(out_name)
        Image.fromarray(arr).save(str(dst / out_name), 'PNG')
        out_files.append(out_name)
        n += 1
    print(f"✅ 规范化完成: {n}/{len(files)} 张 → {dst}")
    if input_sig:
        update_cache_section('normalized', {'input': input_sig}, count=n, files=out_files)
    return str(dst), n


def stage1_colmap_sfm_calibration(feature_method='superpoint', matcher_method='superglue', cache_key=None):
    """阶段1: 使用COLMAP SfM进行自动标定"""
    
    print("\n=== 阶段1: COLMAP SfM自动标定 ===")
    
    # 检查输入
    images_dir = Path(PathConfig.ORIGIN_IMAGES)
    if not images_dir.exists():
        print("❌ 输入图像目录不存在")
        return False
    
    # 统计图像数量
    image_files = list_standard_images(images_dir)
    print(f"📷 检测到图像: {len(image_files)} 张")

    if cache_key and stage1_cache_valid(cache_key, len(image_files)):
        print(f"♻️ 复用COLMAP SfM结果: {PathConfig.COLMAP_SFM_DIR}")
        with open(PathConfig.ESTIMATED_CALIB, 'r', encoding='utf-8') as f:
            calib_data = json.load(f)
        cameras_count = len(calib_data) if isinstance(calib_data, list) else len(calib_data['Calibration']['cameras'])
        print(f"📊 SfM统计:")
        print(f"  - 估计相机数: {cameras_count} 台")
        print(f"  - 原始图像: {len(image_files)} 张")
        print(f"  - 重建成功率: {cameras_count/len(image_files)*100:.1f}%")
        return True

    remove_path(PathConfig.COLMAP_SFM_DIR)
    
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
        if result.returncode != 0 and cuda_unavailable(result.stderr):
            print("⚠️ CUDA不可用，改用CPU重新提取COLMAP特征")
            remove_path(PathConfig.COLMAP_DATABASE)
            feature_extract_cmd += ['--SiftExtraction.use_gpu', '0']
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
        if result.returncode != 0 and cuda_unavailable(result.stderr):
            print("⚠️ CUDA不可用，改用CPU重新进行COLMAP特征匹配")
            feature_match_cmd += ['--SiftMatching.use_gpu', '0']
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

        if cache_key:
            update_cache_section('stage1', cache_key, image_count=len(image_files), camera_count=cameras_count)
        
        return True
        
    except FileNotFoundError:
        print("❌ COLMAP未安装，请先安装COLMAP")
        print("安装指南: https://colmap.github.io/install.html")
        return False
    except Exception as e:
        print(f"❌ COLMAP SfM处理出错: {e}")
        return False


def stage2_undistort_with_calibration(undistort_method='custom', cache_key=None):
    """阶段2: 使用估计的标定文件进行去畸变"""
    
    print("\n=== 阶段2: 基于估计标定的去畸变 ===")
    
    # 检查估计标定文件
    if not Path(PathConfig.ESTIMATED_CALIB).exists():
        print("❌ 估计标定文件不存在，请先运行阶段1")
        return False

    expected_count = len(list_standard_images(PathConfig.ORIGIN_IMAGES))
    if cache_key and undistort_cache_valid(cache_key, expected_count):
        print(f"♻️ 复用去畸变结果: {PathConfig.UNDISTORTED_IMAGES}（{expected_count} 张）")
        return True

    for path in (
        PathConfig.UNDISTORT_OUTPUT,
        PathConfig.HLOC_INPUT_DIR,
        PathConfig.HLOC_OUTPUT_DIR,
        Path(PathConfig.OUTPUT_BASE_DIR) / 'hloc_outputs_multicam',
        PathConfig.TRAINING_DATA_DIR,
    ):
        remove_path(path)
    
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
        image_count = len(list_standard_images(undist_dir))
        print(f"✅ 生成去畸变图像: {image_count} 张")
        if image_count == 0:
            print("❌ 去畸变图像目录为空")
            return False
        
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
        
        image_count = len(list_standard_images(undist_dir))
        print(f"✅ 生成去畸变图像: {image_count} 张")
        if image_count == 0:
            print("❌ COLMAP去畸变没有生成可用图像，请检查 sparse 图像名是否与输入目录一致")
            return False
        
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

    if cache_key:
        update_cache_section(
            'undistort',
            cache_key,
            image_count=len(list_standard_images(PathConfig.UNDISTORTED_IMAGES)),
        )
    
    return True


def step2_hloc_reconstruction(feature_method='superpoint', matcher_method='superglue', cache_key=None):
    """步骤2: 使用去畸变图像进行hloc重建"""
    
    print("\n=== 步骤2: hloc点云重建 ===")
    
    # 检查输入
    undist_dir = Path(PathConfig.UNDISTORTED_IMAGES)
    cameras_file = Path(PathConfig.UNDISTORTED_CAMERAS)
    
    if not undist_dir.exists() or not cameras_file.exists():
        print("❌ 缺少去畸变输入文件")
        return False

    if cache_key and hloc_cache_valid(cache_key):
        point_count = count_points(Path(PathConfig.HLOC_SPARSE_DIR) / 'points3D.txt')
        print(f"♻️ 复用hloc点云重建结果: {PathConfig.HLOC_OUTPUT_DIR}")
        print(f"✅ 3D点数量: {point_count}")
        return True

    for path in (
        PathConfig.HLOC_INPUT_DIR,
        PathConfig.HLOC_OUTPUT_DIR,
        Path(PathConfig.OUTPUT_BASE_DIR) / 'hloc_outputs_multicam',
    ):
        remove_path(path)
    
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
        return False
    
    # 统计点云数量
    with open(points_file, 'r') as f:
        point_count = len([line for line in f if not line.startswith('#') and line.strip()])
    
    print(f"✅ 生成3D点数量: {point_count}")
    
    if point_count < 1000:
        print("⚠️ 点云数量较少，可能影响3DGS训练效果")
    elif point_count > 10000:
        print("✅ 点云密度充足，适合3DGS训练")

    if cache_key:
        update_cache_section('hloc', cache_key, point_count=point_count)
    
    return True


def step3_prepare_3dgs_data(use_colmap_sparse=False, cache_key=None, expected_count=None):
    """步骤3: 准备3DGS训练数据"""
    
    print("\n=== 步骤3: 准备3DGS训练数据 ===")

    if cache_key and expected_count and training_cache_valid(cache_key, expected_count):
        print(f"♻️ 复用完整3DGS训练数据: {PathConfig.TRAINING_DATA_DIR}")
        return True
    
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
    training_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制去畸变图像
    images_dir = Path(PathConfig.TRAINING_IMAGES)
    if images_dir.exists():
        shutil.rmtree(images_dir)
    
    shutil.copytree(PathConfig.UNDISTORTED_IMAGES, images_dir)
    print(f"✅ 复制训练图像到: {images_dir}")
    image_count = len(list_standard_images(images_dir))
    if image_count == 0:
        print("❌ 训练图像目录为空")
        return False
    
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
    if cache_key:
        update_cache_section(
            'training',
            cache_key,
            image_count=image_count,
            point_count=count_points(sparse_dir / 'points3D.txt'),
            source='colmap' if use_colmap_sparse else 'hloc',
        )
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
        image_count = len(list_standard_images(images_dir))
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
    parser.add_argument('--input-images', type=str, default=None,
                       help='输入图像目录（覆盖默认路径，供 App 传入用户选择的文件夹）')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出根目录（覆盖默认路径，所有派生路径基于此重算）')

    args = parser.parse_args()

    # 若传入了自定义输入/输出路径，则覆盖 PathConfig 默认值并重算派生路径
    if args.input_images or args.output_dir:
        PathConfig.configure(args.input_images, args.output_dir)

    # 使用命令行参数或默认值
    feature_method = args.feature_method
    matcher_method = args.matcher_method
    undistort_method = args.undistort_method
    original_input_images = PathConfig.ORIGIN_IMAGES
    original_input_sig = input_signature(original_input_images)

    # 规范化输入图像（tif/16bit/非标准命名 → PNG），保证 COLMAP 与 hloc 都能读取
    norm_dir = str(Path(PathConfig.OUTPUT_BASE_DIR) / 'input_normalized')
    new_src, n_imgs = normalize_input_images(PathConfig.ORIGIN_IMAGES, norm_dir, original_input_sig)
    if new_src != PathConfig.ORIGIN_IMAGES:
        PathConfig.ORIGIN_IMAGES = new_src
        print(f"📷 使用规范化后的图像目录: {new_src}（{n_imgs} 张）")

    run_key = pipeline_key(original_input_sig, feature_method, matcher_method, undistort_method)
    expected_count = len(list_standard_images(PathConfig.ORIGIN_IMAGES))

    print("=== 3DGS训练数据完整预处理管线（两阶段处理）===")
    print("阶段1: COLMAP SfM自动标定 → 阶段2: 基于标定文件的3DGS数据准备")

    # 打印路径配置
    PathConfig.print_config()
    print()
    
    # 检查输入图像目录
    if not Path(PathConfig.ORIGIN_IMAGES).exists():
        print("❌ 输入图像目录不存在")
        return False
    
    print(f"🔧 特征提取: {feature_method.upper()}")
    print(f"🔗 特征匹配: {matcher_method.upper()}")
    print(f"🔧 去畸变方法: {undistort_method}")
    
    success = True
    used_full_cache = False

    if args.stage == 'all' and training_cache_valid(run_key, expected_count):
        print(f"♻️ 检测到完整缓存，直接复用训练数据: {PathConfig.TRAINING_DATA_DIR}")
        used_full_cache = True
    
    # 阶段1: COLMAP SfM自动标定
    if not used_full_cache and args.stage in ['1', 'all'] and success:
        success = stage1_colmap_sfm_calibration(feature_method, matcher_method, run_key)
    
    # 阶段2: 基于标定文件的处理
    if not used_full_cache and args.stage in ['2', 'all'] and success:
        # 2.1 去畸变
        success = stage2_undistort_with_calibration(undistort_method, run_key)
        
        # 2.2 hloc重建。若 hloc 未生成点云，COLMAP 去畸变模式可回退到 COLMAP sparse。
        if success:
            hloc_ok = step2_hloc_reconstruction(feature_method, matcher_method, run_key)
            if hloc_ok:
                success = step3_prepare_3dgs_data(
                    use_colmap_sparse=False,
                    cache_key=run_key,
                    expected_count=expected_count,
                )
            elif undistort_method == 'colmap':
                print("⚠️ hloc未生成完整点云，回退使用COLMAP sparse数据准备3DGS训练数据")
                success = step3_prepare_3dgs_data(
                    use_colmap_sparse=True,
                    cache_key=run_key,
                    expected_count=expected_count,
                )
            else:
                success = False
        
    # 验证结果：stage 1 只生成 SfM/标定，不要求完整训练数据。
    if success and args.stage in ['2', 'all']:
        success = verify_training_data()

    if success:
        if args.stage == '1':
            print("\n" + "="*50)
            print("🎉 COLMAP SfM自动标定完成!")
            print(f"\n📁 SfM结果: {PathConfig.COLMAP_SFM_DIR}")
            print(f"📋 估计标定: {PathConfig.ESTIMATED_CALIB}")
            print("\n🚀 下一步: 运行阶段2或完整流程准备3DGS训练数据")
            return True
        
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
    return success


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
