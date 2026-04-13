# 4DGS Data Process

多相机 3D Gaussian Splatting 数据处理工具集，用于从视频到 3DGS 训练数据的完整预处理流程。

## 📋 目录

- [功能特性](#功能特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [详细使用](#详细使用)
- [工作流程](#工作流程)
- [目录结构](#目录结构)
- [依赖项](#依赖项)

## ✨ 功能特性

- 🎥 **视频帧提取**：支持多相机视频批量提帧，可指定帧率、时间范围
- 📸 **首帧管理**：自动提取和复制各相机的首帧图像
- 🔧 **图像去畸变**：支持 SuperGlue 和 COLMAP 两种去畸变方案
- 📐 **多流程支持**：Agisoft、COLMAP、SuperGlue 等多种处理管线
- 🔄 **自动化流程**：完整的从视频到 3DGS 训练数据转换

## 🚀 安装

### 1. 克隆仓库（包含子模块）

```bash
git clone --recurse-submodules https://github.com/liangfcsu/4DGSDataProcess.git
cd 4DGSDataProcess
```

如果已经克隆但没有子模块，运行：
```bash
git submodule update --init --recursive
```

### 2. 安装依赖

```bash
conda create -n dataprocess python=3.8
conda activate dataprocess
pip install opencv-python numpy tqdm
```

### 3. 安装第三方工具（可选）

根据需要安装以下工具：
- **COLMAP**: [安装指南](https://colmap.github.io/install.html)
- **Agisoft Metashape**: 商业软件
- **SuperGlue**: 已包含在 `Hierarchical-Localization` 子模块中

## ⚡ 快速开始

### 基础工作流程

```bash
# 1. 从视频提取帧（所有帧）
python videoprocess_tool/1.0extract_frames_from_videos.py \
  --video-dir data/videos \
  --output-dir data/images

# 2. 提取首帧用于特征匹配
python videoprocess_tool/1.1extract_first_frames.py \
  --video-dir data/videos \
  --output-dir data/first_frames

# 3. 复制首帧到指定目录
python videoprocess_tool/2.0copy_first_images.py \
  --data-dir data/images \
  --output-dir data/first_images

# 4. 去畸变处理（使用 SuperGlue 估计参数）
python videoprocess_tool/3.0undistort_all_frames_batch.py
```

## 📖 详细使用

### 1. 视频帧提取工具

#### 1.0 批量提取所有帧

```bash
python videoprocess_tool/1.0extract_frames_from_videos.py \
  --video-dir <视频目录> \
  --output-dir <输出目录> \
  --fps <提取帧率> \
  --start-second <起始秒数> \
  --end-second <结束秒数> \
  --max-frames <最大帧数>
```

**参数说明：**
- `--video-dir`: 视频文件目录（支持 `cam00.mp4`, `cam001.mp4` 等格式）
- `--output-dir`: 输出根目录，将创建 `cam000/`, `cam001/` 等子目录
- `--fps`: 提取帧率（留空则提取所有帧）
- `--start-second`: 从第几秒开始提取（默认从 0 秒开始）
- `--end-second`: 提取到第几秒（默认到视频结尾）
- `--max-frames`: 每个视频最多提取多少帧（留空则不限制）

**示例：**
```bash
# 提取 2-6 秒之间的所有帧
python videoprocess_tool/1.0extract_frames_from_videos.py \
  --video-dir publicdata/coffee_martini/videos \
  --output-dir publicdata/coffee_martini/images \
  --start-second 2 \
  --end-second 6

# 以 5 FPS 提取前 100 帧
python videoprocess_tool/1.0extract_frames_from_videos.py \
  --video-dir data/videos \
  --output-dir data/images \
  --fps 5 \
  --max-frames 100
```

#### 1.1 提取首帧

```bash
python videoprocess_tool/1.1extract_first_frames.py \
  --video-dir <视频目录> \
  --output-dir <输出目录>
```

从每个视频提取第一帧，输出为 `cam000.png`, `cam001.png` 等单张图片。

### 2. 首帧复制工具

```bash
python videoprocess_tool/2.0copy_first_images.py \
  --data-dir <图像目录> \
  --output-dir <输出目录> \
  --start-cam <起始相机号> \
  --end-cam <结束相机号>
```

从每个相机文件夹（如 `cam000/`, `cam001/`）中复制第一张图片到指定目录。

**示例：**
```bash
python videoprocess_tool/2.0copy_first_images.py \
  --data-dir publicdata/coffee_martini/images \
  --output-dir publicdata/coffee_martini/first_images
```

### 3. 去畸变工具

#### 3.0 使用 SuperGlue 估计参数去畸变

```bash
python videoprocess_tool/3.0undistort_all_frames_batch.py
```

自动读取 SuperGlue 估计的相机参数（`estimated_calib.json`），对所有帧进行去畸变处理。

**默认路径配置：**
- 输入：`publicdata/coffee_martini_files/coffee_martini_images_origin`
- 标定文件：`publicdata/coffee_martini_files/superglue/colmap_sfm/estimated_calib.json`
- 输出：`publicdata/coffee_martini_files/undistorted_all_frames_complete_superglue`

#### 3.1 使用 COLMAP 去畸变

```bash
python videoprocess_tool/3.1undistort_all_frames_batch_colmap.py
```

使用 COLMAP 的 `image_undistorter` 进行去畸变。

### 4. 数据处理脚本

#### COLMAP 处理流程

```bash
# 转换 COLMAP 模型格式
python scripts/colmap_process_scripts/convert_model_bin_to_txt.py

# 格式转换
python scripts/colmap_process_scripts/convert.py
```

#### SuperGlue 无标定流程

```bash
# 完整的 3DGS 管线
python scripts/self_process_scripts_superglue_noncalib/complete_3dgs_pipeline.py
```

这是一个集成脚本，包含：
1. 图像去畸变（用于 HLOC）
2. SuperGlue 特征匹配
3. COLMAP SfM 重建
4. 点云生成
5. 参数转换为 3DGS 格式

## 🔄 工作流程

### 完整流程示例（以 coffee_martini 数据集为例）

```bash
# 步骤 1: 提取视频帧（2-6 秒）
python videoprocess_tool/1.0extract_frames_from_videos.py \
  --video-dir publicdata/coffee_martini/videos \
  --output-dir publicdata/coffee_martini/images_origin \
  --start-second 2 \
  --end-second 6

# 步骤 2: 提取首帧用于 SuperGlue 匹配
python videoprocess_tool/1.1extract_first_frames.py \
  --video-dir publicdata/coffee_martini/videos \
  --output-dir publicdata/coffee_martini/first_frames

# 步骤 3: 运行 SuperGlue 无标定完整流程
python scripts/self_process_scripts_superglue_noncalib/complete_3dgs_pipeline.py

# 步骤 4: 使用估计参数对所有帧去畸变
python videoprocess_tool/3.0undistort_all_frames_batch.py

# 现在 undistorted_all_frames_complete_superglue/ 目录下的数据
# 可直接用于 3DGS 训练
```

## 📁 目录结构

```
4DGSDataProcess/
├── videoprocess_tool/           # 视频处理工具集
│   ├── 1.0extract_frames_from_videos.py    # 批量提帧
│   ├── 1.1extract_first_frames.py          # 提取首帧
│   ├── 2.0copy_first_images.py             # 复制首帧
│   ├── 3.0undistort_all_frames_batch.py    # SuperGlue 去畸变
│   └── 3.1undistort_all_frames_batch_colmap.py  # COLMAP 去畸变
│
├── scripts/                     # 数据处理脚本
│   ├── colmap_process_scripts/           # COLMAP 相关
│   ├── self_process_scripts_agisoft/     # Agisoft 流程
│   ├── self_process_scripts_superglue/   # SuperGlue 有标定流程
│   └── self_process_scripts_superglue_noncalib/  # SuperGlue 无标定流程
│       ├── complete_3dgs_pipeline.py     # 完整管线
│       ├── Hierarchical-Localization/    # HLOC (子模块)
│       └── tool/
│           ├── convert_colmap_to_calib.py
│           ├── generate_pointcloud_multicam.py
│           └── undistort_for_hloc.py
│
├── .gitignore
├── .gitmodules
└── README.md
```

## 📦 依赖项

### Python 依赖
```
opencv-python >= 4.5.0
numpy >= 1.19.0
tqdm >= 4.60.0
```

### 外部工具（可选）
- **COLMAP** (v3.6+): 用于 SfM 重建
- **Agisoft Metashape**: 商业摄影测量软件
- **SuperGlue**: 基于学习的特征匹配（已包含在子模块中）

## 📝 注意事项

1. **视频命名格式**：支持 `cam00.mp4`, `cam001.mp4`, `001.mp4` 等格式
2. **相机编号**：自动从文件名提取编号，支持 0 开始或 1 开始
3. **输出目录**：统一使用 3 位数编号（`cam000`, `cam001` 等）
4. **子模块**：首次克隆后需确保子模块已初始化

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目仅供学术研究使用。

## 🔗 相关链接

- [COLMAP](https://colmap.github.io/)
- [SuperGlue/SuperPoint](https://github.com/magicleap/SuperGluePretrainedNetwork)
- [Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)

---

**维护者**: [@liangfcsu](https://github.com/liangfcsu)  
**最后更新**: 2026年4月13日
