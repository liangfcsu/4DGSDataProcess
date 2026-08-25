# 多相机视频一键转4DGS数据集

该目录是独立工具包，已集中放置流水线代码、HLoc/SuperPoint/SuperGlue代码和权重，不依赖仓库里的`videoprocess_tool`或其他处理脚本目录。运行时仍需要系统中已安装的COLMAP，以及具备CUDA版PyTorch、OpenCV、h5py和pycolmap等依赖的Python环境。入口脚本会优先检测当前环境，再尝试本机常用的4DGS Conda环境。

## 在本目录直接使用

```bash
cd scripts/multicam_4dgs_pipeline

# 可选：先检查代码、权重、Python依赖、CUDA和COLMAP
python check_bundle.py

# 前30帧验证（路径相对于当前目录）
python run_pipeline.py \
  --video-dir ../../data/cook_spinach \
  --output-dir ../../data/cook_spinach_4dgs_test \
  --max-frames 30 \
  --gpus auto
```

如果当前Python环境依赖不完整，可先激活已有环境，或者给入口增加`--python /绝对路径/bin/python`。`requirements.txt`仅包含Python包；COLMAP和适配CUDA版本的PyTorch仍应按当前机器环境安装。

工具包内包含：

```text
multicam_4dgs_pipeline/
├── run_pipeline.py                 # 唯一主入口
├── check_bundle.py                 # 独立性和环境自检
├── extract_frames_from_videos.py   # 多相机提帧、逐帧续传
├── extract_first_frames.py         # 提取标定首帧
├── complete_3dgs_pipeline.py       # 内外参估计和首帧重建
├── undistort_all_frames.py         # 全相机全帧去畸变
├── generate_per_frame_sparse.py    # 多GPU逐帧稀疏点云
├── tool/                           # 标定和重建辅助代码
└── Hierarchical-Localization/      # HLoc、SuperPoint/SuperGlue及三份权重
```

## cook_spinach前30帧验证

```bash
python scripts/multicam_4dgs_pipeline/run_pipeline.py \
  --video-dir data/cook_spinach \
  --output-dir data/cook_spinach_4dgs_test \
  --max-frames 30 \
  --gpus auto
```

输入`cam00.mp4 ~ cam20.mp4`会保持为`cam000 ~ cam020`。前30帧是视频的原始前30帧，不是在整段视频中均匀抽取的30帧。

## 全量处理

前30帧验证通过后运行：

```bash
python scripts/multicam_4dgs_pipeline/run_pipeline.py \
  --video-dir data/cook_spinach \
  --output-dir data/cook_spinach_4dgs \
  --max-frames 0 \
  --gpus auto
```

`--max-frames 0`表示提取全部帧。如果要按时间降采样，可额外传入`--frames-per-second N`。

## 输出与恢复

整条流水线成功后，输出目录只包含：

```text
<output-dir>/
├── images/       # 去畸变后的各相机第1帧（扁平目录）
├── ims/          # 去畸变后的所有相机和帧
├── persparse/    # frameXXX_points3D.txt
└── sparse/0/     # 首帧的COLMAP内外参和稀疏点云
```

成功时中间产物会自动删除。失败或`Ctrl+C`中断时会保留在输出目录同级的`.<output-name>.work`中；使用原命令并加`--resume`续跑。提帧阶段会跳过已完成的相机，部分完成的相机会从已有连续帧的下一帧继续，不会删除已提取的PNG。如果要放弃旧中间结果或替换已有输出，显式加`--overwrite`。

## 多GPU分配

`--gpus auto`会检测所有NVIDIA GPU。逐帧稀疏点云按连续帧区间均分；例如100帧/4卡会自动分为每卡25帧。提帧使用的是OpenCV CPU解码，不会伪装成GPU解码，但会默认按GPU数启动多进程、按相机分片并行提取。
