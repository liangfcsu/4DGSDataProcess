# 多相机视频一键转4DGS数据集

该目录是独立工具包，已集中放置流水线代码、HLoc/SuperPoint/SuperGlue代码和权重。运行时仍需要系统中已安装的COLMAP和具备OpenCV、PyTorch、h5py、pycolmap的Python环境。入口脚本会自动优先检测当前环境，然后尝试本机的`3dgslf`环境。

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

成功时中间产物会自动删除。失败时会保留在输出目录同级的`.<output-name>.work`中；修复问题后，使用原命令并加`--resume`续跑。如果要放弃旧中间结果或替换已有输出，显式加`--overwrite`。

## 多GPU分配

`--gpus auto`会检测所有NVIDIA GPU。逐帧稀疏点云按连续帧区间均分；例如100帧/4卡会自动分为每卡25帧。提帧使用的是OpenCV CPU解码，不会伪装成GPU解码，但会默认按GPU数启动多进程、按相机分片并行提取。

