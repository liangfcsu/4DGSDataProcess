# gs_pipeline — 统一多相机 → 4DGS 数据处理流水线

把仓库里四个历史脚本目录合并成**一条流水线、一个入口**，一条命令覆盖
「输入 × 标定」的全部组合。

| | **无标定**（自动 SfM） | **rig 标定**（refined_rig_group.json） | **libCalib 标定**（calib*.json） |
|---|---|---|---|
| **多相机视频** (`camXX.mp4`) | ✅ | ✅ | ✅ |
| **多相机多帧图像** (`camXXX/` 每台一段序列) | ✅ | ✅ | ✅ |
| **多相机单帧图像** (扁平目录，每张一台相机) | ✅ | ✅ | ✅ |

视频与多帧图像还可加 `--frame N` **只处理第 N 帧**（视频第 N 帧 / 序列第 N 张），
产出单帧数据集，同样与三种标定模式自由组合。

## 一条命令

```bash
cd scripts/gs_pipeline

# 多相机视频 + 无标定（自动估计内外参）
python run.py --input ../../data/cook_spinach --output ../../data/cook_spinach_4dgs --max-frames 30

# 多相机多帧图像 + rig 标定
python run.py --input ../../data/rig_seq --output ../../data/rig_4dgs \
  --rig-json ../../data/rig_seq/refined_rig_group.json

# 多相机单帧图像 + libCalib 标定
python run.py --input ../../data/7.7/Photo --output ../../data/7.7_4dgs \
  --calib-json ../../data/7.7/calib0.3343.json

# 只处理某一帧：视频第 30 帧 / 多帧序列第 30 张（标定与否均可）
python run.py --input ../../data/videos --output ../../data/frame30_4dgs --frame 30
python run.py --input ../../data/rig_seq --output ../../data/rig_frame30 --frame 30 \
  --rig-json ../../data/rig_seq/refined_rig_group.json

# 先看清每一步会执行什么命令，不实际运行（无需 GPU）
python run.py --input ../../data/x --output ../../data/x_4dgs --dry-run
```

`--input-type` 默认 `auto`：目录里有 `*.mp4/.mov/.avi/.mkv` 走视频分支，否则走图像分支。
标定来源二选一（`--rig-json` / `--calib-json`），都不给即无标定自动 SfM。
`--frame N` 只处理指定单帧（视频第 N 帧、多帧序列第 N 张），产出单帧数据集；与三种标定模式自由组合。

## 输出

成功后 `--output` 目录只包含可直接训练的四项：

```
<output>/
├── images/       # 每台相机第 1 帧（去畸变，扁平）
├── ims/          # 全部去畸变帧 camXXX/camXXXframeYYY.png
├── persparse/    # frame{n:03d}_points3D.txt（逐帧稀疏点云）
└── sparse/0/     # 参考模型 cameras.txt / images.txt / points3D.txt
```

## 流水线四阶段

无论哪种输入/标定，都归并到同一条主干；差异只在前两个阶段的“适配器”：

1. **采集** → 统一成 `ims/camXXX/camXXXframe{n:03d}.png`
   - 视频：`extract_frames_from_videos.py` 按相机分片并行提帧
   - 图像：`camXXX/` 子目录按序重命名；扁平目录按排序索引铺成单帧
2. **标定 + 全帧去畸变** → 参考 `sparse/0`（已知位姿，空点）+ 去畸变全帧
   - **rig**：有理径向畸变模型 `(1+k1r²+k2r⁴+k3r⁶)/(1+k4r²+k5r⁴+k6r⁶)`
     逐台建映射并作用到该相机全部帧，写出 PINHOLE 参考模型（COLMAP/OpenCV 无法表达该分式模型，故保留原实现）
   - **libCalib**：`1.convert_calib_to_cameras_json.py` 得到 cameras.json，多项式去畸变 + CTW-Euler→WTC 位姿
   - **无标定**：首帧 `complete_3dgs_pipeline.py` 跑 COLMAP SfM 估参，再 `undistort_all_frames.py` 去畸变
3. **逐帧稀疏点云** → `generate_per_frame_sparse.py` 在去畸变帧上按帧区间多 GPU 三角化
4. **组装** → 生成上面的四目录，硬链接优先，成功后清理中间产物

**关键枢纽**：无论标定与否，第 2 阶段都产出统一的参考 `sparse/0`；其后的去畸变→逐帧三角化→组装完全一致。

## 参数速查

| 参数 | 说明 | 默认 |
|---|---|---|
| `--input` / `--output` | 输入目录 / 最终数据集目录（必填） | — |
| `--input-type` | `auto` / `videos` / `images` | `auto` |
| `--rig-json` / `--calib-json` | rig 标定 / libCalib 标定（互斥） | 无（=自动 SfM） |
| `--max-frames` | 每台相机最多前 N 帧；`0`=全部（仅视频） | `0` |
| `--frames-per-second` | 每秒均匀采样 N 帧（仅视频） | 按原始帧序 |
| `--frame N` | 只处理第 N 帧（视频第 N 帧 / 多帧序列第 N 张），产出单帧数据集 | 不设=全部帧 |
| `--gpus` | `auto` 或 `0,1,2,3` | `auto` |
| `--max-keypoints` / `--resize-max` / `--superglue-weights` | SuperGlue 三角化参数 | `4096` / `4000` / `indoor` |
| `--undistort-workers` | 去畸变 CPU 线程数 | `8` |
| `--rig-pose-mode` / `--rig-radial-mode` / `--rig-small-terms` / `--rig-direction` / `--rig-scale` | rig 去畸变高级项 | `wtc_center` / `direct` / `tangential_d8d9` / `ideal_to_observed` / `1.0` |
| `--overwrite` / `--keep-intermediate` / `--dry-run` | 覆盖已有 / 保留中间 / 仅打印计划 | — |

## 依赖

- **Python**：`numpy opencv-python torch(+CUDA) h5py pycolmap tqdm Pillow packaging`
  （入口自动探测当前环境，缺依赖时可 `--python /abs/bin/python` 指定）
- **COLMAP** 可执行文件在 PATH 上
- **SuperPoint/SuperGlue 权重**：随 `multicam_4dgs_pipeline/Hierarchical-Localization` 提供

## 目录结构与合并说明

本目录是 `multicam_4dgs_pipeline`、`self_process_scripts_superglue`（含 rig）、
`self_process_scripts_superglue_noncalib`、`refined_rig_superglue` 四者合并后的**单一自包含项目**：

```
gs_pipeline/
├── run.py              # 唯一入口，按 输入×标定 分派
├── pipeline/           # 采集/标定/去畸变/逐帧点云/组装 的调度与粘合
│   ├── common.py       # 命名、发现、子进程、环境探测
│   ├── acquire.py      # 输入 → ims/camXXX/camXXXframeYYY.png
│   ├── calibrated.py   # rig(有理径向) + libCalib(多项式) 去畸变 + 参考模型（纯 numpy/cv2）
│   ├── noncalib.py     # 首帧 SfM 自动标定 + 去畸变（调 engine）
│   ├── sparse.py       # 逐帧多 GPU 三角化（调 engine）
│   └── assemble.py     # 组装 images/ ims/ persparse/ sparse/0/
├── stages/             # libCalib calib*.json → cameras.json
└── engine/             # 重建引擎 + 唯一一份 Hierarchical-Localization（含 SuperGlue 权重）
```

合并要点：

- 重的重建能力（提帧、首帧 SfM、去畸变、逐帧三角化）来自原 `multicam_4dgs_pipeline`，
  整体移入 `engine/`，脚本与其内置 HLOC 的相对关系不变，行为未改动。
- rig 有理径向畸变数学（`refined_rig_superglue`）与 rig 位姿合成、libCalib 位姿转换
  已原样/等价内联到 `pipeline/calibrated.py`，不再依赖任何外部目录。
- 三份重复的 `Hierarchical-Localization` 合并为 `engine/` 下的**唯一一份**；
  原四个目录中的另外三个已删除（内容全部被本项目取代，可从 git 历史恢复）。

## 正确性验证

- **rig 标定**：本流水线产出的参考位姿/内参与既有 `refined_rig_superglue` 输出逐条一致
  （最大位姿误差 5e-11，内参完全相同，在 `data/7.7` 64 相机上验证）。
- **libCalib 位姿**：CTW-Euler→COLMAP WTC 转换对 3000 组随机位姿与矩阵法真值一致
  （四元数误差 3e-16，平移误差 4e-15）。
