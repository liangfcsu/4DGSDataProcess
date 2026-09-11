# track_sparse：原始多相机视频到稀疏 3D 轨迹

`scripts/track_sparse` 是一条独立可运行的数据处理流水线。输入可以直接是
`cam00.mp4、cam01.mp4、...` 组成的原始多相机目录；程序会在内部完成抽帧、
首帧 COLMAP SfM 内外参估计、全序列去畸变，然后继续构建带稳定 Track ID 的
时序稀疏点云。不再要求用户先手工运行 `gs_pipeline`。

本实现复用了同仓库 `scripts/gs_pipeline` 中已经验证过的底层抽帧、标定和去畸变
实现，但流程编排、缓存、参数和最终入口都位于 `scripts/track_sparse`。它不会运行
`gs_pipeline` 的逐帧 COLMAP 重建与 4DGS 数据集组装阶段。

## 0. 当前实现与快速开始

当前实现遵循本文与 `稀疏点云轨迹系统_实现指导.md` 的核心原则：使用 SuperPoint/SuperGlue 建立同帧跨相机空间边和同相机跨帧时间边，为二维 Observation 分配稳定的全局 Track ID，然后在每一帧用当前可见的多视角观测重新做鲁棒三角化。逐帧 `persparse` 只参与支持距离和置信度计算，绝不传播身份。

已实现：

- `images.txt` 空观测行、占位行和乱序 ID 的稳健 COLMAP 文本解析。
- 保留真实 `cam_id` / `frame_id` 的数据发现、缺帧检查和 1 像素编码补边告警。
- 基于标定、参考点云重叠、基线和视轴方向的固定 Camera Graph。
- 复用仓库内置 HLOC SuperPoint/SuperGlue 代码和权重，模型输出封装在 adapter 后面。
- 空间匹配置信度 + Sampson 误差过滤、同相机冲突受限的多视角组件合并。
- 两视角 RANSAC 候选、正深度、视差角、逐视角重投影剔除及 SciPy robust least-squares refinement。
- 相邻帧和短跨度时间匹配、目标 feature 唯一占用、周期出生、重复抑制和严格短时重连。
- `new/tentative/active/occluded/lost/ended` 生命周期以及 `static/dynamic/unknown` 分类。
- `tracks.h5`、逐帧 NPZ/PLY、FreeTimeGS 适配 NPZ、2D overlay、3D trajectory PLY 和统计 JSON。
- 输入/配置指纹、模型缓存复用、`--resume`、`--overwrite`、`--dry-run` 和阶段运行。

当前时间跟踪后端是跨帧 SuperGlue feature matching，并默认额外匹配 `t -> t+2` 来恢复一帧缺口。接口已隔离在 `temporal_tracker.py`；CoTracker/TAPIR 和光流双验证属于后续可替换后端，当前版本不会伪造这些模型的 visibility 或 flow confidence。

### 0.1 直接处理 `flame_steak`

完整 300 帧命令如下。`--dataset` 保留为 `--input` 的兼容别名，两种写法等价：

```bash
python3 scripts/track_sparse/run.py \
  --input data/pubulicdata/flame_steak \
  --output outputs/flame_steak_tracks \
  --config scripts/track_sparse/configs/flame_steak.yaml \
  --start-frame 1 \
  --end-frame 300 \
  --gpu 0 \
  --resume
```

`configs/flame_steak.yaml` 集中保存相机图、特征、匹配、出生、时间跟踪、三角化、
生命周期、分类和导出参数。输入输出路径、帧范围、GPU 及断点模式属于运行控制，
继续由 CLI 指定。命令行同名算法参数仍可用于临时覆盖 YAML，但正式处理建议只修改
该配置文件，以保证 `manifest.json` 中记录的参数可复现。

未提供 `--rig-json` 或 `--calib-json` 时，程序自动用所有相机的第一帧估计固定内外参。
`flame_steak/poses_bounds.npy` 只包含 LLFF 位姿/焦距信息而不包含可靠的镜头畸变模型，
默认不会把它冒充成完整标定；程序会重新执行首帧 SfM 和去畸变。

当输入是视频且没有显式给 `--max-frames` 时，`--end-frame 300` 同时作为每路视频
的抽帧上限，因此上述命令不会多抽无用帧。预处理缓存保存在
`outputs/flame_steak_tracks/preprocess/`。

### 0.2 先做小规模 smoke test

建议先让全部相机参与标定，再只用 4 台相机、20 帧验证轨迹参数：

```bash
python3 scripts/track_sparse/run.py \
  --input data/pubulicdata/flame_steak \
  --output outputs/track_sparse_smoke \
  --start-frame 1 \
  --end-frame 20 \
  --cameras 0,1,2,3 \
  --camera-neighbors 3 \
  --min-seed-views 2 \
  --max-keypoints 1024 \
  --resize-max 1024 \
  --debug-frames 1,10,20 \
  --resume
```

CLI 会先检查当前 Python 环境；本机普通 `python3` 缺少科学计算依赖时，会自动尝试项目的 `/opt/4dgs-player/env/bin/python3.11`。也可以显式传入 `--python /path/to/python`。自动标定还要求系统 `PATH` 中存在 `colmap` 可执行文件；Python 包列表见 `requirements.txt`。

先只检查原始输入并打印抽帧、标定和去畸变计划，不写文件：

```bash
python3 scripts/track_sparse/run.py \
  --input data/pubulicdata/flame_steak \
  --output <output-dir> \
  --end-frame 20 \
  --dry-run
```

只完成抽帧、标定和去畸变，不启动轨迹网络：

```bash
python3 scripts/track_sparse/run.py \
  --input data/pubulicdata/flame_steak \
  --output outputs/flame_steak_tracks \
  --end-frame 300 \
  --preprocess-only \
  --resume
```

### 0.3 已知标定与已预处理数据兼容入口

有已知 OpenCV/frames 或 libCalib 内外参时，可跳过自动 SfM：

```bash
python3 scripts/track_sparse/run.py \
  --input <raw-video-or-image-dir> \
  --calib-json <calibration.json> \
  --output <track-output> \
  --resume
```

已有 `gs_pipeline` 最终数据集仍可直接使用，程序会自动识别为 `prepared`：

```bash
python3 scripts/track_sparse/run.py \
  --input data/pubulicdata/cook_spinach_4dgsdata \
  --output data/pubulicdata/cook_spinach_tracks \
  --start-frame 1 \
  --end-frame 300 \
  --spawn-interval 10 \
  --camera-neighbors 6 \
  --min-seed-views 3 \
  --min-triangulation-views 2 \
  --max-reprojection-error 3.0 \
  --min-triangulation-angle 1.0 \
  --resume
```

直接使用 `.work` 中间目录时也可显式指定：

```bash
python3 scripts/track_sparse/run.py \
  --images-dir <work>/undistorted \
  --sparse-dir <work>/calibration/reference_sparse/0 \
  --persparse-dir <work>/persparse_shards \
  --output <track-output> \
  --resume
```

默认参数都在 `configs/default.yaml`。可用 `--config my_config.yaml` 只覆盖需要修改的层级；常用阈值也有同名 CLI 参数。第一次已有输出时必须明确使用 `--resume` 或 `--overwrite`，避免把不同配置的缓存静默混用。

`--resume` 同时校验原视频元数据、标定来源、抽帧参数、去畸变参数以及轨迹配置。
参数或输入发生变化时会拒绝混用旧缓存；确认要重算时使用 `--overwrite`。

`--stages` 会运行到所选最晚阶段，并自动补齐它之前的依赖。例如 `--stages prepare,camera_graph` 不加载模型；`--stages spawn` 会生成特征和空间匹配；完整输出使用默认的 `all`。若输入或配置指纹发生变化，`--resume` 会拒绝旧缓存并提示换输出目录或显式 `--overwrite`。

### 0.4 输出

```text
<output>/
├── preprocess/
│   ├── preprocess_manifest.json # 原始输入与预处理断点清单
│   ├── ims/camXXX/              # 从视频提取的原始帧
│   ├── undistorted/camXXX/      # 轨迹模块实际读取的去畸变帧
│   └── calibration/reference_sparse/0/
│       ├── cameras.txt
│       ├── images.txt
│       └── points3D.txt
├── manifest.json                 # 输入/配置指纹、坐标约定和阶段状态
├── camera_graph.json             # 邻接边、几何分数、验证匹配数
├── features.h5 / matches.h5      # 可断点复用的 HLOC 缓存
├── tracks.h5                     # cameras / observations / samples3d / tracks
├── tracks_summary.json           # 每轨迹摘要和全局指标
├── freetimegs_tracks.npz         # 扁平 4DGS 适配数据
├── exports/
│   ├── frameXXX_track_points.npz
│   └── frameXXX_track_points.ply
└── debug/
    ├── metrics.json
    ├── trajectories.ply
    └── overlays/camXXX_frameYYY.jpg
```

`tracks.h5/observations` 按 `(track_id, frame_id, cam_id)` 排序；`samples3d` 按 `(track_id, frame_id)` 排序。无可靠双视角几何时会保留已有 2D Observation，但 `valid_3d=false`，不会插值伪装成测量值。

### 0.5 测试

```bash
PYTHONPATH=scripts /opt/4dgs-player/env/bin/python3.11 \
  -m unittest discover -s scripts/track_sparse/tests -v
```

## 1. 核心决策

本模块不对相邻帧的独立 3D 点云做 KNN、ICP 或逐帧点 ID 匹配。长期身份建立在 2D 观测上：

```text
固定标定 + 同步多相机图像
          |
          v
同一时刻跨相机匹配，建立多视角 2D 种子
          |
          v
为种子分配全局 Track ID
          |
          v
各相机视频内做长期 2D tracking
          |
          v
每一帧用可见的多相机 2D 观测重新鲁棒三角化
          |
          v
Partial Reliable 3D Tracks
```

一句话原则：**Track 图像中的物理点，然后反复三角化为 3D；不要 Track 每帧独立生成的 3D 点云。**

已有 `persparse/frameXXX_points3D.txt` 可用于结果对照、场景范围估计或临时种子，但其中每帧的 `POINT3D_ID` 都是局部重建 ID，绝不能直接当成跨帧 Track ID。

## 2. 第一版目标与非目标

第一版目标：

- 直接读取原始多相机视频/图像，自行完成抽帧、固定标定和全帧去畸变。
- 兼容读取 `gs_pipeline` 产出的去畸变多相机图像和固定 COLMAP 标定。
- 在若干出生帧上建立可靠的跨相机 2D 对应，生成全局 Track ID。
- 在每个相机内跟踪 2D 点，同时保留 tracker 的可见性与置信度。
- 每帧用不少于 2 个有效视角鲁棒三角化，并逐相机做重投影剔除。
- 允许轨迹晚出生、提前结束和中间缺失，不强行补齐 3D。
- 输出可审计的 2D 观测、3D 样本、置信度、出生/结束时间和动静分类。

第一版不做：

- 不让所有点都覆盖完整视频。
- 不用 3D 最近邻直接决定两个 Track 的身份。
- 不在只有一个可见相机时猜测 3D。
- 不把平滑后的坐标覆盖原始三角化结果。
- 不一开始就修改 `gs_pipeline` 的默认输出或 FreeTimeGS 训练代码；先通过独立导出适配器集成。

## 3. 与当前仓库的数据接口

首选输入是原始视频目录：

```text
<raw>/
├── cam00.mp4
├── cam01.mp4
├── ...
└── poses_bounds.npy              # 可存在；自动 SfM 模式不依赖它
```

以下已预处理布局作为兼容输入继续支持：

```text
<dataset>/
├── ims/
│   ├── cam000/cam000frame001.png
│   ├── cam000/cam000frame002.png
│   └── ...
├── sparse/0/
│   ├── cameras.txt
│   ├── images.txt
│   └── points3D.txt
└── persparse/
    ├── frame001_points3D.txt
    └── ...
```

流水线尚未组装完成时，也应支持显式传入中间目录：

```text
<work>/undistorted/                         # 图像
<work>/calibration/reference_sparse/0/      # 固定位姿和去畸变内参
<work>/persparse/ 或 persparse_shards/gpu*/ # 可选对照点云
```

### 3.1 必须统一的编号和坐标约定

- 当前图像帧号从 `001` 开始；内部保留真实 `frame_id`，不要先减一再到处混用。
- 从 `images.txt` 的 `NAME` 用 `cam(\d+)` 建立 `cam_id` 映射。不要假定 `IMAGE_ID == CAMERA_ID == cam_id + 1`。
- COLMAP 位姿是 world-to-camera：`x_cam = R(qvec) X_world + t`。
- 相机光心为 `C = -R.T @ t`，投影矩阵为 `P = K [R | t]`。
- 几何计算统一使用 `ims/` 中的去畸变图和 `sparse/0/cameras.txt` 中对应的 PINHOLE 内参。
- matcher 或 tracker 若缩放/裁剪图像，写入几何模块前必须恢复为原始去畸变图像的像素坐标。
- 所有 3D 都使用标定的世界坐标系；不要在中间步骤偷偷归一化后忘记记录变换。

`images.txt` 是每张图两行的 COLMAP 文本格式，第二行可能为空或只有占位点。解析时应按状态机处理，不能简单过滤空行后按奇偶行读取。

### 3.2 关于现有 `persparse` 的重要限制

当前 `generate_per_frame_sparse.py` 最终只集中保留逐帧 `points3D.txt`，对应的逐帧 `images.txt`、`features.h5` 和 `matches.h5` 会随临时目录清理。因此：

- `points3D.txt` 的 `TRACK[] = (IMAGE_ID, POINT2D_IDX)` 没有配套逐帧 `images.txt` 时，不能可靠恢复真实 2D 特征坐标。
- 可将 3D 点重投影到其关联相机，作为近似 tracker query，但它不等价于保存下来的原始匹配像素，也缺少匹配置信度和遮挡信息。
- 正式实现优先让 `track_sparse` 自己保存出生帧的 2D features/matches，或给原稀疏生成器增加“保留逐帧观测”的可选模式。
- 样例 `points3D.txt` 中同一个点可能出现重复 `IMAGE_ID`；任何导入器都要按相机去重，不能盲信 `TRACK[]` 已满足一相机一个观测。

## 4. 建议目录结构

```text
scripts/track_sparse/
├── README.md                 # 本文档
├── run.py                    # 唯一 CLI 入口和阶段调度
├── config.py                 # 默认参数、配置加载与校验
├── schema.py                 # Camera/Observation/TrackSample 数据结构
├── io_colmap.py              # cameras.txt、images.txt、points3D.txt 解析
├── dataset.py                # 图像发现、cam/frame 映射、完整性检查
├── camera_graph.py           # 固定 Camera Graph 构建与缓存
├── spatial_matcher.py        # SuperGlue/LightGlue/RoMa 统一接口
├── track_graph.py            # 同帧多视角观测成组及冲突消解
├── temporal_tracker.py       # CoTracker3/TAPIR 统一接口
├── geometry.py               # 投影、三角化、重投影误差、视差角
├── spawn.py                  # 初始出生、周期出生、重复抑制
├── lifecycle.py              # active/lost/ended 状态与重连
├── classify.py               # static/dynamic/unknown 分类
├── export.py                 # HDF5、NPZ、PLY、FreeTimeGS 适配导出
├── visualize.py              # 2D/3D debug 可视化
└── tests/
    ├── test_io_colmap.py
    ├── test_geometry.py
    ├── test_track_graph.py
    └── test_synthetic_triangulation.py
```

模型实现必须藏在 adapter 后面。上层只依赖统一输出，避免以后从 SuperGlue 换到 LightGlue/RoMa、从 CoTracker3 换到 TAPIR 时重写主流程。

建议接口语义如下：

```python
class SpatialMatcher:
    def match(self, image_a, image_b) -> PairMatches:
        """返回原图像素坐标 uv_a/uv_b、score，以及可选 descriptor。"""

class TemporalTracker:
    def track(self, video, queries) -> TemporalTracks:
        """queries 含 query_frame 和原图 uv；返回每帧 uv、visible、confidence。"""
```

不要让上层代码依赖某个模型当前版本的 tensor 形状或可见性字段名字。

## 5. 内部数据模型

### 5.1 Camera

每台相机至少保存：

```text
cam_id, image_id, camera_id
width, height
K[3,3], R_w2c[3,3], t_w2c[3]
P[3,4], center_world[3]
reference_image_name
```

加载后立即运行自检：四元数归一化、`R R.T ~= I`、`det(R) ~= 1`、内参尺寸与实际图像尺寸一致。

### 5.2 Observation

唯一键为 `(track_id, frame_id, cam_id)`。每个键最多一条观测：

```text
track_id: int64
frame_id: int32
cam_id: int16
u, v: float32                  # 原始去畸变图像像素
visible: bool                  # tracker 判断
tracker_confidence: float32
spatial_confidence: float32    # 出生/重连时有值
reprojection_error: float32    # 三角化后回填
is_inlier: bool
source: seed | temporal | reconnect
```

### 5.3 TrackSample

唯一键为 `(track_id, frame_id)`：

```text
x, y, z: float32
valid_3d: bool
num_visible_views: int16
num_inlier_views: int16
reprojection_rmse: float32
min_triangulation_angle_deg: float32
geometry_confidence: float32
```

只有通过几何验证时 `valid_3d=True`。一视角或零视角帧仍可保留 2D Observation，但不生成 3D。

### 5.4 Track 元数据

```text
track_id, birth_frame, last_valid_frame
state: active | lost | ended
class: static | dynamic | unknown
valid_frame_count, longest_valid_run
mean_confidence, median_reprojection_error
color_rgb                         # 从可靠观测鲁棒统计
```

全局 `track_id` 使用单调递增 int64，一旦分配永不复用。轨迹暂时遮挡后恢复，仍使用原 ID；无法可靠证明是同一点时宁愿创建新 ID，也不要错误合并。

## 6. 完整处理流程

### 阶段 A：数据准备与验证

1. 发现所有 `camXXX/camXXXframeYYY.*`。
2. 求相机集合和各相机帧集合，检查同步帧是否齐全。
3. 加载 COLMAP 文本标定并按图像名映射到 `cam_id`。
4. 检查图像是否已经去畸变、尺寸是否匹配内参。
5. 生成 `manifest.json`，记录输入路径、帧范围、相机列表、配置和输入文件摘要，供断点续跑判断缓存是否失效。

任何相机缺帧都不要静默错位。可以允许缺帧，但必须按真实 `frame_id` 标记 missing，不能用列表下标把后一帧顶上来。

### 阶段 B：建立固定 Camera Graph

Camera Graph 只连接有足够视野重叠和有效基线的相机，后续出生帧仅匹配这些边。

推荐步骤：

1. 用标定得到相机光心、主光轴和视锥。
2. 用首帧参考点云、用户给定场景包围盒，或多个相机视锥交集估计工作体积。
3. 对每对相机计算视锥重叠、朝向夹角、基线和预期三角化角。
4. 过滤无重叠、基线过小或朝向明显不合理的边。
5. 每台相机保留分数最高的 `K=4~8` 个邻居，再做无向对称化。
6. 在一个代表帧上统计每条边通过极线几何的匹配数；过弱边删除，断开的相机可补一条最可靠边。

输出 `camera_graph.json`，至少包含边分数、选择原因和验证匹配数。第一版也可以用手工邻接表启动，但接口不能写死相机数量或环形排列。

### 阶段 C：出生帧的跨相机匹配

出生帧包括初始帧和周期帧，例如每 10 帧一次：

1. 每台相机只提取一次该帧特征并缓存。
2. 仅对 Camera Graph 边运行空间 matcher。
3. 用匹配置信度、双向一致性和已知标定的极线/Sampson 误差过滤边。
4. 建立同帧 2D 观测图：节点是 `(frame_id, cam_id, keypoint_id, u, v)`，边是跨相机匹配。
5. 将边按置信度从高到低加入组件；组件内禁止出现同一相机的两个不同 keypoint。
6. 对候选组件做鲁棒三角化和重投影验证。
7. 种子默认要求至少 3 个几何内点视角；覆盖不足区域可退化到高置信 2 视角。
8. 通过验证后才分配 Track ID。

不能直接对所有 pairwise matches 做无约束并查集合并。错误的传递边可能把同一相机的两个点或两个物理点合成一个组件。

### 阶段 D：每台相机内做 2D 时间跟踪

对每个种子组件，分别把它在各相机的种子像素交给 CoTracker3、TAPIR 或后续 tracker：

```text
Track 42, cam005: (birth_frame, u5, v5) -> uv5(t), visible5(t), conf5(t)
Track 42, cam006: (birth_frame, u6, v6) -> uv6(t), visible6(t), conf6(t)
Track 42, cam007: (birth_frame, u7, v7) -> uv7(t), visible7(t), conf7(t)
```

实现要求：

- 按相机批量跟踪 query，避免一个 Track 调一次模型。
- 明确 offline/online 与视频 chunk 策略；长视频分块时保留重叠帧，并在重叠区校验坐标连续性。
- 保存模型原始 `visibility/confidence`，不要只保存阈值后的 bool。
- query frame 可以不同，因此批处理数据结构必须携带每个 query 的出生帧。
- 第一版建议只向出生帧之后跟踪；向前追溯可作为后续重连功能，避免和已有 Track 大量重复。
- tracker 判断不可见、坐标越界或置信度过低时，该相机当帧不参与三角化。
- 可加入 forward/backward consistency 作为额外分数，但不能取代多视角几何验证。

### 阶段 E：逐帧鲁棒三角化

对每个 `(track_id, frame_id)` 收集所有可见且高置信的 2D 观测。建议实现顺序：

1. 至少有 2 个候选视角，否则 3D 无效。
2. 枚举或 RANSAC 采样相机对，用归一化 DLT 生成候选 3D。
3. 拒绝任一用于估计的相机中深度为负的候选。
4. 拒绝三角化夹角过小的候选。
5. 将候选重投影到全部观测相机，计算逐相机像素误差。
6. 取内点最多、误差最小的候选。
7. 用全部内点进行非线性重投影优化，使用 Huber/Cauchy robust loss。
8. 再次计算误差；逐个剔除超阈值相机并重估，直到收敛或只剩 2 视角。
9. 仍有不少于 2 个内点、正深度、足够夹角且误差合格时才输出有效 3D。

重投影定义：

```text
p_hat_c = project(K_c, R_c, t_c, X)
e_c = ||p_hat_c - p_c||_2
```

不要只存平均误差。必须保留每相机误差和 `is_inlier`，这样才能发现某台相机长期漂移或标定异常。

### 阶段 F：周期出生、增加新视角和重复抑制

只在 frame 1 出生会漏掉后来显露的手背、物体背面等区域，因此每 `spawn_interval` 帧重复阶段 C。

对每个新候选先检查是否已被现有 Track 覆盖：

1. 将候选 3D 投影到可见相机。
2. 查找当前已有 Track 在同相机同帧的 2D 观测。
3. 只有在至少两个共享视角上像素位置一致，并且描述子/局部外观或短时运动也一致时，才视为同一 Track。
4. 一致时给旧 Track 增加相机观测或执行重连；否则创建新 Track。

3D KNN 只能用于缩小候选范围，不能单独决定合并。两个相邻表面点在 3D 中很近，不代表它们身份相同。

周期出生还承担“新相机接入”：某点出生时只在 3 台相机可见，后续通过同帧空间匹配和几何验证，可把第 4 台相机的观测加入同一 Track。

### 阶段 G：遮挡、丢失与重连

建议状态机：

```text
active --连续若干帧无可靠 3D--> lost
lost   --空间+时间+几何均通过--> active（保留 ID）
lost   --超过 max_gap 未恢复--> ended
```

- 只有一个相机可见时保留 2D，3D 标为 invalid。
- 无观测时保留缺口，不线性插值冒充测量值。
- 重连至少要求两个视角几何一致，或一个视角的强时间证据加另一个视角确认。
- 离线导出时可另存 `xyz_smoothed`/`xyz_interpolated`，但必须和原始 `xyz`、`valid_3d` 分开，并记录使用方法。

### 阶段 H：静态/动态分类

分类应在完成几何过滤之后进行。不要直接以相邻帧 3D 差分大于某固定世界单位作为唯一规则，因为标定尺度和三角化噪声会影响结果。

推荐：

1. 仅使用有效且高置信的 3D 样本。
2. 用中位数位置拟合静态模型，计算位置残差的 median/MAD。
3. 结合该 Track 的重投影噪声、视差角与时间跨度估计允许的 3D 抖动。
4. 足够长且残差稳定低于噪声阈值标为 `static`。
5. 存在连续、显著且超过噪声的运动标为 `dynamic`。
6. 样本不足或证据冲突标为 `unknown`，不要强分。

可为动态 Track 额外估计速度或局部轨迹，但原始逐帧 3D 仍是事实数据源。

## 7. 推荐输出格式

主输出建议为一个表式 HDF5 文件，而不是“每个 Track 一个 group”，后者在大量短轨迹时元数据开销很大：

```text
<output>/
├── manifest.json
├── camera_graph.json
├── tracks.h5
│   ├── cameras/*
│   ├── observations/track_id, frame_id, cam_id, uv, ...
│   ├── samples3d/track_id, frame_id, xyz, valid_3d, ...
│   └── tracks/track_id, birth_frame, class, ...
├── debug/
│   ├── overlays/...
│   └── metrics.json
└── exports/
    ├── frameXXX_track_points.npz
    ├── frameXXX_track_points.ply
    └── freetimegs_init.npz
```

`tracks.h5` 的所有表按 `(track_id, frame_id, cam_id)` 或 `(track_id, frame_id)` 排序，并在 `manifest.json` 写明 schema 版本、坐标系、像素定义、帧号起点和配置。

逐帧导出必须令点的 `track_id` 保持全局一致。例如：

```text
track_id[N], xyz[N,3], rgb[N,3], confidence[N], valid[N]
```

若为了兼容旧代码导出 COLMAP 风格 `frameXXX_points3D.txt`，可以让第一列使用全局 Track ID，但 `TRACK[]` 必须基于同时导出的逐帧 `images.txt` 正确重建。不要生成看似合法但索引无法对应的伪 COLMAP 文件。

## 8. CLI

当前入口保持为一条命令：

```bash
python scripts/track_sparse/run.py \
  --dataset data/pubulicdata/cook_spinach_4dgsdata \
  --output data/pubulicdata/cook_spinach_tracks \
  --start-frame 1 \
  --end-frame 300 \
  --spawn-interval 10 \
  --camera-neighbors 6 \
  --min-seed-views 3 \
  --min-triangulation-views 2 \
  --max-reprojection-error 3.0 \
  --min-triangulation-angle 1.0 \
  --spatial-matcher superglue \
  --temporal-tracker superglue \
  --resume
```

同时提供 `--images-dir`、`--sparse-dir` 覆盖项，方便直接调试 `.work` 中间结果。建议支持：

```text
--stages prepare,camera_graph,features,spawn,temporal,triangulate,classify,export
--dry-run
--overwrite
--resume
--debug-frames 1,10,20
--debug-track-ids 1,2,3
```

每一阶段写到临时文件，完成后原子重命名，并在 manifest 记录阶段状态。`--resume` 只复用输入摘要和相关配置均一致的缓存。

## 9. 初始参数建议

下面只是适合开始调试的值，不是最终真值：

| 参数 | 初值 | 说明 |
|---|---:|---|
| camera neighbors | 6 | 每相机 4~8 个有效邻居 |
| spawn interval | 10 帧 | 动作快或新表面出现快时改为 5 |
| min seed views | 3 | 2 视角种子需更严置信度 |
| min runtime views | 2 | 少于 2 视角不输出 3D |
| epipolar/Sampson threshold | 1~2 px | 应按分辨率和标定质量调 |
| max reprojection error | 3 px | 可从 2~4 px 扫描 |
| min triangulation angle | 1 度 | 与现有稀疏重建配置一致起步 |
| tracker confidence threshold | 由模型标定 | 不要假设不同 tracker 分数同尺度 |
| max gap | 8~15 帧 | 超过后结束；仍可严格重连 |

所有阈值必须进入配置和输出 manifest，不能散落成源码魔法数字。最终应根据重投影误差分布、人工 overlay 和少量标注调参。

## 10. 推荐开发顺序

### M0：只做 I/O 与几何

- 完成 COLMAP parser、相机映射和投影函数。
- 用已知 3D 点投影到相机，再三角化回来。
- 单元测试坐标约定、正深度和误差计算。

完成标准：无噪声合成数据能以接近数值精度恢复 3D；反转位姿的错误能被测试捕获。

### M1：单出生帧空间建 Track

- 复用仓库现有 HLOC SuperPoint/SuperGlue 作为第一个 `SpatialMatcher`。
- Camera Graph 先允许手工 JSON，再实现自动构图。
- 构建多视角组件并鲁棒三角化。
- 输出该帧的 2D overlay、3D PLY 和组件统计。

完成标准：组件内每台相机最多一个观测；所有有效点正深度；高分位重投影误差可接受。

### M2：单相机时间跟踪

- 接入一个 temporal tracker adapter。
- 先对 1 台相机、10~30 帧、小批 query 调通。
- 验证坐标缩放、可见性、越界处理与分块连续性。

完成标准：overlay 中点跟随同一物理位置，遮挡时 visibility 合理下降。

### M3：多相机逐帧三角化

- 聚合同一 Track 的各相机时间观测。
- 加入 RANSAC、逐视角重投影剔除和非线性优化。
- 输出误差曲线、有效视角数和 3D 轨迹可视化。

完成标准：人为注入一个错误相机观测时能剔除它，且其余视角仍能恢复正确 3D。

### M4：周期出生与生命周期

- 加入每 N 帧出生。
- 完成重复抑制、新相机观测接入、lost/ended 和严格重连。
- 检查 Track ID 稳定性和重复率。

### M5：分类与 FreeTimeGS 导出

- 做 static/dynamic/unknown 分类。
- 导出出生时间、有效时间段、位置、可见性、置信度和可选速度。
- 单独编写 FreeTimeGS adapter，并用小数据确认字段语义后再接训练代码。

## 11. 测试与验收

### 11.1 必须有的自动测试

- COLMAP `qvec/tvec` 到投影矩阵的已知值测试。
- `images.txt` 空第二行、占位行、乱序 ID 的解析测试。
- 三相机合成点三角化测试。
- 加像素高斯噪声后的稳定性测试。
- 加一个 20 px 离群观测后的 RANSAC/重投影剔除测试。
- 小视差、负深度、只有一视角时必须拒绝 3D。
- 多视角组件中同一相机两个 keypoint 的冲突测试。
- 周期出生对已有 Track 的重复抑制测试。

### 11.2 每次真实数据 smoke test

先跑 3~5 台相邻相机、10~30 帧、几百个点，检查：

- 每帧候选数、种子通过率、Track 出生数。
- 有效 3D 比例和每点内点视角数分布。
- 重投影误差 median、P90、P95。
- 正深度比例和小视差拒绝比例。
- tracker 不可见率、几何剔除率、每相机异常率。
- Track 有效长度、缺口长度、重连数和疑似重复数。
- 2D overlay 是否贴住物理点，3D 轨迹是否出现跳变或飞点。

可把 median 重投影误差小于约 2 px、P95 小于约 4 px 作为最初观察线，但必须根据当前 2703×2027 图像、标定质量和 matcher 精度用真实分布调整，不能当成固定验收真值。

## 12. 常见失败模式检查表

出现“大量点无法三角化/全部飞掉”时，依次检查：

1. 是否把 COLMAP world-to-camera 当成 camera-to-world。
2. 是否错误地假定 `cam_id`、`IMAGE_ID`、`CAMERA_ID` 相等。
3. 2D 坐标是否来自去畸变图，却配了原始畸变内参，或反过来。
4. tracker/matcher resize 后是否忘了坐标缩放回原图。
5. 各相机视频是否真的时间同步，是否存在一帧偏移。
6. 同一多视角组件是否混入同一相机的多个点。
7. 三角化相机对是否基线过小或视差角过小。
8. 某一相机是否长期具有明显更大的重投影误差。
9. 周期出生时是否只用 3D 距离合并，造成错误 ID merge。
10. 是否把不可见点、越界点或低 tracker 置信度点继续送进三角化。

出现“轨迹很短”不一定是失败。优先保证几何可靠；多个短而可靠的 partial tracks 通常好于一条被错误重连拉长的轨迹。

## 13. 编码时必须守住的不变量

- 一个 `(track_id, frame_id, cam_id)` 最多一个 2D 观测。
- 一个有效 `(track_id, frame_id)` 至少两个几何内点视角。
- 每个有效 3D 点在所有内点相机中均为正深度。
- 所有参与几何的 2D 都在同一套去畸变原图像素坐标中。
- Track ID 全局唯一、单调分配、永不复用。
- 缺失就是缺失，不用插值伪装成观测。
- 原始观测、过滤标记、优化结果和可选平滑结果分别保存。
- 3D 距离只能作候选门控，不能单独决定 Track 身份。

先实现并测试这些不变量，再增加更复杂的模型或重连策略，能显著降低后续排查 ID switch 和几何漂移的成本。
