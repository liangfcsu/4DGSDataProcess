# track_sparse 使用说明

`run.py` 是唯一启动脚本。`configs/default.yaml` 是唯一内置配置文件，包含数据路径、执行选项和轨迹算法参数。

## 处理数据集

从仓库根目录运行：

```bash
python scripts/track_sparse/run.py
```

当前配置处理 `data/coffee_martini/coffee_martini` 中的 17 路视频，输出到 `outputs/coffee_martini_tracks`。换数据集时，先修改 `configs/default.yaml` 中的 `run.input` 和 `run.output`。再按数据情况修改 `input_type`、帧范围、GPU、缓存模式及下方算法参数。`max_frames` 不应小于 `end_frame`。

也可以临时用 CLI 覆盖 YAML 中的运行选项：

```bash
python scripts/track_sparse/run.py --input data/another_scene --output outputs/another_scene_tracks
```

`--config path/to/scene.yaml` 可指定另一份配置文件；未提供时始终读取 `configs/default.yaml`。CLI 参数的优先级高于 YAML。`run:` 仅控制执行，不参与算法配置和轨迹缓存指纹。

## 目录

```text
scripts/track_sparse/
  run.py                         # 唯一启动入口
  configs/default.yaml           # 输入、输出、运行及算法参数
  core/                          # 配置、数据结构、流程编排
  input/                         # 数据发现、COLMAP 读取、预处理
  geometry/                      # 相机图、三角化、不确定性
  matching/                      # 空间和时间匹配
  tracking/                      # 轨迹生命周期、身份验证
  optimization/                  # 轨迹、相机位姿、运动优化
  exports/                       # 结果导出与可视化
  dependencies/requirements.txt  # Python 依赖
  docs/                          # 文档
  tests/                         # 单元测试
```

原始视频自动标定需要可在 `PATH` 中调用的 `colmap`。安装 Python 依赖可使用 `pip install -r scripts/track_sparse/dependencies/requirements.txt`。`poses_bounds.npy` 不提供镜头畸变模型，因此原始视频流程会执行首帧 SfM 标定与去畸变。
