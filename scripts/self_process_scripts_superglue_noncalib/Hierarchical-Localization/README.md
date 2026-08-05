# Hierarchical-Localization (精简版)

这是 [Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization) 的精简版本，只保留了本项目实际用到的核心功能。

## 保留的功能

- **特征提取**: `hloc/extract_features.py` - SuperPoint 特征提取
- **特征匹配**: `hloc/match_features.py` - SuperGlue 特征匹配
- **图像配对**: `hloc/pairs_from_exhaustive.py` - 穷举配对生成
- **SfM 重建**: `hloc/reconstruction.py` - COLMAP 重建接口

## 目录结构

```
Hierarchical-Localization/
├── hloc/
│   ├── extract_features.py      # 特征提取主模块
│   ├── match_features.py         # 特征匹配主模块
│   ├── pairs_from_exhaustive.py  # 配对生成
│   ├── reconstruction.py         # SfM 重建
│   ├── extractors/
│   │   └── superpoint.py         # SuperPoint 提取器
│   ├── matchers/
│   │   ├── superglue.py          # SuperGlue 匹配器
│   │   └── nearest_neighbor.py   # NN 匹配器
│   └── utils/                    # 工具类
└── third_party/
    └── SuperGluePretrainedNetwork/  # SuperGlue 预训练模型
        ├── models/
        │   ├── superpoint.py
        │   ├── superglue.py
        │   └── weights/          # 预训练权重 (97MB)
        └── match_pairs.py
```

## 删除的内容

- ❌ 所有 demo 和 notebook
- ❌ 其他特征提取器 (DISK, R2D2, D2Net, AlIKED, DogAff 等)
- ❌ 其他匹配器 (LoFTR, LightGlue, AdaLAM 等)
- ❌ 定位管线 (Aachen, 7Scenes, Cambridge, CMU, RobotCar 等)
- ❌ 测试数据集和示例配对
- ❌ 文档和可视化工具

## 安装

### 1. 下载预训练权重

权重文件较大（97MB），需要单独下载：

```bash
cd third_party/SuperGluePretrainedNetwork
mkdir -p models/weights
cd models/weights

# 下载 SuperPoint 权重
wget https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superpoint_v1.pth

# 下载 SuperGlue 权重
wget https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_outdoor.pth
wget https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_indoor.pth
```

### 2. 安装依赖

```bash
pip install torch opencv-python h5py pycolmap
```

## 使用

本精简版专为 `complete_3dgs_pipeline.py` 优化，通过以下方式使用：

```python
from hloc import extract_features, match_features, pairs_from_exhaustive, reconstruction
```

## 原始项目

完整版本请访问：https://github.com/cvg/Hierarchical-Localization

## 许可证

遵循原项目 Apache License 2.0
