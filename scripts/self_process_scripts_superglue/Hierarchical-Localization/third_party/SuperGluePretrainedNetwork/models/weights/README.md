# 权重文件目录

此目录应包含以下预训练权重文件（需单独下载）：

- `superpoint_v1.pth` (5.0 MB)
- `superglue_outdoor.pth` (46 MB)
- `superglue_indoor.pth` (46 MB)

## 下载方法

```bash
# 在当前目录下运行：
wget https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superpoint_v1.pth
wget https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_outdoor.pth
wget https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_indoor.pth
```

或使用提供的下载脚本：

```bash
./download_weights.sh
```
