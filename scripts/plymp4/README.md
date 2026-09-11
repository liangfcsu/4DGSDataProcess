# 将逐帧 PLY 封装成 4DGS MP4

这不是把点云渲染成二维视频，而是把每个完整 PLY 作为一个带时间戳的 MP4 sample。
配套的 4DGS-Player 会按时间轴随机读取、解压并继续用 Gaussian renderer 绘制，
所以播放时仍能旋转、平移和缩放视角。

## 封装

在 `4DGSDataProcess` 仓库根目录执行：

```bash
# 先打包 3 帧做快速验证
python3 scripts/plymp4/pack.py \
  --input data/ply \
  --output data/preview_4dgs.mp4 \
  --end 3 --fps 30 --overwrite

# 打包全部 241 帧
python3 scripts/plymp4/pack.py \
  --input data/ply \
  --output data/sequence_4dgs.mp4 \
  --fps 30 --compression zlib --compression-level 1 --overwrite
```

不压缩可减少 CPU 解码开销，但文件约等于原始 PLY 总大小：

```bash
python3 scripts/plymp4/pack.py --input data/ply \
  --output data/sequence_fast_4dgs.mp4 --compression none --overwrite
```

程序流式读写，不会一次把 9.5 GB 数据装入内存。输出先写入同目录临时文件，成功
后才原子替换目标文件。

## 播放

```bash
cd /home/lf/SAM4T/4DGS/4DGS-Player

# 命令行打开
python player.py /绝对路径/sequence_4dgs.mp4

# 或启动 GUI 后，把 .mp4 文件拖进主视口
python player.py
```

播放器直接从 MP4 做随机读取，不会把 241 个 PLY 解包到临时目录。方向键、时间轴、
空格播放以及自由相机操作和原 PLY 目录模式相同。

详细二进制布局见 [FORMAT.md](FORMAT.md)。这是项目自定义的 `gsp1` 4DGS codec；
普通视频播放器不具备 Gaussian renderer，不能播放该文件。
