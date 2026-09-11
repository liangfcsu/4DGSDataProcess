# 4DGS-in-MP4 v1

这是 4DGS-Player 与 `scripts/plymp4/pack.py` 之间的自定义、无损封装协议。
扩展名为 `.mp4`，底层结构遵循 ISO Base Media File Format，但 `gsp1` 不是通用
播放器支持的视频 codec。

## MP4 结构

```text
ftyp  major_brand=isom, compatible_brands=[isom, iso6, mp41, 4dgs]
mdat
  sample 0
  sample 1
  ...
moov
  mvhd
  trak
    tkhd
    mdia
      mdhd                  # 精确帧率时间基
      hdlr handler=meta
      minf/nmhd/dinf/stbl
        stsd/gsp1/gscC      # gscC 是 UTF-8 JSON manifest
        stts                # 每帧时长
        stsc                # 每 chunk 一帧
        stsz                # 每帧字节数
        co64                # 64 位随机访问偏移，支持 >4 GiB
```

每个样本都是独立随机访问帧；无需从关键帧开始解码。

## `gsp1` sample

所有整数使用 big-endian：

```text
offset  size  field
0       4     magic = "GSP1"
4       1     version = 1
5       1     flags: bit 0 = zlib
6       2     reserved = 0
8       8     解压后的 PLY 字节数
16      4     解压后 PLY 的 CRC32
20      ...   原始 PLY，或逐帧独立 zlib 数据
```

解码所得字节与输入 PLY 完全一致，因此 `x/y/z`、全部球谐系数、opacity、scale、
rotation 以及未知扩展属性都不会因封装而丢失。

## 兼容性

- FFprobe 等工具能识别 MP4 时间轴，但会把 `gsp1` 报告为未知 codec，这是正常的。
- VLC、浏览器 `<video>` 等普通二维播放器无法渲染该轨道。
- 目前由本项目配套修改后的 4DGS-Player 解封装并进行 CUDA Gaussian 渲染。
