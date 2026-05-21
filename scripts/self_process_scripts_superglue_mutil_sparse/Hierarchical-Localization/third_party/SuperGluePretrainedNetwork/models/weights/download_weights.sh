#!/bin/bash
# 下载 SuperPoint 和 SuperGlue 预训练权重

set -e

echo "📥 下载 SuperPoint 权重..."
wget -nc https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superpoint_v1.pth

echo "📥 下载 SuperGlue outdoor 权重..."
wget -nc https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_outdoor.pth

echo "📥 下载 SuperGlue indoor 权重..."
wget -nc https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_indoor.pth

echo "✅ 权重下载完成！"
ls -lh *.pth
