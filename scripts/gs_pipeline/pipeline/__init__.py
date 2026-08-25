"""统一 4DGS 数据处理流水线。

单一入口 run.py 通过本包覆盖输入 {多相机视频 | 多相机多帧图像 | 多相机单帧图像}
与标定 {rig 标定 | libCalib 标定 | 无标定自动 SfM} 的全部组合。

本目录是四个历史脚本目录（multicam_4dgs_pipeline、self_process_scripts_superglue、
self_process_scripts_superglue_noncalib、refined_rig_superglue）的合并结果：
重的重建能力（提帧/SfM/去畸变/逐帧三角化 + 单份 Hierarchical-Localization）收纳在
同级 engine/，libCalib 转换在 stages/，标定粘合与调度在 pipeline/ 与 run.py。
"""
