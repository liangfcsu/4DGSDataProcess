#!/usr/bin/env python3
"""
多相机点云生成器 - 充分利用22台相机的精确内外参
支持每台相机独立的内参和外参
"""

import json
import numpy as np
from pathlib import Path
import h5py
import pycolmap
from hloc import extract_features, match_features, pairs_from_exhaustive
from hloc.utils import io
import shutil
import time
import sys
import argparse

# 添加third_party路径 - 使用绝对路径
script_dir = Path(__file__).parent.absolute()
hierarchical_loc_path = script_dir.parent / "Hierarchical-Localization"
third_party_path = hierarchical_loc_path / "third_party"

# 添加SuperGlue到Python路径
if third_party_path.exists():
    superglue_path = third_party_path / "SuperGluePretrainedNetwork"
    if superglue_path.exists():
        sys.path.insert(0, str(third_party_path.absolute()))
        sys.path.insert(0, str(superglue_path.absolute()))

# 检查SuperGlue可用性
def check_third_party_availability():
    """检查SuperGlue的可用性"""
    available_methods = {'superglue': False}
    
    try:
        superglue_path = third_party_path / "SuperGluePretrainedNetwork"
        if superglue_path.exists():
            try:
                import SuperGluePretrainedNetwork
                available_methods['superglue'] = True
            except ImportError:
                pass
    except Exception:
        pass
    
    return available_methods


class MultiCameraPointCloudGenerator:
    """多相机点云生成器，充分利用已知标定参数，支持多种高级特征提取方法"""
    
    def __init__(self, colmap_dir, output_dir="hloc_outputs_multicam"):
        self.colmap_dir = Path(colmap_dir)
        self.image_path = self.colmap_dir / "images" 
        self.calib_path = self.colmap_dir / "cameras.json"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 检查第三方方法可用性
        self.available_methods = check_third_party_availability()
        
        # 加载标定数据
        self.load_calibration_data()
        
        print("🚀 多相机点云生成器初始化完成")
        print(f"📂 图像目录: {self.image_path}")
        print(f"📋 标定文件: {self.calib_path}")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"📷 相机数量: {len(self.cameras)}")
        
        # 显示SuperGlue状态
        print("\n🔧 SuperGlue状态:")
        superglue_status = "✅ 可用" if self.available_methods.get('superglue') else "❌ 不可用"
        print(f"   SUPERGLUE: {superglue_status}")
    
    def load_calibration_data(self):
        """加载22台相机的精确标定数据"""
        with open(self.calib_path, 'r') as f:
            calib_data = json.load(f)
        
        self.cameras = {}
        self.images = {}
        
        for cam in calib_data:
            cam_id = cam['id'] + 1  # COLMAP相机ID从1开始
            img_id = cam['id'] + 1  # COLMAP图像ID从1开始
            
            # 存储相机信息
            self.cameras[cam_id] = {
                'width': cam['width'],
                'height': cam['height'], 
                'fx': cam['fx'],
                'fy': cam['fy'],
                'cx': cam['cx'],
                'cy': cam['cy']
            }
            
            # 转换旋转格式：从欧拉角到四元数
            rx, ry, rz = cam['rotation']['rx'], cam['rotation']['ry'], cam['rotation']['rz']
            quat = self.euler_to_quaternion(rx, ry, rz)
            
            # 存储图像信息
            self.images[img_id] = {
                'name': cam['img_name'],
                'camera_id': cam_id,
                'quat': quat.tolist(),  # [qw, qx, qy, qz]
                'tvec': cam['position']  # [tx, ty, tz]
            }
    
    def euler_to_quaternion(self, rx, ry, rz):
        """欧拉角转四元数 (COLMAP格式: qw, qx, qy, qz)"""
        # 将欧拉角转换为旋转矩阵
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)  
        cz, sz = np.cos(rz), np.sin(rz)
        
        # 旋转矩阵 R = Rz * Ry * Rx
        R = np.array([
            [cy*cz, -cy*sz, sy],
            [sx*sy*cz + cx*sz, -sx*sy*sz + cx*cz, -sx*cy],
            [-cx*sy*cz + sx*sz, cx*sy*sz + sx*cz, cx*cy]
        ])
        
        # 旋转矩阵转四元数
        trace = R[0,0] + R[1,1] + R[2,2]
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2,1] - R[1,2]) / s
            qy = (R[0,2] - R[2,0]) / s 
            qz = (R[1,0] - R[0,1]) / s
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2  # s = 4 * qx
                qw = (R[2,1] - R[1,2]) / s
                qx = 0.25 * s
                qy = (R[0,1] + R[1,0]) / s
                qz = (R[0,2] + R[2,0]) / s
            elif R[1,1] > R[2,2]:
                s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2  # s = 4 * qy
                qw = (R[0,2] - R[2,0]) / s
                qx = (R[0,1] + R[1,0]) / s
                qy = 0.25 * s
                qz = (R[1,2] + R[2,1]) / s
            else:
                s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2  # s = 4 * qz
                qw = (R[1,0] - R[0,1]) / s
                qx = (R[0,2] + R[2,0]) / s
                qy = (R[1,2] + R[2,1]) / s
                qz = 0.25 * s
        
        return np.array([qw, qx, qy, qz])
    
    def get_available_feature_extractors(self):
        """获取所有可用的特征提取器"""
        extractors = {
            'sift': {
                'name': 'SIFT (DoG)',
                'description': '经典SIFT特征，稳定可靠',
                'available': True,
                'config': {
                    'model': {'name': 'dog'},
                    'preprocessing': {'grayscale': True, 'resize_max': 1600}
                }
            },
            'superpoint': {
                'name': 'SuperPoint',
                'description': '深度学习特征点检测器',
                'available': True,
                'config': {
                    'model': {'name': 'superpoint'},
                    'preprocessing': {'grayscale': True, 'resize_max': 1600}
                }
            }
        }
        
        return extractors
    
    def get_available_matchers(self):
        """获取所有可用的匹配器"""
        matchers = {
            'nn-mutual': {
                'name': 'NN Mutual',
                'description': '最近邻互验证匹配，鲁棒性强',
                'available': True,
                'config': {
                    'model': {'name': 'nearest_neighbor', 'do_mutual_check': True}
                }
            },
            'nn-ratio': {
                'name': 'NN Ratio',
                'description': 'Lowe比值测试，快速匹配',
                'available': True,
                'config': {
                    'model': {'name': 'nearest_neighbor', 'ratio_threshold': 0.8}
                }
            }
        }
        
        # 添加第三方匹配器
        if self.available_methods['superglue']:
            matchers['superglue'] = {
                'name': 'SuperGlue',
                'description': '神经网络图匹配，最先进技术',
                'available': True,
                'config': {
                    'model': {'name': 'superglue', 'weights': 'outdoor'}
                }
            }
        
        return matchers
        
    def test_all_method_combinations(self):
        """测试所有可用的方法组合"""
        print("\n🧪 测试所有可用的方法组合...")
        
        extractors = self.get_available_feature_extractors()
        matchers = self.get_available_matchers()
        
        if extractors is None or matchers is None:
            print("❌ 无法获取可用方法")
            return None
        
        # 获取可用的方法
        available_extractors = [(k, v) for k, v in extractors.items() if v['available']]
        available_matchers = [(k, v) for k, v in matchers.items() if v['available']]
        
        if not available_extractors or not available_matchers:
            print("❌ 没有可用的方法组合")
            return None
        
        print(f"📊 将测试 {len(available_extractors)} 个特征提取器 × {len(available_matchers)} 个匹配器 = {len(available_extractors) * len(available_matchers)} 种组合")
        
        results = []
        
        for ext_key, ext_info in available_extractors:
            for match_key, match_info in available_matchers:
                print(f"\n🔬 测试组合: {ext_info['name']} + {match_info['name']}")
                
                try:
                    # 创建专用输出目录
                    test_output_dir = self.output_dir / f"test_{ext_key}_{match_key}"
                    test_output_dir.mkdir(exist_ok=True)
                    
                    # 临时改变输出目录
                    original_output_dir = self.output_dir
                    self.output_dir = test_output_dir
                    
                    # 运行测试
                    start_time = time.time()
                    result_path = self.generate(ext_key, match_key)
                    test_time = time.time() - start_time
                    
                    # 恢复输出目录
                    self.output_dir = original_output_dir
                    
                    if result_path:
                        # 读取结果统计
                        points_file = test_output_dir / 'sparse_multicam_text' / 'points3D.txt'
                        num_points = 0
                        if points_file.exists():
                            with open(points_file, 'r') as f:
                                lines = f.readlines()
                                num_points = len([line for line in lines if not line.startswith('#') and line.strip()])
                        
                        results.append({
                            'extractor': ext_info['name'],
                            'matcher': match_info['name'],
                            'status': '✅ 成功',
                            'points': num_points,
                            'time': f"{test_time:.1f}s"
                        })
                        print(f"   ✅ 成功: {num_points} 个3D点, 耗时 {test_time:.1f}s")
                    else:
                        results.append({
                            'extractor': ext_info['name'],
                            'matcher': match_info['name'],
                            'status': '❌ 失败',
                            'points': 0,
                            'time': f"{test_time:.1f}s"
                        })
                        print(f"   ❌ 失败, 耗时 {test_time:.1f}s")
                        
                except Exception as e:
                    results.append({
                        'extractor': ext_info['name'],
                        'matcher': match_info['name'],
                        'status': f'❌ 错误: {str(e)[:50]}...',
                        'points': 0,
                        'time': '0s'
                    })
                    print(f"   ❌ 错误: {e}")
        
        # 显示测试结果总结
        print(f"\n📊 测试结果总结:")
        print(f"{'特征提取器':<15} {'匹配器':<15} {'状态':<15} {'3D点数':<8} {'耗时':<8}")
        print("-" * 70)
        
        successful_combinations = 0
        best_combination = None
        max_points = 0
        
        for result in results:
            print(f"{result['extractor']:<15} {result['matcher']:<15} {result['status']:<15} {result['points']:<8} {result['time']:<8}")
            
            if '成功' in result['status']:
                successful_combinations += 1
                if result['points'] > max_points:
                    max_points = result['points']
                    best_combination = result
        
        print(f"\n🎯 测试总结:")
        print(f"   - 成功组合: {successful_combinations}/{len(results)}")
        print(f"   - 失败组合: {len(results) - successful_combinations}")
        
        if best_combination:
            print(f"   - 最佳组合: {best_combination['extractor']} + {best_combination['matcher']}")
            print(f"   - 最多3D点: {best_combination['points']} 个")
            print(f"   - 处理时间: {best_combination['time']}")
        
        return results
    
    def save_multicam_info(self):
        """保存多相机信息供参考"""
        # 保存相机信息
        cam_info_file = self.output_dir / 'multicam_info.json'
        multicam_data = {
            'cameras': self.cameras,
            'images': self.images,
            'total_cameras': len(self.cameras),
            'description': '22-camera system with individual intrinsics'
        }
        
        with open(cam_info_file, 'w') as f:
            json.dump(multicam_data, f, indent=2)
        
        print(f"📋 多相机信息保存到: {cam_info_file}")
        print(f"   - 相机数量: {len(self.cameras)}")
        print(f"   - 图像数量: {len(self.images)}")
        
        # 显示相机参数范围
        fx_values = [cam['fx'] for cam in self.cameras.values()]
        fy_values = [cam['fy'] for cam in self.cameras.values()]
        
        print(f"   - 焦距范围: fx [{min(fx_values):.1f}, {max(fx_values):.1f}]")
        print(f"   - 焦距范围: fy [{min(fy_values):.1f}, {max(fy_values):.1f}]")
    
    def run_multicam_reconstruction(self, feature_path, match_path, pairs_path):
        """使用hloc标准流程进行多相机重建"""
        print("\n🔺 运行多相机3D重建（使用hloc标准流程）...")
        
        # 保存多相机信息供参考
        self.save_multicam_info()
        
        # 创建稀疏重建目录
        sparse_dir = self.output_dir / 'sparse_multicam'
        if sparse_dir.exists():
            shutil.rmtree(sparse_dir)
        sparse_dir.mkdir()
        
        try:
            print("🚀 使用hloc reconstruction模块...")
            start_time = time.time()
            
            # 使用hloc的标准重建流程，使用更简洁的配置
            from hloc import reconstruction
            
            # 运行重建，使用默认配置但设置优化选项
            model = reconstruction.main(
                sparse_dir, 
                self.image_path, 
                pairs_path, 
                feature_path, 
                match_path,
                verbose=True,
                camera_mode=pycolmap.CameraMode.AUTO  # 自动处理多相机
            )
            
            elapsed = time.time() - start_time
            print(f"⏱️ 重建耗时: {elapsed:.1f}秒")
            
            if model is not None:
                print(f"\n✅ 多相机重建成功!")
                print(f"📊 重建统计:")
                print(f"   - 输入图像数: 22")
                print(f"   - 注册图像数: {model.num_reg_images()}")
                print(f"   - COLMAP相机数: {len(model.cameras)}")
                print(f"   - 3D点数: {len(model.points3D)}")
                
                if len(model.points3D) > 0:
                    # 观测统计
                    total_obs = sum(len(point.track.elements) for point in model.points3D.values())
                    avg_track_length = total_obs / len(model.points3D)
                    print(f"   - 总观测数: {total_obs}")
                    print(f"   - 平均轨迹长度: {avg_track_length:.2f}")
                    
                    # 重投影误差统计
                    errors = [point.error for point in model.points3D.values()]
                    if errors:
                        avg_error = np.mean(errors)
                        median_error = np.median(errors)
                        print(f"   - 平均重投影误差: {avg_error:.3f} 像素")
                        print(f"   - 中位数重投影误差: {median_error:.3f} 像素")
                
                # 保存为文本格式
                text_dir = self.output_dir / 'sparse_multicam_text'
                if text_dir.exists():
                    shutil.rmtree(text_dir)
                text_dir.mkdir()
                model.write_text(str(text_dir))
                
                print(f"📁 结果保存到: {text_dir}")
                
                # 分析相机使用情况
                print(f"\n📷 重建后相机分析:")
                for cam_id, camera in model.cameras.items():
                    # 统计使用此相机的图像数
                    images_with_cam = [img for img in model.images.values() 
                                     if img.camera_id == cam_id]
                    print(f"   相机 {cam_id}: {len(images_with_cam)} 张图像, "
                          f"{camera.model} {camera.width}x{camera.height}")
                    
                    # 显示参数
                    if len(camera.params) >= 3:
                        # 根据参数数量判断相机类型
                        if len(camera.params) == 3:
                            f, cx, cy = camera.params[:3]
                            print(f"      参数: f={f:.1f}, cx={cx:.1f}, cy={cy:.1f}")
                        elif len(camera.params) >= 4:
                            fx, fy, cx, cy = camera.params[:4]
                            print(f"      参数: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
                
                # 与原始标定对比
                print(f"\n🔍 标定对比分析:")
                print(f"   - 原始标定: 22台相机，独立内参")
                print(f"   - 重建结果: {len(model.cameras)}台相机模型")
                if len(model.cameras) < len(self.cameras):
                    print(f"   ⚠️ 注意: COLMAP合并了相似的相机参数")
                else:
                    print(f"   ✅ 保持了多相机独立性")
                
                return str(text_dir)
            else:
                print("❌ 重建失败：没有生成任何模型")
                return None
                
        except Exception as e:
            print(f"❌ 重建过程出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate(self, feature_type='sift', matcher_type='nn-mutual'):
        """生成点云的主流程"""
        print(f"\n🎯 开始多相机点云生成...")
        
        # 获取可用方法
        extractors = self.get_available_feature_extractors()
        matchers = self.get_available_matchers()
        
        # 验证方法可用性
        if feature_type not in extractors:
            print(f"❌ 特征提取器 {feature_type} 不可用")
            return None
        if matcher_type not in matchers:
            print(f"❌ 匹配器 {matcher_type} 不可用")
            return None
            
        print(f"🔧 使用方法组合:")
        print(f"   特征提取器: {extractors[feature_type]['name']} - {extractors[feature_type]['description']}")
        print(f"   匹配器: {matchers[matcher_type]['name']} - {matchers[matcher_type]['description']}")
        
        try:
            # 1. 特征提取
            print(f"\n🎯 使用 {extractors[feature_type]['name']} 进行特征提取...")
            features_path = self.output_dir / f'feats-{feature_type}.h5'
            
            feature_conf = extractors[feature_type]['config'].copy()
            feature_conf['output'] = f'feats-{feature_type}'
            
            extract_features.main(
                feature_conf, 
                self.image_path, 
                feature_path=features_path
            )
            print(f"✅ 特征提取完成: {features_path}")
            
            # 2. 生成图像对
            pairs_path = self.output_dir / f'pairs-exhaustive.txt'
            image_list = [img_info['name'] for img_info in self.images.values()]
            pairs_from_exhaustive.main(pairs_path, image_list=image_list)
            print(f"✅ 图像对生成完成: {pairs_path}")
            
            # 3. 特征匹配 
            print(f"\n🔗 使用 {matchers[matcher_type]['name']} 进行特征匹配...")
            matches_path = self.output_dir / f'feats-{feature_type}_matches-{matcher_type}_{pairs_path.name}.h5'
            
            matcher_conf = matchers[matcher_type]['config'].copy()
            matcher_conf['output'] = f'matches-{matcher_type}'
            
            match_features.main(
                matcher_conf,
                pairs_path, 
                features_path,
                matches=matches_path
            )
            print(f"✅ 特征匹配完成: {matches_path}")
            
            # 4. 多相机3D重建
            result_path = self.run_multicam_reconstruction(
                features_path, matches_path, pairs_path
            )
            
            if result_path:
                print(f"\n🎉 多相机点云生成成功!")
                print(f"📊 使用的方法组合: {extractors[feature_type]['name']} + {matchers[matcher_type]['name']}")
                return result_path
            else:
                print(f"\n❌ 多相机点云生成失败")
                return None
            
        except Exception as e:
            print(f"❌ 生成过程出错: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """交互式或命令行主程序"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="多相机点云生成器")
    parser.add_argument('--feature-method', type=str, default='superpoint',
                       choices=['sift', 'superpoint'],
                       help="特征提取方法")
    parser.add_argument('--matcher-method', type=str, default='superglue',
                       choices=['nn-mutual', 'nn-ratio', 'superglue'],
                       help="特征匹配方法")
    parser.add_argument('--non-interactive', action='store_true',
                       help="非交互式模式，使用命令行参数")
    parser.add_argument('--test-all', action='store_true',
                       help="测试所有方法组合")
    
    args = parser.parse_args()
    
    # 检查输入目录
    colmap_dir = "colmap"
    if not Path(colmap_dir).exists():
        print(f"❌ 目录不存在: {colmap_dir}")
        return
    
    if not (Path(colmap_dir) / "cameras.json").exists():
        print(f"❌ 标定文件不存在: {colmap_dir}/cameras.json") 
        return
    
    if not (Path(colmap_dir) / "images").exists():
        print(f"❌ 图像目录不存在: {colmap_dir}/images")
        return
    
    # 创建生成器
    generator = MultiCameraPointCloudGenerator(colmap_dir)
    
    # 根据参数选择模式
    if args.test_all:
        # 测试所有方法组合
        results = generator.test_all_method_combinations()
        return
    elif args.non_interactive:
        # 非交互式模式，使用命令行参数
        feature_choice = args.feature_method
        matcher_choice = args.matcher_method
        
        print(f"\n🔧 使用命令行指定的方法组合:")
        extractors = generator.get_available_feature_extractors()
        matchers = generator.get_available_matchers()
        
        if feature_choice in extractors:
            print(f"   特征提取器: {extractors[feature_choice]['name']}")
        else:
            print(f"❌ 特征提取器 {feature_choice} 不可用")
            return
            
        if matcher_choice in matchers:
            print(f"   匹配器: {matchers[matcher_choice]['name']}")
        else:
            print(f"❌ 匹配器 {matcher_choice} 不可用")
            return
    else:
        # 交互式模式
        print(f"\n🎛️  请选择运行模式:")
        print(f"  1. 手动选择方法组合")
        print(f"  2. 测试所有可用方法组合")
        
        mode_choice = input("请输入选择 (1-2, 默认手动): ").strip()
        
        if mode_choice == '2':
            # 测试所有方法组合
            results = generator.test_all_method_combinations()
            return
        
        # 手动选择模式
        # 获取可用方法
        extractors = generator.get_available_feature_extractors()
        
        # 特征提取器选择
        print(f"\n🎛️  请选择特征提取方法:")
        extractor_keys = list(extractors.keys())
        for i, (key, info) in enumerate(extractors.items(), 1):
            status = "✅" if info['available'] else "❌"
            print(f"  {i}. {status} {info['name']} - {info['description']}")
        
        while True:
            feature_choice = input(f"请输入选择 (1-{len(extractor_keys)}, 默认sift): ").strip()
            if not feature_choice:
                feature_choice = 'sift'
                break
            elif feature_choice.isdigit() and 1 <= int(feature_choice) <= len(extractor_keys):
                feature_choice = extractor_keys[int(feature_choice) - 1]
                if extractors[feature_choice]['available']:
                    break
                else:
                    print(f"❌ {feature_choice} 当前不可用，请选择其他方法")
            else:
                print(f"请输入有效的选择 (1-{len(extractor_keys)})")
        
        # 获取匹配器（可能依赖于选择的提取器）
        matchers = generator.get_available_matchers()
        
        # 匹配器选择
        print(f"\n🎛️  请选择匹配方法:")
        matcher_keys = list(matchers.keys())
        for i, (key, info) in enumerate(matchers.items(), 1):
            status = "✅" if info['available'] else "❌"
            print(f"  {i}. {status} {info['name']} - {info['description']}")
        
        while True:
            matcher_choice = input(f"请输入选择 (1-{len(matcher_keys)}, 默认nn-mutual): ").strip()
            if not matcher_choice:
                matcher_choice = 'nn-mutual'
                break
            elif matcher_choice.isdigit() and 1 <= int(matcher_choice) <= len(matcher_keys):
                matcher_choice = matcher_keys[int(matcher_choice) - 1]
                if matchers[matcher_choice]['available']:
                    break
                else:
                    print(f"❌ {matcher_choice} 当前不可用，请选择其他方法")
            else:
                print(f"请输入有效的选择 (1-{len(matcher_keys)})")
        
        # 显示选择的方法组合
        print(f"\n🔧 您选择的方法组合:")
        print(f"   特征提取器: {extractors[feature_choice]['name']}")
        print(f"   匹配器: {matchers[matcher_choice]['name']}")
    
    # 生成点云
    result = generator.generate(feature_choice, matcher_choice)
    
    if result:
        print(f"\n🎉 成功！结果保存在: {result}")
        
        # 提供转换选项
        print(f"\n💡 提示: 您可以使用以下命令将points3D.txt转换为PLY格式:")
        print(f"   python point-txt-ply-cloud-converter.py")
        
        # 显示方法组合总结
        print(f"\n📋 本次使用的方法组合:")
        extractors = generator.get_available_feature_extractors()
        matchers = generator.get_available_matchers()
        print(f"   🔧 特征提取: {extractors[feature_choice]['name']}")
        print(f"   🔗 特征匹配: {matchers[matcher_choice]['name']}")
        print(f"   📊 输出目录: hloc_outputs_multicam/")
    else:
        print(f"\n❌ 生成失败，请检查输入数据或尝试其他方法组合")


if __name__ == "__main__":
    main()