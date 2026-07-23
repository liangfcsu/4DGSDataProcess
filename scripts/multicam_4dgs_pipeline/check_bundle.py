#!/usr/bin/env python3
"""检查多相机4DGS工具包的本地代码、权重和运行环境。"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = (
    "run_pipeline.py",
    "extract_frames_from_videos.py",
    "extract_first_frames.py",
    "complete_3dgs_pipeline.py",
    "undistort_all_frames.py",
    "generate_per_frame_sparse.py",
    "tool/convert_colmap_to_calib.py",
    "tool/undistort_for_hloc.py",
    "tool/generate_pointcloud_multicam.py",
    "Hierarchical-Localization/hloc/extract_features.py",
    "Hierarchical-Localization/hloc/match_features.py",
    "Hierarchical-Localization/hloc/triangulation.py",
    "Hierarchical-Localization/hloc/reconstruction.py",
    "Hierarchical-Localization/hloc/extractors/superpoint.py",
    "Hierarchical-Localization/hloc/matchers/superglue.py",
)

WEIGHTS = (
    "Hierarchical-Localization/third_party/SuperGluePretrainedNetwork/"
    "models/weights/superpoint_v1.pth",
    "Hierarchical-Localization/third_party/SuperGluePretrainedNetwork/"
    "models/weights/superglue_indoor.pth",
    "Hierarchical-Localization/third_party/SuperGluePretrainedNetwork/"
    "models/weights/superglue_outdoor.pth",
)

PYTHON_MODULES = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "torch": "torch",
    "h5py": "h5py",
    "pycolmap": "pycolmap",
    "tqdm": "tqdm",
    "PIL": "Pillow",
    "packaging": "packaging",
}


def fail(message: str, failures: list[str]) -> None:
    print(f"❌ {message}")
    failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description="检查独立工具包及当前Python运行环境")
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="仅在无GPU的容器/远程检查环境中跳过CUDA可见性检查",
    )
    args = parser.parse_args()
    failures: list[str] = []
    print(f"工具包目录: {ROOT}")

    for relative in REQUIRED_FILES:
        path = ROOT / relative
        if not path.is_file():
            fail(f"缺少代码文件: {relative}", failures)

    for relative in WEIGHTS:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size < 1_000_000:
            fail(f"缺少或损坏的权重: {relative}", failures)
        else:
            print(f"✅ 权重: {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MiB)")

    links = [path for path in ROOT.rglob("*") if path.is_symlink()]
    if links:
        fail("工具包内存在符号链接: " + ", ".join(str(p.relative_to(ROOT)) for p in links), failures)
    else:
        print("✅ 所有代码和权重均为目录内实体文件")

    missing_packages = []
    for module_name, package_name in PYTHON_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # 导入CUDA扩展时也可能抛出非ImportError
            missing_packages.append(package_name)
            print(f"❌ Python模块 {module_name}: {type(exc).__name__}: {exc}")
    if missing_packages:
        failures.append("Python环境缺少依赖: " + ", ".join(sorted(set(missing_packages))))
    else:
        print(f"✅ Python依赖完整: {sys.executable}")
        hloc_root = ROOT / "Hierarchical-Localization"
        superglue_root = hloc_root / "third_party" / "SuperGluePretrainedNetwork"
        sys.path.insert(0, str(superglue_root))
        sys.path.insert(0, str(hloc_root))
        try:
            importlib.import_module("hloc.extractors.superpoint")
            importlib.import_module("hloc.matchers.superglue")
            importlib.import_module("hloc.triangulation")
            print("✅ 工具包内HLoc/SuperPoint/SuperGlue可独立导入")
        except Exception as exc:
            fail(f"工具包内部模块导入失败: {type(exc).__name__}: {exc}", failures)

    if args.skip_gpu:
        print("⏭️ 已跳过CUDA可见性检查")
    else:
        try:
            torch = importlib.import_module("torch")
            if not torch.cuda.is_available():
                fail("PyTorch未检测到可用CUDA GPU", failures)
            else:
                print(f"✅ CUDA可用，GPU数量: {torch.cuda.device_count()}")
        except Exception:
            pass

    colmap = shutil.which("colmap")
    if not colmap:
        fail("PATH中未找到COLMAP", failures)
    else:
        result = subprocess.run(
            [colmap, "-h"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if result.returncode != 0:
            fail(f"COLMAP不可执行: {colmap}", failures)
        else:
            print(f"✅ COLMAP: {colmap}")

    if failures:
        print(f"\n自检失败，共 {len(failures)} 项。请按 README.md 配置后重试。")
        return 1
    print("\n🎉 工具包自检通过，可以从本目录运行 run_pipeline.py。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
