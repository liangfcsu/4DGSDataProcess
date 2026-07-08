#!/usr/bin/env python3
"""
One-command sparse point cloud pipeline for refined_rig_group.json + images.

Pipeline:
1. Convert refined_rig_group.json to calib0.3343.json-style calibration.
2. Convert calibration + images to a COLMAP reference sparse model.
3. Run SuperPoint + SuperGlue + pycolmap triangulation to export a point cloud.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def run_step(cmd, cwd):
    print("\n$ " + " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_exists(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a SuperGlue sparse point cloud from refined rig calibration and images."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/7.7"), help="Dataset directory")
    parser.add_argument("--rig-json", type=Path, default=None, help="Path to refined_rig_group.json")
    parser.add_argument("--images-dir", type=Path, default=None, help="Path to image directory")
    parser.add_argument("--calib-json", type=Path, default=None, help="Output calib0.3343.json path")
    parser.add_argument("--sparse-dir", type=Path, default=None, help="Output COLMAP reference sparse/0 path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output SuperGlue point cloud directory")
    parser.add_argument("--camera-model", choices=["auto", "pinhole", "opencv", "full_opencv"], default="auto")
    parser.add_argument("--superglue-weights", choices=["indoor", "outdoor"], default="indoor")
    parser.add_argument("--max-keypoints", type=int, default=4096)
    parser.add_argument("--resize-max", type=int, default=4000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--match-threshold", type=float, default=0.2)
    parser.add_argument("--prepare-only", action="store_true", help="Only generate calib and reference sparse files")
    parser.add_argument("--skip-calib", action="store_true", help="Reuse existing calib JSON")
    parser.add_argument("--skip-reference-sparse", action="store_true", help="Reuse existing reference sparse model")
    parser.add_argument(
        "--clean-superglue-cache",
        action="store_true",
        help="Remove output/temp before running SuperGlue so features and matches are recomputed",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    rig_json = args.rig_json or data_dir / "refined_rig_group.json"
    images_dir = args.images_dir or data_dir / "Photo"
    calib_json = args.calib_json or data_dir / "calib0.3343.json"
    sparse_dir = args.sparse_dir or data_dir / "sparse" / "0"
    output_dir = args.output_dir or data_dir / "superglue_output"

    ensure_exists(rig_json, "refined rig JSON")
    ensure_exists(images_dir, "image directory")

    python = sys.executable

    if not args.skip_calib:
        run_step(
            [
                python,
                SCRIPT_DIR / "0.convert_refined_rig_group_to_cameras_json.py",
                "--input",
                rig_json,
                "--images-dir",
                images_dir,
                "--output",
                calib_json,
                "--output-format",
                "calib",
                "--backup",
            ],
            REPO_ROOT,
        )
    else:
        ensure_exists(calib_json, "calib JSON")

    if not args.skip_reference_sparse:
        run_step(
            [
                python,
                SCRIPT_DIR / "3.0convert_to_colmap.py",
                "--cameras-json",
                calib_json,
                "--images-dir",
                images_dir,
                "--output-dir",
                sparse_dir,
                "--verify-images",
                "--camera-model",
                args.camera_model,
            ],
            REPO_ROOT,
        )
    else:
        ensure_exists(sparse_dir / "cameras.txt", "reference cameras.txt")
        ensure_exists(sparse_dir / "images.txt", "reference images.txt")
        ensure_exists(sparse_dir / "points3D.txt", "reference points3D.txt")

    if args.prepare_only:
        print("\nPrepared calibration and reference sparse model.")
        print(f"calib: {calib_json}")
        print(f"sparse: {sparse_dir}")
        return 0

    temp_dir = output_dir / "temp"
    if args.clean_superglue_cache and temp_dir.exists():
        shutil.rmtree(temp_dir)

    run_step(
        [
            python,
            SCRIPT_DIR / "4.superglue_simple.py",
            "--images-dir",
            images_dir,
            "--sparse-dir",
            sparse_dir,
            "--output-dir",
            output_dir,
            "--max-keypoints",
            str(args.max_keypoints),
            "--resize-max",
            str(args.resize_max),
            "--num-workers",
            str(args.num_workers),
            "--superglue-weights",
            args.superglue_weights,
            "--match-threshold",
            str(args.match_threshold),
        ],
        REPO_ROOT,
    )

    print("\nPipeline finished.")
    print(f"point cloud: {output_dir / 'pointcloud.ply'}")
    print(f"sparse output: {output_dir / 'sparse' / '0'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
