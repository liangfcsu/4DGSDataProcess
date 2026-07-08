#!/usr/bin/env python3
"""
Run the isolated refined-rig SuperGlue sparse point cloud pipeline.

It writes all outputs under --work-dir and calls the existing SuperGlue/HLoc
runner from scripts/self_process_scripts_superglue.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SUPERGLUE_RUNNER = REPO_ROOT / "scripts" / "self_process_scripts_superglue" / "4.superglue_simple.py"


def run(cmd):
    print("\n$ " + " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], cwd=str(REPO_ROOT), check=True)


def make_variant_name(pose_mode, camera_model, tangential_source, radial_mode):
    return f"{pose_mode}_{camera_model.lower()}_{tangential_source}_{radial_mode}"


def main():
    parser = argparse.ArgumentParser(description="Refined-rig isolated SuperGlue pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/7.7"))
    parser.add_argument("--rig-json", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--pose-mode",
        choices=[
            "ctw",
            "wtc",
            "wtc_center",
            "ctw_translation",
            "ctw_flip_yz",
            "wtc_flip_yz",
            "wtc_center_flip_yz",
            "ctw_translation_flip_yz",
        ],
        default="wtc_center",
    )
    parser.add_argument("--camera-model", choices=["PINHOLE", "OPENCV", "FULL_OPENCV"], default="PINHOLE")
    parser.add_argument("--tangential-source", choices=["auto", "d6d7", "d8d9", "zero"], default="auto")
    parser.add_argument("--radial-mode", choices=["direct", "swap", "negate", "swap_negate"], default="direct")
    parser.add_argument("--superglue-weights", choices=["indoor", "outdoor"], default="indoor")
    parser.add_argument("--resize-max", type=int, default=1600)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--match-threshold", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--clean", action="store_true", help="Remove this variant work directory before running")
    args = parser.parse_args()

    data_dir = args.data_dir
    rig_json = args.rig_json or data_dir / "refined_rig_group.json"
    images_dir = args.images_dir or data_dir / "Photo"
    base_work_dir = args.work_dir or data_dir / "refined_rig_superglue"
    variant_dir = base_work_dir / make_variant_name(
        args.pose_mode, args.camera_model, args.tangential_source, args.radial_mode
    )
    sparse_dir = variant_dir / "sparse" / "0"
    output_dir = variant_dir / "superglue_output"

    if args.clean and variant_dir.exists():
        shutil.rmtree(variant_dir)

    if not rig_json.exists():
        raise FileNotFoundError(rig_json)
    if not images_dir.exists():
        raise FileNotFoundError(images_dir)

    python = sys.executable
    run(
        [
            python,
            SCRIPT_DIR / "convert_refined_rig_to_colmap.py",
            "--input",
            rig_json,
            "--images-dir",
            images_dir,
            "--output-dir",
            sparse_dir,
            "--camera-model",
            args.camera_model,
            "--pose-mode",
            args.pose_mode,
            "--tangential-source",
            args.tangential_source,
            "--radial-mode",
            args.radial_mode,
        ]
    )

    if args.prepare_only:
        print("\nPrepared isolated reference model.")
        print(f"sparse: {sparse_dir}")
        return 0

    run(
        [
            python,
            SUPERGLUE_RUNNER,
            "--images-dir",
            images_dir,
            "--sparse-dir",
            sparse_dir,
            "--output-dir",
            output_dir,
            "--max-keypoints",
            args.max_keypoints,
            "--resize-max",
            args.resize_max,
            "--num-workers",
            args.num_workers,
            "--superglue-weights",
            args.superglue_weights,
            "--match-threshold",
            args.match_threshold,
        ]
    )

    print("\nDone.")
    print(f"variant: {variant_dir}")
    print(f"pointcloud: {output_dir / 'pointcloud.ply'}")
    print(f"sparse: {output_dir / 'sparse' / '0'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
