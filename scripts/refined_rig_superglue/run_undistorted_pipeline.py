#!/usr/bin/env python3
"""
Run refined-rig undistortion with the Eclipse-style rational model, then
triangulate an undistorted PINHOLE sparse model with the existing SuperGlue runner.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
UNDISTORT_SCRIPT = SCRIPT_DIR / "undistort_refined_rig.py"
SUPERGLUE_RUNNER = REPO_ROOT / "scripts" / "self_process_scripts_superglue" / "4.superglue_simple.py"


def run(cmd):
    print("\n$ " + " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], cwd=str(REPO_ROOT), check=True)


def make_variant_name(args):
    return (
        f"{args.direction}_{args.radial_mode}_{args.small_terms}_"
        f"scale{args.scale:g}_{args.pose_mode}"
    )


def refresh_link(link_path, target_path):
    if link_path.is_symlink():
        link_path.unlink()
    if not link_path.exists():
        link_path.symlink_to(target_path.resolve(), target_is_directory=target_path.is_dir())


def make_3dgs_view(variant_dir, undistorted_images_dir, output_dir):
    train_dir = variant_dir / "3dgs"
    train_dir.mkdir(parents=True, exist_ok=True)
    refresh_link(train_dir / "images", undistorted_images_dir)
    refresh_link(train_dir / "sparse", output_dir / "sparse")
    return train_dir


def main():
    parser = argparse.ArgumentParser(description="Undistort refined-rig images and build a SuperGlue sparse model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/7.7"))
    parser.add_argument("--rig-json", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--direction", choices=["ideal_to_observed", "observed_to_ideal"], default="ideal_to_observed")
    parser.add_argument("--radial-mode", choices=["direct", "swap", "negate", "swap_negate"], default="direct")
    parser.add_argument("--small-terms", choices=["none", "tangential_d8d9", "thin_prism_d8d9"], default="tangential_d8d9")
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
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--superglue-weights", choices=["indoor", "outdoor"], default="indoor")
    parser.add_argument("--resize-max", type=int, default=1600)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--match-threshold", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir
    rig_json = args.rig_json or data_dir / "refined_rig_group.json"
    images_dir = args.images_dir or data_dir / "Photo"
    base_work_dir = args.work_dir or data_dir / "refined_rig_undistorted_superglue"
    variant_dir = base_work_dir / make_variant_name(args)
    undistorted_images_dir = variant_dir / "images"
    reference_sparse_dir = variant_dir / "reference_sparse" / "0"
    output_dir = variant_dir / "superglue_output"

    if args.clean and variant_dir.exists():
        shutil.rmtree(variant_dir)

    run(
        [
            sys.executable,
            UNDISTORT_SCRIPT,
            "--input",
            rig_json,
            "--images-dir",
            images_dir,
            "--output-images-dir",
            undistorted_images_dir,
            "--output-sparse-dir",
            reference_sparse_dir,
            "--direction",
            args.direction,
            "--radial-mode",
            args.radial_mode,
            "--small-terms",
            args.small_terms,
            "--pose-mode",
            args.pose_mode,
            "--scale",
            args.scale,
            "--iterations",
            args.iterations,
        ]
    )

    if args.prepare_only:
        print("\nPrepared undistorted PINHOLE reference model.")
        print(f"images: {undistorted_images_dir}")
        print(f"sparse: {reference_sparse_dir}")
        return 0

    run(
        [
            sys.executable,
            SUPERGLUE_RUNNER,
            "--images-dir",
            undistorted_images_dir,
            "--sparse-dir",
            reference_sparse_dir,
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

    train_dir = make_3dgs_view(variant_dir, undistorted_images_dir, output_dir)

    print("\nDone.")
    print(f"variant: {variant_dir}")
    print(f"undistorted images: {undistorted_images_dir}")
    print(f"reference sparse: {reference_sparse_dir}")
    print(f"pointcloud: {output_dir / 'pointcloud.ply'}")
    print(f"training sparse: {output_dir / 'sparse' / '0'}")
    print(f"3dgs data root: {train_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
