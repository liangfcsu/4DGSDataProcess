#!/usr/bin/env python3
"""
Try multiple refined-rig pose/camera-model interpretations without undistorting images.

The script computes SuperPoint/SuperGlue features once on the original images,
then reuses those matches to triangulate several COLMAP reference models.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pycolmap


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
HLOC_ROOT = REPO_ROOT / "scripts" / "self_process_scripts_superglue" / "Hierarchical-Localization"
sys.path.insert(0, str(HLOC_ROOT))

from hloc import extract_features, match_features, triangulation  # noqa: E402


DEFAULT_POSE_MODES = [
    "ctw",
    "wtc_center",
    "wtc",
    "ctw_flip_yz",
    "wtc_center_flip_yz",
    "wtc_flip_yz",
]


def run(cmd):
    print("\n$ " + " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], cwd=str(REPO_ROOT), check=True)


def image_names(images_dir):
    names = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        names.extend(p.name for p in Path(images_dir).glob(ext))
    return sorted(set(names))


def write_exhaustive_pairs(names, pairs_path):
    pairs = []
    for i, name0 in enumerate(names):
        for name1 in names[i + 1 :]:
            pairs.append(f"{name0} {name1}")
    pairs_path.write_text("\n".join(pairs) + "\n", encoding="utf-8")
    return len(pairs)


def ensure_features_and_matches(args, images_dir, shared_dir):
    shared_dir.mkdir(parents=True, exist_ok=True)
    features_path = shared_dir / "features.h5"
    matches_path = shared_dir / "matches.h5"
    pairs_path = shared_dir / "pairs.txt"

    if not pairs_path.exists() or args.overwrite_matches:
        count = write_exhaustive_pairs(image_names(images_dir), pairs_path)
        print(f"Wrote {count} exhaustive pairs to {pairs_path}")

    if not features_path.exists() or args.overwrite_matches:
        feature_conf = {
            "model": {
                "name": "superpoint",
                "nms_radius": 4,
                "keypoint_threshold": args.keypoint_threshold,
                "max_keypoints": args.max_keypoints,
            },
            "preprocessing": {
                "grayscale": True,
                "resize_max": args.resize_max,
            },
        }
        extract_features.main(
            feature_conf,
            images_dir,
            feature_path=features_path,
            overwrite=args.overwrite_matches,
            num_workers=args.num_workers,
        )
    else:
        print(f"Reusing features: {features_path}")

    if not matches_path.exists() or args.overwrite_matches:
        matcher_conf = {
            "model": {
                "name": "superglue",
                "weights": args.superglue_weights,
                "sinkhorn_iterations": args.sinkhorn_iterations,
                "match_threshold": args.match_threshold,
            }
        }
        match_features.main(
            matcher_conf,
            pairs_path,
            features=features_path,
            matches=matches_path,
            num_workers=args.num_workers,
        )
    else:
        print(f"Reusing matches: {matches_path}")

    return pairs_path, features_path, matches_path


def point_count(model_dir):
    reconstruction = pycolmap.Reconstruction(model_dir)
    return reconstruction.num_points3D(), reconstruction


def export_pointcloud_ply(reconstruction, ply_path):
    points = []
    for _, point in reconstruction.points3D.items():
        xyz = point.xyz
        rgb = point.color
        points.append((xyz[0], xyz[1], xyz[2], int(rgb[0]), int(rgb[1]), int(rgb[2])))

    with ply_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in points:
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


def make_variant_name(pose_mode, camera_model, tangential_source, radial_mode):
    return f"{pose_mode}_{camera_model.lower()}_{tangential_source}_{radial_mode}"


def run_variant(args, pose_mode, camera_model, tangential_source, radial_mode, images_dir, pairs_path, features_path, matches_path):
    variant_dir = args.work_dir / make_variant_name(pose_mode, camera_model, tangential_source, radial_mode)
    reference_dir = variant_dir / "reference_sparse" / "0"
    sfm_dir = variant_dir / "sfm"
    sparse_text_dir = variant_dir / "sparse" / "0"

    if args.clean_variant and variant_dir.exists():
        shutil.rmtree(variant_dir)

    run(
        [
            sys.executable,
            SCRIPT_DIR / "convert_refined_rig_to_colmap.py",
            "--input",
            args.rig_json,
            "--images-dir",
            images_dir,
            "--output-dir",
            reference_dir,
            "--camera-model",
            camera_model,
            "--pose-mode",
            pose_mode,
            "--tangential-source",
            tangential_source,
            "--radial-mode",
            radial_mode,
        ]
    )

    if sfm_dir.exists():
        shutil.rmtree(sfm_dir)
    sfm_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    reconstruction = triangulation.main(
        sfm_dir,
        reference_dir,
        images_dir,
        pairs_path,
        features_path,
        matches_path,
        skip_geometric_verification=args.skip_geometric_verification,
        min_match_score=args.min_match_score,
        verbose=args.verbose,
    )
    elapsed = time.time() - start

    sparse_text_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write_text(sparse_text_dir)
    pts = reconstruction.num_points3D()
    if pts > 0:
        export_pointcloud_ply(reconstruction, variant_dir / "pointcloud.ply")

    print(f"RESULT {variant_dir.name}: points={pts}, elapsed={elapsed:.1f}s")
    return {
        "variant": variant_dir.name,
        "points": pts,
        "elapsed": elapsed,
        "variant_dir": variant_dir,
        "sparse_dir": sparse_text_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Try refined-rig sparse reconstruction variants.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/7.7"))
    parser.add_argument("--rig-json", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--pose-modes", nargs="+", default=DEFAULT_POSE_MODES)
    parser.add_argument("--camera-models", nargs="+", choices=["PINHOLE", "OPENCV", "FULL_OPENCV"], default=["FULL_OPENCV"])
    parser.add_argument("--tangential-sources", nargs="+", choices=["auto", "d6d7", "d8d9", "zero"], default=["auto"])
    parser.add_argument(
        "--radial-modes",
        nargs="+",
        choices=["direct", "swap", "negate", "swap_negate"],
        default=["direct"],
    )
    parser.add_argument("--resize-max", type=int, default=1600)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--keypoint-threshold", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--superglue-weights", choices=["indoor", "outdoor"], default="indoor")
    parser.add_argument("--sinkhorn-iterations", type=int, default=100)
    parser.add_argument("--match-threshold", type=float, default=0.2)
    parser.add_argument("--min-match-score", type=float)
    parser.add_argument("--skip-geometric-verification", action="store_true")
    parser.add_argument("--overwrite-matches", action="store_true")
    parser.add_argument("--clean-variant", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.rig_json = args.rig_json or args.data_dir / "refined_rig_group.json"
    images_dir = args.images_dir or args.data_dir / "Photo"
    args.work_dir = args.work_dir or args.data_dir / "refined_rig_superglue_trials"
    shared_dir = args.work_dir / "_shared_matches"

    if not args.rig_json.exists():
        raise FileNotFoundError(args.rig_json)
    if not images_dir.exists():
        raise FileNotFoundError(images_dir)

    pairs_path, features_path, matches_path = ensure_features_and_matches(args, images_dir, shared_dir)

    results = []
    for camera_model in args.camera_models:
        for tangential_source in args.tangential_sources:
            for radial_mode in args.radial_modes:
                for pose_mode in args.pose_modes:
                    results.append(
                        run_variant(
                            args,
                            pose_mode,
                            camera_model,
                            tangential_source,
                            radial_mode,
                            images_dir,
                            pairs_path,
                            features_path,
                            matches_path,
                        )
                    )

    results.sort(key=lambda x: x["points"], reverse=True)
    print("\nSummary:")
    for result in results:
        print(f"{result['variant']}: {result['points']} points -> {result['sparse_dir']}")

    best = results[0] if results else None
    if best:
        print(f"\nBest: {best['variant']} with {best['points']} points")
        if best["points"] > 0:
            print(f"Use sparse model: {best['sparse_dir']}")
            print(f"Point cloud: {best['variant_dir'] / 'pointcloud.ply'}")


if __name__ == "__main__":
    main()
