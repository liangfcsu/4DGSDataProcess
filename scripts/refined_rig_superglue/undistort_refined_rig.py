#!/usr/bin/env python3
"""
Undistort refined_rig_group.json images with a rational Eclipse-style model.

The refined rig file is not a COLMAP calibration file. This script keeps the
logic isolated and writes undistorted images plus a PINHOLE COLMAP reference
model for downstream HLoc/SuperGlue and 3DGS.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from convert_refined_rig_to_colmap import (  # noqa: E402
    compose_pose,
    list_images,
)


def parse_d(d, radial_mode):
    values = [float(x) for x in (d or [])]
    values += [0.0] * max(0, 11 - len(values))
    num = values[0:3]
    den = values[3:6]
    if radial_mode == "direct":
        k1, k2, k3 = num
        k4, k5, k6 = den
    elif radial_mode == "swap":
        k1, k2, k3 = den
        k4, k5, k6 = num
    elif radial_mode == "negate":
        k1, k2, k3 = [-x for x in num]
        k4, k5, k6 = [-x for x in den]
    elif radial_mode == "swap_negate":
        k1, k2, k3 = [-x for x in den]
        k4, k5, k6 = [-x for x in num]
    else:
        raise ValueError(f"Unsupported radial mode: {radial_mode}")
    return {
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "k4": k4,
        "k5": k5,
        "k6": k6,
        "d6": values[6],
        "d7": values[7],
        "d8": values[8],
        "d9": values[9],
        "d10": values[10],
    }


def apply_model(x, y, coeffs, small_terms):
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    num = 1.0 + coeffs["k1"] * r2 + coeffs["k2"] * r4 + coeffs["k3"] * r6
    den = 1.0 + coeffs["k4"] * r2 + coeffs["k5"] * r4 + coeffs["k6"] * r6
    radial = num / np.where(np.abs(den) < 1e-12, 1e-12, den)

    xd = x * radial
    yd = y * radial

    if small_terms == "tangential_d8d9":
        p1 = coeffs["d8"]
        p2 = coeffs["d9"]
        xd = xd + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = yd + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    elif small_terms == "thin_prism_d8d9":
        s1 = coeffs["d8"]
        s2 = coeffs["d9"]
        xd = xd + s1 * r2 + s2 * r4
    elif small_terms == "none":
        pass
    else:
        raise ValueError(f"Unsupported small-terms mode: {small_terms}")

    return xd, yd


def invert_model(x_target, y_target, coeffs, small_terms, iterations):
    x = x_target.copy()
    y = y_target.copy()
    for _ in range(iterations):
        xd, yd = apply_model(x, y, coeffs, small_terms)
        x += x_target - xd
        y += y_target - yd
    return x, y


def build_map(width, height, fx, fy, cx, cy, coeffs, direction, small_terms, iterations, scale):
    new_fx = fx * scale
    new_fy = fy * scale
    new_cx = width * 0.5
    new_cy = height * 0.5

    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    x = (u - new_cx) / new_fx
    y = (v - new_cy) / new_fy

    if direction == "ideal_to_observed":
        xs, ys = apply_model(x, y, coeffs, small_terms)
    elif direction == "observed_to_ideal":
        xs, ys = invert_model(x, y, coeffs, small_terms, iterations)
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    map_x = xs * fx + cx
    map_y = ys * fy + cy
    new_k = (new_fx, new_fy, new_cx, new_cy)
    return map_x.astype(np.float32), map_y.astype(np.float32), new_k


def copy_or_write_image(src, dst, cam, args):
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {src}")
    h, w = img.shape[:2]
    coeffs = parse_d(cam.get("D", []), args.radial_mode)
    map_x, map_y, new_k = build_map(
        w,
        h,
        float(cam["fx"]),
        float(cam["fy"]),
        float(cam["cx"]),
        float(cam["cy"]),
        coeffs,
        args.direction,
        args.small_terms,
        args.iterations,
        args.scale,
    )
    undistorted = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst), undistorted)
    if not ok:
        raise RuntimeError(f"Could not write image: {dst}")
    return new_k


def write_pinhole_colmap(rigs, image_names, new_intrinsics, output_dir, pose_mode):
    output_dir.mkdir(parents=True, exist_ok=True)
    cameras = []
    images = []

    for rig in rigs:
        for cam in rig.get("cameras", []):
            idx = len(cameras)
            fx, fy, cx, cy = new_intrinsics[idx]
            camera_id = idx + 1
            image_id = idx + 1
            width = int(cam["w"])
            height = int(cam["h"])
            pose = compose_pose(rig, cam, pose_mode)
            cameras.append((camera_id, width, height, fx, fy, cx, cy))
            images.append((image_id, pose, camera_id, image_names[idx]))

    with (output_dir / "cameras.txt").open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for camera_id, width, height, fx, fy, cx, cy in cameras:
            f.write(f"{camera_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

    with (output_dir / "images.txt").open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}, mean observations per image: 0\n")
        for image_id, pose, camera_id, name in images:
            qw, qx, qy, qz, tx, ty, tz = pose
            f.write(
                f"{image_id} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                f"{tx:.10f} {ty:.10f} {tz:.10f} {camera_id} {name}\n\n"
            )

    with (output_dir / "points3D.txt").open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0, mean track length: 0\n")


def make_variant_name(args):
    return (
        f"{args.direction}_{args.radial_mode}_{args.small_terms}_"
        f"scale{args.scale:g}_{args.pose_mode}"
    )


def main():
    parser = argparse.ArgumentParser(description="Undistort refined-rig images and write a PINHOLE COLMAP model.")
    parser.add_argument("--input", type=Path, default=Path("data/7.7/refined_rig_group.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/7.7/Photo"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/7.7/refined_rig_undistorted"))
    parser.add_argument("--output-images-dir", type=Path)
    parser.add_argument("--output-sparse-dir", type=Path)
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
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    rigs = data.get("rigs")
    if not isinstance(rigs, list):
        raise ValueError("Input JSON must contain a top-level rigs list")

    variant = make_variant_name(args)
    output_images_dir = args.output_images_dir or args.work_dir / variant / "images"
    output_sparse_dir = args.output_sparse_dir or args.work_dir / variant / "sparse" / "0"
    if args.clean and output_images_dir.parent.exists():
        shutil.rmtree(output_images_dir.parent)

    names = list_images(args.images_dir)
    cams = [cam for rig in rigs for cam in rig.get("cameras", [])]
    if len(names) != len(cams):
        raise ValueError(f"Image count {len(names)} != camera count {len(cams)}")

    new_intrinsics = []
    for idx, (name, cam) in enumerate(zip(names, cams)):
        src = args.images_dir / name
        dst = output_images_dir / name
        new_intrinsics.append(copy_or_write_image(src, dst, cam, args))
        if (idx + 1) % 8 == 0 or idx + 1 == len(cams):
            print(f"Undistorted {idx + 1}/{len(cams)} images")

    write_pinhole_colmap(rigs, names, new_intrinsics, output_sparse_dir, args.pose_mode)
    print(f"Wrote undistorted images: {output_images_dir}")
    print(f"Wrote PINHOLE sparse reference: {output_sparse_dir}")


if __name__ == "__main__":
    main()
