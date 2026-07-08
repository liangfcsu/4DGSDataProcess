#!/usr/bin/env python3
"""
Convert refined_rig_group.json directly to a COLMAP text model.

This is intentionally separate from scripts/self_process_scripts_superglue so
refined-rig experiments do not change the older pipeline files.
"""

import argparse
import json
import math
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mat_vec_mul(m, v):
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def mat_transpose(m):
    return [[m[j][i] for j in range(3)] for i in range(3)]


def vec_add(a, b):
    return [a[i] + b[i] for i in range(3)]


def normalize_matrix(value, name):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a 3x3 matrix")
    out = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"{name} must be a 3x3 matrix")
        out.append([float(x) for x in row])
    return out


def normalize_vec3(value, name):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-value list")
    return [float(x) for x in value]


def quaternion_from_rotation_matrix(r):
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2][1] - r[1][2]) / s
        qy = (r[0][2] - r[2][0]) / s
        qz = (r[1][0] - r[0][1]) / s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        qw = (r[2][1] - r[1][2]) / s
        qx = 0.25 * s
        qy = (r[0][1] + r[1][0]) / s
        qz = (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        qw = (r[0][2] - r[2][0]) / s
        qx = (r[0][1] + r[1][0]) / s
        qy = 0.25 * s
        qz = (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
        qw = (r[1][0] - r[0][1]) / s
        qx = (r[0][2] + r[2][0]) / s
        qy = (r[1][2] + r[2][1]) / s
        qz = 0.25 * s

    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return qw / norm, qx / norm, qy / norm, qz / norm


def list_images(images_dir):
    path = Path(images_dir)
    if not path.exists():
        raise FileNotFoundError(f"Image directory not found: {path}")
    return sorted(p.name for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def parse_distortion(d, tangential_source, radial_mode):
    values = [float(x) for x in (d or [])]
    values += [0.0] * max(0, 11 - len(values))

    numerator = values[0:3]
    denominator = values[3:6]
    if radial_mode == "direct":
        k1, k2, k3 = numerator
        k4, k5, k6 = denominator
    elif radial_mode == "swap":
        k1, k2, k3 = denominator
        k4, k5, k6 = numerator
    elif radial_mode == "negate":
        k1, k2, k3 = [-x for x in numerator]
        k4, k5, k6 = [-x for x in denominator]
    elif radial_mode == "swap_negate":
        k1, k2, k3 = [-x for x in denominator]
        k4, k5, k6 = [-x for x in numerator]
    else:
        raise ValueError(f"Unsupported radial mode: {radial_mode}")

    if tangential_source == "d6d7":
        p1, p2 = values[6], values[7]
    elif tangential_source == "d8d9":
        p1, p2 = values[8], values[9]
    elif tangential_source == "zero":
        p1, p2 = 0.0, 0.0
    else:
        if abs(values[6]) + abs(values[7]) > 1e-12:
            p1, p2 = values[6], values[7]
        elif abs(values[8]) + abs(values[9]) > 1e-12:
            p1, p2 = values[8], values[9]
        else:
            p1, p2 = 0.0, 0.0

    return {
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "k4": k4,
        "k5": k5,
        "k6": k6,
        "p1": p1,
        "p2": p2,
    }


def camera_params(cam, model, tangential_source, radial_mode):
    distortion = parse_distortion(cam.get("D", []), tangential_source, radial_mode)
    fx = float(cam["fx"])
    fy = float(cam["fy"])
    cx = float(cam["cx"])
    cy = float(cam["cy"])

    if model == "PINHOLE":
        return [fx, fy, cx, cy]
    if model == "OPENCV":
        return [fx, fy, cx, cy, distortion["k1"], distortion["k2"], distortion["p1"], distortion["p2"]]
    if model == "FULL_OPENCV":
        return [
            fx,
            fy,
            cx,
            cy,
            distortion["k1"],
            distortion["k2"],
            distortion["p1"],
            distortion["p2"],
            distortion["k3"],
            distortion["k4"],
            distortion["k5"],
            distortion["k6"],
        ]
    raise ValueError(f"Unsupported camera model: {model}")


def compose_pose(rig, cam, pose_mode):
    rig_r = normalize_matrix(rig["rig_rotation"], "rig_rotation")
    rig_t = normalize_vec3(rig["rig_position"], "rig_position")
    cam_r = normalize_matrix(cam.get("R", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "camera R")
    cam_c = normalize_vec3(cam.get("c", [0.0, 0.0, 0.0]), "camera c")
    flip_yz = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    base_pose_mode = pose_mode.replace("_flip_yz", "")

    if base_pose_mode == "ctw":
        r_ctw = mat_mul(rig_r, cam_r)
        center = vec_add(rig_t, mat_vec_mul(rig_r, cam_c))
        r_wtc = mat_transpose(r_ctw)
        t_wtc = [-x for x in mat_vec_mul(r_wtc, center)]
    elif base_pose_mode == "wtc":
        r_wtc = mat_mul(rig_r, cam_r)
        t_wtc = vec_add(rig_t, mat_vec_mul(rig_r, cam_c))
    elif base_pose_mode == "wtc_center":
        r_wtc = mat_mul(rig_r, cam_r)
        center = vec_add(rig_t, mat_vec_mul(rig_r, cam_c))
        t_wtc = [-x for x in mat_vec_mul(r_wtc, center)]
    elif base_pose_mode == "ctw_translation":
        r_ctw = mat_mul(rig_r, cam_r)
        r_wtc = mat_transpose(r_ctw)
        t_wtc = vec_add(rig_t, mat_vec_mul(rig_r, cam_c))
    else:
        raise ValueError(f"Unsupported pose mode: {pose_mode}")

    if pose_mode.endswith("_flip_yz"):
        r_wtc = mat_mul(flip_yz, r_wtc)
        t_wtc = mat_vec_mul(flip_yz, t_wtc)

    qw, qx, qy, qz = quaternion_from_rotation_matrix(r_wtc)
    return qw, qx, qy, qz, t_wtc[0], t_wtc[1], t_wtc[2]


def write_colmap_model(rigs, image_names, output_dir, camera_model, tangential_source, radial_mode, pose_mode):
    output_dir.mkdir(parents=True, exist_ok=True)
    cameras = []
    images = []

    for rig in rigs:
        for cam in rig.get("cameras", []):
            idx = len(cameras)
            if idx >= len(image_names):
                raise ValueError(f"Not enough images for cameras: image index {idx}")

            camera_id = idx + 1
            image_id = idx + 1
            params = camera_params(cam, camera_model, tangential_source, radial_mode)
            pose = compose_pose(rig, cam, pose_mode)
            cameras.append((camera_id, camera_model, int(cam["w"]), int(cam["h"]), params))
            images.append((image_id, pose, camera_id, image_names[idx]))

    if len(cameras) != len(image_names):
        print(f"Warning: {len(cameras)} cameras but {len(image_names)} images")

    with (output_dir / "cameras.txt").open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for camera_id, model, width, height, params in cameras:
            params_str = " ".join(str(x) for x in params)
            f.write(f"{camera_id} {model} {width} {height} {params_str}\n")

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

    return len(cameras)


def main():
    parser = argparse.ArgumentParser(description="Convert refined rig calibration to a COLMAP text model.")
    parser.add_argument("--input", type=Path, default=Path("data/7.7/refined_rig_group.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/7.7/Photo"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/7.7/refined_rig_superglue/sparse/0"))
    parser.add_argument("--camera-model", choices=["PINHOLE", "OPENCV", "FULL_OPENCV"], default="PINHOLE")
    parser.add_argument("--radial-mode", choices=["direct", "swap", "negate", "swap_negate"], default="direct")
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
    parser.add_argument("--tangential-source", choices=["auto", "d6d7", "d8d9", "zero"], default="auto")
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    rigs = data.get("rigs")
    if not isinstance(rigs, list):
        raise ValueError("Input JSON must contain a top-level rigs list")

    image_names = list_images(args.images_dir)
    count = write_colmap_model(
        rigs,
        image_names,
        args.output_dir,
        args.camera_model,
        args.tangential_source,
        args.radial_mode,
        args.pose_mode,
    )
    print(f"Wrote {count} cameras/images to {args.output_dir}")
    print(
        f"camera_model={args.camera_model}, pose_mode={args.pose_mode}, "
        f"tangential_source={args.tangential_source}, radial_mode={args.radial_mode}"
    )


if __name__ == "__main__":
    main()
