#!/usr/bin/env python3
"""
Convert refined_rig_group.json into the calib0.3343.json-style structure used
by the self_process_scripts_superglue pipeline.

Default output pose is compatible with 3.0convert_to_colmap.py defaults:
rotation is a Rodrigues vector for world-to-camera, and translation is COLMAP's
world-to-camera translation vector. Use --pose-output ctw-euler if you want
camera centers plus ZYX Euler angles instead.
"""

import argparse
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DISTORTION_KEYS = (
    "k1",
    "k2",
    "k3",
    "k4",
    "k5",
    "k6",
    "p1",
    "p2",
    "s1",
    "s2",
    "s3",
    "s4",
    "tauX",
    "tauY",
)
PARAMETER_STATES = {
    "f": 0,
    "ar": 2,
    "cx": 0,
    "cy": 0,
    "k1": 0,
    "k2": 0,
    "k3": 2,
    "k4": 2,
    "k5": 2,
    "k6": 2,
    "p1": 2,
    "p2": 2,
    "s1": 2,
    "s2": 2,
    "s3": 2,
    "s4": 2,
    "tauX": 2,
    "tauY": 2,
}


def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mat_vec_mul(m, v):
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def mat_transpose(m):
    return [[m[j][i] for j in range(3)] for i in range(3)]


def vec_add(a, b):
    return [a[i] + b[i] for i in range(3)]


def rotation_matrix_to_euler_zyx(r):
    """Return roll/pitch/yaw for R = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    sy = math.sqrt(r[0][0] * r[0][0] + r[1][0] * r[1][0])
    singular = sy < 1e-6
    if not singular:
        rx = math.atan2(r[2][1], r[2][2])
        ry = math.atan2(-r[2][0], sy)
        rz = math.atan2(r[1][0], r[0][0])
    else:
        rx = math.atan2(-r[1][2], r[1][1])
        ry = math.atan2(-r[2][0], sy)
        rz = 0.0
    return {"rx": rx, "ry": ry, "rz": rz}


def rotation_matrix_to_rodrigues(r):
    trace = r[0][0] + r[1][1] + r[2][2]
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    theta = math.acos(cos_theta)

    if theta < 1e-12:
        return {"rx": 0.0, "ry": 0.0, "rz": 0.0}

    if abs(math.pi - theta) < 1e-5:
        xx = max(0.0, (r[0][0] + 1.0) * 0.5)
        yy = max(0.0, (r[1][1] + 1.0) * 0.5)
        zz = max(0.0, (r[2][2] + 1.0) * 0.5)
        axis = [math.sqrt(xx), math.sqrt(yy), math.sqrt(zz)]
        if r[0][1] < 0.0:
            axis[1] = -axis[1]
        if r[0][2] < 0.0:
            axis[2] = -axis[2]
        norm = math.sqrt(sum(x * x for x in axis))
        if norm < 1e-12:
            axis = [1.0, 0.0, 0.0]
        else:
            axis = [x / norm for x in axis]
    else:
        denom = 2.0 * math.sin(theta)
        axis = [
            (r[2][1] - r[1][2]) / denom,
            (r[0][2] - r[2][0]) / denom,
            (r[1][0] - r[0][1]) / denom,
        ]

    return {"rx": axis[0] * theta, "ry": axis[1] * theta, "rz": axis[2] * theta}


def normalize_matrix(value, name):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a 3x3 matrix")
    matrix = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"{name} must be a 3x3 matrix")
        matrix.append([float(x) for x in row])
    return matrix


def normalize_vec3(value, name):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-value list")
    return [float(x) for x in value]


def list_images(images_dir):
    if images_dir is None:
        return []
    path = Path(images_dir)
    if not path.exists():
        raise FileNotFoundError(f"Images directory not found: {path}")
    return sorted([p.name for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def build_image_name(index, camera, image_names, name_source):
    if name_source == "images" and index < len(image_names):
        return image_names[index]
    if name_source == "camera-info":
        info = str(camera.get("info", "")).strip()
        if info:
            return f"{info}.png"
    return f"{index + 1:03d}.png"


def parse_distortion(values, order):
    d = [float(x) for x in (values or [])]
    distortion = {key: 0.0 for key in DISTORTION_KEYS}

    if order == "opencv":
        # OpenCV common order: k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4.
        mapping = {
            0: "k1",
            1: "k2",
            2: "p1",
            3: "p2",
            4: "k3",
            5: "k4",
            6: "k5",
            7: "k6",
            8: "s1",
            9: "s2",
            10: "s3",
            11: "s4",
        }
    else:
        # Many refined rig files store rational radial terms first.
        mapping = {
            0: "k1",
            1: "k2",
            2: "k3",
            3: "k4",
            4: "k5",
            5: "k6",
            6: "p1",
            7: "p2",
            8: "s1",
            9: "s2",
            10: "s3",
            11: "s4",
        }

    for src_index, key in mapping.items():
        if src_index < len(d):
            distortion[key] = d[src_index]
    return distortion


def compose_camera_pose(rig, camera):
    rig_r = normalize_matrix(rig.get("rig_rotation"), "rig_rotation")
    rig_c = normalize_vec3(rig.get("rig_position"), "rig_position")
    cam_r = normalize_matrix(camera.get("R", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "camera R")
    cam_c = normalize_vec3(camera.get("c", [0.0, 0.0, 0.0]), "camera c")

    r_ctw = mat_mul(rig_r, cam_r)
    center_world = vec_add(rig_c, mat_vec_mul(rig_r, cam_c))
    r_wtc = mat_transpose(r_ctw)
    t_wtc = [-x for x in mat_vec_mul(r_wtc, center_world)]
    return center_world, r_ctw, t_wtc, r_wtc


def parameter(value, name):
    return {"val": float(value), "state": PARAMETER_STATES.get(name, 2)}


def make_calib_camera(out_id, camera, position, rotation, distortion):
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    ar = fy / fx if fx else 1.0
    parameters = {
        "f": parameter(fx, "f"),
        "ar": parameter(ar, "ar"),
        "cx": parameter(camera["cx"], "cx"),
        "cy": parameter(camera["cy"], "cy"),
    }
    for key in DISTORTION_KEYS:
        parameters[key] = parameter(distortion.get(key, 0.0), key)

    model = {
        "polymorphic_id": 2147483649 if out_id == 0 else 1,
        "polymorphic_name": "libCalib::CameraModelOpenCV",
        "ptr_wrapper": {
            "valid": 1,
            "data": {
                "CameraModelCRT": {
                    "CameraModelBase": {
                        "imageSize": {
                            "width": int(camera["w"]),
                            "height": int(camera["h"]),
                        }
                    }
                },
                "parameters": parameters,
            },
        },
    }

    return {
        "model": model,
        "transform": {
            "rotation": {
                "rx": float(rotation["rx"]),
                "ry": float(rotation["ry"]),
                "rz": float(rotation["rz"]),
            },
            "translation": {
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
            },
        },
    }


def make_flat_camera(out_id, camera, image_names, args, position, rotation, distortion):
    return {
        "id": out_id,
        "img_name": build_image_name(out_id, camera, image_names, args.name_source),
        "width": int(camera["w"]),
        "height": int(camera["h"]),
        "position": position,
        "rotation": rotation,
        "fx": float(camera["fx"]),
        "fy": float(camera["fy"]),
        "cx": float(camera["cx"]),
        "cy": float(camera["cy"]),
        "distortion": distortion,
    }


def convert(data, image_names, args):
    rigs = data.get("rigs")
    if not isinstance(rigs, list):
        raise ValueError("Input JSON must contain a top-level 'rigs' list")

    cameras_out = []
    flat_cameras_out = []
    for rig_index, rig in enumerate(rigs):
        rig_cameras = rig.get("cameras", [])
        if not rig_cameras:
            print(f"Warning: rig {rig_index} has no cameras, skipped")
            continue

        for camera in rig_cameras:
            out_id = len(cameras_out)
            center_world, r_ctw, t_wtc, r_wtc = compose_camera_pose(rig, camera)

            if args.pose_output == "ctw-euler":
                position = center_world
                rotation = rotation_matrix_to_euler_zyx(r_ctw)
            else:
                position = t_wtc
                rotation = rotation_matrix_to_rodrigues(r_wtc)

            distortion = parse_distortion(camera.get("D", []), args.distortion_order)
            cameras_out.append(make_calib_camera(out_id, camera, position, rotation, distortion))
            flat_cameras_out.append(make_flat_camera(out_id, camera, image_names, args, position, rotation, distortion))

    if args.output_format == "cameras":
        return {"cameras": flat_cameras_out}

    return {
        "Calibration": {
            "cameras": cameras_out,
            "poses": [],
            "targets": [],
            "distanceConstraints": [],
            "isInitialized": True,
            "isOptimized": True,
        }
    }


def backup_existing(path):
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak_{stamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert refined_rig_group.json to calib0.3343.json-style JSON for self_process_scripts_superglue."
    )
    parser.add_argument("--input", default="data/7.7/refined_rig_group.json", help="Path to refined_rig_group.json")
    parser.add_argument("--images-dir", default="data/7.7/Photo", help="Directory containing images")
    parser.add_argument("--output", default="data/7.7/calib.json", help="Output calib JSON path")
    parser.add_argument(
        "--output-format",
        choices=["calib", "cameras"],
        default="calib",
        help="calib writes Calibration.cameras; cameras writes the older flat cameras.json helper format.",
    )
    parser.add_argument(
        "--pose-output",
        choices=["wtc-rodrigues", "ctw-euler"],
        default="wtc-rodrigues",
        help="wtc-rodrigues works with 3.0convert_to_colmap.py defaults; ctw-euler matches camera-center style.",
    )
    parser.add_argument(
        "--name-source",
        choices=["images", "camera-info", "numeric"],
        default="images",
        help="How to fill img_name. Use images for sorted files from --images-dir.",
    )
    parser.add_argument(
        "--distortion-order",
        choices=["radial-first", "opencv"],
        default="radial-first",
        help="How to map the source D array into k/p/s distortion fields.",
    )
    parser.add_argument("--backup", action="store_true", help="Back up an existing output file before overwriting")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print a summary without writing output")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    image_names = list_images(args.images_dir) if args.name_source == "images" else []
    output_data = convert(data, image_names, args)
    if args.output_format == "calib":
        cameras = output_data["Calibration"]["cameras"]
    else:
        cameras = output_data["cameras"]

    if args.name_source == "images" and len(image_names) != len(cameras):
        print(f"Warning: found {len(image_names)} images but converted {len(cameras)} cameras")

    print(f"Converted {len(cameras)} cameras from {input_path}")
    print(f"Output format: {args.output_format}")
    print(f"Pose output: {args.pose_output}")
    if args.output_format == "cameras" and cameras:
        print(f"First image: {cameras[0]['img_name']}")
        print(f"Last image: {cameras[-1]['img_name']}")

    if args.dry_run:
        return 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.backup:
        backup_path = backup_existing(output_path)
        if backup_path:
            print(f"Backed up existing output to {backup_path}")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
