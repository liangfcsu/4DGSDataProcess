#!/usr/bin/env python3
"""
Undistort images using intrinsics from calib.json (libCalib format).
Assumes calib.json extrinsics are world-to-camera (w2c), but undistortion uses intrinsics only.

Usage:
  python scripts/self_process_scropts_agisoft/1.undistort_from_calib.py \
    --calib data1.15/origin/calib/calib0.4723.json \
    --images_dir data1.15/origin/lf \
    --output_dir data1.15/process/agisoft/undistorted \
    --pattern "*.png"
    --scale 1.0

Notes:
- Reads fx, fy (or f & ar), cx, cy, and distortion (k1,k2,k3,k4,k5,k6,p1,p2) per camera entry.
- Uses OpenCV's standard pinhole distortion model: [k1,k2,p1,p2,k3].
- Higher-order k4..k6 are ignored by cv2.undistort.
- If image dimensions differ from the intrinsics' recorded width/height, we scale fx,fy,cx,cy accordingly.

If you need rectification to a common plane using extrinsics, that's a separate step.
"""

import argparse
import json
import os
from pathlib import Path
import glob
from typing import Dict, Any, Tuple
import copy

import cv2
import numpy as np


def extract_intrinsics(cam_entry: Dict[str, Any]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[int, int], np.ndarray]:
    """Extract (fx, fy), (cx, cy), (width, height), dist_coeffs from a camera entry."""
    model = cam_entry.get('model', {})
    data = model.get('ptr_wrapper', {}).get('data', {})
    params = data.get('parameters', {})
    crt = data.get('CameraModelCRT', {})
    base = crt.get('CameraModelBase', {})
    img_size = base.get('imageSize', {})
    width = int(img_size.get('width', 0))
    height = int(img_size.get('height', 0))

    f = params.get('f', {}).get('val')
    ar = params.get('ar', {}).get('val', 1.0)
    fx = float(f) if f is not None else float(params.get('fx', {}).get('val', 0.0))
    fy = float(fx * ar) if f is not None else float(params.get('fy', {}).get('val', 0.0))
    cx = float(params.get('cx', {}).get('val', 0.0))
    cy = float(params.get('cy', {}).get('val', 0.0))

    # Validate intrinsics
    if fx <= 0 or fy <= 0:
        raise ValueError(f"Invalid focal length: fx={fx}, fy={fy}. Must be positive.")
    if cx < 0 or cy < 0:
        print(f"[WARN] Negative principal point: cx={cx}, cy={cy}")

    # Distortion coefficients
    k1 = float(params.get('k1', {}).get('val', 0.0))
    k2 = float(params.get('k2', {}).get('val', 0.0))
    p1 = float(params.get('p1', {}).get('val', 0.0))
    p2 = float(params.get('p2', {}).get('val', 0.0))
    k3 = float(params.get('k3', {}).get('val', 0.0))
    # OpenCV standard undistort uses only first 5; k4..k6 ignored here
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    return (fx, fy), (cx, cy), (width, height), dist


def build_camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K


def scale_intrinsics(K: np.ndarray, from_size: Tuple[int, int], to_size: Tuple[int, int]) -> np.ndarray:
    """Scale K if image resolution differs from the one stored in calib."""
    w0, h0 = from_size
    w1, h1 = to_size
    if w0 == 0 or h0 == 0 or (w0 == w1 and h0 == h1):
        return K.copy()
    sx = w1 / w0
    sy = h1 / h0
    K_scaled = K.copy()
    K_scaled[0, 0] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[0, 2] *= sx
    K_scaled[1, 2] *= sy
    return K_scaled


def undistort_image_unified(img: np.ndarray, K: np.ndarray, dist: np.ndarray, unified_K: np.ndarray, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Undistort image using original intrinsics but output to unified intrinsics"""
    h, w = img.shape[:2]
    
    # Check if there's actually any distortion to correct
    has_distortion = np.any(np.abs(dist) > 1e-6)
    
    if not has_distortion:
        # No distortion - just apply optional scale to the unified K matrix
        new_K = unified_K.copy()
        if scale != 1.0:
            new_K[0, 0] *= scale
            new_K[1, 1] *= scale
            new_K[0, 2] *= scale
            new_K[1, 2] *= scale
        return img.copy(), new_K
    else:
        # Has distortion - perform undistortion with unified target intrinsics
        new_K = unified_K.copy()
        if scale != 1.0:
            new_K[0, 0] *= scale
            new_K[1, 1] *= scale
            new_K[0, 2] *= scale
            new_K[1, 2] *= scale
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, R=None, newCameraMatrix=new_K, size=(w, h), m1type=cv2.CV_32FC1)
        undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        return undist, new_K


def undistort_image(img: np.ndarray, K: np.ndarray, dist: np.ndarray, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    
    # Check if there's actually any distortion to correct
    has_distortion = np.any(np.abs(dist) > 1e-6)
    
    if not has_distortion:
        # No distortion - just apply optional scale to the original K matrix
        new_K = K.copy()
        if scale != 1.0:
            new_K[0, 0] *= scale
            new_K[1, 1] *= scale
            new_K[0, 2] *= scale
            new_K[1, 2] *= scale
        return img.copy(), new_K
    else:
        # Has distortion - perform actual undistortion
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
        if scale != 1.0:
            new_K = new_K.copy()
            new_K[0, 0] *= scale
            new_K[1, 1] *= scale
            new_K[0, 2] *= scale
            new_K[1, 2] *= scale
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, R=None, newCameraMatrix=new_K, size=(w, h), m1type=cv2.CV_32FC1)
        undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        return undist, new_K
    return undist, new_K


def str2bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    ap = argparse.ArgumentParser(description='Undistort images using calib.json intrinsics (w2c extrinsics ignored).')
    ap.add_argument('--calib', type=str, default='calib_0.5052.json', help='Path to calib.json')
    ap.add_argument('--images_dir', type=str, default='output/binary_masks', help='Input images directory')
    ap.add_argument('--output_dir', type=str, default='output/undistorted_binary_masks', help='Output directory for undistorted images')
    ap.add_argument('--pattern', type=str, default='*.png', help='Glob pattern for images')
    ap.add_argument('--scale', type=float, default=1.0, help='Optional scale factor applied to new camera matrix')
    ap.add_argument('--preserve_alpha', type=str2bool, default=True, help='Preserve and remap alpha channel if present (PNG with transparency)')
    ap.add_argument('--unify_intrinsics', type=str2bool, default=True, help='Use unified average intrinsics for all cameras')
    args = ap.parse_args()

    with open(args.calib, 'r') as f:
        calib = json.load(f)

    cams = calib.get('Calibration', {}).get('cameras', [])
    if not cams:
        raise RuntimeError('No cameras found in calib.json')

    # Collect images
    img_paths = sorted(glob.glob(str(Path(args.images_dir) / args.pattern)))
    if not img_paths:
        print(f'No images found in {args.images_dir} matching {args.pattern}')
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Found {len(img_paths)} images. Undistorting to {args.output_dir} ...')

    # If count matches, map 1:1; else use the first camera intrinsics for all
    per_cam = len(img_paths) == len(cams)

    # Calculate unified intrinsics if requested
    unified_intrinsics = None
    if args.unify_intrinsics:
        print("Computing unified intrinsics from all cameras...")
        fx_list, fy_list, cx_list, cy_list = [], [], [], []
        for cam in cams:
            (fx, fy), (cx, cy), _, _ = extract_intrinsics(cam)
            fx_list.append(fx)
            fy_list.append(fy)
            cx_list.append(cx)
            cy_list.append(cy)
        
        avg_fx = np.mean(fx_list)
        avg_fy = np.mean(fy_list)
        avg_cx = np.mean(cx_list)
        avg_cy = np.mean(cy_list)
        unified_intrinsics = (avg_fx, avg_fy, avg_cx, avg_cy)
        print(f"Unified intrinsics: fx={avg_fx:.2f}, fy={avg_fy:.2f}, cx={avg_cx:.2f}, cy={avg_cy:.2f}")

    # collect new intrinsics per image
    undist_intrinsics = {}

    for i, img_path in enumerate(img_paths):
        # Use UNCHANGED to keep alpha if present
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'[WARN] Failed to read {img_path}')
            continue
        cam_idx = i if per_cam else 0
        (fx, fy), (cx, cy), (w0, h0), dist = extract_intrinsics(cams[cam_idx])
        K = build_camera_matrix(fx, fy, cx, cy)
        h, w = img.shape[:2]
        K_scaled = scale_intrinsics(K, (w0, h0), (w, h))
        
        # Choose undistortion method based on unify_intrinsics flag
        if args.unify_intrinsics:
            # Use unified intrinsics as target
            unified_K = build_camera_matrix(*unified_intrinsics)
            unified_K_scaled = scale_intrinsics(unified_K, (w0, h0), (w, h))
            
            # If alpha channel exists and user requested preservation, split and remap all channels
            if args.preserve_alpha and img.ndim == 3 and img.shape[2] == 4:
                bgr = img[:, :, :3]
                alpha = img[:, :, 3]
                undist_bgr, new_K = undistort_image_unified(bgr, K_scaled, dist, unified_K_scaled, scale=args.scale)
                # Remap alpha with same maps for consistency - use the SAME new_K from BGR undistortion
                map1, map2 = cv2.initUndistortRectifyMap(K_scaled, dist, R=None, newCameraMatrix=new_K, size=(w, h), m1type=cv2.CV_32FC1)
                undist_alpha = cv2.remap(alpha, map1, map2, interpolation=cv2.INTER_NEAREST)
                undist = np.dstack([undist_bgr, undist_alpha])
            else:
                undist, new_K = undistort_image_unified(img, K_scaled, dist, unified_K_scaled, scale=args.scale)
        else:
            # Use original per-camera undistortion logic
            # If alpha channel exists and user requested preservation, split and remap all channels
            if args.preserve_alpha and img.ndim == 3 and img.shape[2] == 4:
                bgr = img[:, :, :3]
                alpha = img[:, :, 3]
                undist_bgr, new_K = undistort_image(bgr, K_scaled, dist, scale=args.scale)
                # Remap alpha with same maps for consistency - use the SAME new_K from BGR undistortion
                map1, map2 = cv2.initUndistortRectifyMap(K_scaled, dist, R=None, newCameraMatrix=new_K, size=(w, h), m1type=cv2.CV_32FC1)
                undist_alpha = cv2.remap(alpha, map1, map2, interpolation=cv2.INTER_NEAREST)
                undist = np.dstack([undist_bgr, undist_alpha])
            else:
                undist, new_K = undistort_image(img, K_scaled, dist, scale=args.scale)
        
        out_path = Path(args.output_dir) / Path(img_path).name
        cv2.imwrite(str(out_path), undist)
        print(f'  - {Path(img_path).name} -> {out_path}')

        # record intrinsics for this image
        fx_new = float(new_K[0, 0])
        fy_new = float(new_K[1, 1])
        cx_new = float(new_K[0, 2])
        cy_new = float(new_K[1, 2])
        undist_intrinsics[Path(img_path).name] = {
            'width': int(undist.shape[1]),
            'height': int(undist.shape[0]),
            'fx': fx_new, 'fy': fy_new, 'cx': cx_new, 'cy': cy_new,
            'distortion': {'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0, 'k3': 0.0}
        }

    # write calib for undistorted outputs, preserving original format and extrinsics
    # We will clone the original calib structure and update camera intrinsics
    calib_clone = copy.deepcopy(calib)  # More efficient deep copy
    cams_clone = calib_clone.get('Calibration', {}).get('cameras', [])

    def set_param(params_dict: Dict[str, Any], key: str, val: float):
        if key in params_dict and isinstance(params_dict[key], dict):
            # libCalib format: params[key] = { "val": number, ... }
            params_dict[key]['val'] = float(val)
        else:
            # if missing, create it minimally
            params_dict[key] = {'val': float(val)}

    for i, cam_entry in enumerate(cams_clone):
        # map camera to corresponding image's new intrinsics when available
        img_name = None
        if per_cam and i < len(img_paths):
            img_name = Path(img_paths[i]).name
        else:
            # fallback to first image's intrinsics if counts mismatch
            img_name = Path(img_paths[0]).name

        new_intr = undist_intrinsics.get(img_name)
        if not new_intr:
            continue

        model = cam_entry.get('model', {})
        data = model.get('ptr_wrapper', {}).get('data', {})
        params = data.setdefault('parameters', {})
        
        # If unified intrinsics are used, all cameras get the same intrinsics
        if args.unify_intrinsics:
            # Use the unified intrinsics (should be the same for all images)
            new_fx = new_intr['fx']
            new_fy = new_intr['fy']
            new_cx = new_intr['cx'] 
            new_cy = new_intr['cy']
        else:
            # Use individual camera intrinsics
            new_fx = new_intr['fx']
            new_fy = new_intr['fy']
            new_cx = new_intr['cx']
            new_cy = new_intr['cy']
        
        new_ar = new_fy / new_fx if new_fx > 0 else 1.0
        
        # Update intrinsics parameters - use f + ar format to preserve aspect ratio
        set_param(params, 'f', new_fx)  # Set f = fx
        set_param(params, 'ar', new_ar)  # Set ar = fy/fx
        set_param(params, 'fx', new_fx)
        set_param(params, 'fy', new_fy)
        set_param(params, 'cx', new_cx)
        set_param(params, 'cy', new_cy)
        # Zero distortion (OpenCV k1,k2,p1,p2,k3)
        set_param(params, 'k1', 0.0)
        set_param(params, 'k2', 0.0)
        set_param(params, 'p1', 0.0)
        set_param(params, 'p2', 0.0)
        set_param(params, 'k3', 0.0)
        # preserve/adjust imageSize if present
        crt = data.get('CameraModelCRT', {})
        base = crt.get('CameraModelBase', {})
        img_size = base.get('imageSize', {})
        img_size['width'] = int(new_intr['width'])
        img_size['height'] = int(new_intr['height'])
        base['imageSize'] = img_size
        crt['CameraModelBase'] = base
        data['CameraModelCRT'] = crt
        model['ptr_wrapper']['data'] = data
        cam_entry['model'] = model

    out_json = Path(args.output_dir) / 'calib_undistorted.json'
    with out_json.open('w') as jf:
        json.dump(calib_clone, jf, indent=2)
    print(f'Wrote undistorted calib (preserving extrinsics and format) to {out_json}')

    print('Done.')


if __name__ == '__main__':
    main()
