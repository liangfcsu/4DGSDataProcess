"""COLMAP text readers with explicit world-to-camera semantics."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from track_sparse.core.schema import Camera, ColmapCamera, ColmapImage, Point3D


CAM_RE = re.compile(r"cam[_-]?0*(\d+)", re.IGNORECASE)


def camera_matrix(model: str, params: np.ndarray) -> np.ndarray:
    model = model.upper()
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        if len(params) < 3:
            raise ValueError(f"{model} 内参数量不足")
        f, cx, cy = params[:3]
        fx = fy = f
    elif model in {"PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"}:
        if len(params) < 4:
            raise ValueError(f"{model} 内参数量不足")
        fx, fy, cx, cy = params[:4]
    else:
        raise ValueError(f"暂不支持 COLMAP 相机模型: {model}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def read_cameras_text(path: str | Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5:
                raise ValueError(f"{path}:{line_no}: cameras.txt 行格式错误")
            camera_id = int(fields[0])
            model = fields[1].upper()
            params = np.asarray([float(v) for v in fields[4:]], dtype=np.float64)
            if camera_id in cameras:
                raise ValueError(f"{path}:{line_no}: CAMERA_ID {camera_id} 重复")
            cameras[camera_id] = ColmapCamera(
                camera_id=camera_id,
                model=model,
                width=int(fields[2]),
                height=int(fields[3]),
                params=params,
                K=camera_matrix(model, params),
            )
    if not cameras:
        raise ValueError(f"未从 {path} 读到相机")
    return cameras


def _looks_like_image_metadata(fields: list[str]) -> bool:
    if len(fields) != 10:
        return False
    try:
        int(fields[0])
        [float(v) for v in fields[1:8]]
        int(fields[8])
    except ValueError:
        return False
    return True


def _parse_image_metadata(fields: list[str], path: Path, line_no: int) -> ColmapImage:
    if not _looks_like_image_metadata(fields):
        raise ValueError(f"{path}:{line_no}: images.txt 图像元数据格式错误")
    return ColmapImage(
        image_id=int(fields[0]),
        qvec=np.asarray([float(v) for v in fields[1:5]], dtype=np.float64),
        tvec=np.asarray([float(v) for v in fields[5:8]], dtype=np.float64),
        camera_id=int(fields[8]),
        name=fields[9],
    )


def read_images_text(path: str | Path) -> dict[int, ColmapImage]:
    """Read COLMAP's two-line image records without dropping empty second lines."""
    path = Path(path)
    images: dict[int, ColmapImage] = {}
    pending: ColmapImage | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            stripped = raw.strip()
            if stripped.startswith("#"):
                continue
            fields = stripped.split()
            if pending is None:
                if not fields:
                    continue
                pending = _parse_image_metadata(fields, path, line_no)
                continue

            # A missing/stripped POINTS2D line is seen in real generated models.
            if fields and _looks_like_image_metadata(fields):
                if pending.image_id in images:
                    raise ValueError(f"{path}:{line_no}: IMAGE_ID {pending.image_id} 重复")
                images[pending.image_id] = pending
                pending = _parse_image_metadata(fields, path, line_no)
                continue

            if fields:
                if len(fields) % 3:
                    raise ValueError(f"{path}:{line_no}: POINTS2D 字段数不是 3 的倍数")
                pending.points2d = np.asarray(fields, dtype=np.float64).reshape(-1, 3)
            if pending.image_id in images:
                raise ValueError(f"{path}:{line_no}: IMAGE_ID {pending.image_id} 重复")
            images[pending.image_id] = pending
            pending = None
    if pending is not None:
        if pending.image_id in images:
            raise ValueError(f"{path}: IMAGE_ID {pending.image_id} 重复")
        images[pending.image_id] = pending
    if not images:
        raise ValueError(f"未从 {path} 读到图像位姿")
    return images


def read_points3d_text(path: str | Path, limit: int | None = None) -> dict[int, Point3D]:
    points: dict[int, Point3D] = {}
    path = Path(path)
    if not path.is_file():
        return points
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 8 or (len(fields) - 8) % 2:
                raise ValueError(f"{path}:{line_no}: points3D.txt 行格式错误")
            point_id = int(fields[0])
            track = [(int(fields[i]), int(fields[i + 1])) for i in range(8, len(fields), 2)]
            points[point_id] = Point3D(
                point3d_id=point_id,
                xyz=np.asarray(fields[1:4], dtype=np.float64),
                rgb=np.asarray(fields[4:7], dtype=np.uint8),
                error=float(fields[7]),
                track=track,
            )
            if limit is not None and len(points) >= limit:
                break
    return points


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, dtype=np.float64)
    norm = float(np.linalg.norm(qvec))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("无效的零四元数")
    qw, qx, qy, qz = qvec / norm
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def parse_cam_id(name: str) -> int:
    match = CAM_RE.search(Path(name).name)
    if not match:
        raise ValueError(f"无法从 COLMAP 图像名解析 cam_id: {name}")
    return int(match.group(1))


def load_rig(sparse_dir: str | Path) -> dict[int, Camera]:
    sparse_dir = Path(sparse_dir)
    colmap_cameras = read_cameras_text(sparse_dir / "cameras.txt")
    colmap_images = read_images_text(sparse_dir / "images.txt")
    rig: dict[int, Camera] = {}
    for image in colmap_images.values():
        cam_id = parse_cam_id(image.name)
        if cam_id in rig:
            raise ValueError(f"images.txt 中 cam_id={cam_id} 出现多次")
        if image.camera_id not in colmap_cameras:
            raise ValueError(f"IMAGE_ID {image.image_id} 引用了不存在的 CAMERA_ID {image.camera_id}")
        intrinsic = colmap_cameras[image.camera_id]
        R = qvec_to_rotmat(image.qvec)
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(R), 1.0, atol=1e-7):
            raise ValueError(f"cam_id={cam_id} 的旋转矩阵未通过正交性检查")
        t = image.tvec.astype(np.float64)
        center = -R.T @ t
        P = intrinsic.K @ np.column_stack((R, t))
        rig[cam_id] = Camera(
            cam_id=cam_id,
            image_id=image.image_id,
            camera_id=image.camera_id,
            width=intrinsic.width,
            height=intrinsic.height,
            K=intrinsic.K.copy(),
            R_w2c=R,
            t_w2c=t,
            P=P,
            center_world=center,
            reference_image_name=image.name,
        )
    return dict(sorted(rig.items()))

