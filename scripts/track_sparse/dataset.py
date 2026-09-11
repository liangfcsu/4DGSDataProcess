"""Dataset discovery and frame/camera indexing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .schema import Camera, ImageRecord


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
CAM_RE = re.compile(r"cam[_-]?0*(\d+)", re.IGNORECASE)
FRAME_RE = re.compile(r"frame[_-]?0*(\d+)", re.IGNORECASE)


@dataclass(slots=True)
class DatasetIndex:
    images_dir: Path
    sparse_dir: Path
    images: dict[tuple[int, int], ImageRecord]
    cam_ids: list[int]
    frame_ids: list[int]
    missing: list[tuple[int, int]]

    def get(self, cam_id: int, frame_id: int) -> ImageRecord | None:
        return self.images.get((cam_id, frame_id))

    def image_names(self) -> list[str]:
        return [self.images[key].relative_name for key in sorted(self.images)]


def _find_images_dir(dataset: Path | None, override: Path | None) -> Path:
    if override:
        return override.expanduser().resolve()
    if dataset is None:
        raise ValueError("必须提供 --input/--dataset 或 --images-dir")
    candidates = [dataset / "ims", dataset / "undistorted", dataset / "images"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"无法在 {dataset} 下找到 ims/、undistorted/ 或 images/")


def _find_sparse_dir(dataset: Path | None, override: Path | None) -> Path:
    if override:
        result = override.expanduser().resolve()
        if (result / "0" / "cameras.txt").is_file():
            result = result / "0"
        return result
    if dataset is None:
        raise ValueError("必须提供 --input/--dataset 或 --sparse-dir")
    candidates = [
        dataset / "sparse" / "0",
        dataset / "calibration" / "reference_sparse" / "0",
        dataset / "reference_sparse" / "0",
    ]
    for candidate in candidates:
        if (candidate / "cameras.txt").is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"无法在 {dataset} 下找到 COLMAP sparse/0 标定")


def discover_dataset(
    dataset: str | Path | None,
    images_dir: str | Path | None,
    sparse_dir: str | Path | None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    camera_filter: list[int] | None = None,
) -> DatasetIndex:
    dataset_path = Path(dataset).expanduser().resolve() if dataset else None
    images_root = _find_images_dir(dataset_path, Path(images_dir) if images_dir else None)
    sparse_root = _find_sparse_dir(dataset_path, Path(sparse_dir) if sparse_dir else None)
    images: dict[tuple[int, int], ImageRecord] = {}
    for path in sorted(images_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        cam_match = CAM_RE.search(path.name) or CAM_RE.search(path.parent.name)
        frame_match = FRAME_RE.search(path.name)
        if not cam_match or not frame_match:
            continue
        cam_id, frame_id = int(cam_match.group(1)), int(frame_match.group(1))
        if camera_filter is not None and cam_id not in camera_filter:
            continue
        if start_frame is not None and frame_id < start_frame:
            continue
        if end_frame is not None and frame_id > end_frame:
            continue
        key = (cam_id, frame_id)
        if key in images:
            raise ValueError(f"重复图像键 cam={cam_id}, frame={frame_id}: {images[key].path} / {path}")
        images[key] = ImageRecord(
            cam_id=cam_id,
            frame_id=frame_id,
            path=path.resolve(),
            relative_name=path.relative_to(images_root).as_posix(),
        )
    if not images:
        raise FileNotFoundError(f"{images_root} 中未发现 camXXXframeYYY 图像")
    cam_ids = sorted({key[0] for key in images})
    frame_ids = sorted({key[1] for key in images})
    missing = [(cam, frame) for cam in cam_ids for frame in frame_ids if (cam, frame) not in images]
    return DatasetIndex(images_root, sparse_root, images, cam_ids, frame_ids, missing)


def validate_dataset(index: DatasetIndex, rig: dict[int, Camera], tolerance_px: int = 1) -> list[str]:
    warnings: list[str] = []
    missing_calib = sorted(set(index.cam_ids) - set(rig))
    if missing_calib:
        raise ValueError(f"以下图像相机没有固定标定: {missing_calib}")
    for cam_id in index.cam_ids:
        records = [record for (cam, _), record in index.images.items() if cam == cam_id]
        if not records:
            continue
        with Image.open(records[0].path) as image:
            width, height = image.size
        for record in records:
            record.width, record.height = width, height
        camera = rig[cam_id]
        dw, dh = abs(width - camera.width), abs(height - camera.height)
        if dw > tolerance_px or dh > tolerance_px:
            raise ValueError(
                f"cam{cam_id:03d} 图像尺寸 {width}x{height} 与标定 "
                f"{camera.width}x{camera.height} 不一致"
            )
        if dw or dh:
            warnings.append(
                f"cam{cam_id:03d}: 图像 {width}x{height} 与标定 {camera.width}x{camera.height} "
                f"相差 {dw}x{dh} 像素（在容差内）"
            )
    if index.missing:
        warnings.append(f"发现 {len(index.missing)} 个缺帧位置；将按真实 frame_id 保留 missing")
    return warnings
