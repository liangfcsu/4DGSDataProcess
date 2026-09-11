"""Raw multi-camera input preparation for the sparse-track pipeline.

This module intentionally lives under ``track_sparse`` so users only need the
``track_sparse/run.py`` entry point.  The low-level, already-tested extraction
and calibration implementations are loaded from the sibling ``gs_pipeline``
package under a private module name; no 4DGS assembly or per-frame COLMAP job is
run here.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
GS_PIPELINE_ROOT = Path(__file__).resolve().parents[1] / "gs_pipeline"
GS_PIPELINE_PACKAGE = "_track_sparse_gs_pipeline"
CONVERT_CALIB = GS_PIPELINE_ROOT / "stages" / "convert_calib_to_cameras_json.py"


@dataclass(slots=True)
class PreprocessOptions:
    input_dir: Path
    work_dir: Path
    input_type: str
    python: Path
    gpu_ids: list[int | str]
    max_frames: int
    frames_per_second: int | None
    rig_json: Path | None
    calib_json: Path | None
    undistort_workers: int
    resume: bool
    overwrite: bool
    dry_run: bool
    rig_pose_mode: str = "wtc_center"
    rig_radial_mode: str = "direct"
    rig_small_terms: str = "tangential_d8d9"
    rig_direction: str = "ideal_to_observed"
    rig_scale: float = 1.0
    rig_iterations: int = 8


@dataclass(slots=True)
class PreparedInput:
    source_dir: Path
    input_type: str
    dataset_dir: Path
    images_dir: Path | None
    sparse_dir: Path | None
    preprocess_dir: Path | None
    reused: bool
    dry_run: bool
    metadata: dict

    def summary(self) -> dict:
        return {
            "source": str(self.source_dir),
            "input_type": self.input_type,
            "dataset": str(self.dataset_dir),
            "images_dir": str(self.images_dir) if self.images_dir else None,
            "sparse_dir": str(self.sparse_dir) if self.sparse_dir else None,
            "preprocess_dir": str(self.preprocess_dir) if self.preprocess_dir else None,
            "reused": self.reused,
            "dry_run": self.dry_run,
            **self.metadata,
        }

    def tracking_source(self) -> dict:
        """Stable source metadata; unlike ``summary``, resume state is excluded."""
        result = {
            "source": str(self.source_dir),
            "input_type": self.input_type,
            "dataset": str(self.dataset_dir),
            "images_dir": str(self.images_dir) if self.images_dir else None,
            "sparse_dir": str(self.sparse_dir) if self.sparse_dir else None,
        }
        for key in ("preprocess_fingerprint", "calibration", "max_frames"):
            if key in self.metadata:
                result[key] = self.metadata[key]
        return result


def _atomic_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


@lru_cache(maxsize=1)
def _load_gs_pipeline() -> SimpleNamespace:
    """Load sibling gs_pipeline modules without colliding with track_sparse.pipeline."""
    package_dir = GS_PIPELINE_ROOT / "pipeline"
    init_file = package_dir / "__init__.py"
    if not init_file.is_file():
        raise FileNotFoundError(f"缺少 gs_pipeline 包: {package_dir}")
    if GS_PIPELINE_PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            GS_PIPELINE_PACKAGE,
            init_file,
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载 gs_pipeline 包: {package_dir}")
        package = importlib.util.module_from_spec(spec)
        sys.modules[GS_PIPELINE_PACKAGE] = package
        spec.loader.exec_module(package)
    return SimpleNamespace(
        acquire=importlib.import_module(f"{GS_PIPELINE_PACKAGE}.acquire"),
        calibrated=importlib.import_module(f"{GS_PIPELINE_PACKAGE}.calibrated"),
        noncalib=importlib.import_module(f"{GS_PIPELINE_PACKAGE}.noncalib"),
        common=importlib.import_module(f"{GS_PIPELINE_PACKAGE}.common"),
    )


def _prepared_roots(root: Path) -> tuple[Path, Path] | None:
    images = next(
        (path for path in (root / "undistorted", root / "ims", root / "images") if path.is_dir()),
        None,
    )
    sparse = next(
        (
            path
            for path in (
                root / "sparse" / "0",
                root / "calibration" / "reference_sparse" / "0",
                root / "reference_sparse" / "0",
            )
            if (path / "cameras.txt").is_file() and (path / "images.txt").is_file()
        ),
        None,
    )
    return (images.resolve(), sparse.resolve()) if images and sparse else None


def detect_input_type(input_dir: Path, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if _prepared_roots(input_dir) is not None:
        return "prepared"
    if any(path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES for path in input_dir.iterdir()):
        return "videos"
    return "images"


def _calib_schema(path: Path, mode: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"标定文件不存在: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法解析标定文件 {path}: {exc}") from exc
    has_rigs = isinstance(data, dict) and isinstance(data.get("rigs"), list)
    has_frames = isinstance(data, dict) and isinstance(data.get("frames"), list)
    looks_libcalib = isinstance(data, list) or (
        isinstance(data, dict) and ("Calibration" in data or "cameras" in data)
    )
    if mode == "rig":
        if not has_rigs:
            raise ValueError(f"--rig-json 需要含顶层 rigs 列表: {path}")
        return "rig"
    if has_rigs:
        raise ValueError(f"{path} 是 rig 标定；请使用 --rig-json")
    if has_frames:
        return "opencv_frames"
    if looks_libcalib:
        return "libcalib"
    raise ValueError(f"无法识别 --calib-json 结构: {path}")


def _source_files(input_dir: Path, input_type: str) -> list[Path]:
    if input_type == "videos":
        return sorted(
            path for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )
    return sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _source_fingerprint(options: PreprocessOptions, input_type: str, calib_schema: str) -> tuple[str, int]:
    files = _source_files(options.input_dir, input_type)
    if not files:
        kind = "视频" if input_type == "videos" else "图像"
        raise FileNotFoundError(f"输入目录中没有可识别的{kind}: {options.input_dir}")
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        digest.update(
            f"{path.relative_to(options.input_dir)}:{stat.st_size}:{stat.st_mtime_ns}".encode()
        )
    for path in (options.rig_json, options.calib_json):
        if path:
            stat = path.stat()
            digest.update(f"calib:{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    settings = {
        "input_type": input_type,
        "max_frames": options.max_frames,
        "frames_per_second": options.frames_per_second,
        "calib_schema": calib_schema,
        "rig_pose_mode": options.rig_pose_mode,
        "rig_radial_mode": options.rig_radial_mode,
        "rig_small_terms": options.rig_small_terms,
        "rig_direction": options.rig_direction,
        "rig_scale": options.rig_scale,
        "rig_iterations": options.rig_iterations,
    }
    digest.update(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest(), len(files)


def _validate_products(work_dir: Path) -> tuple[Path, Path, int, int]:
    gs = _load_gs_pipeline()
    images_dir = work_dir / "undistorted"
    sparse_dir = work_dir / "calibration" / "reference_sparse" / "0"
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        if not (sparse_dir / name).is_file():
            raise FileNotFoundError(f"预处理产物不完整，缺少: {sparse_dir / name}")
    sequences = gs.common.discover_cam_sequences(images_dir)
    frame_count = gs.common.frame_count_of(sequences)
    return images_dir.resolve(), sparse_dir.resolve(), len(sequences), frame_count


def _run_known_calibration(
    options: PreprocessOptions,
    sequences: dict,
    cam_ids: list[int],
    calib_schema: str,
) -> tuple[Path, Path]:
    gs = _load_gs_pipeline()
    work = options.work_dir
    if options.rig_json:
        cameras = gs.calibrated.load_rig_calibration(options.rig_json, cam_ids, {
            "pose_mode": options.rig_pose_mode,
            "radial_mode": options.rig_radial_mode,
            "small_terms": options.rig_small_terms,
            "direction": options.rig_direction,
            "scale": options.rig_scale,
            "iterations": options.rig_iterations,
        })
    elif calib_schema == "opencv_frames":
        cameras = gs.calibrated.load_opencv_frames_calibration(options.calib_json, cam_ids)
    else:
        if not CONVERT_CALIB.is_file():
            raise FileNotFoundError(f"缺少标定转换脚本: {CONVERT_CALIB}")
        seed = work / "calibration" / "seed_first_frames"
        gs.noncalib._build_first_frames(work / "ims", seed)
        cameras_json = work / "calibration" / "cameras.json"
        gs.common.run_checked([
            str(options.python), str(CONVERT_CALIB),
            "--calib", str(options.calib_json),
            "--images-dir", str(seed),
            "--output", str(cameras_json),
            "--verify-images",
        ])
        cameras = gs.calibrated.load_libcalib_calibration(cameras_json, cam_ids)

    undistorted = work / "undistorted"
    reference_sparse = work / "calibration" / "reference_sparse" / "0"
    new_intrinsics = gs.calibrated.undistort_all(
        cameras, sequences, undistorted, options.undistort_workers
    )
    gs.calibrated.build_reference_sparse(cameras, new_intrinsics, reference_sparse)
    return reference_sparse, undistorted


def prepare_input(options: PreprocessOptions) -> PreparedInput:
    """Normalize raw input and return paths consumable by the tracking pipeline."""
    input_dir = options.input_dir.expanduser().resolve()
    work = options.work_dir.expanduser().resolve()
    options.input_dir = input_dir
    options.work_dir = work
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    if options.max_frames < 0:
        raise ValueError("--max-frames 不能小于 0")
    if options.frames_per_second is not None and options.frames_per_second <= 0:
        raise ValueError("--frames-per-second 必须大于 0")
    if options.undistort_workers <= 0:
        raise ValueError("--undistort-workers 必须大于 0")
    if options.rig_json and options.calib_json:
        raise ValueError("--rig-json 与 --calib-json 不能同时使用")

    input_type = detect_input_type(input_dir, options.input_type)
    if input_type == "prepared":
        if options.rig_json or options.calib_json:
            raise ValueError("已预处理数据集不能再指定 --rig-json/--calib-json")
        roots = _prepared_roots(input_dir)
        if roots is None:
            raise FileNotFoundError(f"{input_dir} 不是完整的已预处理数据集")
        images_dir, sparse_dir = roots
        return PreparedInput(
            source_dir=input_dir,
            input_type=input_type,
            dataset_dir=input_dir,
            images_dir=images_dir,
            sparse_dir=sparse_dir,
            preprocess_dir=None,
            reused=True,
            dry_run=options.dry_run,
            metadata={"preprocessing": "not_required"},
        )
    if input_type not in {"videos", "images"}:
        raise ValueError(f"未知输入类型: {input_type}")
    if work == input_dir or work in input_dir.parents or input_dir in work.parents:
        raise ValueError("预处理目录与原始输入目录不能相同或互相包含")

    calib_schema = "auto_sfm"
    if options.rig_json:
        options.rig_json = options.rig_json.expanduser().resolve()
        calib_schema = _calib_schema(options.rig_json, "rig")
    elif options.calib_json:
        options.calib_json = options.calib_json.expanduser().resolve()
        calib_schema = _calib_schema(options.calib_json, "calib")
    fingerprint, source_count = _source_fingerprint(options, input_type, calib_schema)
    if calib_schema == "auto_sfm" and (input_dir / "poses_bounds.npy").is_file():
        print(
            "  提示: 检测到 poses_bounds.npy；它不含可靠的镜头畸变模型，"
            "本流程仍会用多相机首帧重新估计内外参并据此去畸变。"
        )
    if calib_schema == "auto_sfm" and not options.dry_run and shutil.which("colmap") is None:
        raise RuntimeError("自动首帧 SfM 需要 COLMAP，但当前 PATH 中未找到 colmap")
    manifest_path = work / "preprocess_manifest.json"
    old_manifest = None
    if manifest_path.is_file():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if old_manifest and old_manifest.get("fingerprint") != fingerprint and options.resume:
        raise ValueError(
            "--resume 检测到原始输入或预处理参数已经变化；请换 --output，"
            "或确认后使用 --overwrite"
        )
    if old_manifest and not options.resume and not options.overwrite and not options.dry_run:
        raise FileExistsError(f"预处理缓存已存在；请使用 --resume 或 --overwrite: {work}")

    if options.resume and old_manifest and old_manifest.get("status") == "completed":
        try:
            images_dir, sparse_dir, camera_count, frame_count = _validate_products(work)
            expected = old_manifest.get("stages", {}).get("calibrate_undistort", {})
            if camera_count != int(expected.get("camera_count", camera_count)):
                raise ValueError(
                    f"去畸变相机数变化: {camera_count} != {expected.get('camera_count')}"
                )
            if frame_count != int(expected.get("frame_count", frame_count)):
                raise ValueError(
                    f"去畸变帧数变化: {frame_count} != {expected.get('frame_count')}"
                )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"  ⚠️ 预处理缓存不完整，将从已有抽帧继续修复: {exc}")
            old_manifest["status"] = "running"
        else:
            return PreparedInput(
                source_dir=input_dir,
                input_type=input_type,
                dataset_dir=work,
                images_dir=images_dir,
                sparse_dir=sparse_dir,
                preprocess_dir=work,
                reused=True,
                dry_run=False,
                metadata={
                    "preprocessing": "reused",
                    "preprocess_fingerprint": fingerprint,
                    "camera_count": camera_count,
                    "frame_count": frame_count,
                    "calibration": calib_schema,
                },
            )

    gs = _load_gs_pipeline()
    if options.dry_run:
        print("\n[预处理 1/2] 原始输入 → 规范多相机帧")
        if input_type == "videos":
            gs.acquire.acquire_from_videos(
                input_dir, work / "ims", options.python, options.gpu_ids,
                max_frames=options.max_frames,
                frames_per_second=options.frames_per_second,
                resume=options.resume,
                dry_run=True,
            )
        else:
            gs.acquire.acquire_from_images(input_dir, work / "ims", dry_run=True)
        print(f"\n[预处理 2/2] 首帧标定 + 全帧去畸变（{calib_schema}）")
        if calib_schema == "auto_sfm":
            gs.noncalib.calibrate_and_undistort_sfm(
                work, work / "ims", options.python, options.gpu_ids,
                undistort_workers=options.undistort_workers,
                dry_run=True,
            )
        else:
            print("  将读取已知内外参、生成每相机去畸变映射，并写出固定 COLMAP sparse/0。")
        return PreparedInput(
            source_dir=input_dir,
            input_type=input_type,
            dataset_dir=work,
            images_dir=work / "undistorted",
            sparse_dir=work / "calibration" / "reference_sparse" / "0",
            preprocess_dir=work,
            reused=False,
            dry_run=True,
            metadata={
                "preprocessing": "planned",
                "preprocess_fingerprint": fingerprint,
                "source_file_count": source_count,
                "max_frames": options.max_frames or "all",
                "calibration": calib_schema,
            },
        )

    if options.overwrite and work.is_dir():
        shutil.rmtree(work)
        old_manifest = None
    elif work.exists() and not options.resume and old_manifest is None:
        raise FileExistsError(f"预处理目录已存在但没有清单；请使用 --resume 或 --overwrite: {work}")
    work.mkdir(parents=True, exist_ok=True)
    manifest = old_manifest or {
        "version": 1,
        "status": "running",
        "fingerprint": fingerprint,
        "source": str(input_dir),
        "input_type": input_type,
        "source_file_count": source_count,
        "calibration": calib_schema,
        "settings": {
            key: value
            for key, value in asdict(options).items()
            if key not in {"python", "gpu_ids", "resume", "overwrite", "dry_run"}
        },
        "stages": {},
    }
    # Dataclass paths are not JSON serializable.
    manifest["settings"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in manifest["settings"].items()
    }
    _atomic_json(manifest, manifest_path)

    print("\n[预处理 1/2] 原始输入 → 规范多相机帧")
    ims_dir = work / "ims"
    if input_type == "videos":
        gs.acquire.acquire_from_videos(
            input_dir, ims_dir, options.python, options.gpu_ids,
            max_frames=options.max_frames,
            frames_per_second=options.frames_per_second,
            resume=options.resume,
            dry_run=False,
        )
    else:
        gs.acquire.acquire_from_images(input_dir, ims_dir, dry_run=False)
    sequences = gs.common.discover_cam_sequences(ims_dir)
    frame_count = gs.common.frame_count_of(sequences)
    cam_ids = sorted(sequences)
    manifest["stages"]["extract"] = {
        "status": "completed", "camera_count": len(cam_ids), "frame_count": frame_count
    }
    _atomic_json(manifest, manifest_path)

    print(f"\n[预处理 2/2] 首帧标定 + 全帧去畸变（{calib_schema}）")
    if calib_schema == "auto_sfm":
        reference_sparse, undistorted = gs.noncalib.calibrate_and_undistort_sfm(
            work, ims_dir, options.python, options.gpu_ids,
            undistort_workers=options.undistort_workers,
            dry_run=False,
        )
    else:
        reference_sparse, undistorted = _run_known_calibration(
            options, sequences, cam_ids, calib_schema
        )
    images_dir, sparse_dir, camera_count, output_frame_count = _validate_products(work)
    if reference_sparse.resolve() != sparse_dir or undistorted.resolve() != images_dir:
        raise RuntimeError("预处理内部输出路径不一致")
    manifest["stages"]["calibrate_undistort"] = {
        "status": "completed",
        "camera_count": camera_count,
        "frame_count": output_frame_count,
        "images_dir": str(images_dir),
        "sparse_dir": str(sparse_dir),
    }
    manifest["status"] = "completed"
    _atomic_json(manifest, manifest_path)
    return PreparedInput(
        source_dir=input_dir,
        input_type=input_type,
        dataset_dir=work,
        images_dir=images_dir,
        sparse_dir=sparse_dir,
        preprocess_dir=work,
        reused=False,
        dry_run=False,
        metadata={
            "preprocessing": "completed",
            "preprocess_fingerprint": fingerprint,
            "camera_count": camera_count,
            "frame_count": output_frame_count,
            "calibration": calib_schema,
        },
    )
