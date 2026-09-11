"""标定分支：把已知内外参（rig 标定 / libCalib 标定）统一为
「逐帧去畸变 + PINHOLE 参考稀疏模型」。整段代码不依赖 torch/hloc，
仅需 numpy/cv2/stdlib，因此可脱离 GPU 环境单独测试。

两种标定来源共用一个内部结构 CameraCalib，再走同一套
undistort_all() 与 build_reference_sparse()：

- rig（refined_rig_group.json）：有理径向畸变模型
  radial = (1+k1 r²+k2 r⁴+k3 r⁶)/(1+k4 r²+k5 r⁶+k6 r⁶)，
  该模型 COLMAP/OpenCV 无法表达，其实现已从 refined_rig_superglue 原样内联到本模块。
- libCalib（calib*.json → 扁平 cameras.json）：标准多项式径向+切向畸变，
  用 cv2 去畸变，位姿沿用 3.0convert_to_colmap 的 CTW-Euler→WTC 约定。
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .common import (
    PipelineError,
    cam_dirname,
    parse_cam_id,
    reference_image_name,
)


# ---------------------------------------------------------------------------
# rig 位姿合成（自 refined_rig_superglue/convert_refined_rig_to_colmap.py 原样内联，
# 纯标准库；把 rig 外参 + 每相机 R/c 组合为 COLMAP world-to-camera 位姿）
# ---------------------------------------------------------------------------

def _mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _mat_vec_mul(m, v):
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def _mat_transpose(m):
    return [[m[j][i] for j in range(3)] for i in range(3)]


def _vec_add(a, b):
    return [a[i] + b[i] for i in range(3)]


def _normalize_matrix(value, name):
    if not isinstance(value, list) or len(value) != 3 or any(
        not isinstance(row, list) or len(row) != 3 for row in value
    ):
        raise ValueError(f"{name} must be a 3x3 matrix")
    return [[float(x) for x in row] for row in value]


def _normalize_vec3(value, name):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-value list")
    return [float(x) for x in value]


def _quaternion_from_rotation_matrix(r):
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


def compose_pose(rig, cam, pose_mode):
    """rig 外参 + 相机 R/c → COLMAP (qw,qx,qy,qz,tx,ty,tz)（world-to-camera）。"""
    rig_r = _normalize_matrix(rig["rig_rotation"], "rig_rotation")
    rig_t = _normalize_vec3(rig["rig_position"], "rig_position")
    cam_r = _normalize_matrix(cam.get("R", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "camera R")
    cam_c = _normalize_vec3(cam.get("c", [0.0, 0.0, 0.0]), "camera c")
    flip_yz = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    base = pose_mode.replace("_flip_yz", "")

    if base == "ctw":
        r_ctw = _mat_mul(rig_r, cam_r)
        center = _vec_add(rig_t, _mat_vec_mul(rig_r, cam_c))
        r_wtc = _mat_transpose(r_ctw)
        t_wtc = [-x for x in _mat_vec_mul(r_wtc, center)]
    elif base == "wtc":
        r_wtc = _mat_mul(rig_r, cam_r)
        t_wtc = _vec_add(rig_t, _mat_vec_mul(rig_r, cam_c))
    elif base == "wtc_center":
        r_wtc = _mat_mul(rig_r, cam_r)
        center = _vec_add(rig_t, _mat_vec_mul(rig_r, cam_c))
        t_wtc = [-x for x in _mat_vec_mul(r_wtc, center)]
    elif base == "ctw_translation":
        r_ctw = _mat_mul(rig_r, cam_r)
        r_wtc = _mat_transpose(r_ctw)
        t_wtc = _vec_add(rig_t, _mat_vec_mul(rig_r, cam_c))
    else:
        raise ValueError(f"Unsupported pose mode: {pose_mode}")

    if pose_mode.endswith("_flip_yz"):
        r_wtc = _mat_mul(flip_yz, r_wtc)
        t_wtc = _mat_vec_mul(flip_yz, t_wtc)

    qw, qx, qy, qz = _quaternion_from_rotation_matrix(r_wtc)
    return qw, qx, qy, qz, t_wtc[0], t_wtc[1], t_wtc[2]


# ---------------------------------------------------------------------------
# 有理径向畸变模型（自 refined_rig_superglue/undistort_refined_rig.py 原样内联，
# 仅依赖 numpy，便于本包自包含与脱离 cv2 环境测试位姿/内参）
#   radial = (1 + k1 r² + k2 r⁴ + k3 r⁶) / (1 + k4 r² + k5 r⁴ + k6 r⁶)
# COLMAP/OpenCV 的多项式模型无法表达该分式形式，因此必须保留此实现。
# ---------------------------------------------------------------------------

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
    return {"k1": k1, "k2": k2, "k3": k3, "k4": k4, "k5": k5, "k6": k6,
            "d6": values[6], "d7": values[7], "d8": values[8], "d9": values[9], "d10": values[10]}


def _apply_model(x, y, coeffs, small_terms):
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    num = 1.0 + coeffs["k1"] * r2 + coeffs["k2"] * r4 + coeffs["k3"] * r6
    den = 1.0 + coeffs["k4"] * r2 + coeffs["k5"] * r4 + coeffs["k6"] * r6
    radial = num / np.where(np.abs(den) < 1e-12, 1e-12, den)
    xd = x * radial
    yd = y * radial
    if small_terms == "tangential_d8d9":
        p1, p2 = coeffs["d8"], coeffs["d9"]
        xd = xd + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = yd + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    elif small_terms == "thin_prism_d8d9":
        s1, s2 = coeffs["d8"], coeffs["d9"]
        xd = xd + s1 * r2 + s2 * r4
    elif small_terms == "none":
        pass
    else:
        raise ValueError(f"Unsupported small-terms mode: {small_terms}")
    return xd, yd


def _invert_model(x_target, y_target, coeffs, small_terms, iterations):
    x = x_target.copy()
    y = y_target.copy()
    for _ in range(iterations):
        xd, yd = _apply_model(x, y, coeffs, small_terms)
        x += x_target - xd
        y += y_target - yd
    return x, y


# ---------------------------------------------------------------------------
# 多项式（libCalib）去畸变（自 self_process_scripts_superglue/2.0undistort_images.py 的
# undistort_colmap_equiv 原样移植）：主点强制居中、按四角射线拟合新焦距、迭代逆映射。
# 之所以不用 cv2.undistort/getOptimalNewCameraMatrix：libCalib 的畸变约定与 OpenCV 相反，
# 直接喂给 cv2 会得到几何错误的去畸变图（特征匹配变差、点数骤降）。
# ---------------------------------------------------------------------------

def _distort_points(x, y, k1, k2, k3, p1, p2):
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    x_dist = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_dist = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return x_dist, y_dist


def polynomial_build_map(width, height, fx, fy, cx, cy, k1, k2, k3, p1, p2, iter_count=5):
    """返回 (map_x, map_y, (fx_new,fy_new,cx_new,cy_new))；主点居中，新焦距覆盖四角射线。"""
    W, H = int(width), int(height)
    # Step1: 拟合四角射线得到新焦距，主点强制居中
    max_x = max_y = 0.0
    for u, v in ((0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)):
        x = (u - cx) / fx
        y = (v - cy) / fy
        x_d, y_d = _distort_points(x, y, k1, k2, k3, p1, p2)
        max_x = max(max_x, abs(x_d))
        max_y = max(max_y, abs(y_d))
    fx_new = (W / 2) / max_x
    fy_new = (H / 2) / max_y
    cx_new = W / 2
    cy_new = H / 2
    # Step2: 迭代逆映射构建去畸变表
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    x_n = (u - cx_new) / fx_new
    y_n = (v - cy_new) / fy_new
    x, y = x_n.copy(), y_n.copy()
    for _ in range(iter_count):
        x_d, y_d = _distort_points(x, y, k1, k2, k3, p1, p2)
        x = x_n - (x_d - x)
        y = y_n - (y_d - y)
    map_x = (x * fx + cx).astype(np.float32)
    map_y = (y * fy + cy).astype(np.float32)
    return map_x, map_y, (fx_new, fy_new, cx_new, cy_new)


def rational_build_map(width, height, fx, fy, cx, cy, coeffs, direction, small_terms, iterations, scale):
    new_fx = fx * scale
    new_fy = fy * scale
    new_cx = width * 0.5
    new_cy = height * 0.5
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    x = (u - new_cx) / new_fx
    y = (v - new_cy) / new_fy
    if direction == "ideal_to_observed":
        xs, ys = _apply_model(x, y, coeffs, small_terms)
    elif direction == "observed_to_ideal":
        xs, ys = _invert_model(x, y, coeffs, small_terms, iterations)
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    map_x = xs * fx + cx
    map_y = ys * fy + cy
    return map_x.astype(np.float32), map_y.astype(np.float32), (new_fx, new_fy, new_cx, new_cy)


# ---------------------------------------------------------------------------
# 内部相机结构
# ---------------------------------------------------------------------------

@dataclass
class CameraCalib:
    cam_id: int                       # 物理相机编号，用于输出命名
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str                        # 'rational' | 'polynomial' | 'opencv'
    pose: tuple                       # COLMAP world-to-camera (qw,qx,qy,qz,tx,ty,tz)
    coeffs: dict | None = None        # rational: parse_d 结果
    dist: list | None = None          # polynomial: [k1,k2,p1,p2,k3]
    rational_opts: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 位姿转换（libCalib）：rotation 是 world-to-camera 的 Rodrigues(轴角)向量，
# translation 就是 t_wtc（不是相机中心）。等价于 3.0convert_to_colmap 的默认
# --quaternion-method rodrigues --pose-format wtc。
# 该约定用非标定 SfM 的相机位姿作真值验证过：对齐 RMS 0.028（场景尺度 ~10），
# 而「CTW 欧拉 + 相机中心」的解释 RMS 0.73，明显错误。
# ---------------------------------------------------------------------------

def _rodrigues_to_quaternion(rx: float, ry: float, rz: float) -> tuple[float, float, float, float]:
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-12:
        return 1.0, 0.0, 0.0, 0.0
    half = angle * 0.5
    s = math.sin(half) / angle
    return math.cos(half), rx * s, ry * s, rz * s


def rodrigues_wtc_to_colmap(rotation, position) -> tuple:
    """WTC Rodrigues 旋转 + t_wtc → COLMAP (qw,qx,qy,qz,tx,ty,tz)。"""
    qw, qx, qy, qz = _rodrigues_to_quaternion(rotation["rx"], rotation["ry"], rotation["rz"])
    return qw, qx, qy, qz, float(position[0]), float(position[1]), float(position[2])


# ---------------------------------------------------------------------------
# 标定加载：把不同来源统一成按 cam_ids 顺序对齐的 CameraCalib 列表
# ---------------------------------------------------------------------------

def load_rig_calibration(rig_json: Path, cam_ids: list[int], opts: dict) -> list[CameraCalib]:
    """refined_rig_group.json → CameraCalib[]（有理畸变）。

    rig 内相机按文件顺序展平，与 cam_ids（已排序）逐个配对。
    """
    import json

    data = json.loads(Path(rig_json).read_text(encoding="utf-8"))
    rigs = data.get("rigs")
    if not isinstance(rigs, list):
        raise PipelineError("rig JSON 缺少顶层 rigs 列表")

    flat: list[tuple[dict, dict]] = [(rig, cam) for rig in rigs for cam in rig.get("cameras", [])]
    if len(flat) != len(cam_ids):
        raise PipelineError(
            f"rig 相机数 {len(flat)} 与输入相机数 {len(cam_ids)} 不一致，无法按序配对"
        )

    pose_mode = opts.get("pose_mode", "wtc_center")
    radial_mode = opts.get("radial_mode", "direct")
    cameras: list[CameraCalib] = []
    for cam_id, (rig, cam) in zip(cam_ids, flat):
        pose = compose_pose(rig, cam, pose_mode)
        coeffs = parse_d(cam.get("D", []), radial_mode)
        cameras.append(CameraCalib(
            cam_id=cam_id,
            width=int(cam["w"]),
            height=int(cam["h"]),
            fx=float(cam["fx"]),
            fy=float(cam["fy"]),
            cx=float(cam["cx"]),
            cy=float(cam["cy"]),
            model="rational",
            pose=pose,
            coeffs=coeffs,
            rational_opts={
                "direction": opts.get("direction", "ideal_to_observed"),
                "small_terms": opts.get("small_terms", "tangential_d8d9"),
                "iterations": int(opts.get("iterations", 8)),
                "scale": float(opts.get("scale", 1.0)),
            },
        ))
    return cameras


def load_libcalib_calibration(cameras_json: Path, cam_ids: list[int]) -> list[CameraCalib]:
    """扁平 cameras.json（1.convert_calib_to_cameras_json 产物）→ CameraCalib[]。

    每台相机 position=相机中心(world)，rotation=CTW ZYX 欧拉角；转为 COLMAP WTC。
    畸变按 OpenCV [k1,k2,p1,p2,k3]。
    """
    import json

    data = json.loads(Path(cameras_json).read_text(encoding="utf-8"))
    entries = data["cameras"] if isinstance(data, dict) and "cameras" in data else data
    if not isinstance(entries, list):
        raise PipelineError("libCalib cameras.json 结构无法识别（期望 {cameras:[...]} 或列表）")
    if len(entries) != len(cam_ids):
        raise PipelineError(
            f"标定相机数 {len(entries)} 与输入相机数 {len(cam_ids)} 不一致，无法按序配对"
        )

    cameras: list[CameraCalib] = []
    for cam_id, entry in zip(cam_ids, entries):
        rotation = entry.get("rotation", {"rx": 0.0, "ry": 0.0, "rz": 0.0})
        position = entry.get("position", [0.0, 0.0, 0.0])
        pose = rodrigues_wtc_to_colmap(rotation, position)
        dist_raw = entry.get("distortion", {})
        if isinstance(dist_raw, dict):
            dist = [dist_raw.get("k1", 0.0), dist_raw.get("k2", 0.0),
                    dist_raw.get("p1", 0.0), dist_raw.get("p2", 0.0), dist_raw.get("k3", 0.0)]
        else:
            padded = list(dist_raw) + [0.0] * 5
            dist = [padded[0], padded[1], padded[2], padded[3], padded[4]]
        cameras.append(CameraCalib(
            cam_id=cam_id,
            width=int(entry["width"]),
            height=int(entry["height"]),
            fx=float(entry["fx"]),
            fy=float(entry["fy"]),
            cx=float(entry["cx"]),
            cy=float(entry["cy"]),
            model="polynomial",
            pose=pose,
            dist=[float(x) for x in dist],
        ))
    return cameras


def _matrix4(value, name: str) -> list[list[float]]:
    """接受展平的 16 项或 4x4 嵌套矩阵。"""
    if isinstance(value, list) and len(value) == 16:
        values = [float(x) for x in value]
        return [values[i:i + 4] for i in range(0, 16, 4)]
    if isinstance(value, list) and len(value) == 4 and all(
        isinstance(row, list) and len(row) == 4 for row in value
    ):
        return [[float(x) for x in row] for row in value]
    raise PipelineError(f"{name} 必须是展平的 16 项或 4x4 矩阵")


def _pose_from_frame_matrix(entry: dict, cam_id: int) -> tuple:
    """frames 标定的 w2c/c2w → COLMAP world-to-camera 位姿。"""
    if entry.get("w2c") is not None:
        w2c = _matrix4(entry["w2c"], f"相机 {cam_id} w2c")
        rotation = [row[:3] for row in w2c[:3]]
        translation = [w2c[i][3] for i in range(3)]
    elif entry.get("c2w") is not None:
        c2w = _matrix4(entry["c2w"], f"相机 {cam_id} c2w")
        rotation_ctw = [row[:3] for row in c2w[:3]]
        rotation = _mat_transpose(rotation_ctw)
        center = [c2w[i][3] for i in range(3)]
        translation = [-x for x in _mat_vec_mul(rotation, center)]
    else:
        raise PipelineError(f"相机 {cam_id} 缺少 w2c/c2w 位姿矩阵")
    qw, qx, qy, qz = _quaternion_from_rotation_matrix(rotation)
    return qw, qx, qy, qz, translation[0], translation[1], translation[2]


def load_opencv_frames_calibration(calib_json: Path, cam_ids: list[int]) -> list[CameraCalib]:
    """加载 ``{frames:[...]}`` OpenCV 标定。

    每项通过 filename/camera_id 与输入的 camXXX 对齐，直接复用 w2c；不会按
    JSON 数组顺序配对。两边完整且只有恒定编号偏移时自动修正偏移，避免字典序、
    缺号或采集/标定编号规则不同导致外参套到错误相机。
    """
    import json

    data = json.loads(Path(calib_json).read_text(encoding="utf-8"))
    frames = data.get("frames") if isinstance(data, dict) else None
    if not isinstance(frames, list) or not frames:
        raise PipelineError("frames 标定 JSON 缺少非空顶层 frames 列表")

    by_cam_id: dict[int, dict] = {}
    for index, entry in enumerate(frames):
        if not isinstance(entry, dict):
            raise PipelineError(f"frames[{index}] 不是对象")
        filename_id = parse_cam_id(str(entry.get("filename", "")))
        explicit_id = entry.get("camera_id")
        try:
            explicit_id = int(explicit_id) if explicit_id is not None else None
        except (TypeError, ValueError) as exc:
            raise PipelineError(f"frames[{index}].camera_id 不是整数") from exc
        if filename_id is not None and explicit_id is not None and filename_id != explicit_id:
            raise PipelineError(
                f"frames[{index}] 编号冲突: filename={entry.get('filename')}，"
                f"camera_id={explicit_id}"
            )
        cam_id = filename_id if filename_id is not None else explicit_id
        if cam_id is None:
            raise PipelineError(f"frames[{index}] 无法解析相机编号")
        if cam_id in by_cam_id:
            raise PipelineError(f"frames 标定包含重复相机编号: {cam_id}")
        by_cam_id[cam_id] = entry

    input_ids = set(cam_ids)
    calib_ids = set(by_cam_id)
    ordered_input = sorted(input_ids)
    ordered_calib = sorted(calib_ids)

    # 某些采集系统用视频序号 cam_1..cam_N，而标定系统用设备号
    # cam_2..cam_(N+1)。只有两边数量完全相同且排序后一一呈唯一恒定偏移时，
    # 才自动应用该偏移；其他不完整情况仍严格按同号匹配，避免静默错配外参。
    offsets = {calib_id - input_id
               for input_id, calib_id in zip(ordered_input, ordered_calib)}
    use_offset = len(ordered_input) == len(ordered_calib) and len(offsets) == 1
    id_offset = next(iter(offsets)) if use_offset else 0
    if use_offset:
        entry_by_input = {
            input_id: by_cam_id[input_id + id_offset] for input_id in ordered_input
        }
        matched_ids = ordered_input
        if id_offset:
            sign = "+" if id_offset > 0 else ""
            print(
                f"  ⚠️ 检测到标定编号整体偏移 {sign}{id_offset}，自动映射: "
                f"输入 cam{ordered_input[0]:03d} → 标定 cam{ordered_input[0] + id_offset:03d}，"
                f"共 {len(matched_ids)} 台"
            )
    else:
        matched_ids = sorted(input_ids & calib_ids)
        entry_by_input = {cam_id: by_cam_id[cam_id] for cam_id in matched_ids}
        if not matched_ids:
            raise PipelineError(
                f"frames 标定与输入相机编号没有交集：输入={ordered_input[:8]}，"
                f"标定={ordered_calib[:8]}"
            )
        missing = sorted(input_ids - calib_ids)
        extra = sorted(calib_ids - input_ids)
        if missing:
            print(f"  ⚠️ {len(missing)} 台输入相机没有标定，已跳过: "
                  f"{', '.join('cam%03d' % x for x in missing[:12])}"
                  f"{' …' if len(missing) > 12 else ''}")
        if extra:
            print(f"  ⚠️ 标定文件中 {len(extra)} 台相机没有对应输入，已忽略: "
                  f"{', '.join('cam%03d' % x for x in extra[:12])}"
                  f"{' …' if len(extra) > 12 else ''}")

    cameras: list[CameraCalib] = []
    for cam_id in matched_ids:
        entry = entry_by_input[cam_id]
        intrinsic = entry.get("intrinsic")
        if isinstance(intrinsic, list) and len(intrinsic) == 3 and all(
            isinstance(row, list) and len(row) == 3 for row in intrinsic
        ):
            intrinsic = [x for row in intrinsic for x in row]
        if not isinstance(intrinsic, list) or len(intrinsic) != 9:
            raise PipelineError(f"相机 {cam_id} intrinsic 必须是 9 项或 3x3 矩阵")
        image_size = entry.get("image_size")
        if not isinstance(image_size, list) or len(image_size) != 2:
            raise PipelineError(f"相机 {cam_id} image_size 必须是 [width, height]")

        model = str(entry.get("camera_model", "OPENCV")).upper()
        if model not in {"OPENCV", "FULL_OPENCV"}:
            raise PipelineError(f"相机 {cam_id} 暂不支持 camera_model={model}")
        dist_raw = entry.get("distortion", [])
        if isinstance(dist_raw, dict):
            dist = [dist_raw.get(name, 0.0) for name in
                    ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6",
                     "s1", "s2", "s3", "s4")]
        elif isinstance(dist_raw, list):
            dist = dist_raw
        else:
            raise PipelineError(f"相机 {cam_id} distortion 必须是列表或对象")

        cameras.append(CameraCalib(
            cam_id=cam_id,
            width=int(image_size[0]),
            height=int(image_size[1]),
            fx=float(intrinsic[0]),
            fy=float(intrinsic[4]),
            cx=float(intrinsic[2]),
            cy=float(intrinsic[5]),
            model="opencv",
            pose=_pose_from_frame_matrix(entry, cam_id),
            dist=[float(x) for x in dist],
        ))
    print(f"  已加载 frames/OpenCV 标定: {len(cameras)} 台相机")
    return cameras


# ---------------------------------------------------------------------------
# 去畸变映射（两种模型统一成 map_x/map_y + 新内参）
# ---------------------------------------------------------------------------

def build_undistort_map(calib: CameraCalib):
    """返回 (map_x, map_y, (fx,fy,cx,cy), (width,height))，供 cv2.remap 复用。"""
    w, h = calib.width, calib.height
    if calib.model == "rational":
        opts = calib.rational_opts
        map_x, map_y, new_k = rational_build_map(
            w, h, calib.fx, calib.fy, calib.cx, calib.cy, calib.coeffs,
            opts["direction"], opts["small_terms"], opts["iterations"], opts["scale"],
        )
        new_fx, new_fy, new_cx, new_cy = new_k
        return map_x, map_y, (new_fx, new_fy, new_cx, new_cy), (w, h)

    if calib.model == "polynomial":
        # libCalib：dist = [k1, k2, p1, p2, k3]
        k1, k2, p1, p2, k3 = (list(calib.dist) + [0.0] * 5)[:5]
        map_x, map_y, new_k = polynomial_build_map(
            w, h, calib.fx, calib.fy, calib.cx, calib.cy, k1, k2, k3, p1, p2)
        return map_x, map_y, new_k, (w, h)

    if calib.model == "opencv":
        import cv2

        camera_matrix = np.array([
            [calib.fx, 0.0, calib.cx],
            [0.0, calib.fy, calib.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        distortion = np.asarray(calib.dist or [0.0] * 5, dtype=np.float64)
        new_matrix, _roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, distortion, (w, h), alpha=0.0
        )
        # 下游 3DGS（含 FastGS）只用 fx/fy 推 FoV、假定主点位于图像中心，完全忽略
        # cx/cy；COLMAP getOptimalNewCameraMatrix + ROI 裁剪会把主点留在非中心处，
        # 导致每台相机渲染都有几十像素的平移，多视角互相打架、训练糊成一团。
        # 这里强制主点居中，并保持整幅尺寸（不做 ROI 裁剪），与 rational/polynomial
        # 两个分支的居中约定保持一致。
        #
        # 另外 getOptimalNewCameraMatrix 会分别拟合水平/垂直缩放，即使原始标定
        # fx≈fy（方形像素）也会凭空造出 ~2% 的 fx≠fy 各向异性；3DGS 用 fx 推 FovX、
        # fy 推 FovY，会把每帧纵向拉伸。取两者较小值作为统一焦距，恢复方形像素、
        # 且不引入额外裁剪。
        new_matrix = new_matrix.copy()
        focal = min(new_matrix[0, 0], new_matrix[1, 1])
        new_matrix[0, 0] = focal
        new_matrix[1, 1] = focal
        new_matrix[0, 2] = w / 2.0
        new_matrix[1, 2] = h / 2.0
        map_x, map_y = cv2.initUndistortRectifyMap(
            camera_matrix, distortion, None, new_matrix, (w, h), cv2.CV_32FC1
        )
        new_k = (new_matrix[0, 0], new_matrix[1, 1],
                 new_matrix[0, 2], new_matrix[1, 2])
        return map_x, map_y, new_k, (w, h)

    raise PipelineError(f"未知畸变模型: {calib.model}")


# ---------------------------------------------------------------------------
# 逐帧去畸变 + 参考稀疏模型
# ---------------------------------------------------------------------------

def undistort_all(cameras: list[CameraCalib], sequences: dict[int, list[Path]],
                  out_dir: Path, workers: int) -> dict[int, tuple]:
    """对每台相机的所有帧套用同一张去畸变映射。

    返回 {cam_id: (fx,fy,cx,cy,width,height)}（去畸变后的新内参）。
    """
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    new_intrinsics: dict[int, tuple] = {}

    def process_camera(calib: CameraCalib) -> tuple[int, tuple]:
        frames = sequences.get(calib.cam_id, [])
        if not frames:
            raise PipelineError(f"相机 {calib.cam_id} 没有可去畸变的帧")
        map_x, map_y, intr, (w, h) = build_undistort_map(calib)
        cam_out = out_dir / cam_dirname(calib.cam_id)
        cam_out.mkdir(parents=True, exist_ok=True)
        for frame_path in frames:
            img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if img is None:
                raise PipelineError(f"无法读取图像: {frame_path}")
            if (img.shape[1], img.shape[0]) != (w, h):
                # 标定尺寸与实拍尺寸不一致时以实际图像为准，重建映射。
                calib.width, calib.height = img.shape[1], img.shape[0]
                map_x, map_y, intr, (w, h) = build_undistort_map(calib)
            undistorted = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            ok, buffer = cv2.imencode(".png", undistorted)
            if not ok:
                raise PipelineError(f"编码去畸变图像失败: {frame_path.name}")
            buffer.tofile(str(cam_out / frame_path.name))
        return calib.cam_id, (intr[0], intr[1], intr[2], intr[3], w, h)

    errors = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(process_camera, calib): calib for calib in cameras}
        for future in as_completed(futures):
            try:
                cam_id, intr = future.result()
                new_intrinsics[cam_id] = intr
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
    if errors:
        raise PipelineError("去畸变失败: " + "; ".join(errors[:5]))
    return new_intrinsics


def build_reference_sparse(cameras: list[CameraCalib], new_intrinsics: dict[int, tuple],
                           out_sparse_dir: Path) -> None:
    """写出 PINHOLE 参考稀疏模型 sparse/0/{cameras,images,points3D}.txt。

    每台相机一条记录，位姿为已知外参，points3D 留空（供逐帧三角化填充）。
    图像名用 reference_image_name(cam_id)，逐帧引擎按 cam 编号对齐。
    """
    out_sparse_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(cameras, key=lambda c: c.cam_id)

    with (out_sparse_dir / "cameras.txt").open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(ordered)}\n")
        for i, calib in enumerate(ordered):
            fx, fy, cx, cy, w, h = new_intrinsics[calib.cam_id]
            f.write(f"{i + 1} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")

    with (out_sparse_dir / "images.txt").open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(ordered)}, mean observations per image: 0\n")
        for i, calib in enumerate(ordered):
            qw, qx, qy, qz, tx, ty, tz = calib.pose
            name = reference_image_name(calib.cam_id)
            f.write(
                f"{i + 1} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                f"{tx:.10f} {ty:.10f} {tz:.10f} {i + 1} {name}\n\n"
            )

    with (out_sparse_dir / "points3D.txt").open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0, mean track length: 0\n")
