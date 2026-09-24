#!/usr/bin/env python3
"""Launch track_sparse with execution controls read from the same YAML config."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - depends on the invoking runtime
    raise SystemExit(
        "启动脚本需要 PyYAML；请使用 /opt/4dgs-player/env/bin/python3.11 运行"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RUN_SCRIPT = SCRIPT_DIR / "run.py"

VALUE_OPTIONS = {
    "input": "--input",
    "input_type": "--input-type",
    "images_dir": "--images-dir",
    "sparse_dir": "--sparse-dir",
    "persparse_dir": "--persparse-dir",
    "output": "--output",
    "preprocess_dir": "--preprocess-dir",
    "gpu": "--gpu",
    "max_frames": "--max-frames",
    "frames_per_second": "--frames-per-second",
    "rig_json": "--rig-json",
    "calib_json": "--calib-json",
    "undistort_workers": "--undistort-workers",
    "rig_pose_mode": "--rig-pose-mode",
    "rig_radial_mode": "--rig-radial-mode",
    "rig_small_terms": "--rig-small-terms",
    "rig_direction": "--rig-direction",
    "rig_scale": "--rig-scale",
    "rig_iterations": "--rig-iterations",
    "start_frame": "--start-frame",
    "end_frame": "--end-frame",
    "cameras": "--cameras",
    "stages": "--stages",
    "debug_frames": "--debug-frames",
    "log_level": "--log-level",
}

PATH_OPTIONS = {
    "input",
    "images_dir",
    "sparse_dir",
    "persparse_dir",
    "output",
    "preprocess_dir",
    "rig_json",
    "calib_json",
}

FLAG_OPTIONS = {
    "resume": "--resume",
    "overwrite": "--overwrite",
    "preprocess_only": "--preprocess-only",
    "dry_run": "--dry-run",
}

LIST_OPTIONS = {"cameras", "stages", "debug_frames"}
SPECIAL_OPTIONS = {"python"}


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise ValueError(f"无法读取配置文件 {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML 格式错误 {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("YAML 根节点必须是映射")
    run = payload.get("run")
    if not isinstance(run, dict):
        raise ValueError("配置文件必须包含 run: 执行参数")
    return run


def _resolve_project_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def build_command(config_path: str | Path) -> list[str]:
    """Build the exact run.py command without starting the workload."""

    config = Path(config_path).expanduser().resolve()
    run = _read_yaml(config)
    unknown = sorted(set(run) - set(VALUE_OPTIONS) - set(FLAG_OPTIONS) - SPECIAL_OPTIONS)
    if unknown:
        raise ValueError("run 中存在未知参数: " + ", ".join(unknown))
    if bool(run.get("resume")) and bool(run.get("overwrite")):
        raise ValueError("run.resume 与 run.overwrite 不能同时为 true")
    if not run.get("output"):
        raise ValueError("run.output 不能为空")
    has_input = bool(run.get("input"))
    has_prepared_pair = bool(run.get("images_dir")) and bool(run.get("sparse_dir"))
    if has_input == has_prepared_pair:
        raise ValueError(
            "run 中必须提供 input，或同时提供 images_dir+sparse_dir，且不能混用"
        )

    python_value = run.get("python") or sys.executable
    python_path = _resolve_project_path(python_value)
    if not python_path.is_file():
        raise ValueError(f"Python 解释器不存在: {python_path}")

    command = [str(python_path), str(RUN_SCRIPT), "--config", str(config)]
    for key, option in VALUE_OPTIONS.items():
        value = run.get(key)
        if value is None or value == "":
            continue
        if key in PATH_OPTIONS:
            value = _resolve_project_path(value)
        elif key in LIST_OPTIONS:
            if isinstance(value, (list, tuple)):
                value = ",".join(str(item) for item in value)
            elif not isinstance(value, str):
                raise ValueError(f"run.{key} 必须是列表或逗号分隔字符串")
        command.extend((option, str(value)))
    for key, option in FLAG_OPTIONS.items():
        value = run.get(key, False)
        if not isinstance(value, bool):
            raise ValueError(f"run.{key} 必须是 true 或 false")
        if value:
            command.append(option)
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="从单个 YAML 启动 track_sparse 完整处理流程"
    )
    parser.add_argument("config", type=Path, help="含 run: 和算法参数的 YAML")
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="只打印最终 run.py 命令，不启动处理",
    )
    args = parser.parse_args(argv)
    try:
        command = build_command(args.config)
    except ValueError as exc:
        parser.error(str(exc))
    print("\n启动 track_sparse：\n")
    print("$ " + shlex.join(command), flush=True)
    if args.print_only:
        return 0
    os.chdir(PROJECT_ROOT)
    os.execv(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
