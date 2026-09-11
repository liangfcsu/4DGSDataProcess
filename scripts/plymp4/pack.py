#!/usr/bin/env python3
"""Losslessly mux a per-frame Gaussian PLY sequence into an MP4 container.

This is a custom timed metadata codec (sample entry ``gsp1``), not a 2D video
codec.  Each independently decodable MP4 sample contains one complete PLY.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import time
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO, Sequence


SAMPLE_MAGIC = b"GSP1"
SAMPLE_VERSION = 1
SAMPLE_FLAG_ZLIB = 1
SAMPLE_HEADER = struct.Struct(">4sBBHQI")
MP4_EPOCH_OFFSET = 2_082_844_800
COPY_CHUNK_SIZE = 4 * 1024 * 1024


class PackError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlyInfo:
    path: Path
    format: str
    vertex_count: int
    properties: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PackedSample:
    offset: int
    size: int
    original_size: int
    crc32: int


def _box(box_type: bytes, payload: bytes) -> bytes:
    if len(box_type) != 4:
        raise ValueError("MP4 box type must be four bytes")
    total = 8 + len(payload)
    if total < 2**32:
        return struct.pack(">I4s", total, box_type) + payload
    return struct.pack(">I4sQ", 1, box_type, 16 + len(payload)) + payload


def _full_box(box_type: bytes, version: int, flags: int, payload: bytes) -> bytes:
    prefix = bytes((version,)) + int(flags).to_bytes(3, "big")
    return _box(box_type, prefix + payload)


def _unity_matrix() -> bytes:
    return struct.pack(
        ">9I",
        0x00010000, 0, 0,
        0, 0x00010000, 0,
        0, 0, 0x40000000,
    )


def _language_code(language: str = "und") -> int:
    if len(language) != 3 or not language.islower():
        language = "und"
    return sum((ord(char) - 0x60) << shift for char, shift in zip(language, (10, 5, 0)))


def _build_moov(
    samples: Sequence[PackedSample],
    fps: Fraction,
    manifest: dict[str, object],
) -> bytes:
    sample_count = len(samples)
    media_timescale = fps.numerator
    sample_delta = fps.denominator
    media_duration = sample_count * sample_delta
    movie_timescale = 1000
    movie_duration = round(sample_count * movie_timescale / float(fps))
    now = int(time.time()) + MP4_EPOCH_OFFSET

    mvhd_payload = (
        struct.pack(">QQIQ", now, now, movie_timescale, movie_duration)
        + struct.pack(">IhH", 0x00010000, 0x0100, 0)
        + b"\0" * 8
        + _unity_matrix()
        + b"\0" * 24
        + struct.pack(">I", 2)
    )
    mvhd = _full_box(b"mvhd", 1, 0, mvhd_payload)

    tkhd_payload = (
        struct.pack(">QQIIQ", now, now, 1, 0, movie_duration)
        + b"\0" * 8
        + struct.pack(">hhhh", 0, 0, 0, 0)
        + _unity_matrix()
        + struct.pack(">II", 0, 0)
    )
    tkhd = _full_box(b"tkhd", 1, 0x000007, tkhd_payload)

    mdhd_payload = (
        struct.pack(">QQIQ", now, now, media_timescale, media_duration)
        + struct.pack(">HH", _language_code(), 0)
    )
    mdhd = _full_box(b"mdhd", 1, 0, mdhd_payload)
    hdlr = _full_box(
        b"hdlr", 0, 0,
        struct.pack(">I4sIII", 0, b"meta", 0, 0, 0)
        + b"4D Gaussian Splat PLY sequence\0",
    )

    config_json = json.dumps(
        manifest, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    sample_entry = _box(
        b"gsp1",
        b"\0" * 6 + struct.pack(">H", 1) + _box(b"gscC", config_json),
    )
    stsd = _full_box(b"stsd", 0, 0, struct.pack(">I", 1) + sample_entry)
    stts = _full_box(
        b"stts", 0, 0,
        struct.pack(">III", 1, sample_count, sample_delta),
    )
    stsc = _full_box(b"stsc", 0, 0, struct.pack(">IIII", 1, 1, 1, 1))
    stsz = _full_box(
        b"stsz", 0, 0,
        struct.pack(">II", 0, sample_count)
        + b"".join(struct.pack(">I", sample.size) for sample in samples),
    )
    co64 = _full_box(
        b"co64", 0, 0,
        struct.pack(">I", sample_count)
        + b"".join(struct.pack(">Q", sample.offset) for sample in samples),
    )
    stbl = _box(b"stbl", stsd + stts + stsc + stsz + co64)
    nmhd = _full_box(b"nmhd", 0, 0, b"")
    url = _full_box(b"url ", 0, 1, b"")
    dref = _full_box(b"dref", 0, 0, struct.pack(">I", 1) + url)
    dinf = _box(b"dinf", dref)
    minf = _box(b"minf", nmhd + dinf + stbl)
    mdia = _box(b"mdia", mdhd + hdlr + minf)
    trak = _box(b"trak", tkhd + mdia)
    return _box(b"moov", mvhd + trak)


def _read_ply_info(path: Path) -> PlyInfo:
    ply_format = ""
    vertex_count = -1
    properties: list[tuple[str, str]] = []
    in_vertex = False
    try:
        with path.open("rb") as source:
            if source.readline().strip() != b"ply":
                raise PackError(f"不是 PLY 文件：{path}")
            header_bytes = 4
            while True:
                raw = source.readline()
                header_bytes += len(raw)
                if not raw or header_bytes > 4 * 1024 * 1024:
                    raise PackError(f"PLY header 不完整或过大：{path}")
                try:
                    line = raw.decode("ascii").strip()
                except UnicodeDecodeError as exc:
                    raise PackError(f"PLY header 不是 ASCII：{path}") from exc
                fields = line.split()
                if not fields or fields[0] in ("comment", "obj_info"):
                    continue
                if fields[0] == "format" and len(fields) >= 2:
                    ply_format = fields[1]
                elif fields[0] == "element" and len(fields) == 3:
                    in_vertex = fields[1] == "vertex"
                    if in_vertex:
                        vertex_count = int(fields[2])
                elif fields[0] == "property" and in_vertex:
                    if len(fields) == 3:
                        properties.append((fields[2], fields[1]))
                    elif len(fields) == 5 and fields[1] == "list":
                        properties.append((fields[4], f"list:{fields[2]}:{fields[3]}"))
                    else:
                        raise PackError(f"无法解析 PLY property：{path}: {line}")
                elif fields[0] == "end_header":
                    break
    except OSError as exc:
        raise PackError(f"无法读取 {path}：{exc}") from exc

    if not ply_format or vertex_count < 0 or not properties:
        raise PackError(f"PLY 缺少 format/vertex/property：{path}")
    property_names = {name for name, _ in properties}
    required = {"x", "y", "z"}
    if not required <= property_names:
        raise PackError(f"PLY 缺少坐标字段 {sorted(required - property_names)}：{path}")
    return PlyInfo(path, ply_format, vertex_count, tuple(properties))


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def _discover(args: argparse.Namespace) -> list[Path]:
    input_dir = Path(args.input).expanduser().resolve()
    if not input_dir.is_dir():
        raise PackError(f"输入目录不存在：{input_dir}")
    files = sorted((p for p in input_dir.glob(args.pattern) if p.is_file()), key=_natural_key)
    if not files:
        raise PackError(f"{input_dir} 中没有匹配 {args.pattern!r} 的 PLY")
    if args.start < 1 or args.stride < 1 or args.end < 0:
        raise PackError("--start/--stride 必须 >= 1，--end 必须 >= 0")
    end = len(files) if args.end == 0 else args.end
    if end < args.start:
        raise PackError("--end 不能小于 --start")
    selected = files[args.start - 1:end:args.stride]
    if not selected:
        raise PackError("帧范围中没有 PLY 文件")
    return selected


def _write_sample(
    destination: BinaryIO,
    source_path: Path,
    compression: str,
    level: int,
) -> PackedSample:
    sample_offset = destination.tell()
    original_size = source_path.stat().st_size
    destination.write(b"\0" * SAMPLE_HEADER.size)
    checksum = 0
    bytes_read = 0
    compressor = zlib.compressobj(level) if compression == "zlib" else None
    with source_path.open("rb") as source:
        while True:
            chunk = source.read(COPY_CHUNK_SIZE)
            if not chunk:
                break
            bytes_read += len(chunk)
            checksum = zlib.crc32(chunk, checksum)
            encoded = compressor.compress(chunk) if compressor is not None else chunk
            if encoded:
                destination.write(encoded)
    if compressor is not None:
        destination.write(compressor.flush())
    if bytes_read != original_size:
        raise PackError(f"封装期间源文件大小发生变化：{source_path}")

    sample_end = destination.tell()
    flags = SAMPLE_FLAG_ZLIB if compression == "zlib" else 0
    destination.seek(sample_offset)
    destination.write(SAMPLE_HEADER.pack(
        SAMPLE_MAGIC, SAMPLE_VERSION, flags, 0, original_size, checksum & 0xFFFFFFFF
    ))
    destination.seek(sample_end)
    return PackedSample(sample_offset, sample_end - sample_offset, original_size, checksum & 0xFFFFFFFF)


def pack(args: argparse.Namespace) -> Path:
    if args.fps <= 0 or args.fps > 1000:
        raise PackError("--fps 必须在 (0, 1000] 范围内")
    if args.compression_level < 0 or args.compression_level > 9:
        raise PackError("--compression-level 必须在 0..9 范围内")
    paths = _discover(args)
    output = Path(args.output).expanduser().resolve()
    if output.suffix.lower() != ".mp4":
        raise PackError("输出文件必须使用 .mp4 扩展名")
    if output.exists() and not args.overwrite:
        raise PackError(f"输出已存在：{output}；如需覆盖请加 --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"检查 {len(paths)} 帧 PLY schema……")
    infos = [_read_ply_info(path) for path in paths]
    expected_schema = (infos[0].format, infos[0].properties)
    for info in infos[1:]:
        if (info.format, info.properties) != expected_schema:
            raise PackError(
                f"各帧 PLY schema 不一致：{infos[0].path.name} 与 {info.path.name}"
            )

    fps = Fraction(str(args.fps)).limit_denominator(1_000_000)
    property_names = [name for name, _ in infos[0].properties]
    rest_count = sum(name.startswith("f_rest_") for name in property_names)
    coefficient_count = rest_count // 3 + 1
    sh_degree = max(0, int(coefficient_count**0.5) - 1)
    manifest: dict[str, object] = {
        "format": "4dgs-ply-sequence",
        "version": 1,
        "codec": "gsp1",
        "compression": args.compression,
        "fps": {"numerator": fps.numerator, "denominator": fps.denominator},
        "frame_count": len(paths),
        "frame_names": [path.name for path in paths],
        "vertex_counts": [info.vertex_count for info in infos],
        "ply_format": infos[0].format,
        "properties": [[name, kind] for name, kind in infos[0].properties],
        "sh_degree": sh_degree,
        "source_size": sum(path.stat().st_size for path in paths),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    temp = output.with_name(f".{output.name}.part-{os.getpid()}")
    samples: list[PackedSample] = []
    start_time = time.monotonic()
    try:
        with temp.open("w+b") as destination:
            ftyp = _box(b"ftyp", b"isom" + struct.pack(">I", 0x200) + b"isomiso6mp414dgs")
            destination.write(ftyp)
            mdat_start = destination.tell()
            destination.write(struct.pack(">I4sQ", 1, b"mdat", 0))

            for index, path in enumerate(paths, start=1):
                sample = _write_sample(
                    destination, path, args.compression, args.compression_level
                )
                samples.append(sample)
                elapsed = time.monotonic() - start_time
                rate = index / elapsed if elapsed else 0.0
                eta = (len(paths) - index) / rate if rate else 0.0
                ratio = sum(item.size for item in samples) / sum(
                    item.original_size for item in samples
                )
                print(
                    f"\r封装：{index:>4}/{len(paths)}  {rate:5.2f} 帧/秒  "
                    f"比例 {ratio:5.1%}  ETA {eta:6.1f} 秒",
                    end="", flush=True,
                )

            mdat_end = destination.tell()
            destination.seek(mdat_start + 8)
            destination.write(struct.pack(">Q", mdat_end - mdat_start))
            destination.seek(mdat_end)
            destination.write(_build_moov(samples, fps, manifest))
            destination.flush()
            os.fsync(destination.fileno())
        print()
        os.replace(temp, output)
    except BaseException:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        raise

    source_size = int(manifest["source_size"])
    result_size = output.stat().st_size
    print(
        f"完成：{output}\n"
        f"  {len(paths)} 帧，{float(fps):g} fps，时长 {len(paths) / float(fps):.3f} 秒\n"
        f"  源数据 {source_size / 1024**3:.3f} GiB -> MP4 {result_size / 1024**3:.3f} GiB "
        f"({result_size / source_size:.1%})"
    )
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将逐帧 4DGS PLY 无损封装为可随机访问的自定义 MP4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default="data/ply", help="PLY 序列目录")
    parser.add_argument("--output", default="data/sequence_4dgs.mp4", help="输出 MP4")
    parser.add_argument("--pattern", default="*.ply", help="输入 glob")
    parser.add_argument("--fps", type=float, default=30.0, help="容器时间轴帧率")
    parser.add_argument("--start", type=int, default=1, help="起始帧（1 起）")
    parser.add_argument("--end", type=int, default=0, help="结束帧（含）；0=最后")
    parser.add_argument("--stride", type=int, default=1, help="帧采样步长")
    parser.add_argument(
        "--compression", choices=("none", "zlib"), default="zlib",
        help="每帧独立压缩；none 读取最快，zlib 文件较小",
    )
    parser.add_argument(
        "--compression-level", type=int, default=1,
        help="zlib 级别；这类浮点数据提高级别通常收益很小",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        pack(parse_args(argv))
    except PackError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已取消；未完成的临时文件已清理。", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
