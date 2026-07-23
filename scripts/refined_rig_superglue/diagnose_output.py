#!/usr/bin/env python3
"""
Diagnose a refined-rig SuperGlue output directory.
"""

import argparse
import sqlite3
from pathlib import Path

import h5py
import pycolmap


def count_h5_match_pairs(matches_path):
    if not matches_path.exists():
        return None
    pair_count = 0
    match_count = 0
    with h5py.File(matches_path, "r") as f:
        for name0 in f:
            group0 = f[name0]
            if not isinstance(group0, h5py.Group):
                continue
            for name1 in group0:
                group = group0[name1]
                if isinstance(group, h5py.Group) and "matches0" in group:
                    matches0 = group["matches0"][()]
                    pair_count += 1
                    match_count += int((matches0 > -1).sum())
    return pair_count, match_count


def count_database_matches(database_path):
    if not database_path.exists():
        return None
    con = sqlite3.connect(str(database_path))
    try:
        cur = con.cursor()
        raw_pairs, raw_matches = cur.execute(
            "select count(*), coalesce(sum(rows), 0) from matches"
        ).fetchone()
        verified_pairs, verified_matches = cur.execute(
            "select count(*), coalesce(sum(rows), 0) from two_view_geometries"
        ).fetchone()
        return raw_pairs, raw_matches, verified_pairs, verified_matches
    finally:
        con.close()


def print_reconstruction(path, label):
    if not path.exists():
        print(f"{label}: missing ({path})")
        return
    try:
        reconstruction = pycolmap.Reconstruction(path)
    except Exception as exc:
        print(f"{label}: cannot read ({exc})")
        return
    print(f"{label}:")
    print(reconstruction.summary())


def main():
    parser = argparse.ArgumentParser(description="Diagnose SuperGlue triangulation output.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/7.7/refined_rig_superglue/ctw_full_opencv_auto/superglue_output"),
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    temp_dir = output_dir / "temp"
    matches_path = temp_dir / "matches.h5"
    database_path = temp_dir / "sfm" / "database.db"

    h5_counts = count_h5_match_pairs(matches_path)
    if h5_counts is None:
        print(f"matches.h5: missing ({matches_path})")
    else:
        pair_count, match_count = h5_counts
        print(f"matches.h5: {pair_count} pairs, {match_count} raw SuperGlue matches")

    db_counts = count_database_matches(database_path)
    if db_counts is None:
        print(f"database.db: missing ({database_path})")
    else:
        raw_pairs, raw_matches, verified_pairs, verified_matches = db_counts
        print(f"database matches: {raw_pairs} pairs, {raw_matches} imported matches")
        print(f"verified geometries: {verified_pairs} pairs, {verified_matches} inlier matches")

    print_reconstruction(temp_dir / "sfm", "temporary sfm")
    print_reconstruction(output_dir / "sparse" / "0", "exported sparse")


if __name__ == "__main__":
    main()
