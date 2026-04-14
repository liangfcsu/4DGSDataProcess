import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pycolmap
from tqdm import tqdm

from . import logger
from .utils.io import get_keypoints, get_matches


class OutputCapture:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def __enter__(self):
        if not self.verbose:
            pycolmap.logging.alsologtostderr = False

    def __exit__(self, exc_type, *args):
        if not self.verbose:
            pycolmap.logging.alsologtostderr = True


def create_db_from_model(
    reconstruction: pycolmap.Reconstruction, database_path: Path
) -> Dict[str, int]:
    if database_path.exists():
        logger.warning("The database already exists, deleting it.")
        database_path.unlink()

    with pycolmap.Database.open(database_path) as db:
        for _, camera in reconstruction.cameras.items():
            db.write_camera(camera, use_camera_id=True)
        for _, rig in reconstruction.rigs.items():
            db.write_rig(rig, use_rig_id=True)
        for _, frame in reconstruction.frames.items():
            db.write_frame(frame, use_frame_id=True)
        for image_id, image in reconstruction.images.items():
            db.write_image(image, use_image_id=True)
    return {image.name: image_id for image_id, image in reconstruction.images.items()}


def import_features(
    image_ids: Dict[str, int], db: pycolmap.Database, features_path: Path
):
    logger.info("Importing features into the database...")
    for image_name, image_id in tqdm(image_ids.items()):
        keypoints = get_keypoints(features_path, image_name)
        keypoints += 0.5  # COLMAP coordinates origin
        db.write_keypoints(image_id, keypoints)


def import_matches(
    image_ids: Dict[str, int],
    db: pycolmap.Database,
    pairs_path: Path,
    matches_path: Path,
    min_match_score: Optional[float] = None,
    skip_geometric_verification: bool = False,
):
    logger.info("Importing matches into the database...")

    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]

    matched = set()
    for name0, name1 in tqdm(pairs):
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        matches, scores = get_matches(matches_path, name0, name1)
        if min_match_score is not None:
            matches = matches[scores > min_match_score]
        db.write_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}

        if skip_geometric_verification:
            db.write_two_view_geometry(
                id0, id1, pycolmap.TwoViewGeometry(inlier_matches=matches)
            )


def estimation_and_geometric_verification(
    database_path: Path, pairs_path: Path, verbose: bool = False
):
    logger.info("Performing geometric verification of the matches...")
    with OutputCapture(verbose):
        pycolmap.verify_matches(
            database_path,
            pairs_path,
            options=dict(ransac=dict(max_num_trials=20000, min_inlier_ratio=0.1)),
        )


def run_triangulation(
    model_path: Path,
    database_path: Path,
    image_dir: Path,
    reference_model: pycolmap.Reconstruction,
    verbose: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> pycolmap.Reconstruction:
    model_path.mkdir(parents=True, exist_ok=True)
    logger.info("Running 3D triangulation...")
    options = options or {}
    with OutputCapture(verbose):
        reconstruction = pycolmap.triangulate_points(
            reference_model, database_path, image_dir, model_path, options=options
        )
    return reconstruction


def main(
    sfm_dir: Path,
    reference_model: Path,
    image_dir: Path,
    pairs: Path,
    features: Path,
    matches: Path,
    skip_geometric_verification: bool = False,
    estimate_two_view_geometries: bool = False,
    min_match_score: Optional[float] = None,
    verbose: bool = False,
    mapper_options: Optional[Dict[str, Any]] = None,
) -> pycolmap.Reconstruction:
    del estimate_two_view_geometries  # Kept for API compatibility.

    assert reference_model.exists(), reference_model
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"
    reference = pycolmap.Reconstruction(reference_model)

    image_ids = create_db_from_model(reference, database)
    with pycolmap.Database.open(database) as db:
        import_features(image_ids, db, features)
        import_matches(
            image_ids,
            db,
            pairs,
            matches,
            min_match_score,
            skip_geometric_verification,
        )

    if not skip_geometric_verification:
        estimation_and_geometric_verification(database, pairs, verbose)

    reconstruction = run_triangulation(
        sfm_dir, database, image_dir, reference, verbose, mapper_options
    )
    logger.info(
        "Finished the triangulation with statistics:\n%s", reconstruction.summary()
    )
    return reconstruction


def parse_option_args(args: List[str], default_options) -> Dict[str, Any]:
    options = {}
    for arg in args:
        idx = arg.find("=")
        if idx == -1:
            raise ValueError("Options format: key1=value1 key2=value2 etc.")
        key, value = arg[:idx], arg[idx + 1 :]
        if not hasattr(default_options, key):
            raise ValueError(
                f'Unknown option "{key}", allowed options and default values'
                f" for {default_options.summary()}"
            )
        value = eval(value)
        target_type = type(getattr(default_options, key))
        if not isinstance(value, target_type):
            raise ValueError(
                f'Incorrect type for option "{key}":' f" {type(value)} vs {target_type}"
            )
        options[key] = value
    return options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfm_dir", type=Path, required=True)
    parser.add_argument("--reference_sfm_model", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)

    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)

    parser.add_argument("--skip_geometric_verification", action="store_true")
    parser.add_argument("--min_match_score", type=float)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--mapper_options",
        nargs="+",
        default=[],
        help="List of key=value from {}".format(
            pycolmap.IncrementalMapperOptions().todict()
        ),
    )
    args = parser.parse_args().__dict__

    mapper_options = parse_option_args(
        args.pop("mapper_options"), pycolmap.IncrementalMapperOptions()
    )

    main(
        sfm_dir=args["sfm_dir"],
        reference_model=args["reference_sfm_model"],
        image_dir=args["image_dir"],
        pairs=args["pairs"],
        features=args["features"],
        matches=args["matches"],
        skip_geometric_verification=args["skip_geometric_verification"],
        min_match_score=args["min_match_score"],
        verbose=args["verbose"],
        mapper_options=mapper_options,
    )