# refined_rig_superglue

This folder is an isolated pipeline for datasets like `data/7.7/refined_rig_group.json`.
It does not write into `scripts/self_process_scripts_superglue`; it only calls the
existing SuperGlue/HLoc runner there.

## Recommended path for `data/7.7`

The `D` field in `refined_rig_group.json` should not be pushed directly into
COLMAP `OPENCV` or `FULL_OPENCV`. In this file, `D[0:3]` and `D[3:6]` behave
like the numerator and denominator of a rational radial model. The tested
undistortion uses:

```text
x_observed = x_ideal * (1 + D0*r2 + D1*r4 + D2*r6)
             / (1 + D3*r2 + D4*r4 + D5*r6)
y_observed = y_ideal * (1 + D0*r2 + D1*r4 + D2*r6)
             / (1 + D3*r2 + D4*r4 + D5*r6)
```

with `D[8]` and `D[9]` as small tangential terms. After undistortion, the
downstream sparse model is written as `PINHOLE`.

Run:

```bash
/usr/local/miniconda/envs/tool/bin/python scripts/refined_rig_superglue/run_undistorted_pipeline.py \
  --clean \
  --resize-max 1600 \
  --max-keypoints 2048 \
  --num-workers 0
```

Tested result on `data/7.7`:

```text
variant: data/7.7/refined_rig_undistorted_superglue/ideal_to_observed_direct_tangential_d8d9_scale1_wtc_center
points3D: 13764
mean_reprojection_error: 1.58384 px
3dgs data root: data/7.7/refined_rig_undistorted_superglue/ideal_to_observed_direct_tangential_d8d9_scale1_wtc_center/3dgs
```

The 3DGS data root contains `images -> ../images` and
`sparse -> ../superglue_output/sparse`, so it has the usual:

```text
3dgs/images/
3dgs/sparse/0/
```

## Prepare a COLMAP reference model

```bash
/usr/local/miniconda/envs/tool/bin/python scripts/refined_rig_superglue/run_pipeline.py \
  --prepare-only
```

Default output:

```text
data/7.7/refined_rig_superglue/wtc_center_pinhole_auto/sparse/0
```

## Run SuperGlue triangulation

```bash
/usr/local/miniconda/envs/tool/bin/python scripts/refined_rig_superglue/run_pipeline.py \
  --clean \
  --resize-max 1600 \
  --max-keypoints 2048
```

Default output:

```text
data/7.7/refined_rig_superglue/wtc_center_pinhole_auto/superglue_output/pointcloud.ply
```

For `data/7.7/refined_rig_group.json`, the tested working setup is:

```text
pose-mode: wtc_center
camera-model: PINHOLE
```

This uses the original `Photo` images without undistortion and ignores `D`.
The direct `OPENCV` and `FULL_OPENCV` mappings of `D` produced 0 points in tests.

## If triangulation returns 0 points

The recommended diagnostic is to reuse one set of SuperGlue matches and try
multiple pose interpretations:

```bash
/usr/local/miniconda/envs/tool/bin/python scripts/refined_rig_superglue/run_trials.py \
  --resize-max 1600 \
  --max-keypoints 2048 \
  --clean-variant
```

This keeps the original `data/7.7/Photo` images untouched and triangulates with
the refined-rig intrinsics/extrinsics directly. Outputs are written to:

```text
data/7.7/refined_rig_superglue_trials/
```

On `data/7.7`, the best tested trial was:

```text
data/7.7/refined_rig_superglue_trials/wtc_center_pinhole_auto/sparse/0
data/7.7/refined_rig_superglue_trials/wtc_center_pinhole_auto/pointcloud.ply
```

It reconstructed 9125 points with mean reprojection error about 1.59 px.

Try pose convention first:

```bash
/usr/local/miniconda/envs/tool/bin/python scripts/refined_rig_superglue/run_pipeline.py \
  --pose-mode wtc \
  --clean \
  --resize-max 1600 \
  --max-keypoints 2048
```

Try tangential distortion mapping:

```bash
/usr/local/miniconda/envs/tool/bin/python scripts/refined_rig_superglue/run_pipeline.py \
  --tangential-source d8d9 \
  --clean \
  --resize-max 1600 \
  --max-keypoints 2048
```

Try no distortion as a diagnostic only:

```bash
/usr/local/miniconda/envs/tool/bin/python scripts/refined_rig_superglue/run_pipeline.py \
  --camera-model PINHOLE \
  --clean \
  --resize-max 1600 \
  --max-keypoints 2048
```

If `PINHOLE` gives points but `FULL_OPENCV` does not, the likely issue is the
distortion model/order. If `wtc` gives points but `ctw` does not, the likely issue
is pose convention.

## Diagnose a finished run

```bash
/usr/local/miniconda/envs/tool/bin/python scripts/refined_rig_superglue/diagnose_output.py \
  --output-dir data/7.7/refined_rig_superglue/ctw_full_opencv_auto/superglue_output
```

This reports raw SuperGlue matches, verified two-view geometries, and the final
pycolmap reconstruction summary.
