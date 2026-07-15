# DLOSeg

Segmentation and geometric reconstruction of **Deformable Linear Objects** (wires, cables) from RGB / stereo images.

The core idea: segment the DLO from an image (text-prompted GroundingDINO + MobileSAMv2), then convert the binary mask into a topological graph and fit **B-splines** to recover each wire as a smooth parametric curve — even through crossings.

📚 **Full documentation lives in [`docs/`](docs/README.md)** — overview, setup, pipeline internals, datasets, and a reference for every script.

```
RGB image ──► segmentors (GroundingDINO + MobileSAMv2) ──► binary mask
binary mask ──► dloseg.graph (skeleton → graph → prune → splines) ──► B-splines per wire
left/right splines ──► dloseg.recon3d (epipolar matching + triangulation) ──► 3D spline
```

## Repository layout

```
src/
  dloseg/                 # the installable Python package
    graph/                # core 2D pipeline: mask → graph → B-splines
      dlo_graph.py        #   DLOGraph — the central data structure
      pipeline.py         #   get_spline() orchestration
      bspline_fitting.py  #   2D spline smoothing/fitting backends
    recon3d/              # 3D reconstruction work
      stereo.py           #   ZED calibration parsing + stereo rectification
      bspline_3d_recon.py #   stereo triangulation of 2D splines → 3D spline
    zed/                  # ZED stereo camera tooling
      calibration_data/   #   camera calibration YAML (used by the pipeline)
      depth/              #   depth / point-cloud extraction scripts
      camera_streaming/   #   network sender/receiver for the camera feed
      record/             #   SVO recording
  segmentors/             # vendored third-party code (MobileSAMv2) + glue
    full_pipe_line.py     #   text-prompted wire segmentation entry point
    GroundingDINO/        #   configs + weights for GroundingDINO
scripts/                  # runnable entry points
  extract_spline.py       #   single mask → B-splines (visual demo)
  benchmark_sbhc.py       #   run the pipeline over the whole SBHC dataset
  extract_spline_video.py #   live spline extraction on a video
  video_to_frames.py      #   dump video frames to images
  zed_svo_export.py       #   export frames from a ZED .svo2 recording
DATASETS/                 # datasets (SBHC: S1/S2/S3 = 1/2/3 wires per image)
outputs/                  # generated masks / results (gitignored)
archive/                  # retired experiments (gitignored, kept locally)
```

## Setup

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> && cd DLOSeg
uv sync                       # creates .venv and installs everything declared
```

Extras that `uv sync` does **not** manage:

- **pyzed** (only for the ZED camera scripts) — installed by the ZED SDK:
  `.venv/bin/python /usr/local/zed/get_python_api.py`
  Re-run this after any `uv sync`, which prunes undeclared packages.
- **Model checkpoints** for segmentation (gitignored) — place under
  `src/segmentors/MobileSAMv2/weight/` and `src/segmentors/GroundingDINO/weights/`
  (see `src/segmentors/full_pipe_line.py --help` for the expected files).

## Quick start

```bash
# Fit B-splines to a ground-truth mask and plot them (needs a display)
uv run scripts/extract_spline.py

# Benchmark the full SBHC dataset (headless-safe)
MPLBACKEND=Agg uv run scripts/benchmark_sbhc.py

# Text-prompted wire segmentation on a folder of images (needs checkpoints + GPU)
uv run src/segmentors/full_pipe_line.py --img_path <folder>
```

All scripts resolve dataset paths from the repo root — they can be launched from any directory. Tunable parameters live in the `config` dict at the top of each script's `__main__` block.

## The pipeline in one paragraph

`DLOGraph.load_from_mask` pads and morphologically cleans the mask, thins it to a 1-px skeleton (Guo–Hall), builds a k-NN graph over skeleton pixels and reduces it to an MST. `prune_short_branches_and_delete_junctions` removes noise branches and junction nodes, leaving simple paths. Each path is smoothed with a parametric B-spline, then `reconstruct_dlo_2` re-connects branch endpoints across removed junctions by minimizing total turning angle — this is what disentangles wire crossings. Finally `fit_bspline_to_graph` fits one B-spline per leaf-to-leaf path; results land in `graph.full_bsplines`. On SBHC the full pipeline runs at ~50 FPS per 256×256 mask (CPU).
