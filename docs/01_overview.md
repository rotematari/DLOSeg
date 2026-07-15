# 01 — Overview

DLOSeg reconstructs **Deformable Linear Objects** (wires, cables) from images as smooth parametric curves. The end goal is a 3D B-spline per wire, obtained from a stereo camera, robust to wires crossing each other.

## The full picture

```
                ┌─────────────────────────────┐
 RGB image ────►│ segmentors/                 │────► binary mask (per wire)
                │ GroundingDINO ("wire.")     │
                │  → boxes → MobileSAMv2      │
                └─────────────────────────────┘
                ┌─────────────────────────────┐
 binary mask ──►│ dloseg.recon3d.             │────► 2D B-splines (per wire)
                │ skeleton → k-NN graph → MST │
                │  → prune → smooth → resolve │
                │  crossings → fit B-splines  │
                └─────────────────────────────┘
                ┌─────────────────────────────┐
 left+right ───►│ dloseg.recon3d.             │────► 3D B-spline
 2D splines     │ bspline_3d_recon            │
                │ epipolar match → triangulate│
                └─────────────────────────────┘
```

Each stage works standalone:

- **Segmentation** ([04](04_segmentation.md)) needs GPU + model checkpoints; produces masks like those already in `outputs/`.
- **Graph pipeline** ([03](03_pipeline_graph.md)) is pure CPU (numpy/scipy/networkx/OpenCV), ~50 FPS on 256×256 masks. This is the scientific core of the repo.
- **Stereo/3D** ([05](05_stereo_3d.md)) needs a calibrated stereo pair (ZED 2i); currently demoed with mock data.

## Where things live

| Path | What |
|------|------|
| `src/dloseg/graph/dlo_graph.py` | `DLOGraph` — the central class |
| `src/dloseg/graph/pipeline.py` | `get_spline()` orchestration of the 2D pipeline |
| `src/dloseg/graph/bspline_fitting.py` | 2D spline smoothing/fitting backends |
| `src/dloseg/recon3d/` | 3D work: stereo helpers (`stereo.py`) + triangulation (`bspline_3d_recon.py`) |
| `src/dloseg/zed/` | ZED camera tooling (calibration, depth, streaming, recording) |
| `src/segmentors/` | vendored MobileSAMv2 + GroundingDINO glue |
| `scripts/` | runnable entry points ([07](07_scripts.md)) |
| `DATASETS/` | evaluation data ([06](06_datasets.md)) |

## Status / roadmap notes

- The graph pipeline is validated on the full SBHC dataset (300 images, 0 failures).
- 3D reconstruction (`bspline_3d_recon.py`) runs on synthetic data; wiring it to real ZED captures (via `outputs/seg_data_720_15fps` left/right masks) is the active work front.
- Retired approaches live in the gitignored `archive/`: a learned RGB→heatmap model (SegFormer), DER/DEFORM rod simulation, DIGIT tactile sensing.

Next: [02 — Setup](02_setup.md)
