# 07 — Scripts reference

All entry points live in `scripts/`. Each resolves data paths from the repo root (`REPO_ROOT` constant), so they run from any working directory. Parameters live in the `config` dict inside each script's `__main__` block; graph-pipeline keys are documented in [03 — Graph pipeline](03_pipeline_graph.md#config-reference).

## `extract_spline.py`

Single-mask visual demo of the graph pipeline.

```bash
uv run scripts/extract_spline.py
```

- Input: `config['mask_l_path']` (a binary mask, e.g. from `outputs/seg_data_720_15fps/`).
- Shows each pipeline stage if the `show_*` flags are on, then plots the final B-splines next to the real RGB image.
- Needs a display; use `MPLBACKEND=Agg` to run headless (plots are suppressed).

## `benchmark_sbhc.py`

Batch run over the full SBHC dataset (S1–S3, 300 masks).

```bash
MPLBACKEND=Agg uv run scripts/benchmark_sbhc.py
```

- Prints average per-image time / FPS and lists failed images.
- A commented block inside saves per-wire spline `.npy` predictions to `SBHC/*/spline_preds/` and overlay plots to `plot_preds/` — enable when regenerating evaluation data.

## `extract_spline_video.py`

Frame-by-frame live spline extraction on a video.

```bash
uv run scripts/extract_spline_video.py   # set config['video_path'] first
```

- Thresholds each frame (gray > 80) into a mask → pipeline → draws splines in an OpenCV window. Press `q` to quit.

## `video_to_frames.py`

Dump every frame of a video to `frame_NNNNNN.png` files. Edit the paths in `__main__`.

## `zed_svo_export.py`

Export frames from a recorded ZED `.svo2` file (left/right or left/depth depending on `mode`). Edit `svo_input_path` / `output_dir` at the top. Requires **pyzed** ([02 — Setup](02_setup.md#-uv-sync-prunes-undeclared-packages)).

## Segmentation entry point (not in `scripts/`)

`src/segmentors/full_pipe_line.py` — text-prompted segmentation over a folder of images; see [04 — Segmentation](04_segmentation.md). It stays next to the vendored code it glues together.

## Adding a new script

1. Put it in `scripts/`, with a module docstring saying what it does and how to run it.
2. Import library code from `dloseg.*` — never duplicate pipeline logic into the script.
3. Anchor file paths with the `REPO_ROOT` pattern used by the existing scripts.
