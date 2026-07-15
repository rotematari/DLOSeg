# 06 — Datasets

Everything lives under `DATASETS/` at the repo root. Scripts reference these paths relative to the repo root (via their `REPO_ROOT` constant), so keep the layout.

## SBHC — the primary benchmark

Synthetic wire images grouped by wire count: **S1 / S2 / S3 = 1 / 2 / 3 wires per image**, 100 images each.

```
SBHC/S{1,2,3}/
  images/        # RGB renders (imgN.jpg)
  gt_images/     # binary ground-truth masks (imgN.png) — pipeline input
  gt_labels/     # ground-truth spline points (imgN.npy)
  spline_preds/  # pipeline outputs: predicted spline points (.npy)
  plot_preds/    # pipeline outputs: overlay plots
  predicts/      # segmentation predictions
  sd_variation/  # stable-diffusion-augmented variants
```

`scripts/benchmark_sbhc.py` sweeps all of `S1–S3/gt_images` ([07](07_scripts.md)).

## Other datasets (segmentation evaluation)

| Dataset | Content | Layout |
|---------|---------|--------|
| `BWH/` | wire-harness images | `imgs/`, `labels/`, `masks/`, `masks_r101/`, `predict_labels*/` |
| `EWD/` | electrical-wire test set | `test_imgs/`, `test_labels/`, `test_masks_r50/`, `test_masks_r101/`, `test_predicts_r101/` |
| `LABD_Real/`, `LABD_Syn/` | lab DLO images, real & synthetic | `imgs/`, `json/`, `labels/`, `masks/`, `predict_*` |

The `*_r50` / `*_r101` suffixes are backbone variants of the model that produced those masks.

## Stereo captures

- `DATASETS/data_720_15fps/` — raw 720p frames from the ZED (side-by-side stereo, `img_NN.png`).
- `outputs/seg_data_720_15fps/` — per-frame left/right wire masks produced by the segmentors (`img_NN_{left,right}_mask_0.png`). These are the intended inputs for the [3D reconstruction](05_stereo_3d.md) work.

## Conventions

- `outputs/` is **gitignored** — generated artifacts only, safe to delete.
- Masks are single-channel PNG, wire = white (>0), background = 0.
- Spline predictions are `.npy` arrays of shape `(n_points, 2)` in `(x, y)` pixel coordinates (note: `.npy` is gitignored, so `spline_preds/` content stays local).

Next: [07 — Scripts reference](07_scripts.md)
