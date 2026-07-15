# 05 — Stereo & 3D reconstruction

Lift two 2D B-splines (left/right rectified views) to a single 3D B-spline. Code: `src/dloseg/recon3d/bspline_3d_recon.py` + calibration helpers in `src/dloseg/recon3d/stereo.py`.

## Prerequisites

- A **rectified** stereo pair (epipolar lines horizontal), so that corresponding points share the same y coordinate.
- Camera projection matrices `P1`, `P2` from calibration.

## ZED calibration

`stereo.get_zed_calibration(yaml_path, res)` parses the ZED factory calibration YAML (`src/dloseg/zed/calibration_data/zed_2i_cal_data.yaml`) into `K1/D1/K2/D2/R/T/P1/P2/baseline` for a chosen resolution (`'2K' | '1080p' | '720p' | 'VGA'`).

`stereo.rectify_stereo_pair(left, right, calib, alpha)` wraps `cv2.stereoRectify` + `initUndistortRectifyMap` + `remap`. Use the **returned** `P1/P2` (printed by the function) for triangulation of rectified images — not the raw calibration ones.

## Reconstruction pipeline (`triangulate_and_reconstruct`)

1. **Fit** parametric B-splines to the left and right 2D point sets (`splprep`, s=0).
2. **Correspond** by normalized **arc length** (default): both splines trace the same wire end-to-end, so u on the left maps to ~u on the right (after orientation alignment); a local root-solve then refines u to satisfy the epipolar constraint `y_right(u) == y_left`. (A global y-search mode, `matching='epipolar'`, exists but mismatches on hairpin-shaped wires.)
3. **Triangulate** matched pairs with `cv2.triangulatePoints(P1, P2, ...)` → 3D points.
4. **Fit** a 3D B-spline through the triangulated points (`splprep`, 3D).

`visualize_reconstruction` shows the 2D inputs, the raw 3D cloud, and the final 3D spline side by side.

## Running on real data

```bash
MPLBACKEND=Agg uv run scripts/recon3d_demo.py
```

End-to-end on a captured ZED frame: loads the left/right masks from `outputs/seg_data_720_15fps/`, rectifies with the factory calibration, extracts 2D splines, triangulates, and saves figures + `points3d.npy` to `outputs/recon3d/`. Verified on `img_04`: 99 points, Z ∈ [1.06, 1.16] m.

`uv run python src/dloseg/recon3d/bspline_3d_recon.py` still runs the synthetic mock demo.

## Known limitations

- Arc-length matching assumes both views see the **whole wire** (heavy occlusion of one end in one view shifts the parameterization). `z_range` filtering in `triangulate_and_reconstruct` drops residual mismatches.
- `get_zed_calibration` follows the OpenCV convention (T maps left→right frame, `t_x = -baseline`); ZED YAML Baseline/TY/TZ are in millimeters.
- Wire identity across views is assumed (spline i on the left ↔ spline i on the right); with multiple wires you must match them first (e.g. by endpoint epipolar consistency).

## Related ZED tooling (`src/dloseg/zed/`)

| Path | What |
|------|------|
| `depth/depth_to_point_cloud.py`, `depth/stereo_to_point_cloud.py` | depth-map / stereo → point cloud experiments |
| `depth/depth_from_seg.py` | depth restricted to a segmentation mask |
| `camera_streaming/` | socket sender/receiver for a remote camera feed |
| `record/svo_recording.py` | record `.svo2` clips |
| `scripts/zed_svo_export.py` | export frames from an `.svo2` recording ([07](07_scripts.md)) |

Next: [06 — Datasets](06_datasets.md)
