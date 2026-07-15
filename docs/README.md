# DLOSeg documentation

Wiki-style docs for the DLOSeg repository. Start with the overview, then jump to the page that matches your task.

| Page | Read when you want to… |
|------|------------------------|
| [01 — Overview](01_overview.md) | understand what DLOSeg does end-to-end and how the pieces connect |
| [02 — Setup](02_setup.md) | install the environment on a new machine (uv, pyzed, checkpoints) |
| [03 — Graph pipeline](03_pipeline_graph.md) | understand / tune the mask → B-spline core (`dloseg.graph`) |
| [04 — Segmentation](04_segmentation.md) | produce DLO masks from RGB with GroundingDINO + MobileSAMv2 |
| [05 — Stereo & 3D](05_stereo_3d.md) | go from left/right 2D splines to a 3D spline (ZED calibration, triangulation) |
| [06 — Datasets](06_datasets.md) | know what's in `DATASETS/` and the expected folder conventions |
| [07 — Scripts reference](07_scripts.md) | run any entry point in `scripts/` and know its knobs |

## Conventions

- Code lives in `src/dloseg/` (installable package) and `src/segmentors/` (vendored third-party + glue). Entry points live in `scripts/`.
- Every script resolves data paths from the **repo root** via a `REPO_ROOT` constant — run them from anywhere.
- Tunable parameters live in a `config` dict at the top of each script's `__main__` block; [03 — Graph pipeline](03_pipeline_graph.md) documents every key.
- `archive/` is gitignored: retired experiments stay on Rotem's machine only. If something seems missing from history, look there first.
