# 02 — Setup

## Requirements

- Linux, Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) for environment management
- NVIDIA GPU + CUDA driver (only for segmentation; the graph pipeline is CPU)
- ZED SDK (only for the camera scripts)

## Install

```bash
git clone <repo-url> && cd DLOSeg
uv sync
```

`uv sync` creates `.venv/` and installs everything declared in `pyproject.toml`, including torch (CUDA build), `groundingdino-py`, and the editable `dloseg` package itself.

### ⚠️ uv sync prunes undeclared packages

`uv sync` removes anything from `.venv` that is not declared in `pyproject.toml`. Two things are installed outside of uv and **must be re-installed after every sync**:

1. **pyzed** (ZED SDK Python API — proprietary, not on PyPI):

   ```bash
   .venv/bin/python /usr/local/zed/get_python_api.py
   ```

   Requires the ZED SDK installed at `/usr/local/zed` ([stereolabs.com/developers](https://www.stereolabs.com/developers/)).

2. Nothing else — if you find yourself `pip install`-ing into the venv, add the package to `pyproject.toml` instead, or it will vanish on the next sync.

## Model checkpoints (segmentation only)

Checkpoints are gitignored. Place them at:

| File | Where to get it |
|------|-----------------|
| `src/segmentors/MobileSAMv2/weight/l2.pt` (efficientvit-L2 encoder) | [MobileSAMv2 release](https://github.com/ChaoningZhang/MobileSAM) |
| `src/segmentors/MobileSAMv2/weight/ObjectAwareModel.pt` | same |
| `src/segmentors/MobileSAMv2/PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt` | same |
| `src/segmentors/GroundingDINO/weights/groundingdino_swint_ogc.pth` | [GroundingDINO release](https://github.com/IDEA-Research/GroundingDINO) |

## Smoke test

```bash
# CPU-only, headless-safe — should print ~50 FPS with 0 errors
MPLBACKEND=Agg uv run scripts/benchmark_sbhc.py

# Interactive single-image demo (needs a display)
uv run scripts/extract_spline.py
```

If `import torch` fails with `libcudnn.so.9` errors after a partial sync, force-reinstall the CUDA wheels:

```bash
uv sync --reinstall-package torch --reinstall-package nvidia-cudnn-cu13
```

Next: [03 — Graph pipeline](03_pipeline_graph.md)
