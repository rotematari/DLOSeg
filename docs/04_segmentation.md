# 04 — Segmentation (`src/segmentors`)

Text-prompted wire segmentation: **GroundingDINO** finds boxes matching a text prompt, **MobileSAMv2** turns each box into a pixel mask. No training needed — the prompt (default `"wire."`) is the only task specification.

## Entry point

```bash
uv run src/segmentors/full_pipe_line.py --img_path <folder-of-images>/
```

Key arguments (see `parse_args` in the script for all):

| Flag | Default | Meaning |
|------|---------|---------|
| `--img_path` | `src/segmentors/MobileSAMv2/test_images/` | folder of input images |
| `--output_dir` | `src/segmentors/outputs/` | where rendered masks are saved |
| `--encoder_type` | `efficientvit_l2` | SAM image encoder: `efficientvit_l2` (fast) / `tiny_vit` / `sam_vit_h` (accurate) |
| `--conf` / `--iou` | 0.4 / 0.9 | detection thresholds |

The text prompt and DINO thresholds (`TEXT_PROMPT`, `BOX_TRESHOLD`, `TEXT_TRESHOLD`) are constants inside `main()` — edit there to segment something other than wires.

## How it flows

1. `groundingdino.util.inference.predict` → boxes for the prompt (`cxcywh`, normalized).
2. Boxes converted to `xyxy` pixel coords → SAM box prompts.
3. MobileSAMv2 decodes masks for up to 320 boxes per batch, single forward pass.
4. Masks sorted by area, rendered with random colors, saved to `--output_dir`.

Per-stage timing is printed (DINO is the slow part; MobileSAMv2 with the efficientvit-L2 encoder is ~real-time on a modern GPU).

## Dependencies & weights

- `groundingdino-py` is declared in `pyproject.toml` (imports as `groundingdino`).
- The vendored MobileSAMv2 tree (`src/segmentors/MobileSAMv2/`) is third-party code — **do not refactor**; it is imported as `segmentors.MobileSAMv2.*` and the glue script bootstraps `sys.path` itself.
- Checkpoint locations: see [02 — Setup](02_setup.md#model-checkpoints-segmentation-only).

## Producing masks for the graph pipeline

The graph pipeline ([03](03_pipeline_graph.md)) consumes single-channel binary masks. `full_pipe_line.py` currently saves rendered visualizations; for pipeline-ready masks save `sam_mask` tensors as PNG instead (`cv2.imwrite(name, (mask*255).astype(np.uint8))`) — the stereo captures in `outputs/seg_data_720_15fps/` were produced this way.

Next: [05 — Stereo & 3D](05_stereo_3d.md)
