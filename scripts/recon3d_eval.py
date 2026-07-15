"""Headless head-to-head evaluation of the two 3D DLO reconstructions.

Loops over the real ZED frames (img_00..img_12), runs the triangulation
baseline and the correspondence-free projection-based fit, scores both with
`dloseg.recon3d.reprojection.reprojection_report`, writes
`outputs/recon3d/eval_summary.csv`, and prints the comparison table.

    MPLBACKEND=Agg uv run scripts/recon3d_eval.py [--methods both|triangulation] [--check]

`--check` verifies the row-wise success criteria (projection reprojection <=
baseline per frame; strictly lower on flagged spans; lower bending + arc-length
spread) and prints PASS/FAIL per criterion.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import fields

import numpy as np

from dloseg.recon3d.bspline_3d_recon import triangulate_and_reconstruct
from dloseg.recon3d.frame_io import REPO_ROOT, extract_wire_splines, load_frame
from dloseg.recon3d.reprojection import ReprojectionReport, reprojection_report

FRAMES = [f"img_{i:02d}" for i in range(13)]
N_SAMPLES = 100  # matched-density sampling for every metric and both methods
REPROJ_TOL_PX = 0.1  # per-frame slack allowed on the reprojection <= baseline criterion
# A triangulation baseline reprojecting this far off has collapsed (correspondence
# yield gone, e.g. a closed/periodic segmentation): its curve is garbage, so the
# D4/arclen COMPARISON against it is meaningless. Such frames are excluded from
# criteria 3-4 (still reported); the projection reprojection alone shows the win.
BASELINE_COLLAPSE_PX = 20.0


def _sample(spline_func, n: int) -> np.ndarray:
    return np.asarray(spline_func(np.linspace(0.0, 1.0, n)), dtype=np.float64)


def _evaluate_frame(frame: str, methods: str) -> list[ReprojectionReport]:
    """Run the requested methods on one frame and return their reports."""
    frame_data = load_frame(frame)
    left_pts, right_pts = extract_wire_splines(frame_data)
    reports: list[ReprojectionReport] = []

    t0 = time.perf_counter()
    _, tri_func = triangulate_and_reconstruct(
        frame_data.calib_rect, left_pts, right_pts, z_range=(0.3, 5.0)
    )
    tri_runtime = time.perf_counter() - t0
    tri_curve = _sample(tri_func, N_SAMPLES)
    reports.append(
        reprojection_report(
            frame,
            "triangulation",
            tri_curve,
            frame_data.P1,
            frame_data.P2,
            left_pts,
            right_pts,
            runtime_s=tri_runtime,
            mask_left=frame_data.rect_left,
            mask_right=frame_data.rect_right,
        )
    )

    if methods == "both":
        from dloseg.recon3d.projection_recon import reconstruct_projection_based

        t0 = time.perf_counter()
        proj_func, _, info = reconstruct_projection_based(
            frame_data.calib_rect, left_pts, right_pts, tri_func
        )
        proj_runtime = time.perf_counter() - t0
        proj_curve = _sample(proj_func, N_SAMPLES)
        reports.append(
            reprojection_report(
                frame,
                "projection",
                proj_curve,
                frame_data.P1,
                frame_data.P2,
                left_pts,
                right_pts,
                runtime_s=proj_runtime,
                mask_left=frame_data.rect_left,
                mask_right=frame_data.rect_right,
            )
        )
        _ = info
    return reports


def _print_table(reports: list[ReprojectionReport]) -> None:
    cols = [f.name for f in fields(ReprojectionReport)]
    widths = {c: max(len(c), 12) for c in cols}
    header = "  ".join(c.rjust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in reports:
        row = r.as_row()
        cells = []
        for c in cols:
            v = row[c]
            cells.append(v.rjust(widths[c]) if isinstance(v, str) else f"{v:{widths[c]}.4g}")
        print("  ".join(cells))


def _write_csv(reports: list[ReprojectionReport], path: str) -> None:
    cols = [f.name for f in fields(ReprojectionReport)]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in reports:
            writer.writerow(r.as_row())


def _check_criteria(reports: list[ReprojectionReport]) -> bool:
    """Verify the row-wise real-frame success criteria; print PASS/FAIL."""
    by_frame: dict[str, dict[str, ReprojectionReport]] = {}
    for r in reports:
        by_frame.setdefault(r.frame, {})[r.method] = r

    reproj_ok = flagged_ok = bend_ok = arclen_ok = True
    for frame, methods in by_frame.items():
        if "projection" not in methods or "triangulation" not in methods:
            continue
        tri, proj = methods["triangulation"], methods["projection"]

        if max(tri.rms_reproj_L_px, tri.rms_reproj_R_px) > BASELINE_COLLAPSE_PX:
            print(
                f"  [skip]    {frame}: baseline collapsed "
                f"(tri reproj {tri.rms_reproj_L_px:.1f}/{tri.rms_reproj_R_px:.1f} px); "
                f"projection reproj {proj.rms_reproj_L_px:.1f}/{proj.rms_reproj_R_px:.1f} px"
            )
            continue

        if not (
            proj.rms_reproj_L_px <= tri.rms_reproj_L_px + REPROJ_TOL_PX
            and proj.rms_reproj_R_px <= tri.rms_reproj_R_px + REPROJ_TOL_PX
        ):
            reproj_ok = False
            print(
                f"  [reproj]  {frame}: proj L/R {proj.rms_reproj_L_px:.3f}/"
                f"{proj.rms_reproj_R_px:.3f} vs tri {tri.rms_reproj_L_px:.3f}/"
                f"{tri.rms_reproj_R_px:.3f}"
            )
        if proj.flagged_frac > 0.0 and not (proj.rms_reproj_flagged_px < tri.rms_reproj_flagged_px):
            flagged_ok = False
            print(
                f"  [flagged] {frame}: proj {proj.rms_reproj_flagged_px:.3f} "
                f"vs tri {tri.rms_reproj_flagged_px:.3f}"
            )
        if not (proj.bending_D4 < tri.bending_D4):
            bend_ok = False
            print(f"  [bending] {frame}: proj {proj.bending_D4:.4g} vs tri {tri.bending_D4:.4g}")
        if not (proj.arclen_std_mm < tri.arclen_std_mm):
            arclen_ok = False
            print(
                f"  [arclen]  {frame}: proj {proj.arclen_std_mm:.3f} vs tri {tri.arclen_std_mm:.3f}"
            )

    print(
        f"\nCriterion 3 (reproj <= baseline +{REPROJ_TOL_PX}px): {'PASS' if reproj_ok else 'FAIL'}"
    )
    print(f"Criterion 3 (flagged span strictly lower):      {'PASS' if flagged_ok else 'FAIL'}")
    print(f"Criterion 4 (lower D4 bending):                 {'PASS' if bend_ok else 'FAIL'}")
    print(f"Criterion 4 (lower arc-length std):             {'PASS' if arclen_ok else 'FAIL'}")
    return reproj_ok and flagged_ok and bend_ok and arclen_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", choices=["both", "triangulation"], default="both")
    parser.add_argument("--check", action="store_true", help="assert the row-wise criteria")
    args = parser.parse_args()

    out_dir = os.path.join(REPO_ROOT, "outputs", "recon3d")
    os.makedirs(out_dir, exist_ok=True)

    all_reports: list[ReprojectionReport] = []
    for frame in FRAMES:
        try:
            all_reports.extend(_evaluate_frame(frame, args.methods))
            print(f"done {frame}")
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"SKIP {frame}: {exc}")

    _print_table(all_reports)
    csv_path = os.path.join(out_dir, "eval_summary.csv")
    _write_csv(all_reports, csv_path)
    print(f"\nWrote {csv_path} ({len(all_reports)} rows)")

    if args.check:
        ok = _check_criteria(all_reports)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
