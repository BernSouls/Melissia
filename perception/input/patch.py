"""patch_extraction.py — image-patch region extraction.

Extracted verbatim from the legacy `_multicat_real_probe.py` scaffold so
that the vision harness (and its dependent probes) can obtain region
extraction without executing the probe's module-level monkey-patches on
`lenses.py`. Those patches (an Orientation split into arity-2 theta +
synthetic Elongation, and an anchor-class override) were found in Phase
K.0 to silently disable the entire V2+ lens tier: the split rewrites
Orientation from arity 3 to arity 2, and every higher lens reads its
input by arity, so Contour / Junction / CurveComplex / Part / ObjectShape
returned zero detections on every step. The monkey-patches remain in
`_multicat_real_probe.py` for historical reference; this module is what
the harness imports.

No behavior change vs. the source function in `_multicat_real_probe.py`.
Behavior change is only that importing this module does not execute the
probe's side effects.

Determinism: byte-identical output for the same input, same as the source.
"""

from __future__ import annotations

import math

import numpy as np

# Module-level configuration, mirrors the original probe.
# The harness sets `patch_extraction.GRID_N = grid` before calling.
GRID_N: int = 4


# -------------------------------------------------------------------------
# Region extraction (verbatim from _multicat_real_probe.py — do not modify).
# -------------------------------------------------------------------------

def _regions_from_patches(arr):
    """Fixed-grid patch extractor with a hard cap on patch count.

    Down-samples to PATCH_GRID x PATCH_GRID patches with stride
    equal to grid size (no overlap). Cap is enforced by choosing
    grid size from MAX_PATCHES, not by trimming after the fact â€”
    this preserves determinism and avoids producing a
    variable-size graph."""
    import math

    # Grid size: GRID_N x GRID_N (env-controlled). At 4 -> 16 patches,
    # at 8 -> 64 patches. Anchor classes are grid-cell ids, so the
    # signature lattice resolution tracks the grid exactly.
    G = int(GRID_N)
    if G < 2:
        G = 2

    H, W = arr.shape[:2]
    ps_h = max(1, H // G)
    ps_w = max(1, W // G)
    arr_f = arr.astype(np.float32) / 255.0
    gray = arr_f[..., :3].mean(axis=-1)

    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    grad_ang = np.arctan2(gy, gx)

    diag = math.sqrt(H * H + W * W)

    # ---- [RICH PATCH] sub-cell analysis so shape channels can vary ----
    SUBGRID = 4
    ps_h = max(1, H // G)
    ps_w = max(1, W // G)
    sub_h = max(1, ps_h // SUBGRID)
    sub_w = max(1, ps_w // SUBGRID)

    # One scene-level mean gradient magnitude, computed once per image.
    scene_mean_mag = float(grad_mag.mean())

    regions = []
    for gy_i in range(G):
        for gx_i in range(G):
            y0 = gy_i * ps_h
            x0 = gx_i * ps_w
            y1 = min(H, y0 + ps_h)
            x1 = min(W, x0 + ps_w)
            prgb = arr_f[y0:y1, x0:x1, :3]
            pmag = grad_mag[y0:y1, x0:x1]
            pgy = gy[y0:y1, x0:x1]
            pgx = gx[y0:y1, x0:x1]

            mean_rgb = tuple(float(v) for v in
                             prgb.reshape(-1, 3).mean(axis=0))

            # Sub-cell gradient statistics (vectorized).
            sub_mag = np.zeros((SUBGRID, SUBGRID), dtype=np.float64)
            sub_ang = np.zeros((SUBGRID, SUBGRID), dtype=np.float64)
            for sy in range(SUBGRID):
                ay0 = sy * sub_h
                ay1 = min(y1 - y0, ay0 + sub_h)
                for sx in range(SUBGRID):
                    ax0 = sx * sub_w
                    ax1 = min(x1 - x0, ax0 + sub_w)
                    if ay1 <= ay0 or ax1 <= ax0:
                        continue
                    cell = pmag[ay0:ay1, ax0:ax1]
                    sub_mag[sy, sx] = float(cell.mean())
                    sub_ang[sy, sx] = math.atan2(
                        float(pgy[ay0:ay1, ax0:ax1].sum()),
                        float(pgx[ay0:ay1, ax0:ax1].sum()),
                    )

            edge_frac = float(
                (sub_mag > scene_mean_mag * 1.5).sum() / sub_mag.size
            )

            w_sum = float(sub_mag.sum())
            if w_sum < 1e-6:
                C = 0.0
                S = 0.0
            else:
                C = float((sub_mag * np.cos(2.0 * sub_ang)).sum() / w_sum)
                S = float((sub_mag * np.sin(2.0 * sub_ang)).sum() / w_sum)
            anisotropy = math.sqrt(C * C + S * S)

            def _clamp(v, lo, hi):
                return max(lo, min(hi, v))

            circularity = _clamp(1.0 - edge_frac, 0.0, 1.0)
            complexity = _clamp(edge_frac, 0.0, 1.0)
            elongation = _clamp(1.0 - anisotropy, 0.1, 1.0)
            # F.0.2: PER-PATCH roughness.
            # Was `_clamp(scene_mean_mag * 2.0, ...)`, where scene_mean_mag
            # is a scene-level scalar hoisted out of this loop and copied
            # verbatim into every region -- so all 64 patches shared one
            # identical float. `sub_mag` is this patch's own SUBGRID x
            # SUBGRID array of sub-cell mean gradient magnitudes, so its
            # mean is a genuine per-patch measurement.
            # `scene_mean_mag` is still used above, correctly, as the
            # scene-relative threshold for edge_frac.
            roughness = _clamp(float(sub_mag.mean()) * 2.0, 0.0, 1.0)
            isotropy = _clamp(1.0 - anisotropy, 0.0, 1.0)
            major_ang = (0.5 * math.atan2(S, C)) % math.pi

            cx = ((x0 + x1) / 2.0) / max(W, 1)
            cy = ((y0 + y1) / 2.0) / max(H, 1)
            pr = 0.5 * math.sqrt(ps_h * ps_h + ps_w * ps_w)

            regions.append({
                "centroid": (cx * W, cy * H),
                "radius_px": pr,
                "radius_norm": pr / (0.35 * diag + 1e-6),
                "rel_position": (2.0 * cx - 1.0, 2.0 * cy - 1.0),
                "rel_x": cx,
                "rel_y": cy,
                "area": ps_h * ps_w,
                "mean_rgb": mean_rgb,
                "elongation": elongation,
                "major_axis_angle": major_ang,
                "circularity": circularity,
                "complexity": complexity,
                "roughness": roughness,
                "isotropy": isotropy,
                "boundary_polyline": [],
                # Grid coordinates: consumed by the probe's
                # _anchor_class_bin monkey-patch so the anchor class is
                # the exact cell id, independent of image aspect ratio.
                "grid_row": int(gy_i),
                "grid_col": int(gx_i),
            })
    assert len(regions) == GRID_N * GRID_N, len(regions)
    return regions
