"""perception/contour_structure.py -- contour tracing, structure and lattice
symbols moved VERBATIM from perception/contours.py (ITEM 4 deletion prep,
MOVE ONLY: nothing deleted). perception/contours.py keeps a thin re-export
shim so every existing import path keeps working. Importers per
diagnostics/_item4_symbol_audit.md.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from scipy.ndimage import maximum_filter

import parameters as P
from core.hashing import (  # noqa: F401
    clamp,
)
from perception.polar import (  # noqa: F401
    _angle_diff,
    _axial_angle_diff,
    _axial_wrap_sf,
    signed_menger,
)


def _local_edge_at(lum, cy, cx):
    """(contrast, polarity, sharpness, orientation) at one pixel."""
    h, w = lum.shape

    def at(y, x):
        return lum[max(0, min(h - 1, y)), max(0, min(w - 1, x))]

    gx = (at(cy, cx + 1) - at(cy, cx - 1)) * 0.5
    gy = (at(cy + 1, cx) - at(cy - 1, cx)) * 0.5
    contrast = math.sqrt(gx * gx + gy * gy)
    gxx = at(cy, cx + 1) - 2 * at(cy, cx) + at(cy, cx - 1)
    gyy = at(cy + 1, cx) - 2 * at(cy, cx) + at(cy - 1, cx)
    lap = gxx + gyy
    polarity = 1.0 if lap < 0 else -1.0
    sharpness = min(1.0, abs(lap) / (contrast + 1e-6))
    orient = math.atan2(gy, gx) % math.pi
    return (contrast, polarity, sharpness, orient)

def _polyline_arc_length(polyline: List[Tuple[int, int]]) -> float:
    """Sum of consecutive Euclidean step lengths. Diagonals cost √2."""
    if len(polyline) < 2:
        return 0.0
    total = 0.0
    for i in range(len(polyline) - 1):
        dr = polyline[i + 1][0] - polyline[i][0]
        dc = polyline[i + 1][1] - polyline[i][1]
        total += math.sqrt(dr * dr + dc * dc)
    return total

def _tangent_at(
    polyline: List[Tuple[int, int]], at_start: bool, span: int = 3
) -> float:
    """Local tangent (atan2(dr, dc)) at one end of an ordered polyline,
    using up to `span` pixels of local support."""
    n = len(polyline)
    if n < 2:
        return 0.0
    span = max(1, min(span, n - 1))
    if at_start:
        p1, p2 = polyline[0], polyline[span]
    else:
        p1, p2 = polyline[n - 1 - span], polyline[n - 1]
    return math.atan2(p2[0] - p1[0], p2[1] - p1[1])

def _moore_neighbor_trace(
    coords: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """[v9/v10] Moore-Neighbor boundary tracing.

    Returns the OUTER contour as an ordered list of (r, c) pixels,
    traversed clockwise from the topmost-leftmost boundary pixel. Uses
    a simplified stopping criterion (halts upon re-entering the start
    pixel). For non-simple boundaries, the trace may halt early, but
    the closure check (trace[0] == trace[-1]) safely rejects them as [].

    CONVENTION: prev_dir_idx is the DIRECTION OF MOTION (set to
    found_idx after each move), not the backtrack direction. Scan start
    offset is +5 (backtrack +4, one clockwise step past it).

    Returns [] on any failure (sentinel — callers treat as "not traced"):
      - region area < 3;
      - fewer than 3 boundary pixels;
      - start pixel is isolated;
      - trace length < 3;
      - trace length < len(boundary) // 3;
      - trace does not close.
    """
    if len(coords) < 3:
        return []
    coord_set = set(coords)

    def _is_boundary(r: int, c: int) -> bool:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) not in coord_set:
                return True
        return False

    boundary = [(r, c) for (r, c) in coords if _is_boundary(r, c)]
    if len(boundary) < 3:
        return []

    start = min(boundary, key=lambda p: (p[0], p[1]))
    bnd_set = set(boundary)
    assert start in bnd_set, "start pixel must be in bnd_set"

    dirs = [
        (0, -1), (-1, -1), (-1, 0), (-1, 1),
        (0, 1), (1, 1), (1, 0), (1, -1),
    ]

    prev_dir_idx = 0
    found = False
    for k in range(8):
        di = (prev_dir_idx + k) % 8
        dr, dc = dirs[di]
        if (start[0] + dr, start[1] + dc) in bnd_set:
            prev_dir_idx = di
            found = True
            break
    if not found:
        return []

    trace = [start]
    trace_set: Set[Tuple[int, int]] = {start}
    curr = start
    max_steps = 8 * len(boundary) + 8
    for _ in range(max_steps):
        search_start = (prev_dir_idx + 5) % 8
        found_idx = None
        for k in range(8):
            di = (search_start + k) % 8
            dr, dc = dirs[di]
            nxt = (curr[0] + dr, curr[1] + dc)
            if nxt in bnd_set:
                found_idx = di
                break
        if found_idx is None:
            break
        dr, dc = dirs[found_idx]
        nxt = (curr[0] + dr, curr[1] + dc)
        if nxt == start and len(trace) >= 3:
            trace.append(start)
            break
        if nxt in trace_set:
            trace.append(nxt)
            break
        trace.append(nxt)
        trace_set.add(nxt)
        curr = nxt
        prev_dir_idx = found_idx

    if len(trace) < 3:
        return []
    if len(trace) < len(boundary) // 3:
        return []
    if trace[0] != trace[-1]:
        return []
    return trace

def _detect_t_junctions(
    regions: List[Dict[str, Any]],
) -> List[Tuple[int, int, Tuple[float, float], float]]:
    """[v9] Geometric T-junction detection. Returns
    [(stem_region_idx, top_region_idx, contact_point, confidence)]."""
    tjs: List[Tuple[int, int, Tuple[float, float], float]] = []
    for i, r_stem in enumerate(regions):
        poly_stem = r_stem.get("boundary_polyline", [])
        if not poly_stem or len(poly_stem) < 2:
            continue
        for is_start in (True, False):
            pt = poly_stem[0] if is_start else poly_stem[-1]
            tan_stem_in = _tangent_at(poly_stem, at_start=is_start)
            tan_stem_out = (tan_stem_in + math.pi) % (2.0 * math.pi)
            for j, r_top in enumerate(regions):
                if i == j:
                    continue
                poly_top = r_top.get("boundary_polyline", [])
                if not poly_top or len(poly_top) < 2:
                    continue
                top_arc = _polyline_arc_length(poly_top)
                if top_arc < P.T_JUNCTION_MIN_FOREIGN_ARC:
                    continue
                best_d = float("inf")
                best_idx = -1
                for k, q in enumerate(poly_top):
                    dd = math.sqrt(
                        (pt[0] - q[0]) ** 2 + (pt[1] - q[1]) ** 2
                    )
                    if dd < best_d:
                        best_d = dd
                        best_idx = k
                if best_d > P.T_JUNCTION_MAX_GAP:
                    continue
                lo = max(0, best_idx - 3)
                hi = min(len(poly_top) - 1, best_idx + 3)
                if hi - lo < 1:
                    continue
                p_lo, p_hi = poly_top[lo], poly_top[hi]
                tan_top = math.atan2(
                    p_hi[0] - p_lo[0], p_hi[1] - p_lo[1]
                )
                ang = abs(_angle_diff(tan_stem_out, tan_top))
                ang = min(ang, math.pi - ang)
                perp_dev = abs(ang - math.pi / 2.0)
                if perp_dev > P.T_JUNCTION_MAX_PERP_DEV:
                    continue
                gap_conf = 1.0 - best_d / P.T_JUNCTION_MAX_GAP
                perp_conf = 1.0 - perp_dev / P.T_JUNCTION_MAX_PERP_DEV
                len_conf = min(
                    1.0, top_arc / (2.0 * P.T_JUNCTION_MIN_FOREIGN_ARC)
                )
                confidence = gap_conf * perp_conf * len_conf
                if confidence < P.OCCLUSION_MIN_CONFIDENCE:
                    continue
                contact_pt = (
                    (pt[0] + poly_top[best_idx][0]) / 2.0,
                    (pt[1] + poly_top[best_idx][1]) / 2.0,
                )
                tjs.append((i, j, contact_pt, confidence))
    return tjs

def _front_end_scale(h: int, w: int) -> float:
    """Scale of this frame relative to the calibrated reference resolution.

    Pixel-denominated front-end thresholds are multiplied by this factor and
    the Gabor cycles-per-pixel band is divided by it, so the same physical
    tuning holds at any resolution. 1.0 at the 256x256 reference frame, so
    behaviour there is unchanged. See the note on
    P.FRONT_END_REF_DIAGONAL_PX for the A/B/C experiment that motivated this.
    """
    ref = float(getattr(P, "FRONT_END_REF_DIAGONAL_PX", 362.0386719678744))
    return math.hypot(float(h), float(w)) / max(ref, 1e-6)

def _extract_association_field_contours(
    complex_energy: np.ndarray,
    bank_theta: np.ndarray,
    bank_sf: np.ndarray,
    h: int,
    w: int,
) -> List[Dict[str, Any]]:
    """Extract continuous curvilinear contours from oriented energy via association fields.

    Field, Hayes & Hess (1993): Collinear and co-circular integration along
    the edge tangent ridge (orthogonal to Gabor carrier orientation).

    [Scale-relative] Pixel-denominated constants (sigma_dist, min_length,
    max_seeds, closure distance) are scaled by _front_end_scale(h, w); the
    normalised/angular ones are resolution-independent and left as-is.
    """
    scale = _front_end_scale(h, w)
    min_len = getattr(P, "CONTOUR_MIN_LENGTH", 12) * scale
    s_theta = getattr(P, "CONTOUR_SIGMA_THETA", 0.45)
    s_dist = getattr(P, "CONTOUR_SIGMA_DIST", 3.5) * scale
    p_cos = getattr(P, "CONTOUR_COS_POWER", 2.0)
    s_thresh = getattr(P, "CONTOUR_STRENGTH_THRESHOLD", 0.12)
    max_steps = getattr(P, "CONTOUR_MAX_TRACING_STEPS", 350)
    max_seeds = max(1, int(round(getattr(P, "CONTOUR_MAX_SEEDS", 180) * scale)))

    nt = len(bank_theta)

    # 1. Energy pooling and dominant carrier orientation
    energy_over_sf = complex_energy.max(axis=1)  # (nt, H, W)
    strength_map = energy_over_sf.max(axis=0).astype(np.float64)  # (H, W)
    carrier_idx = np.argmax(energy_over_sf, axis=0)  # (H, W)

    # [Front-end fix 2026-10-10] Global-max normalisation let the frame-edge
    # convolution artifacts (the reflect-padded lowest-SF kernel manufactures
    # ridges at ~3*sigma_x inside the border) set the reference, so a small
    # object's edge energy was diluted below CONTOUR_STRENGTH_THRESHOLD and it
    # vanished entirely -- measured on the synthetic zoom sweep: a disc at
    # r <= 0.20 of a 256^2 frame yielded ZERO contours. Normalise against a
    # high percentile of the INTERIOR, with the border margin taken from the
    # lowest-SF kernel's support (scale-relative via _front_end_scale).
    sf_min = float(np.min(bank_sf)) if len(bank_sf) else P.GABOR_SF_MIN
    # bank_sf is already scale-adjusted by GaborBank.build_default(scale), so
    # this is the EFFECTIVE kernel sigma in pixels at this resolution.
    sigma_min = P.GABOR_SIGMA_X_FRAC / max(sf_min, 1e-6)
    margin = int(max(1, math.ceil(3.0 * sigma_min)))
    margin = min(margin, max(1, (min(h, w) - 1) // 2))
    interior = strength_map[margin:h - margin, margin:w - margin]
    if interior.size == 0:
        interior = strength_map
    s_ref = float(np.percentile(interior, P.CONTOUR_INTERIOR_PERCENTILE))
    if s_ref < 1e-9:
        s_ref = float(strength_map.max())
    # [front-end fix, gate 1] Absolute floor. A percentile reference has no
    # notion of "is there structure at all": on a uniform field it normalises
    # numerical noise into ridges. Measured on a blank 256^2 frame the interior
    # p99.5 is 6.9e-07 while the hardest real small object is 2.75e-01, so the
    # floor separates the two by ~5 orders of magnitude.
    if s_ref < P.CONTOUR_MIN_ABS_STRENGTH:
        return []
    strength_norm = strength_map / s_ref

    # 2. Physics Correction: Gabor carrier is normal; tangent is theta_carrier + pi/2
    carrier_rad = carrier_idx.astype(np.float64) * (math.pi / float(nt))
    tangent_rad = (carrier_rad + (math.pi / 2.0)) % math.pi

    # 3. Dense Non-Maximum Suppression with bounded seed budget
    local_max = maximum_filter(strength_norm, size=3)
    ridge_mask = (strength_norm == local_max) & (strength_norm >= s_thresh)

    ys, xs = np.where(ridge_mask)
    if len(xs) == 0:
        return []

    ridge_seeds = [(int(x), int(y), float(strength_norm[y, x])) for x, y in zip(xs, ys)]
    ridge_seeds.sort(key=lambda s: -s[2])
    ridge_seeds = ridge_seeds[:max_seeds]

    used = np.zeros((h, w), dtype=bool)
    contours: List[Dict[str, Any]] = []

    neighbor_offsets = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]

    def _trace_branch(start_x: int, start_y: int, heading_angle: float) -> Tuple[List[Tuple[int, int]], List[float], List[float]]:
        path: List[Tuple[int, int]] = []
        thetas: List[float] = []
        strengths: List[float] = []

        cx, cy = start_x, start_y
        curr_heading = heading_angle

        for _ in range(max_steps):
            best_next = None
            best_w = 0.0
            best_heading = curr_heading

            for dx, dy in neighbor_offsets:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < w and 0 <= ny < h) or used[ny, nx]:
                    continue
                if strength_norm[ny, nx] < (s_thresh * 0.4):
                    continue

                step_dist = math.hypot(dx, dy)
                step_dir = math.atan2(dy, dx)

                # Forward cone check: must step in forward hemisphere
                d_head = abs((step_dir - curr_heading + math.pi) % (2.0 * math.pi) - math.pi)
                if d_head > (math.pi / 2.0):
                    continue

                # Tangent agreement (axial mod pi)
                n_tangent = tangent_rad[ny, nx]
                d_theta = _axial_angle_diff(n_tangent, curr_heading)

                # Association field equation
                w_assoc = (
                    math.exp(-(d_theta ** 2) / (s_theta ** 2))
                    * math.exp(-(step_dist ** 2) / (s_dist ** 2))
                    * (max(0.0, math.cos(d_head)) ** p_cos)
                    * float(strength_norm[ny, nx])
                )

                if w_assoc > best_w:
                    best_w = w_assoc
                    best_next = (nx, ny)
                    best_heading = step_dir

            if best_next is None or best_w < 0.05:
                break

            nx, ny = best_next
            path.append((nx, ny))
            thetas.append(float(tangent_rad[ny, nx]))
            strengths.append(float(strength_norm[ny, nx]))
            used[ny, nx] = True

            cx, cy = nx, ny
            curr_heading = best_heading

        return path, thetas, strengths

    for sx, sy, _ in ridge_seeds:
        if used[sy, sx]:
            continue

        used[sy, sx] = True
        tan_0 = tangent_rad[sy, sx]

        # Trace both branches starting from seed
        fwd_pts, fwd_th, fwd_st = _trace_branch(sx, sy, tan_0)
        bwd_pts, bwd_th, bwd_st = _trace_branch(sx, sy, (tan_0 + math.pi) % (2.0 * math.pi))

        full_pts = list(reversed(bwd_pts)) + [(sx, sy)] + fwd_pts
        full_th = list(reversed(bwd_th)) + [float(tan_0)] + fwd_th
        full_st = list(reversed(bwd_st)) + [float(strength_norm[sy, sx])] + fwd_st

        if len(full_pts) < min_len:
            continue

        d_close = math.hypot(full_pts[0][0] - full_pts[-1][0], full_pts[0][1] - full_pts[-1][1])
        is_closed = (d_close <= 3.0 * scale and len(full_pts) >= 16)
        if is_closed:
            full_pts.append(full_pts[0])
            full_th.append(full_th[0])
            full_st.append(full_st[0])

        arc_len = sum(
            math.hypot(full_pts[i+1][0] - full_pts[i][0], full_pts[i+1][1] - full_pts[i][1])
            for i in range(len(full_pts) - 1)
        )

        contours.append({
            "polyline": full_pts,
            "theta_profile": full_th,
            "strength_profile": full_st,
            "length": float(arc_len),
            "is_closed": bool(is_closed),
        })

    return contours

class V1RetinotopicLattice:
    """Tessellates the visual array into an overlapping retinotopic coordinate sheet
    of cortical hypercolumns (Hubel & Wiesel 1968; Mountcastle 1997).
    """

    def __init__(self, density: int = None, overlap_factor: float = None):
        self.density = density or getattr(P, "RETINOTOPIC_GRID_DENSITY", 16)
        self.overlap_factor = overlap_factor or getattr(P, "RETINOTOPIC_RF_OVERLAP", 1.6)

    def tile_visual_field(self, h: int, w: int) -> List[Dict[str, Any]]:
        stride_y = h / float(self.density)
        stride_x = w / float(self.density)
        radius_px = max(2.0, (min(stride_y, stride_x) / 2.0) * self.overlap_factor)
        diag_ref = math.sqrt(h * h + w * w)

        hypercolumns: List[Dict[str, Any]] = []
        for r_idx in range(self.density):
            cy = (r_idx + 0.5) * stride_y
            for c_idx in range(self.density):
                cx = (c_idx + 0.5) * stride_x
                rel_x = (cx / w) * 2.0 - 1.0
                rel_y = (cy / h) * 2.0 - 1.0

                hypercolumns.append({
                    "anchor_id": len(hypercolumns),
                    "centroid": (float(cx), float(cy)),
                    "grid_coord": (r_idx, c_idx),
                    "radius_px": float(radius_px),
                    "radius_norm": float(radius_px / (0.35 * diag_ref + 1e-6)),
                    "rel_position": (float(rel_x), float(rel_y)),
                    "rel_x": float((rel_x + 1.0) / 2.0),
                    "rel_y": float((rel_y + 1.0) / 2.0),
                    "area": int(math.pi * radius_px * radius_px),
                    "circularity": 1.0,
                    "complexity": 0.1,
                    "elongation": 1.0,
                    "major_axis_angle": 0.0,
                    "roughness": 0.0,
                    "isotropy": 1.0,
                    "boundary_polyline": None,
                    "is_boundary": False,
                })
        return hypercolumns

    def tile_with_features(
        self,
        h: int,
        w: int,
        complex_energy: np.ndarray,
        bank_theta: np.ndarray,
        bank_sf: np.ndarray,
        lum: np.ndarray,
        clean_rgb: np.ndarray,
    ) -> List[Dict[str, Any]]:
        regions = self.tile_visual_field(h, w)
        nt = len(bank_theta)

        for r in regions:
            cx = max(0, min(w - 1, int(round(r["centroid"][0]))))
            cy = max(0, min(h - 1, int(round(r["centroid"][1]))))
            rad = max(2, int(r["radius_px"]))
            y0, y1 = max(0, cy - rad), min(h, cy + rad + 1)
            x0, x1 = max(0, cx - rad), min(w, cx + rad + 1)

            win_energy = complex_energy[:, :, y0:y1, x0:x1]
            if win_energy.size > 0:
                marginal_th = win_energy.sum(axis=(1, 2, 3))
                tot_e = float(marginal_th.sum())
                if tot_e > 1e-9:
                    p_th = marginal_th / tot_e
                    c_m = float(np.sum(p_th * np.cos(2 * bank_theta)))
                    s_m = float(np.sum(p_th * np.sin(2 * bank_theta)))
                    r_len = math.hypot(c_m, s_m)
                    r["elongation"] = clamp(1.0 - r_len, 0.0, 1.0)
                    r["major_axis_angle"] = float(0.5 * math.atan2(s_m, c_m)) % math.pi
                    r["complexity"] = clamp(r_len, 0.0, 1.0)
                else:
                    r["elongation"] = 1.0
                    r["major_axis_angle"] = 0.0
                    r["complexity"] = 0.0

                win_spatial = win_energy.sum(axis=(0, 1))
                tot_sp = float(win_spatial.sum())
                if tot_sp > 1e-9:
                    yy, xx = np.mgrid[0:win_spatial.shape[0], 0:win_spatial.shape[1]]
                    w_n = win_spatial / tot_sp
                    my = float((w_n * yy).sum())
                    mx = float((w_n * xx).sum())
                    dists = np.hypot(yy - my, xx - mx)
                    max_d = float(dists.max()) if dists.size > 0 else 1.0
                    r["circularity"] = clamp(float((w_n * dists).sum()) / (max_d + 1e-9), 0.0, 1.0)
                else:
                    r["circularity"] = 0.5
            else:
                r["elongation"] = 1.0
                r["major_axis_angle"] = 0.0
                r["circularity"] = 0.5
                r["complexity"] = 0.0

            lum_win = lum[y0:y1, x0:x1]
            if lum_win.size > 4:
                r["roughness"] = clamp(float(np.std(lum_win)) * 2.0, 0.0, 1.0)
            else:
                r["roughness"] = 0.0
            r["isotropy"] = clamp(r["elongation"], 0.0, 1.0)

            # True Linear RGB for Color lens compatibility
            pix_rgb = clean_rgb[y0:y1, x0:x1]
            if pix_rgb.size >= 3:
                r["mean_rgb"] = tuple(float(x) for x in np.mean(pix_rgb, axis=(0, 1))[:3])
            else:
                r["mean_rgb"] = (float(clean_rgb[cy, cx, 0]), float(clean_rgb[cy, cx, 1]), float(clean_rgb[cy, cx, 2]))

        return regions

def _resample_contours(contours, step, min_points=12):
    pts = []  # (x, y, theta_axial, contour_idx, arc)
    for ci, c in enumerate(contours):
        pl = c.get("polyline")
        if not pl or len(pl) < min_points:
            continue
        P_ = [(float(x), float(y)) for x, y in pl]
        cum = [0.0]
        for i in range(1, len(P_)):
            cum.append(cum[-1] + math.hypot(P_[i][0] - P_[i-1][0], P_[i][1] - P_[i-1][1]))
        if cum[-1] < 4 * step:
            continue
        smp, j = [], 0
        for k in range(int(cum[-1] / step) + 1):
            s = min(cum[-1], k * step)
            while j < len(cum) - 2 and cum[j+1] < s:
                j += 1
            seg = cum[j+1] - cum[j]
            f = 0.0 if seg < 1e-9 else (s - cum[j]) / seg
            smp.append((P_[j][0] + f * (P_[j+1][0] - P_[j][0]), P_[j][1] + f * (P_[j+1][1] - P_[j][1]), s))
        for k, (x, y, s) in enumerate(smp):
            a, b = smp[max(0, k-2)], smp[min(len(smp)-1, k+2)]
            pts.append((x, y, math.atan2(b[1] - a[1], b[0] - a[0]) % math.pi, ci, s))
    return pts

def _segment_profile(r, tol):
    """Split a radius profile at significant interior extrema (hysteresis tol).
    Returns (cut_indices, kinds) with kinds 'min' (neck) or 'max' (bulge/seed)."""
    cuts, kinds, ext_v, ext_i, mode = [], [], r[0], 0, 0
    for i in range(1, len(r)):
        v = r[i]
        if mode == 0:
            if v - ext_v > tol:
                mode, ext_v, ext_i = 1, v, i
            elif ext_v - v > tol:
                mode, ext_v, ext_i = -1, v, i
        elif mode == 1:
            if v > ext_v:
                ext_v, ext_i = v, i
            elif ext_v - v > tol:
                cuts.append(ext_i)
                kinds.append("max")
                mode, ext_v, ext_i = -1, v, i
        else:
            if v < ext_v:
                ext_v, ext_i = v, i
            elif v - ext_v > tol:
                cuts.append(ext_i)
                kinds.append("min")
                mode, ext_v, ext_i = 1, v, i
    return cuts, kinds

def extract_shock_structure(
    contours,
    step_frac=0.03,
    r_min_frac=0.03,
    r_max_frac=1.05,
    tol_sym=math.pi / 12,
    tol_taper=math.pi / 6,
    min_run=4,
    prune_aspect=0.5,
    join_frac=1.2,
):
    """Extract pruned shock graph branches from symmetric boundary ribbons."""
    allp = [(float(x), float(y)) for c in contours for x, y in (c.get("polyline") or [])]
    if len(allp) < 24:
        return [], [], [], {}
    cx = sum(p[0] for p in allp) / len(allp)
    cy = sum(p[1] for p in allp) / len(allp)
    ds = sorted(math.hypot(p[0] - cx, p[1] - cy) for p in allp)
    R = ds[int(0.9 * (len(ds) - 1))]
    if R < 1e-6:
        return [], [], [], {}
    step = max(1.5, step_frac * R)
    r_min = r_min_frac * R
    r_max = r_max_frac * R

    pts = _resample_contours(contours, step)
    N = len(pts)
    if N < 2 * min_run:
        return [], [], [], {}
    cell = max(R, 1.0)
    grid = defaultdict(list)
    for i, p in enumerate(pts):
        grid[(int(p[0] // cell), int(p[1] // cell))].append(i)

    def near(x, y, rad):
        gx, gy, cr, out = int(x // cell), int(y // cell), int(rad // cell) + 1, []
        for dx in range(-cr, cr + 1):
            for dy in range(-cr, cr + 1):
                for k in grid.get((gx + dx, gy + dy), ()):
                    if math.hypot(pts[k][0] - x, pts[k][1] - y) <= rad:
                        out.append(k)
        return out

    pairs = {}
    for i, (xi, yi, ti, ci, si) in enumerate(pts):
        for j in sorted(near(xi, yi, 2.0 * r_max)):
            if j <= i:
                continue
            xj, yj, tj, cj, sj = pts[j]
            d = math.hypot(xj - xi, yj - yi)
            if d < 2.0 * r_min or (ci == cj and abs(si - sj) < 1.25 * d):
                continue
            g = math.atan2(yj - yi, xj - xi)
            ai, aj = _axial_wrap_sf(ti - g), _axial_wrap_sf(tj - g)
            if min(abs(ai), abs(aj)) < math.pi / 2 - tol_taper or abs(_axial_wrap_sf(ai + aj)) > tol_sym:
                continue
            mx, my, r = 0.5 * (xi + xj), 0.5 * (yi + yj), 0.5 * d
            blocked = False
            for k in near(mx, my, 0.8 * r):  # Blum maximal disk criterion
                if k in (i, j):
                    continue
                if (pts[k][3] == ci and abs(pts[k][4] - si) <= 2.5 * step) or \
                   (pts[k][3] == cj and abs(pts[k][4] - sj) <= 2.5 * step):
                    continue
                blocked = True
                break
            if not blocked:
                pairs[(i, j)] = (mx, my, r)
    parent = {k: k for k in pairs}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (i, j) in sorted(pairs):
        for di in (-2, -1, 0, 1, 2):
            for dj in (-2, -1, 0, 1, 2):
                a, b = i + di, j + dj
                if (di == 0 and dj == 0) or not (0 <= a < N and 0 <= b < N) or a == b:
                    continue
                if pts[a][3] != pts[i][3] or pts[b][3] != pts[j][3]:
                    continue
                key = (a, b) if a < b else (b, a)
                if key in pairs:
                    ra, rb = find((i, j)), find(key)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)

    comps = defaultdict(list)
    for k in sorted(pairs):
        comps[find(k)].append(k)

    segs = []
    for root in sorted(comps):
        ks = comps[root]
        if len(ks) < min_run:
            continue
        M = [pairs[k] for k in ks]
        ax_ = sum(m[0] for m in M) / len(M)
        ay_ = sum(m[1] for m in M) / len(M)
        sxx = sum((m[0] - ax_) ** 2 for m in M)
        syy = sum((m[1] - ay_) ** 2 for m in M)
        sxy = sum((m[0] - ax_) * (m[1] - ay_) for m in M)
        ang = 0.5 * math.atan2(2 * sxy, sxx - syy)
        ux, uy = math.cos(ang), math.sin(ang)
        slots = defaultdict(list)
        for t_, m in enumerate(M):
            slots[int(round(((m[0] - ax_) * ux + (m[1] - ay_) * uy) / step))].append(t_)
        axis = [
            (
                sum(M[t_][0] for t_ in v) / len(v),
                sum(M[t_][1] for t_ in v) / len(v),
                sum(M[t_][2] for t_ in v) / len(v),
            )
            for _, v in sorted(slots.items())
        ]
        sup = sorted({pts[k[0]][3] for k in ks} | {pts[k[1]][3] for k in ks})
        rs = [a[2] for a in axis]
        mean_r = sum(rs) / len(rs)
        tol_r = max(0.15 * mean_r, 0.5 * step)
        cuts, kinds = _segment_profile(rs, tol_r) if len(axis) >= 3 else ([], [])
        valid_cuts = [c for c in cuts if 0 < c < len(axis) - 1]
        bounds = [0] + sorted(set(valid_cuts)) + [len(axis) - 1]
        for si_, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
            seg_ax = axis[lo:hi + 1]
            if len(seg_ax) < 1:
                continue
            L = sum(
                math.hypot(seg_ax[q+1][0] - seg_ax[q][0], seg_ax[q+1][1] - seg_ax[q][1])
                for q in range(len(seg_ax) - 1)
            )
            sr = [a[2] for a in seg_ax]
            mr = sum(sr) / len(sr)
            if L < 1.0 * mr:
                order = 4  # Fourth-order: radial seed/extremum
            elif max(sr) - min(sr) <= tol_r:
                order = 3  # Third-order: constant width
            else:
                order = 1  # First-order: monotone tapering
            if order != 4 and (L < prune_aspect * 2 * mr or mr < r_min):
                continue  # Prune stubs
            if sr[0] < sr[-1]:
                seg_ax = seg_ax[::-1]
                sr = sr[::-1]  # Canonical thick-to-thin traversal
            taper = (sr[-1] - sr[0]) / max(L, 1e-6)

            # Translation-invariant signed curvature using object centroid (cx, cy)
            kap = (
                signed_menger(seg_ax[0][:2], seg_ax[len(seg_ax) // 2][:2], seg_ax[-1][:2], (cx, cy)) * R
                if len(seg_ax) >= 3 else 0.0
            )

            segs.append({
                "id": len(segs),
                "ribbon": root,
                "slot": si_,
                "contours": sup,
                "axis": seg_ax,
                "order": order,
                "length": L,
                "mean_r": mr,
                "taper": taper,
                "curv": kap,
                "event_before": (kinds[si_-1] if si_ > 0 and si_-1 < len(kinds) else None),
            })

    joints = []
    for a in range(len(segs)):
        for b in range(a + 1, len(segs)):
            A, B = segs[a], segs[b]
            trunk, limb = (
                (A, B)
                if (A["mean_r"], A["length"], -A["id"]) >= (B["mean_r"], B["length"], -B["id"])
                else (B, A)
            )
            best = None
            for e in (limb["axis"][0], limb["axis"][-1]):
                for q, tp in enumerate(trunk["axis"]):
                    dd = math.hypot(e[0] - tp[0], e[1] - tp[1])
                    if best is None or dd < best[0]:
                        best = (dd, e is limb["axis"][0], q)
            if best is None or best[0] > join_frac * max(trunk["mean_r"], limb["mean_r"]):
                continue
            _, at_start, q = best
            apos = q / max(1, len(trunk["axis"]) - 1)
            same = A["ribbon"] == B["ribbon"] and abs(A["slot"] - B["slot"]) == 1
            jtype = 2 if same else (0 if (apos <= 0.15 or apos >= 0.85) else 1)
            l0, l1 = (limb["axis"][0], limb["axis"][-1]) if at_start else (limb["axis"][-1], limb["axis"][0])
            t0, t1 = (
                trunk["axis"][q],
                trunk["axis"][min(q + 1, len(trunk["axis"]) - 1)]
                if q < len(trunk["axis"]) - 1
                else trunk["axis"][max(q - 1, 0)],
            )
            ang_l = math.atan2(l1[1] - l0[1], l1[0] - l0[0])
            ang_t = math.atan2(t1[1] - t0[1], t1[0] - t0[0])
            joints.append({
                "trunk": trunk["id"],
                "limb": limb["id"],
                "type": jtype,
                "attach": apos,
                "angle": (ang_l - ang_t) % (2 * math.pi),
                "ratio": limb["mean_r"] / max(trunk["mean_r"], 1e-6),
                "signal": math.exp(-best[0] / max(trunk["mean_r"], 1e-6)),
            })

    par = list(range(len(segs)))

    def f2(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for j in joints:
        ra, rb = f2(j["trunk"]), f2(j["limb"])
        if ra != rb:
            par[max(ra, rb)] = min(ra, rb)
    groups = defaultdict(list)
    for s in segs:
        groups[f2(s["id"])].append(s["id"])

    ens = []
    for root in sorted(groups):
        ids = groups[root]
        idset = set(ids)
        js = [j for j in joints if j["trunk"] in idset and j["limb"] in idset]
        cnt = [sum(1 for i in ids if segs[i]["order"] == o) for o in (1, 2, 3, 4)]
        cnt[1] += sum(1 for j in js if j["type"] == 2)  # Second-order neck events
        genus = max(0, len(js) - len(ids) + 1)
        P_axis = [(a[0], a[1]) for i in ids for a in segs[i]["axis"]]
        mx_ = sum(p[0] for p in P_axis) / len(P_axis)
        my_ = sum(p[1] for p in P_axis) / len(P_axis)
        cxx = sum((p[0] - mx_) ** 2 for p in P_axis) / len(P_axis)
        cyy = sum((p[1] - my_) ** 2 for p in P_axis) / len(P_axis)
        cxy = sum((p[0] - mx_) * (p[1] - my_) for p in P_axis) / len(P_axis)
        tr = cxx + cyy
        dt = cxx * cyy - cxy * cxy
        disc = math.sqrt(max(tr * tr / 4 - dt, 0.0))
        l1 = tr / 2 + disc
        l2 = max(tr / 2 - disc, 1e-9)
        pa = 0.5 * math.atan2(2 * cxy, cxx - cyy)
        ux, uy = math.cos(pa), math.sin(pa)
        refl = [
            (
                mx_ + 2 * ((p[0] - mx_) * ux + (p[1] - my_) * uy) * ux - (p[0] - mx_),
                my_ + 2 * ((p[0] - mx_) * ux + (p[1] - my_) * uy) * uy - (p[1] - my_),
            )
            for p in P_axis
        ]
        md = sum(min(math.hypot(r_[0] - q[0], r_[1] - q[1]) for q in P_axis) for r_ in refl) / len(refl)
        sym = max(0.0, 1.0 - md / (0.25 * R))
        asp = math.sqrt(l1 / l2)
        bins = (
            float(min(3, cnt[0])),
            float(min(3, cnt[1])),
            float(min(3, cnt[2])),
            float(min(3, cnt[3])),
            float(min(2, genus)),
            float(min(7, int(sym * 8.0))),
            float(min(7, len(ids) - 1)),
            float(min(7, int(math.log2(max(1.0, asp)) * 2.0))),
        )
        ens.append({
            "id": len(ens),
            "segments": ids,
            "contours": sorted({c for i in ids for c in segs[i]["contours"]}),
            "bins": bins,
            "angles": [j["angle"] for j in js],
            "sym_axis": pa % math.pi,
        })
    return segs, joints, ens, {"R": R, "step": step}

def ribbon_bins(seg, R):
    aspect = seg["length"] / max(2.0 * seg["mean_r"], 1e-6)
    taper_norm = clamp(-seg["taper"] / 0.5, 0.0, 1.0)
    taper_bin = float(min(7, int(taper_norm * 8.0)))
    # Signed Menger curvature mapped across [-4.0, 4.0] -> [0.0, 1.0] -> 0..7
    curv_norm = clamp((seg["curv"] / 8.0) + 0.5, 0.0, 1.0)
    curv_bin = float(min(7, int(curv_norm * 8.0)))
    r_norm = clamp(seg["mean_r"] / (0.6 * R), 0.0, 1.0)
    r_bin = float(min(7, int(r_norm * 8.0)))
    aspect_bin = float(min(7, int(math.log2(max(1.0, aspect)) * 2.0)))
    order_bin = float(seg["order"] - 1)
    return (order_bin, aspect_bin, taper_bin, curv_bin, r_bin)
