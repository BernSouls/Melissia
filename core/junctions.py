"""core/junctions.py -- T-junction detection over segmented regions.

MOVE ONLY from primitives.py. Zero project imports except core.hashing,
core.spatial and parameters (a leaf module).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from core.hashing import clamp
from core.spatial import SpatialHash, anchor_in_boxes, tangent_at

import parameters as P

def detect_junctions(regions, max_gap=P.JUNCTION_MAX_GAP,
                     perp_tol=P.JUNCTION_T_PERP_TOL):
    """T/Y/X junction classification.

    Returns list of (kind, stem, top_indices, contact_pt, confidence).
    `top_indices` is ALWAYS a tuple.

    [B.25 HAZARD -- read before wiring a consumer] This is NOT
    `sensory_floor._detect_t_junctions`. That function returns 4-tuples
    (stem, top, contact, confidence) and IS what `binder._run_occlusion`
    unpacks; this one returns 5-tuples with a `kind` prefix. They are two
    different functions with different schemas. Passing this function's
    output to `_run_occlusion` would introduce the crash that the B.25
    "correction" claimed to fix. Wiring happens only in the commit that
    adds this function's consumer.

    [B.20, adapted] `max_gap` / `perp_tol` parameterize what the draft's
    body inlined as literals (`dd < 3.0`, `perp < 0.15`). Bound to
    P.JUNCTION_MAX_GAP / P.JUNCTION_T_PERP_TOL in the B.15 commit, same
    values -- and NOT the T_JUNCTION_MAX_GAP / T_JUNCTION_MAX_PERP_DEV
    pair (6.0 / 0.5) that sensory_floor's 4-tuple detector reads.
    """
    junctions = []
    for i, r_stem in enumerate(regions):
        poly = r_stem.get("boundary_polyline", [])
        if len(poly) < 2:
            continue
        for at_start in (True, False):
            pt = poly[0] if at_start else poly[-1]
            tan_stem = tangent_at(poly, at_start=at_start)
            touches = []
            for j, r_other in enumerate(regions):
                if i == j:
                    continue
                poly_o = r_other.get("boundary_polyline", [])
                if len(poly_o) < 2:
                    continue
                for q_start in (True, False):
                    q = poly_o[0] if q_start else poly_o[-1]
                    dd = math.hypot(pt[0] - q[0], pt[1] - q[1])
                    if dd < max_gap:
                        t_o = tangent_at(poly_o, at_start=q_start)
                        d = abs(tan_stem - t_o) % math.pi
                        touches.append((j, d, dd))
            if not touches:
                continue
            if len(touches) == 1:
                j, ang, dd = touches[0]
                perp = abs(ang - math.pi / 2.0)
                kind = "T" if perp < perp_tol else "Y"
                junctions.append((kind, i, (j,), pt, 1.0 - dd / max_gap))
            elif len(touches) == 2:
                js = tuple(t[0] for t in touches)
                angs = [t[1] for t in touches]
                kind = "T" if all(abs(a - math.pi / 2.0) < perp_tol
                                   for a in angs) else "Y"
                conf = 1.0 - max(t[2] for t in touches) / max_gap
                junctions.append((kind, i, js, pt, conf))
            else:
                js = tuple(t[0] for t in touches[:3])
                conf = 1.0 - max(t[2] for t in touches) / max_gap
                junctions.append(("X", i, js, pt, conf))
    return junctions

