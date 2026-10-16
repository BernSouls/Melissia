"""perception/anti_phase.py -- negative (anti-phase) coupling sources.

The coupled-phase oscillator (dynamics/oscillator.py) separates two cliques
only when the coupling matrix can say NO: a negative edge weight drives the
phase difference toward pi instead of toward 0. Measured before this module
existed (diagnostics/_a6_maskfree_result.json, M4/A6 result): the observed
graph for two-panel composites carried 146 edges, 146 with sign=+1 and ZERO
anti-phase, 118 of them EDGE_CONTACT proximity noise -- so two-panel
separation could not be positive for a structural reason, not a tuning one.

Two sources, both geometric, both computable from the live field's own
detections:

  1. Border ownership -- opposite flank owners across a shared boundary.
     A BorderOwnership detection names two entities: the entity whose
     boundary it is (rho "entity") and the entity on the far side of that
     boundary (rho "neighbor"). Those two are the owners of the opposite
     flanks of one shared boundary, which is the figure-ground / occlusion
     relation, and it is a reason to separate rather than to bind.

  2. Containment -- closed-chain interior vs exterior. A PEAK_CHAIN whose
     closure slot is set and whose member ring encloses another
     detection's anchor separates that detection (interior) from the ring
     (exterior): the ring is one surface's silhouette, the enclosed
     structure sits behind it on another depth layer.

Both sources are pure functions of detections plus anchor positions, so
they are testable without an image, and deterministic: every iteration is
over sorted keys and every tie-break is explicit.

WHY THE DIRECTION SLOT IS NOT READ
----------------------------------
The BorderOwnership direction slot (quantized[1]) looks like a direction:
three readers in the tree treat it as an angle, in three different
conventions --
  * perception/lenses/v2.py ContourLens: an 8-bin direction over 2*pi,
    compared with an edge bearing within one bin;
  * orchestration/binder.py `_run_border_ownership_edge`:
    (bin + 0.5) * pi / 8, an AXIAL angle over pi, compared with an axial
    bearing;
  * orchestration/binder.py `_bo_depth_constraints`:
    (bin + 0.5) / 8 * 2 - 1, a SIGNED scalar.
Only the third matches the producer: the lens emits
`_q_signed(signed_dir, max_bins=8)`, and `_q_signed` maps [-1, 1] to 0..7
via (v + 1) / 2 * max_bins. `signed_dir` is a weighted sum of signed
evidence cues (curvature, texture, colour, gravity), NOT an angle, so the
angle readings are a category error. Measured on the Q3 composites
(diagnostics/_q3_bopairs.py): the slot decodes to a signed value of
p50 = +0.125 with 100% of detections positive, i.e. it carries no side
information at all in this corpus. A rule that gates on it inherits the
category error and rejects arbitrarily, which is what happened to the
first attempt (1 accidental pair out of 15 detections).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from perception.chain_polylines import member_positions, order_ring_positions

__all__ = [
    "border_ownership_claims",
    "ring_polygon",
    "point_in_polygon",
    "containment_pairs",
]

#: Minimum border-ownership strength (|signed_dir|, already in [0, 1]) for a
#: flank claim to count. Calibrated by a stated rule (diagnostics/
#: _q3_strength.py, measured before this line was written): the p10 of |z|
#: over 552 BorderOwnership detections on 3 composites + 2 real images is
#: 0.200 (p50 0.214, max 0.347), and the rule is "p10 rounded down to two
#: decimals", so the threshold is 0.19 and about 90% of claims pass.
MIN_CLAIM_STRENGTH = 0.19
#: Minimum members for a closed chain to count as a ring. A 3-point closed
#: contour is a triangle, which is a legitimate ring; the caller also passes
#: an area floor, so a tiny ring cannot enclose anything.
_MIN_RING_MEMBERS = 3

def _claim_endpoints(det: Any) -> Optional[Tuple[Any, Any]]:
    """(entity_anchor, neighbour_anchor) of a BorderOwnership detection."""
    ent = getattr(det, "alpha", None)
    nb = None
    for role in getattr(det, "rho", ()) or ():
        if len(role) >= 2 and role[0] == "neighbor":
            nb = role[1]
            break
    if ent is None or nb is None or ent == nb:
        return None
    return ent, nb


def border_ownership_claims(
    bo_dets: Sequence[Any],
    min_strength: float = MIN_CLAIM_STRENGTH,
) -> List[Tuple[Any, Any, float]]:
    """Anti-phase pairs from opposite flank owners across a shared boundary.

    A BorderOwnership detection names the entity whose boundary it is (its
    anchor) and the entity on the far side of that boundary (rho
    ``neighbor``). Those two are the owners of the opposite flanks of one
    shared boundary, which separates them: the figure side stands in front
    of the ground side. The detection's ``z`` is the magnitude of the
    evidence sum, so it is a confidence, and it is the ONLY quantity this
    rule uses -- the direction slot is a signed evidence scalar, not an
    angle (module docstring), so it is deliberately not consulted.

    Multiple vertices of the same boundary are combined per ordered pair by
    maximum strength, so an over-sampled boundary cannot inflate a pair.
    Pairs are returned in deterministic (repr, repr) order.
    """
    directed: Dict[Tuple[Any, Any], float] = {}
    for det in bo_dets:
        pair = _claim_endpoints(det)
        if pair is None:
            continue
        strength = abs(float(getattr(det, "z", 0.0) or 0.0))
        if strength < float(min_strength):
            continue
        a, b = pair
        prev = directed.get((a, b))
        if prev is None or strength > prev:
            directed[(a, b)] = strength
    out = []
    for (a, b), s in directed.items():
        lo, hi = (a, b) if repr(a) < repr(b) else (b, a)
        out.append((lo, hi, s))
    out.sort(key=lambda t: (repr(t[0]), repr(t[1])))
    return out

def ring_polygon(
    det: Any,
    anchor_positions: Optional[Dict[Any, Any]],
    min_area: float = 0.0,
) -> Optional[List[Tuple[float, float]]]:
    """Ordered polygon of a closed PEAK_CHAIN, or None when it is not a ring.

    The chain's rho order is a valid traversal for closed chains emitted by
    form_chains' spine walker, but the ring flag (closure slot) is the
    authority: a chain without it is never a container, however its members
    happen to be ordered.
    """
    q = getattr(getattr(det, "sigma", None), "quantized", None)
    if not q or len(q) < 4 or not anchor_positions:
        return None
    if float(q[3]) < 0.5:
        return None
    pts = member_positions(det, anchor_positions)
    if len(pts) < _MIN_RING_MEMBERS:
        return None
    order = order_ring_positions(list(pts))
    if order:
        poly = [pts[i] for i in order]
    else:
        poly = list(pts)
    area = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    if 0.5 * abs(area) < float(min_area):
        return None
    return poly


def point_in_polygon(pt: Tuple[float, float],
                     poly: Sequence[Tuple[float, float]]) -> bool:
    """Even-odd ray cast, boundary points excluded."""
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xin = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xin:
                inside = not inside
    return inside


def _ring_scale(poly: Sequence[Tuple[float, float]]) -> float:
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    rms = math.sqrt(sum((p[0] - cx) ** 2 + (p[1] - cy) ** 2
                        for p in poly) / len(poly))
    return rms


def containment_pairs(
    chain_dets: Sequence[Any],
    other_dets: Sequence[Any],
    anchor_positions: Optional[Dict[Any, Any]],
    min_area: float = 0.0,
) -> List[Tuple[Any, Any, float]]:
    """(ring_anchor, contained_anchor, depth) for closed-chain containment.

    ``depth`` is exp(-(distance from the ring centre) / (ring RMS radius)),
    so a structure at the centre of a ring is strongly contained and one
    hugging the boundary is weakly contained. Anchors that are members of
    the ring itself are excluded (a ring does not contain its own contour).
    """
    if not anchor_positions:
        return []
    rings = []
    for det in chain_dets:
        poly = ring_polygon(det, anchor_positions, min_area=min_area)
        if poly is None:
            continue
        members = {role[1] for role in (getattr(det, "rho", ()) or ())
                   if len(role) >= 2 and role[0] == "member"}
        rings.append((getattr(det, "alpha", None), poly, members))
    if not rings:
        return []
    out: List[Tuple[Any, Any, float]] = []
    for ring_anchor, poly, members in rings:
        scale = _ring_scale(poly)
        if scale <= 1e-9:
            continue
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        for det in other_dets:
            anchor = getattr(det, "alpha", None)
            if anchor is None or anchor == ring_anchor or anchor in members:
                continue
            pt = anchor_positions.get(anchor)
            if pt is None or len(pt) < 2:
                continue
            if not point_in_polygon((float(pt[0]), float(pt[1])), poly):
                continue
            d = math.hypot(float(pt[0]) - cx, float(pt[1]) - cy)
            depth = math.exp(-d / scale)
            out.append((ring_anchor, anchor, depth))
    out.sort(key=lambda t: (repr(t[0]), repr(t[1])))
    return out
