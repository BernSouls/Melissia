"""perception/ownership.py -- the ONE border-ownership reader.

WHY THIS MODULE EXISTS. `BorderOwnershipLens` (perception/lenses/v2.py) emits
one detection per sampled boundary vertex with signature (arity 4):

    (arc_pos_bin, signed_dir_bin, curvature_bin, strength_bin)

`signed_dir_bin` is `_q_signed(signed_dir, max_bins=8)`, i.e. a SIGNED SCALAR
in [-1, 1] mapped to 0..7 -- NOT an angle, and NOT a cyclic direction. Three
readers in the tree disagreed about that slot:

  * perception/lenses/v2.py ContourLens: treated it as an 8-bin direction over
    2*pi and compared it with an edge bearing (a category error);
  * orchestration/binder.py `_run_border_ownership_edge`: treated it as an
    AXIAL angle over pi: `(bin + 0.5) * pi / 8` (a category error);
  * orchestration/binder.py `_bo_depth_constraints`: `(bin + 0.5)/8*2 - 1`, a
    signed scalar -- the ONLY reading that matches the producer.

This module is the single source of truth for that decode, so the readers
cannot drift again. The signed scalar is the edge-level FIGURE-GROUND signal:
`signed_dir > 0` means the detection's entity (its `alpha`) is the figure side
of the boundary, `< 0` means the named neighbour in `rho` ("neighbor") is.

SIGNED EDGE VOCABULARY. A border belongs to exactly one of its two flank
owners. We expose `figure_ground_pair` so the binder can emit the relation as
a SIGNED edge: positive coupling binds the border to the figure it owns and
negative coupling (OCCLUSION, sign=-1) marks the ground / T-occlusion side.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

__all__ = [
    "BO_BINS",
    "ownership_signed",
    "ownership_strength",
    "figure_ground_pair",
    "combine_ownership_cues",
]

#: Bin count of the signed-direction slot (the producer's `_q_signed(..., 8)`).
BO_BINS = 8


def _quantized(det: Any) -> Tuple[float, ...]:
    q = getattr(getattr(det, "sigma", None), "quantized", None)
    return tuple(q) if q else ()


def ownership_signed(det: Any) -> float:
    """The signed figure-ground value of a BorderOwnership detection, in
    [-1, 1]: +1 = the entity (`alpha`) is the figure, -1 = the neighbour is.
    Decoded as the bin CENTRE of the `_q_signed(..., 8)` slot, matching the
    producer and `_bo_depth_constraints`."""
    q = _quantized(det)
    if len(q) < 2:
        return 0.0
    return (float(q[1]) + 0.5) / BO_BINS * 2.0 - 1.0


def ownership_strength(det: Any) -> float:
    """The magnitude of the ownership signal in [0, 1] (bin centre of slot 3,
    falling back to `z`)."""
    q = _quantized(det)
    if len(q) >= 4:
        return (float(q[3]) + 0.5) / BO_BINS
    return abs(float(getattr(det, "z", 0.0) or 0.0))


def _neighbor(det: Any) -> Optional[Any]:
    for role in getattr(det, "rho", ()) or ():
        if len(role) >= 2 and role[0] == "neighbor":
            return role[1]
    return None


def figure_ground_pair(det: Any) -> Optional[Tuple[Any, Any]]:
    """(figure_anchor, ground_anchor) of a BorderOwnership detection, or None
    when it names no distinct neighbour. The ownership sign decides which of
    the two flank owners is the figure."""
    ent = getattr(det, "alpha", None)
    nb = _neighbor(det)
    if ent is None or nb is None or ent == nb:
        return None
    if ownership_signed(det) >= 0.0:
        return ent, nb
    return nb, ent


def combine_ownership_cues(
    cues: Mapping[str, Optional[float]],
    weights: Mapping[str, float],
) -> float:
    """Weighted, normalised mean of the per-vertex ownership cues.

    A cue that is `None` (no evidence: e.g. enclosure on a blank region, or a
    convexity term on an open chain) is omitted AND its weight is dropped from
    the normaliser, so the surviving cues keep full authority instead of being
    diluted toward zero. The weights are the `BO_WEIGHTS` family: the module
    that names a cue owns its weight, and this combiner is the only consumer.
    """
    num = 0.0
    den = 0.0
    for name, w in weights.items():
        v = cues.get(name)
        if v is None:
            continue
        wf = float(w)
        num += wf * float(v)
        den += abs(wf)
    if den <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, num / den))
