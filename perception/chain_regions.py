"""perception/chain_regions.py -- [W1] Field-based region formation.

The old eye's identity-carrying lenses (Boundary/Color/Texture/Surface and
the Shape family) read REGIONS produced by the segmenter: a list of entity
indices with a boundary polyline, a position, an area, and interior colour /
texture samples. Removing the segmenter therefore removes not just the
polylines but the entities those lenses iterate.

This module rebuilds region-like entities from the field's own output, so
the same lenses have something to emit for without a boundary tracer:

  * CLOSED chains (closure_bin == 1) are rings -- they enclose an area, so
    they become regions with a shoelace interior and an interior centroid.
  * OPEN chains become bounded envelopes -- a region whose "boundary" is
    the chain plus the chord joining its ends, which is the convex-hull-free
    envelope a contour-plus-closure pair would produce.

Colour and texture are sampled from the channels the front end already
computed (`compute_opponent_retinex` luminance + opponent planes), so
surface and texture descriptors keep working on real image content rather
than on a synthetic fill.

Neural analogue
---------------
A closed contour in V2/V4 behaves as a figure: the interior is assigned
surface properties (colour, texture) and the exterior becomes ground. That
is exactly the role this module gives a closed chain, and the reason an
open chain still yields an interior is that biological contours are
frequently interrupted by occlusion -- the continuations are bridged by
illusory-contour units rather than rejected.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import parameters as P

from perception.chain_polylines import (  # noqa: F401
    chain_polyline_for_detection,
    close_polyline,
    member_positions,
)

# interior sampling grid: odd so it always has an exact centre pixel.
INTERIOR_GRID_N = 5

# minimum enclosed area for a closed chain to become a region.
#
# Expressed as a FRACTION OF IMAGE AREA (see P.FIELD_REGION_MIN_AREA_FRAC),
# so the floor is scale-free and identical at 128, 256, or any crop -- an
# absolute pixel floor would tighten as resolution rises and silently reject
# every real figure at high resolution. Only used as a fallback when no
# image extent is known.
MIN_REGION_AREA_FRAC_FALLBACK = 1e-4

# a chain polyline with fewer points than this cannot carry a shape.
CHAIN_MIN_POLYLINE = 3


def _point_in_ring(x: float, y: float, pts: List[Tuple[float, float]]) -> bool:
    """Even-odd ray-cast test on a simple polygon (closed walk)."""
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _interior_centroid(pts: List[Tuple[float, float]],
                       g: Dict[str, Any]) -> Tuple[float, float]:
    """Innermost interior grid point of the ring, else the centroid.

    Samples a `INTERIOR_GRID_N` x `INTERIOR_GRID_N` grid spanning the bbox
    and returns the interior point closest to the ring's centre. Sampling an
    actual interior pixel -- rather than blindly using the mean of vertices
    -- is what lets interior colour come from inside the figure instead of
    on its contour, where a boundary pixel would systematically bias surface
    colour towards the background.

    Searching from the centre outward also makes the choice of interior
    sample stable against small contour jitter: a near-symmetric figure
    keeps the same sample under modest deformation, which a
    raster-order search would not.
    """
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mid_x = (max(xs) + min(xs)) / 2.0
    mid_y = (max(ys) + min(ys)) / 2.0
    n = INTERIOR_GRID_N
    cands = []
    for gy in range(n):
        for gx in range(n):
            x = min(xs) + (max(xs) - min(xs)) * (gx + 0.5) / n
            y = min(ys) + (max(ys) - min(ys)) * (gy + 0.5) / n
            if _point_in_ring(x, y, pts):
                cands.append(((x - mid_x) ** 2 + (y - mid_y) ** 2, x, y))
    if cands:
        cands.sort()
        best = cands[0]
        return (best[1], best[2])
    return g["centroid"]


class FieldRegionSource:
    """Region entities derived from field chains.

    Attributes
    ----------
    entities : dict anchor -> entity
        Each entity has `anchor`, `polyline` (closed walk), `closed`,
        `area`, `centroid`, `interior`, `length`, plus sampling results
        once `attach_surface` has been called.
    order : list of anchors
        Deterministic emission order (by anchor), so re-running produces
        byte-identical detections.
    """

    def __init__(self) -> None:
        self.entities: Dict[Any, Dict[str, Any]] = {}
        self.order: List[Any] = []

    # -- construction ---------------------------------------------------
    def from_chain_detections(
        self,
        detections: List[Any],
        anchor_positions: Optional[Dict[Any, Any]],
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> "FieldRegionSource":
        """Build entities from chains.

        `image_shape` is (height, width) of the frame the chains were
        extracted from, used to turn the scale-free area FRACTION into an
        absolute floor. Without it the fallback fraction is applied to a
        nominal 512^2 so the gate still behaves rather than vanishing.
        """
        if image_shape is not None:
            try:
                area_px = float(int(image_shape[0]) * int(image_shape[1]))
            except (TypeError, IndexError, ValueError):
                area_px = float(512 * 512)
        else:
            area_px = float(512 * 512)
        frac = float(getattr(P, "FIELD_REGION_MIN_AREA_FRAC",
                             MIN_REGION_AREA_FRAC_FALLBACK))
        min_area = max(1.0, area_px * frac)

        for det in detections or ():
            g = chain_polyline_for_detection(det, anchor_positions)
            if g is None:
                continue
            pts = g["points"]
            closed = g["closed"]
            if closed:
                if g["area"] < min_area:
                    continue
                walk = list(g["walk"])
                interior = _interior_centroid(walk, g)
            else:
                # open chain: envelope = chain + closing chord
                if len(pts) < 2:
                    continue
                walk = list(pts) + [pts[0]]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                span_x = max(xs) - min(xs)
                span_y = max(ys) - min(ys)
                cross = 0.0
                for i in range(len(pts) - 1):
                    cross += pts[i][0] * pts[i + 1][1]
                    cross -= pts[i + 1][0] * pts[i][1]
                cross += pts[-1][0] * pts[0][1] - pts[0][0] * pts[-1][1]
                area = 0.5 * abs(cross)
                if area < min_area and max(span_x, span_y) < 4.0:
                    continue
                interior = g["centroid"]
            self.entities[det.alpha] = dict(
                anchor=det.alpha,
                polyline=walk,
                closed=closed,
                area=g["area"] if closed else area,
                centroid=g["centroid"],
                interior=interior,
                length=g["length"],
                source="field",
            )
        self.order = sorted(self.entities, key=_sort_key)
        return self

    # -- surface sampling ----------------------------------------------
    def attach_surface(self, channels: Optional[Dict[str, Any]]) -> None:
        """Sample luminance/opponent at each entity's interior point.

        `channels` is the dict from `compute_opponent_retinex` (luminance +
        opponent planes). Sampling is nearest-pixel with bounds clamping; an
        entity whose interior falls outside the plane keeps neutral values
        rather than being dropped, so a figure at the frame edge still
        produces a region.
        """
        if not channels:
            for e in self.entities.values():
                e["lum"] = 0.5
                e["opponent"] = (0.0, 0.0)
            return
        lum = np.asarray(channels.get("lum",
                                      channels.get("luminance", [])),
                         dtype=np.float64)
        opp = channels.get("opponent") or channels.get("opp") or None
        opp = np.asarray(opp, dtype=np.float64) if opp is not None else None

        def _sample(plane, x: float, y: float, default: float) -> float:
            if plane is None or plane.size == 0 or plane.ndim < 2:
                return default
            h, w = plane.shape[0], plane.shape[1]
            xi = int(min(w - 1, max(0, round(x))))
            yi = int(min(h - 1, max(0, round(y))))
            v = float(plane[yi, xi])
            return v if math.isfinite(v) else default

        for e in self.entities.values():
            x, y = e["interior"]
            e["lum"] = _sample(lum, x, y, 0.5)
            if opp is not None and opp.ndim == 3:
                e["opponent"] = (
                    _sample(opp[..., 0], x, y, 0.0),
                    _sample(opp[..., 1], x, y, 0.0),
                )
            else:
                e["opponent"] = (0.0, 0.0)
        return

    # -- access ---------------------------------------------------------
    def entity_anchors(self) -> List[Any]:
        return list(self.order)

    def get(self, anchor: Any) -> Optional[Dict[str, Any]]:
        return self.entities.get(anchor)

    def __len__(self) -> int:
        return len(self.entities)


def _sort_key(k: Any) -> Tuple[int, Any]:
    """Deterministic ordering for mixed anchor types.

    Anchors can be ints (chain anchors) or other hashables; grouping by
    type name then value keeps the order stable and reproducible.
    """
    try:
        return (0, int(k))
    except (TypeError, ValueError):
        return (1, str(k))


def build_field_regions(
    chain_detections: List[Any],
    anchor_positions: Optional[Dict[Any, Any]],
    channels: Optional[Dict[str, Any]] = None,
    image_shape: Optional[Tuple[int, int]] = None,
) -> FieldRegionSource:
    """One-call construction: chains -> entities -> sampled surfaces."""
    src = FieldRegionSource().from_chain_detections(chain_detections,
                                                    anchor_positions,
                                                    image_shape)
    src.attach_surface(channels)
    return src


def variant_fingerprint() -> Dict[str, Any]:
    """The arm parameters that change what a bake-off measures.

    Recorded into every diagnostic artifact. A result measured without the
    arm stamped is not comparable to one measured with it, because
    FIELD_REGION_SOURCE swaps the whole entity source and the area fraction
    changes which chains become figures.
    """
    return {
        "field_region_source": bool(getattr(P, "FIELD_REGION_SOURCE", False)),
        "field_region_min_area_frac": float(
            getattr(P, "FIELD_REGION_MIN_AREA_FRAC",
                    MIN_REGION_AREA_FRAC_FALLBACK)),
    }


if __name__ == "__main__":
    # smoke: a 20x20 square ring must yield ONE entity with area 400.
    sq = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0), (0.0, 0.0)]
    det = type("D", (), {
        "alpha": 7,
        "sigma": type("S", (), {"quantized": (4, 0, 0, 1, 0, 0)})(),
        "rho": tuple([("entity", 7, 0, "+")] +
                     [("member", k, i, "+") for i, k in
                      enumerate(["a", "b", "c", "d"])]),
    })()
    pos = {"a": (0.0, 0.0), "b": (20.0, 0.0), "c": (20.0, 20.0),
           "d": (0.0, 20.0)}
    src = build_field_regions([det], pos)
    print("entities:", len(src), "anchors:", src.entity_anchors())
    if len(src):
        e = src.get(7)
        print("closed:", e["closed"], "area:", e["area"],
              "interior:", tuple(round(v, 2) for v in e["interior"]))
