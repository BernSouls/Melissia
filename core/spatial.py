"""core/spatial.py -- spatial hashing, box queries, image pyramids.

MOVE ONLY from primitives.py. Zero project imports except core.hashing and
parameters (a leaf module).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from core.hashing import clamp, serialize_value

import parameters as P

class SpatialHash:
    """Uniform grid over 2D anchor positions.

    Deterministic iteration; cells keyed by (int, int). Neighbors
    queries return [(anchor_id, (x, y)), ...].
    """
    def __init__(self, positions, cell_size):
        self.cell_size = max(float(cell_size), 1e-9)
        self._cells = defaultdict(list)
        for anchor, pos in positions.items():
            if pos is None or len(pos) < 2:
                continue
            ci = int(pos[0] // self.cell_size)
            cj = int(pos[1] // self.cell_size)
            self._cells[(ci, cj)].append(
                (anchor, (float(pos[0]), float(pos[1]))))
        for key in self._cells:
            self._cells[key].sort(key=lambda ap: serialize_value(ap[0]))

    def neighbors(self, pos, radius):
        r_cells = int(math.ceil(radius / self.cell_size)) + 1
        ci = int(pos[0] // self.cell_size)
        cj = int(pos[1] // self.cell_size)
        r2 = radius * radius
        out = []
        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):
                for anchor, p in self._cells.get((ci + di, cj + dj), ()):
                    dx = p[0] - pos[0]
                    dy = p[1] - pos[1]
                    if dx * dx + dy * dy <= r2:
                        out.append((anchor, p))
        return out

def anchor_in_boxes(anchor_pos, boxes):
    """True if (x, y) lies inside any (x0, y0, x1, y1) box."""
    if not boxes:
        return True
    x, y = anchor_pos
    for (x0, y0, x1, y1) in boxes:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False

def tangent_at(polyline, at_start, span=3):
    """Local tangent at one end of an ordered polyline."""
    n = len(polyline)
    if n < 2:
        return 0.0
    span = max(1, min(span, n - 1))
    if at_start:
        p1, p2 = polyline[0], polyline[span]
    else:
        p1, p2 = polyline[n - 1 - span], polyline[n - 1]
    return math.atan2(p2[0] - p1[0], p2[1] - p1[1])

def normalize_2d(response, sigma=None, pool_radius=None):
    """Divisive normalization with an isotropic Gaussian pool.

    Heeger 1992; Carandini & Heeger 2012; Carandini et al. 1997.
    """
    sigma = P.DIVISIVE_SIGMA if sigma is None else sigma
    pool_radius = (
        P.DIVISIVE_POOL_RADIUS if pool_radius is None else pool_radius
    )
    pooled = gaussian_filter(
        response.astype(np.float32), sigma=pool_radius, mode="reflect"
    )
    return (response / (sigma + pooled)).astype(np.float32)

def _blur_2x(img):
    b = gaussian_filter(img, sigma=1.0, mode="reflect")
    return b[::2, ::2]

def build_pyramid(img, n_levels=None):
    """Gaussian pyramid; levels[0] is the input, each next level halves."""
    n_levels = n_levels or P.PYRAMID_N_LEVELS
    img = img.astype(np.float32)
    levels = [img]
    for _ in range(n_levels - 1):
        levels.append(_blur_2x(levels[-1]))
    return levels


# --- moved from orchestration/pipeline.py (Move 3.5): pure geometry ---
def _pos_dist(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Euclidean distance over at most 3 dims (workspace is 2D/3D)."""
    dims = min(len(a), len(b), 3)
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(dims)))
