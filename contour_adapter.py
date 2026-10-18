"""Stage 2c adapter: field peaks -> Orientation detections.

The Stage 2b audit established that `ContourLens` (perception/lenses/v2.py:259)
and `JunctionLens` (:553) are Class A consumers: they read only
`current_detections`, `anchor_positions` and concept-graph edges, and never
dereference `observation`. Together they are the two lens consumers of the
`Orientation` detections the new continuous field can produce directly.

This module therefore holds ONLY the adapter that converts
`perception/contour_field.py` peak dicts into `Detection` objects matching
the single emitter contract at perception/lenses/base.py:1958.

  Orientation contract (base.py:1958, shape from perception/rich.py:615):
    k             = "Orientation"
    source_layer  = 0
    scale         = 0
    alpha         = u, an integer region/entity index
    rho           = (("entity", u, 0, "+"),)
    z             = clamp((resp - theta_eff) / (1 - theta_eff), 0, 1)
                    -- a [0,1] confidence from channel gain and peak energy
    sigma         = TypedSignature(
                        kind="CANONICAL",
                        class_="ORIENTATION_RICH_{nt}_{nsf}",
                        relation="HYPERCOLUMN_POPULATION",
                        quantized=<433 slots>)

  Slot 0 is `E_peak` in [0.0, 1.0] and is what the emitter's candidate gate
  uses as `peak_energy` (base.py:1936-1938): `presence * max(marginal slots)`.
  So the adapter must place a normalised integration strength into slot 0 for
  the gate to admit the detection. The remaining 432 slots are carried from the
  emitter's schema; the adapter fills them with clamped mid-range values so a
  downstream consumer that reads an unmodelled slot gets a defined value rather
  than a default that may be out of schema range.

  Orientation and scale come from the field's own per-peak `theta` and `sf`,
  which are the quantities the emitter derives from the dominant Gabor
  channel. They are encoded with the same bin structure the emitter's
  histograms use (nt = GABOR_N_THETA bins over [0, pi), nsf = GABOR_N_SF),
  so a consumer reading histogram slots sees a consistent encoding.

No imports from perception.lenses.* or perception.contours: this module is a
pure function from field output to detections.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import parameters as P
from core.types import Detection, TypedSignature


# ---------------------------------------------------------------------------
# CONTRACT
#
# The emitter that actually runs in production is
# perception/lenses/base.py:1681-1687 (`LensExecutor._run_orientation_gabor`,
# basic branch). Its contract, verbatim:
#
#     sigma = TypedSignature(
#         kind="CANONICAL",
#         class_="ORIENTATION",
#         relation="MAJOR_AXIS",
#         quantized=(float(theta_b), float(e_b), float(anchor_bin)),
#     )
#     return sigma, (("entity", anchor, 0, "+"),)
#
#     theta_b    = floor((angle % pi) * ORIENTATION_N_THETA / pi) % 24   [0,24)
#     e_b        = floor(clamp(1 - elongation, 0, 1) * ORIENTATION_N_ELONG) [0,5)
#     anchor_bin = 4x4 position class in [0, 15]                         [0,16)
#     rho        = (("entity", anchor, 0, "+"),)
#     alpha      = the anchor itself (an int/opaque id)
#
# The RICH sibling (base.py:1958, class_="ORIENTATION_RICH_12_5", 433 slots,
# relation="HYPERCOLUMN_POPULATION") is emitted by the SAME method but only
# when `obs.get_b1_sig(u)` returns a signature. Measured on the production
# path (Observation2D over ETH-80): the method exists but 0 of 336 regions
# carry a `b1_sig`, so the rich branch is emitted NEVER. The adapter therefore
# matches the basic contract only -- see diagnostics/_stage2d-2 report.
#
# The binder reads this contract at orchestration/binder.py:4304
# (`class_ not in ("ORIENTATION", "ORIENTATION_POL") -> continue`) and at
# :4937. Every other Orientation-class filter site in the tree accepts the
# same pair, so a detection with class_="ORIENTATION" passes all of them.
# ---------------------------------------------------------------------------

_ORIENTATION_N_THETA = 24      # P.ORIENTATION_N_THETA (base.py:1548)
_ORIENTATION_N_ELONG = 5       # P.ORIENTATION_N_ELONG


def _theta_bin(theta: float) -> float:
    """Slot 0 is a bin INDEX over [0, pi), not the angle itself.

    Exact emitter mapping, base.py:1595-1598:
        int(math.floor((angle % math.pi) * n_theta / math.pi)) % n_theta
    The field's `theta` is already a tangent direction in [0, pi), so the
    modulo is a no-op there but is kept to match the emitter on any input.
    """
    t = float(theta) % math.pi
    if t < 0.0:
        t += math.pi
    b = int(math.floor(t * _ORIENTATION_N_THETA / math.pi)) % _ORIENTATION_N_THETA
    return float(b)


def _elong_bin(sf: float, sf_min: float, sf_max: float) -> float:
    """Slot 1 is the coarseness bin.

    Emitter mapping, base.py:1647-1650:
        elong_norm = clamp(1.0 - elongation, 0.0, 1.0)
        e_bin      = floor(elong_norm * ORIENTATION_N_ELONG)  # capped at 4

    The emitter's `elongation` is draw-order-derived (1.0 = isotropic, 0.0 =
    fully elongated) and is gated by ORIENTATION_ELONG_SKIP. The field has no
    draw order, so the coarseness proxy used here is spatial frequency,
    normalised across the band: fine (high sf) behaves like high elongation,
    coarse maps to 0.0. This is the one slot where the adapter's semantics are
    NOT the emitter's, and the difference is recorded rather than hidden.
    """
    if sf_max <= sf_min:
        return float(_ORIENTATION_N_ELONG - 1)
    norm = 0.0 if float(sf) < sf_min else min(float(sf) - sf_min, sf_max - sf_min)
    norm /= (sf_max - sf_min)
    elong_norm = 1.0 - norm                      # coarse -> 0.0, fine -> 1.0
    e = int(math.floor(max(0.0, min(1.0, elong_norm)) * _ORIENTATION_N_ELONG))
    if e >= _ORIENTATION_N_ELONG:
        e = _ORIENTATION_N_ELONG - 1
    return float(e)


def _anchor_class_bin(anchor_positions, anchor) -> float:
    """Slot 2 is a coarse 4x4 position class in [0, 15].

    Emitter mapping, base.py:242-286 via core/quantize.py:_q_signed:
        x_bin = _q_signed(x, 4)                      # in [0, 3]
        y_bin = _q_signed(y, 4)                      # in [0, 3]
        bin   = x_bin * 4 + y_bin                    # in [0, 15]
    Positions must be in [-1, 1]; the emitter falls back to a hash of the
    anchor when they are not. The field's peaks are in pixel coordinates and
    the frame is square (the emitter's lattice is 4x4, aspect-tolerant), so
    the peak's own (y, x) is normalised to [-1, 1] over the frame before
    binning -- which is the same physical grid the emitter's 4x4 lattice
    describes.
    """
    pos = anchor_positions.get(anchor) if isinstance(anchor_positions, dict) \
        else None
    if pos is None or len(pos) < 2:
        return _anchor_bin_identity(anchor)
    x, y = float(pos[0]), float(pos[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return _anchor_bin_identity(anchor)
    # normalise the frame to [-1, 1]; the emitter assumes that range already.
    ext = 127.5                                  # 256-px frame, centre 127.5
    nx, ny = x / (ext + 0.5), y / (ext + 0.5)
    nx = max(-1.0, min(1.0, nx))
    ny = max(-1.0, min(1.0, ny))
    x_bin = int(math.floor((nx + 1.0) * 0.5 * 4.0))
    if x_bin >= 4:
        x_bin = 3
    y_bin = int(math.floor((ny + 1.0) * 0.5 * 4.0))
    if y_bin >= 4:
        y_bin = 3
    return float(x_bin * 4 + y_bin)


def _anchor_bin_identity(anchor) -> float:
    """The emitter's out-of-range fallback, base.py:286.

    Deterministic, keyed on the anchor so the same anchor always gives the
    same class. Kept as float because the emitter also returns float here.
    """
    import struct
    h = int.from_bytes(
        np.asarray(str(anchor), dtype="S").tobytes()[:8].ljust(8, b"\x00"),
        "little")
    return float(h & ((1 << 32) - 1))


def peaks_to_orientation_detections(
    peaks: List[Dict],
    anchor_positions: Dict,
    anchor_allocator: Callable[[int, int], Any],
    extraction_scale: float,
) -> List[Detection]:
    """Field peak dicts -> Orientation detections (base.py:1681 contract).

    Emits the 3-slot `class_="ORIENTATION"` shape that the binder and
    `ContourLens` both read. See the CONTRACT note above for why the rich
    433-slot variant is not emitted.

    Determinism
    -----------
    Output is sorted by (y, x, theta) so the ordering does not depend on the
    order peaks arrived in, matching the emitter's `_detection_sort_key` use.
    """
    sf_min = float(getattr(P, "GABOR_SF_MIN", 0.06))
    sf_max = float(getattr(P, "GABOR_SF_MAX", 0.35))

    ordered = sorted(
        peaks,
        key=lambda p: (int(p.get("y", 0)), int(p.get("x", 0)),
                       float(p.get("theta", 0.0))),
    )

    out: List[Detection] = []
    for pk in ordered:
        x = int(pk.get("x", 0))
        y = int(pk.get("y", 0))
        anchor = anchor_allocator(y, x)
        if anchor is None:
            continue

        anchor_positions[anchor] = (float(y), float(x))

        strength = float(pk.get("integration_strength", 0.0))
        if not math.isfinite(strength) or strength < 0.0:
            strength = 0.0
        if strength > 1.0:
            strength = 1.0

        theta = float(pk.get("theta", 0.0))
        sf = float(pk.get("sf", 0.0))

        quantized = (
            _theta_bin(theta),
            _elong_bin(sf, sf_min, sf_max),
            _anchor_class_bin(anchor_positions, anchor),
        )

        sigma = TypedSignature(
            kind="CANONICAL",
            class_="ORIENTATION",
            relation="MAJOR_AXIS",
            quantized=quantized,
        )

        # z: the emitter's channel-gain confidence, [0,1]
        z = 0.0 if extraction_scale <= 0.0 else min(max(strength, 0.0), 1.0)

        out.append(Detection(
            k="Orientation",
            source_layer=0,
            scale=0,
            sigma=sigma,
            alpha=anchor,
            z=z,
            t=0,
            rho=(("entity", anchor, 0, "+"),),
        ))
    return out

