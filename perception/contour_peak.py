"""perception/contour_peak.py -- CONTOUR_PEAK detections from field peaks.

Stage A2 of the front-end replacement. Stage A0/A1 built
`perception.contour_field` and validated it (deterministic, blank-safe,
persistence-passing). This module turns that field's peak list — peaks of the
continuous V1/V2 integration field, NOT discrete shock segments — into
`Detection` objects that downstream consumers will eventually read.

Additive: nothing here mutates the old front end, the old lenses, or the
pipeline, and nothing consumes CONTOUR_PEAK yet. A4 migrates consumers; this
stage only makes the detections exist.

WHAT THIS DOES NOT PROVE
------------------------
* That downstream lenses can read CONTOUR_PEAK. That is A4 work.
* That these signatures are discriminative. That is A3 (the bake-off).
* That the anchor scheme is final. It is caller-supplied; A4 reconciles it.

The anchor interface is deliberately pass-through: the caller owns the allocator
and the anchor_positions dict, because anchor identity is a pipeline-wide
decision (Position detections, border ownership and the spatial binders must
all agree on it) and A2 must not pre-empt it. Every slots bin below is stated
explicitly in this docstring so it can be re-derived or re-tuned without
reading the code.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Tuple

from core.types import Detection, TypedSignature

K_PEAK = "CONTOUR_PEAK"
CLASS_PEAK = "CONTOUR_PEAK"
RELATION_PEAK = "CONTOUR_PROFILE"

# --- slot bin definitions (fixed, documented, tunable) --------------------
# Measured over 2643 peaks from 5 real ETH-80 images:
#   signed_curvature      min -0.523  p50 -0.001  max 0.521   (2643 distinct)
#   bend_angle            min  0.000  p50  0.189  max 7.218
#   integration_strength  min  0.029  p50 17.015  max 25.446  NOT in [0,1]
#   coherence             min  0.086  p50  0.098  max 0.147   narrow, low
# Bin edges are chosen from these observed ranges; anything outside the range
# is clipped to the nearest edge, which keeps the signature total.
N_CURV_BINS = 8
CURV_CLIP = 0.6          # |signed_curvature| observed max 0.521; clip at 0.6
N_BEND_BINS = 8
BEND_CLIP = 1.5          # observed p99 1.182; clip at 1.5 rather than the
                         # outlier max 7.2, so a single wild bend does not
                         # collapse the top bin's meaning
N_STRENGTH_BINS = 8
STRENGTH_CLIP = 20.0     # observed p50 17.0, max 25.4; clip at 20
N_COHERENCE_BINS = 8
COHERENCE_CLIP = 0.15    # observed max 0.147; clip at 0.15

Z_CLAMP_LO, Z_CLAMP_HI = 0.0, 1.0


def _clip_bin(value, clip, n_bins):
    """Map a float to a bin index in [0, n_bins-1] by clipping to a range."""
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return 0
    v = float(value)
    if v < 0.0:
        v = 0.0
    if v > clip:
        v = clip
    q = int(math.floor(v / (clip / float(n_bins))))
    if q >= n_bins:
        q = n_bins - 1
    return q


def peaks_to_contour_peak_detections(
    peaks: List[Dict],
    anchor_allocator: Callable[[int, int], Any],
    anchor_positions: Dict,
    extraction_scale: float,
    preserve_input_order: bool = False,
    t: int = 0,
) -> List[Detection]:
    """Field peak dicts -> one CONTOUR_PEAK Detection per peak.

    Anchor is supplied by `anchor_allocator(y, x)`; the caller owns the
    allocator and `anchor_positions`. `anchor_positions` is written here only so
    that later callers (e.g. Position detections) can share the same anchors.

    `extraction_scale` is accepted for signature compatibility with the other
    adapters and is NOT used: CONTOUR_PEAK carries no spatial extent, and the
    bin edges above are absolute rather than scale-relative so that a peak
    means the same thing at every image size.
    """
    # Chain membership refers to the original field-list index, not sorted order.
    ordered = list(peaks) if preserve_input_order else sorted(
        peaks,
        key=lambda p: (int(p.get("y", 0)), int(p.get("x", 0)),
                       float(p.get("theta", 0.0))),
    )

    out: List[Detection] = []
    for pk in ordered:
        y = int(pk.get("y", 0))
        x = int(pk.get("x", 0))
        anchor = anchor_allocator(y, x)
        if anchor is None:
            continue

        anchor_positions[anchor] = (float(x), float(y))

        theta_idx = int(pk.get("theta_idx", 0)) % 12
        sf_idx = int(pk.get("sf_idx", 0))
        phase_bin = int(pk.get("phase_bin", 0)) % 4
        endstop_flag = 1 if int(pk.get("endstop_flag", 0)) else 0
        curv_bin = _clip_bin(pk.get("signed_curvature"), CURV_CLIP, N_CURV_BINS)
        bend_bin = _clip_bin(pk.get("bend_angle"), BEND_CLIP, N_BEND_BINS)
        strength_bin = _clip_bin(pk.get("integration_strength"),
                                 STRENGTH_CLIP, N_STRENGTH_BINS)
        coherence_bin = _clip_bin(pk.get("coherence"), COHERENCE_CLIP,
                                  N_COHERENCE_BINS)

        quantized = (
            float(theta_idx),
            float(sf_idx),
            float(curv_bin),
            float(bend_bin),
            float(phase_bin),
            float(endstop_flag),
            float(strength_bin),
            float(coherence_bin),
        )

        sigma = TypedSignature(
            kind="CANONICAL",
            class_=CLASS_PEAK,
            relation=RELATION_PEAK,
            quantized=quantized,
        )

        # z = integration_strength clamped to [0, 1]: the raw field value is
        # not normalized (observed p50 17.0 on a 0..25 scale), so the clamp is
        # a saturating confidence rather than a rescale, deliberately loud at
        # the top bin instead of pretending to be bounded.
        strength = pk.get("integration_strength", 0.0)
        if not isinstance(strength, (int, float)) or \
                not math.isfinite(float(strength)):
            strength = 0.0
        z = float(min(max(strength, Z_CLAMP_LO), Z_CLAMP_HI))

        out.append(Detection(
            k=K_PEAK,
            source_layer=1,
            scale=0,
            sigma=sigma,
            alpha=anchor,
            z=z,
            t=t,
            rho=(("entity", anchor, 0, "+"),),
        ))
    return out


def peaks_to_position_detections(
    peaks: List[Dict],
    anchor_allocator: Callable[[int, int], Any],
    anchor_positions: Dict,
) -> List[Detection]:
    """One POSITION detection per peak, sharing the peak's anchor.

    Layout matched from perception/lenses/base.py:1495-1503 -- arity 3,
    `(x_bin, y_bin, z_bin)`, each binned to 16 slots over a [-1, 1] normalised
    frame by core/quantize.py:_q_signed:

        sigma = TypedSignature(kind="CANONICAL", class_="POSITION",
                               relation="OBJECT_CENTRIC_REL",
                               quantized=(float(x_b), float(y_b), float(z_b)))

    z_b is 7.5 -- the centre of the third slot's range, the same
    "information absent" value the pipeline uses for unbinned depth slots
    (see EDGE_OCCLUSION's stack_bin 8.0 convention). A peak carries no depth
    channel, and a fabricated 0.0 would assert flatness the field never
    measured.

    Frame positions are normalised over a 256-px field (centre 127.5), which is
    the size `field_from_luminance` was validated on. Callers working at other
    sizes must normalise before calling.
    """
    ordered = sorted(
        peaks,
        key=lambda p: (int(p.get("y", 0)), int(p.get("x", 0))),
    )

    out: List[Detection] = []
    for pk in ordered:
        y = float(pk.get("y", 0))
        x = float(pk.get("x", 0))
        anchor = anchor_allocator(int(round(y)), int(round(x)))
        if anchor is None:
            continue
        anchor_positions[anchor] = (y, x)

        # _q_signed(v, 16): (clamp(v,-1,1) + 1) / 2 * 16, capped at 15.
        def _qs(v):
            c = max(-1.0, min(1.0, v))
            q = int(math.floor((c + 1.0) * 0.5 * 16.0))
            return 15 if q >= 16 else q

        sigma = TypedSignature(
            kind="CANONICAL",
            class_="POSITION",
            relation="OBJECT_CENTRIC_REL",
            quantized=(float(_qs(x / 127.5 - 1.0)),
                       float(_qs(y / 127.5 - 1.0)),
                       7.5),
        )
        out.append(Detection(
            k="Position",
            source_layer=0,
            scale=0,
            sigma=sigma,
            alpha=anchor,
            z=1.0,
            t=0,
            rho=(("entity", anchor, 0, "+"),),
        ))
    return out
