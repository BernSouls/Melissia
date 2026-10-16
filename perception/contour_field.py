"""perception/contour_field.py — Continuous V1/V2/V4 contour field.

Replaces the shock-graph segmenter (perception/contours.py). Peaks of a
continuous integration field are the primitives; no region partition,
no medial axis, no segments.

Authoritative spec: master plan v17/v18, Steps A0 and A1.

This module is standalone. It returns numpy arrays and a list of peak
dicts. It does not emit Detection objects, does not touch the graph,
does not import from binder / mapper / cores / distributor / the old
contours module. Peak-to-Detection conversion is Stage 2.

Determinism
-----------
- float32 throughout; Python floats are cast to np.float32 before any
  array arithmetic.
- integer-pixel shifts only; `int(round(k * cos(theta)))` is IEEE-754
  deterministic on the same input.
- sorted iteration for every tie-break.
- direct convolution (scipy.ndimage.convolve, mode="reflect") for
  kernels at or below FFT_KERNEL_THRESHOLD; fftconvolve otherwise. BLAS
  threading is capped at the project's standard single-worker setting
  (enforced at the top of perception.filters, inherited here).

Runtime note
------------
The beam integration is a shift-and-sum over integer offsets along
each orientation. O(n_theta * n_sf * K) numpy slice operations, K =
int(3 * wavelength). No dense O(N^2) pair loop. Measured runtime is
reported by the diagnostics block at the bottom of this file.

Documented approximations
-------------------------
1. AF angular tolerance (|theta_p - theta_q| < 0.35 rad) is applied
   by treating each orientation channel independently. A fractional
   cross-channel blend is a future refinement: the +-1 index gap is
   pi/12 = 15 deg, which falls INSIDE the ~20 deg tolerance, so a naive
   blend at neighbouring channels would over-count collinear support.
2. Cocircular support (plan: circle through two orientations with
   radius in [5, 200], tangent error < 0.15) is approximated here by a
   single wide isotropic Gaussian of the summed complex field. The
   full pair computation is O(N^2) over orientation pairs and is not
   implemented.
3. V4 curvature in this module is the directional derivative of the
   dominant-orientation field along its perpendicular. Menger
   curvature on the ridge polyline (A1) is the definitive measure and
   is computed at peak positions after ridge extraction.
4. Peak descriptor `phase_bin` is 4-bin (plan: 4 bins). Ridge peak
   spacing is 1.0 * median_wavelength (plan: 1.0).
5. Peak gating is relative-AND-absolute. The relative term keeps the
   top of the field on real images; the absolute term
   (PEAK_MIN_INTEGRATION_ABS) exists because a relative threshold
   alone normalises noise against itself and lets a structure-free
   frame through.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import convolve as nd_convolve
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve

import parameters as P
from perception.filters import GaborBank


# =============================================================================
# Parameters
# =============================================================================

def _p(name: str, default):
    return getattr(P, name, default)


_FRONT_END_REF_DIAGONAL_PX = _p("FRONT_END_REF_DIAGONAL_PX",
                                362.0386719678744)

_GABOR_N_THETA = int(_p("GABOR_N_THETA", 12))
_GABOR_N_SF = int(_p("GABOR_N_SF", 5))
_GABOR_SF_MIN = float(_p("GABOR_SF_MIN", 0.06))
_GABOR_SF_MAX = float(_p("GABOR_SF_MAX", 0.35))

_DIVISIVE_SIGMA = float(_p("DIVISIVE_SIGMA", 0.1))
_DIVISIVE_POOL_RADIUS = float(_p("DIVISIVE_POOL_RADIUS", 2.0))
_DIVISIVE_POOL_SIGMA = float(_p("DIVISIVE_POOL_SIGMA", 2.0))

_AF_LONGITUDINAL_MULT = float(_p("AF_LONGITUDINAL_MULT", 3.0))
# [PLACEHOLDER] not read by this module: the same-channel beam is
# perpendicular-offset-free by construction, and the angular tolerance is
# applied channel-wise. Kept as named placeholders for a later
# cross-channel association field.
_AF_PERP_FRAC = float(_p("AF_PERP_FRAC", 0.3))
_AF_MAX_DELTA_THETA = float(_p("AF_MAX_DELTA_THETA", 0.35))

_COCIRCULAR_WIDE_SIGMA_FRAC = float(_p("COCIRCULAR_WIDE_SIGMA_FRAC", 2.0))
_COCIRCULAR_WEIGHT = float(_p("COCIRCULAR_WEIGHT", 0.25))

_PEAK_RIDGE_SPACING_FRAC = float(_p("PEAK_RIDGE_SPACING_FRAC", 1.0))
# [FIX 6] perpendicular suppression half-width as a fraction of the median
# wavelength. Two contours closer than this perpendicular to the tangent
# are treated as the same contour; further apart they are parallel
# structure and both survive.
_SPACING_PERP_FRAC = float(_p("PEAK_SPACING_PERP_FRAC", 0.3))
_PEAK_MIN_INTEGRATION_FRAC = float(_p("PEAK_MIN_INTEGRATION_FRAC", 0.15))
# [BLANK-FRAME FIX] Absolute floor on summed integration. The relative
# threshold above normalises noise against itself, so a structure-free
# frame still passes it. Same construction as the gate-1 floor in
# perception/contours.py (CONTOUR_MIN_ABS_STRENGTH).
# 10.0 is ~2 orders of magnitude above the noise floor measured on a
# flat 256^2 grey plate (integration_max 18.27 there, but with a peak
# density of 300 -- real scenes reach 1e3-1e4). Tuned so the gate is
# strict on blanks and inert on real images.
# [FIX 4] retuned for mass-normalised integration: values now shrink by
# the beam weight total (order 5-30) relative to the un-normalised sum.
# Set from the measured blank/real ratio, not from the stale 18.27 figure.
_PEAK_MIN_INTEGRATION_ABS = float(_p("PEAK_MIN_INTEGRATION_ABS", 1e-3))
# [BORDER-MARGIN FIX] minimum fraction of each axis that must remain inside
# the artifact-band discard. Caps the margin on small frames / long-
# wavelength banks, where kernel_support + beam_reach would remove most of
# the image. 0.5 = keep at least half of each axis.
_FIELD_MIN_INTERIOR_FRAC = float(_p("FIELD_MIN_INTERIOR_FRAC", 0.5))
_ENDSTOP_MAX_RADIUS_FRAC = float(_p("ENDSTOP_MAX_RADIUS_FRAC", 2.0))
_CURVATURE_MAX_RADIUS_FRAC = float(_p("CURVATURE_MAX_RADIUS_FRAC", 2.0))

_FFT_KERNEL_THRESHOLD = min(int(_p("FFT_KERNEL_THRESHOLD", 31)), 7)  # production bank: FFT for 29+ px
_PADDING_TEST_IMAGE_SIZE = 128

# [BUG-03] Gabor bank cache keyed by scale
_GABOR_BANK_CACHE: Dict[float, GaborBank] = {}

# [BUG-04] Minimum peak-extraction margin; increase for wider kernels.
_PEAK_BORDER_MARGIN_PX = 16


# =============================================================================
# Front-end scale
# =============================================================================

def front_end_scale(h: int, w: int) -> float:
    """Image-diagonal ratio versus the reference. 1.0 at 256x256."""
    return math.hypot(float(h), float(w)) / max(
        _FRONT_END_REF_DIAGONAL_PX, 1e-6
    )


# =============================================================================
# V1: Gabor responses and divisive normalization
# =============================================================================

def _convolve_same(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Reflect-padded same-size convolution.

    Direct for small kernels (deterministic); fftconvolve for larger
    kernels, assuming single-threaded BLAS (set by the project
    convention at perception.filters import time).
    """
    if kernel.shape[0] <= _FFT_KERNEL_THRESHOLD:
        return nd_convolve(img, kernel, mode="reflect").astype(np.float32)
    # [FIX 3] scipy.signal.fftconvolve has no padding-mode argument, so
    # mode="same" ZERO-pads -- making large-kernel and small-kernel
    # responses non-comparable at the border. Pad manually so both paths
    # reflect-pad identically.
    H, W = img.shape
    k = int(kernel.shape[0])
    ph = pw = k // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="symmetric")
    return fftconvolve(padded, kernel, mode="valid").astype(np.float32)


def _border_margin_px(bank) -> int:
    """Suppress padding artifacts at extraction, not in convolution.

    The minimum is ~16 px; wider kernel support requires a wider margin.
    """
    return max(_PEAK_BORDER_MARGIN_PX, int(bank.kernels.shape[-1]) // 2)


def v1_responses(
    lum: np.ndarray,
    scale: float = 1.0,
    bank: Optional[GaborBank] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V1 simple, complex, and phase responses.

    simple  : (n_theta, n_sf, 2, H, W) float32 — even and odd
    complex : (n_theta, n_sf, H, W)    float32 — sqrt(even^2 + odd^2)
    phase   : (n_theta, n_sf, H, W)    float32 — atan2(odd, even)
    """
    if bank is None:
        # [BUG-03] Cache Gabor bank keyed by scale
        scale_key = float(scale)
        bank = _GABOR_BANK_CACHE.get(scale_key)
        if bank is None:
            bank = GaborBank.build_default(scale=scale)
            _GABOR_BANK_CACHE[scale_key] = bank
    n_theta, n_sf, n_phase = bank.kernels.shape[:3]
    H, W = lum.shape
    simple = np.zeros((n_theta, n_sf, n_phase, H, W), dtype=np.float32)
    for it in range(n_theta):
        for isf in range(n_sf):
            for ip in range(n_phase):
                simple[it, isf, ip] = _convolve_same(
                    lum, bank.kernels[it, isf, ip]
                )
    complex_ = np.sqrt(
        simple[:, :, 0].astype(np.float32) ** 2
        + simple[:, :, 1].astype(np.float32) ** 2
    )
    phase = np.arctan2(simple[:, :, 1], simple[:, :, 0])
    return simple, complex_.astype(np.float32), phase.astype(np.float32)


def _gaussian_kernel_1d(radius: int, sigma: float) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2.0 * np.float32(sigma) ** 2))
    k /= (k.sum() + np.float32(1e-12))
    return k.astype(np.float32)


def _blur_sep(img: np.ndarray, radius: int, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with reflect padding."""
    k = _gaussian_kernel_1d(radius, sigma)
    H, W = img.shape
    pad = np.pad(img, ((0, 0), (radius, radius)), mode="reflect")
    out = np.zeros_like(img)
    for i in range(k.shape[0]):
        w = k[i]
        out += w * pad[:, i:i + W]
    pad = np.pad(out, ((radius, radius), (0, 0)), mode="reflect")
    out2 = np.zeros_like(out)
    for i in range(k.shape[0]):
        w = k[i]
        out2 += w * pad[i:i + H, :]
    return out2.astype(np.float32)


def divisive_normalize(
    complex_: np.ndarray,
    sigma: Optional[float] = None,
    pool_radius: Optional[float] = None,
    pool_sigma: Optional[float] = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Divisive normalization: r / (sigma + pooled_summed_activity).

    Pool is a Gaussian blur of the sum over (theta, sf) of the complex
    magnitude, then broadcast back over the channel dimension. This
    follows the V1 convention that the divisor is the local total
    activity across channels, not the same channel alone. A channel-
    wise anisotropic pool is a future refinement.
    """
    sig = _DIVISIVE_SIGMA if sigma is None else float(sigma)
    pr = _DIVISIVE_POOL_RADIUS if pool_radius is None else float(pool_radius)
    ps = _DIVISIVE_POOL_SIGMA if pool_sigma is None else float(pool_sigma)
    # [FIX 5] the pool radius is a spatial scale, so it must follow the
    # image diagonal rather than being an absolute pixel count.
    r = int(round(pr * max(float(scale), 1e-6)))
    summed = complex_.sum(axis=(0, 1)).astype(np.float32)
    pooled = _blur_sep(summed, r, ps)
    denom = np.float32(sig) + pooled[None, None, :, :]
    return (complex_ / denom).astype(np.float32)


# =============================================================================
# V2: integration field and end-stop field
# =============================================================================

def _accumulate_shift(
    target: np.ndarray,
    source: np.ndarray,
    dx: int,
    dy: int,
    w: np.float32,
) -> None:
    """target[y+dy, x+dx] += w * source[y, x], in-place, zero-fill."""
    H, W = source.shape
    ty0 = max(0, dy); ty1 = min(H, H + dy)
    tx0 = max(0, dx); tx1 = min(W, W + dx)
    if ty0 >= ty1 or tx0 >= tx1:
        return
    sy0 = ty0 - dy; sy1 = ty1 - dy
    sx0 = tx0 - dx; sx1 = tx1 - dx
    target[ty0:ty1, tx0:tx1] += w * source[sy0:sy1, sx0:sx1]


def _beam_into(
    target: np.ndarray,
    channel: np.ndarray,
    theta_idx: int,
    wavelength: float,
    direction: int,
) -> float:
    """Accumulate the collinear beam sum into `target`.

    direction = +1 for forward along theta, -1 for backward.
    Integer-pixel shifts; the perpendicular offset is zero by
    construction (same channel), satisfying AF_PERP_FRAC < 0.3 *
    wavelength exactly for the same-channel case.
    """
    H, W = channel.shape
    theta = (theta_idx + 0.5) * math.pi / _GABOR_N_THETA
    ct = math.cos(theta) * direction
    st = math.sin(theta) * direction
    K = int(round(_AF_LONGITUDINAL_MULT * wavelength))
    sigma_along = _AF_LONGITUDINAL_MULT * wavelength / 2.0

    # [FIX 4] return the weight total so the caller can mass-normalise:
    # without it, coarse scales accumulate ~5x the weight of fine ones
    # and the sf_idx argmax is biased toward the coarse end.
    weight_total = 1.0
    target += channel  # k = 0 self-term

    if K < 1:
        return float(weight_total)

    for k in range(1, K + 1):
        dx = int(round(k * ct))
        dy = int(round(k * st))
        if dx == 0 and dy == 0:
            continue
        w = math.exp(-(k * k) / (2.0 * sigma_along * sigma_along))
        weight_total += w
        _accumulate_shift(target, channel, dx, dy, np.float32(w))
    return float(weight_total)


def integration_field(
    complex_: np.ndarray,
    sfs: np.ndarray,
    cocircular_wide: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """V2 integration and end-stop fields.

    Returns
    -------
    integration : (n_theta, n_sf, H, W) float32
        forward + backward collinear support, plus optional wide-
        context term.
    endstop     : (n_theta, n_sf, H, W) float32
        |forward - backward| normalized by forward + backward.
    """
    n_theta, n_sf, H, W = complex_.shape
    fwd = np.zeros_like(complex_)
    bwd = np.zeros_like(complex_)

    for it in range(n_theta):
        for isf in range(n_sf):
            ch = complex_[it, isf]
            # [BUG-03] Short-circuit zero-energy channels
            if ch.max() <= 0.0:
                continue
            sf = float(sfs[isf])
            wavelength = 1.0 / max(sf, 1e-6)
            wt = _beam_into(fwd[it, isf], ch, it, wavelength, +1)
            _beam_into(bwd[it, isf], ch, it, wavelength, -1)
            # [FIX 4] mass-normalise each channel so coarse and fine
            # channels share one scale.
            wtn = np.float32(max(wt, 1e-9))
            fwd[it, isf] /= wtn
            bwd[it, isf] /= wtn

    integration = (fwd + bwd).astype(np.float32)
    if cocircular_wide is not None:
        integration = (
            integration
            + np.float32(_COCIRCULAR_WEIGHT) * cocircular_wide
        ).astype(np.float32)

    denom = (fwd + bwd + np.float32(1e-6)).astype(np.float32)
    endstop = (np.abs(fwd - bwd) / denom).astype(np.float32)
    return integration, endstop


def cocircular_context(
    complex_: np.ndarray, wavelength_mean: float,
) -> np.ndarray:
    """Broad-context support: wide Gaussian blur of the summed complex.

    Documented approximation of the plan's cocircular support. The
    full pair computation (circle through two orientations with
    radius in [5, 200] and tangent error < 0.15) is O(N^2) over
    orientation pairs and is not implemented.

    Returns shape (1, 1, H, W); callers broadcast.
    """
    summed = complex_.sum(axis=(0, 1)).astype(np.float32)
    sigma = _COCIRCULAR_WIDE_SIGMA_FRAC * wavelength_mean
    r = int(math.ceil(3.0 * sigma))
    wide = _blur_sep(summed, r, sigma)
    return wide[None, None, :, :].astype(np.float32)


# =============================================================================
# V4: continuous curvature
# =============================================================================

def _continuous_orientation(integration: np.ndarray) -> np.ndarray:
    """Population-code continuous orientation from the integration field.

    Returns (H, W) float32 in [0, pi). Orientation is pi-periodic, so the
    circular mean is computed in doubled-angle space.
    """
    n_theta = integration.shape[0]
    summed = integration.sum(axis=1).astype(np.float32)
    thetas = ((np.arange(n_theta, dtype=np.float32) + 0.5)
              * math.pi / n_theta).astype(np.float32)
    c = np.cos(2.0 * thetas).astype(np.float32)
    sn = np.sin(2.0 * thetas).astype(np.float32)
    cos_sum = (summed * c[:, None, None]).sum(axis=0)
    sin_sum = (summed * sn[:, None, None]).sum(axis=0)
    two_theta = np.arctan2(sin_sum, cos_sum)
    theta = 0.5 * two_theta
    theta = np.where(theta < 0.0, theta + math.pi, theta)
    return theta.astype(np.float32)


def curvature_field(
    integration: np.ndarray,
    wavelength_mean: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Signed curvature and bend angle.

    [FIX 1] kappa = d(theta)/ds where s is arc length ALONG THE TANGENT --
    the derivative of the continuous orientation field in its own tangent
    direction. The previous version took the derivative PERPENDICULAR to
    the tangent, which is ~0 for a circle and exactly 0 for a line.

    Continuous (population-code) orientation replaces the argmax bin, so
    the finite difference is not quantised to 15-degree steps. [FIX 4/5]
    bend_angle uses wavelength_mean supplied by the caller from the actual
    sfs array rather than an unscaled constant.
    """
    H, W = integration.shape[2], integration.shape[3]
    theta_cont = _continuous_orientation(integration)

    tan_x = np.cos(theta_cont)
    tan_y = np.sin(theta_cont)
    eps = np.float32(3.0)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x2 = np.clip(xx + eps * tan_x, 0.0, np.float32(W - 1))
    y2 = np.clip(yy + eps * tan_y, 0.0, np.float32(H - 1))
    x2i = np.clip(np.round(x2).astype(np.int32), 0, W - 1)
    y2i = np.clip(np.round(y2).astype(np.int32), 0, H - 1)
    theta_2 = theta_cont[y2i, x2i]
    d_theta = theta_2 - theta_cont
    # wrap in doubled-angle space: orientation is pi-periodic
    d_theta = np.arctan2(np.sin(2.0 * d_theta),
                         np.cos(2.0 * d_theta)) / 2.0
    kappa = (d_theta / eps).astype(np.float32)
    bend = (np.abs(kappa) * np.float32(2.0 * wavelength_mean)).astype(
        np.float32
    )
    return kappa, bend

def _sample_bilinear(
    field: np.ndarray, x: np.ndarray, y: np.ndarray,
) -> np.ndarray:
    H, W = field.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x0c = np.clip(x0, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)
    v00 = field[y0c, x0c]
    v10 = field[y0c, x1]
    v01 = field[y1, x0c]
    v11 = field[y1, x1]
    return (
        v00 * (1.0 - wx) * (1.0 - wy)
        + v10 * wx * (1.0 - wy)
        + v01 * (1.0 - wx) * wy
        + v11 * wx * wy
    ).astype(np.float32)


def ridge_nms(
    integration: np.ndarray, min_integration: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Binary mask of ridge pixels and the summed intensity.

    Ridge pixel iff I(x, y) strictly exceeds both perpendicular
    samples at each of eps = 1 and 2 pixels, and exceeds the
    integration floor. Ties are rejected.
    """
    n_theta, n_sf, H, W = integration.shape
    I = integration.sum(axis=(0, 1)).astype(np.float32)
    flat = integration.reshape(n_theta * n_sf, H, W)
    flat_idx = flat.argmax(axis=0)
    theta_idx = (flat_idx // n_sf).astype(np.float32)
    theta_rad = ((theta_idx + 0.5) * math.pi / _GABOR_N_THETA).astype(
        np.float32
    )
    perp_x = -np.sin(theta_rad)
    perp_y = np.cos(theta_rad)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    mask = I > np.float32(min_integration)
    for distance in (1.0, 2.0):
        eps = np.float32(distance)
        I_p = _sample_bilinear(I, xx + eps * perp_x, yy + eps * perp_y)
        I_m = _sample_bilinear(I, xx - eps * perp_x, yy - eps * perp_y)
        mask &= (I > I_p) & (I > I_m)
    return mask, I


def _greedy_stride_nms(
    pixels: List[Tuple[float, int, int]], min_dist: float,
) -> List[Tuple[float, int, int]]:
    """Deterministic greedy NMS over a sorted pixel list.

    Caller must sort by (-value, y, x) beforehand. Accepts pixel if
    no already-accepted pixel is within Euclidean min_dist.
    """
    accepted: List[Tuple[float, int, int]] = []
    min_d2 = min_dist * min_dist
    for v, y, x in pixels:
        ok = True
        for _, ay, ax in accepted:
            dy = y - ay
            dx = x - ax
            if dy * dy + dx * dx < min_d2:
                ok = False
                break
        if ok:
            accepted.append((v, y, x))
    return accepted


def _local_maxima_2d(
    field: np.ndarray, radius: int,
) -> List[Tuple[int, int, float]]:
    """Local maxima of `field` with the given NMS radius.

    Deterministic: maximum_filter is single-threaded; ties broken by
    sorted (y, x) order.
    """
    size = 2 * max(1, radius) + 1
    maxf = maximum_filter(field, size=size, mode="reflect")
    ys, xs = np.where(field >= maxf)
    pairs = sorted(
        ((int(y), int(x), float(field[y, x]))
         for y, x in zip(ys.tolist(), xs.tolist())),
        key=lambda t: (-t[2], t[0], t[1]),
    )
    return pairs


def _directional_nms(
    samples: List[Tuple[float, int, int, float]],
    spacing_along: float,
    spacing_perp: float,
) -> List[Tuple[float, int, int, float]]:
    """[FIX 6] Suppress a sample only if an accepted one lies within
    spacing_along ALONG its tangent AND within spacing_perp perpendicular
    to it. Strength-ordered isotropic NMS lets the first of two parallel
    contours kill the second; this spares parallel structure.

    samples: (value, y, x, theta). Caller sorts by (-value, y, x, theta).
    """
    accepted: List[Tuple[float, int, int, float]] = []
    sa2 = spacing_along * spacing_along
    sp2 = spacing_perp * spacing_perp
    for v, y, x, th in samples:
        tx = math.cos(th)
        ty = math.sin(th)
        ok = True
        for _v, ay, ax, _ath in accepted:
            dx = x - ax
            dy = y - ay
            along = dx * tx + dy * ty
            perp = -dx * ty + dy * tx
            if along * along <= sa2 and perp * perp <= sp2:
                ok = False
                break
        if ok:
            accepted.append((v, y, x, th))
    return accepted


def _union_by_proximity(
    samples: List[Tuple[float, int, int]], min_dist: float,
) -> List[Tuple[float, int, int]]:
    samples = sorted(samples, key=lambda t: (-t[0], t[1], t[2]))
    return _greedy_stride_nms(samples, min_dist)


def _dominant_theta_rad(integration: np.ndarray, y: int, x: int) -> float:
    """Dominant orientation in radians at one pixel (channel argmax).

    integration is indexed [theta, sf, y, x], so the pixel slice is
    (n_theta, n_sf) -- argmax over theta first, then sf.
    """
    cell = integration[:, :, y, x]        # (n_theta, n_sf)
    it = int(np.argmax(cell.max(axis=1)))
    return (it + 0.5) * math.pi / integration.shape[0]


def extract_peaks(
    integration: np.ndarray,
    endstop: np.ndarray,
    curvature_signed: np.ndarray,
    bend_angle: np.ndarray,
    complex_: np.ndarray,
    phase: np.ndarray,
    sfs: np.ndarray,
    border_margin_px: int = 0,
) -> List[Dict]:
    """Peak extraction per A1.

    Union of ridge samples (greedy NMS at 1.0 * median_wavelength),
    end-stop maxima, and |curvature| maxima.

    Each peak is a dict with position, dominant theta and sf,
    signed curvature, bend angle, phase, end-stop flag, integration
    strength, coherence.
    """
    n_theta, n_sf, H, W = integration.shape
    wavelengths = 1.0 / np.maximum(sfs.astype(np.float32), 1e-6)
    median_wavelength = float(np.median(wavelengths))
    peak_spacing = _PEAK_RIDGE_SPACING_FRAC * median_wavelength
    endstop_radius = int(
        round(_ENDSTOP_MAX_RADIUS_FRAC * median_wavelength)
    )
    curv_radius = int(
        round(_CURVATURE_MAX_RADIUS_FRAC * median_wavelength)
    )

    I = integration.sum(axis=(0, 1)).astype(np.float32)
    I_max = float(I.max()) if I.size else 0.0
    # [BLANK-FRAME FIX] relative AND absolute. The relative term alone
    # normalises noise against itself; the absolute term keeps a
    # structure-free frame out (0 peaks).
    min_integration = max(
        _PEAK_MIN_INTEGRATION_FRAC * I_max,
        _PEAK_MIN_INTEGRATION_ABS,
    )
    mask, _ = ridge_nms(integration, min_integration)

    ys, xs = np.where(mask)
    ridge_pixels: List[Tuple[float, int, int]] = [
        (float(I[y, x]), int(y), int(x))
        for y, x in zip(ys.tolist(), xs.tolist())
    ]
    ridge_pixels.sort(key=lambda t: (-t[0], t[1], t[2]))
    # [BORDER-MARGIN FIX] drop the reflect-padding artifact band. Uses
    # >= so a pixel exactly on the margin boundary is kept.
    if border_margin_px > 0:
        # [BORDER-MARGIN FIX] cap so a long-wavelength bank cannot gut a
        # small frame: keep at least FIELD_MIN_INTERIOR_FRAC of each axis.
        _h, _w = int(H), int(W)
        _m_raw = int(border_margin_px)
        m = min(_m_raw, int(_FIELD_MIN_INTERIOR_FRAC * min(_h, _w)))
        m = max(0, m)
        ridge_pixels = [
            t for t in ridge_pixels
            if m <= t[1] < H - m and m <= t[2] < W - m
        ]
    # [FIX 6] direction-aware NMS: parallel contours within one
    # wavelength no longer suppress each other.
    _ridge_dn = [
        (v, int(y), int(x), _dominant_theta_rad(integration, int(y), int(x)))
        for (v, y, x) in ridge_pixels
    ]
    _ridge_dn.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
    ridge_samples = [
        (v, y, x)
        for (v, y, x, _th) in _directional_nms(
            _ridge_dn, peak_spacing,
            _SPACING_PERP_FRAC * median_wavelength)
    ]

    endstop_sum = endstop.sum(axis=(0, 1)).astype(np.float32)
    # [BLANK-FRAME FIX] an end-stop candidate must sit on real integration
    # energy. Without this, float noise fed through the beam sum produces an
    # end-stop signature on a structure-free frame and `v > 0.5` passes.
    endstop_candidates = [
        (v, y, x)
        for (y, x, v) in _local_maxima_2d(endstop_sum, endstop_radius)
        if v > 0.5 and I[y, x] >= min_integration
    ]
    # [BORDER-MARGIN FIX] same discard for the end-stop candidate set
    if border_margin_px > 0:
        # [BORDER-MARGIN FIX] cap so a long-wavelength bank cannot gut a
        # small frame: keep at least FIELD_MIN_INTERIOR_FRAC of each axis.
        _h, _w = int(H), int(W)
        _m_raw = int(border_margin_px)
        m = min(_m_raw, int(_FIELD_MIN_INTERIOR_FRAC * min(_h, _w)))
        m = max(0, m)
        endstop_candidates = [
            t for t in endstop_candidates
            if m <= t[1] < H - m and m <= t[2] < W - m
        ]
    _es_dn = [
        (v, int(y), int(x), _dominant_theta_rad(integration, int(y), int(x)))
        for (v, y, x) in endstop_candidates
    ]
    _es_dn.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
    endstop_peaks = [
        (v, y, x)
        for (v, y, x, _th) in _directional_nms(
            _es_dn, peak_spacing,
            _SPACING_PERP_FRAC * median_wavelength)
    ]

    curv_abs = np.abs(curvature_signed).astype(np.float32)
    # [BLANK-FRAME FIX] same energy requirement for curvature candidates.
    curv_candidates = [
        (v, y, x)
        for (y, x, v) in _local_maxima_2d(curv_abs, curv_radius)
        if v > 0.0 and I[y, x] >= min_integration
    ]
    # [BORDER-MARGIN FIX] same discard for the curvature candidate set
    if border_margin_px > 0:
        # [BORDER-MARGIN FIX] cap so a long-wavelength bank cannot gut a
        # small frame: keep at least FIELD_MIN_INTERIOR_FRAC of each axis.
        _h, _w = int(H), int(W)
        _m_raw = int(border_margin_px)
        m = min(_m_raw, int(_FIELD_MIN_INTERIOR_FRAC * min(_h, _w)))
        m = max(0, m)
        curv_candidates = [
            t for t in curv_candidates
            if m <= t[1] < H - m and m <= t[2] < W - m
        ]
    _cv_dn = [
        (v, int(y), int(x), _dominant_theta_rad(integration, int(y), int(x)))
        for (v, y, x) in curv_candidates
    ]
    _cv_dn.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
    curv_peaks = [
        (v, y, x)
        for (v, y, x, _th) in _directional_nms(
            _cv_dn, peak_spacing,
            _SPACING_PERP_FRAC * median_wavelength)
    ]

    # [FIX 6] the final union is directional too: same-location, different
    # source peaks must merge, parallel contours must not.
    _uni_dn = [
        (v, int(y), int(x), _dominant_theta_rad(integration, int(y), int(x)))
        for (v, y, x) in (ridge_samples + endstop_peaks + curv_peaks)
    ]
    _uni_dn.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
    all_peaks = [
        (v, y, x)
        for (v, y, x, _th) in _directional_nms(
            _uni_dn, peak_spacing,
            _SPACING_PERP_FRAC * median_wavelength)
    ]

    peaks: List[Dict] = []
    for v, y, x in all_peaks:
        best_val = -np.inf
        best_theta = 0
        best_sf_here = 0
        for isf in range(n_sf):
            row = integration[:, isf, y, x]
            it = int(np.argmax(row))
            val = float(row[it])
            if val > best_val:
                best_val = val
                best_theta = it
                best_sf_here = isf
        theta_rad = (best_theta + 0.5) * math.pi / _GABOR_N_THETA

        phase_val = float(phase[best_theta, best_sf_here, y, x])
        phase_bin = int(((phase_val + math.pi) / (2 * math.pi)) * 4) % 4

        endstop_val = float(endstop[best_theta, best_sf_here, y, x])
        endstop_flag = 1 if endstop_val > 0.5 else 0

        theta_row = integration[:, best_sf_here, y, x]
        total = float(theta_row.sum()) + 1e-6
        top = float(theta_row.max())
        coherence = top / total

        peaks.append({
            "x": int(x),
            "y": int(y),
            "theta": float(theta_rad),
            "theta_idx": int(best_theta),
            "sf_idx": int(best_sf_here),
            "sf": float(sfs[best_sf_here]),
            "signed_curvature": float(curvature_signed[y, x]),
            "bend_angle": float(bend_angle[y, x]),
            "phase": float(phase_val),
            "phase_bin": int(phase_bin),
            "endstop_flag": int(endstop_flag),
            "integration_strength": float(v),
            "coherence": float(coherence),
        })
    return peaks


# =============================================================================
# Convenience top-level
# =============================================================================

def field_from_luminance(
    lum: np.ndarray, scale: Optional[float] = None,
) -> Dict:
    """Run the full field pipeline on a luminance image.

    Returns a dict with: simple, complex, phase, integration,
    endstop, curvature_signed, bend_angle, peaks, scale, sfs.
    """
    if lum.dtype != np.float32:
        lum = lum.astype(np.float32)
    H, W = lum.shape
    if scale is None:
        scale = front_end_scale(H, W)

    # [BUG-03] Use cached Gabor bank
    scale_key = float(scale)
    bank = _GABOR_BANK_CACHE.get(scale_key)
    if bank is None:
        bank = GaborBank.build_default(scale=scale)
        _GABOR_BANK_CACHE[scale_key] = bank
    simple, complex_raw, phase = v1_responses(lum, scale=scale, bank=bank)
    complex_ = divisive_normalize(complex_raw, scale=scale)

    sfs = np.geomspace(
        _GABOR_SF_MIN / max(scale, 1e-6),
        _GABOR_SF_MAX / max(scale, 1e-6),
        _GABOR_N_SF,
    ).astype(np.float32)
    wavelength_mean = float(np.median(1.0 / np.maximum(sfs, 1e-6)))
    cocirc = cocircular_context(complex_, wavelength_mean)

    integration, endstop = integration_field(
        complex_, sfs, cocircular_wide=cocirc,
    )
    curvature_signed, bend_angle = curvature_field(
        integration, wavelength_mean)
    peaks = extract_peaks(
        integration, endstop, curvature_signed, bend_angle,
        complex_, phase, sfs,
        border_margin_px=_border_margin_px(bank),
    )
    return {
        "simple": simple,
        "complex": complex_,
        "phase": phase,
        "integration": integration,
        "endstop": endstop,
        "curvature_signed": curvature_signed,
        "bend_angle": bend_angle,
        "peaks": peaks,
        "scale": float(scale),
        "sfs": sfs.tolist(),
    }


# =============================================================================
# Diagnostics
# =============================================================================

def test_padding_schemes(
    size: int = _PADDING_TEST_IMAGE_SIZE,
) -> Dict:
    """Compare padding modes at the border.

    For each mode, convolve a random luminance image with one mid-
    scale Gabor kernel and report the ratio of mean |response| in the
    3-pixel border band to the mean |response| in the 3-pixel inset
    interior. A ratio near 1.0 means the padding does not manufacture
    border artifacts. This formalizes the 2026-10-10 front-end fix's
    finding that reflect padding did manufacture artifacts on the
    previous pipeline.

    fftconvolve is used only for large kernels and does not accept a
    padding-mode argument; the padding modes reported are those that
    ndimage.convolve accepts. If the probe kernel exceeds the FFT
    threshold the mode argument is ignored and the same value is
    reported for all three modes, which the caller can detect by
    inspecting `kernel_size`.
    """
    rng = np.random.default_rng(0)
    lum = rng.random((size, size), dtype=np.float32)
    bank = GaborBank.build_default(scale=front_end_scale(size, size))
    k = bank.kernels[3, 2, 0]
    kernel_size = int(k.shape[0])
    use_direct = kernel_size <= _FFT_KERNEL_THRESHOLD

    b = 3
    results: Dict[str, Dict[str, float]] = {}
    modes = ("reflect", "constant", "nearest") if use_direct else ("reflect",)
    for mode in modes:
        if use_direct:
            resp = nd_convolve(lum, k, mode=mode).astype(np.float32)
        else:
            resp = fftconvolve(lum, k, mode="same").astype(np.float32)
        border = np.concatenate([
            np.abs(resp[:b, :]).ravel(),
            np.abs(resp[-b:, :]).ravel(),
            np.abs(resp[:, :b]).ravel(),
            np.abs(resp[:, -b:]).ravel(),
        ])
        interior = np.abs(resp[b:-b, b:-b]).ravel()
        bm = float(border.mean()) if border.size else 0.0
        im = float(interior.mean()) if interior.size else 0.0
        results[mode] = {
            "border_mean": bm,
            "interior_mean": im,
            "ratio": bm / (im + 1e-9),
        }
    results["kernel_size"] = float(kernel_size)
    results["use_direct"] = float(1.0 if use_direct else 0.0)
    return results


def blank_frame_check(size: int = _PADDING_TEST_IMAGE_SIZE) -> Dict:
    """Blank-frame gate: uniform image yields 0 peaks."""
    # [BLANK-FRAME FIX] A blank scene is a uniform NONZERO grey plate, not
    # a zero frame: with all-zero luminance the V1 response is identically
    # zero and the gate tests nothing. The nonzero case is the one that
    # exposed the relative-threshold failure (300 peaks).
    lum = np.full((size, size), 0.5, dtype=np.float32)
    result = field_from_luminance(lum)
    return {
        "peak_count": len(result["peaks"]),
        "integration_max": float(result["integration"].max()),
    }


def determinism_check(size: int = 128, seed: int = 0) -> Dict:
    """Run field twice on the same input; report byte-identity."""
    rng = np.random.default_rng(seed)
    lum = rng.random((size, size), dtype=np.float32)
    a = field_from_luminance(lum)
    b = field_from_luminance(lum)
    return {
        "integration_identical": bool(
            np.array_equal(a["integration"], b["integration"])
        ),
        "complex_identical": bool(
            np.array_equal(a["complex"], b["complex"])
        ),
        "endstop_identical": bool(
            np.array_equal(a["endstop"], b["endstop"])
        ),
        "peaks_identical": a["peaks"] == b["peaks"],
    }


# =============================================================================
# __main__: diagnostic block
# =============================================================================

if __name__ == "__main__":
    import sys
    import time

    size = 128
    rng = np.random.default_rng(0)
    lum = rng.random((size, size), dtype=np.float32)

    print("=== determinism ===")
    print(determinism_check(size=size, seed=0))

    print("=== blank frame ===")
    print(blank_frame_check(size=size))

    print("=== padding schemes ===")
    for mode, stat in test_padding_schemes(size=size).items():
        print(f"  {mode}: {stat}")

    print("=== runtime (mean of 5) ===")
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = field_from_luminance(lum)
        times.append(time.perf_counter() - t0)
    times.sort()
    print(f"  mean={sum(times)/len(times):.3f}s  p95={times[-1]:.3f}s")

    print("=== peak count ===")
    r = field_from_luminance(lum)
    print(f"  n_peaks={len(r['peaks'])}")
    sys.exit(0)
