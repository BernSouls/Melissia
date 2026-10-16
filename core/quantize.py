"""core/quantize.py -- pure bin/angle quantizers shared by the derived lenses.

Sources: primitives.py (Move 2), lenses.py (Move 3c) and
perception/lenses/base.py (Move 3.5, _q_linear_safe). Zero project imports
by contract.
"""
from __future__ import annotations

import math


_MAX_BINS = 20  # Generic cap for signature quantization


def q_linear(value: float, n_bins: int, lo: float = 0.0, hi: float = 1.0) -> int:
    """[C-ARITY.1 / C1] Canonical quantizer. Quantize value into
    [0, n_bins-1] over [lo, hi].

    Deterministic: floor of normalized value, clamped to [0, n_bins-1].
    Bin k means value in [lo + k*(hi-lo)/n_bins, lo + (k+1)*(hi-lo)/n_bins).

    [V9 Track 2] Prerequisite for the arity work: C-ARITY.2 (Contour
    curvature-profile slots) and C-ARITY.3 (ObjectShape symmetry/aspect/
    identity histogram) both quantize through this function.

    Scope note on the other two wrappers, so nobody "unifies" them by
    mistake:
      * `_q_unsigned` / `_q_signed` below already reproduce the drafts'
        bodies line-for-line (verified), so they are NOT rewritten here --
        rewriting them is a no-op with a fingerprint risk and zero gain.
      * `_q_arc_len` already derives its step from
        P.ORIENTATION_SEGMENT_ARC_MAX / P.ORIENTATION_SEGMENT_N_ARC
        (:325-327). The draft's B.26a calls that rewrite a behaviour
        change and gates it; measured, it is arithmetically identical on
        every input domain (only the exception type on +/-inf differs).
        Left alone -- it is already canonical and the arity work does not
        need it.
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    if not math.isfinite(value):
        raise ValueError(f"cannot quantize non-finite value: {value}")
    if hi <= lo:
        raise ValueError(f"hi must be > lo, got lo={lo}, hi={hi}")
    v = max(lo, min(hi, value))
    norm = (v - lo) / (hi - lo)
    q = int(math.floor(norm * n_bins))
    return min(n_bins - 1, max(0, q))


def _q_unsigned(value: float, step: float, max_bins: int = _MAX_BINS) -> int:
    """Quantize a non-negative scalar into [0, max_bins].

    Deterministic: floor of value / step, clamped to [0, max_bins].
    Raises on non-finite input. Raises on non-positive step.
    """
    if step <= 0.0:
        raise ValueError(f"quantize step must be positive, got {step}")
    if not math.isfinite(value):
        raise ValueError(f"cannot quantize non-finite value: {value}")
    v = max(0.0, value)
    q = int(math.floor(v / step))
    if q > max_bins:
        return max_bins
    return q


def _q_signed(value: float, max_bins: int = 8) -> int:
    """Quantize a signed value in [-1, 1] into [0, max_bins - 1].

    Values outside [-1, 1] are clamped. The mapping is (v + 1) / 2 * max_bins,
    floored, and capped at max_bins - 1. Deterministic.
    """
    if not math.isfinite(value):
        raise ValueError(f"cannot quantize non-finite value: {value}")
    if max_bins <= 0:
        raise ValueError(f"max_bins must be positive, got {max_bins}")
    v = max(-1.0, min(1.0, value))
    norm = (v + 1.0) * 0.5
    q = int(math.floor(norm * max_bins))
    if q >= max_bins:
        return max_bins - 1
    return q


def _q_linear_safe(v: float, n_bins: int, lo: float, hi: float) -> int:
    """Wrapper for q_linear that enforces half-open [lo, hi) contract."""
    v = max(lo, min(hi, float(v)))
    return int(min(n_bins - 1, max(0, q_linear(v, n_bins, lo, hi))))


def _aang(a: float, b: float) -> float:
    """Undirected angle difference in [0, pi/2] (orientations are axial)."""
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


# ---------------------------------------------------------------------------
# [AUDIT #7/#11] SINGLE SOURCE OF TRUTH for the orientation convention.
# ---------------------------------------------------------------------------
def carrier_to_tangent(theta_rad: float) -> float:
    """Gabor/bank theta is the CARRIER (normal) axis; the edge tangent is
    theta + pi/2. Every lens that reasons about edge DIRECTION (B5, B7, B17)
    must call this exactly once. Do not re-derive it locally."""
    return (theta_rad + math.pi / 2.0) % math.pi


def _opens_toward(det, vec, n_theta: int) -> bool:
    """Does this EndStopped terminator face along `vec`?

    [AUDIT #7 FIX] `ends` (D1 slot 2): 0 = both flanks empty (short bar/dot,
    directionless -> accept), 1 = open toward -a, 2 = open toward +a, where
    a = the bar's long axis = (-sin th, cos th) in (x, y).
    """
    if len(det.sigma.quantized) < 3:
        return True
    ends = int(det.sigma.quantized[2])
    if ends == 0:
        return True
    th = (det.sigma.quantized[0] + 0.5) * math.pi / n_theta
    a = (-math.sin(th), math.cos(th))  # +a in (x, y)
    return (ends == 2) == ((a[0] * vec[0] + a[1] * vec[1]) > 0)


def theta_to_bin(it: int, nt: int, n_theta: int) -> float:
    """[AUDIT #6/#13 FIX] Map a bank theta INDEX to its signature bin."""
    return float((it * n_theta) // nt)
