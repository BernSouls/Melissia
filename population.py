"""core/population.py -- population coding primitives.

THESIS (representation is the hard problem; matching is trivial afterwards):
a quantity is held as GRADED ACTIVITY over overlapping tuning curves, and
similarity is read as the OVERLAP of those activity vectors. Bins are retained
only as LOOKUP KEYS (for identity, indexing and the coarse graph). This is
robust to bin-edge wobble -- the failure mode of argmax binning, where a value
that sits just across a bin edge scores maximally far from its true neighbour.

Design decisions (mine, stated so they can be audited):
  * Tuning width. A population bump has standard deviation `sigma_bins`
    expressed in BIN units. The default is 1.0 bin, i.e. the tuning width is
    the front-end resolution: adjacent-bin overlap is high (exp(-0.5) ~ 0.61 at
    one bin) which is exactly what removes edge wobble. For orientation the
    filter bank has GABOR_N_THETA channels spanning [0, pi), so one channel = one
    bin and the width IS the filter bandwidth, not a tuned free parameter.
  * Circular axes use a WRAPPED Gaussian in bin units; no doubling trick is
    needed because the axis is already the pi-periodic bin odometer.
  * Activity vectors are non-negative and L1-normalised, so they are
    distributions; the scale (how much evidence) travels separately in `z`,
    never inside the shape vector.
   * Similarity is the CHANCE-CORRECTED overlap: Pearson correlation of the
     activity vectors (mean subtracted, i.e. against the uniform null), read as
     its non-negative part. Identical shapes -> 1.0; unrelated bumps (orthogonal
     or negatively correlated, as disjoint supports always are) -> 0.0; a
     uniform (uninformative) vector -> 0.0. This is the overlap reading, and it
     agrees with the fuzzy-Gaussian branch's "far apart -> ~0" behaviour.
  * Deterministic; bounded compute (n_bins adds per axis); no randomness, no
    iteration to convergence, no ODEs.

Nothing here imports the rest of the project, so it can sit under core/.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

DEFAULT_SIGMA_BINS = 1.0


@dataclass(frozen=True)
class PopAxis:
    """One tuning-curve axis of a population code.

    name      : stable identifier (documentation + diagnostics)
    n_bins    : number of tuning curves on the axis
    circular  : True if the axis wraps (last bin adjacent to first)
    sigma_bins: tuning width, in bins
    weight    : contribution to the combined similarity (>= 0)
    """
    name: str
    n_bins: int
    circular: bool = False
    sigma_bins: float = DEFAULT_SIGMA_BINS
    weight: float = 1.0

    def __post_init__(self):
        if self.n_bins <= 0:
            raise ValueError(f"PopAxis {self.name!r}: n_bins must be > 0")
        if self.sigma_bins <= 0.0:
            raise ValueError(f"PopAxis {self.name!r}: sigma_bins must be > 0")
        if self.weight < 0.0:
            raise ValueError(f"PopAxis {self.name!r}: weight must be >= 0")


def _wrapped_bin_distance(d: float, n_bins: int, circular: bool) -> float:
    d = abs(d)
    if circular and n_bins > 0:
        d = min(d, n_bins - d)
    return d


def soft_bump_bins(center_bins: float, axis: PopAxis) -> Tuple[float, ...]:
    """A normalised tuning bump centred at `center_bins` (a continuous bin index).

    Deterministic Gaussian profile; circular axes wrap. Returns an L1-normalised
    tuple of length `axis.n_bins`.
    """
    n = axis.n_bins
    s = axis.sigma_bins
    act = []
    for j in range(n):
        d = _wrapped_bin_distance(j - center_bins, n, axis.circular)
        act.append(math.exp(-0.5 * (d / s) ** 2))
    total = sum(act)
    if total <= 0.0:
        return tuple(1.0 / n for _ in range(n))
    return tuple(a / total for a in act)


def mixture_bins(center_bins_list: Sequence[float], axis: PopAxis,
                 weights: Optional[Sequence[float]] = None) -> Tuple[float, ...]:
    """A break-stable distribution: the normalised sum of member bumps.

    Used for ensembles (a chain has a DISTRIBUTION of member orientations, not a
    single mean). Weights default to uniform.
    """
    n = axis.n_bins
    acc = [0.0] * n
    if not center_bins_list:
        return tuple(1.0 / n for _ in range(n))
    if weights is None:
        weights = [1.0] * len(center_bins_list)
    for c, w in zip(center_bins_list, weights):
        bump = soft_bump_bins(c, axis)
        for j in range(n):
            acc[j] += float(w) * bump[j]
    total = sum(acc)
    if total <= 0.0:
        return tuple(1.0 / n for _ in range(n))
    return tuple(a / total for a in acc)


def centered_cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation of two activity vectors (chance correction = the
    uniform null). Returns 0.0 when either vector has ~no variance (uniform),
    i.e. carries no information -> neutral similarity (see axis_similarity)."""
    n = len(a)
    if n == 0 or len(b) != n:
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    num = 0.0
    va = 0.0
    vb = 0.0
    for x, y in zip(a, b):
        dx = x - ma
        dy = y - mb
        num += dx * dy
        va += dx * dx
        vb += dy * dy
    if va <= 1e-18 or vb <= 1e-18:
        return 0.0
    return max(-1.0, min(1.0, num / math.sqrt(va * vb)))


def axis_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Chance-corrected overlap of one axis, in [0, 1]: the non-negative part of
    the Pearson correlation of the activity vectors. Identical -> 1.0, unrelated
    (orthogonal or disjoint) -> 0.0."""
    c = centered_cosine(a, b)
    return c if c > 0.0 else 0.0


def population_similarity(
    pop_a: Sequence[Sequence[float]],
    pop_b: Sequence[Sequence[float]],
    axes: Sequence[PopAxis],
) -> float:
    """Weighted mean of per-axis chance-corrected overlaps.

    Returns 0.0 on layout mismatch (a population is only comparable to a
    population of the same class/layout), which the caller reads as
    "population not applicable" and falls back to bins.
    """
    if pop_a is None or pop_b is None:
        return 0.0
    if len(pop_a) != len(pop_b) or len(pop_a) != len(axes):
        return 0.0
    total = 0.0
    wsum = 0.0
    for axis, va, vb in zip(axes, pop_a, pop_b):
        if len(va) != axis.n_bins or len(vb) != axis.n_bins:
            return 0.0
        total += axis.weight * axis_similarity(va, vb)
        wsum += axis.weight
    if wsum <= 0.0:
        return 0.0
    return total / wsum


def validate_population(pop: Sequence[Sequence[float]],
                        axes: Sequence[PopAxis]) -> None:
    """Contract check: one vector per axis, each of length axis.n_bins,
    non-negative and finite. Raises ValueError on violation."""
    if len(pop) != len(axes):
        raise ValueError(
            f"population has {len(pop)} axes but {len(axes)} were declared")
    for axis, v in zip(axes, pop):
        if len(v) != axis.n_bins:
            raise ValueError(
                f"axis {axis.name!r}: vector length {len(v)} != n_bins "
                f"{axis.n_bins}")
        for a in v:
            if not math.isfinite(a) or a < 0.0:
                raise ValueError(
                    f"axis {axis.name!r}: activity must be finite and >= 0")
