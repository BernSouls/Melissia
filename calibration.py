"""matching/calibration.py -- null-calibrated, background-calibrated scoring.

Phase 5 scoring layer. Three ideas, each with a stated null model:

1. **Null-calibrated mass.** A raw match fraction is only evidence to the
   extent it exceeds what an uninformative correspondence would produce. The
   null is a random pairing: its expected similarity-weighted coverage is the
   mean pairwise similarity of the candidate matrix (``NullModel.from_matrix``)
   or, for a pure count coverage, the cardinality ratio ``min/max``
   (``NullModel.from_sizes``). Calibration maps an observed coverage to
   ``(observed - chance) / (1 - chance)`` clamped to ``[0, 1]``. The node
   similarity itself is already chance-corrected (``population_similarity`` is
   the Pearson overlap of activity vectors against the uniform null), so the
   two calibrations compose rather than compete.

2. **No 0.85 conjunction floor.** A conjunctive relational multiplier must be
   able to reach 0 when no relation is preserved. ``conjunction`` returns the
   raw consistency; the historical floor is available only by explicitly
   passing it, so it can never leak back in as a hidden default.

3. **Background-calibrated per-class scoring.** A lens class that matches by
   chance often must not be credited as much. ``background_weights`` gives each
   class an inverse-frequency (idf-style) weight from its background rate.

All operators are pure, total and deterministic. No corpus state, no tuning
against a score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

from parameters import EPSILON

#: Smoothing mass added to both numerator and denominator of the background
#: rate. One pseudo-observation: a class seen zero times still gets a finite
#: (maximal) weight, and a class seen every draw gets a finite (minimal) one.
BACKGROUND_SMOOTHING = 1.0


def chance_fraction(n_a: int, n_b: int) -> float:
    """Chance COUNT coverage of a bijection between sides of sizes n_a, n_b.

    A bijection covers exactly min(n_a, n_b) of the max(n_a, n_b) pairs, so the
    uninformative count coverage fraction is min/max. 0 when either side is
    empty. NOTE: for weighted coverage use ``NullModel.from_matrix``; this
    count null is 1.0 whenever the two sides are the same size.
    """
    if n_a <= 0 or n_b <= 0:
        return 0.0
    return min(n_a, n_b) / float(max(n_a, n_b))


def mean_pairwise_similarity(matrix: Sequence[Sequence[float]]) -> float:
    """Null expected similarity-weighted coverage: the mean matrix entry.

    Under a random pairing, the expected per-pair similarity is the mean of the
    whole candidate matrix, so this is the coverage a random correspondence
    would achieve. Returns 0.0 for an empty matrix.
    """
    total = 0.0
    n = 0
    for row in matrix:
        for v in row:
            total += float(v)
            n += 1
    if n == 0:
        return 0.0
    return total / n


def calibrate_fraction(observed: float, chance: float) -> float:
    """Map an observed [0,1] fraction to evidence above chance, in [0,1].

    (observed - chance) / (1 - chance), clamped. chance >= 1 means the null
    already explains everything, so the observation carries no evidence -> 0.0.
    """
    if chance >= 1.0:
        return 0.0
    v = (float(observed) - float(chance)) / (1.0 - float(chance))
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


@dataclass(frozen=True)
class NullModel:
    """A null coverage a score is calibrated against."""
    chance: float

    @classmethod
    def from_sizes(cls, n_a: int, n_b: int) -> "NullModel":
        return cls(chance_fraction(n_a, n_b))

    @classmethod
    def from_matrix(cls, matrix: Sequence[Sequence[float]]) -> "NullModel":
        return cls(mean_pairwise_similarity(matrix))

    def calibrate(self, observed: float) -> float:
        return calibrate_fraction(observed, self.chance)

    def chance_mass(self, total: float) -> float:
        """Expected matched mass under the null for a side of mass `total`."""
        return self.chance * float(total)


def conjunction(relational_consistency: float, floor: float = 0.0) -> float:
    """Conjunctive relational multiplier with an explicit floor.

    Default floor 0.0 removes the historical 0.85 floor: an uninformative
    relation set yields 0, not 0.85. Pass floor=0.85 only to reproduce the old
    behaviour deliberately.
    """
    v = float(relational_consistency)
    if v < floor:
        return float(floor)
    return 1.0 if v > 1.0 else (0.0 if v < 0.0 else v)


def evidence_score_calibrated(
    node_coverage: float,
    relational_consistency: float,
    chance: float,
    floor: float = 0.0,
) -> float:
    """Conjunctive evidence score calibrated against the null.

    score = calibrate(node_coverage, chance) * conjunction(relational_consistency)

    With ``floor=0.0`` (default) an uninformative relation set zeroes the score
    instead of inheriting a 0.85 prior.
    """
    return NullModel(chance).calibrate(node_coverage) * conjunction(
        relational_consistency, floor=floor)


def background_weights(
    counts: Mapping[str, float],
    draws: float,
    smoothing: float = BACKGROUND_SMOOTHING,
) -> Dict[str, float]:
    """Per-class inverse-frequency weights from background match counts.

    weight(c) = log((draws + s) / (count(c) + s)), clamped to >= 0 and
    normalised so the maximum weight is 1.0. A class that matched on every
    draw gets weight ~0 (it carries no discriminative evidence); a class that
    never matched by chance gets the maximal weight 1.0.
    """
    s = float(smoothing)
    if draws < 0.0:
        draws = 0.0
    raw: Dict[str, float] = {}
    for cls, c in counts.items():
        c = float(c)
        if c < 0.0:
            c = 0.0
        w = math.log((draws + s) / (c + s))
        raw[cls] = w if w > 0.0 else 0.0
    wmax = max(raw.values()) if raw else 0.0
    if wmax <= 0.0:
        return {cls: 0.0 for cls in raw}
    return {cls: raw[cls] / wmax for cls in raw}


def weighted_class_score(
    class_scores: Mapping[str, float],
    weights: Mapping[str, float],
    default_weight: float = 1.0,
) -> float:
    """Background-weighted mean of per-class scores.

    A class absent from `weights` (no background estimate) keeps
    `default_weight`, so an unscored class is neither credited nor suppressed
    by a missing estimate.
    """
    if not class_scores:
        return 0.0
    total = 0.0
    wsum = 0.0
    for cls in sorted(class_scores):
        w = float(weights.get(cls, default_weight))
        total += w * float(class_scores[cls])
        wsum += w
    if wsum <= EPSILON:
        return 0.0
    return total / wsum
