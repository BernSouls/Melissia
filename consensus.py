"""matching/consensus.py -- multi-vote consensus with stored disagreement.

Phase 5 generalisation layer. A learned structural template is the CONSENSUS of
the positive exemplars it was induced from. The existing code stores the
consensus value and a single scalar consistency; this layer additionally stores,
per slot, the WEIGHTED VOTE SUPPORT and the DISAGREEMENT, so a downstream matcher
can read how contested each decision was rather than only how confident the
aggregate is.

Design decisions (mine, stated so they can be audited):

* **Vote, not average.** Categorical (identity/NAP) slots take the weighted
  MODE and are never averaged into an invented middle value. Graded linear slots
  take the weighted MEDOID (a real observed value nearest the weighted mean),
  never the mean itself, so no unsupported value is invented. Circular slots
  take the medoid nearest the weighted circular mean.

* **Support is agreement, disagreement is its complement.** For categorical and
  linear slots support is the share of vote weight within tolerance of the
  consensus (exact for identity, half a bin for linear). For circular slots it is
  the resultant length R (the concentration of the vote on the circle).
  disagreement = 1 - support. Both are in [0, 1].

* **No invented edges.** ``generalize_edges`` keeps an edge only when its
  support weight reaches ``min_support``; unsupported candidate edges are
  dropped, never emitted with a default weight. Every kept edge carries its
  support weight.

Pure and deterministic: ties break toward the smaller value, iteration sorts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SlotVote:
    value: float
    support: float        # weighted agreement with the consensus, in [0, 1]
    disagreement: float   # 1 - support, in [0, 1]
    n_weight: float       # total vote weight that reached this slot


@dataclass(frozen=True)
class VoteConsensus:
    values: Tuple[float, ...]
    support: Tuple[float, ...]
    disagreement: Tuple[float, ...]
    n_voters: int

    def mean_support(self) -> float:
        if not self.support:
            return 0.0
        return sum(self.support) / len(self.support)

    def mean_disagreement(self) -> float:
        if not self.disagreement:
            return 0.0
        return sum(self.disagreement) / len(self.disagreement)


def _weights(n: int, weights: Optional[Sequence[float]]) -> List[float]:
    if weights is None:
        return [1.0] * n
    out: List[float] = []
    for i in range(n):
        w = float(weights[i]) if i < len(weights) else 1.0
        out.append(w if w > 0.0 and math.isfinite(w) else 0.0)
    return out


def slot_vote(
    values: Sequence[float],
    *,
    weights: Optional[Sequence[float]] = None,
    identity: bool = False,
    cycle: Optional[float] = None,
) -> SlotVote:
    """Weighted vote for one slot. See module docstring for the semantics."""
    pairs = [(float(v), w) for v, w in zip(values, _weights(len(values), weights))
             if math.isfinite(float(v)) and w > 0.0]
    if not pairs:
        return SlotVote(0.0, 0.0, 1.0, 0.0)
    total_w = sum(w for _, w in pairs)

    if identity:
        # categorical weighted mode; ties -> smaller value
        counts: Dict[int, float] = {}
        for v, w in pairs:
            k = int(round(v))
            counts[k] = counts.get(k, 0.0) + w
        mode = min(
            sorted(counts),
            key=lambda k: (-counts[k], k),
        )
        support = counts[mode] / total_w
        return SlotVote(float(mode), support, 1.0 - support, total_w)

    if cycle is not None and cycle > 0.0:
        # circular: weighted resultant vector
        s = sum(w * math.sin(2.0 * math.pi * v / cycle) for v, w in pairs)
        c = sum(w * math.cos(2.0 * math.pi * v / cycle) for v, w in pairs)
        r = math.hypot(s, c) / total_w
        mean_ang = math.atan2(s, c) % (2.0 * math.pi)
        mean_val = (mean_ang / (2.0 * math.pi)) * cycle

        def _circ_dist(v: float) -> float:
            d = abs(v - mean_val) % cycle
            return min(d, cycle - d)

        medoid = min(pairs, key=lambda vw: (_circ_dist(vw[0]), vw[0]))[0]
        support = min(1.0, max(0.0, r))
        return SlotVote(float(medoid), support, 1.0 - support, total_w)

    # linear graded: weighted medoid nearest the weighted mean
    mean = sum(v * w for v, w in pairs) / total_w
    medoid = min(pairs, key=lambda vw: (abs(vw[0] - mean), vw[0]))[0]
    near = sum(w for v, w in pairs if abs(v - medoid) <= 0.5)
    support = near / total_w
    return SlotVote(float(medoid), support, 1.0 - support, total_w)


def vote_consensus(
    per_slot_values: Sequence[Sequence[float]],
    *,
    identity_slots: Iterable[int] = (),
    cycles: Optional[Mapping[int, float]] = None,
    weights: Optional[Sequence[float]] = None,
) -> VoteConsensus:
    """Multi-vote consensus over aligned exemplar slots.

    per_slot_values[i] is the list of observed values for slot i across the
    voters (an empty list marks a slot no voter filled -> value 0, support 0).
    """
    id_slots = set(identity_slots)
    cycles = dict(cycles or {})
    n_slots = len(per_slot_values)
    values: List[float] = []
    support: List[float] = []
    disagreement: List[float] = []
    for i in range(n_slots):
        sv = slot_vote(
            per_slot_values[i],
            weights=weights,
            identity=(i in id_slots),
            cycle=cycles.get(i),
        )
        values.append(sv.value)
        support.append(sv.support)
        disagreement.append(sv.disagreement)
    n_voters = len(weights) if weights is not None else max(
        (len(c) for c in per_slot_values), default=0)
    return VoteConsensus(
        values=tuple(values),
        support=tuple(support),
        disagreement=tuple(disagreement),
        n_voters=n_voters,
    )


def generalize_edges(
    support_weights: Mapping[object, float],
    min_support: float,
) -> Dict[object, float]:
    """Keep only edges whose support weight reaches `min_support`.

    `support_weights` maps a canonical edge key to its (normalised) support in
    [0, 1]. The result is the surviving subset with its weights, deterministic
    by sorted key. No edge is invented and no dropped edge is re-added with a
    default weight.
    """
    return {
        key: float(support_weights[key])
        for key in sorted(support_weights, key=lambda k: repr(k))
        if float(support_weights[key]) >= min_support
    }


def support_weights(
    edge_presence: Sequence[Mapping[object, float]],
    voter_weights: Optional[Sequence[float]] = None,
) -> Dict[object, float]:
    """Normalised support weight per candidate edge across voters.

    `edge_presence[e]` is voter e's weight for edge `e` (0 if absent), or 1.0
    if present. The support of an edge is the sum of voter weights that carried
    it, divided by the total voter weight, so it is a vote share in [0, 1].
    """
    n = len(edge_presence)
    w = _weights(n, voter_weights)
    total = sum(w)
    if total <= 0.0:
        return {}
    acc: Dict[object, float] = {}
    for e in range(n):
        for key, present in edge_presence[e].items():
            if present:
                acc[key] = acc.get(key, 0.0) + w[e]
    return {key: acc[key] / total for key in acc}
