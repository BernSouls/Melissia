"""matching/fuzzy.py -- unified fuzzy graded structural matcher (Phase 5).

One entry point that composes the Phase 5 layers:

  * node similarity       : the population-overlap similarity already produced
                            by ``TypedSignature.similarity`` (chance-corrected
                            Pearson overlap against the uniform null);
  * global alignment      : ``matching/alignment`` global-consistency
                            alignment seeded from the approximate Hungarian,
                            with an explicit unmatched cost and symmetric
                            coverage;
  * calibration           : ``matching/calibration`` null-calibrated,
                            floor-free, background-weighted scoring;
  * consensus             : ``matching/consensus`` multi-vote consensus with
                            stored disagreement (``fuzzy_consensus``);
  * search                : ``matching/search`` best-first extension with
                            anytime budgets (available for template growth).

The pairwise result is a ``FuzzyMatch`` carrying every component separately, so
a caller (or an ablation) can read or recombine them; ``score`` is the documented
default combination.

Pure and deterministic. No corpus state, no scoring tuned against an outcome.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import parameters as P
from core.graph_types import Subgraph
from core.types import UNDIRECTED_EDGE_TYPES

from .calibration import NullModel, mean_pairwise_similarity
from .structural_similarity import (
    _lookup_edge_params,
    _pair_edge_sim_typed,
    _sig_sim,
)

SignatureProvider = object  # (node_id) -> Optional[Tuple[lens_type, TypedSignature]]


@dataclass(frozen=True)
class FuzzyMatch:
    """Components of a fuzzy graded match between two subgraphs."""
    mapping: Dict[int, int]
    quality: Dict[int, float]
    node_coverage: float              # symmetric Dice count 2|M|/(|A|+|B|)
    node_coverage_weighted: float     # similarity-weighted Dice
    node_coverage_calibrated: float   # weighted Dice calibrated above chance
    node_quality_mean: float
    chance: float                     # null expected weighted coverage
    systematicity: float              # conjunctive typed-edge preservation
    fwd_edge_cov: float
    bwd_edge_cov: float
    edge_coverage: float              # 0.5*(fwd+bwd)
    edge_param_mean: float            # mean graded edge-param sim over kept edges
    unmatched_a: Tuple[int, ...]
    unmatched_b: Tuple[int, ...]
    score: float
    budget_exhausted: bool = False


def _edge_param_mean(
    sg_a: Subgraph,
    sg_b: Subgraph,
    mapping: Mapping[int, int],
    use_confidence: bool,
) -> float:
    if not sg_a.edges:
        return 1.0  # vacuous: no edge evidence to contradict
    b_und = {
        (min(s, d), max(s, d), et)
        for (s, d, et) in sg_b.edges if et in UNDIRECTED_EDGE_TYPES
    }
    b_dir = {(s, d, et) for (s, d, et) in sg_b.edges if et not in UNDIRECTED_EDGE_TYPES}
    total = 0.0
    n = 0
    for (s, d, et) in sorted(sg_a.edges, key=lambda t: (t[0], t[1], getattr(t[2], "value", 0))):
        ms = mapping.get(s)
        md = mapping.get(d)
        if ms is None or md is None:
            continue
        if et in UNDIRECTED_EDGE_TYPES:
            lo, hi = (ms, md) if ms < md else (md, ms)
            present = (lo, hi, et) in b_und
        else:
            present = (ms, md, et) in b_dir
        if not present:
            continue
        ep_a = _lookup_edge_params(sg_a.edge_params, s, d, et)
        ep_b = _lookup_edge_params(sg_b.edge_params, ms, md, et)
        total += _pair_edge_sim_typed(
            s, d, et, ep_a, ms, md, et, ep_b, use_confidence,
        )
        n += 1
    return total / n if n else 1.0


def fuzzy_match(
    sg_a: Subgraph,
    sg_b: Subgraph,
    sig_provider,
    *,
    unmatched_cost: Optional[float] = None,
    systematicity_weight: Optional[float] = None,
    rounds: Optional[int] = None,
    max_pair_evals: Optional[int] = None,
    edge_weight: Optional[float] = None,
    use_confidence: bool = False,
) -> FuzzyMatch:
    """Fuzzy graded match of `sg_a` onto `sg_b`.

    Node similarity is ``_sig_sim`` (which routes through the
    population-overlap path when both signatures carry a population). The
    alignment is global-consistency refinement seeded from the approximate
    Hungarian; node coverage is calibrated against the null expected coverage
    (the mean pairwise similarity), and the relational multiplier has NO 0.85
    floor.
    """
    if unmatched_cost is None:
        unmatched_cost = float(getattr(P, "MATCH_UNMATCHED_COST", 0.0))
    if systematicity_weight is None:
        systematicity_weight = float(getattr(P, "MATCH_SYSTEMATICITY_WEIGHT", 0.5))
    if rounds is None:
        rounds = int(getattr(P, "MATCH_ALIGN_ROUNDS", 4))
    if max_pair_evals is None:
        max_pair_evals = getattr(P, "MATCH_ALIGN_MAX_PAIR_EVALS", 200000)
    if edge_weight is None:
        edge_weight = float(getattr(P, "MATCH_EDGE_WEIGHT", 0.5))

    nodes_a = sorted(sg_a.nodes)
    nodes_b = sorted(sg_b.nodes)

    # The similarity matrix is built once here so the null (mean pairwise
    # similarity) and the alignment share it. Indexed [i][j] over the sorted
    # node lists.
    sigs_a = [(sig_provider(u) or (None, None))[1] for u in nodes_a]
    sigs_b = [(sig_provider(v) or (None, None))[1] for v in nodes_b]
    matrix: List[List[float]] = [
        [_sig_sim(sa, sb) for sb in sigs_b] for sa in sigs_a
    ]

    chance = mean_pairwise_similarity(matrix)

    from .alignment import global_consistency_alignment
    al = global_consistency_alignment(
        nodes_a,
        nodes_b,
        None,
        edges_a=sg_a.edges,
        edges_b=sg_b.edges,
        unmatched_cost=unmatched_cost,
        systematicity_weight=systematicity_weight,
        rounds=rounds,
        max_pair_evals=max_pair_evals,
        sim=matrix,
    )

    ntot = len(nodes_a) + len(nodes_b)
    weighted = 2.0 * sum(al.quality.values()) / ntot if ntot else 0.0
    node_cal = NullModel(chance).calibrate(weighted)
    quality_mean = (
        sum(al.quality.values()) / len(al.quality) if al.quality else 0.0
    )
    edge_param_mean = _edge_param_mean(sg_a, sg_b, al.mapping, use_confidence)
    edge_term = al.edge_coverage() * edge_param_mean
    score = node_cal * ((1.0 - edge_weight) + edge_weight * edge_term)

    return FuzzyMatch(
        mapping=dict(al.mapping),
        quality=dict(al.quality),
        node_coverage=al.node_coverage,
        node_coverage_weighted=weighted,
        node_coverage_calibrated=node_cal,
        node_quality_mean=quality_mean,
        chance=chance,
        systematicity=al.systematicity,
        fwd_edge_cov=al.fwd_edge_cov,
        bwd_edge_cov=al.bwd_edge_cov,
        edge_coverage=al.edge_coverage(),
        edge_param_mean=edge_param_mean,
        unmatched_a=al.unmatched_a,
        unmatched_b=al.unmatched_b,
        score=score,
        budget_exhausted=al.budget_exhausted,
    )


def fuzzy_consensus(
    reference: Subgraph,
    exemplars: Sequence[Subgraph],
    sig_provider,
    *,
    identity_slots: Sequence[int] = (),
    cycles: Optional[Mapping[int, float]] = None,
    voter_weights: Optional[Sequence[float]] = None,
    min_support: Optional[float] = None,
):
    """Generalise a template from exemplars aligned onto a reference.

    Each exemplar is fuzzy-matched onto the reference; for every reference node
    the matched exemplar's quantized slot values become one vote. Returns
    ``(per_node_consensus, edge_support)`` where ``per_node_consensus`` maps a
    reference node to a ``VoteConsensus`` and ``edge_support`` maps each
    reference edge key to its vote share, filtered by ``min_support`` so that no
    edge is invented.
    """
    from .consensus import (
        generalize_edges,
        support_weights,
        vote_consensus,
    )

    if min_support is None:
        min_support = float(getattr(P, "MATCH_MIN_CONSENSUS_SUPPORT", 0.5))

    ref_nodes = sorted(reference.nodes)
    slot_values: Dict[int, List[List[float]]] = {n: [] for n in ref_nodes}
    edge_presence: List[Dict[object, float]] = []

    for ex in exemplars:
        fm = fuzzy_match(ex, reference, sig_provider)
        # fm.mapping: exemplar node -> reference node. Invert to reference -> ex.
        by_ref: Dict[int, int] = {rv: ev for ev, rv in fm.mapping.items()}
        for rv in ref_nodes:
            info = sig_provider(by_ref[rv]) if rv in by_ref else None
            sig = info[1] if info else None
            slot_values[rv].append(list(sig.quantized) if sig else [])
        # Which reference edges did this exemplar preserve?
        present: Dict[object, float] = {}
        for (s, d, et) in reference.edges:
            es, ed = by_ref.get(s), by_ref.get(d)
            if es is None or ed is None:
                continue
            if et in UNDIRECTED_EDGE_TYPES:
                want = (min(es, ed), max(es, ed), et)
                have = {(min(a, b), max(a, b), c) for (a, b, c) in ex.edges}
                present[(s, d, et)] = 1.0 if want in have else 0.0
            else:
                present[(s, d, et)] = 1.0 if (es, ed, et) in set(ex.edges) else 0.0
        edge_presence.append(present)

    per_node = {}
    for rv in ref_nodes:
        cols = slot_values[rv]
        n_slots = max((len(c) for c in cols), default=0)
        rows = [
            [c[i] for c in cols if i < len(c)]
            for i in range(n_slots)
        ]
        per_node[rv] = vote_consensus(
            rows,
            identity_slots=identity_slots,
            cycles=cycles,
            weights=voter_weights,
        )

    support = support_weights(edge_presence, voter_weights)
    kept_edges = generalize_edges(support, min_support)
    return per_node, kept_edges
