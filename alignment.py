"""matching/alignment.py -- global-consistency graph alignment.

The correspondence layer of the fuzzy graded matcher (Phase 5).

Given two node sets, a node-similarity function (the *population-overlap*
similarity produced by ``TypedSignature.similarity``) and the typed edge sets,
this module produces a bijective correspondence that maximises not just local
node similarity but GLOBAL CONSISTENCY: neighbouring nodes that are connected in
one graph are preferred to map onto neighbouring connected nodes in the other
(systematicity). It is seeded from the existing approximate Hungarian
(``_max_weight_bipartite_matching``) and refined by bounded fixed-point
re-assignment.

Decisions (mine, stated so they can be audited):

* **Explicit unmatched cost.** A pair is only ever matched when its final
  quality is at least ``unmatched_cost``. Leaving a node unmatched is therefore
  a decision with a stated price rather than an accident of ``> 0`` pruning.
  (Implemented as a quality floor; the existing greedy solver already refuses a
  zero-weight partner, and a floor expresses the same "not worth matching"
  choice at O(n*m) instead of a squared augmented matrix.)

* **Symmetric coverage.** Coverage is the Dice overlap ``2|M| / (|A|+|B|)``, so
  a mapping that explains both sides is rewarded and a mapping that covers one
  side by leaving the other entirely unmatched is not. Forward/backward edge
  coverage are reported separately.

* **Systematicity.** The conjunctive (min) fraction of typed edges preserved in
  each direction. Min, not mean: a clean A side must not compensate a
  contradicted B side.

* **Determinism.** Every iteration sorts its inputs; ties break on
  ``(weight desc, row asc, col asc)``; the returned mapping is a sorted dict.

Pure, stateless, budgeted, no randomness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from core.types import UNDIRECTED_EDGE_TYPES
from matching.structural_similarity import _max_weight_bipartite_matching

#: (u, v) -> similarity in [0, 1]. 0.0 means "not comparable".
SimFn = Callable[[int, int], float]
#: node -> list of (neighbour, edge_type, outward). `outward` True when the
#: node is the edge's source.
Incidence = Dict[int, List[Tuple[int, object, bool]]]
Edge = Tuple[int, int, object]


@dataclass(frozen=True)
class Alignment:
    """Result of a global-consistency alignment.

    mapping      : A-node -> B-node, sorted by A-node for determinism.
    quality      : A-node -> final pair quality in [0, 1] (= sim * relation).
    systematicity: conjunctive typed-edge preservation, in [0, 1].
    fwd_edge_cov : fraction of A edges preserved in B, in [0, 1].
    bwd_edge_cov : fraction of B edges preserved in A, in [0, 1].
    node_coverage: symmetric Dice 2|M| / (|A|+|B|), in [0, 1].
    unmatched_a  : sorted A nodes left unmatched.
    unmatched_b  : sorted B nodes left unmatched.
    rounds_used  : refinement sweeps actually performed.
    budget_exhausted: True when the pair-evaluation budget stopped refinement.
    """
    mapping: Dict[int, int]
    quality: Dict[int, float]
    systematicity: float
    fwd_edge_cov: float
    bwd_edge_cov: float
    node_coverage: float
    unmatched_a: Tuple[int, ...] = ()
    unmatched_b: Tuple[int, ...] = ()
    rounds_used: int = 0
    budget_exhausted: bool = False

    def edge_coverage(self) -> float:
        return 0.5 * (self.fwd_edge_cov + self.bwd_edge_cov)


def build_incidence(edges: Iterable[Edge]) -> Incidence:
    """node -> [(neighbour, et, outward)] from a typed edge iterable."""
    inc: Incidence = {}
    for e in sorted(edges, key=lambda t: (min(t[0], t[1]), max(t[0], t[1]),
                                          getattr(t[2], "value", 0))):
        s, d, et = e
        inc.setdefault(s, []).append((d, et, True))
        inc.setdefault(d, []).append((s, et, False))
    return inc


def _edge_indexes(edges: Iterable[Edge]) -> Tuple[Set[Edge], Set[Edge]]:
    directed: Set[Edge] = set()
    undirected: Set[Edge] = set()
    for s, d, et in edges:
        if et in UNDIRECTED_EDGE_TYPES:
            undirected.add((min(s, d), max(s, d), et))
        else:
            directed.add((s, d, et))
    return directed, undirected


def _preserved_edge(
    a_src: int, a_dst: int, et: object,
    mapping: Dict[int, int],
    b_dir: Set[Edge], b_und: Set[Edge],
) -> bool:
    b_src = mapping.get(a_src)
    b_dst = mapping.get(a_dst)
    if b_src is None or b_dst is None:
        return False
    if et in UNDIRECTED_EDGE_TYPES:
        lo, hi = (b_src, b_dst) if b_src < b_dst else (b_dst, b_src)
        return (lo, hi, et) in b_und
    return (b_src, b_dst, et) in b_dir


def systematicity(
    mapping: Dict[int, int],
    edges_a: Sequence[Edge],
    edges_b: Sequence[Edge],
) -> Tuple[float, float, float]:
    """Return (conjunctive, forward, backward) typed-edge preservation.

    conjunctive = min(forward, backward), with an edgeless side read as 1.0
    (vacuous), matching the codebase's conjunction convention.
    """
    a_dir, a_und = _edge_indexes(edges_a)
    b_dir, b_und = _edge_indexes(edges_b)
    rev = {v: u for u, v in mapping.items()}

    if edges_a:
        fwd = sum(
            1 for (s, d, et) in edges_a
            if _preserved_edge(s, d, et, mapping, b_dir, b_und)
        ) / float(len(edges_a))
    else:
        fwd = 1.0
    if edges_b:
        bwd = sum(
            1 for (s, d, et) in edges_b
            if _preserved_edge(s, d, et, rev, a_dir, a_und)
        ) / float(len(edges_b))
    else:
        bwd = 1.0
    return min(fwd, bwd), fwd, bwd


def _relational_agreement(
    u: int,
    v: int,
    mapping: Dict[int, int],
    inc_a: Incidence,
    inc_b: Incidence,
    b_dir: Set[Edge],
    b_und: Set[Edge],
    a_dir: Set[Edge],
    a_und: Set[Edge],
    rev: Dict[int, int],
) -> float:
    """Conjunctive mapped-context preservation for pair (u, v).

    Fraction of u's typed relations whose images under `mapping` exist at v,
    min the same from v's side. Relation-free nodes are 1.0 (identified by
    signature alone). Missing relation-partners do not earn occlusion credit
    here: this operator has no geometry, so it stays conservative.
    """
    edges_u = inc_a.get(u)
    if edges_u:
        kept = 0.0
        for w, et, outward in edges_u:
            mw = mapping.get(w)
            if mw is None:
                continue
            if et in UNDIRECTED_EDGE_TYPES:
                lo, hi = (v, mw) if v < mw else (mw, v)
                if (lo, hi, et) in b_und:
                    kept += 1.0
            else:
                want = (v, mw, et) if outward else (mw, v, et)
                if want in b_dir:
                    kept += 1.0
        agree_u = kept / len(edges_u)
    else:
        agree_u = 1.0

    edges_v = inc_b.get(v)
    if edges_v:
        kept = 0.0
        counted = 0
        for x, et, outward in edges_v:
            ux = rev.get(x)
            if ux is None:
                continue
            counted += 1
            if et in UNDIRECTED_EDGE_TYPES:
                lo, hi = (u, ux) if u < ux else (ux, u)
                if (lo, hi, et) in a_und:
                    kept += 1.0
            else:
                want = (ux, u, et) if outward else (u, ux, et)
                if want in a_dir:
                    kept += 1.0
        agree_v = kept / counted if counted else 1.0
    else:
        agree_v = 1.0
    return min(agree_u, agree_v)


def _assign(sim: List[List[float]],
            nodes_a: List[int],
            nodes_b: List[int],
            rel_weight: Optional[Callable[[int, int], float]] = None,
            beta: float = 0.0) -> Dict[int, int]:
    """One bipartite assignment. When rel_weight is given, the matrix is
    ``sim * (1 - beta + beta * rel)`` (relational modulation)."""
    if rel_weight is None:
        w = sim
    else:
        w = [
            [sim[i][j] * (1.0 - beta + beta * rel_weight(i, j))
             for j in range(len(nodes_b))]
            for i in range(len(nodes_a))
        ]
    pairs = _max_weight_bipartite_matching(w)
    m: Dict[int, int] = {}
    for i, j in pairs:
        if w[i][j] > 0.0:
            m[nodes_a[i]] = nodes_b[j]
    return m


def global_consistency_alignment(
    nodes_a: Sequence[int],
    nodes_b: Sequence[int],
    sim_fn: SimFn,
    edges_a: Sequence[Edge] = (),
    edges_b: Sequence[Edge] = (),
    *,
    unmatched_cost: float = 0.0,
    systematicity_weight: float = 0.5,
    rounds: int = 6,
    max_pair_evals: Optional[int] = None,
    sim: Optional[Sequence[Sequence[float]]] = None,
) -> Alignment:
    """Global-consistency alignment seeded from the approximate Hungarian.

    Steps:
      1. Build the node-similarity matrix (or use a supplied `sim`, indexed
         ``[i][j]`` over the SORTED node lists); prune pairs below
         `unmatched_cost`.
      2. Seed the correspondence with the approximate Hungarian on it.
      3. Refine: re-assign on ``sim * (1-beta) + sim*beta*rel_agreement`` until
         the mapping stabilises, the round budget is spent, or the pair-eval
         budget is exhausted (anytime: the last stable mapping is returned).
      4. Keep only pairs whose final quality (sim * relational agreement) is at
         least `unmatched_cost`.

    `unmatched_cost == 0.0` recovers the classic "any positive similarity may
    match" behaviour; raising it makes unmatched an explicit, priced decision.
    """
    nodes_a = sorted(set(nodes_a))
    nodes_b = sorted(set(nodes_b))
    empty = Alignment({}, {}, 1.0 if not edges_a and not edges_b else 0.0,
                      1.0, 1.0, 0.0,
                      tuple(nodes_a), tuple(nodes_b))
    if not nodes_a or not nodes_b:
        return empty

    # 1. similarity matrix (supplied or built), pruned by the unmatched cost.
    pair_evals = 0
    if sim is None:
        sim = [[0.0] * len(nodes_b) for _ in nodes_a]
        for i, u in enumerate(nodes_a):
            for j, v in enumerate(nodes_b):
                sim[i][j] = float(sim_fn(u, v))
                pair_evals += 1
    matrix: List[List[float]] = [
        [(0.0 if float(sim[i][j]) < unmatched_cost else float(sim[i][j]))
         for j in range(len(nodes_b))]
        for i in range(len(nodes_a))
    ]

    if not any(any(row) for row in matrix):
        return empty

    inc_a = build_incidence(edges_a)
    inc_b = build_incidence(edges_b)
    a_dir, a_und = _edge_indexes(edges_a)
    b_dir, b_und = _edge_indexes(edges_b)
    has_edges = bool(edges_a) or bool(edges_b)

    mapping = _assign(matrix, nodes_a, nodes_b)

    rounds_used = 0
    budget_exhausted = False
    if has_edges and rounds > 1:
        for _ in range(max(0, rounds - 1)):
            if max_pair_evals is not None and pair_evals >= max_pair_evals:
                budget_exhausted = True
                break
            rev = {v: u for u, v in mapping.items()}

            def rel_weight(i: int, j: int,
                           _rev=rev) -> float:
                nonlocal pair_evals
                pair_evals += 1
                return _relational_agreement(
                    nodes_a[i], nodes_b[j], mapping,
                    inc_a, inc_b, b_dir, b_und, a_dir, a_und, _rev,
                )

            new_mapping = _assign(
                matrix, nodes_a, nodes_b, rel_weight,
                beta=float(systematicity_weight),
            )
            rounds_used += 1
            if new_mapping == mapping:
                break
            mapping = new_mapping

    # 2. Final quality (sim * relational agreement) and the unmatched floor.
    rev_final = {v: u for u, v in mapping.items()}
    quality: Dict[int, float] = {}
    filtered: Dict[int, int] = {}
    idx_a = {n: i for i, n in enumerate(nodes_a)}
    idx_b = {n: j for j, n in enumerate(nodes_b)}
    for u, v in sorted(mapping.items()):
        s = matrix[idx_a[u]][idx_b[v]]
        if has_edges:
            rel = _relational_agreement(
                u, v, mapping, inc_a, inc_b, b_dir, b_und, a_dir, a_und,
                rev_final,
            )
        else:
            rel = 1.0
        q = s * (1.0 - systematicity_weight + systematicity_weight * rel)
        if q < unmatched_cost:
            continue
        filtered[u] = v
        quality[u] = q

    sys_conj, fwd, bwd = systematicity(filtered, edges_a, edges_b)
    n_match = len(filtered)
    node_cov = (2.0 * n_match) / float(len(nodes_a) + len(nodes_b))
    unmatched_a = tuple(u for u in nodes_a if u not in filtered)
    unmatched_b = tuple(v for v in nodes_b if v not in set(filtered.values()))

    return Alignment(
        mapping=dict(sorted(filtered.items())),
        quality=quality,
        systematicity=sys_conj,
        fwd_edge_cov=fwd,
        bwd_edge_cov=bwd,
        node_coverage=node_cov,
        unmatched_a=unmatched_a,
        unmatched_b=unmatched_b,
        rounds_used=rounds_used,
        budget_exhausted=budget_exhausted,
    )
