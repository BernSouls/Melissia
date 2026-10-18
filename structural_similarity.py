"""
structural_similarity.py — Pure Structural Comparison Operator (§1.5, §4.12, §5.3.2, §6.4.1).

Stateless mathematical operator over already-extracted graphs and neighborhoods.
Evaluates:
  1. Node overlap, cardinality, lens types, and graded signature alignment.
  2. Edge counts, binder relation channels, and topological isomorphism.
  3. Exemplar-guided structural template intersection and materialization.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np

from core.hashing import clamp  # noqa: F401
from core.registries import (  # noqa: F401
    CIRCULAR_CYCLE_LENGTHS,
    CONSTITUTIONAL_RANGES,
)
from core.schema import (  # noqa: F401
    CIRCULAR_EDGE_CYCLE_LENGTHS,
    CONSTITUTIONAL_EDGE_RANGES,
    EDGE_IDENTITY_SLOTS,
    EdgeParams,
    EdgeParamsMixture,
    UNKNOWN,
    coarsen_edge_params_for_wl,
    get_compatible_types,
    polymorphic_edge_similarity,
    reverse_edge_params,
)
from parameters import EPSILON  # noqa: F401
from sensory_floor import TypedSignature
from core.graph_types import Subgraph, anchor_positions_for
from core.types import (
    EdgeType,
    UNDIRECTED_EDGE_TYPES,
)
import parameters as P

# Signature provider: (node_id) -> Optional[Tuple[lens_type: str, TypedSignature]]
SignatureProvider = Callable[[int], Optional[Tuple[str, TypedSignature]]]

# Parameterized weights with fallbacks
W_SIZE = getattr(P, "STRUCT_SIM_W_SIZE", 0.3)
W_TYPE = getattr(P, "STRUCT_SIM_W_TYPE", 0.3)
W_ALIGN = getattr(P, "STRUCT_SIM_W_ALIGN", 0.4)
DEFAULT_ALPHA = getattr(P, "STRUCT_SIM_ALPHA", 0.5)


# =============================================================================
# [S5] RELATIONAL EDGE PARAMETER LOOKUP & SIMILARITY BRIDGES
# =============================================================================

def _lookup_edge_params(params: Optional[Dict], s: int, d: int, et: EdgeType) -> Optional[Any]:
    """Robust 3-way key lookup covering direct, canonical, and reversed undirected conventions."""
    if not params:
        return None
    ep = params.get((s, d, et))
    if ep is None and et in UNDIRECTED_EDGE_TYPES:
        ep = params.get((min(s, d), max(s, d), et))
        if ep is None:
            ep = params.get((max(s, d), min(s, d), et))
    return ep


def _edge_class_name(et: EdgeType, ep: Optional[Any]) -> str:
    """Resolve an edge's schema class name: live params win, else EDGE_<ENUM>."""
    if ep is not None:
        cls = getattr(ep, "class_", None)
        if cls:
            return str(cls)
    et_name = et.name if hasattr(et, 'name') else str(et)
    return "EDGE_" + et_name


def _b_key_index(params_b: Optional[Dict[Tuple[int, int, EdgeType], Any]]):
    """[PERF] Endpoint-pair index over params_b keys.

    Replaces the per-call `set(params_b.keys())` full scan in
    _best_cross_type_match (O(calls x |keys|) -> O(calls) dict lookups).
    Iteration order is the dict's insertion order -- identical to what
    `for _k in set_b_keys:` visited before; membership of the resulting
    candidate list only depends on the key SET, and candidates are sorted
    canonically downstream, so output is unchanged. Memoised by object
    identity + length; dicts mutated in place between calls with the same
    length would be a TOCTOU we deliberately do not guard (params dicts
    are write-once at subgraph build time)."""
    if not params_b:
        return {}, {}
    exact: Dict[Tuple[int, int], list] = {}
    canon: Dict[Tuple[int, int], list] = {}
    for _k in params_b:
        if not (isinstance(_k, tuple) and len(_k) == 3):
            continue
        _s, _d, _et = _k
        if not isinstance(_et, EdgeType):
            continue
        exact.setdefault((_s, _d), []).append(_k)
        canon.setdefault((min(_s, _d), max(_s, _d)), []).append(_k)
    return exact, canon


def _get_b_key_index(params_b):
    """[PERF] Build the endpoint-pair index for this call.

    [SAFETY] This deliberately rebuilds the index on every call rather than
    memoising it. A (id(params_b), len(params_b)) cache was tried and is
    removed: CPython reuses the address of a freed dict, so a later dict of
    the same length allocated at the same address silently received the
    earlier dict's index. Verified reproducible with a direct probe (the
    second dict's lookup returned the first dict's keys).

    The win that matters does not come from the cache. _best_cross_type_match
    previously rebuilt set(params_b.keys()) once per call and then scanned
    every key for every candidate edge; the index is still built once per call
    here, and the per-candidate scan becomes dict lookups. That is the same
    asymptotic improvement, with no stale state.
    """
    return _b_key_index(params_b)


def _best_cross_type_match(
    src: int,
    dst: int,
    et: EdgeType,
    mapped_src: int,
    mapped_dst: int,
    params_a: Optional[Dict[Tuple[int, int, EdgeType], Any]],
    params_b: Optional[Dict[Tuple[int, int, EdgeType], Any]],
    use_confidence: bool = False,
) -> float:
    """[Stage 2] Best polymorphic partner for (src,dst,et) among B-edges
    spanning (mapped_src, mapped_dst) whose class is identical OR declared
    compatible via get_compatible_types. Same-type fast path preserves the
    exact legacy behavior (canonical-key lookup + orientation-aware
    reversal)."""
    if params_b is None:
        params_b = {}
    exact_b, canon_b = _get_b_key_index(params_b)
    cls_a_hint = None
    if params_a is not None:
        for _k in ((src, dst, et), (min(src, dst), max(src, dst), et),
                   (max(src, dst), min(src, dst), et)):
            _hit = params_a.get(_k)
            if _hit is not None:
                cls_a_hint = getattr(_hit, "class_", None)
                break
    if cls_a_hint is None:
        cls_a_hint = "EDGE_" + et.name

    lo, hi = (mapped_src, mapped_dst) if mapped_src <= mapped_dst else (mapped_dst, mapped_src)
    compat = get_compatible_types(str(cls_a_hint))
    seen_pairs = set()
    candidates: List[Tuple[int, int, EdgeType]] = []
    for _group in (exact_b.get((mapped_src, mapped_dst), ()),
                   exact_b.get((lo, hi), ()),
                   canon_b.get((lo, hi), ())):
        for _k in _group:
            if _k in seen_pairs:
                continue
            seen_pairs.add(_k)
            _s, _d, _et = _k
            if (_s, _d) != (mapped_src, mapped_dst) and (_s, _d) != (lo, hi):
                continue
            _cls_b = _edge_class_name(_et, params_b.get(_k))
            if _et == et or _cls_b in compat:
                candidates.append((_s, _d, _et))
    # Fallback: B-side has no params dict entry at all — test the direct
    # same-type edge so missing-params pairs still score the neutral term.
    if not candidates:
        candidates.append((mapped_src, mapped_dst, et))

    use_conf = bool(getattr(P, "EDGE_MATCH_USE_CONFIDENCE", use_confidence))
    best = 0.0
    for _ck in sorted(candidates, key=lambda t: (t[2].name, t[0], t[1])):
        _p_b = _lookup_edge_params(params_b, _ck[0], _ck[1], _ck[2])
        _cls_b = _edge_class_name(_ck[2], _p_b)
        _same = (_ck[2] == et)
        if not _same and _cls_b not in get_compatible_types(str(cls_a_hint)):
            continue
        _p_a = _lookup_edge_params(params_a, src, dst, et)
        if _p_a is None and _cls_b in get_compatible_types(str(cls_a_hint)):
            # Cross-type with missing A-side params: neutral handling.
            base_neutral = float(getattr(P, "EDGE_PARAM_ONESIDED_SIM", 0.5))
            prior = float(getattr(P, "EDGE_CROSS_FAMILY_PRIOR", 0.90))
            _sim = base_neutral * prior
        else:
            _sim = _pair_edge_sim_typed(
                src, dst, et, _p_a, _ck[0], _ck[1], _ck[2], _p_b,
                use_confidence=use_conf,
            )
        if _sim > best:
            best = _sim
    return float(best)


def _pair_edge_sim_typed(
    a1: int, a2: int, et_a: EdgeType, ep_a: Optional[Any],
    b1: int, b2: int, et_b: EdgeType, ep_b: Optional[Any],
    use_confidence: bool = False,
) -> float:
    """[Stage 2] Pairwise edge similarity supporting compatible cross-type
    matching. Same-type (or compatible) pairs score via polymorphic
    projection; missing params score the neutral one-sided term
    (× cross-family prior when the types differ)."""
    class_1 = _edge_class_name(et_a, ep_a)
    class_2 = _edge_class_name(et_b, ep_b)
    if et_a == et_b or class_2 in get_compatible_types(class_1):
        if ep_a is None or ep_b is None:
            base_neutral = float(getattr(P, "EDGE_PARAM_ONESIDED_SIM", 0.5))
            prior = 1.0 if et_a == et_b else float(getattr(P, "EDGE_CROSS_FAMILY_PRIOR", 0.90))
            return float(base_neutral * prior)
        _p_b = ep_b
        if et_a in UNDIRECTED_EDGE_TYPES and (a1 < a2) != (b1 < b2):
            _p_b = reverse_edge_params(_p_b) if not isinstance(_p_b, EdgeParamsMixture) else _p_b.flipped_copy()
        use_conf = bool(getattr(P, "EDGE_MATCH_USE_CONFIDENCE", use_confidence))
        return float(polymorphic_edge_similarity(ep_a, _p_b, use_confidence=use_conf))
    return 0.0


def edge_params_similarity(a: Optional[Any], b: Optional[Any], flipped: bool = False) -> float:
    """Evaluate relational parameter similarity, flipping endpoint conventions when required.

    [Stage 2] Delegates to polymorphic_edge_similarity so compatible
    cross-schema pairs score the shared-subspace projection instead of 0.0.
    Signature preserved (flipped= kwarg) for existing callers.
    """
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return float(getattr(P, "EDGE_PARAM_ONESIDED_SIM", 0.5))
    if flipped:
        a = a.flipped_copy() if isinstance(a, EdgeParamsMixture) else reverse_edge_params(a)
    use_conf = bool(getattr(P, "EDGE_MATCH_USE_CONFIDENCE", False))
    return float(polymorphic_edge_similarity(a, b, use_confidence=use_conf))


def _pair_edge_sim(a1: int, a2: int, b1: int, b2: int, et: EdgeType, p_a: Optional[Dict], p_b: Optional[Dict]) -> float:
    """Evaluate parameter agreement between edge (a1, a2) in graph A and (b1, b2) in graph B."""
    ep_a = _lookup_edge_params(p_a, a1, a2, et)
    ep_b = _lookup_edge_params(p_b, b1, b2, et)
    flipped = (et in UNDIRECTED_EDGE_TYPES) and ((a1 < a2) != (b1 < b2))
    return edge_params_similarity(ep_a, ep_b, flipped=flipped)


# =============================================================================
# [PASS 25] V4 BOUNDARY CONFORMATION & TURNING FUNCTION OPERATORS
# =============================================================================

def _resample_turn_sequence(turns: Sequence[float], target_n: int) -> np.ndarray:
    """Arc-length resample a periodic turn sequence using continuous circle trigonometry."""
    n = len(turns)
    if n == target_n:
        return np.asarray(turns, dtype=np.float64)
    orig_s = np.linspace(0.0, 1.0, n, endpoint=False)
    target_s = np.linspace(0.0, 1.0, target_n, endpoint=False)
    sin_t = np.sin(turns)
    cos_t = np.cos(turns)
    orig_s_ext = np.concatenate([orig_s, [1.0]])
    sin_ext = np.concatenate([sin_t, [sin_t[0]]])
    cos_ext = np.concatenate([cos_t, [cos_t[0]]])
    new_sin = np.interp(target_s, orig_s_ext, sin_ext)
    new_cos = np.interp(target_s, orig_s_ext, cos_ext)
    return np.arctan2(new_sin, new_cos)


def turning_function_distance(
    turns_a: Sequence[float],
    turns_b: Sequence[float],
    closed: bool = True,
    max_length_ratio: float = 0.35,
) -> Tuple[float, List[Tuple[int, int]]]:
    """Computes V4 boundary conformation distance between two turn sequences.
    Closed contours: circular shift matching with direction invariance (clockwise / counter-clockwise).
    Open contours: Sakoe-Chiba banded Dynamic Time Warping with prefix/suffix recovery.
    """
    n_a, n_b = len(turns_a), len(turns_b)
    if n_a == 0 or n_b == 0:
        return float("inf"), []

    # Bounded length guardrail: allows small contour comparisons while rejecting major segment mismatches
    max_len = max(n_a, n_b)
    allowed_diff = max(2, int(max_len * max_length_ratio))
    if abs(n_a - n_b) > allowed_diff:
        return float("inf"), []

    if closed:
        N = max(n_a, n_b)
        A = _resample_turn_sequence(turns_a, N)
        B = _resample_turn_sequence(turns_b, N)
        best_dist = float("inf")
        best_shift = 0
        best_reflected = False

        # [PERF] Batched circular-shift search. The old loop paid one
        # np.roll + one np.mean dispatch per shift (O(N^2) allocations).
        # Here the full pairwise squared-difference matrix
        # W[j, k] = wrapped(A[j] - B[k]) ** 2 is built once per winding
        # view; every shift s is the diagonal sample sel[j, s] =
        # W[j, (j - s) % N], reduced with a C-ordered contiguous sum so the
        # accumulation order matches the original np.mean exactly. Verified
        # bit-identical (0 ULP) to the roll-loop for all N in [2, 70) over
        # random turn sequences, both forward and reflected views.
        # Tie-break semantics unchanged.
        js = np.arange(N)[:, None]
        ks = (js - np.arange(N)[None, :]) % N  # ks[j, s] = (j - s) % N
        # Evaluate both forward and reflected (opposite winding direction) views
        for target_seq, reflected in ((B, False), (-B[::-1], True)):
            sq = ((A[:, None] - target_seq[None, :] + math.pi)
                  % (2.0 * math.pi) - math.pi) ** 2
            dists = np.ascontiguousarray(sq[js, ks].T).sum(axis=1) / N
            order = np.argsort(dists, kind="stable")
            for s in order.tolist():
                d = float(dists[s])
                # Deterministic tie-breaking on float distance
                if d < best_dist - 1e-9 or (abs(d - best_dist) <= 1e-9 and (s, int(reflected)) < (best_shift, int(best_reflected))):
                    best_dist = d
                    best_shift = s
                    best_reflected = reflected

        alignment: List[Tuple[int, int]] = []
        for i in range(n_a):
            target_idx = int(round((i / n_a) * N)) % N
            if best_reflected:
                mapped_N = (N - 1 - ((target_idx - best_shift) % N)) % N
            else:
                mapped_N = (target_idx + best_shift) % N
            j = min(n_b - 1, int(round((mapped_N / N) * n_b)))
            alignment.append((i, j))

        return math.sqrt(best_dist), alignment

    else:
        # Sakoe-Chiba Banded DTW for open contours
        def _run_dtw(seq_a: Sequence[float], seq_b: Sequence[float]) -> Tuple[float, List[Tuple[int, int]]]:
            na, nb = len(seq_a), len(seq_b)
            window = max(abs(na - nb) + 2, int(max(na, nb) * 0.25))
            dtw = np.full((na + 1, nb + 1), np.inf, dtype=np.float64)
            dtw[0, 0] = 0.0
            for i in range(1, na + 1):
                j_start = max(1, i - window)
                j_end = min(nb, i + window)
                for j in range(j_start, j_end + 1):
                    d_theta = (seq_a[i - 1] - seq_b[j - 1] + math.pi) % (2.0 * math.pi) - math.pi
                    cost = d_theta ** 2
                    dtw[i, j] = cost + min(dtw[i - 1, j - 1], dtw[i - 1, j], dtw[i, j - 1])

            if math.isinf(dtw[na, nb]):
                return float("inf"), []

            # Deterministic backtracking to (0, 0)
            align: List[Tuple[int, int]] = []
            curr_i, curr_j = na, nb
            while curr_i > 0 or curr_j > 0:
                if curr_i > 0 and curr_j > 0:
                    diag = dtw[curr_i - 1, curr_j - 1]
                    up = dtw[curr_i - 1, curr_j]
                    left = dtw[curr_i, curr_j - 1]
                    m = min(diag, up, left)
                    # Diagonal match takes priority on ties
                    if diag == m:
                        curr_i -= 1
                        curr_j -= 1
                        align.append((curr_i, curr_j))
                    elif up == m:
                        curr_i -= 1
                        align.append((curr_i, curr_j))
                    else:
                        curr_j -= 1
                        align.append((curr_i, curr_j))
                elif curr_i > 0:
                    curr_i -= 1
                    align.append((curr_i, 0))
                else:
                    curr_j -= 1
                    align.append((0, curr_j))

            align.reverse()
            rms = math.sqrt(dtw[na, nb] / max(len(align), 1))
            return rms, align

        dist_fwd, align_fwd = _run_dtw(turns_a, turns_b)
        rev_b = [-v for v in reversed(turns_b)]
        dist_rev, align_rev = _run_dtw(turns_a, rev_b)
        if dist_rev < dist_fwd - 1e-9:
            mapped_align = [(i, (n_b - 1 - j)) for (i, j) in align_rev]
            return dist_rev, mapped_align
        return dist_fwd, align_fwd


# =============================================================================
# 1. NODE DIMENSION HELPERS
# =============================================================================

def turning_function_similarity(
    turns_a: Sequence[float],
    turns_b: Sequence[float],
    closed: bool = True,
    sigma: float = 0.45,
) -> float:
    """Computes Gaussian boundary conformation similarity in [0, 1]."""
    dist, _ = turning_function_distance(turns_a, turns_b, closed=closed)
    if math.isinf(dist):
        return 0.0
    return float(math.exp(-0.5 * (dist / sigma) ** 2))


# =============================================================================
# CONSENSUS EDGE PARAMETER SYNTHESIS
# =============================================================================

def build_consensus_edge_params(
    params_list: Sequence[EdgeParams],
    consensus_identity_threshold: float = 0.40,
) -> Optional[EdgeParams]:
    """Constructs consensus edge parameters across positive template instances.
    Categorical slots use MODE with deterministic tie-breaking. Graded slots use MEDOID.
    """
    if not params_list:
        return None

    ref = params_list[0]
    same_class = [
        p for p in params_list
        if p.class_ == ref.class_ and len(p.quantized) == len(ref.quantized)
    ]
    if not same_class:
        return None

    n_slots = len(ref.quantized)
    id_slots = (
        set(ref.identity_slots) if ref.identity_slots is not None
        else set(EDGE_IDENTITY_SLOTS.get(ref.class_, ()))
    )
    cycles = CIRCULAR_EDGE_CYCLE_LENGTHS.get(ref.class_, {})
    consensus_q: List[float] = []

    for i in range(n_slots):
        vals = [
            p.quantized[i] for p in same_class
            if p.quantized[i] != UNKNOWN and math.isfinite(p.quantized[i])
        ]
        if not vals:
            consensus_q.append(UNKNOWN)
            continue

        if i in id_slots:
            # Identity Slot (NAP): Categorical MODE with deterministic tie-breaking
            counts: Dict[int, int] = defaultdict(int)
            for v in vals:
                counts[int(round(v))] += 1

            # Deterministic sort: highest frequency wins; lowest integer breaks ties
            sorted_candidates = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            top_val, top_count = sorted_candidates[0]

            if top_count / len(vals) >= consensus_identity_threshold:
                consensus_q.append(float(top_val))
            else:
                consensus_q.append(UNKNOWN)
        else:
            # Graded Slot: Metric Property
            sorted_vals = sorted(vals)
            if i in cycles:
                # Circular metric slot
                cycle = float(cycles[i])
                angles = [(2.0 * math.pi * v) / cycle for v in vals]
                s = sum(math.sin(a) for a in angles)
                c = sum(math.cos(a) for a in angles)
                r = math.hypot(s, c)
                # If angular dispersion is nearly uniform, mark as UNKNOWN
                if r / len(vals) < 0.25:
                    consensus_q.append(UNKNOWN)
                else:
                    mean_ang = math.atan2(s, c) % (2.0 * math.pi)
                    circ_mean_val = (mean_ang / (2.0 * math.pi)) * cycle

                    def _circ_dist(v: float) -> float:
                        d = abs(v - circ_mean_val) % cycle
                        return min(d, cycle - d)

                    # Medoid instance closest to circular mean
                    medoid = min(sorted_vals, key=lambda v: (_circ_dist(v), v))
                    consensus_q.append(float(medoid))
            else:
                # Linear metric slot: pick medoid closest to median
                median_val = sorted_vals[len(sorted_vals) // 2]
                medoid = min(sorted_vals, key=lambda v: (abs(v - median_val), v))
                consensus_q.append(float(medoid))

    # Reconstruct consensus sign and confidence
    signs = [p.sign for p in same_class]
    consensus_sign = 1 if signs.count(1) >= signs.count(-1) else -1
    mean_conf = sum(p.confidence for p in same_class) / len(same_class)

    provisional = EdgeParams(
        class_=ref.class_,
        quantized=tuple(consensus_q),
        sign=consensus_sign,
        confidence=1.0,
        identity_slots=tuple(sorted(id_slots)),
    )
    sims = [provisional.similarity(p) for p in same_class]
    param_consistency = sum(sims) / len(sims) if sims else 1.0

    return EdgeParams(
        class_=ref.class_,
        quantized=tuple(consensus_q),
        sign=consensus_sign,
        confidence=float(clamp(param_consistency * mean_conf, 0.0, 1.0)),
        identity_slots=tuple(sorted(id_slots)),
    )


# =============================================================================
# SIGNATURE SIMILARITY DISPATCHER (With Safe Contour Alignment)
# =============================================================================

def _sig_sim(a: Optional[TypedSignature], b: Optional[TypedSignature]) -> float:
    """Signature similarity with V4 boundary conformation turning-function alignment.

    NO exception swallow (Phase A contract retained): a raise from
    TypedSignature.similarity is a data-integrity bug and must surface.
    The `None` short-circuit is kept: a missing signature is a legitimate
    "no evidence" state, not an error.
    """
    if a is None or b is None:
        return 0.0

    # V4 Boundary Conformation: evaluate CONTOUR sequences with turning function alignment
    if (
        getattr(a, "class_", "") in ("CONTOUR", "CONTOUR_CONTENT")
        and getattr(b, "class_", "") in ("CONTOUR", "CONTOUR_CONTENT")
        and len(a.quantized) > 7
        and len(b.quantized) > 7
    ):
        # DYNAMIC LOOKUP: Prevents drift if CONSTITUTIONAL_RANGES changes
        full_ranges = CONSTITUTIONAL_RANGES.get("CONTOUR", ())
        base_ranges = full_ranges[:7] if len(full_ranges) >= 7 else (8.0, 12.0, 8.0, 2.0, 8.0, 8.0, 8.0)
        # [Pass 25 drift-hazard fix] Signature-class cycles live in
        # CIRCULAR_CYCLE_LENGTHS; CIRCULAR_EDGE_CYCLE_LENGTHS has no "CONTOUR"
        # key and would silently drop the slot-1 cycle-24 wrap.
        base_cycles = CIRCULAR_CYCLE_LENGTHS.get("CONTOUR", {})

        total_sq = 0.0
        for i in range(7):
            d = abs(a.quantized[i] - b.quantized[i])
            if i in base_cycles:
                n = base_cycles[i]
                d = min(d, n - d)
            sigma = base_ranges[i] / 3.0 if base_ranges[i] > 0.0 else 1.0
            total_sq += (d / sigma) ** 2

        base_sim = math.exp(-total_sq / 7.0)
        if base_sim <= 0.05:
            return 0.0

        # Curvature sequence: slots 7 to end (discrete signed curvature levels centered at 4.0)
        # Open-ended slicing guarantees resilience against CONTOUR_N_SEQ parameter changes
        turns_a = [(float(q) - 4.0) * (math.pi / 4.0) for q in a.quantized[7:]]
        turns_b = [(float(q) - 4.0) * (math.pi / 4.0) for q in b.quantized[7:]]
        is_closed = (float(a.quantized[3]) >= 0.5)

        curv_sim = turning_function_similarity(turns_a, turns_b, closed=is_closed, sigma=0.45)
        # 35% base invariant features + 65% boundary conformation curvature sequence
        return float(clamp(0.35 * base_sim + 0.65 * curv_sim, 0.0, 1.0))

    return float(a.similarity(b))


# [H.4b Gestalt boost] Multiplier applied to a matched node's mass when
# the node's lens type is a high-tier structural concept. Applied at
# score computation, not the matching weight matrix: the matcher is
# per-lens-type and a weight-matrix boost would be uniform within a type.
GESTALT_LENS_BOOST: Dict[str, float] = {
    "CurveComplex": 1.5,
    "Contour": 1.2,
    "Junction": 1.2,
}


def _max_weight_bipartite_matching(
    weights: List[List[float]],
    top_k: int = 8,
    max_local_opt_passes: int = 3,
) -> List[Tuple[int, int]]:
    """Approximate max-weight bipartite matching: sparse greedy + 2-opt.

    Replaces the O(N^3) Hungarian solver that was the dominant cost per
    MCS call (~200ms per call for 1300-element positives).

    Algorithm:
      1. Per-row top-K pruning (ties by column index ascending).
      2. Noise-floor union: also keep every pair with weight >= THETA_SIM.
         (Replaces the earlier median-threshold union, which on large
         matrices included up to half of all positive entries.)
      3. Global sort by (weight desc, row asc, col asc).
      4. Greedy assignment.
      5. 2-opt local improvement, bounded to max_local_opt_passes.

    Determinism (§9): sort keys are (weight, row, col) — canonical.
    Same input -> same output, run to run.

    Edge cases: empty matrix or all-zero weights -> [].
    """
    n = len(weights)
    if n == 0:
        return []
    m = len(weights[0]) if weights else 0
    if m == 0:
        return []

    positive_entries: List[List[Tuple[float, int]]] = []
    for i in range(n):
        row_entries = [(w, j) for j, w in enumerate(weights[i]) if w > 0.0]
        positive_entries.append(row_entries)

    if not any(positive_entries):
        return []

    candidates_set: Set[Tuple[int, int, float]] = set()

    # Phase 1: per-row top-K.
    for i in range(n):
        row_entries = positive_entries[i]
        if not row_entries:
            continue
        row_entries.sort(key=lambda x: (-x[0], x[1]))
        for w, j in row_entries[:top_k]:
            candidates_set.add((i, j, w))

    # Phase 2: noise-floor union.
    theta_sim = float(getattr(P, "THETA_SIM", 0.7))
    for i in range(n):
        for w, j in positive_entries[i]:
            if w >= theta_sim:
                candidates_set.add((i, j, w))

    # Phase 3: global sort.
    candidates = sorted(candidates_set, key=lambda x: (-x[2], x[0], x[1]))

    # Phase 4: greedy assignment.
    row_used = [False] * n
    col_used = [False] * m
    assignment: Dict[int, int] = {}
    weight_of: Dict[Tuple[int, int], float] = {
        (i, j): w for (i, j, w) in candidates_set
    }
    # [H.4b adaptive fix] candidates hold (row, col, weight) triples,
    # ranked by (-weight, row, col) in Phase 3; the drafted unpack was
    # (weight, row, col), which made row/col indices floats.
    for i, j, _w in candidates:

        if row_used[i] or col_used[j]:
            continue
        row_used[i] = True
        col_used[j] = True
        assignment[i] = j

    # Phase 5: 2-opt local improvement.
    _EPS = 1e-9
    for _ in range(max_local_opt_passes):
        improved = False
        rows = sorted(assignment.keys())
        for ai in range(len(rows)):
            for bi in range(ai + 1, len(rows)):
                ra, rb = rows[ai], rows[bi]
                ca, cb = assignment[ra], assignment[rb]
                w_ax_by = weight_of.get((ra, ca), 0.0) + weight_of.get(
                    (rb, cb), 0.0
                )
                w_ay_bx = weight_of.get((ra, cb), 0.0) + weight_of.get(
                    (rb, ca), 0.0
                )
                if w_ay_bx > w_ax_by + _EPS:
                    assignment[ra], assignment[rb] = cb, ca
                    improved = True
        if not improved:
            break

    pairs = [
        (r, c) for r, c in assignment.items()
        if weight_of.get((r, c), 0.0) > 0.0
    ]
    pairs.sort()
    return pairs

def node_structural_alignment(
    nodes_a: FrozenSet[int],
    nodes_b: FrozenSet[int],
    sig_provider: SignatureProvider,
) -> Tuple[float, Dict[int, int]]:
    """Compare node sets over cardinality, lens types, and signature alignment.
    Returns (node_similarity_score in [0, 1], node_mapping {u_a: u_b}).
    """
    if not nodes_a and not nodes_b:
        return 1.0, {}
    if not nodes_a or not nodes_b:
        return 0.0, {}

    sigs_a: Dict[int, Tuple[str, TypedSignature]] = {}
    for n in nodes_a:
        info = sig_provider(n)
        if info is not None:
            sigs_a[n] = info

    sigs_b: Dict[int, Tuple[str, TypedSignature]] = {}
    for n in nodes_b:
        info = sig_provider(n)
        if info is not None:
            sigs_b[n] = info

    len_a = len(sigs_a)
    len_b = len(sigs_b)
    if len_a == 0 and len_b == 0:
        return 1.0, {}
    if len_a == 0 or len_b == 0:
        return 0.0, {}

    # 1. Cardinality ratio
    size_ratio = min(len_a, len_b) / (max(len_a, len_b) + EPSILON)

    # 2. Lens type multiset overlap
    types_a: Dict[str, int] = defaultdict(int)
    for lt, _ in sigs_a.values():
        types_a[lt] += 1

    types_b: Dict[str, int] = defaultdict(int)
    for lt, _ in sigs_b.values():
        types_b[lt] += 1

    all_types = set(types_a.keys()) | set(types_b.keys())
    type_min_sum = sum(min(types_a[t], types_b[t]) for t in all_types)
    type_max_sum = sum(max(types_a[t], types_b[t]) for t in all_types)
    type_overlap = type_min_sum / (type_max_sum + EPSILON) if type_max_sum > 0 else 0.0

    # 3. Graded signature alignment per lens type
    mapping: Dict[int, int] = {}
    matched_sig_sum = 0.0

    for lt in sorted(set(types_a.keys()) & set(types_b.keys())):
        list_a = sorted([n for n, (t, _) in sigs_a.items() if t == lt])
        list_b = sorted([n for n, (t, _) in sigs_b.items() if t == lt])
        weights = [[_sig_sim(sigs_a[u][1], sigs_b[v][1]) for v in list_b] for u in list_a]
        assignment = _max_weight_bipartite_matching(weights)
        for i_a, i_b in assignment:
            w = weights[i_a][i_b]
            if w > 0.0:
                mapping[list_a[i_a]] = list_b[i_b]
                matched_sig_sum += w

    matched_count = len(mapping)
    sig_alignment = matched_sig_sum / (max(matched_count, 1) + EPSILON) if matched_count > 0 else 0.0

    node_score = W_SIZE * size_ratio + W_TYPE * type_overlap + W_ALIGN * sig_alignment
    return clamp(node_score, 0.0, 1.0), mapping


# =============================================================================
# 1b. ORGANIZATION-GROUNDED CORRESPONDENCE (§1.4/§1.5)
# =============================================================================

def _relational_agreement(
    u: int,
    v: int,
    mapping: Dict[int, int],
    inc_a: Dict[int, List[Tuple[int, EdgeType, bool]]],
    inc_b: Dict[int, List[Tuple[int, EdgeType, bool]]],
    b_dir: Set[Tuple[int, int, EdgeType]],
    b_und: Set[Tuple[int, int, EdgeType]],
    a_dir: Set[Tuple[int, int, EdgeType]],
    a_und: Set[Tuple[int, int, EdgeType]],
    rev: Dict[int, int] = None,
) -> float:
    """Conjunctive (min) mapped-context preservation for pair (u in A, v in B).

    Reference side: fraction of u's relations whose images under `mapping`
    exist at v. Target side: fraction of v's relations whose partner has a
    preimage under `mapping` and whose preimage exists at u. Min, not mean:
    a clean reference side must not compensate a contradicted target side.
    Relation-free nodes are identified by signature alone (1.0).

    [PERF] `rev` (the inverse mapping) may be passed in by callers that
    evaluate many pairs against one fixed mapping — it was previously
    rebuilt from scratch inside this function on every call. When omitted
    it is derived here; the value is identical either way.
    """
    anchors_a = getattr(inc_a, "anchors", None) or {}
    anchors_b = getattr(inc_b, "anchors", None) or {}
    junc_a = getattr(inc_a, "junctions", None) or ()
    junc_b = getattr(inc_b, "junctions", None) or ()
    tj_a = getattr(inc_a, "tj_index", None)
    tj_b = getattr(inc_b, "tj_index", None)
    p_a = getattr(inc_a, "params", {})
    p_b = getattr(inc_b, "params", {})

    edges_u = inc_a.get(u)
    if edges_u:
        kept = 0.0
        for w, et, outward in edges_u:
            mw = mapping.get(w)
            if mw is None:
                if _t_junction_near_nodes(anchors_a.get(u),
                                          anchors_a.get(w), junc_a,
                                          index=tj_a):
                    kept += P.PARTIAL_OCCLUSION_CREDIT
                continue
            if et in UNDIRECTED_EDGE_TYPES:
                lo, hi = (v, mw) if v < mw else (mw, v)
                ok = (lo, hi, et) in b_und
            else:
                ok = ((v, mw, et) if outward else (mw, v, et)) in b_dir
            if ok:
                a1, a2 = (u, w) if outward else (w, u)
                b1, b2 = (v, mw) if outward else (mw, v)
                kept += _pair_edge_sim(a1, a2, b1, b2, et, p_a, p_b)
        agree_u = kept / len(edges_u)
    else:
        agree_u = 1.0

    edges_v = inc_b.get(v)
    if edges_v:
        if rev is None:
            rev = {vv: uu for uu, vv in mapping.items()}
        kept = 0.0
        counted = 0
        for x, et, outward in edges_v:
            ux = rev.get(x)
            if ux is None:
                if _t_junction_near_nodes(anchors_b.get(v),
                                          anchors_b.get(x), junc_b,
                                          index=tj_b):
                    kept += P.PARTIAL_OCCLUSION_CREDIT
                continue
            counted += 1
            if et in UNDIRECTED_EDGE_TYPES:
                lo, hi = (u, ux) if u < ux else (ux, u)
                ok = (lo, hi, et) in a_und
            else:
                ok = ((ux, u, et) if outward else (u, ux, et)) in a_dir
            if ok:
                b1, b2 = (v, x) if outward else (x, v)
                a1, a2 = (u, ux) if outward else (ux, u)
                kept += _pair_edge_sim(a1, a2, b1, b2, et, p_a, p_b)
        agree_v = min(1.0, kept / counted) if counted else 1.0
    else:
        agree_v = 1.0
    return min(agree_u, agree_v)


class _IncidenceDict(dict):
    """[V9 B.12 / S5] Incidence map that also carries its side's occupancy
    and edge_params."""

    __slots__ = ("anchors", "junctions", "params", "tj_index")

    def __init__(self, *args, anchors=None, junctions=None, params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchors = anchors or {}
        self.junctions = junctions or ()
        self.params = params or {}
        self.tj_index = None


def _tj_spatial_index(junctions, radius: float):
    """[PERF] Uniform-grid index over T-junction contact points.

    Builds a dict cell -> list of (cx, cy) contacts plus a pre-flattened
    float tuple per junction so the hot query in _t_junction_near_nodes is
    O(cells touched) instead of O(|junctions|). The conservative schema
    rules are preserved exactly: a junction contributes only when it
    unpacks to 4 fields AND its contact supports [0]/[1] indexing; those
    try/except filters run at build time, once, not per query."""
    if not junctions or radius <= 0.0:
        return None
    cells = {}
    for j in junctions:
        try:
            _stem_idx, _top_idx, contact, _confidence = j
            cx = float(contact[0]); cy = float(contact[1])
        except (TypeError, ValueError, IndexError):
            continue
        key = (int(math.floor(cx / radius)), int(math.floor(cy / radius)))
        cells.setdefault(key, []).append((cx, cy))
    return {"cells": cells, "r": float(radius)}


def _t_junction_near_nodes(anchor_a, anchor_b, junctions,
                           radius: float = None, index=None) -> bool:
    """[V9 B.12] Is a scene T-junction within `T_JUNCTION_RADIUS` of either
    endpoint? Conservative: an unrecognised junction schema credits
    nothing rather than guessing an unpack order.

    [PERF] When `index` (from _tj_spatial_index) is supplied, only the grid
    cells overlapping each endpoint's query disc are scanned; the distance
    predicate is unchanged (< r, strict), so results are identical."""
    if anchor_a is None or anchor_b is None or not junctions:
        return False
    r = P.T_JUNCTION_RADIUS if radius is None else radius
    ax, ay = anchor_a[0], anchor_a[1]
    bx, by = anchor_b[0], anchor_b[1]
    if index is not None:
        rr = index["r"]
        seen = set()
        for px, py in ((ax, ay), (bx, by)):
            c0 = int(math.floor((px - rr) / rr)); c1 = int(math.floor((px + rr) / rr))
            d0 = int(math.floor((py - rr) / rr)); d1 = int(math.floor((py + rr) / rr))
            for ci in range(c0, c1 + 1):
                for dj in range(d0, d1 + 1):
                    if (ci, dj) in seen:
                        continue
                    seen.add((ci, dj))
                    for cx, cy in index["cells"].get((ci, dj), ()):
                        if (math.hypot(cx - ax, cy - ay) < r
                                or math.hypot(cx - bx, cy - by) < r):
                            return True
        return False
    for j in junctions:
        try:
            _stem_idx, _top_idx, contact, _confidence = j
        except (TypeError, ValueError):
            continue
        try:
            cx, cy = contact[0], contact[1]
        except (TypeError, IndexError):
            continue
        if (math.hypot(cx - ax, cy - ay) < r
                or math.hypot(cx - bx, cy - by) < r):
            return True
    return False


def _incidence(sg: Subgraph, sigs: Dict[int, Tuple[str, TypedSignature]]):
    inc: Dict[int, List[Tuple[int, EdgeType, bool]]] = defaultdict(list)
    for srt, dst, et in sg.edges:
        if srt in sigs and dst in sigs:
            inc[srt].append((dst, et, True))
            inc[dst].append((srt, et, False))
    junctions = getattr(sg, "t_junctions", None) or ()
    out = _IncidenceDict(
        inc,
        anchors=getattr(sg, "anchor_of_node", None) or {},
        junctions=junctions,
        params=getattr(sg, "edge_params", None) or {},
    )
    # [PERF] one-time spatial index for the occlusion-credit queries
    out.tj_index = _tj_spatial_index(junctions, float(P.T_JUNCTION_RADIUS))
    return out


def _edge_indexes(sg: Subgraph, sigs: Dict[int, Tuple[str, TypedSignature]]):
    directed: Set[Tuple[int, int, EdgeType]] = set()
    undirected: Set[Tuple[int, int, EdgeType]] = set()
    for srt, dst, et in sg.edges:
        if srt in sigs and dst in sigs:
            if et in UNDIRECTED_EDGE_TYPES:
                undirected.add((min(srt, dst), max(srt, dst), et))
            else:
                directed.add((srt, dst, et))
    return directed, undirected


def organization_alignment(
    sg_a: Subgraph,
    sg_b: Subgraph,
    sig_provider: SignatureProvider,
    rounds: int = 3,
) -> Tuple[Dict[int, int], Dict[int, float]]:
    """Grounded fixed-point correspondence (multi-start deterministic).

    q(u, v) = sig_sim(u, v) * relational_agreement(u, v | M); Hungarian
    re-assignment iterates until M stabilizes or the budget is spent.
    Seeds: (1) signature-bag assignment, (2) degree-profile-weighted
    assignment; the fixed point with the higher total grounded quality is
    returned. Matches are pairs with q >= THETA_MATCH (P.THETA_SIM).
    """
    sigs_a: Dict[int, Tuple[str, TypedSignature]] = {}
    for n in sg_a.nodes:
        info = sig_provider(n)
        if info is not None:
            sigs_a[n] = info
    sigs_b: Dict[int, Tuple[str, TypedSignature]] = {}
    for n in sg_b.nodes:
        info = sig_provider(n)
        if info is not None:
            sigs_b[n] = info
    if not sigs_a or not sigs_b:
        return {}, {}

    inc_a = _incidence(sg_a, sigs_a)
    inc_b = _incidence(sg_b, sigs_b)
    a_dir, a_und = _edge_indexes(sg_a, sigs_a)
    b_dir, b_und = _edge_indexes(sg_b, sigs_b)

    types_a: Dict[str, List[int]] = defaultdict(list)
    for n in sorted(sigs_a):
        types_a[sigs_a[n][0]].append(n)
    types_b: Dict[str, List[int]] = defaultdict(list)
    for n in sorted(sigs_b):
        types_b[sigs_b[n][0]].append(n)
    common = sorted(set(types_a) & set(types_b))

    def _profile(sg, sigs, n):
        prof: Dict[EdgeType, int] = defaultdict(int)
        for _w, et, _o in inc_a.get(n, ()) if sigs is sigs_a else inc_b.get(n, ()):
            prof[et] += 1
        return prof

    def _assign(weights_fn) -> Dict[int, int]:
        m: Dict[int, int] = {}
        for lt in common:
            la, lb = types_a[lt], types_b[lt]
            weights = [[weights_fn(u, v) for v in lb] for u in la]
            for ia, ib in _max_weight_bipartite_matching(weights):
                if weights[ia][ib] > 0.0:
                    m[la[ia]] = lb[ib]
        return m

    def _sig_w(u, v):
        return _sig_sim(sigs_a[u][1], sigs_b[v][1])

    def _prof_w(u, v):
        base = _sig_sim(sigs_a[u][1], sigs_b[v][1])
        if base <= 0.0:
            return 0.0
        pa, pb = _profile(sg_a, sigs_a, u), _profile(sg_b, sigs_b, v)
        keys = set(pa) | set(pb)
        if not keys:
            return base
        inter = sum(min(pa.get(k, 0), pb.get(k, 0)) for k in keys)
        union = sum(max(pa.get(k, 0), pb.get(k, 0)) for k in keys)
        return base * (0.5 + 0.5 * (inter / union if union else 1.0))

    def _refine(seed: Dict[int, int]):
        mapping = dict(seed)
        for _ in range(max(0, rounds - 1)):
            rev_map = {vv: uu for uu, vv in mapping.items()}

            def w_fn(u, v, m=mapping, rv=rev_map):
                base = _sig_sim(sigs_a[u][1], sigs_b[v][1])
                if base <= 0.0:
                    return 0.0
                agree = _relational_agreement(u, v, m, inc_a, inc_b, b_dir, b_und, a_dir, a_und, rev=rv)
                return base * agree if agree > 0.0 else 0.0
            new_m = _assign(w_fn)
            if new_m == mapping:
                break
            mapping = new_m
        qvals: Dict[int, float] = {}
        rev_final = {vv: uu for uu, vv in mapping.items()}
        for u, v in mapping.items():
            base = _sig_sim(sigs_a[u][1], sigs_b[v][1])
            if base <= 0.0:
                continue
            agree = _relational_agreement(u, v, mapping, inc_a, inc_b, b_dir, b_und, a_dir, a_und, rev=rev_final)
            q = base * agree
            if q > 0.0:
                qvals[u] = q
        return mapping, qvals, sum(qvals.values())

    results = [_refine(_assign(_sig_w)), _refine(_assign(_prof_w))]
    results.sort(key=lambda r: -r[2])
    mapping, qvals, _tot = results[0]
    theta = float(getattr(P, "THETA_SIM", 0.7))
    matches = {u: q for u, q in qvals.items() if q >= theta}
    return mapping, matches


# =============================================================================
# 2. EDGE DIMENSION HELPERS
# =============================================================================

def edge_structural_alignment(
    edges_a: FrozenSet[Tuple[int, int, EdgeType]],
    edges_b: FrozenSet[Tuple[int, int, EdgeType]],
    node_mapping: Dict[int, int],
    params_a: Optional[Dict[Tuple[int, int, EdgeType], Any]] = None,
    params_b: Optional[Dict[Tuple[int, int, EdgeType], Any]] = None,
) -> float:
    """Compare edge sets over cardinality, binder types, and topological alignment
    modulated by relational EdgeParams similarity."""
    if not edges_a and not edges_b:
        return 1.0
    if not edges_a or not edges_b:
        return 0.0

    len_a = len(edges_a)
    len_b = len(edges_b)
    edge_ratio = min(len_a, len_b) / (max(len_a, len_b) + EPSILON)

    # 1. Binder channel multiset overlap
    types_a: Dict[EdgeType, int] = defaultdict(int)
    for _, _, et in edges_a:
        types_a[et] += 1

    types_b: Dict[EdgeType, int] = defaultdict(int)
    for _, _, et in edges_b:
        types_b[et] += 1

    all_ets = set(types_a.keys()) | set(types_b.keys())
    et_min_sum = sum(min(types_a[et], types_b[et]) for et in all_ets)
    et_max_sum = sum(max(types_a[et], types_b[et]) for et in all_ets)
    binder_overlap = et_min_sum / (et_max_sum + EPSILON) if et_max_sum > 0 else 0.0

    # 2. Topological alignment modulated by relational EdgeParams similarity
    set_b = set(edges_b)
    aligned_w = 0.0
    for src, dst, et in edges_a:
        mapped_src = node_mapping.get(src)
        mapped_dst = node_mapping.get(dst)
        if mapped_src is not None and mapped_dst is not None:
            if et in UNDIRECTED_EDGE_TYPES:
                lo, hi = (mapped_src, mapped_dst) if mapped_src < mapped_dst else (mapped_dst, mapped_src)
                is_matched = ((lo, hi, et) in set_b) or ((mapped_src, mapped_dst, et) in set_b)
            else:
                is_matched = (mapped_src, mapped_dst, et) in set_b

            if is_matched:
                aligned_w += _pair_edge_sim(src, dst, mapped_src, mapped_dst, et, params_a, params_b)

    topo_align = aligned_w / (max(len_a, len_b) + EPSILON)
    edge_score = W_SIZE * edge_ratio + W_TYPE * binder_overlap + W_ALIGN * topo_align
    return clamp(edge_score, 0.0, 1.0)


# =============================================================================
# 3. PUBLIC SUBGRAPH STRUCTURAL SIMILARITY
# =============================================================================

def subgraph_structural_similarity(
    sg_a: Subgraph,
    sg_b: Subgraph,
    sig_provider: SignatureProvider,
    alpha: Optional[float] = None,
) -> float:
    """Unified structural comparison between two subgraphs with full EdgeParams sensitivity.
    Returns similarity in [0.0, 1.0].
    """
    if alpha is None:
        alpha = DEFAULT_ALPHA
    node_score, mapping = node_structural_alignment(sg_a.nodes, sg_b.nodes, sig_provider)
    p_a = getattr(sg_a, "edge_params", None)
    p_b = getattr(sg_b, "edge_params", None)
    edge_score = edge_structural_alignment(sg_a.edges, sg_b.edges, mapping, params_a=p_a, params_b=p_b)
    combined = alpha * node_score + (1.0 - alpha) * edge_score
    return clamp(combined, 0.0, 1.0)


def node_structural_alignment_positional(
    nodes_a: FrozenSet[int],
    nodes_b: FrozenSet[int],
    sig_provider: SignatureProvider,
    rel_pos_provider: Callable[[int], Optional[Tuple[float, float]]],
    pos_alpha: float = 8.0,
    pos_beta: float = 0.5,
) -> Tuple[float, Dict[int, int]]:
    """One-to-one node alignment modulated by envelope-relative position.

    Same-type signature similarity is multiplied by a positional factor
        factor = (1 - pos_beta) + pos_beta * exp(-pos_alpha * dist^2)
    where dist is the Euclidean distance between the two nodes'
    envelope-relative positions (each in [0,1]^2). Nodes lacking a position
    get factor = 1.0 (neutral, signature-only). The blended matrix is
    solved with a single Hungarian pass, preserving bijection and global
    optimality.

    Position can only DOWN-weight (factor <= 1.0), never inflate
    similarity, so no false high-similarity matches are created.
    """
    if not nodes_a and not nodes_b:
        return 1.0, {}
    if not nodes_a or not nodes_b:
        return 0.0, {}

    list_a = sorted(n for n in nodes_a if sig_provider(n) is not None)
    list_b = sorted(n for n in nodes_b if sig_provider(n) is not None)
    if not list_a or not list_b:
        return 0.0, {}

    weights = [[0.0 for _ in list_b] for _ in list_a]
    for i, u in enumerate(list_a):
        tu, su = sig_provider(u)
        pu = rel_pos_provider(u) if rel_pos_provider is not None else None
        for j, v in enumerate(list_b):
            tv, sv = sig_provider(v)
            if tu != tv:
                continue
            sig_s = _sig_sim(su, sv)
            if sig_s <= 0.0:
                continue
            pv = rel_pos_provider(v) if rel_pos_provider is not None else None
            if pu is not None and pv is not None:
                dx = pu[0] - pv[0]
                dy = pu[1] - pv[1]
                pos_s = math.exp(-pos_alpha * (dx * dx + dy * dy))
                factor = (1.0 - pos_beta) + pos_beta * pos_s
            else:
                factor = 1.0
            weights[i][j] = sig_s * factor

    assignment = _max_weight_bipartite_matching(weights)
    mapping: Dict[int, int] = {}
    total = 0.0
    for i, j in assignment:
        w = weights[i][j]
        if w > 0.0:
            mapping[list_a[i]] = list_b[j]
            total += w

    denom = float(max(len(list_a), len(list_b)))
    score = total / (denom + EPSILON)
    return clamp(score, 0.0, 1.0), mapping



def organizational_similarity(
    sg_a: Subgraph,
    sg_b: Subgraph,
    sig_provider: SignatureProvider,
    theta_node: float = 0.55,
    lam: float = 0.5,
    refine_iters: int = 3,
    edge_weight: float = 0.85,
) -> float:
    """Conjunctive organizational-overlap similarity (§1.5).

    Scores matched ORGANIZATIONAL STRUCTURE, not additive bag statistics.
    An edge contributes only if ALL hold (conjunctive):
      1. Both endpoint nodes map to same-lens-type nodes in the other graph
         with signature similarity >= theta_node.
      2. The corresponding edge (same EdgeType, undirected-aware) exists
         there — in the SAME direction for directed types.
      3. Contribution is weighted by the product of endpoint signature
         similarities (node-quality gate).

    The node correspondence is made organization-aware via graduated-
    assignment refinement rewarding mappings that preserve same-type edge
    neighborhoods.

    [FIXES over the supplied patch]
      - directed edge preservation no longer accepts the reversed tuple;
      - undirected edges populate BOTH adjacency directions so max-endpoint
        nodes participate in edge-aware refinement;
      - both-edgeless identity limit returns node_cov (not 0.5*node_cov).
    """
    nodes_a = sorted(n for n in sg_a.nodes if sig_provider(n) is not None)
    nodes_b = sorted(n for n in sg_b.nodes if sig_provider(n) is not None)
    if not nodes_a or not nodes_b:
        return 0.0

    sig_a = {n: sig_provider(n) for n in nodes_a}
    sig_b = {n: sig_provider(n) for n in nodes_b}
    na, nb = len(nodes_a), len(nodes_b)

    # 1. Compatibility matrix: lens-type-gated signature similarity.
    M0 = np.zeros((na, nb), dtype=np.float64)
    for i, ua in enumerate(nodes_a):
        ta, sa = sig_a[ua]
        for j, ub in enumerate(nodes_b):
            tb, sb = sig_b[ub]
            if ta == tb:
                M0[i, j] = _sig_sim(sa, sb)

    idx_a = {n: i for i, n in enumerate(nodes_a)}
    idx_b = {n: i for i, n in enumerate(nodes_b)}

    def build_adj(edges, idx):
        # Neighborhood context for refinement includes BOTH in- and
        # out-directions for every edge type: a directed edge's dst node
        # otherwise has an empty neighborhood and its compatibility decays
        # below theta_node under refinement, dropping it from the
        # correspondence (measured: identical graphs scored 0.46).
        # Directionality itself is enforced later, in preserved().
        adj = [{} for _ in range(len(idx))]
        for src, dst, et in edges:
            if src in idx and dst in idx:
                adj[idx[src]].setdefault(et, []).append(idx[dst])
                adj[idx[dst]].setdefault(et, []).append(idx[src])
        return adj

    adj_a = build_adj(sg_a.edges, idx_a)
    adj_b = build_adj(sg_b.edges, idx_b)

    # 2. Graduated-assignment edge-aware refinement.
    M = M0.copy()
    for _ in range(refine_iters):
        S = np.zeros((na, nb), dtype=np.float64)
        for i in range(na):
            for j in range(nb):
                acc = 0.0
                for et, nbrs_a in adj_a[i].items():
                    nbrs_b = adj_b[j].get(et)
                    if not nbrs_b:
                        continue
                    for k in nbrs_a:
                        row = M[k]
                        for l in nbrs_b:
                            acc += row[l]
                S[i, j] = acc
        smax = float(S.max()) if S.size else 0.0
        if smax <= 0.0:
            # No edge context at all: keep pure signature compatibility.
            # Blending with an all-zero S would shrink M below theta_node
            # and empty the correspondence (measured: edgeless identity
            # scored 0.0).
            continue
        S /= smax
        M = (1.0 - lam) * M0 + lam * S

    # 3. Hard assignment (Hungarian), gated by theta_node.
    assignment = _max_weight_bipartite_matching(M.tolist())
    pi = {}
    pi_rev = {}
    for i, j in assignment:
        if M[i, j] >= theta_node:
            pi[nodes_a[i]] = nodes_b[j]
            pi_rev[nodes_b[j]] = nodes_a[i]

    s_node_cov = len(pi) / float(max(na, nb))

    has_edges_a = len(sg_a.edges) > 0
    has_edges_b = len(sg_b.edges) > 0

    if not has_edges_a and not has_edges_b:
        # Identity limit of the coverage formula with no organization on
        # either side: pure node coverage.
        return clamp(s_node_cov, 0.0, 1.0)
    if not (has_edges_a and has_edges_b):
        return clamp((1.0 - edge_weight) * s_node_cov, 0.0, 1.0)

    edge_set_a = set(sg_a.edges)
    edge_set_b = set(sg_b.edges)

    def preserved(u, v, et, pi_map, target_set):
        mu, mv = pi_map.get(u), pi_map.get(v)
        if mu is None or mv is None:
            return False
        if et in UNDIRECTED_EDGE_TYPES:
            lo, hi = (mu, mv) if mu < mv else (mv, mu)
            return (lo, hi, et) in target_set
        return (mu, mv, et) in target_set

    # 4a. Forward coverage: A edges preserved in B.
    # [SPEC FIX] contrib weights are the signature similarities
    # sim(sigma_u, sigma_pi(u)) * sim(sigma_v, sigma_pi(v)) — i.e. M0, the
    # type-gated signature compatibility — not the refined blend M (which
    # leaks cross-node similarity noise into endpoint quality and caps
    # identical-graph scores below 1.0).
    forward = 0.0
    for u, v, et in sg_a.edges:
        if preserved(u, v, et, pi, edge_set_b):
            forward += M0[idx_a[u], idx_b[pi[u]]] * M0[idx_a[v], idx_b[pi[v]]]
    fwd_cov = forward / float(len(sg_a.edges))

    # 4b. Backward coverage: B edges preserved in A.
    backward = 0.0
    for u, v, et in sg_b.edges:
        if preserved(u, v, et, pi_rev, edge_set_a):
            backward += (
                M0[idx_a[pi_rev[u]], idx_b[u]] * M0[idx_a[pi_rev[v]], idx_b[v]]
            )
    bwd_cov = backward / float(len(sg_b.edges))

    s_edge_org = 0.5 * (fwd_cov + bwd_cov)
    return clamp(
        edge_weight * s_edge_org + (1.0 - edge_weight) * s_node_cov,
        0.0,
        1.0,
    )


def neighborhood_structural_similarity(
    u: int,
    v: int,
    cg: Any,
    sig_provider: SignatureProvider,
    radius: int = 1,
    alpha: Optional[float] = None,
) -> float:
    """Compare the structural neighborhoods around node u and node v in cg."""
    sg_u = _extract_neighborhood_subgraph(u, cg, radius)
    sg_v = _extract_neighborhood_subgraph(v, cg, radius)
    return subgraph_structural_similarity(sg_u, sg_v, sig_provider, alpha=alpha)


def _extract_neighborhood_subgraph(center: int, cg: Any, radius: int) -> Subgraph:
    visited: Set[int] = {center}
    frontier: Set[int] = {center}
    edges: Set[Tuple[int, int, EdgeType]] = set()

    for _ in range(radius):
        next_frontier: Set[int] = set()
        for n in frontier:
            for e in cg.get_outgoing_edges(n):
                edges.add((e.parent, e.child, e.edge_type))
                if e.child not in visited:
                    visited.add(e.child)
                    next_frontier.add(e.child)
            for e in cg.get_incoming_edges(n):
                edges.add((e.parent, e.child, e.edge_type))
                if e.parent not in visited:
                    visited.add(e.parent)
                    next_frontier.add(e.parent)
        frontier = next_frontier
        if not frontier:
            break

    return Subgraph(
        nodes=frozenset(visited),
        edges=frozenset(edges),
        anchor_of_node=anchor_positions_for(
            visited, cg.pattern_node_position
        ),
    )


# =============================================================================
# 4. STRUCTURAL TEMPLATE & INTERSECTION OPERATORS (§5.3.2)
# =============================================================================

@dataclass
class StructuralTemplate:
    """§5.3.2 v32.2 (graded): nodes carry (lens_type, consensus_sig,
    consistency); edges carry (EdgeType, src_idx, dst_idx, consistency)
    or with consensus EdgeParams (EdgeType, src_idx, dst_idx, consistency, edge_params).
    Consistency ∈ [0, 1]."""
    nodes: List[Tuple[str, TypedSignature, float]] = field(default_factory=list)
    edges: List[Tuple] = field(default_factory=list)


def structural_intersection(
    positives: List[Subgraph],
    sig_provider: SignatureProvider,
    theta_sim: float = 0.70,                     # UNUSED — retained for compat
    min_positive_ratio: float = 0.0,             # DEPRECATED alias; ignored
    rel_pos_provider: Optional[Callable[[int], Optional[Tuple[float, float]]]] = None,
    pos_alpha: float = 8.0,
    pos_beta: float = 0.5,
    min_frequency: Optional[float] = None,
    min_consistency: float = None,
    n_neg: int = 8,
    diag: Optional[Dict[str, Any]] = None,   # [G.1.6a] instrumentation
) -> Optional[StructuralTemplate]:
    """[v32.2 GRADED] Permissive template builder. Every exemplar node is
    a candidate; consistency is graded. Downstream filter applies mass.

    [FIX 2] Recurrence rule: when min_frequency is None, it is computed
    from _wilson_min_recurrence(n_pos, n_neg) -- the minimum count pc such
    that Wilson-lower(pc/n_pos) > Wilson-upper(0/n_neg). For n_pos=7,
    n_neg=8 this gives pc=4 (frequency 4/7 ≈ 0.571). THE single source of
    truth for the recurrence rule is _wilson_min_recurrence in
    memory/distributions.py; THETA_MIN_FREQ/_text.py carries no default.
    The rule is applied at the TEMPLATE step (intersection), not at the
    mass step (_lcb_beats_chance in memory/distributions.py tests whether a
    given count beats chance).
    """
    if min_frequency is None:
        from memory.distributions import _wilson_min_recurrence

        min_frequency = _wilson_min_recurrence(len(positives), n_neg) / float(
            len(positives)
        )
    if min_consistency is None:
        min_consistency = 0.0  # THETA_MIN_CONSISTENCY removed (dead 0.0 knob)

    if not positives:
        return None

    if len(positives) == 1:
        p = positives[0]
        node_list: List[Tuple[str, TypedSignature, float]] = []
        node_to_idx: Dict[int, int] = {}
        for nid in sorted(p.nodes):
            info = sig_provider(nid)
            if info is not None:
                node_to_idx[nid] = len(node_list)
                node_list.append((info[0], info[1], 1.0))
        edge_list: List[Tuple] = []
        p_params = getattr(p, "edge_params", None) or {}
        for src, dst, et in p.edges:
            if src in node_to_idx and dst in node_to_idx:
                canon = (min(src, dst), max(src, dst), et) if et in UNDIRECTED_EDGE_TYPES else (src, dst, et)
                ep = p_params.get((src, dst, et)) or p_params.get(canon)
                if ep is None and et in UNDIRECTED_EDGE_TYPES:
                    ep = p_params.get((dst, src, et))
                edge_list.append((et, node_to_idx[src], node_to_idx[dst], 1.0, ep))
        return StructuralTemplate(nodes=node_list, edges=edge_list)

    exemplar = min(positives, key=lambda sg: (len(sg.nodes), len(sg.edges), sorted(sg.nodes)))
    other_positives = [sg for sg in positives if sg is not exemplar]

    if diag is not None:
        diag["inter_calls"] = diag.get("inter_calls", 0) + 1
        diag["inter_positives_count"] = len(positives)
        diag["inter_exemplar_nodes"] = len(exemplar.nodes)
        diag["inter_exemplar_edges"] = len(exemplar.edges)

    def _align(other_sg: "Subgraph") -> Dict[int, int]:
        if rel_pos_provider is not None:
            _, mapping = node_structural_alignment_positional(
                exemplar.nodes, other_sg.nodes, sig_provider,
                rel_pos_provider, pos_alpha, pos_beta,
            )
            return mapping
        mapping, _matches = organization_alignment(exemplar, other_sg, sig_provider)
        return mapping

    mappings: List[Dict[int, int]] = [_align(o) for o in other_positives]

    total_positives = len(positives)
    consensus_nodes: List[Tuple[str, TypedSignature, float]] = []
    exemplar_to_template: Dict[int, int] = {}

    for u in sorted(exemplar.nodes):
        u_info = sig_provider(u)
        if u_info is None:
            continue
        u_type, u_sig = u_info

        aligned_count = 1
        aligned_sigs: List[TypedSignature] = [u_sig]
        for other, mapping in zip(other_positives, mappings):
            v = mapping.get(u)
            if v is None:
                continue
            v_info = sig_provider(v)
            if v_info is None or v_info[0] != u_type:
                continue
            aligned_count += 1
            aligned_sigs.append(v_info[1])

        frequency = aligned_count / total_positives
        if diag is not None:
            # aligned_count == int(frequency * total_positives), so the
            # bucket key is the raw alignment count across positives.
            _b = f"inter_node_freq_bin_{aligned_count}"
            diag[_b] = diag.get(_b, 0) + 1
        if frequency < min_frequency:
            continue

        if len(aligned_sigs) < 2:
            consistency = 1.0
        else:
            total_sim = 0.0
            n_pairs = 0
            for i in range(len(aligned_sigs)):
                for j in range(i + 1, len(aligned_sigs)):
                    total_sim += _sig_sim(aligned_sigs[i], aligned_sigs[j])
                    n_pairs += 1
            consistency = total_sim / n_pairs if n_pairs > 0 else 1.0

        if consistency < min_consistency:
            continue

        best_sig = max(aligned_sigs, key=lambda s: min(_sig_sim(s, o) for o in aligned_sigs))

        template_idx = len(consensus_nodes)
        consensus_nodes.append((u_type, best_sig, float(consistency)))
        exemplar_to_template[u] = template_idx

    if diag is not None:
        diag["inter_consensus_nodes"] = len(consensus_nodes)

    if not consensus_nodes:
        return None

    other_edge_sets: List[Set[Tuple[int, int, EdgeType]]] = [set(o.edges) for o in other_positives]

    positive_edges_in_exemplar_space: List[Set[Tuple[int, int, EdgeType]]] = []
    positive_edge_params_in_exemplar_space: List[Dict[Tuple[int, int, EdgeType], EdgeParams]] = []

    ex_canonical: Set[Tuple[int, int, EdgeType]] = set()
    ex_params_map: Dict[Tuple[int, int, EdgeType], EdgeParams] = {}
    ex_edge_params = getattr(exemplar, "edge_params", None) or {}
    for (s, d, et) in exemplar.edges:
        if et in UNDIRECTED_EDGE_TYPES:
            a, b = (s, d) if s < d else (d, s)
            canon = (a, b, et)
        else:
            canon = (s, d, et)
        ex_canonical.add(canon)
        ep = ex_edge_params.get((s, d, et)) or ex_edge_params.get(canon)
        if ep is None and et in UNDIRECTED_EDGE_TYPES:
            ep = ex_edge_params.get((d, s, et))
        if ep is not None:
            ex_params_map[canon] = ep

    positive_edges_in_exemplar_space.append(ex_canonical)
    positive_edge_params_in_exemplar_space.append(ex_params_map)

    for other_edges, other_sg, mapping in zip(other_edge_sets, other_positives, mappings):
        reverse = {v: u for u, v in mapping.items()}
        mapped: Set[Tuple[int, int, EdgeType]] = set()
        mapped_params: Dict[Tuple[int, int, EdgeType], EdgeParams] = {}
        o_edge_params = getattr(other_sg, "edge_params", None) or {}
        for (o_s, o_d, o_et) in other_edges:
            ex_s = reverse.get(o_s)
            ex_d = reverse.get(o_d)
            if ex_s is None or ex_d is None:
                continue
            if o_et in UNDIRECTED_EDGE_TYPES:
                a, b = (ex_s, ex_d) if ex_s < ex_d else (ex_d, ex_s)
                canon = (a, b, o_et)
                o_canon = (min(o_s, o_d), max(o_s, o_d), o_et)
            else:
                canon = (ex_s, ex_d, o_et)
                o_canon = (o_s, o_d, o_et)
            mapped.add(canon)
            ep = o_edge_params.get((o_s, o_d, o_et)) or o_edge_params.get(o_canon)
            if ep is None and o_et in UNDIRECTED_EDGE_TYPES:
                ep = o_edge_params.get((o_d, o_s, o_et))
            if ep is not None:
                mapped_params[canon] = ep

        positive_edges_in_exemplar_space.append(mapped)
        positive_edge_params_in_exemplar_space.append(mapped_params)

    candidate_edges_set: Set[Tuple[int, int, EdgeType]] = set()
    for s in positive_edges_in_exemplar_space:
        candidate_edges_set |= s

    consensus_edges: List[Tuple] = []
    _drop_endpoint = 0
    _drop_freq = 0
    _drop_consistency = 0
    for (src, dst, et) in sorted(candidate_edges_set, key=lambda e: (e[0], e[1], e[2].value)):
        if src not in exemplar_to_template or dst not in exemplar_to_template:
            _drop_endpoint += 1
            continue
        present_count = sum(
            1 for s in positive_edges_in_exemplar_space
            if (src, dst, et) in s
        )
        if diag is not None:
            _b = f"inter_edge_freq_bin_{present_count}"
            diag[_b] = diag.get(_b, 0) + 1
        frequency = present_count / total_positives
        if frequency < min_frequency:
            _drop_freq += 1
            continue

        aligned_eps: List[EdgeParams] = []
        for p_map in positive_edge_params_in_exemplar_space:
            if (src, dst, et) in p_map:
                aligned_eps.append(p_map[(src, dst, et)])

        # ----- V4 consensus (replaces mean+circular-mean block) -----
        consensus_ep = build_consensus_edge_params(aligned_eps)

        edge_consistency = float(frequency) * float(consensus_ep.confidence if consensus_ep else 1.0)

        if edge_consistency < min_consistency:
            _drop_consistency += 1
            continue
        consensus_edges.append(
            (et, exemplar_to_template[src], exemplar_to_template[dst], edge_consistency, consensus_ep)
        )

    if diag is not None:
        diag["inter_candidate_edges_pre_endpoint_filter"] = len(
            candidate_edges_set)
        diag["inter_edges_dropped_missing_endpoint"] = (
            diag.get("inter_edges_dropped_missing_endpoint", 0)
            + _drop_endpoint
        )
        diag["inter_edges_dropped_by_frequency"] = (
            diag.get("inter_edges_dropped_by_frequency", 0) + _drop_freq
        )
        diag["inter_edges_dropped_by_consistency"] = (
            diag.get("inter_edges_dropped_by_consistency", 0)
            + _drop_consistency
        )
        diag["inter_consensus_edges"] = (
            diag.get("inter_consensus_edges", 0) + len(consensus_edges)
        )

    return StructuralTemplate(nodes=consensus_nodes, edges=consensus_edges)


def materialize_template(
    template: StructuralTemplate,
    exemplar: Subgraph,
    sig_provider: SignatureProvider,
) -> Subgraph:
    """Materialize a canonical representative Subgraph (§5.3.2 v32.2).
    Undirected edges are canonicalized at insertion so downstream mass
    lookups (which canonicalize) find them."""
    exemplar_by_type: Dict[str, List[int]] = defaultdict(list)
    exemplar_sigs: Dict[int, Tuple[str, TypedSignature]] = {}
    for nid in sorted(exemplar.nodes):
        info = sig_provider(nid)
        if info is not None:
            exemplar_by_type[info[0]].append(nid)
            exemplar_sigs[nid] = info

    assignments: List[Optional[int]] = [None] * len(template.nodes)
    node_consistency: List[float] = [1.0] * len(template.nodes)
    used: Set[int] = set()
    for i, entry in enumerate(template.nodes):
        if len(entry) == 3:
            nt, consensus_sig, cons = entry
        else:
            nt, consensus_sig = entry
            cons = 1.0
        best_nid = None
        best_sim = -1.0
        for nid in exemplar_by_type.get(nt, []):
            if nid in used:
                continue
            sim = _sig_sim(exemplar_sigs[nid][1], consensus_sig)
            if sim > best_sim:
                best_sim = sim
                best_nid = nid
        if best_nid is not None:
            assignments[i] = best_nid
            node_consistency[i] = float(cons)
            used.add(best_nid)

    mat_nodes = frozenset(n for n in assignments if n is not None)
    mat_node_weights: Dict[int, float] = {}
    for i, nid in enumerate(assignments):
        if nid is not None:
            mat_node_weights[nid] = node_consistency[i]

    mat_edges: Set[Tuple[int, int, EdgeType]] = set()
    mat_edge_weights: Dict[Tuple[int, int, EdgeType], float] = {}
    mat_edge_params: Dict[Tuple[int, int, EdgeType], EdgeParams] = {}
    for entry in template.edges:
        if len(entry) >= 5:
            et, src_idx, dst_idx, cons, ep = entry[:5]
        elif len(entry) == 4:
            et, src_idx, dst_idx, cons = entry
            ep = None
        else:
            et, src_idx, dst_idx = entry
            cons = 1.0
            ep = None
        if src_idx < len(assignments) and dst_idx < len(assignments):
            s_nid = assignments[src_idx]
            d_nid = assignments[dst_idx]
            if s_nid is not None and d_nid is not None:
                if et in UNDIRECTED_EDGE_TYPES:
                    key = (min(s_nid, d_nid), max(s_nid, d_nid), et)
                else:
                    key = (s_nid, d_nid, et)
                mat_edges.add(key)
                mat_edge_weights[key] = float(cons)
                if ep is not None:
                    mat_edge_params[key] = ep

    return Subgraph(
        nodes=mat_nodes,
        edges=frozenset(mat_edges),
        edge_params=mat_edge_params,
        weights=mat_edge_weights,
        node_weights=mat_node_weights,
        anchor_of_node=anchor_positions_for(
            mat_nodes, exemplar.anchor_of_node
        ),
    )


# [H.4b] Compatibility alias. Historical name resolves to the redesigned
# matcher; any external caller using the old name gets the new behaviour.
_hungarian_max_bipartite = _max_weight_bipartite_matching
