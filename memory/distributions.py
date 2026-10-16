"""memory/distributions.py -- extracted from consolidator.py (Move 5a).

MOVE ONLY: whole top-level statements, verbatim.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections import Counter
from dataclasses import dataclass
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import parameters as P

# [defect fix] These three constants were defined in consolidator.py but
# used here; importing them back would be circular (consolidator imports
# this module), so they are defined here and re-exported by consolidator.

# Phase A Step 2 -- bound at load to avoid hot-path P.* lookups.
_GRADED_FALLBACK_SCALE = P.GRADED_FALLBACK_SCALE
_MAX_GRADED_EDGES_PER_IMAGE = P.MAX_GRADED_EDGES_PER_IMAGE

# FIX-CATCH (Phase A review): cap-hit sentinel key. Stored inside
# _SigIndex._sim_cache under a non-bytes key so it can never collide
# with a real (bytes, bytes) similarity entry.
_CAP_HIT_SENTINEL = b"\x00phase_a_edge_cap_hits\x00"

# --------------------------------------------------------------------------
# [Step B1] Evidence-weighting swap: log-likelihood ratio (Approach A)
#          at every site where raw counts are available, bounded relative
#          evidence (Approach B) at the two probability-only sites.
#
# Why not log(pf/nf) directly: a zero count gives log(0) = -inf. Jeffreys
# prior (+0.5 on each side) bounds the ratio and, for n_pos == n_neg, keeps
# the sign of pf - nf, which is what the exact-first guard relies on.
# Count sensitivity is the point: a feature seen 3 times no longer scores
# like one seen 40 times, which the plain difference could not express.

# [Step B1] Jeffreys smoothing constant for the count-based LLR.
_LLR_PRIOR: float = 0.5

# [Step B1] Epsilon for Approach B (probabilities-only sites).
BOUNDED_EVIDENCE_EPS: float = 1e-6


def _llr_from_counts(pos_count, neg_count, n_pos, n_neg) -> float:
    """Log-likelihood ratio of two rates, Jeffreys-smoothed."""
    import math
    pr = (pos_count + _LLR_PRIOR) / (float(max(1, n_pos)) + 1.0)
    nr = (neg_count + _LLR_PRIOR) / (float(max(1, n_neg)) + 1.0)
    return float(math.log(pr / nr))


def _bounded_rel_evidence(pf, nf) -> float:
    """Approach B for sites holding probabilities only (no counts)."""
    return (pf - nf) / (pf + nf + BOUNDED_EVIDENCE_EPS)


# ---------------------------------------------------------------------------
# [ITEM 3 / K2] ONE shared mass function for every mass site.
#
# RULE K2-LCB -- pre-registered 2026-10-16, BEFORE any re-measurement result
# was viewed: a core element receives mass only when its evidence beats
# chance, i.e. the one-sided 95% lower Wilson bound of the positive rate
# minus the one-sided 95% upper Wilson bound of the negative rate is > 0
# (z = 1.645, a conventional confidence level -- never swept). The cutoff is
# a function of the sample sizes alone; there is no tuned threshold.
#
# Scoring-facing: shipped DARK behind P.MASS_LCB_GATE (default False),
# label UNCONFIRMED -- one object per class, views only.
# ---------------------------------------------------------------------------
_LCB_Z: float = 1.645  # one-sided 95%; conventional level, never tuned.


def _wilson_bound(p_hat: float, n: int, z: float, upper: bool) -> float:
    """Wilson score interval bound (handles p=0 and p=1 without collapse)."""
    if n <= 0:
        return 1.0 if upper else 0.0
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(
        p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))
    return center + half if upper else center - half


def _lcb_beats_chance(pos_count: int, neg_count: int, n_pos: int, n_neg: int,
                      z: Optional[float] = None) -> bool:
    """RULE K2-LCB: Wilson-lower(p_pos) - Wilson-upper(p_neg) > 0."""
    if z is None:
        z = _LCB_Z
    lp = _wilson_bound(pos_count / float(max(1, n_pos)), max(1, n_pos), z,
                       upper=False)
    un = _wilson_bound(neg_count / float(max(1, n_neg)), max(1, n_neg), z,
                       upper=True)
    return (lp - un) > 0.0


def _shared_mass(disc, prior, exponent) -> float:
    """THE mass function: mass = max(0, disc)**exponent * prior.

    Contract (v32.2, unchanged): disc <= 0 or prior <= 0 -> 0.0; empty prior
    map is resolved by the caller to prior = 1.0 before calling here."""
    if disc is None or disc <= 0.0:
        return 0.0
    if prior is None or prior <= 0.0:
        return 0.0
    return (float(disc) ** exponent) * float(prior)


def _mass_from_counts(pos_count, neg_count, n_pos, n_neg, disc, prior,
                      exponent) -> float:
    """Single entry point used by ALL mass sites (the four inline sites in
    _content_discriminative_filter, _mass_for_key, and the two extension
    mass helpers): optional RULE K2-LCB gate (dark flag, live read so tests
    can toggle without reimport) followed by _shared_mass."""
    if getattr(P, "MASS_LCB_GATE", False):
        if not _lcb_beats_chance(pos_count, neg_count, n_pos, n_neg):
            return 0.0
    return _shared_mass(disc, prior, exponent)

from core.graph_types import (  # noqa: F401
    Subgraph,
)
from core.types import (  # noqa: F401
    EdgeType,
    TypedSignature,
    UNDIRECTED_EDGE_TYPES,
)

def _mass_for_key(key, pos_counter, neg_counter, n_pos, n_neg, exponent):
    """Compute (mass, disc). mass via the ONE shared mass function
    (_mass_from_counts: RULE K2-LCB gate when dark flag on, else plain);
    disc = Jeffreys-smoothed LLR of the two rates. Caller multiplies mass
    by any consistency prior (prior = 1.0 here)."""
    # [Step B1] LLR from raw counts (was: pf - nf)
    d = _llr_from_counts(
        pos_counter.get(key, 0),
        neg_counter.get(key, 0),
        n_pos, n_neg,
    )
    mass = _mass_from_counts(
        pos_counter.get(key, 0), neg_counter.get(key, 0),
        n_pos, n_neg, d, 1.0, exponent,
    )
    return mass, d

@dataclass
class _SigIndex:
    pos_count: Dict[Tuple[str, Tuple[float, ...]], int]
    neg_count: Dict[Tuple[str, Tuple[float, ...]], int]
    pos_sig: Dict[Tuple[str, Tuple[float, ...]], TypedSignature]
    neg_sig: Dict[Tuple[str, Tuple[float, ...]], TypedSignature]
    pos_sigs_by_lt: Dict[str, List[List[TypedSignature]]]
    neg_sigs_by_lt: Dict[str, List[List[TypedSignature]]]
    pos_edge_count: Dict[Tuple, int]
    neg_edge_count: Dict[Tuple, int]
    pos_edge_sigs: Dict[Tuple, Tuple[TypedSignature, TypedSignature]]
    neg_edge_sigs: Dict[Tuple, Tuple[TypedSignature, TypedSignature]]
    pos_edge_sigs_by_et: Dict[EdgeType, List[List[Tuple[TypedSignature, TypedSignature]]]]
    neg_edge_sigs_by_et: Dict[EdgeType, List[List[Tuple[TypedSignature, TypedSignature]]]]
    n_pos: int
    n_neg: int
    _sim_cache: Dict[Any, float] = field(default_factory=dict)

    def sim_cached(self, a: TypedSignature, b: TypedSignature) -> float:
        # FIX-CATCH vs patch draft: canonicalize order so (a,b) and (b,a)
        # share one cache entry (similarity is symmetric).
        sa, sb = a.serialize(), b.serialize()
        key = (sa, sb) if sa <= sb else (sb, sa)
        v = self._sim_cache.get(key)
        if v is None:
            v = a.similarity(b)
            self._sim_cache[key] = v
        return v

def _build_sig_index(
    positives: List[Subgraph],
    negatives: List[Subgraph],
    sig_lookup: SignatureProvider,
) -> _SigIndex:
    pos_count: Dict[Tuple[str, Tuple[float, ...]], int] = defaultdict(int)
    neg_count: Dict[Tuple[str, Tuple[float, ...]], int] = defaultdict(int)
    pos_sig: Dict[Tuple[str, Tuple[float, ...]], TypedSignature] = {}
    neg_sig: Dict[Tuple[str, Tuple[float, ...]], TypedSignature] = {}
    pos_sigs_by_lt: Dict[str, List[List[TypedSignature]]] = defaultdict(list)
    neg_sigs_by_lt: Dict[str, List[List[TypedSignature]]] = defaultdict(list)

    for p in positives:
        per_lt: Dict[str, List[TypedSignature]] = defaultdict(list)
        seen: Set[Tuple[str, Tuple[float, ...]]] = set()
        for nid in p.nodes:
            info = sig_lookup(nid)
            if info is None:
                continue
            lt, sig = info
            key = (lt, tuple(float(x) for x in sig.quantized))
            if key in seen:
                continue
            seen.add(key)
            per_lt[lt].append(sig)
            pos_count[key] += 1
            pos_sig[key] = sig
        for lt, sigs in per_lt.items():
            pos_sigs_by_lt[lt].append(sigs)

    for n in negatives:
        per_lt_n: Dict[str, List[TypedSignature]] = defaultdict(list)
        seen_n: Set[Tuple[str, Tuple[float, ...]]] = set()
        for nid in n.nodes:
            info = sig_lookup(nid)
            if info is None:
                continue
            lt, sig = info
            key = (lt, tuple(float(x) for x in sig.quantized))
            if key in seen_n:
                continue
            seen_n.add(key)
            per_lt_n[lt].append(sig)
            neg_count[key] += 1
            neg_sig[key] = sig
        for lt, sigs in per_lt_n.items():
            neg_sigs_by_lt[lt].append(sigs)

    pos_edge_count: Dict[Tuple, int] = defaultdict(int)
    neg_edge_count: Dict[Tuple, int] = defaultdict(int)
    pos_edge_sigs: Dict[Tuple, Tuple[TypedSignature, TypedSignature]] = {}
    neg_edge_sigs: Dict[Tuple, Tuple[TypedSignature, TypedSignature]] = {}
    pos_edge_sigs_by_et: Dict[EdgeType, List[List[Tuple[TypedSignature, TypedSignature]]]] = defaultdict(list)
    neg_edge_sigs_by_et: Dict[EdgeType, List[List[Tuple[TypedSignature, TypedSignature]]]] = defaultdict(list)

    def _edge_key_and_pair(s, d, et, sig_lookup):
        info_s = sig_lookup(s)
        info_d = sig_lookup(d)
        if info_s is None or info_d is None:
            return None, None
        s_lt, s_sig = info_s
        d_lt, d_sig = info_d
        s_key = (s_lt, tuple(float(x) for x in s_sig.quantized))
        d_key = (d_lt, tuple(float(x) for x in d_sig.quantized))
        sig_pair = (s_sig, d_sig)
        if et in UNDIRECTED_EDGE_TYPES and s_key > d_key:
            s_key, d_key = d_key, s_key
            sig_pair = (d_sig, s_sig)
        return (et, s_key, d_key), sig_pair

    for p in positives:
        seen_edges: Set[Tuple] = set()
        per_et: Dict[EdgeType, List[Tuple[TypedSignature, TypedSignature]]] = defaultdict(list)
        for (s, d, et) in p.edges:
            e_key, sig_pair = _edge_key_and_pair(s, d, et, sig_lookup)
            if e_key is None:
                continue
            if e_key in seen_edges:
                continue
            seen_edges.add(e_key)
            pos_edge_count[e_key] += 1
            pos_edge_sigs[e_key] = sig_pair
            per_et[et].append(sig_pair)
        for et, sigs in per_et.items():
            pos_edge_sigs_by_et[et].append(sigs)

    for n in negatives:
        seen_edges_n: Set[Tuple] = set()
        per_et_n: Dict[EdgeType, List[Tuple[TypedSignature, TypedSignature]]] = defaultdict(list)
        for (s, d, et) in n.edges:
            e_key, sig_pair = _edge_key_and_pair(s, d, et, sig_lookup)
            if e_key is None:
                continue
            if e_key in seen_edges_n:
                continue
            seen_edges_n.add(e_key)
            neg_edge_count[e_key] += 1
            neg_edge_sigs[e_key] = sig_pair
            per_et_n[et].append(sig_pair)
        for et, sigs in per_et_n.items():
            neg_edge_sigs_by_et[et].append(sigs)

    return _SigIndex(
        pos_count=pos_count, neg_count=neg_count,
        pos_sig=pos_sig, neg_sig=neg_sig,
        pos_sigs_by_lt=dict(pos_sigs_by_lt),
        neg_sigs_by_lt=dict(neg_sigs_by_lt),
        pos_edge_count=pos_edge_count, neg_edge_count=neg_edge_count,
        pos_edge_sigs=pos_edge_sigs, neg_edge_sigs=neg_edge_sigs,
        pos_edge_sigs_by_et=dict(pos_edge_sigs_by_et),
        neg_edge_sigs_by_et=dict(neg_edge_sigs_by_et),
        n_pos=len(positives), n_neg=len(negatives),
        _sim_cache={},
    )

def _filter_disc_node(
    core_sig: TypedSignature,
    core_lt: str,
    idx: _SigIndex,
) -> Tuple[float, float, float, str]:
    """Return (disc_exact, disc_graded, disc_used, which_won)."""
    exact_key = (core_lt, tuple(float(x) for x in core_sig.quantized))
    # [Step B1] LLR from raw counts (was: pf_e - nf_e)
    disc_exact = _llr_from_counts(
        idx.pos_count.get(exact_key, 0),
        idx.neg_count.get(exact_key, 0),
        idx.n_pos, idx.n_neg,
    )
    pos_sum = 0.0
    for sigs in idx.pos_sigs_by_lt.get(core_lt, ()):
        best = 0.0
        for sig in sigs:
            s = idx.sim_cached(core_sig, sig)
            if s > best:
                best = s
        pos_sum += best
    pf_g = pos_sum / max(1, idx.n_pos)
    neg_sum = 0.0
    for sigs in idx.neg_sigs_by_lt.get(core_lt, ()):
        best = 0.0
        for sig in sigs:
            s = idx.sim_cached(core_sig, sig)
            if s > best:
                best = s
        neg_sum += best
    nf_g = neg_sum / max(1, idx.n_neg)
    disc_graded = pf_g - nf_g
    if disc_exact != 0.0:
        return disc_exact, disc_graded, disc_exact, "exact"
    # FIX-CATCH: non-positive graded fallback must yield 0.0 used, never a
    # negative mass base (float(neg)**exp would be complex/nan downstream).
    if disc_graded <= 0.0:
        return disc_exact, disc_graded, 0.0, "graded_fallback"
    return disc_exact, disc_graded, disc_graded * _GRADED_FALLBACK_SCALE, "graded_fallback"

def _filter_disc_edge(
    core_src_nid: int,
    core_dst_nid: int,
    core_et: EdgeType,
    sig_lookup: SignatureProvider,
    idx: _SigIndex,
) -> Tuple[float, float, float, str]:
    """Return (disc_exact, disc_graded, disc_used, which_won)."""
    info_s = sig_lookup(core_src_nid)
    info_d = sig_lookup(core_dst_nid)
    if info_s is None or info_d is None:
        return 0.0, 0.0, 0.0, "unresolvable"
    s_lt, s_sig = info_s
    d_lt, d_sig = info_d
    s_key = (s_lt, tuple(float(x) for x in s_sig.quantized))
    d_key = (d_lt, tuple(float(x) for x in d_sig.quantized))
    ordered_src, ordered_dst = s_sig, d_sig
    if core_et in UNDIRECTED_EDGE_TYPES and s_key > d_key:
        s_key, d_key = d_key, s_key
        ordered_src, ordered_dst = d_sig, s_sig
    exact_key = (core_et, s_key, d_key)
    # [Step B1] LLR from raw counts (was: pf_e - nf_e)
    disc_exact = _llr_from_counts(
        idx.pos_edge_count.get(exact_key, 0),
        idx.neg_edge_count.get(exact_key, 0),
        idx.n_pos, idx.n_neg,
    )
    cap = _MAX_GRADED_EDGES_PER_IMAGE
    pos_sum = 0.0
    for img_edges in idx.pos_edge_sigs_by_et.get(core_et, ()):
        best = 0.0
        # FIX-CATCH: deterministic truncation (first-N insertion order) and
        # cap-hit accounting via sentinel counter on the shared cache.
        if cap is not None and len(img_edges) > cap:
            idx._sim_cache[_CAP_HIT_SENTINEL] = (
                idx._sim_cache.get(_CAP_HIT_SENTINEL, 0.0) + 1.0
            )
            img_edges = img_edges[:cap]
        for (es, ed) in img_edges:
            sim_s = idx.sim_cached(ordered_src, es)
            if sim_s <= 0.0:
                continue
            sim_d = idx.sim_cached(ordered_dst, ed)
            if sim_d <= 0.0:
                continue
            score = sim_s * sim_d
            if score > best:
                best = score
        pos_sum += best
    pf_g = pos_sum / max(1, idx.n_pos)
    neg_sum = 0.0
    for img_edges in idx.neg_edge_sigs_by_et.get(core_et, ()):
        best = 0.0
        if cap is not None and len(img_edges) > cap:
            idx._sim_cache[_CAP_HIT_SENTINEL] = (
                idx._sim_cache.get(_CAP_HIT_SENTINEL, 0.0) + 1.0
            )
            img_edges = img_edges[:cap]
        for (es, ed) in img_edges:
            sim_s = idx.sim_cached(ordered_src, es)
            if sim_s <= 0.0:
                continue
            sim_d = idx.sim_cached(ordered_dst, ed)
            if sim_d <= 0.0:
                continue
            score = sim_s * sim_d
            if score > best:
                best = score
        neg_sum += best
    nf_g = neg_sum / max(1, idx.n_neg)
    disc_graded = pf_g - nf_g
    if disc_exact != 0.0:
        return disc_exact, disc_graded, disc_exact, "exact"
    if disc_graded <= 0.0:
        return disc_exact, disc_graded, 0.0, "graded_fallback"
    return disc_exact, disc_graded, disc_graded * _GRADED_FALLBACK_SCALE, "graded_fallback"

def _compute_node_mass(
    nid: int,
    sig_lookup: SignatureProvider,
    sig_index: Optional[_SigIndex],
    filter_counters: Dict,
    disc_exponent: float,
    dual_enabled: bool,
) -> float:
    """Mass for an extension-added node (prior = 1.0; no materialization prior)."""
    info = sig_lookup(nid)
    if info is None:
        return 0.0
    core_lt, core_sig = info
    k = (core_lt, tuple(float(x) for x in core_sig.quantized))
    if dual_enabled and sig_index is not None:
        _de, _dg, disc_used, _ww = _filter_disc_node(core_sig, core_lt, sig_index)
        # [ITEM 3] shared mass site (RULE K2-LCB gate when dark flag on).
        return _mass_from_counts(
            sig_index.pos_count.get(k, 0), sig_index.neg_count.get(k, 0),
            sig_index.n_pos, sig_index.n_neg, disc_used, 1.0, disc_exponent,
        )
    # [Step B1] LLR from raw counts (was: pf - nf). This fallback is the
    # LIVE mass path whenever sig_index is None, which is why the first
    # attempt (three named functions only) produced a measurable no-op.
    disc = _llr_from_counts(
        filter_counters["pos_node"].get(k, 0),
        filter_counters["neg_node"].get(k, 0),
        filter_counters["n_pos"], filter_counters["n_neg"],
    )
    return _mass_from_counts(
        filter_counters["pos_node"].get(k, 0),
        filter_counters["neg_node"].get(k, 0),
        filter_counters["n_pos"], filter_counters["n_neg"],
        disc, 1.0, disc_exponent,
    )

def _compute_edge_mass(
    e: Tuple[int, int, EdgeType],
    sig_lookup: SignatureProvider,
    sig_index: Optional[_SigIndex],
    filter_counters: Dict,
    disc_exponent: float,
    dual_enabled: bool,
) -> float:
    """Mass for an extension-added edge (prior = 1.0)."""
    s, d, et = e
    if dual_enabled and sig_index is not None:
        info_s = sig_lookup(s)
        info_d = sig_lookup(d)
        if info_s is None or info_d is None:
            return 0.0
        s_lt, s_sig = info_s
        d_lt, d_sig = info_d
        s_key = (s_lt, tuple(float(x) for x in s_sig.quantized))
        d_key = (d_lt, tuple(float(x) for x in d_sig.quantized))
        if et in UNDIRECTED_EDGE_TYPES and s_key > d_key:
            s_key, d_key = d_key, s_key
        k = (et, s_key, d_key)
        _de, _dg, disc_used, _ww = _filter_disc_edge(s, d, et, sig_lookup, sig_index)
        # [ITEM 3] shared mass site (RULE K2-LCB gate when dark flag on).
        return _mass_from_counts(
            sig_index.pos_edge_count.get(k, 0),
            sig_index.neg_edge_count.get(k, 0),
            sig_index.n_pos, sig_index.n_neg, disc_used, 1.0, disc_exponent,
        )
    info_s = sig_lookup(s)
    info_d = sig_lookup(d)
    if info_s is None or info_d is None:
        return 0.0
    s_lt, s_sig = info_s
    d_lt, d_sig = info_d
    s_key = (s_lt, tuple(float(x) for x in s_sig.quantized))
    d_key = (d_lt, tuple(float(x) for x in d_sig.quantized))
    if et in UNDIRECTED_EDGE_TYPES and s_key > d_key:
        s_key, d_key = d_key, s_key
    k = (et, s_key, d_key)
    # [Step B1] LLR from raw counts (was: pf - nf); live fallback path
    disc = _llr_from_counts(
        filter_counters["pos_edge"].get(k, 0),
        filter_counters["neg_edge"].get(k, 0),
        filter_counters["n_pos"], filter_counters["n_neg"],
    )
    return _mass_from_counts(
        filter_counters["pos_edge"].get(k, 0),
        filter_counters["neg_edge"].get(k, 0),
        filter_counters["n_pos"], filter_counters["n_neg"],
        disc, 1.0, disc_exponent,
    )

def _dist(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"n": 0, "min": 0.0, "median": 0.0, "mean": 0.0,
                "max": 0.0, "p90": 0.0}
    sv = sorted(vals)
    n = len(sv)
    p90_idx = n - 1 if n < 10 else int(0.9 * (n - 1))
    return {
        "n": n, "min": float(sv[0]), "median": float(sv[n // 2]),
        "mean": float(sum(sv) / n), "max": float(sv[-1]),
        "p90": float(sv[p90_idx]),
    }

def _build_df_content_distribution(
    subgraphs: List[Subgraph],
    sig_lookup: SignatureProvider,
) -> Dict[str, Dict[Tuple[float, ...], float]]:
    """[v32.4] Document-frequency histogram per lens type: for each lens
    type and signature tuple, the fraction of subgraphs containing it."""
    from collections import Counter
    per_image: List[Dict[str, set]] = []
    for sg in subgraphs:
        img_seen: Dict[str, set] = {}
        for nid in sg.nodes:
            info = sig_lookup(nid)
            if info is None:
                continue
            lt, sig = info
            key = tuple(float(x) for x in sig.quantized)
            img_seen.setdefault(lt, set()).add(key)
        per_image.append(img_seen)
    n = max(1, len(per_image))
    lens_keys: Dict[str, Counter] = {}
    for img_seen in per_image:
        for lt, keys in img_seen.items():
            for k in keys:
                lens_keys.setdefault(lt, Counter())[k] += 1
    return {lt: {k: c / n for k, c in counter.items()}
            for lt, counter in lens_keys.items()}

def _build_df_structural_distribution(
    subgraphs: List[Subgraph],
    sig_lookup: SignatureProvider,
) -> Dict[EdgeType, Dict[Tuple, float]]:
    """[v32.4] DF histogram per edge type over signature-inclusive keys.
    Undirected edges canonicalized by tuple comparison of endpoint keys."""
    from collections import Counter
    per_image: List[Dict[EdgeType, set]] = []
    for sg in subgraphs:
        img_seen: Dict[EdgeType, set] = {}
        for (s, d, et) in sg.edges:
            info_s = sig_lookup(s)
            info_d = sig_lookup(d)
            if info_s is None or info_d is None:
                continue
            s_k = (info_s[0], tuple(float(x) for x in info_s[1].quantized))
            d_k = (info_d[0], tuple(float(x) for x in info_d[1].quantized))
            if et in UNDIRECTED_EDGE_TYPES:
                key = (et, s_k, d_k) if s_k <= d_k else (et, d_k, s_k)
            else:
                key = (et, s_k, d_k)
            img_seen.setdefault(et, set()).add(key)
        per_image.append(img_seen)
    n = max(1, len(per_image))
    edge_keys: Dict[EdgeType, Counter] = {}
    for img_seen in per_image:
        for et, keys in img_seen.items():
            for k in keys:
                edge_keys.setdefault(et, Counter())[k] += 1
    return {et: {k: c / n for k, c in counter.items()}
            for et, counter in edge_keys.items()}

def _build_discriminative_content_distribution(
    pos_dist: Dict[str, Dict[Tuple[float, ...], float]],
    neg_dist: Dict[str, Dict[Tuple[float, ...], float]],
    mode: str = None,
) -> Dict:
    """[v32.5] Discriminability-weighted class content distribution.

    Returns:
      per_lens                          -> Dict[lens_type, Dict[key, weight]]
      global / log / class_global       -> Dict['__global__',
                                                 Dict[(lens_type, key), weight]]
    """
    if mode is None:
        mode = P.DIST_NORM_MODE
    if mode not in P.VALID_DIST_NORM_MODES:
        raise ValueError(f"unknown distribution mode: {mode!r}")

    raw: Dict[str, Dict[Tuple[float, ...], float]] = {}
    for lt in pos_dist:
        for key, pf in pos_dist[lt].items():
            nf = neg_dist.get(lt, {}).get(key, 0.0)
            # [Step B1/Approach B] probabilities only here (already-df
            # normalised), so bounded relative evidence, not LLR:
            # a zero-count feature no longer scores like a certain one.
            d = _bounded_rel_evidence(pf, nf)
            if d > 0.0:
                raw.setdefault(lt, {})[key] = d

    if mode == "per_lens":
        out: Dict[str, Dict[Tuple[float, ...], float]] = {}
        for lt, weights in raw.items():
            tot = sum(weights.values())
            if tot > 0.0:
                out[lt] = {k: v / tot for k, v in weights.items()}
        return out

    flat: Dict[Tuple[str, Tuple[float, ...]], float] = {}
    for lt, weights in raw.items():
        for k, v in weights.items():
            flat[(lt, k)] = math.log(1.0 + v) if mode == "log" else v
    tot = sum(flat.values())
    if tot <= 0.0:
        return {}
    return {"__global__": {k: v / tot for k, v in flat.items()}}

def _build_discriminative_structural_distribution(
    pos_dist: Dict[EdgeType, Dict[Tuple, float]],
    neg_dist: Dict[EdgeType, Dict[Tuple, float]],
    mode: str = None,
) -> Dict:
    """[v32.5] Same as content; structural keys are
    (EdgeType, (src_lens, src_quant), (dst_lens, dst_quant)) in the
    flattened __global__ form."""
    if mode is None:
        mode = P.DIST_NORM_MODE
    if mode not in P.VALID_DIST_NORM_MODES:
        raise ValueError(f"unknown distribution mode: {mode!r}")

    raw: Dict[EdgeType, Dict[Tuple, float]] = {}
    for et in pos_dist:
        for key, pf in pos_dist[et].items():
            nf = neg_dist.get(et, {}).get(key, 0.0)
            # [Step B1/Approach B] probabilities only here
            d = _bounded_rel_evidence(pf, nf)
            if d > 0.0:
                raw.setdefault(et, {})[key] = d

    if mode == "per_lens":
        out: Dict[EdgeType, Dict[Tuple, float]] = {}
        for et, weights in raw.items():
            tot = sum(weights.values())
            if tot > 0.0:
                out[et] = {k: v / tot for k, v in weights.items()}
        return out

    flat: Dict[Tuple[EdgeType, Tuple], float] = {}
    for et, weights in raw.items():
        for k, v in weights.items():
            flat[(et, k)] = math.log(1.0 + v) if mode == "log" else v
    tot = sum(flat.values())
    if tot <= 0.0:
        return {}
    return {"__global__": {k: v / tot for k, v in flat.items()}}

def build_discriminative_content_distribution(
    positives: List[Subgraph],
    negatives: List[Subgraph],
    sig_lookup: SignatureProvider,
) -> Dict[str, Dict[Tuple[float, ...], float]]:
    """[v32.4] Public wrapper: DF for positives and negatives, then
    discriminability-weighted and normalized per lens type."""
    pos_df = _build_df_content_distribution(positives, sig_lookup)
    neg_df = _build_df_content_distribution(negatives, sig_lookup)
    return _build_discriminative_content_distribution(pos_df, neg_df)

def build_discriminative_structural_distribution(
    positives: List[Subgraph],
    negatives: List[Subgraph],
    sig_lookup: SignatureProvider,
) -> Dict[EdgeType, Dict[Tuple, float]]:
    pos_df = _build_df_structural_distribution(positives, sig_lookup)
    neg_df = _build_df_structural_distribution(negatives, sig_lookup)
    return _build_discriminative_structural_distribution(pos_df, neg_df)

def _build_query_content_distribution(
    q_content: Subgraph,
    sig_lookup: SignatureProvider,
    mode: str = None,
) -> Dict:
    if mode is None:
        mode = P.DIST_NORM_MODE
    if mode not in P.VALID_DIST_NORM_MODES:
        raise ValueError(f"unknown distribution mode: {mode!r}")

    from collections import Counter

    # Presence semantics, not multiplicity. Content-tier subgraphs already
    # collapse identical signatures to one node.
    counts: Dict[str, Counter] = {}
    for nid in q_content.nodes:
        info = sig_lookup(nid)
        if info is None:
            continue
        lt, sig = info
        key = tuple(float(x) for x in sig.quantized)
        counts.setdefault(lt, Counter())[key] += 1

    if mode == "per_lens":
        out: Dict[str, Dict[Tuple[float, ...], float]] = {}
        for lt, counter in counts.items():
            tot = sum(counter.values())
            if tot > 0:
                out[lt] = {k: v / tot for k, v in counter.items()}
        return out

    if mode == "class_global":
        # Flattened __global__ shape, per-lens normalized values.
        flat: Dict[Tuple[str, Tuple[float, ...]], float] = {}
        for lt, counter in counts.items():
            tot = sum(counter.values())
            if tot <= 0:
                continue
            for k, v in counter.items():
                flat[(lt, k)] = v / tot
        if not flat:
            return {}
        return {"__global__": flat}

    # global / log query: flat presence, globally normalized.
    flat_g: Dict[Tuple[str, Tuple[float, ...]], float] = {}
    for nid in q_content.nodes:
        info = sig_lookup(nid)
        if info is None:
            continue
        lt, sig = info
        key = (lt, tuple(float(x) for x in sig.quantized))
        flat_g[key] = 1.0
    tot = sum(flat_g.values())
    if tot <= 0.0:
        return {}
    return {"__global__": {k: v / tot for k, v in flat_g.items()}}

def _build_query_structural_distribution(
    q_content: Subgraph,
    sig_lookup: SignatureProvider,
    mode: str = None,
) -> Dict:
    if mode is None:
        mode = P.DIST_NORM_MODE
    if mode not in P.VALID_DIST_NORM_MODES:
        raise ValueError(f"unknown distribution mode: {mode!r}")

    from collections import Counter

    def _struct_key(s: int, d: int, et: EdgeType):
        info_s = sig_lookup(s)
        info_d = sig_lookup(d)
        if info_s is None or info_d is None:
            return None
        s_k = (info_s[0], tuple(float(x) for x in info_s[1].quantized))
        d_k = (info_d[0], tuple(float(x) for x in info_d[1].quantized))
        if et in UNDIRECTED_EDGE_TYPES:
            return (s_k, d_k) if s_k <= d_k else (d_k, s_k)
        return (s_k, d_k)

    counts: Dict[EdgeType, Counter] = {}
    for (s, d, et) in q_content.edges:
        sk = _struct_key(s, d, et)
        if sk is None:
            continue
        counts.setdefault(et, Counter())[sk] += 1

    if mode == "per_lens":
        out: Dict[EdgeType, Dict[Tuple, float]] = {}
        for et, counter in counts.items():
            tot = sum(counter.values())
            if tot > 0:
                out[et] = {k: v / tot for k, v in counter.items()}
        return out

    if mode == "class_global":
        flat: Dict[Tuple[EdgeType, Tuple], float] = {}
        for et, counter in counts.items():
            tot = sum(counter.values())
            if tot <= 0:
                continue
            for k, v in counter.items():
                flat[(et, k)] = v / tot
        if not flat:
            return {}
        return {"__global__": flat}

    flat_g: Dict[Tuple[EdgeType, Tuple], float] = {}
    for et, counter in counts.items():
        for k in counter.keys():
            flat_g[(et, k)] = 1.0
    tot = sum(flat_g.values())
    if tot <= 0.0:
        return {}
    return {"__global__": {k: v / tot for k, v in flat_g.items()}}

def _distribution_intersection(q_dist: Dict, c_dist: Dict) -> float:
    """[v32.4] Histogram intersection between normalized per-channel
    distributions, averaged over channels present in c_dist. Channels
    present in c_dist but absent from q_dist contribute 0 to the numerator
    but still count in the denominator (coverage penalty)."""
    if not c_dist:
        return 0.0
    total = 0.0
    for top_key in sorted(c_dist.keys(), key=str):
        if top_key not in q_dist:
            continue
        q_lt = q_dist[top_key]
        c_lt = c_dist[top_key]
        if not c_lt:
            continue
        for k in sorted(set(q_lt) & set(c_lt), key=str):
            qv = q_lt[k]
            cv = c_lt[k]
            total += qv if qv < cv else cv
    return total / len(c_dist)

def classify_by_distribution(
    q_content: Subgraph,
    q_sig_lookup: SignatureProvider,
    class_content_dists,
    class_struct_dists,
    mode: str = None,
) -> Dict[str, Dict[str, float]]:
    if mode is None:
        mode = P.DIST_NORM_MODE
    if mode not in P.VALID_DIST_NORM_MODES:
        raise ValueError(f"unknown distribution mode: {mode!r}")

    q_content_dist = _build_query_content_distribution(
        q_content, q_sig_lookup, mode=mode
    )
    q_struct_dist = _build_query_structural_distribution(
        q_content, q_sig_lookup, mode=mode
    )
    content_scores: Dict[str, float] = {}
    struct_scores: Dict[str, float] = {}
    for cls, c_dist in class_content_dists.items():
        content_scores[cls] = _distribution_intersection(q_content_dist, c_dist)
    for cls, s_dist in class_struct_dists.items():
        struct_scores[cls] = _distribution_intersection(q_struct_dist, s_dist)
    combined: Dict[str, float] = {}
    for cls in content_scores:
        combined[cls] = (
            0.5 * content_scores[cls] + 0.5 * struct_scores.get(cls, 0.0)
        )
    return {"content": content_scores, "structural": struct_scores,
            "combined": combined}

