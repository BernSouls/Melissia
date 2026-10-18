"""consolidator.py -- COMPATIBILITY SHELL (reorg Move 5a).

Aspect-graph machinery moved to `memory/aspects.py`; the distribution
engine to `memory/distributions.py`. The `Consolidator` class stays here
for now (Moves 5b/5c continue the split). Everything moved is
re-exported so existing imports keep working.

Original preserved in consolidator.py.pre_move5a.
"""
from __future__ import annotations

from itertools import chain
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)
import parameters as P
from core.hashing import (  # noqa: F401
    clamp,
    hash_128,
    hash_64,
    serialize_bytes,
    serialize_int,
    serialize_tuple,
    serialize_value,
)
from core.registries import CONSTITUTIONAL_RANGES  # noqa: F401
from core.schema import coarsen_edge_params_for_wl  # noqa: F401
from sensory_floor import TypedSignature, ribbon_bins
from core.hashing import _node_signature_hash
from core.types import (
    EdgeType,
    REPRODUCIBLE_EDGE_TYPES,
    UNDIRECTED_EDGE_TYPES,
)
from concept_graph import Concept, ConceptStatus, ConceptFamily
from core.graph_types import Subgraph, anchor_positions_for, edge_params_for
from memory.content import ContentSignatureStore, build_content_subgraph
from matching.structural_similarity import (
    StructuralTemplate,
    structural_intersection as _structural_intersection,
    materialize_template as _materialize,
    _max_weight_bipartite_matching as _max_weight_match,
    GESTALT_LENS_BOOST,
)


# --- moved surface, re-exported (Move 5a) ---
from memory.aspects import (  # noqa: F401
    AspectGraph,
    PartDesc,
    ViewRecord,
    _ENSEMBLE_R,
    _RIBBON_R,
    _awrap,
    _bins_sim,
    _dOmega,
    _greedy_match,
    build_view_descriptor,
    build_view_descriptor_from_chains,
    match_views,
)

from memory.distributions import (  # noqa: F401
    _CAP_HIT_SENTINEL,
    _GRADED_FALLBACK_SCALE,
    _MAX_GRADED_EDGES_PER_IMAGE,
    _SigIndex,
    _build_df_content_distribution,
    _build_df_structural_distribution,
    _build_discriminative_content_distribution,
    _build_discriminative_structural_distribution,
    _build_query_content_distribution,
    _build_query_structural_distribution,
    _build_sig_index,
    _compute_edge_mass,
    _compute_node_mass,
    _dist,
    _distribution_intersection,
    _filter_disc_edge,
    _filter_disc_node,
    _mass_for_key,
    _mass_from_counts,
    _wilson_min_recurrence,
    build_discriminative_content_distribution,
    build_discriminative_structural_distribution,
    classify_by_distribution,
)





# --- moved to memory/resonance.py (Move 5b), re-exported ---
from memory.resonance import (  # noqa: F401
    CoreKind,
    CorePayload_Action,
    CorePayload_Category,
    CoreStatus,
    ReductionTemplate,
    _merge_distribution,
    vigilance_candidates,
)


# --- moved to memory/cores.py (Move 5c), re-exported ---
from memory.cores import (  # noqa: F401
    CORE_SIZE_CAP_EDGES,
    CORE_SIZE_CAP_NODES,
    CategoryLabel,
    CategoryManager,
    CoreEntry,
    CoreLibrary,
    HISTORY_MAXLEN,
    InvertedTypeIndex,
    KAPPA_RELIABILITY,
    K_CONSOLIDATE,
    M_PLATEAU,
    N_DEMOTE,
    N_FUSE_FALSE_POS_MAX,
    N_FUSE_QUERY_MIN,
    N_QUERY_FLOOR_FOR_DEMOTION,
    Q3_SEPARATION_THRESHOLD,
    Q3_SPLIT_MIN_HISTORY,
    Q3_SPLIT_MIN_MODE,
    Q5_AGE_MIN,
    Q5_NEG_MIN,
    Q5_POS_MIN,
    RecognizeResult,
    THETA_CATEGORY_DIVERSITY,
    THETA_CATEGORY_INSTANCES,
    THETA_CONTAINMENT,
    THETA_CREATE,
    THETA_FUSE,
    THETA_GROWTH_PLATEAU,
    THETA_PAIR_TRIALS,
    THETA_RECOGNIZE_CATEGORY,
    THETA_RECOGNIZE_MARGIN,
    THETA_UPDATE,
    T_STABLE_MIN,
    _CONTAINMENT_EDGE_TYPES_DIRECTED,
    _CONTAINMENT_EDGE_TYPES_UNDIRECTED,
    _EPS,
    _IT_EDGE_TYPES,
    _IT_LENSES,
    _LABEL_TO_CIDS_SOFT_CAP,
    _V1_EDGE_TYPES,
    _V1_LENSES,
    _V2_EDGE_TYPES,
    _V2_LENSES,
    _V4_EDGE_TYPES,
    _V4_LENSES,
    _extract_actions_from_subgraph,
    _is_effect_atom_label,
    _mcs_accumulate,
    _replay,
    _sig_similarity,
    _subgraph_size,
    _subgraph_wl_hash,
    decide_with_gap,
    edge_tier,
    evidence_score,
    lens_tier,
    mcs_contained,
    mcs_score,
    mcs_similarity,
)
# §5.3.2 structural template machinery — single implementation, owned by
# structural_similarity.py (the pure comparison operator module). The
# Consolidator delegates extraction and materialization to it; the aliases
# preserve the module-internal names (and the mapper.py import surface).
# materialize_template returns exemplar-resident node IDs (§5.3.2
# canonical representative), resolvable through the standard signature
# provider — no auxiliary signature registration is required.


# =============================================================================
# CONSTITUTIONAL PARAMETERS (§5.13)
# =============================================================================

N_REDUCTION_TEMPLATES: int = P.N_REDUCTION_TEMPLATES
KAPPA_INIT: float = P.KAPPA_INIT
MAX_MCS_CANDIDATES: int = P.MAX_MCS_CANDIDATES
MCS_BUDGET_PER_STEP: int = P.MCS_BUDGET_PER_STEP
MCS_MEMO_MAX_ENTRIES: int = P.MCS_MEMO_MAX_ENTRIES
B0_PREFILTER_ENABLED: bool = P.B0_PREFILTER_ENABLED

# [Bundle A] Sentinel distinguishing "memo miss" from a cached 0.0 score.
_MCS_MISSING = object()

# [H.1] Extension-progress instrumentation flag. Adds one `_mcs_margin`
# call (15 MCS calls) after each successful extension. Diagnostic-only.
EXTENSION_PROGRESS_INSTRUMENTATION: bool = True
K_AMB: int = P.K_AMB

# structural_intersection consensus threshold (§5.3.2).
THETA_SIM_CONSENSUS: float = P.THETA_SIM_CONSENSUS

# Transfer gate referenced for the reliability invariant in §5.6.
TAU_TRANSFER: float = P.TAU_TRANSFER




# =============================================================================
# ENUMS AND TYPE ALIASES
# =============================================================================






# Signature provider: (node_id) -> (lens_type, TypedSignature) or None if pruned.
SignatureProvider = Callable[[int], Optional[Tuple[str, TypedSignature]]]


# =============================================================================
# DATA STRUCTURES
# =============================================================================


# StructuralTemplate (§5.3.2) is defined in structural_similarity.py and
# re-exported above: one canonical definition, no parallel dataclass.















def _build_containment_maps(
    sg: Subgraph,
    sigs: Dict[int, Tuple[str, TypedSignature]],
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """[K.7] (parents_of, children_of) for containment edges."""
    parents_of: Dict[int, Set[int]] = defaultdict(set)
    children_of: Dict[int, Set[int]] = defaultdict(set)
    for (u, v, et) in sg.edges:
        if et in _CONTAINMENT_EDGE_TYPES_DIRECTED:
            parents_of[v].add(u)
            children_of[u].add(v)
        elif et in _CONTAINMENT_EDGE_TYPES_UNDIRECTED:
            u_lt = sigs.get(u, ("", None))[0]
            v_lt = sigs.get(v, ("", None))[0]
            if u_lt == "Part" and v_lt != "Part":
                parents_of[v].add(u)
                children_of[u].add(v)
            elif v_lt == "Part" and u_lt != "Part":
                parents_of[u].add(v)
                children_of[v].add(u)
    return parents_of, children_of


def _remove_nodes_and_incident_edges(
    sg: Subgraph, nodes: FrozenSet[int]
) -> Subgraph:
    """[K.7] Drop the given nodes and every incident edge."""
    new_nodes = sg.nodes - nodes
    new_edges = frozenset(
        (s, d, et) for (s, d, et) in sg.edges
        if s not in nodes and d not in nodes
    )
    return Subgraph(
        nodes=new_nodes,
        edges=new_edges,
        # [Pass 24] Preserve edge_params for surviving edges. Dropping them
        # here silently disabled graded edge matching for every subgraph
        # that passed through node removal.
        edge_params={
            k: v for k, v in (getattr(sg, "edge_params", None) or {}).items()
            if k in new_edges
        },
        anchor_of_node=anchor_positions_for(new_nodes, sg.anchor_of_node),
        node_weights={
            k: v for k, v in (sg.node_weights or {}).items()
            if k not in nodes
        },
        weights={
            k: v for k, v in (sg.weights or {}).items()
            if k in new_edges
        },
    )


def _content_discriminative_filter(
    core: Subgraph,
    positives: List[Subgraph],
    negatives: List[Subgraph],
    sig_lookup: SignatureProvider,
    theta_disc_norm: float,       # unused; retained for signature compat
    theta_pos_min: float,         # unused
    pool_sigma_floor: float,      # unused
    top_k_nodes: int = None,
    top_k_edges: int = None,
    disc_exponent: float = None,
    sig_index: Optional[_SigIndex] = None,
    dual_enabled: Optional[bool] = None,
) -> Tuple[Subgraph, Dict[str, Any], Dict[str, Any]]:
    """[v32.2 GRADED] final_mass = prior * max(0, disc)**DISC_EXPONENT.
    Prior contract:
      - empty dict → 1.0 per element
      - non-empty dict, missing key → 0.0 (fail-safe)
    Returns (subgraph, diag, counters)."""
    from collections import Counter

    if top_k_nodes is None:
        top_k_nodes = int(P.GRADED_TOP_K_NODES)
    if top_k_edges is None:
        top_k_edges = int(P.GRADED_TOP_K_EDGES)
    if disc_exponent is None:
        disc_exponent = float(P.DISC_EXPONENT)
    # Phase A Step 2 gate. Read live (P.DUAL_FILTER_ENABLED, not the
    # load-bound copy) so tests can toggle without reimport.
    if dual_enabled is None:
        dual_enabled = bool(P.DUAL_FILTER_ENABLED)
    if dual_enabled and sig_index is None:
        sig_index = _build_sig_index(positives, negatives, sig_lookup)

    diag: Dict[str, Any] = {
        "node_disc": {}, "edge_disc": {},
        "node_prior": {}, "edge_prior": {},
        "node_mass": {}, "edge_mass": {},
        # Phase A Step 2 dual-filter diagnostics (exact vs graded fallback).
        "node_disc_exact": {}, "node_disc_graded": {}, "node_disc_used": {},
        "node_which_won": {}, "edge_disc_exact": {}, "edge_disc_graded": {},
        "edge_disc_used": {}, "edge_which_won": {},
        "node_exact_won_count": 0, "node_graded_won_count": 0,
        "edge_exact_won_count": 0, "edge_graded_won_count": 0,
        "nodes_dropped_unresolvable": 0, "edges_dropped_unresolvable": 0,
        "edge_graded_cap_hits": 0,
        "nodes_kept": 0, "edges_kept": 0,
        "nodes_dropped_zero_mass": 0, "edges_dropped_zero_mass": 0,
        "nodes_dropped_topk": 0, "edges_dropped_topk": 0,
        # [J.2] Endpoint-stage instrumentation (see the top-K block below).
        "nodes_post_topk": 0,
        "edges_viable_pre_topk": 0, "edges_dropped_by_endpoint": 0,
        "node_mass_distribution": {}, "edge_mass_distribution": {},
    }

    # [K.1] Tier attrition — stage 1: what's entering the filter?
    for nid in core.nodes:
        info = sig_lookup(nid)
        lt = info[0] if info is not None else ""
        t = lens_tier(lt)
        k = (f"attrition_in_nodes_tier_{t}" if t >= 0
             else "attrition_in_nodes_tier_unknown")
        diag[k] = diag.get(k, 0) + 1
    for (_s, _d, _et) in core.edges:
        t = edge_tier(_et)
        k = (f"attrition_in_edges_tier_{t}" if t >= 0
             else "attrition_in_edges_tier_unknown")
        diag[k] = diag.get(k, 0) + 1

    def node_key(nid: int):
        info = sig_lookup(nid)
        if info is None:
            return None
        return (info[0], info[1].quantized)

    def edge_key(s: int, d: int, et: EdgeType):
        ks = node_key(s)
        kd = node_key(d)
        if ks is None or kd is None:
            return None
        if et in UNDIRECTED_EDGE_TYPES and ks > kd:
            ks, kd = kd, ks
        return (et, ks, kd)

    def canon_e(e):
        if e[2] in UNDIRECTED_EDGE_TYPES:
            return (min(e[0], e[1]), max(e[0], e[1]), e[2])
        return e

    def presence_counts(subgraphs, is_edge):
        counts: Counter = Counter()
        for sg in subgraphs:
            seen = set()
            if is_edge:
                for (s, d, et) in sg.edges:
                    k = edge_key(s, d, et)
                    if k is not None and k not in seen:
                        seen.add(k)
            else:
                for nid in sg.nodes:
                    k = node_key(nid)
                    if k is not None and k not in seen:
                        seen.add(k)
            for k in seen:
                counts[k] += 1
        return counts

    pos_node = presence_counts(positives, is_edge=False)
    neg_node = presence_counts(negatives, is_edge=False)
    pos_edge = presence_counts(positives, is_edge=True)
    neg_edge = presence_counts(negatives, is_edge=True)

    n_pos = max(1, len(positives))
    n_neg = max(1, len(negatives))

    core_node_prior = getattr(core, "node_weights", None) or {}
    core_edge_prior = getattr(core, "weights", None) or {}

    node_entries: List[Tuple[float, int, float, float]] = []
    for nid in core.nodes:
        k = node_key(nid)
        if k is None:
            # FIX-CATCH: count unresolvable nodes (was silently dropped with
            # no counter, so Step 2's node_dropped_unresolvable is new, not
            # a rename — exact path behavior otherwise unchanged).
            diag["nodes_dropped_unresolvable"] += 1
            continue
        if dual_enabled:
            info = sig_lookup(nid)
            core_lt, core_sig = info
            de, dg, du, ww = _filter_disc_node(core_sig, core_lt, sig_index)
            prior = float(core_node_prior.get(nid, 0.0)) if core_node_prior else 1.0
            # [ITEM 3] shared mass site (RULE K2-LCB gate when dark flag on).
            mass = _mass_from_counts(
                pos_node.get(k, 0), neg_node.get(k, 0),
                n_pos, n_neg, du, prior, disc_exponent,
            )
            diag["node_disc_exact"][nid] = de
            diag["node_disc_graded"][nid] = dg
            diag["node_disc_used"][nid] = du
            diag["node_which_won"][nid] = ww
            diag["node_disc"][nid] = du
            diag["node_prior"][nid] = prior
            diag["node_mass"][nid] = mass
            if ww == "exact":
                diag["node_exact_won_count"] += 1
            elif ww == "graded_fallback":
                diag["node_graded_won_count"] += 1
            if mass > 0.0:
                node_entries.append((mass, nid, du, prior))
            else:
                diag["nodes_dropped_zero_mass"] += 1
            continue
        d_val = (pos_node.get(k, 0) / n_pos) - (neg_node.get(k, 0) / n_neg)
        prior = float(core_node_prior.get(nid, 0.0)) if core_node_prior else 1.0
        # [ITEM 3] shared mass site (RULE K2-LCB gate when dark flag on).
        mass = _mass_from_counts(
            pos_node.get(k, 0), neg_node.get(k, 0),
            n_pos, n_neg, d_val, prior, disc_exponent,
        )
        diag["node_disc"][nid] = d_val
        diag["node_prior"][nid] = prior
        diag["node_mass"][nid] = mass
        diag["node_disc_exact"][nid] = d_val
        diag["node_disc_graded"][nid] = d_val
        diag["node_disc_used"][nid] = d_val
        diag["node_which_won"][nid] = "exact_only"
        if mass > 0.0:
            node_entries.append((mass, nid, d_val, prior))
        else:
            diag["nodes_dropped_zero_mass"] += 1

    edge_entries: List[Tuple[float, Tuple[int, int, EdgeType], float, float]] = []
    for e in core.edges:
        ce = canon_e(e)
        k = edge_key(ce[0], ce[1], ce[2])
        if k is None:
            diag["edges_dropped_unresolvable"] += 1
            continue
        if dual_enabled:
            de, dg, du, ww = _filter_disc_edge(ce[0], ce[1], ce[2], sig_lookup, sig_index)
            prior = float(core_edge_prior.get(ce, 0.0)) if core_edge_prior else 1.0
            # [ITEM 3] shared mass site (RULE K2-LCB gate when dark flag on).
            mass = _mass_from_counts(
                pos_edge.get(k, 0), neg_edge.get(k, 0),
                n_pos, n_neg, du, prior, disc_exponent,
            )
            diag["edge_disc_exact"][ce] = de
            diag["edge_disc_graded"][ce] = dg
            diag["edge_disc_used"][ce] = du
            diag["edge_which_won"][ce] = ww
            diag["edge_disc"][ce] = du
            diag["edge_prior"][ce] = prior
            diag["edge_mass"][ce] = mass
            if ww == "exact":
                diag["edge_exact_won_count"] += 1
            elif ww == "graded_fallback":
                diag["edge_graded_won_count"] += 1
            if mass > 0.0:
                edge_entries.append((mass, ce, du, prior))
            else:
                diag["edges_dropped_zero_mass"] += 1
            continue
        d_val = (pos_edge.get(k, 0) / n_pos) - (neg_edge.get(k, 0) / n_neg)
        prior = float(core_edge_prior.get(ce, 0.0)) if core_edge_prior else 1.0
        # [ITEM 3] shared mass site (RULE K2-LCB gate when dark flag on).
        mass = _mass_from_counts(
            pos_edge.get(k, 0), neg_edge.get(k, 0),
            n_pos, n_neg, d_val, prior, disc_exponent,
        )
        diag["edge_disc"][ce] = d_val
        diag["edge_prior"][ce] = prior
        diag["edge_mass"][ce] = mass
        diag["edge_disc_exact"][ce] = d_val
        diag["edge_disc_graded"][ce] = d_val
        diag["edge_disc_used"][ce] = d_val
        diag["edge_which_won"][ce] = "exact_only"
        if mass > 0.0:
            edge_entries.append((mass, ce, d_val, prior))
        else:
            diag["edges_dropped_zero_mass"] += 1

    # [K.1] Tier attrition — stage 2: which nodes have positive mass?
    # Placed after both mass loops and before both sorts, so it counts
    # exactly the entries that survived the zero-mass gate.
    for entry in node_entries:
        nid = entry[1]
        info = sig_lookup(nid)
        lt = info[0] if info is not None else ""
        t = lens_tier(lt)
        k = (f"attrition_mass_nodes_tier_{t}" if t >= 0
             else "attrition_mass_nodes_tier_unknown")
        diag[k] = diag.get(k, 0) + 1
    for entry in edge_entries:
        et = entry[1][2]
        t = edge_tier(et)
        k = (f"attrition_mass_edges_tier_{t}" if t >= 0
             else "attrition_mass_edges_tier_unknown")
        diag[k] = diag.get(k, 0) + 1

    node_entries.sort(key=lambda x: (-x[0], x[1]))
    edge_entries.sort(key=lambda x: (-x[0], x[1][0], x[1][1], x[1][2].value))

    if len(node_entries) > top_k_nodes:
        diag["nodes_dropped_topk"] = len(node_entries) - top_k_nodes
        node_entries = node_entries[:top_k_nodes]

    # [J.2] The node set is fixed once node top-K has run; edge top-K never
    # touches node_entries, so hoisting this above the edge block cannot
    # change any value -- it only lets the endpoint stage be measured
    # BEFORE edge top-K, which is the stage the historical order hid.
    kept_nodes = {nid for _, nid, _, _ in node_entries}
    diag["nodes_post_topk"] = len(kept_nodes)
    # An edge is "viable" iff both endpoints survived node top-K -- the
    # constraint that makes it an edge OF this subgraph. Measured over all
    # mass-positive edges (post zero-mass, pre edge top-K).
    diag["edges_viable_pre_topk"] = sum(
        1 for _, e, _, _ in edge_entries
        if e[0] in kept_nodes and e[1] in kept_nodes
    )
    diag["edges_dropped_by_endpoint"] = (
        len(edge_entries) - diag["edges_viable_pre_topk"]
    )

    if len(edge_entries) > top_k_edges:
        diag["edges_dropped_topk"] = len(edge_entries) - top_k_edges
        edge_entries = edge_entries[:top_k_edges]

    kept_edges_pre = {e for _, e, _, _ in edge_entries}

    # [K.1] Tier attrition — stage 3: what survived top-K?
    for nid in kept_nodes:
        info = sig_lookup(nid)
        lt = info[0] if info is not None else ""
        t = lens_tier(lt)
        k = (f"attrition_topk_nodes_tier_{t}" if t >= 0
             else "attrition_topk_nodes_tier_unknown")
        diag[k] = diag.get(k, 0) + 1
    for e in kept_edges_pre:
        t = edge_tier(e[2])
        k = (f"attrition_topk_edges_tier_{t}" if t >= 0
             else "attrition_topk_edges_tier_unknown")
        diag[k] = diag.get(k, 0) + 1

    final_edges = {e for e in kept_edges_pre if e[0] in kept_nodes and e[1] in kept_nodes}

    # [K.1] Tier attrition — stage 4: what's in the final core?
    for nid in kept_nodes:
        info = sig_lookup(nid)
        lt = info[0] if info is not None else ""
        t = lens_tier(lt)
        k = (f"attrition_final_nodes_tier_{t}" if t >= 0
             else "attrition_final_nodes_tier_unknown")
        diag[k] = diag.get(k, 0) + 1
    for e in final_edges:
        t = edge_tier(e[2])
        k = (f"attrition_final_edges_tier_{t}" if t >= 0
             else "attrition_final_edges_tier_unknown")
        diag[k] = diag.get(k, 0) + 1

    node_weights = {nid: m for m, nid, _, _ in node_entries if nid in kept_nodes}
    edge_weights = {e: m for m, e, _, _ in edge_entries if e in final_edges}

    diag["nodes_kept"] = len(kept_nodes)
    diag["edges_kept"] = len(final_edges)
    diag["node_mass_distribution"] = _dist(list(node_weights.values()))
    diag["edge_mass_distribution"] = _dist(list(edge_weights.values()))
    # Phase A Step 2: surface edge-cap truncation volume from the shared
    # sentinel counter (0 when the cap never engaged).
    if dual_enabled and sig_index is not None:
        diag["edge_graded_cap_hits"] = int(
            sig_index._sim_cache.get(_CAP_HIT_SENTINEL, 0.0)
        )

    core_edge_params = getattr(core, "edge_params", None) or {}
    filtered_edge_params = {e: core_edge_params[e] for e in final_edges if e in core_edge_params}

    if dual_enabled and sig_index is not None:
        # FIX-CATCH: preserve backward-compatible filter_counters shape by
        # projecting the shared index's exact-key counts (same content as
        # pos_node/neg_node/pos_edge/neg_edge above; keys were built with
        # identical float(quantized) normalization).
        return (
            Subgraph(
                nodes=frozenset(kept_nodes),
                edges=frozenset(final_edges),
                edge_params=filtered_edge_params,
                anchor_of_node=anchor_positions_for(
                    kept_nodes, core.anchor_of_node
                ),
                node_weights=node_weights,
                weights=edge_weights,
            ),
            diag,
            {
                "pos_node": dict(sig_index.pos_count),
                "neg_node": dict(sig_index.neg_count),
                "pos_edge": dict(sig_index.pos_edge_count),
                "neg_edge": dict(sig_index.neg_edge_count),
                "n_pos": sig_index.n_pos, "n_neg": sig_index.n_neg,
            },
        )
    return (
        Subgraph(
            nodes=frozenset(kept_nodes),
            edges=frozenset(final_edges),
            edge_params=filtered_edge_params,
            anchor_of_node=anchor_positions_for(
                kept_nodes, core.anchor_of_node
            ),
            node_weights=node_weights,
            weights=edge_weights,
        ),
        diag,
        {
            "pos_node": pos_node, "neg_node": neg_node,
            "pos_edge": pos_edge, "neg_edge": neg_edge,
            "n_pos": n_pos, "n_neg": n_neg,
        },
    )










# =============================================================================
# SUBGRAPH HASHING (WL CANONICAL)
# =============================================================================
#
# [NOTE] This mirrors `builder._wl_canonical_hash`. Both use
# `binder._node_signature_hash` for initial colors, so identical members +
# edges + underlying signatures produce identical hashes. A future refactor
# should extract the shared core to `primitives.wl_canonical_hash`; the
# duplication is deliberate here to avoid a cross-module refactor inside the
# Consolidator milestone.
# =============================================================================






# =============================================================================
# SIGNATURE SIMILARITY
# =============================================================================




# =============================================================================
# MAX-WEIGHT BIPARTITE MATCHING (HUNGARIAN, O(n^3))
# =============================================================================














# =============================================================================
# STRUCTURAL INTERSECTION / MATERIALIZATION (§5.3.2) — fully delegated.
# `StructuralTemplate`, `_structural_intersection`, and `_materialize` are
# import aliases of structural_similarity's single canonical implementation
# (see module header). Materialization returns exemplar-resident node IDs —
# genuine observation subgraphs resolvable through the standard signature
# provider — so no auxiliary ID namespace or signature registration exists.
# =============================================================================


# =============================================================================
# REPLAY (§5.3.4)
# =============================================================================






# =============================================================================
# CONSOLIDATOR
# =============================================================================


class Consolidator:
    """Two-mode compression engine (§5.12)."""

    def __init__(
        self,
        concept_graph: Any,
        signature_provider: SignatureProvider,
    ) -> None:
        self.cg = concept_graph
        self._sig_provider = signature_provider
        self.core_library = CoreLibrary()
        self.reduction_templates: Dict[Tuple, ReductionTemplate] = {}
        self._current_step: int = -1
        self._mcs_calls_this_step: int = 0
        # [MCS DIAGNOSTIC] Tracks candidate identities that were denied MCS
        # evaluation and when; used to compute the reappearance rate that
        # separates transient under-provisioning (denied once, never seen
        # again) from structural demand growth (denied every appearance).
        self._mcs_denied_hashes: Dict[bytes, int] = {}
        self._mcs_reappearance_count: int = 0
        self._mcs_unique_denied_total: int = 0
        # [Bundle A] Per-call (discriminative_core scope) MCS memo. None
        # outside a discriminative call so stray _mcs_cached callers fall
        # through to compute+budget (no cross-call stale scores).
        self._mcs_memo_call: Optional[Dict[Any, float]] = None
        self._mcs_memo_size_max: int = 0
        self._sig_cache: Dict[int, Optional[Tuple[str, TypedSignature]]] = {}
        # §5.3.4 action-resolution layer (injected by the pipeline; see
        # pipeline_support.make_action_extractor). None ⇒ replay counts
        # `replay_no_extractor` and fails honestly.
        self._action_extractor: Optional[
            Callable[[Subgraph, int], Optional[List[Any]]]
        ] = None
        # D4 schema-level confirmation policy (injected by the pipeline;
        # answers "is this action's causal link counterfactually
        # validated?"). None ⇒ only the literal per-state N_edge count
        # confirms (ruling Item 1, route (a)).
        self._schema_confirmer: Optional[Callable[[Any], bool]] = None

        self.diagnostics: Dict[str, Any] = {
            "pairs_pruned": 0,
            "pairs_failed_replay": 0,
            "empty_core_prune_bug": 0,
            "core_edges_filtered_transient": 0,
            "core_filter_skipped_low_repro": 0,
            "core_filter_skipped_pure_transient": 0,
            "replay_no_extractor": 0,
            "replay_action_extraction_failed": 0,
            "replay_unconfirmed_transition": 0,
            "replay_nondominant_transition": 0,
            "bootstrap_cores_extracted": 0,
            "category_cores_extracted": 0,
            "discriminative_extracted": 0,
            "discriminative_no_intersection": 0,
            # [DEADLOCK FIX] Category/consolidation skip diagnostics
            "category_skipped_no_negatives": 0,
            "category_skipped_no_positives": 0,
            "category_skipped_all_overlap": 0,
            # [DEADLOCK FIX] Extension loop termination diagnostics
            "discriminative_initial_core_too_large": 0,
            "discriminative_extension_abandoned": 0,
            "discriminative_extension_cap_reached": 0,
            "core_library_size": 0,
            "reduction_templates_stored": 0,
            "mcs_budget_exhausted": 0,
    # [v32 CONTENT TIER] Pooling diagnostics.
    "content_negatives_dropped_empty": 0,
    "content_nodes_dropped_off_tier": 0,
    "content_edges_dropped_off_tier": 0,
    # [v32.1 CONTENT FILTER] See _content_discriminative_filter.
    "discriminative_core_empty_before_filter": 0,
    "discriminative_core_empty_after_filter": 0,
    "filter_nodes_kept_by_edge": 0,
    "filter_nodes_kept_by_own_freq": 0,
    "filter_nodes_dropped_by_prevalence": 0,
    "filter_nodes_dropped_by_discriminability": 0,
    "filter_nodes_dropped_unresolvable": 0,
    "filter_edges_kept": 0,
    "filter_edges_dropped_by_prevalence": 0,
    "filter_edges_dropped_by_discriminability": 0,
    "filter_edges_dropped_unresolvable_or_missing_endpoint": 0,
    "filter_pool_by_lens_type": {},
    "filter_edge_pool_by_types": {},
    "filter_node_score_distribution": {},
    "filter_edge_score_distribution": {},
    "filter_runs": {},
    # [v32.2 GRADED] mass pipeline diagnostics.
    "filter_node_mass": {},
    "filter_edge_mass": {},
    "filter_node_prior": {},
    "filter_edge_prior": {},
    "filter_node_mass_distribution": {},
    "filter_edge_mass_distribution": {},
    "filter_nodes_dropped_zero_mass": 0,
    "filter_edges_dropped_zero_mass": 0,
    "filter_nodes_dropped_topk": 0,
    "filter_edges_dropped_topk": 0,
            # [MCS DIAGNOSTIC] Per-step submission tracking for demand-growth
            # analysis. Cleared at the start of each step via
            # _reset_step_budget.
            "mcs_submitted_this_step": 0,
            "mcs_evaluated_this_step": 0,
            "mcs_denied_this_step": 0,
            "mcs_denied_untracked_this_step": 0,
            # [Bundle B0] prefilter elimination counters (cumulative).
            "prefilter_tier1_eliminated": 0,
            "prefilter_tier2_eliminated": 0,
            "prefilter_tier3_eliminated": 0,
            "prefilter_passed": 0,
            # [Bundle A] memo diagnostics
            "mcs_cache_hits": 0,
            "mcs_memo_overflow": 0,
            "mcs_memo_size_max": 0,
            # [Pass 24] EdgeParams population at classification time. A high
            # edge_params_empty_at_classification share means the graded path
            # in _mcs_accumulate is running ungraded -- starved of data
            # rather than absent by design.
            "edge_params_populated": 0,
            "edge_params_empty_at_classification": 0,
            "edge_params_edges_present": 0,
            # Cumulative counters (never cleared within an episode).
            "mcs_submitted_total": 0,
            "mcs_evaluated_total": 0,
            "mcs_denied_total": 0,
            "extension_skipped_budget": 0,
            "discriminative_budget_exhausted": 0,
            # [G.2.1] Number of `_discriminative_core_impl` entries at
            # the current step. Counts real extraction attempts, not
            # wrapper calls that return early on empty positives or
            # negatives (those consume no budget). Reset by
            # `_reset_step_budget` when `t` advances; incremented at
            # the top of `_discriminative_core_impl`.
            "dc_calls_this_step": 0,
            # [F6 falsifier] extension-loop exit-path instrumentation.
            # Cumulative counters (bumped per call, never reset):
            "dc_entry_classified": 0,
            "dc_entry_not_classified": 0,
            "dc_exit_classified": 0,
            "dc_exit_break": 0,
            # [H.4c] Break-exit cause: converged (the extension loop ran
            # out of admissible candidates with no trial denied) vs
            # truncated (at least one trial was denied by the MCS budget).
            # Cumulative; dc_break_converged + dc_break_truncated must
            # equal dc_exit_break on every run.
            # [I.2] Downstream sparsity, one row per second:
            "sparsity_attempts": 0,
            "sparsity_template_nodes": 0,
            "sparsity_template_edges": 0,
            "sparsity_materialize_nodes": 0,
            "sparsity_materialize_edges": 0,
            "sparsity_filter_nodes": 0,
            "sparsity_filter_edges": 0,
            "sparsity_prune_nodes": 0,
            "sparsity_prune_edges": 0,
            "sparsity_filter_nodes_kept": 0,
            "sparsity_filter_edges_kept": 0,
            "dc_break_converged": 0,
            "dc_break_truncated": 0,
            # [I.4] Member-set family lookup (was: WL-hash lookup).
            "family_lookup_hits": 0,
            "family_lookup_miss": 0,
            "dc_exit_break_before_ext": 0,
            "dc_exit_break_extensions_seen": 0,
            "dc_registered_validated": 0,
            "dc_registered_unvalidated": 0,
            "dc_registered_existing_hash_short_circuit": 0,
            # Per-call counters (overwritten at each break; the last
            # break's values are what a single-run dump shows):
            "dc_last_break_extensions_seen": 0,
            "dc_last_break_skipped_delta": 0,
            "dc_last_ext_margin": float("nan"),
            # [I.3] Per-call margin trajectory (cleared at each call entry).
            "dc_ext_margin_trace": [],
            "dc_last_ext_count": 0,
            "dc_last_ext_mcs_evaluated": 0,
            # [K.7] Core-minimization accounting. Pre-seeded so the harness's
            # per-class delta panel reports them even on a run where the
            # minimizer never runs (class_result False -> absent would hide
            # the difference between "not run" and "ran and removed nothing").
            "k7_removed_subtrees": 0,
            "k7_removed_nodes": 0,
            "k7_removed_edges": 0,
            "k7_aborted_budget": 0,
            "k7_pre_nodes": 0,
            "k7_pre_edges": 0,
            "k7_post_nodes": 0,
            "k7_post_edges": 0,
            # [L.1] Near-miss minimization. The two margins are ACCUMULATED
            # (summed over near-miss entries in the run) so the harness's
            # per-class float delta equals the class's own margin when the
            # class has one entry -- the battery's case. A class with several
            # entries reports their sum. The entry_* pair exists because
            # "entered == 0" is ambiguous otherwise: unevaluable means budget
            # denial at the entry check, whereas a recorded margin that fails
            # the two preconditions means the core was simply out of range.
            "k7_near_miss_entered": 0,
            "k7_near_miss_classified_after": 0,
            "k7_near_miss_margin_before": 0.0,
            "k7_near_miss_margin_after": 0.0,
            "k7_near_miss_budget_exhausted": 0,
            "k7_near_miss_entry_unevaluable": 0,
            "k7_near_miss_entry_min_pos": 0.0,
            "k7_near_miss_entry_max_neg": 0.0,
            # [L.1 read-out] The two legs of the post-minimization margin,
            # kept separately from k7_near_miss_margin_after (the difference)
            # so a rising margin can be attributed: containment improved
            # (min_pos up, toward the 0.95 classification bar) or separation
            # improved (max_neg down). They answer different questions --
            # whether the diagnostic rule is nearly there, versus whether
            # minimization is finding separation but not recognition.
            "k7_near_miss_min_pos_after": 0.0,
            "k7_near_miss_max_neg_after": 0.0,
            # [L.2] Accepted subtree removals by the root's lens tier. The
            # buckets cover tiers 0-3 only; a root whose lens type is
            # unclassified (lens_tier == -1) is removed but counted in no
            # bucket, so these four can sum to less than k7_removed_subtrees.
            "k7_removed_by_tier_0": 0,
            "k7_removed_by_tier_1": 0,
            "k7_removed_by_tier_2": 0,
            "k7_removed_by_tier_3": 0,
            # [L.3] Post-extension filter fallback: the unfiltered core
            # classified where the filtered+extended core did not.
            "k7_filter_fallback_post": 0,
            "current_step": -1,
        }

    # -------------------------------------------------------------------------
    # Signature provider (memoized wrapper)
    # -------------------------------------------------------------------------

    def _sig_lookup(
        self,
        nid: int,
    ) -> Optional[Tuple[str, TypedSignature]]:
        cached = self._sig_cache.get(nid, None)
        if cached is not None or nid in self._sig_cache:
            return cached
        info = self._sig_provider(nid)
        self._sig_cache[nid] = info
        return info

    def _invalidate_sig_cache(self) -> None:
        self._sig_cache.clear()

    def _reset_step_budget(self, t: int) -> None:
        """Reset the per-STEP diagnostic counters when `t` advances.

        [G.2.1] The MCS ledger that governs the budget check is no
        longer reset here — see `_begin_extraction_budget`. The two are
        deliberately decoupled because the pipeline can invoke multiple
        extractions at one step (see pipeline._consolidation_sweep and
        pipeline._update_label_windows), and the shared-ledger reset was
        the root cause of the car/cow starvation this fix addresses.

        `_mcs_calls_this_step` is deliberately NOT reset here. It is
        reset by `_begin_extraction_budget`, which is the sole reset
        point for the per-extraction ledger. Any new MCS entry point
        must call `_begin_extraction_budget` at its top, or it will
        inherit a stale ledger from a prior extraction at the same `t`.
        """
        if self._current_step != t:
            self._current_step = t
            self.diagnostics["current_step"] = t
            self.diagnostics["mcs_submitted_this_step"] = 0
            self.diagnostics["mcs_evaluated_this_step"] = 0
            self.diagnostics["mcs_denied_this_step"] = 0
            self.diagnostics["mcs_denied_untracked_this_step"] = 0
            self.diagnostics["dc_calls_this_step"] = 0

    def _begin_extraction_budget(self) -> None:
        """Reset the per-EXTRACTION MCS ledger.

        [G.2.1 FIX — root cause of dc_budget_exhausted on car/cow]

        The pipeline invokes `prune_path` and `discriminative_core`
        multiple times at the same `t`:

          * pipeline._consolidation_sweep loops `for concept in all_live:
            _category_core(concept, t)` — N discriminative extractions
            at one `t`.
          * pipeline._update_label_windows hands a *batch* of label
            signals to `category_manager.step(t, label_signals)`.
          * pipeline._trigger_bootstrap_consolidation calls
            `_category_core` per newly crystallized concept at one `t`.
          * pipeline._consume_consolidation_handoffs and
            _consume_consolidation_pairs loop over multiple
            `prune_path(t, ...)` calls at one `t`.

        Before this fix, all of those shared a single ledger keyed on
        `_current_step`. The first extraction drained it, and every
        subsequent one returned `None` from `_mcs_cached`, which the
        extension loop treats as fatal. Under the shared ledger, one
        class's successful extraction starved every peer in the same
        step; the same hazard existed for `prune_path`.

        Both entry points (`prune_path` and `_discriminative_core_impl`)
        call this at their top. When both fire in the same step, each
        gets its own full ledger, so a heavy `prune_path` cannot starve
        a following `discriminative_core` (or vice versa).

        The parameter name `MCS_BUDGET_PER_STEP` no longer reflects its
        scope; the binding now bounds work per EXTRACTION. Renaming is
        deferred to avoid a cross-module parameter change.
        """
        self._mcs_calls_this_step = 0

    def _consume_mcs_budget(self, n: int = 1) -> bool:
        """Return True if the budget allows n more MCS calls this step.
        [MCS DIAGNOSTIC] Tracks submitted/evaluated/denied counts per step
        and cumulatively for demand-growth correlation analysis."""
        self.diagnostics["mcs_submitted_this_step"] += n
        self.diagnostics["mcs_submitted_total"] += n
        if self._mcs_calls_this_step + n > MCS_BUDGET_PER_STEP:
            self.diagnostics["mcs_denied_this_step"] += n
            self.diagnostics["mcs_denied_total"] += n
            self.diagnostics["mcs_budget_exhausted"] += 1
            return False
        self._mcs_calls_this_step += n
        self.diagnostics["mcs_evaluated_this_step"] += n
        self.diagnostics["mcs_evaluated_total"] += n
        return True

    # [MCS DIAGNOSTIC] Per-step cap on denial-identity tracking: beyond
    # this many denials in one step the reappearance statistics switch to
    # count-only for the remainder of the step (bounded sampling, reported
    # via mcs_denied_untracked_this_step) — tracking every denial at
    # consolidation-tick demand (millions) stalled the tick when identities
    # were serialized+sha256'd per denial.
    MCS_DENIAL_TRACK_CAP_PER_STEP: int = 4096

    def _mcs_candidate_identity(self, Q: Subgraph):
        """[MCS DIAGNOSTIC] Process-stable identity of a query subgraph:
        (frozenset of node ids, frozenset of (src, dst, edge_type.value)
        int-tuples). Int/tuple hashes are NOT PYTHONHASHSEED-randomized,
        and construction is C-speed — no serialization, no sha256 (a
        sha256-per-denial identity stalled consolidation ticks at
        million-denial demand). Deterministic across processes (§9)."""
        return (
            frozenset(Q.nodes),
            frozenset((int(a), int(b), int(et.value)) for a, b, et in Q.edges),
        )

    def _track_mcs_denial(self, candidate_hash: bytes, t: int) -> None:
        """Record that a candidate was denied MCS evaluation this step.
        If this hash was previously denied, increment reappearance count."""
        prev_step = self._mcs_denied_hashes.get(candidate_hash)
        if prev_step is not None:
            self._mcs_reappearance_count += 1
        else:
            self._mcs_unique_denied_total += 1
        self._mcs_denied_hashes[candidate_hash] = t
        # Bound the tracking dict to prevent unbounded growth.
        # Evict entries older than 2 x MCS_BUDGET_PER_STEP steps.
        if len(self._mcs_denied_hashes) > 4096:
            cutoff = t - 2 * MCS_BUDGET_PER_STEP
            stale = [h for h, s in self._mcs_denied_hashes.items() if s < cutoff]
            for h in stale:
                del self._mcs_denied_hashes[h]

    def _mcs_identity(self, sg: Subgraph):
        """[Bundle A] Process-stable identity: node id set + edge
        (src, dst, edge_type.value) int-tuple set. Int/tuple hashes are not
        PYTHONHASHSEED-randomized (determinism mandate). No edge weights in
        the key: mcs_score uses no Edge.weight (verified prerequisite 1)."""
        return (
            frozenset(sg.nodes),
            frozenset((int(s), int(d), int(et.value)) for s, d, et in sg.edges),
        )

    def _mcs_pair_key(self, query: Subgraph, target: Subgraph):
        return (self._mcs_identity(query), self._mcs_identity(target))

    def _mcs_cached(self, query: Subgraph, target: Subgraph) -> Optional[float]:
        """[Bundle A] Memoized + budgeted MCS. Cache hits bypass the budget
        (duplicate containment checks within one discriminative call are
        pure waste, not demand). Denials keep the prior-bundle rejection
        tracking so the demand diagnostic stays live.
        [Batch 4] Returns None on budget denial ("unknown"), never 0.0."""
        key = self._mcs_pair_key(query, target)
        memo = self._mcs_memo_call
        if memo is not None:
            hit = memo.get(key, _MCS_MISSING)
            if hit is not _MCS_MISSING:
                self.diagnostics["mcs_cache_hits"] += 1
                return hit

        if not self._consume_mcs_budget(1):
            if (
                self.diagnostics["mcs_denied_this_step"]
                <= self.MCS_DENIAL_TRACK_CAP_PER_STEP
            ):
                self._track_mcs_denial(
                    self._mcs_candidate_identity(query), self._current_step
                )
            else:
                self.diagnostics["mcs_denied_untracked_this_step"] += 1
            # [Batch 4] Denial is "unknown," not "0.0 overlap." Callers must
            # distinguish. Treating denial as 0.0 was the root cause of the
            # t=500 sweep deciding on denial-zeros.
            return None

        result = self._mcs_compute(query, target)

        if memo is not None and len(memo) < MCS_MEMO_MAX_ENTRIES:
            memo[key] = result
        elif memo is not None:
            self.diagnostics["mcs_memo_overflow"] += 1
        return result

    def _mcs_compute(
        self,
        Q: Subgraph,
        C: Subgraph,
    ) -> float:
        """[Bundle A] Compute-only MCS. Budget + memo live in _mcs_cached."""
        q_sigs: Dict[int, Tuple[str, TypedSignature]] = {}
        for nid in Q.nodes:
            info = self._sig_lookup(nid)
            if info is not None:
                q_sigs[nid] = info
        c_sigs: Dict[int, Tuple[str, TypedSignature]] = {}
        for nid in C.nodes:
            info = self._sig_lookup(nid)
            if info is not None:
                c_sigs[nid] = info
        # [Batch 5 integration] mcs_score now returns (score, corr);
        # _mcs_compute is compute-only and discards the correspondence.
        score, _ = mcs_score(Q, C, q_sigs, c_sigs)
        return score

    def _evidence_compute(self, Q: Subgraph, C: Subgraph) -> float:
        # [Pass 24 diagnostic] Count matched edges carrying EdgeParams on each
        # side. An unpopulated Subgraph does NOT raise -- _mcs_accumulate
        # falls back to an ungraded param_sim -- so the graded path can be
        # silently inert. These counters are the only external way to see it.
        q_params = getattr(Q, "edge_params", None) or {}
        c_params = getattr(C, "edge_params", None) or {}
        q_pop = sum(1 for e in Q.edges if e in q_params)
        c_pop = sum(1 for e in C.edges if e in c_params)
        self.diagnostics["edge_params_populated"] += (
            1 if (q_pop or c_pop) else 0
        )
        self.diagnostics["edge_params_empty_at_classification"] += (
            1 if not (q_pop or c_pop) else 0
        )
        self.diagnostics["edge_params_edges_present"] += q_pop + c_pop

        q_sigs: Dict[int, Tuple[str, TypedSignature]] = {}
        for nid in Q.nodes:
            info = self._sig_lookup(nid)
            if info is not None:
                q_sigs[nid] = info
        c_sigs: Dict[int, Tuple[str, TypedSignature]] = {}
        for nid in C.nodes:
            info = self._sig_lookup(nid)
            if info is not None:
                c_sigs[nid] = info
        score, _ = evidence_score(Q, C, q_sigs, c_sigs)
        return score

    def _evidence_cached(self, query: Subgraph, target: Subgraph) -> Optional[float]:
        """Memoized + budgeted evidence. Cache hits bypass budget.
        Returns None on budget denial."""
        key = self._mcs_pair_key(query, target)
        memo = self._mcs_memo_call
        if memo is not None:
            hit = memo.get(key, _MCS_MISSING)
            if hit is not _MCS_MISSING:
                self.diagnostics["mcs_cache_hits"] += 1
                return hit
        if not self._consume_mcs_budget(1):
            if (
                self.diagnostics["mcs_denied_this_step"]
                <= self.MCS_DENIAL_TRACK_CAP_PER_STEP
            ):
                self._track_mcs_denial(
                    self._mcs_candidate_identity(query), self._current_step
                )
            else:
                self.diagnostics["mcs_denied_untracked_this_step"] += 1
            return None
        result = self._evidence_compute(query, target)
        if memo is not None and len(memo) < MCS_MEMO_MAX_ENTRIES:
            memo[key] = result
        elif memo is not None:
            self.diagnostics["mcs_memo_overflow"] += 1
        return result

    def _evidence_margin(
        self,
        core: Subgraph,
        positives: List[Subgraph],
        negatives: List[Subgraph],
    ) -> Optional[float]:
        if not positives:
            return None
        min_pos: Optional[float] = None
        for p in positives:
            # [Pass 19] ARGUMENT ORDER FIX. evidence_score(Q, C) computes
            # core_coverage = matched_c / (c_mass + eps), i.e. the fraction
            # of the SECOND argument (the Core) that the query supports.
            # These call sites passed (core, p), putting the TRAINING IMAGE in
            # the denominator and asking "what fraction of the whole image
            # does this core explain?" -- the inverse of the intended
            # question. Measured consequence: a small core scored ~0.007
            # against a ~2000-edge positive, so only a 1725-edge core could
            # ever clear THETA_EVIDENCE_POS = 0.35, and the margin was a
            # function of core SIZE rather than of how well the image
            # contains the core. (p, core) is the asymmetric recognition test
            # the rest of the system already uses in mapper.query_category.
            score = self._evidence_cached(p, core)
            if score is None:
                return None
            if min_pos is None or score < min_pos:
                min_pos = score
        max_neg = 0.0
        for n in negatives:
            score = self._evidence_cached(n, core)
            if score is None:
                return None
            if score > max_neg:
                max_neg = score
        return min_pos - max_neg

    # -------------------------------------------------------------------------
    # §5.3.1 Sufficient mode
    # -------------------------------------------------------------------------

    def prune_path(
        self,
        t: int,
        path: Subgraph,
        outcome: Subgraph,
        action_edge_map: Any,
        state_pattern_provider: Callable[[int], Optional[Subgraph]],
        current_state: int,
        source_concept_id: Optional[int] = None,
    ) -> Optional[CoreEntry]:
        """§5.12: element-wise pruning with replay verification.

        Returns the CoreEntry on success, None if the path does not produce
        the outcome. On success, the entry is inserted into the library (or
        the existing entry is refreshed via t_last_validated).
        """
        self._reset_step_budget(t)
        self._begin_extraction_budget()
        self._invalidate_sig_cache()

        # Verify the full path first; a path that doesn't work cannot be
        # minimized into a working core.
        if not _replay(
            path,
            outcome,
            current_state,
            action_edge_map,
            state_pattern_provider,
            self._sig_lookup,
            action_extractor=self._action_extractor,
            diagnostics=self.diagnostics,
            schema_confirmer=self._schema_confirmer,
            concept_graph=self.cg,
        ):
            self.diagnostics["pairs_failed_replay"] += 1
            return None

        # Canonical path order (§5.3.1): topological sort of SEQUENCE edges,
        # with nodes not on any SEQUENCE edge appended in ascending hash order.
        path_order = self._canonical_path_order(path)

        i = 0
        while i < len(path_order):
            candidate = path_order[:i] + path_order[i + 1 :]
            candidate_subgraph = self._elements_to_subgraph(candidate, path)
            if _replay(
                candidate_subgraph,
                outcome,
                current_state,
                action_edge_map,
                state_pattern_provider,
                self._sig_lookup,
                action_extractor=self._action_extractor,
                diagnostics=self.diagnostics,
                schema_confirmer=self._schema_confirmer,
                concept_graph=self.cg,
            ):
                path_order = candidate
                # Do not advance i; the element at i is now a new element.
            else:
                i += 1

        core_subgraph = self._elements_to_subgraph(path_order, path)
        # [H1 GUARD / Phase 3] The prune loop can shrink path_order to
        # empty: _replay([]) vacuously "succeeds" when the outcome already
        # holds in the current state (empty action list -> s_sim =
        # current_state -> MCS(O, current patterns) >= theta). An empty
        # core is unusable (no signatures, never retrievable/matchable) —
        # confirmed root cause of the empty-signature ACTION core
        # 4cbbd8ca (nodes=0, edges=0, sigs=0). Skip, do not store.
        if not core_subgraph.nodes:
            self.diagnostics["empty_core_prune_bug"] += 1
            self.core_library.diagnostics["empty_core_prune_bug"] += 1
            return None

        # [v8 SC filter — v8 post-mortem] Empirical filter: transient edge
        # types fire at specific onset moments and do not recur in queries,
        # so cores carrying them cannot be re-matched. Measured on rooms
        # (2000 steps, 4 configs): 2-node/9-edge cores scored 0.4545 against
        # every query, below the 0.5 family threshold. After filtering to
        # the 5 reproducible edges the same cores score 0.714 — above 0.5.
        #
        # Filter set lives in binder.REPRODUCIBLE_EDGE_TYPES with its own
        # rationale comment. Do NOT tighten it on taxonomy grounds.
        #
        # Exemption: if filtering would leave < 2 reproducible edges, keep
        # the original core. Two cases:
        #   - 1 edge: too few to discriminate — filtering would produce a
        #     trivial min-normalized match. Keep the original 9-edge
        #     over-specified core (which at least doesn't match anything
        #     spurious).
        #   - 0 edges: purely causal skeleton (SEQUENCE
        #     is the _map_family Rule 1 signature for EVENT_CAUSAL and
        #     TEMPORAL_CHAIN). Filtering to empty would delete causal
        #     reasoning from the library. Keep unfiltered.
        repro_edges = frozenset(
            (s, d, et)
            for (s, d, et) in core_subgraph.edges
            if et in REPRODUCIBLE_EDGE_TYPES
        )
        if len(repro_edges) >= 2:
            self.diagnostics["core_edges_filtered_transient"] += (
                len(core_subgraph.edges) - len(repro_edges)
            )
            core_subgraph = Subgraph(
                nodes=core_subgraph.nodes,
                edges=repro_edges,
                anchor_of_node=anchor_positions_for(
                    core_subgraph.nodes, core_subgraph.anchor_of_node
                ),
            )
        elif repro_edges:
            self.diagnostics["core_filter_skipped_low_repro"] += 1
        else:
            self.diagnostics["core_filter_skipped_pure_transient"] += 1

        core_hash = _subgraph_wl_hash(core_subgraph, self._sig_lookup)

        # Dedup by path hash (the original path's identity, not the core's).
        path_hash = _subgraph_wl_hash(path, self._sig_lookup)
        existing = self.core_library.lookup_by_path(path_hash, CoreKind.ACTION)
        if existing is not None:
            existing.t_last_validated = t
            return existing

        # Structural identity collision check.
        existing_by_hash = self.core_library.lookup_by_hash(
            core_hash,
            CoreKind.ACTION,
        )
        if existing_by_hash is not None:
            existing_by_hash.t_last_validated = t
            # [SPEC COMPLIANCE §5.8.1] Record the path alias so future
            # trials of this path short-circuit via entries_by_path
            # instead of re-pruning from scratch.
            idx = self.core_library.entries_by_hash.get(
                (existing_by_hash.canonical_hash, CoreKind.ACTION)
            )
            if idx is not None:
                self.core_library.entries_by_path[(path_hash, CoreKind.ACTION)] = idx
            return existing_by_hash

        family = self._family_of_subgraph(path)
        core_node_sigs = self._collect_signatures(core_subgraph)

        entry = CoreEntry(
            canonical_hash=core_hash,
            kind=CoreKind.ACTION,
            family=family,
            core_subgraph=core_subgraph,
            core_node_signatures=core_node_sigs,
            # extraction_confidence = 1.0 by construction (§5.6). Initial
            # reliability = κ_init.
            reliability=1.0 * KAPPA_INIT,
            context_tags=set(),
            t_created=t,
            t_last_validated=t,
            payload=CorePayload_Action(
                consequence_subgraph=outcome,
                path_hash=path_hash,
            ),
            source_concept_id=source_concept_id,
        )
        self.core_library.add(entry)
        self.diagnostics["pairs_pruned"] += 1
        self.diagnostics["core_library_size"] = self.core_library.size()
        self._update_reduction_template(path, core_subgraph)
        return entry

    # -------------------------------------------------------------------------
    # §5.3.2 Discriminative mode
    # -------------------------------------------------------------------------

    def discriminative_core(
        self, t, label, positives, negatives,
        source_concept_id: Optional[int] = None,
        initial_reliability_factor: float = 1.0,
    ):
        """[v32 CONTENT TIER] Pool positives and negatives to content tier,
        then extract a CATEGORY core on the pooled subgraphs. The instance-
        tier signature provider is swapped for the content store's provider
        for the duration of the extraction.

        [Bundle A] The per-call MCS memo lives exactly for the duration of
        one discriminative extraction; peak size is recorded for the memory
        snapshot and cleared in finally so a raised extraction cannot leak a
        stale memo.
        """
        # [DEADLOCK FIX — Phase 1] The discriminative criterion (§5.8.2)
        # requires negatives to discriminate against. Without them, the
        # operation degenerates to pure intersection, which is not its
        # purpose. Callers wanting pure intersection should call
        # structural_intersection directly.
        #
        # Guard ordering: negatives first (the discriminative criterion
        # requires them), then positives (containment requires them).
        #
        # [ADAPTATION] The directive showed this guard on a bare def; the
        # real method is the Bundle A memo wrapper (*args, **kwargs). The
        # signature is made explicit — both call sites (pipeline
        # _update_label_windows and _category_core) pass exactly these
        # four positional args — and the guard runs BEFORE the memo
        # lifecycle: a guarded call performs no extraction, so it must
        # not reset or leave behind a per-call memo.
        if not negatives:
            if not positives:
                self.diagnostics["category_skipped_no_positives"] += 1
            else:
                self.diagnostics["category_skipped_no_negatives"] += 1
            return None

        if not positives:
            self.diagnostics["category_skipped_no_positives"] += 1
            return None

        # [v32 CONTENT TIER] Pool positives to content tier.
        content_store = ContentSignatureStore()
        content_positives: List[Subgraph] = []
        for p in positives:
            cp = build_content_subgraph(
                p, self._sig_lookup, content_store, self.diagnostics
            )
            if len(cp.nodes) == 0:
                self.diagnostics["category_skipped_no_positives"] += 1
                return None
            content_positives.append(cp)

        # [v32 CONTENT TIER] Pool negatives to content tier. A negative
        # that pools to empty cannot discriminate, so it is dropped
        # rather than aborting the extraction.
        content_negatives: List[Subgraph] = []
        empty_negatives = 0
        for n in negatives:
            cn = build_content_subgraph(
                n, self._sig_lookup, content_store, self.diagnostics
            )
            if len(cn.nodes) == 0:
                empty_negatives += 1
                continue
            content_negatives.append(cn)
        if empty_negatives:
            self.diagnostics["content_negatives_dropped_empty"] = (
                self.diagnostics.get("content_negatives_dropped_empty", 0)
                + empty_negatives
            )
        if not content_negatives:
            self.diagnostics["category_skipped_no_negatives"] += 1
            return None

        saved_provider = self._sig_provider
        saved_cache = self._sig_cache
        self._sig_provider = content_store.get
        self._sig_cache = {}

        self._mcs_memo_call = {}
        self.diagnostics["mcs_cache_hits"] = 0
        self.diagnostics["mcs_memo_overflow"] = 0
        try:
            return self._discriminative_core_impl(
                t, label, content_positives, content_negatives,
                source_concept_id=source_concept_id,
                initial_reliability_factor=initial_reliability_factor,
            )
        finally:
            if self._mcs_memo_call is not None:
                self.diagnostics["mcs_memo_size_max"] = max(
                    self.diagnostics["mcs_memo_size_max"],
                    len(self._mcs_memo_call),
                )
            self._mcs_memo_call = None
            self._sig_provider = saved_provider
            self._sig_cache = saved_cache

    def _scores(
        self,
        subgraph: Subgraph,
        positives: List[Subgraph],
        negatives: List[Subgraph],
    ) -> Tuple[Optional[float], Optional[float]]:
        """[L.1] (min containment over positives, max over negatives).

        Returns (None, None) if any comparison is unevaluable (MCS budget
        denial) -- the caller must treat that as "no information", never as
        a score of 0.0 (the Batch 4 rule).

        Uses _mcs_cached, so it consumes the per-extraction ledger and reuses
        the per-call memo; repeated trials of the same pair are cache hits and
        cost no budget.
        """
        pos: List[float] = []
        for p in positives:
            # [Pass 19] ARGUMENT ORDER FIX -- see _evidence_margin. The core
            # must be the second argument so it is the thing whose coverage
            # is measured.
            s = self._evidence_cached(p, subgraph)
            if s is None:
                return None, None
            pos.append(s)
        neg: List[float] = []
        for n in negatives:
            s = self._evidence_cached(n, subgraph)
            if s is None:
                return None, None
            neg.append(s)
        return (min(pos) if pos else 1.0), (max(neg) if neg else 0.0)

    def _accept_near_miss(
        self,
        trial: Subgraph,
        positives: List[Subgraph],
        negatives: List[Subgraph],
        current_min_pos: float,
        current_max_neg: float,
    ) -> Tuple[bool, bool]:
        """[L.1] (proceed, accept) for one removal trial in near-miss mode.

        accept iff (a) no negative has crossed
        THETA_CONTAINMENT - NEAR_MISS_NEGATIVE_MARGIN, and (b) the margin
        min_pos - max_neg strictly increases (1e-9 tolerance).
        proceed=False means the trial was unevaluable (budget denial).
        """
        min_pos, max_neg = self._scores(trial, positives, negatives)
        if min_pos is None:
            return False, False
        if max_neg >= P.THETA_CONTAINMENT - P.NEAR_MISS_NEGATIVE_MARGIN:
            return True, False
        current_margin = current_min_pos - current_max_neg
        new_margin = min_pos - max_neg
        if new_margin <= current_margin + 1e-9:
            return True, False
        return True, True

    def _minimize_core(
        self,
        core: Subgraph,
        positives: List[Subgraph],
        negatives: List[Subgraph],
        pf_ctx: Any,
        mode: str = "strict",                 # "strict" | "near_miss"
        initial_min_pos: float = 1.0,
        initial_max_neg: float = 0.0,
    ) -> Subgraph:
        """[K.7] Minimize a classifying core to its essential pattern.

        Only called when the extension loop exited with class_result
        True. Every accepted removal must preserve classification.
        A size-ratio + absolute floor prevents trivial collapse: the
        containment classification criterion alone would let a 2-edge
        core score 1.0 against every positive, so the floor is required.

        Phases:
          A. Bottom-up subtree removal (post-order DFS; leaves first).
          B. Single-node removal (removes a container whose edges are
             redundant but whose children are essential).
          C. Edge removal, ascending mass.
          D. Isolated-node removal (unconditional; no classification
             cost).

        Every classify call checks for None (budget denial) and
        gracefully aborts the phase, returning the partially minimized
        core.
        """
        _pre_nodes = len(core.nodes)
        _pre_edges = len(core.edges)
        floor = max(
            P.MIN_CORE_STRUCTURAL_ELEMENTS,
            int(P.MIN_CORE_SIZE_RATIO * _pre_edges),
        )
        current = core
        _rm_subtrees = _rm_nodes = _rm_edges = 0
        _aborted = False
        # [L.1] Near-miss state. In "strict" mode both are unused. The margin
        # is carried across trials, so "accept" means STRICTLY better than the
        # best margin seen so far, not merely better than the entry margin.
        current_min_pos = initial_min_pos
        current_max_neg = initial_max_neg
        # [L.1] Per-call trial gate. NOTE: this counts TRIALS, not MCS calls,
        # despite the parameter's name -- one trial costs up to
        # len(positives) + len(negatives) = 15 _mcs_cached calls. Kept as
        # specified; the binding constraint in practice is the extraction
        # ledger that _mcs_cached enforces (MCS_BUDGET_PER_STEP). Reported.
        _calls = [0]

        def _budget_check() -> bool:
            _calls[0] += 1
            return _calls[0] <= P.NEAR_MISS_MCS_BUDGET

        def _accept(result: Optional[bool]) -> Tuple[bool, bool]:
            """(proceed, accept). proceed=False -> budget abort."""
            if result is None:
                return False, False
            return True, result is True

        # --- Phase A — bottom-up subtree removal.
        # [Pass 19 BUG FIX] This passed `self._sig_lookup` (a bound METHOD)
        # straight through, but _build_containment_maps expects a mapping and
        # calls sigs.get(u, ("", None))[0]. Every undirected containment edge
        # therefore raised AttributeError: 'function' object has no attribute
        # 'get'. The K.7 minimiser had never actually completed a run in this
        # harness -- which is why k7_pre_edges / k7_post_edges never appeared
        # in any log and why near-miss was unreachable. Materialise the
        # mapping once over the core's own nodes.
        _core_sigs: Dict[int, Tuple[str, TypedSignature]] = {
            n: self._sig_lookup(n) for n in current.nodes
        }
        _parents, children_of = _build_containment_maps(current, _core_sigs)
        removal_order: List[int] = []
        visited: Set[int] = set()

        def _dfs(n: int) -> None:
            if n in visited:
                return
            visited.add(n)
            for c in sorted(children_of.get(n, ())):
                _dfs(c)
            removal_order.append(n)

        # [L.2] Tier-ordered traversal. The DFS still emits a node after all
        # of its children (within-subtree order unchanged); only the order
        # ACROSS roots changes: tier-0 subtrees are offered for removal before
        # tier-1, before tier-2, before tier-3. Lower-tier structure is
        # therefore stripped first, and higher-tier structure survives unless
        # classification actually demands its removal. No change to
        # structural_intersection or materialize_template -- the intersection
        # itself is NOT tier-filtered.
        _tier_of = lambda nid: lens_tier(
            self._sig_lookup(nid)[0] if self._sig_lookup(nid) else ""
        )
        roots_all = sorted(
            n for n in current.nodes
            if n not in _parents or not _parents[n]
        )
        # (removal_order / visited are declared above, before _dfs.)
        for tier in (0, 1, 2, 3):
            for r in roots_all:
                if _tier_of(r) != tier:
                    continue
                _dfs(r)
        # Orphan nodes (no containment edges at all, or a lens type lens_tier
        # does not map: _tier_of == -1) are appended last and belong to no
        # tier bucket -- so sum(k7_removed_by_tier_*) can be <
        # k7_removed_subtrees. Reported, not papered over.
        for n in sorted(current.nodes):
            if n not in visited:
                _dfs(n)

        for node in removal_order:
            if node not in current.nodes:
                continue
            subtree: Set[int] = set()
            frontier: List[int] = [node]
            while frontier:
                n = frontier.pop()
                if n in current.nodes and n not in subtree:
                    subtree.add(n)
                    frontier.extend(sorted(children_of.get(n, ())))
            if len(subtree) == len(current.nodes):
                continue
            trial = _remove_nodes_and_incident_edges(
                current, frozenset(subtree)
            )
            if len(trial.edges) < floor:
                continue
            if mode == "strict":
                proceed, accept = _accept(
                    self._classify_prefiltered(
                        trial, positives, negatives, pf_ctx
                    )
                )
            else:  # near_miss
                if not _budget_check():
                    self.diagnostics["k7_near_miss_budget_exhausted"] = (
                        self.diagnostics.get(
                            "k7_near_miss_budget_exhausted", 0
                        ) + 1
                    )
                    _aborted = True
                    break
                proceed, accept = self._accept_near_miss(
                    trial, positives, negatives,
                    current_min_pos, current_max_neg,
                )
                if accept:
                    _new_pos, _new_neg = self._scores(
                        trial, positives, negatives
                    )
                    if _new_pos is not None:
                        current_min_pos, current_max_neg = _new_pos, _new_neg
            if not proceed:
                _aborted = True
                break
            if accept:
                current = trial
                _rm_subtrees += 1
                # [L.2] Attribute the removal to the subtree root's tier.
                _t = _tier_of(node)
                if _t in (0, 1, 2, 3):
                    _key = f"k7_removed_by_tier_{_t}"
                    self.diagnostics[_key] = (
                        self.diagnostics.get(_key, 0) + 1
                    )

        # --- Phase B — single-node removal.
        if not _aborted:
            _deg: Dict[int, int] = {n: 0 for n in current.nodes}
            for (s, d, _et) in current.edges:
                _deg[s] = _deg.get(s, 0) + 1
                _deg[d] = _deg.get(d, 0) + 1
            nodes_sorted = sorted(
                current.nodes, key=lambda n: (-_deg.get(n, 0), n)
            )
            for node in nodes_sorted:
                if node not in current.nodes:
                    continue
                trial = _remove_nodes_and_incident_edges(
                    current, frozenset({node})
                )
                if len(trial.edges) < floor:
                    continue
                if mode == "strict":
                    proceed, accept = _accept(
                        self._classify_prefiltered(
                            trial, positives, negatives, pf_ctx
                        )
                    )
                else:  # near_miss
                    if not _budget_check():
                        self.diagnostics["k7_near_miss_budget_exhausted"] = (
                            self.diagnostics.get(
                                "k7_near_miss_budget_exhausted", 0
                            ) + 1
                        )
                        _aborted = True
                        break
                    proceed, accept = self._accept_near_miss(
                        trial, positives, negatives,
                        current_min_pos, current_max_neg,
                    )
                    if accept:
                        _new_pos, _new_neg = self._scores(
                            trial, positives, negatives
                        )
                        if _new_pos is not None:
                            current_min_pos, current_max_neg = _new_pos, _new_neg
                if not proceed:
                    _aborted = True
                    break
                if accept:
                    current = trial
                    _rm_nodes += 1

        # --- Phase C — edge removal, ascending mass.
        if not _aborted:
            edge_list = sorted(
                list(current.edges),
                key=lambda e: (
                    float((current.weights or {}).get(e, 0.0)),
                    e[0], e[1], e[2].value,
                ),
            )
            for e in edge_list:
                if e not in current.edges:
                    continue
                trial = Subgraph(
                    nodes=current.nodes,
                    edges=frozenset(current.edges - {e}),
                    edge_params={
                        k: v for k, v in (current.edge_params or {}).items()
                        if k != e
                    },
                    anchor_of_node=anchor_positions_for(
                        current.nodes, current.anchor_of_node
                    ),
                    node_weights=current.node_weights,
                    weights={
                        k: v for k, v in (current.weights or {}).items()
                        if k != e
                    },
                )
                if len(trial.edges) < floor:
                    continue
                if mode == "strict":
                    proceed, accept = _accept(
                        self._classify_prefiltered(
                            trial, positives, negatives, pf_ctx
                        )
                    )
                else:  # near_miss
                    if not _budget_check():
                        self.diagnostics["k7_near_miss_budget_exhausted"] = (
                            self.diagnostics.get(
                                "k7_near_miss_budget_exhausted", 0
                            ) + 1
                        )
                        _aborted = True
                        break
                    proceed, accept = self._accept_near_miss(
                        trial, positives, negatives,
                        current_min_pos, current_max_neg,
                    )
                    if accept:
                        _new_pos, _new_neg = self._scores(
                            trial, positives, negatives
                        )
                        if _new_pos is not None:
                            current_min_pos, current_max_neg = _new_pos, _new_neg
                if not proceed:
                    _aborted = True
                    break
                if accept:
                    current = trial
                    _rm_edges += 1

        # --- Phase D — isolated nodes (unconditional).
        if not _aborted:
            _deg2: Dict[int, int] = {n: 0 for n in current.nodes}
            for (s, d, _et) in current.edges:
                _deg2[s] = _deg2.get(s, 0) + 1
                _deg2[d] = _deg2.get(d, 0) + 1
            isolated = sorted(
                (n for n in current.nodes if _deg2.get(n, 0) == 0),
                key=lambda n: (
                    float((current.node_weights or {}).get(n, 0.0)), n,
                ),
            )
            for n in isolated:
                if n not in current.nodes:
                    continue
                current = Subgraph(
                    nodes=current.nodes - {n},
                    edges=current.edges,
                    edge_params=current.edge_params,
                    anchor_of_node=anchor_positions_for(
                        current.nodes - {n}, current.anchor_of_node
                    ),
                    node_weights={
                        k: v
                        for k, v in (current.node_weights or {}).items()
                        if k != n
                    },
                    weights=current.weights,
                )
                _rm_nodes += 1

        # --- Diagnostics.
        self.diagnostics["k7_removed_subtrees"] = (
            self.diagnostics.get("k7_removed_subtrees", 0) + _rm_subtrees
        )
        self.diagnostics["k7_removed_nodes"] = (
            self.diagnostics.get("k7_removed_nodes", 0) + _rm_nodes
        )
        self.diagnostics["k7_removed_edges"] = (
            self.diagnostics.get("k7_removed_edges", 0) + _rm_edges
        )
        self.diagnostics["k7_pre_nodes"] = (
            self.diagnostics.get("k7_pre_nodes", 0) + _pre_nodes
        )
        self.diagnostics["k7_pre_edges"] = (
            self.diagnostics.get("k7_pre_edges", 0) + _pre_edges
        )
        self.diagnostics["k7_post_nodes"] = (
            self.diagnostics.get("k7_post_nodes", 0) + len(current.nodes)
        )
        self.diagnostics["k7_post_edges"] = (
            self.diagnostics.get("k7_post_edges", 0) + len(current.edges)
        )
        if _aborted:
            self.diagnostics["k7_aborted_budget"] = (
                self.diagnostics.get("k7_aborted_budget", 0) + 1
            )
        return current

    def _art_resonate(
        self,
        label: Any,
        core_subgraph: "Subgraph",
        core_node_sigs: Dict[int, Tuple[str, "TypedSignature"]],
        positives: List["Subgraph"],
        negatives: List["Subgraph"],
        t: int,
    ) -> Optional["CoreEntry"]:
        """ARTMAP-style registration: merge into same-label prototype when mutual
        resonance >= ART_VIGILANCE and exceeds resonance with any conflicting class.
        """
        rho = P.ART_VIGILANCE
        if rho is None:
            return None

        cands = [
            e for e in self.core_library.entries
            if e.kind == CoreKind.CATEGORY and e.core_node_signatures
            and getattr(e, "core_status", CoreStatus.CRYSTALLIZED) != CoreStatus.REJECTED
        ]
        cands.sort(key=lambda e: e.canonical_hash)

        best_same: Optional["CoreEntry"] = None
        r_same = -1.0
        r_other = -1.0

        for e in cands[: P.ART_MAX_CANDIDATES]:
            s_ab, _ = evidence_score(core_subgraph, e.core_subgraph, core_node_sigs, e.core_node_signatures)
            s_ba, _ = evidence_score(e.core_subgraph, core_subgraph, e.core_node_signatures, core_node_sigs)
            r = min(s_ab, s_ba)  # Mutual conjunctive resonance
            if getattr(e.payload, "label", None) == label:
                if r > r_same:
                    best_same, r_same = e, r
            elif r > r_other:
                r_other = r

        d = self.diagnostics
        if r_other >= rho:
            d["art_label_conflicts"] = d.get("art_label_conflicts", 0) + 1
        if best_same is None or r_same < rho or r_same <= r_other:
            d["art_new"] = d.get("art_new", 0) + 1
            return None

        pl = best_same.payload
        n_old = getattr(pl, "merge_count", len(pl.supporting_examples))
        n_new = len(positives)

        cd = build_discriminative_content_distribution(positives, negatives, self._sig_lookup)
        sd = build_discriminative_structural_distribution(positives, negatives, self._sig_lookup)

        best_same.content_distribution = _merge_distribution(best_same.content_distribution, cd, n_old, n_new)
        best_same.structural_distribution = _merge_distribution(best_same.structural_distribution, sd, n_old, n_new)

        pl.merge_count = n_old + n_new

        for src, dst in ((positives, pl.supporting_examples), (negatives, pl.discriminating_examples)):
            seen = set(dst)
            for sg in src:
                h = _subgraph_wl_hash(sg, self._sig_lookup)
                if h not in seen:
                    dst.append(h)
                    seen.add(h)

        best_same.t_last_validated = t
        d["art_merged"] = d.get("art_merged", 0) + 1
        return best_same


    def _discriminative_core_impl(
        self,
        t: int,
        label: Any,
        positives: List[Subgraph],
        negatives: List[Subgraph],
        source_concept_id: Optional[int] = None,
        initial_reliability_factor: float = 1.0,
    ) -> Optional[CoreEntry]:
        """§5.12: intersection + minimal extension + prune.

        Returns the CoreEntry on success, None if no structural intersection
        exists (spec §5.3.2: no disjunctive cores in v31).
        """
        self._reset_step_budget(t)
        self._begin_extraction_budget()
        self._invalidate_sig_cache()
        self.diagnostics["dc_calls_this_step"] = (
            self.diagnostics.get("dc_calls_this_step", 0) + 1
        )
        # [I.2] Downstream sparsity: one row per extraction attempt. Each
        # later stage accumulates its stage's element counts, so the
        # attrition table is (stage_total / sparsity_attempts) per stage.
        self.diagnostics["sparsity_attempts"] = (
            self.diagnostics.get("sparsity_attempts", 0) + 1
        )
        # [I.3] Clear the per-call trace at entry, so a call that breaks
        # early cannot report the previous call's trajectory.
        self.diagnostics["dc_ext_margin_trace"] = []
        # [I.6] Clear the per-call margin too: an early break must not
        # leave a stale margin for the registration-time validation factor.
        self.diagnostics["dc_last_ext_margin"] = float("nan")

        # [G.1.5] Per-attempt log. Diagnostic only; no behaviour change.
        # min_positive_size is computed below the empty-positives guard, so
        # it is recomputed here defensively rather than referenced early.
        self.diagnostics["dc_attempts"] = (
            self.diagnostics.get("dc_attempts", 0) + 1
        )
        self.diagnostics["dc_last_label"] = str(label)[:80]
        self.diagnostics["dc_last_min_positive_size"] = (
            min((_subgraph_size(p) for p in positives), default=-1))
        self.diagnostics["dc_last_positives_count"] = len(positives)
        self.diagnostics["dc_last_negatives_count"] = len(negatives)

        # [DEADLOCK FIX — Phase 2 / FIX A] Defense-in-depth guard against
        # empty positives. The Phase 1 guard in discriminative_core should
        # have caught this before _discriminative_core_impl is called, but
        # if it fires, use the correct counter.
        if not positives:
            self.diagnostics["category_skipped_no_positives"] += 1
            self.diagnostics["dc_empty_positives"] = (
                self.diagnostics.get("dc_empty_positives", 0) + 1
            )
            return None

        # [DEADLOCK FIX — Phase 2] Compute the tightest termination bound.
        # The bound matches the gate that makes classification permanently
        # impossible: _classify returns False whenever
        # _subgraph_size(core) > _subgraph_size(p) for ANY positive p
        # (gate verified: total size = nodes + edges, not node count, not
        # MCS min-nodes), and the extension loop only ever ADDS elements —
        # so once the core exceeds the smallest positive, no extension can
        # recover and _classify_prefiltered can never return True again.
        min_positive_size = min(_subgraph_size(p) for p in positives)

        # [REVERTED 2026-09-30] The strict-intersection probe pinned
        # `min_frequency=1.0` here (dc72cae). User-directed revert: held-out
        # fell 4/9 -> 2/9 and its diagnostic purpose was not validated, so
        # [ITEM B / Item 1] Recurrence floor from the Wilson test: a node or
        # edge must recur in at least the number of views where it beats
        # chance against the negatives. For n_pos=7, n_neg=8 this gives
        # min_count=4 (frequency 4/7 ≈ 0.571), replacing the hardcoded 0.5.
        n_pos = len(positives)
        n_neg = len(negatives)
        min_count = _wilson_min_recurrence(n_pos, n_neg)
        min_freq = min_count / float(n_pos)
        template = _structural_intersection(
            positives,
            self._sig_lookup,
            THETA_SIM_CONSENSUS,
            diag=self.diagnostics,
            min_frequency=min_freq,
        )
        self.diagnostics["dc_wilson_min_count"] = min_count
        self.diagnostics["dc_wilson_min_freq"] = min_freq
        self.diagnostics["dc_wilson_n_pos"] = n_pos
        self.diagnostics["dc_wilson_n_neg"] = n_neg
        if template is None:
            self.diagnostics["discriminative_no_intersection"] += 1
            self.diagnostics["dc_no_intersection"] = (
                self.diagnostics.get("dc_no_intersection", 0) + 1
            )
            return None

        # [I.2] Template survives the intersection (including the content
        # filter in _structural_intersection).
        self.diagnostics["sparsity_template_nodes"] = (
            self.diagnostics.get("sparsity_template_nodes", 0)
            + len(template.nodes)
        )
        self.diagnostics["sparsity_template_edges"] = (
            self.diagnostics.get("sparsity_template_edges", 0)
            + len(template.edges)
        )

        # Deterministic exemplar selection: lowest canonical hash. Retained
        # for the ReductionTemplate structural profile (§5.4), which is
        # measured example-size-relative.
        exemplar = min(
            positives,
            key=lambda sg: _subgraph_wl_hash(sg, self._sig_lookup),
        )
        # §5.3.2 materialization: the canonical representative is drawn
        # from the exemplar's own nodes (highest-similarity signature per
        # template component), so every core node ID is resolvable through
        # the standard signature provider — WL hashing, MCS classification,
        # and CoreEntry.core_node_signatures need no auxiliary lookup.
        core_subgraph = _materialize(template, exemplar, self._sig_lookup)
        # [I.2] Materialized core, before the content discriminative filter.
        self.diagnostics["sparsity_materialize_nodes"] = (
            self.diagnostics.get("sparsity_materialize_nodes", 0)
            + len(core_subgraph.nodes)
        )
        self.diagnostics["sparsity_materialize_edges"] = (
            self.diagnostics.get("sparsity_materialize_edges", 0)
            + len(core_subgraph.edges)
        )

        # [v32.1 CONTENT FILTER] See _content_discriminative_filter.
        if len(core_subgraph.nodes) == 0:
            self.diagnostics["discriminative_core_empty_before_filter"] += 1
            self.diagnostics["dc_empty_before_filter"] = (
                self.diagnostics.get("dc_empty_before_filter", 0) + 1
            )
            return None

        # [L.3] Retain the unfiltered (materialized, pre-extension) core.
        # Reference only, no copy: `core_subgraph` is rebound by the filter
        # and by the extension loop, never mutated in place.
        unfiltered_core = core_subgraph

        # Phase A Step 2: build the shared dual-filter index once and reuse
        # it in the extension loop (no rebuild; sim cache shared).
        dual_enabled = bool(P.DUAL_FILTER_ENABLED)
        sig_index = (
            _build_sig_index(positives, negatives, self._sig_lookup)
            if dual_enabled else None
        )

        core_subgraph, filter_diag, filter_counters = _content_discriminative_filter(
            core_subgraph,
            positives,
            negatives,
            self._sig_lookup,
            theta_disc_norm=P.THETA_DISC_NORM,       # unused, ignored
            theta_pos_min=P.THETA_POS_MIN_CONTENT,   # unused, ignored
            pool_sigma_floor=P.POOL_SIGMA_FLOOR,     # unused, ignored
            sig_index=sig_index,
            dual_enabled=dual_enabled,
            # [Phase L] The K.7 branch disabled top-K here (top_k_* = 10000).
            # That was the K.7 trap: unbounded candidate pools multiplied by
            # unbounded comparisons, ~14 min/class at budget 100000. Reverted
            # -- the filter runs at the global GRADED_TOP_K_NODES/EDGES = 100
            # (the strict-intersection probe also showed top-K is nearly inert
            # once the template is strict: 43 vs 46 edges for car).
        )
        if not core_subgraph.nodes:
            self.diagnostics["discriminative_core_empty_after_filter"] += 1
            self.diagnostics["dc_empty_after_filter"] = (
                self.diagnostics.get("dc_empty_after_filter", 0) + 1
            )
            return None

        for k, v in filter_diag.items():
            target = f"filter_{k}"
            if isinstance(v, dict):
                self.diagnostics[target] = v
            else:
                self.diagnostics[target] = self.diagnostics.get(target, 0) + v

        runs = self.diagnostics.setdefault("filter_runs", {})
        label_key = str(label) if label is not None else "<none>"
        if len(label_key) > 120:
            label_key = label_key[:117] + "..."
        runs[label_key] = {
            "node_mass_distribution": filter_diag.get("node_mass_distribution"),
            "edge_mass_distribution": filter_diag.get("edge_mass_distribution"),
            "nodes_kept": filter_diag.get("nodes_kept"),
            "edges_kept": filter_diag.get("edges_kept"),
        }
        while len(runs) > 100:
            runs.pop(next(iter(runs)))

        # [I.2] Filter stage. `filter_nodes/edges` are the post-filter
        # magnitudes actually carried forward; `..._kept` is the filter's
        # own self-report, so the two disagreeing is itself a finding.
        self.diagnostics["sparsity_filter_nodes"] = (
            self.diagnostics.get("sparsity_filter_nodes", 0)
            + len(core_subgraph.nodes)
        )
        self.diagnostics["sparsity_filter_edges"] = (
            self.diagnostics.get("sparsity_filter_edges", 0)
            + len(core_subgraph.edges)
        )
        self.diagnostics["sparsity_filter_nodes_kept"] = (
            self.diagnostics.get("sparsity_filter_nodes_kept", 0)
            + int(filter_diag.get("nodes_kept", 0))
        )
        self.diagnostics["sparsity_filter_edges_kept"] = (
            self.diagnostics.get("sparsity_filter_edges_kept", 0)
            + int(filter_diag.get("edges_kept", 0))
        )

        # [DEADLOCK FIX — Phase 2] Entry check: if the initial template is
        # already too large, the loop can never succeed. This is a
        # different failure from "the loop grew into impossibility" — it
        # means the materialized template or prune_path produced an
        # oversized initial core.
        if _subgraph_size(core_subgraph) > min_positive_size:
            self.diagnostics["discriminative_initial_core_too_large"] += 1
            self.diagnostics["dc_initial_too_large"] = (
                self.diagnostics.get("dc_initial_too_large", 0) + 1
            )
            return None

        # Extension loop: add the smallest discriminating element until
        # classification passes. Added elements are tracked: they — and
        # only they — are candidates for the prune step below.
        extensions_count = 0
        extension_elements: Set[Any] = set()
        # [Bundle B0] prefilter context (positive-side indexes) once per call.
        pf_ctx = self._prefilter_context(positives)
        class_result = self._classify_prefiltered(
            core_subgraph, positives, negatives, pf_ctx
        )
        # [F6 falsifier] Snapshot the cumulative extension-skip counter so
        # the per-call delta can be computed at break time. Also records
        # the entry-classification state so the entry/exit identity can
        # be verified. `not_classified` covers both `False` and `None`
        # (budget-denied at the entry classification); the distinction
        # is recovered from `dc_budget_exhausted` in the exit counters.
        _ext_skipped_at_entry = self.diagnostics.get(
            "extension_skipped_budget", 0
        )
        if class_result is True:
            self.diagnostics["dc_entry_classified"] = (
                self.diagnostics.get("dc_entry_classified", 0) + 1
            )
        else:
            self.diagnostics["dc_entry_not_classified"] = (
                self.diagnostics.get("dc_entry_not_classified", 0) + 1
            )
        while class_result is not True:
            if class_result is None:
                # [Batch 4] Unevaluable. Abandon immediately rather than
                # burn budget on a candidate whose scores are unknown.
                self.diagnostics["discriminative_budget_exhausted"] += 1
                self.diagnostics["dc_budget_exhausted"] = (
                    self.diagnostics.get("dc_budget_exhausted", 0) + 1
                )
                return None
            # class_result is False.
            # [DEADLOCK FIX — Phase 2] Termination: classification is
            # provably impossible once the core exceeds the smallest
            # positive. No extension can recover from this state.
            if _subgraph_size(core_subgraph) > min_positive_size:
                self.diagnostics["discriminative_extension_abandoned"] += 1
                self.diagnostics["dc_extension_abandoned"] = (
                    self.diagnostics.get("dc_extension_abandoned", 0) + 1
                )
                return None

            # [DEADLOCK FIX — Phase 2b] High-ceiling fail-safe. Even if
            # the size bound doesn't fire (e.g., all positives are very
            # large), cap the total extension count to prevent unbounded
            # CPU consumption. This is a safety net, not a fix.
            #
            # 200 is evidence-based: the observed minigrid deadlock
            # reached extensions_count 1076 before py-spy was taken.
            # 200 is well below 1076, ensuring the cap fires before
            # the observed blowup. Not derived from a principled
            # analysis of the extension space; chosen as "much less
            # than observed failure point."
            if extensions_count > 200:
                self.diagnostics["discriminative_extension_cap_reached"] += 1
                self.diagnostics["dc_extension_cap_reached"] = (
                    self.diagnostics.get("dc_extension_cap_reached", 0) + 1
                )
                return None

            best_ext = self._find_best_extension(
                core_subgraph,
                positives,
                negatives,
                sig_index=sig_index,
                filter_counters=filter_counters,
                dual_enabled=dual_enabled,
            )
            if best_ext is None:
                # [F6 falsifier] Break exit — record the state at the
                # moment the loop abandons. Exit-kind classification
                # (dc_exit_classified / dc_exit_break) is bumped once,
                # after the loop, in Anchor 3; this block records the
                # break-specific state only.
                #
                # `dc_last_break_skipped_delta` is the per-call signal
                # that separates case (a) from {b,c}:
                #   delta == 0 → no trial was denied; the candidate pool
                #                was genuinely exhausted → case (a).
                #   delta > 0  → at least one trial was denied → case
                #                (b) or (c); the 1e9-budget run splits.
                _ext_skipped_at_break = self.diagnostics.get(
                    "extension_skipped_budget", 0
                )
                if extensions_count == 0:
                    self.diagnostics["dc_exit_break_before_ext"] = (
                        self.diagnostics.get("dc_exit_break_before_ext", 0)
                        + 1
                    )
                # Cumulative across breaks in the run (sanity check).
                self.diagnostics["dc_exit_break_extensions_seen"] = (
                    self.diagnostics.get("dc_exit_break_extensions_seen", 0)
                    + extensions_count
                )
                # Per-call snapshots (overwritten at each break; the last
                # break's values are what a single-run dump shows).
                self.diagnostics["dc_last_break_extensions_seen"] = (
                    extensions_count
                )
                _skipped_delta = (
                    _ext_skipped_at_break - _ext_skipped_at_entry
                )
                self.diagnostics["dc_last_break_skipped_delta"] = (
                    _skipped_delta
                )
                # [H.4c] Classify the break cause from the same delta that
                # is already recorded above: no trial was denied anywhere
                # in this call (delta 0) means the loop exhausted its
                # admissible candidates -- converged. A positive delta
                # means the MCS budget denied at least one trial, so the
                # loop abandoned a pool it had not finished exploring --
                # truncated. Written here, at the single break site, so
                # the two counters and dc_exit_break cannot drift apart.
                if _skipped_delta == 0:
                    self.diagnostics["dc_break_converged"] = (
                        self.diagnostics.get("dc_break_converged", 0) + 1
                    )
                else:
                    self.diagnostics["dc_break_truncated"] = (
                        self.diagnostics.get("dc_break_truncated", 0) + 1
                    )
                break
            new_nodes, new_edges = best_ext

            valid_new_nodes: Set[int] = set()
            for nid in new_nodes:
                if self._sig_lookup(nid) is not None:
                    valid_new_nodes.add(nid)

            valid_new_edges: Set[Tuple[int, int, EdgeType]] = set()
            for e in new_edges:
                if self._sig_lookup(e[0]) is not None and self._sig_lookup(e[1]) is not None:
                    s, d_, et = e
                    if et in UNDIRECTED_EDGE_TYPES:
                        valid_new_edges.add((min(s, d_), max(s, d_), et))
                    else:
                        valid_new_edges.add(e)

            existing_node_weights = dict(core_subgraph.node_weights) if core_subgraph.node_weights else {}
            existing_edge_weights = dict(core_subgraph.weights) if core_subgraph.weights else {}

            for nid in valid_new_nodes:
                # FIX-CATCH (Phase A review): exact-only keys must use
                # float(quantized) normalization — raw tuples missed every
                # lookup (float-key counters vs raw-key query → pf=nf=0 →
                # mass forced 0). Dual path goes through _compute_node_mass.
                m = _compute_node_mass(
                    nid, self._sig_lookup, sig_index, filter_counters,
                    P.DISC_EXPONENT, dual_enabled,
                )
                existing_node_weights[nid] = m

            for e in valid_new_edges:
                # s, d_, et already canonicalized above.
                m = _compute_edge_mass(
                    e, self._sig_lookup, sig_index, filter_counters,
                    P.DISC_EXPONENT, dual_enabled,
                )
                existing_edge_weights[e] = m

            core_subgraph = Subgraph(
                nodes=frozenset(core_subgraph.nodes | valid_new_nodes),
                edges=frozenset(core_subgraph.edges | valid_new_edges),
                anchor_of_node=anchor_positions_for(
                    core_subgraph.nodes | valid_new_nodes,
                    core_subgraph.anchor_of_node,
                    self.cg.pattern_node_position,
                ),
                node_weights=existing_node_weights,
                weights=existing_edge_weights,
            )
            extension_elements |= set(valid_new_nodes) | set(valid_new_edges)
            extensions_count += 1
            if EXTENSION_PROGRESS_INSTRUMENTATION:
                _margin = self._evidence_margin(
                    core_subgraph, positives, negatives
                )
                _m_val = (
                    float(_margin) if _margin is not None else float("nan")
                )
                self.diagnostics["dc_last_ext_margin"] = _m_val
                self.diagnostics["dc_last_ext_count"] = extensions_count
                self.diagnostics["dc_last_ext_mcs_evaluated"] = (
                    self.diagnostics.get("mcs_evaluated_this_step", 0)
                )
                # [I.3] Per-iteration trajectory. Appended in loop order,
                # so trace[i] is the margin after the i-th extension.
                _trace = self.diagnostics.get("dc_ext_margin_trace")
                if not isinstance(_trace, list):
                    _trace = []
                    self.diagnostics["dc_ext_margin_trace"] = _trace
                _trace.append(_m_val)
            class_result = self._classify_prefiltered(
                core_subgraph, positives, negatives, pf_ctx
            )

        # [F6 falsifier] Distinguish the two loop exits. The loop's own
        # condition (`while class_result is not True`) guarantees that
        # falling out with `class_result is True` is a classified exit,
        # and falling out via `break` (the only other way out that does
        # not return None) leaves `class_result` False.
        #
        # This is the SINGLE site where the exit kind is written. Anchor
        # 2 does not bump dc_exit_break or dc_exit_classified.
        if class_result is True:
            self.diagnostics["dc_exit_classified"] = (
                self.diagnostics.get("dc_exit_classified", 0) + 1
            )
        else:
            self.diagnostics["dc_exit_break"] = (
                self.diagnostics.get("dc_exit_break", 0) + 1
            )

        # Prune: remove any EXTENSION-ADDED element (node OR edge) that is
        # no longer necessary for classification. §5.3.2's "core = core -
        # {element}" is not node-restricted; the extension step adds edges,
        # and those edges must be prunable.
        #
        # [SPEC INTERPRETATION §5.3.2/§5.11/§6.4] The prune step's stated
        # contract is cleanup of the extension step's surplus ("The
        # extension step may have added more than needed"). The
        # intersection base is NOT reduction-eligible: it is the unique,
        # order-independent "guaranteed-shared structure" (§5.3.2), and
        # under the min-normalized MCS (§6.4) virtually every removal
        # trial of a base element passes containment against the positives
        # — the documented small-query property — so an unrestricted prune
        # collapses every category core to a minimal pair that happens to
        # dodge the negatives, destroying the matchable unit (§5.11) that
        # Mapper recognition depends on. Base elements live in the reserved
        # template namespace and extension elements are real graph IDs, so
        # the two sets are disjoint by construction.
        #
        # Snapshot the element list once so mid-loop removals from
        # core_subgraph do not invalidate iteration.
        # [K.7] The §5.3.2 prune loop that stood here examined only
        # extension-added elements (see the commentary above, retained as
        # rollback material). _minimize_core subsumes it and goes further:
        # bottom-up subtree removal, single-node removal, ascending-mass
        # edge removal and unconditional isolated-node removal, over the
        # whole core, each stage bounded by the floor in parameters.py.
        #
        # [K.7] Minimize the core to its essential pattern. Runs only
        # when the extension loop exited with class_result True. On
        # unclassified break exits (the current battery's state, F6) the
        # core registers unvalidated as before — minimization of an
        # unclassified core under a permissive constraint strips it to
        # the floor and silently corrupts held-out.
        #
        # [BUDGET FIX] The minimization phase gets its own MCS ledger.
        # The extension loop arrives here having saturated the extraction
        # ledger (mcs_evaluated_this_step == MCS_BUDGET_PER_STEP, denials
        # > 0, in every class), so every post-loop MCS call -- near-miss
        # entry scoring, removal trials, reclassification and the L.3
        # fallback -- was denied by construction and near-miss
        # minimization could never run (L13 report §3; car entered and
        # aborted on its first trial, k7_aborted_budget = 1).
        #
        # `_begin_extraction_budget` resets `_mcs_calls_this_step`, the
        # counter `_consume_mcs_budget` actually checks. It is not a
        # parameter change, and it deliberately does NOT reset the
        # diagnostic counters: mcs_evaluated_this_step / mcs_denied_this_step
        # / *_total and dc_last_ext_mcs_evaluated keep accumulating over
        # the whole extraction, so the loop's saturation stays visible.
        #
        # Scope: both minimization modes. Strict mode is unreachable today
        # (no class classifies), but it would hit the same wall the moment
        # one does, and the K.7 goal is minimization of classified cores.
        self._begin_extraction_budget()
        if class_result is True:
            core_subgraph = self._minimize_core(
                core_subgraph, positives, negatives, pf_ctx, mode="strict"
            )
        else:
            # [L.1] Near-miss entry test. Three outcomes are instrumented
            # separately because "not entered" has three different meanings:
            # unevaluable (budget denial at the entry check), too far below
            # NEAR_MISS_THRESHOLD, or a negative already inside the margin.
            _entry_min_pos, _entry_max_neg = self._scores(
                core_subgraph, positives, negatives
            )
            if _entry_min_pos is None:
                self.diagnostics["k7_near_miss_entry_unevaluable"] = (
                    self.diagnostics.get(
                        "k7_near_miss_entry_unevaluable", 0
                    ) + 1
                )
            else:
                self.diagnostics["k7_near_miss_entry_min_pos"] = (
                    self.diagnostics.get(
                        "k7_near_miss_entry_min_pos", 0.0
                    ) + _entry_min_pos
                )
                self.diagnostics["k7_near_miss_entry_max_neg"] = (
                    self.diagnostics.get(
                        "k7_near_miss_entry_max_neg", 0.0
                    ) + _entry_max_neg
                )
            if (_entry_min_pos is not None
                    and _entry_min_pos >= P.NEAR_MISS_THRESHOLD
                    and _entry_max_neg < P.THETA_EVIDENCE_NEG - P.NEAR_MISS_NEGATIVE_MARGIN):
                self.diagnostics["k7_near_miss_entered"] = (
                    self.diagnostics.get("k7_near_miss_entered", 0) + 1
                )
                self.diagnostics["k7_near_miss_margin_before"] = (
                    self.diagnostics.get(
                        "k7_near_miss_margin_before", 0.0
                    ) + (_entry_min_pos - _entry_max_neg)
                )
                core_subgraph = self._minimize_core(
                    core_subgraph, positives, negatives, pf_ctx,
                    mode="near_miss",
                    initial_min_pos=_entry_min_pos,
                    initial_max_neg=_entry_max_neg,
                )
                class_result = self._classify_prefiltered(
                    core_subgraph, positives, negatives, pf_ctx
                )
                _after_min_pos, _after_max_neg = self._scores(
                    core_subgraph, positives, negatives
                )
                if _after_min_pos is not None:
                    self.diagnostics["k7_near_miss_margin_after"] = (
                        self.diagnostics.get(
                            "k7_near_miss_margin_after", 0.0
                        ) + (_after_min_pos - _after_max_neg)
                    )
                    # [L.1 read-out] The same post-minimization _scores call
                    # already returns both legs of the margin; record them
                    # separately. No extra MCS work, no control flow, no
                    # threshold: this is the read-out only.
                    self.diagnostics["k7_near_miss_min_pos_after"] = (
                        self.diagnostics.get(
                            "k7_near_miss_min_pos_after", 0.0
                        ) + _after_min_pos
                    )
                    self.diagnostics["k7_near_miss_max_neg_after"] = (
                        self.diagnostics.get(
                            "k7_near_miss_max_neg_after", 0.0
                        ) + _after_max_neg
                    )
                if class_result is True:
                    self.diagnostics["k7_near_miss_classified_after"] = (
                        self.diagnostics.get(
                            "k7_near_miss_classified_after", 0
                        ) + 1
                    )

        # [L.3] If the extended+filtered core still fails, check the
        # unfiltered core. The filter is a performance optimisation; if it
        # removed the wrong elements, restore the unfiltered version. This is
        # the POST-extension check: it runs on the materialized core the
        # filter started from, before the extension loop grew it.
        if class_result is not True:
            unfiltered_result = self._classify_prefiltered(
                unfiltered_core, positives, negatives, pf_ctx
            )
            # [Pass 19 — TRIED AND REVERTED, recorded so it is not retried
            # blind] Minimising the unfiltered core to buy specificity before
            # declining the fallback. Implemented, measured, rejected:
            #   * mechanism confirmed — with the amodal/shared-boundary
            #     emitters on, cow's unfiltered core fails classification on
            #     the NEGATIVE leg only. max_neg rises 0.4638 -> 0.6097
            #     against THETA_EVIDENCE_NEG = 0.60 (overshoot 0.0097), while
            #     min_pos 0.5476 stays far above THETA_EVIDENCE_POS = 0.35.
            #     L.3 then declines and the 10-edge pre-filtered core is
            #     registered instead (19n/10e, reliability 0.389), small
            #     enough to sit inside apple9 and flip it to cow.
            #   * but near-miss minimisation on a 2397-edge core barely
            #     shrinks it at a 60% floor, so car and cow both adopt
            #     enormous cores (2912 edges / 50 nodes and 1438 / 32,
            #     reliability 0.800) and OVERALL argmax fell 7/9 -> 6/9.
            # It trades one pathology for a worse one. The remaining levers
            # are all threshold or policy moves (THETA_EVIDENCE_NEG, or
            # redesigning L.3), which diagnostics/Kc2_specificity_result.md
            # records as requiring explicit direction rather than a silent
            # code change. Left as a comment pending that decision.
            if unfiltered_result is True:
                # [Pass 20b] Margin-maximizing trim of the ACCEPTED unfiltered
                # core, before it is registered.
                #
                # Measured motivation (diagnostics/_p20_coremass.jsonl): cow's
                # unfiltered core classifies, but barely -- min_pos 0.513,
                # max_neg 0.459, margin 0.054. That margin does not survive
                # held-out viewpoint change, which is why cow loses every test
                # image while still being the best MCS match. The FILTERED
                # core cannot be used instead: at 7 edges it scores 0.863-0.966
                # on positives AND 0.726-0.897 on negatives, i.e. it is
                # vacuously contained in everything, because
                # _content_discriminative_filter scores elements individually
                # while cow's discrimination is combinatorial.
                #
                # So the core is neither too large to classify nor usable once
                # filtered. Removing elements that only add non-discriminative
                # mass should widen the margin. This is the inverse of the
                # extension loop: add-to-classify vs remove-for-margin.
                #
                # Implementation note: this deliberately reuses
                # _minimize_core(mode="near_miss") rather than a bespoke
                # greedy loop. A hand-written pass over all 2412 edges is
                # 2412 x 15 ~= 36,000 MCS evaluations, against ceilings of
                # NEAR_MISS_MCS_BUDGET=4000 trials and MCS_BUDGET_PER_STEP=10000
                # -- it would be denied ~72% of its own work and abort. The
                # existing minimiser applies the same greedy margin-increase
                # rule with the same memo and the same budget guards, plus the
                # MIN_CORE_SIZE_RATIO floor that stops it collapsing toward the
                # vacuous 7-edge failure mode.
                #
                # Distinct from the Pass 19 attempt, which minimized only
                # unfiltered cores that had FAILED classification (where
                # minimization is a no-op anyway, since strict mode accepts
                # only removals that preserve classification). Here the core
                # already passes, so there is real margin headroom to work.
                #
                # The trim is accepted ONLY if the result still classifies.
                # A trimmed core that fails is worse than the original: it
                # would be registered unvalidated and re-enter the vacuity
                # trap from the other direction.
                _u_pos, _u_neg = self._scores(
                    unfiltered_core, positives, negatives
                )
                _trimmed = self._minimize_core(
                    unfiltered_core, positives, negatives, pf_ctx,
                    mode="near_miss",
                    initial_min_pos=_u_pos if _u_pos is not None else 1.0,
                    initial_max_neg=_u_neg if _u_neg is not None else 0.0,
                )
                _trim_result = self._classify_prefiltered(
                    _trimmed, positives, negatives, pf_ctx
                )
                _t_pos, _t_neg = self._scores(_trimmed, positives, negatives)
                _u_margin = ((_u_pos or 0.0) - (_u_neg or 0.0))
                _t_margin = ((_t_pos or 0.0) - (_t_neg or 0.0))
                self.diagnostics["k7_l3_trim_edges_before"] = (
                    self.diagnostics.get("k7_l3_trim_edges_before", 0)
                    + len(unfiltered_core.edges)
                )
                self.diagnostics["k7_l3_trim_edges_after"] = (
                    self.diagnostics.get("k7_l3_trim_edges_after", 0)
                    + len(_trimmed.edges)
                )
                self.diagnostics["k7_l3_trim_margin_before"] = (
                    self.diagnostics.get("k7_l3_trim_margin_before", 0.0)
                    + _u_margin
                )
                self.diagnostics["k7_l3_trim_margin_after"] = (
                    self.diagnostics.get("k7_l3_trim_margin_after", 0.0)
                    + _t_margin
                )
                if _trim_result is True:
                    core_subgraph = _trimmed
                    self.diagnostics["k7_l3_trim_accepted"] = (
                        self.diagnostics.get("k7_l3_trim_accepted", 0) + 1
                    )
                else:
                    core_subgraph = unfiltered_core
                    self.diagnostics["k7_l3_trim_rejected"] = (
                        self.diagnostics.get("k7_l3_trim_rejected", 0) + 1
                    )
                class_result = True
                self.diagnostics["k7_filter_fallback_post"] = (
                    self.diagnostics.get("k7_filter_fallback_post", 0) + 1
                )

        # [I.2] Prune stage: the elements that survive the §5.3.2 prune
        # loop. This is the magnitude that gets hashed and registered.
        self.diagnostics["sparsity_prune_nodes"] = (
            self.diagnostics.get("sparsity_prune_nodes", 0)
            + len(core_subgraph.nodes)
        )
        self.diagnostics["sparsity_prune_edges"] = (
            self.diagnostics.get("sparsity_prune_edges", 0)
            + len(core_subgraph.edges)
        )

        core_hash = _subgraph_wl_hash(core_subgraph, self._sig_lookup)
        existing = self.core_library.lookup_by_hash(core_hash, CoreKind.CATEGORY)
        if existing is not None:
            # [Phase 1 Decision 2 / V6-02] Purge-then-create on REJECTED.
            if getattr(existing, "core_status", CoreStatus.CRYSTALLIZED) == CoreStatus.REJECTED:
                self.core_library.remove(core_hash, CoreKind.CATEGORY)
                existing = None
            else:
                # [F6 falsifier] Existing-hash short-circuit: no new
                # registration occurred. Counted separately so Identity 3
                # can be stated as `==` rather than `<=`. The REJECTED
                # branch above falls through to the normal registration
                # block and is NOT counted here.
                self.diagnostics[
                    "dc_registered_existing_hash_short_circuit"
                ] = (
                    self.diagnostics.get(
                        "dc_registered_existing_hash_short_circuit", 0
                    ) + 1
                )
                existing.t_last_validated = t
                return existing

        core_node_sigs = self._collect_signatures(core_subgraph)

        # [Stage 4] ARTMAP prototype resonance: when ART_VIGILANCE is set, a
        # fresh extraction first attempts to merge into an existing same-label
        # prototype whose mutual conjunctive resonance clears the vigilance and
        # beats any conflicting-class resonance. Returns the merged entry, or
        # None to fall through to fresh registration below.
        if P.ART_VIGILANCE is not None:
            merged = self._art_resonate(
                label, core_subgraph, core_node_sigs, positives, negatives, t
            )
            if merged is not None:
                return merged

        # [v32.4] Discriminability-weighted class distributions, computed from
        # the content-tier positives and negatives. Provider at this point is
        # still content_store.get (set by the wrapper). Independent of the core.
        content_dist = build_discriminative_content_distribution(
            positives, negatives, self._sig_lookup
        )
        struct_dist = build_discriminative_structural_distribution(
            positives, negatives, self._sig_lookup
        )

        # [Batch D3] Diversity-softened initial reliability. The caller
        # supplies div(W+)-derived factor; the clamp is repeated here
        # because discriminative_core is a public API and the reliability
        # ceiling (kappa_init) must not depend on caller discipline. The
        # existing-by-hash path above deliberately preserves the already
        # stored reliability: re-extraction must not demote a core that has
        # since earned reliability (§5.6).
        reliability_factor = min(
            1.0, max(0.0, float(initial_reliability_factor))
        )
        # [I.6] Margin-derived validation factor. A break exit did not
        # satisfy _classify, but how far short it fell is informative.
        # Sigmoid on the margin: near-miss ~ validated; far-miss heavily
        # attenuated.
        #
        # Grounding: Signal Detection Theory -- confidence is monotone in
        # distance from the decision boundary.
        #
        # Bounds [0.05, 0.95]: unvalidated cores always contribute
        # (partial evidence retained) but never fully match a validated
        # core at the same score. The margin is the one recorded at the
        # LAST _mcs_margin evaluation of the extension loop (cleared to
        # NaN at call entry by I.3, so an early break cannot inherit a
        # stale value from a previous call).
        if class_result is True:
            validation_factor = 1.0
        else:
            _margin = self.diagnostics.get(
                "dc_last_ext_margin", float("nan")
            )
            if _margin != _margin:   # NaN: no margin was measured
                validation_factor = 0.5
            else:
                _m = max(-5.0, min(5.0, float(_margin)))
                _tau = 0.1
                _s = 1.0 / (1.0 + math.exp(-_m / _tau))
                validation_factor = min(0.95, max(0.05, _s))
        self.diagnostics["dc_last_validation_factor"] = validation_factor
        entry = CoreEntry(
            canonical_hash=core_hash,
            kind=CoreKind.CATEGORY,
            family="",  # ignored for CATEGORY (§5.6)
            validated=(class_result is True),
            core_subgraph=core_subgraph,
            core_node_signatures=core_node_sigs,
            reliability=(
                KAPPA_INIT * reliability_factor * validation_factor
            ),
            context_tags=set(),
            t_created=t,
            t_last_validated=t,
            payload=CorePayload_Category(
                label=label,
                # [v32 CONTENT TIER] The provider has been swapped to the
                # content store by the wrapper; `p`/`n` here are
                # content-tier subgraphs and the hashes are content-tier
                # WL hashes. Mixed-library cores are safely distinguished
                # by core_tier at retrieval.
                supporting_examples=[
                    _subgraph_wl_hash(p, self._sig_lookup) for p in positives
                ],
                discriminating_examples=[
                    _subgraph_wl_hash(n, self._sig_lookup) for n in negatives
                ],
                # [Stage 4] Seed merge_count with the number of supporting
                # examples so the first ARTMAP merge blends distributions with
                # the correct support weight.
                merge_count=len(positives),
            ),
            source_concept_id=source_concept_id,
            # [v32 CONTENT TIER] CATEGORY cores are content-tier; the
            # wrapper pooled positives/negatives and swapped the provider
            # before calling this impl. The ACTION core in prune_path
            # keeps the "instance" default.
            core_tier="content",
            content_distribution=content_dist,
            structural_distribution=struct_dist,
        )
        self.core_library.add(entry)
        self.diagnostics["discriminative_extracted"] += 1
        self.diagnostics["core_library_size"] = self.core_library.size()
        # [F6 falsifier] Attribute the registration to its exit kind. A
        # classified exit registers a validated core; a break exit
        # registers an unvalidated core (§5.12 contract violation).
        if class_result is True:
            self.diagnostics["dc_registered_validated"] = (
                self.diagnostics.get("dc_registered_validated", 0) + 1
            )
        else:
            self.diagnostics["dc_registered_unvalidated"] = (
                self.diagnostics.get("dc_registered_unvalidated", 0) + 1
            )
        self._update_reduction_template(
            exemplar, core_subgraph, extras=extensions_count
        )
        return entry

    # -------------------------------------------------------------------------
    # Classification
    # -------------------------------------------------------------------------

    def _prefilter_context(self, positives: List[Subgraph]):
        """[Bundle B0] Positive-side indexes computed once per
        classification scope: lens-type union, per-positive edge-type
        multisets, and the largest positive size."""
        lens_union: Set[str] = set()
        edge_counts: List[Dict[Any, int]] = []
        max_size = 0
        for p in positives:
            for nid in p.nodes:
                info = self._sig_lookup(nid)
                if info is not None:
                    lens_union.add(info[0])
            petc: Dict[Any, int] = {}
            for _s, _d, et in p.edges:
                petc[et] = petc.get(et, 0) + 1
            edge_counts.append(petc)
            ps = _subgraph_size(p)
            if ps > max_size:
                max_size = ps
        return (lens_union, edge_counts, max_size)

    def _prefilter_fails(self, core: Subgraph, ctx) -> int:
        """[Bundle B0] Necessary-condition checks for "core classifies
        against positives". Returns the eliminating tier (1/2/3) or 0 when
        the candidate passes. Every eliminated core would evaluate
        _classify to False anyway (missing lens type => zero containment in
        some positive; uncovered edge-type multiset => containment
        impossible; oversized core => min-normalized MCS cannot reach
        THETA_CONTAINMENT), so substituting False is semantics-preserving.
        """
        if not B0_PREFILTER_ENABLED:
            return 0
        lens_union, edge_counts, max_size = ctx
        # Tier 1: lens-type intersection floor.
        core_lens: Set[str] = set()
        for nid in core.nodes:
            info = self._sig_lookup(nid)
            if info is not None:
                core_lens.add(info[0])
        if not core_lens <= lens_union:
            self.diagnostics["prefilter_tier1_eliminated"] += 1
            return 1
        # Tier 2: edge-type multiset subtyping.
        if core.edges:
            core_counts: Dict[Any, int] = {}
            for _s, _d, et in core.edges:
                core_counts[et] = core_counts.get(et, 0) + 1
            subsumed = any(
                all(pc.get(et, 0) >= cnt for et, cnt in core_counts.items())
                for pc in edge_counts
            )
            if not subsumed:
                self.diagnostics["prefilter_tier2_eliminated"] += 1
                return 2
        # Tier 3: node-count envelope.
        if _subgraph_size(core) > max_size:
            self.diagnostics["prefilter_tier3_eliminated"] += 1
            return 3
        self.diagnostics["prefilter_passed"] += 1
        return 0

    def _classify_prefiltered(self, core, positives, negatives, ctx) -> Optional[bool]:
        """[Bundle B0] _classify with necessary-condition substitution:
        prefilter-eliminated cores are known-False without any MCS call;
        survivors pay the full classification cost. Control flow identical
        to unfiltered _classify (margin scoring downstream unchanged).
        [Batch 4] Propagates None from _classify. Prefilter failure is a
        definite False (candidate provably cannot classify). Unevaluable
        is None."""
        if self._prefilter_fails(core, ctx):
            return False
        return self._classify(core, positives, negatives)

    def _classify(
        self,
        core: Subgraph,
        positives: List[Subgraph],
        negatives: List[Subgraph],
    ) -> Optional[bool]:
        """Evidence-based classification.

        True  — score >= THETA_EVIDENCE_POS against every positive AND
                < THETA_EVIDENCE_NEG against every negative.
        False — violates either gate.
        None  — budget denial.

        The size gate is removed. Evidence normalizes by total mass on both
        sides, so an oversized core scores low without an early return.
        """
        if _subgraph_size(core) == 0:
            return False
        for p in positives:
            # [Pass 19] ARGUMENT ORDER FIX -- see _evidence_margin. This is
            # the classification gate itself, so the inversion was load-
            # bearing here: it made the positive score a measure of core
            # SIZE. It is also why the docstring's promise ("an oversized
            # core scores low") only held by accident of the image's mass
            # sitting in the denominator.
            score = self._evidence_cached(p, core)
            if score is None:
                return None
            if score < P.THETA_EVIDENCE_POS:
                return False
        for n in negatives:
            score = self._evidence_cached(n, core)
            if score is None:
                return None
            if score >= P.THETA_EVIDENCE_NEG:
                return False
        return True

    def _mcs_margin(
        self,
        core: Subgraph,
        positives: List[Subgraph],
        negatives: List[Subgraph],
    ) -> Optional[float]:
        """Returns float or None.

        None means one or more MCS calls were unevaluable. A margin
        computed from partial evaluations is built on unknown inputs and
        must not be used for ranking.

        [Batch 4] Empty positives returns None (was 0.0). Callers skip
        None margins; no positives means no classification is possible,
        which is failure, not zero-margin.
        """
        if not positives:
            return None
        # [P0-DOC / STRUCTURAL DEBT] This extension loop re-submits the
        # same containment checks thousands of times per consolidation tick
        # (measured: ~196 unique pairs behind ~1.26M submissions). The
        # per-call memo (_mcs_memo_call) collapses the duplicates; the
        # underlying algorithmic redundancy (re-testing every extension
        # candidate against every example from scratch) remains structural
        # debt slated for a discriminative-core redesign, not this bundle.
        min_pos: Optional[float] = None
        for p in positives:
            score = self._mcs_cached(core, p)
            if score is None:
                return None
            if min_pos is None or score < min_pos:
                min_pos = score
        max_neg = 0.0
        for n in negatives:
            score = self._mcs_cached(core, n)
            if score is None:
                return None
            if score > max_neg:
                max_neg = score
        return min_pos - max_neg

    def _find_best_extension(
        self,
        core: Subgraph,
        positives: List[Subgraph],
        negatives: List[Subgraph],
        sig_index: Optional[_SigIndex] = None,
        filter_counters: Optional[Dict] = None,
        dual_enabled: bool = False,
    ) -> Optional[Tuple[Set[int], Set[Tuple[int, int, EdgeType]]]]:
        """§5.3.2: smallest element appearing in some positive, no negative,
        not already in core, that improves classification the most.

        [I.5] Candidates are TRIALED in mass-priority order (highest mass
        first) rather than in raw ID order: the loop breaks on the first
        trial that improves classification, so order decides which
        element is added. The returned element is therefore unchanged as
        a rule, but the truncation point under the MCS budget moves.
        """
        # [I.5 adaptive fix] `_compute_node_mass` writes into its counters
        # dict, so a dual-enabled call with no counters supplied must not
        # reach it with None.
        if filter_counters is None:
            filter_counters = {}
        neg_node_union: Set[int] = set()
        neg_edge_union: Set[Tuple[int, int, EdgeType]] = set()
        for n in negatives:
            neg_node_union |= n.nodes
            neg_edge_union |= n.edges
        # [Bundle B0] prefilter context for trial-core substitution.
        pf_ctx = self._prefilter_context(positives)

        # Candidate nodes.
        candidate_nodes: Set[int] = set()
        for p in positives:
            for nid in p.nodes:
                if nid in core.nodes:
                    continue
                if nid in neg_node_union:
                    continue
                candidate_nodes.add(nid)

        # Candidate edges. Only edges whose both endpoints are already in
        # the core are valid extensions; adding an edge whose endpoint is
        # absent produces a malformed Subgraph and corrupts downstream MCS.
        candidate_edges: Set[Tuple[int, int, EdgeType]] = set()
        for p in positives:
            for e in p.edges:
                if e in core.edges:
                    continue
                if e in neg_edge_union:
                    continue
                if e[0] not in core.nodes or e[1] not in core.nodes:
                    continue
                candidate_edges.add(e)

        if not candidate_nodes and not candidate_edges:
            return None

        # [I.5] Mass-based priority. Reuses the same mass formula the
        # discriminative filter applies at seeding:
        #   mass = (prior * max(0, disc)) ** DISC_EXPONENT
        # where disc = pf - nf. A high-mass candidate appears in positives
        # and not in negatives -- exactly the candidate whose inclusion is
        # most likely to move _classify toward True.
        #
        # Fallback when the dual filter is off: for each candidate node,
        # (best positive similarity - best negative similarity) over
        # same-lens nodes. Pre-grouped by lens to bound cost.
        def _node_priority(nid: int) -> float:
            if dual_enabled and sig_index is not None:
                return _compute_node_mass(
                    nid, self._sig_lookup, sig_index, filter_counters,
                    P.DISC_EXPONENT, dual_enabled,
                )
            info = self._sig_lookup(nid)
            if info is None:
                return 0.0
            _lt, sig = info
            pos_best = 0.0
            for p_sig in pos_sigs_by_lens.get(_lt, ()):
                s_v = _sig_similarity(sig, p_sig)
                if s_v > pos_best:
                    pos_best = s_v
            neg_best = 0.0
            for n_sig in neg_sigs_by_lens.get(_lt, ()):
                s_v = _sig_similarity(sig, n_sig)
                if s_v > neg_best:
                    neg_best = s_v
            return max(0.0, pos_best - neg_best)

        def _edge_priority(e: Tuple[int, int, EdgeType]) -> float:
            if dual_enabled and sig_index is not None:
                return _compute_edge_mass(
                    e, self._sig_lookup, sig_index, filter_counters,
                    P.DISC_EXPONENT, dual_enabled,
                )
            return 0.0

        # [I.5 pre-group] Same-lens signature sets, built once per call.
        # [I.5 adaptive fix] Sets, not lists: the fallback priority is a
        # MAX over the group, and duplicates cannot change a max, so
        # de-duplicating cuts the fallback's cost by the window-overlap
        # factor without altering any priority. A duplicate-heavy battery
        # is otherwise quadratic in nodes-per-window x windows.
        pos_sigs_by_lens: Dict[str, Set[TypedSignature]] = {}
        neg_sigs_by_lens: Dict[str, Set[TypedSignature]] = {}
        for p_sg in positives:
            for p_nid in p_sg.nodes:
                p_info = self._sig_lookup(p_nid)
                if p_info is None:
                    continue
                pos_sigs_by_lens.setdefault(p_info[0], set()).add(p_info[1])
        for n_sg in negatives:
            for n_nid in n_sg.nodes:
                n_info = self._sig_lookup(n_nid)
                if n_info is None:
                    continue
                neg_sigs_by_lens.setdefault(n_info[0], set()).add(n_info[1])

        ordered_nodes = sorted(
            candidate_nodes,
            key=lambda nid: (
                -_node_priority(nid), hash_128(serialize_int(nid))
            ),
        )
        ordered_edges = sorted(
            candidate_edges,
            key=lambda e: (
                -_edge_priority(e), e[0], e[1], e[2].value,
            ),
        )

        best_ext: Optional[Tuple[Set[int], Set[Tuple[int, int, EdgeType]]]] = None
        best_score = -float("inf")
        # [Determinism guard] Lexicographic key: bucketed score, then the
        # stable content hash. Comparing the WHOLE key -- never a hand-rolled
        # running max -- keeps the acceptance bar monotonic: a hash win for a
        # lower-scoring candidate cannot lower the bar for later comparisons,
        # and the outcome is order-independent. _EVID_TIE_GRID is sized from
        # the two-seed probe (min positive top-two gap / 10), never tuned to
        # benchmark numbers.
        best_key: Optional[Tuple[int, bytes]] = None

        # Node extensions.
        core_params = getattr(core, "edge_params", None) or {}
        for nid in ordered_nodes:   # [I.5] mass-priority order
            new_core = Subgraph(
                nodes=frozenset(core.nodes | {nid}),
                edges=core.edges,
                edge_params=core_params,
                anchor_of_node=anchor_positions_for(
                    core.nodes | {nid},
                    core.anchor_of_node,
                    self.cg.pattern_node_position,
                ),
            )
            class_result = self._classify_prefiltered(
                new_core, positives, negatives, pf_ctx
            )
            if class_result is True:
                return ({nid}, set())
            if class_result is None:
                self.diagnostics["extension_skipped_budget"] += 1
                continue
            # class_result is False — proceed to margin scoring.
            score = self._evidence_margin(new_core, positives, negatives)
            if score is None:
                self.diagnostics["extension_skipped_budget"] += 1
                continue
            tb = hash_128(serialize_int(nid))
            key = (round(score / P._EVID_TIE_GRID), tb)
            if best_key is None or key > best_key:
                best_key = key
                best_ext = ({nid}, set())
                best_score = key[0] * P._EVID_TIE_GRID

        # Edge extensions.
        for e in ordered_edges:   # [I.5] mass-priority order
            new_params = dict(core_params)
            edge_obj = self.cg.edges.get(e)
            if edge_obj is not None and getattr(edge_obj, "params", None) is not None:
                new_params[e] = edge_obj.params
            new_core = Subgraph(
                nodes=core.nodes,
                edges=frozenset(core.edges | {e}),
                edge_params=new_params,
                anchor_of_node=anchor_positions_for(
                    core.nodes, core.anchor_of_node
                ),
            )
            class_result = self._classify_prefiltered(
                new_core, positives, negatives, pf_ctx
            )
            if class_result is True:
                return (set(), {e})
            if class_result is None:
                self.diagnostics["extension_skipped_budget"] += 1
                continue
            # class_result is False — proceed to margin scoring.
            score = self._evidence_margin(new_core, positives, negatives)
            if score is None:
                self.diagnostics["extension_skipped_budget"] += 1
                continue
            tb = hash_128(
                serialize_tuple(
                    [
                        serialize_int(e[0]),
                        serialize_int(e[1]),
                        serialize_int(e[2].value),
                    ]
                )
            )
            key = (round(score / P._EVID_TIE_GRID), tb)
            if best_key is None or key > best_key:
                best_key = key
                best_ext = (set(), {e})
                best_score = key[0] * P._EVID_TIE_GRID

        return best_ext

    # -------------------------------------------------------------------------
    # Path ordering and subgraph helpers
    # -------------------------------------------------------------------------

    def _canonical_path_order(self, G: Subgraph) -> List[int]:
        """§5.3.1: canonical path order.

        Returns node IDs in topological sort order of the SEQUENCE edges,
        followed by nodes not on any SEQUENCE edge in ascending canonical
        order. Node-only by construction; edges are not returned as
        elements (see `_elements_to_subgraph`, which induces them).
        """
        seq_edges = [
            (src, dst) for (src, dst, et) in G.edges if et == EdgeType.SEQUENCE
        ]

        on_seq: Set[int] = set()
        for src, dst in seq_edges:
            on_seq.add(src)
            on_seq.add(dst)

        if seq_edges:
            out_adj: Dict[int, List[int]] = defaultdict(list)
            in_degree: Dict[int, int] = {nid: 0 for nid in on_seq}
            for src, dst in seq_edges:
                out_adj[src].append(dst)
                in_degree[dst] = in_degree.get(dst, 0) + 1
            queue: deque = deque(sorted(n for n in on_seq if in_degree[n] == 0))
            ordered_seq: List[int] = []
            while queue:
                n = queue.popleft()
                ordered_seq.append(n)
                for nb in sorted(out_adj.get(n, ())):
                    in_degree[nb] -= 1
                    if in_degree[nb] == 0:
                        queue.append(nb)
            remaining = sorted(n for n in on_seq if n not in set(ordered_seq))
            ordered_seq.extend(remaining)
        else:
            ordered_seq = []

        off_seq = sorted(
            [n for n in G.nodes if n not in on_seq],
            key=self._node_canonical_key,
        )
        return ordered_seq + off_seq

    def _node_canonical_key(self, nid: int) -> Tuple[bytes, bytes, int]:
        """§5.3.1: ascending canonical-hash order for a node not on any
        SEQUENCE edge.

        A pattern node's canonical identity is (lens_type, signature_hash)
        per §1.11. Node ID appended as a final tie-break for stability
        against pruned nodes whose signature is unavailable.
        """
        info = self._sig_lookup(nid)
        if info is None:
            return (b"_PRUNED_", b"\x00" * 16, nid)
        return (info[0].encode("utf-8"), _node_signature_hash(info[1]), nid)

    def _element_sort_key(self, element: Any) -> Tuple[int, int]:
        """Canonical sort key for a mixed node/edge element list.

        Nodes are keyed by (tag=0, 128-bit hash of the node id); edges are
        keyed by (tag=1, EdgeType.value). Tag disambiguation keeps node and
        edge elements on one totally-ordered key space for deterministic
        iteration in §5.3.2's prune loop. Nodes in `_canonical_path_order`
        use `_node_canonical_key` instead, which keys on the pattern node's
        (lens_type, signature_hash) identity per §1.11.
        """
        if isinstance(element, int):
            return (
                0,
                int.from_bytes(hash_128(serialize_int(element)), "big") % (1 << 128),
            )
        edge = element
        et = edge[2] if len(edge) > 2 else edge[-1]
        et_val = et.value if isinstance(et, EdgeType) else int(et)
        return (1, et_val)

    def _elements_to_subgraph(
        self,
        elements: List[Any],
        reference: Subgraph,
    ) -> Subgraph:
        """Reconstruct a subgraph from an ordered element list.

        Elements are node IDs. Edges are the induced edges among the given
        nodes (drawn from the reference subgraph's edge set).
        """
        node_set = {e for e in elements if isinstance(e, int)}
        member_set = node_set
        edges = frozenset(
            (src, dst, et)
            for (src, dst, et) in reference.edges
            if src in member_set and dst in member_set
        )
        ref_params = getattr(reference, "edge_params", None) or {}
        # [Pass 24] Inherit from the reference first, then fall back to the
        # concept graph's edge store for any edge the reference lacks. The
        # reference may itself have been built before edge_params were
        # populated, in which case inheriting alone yields an empty dict and
        # silently disables graded edge matching for the whole ACTION core.
        edge_params: Dict[Tuple[int, int, Any], Any] = {}
        missing = []
        for e in edges:
            if e in ref_params:
                edge_params[e] = ref_params[e]
            else:
                missing.append(e)
        if missing:
            for key, params in edge_params_for(self.cg, missing).items():
                edge_params.setdefault(key, params)
        return Subgraph(
            nodes=frozenset(node_set),
            edges=edges,
            edge_params=edge_params,
            anchor_of_node=anchor_positions_for(
                node_set,
                reference.anchor_of_node,
                self.cg.pattern_node_position,
            ),
        )

    def _collect_signatures(
        self,
        G: Subgraph,
    ) -> Dict[int, Tuple[str, TypedSignature]]:
        result: Dict[int, Tuple[str, TypedSignature]] = {}
        for nid in G.nodes:
            info = self._sig_lookup(nid)
            if info is not None:
                result[nid] = info
        return result

    def _family_of_subgraph(self, G: Subgraph) -> str:
        """Recover the family of a subgraph via member-set identity.

        [I.4] Member set (pattern-node IDs) is stable: nodes are keyed
        by (lens_type, signature_hash) per §1.11. Edges drift under EMA
        decay and pruning, so the historical WL-hash lookup missed
        whenever the live subgraph's edges differed from
        crystallization-time edges. Member-set lookup does not depend on
        edges.

        On multiple matches, pick highest layer, tie-break lowest id.
        """
        key = frozenset(G.nodes)
        if not key:
            return ""
        candidates = self.cg.concepts_by_member_set(key)
        if not candidates:
            self.diagnostics["family_lookup_miss"] = (
                self.diagnostics.get("family_lookup_miss", 0) + 1
            )
            return ""

        best_cid: Optional[int] = None
        best_layer = -1
        for cid in sorted(candidates):
            concept = self.cg.concepts.get(cid)
            if concept is None:
                continue
            status_name = getattr(concept.status, "name", "")
            if status_name in ("TOMBSTONED", "REJECTED"):
                continue
            layer = int(getattr(concept, "layer", 0))
            if layer > best_layer or (
                layer == best_layer
                and (best_cid is None or cid < best_cid)
            ):
                best_layer = layer
                best_cid = cid

        if best_cid is None:
            self.diagnostics["family_lookup_miss"] = (
                self.diagnostics.get("family_lookup_miss", 0) + 1
            )
            return ""
        concept = self.cg.concepts[best_cid]
        fam = getattr(concept, "family", None)
        if fam is None:
            return ""
        self.diagnostics["family_lookup_hits"] = (
            self.diagnostics.get("family_lookup_hits", 0) + 1
        )
        return fam.name

    # -------------------------------------------------------------------------
    # Reduction templates (§5.4)
    # -------------------------------------------------------------------------

    def get_reduction_template(
        self,
        structural_profile: Tuple,
    ) -> Optional[ReductionTemplate]:
        return self.reduction_templates.get(structural_profile)

    def update_reduction_template(
        self,
        structural_profile: Tuple,
        observed_ratio: float,
        observed_extras: int,
    ) -> None:
        tpl = self.reduction_templates.get(structural_profile)
        if tpl is None:
            if len(self.reduction_templates) >= N_REDUCTION_TEMPLATES:
                self._evict_reduction_template()
            tpl = ReductionTemplate(structural_profile=structural_profile)
            self.reduction_templates[structural_profile] = tpl
        # Sample-count-weighted running average.
        n = tpl.sample_count
        tpl.observed_ratio = (tpl.observed_ratio * n + observed_ratio) / (n + 1)
        tpl.observed_extras_count = (
            tpl.observed_extras_count * n + observed_extras
        ) / (n + 1)
        tpl.sample_count = n + 1
        self.diagnostics["reduction_templates_stored"] = len(self.reduction_templates)

    def _update_reduction_template(
        self,
        example: Subgraph,
        core: Subgraph,
        extras: int = 0,
    ) -> None:
        """§5.4: record the compression prior.

        `example` is the input artifact (path in sufficient mode, exemplar
        in discriminative mode). `extras` is the number of extensions
        applied during the discriminative extension loop; it is 0 for
        sufficient mode where extensions do not occur, not the count of
        pruned elements.
        """
        profile = self._structural_profile(example)
        example_size = _subgraph_size(example)
        core_size = _subgraph_size(core)
        ratio = core_size / max(1, example_size)
        self.update_reduction_template(profile, ratio, extras)

    def _structural_profile(self, G: Subgraph) -> Tuple:
        """§5.4: structural profile key.

        (n_nodes_bucket, n_edges_bucket, density_bucket, dominant_edge_types)
        """
        n_nodes = len(G.nodes)
        n_edges = len(G.edges)
        density = n_edges / max(1, n_nodes)
        # Bucketization: 5-node and 5-edge buckets, 0.2 density buckets.
        node_b = n_nodes // 5
        edge_b = n_edges // 5
        dens_b = int(density / 0.2)
        # Dominant edge types: top-3 by frequency, deterministic tie-break
        # by EdgeType value.
        counts: Dict[EdgeType, int] = defaultdict(int)
        for _s, _d, et in G.edges:
            counts[et] += 1
        sorted_types = sorted(
            counts.items(),
            key=lambda kv: (-kv[1], kv[0].value),
        )[:3]
        dominant = tuple(et for (et, _c) in sorted_types)
        return (node_b, edge_b, dens_b, dominant)

    def _evict_reduction_template(self) -> None:
        if not self.reduction_templates:
            return
        # Deterministic eviction: lowest sample_count, tie-break by
        # canonical hash of the profile tuple.
        sorted_items = sorted(
            self.reduction_templates.items(),
            key=lambda kv: (kv[1].sample_count, hash_128(serialize_value(kv[0]))),
        )
        if sorted_items:
            victim_key = sorted_items[0][0]
            del self.reduction_templates[victim_key]

    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------

    def set_action_extractor(
        self,
        extractor: Callable[[Subgraph, int], Optional[List[Any]]],
    ) -> None:
        """Inject the §5.3.4 actions(G) resolver (wired by the pipeline
        from pipeline_support.make_action_extractor)."""
        self._action_extractor = extractor

    def set_schema_confirmer(
        self,
        confirmer: Callable[[Any], bool],
    ) -> None:
        """Inject the D4 schema-level confirmation policy (action →
        counterfactually validated?). Used by the ruling Item-1 gate as
        the alternative to per-state mass ≥ N_edge."""
        self._schema_confirmer = confirmer

    def get_core_library(self) -> CoreLibrary:
        return self.core_library

    def get_mcs_diagnostics(self) -> Dict[str, Any]:
        """Complete MCS diagnostic snapshot for demand-growth analysis."""
        d = self.diagnostics
        return {
            "submitted_this_step": d.get("mcs_submitted_this_step", 0),
            "evaluated_this_step": d.get("mcs_evaluated_this_step", 0),
            "denied_this_step": d.get("mcs_denied_this_step", 0),
            "submitted_total": d.get("mcs_submitted_total", 0),
            "evaluated_total": d.get("mcs_evaluated_total", 0),
            "denied_total": d.get("mcs_denied_total", 0),
            "budget_exhausted_events": d.get("mcs_budget_exhausted", 0),
            "reappearance_count": self._mcs_reappearance_count,
            "unique_denied_total": self._mcs_unique_denied_total,
            "reappearance_rate": (
                self._mcs_reappearance_count / max(1, self._mcs_unique_denied_total)
            ),
            "denied_hashes_tracked": len(self._mcs_denied_hashes),
            "denied_untracked_this_step": d.get(
                "mcs_denied_untracked_this_step", 0
            ),
            "denial_track_cap_per_step": self.MCS_DENIAL_TRACK_CAP_PER_STEP,
            "budget_per_step": MCS_BUDGET_PER_STEP,
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        # [MCS DIAGNOSTIC] counters live in self.diagnostics (auto-exposed);
        # reappearance tracking lives in instance attrs — merge both.
        out = dict(self.diagnostics)
        out["mcs_reappearance_count"] = self._mcs_reappearance_count
        out["mcs_unique_denied_total"] = self._mcs_unique_denied_total
        out["mcs_reappearance_rate"] = (
            self._mcs_reappearance_count / max(1, self._mcs_unique_denied_total)
        )
        out["mcs_denied_hashes_tracked"] = len(self._mcs_denied_hashes)
        return out

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Clear episode-local state. Core library and reduction templates
        persist across episodes (§4.3)."""
        self._current_step = -1
        self._mcs_calls_this_step = 0
        self._sig_cache.clear()


# =============================================================================
# CATEGORY MANAGER (merged from category_manager.py)
# =============================================================================