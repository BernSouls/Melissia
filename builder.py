"""
builder.py — Spine-Guided Commitment Search over candidate subgraphs.

Implements §4 of the v31 update specification (15_UpdateTOBLUEPRINT(17).md).

Determinism: every comparison, tie-break, and ordering is by 128-bit canonical
hash (bytes, lexicographic) or by canonical serialized bytes. Same input
sequence produces byte-identical output, run to run.

Dependencies on other modules:
  - primitives: hash_128, serialize_*, hash_64, clamp
  - parameters: single source of truth for every §4.22 constant (the
    module-level names below are bindings, not independent definitions)
  - sensory_floor: Detection, TypedSignature, serialize_detection
  - concept_graph: Concept, ConceptStatus, ConceptFamily, CrossSection, Edge
  - binder: EdgeType, SELF_BINDER_TYPES, _node_signature_hash, _map_family

[SPEC REQUIREMENT] binder must export `_map_family(original_detections, edges)
-> Optional[ConceptFamily]`. If it does not, importing this module fails with
an explicit ImportError (fail-loudly; architectural drift is not tolerated).
"""

from __future__ import annotations

import math
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from core.hashing import (  # noqa: F401
    clamp,
    hash_128,
    hash_64,
    serialize_bytes,
    serialize_int,
    serialize_tuple,
    serialize_value,
)

from sensory_floor import Detection, TypedSignature, serialize_detection
from concept_graph import (
    Concept,
    ConceptStatus,
    ConceptFamily,
    CrossSection,
)
from orchestration.binder import _map_family
from core.hashing import _node_signature_hash
from core.types import EdgeType
from core.handoff import ConsolidationHandoff
from core.graph_types import Subgraph, anchor_positions_for


# =============================================================================
# CONSTITUTIONAL PARAMETERS (§4.22) — bound from parameters.py
# =============================================================================
# Single source of truth: every value below is the parameters.py binding.
# The former module-level literals were a parallel definition that could
# silently diverge from the canonical table (hardcoding drift).

import parameters as P

TAU_FRESH: float = P.TAU_FRESH
ETA_FRESH: float = P.ETA_FRESH
W_STALL: int = P.W_STALL
STALL_THRESHOLD: float = P.STALL_THRESHOLD
KAPPA_REF: float = P.KAPPA_REF
THETA_CURIOUS: float = P.THETA_CURIOUS
THETA_STERILE: float = P.THETA_STERILE
TAU_EXCURSION_MAX: int = P.TAU_EXCURSION_MAX
TAU_EPISTEMIC_MAX: int = P.TAU_EPISTEMIC_MAX
K_MAIN: int = P.K_MAIN
THETA_BASIN_FULL_RECOMPUTE: float = P.THETA_BASIN_FULL_RECOMPUTE
THETA_PRUNE_FRONTIER: float = P.THETA_PRUNE_FRONTIER
K_PRUNE_FRONTIER: int = P.K_PRUNE_FRONTIER
# [Batch D / P1-1] Endogenous cadence revision bounds.
W_STALL_REVISION: int = P.W_STALL_REVISION
K_MAIN_REVISION: int = P.K_MAIN_REVISION
TAU_EXCURSION_REVISION: int = P.TAU_EXCURSION_REVISION
TAU_EPISTEMIC_REVISION: int = P.TAU_EPISTEMIC_REVISION
K_PRUNE_FRONTIER_REVISION: int = P.K_PRUNE_FRONTIER_REVISION
K_FERTILE_REVISION: int = P.K_FERTILE_REVISION

F_RECURSIVE_DEPTH: int = P.F_RECURSIVE_DEPTH
LAMBDA_F_RECURSIVE: float = P.LAMBDA_RECURSIVE
THETA_FERTILE_PERCENTILE: float = P.THETA_FERTILE_PERCENTILE
RF_MIN_FERTILE: float = P.RF_MIN_FERTILE
K_FERTILE: int = P.K_FERTILE
FERTILE_MIN_POPULATION: int = P.FERTILE_MIN_POPULATION

CORE_SIZE_CAP_NODES: int = P.CORE_SIZE_CAP_NODES
CORE_SIZE_CAP_EDGES: int = P.CORE_SIZE_CAP_EDGES
BUILDER_CANDIDATE_MAX: int = P.BUILDER_CANDIDATE_MAX

THETA_OPP_SUBGRAPH: int = P.THETA_OPP_SUBGRAPH
THETA_CONF_SUBGRAPH: int = P.THETA_CONF_SUBGRAPH
THETA_SPAN_SUBGRAPH: int = P.THETA_SPAN_SUBGRAPH
THETA_ANCHOR_SUBGRAPH_SINGLE: int = P.THETA_ANCHOR_SUBGRAPH_SINGLE
THETA_ANCHOR_SUBGRAPH_DEFAULT: int = P.THETA_ANCHOR_SUBGRAPH_DEFAULT

W_SUCCESSOR: int = P.W_SUCCESSOR
THETA_PAIR_TRIALS: int = P.THETA_PAIR_TRIALS

TRANSPOSITION_TABLE_MAX: int = P.TRANSPOSITION_TABLE_MAX
FRONTIER_MAX: int = P.FRONTIER_MAX

TAU_MIN: float = P.TAU_MIN
EPS_REF: float = P.EPS_REF

W_SEEDING_MAX: int = P.W_SEEDING_MAX
MAX_ANCHORS_PER_CANDIDATE: int = P.MAX_ANCHORS_PER_CANDIDATE
MAX_SURPRISE_QUEUE: int = P.MAX_SURPRISE_QUEUE
ETA_SURPRISE: float = P.ETA_SURPRISE
BUILDER_EXPANSIONS_PER_STEP: int = P.BUILDER_EXPANSIONS_PER_STEP
TAU_SAMPLE_SIZE: int = P.TAU_SAMPLE_SIZE

ALPHA_V: float = P.BUILDER_ALPHA
BETA_V: float = P.BUILDER_BETA
GAMMA_V: float = P.BUILDER_GAMMA

# §4.17 per-family anchor thresholds: single-anchor families per spec.
_SINGLE_ANCHOR_FAMILIES: FrozenSet[str] = frozenset(
    {
        "SPATIAL_OBJECT",
        "EVENT_CAUSAL",
        "TEMPORAL_CHAIN",
    }
)

# §4.14 non-family-forming edge types — bound from the parameters.py
# classification (same source binder.py and concept_graph.py consume).
_NON_FAMILY_FORMING_EDGE_TYPES: FrozenSet[EdgeType] = frozenset(
    et for et in EdgeType if et.name in P.NON_FAMILY_FORMING_EDGE_TYPES
)

_EPS = P.EPSILON  # §2.1.20: all divisions use x/(y+ε), ε = 1e-9


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class BuilderConfig:
    """Runtime-tunable Builder configuration. Defaults match §4.22."""

    tau_fresh: float = TAU_FRESH
    eta_fresh: float = ETA_FRESH
    w_stall: int = W_STALL
    stall_threshold: float = STALL_THRESHOLD
    kappa_ref: float = KAPPA_REF
    theta_curious: float = THETA_CURIOUS
    theta_sterile: float = THETA_STERILE
    tau_excursion_max: int = TAU_EXCURSION_MAX
    tau_epistemic_max: int = TAU_EPISTEMIC_MAX
    k_main: int = K_MAIN
    # [Batch D / P1-1] endogenous revision bounds
    w_stall_revision: int = W_STALL_REVISION
    k_main_revision: int = K_MAIN_REVISION
    tau_excursion_revision: int = TAU_EXCURSION_REVISION
    tau_epistemic_revision: int = TAU_EPISTEMIC_REVISION
    k_prune_frontier_revision: int = K_PRUNE_FRONTIER_REVISION
    k_fertile_revision: int = K_FERTILE_REVISION
    theta_basin_full_recompute: float = THETA_BASIN_FULL_RECOMPUTE
    theta_prune_frontier: float = THETA_PRUNE_FRONTIER
    k_prune_frontier: int = K_PRUNE_FRONTIER
    f_recursive_depth: int = F_RECURSIVE_DEPTH
    lambda_f_recursive: float = LAMBDA_F_RECURSIVE
    theta_fertile_percentile: float = THETA_FERTILE_PERCENTILE
    rf_min_fertile: float = RF_MIN_FERTILE
    # [P0-2]
    fertile_admit_margin: float = P.FERTILE_ADMIT_MARGIN
    fertile_retighten_margin: float = P.FERTILE_RETIGHTEN_MARGIN
    # [P0-3]
    theta_min_present_ratio: float = P.THETA_MIN_PRESENT_RATIO
    k_fertile: int = K_FERTILE
    fertile_min_population: int = FERTILE_MIN_POPULATION
    core_size_cap_nodes: int = CORE_SIZE_CAP_NODES
    core_size_cap_edges: int = CORE_SIZE_CAP_EDGES
    builder_candidate_max: int = BUILDER_CANDIDATE_MAX
    theta_opp_subgraph: int = THETA_OPP_SUBGRAPH
    theta_conf_subgraph: int = THETA_CONF_SUBGRAPH
    theta_span_subgraph: int = THETA_SPAN_SUBGRAPH
    w_successor: int = W_SUCCESSOR
    theta_pair_trials: int = THETA_PAIR_TRIALS
    transposition_table_max: int = TRANSPOSITION_TABLE_MAX
    frontier_max: int = FRONTIER_MAX
    tau_min: float = TAU_MIN
    eps_ref: float = EPS_REF
    w_seeding_max: int = W_SEEDING_MAX
    max_anchors_per_candidate: int = MAX_ANCHORS_PER_CANDIDATE
    max_surprise_queue: int = MAX_SURPRISE_QUEUE
    eta_surprise: float = ETA_SURPRISE
    builder_expansions_per_step: int = BUILDER_EXPANSIONS_PER_STEP
    tau_sample_size: int = TAU_SAMPLE_SIZE
    alpha_v: float = ALPHA_V    # §4.6.2 constitutional weights
    beta_v: float = BETA_V
    gamma_v: float = GAMMA_V

    def __post_init__(self) -> None:
        if self.w_stall <= self.theta_conf_subgraph:
            raise ValueError(
                f"W_stall ({self.w_stall}) must exceed θ_conf_subgraph "
                f"({self.theta_conf_subgraph}) to avoid premature stall "
                f"detection. See §4.17 constraint."
            )
        # [P0-3] Guard the partial-observation threshold.
        if not (0.0 < self.theta_min_present_ratio <= 1.0):
            raise ValueError(
                f"theta_min_present_ratio must be in (0, 1], got "
                f"{self.theta_min_present_ratio}"
            )


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(eq=False)   # identity is canonical_hash; no value equality
class CandidateSubgraph:
    """A candidate subgraph in the Builder's search space (§4.3.1).

    Field provenance:
      - canonical_hash, cross_section_key: IMMUTABLE after creation.
      - n_conf: distinct-step re-observations (§4.17). Bumped at most once
        per step via `last_n_conf_step`. Zero for freshly created candidates.
      - last_n_conf_step: guard for "at most once per step" (§4.17).
    """

    canonical_hash: bytes
    cross_section_key: bytes
    members: FrozenSet[int]
    edges: FrozenSet[Tuple[int, int, EdgeType]]
    anchors: Tuple[Tuple[Any, int], ...]
    contexts: FrozenSet[bytes]
    t_created: int
    T_first: int
    T_last: int
    basin_id: int = -1
    parent_hash: Optional[bytes] = None
    construction_op: str = "SEED"
    original_detections: FrozenSet[Detection] = field(default_factory=frozenset)
    original_detection_hashes: FrozenSet[bytes] = field(default_factory=frozenset)
    n_conf: int = 0
    last_n_conf_step: int = -1
    V: float = 0.0
    V_step: int = -1
    low_v_steps: int = 0
    # [Batch D / P1-1] revision-based pruning bound fields.
    low_v_revision_events: int = 0
    last_low_v_rev: int = -1
    # §4.10.2: candidates produced during EPISTEMIC_EXCURSION do not count
    # for Spine stall detection.
    epistemic_origin: bool = False
    # Per-candidate caches (invalidated by step / detection-set growth).
    f_exp_cache: float = 0.0
    f_exp_step: int = -1
    # [P0-3] Running accumulator for fractional membership credit.
    member_accum: float = 0.0
    family_cache_n: int = -1
    family_cache: Optional[ConceptFamily] = None

    def size(self) -> int:
        return len(self.members) + len(self.edges)


@dataclass
class Basin:
    id: int
    members: Set[bytes] = field(default_factory=set)
    cumulative_fertility: float = 0.0
    reference_V: float = 0.0
    # [Bundle A] bounded reference_V history (deque maxlen=1000, created
    # lazily on first sample so the dataclass stays import-light).
    reference_V_history: Any = None


@dataclass
class CrystallizationTrace:
    path_hash: bytes
    basin_id: int
    t_crystallized: int
    best_successor_hash: Optional[bytes] = None
    best_successor_V: float = 0.0
    # [B1/D6] State hash at crystallization — the faithful replay start
    # for the Consolidator's §5.3.4 prune_path (the situation in which
    # the path actually occurred). s_t is P's crystallization state;
    # successor_s_t is recorded when the best successor binds.
    s_t: Optional[int] = None
    successor_s_t: Optional[int] = None


@dataclass
class BuilderOutput:
    cross_sections: List[CrossSection]
    path_outcome_pairs: List[Tuple[bytes, bytes]]
    diagnostics: Dict[str, Any]
    # [B1/D6] Rich consolidation handoffs: the same resolved (P, S)
    # pairs as path_outcome_pairs, packaged with INDUCED subgraphs and
    # crystallization states so the pipeline can call prune_path without
    # reconstruction guesswork. The hash pairs remain for diagnostics
    # and backward compatibility.
    handoffs: List[ConsolidationHandoff] = field(default_factory=list)


# =============================================================================
# CANONICAL ROLE TUPLE ENCODING (§4.14)
# =============================================================================


def _serialize_role_tuple(role: Tuple) -> bytes:
    """[SPEC COMPLIANCE §4.14] Canonical byte encoding of a role descriptor
    tuple (role_name, value, index, polarity) for deterministic sorting and
    set deduplication.

    NOTE: `primitives.py` does not export serialize_role_tuple. This module
    provides it locally; if primitives later exports an authoritative
    version, replace this definition with an import.
    """
    if not isinstance(role, tuple) or len(role) != 4:
        raise ValueError(f"role must be a 4-tuple, got {role!r}")
    role_name, value, index, polarity = role
    return serialize_tuple(
        [
            serialize_bytes(str(role_name).encode("utf-8")),
            serialize_value(value),
            serialize_int(int(index)),
            serialize_bytes(str(polarity).encode("utf-8")),
        ]
    )


# =============================================================================
# CANONICAL SUBGRAPH HASHING (§4.12)
# =============================================================================


def _wl_canonical_hash(
    members: FrozenSet[int],
    edges: FrozenSet[Tuple[int, int, EdgeType]],
    cg: Any,
) -> bytes:
    """Compute the 128-bit WL canonical hash of a typed, directed subgraph.

    [SPEC COMPLIANCE §4.12] The full refinement loop runs before any node-ID
    tie-break. Early injection of node ID (present in prior iterations) is
    rejected because it would bypass WL refinement for subgraphs with
    duplicate signatures. Node-ID tie-breaking is applied ONLY at the final
    canonical ordering when the partition remains non-discrete.
    """
    if not members:
        return hash_128(serialize_tuple([serialize_int(0)]))

    members_sorted: Tuple[int, ...] = tuple(sorted(members))

    node_sig: Dict[int, Tuple[str, bytes]] = {}
    for nid in members_sorted:
        sig = cg.get_pattern_node_signature(nid)
        if sig is None:
            node_sig[nid] = ("_PRUNED_", b"\x00" * 16)
        else:
            node_sig[nid] = (sig[0], sig[1])

    member_set = set(members_sorted)
    out_adj: Dict[int, List[Tuple[EdgeType, int, Tuple[float, ...]]]] = {nid: [] for nid in members_sorted}
    in_adj: Dict[int, List[Tuple[EdgeType, int, Tuple[float, ...]]]] = {nid: [] for nid in members_sorted}
    for src, dst, et in edges:
        if src in member_set and dst in member_set:
            edge_obj = getattr(cg, "get_edge", lambda s, d, t: None)(src, dst, et)
            if edge_obj is None and hasattr(cg, "edges"):
                edge_obj = cg.edges.get((src, dst, et))
            ep = getattr(edge_obj, "params", None) if edge_obj is not None else None
            p_bins = tuple(float(x) for x in ep.quantized) if ep is not None and hasattr(ep, "quantized") else ()
            out_adj[src].append((et, dst, p_bins))
            in_adj[dst].append((et, src, p_bins))

    # Initial partition strictly by (lens_type, signature_hash).
    sig_to_part: Dict[Tuple[str, bytes], int] = {}
    partition_of: Dict[int, int] = {}
    for nid in members_sorted:
        key = node_sig[nid]
        if key not in sig_to_part:
            sig_to_part[key] = len(sig_to_part)
        partition_of[nid] = sig_to_part[key]

    n = len(members_sorted)
    max_iters = max(1, n - 1)
    for _ in range(max_iters):
        colors: Dict[int, Tuple] = {}
        for nid in members_sorted:
            out_info = tuple(
                sorted((et.value, partition_of[nb], p_bins) for et, nb, p_bins in out_adj[nid])
            )
            in_info = tuple(
                sorted((et.value, partition_of[nb], p_bins) for et, nb, p_bins in in_adj[nid])
            )
            colors[nid] = (partition_of[nid], out_info, in_info)

        color_to_new: Dict[Tuple, int] = {}
        new_partition: Dict[int, int] = {}
        for nid in members_sorted:
            c = colors[nid]
            if c not in color_to_new:
                color_to_new[c] = len(color_to_new)
            new_partition[nid] = color_to_new[c]

        if all(new_partition[nid] == partition_of[nid] for nid in members_sorted):
            break

        # Deterministic renumbering by sorted group representative.
        group_sigs: Dict[int, List[Tuple[str, bytes]]] = defaultdict(list)
        for nid in members_sorted:
            group_sigs[new_partition[nid]].append(node_sig[nid])
        group_repr = {p: tuple(sorted(sigs)) for p, sigs in group_sigs.items()}
        sorted_groups = sorted(group_repr.items(), key=lambda kv: kv[1])
        p_to_new_p = {p: i for i, (p, _) in enumerate(sorted_groups)}
        partition_of = {nid: p_to_new_p[new_partition[nid]] for nid in members_sorted}

        if len(set(partition_of.values())) == n:
            break

    # [SPEC COMPLIANCE §4.12] Post-refinement tie-break. If the partition is
    # non-discrete, use node ID as the terminal discriminator. Otherwise use
    # (partition, (lens_type, sig_hash)) which is guaranteed discrete.
    if len(set(partition_of.values())) < n:
        ordered = sorted(members_sorted, key=lambda nid: (partition_of[nid], nid))
    else:
        ordered = sorted(
            members_sorted,
            key=lambda nid: (partition_of[nid], node_sig[nid]),
        )
    node_pos = {nid: i for i, nid in enumerate(ordered)}

    node_reprs = [
        serialize_tuple(
            [
                serialize_bytes(node_sig[nid][0].encode()),
                serialize_bytes(node_sig[nid][1]),
            ]
        )
        for nid in ordered
    ]

    edge_tuples: List[Tuple[int, int, int, Tuple[float, ...]]] = []
    for src, dst, et in edges:
        if src in node_pos and dst in node_pos:
            edge_obj = getattr(cg, "get_edge", lambda s, d, t: None)(src, dst, et)
            if edge_obj is None and hasattr(cg, "edges"):
                edge_obj = cg.edges.get((src, dst, et))
            ep = getattr(edge_obj, "params", None) if edge_obj is not None else None
            p_bins = tuple(float(x) for x in ep.quantized) if ep is not None and hasattr(ep, "quantized") else ()
            edge_tuples.append((node_pos[src], node_pos[dst], et.value, p_bins))
    edge_tuples.sort()
    edge_reprs = [
        serialize_tuple(
            [
                serialize_int(sp),
                serialize_int(dp),
                serialize_int(ev),
                serialize_tuple([serialize_value(b) for b in p_bins]),
            ]
        )
        for (sp, dp, ev, p_bins) in edge_tuples
    ]

    combined = serialize_tuple(
        [
            serialize_int(len(node_reprs)),
            serialize_tuple(node_reprs),
            serialize_int(len(edge_reprs)),
            serialize_tuple(edge_reprs),
        ]
    )
    return hash_128(combined)


def _negate_bytes(b: bytes) -> bytes:
    """Byte-wise complement so that 'lower canonical hash wins' becomes a
    MAXIMUM over the negated bytes — lets the deterministic spec tie-break
    compose into a single descending sort key tuple."""
    return bytes(0xFF ^ x for x in b)


def _canonical_detection_set_hash(detection_hashes: FrozenSet[bytes]) -> bytes:
    """Detection-set hash (§4.14)."""
    sorted_hashes = sorted(detection_hashes)
    return hash_128(serialize_tuple([serialize_bytes(h) for h in sorted_hashes]))


def _detection_hash(det: Detection) -> bytes:
    return hash_128(serialize_detection(det))


# =============================================================================
# FAMILY-FORMING TEST (§4.14)
# =============================================================================


def _is_family_forming(candidate: CandidateSubgraph) -> bool:
    if not candidate.edges:
        return False
    for _src, _dst, et in candidate.edges:
        if et not in _NON_FAMILY_FORMING_EDGE_TYPES:
            return True
    return False


# =============================================================================
# NODE LAYER LOOKUP (for synthetic edge detections §4.14)
# =============================================================================


def _node_layer(cg: Any, nid: int) -> int:
    """Recover the highest layer of any live concept containing this node.
    Returns 0 if no live concept contains the node.
    """
    cids = cg._node_to_concepts.get(nid, set())  # noqa: SLF001 (ConceptGraph exposes this index)
    best = 0
    for cid in cids:
        concept = cg.concepts.get(cid)
        if concept is None:
            continue
        if concept.status == ConceptStatus.TOMBSTONED:
            continue
        if concept.layer > best:
            best = concept.layer
    return best


# =============================================================================
# SYNTHETIC EDGE DETECTIONS (§4.14)
# =============================================================================


def _synthesize_edge_detection(
    edge: Tuple[int, int, EdgeType],
    cg: Any,
) -> Optional[Detection]:
    """[SPEC COMPLIANCE §4.14] source_layer computed from node layers per
    edge type. Previously hardcoded to 0; corrected here.
    """
    src, dst, et = edge

    if et == EdgeType.SEQUENCE:
        src_layer = max(_node_layer(cg, src), _node_layer(cg, dst))
        sigma = TypedSignature.opaque(
            "SEQUENCE",
            hash_64(serialize_tuple([serialize_int(src), serialize_int(dst)])),
        )
        rho = (("antecedent", src, 0, "+"), ("consequent", dst, 1, "+"))
        return Detection(
            k="Sequence",
            source_layer=src_layer,
            scale=0,
            sigma=sigma,
            alpha=src,
            z=1.0,
            t=-1,
            rho=rho,
        )

    if et == EdgeType.CONTACT:
        a_id, b_id = (src, dst) if src < dst else (dst, src)
        sigma = TypedSignature.opaque(
            "EDGE",
            hash_64(serialize_tuple([serialize_int(a_id), serialize_int(b_id)])),
        )
        rho = (("edge", src, 0, "+"), ("edge", dst, 1, "+"))
        return Detection(
            k="Edge",
            source_layer=0,
            scale=0,
            sigma=sigma,
            alpha=(a_id, b_id),
            z=1.0,
            t=-1,
            rho=rho,
        )

    if et in (EdgeType.TOPOLOGY, EdgeType.DEGREE, EdgeType.CO_MOTION):
        tag = {
            EdgeType.TOPOLOGY: "TOPOLOGY",
            EdgeType.DEGREE: "DEGREE",
            EdgeType.CO_MOTION: "CO_MOTION",
        }[et]
        lens = {
            EdgeType.TOPOLOGY: "Topology",
            EdgeType.DEGREE: "Degree",
            EdgeType.CO_MOTION: "Co-motion",
        }[et]
        sigma = TypedSignature.opaque(
            tag,
            hash_64(
                serialize_tuple(
                    [
                        serialize_int(src),
                        serialize_int(dst),
                        serialize_int(et.value),
                    ]
                )
            ),
        )
        rho = (("src", src, 0, "+"), ("dst", dst, 1, "+"))
        return Detection(
            k=lens,
            source_layer=0,
            scale=0,
            sigma=sigma,
            alpha=src,
            z=1.0,
            t=-1,
            rho=rho,
        )

    return None


# =============================================================================
# SUBGRAPH → CROSSSECTION CONVERSION (§4.14)
# =============================================================================


def subgraph_to_cross_section(
    candidate: CandidateSubgraph,
    cg: Any,
) -> Optional[CrossSection]:
    """Convert a CandidateSubgraph to a CrossSection (§4.14).

    Family resolution is delegated to `binder._map_family`. Role descriptors
    are sorted by `_serialize_role_tuple` (§4.14).
    """
    if not _is_family_forming(candidate):
        return None

    family = _map_family(
        candidate.original_detections,
        candidate.edges,
        diag=cg.diagnostics,
    )
    cg.diagnostics["map_family_checked"] = (
        cg.diagnostics.get("map_family_checked", 0) + 1
    )
    if family is None:
        return None

    alpha_counts: Dict[bytes, Tuple[Any, int]] = defaultdict(lambda: (None, 0))
    for d in candidate.original_detections:
        # Canonical serialization only. The former repr() fallback embedded
        # memory addresses (<object at 0x...>) into the anchor tie-break,
        # poisoning cross-run determinism (§9). A non-serializable alpha is
        # an upstream contract violation and must fail loudly.
        alpha_key = serialize_value(d.alpha)
        prev_alpha, prev_count = alpha_counts[alpha_key]
        if prev_alpha is None:
            alpha_counts[alpha_key] = (d.alpha, prev_count + 1)
        else:
            alpha_counts[alpha_key] = (prev_alpha, prev_count + 1)

    if not alpha_counts:
        return None

    def _alpha_tb(kv: Tuple[bytes, Tuple[Any, int]]) -> Tuple[int, bytes]:
        return (-kv[1][1], kv[0])

    best_alpha_key, _ = sorted(alpha_counts.items(), key=_alpha_tb)[0]
    best_alpha = alpha_counts[best_alpha_key][0]

    # [SPEC COMPLIANCE §4.14] `detections` is the union of
    # original_detections across all members — synthetic edge detections
    # are produced SOLELY to populate roleDescriptors and must not enter
    # `detections` (they would contaminate step-21 candidate-statistic
    # matching, orthogonality tests, and sourceLayers with role-grammar
    # signatures that no lens ever emitted).
    detections: FrozenSet[Detection] = candidate.original_detections

    role_set: Set[Tuple] = set()
    for d in candidate.original_detections:
        for role in d.rho:
            role_set.add(tuple(role))
    # (b) role annotations derived from S's edges: one synthetic detection
    # per order-1 edge, contributing its role-grammar rho entries. The rho
    # 4-tuples are the grammar-conformant encoding of the spec's
    # (edge_type, src, dst, idx, polarity) annotation: the edge type is
    # carried by the role names/sigma tag, the referents by the values, and
    # the ordinal by the deterministic sorted-edge iteration below.
    sorted_edges = sorted(candidate.edges, key=lambda e: (e[2].value, e[0], e[1]))
    for (edge_src, edge_dst, _et) in sorted_edges:
        synth = _synthesize_edge_detection((edge_src, edge_dst, _et), cg)
        if synth is not None:
            for role in synth.rho:
                role_set.add(tuple(role))
    role_descriptors: Tuple[Tuple, ...] = tuple(
        sorted(role_set, key=_serialize_role_tuple)
    )

    # §4.14: sourceLayers = set of source_layer values of the MEMBER
    # detections (original_detections), not of synthetic edge detections.
    source_layers: Set[int] = {d.source_layer for d in detections}

    return CrossSection(
        family=family.name,
        key=candidate.cross_section_key,
        anchor=best_alpha,
        detections=frozenset(detections),
        roleDescriptors=role_descriptors,
        sourceLayers=frozenset(source_layers),
    )


# =============================================================================
# BUILDER
# =============================================================================


class Builder:
    """Spine-Guided Commitment Search over candidate subgraphs."""

    def __init__(
        self,
        concept_graph: Any,
        fertility_engine: Any,
        config: Optional[BuilderConfig] = None,
        prediction_module: Optional[Any] = None,
    ) -> None:
        self.cg = concept_graph
        self.fert = fertility_engine
        self.cfg = config if config is not None else BuilderConfig()
        self.prediction = prediction_module

        # Search space.
        self.frontier: List[CandidateSubgraph] = []
        self.frontier_hashes: Set[bytes] = set()
        # O(1) hash → candidate view, maintained at every frontier mutation
        # site (add / expand-remove / evict / prune / crystallize-remove).
        self._frontier_by_hash: Dict[bytes, CandidateSubgraph] = {}
        # §4.11.1 transposition table: canonical_hash →
        # (step_of_expansion, n_conf_at_expansion). The n_conf component
        # supports the bootstrap evidence-unlock rule (see _expand_mode).
        self.transposition_table: Dict[bytes, Tuple[int, bytes, int, int]] = {}
        # §4.12 WL-hash memoization, bounded FIFO at TRANSPOSITION_TABLE_MAX.
        self._wl_memo: "OrderedDict[Tuple[FrozenSet[int], FrozenSet[Tuple[int, int, EdgeType]]], bytes]" = OrderedDict()
        # [WL LRU] capacity bound (single source: cfg.transposition_table_max,
        # wired from parameters.TRANSPOSITION_TABLE_MAX).
        self._wl_memo_max: int = self.cfg.transposition_table_max
        self.crystallization_traces: List[CrystallizationTrace] = []
        self.pair_trial_counts: Dict[Tuple[bytes, bytes], int] = defaultdict(int)

        # Basins / Spine.
        self.basins: Dict[int, Basin] = {}
        self.next_basin_id: int = 0
        self.spine_basin_id: Optional[int] = None
        self.last_basin_recompute_step: int = -K_MAIN
        self.last_spine_recompute_step: int = -K_MAIN
        # §4.9/§4.10.3: Spine basin split during EXCURSION/EPISTEMIC defers
        # SPINE_REASSIGN to the excursion's resolution.
        self._spine_reassign_deferred: bool = False
        # O(1) canonical_hash → basin_id reverse index (§4.9 incremental
        # updates and §4.15.2 same-basin successor matching).
        self._hash_to_basin: Dict[bytes, int] = {}

        # Incremental basin update state.
        self._dirty_basin_nodes: Set[int] = set()
        self._subgraph_index: Dict[int, Set[bytes]] = defaultdict(set)
        self._subgraph_members_cache: Dict[bytes, FrozenSet[int]] = {}

        # State machine.
        self.mode: str = "SEEDING"
        self.stall_counter: int = 0
        self.t_excursion_start: int = -1
        self.t_epistemic_start: int = -1
        self.excursion_basin_id: Optional[int] = None
        self._excursion_entry_concepts: FrozenSet[int] = frozenset()
        # §4.10.3 sterility: no excursion-basin candidate at V ≥ θ_sterile
        # for W_stall CONSECUTIVE steps (per-step counter, not one-shot).
        self._excursion_sterile_counter: int = 0
        self._last_spine_crystallization_step: int = -1
        self.current_scope_members: FrozenSet[int] = frozenset()
        self.current_scope_expanded: Set[bytes] = set()

        # Surprise queue.
        self.surprise_queue: deque = deque()
        # [SPEC COMPLIANCE §4.11] Current surprise magnitude for
        # EPISTEMIC_EXCURSION priority. Set on entry, cleared on exit.
        self.current_surprise_value: float = 0.0

        # Crystallization snapshot.
        self.crystallized_snapshot: FrozenSet[bytes] = frozenset()

        # Caches.
        self._theta_fertile: float = RF_MIN_FERTILE
        self._theta_fertile_step: int = -(10**9)
        # [P0-2] Cold-start rescue state.
        self._fertile_relaxed: bool = False
        # [Bundle A] fertile-set membership turnover: bumped when the H
        # membership snapshot changes between k_fertile recomputations.
        self._fertile_set_prev = None
        self.fertile_set_turnover_count: int = 0
        # [Bundle A] spine reference_V history sampled on edge_revision change.
        self._last_edge_rev_sample: int = -1
        # [Batch D / P1-1] Endogenous cadence checkpoints (MED-05).
        _g0 = (
            self.cg.graph_revision()
            if hasattr(self.cg, "graph_revision")
            else 0
        )
        self._spine_last_progress_t: int = -1
        self._spine_last_progress_rev: int = _g0
        self._excursion_last_progress_t: int = -1
        self._excursion_last_progress_rev: int = _g0
        self._excursion_v_max_seen: float = -1.0
        self._epistemic_last_progress_t: int = -1
        self._epistemic_last_progress_rev: int = _g0
        self._last_basin_rev: int = _g0
        self._last_fertile_rev: int = _g0
        self._pattern_adj_cache: Dict[int, Set[int]] = {}
        self._pattern_adj_signature: Tuple[int, int, int] = (-1, -1, -1)
        # Directed successor adjacency for F_recursive (§4.7 direction spec).
        self._pattern_adj_dir_cache: Dict[int, Set[int]] = {}
        self._pattern_adj_dir_signature: Tuple[int, int, int] = (-1, -1, -1)
        # Per-step node-level caches for R(v) and proximity(v) — the
        # spec-mandated O(1) read cost for cached entries (§4.7); topology
        # changes between steps invalidate by step key.
        self._reach_cache: Dict[int, float] = {}
        self._reach_cache_step: int = -1
        # Step-scoped memo for _induced_edges over member sets (used by
        # the exact child-construction algebra; mirrors _reach_cache).
        self._induced_memo: Dict[FrozenSet[int], FrozenSet[Tuple[int, int, Any]]] = {}
        self._induced_memo_step: int = -1
        # Step-scoped memo for expansion footprints (transposition unlock).
        self._footprint_memo: Dict[FrozenSet[int], bytes] = {}
        self._footprint_memo_step: int = -1
        self._prox_cache: Dict[int, float] = {}
        self._prox_cache_step: int = -1
        self._tau_cache: float = TAU_MIN
        self._tau_cache_step: int = -(10**9)
        self._tau_active_size: int = 0
        self._tau_active_set: FrozenSet[int] = frozenset()
        self._fertile_cache: List[Tuple[int, Concept]] = []
        self._fertile_cache_step: int = -1
        self._tombstone_cache: Dict[bytes, int] = {}
        self._tombstone_cache_step: int = -1

        # Per-step detection state.
        self._step_detection_map: Dict[int, Detection] = {}
        self.node_detection_map: Dict[int, Detection] = {}
        self._current_s_t_bytes: bytes = b""
        self._current_step: int = -1

        # Diagnostics.
        self.diagnostics: Dict[str, Any] = {
            "dropped_no_family": 0,
            "seeded_total": 0,
            "expanded_total": 0,
            "crystallized_total": 0,
            "frontier_evictions": 0,
            "transposition_evictions": 0,
            "uncrystallizable_scaffold_count": 0,
            "pairs_emitted_total": 0,
            "pair_trials_purged": 0,
            "incremental_basin_updates": 0,
            "full_basin_recomputes": 0,
            "basin_splits": 0,
            "basin_split_checks": 0,
            "handoffs_emitted": 0,
            "handoffs_unresolvable": 0,
            "epistemic_seeds": 0,
            "epistemic_scope_empty_dropped": 0,
            "membership_reobservations": 0,
            "membership_reobservation_partial": 0,
            # [P0 v4 Phase 4] all nine v4 counters consolidated here
            # (Issue 3 fix: membership_reobservation_edge_rejected moved
            # from Phase 7 to this list; Issue 4 fix: renamed to
            # theta_fertile_empty_state_recomputes — per-recompute, not
            # per-step; Issue 6 fix: prune_frontier_active_steps makes the
            # pruning-guard transition observable).
            "membership_reobservation_edge_rejected": 0,
            "fertile_rescue_fired": 0,
            "fertile_rescue_top_val": 0.0,
            "fertile_rescue_fired_meaningful": 0,
            "fertile_rescue_fired_marginal": 0,
            "bootstrap_gate_active_steps": 0,
            "theta_fertile_empty_state_recomputes": 0,
            "prune_frontier_active_steps": 0,
            "cold_start_max_n_conf": 0,
            "spine_splits_deferred": 0,
            "current_step": -1,
        }

        # Deferred seeding-start tracker.
        self._seeding_start: Optional[int] = None

        # Trace-opening guard: (path_hash -> t_opened) so we never open the
        # same trace twice.
        self._opened_traces: Set[bytes] = set()

        # [Batch 4 / Phase 3] CDP predicate probe state (read-side only).
        self._cdp_predicate_probe_enabled: bool = False
        self._cdp_predicate_max: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------------------------------
    # Public API (§4.21)
    # -------------------------------------------------------------------------

    def step(
        self,
        t: int,
        D_bind: Set[Detection],
        s_t: int,
    ) -> BuilderOutput:
        self.diagnostics["current_step"] = t
        self._current_step = t
        s_t_bytes: bytes = serialize_int(s_t)

        # 0. Drain surprise signals from Prediction (§8.11 step 8c) — at
        # the START of the Builder step. drain_surprises() is contractually
        # non-raising (§8.6.5); a failure here is a defect and propagates
        # (the former silent `except Exception: signals = []` swallowed it).
        if self.prediction is not None:
            for sig in self.prediction.drain_surprises():
                if len(self.surprise_queue) >= self.cfg.max_surprise_queue:
                    self.surprise_queue.popleft()
                    self.diagnostics["surprise_queue_evicted"] = (
                        self.diagnostics.get("surprise_queue_evicted", 0) + 1
                    )
                self.surprise_queue.append(sig)

        # Record step state (detection map for anchor recovery, §4.3.1).
        self._record_step_state(D_bind, s_t_bytes)

        # §4.21 internal pipeline order (normative):
        # 1. Seed candidates from D_bind (§4.13).
        self._seed_from_detections(t, D_bind, s_t_bytes)

        # 1b. Structural re-observation (SANCTIONED v31 EXTENSION — see
        # _check_frontier_reobservation). Runs before expansion so
        # confirmations are captured before a candidate is expanded away.
        self._check_frontier_reobservation(t, s_t_bytes)

        # 2. Detect newly crystallized subgraphs by diffing the ConceptGraph
        #    crystallized set; open traces (§4.15.1). Trace successor
        #    matching requires basin membership, so trace UPDATES run after
        #    the basin recomputation below (basins are batch-recomputed once
        #    per step per §4.9; matching before that would pair against the
        #    previous step's basin state).
        newly_crystallized_hashes = self._detect_new_crystallizations(t, s_t)

        # 2a. [Seam 3/D1] Immediate basin splitting under edge removal.
        #     The ConceptGraph records every edge removal / node prune
        #     (step_28 inactivity prunes, pattern-node TTL prunes); the
        #     Builder drains those events here — once per step, batched
        #     (K5) — and re-checks the affected basins for connectivity
        #     splits instead of waiting up to K_main = 500 steps for the
        #     full recompute (§4.9 "immediate split handling").
        removed_nodes = self.cg.drain_removed_structure_nodes()
        if removed_nodes:
            self._check_basin_splits(removed_nodes)
            self._dirty_basin_nodes.update(removed_nodes)

        # 3. Incremental basin recomputation if topology changed (§4.9).
        self._maybe_recompute_basins(t)

        # 2b. Update open traces with newly crystallized successors
        #     (§4.15.2 — basin membership now current).
        self._update_traces(t, newly_crystallized_hashes, s_t)

        # 4. Update state machine (stall check, transitions, excursion
        #    timeout) (§4.10).
        self._update_state_machine(t)

        # 5-6. Select next candidate per mode and expand (§4.11). Steps
        #      1-5 execute atomically before expansion; no basin
        #      recomputation occurs between selection and expansion.
        self._expand_step(t)

        # 7. Update reference_V (§4.10.1).
        self._update_reference_V(t)

        # 8. Identify candidates crossing the crystallization predicate
        #    (§4.17). Also refreshes the scaffold-population gauge: the
        #    spec tracks `uncrystallizable_scaffold_count` as the frontier
        #    population of non-family-forming candidates (§4.14), not a
        #    per-step emission skip count.
        eligible = self._collect_eligible_for_crystallization(t)

        # 9. Convert to CrossSection objects (§4.14).
        cross_sections: List[CrossSection] = []
        for cand in eligible:
            cs = subgraph_to_cross_section(cand, self.cg)
            if cs is None:
                self.diagnostics["dropped_no_family"] += 1
                continue
            cross_sections.append(cs)

        # 10. Check and emit path/outcome pairs (§4.15.3) + B1 handoffs.
        path_outcome_pairs, handoffs = self._resolve_traces(t)

        # 11. Update crystallized_snapshot.
        self.crystallized_snapshot = self._collect_crystallized_hashes()

        # Frontier lifecycle pruning (§4.11.1) — exact spec rule:
        # V < θ_prune_frontier for K_prune_frontier consecutive steps.
        self._prune_frontier(t)

        # Bound _opened_traces to the live crystallized set (a
        # re-crystallization of an evicted hash may open a fresh trace).
        self._opened_traces.intersection_update(self.crystallized_snapshot)

        # Purge pair trial counters whose structures are no longer
        # crystallized — trials count episodes of EXISTING structure; a
        # dead (evicted/tombstoned) hash's counter is meaningless state.
        # Bounded: registry size ≤ live crystallized pairs.
        dead_pairs = [
            key
            for key in self.pair_trial_counts
            if key[0] not in self.crystallized_snapshot
            or key[1] not in self.crystallized_snapshot
        ]
        for key in dead_pairs:
            del self.pair_trial_counts[key]
        self.diagnostics["pair_trials_purged"] += len(dead_pairs)

        diagnostics = {
            "mode": self.mode,
            "spine_basin_id": self.spine_basin_id,
            "reference_V": (
                self.basins[self.spine_basin_id].reference_V
                if self.spine_basin_id is not None
                and self.spine_basin_id in self.basins
                else 0.0
            ),
            "frontier_size": len(self.frontier),
            "transposition_table_size": len(self.transposition_table),
            "basin_count": len(self.basins),
            "stall_counter": self.stall_counter,
            "theta_fertile": self._theta_fertile,
            "tau": self._tau_cache,
            "surprise_queue_size": len(self.surprise_queue),
            "dirty_basin_nodes": len(self._dirty_basin_nodes),
        }

        return BuilderOutput(
            cross_sections=cross_sections,
            path_outcome_pairs=path_outcome_pairs,
            diagnostics=diagnostics,
            handoffs=handoffs,
        )

    def get_frontier(self) -> List[CandidateSubgraph]:
        return list(self.frontier)

    def get_spine(self) -> Optional[Basin]:
        if self.spine_basin_id is None:
            return None
        return self.basins.get(self.spine_basin_id)

    def get_mode(self) -> str:
        return self.mode

    def enable_cdp_predicate_probe(self) -> None:
        """[Batch 4 / Phase 3] Enable read-side CDP predicate probe."""
        self._cdp_predicate_probe_enabled = True

    def dump_cdp_predicate_probe(self, path: str) -> None:
        """[Batch 4 / Phase 3] Dump per-hash max-value probe records."""
        import json
        with open(path, "w") as f:
            json.dump(self._cdp_predicate_max, f, indent=2)

    def get_transposition_table_size(self) -> int:
        return len(self.transposition_table)

    def get_diagnostics(self) -> Dict[str, Any]:
        """§4.21: diagnostic state — counters plus the live search state
        (frontier contents size, Spine identity, current mode,
        transposition table size, drop rates)."""
        out = dict(self.diagnostics)
        out.update(
            {
                "mode": self.mode,
                "spine_basin_id": self.spine_basin_id,
                "frontier_size": len(self.frontier),
            # [MCS DIAGNOSTIC] Live edge/node counts for correlating MCS
            # demand with graph size (transient vs structural demand).
            # frontier_size already present above; not duplicated.
            "live_edge_count": len(self.cg.edges),
            "live_node_count": len(self.cg._pattern_node_key_by_id),
                "transposition_table_size": len(self.transposition_table),
                "basin_count": len(self.basins),
                "stall_counter": self.stall_counter,
                "theta_fertile": self._theta_fertile,
                "tau": self._tau_cache,
                "surprise_queue_size": len(self.surprise_queue),
                "open_traces": len(self.crystallization_traces),
                "reference_V": (
                    self.basins[self.spine_basin_id].reference_V
                    if self.spine_basin_id is not None
                    and self.spine_basin_id in self.basins
                    else 0.0
                ),
            }
        )
        return out

    # -------------------------------------------------------------------------
    # Per-step detection state
    # -------------------------------------------------------------------------

    def _record_step_state(
        self,
        D_bind: Set[Detection],
        s_t_bytes: bytes,
    ) -> None:
        self._current_s_t_bytes = s_t_bytes
        self._step_detection_map = {}
        for d in sorted(D_bind, key=lambda dd: serialize_detection(dd)):
            sig_hash = _node_signature_hash(d.sigma)
            nid = self.cg.get_or_create_pattern_node(
                d.k, sig_hash, signature=d.sigma
            )
            if nid not in self._step_detection_map:
                self._step_detection_map[nid] = d
            if nid not in self.node_detection_map:
                self.node_detection_map[nid] = d

        # Bound node_detection_map growth (no unbounded state): sweep on
        # the parameters-derived cadence, and only when the map has bloated
        # past the parameters-derived factor of the live node count.
        t = self._current_step
        if t > 0 and t % P.NODE_DETECTION_SWEEP_PERIOD == 0:
            live_nodes = self._live_pattern_node_ids()
            if len(self.node_detection_map) > P.NODE_DETECTION_BLOAT_FACTOR * len(
                live_nodes
            ):
                self.node_detection_map = {
                    k: v
                    for k, v in self.node_detection_map.items()
                    if k in live_nodes
                }

    def _live_pattern_node_ids(self) -> Set[int]:
        """Live pattern-node IDs (§1.11 store). Consolidated access point:
        prefers a public ConceptGraph accessor when present, falls back to
        the maintained index. Flagged for concept_graph.py to expose
        `pattern_node_ids()` publicly (module-boundary rule §1.10.3)."""
        accessor = getattr(self.cg, "pattern_node_ids", None)
        if callable(accessor):
            return set(accessor())
        return set(self.cg._pattern_node_key_by_id.keys())  # noqa: SLF001

    def _canonical_hash(
        self,
        members: FrozenSet[int],
        edges: FrozenSet[Tuple[int, int, EdgeType]],
    ) -> bytes:
        """§4.12 WL canonical hash with the spec-mandated memoization:
        keyed by (frozenset(members), frozenset(edges)), bounded FIFO at
        TRANSPOSITION_TABLE_MAX entries. Pure function of the subgraph and
        the immutable node identities, so the memo is safe across steps."""
        key = (members, edges)
        cached = self._wl_memo.get(key)
        if cached is not None:
            # [WL LRU] move accessed entry to end (most recently used);
            # hot subgraphs survive, cold ones evict first. Hash output
            # unchanged — only eviction order differs from former FIFO.
            self._wl_memo.move_to_end(key)
            return cached
        h = _wl_canonical_hash(members, edges, self.cg)
        self._wl_memo[key] = h
        # LRU eviction: remove oldest (first) entry while over capacity.
        while len(self._wl_memo) > self._wl_memo_max:
            self._wl_memo.popitem(last=False)
        return h

    def _resolve_detection_for_node(
        self,
        nid: int,
        t: int,
    ) -> Optional[Detection]:
        """[SPEC COMPLIANCE §4.3.1] Anchor recovery prioritizes the current
        step's detection, falling back to the persistent node → detection
        map so persistent nodes observed in earlier steps still supply an
        anchor on extension.
        """
        d = self._step_detection_map.get(nid)
        if d is not None:
            return d
        return self.node_detection_map.get(nid)

    # -------------------------------------------------------------------------
    # §4.13 Seeding
    # -------------------------------------------------------------------------

    def _seed_from_detections(
        self,
        t: int,
        D_bind: Set[Detection],
        s_t_bytes: bytes,
    ) -> None:
        for d in sorted(D_bind, key=lambda dd: serialize_detection(dd)):
            sig_hash = _node_signature_hash(d.sigma)
            nid = self.cg.get_or_create_pattern_node(
                d.k, sig_hash, signature=d.sigma
            )
            det_hash = _detection_hash(d)

            members = frozenset({nid})
            # §4.13 (literal): a seeded candidate has edges = {}.
            # Live induced edges enter candidate edge sets through the §4.4
            # construction operations: EXTEND scans the NEW node's full
            # adjacency, and EDGE_ADD adds missing intra-member edges to an existing subgraph.
            # Seeding with live induced edges instead would split a node's evidence
            # across two canonical identities (({n}, ∅) vs ({n}, {edges}))
            # as the edges fire and decay, diluting the N_opp/N_conf
            # accumulation §4.17 crystallization depends on.
            edges: FrozenSet[Tuple[int, int, EdgeType]] = frozenset()
            canonical = self._canonical_hash(members, edges)

            existing = self._find_frontier_by_hash(canonical)
            if existing is not None:
                existing.T_last = t
                if s_t_bytes not in existing.contexts:
                    existing.contexts = existing.contexts | {s_t_bytes}
                existing.original_detections = existing.original_detections | {d}
                existing.original_detection_hashes = (
                    existing.original_detection_hashes | {det_hash}
                )
                existing.anchors = self._append_anchor(
                    existing.anchors,
                    (d.alpha, t),
                )
                self._bump_n_conf(existing, t)
                existing.V_step = -1
                continue

            if canonical in self.crystallized_snapshot:
                continue
            if self._is_tombstoned(canonical):
                continue
            # [SPEC COMPLIANCE §4.13] The transposition table gates
            # EXPANSION, not seeding. A re-seed of a previously expanded
            # hash is a legal N_conf increment; the spec mechanism that
            # keeps such a re-seed from stealing expansion budget is the
            # transposition skip in _expand_mode (a candidate whose hash is
            # already in the table is never selected for expansion). The
            # former 3-step "expansion cooldown" was an off-spec scheduler
            # hack contradicting this very rule; removed.
            cross_section_key = _canonical_detection_set_hash(frozenset({det_hash}))
            cand = CandidateSubgraph(
                canonical_hash=canonical,
                cross_section_key=cross_section_key,
                members=members,
                edges=edges,
                anchors=((d.alpha, t),),
                contexts=frozenset({s_t_bytes}),
                t_created=t,
                T_first=t,
                T_last=t,
                parent_hash=None,
                construction_op="SEED",
                original_detections=frozenset({d}),
                original_detection_hashes=frozenset({det_hash}),
                n_conf=0,
                last_n_conf_step=-1,
            )
            self._add_to_frontier(cand)
            self.diagnostics["seeded_total"] += 1

    def _bump_n_conf(self, cand: CandidateSubgraph, t: int) -> None:
        """[SPEC COMPLIANCE §4.17] Increment N_conf at most once per step."""
        if cand.last_n_conf_step < t:
            cand.n_conf += 1
            cand.last_n_conf_step = t
        # [P0-3] cold-start observability: track the best N_conf any
        # candidate reached during the seeding window.
        if t <= self.cfg.w_seeding_max and cand.n_conf > self.diagnostics.get(
            "cold_start_max_n_conf", 0
        ):
            self.diagnostics["cold_start_max_n_conf"] = cand.n_conf

    def _bump_n_conf_weighted(
        self, cand: CandidateSubgraph, t: int, weight: float
    ) -> None:
        """[P0-3] Accumulate fractional confirmation credit, at most once
        per step. The step guard preserves §4.17's "count distinct t′"
        semantics. If _bump_n_conf was already called this step (e.g.
        re-seeding), this returns early — a candidate both re-seeded and
        partially re-observed in one step counts once.

        [ADAPTATION v4] Renamed from the v3-era `_accumulate_membership`;
        the cold-start observability tail is retained (Session-5
        instrumentation, observability-only — no behavior change)."""
        if cand.last_n_conf_step >= t:
            return
        cand.last_n_conf_step = t
        cand.member_accum += weight
        while cand.member_accum >= 1.0:
            cand.n_conf += 1
            cand.member_accum -= 1.0
        # [P0-3] cold-start observability (seeding window).
        if t <= self.cfg.w_seeding_max and cand.n_conf > self.diagnostics.get(
            "cold_start_max_n_conf", 0
        ):
            self.diagnostics["cold_start_max_n_conf"] = cand.n_conf

    def _check_frontier_reobservation(self, t: int, s_t_bytes: bytes) -> None:
        """[P0-3] Membership re-observation with fractional accumulation.

        Replaces the strict "ALL members AND all edges present" rule: on
        sparse environments the perfect conjunction made N_conf >= 3
        statistically unreachable for minimal causal pairs (futurework2
        Item 4). A candidate is now partially re-observed when at least
        theta_min_present_ratio of its members are present this step AND
        every edge with BOTH endpoints present is active per §1.12. The
        fractional credit accumulates in member_accum; each crossing of
        1.0 increments N_conf once — preserving §4.17's integer semantics
        and the "candidate must recur" principle while removing the
        perfect-conjunction requirement.
        """
        current_node_ids: Set[int] = set(self._step_detection_map.keys())
        if not current_node_ids:
            return
        floor = self.cfg.theta_min_present_ratio

        for cand in self.frontier:
            if cand.last_n_conf_step >= t or not cand.members:
                continue

            # Zero-allocation presence count.
            present_count = 0
            for m in cand.members:
                if m in current_node_ids:
                    present_count += 1
            if present_count == 0:
                continue
            present_ratio = present_count / len(cand.members)
            if present_ratio < floor:
                continue

            edges_ok = True
            for (src, dst, et) in cand.edges:
                if src not in current_node_ids or dst not in current_node_ids:
                    continue
                # [ADAPTATION] this tree exposes the edge store directly
                # (ConceptGraph has no get_edge accessor); edges.get on the
                # (parent, child, EdgeType) key is the canonical lookup.
                edge = self.cg.edges.get((src, dst, et))
                if (
                    edge is None
                    or edge.weight < P.THETA_EDGE_ACTIVE
                    or edge.inactive_time >= P.TAU_EDGE_INACTIVE
                ):
                    edges_ok = False
                    break
            if not edges_ok:
                self.diagnostics["membership_reobservation_edge_rejected"] += 1
                continue

            if s_t_bytes not in cand.contexts:
                cand.contexts = cand.contexts | {s_t_bytes}
            cand.T_last = t
            self._bump_n_conf_weighted(cand, t, present_ratio)
            cand.V_step = -1
            cand.f_exp_step = -1
            self.diagnostics["membership_reobservations"] += 1
            if present_ratio < 1.0:
                self.diagnostics["membership_reobservation_partial"] += 1
    def _update_state_machine(self, t: int) -> None:
        if self.mode == "SEEDING":
            if self._seeding_start is None:
                self._seeding_start = t
            forced = (t - self._seeding_start) >= self.cfg.w_seeding_max
            # §4.10.3: SEEDING → COMMITTED on the FIRST CRYSTALLIZATION
            # forming at least one basin. Candidate-only basins (no
            # crystallized members) do not trigger the transition — the
            # Spine is a fertility construct and requires realized
            # structure. (Basins may exist from uncrystallized candidates
            # alone; gating on crystallized_snapshot keeps the transition
            # faithful to "the first crystallization triggers".)
            if (
                self.crystallized_snapshot
                and self.basins
                and self.spine_basin_id is not None
            ):
                self._enter_committed(t)
                return
            # Forced transition after W_seeding_max steps REGARDLESS (§4.10.3
            # — no frontier/basin precondition). Without basins the Builder
            # runs COMMITTED with spine None, i.e. the §4.11 global
            # exploration fallback, until structure crystallizes.
            if forced:
                self.mode = "COMMITTED"
                self._mark_spine_progress(t)
                self.spine_basin_id = self._pick_highest_fertility_basin()
                if (
                    self.spine_basin_id is not None
                    and self.spine_basin_id in self.basins
                ):
                    self.basins[
                        self.spine_basin_id
                    ].reference_V = self._current_max_spine_V(t)
                    self._sample_reference_v_history()
                self.last_spine_recompute_step = t
            return

        if self.mode == "COMMITTED":
            # [SPEC COMPLIANCE §4.10] stall_counter is incremented here, not
            # inside the predicate, ensuring single increment per step.
            if self._check_stall_condition(t):
                self.stall_counter += 1
            else:
                self.stall_counter = 0
            # [Batch D / P1-1] OR-of-bounds (HIGH-03): stalled on step
            # count OR on size-normalized graph turnover without Spine
            # progress (adaptation: normalized — see _graph_drift).
            if (
                self.stall_counter >= self.cfg.w_stall
                or (
                    self._graph_drift(self._spine_last_progress_rev)
                    >= self.cfg.w_stall_revision
                )
            ):
                self.mode = "STALLED"
                self.stall_counter = 0
            return

        if self.mode == "STALLED":
            # Priority 1: epistemic excursion (§4.10.3). Entries are
            # consumed one per EPISTEMIC_EXCURSION entry (§4.10.5). A
            # signal whose scope is empty cannot seed exploration (§4.11
            # scope identification needs members); it is consumed, counted
            # observably, and the next signal is considered in the same
            # step. Surprise magnitude is authoritative from the signal
            # (prediction.py owns the §8.6.1 formulas — never re-derived
            # here; the former duplicated fallback formulas had already
            # diverged from the spec once).
            while self.surprise_queue:
                sig = self.surprise_queue.popleft()
                scope = getattr(sig, "scope", None)
                scope_nodes = frozenset(getattr(scope, "nodes", ()) or ())
                if not scope_nodes:
                    self.diagnostics["epistemic_scope_empty_dropped"] += 1
                    continue
                self.current_surprise_value = max(
                    0.0, min(1.0, float(getattr(sig, "surprise", 0.0)))
                )
                self.current_scope_members = scope_nodes
                self.current_scope_expanded = set()
                self.t_epistemic_start = t
                # [Batch D / P1-1] entry checkpoint.
                self._epistemic_last_progress_t = t
                self._epistemic_last_progress_rev = (
            self.cg.graph_revision()
            if hasattr(self.cg, "graph_revision")
            else 0
        )
                self.mode = "EPISTEMIC_EXCURSION"
                return
            # Priority 2: curiosity-triggered excursion.
            curious = self._pick_curious_basin(exclude=self.spine_basin_id)
            if curious is not None:
                self.excursion_basin_id = curious
                self.t_excursion_start = t
                # [Batch D / P1-1] entry checkpoint.
                self._excursion_last_progress_t = t
                self._excursion_last_progress_rev = (
            self.cg.graph_revision()
            if hasattr(self.cg, "graph_revision")
            else 0
        )
                self._excursion_v_max_seen = -1.0
                self._excursion_entry_concepts = set(self.cg.concepts.keys())
                self.mode = "EXCURSION"
                return
            # Priority 3: return to Spine if it has unexplored candidates.
            if self.spine_basin_id is not None:
                spine_frontier = [
                    c for c in self.frontier if c.basin_id == self.spine_basin_id
                ]
                if spine_frontier:
                    self._enter_committed(t)
                    return
            # Priority 4: SPINE_REASSIGN — recompute the Spine immediately
            # (§4.10.3). §4.10.1: reference_V RESETS on re-entering
            # COMMITTED from SPINE_REASSIGN (unlike priority 3, which
            # preserves it through STALLED without excursion).
            self._reassign_spine(t)
            self.last_spine_recompute_step = t
            if self.spine_basin_id is not None:
                self._enter_committed(t, reset_reference=True)
            else:
                # No basins: §4.10.3 global exploration mode.
                self.mode = "COMMITTED"
                self._mark_spine_progress(t)
            return

        if self.mode == "EXCURSION":
            elapsed = t - self.t_excursion_start
            timeout = elapsed >= self.cfg.tau_excursion_max
            # [Batch D / P1-1] OR-of-bounds: exit on steps OR on graph
            # churn without a new excursion-basin max-V (MED-05).
            _grev = (
            self.cg.graph_revision()
            if hasattr(self.cg, "graph_revision")
            else 0
        )
            _mx = self._excursion_basin_max_v(t)
            if _mx > self._excursion_v_max_seen:
                self._excursion_v_max_seen = _mx
                self._excursion_last_progress_t = t
                self._excursion_last_progress_rev = _grev
            if (
                self._graph_drift(self._excursion_last_progress_rev)
                >= self.cfg.tau_excursion_revision
            ):
                timeout = True
            # §4.10.3 sterility: no candidate achieves V ≥ θ_sterile for
            # W_stall CONSECUTIVE steps — tracked per step, not one-shot.
            if _mx < self.cfg.theta_sterile:
                self._excursion_sterile_counter += 1
            else:
                self._excursion_sterile_counter = 0
            sterile = self._excursion_sterile_counter >= self.cfg.w_stall
            crystallized = self._excursion_crystallized(t)
            if crystallized:
                self._excursion_last_progress_t = t
                self._excursion_last_progress_rev = _grev
            if timeout or sterile or crystallized:
                self.excursion_basin_id = None
                self._excursion_sterile_counter = 0
                # Excursion resolution point: process any Spine-basin split
                # that was deferred during the excursion (§4.10.3).
                self._spine_reassign_deferred = False
                self._reassign_spine(t)
                self.last_spine_recompute_step = t
                self._enter_committed(t)
            return

        if self.mode == "EPISTEMIC_EXCURSION":
            elapsed = t - self.t_epistemic_start
            timeout = elapsed >= self.cfg.tau_epistemic_max
            # [Batch D / P1-1] OR-of-bounds on scope-expansion churn
            # (size-normalized turnover — see _graph_drift).
            if (
                self._graph_drift(self._epistemic_last_progress_rev)
                >= self.cfg.tau_epistemic_revision
            ):
                timeout = True
            remaining = [
                c
                for c in self.frontier
                if (c.members & self.current_scope_members)
                and c.canonical_hash not in self.current_scope_expanded
            ]
            done = not remaining
            if timeout or done:
                self.current_scope_members = frozenset()
                self.current_scope_expanded = set()
                self.current_surprise_value = 0.0
                # Deferred Spine-split reassignment (if any) resolves via
                # the STALLED transition priorities on the next step.
                self._spine_reassign_deferred = False
                self.mode = "STALLED"
            return

    def _graph_size(self) -> int:
        """[Batch D / P1-1 adaptation] Live graph size (edges + live
        pattern nodes), cached per step. Normalizer for revision drift."""
        if getattr(self, "_graph_size_step", -2) != self._current_step:
            self._graph_size_cache = len(self.cg.edges) + len(
                self.cg.get_live_pattern_node_ids()
            )
            self._graph_size_step = self._current_step
        return max(1, self._graph_size_cache)

    def _graph_drift(self, since_rev: int) -> float:
        """[Batch D / P1-1 adaptation] Graph-size-normalized revision drift.

        The v5 draft's raw revision deltas assumed ~1 revision/step.
        Measured on the post-Batch-E tree, rooms churns ~27 revisions per
        step (2700 revisions / 100 steps), so raw bounds fired ~30x faster
        than their step counterparts and destroyed the §4.5 commitment
        hysteresis (canary evidence: rooms solve 425 -> unsolved at 2000,
        schemas_bound 23 -> 16). P1-1's parent directive mandates the
        normalizer ("divided by graph size"): revision bounds are expressed
        in TURNOVER units — accumulated graph changes divided by the graph's
        own size — which is environment-scale-free by construction. On
        churn-heavy graphs the step bound of each OR dominates (pins hold);
        on slow graphs the turnover bound fires proportionally to actual
        change, which is the endogenous-cadence intent."""
        delta = self.cg.graph_revision() - since_rev
        return delta / self._graph_size()

    def _mark_spine_progress(self, t: int) -> None:
        """[Batch D / P1-1 MED-05] Spine progress checkpoint: refreshed on
        spine-basin crystallization, reference_V improvement, and every
        (re)entry into COMMITTED."""
        self._spine_last_progress_t = t
        self._spine_last_progress_rev = (
            self.cg.graph_revision()
            if hasattr(self.cg, "graph_revision")
            else 0
        )

    def _enter_committed(self, t: int, reset_reference: bool = False) -> None:
        """§4.10.1 reference_V lifecycle: initialized on SEEDING→COMMITTED,
        RESET on re-entry from EXCURSION or SPINE_REASSIGN
        (reset_reference=True), PRESERVED through COMMITTED→STALLED→
        COMMITTED without excursion."""
        prev = self.mode
        self.mode = "COMMITTED"
        self.stall_counter = 0
        self._mark_spine_progress(t)
        if self.spine_basin_id is None or self.spine_basin_id not in self.basins:
            return
        if reset_reference or prev in ("EXCURSION", "SEEDING"):
            self.basins[self.spine_basin_id].reference_V = self._current_max_spine_V(t)
            self._sample_reference_v_history()

    def _check_stall_condition(self, t: int) -> bool:
        """Pure predicate: both stall conditions (§4.10.2) must hold this step.

        No side effect on stall_counter; the state machine handles that.
        """
        if self.spine_basin_id is None or self.spine_basin_id not in self.basins:
            return True
        spine = self.basins[self.spine_basin_id]
        if spine.reference_V < self.cfg.eps_ref:
            return True

        max_v = 0.0
        for cand in self.frontier:
            if cand.basin_id != self.spine_basin_id:
                continue
            # §4.10.2 epistemic-excursion exclusion: candidates produced
            # during EPISTEMIC_EXCURSION never count for Spine stall
            # detection — a surprise-driven high-V candidate must not mask
            # a stalled Spine.
            if cand.epistemic_origin:
                continue
            v = self._compute_V(cand, t)
            if v > max_v:
                max_v = v
        stall_threshold_v = self.cfg.stall_threshold * spine.reference_V
        condition_no_v = max_v < stall_threshold_v

        # §4.10.2 condition 1 is PER-STEP: "no new subgraph in the Spine has
        # crossed the crystallization predicate [this step]". Both conditions
        # must hold on each of the W_stall steps; a crystallization at step
        # t0 resets the counter at t0 only, so STALLED is reached at t0 +
        # W_stall. (The former windowed test `last_cryst < t - W_stall`
        # delayed every stall by an extra W_stall steps.)
        condition_no_cryst = self._last_spine_crystallization_step != t

        return condition_no_v and condition_no_cryst

    def _current_max_spine_V(self, t: int) -> float:
        if self.spine_basin_id is None or self.spine_basin_id not in self.basins:
            return 0.0
        m = 0.0
        for cand in self.frontier:
            if cand.basin_id != self.spine_basin_id:
                continue
            v = self._compute_V(cand, t)
            if v > m:
                m = v
        return m

    def _update_reference_V(self, t: int) -> None:
        if self.spine_basin_id is None or self.spine_basin_id not in self.basins:
            return
        m = self._current_max_spine_V(t)
        spine = self.basins[self.spine_basin_id]
        _prev_ref = spine.reference_V
        if spine.reference_V == 0.0:
            spine.reference_V = m
            self._sample_reference_v_history()
        else:
            spine.reference_V = (
                1.0 - self.cfg.kappa_ref
            ) * spine.reference_V + self.cfg.kappa_ref * m
            self._sample_reference_v_history()
        # [Batch D / P1-1 MED-05] reference_V improvement = Spine progress.
        if m > _prev_ref:
            self._mark_spine_progress(t)

    # -------------------------------------------------------------------------
    # §4.9 Basins and Spine — incremental
    # -------------------------------------------------------------------------

    def _mark_dirty_node(self, nid: int) -> None:
        self._dirty_basin_nodes.add(nid)

    def _mark_dirty_members(self, members: FrozenSet[int]) -> None:
        self._dirty_basin_nodes.update(members)

    def _maybe_recompute_basins(self, t: int) -> None:
        # [Batch D / P1-1] OR-of-bounds cadence. ADAPTATION NOTE: HIGH-03
        # said "pure-revision" for basins, but with size-normalized drift
        # a pure-revision cadence has no calibrated floor on current
        # benchmarks (500 turnovers >> any 2000-step run), which would
        # silently disable periodic spine recomputation. The step bound
        # (k_main) is retained as the OR's floor — strictly no-slower than
        # the pre-P1-1 tree — and the turnover bound accelerates
        # recomputation on graphs that churn faster than one turnover per
        # k_main steps.
        _grev = (
            self.cg.graph_revision()
            if hasattr(self.cg, "graph_revision")
            else 0
        )
        cadence_due = (
            t - self.last_basin_recompute_step >= self.cfg.k_main
            or self._graph_drift(self._last_basin_rev)
            >= self.cfg.k_main_revision
        )
        # O(1) denominator: the subgraph index keys are exactly the nodes
        # occupied by concepts and frontier candidates (maintained at every
        # add/remove site; rebuilt on full recompute).
        total_occupied = len(self._subgraph_index)

        if cadence_due:
            self._full_recompute_basins(t, spine_cadence=True)
            self.last_basin_recompute_step = t
            self._last_basin_rev = _grev
            self.last_spine_recompute_step = t
            self._dirty_basin_nodes.clear()
            self.diagnostics["full_basin_recomputes"] += 1
            return

        if not self._dirty_basin_nodes:
            return

        dirty_ratio = len(self._dirty_basin_nodes) / max(1, total_occupied)
        if dirty_ratio > self.cfg.theta_basin_full_recompute:
            # §4.9 fallback full recompute — still mid-cycle for the Spine:
            # continuity rules apply (no free reassignment).
            self._full_recompute_basins(t, spine_cadence=False)
            self.last_basin_recompute_step = t
            self._last_basin_rev = _grev
            self.diagnostics["full_basin_recomputes"] += 1
        else:
            self._incremental_basin_update(self._dirty_basin_nodes)
            self.diagnostics["incremental_basin_updates"] += 1
        self._dirty_basin_nodes.clear()

    def _incremental_basin_update(self, dirty_nodes: Set[int]) -> None:
        """[SPEC COMPLIANCE §4.9] Local Union-Find over dirty nodes and their
        1-hop neighbors. Reassigns only affected subgraphs to basins.

        1-hop closure is sufficient under the basin invariant (basins are
        maximally connected sets): any subgraph further away was already in
        the same basin as the 1-hop neighbor before the change, so merging
        the basins found among local nodes transitively absorbs it.

        Subgraphs whose member set does not intersect the dirty neighborhood
        retain their current basin assignment. The Spine is NEVER reassigned
        here (§4.9: "Incremental basin updates do not reassign the Spine
        mid-cycle") — only preserved, ID-remapped through merges, or
        reassigned on split/vanish (immediately, unless deferred by an
        active excursion per §4.10.3).
        """
        old_spine_id = self.spine_basin_id
        old_spine_members: FrozenSet[bytes] = (
            frozenset(self.basins[old_spine_id].members)
            if old_spine_id is not None and old_spine_id in self.basins
            else frozenset()
        )
        # 1. Collect local neighborhood of dirty nodes (1-hop). Sorted
        # iteration for deterministic downstream set-insertion order.
        local_nodes: Set[int] = set(dirty_nodes)
        for nid in sorted(dirty_nodes):
            for edge in self.cg.get_outgoing_edges(nid):
                local_nodes.add(edge.child)
            for edge in self.cg.get_incoming_edges(nid):
                local_nodes.add(edge.parent)

        # 2. Union-Find over local nodes only.
        parent: Dict[int, int] = {nid: nid for nid in local_nodes}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for nid in local_nodes:
            for edge in self.cg.get_outgoing_edges(nid):
                if edge.child in local_nodes:
                    union(nid, edge.child)

        # 3. For each local component, find which subgraphs belong to it,
        #    and what basins those subgraphs currently belong to.
        comp_to_hashes: Dict[int, Set[bytes]] = defaultdict(set)
        for nid in sorted(local_nodes):
            root = find(nid)
            for h in self._subgraph_index.get(nid, ()):
                comp_to_hashes[root].add(h)

        # 4. Merge/assign basins for each affected component.
        for _root, hashes in comp_to_hashes.items():
            if not hashes:
                continue
            existing_basin_ids = {self._basin_of_hash(h) for h in hashes}
            existing_basin_ids.discard(-1)
            if len(existing_basin_ids) == 1:
                basin_id = next(iter(existing_basin_ids))
            elif len(existing_basin_ids) > 1:
                # Merge all into one.
                basin_id = self._merge_basins(existing_basin_ids)
            else:
                basin_id = self._allocate_basin_id()
                self.basins[basin_id] = Basin(id=basin_id)

            for h in hashes:
                self.basins[basin_id].members.add(h)
                self._assign_basin(h, basin_id)

        # 5. Recompute cumulative fertility for affected basins. §4.9:
        # cumulative fertility is strictly a reflection of the CURRENT
        # member set (Σ EF_combined × layer over crystallized members),
        # never a historical artifact.
        for bid in {
            self._basin_of_hash(h) for comp in comp_to_hashes.values() for h in comp
        }:
            if bid == -1 or bid not in self.basins:
                continue
            self._recompute_basin_fertility(bid)

        # 6. Spine continuity (§4.9/§4.10.3): preserve unless split/vanished.
        self._apply_spine_continuity(old_spine_id, old_spine_members)

    # -------------------------------------------------------------------------
    # §4.9 immediate basin splitting (Seam 3 / D1)
    # -------------------------------------------------------------------------

    def _members_of_hash(self, h: bytes) -> FrozenSet[int]:
        """Member pattern nodes of a basin-resident subgraph hash —
        crystallized concept first, then frontier candidate."""
        concept = self.cg.lookup_by_canonical_hash(h)
        if concept is not None and concept.members:
            return frozenset(concept.members)
        cand = self._frontier_by_hash.get(h)
        if cand is not None:
            return frozenset(cand.members)
        cached = self._subgraph_members_cache.get(h)
        if cached:
            return frozenset(cached)
        return frozenset()

    def _check_basin_splits(self, removed_nodes: Set[int]) -> None:
        """[Seam 3/D1] Batched split check for basins touched by edge
        removals / node prunes (K5: once per Builder step over the union
        of the drained events, not per event).

        The connectivity predicate is edge PRESENCE — the same predicate
        _full_recompute_basins unions over — so splits here and merges
        there can never fight: an edge whose weight decayed but which
        still exists does NOT split a basin (it will be re-merged at the
        next recompute otherwise); an edge REMOVED by step_28's
        inactivity prune (or by a pattern-node prune) splits immediately
        instead of after up to K_main = 500 stale steps.
        """
        affected: Set[int] = set()
        for nid in sorted(removed_nodes):
            for h in self._subgraph_index.get(nid, ()):
                bid = self._basin_of_hash(h)
                if bid != -1:
                    affected.add(bid)
        if not affected:
            return
        self.diagnostics["basin_split_checks"] += len(affected)
        old_spine_id = self.spine_basin_id
        old_spine_members: FrozenSet[bytes] = (
            frozenset(self.basins[old_spine_id].members)
            if old_spine_id is not None and old_spine_id in self.basins
            else frozenset()
        )
        split_happened = False
        for bid in sorted(affected):
            if self._split_basin_if_disconnected(bid):
                split_happened = True
        if split_happened:
            self._apply_spine_continuity(old_spine_id, old_spine_members)

    def _split_basin_if_disconnected(self, bid: int) -> bool:
        """Recompute the connected components of one basin over its member
        subgraphs' pattern nodes (undirected traversal of PRESENT edges,
        matching _full_recompute_basins). If the basin fell apart, the
        component holding the most member hashes keeps the basin id
        (ties → lowest component index, deterministic); every other
        component's hashes move to freshly allocated basins. Returns True
        iff a split was performed."""
        basin = self.basins.get(bid)
        if basin is None or not basin.members:
            return False
        hash_members: Dict[bytes, FrozenSet[int]] = {}
        nodes: Set[int] = set()
        for h in sorted(basin.members):
            members = self._members_of_hash(h)
            if not members:
                continue
            hash_members[h] = members
            nodes.update(members)
        if len(hash_members) < 2 or not nodes:
            return False
        # Adjacency over present edges restricted to basin member nodes.
        adj: Dict[int, Set[int]] = defaultdict(set)
        for nid in sorted(nodes):
            for edge in self.cg.get_outgoing_edges(nid):
                if edge.child in nodes:
                    adj[nid].add(edge.child)
                    adj[edge.child].add(nid)
            for edge in self.cg.get_incoming_edges(nid):
                if edge.parent in nodes:
                    adj[nid].add(edge.parent)
                    adj[edge.parent].add(nid)
        # Components (deterministic iteration).
        seen: Set[int] = set()
        comps: List[Set[int]] = []
        for start in sorted(nodes):
            if start in seen:
                continue
            comp: Set[int] = set()
            stack = [start]
            seen.add(start)
            while stack:
                x = stack.pop()
                comp.add(x)
                for y in sorted(adj.get(x, ())):
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            comps.append(comp)
        if len(comps) <= 1:
            return False
        comp_of: Dict[int, int] = {}
        for ci, comp in enumerate(comps):
            for nid in comp:
                comp_of[nid] = ci
        hash_comp: Dict[bytes, int] = {}
        for h, members in hash_members.items():
            hash_comp[h] = min(comp_of[nid] for nid in members if nid in comp_of)
        counts: Dict[int, int] = defaultdict(int)
        for ci in hash_comp.values():
            counts[ci] += 1
        keep = min(counts.keys(), key=lambda ci: (-counts[ci], ci))
        moved = 0
        for ci in sorted(k for k in counts.keys() if k != keep):
            new_id = self._allocate_basin_id()
            self.basins[new_id] = Basin(id=new_id)
            for h in sorted(h for h, c in hash_comp.items() if c == ci):
                self.basins[new_id].members.add(h)
                self._assign_basin(h, new_id)
                moved += 1
            self._recompute_basin_fertility(new_id)
        if moved:
            self._recompute_basin_fertility(bid)
            self.diagnostics["basin_splits"] += 1
            return True
        return False

    def _recompute_basin_fertility(self, bid: int) -> None:
        b = self.basins[bid]
        total = 0.0
        for h in b.members:
            concept = self.cg.lookup_by_canonical_hash(h)
            if concept is None:
                continue
            total += getattr(concept, "EF_combined", 0.0) * max(1, concept.layer)
        b.cumulative_fertility = total

    def _apply_spine_continuity(
        self, old_spine_id: Optional[int], old_spine_members: FrozenSet[bytes]
    ) -> None:
        """Keep the Spine stable across mid-cycle basin updates. Reassign
        only when the Spine basin vanished or SPLIT (its former members now
        span multiple basins). Splits during EXCURSION/EPISTEMIC_EXCURSION
        defer reassignment to the excursion's resolution (§4.10.3)."""
        if old_spine_id is None:
            # No Spine yet: first identification, not reassignment.
            # §4.10.3: "the first crystallization triggers basin
            # recomputation. If at least one basin is formed, the Spine is
            # identified." Without this branch the SEEDING gate (which
            # requires spine_basin_id) could never open and the Builder
            # would sit in SEEDING until the W_seeding_max force.
            if self.basins:
                self._reassign_spine_quietly()
                self.last_spine_recompute_step = self._current_step
            return
        if old_spine_id in self.basins and not old_spine_members:
            return  # Spine ID alive (and had no crystallized members to split)
        if not old_spine_members:
            self._trigger_spine_reassign()
            return
        current_ids = {
            self._hash_to_basin.get(h, -1) for h in old_spine_members
        }
        current_ids.discard(-1)
        if len(current_ids) == 1:
            # Merged or intact: follow the (possibly remapped) basin ID.
            self.spine_basin_id = next(iter(current_ids))
            return
        # Vanished (0) or split (>1) → SPINE_REASSIGN fires (§4.9).
        self._trigger_spine_reassign()

    def _trigger_spine_reassign(self) -> None:
        if self.mode in ("EXCURSION", "EPISTEMIC_EXCURSION"):
            # §4.10.3: deferred to the excursion's resolution.
            self._spine_reassign_deferred = True
            self.diagnostics["spine_splits_deferred"] += 1
            return
        self._reassign_spine_quietly()
        self.last_spine_recompute_step = self._current_step

    def _basin_of_hash(self, h: bytes) -> int:
        # O(1) via the reverse index (the former all-basins member scan was
        # O(basins × members) per lookup, executed per trace and per
        # incremental component).
        return self._hash_to_basin.get(h, -1)

    def _assign_basin(self, h: bytes, basin_id: int) -> None:
        prev = self._hash_to_basin.get(h)
        if prev is not None and prev != basin_id and prev in self.basins:
            self.basins[prev].members.discard(h)
        self._hash_to_basin[h] = basin_id

    def _unassign_hash(self, h: bytes) -> None:
        prev = self._hash_to_basin.pop(h, None)
        if prev is not None and prev in self.basins:
            self.basins[prev].members.discard(h)

    def _merge_basins(self, basin_ids: Set[int]) -> int:
        """§4.9 basin merge: child inherits the union of member sets and
        cumulative fertility; no full recomputation; resorting deferred to
        the next K_main cadence."""
        sorted_ids = sorted(basin_ids)
        target = sorted_ids[0]
        for bid in sorted_ids[1:]:
            src = self.basins.pop(bid, None)
            if src is None:
                continue
            self.basins[target].members.update(src.members)
            self.basins[target].cumulative_fertility += src.cumulative_fertility
            for h in src.members:
                if self._hash_to_basin.get(h) == bid:
                    self._hash_to_basin[h] = target
        # ID remap is continuity, not reassignment: if the Spine was among
        # the merged basins it follows the target ID (§4.9 stability).
        if self.spine_basin_id in basin_ids and self.spine_basin_id != target:
            self.spine_basin_id = target
        return target

    def _full_recompute_basins(self, t: int, spine_cadence: bool = True) -> None:
        """Full Union-Find recompute (§4.9 fallback / K_main cadence).

        spine_cadence=True (K_main boundary): the Spine is recomputed
        freely. spine_cadence=False (dirty-overflow fallback): mid-cycle —
        Spine continuity rules apply (preserve unless split/vanished)."""
        old_spine_id = self.spine_basin_id
        old_spine_members: FrozenSet[bytes] = (
            frozenset(self.basins[old_spine_id].members)
            if old_spine_id is not None and old_spine_id in self.basins
            else frozenset()
        )
        occ: Set[int] = set()
        concept_membership: Dict[int, List[bytes]] = defaultdict(list)
        for _cid, concept in self.cg.concepts.items():
            if concept.status not in (
                ConceptStatus.CRYSTALLIZED,
                ConceptStatus.STABLE,
            ):
                continue
            ch = concept.canonical_hash
            if not ch:
                continue
            for nid in concept.members:
                occ.add(nid)
                concept_membership[nid].append(ch)

        candidate_membership: Dict[int, List[bytes]] = defaultdict(list)
        for cand in self.frontier:
            for nid in cand.members:
                occ.add(nid)
                candidate_membership[nid].append(cand.canonical_hash)

        occ_set = occ
        adj: Dict[int, Set[int]] = defaultdict(set)
        for (src, dst, _et), _edge in self.cg.edges.items():
            if src in occ_set and dst in occ_set:
                adj[src].add(dst)
                adj[dst].add(src)

        parent: Dict[int, int] = {nid: nid for nid in occ}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for nid in sorted(adj.keys()):
            for nb in sorted(adj[nid]):
                union(nid, nb)

        comp_to_hashes: Dict[int, Set[bytes]] = defaultdict(set)
        for nid in sorted(occ):
            r = find(nid)
            for ch in concept_membership.get(nid, []):
                comp_to_hashes[r].add(ch)
            for ch in candidate_membership.get(nid, []):
                comp_to_hashes[r].add(ch)

        new_basins: Dict[int, Basin] = {}
        old_by_hash: Dict[bytes, int] = {}
        for bid, b in self.basins.items():
            for h in b.members:
                old_by_hash[h] = bid

        # Split safety: when one old basin's members land in MULTIPLE new
        # components, only the first component may reuse the old ID — the
        # former implementation handed the same ID to every fragment and
        # silently overwrote earlier ones (losing members and fertility).
        used_bids: Set[int] = set()
        for _r, hashes in sorted(comp_to_hashes.items(), key=lambda kv: sorted(kv[1])):
            old_ids = {old_by_hash[h] for h in hashes if h in old_by_hash}
            bid: Optional[int] = None
            if len(old_ids) == 1:
                candidate_bid = next(iter(old_ids))
                if candidate_bid not in used_bids:
                    bid = candidate_bid
            if bid is None:
                bid = self._allocate_basin_id()
                while bid in used_bids:
                    bid = self._allocate_basin_id()
            used_bids.add(bid)
            cumulative = 0.0
            for h in hashes:
                concept = self.cg.lookup_by_canonical_hash(h)
                if concept is None:
                    continue
                cumulative += getattr(concept, "EF_combined", 0.0) * max(
                    1, concept.layer
                )
            new_basins[bid] = Basin(
                id=bid,
                members=set(hashes),
                cumulative_fertility=cumulative,
            )

        # Preserve reference_V for retained basin IDs.
        for bid in sorted(new_basins.keys()):
            nb = new_basins[bid]
            if bid in self.basins:
                nb.reference_V = self.basins[bid].reference_V

        # Rebuild indexes.
        self._subgraph_index = defaultdict(set)
        self._subgraph_members_cache = {}
        self._hash_to_basin = {}

        self.basins = new_basins

        hash_to_new_bid: Dict[bytes, int] = {}
        for bid, b in new_basins.items():
            for h in b.members:
                hash_to_new_bid[h] = bid
        self._hash_to_basin = hash_to_new_bid

        for cand in self.frontier:
            cand.basin_id = hash_to_new_bid.get(cand.canonical_hash, -1)

        # Rebuild the subgraph index from scratch.
        for _cid, concept in self.cg.concepts.items():
            if concept.status not in (
                ConceptStatus.CRYSTALLIZED,
                ConceptStatus.STABLE,
            ):
                continue
            ch = concept.canonical_hash
            if not ch:
                continue
            self._subgraph_members_cache[ch] = concept.members
            for nid in concept.members:
                self._subgraph_index[nid].add(ch)
        for cand in self.frontier:
            self._subgraph_members_cache[cand.canonical_hash] = cand.members
            for nid in cand.members:
                self._subgraph_index[nid].add(cand.canonical_hash)

        if spine_cadence:
            # K_main cadence: free Spine recomputation (§4.9).
            self._reassign_spine_quietly()
        else:
            self._apply_spine_continuity(old_spine_id, old_spine_members)

    def _allocate_basin_id(self) -> int:
        bid = self.next_basin_id
        self.next_basin_id += 1
        return bid

    def _reassign_spine_quietly(self) -> None:
        """Spine = argmax over basins of Σ EF_combined × layer (§4.9).
        Tie-break: lower basin ID — the same deterministic rule §4.10.3
        specifies for excursion selection (the former min-member-hash
        tie-break made the two Spine pickers disagree)."""
        if not self.basins:
            self.spine_basin_id = None
            return
        best_id: Optional[int] = None
        best_fert = -1.0
        for bid in sorted(self.basins.keys()):
            fert = self.basins[bid].cumulative_fertility
            if fert > best_fert:
                best_id = bid
                best_fert = fert
        self.spine_basin_id = best_id

    def _reassign_spine(self, t: int) -> None:
        if not self.basins:
            self._full_recompute_basins(t)
        self._reassign_spine_quietly()

    def _pick_highest_fertility_basin(self) -> Optional[int]:
        if not self.basins:
            return None
        return max(
            self.basins.items(),
            key=lambda kv: (kv[1].cumulative_fertility, -kv[0]),
        )[0]

    def _basin_curiosity_map(self) -> Dict[int, float]:
        """§4.8 Curiosity(B) for every basin, computed in ONE frontier pass
        (the former per-basin full-frontier scan was O(basins × frontier)).

            F_expected(B) = max over c ∈ B of F_expected(c)   (ceiling — max,
                                                        not mean, per §4.8)
            ConfirmationRatio(B) = min(1, ΣN_conf / (ΣN_opp + ε)), 0 if ΣN_opp=0
            Curiosity(B) = F_expected(B) × (1 − ConfirmationRatio(B))
        """
        f_max: Dict[int, float] = {}
        conf_sum: Dict[int, int] = defaultdict(int)
        opp_sum: Dict[int, int] = defaultdict(int)
        for cand in self.frontier:
            bid = cand.basin_id
            v_expected = self._compute_F_expected(cand)
            if v_expected > f_max.get(bid, 0.0):
                f_max[bid] = v_expected
            conf_sum[bid] += cand.n_conf
            opp_sum[bid] += len(cand.contexts)
        result: Dict[int, float] = {}
        for bid in self.basins:
            fm = f_max.get(bid, 0.0)
            opp = opp_sum.get(bid, 0)
            ratio = 0.0 if opp == 0 else min(1.0, conf_sum.get(bid, 0) / (opp + _EPS))
            result[bid] = fm * (1.0 - ratio)
        return result

    def _pick_curious_basin(self, exclude: Optional[int]) -> Optional[int]:
        """§4.10.3: highest Curiosity ≥ θ_curious among non-Spine basins;
        tie-break by lower basin ID (deterministic)."""
        if not self.basins:
            return None
        curiosity = self._basin_curiosity_map()
        best_id: Optional[int] = None
        best_cur = -1.0
        for bid in sorted(self.basins.keys()):
            if exclude is not None and bid == exclude:
                continue
            cur = curiosity.get(bid, 0.0)
            if cur < self.cfg.theta_curious:
                continue
            if cur > best_cur:
                best_id = bid
                best_cur = cur
        return best_id

    def _excursion_basin_max_v(self, t: int) -> float:
        """Max V over the excursion basin's frontier candidates (§4.10.3
        sterility input). An empty/unknown basin yields -1.0 so the sterile
        counter advances (nothing to explore is maximally sterile)."""
        if self.excursion_basin_id is None:
            return -1.0
        best = -1.0
        for cand in self.frontier:
            if cand.basin_id != self.excursion_basin_id:
                continue
            v = self._compute_V(cand, t)
            if v > best:
                best = v
        return best

    def _excursion_crystallized(self, t: int) -> bool:
        if self.excursion_basin_id is None:
            return False
        entry = getattr(self, "_excursion_entry_concepts", None)
        if entry is None:
            return False
        for cid, concept in self.cg.concepts.items():
            if cid in entry:
                continue
            if concept.status in (
                ConceptStatus.CRYSTALLIZED,
                ConceptStatus.STABLE,
            ):
                ch = concept.canonical_hash
                if ch and self.excursion_basin_id in self.basins:
                    if ch in self.basins[self.excursion_basin_id].members:
                        return True
        return False

    # -------------------------------------------------------------------------
    # §4.11 Selection & Expansion
    # -------------------------------------------------------------------------

    def _expand_step(self, t: int) -> None:
        if self.mode == "STALLED":
            return
        if self.mode == "SEEDING":
            self._expand_mode(t, priority_fn=lambda c: self._compute_V(c, t))
            return
        if self.mode == "COMMITTED":
            # §4.11 COMMITTED: expand the Spine's frontier — commitment is
            # the whole point of the algorithm. The former implementation
            # expanded the global frontier, abandoning the Spine whenever a
            # non-Spine candidate out-prioritized it (exactly the
            # oscillation §4.5 forbids).
            if self.spine_basin_id is not None:

                def _prio(c: CandidateSubgraph) -> float:
                    v = self._compute_V(c, t)
                    fresh = math.exp(
                        -max(0, t - c.t_created) / self.cfg.tau_fresh
                    )
                    return v + self.cfg.eta_fresh * fresh

                self._expand_mode(
                    t, priority_fn=_prio, basin_filter=self.spine_basin_id
                )
                return
            # No Spine: §4.11 global exploration fallback — top-K by
            # Curiosity(B) of the containing basin; no basins → by V(S).
            if self.basins:
                cur_by_basin = self._basin_curiosity_map()

                def _prio_cur(c: CandidateSubgraph) -> float:
                    return cur_by_basin.get(c.basin_id, 0.0)

                self._expand_mode(t, priority_fn=_prio_cur)
            else:
                self._expand_mode(
                    t, priority_fn=lambda c: self._compute_V(c, t)
                )
            return
        if self.mode == "EXCURSION":
            self._expand_mode(
                t,
                priority_fn=lambda c: self._compute_V(c, t),
                basin_filter=self.excursion_basin_id,
            )
            return
        if self.mode == "EPISTEMIC_EXCURSION":
            # [SPEC COMPLIANCE §4.11] priority(S) = V(S) + η_surprise·surprise
            surprise_bonus = self.cfg.eta_surprise * self.current_surprise_value

            def _prio(c: CandidateSubgraph) -> float:
                return self._compute_V(c, t) + surprise_bonus

            self._expand_mode(
                t,
                priority_fn=_prio,
                scope_filter=self.current_scope_members,
            )
            return

    def _expand_mode(
        self,
        t: int,
        priority_fn,
        basin_filter: Optional[int] = None,
        scope_filter: Optional[FrozenSet[int]] = None,
    ) -> None:
        k = max(1, self.cfg.builder_expansions_per_step)

        pool = list(self.frontier)
        if basin_filter is not None:
            pool = [c for c in pool if c.basin_id == basin_filter]
        if scope_filter is not None and scope_filter:
            pool = [c for c in pool if c.members & scope_filter]

        # Expansion eligibility — two regimes:
        #
        # POST-BOOTSTRAP (fertile set H ≠ ∅): §4.11.1 literal — the
        # transposition table prevents re-expansion of structurally
        # identical subgraphs; V carries the fertility signal that makes
        # §4.4's "select the neighbor maximizing V(S′)" meaningful.
        #
        # BOOTSTRAP (H = ∅, SANCTIONED EXTENSION): every V ≡ 0 (§4.7:
        # proximity ≡ 0 when H = ∅), so neighbor selection degenerates to
        # a canonical-hash lottery and one-shot expansion makes minimal
        # causal pairs effectively unformable (a singleton's single
        # expansion either draws the right neighbor or the structure is
        # lost to the table forever). The regime-correct signal is
        # RECURRENCE EVIDENCE, the only information that exists pre-
        # fertility: expand only candidates with n_conf ≥ 1, and re-
        # expand an already-expanded structure only when it has accrued
        # NEW confirmations since that expansion. Effects: (a) expansion
        # compute is allocated in evidence order (the γ·W term of the
        # spec's own V ranks the most-confirmed first); (b) unconfirmed
        # expansion chains — the V ≡ 0 "blob" failure mode — cannot form;
        # (c) re-seeded event nodes retry structure formation as the
        # edge set accumulates, which is how SEQUENCE
        # pairs get built. The gate self-disables the moment the first
        # concept makes H non-empty; no environment constants involved.
        #
        # [DESIGN CHOICE — P0-2 interaction] The cold-start rescue also
        # affects the bootstrap gate here: under rescue, H ≠ ∅ as soon as
        # any concept with EF > 0 crystallizes, so candidates with
        # n_conf = 0 can be expanded earlier than before. This is ACCEPTED
        # because V is computed against H, and a weak H gives a weak V
        # signal which does not force expansion. The pruning guard in
        # _prune_frontier (Phase 10) preserves meaningful-fertility
        # gating for pruning independently.
        # [GUARD] This counter increments at most once per step.
        # _expand_step calls _expand_mode exactly once per step.
        # If a future refactor calls _expand_mode in a loop, this
        # counter will over-count. Verify the call structure before
        # relying on this diagnostic.
        bootstrap = not self._fertile_concepts()
        if bootstrap:
            self.diagnostics["bootstrap_gate_active_steps"] += 1
        eligible: List[CandidateSubgraph] = []
        for c in pool:
            entry = self.transposition_table.get(c.canonical_hash)
            if entry is not None:
                # §4.11.1: the memo prevents RE-EXPANSION OF STRUCTURALLY
                # IDENTICAL SUBGRAPHS — its purpose is skipping REDUNDANT
                # work. A re-expansion is redundant exactly when it would
                # regenerate the same children and re-present them to an
                # unchanged frontier. Two non-redundant novelty signals
                # unlock exactly one retry each:
                #
                # (1) CONTEXT NOVELTY — the live expansion footprint (the
                #     graph edges the §4.4 operations read around the
                #     candidate's members) changed since the memoized
                #     expansion. New SEQUENCE edges can
                #     now unite into Rule-1 (§4.14) causal children that
                #     did not exist before.
                # (2) EVIDENCE NOVELTY WITH A PRODUCTIVE LINEAGE — the
                #     candidate accrued new confirmations (re-seed /
                #     cross-parent regeneration, §4.17) AND its last
                #     expansion generated at least one child. This is the
                #     regeneration-merge engine: re-expanding P rebuilds
                #     its best VIABLE child (selection-time filtering
                #     below); under converged live-induced identity the
                #     rebuilt child hash-matches the frontier resident
                #     and MERGES, incrementing its N_conf/span. Without
                #     the retry, a stable store gives each parent one
                #     one-shot expansion and multi-node structures are
                #     regenerated at most once (measured: 17 merges over
                #     293 causal both-edge hashes, max N_conf = 1 <
                #     θ_conf = 3 — §4.17 was structurally unreachable
                #     and EVENT_CAUSAL crystallization deadlocked; 201
                #     of 220 causal structures were expanded-away at
                #     median 12 steps of age with zero confirmations).
                #     The `children > 0` clause makes the retry
                #     self-limiting: an exhausted parent (every child
                #     retired by transposition/crystallization/
                #     tombstone) stops consuming the K-slot budget even
                #     if re-seeding keeps raising its n_conf — this is
                #     what prevents the monopolization pathology of the
                #     former unconditional n_conf unlock (measured then:
                #     the same parent expanded 275/300 steps while other
                #     candidates starved).
                _t_expanded, fp_expanded, nconf_expanded, kids_expanded = entry
                context_changed = (
                    fp_expanded != self._expansion_footprint(c.members)
                )
                evidence_retry = c.n_conf > nconf_expanded and kids_expanded > 0
                if not (context_changed or evidence_retry):
                    continue
            if bootstrap and c.n_conf < 1:
                # BOOTSTRAP regime gate (H = ∅): V ≡ 0 for everything, so
                # expansion compute is allocated by the only available
                # signal — recurrence evidence. Self-disables once the
                # first concept makes H non-empty.
                continue
            eligible.append(c)
        pool = eligible

        # [SPEC GROUNDED DEVIATION §4.10.3(b)/§4.17] EPISTEMIC IN-FLIGHT
        # GUARD. The excursion's own completion condition treats a scope
        # candidate as resolved when it is "expanded OR CRYSTALLIZED", and
        # §4.10.4 defines the window's purpose as surprise RESOLUTION —
        # i.e. crystallizing a model of the violated contingency. But
        # expanding a candidate REMOVES it from the frontier (§4.11.1) and
        # its children start at N_conf = 0 (§4.3.1), so consuming a
        # candidate mid-accumulation destroys its in-flight §4.17
        # evidence. Measured consequence without the guard (CausalChain
        # Autopilot, 1800 steps): 185/185 expansions of SEQUENCE
        # (Rule-1 / EVENT_CAUSAL) candidates happened
        # inside EPISTEMIC_EXCURSION windows, 42 of 48 small causal pairs
        # were consumed at N_conf ≤ 1, and 5 more at N_conf = 2 with every
        # other predicate clause already satisfied — one regeneration away
        # from crystallizing. The surprise windows themselves fire every
        # action→effect cycle BECAUSE no causal concept exists yet, so the
        # window kept eating the exact evidence whose crystallization
        # would end the surprises: a self-sustaining consumption loop,
        # 0 EVENT_CAUSAL concepts in 3000 steps.
        #
        # The guard: while a surprise window is open, top-K selection
        # skips candidates that are family-forming AND have begun
        # accumulating confirmations (0 < N_conf < θ_conf) — they are
        # in-flight toward crystallization and the predicate check (step
        # 8 of the §8.11 order) runs every step regardless of mode, so
        # they exit the frontier as concepts the moment N_conf reaches
        # θ_conf. Scaffold singletons (no family-forming edge) are NOT
        # protected — they remain expandable, and their expansions are
        # precisely what regenerates the protected residents (child-add
        # hash-match merge, §4.17 "produced by expansion from a different
        # parent"), so the evidence engine keeps running inside the
        # window. Candidates at N_conf = 0 stay freely expandable: the
        # depth-first exploration cascade is untouched. All other modes
        # (SEEDING/COMMITTED/EXCURSION) keep the literal spec selection —
        # the primary fertility drive's cascade is not modified.
        if self.mode == "EPISTEMIC_EXCURSION":
            pool = [c for c in pool if not self._in_flight_for_crystallization(c)]

        # [SPEC COMPLIANCE §4.11] If the scope has no frontier candidates,
        # seed the scope directly.
        if scope_filter is not None and not pool:
            self._seed_scope_directly(scope_filter, t)
            pool = [c for c in self.frontier if c.members & scope_filter]
            if self.mode == "EPISTEMIC_EXCURSION":
                pool = [
                    c for c in pool if not self._in_flight_for_crystallization(c)
                ]
            if not pool:
                return

        # §4.11 batch-expansion atomicity: top-K selected and V values read
        # BEFORE any expansion occurs; all expansions run against the
        # step-start frontier snapshot.
        scored: List[Tuple[float, bytes, CandidateSubgraph]] = [
            (priority_fn(c), c.canonical_hash, c) for c in pool
        ]
        scored.sort(key=lambda x: (-x[0], x[1]))
        top_k = scored[:k]

        top_k_hashes = {c.canonical_hash for _, _, c in top_k}
        self.frontier = [
            c for c in self.frontier if c.canonical_hash not in top_k_hashes
        ]
        for h in top_k_hashes:
            self.frontier_hashes.discard(h)
            self._frontier_by_hash.pop(h, None)
            self._remove_hash_from_all_basins(h)

        all_children: List[CandidateSubgraph] = []
        top_k_parents: List[CandidateSubgraph] = []
        children_per_parent: Dict[bytes, int] = {}
        for _prio_val, _h, parent in top_k:
            top_k_parents.append(parent)
            self.cg.unregister_candidate_nodes(
                parent.canonical_hash,
                parent.members,
            )
            self._remove_from_subgraph_index(parent.canonical_hash)
            if scope_filter:
                self.current_scope_expanded.add(parent.canonical_hash)
                # [Batch D / P1-1 MED-05] Epistemic progress checkpoint:
                # a scope candidate was expanded.
                if self.mode == "EPISTEMIC_EXCURSION":
                    self._epistemic_last_progress_t = self._current_step
                    self._epistemic_last_progress_rev = (
            self.cg.graph_revision()
            if hasattr(self.cg, "graph_revision")
            else 0
        )
            children = self._generate_children(parent, t)
            children_per_parent[parent.canonical_hash] = len(children)
            if self.mode == "EPISTEMIC_EXCURSION":
                # §4.10.2: mark children produced during the excursion so
                # they are excluded from Spine stall detection.
                for ch in children:
                    ch.epistemic_origin = True
            all_children.extend(children)
            self.diagnostics["expanded_total"] += 1

        # §4.11: "The transposition table is updated after all K
        # expansions, not between them." Entry layout:
        #   (step, footprint_at_expansion, n_conf_at_expansion, n_children)
        # The footprint and the (n_conf, n_children) pair are the two
        # unlock signals checked at eligibility time (see _expand_mode):
        # context novelty (footprint changed) or evidence novelty with a
        # productive previous expansion (new confirmations AND the last
        # expansion generated at least one child).
        for parent in top_k_parents:
            self.transposition_table[parent.canonical_hash] = (
                t,
                self._expansion_footprint(parent.members),
                parent.n_conf,
                children_per_parent.get(parent.canonical_hash, 0),
            )

        all_children.sort(key=lambda c: c.canonical_hash)
        for child in all_children:
            if child.canonical_hash in self.transposition_table:
                continue
            if child.canonical_hash in self.crystallized_snapshot:
                continue
            if self._is_tombstoned(child.canonical_hash):
                continue
            existing = self._find_frontier_by_hash(child.canonical_hash)
            if existing is not None:
                # §4.17: production by expansion from a different parent is
                # a temporal re-observation of the same structure.
                existing.T_last = child.T_last
                existing.contexts = existing.contexts | child.contexts
                existing.original_detections = (
                    existing.original_detections | child.original_detections
                )
                existing.original_detection_hashes = (
                    existing.original_detection_hashes | child.original_detection_hashes
                )
                existing.anchors = self._merge_anchors(
                    existing.anchors,
                    child.anchors,
                )
                self._bump_n_conf(existing, t)
                existing.V_step = -1
                existing.f_exp_step = -1
                continue
            self._add_to_frontier(child)

        self._evict_transposition_table_if_over_cap()
        self._evict_frontier_if_over_cap()

    def _seed_scope_directly(
        self,
        scope_members: FrozenSet[int],
        t: int,
    ) -> None:
        """[SPEC COMPLIANCE §4.11] Direct scope seeding for EPISTEMIC
        EXCURSION when no frontier candidate overlaps the scope.
        """
        if not scope_members:
            return
        edges = self._induced_edges(scope_members)
        canonical = self._canonical_hash(scope_members, edges)
        if canonical in self.frontier_hashes:
            return
        if canonical in self.transposition_table:
            return
        if canonical in self.crystallized_snapshot:
            return
        if self._is_tombstoned(canonical):
            return

        # cross_section_key = hash_128(serialize_tuple(sorted(sig_hashes))).
        sig_hashes: List[bytes] = []
        for nid in scope_members:
            sig = self.cg.get_pattern_node_signature(nid)
            if sig is not None:
                sig_hashes.append(sig[1])
        cs_key = hash_128(
            serialize_tuple([serialize_bytes(h) for h in sorted(sig_hashes)])
        )

        cand = CandidateSubgraph(
            canonical_hash=canonical,
            cross_section_key=cs_key,
            members=scope_members,
            edges=edges,
            anchors=(),
            contexts=frozenset({self._current_s_t_bytes}),
            t_created=t,
            T_first=t,
            T_last=t,
            parent_hash=None,
            construction_op="SEED",
            original_detections=frozenset(),
            original_detection_hashes=frozenset(),
            n_conf=0,
            last_n_conf_step=-1,
            epistemic_origin=True,
        )
        self._add_to_frontier(cand)
        self.diagnostics["epistemic_seeds"] += 1

    def _evict_transposition_table_if_over_cap(self) -> None:
        cap = self.cfg.transposition_table_max
        if len(self.transposition_table) <= cap:
            return
        to_remove = len(self.transposition_table) - cap
        for _ in range(to_remove):
            try:
                first_key = next(iter(self.transposition_table))
            except StopIteration:
                break
            del self.transposition_table[first_key]
            self.diagnostics["transposition_evictions"] += 1

    def _generate_children(
        self,
        parent: CandidateSubgraph,
        t: int,
    ) -> List[CandidateSubgraph]:
        children: List[CandidateSubgraph] = []
        ext = self._best_extension(parent, t)
        if ext is not None:
            children.append(ext)
        merge = self._best_merge(parent, t)
        if merge is not None:
            children.append(merge)
        edge_add = self._best_edge_addition(parent, t)
        if edge_add is not None:
            children.append(edge_add)
        return children

    def _link_evidence(self, nid: int, members: FrozenSet[int]) -> float:
        """Total §1.12 edge weight between `nid` and the member set — the
        graph's own accumulated evidence that this neighbor belongs to
        this assembly (Hebbian: synaptic strength)."""
        terms = []
        for edge in self.cg.get_outgoing_edges(nid):
            if edge.child in members:
                terms.append((edge.child, edge.edge_type.value, 0, max(0.0, edge.weight)))
        for edge in self.cg.get_incoming_edges(nid):
            if edge.parent in members:
                terms.append((edge.parent, edge.edge_type.value, 1, max(0.0, edge.weight)))
        # [DETERMINISM §9] float addition is not associative: summing in
        # adjacency order made exact Hebbian ties seed-flip at ULP level
        # (measured: EXTEND chose node 22 vs 23 across PYTHONHASHSEED).
        # Sum in canonical sorted order instead.
        total = 0.0
        for _k in sorted(terms):
            total += _k[3]
        return total

    def _in_flight_for_crystallization(self, cand: CandidateSubgraph) -> bool:
        """True iff the candidate is mid-flight through the §4.17/§4.20
        crystallization pipeline and consuming it now would destroy
        in-progress evidence (see the EPISTEMIC IN-FLIGHT GUARD in
        _expand_mode). Two phases are in-flight:

        1. ACCUMULATION — family-forming with 0 < N_conf < θ_conf: one or
           two more regenerations would satisfy the Builder predicate.
        2. EMISSION — predicate already satisfied (N_conf ≥ θ_conf) but
           the canonical hash has not appeared in the ConceptGraph's
           crystallized set yet: the candidate must REMAIN in the
           frontier and keep being emitted for ~θ_conf consecutive steps,
           because the ConceptGraph-side candidate (§4.20: the existing
           crystallization pipeline re-checks N_opp/N_conf/N_anchor with
           its own counters) accrues +1 N_conf per arriving CrossSection
           and only crystallizes the concept on the third. Expansion
           during this window removes the emitter from the frontier and
           the ConceptGraph-side candidate stalls below its threshold —
           measured: 16 both-edge candidates consumed by epistemic
           expansion AT N_conf = 3, zero concepts produced from them.

        Scaffold candidates (no family-forming edge) are never in-flight:
        they cannot crystallize, and they are the regeneration engine —
        their expansions are what merge-confirm the protected residents.
        """
        if cand.n_conf <= 0:
            return False
        if not _is_family_forming(cand):
            return False
        return cand.canonical_hash not in self.crystallized_snapshot

    def _child_is_viable(self, canonical: bytes) -> bool:
        """[SPEC COHERENCE §4.4] Selection-time viability of a generated
        child: a child whose canonical hash is already retired (expanded
        and transposed, crystallized, or tombstoned) is dropped at the
        add loop — selecting it means the operation produces NOTHING
        this expansion. §4.4's "select the neighbor/partner/edge that
        maximizes V(S')" is only meaningful over S' that can actually
        enter the frontier: either fresh, or hash-matching a resident
        candidate (whose §4.17 statistics the regeneration then
        advances). Filtering at selection — instead of wasting the
        expansion on an add-loop drop — is what lets the
        regeneration-merge engine walk down its argmax ordering as
        rungs crystallize or transpose.
        """
        if canonical in self.transposition_table:
            return False
        if canonical in self.crystallized_snapshot:
            return False
        if self._is_tombstoned(canonical):
            return False
        return True

    def _best_extension(
        self,
        parent: CandidateSubgraph,
        t: int,
    ) -> Optional[CandidateSubgraph]:
        """§4.4: select the neighbor whose addition maximizes V(S′).

        SANCTIONED EXTENSION (Hebbian tie-break): when child values are
        EXACTLY tied — which is the whole bootstrap regime (H = ∅ ⇒ V ≡ 0
        per §4.7) — the spec's hash-order tie-break freezes neighbor
        selection onto one arbitrary (lowest-hash) neighbor forever, and
        deterministic retries can never reach the causally-relevant
        neighbor (e.g. the SEQUENCE-linked effect node of a toggle
        event). Ties are therefore broken by accumulated edge-weight
        evidence between the neighbor and the parent assembly FIRST —
        assemblies grow along their strongest synapses — and only exact
        evidence ties fall through to the spec's canonical-hash order.
        Once V differentiates (H ≠ ∅), this rule is inert."""
        candidate_nodes = self._neighbor_nodes_of(parent)
        if not candidate_nodes:
            return None
        # [SPEC COHERENCE §4.12/§4.17] Child edge sets must be the LIVE
        # induced subgraph over the child's members, not the parent's
        # frozen prefix plus the new node's scan. The frozen prefix is
        # path-dependent: it carries gaps for edges born after the parent
        # was constructed, so independent lineages converging on the same
        # member set produced DIFFERENT canonical hashes (measured: 2–7
        # distinct hashes per member set among Rule-1 candidates, every
        # one at n_conf=0). §4.17's N_conf re-observation ("produced by
        # expansion from a different parent" with the same canonical
        # hash) could then never fire for multi-node structures, and no
        # EVENT_CAUSAL/TEMPORAL_CHAIN concept could crystallize. With the
        # parent side refreshed to its current induced set, the EXTEND
        # algebra below is exact: child.edges == induced(child.members),
        # a function of the member set alone — lineages converge, hashes
        # recur, statistics accumulate. Refresh is computed once per
        # expansion (memoized per step); per-trial cost stays O(deg(nid)).
        refreshed_parent_edges = self._induced_edges_memo(parent.members)
        best: Optional[CandidateSubgraph] = None
        best_key: Optional[Tuple[float, float, bytes]] = None
        for nid in sorted(candidate_nodes):
            new_members = parent.members | {nid}
            if len(new_members) > self.cfg.builder_candidate_max:
                continue
            new_edges = self._extend_induced_edges(
                refreshed_parent_edges, parent.members, nid
            )
            child = self._make_child(
                parent,
                new_members,
                new_edges,
                t,
                op="EXTEND",
            )
            if not self._child_is_viable(child.canonical_hash):
                continue
            v = self._compute_V(child, t)
            key = (v, self._link_evidence(nid, parent.members), _negate_bytes(child.canonical_hash))
            if best_key is None or key > best_key:
                best, best_key = child, key
        return best

    def _best_merge(
        self,
        parent: CandidateSubgraph,
        t: int,
    ) -> Optional[CandidateSubgraph]:
        """[SPEC COMPLIANCE §4.4] Merge requires V(union) > max(V(a), V(b))."""
        parent_v = self._compute_V(parent, t)
        overlap_candidates = [
            c
            for c in self.frontier
            if c.canonical_hash != parent.canonical_hash
            and (c.members & parent.members)
        ]
        if not overlap_candidates:
            return None
        best: Optional[CandidateSubgraph] = None
        best_key: Optional[Tuple[float, float, bytes]] = None
        for other in overlap_candidates:
            other_v = self._compute_V(other, t)
            new_members = parent.members | other.members
            if len(new_members) > self.cfg.builder_candidate_max:
                continue
            new_edges = self._union_induced_edges(parent, other, new_members)
            child = self._make_child_merge(
                parent,
                other,
                new_members,
                new_edges,
                t,
            )
            if not self._child_is_viable(child.canonical_hash):
                continue
            child_v = self._compute_V(child, t)
            # GUARD: only accept if the union improves over both parents.
            if child_v <= max(parent_v, other_v):
                continue
            # Hebbian tie-break (see _best_extension).
            evidence = 0.0
            for nid in other.members:
                evidence += self._link_evidence(nid, parent.members)
            key = (child_v, evidence, _negate_bytes(child.canonical_hash))
            if best_key is None or key > best_key:
                best, best_key = child, key
        return best

    def _best_edge_addition(
        self,
        parent: CandidateSubgraph,
        t: int,
    ) -> Optional[CandidateSubgraph]:
        existing_keys = parent.edges
        addable: Set[Tuple[int, int, EdgeType]] = set()
        for nid in parent.members:
            for edge in self.cg.get_outgoing_edges(nid):
                if edge.child in parent.members:
                    key = (edge.parent, edge.child, edge.edge_type)
                    if key not in existing_keys:
                        addable.add(key)
            for edge in self.cg.get_incoming_edges(nid):
                if edge.parent in parent.members:
                    key = (edge.parent, edge.child, edge.edge_type)
                    if key not in existing_keys:
                        addable.add(key)
        if not addable:
            return None
        best: Optional[CandidateSubgraph] = None
        best_key: Optional[Tuple[float, float, bytes]] = None
        for ek in sorted(addable, key=lambda e: (e[0], e[1], e[2].value)):
            new_edges = parent.edges | {ek}
            child = self._make_child(
                parent,
                parent.members,
                new_edges,
                t,
                op="EDGE_ADD",
            )
            if not self._child_is_viable(child.canonical_hash):
                continue
            v = self._compute_V(child, t)
            # Hebbian tie-break (see _best_extension): stronger synapse
            # first when V carries no signal.
            edge_obj = self.cg.get_edge(ek[0], ek[1], ek[2])
            evidence = max(0.0, edge_obj.weight) if edge_obj is not None else 0.0
            key = (v, evidence, _negate_bytes(child.canonical_hash))
            if best_key is None or key > best_key:
                best, best_key = child, key
        return best

    def _neighbor_nodes_of(self, cand: CandidateSubgraph) -> Set[int]:
        result: Set[int] = set()
        for nid in cand.members:
            for edge in self.cg.get_outgoing_edges(nid):
                if edge.child not in cand.members:
                    result.add(edge.child)
            for edge in self.cg.get_incoming_edges(nid):
                if edge.parent not in cand.members:
                    result.add(edge.parent)
        return result

    def _induced_edges(
        self,
        members: FrozenSet[int],
    ) -> FrozenSet[Tuple[int, int, EdgeType]]:
        result: Set[Tuple[int, int, EdgeType]] = set()
        member_set = set(members)
        for nid in members:
            for edge in self.cg.get_outgoing_edges(nid):
                if edge.child in member_set:
                    result.add((edge.parent, edge.child, edge.edge_type))
            for edge in self.cg.get_incoming_edges(nid):
                if edge.parent in member_set:
                    result.add((edge.parent, edge.child, edge.edge_type))
        return frozenset(result)

    def _induced_edges_memo(
        self,
        members: FrozenSet[int],
    ) -> FrozenSet[Tuple[int, int, EdgeType]]:
        """Step-scoped memoization of _induced_edges.

        The edge store is stable within a Builder step (binder and
        self-binder writes for step t complete before builder.step(t)
        runs; pruning runs in the pipeline's edge-update phase), so a
        per-step cache is exact. Child construction (§4.4 EXTEND/MERGE)
        reads each parent's induced set once per expansion and reuses it
        across neighbor/partner trials.
        """
        t = self._current_step
        if self._induced_memo_step != t:
            self._induced_memo = {}
            self._induced_memo_step = t
        cached = self._induced_memo.get(members)
        if cached is not None:
            return cached
        result = self._induced_edges(members)
        self._induced_memo[members] = result
        return result

    def _expansion_footprint(self, members: FrozenSet[int]) -> bytes:
        """Canonical hash of the LIVE expansion footprint of a member set:
        every graph edge touching any member (intra-member, outgoing, and
        incoming — exactly the edge population the three §4.4 construction
        operations read when expanding a candidate with these members:
        EXTEND scans neighbor adjacency, MERGE cross-scans, EDGE_ADD
        enumerates intra-member edges).

        Used as the transposition-table unlock signal (§4.11.1): the memo
        exists "to prevent re-expansion of structurally identical
        subgraphs" — its purpose is avoiding REDUNDANT re-expansion. A
        re-expansion is redundant exactly when it would generate the same
        children; with deterministic selection, child sets are a function
        of (members, frozen edges, live footprint, V landscape). When the
        footprint has changed since the memoized expansion, the candidate
        is contextually not the subgraph that was memoized — new SEQUENCE
        edges around it can now unite into Rule-1
        (§4.14) causal children that did not exist before — so the retry
        is productive, not redundant. When the footprint is unchanged, the
        memo stands absolutely.

        This replaces the former confirmation-count unlock (entry n_conf),
        which misfired in BOTH directions: (a) post-bootstrap it let a
        permanently re-seeded high-V singleton re-unlock every single step
        and monopolize the K-slot budget (measured: same parent expanded
        275/300 steps), starving other candidates; and (b)
        it unlocked retries that produced identical children (pure waste)
        whenever n_conf rose without any graph change. The footprint is
        the exact novelty signal: it changes only when the graph around
        the candidate actually changed. Deterministic (§9): canonical
        serialization of sorted (parent, child, edge_type) triples.
        """
        t = self._current_step
        if self._footprint_memo_step != t:
            self._footprint_memo = {}
            self._footprint_memo_step = t
        cached = self._footprint_memo.get(members)
        if cached is not None:
            return cached
        triples: Set[Tuple[int, int, int]] = set()
        for nid in members:
            for edge in self.cg.get_outgoing_edges(nid):
                triples.add((edge.parent, edge.child, edge.edge_type.value))
            for edge in self.cg.get_incoming_edges(nid):
                triples.add((edge.parent, edge.child, edge.edge_type.value))
        fp = hash_128(
            serialize_tuple(
                [
                    serialize_tuple(
                        [serialize_int(s), serialize_int(d), serialize_int(et)]
                    )
                    for s, d, et in sorted(triples)
                ]
            )
        )
        self._footprint_memo[members] = fp
        return fp

    def _extend_induced_edges(
        self,
        base_edges: FrozenSet[Tuple[int, int, EdgeType]],
        base_members: FrozenSet[int],
        new_nid: int,
    ) -> FrozenSet[Tuple[int, int, EdgeType]]:
        """[PERFORMANCE — identical result] induced(base ∪ {new_nid}) from
        the already-computed induced(base): only the new node's adjacency
        is scanned (O(degree(new_nid)) instead of O(Σ degree(members))).
        _best_extension/_best_merge call this once per neighbor trial;
        the full recompute dominated Builder expansion once spatial
        channels revived the edge population (measured: 8.4M dict gets /
        80 steps). Set algebra is exact: every induced edge either lies
        inside base (already present) or touches new_nid (scanned here,
        self-loops included)."""
        result: Set[Tuple[int, int, EdgeType]] = set(base_edges)
        for edge in self.cg.get_outgoing_edges(new_nid):
            if edge.child in base_members or edge.child == new_nid:
                result.add((edge.parent, edge.child, edge.edge_type))
        for edge in self.cg.get_incoming_edges(new_nid):
            if edge.parent in base_members or edge.parent == new_nid:
                result.add((edge.parent, edge.child, edge.edge_type))
        return frozenset(result)

    def _union_induced_edges(
        self,
        a: "CandidateSubgraph",
        b: "CandidateSubgraph",
        new_members: FrozenSet[int],
    ) -> FrozenSet[Tuple[int, int, EdgeType]]:
        """[SPEC COHERENCE §4.12/§4.17] EXACT induced(a.members ∪
        b.members) over the live graph — see _best_extension rationale:
        merge children must be a function of the member set alone so
        independent lineages converge on one canonical hash.

        Exactness: every induced edge of the union lies inside a (∈
        induced(a)), inside b (∈ induced(b)), or crosses between the two
        — and every cross edge touches BOTH sides, so scanning the
        smaller side's live adjacency filtered by the union captures all
        of them (self-loops included: a self-loop on n ∈ a is in
        induced(a)). Both induced() reads are step-memoized; the
        cross-scan is O(|smaller side| · deg).
        """
        result: Set[Tuple[int, int, EdgeType]] = set(
            self._induced_edges_memo(a.members)
        )
        result.update(self._induced_edges_memo(b.members))
        member_set = set(new_members)
        scan_side = a.members if len(a.members) <= len(b.members) else b.members
        for nid in scan_side:
            for edge in self.cg.get_outgoing_edges(nid):
                if edge.child in member_set:
                    result.add((edge.parent, edge.child, edge.edge_type))
            for edge in self.cg.get_incoming_edges(nid):
                if edge.parent in member_set:
                    result.add((edge.parent, edge.child, edge.edge_type))
        return frozenset(result)

    def _filter_live_edges(
        self,
        edges: FrozenSet[Tuple[int, int, EdgeType]],
    ) -> FrozenSet[Tuple[int, int, EdgeType]]:
        """[SPEC COHERENCE §4.12] Restrict a constructed child's edge set
        to edges that currently EXIST in the pattern graph.

        A CandidateSubgraph is, per §4.12, a subgraph of the pattern
        graph: its canonical hash "equals the same value for any two
        structurally isomorphic subgraphs". Parent candidates inherit
        their own frozen edge sets into children (§4.4 EXTEND/MERGE/
        EDGE_ADD); when a graph edge later dies (EMA decay + per-type
        inactivity pruning, §1.12), the inherited tuple no longer denotes
        any edge of the graph. Keeping such phantom tuples has two
        measured consequences (CausalChain autopilot, 900 steps):

        1. IDENTITY FRAGMENTATION. Child edge sets become path-dependent
           historical accumulations (observed: 157–229 frozen tuples of
           which 81–97 were dead, vs a live induced set of ~119–147).
           Two lineages converging on the SAME live structure therefore
           produce different canonical hashes, so the §4.17 N_conf
           re-observation mechanism ("produced by expansion from a
           different parent" with the same canonical hash) can never
           fire: every multi-node candidate stayed at n_conf=0, span=0
           and no EVENT_CAUSAL concept could crystallize even though
           109 frontier candidates resolved to that family.

        2. PHANTOM STRUCTURE DOWNSTREAM. Crystallization, CrossSection
           emission, and Consolidator input would all carry edges that
           do not exist in the graph — while every other consumer of
           subgraph structure (concepts via _concept_V_via_builder, the
           Mapper's situation subgraph, the B1 consolidation handoffs)
           reads the LIVE induced subgraph.

        Filtering happens at child-construction time only; existing
        candidates keep their IMMUTABLE canonical_hash (§4.3.1), and
        spec-literal seeding (edges = {}, §4.13) is untouched. Under a
        stabilized edge store the filtered construction is a function of
        the member set alone (scan-determined), so independent lineages
        converge on identical hashes and N_conf/span accumulate as §4.17
        intends.
        """
        store = self.cg.edges
        return frozenset(e for e in edges if e in store)

    def _make_child(
        self,
        parent: CandidateSubgraph,
        members: FrozenSet[int],
        edges: FrozenSet[Tuple[int, int, EdgeType]],
        t: int,
        op: str,
    ) -> CandidateSubgraph:
        edges = self._filter_live_edges(edges)
        canonical = self._canonical_hash(members, edges)
        new_det_hashes = set(parent.original_detection_hashes)
        new_dets = set(parent.original_detections)
        anchors = parent.anchors
        added_nodes = members - parent.members
        # §4.3.1 update-on-extension: inherit detections, add the new
        # nodes' originating detections, append their alphas to the anchor
        # tuple — one resolution pass for both.
        for nid in sorted(added_nodes):
            det = self._resolve_detection_for_node(nid, t)
            if det is not None:
                new_dets.add(det)
                new_det_hashes.add(_detection_hash(det))
                anchors = self._append_anchor(anchors, (det.alpha, t))
        cs_key = _canonical_detection_set_hash(frozenset(new_det_hashes))
        contexts = parent.contexts | {self._current_s_t_bytes}

        return CandidateSubgraph(
            canonical_hash=canonical,
            cross_section_key=cs_key,
            members=members,
            edges=edges,
            anchors=anchors,
            contexts=contexts,
            t_created=t,
            # §4.3.1: T_first is the step of FIRST OBSERVATION of THIS
            # structure. The update-on-extension rules enumerate exactly
            # which fields are inherited (anchors, detections, hashes,
            # contexts, key) — T_first is not among them. Inheriting the
            # parent's T_first would gift a brand-new structure an
            # artificial temporal span and erode the §4.17 span clause.
            T_first=t,
            T_last=t,
            parent_hash=parent.canonical_hash,
            construction_op=op,
            original_detections=frozenset(new_dets),
            original_detection_hashes=frozenset(new_det_hashes),
            n_conf=0,
            last_n_conf_step=-1,
            epistemic_origin=parent.epistemic_origin
            and self.mode == "EPISTEMIC_EXCURSION",
        )

    def _make_child_merge(
        self,
        a: CandidateSubgraph,
        b: CandidateSubgraph,
        members: FrozenSet[int],
        edges: FrozenSet[Tuple[int, int, EdgeType]],
        t: int,
    ) -> CandidateSubgraph:
        edges = self._filter_live_edges(edges)  # §4.12 subgraph coherence
        canonical = self._canonical_hash(members, edges)
        combined_dets = a.original_detections | b.original_detections
        combined_hashes = a.original_detection_hashes | b.original_detection_hashes
        cs_key = _canonical_detection_set_hash(combined_hashes)
        anchors = self._merge_anchors(a.anchors, b.anchors)
        contexts = a.contexts | b.contexts | {self._current_s_t_bytes}

        # [SPEC COMPLIANCE §4.3.1 / §4.17] A merge produces a new canonical
        # hash, so N_conf resets to 0. It is incremented only on future
        # re-seeds or cross-parent expansions. T_first is this merged
        # structure's first observation: t (see _make_child rationale —
        # the §4.3.1 inheritance enumeration does not include T_first).
        return CandidateSubgraph(
            canonical_hash=canonical,
            cross_section_key=cs_key,
            members=members,
            edges=edges,
            anchors=anchors,
            contexts=contexts,
            t_created=t,
            T_first=t,
            T_last=t,
            parent_hash=a.canonical_hash,
            construction_op="MERGE",
            original_detections=combined_dets,
            original_detection_hashes=combined_hashes,
            n_conf=0,
            last_n_conf_step=-1,
            epistemic_origin=(
                self.mode == "EPISTEMIC_EXCURSION"
                and (a.epistemic_origin or b.epistemic_origin)
            ),
        )

    def _append_anchor(
        self,
        anchors: Tuple[Tuple[Any, int], ...],
        new_anchor: Tuple[Any, int],
    ) -> Tuple[Tuple[Any, int], ...]:
        combined = list(anchors) + [new_anchor]
        combined.sort(key=lambda a: (a[1], hash_128(serialize_value(a[0]))))
        if len(combined) > self.cfg.max_anchors_per_candidate:
            combined = combined[-self.cfg.max_anchors_per_candidate :]
        return tuple(combined)

    def _merge_anchors(
        self,
        a: Tuple[Tuple[Any, int], ...],
        b: Tuple[Tuple[Any, int], ...],
    ) -> Tuple[Tuple[Any, int], ...]:
        combined = list(a) + list(b)
        seen: Set[bytes] = set()
        deduped = []
        for an in combined:
            # Canonical serialization only (no repr() fallback — see
            # subgraph_to_cross_section for the determinism rationale).
            key = serialize_tuple(
                [
                    serialize_value(an[0]),
                    serialize_int(an[1]),
                ]
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(an)
        deduped.sort(key=lambda x: (x[1], hash_128(serialize_value(x[0]))))
        if len(deduped) > self.cfg.max_anchors_per_candidate:
            deduped = deduped[-self.cfg.max_anchors_per_candidate :]
        return tuple(deduped)

    # -------------------------------------------------------------------------
    # Frontier management
    # -------------------------------------------------------------------------

    def _add_to_frontier(self, cand: CandidateSubgraph) -> None:
        self.frontier.append(cand)
        self.frontier_hashes.add(cand.canonical_hash)
        self._frontier_by_hash[cand.canonical_hash] = cand
        self.cg.register_candidate_nodes(cand.canonical_hash, cand.members)
        # Index for incremental basin updates.
        self._subgraph_members_cache[cand.canonical_hash] = cand.members
        for nid in cand.members:
            self._subgraph_index[nid].add(cand.canonical_hash)
        self._mark_dirty_members(cand.members)

    def _drop_frontier_entry(self, cand: CandidateSubgraph) -> None:
        """Shared removal bookkeeping for a candidate leaving the frontier
        (expansion, eviction, pruning, or crystallization)."""
        self.frontier_hashes.discard(cand.canonical_hash)
        self._frontier_by_hash.pop(cand.canonical_hash, None)
        self.cg.unregister_candidate_nodes(cand.canonical_hash, cand.members)
        self._remove_from_subgraph_index(cand.canonical_hash)
        self._remove_hash_from_all_basins(cand.canonical_hash)

    def _remove_from_subgraph_index(self, h: bytes) -> None:
        members = self._subgraph_members_cache.pop(h, None)
        if members is None:
            return
        for nid in members:
            s = self._subgraph_index.get(nid)
            if s is not None:
                s.discard(h)
                if not s:
                    del self._subgraph_index[nid]

    def _remove_hash_from_all_basins(self, h: bytes) -> None:
        """[SPEC COMPLIANCE §4.3.2] Basins hold 'CandidateSubgraphs and
        crystallized subgraphs'. A candidate that leaves the frontier is
        neither, so its hash must be dropped from its basin's member set
        (O(1) via the reverse index).

        Crystallized hashes are NOT removed by this helper; the caller
        decides whether the hash is departing as a non-crystallized candidate.
        """
        bid = self._hash_to_basin.pop(h, None)
        if bid is not None and bid in self.basins:
            self.basins[bid].members.discard(h)

    def _find_frontier_by_hash(
        self,
        canonical_hash: bytes,
    ) -> Optional[CandidateSubgraph]:
        # O(1) (the former list scan ran per detection per step).
        return self._frontier_by_hash.get(canonical_hash)

    def _evict_frontier_if_over_cap(self) -> None:
        if len(self.frontier) <= self.cfg.frontier_max:
            return
        t = self._current_step

        def _score(c: CandidateSubgraph) -> Tuple[float, bytes]:
            v = self._compute_V(c, t)
            fresh = math.exp(-max(0, t - c.t_created) / self.cfg.tau_fresh)
            return (v + self.cfg.eta_fresh * fresh, c.canonical_hash)

        self.frontier.sort(key=_score)
        to_evict = len(self.frontier) - self.cfg.frontier_max
        evicted = self.frontier[:to_evict]
        self.frontier = self.frontier[to_evict:]
        for c in evicted:
            self._drop_frontier_entry(c)
            self.diagnostics["frontier_evictions"] += 1

    def _prune_frontier(self, t: int) -> None:
        """§4.11.1 rule 3, exactly: a candidate is removed when
        `V(S) < θ_prune_frontier` for `K_prune_frontier` CONSECUTIVE steps
        and it has not been selected for expansion in that window.

        The "not selected" clause is satisfied structurally: selection
        removes the candidate from the frontier (§4.11.1 rule 1), so any
        candidate still resident here was not selected this step; a
        re-seed of an expanded hash creates a fresh candidate whose
        low-V streak restarts at zero.

        The former extra protections (age < τ_fresh immunity, a
        last-confirmation gate, and a low-V streak that only advanced when
        the candidate was ALSO unconfirmed for K steps) were fabricated
        drift: they made periodically re-observed-but-barren candidates
        unprunable even though V — which already includes the witness term
        W(N_conf) — stayed below θ_prune_frontier.

        BOOTSTRAP DOMAIN GUARD (sanctioned extension): "proven
        non-productive" presupposes productivity is MEASURABLE. While the
        fertile set H is empty (§4.7: proximity ≡ 0 when H = ∅ — no
        concepts have crystallized yet), every candidate has V ≡ 0 and the
        rule would cannibalize exactly the evidence-bearing structures the
        first crystallization depends on: a deadlock (pruning prevents the
        concepts that would make pruning meaningful). V-pruning therefore
        activates only once H contains a concept with EF ≥ RF_MIN_FERTILE
        ([P0-2 FIX] strengthened from "H ≠ ∅": under the cold-start rescue
        H can be non-empty while every member sits below the fertility
        floor, and pruning must keep waiting for meaningful fertility);
        until then the frontier is bounded
        solely by the §4.11.1 capacity rule (FRONTIER_MAX, lowest-priority
        eviction), which remains active in all regimes."""
        # [P0-2 FIX] Pruning requires at least one concept with meaningful
        # fertility (EF >= rf_min_fertile). Under the cold-start rescue,
        # H may contain concepts with EF < rf_min_fertile; pruning must
        # wait for meaningful fertility, preserving pre-rescue semantics.
        # Reuses the step-cached _fertile_concepts() — no fresh scan.
        H = self._fertile_concepts()
        if not any(
            getattr(c, "EF_combined", 0.0) >= self.cfg.rf_min_fertile
            for _cid, c in H
        ):
            return

        self.diagnostics["prune_frontier_active_steps"] += 1

        survivors: List[CandidateSubgraph] = []
        for c in self.frontier:
            v = self._compute_V(c, t)
            if v < self.cfg.theta_prune_frontier:
                c.low_v_steps += 1
                # [Batch D / P1-1] Endogenous OR bound: graph revisions
                # observed while continuously low-V.
                _grev_now = (
            self.cg.graph_revision()
            if hasattr(self.cg, "graph_revision")
            else 0
        )
                if c.last_low_v_rev < 0:
                    c.last_low_v_rev = _grev_now
                c.low_v_revision_events = _grev_now - c.last_low_v_rev
                if (
                    c.low_v_steps >= self.cfg.k_prune_frontier
                    or self._graph_drift(c.last_low_v_rev)
                    >= self.cfg.k_prune_frontier_revision
                ):
                    self._drop_frontier_entry(c)
                    self.diagnostics["frontier_evictions"] += 1
                    continue
            else:
                c.low_v_steps = 0
                c.low_v_revision_events = 0
                c.last_low_v_rev = -1
            survivors.append(c)
        self.frontier = survivors

    # -------------------------------------------------------------------------
    # §4.6 / §4.7 Value function
    # -------------------------------------------------------------------------

    def _compute_V(self, cand: CandidateSubgraph, t: int) -> float:
        if cand.V_step == t:
            return cand.V
        f_exp = self._compute_F_expected(cand)
        f_rec_raw = self._compute_F_recursive(cand)
        # §4.6.3: F_RECURSIVE_NORM_SCALE(t) = λ · θ_fertile(t). θ_fertile
        # already carries the RF_MIN_FERTILE floor by construction (§4.7).
        norm_scale = self.cfg.lambda_f_recursive * self._theta_fertile
        f_rec_norm = min(1.0, f_rec_raw / (norm_scale + _EPS))
        w = self._compute_W(cand)
        # §4.6.2 constitutional weights via config (defaults = parameters).
        v = (
            self.cfg.alpha_v * f_exp
            + self.cfg.beta_v * f_rec_norm
            + self.cfg.gamma_v * w
        )
        cand.V = v
        cand.V_step = t
        return v

    def _compute_F_expected(self, cand: CandidateSubgraph) -> float:
        """§4.7: Size-normalized average proximity to fertile concepts.
        When H = ∅ or fertile concepts have no members, proximity is
        mathematically 0.0. Exploration baseline is handled by Freshness.

        Cached per candidate per step: Curiosity, stall checks, and
        selection all read F_expected within one step; H and τ are
        step-cached, so the cache is consistent."""
        t = self._current_step
        if cand.f_exp_step == t:
            return cand.f_exp_cache
        value = 0.0
        if cand.members:
            H = self._fertile_concepts()
            if H and any(concept.members for _cid, concept in H):
                tau = self._compute_tau()
                total = sum(self._proximity(v, H, tau) for v in cand.members)
                value = total / len(cand.members)
        cand.f_exp_cache = value
        cand.f_exp_step = t
        return value

    def _compute_F_recursive(self, cand: CandidateSubgraph) -> float:
        if not cand.members:
            return 0.0
        total = sum(self._recursive_reach(v) for v in cand.members)
        return total / len(cand.members)

    def _recursive_reach(self, v: int) -> float:
        """R(v) per §4.7, over DIRECTED successor reachability:

            R(v) = Σ_{c ∈ N_D⁺(v)} λ^{d(v,c)} · RF(c) / |N_D⁺(v)|

        §4.7 direction specification: "N_D⁺(v) is the set of concepts
        reachable from v via DIRECTED edges... only successors count.
        Predecessors are excluded." The former undirected BFS measured
        proximity, not enablement. Concepts CONTAINING v (d = 0) are
        members of N_D⁺ (d ≤ D) with weight λ⁰ = 1 — the former d==0
        exclusion had no spec basis.

        RF(c) here is EF_combined-derived (§4.7 note / §8.12). Cached per
        node per step (§4.7 caching: O(1) read for cached entries;
        topology changes between steps invalidate by step key)."""
        t = self._current_step
        if self._reach_cache_step != t:
            self._reach_cache = {}
            self._reach_cache_step = t
        cached = self._reach_cache.get(v)
        if cached is not None:
            return cached

        H = self._fertile_concepts()
        if not H:
            self._reach_cache[v] = 0.0
            return 0.0
        adj = self._pattern_adjacency_directed()
        D = self.cfg.f_recursive_depth
        lambda_d = self.cfg.lambda_f_recursive

        reachable: Dict[int, int] = {v: 0}
        frontier = deque([(v, 0)])
        while frontier:
            nid, dist = frontier.popleft()
            if dist >= D:
                continue
            for nb in adj.get(nid, ()):
                if nb in reachable and reachable[nb] <= dist + 1:
                    continue
                reachable[nb] = dist + 1
                frontier.append((nb, dist + 1))

        accum = 0.0
        count = 0
        for _cid, concept in H:
            best_d: Optional[int] = None
            for m in concept.members:
                d = reachable.get(m)
                if d is None:
                    continue
                if best_d is None or d < best_d:
                    best_d = d
                    if best_d == 0:
                        break
            if best_d is None:
                continue
            rf_c = getattr(concept, "EF_combined", 0.0)
            accum += (lambda_d**best_d) * rf_c
            count += 1
        result = 0.0 if count == 0 else accum / count
        self._reach_cache[v] = result
        return result

    def _compute_W(self, cand: CandidateSubgraph) -> float:
        return min(1.0, cand.n_conf / (self.cfg.theta_conf_subgraph + _EPS))

    # -------------------------------------------------------------------------
    # Fertility thresholds and distances
    # -------------------------------------------------------------------------

    def _fertile_concepts(self) -> List[Tuple[int, Concept]]:
        t = self._current_step
        if self._fertile_cache_step == t:
            return self._fertile_cache
        self._maybe_recompute_theta_fertile()
        theta = self._theta_fertile
        result: List[Tuple[int, Concept]] = []
        for cid, concept in self.cg.concepts.items():
            if concept.status not in (
                ConceptStatus.CRYSTALLIZED,
                ConceptStatus.STABLE,
            ):
                continue
            if getattr(concept, "dormant", False):
                continue
            if getattr(concept, "soft_dormant", False):
                continue
            if getattr(concept, "EF_combined", 0.0) >= theta:
                result.append((cid, concept))
        self._fertile_cache = result
        self._fertile_cache_step = t
        return result

    def _snapshot_fertile_turnover(self) -> None:
        """[Bundle A] fertile-set membership turnover snapshot: bump the
        counter when H's membership changed since the previous snapshot
        (called at every _maybe_recompute_theta_fertile exit, i.e. at
        k_fertile cadence)."""
        # _fertile_concepts returns (cid, Concept) pairs; the per-step
        # cache + k_fertile gate make this re-entrant call safe (inner
        # _maybe_recompute early-returns).
        h_now = frozenset(cid for cid, _c in self._fertile_concepts())
        if self._fertile_set_prev is not None and h_now != self._fertile_set_prev:
            self.fertile_set_turnover_count += 1
        self._fertile_set_prev = h_now

    def _maybe_recompute_theta_fertile(self) -> None:
        """§4.7 dynamic θ_fertile with cold-start rescue and hysteresis.

        Three-state logic:
          TIGHT   — theta = max(rf_min_fertile, percentile_value).
          RELAXED — theta = top_val * (1 - admit_margin). Admits the
                    top concept by construction. No floor. No
                    min(candidate, ...). The rescue exists to admit
                    concepts below rf_min_fertile.
          EMPTY   — no crystallized concepts; theta = rf_min_fertile.

        Hysteresis prevents oscillation: exit RELAXED only when
        top_val >= candidate * (1 + retighten_margin).
        """
        t = self._current_step
        # [Batch D / P1-1 HIGH-03] OR-of-bounds: recompute on step cadence
        # OR on graph drift; both checkpoints update on every fire.
        _grev = (
            self.cg.graph_revision()
            if hasattr(self.cg, "graph_revision")
            else 0
        )
        if (
            t - self._theta_fertile_step < self.cfg.k_fertile
            and self._graph_drift(self._last_fertile_rev)
            < self.cfg.k_fertile_revision
        ):
            return

        vals: List[float] = []
        for _cid, concept in self.cg.concepts.items():
            if concept.status not in (
                ConceptStatus.CRYSTALLIZED,
                ConceptStatus.STABLE,
            ):
                continue
            ef = getattr(concept, "EF_combined", 0.0)
            if math.isfinite(ef):
                vals.append(ef)

        if not vals:
            # No crystallized concepts. Bootstrap regime handles selection.
            self._theta_fertile = self.cfg.rf_min_fertile
            self._fertile_relaxed = False
            self._theta_fertile_step = t
            self._last_fertile_rev = self.cg.graph_revision()
            self.diagnostics["theta_fertile_empty_state_recomputes"] += 1
            self._snapshot_fertile_turnover()
            return

        top_val = max(vals)

        if top_val <= 0.0:
            # All concepts have zero fertility. Nothing to admit.
            self._theta_fertile = self.cfg.rf_min_fertile
            self._fertile_relaxed = False
            self._theta_fertile_step = t
            self._last_fertile_rev = self.cg.graph_revision()
            self.diagnostics["theta_fertile_empty_state_recomputes"] += 1
            self._snapshot_fertile_turnover()
            return

        if len(vals) < self.cfg.fertile_min_population:
            candidate = self.cfg.rf_min_fertile
        else:
            vals.sort()
            k = int(len(vals) * self.cfg.theta_fertile_percentile / 100.0)
            k = max(0, min(len(vals) - 1, k))
            candidate = max(self.cfg.rf_min_fertile, vals[k])

        if not self._fertile_relaxed:
            if candidate > top_val:
                # Cold-start rescue: the percentile threshold excludes
                # every concept. Relax to admit the top one.
                self._fertile_relaxed = True
                self.diagnostics["fertile_rescue_fired"] += 1
                self.diagnostics["fertile_rescue_top_val"] = top_val
                # Reporting threshold only; 0.1 is chosen as roughly the
                # floor of measurable fertility from activation credit.
                if top_val >= 0.1:
                    self.diagnostics["fertile_rescue_fired_meaningful"] += 1
                else:
                    self.diagnostics["fertile_rescue_fired_marginal"] += 1
            else:
                # Normal operation. Stay tight.
                self._theta_fertile = candidate
                self._theta_fertile_step = t
                self._last_fertile_rev = self.cg.graph_revision()
                self._snapshot_fertile_turnover()
                return

        # RELAXED state: check if we can re-tighten.
        if top_val >= candidate * (1.0 + self.cfg.fertile_retighten_margin):
            self._fertile_relaxed = False
            self._theta_fertile = candidate
        else:
            # Stay relaxed. Admit the top concept by construction.
            # No floor. No min(candidate, ...).
            self._theta_fertile = top_val * (1.0 - self.cfg.fertile_admit_margin)
        self._theta_fertile_step = t
        self._last_fertile_rev = self.cg.graph_revision()
        self._snapshot_fertile_turnover()
    def _canonical_node_sort_key(self, nid: int) -> Tuple[bytes, bytes, int]:
        """Canonical-hash ordering of a node (§4.7 τ sampling): the node's
        identity is (lens_type, signature_hash); node ID is the terminal
        discriminator. Pruned/unknown signatures sort last, deterministically."""
        sig = self.cg.get_pattern_node_signature(nid)
        if sig is None:
            return (b"\xff", b"\xff" * 16, nid)
        return (sig[0].encode("utf-8"), sig[1], nid)

    def _compute_tau(self) -> float:
        """§4.7:
            τ = max(τ_min, median({BFS_distance(v,u) : v,u ∈ active_set, v≠u}))
        recomputed every TAU_RECOMPUTE_CADENCE steps, or whenever the active
        set changes by more than TAU_ACTIVE_DRIFT_FRACTION between steps —
        measured as the SYMMETETRIC DIFFERENCE ratio |A_t △ A_{t−1}| /
        max(|A_t|, |A_{t−1}|, 1) (the former size-difference test missed
        equal-size membership churn entirely). When |active_set| >
        TAU_SAMPLE_SIZE, τ is computed from a deterministic sample chosen
        by CANONICAL-HASH ORDERING of the active set (the former
        node-ID stride sample was neither). BFS depth is capped at
        BFS_MAX — pairs beyond the cap are unreachable at decision-relevant
        scale and excluded from the median."""
        t = self._current_step
        active_nodes: Set[int] = set()
        for _cid, concept in self.cg.concepts.items():
            if getattr(concept, "A", 0) != 1:
                continue
            active_nodes.update(concept.members)
        size = len(active_nodes)
        if size < 2:
            self._tau_cache = self.cfg.tau_min
            self._tau_cache_step = t
            self._tau_active_size = size
            self._tau_active_set = frozenset(active_nodes)
            return self._tau_cache

        cadence = t - self._tau_cache_step >= P.TAU_RECOMPUTE_CADENCE
        drift = False
        prev_set = self._tau_active_set
        if prev_set:
            sym_diff = len(active_nodes ^ set(prev_set))
            denom = max(size, len(prev_set), 1)
            if sym_diff / denom > P.TAU_ACTIVE_DRIFT_FRACTION:
                drift = True
        if not (cadence or drift):
            self._tau_active_size = size
            return self._tau_cache

        adj = self._pattern_adjacency()
        ordered_active = sorted(
            active_nodes, key=self._canonical_node_sort_key
        )
        sampled = ordered_active[: self.cfg.tau_sample_size]

        distances: List[float] = []
        for i, src in enumerate(sampled):
            dists = self._bfs_distances(src, adj, max_hops=P.BFS_MAX)
            for j in range(i + 1, len(sampled)):
                dst = sampled[j]
                d = dists.get(dst)
                if d is not None and d > 0:
                    distances.append(float(d))

        if not distances:
            self._tau_cache = self.cfg.tau_min
        else:
            distances.sort()
            mid = len(distances) // 2
            median = (
                (distances[mid - 1] + distances[mid]) / 2.0
                if len(distances) % 2 == 0
                else distances[mid]
            )
            self._tau_cache = max(self.cfg.tau_min, median)
        self._tau_cache_step = t
        self._tau_active_size = size
        self._tau_active_set = frozenset(active_nodes)
        return self._tau_cache

    def _bfs_distances(
        self,
        src: int,
        adj: Dict[int, Set[int]],
        max_hops: int = 20,
    ) -> Dict[int, int]:
        dists: Dict[int, int] = {src: 0}
        frontier = deque([(src, 0)])
        while frontier:
            nid, d = frontier.popleft()
            if d >= max_hops:
                continue
            for nb in adj.get(nid, ()):
                if nb in dists:
                    continue
                dists[nb] = d + 1
                frontier.append((nb, d + 1))
        return dists

    def _sample_reference_v_history(self) -> None:
        """[Bundle A] append (edge_revision, reference_V) to the spine's
        bounded history whenever the edge revision advances; no-op else."""
        spine_id = self.spine_basin_id
        if spine_id is None or spine_id not in self.basins:
            return
        spine = self.basins[spine_id]
        if spine.reference_V_history is None:
            spine.reference_V_history = deque(maxlen=1000)
        edge_rev = getattr(self.cg, "edge_revision", None)
        if edge_rev is None:
            edge_rev = len(self.cg.edges)
        if int(edge_rev) == self._last_edge_rev_sample:
            return
        self._last_edge_rev_sample = int(edge_rev)
        spine.reference_V_history.append((int(edge_rev), spine.reference_V))

    def _graph_revision_signature(self) -> Tuple[int, int, int]:
        """Cache-invalidating signature for pattern-graph topology.

        The former length-only signature (len(edges), len(nodes)) missed
        same-count churn — an edge eviction plus an edge addition in one
        step preserved both lengths and served a STALE adjacency to the
        value function. ConceptGraph maintains `edge_revision` (monotonic
        counter bumped on every edge mutation — the same revision source
        fertility_engine's _is_active_edge memoization uses); when present
        it is authoritative. Node-store revision falls back to the live
        node count when no counter is exposed."""
        edge_rev = getattr(self.cg, "edge_revision", None)
        if edge_rev is None:
            edge_rev = len(self.cg.edges)
        node_rev = getattr(self.cg, "node_revision", None)
        if node_rev is None:
            node_rev = len(self._live_pattern_node_ids())
        return (int(edge_rev), int(node_rev), 0)

    def _pattern_adjacency(self) -> Dict[int, Set[int]]:
        """UNDIRECTED pattern-node adjacency — for F_expected proximity,
        τ computation, and basin connectivity (§4.7 'BFS runs over
        pattern-node-to-pattern-node edges'; §4.9 'connectivity by graph
        edges (any type)')."""
        sig = self._graph_revision_signature()
        if sig == self._pattern_adj_signature and self._pattern_adj_cache:
            return self._pattern_adj_cache
        pattern_node_set = self._live_pattern_node_ids()
        adj: Dict[int, Set[int]] = defaultdict(set)
        for (src, dst, _et), _edge in self.cg.edges.items():
            if src in pattern_node_set and dst in pattern_node_set:
                adj[src].add(dst)
                adj[dst].add(src)
        self._pattern_adj_cache = dict(adj)
        self._pattern_adj_signature = sig
        return self._pattern_adj_cache

    def _pattern_adjacency_directed(self) -> Dict[int, Set[int]]:
        """DIRECTED successor adjacency — exclusively for F_recursive's
        N_D⁺ (§4.7 direction specification)."""
        sig = self._graph_revision_signature()
        if sig == self._pattern_adj_dir_signature and self._pattern_adj_dir_cache:
            return self._pattern_adj_dir_cache
        pattern_node_set = self._live_pattern_node_ids()
        adj: Dict[int, Set[int]] = defaultdict(set)
        for (src, dst, _et), _edge in self.cg.edges.items():
            if src in pattern_node_set and dst in pattern_node_set:
                adj[src].add(dst)
        self._pattern_adj_dir_cache = dict(adj)
        self._pattern_adj_dir_signature = sig
        return self._pattern_adj_dir_cache

    def _proximity(
        self,
        v: int,
        H: List[Tuple[int, Concept]],
        tau: float,
    ) -> float:
        """max over c ∈ H of exp(−dist(v,c)/τ) (§4.7). Unreachable concepts
        contribute 0 (treated non-fertile, not infinitely distant).

        The BFS hop bound is DERIVED, not magic: beyond d hops with
        exp(−d/τ) < θ_prune_frontier, a member's proximity contribution
        cannot lift any candidate above the frontier-pruning floor, so
        deeper exploration cannot change any decision.

        Cached per node per step (H and τ are step-cached)."""
        if not H:
            return 0.0
        t = self._current_step
        if self._prox_cache_step != t:
            self._prox_cache = {}
            self._prox_cache_step = t
        cached = self._prox_cache.get(v)
        if cached is not None:
            return cached

        adj = self._pattern_adjacency()
        hop_bound = int(tau * math.log(1.0 / self.cfg.theta_prune_frontier)) + 1
        dists = self._bfs_distances(v, adj, max_hops=hop_bound)
        best = 0.0
        for _cid, concept in H:
            best_d: Optional[int] = None
            for m in concept.members:
                if m == v:
                    best_d = 0
                    break
                if m in dists:
                    d = dists[m]
                    if best_d is None or d < best_d:
                        best_d = d
            if best_d is None:
                continue
            # §2.1.20: exponentials clamp to [−50, 50].
            val = math.exp(clamp(-best_d / (tau + _EPS), -50.0, 50.0))
            if val > best:
                best = val
        self._prox_cache[v] = best
        return best

    # -------------------------------------------------------------------------
    # §4.15 Crystallization detection & traces
    # -------------------------------------------------------------------------

    def _detect_new_crystallizations(self, t: int, s_t: Optional[int] = None) -> Set[bytes]:
        current = self._collect_crystallized_hashes()
        new = current - self.crystallized_snapshot
        if new:
            self.diagnostics["crystallized_total"] += len(new)
            # [Batch D / P1-1 MED-05] Spine-basin crystallization =
            # progress. Checked BEFORE the frontier is filtered below.
            if self.spine_basin_id is not None:
                for _c in self.frontier:
                    if (
                        _c.canonical_hash in new
                        and _c.basin_id == self.spine_basin_id
                    ):
                        self._mark_spine_progress(t)
                        break

            # Capture basin_id from the candidate before removal.
            basin_by_hash: Dict[bytes, int] = {}
            removed = [c for c in self.frontier if c.canonical_hash in new]
            for c in removed:
                basin_by_hash[c.canonical_hash] = c.basin_id

            self.frontier = [c for c in self.frontier if c.canonical_hash not in new]
            for c in removed:
                # The hash departs the frontier as a CRYSTALLIZED subgraph:
                # its candidate-node registration and index entries are
                # replaced by the concept registrations below, but its
                # basin membership is retained (§4.3.2 basins hold
                # crystallized subgraphs).
                self.frontier_hashes.discard(c.canonical_hash)
                self._frontier_by_hash.pop(c.canonical_hash, None)
                self.cg.unregister_candidate_nodes(c.canonical_hash, c.members)
                self._remove_from_subgraph_index(c.canonical_hash)

            # Index new concept members.
            for h in new:
                concept = self.cg.lookup_by_canonical_hash(h)
                if concept is None:
                    continue
                self._subgraph_members_cache[h] = concept.members
                for nid in concept.members:
                    self._subgraph_index[nid].add(h)
                self._mark_dirty_members(concept.members)

            # Open traces (skip if already opened for this path hash).
            for h in new:
                if h in self._opened_traces:
                    continue
                # Basin membership may not be computed yet for pipeline-path
                # crystallizations (basins recompute at step 3, after this
                # detection); _update_traces resolves basin_id == -1 lazily
                # once the recomputation has run.
                basin_id = basin_by_hash.get(h, self._basin_of_hash(h))
                self.crystallization_traces.append(
                    CrystallizationTrace(
                        path_hash=h,
                        basin_id=basin_id,
                        t_crystallized=t,
                        s_t=s_t,
                    )
                )
                self._opened_traces.add(h)
                # §4.15.3: "If P crystallizes again in a new episode, the
                # counter starts over." A fresh trace for path P restarts
                # every (P, ·) trial counter.
                for key in [k for k in self.pair_trial_counts if k[0] == h]:
                    self.pair_trial_counts[key] = 0
                if basin_id == self.spine_basin_id and self.spine_basin_id is not None:
                    self._last_spine_crystallization_step = t
        return new

    def _collect_crystallized_hashes(self) -> FrozenSet[bytes]:
        result: Set[bytes] = set()
        for _cid, concept in self.cg.concepts.items():
            if concept.status not in (
                ConceptStatus.CRYSTALLIZED,
                ConceptStatus.STABLE,
            ):
                continue
            ch = getattr(concept, "canonical_hash", b"")
            if ch:
                result.add(ch)
        return frozenset(result)

    def _update_traces(
        self,
        t: int,
        newly_crystallized: Set[bytes],
        s_t: Optional[int] = None,
    ) -> None:
        for trace in self.crystallization_traces:
            if trace.best_successor_hash is not None:
                continue
            # Never pair a trace with a successor that crystallized at the
            # same step the trace was opened.
            if trace.t_crystallized >= t:
                continue
            if t - trace.t_crystallized > self.cfg.w_successor:
                continue
            # Lazy basin resolution: pipeline-path crystallizations open
            # traces with basin_id == -1 (basins recompute after detection);
            # resolve from the reverse index once membership exists.
            if trace.basin_id == -1:
                resolved = self._basin_of_hash(trace.path_hash)
                if resolved != -1:
                    trace.basin_id = resolved
            if trace.basin_id == -1:
                continue
            for h in sorted(newly_crystallized):
                if h == trace.path_hash:
                    continue
                basin_id = self._basin_of_hash(h)
                # Unassigned (-1) is NOT a basin: the former equality test
                # paired unrelated pipeline-path crystallizations with
                # basin-less traces (-1 == -1), fabricating path/outcome
                # pairs across disconnected regions.
                if basin_id == -1 or basin_id != trace.basin_id:
                    continue
                concept = self.cg.lookup_by_canonical_hash(h)
                if concept is None:
                    continue
                v = self._concept_V_via_builder(concept, t)
                if v > trace.best_successor_V:
                    trace.best_successor_hash = h
                    trace.best_successor_V = v
                    trace.successor_s_t = s_t

    def _concept_V_via_builder(self, concept: Concept, t: int) -> float:
        """[SPEC COMPLIANCE §4.15.2] Compute V(S) for a crystallized concept
        by reconstructing a temporary CandidateSubgraph and calling the
        Builder's value function (not a surrogate).
        """
        members = concept.members
        edges = self._induced_edges(members)
        temp = CandidateSubgraph(
            canonical_hash=concept.canonical_hash,
            cross_section_key=concept.cross_section_key,
            members=members,
            edges=edges,
            anchors=(),
            contexts=frozenset(),
            t_created=concept.T_cryst,
            T_first=concept.T_cryst,
            T_last=t,
            parent_hash=None,
            construction_op="SEED",
            original_detections=frozenset(),
            original_detection_hashes=frozenset(),
            n_conf=getattr(concept, "N_conf", 0),
            last_n_conf_step=-1,
        )
        return self._compute_V(temp, t)

    def _resolve_traces(
        self, t: int
    ) -> Tuple[List[Tuple[bytes, bytes]], List[ConsolidationHandoff]]:
        """[SPEC COMPLIANCE §4.15.3] Resolve expired traces and return the
        (P_hash, S_hash) pairs that reached the emission threshold this
        step, PLUS the B1/D6 ConsolidationHandoffs packaging the same
        events with induced subgraphs and crystallization states.

        The emission happens here, not in step(), because expired traces are
        removed from `crystallization_traces` in this method. Any downstream
        iteration over `crystallization_traces` cannot see them.
        """
        survivors: List[CrystallizationTrace] = []
        resolved_pairs: List[Tuple[bytes, bytes]] = []
        handoffs: List[ConsolidationHandoff] = []
        for trace in self.crystallization_traces:
            if t - trace.t_crystallized > self.cfg.w_successor:
                if trace.best_successor_hash is not None:
                    key = (trace.path_hash, trace.best_successor_hash)
                    self.pair_trial_counts[key] += 1
                    if self.pair_trial_counts[key] >= self.cfg.theta_pair_trials:
                        resolved_pairs.append(key)
                        self.pair_trial_counts[key] = 0
                        self.diagnostics["pairs_emitted_total"] += 1
                        handoff = self._make_handoff(trace, t)
                        if handoff is not None:
                            handoffs.append(handoff)
                            self.diagnostics["handoffs_emitted"] += 1
                        else:
                            self.diagnostics["handoffs_unresolvable"] += 1
                continue
            survivors.append(trace)
        self.crystallization_traces = survivors
        return resolved_pairs, handoffs

    def _make_handoff(
        self, trace: CrystallizationTrace, t: int
    ) -> Optional[ConsolidationHandoff]:
        """Package a resolved trace into a ConsolidationHandoff (B1/D6).

        Subgraphs are INDUCED FROM THE LIVE GRAPH at emission time
        (single source of truth — the draft's plan of storing member/edge
        snapshots on the trace would carry stale structure across the
        θ_pair_trials accumulation window). None when either concept has
        since left the graph; the hash pair is still emitted so the
        pipeline's reconstruction path can count the loss observably.
        """
        p_sg = self._subgraph_of_hash(trace.path_hash)
        s_sg = self._subgraph_of_hash(trace.best_successor_hash)
        if p_sg is None or s_sg is None:
            return None
        return ConsolidationHandoff(
            P_subgraph=p_sg,
            S_subgraph=s_sg,
            s_at_path=trace.s_t,
            s_at_successor=trace.successor_s_t,
            t=t,
            source_hashes=(trace.path_hash, trace.best_successor_hash or b""),
        )

    def _subgraph_of_hash(self, canonical_hash: Optional[bytes]) -> Optional[Subgraph]:
        """Induce the Subgraph of a crystallized concept from the live
        ConceptGraph (members + induced edges)."""
        if not canonical_hash:
            return None
        concept = self.cg.lookup_by_canonical_hash(canonical_hash)
        if concept is None:
            return None
        members = frozenset(concept.members)
        if not members:
            return None
        return Subgraph(
            nodes=members,
            edges=self._induced_edges(members),
            anchor_of_node=anchor_positions_for(
                members, self.cg.pattern_node_position
            ),
        )

    # -------------------------------------------------------------------------
    # §4.17 Crystallization predicate
    # -------------------------------------------------------------------------

    def _collect_eligible_for_crystallization(
        self,
        t: int,
    ) -> List[CandidateSubgraph]:
        eligible: List[CandidateSubgraph] = []
        scaffold_population = 0
        for c in self.frontier:
            if not _is_family_forming(c):
                # §4.14: diagnostics.uncrystallizable_scaffold_count tracks
                # the POPULATION of scaffold candidates resident in the
                # frontier (expandable, not crystallization-eligible).
                scaffold_population += 1
                continue
            if self._crystallization_predicate(c, t):
                eligible.append(c)
        self.diagnostics["uncrystallizable_scaffold_count"] = scaffold_population
        return eligible

    def _candidate_family(self, cand: CandidateSubgraph) -> Optional[ConceptFamily]:
        """Family resolution via the Binder's _map_family (§4.14), cached
        per candidate. The cache is keyed by the detection-hash count:
        original_detection_hashes only grows (seed/extension/merge), so a
        count change is the exact invalidation signal (a family can flip
        from None to resolved as detections accumulate)."""
        n = len(cand.original_detection_hashes)
        if cand.family_cache_n == n:
            fam = cand.family_cache  # includes cached None (unresolved yet)
        else:
            fam = _map_family(
                cand.original_detections,
                cand.edges,
                diag=self.cg.diagnostics,
            )
            self.cg.diagnostics["map_family_checked"] = (
                self.cg.diagnostics.get("map_family_checked", 0) + 1
            )
            cand.family_cache = fam
            cand.family_cache_n = n
        # [Batch 4 / Phase 3] CDP predicate probe. Per-hash max values,
        # dumped at end. Bounded by unique CDP candidate hashes.
        if (
            self._cdp_predicate_probe_enabled
            and fam is not None
            and fam.name == "CROSS_DOMAIN_PATTERN"
            and cand.canonical_hash is not None
        ):
            h = cand.canonical_hash.hex()
            rec = self._cdp_predicate_max.setdefault(h, {
                "first_t": self._current_step,
                "max_N_opp": 0,
                "max_N_conf": 0,
                "max_span": 0,
                "max_N_anchor": 0,
                "N_opp_threshold": self.cfg.theta_opp_subgraph,
                "N_conf_threshold": self.cfg.theta_conf_subgraph,
                "span_threshold": self.cfg.theta_span_subgraph,
                "N_anchor_threshold": None,
            })
            if len(cand.contexts) > rec["max_N_opp"]:
                rec["max_N_opp"] = len(cand.contexts)
            if cand.n_conf > rec["max_N_conf"]:
                rec["max_N_conf"] = cand.n_conf
            span_now = cand.T_last - cand.T_first
            if span_now > rec["max_span"]:
                rec["max_span"] = span_now
            if len(cand.anchors) > rec["max_N_anchor"]:
                rec["max_N_anchor"] = len(cand.anchors)
            # Mirror the actual anchor threshold from the crystallization
            # predicate (builder._crystallization_predicate inline
            # conditional on _SINGLE_ANCHOR_FAMILIES; CDP is not in that
            # set so the module-level DEFAULT threshold applies, exactly
            # as the predicate computes it).
            if rec["N_anchor_threshold"] is None:
                rec["N_anchor_threshold"] = (
                    THETA_ANCHOR_SUBGRAPH_SINGLE
                    if fam.name in _SINGLE_ANCHOR_FAMILIES
                    else THETA_ANCHOR_SUBGRAPH_DEFAULT
                )
        return fam

    def _crystallization_predicate(
        self,
        cand: CandidateSubgraph,
        t: int,
    ) -> bool:
        # §4.17: IsFamilyForming ∧ N_opp ≥ θ_opp ∧ N_conf ≥ θ_conf
        #        ∧ N_anchor ≥ θ_anchor(family) ∧ span ≥ θ_span ∧ ¬Tombstone.
        # IsFamilyForming and the emission restriction are ONE gate (§4.17):
        # scaffolds fail here and are never emitted as CrossSections.
        if not _is_family_forming(cand):
            return False
        n_opp = len(cand.contexts)
        if n_opp < self.cfg.theta_opp_subgraph:
            return False
        if cand.n_conf < self.cfg.theta_conf_subgraph:
            return False
        family = self._candidate_family(cand)
        if family is None:
            return False
        anchor_thresh = (
            THETA_ANCHOR_SUBGRAPH_SINGLE
            if family.name in _SINGLE_ANCHOR_FAMILIES
            else THETA_ANCHOR_SUBGRAPH_DEFAULT
        )
        if len(cand.anchors) < anchor_thresh:
            return False
        span = cand.T_last - cand.T_first
        if span < self.cfg.theta_span_subgraph:
            return False
        if self._is_tombstoned(cand.canonical_hash):
            return False
        return True

    def _is_tombstoned(self, canonical_hash: bytes) -> bool:
        # Rebuild tombstone cache once per step.
        t = self._current_step
        if self._tombstone_cache_step != t:
            self._tombstone_cache = {}
            for entry in self.cg.tombstones:
                if entry.expiry > self.cg.t:
                    # Keep the latest expiry if duplicated.
                    prev = self._tombstone_cache.get(entry.signature_hash, -1)
                    if entry.expiry > prev:
                        self._tombstone_cache[entry.signature_hash] = entry.expiry
            self._tombstone_cache_step = t
        return canonical_hash in self._tombstone_cache

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset Builder-owned search state. The concept graph, its
        canonical-hash index, and the transposition table PERSIST (the
        graph persists across episode boundaries per v30 §4.3; the
        transposition table reflects the persistent graph's expansion
        history). crystallized_snapshot persists with the graph."""
        self.frontier = []
        self.frontier_hashes = set()
        self._frontier_by_hash = {}
        self._wl_memo.clear()
        self._induced_memo.clear()
        self._induced_memo_step = -1
        self._footprint_memo.clear()
        self._footprint_memo_step = -1
        self.crystallization_traces = []
        self.pair_trial_counts = defaultdict(int)
        self.basins = {}
        self.next_basin_id = 0
        self.spine_basin_id = None
        self.last_basin_recompute_step = -K_MAIN
        self.last_spine_recompute_step = -K_MAIN
        self._spine_reassign_deferred = False
        self._hash_to_basin = {}
        self._dirty_basin_nodes = set()
        self._subgraph_index = defaultdict(set)
        self._subgraph_members_cache = {}
        self.mode = "SEEDING"
        self.stall_counter = 0
        self.t_excursion_start = -1
        self.t_epistemic_start = -1
        self.excursion_basin_id = None
        self._excursion_entry_concepts = frozenset()
        self._excursion_sterile_counter = 0
        self._last_spine_crystallization_step = -1
        self.current_scope_members = frozenset()
        self.current_scope_expanded = set()
        self.current_surprise_value = 0.0
        self.surprise_queue.clear()
        # [Batch D / P1-1] checkpoint reset mirrors __init__.
        _g0 = (
            self.cg.graph_revision()
            if hasattr(self.cg, "graph_revision")
            else 0
        )
        self._spine_last_progress_t = -1
        self._spine_last_progress_rev = _g0
        self._excursion_last_progress_t = -1
        self._excursion_last_progress_rev = _g0
        self._excursion_v_max_seen = -1.0
        self._epistemic_last_progress_t = -1
        self._epistemic_last_progress_rev = _g0
        self._last_basin_rev = _g0
        self._last_fertile_rev = _g0
        self._theta_fertile = RF_MIN_FERTILE
        self._theta_fertile_step = -(10**9)
        self._fertile_relaxed = False
        self._pattern_adj_cache = {}
        self._pattern_adj_signature = (-1, -1, -1)
        self._pattern_adj_dir_cache = {}
        self._pattern_adj_dir_signature = (-1, -1, -1)
        self._reach_cache = {}
        self._reach_cache_step = -1
        self._prox_cache = {}
        self._prox_cache_step = -1
        self._tau_cache = TAU_MIN
        self._tau_cache_step = -(10**9)
        self._tau_active_size = 0
        self._tau_active_set = frozenset()
        self._fertile_cache = []
        self._fertile_cache_step = -1
        self._tombstone_cache = {}
        self._tombstone_cache_step = -1
        self._seeding_start = None
        self._opened_traces = set()
        self._step_detection_map = {}
