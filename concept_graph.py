"""
MODULE 4: ConceptGraph — v31 (final)

Central structural repository. Owns:
  - The pattern-node store (§1.11)
  - The edge store keyed by (parent, child, EdgeType) (§1.12) with
    maintained forward and reverse adjacency lists for O(deg) traversal
  - Candidate management and crystallization (§4.17)
  - Witness evaluation, status transitions, DAG guard
  - Capacity, dormancy, pruning, operator promotion
  - §4.20 (family, cross_section_key) and canonical_hash indexes

Does NOT own:
  - Causal/counterfactual attribution (moves to prediction.py per §8.9;
    retained here as [TEMP: MIGRATION] until prediction is wired).
  - Builder/Consolidator/Mapper/Executor state.
  - Detection or TypedSignature definitions (owned by sensory_floor).

Public interface for binder.py (§1.11, §1.12):
    get_or_create_pattern_node(lens_type, signature_hash, detection_hash=None) -> int
    get_pattern_node_signature(nid) -> Optional[Tuple[str, bytes]]
    get_pattern_node_detection_hash(nid) -> Optional[bytes]
    upsert_edge(src, dst, edge_type, signal, t, lag=0) -> None
    get_edge(src, dst, edge_type=None) -> Optional[Edge]
    get_outgoing_edges(src, edge_type=None) -> List[Edge]
    get_incoming_edges(dst, edge_type=None) -> List[Edge]
    iter_edges() -> Iterator[Edge]

Public interface for builder.py (§1.11):
    register_candidate_nodes(canonical_hash, node_ids) -> None
    unregister_candidate_nodes(canonical_hash, node_ids) -> None

Determinism: every iteration used for decisions is sorted or keyed by
deterministic integer IDs.
"""

from __future__ import annotations

import math
from typing import (
    AbstractSet,
    Any,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict

from core.hashing import (  # noqa: F401
    clamp,
    hash_128,
    hash_64,
    heaviside,
    serialize_bytes,
    serialize_float,
    serialize_int,
    serialize_set,
    serialize_tuple,
    serialize_value,
)
from core.registries import (  # noqa: F401
    ACTION_NOOP,
    ACTION_NULL,
    is_noop_action,
    is_probe_action,
    is_teleological_action,
)
from core.schema import (  # noqa: F401
    EdgeParams,
    EdgeParamsMixture,
)
import parameters as P

from sensory_floor import TypedSignature, Detection
from core.registries import SELF_BINDER_TYPES
from core.hashing import _node_signature_hash
from core.registries import edge_type_layer
from core.types import (
    EdgeType,
    HEBBIAN_BOOST_TYPES,
)

# =============================================================================
# CONSTANTS — aliased from parameters for caller compatibility
# =============================================================================

EPSILON = P.EPSILON

THETA_OPP = P.THETA_OPP
THETA_CONF = P.THETA_CONF_DEFAULT
THETA_CONF_MIN = P.THETA_CONF_MIN
THETA_CONF_MAX = P.THETA_CONF_MAX
THETA_ANCHOR = P.THETA_ANCHOR_DEFAULT
THETA_SPAN = P.THETA_SPAN
THETA_SEQ = 2
THETA_STRUCT = 2
ARITY_MAX = 3
TAU_FORGET = 1000
TAU_TOMBSTONE = 1000

THETA_WITNESS = P.THETA_WITNESS
THETA_DECISIVE = P.THETA_DECISIVE
ETA_ARITY = P.ETA_ARITY
RHO_SYM_MAX = P.RHO_SYM_MAX
N_MAX_AMBIGUITY = P.N_MAX_AMBIGUITY
N_LN = P.N_LN
THETA_LN = P.THETA_LN
TAU_AMBIGUITY = P.TAU_AMBIGUITY

EPS_INITIAL = P.EPS_INITIAL
C_ENABLEMENT = P.C_ENABLEMENT
KAPPA_SPEC = P.KAPPA_SPEC
C_MAX = P.C_MAX

W_INIT = P.W_INIT
W_MAX = P.W_MAX
BETA_W = P.BETA_W
DELTA_W = P.DELTA_W
ETA_W = P.ETA_W
THETA_EDGE_ACTIVE = P.THETA_EDGE_ACTIVE
THETA_EDGE_PRUNE = 0.01
TAU_EDGE_INACTIVE = P.TAU_EDGE_INACTIVE

# [Phase 1 Decision 1] Three-signal pruning thresholds.
THETA_A = P.THETA_A
THETA_RF = P.THETA_RF
THETA_FERTILE_PROTECT = P.THETA_FERTILE_PROTECT
THETA_PRUNE_SPECIFICITY = P.THETA_PRUNE_SPECIFICITY
N_CAND_MAX = P.N_CAND_MAX
N_LAYER_MAX = P.N_LAYER_MAX
N_NODE_MAX = P.N_NODE_MAX
TAU_DORMANT = 1000
TAU_SOFT_DORMANT = 2000
RF_SOFT_MIN = 1.0
THETA_BLOCK_MAX = 10
N_TOMB_MAX = P.N_TOMB_MAX


BFS_MAX = P.BFS_MAX
D_MAX = P.D_MAX

GAIN_ETA_W = 0.01
GAIN_TAU_DORMANT = 0.01
DELTA_PI_MAX = 0.01

TAU_PROMO = 3
N_OP_MAX = 20
TAU_OP_INACTIVE = 5000
TAU_OP_COOLDOWN = 200
THETA_PROMO_OP = 0.5
THETA_HYST_OP = 0.5

KAPPA_BIRTH = 0.05
N_CHUNK_MAX = 500
THETA_CONF_CHUNK = THETA_CONF
THETA_ANCHOR_CHUNK = 2
THETA_LENS_CHUNK = 2

TAU_REVERT = P.TAU_REVERT

# Cap on pattern-node tombstone set. Pruned-node keys are only meaningful
# while the same lens/signature might be re-observed; older entries are
# historical noise that would grow without bound.
_PATTERN_TOMBSTONE_CAP = N_TOMB_MAX * 2


# =============================================================================
# ENUMS
# =============================================================================


class ConceptStatus(Enum):
    PENDING = auto()
    CRYSTALLIZED = auto()
    STABLE = auto()
    AMBIGUOUS = auto()
    REJECTED = auto()
    TOMBSTONED = auto()


class ConceptFamily(Enum):
    """§11.1: STRUCTURAL_ROLE and OBJECT_ROLE removed. §11.2:
    EMERGENT_COMPOSITION removed (Operator Emission is a mechanism, not a
    family). Remaining families describe subgraph shapes."""

    SPATIAL_OBJECT = auto()
    EVENT_CAUSAL = auto()
    TEMPORAL_CHAIN = auto()
    RECURSIVE_COMPOSITION = auto()
    CROSS_DOMAIN_PATTERN = auto()
    CHUNK = auto()
    ATOMIC_ACTION = auto()

    @property
    def is_causal(self) -> bool:
        return self in (ConceptFamily.EVENT_CAUSAL, ConceptFamily.TEMPORAL_CHAIN)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class CrossSection:
    """Pipeline crystallization input. key is typed as Any for v30 compat;
    §4.14 v31 callers pass bytes. ConceptGraph normalises to bytes via
    _to_bytes before indexing."""

    family: str
    key: Any
    anchor: Any
    detections: FrozenSet[Detection]
    roleDescriptors: Tuple[Tuple, ...]
    sourceLayers: FrozenSet[int]


@dataclass
class Candidate:
    K_C: tuple
    family: ConceptFamily
    anchorKind: str
    combinationType: str
    N_opp: int = 0
    N_conf: int = 0
    N_conf_by_lens: Dict[str, int] = field(default_factory=dict)
    N_anchor: int = 0
    T_first: int = -1
    T_last: int = -1
    Anchors: Set[Any] = field(default_factory=set)
    status: ConceptStatus = ConceptStatus.PENDING
    blockedCount: int = 0
    last_cross_section: Optional[CrossSection] = None
    concept_id: Optional[int] = None
    enabling_pre_states: Set[Any] = field(default_factory=set)
    triggering_action: Optional[Any] = None
    # §4.3.1 — populated by Builder or derived for v30 candidates.
    canonical_hash: Optional[bytes] = None
    cross_section_key: Optional[bytes] = None
    # §1.11 — pattern nodes this candidate references, cached for O(members)
    # de-indexing.
    node_ids: FrozenSet[int] = field(default_factory=frozenset)
    # Builder-only: original detections & their hashes.
    original_detections: FrozenSet[Detection] = field(default_factory=frozenset)
    original_detection_hashes: FrozenSet[bytes] = field(default_factory=frozenset)
    contexts: FrozenSet[bytes] = field(default_factory=frozenset)


@dataclass
class Concept:
    id: int
    family: ConceptFamily
    layer: int
    sigma: TypedSignature
    status: ConceptStatus
    evidence_registry: List[Tuple] = field(default_factory=list)
    credit: float = 0.25
    fertility: float = 0.0
    parent_ids: Set[int] = field(default_factory=set)
    child_ids: Set[int] = field(default_factory=set)
    chain_links: List[int] = field(default_factory=list)
    T_cryst: int = 0
    N_conf: int = 0
    N_opp: int = 0
    N_anchor: int = 0
    Reliability: float = 0.0
    promoted: bool = False
    op_demoted_at: int = -1
    promo_count: int = 0
    inactive_L_P: int = 0
    operator_fatigue: float = 0.0
    operator_contextual_modulation: float = 0.0
    operator_threshold: float = 0.0
    operator_active: bool = False
    fail_streak: int = 0
    # [Batch D / P1-4] consecutive plan-failure blame streak (capped at
    # TAU_REVERT; reset on any structural-progress event).
    plan_blame_streak: int = 0
    last_structural_progress: int = 0
    chunk_members: Set[int] = field(default_factory=set)
    chunk_depth: int = 0
    adapt_P: float = 0.0
    last_activation: int = 0
    dormant: bool = False
    soft_dormant: bool = False
    exhausted: int = 0
    inactive: int = 0
    N_sym: int = 0
    N_conf_LN: int = 0
    N_opp_LN: int = 0
    ambiguityCount: int = 0
    role_descriptors: Tuple[Tuple, ...] = field(default_factory=tuple)
    provisional: bool = False
    CF: Dict = field(default_factory=dict)
    Anchors: Set[Any] = field(default_factory=set)
    EF: float = 0.0
    EF_eff: float = 0.0
    EF_combined: float = 0.0
    RF: float = 0.0
    GP: float = 1.0
    A: int = 0
    onset: int = 0
    onset_time: int = -1
    A_prev: int = 0
    # §1.12: TRANSITION edge signal reads parent's onset at t-1.
    onset_prev: int = 0
    step_F_count: int = 0
    step_S_count: int = 0
    elig: float = 0.0
    action: Any = None
    effect_id: Optional[Any] = None
    # [SCHEMA MULTI-BINDING §1.11] Under anchor-invariance, one causal
    # schema accumulates one (action, effect) binding per anchor instance
    # (e.g. toggle_switch_0→door_0 and toggle_switch_2→door_1 share the
    # same canonical_hash). The scalar action/effect_id above record the
    # first-seen binding; these maps hold the full set so that
    # _resolve_goal_action can select the binding valid in the current
    # context.
    actions: Set[Any] = field(default_factory=set)
    effect_ids_by_action: Dict[Any, Any] = field(default_factory=dict)
    # [Batch D1] Peak N_cf = min(C_opp, C_opp_not) observed while each
    # (action -> effect) binding was live. N_cf decays at gamma_cf
    # between executions, so the CURRENT value understates the evidence
    # the binding was committed on; the utility engine's anti-undo and
    # goal-satisfied gates read this peak instead of the live counter.
    # Written by Pipeline._update_binding_peaks_and_expiry; entries are
    # dropped when the corresponding binding expires.
    effect_confidence: Dict[Any, float] = field(default_factory=dict)
    effect_is_concept: bool = False
    enabling_pre_states: Optional[Set[Any]] = None
    anchor: Any = None
    # [Seam 2] Dual grounding for causal concepts: the TRIGGER entity
    # anchor (physical, positioned — e.g. the switch) alongside the
    # effect atom (often an unpositioned flag, e.g. door_open at
    # (-1,-1)). Written at binding-commit time by the Pipeline; read by
    # UtilityEngine._resolve_goal_action's anchor-approach fallback,
    # which prefers whichever candidate has a real position.
    trigger_anchor: Any = None
    arity: int = 1
    supportTrace: float = 0.0
    M_opp: float = 0.0
    M_conf: float = 0.0
    # §1.11 pattern-node membership.
    members: FrozenSet[int] = field(default_factory=frozenset)
    # §4.17 canonical identity (128-bit bytes).
    canonical_hash: bytes = b""
    # §4.20 (family, cross_section_key) routing key.
    cross_section_key: bytes = b""
    # [Phase 5 provenance + Batch B B5] Track N_conf attribution by source.
    # n_conf_from_observation = count incremented in step_21 (observation path).
    # n_conf_from_reasoning = count incremented in Mapper._write_back (reasoning path).
    # Invariant: n_conf_from_observation + n_conf_from_reasoning == N_conf for live concepts.
    n_conf_from_observation: int = 0
    n_conf_from_reasoning: int = 0
    # [Phase 5] Step at which this concept's N_conf was last incremented by
    # the Mapper's reasoning-pass write-back. Guards against double-count
    # when query_action and query_category both fire in the same step.
    last_witness_update: int = -1
    # [V9 Track 2 - B.11 / B.20] Union of observation anchors this
    # concept's evidence has been seen from. Declared as a field (rather
    # than the draft's bare __init__ attribute) so every construction
    # path carries it; compare=False keeps `Concept.__eq__` exactly as it
    # was -- the same reason the Subgraph occupancy fields are excluded.
    # An empty set means "not tracked yet", not "no anchors".
    persistent_envelope: FrozenSet[Any] = field(
        default_factory=frozenset, compare=False
    )

    def update_persistent_envelope(self, anchors) -> None:
        """[V9 Track 2 - B.11, docstring per B.20]

        Persist the union of anchors observed in this concept's evidence.
        Single-threaded per step: concurrent calls corrupt state, so
        ConceptGraph mutations must be serialized by the caller.
        """
        new = self.persistent_envelope | frozenset(anchors)
        object.__setattr__(self, "persistent_envelope", new)


@dataclass
class Edge:
    """§1.12 typed relation. edge_type distinguishes the channel; lag is
    meaningful only for SEQUENCE."""

    parent: int
    child: int
    weight: float
    edge_type: EdgeType
    lag: int = 0
    inactive_time: int = 0
    CoAct_EMA: float = 0.0
    # [Batch E] Steps since endpoints last carried positive co-activation
    # signal. Default 10**9 encodes "never observed". With the pattern-edge
    # branch of step_28_update_edges now writing this field (SC-12 parity
    # with the concept-edge branch), default 0 would falsely satisfy
    # mapper._edge_included clause (3) for the first TAU_SITUATION_COACT
    # steps of every fresh pattern edge.
    time_since_coact: int = 10**9
    # Signal recorded by the last upsert (binder path). Read once by
    # step_28_update_edges at observed_at == t.
    pending_signal: float = 0.0
    observed_at: int = -1
    initialized: bool = False
    # [Hebbian] Per-edge crystallization-evidence EMA. Bumped toward 1.0
    # when the edge is stamped by a crystallization in the current step;
    # decays at BETA_E every other step. Feeds the boost factor
    # (1 + ETA_E · e) applied to the firing signal in
    # step_28_update_edges for edge types in HEBBIAN_BOOST_TYPES.
    # compare=False: not part of structural identity; prevents the new
    # field from participating in dataclass __eq__ if any diagnostic
    # or test compares edges directly.
    structural_evidence: float = field(default=0.0, compare=False)
    # [C1] Per-edge structural arrangement container (EdgeParams from prediction.py).
    # compare=False: not part of structural equality comparison.
    params: Optional[Any] = field(default=None, compare=False)


@dataclass
class TombstoneEntry:
    """§4.17: keyed by the 128-bit WL canonical hash of the rejected
    candidate (bytes)."""

    signature_hash: bytes
    expiry: int


# =============================================================================
# CONCEPT GRAPH
# =============================================================================


class ConceptGraph:
    def __init__(self, params: Optional[dict] = None):
        # ---- Core stores -------------------------------------------------
        self.concepts: Dict[int, Concept] = {}
        self.candidates: Dict[tuple, Candidate] = {}
        # §1.12 edges keyed by (parent, child, EdgeType).
        self.edges: Dict[Tuple[int, int, EdgeType], Edge] = {}
        # Forward and reverse adjacency for O(deg) traversals.
        # _outgoing[src] = set of (dst, edge_type) pairs.
        # _incoming[dst] = set of (src, edge_type) pairs.
        # [DETERMINISM PERF] sorted adjacency cache: the canonical order
        # (other, edge_type.value) is weight-independent, so it only changes
        # with topology. _topo_version bumps on every edge create/remove;
        # readers rebuild at most once per topology change instead of
        # sorting on every call (the per-call sort doubled rooms ms/step).
        self._topo_version = 0
        self._out_cache: Dict[int, Tuple[int, List]] = {}
        self._in_cache: Dict[int, Tuple[int, List]] = {}
        self._outgoing: Dict[int, Set[Tuple[int, EdgeType]]] = defaultdict(set)
        self._incoming: Dict[int, Set[Tuple[int, EdgeType]]] = defaultdict(set)
        self.tombstones: List[TombstoneEntry] = []
        self.next_concept_id = 0
        self.t = 0

        # Monotonic counter bumped once per edge-topology mutation pass
        # (step_28_update_edges). Downstream engines (FertilityEngine's
        # active-edge memo) key their per-step caches on this so they
        # invalidate by construction instead of guessing when edges changed.
        self.edge_revision: int = 0
        # [Bundle A] monotonic pattern-node store revision: bumped on node
        # add and node remove ONLY (builder's _graph_revision_signature
        # getattr-anticipated this counter; same-count node churn now
        # invalidates adjacency caches precisely).
        self.node_revision: int = 0
        # [Batch D / P1-1] Endogenous revision counters (never reset).
        # topology_revision: every edge create/remove, concept-edge
        # instantiation, and node eviction. concept_revision: every
        # crystallization and eviction. graph_revision() aggregates
        # topology_revision with node_revision for cadence consumers.
        self.topology_revision: int = 0
        self.concept_revision: int = 0

        self.last_state_hash: Optional[int] = None
        self.current_state_hash: Optional[int] = None
        self.entity_atom_map: Dict[Any, Set[int]] = {}
        self.birth_ema: Dict[int, float] = {}
        self.action_to_concepts: Dict[Any, Set[int]] = defaultdict(set)

        # ---- §1.11 pattern-node store ------------------------------------
        self._pattern_node_id_by_key: Dict[Tuple[str, bytes], int] = {}
        self._pattern_node_key_by_id: Dict[int, Tuple[str, bytes]] = {}
        self._pattern_node_detection_hash: Dict[int, bytes] = {}
        self._pattern_node_last_seen: Dict[int, int] = {}
        # [v31 Stage D1] Full TypedSignature per node, for the MCS-based
        # signature_provider consumed by Consolidator/Mapper/Executor.
        # §1.11 identity remains (lens_type, signature_hash); this field
        # is auxiliary and used only by graded-similarity consumers.
        self._pattern_node_signature: Dict[int, TypedSignature] = {}
        # [B2/§5.3.4] Action-vocabulary registry: pattern-node id → the
        # EXACT action object. Written by the Pipeline at execution time
        # (Action-lens pre-registration path); read by the action decoders
        # (pipeline_support.decode_action_sequence) for §5.3.4 replay and
        # Executor chunk expansion. Values are the ORIGINAL objects —
        # decode returns the stored object directly (K1: no
        # re-serialization, no string inference; the env's action type
        # identity is preserved by construction). Cleaned in
        # _prune_pattern_nodes alongside the other per-node stores.
        self._pattern_node_action_alpha: Dict[int, Any] = {}
        # 64-bit signature collision counter (distinct actions mapping to
        # one node). Practically impossible (≤ 2⁻⁶⁴); observable, not
        # silent — a collision would corrupt vocabulary recovery.
        self.action_alpha_collisions: int = 0
        # [V9 Track 2 - B.26] node_id -> the observation anchor the
        # pattern node was FIRST bound from, and that anchor's (x, y)
        # position. Written by the Binder where node ids are minted;
        # read by the B.26 population step that fills
        # Subgraph.anchor_of_node. Inert until then.
        #
        # These live on the GRAPH, not on the Binder, for lifetime
        # reasons rather than taste:
        #   * reachable -- every module that builds a Subgraph already
        #     holds a ConceptGraph (Consolidator/Mapper/Builder/
        #     PipelineSupport as self.cg; StructuralSimilarity and
        #     Prediction take one as a parameter), so populating the
        #     construction sites needs no provider threading;
        #   * no staleness -- each entry is popped with its node in
        #     _prune_pattern_nodes, and the store dies with the graph,
        #     so a reused node id can never inherit an older anchor;
        #   * graph-proportional -- one entry per LIVE node id, same
        #     lifetime as _pattern_node_signature. No cap, and none
        #     needed: the bound is the pattern-node count (1239 on this
        #     corpus).
        self._pattern_node_anchor: Dict[int, Any] = {}
        self._pattern_node_position: Dict[int, Tuple[float, float]] = {}
        # [Seam 3/D1] Structural-removal events for the Builder's basin
        # split check: node ids whose incident edges were removed or which
        # were themselves pruned. Drained once per Builder step.
        self._removed_structure_nodes: Set[int] = set()
        # Bounded; see _PATTERN_TOMBSTONE_CAP. Used to prevent recreating a
        # node for a pruned key within a short window.
        self._pattern_node_tombstones: List[Tuple[str, bytes]] = []
        self._pattern_node_tombstone_set: Set[Tuple[str, bytes]] = set()
        # node_id -> set of candidate canonical-hash bytes
        self._node_to_candidates: Dict[int, Set[bytes]] = defaultdict(set)
        # node_id -> set of concept_ids whose members include this node
        self._node_to_concepts: Dict[int, Set[int]] = defaultdict(set)
        # [I.4] Member-set -> concept id index for family lookup.
        # Family depends on detection evidence (which lenses fired) --
        # the member set. It does NOT depend on edge structure; edges
        # drift under EMA decay and pruning, so WL hashes of live
        # subgraphs do not match crystallization-time hashes. The member
        # set is stable because pattern-node identity is
        # (lens_type, signature_hash) per §1.11.
        self._concepts_by_member_set: Dict[
            FrozenSet[int], List[int]
        ] = defaultdict(list)

        # ---- §4.20 lookup indexes ----------------------------------------
        self._family_key_index: Dict[Tuple[str, bytes], int] = {}
        self._canonical_hash_index: Dict[bytes, int] = {}

        # ---- Reliability counters ----------------------------------------
        self.gamma_rel = 0.999
        self.prev_opportunity_occurred: Dict[int, float] = {}
        self.prev_confirmation_occurred: Dict[int, float] = {}

        # ---- §1.12 per-type edge homeostasis -----------------------------
        # Each edge type carries its own active threshold, target rate,
        # emission-rate EMA, and context/fatigue traces. v31 initializes all
        # types to the same values; type-specific tuning is deferred to v32.
        self.edge_type_theta: Dict[EdgeType, float] = {
            et: THETA_EDGE_ACTIVE for et in EdgeType
        }
        self.edge_type_r_bar: Dict[EdgeType, float] = {et: 0.0 for et in EdgeType}
        self.edge_type_C: Dict[EdgeType, float] = {et: 0.0 for et in EdgeType}
        self.edge_type_F: Dict[EdgeType, float] = {et: 0.0 for et in EdgeType}
        self.rho_edge_target = 0.3
        self.kappa_edge_homeostasis = 0.05

        # ---- [P2-3/P2-4] diagnostics & homeostasis state ------------------
        # Counter dict surfaced through the pipeline's DIAG tick
        # (_graph_diagnostic). Ints count events; edge_entropy_current is
        # stored separately (float).
        self.diagnostics: Dict[str, Any] = {
            "pattern_nodes_preserved_articulation": 0,
            "edge_entropy_rebalances": 0,
            # [Batch 3 Phase 3] Rule-4 (CROSS_DOMAIN_PATTERN) family-
            # resolution instrumentation. Incremented at EVERY
            # binder._map_family call site (both live in builder.py and
            # both have access to this graph: subgraph_to_cross_section
            # via its cg parameter, Builder._candidate_family via
            # self.cg). Read-side only: no control flow consumes these
            # counters; deterministic integer increments.
            "map_family_checked": 0,
            "map_family_rule4_fired": 0,
            # [§P3 measurement-only / Batch B] Read-only instrumentation
            # for the pruning-candidates-scored calibration. Feeds the
            # eventual fertility-protection threshold fix.
            "pruning_candidates_scored": 0,
            "pruning_rf_max": 0.0,
            "pruning_rf_sum": 0.0,
            "pruning_ef_max": 0.0,
            "pruning_ef_sum": 0.0,
            # [Phase 1 Decision 1] Concepts evicted by the three-signal
            # policy since process start.
            "pruning_evictions": 0,
        }
        # [P2-3] Tarjan articulation-point cache for _prune_pattern_nodes:
        # the set of live pattern-node ids whose removal would split the
        # pattern-node subgraph. Recomputed at most once per
        # P.TARJAN_CADENCE steps. None = never computed.
        self._tarjan_cache: Optional[Set[int]] = None
        self._tarjan_cache_step: int = 0
        # [P2-4] Entropy-homeostasis state: the set of over-dominant edge
        # types observed on the previous step_38 pass (for rebalance-edge
        # detection), the feedback gain, and the current Shannon entropy
        # (base 2) of the edge-type share distribution.
        self._prev_dominant_types: Set[EdgeType] = set()
        self.kappa_entropy_feedback = P.KAPPA_ENTROPY_FEEDBACK
        self.edge_entropy_current: float = 0.0

        # ---- Params -------------------------------------------------------
        self.params: Dict[str, Any] = {
            "theta_opp": THETA_OPP,
            "theta_conf": THETA_CONF,
            "theta_conf_min": THETA_CONF_MIN,
            "theta_conf_max": THETA_CONF_MAX,
            "theta_anchor": THETA_ANCHOR,
            "theta_span": THETA_SPAN,
            "theta_witness": THETA_WITNESS,
            "theta_decisive": THETA_DECISIVE,
            "rho_sym_max": RHO_SYM_MAX,
            "N_maxAmbiguity": N_MAX_AMBIGUITY,
            "N_candMax": N_CAND_MAX,
            "N_layerMax": N_LAYER_MAX,
            "N_nodeMax": N_NODE_MAX,
            "tau_dormant": TAU_DORMANT,
            "tau_softDormant": TAU_SOFT_DORMANT,
            "RF_softMin": RF_SOFT_MIN,
            "tau_tombstone": TAU_TOMBSTONE,
            "tau_forget": TAU_FORGET,
            "w_init": W_INIT,
            "eta_w": ETA_W,
            "w_max": W_MAX,
            "beta_w": BETA_W,
            "tau_edgeInactive": TAU_EDGE_INACTIVE,
            "theta_edgeActive": THETA_EDGE_ACTIVE,
            "theta_blockMax": THETA_BLOCK_MAX,
            "N_tombMax": N_TOMB_MAX,
            "epsilon": EPSILON,
            "delta_w": DELTA_W,
            "family_conf_thresholds": {},
            "lens_thresholds": {"Sequence": THETA_SEQ, "Structure": THETA_STRUCT},
            "theta_conf_chunk": THETA_CONF_CHUNK,
            "theta_anchor_chunk": THETA_ANCHOR_CHUNK,
            "theta_lens_chunk": THETA_LENS_CHUNK,
        }
        if params:
            self.params.update(params)
        for fam in ConceptFamily:
            self.params["family_conf_thresholds"][fam.name] = self.params["theta_conf"]

        # ---- [TEMP: MIGRATION] counterfactual machinery (§8.9) -----------
        # Retained until prediction.py is wired. `prediction_module` may be
        # set externally; when present, witness/gate queries prefer it.
        self.prediction_module: Optional[Any] = None
        self.gamma_cf = P.GAMMA_CF
        self.w_confirm = 5
        self.rho_smooth = P.RHO_SMOOTH
        self.N_cf_min = P.N_CF_MIN
        self.eps_hyst_min = P.EPS_HYST_NEC_MIN
        self.c_hyst = P.C_HYST_NECESSITY
        self.eps_count = 1e-9
        self.no_op_actions = {ACTION_NULL, ACTION_NOOP}
        self.effect_flags: Set[Any] = set()
        self.C_opp: Dict[Any, float] = defaultdict(float)
        self.C_opp_not: Dict[Any, float] = defaultdict(float)
        self.resolved_count: Dict[Any, float] = defaultdict(float)
        self.resolved_real_total: float = 0.0
        self.C_succ_a: Dict[Tuple[Any, Any], float] = defaultdict(float)
        self.C_succ_real_total: Dict[Any, float] = defaultdict(float)
        self.prev_bar_p_a: Dict[Tuple[Any, Any], float] = defaultdict(float)
        self.prev_bar_p_not_a: Dict[Tuple[Any, Any], float] = defaultdict(float)
        self.flag_init_p_a: Dict[Tuple[Any, Any], bool] = defaultdict(bool)
        self.flag_init_p_not_a: Dict[Tuple[Any, Any], bool] = defaultdict(bool)
        self.prev_H_nec: Dict[Tuple[Any, Any], int] = defaultdict(int)
        self.Gate_bind: Dict[Tuple[Any, Any], int] = defaultdict(int)
        self.Gate_undo: Dict[Tuple[Any, Any], int] = defaultdict(int)
        self.bar_dp_dict: Dict[Tuple[Any, Any], float] = defaultdict(float)

    # -------------------------------------------------------------------------
    # ID allocation
    # -------------------------------------------------------------------------

    def _allocate_id(self) -> int:
        nid = self.next_concept_id
        self.next_concept_id += 1
        return nid

    # -------------------------------------------------------------------------
    # §1.11 Pattern-node store
    # -------------------------------------------------------------------------

    def graph_revision(self) -> int:
        """[Batch D / P1-1] Endogenous graph-change signal: edge-level
        topology revisions plus pattern-node revisions. Cadence consumers
        (Builder basin recompute, stall/excursion/epistemic bounds,
        frontier pruning, theta_fertile, analogical decay) measure
        waiting in graph-change events rather than elapsed steps."""
        return self.topology_revision + self.node_revision

    def get_or_create_pattern_node(
        self,
        lens_type: str,
        signature_hash: bytes,
        detection_hash: Optional[bytes] = None,
        signature: Optional[TypedSignature] = None,
    ) -> int:
        key = (lens_type, signature_hash)
        nid = self._pattern_node_id_by_key.get(key)
        if nid is not None:
            self._pattern_node_last_seen[nid] = self.t
            if detection_hash is not None:
                self._pattern_node_detection_hash[nid] = detection_hash
            # First-write-wins for the full signature: subsequent calls
            # with the same (lens_type, signature_hash) are guaranteed to
            # be equivalent (signature_hash is a deterministic function of
            # signature), so overwriting is unnecessary.
            if signature is not None and nid not in self._pattern_node_signature:
                self._pattern_node_signature[nid] = signature
            return nid
        nid = self._allocate_id()
        self._pattern_node_id_by_key[key] = nid
        self._pattern_node_key_by_id[nid] = key
        self.node_revision += 1  # [Bundle A] node add
        if detection_hash is not None:
            self._pattern_node_detection_hash[nid] = detection_hash
        if signature is not None:
            self._pattern_node_signature[nid] = signature
        self._pattern_node_last_seen[nid] = self.t
        return nid

    def get_pattern_node_signature(self, nid: int) -> Optional[Tuple[str, bytes]]:
        return self._pattern_node_key_by_id.get(nid)

    def get_pattern_node_detection_hash(self, nid: int) -> Optional[bytes]:
        return self._pattern_node_detection_hash.get(nid)

    def get_pattern_node_full_signature(self, nid: int) -> Optional[TypedSignature]:
        """[v31 Stage D1] Return the full TypedSignature registered for a
        pattern node, or None if no signature was supplied at creation.

        Consumers building a SignatureProvider reconstruct the
        (lens_type, TypedSignature) tuple from this and
        get_pattern_node_signature(nid). The pair is what the
        Consolidator/Mapper/Executor MCS machinery expects.
        """
        return self._pattern_node_signature.get(nid)

    # -------------------------------------------------------------------------
    # [B2] Action-vocabulary registry + node introspection accessors
    # -------------------------------------------------------------------------

    def register_action_alpha(self, nid: int, action: Any) -> None:
        """Bind pattern node `nid` (an Action-lens node) to the exact
        action object. First-write-wins: node identity is the 64-bit
        canonical hash of the action, so re-registration is idempotent;
        a DIFFERENT value for the same node would mean a 64-bit signature
        collision — counted observably, never silently overwritten."""
        existing = self._pattern_node_action_alpha.get(nid)
        if existing is None:
            self._pattern_node_action_alpha[nid] = action
        elif existing is not action and serialize_value(existing) != serialize_value(action):
            self.action_alpha_collisions += 1

    def get_node_action(self, nid: int) -> Optional[Any]:
        """Exact action object registered for an Action-lens pattern node,
        or None (non-action node / pruned registry entry). Decode returns
        the stored object directly — the same hashable value the
        environment's action_edge_map is keyed by (K1 round-trip)."""
        return self._pattern_node_action_alpha.get(nid)

    # -------------------------------------------------------------------------
    # [V9 Track 2 - B.26] anchor / position registry
    # -------------------------------------------------------------------------

    def note_pattern_node_anchor(self, nid: int, anchor: Any) -> None:
        """Bind pattern node `nid` to the observation anchor it was first
        bound from.

        FIRST-write-wins, for the same reason register_action_alpha is: a
        pattern node is keyed by (lens_type, signature_hash), so many
        detections can reach one node, and which anchor is recorded must not
        depend on emitter order. Callers iterate deterministic structures,
        so the first write is deterministic. Overwriting would make the
        entry depend on the last emitter to run -- a silent, run-order
        dependent change in a node's recorded geometry.
        """
        if nid not in self._pattern_node_anchor:
            self._pattern_node_anchor[nid] = anchor

    def pattern_node_anchor_map(self) -> Dict[int, Any]:
        """node id -> anchor. Defensive copy."""
        return dict(self._pattern_node_anchor)

    def note_pattern_node_position(self, nid: int, position: Any) -> None:
        """Bind pattern node `nid` to its anchor's (x, y) position.

        The caller supplies the position of the step in which the node was
        MINTED -- never a lookup at read time. That distinction is the whole
        point: `Binder._step_anchor_positions` is assigned at the end of
        Binder.step, so anything that captures positions earlier reads the
        PREVIOUS step's map (the trap that removed `node_positions()`).

        First-write-wins, like the anchor. Stores the (x, y) pair only;
        extra dimensions of a 3D anchor are dropped, matching
        Subgraph.anchor_of_node's declared Tuple[float, float].
        """
        if nid in self._pattern_node_position:
            return
        if position is None or len(position) < 2:
            return
        self._pattern_node_position[nid] = (
            float(position[0]),
            float(position[1]),
        )

    def pattern_node_position_map(self) -> Dict[int, Tuple[float, float]]:
        """node id -> (x, y) of the anchor whose minting step had a position
        for it. Nodes minted from an anchor with no position are absent.
        Defensive copy."""
        return dict(self._pattern_node_position)

    def pattern_node_position(
        self, nid: int
    ) -> Optional[Tuple[float, float]]:
        """(x, y) recorded for `nid`, or None if the node is unknown or
        was minted with no position available.

        Single-node lookup, deliberately: the Subgraph construction
        sites that hold a ConceptGraph need per-node resolution, and
        several of them run inside consolidation loops. Copying the
        whole registry there would be O(nodes x registry) for no gain.
        """
        return self._pattern_node_position.get(nid)

    def get_node_lens_type(self, nid: int) -> Optional[str]:
        """Lens type of a pattern node (§1.11 identity component 1)."""
        key = self._pattern_node_key_by_id.get(nid)
        return key[0] if key is not None else None

    def drain_removed_structure_nodes(self) -> Set[int]:
        """[Seam 3/D1] Return and clear the node ids touched by edge
        removals / node prunes since the last drain. The Builder consumes
        this once per step to run immediate basin-split checks instead of
        waiting for the K_main full recompute (bounded staleness was the
        accepted v31 compromise; §4.9 calls for immediate splits)."""
        out = self._removed_structure_nodes
        self._removed_structure_nodes = set()
        return out

    def node_to_concepts(self, nid: int) -> Set[int]:
        """[v31 Stage D1] Public accessor for the node → concept reverse
        index. Returns a defensive copy so callers cannot mutate the
        internal index. Empty set if the node is not referenced by any
        live concept.

        Mapper and UtilityEngine previously reached into the private
        `_node_to_concepts` field directly; this is the sanctioned
        read-only alternative.
        """
        return set(self._node_to_concepts.get(nid, set()))

    def concepts_by_member_set(
        self, members: FrozenSet[int]
    ) -> List[int]:
        """[I.4] Concept ids whose member set exactly matches `members`.

        Returns a defensive copy. Empty list if no concept has this exact
        member set.
        """
        return list(self._concepts_by_member_set.get(members, []))

    def iter_pattern_nodes(self) -> Iterator[Tuple[int, str, bytes]]:
        for nid, (lens, sig) in self._pattern_node_key_by_id.items():
            yield nid, lens, sig

    def get_diagnostics(self) -> Dict[str, int]:
        """[Bundle A] revision counters for the pipeline memory snapshot."""
        return {
            "node_revision": self.node_revision,
            "edge_revision": self.edge_revision,
            "topology_revision": self.topology_revision,
            "concept_revision": self.concept_revision,
        }

    def get_live_pattern_node_ids(self) -> AbstractSet[int]:
        """§1.11/§1.13: read-only live view of every pattern-node ID currently
        in the single canonical node store.

        This is the sanctioned public alternative to reaching into the private
        `_pattern_node_key_by_id` dict from other modules — the same boundary
        discipline `node_to_concepts()` documents for the reverse index. It
        returns the dict's live key view (not a copy), so:
          - `len(...)` is O(1),
          - membership tests are O(1),
          - the view tracks pruning/eviction automatically (a node evicted by
            `_prune_pattern_nodes` disappears from it with no stale snapshot),
          - callers cannot mutate the store through it (KeysView is read-only).

        Builder uses this for its node-map sweep and pattern-adjacency cache
        signature instead of touching the private field directly.
        """
        return self._pattern_node_key_by_id.keys()

    def register_candidate_nodes(
        self, canonical_hash: bytes, node_ids: Iterable[int]
    ) -> None:
        """§1.11: called by the Builder when it seeds, extends, or merges a
        CandidateSubgraph. Keeps referenced pattern nodes from being pruned."""
        for nid in node_ids:
            self._node_to_candidates[nid].add(canonical_hash)

    def unregister_candidate_nodes(
        self, canonical_hash: bytes, node_ids: Iterable[int]
    ) -> None:
        """§1.11: called by the Builder on candidate eviction."""
        for nid in node_ids:
            s = self._node_to_candidates.get(nid)
            if s is not None:
                s.discard(canonical_hash)
                if not s:
                    self._node_to_candidates.pop(nid, None)

    def _compute_articulation_points(self) -> Set[int]:
        """[P2-3] Tarjan articulation points of the live pattern-node
        subgraph (iterative DFS — no recursion-depth limit on long
        structure chains).

        The induced subgraph is over live pattern-node ids only. Pattern
        nodes and concepts share one id space (_allocate_id /
        next_concept_id), so membership in _pattern_node_key_by_id is an
        exact live-pattern-node filter and the induced-subgraph Tarjan
        result is sound: articulation points are precisely the nodes whose
        removal splits the pattern-node connectivity the Builder relies on
        for multi-node pattern re-detection.

        Determinism: live nodes are visited in sorted id order and each
        adjacency list is sorted, so discovery numbering (and hence the
        returned set) is independent of dict/set insertion order.
        """
        live = sorted(self._pattern_node_key_by_id.keys())
        live_set = set(live)
        adj: Dict[int, List[int]] = {}
        for n in live:
            nbrs = set()
            for dst, _et in self._outgoing.get(n, ()):  # type: ignore[misc]
                if dst in live_set:
                    nbrs.add(dst)
            for src, _et in self._incoming.get(n, ()):  # type: ignore[misc]
                if src in live_set:
                    nbrs.add(src)
            adj[n] = sorted(nbrs)

        discovery: Dict[int, int] = {}
        low: Dict[int, int] = {}
        parent: Dict[int, int] = {}
        timer = [0]
        articulation: Set[int] = set()

        for root in live:
            if root in discovery:
                continue
            discovery[root] = timer[0]
            low[root] = timer[0]
            timer[0] += 1
            parent[root] = -1
            child_count_start = 0
            # Stack frames: (node, next-adjacency index).
            stack: List[Tuple[int, int]] = [(root, 0)]
            while stack:
                u, i = stack[-1]
                if i < len(adj[u]):
                    stack[-1] = (u, i + 1)
                    v = adj[u][i]
                    if v not in discovery:
                        parent[v] = u
                        discovery[v] = timer[0]
                        low[v] = timer[0]
                        timer[0] += 1
                        if u == root:
                            child_count_start += 1
                        stack.append((v, 0))
                    elif v != parent.get(u, -1):
                        # Back edge to an ancestor / already-visited node.
                        if discovery[v] < low[u]:
                            low[u] = discovery[v]
                else:
                    stack.pop()
                    if stack:
                        p = stack[-1][0]
                        if low[u] < low[p]:
                            low[p] = low[u]
                        # Non-root rule: p is an articulation point when a
                        # child subtree cannot reach above p.
                        if parent.get(p, -1) != -1 and low[u] >= discovery[p]:
                            articulation.add(p)
            # Root rule: DFS-tree root with more than one child.
            if child_count_start > 1:
                articulation.add(root)
        return articulation

    def _prune_pattern_nodes(self) -> int:
        """§1.11 + [P2-3]: evict pattern nodes with (a) no live concept,
        (b) no live candidate, (c) no observation in PATTERN_NODE_TTL
        steps, and (d) NOT an articulation point of the live pattern-node
        subgraph.

        Rationale for (d): removing an articulation point splits a
        connected structure basin into components that can never re-merge
        through the graph, permanently blinding the Builder to multi-node
        patterns spanning the split. TTL-stale articulation points are
        therefore preserved (and counted) until their neighbourhood
        re-observes them or the topology changes.
        """
        ttl = P.PATTERN_NODE_TTL
        cutoff = self.t - ttl
        # [P2-3] Sorted candidate iteration: TTL eviction order must be
        # insertion-order-independent for determinism.
        candidates = sorted(
            nid
            for nid, last_seen in self._pattern_node_last_seen.items()
            if last_seen <= cutoff
            and not self._node_to_concepts.get(nid)
            and not self._node_to_candidates.get(nid)
        )
        if not candidates:
            return 0
        cadence = getattr(P, "TARJAN_CADENCE", 1)
        if (
            self._tarjan_cache is None
            or self.t - self._tarjan_cache_step >= cadence
        ):
            self._tarjan_cache = self._compute_articulation_points()
            self._tarjan_cache_step = self.t
        articulation = self._tarjan_cache
        to_remove = []
        for nid in candidates:
            if nid in articulation:
                self.diagnostics["pattern_nodes_preserved_articulation"] += 1
                continue
            to_remove.append(nid)
        for nid in to_remove:
            # [MEMORY HYGIENE] Purge all incident edges before removing
            # the node's metadata. Without this, over thousands of steps
            # the edge store accumulates orphan edges pointing to pruned
            # node IDs, wasting memory and forcing get_outgoing_edges /
            # get_incoming_edges lookups against the pattern-node store
            # for missing keys on every traversal.
            dead_edge_keys = []
            for dst, et in list(self._outgoing.get(nid, set())):
                dead_edge_keys.append((nid, dst, et))
            for src, et in list(self._incoming.get(nid, set())):
                dead_edge_keys.append((src, nid, et))
            for e_key in dead_edge_keys:
                self._remove_edge(e_key)

            key = self._pattern_node_key_by_id.pop(nid, None)
            if key is not None:
                self._pattern_node_id_by_key.pop(key, None)
                self.node_revision += 1  # [Bundle A] node remove
                if key not in self._pattern_node_tombstone_set:
                    self._pattern_node_tombstones.append(key)
                    self._pattern_node_tombstone_set.add(key)
                    if len(self._pattern_node_tombstones) > _PATTERN_TOMBSTONE_CAP:
                        evict = self._pattern_node_tombstones.pop(0)
                        self._pattern_node_tombstone_set.discard(evict)
            self._pattern_node_detection_hash.pop(nid, None)
            self._pattern_node_signature.pop(nid, None)
            self._pattern_node_action_alpha.pop(nid, None)
            self._pattern_node_anchor.pop(nid, None)
            self._pattern_node_position.pop(nid, None)
            self._pattern_node_last_seen.pop(nid, None)
            self._node_to_candidates.pop(nid, None)
            self._node_to_concepts.pop(nid, None)
            # [Seam 3/D1] A pruned node severs every basin path through
            # it — record it for the Builder's split check.
            self._removed_structure_nodes.add(nid)
        return len(to_remove)

    def _is_pattern_node_tombstoned(
        self, lens_type: str, signature_hash: bytes
    ) -> bool:
        return (lens_type, signature_hash) in self._pattern_node_tombstone_set

    # -------------------------------------------------------------------------
    # §1.12 Edge store
    # -------------------------------------------------------------------------

    def upsert_edge(
        self,
        src: int,
        dst: int,
        edge_type: EdgeType,
        signal: float,
        t: int,
        lag: int = 0,
        params: Optional[Any] = None,
    ) -> None:
        """Binder write path. Records the raw signal for this step; the
        weight EMA is applied by step_28_update_edges.

        Self-loop guard (§3.4): only SELF_BINDER_TYPES may have src == dst.
        """
        if src == dst and edge_type not in SELF_BINDER_TYPES:
            return
        key = (src, dst, edge_type)
        edge = self.edges.get(key)
        if edge is None:
            self.edges[key] = Edge(
                parent=src,
                child=dst,
                weight=0.0,
                edge_type=edge_type,
                lag=lag,
                pending_signal=signal,
                observed_at=t,
                initialized=False,
                params=params,
            )
            self._outgoing[src].add((dst, edge_type))
            self._incoming[dst].add((src, edge_type))
            self._topo_version += 1
            self.topology_revision += 1  # [Batch D / P1-1]
        else:
            prev_t = edge.observed_at  # Capture previous observation step before update
            if edge.observed_at == t:
                edge.pending_signal = max(edge.pending_signal, signal)
            else:
                edge.pending_signal = signal
            edge.observed_at = t
            if lag != 0:
                edge.lag = lag

            # [S3] Accumulate multimodal parameter distributions decaying strictly by time
            if params is not None:
                cur = edge.params
                if cur is None:
                    edge.params = params
                elif isinstance(cur, EdgeParamsMixture):
                    if isinstance(params, EdgeParamsMixture):
                        for p, m in zip(params.protos, params.mass):
                            cur.observe(p, t, weight=m * signal)
                    else:
                        cur.observe(params, t, weight=signal)
                elif isinstance(cur, EdgeParams):
                    mix = EdgeParamsMixture.from_point(cur, prev_t)
                    if isinstance(params, EdgeParamsMixture):
                        for p, m in zip(params.protos, params.mass):
                            mix.observe(p, t, weight=m * signal)
                    else:
                        mix.observe(params, t, weight=signal)
                    edge.params = mix

    def get_edge(
        self,
        src: int,
        dst: int,
        edge_type: Optional[EdgeType] = None,
    ) -> Optional[Edge]:
        """With edge_type: O(1). Without: iterate EdgeType in enum order
        and return the first match (deterministic)."""
        if edge_type is not None:
            return self.edges.get((src, dst, edge_type))
        for et in EdgeType:
            e = self.edges.get((src, dst, et))
            if e is not None:
                return e
        return None

    def get_outgoing_edges(
        self, src: int, edge_type: Optional[EdgeType] = None
    ) -> List[Edge]:
        cached = self._out_cache.get(src)
        if cached is None or cached[0] != self._topo_version:
            out = []
            for dst, et in self._outgoing.get(src, set()):
                e = self.edges.get((src, dst, et))
                if e is not None:
                    out.append(e)
            # [DETERMINISM §9] canonical order (see _topo_version note):
            # EdgeType hashes by Enum _name_ (a str), so raw set iteration
            # is PYTHONHASHSEED-randomized; every consumer must see ONE
            # order. Cached; rebuilt only on topology change.
            out.sort(key=lambda e: (e.child, e.edge_type.value))
            self._out_cache[src] = (self._topo_version, out)
        else:
            out = cached[1]
        if edge_type is None:
            return list(out)
        return [e for e in out if e.edge_type == edge_type]

    def get_incoming_edges(
        self, dst: int, edge_type: Optional[EdgeType] = None
    ) -> List[Edge]:
        cached = self._in_cache.get(dst)
        if cached is None or cached[0] != self._topo_version:
            out = []
            for src, et in self._incoming.get(dst, set()):
                e = self.edges.get((src, dst, et))
                if e is not None:
                    out.append(e)
            # [DETERMINISM §9] canonical order; cached (see _topo_version).
            out.sort(key=lambda e: (e.parent, e.edge_type.value))
            self._in_cache[dst] = (self._topo_version, out)
        else:
            out = cached[1]
        if edge_type is None:
            return list(out)
        return [e for e in out if e.edge_type == edge_type]

    def iter_edges(self) -> Iterator[Edge]:
        return iter(self.edges.values())

    def get_active_edges(self) -> Dict[Tuple[int, int, EdgeType], Edge]:
        """Returns edges whose weight exceeds their type's active
        threshold."""
        out = {}
        for k, e in self.edges.items():
            th = self.edge_type_theta.get(e.edge_type, THETA_EDGE_ACTIVE)
            if e.weight >= th:
                out[k] = e
        return out

    def get_coact_ema(self) -> Dict[Tuple[int, int, EdgeType], float]:
        return {k: e.CoAct_EMA for k, e in self.edges.items()}

    def get_time_since_coact(self) -> Dict[Tuple[int, int, EdgeType], int]:
        return {k: e.time_since_coact for k, e in self.edges.items()}

    def _remove_edge(self, key: Tuple[int, int, EdgeType]) -> None:
        e = self.edges.pop(key, None)
        if e is None:
            return
        # [Seam 3/D1] Every edge removal (step_28 inactivity prune,
        # pattern-node prune, explicit deletion) is a potential basin
        # split event — record both endpoints for the Builder's drain.
        self._removed_structure_nodes.add(e.parent)
        self._removed_structure_nodes.add(e.child)
        self._topo_version += 1
        self.topology_revision += 1  # [Batch D / P1-1]
        s = self._outgoing.get(e.parent)
        if s is not None:
            s.discard((e.child, e.edge_type))
            if not s:
                self._outgoing.pop(e.parent, None)
        s = self._incoming.get(e.child)
        if s is not None:
            s.discard((e.parent, e.edge_type))
            if not s:
                self._incoming.pop(e.child, None)

    # -------------------------------------------------------------------------
    # Reliability counters (stay in ConceptGraph; feed witness)
    # -------------------------------------------------------------------------

    def check_necessity(self, action: Any, effect_id: Any) -> Tuple[bool, float, int]:
        """[TEMP: MIGRATION] Adapter. Prefers prediction_module when set."""
        if self.prediction_module is not None:
            is_nec = self.prediction_module.gate_undo(action, effect_id)
            delta = self.prediction_module.causal_contrast(action, effect_id)
            n_cf = min(
                self.prediction_module.C_opp.get(action, 0.0),
                self.prediction_module.C_opp_not.get(action, 0.0),
            )
            return is_nec, delta, int(n_cf)
        if action is None or effect_id is None:
            return False, 0.0, 0
        key = (action, effect_id)
        N_cf = min(self.C_opp.get(action, 0.0), self.C_opp_not.get(action, 0.0))
        bar_dp = self.bar_dp_dict.get(key, 0.0)
        is_necessary = (
            bool(self.prev_H_nec.get(key, 0))
            and N_cf >= self.N_cf_min
            and bar_dp >= 0.40
        )
        return is_necessary, bar_dp, int(N_cf)

    def step_reliability_begin(self, t: int) -> None:
        self.t = t
        for P_c in self.concepts.values():
            opp_inc = 1.0 if self.prev_opportunity_occurred.get(P_c.id, 0.0) else 0.0
            conf_inc = 1.0 if self.prev_confirmation_occurred.get(P_c.id, 0.0) else 0.0
            P_c.M_opp = self.gamma_rel * getattr(P_c, "M_opp", 0.0) + opp_inc
            P_c.M_conf = self.gamma_rel * getattr(P_c, "M_conf", 0.0) + conf_inc
            P_c.Reliability = min(1.0, P_c.M_conf / (P_c.M_opp + EPSILON))

    def store_reliability_indicators(
        self, t: int, opp_map: Dict[int, bool], conf_map: Dict[int, bool]
    ) -> None:
        for P_c in self.concepts.values():
            self.prev_opportunity_occurred[P_c.id] = (
                1.0 if opp_map.get(P_c.id, False) else 0.0
            )
            self.prev_confirmation_occurred[P_c.id] = (
                1.0 if conf_map.get(P_c.id, False) else 0.0
            )

    # -------------------------------------------------------------------------
    # [TEMP: MIGRATION] Counterfactual attribution (§8.3)
    # -------------------------------------------------------------------------

    def step_counterfactual(self, *args, **kwargs) -> None:
        """[DEPRECATED §8.9] Causal counterfactual attribution moved to
        PredictionModule. Retained solely as a delegation shim so any
        legacy caller routes through the PredictionModule instead of
        duplicating the §8.3 machinery."""
        if self.prediction_module is not None and hasattr(
            self.prediction_module, "step_causal"
        ):
            self.prediction_module.step_causal(*args, **kwargs)

    def gate_bind(self, a: Any, b: Any) -> bool:
        if self.prediction_module is not None:
            return self.prediction_module.gate_bind(a, b)
        # Bootstrap fallback (§8.3.3): permit binding when PredictionModule
        # is not yet wired or counterfactual samples are below N_cf_min.
        return True

    def gate_undo(self, a: Any, b: Any) -> bool:
        if self.prediction_module is not None:
            return self.prediction_module.gate_undo(a, b)
        if a is None or b is None:
            return False
        return bool(self.Gate_undo.get((a, b), 0))

    # -------------------------------------------------------------------------
    # §1.12 / §2.1.18 Homeostasis
    # -------------------------------------------------------------------------

    def step_38_homeostasis(self, t: int) -> None:
        # §1.12: per-type edge threshold homeostasis.
        # [P2-4] Entropy homeostasis: track the Shannon entropy (base 2) of
        # edge-type shares and add a dominance-proportional θ nudge so
        # over-represented types are gently suppressed, diversifying the
        # edge-type distribution.
        per_type_counts: Dict[EdgeType, Tuple[int, int]] = {
            et: (0, 0) for et in EdgeType
        }
        for e in self.edges.values():
            active, total = per_type_counts[e.edge_type]
            th = self.edge_type_theta.get(e.edge_type, THETA_EDGE_ACTIVE)
            is_active = e.weight >= th
            per_type_counts[e.edge_type] = (active + (1 if is_active else 0), total + 1)

        total_edges = len(self.edges)
        # [v8] Effective type count = types PRESENT in the edge store,
        # not the enum size. Decouples new enum members from the entropy
        # homeostasis of existing channels: adding an unused type no
        # longer raises the dominance threshold for existing types.
        types_present_count = sum(
            1 for counts in per_type_counts.values() if counts[1] > 0
        )
        num_types = max(1, types_present_count)
        uniform_share = 1.0 / num_types

        # Shannon entropy (base 2) over type shares; 0.0 when graph empty.
        entropy = 0.0
        if total_edges > 0:
            for et in EdgeType:
                _, tot = per_type_counts[et]
                if tot > 0:
                    share_e = tot / total_edges
                    entropy -= share_e * math.log2(share_e)
        self.edge_entropy_current = entropy

        curr_dominant: Set[EdgeType] = set()
        for et in EdgeType:
            active, total = per_type_counts[et]
            r = active / (total + EPSILON) if total > 0 else 0.0
            r_bar_prev = self.edge_type_r_bar.get(et, 0.0)
            r_bar_new = (
                1.0 - self.kappa_edge_homeostasis
            ) * r_bar_prev + self.kappa_edge_homeostasis * r
            self.edge_type_r_bar[et] = r_bar_new
            share = total / total_edges if total_edges > 0 else 0.0
            dominance = max(0.0, share - uniform_share)
            if share > 2.0 * uniform_share and total > 0:
                curr_dominant.add(et)
                if et not in self._prev_dominant_types:
                    self.diagnostics["edge_entropy_rebalances"] += 1
            new_theta = (
                self.edge_type_theta.get(et, THETA_EDGE_ACTIVE)
                + self.kappa_edge_homeostasis * (r_bar_new - self.rho_edge_target)
                + self.kappa_entropy_feedback * dominance
            )
            self.edge_type_theta[et] = clamp(new_theta, THETA_EDGE_PRUNE, W_MAX)
        self._prev_dominant_types = curr_dominant

        # Legacy η_w and τ_dormant homeostasis (§2.1.18).
        total_edges = len(self.edges)
        if total_edges > 0:
            active_edges = sum(
                1
                for e in self.edges.values()
                if e.weight >= self.params["theta_edgeActive"]
            )
            rho_edge = active_edges / (total_edges + EPSILON)
            delta_w = clamp(GAIN_ETA_W * (0.3 - rho_edge), -DELTA_PI_MAX, DELTA_PI_MAX)
            self.params["eta_w"] = clamp(self.params["eta_w"] + delta_w, 1e-3, 1.0)
        total_nodes = len(self.concepts)
        if total_nodes > 0:
            dormant_nodes = sum(1 for c in self.concepts.values() if c.dormant)
            rho_dormant = dormant_nodes / (total_nodes + EPSILON)
            delta_dorm = clamp(
                GAIN_TAU_DORMANT * (rho_dormant - 0.1) * self.params["tau_dormant"],
                -10.0,
                10.0,
            )
            self.params["tau_dormant"] = int(
                clamp(self.params["tau_dormant"] + delta_dorm, 100, 10000)
            )

    # -------------------------------------------------------------------------
    # Seeding
    # -------------------------------------------------------------------------

    def seed_atomic_actions(self, actions: List[Any]) -> None:
        for action in actions:
            if any(c.action == action for c in self.concepts.values()):
                continue
            concept_id = self._allocate_id()
            concept = Concept(
                id=concept_id,
                family=ConceptFamily.ATOMIC_ACTION,
                layer=1,
                sigma=TypedSignature.opaque(
                    "ATOMIC_ACTION", hash_64(serialize_value(action))
                ),
                status=ConceptStatus.STABLE,
                credit=EPS_INITIAL,
                action=action,
                enabling_pre_states=set(),
                effect_id=None,
            )
            self.concepts[concept_id] = concept

    # -------------------------------------------------------------------------
    # Step 21 — candidate statistics
    # -------------------------------------------------------------------------

    def _is_inert_or_noop(self, action: Any) -> bool:
        """Constitutional null check (§2.6).
        Returns True for no-ops, probes, and teleological goal references.
        Decomposed internally for auditable extensibility."""
        env = self.params.get("env")
        return (
            is_noop_action(action)
            or is_probe_action(action, env)
            or is_teleological_action(action)
        )

    def step_21_update_candidate_stats(
        self,
        valid_cross_sections,
        t: int,
        prev_action: Any = None,
        a_hist: Optional[Dict[int, Any]] = None,
        onset_history: Optional[Dict[Any, Dict[int, int]]] = None,
    ) -> None:
        self.t = t
        updated_this_step = set()
        for cs in valid_cross_sections:
            try:
                fam = ConceptFamily[cs.family]
            except KeyError:
                continue
            key = cs.key
            cand_key = (fam, key)
            if cand_key in updated_this_step:
                continue
            updated_this_step.add(cand_key)

            existing = self._find_existing_concept_by_cross_section(cs)
            if existing is not None:
                existing.N_opp += 1
                existing.N_conf += 1
                existing.n_conf_from_observation += 1
                if existing.Anchors is None:
                    existing.Anchors = set()
                existing.Anchors.add(cs.anchor)
                existing.N_anchor = len(existing.Anchors)
                if (
                    self.last_state_hash is not None
                    and existing.enabling_pre_states is not None
                ):
                    existing.enabling_pre_states.add(self.last_state_hash)
                if existing.layer >= 2 and existing.family in (
                    ConceptFamily.EVENT_CAUSAL,
                    ConceptFamily.TEMPORAL_CHAIN,
                ):
                    existing.N_opp_LN += 1
                    existing.N_conf_LN += 1

                # [SCHEMA MULTI-BINDING §1.11] Extract the current
                # instance's (action, effect) binding and merge it into
                # the existing schema concept. Under anchor-invariance,
                # switch 2's cross-section matches switch 0's
                # cross_section_key and routes here; without this merge
                # the concept remains bound only to switch 0's action.
                if existing.family in (
                    ConceptFamily.EVENT_CAUSAL,
                    ConceptFamily.TEMPORAL_CHAIN,
                ):
                    # [COUNTERFACTUAL-GATED BINDING §8.3 — v31 NOTE]
                    # UNREACHABLE in v31: bound_effect is extracted from
                    # det.rho "transitionEffect" roles. Binding is performed by
                    # Pipeline._bind_causal_schemas instead (H_nec-
                    # validated, atom-namespace). Kept for reference only.
                    bound_effect = None
                    for det in cs.detections:
                        for role in det.rho:
                            if (
                                role[0] == "transitionEffect"
                                and isinstance(role[1], int)
                            ):
                                bound_effect = role[1]
                                break
                        if bound_effect is not None:
                            break

                    effect_is_fresh = False
                    if bound_effect is not None and onset_history is not None:
                        effect_is_fresh = (
                            onset_history.get(bound_effect, {}).get(t, 0) == 1
                        )
                    elif bound_effect is not None:
                        effect_is_fresh = any(
                            getattr(d, "k", None) == "Onset"
                            and getattr(d, "t", None) == t
                            for d in cs.detections
                        )

                    if (
                        effect_is_fresh
                        and prev_action is not None
                        and not self._is_inert_or_noop(prev_action)
                    ):
                        # Route through Counterfactual Gate
                        if self.gate_bind(prev_action, bound_effect):
                            existing.actions.add(prev_action)
                            self.action_to_concepts[prev_action].add(existing.id)
                            if bound_effect is not None:
                                existing.effect_ids_by_action[prev_action] = (
                                    bound_effect
                                )
                continue

            if cand_key not in self.candidates:
                cand = Candidate(
                    K_C=tuple([key]),
                    family=fam,
                    anchorKind="",
                    combinationType="ordinary",
                    T_first=t,
                    T_last=t,
                )
                self.candidates[cand_key] = cand
            else:
                cand = self.candidates[cand_key]

            cand.N_opp += 1
            cand.T_last = t
            cand.Anchors.add(cs.anchor)
            cand.N_anchor = len(cand.Anchors)
            cand.N_conf += 1
            cand.last_cross_section = cs

            # §4.3.1: cache cross_section_key and canonical_hash (derived for
            # v30 candidates), and register pattern nodes on the candidate
            # so both indexes and pruning are consistent.
            if cand.cross_section_key is None:
                cand.cross_section_key = self._to_bytes(cs.key)
            if cand.canonical_hash is None:
                cand.canonical_hash = self._derive_canonical_hash(cand, cs)
            node_ids = self._candidate_node_ids(cs)
            cand.node_ids = frozenset(node_ids)
            self.register_candidate_nodes(cand.canonical_hash, node_ids)

            if fam in (ConceptFamily.EVENT_CAUSAL, ConceptFamily.TEMPORAL_CHAIN):
                if self.last_state_hash is not None:
                    cand.enabling_pre_states.add(self.last_state_hash)
                # Strict 1-step causal attribution: action immediately
                # preceding transition
                if prev_action is not None and not self._is_inert_or_noop(prev_action):
                    cand.triggering_action = prev_action

            seen_lenses = set()
            for det in cs.detections:
                if det.k not in seen_lenses:
                    seen_lenses.add(det.k)
                    cand.N_conf_by_lens[det.k] = cand.N_conf_by_lens.get(det.k, 0) + 1

    def _to_bytes(self, key: Any) -> bytes:
        if isinstance(key, bytes):
            return key
        return hash_128(serialize_value(key))

    def _candidate_node_ids(self, cs: CrossSection) -> Set[int]:
        """Return the pattern-node IDs a CrossSection references, creating
        nodes on demand. The full TypedSignature is registered so that
        downstream MCS consumers (via the SignatureProvider) can compute
        graded similarity without re-instrumenting the graph.
        """
        node_ids: Set[int] = set()
        for det in cs.detections:
            nid = self.get_or_create_pattern_node(
                det.k, _node_signature_hash(det.sigma), signature=det.sigma
            )
            node_ids.add(nid)
        return node_ids

    def _derive_canonical_hash(self, cand: Candidate, cs: CrossSection) -> bytes:
        """§4.17: 128-bit WL canonical hash. Builder-supplied candidates
        carry their own; v30 CrossSection-based candidates fall back to a
        hash of (family, cross_section_key)."""
        if cand.canonical_hash is not None:
            return cand.canonical_hash
        return hash_128(
            serialize_tuple(
                [
                    serialize_bytes(cand.family.name.encode()),
                    serialize_bytes(self._to_bytes(cs.key)),
                ]
            )
        )

    def step_21_update_chunk_stats(self, D_t_chunk, t: int) -> None:
        self.t = t
        seen_keys = set()
        for det in D_t_chunk:
            if det.k != "Chunk":
                continue
            key = None
            for role in det.rho:
                if role[0] == "chunkKey":
                    key = role[1]
                    break
            if key is None:
                continue
            cand_key = (ConceptFamily.CHUNK, key)
            if cand_key in seen_keys:
                continue
            seen_keys.add(cand_key)
            if cand_key not in self.candidates:
                cand = Candidate(
                    K_C=tuple([key]),
                    family=ConceptFamily.CHUNK,
                    anchorKind="chunkKey",
                    combinationType="chunk",
                    T_first=t,
                    T_last=t,
                )
                self.candidates[cand_key] = cand
            else:
                cand = self.candidates[cand_key]
            cand.N_opp += 1
            cand.T_last = t
            cand.Anchors.add(det.alpha)
            cand.N_anchor = len(cand.Anchors)
            cand.last_cross_section = CrossSection(
                family="CHUNK",
                key=key,
                anchor=det.alpha,
                detections=frozenset([det]),
                roleDescriptors=det.rho,
                sourceLayers=frozenset([det.source_layer]),
            )
            cand.N_conf += 1
            cand.N_conf_by_lens["Chunk"] = cand.N_conf_by_lens.get("Chunk", 0) + 1
            if cand.cross_section_key is None:
                cand.cross_section_key = self._to_bytes(key)
            if cand.canonical_hash is None:
                cand.canonical_hash = hash_128(
                    serialize_tuple(
                        [
                            serialize_bytes(b"CHUNK"),
                            serialize_value(key),
                        ]
                    )
                )
            node_ids = self._candidate_node_ids(cand.last_cross_section)
            cand.node_ids = frozenset(node_ids)
            self.register_candidate_nodes(cand.canonical_hash, node_ids)

    # -------------------------------------------------------------------------
    # Step 23 — capacity and forgetting
    # -------------------------------------------------------------------------

    def step_23_candidate_forgetting_and_capacity(self) -> None:
        to_remove = []
        for key, cand in self.candidates.items():
            if cand.status == ConceptStatus.PENDING and (
                self.t - cand.T_last > self.params["tau_forget"]
            ):
                to_remove.append(key)
        for key in to_remove:
            self._remove_candidate(key)
        if len(self.candidates) > self.params["N_candMax"]:

            def capacity_order(cand):
                span = cand.T_last - cand.T_first if cand.T_first != -1 else 0
                age = self.t - cand.T_first if cand.T_first != -1 else 0
                hash_val = hash_64(
                    serialize_set([serialize_value(v) for v in cand.K_C])
                )
                return (cand.N_conf, span, cand.N_anchor, -age, hash_val)

            sorted_cands = sorted(self.candidates.values(), key=capacity_order)
            for cand in sorted_cands:
                if len(self.candidates) <= self.params["N_candMax"]:
                    break
                key = (cand.family, cand.K_C[0])
                self._remove_candidate(key)
        self.step_23_update_family_thresholds()
        self._prune_pattern_nodes()

    def _remove_candidate(self, cand_key: Tuple) -> None:
        cand = self.candidates.get(cand_key)
        if cand is None:
            return
        if cand.canonical_hash is not None and cand.node_ids:
            self.unregister_candidate_nodes(cand.canonical_hash, cand.node_ids)
        del self.candidates[cand_key]

    def step_23_update_family_thresholds(self) -> None:
        family_conf = defaultdict(int)
        family_opp = defaultdict(int)
        for cand in self.candidates.values():
            fam_name = cand.family.name
            family_conf[fam_name] += cand.N_conf
            family_opp[fam_name] += cand.N_opp
        for fam in ConceptFamily:
            fam_name = fam.name
            conf = family_conf.get(fam_name, 0)
            opp = family_opp.get(fam_name, 0)
            if opp == 0:
                continue
            rho_f = conf / (opp + EPSILON)
            current = self.params["family_conf_thresholds"].get(
                fam_name, self.params["theta_conf"]
            )
            delta = 0.05 * (rho_f - 0.2)
            self.params["family_conf_thresholds"][fam_name] = clamp(
                current + delta, THETA_CONF_MIN, THETA_CONF_MAX
            )

    # -------------------------------------------------------------------------
    # State hash tracking
    # -------------------------------------------------------------------------

    def set_current_state_hash(self, state_hash: int) -> None:
        self.current_state_hash = state_hash
        self.last_state_hash = state_hash

    # -------------------------------------------------------------------------
    # §4.20 existing-concept lookup
    # -------------------------------------------------------------------------

    def _find_existing_concept_by_cross_section(
        self, cs: CrossSection
    ) -> Optional[Concept]:
        family_name = cs.family if isinstance(cs.family, str) else cs.family.name
        key_b = self._to_bytes(cs.key)
        cid = self._family_key_index.get((family_name, key_b))
        if cid is None:
            return None
        concept = self.concepts.get(cid)
        if concept is None or concept.status == ConceptStatus.TOMBSTONED:
            return None
        return concept

    def lookup_by_canonical_hash(self, canonical_hash: bytes) -> Optional[Concept]:
        cid = self._canonical_hash_index.get(canonical_hash)
        if cid is None:
            return None
        c = self.concepts.get(cid)
        if c is None or c.status == ConceptStatus.TOMBSTONED:
            return None
        return c

    def lookup_by_cross_section_key(
        self, family_name: str, key_bytes: bytes
    ) -> Optional[Concept]:
        cid = self._family_key_index.get((family_name, key_bytes))
        if cid is None:
            return None
        c = self.concepts.get(cid)
        if c is None or c.status == ConceptStatus.TOMBSTONED:
            return None
        return c

    # -------------------------------------------------------------------------
    # Parent extraction / role helpers
    # -------------------------------------------------------------------------

    def _stable_role_descriptor(self, role):
        role_name, value, index, polarity = role
        if isinstance(value, int) and value in self.concepts:
            value = self.concepts[value].sigma
        return (role_name, value, index, polarity)

    def _extract_chain_links(self, cs: CrossSection) -> Optional[List[int]]:
        if cs is None:
            return None
        has_marker = any(role[0] == "causalSequence" for role in cs.roleDescriptors)
        if not has_marker:
            return None
        ante = None
        cons = None
        for role in cs.roleDescriptors:
            if role[0] == "antecedent" and isinstance(role[1], int):
                ante = role[1]
            elif role[0] == "consequent" and isinstance(role[1], int):
                cons = role[1]
        if ante is None or cons is None or ante == cons:
            return None
        causal_fams = (ConceptFamily.EVENT_CAUSAL, ConceptFamily.TEMPORAL_CHAIN)
        P_concept = self.concepts.get(ante)
        Q_concept = self.concepts.get(cons)
        if P_concept is None or Q_concept is None:
            return None
        if P_concept.family not in causal_fams or Q_concept.family not in causal_fams:
            return None
        return [ante, cons]

    # §2.7.1 role grammar: the "entity" role is the sole anchor-carrier — its
    # value is an entity index u ∈ [0, N_anchor_max), NOT a concept sourceID.
    # Entity indices and concept IDs are both small ints drawn from ranges that
    # overlap (concept IDs start at 0 via the shared _allocate_id counter), so a
    # pure value-in-V_t test would mistake an anchor for a concept parent. Every
    # other role carries either a concept sourceID (antecedent/parent/member/
    # substrateSource/...), a pattern-node id (disjoint namespace, never in
    # self.concepts), an operator/hash id (large int, never a concept id), or a
    # signature (non-int). Excluding the one anchor role by its grammar-defined
    # name is the principled disambiguation; it is NOT the incomplete family
    # whitelist that §2.8.1.2 forbids.
    _ANCHOR_ROLE_NAMES = frozenset({"entity"})

    def _extract_parents(self, cs: CrossSection) -> Set[int]:
        """§2.9.3 + §2.8.1.2 (Binding Layer Robustness Contract).

        A parent is ANY role descriptor whose VALUE maps to a concept in V_t,
        regardless of role NAME. The role name is metadata for family mapping
        only and MUST NOT gate parent extraction — that is the exact
        `_recursive_key()` defect §2.8.1.2 was written to forbid (a name
        whitelist silently dropped concept detections and blocked RECURSIVE_COMPOSITION for 70+
        steps). The single exception is the grammar-defined anchor role
        ("entity"), whose value is an entity index rather than a sourceID; it is
        excluded to prevent anchor/concept id collisions (see
        _ANCHOR_ROLE_NAMES).

        Concept IDs and pattern-node IDs share one monotonic counter but live in
        disjoint namespaces (§1.11), so `value in self.concepts` is an
        unambiguous concept test: a node id can never alias a concept id. "Raw
        detections with non-concept sourceIDs contribute no parents" (§2.9.3)
        falls out for free — a raw sourceID is not in self.concepts.
        """
        parent_ids: Set[int] = set()
        for det in cs.detections:
            for role in det.rho:
                if len(role) < 2:
                    continue
                if role[0] in self._ANCHOR_ROLE_NAMES:
                    continue
                value = role[1]
                if isinstance(value, int) and not isinstance(value, bool):
                    if value in self.concepts:
                        parent_ids.add(value)
        return parent_ids

    def _build_concept_sigma(self, fam, key, layer, role_descriptors):
        role_hashes = []
        for role in role_descriptors:
            role_name, value, index, polarity = role
            if isinstance(value, int) and value in self.concepts:
                value = self.concepts[value].sigma
            stable_role = (role_name, value, index, polarity)
            role_hashes.append(hash_64(serialize_value(stable_role)))
        return TypedSignature.opaque(
            "CONCEPT_SIGNATURE",
            hash_64(
                serialize_tuple(
                    [
                        serialize_bytes(fam.name.encode()),
                        serialize_value(key),
                        serialize_int(layer),
                        serialize_set([serialize_int(h) for h in sorted(role_hashes)]),
                    ]
                )
            ),
        )

    def _compute_layer(
        self, cs: CrossSection, parent_layers: List[int], source_layer: int
    ) -> int:
        """§3.3: layer is a function of the subgraph's edge hierarchy. v31 has
        NO order-2 edge type — structural similarity was removed from the edge
        taxonomy (it is a pure measurement; see structural_similarity.py) — so
        edge_layer = 0 for every type and the layer reduces to
        max(parent_layers) + 1 combined with source_layer. The role-descriptor
        branch is retained as the extension point for a future order-2 binder
        (the role name is the edge type's role descriptor, not the edge type
        itself), which keeps the general case explicit."""
        edge_layers = []
        for det in cs.detections:
            for role in det.rho:
                if role[0] == "structural_similarity":
                    edge_layers.append(1)
                    break
        layer_from_edges = (1 + max(edge_layers)) if edge_layers else 1
        layer_from_parents = 1 + (max(parent_layers) if parent_layers else 0)
        return max(layer_from_edges, layer_from_parents, 1 + source_layer)

    def _anchor_kind(self, alpha) -> str:
        if alpha is None:
            return ""
        if isinstance(alpha, int):
            return "entity"
        if isinstance(alpha, tuple) and len(alpha) == 2:
            return "pair"
        return "hash"

    def _reverse_role_descriptors(self, roles):
        reversed_roles = []
        for role in roles:
            role_name, value, index, polarity = role
            if role_name == "antecedent":
                reversed_roles.append(("consequent", value, index, polarity))
            elif role_name == "consequent":
                reversed_roles.append(("antecedent", value, index, polarity))
            else:
                reversed_roles.append(role)
        return tuple(sorted(reversed_roles, key=lambda r: serialize_value(r[1])))

    # -------------------------------------------------------------------------
    # Tombstones
    # -------------------------------------------------------------------------

    def step_24_tombstone_expiry(self) -> None:
        self.tombstones = [t for t in self.tombstones if t.expiry > self.t]

    def _add_tombstone(
        self,
        concept: Optional[Concept] = None,
        family_name: Optional[str] = None,
        K_C: Optional[Any] = None,
    ) -> None:
        if family_name is not None and K_C is not None:
            sig_hash = hash_128(
                serialize_tuple(
                    [
                        serialize_bytes(family_name.encode()),
                        serialize_value(K_C),
                    ]
                )
            )
        elif concept is not None and concept.canonical_hash:
            sig_hash = concept.canonical_hash
        else:
            return
        self.tombstones.append(
            TombstoneEntry(
                signature_hash=sig_hash,
                expiry=self.t + self.params["tau_tombstone"],
            )
        )
        if len(self.tombstones) > self.params["N_tombMax"]:
            self.tombstones.sort(key=lambda x: x.expiry)
            self.tombstones = self.tombstones[: self.params["N_tombMax"]]

    def _has_tombstone(self, cand: Candidate) -> bool:
        ch = cand.canonical_hash
        if ch is None:
            ch = hash_128(
                serialize_tuple(
                    [
                        serialize_bytes(cand.family.name.encode()),
                        serialize_value(cand.K_C),
                    ]
                )
            )
        return any(
            t.signature_hash == ch and t.expiry > self.t for t in self.tombstones
        )

    # -------------------------------------------------------------------------
    # Step 25 — crystallization
    # -------------------------------------------------------------------------

    def step_25_evaluate_crystallization(
        self,
        t: int,
        activation_engine=None,
        active_plans=None,
        s_t=None,
        action_t=None,
    ) -> List[Concept]:
        self.t = t
        newly_crystallized = []
        for key, cand in list(self.candidates.items()):
            if cand.status != ConceptStatus.PENDING:
                continue
            if not self._crystallization_predicate(cand, t, active_plans):
                continue
            concept = self._crystallize(cand, t, activation_engine, s_t, action_t)
            if concept is not None:
                if concept.layer >= 2 and concept.family in (
                    ConceptFamily.EVENT_CAUSAL,
                    ConceptFamily.TEMPORAL_CHAIN,
                ):
                    self.update_layer_n_evidence(concept.id, confirmed=True, count=1)
                newly_crystallized.append(concept)
        self.update_birth_ema(t)
        return newly_crystallized

    def update_birth_ema(self, t: int) -> None:
        for concept_id, concept in self.concepts.items():
            b_P = self.birth_ema.get(concept_id, 0.0)
            had_birth = any(
                c in self.concepts and self.concepts[c].T_cryst == t
                for c in concept.child_ids
            )
            b_P = (1.0 - KAPPA_BIRTH) * b_P + KAPPA_BIRTH * (1.0 if had_birth else 0.0)
            self.birth_ema[concept_id] = b_P

    def get_birth_ema(self, concept_id: int) -> float:
        return self.birth_ema.get(concept_id, 0.0)

    def update_entity_atom_map(self, entity_map) -> None:
        self.entity_atom_map = entity_map

    def get_atoms_for_entity(self, entity_id) -> Set[int]:
        return self.entity_atom_map.get(entity_id, set())

    def update_layer_n_evidence(self, concept_id: int, confirmed=True, count=1) -> None:
        concept = self.concepts.get(concept_id)
        if concept is None:
            return
        if concept.layer >= 2 and concept.family in (
            ConceptFamily.EVENT_CAUSAL,
            ConceptFamily.TEMPORAL_CHAIN,
        ):
            concept.N_opp_LN += count
            if confirmed:
                concept.N_conf_LN += count

    def _crystallization_predicate(self, cand, t, active_plans=None) -> bool:
        if cand.status != ConceptStatus.PENDING:
            return False
        if cand.N_opp < self.params["theta_opp"]:
            return False
        if cand.family == ConceptFamily.CHUNK:
            fam_thresh = self.params.get("theta_conf_chunk", THETA_CONF_CHUNK)
        else:
            fam_thresh = self.params["family_conf_thresholds"].get(
                cand.family.name, self.params["theta_conf"]
            )
        if cand.N_conf < fam_thresh:
            return False
        if cand.family == ConceptFamily.CHUNK:
            anchor_thresh = self.params.get("theta_anchor_chunk", THETA_ANCHOR_CHUNK)
        elif cand.family in (
            ConceptFamily.SPATIAL_OBJECT,
            ConceptFamily.EVENT_CAUSAL,
            ConceptFamily.TEMPORAL_CHAIN,
        ):
            anchor_thresh = 1
        else:
            anchor_thresh = self.params["theta_anchor"]
        if cand.N_anchor < anchor_thresh:
            return False
        span = cand.T_last - cand.T_first if cand.T_first != -1 else 0
        if span < self.params["theta_span"]:
            return False
        for lens, count in cand.N_conf_by_lens.items():
            if cand.family == ConceptFamily.CHUNK:
                lens_thresh = self.params.get("theta_lens_chunk", THETA_LENS_CHUNK)
            else:
                lens_thresh = self.params["lens_thresholds"].get(lens, 2)
            if count < lens_thresh:
                return False
        if cand.family == ConceptFamily.CHUNK:
            if not self._chunk_crystallization_predicate(cand, t, active_plans):
                return False
        if self._has_tombstone(cand):
            return False
        if cand.last_cross_section is not None:
            parent_ids = self._extract_parents(cand.last_cross_section)
            for pid in parent_ids:
                parent = self.concepts.get(pid)
                if parent is None:
                    return False
                if parent.status not in (
                    ConceptStatus.CRYSTALLIZED,
                    ConceptStatus.STABLE,
                ):
                    return False
                if cand.family in (
                    ConceptFamily.TEMPORAL_CHAIN,
                    ConceptFamily.RECURSIVE_COMPOSITION,
                ):
                    if parent.family in (
                        ConceptFamily.EVENT_CAUSAL,
                        ConceptFamily.TEMPORAL_CHAIN,
                    ):
                        if parent.status != ConceptStatus.STABLE:
                            return False
        return True

    def _chunk_crystallization_predicate(self, cand, t, active_plans=None) -> bool:
        if cand.last_cross_section is None:
            return False
        role_hashes = []
        for role in cand.last_cross_section.roleDescriptors:
            stable_role = self._stable_role_descriptor(role)
            role_hashes.append(hash_64(serialize_value(stable_role)))
        parent_ids = set()
        for det in cand.last_cross_section.detections:
            for role in det.rho:
                if role[0] == "member" and isinstance(role[1], int):
                    parent_ids.add(role[1])
        source_layer = max(d.source_layer for d in cand.last_cross_section.detections)
        parent_layers = [
            self.concepts[p].layer for p in parent_ids if p in self.concepts
        ]
        layer = 1 + max(max(parent_layers) if parent_layers else 0, source_layer)
        cand_sigma = TypedSignature.opaque(
            "CONCEPT_SIGNATURE",
            hash_64(
                serialize_tuple(
                    [
                        serialize_bytes(cand.family.name.encode()),
                        serialize_value(cand.K_C),
                        serialize_int(layer),
                        serialize_set([serialize_int(h) for h in sorted(role_hashes)]),
                    ]
                )
            ),
        )
        for q in self.concepts.values():
            if (
                q.family == ConceptFamily.CHUNK
                and q.status in (ConceptStatus.CRYSTALLIZED, ConceptStatus.STABLE)
                and q.sigma == cand_sigma
            ):
                return False
        for p in parent_ids:
            if p not in self.concepts:
                return False
            pc = self.concepts[p]
            if pc.status not in (
                ConceptStatus.CRYSTALLIZED,
                ConceptStatus.STABLE,
            ):
                return False
            if pc.dormant or pc.soft_dormant or pc.provisional:
                return False
        active_chunks = sum(
            1
            for c in self.concepts.values()
            if c.family == ConceptFamily.CHUNK
            and c.status in (ConceptStatus.CRYSTALLIZED, ConceptStatus.STABLE)
            and not c.dormant
            and not c.soft_dormant
        )
        if active_chunks < N_CHUNK_MAX:
            return True
        protected = set()
        if active_plans:
            for plan in active_plans:
                if hasattr(plan, "basis_snapshot"):
                    protected.update(plan.basis_snapshot)
        for c in self.concepts.values():
            if (
                c.family == ConceptFamily.CHUNK
                and c.status in (ConceptStatus.CRYSTALLIZED, ConceptStatus.STABLE)
                and not c.dormant
                and not c.soft_dormant
                and c.id not in protected
            ):
                return True
        return False

    def _crystallize(
        self, cand, t, activation_engine=None, s_t=None, action_t=None
    ) -> Optional[Concept]:
        cs = cand.last_cross_section
        if cs is None:
            return None
        concept_id = self._allocate_id()
        cand.concept_id = concept_id
        cand.status = ConceptStatus.CRYSTALLIZED

        parent_ids = self._extract_parents(cs)
        source_layer = max(d.source_layer for d in cs.detections)
        parent_layers = [
            self.concepts[p].layer for p in parent_ids if p in self.concepts
        ]
        layer = self._compute_layer(cs, parent_layers, source_layer)

        if layer > D_MAX:
            self._fail_crystallize(cand)
            return None

        active_in_layer = sum(
            1
            for c in self.concepts.values()
            if c.layer == layer
            and not c.dormant
            and not c.soft_dormant
            and c.status
            in (
                ConceptStatus.CRYSTALLIZED,
                ConceptStatus.STABLE,
                ConceptStatus.AMBIGUOUS,
            )
        )
        active_total = sum(
            1
            for c in self.concepts.values()
            if not c.dormant
            and not c.soft_dormant
            and c.status
            in (
                ConceptStatus.CRYSTALLIZED,
                ConceptStatus.STABLE,
                ConceptStatus.AMBIGUOUS,
            )
        )
        if (
            active_in_layer >= self.params["N_layerMax"]
            or active_total >= self.params["N_nodeMax"]
        ):
            self._fail_crystallize(cand)
            return None

        sigma = self._build_concept_sigma(
            cand.family, cand.K_C, layer, cs.roleDescriptors
        )
        for existing in self.concepts.values():
            if (
                existing.family == cand.family
                and existing.status
                in (
                    ConceptStatus.CRYSTALLIZED,
                    ConceptStatus.STABLE,
                    ConceptStatus.AMBIGUOUS,
                )
                and existing.sigma == sigma
            ):
                self._fail_crystallize(cand)
                return None

        evidence = []
        for i, det in enumerate(cs.detections):
            evidence.append(
                (
                    i,
                    det.sigma,
                    det.k,
                    det.source_layer,
                    det.scale,
                    self._anchor_kind(det.alpha),
                )
            )

        # §1.11: members from cross-section detections.
        members = frozenset(self._candidate_node_ids(cs))
        canonical_hash = (
            cand.canonical_hash
            if cand.canonical_hash is not None
            else self._derive_canonical_hash(cand, cs)
        )
        cross_section_key = (
            cand.cross_section_key
            if cand.cross_section_key is not None
            else self._to_bytes(cs.key)
        )

        concept = Concept(
            id=concept_id,
            family=cand.family,
            layer=layer,
            sigma=sigma,
            status=ConceptStatus.CRYSTALLIZED,
            evidence_registry=evidence,
            T_cryst=t,
            N_conf=cand.N_conf,
            N_opp=cand.N_opp,
            N_anchor=cand.N_anchor,
            parent_ids=parent_ids,
            role_descriptors=cs.roleDescriptors,
            members=members,
            canonical_hash=canonical_hash,
            cross_section_key=cross_section_key,
        )
        concept.n_conf_from_observation = cand.N_conf
        concept.n_conf_from_reasoning = 0
        concept.last_structural_progress = t
        self.concept_revision += 1  # [Batch D / P1-1] crystallization
        concept.arity = len(cs.detections)
        concept.anchor = cs.anchor
        concept.Anchors = set(getattr(cand, "Anchors", set()))

        if cand.family in (
            ConceptFamily.EVENT_CAUSAL,
            ConceptFamily.TEMPORAL_CHAIN,
        ):
            effect_id = None
            for det in cs.detections:
                for role in det.rho:
                    if role[0] == "transitionEffect" and isinstance(role[1], int):
                        effect_id = role[1]
                        break
                if effect_id is not None:
                    break
            concept.effect_id = effect_id
            concept.effect_is_concept = (
                isinstance(effect_id, int) and effect_id in self.concepts
            )
            concept.enabling_pre_states = (
                set(cand.enabling_pre_states)
                if cand.enabling_pre_states
                else ({s_t} if s_t is not None else set())
            )
            # [ACTION FILTER] `cand.triggering_action` is populated in
            # `step_21_update_candidate_stats` and has already passed
            # `_is_inert_or_noop`. The `action_t` fallback comes
            # straight from the pipeline's previous action and has NOT
            # been filtered — without this guard a no-op, probe, or
            # goal-reference step that happened to coincide with an
            # effect onset would bind as the causal action and mint a
            # spurious concept. Note the predicate's polarity: it
            # returns True for inert actions, so we keep `action_t`
            # only when it is NOT inert.
            cand_act = (
                cand.triggering_action
                if cand.triggering_action is not None
                else (
                    action_t
                    if not self._is_inert_or_noop(action_t)
                    else None
                )
            )
            if effect_id is not None and cand_act is not None:
                concept.action = cand_act
            else:
                concept.action = None

        if cand.family == ConceptFamily.TEMPORAL_CHAIN:
            chain_links = self._extract_chain_links(cs)
            if chain_links is None:
                self._fail_crystallize(cand)
                return None
            concept.chain_links = chain_links
            first = self.concepts.get(chain_links[0])
            last = self.concepts.get(chain_links[-1])
            if first is not None and getattr(first, "action", None) is not None:
                concept.action = first.action
            if last is not None:
                concept.effect_id = getattr(last, "effect_id", concept.effect_id)
                concept.effect_is_concept = getattr(last, "effect_is_concept", False)

        # Register everywhere.
        self.concepts[concept_id] = concept
        self._family_key_index[(cand.family.name, cross_section_key)] = concept_id
        if canonical_hash not in self._canonical_hash_index:
            self._canonical_hash_index[canonical_hash] = concept_id
        for nid in members:
            self._node_to_concepts[nid].add(concept_id)
        # [I.4] Register the member-set key (may repeat: several concepts
        # can share a member set across families/layers).
        self._concepts_by_member_set[frozenset(members)].append(concept_id)

        if concept.action is not None:
            concept.actions.add(concept.action)
            if concept.effect_id is not None:
                concept.effect_ids_by_action[concept.action] = concept.effect_id
            self.action_to_concepts[concept.action].add(concept_id)

        for p in parent_ids:
            if p in self.concepts:
                self.concepts[p].child_ids.add(concept_id)
                self._instantiate_concept_edge(p, concept_id, t)

        if cand.family == ConceptFamily.CHUNK:
            concept.chunk_members = set(parent_ids)
            concept.chunk_depth = 1 + max(
                [
                    self.concepts[p].chunk_depth
                    for p in parent_ids
                    if p in self.concepts
                    and self.concepts[p].family == ConceptFamily.CHUNK
                ],
                default=0,
            )

        # [SCHEMA MULTI-BINDING §1.11] The scalar binding is already in
        # `actions` and `effect_ids_by_action` above; nothing else to seed
        # for the first instance. Subsequent instances accumulate via
        # step_21_update_candidate_stats.
        cand_key = (cand.family, cand.K_C[0])
        if cand_key in self.candidates:
            self._remove_candidate(cand_key)

        return concept

    def _fail_crystallize(self, cand: Candidate) -> None:
        cand.blockedCount += 1
        cand.status = (
            ConceptStatus.TOMBSTONED
            if cand.blockedCount >= self.params["theta_blockMax"]
            else ConceptStatus.PENDING
        )

    def _instantiate_concept_edge(self, p: int, c: int, t: int) -> None:
        """Concept-level ancestry edge. Its signal is derived from activation
        in step_28_update_edges (§1.12 CO_ACTIVATION signal); initialized to
        zero weight so the first EMA observation sets it."""
        key = (p, c, EdgeType.CO_ACTIVATION)
        if key in self.edges:
            return
        self.edges[key] = Edge(
            parent=p,
            child=c,
            weight=0.0,
            edge_type=EdgeType.CO_ACTIVATION,
            lag=0,
            pending_signal=0.0,
            observed_at=-1,
            initialized=False,
        )
        self._outgoing[p].add((c, EdgeType.CO_ACTIVATION))
        self._incoming[c].add((p, EdgeType.CO_ACTIVATION))
        self._topo_version += 1
        self.topology_revision += 1  # [Batch D / P1-1]

    # -------------------------------------------------------------------------
    # Witness and status (§4.17 witness path)
    # -------------------------------------------------------------------------

    def step_28_update_witness_and_status(self, t: int) -> None:
        self.t = t
        sym_index = defaultdict(list)
        for c in self.concepts.values():
            if (
                c.status in (ConceptStatus.CRYSTALLIZED, ConceptStatus.STABLE)
                and c.role_descriptors
            ):
                rev = self._reverse_role_descriptors(c.role_descriptors)
                key = serialize_set([serialize_value(role) for role in rev])
                sym_index[key].append((c.id, c.N_conf))

        for concept in self.concepts.values():
            if concept.status == ConceptStatus.CRYSTALLIZED:
                if concept.N_opp > 0:
                    concept.Reliability = concept.N_conf / (concept.N_opp + EPSILON)
                N_conf = concept.N_conf
                if not concept.role_descriptors or concept.family in (
                    ConceptFamily.SPATIAL_OBJECT,
                    ConceptFamily.CHUNK,
                ):
                    N_sym = 0
                else:
                    rev = self._reverse_role_descriptors(concept.role_descriptors)
                    rev_key = serialize_set([serialize_value(role) for role in rev])
                    N_sym = sum(
                        n for cid, n in sym_index.get(rev_key, []) if cid != concept.id
                    )
                concept.N_sym = N_sym
                rho_sym = N_sym / (concept.N_opp + EPSILON) if concept.N_opp > 0 else 0
                Dec = (N_conf - N_sym) / (N_conf + N_sym + EPSILON)
                Dec_star = Dec - ETA_ARITY * (len(concept.parent_ids) - 2)

                if concept.family in (
                    ConceptFamily.EVENT_CAUSAL,
                    ConceptFamily.TEMPORAL_CHAIN,
                ):
                    cand_act = getattr(concept, "action", None)
                    eff_id = getattr(concept, "effect_id", None)
                    if cand_act is not None and eff_id is not None:
                        if self.prediction_module is not None:
                            bar_p_a = self.prediction_module.causal_predict(
                                cand_act, eff_id
                            )
                            bar_p_not_a = 1.0 - bar_p_a
                            n_cf = min(
                                self.prediction_module.C_opp.get(cand_act, 0.0),
                                self.prediction_module.C_opp_not.get(cand_act, 0.0),
                            )
                        else:
                            bar_p_a = self.prev_bar_p_a.get((cand_act, eff_id), 0.0)
                            bar_p_not_a = self.prev_bar_p_not_a.get(
                                (cand_act, eff_id), 0.0
                            )
                            n_cf = min(
                                self.C_opp.get(cand_act, 0.0),
                                self.C_opp_not.get(cand_act, 0.0),
                            )
                        if (
                            n_cf >= self.N_cf_min
                            and bar_p_not_a >= 0.60
                            and bar_p_a <= 0.20
                        ):
                            Dec_star -= ETA_ARITY

                anchor_thresh = (
                    1
                    if concept.family
                    in (
                        ConceptFamily.SPATIAL_OBJECT,
                        ConceptFamily.EVENT_CAUSAL,
                        ConceptFamily.TEMPORAL_CHAIN,
                    )
                    else self.params["theta_anchor"]
                )
                if (
                    N_conf >= self.params["theta_witness"]
                    and Dec_star >= self.params["theta_decisive"]
                    and rho_sym <= self.params["rho_sym_max"]
                    and concept.N_anchor >= anchor_thresh
                ):
                    if concept.layer >= 2 and concept.family in (
                        ConceptFamily.EVENT_CAUSAL,
                        ConceptFamily.TEMPORAL_CHAIN,
                    ):
                        if (
                            concept.N_opp_LN >= N_LN
                            and (concept.N_conf_LN / (concept.N_opp_LN + EPSILON))
                            >= THETA_LN
                        ):
                            concept.status = ConceptStatus.STABLE
                    else:
                        concept.status = ConceptStatus.STABLE
                elif (
                    N_conf >= self.params["theta_witness"]
                    and Dec_star < self.params["theta_decisive"]
                ):
                    concept.status = ConceptStatus.AMBIGUOUS
                    concept.ambiguityCount += 1
                    if concept.ambiguityCount > self.params["N_maxAmbiguity"] or (
                        self.t - concept.T_cryst > TAU_AMBIGUITY
                    ):
                        concept.status = ConceptStatus.REJECTED
                        self._add_tombstone(concept=concept)

            if concept.status == ConceptStatus.CRYSTALLIZED and concept.layer >= 2:
                if concept.family in (
                    ConceptFamily.EVENT_CAUSAL,
                    ConceptFamily.TEMPORAL_CHAIN,
                ):
                    for p_id in concept.parent_ids:
                        parent = self.concepts.get(p_id)
                        if parent and parent.last_activation >= self.t - 1:
                            concept.N_opp_LN += 1
                            if parent.last_activation == self.t:
                                concept.N_conf_LN += 1
                            break
            elif concept.status == ConceptStatus.AMBIGUOUS:
                if concept.ambiguityCount > self.params["N_maxAmbiguity"] or (
                    self.t - concept.T_cryst > TAU_AMBIGUITY
                ):
                    concept.status = ConceptStatus.REJECTED
                    self._add_tombstone(concept=concept)

    # -------------------------------------------------------------------------
    # Dormancy and capacity
    # -------------------------------------------------------------------------

    def step_35_dormancy_sweep(
        self, activation_engine, fertility_engine, planning_engine=None
    ) -> None:
        for concept in self.concepts.values():
            # [ATOMIC_ACTION EXEMPTION] Atomic actions are seeded
            # primitives, not learned structures. They never activate
            # (they are not detections), never have children, and would
            # otherwise exhaust dormancy and be tombstoned after
            # tau_dormant steps. Keep them permanently live.
            if concept.family == ConceptFamily.ATOMIC_ACTION:
                concept.exhausted = 0
                concept.inactive = 0
                concept.dormant = False
                concept.soft_dormant = False
                continue
            if concept.status in (
                ConceptStatus.STABLE,
                ConceptStatus.CRYSTALLIZED,
            ):
                A_P = activation_engine.get_activation(concept.id, self.t)
                has_new_child = any(
                    c in self.concepts and self.concepts[c].T_cryst == self.t
                    for c in concept.child_ids
                )
                has_plan_success = False
                if planning_engine is not None:
                    has_plan_success = planning_engine.has_plan_success(
                        concept.id, self.t
                    )
                contrib = A_P or has_new_child or has_plan_success
                if contrib:
                    concept.exhausted = 0
                    concept.inactive = 0
                else:
                    concept.exhausted += 1
                    concept.inactive += 1
                if concept.exhausted >= self.params["tau_dormant"]:
                    concept.dormant = True
                    concept.soft_dormant = False
                else:
                    concept.dormant = False
                RF = fertility_engine.get_RF(concept.id) if fertility_engine else 0
                if (
                    concept.inactive >= self.params["tau_softDormant"]
                    and RF < self.params["RF_softMin"]
                ):
                    concept.soft_dormant = True
                else:
                    concept.soft_dormant = False

    def step_36_capacity_eviction(self, fertility_engine, planning_engine=None) -> None:
        protected = set()
        if planning_engine is not None:
            for plan in getattr(planning_engine, "active_plans", []):
                if hasattr(plan, "basis_snapshot"):
                    protected.update(plan.basis_snapshot)
        layer_counts = defaultdict(int)
        active_nodes = []
        for concept in self.concepts.values():
            if concept.dormant or concept.soft_dormant:
                continue
            if concept.status not in (
                ConceptStatus.CRYSTALLIZED,
                ConceptStatus.STABLE,
                ConceptStatus.AMBIGUOUS,
            ):
                continue
            active_nodes.append(concept)
            layer_counts[concept.layer] += 1
        for layer, count in layer_counts.items():
            if count > self.params["N_layerMax"]:
                layer_nodes = [c for c in active_nodes if c.layer == layer]
                layer_nodes.sort(
                    key=lambda c: (
                        fertility_engine.get_RF(c.id) if fertility_engine else 0,
                        c.id,
                    )
                )
                excess = count - self.params["N_layerMax"]
                evicted = 0
                for node in layer_nodes:
                    if evicted >= excess:
                        break
                    if node.id in protected:
                        continue
                    self._evict_node(node)
                    evicted += 1
        active_nodes = [
            c
            for c in self.concepts.values()
            if not c.dormant
            and not c.soft_dormant
            and c.status
            in (
                ConceptStatus.CRYSTALLIZED,
                ConceptStatus.STABLE,
                ConceptStatus.AMBIGUOUS,
            )
        ]
        active_count = len(active_nodes)
        if active_count > self.params["N_nodeMax"]:
            dormant_nodes = [
                c
                for c in self.concepts.values()
                if (c.dormant or c.soft_dormant) and c.id not in protected
            ]
            excess = active_count - self.params["N_nodeMax"]
            evicted = 0
            for node in dormant_nodes:
                if evicted >= excess:
                    break
                self._evict_node(node)
                evicted += 1
            active_nodes = [
                c
                for c in self.concepts.values()
                if not c.dormant
                and not c.soft_dormant
                and c.status
                in (
                    ConceptStatus.CRYSTALLIZED,
                    ConceptStatus.STABLE,
                    ConceptStatus.AMBIGUOUS,
                )
            ]
            active_count = len(active_nodes)
            if active_count > self.params["N_nodeMax"]:
                excess = active_count - self.params["N_nodeMax"]
                active_nodes.sort(
                    key=lambda c: (
                        fertility_engine.get_RF(c.id) if fertility_engine else 0,
                        c.id,
                    )
                )
                evicted = 0
                for node in active_nodes:
                    if evicted >= excess:
                        break
                    if node.id in protected:
                        continue
                    self._evict_node(node)
                    evicted += 1

    def _evict_node(self, concept: Concept) -> None:
        concept.status = ConceptStatus.TOMBSTONED
        # [Batch D / P1-1] eviction is a concept- and topology-level change.
        self.concept_revision += 1
        self.topology_revision += 1
        # [SCHEMA MULTI-BINDING §1.11] Clean up all bound actions, not
        # just the scalar. Under multi-binding, action_to_concepts holds
        # entries for every action registered on the concept.
        actions_to_clean = set(getattr(concept, "actions", set()))
        if getattr(concept, "action", None) is not None:
            actions_to_clean.add(concept.action)
        for a in actions_to_clean:
            self.action_to_concepts[a].discard(concept.id)
        if concept.cross_section_key:
            self._family_key_index.pop(
                (concept.family.name, concept.cross_section_key), None
            )
        if concept.canonical_hash:
            self._canonical_hash_index.pop(concept.canonical_hash, None)
        for nid in list(concept.members):
            s = self._node_to_concepts.get(nid)
            if s is not None:
                s.discard(concept.id)
                if not s:
                    self._node_to_concepts.pop(nid, None)
        # [I.4] Drop the member-set entry so an evicted concept cannot be
        # returned by concepts_by_member_set (the status filter in
        # Consolidator._family_of_subgraph is a second line of defence).
        if concept.members:
            _ms_key = frozenset(concept.members)
            _ms_lst = self._concepts_by_member_set.get(_ms_key)
            if _ms_lst is not None:
                try:
                    _ms_lst.remove(concept.id)
                except ValueError:
                    pass
                if not _ms_lst:
                    del self._concepts_by_member_set[_ms_key]
        # Remove concept-level edges in both directions.
        to_remove = []
        for dst, et in list(self._outgoing.get(concept.id, set())):
            to_remove.append((concept.id, dst, et))
        for src, et in list(self._incoming.get(concept.id, set())):
            to_remove.append((src, concept.id, et))
        for key in to_remove:
            self._remove_edge(key)
        for p in list(concept.parent_ids):
            if p in self.concepts:
                self.concepts[p].child_ids.discard(concept.id)
        for c in list(concept.child_ids):
            if c in self.concepts:
                self.concepts[c].parent_ids.discard(concept.id)
        self._add_tombstone(concept=concept)
        self.birth_ema.pop(concept.id, None)

    # -------------------------------------------------------------------------
    # Pruning test
    # -------------------------------------------------------------------------

    def step_pruning(self, activation_engine, fertility_engine, t) -> None:
        """[Phase 1 Decision 1] Three-signal direct pruning.

        A concept is prunable iff ALL hold:
          (1) status == STABLE
          (2) dormant == True
          (3) family != ATOMIC_ACTION
          (4) A_bar(c) < THETA_A
          (5) RF(c) < THETA_RF
          (6) no child crystallization in the last TAU_RECENT steps

        §P3 instrumentation samples BEFORE the (4)/(5) cuts so the
        calibration denominator includes saved-by-high-RF concepts.
        Per-step cap (MAX_PRUNE_PER_STEP) prevents mass extinction.
        """
        evicted_this_step = 0
        for concept in list(self.concepts.values()):
            if evicted_this_step >= P.MAX_PRUNE_PER_STEP:
                break
            if concept.status != ConceptStatus.STABLE:
                continue
            if not concept.dormant:
                continue
            if concept.family == ConceptFamily.ATOMIC_ACTION:
                continue

            rf_val = float(getattr(concept, "RF", 0.0))
            ef_val = float(getattr(concept, "EF_combined", 0.0))
            self.diagnostics["pruning_candidates_scored"] += 1
            self.diagnostics["pruning_rf_sum"] += rf_val
            self.diagnostics["pruning_ef_sum"] += ef_val
            if rf_val > self.diagnostics["pruning_rf_max"]:
                self.diagnostics["pruning_rf_max"] = rf_val
            if ef_val > self.diagnostics["pruning_ef_max"]:
                self.diagnostics["pruning_ef_max"] = ef_val

            spec_val = fertility_engine.get_specificity(concept.id)
            self.diagnostics["pruning_spec_sum"] = (
                self.diagnostics.get("pruning_spec_sum", 0.0) + spec_val
            )
            if spec_val > self.diagnostics.get("pruning_spec_max", 0.0):
                self.diagnostics["pruning_spec_max"] = spec_val
            if spec_val < self.diagnostics.get("pruning_spec_min", 1.0):
                self.diagnostics["pruning_spec_min"] = spec_val

            A_bar = activation_engine.get_average_activation(concept.id)
            if A_bar >= THETA_A:
                continue
            if rf_val >= THETA_RF:
                continue
            # [§P3] Fertility-protected pruning: skip concepts that are
            # currently fertile. ef_val is already read above for
            # instrumentation, so this costs one comparison.
            if ef_val >= THETA_FERTILE_PROTECT:
                continue
            # [§P3] Specificity gate: protect concepts whose specificity is
            # at or above the threshold (they fire selectively, not hub-like).
            if spec_val >= THETA_PRUNE_SPECIFICITY:
                continue
            recent_child = any(
                c in self.concepts
                and self.concepts[c].T_cryst > t - P.TAU_RECENT
                for c in concept.child_ids
            )
            if recent_child:
                continue

            self._evict_node(concept)
            self.diagnostics["pruning_evictions"] += 1
            evicted_this_step += 1

    def get_concept(self, id) -> Optional[Concept]:
        return self.concepts.get(id)

    def get_concept_sigma(self, concept_id) -> Optional[TypedSignature]:
        concept = self.concepts.get(concept_id)
        return concept.sigma if concept else None

    def get_active_concepts(self) -> Set[int]:
        return {
            id
            for id, c in self.concepts.items()
            if not c.dormant
            and not c.soft_dormant
            and c.status
            in (
                ConceptStatus.CRYSTALLIZED,
                ConceptStatus.STABLE,
                ConceptStatus.AMBIGUOUS,
            )
        }

    def get_candidate(self, key) -> Optional[Candidate]:
        return self.candidates.get(key)

    def get_tombstone_hash(self, sig_hash) -> bool:
        return any(
            t.signature_hash == sig_hash and t.expiry > self.t for t in self.tombstones
        )

    def get_ancestors(self, concept_id) -> Set[int]:
        concept = self.concepts.get(concept_id)
        if not concept:
            return set()
        ancestors = set()
        stack = list(concept.parent_ids)
        while stack:
            pid = stack.pop()
            if pid in ancestors:
                continue
            ancestors.add(pid)
            parent = self.concepts.get(pid)
            if parent:
                stack.extend(parent.parent_ids)
        return ancestors

    # NOTE: get_atoms_for_entity is defined once, beside its writer
    # update_entity_atom_map (§ pattern-node/entity section). A duplicate
    # definition here previously shadowed it; removed as dead code.

    def get_contributing_lenses(self, concept_id) -> Set[str]:
        concept = self.concepts.get(concept_id)
        return {evid[2] for evid in concept.evidence_registry} if concept else set()

    def get_source_ids(self, detections) -> Set[int]:
        ids = set()
        for d in detections:
            for role in d.rho:
                if len(role) >= 2 and isinstance(role[1], int):
                    ids.add(role[1])
        return ids

    # -------------------------------------------------------------------------
    # Operators
    # -------------------------------------------------------------------------

    def step_40_update_operator_promotion(self, promotion_candidates, t) -> None:
        self.t = t
        promoted_count = sum(1 for c in self.concepts.values() if c.promoted)
        for concept_id, eligible, score in promotion_candidates:
            concept = self.concepts.get(concept_id)
            if not concept:
                continue
            if not eligible:
                concept.promo_count = 0
                continue
            if (
                concept.op_demoted_at != -1
                and t < concept.op_demoted_at + TAU_OP_COOLDOWN
            ):
                continue
            if score >= THETA_PROMO_OP:
                concept.promo_count = min(concept.promo_count + 1, TAU_PROMO)
            else:
                concept.promo_count = max(concept.promo_count - 1, 0)
            if concept.promo_count >= TAU_PROMO and not concept.promoted:
                if promoted_count < N_OP_MAX:
                    concept.promoted = True
                    concept.operator_active = True
                    concept.operator_threshold = THETA_HYST_OP
                    concept.operator_fatigue = 0.0
                    concept.operator_contextual_modulation = 0.0
                    promoted_count += 1
            if concept.promoted and concept.status in (
                ConceptStatus.REJECTED,
                ConceptStatus.TOMBSTONED,
            ):
                concept.promoted = False
                concept.op_demoted_at = t
                concept.promo_count = 0
                concept.operator_active = False
                concept.inactive_L_P = 0
                promoted_count -= 1

    def step_40_update_operator_demotion(self, operator_emissions, t) -> None:
        self.t = t
        for concept_id, concept in self.concepts.items():
            if not concept.promoted:
                continue
            emissions = operator_emissions.get(concept_id, 0)
            if emissions > 0:
                concept.inactive_L_P = 0
            else:
                concept.inactive_L_P = min(TAU_OP_INACTIVE, concept.inactive_L_P + 1)
            if concept.inactive_L_P >= TAU_OP_INACTIVE or concept.status in (
                ConceptStatus.REJECTED,
                ConceptStatus.TOMBSTONED,
            ):
                concept.promoted = False
                concept.op_demoted_at = t
                concept.promo_count = 0
                concept.operator_active = False
                concept.inactive_L_P = 0

    # -------------------------------------------------------------------------
    # §1.12 Edge weight update (type-specific EMA)
    # -------------------------------------------------------------------------

    def step_28_update_edges(self, activation_engine, t) -> None:
        # [Hebbian 1b] Collect the set of pattern node ids that belong to
        # any concept that crystallized at the current step. A concept
        # whose T_cryst equals t was crystallized this step; its members
        # are pattern node ids. Concept edges (endpoints are concept
        # ids, not pattern ids) never match — only pattern edges are
        # eligible for stamping, which is correct: crystallization is a
        # pattern-level event.
        #
        # Union across concepts (single set, single pass over edges) is
        # O(E) total. The naive per-concept loop would be O(C × E) and
        # E can reach N_EDGE_MAX = 100,000.
        stamped_member_union: Set[int] = set()
        for concept in self.concepts.values():
            if getattr(concept, "T_cryst", -1) == t:
                stamped_member_union.update(concept.members)

        self.t = t
        beta_edge = P.BETA_W
        w_seq = getattr(P, "W_SEQ", 20)
        to_remove = []

        for key, edge in list(self.edges.items()):
            observed = edge.observed_at == t
            if observed:
                edge.inactive_time = 0
            else:
                edge.inactive_time += 1

            if self._is_concept_edge(edge):
                signal = self._concept_edge_signal(edge, activation_engine, t, w_seq)
                edge.CoAct_EMA = (
                    1 - P.KAPPA_COACT
                ) * edge.CoAct_EMA + P.KAPPA_COACT * signal
                if signal > 0:
                    edge.time_since_coact = 0
                else:
                    edge.time_since_coact += 1
            else:
                # Binder-emitted pattern-node edge: signal is the value
                # recorded by upsert_edge this step (or 0 if not observed).
                signal = edge.pending_signal if observed else 0.0
                edge.CoAct_EMA = (
                    1 - P.KAPPA_COACT
                ) * edge.CoAct_EMA + P.KAPPA_COACT * signal
                # [Batch E / SC-12] Recency bookkeeping, mirroring the
                # concept-edge branch. Feeds mapper._edge_included clause
                # (3). No effect on weight; no effect on spreading (RULE 3).
                if signal > 0:
                    edge.time_since_coact = 0
                else:
                    edge.time_since_coact += 1

            # [Hebbian 1b] Structural-evidence EMA. Bumped when both
            # endpoints of the edge are members of a concept that
            # crystallized this step; decayed otherwise. Runs for all edges
            # (including non-boosted types) — evidence is a fact about the
            # edge's history, not tied to whether the boost is applied.
            if (edge.parent in stamped_member_union
                    and edge.child in stamped_member_union):
                edge.structural_evidence = (
                    P.BETA_E * edge.structural_evidence + (1.0 - P.BETA_E)
                )
            elif edge.structural_evidence > 0.0:
                edge.structural_evidence *= P.BETA_E

            # [Hebbian 1a] Firing-signal boost. Clamp defensively to [0, 1]:
            # the equilibrium bound w* ≤ (1 + ETA_E) · s assumes a
            # normalized signal; the guard keeps it robust to any future
            # change in _concept_edge_signal's contract.
            raw_signal = signal
            if raw_signal < 0.0:
                raw_signal = 0.0
            elif raw_signal > 1.0:
                raw_signal = 1.0

            if edge.edge_type in HEBBIAN_BOOST_TYPES:
                effective_signal = raw_signal * (
                    1.0 + P.ETA_E * edge.structural_evidence
                )
            else:
                effective_signal = raw_signal

            if not edge.initialized:
                edge.weight = effective_signal
                edge.initialized = True
            else:
                edge.weight = (
                    beta_edge * edge.weight
                    + (1.0 - beta_edge) * effective_signal
                )

            if edge.inactive_time >= P.TAU_EDGE_INACTIVE:
                to_remove.append(key)

        for key in to_remove:
            self._remove_edge(key)

        # [Hebbian 1c] Evidence-distribution diagnostics. Feeds the
        # canary JSON via the concept_graph_diag export path added
        # earlier this session. Observability only — no behavior.
        n_at_max = 0
        n_positive = 0
        sum_evidence = 0.0
        for e in self.edges.values():
            if e.weight >= P.W_MAX - 1e-9:
                n_at_max += 1
            if e.structural_evidence > 1e-9:
                n_positive += 1
            sum_evidence += e.structural_evidence
        self.diagnostics["edges_at_weight_max"] = n_at_max
        self.diagnostics["edges_with_positive_evidence"] = n_positive
        self.diagnostics["mean_structural_evidence"] = (
            sum_evidence / max(1, len(self.edges))
        )

        # Edge weights, inactive_time, and topology have all settled for this
        # step. Bump the revision so cached active-edge predicates downstream
        # (FertilityEngine._is_active_edge memo) are recomputed against the
        # fresh state on the next step30 pass.
        self.edge_revision += 1

    def _is_concept_edge(self, edge: Edge) -> bool:
        """A concept-level edge has both endpoints in self.concepts. Pattern-
        node edges have endpoints in _pattern_node_key_by_id."""
        return edge.parent in self.concepts and edge.child in self.concepts

    def _concept_edge_signal(
        self, edge: Edge, activation_engine, t: int, w_seq: int
    ) -> float:
        p = edge.parent
        c = edge.child
        et = edge.edge_type
        if et == EdgeType.TRANSITION:
            parent = self.concepts.get(p)
            child = self.concepts.get(c)
            if parent is None or child is None:
                return 0.0
            return 1.0 if (parent.onset_prev == 1 and child.onset == 1) else 0.0
        # CO_ACTIVATION and legacy v30 signal.
        child = self.concepts.get(c)
        parent = self.concepts.get(p)
        onset_c = activation_engine.get_onset(c, t)
        if child is not None and child.family == ConceptFamily.TEMPORAL_CHAIN:
            parent_recent = (
                parent is not None
                and (t - getattr(parent, "last_activation", -999)) <= w_seq
            )
            return 1.0 if (parent_recent and onset_c == 1) else 0.0
        A_p = activation_engine.get_activation(p, t)
        return 1.0 if (A_p == 1 and onset_c == 1) else 0.0

    # -------------------------------------------------------------------------
    # Progress and disablement
    # -------------------------------------------------------------------------

    def step_34_update_structural_progress(self, progress_events, t) -> None:
        self.t = t
        for concept_id, event_type in progress_events:
            concept = self.concepts.get(concept_id)
            if not concept:
                continue
            if event_type in (
                "child_crystallization",
                "plan_success",
                "confirmed_subgoal",
            ):
                concept.last_structural_progress = t
                concept.fail_streak = 0
                # [Batch D / P1-4 SC-14] plan_success resets the blame streak.
                concept.plan_blame_streak = 0
            elif event_type == "progress_failure":
                # §2.31.6: failStreak(P,t+1) = min(τ_revert, failStreak(P,t)+1).
                # The cap is normative — an uncapped counter is unbounded state
                # and would make the "streak >= τ_revert" disable test fire on a
                # value the spec never allows the streak to exceed.
                concept.fail_streak = min(TAU_REVERT, concept.fail_streak + 1)
