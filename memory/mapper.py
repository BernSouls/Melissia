"""
mapper.py — Search Function 3: Reasoner / Mapper (v31 §6).

Answers two questions per query:
  - "What is this?"   → extract the situation's core, match against cores.
  - "Have I seen it?" → return what the matched core enables.

The Mapper does NOT search. It extracts, matches, returns. No pathfinding,
no state-space navigation. The Executor performs pathfinding when a
transfer's consequence requires action sequencing.

Determinism: every comparison, tie-break, and ordering is by canonical hash
(bytes, lexicographic) or by canonical serialized bytes. Same input
sequence produces byte-identical output, run to run.

Dependencies:
  - primitives: serialize_int, serialize_bytes, serialize_tuple, hash_128
  - sensory_floor: TypedSignature
  - binder: EdgeType, _node_signature_hash
  - concept_graph: Concept, ConceptStatus
  - prediction: Subgraph
  - consolidator: CoreEntry, CoreKind, CoreLibrary, CorePayload_Action,
                  CorePayload_Category, SignatureProvider, mcs_score,
                  _structural_intersection, _materialize, _subgraph_wl_hash,
                  _subgraph_size
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from dataclasses import replace as _dc_replace
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Set, Tuple

from core.hashing import (  # noqa: F401
    clamp,
    hash_128,
    serialize_int,
    serialize_tuple,
    serialize_value,
)

from sensory_floor import TypedSignature
from core.hashing import _node_signature_hash
from core.types import EdgeType, UNDIRECTED_EDGE_TYPES
from core.graph_types import Subgraph, anchor_positions_for
from memory.content import ContentSignatureStore, build_content_subgraph

from consolidator import (
    CategoryLabel,
    CoreEntry,
    CoreKind,
    CoreLibrary,
    CorePayload_Action,
    CorePayload_Category,
    CoreStatus,
    SignatureProvider,
    mcs_score,
    _materialize,
    _structural_intersection,
    _subgraph_size,
    _subgraph_wl_hash,
)


# =============================================================================
# PARAMETERS (§6.8) — bound from parameters.py (single source of truth)
# =============================================================================

import parameters as P

H_SITUATION: int = P.H_SITUATION
THETA_SITUATION_EDGE: float = P.THETA_SITUATION_EDGE
MAX_SITUATION_SIZE: int = P.MAX_SITUATION_SIZE

THETA_CONTAINMENT: float = P.THETA_CONTAINMENT
TAU_TRANSFER: float = P.TAU_TRANSFER
QUERY_CACHE_TTL: int = P.QUERY_CACHE_TTL
THETA_CATEGORY_MIN_POSITIVES: int = P.THETA_CATEGORY_MIN_POSITIVES

# §5.13: CORE_SIZE_CAP is stated as "20 nodes / 40 edges". The Mapper's
# compression-skip gate (§6.3) uses the node-count component only; the
# edge-count component is enforced by Consolidator at extraction time and
# is not consulted here. See _extract_query_core_action for the full
# rationale on the §6.3 vs §6.4 units disagreement.
CORE_SIZE_CAP_NODES: int = P.CORE_SIZE_CAP_NODES

# §6.5 Level-B matching thresholds.
THETA_MCS_BY_FAMILY: Dict[str, float] = P.THETA_MCS_BY_FAMILY
THETA_MCS_DEFAULT: float = P.THETA_MCS_DEFAULT
THETA_MCS_CATEGORY: float = P.THETA_MCS_CATEGORY

# [Step 6a] Evidence matching thresholds
THETA_EVIDENCE_BY_FAMILY: Dict[str, float] = P.THETA_EVIDENCE_BY_FAMILY
THETA_EVIDENCE_DEFAULT: float = P.THETA_EVIDENCE_DEFAULT
THETA_EVIDENCE_CATEGORY: float = P.THETA_EVIDENCE_CATEGORY

# Query-time label-history bound (used for discriminative compression
# at query time when >= THETA_CATEGORY_MIN_POSITIVES windows accumulate).
LABEL_HISTORY_MAXLEN: int = P.LABEL_HISTORY_MAXLEN

# Bounded gap queue (§6.6 GapSignal).
GAP_QUEUE_MAX: int = P.GAP_QUEUE_MAX

THETA_IDENTIFY_DELTA: float = P.THETA_IDENTIFY_DELTA
THETA_FIRED_COVERAGE: float = P.THETA_FIRED_COVERAGE
MIN_MATCHED_ELEMENTS: int = P.MIN_MATCHED_ELEMENTS
MIN_MATCHED_EDGES: int = P.MIN_MATCHED_EDGES
MAX_IDENTITIES_PER_BUCKET: int = P.MAX_IDENTITIES_PER_BUCKET

H_FIRE: int = P.H_FIRE
LAMBDA_ACT: float = P.LAMBDA_ACT
THETA_SITUATION: float = P.THETA_SITUATION
W_SITUATION_RECENCY: int = P.W_SITUATION_RECENCY
TAU_SITUATION_COACT: int = P.TAU_SITUATION_COACT
KAPPA_REL: float = P.KAPPA_REL

# Import-time guards (raise, not assert, so -O cannot disable them).
if THETA_IDENTIFY_DELTA < 0.0:
    raise ValueError(
        f"THETA_IDENTIFY_DELTA ({THETA_IDENTIFY_DELTA}) must be >= 0."
    )
if not (0.0 < THETA_FIRED_COVERAGE <= 1.0):
    raise ValueError(
        f"THETA_FIRED_COVERAGE ({THETA_FIRED_COVERAGE}) must be in (0, 1]."
    )
if MAX_IDENTITIES_PER_BUCKET < 1:
    raise ValueError(
        f"MAX_IDENTITIES_PER_BUCKET ({MAX_IDENTITIES_PER_BUCKET}) must be >= 1."
    )
if MIN_MATCHED_ELEMENTS < 1:
    raise ValueError(
        f"MIN_MATCHED_ELEMENTS ({MIN_MATCHED_ELEMENTS}) must be >= 1."
    )
if MIN_MATCHED_EDGES < 1:
    raise ValueError(
        f"MIN_MATCHED_EDGES ({MIN_MATCHED_EDGES}) must be >= 1."
    )
if not (0.0 <= P.MIN_MATCHED_ELEMENTS_PCT <= 1.0):
    raise ValueError(
        f"MIN_MATCHED_ELEMENTS_PCT ({P.MIN_MATCHED_ELEMENTS_PCT}) "
        "must be in [0, 1]."
    )
if P.MIN_MATCHED_ELEMENTS_FLOOR < 0:
    raise ValueError(
        f"MIN_MATCHED_ELEMENTS_FLOOR ({P.MIN_MATCHED_ELEMENTS_FLOOR}) "
        "must be >= 0."
    )
if not (0.0 <= P.MIN_MATCHED_EDGES_PCT <= 1.0):
    raise ValueError(
        f"MIN_MATCHED_EDGES_PCT ({P.MIN_MATCHED_EDGES_PCT}) must be in [0, 1]."
    )
if P.MIN_MATCHED_EDGES_FLOOR < 0:
    raise ValueError(
        f"MIN_MATCHED_EDGES_FLOOR ({P.MIN_MATCHED_EDGES_FLOOR}) must be >= 0."
    )


def _weight_subgraph_from_distributions(
    subgraph: Subgraph,
    sigs: Dict[int, Tuple[str, TypedSignature]],
    content_dist: Optional[Dict],
    structural_dist: Optional[Dict],
) -> Subgraph:
    """Copy `subgraph` with node_weights/weights from discriminative
    distributions.

    Only the CORE (target) side is ever passed through here; the query
    side keeps uniform mass. Missing entries default to 1.0, which is the
    neutral multiplicative identity, so an unweighted feature contributes
    exactly as it did before.

    Keys are cast to float explicitly to match how distributions.py builds
    them (float(quantized)), so int/float key mismatches cannot occur.
    """
    if not getattr(P, "EVIDENCE_DISCRIMINATIVE_MASS", True):
        return subgraph
    if not content_dist and not structural_dist:
        return subgraph

    def _content_weight(lt: str, qt) -> Optional[float]:
        if not content_dist:
            return None
        qt_f = tuple(float(x) for x in qt)
        inner = content_dist.get(lt)
        if isinstance(inner, dict):
            v = inner.get(qt_f)
            return None if v is None else float(v)
        return None

    def _structural_weight(et, s_info, d_info) -> Optional[float]:
        if not structural_dist:
            return None
        s_lt, s_sig = s_info
        d_lt, d_sig = d_info
        s_q = tuple(float(x) for x in s_sig.quantized)
        d_q = tuple(float(x) for x in d_sig.quantized)
        inner = structural_dist.get(et)
        if not isinstance(inner, dict):
            return None
        # mirror the store's canonicalisation so we hit on either placement
        s_k = (s_lt, s_q)
        d_k = (d_lt, d_q)
        first, second = (s_k, d_k) if s_k <= d_k else (d_k, s_k)
        candidates = (
            (first[0], first[1], second[0], second[1]),
            (s_lt, s_q, d_lt, d_q),
            (d_lt, d_q, s_lt, s_q),
        )
        for k in candidates:
            v = inner.get(k)
            if v is not None:
                return float(v)
        return None

    new_node_weights = dict(getattr(subgraph, "node_weights", None) or {})
    for nid in subgraph.nodes:
        if nid in new_node_weights:
            continue
        info = sigs.get(nid)
        if info is None:
            new_node_weights[nid] = 1.0
            continue
        w = _content_weight(info[0], info[1].quantized)
        new_node_weights[nid] = w if w is not None else 1.0

    new_edge_weights = dict(getattr(subgraph, "weights", None) or {})
    for edge in subgraph.edges:
        s, d, et = edge
        canon = (min(s, d), max(s, d), et) if et in UNDIRECTED_EDGE_TYPES else (s, d, et)
        if canon in new_edge_weights:
            continue
        s_info = sigs.get(s)
        d_info = sigs.get(d)
        if s_info is None or d_info is None:
            new_edge_weights[canon] = 1.0
            continue
        w = _structural_weight(et, s_info, d_info)
        new_edge_weights[canon] = w if w is not None else 1.0

    return _dc_replace(subgraph, node_weights=new_node_weights,
                      weights=new_edge_weights)
# =============================================================================
# DATA STRUCTURES (§6.7)
# =============================================================================


@dataclass(frozen=True)
class Transfer:
    """§6.7: a matched action core's transferred consequence."""

    core_hash: bytes
    consequence_subgraph: Subgraph
    confidence: float  # MCS × reliability
    # [F2] Source concept for enabling-context resolution. Populated from
    # the matched CoreEntry's source_concept_id. None for CATEGORY or
    # label-path cores where no concept of origin is recorded.
    source_concept_id: Optional[int] = None


@dataclass
class CategoryMatch:
    """§6.7: a set of (label, confidence) pairs, sorted descending."""

    labels: List[Tuple[Any, float]] = field(default_factory=list)


@dataclass(frozen=True)
class GapSignal:
    """§6.7: emitted when no core matches above θ_MCS."""

    query_core: Subgraph
    kind: CoreKind


@dataclass(frozen=True)
class Identity:
    """One identity attempt that fired. Produced by query_identification.

    score is MCS (min-normalized). matched_elements is the integer
    count of matched nodes + matched edges, back-computed exactly from
    the score and min(|Q|, |C|). Both query_identification gates read
    this field: matched_elements >= max(MIN_MATCHED_ELEMENTS_FLOOR,
    int(MIN_MATCHED_ELEMENTS_PCT * min_size)) and matched_edges =
    matched_elements - len(query_members) >=
    max(MIN_MATCHED_EDGES_FLOOR, int(MIN_MATCHED_EDGES_PCT * min_size)).

    Equality and hash use (core_hash, query_members) only.

    rank_score is the Kc2 balanced-coverage similarity (mcs_similarity),
    carried for ordering only. It is deliberately NOT part of
    __hash__/__eq__, and score/confidence/matched_elements stay
    containment-derived so every existing gate keeps its metric.
    """
    core_hash: bytes
    score: float
    correspondence: Mapping[int, int]  # MappingProxyType at runtime
    confidence: float
    matched_elements: int
    query_members: FrozenSet[int]
    rank_score: float = 0.0

    def __hash__(self) -> int:
        return hash((self.core_hash, self.query_members))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Identity):
            return NotImplemented
        return (
            self.core_hash == other.core_hash
            and self.query_members == other.query_members
        )


@dataclass(frozen=True)
class IdentitySet:
    """Output of query_identification. See its docstring for semantics."""
    full: Tuple[Identity, ...]
    partial: Tuple[Identity, ...]
    gap: FrozenSet[int]
    failure_mode: str


# =============================================================================
# MAPPER
# =============================================================================


def _label_sort_key(label: Any) -> bytes:
    """[SPEC COMPLIANCE §9] Deterministic byte key for a category label.

    Uses the canonical serialize_value encoding. Labels are architecture
    values (detections/role values per §5.6) and are contractually
    serializable; a non-serializable label is an upstream defect and fails
    loudly. The former str()-fallback could embed memory addresses
    (`<object at 0x...>`), poisoning cross-run determinism.
    """
    return serialize_value(label)


class Mapper:
    """Core retrieval and transfer via structural matching (§6.1).

    Gap signals (§6.6.2): §6.6 text says "return GapSignal", but §6.7 types
    `query_action` as Optional[Transfer] and `query_category` as
    CategoryMatch. A GapSignal cannot be returned from either without
    violating the interface contract. Resolution: query methods return
    no-match (None / empty CategoryMatch), and gaps are queued internally
    on `_gap_queue`. The pipeline drains them via `drain_gaps()`.
    """

    def __init__(
        self,
        concept_graph: Any,
        core_library: CoreLibrary,
        signature_provider: SignatureProvider,
        prediction_module: Optional[Any] = None,
        near_miss_debug: bool = False,
    ) -> None:
        self.cg = concept_graph
        self.cl = core_library
        self._sig_provider = signature_provider
        self.prediction = prediction_module

        self.near_miss_debug = near_miss_debug
        self.near_miss_dumps: List[Dict[str, Any]] = []
        # Caches.
        self._sig_cache: Dict[int, Optional[Tuple[str, TypedSignature]]] = {}
        # [MACRO-TRANSFER DIAGNOSTIC] cumulative retrieval counters.
        self.transfer_loop_diag: Dict[str, int] = defaultdict(int)
        for _kind_label in ("action", "category"):
            for _mode in (
                "COMPOSITE_FIRED",
                "COMPOSITE_FAILED_PARTS_FIRED",
                "COMPOSITE_FAILED_PARTIAL",
                "COMPOSITE_FAILED_NOVEL",
            ):
                self.transfer_loop_diag[f"query_{_kind_label}_mode_{_mode}"] = 0
        # [D2 / R5] Pre-seeded counter: query-core extractions that
        # short-circuited to the raw window because the situation window
        # exceeds CORE_SIZE_CAP_NODES -- the default path at situation
        # scale (see _extract_query_core_category).
        self.transfer_loop_diag["query_core_raw_over_cap"] = 0
        # Step-scoped memo for the focus-free situation subgraph (see
        # _begin_step). Tuple of (t, Subgraph) or None.
        self._situation_memo: Optional[Tuple[int, Subgraph]] = None
        # [D2 / R5b] Step-scoped window-hash memo. Keyed on object identity
        # (`is`) so the same situation Subgraph passed through
        # query_category's hash gate and _get_or_build_query_core_category's
        # cache-key construction is hashed once per step. Cleared in
        # _begin_step. Correctness never depends on it: a different object
        # with equal content simply re-hashes.
        self._step_window_hash: Optional[Tuple[int, Any, bytes]] = None
        # Query cache: key = (situation_hash, kind, outcome_hash_or_None).
        # `None` (not b"") marks "no expected outcome" — b"" could collide
        # with a legitimately empty outcome subgraph hash (§6.3).
        self._query_cache: Dict[
            Tuple[bytes, CoreKind, Optional[bytes]], Tuple[Subgraph, int]
        ] = {}

        # Per-label window history, bounded FIFO. Used for query-time
        # discriminative compression (§6.3).
        self._label_history: Dict[Any, deque] = defaultdict(
            lambda: deque(maxlen=LABEL_HISTORY_MAXLEN)
        )
        # [Batch D2] Last window hash appended to `_label_history` per
        # label. query_category is now called every step by the Pipeline,
        # so appending unconditionally would fill the bounded FIFO with
        # repeats of ONE situation when the scene is static — destroying
        # exactly the contrast (§6.3 discriminative compression) the
        # history exists to provide. The gate keeps the history a set of
        # structurally DISTINCT observations (same policy as
        # ObservationWindowBuffer.record's content dedup).
        self._label_hash_history: Dict[Any, bytes] = {}

        # Bounded gap queue.
        self._gap_queue: deque = deque()

        # [Phase 1 Decision 4 / P2-7] Confirmed similarity graph over
        # CATEGORY cores only (populated by match_all when kind is
        # CATEGORY). Filter, not metric.
        self._similarity_edges: Dict[Tuple[bytes, bytes], float] = {}
        self._similarity_neighbors: Dict[bytes, Set[bytes]] = defaultdict(set)
        self._seed_hit_streak: Dict[bytes, int] = defaultdict(int)
        self._force_flat_next: Set[bytes] = set()

        # Step bookkeeping.
        self._current_step: int = -1

        # Last query core (diagnostics).
        self._last_query_core: Optional[Subgraph] = None
        self._last_query_kind: Optional[CoreKind] = None

        # Diagnostics.
        self.diagnostics: Dict[str, Any] = {
            "action_queries": 0,
            "category_queries": 0,
            "transfers_emitted": 0,
            "category_labels_returned": 0,
            "gaps_emitted": 0,
            "query_cache_hits": 0,
            "query_cache_misses": 0,
            "transfers_register_failures": 0,
            "current_step": -1,
            "failure_mode_COMPOSITE_FIRED": 0,
            "failure_mode_COMPOSITE_FAILED_PARTS_FIRED": 0,
            "failure_mode_COMPOSITE_FAILED_PARTIAL": 0,
            "failure_mode_COMPOSITE_FAILED_NOVEL": 0,
            "identities_capped_this_call": 0,
            "parts_count_full": 0,
            "parts_count_partial": 0,
            "reasoning_pass_empty_library": 0,
            "reasoning_pass_empty_matchable": 0,
            "fired_situation_calls": 0,
            "fired_situation_nodes_min": 999999,
            "fired_situation_nodes_max": 0,
            "fired_situation_edges_min": 999999,
            "fired_situation_edges_max": 0,
            # [Batch E] Count of induced situation-subgraph edges admitted
            # SOLELY by the recency clauses (weight below
            # THETA_SITUATION_EDGE). MED-07: at call sites, not inside
            # _edge_included.
            "mapper_edges_included_by_recency": 0,
        }
        self.diagnostics["full_identities_fired"] = 0
        self.diagnostics["identity_coverage_max"] = 0.0
        self.diagnostics["concept_write_backs"] = 0
        self.diagnostics["concept_write_back_skipped_no_source"] = 0
        self.diagnostics["concept_write_back_skipped_dead_concept"] = 0
        self.diagnostics["concept_write_back_skipped_idempotent"] = 0
        for i in range(21):
            self.diagnostics[f"matched_elements_bin_{i:02d}"] = 0

    # -------------------------------------------------------------------------
    # Public API (§6.7)
    # -------------------------------------------------------------------------

    def query_action(
        self,
        t: int,
        situation_subgraph: Subgraph,
        expected_outcome: Optional[Subgraph] = None,
    ) -> Optional[Transfer]:
        """§6.6.1 / Phase 4. Emits a Transfer only on COMPOSITE_FIRED."""
        self._begin_step(t)
        self.diagnostics["action_queries"] += 1

        if _subgraph_size(situation_subgraph) == 0:
            # [BluePrint v8 integration] This short-circuit returns before
            # query_identification, so the per-query mode log must be
            # updated here: verification item 7 requires the four
            # query_action_mode_* counters to sum to action_queries.
            # NOVEL is the faithful label (nothing was identified).
            # `failure_mode_*` is deliberately NOT touched — that family
            # counts identification passes (query_identification's own
            # domain, verification item 5) and this is not one.
            self.transfer_loop_diag[
                "query_action_mode_COMPOSITE_FAILED_NOVEL"
            ] += 1
            self._enqueue_gap(
                GapSignal(query_core=situation_subgraph, kind=CoreKind.ACTION)
            )
            return None

        query_core = self._get_or_build_query_core_action(
            situation_subgraph, expected_outcome, t
        )
        self._last_query_core = query_core
        self._last_query_kind = CoreKind.ACTION
        self.transfer_loop_diag["query_action_calls"] += 1

        idents = self.query_identification(t, query_core, CoreKind.ACTION)

        self.transfer_loop_diag[
            f"query_action_mode_{idents.failure_mode}"
        ] += 1

        if idents.failure_mode != "COMPOSITE_FIRED":
            if not idents.full and not idents.partial:
                self._enqueue_gap(
                    GapSignal(query_core=query_core, kind=CoreKind.ACTION)
                )
            return None

        self.transfer_loop_diag["query_action_matches_nonempty"] += 1

        # Phase 5: write back inside the pass, ACTION kind only.
        self._write_back(t, idents.full, CoreKind.ACTION)

        best: Optional[Transfer] = None
        best_entry: Optional[CoreEntry] = None
        best_key: Tuple[float, float, bytes] = (-1.0, -1.0, b"\xff" * 16)
        for ident in idents.full:
            entry = self.cl.lookup_by_hash(ident.core_hash, CoreKind.ACTION)
            if entry is None:
                continue
            if ident.confidence < TAU_TRANSFER:
                self.transfer_loop_diag["transfer_tau_rejected"] += 1
                continue
            payload = entry.payload
            if not isinstance(payload, CorePayload_Action):
                continue
            # Primary: quality (score). Tie-break: matched_elements
            # (larger explained structure). Final: canonical hash for
            # determinism. All candidates already passed
            # MIN_MATCHED_ELEMENTS + MIN_MATCHED_EDGES + strong_thresh.
            key = (
                -ident.score,
                -ident.matched_elements,
                entry.canonical_hash,
            )
            if best is None or key < best_key:
                best = Transfer(
                    core_hash=entry.canonical_hash,
                    consequence_subgraph=payload.consequence_subgraph,
                    confidence=ident.confidence,
                    source_concept_id=entry.source_concept_id,
                )
                best_entry = entry
                best_key = key

        if best is None:
            # Matches existed, but none passed τ_transfer. Not a gap (§6.6).
            return None

        # 4. Register the transfer with Prediction (§8.5.1).
        if self.prediction is not None:
            effect_atoms: FrozenSet[Any] = frozenset()
            if best_entry is not None and best_entry.source_concept_id is not None:
                src_concept = self.cg.concepts.get(best_entry.source_concept_id)
                if src_concept is not None:
                    status_name = getattr(src_concept.status, "name", "")
                    if status_name not in ("TOMBSTONED", "REJECTED"):
                        effect_atoms = frozenset(
                            (getattr(src_concept, "effect_ids_by_action", None) or {}).values()
                        )
            try:
                self.prediction.register_transfer(
                    core_hash=best.core_hash,
                    predicted_consequence=best.consequence_subgraph,
                    confidence=best.confidence,
                    t_predicted=t,
                    horizon=P.W_VALIDATION,
                    effect_atoms=effect_atoms,
                )
            except Exception:
                # [SPEC COMPLIANCE §6.7] register_transfer is defined as
                # O(1) and must not raise. A raised exception indicates a
                # downstream defect. We record it and continue rather than
                # abort the query — the Transfer has already been computed
                # and the caller must not lose it — but the failure is
                # surfaced through diagnostics so it is not silently
                # swallowed.
                self.diagnostics["transfers_register_failures"] += 1

        self.diagnostics["transfers_emitted"] += 1
        self.transfer_loop_diag["query_action_hits"] += 1
        return best

    def query_category(self, t, label_signal, window_subgraph,
                      read_only: bool = False) -> CategoryMatch:
        """Rank CATEGORY cores against a window.

        Args:
            t: pipeline step.
            label_signal: a single ("class", label) tuple, OR a sequence of
                such tuples. The scoring loop is label-agnostic (candidate
                retrieval is unfiltered by label and the query-core cache key
                is label-independent), so a sequence returns the ranked union
                across all requested labels in one CategoryMatch. Use a
                sequence to read the true top-1/top-2 cross-class margin.
            window_subgraph: the observation window to score.
            read_only: when True, suppress every CONCLUSION so a probe cannot
                perturb accumulation state: the _begin_step step advance,
                _enqueue_gap, and the concept/reliability write-back. The
                label-history append is KEPT, because the query core is
                materialised from that history and dropping it would change
                what is measured. A read-only probe is "reaches no
                conclusions", not "changes no state". Caches are still
                cleared so memo behaviour is identical in both modes.

        Returns:
            CategoryMatch with labels sorted by rank_score desc.
        """
        # [read-only probe] A measurement pass must not perturb accumulation
        # state. The caches are still cleared (memo behaviour must stay
        # identical for both modes) but the step counter does not advance.
        if read_only:
            if self._current_step != t:
                self._sig_cache.clear()
                self._situation_memo = None
                self._step_window_hash = None
                self._evict_query_cache(t)
        else:
            self._begin_step(t)
        self.diagnostics["category_queries"] += 1

        # [multi-label] label_signal may be one tuple or a sequence of them.
        # The scoring loop below is label-agnostic already, so a sequence
        # simply widens which labels the returned union covers.
        _multi_label = False
        _labels_requested = (label_signal,)
        if isinstance(label_signal, (list, tuple, set, frozenset)):
            _seq = list(label_signal)
            if _seq and all(
                isinstance(x, tuple) and len(x) == 2
                for x in _seq
            ):
                _multi_label = True
                _labels_requested = tuple(_seq)

        if _subgraph_size(window_subgraph) == 0:
            self.transfer_loop_diag[
                "query_category_mode_COMPOSITE_FAILED_NOVEL"
            ] += 1
            if not read_only:
                self._enqueue_gap(
                    GapSignal(query_core=window_subgraph, kind=CoreKind.CATEGORY)
                )
            return CategoryMatch(labels=[])

        window_hash = self._window_hash(t, window_subgraph)
        # [read-only probe] The history append is QUERY SEMANTICS, not a
        # side-effect: _extract_query_core_category materialises its query
        # core from the label history (mapper.py:1665), so skipping the
        # append would make the probe measure a DIFFERENT query. It is kept
        # in read_only mode. What read_only suppresses is CONCLUSIONS:
        # _write_back, _enqueue_gap and the step advance below.
        for _ls in _labels_requested:
            history = self._label_history[_ls]
            prev_hash = self._label_hash_history.get(_ls)
            if prev_hash != window_hash:
                history.append(window_subgraph)
                self._label_hash_history[_ls] = window_hash

        # [multi-label] The query core must be LABEL-INDEPENDENT: scoring
        # each class against a query core materialised from a different
        # class's history is not a fair comparison, and for windows above
        # CORE_SIZE_CAP_NODES the two paths diverge.
        _qc_label = None if _multi_label else label_signal
        if read_only:
            # A measurement probe must neither READ nor WRITE the query-core
            # cache. Its key is label-independent while the cached value
            # depends on the label that populated it, so a warm cache would
            # hand back another class's materialised template and silently
            # bias the cross-class ranking. Build directly instead; label
            # None forces the raw-window branch with no history input.
            query_core = self._extract_query_core_category(
                window_subgraph, _qc_label
            )
        else:
            query_core = self._get_or_build_query_core_category(
                window_subgraph, _qc_label, t
            )

        content_store = ContentSignatureStore()
        content_query_core = build_content_subgraph(
            query_core, self._sig_lookup, content_store, self.diagnostics
        )

        saved_provider = self._sig_provider
        saved_cache = self._sig_cache
        self._sig_provider = content_store.get
        self._sig_cache = {}
        try:
            self._last_query_core = content_query_core
            self._last_query_kind = CoreKind.CATEGORY

            query_lens: Set[str] = set()
            for nid in content_query_core.nodes:
                info = self._sig_lookup(nid)
                if info is not None:
                    query_lens.add(info[0])

            candidate_idxs = self.cl.get_candidates_for_query(
                query_lens, CoreKind.CATEGORY
            )
            candidate_idxs = self._filter_content_tier(
                candidate_idxs, CoreKind.CATEGORY
            )

            query_sigs = self._collect_signatures(content_query_core)

            from consolidator import evidence_score, decide_with_gap

            per_label_best: Dict[Any, float] = {}
            per_core: List[Tuple[CoreEntry, float, Dict[int, int]]] = []
            for idx in candidate_idxs:
                entry = self.cl.entries[idx]
                if getattr(entry, "core_status", CoreStatus.CRYSTALLIZED) == CoreStatus.REJECTED:
                    continue
                if not isinstance(entry.payload, CorePayload_Category):
                    self.diagnostics["evidence_skipped_payload"] = (
                        self.diagnostics.get("evidence_skipped_payload", 0) + 1
                    )
                    continue
                # [GAS LINE] weight the CORE side by its discriminative
                # distributions; the query side stays uniform.
                weighted_core = _weight_subgraph_from_distributions(
                    entry.core_subgraph,
                    entry.core_node_signatures,
                    getattr(entry, "content_distribution", None),
                    getattr(entry, "structural_distribution", None),
                )
                ev, corr = evidence_score(
                    content_query_core,
                    weighted_core,
                    query_sigs,
                    entry.core_node_signatures,
                )
                label = entry.payload.label
                if ev > per_label_best.get(label, -1.0):
                    per_label_best[label] = ev
                per_core.append((entry, ev, corr))

            accepted, predicted_label, gap = decide_with_gap(per_label_best)

            if accepted and per_label_best.get(predicted_label, 0.0) < P.EVIDENCE_MIN_BALANCED:
                accepted = False

            labels: List[CategoryLabel] = []
            for entry, ev, corr in per_core:
                labels.append(CategoryLabel(
                    core_hash=entry.canonical_hash,
                    label=entry.payload.label,
                    raw_score=float(ev),
                    confidence=float(ev) * float(entry.reliability),
                    rank_score=float(ev),
                ))
            labels.sort(key=lambda cl: (-cl.rank_score, _label_sort_key(cl.label)))

            if not per_core:
                if not read_only:
                    self._enqueue_gap(
                        GapSignal(query_core=content_query_core, kind=CoreKind.CATEGORY)
                    )

            self.diagnostics["evidence_accepted"] = (
                self.diagnostics.get("evidence_accepted", 0) + (1 if accepted else 0)
            )
            self.diagnostics["evidence_gap"] = float(gap)
            self.diagnostics["evidence_predicted_label"] = str(predicted_label)
            self.diagnostics["category_labels_returned"] = (
                self.diagnostics.get("category_labels_returned", 0) + len(labels)
            )

            if accepted and not read_only:
                accepted_entry = next(
                    (e for e, _, _ in per_core if e.payload.label == predicted_label),
                    None,
                )
                if accepted_entry is not None:
                    identity_like = Identity(
                        core_hash=accepted_entry.canonical_hash,
                        score=per_label_best.get(predicted_label, 0.0),
                        correspondence=MappingProxyType({}),
                        confidence=per_label_best.get(predicted_label, 0.0)
                            * float(accepted_entry.reliability),
                        matched_elements=0,
                        query_members=frozenset(),
                        rank_score=per_label_best.get(predicted_label, 0.0),
                    )
                    if not read_only:
                        self._write_back(t, (identity_like,), CoreKind.CATEGORY)

            return CategoryMatch(labels=labels)
        finally:
            self._sig_provider = saved_provider
            self._sig_cache = saved_cache

    def _write_back(
        self,
        t: int,
        idents: Tuple[Identity, ...],
        kind: CoreKind,
    ) -> None:
        """Phase 5: successful strong unification updates the matched
        CoreEntry's reliability EMA.

        Concept-level N_conf is deliberately NOT written: the CoreEntry's
        canonical hash is computed over the (possibly pruned) core
        subgraph; the Concept's canonical hash is a different derivation.
        Concept evidence continues to accumulate through
        concept_graph.step_21 and step_28. Follow-up: add
        source_concept_id to CoreEntry.

        Kind-scoped (no ACTION→CATEGORY fallthrough). Idempotent per
        step via t_last_validated. Weak identities never write back.
        """
        seen: Set[bytes] = set()
        for ident in idents:
            if ident.core_hash in seen:
                continue
            seen.add(ident.core_hash)
            entry = self.cl.lookup_by_hash(ident.core_hash, kind)
            if entry is None:
                continue
            if entry.t_last_validated == t:
                continue
            entry.reliability = clamp(
                entry.reliability
                + KAPPA_REL * (ident.confidence - entry.reliability),
                0.0,
                1.0,
            )
            entry.t_last_validated = t
            if entry.source_concept_id is not None:
                concept = self.cg.get_concept(entry.source_concept_id)
                if concept is None:
                    self.diagnostics["concept_write_back_skipped_no_source"] += 1
                    continue
                if concept.last_witness_update >= t:
                    self.diagnostics["concept_write_back_skipped_idempotent"] += 1
                    continue
                concept.N_conf += 1
                concept.n_conf_from_reasoning += 1
                if concept.N_conf <= 0:
                    self.diagnostics["concept_write_back_skipped_dead_concept"] += 1
                    continue
                concept.last_witness_update = t
                self.diagnostics["concept_write_backs"] += 1

    def _filter_content_tier(
        self, candidate_idxs: List[int], kind: CoreKind
    ) -> List[int]:
        """[v32 CONTENT TIER] Keep only cores whose tier matches the query
        kind. ACTION cores are instance-tier; CATEGORY cores are content-tier.
        Content IDs >= 2^63, instance IDs are small integers — cross-tier MCS
        is impossible.

        [MIGRATION] Existing CATEGORY cores extracted before v32 carry the
        default core_tier='instance' and are excluded by this filter. They
        are not deleted; they simply do not participate in content-tier
        queries. New CATEGORY cores are content-tier. In a persistent
        library spanning the upgrade, both tiers coexist harmlessly."""
        if kind == CoreKind.ACTION:
            return [
                i for i in candidate_idxs
                if self.cl.entries[i].core_tier == "instance"
            ]
        return [
            i for i in candidate_idxs
            if self.cl.entries[i].core_tier == "content"
        ]

    def match_all(
        self,
        t: int,
        query_core: Subgraph,
        kind: CoreKind,
    ) -> List[Tuple[CoreEntry, float, Dict[int, int]]]:
        """§6.7: all cores of `kind` with MCS scores and node
        correspondences, sorted by score descending (tie-break: canonical
        hash ascending). Filtered by the per-family θ_MCS in §6.5.

        Each returned row is one identity attempt: the query matched
        against one library core at one level. The library provides the
        levels; no decomposition of the query is required.
        """
        if _subgraph_size(query_core) == 0:
            return []

        # Level-1: type-set retrieval (§5.7 / §6.5).
        query_lens: Set[str] = set()
        for nid in query_core.nodes:
            info = self._sig_lookup(nid)
            if info is not None:
                query_lens.add(info[0])
        # [v13 / audit Finding 2] Bound the similarity-graph state.
        self._prune_similarity_state()

        candidate_idxs = self.cl.get_candidates_for_query(query_lens, kind)
        # [v32 CONTENT TIER] Keep only cores of the matching tier.
        candidate_idxs = self._filter_content_tier(candidate_idxs, kind)
        d = self.transfer_loop_diag
        d["get_candidates_for_query_calls"] += 1
        d["get_candidates_for_query_total_returned"] += len(candidate_idxs)

        # [Phase 1 Decision 4] Activation gate: at library size ≥
        # N_SIMILARITY_ACTIVATE, restrict candidates to the confirmed-
        # similarity neighborhood, subject to anti-ossification. Empty
        # graph falls through to the flat candidate list.
        if (
            kind == CoreKind.CATEGORY
            and len(self.cl.entries) >= P.N_SIMILARITY_ACTIVATE
            and candidate_idxs
        ):
            seed_hashes = [
                self.cl.entries[i].canonical_hash
                for i in candidate_idxs[: P.K_SEED]
            ]
            any_forced = any(s in self._force_flat_next for s in seed_hashes)
            if not any_forced:
                neighbor_hashes = set(
                    self.lookup_similar(
                        seed_hashes,
                        edge_threshold=P.THETA_TRUST_NEIGHBORHOOD,
                    )
                )
                allowed_idxs = [
                    i for i in candidate_idxs
                    if self.cl.entries[i].canonical_hash in neighbor_hashes
                ]
                if allowed_idxs:
                    candidate_idxs = allowed_idxs
                for s in seed_hashes:
                    self._seed_hit_streak[s] += 1
                    if self._seed_hit_streak[s] >= P.N_FORCE_FLAT:
                        self._force_flat_next.add(s)
                        self._seed_hit_streak[s] = 0
            else:
                for s in seed_hashes:
                    self._force_flat_next.discard(s)
                    self._seed_hit_streak[s] = 0
        if not candidate_idxs:
            d["get_candidates_for_query_empty_returns"] += 1
            # [RETRIEVAL-FAILURE LOG / Phase 2 next-action] On an empty
            # return while same-kind cores exist, record the lens-type
            # mismatch: query lens set vs each stored core's lens types
            # (first occurrence verbatim + count). Deterministic: sorted
            # tuples of lens-type strings.
            stored = [
                e for e in self.cl.entries if e.kind == kind
            ]
            if stored:
                d["retrieval_empty_with_stored_cores"] += 1
                if d["retrieval_empty_with_stored_cores"] == 1:
                    q_lens = tuple(sorted(query_lens))
                    stored_lens = tuple(
                        sorted(
                            (
                                e.canonical_hash.hex(),
                                tuple(
                                    sorted(
                                        info[0]
                                        for info in e.core_node_signatures.values()
                                    )
                                ),
                            )
                            for e in stored
                        )
                    )
                    d["first_failed_query_lens"] = q_lens
                    d["first_failed_stored_lens"] = stored_lens
        if not candidate_idxs:
            return []

        # Level-2: full MCS on candidates (§6.4.1).
        query_sigs = self._collect_signatures(query_core)
        results: List[Tuple[CoreEntry, float, float, Dict[int, int]]] = []
        for idx in candidate_idxs:
            entry = self.cl.entries[idx]
            from consolidator import evidence_score
            # [GAS LINE] weight the CORE side by its discriminative
            # distributions; the query side stays uniform.
            weighted_core = _weight_subgraph_from_distributions(
                entry.core_subgraph,
                entry.core_node_signatures,
                getattr(entry, "content_distribution", None),
                getattr(entry, "structural_distribution", None),
            )
            ev, corr = evidence_score(
                query_core,
                weighted_core,
                query_sigs,
                entry.core_node_signatures,
            )
            if ev < self._theta_evidence_for_entry(entry):
                self.transfer_loop_diag["match_all_evidence_below_threshold"] += 1
                continue
            results.append((entry, ev, ev, corr))

        results.sort(key=lambda x: (-x[2], x[0].canonical_hash))
        # [Phase 1 Decision 4 / v13 audit Finding 1] Similarity edges
        # recorded for CATEGORY cores only — the activation gate only
        # traverses for CATEGORY. Recording is the sole population path.
        if (
            kind == CoreKind.CATEGORY
            and results
            and results[0][1] >= P.SIMILARITY_EDGE_THRESHOLD
        ):
            top_entry, top_score, _top_rank, _ = results[0]
            for entry, score, _rank, _ in results[1:]:
                # [Kc2 adaptation] Was `break`, which was valid only while
                # `results` was sorted by containment descending (first
                # sub-threshold entry implied all later ones were too).
                # The rank sort removes that invariant, so `break` would
                # silently drop edges that pass 0.60. Filter unchanged:
                # containment >= P.SIMILARITY_EDGE_THRESHOLD.
                if score < P.SIMILARITY_EDGE_THRESHOLD:
                    continue
                self.confirm_similarity(
                    top_entry.canonical_hash,
                    entry.canonical_hash,
                    min(top_score, score),
                )
        return results

    def query_identification(
        self,
        t: int,
        query_core: Subgraph,
        kind: CoreKind = CoreKind.ACTION,
    ) -> IdentitySet:
        """Thesis §3.4 / Phases 2 and 4.

        Match the query against every core in the library (one identity
        attempt per core, at every level the library provides). Classify
        into the four thesis §3.5 modes.

        No decomposition. The library IS the vocabulary of levels. MCS
        min-normalization lets a small core score high wherever its
        pattern appears — that is the "blue-part IS blue-core" attempt.

        SC11 size gate: FIRED requires THREE conditions:
        score >= strong_threshold AND matched_elements >= max(FLOOR,
        int(PCT * min_size)) AND matched_edges >= max(FLOOR,
        int(PCT * min_size)). Score alone is min-normalized and would
        call a 1-node match against a 30-node query FIRED. The gates are
        percentage-based (H.4b.7): the absolute integers
        MIN_MATCHED_ELEMENTS / MIN_MATCHED_EDGES were calibrated for
        Hungarian's denser correspondences and are now deprecated.

        Failure modes:
          COMPOSITE_FIRED — some identity passes both gates.
          COMPOSITE_FAILED_PARTS_FIRED — no such identity, but the union
              of all weak identities covers every matchable query node.
          COMPOSITE_FAILED_PARTIAL — the union covers some but not all.
          COMPOSITE_FAILED_NOVEL — nothing matched (or matchable empty).

        query_members, gap, and correspondence refer to query_core nodes.
        If the caller passed a pruned query_core (see
        _get_or_build_query_core_action), the correspondence is with the
        pruned nodes, not the raw situation.

        Calls _begin_step(t) internally (idempotent).
        """
        self._begin_step(t)

        if not self.cl.entries:
            self.diagnostics["reasoning_pass_empty_library"] += 1
            self.diagnostics["failure_mode_COMPOSITE_FAILED_NOVEL"] += 1
            return IdentitySet(
                full=(),
                partial=(),
                gap=frozenset(query_core.nodes),
                failure_mode="COMPOSITE_FAILED_NOVEL",
            )

        matchable = frozenset(
            nid for nid in query_core.nodes if self._sig_lookup(nid) is not None
        )
        if not matchable:
            self.diagnostics["reasoning_pass_empty_matchable"] += 1
            self.diagnostics["failure_mode_COMPOSITE_FAILED_NOVEL"] += 1
            return IdentitySet(
                full=(),
                partial=(),
                gap=frozenset(query_core.nodes),
                failure_mode="COMPOSITE_FAILED_NOVEL",
            )
        matchable_size = len(matchable)

        matches = self.match_all(t, query_core, kind)

        full: List[Identity] = []
        partial: List[Identity] = []
        capped = 0
        for entry, score, rank_score, corr in matches:
            members = frozenset(corr.keys())
            if not members:
                # Defensive: match_all's family-threshold filter
                # guarantees a positive MCS, which requires non-empty
                # correspondence.
                continue
            coverage = len(members) / matchable_size
            if coverage > self.diagnostics["identity_coverage_max"]:
                self.diagnostics["identity_coverage_max"] = coverage
            min_size = min(
                len(query_core.nodes) + len(query_core.edges),
                len(entry.core_subgraph.nodes) + len(entry.core_subgraph.edges),
            )
            matched_elements = int(round(score * min_size))
            matched_edges = matched_elements - len(members)
            self.diagnostics[
                f"matched_elements_bin_{min(matched_elements, 20):02d}"
            ] += 1
            strong_thresh = (
                self._theta_evidence_for_entry(entry) + THETA_IDENTIFY_DELTA
            )
            if self.near_miss_debug and matched_elements == 4:
                core_members = set(corr.values())
                q_edges = {
                    e for e in query_core.edges
                    if e[0] in members and e[1] in members
                }
                c_edges = {
                    e for e in entry.core_subgraph.edges
                    if e[0] in core_members and e[1] in core_members
                }
                mapped_q_etypes = {
                    (corr[u], corr[v], et) for u, v, et in q_edges
                }
                c_etypes = {(cu, cv, et) for cu, cv, et in c_edges}
                missing = c_etypes - mapped_q_etypes
                self.near_miss_dumps.append({
                    "score": score,
                    "matched_elements": matched_elements,
                    "matched_edges": matched_edges,
                    "min_size": min_size,
                    "query_members": sorted(members),
                    "core_members": sorted(core_members),
                    "core_hash": entry.canonical_hash,
                    "core_edge_types_present": sorted({
                        str(et) for _, _, et in c_edges
                    }),
                    "query_edge_types_present": sorted({
                        str(et) for _, _, et in q_edges
                    }),
                    "mapped_query_edge_types": sorted({
                        (cu, cv, str(et)) for cu, cv, et in mapped_q_etypes
                    }),
                    "missing_core_edges": sorted({
                        (cu, cv, str(et)) for cu, cv, et in missing
                    }),
                    "fired": bool(score >= strong_thresh),
                })
            ident = Identity(
                core_hash=entry.canonical_hash,
                score=score,
                correspondence=MappingProxyType(dict(corr)),
                confidence=score * float(entry.reliability),
                matched_elements=int(round(score * min_size)),
                query_members=members,
                rank_score=float(rank_score),
            )
            if score >= strong_thresh:
                if len(full) < MAX_IDENTITIES_PER_BUCKET:
                    full.append(ident)
                    self.diagnostics["full_identities_fired"] += 1
                else:
                    capped += 1
            else:
                if len(partial) < MAX_IDENTITIES_PER_BUCKET:
                    partial.append(ident)
                else:
                    capped += 1
        if capped:
            self.diagnostics["identities_capped_this_call"] += capped

        full = self._dedup_identities(full)
        partial = self._dedup_identities(partial)

        if full:
            mode = "COMPOSITE_FIRED"
        elif not partial:
            mode = "COMPOSITE_FAILED_NOVEL"
        else:
            union_covered: Set[int] = set()
            for ident in partial:
                union_covered |= ident.query_members
            if union_covered >= matchable:
                mode = "COMPOSITE_FAILED_PARTS_FIRED"
            else:
                mode = "COMPOSITE_FAILED_PARTIAL"

        covered: Set[int] = set()
        for ident in full:
            covered |= ident.query_members
        for ident in partial:
            covered |= ident.query_members
        gap = frozenset(query_core.nodes - covered)

        self.diagnostics[f"failure_mode_{mode}"] += 1
        self.diagnostics["parts_count_full"] += len(full)
        self.diagnostics["parts_count_partial"] += len(partial)

        return IdentitySet(
            full=tuple(full),
            partial=tuple(partial),
            gap=gap,
            failure_mode=mode,
        )

    def _dedup_identities(self, idents: List[Identity]) -> List[Identity]:
        """Deterministic dedup + sort by
        (-rank_score, core_hash, min(query_members))."""
        seen: Set[Tuple[bytes, FrozenSet[int]]] = set()
        out: List[Identity] = []
        for ident in idents:
            key = (ident.core_hash, ident.query_members)
            if key in seen:
                continue
            seen.add(key)
            out.append(ident)
        out.sort(
            key=lambda i: (
                -i.rank_score,
                i.core_hash,
                min(i.query_members) if i.query_members else 0,
            )
        )
        return out

    def mcs_containment(self, query: Subgraph, target: Subgraph) -> bool:
        """§8.5.2/§5.3.2 containment predicate: `query ⊆ target` iff
        MCS_score(query, target) ≥ θ_containment (0.95), computed with the
        typed-constrained MCS (§6.4.1). Public: the Prediction Module
        consumes this via injection (set_containment_checker) for transfer
        validation — it must not maintain a private exact-subset surrogate.

        An empty query is contained in anything (no structural claim to
        contradict); this mirrors the vacuous-truth convention of §5.3.1
        replay for empty cores, distinct from §6.4's MCS_score(empty,·)=0
        matching convention (an empty situation carries no information).
        """
        if _subgraph_size(query) == 0:
            return True
        if _subgraph_size(target) == 0:
            return False
        score, _ = mcs_score(
            query,
            target,
            self._collect_signatures(query),
            self._collect_signatures(target),
        )
        return score >= THETA_CONTAINMENT

    # -------------------------------------------------------------------------
    # Situation subgraph construction (§6.2)
    # -------------------------------------------------------------------------

    def build_situation_subgraph(
        self,
        t: int,
        focus_node: Optional[int] = None,
    ) -> Subgraph:
        """§6.2 + Phase 3.

        Focus-free: activation-driven spreading (`_fired_situation`). The
        graph fires from every active concept's members, spreads along
        edges with LAMBDA_ACT decay, and the situation is the subgraph
        whose accumulated activation clears THETA_SITUATION. Per-step
        memo preserved.

        focus_node: single-seed BFS (unchanged).

        Induced-edge inclusion uses _edge_included (Batch E).
        Spreading itself is unchanged.
        """
        self._begin_step(t)
        if focus_node is None:
            if self._situation_memo is not None:
                memo_t, memo_sg = self._situation_memo
                if memo_t == t:
                    return memo_sg
            seeds = self._collect_fired_seeds()
            result = self._fired_situation(t, seeds)
            self._situation_memo = (t, result)
            self.diagnostics["fired_situation_calls"] += 1
            n = len(result.nodes)
            e = len(result.edges)
            if n < self.diagnostics["fired_situation_nodes_min"]:
                self.diagnostics["fired_situation_nodes_min"] = n
            if n > self.diagnostics["fired_situation_nodes_max"]:
                self.diagnostics["fired_situation_nodes_max"] = n
            if e < self.diagnostics["fired_situation_edges_min"]:
                self.diagnostics["fired_situation_edges_min"] = e
            if e > self.diagnostics["fired_situation_edges_max"]:
                self.diagnostics["fired_situation_edges_max"] = e
            return result

        # focus_node path: existing BFS body unchanged below.
        pattern_nodes = self._live_pattern_node_ids()
        if focus_node not in pattern_nodes:
            return Subgraph(nodes=frozenset(), edges=frozenset())
        seeds: Set[int] = {focus_node}

        # 2. BFS expansion within H_SITUATION hops.
        visited: Set[int] = set(seeds)
        frontier: Set[int] = set(seeds)
        for _ in range(H_SITUATION):
            next_frontier: Set[int] = set()
            for nid in sorted(frontier):
                for edge in self.cg.get_outgoing_edges(nid):
                    nb = edge.child
                    if nb in pattern_nodes and nb not in visited:
                        next_frontier.add(nb)
                for edge in self.cg.get_incoming_edges(nid):
                    nb = edge.parent
                    if nb in pattern_nodes and nb not in visited:
                        next_frontier.add(nb)
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

        # 3. Induced edges, filtered by recency-aware inclusion (Batch E).
        #    The `edges` set deduplicates so the recency counter increments
        #    at most once per unique (parent, child, edge_type) triple,
        #    independent of iteration order over `visited`.
        edges: Set[Tuple[int, int, EdgeType]] = set()
        for nid in visited:
            for edge in self.cg.get_outgoing_edges(nid):
                if edge.child not in visited:
                    continue
                key = (edge.parent, edge.child, edge.edge_type)
                if key in edges:
                    continue
                if self._edge_included(edge, t):
                    if edge.weight < THETA_SITUATION_EDGE:
                        self.diagnostics[
                            "mapper_edges_included_by_recency"
                        ] += 1
                    edges.add(key)
            for edge in self.cg.get_incoming_edges(nid):
                if edge.parent not in visited:
                    continue
                key = (edge.parent, edge.child, edge.edge_type)
                if key in edges:
                    continue
                if self._edge_included(edge, t):
                    if edge.weight < THETA_SITUATION_EDGE:
                        self.diagnostics[
                            "mapper_edges_included_by_recency"
                        ] += 1
                    edges.add(key)

        # 4-5. Cap and prune by activation × fertility.
        if len(visited) > MAX_SITUATION_SIZE:
            scored: List[Tuple[float, Tuple[bytes, bytes, int], int]] = []
            for nid in visited:
                a = self._node_activation(nid)
                ef = self._node_max_ef(nid)
                scored.append((a * ef, self._node_canonical_key(nid), nid))
            scored.sort(key=lambda x: (-x[0], x[1]))
            kept = {nid for _, _, nid in scored[:MAX_SITUATION_SIZE]}
            visited = kept
            edges = {e for e in edges if e[0] in kept and e[1] in kept}

        edge_params = {
            e: self.cg.edges[e].params
            for e in edges
            if e in self.cg.edges and self.cg.edges[e].params is not None
        }

        result = Subgraph(
            nodes=frozenset(visited),
            edges=frozenset(edges),
            edge_params=edge_params,
            anchor_of_node=anchor_positions_for(
                visited, self.cg.pattern_node_position
            ),
        )
        return result

    def _edge_included(self, e: Any, t: int) -> bool:
        """[Batch E] Recency-aware edge inclusion for situation subgraphs.

        An edge enters the induced situation-subgraph edge set iff any of:

          (1) e.weight >= THETA_SITUATION_EDGE — the pre-Batch-E gate.

          (2) e.observed_at >= t - W_SITUATION_RECENCY
              AND e.pending_signal > 0.0
              The most recent binder observation of this edge was within
              the recency window AND carried positive signal. The
              pending_signal gate excludes absence evidence: observed_at
              is stamped on every upsert (including signal=0.0), so
              without the gate a fresh zero-signal edge would enter the
              situation as structure for ~W_SITUATION_RECENCY steps
              (audit E6-03). A subsequent zero-signal observation
              overwrites pending_signal and correctly revokes admission.

          (3) e.time_since_coact <= TAU_SITUATION_COACT
              The endpoints co-activated within the co-activation window.
              Written by step_28's concept-edge branch (SC-12) and, as of
              Batch E, its pattern-edge branch. Positive-evidence by
              construction: tsc resets to 0 only when signal > 0.

        Activation spreading is NOT affected: _fired_situation's spread
        loop keeps its raw `e.weight` test (Batch E RULE 3). Only the
        final induced-edge set uses this predicate. Counter increments
        for the recency-only case live at the call sites, not here
        (MED-07).
        """
        if e.weight >= THETA_SITUATION_EDGE:
            return True
        if (
            e.observed_at >= t - W_SITUATION_RECENCY
            and e.pending_signal > 0.0
        ):
            return True
        if e.time_since_coact <= TAU_SITUATION_COACT:
            return True
        return False

    def _collect_fired_seeds(self) -> FrozenSet[int]:
        """Members of every concept with A > 0, live pattern nodes only."""
        pattern_nodes = self._live_pattern_node_ids()
        seeds: Set[int] = set()
        for _cid, concept in self.cg.concepts.items():
            if getattr(concept, "A", 0) > 0:
                for nid in concept.members:
                    if nid in pattern_nodes:
                        seeds.add(nid)
        return frozenset(seeds)

    def _fired_situation(
        self, t: int, seed_members: FrozenSet[int]
    ) -> Subgraph:
        """Activation spreads with LAMBDA_ACT decay; max-accumulation;
        layer-snapshot order-independent. Induced-edge collection uses
        recency-aware inclusion (Batch E: _edge_included). Spreading
        weight is unchanged."""
        activation: Dict[int, float] = {n: 1.0 for n in seed_members}
        frontier: Set[int] = set(seed_members)
        for _ in range(H_FIRE):
            current_acts: Dict[int, float] = {
                n: activation[n] for n in frontier
            }
            next_frontier: Set[int] = set()
            for n in sorted(frontier):
                a_n = current_acts[n]
                for e in self.cg.get_outgoing_edges(n):
                    contrib = a_n * LAMBDA_ACT * e.weight
                    if contrib > activation.get(e.child, 0.0):
                        activation[e.child] = contrib
                        next_frontier.add(e.child)
                for e in self.cg.get_incoming_edges(n):
                    contrib = a_n * LAMBDA_ACT * e.weight
                    if contrib > activation.get(e.parent, 0.0):
                        activation[e.parent] = contrib
                        next_frontier.add(e.parent)
            frontier = next_frontier
            if not frontier:
                break

        kept = {n for n, a in activation.items() if a >= THETA_SITUATION}
        pattern_nodes = self._live_pattern_node_ids()
        kept &= pattern_nodes
        if len(kept) > MAX_SITUATION_SIZE:
            ordered = sorted(
                kept,
                key=lambda n: (-activation[n], self._node_canonical_key(n)),
            )
            kept = set(ordered[:MAX_SITUATION_SIZE])

        edges: Set[Tuple[int, int, EdgeType]] = set()
        for n in sorted(kept):
            for e in self.cg.get_outgoing_edges(n):
                if e.child in kept and self._edge_included(e, t):
                    # [MED-07] Counter at call site; predicate stays pure.
                    if e.weight < THETA_SITUATION_EDGE:
                        self.diagnostics[
                            "mapper_edges_included_by_recency"
                        ] += 1
                    edges.add((e.parent, e.child, e.edge_type))

        return Subgraph(
            nodes=frozenset(kept),
            edges=frozenset(edges),
            anchor_of_node=anchor_positions_for(
                kept, self.cg.pattern_node_position
            ),
        )

    # -------------------------------------------------------------------------
    # Gap queue (diagnostic channel for §6.6 GapSignal)
    # -------------------------------------------------------------------------

    def drain_gaps(self) -> List[GapSignal]:
        """Return and clear the queued gaps. Called by the pipeline each
        step to feed the Builder's exploration scope."""
        out = list(self._gap_queue)
        self._gap_queue.clear()
        return out

    def _enqueue_gap(self, gap: GapSignal) -> None:
        if len(self._gap_queue) >= GAP_QUEUE_MAX:
            self._gap_queue.popleft()
        self._gap_queue.append(gap)
        self.diagnostics["gaps_emitted"] += 1

    # -------------------------------------------------------------------------
    # Similarity graph (Phase 1 Decision 4 / P2-7)
    # -------------------------------------------------------------------------

    def confirm_similarity(
        self, core_a: bytes, core_b: bytes, mcs_score_val: float
    ) -> None:
        """Symmetric confirmed edge. Only recorded above
        SIMILARITY_EDGE_THRESHOLD. Neighbor cap enforced with
        drop-lowest-score / canonical-hash tie-break."""
        if core_a == core_b:
            return
        if mcs_score_val < P.SIMILARITY_EDGE_THRESHOLD:
            return
        a, b = (core_a, core_b) if core_a < core_b else (core_b, core_a)
        prev = self._similarity_edges.get((a, b))
        if prev is None or mcs_score_val > prev:
            self._similarity_edges[(a, b)] = mcs_score_val
        if b not in self._similarity_neighbors[a]:
            self._similarity_neighbors[a].add(b)
            self._similarity_neighbors[b].add(a)
            self._enforce_neighbor_cap(a)
            self._enforce_neighbor_cap(b)

    def _enforce_neighbor_cap(self, core: bytes) -> None:
        cap = P.MAX_SIMILARITY_NEIGHBORS
        neighbors = self._similarity_neighbors.get(core)
        if not neighbors or len(neighbors) <= cap:
            return
        def key(nb: bytes) -> Tuple[float, bytes]:
            a, b = (core, nb) if core < nb else (nb, core)
            return (self._similarity_edges.get((a, b), 0.0), nb)
        ordered = sorted(neighbors, key=key)
        drop = len(ordered) - cap
        for nb in ordered[:drop]:
            self._similarity_neighbors[core].discard(nb)
            self._similarity_neighbors[nb].discard(core)
            a, b = (core, nb) if core < nb else (nb, core)
            self._similarity_edges.pop((a, b), None)

    def lookup_similar(
        self,
        seed_cores: List[bytes],
        max_hops: int = P.MAX_HOPS,
        edge_threshold: float = P.SIMILARITY_EDGE_THRESHOLD,
    ) -> List[bytes]:
        """BFS from seeds over confirmed edges only. Reached set capped
        at K_NEIGHBOR_SCAN, sorted canonical hash."""
        reached: Set[bytes] = set()
        frontier: Set[bytes] = set(seed_cores)
        for _ in range(max_hops):
            next_frontier: Set[bytes] = set()
            for core in frontier:
                for nb in self._similarity_neighbors.get(core, ()):
                    a, b = (core, nb) if core < nb else (nb, core)
                    w = self._similarity_edges.get((a, b), 0.0)
                    if w < edge_threshold:
                        continue
                    if nb not in reached and nb not in frontier:
                        next_frontier.add(nb)
            reached |= frontier
            frontier = next_frontier
            if not frontier:
                break
        reached |= frontier
        ordered = sorted(reached)
        return ordered[: P.K_NEIGHBOR_SCAN]

    def _prune_similarity_state(self) -> None:
        """[v13 / audit Finding 2 + SC-XX] Bound the state.

        Transient structures clear on cap-exceed. Persistent graph is
        liveness-filtered (drops entries whose endpoints are no longer
        in the library — covers Manager-driven removals of cores from
        Q1 hash-change, Q2 fuse, Q3 split). Both checks are O(1).
        Called at the top of match_all."""
        if len(self._seed_hit_streak) > P.SIMILARITY_STATE_CAP:
            self._seed_hit_streak.clear()
        if len(self._force_flat_next) > P.SIMILARITY_STATE_CAP:
            self._force_flat_next.clear()
        if len(self._similarity_edges) > P.SIMILARITY_EDGE_CAP:
            live = {k[0] for k in self.cl.entries_by_hash.keys()}
            self._similarity_edges = {
                (a, b): w
                for (a, b), w in self._similarity_edges.items()
                if a in live and b in live
            }
            new_neighbors: Dict[bytes, Set[bytes]] = defaultdict(set)
            for k, v in self._similarity_neighbors.items():
                if k not in live:
                    continue
                filtered = {n for n in v if n in live}
                if filtered:
                    new_neighbors[k] = filtered
            self._similarity_neighbors = new_neighbors

    # -------------------------------------------------------------------------
    # Query core extraction (§6.3)
    # -------------------------------------------------------------------------

    def _get_or_build_query_core_action(
        self,
        situation: Subgraph,
        expected_outcome: Optional[Subgraph],
        t: int,
    ) -> Subgraph:
        situation_hash = _subgraph_wl_hash(situation, self._sig_lookup)
        # [SPEC INTERPRETATION §6.3] The spec keys the query cache by
        # situation hash alone. If a caller varies expected_outcome on the
        # same situation, that key collides. We append the outcome hash to
        # the key when an outcome is provided, and None when it is not —
        # a sentinel that cannot collide with any real WL hash.
        if expected_outcome is None:
            outcome_hash: Optional[bytes] = None
        else:
            outcome_hash = _subgraph_wl_hash(expected_outcome, self._sig_lookup)
        key: Tuple[bytes, CoreKind, Optional[bytes]] = (
            situation_hash,
            CoreKind.ACTION,
            outcome_hash,
        )
        cached = self._query_cache.get(key)
        if cached is not None and (t - cached[1]) <= QUERY_CACHE_TTL:
            self.diagnostics["query_cache_hits"] += 1
            return cached[0]
        self.diagnostics["query_cache_misses"] += 1

        core = self._extract_query_core_action(situation, expected_outcome)
        self._query_cache[key] = (core, t)
        return core

    def _extract_query_core_action(
        self,
        situation: Subgraph,
        expected_outcome: Optional[Subgraph],
    ) -> Subgraph:
        """§6.3: sufficient-mode element-wise pruning against the expected
        outcome. If no outcome is expected, skip compression and match the
        raw situation subgraph."""
        if expected_outcome is None:
            # §6.3: "If no outcome is expected, extraction is skipped."
            return situation
        # [SPEC INTERPRETATION §6.3] The size check is on node count, not
        # the node+edge sum. §5.13 gives CORE_SIZE_CAP as "20 nodes / 40
        # edges" — a node-count cap of 20 — and §6.3's own edge-case
        # examples (empty situation, single-node situation) are phrased in
        # nodes. §6.4's `|X| = |X.nodes| + |X.edges|` is a local definition
        # for MCS-score normalization, not the unit of this gate.
        if len(situation.nodes) <= CORE_SIZE_CAP_NODES:
            return situation
        return self._prune_situation_against(situation, expected_outcome)

    def _prune_situation_against(
        self,
        situation: Subgraph,
        outcome: Subgraph,
    ) -> Subgraph:
        """Element-wise pruning, deleting elements whose removal keeps
        MCS(situation, outcome) >= θ_containment.

        [SPEC INTERPRETATION §6.3] §6.3's sufficient-mode test is
        "replay(P_core) produces the outcome". A situation snapshot has no
        action sequence; the "produces" predicate degenerates to a check
        that the reduced situation still structurally overlaps the expected
        outcome. MCS ≥ θ_containment is that check, using the same
        containment threshold the rest of the module uses for signatures.
        """
        outcome_sigs = self._collect_signatures(outcome)
        current = situation

        elements: List[Any] = list(current.nodes) + list(current.edges)
        elements.sort(key=self._element_sort_key)

        i = 0
        while i < len(elements):
            element = elements[i]

            # Attempt removal of `element` from `current`.
            if isinstance(element, int):
                if element not in current.nodes:
                    # Already removed as a side effect of a prior step
                    # (never happens for nodes, but defensive).
                    elements.pop(i)
                    continue
                trial_nodes = current.nodes - {element}
                trial_edges = frozenset(
                    (s, d, et)
                    for (s, d, et) in current.edges
                    if s != element and d != element
                )
            else:
                if element not in current.edges:
                    # Removed as a side effect of a node removal.
                    elements.pop(i)
                    continue
                trial_nodes = current.nodes
                trial_edges = current.edges - {element}

            trial = Subgraph(
                nodes=frozenset(trial_nodes),
                edges=frozenset(trial_edges),
                anchor_of_node=anchor_positions_for(
                    trial_nodes,
                    current.anchor_of_node,
                    self.cg.pattern_node_position,
                ),
            )
            trial_sigs = self._collect_signatures(trial)
            mcs, _ = mcs_score(trial, outcome, trial_sigs, outcome_sigs)
            if mcs >= THETA_CONTAINMENT:
                current = trial
                elements.pop(i)
                # Do not advance i: the next unprocessed element is now at i.
            else:
                i += 1
        return current

    def _get_or_build_query_core_category(
        self,
        window: Subgraph,
        label: Any,
        t: int,
    ) -> Subgraph:
        window_hash = self._window_hash(t, window)
        # [D2 / R5b] Post-R5 the query core IS the raw window for every
        # window above CORE_SIZE_CAP_NODES, independent of the label
        # history: the projection that consumed the history is
        # unreachable. A history-derived cache key therefore bought no
        # discrimination while guaranteeing a miss whenever the history
        # tail changed -- three extra WL hashes per call for nothing.
        # hist_state is fixed to None; the TTL (QUERY_CACHE_TTL) is now
        # the only invalidation.
        #
        # If R5 is ever removed (a future benchmark produces windows
        # short enough to compress meaningfully), restore the
        # history-derived key alongside it.
        hist_state: Optional[bytes] = None
        key: Tuple[bytes, CoreKind, Optional[bytes]] = (
            window_hash,
            CoreKind.CATEGORY,
            hist_state,
        )
        cached = self._query_cache.get(key)
        if cached is not None and (t - cached[1]) <= QUERY_CACHE_TTL:
            self.diagnostics["query_cache_hits"] += 1
            return cached[0]
        self.diagnostics["query_cache_misses"] += 1

        core = self._extract_query_core_category(window, label)
        self._query_cache[key] = (core, t)
        return core

    def _extract_query_core_category(
        self,
        window: Subgraph,
        label: Any,
    ) -> Subgraph:
        """§6.3: discriminative-mode intersection + minimal extension.
        Requires at least θ_category_min_positives positive windows; if
        fewer are available, match the raw situation subgraph directly.

        [SPEC INTERPRETATION §6.3] At query time we run the intersection
        + materialize steps only. The extension and prune steps require a
        stable positive/negative classification set, which the query-time
        context does not provide. The Consolidator's discriminative_core
        is the canonical path for building CATEGORY cores; the Mapper's
        query-time core is a lightweight projection for matching.
        """
        # [SPEC INTERPRETATION §6.3] Node-count gate, matching the action
        # path's reading. See _extract_query_core_action for rationale.
        if len(window.nodes) <= CORE_SIZE_CAP_NODES:
            return window

        # [D2 / R5] Windows above the core size cap: the §6.3 projection
        # is a no-op at situation scale (measured 2026-09-24: 96% node
        # retention on minigrid's 50-node windows, ~1.7 s/call). Post-R5
        # the query core for this window is the raw window. The historical
        # projection path below is unreachable while this early return is
        # present; remove only if a future benchmark produces
        # sub-20-node windows where the projection would compress
        # meaningfully.
        self.transfer_loop_diag["query_core_raw_over_cap"] += 1
        return window

        history = self._label_history.get(label)
        if history is None or len(history) < THETA_CATEGORY_MIN_POSITIVES:
            return window

        positives = list(history)[-THETA_CATEGORY_MIN_POSITIVES:]
        # Explicit consensus threshold: the delegated operator's default
        # (0.75) is the standalone-vision-benchmark reading; the pipeline
        # path keeps the constitutional THETA_SIM_CONSENSUS binding.
        template = _structural_intersection(
            positives, self._sig_lookup, P.THETA_SIM_CONSENSUS
        )
        if template is None or not template.nodes:
            return window

        # Deterministic exemplar: lowest WL hash (§5.3.2 representative).
        exemplar = min(
            positives,
            key=lambda sg: _subgraph_wl_hash(sg, self._sig_lookup),
        )
        materialized = _materialize(template, exemplar, self._sig_lookup)
        if _subgraph_size(materialized) == 0:
            return window
        return materialized

    # -------------------------------------------------------------------------
    # Matching helpers
    # -------------------------------------------------------------------------

    def _theta_mcs_for_entry(self, entry: CoreEntry) -> float:
        """§6.5 Level-B threshold."""
        if entry.kind == CoreKind.CATEGORY:
            return THETA_MCS_CATEGORY
        return THETA_MCS_BY_FAMILY.get(entry.family, THETA_MCS_DEFAULT)

    def _theta_evidence_for_entry(self, entry: CoreEntry) -> float:
        if entry.kind == CoreKind.CATEGORY:
            return THETA_EVIDENCE_CATEGORY
        return THETA_EVIDENCE_BY_FAMILY.get(entry.family, THETA_EVIDENCE_DEFAULT)

    def _sig_lookup(self, nid: int) -> Optional[Tuple[str, TypedSignature]]:
        if nid in self._sig_cache:
            return self._sig_cache[nid]
        info = self._sig_provider(nid)
        self._sig_cache[nid] = info
        return info

    def _concepts_containing(self, nid: int) -> Set[int]:
        """Return the set of concept IDs whose members include this node.

        Uses ConceptGraph._node_to_concepts if present (a maintained reverse
        index in the current implementation). Falls back to a linear scan
        so the Mapper remains correct if that private index is ever removed.
        """
        rev = getattr(self.cg, "_node_to_concepts", None)
        if rev is not None:
            return rev.get(nid, set())
        result: Set[int] = set()
        for cid, concept in self.cg.concepts.items():
            if nid in concept.members:
                result.add(cid)
        return result

    def _live_pattern_node_ids(self) -> Set[int]:
        """Return the set of currently-live pattern node IDs.

        Uses ConceptGraph._pattern_node_key_by_id (spec §1.11 pattern-node
        store). Falls back to an empty set if the attribute is absent;
        callers handle the empty case by producing an empty situation.
        """
        store = getattr(self.cg, "_pattern_node_key_by_id", None)
        if store is None:
            return set()
        return set(store.keys())

    def _collect_signatures(self, G: Subgraph) -> Dict[int, Tuple[str, TypedSignature]]:
        result: Dict[int, Tuple[str, TypedSignature]] = {}
        for nid in G.nodes:
            info = self._sig_lookup(nid)
            if info is not None:
                result[nid] = info
        return result

    def _node_activation(self, nid: int) -> float:
        """§1.11: A_node(n, t) = max over concepts c : n ∈ members(c) of
        A_concept(c, t)."""
        best = 0.0
        for cid in self._concepts_containing(nid):
            concept = self.cg.concepts.get(cid)
            if concept is not None:
                a = float(getattr(concept, "A", 0))
                if a > best:
                    best = a
        return best

    def _node_max_ef(self, nid: int) -> float:
        """§1.11: EF is a concept-level property; per-node aggregate is
        max over containing concepts."""
        best = 0.0
        for cid in self._concepts_containing(nid):
            concept = self.cg.concepts.get(cid)
            if concept is not None:
                ef = float(getattr(concept, "EF_combined", 0.0))
                if ef > best:
                    best = ef
        return best

    def _node_canonical_key(self, nid: int) -> Tuple[bytes, bytes, int]:
        """Deterministic ordering key for a node: (lens_type,
        signature_hash, node_id). Falls back to a _PRUNED_ sentinel for
        nodes whose signature is no longer available."""
        info = self._sig_lookup(nid)
        if info is None:
            return (b"_PRUNED_", b"\x00" * 16, nid)
        return (info[0].encode("utf-8"), _node_signature_hash(info[1]), nid)

    def _element_sort_key(self, element: Any) -> Tuple[int, bytes]:
        """Deterministic sort key for a mixed node/edge element list."""
        if isinstance(element, int):
            return (0, hash_128(serialize_int(element)))
        et = element[2]
        et_val = et.value if isinstance(et, EdgeType) else int(et)
        return (
            1,
            hash_128(
                serialize_tuple(
                    [
                        serialize_int(element[0]),
                        serialize_int(element[1]),
                        serialize_int(et_val),
                    ]
                )
            ),
        )

    # -------------------------------------------------------------------------
    # Step bookkeeping
    # -------------------------------------------------------------------------

    def _window_hash(self, t: int, sg: Subgraph) -> bytes:
        """[D2 / R5b] Memoized `_subgraph_wl_hash`.

        Step-scoped and keyed on object IDENTITY (`is`): the same Subgraph
        object asked for its hash more than once within a step -- the
        query_category hash gate, the query-cache key -- hashes once
        instead of once per consumer. Correctness never depends on the
        memo: a different object with equal content simply re-hashes (the
        WL hash is canonical, so the values agree).
        """
        memo = self._step_window_hash
        if memo is not None and memo[0] == t and memo[1] is sg:
            return memo[2]
        h = _subgraph_wl_hash(sg, self._sig_lookup)
        self._step_window_hash = (t, sg, h)
        return h

    def _begin_step(self, t: int) -> None:
        if self._current_step != t:
            self._current_step = t
            self.diagnostics["current_step"] = t
            self._sig_cache.clear()
            # [PERFORMANCE] Situation-subgraph memo is step-scoped: both
            # intra-step callers (UtilityEngine step_41 and Prediction's
            # situation scope) read identical graph state — activations
            # settle in Phase 3, edges in Phase 5, both BEFORE the first
            # call — so the second build is pure duplication (measured
            # ~25 ms/step on rich graphs).
            self._situation_memo = None
            self._step_window_hash = None
            self._evict_query_cache(t)

    def _evict_query_cache(self, t: int) -> None:
        stale = [
            k
            for k, (_v, t_cached) in self._query_cache.items()
            if t - t_cached > QUERY_CACHE_TTL
        ]
        for k in stale:
            del self._query_cache[k]

    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------

    def get_last_query_core(self) -> Optional[Subgraph]:
        return self._last_query_core

    def get_last_query_kind(self) -> Optional[CoreKind]:
        return self._last_query_kind

    def get_diagnostics(self) -> Dict[str, Any]:
        out = dict(self.diagnostics)
        # [SPEC COMPLIANCE §5.7] The drop happens in
        # CoreLibrary.get_candidates_for_query; expose it here under the
        # spec-mandated diagnostic name.
        out["mcs_budget_dropped"] = self.cl.dropped_by_budget_count
        out["similarity_graph_edge_count"] = len(self._similarity_edges)
        out["similarity_graph_library_size"] = len(self.cl.entries)
        out.update(self.transfer_loop_diag)
        return out

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Clear episode-local state. The core library is owned by the
        Consolidator and persists across episodes (§4.3)."""
        self._sig_cache.clear()
        self._query_cache.clear()
        self._label_history.clear()
        self._label_hash_history.clear()
        self._gap_queue.clear()
        self._current_step = -1
        self._last_query_core = None
        self._last_query_kind = None
