"""
MODULE 9: Pipeline — v31 orchestrator.

Pure orchestration of the per-step loop. No business logic.
"""

from __future__ import annotations
from core.hashing import (  # noqa: F401
    clamp,
    hash_128,
    hash_64,
    serialize_int,
    serialize_value,
)
import math
import time
import threading
import itertools
import logging
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple
from collections import defaultdict, deque

import parameters as P
from core.hashing import (
    _detection_sort_key,
    _node_signature_hash,
)
from core.spatial import _pos_dist
from core.types import EdgeType
from concept_graph import ConceptFamily, ConceptStatus
from consolidator import (
    CoreEntry,
    CoreKind,
    CategoryManager,
    AspectGraph,
    build_view_descriptor,
    build_view_descriptor_from_chains,
)
from core.graph_types import Subgraph, anchor_positions_for, edge_params_for
from matching.structural_similarity import subgraph_structural_similarity
from core.handoff import ConsolidationHandoff, LabelSignal, MacroExecutionOutcome
import derived_lenses as derived_lenses_mod  # [MEMORY DIAGNOSTIC] _SIM_MEMO
from derived_lenses import (
    ShockEnsembleLens,
    BorderOwnershipLens,
    ContourGainState,
    ContourLens,
    CurveComplexLens,
    JunctionLens,
    EndStoppedLens,
    TextureBoundaryLens,
    SurfaceLens,
    RegionShapeLens,
    IllusoryContourLens,
    HolisticShapeLens,
)
from perception.lenses.peak_lens import PeakLens
from perception.lenses.chain_lens import ChainLens  # [A5]
from lenses import PolarityLens
from sensory_floor import (
    ACTION_LENS_NAME,
    Detection,
    TypedSignature,
    resolve_anchor_position,
    serialize_detection,
)
from core.registries import (  # noqa: F401
    is_action_lens_eligible,
    is_noop_action,
    is_probe_action,
    is_teleological_action,
)

logger = logging.getLogger(__name__)
from core.registries import (  # noqa: F401
    is_action_lens_eligible,
    is_noop_action,
    is_probe_action,
    is_teleological_action,
)



# --- moved to action/decode.py (Move 5d), re-exported ---
from action.decode import decode_action_sequence  # noqa: F401
# -----------------------------------------------------------------------------
# CONSTANTS — bound from parameters.py (single source of truth)
# -----------------------------------------------------------------------------
DIAG_INTERVAL = P.DIAG_INTERVAL
EBAF_TEST_MODE = P.EBAF_TEST_MODE
T_INJECT = P.EBAF_T_INJECT

KAPPA_ACT = P.KAPPA_ACT
BETA_SUPPORT = P.BETA_SUPPORT
K_CONTEXT = P.K_CONTEXT

# §8.11 analogical registry cadence.
K_ANALOGICAL = P.K_ANALOGICAL


# =============================================================================
# [PHASE B] B.4 SURROUND SUPPRESSION
# =============================================================================


def observation_modality(observation: Any) -> str:
    """'image' or 'symbolic' for this observation.

    Reuses the codebase's existing capability discriminator rather than
    inventing a second one: the Boundary lens (lenses.py) already splits
    behaviour on "does this observation expose per-entity shape channels"
    (Observation2D/Observation3D carry `get_circularity`; SymbolicObservation
    does not). B.1 scopes the whole higher tier to image modality, so that
    same probe is the correct gate here.
    """
    return "image" if hasattr(observation, "get_circularity") else "symbolic"


def _anchor_pos(
    anchor: Any,
    anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
) -> Optional[Tuple[float, ...]]:
    if not anchor_positions:
        return None
    return anchor_positions.get(anchor)




def surround_suppress(
    dets: FrozenSet[Detection],
    anchor_positions: Dict[Any, Tuple[float, ...]],
    observation_modality_: str,
    extraction_scale: float,
) -> FrozenSet[Detection]:
    """B.4 non-max suppression of near-duplicate V1 detections.

    Within one lens type, a detection is dropped when a strictly higher
    scoring detection sits within KAPPA_SURROUND_SUPPRESS * extraction_scale
    and is at least as similar (THETA_SURROUND_SUPPRESS).

    [PHASE B] Step 4. The neighbourhood radius is a multiple of the
    extraction quantum, NOT an absolute length: the draft's absolute 1.5
    was in grid-cell units while anchor_positions is in pixels/meters, so
    on the Phase-A grid (min anchor separation 32px) it suppressed nothing
    at all. A non-positive extraction_scale means the observation exposes
    no region radii, so there is no metric neighbourhood to work in and
    the set passes through untouched.
    """
    if not anchor_positions or observation_modality_ == "symbolic":
        return dets
    if extraction_scale <= 0.0:
        return dets
    sample_pos = next(iter(anchor_positions.values()), None)
    if sample_pos is None:
        return dets
    dims = min(3, len(sample_pos))
    if any(len(p) < dims for p in anchor_positions.values()):
        return dets
    radius = P.KAPPA_SURROUND_SUPPRESS * extraction_scale
    by_k: Dict[str, List[Detection]] = defaultdict(list)
    for d in sorted(dets, key=_detection_sort_key):
        by_k[d.k].append(d)
    kept: Set[Detection] = set()
    for k, group in sorted(by_k.items()):
        group_sorted = sorted(
            group, key=lambda d: (-d.z, hash_64(serialize_detection(d)))
        )
        grid: Dict[
            Tuple[int, ...],
            List[Tuple[TypedSignature, Tuple[float, ...]]],
        ] = defaultdict(list)
        cell_size = radius
        for d in group_sorted:
            pos = _anchor_pos(d.alpha, anchor_positions)
            if pos is None:
                kept.add(d)
                continue
            cell = tuple(
                int(math.floor(pos[i] / cell_size)) for i in range(dims)
            )
            suppressed = False
            for off in itertools.product((-1, 0, 1), repeat=dims):
                nc = tuple(cell[i] + off[i] for i in range(dims))
                for (other_sig, other_pos) in grid.get(nc, ()):
                    if _pos_dist(pos, other_pos) < radius:
                        if (
                            d.sigma.similarity(other_sig)
                            >= P.THETA_SURROUND_SUPPRESS
                        ):
                            suppressed = True
                            break
                if suppressed:
                    break
            if not suppressed:
                kept.add(d)
                grid[cell].append((d.sigma, pos))
    return frozenset(kept)


class Pipeline:
    """Orchestrates the v31 per-step loop."""

    def __init__(
        self,
        env,
        sensory_floor,
        derived_lenses,
        binder,
        builder,
        providers,
        prediction,
        concept_graph,
        activation_engine,
        fertility_engine,
        utility_engine,
        consolidator=None,
        structural_potential_map=None,
        mapper=None,
        innate_reflex=None,
        optimize_recognize_cache: bool = False,
        self_binders=None,
    ):
        self.env = env
        self.sensory_floor = sensory_floor
        self.derived_lenses = derived_lenses
        self.binder = binder
        self.self_binders = self_binders
        self.builder = builder
        self.providers = providers
        self.prediction = prediction
        self.consolidator = consolidator
        self.mapper = mapper
        # [PHASE B] B.13: the six higher-tier lens instances. Held on the
        # Pipeline (not the lens executor) because they are a SECOND tier —
        # they consume this step's V1 output, whereas the executor's lenses
        # produce it. Inert while HIGHER_LENS_ENABLED is False.
        self.border_ownership_lens = BorderOwnershipLens()
        self.surface_lens = SurfaceLens()
        self.contour_lens = ContourLens()
        self.junction_lens = JunctionLens()
        self.peak_lens = PeakLens()
        self.chain_lens = ChainLens()  # [A5]
        self.illusory_contour_lens = IllusoryContourLens()
        self.curve_complex_lens = CurveComplexLens()
        self.region_shape_lens = RegionShapeLens()
        self.holistic_shape_lens = HolisticShapeLens()
        # [Stage 4] Shock ensemble lens for medial axis ribbons/ensembles.
        self.shock_ensemble_lens = ShockEnsembleLens()
        # [Stage 5] Aspect graph memory engine (inert unless P.ASPECT_ENABLED).
        self.aspect_graph = AspectGraph()
        # [B.30 Tier-1 Fix A] Luminance-polarity lens. Runs from the
        # raw-image route's higher-tier pass only; region observations
        # carry no Gabor responses and it emits nothing.
        self.polarity_lens = PolarityLens()
        self.endstopped_lens = EndStoppedLens()
        self.texture_boundary_lens = TextureBoundaryLens()
        # [PHASE B] B.14 recurrent contour grouping (Step 9 wires it).
        self.contour_gain_state = ContourGainState()
        # [PHASE B] Step 4: one-shot "this lens is wired but inert" flags,
        # so a silently-zero lens is reported exactly once per run rather
        # than every step. Steps 5-9 add a flag per lens.
        self._border_ownership_silent_logged: bool = False
        # [PHASE B] Step 5.
        self._contour_silent_logged: bool = False
        self._junction_silent_logged: bool = False
        self._chain_silent_logged: bool = False  # [A5]
        # [PHASE B] Step 6.
        self._curve_complex_silent_logged: bool = False
        # [PHASE B] Step 7.
        self._part_silent_logged: bool = False
        # [PHASE B] Step 8 (HolisticShape).
        self._holistic_shape_silent_logged: bool = False
        # [Stage 4] Shock ensemble lens.
        self._shock_ensemble_silent_logged: bool = False
        # [PHASE B] Last step's extraction quantum / modality, for the
        # higher-lens probe.
        self._last_extraction_scale: float = 0.0
        self._last_modality: str = "symbolic"
        # [PHASE B] higher-lens diagnostics, surfaced by get_size_snapshot
        # so a silent zero-emission lens is visible.
        self._higher_lens_diagnostics: Dict[str, Dict[str, int]] = {}
        # [V9 Phase 4e / draft B.14] Settle loop, built lazily and used
        # only for raw-image observations (see step()).
        self._settle_loop: Optional[Any] = None
        self._higher_tier_candidates: Dict[str, int] = {}
        self._higher_tier_emissions: Dict[str, int] = {}
        # Unprocessed (P_hash, S_hash) pairs awaiting subgraph
        # reconstruction. Bounded: drained every step when the
        # Consolidator is wired; capped otherwise (see _consume_pairs).
        self._pending_consolidation_pairs: List[Tuple[bytes, bytes]] = []
        # [B1/D6] Rich handoffs from the Builder's resolved §4.15.3
        # traces: induced subgraphs + crystallization states. Drained
        # every step; a handoff supersedes its raw hash pair (the pair
        # list stays for diagnostics and for unpackageable resolutions).
        self._pending_handoffs: List[ConsolidationHandoff] = []
        # [Seam 1] Buffered macro-execution outcomes awaiting their
        # §8.5 transfer validation (bounded per core, retention
        # 2·W_VALIDATION steps).
        self._macro_outcomes_by_core: Dict[bytes, List[MacroExecutionOutcome]] = {}
        self.consolidation_diagnostics: Dict[str, int] = defaultdict(int)
        # [Probe B.1] Materialize the bootstrap counter keys eagerly so
        # end-of-run dumps distinguish "wired but zero" from "absent"
        # (defaultdict only creates keys on increment). The C-introduced
        # bootstrap_single_binding_fallback is intentionally not pre-set
        # (appears on first increment).
        for _k in (
            "bootstrap_unbound",
            "bootstrap_no_sequence_edge",
            "bootstrap_replay_failed",
            "bootstrap_no_replayable_start",
            "bootstrap_no_aem",
            "bootstrap_path_unresolvable",
            "bootstrap_cores_extracted",
        ):
            self.consolidation_diagnostics[_k] = 0
        # [B3/§5.8.2] Per-label observation-window ring buffer feeding
        # the Consolidator's discriminative trigger. Labels are derived
        # from the pipeline's own effect-atom tracking (env-agnostic);
        # windows come from the StatePatternIndex (already maintained).
        self.window_buffer = ObservationWindowBuffer(
            capacity_per_label=P.LABEL_WINDOW_CAPACITY,
            min_instances=P.LABEL_MIN_INSTANCES,
            max_age_steps=P.LABEL_MAX_AGE_STEPS,
        )
        self.window_buffer.attach_sig_provider(
            self.providers.signature_provider
        )
        # [Phase 1 Decision 2] Category lifecycle manager. Extractor
        # stays on the pipeline (injected as a callable). Constructed
        # only when a Consolidator is wired; otherwise None.
        self.category_manager = None
        if consolidator is not None:
            self.category_manager = CategoryManager(
                core_library=consolidator.get_core_library(),
                mapper=mapper,
                window_buffer=self.window_buffer,
                concept_graph=concept_graph,
                consolidation_diagnostics=self.consolidation_diagnostics,
                extract_fn=lambda lbl, pos, neg, tt: self._extract_category_core(
                    tt, lbl, pos, neg, source_concept_id=None,
                ),
                diversity_fn=self._compute_diversity,
            )
            self.category_manager.start_session()
        # Consolidator lifecycle state (rulings: Trigger 1A/1C/2A).
        # Bounded by concept count; concepts are capacity-bounded (§2.1.15).
        self._category_consolidated: Set[bytes] = set()
        # [G.2.2] Bounded-retry counter for concepts whose extraction
        # returned None. Caps re-attempts so a genuinely unextractable
        # concept does not consume sweep budget forever, while allowing
        # a budget-starved concept to succeed on a subsequent sweep.
        #
        # Persists across episodes (like `_category_consolidated` and the
        # Core Library per §4.3): a concept that exhausted its retry
        # budget in episode N is not re-attempted in episode N+1. This
        # matches library persistence semantics — the abstraction the
        # core would have produced is a run-scoped artifact, not
        # episode-scoped.
        self._category_consolidation_attempts: Dict[bytes, int] = defaultdict(int)
        self._bootstrap_attempts: Dict[bytes, int] = defaultdict(int)
        # [Batch D2] Step-scoped published surface for §6.6.2 category
        # matches: matched label -> best confidence among this step's
        # category queries. Rebuilt at the start of every category-query
        # pass (so it always describes exactly one step) and bounded by
        # K_CATEGORY_QUERY_MAX queries x the labels each match returns.
        self._category_match_this_step: Dict[Any, float] = {}
        # concept id → member pattern nodes carrying the concept's bound
        # effect atoms (Onset-lens detections; node namespace). Used to
        # build Trigger 1A consequence subgraphs. Bounded by concept count.
        self._effect_nodes_by_concept: Dict[int, Set[int]] = {}
        # Event-time schema binding (v31 wiring): 6-tuples
        # (action, atom, cid, onset_nodes, onset_t, homing_t) buffered at
        # the onset step — when attribution is observable — and committed
        # once Prediction validates necessity (H_nec, §8.9). [Batch D1b]
        # cid=None marks a VOID (rise with no active causal claimant);
        # homing_t is the step _reattribute_pending_voids claimed it
        # (None for direct registrations). Bounded deque; stale triples
        # for dead concepts are dropped on commit; void / homed entries
        # expire on W_REATTRIB / W_HOMED_TTL.
        self._pending_bindings: deque = deque(maxlen=128)
        # frozenset(members) → sole validated action, for the extractor's
        # single-action fallback (1-step causal schemas have no SEQUENCE
        # structure to walk). Bounded by concept count.
        self._single_action_by_members: Dict[frozenset, Any] = {}
        # Actions whose causal schema link is counterfactually validated
        # (H_nec at binding-commit time) — the D4 confirmation route for
        # the Consolidator's ruling Item-1 replay gate. Bounded by
        # repertoire size.
        self._validated_binding_actions: Set[Any] = set()

        # §5.3.4 wiring: the Consolidator's replay resolves path subgraphs
        # to real actions through the injected extractor. [B2] Primary
        # route is the Action-lens registry decode (exact action objects,
        # state-independent); the AEM walk survives as the legacy-concept
        # fallback. Without either, replay fails honestly and counts
        # `replay_action_extraction_failed` — the former node-ID
        # "extraction" could never match an action and silently starved
        # the CoreLibrary.
        if self.consolidator is not None:
            aem = getattr(env, "action_edge_map", None)
            rep_actions = getattr(env, "action_rep", None)
            if aem is not None and rep_actions:
                base_extractor = make_action_extractor(
                    aem,
                    self.providers.state_pattern_provider,
                    list(rep_actions),
                    concept_graph=concept_graph,
                )
                pipeline_ref = self

                def _extractor_with_binding_fallback(G, s0):
                    seq = base_extractor(G, s0)
                    if seq is None or seq:
                        return seq
                    # seq == []: G carries no SEQUENCE structure. For a
                    # 1-step causal schema the validated binding IS the
                    # action sequence (§5.3.4 actions(G) for single-action
                    # cores — the ruling's "Near Switch → Toggle → Door
                    # Open" example). Registry lookup by member set is
                    # exact and deterministic.
                    fallback = pipeline_ref._single_action_by_members.get(
                        frozenset(G.nodes)
                    )
                    return [fallback] if fallback is not None else seq

                self.consolidator.set_action_extractor(
                    _extractor_with_binding_fallback
                )
                pipeline_ref2 = self
                self.consolidator.set_schema_confirmer(
                    lambda action: action in pipeline_ref2._validated_binding_actions
                )
        self.concept_graph = concept_graph
        self.activation_engine = activation_engine
        self.fertility_engine = fertility_engine
        self.utility_engine = utility_engine
        self.structural_potential_map = structural_potential_map
        self.innate_reflex = innate_reflex

        # [COUNTERFACTUAL-GATED BINDING §8.3] Wire the prediction module into
        # the concept graph so gate_bind consults counterfactual evidence.
        self.concept_graph.prediction_module = prediction

        # The former attention_gate/attention_threshold flags gated D_bind
        # by concept attention. That contradicts v30 §1.3 (attention is
        # READ-ONLY and selects concepts for proposal generation — it never
        # filters binding evidence) and its keep-set hard-coded lens names
        # ("Boundary", "Topology", "Degree" — two of which are BINDERS in
        # the v31 taxonomy). Removed; D_bind = D_base ∪ D_op per §4.2
        # step 17. Proposal-level gating lives in UtilityEngine's
        # top-K_plan goal pool, which uses contextual fertility as the
        # blueprint specifies.
        self.optimize_recognize_cache = optimize_recognize_cache

        self._current_concept_effects: Set[int] = set()

        self.learning_enabled = True

        self.t = 0
        self.e = 0

        self.prev_state = None
        self.prev_action = None
        self.prev_D_bind: Set = set()
        self.prev_D_rec: Set = set()
        self.last_state_hash = None
        self.all_effect_flags: Set[Any] = set()
        # [Batch B] Effect-presence register. The prediction module
        # holds a REFERENCE to this exact dict after wiring; the only
        # legal mutation is .clear() (or per-key writes). Never rebind
        # (e.g. self.effect_presence_register = {}) — that would orphan
        # prediction's reference.
        self.effect_presence_register: Dict[int, bool] = {}

        self.a_hist: Dict[int, Any] = {}
        self.onset_history: Dict[Any, Dict[int, int]] = defaultdict(dict)
        self.prev_effect_flags: Dict[Any, int] = {}
        self._validated_transfer_outcomes: List[Tuple[bytes, bool, int, int]] = []
        # [Batch D / P1-6 MED-04] Known action vocabulary: env repertoire
        # + already-crystallized action concepts. Novel eligible actions
        # crystallize a new ATOMIC_ACTION concept at execution time.
        self._known_actions: Set[Any] = set(
            getattr(env, "action_rep", ()) or ()
        )
        for _c in concept_graph.concepts.values():
            if getattr(_c, "action", None) is not None:
                self._known_actions.add(_c.action)

        if KAPPA_ACT >= (1 - BETA_SUPPORT) / 4:
            raise RuntimeError(
                "Timescale separation invariant violated: κ_act must be < (1-β_support)/4"
            )

        self.diagnostics_log: List[Dict] = []
        # [Batch 3 Phase 3] Pipeline-level scalar diagnostics surface.
        # [ADAPTATION] Pipeline previously had no plain `diagnostics`
        # dict (only diagnostics_log); created here because the Batch-3
        # CDP tracker writes cdp_count_current to it. Nothing else reads
        # or writes it, so default-off runs are byte-identical.
        self.diagnostics: Dict[str, Any] = {}
        # [Batch 3 Phase 3] CDP-extinction tracker (flag-gated, default
        # OFF; read-side observation only — no inference, matching,
        # family-resolution, or graph-dynamics writes).
        self._cdp_tracking_enabled: bool = False
        self._cdp_tracker: Dict[int, int] = {}
        self._cdp_transitions: List[Dict] = []
        # [P0-4] Profiling harness.
        self._profiling_enabled: bool = False
        self._profiling_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "total_s": 0.0}
        )
        self.max_diagnostics = 1000
        self._proposal_origin_counts: Dict[str, int] = defaultdict(int)

    def set_learning_mode(self, enabled: bool):
        self.learning_enabled = enabled

    def enable_cdp_tracking(self) -> None:
        """[Batch 3 Phase 3] Turn on the read-side CROSS_DOMAIN_PATTERN
        tracker (per-DIAG-tick population snapshots + entered/left
        transition log). Observation only; no behavioral effect."""
        self._cdp_tracking_enabled = True

    # -------------------------------------------------------------------------
    # Step loop
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # [P0-4] Profiling harness
    # -------------------------------------------------------------------------

    def enable_profiling(self) -> None:
        self._profiling_enabled = True

    def _log_p1_3_triggers(self, event: Dict, t: int) -> None:
        """[P1-3 / Bundle A] rolling trigger log for fertile-threshold
        preconditions of succeeded plans. Bounded at 10,000 entries; once
        capped, further entries increment a counter instead (reported in
        consolidation diagnostics, never silent)."""
        import json
        import os
        if not hasattr(self, "_p1_3_log_count"):
            self._p1_3_log_count = 0
            self._p1_3_log_capped = 0
            self._p1_3_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "audit_out",
                "p1_3_trigger_log.jsonl",
            )
        threshold = getattr(self.builder, "_theta_fertile", None)
        if threshold is None:
            return
        for cid in event.get("basis", ()):
            concept = self.concept_graph.concepts.get(cid)
            if concept is None:
                continue
            rf = getattr(concept, "RF", 0.0)
            if rf < threshold:
                if self._p1_3_log_count >= 10_000:
                    self._p1_3_log_capped += 1
                    self.consolidation_diagnostics["p1_3_log_capped"] = (
                        self._p1_3_log_capped
                    )
                    continue
                # [FIX 2026-09-25] A diagnostics logger must never kill the
                # cognitive loop: create the audit_out directory on demand and
                # degrade any OSError to a counted diagnostic instead of
                # propagating out of pipeline.step().
                try:
                    os.makedirs(
                        os.path.dirname(self._p1_3_path), exist_ok=True
                    )
                    with open(self._p1_3_path, "a", encoding="utf-8") as fh:
                        fh.write(
                            json.dumps(
                                {
                                    "plan_id": event.get("core_hash", b"").hex()
                                    if isinstance(event.get("core_hash"), bytes)
                                    else str(event.get("core_hash")),
                                    "precondition_id": cid,
                                    "RF": rf,
                                    "fertile_threshold": threshold,
                                    "step": t,
                                }
                            )
                            + "\n"
                        )
                except OSError:
                    self.consolidation_diagnostics["p1_3_log_write_failed"] = (
                        self.consolidation_diagnostics.get(
                            "p1_3_log_write_failed", 0
                        )
                        + 1
                    )
                    continue
                self._p1_3_log_count += 1

    def _run_phase(self, label: str, fn: Callable, *args, **kwargs) -> Any:
        """Wrap a phase call with perf_counter timing when profiling is on.
        When off this compiles down to a single branch prediction."""
        if not self._profiling_enabled:
            return fn(*args, **kwargs)
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            s = self._profiling_stats[label]
            s["calls"] += 1
            s["total_s"] += dt

    def get_profiling_report(self) -> Dict[str, Any]:
        out = {}
        for label, s in self._profiling_stats.items():
            out[label] = {
                "calls": s["calls"],
                "total_s": s["total_s"],
                "mean_ms": 1000.0 * s["total_s"] / max(1, s["calls"]),
            }
        return out

    def get_size_snapshot(self) -> Dict[str, int]:
        return {
            "edges": len(self.concept_graph.edges),
            "pattern_nodes": len(self.concept_graph.get_live_pattern_node_ids()),
            "concepts": len(self.concept_graph.concepts),
            "candidates": len(self.concept_graph.candidates),
            "frontier": len(self.builder.frontier),
            "transposition_table": len(self.builder.transposition_table),
            "basins": len(self.builder.basins),
            "core_library": (
                self.consolidator.core_library.size()
                if self.consolidator is not None else 0
            ),
            # [MEMORY DIAGNOSTIC] Edge store memory proxy.
            "edge_store_count": len(self.concept_graph.edges),
            # [MEMORY DIAGNOSTIC] Pattern node store count.
            "pattern_node_count": len(
                self.concept_graph._pattern_node_key_by_id
            ),
            # [MEMORY DIAGNOSTIC] Binder M_prev pressure (all channels).
            "binder_m_prev_total": sum(
                len(ch.get("M_prev", {}))
                for ch in self.binder.channels.values()
            ) if hasattr(self.binder, "channels") else 0,
            # [MEMORY DIAGNOSTIC] Derived lens cache pressure.
            "derived_lens_recog_cache": len(
                getattr(self.derived_lenses, "_recog_evidence_cache", {})
            ) if hasattr(self, "derived_lenses") else 0,
            "derived_lens_sim_memo": len(
                getattr(derived_lenses_mod, "_SIM_MEMO", {})
            ),
            # [MEMORY DIAGNOSTIC] Builder WL memo + transposition table.
            "builder_wl_memo_size": len(
                getattr(self.builder, "_wl_memo", {})
            ) if hasattr(self.builder, "_wl_memo") else 0,
            "builder_transposition_size": len(
                getattr(self.builder, "transposition_table", {})
            ) if hasattr(self.builder, "transposition_table") else 0,
            # [MEMORY DIAGNOSTIC] Consolidator core library (duplicate of
            # core_library under the directive's field name).
            "core_library_size": (
                self.consolidator.core_library.size()
                if hasattr(self, "consolidator") and self.consolidator is not None
                else 0
            ),
            # [Bundle A] MCS memo + revision counters.
            "mcs_memo_size": self.consolidator.diagnostics.get(
                "mcs_memo_size_max", 0
            ) if self.consolidator is not None else 0,
            "mcs_cache_hits": self.consolidator.diagnostics.get(
                "mcs_cache_hits", 0
            ) if self.consolidator is not None else 0,
            "mcs_memo_overflow": self.consolidator.diagnostics.get(
                "mcs_memo_overflow", 0
            ) if self.consolidator is not None else 0,
            "node_revision": self.concept_graph.node_revision,
            # [Stage 5] Aspect graph memory size (inert when ASPECT_ENABLED=False).
            "aspect_views": len(self.aspect_graph.views),
            "aspect_protos": len(self.aspect_graph.protos),
            "aspect_links": len(self.aspect_graph.links),
            "fertile_set_turnover_count": getattr(
                self.builder, "fertile_set_turnover_count", 0
            ),
            "reference_V_history_len": len(
                (
                    self.builder.basins[self.builder.spine_basin_id]
                    .reference_V_history
                )
                or ()
            )
            if (
                self.builder.spine_basin_id is not None
                and self.builder.spine_basin_id in self.builder.basins
            )
            else 0,
        }

    def _build_concept_effects(self) -> Set[int]:
        """Effect atoms relevant to the current reasoning episode.

        Two sources, unioned:
          (A) Concepts with A == 1 at the previous step. Activation is
              the graph's signal for 'currently firing'. The activation
              engine writes concept.A at Phase 3, so at lens time we
              read the previous step's value (one-step lag; accepted
              because the pipeline phase order runs lenses before
              activation).
          (B) The source_concept_id of the active transfer, when one is
              set. This is the explicit 'what is being pursued' signal.

        Both are read from state that exists at lens-run time; neither
        depends on step_41's scratch _action_targets map (which is
        reset each step and only sparsely populated)."""
        effects: Set[int] = set()

        # Source A — active concepts.
        for _cid, concept in self.concept_graph.concepts.items():
            if getattr(concept, "A", 0) != 1:
                continue
            bindings = getattr(concept, "effect_ids_by_action", None)
            if not bindings:
                continue
            for atom in bindings.values():
                if atom is not None:
                    effects.add(atom)

        # Source B — active transfer's source concept.
        if self.utility_engine is not None:
            transfer = getattr(self.utility_engine, "_active_transfer", None)
            if transfer is not None:
                cid = getattr(transfer, "source_concept_id", None)
                if cid is not None:
                    concept = self.concept_graph.concepts.get(cid)
                    if concept is not None:
                        bindings = (
                            getattr(concept, "effect_ids_by_action", None)
                            or {}
                        )
                        for atom in bindings.values():
                            if atom is not None:
                                effects.add(atom)

        return effects


    # ------------------------------------------------------------------
    # [V9 Phase 4e / draft B.14] Raw-image settle route.
    # ------------------------------------------------------------------

    def _derive_anchor_positions(self, observation):
        """Anchor positions from the observation, both modality shapes.

        Extracted verbatim from `step` (same statements, same order) so the
        region path is byte-identical and the raw-image route can obtain
        the map before Step 2.
        """
        anchor_positions: Dict[Any, Tuple[float, ...]] = {}
        positions_attr = getattr(observation, "positions", None)
        if positions_attr is None:
            getter = getattr(observation, "get_positions", None)
            if callable(getter):
                positions_attr = getter()
        if positions_attr is not None:
            if isinstance(positions_attr, dict):
                items = positions_attr.items()
            else:
                items = enumerate(positions_attr)
            for idx, pos in items:
                if pos is None:
                    continue
                if not isinstance(pos, (list, tuple)):
                    continue
                if len(pos) < 2:
                    continue
                try:
                    x = float(pos[0])
                    y = float(pos[1])
                except (TypeError, ValueError):
                    continue
                if x == -1.0 and y == -1.0:
                    continue
                if len(pos) >= 3:
                    try:
                        z = float(pos[2])
                    except (TypeError, ValueError):
                        z = 0.0
                    if not math.isfinite(z):
                        z = 0.0
                else:
                    z = 0.0
                anchor_positions[idx] = (x, y, z)
        return anchor_positions

    def _derive_extraction_scale(self, observation) -> float:
        """Extraction quantum (median region radius), 0.0 if unavailable."""
        extraction_scale = 0.0
        _es_getter = getattr(observation, "get_extraction_scale", None)
        if callable(_es_getter):
            try:
                extraction_scale = float(_es_getter() or 0.0)
            except (TypeError, ValueError):
                extraction_scale = 0.0
        return extraction_scale

    def _get_settle_loop(self):
        if self._settle_loop is None:
            from dynamics.settle import SettleLoop, DefaultDerivedRunner
            self._settle_loop = SettleLoop(
                sensory_floor=self.sensory_floor,
                binder=self.binder,
                self_binders=self.self_binders,
                derived=DefaultDerivedRunner(
                    self.derived_lenses, self.concept_graph),
                cg=self.concept_graph,
                higher_lens_runner=self._higher_lens_runner,
            )
        return self._settle_loop

    def _higher_lens_runner(self, t, D_v1, obs, cg, envelope_mask=None):
        """Higher-tier pass for the settle route (draft B.14 shape).

        The six lenses in dependency order over the step's detections.
        Anchors come from the observation (the draft's B21 note: never from
        a Pipeline field nothing populates). The region path does NOT use
        this runner -- it keeps the inline block, which also binds and
        emits per tier.
        """
        anchor_positions = {}
        n_atoms = obs.num_atoms() if hasattr(obs, "num_atoms") else 0
        if hasattr(obs, "get_region_centroid_by_anchor"):
            for u in range(n_atoms):
                p = obs.get_region_centroid_by_anchor(u)
                if p is not None:
                    anchor_positions[u] = p
        extraction_scale = self._derive_extraction_scale(obs)
        current_dets: Dict[str, List[Detection]] = defaultdict(list)
        for d in D_v1:
            current_dets[d.k].append(d)

        out: Set[Detection] = set()
        # Same provider preference as the inline block: the pipeline's own
        # provider for the contour/junction/… tier, the floor's as fallback.
        sig_provider = getattr(
            self.providers, "signature_provider", None
        ) or getattr(self.sensory_floor, "signature_provider", None)
        for lens in (self.border_ownership_lens, self.surface_lens, self.contour_lens,
                     self.junction_lens, self.illusory_contour_lens, self.curve_complex_lens,
                     self.region_shape_lens, self.polarity_lens, self.endstopped_lens,
                     self.texture_boundary_lens):
            results = lens.step(
                t, cg, sig_provider, anchor_positions, None,
                dict(current_dets),
                extraction_scale=extraction_scale,
                envelope_mask=envelope_mask,
                observation=obs)
            for d in results:
                current_dets[d.k].append(d)
                out.add(d)
        return out

    def _settle_loop_step(self, observation, dir_tuples, _ap, _es):
        """Run one real step through the settle loop (raw-image route)."""
        loop = self._get_settle_loop()
        dets, _derived, S_t, s_t, T_prev = loop.step(
            self.t, observation, self.prev_action, dir_tuples,
            self._current_concept_effects,
            anchor_positions=_ap, extraction_scale=_es)
        return dets, S_t, s_t, T_prev

    def step(self):
        """Execute one full v31 step."""
        t = self.t
        # [PHASE B] B.3: open the binder step lifecycle before any
        # phase runs, so finalize_step() in the `finally` below owns
        # the single per-step commit of the previous-step pattern sets.
        self.binder.note_step_begin(t)
        try:

            if hasattr(self.concept_graph, "step_reliability_begin"):
                self.concept_graph.step_reliability_begin(self.t)

            # Step 1: Observation
            observation = self.env.observe()

            # Step 1.5: Innate reflex
            dir_tuples = None
            reflex = self.innate_reflex
            if reflex is None:
                reflex = getattr(self.utility_engine, "innate_reflex", None)
            if reflex is not None:
                reflex.update(self.t, observation)
                dir_tuples = reflex.get_dir_tuples()

            # Step 2: Sensory floor
            self._current_concept_effects = self._build_concept_effects()
            # [V9 Phase 4e / draft B.14] Raw-image observations go
            # through the settle loop; region observations keep the
            # single-pass floor call (byte-identical). The gate is the
            # DATA (a pixel-built Observation2D), not hasattr.
            if getattr(observation, "_raw_image", None) is not None:
                _ap = self._derive_anchor_positions(observation)
                _es = self._derive_extraction_scale(observation)
                D_raw, S_t, s_t, T_prev = self._settle_loop_step(
                    observation, dir_tuples, _ap, _es
                )
            else:
                D_raw, S_t, s_t, T_prev = self._run_phase(
                    "lenses",  # [P0 v4] label per Phase 8.3
                    self.sensory_floor.step,
                    observation,
                    a_prev=self.prev_action,
                    dir_tuples=dir_tuples,
                    concept_effects=self._current_concept_effects,
                    gain_by_anchor=(
                        self.contour_gain_state.gain
                        if P.RECURRENT_GROUPING_ENABLED else None
                    ),
                )
            self.last_state_hash = s_t
            self.concept_graph.set_current_state_hash(s_t)

            if hasattr(self.concept_graph, "update_entity_atom_map") and S_t is not None:
                self.concept_graph.update_entity_atom_map(S_t.entity_members)

            new_anchor_spawned = self.sensory_floor.new_anchor_spawned

            # Phase 2: Action-edge map + structural potential
            if self.learning_enabled:
                # §2.22 delayed-previous-transition update; the returned edge
                # events (newly confirmed edges, §2.19 EdgeConfirmed) feed the
                # potential map's ε_edge bonus. Potential-map updates live in
                # the pipeline (core state), not inside the environment.
                edge_events = self.env.update_action_edge_map(T_prev)
                if T_prev is not None and self.structural_potential_map is not None:
                    self.structural_potential_map.update_from_transition(
                        T_prev,
                        new_edge_count=int(
                            (edge_events or {}).get("just_confirmed", False)
                        ),
                    )

            # Step 8b (§8.11): validate pending transfers whose horizon expired.
            # §8.7 interface: step_validate_transfers(t, concept_graph). The
            # situation subgraph (§6.2) and MCS containment (§6.4.1) reach the
            # Prediction module through the Mapper injection wired in main.py —
            # no activation_engine parameter, no swallowed exceptions (the
            # former print-thrice-then-silence handler hid spec/implementation
            # mismatches behind a counter).
            if self.prediction is not None and self.learning_enabled:
                self.prediction.step_validate_transfers(t, self.concept_graph)

            # §8.5.3 → v31 replacement for the deleted PlanningEngine's
            # plan_outcomes: validated transfer outcomes feed FertilityEngine's
            # plan-success contextual-fertility increments (step 31) and
            # step-blame credit (step 33). Drained every step; buffered for the
            # phase-6 calls below.
            self._validated_transfer_outcomes = (
                self.prediction.drain_validated_transfers()
                if self.prediction is not None and self.learning_enabled
                else []
            )

            # Periodic (§8.11 steps 8e/8f): analogical register + decay
            if self.prediction is not None and self.learning_enabled and t > 0:
                if t % K_ANALOGICAL == 0:
                    self.prediction.step_analogical_register(t, self.concept_graph)
                    self.prediction.step_analogical_decay(t, self.concept_graph)

            # Phase 3: Context + activation
            active_concepts = self.concept_graph.get_active_concepts()
            if active_concepts:
                ranked = sorted(
                    active_concepts,
                    key=lambda cid: math.log(
                        1 + getattr(self.concept_graph.concepts[cid], "EF_combined", 0.0)
                    ),
                    reverse=True,
                )
                top_k = ranked[:K_CONTEXT]
            else:
                top_k = []
            if hasattr(self.activation_engine, "compute_context_signature"):
                context_sig = self.activation_engine.compute_context_signature(s_t, top_k)
            else:
                context_sig = s_t

            self.activation_engine.set_previous_detections(
                self.prev_D_bind, self.prev_D_rec
            )
            if hasattr(self.activation_engine, "set_time"):
                self.activation_engine.set_time(t)
            elif hasattr(self.activation_engine, "t"):
                self.activation_engine.t = t
            self.activation_engine.step(t, None, None, s_t)

            if hasattr(self.activation_engine, "compute_onsets"):
                self.activation_engine.compute_onsets(t)

            # [SPATIAL BINDERS] Topology/Degree/Contact require anchor_positions.
            # Extract from the observation's `positions` mapping — every symbolic
            # env in the tree populates this attribute. Non-spatial atoms are
            # encoded as either `None` (absent) or `(-1, -1)` (flag placeholder);
            # both are skipped. Non-numeric or malformed position values are
            # skipped rather than raising, since the pipeline is not the right
            # place to fail on an env-side data quality issue.
            #
            # Co-motion requires anchor_velocities, which symbolic envs don't
            # expose; that binder correctly skips when velocities are absent.
            anchor_positions = self._derive_anchor_positions(observation)

            # [v9] T-junction extraction for the OCCLUSION binder. Symbolic
            # observations lack get_t_junctions(), so this stays None and the
            # binder produces no edges (the 3D depth branch is separately
            # gated on 3-component positions).
            t_junctions = None
            if hasattr(observation, "get_t_junctions"):
                t_junctions = observation.get_t_junctions()

            extraction_scale = self._derive_extraction_scale(observation)
            modality = observation_modality(observation)
            # Retained for diagnostics: the higher-lens probe reports this,
            # and it is the single number that decides whether every
            # higher-tier spatial gate is satisfiable.
            self._last_extraction_scale = extraction_scale
            self._last_modality = modality

            # B.4 surround suppression. Runs before D_base so the
            # suppressed set is what recognition, operator emission, the
            # binders and the Builder all see — suppressing only at the
            # binder would leave the higher tier reading a different
            # detection set from the one V1 was bound on.
            D_v1_suppressed: FrozenSet[Detection] = frozenset(D_raw)
            if P.HIGHER_LENS_ENABLED:
                D_v1_suppressed = surround_suppress(
                    frozenset(D_raw),
                    anchor_positions,
                    modality,
                    extraction_scale,
                )

            # Phase 4: Form the base detection set.
            #
            # [v31 FLAT TAXONOMY] There is no "higher lens" partition. All
            # lens output is D_raw. Derived lenses (recognition, operator,
            # chunk) contribute to D_op, D_rec, D_chunk below, not to D_base.
            # Concept-derived lenses from v30 are gone.
            #
            # [PHASE B] Step 4: with the higher tier enabled, D_base is the
            # SUPPRESSED set. D_raw keeps its pre-suppression meaning for
            # diagnostics and for the flag-off rollback path.
            D_base = set(D_v1_suppressed)

            if self.optimize_recognize_cache:
                recognized_raw = self.derived_lenses.recognize_raw(
                    list(D_v1_suppressed),
                    self.concept_graph.concepts,
                    t,
                    min_similarity=0.9,
                )
                D_base_for_binding = D_base - recognized_raw
            else:
                D_base_for_binding = D_base

            D_rec = self.derived_lenses.step_13_recognition(
                D_raw=list(D_v1_suppressed),
                concepts=self.concept_graph.concepts,
                t=t,
            )

            ef_eff_proposal = {
                cid: getattr(c, "EF_eff", 0.0)
                for cid, c in self.concept_graph.concepts.items()
            }
            rf = {
                cid: getattr(c, "RF", 0.0) for cid, c in self.concept_graph.concepts.items()
            }
            D_op = self.derived_lenses.step_14_operator_emission(
                D_base=list(D_base_for_binding),
                concepts=self.concept_graph.concepts,
                edges=self.concept_graph.edges,
                active_concepts=self.concept_graph.get_active_concepts(),
                t=t,
                ef_eff_proposal=ef_eff_proposal,
                rf=rf,
            )

            chunk_candidates = self.derived_lenses.step_15_chunk_candidates(
                active_plans={},
                concepts=self.concept_graph.concepts,
                edges=self.concept_graph.edges,
                coact_ema=self.concept_graph.get_coact_ema(),
                time_since_coact=self.concept_graph.get_time_since_coact(),
                active_concepts=self.concept_graph.get_active_concepts(),
                t=t,
            )
            D_chunk = self.derived_lenses.step_16_chunk_emission(
                candidates=chunk_candidates,
                concepts=self.concept_graph.concepts,
                edges=self.concept_graph.edges,
                coact_ema=self.concept_graph.get_coact_ema(),
                t=t,
                context_sig=context_sig,
                active_plans={},
            )

            # §4.2 step 17: D_bind = D_base ∪ D_op — complete binding
            # evidence. (Attention gating is read-only and applies to proposal
            # generation, never to the binding input; see constructor note.)
            D_bind = D_base_for_binding.union(set(D_op))

            # v31 Edge-emission pipeline: self-binders → binder → builder.
            # [PHASE B] Step 4: D_bind is already built from the SUPPRESSED
            # base (see Phase 4 above), so there is no second suppression
            # pass here — suppressing again at this point would also hit
            # the D_op detections, which is not what B.4 asks for.
            if self.self_binders is not None:
                self._run_phase(
                    "binders.self", self.self_binders.step, t, frozenset(D_bind),
                )

            # [Stage 4] Extract shock skeleton joints for SKELETAL_JOINT edges.
            shock_struct = (
                observation.get_shock_structure()
                if hasattr(observation, "get_shock_structure")
                else None
            )
            shock_joints = []
            if shock_struct:
                _segs, _joints, _ens, _meta = shock_struct
                shock_joints = [
                    (j["trunk"], j["limb"], j["signal"], j["angle"], j["ratio"], j["type"], j["attach"])
                    for j in _joints
                ]

            if self.self_binders is not None:
                self._run_phase(
                    "binders.self", self.self_binders.step, t, frozenset(D_bind),
                )
            binder_sections = self._run_phase(
                "binders.cross",
                self.binder.step,
                t,
                frozenset(D_bind),
                anchor_positions=anchor_positions if anchor_positions else None,
                anchor_velocities=None,
                t_junctions=t_junctions,
                extraction_scale=extraction_scale,
                structural_anchors=getattr(observation, "get_structural_anchors", lambda: None)(),
                surface_anchors=getattr(observation, "get_surface_anchors", lambda: None)(),
                contour_surface_contacts=getattr(observation, "get_contour_surface_contacts", lambda: None)(),
                shock_joints=shock_joints,
            )
            # Phase B higher-lens detections. D_v4 / D_part / D_ait stay empty
            # until Steps 6-8 implement their lenses; D_bo and D_v2 are
            # populated inside the flag block.
            D_bo: List[Detection] = []
            D_contour: List[Detection] = []
            D_junction: List[Detection] = []
            D_v2: List[Detection] = []
            D_v4: List[Detection] = []
            D_part: List[Detection] = []
            D_ait: List[Detection] = []
            D_shock: List[Detection] = []

            if P.HIGHER_LENS_ENABLED:
                current_detections_by_k: Dict[str, List[Detection]] = (
                    defaultdict(list)
                )
                for d in sorted(D_bind, key=_detection_sort_key):
                    current_detections_by_k[d.k].append(d)

                cg = self.concept_graph
                sig_provider = getattr(
                    self.sensory_floor, "signature_provider", None
                )

                D_bo = sorted(
                    self.border_ownership_lens.step(
                        t, cg, sig_provider, anchor_positions, None,
                        dict(current_detections_by_k), extraction_scale,
                        observation=observation,
                    ),
                    key=_detection_sort_key,
                )
                self.binder.bind_intra_step(t, frozenset(D_bo))

                # ---- V2: Surface (runs alongside BorderOwnership) ----
                D_surface = sorted(
                    self.surface_lens.step(
                        t, cg, sig_provider, anchor_positions, None,
                        {
                            "Boundary": current_detections_by_k.get("Boundary", []),
                            "BorderOwnership": D_bo,
                            "Color": current_detections_by_k.get("Color", []),
                            "Texture": current_detections_by_k.get("Texture", []),
                        },
                        extraction_scale=extraction_scale,
                    ),
                    key=_detection_sort_key,
                )
                self.binder.bind_intra_step(t, frozenset(D_surface))

                # A lens that is enabled, wired and reachable but has
                # emitted nothing is the exact failure this tier is prone
                # to (Step 4 shipped one: absolute radii no anchor pair
                # could satisfy). Log it once, loudly, rather than letting
                # a silently-inert lens pass the gate.
                #
                # Gated on modality == "image": B.1 scopes the higher tier
                # to image modality, so a symbolic observation emitting
                # nothing is CORRECT rather than a fault. Without this gate
                # every symbolic run (the main test environment) would
                # raise a false alarm.
                bo_diag = self.border_ownership_lens.get_diagnostics()
                if (
                    bo_diag["emitted_total"] == 0
                    and modality == "image"
                    and self.border_ownership_lens._candidates_evaluated > 0
                    and not self._border_ownership_silent_logged
                ):
                    self._border_ownership_silent_logged = True
                    logger.error(
                        "Phase B: BorderOwnershipLens emitted 0 detections "
                        "(extraction_scale=%r, modality=%r, anchors=%d, "
                        "Boundary=%d, Color=%d, Texture=%d). The V2 "
                        "figure-ground tier is inert.",
                        extraction_scale, modality, len(anchor_positions),
                        len(current_detections_by_k.get("Boundary", [])),
                        len(current_detections_by_k.get("Color", [])),
                        len(current_detections_by_k.get("Texture", [])),
                    )

                # ---- V2: Contour + Junction (Step 5) ----
                # These read the V1 edge graph written by binder.step above.
                D_contour = sorted(
                    self.contour_lens.step(
                        t, self.concept_graph,
                        self.providers.signature_provider,
                        anchor_positions, None,
                        {
                            "Orientation":
                                current_detections_by_k.get("Orientation", []),
                            "BorderOwnership": D_bo,
                        },
                        extraction_scale=extraction_scale,
                    ),
                    key=_detection_sort_key,
                )
                D_junction = sorted(
                    self.junction_lens.step(
                        t, self.concept_graph,
                        self.providers.signature_provider,
                        anchor_positions, None,
                        {
                            "Orientation":
                                current_detections_by_k.get("Orientation", []),
                        },
                        extraction_scale=extraction_scale,
                    ),
                    key=_detection_sort_key,
                )
                D_v2 = list(D_contour) + list(D_junction)
                # Register BEFORE emitting: the v2 emitters read the node
                # ids that bind_intra_step just minted.
                # V1 peaks own their anchors; V2 chains reference those
                # anchors and both are registered before the single v2 pass.
                D_peak = sorted(
                    self.peak_lens.step(
                        t, self.concept_graph,
                        self.providers.signature_provider,
                        anchor_positions, None, {},
                        extraction_scale=extraction_scale,
                        observation=obs,
                    ),
                    key=_detection_sort_key,
                )
                D_chain = sorted(
                    self.chain_lens.step(
                        t, self.concept_graph,
                        self.providers.signature_provider,
                        anchor_positions, None,
                        {
                            "Orientation":
                                current_detections_by_k.get("Orientation", []),
                        },
                        extraction_scale=extraction_scale,
                        observation=obs,
                    ),
                    key=_detection_sort_key,
                )
                D_v2 = D_v2 + list(D_peak) + list(D_chain)
                self.binder.bind_intra_step(t, frozenset(D_v2))
                self.binder.emit_pending_edges("v2")

                # ---- V2: Illusory Contours (Modal Completion) ----
                D_illusory = sorted(
                    self.illusory_contour_lens.step(
                        t, self.concept_graph,
                        self.providers.signature_provider,
                        anchor_positions, None,
                        {
                            "EndStopped": current_detections_by_k.get("EndStopped", []),
                            "Termination": current_detections_by_k.get("Termination", []),
                        },
                        extraction_scale=extraction_scale,
                    ),
                    key=_detection_sort_key,
                )
                for d in D_illusory:
                    inds = [r for r in d.rho if r[0] in ("inducer_1", "inducer_2")]
                    pts = [anchor_positions.get(r[1]) for r in inds]
                    pts = [p for p in pts if p]
                    if len(pts) == 2:
                        anchor_positions[d.alpha] = (
                            (pts[0][0] + pts[1][0]) / 2.0,
                            (pts[0][1] + pts[1][1]) / 2.0,
                            0.0,
                        )
                self.binder.bind_intra_step(t, frozenset(D_illusory))
                D_v2 = (list(D_contour) + list(D_junction) + list(D_peak)
                        + list(D_chain) + list(D_illusory))

                # ---- V4: CurveComplex (Step 6) ----
                D_v4 = sorted(
                    self.curve_complex_lens.step(
                        t, self.concept_graph,
                        self.providers.signature_provider,
                        anchor_positions, None,
                        {"Contour": D_contour, "Junction": D_junction},
                        extraction_scale=extraction_scale,
                    ),
                    key=_detection_sort_key,
                )
                # Register BEFORE emitting, as at v2: the containment
                # emitter reads the node ids bind_intra_step just minted.
                self.binder.bind_intra_step(t, frozenset(D_v4))
                self.binder.emit_pending_edges("v4")

                # ---- AIT: HolisticShape (Step 6) ----
                D_ait = sorted(
                    self.holistic_shape_lens.step(
                        t, self.concept_graph,
                        self.providers.signature_provider,
                        anchor_positions, None,
                        {},  # no additional detections needed
                        extraction_scale=extraction_scale,
                        observation=observation,
                    ),
                    key=_detection_sort_key,
                )
                self.binder.bind_intra_step(t, frozenset(D_ait))
                # No pending edges for AIT; it is a terminal shape descriptor.

                # [Stage 4] Shock Ensemble Lens (medial axis ribbons/ensembles).
                D_shock = sorted(
                    self.shock_ensemble_lens.step(
                        t, self.concept_graph,
                        self.providers.signature_provider,
                        anchor_positions, None,
                        {},
                        extraction_scale=extraction_scale,
                        observation=observation,
                    ),
                    key=_detection_sort_key,
                )
                self.binder.bind_intra_step(t, frozenset(D_shock))

                # [Phase B B.14] Recurrent contour grouping: an anchor that
                # keeps appearing as a Contour member accumulates gain, which
                # widens its Orientation response next step. Decay is applied
                # AFTER accumulation so a fresh contour is not decayed away in
                # the same step it is earned. Gain is therefore inherently
                # ONE STEP LAGGED: a contour seen at step t first influences
                # Orientation at t+1.
                if P.RECURRENT_GROUPING_ENABLED:
                    for det in D_contour:
                        for role in det.rho:
                            if role[0] == "member":
                                self.contour_gain_state.bounded_increment(
                                    role[1]
                                )
                    self.contour_gain_state.decay()

                for _lens, _flag in (
                    (self.contour_lens, "_contour_silent_logged"),
                    (self.junction_lens, "_junction_silent_logged"),
                    (self.chain_lens, "_chain_silent_logged"),  # [A5]
                    (self.curve_complex_lens, "_curve_complex_silent_logged"),
                    (self.holistic_shape_lens, "_holistic_shape_silent_logged"),
                    (self.shock_ensemble_lens, "_shock_ensemble_silent_logged"),
                ):
                    if (
                        extraction_scale > 0.0
                        and _lens._emitted_total == 0
                        and _lens._candidates_evaluated > 0
                        and not getattr(self, _flag)
                    ):
                        setattr(self, _flag, True)
                        logger.error(
                            "Phase B: %s emitted 0 detections despite "
                            "candidates_evaluated=%d (extraction_scale=%r, "
                            "modality=%r, Orientation=%d). Candidates were "
                            "found and discarded — the tier is inert.",
                            type(_lens).__name__,
                            _lens._candidates_evaluated,
                            extraction_scale, modality,
                            len(current_detections_by_k.get("Orientation", [])),
                        )

                # Steps 6-8 add CurveComplex and ObjectShape. Their
                # lenses still raise NotImplementedError, so they are
                # deliberately NOT called here; D_v4 / D_part / D_ait stay
                # empty and their edge channels stay at zero until each
                # step lands.
                # B.14: the gain state is constructed in Step 4 and wired
                # in Step 9, when recurrent contour grouping is live.

                self._higher_lens_diagnostics = {
                    "BorderOwnership": bo_diag,
                    "Contour": self.contour_lens.get_diagnostics(),
                    "Junction": self.junction_lens.get_diagnostics(),
                    "CurveComplex":
                        self.curve_complex_lens.get_diagnostics(),
                    "HolisticShape":
                        self.holistic_shape_lens.get_diagnostics(),
                    "ShockEnsemble":
                        self.shock_ensemble_lens.get_diagnostics(),
                }
                # [PHASE B] Steps 5-6. candidates_evaluated is what separates
                # "nothing to find" (0) from "found candidates, discarded
                # them all" (>0 with emitted 0) — the latter is a bug.
                self._higher_tier_candidates = {
                    "BorderOwnership":
                        self.border_ownership_lens._candidates_evaluated,
                    "Contour": self.contour_lens._candidates_evaluated,
                    "Junction": self.junction_lens._candidates_evaluated,
                    "CurveComplex":
                        self.curve_complex_lens._candidates_evaluated,
                    "HolisticShape":
                        self.holistic_shape_lens._candidates_evaluated,
                    "ShockEnsemble":
                        self.shock_ensemble_lens._candidates_evaluated,
                }
                self._higher_tier_emissions = {
                    "BorderOwnership":
                        self.border_ownership_lens._emitted_total,
                    "Contour": self.contour_lens._emitted_total,
                    "Junction": self.junction_lens._emitted_total,
                    "CurveComplex":
                        self.curve_complex_lens._emitted_total,
                    "HolisticShape":
                        self.holistic_shape_lens._emitted_total,
                    "ShockEnsemble":
                        self.shock_ensemble_lens._emitted_total,
                }

            # Union order is fixed (V1, then v2, v4, lo/pIT, aIT) so the
            # resulting set is deterministic run to run.
            D_bind_all = (
                D_bind
                .union(D_bo)
                .union(D_surface)
                .union(D_v2)
                .union(D_v4)
                .union(D_ait)
                .union(D_shock)
            )

            # [B2] Situation windows represent STATES, not the arriving
            # action: the Action-lens detection present in D_bind at step t
            # records a_{t−1} (the action that LED to s_t). Including it
            # would make window identity route-dependent (the same state
            # reached by different actions would compare unequal), breaking
            # §5.3.4 replay matching and B3's content dedup. Action nodes
            # still enter the Builder's candidate stream (D_bind_all below is
            # unfiltered) — only the state index sees the filtered set.
            # D_bind_all includes higher-lens features once Step 3 lands.
            self.providers.state_index.record(
                s_t, {d for d in D_bind_all if d.k != ACTION_LENS_NAME}
            )

            # [Stage 5] Aspect graph: record this view with its dataset
            # metadata. The descriptor is now built from the live field's
            # PEAK_CHAINs rather than from shock ribbons: the segmenter
            # region path is empty, so shock structures are unreachable and
            # this would otherwise never fire. Falls back to the shock
            # descriptor when a view somehow has no chains.
            if P.ASPECT_ENABLED:
                _vm = getattr(observation, "view_meta", None)
                if _vm is not None:
                    _desc = build_view_descriptor_from_chains(
                        chain, anchor_positions, extraction_scale)
                    if _desc is None and hasattr(
                            observation, "get_shock_structure"):
                        _desc = build_view_descriptor(
                            observation.get_shock_structure())
                    if _desc is not None:
                        self.aspect_graph.observe(
                            t,
                            _vm["instance"],
                            _vm.get("label"),
                            _vm["az"],
                            _vm["el"],
                            _desc,
                        )

            builder_output = self._run_phase(
                "builder", self.builder.step, t, D_bind_all, s_t
            )

            # [SPEC COMPLIANCE §3.2] "Its proposed cross-sections are unioned
            # with the Binder's output and processed together by
            # step_21_update_candidate_stats." The former wiring passed ONLY
            # builder_output.cross_sections — the Binder's valid cross-sections
            # (the primary crystallization engine, §4.20) were silently
            # discarded, starving the concept graph of every binder-derived
            # candidate. Deduplication by (family, cross_section_key) is
            # step_21's job (§4.20); the union here must not pre-deduplicate
            # by content since both sources compute identical keys.
            if binder_sections is None:
                # v31 §2.9: the Binder is the edge-emitting observation engine
                # — step() writes EdgeType-tagged edges into the ConceptGraph
                # and returns None BY CONTRACT. CrossSections originate from
                # the Builder (the sole proposer, §4.1/§4.20); the union below
                # stays so any future binder-emitted sections are consumed per
                # §3.2 without rewiring.
                binder_sections = []
            valid_cross_sections = list(binder_sections) + list(
                builder_output.cross_sections
            )

            # §5.8.1 action-mode consolidation. The Builder emits resolved
            # §4.15.3 (P, S) trace pairs in two forms:
            #   * handoffs (B1/D6) — induced subgraphs + the state hash at
            #     P's crystallization, the faithful §5.3.4 replay start; and
            #   * raw (P_hash, S_hash) pairs — kept for diagnostics and as
            #     the reconstruction path for any resolution the Builder
            #     could not package (concept left the graph between resolve
            #     and packaging).
            # A handoff SUPERSEDES its raw pair (same event — double
            # consumption would double-spend the replay budget).
            if getattr(builder_output, "handoffs", None):
                self._pending_handoffs.extend(builder_output.handoffs)
            if builder_output.path_outcome_pairs:
                self._pending_consolidation_pairs.extend(
                    builder_output.path_outcome_pairs
                )
            covered = self._consume_consolidation_handoffs(t, s_t)
            self._consume_consolidation_pairs(t, s_t, skip=covered)

            # Phase 5: Candidates and graph updates
            if self.learning_enabled:
                self.concept_graph.step_21_update_chunk_stats(set(D_chunk), t)
                self._run_phase(
                    "graph.candidates",
                    self.concept_graph.step_21_update_candidate_stats,
                    valid_cross_sections,
                    t,
                    prev_action=self.prev_action,
                    a_hist=self.a_hist,
                    onset_history=self.onset_history,
                )
                self.concept_graph.step_23_candidate_forgetting_and_capacity()
                self.concept_graph.step_24_tombstone_expiry()

                # (set_current_state_hash already ran in Phase 1 — the second
                # call was a redundant re-entrant write.)
                newly_crystallized = self._run_phase(
                    "graph.crystallization",
                    self.concept_graph.step_25_evaluate_crystallization,
                    t=t,
                    activation_engine=self.activation_engine,
                    active_plans=[],
                    s_t=s_t,
                    action_t=self.prev_action,
                )
                self._run_phase(
                    "graph.edges",
                    self.concept_graph.step_28_update_edges,
                    self.activation_engine,
                    t,
                )
                self._run_phase(
                    "graph.witness",
                    self.concept_graph.step_28_update_witness_and_status,
                    t,
                )

                if hasattr(self.derived_lenses, "update_recognition_drift"):
                    self.derived_lenses.update_recognition_drift(
                        self.concept_graph.concepts
                    )

                # Commit any bindings whose necessity validated since their
                # onset-step registration (event-time attribution below), so
                # freshly crystallized concepts carry consequence packaging.
                self._commit_pending_bindings(t, observation)

                # TRIGGER 1A (ruling): single-concept bootstrap consolidation.
                # A freshly crystallized causal schema is immediately packaged
                # (path = its organization, outcome = its effect pattern,
                # replay start = its enabling state) and offered to
                # prune_path — populating the CoreLibrary on the FIRST door
                # opening instead of waiting for §4.15 (P,S) pair episodes
                # that sparse environments may never repeat three times.
                self._trigger_bootstrap_consolidation(t, newly_crystallized)

                # TRIGGER 1C + 2 (ruling): periodic consolidation sweep at the
                # §5.9 K_consolidate cadence — retries bootstrap cores whose
                # transitions were not yet confirmed (N_edge=5) at
                # crystallization time, and extracts multi-anchor category
                # cores (Trigger 2A). Replaces the former run_periodic
                # hasattr-probe, which the Consolidator never exposed.
                if t > 0 and t % P.K_CONSOLIDATE == 0:
                    self._consolidation_sweep(t)
            else:
                newly_crystallized = []

            # Phase 6: Fertility
            self._run_phase(
                "fertility", self.fertility_engine.step30_recompute_RF_GP_EF, t
            )

            if self.learning_enabled:
                # v31 plan-outcome events: validated transfers (§8.5) replace
                # the deleted PlanningEngine's plan_outcomes as the source of
                # plan successes/failures. They are marshaled into the EXISTING
                # step31/step33 plan-event interface ({"basis", "trace"} dicts):
                # the basis is the transfer's realization basis (live concepts
                # overlapping the consequence — utility_engine.transfer_basis,
                # the structural successor of basisSnapshot), and the
                # eligibility trace is uniform over the basis (no plan-stack
                # machinery survives in v31 to weight it further). The former
                # hardwired empty lists silently disabled plan-success
                # contextual fertility (§2.15.3) and step-blame credit (§2.10.2)
                # — the "orphaned planner dependencies" defect.
                plan_successes: List[Dict] = []
                plan_failures: List[Dict] = []
                # [Seam 1] Macro-execution outcomes close the orphaned-
                # planner-listener defect WITHOUT inventing a parallel credit
                # path: the UtilityEngine records how much of each transferred
                # ActionSequence was actually executed; outcomes are BUFFERED
                # here until Prediction's §8.5 validation for the same core
                # arrives (horizon W_VALIDATION later — a same-step merge
                # would almost never coincide), then GATE and SCALE the
                # credit event (K4: partial executions credit proportionally;
                # unexecuted validations — consequence appeared by other
                # means — credit nothing). FertilityEngine's step31/step33
                # interface is untouched (constitutional file).
                for oc in self.utility_engine.drain_macro_outcomes():
                    bucket = self._macro_outcomes_by_core.setdefault(
                        oc.core_hash, []
                    )
                    bucket.append(oc)
                    if len(bucket) > 32:  # bounded per core
                        del bucket[: len(bucket) - 32]
                retention_floor = t - 2 * P.W_VALIDATION
                for ch in list(self._macro_outcomes_by_core.keys()):
                    kept = [
                        oc
                        for oc in self._macro_outcomes_by_core[ch]
                        if oc.t >= retention_floor
                    ]
                    if kept:
                        self._macro_outcomes_by_core[ch] = kept
                    else:
                        del self._macro_outcomes_by_core[ch]
                for core_hash, observed, t_val, t_pred in self._validated_transfer_outcomes:
                    basis = self.utility_engine.transfer_basis(core_hash)
                    if not basis:
                        self.consolidation_diagnostics[
                            "transfer_outcome_basis_empty"
                        ] += 1
                        continue
                    # Match the LATEST execution of this core that precedes
                    # the validation (one consumption per validation).
                    bucket = self._macro_outcomes_by_core.get(core_hash)
                    oc = None
                    if bucket:
                        eligible = [o for o in bucket if o.t <= t]
                        if eligible:
                            oc = max(eligible, key=lambda o: o.t)
                            bucket.remove(oc)
                    completion = oc.completion_ratio if oc is not None else 0.0

                    # Pop the selection record for this validation window.
                    sel_t = self.utility_engine.pop_selection_record(
                        core_hash, t_pred
                    )
                    if sel_t is None:
                        self.consolidation_diagnostics[
                            "transfer_selection_pop_misses"
                        ] += 1

                    # Synthesize credit only when no macro was in flight.
                    if oc is None and observed:
                        if sel_t is not None and t_pred <= sel_t <= t_val:
                            trace = {cid: 1.0 / len(basis) for cid in basis}
                            event = {
                                "basis": basis,
                                "trace": trace,
                                "core_hash": core_hash,
                            }
                            plan_successes.append(event)
                            self.consolidation_diagnostics[
                                "transfer_credit_synthesized"
                            ] += 1
                            self._log_p1_3_triggers(event, t)
                            continue

                    if observed and completion <= 0.0:
                        # Consequence materialized without the macro being
                        # executed — correlation, not plan success. Credit
                        # would reinforce concepts that did nothing.
                        self.consolidation_diagnostics[
                            "transfer_validated_not_executed"
                        ] += 1
                        continue
                    if not observed and completion <= 0.0:
                        # Neither executed nor observed: no evidence about
                        # the schema either way — no blame event.
                        self.consolidation_diagnostics[
                            "transfer_unvalidated_not_executed"
                        ] += 1
                        continue
                    # Execution-gated event: trace weight scaled by the
                    # executed fraction of the plan (full completion → the
                    # former uniform trace; interruption → proportional).
                    trace = {cid: completion / len(basis) for cid in basis}
                    event = {"basis": basis, "trace": trace, "core_hash": core_hash}
                    if observed:
                        plan_successes.append(event)
                        self.consolidation_diagnostics[
                            "macro_credit_success"
                        ] += 1
                        # [P1-3 TRIGGER LOGGING / Bundle A] for each precondition
                        # concept of a succeeded plan whose RF sits below the
                        # fertile threshold, log the trigger tuple to a rolling
                        # file (cap 10,000 entries). Consumed only when the
                        # P1-3 design pass triggers; pure observation.
                        self._log_p1_3_triggers(event, t)
                    else:
                        plan_failures.append(event)
                        self.consolidation_diagnostics["macro_credit_failure"] += 1
                # [Batch D / P1-3] Credit propagation to preconditions: walk
                # backward along SEQUENCE/TRANSITION edges from each succeeded
                # plan's basis; enabling concepts earn discounted credit.
                if plan_successes:
                    for event in plan_successes:
                        basis = event.get("basis", [])
                        if not basis:
                            continue
                        precond = self._compute_precondition_credit(set(basis))
                        for pid, cred in precond.items():
                            concept = self.concept_graph.concepts.get(pid)
                            if concept is not None:
                                concept.credit = min(
                                    P.C_MAX,
                                    concept.credit + P.C_ENABLEMENT * cred,
                                )
                        self.consolidation_diagnostics[
                            "precondition_credits_assigned"
                        ] += len(precond)

                self._validated_transfer_outcomes = []
                self.fertility_engine.step31_update_contextual_fertility(
                    t=t,
                    context_signature=context_sig,
                    plan_successes=plan_successes,
                    plan_failures=plan_failures,
                    newly_crystallized=newly_crystallized,
                )
                self.fertility_engine.step33_credit_updates(
                    t=t,
                    plan_successes=plan_successes,
                    plan_failures=plan_failures,
                )

                # [Batch D / P1-4 HIGH-01] Partial-failure attribution: the
                # blame loop iterates plan_failures HERE (in scope, K4-gated),
                # not the cleared _validated_transfer_outcomes.
                for event in plan_failures:
                    core_hash = event.get("core_hash")
                    if core_hash is None:
                        continue
                    chain = self._reconstruct_chain(core_hash)
                    if not chain:
                        continue
                    break_idx = self._find_break_idx(chain, t)
                    for i, cid in enumerate(chain):
                        blame = (
                            P.GAMMA_BLAME ** (break_idx - i)
                            if i < break_idx
                            else 1.0
                            if i == break_idx
                            else 0.0
                        )
                        concept = self.concept_graph.concepts.get(cid)
                        if concept is not None and blame > 0.5:
                            concept.plan_blame_streak = min(
                                P.TAU_REVERT, concept.plan_blame_streak + 1
                            )
                    self.consolidation_diagnostics["blame_events"] += 1

                # [Batch D / P1-4] Route plan successes into the structural-
                # progress channel (resets fail/blame streaks, step_34).
                if plan_successes:
                    plan_progress_events = []
                    for event in plan_successes:
                        for cid in event.get("basis", []):
                            plan_progress_events.append((cid, "plan_success"))
                    if plan_progress_events:
                        self.concept_graph.step_34_update_structural_progress(
                            plan_progress_events, t
                        )

                if newly_crystallized:
                    progress_events = []
                    for concept in newly_crystallized:
                        for p_id in concept.parent_ids:
                            progress_events.append((p_id, "child_crystallization"))
                    self.concept_graph.step_34_update_structural_progress(
                        progress_events, t
                    )


                # Phase 7: Pruning/capacity/homeostasis
                self.concept_graph.step_35_dormancy_sweep(
                    self.activation_engine, self.fertility_engine, planning_engine=None
                )
                self.concept_graph.step_36_capacity_eviction(
                    self.fertility_engine, planning_engine=None
                )
                self.concept_graph.step_pruning(
                    self.activation_engine, self.fertility_engine, t
                )
                if hasattr(self.concept_graph, "step_38_homeostasis"):
                    self.concept_graph.step_38_homeostasis(t)

                self.fertility_engine.step39_decay_contextual_fertility()
                self.fertility_engine.step39b_update_ebaf(t)

                ef_eff_proposal = {
                    cid: getattr(c, "EF_eff", 0.0)
                    for cid, c in self.concept_graph.concepts.items()
                }
                rf = {
                    cid: getattr(c, "RF", 0.0)
                    for cid, c in self.concept_graph.concepts.items()
                }
                promotion_candidates = self.derived_lenses.compute_promotion_candidates(
                    self.concept_graph.concepts, t, ef_eff_proposal, rf
                )
                self.concept_graph.step_40_update_operator_promotion(
                    promotion_candidates, t
                )

            # Operator demotion runs unconditionally (cleanup path).
            operator_emissions = self.derived_lenses.get_operator_emissions(t)
            self.concept_graph.step_40_update_operator_demotion(operator_emissions, t)
            self.fertility_engine.step40b_compute_ebaf(t, context_sig)

            # v30 §2.32.8 test-only credit injection (flag lives in
            # parameters.py). When the test mode is ENABLED, the hook is a
            # hard contract — a missing method fails loudly instead of
            # silently skipping the injection the run was configured for.
            if EBAF_TEST_MODE and t == T_INJECT:
                self.fertility_engine.credit_injection_test()

            # ---------------------------------------------------------------------
            # [Batch D2] §6.6.2 CATEGORY-QUERY CALLER
            # ---------------------------------------------------------------------
            # "What is this?" asked once per step over the step's own situation
            # subgraph. This is the consumer CategoryMatch / CATEGORY cores
            # never had: the Mapper could extract and store category cores but
            # no code ever queried them, so their reliability was never
            # exercised (category_labels_returned could only be 0).
            #
            # The situation subgraph is memoized for this t by the earlier
            # consumers (UtilityEngine step_41 via the situation provider), and
            # Mapper._begin_step is per-t idempotent, so this costs one lookup -
            # not one rebuild. Cost is bounded by K_CATEGORY_QUERY_MAX
            # identification passes; labels are ordered by serialize_value
            # (determinism, §9).
            if (
                self.learning_enabled
                and self.mapper is not None
                and self.category_manager is not None
            ):
                self._category_match_this_step = {}
                situation_for_cat = self.mapper.build_situation_subgraph(t)
                labels_sorted = sorted(
                    self._derive_label_signals(t),
                    key=lambda s: serialize_value(s.label),
                )[:P.K_CATEGORY_QUERY_MAX]
                labels_queried = 0
                labels_recognized = 0
                for sig in labels_sorted:
                    rec = self.category_manager.recognize(
                        sig.label, sig.polarity, situation_for_cat, t,
                    )
                    labels_queried += 1
                    if rec.assigned_hash is not None:
                        labels_recognized += 1
                        prev_score = self._category_match_this_step.get(
                            rec.label, 0.0
                        )
                        if rec.top_candidate_score > prev_score:
                            self._category_match_this_step[rec.label] = (
                                rec.top_candidate_score
                            )
                    self.category_manager.observe(
                        sig.label, sig.polarity, situation_for_cat,
                        rec.assigned_hash, rec.top_candidate_hash,
                        rec.top_candidate_score, t,
                    )
                self.consolidation_diagnostics[
                    "category_queries_issued"
                ] += labels_queried
                self.consolidation_diagnostics[
                    "category_queries_recognized"
                ] += labels_recognized
                # §6.6/§6.7 GapSignal channel: drain the Mapper's bounded queue
                # so it cannot sit permanently at its GAP_QUEUE_MAX cap. In v31
                # the queue is a diagnostic channel (no exploration consumer
                # exists), so the drain counts rather than routes.
                for _ in self.mapper.drain_gaps():
                    self.consolidation_diagnostics["gaps_drained"] += 1

            # Phase 8: Action selection
            self.utility_engine.set_time(t, s_t)
            proposals = self._run_phase(
                "utility.proposals",
                self.utility_engine.step_41_generate_proposals,
                s_t,
                observation,
            )
            utilities = self._run_phase(
                "utility.utilities",
                self.utility_engine.step_42_compute_utilities,
                proposals,
                s_t,
            )
            selected_action = self._run_phase(
                "utility.selection", self.utility_engine.step_43_select_action, utilities, s_t
            )

            # Phase 9: Execution (§4.2 step 44). One observation per step —
            # the former post-execution env.observe() + private state-hash
            # recomputation existed only to feed a `state_changed` flag for
            # the utility engine's off-spec "spatial refractory isolation".
            # Worse, on 3D environments observe() ADVANCES the adapter frame
            # (step_count, Schmitt state, LK history), so the double-observe
            # ran the sensory floor at 2× the cognitive clock: velocities
            # halved, τ_persist windows halved. Environments refresh their
            # effect-marker state inside execute_action (pybullet_env does).
            action_result = self._run_phase(
                "execution", self.env.execute_action, selected_action
            )

            self.env.record_action(selected_action, action_result)

            # [B2] Execution-time Action-pattern onset pre-registration. The
            # Action lens will emit the identical pattern at step t+1 (when
            # this action arrives as a_prev), but its ONSET belongs to NOW —
            # the step the action was taken. Pinning the onset here makes the
            # SEQUENCE binder attribute the effect patterns onsetting at t+1
            # to THIS action (lag = 1), instead of misattributing them to the
            # previous action (co-onset, lag 0, no edge). Also registers the
            # exact action object in the ConceptGraph's action-alpha registry
            # — the §5.3.4 vocabulary-recovery source (K1: the ORIGINAL
            # object, never a re-serialization).
            if self.learning_enabled and is_action_lens_eligible(selected_action):
                # [Batch D / P1-6] Runtime vocabulary extension: a novel
                # eligible action crystallizes a new ATOMIC_ACTION concept.
                if selected_action not in self._known_actions:
                    self._known_actions.add(selected_action)
                    self.concept_graph.seed_atomic_actions([selected_action])
                    self.consolidation_diagnostics["atomic_actions_added"] += 1
                action_det = self.sensory_floor.make_action_detection(
                    selected_action, t
                )
                if action_det is not None:
                    action_nid = self.binder.note_executed_action(t, action_det)
                    self.concept_graph.register_action_alpha(
                        action_nid, selected_action
                    )

            if isinstance(action_result, dict):
                success = action_result.get("success", False)
            else:
                success = bool(action_result)
            self.utility_engine.update_state_after_action(
                executed_action=selected_action,
                success=success,
                new_anchor_spawned=new_anchor_spawned,
            )

            # §2.19 structural potential: per-step decay/cleanup pass over the
            # potential map (moved here from main.py — orchestration belongs to
            # the Pipeline, and a runner that forgets the call silently freezes
            # potential decay). v31 has no plans, so the target set is empty.
            if self.structural_potential_map is not None:
                self.structural_potential_map.step_update(s_t, [])

            # Post-execution: onset history + prediction causal update.
            # Single source of truth for w_confirm: parameters.py (§8.8). The
            # pruning window is DERIVED: step_causal resolves trials at
            # j = t − w_confirm reading onsets in [j+1, j+w_confirm], so the
            # oldest onset ever read at step t is t − 2·w_confirm + 1
            # (the former `− 10` was a magic margin).
            w_confirm = P.W_CONFIRM
            prune_before = self.t - 2 * w_confirm
            for b in list(self.onset_history.keys()):
                for k in list(self.onset_history[b].keys()):
                    if k < prune_before:
                        del self.onset_history[b][k]
                if not self.onset_history[b]:
                    del self.onset_history[b]

            self.a_hist[self.t] = selected_action
            # a_hist is read only at t−1 and t−w_confirm (§8.3.1); bound it.
            a_hist_floor = self.t - 2 * w_confirm
            for old_t in [k for k in self.a_hist if k < a_hist_floor]:
                del self.a_hist[old_t]

            if hasattr(observation, "get_effect_indices"):
                current_effect_indices = set(observation.get_effect_indices() or [])
            else:
                current_effect_indices = set(
                    getattr(observation, "effect_indices", []) or []
                )
            self.all_effect_flags.update(current_effect_indices)

            presence_attr = getattr(observation, "presence", None)
            for b in sorted(self.all_effect_flags, key=serialize_value):
                if b in current_effect_indices:
                    if presence_attr is not None:
                        v_b = (
                            1
                            if (
                                presence_attr[b]
                                if isinstance(presence_attr, dict)
                                else (presence_attr[b] if b < len(presence_attr) else True)
                            )
                            else 0
                        )
                    else:
                        v_b = 1
                else:
                    v_b = 0
                # [Batch B] Publish per-atom presence for the prediction
                # module's persistence fallback. v_b is 0/1 here, so bool()
                # is a faithful coercion.
                self.effect_presence_register[b] = bool(v_b)
                v_prev = self.prev_effect_flags.get(b, 0)
                onset_b = (1 if (v_b == 1 and v_prev == 0) else 0) if self.t > 0 else 0
                self.onset_history[b][self.t] = onset_b
                self.prev_effect_flags[b] = v_b
                if (
                    onset_b == 1
                    and self.prediction is not None
                    and self.learning_enabled
                ):
                    self.prediction.notify_effect_rise(b, self.t)

            # §8.11 step 8a: streaming counterfactual, post-execution
            if self.learning_enabled and self.prediction is not None:
                configured_actions = getattr(
                    self.env, "action_rep", None
                ) or self.concept_graph.params.get("action_rep", [])
                real_actions = [
                    a
                    for a in configured_actions
                    if not is_noop_action(a)
                    and not is_probe_action(a, self.env)
                    and not is_teleological_action(a)
                ]
                # §8.11 step 8a — every step, synchronous, no exception
                # swallowing (a failing counterfactual update is a defect, not
                # a degraded mode).
                self.prediction.step_causal(
                    self.t,
                    self.a_hist,
                    self.onset_history,
                    real_actions,
                    list(self.all_effect_flags),
                )

                # [Batch D1b] Home pending voids (rises buffered earlier with
                # cid=None because no causal concept was ACTIVE to claim them)
                # to causal concepts onsetting THIS step — first-write-wins,
                # K_REATTRIB_MAX per step — before this step's registration
                # appends new candidates. Pipeline-side only; activation_engine
                # untouched (D1b design draft §2/§6).
                self._reattribute_pending_voids(self.t)
                # EVENT-TIME SCHEMA BINDING (v31 wiring): register attribution
                # candidates for atoms that ROSE this step, then commit any
                # pending triples whose necessity has validated. Runs every
                # step (registration is O(active concepts × rising atoms);
                # commitment is O(pending) with a 128-entry bound).
                rising = {
                    b
                    for b in self.all_effect_flags
                    if self.onset_history.get(b, {}).get(self.t, 0) == 1
                }
                if rising:
                    # [Batch C+ timing fix] A rising flag at step t reflects
                    # the state observed at step-start - the effect of the
                    # action executed at step t-1. Registering with
                    # `selected_action` (= a_hist[t]) paired the wrong cause
                    # and made the subsequent check_necessity call unable
                    # to validate (the effect does not rise again after t).
                    # The correct cause is a_hist[t-1], which equals
                    # prev_action at this point. None only on t=0 or after
                    # a_hist pruning; skip registration in that case.
                    cause_action = self.a_hist.get(self.t - 1)
                    if cause_action is not None:
                        self._register_event_bindings(
                            self.t, cause_action, rising, D_bind
                        )
                    else:
                        self.consolidation_diagnostics[
                            "binding_registration_skipped_no_cause"
                        ] += 1
                self._commit_pending_bindings(self.t, observation)
                # [Batch D1] Peak track + provisional expiry run every step,
                # independent of _pending_bindings state: the D1 tier can
                # commit a binding whose N_cf later decays and whose contrast
                # later turns negative, and that binding must be withdrawn
                # rather than persist as a fossil.
                self._update_binding_peaks_and_expiry(self.t)

                # [B3/§5.8.2] Label-window accumulation for the discriminative
                # consolidation trigger. Labels are derived from the
                # pipeline's OWN effect-atom tracking — no env-specific code,
                # no external label provider required.
                self._update_label_windows(self.t, s_t)

            # Habituation
            interactions = reflex.consume_interactions() if reflex is not None else set()
            crystallized_anchors = {
                c.anchor
                for c in newly_crystallized
                if getattr(c, "anchor", None) is not None
            }
            if reflex is not None:
                reflex.post_update(self.t, interactions, crystallized_anchors)

            # Reliability indicators
            if hasattr(self.concept_graph, "store_reliability_indicators"):
                presence = getattr(observation, "presence", None)
                if presence is not None and not isinstance(presence, dict):
                    entity_indices = getattr(observation, "entity_indices", [])
                    presence_map = {
                        i: bool(presence[i]) if i < len(presence) else True
                        for i in entity_indices
                    }
                else:
                    presence_map = presence or {}
                opp_map = {}
                conf_map = {}
                # Loop variable renamed from `P` — it shadowed the module-level
                # `import parameters as P`, turning every earlier P.* read in
                # step() into a latent UnboundLocalError.
                for concept in self.concept_graph.concepts.values():
                    anchor_present = True
                    if concept.anchor is not None:
                        anchor_present = bool(presence_map.get(concept.anchor, True))
                    opp_map[concept.id] = anchor_present
                    conf_map[concept.id] = getattr(concept, "A", 0) == 1
                self.concept_graph.store_reliability_indicators(self.t, opp_map, conf_map)

            # Step 46
            self.t += 1

            self.prev_state = S_t
            self.prev_action = selected_action
            self.prev_D_bind = D_bind
            self.prev_D_rec = D_rec
            self.last_state_hash = s_t

            if self.t % DIAG_INTERVAL == 0:
                self._run_diagnostics(
                    t,
                    D_raw,
                    D_rec,
                    D_op,
                    D_chunk,
                    D_bind,
                    valid_cross_sections,
                    proposals,
                    utilities,
                    selected_action,
                    s_t,
                )
                # [P0-4] size + profile snapshot piggybacked on the diag tick.
                self.diagnostics_log[-1]["sizes"] = self.get_size_snapshot()
                if self._profiling_enabled:
                    self.diagnostics_log[-1]["profile"] = self.get_profiling_report()

            return {
                "t": self.t - 1,
                "action": selected_action,
                "D_raw": D_raw,
                "D_rec": D_rec,
                "D_op": D_op,
                "D_chunk": D_chunk,
                "valid_cross_sections": valid_cross_sections,
            }
        finally:
            # Runs on the normal path AND on an exception, so a failed
            # step cannot leave the accumulators dirty for the next one.
            self.binder.finalize_step()

    # -------------------------------------------------------------------------
    # Boundary management
    # -------------------------------------------------------------------------
    def episode_reset(self):
        """§4.3 episode boundary: reset episode-local state; PRESERVE the
        graph, CF, Φ, action-edge map, tombstones, registries, and the
        Core Library (§10 memory system 16 persists across episodes).
        Builder search state persists with the graph (its frontier,
        basins, and transposition history reference graph structure)."""
        if self.innate_reflex is not None and hasattr(self.innate_reflex, "reset"):
            self.innate_reflex.reset()
        if hasattr(self.utility_engine, "reset_episode"):
            self.utility_engine.reset_episode()
        if hasattr(self.concept_graph, "reset_episode_counters"):
            self.concept_graph.reset_episode_counters()
        if self.mapper is not None:
            # Mapper's own episode-local reset (caches, label history,
            # gap queue); the core library is Consolidator-owned and
            # persists (§4.3 / mapper.reset docstring).
            self.mapper.reset()
        self.prev_state = None
        self.prev_action = None
        self.prev_D_bind = set()
        self.prev_D_rec = set()
        self.a_hist.clear()
        self.onset_history.clear()
        self.prev_effect_flags.clear()
        self.all_effect_flags.clear()
        # [Batch B] Clear the register BEFORE resetting prediction so the
        # module's reference is never stale and never observed mid-clear.
        self.effect_presence_register.clear()
        if self.prediction is not None:
            self.prediction.reset_episode_local_state()
        # [Batch 3 Phase 3] Clear CDP tracker state at the episode
        # boundary (the enable flag itself persists: tracking is a
        # run-level observation mode, not episode-local state).
        self._cdp_tracker.clear()
        self._cdp_transitions.clear()
        self.diagnostics.pop("cdp_count_current", None)
        self.e += 1

    def transfer_boundary(self):
        if hasattr(self.concept_graph, "mark_environment_bound"):
            self.concept_graph.mark_environment_bound()

    # -------------------------------------------------------------------------
    # Consolidation (§5.8.1 action-mode trigger consumption)
    # -------------------------------------------------------------------------

    def _reconstruct_subgraph(self, canonical_hash: bytes) -> Optional[Subgraph]:
        """Rebuild the Subgraph of a crystallized concept from its pattern
        nodes and the induced edges in the ConceptGraph edge store
        (§1.12 keying (parent, child, EdgeType))."""
        concept = self.concept_graph.lookup_by_canonical_hash(canonical_hash)
        if concept is None:
            return None
        members = frozenset(concept.members)
        edges = set()
        for nid in members:
            for edge in self.concept_graph.get_outgoing_edges(nid):
                if edge.child in members:
                    edges.add((edge.parent, edge.child, edge.edge_type))
        return Subgraph(
            nodes=members,
            edges=frozenset(edges),
            anchor_of_node=anchor_positions_for(
                members, self.concept_graph.pattern_node_position
            ),
        )

    def _consume_consolidation_handoffs(self, t: int, s_t: int) -> Set[Tuple[bytes, bytes]]:
        """[B1/D6] Marshal Builder handoffs into Consolidator prune_path
        calls. Returns the set of source hash pairs covered by handoffs
        (the raw-pair consumer skips them — one event, one consumption).

        Replay start (root-cause fix over the draft, which replayed from
        the SUCCESSOR's state): §5.3.4 replay must simulate the path from
        the situation in which the path actually occurred — P's
        crystallization state. Fallbacks (successor state, then current
        state) keep the K7 None-state cases honest but opportunistic;
        each start is counted so the yield per route is observable.
        """
        if not self._pending_handoffs:
            return set()
        handoffs = self._pending_handoffs
        self._pending_handoffs = []
        covered: Set[Tuple[bytes, bytes]] = set()
        if self.consolidator is None:
            self.consolidation_diagnostics[
                "handoffs_dropped_no_consolidator"
            ] += len(handoffs)
            return covered
        aem = getattr(self.env, "action_edge_map", None)
        if aem is None:
            self.consolidation_diagnostics[
                "handoffs_dropped_no_action_edge_map"
            ] += len(handoffs)
            return covered
        for h in handoffs:
            covered.add(tuple(h.source_hashes))
            entry = None
            route = None
            h_source = None
            if h.source_hashes:
                src = self.concept_graph.lookup_by_canonical_hash(
                    h.source_hashes[0]
                )
                if src is not None and src.status in (
                    ConceptStatus.CRYSTALLIZED,
                    ConceptStatus.STABLE,
                ):
                    h_source = src.id
            for route, start in (
                ("path_state", h.s_at_path),
                ("successor_state", h.s_at_successor),
                ("current_state", s_t),
            ):
                if start is None:
                    continue
                entry = self.consolidator.prune_path(
                    t,
                    h.P_subgraph,
                    h.S_subgraph,
                    aem,
                    self.providers.state_pattern_provider,
                    start,
                    source_concept_id=h_source,
                )
                if entry is not None:
                    break
            if entry is None:
                self.consolidation_diagnostics["handoffs_prune_failed"] += 1
            else:
                self.consolidation_diagnostics["handoffs_cores_extracted"] += 1
                self.consolidation_diagnostics[
                    f"handoff_route_{route}"
                ] += 1
        return covered

    def _consume_consolidation_pairs(
        self, t: int, s_t: int, skip: Optional[Set[Tuple[bytes, bytes]]] = None
    ) -> None:
        """Marshal emitted (P_hash, S_hash) pairs into Consolidator
        prune_path calls (§5.8.1/§5.12). Synchronous, in-step, bounded:
        the pending list is drained every step when the Consolidator is
        wired and hard-capped otherwise so an unwired system cannot leak
        memory (the former list grew without bound and was never read).

        prune_path returning None is a legitimate outcome (replay FAIL —
        the path does not reproduce the outcome from the current state);
        it is counted, not retried here. The Builder re-emits the pair
        after θ_pair_trials further episodes (§4.15.3), which is the
        spec's own retry mechanism."""
        if not self._pending_consolidation_pairs:
            return
        if self.consolidator is None:
            dropped = len(self._pending_consolidation_pairs)
            self._pending_consolidation_pairs.clear()
            self.consolidation_diagnostics["pairs_dropped_no_consolidator"] += dropped
            return
        pairs = self._pending_consolidation_pairs
        self._pending_consolidation_pairs = []
        for p_hash, s_hash in pairs:
            if skip is not None and (p_hash, s_hash) in skip:
                # Already consumed through its rich handoff (B1) — one
                # event, one consumption.
                self.consolidation_diagnostics["pairs_covered_by_handoff"] += 1
                continue
            path = self._reconstruct_subgraph(p_hash)
            outcome = self._reconstruct_subgraph(s_hash)
            if path is None or outcome is None:
                # A crystallized hash from this step's Builder output that
                # cannot be resolved is a graph/index defect: count it
                # observably (silent skips are how this pathway died once
                # already).
                self.consolidation_diagnostics["pairs_unresolvable"] += 1
                continue
            aem = getattr(self.env, "action_edge_map", None)
            if aem is None:
                self.consolidation_diagnostics["pairs_no_action_edge_map"] += 1
                continue
            # §5.3.4 replay simulates from the CURRENT state; the
            # Consolidator resolves the path's action sequence through the
            # state-pattern index (§5.12 implementation signature).
            p_src = None
            src = self.concept_graph.lookup_by_canonical_hash(p_hash)
            if src is not None and src.status in (
                ConceptStatus.CRYSTALLIZED,
                ConceptStatus.STABLE,
            ):
                p_src = src.id
            entry = self.consolidator.prune_path(
                t,
                path,
                outcome,
                aem,
                self.providers.state_pattern_provider,
                s_t,
                source_concept_id=p_src,
            )
            if entry is None:
                self.consolidation_diagnostics["pairs_prune_failed"] += 1
            else:
                self.consolidation_diagnostics["cores_extracted"] += 1

        # (The §5.9 periodic cadence is driven by _consolidation_sweep in
        # Phase 5 — Trigger 1C/2A per the architect ruling.)

    # -------------------------------------------------------------------------
    # Consolidator lifecycle (architect ruling: Triggers 1A / 1C / 2A)
    # -------------------------------------------------------------------------

    def _register_event_bindings(
        self, t: int, action: Any, rising_atoms: Set[Any], detections: Set[Any]
    ) -> None:
        """Event-time schema attribution (v31 wiring; see concept_graph
        step_21 note): when effect atoms RISE at step t and a real action
        was just executed, every causal schema ACTIVE at t is a binding
        candidate for (action, atom). Attribution is only observable at
        the event — emission-time binding (the disabled step_21 block)
        could never see it. Triples are buffered; commitment waits for
        Prediction's counterfactual validation (H_nec), so correlation
        alone never binds."""
        if (
            action is None
            or is_noop_action(action)
            or is_probe_action(action, self.env)
            or is_teleological_action(action)
        ):
            return
        act = self.activation_engine
        if act is None:
            return
        # Onset-lens pattern nodes for the rising atoms (consequence
        # material for Trigger 1A), created in the ConceptGraph's node
        # space so state windows and MCS see the same ids.
        onset_nodes: Set[int] = set()
        for det in detections:
            if getattr(det, "k", None) == "Onset" and getattr(det, "alpha", None) in rising_atoms:
                onset_nodes.add(
                    self.concept_graph.get_or_create_pattern_node(
                        det.k, _node_signature_hash(det.sigma), signature=det.sigma
                    )
                )
        registered = False
        any_active_causal = False
        for cid in sorted(self.concept_graph.concepts.keys()):
            concept = self.concept_graph.concepts[cid]
            if concept.family not in (
                ConceptFamily.EVENT_CAUSAL,
                ConceptFamily.TEMPORAL_CHAIN,
            ):
                continue
            if concept.status in (
                ConceptStatus.TOMBSTONED,
                ConceptStatus.REJECTED,
            ):
                continue
            if act.get_activation(cid, t) <= 0:
                continue
            any_active_causal = True
            for atom in sorted(rising_atoms, key=serialize_int):
                self._pending_bindings.append(
                    (action, atom, cid, frozenset(onset_nodes), t, None)
                )
                registered = True
        if not any_active_causal:
            # [Batch D1b / MED-03] VOID entries: rise occurred, no active
            # causal concept to claim it. cid=None marks the void;
            # _reattribute_pending_voids may home it within W_REATTRIB.
            for atom in sorted(rising_atoms, key=serialize_int):
                self._pending_bindings.append(
                    (action, atom, None, frozenset(onset_nodes), t, None)
                )
        if registered:
            self.consolidation_diagnostics["binding_candidates_registered"] += 1

    def _commit_pending_bindings(self, t: int, observation: Any = None) -> None:
        """Commit buffered (action, atom, concept) triples whose necessity
        Prediction has since validated (check_necessity → H_nec, §8.9).
        Writes concept.actions / effect_ids_by_action / action_to_concepts
        under §1.10-2 (Pipeline-orchestrated module-state updates). Dead
        concepts and superseded triples are dropped; the deque bound (128)
        caps memory.

        [Seam 2] On the first committed binding, also grounds the concept's
        TRIGGER anchor: the positioned entity through which the effect atom
        (often an unpositioned flag) is reachable. UtilityEngine's goal
        fallback prefers concept.anchor but falls through to this when the
        primary anchor has no physical position."""
        if self.prediction is None or not self._pending_bindings:
            return
        for triple in list(self._pending_bindings):
            action, atom, cid, onset_nodes, onset_t, homing_t = triple

            # [Batch D1b] Void: never a commit candidate. Owns the
            # W_REATTRIB clock from its rise step.
            if cid is None:
                if t - onset_t > P.W_REATTRIB:
                    self._pending_bindings.remove(triple)
                    self.consolidation_diagnostics["pending_voids_expired"] += 1
                continue

            # [Batch D1b] Homed-but-uncommitted: expires on its own clock
            # so a lapsed claim is released.
            if homing_t is not None and t - homing_t > P.W_HOMED_TTL:
                self._pending_bindings.remove(triple)
                self.consolidation_diagnostics["pending_homed_expired"] += 1
                continue

            concept = self.concept_graph.concepts.get(cid)
            if concept is None or concept.status in (
                ConceptStatus.TOMBSTONED,
                ConceptStatus.REJECTED,
            ):
                self._pending_bindings.remove(triple)
                continue

            if concept.effect_ids_by_action.get(action) == atom:
                self._pending_bindings.remove(triple)
                continue
            nec = self.prediction.check_necessity(action, atom)
            if nec.get("H_nec") != 1:
                continue  # keep buffered; validation may come later
            concept.actions.add(action)
            concept.effect_ids_by_action[action] = atom
            self.concept_graph.action_to_concepts.setdefault(action, set()).add(cid)
            if getattr(concept, "trigger_anchor", None) is None:
                trigger = self._resolve_trigger_anchor(concept, atom, observation)
                if trigger is not None:
                    concept.trigger_anchor = trigger
                    self.consolidation_diagnostics["trigger_anchors_grounded"] += 1
            if onset_nodes:
                self._effect_nodes_by_concept.setdefault(cid, set()).update(onset_nodes)
            # Single-action registry for the extractor fallback: only when
            # exactly one validated action is bound (multi-action schemas
            # consolidate through the §4.15 chained pathway instead).
            if len(concept.effect_ids_by_action) == 1:
                self._single_action_by_members[frozenset(concept.members)] = action
            elif len(concept.effect_ids_by_action) > 1:
                self._single_action_by_members.pop(frozenset(concept.members), None)
            self._validated_binding_actions.add(action)
            self._pending_bindings.remove(triple)
            self.consolidation_diagnostics["schemas_bound"] += 1
            if homing_t is not None:
                self.consolidation_diagnostics["schemas_bound_reattributed"] += 1
            else:
                # Report split: commits without homing_t (design §4
                # invariant: schemas_bound == direct + reattributed).
                self.consolidation_diagnostics["schemas_bound_direct"] += 1

    def _reattribute_pending_voids(self, t: int) -> None:
        """[Batch D1b] Home pending void entries to concepts that onset
        this step. First-write-wins per concept: a concept may hold at
        most one pending homed triple and must not have any committed
        binding (CRIT-01 — no double-claim).

        Void = (action, atom, None, onset_nodes, onset_t, None): the rise
        was buffered but no causal concept was ACTIVE to claim it. A homed
        triple replaces cid and stamps homing_t = t, which the commit
        loop's W_HOMED_TTL clock then owns. FIFO over the deque (oldest
        voids homed first), capped at K_REATTRIB_MAX claims per step;
        voids older than W_REATTRIB expire here and in the commit loop."""
        if not self._pending_bindings:
            return

        # Concepts with any pending cid-set triple (direct OR homed):
        # reserving both keeps a void from claiming a concept that
        # already has a candidate in flight (invariant: at most one
        # homed triple per concept).
        homed_cids: Set[int] = set()
        for triple in self._pending_bindings:
            if triple[2] is not None:
                homed_cids.add(triple[2])

        # Concepts onsetting this step, deterministic order.
        onsetting: List[int] = []
        for cid in sorted(self.concept_graph.concepts.keys()):
            c = self.concept_graph.concepts.get(cid)
            if c is None:
                continue
            if c.family not in (
                ConceptFamily.EVENT_CAUSAL,
                ConceptFamily.TEMPORAL_CHAIN,
            ):
                continue
            if c.status in (
                ConceptStatus.TOMBSTONED,
                ConceptStatus.REJECTED,
            ):
                continue
            if getattr(c, "onset", 0) == 1:
                onsetting.append(cid)
        if not onsetting:
            return

        claims_this_step = 0
        to_remove: List[Any] = []
        # FIFO over the deque: oldest voids are homed first (MED-04).
        for triple in list(self._pending_bindings):
            if claims_this_step >= P.K_REATTRIB_MAX:
                break
            action, atom, cid, onset_nodes, onset_t, _homing_t = triple
            if cid is not None:
                continue  # already homed (or a direct registration)
            if t - onset_t > P.W_REATTRIB:
                to_remove.append(triple)
                self.consolidation_diagnostics["pending_voids_expired"] += 1
                continue
            for candidate_cid in onsetting:
                if candidate_cid in homed_cids:
                    continue
                candidate = self.concept_graph.concepts.get(candidate_cid)
                if candidate is None:
                    continue
                if not (onset_nodes & candidate.members):
                    continue
                # Home: replace tuple with cid + homing_t set.
                idx = self._pending_bindings.index(triple)
                self._pending_bindings[idx] = (
                    action, atom, candidate_cid, onset_nodes, onset_t, t,
                )
                homed_cids.add(candidate_cid)
                claims_this_step += 1
                self.consolidation_diagnostics[
                    "schemas_bound_reattributed_prelim"
                ] += 1
                break

        for t_rem in to_remove:
            try:
                self._pending_bindings.remove(t_rem)
            except ValueError:
                pass

    def _resolve_trigger_anchor(
        self, concept: Any, effect_atom: Any, observation: Any
    ) -> Optional[Any]:
        """[Seam 2] Pick the concept's trigger-entity anchor: among the
        anchors the concept itself recorded (cross-section Anchors ∪
        primary anchor), the first — in deterministic canonical order —
        that is NOT the effect atom and HAS a real physical position in
        the current observation. None when no candidate qualifies (the
        utility fallback then behaves exactly as before)."""
        candidates: Set[Any] = set(getattr(concept, "Anchors", ()) or ())
        primary = getattr(concept, "anchor", None)
        if primary is not None:
            candidates.add(primary)
        candidates.discard(effect_atom)
        for anchor in sorted(candidates, key=serialize_value):
            if resolve_anchor_position(observation, anchor) is not None:
                return anchor
        return None

    def _update_binding_peaks_and_expiry(self, t: int) -> None:
        """[Batch D1] Peak track + provisional expiry for committed
        (action -> effect) bindings. Called every step, independent of
        `_pending_bindings` state (a committed binding no longer has a
        pending triple, yet it is exactly what must be re-examined).

        PEAK TRACK: N_cf = min(C_opp[a], C_opp_not[a]) decays at
        gamma_cf between executions, so the LIVE value understates the
        evidence a binding was committed on (measured: 6.3 -> 2.8 -> 0.3
        between toggle bursts). concept.effect_confidence[action] records
        the highest N_cf seen while the binding is live — monotone, only
        raises — and is what the utility engine's anti-undo and
        goal-satisfied gates read.

        EXPIRY: a binding whose necessity has lapsed (H_nec == 0) AND
        whose contrast has gone NEGATIVE (bar_dp < 0) is disconfirmed —
        the action is now observed to succeed as often WITHOUT this
        effect as with it. The provisional binding is withdrawn from
        effect_ids_by_action / actions / action_to_concepts; the
        single-action and effect-node registries are pruned when the
        concept retains no binding, and the global
        `_validated_binding_actions` entry is dropped once no concept
        retains that action (the extractor fallback reads that set).
        """
        if self.prediction is None:
            return
        pred = self.prediction
        for cid in list(self.concept_graph.concepts.keys()):
            concept = self.concept_graph.concepts.get(cid)
            if concept is None:
                continue
            eff_map = getattr(concept, "effect_ids_by_action", None)
            if not eff_map:
                continue
            for action in list(eff_map.keys()):
                atom = eff_map[action]
                c_opp = pred.C_opp.get(action, 0.0)
                c_opp_not = pred.C_opp_not.get(action, 0.0)
                current_n_cf = min(c_opp, c_opp_not)
                current_h_nec = int(pred.prev_H_nec.get((action, atom), 0))
                bar_dp = pred.bar_dp_dict.get((action, atom), 0.0)
                prev_peak = concept.effect_confidence.get(action, 0.0)
                if current_n_cf > prev_peak:
                    concept.effect_confidence[action] = current_n_cf
                if current_h_nec == 0 and bar_dp < 0.0:
                    del eff_map[action]
                    concept.effect_confidence.pop(action, None)
                    concept.actions.discard(action)
                    self.concept_graph.action_to_concepts[action].discard(cid)
                    self.consolidation_diagnostics[
                        "provisional_bindings_expired"
                    ] += 1
                    if not eff_map:
                        self._effect_nodes_by_concept.pop(cid, None)
                        self._single_action_by_members.pop(
                            frozenset(concept.members), None
                        )
                        any_retains = any(
                            action in c.effect_ids_by_action
                            for c in self.concept_graph.concepts.values()
                        )
                        if not any_retains:
                            self._validated_binding_actions.discard(action)

    # -------------------------------------------------------------------------
    # [B3/§5.8.2] Discriminative consolidation trigger (label windows)
    # -------------------------------------------------------------------------

    def _derive_label_signals(self, t: int) -> List[LabelSignal]:
        """Label signals from the pipeline's OWN effect-atom tracking —
        environment-agnostic by construction: every symbolic env exposes
        effect indices through the observation, and the pipeline already
        maintains per-atom present/absent values (prev_effect_flags) for
        the §8.3 onset machinery. An atom PRESENT at t labels the current
        situation positive; ABSENT labels it negative. No external label
        provider, no env-specific semantics."""
        signals: List[LabelSignal] = []
        for b in sorted(self.all_effect_flags, key=serialize_value):
            v_b = 1 if self.prev_effect_flags.get(b, 0) else -1
            signals.append(LabelSignal(label=("effect_atom", b), polarity=v_b, t=t))
        return signals

    def _update_label_windows(self, t: int, s_t: int) -> None:
        """Accumulate per-label observation windows and fire the §5.8.2
        discriminative trigger when a label is ready (≥ θ_instances
        DISTINCT positive windows, ≥ 1 negative, diversity ≥ θ_diversity).
        Windows come from the StatePatternIndex (already maintained for
        §5.3.4 replay) — no extra observation machinery. Firing calls the
        Consolidator's public discriminative_core, which registers the
        CATEGORY core in the shared CoreLibrary (D8: Mapper.query_category
        then has real learned categories to match)."""
        if self.consolidator is None or not self.all_effect_flags:
            return
        window = self.providers.state_index(s_t)
        if window is None or not window.nodes:
            return
        self.window_buffer.evict_stale(t)
        label_signals = self._derive_label_signals(t)
        for sig in label_signals:
            self.window_buffer.record(sig.label, sig.polarity, window, t)
        # [Phase 1 Decision 2 / V5-04] Manager owns the ready-trigger +
        # extraction for label-window categories. Trigger-2A (concept
        # sigma labels) still calls _extract_category_core directly.
        if self.category_manager is not None:
            self.category_manager.step(t, label_signals)

    def _bootstrap_action_core(self, concept: Any, t: int) -> None:
        """Trigger 1A/1C packaging (ruling): prune_path over (concept
        organization → state-change consequence) replayed from a REAL
        enabling state. Start states come from the AEM (states where a
        bound action is recorded), preferring the concept's own
        enabling_pre_states — the recorded set also contains mere
        observation states, from which the action was never executed.
        Consequence O = the state-change pattern (window(s_post) \
        window(s_pre), i.e. E_P as "effect pattern / state change" per the
        ruling) plus any Onset-lens nodes registered at binding time.
        Dedup and the confirmed-transition permanence gate (N_edge=5,
        θ_dominance=0.8 — ruling Item 1) are enforced inside the
        Consolidator's replay."""
        ch = getattr(concept, "canonical_hash", b"")
        if not ch:
            return
        bound = dict(getattr(concept, "effect_ids_by_action", {}) or {})
        if not bound:
            self.consolidation_diagnostics["bootstrap_unbound"] += 1
            return
        path = self._reconstruct_subgraph(ch)
        if path is None:
            self.consolidation_diagnostics["bootstrap_path_unresolvable"] += 1
            return
        # [CORRECTION 2] Sequence-edge safety guard. The §5.3.4 replay derives
        # the path's action sequence from the topological order of its SEQUENCE
        # edges. D1c (2026-09-24) admits single-bound 1-step schemas via the
        # fallback extractor; the check now refuses only subgraphs with neither
        # SEQUENCE structure nor a single-action binding.
        has_sequence = any(
            et == EdgeType.SEQUENCE for (_src, _dst, et) in path.edges
        )
        has_single_binding = (
            frozenset(path.nodes) in self._single_action_by_members
        )
        if not has_sequence and not has_single_binding:
            self.consolidation_diagnostics["bootstrap_no_sequence_edge"] += 1
            return
        if not has_sequence and has_single_binding:
            # 1-step causal schema with a committed single binding. Resolves
            # via the extractor fallback registry (single-bound only; multi-
            # bound schemas are out of D1c's scope, see multi-binding policy).
            self.consolidation_diagnostics["bootstrap_single_binding_fallback"] += 1
        aem = getattr(self.env, "action_edge_map", None)
        if aem is None:
            self.consolidation_diagnostics["bootstrap_no_aem"] += 1
            return
        provider = self.providers.state_pattern_provider
        # AEM-recorded start states for each bound action (deterministic
        # canonical order), preferring the concept's enabling states.
        eps = set(getattr(concept, "enabling_pre_states", ()) or ())
        attempts = 0
        onset_nodes = set(self._effect_nodes_by_concept.get(concept.id, ()))
        for action in sorted(bound.keys(), key=lambda a: serialize_value(a)):
            # Evidence-first scan: start states ordered by transition mass
            # (descending), enabling states preferred within equal mass,
            # canonical serialization as the final deterministic
            # tie-break. The permanence gate (ruling Item 1) only admits
            # mass ≥ N_edge transitions, so the best-evidenced starts are
            # exactly the replayable ones — a static hash order could
            # strand them behind the per-call attempt bound while the
            # evidence accumulated elsewhere.
            get_mass = getattr(aem, "get_mass", None)
            states = sorted(
                (s for (s, a), _info in aem.items() if a == action),
                key=lambda s: (
                    -(float(get_mass(s, action)) if callable(get_mass) else 0.0),
                    0 if s in eps else 1,
                    serialize_int(s),
                ),
            )
            ordered = states
            for s_pre in ordered:
                if attempts >= 8:  # bounded replays per concept per call
                    break
                s_post = aem.predict(s_pre, action)
                if s_post is None:
                    continue
                w_pre = provider(s_pre)
                w_post = provider(s_post)
                if w_pre is None or w_post is None:
                    continue
                delta = set(w_post.nodes) - set(w_pre.nodes)
                o_nodes = (delta | onset_nodes) & set(w_post.nodes)
                if not o_nodes:
                    # Falling-edge transition (effect REMOVED: the toggle
                    # that closed an open door) — no new pattern exists to
                    # consolidate. Skip without counting: the opening
                    # instance of the same action is scanned next.
                    continue
                o_edges = {e for e in w_post.edges if e[0] in o_nodes and e[1] in o_nodes}
                outcome = Subgraph(
                    nodes=frozenset(o_nodes),
                    edges=frozenset(o_edges),
                    anchor_of_node=anchor_positions_for(
                        o_nodes, self.concept_graph.pattern_node_position
                    ),
                )
                attempts += 1
                self._bootstrap_attempts[ch] += 1
                entry = self.consolidator.prune_path(
                    t, path, outcome, aem, provider, s_pre,
                    source_concept_id=concept.id,
                )
                if entry is not None:
                    self.consolidation_diagnostics["bootstrap_cores_extracted"] += 1
                    return
        if attempts == 0:
            self.consolidation_diagnostics["bootstrap_no_replayable_start"] += 1
        else:
            self.consolidation_diagnostics["bootstrap_replay_failed"] += 1

    def _trigger_bootstrap_consolidation(self, t: int, newly_crystallized: List[Any]) -> None:
        if self.consolidator is None or not newly_crystallized:
            return
        for concept in newly_crystallized:
            if concept.family not in (
                ConceptFamily.EVENT_CAUSAL,
                ConceptFamily.TEMPORAL_CHAIN,
            ):
                continue
            if concept.status in (ConceptStatus.TOMBSTONED,):
                continue
            self._bootstrap_action_core(concept, t)
            # Trigger 2A rides the same event: multi-anchor concepts get a
            # category core immediately when the anchor evidence exists.
            self._category_core(concept, t)

    def _category_core(self, concept: Any, t: int) -> None:
        """Trigger 2A (ruling): multi-anchor crystallization → discriminative
        category core. Positives: recorded situation windows containing the
        concept's member patterns (each window = one anchor-instance
        observation); negatives: windows with zero overlap from the same
        recorded range (deterministic sampling — §9 forbids randomness).
        Label: the concept's signature (§5.6 category identification)."""
        ch = getattr(concept, "canonical_hash", b"")
        if not ch or ch in self._category_consolidated:
            return
        if getattr(concept, "N_anchor", 0) < P.THETA_ANCHOR_DEFAULT:
            return
        members = set(getattr(concept, "members", ()) or ())
        if not members:
            return
        index = self.providers.state_index
        positives = index.windows_containing(members, P.CATEGORY_WINDOW_MAX)
        if not positives:
            self.consolidation_diagnostics["category_insufficient_windows"] += 1
            return
        negatives = index.windows_without(members, len(positives))
        if not negatives and len(index) > 0:
            # Distinguish WHY negatives are empty.
            # [ISSUE 4 FIX / ADAPTATION — FIX B] StatePatternIndex is
            # label-agnostic: windows are keyed by state id and carry no
            # .label, and count_for_label does not exist, so a per-label
            # window count is unavailable. The faithful equivalent of
            # "buffer has windows but none discriminate" is the total
            # recorded-window count: len(index) == 0 ⇒ buffer empty (only
            # category_skipped_no_negatives fires, inside
            # discriminative_core); len(index) > 0 with empty negatives ⇒
            # every recorded window overlaps this concept's members ⇒
            # all_overlap. len() on the backing dict is O(1), so FIX B's
            # per-sweep caching is unnecessary (and _category_core is also
            # invoked outside sweeps by _trigger_bootstrap_consolidation,
            # where a sweep-scoped cache would not apply).
            self.consolidator.diagnostics[
                "category_skipped_all_overlap"
            ] += 1
            # Fall through to the extractor, which reaches
            # discriminative_core: empty negatives are caught there and
            # yield None.
        # [G.2.2 FIX — silent-loss bug]
        # The previous version marked the concept consolidated
        # unconditionally, ignoring the extractor's return value. Since
        # _category_core is entered under `if ch in self._category_consolidated:
        # return` (see the top of this function), a concept whose
        # extraction returned None — whether because of budget starvation,
        # an empty intersection, or a uniform positive set — was marked
        # done and NEVER retried. A resource-contention event became
        # permanent data loss.
        #
        # This fix marks the concept consolidated only when the extractor
        # returned a CoreEntry, and caps re-attempts so a genuinely
        # unextractable concept does not re-enter the sweep forever. The
        # cap semantics: a concept receives up to
        # MAX_CATEGORY_EXTRACTION_ATTEMPTS real extraction attempts; the
        # (N+1)-th sweep encounter marks it abandoned without calling the
        # extractor again.
        attempt_count = self._category_consolidation_attempts.get(ch, 0)
        if attempt_count >= P.MAX_CATEGORY_EXTRACTION_ATTEMPTS:
            self._category_consolidated.add(ch)
            self._category_consolidation_attempts.pop(ch, None)
            self.consolidation_diagnostics[
                "category_core_extraction_abandoned"
            ] += 1
            return

        entry = self._extract_category_core(
            t, getattr(concept, "sigma", None), positives, negatives,
            source_concept_id=concept.id,
        )
        if entry is not None:
            self._category_consolidated.add(ch)
            self._category_consolidation_attempts.pop(ch, None)
        else:
            self._category_consolidation_attempts[ch] = attempt_count + 1
            self.consolidation_diagnostics[
                "category_core_extraction_deferred"
            ] += 1

    def _consolidation_sweep(self, t: int) -> None:
        """Trigger 1C + periodic 2A sweep (§5.9 cadence): retry bootstrap
        action cores for causal concepts whose transitions have since
        reached confirmation, and extract category cores for concepts that
        accrued ≥ 2 anchors after crystallization. Deterministic iteration
        (sorted concept ids); bounded by concept count."""
        if self.consolidator is None:
            return
        causals: List[Any] = []
        all_live: List[Any] = []
        for cid in sorted(self.concept_graph.concepts.keys()):
            concept = self.concept_graph.concepts[cid]
            if concept.status in (ConceptStatus.TOMBSTONED,):
                continue
            if concept.status is not ConceptStatus.REJECTED:
                all_live.append(concept)
            if concept.family in (
                ConceptFamily.EVENT_CAUSAL,
                ConceptFamily.TEMPORAL_CHAIN,
            ) and concept.status is not ConceptStatus.REJECTED:
                causals.append(concept)
        self._commit_pending_bindings(t)
        # [Batch D1] Same every-step pass as post-execution (the sweep's
        # commit may add a binding whose peak must be recorded immediately;
        # the pass is idempotent within a step).
        self._update_binding_peaks_and_expiry(t)
        for concept in causals:
            self._bootstrap_action_core(concept, t)
        for concept in all_live:
            self._category_core(concept, t)

    # -------------------------------------------------------------------------
    # [Batch D3] Shared category-core producer
    # -------------------------------------------------------------------------

    def _extract_category_core(
        self,
        t: int,
        label: Any,
        positives: List[Subgraph],
        negatives: List[Subgraph],
        source_concept_id: Optional[int] = None,
    ) -> Optional[CoreEntry]:
        """Single producer for every §5.8.2 CATEGORY core.

        Both producers — Trigger 2A (`_category_core`, anchor-evidence
        windows) and the label-window trigger (`_update_label_windows`,
        effect-atom label windows) — route here, so the extraction policy
        exists exactly once:

          1. |W+| >= THETA_CATEGORY_MIN_POSITIVES (structural minimum for
             a discriminative intersection to exist at all);
          2. div(W+) >= MIN_DIVERSITY_FLOOR (reject degenerate positive
             sets — structural spread ~ 0 carries no discriminating
             structure);
          3. diversity is otherwise NOT a veto but a soft reliability
             factor: factor = clamp(div/TARGET_DIVERSITY,
             MIN_RELIABILITY_FACTOR, 1.0), applied as
             reliability = KAPPA_INIT * factor by the Consolidator.

        Precondition: self.consolidator is not None — every caller
        (_consolidation_sweep, _trigger_bootstrap_consolidation,
        _update_label_windows) guards it.
        """
        if len(positives) < P.THETA_CATEGORY_MIN_POSITIVES:
            self.consolidation_diagnostics[
                "category_insufficient_windows"
            ] += 1
            return
        diversity = self._compute_diversity(positives)
        if diversity < P.MIN_DIVERSITY_FLOOR:
            self.consolidation_diagnostics["category_uniform_windows"] += 1
            return
        factor = max(
            P.MIN_RELIABILITY_FACTOR,
            min(1.0, diversity / P.TARGET_DIVERSITY),
        )
        entry = self.consolidator.discriminative_core(
            t, label, positives, negatives,
            source_concept_id=source_concept_id,
            initial_reliability_factor=factor,
        )
        if entry is None:
            self.consolidation_diagnostics["category_no_intersection"] += 1
            return None
        self.consolidation_diagnostics["category_cores_extracted"] += 1
        return entry

    def _compute_diversity(self, positives: List[Subgraph]) -> float:
        """Mean pairwise structural DISTANCE of a positive window set:
        1 - mean_{i<j} Jaccard(node_i, node_j).

        Node-set Jaccard (not the signature-aware
        subgraph_structural_similarity) is deliberate: this factor only
        scales an initial reliability, so an O(n^2) node-level estimate
        over <= CATEGORY_WINDOW_MAX windows is sufficient and stays off
        the §5.8.2 signature path. 1.0 for fewer than two windows (a
        single window carries no measured spread and must not be
        penalised). Deterministic: no RNG, arrival-order-independent.
        """
        if len(positives) < 2:
            return 1.0
        total = 0.0
        n = 0
        for i in range(len(positives)):
            for j in range(i + 1, len(positives)):
                a, b = positives[i].nodes, positives[j].nodes
                union = a | b
                if not union:
                    continue
                total += len(a & b) / len(union)
                n += 1
        return 1.0 - (total / n) if n else 1.0

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    def _run_diagnostics(
        self,
        t,
        D_raw,
        D_rec,
        D_op,
        D_chunk,
        D_bind,
        valid_cross_sections,
        proposals,
        utilities,
        action,
        s_t,
    ):
        diag = {}
        # [PROBE FIX] the entry previously carried no step index at all
        # (probe read entry.get("t") -> None); populate from the step arg.
        diag["t"] = t
        # [Batch 3 Phase 3] CDP-extinction tracker, per DIAG tick.
        # [ADAPTATION] The directive located this "inside the existing
        # self.t % DIAG_INTERVAL == 0 block in _run_diagnostics"; the
        # cadence block actually lives in step() and calls this method,
        # so the update runs here (identical cadence). Events carry the
        # DIAG label `t` (== self.t - 1 at this point), matching the
        # "=== DIAGNOSTIC at t=..." headers. Iteration is sorted() for
        # canonical event order (determinism mandate §9). Read-side only.
        if self._cdp_tracking_enabled:
            current_cdps = {
                c.id
                for c in self.concept_graph.concepts.values()
                if c.family == ConceptFamily.CROSS_DOMAIN_PATTERN
            }
            for cid in sorted(current_cdps - set(self._cdp_tracker.keys())):
                c = self.concept_graph.concepts[cid]
                self._cdp_transitions.append({
                    "t": t, "event": "entered", "cdp_id": cid,
                    "T_cryst": c.T_cryst, "status": c.status.name,
                    "N_conf": c.N_conf, "N_opp": c.N_opp,
                })
                self._cdp_tracker[cid] = t
            for cid in sorted(set(self._cdp_tracker.keys()) - current_cdps):
                c = self.concept_graph.concepts.get(cid)
                if c is None:
                    cause = "evicted"
                elif c.status.name == "TOMBSTONED":
                    cause = "tombstoned"
                elif c.status.name == "REJECTED":
                    cause = "rejected"
                elif getattr(c, "dormant", False):
                    cause = "dormant"
                else:
                    cause = "unknown"
                self._cdp_transitions.append({
                    "t": t, "event": "left", "cdp_id": cid, "cause": cause,
                })
                del self._cdp_tracker[cid]
            self.diagnostics["cdp_count_current"] = len(current_cdps)
            diag["cdp_count_current"] = len(current_cdps)
        self.diagnostics["concept_effects_size"] = len(
            self._current_concept_effects
        )
        self.diagnostics["effect_register_size"] = len(
            self.effect_presence_register
        )
        diag["lens"] = {
            "D_raw_count": len(D_raw),
            "D_rec_count": len(D_rec),
            "D_op_count": len(D_op),
            "D_chunk_count": len(D_chunk),
            "D_bind_count": len(D_bind),
        }
        diag["graph"] = self._graph_diagnostic()
        diag["activation"] = self._activation_diagnostic()
        diag["fertility"] = self._fertility_diagnostic()
        diag["builder"] = self._builder_diagnostic()
        # [Batch-1 ADAPTATION] Binder DIAG surface: nothing previously
        # printed Binder.diagnostics at DIAG ticks. Read-only observer per
        # §4.7 (no behavior change). Surfaces the [P2-5] steady-pair counter
        # and per-type emission counts (incl. SYNCHRONY after Phase D).
        # Sorted so the stdout repr is seed-independent.
        _binder_ref = getattr(self, "binder", None)
        diag["binder"] = (
            {
                "co_activation_steady_pairs_skipped": (
                    _binder_ref.diagnostics.get(
                        "co_activation_steady_pairs_skipped", 0
                    )
                ),
                "emitted_by_type": dict(
                    sorted(_binder_ref.diagnostics["emitted_by_type"].items())
                ),
            }
            if _binder_ref is not None
            else {}
        )

        causal_concepts = [
            c
            for c in self.concept_graph.concepts.values()
            if c.family in (ConceptFamily.EVENT_CAUSAL, ConceptFamily.TEMPORAL_CHAIN)
            and c.status in (ConceptStatus.CRYSTALLIZED, ConceptStatus.STABLE)
        ]
        diag["causal_concepts"] = {
            "count": len(causal_concepts),
            "ids": [c.id for c in causal_concepts[:10]],
            "families": sorted({c.family.name for c in causal_concepts}),
        }
        sorted_u = sorted(utilities.items(), key=lambda kv: (-kv[1], str(kv[0])))
        diag["utility"] = {
            "proposal_count": len(proposals),
            "utility_stats": self._stats(list(utilities.values())),
            "top5_utilities": [(str(a), round(u, 4)) for a, u in sorted_u[:5]],
            "selected_action": action,
            "rho_spawn": round(getattr(self.utility_engine, "rho_spawn", 0.0), 4),
        }

        # Track winning proposal source (proposal-origin percentage tracking)
        winner_source = "MOTOR"
        if action in self.utility_engine._plan_actions(s_t):
            winner_source = "TRANSFER"
        elif any(
            getattr(c, "action", None) == action
            for c in self.concept_graph.concepts.values()
            # [Patch 1] ATOMIC_ACTION concepts mirror the motor repertoire;
            # counting them as GOAL mislabeled every motor action (the
            # "88.7% GOAL" contamination). Diagnostic label only — no
            # behavior change.
            if getattr(c, "family", None) != ConceptFamily.ATOMIC_ACTION
        ):
            winner_source = "GOAL"
        elif action == getattr(self.utility_engine, "_orient_action", None):
            winner_source = "ORIENT"
        self._proposal_origin_counts[winner_source] += 1
        _total = sum(self._proposal_origin_counts.values())
        if _total > 0:
            diag["proposal_origin_pct"] = {
                k: round(v / _total * 100, 1)
                for k, v in self._proposal_origin_counts.items()
            }
        else:
            diag["proposal_origin_pct"] = {}
        diag["proposal_origin"] = winner_source

        # [Floor-fix verification] theta trajectory in every DIAG block.
        # Diagnostic-only: reads the UtilityEngine's homeostatic state,
        # changes no behavior. Absent attributes print as None (cold
        # start, before the first homeostasis update initialises them).
        diag["theta_frontier_current"] = getattr(
            self.utility_engine, "_theta_frontier", None
        )
        diag["r_bar_goal_current"] = getattr(
            self.utility_engine, "_r_bar_goal", None
        )
        if self.mapper is not None:
            md = self.mapper.get_diagnostics()
            diag["fired_situation_calls"] = md.get("fired_situation_calls", 0)
        if self.category_manager is not None:
            diag["category_manager"] = self.category_manager.get_diagnostics()

        self.diagnostics_log.append(diag)
        if len(self.diagnostics_log) > self.max_diagnostics:
            self.diagnostics_log.pop(0)

        print(f"=== DIAGNOSTIC at t={t} ===")
        for k, v in diag.items():
            print(f"{k}: {v}")

    def _lift_to_concepts(self, nid: int) -> Set[int]:
        """[Batch D / P1-3 MED-05] Concept ids associated with a pattern
        node: member concepts plus the ATOMIC_ACTION concept whose action
        matches the node's registered action (if any)."""
        result: Set[int] = set(self.concept_graph.node_to_concepts(nid))
        action = self.concept_graph.get_node_action(nid)
        if action is not None:
            for cid, c in self.concept_graph.concepts.items():
                if (
                    c.family == ConceptFamily.ATOMIC_ACTION
                    and getattr(c, "action", None) == action
                ):
                    result.add(cid)
        return result

    def _compute_precondition_credit(self, basis_concepts: Set[int]) -> Dict[int, float]:
        """[Batch D / P1-3] Backward walk along SEQUENCE/TRANSITION edges
        from the plan basis; credit decays by D_C per hop and by the edge
        weight transform w/(w+1); capped at MAX_HOPS_CREDIT hops."""
        basis_nodes: Set[int] = set()
        for cid in basis_concepts:
            c = self.concept_graph.concepts.get(cid)
            if c is not None:
                basis_nodes.update(c.members)
        if not basis_nodes:
            return {}
        reached = set(basis_nodes)
        mass: Dict[int, float] = {nid: 1.0 for nid in basis_nodes}
        credit: Dict[int, float] = defaultdict(float)
        for _hop in range(P.MAX_HOPS_CREDIT):
            nxt: Dict[int, float] = defaultdict(float)
            for nid, m in mass.items():
                for e in self.concept_graph.get_incoming_edges(nid):
                    if e.edge_type not in (EdgeType.SEQUENCE, EdgeType.TRANSITION):
                        continue
                    if e.parent in reached:
                        continue
                    w = e.weight / (e.weight + 1.0)
                    nxt[e.parent] += m * P.D_C * w
            if not nxt:
                break
            for nid, v in nxt.items():
                reached.add(nid)
                for cid in self._lift_to_concepts(nid):
                    if cid in basis_concepts:
                        continue
                    credit[cid] += v
            mass = nxt
        return dict(credit)

    def _reconstruct_chain(self, core_hash) -> List[int]:
        """[Batch D / P1-4] Causal chain of a transferred core: member
        pattern nodes of the core subgraph, lifted to concepts, ordered by
        Kahn topological sort over SEQUENCE/TRANSITION edges.

        ADAPTATION NOTE: the v5 draft says "unchanged from v4", but v4
        never landed in this tree — no prior implementation exists. This
        is a fresh implementation of the described algorithm (Kahn + lift
        via _lift_to_concepts), deterministic by canonical node ordering."""
        entry = None
        for e in self.consolidator.get_core_library().entries:
            if e.canonical_hash == core_hash:
                entry = e
                break
        if entry is None:
            return []
        members = set(entry.core_subgraph.nodes)
        if not members:
            return []
        indeg: Dict[int, int] = {n: 0 for n in members}
        adj: Dict[int, List[int]] = {n: [] for n in members}
        for n in sorted(members):
            for e in self.concept_graph.get_outgoing_edges(n):
                if (
                    e.child in members
                    and e.edge_type in (EdgeType.SEQUENCE, EdgeType.TRANSITION)
                ):
                    adj[n].append(e.child)
                    indeg[e.child] += 1
        queue = sorted(n for n in members if indeg[n] == 0)
        order: List[int] = []
        ordered_set: Set[int] = set()
        while queue:
            n = queue.pop(0)
            order.append(n)
            ordered_set.add(n)
            for m in sorted(adj[n]):
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
            queue.sort()
        for n in sorted(members):
            if n not in ordered_set:
                order.append(n)
        chain: List[int] = []
        seen: Set[int] = set()
        for n in order:
            for cid in sorted(self._lift_to_concepts(n)):
                if cid not in seen:
                    seen.add(cid)
                    chain.append(cid)
        return chain

    def _find_break_idx(self, chain: List[int], t_at: int) -> int:
        """[Batch D / P1-4] Index of the chain step where activation died
        (last still-active concept); defaults to the final step."""
        act = self.activation_engine
        if act is None or not chain:
            return len(chain) - 1
        try:
            _ = act.get_activation(chain[-1], t_at)
        except (KeyError, IndexError, TypeError):
            return len(chain) - 1
        last = 0
        for i, cid in enumerate(chain):
            try:
                if act.get_activation(cid, t_at) > 0:
                    last = i
            except (KeyError, IndexError, TypeError):
                continue
        return last

    def _graph_diagnostic(self):
        concepts = self.concept_graph.concepts
        by_layer = defaultdict(int)
        by_status = defaultdict(int)
        by_family = defaultdict(int)
        for c in concepts.values():
            by_layer[c.layer] += 1
            by_status[c.status.name] += 1
            by_family[c.family.name] += 1
        # [Batch D / P1-4] top-5 concepts by plan_blame_streak.
        top_blame = sorted(
            (
                (cid, c.plan_blame_streak)
                for cid, c in concepts.items()
                if getattr(c, "plan_blame_streak", 0) > 0
            ),
            key=lambda kv: (-kv[1], kv[0]),
        )[:5]
        return {
            "concept_count_by_layer": dict(by_layer),
            "concept_count_by_status": dict(by_status),
            "concept_count_by_family": dict(by_family),
            "top5_plan_blame": top_blame,
            "active_edges": len(self.concept_graph.get_active_edges()),
            # [P2-3] articulation-safe pruning counter; [P2-4] entropy
            # homeostasis counter + current type-share entropy (base 2).
            "pattern_nodes_preserved_articulation": self.concept_graph.diagnostics.get(
                "pattern_nodes_preserved_articulation", 0
            ),
            "edge_entropy_rebalances": self.concept_graph.diagnostics.get(
                "edge_entropy_rebalances", 0
            ),
            "edge_entropy_current": round(
                float(self.concept_graph.edge_entropy_current), 4
            ),
        }

    @staticmethod
    def _stats(values):
        vals = list(values)
        if not vals:
            return {"n": 0, "min": 0.0, "mean": 0.0, "max": 0.0}
        # [DETERMINISM §9] min/max over an insertion-ordered dict of
        # values can return an int 0 or a float 0.0 depending on which
        # zero was inserted first (proposal-set iteration order is
        # seed-randomized pre-fix); round() preserves the type, so the
        # diagnostic repr differed across seeds. Coerce to float and use
        # math.fsum (exact, order-independent) for the mean.
        return {
            "n": len(vals),
            "min": round(float(min(vals)), 4),
            "mean": round(math.fsum(vals) / len(vals), 4),
            "max": round(float(max(vals)), 4),
        }

    def _activation_diagnostic(self):
        concepts = self.concept_graph.concepts
        active_ids = [cid for cid, c in concepts.items() if c.A == 1]
        adapt_vals = [getattr(c, "adapt_P", 0.0) for c in concepts.values()]
        support_vals = [getattr(c, "supportTrace", 0.0) for c in concepts.values()]
        return {
            "active_count": len(active_ids),
            "active_ids_sample": active_ids[:10],
            "adapt": self._stats(adapt_vals),
            "supportTrace": self._stats(support_vals),
        }

    def _fertility_diagnostic(self):
        concepts = self.concept_graph.concepts
        credit_vals = [getattr(c, "credit", 0.0) for c in concepts.values()]
        rf_vals = [getattr(c, "RF", 0.0) for c in concepts.values()]
        ef_combined_vals = [getattr(c, "EF_combined", 0.0) for c in concepts.values()]
        top = sorted(
            ((cid, getattr(c, "EF_combined", 0.0)) for cid, c in concepts.items()),
            key=lambda kv: -kv[1],
        )[:5]
        return {
            "credit": self._stats(credit_vals),
            "RF": self._stats(rf_vals),
            "EF_combined": self._stats(ef_combined_vals),
            "top5_by_EF_combined": top,
        }

    def _builder_diagnostic(self):
        # §4.21 Builder diagnostic interface is mandatory — the former
        # five-layer try/except turned any Builder wiring defect into
        # empty diagnostics (§4.7 requires diagnostics be read-only
        # observers, not error sinks).
        bd = self.builder.get_diagnostics()
        spine = self.builder.get_spine()
        core_library = getattr(self.utility_engine, "core_library", None)
        return {
            "mode": self.builder.get_mode(),
            "spine_basin_id": spine.id if spine is not None else None,
            "frontier_size": len(self.builder.get_frontier()),
            "core_library_size": (
                core_library.size() if core_library is not None else 0
            ),
            "drop_no_family": bd.get("dropped_no_family", 0),
            "scaffold_population": bd.get(
                "uncrystallizable_scaffold_count", 0
            ),
            # [P0 v4 V7] Surface the nine v4 counters (plus the two
            # Session-5 membership counters they interpret) at
            # DIAG_INTERVAL cadence. [ADAPTATION] the directive does not
            # specify the surfacing site; this curated dict is what the
            # DIAGNOSTIC block prints, so the counters are exposed here
            # (read-only observers per §4.7 — no behavior change).
            "p0": {
                "theta_fertile": bd.get("theta_fertile", 0.0),
                "fertile_relaxed": self.builder._fertile_relaxed,
                "membership_reobservations": bd.get(
                    "membership_reobservations", 0
                ),
                "membership_reobservation_partial": bd.get(
                    "membership_reobservation_partial", 0
                ),
                "membership_reobservation_edge_rejected": bd.get(
                    "membership_reobservation_edge_rejected", 0
                ),
                "fertile_rescue_fired": bd.get("fertile_rescue_fired", 0),
                "fertile_rescue_top_val": bd.get("fertile_rescue_top_val", 0.0),
                "fertile_rescue_fired_meaningful": bd.get(
                    "fertile_rescue_fired_meaningful", 0
                ),
                "fertile_rescue_fired_marginal": bd.get(
                    "fertile_rescue_fired_marginal", 0
                ),
                "bootstrap_gate_active_steps": bd.get(
                    "bootstrap_gate_active_steps", 0
                ),
                "theta_fertile_empty_state_recomputes": bd.get(
                    "theta_fertile_empty_state_recomputes", 0
                ),
                "prune_frontier_active_steps": bd.get(
                    "prune_frontier_active_steps", 0
                ),
                "cold_start_max_n_conf": bd.get("cold_start_max_n_conf", 0),
            },
            "consolidation": dict(self.consolidation_diagnostics),
        }


# =============================================================================
# Signature provider (Item 4)
# =============================================================================


def make_signature_provider(
    concept_graph: Any,
) -> Callable[[int], Optional[Tuple[str, Any]]]:
    """Return a callable `(nid) -> Optional[(lens_type, TypedSignature)]`.

    Reads from the D1 pattern-node signature store. Returns None for
    unknown or signature-less nodes.

    The returned tuple's second element is the raw TypedSignature object
    from the graph's store — it is not copied. Callers must not mutate it;
    TypedSignature is a frozen dataclass, so mutability is already
    prevented by construction.
    """

    def _provider(nid: int) -> Optional[Tuple[str, Any]]:
        key = concept_graph.get_pattern_node_signature(nid)
        if key is None:
            return None
        sig = concept_graph.get_pattern_node_full_signature(nid)
        if sig is None:
            return None
        return (key[0], sig)

    return _provider


# =============================================================================
# State-pattern index (Item 5)
# =============================================================================


class StatePatternIndex:
    """Maintains `state_hash -> Subgraph` for the Executor's target_ok.

    The pipeline calls `record(s_t, D_bind)` after each Binder run to
    capture the induced pattern-node subgraph observed at `s_t`. The
    Executor queries the same hash via `__call__` and receives the
    Subgraph whose `nodes` are the pattern nodes that participated in
    that step's binding, and whose `edges` are the induced edges among
    those nodes drawn from the concept graph's edge store.

    Bounded via FIFO eviction: the least-recently-recorded state is
    dropped when capacity is exceeded. Capacity default (10000) matches
    the Executor's MAX_EXPANSIONS, so any state the Executor could
    plausibly need is retained.
    """

    def __init__(
        self,
        concept_graph: Any,
        capacity: int = P.STATE_PATTERN_INDEX_CAPACITY,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.cg = concept_graph
        self.capacity = capacity
        self._index: Dict[int, Subgraph] = {}

    def record(self, s_t: int, D_bind: Set[Detection]) -> None:
        """Record the pattern-node Subgraph observed at state `s_t`.

        Member set is the union of pattern nodes referenced by D_bind.
        Edge set is the induced subgraph of those nodes in the concept
        graph's edge store, at the moment record() is called.

        Recording the same s_t twice replaces the previous entry
        (idempotent under the assumption that the same state hash is
        observed over the same pattern-node set — true whenever the
        pattern-node identity is (lens_type, signature_hash) as §1.11
        requires).
        """
        nodes: Set[int] = set()
        lookup = getattr(self.cg, "get_pattern_node_id", None)
        for d in D_bind:
            if callable(lookup):
                nid = lookup(d.k, _node_signature_hash(d.sigma))
            else:
                # Consolidated fallback: ConceptGraph's key→id index.
                # Flagged for concept_graph.py to expose the public
                # accessor (module-boundary rule §1.10.3).
                nid = self.cg._pattern_node_id_by_key.get(  # noqa: SLF001
                    (d.k, _node_signature_hash(d.sigma))
                )
            if nid is not None:
                nodes.add(nid)

        edges: Set[Tuple[int, int, EdgeType]] = set()
        for nid in nodes:
            for edge in self.cg.get_outgoing_edges(nid):
                if edge.child in nodes:
                    edges.add((edge.parent, edge.child, edge.edge_type))
            for edge in self.cg.get_incoming_edges(nid):
                if edge.parent in nodes:
                    edges.add((edge.parent, edge.child, edge.edge_type))

        self._index[s_t] = Subgraph(
            nodes=frozenset(nodes),
            edges=frozenset(edges),
            # [Pass 24] Populate edge_params from the concept graph's edge
            # store. Without this, _mcs_accumulate's graded edge matching
            # never fires: qp/cp are both None and param_sim falls through
            # to the ungraded default for every edge, so the graded path is
            # silently inert rather than raising.
            edge_params=edge_params_for(self.cg, edges),
            anchor_of_node=anchor_positions_for(
                nodes, self.cg.pattern_node_position
            ),
        )

        # FIFO eviction: drop oldest insertion-order entry.
        while len(self._index) > self.capacity:
            first_key = next(iter(self._index))
            del self._index[first_key]

    def __call__(self, s_t: int) -> Optional[Subgraph]:
        """Return the recorded Subgraph for `s_t`, or None if unrecorded."""
        return self._index.get(s_t)

    def windows_containing(self, nodes: Set[int], limit: int) -> List[Subgraph]:
        """Recorded situation windows whose node set overlaps `nodes`,
        in insertion (= temporal) order, capped at `limit`. Deterministic
        positive-example source for §5.3.2 discriminative consolidation
        (Trigger 2A)."""
        out: List[Subgraph] = []
        target = set(nodes)
        for sg in self._index.values():
            if sg.nodes & target:
                out.append(sg)
                if len(out) >= limit:
                    break
        return out

    def windows_without(self, nodes: Set[int], limit: int) -> List[Subgraph]:
        """Recorded situation windows with ZERO overlap with `nodes`, in
        insertion order, capped at `limit` — the deterministic stand-in
        for §5.8.2's "sample of windows where the signal did not appear"
        (randomness would violate §9 determinism)."""
        out: List[Subgraph] = []
        target = set(nodes)
        for sg in self._index.values():
            if not (sg.nodes & target):
                out.append(sg)
                if len(out) >= limit:
                    break
        return out

    def __len__(self) -> int:
        return len(self._index)

    def clear(self) -> None:
        """Drop all records. Called at episode boundaries if needed."""
        self._index.clear()


# =============================================================================
# Bundled provider container (convenience)
# =============================================================================


# =============================================================================
# Action-vocabulary decoding (B2 / §5.3.4)
# =============================================================================


def topo_sort_sequence_strict(
    nodes: Set[int], edges: List[Tuple[int, int]]
) -> Optional[List[int]]:
    """Deterministic Kahn topological sort over SEQUENCE edges.

    Ties break by ascending canonical node hash (the §5.3.1 canonical
    path order). Returns None on a cycle — an honest structural failure
    (K8): SEQUENCE edges are emitted with strictly positive temporal lag,
    so a cycle means the candidate merged incompatible episodes. The
    draft raised TemporalBindingError here; replay treats an unrealizable
    path as FAIL with a diagnostic counter, and an exception would abort
    the whole pipeline step over one corrupt candidate.
    """
    indeg: Dict[int, int] = {n: 0 for n in nodes}
    succ: Dict[int, List[int]] = {n: [] for n in nodes}
    for u, v in edges:
        if u in indeg and v in indeg:
            succ[u].append(v)
            indeg[v] += 1
    ready = sorted(
        (n for n in nodes if indeg[n] == 0),
        key=lambda nid: hash_128(serialize_int(nid)),
    )
    order: List[int] = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        newly: List[int] = []
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                newly.append(m)
        if newly:
            newly.sort(key=lambda nid: hash_128(serialize_int(nid)))
            ready.extend(newly)
            ready.sort(key=lambda nid: hash_128(serialize_int(nid)))
    if len(order) != len(nodes):
        return None  # cycle
    return order




def make_action_extractor(
    action_edge_map: Any,
    state_pattern_provider: Callable[[int], Optional[Subgraph]],
    action_repertoire: List[Any],
    concept_graph: Any = None,
) -> Callable[[Subgraph, int], Optional[List[Any]]]:
    """§5.3.4 step 1: extract the action sequence implied by a subgraph.

        actions(G) = ordered action descriptors determined by G's structure

    Resolution order (B2 root-cause fix):
      1. REGISTRY DECODE — when `concept_graph` is wired, Action-lens
         nodes in G's SEQUENCE topology decode to exact action objects
         (dictionary lookup; see decode_action_sequence). This is the
         primary route: it is exact, state-independent, and works for
         multi-action chains.
      2. AEM WALK (legacy fallback) — for concepts crystallized before
         the Action lens existed (no Action nodes): topological
         SEQUENCE walk realized against recorded dynamics (each step
         resolved to the max-mass action whose predicted successor's
         situation window contains the next node).
      3. None — neither route can realize the path; replay FAILs per
         §5.3.4, counted by the Consolidator's diagnostics.
    """

    def _extract(G: Subgraph, start_state: int) -> Optional[List[Any]]:
        if concept_graph is not None:
            decoded = decode_action_sequence(G, concept_graph)
            if decoded is not None:
                return decoded
        seq_edges = [
            (s, d) for (s, d, et) in G.edges if et == EdgeType.SEQUENCE
        ]
        if not seq_edges:
            return []

        # Topological order along SEQUENCE edges (Kahn, deterministic:
        # in-degree ties broken by ascending canonical node hash).
        nodes = sorted(
            {n for e in seq_edges for n in e},
            key=lambda nid: hash_128(serialize_int(nid)),
        )
        indeg: Dict[int, int] = {n: 0 for n in nodes}
        succ: Dict[int, List[int]] = {n: [] for n in nodes}
        for s, d in seq_edges:
            if s in indeg and d in indeg:
                succ[s].append(d)
                indeg[d] += 1
        order: List[int] = []
        ready = sorted(
            (n for n in nodes if indeg[n] == 0),
            key=lambda nid: hash_128(serialize_int(nid)),
        )
        while ready:
            n = ready.pop(0)
            order.append(n)
            for m in succ[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
            ready.sort(key=lambda nid: hash_128(serialize_int(nid)))
        if len(order) != len(nodes):
            return None  # cyclic SEQUENCE structure: not a replayable path

        actions: List[Any] = []
        state = start_state
        for i in range(len(order) - 1):
            target_node = order[i + 1]
            best: Optional[Tuple[float, bytes, Any]] = None
            for a in action_repertoire:
                outcomes = action_edge_map.get_outcomes(state, a)
                if not outcomes:
                    continue
                total_mass = float(sum(outcomes.values()))
                predicted = action_edge_map.predict(state, a)
                if predicted is None:
                    continue
                pattern = state_pattern_provider(predicted)
                if pattern is None or target_node not in pattern.nodes:
                    continue
                key = (-total_mass, serialize_value(a))
                if best is None or key < (best[0], best[1]):
                    best = (-total_mass, serialize_value(a), a)
            if best is None:
                return None  # step unrealizable in recorded dynamics
            actions.append(best[2])
            state = action_edge_map.predict(state, best[2])
        return actions

    return _extract


class PipelineProviders:
    """Groups the provider callables for pipeline construction.

    Use:
        providers = PipelineProviders(concept_graph)
        # During step, after Binder runs:
        providers.state_index.record(s_t, D_bind)
        # When constructing Mapper/Executor:
        executor = Executor(
            concept_graph,
            providers.signature_provider,
            providers.state_pattern_provider,
            is_monotonic_fn=env.is_monotonic,
        )
        # When wiring the Consolidator's replay (§5.3.4):
        extractor = make_action_extractor(
            env.action_edge_map,
            providers.state_pattern_provider,
            action_repertoire,
        )
    """

    def __init__(
        self,
        concept_graph: Any,
        capacity: int = P.STATE_PATTERN_INDEX_CAPACITY,
    ) -> None:
        self.cg = concept_graph
        self.state_index = StatePatternIndex(concept_graph, capacity=capacity)
        self.signature_provider = make_signature_provider(concept_graph)
        # Expose the index's __call__ directly for binding into the Executor.
        self.state_pattern_provider = self.state_index.__call__


# =============================================================================
# OBSERVATION WINDOW BUFFER (merged from observation_window_buffer.py)
# =============================================================================

class ConfigurationError(ValueError):
    """Raised on invalid construction or missing required wiring."""


_WindowKey = Tuple[frozenset, frozenset]


def _window_content_key(window: Subgraph) -> _WindowKey:
    """Hashable content identity of a Subgraph (dataclass is unhashable)."""
    return (frozenset(window.nodes), frozenset(window.edges))


class ObservationWindowBuffer:
    """Bounded, thread-safe, per-label positive/negative window store."""

    def __init__(
        self,
        capacity_per_label: int,
        min_instances: int,
        max_age_steps: int,
    ) -> None:
        if capacity_per_label <= 0:
            raise ConfigurationError("capacity_per_label must be > 0")
        if min_instances <= 0:
            raise ConfigurationError("min_instances must be > 0")
        if min_instances > capacity_per_label:
            raise ConfigurationError(
                "min_instances must be <= capacity_per_label (the trigger "
                "could never fire otherwise)"
            )
        if max_age_steps <= 0:
            raise ConfigurationError("max_age_steps must be > 0")
        self._capacity = int(capacity_per_label)
        self._min_instances = int(min_instances)
        self._max_age = int(max_age_steps)
        self._lock = threading.RLock()
        # label -> deque of (t, window)
        self._pos: Dict[Any, Deque[Tuple[int, Subgraph]]] = {}
        self._neg: Dict[Any, Deque[Tuple[int, Subgraph]]] = {}
        # label -> content keys, parallel to the deques (positives) /
        # set mirror (negatives) for O(capacity) dedup.
        self._pos_keys: Dict[Any, Deque[_WindowKey]] = {}
        self._neg_keys: Dict[Any, set] = {}
        # label -> deque parallel to _pos: entry i holds the structural
        # DISTANCES from window i to each window inserted BEFORE it
        # (insertion order). Sum over all entries = Σ_{i<j} d(wᵢ, wⱼ).
        self._pos_pair_lists: Dict[Any, Deque[List[float]]] = {}
        # label -> [Σ distances, pair count]
        self._pos_pairs: Dict[Any, List[float]] = {}
        # Mutation counter per label (consistency check for the
        # compute-outside-lock protocol).
        self._version: Dict[Any, int] = {}
        self._sig_provider: Optional[Callable[[int], Any]] = None
        # Diagnostics (observable, fixed literal keys only).
        self.diagnostics: Dict[str, int] = {
            "recorded_positives": 0,
            "recorded_negatives": 0,
            "duplicate_windows_skipped": 0,
            "stale_windows_evicted": 0,
            "labels_ready": 0,
            "labels_popped": 0,
            "version_conflicts": 0,
        }

    # -------------------------------------------------------------------------
    # Wiring
    # -------------------------------------------------------------------------

    def attach_sig_provider(self, provider: Callable[[int], Any]) -> None:
        """Attach the (nid) -> (lens_type, TypedSignature) provider used by
        subgraph_structural_similarity. Required before diversity gates."""
        self._sig_provider = provider

    # -------------------------------------------------------------------------
    # Recording
    # -------------------------------------------------------------------------

    def record(
        self, label: Any, polarity: int, window: Subgraph, t: int
    ) -> bool:
        """Buffer one labeled window. Returns True iff it was stored
        (False → content-duplicate of an already-buffered sibling or an
        empty window).

        Positive similarity work runs OUTSIDE the lock: snapshot the
        existing windows + version, compute, then re-acquire and verify
        the version before appending (single bounded retry; a second
        conflict stores with in-lock computation — correctness over
        latency, and conflicts are impossible in the single-threaded
        pipeline).
        """
        if window is None or not window.nodes:
            return False
        key = _window_content_key(window)
        if polarity > 0:
            return self._record_positive(label, key, window, t)
        with self._lock:
            dq = self._neg.get(label)
            if dq is None:
                dq = deque(maxlen=self._capacity)
                self._neg[label] = dq
                self._neg_keys[label] = set()
            keys = self._neg_keys[label]
            if key in keys:
                self.diagnostics["duplicate_windows_skipped"] += 1
                return False
            if len(dq) == self._capacity:
                keys.discard(_window_content_key(dq[0][1]))
            dq.append((t, window))
            keys.add(key)
            self.diagnostics["recorded_negatives"] += 1
            return True

    def _record_positive(
        self, label: Any, key: _WindowKey, window: Subgraph, t: int
    ) -> bool:
        for _attempt in range(2):
            with self._lock:
                self._ensure_label(label)
                dq = self._pos[label]
                if key in set(self._pos_keys[label]):
                    self.diagnostics["duplicate_windows_skipped"] += 1
                    return False
                existing = [w for _tt, w in dq]
                version = self._version[label]
            # Heavy work OUTSIDE the lock.
            dists = self._distances_to(window, existing)
            with self._lock:
                if self._version.get(label, 0) != version:
                    self.diagnostics["version_conflicts"] += 1
                    continue  # bounded retry
                self._append_positive(label, key, window, t, dists)
                self.diagnostics["recorded_positives"] += 1
                return True
        # Second conflict (concurrent mutation): store with in-lock math.
        with self._lock:
            self._ensure_label(label)
            existing = [w for _tt, w in self._pos[label]]
            dists = self._distances_to(window, existing)
            self._append_positive(label, key, window, t, dists)
            self.diagnostics["recorded_positives"] += 1
            return True

    def _ensure_label(self, label: Any) -> None:
        if label not in self._pos:
            self._pos[label] = deque(maxlen=self._capacity)
            self._pos_keys[label] = deque(maxlen=self._capacity)
            self._pos_pair_lists[label] = deque(maxlen=self._capacity)
            self._pos_pairs[label] = [0.0, 0.0]
            self._version.setdefault(label, 0)

    def _append_positive(
        self,
        label: Any,
        key: _WindowKey,
        window: Subgraph,
        t: int,
        dists: List[float],
    ) -> None:
        """Caller holds the lock. Appends and maintains aggregates;
        handles capacity eviction as O(n) list surgery."""
        dq = self._pos[label]
        keys_dq = self._pos_keys[label]
        pairs_dq = self._pos_pair_lists[label]
        pairs = self._pos_pairs[label]
        at_capacity = len(dq) == self._capacity
        # Capture the doomed list BEFORE the append (maxlen drops it).
        evicted_list = list(pairs_dq[0]) if at_capacity else None
        dq.append((t, window))
        keys_dq.append(key)
        pairs_dq.append(list(dists))
        pairs[0] += float(sum(dists))
        pairs[1] += float(len(dists))
        if at_capacity:
            # (a) drop the evicted window's own predecessor distances;
            pairs[0] = max(0.0, pairs[0] - float(sum(evicted_list)))
            pairs[1] = max(0.0, pairs[1] - float(len(evicted_list)))
            # (b) drop the evicted window's distance from EVERY survivor's
            # predecessor list — it was the oldest window, hence each
            # survivor's first predecessor (including the new window's).
            for lst in pairs_dq:
                if lst:
                    d0 = lst.pop(0)
                    pairs[0] = max(0.0, pairs[0] - d0)
                    pairs[1] = max(0.0, pairs[1] - 1.0)
        self._version[label] = self._version.get(label, 0) + 1

    def _distances_to(self, window: Subgraph, existing: List[Subgraph]) -> List[float]:
        """Structural distances from each existing window (insertion
        order) to `window`. Requires the signature provider (fail-loud)."""
        if not existing:
            return []
        if self._sig_provider is None:
            raise ConfigurationError(
                "ObservationWindowBuffer positive recording requires "
                "attach_sig_provider() before diversity can be computed"
            )
        out: List[float] = []
        for w in existing:
            sim = subgraph_structural_similarity(w, window, self._sig_provider)
            out.append(1.0 - float(sim))
        return out

    # -------------------------------------------------------------------------
    # Stale eviction (K2)
    # -------------------------------------------------------------------------

    def evict_stale(self, current_t: int) -> None:
        with self._lock:
            self._evict_bucket(self._pos, current_t, positive=True)
            self._evict_bucket(self._neg, current_t, positive=False)

    def _evict_bucket(
        self,
        bucket: Dict[Any, Deque[Tuple[int, Subgraph]]],
        current_t: int,
        positive: bool,
    ) -> None:
        # [DETERMINISM §9] repr() as a sort key is forbidden by the
        # canonical-ordering directive (process-dependent for any custom
        # label object). serialize_value yields canonical bytes for any
        # label type; eviction itself is per-label independent, so this
        # key only fixes the diagnostic/iteration order.
        for label in sorted(bucket.keys(), key=lambda k: serialize_value(k)):
            dq = bucket[label]
            changed = False
            while dq and current_t - dq[0][0] > self._max_age:
                dq.popleft()
                changed = True
                self.diagnostics["stale_windows_evicted"] += 1
                if positive:
                    keys_dq = self._pos_keys.get(label)
                    if keys_dq:
                        keys_dq.popleft()
                    pairs_dq = self._pos_pair_lists.get(label)
                    pairs = self._pos_pairs.get(label)
                    if pairs_dq and pairs is not None:
                        evicted_dists = pairs_dq.popleft()
                        pairs[0] = max(0.0, pairs[0] - float(sum(evicted_dists)))
                        pairs[1] = max(0.0, pairs[1] - float(len(evicted_dists)))
                        # The evicted window was the oldest, so it is the
                        # first predecessor entry of every survivor list.
                        for lst in pairs_dq:
                            if lst:
                                d0 = lst.pop(0)
                                pairs[0] = max(0.0, pairs[0] - d0)
                                pairs[1] = max(0.0, pairs[1] - 1.0)
                else:
                    keys = self._neg_keys.get(label)
                    if keys is not None:
                        # Rebuild the key mirror from the surviving deque —
                        # cheap (≤ capacity) and immune to drift.
                        keys.clear()
                        for _tt, w in dq:
                            keys.add(_window_content_key(w))
            if changed and positive:
                self._version[label] = self._version.get(label, 0) + 1
            if not dq:
                del bucket[label]
                if positive:
                    self._pos_keys.pop(label, None)
                    self._pos_pair_lists.pop(label, None)
                    self._pos_pairs.pop(label, None)
                else:
                    self._neg_keys.pop(label, None)

    # -------------------------------------------------------------------------
    # Trigger predicate + drain
    # -------------------------------------------------------------------------

    def ready(self, label: Any) -> bool:
        """O(1) trigger check — the pairwise-diversity mean is maintained
        incrementally by record()/eviction, never recomputed here."""
        with self._lock:
            pos = self._pos.get(label)
            neg = self._neg.get(label)
            if pos is None or len(pos) < self._min_instances:
                return False
            if neg is None or len(neg) < 1:
                return False
            # [Batch D3] The diversity gate is gone: readiness is purely
            # (distinct positives >= theta_instances) AND (>= 1 negative).
            # Diversity is consumed downstream as a soft reliability
            # factor, not as a veto -- see Pipeline._extract_category_core.
            self.diagnostics["labels_ready"] += 1
            return True

    def diversity(self, label: Any) -> float:
        """Current mean pairwise distance of the label's positives (0.0 if
        fewer than two are buffered). Diagnostic accessor."""
        with self._lock:
            pairs = self._pos_pairs.get(label)
            if pairs is None or pairs[1] <= 0.0:
                return 0.0
            return pairs[0] / pairs[1]

    def counts(self, label: Any) -> Tuple[int, int]:
        with self._lock:
            return (
                len(self._pos.get(label, ())),
                len(self._neg.get(label, ())),
            )

    def labels(self) -> List[Any]:
        with self._lock:
            return sorted(
                set(self._pos.keys()) | set(self._neg.keys()),
                key=lambda k: serialize_value(k),
            )

    def pop_windows(
        self, label: Any
    ) -> Tuple[List[Subgraph], List[Subgraph]]:
        with self._lock:
            pos = [w for _t, w in self._pos.get(label, deque())]
            neg = [w for _t, w in self._neg.get(label, deque())]
            self._pos.pop(label, None)
            self._neg.pop(label, None)
            self._pos_keys.pop(label, None)
            self._neg_keys.pop(label, None)
            self._pos_pair_lists.pop(label, None)
            self._pos_pairs.pop(label, None)
            self._version.pop(label, None)
            if pos or neg:
                self.diagnostics["labels_popped"] += 1
            return pos, neg

    def clear(self) -> None:
        with self._lock:
            self._pos.clear()
            self._neg.clear()
            self._pos_keys.clear()
            self._neg_keys.clear()
            self._pos_pair_lists.clear()
            self._pos_pairs.clear()
            self._version.clear()
