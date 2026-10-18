"""
Module 8: UtilityEngine — v31 integration.

Implements §2.20 Action Utility Function: Proposal Construction,
Utility Computation, Action Selection.

v31 changes vs. v30:
  - `planning_engine` is no longer a dependency. Its roles are split:
      * plan-driven proposals          → Mapper + Executor (A_t^transfer)
      * recursive distance             → §2.25 DFS over parent_ids with the
                                         enabling_pre_states BFS rule for
                                         parentless Layer-1 concepts
      * anti-undo guard                → prediction.gate_undo (§8.3.4/§8.11
                                         step 8d) AND effect presence via
                                         the architectural effect-marker
                                         interface — verified causality,
                                         not env-specific name checks
  - `_resolve_goal_action` no longer consults planning.

The §2.20.3 utility formula is a CLOSED FORM and is implemented verbatim:

    U_raw = U_cap·[Σ ŵᵢ·Normᵢ + ŵ_persist·PlanPersist] − η_conflict·Conflict
    U_t   = clamp(U_raw + σ_noise·τ_break(a), 0, U_cap)

The former "§2.20.3 formula coefficients" (reversal penalty, stay
discount, plan bonus, orient gain, destination richness) and the
"adaptive inertia" mechanism had no blueprint basis — utility terms may
only come from the eight §1.2 sources — and were removed as architectural
drift. Reflex-orient proposals remain (bounded motor expansion) and
compete through the standard terms via their bound target concept.

[REJECTED AUDIT FIX] An earlier audit proposed returning inf for
active-but-parentless non-atomic concepts. This is incorrect: active
concepts are pre-filtered from the goal subset, so the RD=0 branch is
only reached as an intermediate, where "active ⇒ realized" is correct.
Rejecting the fix preserves the correct RD(P) = 1 for a child of an
active parent.
"""

from __future__ import annotations

import math
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
from collections import defaultdict

from concept_graph import ConceptStatus, ConceptFamily
from sensory_floor import resolve_anchor_position

import parameters as P
from core.hashing import (  # noqa: F401
    clamp,
    hash_64,
    serialize_int,
    serialize_value,
)
from core.registries import (  # noqa: F401
    ACTION_NULL,
    is_noop_action,
    is_probe_action,
)

from consolidator import CoreKind, CoreLibrary  # noqa: F401 (CoreLibrary for typing)
from core.handoff import MacroExecutionOutcome


# =============================================================================
# CONSTANTS (§2.20 / §2.1.12 / §2.1.21)
# =============================================================================

# Every constant is sourced from parameters.py (single source of truth).
# The former inline literals (eta_layer=0.1, eta_conflict=5.0, ...) were a
# parallel definition of the v30 §2.1.12 table; the former "§23.1 formula
# coefficients" (reversal/plan-bonus/orient/dest-richness/inertia) had
# no blueprint basis at all and were deleted from parameters.py as drift.
# Exception (auditor ruling R5 §2.4): the standard v30 stay discount is
# reinstated — see P.U_STAY_DISCOUNT.
CONST = {
    "eps": P.EPSILON,
    "U_cap": P.U_CAP,
    "stay_discount": P.U_STAY_DISCOUNT,
    "Omega_probe": P.OMEGA_PROBE,
    "eta_layer": P.ETA_LAYER,
    "M_layer_max": P.M_LAYER_MAX,
    "theta_abstract": P.THETA_ABSTRACT,
    "eta_planPersist": P.ETA_PLAN_PERSIST,
    "eta_hysteresis": P.ETA_HYSTERESIS,
    "eta_fatigue": P.ETA_FATIGUE,
    "eta_fatigueDecay": P.ETA_FATIGUE_DECAY,
    "tau_satiety": P.TAU_SATIETY,
    "tau_fail": P.TAU_FAIL,
    "tau_failMax": P.TAU_FAIL_MAX,
    "N_motorFailMax": P.N_MOTOR_FAIL_MAX,
    "sigma_noise": P.SIGMA_NOISE,
    "K_plan": P.K_PLAN,
    "tau_meaning": P.TAU_MEANING,
    "xi_attempt": P.XI_ATTEMPT,
    "xi_entropy": P.XI_ENTROPY,
    "C_planMax": P.C_PLAN_MAX,
    "eta_conflict": P.ETA_CONFLICT,
    "tau_provisional": P.TAU_PROVISIONAL,
    "theta_sim": P.THETA_SIM,
    "theta_exploration": P.THETA_EXPLORATION,
    "kappa_protect": P.KAPPA_PROTECT,
    "Phi_max": P.PHI_MAX,
    "N_edge": P.N_EDGE,
    "theta_dominance": P.THETA_DOMINANCE,
    "N_outcomeMax": P.N_OUTCOME_MAX,
    "RF_max": P.RF_MAX,
    "rho_spawn_star": P.RHO_SPAWN_STAR_DEFAULT,
    "kappa_ema": P.KAPPA_EMA,
    "lambda": P.LAMBDA_DIST,
    "w1": P.W_FERT,
    "w2": P.W_ABSTRACT,
    "w3": P.W_REAL,
    "w4": P.W_PROTECT,
    "w5": P.W_CONSOL,
    "w6": P.W_MOTOR,
    # §2.20.1 targeted anchor fallback: the maximum anchor distance at which a
    # goal concept's own anchor may be approached directly (locomotion bridge
    # beyond it). Sourced from P.ANCHOR_FALLBACK_MAX_DIST — no inline literal.
    "anchor_fallback_max_dist": P.ANCHOR_FALLBACK_MAX_DIST,
    # [Drive Remediation Phase 0] Frontier-gated goal proposal +
    # fertility-to-locomotion inheritance (Strategic Fixes 1 & 2).
    "theta_frontier":     P.THETA_FRONTIER,
    "frontier_cache_ttl": P.FRONTIER_CACHE_TTL,
    "frontier_horizon":   P.FRONTIER_HORIZON,
    "k_plan_scan":        P.K_PLAN_SCAN,
    "w_fert_inherit":     P.W_FERT_INHERIT,
    "fert_inherit_top_k": P.FERT_INHERIT_TOP_K,
    # §2.25 DFS depth guard for the parent_ids RD proxy. Derived, not
    # magic: a realization basis deeper than twice the max plan cost can
    # never yield an affordable plan (RealizationValue decays as
    # exp(−λ·RD) and RD > C_planMax is rejected at proposal time).
    "rd_visited_max": 2 * P.C_PLAN_MAX,
}

# NOTE: the former second `from primitives import ...` block and the local
# `ACTION_NULL = "null"` redefinition were removed — the primitives module
# is the single source for the canonical null-action identity (a hardcoded
# env action-name literal shadowing the import is exactly the
# object-name-semantics drift v30 §1.9 prohibits).


# =============================================================================
# UTILITY ENGINE
# =============================================================================


class UtilityEngine:
    def __init__(
        self,
        concept_graph: Any,
        fertility_engine: Any,
        action_edge_map: Any,
        structural_potential_map: Any,
        action_repertoire: List[Any],
        mapper: Optional[Any] = None,
        executor: Optional[Any] = None,
        prediction_module: Optional[Any] = None,
        core_library: Optional["CoreLibrary"] = None,
        rho_spawn_star: float = 0.0,
        innate_reflex: Optional[Any] = None,
        env: Optional[Any] = None,
        activation_engine: Optional[Any] = None,
        ablations: FrozenSet[str] = frozenset(),
    ) -> None:
        self.graph = concept_graph
        self.fertility = fertility_engine
        self.action_edge = action_edge_map
        self.phi = structural_potential_map
        self.action_rep = list(action_repertoire)
        self.mapper = mapper
        self.executor = executor
        self.prediction = prediction_module
        self.core_library = core_library
        self.rho_spawn_star = rho_spawn_star
        self.innate_reflex = innate_reflex
        self.env = env
        self.act = activation_engine
        # [AUDIT SEAM] Diagnostic ablation switches — the ONLY sanctioned
        # way to neutralize a drive for the fertility-vs-noise audit
        # (regime C). "fertility" zeros every concept-fertility-derived
        # utility term (FertNorm, AbstractNorm, EF-ranked goal pool,
        # transfer fertility) while leaving structural potential (Φ),
        # RD, motor, and consolidation terms live, so the audit isolates
        # exactly the drive under question. Empty in production.
        self.ablations: FrozenSet[str] = frozenset(ablations)

        # [Seam 1] Macro-execution tracking: the transferred
        # ActionSequence currently being executed (if any) and the
        # completed/interrupted outcomes awaiting pipeline drain.
        self._macro_in_flight: Optional[Dict[str, Any]] = None
        self.macro_outcomes: List[MacroExecutionOutcome] = []

        # Observation cache set by the pipeline.
        self._observation: Optional[Any] = None

        # Orienting reflex target binding.
        self._orient_action: Optional[Any] = None
        self._orient_target_concept: Optional[int] = None

        # [SPEC INTERPRETATION §2.20.1 / §3.2] Transfer-driven plan state.
        self._active_transfer: Optional[Any] = None
        self._active_sequence: Optional[Any] = None
        self._transfer_costs: Dict[bytes, float] = {}

        # [SPEC GAP §2.25] Per-step RD cache, keyed by concept_id.
        self._rd_cache: Dict[int, float] = {}
        self._rd_cache_state: Optional[int] = None

        # Internal state (§2.20.2).
        self.fatigue: Dict[Any, float] = defaultdict(float)
        self.N_attempt: Dict[Any, int] = defaultdict(int)
        self.last_goal_success: Dict[int, int] = {}
        self._last_selected: Optional[Any] = None
        self.rho_spawn: float = 0.0
        self.refractory_until: Dict[Any, int] = {}
        self.consecutive_fail_count: Dict[Any, int] = defaultdict(int)

        # (The former "Adaptive Inertia" state — per-action homeostatic
        # thresholds, EMA, and the destination-entropy cache — implemented
        # utility terms with no §2.20.3 basis; removed with the terms.)

        # Target binding map populated by step_41, consumed by step_42.
        self._action_targets: Dict[Any, Any] = {}
        # [F2] Action -> (core_hash, t_predicted), per-step attribution.
        self._action_transfer_source: Dict[Any, Tuple[bytes, int]] = {}
        # [Batch D / P1-5] Endogenous goal suppression: cid -> topology
        # revision at suppression; revived after N_REVIVE_MIN revisions.
        self._suppressed_goals: Dict[int, int] = {}
        self._reach_cache: Dict[Tuple[Any, Any], float] = {}
        self._reach_cache_state: Optional[Any] = None
        # [F2] (core_hash, t_predicted) -> selection step.
        self._transfer_action_selected: Dict[Tuple[bytes, int], int] = {}

        # Per-step transfer fertility cache.
        self._transfer_fert_cache: Dict[bytes, float] = {}

        # Per-step AEM outcome cache. Keyed by (state, action).
        self._outcomes_cache: Dict[Tuple[Any, Any], Dict[Any, int]] = {}
        # [Phase 3] per-step memo for AEM geodesic distances used by the
        # fertility-to-locomotion inheritance bonus. Cleared in set_time.
        self._aem_dist_cache: Dict[Tuple[Any, FrozenSet[Any]], Optional[int]] = {}
        # [Patch L1c, STRICT variant] Frontier cache stores the TOPOLOGY
        # only: the set of states within FRONTIER_HORIZON AEM hops of the
        # start state. That set depends solely on AEM edges, so caching it
        # cannot change decisions; Phi is re-read fresh EVERY step over the
        # cached neighborhood, keeping _reaches_frontier bit-identical to
        # the uncached BFS (the original bool-memo design went stale
        # against per-step Phi decay and shifted the rooms solve step
        # 329 -> 436, violating the behavior-preserving mandate).
        # Generation tag = (AEM edge count, t // FRONTIER_CACHE_TTL); the
        # TTL term is a belt-and-braces refresh (no behavioral effect).
        self._frontier_cache: Dict[Any, FrozenSet[Any]] = {}
        self._frontier_cache_gen: Optional[Tuple[int, int]] = None
        self._goal_concepts_cache: Optional[List[Any]] = None
        self._goal_concepts_cache_t: int = -1

        # Per-step inverted index action → bound concept ids, built from
        # ConceptGraph.action_to_concepts (anti-undo and satiety run in
        # O(A + matches) instead of O(A × C) full-graph scans).
        self._action_concept_index: Dict[Any, List[Any]] = {}

        # Current step's state hash, set by set_time.
        self._current_state: Optional[int] = None

        self.t = 0
        # [P1-2 MEASUREMENT — Batch 2] Opt-in instrumentation for the
        # frontier-gate occupancy measurement
        # (diagnostics/_p1_2_measure.py). Default False: zero behavioral
        # effect, zero output. When True, step_41 appends (t, admitted
        # pool size, per-candidate frontier-reach margins) to p1_2_log;
        # no behavioral path reads or writes these two attributes.
        self.p1_2_measure = False
        self.p1_2_log: List[Tuple[int, int, List[float]]] = []
        # [Batch B B2] Unified θ initialization. set_theta_frontier() is the
        # only sanctioned way to pin θ; cold-start and reset_episode read from
        # _theta_frontier_initial. _r_bar_goal and _theta_frontier are seeded
        # so reset_episode and pre-step reads succeed unconditionally.
        self._theta_frontier_initial: float = P.THETA_FRONTIER
        self._theta_frontier_frozen: bool = False
        self._theta_frontier: float = P.THETA_FRONTIER
        self._r_bar_goal: float = 0.0
        # Counters for spec-mandated observability (e.g. the goal-satisfied
        # skip); unbounded keys are impossible — fixed literal names only.
        self.diagnostics: Dict[str, int] = defaultdict(int)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def set_time(self, t: int, current_state: Optional[int] = None) -> None:
        self.t = t
        if current_state is not None:
            self._current_state = current_state
        self._rd_cache.clear()
        # [Batch D / P1-5 SC-2] reach cache is per-step: clear on advance.
        self._reach_cache.clear()
        self._reach_cache_state = None
        self._rd_cache_state = current_state
        self._transfer_fert_cache.clear()
        self._outcomes_cache.clear()
        self._aem_dist_cache.clear()

    def reset_episode(self) -> None:
        # An unfinished macro at the episode boundary is an interruption
        # (Seam 1) — recorded, not silently dropped.
        if self._macro_in_flight is not None:
            self._emit_macro_outcome(success=False)
        self.fatigue.clear()
        self.N_attempt.clear()
        self.refractory_until.clear()
        self.consecutive_fail_count.clear()
        self.last_goal_success.clear()
        self._last_selected = None
        self._active_transfer = None
        self._active_sequence = None
        self._transfer_costs.clear()
        self._rd_cache.clear()
        self._rd_cache_state = None
        self._transfer_fert_cache.clear()
        self._outcomes_cache.clear()
        self._action_targets.clear()
        self._action_concept_index.clear()
        self._action_transfer_source.clear()
        self._transfer_action_selected.clear()
        if hasattr(self, "_r_bar_goal"):
            self._r_bar_goal = 0.0
            self._theta_frontier = self._theta_frontier_initial

    def set_theta_frontier(self, value: float, frozen: bool = True) -> None:
        """Set the frontier gate threshold. If frozen=True, the value is
        pinned: cold-start init and reset_episode restore it, and the
        homeostasis update in step_41 does not modify it.

        _r_bar_goal is seeded (0.0) so the unfrozen homeostasis branch
        reads a defined value on the first step_41 after a manual set.
        """
        self._theta_frontier = value
        self._theta_frontier_initial = value
        self._theta_frontier_frozen = frozen
        if not hasattr(self, "_r_bar_goal"):
            self._r_bar_goal = 0.0

    # -------------------------------------------------------------------------
    # Observation accessors
    # -------------------------------------------------------------------------

    def _is_action_valid(self, action: Any, observation: Any = None) -> bool:
        """Environment interface gate (§2.2): environments that expose
        is_action_valid constrain the executable set. A raising
        implementation is an env defect and propagates (the former
        `except Exception: return False` silently converted env bugs into
        'action invalid' verdicts)."""
        if self.env is None or not hasattr(self.env, "is_action_valid"):
            return True
        return bool(self.env.is_action_valid(action, observation))

    def _get_outcomes(self, state: Any, action: Any) -> Dict[Any, int]:
        """Return {next_state: count} for (state, action).

        Canonical ActionEdgeMap interface (§2.22): get_outcomes(s, a).
        Both AEM implementations in the tree (environment.py's
        SimpleActionEdgeMap and pybullet_env.ActionEdgeMap) provide it —
        the pybullet map was missing the accessor and utility_engine
        compensated with an O(|E|) items() aggregation per query; that
        shape-sniffing fallback was removed together with the aggregation
        path once the interface was unified at the source.
        """
        cache_key = (state, action)
        cached = self._outcomes_cache.get(cache_key)
        if cached is not None:
            return cached
        result = dict(self.action_edge.get_outcomes(state, action))
        self._outcomes_cache[cache_key] = result
        return result

    def _effect_id_for_action(self, concept: Any, action: Any) -> Any:
        """Return the effect pattern-node id bound to `action` on this
        concept, or the scalar `effect_id` if the per-action map is not
        populated. Returns None if the concept has no binding for the
        action.

        NOTE: this value is a pattern node id, not an environment effect atom.
        It is retained as a fallback only; prefer `_action_effect_atom_for` below, which
        routes through the env API when available.
        """
        mapping = getattr(concept, "effect_ids_by_action", None)
        if mapping:
            bound = mapping.get(action)
            if bound is not None:
                return bound
        if getattr(concept, "action", None) == action:
            return getattr(concept, "effect_id", None)
        return None

    def _action_effect_atom_for(self, concept: Any, action: Any) -> Any:
        """Return the environment's effect atom corresponding to `action`.

        Prefers the env API. When the env provides the API but returns
        None for this action, that None is authoritative — the action has
        no environmental effect. Only when the env does not expose
        `get_action_effect_atom` at all does this fall back to the
        concept's stored pattern-node id, which is a different namespace
        and may coincidentally collide with an env atom index.
        """
        if self.env is not None and hasattr(self.env, "get_action_effect_atom"):
            return self.env.get_action_effect_atom(action)
        return self._effect_id_for_action(concept, action)

    def _has_effect_marker(self, effect_id: Any, t: int) -> bool:
        """Effect-presence oracle via the architectural environment
        interface (§2.2: environments expose effect markers). This is the
        PRESENCE half of the anti-undo guard; the CAUSAL half is
        prediction.gate_undo (§8.3.4). A raising env implementation is an
        env defect and propagates."""
        if effect_id is None:
            return False
        if self.env is not None and hasattr(self.env, "effect_marker_checker"):
            return bool(self.env.effect_marker_checker(effect_id, t))
        return False

    def _build_action_concept_index(self) -> None:
        """Per-step inverted index action → bound live concepts, sourced
        from ConceptGraph.action_to_concepts with a per-concept action-set
        reconciliation (multi-binding)."""
        self._action_concept_index = {}
        rev = getattr(self.graph, "action_to_concepts", None)
        if rev is not None:
            for a, cids in rev.items():
                live = []
                for cid in cids:
                    c = self.graph.concepts.get(cid)
                    if c is not None and c.status in (
                        ConceptStatus.CRYSTALLIZED,
                        ConceptStatus.STABLE,
                    ):
                        live.append(c)
                if live:
                    self._action_concept_index[a] = live
            return
        for c in self.graph.concepts.values():
            if c.status not in (ConceptStatus.CRYSTALLIZED, ConceptStatus.STABLE):
                continue
            c_actions = set(getattr(c, "actions", set()) or ())
            if getattr(c, "action", None) is not None:
                c_actions.add(c.action)
            for a in c_actions:
                self._action_concept_index.setdefault(a, []).append(c)

    def _gate_undo_verified(self, action: Any, effect: Any) -> bool:
        """§8.11 step 8d: the anti-undo causal criterion is
        prediction.gate_undo(action, effect) — verified counterfactual
        causality (H_nec ∧ N_cf ≥ N_cf_min ∧ ΔP̄ ≥ 0.40). Without a wired
        Prediction module no link is verified and the guard is inert
        (fail-open on unverified causality is the spec semantics:
        Gate_undo defaults to 0)."""
        if self.prediction is None:
            return False
        return bool(self.prediction.gate_undo(action, effect))

    def _effect_already_achieved(self, concept: Any, action: Any) -> bool:
        """Anti-undo for a specific (concept, action) binding: verified
        causal link AND effect already present."""
        effect_atom = self._action_effect_atom_for(concept, action)
        if effect_atom is None:
            return False
        # [Batch D1] Peak-gated: a tiered commit (THETA_CF_COMMIT) is not
        # exempt from this route's own evidence requirement — the binding
        # must have carried N_CF_MIN of peak opportunity at some point.
        if concept.effect_confidence.get(action, 0.0) < P.N_CF_MIN:
            return False
        if not self._gate_undo_verified(action, effect_atom):
            return False
        return self._has_effect_marker(effect_atom, self.t)

    def _action_effect_achieved(self, action: Any) -> bool:
        """Anti-undo across all concepts bound to `action` (multi-binding)."""
        for c in self._action_concept_index.get(action, ()):
            # [Batch D1] Peak gate per concept: see _action_binding_satisfied
            # (bound but never peak-validated => not an anti-undo candidate).
            if c.effect_confidence.get(action, 0.0) < P.N_CF_MIN:
                continue
            if self._effect_already_achieved(c, action):
                return True
        return False

    def _action_binding_satisfied(self, action: Any) -> bool:
        """GOAL-SATISFIED SKIP PARITY (auditor ruling R5-O1; motor subset).

        True when `action` carries a COMMITTED counterfactual-validated
        binding (concept.effect_ids_by_action is written exclusively on
        H_nec validation — Pipeline._commit_pending_bindings, §8.9) whose
        effect marker is PRESENT right now: re-executing the action would
        be redundant (idempotent effect) or destructive (the toggle that
        closes an open door).

        This is the durable complement to the live gate_undo anti-undo
        route (§8.11 step 8d, `_action_effect_achieved` above). Gate_undo
        re-demands N_cf = min(C_opp[a], C_opp_not[a]) ≥ N_cf_min EVERY
        step, and both counters decay at γ_cf = 0.99: for a sparse action
        (MultiRoom toggles fire ~26× per 2000 steps) N_cf collapses
        between bursts (measured 6.3 → 2.8 → 0.3 → 0.0), so the live
        guard flickers off exactly when the achieved effect is most
        exposed to motor-noise re-selection. The validated binding is the
        permanent record that the H_nec criterion (N_cf ≥ 5 ∧ ΔP̄ ≥ 0.40
        — D4, ratified) WAS met; validation does not expire from
        opportunity starvation. Biologically: extinction of a learned
        affordance requires disconfirmation, not mere absence of recent
        execution.

        Environment-agnostic by construction: consumes only the validated-
        binding registry and the architectural env effect-marker interface
        (§2.2) — no action names, no env branching. Same adjudicated
        criterion as the concept-goal subset's GOAL-SATISFIED SKIP; this
        method extends it to the motor subset, which was the remaining
        route by which noise could re-toggle an achieved effect.
        """
        for c in self._action_concept_index.get(action, ()):
            atom = c.effect_ids_by_action.get(action)
            if atom is None:
                continue
            # [Batch D1] Peak-gated. The commit was tiered
            # (THETA_CF_COMMIT), so the LIVE N_cf may sit below the
            # historical N_CF_MIN while the link is real; the peak
            # recorded while the binding was live is the durable proof. A
            # binding whose peak never reached N_CF_MIN is not a validated
            # link and must not exempt the action.
            if c.effect_confidence.get(action, 0.0) < P.N_CF_MIN:
                continue
            if self._has_effect_marker(atom, self.t):
                return True
        return False

    # -------------------------------------------------------------------------
    # Step 41: Proposal Generation (§2.20.1)
    # -------------------------------------------------------------------------

    def step_41_generate_proposals(
        self, current_state: Any, observation: Any = None
    ) -> Set[Any]:
        proposals: Set[Any] = set()
        if observation is not None:
            self._observation = observation
        self._current_state = current_state

        # Reset per-step caches.
        self._transfer_fert_cache.clear()
        self._outcomes_cache.clear()
        self._action_targets = {}
        self._action_transfer_source.clear()
        _stale_cutoff = self.t - 2 * int(P.W_VALIDATION)
        _stale_keys = [
            k for k in self._transfer_action_selected
            if k[1] < _stale_cutoff
        ]
        for _k in _stale_keys:
            del self._transfer_action_selected[_k]

        # [Batch D / P1-5] Revive suppressed goals on graph growth.
        self._maybe_revive_suppressed()

        # [Batch D / P1-6 MED-04] Observed repertoire: static env actions
        # plus every ATOMIC_ACTION concept the graph has crystallized
        # (runtime vocabulary extension reads from the graph, not a fixed
        # list). Seeded, so no first-observation double-count.
        observed_actions = list(self.action_rep)
        for _c in self.graph.concepts.values():
            if (
                _c.family == ConceptFamily.ATOMIC_ACTION
                and _c.action is not None
                and _c.action not in observed_actions
            ):
                observed_actions.append(_c.action)

        # --- 1. Probe subset (§2.20.1) ------------------------------------
        if self.rho_spawn < self.rho_spawn_star:
            for a in observed_actions:
                if not self._refractory(a) and self._is_action_valid(a, observation):
                    proposals.add(a)

        # --- 2. Transfer subset (v31 replacement for A_t^plan) ------------
        # [SPEC INTERPRETATION §2.20.1 / §3.2] A_t^transfer =
        # {first_action(ActionSequence)} if a transfer is active, ∅ otherwise.
        self._active_transfer = None
        self._active_sequence = None
        self._transfer_costs.clear()

        if (
            self.mapper is not None
            and self.executor is not None
            and self.core_library is not None
        ):
            # §3.2: Mapper feeds proposal generation; Executor sequences
            # actions for the transferred consequence. These are core v31
            # calls on wired modules — exceptions propagate (the former
            # try/except triple silently degraded the entire transfer path
            # to a no-op on any defect).
            situation = self.mapper.build_situation_subgraph(self.t)
            if situation.nodes:
                transfer = self.mapper.query_action(self.t, situation)
                if transfer is not None:
                    self._active_transfer = transfer
                    source_cid = getattr(transfer, "source_concept_id", None)
                    concept = self.graph.concepts.get(source_cid) if source_cid is not None else None
                    if concept is not None and concept.status in (ConceptStatus.TOMBSTONED, ConceptStatus.REJECTED):
                        concept = None
                    first_action: Optional[Any] = None
                    ladder_used = False
                    transfer_satisfied = False
                    if concept is not None:
                        bound_atoms = getattr(concept, "effect_ids_by_action", None) or {}
                        if bound_atoms and any(self._has_effect_marker(atom, self.t) for atom in bound_atoms.values()):
                            self.diagnostics["transfer_ladder_skipped_achieved"] += 1
                            transfer_satisfied = True
                        else:
                            first_action = self._resolve_goal_action(concept.id, current_state)
                            ladder_used = first_action is not None
                    if not transfer_satisfied and first_action is None:
                        seq = self.executor.find_actions(
                            self.t, transfer.consequence_subgraph, self.action_edge,
                            current_state, core_library=self.core_library,
                        )
                        if seq is not None and seq.actions:
                            self._active_sequence = seq
                            self._transfer_costs[transfer.core_hash] = float(seq.reported_cost)
                            first_action = seq.actions[0]
                            self.diagnostics["transfer_executor_fallback_used"] += 1
                    if first_action is not None and not self._refractory(first_action) and self._is_action_valid(first_action, observation):
                        proposals.add(first_action)
                        self.diagnostics["transfer_proposals_emitted"] += 1
                        if ladder_used and concept is not None:
                            self.diagnostics["transfer_ladder_used"] += 1
                            self._action_targets[first_action] = concept.id
                            if self.env is not None and hasattr(self.env, "get_action_effect_atom") and self.prediction is not None:
                                self.prediction.narrow_validation_atoms(transfer.core_hash, self.env.get_action_effect_atom(first_action))
                        else:
                            self._action_targets[first_action] = transfer.core_hash
                        self._action_transfer_source[first_action] = (transfer.core_hash, self.t)
        if self._active_transfer is not None:
            self.diagnostics["active_transfer_set"] += 1

        # --- 3. Goal subset (§2.20.1) -------------------------------------
        # A_goal = {a_P : P ∈ top_K_plan(V_t, EF_eff^proposal), RD < ∞,
        #           ¬GoalSatiety, FertNorm ≥ θ_abstract}
        # The top-K_plan POOL is selected by proposal-safe EF_combined
        # (dormant/soft-dormant/tombstoned concepts score 0 and cannot
        # consume pool slots — v30 §2.15.6), then the remaining filters
        # apply WITHIN the pool. The former walk-until-K-survivors over
        # the full graph replaced the spec's bounded pool with an
        # unbounded scan and let low-EF concepts enter the goal set.
        def _proposal_safe_ef(n: Any) -> float:
            if n.status not in (ConceptStatus.CRYSTALLIZED, ConceptStatus.STABLE):
                return -1.0
            if "fertility" in self.ablations:
                # [AUDIT SEAM] Regime C: the EF-ranked goal pool becomes
                # id-ordered (all live concepts tie at 0.0).
                return 0.0
            if getattr(n, "dormant", False) or getattr(n, "soft_dormant", False):
                return 0.0
            if getattr(n, "tombstoned", False):
                return 0.0
            return getattr(n, "EF_combined", 0.0)

        # Spatial discovery of bottlenecks (doorways) is the constitutional role of the
        # InnateReflex orienting beacon (Update 12), NOT the motor system. The motor
        # system is coordinate-free topological; it cannot follow a gradient to a state
        # that has never been observed. Division of labor:
        #   Reflex discovers the bottleneck -> AEM confirms the transition ->
        #   Phi/geodesic exploits it -> Concept Graph chains it.
        # (Option-B guard: NO Phi-gradient compass, NO 2-hop motor hacks —
        #  rejected as violations of the environment-agnostic foundation.)
        # [Phase 2 — Frontier-gated goal pool] Scan the top K_PLAN_SCAN
        # causal-family (EVENT_CAUSAL / TEMPORAL_CHAIN) concepts by
        # proposal-safe EF_combined; admit a concept only if pursuing it
        # can REACH frontier territory (Φ > θ_frontier within
        # FRONTIER_HORIZON AEM hops of one of its post/enabling states).
        # Structurally exhausted goals (every reachable state already
        # mapped) no longer consume pool slots — the cure for compulsive
        # re-visitation. Concepts without recorded post-states pass
        # (do not over-suppress). All existing per-node gates below
        # (activation, goal-satisfied, satiety, RD, θ_abstract) apply
        # to the gated pool unchanged.
        causal = [
            c for c in self.graph.concepts.values()
            if getattr(c, "family", None) in (
                ConceptFamily.EVENT_CAUSAL, ConceptFamily.TEMPORAL_CHAIN
            )
        ]
        causal.sort(key=lambda n: (-_proposal_safe_ef(n), serialize_value(n.id)))
        pool: List[Any] = []
        for c in causal[: int(CONST["k_plan_scan"])]:
            if _proposal_safe_ef(c) < 0.0:
                continue
            if not self._concept_leads_to_frontier(c):
                self.diagnostics["goal_skipped_no_frontier"] += 1
                continue
            pool.append(c)
            if len(pool) >= CONST["K_plan"]:
                break

        # [P1-2 MEASUREMENT HOOK — Batch 2; measurement-only] When the
        # opt-in flag is set, record for THIS step: (t, admitted pool
        # size, frontier-reach margin of every safe-EF candidate in the
        # scan window). The margins let diagnostics/_p1_2_measure.py
        # replay the gate at any hypothetical fixed theta offline
        # (pool_size(theta) = min(#margin > theta, K_plan)) without
        # touching behavior here. Flag defaults to False — production
        # and baseline runs pay one boolean test per step, emit nothing,
        # and their stdout stays byte-identical.
        if self.p1_2_measure:
            margins: List[float] = []
            for cand in causal[: int(CONST["k_plan_scan"])]:
                if _proposal_safe_ef(cand) < 0.0:
                    continue
                margins.append(self._p1_2_max_reach_phi(cand))
            self.p1_2_log.append((self.t, len(pool), margins))

        # [Batch 4 / P1-2] Frontier-gate homeostasis (Option B).
        # Cold start: _r_bar_goal at 0.0 (matches measured reality).
        if not hasattr(self, "_r_bar_goal"):
            self._r_bar_goal = 0.0
            self._theta_frontier = self._theta_frontier_initial
        scanned = max(1, int(CONST["k_plan_scan"]))
        r_goal = len(pool) / scanned
        if not self._theta_frontier_frozen:
            self._r_bar_goal = (
                (1.0 - P.KAPPA_EMA_FRONTIER) * self._r_bar_goal
                + P.KAPPA_EMA_FRONTIER * r_goal
            )
            self._theta_frontier = clamp(
                self._theta_frontier
                + P.KAPPA_FRONTIER * (self._r_bar_goal - P.RHO_GOAL_TARGET),
                P.THETA_FRONTIER_FLOOR,
                P.THETA_FRONTIER_MAX,
            )
        self.diagnostics["theta_frontier_current"] = self._theta_frontier
        self.diagnostics["r_bar_goal_current"] = self._r_bar_goal

        for node in pool:
            # [Batch D / P1-5] Structural goal-abandon: suppressed until the
            # graph grows; unreachable goals are skipped endogenously.
            if node.id in self._suppressed_goals:
                continue
            if self._has_no_path_to_goal(node.id, current_state):
                self._suppressed_goals[node.id] = getattr(
                    self.graph, "topology_revision", 0
                )
                self.diagnostics["goal_suppressed_no_path"] += 1
                continue
            if _proposal_safe_ef(node) < 0.0:
                continue
            if getattr(node, "dormant", False) or getattr(
                node, "soft_dormant", False
            ):
                continue
            if node.family == ConceptFamily.ATOMIC_ACTION:
                # Atomic actions reach the proposal set through the motor
                # subset (§2.20.1); listing them as concept goals would
                # double-count the same action under two targets.
                continue
            if (
                self.act is not None
                and self.act.get_activation(node.id, self.t) == 1
            ):
                # Already realized this step (see [REJECTED AUDIT FIX] in
                # the module header for the RD=0 intermediate semantics).
                continue
            P_id = node.id
            if self._goal_satiety(P_id):
                continue

            # GOAL-SATISFIED SKIP (architect ruling — locomotion gap):
            # when a concept's counterfactually VALIDATED effect (bindings
            # are written only on H_nec, see Pipeline._bind_causal_schemas)
            # is PRESENT right now, the goal is satisfied — re-proposing
            # its affordance would destroy the achieved effect (the toggle
            # that closes an open door), and the locomotion bridge would
            # loiter at a consumed goal forever. Complements GoalSatiety
            # (§2.20.2, τ_satiety=30) for persistent LEVEL effects; uses
            # only spec-native signals (env effect markers × validated
            # bindings), so it is environment-agnostic by construction.
            bound_effects = set(
                getattr(node, "effect_ids_by_action", {}).values()
            )
            if bound_effects and any(
                self._has_effect_marker(atom, self.t) for atom in bound_effects
            ):
                self.diagnostics["goal_skipped_satisfied"] += 1
                continue

            # §2.25 realization distance: parent_ids recursion + the
            # enabling_pre_states BFS rule for parentless Layer-1 concepts.
            rd = self._concept_realization_distance(P_id, current_state)
            if rd == math.inf or rd > CONST["C_planMax"]:
                continue
            if self._fert_norm(P_id) < CONST["theta_abstract"]:
                continue

            concrete_a = self._resolve_goal_action(P_id, current_state)
            if (
                concrete_a is not None
                and not self._refractory(concrete_a)
                and self._is_action_valid(concrete_a, observation)
            ):
                # Anti-undo guard (§8.3.4 / §8.11 step 8d): skip the action
                # only when Prediction has VERIFIED the action→effect link
                # (gate_undo) AND the bound effect is already present.
                # Verified causality is the architectural criterion — the
                # former marker-only check trusted unverified bindings and
                # the env-name-sniffing variant around it was env-specific
                # drift.
                if self._effect_already_achieved(node, concrete_a):
                    continue
                proposals.add(concrete_a)
                curr_target = self._action_targets.get(concrete_a)
                if curr_target is None or self._fert_norm(P_id) > self._fert_norm(
                    curr_target
                ):
                    self._action_targets[concrete_a] = P_id

        # --- 4. Frontier subset (§2.20.1) ---------------------------------
        # A_frontier = {a : ∃s′, M_{s_t,a} ≥ N_edge,
        #                    dominance_{s_t,a,s′} ≥ θ_dominance,
        #                    Φ_t(s′) ≥ θ_exploration}
        # Evaluated per repertoire action via the canonical AEM accessor —
        # O(|A_rep|) instead of the former O(|E|) full-map items() scan
        # every step.
        # Ruling Item 1 (exploration side): the frontier subset navigates
        # TENTATIVE transitions — the threshold is the AEM instance's own
        # N_edge (bootstrap environments declare 1). The permanence gate
        # (P.N_EDGE = 5, θ_dominance = 0.8) lives where cores are ENCODED:
        # the Consolidator's replay. Using the consolidation threshold
        # here starved frontier proposals in every sparse environment.
        n_edge_explore = getattr(self.action_edge, "N_edge", CONST["N_edge"])
        for a in observed_actions:
            if self._refractory(a) or not self._is_action_valid(a, observation):
                continue
            outcomes = self._get_outcomes(current_state, a)
            if not outcomes:
                continue
            mass = float(sum(outcomes.values()))
            if mass < n_edge_explore:
                continue
            for s_prime, m in outcomes.items():
                dominance = m / (mass + CONST["eps"])
                if (
                    dominance >= CONST["theta_dominance"]
                    and self.phi.get(s_prime, 0) >= CONST["theta_exploration"]
                ):
                    proposals.add(a)
                    break

        # --- 5. Motor subset (§2.20.1) + anti-undo ------------------------
        # A_motor = {a ∈ A_rep : ¬Refractory(a,t)} — the spec subset has no
        # probe exclusion (probe-ness is handled by the utility BRANCH in
        # step_42, not by the proposal set) and no inertness filter (the
        # former "adaptive inertia" hard-zero was an off-spec conditional
        # bypass; repeated unproductive actions are bounded by Refractory
        # escalation and the N_attempt term of MotorNorm, exactly as the
        # blueprint prescribes).
        self._build_action_concept_index()
        for a in observed_actions:
            if self._refractory(a) or not self._is_action_valid(a, observation):
                continue
            # Subcortical habituation gating is the reflex module's own
            # public interface (12_Update reflex layer); a raising
            # implementation is a defect and propagates (the former
            # `except Exception: pass` swallowed it).
            if self.innate_reflex is not None and hasattr(
                self.innate_reflex, "is_affordance_satiated"
            ):
                if self.innate_reflex.is_affordance_satiated(
                    a, self.t, observation
                ):
                    continue

            # Anti-undo guard (§8.3.4 / §8.11 step 8d): skip a when
            # Prediction has verified a→b causal (gate_undo) AND the bound
            # effect b is already present. Multi-binding is covered via the
            # per-step action→concepts index (O(A + matches), replacing the
            # former per-action full-graph concept scan).
            if self._action_effect_achieved(a):
                self.diagnostics["motor_skipped_anti_undo"] += 1
                continue
            # Goal-satisfied skip parity (auditor ruling R5-O1): the same
            # adjudicated criterion the concept-goal subset uses — committed
            # H_nec-validated binding × effect marker PRESENT — also exempts
            # the action here, so the durable route covers the sparse-action
            # windows where the live gate_undo's decayed N_cf flickers the
            # §8.11-8d guard off (see _action_binding_satisfied).
            if self._action_binding_satisfied(a):
                self.diagnostics["motor_skipped_satisfied"] += 1
                continue

            proposals.add(a)

        # --- 6. Orienting reflex subset (§2.20.1) -------------------------
        self._orient_action = None
        self._orient_target_concept = None
        if self.innate_reflex is not None:
            obs = observation if observation is not None else self._observation
            if hasattr(self.innate_reflex, "get_approach_action_with_target"):
                approach_data = self.innate_reflex.get_approach_action_with_target(
                    self.t, obs, reject_fn=self._refractory
                )
            else:
                approach_a = self.innate_reflex.get_approach_action(
                    obs, reject_fn=self._refractory
                )
                approach_data = (
                    (approach_a, None) if approach_a is not None else None
                )

            if approach_data is not None:
                approach_a, target_anchor = approach_data
                if self._is_action_valid(approach_a, observation):
                    proposals.add(approach_a)
                    self._orient_action = approach_a
                    if target_anchor is not None:
                        matching = [
                            c
                            for c in self.graph.concepts.values()
                            if getattr(c, "anchor", None) == target_anchor
                            and c.status
                            in (ConceptStatus.CRYSTALLIZED, ConceptStatus.STABLE)
                        ]
                        if matching:
                            def _tb(c: Any) -> int:
                                sig = getattr(c, "sigma", None)
                                if sig is not None and hasattr(sig, "serialize"):
                                    return hash_64(sig.serialize())
                                return hash_64(serialize_value(c.id))

                            matching.sort(
                                key=lambda c: (
                                    getattr(c, "layer", 0),
                                    getattr(c, "EF_combined", 0.0),
                                    _tb(c),
                                ),
                                reverse=True,
                            )
                            self._orient_target_concept = matching[0].id

        if not proposals:
            proposals.add(ACTION_NULL)
        return proposals

    # -------------------------------------------------------------------------
    # [SPEC GAP §2.25] Lightweight recursive realization distance
    # -------------------------------------------------------------------------

    def _concept_realization_distance(
        self,
        concept_id: int,
        current_state: int,
        _visited: Optional[Set[int]] = None,
    ) -> float:
        """[SPEC GAP §2.25] v31 removes planning_engine without nominating
        a replacement for RealizationBasis_t(P). This function combines
        v30's structural rules with the concept graph's parent_ids:

            RD(P, S_t) = 0                              if A_P(t) = 1
                       = 1                              if P is atomic and
                                                          executable from s_t
                       = 1 + min over parents(p) RD(p)  otherwise
                       = inf                            if no path exists

        [REJECTED AUDIT FIX] An earlier audit proposed returning inf for
        active-but-parentless non-atomic concepts. This is incorrect: no
        active concept ever becomes a goal (step_41 filters them), so the
        RD=0 branch is only reached as an intermediate node during parent
        traversal. There, "active ⇒ realized" is the correct definition —
        activation is the architectural signal for "this pattern is
        currently present." Rejecting the fix preserves RD(P) = 1 for a
        child of an active parent.
        """
        if self._rd_cache_state != current_state:
            self._rd_cache.clear()
            self._rd_cache_state = current_state

        if _visited is None:
            _visited = set()

        cached = self._rd_cache.get(concept_id)
        if cached is not None:
            return cached

        if concept_id in _visited:
            return math.inf
        if len(_visited) > CONST["rd_visited_max"]:
            return math.inf

        concept = self.graph.concepts.get(concept_id)
        if concept is None:
            self._rd_cache[concept_id] = math.inf
            return math.inf

        # [SELF-AUDIT SA1] Atomic actions: RD = 1 if executable from s_t.
        if concept.family == ConceptFamily.ATOMIC_ACTION:
            action = getattr(concept, "action", None)
            if action is not None and self._get_outcomes(current_state, action):
                self._rd_cache[concept_id] = 1.0
                return 1.0
            self._rd_cache[concept_id] = math.inf
            return math.inf

        # Active concept → realized (RD = 0).
        if getattr(concept, "A", 0) == 1:
            self._rd_cache[concept_id] = 0.0
            return 0.0

        # [LAYER 1 RD] Causal and action concepts have no concept parents
        # but do carry enabling_pre_states — the set of state hashes from
        # which their action can be executed. Compute RD as the minimum
        # over those states of (BFS(current_state, enabling_state) + 1
        # action execution). Without this, every Layer 1 concept
        # evaluates to RD = inf and is permanently excluded from goal
        # proposals.
        eps = getattr(concept, "enabling_pre_states", None)
        if eps:
            best_state_dist = math.inf
            for s in eps:
                # shortest_path_length is part of the canonical
                # ActionEdgeMap interface (BFS over the action-edge graph
                # is the §1.9-mandated navigation primitive); a missing
                # implementation is a wiring defect and propagates (the
                # former except-(AttributeError, TypeError) fallback
                # silently returned inf and re-created the very Layer-1
                # RD bug this branch fixes).
                d = self.action_edge.shortest_path_length(
                    current_state, s, max_depth=int(CONST["C_planMax"])
                )
                if d < best_state_dist:
                    best_state_dist = d
            if math.isfinite(best_state_dist):
                # +1 action execution — for ANY action binding (scalar
                # `action` or multi-bound `actions`); the former scalar-
                # only check undercounted multi-bound schemas by 1.
                has_action = getattr(concept, "action", None) is not None or bool(
                    getattr(concept, "actions", None)
                )
                action_cost = 1.0 if has_action else 0.0
                result = min(
                    best_state_dist + action_cost,
                    float(CONST["C_planMax"]),
                )
                self._rd_cache[concept_id] = result
                return result

        _visited.add(concept_id)
        best = math.inf
        for pid in sorted(concept.parent_ids):
            sub = self._concept_realization_distance(
                pid, current_state, _visited
            )
            if sub < best:
                best = sub
        _visited.discard(concept_id)

        if best < math.inf:
            best = min(best + 1.0, float(CONST["C_planMax"]))

        self._rd_cache[concept_id] = best
        return best

    def _resolve_goal_action(
        self, concept_id: int, current_state: Any
    ) -> Optional[Any]:
        """Resolve a concept goal to a concrete executable action.
        Under Schema Multi-Binding, iterate valid actions deterministically.

        Resolution order (§2.20.1):

          1. Direct affordance — a terminal action of the concept valid now.
          2. Active transfer macro-sequence step.
          3. LOCOMOTION BRIDGE (architect ruling — "Executor/Action-Edge
             bridge for goals"): when the terminal affordance is NOT valid
             from the current state (the classic far-away goal: the switch
             schema is fertile but the agent stands three tiles away),
             resolve the next movement step toward the concept's enabling
             state via BFS over the action-edge graph (§1.9's mandated
             navigation primitive) and return it.
          4. TARGETED ANCHOR FALLBACK with distance guard (Correction 1):
             when no AEM path exists (the action graph has not recorded the
             needed transitions yet) but the concept declares an anchor
             within ANCHOR_FALLBACK_MAX_DIST of the agent, ask the reflex for
             one step of approach to *that* anchor through the public
             get_approach_action_for_anchor API. The distance guard is what
             keeps this a *targeted* fallback: a distant anchor is a
             navigation problem (route it through the bridge), not an
             affordance that can be executed from here.

        step_41 binds target(action) = P for the returned action, so a bridged
        or fallen-back movement inherits w1·FertNorm(P) and
        w3·RealizationValue = exp(−λ·RD(P, S_t)) through the CLOSED-FORM
        §2.20.3 utility — a continuous fertility gradient pulling the agent
        toward fertile goals. No new utility terms are introduced; these
        branches only supply the target binding the spec's own goal subset
        requires.
        """
        concept = self.graph.concepts.get(concept_id)
        if concept is None:
            return None

        # 1. Multi-binding path: a terminal affordance valid right now.
        actions_set = getattr(concept, "actions", None)
        if actions_set:
            # Canonical serialization for the deterministic iteration
            # order (str() on non-str actions is type-dependent and, for
            # plain objects, address-poisoned).
            for action in sorted(actions_set, key=lambda a: serialize_value(a)):
                if self._is_action_valid(action, self._observation):
                    return action

        # 1b. Scalar affordance fallback.
        own_action = getattr(concept, "action", None)
        if own_action is not None:
            if self._is_action_valid(own_action, self._observation):
                return own_action

        # 2. Active transfer macro-step.
        if (
            self._active_transfer is not None
            and self._active_sequence is not None
            and self._active_sequence.actions
        ):
            consequence = getattr(
                self._active_transfer, "consequence_subgraph", None
            )
            if consequence is not None and (consequence.nodes & concept.members):
                return self._active_sequence.actions[0]

        # 3. Locomotion bridge: first valid step of the action-edge path
        # to the nearest recorded enabling state. Deterministic: enabling
        # states in canonical-serialization order; the AEM's BFS path is
        # deterministic; each path step must be currently valid and non-
        # refractory (walls/closed doors), else the next enabling state is
        # tried. Depth-bounded by C_planMax — the same horizon the RD
        # proxy uses, so a bridged action always has finite
        # RealizationValue (no proposals the utility cannot score).
        if current_state is not None:
            eps = getattr(concept, "enabling_pre_states", None)
            bfs = getattr(self.action_edge, "bfs_path", None)
            if eps and callable(bfs):
                for s_enable in sorted(eps, key=lambda s: serialize_int(int(s))):
                    try:
                        path = bfs(
                            current_state,
                            int(s_enable),
                            max_depth=int(CONST["C_planMax"]),
                        )
                    except Exception:
                        path = None
                    if not path:
                        continue
                    for step_action in path:
                        if self._is_action_valid(
                            step_action, self._observation
                        ) and not self._refractory(step_action):
                            self.diagnostics["goal_bridge_used"] += 1
                            return step_action
                self.diagnostics["goal_bridge_exhausted"] += 1

        # 4. Targeted anchor fallback with distance guard (Correction 1).
        # Only reached when no terminal affordance is valid and no AEM path
        # exists: the concept's own anchor is close enough to walk to in one
        # step, so the reflex supplies the direction. The returned action must
        # still pass the environment validity gate; the reflex is asked
        # through its public API (never a private direction extractor) and is
        # handed the refractory predicate so a saturated direction is not
        # re-proposed.
        #
        # [Seam 2 — flag/atom dual grounding] A causal concept's primary
        # anchor is frequently its EFFECT atom — often an unpositioned flag
        # (symbolic envs park flags at the (-1,-1) placeholder). The former
        # single-anchor probe then either skipped the fallback entirely or,
        # worse, computed a spurious approach vector toward (-1,-1).
        # Concepts now carry BOTH groundings (anchor + trigger_anchor, the
        # positioned entity written at binding-commit time); this branch
        # tries each in order and approaches the first with a REAL position.
        # [Phase 4] Reflex fallback ONLY for physically-grounded sensory
        # goals. Abstract causal concepts navigate via the AEM geodesic
        # (steps 1–3 above) and fertility inheritance (Phase 3) — the
        # reflex is decoupled from conceptual locomotion.
        if self.innate_reflex is not None and not self._is_abstract_goal(concept):
            obs = self._observation
            if obs is not None:
                agent_pos = self._anchor_position(
                    getattr(self.innate_reflex, "agent_anchor", 0), obs
                )
                if agent_pos is not None:
                    seen_anchors: Set[Any] = set()
                    for anchor in (
                        getattr(concept, "anchor", None),
                        getattr(concept, "trigger_anchor", None),
                    ):
                        if anchor is None or anchor in seen_anchors:
                            continue
                        seen_anchors.add(anchor)
                        target_pos = self._anchor_position(anchor, obs)
                        if target_pos is None:
                            continue
                        dist = math.sqrt(
                            sum(
                                (a - b) ** 2
                                for a, b in zip(agent_pos, target_pos)
                            )
                        )
                        if dist > CONST["anchor_fallback_max_dist"]:
                            continue
                        direction = None
                        try:
                            direction = (
                                self.innate_reflex.get_approach_action_for_anchor(
                                    obs, anchor, reject_fn=self._refractory
                                )
                            )
                        except (AttributeError, KeyError, TypeError, IndexError):
                            self.diagnostics["anchor_fallback_reflex_errors"] += 1
                            direction = None
                        if direction and self._is_action_valid(direction, obs):
                            self.diagnostics["anchor_fallback_used"] += 1
                            return direction

        return None

    def _anchor_position(
        self, anchor: Any, obs: Any
    ) -> Optional[Tuple[float, float]]:
        """[Seam 2] Real 2D position of an anchor in the current
        observation, or None. Delegates to the single source of truth in
        sensory_floor.resolve_anchor_position (shared with the Pipeline's
        trigger-anchor grounding): dict or sequence position conventions,
        (-1, -1) flag placeholders rejected — flag atoms are skipped,
        never approached."""
        return resolve_anchor_position(obs, anchor)

    # -------------------------------------------------------------------------
    # [Phase 2] Frontier-gated goal proposal — helpers
    # -------------------------------------------------------------------------

    def _phi_at(self, state: Any) -> float:
        """Structural potential Φ(state). The engine holds the map as
        `self.phi` (ctor arg `structural_potential_map`); both names are
        probed so the helper is wiring-agnostic."""
        spm = getattr(self, "phi", None)
        if spm is None:
            spm = getattr(self, "structural_potential_map", None)
        if spm is None:
            return 0.0
        return float(spm.get(state, 0.0))

    def _has_no_path_to_goal(self, cid, current_state):
        """[Batch D / P1-5 HIGH-02] True when the AEM has no path (horizon
        C_PLAN_MAX, matching the RD gate) from the current state to ANY of
        the concept's enabling pre-states. Per-step memoized. Concepts
        without recorded pre-states pass (do not over-suppress)."""
        if self._reach_cache_state != current_state:
            self._reach_cache.clear()
            self._reach_cache_state = current_state
        concept = self.graph.concepts.get(cid)
        if concept is None:
            return True
        eps = getattr(concept, "enabling_pre_states", None) or set()
        if not eps:
            return False
        for s_pre in eps:
            key = (current_state, s_pre)
            if key not in self._reach_cache:
                self._reach_cache[key] = self.action_edge.shortest_path_length(
                    current_state, s_pre, max_depth=int(P.C_PLAN_MAX)
                )
            if math.isfinite(self._reach_cache[key]):
                return False
        return True

    def _maybe_revive_suppressed(self):
        """[Batch D / P1-5] Goals re-wake when the graph grows
        (N_REVIVE_MIN topology revisions since suppression), not on a
        timer — the endogenous signal the P1-5 design demands."""
        if not self._suppressed_goals:
            return
        rev = getattr(self.graph, "topology_revision", 0)
        revived = [
            cid
            for cid, rev_at in sorted(self._suppressed_goals.items())
            if rev - rev_at >= P.N_REVIVE_MIN
        ]
        for cid in revived:
            self._suppressed_goals.pop(cid, None)
            self.diagnostics["goal_revived"] += 1

    def _aem_neighbors(self, state: Any) -> List[Any]:
        """AEM one-hop successors of `state`.

        [Phase 2 adaptation] SimpleActionEdgeMap exposes neither
        `successors` nor `get_outgoing`; its canonical API is the
        per-(state, action) outcome dict. The union of recorded outcomes
        over the action repertoire is the TENTATIVE transition graph —
        the same exploration-side convention as the §2.20.1 frontier
        subset (Ruling Item 1): permanence gating lives at consolidation,
        not at exploration."""
        ae = getattr(self, "action_edge", None)
        if ae is None:
            return []
        if hasattr(ae, "successors"):
            return list(ae.successors(state))
        if hasattr(ae, "get_outgoing"):
            return [e.child for e in ae.get_outgoing(state)]
        nbrs: List[Any] = []
        seen: Set[Any] = set()
        for a in self.action_rep:
            for ns in self._get_outcomes(state, a):
                if ns not in seen:
                    seen.add(ns)
                    nbrs.append(ns)
        return nbrs

    def _aem_edge_count(self) -> int:
        """[Patch L1d] Recorded-transition count of the action-edge map.
        SimpleActionEdgeMap exposes neither `num_edges` nor `edges`; its
        `mass` dict (one entry per observed (state, action) pair) is the
        faithful edge-population proxy — it grows exactly when the AEM
        learns a new transition."""
        ae = getattr(self, "action_edge", None)
        if ae is None:
            return 0
        if hasattr(ae, "num_edges"):
            return int(ae.num_edges)
        if hasattr(ae, "edges"):
            return len(ae.edges)
        mass = getattr(ae, "mass", None)
        if mass is not None:
            return len(mass)
        return 0

    def _frontier_neighborhood(self, start_state: Any) -> FrozenSet[Any]:
        """[Patch L1 strict] All states within FRONTIER_HORIZON AEM hops of
        `start_state` (start included). Topology-only: recomputed solely on
        AEM growth or TTL refresh, never on Phi changes."""
        gen = (self._aem_edge_count(), self.t // int(CONST["frontier_cache_ttl"]))
        if self._frontier_cache_gen != gen:
            self._frontier_cache.clear()
            self._frontier_cache_gen = gen
        cached = self._frontier_cache.get(start_state)
        if cached is not None:
            return cached
        visited = {start_state}
        frontier = [start_state]
        for _ in range(int(CONST["frontier_horizon"])):
            nxt: List[Any] = []
            for s in frontier:
                for nbr in self._aem_neighbors(s):
                    if nbr not in visited:
                        visited.add(nbr)
                        nxt.append(nbr)
            frontier = nxt
            if not frontier:
                break
        nb = frozenset(visited)
        self._frontier_cache[start_state] = nb
        return nb

    def _reaches_frontier(self, start_state: Any) -> bool:
        """Cached frontier reachability, bit-identical to the uncached BFS:
        the cached part is the hop-ball (topology); Phi is evaluated fresh
        this step over every state in it. Invalidates when the AEM gains
        edges or every FRONTIER_CACHE_TTL steps."""
        theta = getattr(self, "_theta_frontier", CONST["theta_frontier"])
        for s in self._frontier_neighborhood(start_state):
            if self._phi_at(s) > theta:
                return True
        return False

    def _reaches_frontier_uncached(self, start_state: Any) -> bool:
        """True iff some state within FRONTIER_HORIZON hops has
        Φ > θ_frontier (start state included)."""
        theta = getattr(self, "_theta_frontier", CONST["theta_frontier"])
        if self._phi_at(start_state) > theta:
            return True
        visited = {start_state}
        frontier = [start_state]
        for _ in range(int(CONST["frontier_horizon"])):
            nxt: List[Any] = []
            for s in frontier:
                for nbr in self._aem_neighbors(s):
                    if nbr in visited:
                        continue
                    if self._phi_at(nbr) > theta:
                        return True
                    visited.add(nbr)
                    nxt.append(nbr)
            frontier = nxt
            if not frontier:
                break
        return False

    def _concept_leads_to_frontier(self, concept: Any) -> bool:
        """Reachability-to-frontier gate. Propose a concept only if
        pursuing it can reach unexplored territory. Unknown post-states
        are allowed (do not over-suppress young concepts). Reachability —
        not the local post-state Φ alone — so the concept reactivates
        when the world changes (a door closes; re-opening unlocks new
        territory again)."""
        post_states = getattr(concept, "post_states", None) \
            or getattr(concept, "enabling_pre_states", None)
        if not post_states:
            return True
        for ps in post_states:
            if self._reaches_frontier(ps):
                return True
        return False

    def _p1_2_max_reach_phi(self, concept: Any) -> float:
        """[P1-2 MEASUREMENT ONLY — Batch 2] Frontier-reach margin of a
        concept: max Phi over the FRONTIER_HORIZON-hop AEM neighborhood of
        its post/enabling states; +inf when no states are recorded. Exact
        margin form of _concept_leads_to_frontier: that gate passes at
        threshold theta iff this value > theta (same post_states-or-
        enabling_pre_states fallback, same empty-passes rule, same cached
        neighborhoods and _phi_at reads). Never called from a behavioral
        path — only from the p1_2_measure hook in step_41."""
        post_states = getattr(concept, "post_states", None) \
            or getattr(concept, "enabling_pre_states", None)
        if not post_states:
            return math.inf
        best = -math.inf
        for ps in post_states:
            for s in self._frontier_neighborhood(ps):
                v = self._phi_at(s)
                if v > best:
                    best = v
        return best

    # -------------------------------------------------------------------------
    # [Phase 3] Fertility-to-locomotion inheritance — helpers
    # -------------------------------------------------------------------------

    def _active_goal_concepts(self) -> List[Any]:
        """Top-K strongest causal concepts by EF_combined, cached per step.
        Deterministic tie-break by serialized concept id (§9)."""
        if self._goal_concepts_cache is not None and self._goal_concepts_cache_t == self.t:
            return self._goal_concepts_cache
        causal = [
            c for c in self.graph.concepts.values()
            if getattr(c, "family", None) in (
                ConceptFamily.EVENT_CAUSAL, ConceptFamily.TEMPORAL_CHAIN
            ) and getattr(c, "EF_combined", 0.0) > 0.0
        ]
        causal.sort(
            key=lambda c: (-getattr(c, "EF_combined", 0.0), serialize_value(c.id))
        )
        top = causal[: int(CONST["fert_inherit_top_k"])]
        self._goal_concepts_cache = top
        self._goal_concepts_cache_t = self.t
        return top

    def _min_aem_distance(self, start: Any, targets) -> Optional[int]:
        """Fewest AEM hops from `start` to any state in `targets`, or None
        if unreachable within 2×FRONTIER_HORIZON. Memoized per step."""
        ae = getattr(self, "action_edge", None)
        if ae is None or not targets:
            return None
        target_set = frozenset(targets)
        key = (start, target_set)
        if key in self._aem_dist_cache:
            return self._aem_dist_cache[key]
        result: Optional[int] = None
        if start in target_set:
            result = 0
        else:
            visited = {start}
            frontier = [start]
            depth = 0
            max_depth = int(CONST["frontier_horizon"]) * 2
            while frontier and depth < max_depth and result is None:
                nxt: List[Any] = []
                for s in frontier:
                    found = False
                    for nbr in self._aem_neighbors(s):
                        if nbr in target_set:
                            result = depth + 1
                            found = True
                            break
                        if nbr not in visited:
                            visited.add(nbr)
                            nxt.append(nbr)
                    if found:
                        break
                frontier = nxt
                depth += 1
        self._aem_dist_cache[key] = result
        return result

    def _fertility_inheritance(self, current_state: Any, action: Any) -> float:
        """[Phase 3] Additive utility bonus for actions whose predicted
        successor moves CLOSER (AEM geodesic) to a high-EF goal concept's
        enabling states. Replaces the reflex as the approach mechanism for
        conceptual goals (Phase 4 decoupling).

        FertNorm uses the §2.20.3 log-normalization; approach_gain is the
        fractional distance reduction; W_FERT_INHERIT scales the term.
        Fertility-derived ⇒ reads 0.0 under the "fertility" [AUDIT SEAM]
        (regimes C/E), keeping the ablation ladder honest."""
        if "fertility" in self.ablations:
            return 0.0
        if current_state is None:
            return 0.0
        ae = getattr(self, "action_edge", None)
        predicted_s: Any = None
        predict = getattr(ae, "predict", None)
        if callable(predict):
            # §5.3.4 canonical deterministic successor (count argmax,
            # tie-break by serialized state) — preferred over a raw
            # insertion-order max().
            predicted_s = predict(current_state, action)
        if predicted_s is None:
            outcomes = self._get_outcomes(current_state, action)
            if not outcomes:
                return 0.0
            predicted_s = sorted(
                outcomes.items(), key=lambda kv: (-kv[1], serialize_int(kv[0]))
            )[0][0]
        best = 0.0
        for concept in self._active_goal_concepts():
            fert = getattr(concept, "EF_combined", 0.0)
            enabling = getattr(concept, "enabling_pre_states", None)
            if not enabling:
                continue
            d_now = self._min_aem_distance(current_state, enabling)
            d_next = self._min_aem_distance(predicted_s, enabling)
            if d_now is None or d_next is None or d_now <= 0:
                continue
            if d_next < d_now:
                fert_norm = math.log(1.0 + fert) / math.log(
                    1.0 + CONST["RF_max"] + CONST["eps"]
                )
                approach_gain = (d_now - d_next) / float(max(d_now, 1))
                bonus = fert_norm * approach_gain * CONST["w_fert_inherit"]
                if bonus > best:
                    best = bonus
        return best

    # -------------------------------------------------------------------------
    # [Phase 4] Reflex decoupling from conceptual locomotion — helpers
    # -------------------------------------------------------------------------

    def _is_abstract_goal(self, concept: Any) -> bool:
        """True for causal/temporal schemas or goals whose anchor has no
        real position. Abstract goals navigate via the AEM geodesic
        (_resolve_goal_action steps 1–3) plus fertility inheritance —
        never via the reflex anchor fallback."""
        if getattr(concept, "family", None) in (
            ConceptFamily.EVENT_CAUSAL, ConceptFamily.TEMPORAL_CHAIN
        ):
            return True
        # Explicit None-chain, NOT `trigger_anchor or anchor`: anchor id 0
        # (the agent effector) is falsy and `or` would skip past it.
        anchor = getattr(concept, "trigger_anchor", None)
        if anchor is None:
            anchor = getattr(concept, "anchor", None)
        if anchor is not None and not self._anchor_has_position(anchor):
            return True
        return False

    def _anchor_has_position(self, anchor: Any) -> bool:
        """Delegates to the Seam-2 single source of truth
        (sensory_floor.resolve_anchor_position via _anchor_position):
        None AND the (-1, -1) sentinel both mean "no position", so the
        utility engine and the pipeline trigger-grounding cannot
        desynchronize."""
        return self._anchor_position(anchor, self._observation) is not None

    # -------------------------------------------------------------------------
    # Step 42: Utility Computation (§2.20.3)
    # -------------------------------------------------------------------------

    def step_42_compute_utilities(
        self, proposals: Set[Any], current_state: Any
    ) -> Dict[Any, float]:
        if self._current_state != current_state:
            self._active_sequence = None
            self._active_transfer = None
            self._transfer_costs.clear()
        self._current_state = current_state

        utilities: Dict[Any, float] = {}
        plan_actions = self._plan_actions(current_state)

        # [DETERMINISM §9] proposals is a set of action values (strs):
        # iteration order is PYTHONHASHSEED-randomized, which made the
        # utilities dict insertion order seed-dependent. Serialize-
        # bytes sort is canonical and process-stable.
        for a in sorted(proposals, key=lambda a: serialize_value(a)):
            if a == ACTION_NULL:
                utilities[a] = 0.0
                continue

            if (
                not plan_actions
                and self.rho_spawn < self.rho_spawn_star
                and self._is_pure_probe(a, plan_actions)
            ):
                utilities[a] = self._probe_utility(a, current_state)
            else:
                utilities[a] = self._nonprobe_utility(a, current_state)

        return utilities

    def _is_pure_probe(self, a: Any, plan_actions: Set[Any]) -> bool:
        if a in plan_actions:
            return False
        if isinstance(a, tuple):
            return False
        if self._refractory(a):
            return False
        return is_probe_action(a, self.env)

    # -------------------------------------------------------------------------
    # Step 43: Action Selection (§2.20.3)
    # -------------------------------------------------------------------------

    def step_43_select_action(
        self, utilities: Dict[Any, float], current_state: Any = None
    ) -> Any:
        if not utilities:
            return ACTION_NULL
        sorted_actions = sorted(
            utilities.items(),
            key=lambda kv: (-kv[1], -self._tie_break(kv[0])),
        )
        selected = sorted_actions[0][0]
        self._last_selected = selected
        src = self._action_transfer_source.get(selected)
        if src is not None:
            core_hash, t_pred = src
            self._transfer_action_selected.setdefault((core_hash, t_pred), self.t)
        self._note_macro_start(selected)
        return selected

    # -------------------------------------------------------------------------
    # [Seam 1] Macro-execution tracking (Executor/ActionSequence outcomes)
    # -------------------------------------------------------------------------

    def _note_macro_start(self, selected: Any) -> None:
        """Open (or refresh) the in-flight macro record when the selected
        action is the first step of an active transferred ActionSequence.
        The ORIGINAL plan is tracked (not each step's re-planned
        remainder): the audit question is whether the agent carried the
        transferred macro through, and re-planning mid-flight must not
        launder an interruption into a fresh plan."""
        if self._active_sequence is None or not self._active_sequence.actions:
            return
        if self._active_transfer is None:
            return
        planned = tuple(self._active_sequence.actions)
        core_hash = self._active_transfer.core_hash
        if selected != planned[0]:
            return
        m = self._macro_in_flight
        if m is not None and m["core_hash"] == core_hash and m["completed"] > 0:
            return  # already executing this macro
        if m is not None and m["core_hash"] != core_hash:
            # A different macro supersedes an unfinished one — the
            # superseded execution is an interruption, recorded honestly.
            self._emit_macro_outcome(success=False)
        self._macro_in_flight = {
            "core_hash": core_hash,
            "planned": planned,
            "completed": 0,
            "t_start": self.t,
        }

    def _advance_macro(self, executed_action: Any) -> None:
        """Post-execution macro bookkeeping: advance on the planned next
        step, emit completion, or emit an interruption with partial
        progress (K4: partial executions are credited proportionally by
        the pipeline's completion-ratio trace scaling, never as full
        successes or full failures)."""
        m = self._macro_in_flight
        if m is None:
            return
        planned = m["planned"]
        comp = m["completed"]
        if comp < len(planned) and executed_action == planned[comp]:
            m["completed"] = comp + 1
            if m["completed"] >= len(planned):
                self._emit_macro_outcome(success=True)
        else:
            self._emit_macro_outcome(success=False)

    def _emit_macro_outcome(self, success: bool) -> None:
        m = self._macro_in_flight
        self._macro_in_flight = None
        if m is None:
            return
        core_hash = m["core_hash"]
        try:
            basis = tuple(self.transfer_basis(core_hash))
        except Exception:
            basis = ()
        self.macro_outcomes.append(
            MacroExecutionOutcome(
                core_hash=core_hash,
                success=success,
                steps_used=int(m["completed"]),
                steps_planned=len(m["planned"]),
                reward_delta=0.0,
                concepts_involved=basis,
                t=self.t,
            )
        )
        # Hard bound (the pipeline drains every step when wired; an
        # unwired engine must not leak).
        if len(self.macro_outcomes) > 256:
            del self.macro_outcomes[: len(self.macro_outcomes) - 256]

    def pop_selection_record(
        self, core_hash: bytes, t_pred: int
    ) -> Optional[int]:
        """[F2] Retire and return the selection step for a transfer window."""
        return self._transfer_action_selected.pop((core_hash, t_pred), None)

    def drain_macro_outcomes(self) -> List[MacroExecutionOutcome]:
        """Return and clear completed/interrupted macro-execution records.
        Consumed by the Pipeline, which merges them with Prediction's
        transfer validations into the FertilityEngine plan-event
        interface (Seam 1: execution-gated, proportionally scaled
        credit — the orphaned-planner-listener defect closed without
        inventing a parallel credit path)."""
        out = list(self.macro_outcomes)
        self.macro_outcomes.clear()
        return out

    # -------------------------------------------------------------------------
    # Helpers (§2.20.2)
    # -------------------------------------------------------------------------

    def _probe_obligation(self, a: Any) -> bool:
        return self.rho_spawn < self.rho_spawn_star

    def _goal_satiety(self, P: int) -> bool:
        # §2.20.2: GoalSatiety(P,t) = 1[t − lastGoalSuccess(P) ≤ τ_satiety]
        last = self.last_goal_success.get(P, -math.inf)
        return (self.t - last) <= CONST["tau_satiety"]

    def _refractory(self, a: Any) -> bool:
        until = self.refractory_until.get(a, -1)
        return self.t < until

    def _probe_utility(self, a: Any, current_state: Any = None) -> float:
        urgency = self._probe_urgency(a, current_state)
        return (
            CONST["Omega_probe"] * urgency
            + CONST["sigma_noise"] * self._tie_break(a)
        )

    def _probe_urgency(self, a: Any, current_state: Any = None) -> float:
        fatigue = self.fatigue[a]
        obligation = 1.0 if self.rho_spawn < self.rho_spawn_star else 0.0
        return clamp(1.0 - CONST["eta_fatigue"] * fatigue, 0.0, 1.0) * obligation

    def _nonprobe_utility(self, a: Any, current_state: Any) -> float:
        plan_actions = self._plan_actions(current_state)
        target = self._target(a, current_state, plan_actions)
        frontier = self._frontier(a, target, current_state, plan_actions)

        fert_norm = self._fert_norm(target)

        layer = self._target_layer(target)
        abstract = (
            min(CONST["M_layer_max"], 1 + CONST["eta_layer"] * layer) * fert_norm
        )
        abstract_norm = abstract / CONST["M_layer_max"]

        rd = self._realization_distance_for_target(target, current_state)
        if rd < math.inf:
            exponent = clamp(-CONST["lambda"] * rd, -50.0, 50.0)
            real_val = math.exp(exponent)
        else:
            real_val = 0.0

        need = self._need(target)
        prov = self._prov(target)
        consol = clamp(need + prov, 0, 1)

        phi_val = self.phi.get(frontier, 0)
        if phi_val > 0:
            phi_norm = math.log(1 + phi_val) / math.log(
                1 + CONST["Phi_max"] + CONST["eps"]
            )
        else:
            phi_norm = 0.0
        protect_norm = (
            CONST["kappa_protect"] / (1 + CONST["kappa_protect"])
        ) * phi_norm

        attempt_val = 1.0 / (1.0 + self.N_attempt[a])
        entropy = self._entropy(a, current_state)

        motor_norm = (
            CONST["xi_attempt"] * attempt_val + CONST["xi_entropy"] * entropy
        ) / (CONST["xi_attempt"] + CONST["xi_entropy"])

        plan_persist = 1 if a in plan_actions else 0

        conflict = self._conflict(
            a, target, plan_actions, current_state=current_state
        )

        # §2.20.3 CLOSED FORM — verbatim:
        #   U_raw = U_cap·[Σ ŵᵢ·Normᵢ + ŵ_persist·PlanPersist
        #                   − stay_discount·1[a ∈ a_null]]
        #           − η_conflict·Conflict
        #   U_t   = clamp(U_raw + σ_noise·τ_break(a), 0, U_cap)
        #
        # The former extra in-bracket terms (plan_bonus, orient_gain·0.30,
        # dest_richness·0.20, −reversal_penalty) and the `is_inert → return
        # 0.0` hard bypass had no blueprint basis: §1.2 permits utility to
        # be generated ONLY from the enumerated sources, and destination
        # "richness" double-counts signals the closed form already carries
        # (outcome Entropy inside MotorNorm; Φ(s′) inside ProtectNorm via
        # the frontier map; target fertility inside FertNorm). Reflex-orient
        # proposals compete through their bound target concept like any
        # other proposal.
        #
        # STAY DISCOUNT (auditor ruling R5 §2.4 — standard v30 term,
        # reinstated): the constitutional no-op class a_null (§2.6:
        # None/"null"/"stay" — primitives.is_noop_action, never env action
        # names) is discounted by P.U_STAY_DISCOUNT inside the U_cap
        # bracket. Rationale (O2, stay-dominance in flat regions): where
        # every action's target-dependent norms are zero, utilities collapse
        # to MotorNorm + noise and the deterministic tie-break let passive
        # idling win ~35% of steps in unmapped territory. The discount
        # (effective 0.10·U_cap = 5.0 points) exceeds the maximum score of
        # a targetless action at a zero tie (w6·MotorNorm·U_cap/W_total ≤
        # 0.4·50/4.1 ≈ 4.88 < 5.0), so exploratory movement ALWAYS beats idling at zero ties, while any
        # genuine goal/frontier utility on the no-op itself (≫ 0.1) still
        # survives the discount — no hard bypass, no forced action.
        stay_disc = CONST["stay_discount"] if is_noop_action(a) else 0.0

        norm_components = [
            fert_norm,
            abstract_norm,
            real_val,
            protect_norm,
            consol,
            motor_norm,
        ]
        weights = [
            CONST["w1"],
            CONST["w2"],
            CONST["w3"],
            CONST["w4"],
            CONST["w5"],
            CONST["w6"],
        ]
        W_total = sum(weights) + CONST["eta_planPersist"]
        weighted_sum = sum(w * v for w, v in zip(weights, norm_components))
        w_persist = CONST["eta_planPersist"] / W_total

        # [Phase 3 — fertility-to-locomotion inheritance] Additive
        # in-bracket term (before clamping, per the patch directive).
        # Fertility-derived: reads 0.0 under the "fertility" audit seam.
        fert_inherit = self._fertility_inheritance(current_state, a)

        raw = CONST["U_cap"] * (
            weighted_sum / W_total + w_persist * plan_persist + fert_inherit
            - stay_disc
        ) - CONST["eta_conflict"] * conflict
        utility = clamp(
            raw + CONST["sigma_noise"] * self._tie_break(a),
            0,
            CONST["U_cap"],
        )
        return utility

    # -------------------------------------------------------------------------
    # Target resolution
    # -------------------------------------------------------------------------

    def _target(
        self,
        a: Any,
        current_state: Any,
        plan_actions: Optional[Set[Any]] = None,
    ) -> Any:
        if a in self._action_targets:
            return self._action_targets[a]

        if plan_actions is not None and a in plan_actions:
            if self._active_transfer is not None:
                return self._active_transfer.core_hash

        if (
            a == getattr(self, "_orient_action", None)
            and getattr(self, "_orient_target_concept", None) is not None
        ):
            return self._orient_target_concept

        if isinstance(a, tuple) and len(a) == 2 and a[0] == "goal":
            return a[1]

        outcomes = self._get_outcomes(current_state, a)
        if outcomes:
            best_s = sorted(
                outcomes.items(),
                key=lambda kv: (-kv[1], serialize_int(kv[0])),
            )[0][0]
            return best_s
        return current_state

    def _frontier(
        self,
        a: Any,
        target: Any,
        current_state: Any,
        plan_actions: Optional[Set[Any]] = None,
    ) -> Any:
        """§2.20.2 frontier map:
            frontier(a) = s′                        if a has dominant outcome s′
                        = enablingState(target(a))  if a ∈ A_goal ∪ A_plan
                        = s_t                       otherwise.
        v31 removed planning_engine's enablingState(P); the surviving
        structural source is the concept's `enabling_pre_states` set (the
        same source the §2.25 RD rule uses). The CLOSEST enabling state
        (BFS over the action-edge graph — the §1.9-mandated replacement
        for coordinate navigation) is selected; ties break by lower state
        hash for determinism."""
        outcomes = self._get_outcomes(current_state, a)
        if outcomes:
            best_s = sorted(
                outcomes.items(),
                key=lambda kv: (-kv[1], serialize_int(kv[0])),
            )[0][0]
            return best_s
        is_plan = plan_actions is not None and a in plan_actions
        if is_plan or self._is_goal(a) or a in self._action_targets:
            eps = self._target_enabling_states(target)
            if eps and current_state is not None:
                best = None
                best_key = None
                for s in eps:
                    d = self.action_edge.shortest_path_length(
                        current_state, s, max_depth=int(CONST["C_planMax"])
                    )
                    if not math.isfinite(d):
                        continue
                    key = (d, serialize_int(s))
                    if best_key is None or key < best_key:
                        best_key = key
                        best = s
                if best is not None:
                    return best
        return current_state

    def _target_enabling_states(self, target: Any) -> Set[Any]:
        """Enabling pre-states of a concept target (v31 survivor of the
        planning-era enablingState map). Transfer targets (bytes) have no
        pre-state set at the concept level and resolve to ∅ → s_t."""
        if isinstance(target, int):
            concept = self.graph.concepts.get(target)
            if concept is not None:
                return set(getattr(concept, "enabling_pre_states", None) or ())
        return set()

    def _is_goal(self, a: Any) -> bool:
        return isinstance(a, tuple) and len(a) == 2 and a[0] == "goal"

    def _is_plan(
        self,
        a: Any,
        current_state: Any = None,
        plan_actions: Optional[Set[Any]] = None,
    ) -> bool:
        if plan_actions is None:
            plan_actions = self._plan_actions(current_state)
        return a in plan_actions

    def _plan_actions(self, current_state: Any = None) -> Set[Any]:
        """[SPEC INTERPRETATION §2.20.1] A_t^transfer = {first action of
        the active ActionSequence}, or ∅ if no active transfer."""
        if self._active_sequence is not None and self._active_sequence.actions:
            return {self._active_sequence.actions[0]}
        if self._active_transfer is not None:
            tc_hash = self._active_transfer.core_hash
            for action, (ch, _t_pred) in self._action_transfer_source.items():
                if ch == tc_hash:
                    return {action}
        return set()

    # -------------------------------------------------------------------------
    # Formula components
    # -------------------------------------------------------------------------

    def _target_layer(self, target: Any) -> int:
        if isinstance(target, bytes):
            return 1
        if isinstance(target, int) and target in self.graph.concepts:
            return self.graph.concepts[target].layer
        return 0

    def _fert_norm(self, P: Any) -> float:
        """Normalized fertility per §2.20.3 / Update 11 §4.1:
          - Transfer Core Hash / Concept ID: log(1 + EF_combined) / log(1 + RF_max)
          - State Hash (bytes or int): PhiNorm(s) = log(1 + Phi(s)) / log(1 + Phi_max)

        The target of a utility term may be a transfer core hash (bytes), a
        concept id (int indexing the ConceptGraph), or a raw state hash (the
        structural-potential map's domain). The byte form is therefore
        resolved in that order: transfer core first, then state hash — a byte
        key absent from the core library is a state, not a failure.
        """
        if "fertility" in self.ablations:
            # [AUDIT SEAM] Regime C: every fertility-derived term reads 0.
            return 0.0
        if isinstance(P, bytes):
            # 1. Could be a transfer core hash.
            transfer_fert = self._transfer_fertility(P)
            if transfer_fert > 0.0:
                return transfer_fert
            # 2. Could be a serialized byte state hash.
            phi_val = self.phi.get(P, 0.0) if self.phi else 0.0
            if phi_val > 0.0:
                return math.log(1 + phi_val) / math.log(
                    1 + CONST["Phi_max"] + CONST["eps"]
                )
            return 0.0

        if isinstance(P, int):
            # 3. Concept ID.
            if P in self.graph.concepts:
                node = self.graph.concepts[P]
                ef = getattr(node, "EF_combined", 0.0)
                return math.log(1 + ef) / math.log(
                    1 + CONST["RF_max"] + CONST["eps"]
                )
            # 4. Int state hash.
            phi_val = self.phi.get(P, 0.0) if self.phi else 0.0
            if phi_val > 0.0:
                return math.log(1 + phi_val) / math.log(
                    1 + CONST["Phi_max"] + CONST["eps"]
                )

        return 0.0

    def transfer_basis(self, core_hash: bytes) -> List[Any]:
        """Live concepts overlapping the transfer's consequence subgraph —
        the v31 realization BASIS of a transfer (structural successor of
        the deleted PlanningEngine's basisSnapshot). Consumed by the
        pipeline to convert validated transfer outcomes into the
        plan_successes/plan_failures events FertilityEngine's §2.15.3 CF
        increments and §2.10.2 step-blame already consume."""
        if self.core_library is None:
            return []
        entry = self.core_library.lookup_by_hash(core_hash, CoreKind.ACTION)
        if entry is None:
            return []
        payload = getattr(entry, "payload", None)
        consequence = getattr(payload, "consequence_subgraph", None)
        if consequence is None or not consequence.nodes:
            return []
        basis: Set[int] = set()
        accessor = getattr(self.graph, "get_concepts_containing", None)
        rev = (
            None
            if callable(accessor)
            else getattr(self.graph, "_node_to_concepts", None)
        )
        for nid in consequence.nodes:
            if callable(accessor):
                cids = accessor(nid)
            elif rev is not None:
                cids = rev.get(nid, ())
            else:
                cids = [
                    cid
                    for cid, c in self.graph.concepts.items()
                    if nid in c.members
                ]
            for cid in cids:
                c = self.graph.concepts.get(cid)
                if c is None:
                    continue
                if c.status in (ConceptStatus.CRYSTALLIZED, ConceptStatus.STABLE):
                    basis.add(cid)
        return sorted(basis)

    def _transfer_fertility(self, core_hash: bytes) -> float:
        """Max EF_combined of any live concept overlapping the transfer's
        consequence.

        [SELF-AUDIT SA1/P2] Uses the graph's node→concepts reverse index
        when available; falls back to linear scan otherwise. Converts the
        per-target cost from O(N_concepts) to O(|consequence.nodes| ×
        avg_concepts_per_node).
        """
        cached = self._transfer_fert_cache.get(core_hash)
        if cached is not None:
            return cached
        if "fertility" in self.ablations:
            # [AUDIT SEAM] Regime C.
            self._transfer_fert_cache[core_hash] = 0.0
            return 0.0
        if self.core_library is None:
            self._transfer_fert_cache[core_hash] = 0.0
            return 0.0
        entry = self.core_library.lookup_by_hash(core_hash, CoreKind.ACTION)
        if entry is None:
            self._transfer_fert_cache[core_hash] = 0.0
            return 0.0
        payload = getattr(entry, "payload", None)
        consequence = getattr(payload, "consequence_subgraph", None)
        if consequence is None or not consequence.nodes:
            self._transfer_fert_cache[core_hash] = 0.0
            return 0.0

        best_ef = 0.0
        rev = getattr(self.graph, "_node_to_concepts", None)
        if rev is not None:
            seen_cids: Set[int] = set()
            for nid in consequence.nodes:
                for cid in rev.get(nid, ()):
                    if cid in seen_cids:
                        continue
                    seen_cids.add(cid)
                    c = self.graph.concepts.get(cid)
                    if c is None:
                        continue
                    if c.status not in (
                        ConceptStatus.CRYSTALLIZED,
                        ConceptStatus.STABLE,
                    ):
                        continue
                    ef = getattr(c, "EF_combined", 0.0)
                    if ef > best_ef:
                        best_ef = ef
        else:
            # Fallback: linear scan. Bounded by graph size.
            for concept in self.graph.concepts.values():
                if concept.status not in (
                    ConceptStatus.CRYSTALLIZED,
                    ConceptStatus.STABLE,
                ):
                    continue
                if concept.members & consequence.nodes:
                    ef = getattr(concept, "EF_combined", 0.0)
                    if ef > best_ef:
                        best_ef = ef

        result = math.log(1 + best_ef) / math.log(
            1 + CONST["RF_max"] + CONST["eps"]
        )
        self._transfer_fert_cache[core_hash] = result
        return result

    def _realization_distance_for_target(
        self, target: Any, current_state: Any
    ) -> float:
        if isinstance(target, bytes):
            return self._transfer_costs.get(target, math.inf)
        if isinstance(target, int) and target in self.graph.concepts:
            return self._concept_realization_distance(target, current_state)
        return math.inf

    def _need(self, P: Any) -> float:
        if not isinstance(P, int) or P not in self.graph.concepts:
            return 0.0
        node = self.graph.concepts[P]
        graph_params = getattr(self.graph, "params", {})
        fam_thresholds = graph_params.get("family_conf_thresholds", {})
        fam_conf = fam_thresholds.get(node.family.name, 3)
        n_conf = getattr(node, "N_conf", 0)
        return max(0, fam_conf - n_conf) / (fam_conf + CONST["eps"])

    def _prov(self, P: Any) -> float:
        if not isinstance(P, int) or P not in self.graph.concepts:
            return 0.0
        node = self.graph.concepts[P]
        if not getattr(node, "provisional", False):
            return 0.0
        t_cryst = getattr(node, "T_cryst", self.t)
        return clamp(
            1 - (self.t - t_cryst) / (CONST["tau_provisional"] + CONST["eps"]),
            0,
            1,
        )

    def _entropy(self, a: Any, current_state: Any) -> float:
        outcomes = self._get_outcomes(current_state, a)
        total = sum(outcomes.values())
        if len(outcomes) <= 1 or total < CONST["eps"]:
            return 0.0
        H = 0.0
        for m in outcomes.values():
            p = m / total
            H -= p * math.log(p + CONST["eps"])
        return H / math.log(CONST["N_outcomeMax"] + CONST["eps"])

    def _conflict(
        self,
        a: Any,
        target: Any,
        plan_actions: Optional[Set[Any]] = None,
        current_state: Any = None,
    ) -> int:
        """§2.20.2: Conflict(a,t) = 1[ActivePlans ≠ ∅ ∧ a ∉ A_plan ∧
        target(a) ∈ V_t]. The third clause matters: only CONCEPT targets
        compete with the active plan at the goal level. The former
        implementation penalized EVERY non-plan action during an active
        transfer — including plain motor actions whose target is a state —
        at η_conflict = 5.0, effectively freezing exploration for the
        transfer's duration."""
        if self._active_sequence is None or not self._active_sequence.actions:
            return 0
        if plan_actions is None:
            plan_actions = self._plan_actions(current_state)
        if a in plan_actions:
            return 0
        target_is_concept = isinstance(target, int) and target in self.graph.concepts
        return 1 if target_is_concept else 0

    def _tie_break(self, a: Any) -> float:
        """§2.20.3 (normative): τ_break(a) = hash_64(serialize(a)) / 2⁶⁴.

        A pure function of the action — constant across steps and states,
        which is precisely what makes it a stable deterministic tie-break.
        Two successive drift versions lived here: one hashed self.t
        (flipping ties every step → Brownian dithering), the other hashed
        (action, state, mass, attempts) (better, but still off-spec and
        state-dependent). The spec formula has neither failure mode.
        """
        return hash_64(serialize_value(a)) / (2**64)

    # -------------------------------------------------------------------------
    # Post-action bookkeeping (§2.20.2)
    # -------------------------------------------------------------------------

    def update_state_after_action(
        self,
        executed_action: Any,
        success: bool,
        new_anchor_spawned: bool = False,
    ) -> None:
        """§2.20.2 post-action bookkeeping — and nothing else.

        Removed here (all off-spec):
          * "Adaptive Inertia" homeostasis (per-action θ/EMA controller for
            a utility term that does not exist in §2.20.3).
          * "Spatial refractory isolation" (halving nav fail counts on
            state change) and the τ_navRefractory branch — both keyed off
            `_is_nav_action`, which sniffed for "move_*"/"stay"/"up"…
            action-name strings: exactly the env-specific branching the
            constitution forbids. §2.20.2 specifies ONE refractory rule:
            τ_fail·2^min(n_fail−1, L_max), reset on success.
          * env-causal gating of goal-satiety recording (a
            get_action_effect_atom sniff distinguishing "causal" from
            "transport" actions by environment API). The env-agnostic
            criterion is structural: satiety is recorded for action-bound
            concepts whose bound effect marker is now PRESENT — transport
            actions without effect bindings simply record nothing.
        """
        if (
            self.innate_reflex is not None
            and success
            and hasattr(self.innate_reflex, "notify_interaction")
        ):
            self.innate_reflex.notify_interaction(
                executed_action, self._observation
            )

        # [Seam 1] Macro-execution progress (before fatigue/refractory
        # bookkeeping — the tracker reads only the executed action).
        self._advance_macro(executed_action)

        # Fatigue (§2.20.2): Fatigue(a,t+1) = η_decay·Fatigue(a,t) +
        # (1−η_decay)·1[a_t = a].
        for a in self.fatigue:
            self.fatigue[a] *= CONST["eta_fatigueDecay"]
        self.fatigue[executed_action] += 1 - CONST["eta_fatigueDecay"]

        # Attempt counter (§2.20.2).
        self.N_attempt[executed_action] += 1

        # Spawn-rate EMA (§2.20.2 probe obligation).
        self.rho_spawn = (1 - CONST["kappa_ema"]) * self.rho_spawn + CONST[
            "kappa_ema"
        ] * (1 if new_anchor_spawned else 0)

        if success:
            # §2.20.2: consecutive-failure count resets on success.
            self.consecutive_fail_count[executed_action] = 0
            self.refractory_until[executed_action] = -1
            # Goal satiety (§2.20.2 lastGoalSuccess): record for every
            # live concept bound to the executed action whose bound effect
            # is now present. Effect presence is read through the
            # architectural effect-marker interface (§2.2); causality of
            # the link is established by the counterfactual machinery
            # (prediction gate_bind/gate_undo) — no env-specific action
            # classification is consulted.
            self._build_action_concept_index()
            for concept in self._action_concept_index.get(executed_action, ()):
                effect_atom = self._action_effect_atom_for(
                    concept, executed_action
                )
                if effect_atom is not None and self._has_effect_marker(
                    effect_atom, self.t
                ):
                    self.last_goal_success[concept.id] = self.t
        else:
            # §2.20.2 refractory escalation:
            # refractoryUntil(a) = t_fail + clamp(τ_fail·2^min(n−1, L_max),
            #                              τ_fail, τ_failMax)
            self.consecutive_fail_count[executed_action] += 1
            n_fail = self.consecutive_fail_count[executed_action]
            L_max = math.ceil(
                math.log2(CONST["tau_failMax"] / CONST["tau_fail"])
            )
            exponent = min(n_fail - 1, L_max)
            refractory_duration = clamp(
                CONST["tau_fail"] * (2**exponent),
                CONST["tau_fail"],
                CONST["tau_failMax"],
            )
            self.refractory_until[executed_action] = (
                self.t + refractory_duration
            )

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def utility_diagnostic(
        self,
        proposals: Optional[Set[Any]] = None,
        utilities: Optional[Dict[Any, float]] = None,
    ) -> Dict[str, Any]:
        diag: Dict[str, Any] = {
            "fatigue": dict(self.fatigue),
            "N_attempt": dict(self.N_attempt),
            "rho_spawn": self.rho_spawn,
            "refractory_until": dict(self.refractory_until),
            "active_transfer": (
                self._active_transfer.core_hash.hex()[:8]
                if self._active_transfer is not None
                else None
            ),
            "active_sequence_len": (
                len(self._active_sequence.actions)
                if self._active_sequence is not None
                else 0
            ),
        }
        if proposals is not None:
            diag["proposal_count"] = len(proposals)
            diag["probe_count"] = sum(
                1 for a in proposals if self._is_pure_probe(a, set())
            )
            diag["motor_count"] = sum(
                1
                for a in proposals
                if not self._refractory(a)
                and not self._is_goal(a)
                and not self._is_plan(a)
            )
        if utilities is not None:
            diag["utilities"] = {
                str(a): round(u, 4) for a, u in utilities.items()
            }
        return diag