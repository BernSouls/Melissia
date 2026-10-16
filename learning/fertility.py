"""
MODULE 06: FertilityEngine — v31

Responsibility: Credit, RF, GP, EF, contextual fertility, EF_combined.
Owns: credit, RF, GP, EF, CF, EF_combined.
Interface: reads ConceptGraph topology, writes fertility values to nodes.

Reads: concept graph topology, activation states, plan outcomes.
Writes: credit(P), RF(P), GP(P), EF(P), CF(P, C_t), EF_combined(P, S_t).

v31 changes:
  - EBAF (analogical prototype registry + PF computation) moved to
    prediction.py §8.4 per §8.9. This module no longer owns the registry;
    it queries prediction.analogical_prior() when computing EF_combined.
  - step39b_update_ebaf is retained as a delegating shim so pipeline.py
    does not need to be rewired in the same change.
  - step40b_compute_ebaf consumes evidence sets directly (from each
    concept's own evidence_registry) rather than a pre-computed PF.
  - _is_active_edge now checks across all EdgeType values between a
    (parent, child) pair. Under v31's typed-edge store, more than one
    edge type can connect the same pair, and a caller asking "is this
    pair connected by any active edge?" must not be satisfied by the
    first edge returned by get_edge.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from collections import deque

from sensory_floor import TypedSignature, Detection
from concept_graph import ConceptStatus, ConceptFamily
from core.types import EdgeType
from core.hashing import (  # noqa: F401
    clamp,
    hash_64,
    serialize_bytes,
    serialize_float,
    serialize_int,
    serialize_set,
    serialize_tuple,
    serialize_value,
)
import parameters as P


# =============================================================================
# CONSTANTS
# =============================================================================

EPSILON = P.EPSILON

# Credit parameters
EPS_INITIAL = P.EPS_INITIAL
C_ENABLEMENT = P.C_ENABLEMENT
W_ENABLEMENT = 1.0
C_SHARE = 0.1
C_PLAN = P.C_PLAN
C_SUBGOAL = P.C_SUBGOAL
C_SUBGOAL_EPISODE = 0.5
C_SUBGOAL_LIFETIME = 5.0
RHO_REVERSAL = P.RHO_REVERSAL
C_MAX = P.C_MAX
KAPPA_SPEC = P.KAPPA_SPEC
LAMBDA_ELIG = 0.9
ALPHA_STEP = 0.1
BETA_STEP = 0.1
N_STEP_MIN = P.N_STEP_MIN

# Fertility and distance
TAU_EXHAUST = 5000
TAU_RECENT = P.TAU_RECENT
LAMBDA = 0.15
ALPHA = P.ALPHA_GP
BETA = P.BETA_GP
GAMMA = P.GAMMA_GP
DELTA = P.DELTA_GP
BFS_MAX = P.BFS_MAX
D_MAX = P.D_MAX
RF_MAX = P.RF_MAX
GP_MAX = P.GP_MAX

# Contextual fertility
ETA_CTX = P.ETA_CTX
BETA_CTX = P.BETA_CTX
N_CTX_MAX = P.N_CTX_MAX
K_CONTEXT = P.K_CONTEXT
TAU_CTX_FORGET = 5000
CF_MAX = P.CF_MAX
C_CTX = C_ENABLEMENT

# Analogical prior weight in CF_aug (§8.12)
ETA_EBAF = P.ETA_EBAF

# Numerical safety
KAPPA_EMA = 0.05
THETA_EDGE_ACTIVE = P.THETA_EDGE_ACTIVE
TAU_EDGE_INACTIVE = P.TAU_EDGE_INACTIVE


# =============================================================================
# FERTILITY ENGINE
# =============================================================================


class FertilityEngine:
    def __init__(
        self,
        concept_graph: Any,
        activation_engine: Any,
        planning_engine: Any = None,
        context_signature: int = 0,
        prediction_module: Any = None,
    ):
        self.cg = concept_graph
        self.act = activation_engine
        self.plan = planning_engine
        self.prediction = prediction_module
        self.t = 0
        # Per-revision memo for _is_active_edge. Concept↔concept ancestry
        # edges are created only as CO_ACTIVATION (§_instantiate_concept_edge),
        # but _is_active_edge must stay type-agnostic per §1.12 (any typed edge
        # counts). The scan is O(|EdgeType|) and _compute_GP calls it once per
        # BFS descendant, so without memoization the value function pays
        # O(concepts × descendants × 13) dict probes every step. The memo is
        # keyed on cg.edge_revision, which step_28_update_edges bumps after all
        # weight/inactive_time/topology mutations settle — so the cache is
        # correct by construction and never stale.
        self._active_edge_memo: Dict[Tuple[int, int], bool] = {}
        self._active_edge_memo_rev: int = -1
        self.context_signature = context_signature

        # EMA of activation for specificity weighting.
        self.avg_activation_ema: Dict[int, float] = {}

        # Initialize concept fields.
        for cid, concept in self.cg.concepts.items():
            if not hasattr(concept, "credit"):
                concept.credit = EPS_INITIAL
            if not hasattr(concept, "RF"):
                concept.RF = 0.0
            if not hasattr(concept, "GP"):
                concept.GP = 1.0
            if not hasattr(concept, "EF"):
                concept.EF = 0.0
            if not hasattr(concept, "EF_eff"):
                concept.EF_eff = 0.0
            if not hasattr(concept, "EF_combined"):
                concept.EF_combined = 0.0
            if not hasattr(concept, "CF"):
                concept.CF = {}
            if not hasattr(concept, "step_F_count"):
                concept.step_F_count = 0
            if not hasattr(concept, "step_S_count"):
                concept.step_S_count = 0
            if not hasattr(concept, "elig"):
                concept.elig = 0.0
            self.avg_activation_ema[cid] = 0.0

    # =========================================================================
    # PUBLIC STEP METHODS
    # =========================================================================

    def step30_recompute_RF_GP_EF(self, t: int) -> None:
        """Recompute RF, GP, EF in reverse topological order."""
        self.t = t
        self._apply_enablement_credit(t)
        order = self._reverse_topological_order()
        for cid in order:
            concept = self.cg.concepts[cid]
            if concept.status == ConceptStatus.TOMBSTONED:
                continue
            self._compute_RF(concept)
            self._compute_GP(concept)
            self._compute_EF(concept)

    def get_ef_combined(self, cid: int) -> float:
        concept = self.cg.get_concept(cid)
        if concept is None:
            return 0.0
        return getattr(concept, "EF_combined", 0.0)

    def get_RF(self, cid: int) -> float:
        concept = self.cg.get_concept(cid)
        if concept is None:
            return 0.0
        return getattr(concept, "RF", 0.0)

    def get_specificity(self, cid: int) -> float:
        """§P3: specificity accessor for the pruning predicate.
        Delegates to _specificity so there is exactly one formula.
        Bounded (0, 1]."""
        return self._specificity(cid, 0)

    def get_ef_eff_proposal(self, cid: int) -> float:
        """Proposal-safe effective fertility: 0 if dormant/soft/tombstoned."""
        concept = self.cg.get_concept(cid)
        if concept is None:
            return 0.0
        if (
            concept.status == ConceptStatus.TOMBSTONED
            or getattr(concept, "dormant", False)
            or getattr(concept, "soft_dormant", False)
        ):
            return 0.0
        return getattr(concept, "EF_eff", 0.0)

    def step31_update_contextual_fertility(
        self,
        t: int,
        context_signature: int,
        plan_successes: Optional[List[Dict]] = None,
        plan_failures: Optional[List[Dict]] = None,
        newly_crystallized: Optional[List[Any]] = None,
    ) -> None:
        """Update contextual fertility map and EF_eff."""
        self.t = t
        self.context_signature = context_signature
        if plan_successes is None and self.plan is not None:
            plan_successes = self.plan.plan_outcomes.get("success", [])
        if plan_failures is None and self.plan is not None:
            plan_failures = self.plan.plan_outcomes.get("failure", [])
        plan_successes = plan_successes or []
        plan_failures = plan_failures or []

        # Child crystallization → CF increment for parents.
        if newly_crystallized:
            for child in newly_crystallized:
                child_id = child.id if hasattr(child, "id") else child
                for parent_id, parent in self.cg.concepts.items():
                    if parent.status == ConceptStatus.TOMBSTONED:
                        continue
                    if child_id in parent.child_ids:
                        spec = self._specificity(parent_id, t)
                        child_ef = getattr(self.cg.concepts.get(child_id), "EF", 0.0)
                        increment = C_CTX * child_ef * spec
                        self._increment_CF(parent_id, context_signature, increment)
        else:
            for parent_id, parent in self.cg.concepts.items():
                if parent.status == ConceptStatus.TOMBSTONED:
                    continue
                for child_id in parent.child_ids:
                    child = self.cg.get_concept(child_id)
                    if child and child.T_cryst == t:
                        spec = self._specificity(parent_id, t)
                        increment = C_CTX * getattr(child, "EF", 0.0) * spec
                        self._increment_CF(parent_id, context_signature, increment)

        # Plan success → trace-weighted CF.
        for plan in plan_successes:
            basis = self._extract_basis(plan)
            trace_map = plan.get("trace") or {}
            for cid in basis:
                trace = trace_map.get(cid, 0.0)
                if trace > 0.0:
                    self._increment_CF(cid, context_signature, C_PLAN * RF_MAX * trace)

        # Subgoal confirmations.
        confirmed_ids: List[int] = []
        if self.plan is not None and hasattr(self.plan, "drain_subgoal_confirmations"):
            confirmed_ids = self.plan.drain_subgoal_confirmations()
        for cid in confirmed_ids:
            concept = self.cg.get_concept(cid)
            if concept is not None and concept.status != ConceptStatus.TOMBSTONED:
                self._increment_CF(cid, context_signature, C_SUBGOAL * RF_MAX)

        self._update_EF_eff_all()

    def step33_credit_updates(
        self,
        t: int,
        plan_successes: Optional[List[Dict]] = None,
        plan_failures: Optional[List[Dict]] = None,
    ) -> None:
        """Distribute success credit and apply failure reversal."""
        self.t = t
        if plan_successes is None and self.plan is not None:
            plan_successes = self.plan.plan_outcomes.get("success", [])
        if plan_failures is None and self.plan is not None:
            plan_failures = self.plan.plan_outcomes.get("failure", [])
        plan_successes = plan_successes or []
        plan_failures = plan_failures or []

        self._update_step_blame_counters(t, plan_successes, plan_failures)

        for plan in plan_successes:
            basis = self._extract_basis(plan)
            if not basis:
                continue
            trace_map = plan.get("trace") or {}
            total = sum(trace_map.get(cid, 0.0) for cid in basis)
            if total <= EPSILON:
                continue
            for cid in basis:
                trace = trace_map.get(cid, 0.0)
                if trace <= 0.0:
                    continue
                concept = self.cg.get_concept(cid)
                if concept:
                    concept.credit = clamp(concept.credit + C_PLAN * trace, 0, C_MAX)

        for failure in plan_failures:
            basis = failure.get("basis", [])
            if not basis:
                continue
            blame_weights = {}
            for cid in basis:
                bw = self._compute_blame_weight(cid)
                if bw is not None:
                    blame_weights[cid] = bw
            if not blame_weights:
                continue
            max_blame = max(blame_weights.values())
            for cid, bw in blame_weights.items():
                concept = self.cg.get_concept(cid)
                if not concept:
                    continue
                reversal = RHO_REVERSAL * concept.credit * (bw / (max_blame + EPSILON))
                concept.credit = clamp(concept.credit - reversal, 0, C_MAX)

    def step39_decay_contextual_fertility(self) -> None:
        """Decay all CF entries and enforce per-concept cap."""
        for concept in self.cg.concepts.values():
            cf = getattr(concept, "CF", None)
            if not cf:
                continue
            for ctx in list(cf.keys()):
                cf[ctx] *= BETA_CTX
                if cf[ctx] < EPSILON:
                    del cf[ctx]
            if len(cf) > N_CTX_MAX:
                items = sorted(cf.items(), key=lambda x: (x[1], x[0]))
                for ctx, _val in items[: len(cf) - N_CTX_MAX]:
                    del cf[ctx]

    def step39b_update_ebaf(self, t: int) -> None:
        """[v31] Delegating shim. The EBAF registry now lives in prediction
        (§8.4). If prediction_module is not wired, this is a no-op."""
        self.t = t
        if self.prediction is not None:
            self.prediction.step_analogical_register(t, self.cg)

    def step40b_compute_ebaf(self, t: int, context_signature: int) -> None:
        """Compute EF_combined for all concepts (§8.12).

        EF_combined(P) = EF(P) + ETA_CTX * CF_aug(P, C_t)
        CF_aug(P, C_t) = CF_base(P, C_t) + η_ebaf · PF(P) · RF_MAX

        PF(P) is the analogical prior from prediction (§8.4). If prediction
        is not wired, PF = 0 and CF_aug reduces to CF_base.
        """
        self.t = t
        self.context_signature = context_signature
        prediction = self.prediction
        registry_present = prediction is not None and bool(
            prediction.fertile_evidence_registry
        )
        for cid, concept in self.cg.concepts.items():
            # Dormant and soft-dormant concepts have zero fertility. They
            # should not contribute to the Builder's value function.
            if (
                concept.status == ConceptStatus.TOMBSTONED
                or getattr(concept, "dormant", False)
                or getattr(concept, "soft_dormant", False)
            ):
                concept.EF_combined = 0.0
                continue
            pf = 0.0
            if registry_present:
                evidence_set = tuple(evid[1] for evid in concept.evidence_registry)
                pf = prediction.analogical_prior(evidence_set, t)
            cf_base = concept.CF.get(context_signature, 0.0)
            # CF_aug is clamped to [0, RF_MAX] to prevent unbounded growth.
            cf_aug = clamp(cf_base + ETA_EBAF * pf * RF_MAX, 0.0, RF_MAX)
            concept.EF_combined = clamp(concept.EF + ETA_CTX * cf_aug, 0.0, RF_MAX)

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _apply_enablement_credit(self, t: int) -> None:
        for cid, concept in self.cg.concepts.items():
            if concept.status == ConceptStatus.TOMBSTONED:
                continue
            if getattr(concept, "dormant", False) or getattr(
                concept, "soft_dormant", False
            ):
                continue
            A = self.act.get_activation(cid, t)
            onset = self.act.get_onset(cid, t)
            prev_avg = self.avg_activation_ema.get(cid, 0.0)
            self.avg_activation_ema[cid] = (1 - KAPPA_EMA) * prev_avg + KAPPA_EMA * A
            if A == 1 or onset == 1:
                spec = 1.0 / (1.0 + KAPPA_SPEC * self.avg_activation_ema[cid])
                concept.credit = clamp(concept.credit + C_ENABLEMENT * spec, 0, C_MAX)

    def _specificity(self, cid: int, t: int) -> float:
        avg = self.avg_activation_ema.get(cid, 0.0)
        return 1.0 / (1.0 + KAPPA_SPEC * avg)

    def _is_active_edge(self, parent_id: int, child_id: int) -> bool:
        """True iff any edge type between parent and child is active.

        Under v31's typed-edge store, more than one edge type can connect
        the same (parent, child) pair. A caller asking "is this pair
        connected by an active edge?" must not be satisfied by the first
        edge returned by get_edge; the check iterates over all EdgeType
        values and returns True on the first active one found.

        Memoized per cg.edge_revision: the predicate is a pure function of
        edge weight/inactive_time, both of which only change inside
        step_28_update_edges (which bumps the revision). Within a single
        step30 pass the answer is stable, so _compute_GP's BFS and
        _compute_RF's child loop reuse it instead of re-scanning 13 types.
        """
        rev = getattr(self.cg, "edge_revision", 0)
        if rev != self._active_edge_memo_rev:
            self._active_edge_memo = {}
            self._active_edge_memo_rev = rev
        pair = (parent_id, child_id)
        cached = self._active_edge_memo.get(pair)
        if cached is not None:
            return cached
        active = False
        for edge_type in EdgeType:
            edge = self.cg.get_edge(parent_id, child_id, edge_type)
            if edge is None:
                continue
            if (
                edge.weight >= THETA_EDGE_ACTIVE
                and edge.inactive_time < TAU_EDGE_INACTIVE
            ):
                active = True
                break
        self._active_edge_memo[pair] = active
        return active

    def _compute_RF(self, concept: Any) -> None:
        total = concept.credit
        for child_id in concept.child_ids:
            child = self.cg.get_concept(child_id)
            if child is None:
                continue
            if (
                child.status in (ConceptStatus.CRYSTALLIZED, ConceptStatus.STABLE)
                and not getattr(child, "dormant", False)
                and not getattr(child, "soft_dormant", False)
                and self._is_active_edge(concept.id, child_id)
            ):
                total += child.RF
                if total >= RF_MAX:
                    break
        concept.RF = clamp(total, 0, RF_MAX)

    def _compute_GP(self, concept: Any) -> None:
        depth = concept.layer - 1 if concept.layer > 0 else 0
        branching = 0
        descendants = 0
        recent_births = 0
        for child_id in concept.child_ids:
            child = self.cg.get_concept(child_id)
            if (
                child
                and child.status
                in (
                    ConceptStatus.CRYSTALLIZED,
                    ConceptStatus.STABLE,
                )
                and not getattr(child, "dormant", False)
                and not getattr(child, "soft_dormant", False)
                and self._is_active_edge(concept.id, child_id)
            ):
                branching += 1
                if self.t - child.T_cryst <= TAU_RECENT:
                    recent_births += 1

        visited = set()
        queue: deque = deque()
        for cid in concept.child_ids:
            if self._is_active_edge(concept.id, cid):
                queue.append((cid, 1))
        while queue:
            cid, dist = queue.popleft()
            if dist > BFS_MAX or cid in visited:
                continue
            visited.add(cid)
            child = self.cg.get_concept(cid)
            if (
                child
                and child.status
                in (
                    ConceptStatus.CRYSTALLIZED,
                    ConceptStatus.STABLE,
                )
                and not getattr(child, "dormant", False)
                and not getattr(child, "soft_dormant", False)
            ):
                descendants += 1
                for gcid in child.child_ids:
                    if self._is_active_edge(cid, gcid):
                        queue.append((gcid, dist + 1))

        D_hat_max = max(D_MAX, 1)
        gp = (
            1
            + ALPHA * min(1.0, depth / D_hat_max)
            + BETA * (branching / (branching + 1))
            + GAMMA * (descendants / (descendants + 1))
            + DELTA * (recent_births / (recent_births + 1))
        )
        concept.GP = clamp(gp, 1.0, GP_MAX - EPSILON)

    def _compute_EF(self, concept: Any) -> None:
        concept.EF = clamp(concept.RF * concept.GP / GP_MAX, 0, RF_MAX)

    def _update_EF_eff_all(self) -> None:
        for cid, concept in self.cg.concepts.items():
            if concept.status == ConceptStatus.TOMBSTONED:
                concept.EF_eff = 0.0
                continue
            if not hasattr(concept, "CF") or concept.CF is None:
                concept.CF = {}
            cf_val = concept.CF.get(self.context_signature, 0.0)
            concept.EF_eff = clamp(concept.EF + ETA_CTX * cf_val, 0, RF_MAX)

    def _increment_CF(self, cid: int, context: int, increment: float) -> None:
        concept = self.cg.get_concept(cid)
        if concept is None:
            return
        if not hasattr(concept, "CF") or concept.CF is None:
            concept.CF = {}
        concept.CF[context] = clamp(concept.CF.get(context, 0.0) + increment, 0, CF_MAX)

    @staticmethod
    def _extract_basis(outcome: Optional[Dict[str, Any]]) -> List[int]:
        if not outcome:
            return []
        basis = outcome.get("basis")
        if basis is None:
            plan_obj = outcome.get("plan")
            if plan_obj is not None:
                basis = getattr(plan_obj, "basis_snapshot", [])
        return list(basis) if basis else []

    def _reverse_topological_order(self) -> List[int]:
        visited = set()
        result: List[int] = []
        for start_cid in self.cg.concepts:
            if start_cid in visited:
                continue
            stack = [(start_cid, False)]
            while stack:
                cid, processed = stack.pop()
                if processed:
                    result.append(cid)
                    continue
                if cid in visited:
                    continue
                visited.add(cid)
                stack.append((cid, True))
                concept = self.cg.get_concept(cid)
                if concept:
                    for child_id in concept.child_ids:
                        if child_id not in visited:
                            stack.append((child_id, False))
        return result

    def _update_step_blame_counters(
        self, t: int, plan_successes, plan_failures
    ) -> None:
        for failure in plan_failures:
            for cid in self._extract_basis(failure):
                concept = self.cg.get_concept(cid)
                if concept:
                    concept.step_F_count = getattr(concept, "step_F_count", 0) + 1
        for plan in plan_successes:
            for cid in self._extract_basis(plan):
                concept = self.cg.get_concept(cid)
                if concept:
                    concept.step_S_count = getattr(concept, "step_S_count", 0) + 1

    def _compute_blame_weight(self, cid: int) -> Optional[float]:
        concept = self.cg.get_concept(cid)
        if concept is None:
            return None
        F = getattr(concept, "step_F_count", 0)
        S = getattr(concept, "step_S_count", 0)
        if F + S < N_STEP_MIN:
            return None
        return (F + EPSILON) / (F + S + EPSILON)


# =============================================================================
# DIAGNOSTIC
# =============================================================================


def fertility_diagnostic(fe: FertilityEngine) -> Dict[str, Any]:
    diag: Dict[str, Any] = {
        "credit": {},
        "RF": {},
        "GP": {},
        "EF": {},
        "EF_combined": {},
        "CF_map_size": {},
    }
    for cid, concept in fe.cg.concepts.items():
        diag["credit"][cid] = concept.credit
        diag["RF"][cid] = concept.RF
        diag["GP"][cid] = concept.GP
        diag["EF"][cid] = concept.EF
        diag["EF_combined"][cid] = concept.EF_combined
        diag["CF_map_size"][cid] = len(concept.CF) if concept.CF else 0
    return diag
