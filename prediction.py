"""
prediction.py — Blueprint v31 §8: forward-looking estimator.

Three prediction subsystems plus a surprise signal, all sharing the same
validate-against-observation loop:

    Causal       — P(effect | action) via matched-decay counterfactuals.
    Analogical   — Fertile Evidence Registry (EBAF) prototype prior.
    Transfer     — reliability tracking for Mapper-produced transfers.
    Surprise     — alarm emitted when a prediction is contradicted.

Not owned by this module:
    CF decay on surprise (§8.6.3)       — a consumer action; FertilityEngine
                                          applies it based on drained signals.
    Edge weight modification            — explicitly forbidden by §8.6.3.
    Active subgraph construction        — §6.2, owned by the Mapper and
                                          consumed via set_situation_provider
                                          injection; a semantically identical
                                          local builder is retained for
                                          standalone operation.
    Reliability storage                 — §5.6/§8.9: lives on CoreEntry in
                                          the Core Library (injected via
                                          set_core_library); this module owns
                                          the update logic exclusively.

Determinism: all tie-breaking is by canonical hash; all iteration for
decision purposes is over sorted collections. Same input → byte-identical
output, run to run.

Spec divergences are marked [SPEC DIVERGENCE: ...] with the rationale.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from core.hashing import clamp  # noqa: F401
from core.schema import EdgeParams  # noqa: F401
import parameters as P
# [reorg Move 4b] Subgraph/anchor_positions_for/edge_params_for moved to
# core/graph_types.py; re-exported so rom prediction import Subgraph keeps
# working until the shim is deleted.
from core.graph_types import (  # noqa: F401
    Subgraph,
    anchor_positions_for,
    edge_params_for,
)

from sensory_floor import TypedSignature
from core.types import EdgeType

EPSILON = P.EPSILON


# =============================================================================
# ENUMS AND DATACLASSES
# =============================================================================


class PredictionKind(Enum):
    CAUSAL = auto()
    ANALOGICAL = auto()
    TRANSFER = auto()





@dataclass
class PrototypeEntry:
    """§8.4.1: one entry in the Fertile Evidence Registry."""

    concept_id: int
    sigma: TypedSignature
    evidence_set: Tuple[TypedSignature, ...]
    peak_fertility: float  # normalized F_P ∈ [0, 1]
    # t_last_confirmed is refreshed on every step the owning concept is
    # STABLE and above the EF threshold. Decay therefore measures recency
    # of confirmed relevance, not age since peak. A prototype continuously
    # confirmed fertile carries decay ≈ 1 regardless of when its peak
    # occurred.
    t_last_confirmed: int
    # [Batch D / P1-1] Endogenous confirmation checkpoint: graph_revision()
    # at the last confirmation refresh (OR-of-bounds decay, HIGH-03).
    last_rev_confirmed: int = 0


@dataclass
class TransferPrediction:
    """§8.5.1: a pending transfer prediction awaiting validation."""

    core_hash: bytes
    predicted_consequence: Subgraph
    confidence: float
    t_predicted: int
    horizon: int
    # [F2] Environment effect atoms this transfer predicts. Populated at
    # registration from the source concept's full binding set; optionally
    # narrowed to the specific atom of the proposed action.
    effect_atoms: FrozenSet[Any] = frozenset()
    # [Batch B] Snapshot of atoms already present at t_predicted.
    # Intersected with effect_atoms at validation time (not here) so
    # repeated narrowing preserves the snapshot whole.
    effect_atoms_present_at_prediction: FrozenSet[Any] = frozenset()


@dataclass
class SurpriseSignal:
    """§8.6.2: emitted when a prediction is contradicted."""

    kind: PredictionKind
    subject: Any  # core_hash, (action, effect) key, or concept_id
    predicted: float
    observed: float
    t_emitted: int
    scope: Subgraph
    # [SPEC COMPLIANCE §8.6.1] Pre-computed surprise magnitude. Downstream
    # consumers (Builder's EPISTEMIC_EXCURSION priority, future CF decay
    # application in FertilityEngine) read this value directly; the formula
    # is owned by this module and never re-derived downstream.
    surprise: float = 0.0


# =============================================================================
# PREDICTION MODULE
# =============================================================================


class PredictionModule:
    """Forward-looking estimator. Called every step for causal and transfer
    validation; periodically for analogical registration and decay.

    All constitutional constants are bound from `parameters` (single source
    of truth); the class attributes exist for interface stability only.

    Wiring (performed by main/pipeline at construction time):
        set_core_library(lib)        — authoritative CoreEntry.reliability
                                       storage (§5.6/§8.9: reliability lives on
                                       CoreEntry; this module owns the updates).
        set_situation_provider(fn)   — §6.2 situation subgraph builder owned
                                       by the Mapper (fn(t) -> Subgraph).
        set_containment_checker(fn)  — §6.4.1 typed-constrained MCS
                                       containment (fn(query, target) -> bool,
                                       θ_containment applied inside).
    """

    # ---- Causal parameters (§8.8) ----
    GAMMA_CF = P.GAMMA_CF
    W_CONFIRM = P.W_CONFIRM
    RHO_SMOOTH = P.RHO_SMOOTH
    N_CF_MIN = P.N_CF_MIN
    EPS_HYST_MIN = P.EPS_HYST_NEC_MIN
    EPS_HYST_MAX = P.EPS_HYST_NEC_MAX
    C_HYST = P.C_HYST
    # [Batch D1] Tiered necessity commit (§8.3.3). Bound from parameters
    # like every other constitutional constant; class attributes exist for
    # interface stability. See the §22 block in parameters.py for the
    # rationale (sparse / one-shot effects never reach N_CF_MIN).
    THETA_CF_COMMIT = P.THETA_CF_COMMIT
    THETA_CONTRAST_COMMIT = P.THETA_CONTRAST_COMMIT
    THETA_CONTRAST_STABLE = P.THETA_CONTRAST_STABLE

    # ---- Analogical parameters (§8.8) ----
    THETA_PROTO_EF = P.THETA_PROTO_EF
    N_PROTO_MAX_EBAF = P.N_PROTO_MAX_EBAF
    TAU_PROTO_DECAY = P.TAU_PROTO_DECAY

    # ---- Transfer parameters (§8.8) ----
    KAPPA_REL = P.KAPPA_REL
    W_VALIDATION = P.W_VALIDATION
    MAX_PENDING = P.MAX_PENDING
    W_STALE = P.W_STALE

    # ---- Surprise parameters (§8.8) ----
    THETA_SURPRISE = P.THETA_SURPRISE
    MAX_SURPRISE_PER_STEP = P.MAX_SURPRISE_PER_STEP
    MAX_SURPRISE_QUEUE = P.MAX_SURPRISE_QUEUE

    # ---- Active-subgraph construction (§6.2, fallback path) ----
    H_SITUATION = P.H_SITUATION
    THETA_SITUATION_EDGE = P.THETA_SITUATION_EDGE
    MAX_SITUATION_SIZE = P.MAX_SITUATION_SIZE

    def __init__(self) -> None:
        # ---------------- Causal subsystem state ----------------
        self.C_opp: Dict[Any, float] = defaultdict(float)
        self.C_opp_not: Dict[Any, float] = defaultdict(float)
        self.C_succ_a: Dict[Tuple[Any, Any], float] = defaultdict(float)
        self.C_succ_real_total: Dict[Any, float] = defaultdict(float)
        self.resolved_count: Dict[Any, float] = defaultdict(float)
        self.resolved_real_total: float = 0.0

        self.prev_bar_p_a: Dict[Tuple[Any, Any], float] = defaultdict(float)
        self.prev_bar_p_not_a: Dict[Tuple[Any, Any], float] = defaultdict(float)
        self.flag_init_p_a: Dict[Tuple[Any, Any], bool] = defaultdict(bool)
        self.flag_init_p_not_a: Dict[Tuple[Any, Any], bool] = defaultdict(bool)
        self.bar_dp_dict: Dict[Tuple[Any, Any], float] = defaultdict(float)
        self.prev_H_nec: Dict[Tuple[Any, Any], int] = defaultdict(int)
        self.Gate_bind: Dict[Tuple[Any, Any], int] = defaultdict(int)
        self.Gate_undo: Dict[Tuple[Any, Any], int] = defaultdict(int)

        self.effect_flags: Set[Any] = set()

        # ---------------- Analogical subsystem state ----------------
        self.fertile_evidence_registry: List[PrototypeEntry] = []

        # ---------------- Transfer subsystem state ----------------
        self.pending_transfers: Dict[bytes, List[TransferPrediction]] = defaultdict(
            list
        )
        self.pending_stale_evicted: int = 0
        # Validated transfer outcomes awaiting consumer drain (pipeline →
        # FertilityEngine plan-success/step-blame events; UtilityEngine
        # goal-satiety). Bounded: drained every step by the pipeline.
        self._validated_outcomes: deque = deque()
        # [MACRO-TRANSFER DIAGNOSTIC] cumulative validation counters.
        self.transfers_validated_this_run: int = 0
        self.transfers_observed_true: int = 0
        self.transfers_observed_false: int = 0

        # ---------------- Injected wiring (see class docstring) -------------
        self._core_library: Optional[Any] = None
        self._core_kind_action: Optional[Any] = None
        self._core_kind_category: Optional[Any] = None
        self._situation_provider: Optional[Any] = None
        self._containment_checker: Optional[Any] = None
        # [Batch B] Pipeline's effect-presence register. Never None.
        # An unwired run produces empty snapshots rather than crashing.
        # set_effect_presence_register REPLACES the reference with the
        # pipeline's authoritative dict. The pipeline MUST rebind this
        # via dict.clear() on reset, never via reassignment — otherwise
        # this reference goes stale.
        self._effect_presence_register: Dict[int, bool] = {}
        # [F2] Rising-edge history for transfer validation. Sized to
        # (W_STALE + 2*W_VALIDATION) * 16 events.
        _rise_capacity = (P.W_STALE + 2 * P.W_VALIDATION) * 16
        self._rise_history: deque = deque(maxlen=_rise_capacity)
        # [F2v9 4.2] Pair-level record for the rise-positive gate:
        # (rise_t, t_predicted) appended on every rise-validated transfer.
        self.rise_validation_pairs: List[Tuple[int, int]] = []

        # ---------------- Surprise queue ----------------
        self._surprise_queue: deque = deque()
        self._surprise_overflow: List[SurpriseSignal] = []
        self._current_step: int = -1
        self._surprises_this_step: int = 0
        self._diagnostics: Dict[str, int] = defaultdict(int)

    # -------------------------------------------------------------------------
    # Wiring injection (called once at construction time by main/pipeline)
    # -------------------------------------------------------------------------

    def set_core_library(self, core_library: Any) -> None:
        """§8.9: reliability storage lives on CoreEntry in the Core Library;
        this module owns the update logic. Importing CoreKind lazily keeps
        the Milestone-C build order (prediction before consolidator)."""
        from consolidator import CoreKind  # noqa: PLC0415 (wiring-time import)

        self._core_library = core_library
        self._core_kind_action = CoreKind.ACTION
        self._core_kind_category = CoreKind.CATEGORY

    def set_situation_provider(self, provider: Any) -> None:
        """§6.2 situation subgraph builder (Mapper.build_situation_subgraph)."""
        self._situation_provider = provider

    def set_containment_checker(self, checker: Any) -> None:
        """§6.4.1 typed-constrained MCS containment at θ_containment."""
        self._containment_checker = checker

    def set_effect_presence_register(self, register: Dict[int, bool]) -> None:
        """[Batch B] Direct reference to the pipeline's effect-presence
        register {atom: bool}. Read by the persistence fallback in
        transfer validation. The pipeline must not rebind its register
        after this call — only dict.clear() on reset."""
        self._effect_presence_register = register

    def notify_effect_rise(self, atom: Any, t: int) -> None:
        """[F2] Record an environment effect rising edge."""
        self._rise_history.append((t, atom))

    def _lookup_core_entry(self, core_hash: bytes) -> Optional[Any]:
        """Resolve a CoreEntry by canonical hash. Transfers flow from action
        queries (§6.6.1), so ACTION is checked first; CATEGORY second
        (composite key per §5.5)."""
        if self._core_library is None:
            return None
        by_hash = getattr(self._core_library, "entries_by_hash", None)
        if by_hash is None:
            return None
        for kind in (self._core_kind_action, self._core_kind_category):
            idx = by_hash.get((core_hash, kind))
            if idx is not None:
                return self._core_library.entries[idx]
        return None

    # -------------------------------------------------------------------------
    # Step advancement
    # -------------------------------------------------------------------------

    def _maybe_advance_step(self, t: int) -> None:
        """Reset per-step counters when a new step is entered. Drains
        overflow into the main queue up to the queue cap."""
        if t == self._current_step:
            return
        self._current_step = t
        self._surprises_this_step = 0

        # Drain overflow into the main queue, respecting the global cap.
        while (
            self._surprise_overflow
            and len(self._surprise_queue) < self.MAX_SURPRISE_QUEUE
        ):
            self._surprise_queue.append(self._surprise_overflow.pop(0))

    # =========================================================================
    # CAUSAL PREDICTION (§8.3)
    # =========================================================================

    def step_causal(
        self,
        t: int,
        a_hist: Dict[int, Any],
        onset_history: Dict[Any, Dict[int, int]],
        real_actions: Sequence[Any],
        effect_flags: Sequence[Any],
    ) -> None:
        """Streaming counterfactual update with matched decay. Called every
        step after action execution.

        ONSET SEMANTICS (§8.3.1, auditor ruling R5-O1): "success counters
        ... detect whether `b` FIRED within the window" — FIRED means an
        Onset detection (rising edge, 0→1 transition of the effect flag),
        never passive flag presence. `onset_history[b][k]` is the Pipeline's
        rising-edge record (1 iff v_b(k)=1 ∧ v_b(k−1)=0). Consequence for
        persistent LEVEL effects (e.g. an open-door flag that stays high):
        trials whose window contains no rising edge earn χ=0 even while the
        flag is passively high, so ¬a trials accrue no C_succ and
        bar_ΔP = bar_p_a − bar_p_not_a is PRESERVED (→ 1.0 − 0.0 for a
        reliable cause). A presence-based χ would credit every trial while
        the flag is high, driving p_not_a → 1.0, bar_ΔP → 0.0 and collapsing
        H_nec/gate_undo — the exact O1 failure mode this semantics forbids.
        """
        self._maybe_advance_step(t)

        # Global set of effects ever observed (§2.6.1).
        for b in effect_flags:
            self.effect_flags.add(b)

        all_effects: List[Any] = list(self.effect_flags)
        # Contract (§8.7): `real_actions` is the caller-filtered set of real
        # motor actions (the pipeline excludes no-ops/probes/teleological
        # actions). No hardcoded action-name filtering happens here — the
        # former NO_OP_ACTIONS = {"null", "stay"} literal was env-specific
        # name semantics and has been removed.
        real_actions = list(real_actions)

        gamma = self.GAMMA_CF
        w_confirm = self.W_CONFIRM

        # 1. Decay ALL tracked counters exactly once.
        # Iterating over all keys prevents counters from freezing when an
        # action or effect drops out of the active repertoire.
        for a in list(self.C_opp.keys()):
            self.C_opp[a] *= gamma
            if self.C_opp[a] < EPSILON:
                del self.C_opp[a]

        for a in list(self.C_opp_not.keys()):
            self.C_opp_not[a] *= gamma
            if self.C_opp_not[a] < EPSILON:
                del self.C_opp_not[a]

        for pair in list(self.C_succ_a.keys()):
            self.C_succ_a[pair] *= gamma
            if self.C_succ_a[pair] < EPSILON:
                del self.C_succ_a[pair]

        for b in list(self.C_succ_real_total.keys()):
            self.C_succ_real_total[b] *= gamma
            if self.C_succ_real_total[b] < EPSILON:
                del self.C_succ_real_total[b]

        # 2. Opportunity counters, updated from a_{t-1}.
        prev_a = a_hist.get(t - 1)
        if prev_a in real_actions:
            for a in real_actions:
                if prev_a == a:
                    self.C_opp[a] += 1.0
                else:
                    self.C_opp_not[a] += 1.0

        # 3. Resolve the trial that fired w_confirm steps ago.
        # χ (C_succ credit) is ONSET-based, not level-based (ruling R5-O1):
        # the trial at j succeeds for effect b iff b's rising edge was
        # detected in [j+1, j+w_confirm]. A flag merely STAYING high inside
        # the window earns nothing — an action caused the effect iff the
        # effect transitioned 0→1 after it.
        resolutions: List[Tuple[Any, Any, int]] = []
        if t >= w_confirm:
            j = t - w_confirm
            acted = a_hist.get(j)
            if acted in real_actions:
                self.resolved_real_total += 1.0
                self.resolved_count[acted] += 1.0
                for b in all_effects:
                    chi = 0
                    b_onsets = onset_history.get(b, {})
                    for k in range(j + 1, j + w_confirm + 1):
                        if b_onsets.get(k, 0) == 1:
                            chi = 1
                            break
                    increment = (gamma ** (w_confirm - 1)) * chi
                    self.C_succ_a[(acted, b)] += increment
                    self.C_succ_real_total[b] += increment
                    resolutions.append((acted, b, chi))

        # Snapshot old predictions BEFORE updating with current step's
        # observations to prevent temporal contamination of surprise.
        old_bar_p_a = dict(self.prev_bar_p_a)

        # 4. Probability estimates, filtering, hysteresis, gates.
        for a in real_actions:
            for b in all_effects:
                c_succ_not = max(0.0, self.C_succ_real_total[b] - self.C_succ_a[(a, b)])
                p_a = clamp(
                    self.C_succ_a[(a, b)] / (self.C_opp[a] + EPSILON),
                    0.0,
                    1.0,
                )
                p_not_a = clamp(
                    c_succ_not / (self.C_opp_not[a] + EPSILON),
                    0.0,
                    1.0,
                )

                # §8.3.3: N_cf = min(C_opp[a], C_opp_not[a]). The
                # resolved-trial counters (resolved_count /
                # resolved_real_total) are maintained as migrated
                # counterfactual state (§8.9) and exposed via
                # check_necessity; the gates below use N_cf exactly as
                # specified (the former Gate_bind used an off-spec
                # resolved-count variant).
                N_cf = min(self.C_opp[a], self.C_opp_not[a])

                if self.C_opp[a] > 0 and not self.flag_init_p_a[(a, b)]:
                    self.prev_bar_p_a[(a, b)] = p_a
                    self.flag_init_p_a[(a, b)] = True
                if self.C_opp_not[a] > 0 and not self.flag_init_p_not_a[(a, b)]:
                    self.prev_bar_p_not_a[(a, b)] = p_not_a
                    self.flag_init_p_not_a[(a, b)] = True

                bar_p_a = (
                    self.RHO_SMOOTH * self.prev_bar_p_a[(a, b)]
                    + (1.0 - self.RHO_SMOOTH) * p_a
                )
                bar_p_not_a = (
                    self.RHO_SMOOTH * self.prev_bar_p_not_a[(a, b)]
                    + (1.0 - self.RHO_SMOOTH) * p_not_a
                )
                bar_dp = bar_p_a - bar_p_not_a

                self.prev_bar_p_a[(a, b)] = bar_p_a
                self.prev_bar_p_not_a[(a, b)] = bar_p_not_a
                self.bar_dp_dict[(a, b)] = bar_dp

                # [Batch D1] Tiered necessity commit. N_cf grows only as
                # fast as the action is executed, while the contrast
                # evidence for the link can be decisive after one
                # observation; demanding N_cf >= 5 before H_nec may be 1
                # therefore discarded the strongest evidence available on
                # sparse / one-shot effects (the tool's effects are
                # one-shot: N_cf never reaches 5 within an episode). The
                # commit point is THETA_CF_COMMIT; the required contrast
                # ramps in sqrt(N_cf) from THETA_CONTRAST_COMMIT to
                # THETA_CONTRAST_STABLE at N_CF_MIN, so a low-N_cf commit
                # must clear the HIGHER end of the contrast range first.
                # Gate_bind / Gate_undo below are unchanged (they still
                # demand N_cf >= N_CF_MIN).
                if N_cf < self.THETA_CF_COMMIT:
                    H_nec = 0
                else:
                    eps_hyst = min(
                        0.2,
                        max(
                            self.EPS_HYST_MIN,
                            self.C_HYST / math.sqrt(max(N_cf, 1.0)),
                        ),
                    )
                    span = max(self.N_CF_MIN - self.THETA_CF_COMMIT, 1e-9)
                    progress = math.sqrt(
                        max(
                            0.0,
                            min(1.0, (N_cf - self.THETA_CF_COMMIT) / span),
                        )
                    )
                    theta_dp = (
                        self.THETA_CONTRAST_COMMIT
                        + (
                            self.THETA_CONTRAST_STABLE
                            - self.THETA_CONTRAST_COMMIT
                        )
                        * progress
                    )
                    enter = bar_dp >= theta_dp
                    stay = bar_dp >= theta_dp - eps_hyst
                    H_nec = (
                        1 if (stay if self.prev_H_nec[(a, b)] == 1 else enter) else 0
                    )

                self.prev_H_nec[(a, b)] = H_nec
                # §8.3.3: Gate_bind = 1 iff H_nec ∨ N_cf < N_cf_min.
                self.Gate_bind[(a, b)] = 1 if (H_nec or N_cf < self.N_CF_MIN) else 0
                self.Gate_undo[(a, b)] = (
                    1 if (H_nec and N_cf >= self.N_CF_MIN and bar_dp >= 0.40) else 0
                )

        # 5. Causal surprise for resolved trials.
        scope = self._current_situation_scope(t)
        for acted, b, chi in resolutions:
            # Use the pre-update prediction to avoid contamination.
            pred = old_bar_p_a.get((acted, b), 0.0)
            surprise = _causal_surprise(pred, chi)
            if surprise >= self.THETA_SURPRISE:
                self._emit_surprise(
                    SurpriseSignal(
                        kind=PredictionKind.CAUSAL,
                        subject=(acted, b),
                        predicted=pred,
                        observed=float(chi),
                        t_emitted=t,
                        # §8.6.2: scope = the region where the violation
                        # occurred — the current situation subgraph (§6.2)
                        # when the Mapper provider is wired; empty otherwise
                        # (the Builder counts and drops empty-scope signals).
                        scope=scope if scope is not None else Subgraph(),
                        surprise=surprise,
                    )
                )

    def _current_situation_scope(self, t: int) -> Optional[Subgraph]:
        """§6.2 situation subgraph via the injected Mapper provider."""
        if self._situation_provider is None:
            return None
        return self._situation_provider(t)

    # ---- Causal accessors --------------------------------------------------

    def causal_predict(self, action: Any, effect: Any) -> float:
        return self.prev_bar_p_a.get((action, effect), 0.0)

    def causal_contrast(self, action: Any, effect: Any) -> float:
        return self.bar_dp_dict.get((action, effect), 0.0)

    def gate_bind(self, action: Any, effect: Any) -> bool:
        if action is None or effect is None:
            return False
        return bool(self.Gate_bind.get((action, effect), 1))

    def gate_undo(self, action: Any, effect: Any) -> bool:
        if action is None or effect is None:
            return False
        return bool(self.Gate_undo.get((action, effect), 0))

    def check_necessity(self, action: Any, effect: Any) -> Dict[str, Any]:
        """§8.9 migration: legacy necessity adapter (moved from
        concept_graph.check_necessity). Read-only view over the §8.3
        counterfactual state for a single (action, effect) pair, including
        the resolved-trial counters (resolved_count / resolved_real_total)
        migrated alongside step_counterfactual."""
        pair = (action, effect)
        c_opp = self.C_opp.get(action, 0.0)
        c_opp_not = self.C_opp_not.get(action, 0.0)
        res_a = self.resolved_count.get(action, 0.0)
        return {
            "N_cf": min(c_opp, c_opp_not),
            "N_cf_resolved": min(res_a, max(0.0, self.resolved_real_total - res_a)),
            "bar_p_a": self.prev_bar_p_a.get(pair, 0.0),
            "bar_p_not_a": self.prev_bar_p_not_a.get(pair, 0.0),
            "bar_dp": self.bar_dp_dict.get(pair, 0.0),
            "H_nec": int(self.prev_H_nec.get(pair, 0)),
            "gate_bind": bool(self.Gate_bind.get(pair, 1)),
            "gate_undo": bool(self.Gate_undo.get(pair, 0)),
        }

    # =========================================================================
    # ANALOGICAL PREDICTION (§8.4)
    # =========================================================================

    @staticmethod
    def _graph_rev(concept_graph: Any) -> int:
        """[Batch D / P1-1] Endogenous graph revision, 0 if unavailable."""
        fn = getattr(concept_graph, "graph_revision", None)
        return int(fn()) if callable(fn) else 0

    def step_analogical_register(self, t: int, concept_graph: Any) -> None:
        """Register or refresh prototypes for STABLE concepts.

        [SPEC DEVIATION §8.4.2] Eligibility gates on plain EF, not
        EF_combined. EF_combined includes the analogical prior the
        registry itself produces; gating on it creates a positive-
        feedback loop. The parameter name THETA_PROTO_EF matches the
        corrected gate. Peak fertility is likewise EF-normalized.
        """
        self._maybe_advance_step(t)
        concept_ids = sorted(concept_graph.concepts.keys())
        for cid in concept_ids:
            concept = concept_graph.concepts[cid]
            status_name = getattr(concept.status, "name", str(concept.status))
            if status_name != "STABLE":
                continue

            # [SPEC DEVIATION] Gate on plain EF, not EF_combined. EF_combined
            # includes the analogical prior that the registry itself produces,
            # creating a positive feedback loop. The parameter name
            # THETA_PROTO_EF says "EF", not "EF_combined".
            # peak_fertility also uses plain EF to avoid contaminating the
            # prototype's fertility with the analogical prior it helped produce.
            ef = getattr(concept, "EF", 0.0)
            if ef < self.THETA_PROTO_EF * P.RF_MAX:
                continue
            f_p = ef / P.RF_MAX
            entry = self._find_prototype(cid)
            if entry is None:
                evidence_set = tuple(evid[1] for evid in concept.evidence_registry)
                self.fertile_evidence_registry.append(
                    PrototypeEntry(
                        concept_id=cid,
                        sigma=concept.sigma,
                        evidence_set=evidence_set,
                        peak_fertility=f_p,  # f_p is now plain EF / RF_MAX, not EF_combined
                        t_last_confirmed=t,
                        last_rev_confirmed=self._graph_rev(concept_graph),
                    )
                )
            else:
                if f_p > entry.peak_fertility:
                    entry.peak_fertility = f_p
                entry.t_last_confirmed = t
                entry.last_rev_confirmed = self._graph_rev(concept_graph)
                entry.evidence_set = tuple(
                    evid[1] for evid in concept.evidence_registry
                )

        if len(self.fertile_evidence_registry) > self.N_PROTO_MAX_EBAF:
            self._evict_prototypes(t)

    def step_analogical_decay(self, t: int, concept_graph: Any = None) -> None:
        self._maybe_advance_step(t)
        # [Batch D / P1-1 HIGH-03] OR-of-bounds expiry: an entry expires on
        # step age OR on endogenous graph drift since its last confirmation.
        # ADAPTATION NOTE: the tree's peak*decay >= THETA_PROTO_EF filter is
        # RETAINED (tree wins on formulation); the revision bound is added,
        # not substituted — the draft's pure-expiry body predates the
        # decay-threshold shape that shipped.
        graph_rev = (
            concept_graph.graph_revision()
            if concept_graph is not None
            and hasattr(concept_graph, "graph_revision")
            else None
        )
        # [Batch D / P1-1 adaptation] Revision expiry is size-normalized
        # (turnover units), mirroring the Builder's _graph_drift: raw
        # deltas fire far too fast on churn-heavy graphs (measured ~27
        # revisions/step on rooms post-Batch-E).
        graph_size = 1
        if graph_rev is not None:
            graph_size = max(
                1,
                len(getattr(concept_graph, "edges", {}) or {})
                + len(concept_graph.get_live_pattern_node_ids()),
            )
        survivors: List[PrototypeEntry] = []
        for entry in self.fertile_evidence_registry:
            if (t - entry.t_last_confirmed) >= self.TAU_PROTO_DECAY:
                continue
            if (
                graph_rev is not None
                and (graph_rev - entry.last_rev_confirmed) / graph_size
                >= P.TAU_PROTO_DECAY_REVISION
            ):
                continue
            decay = math.exp(-(t - entry.t_last_confirmed) / self.TAU_PROTO_DECAY)
            if entry.peak_fertility * decay >= self.THETA_PROTO_EF:
                survivors.append(entry)
        self.fertile_evidence_registry = survivors

    def analogical_prior(
        self,
        evidence_set: Sequence[TypedSignature],
        t: int,
    ) -> float:
        """§8.4.3 analogical fertility prior.

        PF(q) = max over prototypes p of:
                    Sim_EBAF(q, p) · p.peak_fertility · decay(t − p.t_last_confirmed)

        Sim_EBAF uses TypedSignature.similarity (§2.21.5), not MCS.
        """
        if not self.fertile_evidence_registry or not evidence_set:
            return 0.0
        max_pf = 0.0
        for proto in self.fertile_evidence_registry:
            sim_max = 0.0
            for qsig in sorted(evidence_set, key=lambda s: s.serialize()):
                for psig in proto.evidence_set:
                    s = qsig.similarity(psig)
                    if s > sim_max:
                        sim_max = s
            # Decay is recency of confirmed relevance, not age of peak.
            # See PrototypeEntry.t_last_confirmed.
            decay = math.exp(-(t - proto.t_last_confirmed) / self.TAU_PROTO_DECAY)
            pf = sim_max * proto.peak_fertility * decay
            if pf > max_pf:
                max_pf = pf
        return max_pf

    def validate_analogical(
        self,
        concept_id: int,
        evidence_set: Sequence[TypedSignature],
        rf_obs_normalized: float,
        t: int,
    ) -> None:
        """Validate observed fertility against the analogical prior and emit
        Analogical surprise when the prediction is contradicted (§8.6.1).

        [SPEC NOTE §8.6.1] The spec formula is
            surprise = |PF_pred · RF_MAX − RF_obs| / RF_MAX.
        Dividing through by RF_MAX makes it algebraically identical to
            surprise = |PF_pred − RF_obs / RF_MAX|,
        i.e. this implementation with rf_obs_normalized = RF_obs / RF_MAX is
        the spec formula, not a divergence: both terms end up on the same
        normalized scale. Callers (FertilityEngine) hold RF_obs on the RF
        scale and pass the normalized quotient.

        Parameter rf_obs_normalized must be in [0, 1].
        """
        self._maybe_advance_step(t)
        if not (0.0 <= rf_obs_normalized <= 1.0):
            raise ValueError(
                f"rf_obs_normalized must be in [0, 1], got {rf_obs_normalized}"
            )
        pf_pred = self.analogical_prior(evidence_set, t)
        surprise = abs(pf_pred - rf_obs_normalized)
        if surprise >= self.THETA_SURPRISE:
            self._emit_surprise(
                SurpriseSignal(
                    kind=PredictionKind.ANALOGICAL,
                    subject=concept_id,
                    predicted=pf_pred,
                    observed=rf_obs_normalized,
                    t_emitted=t,
                    scope=Subgraph(),
                    surprise=surprise,
                )
            )

    # ---- Analogical helpers ------------------------------------------------

    def _find_prototype(self, concept_id: int) -> Optional[PrototypeEntry]:
        for entry in self.fertile_evidence_registry:
            if entry.concept_id == concept_id:
                return entry
        return None

    def _evict_prototypes(self, t: int) -> None:
        def score(entry: PrototypeEntry) -> Tuple[float, bytes]:
            decay = math.exp(-(t - entry.t_last_confirmed) / self.TAU_PROTO_DECAY)
            # Tie-break by canonical bytes.
            return (entry.peak_fertility * decay, entry.sigma.serialize())

        self.fertile_evidence_registry.sort(key=score)
        excess = len(self.fertile_evidence_registry) - self.N_PROTO_MAX_EBAF
        if excess > 0:
            self.fertile_evidence_registry = self.fertile_evidence_registry[excess:]

    # =========================================================================
    # TRANSFER PREDICTION (§8.5)
    # =========================================================================

    def register_transfer(
        self,
        core_hash: bytes,
        predicted_consequence: Subgraph,
        confidence: float,
        t_predicted: int,
        horizon: int,
        effect_atoms: FrozenSet[Any] = frozenset(),
    ) -> None:
        # §8.5.1 registration is O(1) and part of the Mapper→Prediction
        # contract; the core library must be wired at construction time
        # (main.py). A missing library is a wiring defect: fail loudly
        # rather than tracking reliability in an orphaned private cache.
        if self._core_library is None:
            raise RuntimeError(
                "PredictionModule.register_transfer called before "
                "set_core_library(); reliability storage lives on CoreEntry "
                "(§5.6/§8.9). Wire the core library at construction time."
            )
        self._maybe_advance_step(t_predicted)
        # [Batch B] Snapshot atoms already present at prediction time.
        # Distinguishes "already true" (persistence fallback) from
        # "turned on during the window" (rise path). Counter fires when
        # a transfer predicted effects that were all absent at
        # registration — the NORMAL case for causal transfers whose
        # effect has not yet fired. Diagnostic-only; not an error.
        if effect_atoms:
            present_snapshot = frozenset(
                a for a in effect_atoms
                if self._effect_presence_register.get(a, False)
            )
            if not present_snapshot:
                self._diagnostics["transfer_register_atoms_all_absent"] += 1
        else:
            present_snapshot = frozenset()
        self.pending_transfers[core_hash].append(
            TransferPrediction(
                core_hash,
                predicted_consequence,
                confidence,
                t_predicted,
                horizon,
                effect_atoms=effect_atoms,
                effect_atoms_present_at_prediction=present_snapshot,
            )
        )
        total = sum(len(v) for v in self.pending_transfers.values())
        while total > self.MAX_PENDING:
            oldest_key, oldest_time = None, None
            for kh, lst in self.pending_transfers.items():
                if lst and (oldest_time is None or lst[0].t_predicted < oldest_time):
                    oldest_time = lst[0].t_predicted
                    oldest_key = kh
            if oldest_key is None:
                break
            self.pending_transfers[oldest_key].pop(0)
            if not self.pending_transfers[oldest_key]:
                del self.pending_transfers[oldest_key]
            total -= 1

    def step_validate_transfers(self, t: int, concept_graph: Any) -> None:
        """§8.5.2/§8.11 step 8b — validate due transfers against the current
        situation subgraph (§6.2) using typed-constrained MCS containment
        (§6.4.1, θ_containment). Called every step."""
        self._maybe_advance_step(t)

        # [F2v9 4.6] Time-based pruning of the rise history and the pair
        # record. Entries older than the staleness window can never match
        # a due predicate (t_predicted + horizon + extra <= t) for any
        # still-pending transfer, so they are dead weight.
        cutoff = t - (2 * self.W_VALIDATION + self.W_STALE)
        if self._rise_history and self._rise_history[0][0] < cutoff:
            kept = [e for e in self._rise_history if e[0] >= cutoff]
            self._rise_history.clear()
            self._rise_history.extend(kept)
        if self.rise_validation_pairs:
            self.rise_validation_pairs = [
                (rt, t_pred_pair)
                for (rt, t_pred_pair) in self.rise_validation_pairs
                if rt >= cutoff
            ]

        # Staleness pruning (§8.5.1).
        stale_keys = []
        for kh, lst in self.pending_transfers.items():
            still_pending = [
                tp for tp in lst if tp.t_predicted + tp.horizon >= t - self.W_STALE
            ]
            self.pending_stale_evicted += len(lst) - len(still_pending)
            if still_pending:
                self.pending_transfers[kh] = still_pending
            else:
                stale_keys.append(kh)
        for kh in stale_keys:
            del self.pending_transfers[kh]

        due_by_key: List[Tuple[bytes, TransferPrediction]] = []
        for kh, lst in self.pending_transfers.items():
            for tp in lst:
                extra = 1 if tp.effect_atoms else 0
                if tp.t_predicted + tp.horizon + extra <= t:
                    due_by_key.append((kh, tp))
        if not due_by_key:
            return

        if self._containment_checker is None:
            raise RuntimeError(
                "PredictionModule.step_validate_transfers requires the "
                "typed-constrained MCS containment checker (§6.4.1) — call "
                "set_containment_checker(mapper.mcs_containment) at wiring time."
            )

        _active_cache: List[Any] = [None]

        def _get_active() -> Any:
            if _active_cache[0] is None:
                _active_cache[0] = self._active_subgraph(t, concept_graph)
            return _active_cache[0]

        for kh, tp in due_by_key:
            lst = self.pending_transfers.get(kh)
            if lst is not None and tp in lst:
                lst.remove(tp)
                if not lst:
                    del self.pending_transfers[kh]
            if tp.effect_atoms:
                # [F2v9 4.8] Rise validation: the first rising edge after
                # t_predicted matching any bound effect atom validates the
                # transfer. Deterministic: history order = emission order.
                matched_rise = next(
                    (
                        (rise_t, rise_atom)
                        for (rise_t, rise_atom) in self._rise_history
                        if (rise_t > tp.t_predicted)
                        and (rise_atom in tp.effect_atoms)
                    ),
                    None,
                )
                if matched_rise is not None:
                    observed = True
                    self._diagnostics["transfer_rise_validated"] += 1
                    self.rise_validation_pairs.append(
                        (matched_rise[0], tp.t_predicted)
                    )
                else:
                    # [Batch B] Persistence fallback. Atom must have been
                    # present at prediction time AND still be present
                    # now. Snapshot is intersected HERE with the current
                    # (possibly narrowed) effect_atoms; narrowing itself
                    # carries the snapshot whole.
                    currently_present = frozenset(
                        a for a in tp.effect_atoms
                        if self._effect_presence_register.get(a, False)
                    )
                    persisted = (
                        tp.effect_atoms_present_at_prediction
                        & tp.effect_atoms
                        & currently_present
                    )
                    if persisted:
                        observed = True
                        self._diagnostics[
                            "transfer_validated_by_persistence"
                        ] += 1
                    else:
                        self._diagnostics["transfer_atoms_no_rise"] += 1
                        mcs_obs = bool(
                            self._containment_checker(
                                tp.predicted_consequence, _get_active()
                            )
                        )
                        self._diagnostics[
                            "transfer_mcs_disagreement"
                        ] += int(mcs_obs)
                        observed = False
            else:
                self._diagnostics["transfer_no_atoms"] += 1
                observed = bool(
                    self._containment_checker(
                        tp.predicted_consequence, _get_active()
                    )
                )
            self._update_reliability_entry(tp.core_hash, observed)
            # Expose the outcome for consumers (pipeline → FertilityEngine
            # plan-success/step-blame events; §8.5.3 closes the loop).
            self._validated_outcomes.append(
                (tp.core_hash, observed, t, tp.t_predicted)
            )
            if observed:
                self.transfers_observed_true += 1
            else:
                self.transfers_observed_false += 1
            surprise = abs(tp.confidence - (1.0 if observed else 0.0))
            if surprise >= self.THETA_SURPRISE:
                self._emit_surprise(
                    SurpriseSignal(
                        PredictionKind.TRANSFER,
                        tp.core_hash,
                        tp.confidence,
                        1.0 if observed else 0.0,
                        t,
                        tp.predicted_consequence,
                        surprise=surprise,
                    )
                )

    def drain_validated_transfers(self) -> List[Tuple[bytes, bool, int, int]]:
        """Return and clear validated transfer outcomes including t_predicted."""
        out = list(self._validated_outcomes)
        self.transfers_validated_this_run += len(out)
        self._validated_outcomes.clear()
        return out

    def get_reliability(self, core_hash: bytes) -> float:
        """§8.5.4 — reliability as stored on the CoreEntry (§5.6). Unknown
        hashes return κ_init × 1.0 = the v31 initial-reliability invariant
        ('all extracted cores have initial_reliability = 0.8, which exceeds
        τ_transfer = 0.4'), so a fresh core can transfer on its first query.
        The former 0.0 default re-introduced exactly the circularity §5.6
        exists to break."""
        entry = self._lookup_core_entry(core_hash)
        if entry is not None:
            return float(entry.reliability)
        return P.KAPPA_INIT

    def narrow_validation_atoms(self, core_hash: bytes, atom: Any) -> None:
        """[F2] Narrow the latest pending prediction for a core to one atom."""
        if atom is None:
            return
        lst = self.pending_transfers.get(core_hash)
        if not lst:
            return
        tp = lst[-1]
        target = frozenset({atom})
        if tp.effect_atoms == target:
            return
        lst[-1] = TransferPrediction(
            core_hash=tp.core_hash,
            predicted_consequence=tp.predicted_consequence,
            confidence=tp.confidence,
            t_predicted=tp.t_predicted,
            horizon=tp.horizon,
            effect_atoms=target,
            # [Batch B / BUG1] Carry the ORIGINAL snapshot whole.
            # Intersection with (possibly narrowed) effect_atoms
            # happens at validation time, not here.
            effect_atoms_present_at_prediction=(
                tp.effect_atoms_present_at_prediction
            ),
        )

    def _update_reliability_entry(self, core_hash: bytes, observed: bool) -> None:
        """§8.5.3 reliability EMA — written exclusively by this module,
        stored on the CoreEntry (§5.12: 'The Mapper reads entry.reliability
        and never writes to it'; the Consolidator sets it once at
        extraction)."""
        entry = self._lookup_core_entry(core_hash)
        if entry is None:
            # The core was evicted between emission and validation; the
            # outcome is unobservable but must not pass silently.
            self._diagnostics["reliability_update_orphaned"] += 1
            return
        current = float(entry.reliability)
        if observed:
            entry.reliability = min(1.0, current + self.KAPPA_REL * (1.0 - current))
        else:
            entry.reliability = max(0.0, current - self.KAPPA_REL * current)
        entry.t_last_validated = self._current_step

    def _active_subgraph(self, t: int, concept_graph: Any) -> Subgraph:
        """§8.5.2 active_subgraph(t) = the §6.2 situation subgraph. Owned by
        the Mapper; used via injection when wired. The local builder below
        is the same §6.2 algorithm, kept for standalone operation and
        required to stay byte-equivalent in semantics."""
        if self._situation_provider is not None:
            return self._situation_provider(t)
        return self._build_active_subgraph(t, concept_graph)

# [Batch E divergence note] This module's local _build_active_subgraph
# still filters induced edges by the pre-Batch-E weight gate
# (edge.weight >= THETA_SITUATION_EDGE). Under the wired path
# (main.py: set_situation_provider(mapper.build_situation_subgraph)),
# validation situations flow through the Mapper and therefore inherit
# Batch E's recency-aware inclusion automatically. The divergence is
# deliberate: _build_active_subgraph is standalone-mode-only and is out
# of Batch E scope. If standalone validation ever ships to production,
# port _edge_included and the two new parameters into this module.
    def _build_active_subgraph(self, t: int, concept_graph: Any) -> Subgraph:
        """Local §6.2 situation-subgraph builder (fallback when the Mapper
        provider is not injected). Kept semantically identical to
        Mapper.build_situation_subgraph: A = 1 concept members → H_SITUATION
        expansion → θ_situation_edge weight filter → MAX_SITUATION_SIZE cap
        keeping highest A_node × EF_combined, tie-broken by canonical node
        key (lens_type, signature_hash) per §6.2/§9."""
        seed: Set[int] = set()
        for cid in sorted(concept_graph.concepts.keys()):
            c = concept_graph.concepts[cid]
            if c.A != 1:
                continue
            for nid in c.members:
                seed.add(nid)

        frontier: Set[int] = set(seed)
        for _ in range(self.H_SITUATION):
            next_frontier: Set[int] = set()
            for nid in frontier:
                for e in concept_graph.get_outgoing_edges(nid):
                    if e.weight >= self.THETA_SITUATION_EDGE:
                        next_frontier.add(e.child)
                for e in concept_graph.get_incoming_edges(nid):
                    if e.weight >= self.THETA_SITUATION_EDGE:
                        next_frontier.add(e.parent)
            next_frontier -= seed
            seed |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

        nodes = list(seed)
        if len(nodes) > self.MAX_SITUATION_SIZE:
            # Pre-compute per-node (A × EF_combined) once instead of inside
            # the sort key.
            node_scores: Dict[int, float] = {}
            for cid in sorted(concept_graph.concepts.keys()):
                c = concept_graph.concepts[cid]
                ef = getattr(c, "EF_combined", 0.0)
                for nid in c.members:
                    node_scores[nid] = max(node_scores.get(nid, 0.0), c.A * ef)

            def priority(nid: int) -> Tuple[float, Tuple[bytes, bytes, int]]:
                # §6.2: tie-break by canonical hash — the node's canonical
                # identity is (lens_type, signature_hash); node ID is the
                # terminal discriminator for total order.
                return (node_scores.get(nid, 0.0), self._canonical_node_key(concept_graph, nid))

            nodes.sort(key=priority, reverse=True)
            nodes = nodes[: self.MAX_SITUATION_SIZE]
        node_set = frozenset(nodes)

        edges: Set[Tuple[int, int, EdgeType]] = set()
        edge_params: Dict[Tuple[int, int, EdgeType], EdgeParams] = {}
        for nid in node_set:
            for e in concept_graph.get_outgoing_edges(nid):
                if e.child in node_set and e.weight >= self.THETA_SITUATION_EDGE:
                    key = (e.parent, e.child, e.edge_type)
                    edges.add(key)
                    if getattr(e, "params", None) is not None:
                        edge_params[key] = e.params
        return Subgraph(
            nodes=node_set,
            edges=frozenset(edges),
            edge_params=edge_params,
            anchor_of_node=anchor_positions_for(
                node_set, concept_graph.pattern_node_position
            ),
        )

    @staticmethod
    def _canonical_node_key(concept_graph: Any, nid: int) -> Tuple[bytes, bytes, int]:
        sig = concept_graph.get_pattern_node_signature(nid)
        if sig is None:
            return (b"_PRUNED_", b"\x00" * 16, nid)
        return (sig[0].encode("utf-8"), sig[1], nid)

    # =========================================================================
    # SURPRISE (§8.6)
    # =========================================================================

    def _emit_surprise(self, signal: SurpriseSignal) -> None:
        """Enforces global 100-signal bound across queue + overflow."""
        if self._surprises_this_step < self.MAX_SURPRISE_PER_STEP:
            self._surprises_this_step += 1
            if len(self._surprise_queue) >= self.MAX_SURPRISE_QUEUE:
                self._surprise_queue.popleft()
                self._diagnostics["surprise_queue_evicted"] += 1
            self._surprise_queue.append(signal)
        else:
            self._surprise_overflow.append(signal)
            total_buffered = len(self._surprise_queue) + len(self._surprise_overflow)
            while total_buffered > self.MAX_SURPRISE_QUEUE:
                if self._surprise_queue:
                    self._surprise_queue.popleft()
                else:
                    self._surprise_overflow.pop(0)
                self._diagnostics["surprise_queue_evicted"] += 1
                total_buffered -= 1

    def drain_surprises(self) -> List[SurpriseSignal]:
        out = list(self._surprise_queue)
        self._surprise_queue.clear()
        return out

    def surprise_queue_size(self) -> int:
        return len(self._surprise_queue)

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def get_diagnostics(self) -> Dict[str, Any]:
        # [Tier 1 reliability-decay measurement] Histogram over the
        # current reliability values of all tracked CoreEntries. Reveals
        # whether binary validation is burning reliability faster than
        # successes restore it.
        rel_buckets = [0, 0, 0, 0, 0]
        rel_values = []
        if self._core_library is not None:
            for entry in getattr(self._core_library, "entries", []):
                r = float(getattr(entry, "reliability", 0.0))
                rel_values.append(r)
                idx = min(4, int(r * 5))
                if idx < 0:
                    idx = 0
                rel_buckets[idx] += 1
        return {
            "pending_transfer_core_count": len(self.pending_transfers),
            "pending_transfer_total": sum(
                len(v) for v in self.pending_transfers.values()
            ),
            "pending_stale_evicted": self.pending_stale_evicted,
            "transfers_validated_this_run": self.transfers_validated_this_run,
            "transfers_observed_true": self.transfers_observed_true,
            "transfers_observed_false": self.transfers_observed_false,
            "prototype_registry_size": len(self.fertile_evidence_registry),
            "surprise_queue_size": len(self._surprise_queue),
            "surprise_overflow_size": len(self._surprise_overflow),
            "surprises_emitted_this_step": self._surprises_this_step,
            "effect_flags_known": len(self.effect_flags),
            "causal_pairs_tracked": len(self.bar_dp_dict),
            "rise_validation_pairs": list(self.rise_validation_pairs),
            "reliability_buckets": rel_buckets,
            "reliability_count": len(rel_values),
            "reliability_sum": sum(rel_values),
            "reliability_min": min(rel_values) if rel_values else 0.0,
            "reliability_max": max(rel_values) if rel_values else 0.0,
            "counters": dict(self._diagnostics),
        }

    def reset_episode_local_state(self) -> None:
        """[Batch B / BUG2] Clear per-episode state on environment
        reset. Pending transfers and rise history do NOT survive an
        episode boundary: a transfer registered in episode N has its
        snapshot against episode N's register, and validating it
        against episode N+1's register would be meaningless.

        Long-term statistics (C_opp, C_opp_not, C_succ_a,
        C_succ_real_total, effect_flags, Gate_bind, Gate_undo,
        prev_H_nec, fertile_evidence_registry) are intentionally
        preserved: action/effect semantics do not change between
        episodes of the same environment.

        [B3 registry note] Persistence-validated transfers confirm
        that predicted atoms remained true across the window — a
        correlational claim. They do NOT confirm the action caused
        the effect (that is the rise path's role, and Gate_bind /
        Gate_undo's role). Both paths feed the same reliability EMA
        because both confirm a correct prediction; causal depth is
        tracked separately."""
        self.pending_transfers.clear()
        self.pending_stale_evicted = 0
        self._rise_history.clear()
        self.rise_validation_pairs.clear()
        self._validated_outcomes.clear()
        self._surprise_queue.clear()
        self._surprise_overflow.clear()
        self._surprises_this_step = 0


# =============================================================================
# MODULE-LEVEL HELPERS
# =============================================================================


def _causal_surprise(predicted: float, observed: int) -> float:
    """§8.6.1 causal surprise (normative formula):

        surprise = |p̄_a − o| × max(p̄_a, 1 − p̄_a)

    The confidence weight max(p, 1−p) makes surprise peak when the system
    was confident AND wrong. Per the spec's own worked examples: at maximal
    uncertainty (p = 0.5) surprise = 0.25; at p = 0.95, o = 0 it is 0.9025.
    The maximum possible value is 1.0, so no clamping is required.

    (The former |2p−1|·|p−o| variant collapsed to 0 at p = 0.5, directly
    contradicting the spec examples and desensitizing the alarm; removed.)
    """
    p = clamp(predicted, 0.0, 1.0)
    confidence = max(p, 1.0 - p)
    return confidence * abs(p - float(observed))
