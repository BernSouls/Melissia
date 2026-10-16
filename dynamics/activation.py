"""
MODULE 5: ActivationEngine — v31

Responsibility: Activation, support, adaptation, onset, hysteresis,
                recognition evidence integration.
Owns: A_P(t), onset_P(t), onset_prev_P(t), supportTrace(P,t), adapt_P(t).
Reads: D_t^bind, D_t^rec, concept graph topology.
Writes: A_P(t), onset_P(t), onset_prev_P(t), adapt_P(t),
        supportTrace(P,t) on concept nodes.
MUST implement Activation Adaptation (§2.18.2).

v31 additions:
  - onset_prev is saved at the START of every step, before new onsets
    are computed. This is required by §1.12: the TRANSITION edge signal
    reads parent.onset_prev at step t+1 as the parent's onset from
    step t. Saving at the end of the step would make onset_prev equal
    onset_{t+1}, and the signal would test onset_{t+1} ∧ onset_{t+1}.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Set, Optional, Any, FrozenSet, Iterable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import copy

# Canonical shared types (§A.6 type invariance): single definitions live in
# sensory_floor (TypedSignature, Detection) and concept_graph (enums).
from sensory_floor import TypedSignature, Detection
from concept_graph import ConceptStatus, ConceptFamily
from core.hashing import (  # noqa: F401
    clamp,
    hash_64,
    heaviside,
    serialize_bytes,
    serialize_float,
    serialize_int,
    serialize_set,
    serialize_tuple,
)
import parameters as P


# =============================================================================
# ACTIVATION ENGINE PARAMETERS (§2.1.14, §2.1.3)
# =============================================================================

EPSILON = P.EPSILON
THETA_ACT_MIN = P.THETA_ACT_MIN
THETA_ACT_MAX = P.THETA_ACT_MAX
THETA_ACT_FLOOR = P.THETA_ACT_FLOOR
KAPPA_ACT = P.KAPPA_ACT
RHO_ACT_TARGET = P.RHO_ACT_TARGET
KAPPA_SUPPORT = P.KAPPA_SUPPORT
ETA_SUPPORT = P.ETA_SUPPORT
BETA_SUPPORT = P.BETA_SUPPORT
S_MAX = P.S_MAX
SIGMA_ACT_NOISE = P.SIGMA_ACT_NOISE
ETA_PRIME = 0.05
DELTA_ACT = P.DELTA_ACT
ADAPT_RATE = P.ADAPT_RATE
# §2.18.2 mandates adapt_cap > 1.0 - theta_act_floor = 0.95, otherwise
# sustained activation at z=1.0 never deactivates and edge potentiation dies.
ADAPT_CAP = P.ADAPT_CAP
ADAPT_DECAY = P.ADAPT_DECAY

KAPPA_EMA = 0.05
KAPPA_THETA = 0.05
DELTA_THETA_MAX = 0.1


# =============================================================================
# ACTIVATION ENGINE
# =============================================================================


class ActivationEngine:
    def __init__(self, concept_graph, global_theta_act: float = 0.5):
        self.cg = concept_graph
        self.global_theta_act = global_theta_act
        self.r_bar_act = 0.0
        # Family-specific thresholds (default same as global).
        self.family_theta_act = {fam.name: global_theta_act for fam in ConceptFamily}
        # Per-family activation rate EMAs.
        self.family_r_bar_act = {fam.name: 0.0 for fam in ConceptFamily}
        self.prev_bind: Set[Detection] = set()
        self.prev_rec: Set[Detection] = set()
        # Activation history for average activation diagnostics.
        self.activation_history: Dict[int, deque] = {}
        # Timescale separation invariant check (§2.1.14).
        assert KAPPA_ACT < (1 - BETA_SUPPORT) / 4, "Timescale separation violated"

        # Initialize concept state if not already present.
        for concept in self.cg.concepts.values():
            if not hasattr(concept, "A"):
                concept.A = 0
            if not hasattr(concept, "onset"):
                concept.onset = 0
            # §1.12: onset_prev holds the previous step's onset for TRANSITION
            # edges. Initialized to 0 so the first TRANSITION update at t=1
            # reads 0 for the parent side, which is correct — no prior onset.
            if not hasattr(concept, "onset_prev"):
                concept.onset_prev = 0
            if not hasattr(concept, "supportTrace"):
                concept.supportTrace = 0.0
            if not hasattr(concept, "adapt_P"):
                concept.adapt_P = 0.0

        self.thresholds = {cid: self.global_theta_act for cid in self.cg.concepts}
        for cid, concept in self.cg.concepts.items():
            self.thresholds[cid] = self.family_theta_act.get(
                concept.family.name, self.global_theta_act
            )

    # -------------------------------------------------------------------------
    # Main step
    # -------------------------------------------------------------------------

    def step(
        self,
        t: int,
        D_bind: Optional[Set[Detection]],
        D_rec: Optional[Set[Detection]],
        s_t: int,
    ) -> None:
        """Compute activations for step t from the previous step's detections.
        Then store the current detections for the next step.

        [SPEC NOTE §1.12] onset_prev is saved at the START of this method,
        before new onsets are computed. Saving it at the end (as the v30
        code did conceptually) would make onset_prev equal onset at step t,
        so the TRANSITION edge signal at t+1 would test
        onset_{t+1} ∧ onset_{t+1} instead of onset_t ∧ onset_{t+1}.
        """
        # §1.12: persist the previous step's onset for the TRANSITION signal.
        for concept in self.cg.concepts.values():
            concept.onset_prev = concept.onset

        current_bind = self.prev_bind
        current_rec = self.prev_rec

        for concept_id, concept in self.cg.concepts.items():
            if (
                concept.status == ConceptStatus.TOMBSTONED
                or concept.dormant
                or concept.soft_dormant
            ):
                concept.A = 0
                concept.onset = 0
                continue

            # 1. Detection strengths.
            z_bind = self._compute_z_bind(concept, current_bind)
            z_rec = self._compute_z_rec(concept, current_rec)
            z = max(z_bind, z_rec)

            # 2. Top-down priming from children (§2.18.1).
            prime = self._compute_prime(concept, t - 1)

            # 3. Support gain.
            support_gain = concept.supportTrace / (
                concept.supportTrace + KAPPA_SUPPORT + EPSILON
            )

            # 4. Threshold lookup.
            if concept_id not in self.thresholds:
                self.thresholds[concept_id] = self.family_theta_act.get(
                    concept.family.name, self.global_theta_act
                )

            # 5. State-dependent noise.
            noise = self._compute_noise(concept_id, t, s_t)

            # 6. Base threshold (clamped).
            theta_base = (
                self.thresholds[concept_id] - prime - ETA_SUPPORT * support_gain + noise
            )
            theta_base = clamp(theta_base, THETA_ACT_FLOOR, THETA_ACT_MAX)

            # 7. Final threshold with adaptation (no clamp).
            theta_final = theta_base + concept.adapt_P

            # 8. Hysteresis activation.
            A_prev = getattr(concept, "A_prev", 0)
            if t == 0:
                A_prev = 0
            A_new = heaviside(z - theta_final - DELTA_ACT * (1 - 2 * A_prev))

            # 9. Onset.
            onset_new = A_new * (1 - A_prev)

            # 10. Update concept fields.
            concept.A = A_new
            concept.onset = onset_new
            if onset_new == 1:
                concept.onset_time = t
            if A_new == 1:
                concept.last_activation = t

            # 11. Adaptation update for next step.
            if A_new == 1:
                concept.adapt_P = min(ADAPT_CAP, concept.adapt_P + ADAPT_RATE)
            else:
                concept.adapt_P = ADAPT_DECAY * concept.adapt_P

        # Support trace update uses parent onsets computed this step.
        for concept_id, concept in self.cg.concepts.items():
            if (
                concept.status == ConceptStatus.TOMBSTONED
                or concept.dormant
                or concept.soft_dormant
            ):
                continue
            parent_onset_sum = 0.0
            for p_id in concept.parent_ids:
                parent = self.cg.concepts.get(p_id)
                if parent is None:
                    continue
                # v31 ConceptGraph.get_edge with 2 args returns the first
                # matching typed edge; concept-level ancestry edges are
                # CO_ACTIVATION.
                edge = self.cg.get_edge(p_id, concept_id)
                w = edge.weight if edge else 0.0
                parent_onset_sum += w * parent.onset
            concept.supportTrace = clamp(
                BETA_SUPPORT * concept.supportTrace
                + (1 - BETA_SUPPORT) * KAPPA_SUPPORT * parent_onset_sum,
                0,
                S_MAX,
            )

        self._update_global_threshold(t)

        # Save A_prev and history for next step.
        for concept in self.cg.concepts.values():
            concept.A_prev = concept.A
            if concept.id not in self.activation_history:
                self.activation_history[concept.id] = deque(maxlen=100)
            self.activation_history[concept.id].append(concept.A)

        # Update previous detections (None must not clobber).
        if D_bind is not None:
            self.prev_bind = D_bind
        if D_rec is not None:
            self.prev_rec = D_rec

    # -------------------------------------------------------------------------
    # Detection strength
    # -------------------------------------------------------------------------

    def _compute_z_bind(self, concept, detections: Set[Detection]) -> float:
        """Max detection strength from binding evidence.

        SPATIAL_OBJECT activates on the stable entity anchor rather than
        the full evidence signature (which may include dynamic components
        that change with agent movement). Exact signature match for all
        concepts as fallback.
        """
        sigma_set = {sig for (_, sig, _, _, _, _) in concept.evidence_registry}
        max_z = 0.0
        for d in detections:
            if (
                getattr(concept.family, "name", None) == "SPATIAL_OBJECT"
                and concept.anchor is not None
                and d.alpha == concept.anchor
            ):
                max_z = max(max_z, d.z)
            if d.sigma in sigma_set:
                max_z = max(max_z, d.z)
        return max_z

    def _compute_z_rec(self, concept, detections: Set[Detection]) -> float:
        """Max detection strength from recognition evidence where
        recognizedConcept = concept.id."""
        max_z = 0.0
        for d in detections:
            for role in d.rho:
                if role[0] == "recognizedConcept" and role[1] == concept.id:
                    max_z = max(max_z, d.z)
                    break
        return max_z

    def _compute_prime(self, concept, t_minus_1: int) -> float:
        """Top-down priming from active children."""
        prime = 0.0
        for child_id in concept.child_ids:
            child = self.cg.concepts.get(child_id)
            if child is None:
                continue
            A_child = getattr(child, "A_prev", 0)
            edge = self.cg.get_edge(concept.id, child_id)
            w = edge.weight if edge else 0.0
            prime += ETA_PRIME * A_child * w
        return prime

    def _compute_noise(self, concept_id: int, t: int, s_t: int) -> float:
        """State-dependent biological noise (deterministic)."""
        data = serialize_tuple(
            [serialize_int(concept_id), serialize_int(t), serialize_int(s_t)]
        )
        hash_val = hash_64(data)
        normalized = (hash_val / 2**64) - 0.5
        return SIGMA_ACT_NOISE * normalized

    # -------------------------------------------------------------------------
    # Homeostatic threshold update
    # -------------------------------------------------------------------------

    def _update_global_threshold(self, t: int) -> None:
        family_active = {fam.name: 0 for fam in ConceptFamily}
        family_eligible = {fam.name: 0 for fam in ConceptFamily}
        global_active = 0
        global_eligible = 0
        for concept in self.cg.concepts.values():
            if (
                concept.status == ConceptStatus.TOMBSTONED
                or concept.dormant
                or concept.soft_dormant
            ):
                continue
            fam = concept.family.name if concept.family else "GLOBAL"
            if fam in family_active:
                family_eligible[fam] += 1
                if concept.A == 1:
                    family_active[fam] += 1
            global_eligible += 1
            if concept.A == 1:
                global_active += 1

        r_act_global = (
            global_active / (global_eligible + EPSILON) if global_eligible > 0 else 0.0
        )
        self.r_bar_act = (1 - KAPPA_EMA) * self.r_bar_act + KAPPA_EMA * r_act_global
        self.global_theta_act = clamp(
            self.global_theta_act + KAPPA_ACT * (self.r_bar_act - RHO_ACT_TARGET),
            THETA_ACT_MIN,
            THETA_ACT_MAX,
        )

        for fam in ConceptFamily:
            fam_name = fam.name
            r_act_fam = (
                family_active[fam_name] / (family_eligible[fam_name] + EPSILON)
                if family_eligible[fam_name] > 0
                else 0.0
            )
            self.family_r_bar_act[fam_name] = (1 - KAPPA_EMA) * self.family_r_bar_act[
                fam_name
            ] + KAPPA_EMA * r_act_fam
            self.family_theta_act[fam_name] = clamp(
                self.family_theta_act[fam_name]
                + KAPPA_ACT * (self.family_r_bar_act[fam_name] - RHO_ACT_TARGET),
                THETA_ACT_MIN,
                THETA_ACT_MAX,
            )

        for cid, concept in self.cg.concepts.items():
            fam_name = concept.family.name if concept.family else None
            if fam_name and fam_name in self.family_theta_act:
                self.thresholds[cid] = self.family_theta_act[fam_name]
            else:
                self.thresholds[cid] = self.global_theta_act

    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------

    def get_activation(self, concept_id: int, t: int) -> int:
        concept = self.cg.concepts.get(concept_id)
        if concept:
            return getattr(concept, "A", 0)
        return 0

    def get_onset(self, concept_id: int, t: int) -> int:
        concept = self.cg.concepts.get(concept_id)
        if concept:
            return getattr(concept, "onset", 0)
        return 0

    def get_onset_prev(self, concept_id: int, t: int) -> int:
        """§1.12: previous step's onset for TRANSITION signal dispatch.
        Read by ConceptGraph.step_28_update_edges at t+1."""
        concept = self.cg.concepts.get(concept_id)
        if concept:
            return getattr(concept, "onset_prev", 0)
        return 0

    def get_average_activation(self, concept_id: int) -> float:
        history = self.activation_history.get(concept_id)
        if not history:
            return 0.0
        return sum(history) / len(history)

    def compute_context_signature(self, S_t, prev_active_concepts) -> int:
        active_sigs = []
        for cid in prev_active_concepts:
            concept = self.cg.concepts.get(cid)
            if concept:
                active_sigs.append(concept.sigma.serialize())
        return hash_64(serialize_set(active_sigs))

    def set_previous_detections(
        self, D_bind: Set[Detection], D_rec: Set[Detection]
    ) -> None:
        self.prev_bind = D_bind
        self.prev_rec = D_rec

    def get_onset_times(self) -> Dict[int, int]:
        return {
            cid: c.onset_time
            for cid, c in self.cg.concepts.items()
            if c.onset_time >= 0
        }
