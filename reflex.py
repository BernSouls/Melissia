"""
Innate orienting reflex + kinematic salience field + 3D motor approach layer.

Conforms to:
  - 13_UpdateTOBLUEPRINT.md §6   (Directional Relational Invariants in 3D)
  - 13_UpdateTOBLUEPRINT.md §7.1 (Composite 3D Salience Field)
  - 13_UpdateTOBLUEPRINT.md §7.2 (3D Egocentric Vector Decomposition)
  - 12_Update §2.3–§2.5          (Corticotectal Arbitration, Kinematic Velocity, Habituation)

Owns exclusively: filtered positions, pair-keyed sign memory, directional slots
(S_t^dir), salience Psi/Psi_bar, orienting target u*, gate Gamma, habituation
clocks, motor approach output.

Reads read-only: observation (positions/presence/distances/adjacency),
ConceptGraph (anchor/Anchors, A, EF_combined, M_opp, M_conf).

Writes: only its own state; exposes get_dir_tuples(), orienting_target,
gamma(u), bar_psi(u).
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.hashing import (  # noqa: F401
    hash_64,
    serialize_value,
)
from core.registries import (  # noqa: F401
    ACTION_NOOP,
    is_noop_action,
)
import parameters as P

# =============================================================================
# CONSTANTS — sourced from parameters.py (single source of truth)
# =============================================================================

ALPHA_P: float = P.REFLEX_ALPHA_P
ETA_DEAD: float = P.REFLEX_ETA_DEAD
EPS_M: float = P.REFLEX_EPS_M
L_REF: float = P.REFLEX_L_REF
K_DIR_MAX: int = P.REFLEX_K_DIR_MAX
DELTA_RANK: float = P.REFLEX_DELTA_RANK
T_RETAIN: int = P.REFLEX_T_RETAIN
N_ANCHOR_MAX: int = P.N_ANCHOR_MAX
# [CONTOUR TIMEOUT ADAPTATION] The directive's 2*(N_ANCHOR_MAX+5) budget
# assumes N_ANCHOR_MAX is the reflex working-memory cap; in this tree that
# name is a DEPRECATED alias of N_PROTO_MAX (1000), which would make the
# anti-infinite-slide timeout inert (2010 > any run). The constitutional
# reflex cap is N_ANCHOR_MAX_3D (13_Update §3.4) -> 42-slide budget.
N_ANCHOR_MAX_3D: int = P.N_ANCHOR_MAX_3D
V_BASE: float = P.REFLEX_V_BASE
KAPPA_V: float = P.REFLEX_KAPPA_V
EPS_V: float = P.REFLEX_EPS_V
V_MAX: float = P.REFLEX_V_MAX
R_ENV_MIN: float = P.REFLEX_R_ENV_MIN
M_EXP: float = P.REFLEX_M_EXP
W_D: float = P.REFLEX_W_D
W_V: float = P.REFLEX_W_V
W_C: float = P.REFLEX_W_C
W_N: float = P.REFLEX_W_N
V0: float = P.REFLEX_V0
THETA_G: float = P.REFLEX_THETA_G
DELTA_HYST: float = P.REFLEX_DELTA_HYST
BETA_G: float = P.REFLEX_BETA_G
GAMMA_MIN: float = P.REFLEX_GAMMA_MIN
KAPPA_C: float = P.REFLEX_KAPPA_C
T_DWELL: int = P.REFLEX_T_DWELL
BETA_VEL: float = P.REFLEX_BETA_VEL
THETA_VEL: float = P.REFLEX_THETA_VEL
V_THRESH: float = P.REFLEX_V_THRESH
TAU_KIN_SAT: int = P.REFLEX_TAU_KIN_SAT
PSI_CLIP: float = P.REFLEX_PSI_CLIP
EPS_DEAD_MOTOR: float = P.REFLEX_EPS_DEAD_MOTOR
TAU_SATIETY: int = max(1, P.TAU_SATIETY)

# =============================================================================
# MOVEMENT ROLE MAPPING (§7.2)
# =============================================================================

# Axis index → dimension label:
#   d=0: Lateral (X), d=1: Vertical (Y), d=2: Depth (Z)

# Axis → sign → role mapping (pure geometry; the role→action vocabulary
# is environment-declared via `get_role_action_map` / `role_to_actions`)
_AXIS_SIGN_TO_ROLE: Dict[int, Dict[int, str]] = {
    0: {+1: "+X", -1: "-X"},  # Lateral
    1: {+1: "+Y", -1: "-Y"},  # Vertical
    2: {+1: "+Z", -1: "-Z"},  # Depth
}


def _is_nav_action(action: Any, env: Any) -> bool:
    """Query environment interface for navigation classification.
    Conservative fallback: False (does not guess via string names)."""
    if is_noop_action(action):
        return True
    if env is not None and hasattr(env, "is_nav_action"):
        try:
            return bool(env.is_nav_action(action))
        except Exception:
            return False
    return False


# =============================================================================
# MATH HELPERS
# =============================================================================


def _math_sqrt(x: float) -> float:
    return math.sqrt(max(0.0, x))


def _math_exp(x: float) -> float:
    return math.exp(max(-50.0, min(50.0, x)))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _fert_norm(concept: Any) -> float:
    """Normalize fertility (EF_combined) to [0, 1] via log scaling.
    Bug 6 fix: parameter renamed from P to concept to avoid shadowing."""
    ef = getattr(concept, "EF_combined", 0.0)
    return math.log(1.0 + min(ef, 1e9)) / math.log(1.0 + 1e9 + 1e-9)


# =============================================================================
# INNATE REFLEX CLASS
# =============================================================================


class InnateReflex:
    def __init__(
        self,
        env: Any = None,
        agent_anchor: int = 0,
        action_repertoire: Optional[List[Any]] = None,
        concept_graph: Optional[Any] = None,
        satiety_window: int = TAU_SATIETY,
        babble_budget: int = 3,
        role_to_actions: Optional[Dict[str, List[str]]] = None,
    ):
        self.env = env
        self.agent_anchor = agent_anchor
        self.action_repertoire = list(action_repertoire or [])
        self.cg = concept_graph
        self.satiety_window = satiety_window
        self.babble_budget = babble_budget
        self.t: int = 0

        # Motor mapping injection
        if role_to_actions is not None:
            self.role_to_actions = role_to_actions
        elif env is not None and hasattr(env, "get_role_action_map"):
            self.role_to_actions = env.get_role_action_map()
        else:
            self.role_to_actions = {}
            warnings.warn(
                "InnateReflex: no role_to_actions provided. "
                "Motor approach proposals will be unavailable until "
                "the environment supplies a role→action map.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Kinematics (§6.1, 12_Update §2.4)
        self.filtered_pos: Dict[Any, Tuple[float, ...]] = {}
        self.prev_filtered: Dict[Any, Tuple[float, ...]] = {}
        self.vel: Dict[Any, Tuple[float, ...]] = {}
        self.prev_vel: Dict[Any, Tuple[float, ...]] = {}
        self.prev_U: Set[Any] = set()
        self.v_peak: float = V_BASE

        # Directional slots and sign memory (§6, 12_Update §2.2)
        # pair_sign keyed by (agent_anchor, entity, axis) — NOT by slot index
        self.pair_sign: Dict[Tuple[Any, Any, int], int] = {}
        self.prev_slots: Dict[int, Any] = {}
        self.retain: Dict[int, int] = {}
        self.dir_tuples: List[Tuple[int, int, int]] = []

        # Salience and gating (§7.1, 12_Update §2.3, §2.5)
        self.f_v: Dict[Any, float] = {}
        self.bar_v: Dict[Any, float] = {}
        self.h_gate: Dict[Any, int] = {}
        self.persist: Dict[Any, int] = {}
        self.prev_cand: Dict[Any, int] = {}
        self.psi: Dict[Any, float] = {}
        self.bar_psi: Dict[Any, float] = {}
        self.tau_interact: Dict[Any, int] = {}
        self.tau_pursuit: Dict[Any, int] = {}
        self.orienting_target: Optional[Any] = None

        # Motor layer / IOR (12_Update §2.5)
        # `familiar_until` is the satiety horizon: set to
        # (t + satiety_window) when interaction_counts[u] reaches
        # babble_budget. Distinct from `last_seen_step`, which tracks
        # persistence for occlusion memory. Conflating them caused
        # satiety to be overwritten every frame.
        self.familiar_until: Dict[Any, int] = {}
        self.last_seen_step: Dict[Any, int] = {}
        self.interaction_counts: Dict[Any, int] = {}
        # [Patch 2a] Fixation guard: consecutive adjacency-without-interaction
        # ticks per entity. Satiety is NOT applied on first adjacency (the
        # former instant-satiation made the reflex bounce off every entity it
        # reached); only after babble_budget adjacent idle ticks does the
        # entity become familiar. Real interactions reset this counter and
        # run the interaction_counts budget instead (notify_interaction).
        self.adjacency_ticks: Dict[Any, int] = {}
        self.prev_presence: Dict[Any, bool] = {}
        self.contour_commit: Dict[Any, str] = {}
        # [CONTOUR FIX] per-target slide-step counter for the forced
        # reversal timeout (see _extract_ego_directions STEP 2).
        self._contour_steps: Dict[Any, int] = {}
        self._interactions: Set[Any] = set()
        self._anchor_to_concepts: Dict[Any, List[Any]] = defaultdict(list)

        # Bug 3 fix: track warned roles to avoid flooding
        self._warned_roles: Set[str] = set()

        # Validate that declared role_to_actions map intersects with action repertoire
        if self.action_repertoire and self.role_to_actions:
            mapped_actions = {a for acts in self.role_to_actions.values() for a in acts}
            has_directional = bool(set(self.action_repertoire) & mapped_actions)
            if not has_directional:
                warnings.warn(
                    "Action repertoire contains no actions matching the declared role_to_actions map. "
                    "Motor approach will produce no directional proposals.",
                    UserWarning,
                    stacklevel=2,
                )

    # =========================================================================
    # OBSERVATION INTERFACE HELPERS
    # =========================================================================

    def _get_positions(self, observation: Any) -> Dict[Any, Tuple[float, ...]]:
        """Extract positions with fallback chain.
        Bug 1 fix: normalizes list/tuple to dict keyed by index."""
        positions = getattr(observation, "positions", None)
        if positions is not None:
            if isinstance(positions, dict):
                return positions
            return {i: tuple(p) for i, p in enumerate(positions)}
        if hasattr(observation, "get_positions"):
            pos = observation.get_positions()
            if pos is None:
                return {}
            if isinstance(pos, dict):
                return pos
            return {i: tuple(p) for i, p in enumerate(pos)}
        return {}

    def _get_presence_collection(
        self, observation: Any
    ) -> Optional[Union[Dict[Any, bool], List[bool], Tuple[bool, ...]]]:
        """Extract the presence collection from observation.
        Bug 4 fix: renamed from _get_presence for clarity.
        Returns the raw collection (dict/list/tuple) or None."""
        return getattr(observation, "presence", None)

    def _get_entity_indices(self, observation: Any) -> List[int]:
        """Extract entity indices with fallback chain."""
        indices = getattr(observation, "entity_indices", None)
        if indices is not None:
            return list(indices)
        if hasattr(observation, "get_entity_indices"):
            return list(observation.get_entity_indices())
        if hasattr(observation, "num_atoms"):
            return list(range(observation.num_atoms()))
        raise AttributeError(
            "Observation must expose entity_indices, get_entity_indices(), "
            "or num_atoms()"
        )

    def _is_present_at(self, observation: Any, idx: int) -> bool:
        """Check presence for a single index with fallback."""
        presence = self._get_presence_collection(observation)
        if presence is not None:
            if isinstance(presence, dict):
                return presence.get(idx, True)
            if isinstance(presence, (list, tuple)):
                return bool(presence[idx]) if idx < len(presence) else True
        if hasattr(observation, "is_present"):
            return observation.is_present(idx)
        return True

    # =========================================================================
    # MAIN UPDATE PIPELINE
    # =========================================================================

    def update(self, t: int, observation: Any) -> None:
        """Execute orienting pipeline after observation ingestion."""
        self.t = t
        positions = self._get_positions(observation)

        # Reset pursuit and record interaction when agent touches entity (§2.5.2)
        if observation is not None:
            raw_entities = self._get_entity_indices(observation)
            for u in raw_entities:
                if u != self.agent_anchor and self._adj(
                    observation, self.agent_anchor, u
                ):
                    self._interactions.add(u)

        # Extract active entity set
        raw_entities = self._get_entity_indices(observation)
        presence = self._get_presence_collection(observation)

        if presence is not None:
            if isinstance(presence, dict):
                U = {u for u in raw_entities if presence.get(u, True)}
            else:
                U = {u for u in raw_entities if self._is_present_at(observation, u)}
        else:
            U = {u for u in raw_entities if self._is_present_at(observation, u)}

        U.add(self.agent_anchor)
        U = self._truncate(U)

        self._update_filtered(U, positions)
        self._update_velocity(U)

        for u in U:
            self.f_v[u] = self._f_v(u)

        # Build anchor-to-concepts index once per update step
        self._anchor_to_concepts.clear()
        if self.cg is not None:
            for concept in self.cg.concepts.values():
                anchor = getattr(concept, "anchor", None)
                if anchor is not None:
                    self._anchor_to_concepts[anchor].append(concept)
                anchors_set = getattr(concept, "Anchors", None)
                if anchors_set:
                    for a in anchors_set:
                        self._anchor_to_concepts[a].append(concept)

        for u in U:
            self._update_cortical(u)

        for u in U:
            if u != self.agent_anchor:
                self._update_salience(u, positions)

        self._assign_slots(U)
        self._build_dir_tuples()

        # Slot-1 entity represents the current orienting target (§2.2.2)
        self.orienting_target = self.prev_slots.get(1)
        self.prev_U = set(U)

    # =========================================================================
    # TRUNCATION (§3.4.3 capacity bound)
    # =========================================================================

    def _truncate(self, U: Set[Any]) -> Set[Any]:
        cap = N_ANCHOR_MAX - 1
        ext = [u for u in U if u != self.agent_anchor]
        if len(ext) <= cap:
            return U
        ext.sort(key=lambda u: (-self.bar_psi.get(u, 0.0), u))
        return {self.agent_anchor} | set(ext[:cap])

    # =========================================================================
    # KINEMATICS (§6.1, 12_Update §2.4)
    # =========================================================================

    def _update_filtered(self, U: Set[Any], positions: Dict[Any, Any]) -> None:
        self.prev_filtered = dict(self.filtered_pos)
        for u in list(U):
            pos = positions.get(u)
            if pos is None:
                if u in self.prev_filtered:
                    pos = self.prev_filtered[u]
                else:
                    U.discard(u)
                    continue
            if u not in self.filtered_pos:
                self.filtered_pos[u] = tuple(pos)
            else:
                prev = self.filtered_pos[u]
                self.filtered_pos[u] = tuple(
                    (1.0 - ALPHA_P) * prev[d] + ALPHA_P * pos[d]
                    for d in range(len(pos))
                )
        # Persistence-aware eviction. Use `last_seen_step` — NOT
        # `familiar_until` — for the retention decision. `familiar_until`
        # is the satiety horizon and must not be touched here.
        for u in list(self.filtered_pos):
            if u == self.agent_anchor:
                continue
            if u not in positions:
                last_seen = self.last_seen_step.get(u, -1)
                if last_seen < 0 or (self.t - last_seen) > TAU_SATIETY:
                    del self.filtered_pos[u]
            else:
                self.last_seen_step[u] = self.t

    def _update_velocity(self, U: Set[Any]) -> None:
        self.prev_vel = dict(self.vel)
        for u in U:
            if u in self.prev_filtered and u in self.prev_U:
                cur, prev = self.filtered_pos[u], self.prev_filtered[u]
                self.vel[u] = tuple(
                    max(-V_MAX, min(cur[d] - prev[d], V_MAX)) for d in range(len(cur))
                )
            else:
                self.vel[u] = tuple([0.0] * len(self.filtered_pos.get(u, (0.0, 0.0))))
        # Bug 7 fix: removed dead `or [V_BASE]`
        peak = max(
            [V_BASE, (1.0 - KAPPA_V) * self.v_peak]
            + [self._norm(self.prev_vel.get(u)) for u in self.prev_U]
        )
        self.v_peak = peak

    @staticmethod
    def _norm(v: Optional[Tuple[float, ...]]) -> float:
        return _math_sqrt(sum(x * x for x in (v or ())))

    def _f_v(self, u: Any) -> float:
        n = self._norm(self.vel.get(u))
        return n / (n + self.v_peak + EPS_V)

    # =========================================================================
    # CORTICAL VALENCE (12_Update §2.3 — PRESERVED UNCHANGED)
    # =========================================================================

    def _update_cortical(self, u: Any) -> None:
        f_v = self.f_v.get(u, 0.0)
        baseline = min(THETA_G - DELTA_HYST, V0 + (1.0 - V0) * f_v)
        v_concept = 0.0
        # Bug 6 fix: renamed loop variable from P to concept
        for concept in self._anchor_to_concepts.get(u, []):
            m_opp = getattr(concept, "M_opp", 0.0)
            m_conf = getattr(concept, "M_conf", 0.0)
            rel = min(1.0, m_conf / (m_opp + 1e-9))
            val = float(getattr(concept, "A", 0)) * _fert_norm(concept) * rel
            if val > v_concept:
                v_concept = val
        v_raw = max(baseline, v_concept) if v_concept > 0 else baseline
        prev = self.bar_v.get(u, V0)
        self.bar_v[u] = (1.0 - KAPPA_C) * prev + KAPPA_C * v_raw
        bv = self.bar_v[u]
        cand = (
            1
            if bv >= THETA_G + DELTA_HYST
            else (0 if bv <= THETA_G - DELTA_HYST else self.h_gate.get(u, 0))
        )
        prev_h = self.h_gate.get(u, 0)
        prev_cand = self.prev_cand.get(u, prev_h)
        prev_p = self.persist.get(u, 0)
        if cand == prev_h:
            persist = 0
        elif cand == prev_cand:
            persist = prev_p + 1
        else:
            persist = 1
        if persist >= T_DWELL:
            self.h_gate[u] = cand
            persist = 0
        else:
            self.h_gate[u] = prev_h
        self.prev_cand[u] = cand
        self.persist[u] = persist

    def gamma(self, u: Any) -> float:
        g_c = self.h_gate.get(u, 0) / (
            1.0 + _math_exp(-BETA_G * (self.bar_v.get(u, 0.0) - THETA_G))
        )
        return max(g_c, self.f_v.get(u, 0.0) * GAMMA_MIN)

    # =========================================================================
    # SALIENCE (§7.1 — PRESERVED UNCHANGED)
    # =========================================================================

    def _update_salience(self, u: Any, positions: Dict[Any, Any]) -> None:
        raw_u = positions.get(u) if positions else None
        raw_agent = positions.get(self.agent_anchor) if positions else None
        if raw_u is None:
            raw_u = self.filtered_pos.get(u)
        if raw_agent is None:
            raw_agent = self.filtered_pos.get(self.agent_anchor)
        d = self._dist(raw_u, raw_agent)
        f_d = _math_exp(-min(d / R_ENV_MIN, M_EXP))
        nv = self._norm(self.vel.get(u))
        w_kin = 1.0 / (
            1.0 + _math_exp(-BETA_VEL * (nv / max(self.v_peak, EPS_V) - THETA_VEL))
        )
        h_st = _clamp01((self.t - self.tau_interact.get(u, -TAU_SATIETY)) / TAU_SATIETY)
        h_kin = _math_exp(-self.tau_pursuit.get(u, 0) / TAU_KIN_SAT)
        f_nov = w_kin * h_kin + (1.0 - w_kin) * h_st
        raw = (
            W_D * f_d
            + W_V * self.f_v.get(u, 0.0)
            + W_C * self.bar_v.get(u, 0.0)
            + W_N * f_nov
        )
        self.psi[u] = min(PSI_CLIP, raw)
        self.bar_psi[u] = (1.0 - 0.2) * self.bar_psi.get(
            u, self.psi[u]
        ) + 0.2 * self.psi[u]

    # =========================================================================
    # SLOT ASSIGNMENT (12_Update §2.2 — PRESERVED UNCHANGED)
    # =========================================================================

    def _assign_slots(self, U: Set[Any]) -> None:
        E = [u for u in U if u != self.agent_anchor]
        K = min(len(E), K_DIR_MAX)
        for r in list(self.prev_slots.keys()):
            if r > K:
                del self.prev_slots[r]
                self.retain.pop(r, None)
        avail = set(E)
        for r in range(1, K + 1):
            if not avail:
                self.prev_slots.pop(r, None)
                continue
            v_star = max(
                avail,
                key=lambda v: (
                    self.bar_psi.get(v, 0.0),
                    -v if isinstance(v, (int, float)) else -hash_64(serialize_value(v)),
                ),
            )
            prev = self.prev_slots.get(r)
            if (
                prev in avail
                and self.retain.get(r, 0) < T_RETAIN
                and self.bar_psi.get(prev, 0.0)
                >= self.bar_psi.get(v_star, 0.0) - DELTA_RANK
            ):
                u_r = prev
                self.retain[r] = self.retain.get(r, 0) + 1
            else:
                u_r = v_star
                self.retain[r] = 0
            avail.discard(u_r)
            self.prev_slots[r] = u_r

    # =========================================================================
    # DIRECTIONAL TUPLES (§6 — sign memory keyed by (agent, entity, axis))
    # =========================================================================

    def _build_dir_tuples(self) -> None:
        self.dir_tuples = []
        ap = self.filtered_pos.get(self.agent_anchor)
        if ap is None:
            return
        for r, u in sorted(self.prev_slots.items()):
            fp = self.filtered_pos.get(u)
            if fp is None:
                continue
            m_eff = max(self._dist(fp, ap), EPS_M * L_REF)
            tau = ETA_DEAD * m_eff
            dim = min(len(fp), len(ap))
            for d in range(dim):
                delta = fp[d] - ap[d]
                key = (self.agent_anchor, u, d)
                if delta > tau:
                    s = 1
                elif delta < -tau:
                    s = -1
                else:
                    s = self.pair_sign.get(key, 1 if delta >= 0.0 else -1)
                self.pair_sign[key] = s
                self.dir_tuples.append((r, d, s))

    # =========================================================================
    # INTERACTION BUFFER
    # =========================================================================

    def consume_interactions(self) -> Set[Any]:
        """Consume and clear pending interactions buffer."""
        res = set(self._interactions)
        self._interactions.clear()
        return res

    # =========================================================================
    # POST-UPDATE HABITUATION (12_Update §2.5.2 — PRESERVED UNCHANGED)
    # =========================================================================

    def post_update(
        self, t: int, interactions: Set[Any], crystallized: Set[Any]
    ) -> None:
        """Update habituation clocks after action execution."""
        for u in list(self.bar_psi):
            I = u in interactions
            Cr = u in crystallized
            if I:
                self.tau_interact[u] = t
            if I or Cr or u != self.orienting_target:
                self.tau_pursuit[u] = 0
            else:
                self.tau_pursuit[u] = self.tau_pursuit.get(u, 0) + 1

    # =========================================================================
    # PUBLIC GETTERS
    # =========================================================================

    def get_dir_tuples(self) -> List[Tuple[int, int, int]]:
        return list(self.dir_tuples)

    # =========================================================================
    # SATIETY AND IOR (12_Update §2.5)
    # =========================================================================

    def is_entity_satiated(self, idx: Any, t: int) -> bool:
        if self._is_dynamic(idx):
            return False
        return t < self.familiar_until.get(idx, -1)

    def is_affordance_satiated(self, action: Any, t: int, observation: Any) -> bool:
        """Checks if a manipulative affordance targets an entity in IOR satiety."""
        if observation is None:
            return False
        if _is_nav_action(action, self.env):
            return False
        raw_entities = self._get_entity_indices(observation)
        for idx in raw_entities:
            if idx != self.agent_anchor:
                is_pres = self._is_present_at(observation, idx)
                if is_pres and self._adj(observation, self.agent_anchor, idx):
                    if self.is_entity_satiated(idx, t):
                        return True
        return False

    def notify_interaction(self, action: Any, observation: Any) -> None:
        if observation is None or not action or _is_nav_action(action, self.env):
            return
        raw_entities = self._get_entity_indices(observation)
        for idx in raw_entities:
            if idx != self.agent_anchor:
                is_pres = self._is_present_at(observation, idx)
                if is_pres and self._adj(observation, self.agent_anchor, idx):
                    c = self.interaction_counts.get(idx, 0) + 1
                    self.interaction_counts[idx] = c
                    # [Patch 2c] A real interaction resets the idle-adjacency
                    # fixation counter; satiety here is earned through the
                    # interaction budget, not through proximity.
                    self.adjacency_ticks[idx] = 0
                    if c >= self.babble_budget:
                        self.familiar_until[idx] = self.t + self.satiety_window
                        self.interaction_counts[idx] = 0
                    self._interactions.add(idx)
                    return

    # =========================================================================
    # MOTOR APPROACH (§7.2 — 3D Egocentric Vector Decomposition)
    # =========================================================================

    def get_approach_action_for_anchor(
        self,
        observation: Any,
        target_anchor: Any,
        reject_fn: Any = None,
    ) -> Optional[str]:
        """Public API: extract the priority-ordered movement action toward a
        specific target anchor (§7.2).

        Encapsulates directional extraction without relying on private
        methods, so callers that own a *target* (the UtilityEngine's anchor
        fallback, for example) can ask the reflex for one step of approach to
        exactly that anchor instead of for its own salience-selected target.
        Returns None when no directional action is currently valid or when
        every candidate is rejected by `reject_fn` (e.g. refractory).
        """
        if observation is None or target_anchor is None:
            return None
        directions = self._extract_ego_directions(observation, target_anchor)
        for direction in directions:
            if (
                direction
                and self._action_valid(direction, observation)
                and not (reject_fn and reject_fn(direction))
            ):
                return direction
        return None

    def get_approach_action_with_target(
        self,
        t_or_obs: Any,
        observation: Any = None,
        reject_fn: Any = None,
    ) -> Optional[Tuple[str, Any]]:
        """Flexible call convention supporting (t, obs, reject_fn) and (obs, reject_fn)."""
        if isinstance(t_or_obs, (int, float)):
            t = int(t_or_obs)
            obs = observation
        else:
            obs = t_or_obs
            t = self.t
        if obs is None:
            return None
        self.t = t

        # Bootstrap: run one orienting update if kinematic layer is empty
        if obs is not None and not self.filtered_pos:
            self.update(t, obs)

        raw_entities = self._get_entity_indices(obs)
        entities = [i for i in raw_entities if self._is_present_at(obs, i)]

        for idx in entities:
            cur = self._is_present_at(obs, idx)
            if cur != self.prev_presence.get(idx, cur):
                self.familiar_until[idx] = -1
                self.interaction_counts[idx] = 0
                self.adjacency_ticks[idx] = 0
            self.prev_presence[idx] = cur

        for idx in entities:
            if (
                idx != self.agent_anchor
                and self._adj(obs, self.agent_anchor, idx)
                and not self._is_dynamic(idx)
                and not self.is_entity_satiated(idx, t)
                and self._has_interaction(idx, obs, reject_fn=reject_fn)
            ):
                return None

        cand = []
        for idx in entities:
            if idx == self.agent_anchor:
                continue
            reached = (
                self._adj(obs, self.agent_anchor, idx)
                if not self._is_dynamic(idx)
                else False
            )
            if reached:
                # [Patch 2b] FIX: no satiety on first adjacency. Count
                # adjacency ticks; only satiate after babble_budget ticks of
                # adjacency-without-interaction (fixation guard). Real
                # interactions are handled by notify_interaction.
                if not self._is_dynamic(idx) and t >= self.familiar_until.get(idx, -1):
                    self.adjacency_ticks[idx] = self.adjacency_ticks.get(idx, 0) + 1
                    if self.adjacency_ticks[idx] >= self.babble_budget:
                        self.familiar_until[idx] = t + self.satiety_window
                        self.adjacency_ticks[idx] = 0
                continue
            if self.is_entity_satiated(idx, t):
                continue
            cand.append(idx)

        if not cand:
            return None

        cand.sort(
            key=lambda u: (
                0 if u == self.orienting_target else 1,
                -self.bar_psi.get(u, 0.0),
                u,
            )
        )

        for target in cand:
            for direction in self._extract_ego_directions(obs, target):
                if (
                    direction
                    and self._action_valid(direction, obs)
                    and not (reject_fn and reject_fn(direction))
                ):
                    return (direction, target)
        return None

    def get_approach_action(
        self,
        t_or_obs: Any,
        observation: Any = None,
        reject_fn: Any = None,
    ) -> Optional[str]:
        d = self.get_approach_action_with_target(t_or_obs, observation, reject_fn)
        return d[0] if d else None

    # =========================================================================
    # 3D EGOCENTRIC VECTOR DECOMPOSITION (§7.2)
    # =========================================================================

    def _extract_ego_directions(self, obs: Any, target: Any) -> List[str]:
        """Produce priority-ordered movement actions per §7.2 with monotone
        barrier contouring (Bug Algorithm, corrected ordering).

        Order: (1) direct path clear -> take it, exit contour.
               (2) direct path blocked -> follow committed contour direction.
               (3) no valid commitment -> initiate/reverse contour.
        """
        # --- Env-convention path ---
        if self.env is not None and hasattr(self.env, "get_direction_to_anchor"):
            try:
                env_dir = self.env.get_direction_to_anchor(target)
            except Exception:
                env_dir = None
            if env_dir is not None and self._action_valid(env_dir, obs):
                # Direct path is clear: exit contour mode.
                self.contour_commit.pop(target, None)
                self._contour_steps.pop(target, None)
                return [env_dir]

            # Direct path BLOCKED. Follow active contour commitment.
            c = self.contour_commit.get(target)
            if c is not None and self._action_valid(c, obs):
                self._contour_steps[target] = self._contour_steps.get(target, 0) + 1
                return [c]

            # No valid commitment: initiate or reverse contour.
            self._contour_steps.pop(target, None)
            if hasattr(self.env, "last_positions"):
                pos = self.env.last_positions.get(target)
                if pos is not None and hasattr(self.env, "agent_pos"):
                    delta_env = [
                        float(pos[0] - self.env.agent_pos[0]),
                        float(pos[1] - self.env.agent_pos[1]),
                    ]
                    contour_acts = self._contour_3d(target, obs, delta_env, 2)
                    if contour_acts:
                        return contour_acts

        # --- Internal decomposition path ---
        ap = self.filtered_pos.get(self.agent_anchor)
        tp = self.filtered_pos.get(target)
        if ap is None or tp is None:
            return []

        vel = self.vel.get(target, ())
        if self._is_dynamic(target) and vel:
            tp_effective = tuple(
                tp[d] + (vel[d] if d < len(vel) else 0.0) for d in range(len(tp))
            )
        else:
            tp_effective = tp

        dim = min(len(ap), len(tp_effective))
        delta = [tp_effective[d] - ap[d] for d in range(dim)]

        # STEP 1: Check if the PRIMARY direction toward target is valid.
        # If yes, the direct path is clear — exit contour mode and take it.
        # THIS MUST COME BEFORE THE COMMITMENT CHECK.
        primary_axis = max(range(dim), key=lambda d: abs(delta[d])) if dim > 0 else 0
        primary_sign = 1 if delta[primary_axis] > 0 else -1
        primary_role = _AXIS_SIGN_TO_ROLE.get(primary_axis, {}).get(primary_sign)
        primary_action = self._action_for_role(primary_role, obs) if primary_role else None

        if primary_action is not None and self._action_valid(primary_action, obs):
            self.contour_commit.pop(target, None)
            self._contour_steps.pop(target, None)
            return [primary_action]

        # STEP 2: Primary is BLOCKED. Follow active contour commitment.
        c = self.contour_commit.get(target)
        if c is not None and self._action_valid(c, obs):
            steps = self._contour_steps.get(target, 0) + 1
            self._contour_steps[target] = steps
            # Contour timeout: if sliding too long without the primary
            # direction clearing, force a reversal to try the other way.
            if steps > 2 * (N_ANCHOR_MAX_3D + 5):
                self.contour_commit.pop(target, None)
                self._contour_steps.pop(target, None)
            else:
                return [c]

        # STEP 3: No valid commitment (or timed out). Initiate/reverse contour.
        has_meaningful_direction = any(
            abs(delta[d]) > EPS_DEAD_MOTOR for d in range(dim)
        )
        out = []
        if has_meaningful_direction:
            out = self._contour_3d(target, obs, delta, dim)

        if not out:
            if self._action_valid(ACTION_NOOP, obs):
                out = [ACTION_NOOP]

        return out

    def _action_for_role(self, role: str, obs: Any) -> Optional[str]:
        """Find the first valid action in the repertoire matching the movement role.
        Bug 3 fix: warns once per missing role."""
        candidates = self.role_to_actions.get(role, [])
        for action_name in candidates:
            # [REPERTOIRE GUARD] The reflex owns its own copy of the
            # repertoire and must not depend on the env being strict.
            # The injected `role_to_actions` map may include
            # `sway_*` / `heave_*` for 3D envs; a grid env that never
            # exposes them must not have them proposed. Without this,
            # they leak through whenever the env's `is_action_valid`
            # is permissive, and every selected one burns refractory.
            if self.action_repertoire and action_name not in self.action_repertoire:
                continue
            if self._action_valid(action_name, obs):
                return action_name
        # Warn once per missing role — only when the map has NO candidate
        # at all (missing vocabulary). If candidates exist but are all
        # transiently invalid (e.g. blocked by a wall this step), that is
        # normal and not worth warning about.
        if not candidates and role not in self._warned_roles:
            warnings.warn(
                f"No valid action found for movement role '{role}' in repertoire. "
                f"Checked: {candidates}. Motor approach may produce no proposals "
                f"for this axis.",
                UserWarning,
                stacklevel=2,
            )
            self._warned_roles.add(role)
        return None

    def _contour_3d(
        self, target: Any, obs: Any, delta: List[float], dim: int
    ) -> List[str]:
        """Tactile barrier contouring: slide along the orthogonal axis when
        the primary direction is blocked.

        Corrected: uses contour_commit for monotone sliding. Reverses at
        dead-ends (committed action became invalid). At delta=0, alternates
        the preferred direction using the previous commitment to avoid
        always choosing the same wrong way.
        """
        if dim < 2:
            return []

        prev_c = self.contour_commit.get(target)
        primary_axis = max(range(dim), key=lambda d: abs(delta[d]))
        orthogonal_axes = [d for d in range(dim) if d != primary_axis]

        for d in orthogonal_axes:
            if delta[d] != 0:
                signs = (1 if delta[d] > 0 else -1, -1 if delta[d] > 0 else 1)
            else:
                # At delta=0: alternate based on previous commitment.
                # If previously committed to +axis, try -axis first (and
                # vice versa). This prevents always sliding the wrong way.
                if prev_c is not None:
                    prev_role = None
                    for axis_d, role_map in _AXIS_SIGN_TO_ROLE.items():
                        for sign, role in role_map.items():
                            acts = self.role_to_actions.get(role, [])
                            if prev_c in acts:
                                prev_role = (axis_d, sign)
                    if prev_role is not None and prev_role[0] == d:
                        # Previously went prev_role[1]; try the other first.
                        signs = (-prev_role[1], prev_role[1])
                    else:
                        signs = (1, -1)
                else:
                    signs = (1, -1)

            for sign in signs:
                role = _AXIS_SIGN_TO_ROLE.get(d, {}).get(sign)
                if role is None:
                    continue
                action = self._action_for_role(role, obs)
                if action is not None and action != prev_c:
                    self.contour_commit[target] = action
                    self._contour_steps[target] = 0
                    return [action]

        # All orthogonal directions blocked or same as prev. Try prev reversal.
        if prev_c is not None:
            for d in orthogonal_axes:
                for sign in (1, -1):
                    role = _AXIS_SIGN_TO_ROLE.get(d, {}).get(sign)
                    if role is None:
                        continue
                    action = self._action_for_role(role, obs)
                    if action is not None and action != prev_c:
                        self.contour_commit[target] = action
                        self._contour_steps[target] = 0
                        return [action]

        return []

    # =========================================================================
    # INTERACTION DETECTION (affordance check via _is_nav_action)
    # =========================================================================

    def _has_interaction(
        self, idx: Any, observation: Any, reject_fn: Any = None
    ) -> bool:
        if self.env is None or not hasattr(self.env, "is_action_valid"):
            return False
        manip = [
            a
            for a in (getattr(self.env, "action_rep", None) or self.action_repertoire)
            if not _is_nav_action(a, self.env)
        ]
        return any(
            self.env.is_action_valid(a, observation)
            and not (reject_fn and reject_fn(a))
            for a in manip
        )

    # =========================================================================
    # ADJACENCY AND DYNAMIC CLASSIFICATION (PRESERVED UNCHANGED)
    # =========================================================================

    def _adj(self, observation: Any, i: int, j: int) -> bool:
        if not hasattr(observation, "get_adjacency"):
            return False
        return observation.get_adjacency(i, j) == 1

    def _is_dynamic(self, u: Any) -> bool:
        return self._norm(self.vel.get(u)) > V_THRESH

    def _action_valid(self, action: Any, obs: Any) -> bool:
        if self.env is None or not hasattr(self.env, "is_action_valid"):
            return True
        return bool(self.env.is_action_valid(action, obs))

    @staticmethod
    def _dist(a: Optional[Tuple[float, ...]], b: Optional[Tuple[float, ...]]) -> float:
        if a is None or b is None:
            return float("inf")
        return _math_sqrt(sum((a[d] - b[d]) ** 2 for d in range(min(len(a), len(b)))))

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self) -> None:
        self._contour_steps.clear()
        self.__init__(
            self.env,
            self.agent_anchor,
            self.action_repertoire,
            self.cg,
            self.satiety_window,
            self.babble_budget,
        )
