"""
lenses.py — Canonical lens module for Blueprint v31.

Implements §2.2: seven lenses, each producing a single-detection node
about ONE entity at ONE moment.

    Boundary — persistence of an enclosed region
    Color    — mean RGB of a region
    Motion   — velocity of a single entity
    Size     — spatial extent of a region
    Texture  — surface regularity of a region
    Position — egocentric coordinates of a region
    Onset    — recency of a presence transition

Plus the B2 first-class Action lens (an eighth, non-homeostatic channel):

    Action   — the agent's own executed action as a categorical pattern
               node (OPAQUE exact-identity signature). Emitted for the
               previous step's action; its ONSET is pre-registered at
               execution time by the Pipeline (Binder.note_executed_action)
               so SEQUENCE edges attribute effects to the action that
               caused them. Vocabulary recovery for §5.3.4 replay reads
               the ConceptGraph action-alpha registry keyed by the lens's
               pattern nodes.

Each lens:
  - reads properties of a single entity (never of two or more);
  - emits a Detection with a CANONICAL TypedSignature;
  - maintains its own homeostatic state (theta, r_bar, C, F, M_prev);
  - uses the shared emission mechanism in _emit (§2.7).

Modality handling
-----------------
The 3D path (Observation3D) exposes per-anchor physical channels directly
through get_position / get_velocity / get_radius / get_planarity /
get_agent_index. The symbolic path (SymbolicObservation) does not expose
these; only Boundary and Onset run for symbolic modality. This is not a
limitation of the lens design — the physical properties required by the
other lenses are simply not present in symbolic observations.

Color lens dependency
---------------------
The Color lens calls obs.get_anchor_rgb(u) if present. The current
Observation3D does not implement this getter; it is the adapter's
responsibility to compute per-anchor mean RGB from the raw frame. Until
sensory_adapter.py provides it, Color emits nothing. The interface is
defined here so that the adapter change is a one-place update.

Determinism
-----------
Every comparison, tie-break, and ordering in this module is by canonical
bytes (serialize_value, serialize_tuple). Same input sequence produces
byte-identical output, run to run.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from core.hashing import (  # noqa: F401
    clamp,
    hash_64,
    hash_64_signed,
    serialize_bytes,
    serialize_float,
    serialize_int,
    serialize_set,
    serialize_tuple,
    serialize_value,
)
from core.registries import (  # noqa: F401
    CONSTITUTIONAL_RANGES,
    is_action_lens_eligible,
)
from core.quantize import (  # noqa: F401
    _MAX_BINS,
    _q_linear_safe,
    _q_signed,
    _q_unsigned,
    q_linear,
)
import numpy as np
import parameters as P

# TypedSignature and Detection are defined in sensory_floor. Importing
# them here is safe as long as sensory_floor does NOT import lenses at
# module scope. The eventual pipeline integration must lazy-import
# LensExecutor inside SensoryFloor.__init__ or _execute_raw_lenses.
from sensory_floor import (
    TypedSignature,
    Detection,
    ACTION_LENS_NAME,
    ACTION_ROLE_NAME,
    action_signature,
    polar_boundary_fragments,
)
from scipy.ndimage import gaussian_filter


def _second_peak(e: np.ndarray, min_sep: int = 2) -> Tuple[int, int, float]:
    """[AUDIT #13 FIX] Second LOCAL MAXIMUM at least `min_sep` bins away.
    Returns (i1, i2, e2); e2 == 0.0 when no such peak exists.
    """
    n = len(e)
    i1 = int(np.argmax(e))
    best, j2 = 0.0, i1
    for j in range(n):
        if min(abs(j - i1), n - abs(j - i1)) < min_sep:
            continue
        if e[j] >= e[(j - 1) % n] and e[j] >= e[(j + 1) % n] and e[j] > best:
            best, j2 = float(e[j]), j
    return i1, j2, best


def motion_energy_maps(
    Ix: np.ndarray,
    Iy: np.ndarray,
    It: np.ndarray,
    dirs: Sequence[float],
    rf_sigma: float,
) -> np.ndarray:
    """[AUDIT #13 FIX] Direction-energy POPULATION accumulated over the RF."""
    return np.stack([
        gaussian_filter(
            np.maximum(0.0, -(Ix * math.cos(a) + Iy * math.sin(a)) * It),
            rf_sigma,
        )
        for a in dirs
    ])


EPSILON = P.EPSILON

_LENS_NAMES: Tuple[str, ...] = (
    "Boundary",
    "Color",
    "Motion",
    "Size",
    "Texture",
    "Position",
    "Onset",
    "Orientation",
    "EffectPresent",
    "Polarity",
    "OrientationPol",
)

# [Fix 1 — V1 field tiling]
# The 64-default is a global emission cap with no neural analog. V1
# populations tile the visual field densely; selection is by local
# content, not by a fixed population budget. At GRID_N=32 (1024 regions)
# the 64-cap silently windowed 94% of the field to the byte-order-first
# 64 anchors (all presence-EMA responses tied → `_emit`'s top-K
# tie-break fell through to `serialize_tuple([serialize_int(N),
# serialize_value(a)])`, a fixed image-independent subset).
#
# Caps are raised to cover a 64x64 region grid (4096) with headroom for
# every field-tiling V1 lens. `_emit` already reads
# `_LENS_DETECTION_CAPS.get(lens_name, _DEFAULT_LENS_DETECTION_CAP)`
# (lenses.py, _emit), so adding entries here is the entire change.
#
# Self-audit: Position, Motion, Boundary, and Orientation are the same
# class of field-tiling channel. Position feeds DEGREE/TOPOLOGY/CONTACT;
# Motion feeds CO_MOTION; Boundary feeds Rule-2 family resolution;
# Orientation is cap-bound on every raw image (investigation §2). All
# four are included here.
_DEFAULT_LENS_DETECTION_CAP = 64  # unchanged; retains legacy semantics
                                  # for event-based lenses (Onset).
_FIELD_TILING_LENS_CAP = 4096     # 64x64 regions

_LENS_DETECTION_CAPS: Dict[str, int] = {
    "Orientation": _FIELD_TILING_LENS_CAP,  # [v9] raised from 128; the
                                            # 128 cap dropped 80-90% of
                                            # gradient-gate survivors on
                                            # GRID_N=32.
    "Color": _FIELD_TILING_LENS_CAP,
    "Texture": _FIELD_TILING_LENS_CAP,
    "Position": _FIELD_TILING_LENS_CAP,
    "Motion": _FIELD_TILING_LENS_CAP,
    "Boundary": _FIELD_TILING_LENS_CAP,
    "Polarity": _FIELD_TILING_LENS_CAP,  # [B.30] per-anchor field lens
    "OrientationPol": _FIELD_TILING_LENS_CAP,  # [option (b)] per-anchor
}
_ONSET_WINDOW = 10  # Steps over which onset response decays
# Presence EMA decay now lives in parameters.py as PRESENCE_EMA_BETA
# (single source of truth with sensory_floor's β_T).


# =============================================================================
# LENS STATE
# =============================================================================


def _init_lens_state() -> Dict[str, Any]:
    """Per-lens homeostatic state. Structurally identical to the binder
    threshold mechanism of §2.7 (θ_channel_base, r_bar, C, F, M_prev)."""
    return {
        "theta": P.THETA_CHANNEL_BASE_INIT,
        "r_bar": 0.0,
        "C": 0.0,
        "F": 0.0,
        "M_prev": {},
    }


# =============================================================================
# QUANTIZERS
# =============================================================================










def _q_log_unsigned(value: float, max_bins: int = 10) -> int:
    """Quantize a non-negative scalar in log space (fixed Step 4 quantizer).

    Closed form (blueprint §4):
      val_log  = ln(clamp(v, 0.0, 1.0) + 1e-4)
      norm     = (val_log - ln(1e-4)) / (ln(1.0 + 1e-4) - ln(1e-4))
      bin      = min(n_bins - 1, max(0, floor(norm * n_bins)))

    Equal-log-spacings across bins 0..n_bins-1: low-magnitude gradient
    variance is no longer floored into bin 0 (old step=0.1 collapsed
    ~98.6% of regions to bin 0).
    """
    if max_bins <= 0:
        raise ValueError(f"max_bins must be positive, got {max_bins}")
    v = max(0.0, min(1.0, float(value)))
    log_lo = math.log(1e-4)
    log_hi = math.log(1.0 + 1e-4)
    norm = (math.log(v + 1e-4) - log_lo) / (log_hi - log_lo)
    return min(max_bins - 1, max(0, int(math.floor(norm * max_bins))))


def _anchor_class_bin(obs: Any, anchor: Any) -> float:
    """Coarse position class for an anchor, in [0, 15].

    Used in canonical signatures whose slot-2 value must agree across
    images for the same spatial region.

    Design:
      - 4 x-bins x 4 y-bins = 16 classes. Coarser than the prior 8x8
        because position normalization in the observation is
        aspect-ratio-dependent (portrait and landscape images map the
        same physical grid to different cells of a finer lattice). A
        4x4 lattice tolerates ~25% position shift without class change;
        8x8 tolerates ~12%.
      - Positions must be normalized to [-1, 1] for the coarse bin to
        be meaningful. If the observation returns coordinates outside
        that range (e.g. raw world coordinates from Observation3D),
        fall back to the anchor hash so no class degenerates.

    Same physical region across images → same class → same slot-2 value
    → cross-image matching is possible. Different regions → different
    classes → the diff saturates similarity to 0 only when the diff is
    >= 16, which is the full range of the 4x4 lattice.
    """
    getter = getattr(obs, "get_position", None)
    if callable(getter):
        try:
            pos = getter(anchor)
        except (KeyError, IndexError, TypeError):
            pos = None
        if pos is not None and len(pos) >= 2:
            try:
                x = float(pos[0])
                y = float(pos[1])
            except (TypeError, ValueError):
                x = y = float("nan")
            if (
                math.isfinite(x)
                and math.isfinite(y)
                and -1.0 <= x <= 1.0
                and -1.0 <= y <= 1.0
            ):
                x_bin = _q_signed(x, max_bins=4)   # in [0, 3]
                y_bin = _q_signed(y, max_bins=4)   # in [0, 3]
                return float(x_bin * 4 + y_bin)    # in [0, 15]
    return float(hash_64(serialize_value(anchor)) & ((1 << 32) - 1))


def _segment_polyline(
    polyline: List[Tuple[int, int]],
) -> List[Tuple[float, float, Tuple[float, float]]]:
    """[v9/v10.2] Split a boundary polyline into equal-arc segments.

    Returns [(angle_rad, arc_len_px, midpoint), ...].

    angle: chord direction, undirected, in [0, π).
    arc_len: segment arc length in pixels.
    midpoint: point EXACTLY on the polyline at the segment's half-arc
      position (via _point_at_arc interpolation), not the chord midpoint.

    Does NOT call sensory_floor._polyline_arc_length: builds the
    cumulative-arc array locally and derives total = cum[-1].
    """
    n = len(polyline)
    if n < 2:
        return []

    cum = [0.0] * n
    for i in range(1, n):
        dr = polyline[i][0] - polyline[i - 1][0]
        dc = polyline[i][1] - polyline[i - 1][1]
        cum[i] = cum[i - 1] + math.sqrt(dr * dr + dc * dc)
    total = cum[-1]

    if total < P.ORIENTATION_SEGMENT_MIN_ARC:
        return []
    arc_per_seg = max(
        P.ORIENTATION_SEGMENT_MIN_ARC,
        total / P.ORIENTATION_SEGMENT_TARGET_N,
    )
    num_segs = int(math.floor(total / arc_per_seg))
    if num_segs < 1:
        num_segs = 1
    if num_segs > P.ORIENTATION_SEGMENT_MAX_PER_REGION:
        num_segs = P.ORIENTATION_SEGMENT_MAX_PER_REGION
    seg_len = total / num_segs

    def _point_at_arc(target_arc: float) -> Tuple[float, float]:
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < target_arc:
                lo = mid + 1
            else:
                hi = mid
        i = max(1, lo)
        arc0, arc1 = cum[i - 1], cum[i]
        if arc1 <= arc0:
            return float(polyline[i][0]), float(polyline[i][1])
        t = (target_arc - arc0) / (arc1 - arc0)
        r = polyline[i - 1][0] + t * (
            polyline[i][0] - polyline[i - 1][0]
        )
        c = polyline[i - 1][1] + t * (
            polyline[i][1] - polyline[i - 1][1]
        )
        return float(r), float(c)

    segments: List[Tuple[float, float, Tuple[float, float]]] = []
    for k in range(num_segs):
        a_start = k * seg_len
        a_end = (k + 1) * seg_len if k < num_segs - 1 else total
        p_start = _point_at_arc(a_start)
        p_end = _point_at_arc(a_end)
        dr = p_end[0] - p_start[0]
        dc = p_end[1] - p_start[1]
        if abs(dr) < 1e-9 and abs(dc) < 1e-9:
            continue
        angle = math.atan2(dr, dc) % math.pi
        mid = _point_at_arc((a_start + a_end) / 2.0)
        segments.append((angle, a_end - a_start, mid))
    return segments




def _find_local_peaks(x: Any, min_val: float) -> List[Tuple[int, int, float]]:
    """[V9 Phase 4c / draft B.7] Local maxima over a (theta, sf) grid.

    Theta axis is circular; the sf axis is not.
    """
    nt, nsf = x.shape
    peaks = []
    for it in range(nt):
        for isf in range(nsf):
            v = x[it, isf]
            if v < min_val:
                continue
            is_peak = True
            for dit in (-1, 0, 1):
                nit = (it + dit) % nt
                for disf in (-1, 0, 1):
                    if dit == 0 and disf == 0:
                        continue
                    nisf = isf + disf
                    if 0 <= nisf < nsf and x[nit, nisf] > v:
                        is_peak = False
                        break
                if not is_peak:
                    break
            if is_peak:
                peaks.append((it, isf, float(v)))
    return peaks


def _log_contrast_bin(value: float, n_bins: int) -> int:
    """[V9 Phase 4c / draft B.7] Log-compressed contrast bin index."""
    v = max(0.0, value)
    lv = math.log1p(v * 10.0)
    lmax = math.log1p(10.0)
    q = int(math.floor((lv / lmax) * n_bins))
    return min(n_bins - 1, max(0, q))

def _q_arc_len(arc_px: float) -> int:
    """Quantize arc length into [0, ORIENTATION_SEGMENT_N_ARC - 1]."""
    step = P.ORIENTATION_SEGMENT_ARC_MAX / P.ORIENTATION_SEGMENT_N_ARC
    q = int(math.floor(arc_px / step))
    return min(P.ORIENTATION_SEGMENT_N_ARC - 1, max(0, q))



# =============================================================================
# LENS EXECUTOR
# =============================================================================


class LensExecutor:
    """Runs all v31 lenses over a single step and returns the union of
    their detections.

    Stateful. Each lens maintains its own homeostatic state and the
    executor maintains per-anchor history used by Boundary, Texture,
    and Onset. Call run_all once per step in step order.
    """

    def __init__(self, agent_anchor: Any = 0) -> None:
        self.state: Dict[str, Dict[str, Any]] = {
            name: _init_lens_state() for name in _LENS_NAMES
        }
        self.t: int = -1
        self._first_step: bool = True
        # Agent atom convention (symbolic envs place the agent at atom 0;
        # main.py wires InnateReflex with the same value). Action
        # detections carry this as their alpha/anchor.
        self.agent_anchor: Any = agent_anchor

        # [Arm C] Boundary anchor capability gate. Some observations expose
        # per-entity shape channels (Observation2D / Observation3D carry
        # `get_circularity`); symbolic envs (SymbolicObservation) do not.
        # The probe cannot run in __init__ because the executor is built
        # before the first observation exists, so `_run_boundary` probes
        # lazily on first emission and caches the answer here.
        # None = not yet probed.
        self._has_shape_channels: Optional[bool] = None

        # [v8] Orientation capability gate. Probed once on the first
        # `_run_orientation` call, cached thereafter. True iff the
        # observation exposes `get_orientation` (Observation2D with
        # major-axis info; Observation3D and SymbolicObservation
        # return False until their adapters add the accessor).
        self._has_orientation_channels: Optional[bool] = None

        # [Phase B B.14] Staged by run_all for _run_orientation; always
        # None outside a run_all call. Initialised here so a direct
        # _run_orientation call cannot AttributeError.
        self._current_gain_by_anchor: Optional[Dict[Any, float]] = None

        # [Tier 3] EffectPresence tracking. Both dicts are reinitialized
        # by reset() (which calls __init__), so no separate reset path is
        # needed. `_prev_present_effects` is the presence-filtered effect
        # set from the previous step; `_recently_risen_effects` maps
        # effect atom -> expiry step.
        self._prev_present_effects: Set[int] = set()
        self._recently_risen_effects: Dict[int, int] = {}
        self._effect_present_emitted_total: int = 0

        # Per-anchor history
        self._presence_ema: Dict[Any, float] = defaultdict(float)
        self._presence_prev: Dict[Any, bool] = {}
        self._absence_duration: Dict[Any, int] = defaultdict(int)
        self._onset_step: Dict[Any, int] = {}
        self._onset_signature_bin: Dict[Any, int] = {}
        self._prev_planarity: Dict[Any, float] = {}
        self._planarity_delta_ema: Dict[Any, float] = defaultdict(float)

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def run_all(
        self,
        S_t: Any,
        obs: Any,
        t: int,
        dir_tuples: Optional[List[Tuple[int, int, int]]] = None,
        prev_action: Optional[Any] = None,
        concept_effects: Optional[Set[int]] = None,
        gain_by_anchor: Optional[Dict[Any, float]] = None,
        envelope_mask: Optional[
            Tuple[Tuple[float, float, float, float], ...]
        ] = None,
        update_homeostasis: bool = True,
        object_envelope: Optional[
            Tuple[float, float, float, float]
        ] = None,
        scale_level: int = 0,
    ) -> Set[Detection]:
        """Run all lenses over the current step.

        S_t  — StatePattern constructed this step.
        obs  — Observation (Observation3D, SymbolicObservation, or
               ContinuousObservation) used to build S_t.
        t    — Current step index.
        dir_tuples — Optional egocentric direction tuples from the
               InnateReflex. Unused by lenses; accepted for interface
               stability with the v30 pipeline.
        prev_action — The action executed at the end of the previous step
               (B2 Action lens input). None ⇒ no Action detection.
        gain_by_anchor — [Phase B B.14] per-anchor recurrent-grouping gain,
               keyed by the SAME anchor form `_run_orientation` uses (`u`
               for single-region, `(u, seg_idx)` for segments). None or
               empty ⇒ no gain applied, which is the default.
        """
        # Staged on the instance so _run_orientation can read it without a
        # signature change on a dozen call sites. Cleared in the finally
        # below: leaving a stale dict bound would let a later direct
        # _run_orientation call (outside run_all) inherit last step's gain.
        self._current_gain_by_anchor = gain_by_anchor
        # [V9 Phase 4g] Top-down state, staged for the duration of this
        # call in the same idiom as gain_by_anchor. NOTE: no V1 emitter
        # consumes envelope_mask / object_envelope / scale_level yet --
        # the draft's `_emit`-side envelope gain is a separate change.
        # `update_homeostasis` IS consumed here: False skips the
        # presence update, which is what the settle loop's coarse pass
        # requires so it cannot pollute lens state.
        self._current_envelope_mask = envelope_mask
        self._current_object_envelope = object_envelope
        self._current_scale_level = scale_level
        self.t = t

        if update_homeostasis:
            self._update_presence(S_t, t)

        try:
            detections: Set[Detection] = set()
            detections.update(self._run_boundary(S_t, obs))
            detections.update(self._run_contour_fragments(S_t, obs))
            detections.update(self._run_color(S_t, obs))
            detections.update(self._run_motion(S_t, obs))
            # [PHASE F] Size lens purged (tautology on a uniform patch grid);
            # body retained, emission gated.
            if P.SIZE_LENS_ENABLED:
                detections.update(self._run_size(S_t, obs))
            detections.update(self._run_texture(S_t, obs))
            detections.update(self._run_position(S_t, obs))
            detections.update(self._run_onset(S_t))
            detections.update(
                self._run_effect_presence(obs, concept_effects)
            )
            detections.update(self._run_orientation(S_t, obs))
            detections.update(self._run_orientation_pol(S_t, obs))
            detections.update(self._run_action(prev_action))
        finally:
            self._current_gain_by_anchor = None
            self._current_envelope_mask = None
            self._current_object_envelope = None
            self._current_scale_level = 0

        self._first_step = False
        return detections

    # -------------------------------------------------------------------------
    # Action lens (B2, §5.3.4) — the agent's own executed action as a
    # first-class pattern node.
    # -------------------------------------------------------------------------

    def make_action_detection(self, action: Any, t: int) -> Optional[Detection]:
        """Construct the Action-lens Detection for `action` at step `t`,
        or None if the action is not lens-eligible.

        Public because the Pipeline pre-registers the action pattern's
        ONSET at execution time (end of step t, via
        Binder.note_executed_action) so that SEQUENCE edges attribute the
        effect patterns onsetting at t+1 to the action that caused them
        (lag = 1). Emitting only at t+1 — when the lens observes
        a_prev — would onset the action pattern in the SAME step as its
        effects; SEQUENCE requires strictly positive lag, so the
        action→effect edge could never form and the action would instead
        be misattributed as the parent of the NEXT step's effects. The
        pre-registration and this emission produce the identical pattern
        key (lens name + OPAQUE signature hash of the exact action), so
        the t+1 emission extends presence without re-onsetting.

        Deliberate divergence from _emit (§2.7 homeostasis): the Action
        lens is a categorical RECORD of the agent's own output, not a
        perceptual channel competing for emission bandwidth. Routing it
        through theta/fatigue suppression could silently drop an executed
        action from the vocabulary — losing the exact fact §5.3.4 replay
        must recover. z = 1.0 (maximal, certain salience), unconditional.
        """
        if not is_action_lens_eligible(action):
            return None
        sigma = action_signature(action)
        rho = ((ACTION_ROLE_NAME, action, 0, "+"),)
        return Detection(
            k=ACTION_LENS_NAME,
            source_layer=0,
            scale=0,
            sigma=sigma,
            alpha=self.agent_anchor,
            z=1.0,
            t=t,
            rho=rho,
            substrate_hash=sigma.hash_value,
        )

    def _run_action(self, prev_action: Optional[Any]) -> List[Detection]:
        if prev_action is None:
            return []
        det = self.make_action_detection(prev_action, self.t)
        return [det] if det is not None else []

    def reset(self) -> None:
        """Clear all state. For use across episodes.

        [v8] Preserves the configured `agent_anchor`; the previous
        `self.__init__()` form reset it to the default.
        """
        self.__init__(self.agent_anchor)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Read-only snapshot of lens state for diagnostics.

        EffectPresent bypasses _emit, so its channel state remains at
        initial values. The real signal is emitted_total, added below.
        """
        snapshot = {
            name: {
                "theta": self.state[name]["theta"],
                "r_bar": self.state[name]["r_bar"],
                "C": self.state[name]["C"],
                "F": self.state[name]["F"],
                "M_prev_size": len(self.state[name]["M_prev"]),
            }
            for name in _LENS_NAMES
        }
        # Requires "EffectPresent" in _LENS_NAMES (Edit B1). Both edits
        # are in this commit; applied out of order this raises KeyError.
        snapshot["EffectPresent"]["emitted_total"] = (
            self._effect_present_emitted_total
        )
        return snapshot

    # -------------------------------------------------------------------------
    # History maintenance
    # -------------------------------------------------------------------------

    def _update_presence(self, S_t: Any, t: int) -> None:
        """Update presence EMA for physical entities AND active flags/effects.

        Runs once per step, before any lens runs. Guarantees that every
        anchor appearing in S_t.entity_indices has an up-to-date entry
        in _presence_ema.

        [FLAG/EFFECT VISIBILITY §2.2.7] Presence is now tracked for active
        flags and effects as well as physical entities. Switch flips and
        door openings otherwise produce no presence transition, hence no
        Onset detection, hence no causal antecedent/consequent pair at the
        binder — leaving incidental motion as the only visible correlate
        of the transition.
        """
        # Include entities + flags/effects that are currently active
        current = set(S_t.entity_indices)
        if hasattr(S_t, "raw_indices"):
            for i in S_t.raw_indices:
                # Channel 5 is Flag, Channel 6 is Effect; Ch 0 carries the
                # value. A flag/effect only counts as "present" while its
                # value is active, so a 1 -> 0 transition registers as an
                # absence and a 0 -> 1 transition as an onset.
                if (
                    S_t.X[i][5] == 1.0 or S_t.X[i][6] == 1.0
                ) and S_t.X[i][0] > 0.5:
                    current.add(i)

        # Update presence EMA for every anchor ever seen plus every anchor
        # currently visible. Absent anchors decay toward zero; present
        # anchors climb toward one.
        all_anchors = set(self._presence_prev.keys()) | current
        beta_presence = P.PRESENCE_EMA_BETA
        for u in all_anchors:
            if u in current:
                self._presence_ema[u] = (
                    beta_presence * self._presence_ema[u]
                    + (1.0 - beta_presence)
                )
            else:
                self._presence_ema[u] = (
                    beta_presence * self._presence_ema[u]
                )

        # Detect transitions and record onset signature.
        for u in current:
            prev_present = self._presence_prev.get(u, False)
            if not prev_present:
                self._onset_step[u] = t
                absence = self._absence_duration.get(u, 0)
                self._onset_signature_bin[u] = _q_unsigned(
                    float(absence), P.Q_T, max_bins=_MAX_BINS - 1
                )
                self._absence_duration[u] = 0
            else:
                self._absence_duration[u] = 0
            self._presence_prev[u] = True

        # Increment absence duration for anchors that are now missing.
        for u in list(self._presence_prev.keys()):
            if u not in current:
                if self._presence_prev[u]:
                    self._absence_duration[u] = 1
                else:
                    self._absence_duration[u] = self._absence_duration.get(u, 0) + 1
                self._presence_prev[u] = False

        # Prune per-anchor state for anchors absent longer than the onset
        # window. Past _ONSET_WINDOW, an anchor cannot emit onset detections,
        # and its presence_ema has decayed below any lens threshold, so the
        # retained state is dead weight that would grow without bound.
        stale = [
            u for u in list(self._presence_prev.keys())
            if u not in current
            and self._absence_duration.get(u, 0) > _ONSET_WINDOW
        ]
        for u in stale:
            self._presence_ema.pop(u, None)
            self._presence_prev.pop(u, None)
            self._absence_duration.pop(u, None)
            self._onset_step.pop(u, None)
            self._onset_signature_bin.pop(u, None)

    # -------------------------------------------------------------------------
    # Shared emission mechanism
    # -------------------------------------------------------------------------

    def _emit(
        self,
        lens_name: str,
        responses: List[float],
        anchors: List[Any],
        signature_builder: Callable[[Any, int], Tuple[TypedSignature, Tuple]],
        tiebreaker_fn: Callable[[Any], bytes],
        normalize_by_max: bool = True,
        update_homeostasis: bool = True,
    ) -> List[Detection]:
        """Shared lens emission mechanism, structurally identical to the
        v30 lens emitter and to the binder channel mechanism of §2.7.

        Normalizes responses, applies homeostatic theta with hysteresis,
        emits up to D_MAX_DETECTIONS_PER_LENS detections, and updates
        the lens's homeostatic state.
        """
        state = self.state[lens_name]

        if len(anchors) != len(responses):
            raise ValueError(
                f"lens {lens_name}: len(anchors)={len(anchors)} != len(responses)={len(responses)}"
            )

        if not responses:
            return []

        max_resp = max(responses)
        if max_resp <= 0.0:
            return []

        if normalize_by_max:
            norm_resp = [r / max_resp for r in responses]
        else:
            norm_resp = list(responses)

        theta_eff = clamp(
            state["theta"] - P.ETA_CTX_CHANNEL * state["C"] + P.ETA_FAT_CHANNEL * state["F"],
            P.THETA_ACT_FLOOR,
            P.THETA_ACT_MAX,
        )

        identities: List[bytes] = [tiebreaker_fn(a) for a in anchors]

        M_prev: Dict[bytes, int] = state["M_prev"]
        M_hyst: List[int] = []
        for idx in range(len(anchors)):
            prev = M_prev.get(identities[idx], 0)
            threshold = (
                theta_eff if self._first_step
                else theta_eff + P.DELTA_CHANNEL_HYST * (1 - 2 * prev)
            )
            M_hyst.append(1 if norm_resp[idx] >= threshold else 0)

        z_list: List[float] = []
        for idx, m in enumerate(M_hyst):
            if m == 1:
                z = clamp(
                    (norm_resp[idx] - theta_eff) / (1.0 - theta_eff + EPSILON),
                    0.0,
                    1.0,
                )
            else:
                z = 0.0
            z_list.append(z)

        candidates = list(zip(z_list, identities, anchors, range(len(anchors))))
        candidates.sort(key=lambda x: (-x[0], x[1]))
        cap = _LENS_DETECTION_CAPS.get(lens_name, _DEFAULT_LENS_DETECTION_CAP)
        selected = candidates[:cap]

        detections: List[Detection] = []
        for z, _ident, anchor, orig_idx in selected:
            if z <= 0.0:
                continue
            sigma, rho = signature_builder(anchor, orig_idx)
            detections.append(
                Detection(
                    k=lens_name,
                    source_layer=0,
                    scale=0,
                    sigma=sigma,
                    alpha=anchor,
                    z=z,
                    t=self.t,
                    rho=rho,
                )
            )

        # [PASS 27 IN-PLACE MERGE FIX] Merge base candidate states without overwriting M_prev
        for idx, ident in enumerate(identities):
            state["M_prev"][ident] = M_hyst[idx]

        # Bound memory growth over long episodes via LRU pruning
        max_m_prev = getattr(P, "MAX_M_PREV_SIZE", 50000)
        if len(state["M_prev"]) > max_m_prev:
            excess = len(state["M_prev"]) - (max_m_prev // 2)
            for k in list(state["M_prev"].keys())[:excess]:
                del state["M_prev"][k]

        # Optional single-pass homeostatic update
        if update_homeostasis:
            N_active = len(anchors)
            emitted_count = len(detections)
            r = emitted_count / (N_active + EPSILON) if N_active > 0 else 0.0
            state["r_bar"] = (1.0 - P.KAPPA_R_CHANNEL) * state["r_bar"] + P.KAPPA_R_CHANNEL * r
            state["theta"] = clamp(
                state["theta"] + P.KAPPA_CHANNEL_THETA * (state["r_bar"] - P.RHO_CHANNEL_TARGET),
                P.THETA_CHANNEL_MIN,
                P.THETA_CHANNEL_MAX,
            )
            had_emission = 1.0 if emitted_count > 0 else 0.0
            state["C"] = (1.0 - P.KAPPA_C_CHANNEL) * state["C"] + P.KAPPA_C_CHANNEL * had_emission
            state["F"] = (
                P.BETA_F_CHANNEL * state["F"] + (1.0 - P.BETA_F_CHANNEL) * had_emission
            )

        return detections

    # -------------------------------------------------------------------------
    # Boundary lens — persistence of an enclosed region
    # -------------------------------------------------------------------------

    def _run_contour_fragments(self, S_t: Any, obs: Any = None) -> List[Detection]:
        """[Stage 3] Pasupathy & Connor object-centred polar boundary fragments.

        Stream-2 contour polylines are subdivided into arc fragments and each is
        encoded as (kappa, theta, r, tangent-vs-radial) relative to the object
        visual-mass centroid. Emitted directly (bypassing the homeostatic
        `_emit` path, the way the Action lens does) as `CONTOUR_FRAGMENT`
        detections anchored on the owning contour region.
        """
        if obs is None or not hasattr(obs, "get_contours"):
            return []
        frs, meta = polar_boundary_fragments(
            obs.get_contours(),
            P.CONTOUR_FRAGMENT_LEN_FRAC,
            P.CONTOUR_FRAGMENT_KAPPA_CAP,
        )
        if not frs:
            return []
        anchor_of: Dict[Any, Any] = {}
        for u, r in enumerate(getattr(obs, "_regions", [])):  # TODO(agent): use a public accessor if one exists
            ci = r.get("_contour_idx")
            if ci is not None:
                anchor_of.setdefault(ci, u)
        out: List[Detection] = []
        for f in frs:
            a = anchor_of.get(f["contour"], f["contour"])
            sig = TypedSignature(kind="CANONICAL", class_="CONTOUR_FRAGMENT",
                                 relation="OBJECT_CENTERED_V4", quantized=f["bins"])
            out.append(Detection(k="ContourFragment", source_layer=1, scale=0, sigma=sig, alpha=a,
                                 z=min(1.0, f["arc"] / meta["R"]), t=getattr(S_t, "t", 0),
                                 rho=(("entity", a, 0, "+"), ("fragment", f["frag"], 0, "+"))))
        return out

    def _run_boundary(self, S_t: Any, obs: Any = None) -> List[Detection]:
        """Boundary lens: isoperimetric circularity + bounding-box fill
        complexity (CONTOUR_PROFILE). Emits class `BOUNDARY_ANCHORED`
        (arity 3: shape bins + anchor trunk) when the observation exposes
        per-entity shape channels, and class `BOUNDARY` (arity 2, the
        pre-anchor-fix collapsed form) when it does not. See the
        capability gate in `_sig` below and the paired registry entries in
        `primitives.CONSTITUTIONAL_RANGES`."""
        responses: List[float] = []
        anchors: List[Any] = []
        boundary_bins: List[Tuple[int, int]] = []
        step = 1.0 / 10.0

        for u in S_t.entity_indices:
            circ = getattr(obs, "get_circularity", lambda idx: 0.5)(u)
            comp = getattr(obs, "get_complexity", lambda idx: 0.5)(u)

            c_bin = _q_unsigned(clamp(float(circ), 0.0, 1.0), step, max_bins=9)
            comp_bin = _q_unsigned(clamp(float(comp), 0.0, 1.0), step, max_bins=9)

            p = max(self._presence_ema.get(u, 1.0), EPSILON)
            responses.append(p)
            anchors.append(u)
            boundary_bins.append((c_bin, comp_bin))

        # [Arm C] Capability gate. Probe the observation ONCE for per-entity
        # shape channels (`get_circularity`) and cache the answer: the
        # executor is constructed before the first observation exists, so
        # the probe cannot live in __init__. Observations that carry shape
        # channels keep the anchor-in-signature form (class
        # BOUNDARY_ANCHORED, arity 3); observations that do not fall back
        # to the pre-anchor-fix collapsed form (class BOUNDARY, arity 2),
        # where two entities quoted with the same boundary shape
        # deliberately map to one pattern node.
        # [v8] The probe latches on the first observation. The executor
        # must not be shared across modalities within a run; construct
        # separate executors if mixed-modality runs are needed.
        if self._has_shape_channels is None:
            self._has_shape_channels = hasattr(obs, "get_circularity")

        if self._has_shape_channels:
            def _sig(anchor: Any, idx: int) -> Tuple[TypedSignature, Tuple]:
                c_b, comp_b = boundary_bins[idx]
                # [ANCHOR-IN-SIGNATURE FIX / Phase 2 item 1] Fold the entity
                # anchor into the signature so two different entities with
                # the same boundary shape no longer collapse to the same
                # pattern node under §1.11 identity
                # [ANCHOR-IN-SIGNATURE FIX / revised] The anchor occupies
                # slot 2 as a coarse position class in [0, 15] (4x4
                # lattice). Same physical region across images yields
                # the same class; aspect-ratio-induced position shifts
                # up to ~25% stay within the same cell. The slot's
                # registered range is 0.0 — any nonzero diff contributes
                # directly to the sum-form similarity numerator, so class
                # diff of 16 (full range) saturates to 0 while smaller
                # diffs are graded.
                anchor_bin = _anchor_class_bin(obs, anchor)
                sigma = TypedSignature(
                    kind="CANONICAL",
                    class_="BOUNDARY_ANCHORED",
                    relation="CONTOUR_PROFILE",
                    quantized=(float(c_b), float(comp_b), float(anchor_bin)),
                )
                return sigma, (("entity", anchor, 0, "+"),)
        else:
            def _sig(anchor: Any, idx: int) -> Tuple[TypedSignature, Tuple]:
                c_b, comp_b = boundary_bins[idx]
                # [Arm C] Collapsed form for shape-channel-free
                # observations: arity 2, no anchor term. Matches
                # `CONSTITUTIONAL_RANGES["BOUNDARY"] = (10, 10)` and
                # reproduces pre-anchor-fix behaviour (one pattern node per
                # persistence bin).
                sigma = TypedSignature(
                    kind="CANONICAL",
                    class_="BOUNDARY",
                    relation="CONTOUR_PROFILE",
                    quantized=(float(c_b), float(comp_b)),
                )
                return sigma, (("entity", anchor, 0, "+"),)

        return self._emit("Boundary", responses, anchors, _sig, lambda a: serialize_tuple([serialize_int(1), serialize_value(a)]))

    def _run_color(self, S_t: Any, obs: Any) -> List[Detection]:
        # [AUDIT-B 4.2 FIX — WRONG ACCESSOR NAME MADE B8 DEAD.] This probed
        # `get_anchor_opponent`, but Observation2D exposes
        # `get_opponent_channels(u)`. Probe the name that actually exists,
        # and accept the old one as an alias.
        _get_opp = (getattr(obs, "get_opponent_channels", None)
                    or getattr(obs, "get_anchor_opponent", None))
        if _get_opp is None:
            return self._run_color_legacy_rgb(S_t, obs)

        has_opp = False
        for u in S_t.entity_indices:
            opp = _get_opp(u)
            if opp is not None and len(opp) >= 5:
                has_opp = True
                break
        if not has_opp:
            return self._run_color_legacy_rgb(S_t, obs)

        responses: List[float] = []
        anchors: List[Any] = []
        color_bins: List[Tuple[int, int, int, int, int]] = []

        for u in S_t.entity_indices:
            opp = _get_opp(u)
            if opp is None or len(opp) < 5:
                continue
            lum_comp, rg_comp, by_comp, sat_norm, hue = opp
            lum_bin = _q_linear_safe(float(lum_comp), P.COLOR_LUM_BINS, -1.0, 1.0)
            rg_bin = _q_linear_safe(float(rg_comp), P.COLOR_RG_BINS, -1.0, 1.0)
            by_bin = _q_linear_safe(float(by_comp), P.COLOR_BY_BINS, -1.0, 1.0)
            sat_bin = _q_linear_safe(float(sat_norm), P.COLOR_SAT_BINS, 0.0, 1.0)
            hue_bin = _q_linear_safe(float(hue), P.COLOR_HUE_BINS, 0.0, 2 * math.pi)

            color_bins.append((lum_bin, rg_bin, by_bin, sat_bin, hue_bin))
            presence = max(self._presence_ema.get(u, 0.0), EPSILON)
            responses.append(presence * (1.0 + float(sat_norm)))
            anchors.append(u)

        def _sig(anchor: Any, idx: int) -> Tuple[TypedSignature, Tuple]:
            lum_b, rg_b, by_b, sat_b, hue_b = color_bins[idx]
            sigma = TypedSignature(
                kind="CANONICAL",
                class_="COLOR",
                relation="COLOR_REFLECTANCE",
                quantized=(float(lum_b), float(rg_b), float(by_b), float(sat_b), float(hue_b)),
            )
            rho = (("entity", anchor, 0, "+"),)
            return sigma, rho

        def _tb(anchor: Any) -> bytes:
            return serialize_tuple([serialize_int(2), serialize_value(anchor)])

        return self._emit("Color", responses, anchors, _sig, _tb)

    def _run_color_legacy_rgb(self, S_t: Any, obs: Any) -> List[Detection]:
        if not hasattr(obs, "get_anchor_rgb"):
            return []

        responses: List[float] = []
        anchors: List[Any] = []
        color_bins: List[Tuple[int, int, int]] = []
        legacy_bins = int(CONSTITUTIONAL_RANGES["COLOR_LEGACY"][0])
        bin_step = 1.0 / legacy_bins
        max_bin = legacy_bins - 1

        for u in S_t.entity_indices:
            rgb = obs.get_anchor_rgb(u)
            if rgb is None:
                continue
            r, g, b = rgb[0], rgb[1], rgb[2]
            if not (math.isfinite(r) and math.isfinite(g) and math.isfinite(b)):
                continue
            r_c = clamp(r, 0.0, 1.0)
            g_c = clamp(g, 0.0, 1.0)
            b_c = clamp(b, 0.0, 1.0)
            r_bin = _q_unsigned(r_c, bin_step, max_bins=max_bin)
            g_bin = _q_unsigned(g_c, bin_step, max_bins=max_bin)
            b_bin = _q_unsigned(b_c, bin_step, max_bins=max_bin)
            color_bins.append((r_bin, g_bin, b_bin))
            chroma = max(r_c, g_c, b_c) - min(r_c, g_c, b_c)
            presence = max(self._presence_ema.get(u, 0.0), EPSILON)
            responses.append(presence * (1.0 + chroma))
            anchors.append(u)

        def _sig(anchor: Any, idx: int) -> Tuple[TypedSignature, Tuple]:
            r_bin, g_bin, b_bin = color_bins[idx]
            sigma = TypedSignature(
                kind="CANONICAL",
                class_="COLOR_LEGACY",
                relation="COLOR_RAW_RGB",
                quantized=(float(r_bin), float(g_bin), float(b_bin)),
            )
            rho = (("entity", anchor, 0, "+"),)
            return sigma, rho

        def _tb(anchor: Any) -> bytes:
            return serialize_tuple([serialize_int(2), serialize_value(anchor)])

        return self._emit("Color", responses, anchors, _sig, _tb)

    # -------------------------------------------------------------------------
    # Motion lens — velocity & spatiotemporal energy of a single entity
    # -------------------------------------------------------------------------

    def _run_motion(self, S_t: Any, obs: Any) -> List[Detection]:
        if not hasattr(obs, "get_velocity") and not hasattr(obs, "get_motion_gradients"):
            return []

        responses: List[float] = []
        anchors: List[Any] = []
        motion_data: List[Tuple[Tuple[float, ...], int, int]] = []

        n_dirs = P.MOTION_DIRECTION_BINS
        angles = [float(k) * 2.0 * math.pi / float(n_dirs) for k in range(n_dirs)]

        energy_maps = None
        if hasattr(obs, "get_motion_gradients"):
            grads = obs.get_motion_gradients()
            if grads is not None and len(grads) == 3:
                Ix, Iy, It = grads
                energy_maps = motion_energy_maps(Ix, Iy, It, angles, P.MOTION_RF_SIGMA_PX)

        # Energy maps and differential motion are indexed in image pixels.
        # get_position is a view-normalized address, never a pixel fallback.
        # Without a pixel centroid, use velocity evidence (or skip motion)
        # rather than sampling near the image origin.
        get_pos = getattr(obs, "get_region_centroid_by_anchor", None)

        for u in S_t.entity_indices:
            energies = None
            if energy_maps is not None and get_pos is not None:
                p = get_pos(u)
                if p is not None and len(p) >= 2:
                    px, py = int(round(p[0])), int(round(p[1]))
                    H, W = energy_maps.shape[1], energy_maps.shape[2]
                    if 0 <= py < H and 0 <= px < W:
                        energies = np.array([float(energy_maps[k, py, px]) for k in range(n_dirs)], dtype=np.float64)

            if energies is None:
                if not hasattr(obs, "get_velocity"):
                    continue
                vel = obs.get_velocity(u)
                if vel is None:
                    continue
                vx, vy = float(vel[0]), float(vel[1])
                if not (math.isfinite(vx) and math.isfinite(vy)):
                    continue
                energies = np.array([max(0.0, vx * math.cos(a) + vy * math.sin(a)) for a in angles], dtype=np.float64)

            total_energy = float(np.sum(energies)) + 1e-6
            e_max = float(np.max(energies))
            e_mean = float(np.mean(energies))
            peakiness = (e_max - e_mean) / (e_max + e_mean + 1e-6)
            coherence_bin = q_linear(peakiness, P.MOTION_COHERENCE_BINS, 0.0, 1.0)

            c50 = P.MOTION_C50
            speed_ratio = total_energy / (total_energy + 1.0)
            speed_norm = (speed_ratio * speed_ratio) / (speed_ratio * speed_ratio + c50 * c50 + 1e-12)
            speed_bin = q_linear(clamp(speed_norm, 0.0, 1.0), P.MOTION_SPEED_BINS, 0.0, 1.0)

            dir_bins = [q_linear(e / total_energy, 8, 0.0, 1.0) for e in energies]

            i1, i2, e2 = _second_peak(energies, min_sep=2)
            e1 = float(energies[i1])
            transparency = (e2 / (e1 + 1e-6)) if e1 > 1e-6 else 0.0
            transparency_bin = q_linear(transparency, P.MOTION_TRANSPARENCY_BINS, 0.0, 1.0)
            dir1_bin = float(i1)
            dir2_bin = float(i2)

            # Robust closed-form differential motion: divergence, curl, shear
            # Compute from spatial gradients if available
            div_bin = 0.0
            curl_bin = 0.0
            shear_bin = 0.0
            if hasattr(obs, "get_motion_gradients"):
                grads = obs.get_motion_gradients()
                if grads is not None and len(grads) == 3:
                    Ix, Iy, It = grads
                    if get_pos is not None:
                        pos = get_pos(u)
                        if pos is not None and len(pos) >= 2:
                            px, py = int(round(pos[0])), int(round(pos[1]))
                            H, W = Ix.shape
                            if 0 <= py < H and 0 <= px < W:
                                # Spatial gradients at this point
                                # Use simple finite differences for spatial derivatives of velocity
                                # v = (-Ix*It/(Ix^2+Iy^2), -Iy*It/(Ix^2+Iy^2)) approx
                                denom = Ix[py, px]**2 + Iy[py, px]**2 + 1e-6
                                vx = -Ix[py, px] * It[py, px] / denom
                                vy = -Iy[py, px] * It[py, px] / denom
                                # Divergence: dvx/dx + dvy/dy
                                # Curl: dvy/dx - dvx/dy
                                # Shear: sqrt((dvx/dy + dvy/dx)^2 + (dvx/dx - dvy/dy)^2)
                                # Approximate using local Ix, Iy gradients
                                div = (Ix[py, px] * vx + Iy[py, px] * vy) / (math.sqrt(denom) + 1e-6)
                                curl = (Ix[py, px] * vy - Iy[py, px] * vx) / (math.sqrt(denom) + 1e-6)
                                shear = abs(div - curl) / 2.0
                                div_bin = q_linear(clamp(div, -1.0, 1.0), 8, -1.0, 1.0)
                                curl_bin = q_linear(clamp(curl, -1.0, 1.0), 8, -1.0, 1.0)
                                shear_bin = q_linear(clamp(shear, 0.0, 1.0), 8, 0.0, 1.0)

            quantized = (
                tuple(float(b) for b in dir_bins)
                + (float(coherence_bin), float(speed_bin))
                + (float(transparency_bin), float(dir1_bin), float(dir2_bin))
                + (float(div_bin), float(curl_bin), float(shear_bin))
            )

            motion_data.append((quantized, i1, i2))
            responses.append(max(self._presence_ema.get(u, 0.0), EPSILON))
            anchors.append(u)

        def _sig(anchor: Any, idx: int) -> Tuple[TypedSignature, Tuple]:
            q, i1, i2 = motion_data[idx]
            sigma = TypedSignature(
                kind="CANONICAL",
                class_="MOTION_POPULATION",
                relation="SPATIOTEMPORAL_ENERGY",
                quantized=q,
            )
            rho = (
                ("entity", anchor, 0, "+"),
                ("direction", int(i1), 0, "+"),
                ("direction", int(i2), 1, "+"),
            )
            return sigma, rho

        def _tb(anchor: Any) -> bytes:
            return serialize_tuple([serialize_int(3), serialize_value(anchor)])

        return self._emit("Motion", responses, anchors, _sig, _tb)

    # -------------------------------------------------------------------------
    # Size lens — spatial extent of a region
    # -------------------------------------------------------------------------

    def _run_size(self, S_t: Any, obs: Any) -> List[Detection]:
        """Size lens: radial scale + inertia aspect ratio (SIZE 2-channel,
        RADIUS_ASPECT)."""
        responses: List[float] = []
        anchors: List[Any] = []
        size_bins: List[Tuple[int, int]] = []
        step = 1.0 / 10.0

        for u in S_t.entity_indices:
            r_norm = getattr(obs, "get_radius", lambda idx: 0.1)(u)
            r_bin = _q_unsigned(clamp(float(r_norm), 0.0, 1.0), step, max_bins=9)

            elongation = getattr(obs, "get_elongation", lambda idx: 1.0)(u)
            e_bin = _q_unsigned(clamp(float(elongation), 0.0, 1.0), step, max_bins=9)

            size_bins.append((r_bin, e_bin))
            responses.append(max(self._presence_ema.get(u, 1.0), EPSILON))
            anchors.append(u)

        def _sig(anchor: Any, idx: int) -> Tuple[TypedSignature, Tuple]:
            r_b, e_b = size_bins[idx]
            sigma = TypedSignature(
                kind="CANONICAL",
                class_="SIZE",
                relation="RADIUS_ASPECT",
                quantized=(float(r_b), float(e_b)),
            )
            return sigma, (("entity", anchor, 0, "+"),)

        return self._emit("Size", responses, anchors, _sig, lambda a: serialize_tuple([serialize_int(4), serialize_value(a)]))

    def _run_texture(self, S_t: Any, obs: Any) -> List[Detection]:
        """Texture lens: gradient-magnitude roughness + magnitude-weighted
        directional isotropy (TEXTURE 2-channel), plus B4 TEXTURE_RICH
        (244-channel), with unified homeostatic accounting."""
        responses: List[float] = []
        anchors: List[Any] = []
        texture_bins: List[Tuple[int, int]] = []
        step = 1.0 / 10.0

        for u in S_t.entity_indices:
            roughness = getattr(obs, "get_roughness", lambda idx: 0.0)(u)
            isotropy = getattr(obs, "get_isotropy", lambda idx: 0.0)(u)

            rough_c = clamp(float(roughness), 0.0, 1.0)
            iso_c = clamp(float(isotropy), 0.0, 1.0)
            r_bin = _q_log_unsigned(rough_c, max_bins=10)
            i_bin = _q_unsigned(iso_c, step, max_bins=9)

            texture_bins.append((r_bin, i_bin))
            presence = max(self._presence_ema.get(u, 1.0), EPSILON)
            responses.append(presence * (1.0 + rough_c))
            anchors.append(u)

        def _sig(anchor: Any, idx: int) -> Tuple[TypedSignature, Tuple]:
            r_b, i_b = texture_bins[idx]
            sigma = TypedSignature(
                kind="CANONICAL",
                class_="TEXTURE",
                relation="SURFACE_ROUGHNESS_ISOTROPY",
                quantized=(float(r_b), float(i_b)),
            )
            return sigma, (("entity", anchor, 0, "+"),)

        # Defer channel EMA update to end of method
        base_emitted = self._emit(
            "Texture", responses, anchors, _sig,
            lambda a: serialize_tuple([serialize_int(5), serialize_value(a)]),
            update_homeostasis=False,
        )

        rich_emitted: List[Detection] = []
        get_b4 = getattr(obs, "get_b4_sig", None)
        state = self.state["Texture"]
        theta_eff = clamp(
            state["theta"] - P.ETA_CTX_CHANNEL * state["C"] + P.ETA_FAT_CHANNEL * state["F"],
            P.THETA_ACT_FLOOR,
            P.THETA_ACT_MAX,
        )

        if callable(get_b4):
            cap = _LENS_DETECTION_CAPS.get("Texture", _DEFAULT_LENS_DETECTION_CAP)
            M_prev = state["M_prev"]
            rich_candidates = []

            for u in S_t.entity_indices:
                b4_sig = get_b4(u)
                if b4_sig is not None and len(b4_sig.quantized) >= 4:
                    presence = max(self._presence_ema.get(u, 1.0), EPSILON)
                    contrast_energy = clamp(float(b4_sig.quantized[-1]), 0.0, 1.0)
                    resp = presence * contrast_energy

                    ident = serialize_tuple([serialize_int(55), serialize_value(u)])
                    prev_state = M_prev.get(ident, 0)
                    threshold = (
                        theta_eff if self._first_step
                        else theta_eff + P.DELTA_CHANNEL_HYST * (1 - 2 * prev_state)
                    )

                    if resp >= threshold:
                        z_val = clamp((resp - theta_eff) / (1.0 - theta_eff + EPSILON), 0.0, 1.0)
                        rich_candidates.append((z_val, ident, u, b4_sig))
                    else:
                        M_prev[ident] = 0

            rich_candidates.sort(key=lambda x: (-x[0], x[1]))
            selected_rich = rich_candidates[:cap]
            for z_val, ident, u, b4_sig in selected_rich:
                M_prev[ident] = 1
                rich_emitted.append(
                    Detection(
                        k="Texture",
                        source_layer=0,
                        scale=0,
                        sigma=b4_sig,
                        alpha=u,
                        z=z_val,
                        t=self.t,
                        rho=(("entity", u, 0, "+"),),
                    )
                )

        # Unified homeostatic update across all emissions for this channel
        total_emitted = len(base_emitted) + len(rich_emitted)
        n_active = max(len(anchors), 1)
        r_rate = total_emitted / float(n_active)
        state["r_bar"] = (1.0 - P.KAPPA_R_CHANNEL) * state["r_bar"] + P.KAPPA_R_CHANNEL * r_rate
        state["theta"] = clamp(
            state["theta"] + P.KAPPA_CHANNEL_THETA * (state["r_bar"] - P.RHO_CHANNEL_TARGET),
            P.THETA_CHANNEL_MIN,
            P.THETA_CHANNEL_MAX,
        )
        had_emission = 1.0 if total_emitted > 0 else 0.0
        state["C"] = (1.0 - P.KAPPA_C_CHANNEL) * state["C"] + P.KAPPA_C_CHANNEL * had_emission
        state["F"] = P.BETA_F_CHANNEL * state["F"] + (1.0 - P.BETA_F_CHANNEL) * had_emission

        return base_emitted + rich_emitted

    def _run_position(self, S_t: Any, obs: Any) -> List[Detection]:
        """Position lens: object-centric relative coordinates (POSITION
        3-channel, OBJECT_CENTRIC_REL).

        Observation2D supplies relative coordinates directly. For other
        modalities the relative frame is derived closed-form from the
        observation's absolute position envelope (translation/scale
        equivariant), keeping every modality on the same constitutional
        channel semantics.
        """
        responses: List[float] = []
        anchors: List[Any] = []
        pos_bins: List[Tuple[int, int, int]] = []

        rel_getter = getattr(obs, "get_relative_position", None)
        envelope = None
        # Absolute-position fallback ladder (env-agnostic): Observation3D
        # exposes get_position(u); symbolic envs expose a `positions`
        # mapping (dict keyed by atom index) or get_positions() — the
        # same conventions the pipeline's anchor_positions probe uses.
        # WITHOUT this rung, SymbolicObservation (no get_position) left
        # abs_pos empty: every entity quantized to the (0,0,0) bin, all
        # Position detections collapsed to ONE pattern node, and every
        # spatial binder (Topology/Degree/Contact) produced only
        # self-loops — forbidden by §3.4 — i.e. the spatial channel was
        # silently dead on every puzzle env (the D2 "DEGREE dark" report
        # understated it: all three were dark). Flag placeholders
        # ((-1,-1)) and non-finite entries are skipped, never quantized.
        abs_pos: Dict[Any, List[float]] = {}
        # Scene-frame points for the envelope: ALL positioned atoms the
        # observation exposes (entities, flags, effects), not just the
        # entity anchors. FIX (single-entity degeneracy): Level1Causal /
        # FertilityTest declare ONE entity (the agent); switches/lights
        # are flag/effect atoms. With an entity-only envelope the frame
        # collapsed to a point (wx = wy = 0 → dx ≡ 0 → bins frozen at
        # (7,7,7)): the agent's motion — pure translation of the only
        # entity — was invisible to the OBJECT_CENTRIC_REL frame, so no
        # Position signature ever changed and position-delta stayed at
        # 0 candidates on every single-entity env (measured: Level1 and
        # FertilityTest, 200 steps). The observation's absolute-position
        # envelope (docstring above) is the SCENE extent: derived from
        # every valid positioned atom, it is presence-independent (env
        # position maps carry stable slots for flags/effects), translation/
        # scale equivariant, and the agent moves RELATIVE to the scene.
        # Detection anchors remain entity-only (unchanged §2.2 semantics);
        # only the frame widens. Env-agnostic: same dict/sequence ladder,
        # same validity filter, canonical sorted iteration (§9).
        scene_pts: List[List[float]] = []

        def _parse_pos(pos: Any) -> Optional[List[float]]:
            if pos is None or not isinstance(pos, (list, tuple)) or len(pos) < 2:
                return None
            try:
                x = float(pos[0])
                y = float(pos[1])
            except (TypeError, ValueError):
                return None
            if not (math.isfinite(x) and math.isfinite(y)):
                return None
            if x == -1.0 and y == -1.0:
                return None  # symbolic flag placeholder — not a position
            z = 0.0
            if len(pos) >= 3:
                try:
                    z = float(pos[2])
                except (TypeError, ValueError):
                    z = 0.0
                if not math.isfinite(z):
                    z = 0.0
            return [x, y, z]

        if not callable(rel_getter):
            gp = getattr(obs, "get_position", None)
            positions_attr: Any = None
            if not callable(gp):
                positions_attr = getattr(obs, "positions", None)
                if positions_attr is None:
                    getter = getattr(obs, "get_positions", None)
                    if callable(getter):
                        positions_attr = getter()
            for u in S_t.entity_indices:
                pos: Any = None
                if callable(gp):
                    pos = gp(u)
                elif isinstance(positions_attr, dict):
                    pos = positions_attr.get(u)
                elif (
                    positions_attr is not None
                    and isinstance(u, int)
                    and not isinstance(u, bool)
                    and 0 <= u < len(positions_attr)
                ):
                    pos = positions_attr[u]
                parsed = _parse_pos(pos)
                if parsed is None:
                    continue
                abs_pos[u] = parsed
            if not callable(gp) and positions_attr is not None:
                if isinstance(positions_attr, dict):
                    scene_items = sorted(
                        positions_attr.items(), key=lambda kv: serialize_value(kv[0])
                    )
                    scene_vals = [v for _k, v in scene_items]
                else:
                    scene_vals = list(positions_attr)
                for pos in scene_vals:
                    parsed = _parse_pos(pos)
                    if parsed is not None:
                        scene_pts.append(parsed)
            env_src = scene_pts if scene_pts else list(abs_pos.values())
            if env_src:
                xs = [p[0] for p in env_src]
                ys = [p[1] for p in env_src]
                zs = [p[2] for p in env_src]
                envelope = (
                    sum(xs) / len(xs), max(xs) - min(xs),
                    sum(ys) / len(ys), max(ys) - min(ys),
                    sum(zs) / len(zs), max(zs) - min(zs),
                )

        for u in S_t.entity_indices:
            if callable(rel_getter):
                pos = rel_getter(u)
                dx, dy, dz = float(pos[0]), float(pos[1]), float(pos[2] if len(pos) > 2 else 0.0)
            elif envelope is not None and u in abs_pos:
                pos = abs_pos[u]
                cx, wx, cy, wy, cz, wz = envelope
                dx = (float(pos[0]) - cx) / (0.5 * max(wx, 1e-6))
                dy = (float(pos[1]) - cy) / (0.5 * max(wy, 1e-6))
                dz = (float(pos[2]) - cz) / (0.5 * max(wz, 1e-6))
            else:
                dx = dy = dz = 0.0
            x_bin = _q_signed(clamp(dx, -1.0, 1.0), max_bins=16)
            y_bin = _q_signed(clamp(dy, -1.0, 1.0), max_bins=16)
            z_bin = _q_signed(clamp(dz, -1.0, 1.0), max_bins=16)

            pos_bins.append((x_bin, y_bin, z_bin))
            responses.append(max(self._presence_ema.get(u, 1.0), EPSILON))
            anchors.append(u)

        def _sig(anchor: Any, idx: int) -> Tuple[TypedSignature, Tuple]:
            x_b, y_b, z_b = pos_bins[idx]
            sigma = TypedSignature(
                kind="CANONICAL",
                class_="POSITION",
                relation="OBJECT_CENTRIC_REL",
                quantized=(float(x_b), float(y_b), float(z_b)),
            )
            return sigma, (("entity", anchor, 0, "+"),)

        return self._emit("Position", responses, anchors, _sig, lambda a: serialize_tuple([serialize_int(6), serialize_value(a)]))

    # -------------------------------------------------------------------------
    # Orientation lens — major-axis angle of an elongated region
    # -------------------------------------------------------------------------

    def _run_orientation(self, S_t: Any, obs: Any) -> List[Detection]:
        """[v8/v9] Orientation lens with a Gabor branch and two
        geometry emission modes.

        [V9 Phase 4c / draft B.7] When the observation carries Gabor
        responses (Observation2D.from_image), a V1 filter-response
        path runs instead: local peaks over the (theta, sf) grid,
        divisively normalised, emitted as GABOR_RESPONSE signatures.
        The capability test is the DATA, not `hasattr`: every
        Observation2D now has `get_gabor_responses` (4b), and
        region-built instances return None for every anchor.

        [v8/v9] Geometry modes (unchanged):

          - Multi-segment: when obs.get_boundary_polyline(u) returns a
            polyline ≥ ORIENTATION_SEGMENT_MIN_POLYLINE, split into
            equal-arc segments and emit one ORIENTATION_SEGMENT node per
            segment (arity 4: theta, arc_len, chain_id, chain_pos).

          - Single-region fallback: when no polyline is available (short
            contours, 3D observations, symbolic modality), emit one
            ORIENTATION node on the region's inertia major axis.

        The two geometry modes never run on the same anchor in the
        same step.
        """
        gabor_probe = getattr(obs, "get_gabor_responses", None)
        if gabor_probe is not None:
            for _u in S_t.entity_indices:
                if gabor_probe(_u) is not None:
                    return self._run_orientation_gabor(S_t, obs)
        if self._has_orientation_channels is None:
            self._has_orientation_channels = hasattr(obs, "get_orientation")
        if not self._has_orientation_channels:
            return []

        has_polyline = hasattr(obs, "get_boundary_polyline")
        n_theta = P.ORIENTATION_N_THETA
        responses: List[float] = []
        anchors: List[Any] = []
        orient_bins: List[Tuple[float, ...]] = []
        emission_mode: List[str] = []
        closed_flags: List[bool] = []
        num_seg_per_region: List[int] = []

        # [Gradient gate] Only edge-energy patches may seed an orientation.
        # Flat (isotropic) patches pass ORIENTATION_ELONG_SKIP with a
        # near-arbitrary major axis; on a uniform grid they lattice into the
        # one-component ORIENTATION_BEND/JUNCTION blob that welds the whole
        # CurveComplex/Part tier into a single image-spanning component.
        # Modalities without a roughness accessor keep legacy behaviour
        # (no gradient information -> no gradient gate).
        get_roughness = getattr(obs, "get_roughness", None)

        for u in S_t.entity_indices:
            if (
                get_roughness is not None
                and float(get_roughness(u)) < P.ORIENTATION_GRADIENT_MIN
            ):
                continue
            polyline = obs.get_boundary_polyline(u) if has_polyline else None
            if polyline is not None:
                is_closed = (
                    len(polyline) >= 2 and polyline[0] == polyline[-1]
                )
                segments = _segment_polyline(polyline)
                if segments:
                    chain_id = (
                        obs.get_chain_id(u)
                        if hasattr(obs, "get_chain_id")
                        else None
                    )
                    chain_pos_start, chain_pos_end = (
                        obs.get_chain_bounds(u)
                        if hasattr(obs, "get_chain_bounds")
                        else (0.0, 1.0)
                    )
                    chain_id_bin = (
                        hash_64(serialize_int(chain_id)) & ((1 << 32) - 1)
                        if chain_id is not None
                        else hash_64(serialize_value(u)) & ((1 << 32) - 1)
                    )
                    n_seg = len(segments)
                    for seg_idx, (angle, arc_len, _mid) in enumerate(segments):
                        theta_bin = (
                            int(math.floor((angle % math.pi) * n_theta / math.pi))
                            % n_theta
                        )
                        arc_bin = _q_arc_len(arc_len)
                        local_frac = (seg_idx + 0.5) / n_seg
                        global_frac = (
                            chain_pos_start
                            + local_frac * (chain_pos_end - chain_pos_start)
                        )
                        chain_pos_bin = min(
                            P.ORIENTATION_SEGMENT_N_POS - 1,
                            int(
                                math.floor(
                                    global_frac * P.ORIENTATION_SEGMENT_N_POS
                                )
                            ),
                        )
                        resp = min(
                            1.0, arc_len / P.ORIENTATION_SEGMENT_ARC_MAX
                        )
                        if resp < EPSILON:
                            resp = EPSILON
                        responses.append(resp)
                        anchors.append((u, seg_idx))
                        orient_bins.append(
                            (
                                float(theta_bin),
                                float(arc_bin),
                                float(chain_id_bin),
                                float(chain_pos_bin),
                            )
                        )
                        emission_mode.append("segment")
                        closed_flags.append(is_closed)
                        num_seg_per_region.append(n_seg)
                    continue

            # Fallback: single-region emission (v8).
            pair = obs.get_orientation(u)
            if pair is None:
                continue
            angle = float(pair[0])
            elongation = float(pair[1])
            if not (math.isfinite(angle) and math.isfinite(elongation)):
                continue
            if elongation >= P.ORIENTATION_ELONG_SKIP:
                continue
            theta_bin = (
                int(math.floor((angle % math.pi) * n_theta / math.pi))
                % n_theta
            )
            elong_norm = clamp(1.0 - elongation, 0.0, 1.0)
            e_bin = int(math.floor(elong_norm * P.ORIENTATION_N_ELONG))
            if e_bin >= P.ORIENTATION_N_ELONG:
                e_bin = P.ORIENTATION_N_ELONG - 1
            responses.append(max(1.0 - elongation, EPSILON))
            anchors.append(u)
            orient_bins.append((float(theta_bin), float(e_bin)))
            emission_mode.append("single")
            closed_flags.append(False)
            num_seg_per_region.append(1)

        if not responses:
            return []

        def _sig(anchor: Any, idx: int) -> Tuple[TypedSignature, Tuple]:
            if emission_mode[idx] == "segment":
                u, seg_idx = anchor
                theta_b, arc_b, chain_b, pos_b = orient_bins[idx]
                sigma = TypedSignature(
                    kind="CANONICAL",
                    class_="ORIENTATION_SEGMENT",
                    relation="BOUNDARY_SEGMENT",
                    quantized=(theta_b, arc_b, chain_b, pos_b),
                )
                rho = (
                    ("entity", u, 0, "+"),
                    ("segment", seg_idx, 0, "+"),
                    ("num_segments", num_seg_per_region[idx], 0, "+"),
                    ("closed", 1 if closed_flags[idx] else 0, 0, "+"),
                )
                return sigma, rho
            else:
                theta_b, e_b = orient_bins[idx]
                anchor_bin = _anchor_class_bin(obs, anchor)
                sigma = TypedSignature(
                    kind="CANONICAL",
                    class_="ORIENTATION",
                    relation="MAJOR_AXIS",
                    quantized=(float(theta_b), float(e_b), float(anchor_bin)),
                )
                return sigma, (("entity", anchor, 0, "+"),)

        def _tb(anchor: Any) -> bytes:
            if isinstance(anchor, tuple):
                u, seg_idx = anchor
                return serialize_tuple(
                    [
                        serialize_int(8),
                        serialize_value(u),
                        serialize_int(seg_idx),
                    ]
                )
            return serialize_tuple([serialize_int(8), serialize_value(anchor)])

        # [Phase B B.14] Recurrent contour grouping. Applied ONCE, here, in a
        # single pass over the parallel `responses`/`anchors` lists — after
        # both the segment loop and the single-region fallback have
        # populated them, and before _emit's per-anchor max normalisation.
        # Keying off `anchors[idx]` (not a reconstruction from
        # emission_mode) is what lets one loop cover both modes: the list
        # already holds `u` for single-region and `(u, seg_idx)` for
        # segments, which is exactly the key space gain_by_anchor uses.
        gain_by_anchor = self._current_gain_by_anchor
        if gain_by_anchor:
            for idx in range(len(responses)):
                g = gain_by_anchor.get(anchors[idx], 0.0)
                if g > 0.0:
                    responses[idx] = responses[idx] * (1.0 + g)

        return self._emit("Orientation", responses, anchors, _sig, _tb)


    def _run_orientation_gabor(self, S_t: Any, obs: Any) -> List[Detection]:
        """[V9 Phase 4c / draft B.7] Gabor-response Orientation path.

        Runs only when the observation carries Gabor data. Peaks over the
        (theta, sf) grid are divided by the spatial+cross-channel pool
        (Heeger-style divisive normalisation, draft B.7 form), binned, and
        emitted with `serialize_value`-stable anchors.

        Declared adaptation: the live `_emit` takes five parameters -- no
        `neighbor_fn` / `anchor_region_key` -- so the draft's
        neighbor-based suppression is not applied here (register 33).
        """
        responses: List[float] = []
        anchors: List[Any] = []
        bins: List[Tuple[float, ...]] = []
        # [B.45] Raw magnitudes and the within-anchor denominator per
        # candidate; the across-anchor pool needs both after the loop.
        raws: List[float] = []
        pools: List[float] = []

        for u in S_t.entity_indices:
            g = obs.get_gabor_responses(u)
            if g is None:
                continue
            complex_ = g["complex"]
            pooled = g["complex_pooled"]
            simple = g["simple"]
            nt, nsf = complex_.shape

            peaks = _find_local_peaks(complex_, P.ORIENTATION_MIN_RESPONSE)
            for (it, isf, v) in peaks:
                spatial_pool = float(pooled[it, isf])
                cross_channel_pool = float(
                    pooled[max(0, it - 1):it + 2,
                           max(0, isf - 1):isf + 2].mean())
                total_pool = (P.DIVISIVE_W_SPATIAL * spatial_pool
                              + P.DIVISIVE_W_CROSS * cross_channel_pool)
                v_norm = v / (P.DIVISIVE_SIGMA + total_pool)

                contrast_bin = _log_contrast_bin(
                    v_norm, P.ORIENTATION_N_CONTRAST)
                elong = (obs.get_elongation(u)
                         if hasattr(obs, "get_elongation") else 1.0)
                elong_bin = _q_unsigned(
                    clamp(1.0 - elong, 0.0, 1.0),
                    1.0 / P.ORIENTATION_N_ELONG,
                    max_bins=P.ORIENTATION_N_ELONG - 1)

                ev = float(simple[it, isf, 0])
                od = float(simple[it, isf, 1])
                phase_rad = math.atan2(od, ev)

                # [B3b] Polarity asymmetry: complex path
                on_l = float(g["complex_on"][it, isf]) if "complex_on" in g else max(0.0, ev)
                off_l = float(g["complex_off"][it, isf]) if "complex_off" in g else max(0.0, -ev)
                asym = (on_l - off_l) / (on_l + off_l + 1e-6)
                asym_bin = float(min(7, max(0, int(math.floor(((clamp(asym, -1.0, 1.0) + 1.0) / 2.0) * 8.0)))))

                responses.append(v_norm)
                raws.append(v)
                pools.append(total_pool)
                anchors.append((u, it, isf, "complex"))
                bins.append((float(it), float(isf), 0.0,
                             float(contrast_bin), float(elong_bin), 1.0,
                             asym_bin))

                # Simple cell emission
                amp = math.hypot(ev, od) + 1e-9
                if amp / P.SIMPLE_SNR_FLOOR < P.SIMPLE_PHASE_SNR_MIN:
                    continue
                # [B10] Quadrature phase circularly quantized
                phase_bin = float(int((((phase_rad + math.pi) / (2.0 * math.pi)) * P.ORIENTATION_PHASE_BINS)) % P.ORIENTATION_PHASE_BINS)

                # [B10] Contrast via Naka-Rushton at fixed semisaturation C50
                c50 = P.ORIENTATION_C50
                simple_c = (amp * amp) / (amp * amp + c50 * c50 + 1e-12)
                simple_contrast_bin = float(min(7, max(0, int(math.floor(simple_c * float(P.ORIENTATION_CONTRAST_BINS))))))

                # [B3b] Simple path asymmetry is cos(phase)
                simple_asym = math.cos(phase_rad)
                simple_asym_bin = float(min(7, max(0, int(math.floor(((clamp(simple_asym, -1.0, 1.0) + 1.0) / 2.0) * 8.0)))))

                simple_v_norm = amp / (P.DIVISIVE_SIGMA + total_pool)
                responses.append(simple_v_norm * P.SIMPLE_CELL_GAIN)
                raws.append(amp * P.SIMPLE_CELL_GAIN)
                pools.append(total_pool)
                anchors.append((u, it, isf, "simple"))
                bins.append((float(it), float(isf), float(phase_bin),
                             float(simple_contrast_bin),
                             float(elong_bin), 0.0,
                             simple_asym_bin))

        if not responses:
            return []

        # [B.45] Anisotropic divisive normalization (Heeger 1992): the
        # across-anchor pool. Candidate i's pool is weighted by a
        # circularly-wrapped Gaussian on orientation difference, a Gaussian
        # on the log2 spatial-frequency difference and an isotropic
        # Gaussian on spatial distance; the last is in PIXELS (the px
        # centroid accessor, not the normalized-address get_position, whose
        # -1..1 range would make a px radius cover the whole image).
        # Declared integration (register section 40): the term ADDS to the
        # within-anchor denominator above rather than dividing a second
        # time, so the stage stays the canonical x / (sigma + pool).
        get_pos = (getattr(obs, "get_region_centroid_by_anchor", None)
                   or getattr(obs, "get_position", None))
        if responses and get_pos is not None:
            n = len(responses)
            thetas = [0.0] * n
            sfs = [0.0] * n
            positions: List[Optional[Tuple[float, float]]] = [None] * n
            for idx, a in enumerate(anchors):
                u = a[0] if isinstance(a, tuple) else a
                # [B.45 declared fix] Both axes are indexed by the GABOR
                # bank's grid, so the angular step is pi / GABOR_N_THETA.
                # The draft used the geometry-branch constants
                # (ORIENTATION_N_THETA = 24 vs the bank's 12) which halved
                # every orientation difference -- measured in the read-back
                # as a 90-degree pair scoring 45 degrees and suppressing.
                thetas[idx] = bins[idx][0] * math.pi / P.GABOR_N_THETA
                sfs[idx] = bins[idx][1] / max(1, P.GABOR_N_SF - 1)
                p = get_pos(u)
                if p is not None and len(p) >= 2:
                    positions[idx] = (float(p[0]), float(p[1]))

            r2 = (P.DIVISIVE_ANISO_POOL_RADIUS
                  * P.DIVISIVE_ANISO_POOL_RADIUS)
            pooled_across = [0.0] * n
            for i in range(n):
                pi = positions[i]
                if pi is None:
                    continue
                p_sum = 0.0
                w_total = 0.0
                for j in range(n):
                    if i == j:
                        continue
                    pj = positions[j]
                    if pj is None:
                        continue
                    d2 = ((pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2)
                    if d2 > 4.0 * r2:
                        continue
                    w_spatial = math.exp(-d2 / (r2 + EPSILON))
                    d_theta = abs(thetas[i] - thetas[j])
                    d_theta = min(d_theta, math.pi - d_theta)
                    w_theta = math.exp(
                        -0.5 * ((d_theta / P.DIVISIVE_SIGMA_THETA) ** 2))
                    d_sf = math.log2((sfs[i] + EPSILON) / (sfs[j] + EPSILON))
                    w_sf = math.exp(-((d_sf / P.DIVISIVE_SIGMA_SF) ** 2))
                    w = w_theta * w_sf * w_spatial
                    p_sum += w * raws[j]
                    w_total += w
                if getattr(P, "LENS_POOL_NORMALIZE", False):
                    pooled_across[i] = p_sum / (w_total + getattr(P, "LENS_POOL_PRIOR_MASS", 1.0))
                else:
                    pooled_across[i] = p_sum

            for i in range(n):
                final = raws[i] / (P.DIVISIVE_SIGMA
                                   + pools[i] + pooled_across[i])
                responses[i] = final
                b = list(bins[i])
                if b[5] == 1.0:  # complex cell
                    b[3] = float(
                        _log_contrast_bin(final, P.ORIENTATION_N_CONTRAST))
                bins[i] = tuple(b)


        def _sig(anchor: Any, idx: int) -> Tuple[TypedSignature, Tuple]:
            b = bins[idx]
            sigma = TypedSignature(
                kind="CANONICAL", class_="ORIENTATION_GABOR",
                relation="GABOR_RESPONSE",
                quantized=(float(b[0]), float(b[1]), float(b[2]),
                           float(b[3]), float(b[4]), float(b[5]),
                           float(b[6])))
            u, it, isf, kind = anchor
            return sigma, (("entity", u, 0, "+"),
                           ("theta", it, 0, "+"),
                           ("sf", isf, 0, "+"),
                           ("cell", kind, 0, "+"))

        def _tb(anchor: Any) -> bytes:
            u, it, isf, kind = anchor
            return serialize_tuple([serialize_int(8), serialize_value(u),
                                    serialize_int(it), serialize_int(isf),
                                    serialize_bytes(kind.encode())])

        # [B.45] This branch is already divisively normalized: keeping
        # max-norm here would map the strongest response to 1.0 and hide
        # cross-candidate suppression (the draft's own verification).
        # Defer channel EMA update to end of method
        base_emitted = self._emit("Orientation", responses, anchors, _sig, _tb,
                                  normalize_by_max=False,
                                  update_homeostasis=False)

        rich_emitted: List[Detection] = []
        get_b1 = getattr(obs, "get_b1_sig", None)
        state = self.state["Orientation"]
        theta_eff = clamp(
            state["theta"] - P.ETA_CTX_CHANNEL * state["C"] + P.ETA_FAT_CHANNEL * state["F"],
            P.THETA_ACT_FLOOR,
            P.THETA_ACT_MAX,
        )

        if callable(get_b1):
            cap = _LENS_DETECTION_CAPS.get("Orientation", _DEFAULT_LENS_DETECTION_CAP)
            n_marginals = getattr(P, "ORIENTATION_RICH_MARGINAL_BINS", 60)
            M_prev = state["M_prev"]
            rich_candidates = []

            for u in S_t.entity_indices:
                b1_sig = get_b1(u)
                if b1_sig is not None and len(b1_sig.quantized) >= n_marginals:
                    presence = max(self._presence_ema.get(u, 1.0), EPSILON)
                    valid_margs = [float(x) for x in b1_sig.quantized[:n_marginals] if math.isfinite(x)]
                    peak_energy = clamp(max(valid_margs), 0.0, 1.0) if valid_margs else 0.0
                    resp = presence * peak_energy

                    ident = serialize_tuple([serialize_int(88), serialize_value(u)])
                    prev_state = M_prev.get(ident, 0)
                    threshold = (
                        theta_eff if self._first_step
                        else theta_eff + P.DELTA_CHANNEL_HYST * (1 - 2 * prev_state)
                    )

                    if resp >= threshold:
                        z_val = clamp((resp - theta_eff) / (1.0 - theta_eff + EPSILON), 0.0, 1.0)
                        rich_candidates.append((z_val, ident, u, b1_sig))
                    else:
                        M_prev[ident] = 0

            rich_candidates.sort(key=lambda x: (-x[0], x[1]))
            selected_rich = rich_candidates[:cap]
            for z_val, ident, u, b1_sig in selected_rich:
                M_prev[ident] = 1
                rich_emitted.append(
                    Detection(
                        k="Orientation",
                        source_layer=0,
                        scale=0,
                        sigma=b1_sig,
                        alpha=u,
                        z=z_val,
                        t=self.t,
                        rho=(("entity", u, 0, "+"),),
                    )
                )

        # Clean up stale identities whose anchors are absent from the current frame
        active_idents = {
            serialize_tuple([serialize_int(8), serialize_value(u)]) for u in S_t.entity_indices
        } | {
            serialize_tuple([serialize_int(88), serialize_value(u)]) for u in S_t.entity_indices
        }
        state["M_prev"] = {k: v for k, v in state["M_prev"].items() if k in active_idents}

        # Unified homeostatic update across all emissions for this channel
        total_emitted = len(base_emitted) + len(rich_emitted)
        n_active = max(len(anchors), 1)
        r_rate = total_emitted / float(n_active)
        state["r_bar"] = (1.0 - P.KAPPA_R_CHANNEL) * state["r_bar"] + P.KAPPA_R_CHANNEL * r_rate
        state["theta"] = clamp(
            state["theta"] + P.KAPPA_CHANNEL_THETA * (state["r_bar"] - P.RHO_CHANNEL_TARGET),
            P.THETA_CHANNEL_MIN,
            P.THETA_CHANNEL_MAX,
        )
        had_emission = 1.0 if total_emitted > 0 else 0.0
        state["C"] = (1.0 - P.KAPPA_C_CHANNEL) * state["C"] + P.KAPPA_C_CHANNEL * had_emission
        state["F"] = P.BETA_F_CHANNEL * state["F"] + (1.0 - P.BETA_F_CHANNEL) * had_emission

        return base_emitted + rich_emitted

    # -------------------------------------------------------------------------
    # OrientationPol lens -- polarized major-axis orientation (option (b)).
    #
    # Companion to Orientation: the same theta/elongation/anchor-class
    # slots plus the 3-class luminance-polarity slot. The ORIENTATION
    # class is NOT modified and its nodes are untouched; this lens exists
    # so polarity-selective cells have their own pattern nodes instead of
    # collapsing onto the polarity-invariant population (rho is not
    # hashed, so a rho-carried polarity label would be destroyed).
    #
    # Data gate: polarity comes from the observation's Gabor responses.
    # Region observations carry none, so every anchor is skipped, _emit is
    # never called, and the region path is byte-identical by construction.
    # -------------------------------------------------------------------------

    def _run_orientation_pol(self, S_t: Any, obs: Any) -> List[Detection]:
        """Polarized single-region orientation.

        Emits class "ORIENTATION_POL", relation "POLARIZED_MAJOR_AXIS",
        arity 4: (theta bin, elongation bin, coarse position class,
        polarity bin). Mirrors the single-region eligibility of
        `_run_orientation` exactly -- the gradient gate, the
        `get_orientation` capability and ORIENTATION_ELONG_SKIP. Anchors
        whose polarity cannot be derived are SKIPPED, never defaulted to
        balanced (1): a fabricated polarity is worse than a missing one.
        """
        if not hasattr(obs, "get_orientation"):
            return []
        get_roughness = getattr(obs, "get_roughness", None)
        n_theta = P.ORIENTATION_N_THETA
        n_elong = P.ORIENTATION_N_ELONG

        responses: List[float] = []
        anchors: List[Any] = []
        pol_bins: List[Tuple[int, int, int, int]] = []

        for u in S_t.entity_indices:
            # [Gradient gate] Same gate, same anchors as _run_orientation.
            if (
                get_roughness is not None
                and float(get_roughness(u)) < P.ORIENTATION_GRADIENT_MIN
            ):
                continue
            pair = obs.get_orientation(u)
            if pair is None:
                continue
            angle = float(pair[0])
            elongation = float(pair[1])
            if not (math.isfinite(angle) and math.isfinite(elongation)):
                continue
            if elongation >= P.ORIENTATION_ELONG_SKIP:
                continue
            polarity_bin = compute_polarity_bin(obs, u)
            if polarity_bin is None:
                continue  # no fabrication
            theta_bin = (
                int(math.floor((angle % math.pi) * n_theta / math.pi))
                % n_theta
            )
            elong_norm = clamp(1.0 - elongation, 0.0, 1.0)
            e_bin = int(math.floor(elong_norm * n_elong))
            if e_bin >= n_elong:
                e_bin = n_elong - 1
            anchor_bin = int(_anchor_class_bin(obs, u))
            responses.append(max(1.0 - elongation, EPSILON))
            anchors.append(u)
            pol_bins.append((theta_bin, e_bin, anchor_bin,
                             int(polarity_bin)))

        if not responses:
            return []

        def _sig(anchor: Any, idx: int) -> Tuple[TypedSignature, Tuple]:
            tb, eb, ab, pb = pol_bins[idx]
            sigma = TypedSignature(
                kind="CANONICAL",
                class_="ORIENTATION_POL",
                relation="POLARIZED_MAJOR_AXIS",
                quantized=(float(tb), float(eb), float(ab), float(pb)),
            )
            return sigma, (("entity", anchor, 0, "+"),)

        # Tiebreaker integer 9 is unused in this module (Orientation's
        # geometry and Gabor paths both use 8); M_prev is per-lens-name,
        # so the value only has to be stable within this lens.
        return self._emit(
            "OrientationPol", responses, anchors, _sig,
            lambda a: serialize_tuple([serialize_int(9), serialize_value(a)]),
        )


    def _run_onset(self, S_t: Any) -> List[Detection]:
        responses: List[float] = []
        anchors: List[Any] = []
        onset_bins: List[int] = []

        # Check entities AND active flags/effects for appearance onsets
        atoms_to_check = set(S_t.entity_indices)
        if hasattr(S_t, "raw_indices"):
            for i in S_t.raw_indices:
                if (S_t.X[i][5] == 1.0 or S_t.X[i][6] == 1.0) and S_t.X[i][0] > 0.5:
                    atoms_to_check.add(i)

        for u in sorted(atoms_to_check, key=serialize_value):
            onset_step = self._onset_step.get(u)
            if onset_step is None:
                continue
            age = self.t - onset_step
            if age < 0 or age > _ONSET_WINDOW:
                continue
            decay = math.exp(-age * 3.0 / max(_ONSET_WINDOW, 1))
            responses.append(decay)
            anchors.append(u)
            onset_bins.append(self._onset_signature_bin.get(u, 0))

        def _sig(anchor: Any, idx: int) -> Tuple[TypedSignature, Tuple]:
            sigma = TypedSignature(
                kind="CANONICAL",
                class_="ONSET",
                relation="APPEARANCE_TRANSITION",
                quantized=(float(onset_bins[idx]),),
            )
            rho = (("entity", anchor, 0, "+"),)
            return sigma, rho

        def _tb(anchor: Any) -> bytes:
            return serialize_tuple([serialize_int(7), serialize_value(anchor)])

        return self._emit("Onset", responses, anchors, _sig, _tb)

    def _run_effect_presence(
        self,
        obs: Any,
        concept_effects: Optional[Set[int]],
    ) -> List[Detection]:
        """[Tier 3 persistence] Emit one EffectPresent detection per
        effect atom that is (a) present in the current observation,
        AND (b) either bound to a concept the reasoning pass is
        currently pursuing, OR risen within W_SEQ steps.

        Bypasses _emit: homeostatic suppression contradicts the
        purpose of the lens. Presence is a fact, not a graded signal.

        Activation gate (Commit B inertness): when concept_effects is
        None (the pipeline hasn't been wired yet), return [] without
        touching any state. When the pipeline passes a set — even an
        empty set — the lens activates.

        Presence filtering matches Pipeline._derive_label_signals:
        an effect counts as present only when it appears in the
        observation's effect indices AND (where presence is exposed)
        its presence slot is truthy. Observation2D exposes no presence
        list; the filter is a no-op there (and 2D observations carry
        no effect indices anyway)."""
        if concept_effects is None:
            return []

        getter = getattr(obs, "get_effect_indices", None)
        if callable(getter):
            raw_effects = list(getter())
        else:
            raw_effects = list(getattr(obs, "effect_indices", None) or [])

        presence = getattr(obs, "presence", None)

        present_effects: Set[int] = set()
        for eid in raw_effects:
            if presence is None:
                present_effects.add(eid)
                continue
            try:
                if presence[eid]:
                    present_effects.add(eid)
            except (IndexError, TypeError):
                continue

        # Rise detection on the presence-filtered set.
        newly_risen = present_effects - self._prev_present_effects
        for eid in newly_risen:
            self._recently_risen_effects[eid] = self.t + P.W_SEQ
        self._prev_present_effects = present_effects

        # Prune expired.
        for eid in list(self._recently_risen_effects.keys()):
            if self._recently_risen_effects[eid] < self.t:
                del self._recently_risen_effects[eid]

        attention = concept_effects | set(self._recently_risen_effects.keys())
        if not attention:
            return []

        out: List[Detection] = []
        for eid in sorted(present_effects, key=serialize_value):
            if eid not in attention:
                continue
            sigma = TypedSignature.opaque(
                "EFFECT_PRESENT", hash_64(serialize_int(eid))
            )
            out.append(
                Detection(
                    k="EffectPresent",
                    source_layer=0,
                    scale=0,
                    sigma=sigma,
                    alpha=eid,
                    z=1.0,
                    t=self.t,
                    rho=(("entity", eid, 0, "+"),),
                    substrate_hash=sigma.hash_value,
                )
            )
        self._effect_present_emitted_total += len(out)
        return out


# =============================================================================
# [B.30] POLARITY LENS (merged from polarity_lens.py)
# =============================================================================

class PolarityLens:
    """Emits one Polarity detection per anchor."""

    def __init__(self):
        self._emitted_total = 0

    def step(self, t, concept_graph, sig_provider, anchor_positions,
             anchor_velocities, current_detections, extraction_scale=0.0,
             envelope_mask=None, observation=None):
        if observation is None:
            return []
        if not hasattr(observation, "get_gabor_responses"):
            return []
        if not anchor_positions:
            return []

        emitted = []
        for anchor in sorted(anchor_positions.keys(),
                             key=lambda a: (a if isinstance(a, int)
                                            else str(a))):
            g = observation.get_gabor_responses(anchor)
            if g is None:
                continue
            polarity_class = self._anchor_polarity(g)
            sigma = TypedSignature(
                kind="CANONICAL",
                class_="POLARITY",
                relation="LUMINANCE_POLARITY",
                quantized=(float(polarity_class),),
            )
            emitted.append(Detection(
                k="Polarity",
                source_layer=1,
                scale=0,
                sigma=sigma,
                alpha=anchor,
                z=1.0,
                t=t,
                rho=(("entity", anchor, 0, "+"),)))

        self._emitted_total += len(emitted)
        return emitted

    @staticmethod
    def _anchor_polarity(g):
        """[TIER-1 Fix A] Aggregate phase-0 Gabor responses to 3 bins.

        Returns 0 (OFF-dominant), 1 (balanced), 2 (ON-dominant).
        """
        simple = g.get("simple")
        if simple is None or simple.size == 0:
            return 1
        # [V9.3 CORRECTION] The array is (n_theta, n_sf, n_phase) with
        # n_phase == 2 (phases 0 and pi/2), NOT (n_theta, n_sf, 2) as
        # the original fix text claimed. Both happen to index the same
        # way -- `simple[:, :, 0]` is the even (cosine) channel -- but
        # the shape claim was wrong and would mislead anyone writing a
        # guard. See B.6 (`r["gabor_simple"] = simple[:, :, :, cy, cx]`
        # over a 5-D `simple`) and B.3 (`phases = (0.0, pi/2)`).
        even = np.asarray(simple[:, :, 0], dtype=np.float32).ravel()
        weights = np.abs(even)
        total_w = float(weights.sum())
        if total_w < 1e-9:
            return 1
        mean_signed = float((even * weights).sum() / total_w)
        rms = float(np.sqrt((even ** 2).mean())) + 1e-9
        normalized = mean_signed / rms
        if normalized > P.POLARITY_ON_THRESHOLD:
            return 2
        if normalized < -P.POLARITY_ON_THRESHOLD:
            return 0
        return 1



def compute_polarity_bin(observation, u):
    """[Option (b) directive] Polarity class for one anchor, or None.

    The classification itself is `PolarityLens._anchor_polarity` -- this
    wrapper exists so a second lens reuses it verbatim (one implementation,
    two entry shapes: per-anchor accessor here; the lens's own
    per-detection path there). Returns None when the anchor carries no
    Gabor response, which is the "not derivable -> skip" case: callers
    must NOT default to balanced (1).
    """
    getter = getattr(observation, "get_gabor_responses", None)
    if getter is None:
        return None
    g = getter(u)
    if g is None:
        return None
    return PolarityLens._anchor_polarity(g)
