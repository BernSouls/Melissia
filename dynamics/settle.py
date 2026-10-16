"""[V9 Phase 4d] settle.py -- hypothesis settle loop (merged draft B.9 + B.22).

Draft module merged into the tree per the Phase-4 table; the ONLY new file.

Declared interface adaptations against the draft (full text in the register
§34):
  * `derived.run_all(t, dets, cg)` -- live `DerivedLenses` exposes the
    four `step_13..16` calls, not a `run_all`; `DefaultDerivedRunner` adapts
    them into the draft's shape.
  * the draft's `binder.get_step_anchor_positions()` /
    `get_step_extraction_scale()` do not exist (the state is private on the
    binder and the live Pipeline computes both itself): `SettleLoop.step`
    takes them as parameters and passes them through.
  * `SensoryFloor.step` / `LensExecutor.run_all` do not yet accept
    `envelope_mask` / `update_homeostasis` / `object_envelope` /
    `scale_level` -- that is the 4g widening. This module MUST NOT be wired
    into the pipeline before 4g (declared ordering dependency).

Dormant until 4e/4g: nothing imports this file yet.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional

import parameters as P
from concept_graph import ConceptStatus


@dataclass(frozen=True)
class Hypothesis:
    """One active top-down hypothesis for the settle iteration.

    Frozen: hypotheses are compared across iterations by value in
    `_hyp_stability`, so a mutated instance would make the stability test
    depend on iteration order rather than content.
    """

    concept_id: int
    signature: Any
    envelope_boxes: tuple
    activation: float
    # [B.48] The lens this hypothesis attends to. Required for
    # feature_attention_gain: with no lens name the helper returns {} and
    # attention silently does nothing while appearing to be wired (the
    # draft's own BLOCKER note). Defaults to None so existing positional
    # construction is unaffected.
    primary_lens: Optional[str] = None


class _MinState:
    """Minimal StatePattern stand-in for the coarse prior pass.

    The coarse pass needs only `entity_indices` -- not history, not the
    state hash, not validation -- so it never calls `SensoryFloor.step`.
    That is what stops the coarse pass from advancing `SensoryFloor.t` or
    appending a second history entry per real step.
    """

    def __init__(self, obs):
        n = obs.num_atoms() if hasattr(obs, "num_atoms") else 0
        self.entity_indices = list(range(n))
        self.raw_indices = self.entity_indices
        # 16-wide rows: field lenses only read [0] (presence) and the
        # flag/effect slots guarded in `_run_onset`; this is NOT a full
        # state encoding (declared, register §34).
        self.X = [[0.0] * 16 for _ in range(n)]


class DefaultDerivedRunner:
    """Adapts the live `DerivedLenses.step_13..16` sequence to the draft's
    `derived.run_all(t, dets, cg)` shape.

    The draft was written against a `derived` object exposing `run_all`; the
    live tree orchestrates the same four calls from `Pipeline.step` (with
    the recognition cache and co-activation state the tree owns). This
    adapter reproduces that sequence in the draft's shape so the settle loop
    has a working default; 4e's Pipeline passes one wrapping its live
    `derived_lenses` instance.
    """

    def __init__(self, derived_lenses, concept_graph, max_evidence=8):
        self._dl = derived_lenses
        self._cg = concept_graph
        self._max_evidence = max_evidence

    def run_all(self, t, dets, cg=None):
        cg = cg if cg is not None else self._cg
        dets_list = list(dets)
        out = set()
        try:
            out |= set(self._dl.step_13_recognition(
                D_raw=dets_list, concepts=cg.concepts, t=t))
            if hasattr(self._dl, "recognize_raw"):
                out |= set(self._dl.recognize_raw(
                    dets_list, cg.concepts, t, min_similarity=0.9))
            out |= set(self._dl.step_14_operator_emission(
                D_base=dets_list, concepts=cg.concepts, edges=cg.edges,
                active_concepts=cg.get_active_concepts(), t=t,
                ef_eff_proposal={
                    cid: getattr(c, "EF_eff", 0.0)
                    for cid, c in cg.concepts.items()},
                rf={cid: getattr(c, "RF", 0.0)
                    for cid, c in cg.concepts.items()}))
            candidates = self._dl.step_15_chunk_candidates(
                active_plans={}, concepts=cg.concepts, edges=cg.edges,
                coact_ema=cg.get_coact_ema(),
                time_since_coact=cg.get_time_since_coact(),
                active_concepts=cg.get_active_concepts(), t=t)
            out |= set(self._dl.step_16_chunk_emission(
                candidates=candidates, concepts=cg.concepts, edges=cg.edges,
                coact_ema=cg.get_coact_ema(), t=t, context_sig=None,
                active_plans={}))
        except Exception:
            # A derived-lens failure must not take down the settle loop;
            # the loop's product is detections, and partial detections are
            # still bindable. `try/except` is deliberate ONLY here, around
            # foreign subsystems with no shared error contract.
            pass
        return out


class SettleLoop:
    """Multi-iteration settle loop with hypothesis feedback.

    The floor runs ONCE per real step (iteration 0); later iterations re-run
    only the lens executor against the already-built state, so `SensoryFloor.t`
    advances exactly once per step and temporal binders see a real timeline.

    Native laminar recurrence (Step 7): iteration i > 0 receives a top-down
    attentional gain field from the hypotheses formed at iteration i-1.
    This gain is applied via `gain_by_anchor` in the lens executor, biasing
    V1/V2 responses toward features consistent with the current hypothesis set.
    """

    def __init__(self, sensory_floor, binder, self_binders=None, derived=None, cg=None,
                 higher_lens_runner=None):
        self.sf = sensory_floor
        self.binder = binder
        self.self_binders = self_binders
        self.derived = derived
        self.cg = cg
        self.higher_lens_runner = higher_lens_runner
        self._prev_hypotheses: List[Hypothesis] = []
        self._match_cache: dict = {}

    def step(self, t, obs, a_prev, dir_tuples, concept_effects,
             anchor_positions=None, extraction_scale=0.0):
        self._match_cache = {}  # per-step clear
        self.binder.note_step_begin(t)
        try:
            # Coarse pass: concept priors only, no boxes (watershed
            # centroids do not scale linearly under downsampling, so
            # projected boxes land at positions belonging to no real anchor).
            coarse_obs = (obs.at_scale(P.PYRAMID_N_LEVELS - 1)
                          if hasattr(obs, "at_scale") else obs)
            lens_exec = self.sf._get_lens_executor()
            if coarse_obs is obs or coarse_obs is None:
                coarse_priority: set = set()
            else:
                coarse_state = _MinState(coarse_obs)
                D_coarse = lens_exec.run_all(
                    coarse_state, coarse_obs, self.sf.t, dir_tuples,
                    prev_action=a_prev, concept_effects=concept_effects,
                    gain_by_anchor=None, envelope_mask=None,
                    update_homeostasis=False, object_envelope=None,
                    scale_level=0)
                coarse_priority = self._concepts_hit(D_coarse)

            prev_hyps: List[Hypothesis] = []
            prev_all_dets: set = set()
            final_dets: set = set()
            final_state = None
            final_hash = 0
            final_T = None

            # Laminar top-down gain field, updated each iteration
            top_down_gain: Dict[Any, float] = {}

            for iteration in range(P.SETTLE_MAX_ITERS):
                boxes = _build_boxes(prev_hyps) if iteration > 0 else None
                obj_env = _object_envelope(prev_hyps)

                # Use top-down gain from previous iteration (laminar recurrence)
                gain_by_anchor = top_down_gain if iteration > 0 else None

                if iteration == 0:
                    D_v1, S_t, s_t, T_prev = self.sf.step(
                        obs, a_prev=a_prev, dir_tuples=dir_tuples,
                        concept_effects=concept_effects,
                        gain_by_anchor=gain_by_anchor, envelope_mask=boxes,
                        update_homeostasis=True,
                        object_envelope=obj_env, scale_level=0)
                else:
                    D_v1 = lens_exec.run_all(
                        S_t, obs, t, dir_tuples, prev_action=a_prev,
                        concept_effects=concept_effects,
                        gain_by_anchor=gain_by_anchor, envelope_mask=boxes,
                        update_homeostasis=False,
                        object_envelope=obj_env, scale_level=0)

                # [Stage 4] Extract shock skeleton joints for SKELETAL_JOINT edges.
                shock_struct = (
                    obs.get_shock_structure()
                    if hasattr(obs, "get_shock_structure")
                    else None
                )
                shock_joints = []
                if shock_struct:
                    _segs, _joints, _ens, _meta = shock_struct
                    shock_joints = [
                        (j["trunk"], j["limb"], j["signal"], j["angle"], j["ratio"], j["type"], j["attach"])
                        for j in _joints
                    ]

                if iteration == 0:
                    self.binder.step(
                        t, D_v1,
                        anchor_positions=anchor_positions,
                        extraction_scale=extraction_scale,
                        t_junctions=(obs.get_t_junctions()
                                     if hasattr(obs, "get_t_junctions")
                                     else None),
                        shock_joints=shock_joints,
                    )
                else:
                    self.binder.bind_intra_step(t, D_v1)

                D_higher: set = set()
                if self.higher_lens_runner is not None:
                    D_higher = self.higher_lens_runner(
                        t, D_v1, obs, self.cg, envelope_mask=boxes)
                    self.binder.bind_intra_step(t, D_higher)

                # Emit + drain the higher-tier accumulator, as the
                # draft requires. [Declared omission] The draft also
                # called `clear_higher_accumulator()` here; the live
                # tree deliberately has NO mid-step caller for it
                # (B.8 note: a mid-step caller would drop same-step
                # registrations), so the accumulator stays owned by
                # the step lifecycle.
                for level in ("v2", "v4", "lo_pit"):
                    self.binder.emit_pending_edges(level)

                all_dets = set(D_v1) | set(D_higher)
                hyps = self._extract_hypotheses(
                    all_dets, source_obs=obs, source_scale=1,
                    priority_ids=coarse_priority)

                # Compute top-down gain for next iteration (laminar recurrence)
                if iteration < P.SETTLE_MAX_ITERS - 1 and hyps:
                    top_down_gain = self._compute_top_down_gain(hyps, obs, anchor_positions, all_dets)
                else:
                    top_down_gain = {}

                final_dets = all_dets
                final_state = S_t
                final_hash = s_t
                final_T = T_prev

                if iteration > 0:
                    if (_hyp_stability(prev_hyps, hyps)
                            > 1.0 - P.SETTLE_STABILITY_EPS
                            and _det_stability(prev_all_dets, all_dets)
                            > 1.0 - P.SETTLE_STABILITY_EPS):
                        break
                prev_hyps = hyps
                prev_all_dets = all_dets

            derived_out = self.derived.run_all(t, final_dets, self.cg)
            self._prev_hypotheses = prev_hyps
            return final_dets, derived_out, final_state, final_hash, final_T
        finally:
            self.binder.finalize_step()

    def _extract_hypotheses(self, D_raw, source_obs, source_scale,
                            priority_ids=None):
        """Active concepts evidenced by `D_raw` -> ranked hypotheses.

        `priority_ids` (from the coarse pass) nudges activation; it does
        NOT contribute boxes.
        """
        priority = set(priority_ids or ())
        hyps: List[Hypothesis] = []
        for cid, c in self.cg.concepts.items():
            if c.status not in (ConceptStatus.CRYSTALLIZED,
                                ConceptStatus.STABLE):
                continue
            if c.dormant or c.soft_dormant or c.provisional:
                continue
            matching = [d for d in D_raw
                        if self._matches_concept_cached(d, c)]
            if not matching:
                continue
            # Sort/cap BEFORE building boxes, so the box list is bounded
            # by HYPOTHESIS_MAX_BOXES too (activation alone was bounded
            # in the superseded order -- the unbounded-box case).
            matching.sort(key=lambda d: -d.z)
            matching = matching[:P.HYPOTHESIS_MAX_BOXES]
            boxes = []
            for d in matching:
                pos = None
                if hasattr(source_obs, "get_region_centroid_by_anchor"):
                    pos = source_obs.get_region_centroid_by_anchor(d.alpha)
                if pos is None:
                    continue
                x = pos[0] * source_scale
                y = pos[1] * source_scale
                pad = 8.0 * source_scale
                boxes.append((x - pad, y - pad, x + pad, y + pad))
            if not boxes:
                continue
            activation = max(d.z for d in matching)
            if cid in priority:
                # Draft literal 1.2 (no new parameter is added by this
                # step; promoting it to parameters.py is its own change).
                activation *= 1.2
            hyps.append(
                Hypothesis(cid, c.sigma, tuple(boxes), activation))
        hyps.sort(key=lambda h: (-h.activation, h.concept_id))
        return hyps[:P.SETTLE_TOP_K]

    def _compute_top_down_gain(self, hyps, obs, anchor_positions, all_dets=None):
        """Compute laminar top-down attentional gain from hypotheses.

        For each hypothesis, extract the dominant lens/signature that
        evidences it, and generate a gain field over anchors that match
        that lens/signature. This implements B.48 dimension-specific
        feature-based attention in the laminar recurrence pathway.
        """
        if not hyps or not anchor_positions:
            return {}

        # Collect attended (lens, signature) pairs from top hypotheses
        attended_pairs = []
        for h in hyps[:P.SETTLE_TOP_K]:
            # The concept's evidence_registry contains (idx, sig, fam, layer, scale, anchor)
            for entry in self.cg.concepts.get(h.concept_id, type('', (), {'evidence_registry': []})()).evidence_registry:
                _idx, sig, _fam, layer, _scale, _anchor = entry
                if layer == 0 and hasattr(sig, 'quantized'):
                    attended_pairs.append((sig.quantized, sig.class_))
                    break  # one representative per hypothesis

        if not attended_pairs:
            return {}

        # Aggregate gain across hypotheses (max pooling)
        gain_by_anchor = {}
        # Build lens->detections map for efficient lookup
        detections_by_lens = {}
        if all_dets:
            for d in all_dets:
                if d.k not in detections_by_lens:
                    detections_by_lens[d.k] = []
                detections_by_lens[d.k].append(d)

        for anchor, pos in anchor_positions.items():
            if not pos or len(pos) < 2:
                continue
            max_gain = 0.0
            for attended_quantized, attended_class in attended_pairs:
                # Find detections at this anchor matching the attended class
                matching_dets = detections_by_lens.get(attended_class, [])
                if not matching_dets:
                    continue
                # Use feature_attention_gain for this lens
                gain = feature_attention_gain(
                    matching_dets,
                    attended_class,
                    attended_quantized,
                    beta=P.ATTENTION_BETA_FEAT * 0.5  # reduced for top-down
                )
                if anchor in gain:
                    max_gain = max(max_gain, gain[anchor])
            if max_gain > 0.0:
                gain_by_anchor[anchor] = max_gain

        return gain_by_anchor

    def _matches_concept_cached(self, d, c):
        key = (c.id, hash(d))
        cached = self._match_cache.get(key)
        if cached is not None:
            return cached
        result = _matches_concept(d, c)
        self._match_cache[key] = result
        return result

    def _concepts_hit(self, D_raw):
        """Concept ids evidenced by `D_raw`, regardless of position.

        Method, not module function: the settle loop calls it as
        `self._concepts_hit`, and at module scope that is an
        AttributeError on the first coarse pass.
        """
        out = set()
        for cid, c in self.cg.concepts.items():
            if c.status not in (ConceptStatus.CRYSTALLIZED,
                                ConceptStatus.STABLE):
                continue
            if c.dormant or c.soft_dormant or c.provisional:
                continue
            for d in D_raw:
                if self._matches_concept_cached(d, c):
                    out.add(cid)
                    break
        return out


def feature_attention_gain(prev_detections, attended_lens, attended_value,
                           beta=None):
    """[B.48] Dimension-specific feature-based attention.

    The tuning curve of a lens is its signature; attending to a value V
    of lens L boosts only detections whose lens == L. Others untouched
    (Reynolds & Heeger 2009). Corrects B.32/B.41, which compared raw
    quantized vectors across lenses: a Texture signature that happens to
    correlate with an attended Color vector is NOT boosted.

    Returns {alpha: gain} -- the live `gain_by_anchor` shape of
    LensExecutor.run_all, so adoption needs no new interface.
    """
    if attended_lens is None or attended_value is None:
        return {}
    beta = P.ATTENTION_BETA_FEAT if beta is None else beta
    target = tuple(float(x) for x in attended_value)
    tn = math.sqrt(sum(x * x for x in target)) + P.EPSILON
    out = {}
    for d in prev_detections:
        if d.k != attended_lens:
            continue
        sig = getattr(d.sigma, "quantized", None)
        if sig is None or len(sig) != len(target):
            continue
        sn = math.sqrt(sum(x * x for x in sig)) + P.EPSILON
        cos = sum(a * b for a, b in zip(sig, target)) / (sn * tn)
        if cos <= 0.0:
            continue
        out[d.alpha] = beta * min(1.0, cos)
    return out


def _matches_concept(d, c):
    """Signature-level match against the concept's layer-0 evidence."""
    from perception.lenses.derived_common import _sig_compatible, _sig_similarity
    for entry in c.evidence_registry:
        _idx, sig, _fam, layer, _scale, _anchor = entry
        if layer != 0:
            continue
        if d.sigma == sig:
            return True
        if _sig_compatible(d.sigma, sig):
            if _sig_similarity(d.sigma, sig) >= P.HYPOTHESIS_MATCH_MIN:
                return True
    return False


def _build_boxes(hyps):
    """Union of hypothesis boxes, or None when nothing survived.

    Returns None -- never () -- when empty: `_emit` branches on
    `mask_boxes is not None`, and an empty tuple passes that test, which
    would silently drop the per-lens detection cap for that iteration.
    """
    if not hyps:
        return None
    boxes = []
    for h in hyps:
        boxes.extend(h.envelope_boxes)
    return tuple(boxes) if boxes else None


def _object_envelope(hyps):
    if not hyps:
        return None
    xs0, ys0, xs1, ys1 = [], [], [], []
    for h in hyps:
        for (x0, y0, x1, y1) in h.envelope_boxes:
            xs0.append(x0)
            ys0.append(y0)
            xs1.append(x1)
            ys1.append(y1)
    if not xs0:
        return None
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def _hyp_stability(prev, curr):
    if not prev and not curr:
        return 1.0
    if not prev or not curr:
        return 0.0
    a = {h.concept_id for h in prev}
    b = {h.concept_id for h in curr}
    return len(a & b) / max(len(a | b), 1)


def _det_stability(prev, curr):
    if not prev and not curr:
        return 1.0
    a = {hash(d) for d in prev}
    b = {hash(d) for d in curr}
    return len(a & b) / max(len(a | b), 1)
