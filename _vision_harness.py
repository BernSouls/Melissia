"""Route ETH-80 discrimination through the real Consolidator+Mapper path.

The existing probes (eval_discrimination.py, eval_cats_dogs.py) run numpy
1-NN on final-step signatures. They never touch the Consolidator, the
CoreLibrary, or the Mapper -- so every discrimination number measured so
far exercises a path the pipeline never uses.

This harness:
  1. builds a shared stack once (reused across images: the pipeline reuses
     its SensoryFloor/Binder/ConceptGraph; the harness must too)
  2. runs N>=8 train images per class, records windows into the
     ObservationWindowBuffer with label=("class", cls), polarity=+1 for
     same-class and -1 for other classes
  3. fires Consolidator.discriminative_core per class when the buffer is
     ready (>= LABEL_MIN_INSTANCES distinct positives + >=1 negative)
  4. runs held-out images through Mapper.query_category and reports the
     match / wrong-label / no-label outcome per class

Usage:
    python diagnostics/_vision_harness.py --train 8 --test 3 \
        --canon 512 --grid 16 --steps 4

Deviation from the drafted spec: none in the query path -- CategoryLabel
objects are used as drafted. The one change is the corpus: only 10 unique
photographs per class exist (the other 10 ETH-80 files were -map masks),
so the max split is 7 train / 3 test, not 8/3.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO / "src", REPO / "diagnostics", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

os.environ.setdefault("SIMILARITY_MODE", "linear")
os.environ.setdefault("DUAL_FILTER_ENABLED", "0")
os.environ.setdefault("PROBE_MODE", "patches")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import parameters as P  # noqa: E402
from binder import Binder, _detection_sort_key  # noqa: E402
import derived_lenses as DL  # noqa: E402
from concept_graph import ConceptGraph  # noqa: E402
from sensory_floor import Observation2D, SensoryFloor  # noqa: E402
import patch_extraction as PE  # noqa: E402

from pipeline import surround_suppress, observation_modality  # noqa: E402
from activation_engine import ActivationEngine  # noqa: E402
from prediction import PredictionModule  # noqa: E402
from fertility_engine import FertilityEngine  # noqa: E402
from pipeline import PipelineProviders  # noqa: E402
from builder import Builder  # noqa: E402
from consolidator import Consolidator  # noqa: E402
from mapper import Mapper  # noqa: E402
from pipeline import ObservationWindowBuffer  # noqa: E402

DATA = REPO / "data" / "ETH-80-master"   # [G.1.6c] --data overrides; default per architect

# [F6 falsifier] Per-class consolidator diagnostic deltas, keyed by class
# name; written by consolidate_pass, read by main()'s delta panel.
_PER_CLASS_DIAG: Dict[str, Dict[str, Any]] = {}

# [H.1] Per-class extension-state capture. A list of tuples rather than
# a dict, because the values are overwrite-per-call.
_PER_CLASS_EXT_STATE: List[Tuple[str, float, int, int, List[float]]] = []
CLASSES = ("apple", "car", "cow")

# [I.1] (image, t, len(builder_out.cross_sections)) -- reporting only.
_builder_xs: List[Tuple[str, int, int]] = []

# Edge types expected to fire on the vision harness config (static frames,
# no actions, no temporal history). If any of these report zero emissions
# at end of run, a lens tier or binder channel is silently inactive.
#
# Not on this list (see _NOT_EXPECTED_ON_HARNESS for the reasons):
#   CO_ACTIVATION - deliberately off via static_frame=True
#   CO_MOTION     - requires anchor_velocities, which this config lacks
#   CURVATURE     - requires ORIENTATION_SEGMENT detections; this
#                   extraction mode emits only ORIENTATION / MAJOR_AXIS
_EXPECTED_EDGE_TYPES = frozenset({
    # V1 spatial (static-frame capable)
    "CONTACT", "DEGREE", "TOPOLOGY",
    "CO_LOCATED",
    "CONTRAST", "PERIODICITY", "PERSISTENCE",
    # V2 lateral + vertical
    "ORIENTATION_CONTINUITY",
    "ORIENTATION_BEND", "ORIENTATION_JUNCTION",
    "CONTOUR_CONTINUITY", "JUNCTION_CONTOUR",
    "CONTOUR_ORIENTATION", "JUNCTION_ORIENTATION",
    # V4
    "CURVE_CONTAINMENT",
    # IT
    # [Pass 20] OBJECT_PART removed from the expected set: its only emitter
    # (_run_object_part) keyed on ObjectShape detections, and the aIT tier is
    # gone with the oracle.
    "PART_CONTAINMENT", "PART_CONTENT",
})

# Edge types NOT expected on this config (need temporal history, actions,
# or 3D/T-junction input). Reported as informational, not warnings.
#
#   CO_ACTIVATION   - deliberately off via static_frame=True
#   CO_MOTION       - requires anchor_velocities, which this config lacks
#   CURVATURE       - requires ORIENTATION_SEGMENT detections, which this
#                     extraction mode never produces (the runner executed
#                     8x with zero candidates in the K.0-fix run).
#   SURFACE_GROUPING - KNOWN DEFECT, not a config limitation: every input
#                     gate passes (120 Color / 114 Texture / 99 positioned
#                     anchors measured) and the channel dies at its own
#                     spread gate. With SURFACE_GROUPING_SPREAD_MIN=0 it
#                     emits 768 edges. Listed here so the banner does not
#                     double-report an open defect; removing it from this
#                     set is the fix's acceptance test.
_NOT_EXPECTED_ON_HARNESS = frozenset({
    "CO_ACTIVATION", "CO_MOTION",
    "CURVATURE", "SURFACE_GROUPING",
    "TRANSITION", "SEQUENCE", "SYNCHRONY",
    "OCCLUSION",
})



def build_stack():
    """One shared stack. Reused across all images (mirrors the pipeline)."""
    cg = ConceptGraph()
    cg.params["action_rep"] = []
    activation = ActivationEngine(cg)
    prediction = PredictionModule()
    fertility = FertilityEngine(cg, activation, planning_engine=None,
                                prediction_module=prediction)
    providers = PipelineProviders(cg)
    binder = Binder(cg, bootstrap_relax_orthogonality=False)
    builder = Builder(cg, fertility, prediction_module=prediction)
    consolidator = Consolidator(concept_graph=cg,
                                signature_provider=providers.signature_provider)
    core_library = consolidator.get_core_library()
    mapper = Mapper(cg, core_library, providers.signature_provider, prediction)
    prediction.set_core_library(core_library)
    prediction.set_situation_provider(mapper.build_situation_subgraph)
    prediction.set_containment_checker(mapper.mcs_containment)
    window_buffer = ObservationWindowBuffer(
        capacity_per_label=P.LABEL_WINDOW_CAPACITY,
        min_instances=P.LABEL_MIN_INSTANCES,
        max_age_steps=P.LABEL_MAX_AGE_STEPS,
    )
    window_buffer.attach_sig_provider(providers.signature_provider)
    # One shared SensoryFloor (lens homeostatic state evolves across images).
    sensory_floor = SensoryFloor()
    return {
        "cg": cg, "activation": activation, "prediction": prediction,
        "fertility": fertility, "providers": providers,
        "binder": binder,
        "builder": builder, "consolidator": consolidator,
        "core_library": core_library, "mapper": mapper,
        "window_buffer": window_buffer, "sensory_floor": sensory_floor,
        # [I.1] Activation reads the PREVIOUS step's detections.
        "prev_D_bind": frozenset(),
        "prev_D_rec": frozenset(),
    }


def observe_image(path, canon, grid, steps, stack):
    """Run the lens+binder pipeline for `steps` on one image.

    Applies surround_suppress (Phase B B.4) before the higher lenses, so
    the V4/Part/ObjectShape signatures are the ones the pipeline would
    produce. Returns the final-step situation window and its state hash.
    """
    arr = np.array(Image.open(path).convert("RGB").resize(
        (canon, canon), Image.BILINEAR))
    PE.GRID_N = grid
    regions = PE._regions_from_patches(arr)
    obs = Observation2D(regions, arr.shape[:2])
    ap = obs.get_positions() or {}
    es = obs.get_extraction_scale() or 0.0

    # SensoryFloor.step returns (D_raw, S_t, s_t, T_prev); s_t is the int
    # state hash at index 2 and the 4th element is None. The state index
    # is keyed by int, so an unpinned s_t raises TypeError there.
    step_result = stack["sensory_floor"].step(obs)
    D_raw, S_t, s_t = step_result[0], step_result[1], step_result[2]

    # Phase B B.4 -- surround suppression. Same call the pipeline makes.
    modality = observation_modality(obs)
    D_v1_suppressed = surround_suppress(
        frozenset(D_raw), ap, modality, es,
    )

    byk = defaultdict(list)
    for d in sorted(D_v1_suppressed, key=_detection_sort_key):
        byk[d.k].append(d)
    byk = dict(byk)

    cg = stack["cg"]
    b = stack["binder"]
    final_window = None
    for t in range(steps):
        b.note_step_begin(t)
        b.step(t, D_v1_suppressed, anchor_positions=ap or None,
               anchor_velocities=None, static_frame=True,
               extraction_scale=es)
        # [A/B] Three arms:
        #   W1_ROUTING=seg  -> the segmenter observation (from_image), i.e. a
        #                      REAL legacy arm: contours traced, boundary
        #                      polylines present, shape lenses reading them
        #   W1_ROUTING=bak  -> the historical wiring the stored baselines saw
        #                      (S_t instead of obs; region path, so no
        #                      segmenter ran and every polyline was absent)
        #   otherwise       -> field-derived entities on the region path
        _routing = os.environ.get("W1_ROUTING", "fix")
        # [A/B] the field arm engages the chain-geometry entity source. The
        # flag is an ARM parameter, not a default switch (Slice 0b), so it is
        # turned on here for this arm only rather than in parameters.
        if _routing == "field":
            import parameters as _P
            _P.FIELD_REGION_SOURCE = True
        if _routing == "seg":
            obs = Observation2D.from_image(arr, rich=False)
        _obs = S_t if _routing == "bak" else obs
        _bak = _routing == "bak"
        bo = DL.BorderOwnershipLens().step(t, cg, None, ap, None, byk, es, observation=_obs)
        b.bind_intra_step(t, frozenset(bo))
        surf = DL.SurfaceLens().step(
            t, cg, None, ap, None,
            {"Boundary": byk.get("Boundary", []),
             "BorderOwnership": list(bo),
             "Color": byk.get("Color", []),
             "Texture": byk.get("Texture", [])}, es)
        b.bind_intra_step(t, frozenset(surf))
        ct = DL.ContourLens().step(
            t, cg, None, ap, None,
            {"Orientation": byk.get("Orientation", []),
             "BorderOwnership": list(bo)}, es)
        jn = DL.JunctionLens().step(
            t, cg, None, ap, None,
            {"Orientation": byk.get("Orientation", [])}, es)
        # [A5] Peak detections. OWNED by PeakLens: it is the only writer of
        # the PEAK_ANCHOR_BASE+i -> (x, y) mapping that chain members are
        # resolved against. Without this call `ap` carries only the region
        # anchors, every chain member lookup misses, and the field
        # entity source yields nothing -- the field arm silently scores 0.
        pk = DL.PeakLens().step(
            t, cg, None, ap, None,
            {"Orientation": byk.get("Orientation", [])}, es,
            observation=S_t, luminance=arr)
        # [A5] Peak chains. Joined into the v2 registration set BEFORE
        # bind_intra_step/emit_pending_edges so the chain's containment edges
        # are emitted by the SAME single v2 pass -- a second
        # emit_pending_edges("v2") would double-fire every v2 channel.
        chain = DL.ChainLens().step(
            t, cg, None, ap, None,
            {"Orientation": byk.get("Orientation", [])}, es,
            observation=S_t, luminance=arr)
        b.bind_intra_step(t, frozenset(ct) | frozenset(jn) | frozenset(pk)
                        | frozenset(chain))
        b.emit_pending_edges("v2")
        # [W1] The Shape lenses need the chain detections to build their
        # field entities. `current_detections` is threaded here so the lens
        # does not have to reach back into the pipeline for them.
        _w1_dets = {
            "PEAK_CHAIN": list(chain),
            "Boundary": list(ct),
            "Junction": list(jn),
            "BorderOwnership": list(bo),
            "Texture": list(byk.get("Texture", [])),
            "Color": list(byk.get("Color", [])),
        }
        # [A/B] W1_ROUTING=bak reproduces the HISTORICAL wiring the stored
        # baselines were measured under: the Shape lenses were handed S_t
        # (the int state hash) instead of obs, and no current_detections at
        # all. Kept so a baseline-contamination diff is a measurement rather
        # than an assertion about what the old code did.
        _dets = {} if _bak else _w1_dets
        v4_cc = DL.CurveComplexLens().step(
            t, cg, None, ap, None,
            {"Contour": list(ct), "Junction": list(jn)}, es)
        v4_rs = DL.RegionShapeLens().step(
            t, observation=_obs, anchor_positions=ap, extraction_scale=es,
            current_detections=_dets)
        v4_hs = DL.HolisticShapeLens().step(
            t, observation=_obs, anchor_positions=ap, extraction_scale=es,
            current_detections=_dets)
        v4_sf = DL.SurfaceLens().step(
            t, cg, None, ap, None, _dets, es, observation=_obs,
            luminance=arr)
        v4_tb = DL.TextureBoundaryLens().step(
            t, cg, None, ap, None, _dets, es, observation=_obs,
            envelope_mask=None, luminance=arr)
        v4 = v4_cc + v4_rs + v4_hs + v4_sf + v4_tb
        b.bind_intra_step(t, frozenset(v4))
        b.emit_pending_edges("v4")
        # PartLens removed: PART_CONTAINMENT and PART_CONTENT edges now emitted
        # by geometry-based binders from RegionShape and CurveComplex detections.
        b.emit_pending_edges("lo_pit")
        # [Pass 20] The aIT tier is gone: ObjectShapeLens was excised as an
        # oracle (IT objecthood is emergent from consolidated part relations,
        # not a hardcoded whole-object descriptor), so there is no D_ait to
        # register and no "ait" level to emit.
        # [A5] The peak and chain detections belong in D_bind, mirroring the
        # live pipeline (orchestration/pipeline.py:1263-1292: "D_v2 = D_v2 +
        # list(D_peak) + list(D_chain)"), which then feeds D_bind_all and the
        # state index. Omitting them here made every harness window
        # chain-free: the recorded subgraph carried no PEAK_CHAIN node, so
        # chain-level edges (containment, anti-phase containment) could
        # never appear in the measured graph and the state patterns the
        # mapper sees were not the ones the live tree records.
        D_bind_all = (D_v1_suppressed | frozenset(bo) | frozenset(surf)
                      | frozenset(ct) | frozenset(jn) | frozenset(pk)
                      | frozenset(chain) | frozenset(v4))

        # [I.1] Mini-pipeline extension. Phase order mirrors pipeline.py:
        #   Phase 3 (activation) before Phase 4 (builder) before
        #   Phase 5 (candidates -> crystallization -> edge EMA -> witness
        #   -> fertility). The builder reads concept activations for
        #   frontier prioritization, so activation must be current.
        #
        # The vision harness has no actions: prev_action=None throughout.
        # CHUNK crystallization is disabled by passing active_plans=[].
        # _builder_xs records the cross-section count per step (reporting).

        # Phase 3: activation.
        if hasattr(stack["activation"], "set_time"):
            stack["activation"].set_time(t)
        elif hasattr(stack["activation"], "t"):
            stack["activation"].t = t
        stack["activation"].set_previous_detections(
            stack["prev_D_bind"], stack["prev_D_rec"]
        )
        stack["activation"].step(t, None, None, s_t)
        if hasattr(stack["activation"], "compute_onsets"):
            stack["activation"].compute_onsets(t)

        # Phase 4: builder.
        builder_out = stack["builder"].step(t, set(D_bind_all), s_t)
        _builder_xs.append((path.name, t, len(builder_out.cross_sections)))

        # Phase 5a: candidate stats -> crystallization.
        cg.step_21_update_candidate_stats(
            list(builder_out.cross_sections), t,
            prev_action=None,
            a_hist=None,
            onset_history=None,
        )
        cg.step_23_candidate_forgetting_and_capacity()
        cg.step_24_tombstone_expiry()
        cg.step_25_evaluate_crystallization(
            t=t,
            activation_engine=stack["activation"],
            active_plans=[],
            s_t=s_t,
            action_t=None,
        )

        # Phase 5b: edge EMA. [I.1 fix] The activation engine MUST be
        # passed -- `_concept_edge_signal` calls
        # `activation_engine.get_onset(...)`, raising AttributeError on
        # `None` the moment a concept-edge is processed.
        cg.step_28_update_edges(stack["activation"], t)

        # Phase 5c: witness + fertility, after edge EMA.
        cg.step_28_update_witness_and_status(t)
        stack["fertility"].step30_recompute_RF_GP_EF(t)

        # Persist current detections for the next step's activation.
        stack["prev_D_bind"] = D_bind_all
        stack["prev_D_rec"] = frozenset()

        # The record captures the current step's state, after the phase
        # chain has written this step's edges into the graph.
        stack["providers"].state_index.record(s_t, set(D_bind_all))
        # pipeline.py finalizes the Binder at the END of the step (after
        # every phase), so the mini-pipeline runs on unfinalized state.
        b.finalize_step()
        final_window = stack["providers"].state_pattern_provider(s_t)

    return final_window, s_t



def _spread_views(views: List[Path], k: int) -> List[Path]:
    """[MORE-DATA] Pick k views evenly spaced over one instance's sorted
    view list (k == 1 -> the middle view). Deterministic; duplicates are
    impossible while k <= len(views), which callers enforce via min()."""
    if k <= 0:
        return []
    if k >= len(views):
        return list(views)
    if k == 1:
        return [views[len(views) // 2]]
    idx = [round(j * (len(views) - 1) / (k - 1)) for j in range(k)]
    return [views[i] for i in idx]


def _enumerate_stratified(n_train: int, n_test: int):
    """[MORE-DATA] Instance-disjoint, view-spread split over the full corpus.

    The default split takes the first n_train+n_test SORTED paths, which in
    ETH-80-master are all views of instance 1 (paths sort as
    car1-000-000, car1-022-000, ...): train and test share one object and
    the other 9 instances never contribute. This variant:

      1. groups each class's photos by instance (stem digits before the
         first '-'),
      2. assigns instances ~proportionally to the n_train:n_test ratio
         (7:3 over 10 instances -> 7 train / 3 test), keeping train and
         test instances DISJOINT,
      3. spreads each instance's share of the count evenly over its 41
         views, and orders per-class lists view-major (view 0 of every
         instance before view 1 of any) so capacity-bounded buffers
         (LABEL_WINDOW_CAPACITY=8) retain cross-instance diversity,
      4. interleaves classes round-robin so each label's negative deque
         receives a balanced mix instead of one whole class block.

    Deterministic; used only when --stratify is passed, so the canonical
    7/3 protocol output is byte-identical to previous runs.
    """
    per_cls_train: Dict[str, List[Path]] = {}
    per_cls_test: Dict[str, List[Path]] = {}
    for cls in CLASSES:
        paths = sorted(
            p for p in DATA.rglob(f"{cls}*.png")
            if "-map" not in p.name
        )
        groups: Dict[str, List[Path]] = {}
        for p in paths:
            head = p.stem[len(cls):]
            inst = head.split("-", 1)[0]
            groups.setdefault(inst, []).append(p)
        insts = sorted(
            groups,
            key=lambda s: (0, int(s), "") if s.isdigit() else (1, 0, s),
        )
        if len(insts) < 2:
            raise SystemExit(
                f"class {cls}: need >= 2 instances to stratify, "
                f"found {len(insts)} in {DATA}"
            )
        frac = n_test / max(1, (n_train + n_test))
        n_test_inst = max(1, min(len(insts) - 1, round(len(insts) * frac)))
        n_train_inst = len(insts) - n_test_inst
        tr_insts, te_insts = insts[:n_train_inst], insts[n_train_inst:]

        def _select(sel_insts, total):
            base, rem = divmod(total, len(sel_insts))
            counts = [base + (1 if i < rem else 0)
                      for i in range(len(sel_insts))]
            chosen = {
                inst: _spread_views(groups[inst], counts[i])
                for i, inst in enumerate(sel_insts)
            }
            out: List[Path] = []
            for j in range(max(counts, default=0)):
                for inst in sel_insts:
                    if j < len(chosen[inst]):
                        out.append(chosen[inst][j])
            return out

        per_cls_train[cls] = _select(tr_insts, n_train)
        per_cls_test[cls] = _select(te_insts, n_test)
        if (len(per_cls_train[cls]) < n_train
                or len(per_cls_test[cls]) < n_test):
            raise SystemExit(
                f"class {cls}: not enough views for {n_train}/{n_test} "
                f"across {n_train_inst}/{n_test_inst} instances in {DATA}"
            )

    def _interleave(per_cls):
        out = []
        lists = [per_cls[c] for c in CLASSES]
        for i in range(max((len(lst) for lst in lists), default=0)):
            for cls, lst in zip(CLASSES, lists):
                if i < len(lst):
                    out.append((cls, lst[i]))
        return out

    return _interleave(per_cls_train), _interleave(per_cls_test)


def enumerate_split(n_train, n_test, stratify: bool = False):
    if stratify:
        return _enumerate_stratified(n_train, n_test)
    train, test = [], []
    for cls in CLASSES:
        # [G.1.6c] rglob (ETH-80 nests: <class_idx>/<object_idx>/<name>.png)
        # and exclude -map.png masks: they are segmentations, not photos,
        # and would double the nominal count.
        paths = sorted(
            p for p in DATA.rglob(f"{cls}*.png")
            if "-map" not in p.name
        )
        if len(paths) < n_train + n_test:
            raise SystemExit(
                f"class {cls}: need {n_train + n_test} images, found "
                f"{len(paths)} in {DATA}"
            )
        train.extend((cls, p) for p in paths[:n_train])
        test.extend((cls, p) for p in paths[n_train:n_train + n_test])
    return train, test


def train_pass(items, canon, grid, steps, stack, t_counter):
    buf = stack["window_buffer"]
    for cls, path in items:
        window, s_t = observe_image(path, canon, grid, steps, stack)
        t_counter[0] += steps
        if window is None or not window.nodes:
            continue
        buf.record(("class", cls), +1, window, t_counter[0])
        for other in CLASSES:
            if other == cls:
                continue
            buf.record(("class", other), -1, window, t_counter[0])


def consolidate_pass(stack, t_counter, labels, f1_falsifier=False):
    """[G.1.6c] f1_falsifier=True advances t between labels. The canonical
    instrument keeps all labels at one t, which means they share one MCS
    budget ledger (consolidator._reset_step_budget only zeroes on t change).
    """
    buf = stack["window_buffer"]
    cons = stack["consolidator"]
    results = {}
    for label in labels:
        if f1_falsifier:
            t_counter[0] += 1
        t = t_counter[0]
        pos_n, neg_n = buf.counts(label)
        if not buf.ready(label):
            results[label] = ("not_ready", f"t={t} pos={pos_n} neg={neg_n}")
            continue
        pair = buf.pop_windows(label)
        if pair is None:
            results[label] = ("no_pair", "")
            continue
        positives, negatives = pair
        if not positives or not negatives:
            results[label] = ("empty",
                              f"pos={len(positives)} neg={len(negatives)}")
            continue
        # [F6 falsifier] Snapshot the consolidator's full diagnostics
        # dict before and after the call. Per-class deltas are what
        # attribute the break-cause panel (a)/(b)/(c) to a specific
        # class when all classes consolidate at one `t`.
        before = dict(cons.diagnostics)
        import time as _time  # [F6d] local import; reporting only
        _t0 = _time.time()
        entry = cons.discriminative_core(
            t, label, positives, negatives,
            source_concept_id=None, initial_reliability_factor=1.0)
        _dt = _time.time() - _t0
        after = dict(cons.diagnostics)
        # [F6d] Per-class wall-time line so the 1e9-budget run can be
        # aborted per the brief's 10-minute-per-class rule (the pass
        # otherwise prints only after all classes return).
        print(f"    [{label[1] if isinstance(label, tuple) else label}] "
              f"call done in {_dt:.1f}s "
              f"(exit_break +{after.get('dc_exit_break', 0) - before.get('dc_exit_break', 0)}, "
              f"exit_classified +{after.get('dc_exit_classified', 0) - before.get('dc_exit_classified', 0)}, "
              f"ext_skipped +{after.get('extension_skipped_budget', 0) - before.get('extension_skipped_budget', 0)})",
              flush=True)
        # Per-class deltas for scalar int counters only. Non-int values
        # (lists, dicts, floats) are passed through unchanged.
        class_key = label[1] if isinstance(label, tuple) else str(label)
        # [H.1] capture AFTER class_key exists (the brief placed this
        # immediately after `after = dict(...)`, but in this tree
        # class_key is defined here, below the wall-time line).
        _PER_CLASS_EXT_STATE.append((
            class_key,
            float(after.get("dc_last_ext_margin", float("nan"))),
            int(after.get("dc_last_ext_count", 0)),
            int(after.get("dc_last_ext_mcs_evaluated", 0)),
            list(after.get("dc_ext_margin_trace", [])),
        ))
        deltas = {}
        for k, v in after.items():
            if isinstance(v, int) and not isinstance(v, bool):
                deltas[k] = v - before.get(k, 0)
            elif isinstance(v, float):
                deltas[k] = v - float(before.get(k, 0.0))
        _PER_CLASS_DIAG[class_key] = deltas
        if entry is None:
            results[label] = ("failed", "discriminative_core returned None")
        else:
            results[label] = ("extracted",
                              f"t={t} "
                              f"hash={entry.canonical_hash.hex()[:12]} "
                              f"nodes={len(entry.core_subgraph.nodes)} "
                              f"edges={len(entry.core_subgraph.edges)} "
                              f"reliability={entry.reliability:.3f}")
    return results


def margin_pass(items, canon, grid, steps, stack, t_counter):
    """[B1/B2 gate] Per-image accuracy AND the true cross-class margin.

    Uses the mapper's read-only multi-label probe (mapper.py, `read_only=True`
    and a sequence of label signals). One call per image, because the scoring
    loop is label-agnostic and the query-core cache key is label-independent --
    so the returned CategoryMatch already holds the ranked union across ALL
    classes, including labels that were never requested.

    read_only suppresses _begin_step's step advance, _enqueue_gap, the
    label-history append and _write_back, so this pass measurably cannot
    perturb N_conf, reliability EMAs or the gap queue. That is verified by the
    state snapshot printed alongside the rows.

    Emits the row shape the gates need:
        image_id, ground_truth_class, predicted_class, is_correct, margin
    where margin = rank_score(top-1) - rank_score(top-2) over the union.
    """
    mapper = stack["mapper"]
    cons = stack["consolidator"]

    def snap():
        d = cons.diagnostics
        return {k: d.get(k) for k in (
            "category_queries", "evidence_accepted", "gaps_emitted",
            "dc_attempts", "discriminative_extracted")}

    before = snap()
    out = []
    for cls, path in items:
        window, _s_t = observe_image(path, canon, grid, steps, stack)
        t_counter[0] += steps
        if window is None or not window.nodes:
            out.append(dict(image=path.name, truth=cls, pred="", correct=False,
                            margin=0.0, n_labels=0))
            continue
        labels_requested = [("class", c) for c in CLASSES]
        match = mapper.query_category(t_counter[0], labels_requested, window,
                                      read_only=True)
        ranked = list(match.labels or [])
        if not ranked:
            out.append(dict(image=path.name, truth=cls, pred="", correct=False,
                            margin=0.0, n_labels=0))
            continue
        pred = str(ranked[0].label[1]) if ranked[0].label else ""
        margin = (float(ranked[0].rank_score) - float(ranked[1].rank_score)
                  if len(ranked) >= 2 else 0.0)
        out.append(dict(image=path.name, truth=cls, pred=pred,
                        correct=(pred == cls), margin=margin,
                        n_labels=len(ranked)))
    after = snap()
    perturbed = [k for k in before if before[k] != after[k]]
    print("  [non-perturbation check] accumulation state before/after probe: "
          + ("UNCHANGED" if not perturbed
             else "CHANGED on %s" % perturbed))
    print("    before: %s" % before)
    print("    after : %s" % after)
    return out


def query_pass(items, canon, grid, steps, stack, t_counter):
    mapper = stack["mapper"]
    # [Kc2] per_class[cls] = [presence, argmax, total].
    #   presence: the TRUE label appears anywhere in the returned labels
    #             (the harness's original criterion, unchanged).
    #   argmax:   the label query_category ranked FIRST (match.labels[0],
    #             ordered by rank_score after Kc2) IS the true label.
    # The two differ whenever the true label is returned but not ranked
    # top -- the case that made 4/9 presence read as 4/9 accuracy.
    rows, per_class = [], {c: [0, 0, 0] for c in CLASSES}
    for cls, path in items:
        window, s_t = observe_image(path, canon, grid, steps, stack)
        t_counter[0] += steps
        per_class[cls][2] += 1
        if window is None or not window.nodes:
            rows.append((cls, path.name, "no_window", 0.0, "",
                         "", 0.0, 0.0, 0.0))
            continue
        label = ("class", cls)
        match = mapper.query_category(t_counter[0], label, window)
        # CategoryMatch.labels holds CategoryLabel dataclasses
        # (consolidator.py:1134: core_hash, label, raw_score, confidence),
        # despite the List[Tuple[Any, float]] annotation on the dataclass.
        pairs = list(match.labels or [])
        hit = [c for c in pairs if c.label == label]
        if hit:
            per_class[cls][0] += 1
        # [Kc2] Top-1 as the model actually ranked it: query_category
        # returns its labels in rank order, so element 0 is the argmax.
        top1 = pairs[0] if pairs else None
        if top1 is not None and top1.label == label:
            per_class[cls][1] += 1
        t1 = (str(top1.label), top1.confidence, top1.raw_score,
              top1.rank_score) if top1 is not None else ("", 0.0, 0.0, 0.0)
        if hit:
            best = max(hit, key=lambda c: c.confidence)
            rows.append((cls, path.name, "match", best.confidence,
                         best.core_hash.hex()[:12]) + t1)
        elif pairs:
            top = max(pairs, key=lambda c: c.confidence)
            rows.append((cls, path.name, f"wrong:{top.label}",
                         top.confidence, top.core_hash.hex()[:12]) + t1)
        else:
            rows.append((cls, path.name, "no_labels", 0.0, "") + t1)
    return rows, per_class


def _set_corpus(classes: Tuple[str, ...], data: Path) -> None:
    """[H.3] Point the harness at a different (CLASSES, DATA) corpus."""
    global CLASSES, DATA
    CLASSES = classes
    DATA = data


def _print_attrition_trace(stack: Dict[str, Any]) -> None:
    """[K.1] Tier attrition across the content filter's four stages.

    The consolidator merges the filter's local diag dict under a `filter_`
    prefix (`self.diagnostics[f"filter_{k}"] = ...`), so the stage counters
    live at `filter_attrition_<stage>_<kind>_tier_<t>`. Reading them
    unprefixed would print an all-zero table silently.
    Cumulative over the run; per-class deltas also appear in the
    consolidator-deltas panel above.
    """
    cons_diag = stack["consolidator"].diagnostics
    stages = ["in", "mass", "topk", "final"]
    kinds = ["nodes", "edges"]
    tiers = [0, 1, 2, 3, "unknown"]
    print("\n--- tier attrition trace (cumulative) ---")
    print("| stage | kind | t0 | t1 | t2 | t3 | unknown |")
    print("|---|---|---|---|---|---|---|")
    for stage in stages:
        for kind in kinds:
            row = [stage, kind]
            for t in tiers:
                key = f"filter_attrition_{stage}_{kind}_tier_{t}"
                row.append(str(cons_diag.get(key, 0)))
            print("| " + " | ".join(row) + " |")


def _report_emitted_edge_types(stack: Dict[str, Any]) -> int:
    """Print cumulative emitted_by_type for every EdgeType.

    Returns the count of expected-but-zero types (0 == healthy).
    """
    from binder import EdgeType
    binder = stack.get("binder")
    if binder is None:
        print("[emitted-type banner] no binder in stack; skipping.")
        return -1
    emitted = binder.diagnostics.get("emitted_by_type", {}) or {}
    counts = {et.name: int(emitted.get(et.name, 0)) for et in EdgeType}

    print("\n--- emitted_by_type (cumulative over run) ---")
    print("| edge type | count | expected? |")
    print("|---|---|---|")
    expected_zero = 0
    for et in EdgeType:
        name = et.name
        c = counts.get(name, 0)
        if name in _EXPECTED_EDGE_TYPES:
            marker = "yes"
            if c == 0:
                marker = "**EXPECTED BUT ZERO**"
                expected_zero += 1
        elif name in _NOT_EXPECTED_ON_HARNESS:
            marker = "no (by config)"
        else:
            marker = "unclassified"
        print(f"| {name} | {c} | {marker} |")

    if expected_zero:
        print(
            f"\n[emitted-type banner] WARNING: {expected_zero} expected "
            f"edge type(s) emitted zero. The pipeline has a silent "
            f"inactive tier or channel."
        )
    else:
        print(
            "\n[emitted-type banner] OK: every expected edge type fired "
            "at least once."
        )
    return expected_zero


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--test", type=int, default=3)
    ap.add_argument("--stratify", action="store_true",
                    help="[MORE-DATA] Instance-disjoint, view-spread, "
                         "class-interleaved split over the full corpus "
                         "(default split takes only the first sorted "
                         "paths = one instance's views).")
    ap.add_argument("--canon", type=int, default=512)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--data", type=str,
                    default="data/ETH-80-master",
                    help="Corpus root (default: data/ETH-80-master). "
                         "Recursive match, -map.png masks excluded.")
    ap.add_argument("--f1-falsifier", action="store_true",
                    help="[F1 falsifier, harness-side only] Advance t between "
                         "classes in the consolidation pass so each label gets "
                         "a fresh MCS budget ledger.")
    ap.add_argument("--mcs-budget", type=int, default=None,
                    help="Experiment only (NOT a fix): override the module-"
                         "level consolidator.MCS_BUDGET_PER_STEP for this "
                         "run, so _consume_mcs_budget reads the new value. "
                         "parameters.py is untouched.")
    args = ap.parse_args(argv)

    _PER_CLASS_EXT_STATE.clear()   # [PF-6] idempotent across repeat calls

    global DATA
    DATA = Path(args.data)
    if not DATA.is_absolute():
        DATA = REPO / DATA

    if args.train < P.LABEL_MIN_INSTANCES:
        print(f"ERROR: --train {args.train} < LABEL_MIN_INSTANCES "
              f"{P.LABEL_MIN_INSTANCES}. CategoryManager trigger cannot fire.")
        return 1

    os.environ["GRID_N"] = str(args.grid)
    os.environ["CANONICAL_SIZE"] = str(args.canon)

    print(f"data   : {DATA}")
    print(f"config : canon {args.canon}, GRID_N {args.grid}, steps {args.steps}")
    print(f"split  : {args.train} train / {args.test} test per class")
    print(f"f1     : falsifier {'ON (t advances per class)' if args.f1_falsifier else 'off (one t for all classes)'}")
    print(f"trigger: MIN_INSTANCES={P.LABEL_MIN_INSTANCES}, "
          f"capacity={P.LABEL_WINDOW_CAPACITY}, max_age={P.LABEL_MAX_AGE_STEPS}")

    train, test = enumerate_split(args.train, args.test,
                                  stratify=args.stratify)
    _strat_tag = (" [stratified: instance-disjoint, view-spread, interleaved]"
                  if args.stratify else "")
    print(f"        {len(train)} train, {len(test)} test total{_strat_tag}")
    for cls in CLASSES:
        tr = [p.name for c, p in train if c == cls]
        te = [p.name for c, p in test if c == cls]
        print(f"        {cls:6s} train={tr}")
        print(f"        {cls:6s} test ={te}")
    print("--- config record (for the report) ---")
    print(f"  env PROBE_MODE={os.environ.get('PROBE_MODE')} "
          f"SIMILARITY_MODE={os.environ.get('SIMILARITY_MODE')} "
          f"DUAL_FILTER_ENABLED={os.environ.get('DUAL_FILTER_ENABLED')}")
    print(f"  params THETA_MIN_FREQ={P.THETA_MIN_FREQ} "
          f"THETA_MIN_CONSISTENCY={P.THETA_MIN_CONSISTENCY} "
          f"THETA_CLUSTER_CONTENT={P.THETA_CLUSTER_CONTENT} "
          f"THETA_CONTAINMENT={P.THETA_CONTAINMENT} "
          f"B0_PREFILTER_ENABLED={P.B0_PREFILTER_ENABLED}")
    import consolidator as _c0
    print(f"  params MCS_BUDGET_PER_STEP={_c0.MCS_BUDGET_PER_STEP} "
          f"THETA_SIM_CONSENSUS={P.THETA_SIM_CONSENSUS}")

    if args.mcs_budget is not None:
        import consolidator as _cons
        _cons.MCS_BUDGET_PER_STEP = args.mcs_budget
        print(f"NOTE: consolidator.MCS_BUDGET_PER_STEP overridden to "
              f"{args.mcs_budget} (experiment only; parameters.py untouched)")

    stack = build_stack()
    t_counter = [0]

    print("--- train pass (recording windows) ---")
    train_pass(train, args.canon, args.grid, args.steps, stack, t_counter)
    short = []
    for cls in CLASSES:
        pos, neg = stack["window_buffer"].counts(("class", cls))
        print(f"  {cls:6s} pos={pos:2d} neg={neg:2d} (distinct after dedup)")
        if pos < P.LABEL_MIN_INSTANCES:
            short.append(cls)
    if short:
        print(f"\nSTOP: {short} below LABEL_MIN_INSTANCES "
              f"{P.LABEL_MIN_INSTANCES}. The trigger cannot fire.")
        return 2

    stack["window_buffer"].evict_stale(t_counter[0])

    print("\n--- consolidation pass ---")
    t_counter[0] += 1
    labels = [("class", c) for c in CLASSES]
    results = consolidate_pass(stack, t_counter, labels,
                               f1_falsifier=args.f1_falsifier)
    n_ext = 0
    for label in labels:
        status, detail = results[label]
        n_ext += 1 if status == "extracted" else 0
        print(f"  {label[1]:6s} {status:10s} {detail}")
    print(f"\ncore_library size: {stack['core_library'].size()} "
          f"({n_ext} extracted)")

    print("\n--- builder cross_sections per step (first 5, I.1) ---")
    for _name, _t, _n in _builder_xs[:5]:
        print(f"  {_name} t={_t} cross_sections={_n}")

    print("\n--- builder diagnostics (non-zero, I.1) ---")
    for _k in sorted(stack["builder"].diagnostics.keys()):
        _v = stack["builder"].diagnostics[_k]
        if _v:
            print(f"  {_k}: {_v}")

    # [Pass 24] EdgeParams population at classification time. A high
    # empty share means _mcs_accumulate's graded path is running ungraded
    # -- params are starved, not absent by design.
    print("\n--- edge_params population (Pass 24) ---")
    _cd = stack["consolidator"].diagnostics
    _pop = _cd.get("edge_params_populated", 0)
    _empty = _cd.get("edge_params_empty_at_classification", 0)
    _tot = _pop + _empty
    print("  classification calls with params: %d" % _pop)
    print("  classification calls without:     %d" % _empty)
    print("  populated share: %.1f%%"
          % ((_pop / _tot * 100.0) if _tot else 0.0))
    print("  matched edges carrying params:     %d"
          % _cd.get("edge_params_edges_present", 0))

    print("\n--- concept graph (I.1) ---")
    _cg = stack["cg"]
    for _k in sorted(_cg.diagnostics.keys()):
        if _k.startswith("map_family"):
            print(f"  {_k}: {_cg.diagnostics[_k]}")
    print(f"  cg.concepts: {len(_cg.concepts)}")
    _fam_counts: Dict[str, int] = {}
    _status_counts: Dict[str, int] = {}
    for _c in _cg.concepts.values():
        _f = getattr(_c, "family", None)
        _fname = getattr(_f, "name", str(_f))
        _fam_counts[_fname] = _fam_counts.get(_fname, 0) + 1
        _sname = getattr(getattr(_c, "status", None), "name", "?")
        _status_counts[_sname] = _status_counts.get(_sname, 0) + 1
    for _k in sorted(_fam_counts):
        print(f"  family {_k}: {_fam_counts[_k]}")
    for _k in sorted(_status_counts):
        print(f"  status {_k}: {_status_counts[_k]}")

    print("\n--- per-class last-extension state (I.3) ---")
    for cls_key, margin, count, mcs_eval, trace in _PER_CLASS_EXT_STATE:
        trace_str = (
            " -> ".join(f"{m:+.4f}" for m in trace) if trace else "(none)"
        )
        print(f"  [{cls_key}] margin={margin:.6f} count={count} "
              f"mcs_evaluated={mcs_eval}")
        print(f"          trace: {trace_str}")

    print("\n--- per-class consolidator deltas ---")
    for cls_key in sorted(_PER_CLASS_DIAG.keys()):
        d = _PER_CLASS_DIAG[cls_key]
        print(f"\n[{cls_key}]")
        for k in sorted(d.keys()):
            v = d[k]
            if v != 0:
                print(f"  {k}: {v}")

    # [G.1.5] Branch distribution + content-tier budget.
    cons_diag = stack["consolidator"].diagnostics
    print("\n--- consolidator dc_* / content_* diagnostics ---")
    for key in sorted(cons_diag.keys()):
        if (key.startswith("dc_") or key.startswith("content_")
                or key.startswith("inter_") or key.startswith("mcs_")
                or key.startswith("extension_")
                or key.startswith("family_lookup")
                or key.startswith("discriminative_")):
            print(f"  {key}: {cons_diag[key]}")
    _sn = cons_diag.get("content_nodes_in_source", 0)
    _bn = cons_diag.get("content_nodes_built", 0)
    _se = cons_diag.get("content_edges_in_source", 0)
    _be = cons_diag.get("content_edges_built", 0)
    _nr = (_sn / _bn) if _bn else float("inf")
    _er = (_se / _be) if _be else float("inf")
    print(f"  nodes in_source={_sn} built={_bn}  ratio={_nr:.2f}x")
    print(f"  edges in_source={_se} built={_be}  ratio={_er:.2f}x")

    _print_attrition_trace(stack)

    # [B1/B2 gate] The read-only margin probe MUST run on pristine state,
    # BEFORE any mutating query. The mutating `query_pass` below calls
    # _write_back, which EMA-updates entry.reliability and concept.N_conf; those
    # feed candidate selection and signature collection, so a probe run
    # afterwards scores a different candidate set and different raw_scores for
    # the SAME image (measured: raw_score flips on 5 of 9 images, which is the
    # whole 4/9-vs-6/9 discrepancy). Running the probe first makes the
    # measurement independent of harness execution order.
    print("\n--- [B1/B2 gate] per-image accuracy and true cross-class "
          "margin (read-only probe, pristine state) ---")
    print("| image | truth | predicted | correct | margin | labels |")
    print("|---|---|---|---|---|---|")
    t_counter[0] += 1
    mrows = margin_pass(test, args.canon, args.grid, args.steps, stack,
                         t_counter)
    tot_correct = 0
    margins = []
    for r in mrows:
        tot_correct += 1 if r["correct"] else 0
        margins.append(r["margin"])
        print(f"| {r['image']} | {r['truth']} | {r['pred']} | "
              f"{'true' if r['correct'] else 'false'} | {r['margin']:.6f} | "
              f"{r['n_labels']} |")
    n_img = len(mrows)
    if n_img:
        print(f"\nGATE per-image accuracy: {tot_correct}/{n_img} = "
              f"{tot_correct / n_img:.4f}")
        print(f"GATE mean cross-class margin: {sum(margins) / n_img:.6f}")
        _pos = [m for m in margins if m > 0.0]
        print(f"GATE margin > 0 on {len(_pos)}/{n_img} images")

    # NOTE ON ORDER: the B1/B2 margin probe above runs FIRST, on pristine
    # state, and its numbers are the gate numbers. This `query_pass` runs
    # SECOND and is therefore POST-MUTATION: its own _write_back calls have
    # already EMA-updated core reliability and concept N_conf for the images
    # it processed, which measurably changes accuracy (6/9 pristine -> 4/9
    # post-mutation on this split). It is the pipeline's self-report, not the
    # gate figure.
    print("\n--- held-out query pass (MUTATING; runs SECOND, post-mutation) ---")
    t_counter[0] += 1
    rows, per_class = query_pass(test, args.canon, args.grid, args.steps,
                                 stack, t_counter)
    print("| true | image | result | confidence | matched label "
          "| top1 label | top1 conf | top1 raw_score | top1 rank_score |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (cls, name, result, conf, core,
         t1_label, t1_conf, t1_raw, t1_rank) in rows:
        print(f"| {cls} | {name} | {result} | {conf:.3f} | {core} "
              f"| {t1_label} | {t1_conf:.3f} | {t1_raw:.3f} "
              f"| {t1_rank:.3f} |")

    print("\n--- per-class ---")
    tot_ok, tot_n, tot_am = 0, 0, 0
    for cls in CLASSES:
        ok, am, n = per_class[cls]
        tot_ok += ok
        tot_am += am
        tot_n += n
        print(f"  {cls:6s} presence {ok}/{n}   argmax {am}/{n}")
    print(f"\nOVERALL presence: {tot_ok}/{tot_n}")
    print(f"OVERALL argmax:   {tot_am}/{tot_n}")
    print(f"core_library size: {stack['core_library'].size()}")

    n_zero = _report_emitted_edge_types(stack)
    if n_zero > 0:
        print(
            f"\n[emitted-type banner] run completed with {n_zero} "
            f"expected-but-zero edge type(s). Re-run diagnostics before "
            f"trusting these numbers."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))