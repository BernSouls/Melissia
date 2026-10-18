"""core/registries.py -- lens-level constitutional ranges and cycle lengths,
slot weights, arities and action-name helpers.

MOVE ONLY: statements extracted verbatim from primitives.py by
_reorg_move2_schema.py, preserving their order. primitives.py re-exports from
here. Zero project imports except core.hashing and parameters (a leaf module).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.hashing import clamp, hash_64, hash_64_signed, serialize_bytes
from core.population import PopAxis, validate_population

import parameters as P

ACTION_NULL: str = "null"

ACTION_NOOP: str = "stay"

PROBE_ACTION_PREFIX: str = "probe_"

def is_noop_action(action: Any) -> bool:
    """Constitutional null check (§2.6): action is None or explicit no-op ⊘."""
    return action is None or action == ACTION_NULL or action == ACTION_NOOP

def is_probe_action(action: Any, env: Any = None) -> bool:
    """Check if action is an exploratory sensorimotor probe.
    Prefers env-declared classification; falls back to protocol prefix."""
    if env is not None and hasattr(env, "is_probe_action"):
        try:
            return bool(env.is_probe_action(action))
        except Exception:
            pass
    return isinstance(action, str) and action.startswith(PROBE_ACTION_PREFIX)

def is_teleological_action(action: Any) -> bool:
    """Check if action is a goal reference rather than an executable motor command."""
    return isinstance(action, tuple) and len(action) >= 2 and action[0] == "goal"

def is_action_lens_eligible(action: Any) -> bool:
    """Single eligibility predicate for the first-class Action lens (B2).

    PURE by construction — the lens (inside SensoryFloor, which has no env
    reference) and the pipeline's execution-time onset pre-registration
    MUST reach the identical verdict for the same action, or the pattern's
    onset timing desynchronizes from its emission. Consulting an
    env-declared probe classifier here would reintroduce exactly that
    divergence, so this predicate uses only the constitutional checks:
    real, executable motor/manipulation actions are eligible; no-ops,
    probe-prefix actions, and teleological goal references are not.
    """
    if is_noop_action(action):
        return False
    if is_teleological_action(action):
        return False
    if isinstance(action, str) and action.startswith(PROBE_ACTION_PREFIX):
        return False
    return True

CONSTITUTIONAL_RANGES: dict[str, tuple[int, ...]] = {
    # --- Lens classes (§2.2) ---
    # Boundary, collapsed form: Ch 0 = circularity (10 bins), Ch 1 =
    # boundary complexity (10 bins). Emitted by the Boundary lens when the
    # observation exposes NO per-entity shape channels (symbolic envs), where
    # same-shape entities deliberately share one pattern node.
    "BOUNDARY": (10, 10),
    # Boundary, anchored form: [Arm C / ANCHOR-IN-SIGNATURE FIX] Arity-3:
    # (circularity_range, complexity_range, anchor_range). The anchor
    # component is a CATEGORICAL discriminator; its range is 0.0. In the
    # sum-form similarity (total_diff / sum(ranges)), a nonzero anchor
    # difference saturates total_diff → clamps to 0.0 for cross-anchor pairs;
    # same-anchor pairs contribute 0 on this slot, preserving graded shape
    # similarity on slots 0/1 exactly as before. Emitted by the Boundary lens
    # when the observation DOES expose shape channels (Observation2D/3D).
    "BOUNDARY_ANCHORED": (10, 10, 0.0),
    # B.8 Color: Opponent channels (Lum, RG, BY, Sat, Hue).
    "COLOR": (8.0, 16.0, 12.0, 8.0, 8.0),
    "COLOR_LEGACY": (12.0, 12.0, 12.0),
    "COLOR_CONTENT": (8.0, 8.0),
    "COLOR_LEGACY_CONTENT": (8.0, 8.0),
    # Motion: speed (20 bins), 3D direction (8 bins/axis)
    "MOTION": (20, 8, 8, 8),
    # Onset: age-since-onset normalized (20 bins)
    "ONSET": (20,),
    # Polarity: single categorical slot, 3 classes (OFF/balanced/ON).
    # Value is the draft's declared 3.0; no pair of classes can differ
    # by 3.0 (max diff 2), noted for the first POLARITY matching
    # calibration (register section 39).
    "POLARITY": (3.0,),
    # [Option (b)] Polarized orientation: the single-region ORIENTATION
    # slots (theta bin, elongation bin, 4x4 position class) plus the
    # 3-class polarity slot. Ranges mirror ORIENTATION_N_THETA = 24 and
    # ORIENTATION_N_ELONG = 4. NOTE: the adjacent "ORIENTATION" entry
    # still reads (12, 4, 0.0) and the same class also carries arity-6
    # GABOR_RESPONSE signatures -- pre-existing, recorded, not changed.
    "ORIENTATION_POL": (24, 4, 16.0, 3.0),
    # Position: Object-centric relative dx, relative dy, elevation
    # (16 bins per axis — fallback ladder refinement for layout resolution)
    "POSITION": (16, 16, 16),
    # RelativePosition: position relative to macro envelope centroid,
    # normalized by envelope extent. 16 bins per axis (fallback ladder:
    # 10 bins measured too coarse for layout discrimination). View-stable.
    "RELATIVE_POSITION": (16, 16),
    # Size: Ch 0 = radial scale (10 bins), Ch 1 = inertia aspect ratio / roundness (10 bins)
    "SIZE": (10, 10),
    # Texture: Ch 0 = surface roughness (10 bins), Ch 1 = directional isotropy (10 bins)
    "TEXTURE": (10, 10),
    # --- Binder classes (§2.3) ---
    "TOPOLOGY": (4, 20, 20, 20, 20),
    "DEGREE": (10,),
    "EDGE": (10,),
    "CONTACT": (10,),
    "CO_MOTION": (10,),
    "CONTRAST": (10,),
    "PERIODICITY": (20, 20),
    "PERSISTENCE": (10,),
    # --- Structure binder classes ---
    "STRUCTURE": (4, 4, 4, 4, 4, 4, 4, 4),
    # --- Order-2 binder classes (§2.3) ---
    "SEQUENCE": (20, 1, 1, 1, 1, 1, 1, 1, 1),
    # [v8] ORIENTATION signature (arity 3).
    # Slot 0 (angle): circular; range = N_theta / 2 = 12.
    # Slot 1 (elongation): N_ELONG - 1 = 4.
    # Slot 2 (anchor trunk): 0.0 categorical discriminator.
    # NOTE: hard-coded to mirror parameters.ORIENTATION_N_THETA and
    # parameters.ORIENTATION_N_ELONG; primitives.py does not import
    # parameters.py for layering reasons.
    "ORIENTATION": (12, 4, 0.0),
    # [v9] ORIENTATION_SEGMENT signature (arity 4).
    # Slot 0 (theta): circular; range = N_theta / 2 = 12.
    # Slot 1 (arc_len): ORIENTATION_SEGMENT_N_ARC - 1 = 7.
    # Slot 2 (chain_id): 0.0 categorical discriminator.
    # Slot 3 (chain_pos): ORIENTATION_SEGMENT_N_POS - 1 = 7.
    "ORIENTATION_SEGMENT": (12, 7, 0.0, 7),
    # [v32 CONTENT TIER] Position-invariant content classes. Registered
    # here (primitives) rather than parameters.py because parameters.py
    # does not import these dicts — primitives does not import parameters
    # for layering reasons, so the registries must live in primitives.
    # Content signatures are the anchor-stripped forms produced by
    # content.content_signature().
    # BOUNDARY_CONTENT: quantized[:2] of BOUNDARY/BOUNDARY_ANCHORED.
    "BOUNDARY_CONTENT": (10.0, 10.0),
    # ORIENTATION_CONTENT: (theta, e_b). Slot 0 circular; slot 1 elongation.
    "ORIENTATION_CONTENT": (12.0, 5.0),
    # ELONGATION_CONTENT: (e_b,) — arity 1.
    "ELONGATION_CONTENT": (5.0,),
    # --- [PHASE B] higher-lens classes (futurework3.md B.5–B.10) ---
    # Values mirror parameters.py for layering reasons (primitives does
    # not import parameters): ORIENTATION_N_THETA = 24,
    # COLOR_QUANT_BINS = 12.
    #
    # A CIRCULAR slot's range must be exactly N/2 (see the
    # CIRCULAR_CYCLE_LENGTHS note below): the wrap-around formula
    # min(d, N-d) caps that slot's contribution at N/2, so a larger
    # range would under-penalize opposite angles.
    #
    # [B6 FIX v2] BorderOwnership: PER-BOUNDARY-POINT signed direction FIELD.
    # Arity 4: (arc_pos_bin, signed_dir_bin, curvature_bin, strength_bin).
    # Slots 1 and 2 are signed linear ([-1,1] -> 0..7); no slot is circular.
    # One detection is emitted PER SAMPLED VERTEX, so a boundary is a field.
    "BORDER_OWNERSHIP": (8.0, 8.0, 8.0, 8.0),
    # [Stage 3] ContourFragment: Pasupathy & Connor object-centred polar
    # fragment. Arity 4: (kappa_bin 0..8, theta_bin 0..23 circular, r_bin 0..7,
    # tangent_vs_radial_bin 0..7). Slot 1 is the polar angle on a 24-cycle, so
    # its range is 24/2 = 12.0 under the N/2 convention (registered in
    # CIRCULAR_CYCLE_LENGTHS below).
    "CONTOUR_FRAGMENT": (9.0, 12.0, 8.0, 8.0),
    # [Stage 4] Shock-skeleton primitives. RIBBON is a medial shock-axis
    # segment: (order 1..4, aspect, taper, curvature, radius). ENSEMBLE is a
    # grouped shock skeleton: (n_order1, n_order2, n_order3, n_order4, genus,
    # symmetry, n_segments, aspect).
    "RIBBON": (4.0, 8.0, 8.0, 8.0, 8.0),
    "ENSEMBLE": (4.0, 4.0, 4.0, 4.0, 3.0, 8.0, 8.0, 8.0),
    # [B5] Contour: arity 27 (7 base slots + 20 curvature sequence).
    #   0 length_bin, 1 mean_theta_bin (circular on 24 -> range 12),
    #   2 circ_var_bin (dispersion), 3 closure_bin,
    #   4 total_abs_turn_bin, 5 corner_count_bin,
    #   6 chirality_bin (net signed turn / 2pi),
    #   7..26 canonical signed curvature sequence (CONTOUR_N_SEQ = 20 bins, range 8.0).
    "CONTOUR": (8.0, 12.0, 8.0, 2.0, 8.0, 8.0, 8.0) + (8.0,) * 20,
    # [Stage A2] CONTOUR_PEAK: a peak of the continuous V1/V2 integration field
    # (perception/contour_field.py), NOT a shock segment. Arity 8:
    #   0 theta_bin              circular, 12 bins over [0, pi)
    #   1 sf_bin                 graded, 5 bins
    #   2 signed_curvature_bin   graded, 8 bins, clipped to +-0.6
    #   3 bend_angle_bin         graded, 8 bins, clipped to 1.5
    #   4 phase_bin              circular, 4 bins
    #   5 end_stop_flag          binary
    #   6 integration_strength_bin graded, 8 bins, clipped to 20
    #   7 coherence_bin          graded, 8 bins, clipped to 0.15
    # Bin edges are fixed in perception/contour_peak.py and documented there.
    # Circular slots obey the N/2 convention: range[slot] == cycle/2, AND the
    # cycle equals the emitter's bin count (contour_peak.py: theta_idx %12,
    # phase_bin %4) so the wrap actually fires. A cycle larger than the bin
    # count never wraps and makes adjacent bins maximally far apart.
    #   slot 0: theta_bin, contour_peak.py theta_idx in [0,11] -> cycle 12, range 6
    #   slot 4: phase_bin, contour_peak.py phase_bin in [0,3] -> cycle 4, range 2
    "CONTOUR_PEAK": (6.0, 4.0, 7.0, 7.0, 2.0, 1.0, 7.0, 7.0),
    # [A5] PEAK_CHAIN, arity 6. A chain of CONTOUR_PEAK detections grouped by
    # perception/chain_formation.form_chains (proximity + axial orientation +
    # end-stop clearance). Slots:
    #   0 n_peaks_bin          graded, 0..15
    #   1 mean_theta_bin       circular over 12 (see N/2 note below)
    #   2 theta_var_bin        graded, 0..7
    #   3 closure_bin          binary
    #   4 curvature_summary_bin graded, 0..7
    #   5 length_bin           graded, 0..7
    # Bin edges are fixed in perception/lenses/chain_lens.py.
    "PEAK_CHAIN": (15.0, 6.0, 7.0, 1.0, 7.0, 7.0),
    "CONTOUR_CONTENT": (8.0, 2.0, 8.0, 8.0, 8.0),
    # B.7+B21 Junction, arity 19:
    #   0-7 as B7 (degree, ref_theta, topo, spread, gap_0..3)
    #   8  arm_support: mean #oriented segments per arm, capped (linear 8)
    #   9  arm_sharp_mean, 10 arm_sharp_min: doubled-angle resultant of the arm's
    #      segment tangents (8 bins). Only meaningful when arm_support > 1.
    #   11-18 pairwise arm-angle counts over (0, pi], 8 bins of 22.5 deg, capped at 7.
    "JUNCTION": (6.0, 12.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                 8.0, 8.0, 8.0,
                 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0),
    # Content: rotation-invariant. degree, topo, spread, arm_sharp_mean,
    # pair histogram. (arm_sharp_min and ref_theta are dropped: ref_theta is
    # already 0 for symmetric topologies, and min is a duplicate of mean.)
    "JUNCTION_CONTENT": (6.0, 7.0, 8.0, 8.0,
                         8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0),
    # B12 Surface: (shape, lum, hue, sat, tex, grad_mag, grad_side), arity 7.
    # Slot 2 (hue) is circular on 16 -> range 8.0.
    "SURFACE_PATCH": (8.0,) * 7,
    "SURFACE_PATCH_CONTENT": (8.0,) * 6,
    # B.8/B13 CurveComplex: arity 24.
    # 0 n_contours, 1 n_junctions, 2 extent, 3 closure, 4-11 signed spatial bins,
    # 12-23 V4 template population.
    "CURVE_COMPLEX": (8.0,) * 24,
    "CURVE_COMPLEX_CONTENT": (8.0,) * 22,
    # B14+B22 RegionShape: arity 21 (was 18).
    # 0-15 curvature histogram, 16 area, 17 elongation,
    # 18 sym_score (8 bins), 19 sym_axis (circular 12 -> 6.0),
    # 20 sym_fold (k-1, 0..5).
    "REGION_SHAPE": (8.0,) * 18 + (8.0, 6.0, 6.0),
    "REGION_SHAPE_CONTENT": (8.0,) * 18,
    # B.9/B15 Part: arity 23.
    # 0 n_complexes, 1-8 spatial layout, 9-20 V4 templates (12), 21 hue (circ 16 -> 8.0), 22 sat.
    "PART": (8.0,) * 21 + (8.0, 8.0),
    "PART_CONTENT": (8.0,) * 15,
    # B.10/B16 ObjectShape: arity 33 (was 37/17).
    # 0-7 boundary sym, 8-15 part sym, 16 n_parts, 17-24 ring presence, 25-32 ring template.
    # B17 Illusory Contour: V2 modal completion. Arity 3.
    # 0: orientation_bin (circular 24 -> 12.0)
    # 1: length_bin (linear 8, gap distance normalized)
    # 2: strength_bin (linear 8, completion confidence)
    "ILLUSORY_CONTOUR": (12.0, 8.0, 8.0),
    "ILLUSORY_CONTOUR_CONTENT": (12.0,),
    # [D1] End-Stopped (Hypercomplex): line terminations and short bars/dots.
    # (theta_bin, sf_bin, ends_bin, strength_bin).
    "ENDSTOPPED": (12.0, 8.0, 3.0, 8.0),
    "ENDSTOPPED_CONTENT": (12.0, 8.0),
    # [FIGURE-GROUND / singular points] CORNER: a boundary vertex where the
    # tangent turns sharply, derived from the CONTINUOUS FIELD's chain relations
    # (perception/lenses/corner_lens.py reads PEAK_CHAIN member order and the
    # member peak positions), NOT from the dead shock graph. One typed node per
    # local turn-angle extremum. Arity 4:
    #   0 turn_mag_bin    graded, |signed turn| over [0, pi] -> 0..7
    #   1 orient_bin      circular, corner-bisector orientation over [0, pi),
    #                     cycle 12 -> range 6.0 (N/2 convention)
    #   2 prom_bin        graded, local prominence (|turn| - max neighbour
    #                     |turn|) over [0, pi/2] -> 0..7
    #   3 scale_bin       graded, corner arm length / (8 * extraction_scale)
    "CORNER": (8.0, 6.0, 8.0, 8.0),
    "CORNER_CONTENT": (8.0, 6.0, 8.0),
    # [B3a/b] ORIENTATION_GABOR: (theta, sf, phase, contrast, elong, simple_flag, asym)
    # [Pass 17.2] phase slot (slot 2) holds a 0..23 bin (ORIENTATION_PHASE_BINS
    # = 24): range must be N/2 = 12.0 under the circular convention, and slot 2
    # must be listed in CIRCULAR_CYCLE_LENGTHS, or cross-image simple-cell
    # similarity collapses to exp(-400) ~ 0.
    "ORIENTATION_GABOR": (6.0, 4.0, 12.0, 15.0, 4.0, 1.0, 8.0),
    "ORIENTATION_GABOR_CONTENT": (6.0, 4.0),
    # [B11] Motion Population & Content. Arity 16, matching the emitter
    # (lenses/base.py motion emitter). Slots: 0-7 dir_bins (8-way energy
    # fraction, linear 8), 8 coherence, 9 speed, 10 transparency,
    # 11 dir1 / 12 dir2 (circular 8 -> range 4.0), 13 div / 14 curl
    # (signed 8), 15 shear (8). The registry previously carried only the
    # first 13 slots, so every emitted 16-slot motion signature failed the
    # arity length guard and silently scored 0.0.
    "MOTION_POPULATION": (8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                          8.0, 4.0, 4.0, 8.0, 8.0, 8.0),
    "MOTION_CONTENT": (4.0, 4.0, 8.0),
    # [B20] Texture-Defined Boundary
    "TEXTURE_BOUNDARY": (8.0, 12.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0),
    "TEXTURE_BOUNDARY_CONTENT": (12.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0),
    # [aIT Holistic Shape] Arity 6 — global boundary-shape summary emitted by
    # HolisticShapeLens, from the same polyline_shape_descriptor as RegionShape:
    # 0 area_bin (8.0), 1 elong_bin (8.0), 2 sym_score_bin (8.0),
    # 3 sym_axis (circular 12 -> 6.0, 12 = undefined),
    # 4 fold (0..5), 5 curvature-histogram summary bin (8.0).
    "AIT_HOLISTIC_SHAPE": (8.0, 8.0, 8.0, 6.0, 8.0, 8.0),
    "AIT_HOLISTIC_SHAPE_CONTENT": (8.0, 8.0, 8.0, 6.0, 8.0, 8.0),
}

CIRCULAR_CYCLE_LENGTHS: Dict[str, Dict[int, int]] = {
    "ORIENTATION": {0: 24},
    "ORIENTATION_SEGMENT": {0: 24},
    # [v32 CONTENT TIER] theta slot of the anchor-stripped Orientation.
    "ORIENTATION_CONTENT": {0: 24},
    # [PHASE B] futurework3.md B.5–B.7. CONTOUR/JUNCTION cycle is
    # ORIENTATION_N_THETA (24); mirrored here for the same layering
    # reason as the ORIENTATION entry above.
    "CONTOUR": {1: 24},
    # [Stage 3] ContourFragment slot 1 is the polar angle (24-cycle -> range 12).
    "CONTOUR_FRAGMENT": {1: 24},
    # [Stage A2 / A5c] CONTOUR_PEAK: slot 0 runs 0..11 (contour_peak.py
    # theta_idx %12) -> cycle 12, range 6; slot 4 runs 0..3 (phase_bin %4)
    # -> cycle 4, range 2. Cycle must equal the emitted bin count, or the
    # wrap never fires and adjacent bins score maximally far apart.
    "CONTOUR_PEAK": {0: 12, 4: 4},
    # [A5] PEAK_CHAIN: slot 1 is the circular-mean member-theta bin. 12
    # bins over a 24-cycle odometer: the range is 6.0 per the N/2
    # convention, and the emitted bin index is taken mod 12.
    "PEAK_CHAIN": {1: 12},
    "JUNCTION": {1: 24},
    "ENDSTOPPED": {0: 24},
    "ENDSTOPPED_CONTENT": {0: 24},
    # [singular points] CORNER slot 1 is the corner-bisector orientation over a
    # 12-cycle (range 6.0 under the N/2 convention).
    "CORNER": {1: 12},
    "CORNER_CONTENT": {1: 12},
    "ORIENTATION_GABOR": {0: 12, 2: 24},
    "ORIENTATION_GABOR_CONTENT": {0: 12},
    "MOTION_POPULATION": {11: 8, 12: 8},
    "MOTION_CONTENT": {0: 8, 1: 8},
    "TEXTURE_BOUNDARY": {1: 24},
    "TEXTURE_BOUNDARY_CONTENT": {0: 24},
    "COLOR": {4: 16},
    "COLOR_CONTENT": {0: 16},
    "COLOR_LEGACY_CONTENT": {0: 16},
    "SURFACE_PATCH": {2: 16},
    "SURFACE_PATCH_CONTENT": {2: 16},
    "REGION_SHAPE": {19: 12},
    "PART": {21: 16},
    "PART_CONTENT": {13: 16},
    "ILLUSORY_CONTOUR": {0: 24},
    "ILLUSORY_CONTOUR_CONTENT": {0: 24},
    # [aIT Holistic Shape] slot 3 sym_axis is a 12-cycle (12 = undefined),
    # same convention as REGION_SHAPE slot 19.
    "AIT_HOLISTIC_SHAPE": {3: 12},
    "AIT_HOLISTIC_SHAPE_CONTENT": {3: 12},
}

def _verify_circular_ranges(ranges: dict, cycles: dict, enforce_classes: set) -> None:
    """Verifies the N/2 circular range convention.
    Raises RuntimeError for classes in `enforce_classes`.
    Issues a warning for other classes that violate the convention.
    """
    for _cls, _cycles in cycles.items():
        _ranges = ranges.get(_cls)
        if _ranges is None:
            if _cls in enforce_classes:
                raise RuntimeError(
                    f"CIRCULAR_CYCLE_LENGTHS lists {_cls} but it is missing "
                    f"from CONSTITUTIONAL_RANGES"
                )
            continue

        for _slot_idx, _cycle_len in _cycles.items():
            if _slot_idx >= len(_ranges):
                if _cls in enforce_classes:
                    raise RuntimeError(
                        f"CIRCULAR_CYCLE_LENGTHS[{_cls}][{_slot_idx}] out of bounds "
                        f"for arity {len(_ranges)}"
                    )
                continue

            _expected_range = _cycle_len / 2.0
            _actual_range = _ranges[_slot_idx]
            if _actual_range != _expected_range:
                msg = (
                    f"Circular range convention violated: {_cls} slot {_slot_idx} "
                    f"has cycle {_cycle_len} but range {_actual_range}. "
                    f"Expected N/2 = {_expected_range}."
                )
                if _cls in enforce_classes:
                    raise RuntimeError(msg)
                else:
                    warnings.warn(f"{msg} (Legacy class, not enforced at boot)")

_ENFORCE_N_HALF_CONVENTION = {
    "JUNCTION", "COLOR", "COLOR_CONTENT", "COLOR_LEGACY_CONTENT",
    "SURFACE_PATCH", "SURFACE_PATCH_CONTENT", "REGION_SHAPE",
    "PART", "PART_CONTENT", "ILLUSORY_CONTOUR", "ILLUSORY_CONTOUR_CONTENT",
    "TEXTURE_BOUNDARY", "TEXTURE_BOUNDARY_CONTENT",
    # [Pass 17.2] simple-cell phase slot is circular 24 -> range must be 12.
    "ORIENTATION_GABOR", "ORIENTATION_GABOR_CONTENT",
}

_verify_circular_ranges(
    CONSTITUTIONAL_RANGES, CIRCULAR_CYCLE_LENGTHS, _ENFORCE_N_HALF_CONVENTION
)

# --- [POPULATION CODING] tuning-curve axes -------------------------------
# The graded companion of each class's bins. One PopAxis per graded quantity;
# a signature's `population` is a tuple of L1-normalised activity vectors, one
# per axis, in the order listed here. Circular axes use n_bins == the class's
# CIRCULAR_CYCLE_LENGTHS entry for that quantity (checked below). This is the
# "full vector on the detection" the representation thesis requires; the bins
# remain as lookup keys only.
POPULATION_AXES: Dict[str, Tuple[PopAxis, ...]] = {
    # CONTOUR_PEAK: a continuous field peak. Graded quantities match the bin
    # edges fixed in perception/contour_peak.py (sigma default 1 bin = the
    # front-end resolution; the orientation channel spacing GABOR_N_THETA=12
    # over [0, pi) is exactly 1 bin, so the width is the filter bandwidth).
    "CONTOUR_PEAK": (
        PopAxis("theta", 12, circular=True),
        PopAxis("sf", 5, circular=False),
        PopAxis("curvature", 8, circular=False),
        PopAxis("bend", 8, circular=False),
        PopAxis("phase", 4, circular=True),
        PopAxis("strength", 8, circular=False),
        PopAxis("coherence", 8, circular=False),
    ),
    # PEAK_CHAIN: an ensemble. The orientation axis holds the DISTRIBUTION of
    # member orientations (a break-stable mixture), not the circular mean, so a
    # chain is described by its spread of evidence and not one collapsed number.
    "PEAK_CHAIN": (
        PopAxis("theta_dist", 12, circular=True),
        PopAxis("curvature", 8, circular=False),
        PopAxis("length", 8, circular=False),
    ),
}


# --- [POPULATION CODING] structural classes ------------------------------
# The graded companion of the structural node signatures. Rather than
# hand-listing axes (risking an axis that silently DROPS a slot from the
# overlap), the layout is DERIVED from the constitutional registry: one axis
# per quantized slot, n_bins = the emitted bin count (the registered cycle for
# a circular slot, the registered range otherwise). This guarantees the
# population covers EVERY slot -- the overlap then loses nothing relative to
# the bins path -- and satisfies the circular/cycle agreement the verifier
# enforces. A non-graded slot (binary flag, count) still gets a tuning axis; a
# range-1 axis carries no variance and is therefore NEUTRAL (axis_similarity
# -> 0, weight still counted), never a silent drop.
_POPULATION_SLOT_CLASSES: Tuple[str, ...] = (
    "CONTOUR",
    "JUNCTION",
    "BORDER_OWNERSHIP",
    "ENDSTOPPED",
    "POLARITY",
    "SURFACE_PATCH",
    "CONTOUR_FRAGMENT",
    "TEXTURE_BOUNDARY",
    "CORNER",
)


def _axes_from_registry(cls: str) -> Optional[Tuple[PopAxis, ...]]:
    rng = CONSTITUTIONAL_RANGES.get(cls)
    if not rng:
        return None
    cycles = CIRCULAR_CYCLE_LENGTHS.get(cls, {})
    axes = []
    for i, r in enumerate(rng):
        if i in cycles:
            axes.append(PopAxis(f"slot{i}", int(cycles[i]), circular=True))
        else:
            axes.append(PopAxis(f"slot{i}", max(1, int(round(float(r))))))
    return tuple(axes)


for _c in _POPULATION_SLOT_CLASSES:
    _ax = _axes_from_registry(_c)
    if _ax is not None and _c not in POPULATION_AXES:
        POPULATION_AXES[_c] = _ax


def population_for_bins(class_name: str,
                        bins: Sequence[float]) -> Tuple[Tuple[float, ...], ...]:
    """The graded population of one node from its quantized bins: one soft
    tuning bump per registered axis, centred at the bin index. Returns () when
    the class has no registered population or the arity disagrees, so the
    signature falls back to the bins path in similarity (never a silent drop)."""
    axes = POPULATION_AXES.get(class_name)
    if axes is None or len(axes) != len(bins):
        return ()
    from core.population import soft_bump_bins
    return tuple(soft_bump_bins(float(b), ax) for ax, b in zip(axes, bins))


def get_population_axes(class_name: str) -> Optional[Tuple[PopAxis, ...]]:
    """The declared population layout for a class, or None (bins only)."""
    return POPULATION_AXES.get(class_name)


def _verify_population_axes(ranges: dict, cycles: dict, pop: dict) -> None:
    """Each circular axis's n_bins must equal the class's registered cycle for
    the matching quantity; each axis n_bins must equal the class range length
    where the axis maps 1:1 onto slots is NOT assumed (populations are their own
    layout), so only the circular/cycle agreement is enforced."""
    for _cls, _axes in pop.items():
        _cycles = cycles.get(_cls, {})
        _names_cycle = {}
        for _slot, _cyc in _cycles.items():
            _names_cycle[_cyc] = _names_cycle.get(_cyc, 0) + 1
        for _ax in _axes:
            if _ax.circular and _ax.n_bins not in _names_cycle:
                # A circular population axis must correspond to a registered
                # circular slot with the same bin count, or the wrap convention
                # the similarity uses would diverge from the class's.
                import warnings as _w
                _w.warn(
                    f"POPULATION_AXES[{_cls}].{_ax.name}: circular axis with "
                    f"n_bins={_ax.n_bins} has no matching CIRCULAR_CYCLE_LENGTHS "
                    f"entry {_cycles}")


_verify_population_axes(CONSTITUTIONAL_RANGES, CIRCULAR_CYCLE_LENGTHS,
                        POPULATION_AXES)

def b1_class_name(nt: int, nsf: int) -> str:
    """Canonical identifier for B1 hypercolumn population schema."""
    return f"ORIENTATION_RICH_{nt}_{nsf}"

def b4_class_name(nt: int, nsf: int) -> str:
    """Canonical identifier for B4 second-order texture schema."""
    return f"TEXTURE_RICH_{nt}_{nsf}"

def compute_b1_arity(nt: int, nsf: int) -> int:
    """Closed-form arity formula for B1 hypercolumn population signature:
    3*(nt*nsf) + 14*nt + 85 = 433 slots for a 12x5 bank."""
    return 3 * (nt * nsf) + 14 * nt + 85

def compute_b4_arity(nt: int, nsf: int) -> int:
    """Closed-form arity formula for B4 second-order texture signature:
    4*(nt*nsf) + 4 globals = 244 slots for a 12x5 bank."""
    return 4 * (nt * nsf) + 4

_GABOR_NT = getattr(P, "GABOR_N_THETA", 12)

_GABOR_NSF = getattr(P, "GABOR_N_SF", 5)

_B1_DERIVED_ARITY = compute_b1_arity(_GABOR_NT, _GABOR_NSF)

_B4_DERIVED_ARITY = compute_b4_arity(_GABOR_NT, _GABOR_NSF)

_ORIENTATION_MARGINALS = getattr(P, "ORIENTATION_RICH_MARGINAL_BINS", 60)

CONSTITUTIONAL_RANGES["ORIENTATION_RICH_CONTENT"] = (1.0,) * _ORIENTATION_MARGINALS

CONSTITUTIONAL_RANGES["TEXTURE_RICH_CONTENT"] = (1.0, 1.0, 1.0, 1.0)

CONSTITUTIONAL_RANGES["ORIENTATION_RICH"] = (1.0,) * _B1_DERIVED_ARITY

CONSTITUTIONAL_RANGES["TEXTURE_RICH"] = (1.0,) * _B4_DERIVED_ARITY

CONSTITUTIONAL_RANGES[b1_class_name(_GABOR_NT, _GABOR_NSF)] = (1.0,) * _B1_DERIVED_ARITY

CONSTITUTIONAL_RANGES[b4_class_name(_GABOR_NT, _GABOR_NSF)] = (1.0,) * _B4_DERIVED_ARITY

assert len(CONSTITUTIONAL_RANGES["ORIENTATION_RICH"]) == 433, f"Expected 433, got {len(CONSTITUTIONAL_RANGES['ORIENTATION_RICH'])}"

assert len(CONSTITUTIONAL_RANGES["TEXTURE_RICH"]) == 244, f"Expected 244, got {len(CONSTITUTIONAL_RANGES['TEXTURE_RICH'])}"

def get_similarity_ranges(class_name: str) -> tuple[int, ...] | None:
    """Return the constitutional R_j tuple for a CANONICAL signature class,
    or None if unregistered (which forces sim = 0.0)."""
    return CONSTITUTIONAL_RANGES.get(class_name)

SLOT_WEIGHTS: Dict[str, Tuple[float, ...]] = {}

SLOT_WEIGHTS_BY_CONTEXT: Dict[Tuple[str, Any], Tuple[float, ...]] = {}

SLOT_WEIGHT_PRIORS: Dict[str, Tuple[float, ...]] = {}

def resolve_slot_weights(
    class_name: str,
    n_slots: int,
    context: Any = None,
) -> Tuple[float, ...]:
    """context-specific -> class prior -> legacy SLOT_WEIGHTS -> uniform.

    [B23] The arity check RAISES rather than warns. A length mismatch here
    silently misaligns every weight against the wrong slot, degrading
    similarity in a way no harness will attribute to this function.
    """
    candidates = []
    if context is not None:
        candidates.append(SLOT_WEIGHTS_BY_CONTEXT.get((class_name, context)))
    candidates.append(SLOT_WEIGHT_PRIORS.get(class_name))
    candidates.append(globals().get("SLOT_WEIGHTS", {}).get(class_name))  # legacy registry
    for w in candidates:
        if w is not None:
            if len(w) != n_slots:
                raise ValueError(
                    f"slot weights for {class_name}: {len(w)} != {n_slots}"
                )
            return w
    return (1.0,) * n_slots

def get_slot_weights(
    class_name: str,
    n_slots: Optional[int] = None,
    context: Any = None,
) -> Tuple[float, ...]:
    """Back-compatible entry point.

    [B23] n_slots is OPTIONAL here, not positional-required. Existing call
    sites pass only the class name; making it mandatory would break every one
    of them at import.
    """
    if n_slots is None:
        rng = CONSTITUTIONAL_RANGES.get(class_name)
        if not rng:
            raise KeyError(
                f"unknown class {class_name!r} and no n_slots given"
            )
        n_slots = len(rng)
    return resolve_slot_weights(class_name, n_slots, context)

# --- moved from binder.py (Move 3c): edge-layer map + the
# self-binder / order-2 registry constants --------------------------
def edge_type_layer(e_type: EdgeType) -> int:
    """§3.3 layer contribution. Order-2 edge types raise layer by one."""
    return 1 if e_type in ORDER_2_EDGE_TYPES else 0

# --- moved from binder.py (Move 3c): registry constants ------------
SELF_BINDER_TYPES: FrozenSet[EdgeType] = frozenset()

# §2.3: order-2 edge types. Empty in v31 — structural similarity was removed
# from the binder (a comparison emits no relation; see structural_similarity.py)
# and the set is retained only as the single extension point for a future
# order-2 binder. edge_type_layer() therefore contributes 0 for every type.
ORDER_2_EDGE_TYPES: FrozenSet[EdgeType] = frozenset()

