"""core/schema.py -- edge schemas, edge params, polymorphic matching.

MOVE ONLY: statements extracted verbatim from primitives.py by
_reorg_move2_schema.py, preserving their order (module-level registration order
is part of the persisted-memory contract). primitives.py re-exports from here.

Zero project imports except core.hashing and parameters (a leaf module).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace
from dataclasses import replace as _dc_replace  # noqa: F401  (moved import alias)
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

from core.hashing import clamp, hash_128, hash_64, serialize_bytes, serialize_tuple

import parameters as P
from parameters import EPSILON

CONSTITUTIONAL_EDGE_RANGES: Dict[str, Tuple[float, ...]] = {
    # (gap_len, perp_offset, collinearity_error, signed_curv_delta, src_arc, dst_arc)
    # [Pass 21] (gap, perp, cocircular_asym, signed_turn, tail_role,
    #  head_role, polarity_relation). Slot 3 is circular on cycle 8,
    #  so its range is 4.0 under the N_HALF convention -- NOT 8.0.
    "EDGE_ORIENTATION_CONTINUITY": (8.0, 8.0, 8.0, 4.0, 3.0, 3.0, 3.0),
    # (normal_angle, overlap_len, curv_src, curv_dst)
    "EDGE_CONTACT": (12.0, 9.0, 9.0),
    # (color_dist, texture_dist, orient_delta, contrast_step)
    "EDGE_SURFACE_GROUPING": (9.0, 9.0, 9.0, 9.0),
    # (signed_turn, chord_len, curv_rate)
    "EDGE_CURVATURE": (8.0, 8.0, 8.0),
    # (phase_diff, polarity_congruency, salience_ratio)
    # [Pass 21] (scale_ratio, salience_ratio)
    "EDGE_CO_LOCATED": (9.0, 9.0),
    # (angle_delta, bisector_theta, spatial_dist)
    "EDGE_ORIENTATION_BEND": (8.0, 12.0, 8.0),
    "EDGE_ORIENTATION_JUNCTION": (8.0, 12.0, 8.0),
    # [Pass 12] Relational edge types E7–E11
    "EDGE_PARALLEL": (9.0, 9.0, 9.0),
    "EDGE_SYMMETRY": (8.0, 12.0, 8.0),
    "EDGE_PART_SEQUENCE": (8.0, 8.0, 12.0),
    "EDGE_PART_AT_ANGLE": (12.0, 8.0, 8.0),
    "EDGE_BORDER_OWNERSHIP": (8.0, 12.0, 8.0),
    # [Pass 15] High-tier binder edge parameter schemas
    "EDGE_CONTOUR_CONTINUITY": (8.0, 4.0, 8.0),
    "EDGE_JUNCTION_CONTOUR": (8.0, 12.0, 7.0),
    # [Pass 22] CO_MOTION: (mean_speed, dir_theta, speed_diff, dir_diff)
    "EDGE_CO_MOTION": (8.0, 12.0, 8.0, 8.0),
    "EDGE_CONTOUR_ORIENTATION": (8.0,),
    "EDGE_JUNCTION_ORIENTATION": (8.0,),
    # [Pass 18 / C4+C6] (rel_dist, polar_angle, log_scale_ratio). The polar
    # slot is 24 bins on a circle, hence range 12.0 under N_HALF; registered
    # in CIRCULAR_EDGE_CYCLE_LENGTHS below.
    "EDGE_CURVE_CONTAINMENT": (8.0, 12.0, 8.0),
    "EDGE_PART_CONTAINMENT": (8.0, 12.0, 8.0),
    "EDGE_PART_CONTENT": (8.0,),
    "EDGE_OBJECT_PART": (8.0, 12.0, 8.0),
    # [Stage 3] Occlusion: (rank_gap | depth_step, stack_depth | 8 = none, source
    # 0 = T-junction, 1 = depth). Was arity 2 (dz_bin, conf_bin); the depth-DAG
    # path now carries the longest-path rank gap and the weakly-connected stack
    # depth. No circular slot, so no CIRCULAR_EDGE_CYCLE_LENGTHS entry.
    "EDGE_OCCLUSION": (8.0, 9.0, 2.0),
    # [Pass 16] Relational edge types
    "EDGE_CURVATURE_SIGN_CONTINUITY": (8.0, 8.0),
    "EDGE_COCIRCULAR": (9.0, 9.0),
    "EDGE_AMODAL_LINK": (8.0, 8.0),
    "EDGE_SHARED_BOUNDARY": (8.0, 8.0),
    # [Pass 23] values 42-51. Circular slots satisfy range == cycle/2;
    # _verify_edge_circular_ranges enforces this at import.
    "EDGE_CONCENTRICITY": (8.0, 8.0, 8.0, 8.0),
    "EDGE_TEXTURE_BOUNDARY": (8.0, 8.0, 8.0, 8.0),
    "EDGE_SURFACE_FOLD": (12.0, 6.0, 3.0, 8.0),
    "EDGE_RADIALITY": (12.0, 8.0, 8.0, 12.0),
    "EDGE_SPIRALITY": (6.0, 8.0, 3.0, 8.0),
    "EDGE_COLOR_OPPONENT_BORDER": (12.0, 8.0, 8.0, 8.0),
    "EDGE_MOTION_OPPOSITION": (12.0, 8.0, 12.0, 8.0),
    "EDGE_CROSS_SCALE_IDENTITY": (8.0, 12.0, 8.0, 8.0),
    "EDGE_BINOCULAR_DISPARITY": (8.0, 8.0, 4.0, 8.0),
    # [Pass 23 audit] slot 3 is the AXIAL (mod-pi) parent orientation, so it
    # takes a 12-cycle and range 6.0 -- not a 24-cycle. The draft declared a
    # 24-cycle here, which made adjacent axial values score as maximally
    # different under the min(d, cycle-d) wrap.
    "EDGE_LINE_TERMINATION": (5.0, 8.0, 8.0, 6.0),
}

CIRCULAR_EDGE_CYCLE_LENGTHS: Dict[str, Dict[int, int]] = {
    "EDGE_CONTACT": {0: 24},
    "EDGE_CONTOUR_CONTINUITY": {1: 8},
    "EDGE_JUNCTION_CONTOUR": {1: 24},
    "EDGE_CO_MOTION": {1: 24},
    "EDGE_ORIENTATION_CONTINUITY": {3: 8},
    "EDGE_ORIENTATION_BEND": {1: 24},
    "EDGE_ORIENTATION_JUNCTION": {1: 24},
    "EDGE_SYMMETRY": {1: 24},
    "EDGE_PART_SEQUENCE": {2: 24},
    "EDGE_PART_AT_ANGLE": {0: 24},
    "EDGE_BORDER_OWNERSHIP": {1: 24},
    # [Pass 18 / C4+C6] index 1 is the polar_angle slot of the containment
    # edges: 24 bins on a circle, range 12.0 under the N_HALF convention.
    "EDGE_CURVE_CONTAINMENT": {1: 24},
    "EDGE_PART_CONTAINMENT": {1: 24},
    "EDGE_OBJECT_PART": {1: 24},
    # [Pass 23] slot 1 is the AXIAL (mod-pi) fold axis -> 12-cycle.
    "EDGE_SURFACE_FOLD": {1: 12},
    "EDGE_RADIALITY": {0: 24},
    "EDGE_SPIRALITY": {0: 12},
    "EDGE_COLOR_OPPONENT_BORDER": {0: 24},
    "EDGE_MOTION_OPPOSITION": {0: 24, 2: 24},
    # [Pass 23 audit] axial (mod-pi) parent orientation -> 12-cycle.
    "EDGE_LINE_TERMINATION": {3: 12},
    # [Pass 23] slot 1 is phase_alignment: 24 bins on a circle (phase_bin
    # is computed mod 24 in _run_cross_scale_identity), range 12.0 under
    # the N_HALF convention — satisfies _verify_edge_circular_ranges.
    "EDGE_CROSS_SCALE_IDENTITY": {1: 24},
}

def get_edge_similarity_ranges(class_name: str) -> Tuple[float, ...] | None:
    """Return constitutional ranges for edge parameter schemas, or None if unregistered."""
    return CONSTITUTIONAL_EDGE_RANGES.get(class_name)

UNKNOWN: float = -1.0

EDGE_IDENTITY_SLOTS: Dict[str, Tuple[int, ...]] = {
    # (gap, perp, cocircular_asym, signed_turn, tail_role, head_role, polarity_relation)
    # Slots 4 & 5: bipole tail/head roles (0=terminal, 1=weak, 2=strong interior)
    # Slot 6: polarity relation (0=flipped, 1=none, 2=same)
    "EDGE_ORIENTATION_CONTINUITY": (4, 5, 6),
    # (dist, incidence, topo) -> Slot 2: topology class (0..6: L, T, Y, X, Star, etc.)
    "EDGE_JUNCTION_CONTOUR": (2,),
    # (dihedral, fold_axis, convexity, sharpness) -> Slot 2: convexity (0=concave, 1=flat, 2=convex)
    "EDGE_SURFACE_FOLD": (2,),
    # (pitch, consistency, chirality, expansion) -> Slot 2: chirality (0=cw, 1=ambiguous, 2=ccw)
    "EDGE_SPIRALITY": (2,),
    # (junction, sharpness, curvature, parent) -> Slot 0: junction type (0=end, 1=corner, 2=collinear, 3=T, 4=Y)
    "EDGE_LINE_TERMINATION": (0,),
    # (disp, grad, cell_type, conf) -> Slot 2: near/far arrangement (0..3)
    "EDGE_BINOCULAR_DISPARITY": (2,),
    # (dist, congruence) -> Slot 1: congruence (0=congruent, 7=opposite)
    "EDGE_SHARED_BOUNDARY": (1,),
}

EDGE_REVERSAL: Dict[str, Tuple[Tuple[Tuple[int, int], ...], Tuple[int, ...]]] = {
    # (gap, perp, cocircular_asym, signed_turn, tail_role, head_role, polarity_relation)
    # Slots 4 & 5 swap roles. Slot 3 (signed turn on cycle 8) reflects: (8 - b) % 8.
    "EDGE_ORIENTATION_CONTINUITY": (((4, 5),), (3,)),
    # Slot 0 is normal_angle on cycle 24: reversing endpoints rotates normal by 180 deg (+12 bins).
    "EDGE_CONTACT": (((),), (0,)),
    # Slot 1 is signed turn on cycle 8: reflects across center.
    "EDGE_CONTOUR_CONTINUITY": (((),), (1,)),
    # Slot 0 is signed turn on cycle 8: reflects across center.
    "EDGE_CURVATURE": (((),), (0,)),
}

@dataclass(frozen=True)
class Slot:
    """Single relational parameter slot in an EdgeParams schema.

    domain:
        "linear"      — metric distance, ratio, or intensity; Gaussian similarity, no wrap.
        "circular"    — angular; wraps at n_bins (axial mod pi or full 2pi).
        "ordinal"     — discrete ordered steps.
        "categorical" — unordered discrete classes (exact match only).

    reversal: transformation applied when edge endpoints are swapped:
        "same"             — symmetric slot; unchanged.
        "reflect_linear"   — reflects across linear range: (n_bins - 1.0) - v.
        "reflect_circular" — reflects across circular cycle: (cycle - v) % cycle.
        ("swap", target)   — swaps value with slot at index `target`.
        ("rotate", offset) — adds `offset` bins modulo cycle length.

    is_identity: True -> Non-Accidental Property (NAP) hard gate. Differences >= 0.5 yield similarity 0.0.
    """
    tag: str
    domain: Literal["linear", "circular", "ordinal", "categorical"]
    n_bins: float
    reversal: Union[
        Literal["same", "reflect_linear", "reflect_circular"],
        Tuple[Literal["swap"], int],
        Tuple[Literal["rotate"], int],
    ]
    is_identity: bool = False

EDGE_SCHEMAS: Dict[str, Tuple[Slot, ...]] = {
    # 7 slots: dist, perp, cocirc_asym, turn, tail_role, head_role, polarity_relation
    # [Stage 2 / Note 3] Slot 0 renamed gap -> dist so the along-axis
    # bipole gap shares its tag with CC's endpoint distance. Intentional
    # semantic approximation: OC's slot 0 is the along-axis gap between
    # bipole anchors; CC's slot 0 is endpoint distance. Both bin a
    # physical separation to 0..7 with similar normalization, so the
    # cross-schema comparison is approximately meaningful (damped by the
    # cross-family prior). Positional matching by binder.py is unaffected
    # (it emits by position, never by tag).
    "EDGE_ORIENTATION_CONTINUITY": (
        Slot("dist",            "linear",      8.0, "same"),
        Slot("perp",            "linear",      8.0, "same"),
        Slot("cocirc_asym",     "linear",      8.0, "same"),
        Slot("turn",            "circular",    8.0, "reflect_circular"),
        Slot("tail_role",       "categorical", 3.0, ("swap", 5), is_identity=True),
        Slot("head_role",       "categorical", 3.0, ("swap", 4), is_identity=True),
        Slot("polarity_rel",    "ordinal",     3.0, "same",      is_identity=True),
    ),
    # 3 slots: normal_angle (24-cycle), overlap_len, depth_step
    "EDGE_CONTACT": (
        Slot("normal_angle",    "circular",   24.0, ("rotate", 12)),
        Slot("overlap_len",     "linear",      9.0, "same"),
        Slot("depth_step",      "ordinal",     9.0, "reflect_linear"),
    ),
    # 4 slots: color_dist, tex_dist, contrast_step, spatial_dist
    "EDGE_SURFACE_GROUPING": (
        Slot("color_dist",      "linear",      9.0, "same"),
        Slot("tex_dist",        "linear",      9.0, "same"),
        Slot("contrast_step",   "linear",      9.0, "same"),
        Slot("spatial_dist",    "linear",      9.0, "same"),
    ),
    # 3 slots: signed_turn_bin (linear 0..7), chord_len_bin, curv_rate_bin
    "EDGE_CURVATURE": (
        Slot("signed_turn",     "linear",      8.0, "reflect_linear"),
        Slot("chord_len",       "linear",      8.0, "same"),
        Slot("curv_rate",       "linear",      8.0, "same"),
    ),
    # 2 slots: scale_bin, salience_bin
    "EDGE_CO_LOCATED": (
        Slot("scale_ratio",     "linear",      9.0, "same"),
        Slot("salience_ratio",  "linear",      9.0, "same"),
    ),
    # 3 slots: angle_delta_bin, bisector_bin (24-cycle), spatial_dist_bin
    "EDGE_ORIENTATION_BEND": (
        Slot("angle_delta",     "linear",      8.0, "same"),
        Slot("bisector_theta",  "circular",   24.0, "same"),
        Slot("spatial_dist",    "linear",      8.0, "same"),
    ),
    "EDGE_ORIENTATION_JUNCTION": (
        Slot("angle_delta",     "linear",      8.0, "same"),
        Slot("bisector_theta",  "circular",   24.0, "same"),
        Slot("spatial_dist",    "linear",      8.0, "same"),
    ),
    # 3 slots: perp_bin, stag_bin, conv_bin
    "EDGE_PARALLEL": (
        Slot("perp_dist",       "linear",      9.0, "same"),
        Slot("stagger_dist",    "linear",      9.0, "reflect_linear"),
        Slot("convergence",     "linear",      9.0, "reflect_linear"),
    ),
    # 3 slots: sym_bin, axis_bin (24-cycle), dist_bin
    "EDGE_SYMMETRY": (
        Slot("sym_score",       "linear",      8.0, "same"),
        Slot("axis_angle",      "circular",   24.0, "same"),
        Slot("dist",            "linear",      8.0, "same"),
    ),
    # 3 slots: ang_diff_bin (linear 8.0), bearing_bin (circular 24 -> 12.0), curv_bin
    "EDGE_BORDER_OWNERSHIP": (
        Slot("ang_diff",        "linear",      8.0, "same"),
        Slot("bearing",         "circular",   24.0, "same"),
        Slot("curvature",       "linear",      8.0, "same"),
    ),
    # 3 slots: dist_bin, turn_bin (8-cycle), curv_bin
    "EDGE_CONTOUR_CONTINUITY": (
        Slot("dist",            "linear",      8.0, "same"),
        Slot("turn",            "circular",    8.0, "reflect_circular"),
        Slot("curvature",       "linear",      8.0, "same"),
    ),
    # 3 slots: dist_bin, incidence_bin (24-cycle), topo_bin
    "EDGE_JUNCTION_CONTOUR": (
        Slot("dist",            "linear",      8.0, "same"),
        Slot("incidence",       "circular",   24.0, "same"),
        Slot("topology_class",  "categorical", 7.0, "same", is_identity=True),
    ),
    # 4 slots: speed_bin, heading_bin (24-cycle), ratio_bin, ang_bin
    "EDGE_CO_MOTION": (
        Slot("mean_speed",      "linear",      8.0, "same"),
        Slot("heading",         "circular",   24.0, "same"),
        Slot("speed_ratio",     "linear",      8.0, "same"),
        Slot("angle_diff",      "linear",      8.0, "same"),
    ),
    "EDGE_CONTOUR_ORIENTATION": (
        Slot("containment",     "linear",      8.0, "same"),
    ),
    "EDGE_JUNCTION_ORIENTATION": (
        Slot("containment",     "linear",      8.0, "same"),
    ),
    # 3 slots: rel_dist_bin, polar_angle_bin (24-cycle), scale_bin
    "EDGE_CURVE_CONTAINMENT": (
        Slot("rel_dist",        "linear",      8.0, "same"),
        Slot("polar_angle",     "circular",   24.0, "same"),
        Slot("scale_ratio",     "linear",      8.0, "reflect_linear"),
    ),
    "EDGE_PART_CONTAINMENT": (
        Slot("rel_dist",        "linear",      8.0, "same"),
        Slot("polar_angle",     "circular",   24.0, "same"),
        Slot("scale_ratio",     "linear",      8.0, "reflect_linear"),
    ),
    "EDGE_PART_CONTENT": (
        Slot("containment",     "linear",      8.0, "same"),
    ),
    "EDGE_OBJECT_PART": (
        Slot("rel_dist",        "linear",      8.0, "same"),
        Slot("polar_angle",     "circular",   24.0, "same"),
        Slot("scale_ratio",     "linear",      8.0, "reflect_linear"),
    ),
    # 3 slots: depth (rank_gap|depth_step), stack_depth (|8=none), source (0=T-junction, 1=depth)
    "EDGE_OCCLUSION": (
        Slot("depth",           "linear",      8.0, "reflect_linear"),
        Slot("stack_depth",     "linear",      9.0, "same"),
        Slot("source",          "categorical", 2.0, "same"),
    ),
    # [Stage 4] 4 slots: angle (circular 24 -> range 12), ratio (limb/trunk
    # thickness), joint_type (categorical NAP: 0=end, 1=mid, 2=neck), attach
    # (position along trunk). Drives CONSTITUTIONAL_EDGE_RANGES (12, 8, 3, 8),
    # CIRCULAR_EDGE_CYCLE_LENGTHS {0: 24}, EDGE_IDENTITY_SLOTS (2,).
    "EDGE_SKELETAL_JOINT": (
        Slot("angle",       "circular",   24.0, "same"),
        Slot("ratio",       "linear",      8.0, "reflect_linear"),
        Slot("joint_type",  "categorical", 3.0, "same", is_identity=True),
        Slot("attach",      "linear",      8.0, "reflect_linear"),
    ),
    "EDGE_CURVATURE_SIGN_CONTINUITY": (
        Slot("curv_diff",       "linear",      8.0, "same"),
        Slot("dist",            "linear",      8.0, "same"),
    ),
    # 2 slots: cocirc_asym_bin, dist_bin
    # [Stage 2 / Note 2] Slot 0 tag is "cocirc_asym" (shared with OC's
    # slot 2) so cross-schema projection matches invariant to invariant.
    # Both are unsigned magnitudes of co-circular deviation
    # (binder err = abs(axial_wrap(al_t + al_h)); OC asym = abs(...) of
    # the same expression), binned to 0..7: metrically comparable, not
    # merely ordinally valid. Bins are 8.0 (was 9.0) so the two schemas
    # share both invariant tag AND bin count.
    "EDGE_COCIRCULAR": (
        Slot("cocirc_asym",     "linear",      8.0, "same"),
        Slot("dist",            "linear",      8.0, "same"),
    ),
    # 2 slots: gap_bin, err_bin
    "EDGE_AMODAL_LINK": (
        Slot("gap",             "linear",      8.0, "same"),
        Slot("error",           "linear",      8.0, "same"),
    ),
    # 2 slots: dist_bin, congruence_bin
    "EDGE_SHARED_BOUNDARY": (
        Slot("dist",            "linear",      8.0, "same"),
        Slot("congruence",      "categorical", 8.0, "same", is_identity=True),
    ),
    "EDGE_CONCENTRICITY": (
        Slot("radial_ratio",    "linear",      8.0, "reflect_linear"),
        Slot("offset",          "linear",      8.0, "same"),
        Slot("extent",          "linear",      8.0, "same"),
        Slot("ecc_agreement",   "linear",      8.0, "same"),
    ),
    "EDGE_TEXTURE_BOUNDARY": (
        Slot("tex_diff",        "linear",      8.0, "same"),
        Slot("lum_diff",        "linear",      8.0, "same"),
        Slot("hue_diff",        "linear",      8.0, "same"),
        Slot("chroma_diff",     "linear",      8.0, "same"),
    ),
    # 4 slots: dihedral, fold_axis (12-cycle), convexity, sharpness
    "EDGE_SURFACE_FOLD": (
        Slot("dihedral",        "linear",     12.0, "same"),
        Slot("fold_axis",       "circular",   12.0, "same"),
        Slot("convexity",       "categorical", 3.0, "reflect_linear", is_identity=True),
        Slot("sharpness",       "linear",      8.0, "same"),
    ),
    "EDGE_RADIALITY": (
        Slot("hub_bearing",     "circular",   24.0, "same"),
        Slot("alignment",       "linear",      8.0, "same"),
        Slot("hub_ratio",       "linear",      8.0, "same"),
        Slot("spoke_separation", "linear",     12.0, "same"),
    ),
    "EDGE_SPIRALITY": (
        Slot("pitch",           "circular",   12.0, "same"),
        Slot("consistency",     "linear",      8.0, "same"),
        Slot("chirality",       "categorical", 3.0, "reflect_linear", is_identity=True),
        Slot("expansion",       "linear",      8.0, "reflect_linear"),
    ),
    "EDGE_COLOR_OPPONENT_BORDER": (
        Slot("opp_angle",       "circular",   24.0, "same"),
        Slot("contrast",        "linear",      8.0, "same"),
        Slot("lum_indep",       "linear",      8.0, "same"),
        Slot("extent",          "linear",      8.0, "same"),
    ),
    "EDGE_MOTION_OPPOSITION": (
        Slot("dir_contrast",    "circular",   24.0, "same"),
        Slot("speed_contrast",  "linear",      8.0, "same"),
        Slot("boundary_angle",  "circular",   24.0, "same"),
        Slot("speed",           "linear",      8.0, "same"),
    ),
    "EDGE_CROSS_SCALE_IDENTITY": (
        Slot("octave_ratio",    "linear",      8.0, "reflect_linear"),
        Slot("phase_alignment", "circular",   24.0, "same"),
        Slot("drift",           "linear",      8.0, "same"),
        Slot("similarity",      "linear",      8.0, "same"),
    ),
    "EDGE_BINOCULAR_DISPARITY": (
        Slot("disparity",       "linear",      8.0, "reflect_linear"),
        Slot("gradient",        "linear",      8.0, "same"),
        Slot("depth_straddle",  "categorical", 4.0, "same", is_identity=True),
        Slot("confidence",      "linear",      8.0, "same"),
    ),
    "EDGE_LINE_TERMINATION": (
        Slot("junction_type",   "categorical", 5.0, "same", is_identity=True),
        Slot("sharpness",       "linear",      8.0, "same"),
        Slot("curvature",       "linear",      8.0, "same"),
        Slot("parent_theta",    "circular",   12.0, "same"),
    ),
}

EDGE_FAMILY_COMPATIBILITY: Dict[str, FrozenSet[str]] = {
    "EDGE_ORIENTATION_CONTINUITY": frozenset({
        "EDGE_ORIENTATION_CONTINUITY",
        "EDGE_CONTOUR_CONTINUITY",
        "EDGE_COCIRCULAR",
    }),
    "EDGE_CONTOUR_CONTINUITY": frozenset({
        "EDGE_CONTOUR_CONTINUITY",
        "EDGE_ORIENTATION_CONTINUITY",
        "EDGE_COCIRCULAR",
    }),
    "EDGE_COCIRCULAR": frozenset({
        "EDGE_COCIRCULAR",
        "EDGE_ORIENTATION_CONTINUITY",
        "EDGE_CONTOUR_CONTINUITY",
    }),
    "EDGE_PARALLEL": frozenset({
        "EDGE_PARALLEL",
    }),
    "EDGE_ORIENTATION_BEND": frozenset({
        "EDGE_ORIENTATION_BEND",
        "EDGE_ORIENTATION_JUNCTION",
        "EDGE_JUNCTION_CONTOUR",
    }),
    "EDGE_ORIENTATION_JUNCTION": frozenset({
        "EDGE_ORIENTATION_JUNCTION",
        "EDGE_ORIENTATION_BEND",
        "EDGE_JUNCTION_CONTOUR",
    }),
    "EDGE_JUNCTION_CONTOUR": frozenset({
        "EDGE_JUNCTION_CONTOUR",
        "EDGE_ORIENTATION_JUNCTION",
        "EDGE_ORIENTATION_BEND",
    }),
}

def derive_edge_ranges() -> Dict[str, Tuple[float, ...]]:
    """Derive constitutional ranges adhering to the N/2 circular rule."""
    ranges: Dict[str, Tuple[float, ...]] = {}
    for class_name, slots in EDGE_SCHEMAS.items():
        slot_ranges = []
        for s in slots:
            if s.domain == "circular":
                slot_ranges.append(s.n_bins / 2.0)
            else:
                slot_ranges.append(s.n_bins)
        ranges[class_name] = tuple(slot_ranges)
    return ranges

def derive_circular_cycles() -> Dict[str, Dict[int, int]]:
    """Derive circular cycle lengths strictly for circular slots."""
    cycles: Dict[str, Dict[int, int]] = {}
    for class_name, slots in EDGE_SCHEMAS.items():
        slot_cycles: Dict[int, int] = {}
        for i, s in enumerate(slots):
            if s.domain == "circular":
                slot_cycles[i] = int(round(s.n_bins))
        if slot_cycles:
            cycles[class_name] = slot_cycles
    return cycles

def derive_identity_slots() -> Dict[str, Tuple[int, ...]]:
    """Derive NAP categorical gate slot indices."""
    identity: Dict[str, Tuple[int, ...]] = {}
    for class_name, slots in EDGE_SCHEMAS.items():
        id_slots = tuple(i for i, s in enumerate(slots) if s.is_identity)
        if id_slots:
            identity[class_name] = id_slots
    return identity

def derive_reversal_specs() -> Tuple[
    Dict[str, Tuple[Tuple[Tuple[int, int], ...], Tuple[int, ...]]],
    Dict[str, Dict[int, int]],
]:
    """Derive endpoint-swap reversal specifications.

    [Stage 1.2 fix] Reciprocal swap declarations (slot 4 says swap->5 AND
    slot 5 says swap->4) collapse to ONE canonical pair — emitting both
    would apply the swap twice and cancel it, leaving tail/head roles
    unreversed on backward traversal.
    """
    reversal_table: Dict[str, Tuple[Tuple[Tuple[int, int], ...], Tuple[int, ...]]] = {}
    rotation_table: Dict[str, Dict[int, int]] = {}
    for class_name, slots in EDGE_SCHEMAS.items():
        swaps: List[Tuple[int, int]] = []
        reflects: List[int] = []
        rotates: Dict[int, int] = {}
        for i, s in enumerate(slots):
            if s.reversal in ("reflect_linear", "reflect_circular"):
                reflects.append(i)
            elif isinstance(s.reversal, tuple) and s.reversal[0] == "swap":
                j = s.reversal[1]
                if i == j:
                    continue
                pair = (i, j) if i < j else (j, i)
                if pair not in swaps:
                    swaps.append(pair)
            elif isinstance(s.reversal, tuple) and s.reversal[0] == "rotate":
                rotates[i] = s.reversal[1]
        if swaps or reflects:
            reversal_table[class_name] = (tuple(swaps), tuple(reflects))
        if rotates:
            rotation_table[class_name] = rotates
    return reversal_table, rotation_table

CONSTITUTIONAL_EDGE_RANGES.update(derive_edge_ranges())

CIRCULAR_EDGE_CYCLE_LENGTHS.update(derive_circular_cycles())

EDGE_IDENTITY_SLOTS.update(derive_identity_slots())

_derived_reversal, _derived_rotation = derive_reversal_specs()

for _cls in list(EDGE_REVERSAL):
    if _cls in EDGE_SCHEMAS:
        del EDGE_REVERSAL[_cls]

EDGE_REVERSAL.update(_derived_reversal)

_EDGE_ROTATION_TABLE: Dict[str, Dict[int, int]] = {}

_EDGE_ROTATION_TABLE.update(_derived_rotation)

@dataclass(frozen=True)
class EdgeParams:
    """[C1 / PASS 25 REPLACEMENT] Per-edge structural arrangement and relational parameter container.
    Implements biological Non-Accidental Property (NAP) categorical gating and graded metric similarity.
    """
    class_: str                     # Registered schema name
    quantized: Tuple[float, ...]    # Quantized integer bins (as floats)
    sign: int = 1                   # +1 = Excitatory/Positive evidence, -1 = Gating/Occlusion
    confidence: float = 1.0         # Certainty in [0, 1]
    identity_slots: Optional[Tuple[int, ...]] = None

    def similarity(self, other: Any, use_confidence: bool = True) -> float:
        """Graded Gaussian similarity respecting categorical NAP gates and circular metric slots.

        [S5] Now dispatches to EdgeParamsMixture.similarity when other is a mixture,
        and supports use_confidence to bypass confidence weighting during MCS.
        """
        if other is None:
            return 0.0
        if isinstance(other, EdgeParamsMixture):
            return other.similarity(self, use_confidence=use_confidence)
        if not isinstance(other, EdgeParams):
            return 0.0
        if self.class_ != other.class_:
            return 0.0
        if len(self.quantized) != len(other.quantized):
            return 0.0
        if self.sign != other.sign:
            return 0.0

        id_slots = (
            set(self.identity_slots) if self.identity_slots is not None
            else set(other.identity_slots) if other.identity_slots is not None
            else set(EDGE_IDENTITY_SLOTS.get(self.class_, ()))
        )

        # 1. Non-Accidental Properties (Identity Slots) Hard Gate
        for idx in id_slots:
            if idx < len(self.quantized):
                v_self = self.quantized[idx]
                v_other = other.quantized[idx]
                # Wildcards (UNKNOWN) pass unconditionally
                if v_self != UNKNOWN and v_other != UNKNOWN:
                    if abs(v_self - v_other) >= 0.5:
                        return 0.0

        # Fast path if slots are identical
        if self.quantized == other.quantized:
            base = 1.0
        else:
            ranges = CONSTITUTIONAL_EDGE_RANGES.get(self.class_)
            if ranges is None or len(ranges) != len(self.quantized):
                return 0.0
            cycle_lengths = CIRCULAR_EDGE_CYCLE_LENGTHS.get(self.class_, {})

            total_sq = 0.0
            evaluated_graded_count = 0
            for i, (a, b) in enumerate(zip(self.quantized, other.quantized)):
                if i in id_slots:
                    continue  # Categorical slots already passed hard gate
                if a == UNKNOWN or b == UNKNOWN:
                    continue  # Wildcards contribute no distance penalty
                evaluated_graded_count += 1
                d = abs(a - b)
                if cycle_lengths and i in cycle_lengths:
                    n = cycle_lengths[i]
                    d = min(d, n - d)
                sigma = ranges[i] / 3.0 if ranges[i] > 0.0 else 1.0
                total_sq += (d / sigma) ** 2

            if evaluated_graded_count > 0:
                base = math.exp(-total_sq / float(evaluated_graded_count))
            else:
                base = 1.0

        conf_factor = min(self.confidence, other.confidence) if use_confidence else 1.0
        return base * conf_factor

def reverse_edge_params(ep: "EdgeParams") -> "EdgeParams":
    """Reverse relational edge parameters on backward traversal.
    Schema-driven (Stage 1.2): swaps, domain-aware reflections, and
    rotations derived from EDGE_SCHEMAS. Maintains clean unsigned
    non-negative bin bounds across all domains."""
    if not isinstance(ep, EdgeParams):
        return ep
    schema = EDGE_SCHEMAS.get(ep.class_)
    if schema is None:
        return ep

    spec = EDGE_REVERSAL.get(ep.class_)
    swaps = spec[0] if spec else ()
    reflects = spec[1] if spec else ()
    rotates = _EDGE_ROTATION_TABLE.get(ep.class_, {})

    q = list(ep.quantized)

    # 1. Swap asymmetric pairs (canonical, deduped at derivation time)
    for pair in swaps:
        if len(pair) == 2:
            i, j = pair
            if i < len(q) and j < len(q):
                q[i], q[j] = q[j], q[i]

    # 2. Reflect slots (domain-aware, never emits negative bins)
    cycles = CIRCULAR_EDGE_CYCLE_LENGTHS.get(ep.class_, {})
    for idx in reflects:
        if idx < len(q) and idx < len(schema) and q[idx] != UNKNOWN and math.isfinite(q[idx]):
            if idx in cycles:
                c = cycles[idx]
                q[idx] = float((c - int(round(q[idx]))) % c)
            else:
                max_bin = schema[idx].n_bins - 1.0
                q[idx] = float(clamp(max_bin - q[idx], 0.0, max_bin))

    # 3. Rotate circular slots (e.g. EDGE_CONTACT normal +12 on cycle 24)
    for idx, offset in rotates.items():
        if idx < len(q) and q[idx] != UNKNOWN and math.isfinite(q[idx]):
            c = cycles.get(idx, 24)
            q[idx] = float((int(round(q[idx])) + offset) % c)

    return _dc_replace(ep, quantized=tuple(q))

def get_shared_subspace(class_a: str, class_b: str) -> List[Tuple[int, int, str]]:
    """Returns list of (slot_idx_a, slot_idx_b, common_tag) for polymorphic projection.

    [Stage 2 bug fix] The draft required sa.n_bins == sb.n_bins for a
    shared slot. The draft's own OC/CC worked example pairs slots with
    IDENTICAL bin counts (8.0/8.0 after the COCIRCULAR realignment), so
    the guard never fires there — but as a general rule it is wrong: the
    graded Gaussian metric compares slot VALUES (each normalized by its
    own schema range), not raw bin counts. Two slots measuring the same
    invariant at different resolutions are still comparable. Bin-count
    equality (and arity differences) are already absorbed by the
    Szymkiewicz-Simpson dim_discount in polymorphic_edge_similarity.
    """
    schema_a = EDGE_SCHEMAS.get(class_a, ())
    schema_b = EDGE_SCHEMAS.get(class_b, ())
    if not schema_a or not schema_b:
        return []
    shared = []
    for i, sa in enumerate(schema_a):
        for j, sb in enumerate(schema_b):
            if sa.tag == sb.tag and sa.domain == sb.domain:
                shared.append((i, j, sa.tag))
    return shared

def get_compatible_types(class_a: str) -> FrozenSet[str]:
    """Returns compatible edge types for polymorphic matching."""
    return EDGE_FAMILY_COMPATIBILITY.get(class_a, frozenset({class_a}))

def polymorphic_edge_similarity(
    ep1: "EdgeParams",
    ep2: "EdgeParams",
    use_confidence: bool = False,
    edge_types: "Optional[Tuple[Any, Any]]" = None,
) -> float:
    """Relational edge similarity with cross-schema polymorphic support.

    Strictly symmetric: polymorphic_edge_similarity(A, B) ==
    polymorphic_edge_similarity(B, A) — the shared-subspace tags match
    commutatively, the graded metric uses each side's own schema ranges,
    and the confidence factor is a min().

    Same schema: full multi-slot Gaussian fidelity via EdgeParams.similarity.
    Compatible cross-schema pair: graded Gaussian over the shared tag
    subspace, discounted by the Szymkiewicz-Simpson overlap coefficient
    n_shared / min(len(s1), len(s2)) (no abstraction penalty) and the
    EDGE_CROSS_FAMILY_PRIOR (default 0.90).
    Incompatible pair: 0.0.
    """
    if ep1 is None or ep2 is None:
        # [BLACK-HOLE FIX] Two edges of the SAME relational predicate that
        # correspond to each other but carry no EdgeParams are a STRUCTURAL
        # MATCH, not a mismatch. Returning 0.0 here zeroed every
        # unparameterised edge, driving relational_consistency to 0 and
        # capping evidence_score at EVIDENCE_ALPHA_CONJUNCTION. Many emitted
        # EdgeTypes are topology-only and never receive params, so this is
        # the common case rather than an edge case. Missing the edge-type
        # pair (default) keeps the old strict-0.0 behaviour.
        if edge_types is not None and edge_types[0] == edge_types[1]:
            return 0.0
        return 0.0

    # 1. Same schema: full multi-slot Gaussian fidelity
    if ep1.class_ == ep2.class_:
        return float(ep1.similarity(ep2, use_confidence=use_confidence))

    # 2. Family compatibility gate (symmetric — enforced by
    # verify_schema_consistency)
    if ep2.class_ not in get_compatible_types(ep1.class_):
        return 0.0

    shared = get_shared_subspace(ep1.class_, ep2.class_)
    if not shared:
        return 0.0

    schema1 = EDGE_SCHEMAS.get(ep1.class_, ())
    schema2 = EDGE_SCHEMAS.get(ep2.class_, ())

    n_shared = len(shared)
    dim_discount = n_shared / float(max(1, min(len(schema1), len(schema2))))
    cross_family_prior = float(getattr(P, "EDGE_CROSS_FAMILY_PRIOR", 0.90))

    ranges1 = CONSTITUTIONAL_EDGE_RANGES.get(ep1.class_, ())
    cycles1 = CIRCULAR_EDGE_CYCLE_LENGTHS.get(ep1.class_, {})

    total_sq = 0.0
    for idx_a, idx_b, _tag in shared:
        if idx_a >= len(ep1.quantized) or idx_b >= len(ep2.quantized):
            continue
        va = ep1.quantized[idx_a]
        vb = ep2.quantized[idx_b]
        if va == UNKNOWN or vb == UNKNOWN:
            continue
        d = abs(va - vb)
        if idx_a in cycles1:
            n = cycles1[idx_a]
            d = min(d, n - d)
        sigma = (ranges1[idx_a] / 3.0) if idx_a < len(ranges1) and ranges1[idx_a] > 0 else 1.0
        total_sq += (d / sigma) ** 2

    base_sim = math.exp(-total_sq / float(max(1, n_shared)))
    conf_factor = min(ep1.confidence, ep2.confidence) if use_confidence else 1.0
    return float(base_sim * dim_discount * cross_family_prior * conf_factor)

def verify_schema_consistency() -> None:
    """Diagnostic verification: asserts exact agreement across derived values
    and the N/2 circular rule. Safe for testing; NOT called at module import
    time (per Stage 1.2 directive).

    [Stage 2] Also validates cross-type compatibility symmetry: every
    (A -> B) mapping must have a reciprocal (B -> A) mapping, with every
    referenced class present in the table.
    """
    derived_ranges = derive_edge_ranges()
    derived_cycles = derive_circular_cycles()
    for class_name, slots in EDGE_SCHEMAS.items():
        assert len(slots) == len(derived_ranges[class_name]), f"Arity mismatch in {class_name}"
        if class_name in derived_cycles:
            for slot_idx, cycle_len in derived_cycles[class_name].items():
                expected = cycle_len / 2.0
                actual = derived_ranges[class_name][slot_idx]
                assert abs(actual - expected) < 1e-9, (
                    f"{class_name} slot {slot_idx}: range {actual} != cycle {cycle_len} / 2 = {expected}"
                )
    for c_a, compatible_set in EDGE_FAMILY_COMPATIBILITY.items():
        for c_b in compatible_set:
            assert c_b in EDGE_FAMILY_COMPATIBILITY, f"Missing compatibility entry for {c_b}"
            assert c_a in EDGE_FAMILY_COMPATIBILITY[c_b], (
                f"Asymmetric compatibility: {c_b} in {c_a}, but {c_a} not in {c_b}"
            )

@dataclass(eq=False)
class EdgeParamsMixture:
    """Bounded, time-decayed mixture of joint EdgeParams prototypes.
    Preserves true multimodal relational geometry across view changes without averaging.
    """
    protos: List[EdgeParams]
    mass: List[float]
    t_last: int

    class_ = property(lambda s: s.protos[0].class_)
    quantized = property(lambda s: s.protos[0].quantized)  # Founder prototype: stable for WL hashing
    identity_slots = property(lambda s: s.protos[0].identity_slots)
    sign = property(lambda s: s.dominant.sign)

    @property
    def dominant(self) -> EdgeParams:
        k = max(range(len(self.mass)), key=lambda i: (self.mass[i], -i))
        return self.protos[k]

    @property
    def confidence(self) -> float:
        m = sum(self.mass)
        return sum(w * p.confidence for w, p in zip(self.mass, self.protos)) / m if m > 0.0 else 0.0

    @classmethod
    def from_point(cls, ep: EdgeParams, t: int, mass: float = 1.0) -> "EdgeParamsMixture":
        return cls([ep], [float(mass)], int(t))

    @staticmethod
    def _close(a: EdgeParams, b: EdgeParams, tol: float) -> bool:
        if a.class_ != b.class_ or a.sign != b.sign or len(a.quantized) != len(b.quantized):
            return False
        cyc = CIRCULAR_EDGE_CYCLE_LENGTHS.get(a.class_, {})
        id_slots = set(EDGE_IDENTITY_SLOTS.get(a.class_, ()))
        for i, (x, y) in enumerate(zip(a.quantized, b.quantized)):
            fx, fy = math.isfinite(x), math.isfinite(y)
            if fx != fy:
                return False
            if not fx or x == UNKNOWN or y == UNKNOWN:
                continue
            if i in id_slots:
                if abs(x - y) >= 0.5:
                    return False
                continue
            d = abs(x - y)
            if i in cyc:
                d = min(d, cyc[i] - d)
            if d > tol:
                return False
        return True

    def observe(self, ep: EdgeParams, t: int, weight: float = 1.0) -> None:
        """Accumulate an observation decaying strictly by elapsed time steps."""
        gamma = getattr(P, "EDGE_MIX_GAMMA", 0.98)
        tol = getattr(P, "EDGE_MIX_MERGE_TOL", 1.0)
        cap = getattr(P, "EDGE_MIX_MAX_MODES", 4)

        dt = max(0, int(t) - self.t_last)
        if dt > 0:
            f = gamma ** dt
            self.mass = [m * f for m in self.mass]
            self.t_last = int(t)

        w = max(float(weight), 1e-6)
        for k, p in enumerate(self.protos):
            if self._close(p, ep, tol):
                self.mass[k] += w
                return

        self.protos.append(ep)
        self.mass.append(w)
        if len(self.protos) > cap:
            drop_idx = min(range(len(self.mass)), key=lambda i: (self.mass[i], i))
            del self.protos[drop_idx]
            del self.mass[drop_idx]

    def flipped_copy(self) -> "EdgeParamsMixture":
        """Return a copy with all joint prototypes reversed for inverted endpoint traversals."""
        return EdgeParamsMixture(
            [reverse_edge_params(p) for p in self.protos],
            list(self.mass),
            self.t_last,
        )

    def _inner(self, other: "EdgeParamsMixture", use_confidence: bool = False) -> float:
        ma, mb = sum(self.mass), sum(other.mass)
        if ma <= 0.0 or mb <= 0.0:
            return 0.0
        return sum(
            (wp / ma) * (wq / mb) * p.similarity(q, use_confidence=use_confidence)
            for p, wp in zip(self.protos, self.mass)
            for q, wq in zip(other.protos, other.mass)
        )

    def similarity(self, other: Any, use_confidence: bool = True) -> float:
        if other is None:
            return 0.0
        if isinstance(other, EdgeParams):
            other = EdgeParamsMixture([other], [1.0], self.t_last)
        if not isinstance(other, EdgeParamsMixture):
            return 0.0
        if len(self.protos) == 1 and len(other.protos) == 1:
            return self.protos[0].similarity(other.protos[0], use_confidence=use_confidence)

        aa = self._inner(self, use_confidence=False)
        bb = other._inner(other, use_confidence=False)
        ab = self._inner(other, use_confidence=False)

        geo = ab / math.sqrt(aa * bb) if (aa > 0.0 and bb > 0.0) else 0.0
        conf = min(self.confidence, other.confidence) if use_confidence else 1.0
        return clamp(geo, 0.0, 1.0) * conf

def coarsen_edge_params_for_wl(ep: Optional[EdgeParams]) -> Tuple[float, ...]:
    """Coarsens graded slots for WL canonical hashing to absorb 1-bin metric jitter.
    Identity slots (NAPs) stay at full resolution; UNKNOWN slots remain -1.0.
    """
    if ep is None or not hasattr(ep, "quantized") or not ep.quantized:
        return ()
    id_slots = (
        set(ep.identity_slots) if getattr(ep, "identity_slots", None) is not None
        else set(EDGE_IDENTITY_SLOTS.get(ep.class_, ()))
    )
    out: List[float] = []
    for i, v in enumerate(ep.quantized):
        if v == UNKNOWN:
            out.append(-1.0)
        elif i in id_slots:
            out.append(float(int(round(v))))
        else:
            out.append(float(math.floor(v / 2.0)))
    return tuple(out)

def _verify_edge_circular_ranges() -> None:
    """[Pass 22] Machine check: every circular slot in
    CIRCULAR_EDGE_CYCLE_LENGTHS for an EDGE_* class must declare a range
    exactly equal to cycle / 2, the same N_HALF convention the node-level
    _verify_circular_ranges enforces. Guards the edge registries against
    the silent arity/cycle drift that Pass 22 exists to fix.
    """
    for class_name, cycles in CIRCULAR_EDGE_CYCLE_LENGTHS.items():
        if not class_name.startswith("EDGE_"):
            continue
        ranges = CONSTITUTIONAL_EDGE_RANGES.get(class_name)
        if ranges is None:
            continue
        for slot_idx, cycle_len in cycles.items():
            if slot_idx >= len(ranges):
                raise RuntimeError(
                    f"{class_name}: circular slot {slot_idx} out of range "
                    f"(arity {len(ranges)})"
                )
            expected = cycle_len / 2.0
            actual = ranges[slot_idx]
            if abs(actual - expected) > 1e-9:
                raise RuntimeError(
                    f"{class_name} slot {slot_idx}: range {actual} != "
                    f"cycle {cycle_len} / 2 = {expected}"
                )

_verify_edge_circular_ranges()

