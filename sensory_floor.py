"""sensory_floor.py -- sensory_floor.py — Canonical state construction and physical primitive

The perception surface moved to `perception/`; this module re-exports
it so every existing `from sensory_floor import X` keeps working.
Import from `perception.*` in new code.
"""
from __future__ import annotations

from core.hashing import (  # noqa: F401
    clamp,
    hash_64,
    hash_64_signed,
    heaviside,
    serialize_bytes,
    serialize_float,
    serialize_int,
    serialize_set,
    serialize_tuple,
    serialize_value,
)

from core.junctions import (  # noqa: F401
    detect_region_junctions,
)

from core.registries import (  # noqa: F401
    CIRCULAR_CYCLE_LENGTHS,
    CONSTITUTIONAL_RANGES,
    SLOT_WEIGHT_PRIORS,
    get_similarity_ranges,
    get_slot_weights,
)

from core.spatial import (  # noqa: F401
    SpatialHash,
)

from core.types import (  # noqa: F401
    Detection,
    TypedSignature,
    _serialize_any,
    _typed_signature_bytes,
    serialize_detection,
)

from perception.contours import (  # noqa: F401
    V1RetinotopicLattice,
    _angle_diff,
    _autocorr_peak,
    _axial_angle_diff,
    _compute_coarseness,
    _compute_regularity,
    _detect_t_junctions,
    _extract_association_field_contours,
    _front_end_scale,
    _local_edge_at,
    _moore_neighbor_trace,
    _polyline_arc_length,
    _resample_contours,
    _segment_profile,
    _tangent_at,
    compute_texture_rich,
    extract_shock_structure,
    ribbon_bins,
    texture_richness,
)

from perception.filters import (  # noqa: F401
    GaborBank,
    M_srgb_to_xyz,
    M_xyz_to_lms,
    _TINY,
    _conv2d,
    _gabor_kernel,
    center_surround,
    color_constancy,
    color_contrast,
    compute_opponent_retinex,
    compute_responses,
    illuminant_estimate,
    luminance_adapt,
    opponent_channels,
    split_on_off,
    srgb_to_linear,
)

from perception.floor import (  # noqa: F401
    SensoryFloor,
    diagnostic_report,
)

from perception.input.eth80 import (  # noqa: F401
    _ETH80_VIEW,
    parse_eth80_view,
)

from perception.observation import (  # noqa: F401
    ACTION_LENS_NAME,
    ACTION_ROLE_NAME,
    ACTION_SIG_TAG,
    ALPHA_P,
    AdapterConfig,
    AnchorSet,
    AnchorState,
    BETA_T,
    CameraIntrinsics,
    ContinuousObservation,
    DEG_MAX,
    DEG_MAX_3D,
    DELTA_COHERENCE,
    EPSILON,
    EPS_CONTACT,
    EPS_GATE,
    EPS_VEL,
    HASH_BITS,
    H_REF,
    M_KEYPOINTS,
    N_ANCHOR_MAX,
    Observation,
    Observation3D,
    Q_S,
    R_CLUSTER,
    R_CLUTTER,
    R_LOCAL_3D,
    R_MAX_OBJ,
    R_MIN,
    R_MIN_SEP,
    R_STR_3D,
    StatePattern,
    SymbolicObservation,
    TAU_LIFE,
    TAU_NORM,
    TAU_PERSIST,
    THETA_DEPTH_HIGH,
    THETA_DEPTH_LOW,
    Transition,
    U_AGENT,
    V_MAX,
    V_THRESH,
    W_CM,
    W_R,
    W_SEQ,
    W_T,
    _SF_TINY,
    action_signature,
    median,
    normalize,
    resolve_anchor_position,
)

from perception.observation_2d import (  # noqa: F401
    Observation2D,
    _bilinear_resize,
    downsample_observation,
)

from perception.polar import (  # noqa: F401
    _arc_mid,
    _arc_split,
    _axial_wrap_sf,
    polar_boundary_fragments,
    signed_menger,
)

from perception.rich import (  # noqa: F401
    _B1_PREP_CACHE,
    _B1_SCHEMA_CACHE,
    _B1_SLOT_WEIGHTS_BY_CLASS,
    _B1_W_BASE,
    _B1_W_HIST,
    _B1_W_ONOFF_SUBBIN,
    _B1_W_PHASE,
    _B1_W_Q_PHASE_DIFF,
    _B1_W_Q_THETA_DIFF,
    _B1_W_SF_CURV,
    _B1_W_SP_HIST,
    _B1_W_SP_NONHIST,
    _B1_W_SURROUND,
    _B4_REGISTERED_KEYS,
    _B4_W_TEXTURE_RICH,
    _compute_histogram,
    _compute_phase_stats,
    _compute_simple_flag,
    _compute_subbin,
    _envelope_cov,
    _gaussian_filter_sf,
    _gaussian_kernel,
    _norm_weights,
    _phase_step_vec,
    _prepare_bank_b1,
    _wrap_half_pi,
    _wrap_pi,
    b1_class_name,
    b1_region_signature,
    b1_register,
    b1_reset,
    b1_schema,
    b4_class_name,
    b4_n_slots,
    b4_register,
    compute_responses_b1,
)


# --- re-export the similarity dispatch (audit repair) ---
from core.types import (  # noqa: F401
    _SIMILARITY_IMPL,
    _SIM_MODE,
    _SIGMA_FALLBACK_RANGE0,
    _similarity_gaussian,
    _similarity_linear,
)
