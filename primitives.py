"""primitives.py -- COMPATIBILITY SHIM.

Reorganised into `core/schema.py`; re-exported here so the
original import path keeps working.
"""
from __future__ import annotations

from core.schema import (  # noqa: F401
    CIRCULAR_EDGE_CYCLE_LENGTHS,
    CONSTITUTIONAL_EDGE_RANGES,
    EDGE_FAMILY_COMPATIBILITY,
    EDGE_IDENTITY_SLOTS,
    EDGE_REVERSAL,
    EDGE_SCHEMAS,
    EdgeParams,
    EdgeParamsMixture,
    Slot,
    UNKNOWN,
    coarsen_edge_params_for_wl,
    derive_circular_cycles,
    derive_edge_ranges,
    derive_identity_slots,
    derive_reversal_specs,
    get_compatible_types,
    get_edge_similarity_ranges,
    get_shared_subspace,
    polymorphic_edge_similarity,
    reverse_edge_params,
    verify_schema_consistency,
)

from core.registries import (  # noqa: F401
    ACTION_NOOP,
    ACTION_NULL,
    CIRCULAR_CYCLE_LENGTHS,
    CONSTITUTIONAL_RANGES,
    ORDER_2_EDGE_TYPES,
    PROBE_ACTION_PREFIX,
    SELF_BINDER_TYPES,
    SLOT_WEIGHTS,
    SLOT_WEIGHTS_BY_CONTEXT,
    SLOT_WEIGHT_PRIORS,
    _B1_DERIVED_ARITY,
    _B4_DERIVED_ARITY,
    _ENFORCE_N_HALF_CONVENTION,
    _GABOR_NSF,
    _GABOR_NT,
    _ORIENTATION_MARGINALS,
    _verify_circular_ranges,
    b1_class_name,
    b4_class_name,
    compute_b1_arity,
    compute_b4_arity,
    edge_type_layer,
    get_similarity_ranges,
    get_slot_weights,
    is_action_lens_eligible,
    is_noop_action,
    is_probe_action,
    is_teleological_action,
    resolve_slot_weights,
)

from core.quantize import (  # noqa: F401
    _MAX_BINS,
    _aang,
    _opens_toward,
    _q_linear_safe,
    _q_signed,
    _q_unsigned,
    carrier_to_tangent,
    q_linear,
    theta_to_bin,
)

from core.spatial import (  # noqa: F401
    SpatialHash,
    _blur_2x,
    _pos_dist,
    anchor_in_boxes,
    build_pyramid,
    normalize_2d,
    tangent_at,
)

from core.junctions import (  # noqa: F401
    detect_region_junctions,
)

from core.hashing import (  # noqa: F401
    _detection_sort_key,
    _node_signature_hash,
    clamp,
    hash_128,
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

from core.types import (  # noqa: F401
    Detection,
    EdgeType,
    HEBBIAN_BOOST_TYPES,
    REPRODUCIBLE_EDGE_TYPES,
    TypedSignature,
    UNDIRECTED_EDGE_TYPES,
    _SIGMA_FALLBACK_RANGE0,
    _SIMILARITY_IMPL,
    _SIM_MODE,
    _serialize_any,
    _similarity_gaussian,
    _similarity_linear,
    _typed_signature_bytes,
    serialize_detection,
)

