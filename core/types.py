"""core/types.py -- representation types shared by every layer.

MOVE ONLY. Blocks copied verbatim from binder.py (EdgeType + the edge-type
registries) and sensory_floor.py (TypedSignature, its similarity dispatch,
Detection and their serializers). binder.py and sensory_floor.py now import
these from here, so modules that only wanted an enum or a signature no longer pull
in the whole binder/perception stack.

Contract: core/ imports NOTHING else from the project.
EXCEPTIONS (documented, transitional):
  * `import parameters as P` -- parameters.py is a leaf config
    module (imports no project module), so it cannot cycle.
  * the schema lookups `get_similarity_ranges`, `get_slot_weights`,
    `CIRCULAR_CYCLE_LENGTHS`, `EPSILON` come from primitives.py.
    primitives is the module that BECOMES core/schema.py, so this is
    a seam that collapses when primitives is split (next move);
    primitives itself imports only `parameters`, so no cycle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import parameters as P  # noqa: F401  (leaf config; see the note above)
from core.registries import (  # noqa: F401
    CIRCULAR_CYCLE_LENGTHS,
    get_similarity_ranges,
    get_slot_weights,
)
from parameters import EPSILON  # noqa: F401

from core.hashing import (
    hash_64,
    hash_64_signed,
    serialize_bytes,
    serialize_float,
    serialize_int,
    serialize_set,
    serialize_tuple,
    serialize_value,
)


# --- from binder.py: edge-type enum ---
class EdgeType(Enum):
    CO_ACTIVATION = 0
    TRANSITION = 1
    CO_MOTION = 2
    CONTRAST = 3
    PERIODICITY = 4
    PERSISTENCE = 5
    TOPOLOGY = 6
    DEGREE = 7
    CONTACT = 8
    SEQUENCE = 9
    SYNCHRONY = 13  # [P2-2] phase-locked onset synchrony (undirected)
    ORIENTATION_CONTINUITY = 14  # [v8] collinear oriented edges
    SURFACE_GROUPING = 15  # [v8] similar surface properties
    CURVATURE = 16  # [v9] adjacent-segment bend
    OCCLUSION = 17  # [v9] occlusion relation
    # v32 cross-lens binders
    CO_LOCATED = 18
    ORIENTATION_BEND = 19
    ORIENTATION_JUNCTION = 20
    # [PHASE B] Higher-tier structure (futurework3.md B.11). Emitted by
    # Binder.emit_pending_edges, partitioned by level: v2 = 21/22,
    # v4 = 23, lo_pit = 24/25, ait = 26.
    CONTOUR_CONTINUITY = 21
    JUNCTION_CONTOUR = 22
    CURVE_CONTAINMENT = 23
    PART_CONTAINMENT = 24
    PART_CONTENT = 25
    OBJECT_PART = 26
    # [PHASE B] B.11 values 27/28: the V2 lenses' link back to the
    # Orientation segments they summarize. Added in Step 3 (the final
    # spec's B.11 lists eight types, not six).
    CONTOUR_ORIENTATION = 27
    JUNCTION_ORIENTATION = 28
    # [V9 Track 2 - B.8] values 29-32. Additive registration: the
    # emitters land later in Track 2/3, so nothing reads these yet.
    PARALLEL = 31
    CO_LOCATED_CHANNEL = 32
    ILLUSORY_CONTINUITY = 33
    # [Pass 12] Relational edge types E8–E11
    SYMMETRY = 34
    PART_SEQUENCE = 35
    PART_AT_ANGLE = 36
    BORDER_OWNERSHIP_EDGE = 37
    # [Pass 16] Relational edge types
    CURVATURE_SIGN_CONTINUITY = 38
    AMODAL_LINK = 39
    COCIRCULAR = 40
    SHARED_BOUNDARY = 41

    # [Pass 23] V4 non-Cartesian + chromatic/motion/3D relational edges
    CONCENTRICITY = 42
    TEXTURE_BOUNDARY_EDGE = 43
    SURFACE_FOLD = 44
    RADIALITY = 45
    SPIRALITY = 46
    COLOR_OPPONENT_BORDER = 47
    MOTION_OPPOSITION = 48
    CROSS_SCALE_IDENTITY = 49
    BINOCULAR_DISPARITY = 50
    LINE_TERMINATION = 51
    SKELETAL_JOINT = 52


# --- from binder.py: edge-type registries ---
UNDIRECTED_EDGE_TYPES: FrozenSet[EdgeType] = frozenset(
    {
        EdgeType.CO_ACTIVATION,
        EdgeType.CO_MOTION,
        EdgeType.PERIODICITY,
        EdgeType.DEGREE,
        EdgeType.CONTACT,
        EdgeType.SYNCHRONY,
        EdgeType.ORIENTATION_CONTINUITY,
        EdgeType.SURFACE_GROUPING,
        EdgeType.CURVATURE,
        EdgeType.CO_LOCATED,
        EdgeType.ORIENTATION_BEND,
        EdgeType.ORIENTATION_JUNCTION,
        # [PHASE B] B.11: CONTOUR_CONTINUITY (a chain is continuous with
        # its neighbour either way) and PART_CONTENT (content membership
        # is unordered). The other four higher-tier types are directed
        # containment/relation and are deliberately absent here.
        EdgeType.CONTOUR_CONTINUITY,
        EdgeType.PART_CONTENT,
        # [V9 B.8] unordered relations: co-circular arc membership,
        # parallelism, and the co-located-channel link.
        EdgeType.COCIRCULAR,
        EdgeType.PARALLEL,
        EdgeType.CO_LOCATED_CHANNEL,
        EdgeType.ILLUSORY_CONTINUITY,
        EdgeType.SYMMETRY,
        EdgeType.CURVATURE_SIGN_CONTINUITY,
        EdgeType.AMODAL_LINK,
        EdgeType.SHARED_BOUNDARY,
        # [Pass 23] undirected among the new vocabulary
        EdgeType.TEXTURE_BOUNDARY_EDGE,
        EdgeType.RADIALITY,
        EdgeType.SPIRALITY,
        EdgeType.COLOR_OPPONENT_BORDER,
        EdgeType.LINE_TERMINATION,
    }
)

REPRODUCIBLE_EDGE_TYPES: FrozenSet[EdgeType] = frozenset({
    EdgeType.CO_ACTIVATION,
    EdgeType.CO_MOTION,
    EdgeType.CONTRAST,
    EdgeType.PERSISTENCE,
    EdgeType.TOPOLOGY,
    EdgeType.DEGREE,
    EdgeType.CONTACT,
    EdgeType.ORIENTATION_CONTINUITY,
    EdgeType.SURFACE_GROUPING,
    EdgeType.CURVATURE,
    EdgeType.OCCLUSION,
    EdgeType.CO_LOCATED,
    EdgeType.ORIENTATION_BEND,
    EdgeType.ORIENTATION_JUNCTION,
    # [PHASE B] B.11: all six higher-tier types are added IMMEDIATELY —
    # they are structural containment relations, not onset transients.
    # Without membership, consolidator.prune_path strips them from ACTION
    # cores and the higher tier never reaches a crystallized concept.
    EdgeType.CONTOUR_CONTINUITY,
    EdgeType.JUNCTION_CONTOUR,
    EdgeType.CURVE_CONTAINMENT,
    EdgeType.PART_CONTAINMENT,
    EdgeType.PART_CONTENT,
    EdgeType.CONTOUR_ORIENTATION,
    EdgeType.JUNCTION_ORIENTATION,
    # [V9 B.8] 29-32: geometric/relational structure, not onset
    # transients -- the same reasoning as the six higher-tier types.
    EdgeType.AMODAL_LINK,
    EdgeType.COCIRCULAR,
    EdgeType.PARALLEL,
    EdgeType.CO_LOCATED_CHANNEL,
    EdgeType.ILLUSORY_CONTINUITY,
    EdgeType.SYMMETRY,
    EdgeType.BORDER_OWNERSHIP_EDGE,
    EdgeType.CURVATURE_SIGN_CONTINUITY,
    EdgeType.SHARED_BOUNDARY,
    # [Pass 23] all 10 new types are reproducible
    EdgeType.CONCENTRICITY,
    EdgeType.TEXTURE_BOUNDARY_EDGE,
    EdgeType.SURFACE_FOLD,
    EdgeType.RADIALITY,
    EdgeType.SPIRALITY,
    EdgeType.COLOR_OPPONENT_BORDER,
    EdgeType.MOTION_OPPOSITION,
    EdgeType.CROSS_SCALE_IDENTITY,
    EdgeType.BINOCULAR_DISPARITY,
    EdgeType.LINE_TERMINATION,
    EdgeType.SKELETAL_JOINT,
})

HEBBIAN_BOOST_TYPES: FrozenSet[EdgeType] = frozenset({
    EdgeType.CO_ACTIVATION,
    EdgeType.CO_MOTION,
    EdgeType.CONTRAST,
    EdgeType.PERIODICITY,
    EdgeType.PERSISTENCE,
    EdgeType.TOPOLOGY,
    EdgeType.DEGREE,
    EdgeType.CONTACT,
    EdgeType.SYNCHRONY,
    EdgeType.ORIENTATION_CONTINUITY,
    EdgeType.SURFACE_GROUPING,
    EdgeType.CURVATURE,
    EdgeType.OCCLUSION,
    EdgeType.CO_LOCATED,
    EdgeType.ORIENTATION_BEND,
    EdgeType.ORIENTATION_JUNCTION,
    # [PHASE B] B.11: all six higher-tier types boost — they are
    # corroborated by the structure they describe, which is exactly the
    # evidence the Hebbian term is meant to reinforce.
    EdgeType.CONTOUR_CONTINUITY,
    EdgeType.JUNCTION_CONTOUR,
    EdgeType.CURVE_CONTAINMENT,
    EdgeType.PART_CONTAINMENT,
    EdgeType.PART_CONTENT,
    EdgeType.CONTOUR_ORIENTATION,
    EdgeType.JUNCTION_ORIENTATION,
    # [V9 B.8] corroborated by the geometry they describe.
    EdgeType.AMODAL_LINK,
    EdgeType.COCIRCULAR,
    EdgeType.PARALLEL,
    EdgeType.CO_LOCATED_CHANNEL,
    EdgeType.ILLUSORY_CONTINUITY,
    EdgeType.SYMMETRY,
    EdgeType.BORDER_OWNERSHIP_EDGE,
    EdgeType.CURVATURE_SIGN_CONTINUITY,
    EdgeType.SHARED_BOUNDARY,
    # [Pass 23] geometrically corroborated
    EdgeType.CONCENTRICITY,
    EdgeType.TEXTURE_BOUNDARY_EDGE,
    EdgeType.SURFACE_FOLD,
    EdgeType.RADIALITY,
    EdgeType.SPIRALITY,
    EdgeType.COLOR_OPPONENT_BORDER,
    EdgeType.MOTION_OPPOSITION,
    EdgeType.CROSS_SCALE_IDENTITY,
    EdgeType.BINOCULAR_DISPARITY,
    EdgeType.LINE_TERMINATION,
    EdgeType.SKELETAL_JOINT,
})


# --- from sensory_floor.py: typed signatures ---
_SIM_MODE = P.SIMILARITY_MODE

def _typed_signature_bytes(sig: "TypedSignature") -> bytes:
    if sig.kind == "CANONICAL":
        parts = [
            serialize_int(0),
            serialize_bytes(sig.class_.encode()),
            serialize_bytes(sig.relation.encode()),
            serialize_tuple([serialize_float(f) for f in sig.quantized]),
        ]
        return serialize_tuple(parts)
    elif sig.kind == "OPAQUE":
        parts = [
            serialize_int(1),
            serialize_bytes(sig.type_tag.encode()),
            serialize_int(sig.hash_value),
        ]
        return serialize_tuple(parts)
    else:
        raise ValueError(f"Unknown sig kind: {sig.kind}")

@dataclass(frozen=True)
class TypedSignature:
    kind: str
    type_tag: str = ""
    hash_value: int = 0
    class_: str = ""
    relation: str = ""
    quantized: Tuple[float, ...] = ()

    def __post_init__(self):
        b = _typed_signature_bytes(self)
        object.__setattr__(self, "_bytes_cache", b)
        object.__setattr__(self, "_hash_cache", hash_64_signed(b))

    @staticmethod
    def opaque(type_tag: str, hash_value: int) -> "TypedSignature":
        return TypedSignature(kind="OPAQUE", type_tag=type_tag, hash_value=hash_value)

    def __hash__(self) -> int:
        return self._hash_cache

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypedSignature):
            return False
        if self.kind != other.kind:
            return False
        if self.kind == "CANONICAL":
            return (
                self.class_ == other.class_
                and self.relation == other.relation
                and self.quantized == other.quantized
            )
        return self.type_tag == other.type_tag and self.hash_value == other.hash_value

    def serialize(self) -> bytes:
        return self._bytes_cache

    def compatible(self, other) -> bool:
        """Structural compatibility (§A.6): CANONICAL signatures match on
        class, relation, and quantized arity regardless of values."""
        if self.kind == "CANONICAL" and getattr(other, "kind", None) == "CANONICAL":
            return (
                self.class_ == other.class_
                and self.relation == other.relation
                and len(self.quantized) == len(other.quantized)
            )
        return False

    def similarity(self, other, context: Any = None) -> float:
        """Phase A dispatcher: Gaussian (default) or legacy linear L1.

        SIMILARITY_MODE is snapshotted at import (see _SIM_MODE). Both
        branches preserve the §2.21.5 contract (OPAQUE hash equality,
        CANONICAL class/relation/arity gate, unregistered class → 0.0).
        """
        return _SIMILARITY_IMPL(self, other, context=context)

# --- from sensory_floor.py: detections ---
_SIGMA_FALLBACK_RANGE0 = P.SIGMA_FALLBACK_FOR_RANGE0

_SIM_MODE = P.SIMILARITY_MODE

def _similarity_gaussian(self, other, context: Any = None) -> float:
    other_kind = getattr(other, "kind", None)
    if self.kind == "OPAQUE" and other_kind == "OPAQUE":
        return 1.0 if self == other else 0.0
    if self.kind == "CANONICAL" and other_kind == "CANONICAL":
        if self.class_ != other.class_ or self.relation != other.relation:
            return 0.0
        if len(self.quantized) != len(other.quantized):
            return 0.0
        if self.quantized == other.quantized:
            return 1.0
        ranges = get_similarity_ranges(self.class_)
        if ranges is None or len(ranges) != len(self.quantized):
            return 0.0
        cycle_lengths = CIRCULAR_CYCLE_LENGTHS.get(self.class_)
        weights = get_slot_weights(self.class_, len(ranges), context=context)
        W_sum = float(sum(weights)) if weights else float(len(self.quantized))
        if W_sum <= 0.0:
            W_sum = float(len(self.quantized)) or 1.0

        total_sq = 0.0
        for i, (a, b) in enumerate(zip(self.quantized, other.quantized)):
            d = abs(a - b)
            if cycle_lengths and i in cycle_lengths:
                n = cycle_lengths[i]
                d = min(d, n - d)
            if self.class_ == "REGION_SHAPE" and i == 19:
                if (a >= 12.0 or b >= 12.0 or
                        (len(self.quantized) > 18 and (self.quantized[18] < 2.0 or other.quantized[18] < 2.0))):
                    d = 0.0
            if ranges[i] > 0.0:
                sigma = ranges[i] / 3.0
            else:
                sigma = _SIGMA_FALLBACK_RANGE0
            w = weights[i] if (weights and i < len(weights)) else 1.0
            total_sq += (w / W_sum) * (d / sigma) ** 2
        return math.exp(-total_sq)
    return 0.0

def _similarity_linear(self, other, context: Any = None) -> float:
    other_kind = getattr(other, "kind", None)
    if self.kind == "OPAQUE" and other_kind == "OPAQUE":
        return 1.0 if self == other else 0.0
    if self.kind == "CANONICAL" and other_kind == "CANONICAL":
        if self.class_ != other.class_ or self.relation != other.relation:
            return 0.0
        if len(self.quantized) != len(other.quantized):
            return 0.0
        if self.quantized == other.quantized:
            return 1.0
        ranges = get_similarity_ranges(self.class_)
        if ranges is None or len(ranges) != len(self.quantized):
            return 0.0
        weights = get_slot_weights(self.class_, len(ranges), context=context)
        cycle_lengths = CIRCULAR_CYCLE_LENGTHS.get(self.class_)
        total_diff = 0.0
        for i, (a, b) in enumerate(zip(self.quantized, other.quantized)):
            d = abs(a - b)
            if cycle_lengths and i in cycle_lengths:
                n = cycle_lengths[i]
                d = min(d, n - d)
            if self.class_ == "REGION_SHAPE" and i == 19:
                if (a >= 12.0 or b >= 12.0 or
                        (len(self.quantized) > 18 and (self.quantized[18] < 2.0 or other.quantized[18] < 2.0))):
                    d = 0.0
            total_diff += weights[i] * d
        total_weight = sum(w * r for w, r in zip(weights, ranges))
        return max(0.0, 1.0 - total_diff / max(total_weight, EPSILON))
    return 0.0

_SIMILARITY_IMPL = (
    _similarity_gaussian if _SIM_MODE == "gaussian" else _similarity_linear
)

def _serialize_any(value: Any) -> bytes:
    if isinstance(value, bool):
        return serialize_int(int(value))
    if isinstance(value, int):
        return serialize_int(value)
    if isinstance(value, str):
        return serialize_bytes(value.encode())
    if isinstance(value, bytes):
        return serialize_bytes(value)
    if isinstance(value, float):
        return serialize_float(value)
    if isinstance(value, TypedSignature):
        return value.serialize()
    if isinstance(value, tuple):
        return serialize_tuple([_serialize_any(v) for v in value])
    if isinstance(value, (frozenset, set, list)):
        return serialize_set([_serialize_any(v) for v in value])
    raise TypeError(f"Unsupported detection value type: {type(value).__name__}")

def serialize_detection(det: "Detection") -> bytes:
    return serialize_tuple(
        [
            serialize_bytes(det.k.encode()),
            serialize_int(det.source_layer),
            serialize_int(det.scale),
            serialize_bytes(det.sigma.serialize()),
            _serialize_any(det.alpha),
            serialize_float(det.z),
            serialize_int(det.t),
            _serialize_any(det.rho),
            serialize_int(det.substrate_hash),
        ]
    )

class Detection:
    """Frozen value object for lens/binder-emitted detections.

    Deliberately NOT a @dataclass. The dataclass decorator's interaction
    with class-body __hash__ / __eq__ has produced a tp_hash desync on
    Python 3.11.9 where hash(d) dispatches to a slot wrapper instead of
    the class-dict __hash__, silently bypassing the cached serialized
    hash. An explicit class with __slots__ guarantees the declared
    methods are the ones actually invoked by hash() and ==.
    """

    __slots__ = (
        "k",
        "source_layer",
        "scale",
        "sigma",
        "alpha",
        "z",
        "t",
        "rho",
        "substrate_hash",
        "_hash_cache",
        "_curv_profile",
    )

    def __init__(
        self,
        k: str,
        source_layer: int,
        scale: int,
        sigma: TypedSignature,
        alpha: Any,
        z: float,
        t: int,
        rho: Tuple[Tuple, ...],
        substrate_hash: int = 0,
    ):
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "source_layer", source_layer)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "substrate_hash", substrate_hash)
        object.__setattr__(self, "_curv_profile", None)
        object.__setattr__(
            self, "_hash_cache", hash_64_signed(serialize_detection(self))
        )

    def __setattr__(self, name, value):
        raise AttributeError(f"Detection is immutable; cannot set attribute {name!r}")

    def __delattr__(self, name):
        raise AttributeError(
            f"Detection is immutable; cannot delete attribute {name!r}"
        )

    def __hash__(self) -> int:
        return self._hash_cache

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Detection):
            return NotImplemented
        return (
            self.k == other.k
            and self.source_layer == other.source_layer
            and self.scale == other.scale
            and self.sigma == other.sigma
            and self.alpha == other.alpha
            and abs(self.z - other.z) < 1e-9
            and self.t == other.t
            and self.rho == other.rho
            and self.substrate_hash == other.substrate_hash
        )

    def __repr__(self) -> str:
        return (
            f"Detection(k={self.k!r}, source_layer={self.source_layer}, "
            f"scale={self.scale}, z={self.z}, t={self.t}, "
            f"substrate_hash={self.substrate_hash})"
        )

