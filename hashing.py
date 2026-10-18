"""core/hashing.py -- pure serialization, hashing and numeric utilities.

MOVE ONLY. Function bodies extracted VERBATIM from primitives.py by _reorg_move1_core.py, so the hashing contract is
unchanged. primitives.py re-exports from here. Zero project imports by
contract.
"""
from __future__ import annotations

import hashlib
import math
import struct
from typing import Any, Dict, List

_SERIALIZE_INT_CACHE = {}
_SERIALIZE_INT_CACHE_MAX = 1_000_000


def serialize_int(x: int) -> bytes:
    x = int(x)
    cached = _SERIALIZE_INT_CACHE.get(x)
    if cached is not None:
        return cached
    if -(2**63) <= x < 2**63:
        data = struct.pack("<q", x)
    elif 0 <= x < 2**64:
        data = struct.pack("<Q", x)
    else:
        raise ValueError(f"Integer out of range for 64-bit serialization: {x}")
    if len(_SERIALIZE_INT_CACHE) >= _SERIALIZE_INT_CACHE_MAX:
        _SERIALIZE_INT_CACHE.clear()
    _SERIALIZE_INT_CACHE[x] = data
    return data

def serialize_float(x: float) -> bytes:
    """Fixed-point float serialization at 1e-6 resolution.

    Non-finite floats are rejected. The system's values are bounded, but
    a defensive guard catches upstream NaN/inf at the boundary rather than
    letting it propagate into a confusing failure inside math.floor.

    Note: serialize_float(0.0) and serialize_int(0) produce identical bytes.
    Callers know which type they are passing; the collision is theoretical.
    A future revision that requires type-tagged serialization would need
    to add a tag to both functions consistently.
    """
    if math.isnan(x) or math.isinf(x):
        raise ValueError(f"Cannot serialize non-finite float: {x}")
    if x >= 0:
        scaled = math.floor(x * 1_000_000 + 0.5)
    else:
        scaled = math.ceil(x * 1_000_000 - 0.5)
    return serialize_int(int(scaled))

def serialize_bytes(b: bytes) -> bytes:
    """Length-prefixed byte-string serialization.

    Note: serialize_bytes(b"") and serialize_int(0) produce identical bytes
    (both are 8 zero bytes). Callers know which type they are passing; the
    collision is theoretical. A future revision that requires type-tagged
    serialization would need to add a tag to both functions consistently.
    """
    return serialize_int(len(b)) + b

def serialize_tuple(elems: List[bytes]) -> bytes:
    """Length-prefixed tuple serialization.

    Output = serialize_int(len(elems)) + b"".join(elems). Unambiguous:
    no two distinct lists of bytes can produce the same output. This
    replaces the v30 scheme (empty -> b'\\x00', otherwise join), which had
    a theoretical collision between the empty tuple and a singleton tuple
    containing b'\\x00'.

    Every tuple-derived hash in v31 uses this encoding. Do not change it
    after v31 hashing is in production.
    """
    return serialize_int(len(elems)) + b"".join(elems)

def serialize_set(elems: List[bytes]) -> bytes:
    """Length-prefixed, order-independent set serialization.

    Elements are sorted lexicographically before joining, so set iteration
    order does not affect the output. Length-prefixed for the same
    ambiguity-elimination reason as serialize_tuple.
    """
    return serialize_int(len(elems)) + b"".join(sorted(elems))

def serialize_value(value: Any) -> bytes:
    """Universal deterministic serialization for heterogeneous role values."""
    if isinstance(value, bool):
        return serialize_int(1 if value else 0)
    elif isinstance(value, int):
        return serialize_int(value)
    elif isinstance(value, float):
        return serialize_float(value)
    elif isinstance(value, str):
        return serialize_bytes(value.encode("utf-8"))
    elif isinstance(value, bytes):
        return serialize_bytes(value)
    elif hasattr(value, "serialize") and callable(value.serialize):
        return value.serialize()
    elif isinstance(value, tuple):
        return serialize_tuple([serialize_value(v) for v in value])
    elif isinstance(value, (set, frozenset)):
        return serialize_set([serialize_value(v) for v in value])
    elif isinstance(value, list):
        return serialize_tuple([serialize_value(v) for v in value])
    elif value is None:
        return b"\x00"
    else:
        return serialize_bytes(f"{type(value).__name__}:{value}".encode("utf-8"))

def hash_64(data: bytes) -> int:
    """Canonical 64-bit unsigned little-endian SHA-256 slice.

    Return range is [0, 2^64 - 1]. Note that Python's `hash()` builtin
    operates on signed Py_hash_t (= signed 64-bit on all mainstream
    platforms), so any cached value intended to be returned from
    `__hash__` must be masked via `hash_64_signed` to fit [0, 2^63 - 1].

    Used for: OPAQUE TypedSignature hash_value (§4.13), tie-breaker
    hashes in the current architecture, and all legacy v30 callers.
    """
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little", signed=False)

def hash_64_signed(data: bytes) -> int:
    """64-bit hash masked to fit in Py_ssize_t (positive signed 64-bit).

    Use this for any value that will be returned from `__hash__`. Using
    the raw `hash_64` for that purpose causes silent truncation by
    CPython's `PyLong_AsSsize_t`, breaking the contract
    `hash(obj) == obj._hash_cache`.
    """
    return hash_64(data) & 0x7FFFFFFFFFFFFFFF

def hash_128(data: bytes) -> bytes:
    """Canonical 128-bit byte hash: first 16 bytes of SHA-256.

    Returns a fixed-length byte value suitable for dictionary keys,
    comparison, and serialization. Unlike hash_64 (which returns an
    int), hash_128 is a byte value so it composes cleanly with the
    spec's canonical-hash-as-bytes convention (§4.12, §4.13, §4.14).

    Used exclusively for:
      - Canonical subgraph hashing (WL refinement output, §4.12)
      - Detection-set hashing (cross_section_key, §4.14)
      - CANONICAL signature node-identity hashing (§4.13)

    NOT used to construct TypedSignature.hash_value. That value is
    always produced by hash_64 (§4.13 "Hash function terminology").

    Collision bound at 2^-128: with L entries, P_any_random_collision
    <= L^2 * 2^-128. For L = 10^6, P ~ 10^-26. Collisions are treated
    as terminal errors, not engineered around (§4.12).
    """
    return hashlib.sha256(data).digest()[:16]

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def heaviside(x: float) -> int:
    return 1 if x >= 0 else 0


# --- moved from binder.py (Move 3c): pure helpers that the derived
# lenses need, so they no longer import binder at all -----------
def _detection_sort_key(d: Detection) -> Tuple[Any, ...]:
    return (
        d.k,
        _node_signature_hash(d.sigma),
        serialize_value(d.alpha),
        d.t,
    )

def _node_signature_hash(sigma: TypedSignature) -> bytes:
    """§4.13: OPAQUE → serialize_int(hash_value); CANONICAL → hash_128."""
    if sigma.kind == "OPAQUE":
        return serialize_int(sigma.hash_value)
    if sigma.kind == "CANONICAL":
        return hash_128(sigma.serialize())
    raise ValueError(f"Unknown sigma kind: {sigma.kind}")

