"""perception/lenses/derived_common.py -- shared constants and helpers for the derived lenses (Move 3.5, part 2)."""
from __future__ import annotations

import numpy as np
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Sequence
from typing import TYPE_CHECKING
import parameters as P
from concept_graph import (  # noqa: F401
    ARITY_MAX,
    Concept,
    ConceptStatus,
    EPSILON,
    Edge,
    KAPPA_BIRTH,
    TAU_EDGE_INACTIVE,
    TAU_OP_COOLDOWN,
    TAU_REVERT,
    THETA_EDGE_ACTIVE,
    THETA_HYST_OP,
)
from core.hashing import (  # noqa: F401
    _detection_sort_key,
    clamp,
    hash_64,
    serialize_bytes,
    serialize_float,
    serialize_int,
    serialize_set,
    serialize_tuple,
    serialize_value,
)
from core.quantize import carrier_to_tangent  # noqa: F401
from core.registries import (  # noqa: F401
    CONSTITUTIONAL_RANGES,
)
from core.types import (  # noqa: F401
    Detection,
    EdgeType,
    TypedSignature,
)
from learning.fertility import (  # noqa: F401
    KAPPA_EMA,
    RF_MAX,
)
from memory.cores import (  # noqa: F401
    _sig_similarity,
)
from core.spatial import _pos_dist  # noqa: F401
from parameters._text import (  # noqa: F401
    BO_TEXTURE_BIN_MAX,
    B_MAX,
    DELTA_CAT,
    DELTA_THETA_CAT_MAX,
    KAPPA_CAT,
    K_CATEGORIZE,
    RHO_CAT_TARGET,
    RHO_PROGRESS_MIN,
    TAU_PROGRESS,
    THETA_CATEGORIZE,
    THETA_CATEGORIZE_MAX,
    THETA_CATEGORIZE_MIN,
    THETA_SIM,
    W_CHUNK,
    W_PLAN_CHUNK,
)
from perception.depth import (  # noqa: F401
    _extract_member_anchors,
)
from perception.filters import (  # noqa: F401
    _TINY,
)

"""perception/lenses/derived.py -- extracted from derived.py
(Move 3.5).

MOVE ONLY: whole top-level statements, verbatim.
"""

"""
derived_lenses.py — lenses that read from the concept graph.

Under v31's flat extraction taxonomy (§2.1–§2.3), every extractor falls
into one of three categories:

    Lens         — one entity, one moment        → node
    Cross-binder — two or more entities          → edge

This module contains the lenses whose source is the concept graph
itself rather than raw observations. They emit Detection nodes exactly
like raw lenses; the only difference is what they read.

The v30 "higher lens" model — layered lenses composing on top of lower
lenses — is retired. Composition in v31 happens through binding: edges
capture relations between nodes, and crystallization forms concepts
from bound subgraphs. There is no lens hierarchy.

Contents:

  - Recognition lens (step_13_recognition): matches raw detections
    against concept evidence registries; emits Recognition detections
    consumed as activation evidence (z_rec).

  - Operator emission lens (step_14_operator_emission): promoted
    concepts emit OPERATOR_PATTERN detections (§11.2). These flow
    through standard binding; this module does not crystallize.

  - Chunk emission lens (step_15/16): co-activation clusters emit
    CHUNK detections (§5.10). Fast approximate consolidation.

  - Promotion gating: eligibility and scoring for operator promotion.

Removed from v30 (absorbed by the flat taxonomy):

  ConceptSequence   → SEQUENCE binder
  ConceptDegree     → DEGREE binder
  Convergence       → node in-degree, observable directly
  Divergence        → node out-degree, observable directly

The deleted lenses were concept-derived duplicates of edge-emitters.
Under the flat taxonomy they are redundant: the same relation is already
captured as an edge by the corresponding binder. Causality,
in particular, is now expressed as edge structure — a SEQUENCE edge
(temporal priority) inside a crystallizing candidate. Nothing is reified.
"""

EPSILON = 1e-9

# --- Operator emission (§2.28) ---
THETA_PROMO_OP = 0.5

RHO_PROMO_RF = 0.1

SIGMA_MAX_OP = 10

TAU_PROMO_MIN = 200

TAU_OP_COOLDOWN = 200

K_OP = 10

Y_FLOOR_OP = 0.01

THETA_FLOOR_OP = 0.05

ETA_CTX_OP = 0.05

ETA_FAT_OP = 0.05

ETA_FAT_DECAY_OP = 0.95

DELTA_OP = 0.02

THETA_HYST_OP = 0.5

KAPPA_OP = 0.05

RHO_OP_TARGET = 0.2

KAPPA_BIRTH = 0.05

RHO_BIRTH = 0.1

THETA_SIM = 0.7

# --- Chunk emission (§5.10) ---
W_CHUNK = 20

W_PLAN_CHUNK = 100

K_CHUNK_CAND = 20

K_CHUNK_PLAN = math.ceil(K_CHUNK_CAND / 3)

K_CHUNK_STAR = math.ceil(K_CHUNK_CAND / 3)

THETA_CHUNK_EMIT = 0.5

RHO_CHUNK_TARGET = 0.1

KAPPA_CHUNK_HOMEO = 0.05

THETA_CHUNK_EMIT_MIN = 0.1

THETA_CHUNK_EMIT_MAX = 0.9

D_CHUNK_MAX = 3

DELTA_CHUNK = 0.02

B_MAX = 3

THETA_CATEGORIZE = P.THETA_CATEGORIZE

K_CATEGORIZE = P.K_CATEGORIZE

KAPPA_CAT = P.KAPPA_CAT

RHO_CAT_TARGET = P.RHO_CAT_TARGET

THETA_CATEGORIZE_MIN = P.THETA_CATEGORIZE_MIN

THETA_CATEGORIZE_MAX = P.THETA_CATEGORIZE_MAX

DELTA_THETA_CAT_MAX = P.DELTA_THETA_CAT_MAX

TAU_PROGRESS = P.TAU_PROGRESS

RHO_PROGRESS_MIN = P.RHO_PROGRESS_MIN

TAU_REVERT = P.TAU_REVERT

DELTA_CAT = P.DELTA_CAT

KAPPA_EMA = 0.05

# --- Active-edge predicate (§1.12) ---
THETA_EDGE_ACTIVE = 0.1

TAU_EDGE_INACTIVE = 500

# --- Promotion breadth/fertility ---
ARITY_MAX = 3

RF_MAX = 10**9

def _sig_compatible(sig1: TypedSignature, sig2: TypedSignature) -> bool:
    """Structural compatibility (§A.4)."""
    if sig1.kind == "CANONICAL" and sig2.kind == "CANONICAL":
        return (
            sig1.class_ == sig2.class_
            and sig1.relation == sig2.relation
            and len(sig1.quantized) == len(sig2.quantized)
        )
    return False

def _sig_similarity(
    sig1: Optional[TypedSignature], sig2: Optional[TypedSignature]
) -> float:
    """Delegate to TypedSignature.similarity() (§2.21.5)."""
    if sig1 is None or sig2 is None:
        return 0.0
    return sig1.similarity(sig2)

def _compat_key(sig: TypedSignature) -> Optional[Tuple]:
    """Compatibility group key (§A.4 _sig_compatible, hashed form).
    Only CANONICAL signatures are ever compatible; OPAQUE (e.g. Action
    lens) maps to None — exact membership is its only route to a match.
    """
    if sig.kind != "CANONICAL":
        return None
    return ("CANONICAL", sig.class_, sig.relation, len(sig.quantized))

# Exact-keyed similarity memo (canonical serialized bytes on both sides —
# no hash-collision risk, C4 determinism preserved). Detection and
# evidence signatures recur heavily across steps in stable scenes; the
# graded L1 similarity is pure, so memoization is semantics-preserving.
_SIM_MEMO: Dict[Tuple[bytes, bytes], float] = {}

_SIM_MEMO_CAP = 200_000

def _sim_memo(a: TypedSignature, b: TypedSignature) -> float:
    key = (a.serialize(), b.serialize())
    val = _SIM_MEMO.get(key)
    if val is not None:
        return val
    val = a.similarity(b)
    if len(_SIM_MEMO) >= _SIM_MEMO_CAP:
        _SIM_MEMO.clear()  # bounded; deterministic (insertion-order clear)
    _SIM_MEMO[key] = val
    return val

if TYPE_CHECKING:  # pragma: no cover
    from concept_graph import Concept, ConceptStatus, Edge  # noqa: F401

# [PHASE B] B.2: the higher-tier lens registry. Deliberately NOT added to
# lenses._LENS_NAMES — these are a second tier that consumes V1 output, not
# peers of it, and adding them there would make them run inside the V1
# lens executor (and enter D_raw).
HIGHER_LENS_NAMES: Tuple[str, ...] = (
    "BorderOwnership", "Contour", "Junction",
    "TextureBoundary", "Surface", "RegionShape", "IllusoryContour",
    "HolisticShape", "ShockEnsemble",
)

BO_CURVATURE_WEIGHT = getattr(P, "BO_CURVATURE_WEIGHT", 0.30)

BO_TEXTURE_WEIGHT = getattr(P, "BO_TEXTURE_WEIGHT", 0.30)

BO_COLOR_WEIGHT = getattr(P, "BO_COLOR_WEIGHT", 0.20)

BO_GRAVITY_WEIGHT = getattr(P, "BO_GRAVITY_WEIGHT", 0.20)

BO_CUE_SCALE = getattr(P, "BO_CUE_SCALE", 2.0)

BO_GRAVITY_SCALE = getattr(P, "BO_GRAVITY_SCALE", 4.0)

BO_TEXTURE_BIN_MAX = getattr(P, "BO_TEXTURE_BIN_MAX", 9.0)

BO_RICHNESS_FALLBACK = getattr(P, "BO_RICHNESS_FALLBACK", 0.0)

# Import-time guard to catch arity/range drift between primitives and parameters
_CONTOUR_RANGES = CONSTITUTIONAL_RANGES["CONTOUR"]

if (
    len(_CONTOUR_RANGES) != 7 + P.CONTOUR_N_SEQ
    or any(r != P.CONTOUR_SEQ_LEVELS - 1 for r in _CONTOUR_RANGES[7:])
    or P.ORIENTATION_N_THETA % 2 != 0
):
    raise RuntimeError(
        "[B5] CONSTITUTIONAL_RANGES['CONTOUR'] is out of sync with "
        "CONTOUR_N_SEQ / CONTOUR_SEQ_LEVELS / ORIENTATION_N_THETA"
    )

_THETA_FRAME_SIGN = 1

def _signed_log_level(v: float, n_levels: int, v_ref: float) -> int:
    """Antisymmetric signed-log quantizer: level(-v) == (n_levels-1) - level(v)."""
    half = (n_levels - 1) // 2
    a = min(abs(v), math.pi)
    u = math.log1p(a / v_ref) / math.log1p(math.pi / v_ref)
    m = int(math.floor(u * half + 0.5))
    return half - m if v < 0.0 else half + m

def _contour_turns(bins: List[int], periodic: bool, pts: Optional[List[Tuple[float, float]]]):
    """Per-corner (mags in orientation-bin units, signed turns in radians)."""
    n = len(bins)
    m = n if periodic else n - 1
    n_theta = P.ORIENTATION_N_THETA
    half_cycle = n_theta // 2
    unit = math.pi / n_theta
    band_lo = half_cycle - P.CONTOUR_GEOM_BAND_BINS
    tangents = None
    mags, turns = [], []
    for k in range(m):
        j = (k + 1) % n
        d = (bins[j] - bins[k]) % n_theta
        if d > half_cycle:
            d -= n_theta
        mag = abs(d)
        if pts is not None and mag >= band_lo:
            if tangents is None:
                tangents = _oriented_tangents(bins, pts, periodic)
            tu, tw = tangents[k], tangents[j]
            if tu is not None and tw is not None:
                cross = tu[0] * tw[1] - tu[1] * tw[0]
                dot = tu[0] * tw[0] + tu[1] * tw[1]
                ang = math.atan2(cross, dot)
                turns.append(ang)
                mags.append(int(math.floor(abs(ang) / unit + 0.5)))
                continue
        turns.append(0.0 if 2 * mag == n_theta else d * unit)
        mags.append(mag)
    return mags, turns

def _turn_positions(n: int, periodic: bool, pts: Optional[List[Tuple[float, float]]]) -> List[float]:
    m = n if periodic else n - 1
    gaps = None
    if pts is not None:
        gaps = [math.hypot(pts[(k + 1) % n][0] - pts[k][0],
                           pts[(k + 1) % n][1] - pts[k][1]) for k in range(m)]
        if sum(gaps) <= 1e-9:
            gaps = None
    if gaps is None:
        gaps = [1.0] * m
    total = math.fsum(gaps)
    x, s = [], 0.0
    for g in gaps:
        x.append((s + 0.5 * g) / total)
        s += g
    return x

def _splat(x, t, n_seq: int, periodic: bool = True):
    seq = np.zeros(n_seq)
    u = np.asarray(x, dtype=float) * n_seq - 0.5
    j0 = np.floor(u).astype(np.int64)
    f = u - j0
    lo, hi = j0, j0 + 1
    if periodic:
        lo, hi = lo % n_seq, hi % n_seq
    else:
        lo, hi = np.clip(lo, 0, n_seq - 1), np.clip(hi, 0, n_seq - 1)
    np.add.at(seq, lo, t * (1.0 - f))
    np.add.at(seq, hi, t * f)
    return seq

def _canonical_curvature_sequence(mags, turns, x, periodic: bool = True):
    """Return (encoded_levels, canonical_signed_total_turn_radians)."""
    n_seq, levels = P.CONTOUR_N_SEQ, P.CONTOUR_SEQ_LEVELS
    v_ref = math.pi / P.ORIENTATION_N_THETA
    X, T = np.asarray(x, dtype=float), np.asarray(turns, dtype=float)

    def encode(xs, ts):
        s = _splat(xs, ts, n_seq, periodic)
        return tuple(_signed_log_level(float(v), levels, v_ref) for v in s)

    def total(ts):
        return float(math.fsum(float(v) for v in ts))

    if len(X) == 0 or len(T) == 0:
        return tuple([levels // 2] * n_seq), 0.0
    if periodic:
        views = ((X, T), ((1.0 - X) % 1.0, -T))
        top = max(mags) if len(mags) > 0 else 0
        cands = [c for c, g in enumerate(mags) if g == top] if len(mags) > 0 else [0]
        best = None
        best_total = total(T)
        for xs, ts in views:
            for c in cands:
                q = encode((xs - xs[c]) % 1.0, ts)
                if best is None or q > best:
                    best = q
                    best_total = total(ts)
        return (best if best is not None else tuple([levels // 2] * n_seq)), best_total
    q0, q1 = encode(X, T), encode(1.0 - X, -T)
    if q0 >= q1:
        return q0, total(T)
    return q1, total(-T)

def _contour_slots(bins: List[int], closed: bool, pts: Optional[List[Tuple[float, float]]]) -> Tuple[float, ...]:
    n = len(bins)
    n_theta = P.ORIENTATION_N_THETA
    unit = math.pi / n_theta
    periodic = bool(closed) and n >= 3

    ang2 = [2.0 * (b + 0.5) * unit for b in bins]
    s2 = math.fsum(math.sin(a) for a in ang2) / n
    c2 = math.fsum(math.cos(a) for a in ang2) / n
    R = math.hypot(s2, c2)
    if R > 1e-9:
        mean_theta = (0.5 * math.atan2(s2, c2)) % math.pi
        mean_bin = int(math.floor(mean_theta * n_theta / math.pi)) % n_theta
    else:
        mean_bin = 0

    circ_var_bin = min(7, int((1.0 - R) * 8.0))

    mags, turns = _contour_turns(bins, periodic, pts)
    x = _turn_positions(n, periodic, pts)
    seq, canonical_total = _canonical_curvature_sequence(mags, turns, x, periodic)

    total = sum(mags) * unit
    total_bin = min(7, int(8.0 * math.log1p(total) / math.log1p(P.CONTOUR_TOTAL_TURN_MAX)))
    corners = sum(1 for g in mags if g > P.CONTOUR_CORNER_TURN_FRAC * n_theta)
    chir = max(-1.0, min(1.0, canonical_total / (2.0 * math.pi)))
    chirality_bin = min(7, max(0, int(math.floor((chir + 1.0) * 4.0))))

    return (
        float(min(7, n // 4)), float(mean_bin), float(circ_var_bin),
        float(1 if closed else 0), float(total_bin), float(min(7, corners)),
        float(chirality_bin),
    ) + tuple(float(q) for q in seq)

_JUNCTION_CANONICAL_BINS = 64

def qi(v: float, n: int, lo: float = 0.0, hi: float = 1.0) -> float:
    if not math.isfinite(v):
        return float(n // 2)
    u = (min(max(v, lo), hi) - lo) / (hi - lo)
    return float(min(n - 1, int(u * n)))

def qi_circ(v: float, period: float, n: int) -> float:
    return float(int(((v % period) / period) * n) % n)

def circ_mean_bin(bins: List[float], n: int) -> float:
    if not bins:
        return 0.0
    s = sum(math.sin(2 * math.pi * (b + 0.5) / n) for b in bins)
    c = sum(math.cos(2 * math.pi * (b + 0.5) / n) for b in bins)
    ang = math.atan2(s, c) % (2 * math.pi)
    return float(int((ang / (2 * math.pi)) * n) % n)

def texture_richness(det: Optional[Detection]) -> float:
    if det is None:
        return 0.0
    q, c = det.sigma.quantized, det.sigma.class_
    if c.startswith("TEXTURE_RICH_"):
        if len(q) >= 4:
            return float(q[-2]) * (1.0 - float(q[-4]))
        return 0.0
    if len(q) >= 2:
        return (float(q[0]) / BO_TEXTURE_BIN_MAX) * (1.0 - float(q[1]) / BO_TEXTURE_BIN_MAX)
    return 0.0

_TINY = 1e-12

def _resample_closed(pts, n: int):
    P_ = np.asarray(pts, np.float64)
    if P_.ndim != 2 or P_.shape[0] < 3 or P_.shape[1] < 2:
        return None
    seg = np.hypot(*(np.roll(P_, -1, axis=0) - P_).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    tot = cum[-1]
    if tot < _TINY:
        return None
    s = np.linspace(0.0, tot, n, endpoint=False)
    Q = np.concatenate([P_, P_[:1]], axis=0)
    return np.stack([np.interp(s, cum, Q[:, 0]),
                     np.interp(s, cum, Q[:, 1])], axis=1)

def _chamfer(A, B):
    d = np.hypot(A[:, None, 0] - B[None, :, 0],
                 A[:, None, 1] - B[None, :, 1])
    return float(d.min(axis=1).mean())

def contour_self_symmetry(pts):
    R = _resample_closed(pts, P.SYM_RESAMPLE_N)
    if R is None:
        return 0.0, float(P.SYM_AXIS_UNDEFINED_BIN), 0.0
    R = R - R.mean(axis=0)
    rms = math.sqrt(float((R ** 2).sum(axis=1).mean())) + _TINY
    tol = P.SYM_TAU * rms
    scores = []
    for k in range(P.SYM_N_AXES):
        phi = k * math.pi / P.SYM_N_AXES
        c2, s2 = math.cos(2 * phi), math.sin(2 * phi)
        M = np.array([[c2, s2], [s2, -c2]])
        scores.append(max(0.0, 1.0 - _chamfer(R @ M.T, R) / tol))
    best_k = int(np.argmax(scores))
    spread = max(scores) - min(scores)
    axis_bin = (float(P.SYM_AXIS_UNDEFINED_BIN)
                if spread < P.SYM_ISOTROPIC_SPREAD else float(best_k))

    fold = 1
    for k in range(2, 7):
        a = 2 * math.pi / k
        Rk = np.array([[math.cos(a), -math.sin(a)],
                       [math.sin(a), math.cos(a)]])
        if max(0.0, 1.0 - _chamfer(R @ Rk.T, R) / tol) >= P.SYM_FOLD_THRESHOLD:
            fold = k
    return float(qi(max(scores), 8)), float(axis_bin), float(fold - 1)

def _point_in_poly(pt, poly) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside

def reflection_symmetry(kappa, n_axes=8):
    k = np.asarray(kappa, float)
    if k.ndim != 1 or k.size < 2:
        return [0.0] * n_axes
    n = int(k.size)
    den = float(k @ k) + 1e-12
    idx = np.arange(n)
    return [max(0.0, float(k @ k[(m - idx) % n]) / den)
            for m in (np.arange(n_axes) * n // n_axes)]

def _arm_tangent_stats(
    j: int,
    hub_idx: int,
    G_j: Any,
    orient_dets: List[Detection],
    CONTINUITY_NBR: Optional[Dict[int, Set[int]]] = None,
) -> Tuple[int, float]:
    """Support and doubled-angle concentration of the oriented segments on an arm.

    [B21] Support = how many ORIENTATION_CONTINUITY segments the arm is made
    of; sharpness = how well those segments agree in direction (doubled-angle
    resultant, because orientation is axial).
    """
    members = {j} | set((CONTINUITY_NBR or {}).get(j, ()))
    ang2 = []
    for m in members:
        if m >= len(orient_dets):
            continue
        q = orient_dets[m].sigma.quantized
        if len(q) < 1:
            continue
        tan = carrier_to_tangent((q[0] + 0.5) * math.pi / P.ORIENTATION_N_THETA)
        ang2.append(2.0 * tan)
    if not ang2:
        return 0, 0.0
    s = sum(math.sin(a) for a in ang2) / len(ang2)
    c = sum(math.cos(a) for a in ang2) / len(ang2)
    return len(ang2), min(1.0, math.hypot(s, c))

def _collect_continuity_neighbours(
    concept_graph: Any,
    orient_dets: List[Detection],
    det_to_nid: Dict[int, int],
    anchor_positions: Dict[Any, Tuple[float, ...]],
    junction_radius: float,
    grid: Dict[Tuple[int, ...], List[int]],
) -> Dict[int, Set[int]]:
    """Builds instance-level neighbor lookup over outward ORIENTATION_CONTINUITY edges."""
    continuity_nbr: Dict[int, Set[int]] = defaultdict(set)
    if not concept_graph or not orient_dets or not det_to_nid:
        return continuity_nbr

    pattern_continuity: Set[Tuple[int, int]] = set()
    for nid in set(det_to_nid.values()):
        for edge in concept_graph.get_outgoing_edges(nid):
            if edge.edge_type == EdgeType.ORIENTATION_CONTINUITY:
                pattern_continuity.add((nid, edge.child))
                pattern_continuity.add((edge.child, nid))
        for edge in concept_graph.get_incoming_edges(nid):
            if edge.edge_type == EdgeType.ORIENTATION_CONTINUITY:
                pattern_continuity.add((edge.parent, nid))
                pattern_continuity.add((nid, edge.parent))

    for cell, indices in grid.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                for j in grid.get(neighbor_cell, []):
                    for i in indices:
                        if i >= j:
                            continue
                        nid_i = det_to_nid.get(i)
                        nid_j = det_to_nid.get(j)
                        if nid_i is None or nid_j is None:
                            continue
                        if (nid_i, nid_j) not in pattern_continuity:
                            continue
                        pos_i = anchor_positions.get(orient_dets[i].alpha)
                        pos_j = anchor_positions.get(orient_dets[j].alpha)
                        if not pos_i or not pos_j or len(pos_i) < 2 or len(pos_j) < 2:
                            continue
                        if _pos_dist(pos_i, pos_j) <= junction_radius:
                            continuity_nbr[i].add(j)
                            continuity_nbr[j].add(i)
    return continuity_nbr

def _envelope_cov(env: np.ndarray) -> np.ndarray:
    kh, kw = env.shape
    yy, xx = np.mgrid[0:kh, 0:kw].astype(np.float64)
    w = env / (env.sum() + 1e-12)
    dy, dx = yy - (w * yy).sum(), xx - (w * xx).sum()
    return np.array([[(w * dy * dy).sum(), (w * dy * dx).sum()],
                     [(w * dx * dy).sum(), (w * dx * dx).sum()]], dtype=np.float64)

def compute_endstopped_map(
    Eg: np.ndarray,
    thetas: Sequence[float],
    sfs: Sequence[float],
    prep: Optional[Dict[str, Any]] = None,
    kappa: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V1/V2 Continuous Length-Tuned Hypercomplex Filter.
    Hubel & Wiesel (1965); Dobbins, Zucker & Cynader (1987).

    Continuous length tuning: evaluates center length L_opt against
    extended inhibitory end-zones (2.2 * L_opt).
    """
    from scipy import ndimage
    nt, nsf, H, W = Eg.shape
    rel_s = np.zeros_like(Eg)
    rel_d = np.zeros_like(Eg)
    side = np.zeros(Eg.shape, dtype=np.int8)

    for it, th in enumerate(thetas):
        ay, ax = math.cos(th), -math.sin(th)

        for isf in range(nsf):
            if sfs[isf] <= 0:
                continue

            sx = P.GABOR_SIGMA_X_FRAC / max(float(sfs[isf]), 0.05)
            L_opt = max(1.5, kappa * sx)
            L_long = L_opt * 2.2

            c = Eg[it, isf]

            # Receptive field center vs end-zone flank shifts
            f_pos_opt = ndimage.shift(c, (ay * L_opt, ax * L_opt), order=1, mode="nearest")
            f_neg_opt = ndimage.shift(c, (-ay * L_opt, -ax * L_opt), order=1, mode="nearest")
            f_pos_long = ndimage.shift(c, (ay * L_long, ax * L_long), order=1, mode="nearest")
            f_neg_long = ndimage.shift(c, (-ay * L_long, -ax * L_long), order=1, mode="nearest")

            denom = c + P.ENDSTOP_EPS

            # Continuous suppression index
            supp_pos = np.maximum(0.0, c - f_pos_long) / denom
            supp_neg = np.maximum(0.0, c - f_neg_long) / denom
            double_supp = np.maximum(0.0, c - np.maximum(f_pos_long, f_neg_long)) / denom

            rel_s[it, isf] = np.maximum(supp_pos, supp_neg)
            rel_d[it, isf] = double_supp
            side[it, isf] = np.where(f_neg_opt < f_pos_opt, 1, -1)

    return rel_s, rel_d, side

_TIE_MIN_COS = 0.25

def _oriented_tangents(
    bins: Sequence[float],
    pts: Sequence[Tuple[float, float]],
    periodic: bool,
) -> List[Optional[Tuple[float, float]]]:
    n = len(bins)
    unit = math.pi / float(P.ORIENTATION_N_THETA)
    out: List[Optional[Tuple[float, float]]] = []
    for i in range(n):
        if periodic:
            a, b = pts[(i - 1) % n], pts[(i + 1) % n]
        else:
            a, b = pts[max(i - 1, 0)], pts[min(i + 1, n - 1)]
        cx, cy = b[0] - a[0], b[1] - a[1]
        cn = math.hypot(cx, cy)
        th = (float(bins[i]) + 0.5) * unit
        tang_th = carrier_to_tangent(th)
        tx, ty = math.cos(tang_th), math.sin(tang_th)
        dot = cx * tx + cy * ty
        if cn < 1e-6 or abs(dot) < _TIE_MIN_COS * cn:
            out.append(None)
        else:
            sgn = 1.0 if dot > 0 else -1.0
            out.append((sgn * tx, sgn * ty))
    return out

def diagnostic_derived_lenses(
    dl: DerivedLenses,
    D_rec: List[Detection],
    D_op: List[Detection],
    D_chunk: List[Detection],
) -> Dict[str, Any]:
    return {
        "D_rec": len(D_rec),
        "D_op": len(D_op),
        "D_chunk": len(D_chunk),
        "theta_categorize": dl.theta_categorize,
        "theta_chunk_emit": dl.theta_chunk_emit,
    }

CURVATURE_TEMPLATES = [
    ("convex",       [ 1.0,  0.8,  0.6,  0.4,  0.2]),
    ("concave",      [-1.0, -0.8, -0.6, -0.4, -0.2]),
    ("saddle",       [ 1.0,  0.5, -0.2, -0.6, -0.9]),
    ("s_curve",      [ 1.0,  0.6,  0.0, -0.6, -1.0]),
    ("spiral_cw",    [ 0.2,  0.5,  0.8,  0.9,  1.0]),
    ("spiral_ccw",   [-0.2, -0.5, -0.8, -0.9, -1.0]),
    ("corner_n",     [ 0.0,  0.0,  1.0,  0.0,  0.0]),
    ("corner_e",     [ 1.0,  0.7,  0.0,  0.0,  0.0]),
    ("corner_s",     [ 0.0,  0.0, -1.0,  0.0,  0.0]),
    ("corner_w",     [ 0.0,  0.0,  0.0,  0.7,  1.0]),
    ("t_shape",      [ 0.5,  0.5,  1.0,  0.5,  0.5]),
    ("y_shape",      [-0.5, -0.5, -1.0, -0.5, -0.5]),
]

def sample_curvature_profile(contour_det, anchor_positions, n=5):
    """Sample a signed, max-normalized curvature profile along a contour.

    Anchors come from the contour detection's member roles; the profile is
    arc-length resampled to n points (menger curvature at interior points).
    """
    anchors = _extract_member_anchors(contour_det)
    pts = [anchor_positions[a] for a in anchors if a in anchor_positions]
    if len(pts) < 3:
        return np.zeros(n, dtype=np.float32)
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(
            cum[-1]
            + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        )
    total = cum[-1]
    if total < 1e-6:
        return np.zeros(n, dtype=np.float32)
    sample_arcs = np.linspace(0, total, n)
    sampled = []
    j = 0
    for arc in sample_arcs:
        while j < len(cum) - 1 and cum[j + 1] < arc:
            j += 1
        if j >= len(cum) - 1:
            sampled.append(pts[-1])
        else:
            t = (arc - cum[j]) / (cum[j + 1] - cum[j] + 1e-9)
            sampled.append(
                (
                    pts[j][0] + t * (pts[j + 1][0] - pts[j][0]),
                    pts[j][1] + t * (pts[j + 1][1] - pts[j][1]),
                )
            )
    curv = np.zeros(n, dtype=np.float32)
    for i in range(1, n - 1):
        p0, p1, p2 = sampled[i - 1], sampled[i], sampled[i + 1]
        a = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        b = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        c = math.hypot(p2[0] - p0[0], p2[1] - p0[1])
        if a < 1e-6 or b < 1e-6 or c < 1e-6:
            continue
        cross = (
            (p1[0] - p0[0]) * (p2[1] - p1[1])
            - (p1[1] - p0[1]) * (p2[0] - p1[0])
        )
        area2 = abs(cross)
        if a * b * c < 1e-6:
            continue
        kappa = 2.0 * area2 / (a * b * c)
        curv[i] = math.copysign(kappa, cross)
    m = np.abs(curv).max()
    if m > 1e-6:
        curv = curv / m
    return curv.astype(np.float32)

def match_templates(profile):
    """Score a curvature profile against CURVATURE_TEMPLATES.

    Returns (best_index, best_score, softmax_population).
    """
    scores = np.zeros(len(CURVATURE_TEMPLATES), dtype=np.float32)
    for i, (_, tmpl) in enumerate(CURVATURE_TEMPLATES):
        scores[i] = float(
            np.dot(profile, np.array(tmpl, dtype=np.float32)) / 5.0
        )
    exp_s = np.exp((scores - scores.max()) / 0.2)
    pop = exp_s / (exp_s.sum() + 1e-9)
    best = int(np.argmax(scores))
    return best, float(scores[best]), pop



def polyline_shape_descriptor(poly, observation, closed: bool = True):
    """Shared RegionShape/HolisticShape descriptor; preserve binning order.

    `closed` says whether the polyline is a ring: it selects the periodic
    curvature encoding and whether the walk wraps. A closing edge is only
    real for a ring, so an open walk is not given spurious wrap turns.
    """
    n_pts = len(poly)
    periodic = bool(closed) and n_pts >= 3
    raw_area_sum = 0.0
    if periodic:
        for i in range(n_pts):
            p1 = poly[i]
            p2 = poly[(i + 1) % n_pts]
            raw_area_sum += p1[0] * p2[1] - p2[0] * p1[1]
    area = abs(raw_area_sum) / 2.0

    hist = [0.0] * P.REGION_SHAPE_CURV_BINS
    turns = [0.0] * n_pts
    arc_x = [0.0] * n_pts
    _arc = 0.0
    for i in range(n_pts):
        p_prev = (poly[(i - 1) % n_pts] if periodic
                  else poly[max(0, i - 1)])
        p_curr = poly[i]
        p_next = (poly[(i + 1) % n_pts] if periodic
                  else poly[min(n_pts - 1, i + 1)])
        v1x = p_curr[0] - p_prev[0]
        v1y = p_curr[1] - p_prev[1]
        v2x = p_next[0] - p_curr[0]
        v2y = p_next[1] - p_curr[1]
        cross = v1x * v2y - v1y * v2x
        dot = v1x * v2x + v1y * v2y
        theta = math.atan2(cross, dot)
        if raw_area_sum < 0:
            theta = -theta
        turns[i] = theta
        arc_x[i] = _arc
        bin_idx = int(
            math.floor(
                (theta + math.pi) / (2 * math.pi) * P.REGION_SHAPE_CURV_BINS
            )
        )
        bin_idx = min(P.REGION_SHAPE_CURV_BINS - 1, max(0, bin_idx))
        L_local = (math.hypot(v1x, v1y) + math.hypot(v2x, v2y)) / 2.0
        hist[bin_idx] += L_local
        _arc += L_local

    total_L = sum(hist)
    if total_L > 1e-6:
        hist = [h / total_L for h in hist]
    hist_bins = tuple(
        float(qi(math.sqrt(h), P.REGION_SHAPE_HIST_LEVELS, 0.0, 1.0))
        for h in hist
    )

    img_area = float(getattr(observation, "image_area", 0.0)) or 1.0
    area_norm = math.log10(max(area / img_area, 1e-6))
    area_bin = float(qi(area_norm, P.REGION_SHAPE_AREA_BINS, -6.0, 0.0))

    _prof_turns = [turns[i] for i in range(n_pts)]
    _curv_unit = math.pi / max(1, getattr(P, "CONTOUR_SEQ_LEVELS", 9) - 1)
    total_arc = max(_arc, 1e-6)
    arc_x_norm = [a / total_arc for a in arc_x]
    _seq, _total = _canonical_curvature_sequence(
        [int(round(abs(t_) / _curv_unit)) for t_ in _prof_turns],
        _prof_turns,
        arc_x_norm,
        periodic=periodic,
    )

    _cx = sum(p[0] for p in poly) / n_pts
    _cy = sum(p[1] for p in poly) / n_pts
    _sxx = sum((p[0] - _cx) ** 2 for p in poly) / n_pts
    _syy = sum((p[1] - _cy) ** 2 for p in poly) / n_pts
    _sxy = sum((p[0] - _cx) * (p[1] - _cy) for p in poly) / n_pts
    _tr = _sxx + _syy
    _det = _sxx * _syy - _sxy * _sxy
    _disc = math.sqrt(max(0.0, _tr * _tr / 4.0 - _det))
    _l1 = _tr / 2.0 + _disc
    _l2 = _tr / 2.0 - _disc
    elong = math.sqrt(_l1 / (_l2 + 1e-9)) if _l2 > 1e-9 else 1.0
    elong_norm = min(1.0, (elong - 1.0) / 4.0)
    elong_bin = min(7.0, math.floor(elong_norm * P.REGION_SHAPE_ASPECT_BINS))
    sym_bin, axis_bin, fold_bin = contour_self_symmetry(poly)
    return hist_bins, area_bin, elong_bin, sym_bin, axis_bin, fold_bin, _seq


# --------------------------------------------------------------------------
# [W1] Entity source: chain-derived field regions, or the segmenter's.
#
# The Shape/identity lenses used to iterate observation.entity_indices and
# read a segmenter polyline per entity. With the segmenter removed they have
# nothing to iterate. This returns the same two things -- an iterable of
# entity anchors and a polyline accessor -- from whichever source is enabled,
# so a lens body needs no knowledge of which produced them.
#
# Neural analogue: a closed contour in V2/V4 acts as a figure and acquires
# surface properties; the field-derived entity is that figure.
def field_region_source(
    observation: Any,
    current_detections: Optional[Dict[str, List[Any]]],
    anchor_positions: Optional[Dict[Any, Any]],
) -> Optional[Any]:
    """Build (and cache) the field region source for this observation.

    Returns None when there are no chain detections to build from, so the
    caller can fall back to the legacy source. The result is cached on the
    observation so the five Shape lenses share ONE source per step instead
    of rebuilding identical geometry per lens.
    """
    if observation is None:
        return None
    cached = getattr(observation, "_field_region_source", None)
    if cached is not None:
        return cached
    chains = (current_detections or {}).get("PEAK_CHAIN", [])
    if not chains:
        return None
    from perception.chain_regions import build_field_regions
    channels = None
    if hasattr(observation, "get_channels"):
        try:
            channels = observation.get_channels()
        except Exception:
            channels = None
    shape = None
    if observation is not None:
        for attr in ("image_shape", "_lum_adapted", "_raw_image"):
            v = getattr(observation, attr, None)
            if v is not None and hasattr(v, "shape") and len(v.shape) >= 2:
                shape = (int(v.shape[0]), int(v.shape[1]))
                break
    src = build_field_regions(chains, anchor_positions, channels, shape)
    if len(src) == 0:
        return None
    try:
        observation._field_region_source = src
    except (AttributeError, TypeError):
        pass
    return src


def entity_polylines(
    observation: Any,
    current_detections: Optional[Dict[str, List[Any]]],
    anchor_positions: Optional[Dict[Any, Any]],
    use_field: bool = True,
    with_closure: bool = False,
) -> Dict[Any, Any]:
    """anchor -> ordered polyline, for the enabled entity source.

    use_field False gives the legacy segmenter polylines, keyed by the
    entity anchors the segmenter used, so the two sources are drop-in
    interchangeable at the call site.

    with_closure True returns ``anchor -> (polyline, closed)`` instead, so a
    shape descriptor can encode open walks as open rather than assuming every
    polyline is a periodic ring. The legacy segmenter path always reports
    closed (its polylines are region polygons); the field source reports its
    own ring flag.
    """
    def _put(out: Dict[Any, Any], a: Any, poly: Any, closed: bool) -> None:
        out[a] = (poly, bool(closed)) if with_closure else poly

    out: Dict[Any, Any] = {}
    if use_field:
        src = field_region_source(observation, current_detections,
                                 anchor_positions)
        if src is not None:
            for a in src.entity_anchors():
                e = src.get(a)
                if e is not None and len(e["polyline"]) >= 3:
                    _put(out, a, e["polyline"], bool(e.get("closed", True)))
            return out
    if observation is None:
        return out
    idx = (getattr(observation, "entity_indices", None)
           if hasattr(observation, "entity_indices") else None)
    if idx is None:
        getter = getattr(observation, "get_entity_indices", None)
        if callable(getter):
            try:
                idx = getter()
            except Exception:
                idx = None
    if idx is None:
        idx = range(getattr(observation, "_N", 0))
    for u in idx:
        getter = getattr(observation, "get_boundary_polyline", None)
        poly = getter(u) if callable(getter) else None
        if poly and len(poly) >= 3:
            _put(out, u, [(float(p[0]), float(p[1])) for p in poly], True)
    return out
