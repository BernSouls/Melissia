"""perception/rich.py -- extracted from sensory_floor.py (Move 3a).

MOVE ONLY: whole top-level statements, verbatim. The re-export
shim in sensory_floor.py keeps `from sensory_floor import X`
working until the importer repoint retires it.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve as fftconvolve
import math
import warnings
import parameters as P
from core.registries import (  # noqa: F401
    CIRCULAR_CYCLE_LENGTHS,
    CONSTITUTIONAL_RANGES,
    SLOT_WEIGHT_PRIORS,
    b1_class_name,
    b4_class_name,
)
from perception.filters import (  # noqa: F401
    _TINY,
    _conv2d,
)

_B1_SLOT_WEIGHTS_BY_CLASS = {}

_B1_W_BASE = 4.0

_B1_W_HIST = 3.0

_B1_W_PHASE = 3.0

_B1_W_SURROUND = 2.0

_B1_W_SP_HIST = 3.0

_B1_W_SP_NONHIST = 3.0

_B1_W_Q_PHASE_DIFF = 1.0

_B1_W_Q_THETA_DIFF = 0.5

_B1_W_ONOFF_SUBBIN = 1.0

_B1_W_SF_CURV = 0.5

_B1_SCHEMA_CACHE = {}  # C5 fix: cache by (nt, nsf)

_B1_PREP_CACHE = {}    # per-bank prep, keyed on geometry + relevant params

def _envelope_cov(env):
    """Second-moment covariance of a filter-envelope map (real values)."""
    kh, kw = env.shape
    yy, xx = np.mgrid[0:kh, 0:kw].astype(np.float64)
    w = env / (env.sum() + _TINY)
    dy, dx = yy - (w * yy).sum(), xx - (w * xx).sum()
    return np.array([[(w * dy * dy).sum(), (w * dy * dx).sum()],
                     [(w * dy * dx).sum(), (w * dx * dx).sum()]])

def _gaussian_kernel(cov, truncate=3.0, min_var=1e-6):
    """Normalised Gaussian kernel from a 2x2 covariance (eigen-clamped)."""
    vals, vecs = np.linalg.eigh(cov)
    cov = (vecs * np.maximum(vals, min_var)) @ vecs.T
    inv = np.linalg.inv(cov)
    ry = max(1, int(math.ceil(truncate * math.sqrt(cov[0, 0]))))
    rx = max(1, int(math.ceil(truncate * math.sqrt(cov[1, 1]))))
    yy, xx = np.mgrid[-ry:ry + 1, -rx:rx + 1].astype(np.float64)
    g = np.exp(-0.5 * (inv[0, 0] * yy * yy + 2 * inv[0, 1] * yy * xx + inv[1, 1] * xx * xx))
    return g / (g.sum() + _TINY)

def _phase_step_vec(z):
    """Mean local per-pixel phase advance (kx, ky) for a complex kernel."""
    row = np.angle(np.sum(np.conj(z[:-1, :]) * z[1:, :])) if z.shape[0] > 1 else 0.0
    col = np.angle(np.sum(np.conj(z[:, :-1]) * z[:, 1:])) if z.shape[1] > 1 else 0.0
    return np.array([col, row])

def _gaussian_filter_sf(arr, sigma=1.0, mode="reflect"):
    """Spatial Gaussian blur used by B18's surround term."""
    return gaussian_filter(np.asarray(arr, dtype=np.float64), sigma=sigma, mode=mode)

def _norm_weights(bank):
    """[V3.1] Cross-channel competition weights (theta and log-sf).

    Returns (Wt, Ws) row-stochastic pools: Wt competes across orientation,
    Ws across spatial frequency.
    """
    th = np.asarray(bank.theta, dtype=np.float64)
    d = np.abs(th[:, None] - th[None, :]) % math.pi
    d = np.minimum(d, math.pi - d)
    Wt = np.exp(-0.5 * (d / P.B1_NORM_SIGMA_THETA) ** 2)
    Wt /= Wt.sum(1, keepdims=True)
    ls = np.log2(np.maximum(np.asarray(bank.sf, dtype=np.float64), _TINY))
    Ws = np.exp(-0.5 * ((ls[:, None] - ls[None, :]) / max(P.B2_COS_SIGMA_SF, 1e-6)) ** 2)
    Ws /= Ws.sum(1, keepdims=True)
    return Wt, Ws

def _prepare_bank_b1(bank):
    """Per-image prep for B1's divisive normalisation.

    Caches pool/pool_phase/kvec/theta_w plus the B18 additions (norm_wt,
    norm_ws, surround_sigma). Keyed on kernel shape, bank geometry AND all
    relevant parameters so a changed floor/sigma never reuses a stale bank.
    """
    key = (bank.kernels.shape, tuple(bank.theta), tuple(bank.sf),
           float(P.GABOR_COMPLEX_POOL_RF_FRAC), float(P.GABOR_COMPLEX_POOL_SIGMA_THETA),
           float(P.B1_PHASE_ACROSS_SIGMA), float(P.B1_NORM_SIGMA),
           float(P.B1_NORM_SIGMA_THETA), float(P.B18_NOISE_FLOOR),
           float(P.B18_SIGMA0), float(P.B18_SURROUND_RATIO))
    # [Tree note] GaborBank is a FROZEN dataclass, so the prep cache cannot be
    # hung off the bank instance. A module-level dict keyed on the full geometry
    # + parameter key is equivalent (the key already encodes shape and every
    # constant that feeds the prep) and avoids the FrozenInstanceError.
    c = _B1_PREP_CACHE.get(key)
    if c is not None:
        return c

    nph = bank.kernels.shape[2]
    if nph != 2:
        warnings.warn(f"[B.1] Expected quadrature pair, got {nph}. Using first two.")

    nt, nsf = bank.kernels.shape[:2]
    kc = bank.kernels[:, :, 0].astype(np.float64) + 1j * bank.kernels[:, :, 1].astype(np.float64)
    env = np.abs(kc)
    gain = 0.5 * env.sum(axis=(2, 3))

    pool = [[None] * nsf for _ in range(nt)]
    pool_phase = [[None] * nsf for _ in range(nt)]
    kvec = np.zeros((nt, nsf, 2))

    frac = float(P.GABOR_COMPLEX_POOL_RF_FRAC)     # broad energy pool
    sig_c = float(P.B1_PHASE_ACROSS_SIGMA)          # carrier-across phase pool

    for it in range(nt):
        for isf in range(nsf):
            cov = _envelope_cov(env[it, isf])
            pool[it][isf] = _gaussian_kernel(cov * frac ** 2)
            kvec[it, isf] = _phase_step_vec(kc[it, isf])
            n = math.hypot(*kvec[it, isf])
            if n > _TINY:
                # Phase is defined mod 2pi only along the carrier wave;
                # orthogonal to it there is no phase progression. Build the
                # phase pool anisotropic: sigma=0 along k, sig_c across.
                u = np.array([kvec[it, isf][1], kvec[it, isf][0]]) / n   # (col, row)
                p = np.array([-u[1], u[0]])                              # perpendicular
                sa2 = float(p @ cov @ p) * frac ** 2
                pool_phase[it][isf] = _gaussian_kernel(
                    sa2 * np.outer(p, p) + sig_c ** 2 * np.outer(u, u))
            else:
                pool_phase[it][isf] = pool[it][isf]

    th = np.asarray(bank.theta, dtype=np.float64)
    sig = float(P.GABOR_COMPLEX_POOL_SIGMA_THETA)
    if sig <= 0:
        W = np.eye(nt)
    else:
        d = np.abs(th[:, None] - th[None, :]) % math.pi
        d = np.minimum(d, math.pi - d)
        W = np.exp(-0.5 * (d / sig) ** 2)
        W /= (W.sum(axis=1, keepdims=True) + _TINY)

    norm_wt, norm_ws = _norm_weights(bank)

    # [B18] Surround sigma derived from the bank's own RF geometry (pixels).
    rf_sigmas = [math.sqrt(max(np.trace(_envelope_cov(env[it, isf])) / 2.0, 1.0))
                 for it in range(nt) for isf in range(nsf)]
    surround_sigma = P.B18_SURROUND_RATIO * float(np.median(rf_sigmas))

    c = dict(key=key, gain=gain, pool=pool, pool_phase=pool_phase, kvec=kvec,
             theta_w=W, valid_sf=(np.asarray(bank.sf) > 0), env=env,
             norm_wt=norm_wt, norm_ws=norm_ws, surround_sigma=surround_sigma)
    _B1_PREP_CACHE[key] = c
    return c

def compute_responses_b1(bank, img, conv2d=None):
    """[B.1] Run the bank and apply B18's contrast gain control.

    Returns the base Gabor populations plus the gain-normalised energy map
    `Eg_local` (precomputed once per image so `b1_region_signature` never
    re-divides the whole image per region) and the ON/OFF gain-controlled maps.
    `on_v1_local + off_v1_local == complex_local` exactly, because the noise
    floor and `keep` factor are applied to both branches.
    """
    conv2d = conv2d or _conv2d
    prep = _prepare_bank_b1(bank)
    nt, nsf, nph = bank.kernels.shape[:3]
    h, w = img.shape

    simple = np.zeros((nt, nsf, nph, h, w), dtype=np.float32)
    for it in range(nt):
        for isf in range(nsf):
            for ip in range(nph):
                simple[it, isf, ip] = conv2d(img, bank.kernels[it, isf, ip])

    E = np.hypot(simple[:, :, 0], simple[:, :, 1]).astype(np.float32)
    phase = np.arctan2(simple[:, :, 1], simple[:, :, 0]).astype(np.float32)

    # Local energy: spatially pooled (broader than the carrier) BEFORE gain control.
    E_sp = np.empty_like(E)
    for it in range(nt):
        for isf in range(nsf):
            ry, rx = prep["pool"][it][isf].shape
            p = np.pad(E[it, isf][None].astype(np.float64),
                       ((0, 0), (ry // 2, ry // 2), (rx // 2, rx // 2)), mode="reflect")
            E_sp[it, isf] = fftconvolve(p, prep["pool"][it][isf][None],
                                        mode="valid", axes=(1, 2))[0]

    # --- [B18] Contrast gain control (replaces the old Heeger denom) --------
    gain = prep["gain"][:, :, None, None] + _TINY

    # 1. Noise floor, applied to the numerator AND to ON/OFF, so
    #    on_loc + off_loc == E_loc stays EXACT.
    E_c = np.maximum(E_sp - P.B18_NOISE_FLOOR * gain, 0.0)
    keep = E_c / (E_sp + _TINY)                       # in [0, 1]

    # B3c's V1 ON/OFF block lands later; derive ON/OFF from E_c using the
    # quadrature pair. [B18 NOTE] explicit locals check, not dir().
    s0 = simple[:, :, 0]
    on_c = ((E_c + s0 * keep) * 0.5).astype(np.float32)
    off_c = ((E_c - s0 * keep) * 0.5).astype(np.float32)

    # 2. Center pool: shared across theta/sf (competition, NOT blurring).
    Eg_c = E_c / gain
    pool_c = np.einsum("ij,jsyx->isyx", prep["norm_wt"], Eg_c, optimize=True)
    pool_c = np.einsum("ij,ajyx->aiyx", prep["norm_ws"], pool_c, optimize=True)

    # 3. Spatial surround: broad local contrast energy (center-surround).
    surround = _gaussian_filter_sf(Eg_c.mean(axis=(0, 1)),
                                   sigma=prep["surround_sigma"], mode="reflect")

    denom = (P.B18_SIGMA0 + P.B18_W_CENTER * pool_c
             + P.B18_W_SURROUND * surround[None, None]).astype(np.float32)

    E_loc, on_loc, off_loc = E_c / denom, on_c / denom, off_c / denom
    Eg_local = E_loc / gain

    return {"simple": simple, "complex": E, "complex_local": E_loc, "phase": phase,
            "Eg_local": Eg_local, "on_v1_local": on_loc, "off_v1_local": off_loc}

def b1_schema(nt, nsf):
    """B1's slot schema for a (nt, nsf) bank. 433 slots at (12, 5).

    Bug J fix: weight constants are part of the cache key.
    """
    key = (nt, nsf, _B1_W_BASE, _B1_W_HIST, _B1_W_PHASE, _B1_W_SURROUND,
           _B1_W_SP_HIST, _B1_W_SP_NONHIST, _B1_W_Q_PHASE_DIFF,
           _B1_W_Q_THETA_DIFF, _B1_W_ONOFF_SUBBIN, _B1_W_SF_CURV)
    if key in _B1_SCHEMA_CACHE:
        return _B1_SCHEMA_CACHE[key]

    S = []

    def add(name, lo, hi, period, weight):
        S.append((name, lo, hi, period, weight))

    # Per-slot weights derived from block targets
    n_base = 15
    w_base = _B1_W_BASE / n_base

    n_hist = nt * nsf
    w_hist = _B1_W_HIST / (3 * n_hist)

    # Phase block: 33 value slots (Flag 2 fix: validity flags deleted)
    n_phase = 33
    w_phase = _B1_W_PHASE / n_phase

    n_surround = 6 + nt * 2
    w_surround = _B1_W_SURROUND / n_surround

    n_sp_hist = nt * 3
    w_sp_hist = _B1_W_SP_HIST / (4 * n_sp_hist)

    # Bug A fix: phase_rel dropped (sub-pixel drift). Pairwise diffs suffice.
    n_sp_nonhist = 5  # theta_rel, nonempty, R, dtheta, dlogsf
    w_sp_nonhist = _B1_W_SP_NONHIST / (4 * n_sp_nonhist)

    w_q_phase_diff = _B1_W_Q_PHASE_DIFF / 4
    w_q_theta_diff = _B1_W_Q_THETA_DIFF / 2
    w_onoff_subbin = _B1_W_ONOFF_SUBBIN / 4

    # --- Base (15) ---
    add("E_peak", 0.0, 1.0, None, w_base)
    add("energy_conc", 0.0, 1.0, None, w_base)
    add("sf_peak_ratio", 0.0, 2.0, None, w_base)
    add("is_2d_peak", 0.0, 1.0, None, 0.1 * w_base)  # C4 fix: low weight
    add("region_lum_mean", 0.0, 1.0, None, w_base)
    add("region_lum_rms", 0.0, 1.0, None, w_base)
    add("surround_lum_mean", 0.0, 1.0, None, w_base)
    add("lum_mean_diff", -1.0, 1.0, None, w_base)
    add("boundary_interior_ratio", 0.0, 1.0, None, w_base)
    add("simple_flag", 0.0, 1.0, None, w_base)
    add("on_simple_flag", 0.0, 1.0, None, w_base)
    add("off_simple_flag", 0.0, 1.0, None, w_base)
    add("cross_scale_phase", -math.pi, math.pi, 2 * math.pi, w_base)
    add("dtheta", -math.pi / 2, math.pi / 2, math.pi, w_base)
    add("dlogsf", -1.0, 1.0, None, w_base)

    # --- Main histograms (nt x nsf each, 3 channels) ---
    for k in range(n_hist):
        add(f"hist_{k}", 0.0, 1.0, None, w_hist)
    for k in range(n_hist):
        add(f"on_hist_{k}", 0.0, 1.0, None, w_hist)
    for k in range(n_hist):
        add(f"off_hist_{k}", 0.0, 1.0, None, w_hist)

    # --- Phase block: ALL RELATIVE (A8: no phase_mean) ---
    add("phase_R", 0.0, 1.0, None, w_phase)
    add("phase_skew", 0.0, 1.0, None, w_phase)
    for n in ("phase_diff_theta_plus", "phase_diff_theta_minus",
              "phase_diff_theta_plus2", "phase_diff_theta_minus2",
              "phase_diff_sf_plus", "phase_diff_sf_minus",
              "phase_diff_sf_plus2", "phase_diff_sf_minus2",
              "phase_grad_theta", "phase_grad_sf",
              "phase_diff_tp_sp", "phase_diff_tp_sm",
              "phase_diff_tm_sp", "phase_diff_tm_sm"):
        add(n, -math.pi, math.pi, 2 * math.pi, w_phase)
    for n in ("phase_R_theta_plus", "phase_R_theta_minus",
              "phase_R_theta_plus2", "phase_R_theta_minus2",
              "phase_R_sf_plus", "phase_R_sf_minus",
              "phase_R_sf_plus2", "phase_R_sf_minus2",
              "on_phase_R", "off_phase_R", "on_phase_skew", "off_phase_skew"):
        add(n, 0.0, 1.0, None, w_phase)
    # A10 FIX: ON/OFF phase diffs relative to OWN peak
    for n in ("on_phase_diff_theta_plus", "on_phase_diff_theta_minus",
              "off_phase_diff_theta_plus", "off_phase_diff_theta_minus",
              "on_off_phase_diff"):
        add(n, -math.pi, math.pi, 2 * math.pi, w_phase)

    # --- Surround block ---
    add("surround_theta", 0.0, math.pi, math.pi, w_surround)
    add("surround_theta_conc", 0.0, 1.0, None, w_surround)
    add("surround_logsf_mean", -5.0, 0.0, None, w_surround)
    add("surround_logsf_std", 0.0, 2.0, None, w_surround)
    add("surround_on_mean", 0.0, 1.0, None, w_surround)
    add("surround_off_mean", 0.0, 1.0, None, w_surround)
    for k in range(nt * 2):
        add(f"surround_hist_{k}", 0.0, 1.0, None, w_surround)

    # --- Spatial sub-blocks (4 quadrants) ---
    # C8: spatial histograms use n_sf_bins=3 (coarser sf than main hist).
    for q in range(4):
        for k in range(n_sp_hist):
            add(f"sp_hist_{q}_{k}", 0.0, 1.0, None, w_sp_hist)
        add(f"sp_theta_rel_{q}", -math.pi / 2, math.pi / 2, math.pi, w_sp_nonhist)
        # C3: sp_nonempty_q is the guard slot; readers MUST check it first.
        add(f"sp_nonempty_{q}", 0.0, 1.0, None, w_sp_nonhist)
        # Bug B: sp_*_q slots are undefined/guarded when sp_nonempty_q == 0.
        add(f"sp_R_{q}", 0.0, 1.0, None, w_sp_nonhist)
        add(f"sp_dtheta_{q}", -math.pi / 2, math.pi / 2, math.pi, w_sp_nonhist)
        add(f"sp_dlogsf_{q}", -1.0, 1.0, None, w_sp_nonhist)

    # C3: Quadrant phase differences
    for n in ("sp_phase_diff_01", "sp_phase_diff_23",
              "sp_phase_diff_02", "sp_phase_diff_13"):
        add(n, -math.pi, math.pi, 2 * math.pi, w_q_phase_diff)

    # Cross-quadrant theta diffs (A4 FIX: signed)
    for n in ("sp_theta_diff_01", "sp_theta_diff_23"):
        add(n, -math.pi / 2, math.pi / 2, math.pi, w_q_theta_diff)

    # --- ON/OFF sub-bin (C8) ---
    add("on_dtheta", -math.pi / 2, math.pi / 2, math.pi, w_onoff_subbin)
    add("on_dlogsf", -1.0, 1.0, None, w_onoff_subbin)
    add("off_dtheta", -math.pi / 2, math.pi / 2, math.pi, w_onoff_subbin)
    add("off_dlogsf", -1.0, 1.0, None, w_onoff_subbin)

    # --- sf curvature ---
    add("sf_curvature", -1.0, 1.0, None, _B1_W_SF_CURV)

    _B1_SCHEMA_CACHE[key] = S
    return S

def b1_class_name(nt, nsf):
    """A2 FIX: Shape-dependent class name prevents registry collision."""
    return f"ORIENTATION_RICH_{nt}_{nsf}"

_B4_REGISTERED_KEYS = set()

_B4_W_TEXTURE_RICH = 3.0

def b4_class_name(nt, nsf):
    """A2 FIX: Shape-dependent class name prevents registry collision."""
    return f"TEXTURE_RICH_{nt}_{nsf}"

def b4_n_slots(nt, nsf):
    """4*nt*nsf + 4.

    Per (theta, sf) cell: autocorrelation peak, normalised period, and
    relative direction (3), plus the cross-channel covariance with the next
    theta bin (1). Then the FOUR invariant globals, in order: [-4] mean
    orientation concentration, [-3] log-sf centroid, [-2] roughness (variance
    of the theta x sf marginal), [-1] contrast-normalised energy. Note [-2]
    is roughness and [-1] is contrast energy -- NOT the old TEXTURE 2-vector.
    """
    return 4 * nt * nsf + 4

def b4_register(bank):
    """Register TEXTURE_RICH_{nt}_{nsf} into CONSTITUTIONAL_RANGES /
    _B1_SLOT_WEIGHTS_BY_CLASS / SLOT_WEIGHT_PRIORS. Returns the arity.

    [Tree bridge] Same lesson as b1_register: this tree's similarity stack
    resolves weights through SLOT_WEIGHT_PRIORS (resolve_slot_weights), and
    get_similarity_ranges returns None for an unregistered class, which
    makes _similarity_gaussian return 0.0 outright -- the exact silent-zero
    failure b1_reset warns about. All THREE maps are therefore written.

    Writes unconditionally rather than early-returning on the key, for the
    same reason b1_register does: an early return skips re-registration when
    something else has popped the range, and the class then compares as 0.0
    forever. The key is still maintained so b1_reset's B4 sweep stays the
    single reset path.

    Every slot carries range 1.0, so sigma = 1/3 uniformly and the prior is
    uniform. Only relative proportions are read (both kernels divide by the
    weight sum), so the uniform prior is scale-invariant.
    """
    nt, nsf = len(bank.theta), len(bank.sf)
    cls_name = b4_class_name(nt, nsf)
    n_slots = b4_n_slots(nt, nsf)
    CONSTITUTIONAL_RANGES[cls_name] = [1.0] * n_slots
    w = [1.0 / n_slots] * n_slots
    _B1_SLOT_WEIGHTS_BY_CLASS[cls_name] = w
    SLOT_WEIGHT_PRIORS[cls_name] = tuple(w)
    _B4_REGISTERED_KEYS.add((nt, nsf, float(_B4_W_TEXTURE_RICH)))
    return n_slots

def b1_reset():
    """C6 fix: clear all B1 global state. Call between tests.

    [AUDIT #5 FIX] MUST also clear the B4 registry and pop every
    TEXTURE_RICH_* range. Without this, b4_register() early-returns after a
    reset, TEXTURE_RICH has no CONSTITUTIONAL_RANGES entry, and every
    Gabor-path texture comparison silently returns similarity 0.0.
    """
    global _B1_SLOT_WEIGHTS_BY_CLASS, _B1_SCHEMA_CACHE
    for class_name in list(_B1_SLOT_WEIGHTS_BY_CLASS):
        CONSTITUTIONAL_RANGES.pop(class_name, None)
        CIRCULAR_CYCLE_LENGTHS.pop(class_name, None)
        SLOT_WEIGHT_PRIORS.pop(class_name, None)
    _B1_SLOT_WEIGHTS_BY_CLASS.clear()
    _B1_SCHEMA_CACHE.clear()
    _B1_PREP_CACHE.clear()
    # [AUDIT #5 FIX] B4 state is a SEPARATE registry keyed by (nt, nsf, W).
    # Also clear the content-tier rich schemas that were registered directly
    # in primitives.py (not via b1_register): ORIENTATION_RICH_CONTENT and
    # TEXTURE_RICH_CONTENT. Without this, they survive the reset while their
    # weight priors are gone, causing get_slot_weights to return None and
    # silent-zero similarity for the rich content tier.
    for cn in [k for k in CONSTITUTIONAL_RANGES if k.startswith("TEXTURE_RICH_") or k.startswith("ORIENTATION_RICH_")]:
        CONSTITUTIONAL_RANGES.pop(cn, None)
        _B1_SLOT_WEIGHTS_BY_CLASS.pop(cn, None)
        SLOT_WEIGHT_PRIORS.pop(cn, None)
    _b4_keys = globals().get("_B4_REGISTERED_KEYS")
    if _b4_keys is not None:
        _b4_keys.clear()

def b1_register(bank):
    """Register B1 schema into CONSTITUTIONAL_RANGES / CIRCULAR_CYCLE_LENGTHS
    / SLOT_WEIGHT_PRIORS.

    Always writes unconditionally. b1_schema is cached (O(1) on hit);
    b1_register itself is O(n) per call over the slot list, but is
    invoked once per image rather than once per region, so the
    bookkeeping is negligible. This closes the staleness bug where
    mutating and reverting weight constants would skip re-registration.

    [Tree bridge] The blueprint wrote only `_B1_SLOT_WEIGHTS_BY_CLASS`;
    this tree's similarity stack reads SLOT_WEIGHT_PRIORS via
    get_slot_weights(), so we write both (identical normalised priors).
    """
    shape = (len(bank.theta), len(bank.sf))
    nt, nsf = shape
    S = b1_schema(nt, nsf)
    class_name = b1_class_name(nt, nsf)
    CONSTITUTIONAL_RANGES[class_name] = [hi - lo for (_, lo, hi, _, _) in S]
    CIRCULAR_CYCLE_LENGTHS[class_name] = {
        i: p for i, (_, _, _, p, _) in enumerate(S) if p
    }
    # Flag 1 fix: normalize weights to sum to 1.0 per class
    w_list = [w for (_, _, _, _, w) in S]
    total_w = sum(w_list)
    norm = [w / total_w for w in w_list]
    _B1_SLOT_WEIGHTS_BY_CLASS[class_name] = norm
    SLOT_WEIGHT_PRIORS[class_name] = tuple(norm)
    return len(S)

def _compute_histogram(Eg, it_d, isf_d, nt_bins, nsf_bins, n_sf_bins=None):
    """Dominant-peak-aligned histogram: 12 theta bins x (2*half_sf+1) sf slots,
    L2-normalised, clamped to B1_HIST_CLAMP, re-normalised, floored at 0."""
    if n_sf_bins is None:
        n_sf_bins = nsf_bins
    Eg_rolled = np.roll(Eg, -it_d, axis=0)
    half_sf = n_sf_bins // 2
    hist = []
    for i_off in range(nt_bins):
        for j_off in range(-half_sf, half_sf + 1):
            isf_h = isf_d + j_off
            if 0 <= isf_h < nsf_bins:
                hist.append(float(Eg_rolled[i_off, isf_h]))
            else:
                hist.append(0.0)

    hist_arr = np.array(hist, dtype=np.float64)
    norm = np.linalg.norm(hist_arr)
    if norm > _TINY:
        hist_arr /= norm
    hist_arr = np.minimum(hist_arr, float(P.B1_HIST_CLAMP))
    norm = np.linalg.norm(hist_arr)
    if norm > _TINY:
        hist_arr /= norm
    return [float(min(max(v, 0.0), 1.0)) for v in hist_arr]

def _compute_phase_stats(F, it, isf, y0, y1, x0, x1, prep, location=None):
    """(ph_mean, phase_R, phase_skew) from the anisotropic phase pool."""
    H, W = F["simple"].shape[3:]
    if location is not None:
        yc, xc = location
    else:
        Ec = F["complex_local"][it, isf, y0:y1, x0:x1]
        if Ec.size == 0:
            return 0.0, 0.0, 0.0
        py, px = np.unravel_index(int(np.argmax(Ec)), Ec.shape)
        yc, xc = y0 + int(py), x0 + int(px)

    kern = prep["pool_phase"][it][isf]
    ry, rx = kern.shape[0] // 2, kern.shape[1] // 2
    ya, yb = max(0, yc - ry), min(H, yc + ry + 1)
    xa, xb = max(0, xc - rx), min(W, xc + rx + 1)

    g = kern[ya - (yc - ry): yb - (yc - ry), xa - (xc - rx): xb - (xc - rx)]
    sp = F["simple"][it, isf]
    s0p = sp[0, ya:yb, xa:xb].astype(np.float64)
    s1p = sp[1, ya:yb, xa:xb].astype(np.float64)
    zp = s0p + 1j * s1p
    ap = np.abs(zp)

    gz = (g * zp).sum()
    ga = (g * ap).sum()

    if ga > _TINY and abs(gz) > _TINY:
        ph_mean = float(np.angle(gz))
        phase_R = min(abs(gz) / ga, 1.0)
        wgt = g * ap
        wgt_sum = wgt.sum()
        if wgt_sum > _TINY:
            third_moment = (wgt * np.exp(1j * 3 * (np.angle(zp) - ph_mean))).sum() / wgt_sum
            phase_skew = float(abs(third_moment))
        else:
            phase_skew = 0.0
        return ph_mean, phase_R, phase_skew
    return 0.0, 0.0, 0.0

def _compute_simple_flag(F, it, isf, y0, y1, x0, x1, prep):
    """Complex-cell -> simple-cell dominance in [0, 1]."""
    H, W = F["simple"].shape[3:]
    Ec = F["complex_local"][it, isf, y0:y1, x0:x1]
    if Ec.size == 0:
        return 0.0
    py, px = np.unravel_index(int(np.argmax(Ec)), Ec.shape)
    yc, xc = y0 + int(py), x0 + int(px)

    kern = prep["pool_phase"][it][isf]
    ry, rx = kern.shape[0] // 2, kern.shape[1] // 2
    ya, yb = max(0, yc - ry), min(H, yc + ry + 1)
    xa, xb = max(0, xc - rx), min(W, xc + rx + 1)

    g = kern[ya - (yc - ry): yb - (yc - ry), xa - (xc - rx): xb - (xc - rx)]
    sp = F["simple"][it, isf]
    s0p = sp[0, ya:yb, xa:xb].astype(np.float64)
    ap = np.abs(s0p + 1j * sp[1, ya:yb, xa:xb].astype(np.float64))
    ga = (g * ap).sum()
    if ga > _TINY:
        return min(float((g * np.abs(s0p)).sum() / ga), 1.0)
    return 0.0

def _compute_subbin(F, it, isf, y0, y1, x0, x1, yc, xc, prep, sf_arr):
    """A1 FIX: uses sf_arr[isf] instead of gain. Computes dtheta/dlogsf."""
    # Self-audit fix: clamp yc, xc to image bounds
    H, W = F["simple"].shape[3:]
    yc = max(0, min(H - 1, yc))
    xc = max(0, min(W - 1, xc))

    sx_px = P.GABOR_SIGMA_X_FRAC / max(float(sf_arr[isf]), _TINY)
    sy_px = sx_px / P.GABOR_ASPECT
    ry = int(round(1.5 * sy_px))
    rx = int(round(1.5 * sx_px))
    ya, yb = max(0, yc - ry), min(H, yc + ry + 1)
    xa, xb = max(0, xc - rx), min(W, xc + rx + 1)

    if yb - ya < 2 or xb - xa < 2:
        return 0.0, 0.0

    zw = (F["simple"][it, isf, 0, ya:yb, xa:xb].astype(np.float64) +
          1j * F["simple"][it, isf, 1, ya:yb, xa:xb].astype(np.float64))
    qr = np.sum(np.conj(zw[:-1, :]) * zw[1:, :])
    qc = np.sum(np.conj(zw[:, :-1]) * zw[:, 1:])
    if abs(qr) > _TINY and abs(qc) > _TINY:
        q = np.array([np.angle(qc), np.angle(qr)])
        k = prep["kvec"][it, isf]
        nq, nk = np.hypot(*q), np.hypot(*k)
        if nq > _TINY and nk > _TINY:
            uq, uk = complex(*q) / nq, complex(*k) / nk
            dtheta = 0.5 * float(np.angle(uq * uq * np.conj(uk * uk)))
            dlogsf = float(np.clip(np.log2(nq / nk), -1.0, 1.0))
            return dtheta, dlogsf
    return 0.0, 0.0

def _wrap_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def _wrap_half_pi(angle):
    return (angle + math.pi / 2) % math.pi - math.pi / 2

def b1_region_signature(bank, F_lum, F_on, F_off, y0, y1, x0, x1,
                        lum_full, on_img, off_img, cy, cx, radius_px):
    """433-slot orientation-rich signature for one region window.

    All histograms are rolled to the luminance dominant peak (it_d, isf_d)
    so lum/ON/OFF/quadrant histograms share one cross-comparable frame.
    Returns (quantized_tuple, raw_stats_dict).
    """
    prep = _prepare_bank_b1(bank)
    th = np.asarray(bank.theta, dtype=np.float64)
    sf = np.asarray(bank.sf, dtype=np.float64)
    valid = prep["valid_sf"]
    nt_bins, nsf_bins = len(th), len(sf)
    H, W = F_lum["simple"].shape[3:]

    S = b1_schema(nt_bins, nsf_bins)
    # C2 fix: initialize to NaN so "never set" is distinguishable from
    # "measured zero." The quantization step maps NaN -> lo.
    st = dict.fromkeys([s[0] for s in S], float("nan"))

    # --- Luminance channel ---
    # [B18] Eg_local is precomputed once per image by compute_responses_b1.
    Eg_map_full = F_lum["Eg_local"]
    Eg = Eg_map_full[:, :, y0:y1, x0:x1].mean(axis=(2, 3)).astype(np.float64)

    masked_Eg = np.where(valid[None, :], Eg, -np.inf)
    it_d, isf_d = np.unravel_index(int(np.argmax(masked_Eg)), Eg.shape)

    st["E_peak"] = min(max(float(Eg[it_d, isf_d]), 0.0), 1.0)

    tot = Eg.sum()
    if tot > _TINY:
        raw = Eg.max() / (Eg.mean() + _TINY)
        st["energy_conc"] = max(0.0, min(1.0, float(
            (raw - 1.0) / (nt_bins * nsf_bins - 1.0 + _TINY))))

        sf_marg = Eg[it_d, :]
        if 0 < isf_d < nsf_bins - 1:
            neighbors = (sf_marg[isf_d - 1] + sf_marg[isf_d + 1]) / 2.0
            st["sf_peak_ratio"] = float(min(sf_marg[isf_d] / (neighbors + _TINY), 2.0))
            curvature = sf_marg[isf_d - 1] - 2 * sf_marg[isf_d] + sf_marg[isf_d + 1]
            st["sf_curvature"] = float(np.clip(curvature, -1.0, 1.0))
        else:
            st["sf_peak_ratio"] = 1.0

        # A7 FIX: 8-neighbor 2D peak test.
        # A2 NOTE: theta wraps (circular), sf clamps (not circular).
        # This asymmetry is intentional: sf has no bins beyond the bank's range.
        is_2d_peak = True
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                it_nb = (it_d + di) % nt_bins
                isf_nb = isf_d + dj
                if 0 <= isf_nb < nsf_bins:
                    if Eg[it_nb, isf_nb] > Eg[it_d, isf_d]:
                        is_2d_peak = False
                        break
            if not is_2d_peak:
                break
        st["is_2d_peak"] = 1.0 if is_2d_peak else 0.0

    # Dominant peak spatial location
    Ec_d = F_lum["complex_local"][it_d, isf_d, y0:y1, x0:x1]
    py_d, px_d = np.unravel_index(int(np.argmax(Ec_d)), Ec_d.shape)
    yc_d, xc_d = y0 + int(py_d), x0 + int(px_d)

    # Main histograms (A12: ON/OFF rolled to it_d)
    hist_lum = _compute_histogram(Eg, it_d, isf_d, nt_bins, nsf_bins)
    for k, v in enumerate(hist_lum):
        st[f"hist_{k}"] = v

    Eg_on = F_on["Eg_local"][:, :, y0:y1, x0:x1].mean(axis=(2, 3)).astype(np.float64)
    it_on, isf_on = np.unravel_index(
        int(np.argmax(np.where(valid[None, :], Eg_on, -np.inf))), Eg_on.shape)
    # A1 FIX: Cross-comparable reference frame. All histograms roll to (it_d, isf_d).
    # ON/OFF/quadrant histograms describe energy distribution around the lum peak.
    # If ON energy is not at the lum peak, it appears in off-center bins.
    hist_on = _compute_histogram(Eg_on, it_d, isf_d, nt_bins, nsf_bins)
    for k, v in enumerate(hist_on):
        st[f"on_hist_{k}"] = v

    Eg_off = F_off["Eg_local"][:, :, y0:y1, x0:x1].mean(axis=(2, 3)).astype(np.float64)
    it_off, isf_off = np.unravel_index(
        int(np.argmax(np.where(valid[None, :], Eg_off, -np.inf))), Eg_off.shape)
    hist_off = _compute_histogram(Eg_off, it_d, isf_d, nt_bins, nsf_bins)  # A1 FIX
    for k, v in enumerate(hist_off):
        st[f"off_hist_{k}"] = v


    # --- Phase block (A8: no phase_mean emitted, used only as internal ref) ---
    ph_mean, ph_R, ph_skew = _compute_phase_stats(
        F_lum, it_d, isf_d, y0, y1, x0, x1, prep, location=(yc_d, xc_d))
    st["phase_R"] = ph_R
    st["phase_skew"] = ph_skew
    # NOTE: ph_mean is NOT stored. It is the internal reference for all diffs.

    # Marginal theta phase diffs + R (C2)
    # A5 NOTE: For nt <= 4, theta+2 and theta-2 alias. Not a bug for nt=12.
    for offset, name in [(1, "plus"), (-1, "minus"), (2, "plus2"), (-2, "minus2")]:
        it_nb = (it_d + offset) % nt_bins
        ph_nb, R_nb, _ = _compute_phase_stats(
            F_lum, it_nb, isf_d, y0, y1, x0, x1, prep, location=(yc_d, xc_d))
        st[f"phase_diff_theta_{name}"] = _wrap_pi(ph_nb - ph_mean)
        st[f"phase_R_theta_{name}"] = R_nb

    # Marginal sf phase diffs + R. Out-of-range sf neighbors -> NaN,
    # which quantizes to `lo`. See known-limitation note in the schema.
    for offset, name in [(1, "plus"), (-1, "minus"), (2, "plus2"), (-2, "minus2")]:
        isf_nb = isf_d + offset
        if 0 <= isf_nb < nsf_bins:
            ph_nb, R_nb, _ = _compute_phase_stats(
                F_lum, it_d, isf_nb, y0, y1, x0, x1, prep, location=(yc_d, xc_d))
            st[f"phase_diff_sf_{name}"] = _wrap_pi(ph_nb - ph_mean)
            st[f"phase_R_sf_{name}"] = R_nb
        else:
            st[f"phase_diff_sf_{name}"] = float("nan")
            st[f"phase_R_sf_{name}"] = float("nan")

    # C1: Diagonal joint phase diffs
    for dtheta_off, dsf, name in [
        (1, 1, "tp_sp"), (1, -1, "tp_sm"), (-1, 1, "tm_sp"), (-1, -1, "tm_sm")
    ]:
        it_nb = (it_d + dtheta_off) % nt_bins
        isf_nb = isf_d + dsf
        if 0 <= isf_nb < nsf_bins:
            ph_nb, _, _ = _compute_phase_stats(
                F_lum, it_nb, isf_nb, y0, y1, x0, x1, prep, location=(yc_d, xc_d))
            st[f"phase_diff_{name}"] = _wrap_pi(ph_nb - ph_mean)
        else:
            st[f"phase_diff_{name}"] = float("nan")

    # Phase gradients (differences, position-invariant)
    amp_win = F_lum["complex"][:, :, y0:y1, x0:x1]
    ph_win = F_lum["phase"][:, :, y0:y1, x0:x1]
    cos_pm = (amp_win * np.cos(ph_win)).mean(axis=(2, 3))
    sin_pm = (amp_win * np.sin(ph_win)).mean(axis=(2, 3))
    phase_map = np.arctan2(sin_pm, cos_pm)

    dph_th = np.diff(phase_map[:, isf_d])
    wgt_th = Eg[:, isf_d]
    if wgt_th[:-1].sum() > _TINY:
        st["phase_grad_theta"] = float(math.atan2(
            (wgt_th[:-1] * np.sin(dph_th)).sum(),
            (wgt_th[:-1] * np.cos(dph_th)).sum()))

    dph_sf = np.diff(phase_map[it_d, :])
    wgt_sf = Eg[it_d, :]
    if wgt_sf[:-1].sum() > _TINY:
        st["phase_grad_sf"] = float(math.atan2(
            (wgt_sf[:-1] * np.sin(dph_sf)).sum(),
            (wgt_sf[:-1] * np.cos(dph_sf)).sum()))

    # --- ON/OFF phase (A10 FIX: reference own peak) ---
    ph_on_peak, on_ph_R, on_ph_skew = _compute_phase_stats(
        F_on, it_on, isf_on, y0, y1, x0, x1, prep)
    st["on_phase_R"], st["on_phase_skew"] = on_ph_R, on_ph_skew

    ph_off_peak, off_ph_R, off_ph_skew = _compute_phase_stats(
        F_off, it_off, isf_off, y0, y1, x0, x1, prep)
    st["off_phase_R"], st["off_phase_skew"] = off_ph_R, off_ph_skew

    for offset, name in [(1, "plus"), (-1, "minus")]:
        it_nb_on = (it_on + offset) % nt_bins
        ph_nb_on, _, _ = _compute_phase_stats(F_on, it_nb_on, isf_on, y0, y1, x0, x1, prep)
        st[f"on_phase_diff_theta_{name}"] = _wrap_pi(ph_nb_on - ph_on_peak)

        it_nb_off = (it_off + offset) % nt_bins
        ph_nb_off, _, _ = _compute_phase_stats(F_off, it_nb_off, isf_off, y0, y1, x0, x1, prep)
        st[f"off_phase_diff_theta_{name}"] = _wrap_pi(ph_nb_off - ph_off_peak)

    # A6 FIX: use B1_PHASE_R_MIN from parameters
    ph_on_at_lum, on_R_at_lum, _ = _compute_phase_stats(
        F_on, it_d, isf_d, y0, y1, x0, x1, prep, location=(yc_d, xc_d))
    ph_off_at_lum, off_R_at_lum, _ = _compute_phase_stats(
        F_off, it_d, isf_d, y0, y1, x0, x1, prep, location=(yc_d, xc_d))
    if on_R_at_lum > P.B1_PHASE_R_MIN and off_R_at_lum > P.B1_PHASE_R_MIN:
        st["on_off_phase_diff"] = _wrap_pi(ph_on_at_lum - ph_off_at_lum)

    # Simple flags
    st["simple_flag"] = _compute_simple_flag(F_lum, it_d, isf_d, y0, y1, x0, x1, prep)
    st["on_simple_flag"] = _compute_simple_flag(F_on, it_on, isf_on, y0, y1, x0, x1, prep)
    st["off_simple_flag"] = _compute_simple_flag(F_off, it_off, isf_off, y0, y1, x0, x1, prep)


    # Sub-bin (A1 FIX: pass sf array)
    st["dtheta"], st["dlogsf"] = _compute_subbin(
        F_lum, it_d, isf_d, y0, y1, x0, x1, yc_d, xc_d, prep, sf)

    # Bug 1 fix: Compute ON/OFF sub-bin at their OWN spatial peaks,
    # not the luminance peak. Measuring at the lum peak yields noise
    # if the ON/OFF response is weak there.
    Ec_on = F_on["complex_local"][it_on, isf_on, y0:y1, x0:x1]
    py_on, px_on = np.unravel_index(int(np.argmax(Ec_on)), Ec_on.shape)
    yc_on, xc_on = y0 + int(py_on), x0 + int(px_on)
    st["on_dtheta"], st["on_dlogsf"] = _compute_subbin(
        F_on, it_on, isf_on, y0, y1, x0, x1, yc_on, xc_on, prep, sf)

    Ec_off = F_off["complex_local"][it_off, isf_off, y0:y1, x0:x1]
    py_off, px_off = np.unravel_index(int(np.argmax(Ec_off)), Ec_off.shape)
    yc_off, xc_off = y0 + int(py_off), x0 + int(px_off)
    st["off_dtheta"], st["off_dlogsf"] = _compute_subbin(
        F_off, it_off, isf_off, y0, y1, x0, x1, yc_off, xc_off, prep, sf)

    # Cross-scale phase
    valid_sf_indices = np.where(valid)[0]
    if len(valid_sf_indices) > 1:
        lowest_isf = valid_sf_indices[0]
        sp_low = F_lum["simple"][it_d, lowest_isf]
        sp_high = F_lum["simple"][it_d, isf_d]
        kern_d = prep["pool_phase"][it_d][isf_d]
        ry_dk, rx_dk = kern_d.shape[0] // 2, kern_d.shape[1] // 2
        ya_dk, yb_dk = max(0, yc_d - ry_dk), min(H, yc_d + ry_dk + 1)
        xa_dk, xb_dk = max(0, xc_d - rx_dk), min(W, xc_d + rx_dk + 1)
        g_d = kern_d[ya_dk - (yc_d - ry_dk): yb_dk - (yc_d - ry_dk),
                      xa_dk - (xc_d - rx_dk): xb_dk - (xc_d - rx_dk)]
        zp_low = (sp_low[0, ya_dk:yb_dk, xa_dk:xb_dk].astype(np.float64) +
                  1j * sp_low[1, ya_dk:yb_dk, xa_dk:xb_dk].astype(np.float64))
        zp_high = (sp_high[0, ya_dk:yb_dk, xa_dk:xb_dk].astype(np.float64) +
                   1j * sp_high[1, ya_dk:yb_dk, xa_dk:xb_dk].astype(np.float64))
        amp_low = np.abs(zp_low)
        ph_diff = np.angle(zp_low) - np.angle(zp_high)
        wgt_cs = g_d * amp_low
        wgt_cs_sum = wgt_cs.sum()
        if wgt_cs_sum > _TINY:
            st["cross_scale_phase"] = float(math.atan2(
                (wgt_cs * np.sin(ph_diff)).sum(),
                (wgt_cs * np.cos(ph_diff)).sum()))

    # --- Spatial sub-blocks ---
    ym = (y0 + y1) // 2
    xm = (x0 + x1) // 2
    quadrants = [(y0, ym, x0, xm), (y0, ym, xm, x1),
                 (ym, y1, x0, xm), (ym, y1, xm, x1)]
    sp_peaks_rel = []
    sp_phases_rel = []
    for q_idx, (qy0, qy1, qx0, qx1) in enumerate(quadrants):
        is_nonempty = (qy1 > qy0 and qx1 > qx0)
        st[f"sp_nonempty_{q_idx}"] = 1.0 if is_nonempty else 0.0

        if is_nonempty:
            Eg_q = Eg_map_full[:, :, qy0:qy1, qx0:qx1].mean(axis=(2, 3)).astype(np.float64)
            masked_q = np.where(valid[None, :], Eg_q, -np.inf)
            it_q, isf_q = np.unravel_index(int(np.argmax(masked_q)), Eg_q.shape)

            # A4 FIX: signed relative theta in [-pi/2, pi/2)
            sp_theta_rel = _wrap_half_pi(float(th[it_q]) - float(th[it_d]))
            sp_peaks_rel.append(sp_theta_rel)
            st[f"sp_theta_rel_{q_idx}"] = sp_theta_rel

            hist_q = _compute_histogram(Eg_q, it_d, isf_d, nt_bins, nsf_bins, n_sf_bins=3)  # A1 FIX
            for k, v in enumerate(hist_q):
                st[f"sp_hist_{q_idx}_{k}"] = v

            # Bug A fix: sp_phase_rel_q slot dropped to avoid emitting sub-pixel drift.
            # However, we MUST compute the carrier-advance-corrected relative phase
            # internally for the pairwise diffs, otherwise sp_phase_diff_* loses
            # structural stability and becomes sensitive to exact quadrant peak offsets.
            sp_ph, sp_R, _ = _compute_phase_stats(F_lum, it_q, isf_q, qy0, qy1, qx0, qx1, prep)
            Ec_q = F_lum["complex_local"][it_q, isf_q, qy0:qy1, qx0:qx1]
            py_q, px_q = np.unravel_index(int(np.argmax(Ec_q)), Ec_q.shape)
            yc_q, xc_q = qy0 + int(py_q), qx0 + int(px_q)
            kv = prep["kvec"][it_q, isf_q]  # (col, row)
            carrier_advance = kv[0] * (xc_q - xc_d) + kv[1] * (yc_q - yc_d)
            sp_phase_rel = _wrap_pi(sp_ph - ph_mean - carrier_advance)
            sp_phases_rel.append(sp_phase_rel)
            st[f"sp_R_{q_idx}"] = sp_R

            # Sub-bin at quadrant peak.
            st[f"sp_dtheta_{q_idx}"], st[f"sp_dlogsf_{q_idx}"] = _compute_subbin(
                F_lum, it_q, isf_q, qy0, qy1, qx0, qx1, yc_q, xc_q, prep, sf)
        else:
            sp_peaks_rel.append(float("nan"))
            sp_phases_rel.append(float("nan"))


    # C3: Quadrant phase differences
    if len(sp_phases_rel) >= 4:
        pairs = [(0, 1, "01"), (2, 3, "23"), (0, 2, "02"), (1, 3, "13")]
        for a, b, name in pairs:
            if np.isfinite(sp_phases_rel[a]) and np.isfinite(sp_phases_rel[b]):
                st[f"sp_phase_diff_{name}"] = _wrap_pi(sp_phases_rel[a] - sp_phases_rel[b])
            else:
                st[f"sp_phase_diff_{name}"] = float("nan")

    # Cross-quadrant theta diffs (A4/B6 FIX: signed, proper wrap)
    if len(sp_peaks_rel) >= 4:
        for a, b, name in [(0, 1, "01"), (2, 3, "23")]:
            if np.isfinite(sp_peaks_rel[a]) and np.isfinite(sp_peaks_rel[b]):
                st[f"sp_theta_diff_{name}"] = _wrap_half_pi(sp_peaks_rel[a] - sp_peaks_rel[b])
            else:
                st[f"sp_theta_diff_{name}"] = float("nan")

    # --- Luminance & Surround ---
    lum_win = lum_full[y0:y1, x0:x1]
    mean_lum = float(lum_win.mean())
    std_lum = float(lum_win.std())
    st["region_lum_mean"] = mean_lum
    st["region_lum_rms"] = float(std_lum / math.sqrt(mean_lum**2 + std_lum**2 + _TINY))

    yy_img, xx_img = np.ogrid[0:H, 0:W]
    dist_img = np.sqrt((xx_img - cx)**2 + (yy_img - cy)**2)
    surround_mask = (dist_img >= 1.5 * radius_px) & (dist_img <= 2.5 * radius_px)
    if surround_mask.any():
        Eg_sur = (Eg_map_full * surround_mask[None, None, :, :]).sum(axis=(2, 3))
        tot_sur = Eg_sur.sum()
        if tot_sur > _TINY:
            Ev_sur = Eg_sur * valid[None, :]
            c_th = (Ev_sur.sum(axis=1) * np.cos(2 * th)).sum()
            s_th = (Ev_sur.sum(axis=1) * np.sin(2 * th)).sum()
            st["surround_theta"] = (0.5 * math.atan2(s_th, c_th)) % math.pi
            st["surround_theta_conc"] = min(math.hypot(c_th, s_th) / tot_sur, 1.0)
            msf = Ev_sur.sum(axis=0)[valid]
            lsf = np.log2(sf[valid])
            mu = float((msf * lsf).sum() / (msf.sum() + _TINY))
            st["surround_logsf_mean"] = mu
            st["surround_logsf_std"] = float(math.sqrt(
                (msf * (lsf - mu)**2).sum() / (msf.sum() + _TINY)))

            sur_hist_2d = np.zeros((nt_bins, 2))
            lsf_all = np.log2(sf)
            sf_mid = float(np.median(lsf_all[valid]))
            for it_i in range(nt_bins):
                for isf_i in range(nsf_bins):
                    if valid[isf_i]:
                        sf_bin = 0 if lsf_all[isf_i] < sf_mid else 1
                        sur_hist_2d[it_i, sf_bin] += Ev_sur[it_i, isf_i]
            sur_hist_flat = sur_hist_2d.flatten()
            norm = np.linalg.norm(sur_hist_flat)
            if norm > _TINY:
                sur_hist_flat /= norm
            sur_hist_flat = np.minimum(sur_hist_flat, float(P.B1_HIST_CLAMP))
            norm = np.linalg.norm(sur_hist_flat)
            if norm > _TINY:
                sur_hist_flat /= norm
            for k in range(nt_bins * 2):
                st[f"surround_hist_{k}"] = float(min(max(sur_hist_flat[k], 0.0), 1.0))

        st["surround_lum_mean"] = float(lum_full[surround_mask].mean())
        st["lum_mean_diff"] = float(st["region_lum_mean"] - st["surround_lum_mean"])
        st["surround_on_mean"] = float(on_img[surround_mask].mean())
        st["surround_off_mean"] = float(off_img[surround_mask].mean())

    # Boundary-interior ratio
    h_win = y1 - y0
    w_win = x1 - x0
    if h_win > 4 and w_win > 4:
        ring = max(1, int(0.1 * min(h_win, w_win)))
        boundary_mask = np.zeros((h_win, w_win), dtype=bool)
        boundary_mask[:ring, :] = True
        boundary_mask[-ring:, :] = True
        boundary_mask[:, :ring] = True
        boundary_mask[:, -ring:] = True
        interior_mask = ~boundary_mask
        E_win = Eg_map_full[:, :, y0:y1, x0:x1].sum(axis=(0, 1))
        boundary_energy = float(E_win[boundary_mask].sum())
        interior_energy = float(E_win[interior_mask].sum())
        if interior_energy > _TINY:
            st["boundary_interior_ratio"] = float(
                min(boundary_energy / (interior_energy + _TINY), 1.0))

    # Quantize and clamp
    vals = [st[s[0]] for s in S]
    out = []
    for v, (_, lo, hi, per, _) in zip(vals, S):
        if not np.isfinite(v):
            v = lo
        if per:
            v = min(lo + ((v - lo) % per), hi)
        out.append(float(min(max(v, lo), hi)))

    return tuple(out), st



# [ITEM 4] Moved verbatim from perception/contours.py (thin-shim split; nothing deleted).
from scipy.ndimage import maximum_filter  # noqa: E402
from typing import Any, List, Tuple  # noqa: E402
from core.hashing import clamp  # noqa: E402,F401

def _compute_coarseness(window):
    """Low-frequency dominance of a region's pooled complex response.

    [V9.1 M5] Padded-domain Nyquist: the value is expressed in the same
    padded spectral domain the FFT was taken in (normalizing by the
    ORIGINAL support mixed domains and made it padding-dependent).
    """
    if window.size == 0:
        return 0.0
    agg = window.sum(axis=0)
    total = agg.sum(axis=0)
    h, w = total.shape
    hp = 1 << (h - 1).bit_length()
    wp = 1 << (w - 1).bit_length()
    padded = np.zeros((hp, wp), dtype=np.float32)
    padded[:h, :w] = total
    spec = np.abs(np.fft.fft2(padded))
    spec = np.fft.fftshift(spec)
    cy, cx = hp // 2, wp // 2
    yy, xx = np.mgrid[0:hp, 0:wp]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    padded_r_max = math.hypot(hp / 2.0, wp / 2.0) + 1e-9
    spec[r < 2.0] = 0.0
    peak = np.unravel_index(np.argmax(spec), spec.shape)
    return float(min(1.0, r[peak] / padded_r_max))

def _compute_regularity(window):
    """Autocorrelation peak-to-mean ratio of a region's response.

    [V9.1 M5] Plain masked mean over the exclusion-masked array (the old
    sum()/(size-25) divided by a hardcoded cell count that does not match
    the clamped slice near boundaries).
    """
    if window.size == 0:
        return 0.0
    agg = window.sum(axis=(0, 1))
    h, w = agg.shape
    hp = 1 << (h - 1).bit_length()
    wp = 1 << (w - 1).bit_length()
    padded = np.zeros((hp, wp), dtype=np.float32)
    padded[:h, :w] = agg
    f = np.fft.fft2(padded)
    ac = np.abs(np.fft.ifft2(f * np.conj(f)))
    ac = np.fft.fftshift(ac)
    cy, cx = hp // 2, wp // 2
    ac_wo = ac.copy()
    ac_wo[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3] = 0.0
    peak = float(ac_wo.max())
    mean_val = float(ac_wo.mean())
    return float(min(1.0, (peak / (mean_val + 1e-9)) / 10.0))

def _autocorr_peak(energy: np.ndarray, carrier_theta: float) -> Tuple[float, float, float]:
    """(peak, period_norm, rel_dir) of the first non-DC autocorrelation peak.
    rel_dir is measured from the BAR's long axis (carrier_theta + pi/2):
    1.0 = lag ALONG the bars, 0.0 = lag ACROSS the bars.
    """
    H, W = energy.shape
    e = energy - energy.mean()
    var = float((e * e).sum())
    if var < 1e-12:
        return 0.0, 0.0, 0.0
    F = np.fft.rfft2(e, s=(2 * H, 2 * W))
    ac = np.fft.fftshift(np.fft.irfft2(F * np.conj(F), s=(2 * H, 2 * W)))
    cy, cx = H, W
    ac = ac[cy - H // 2:cy + H // 2 + 1, cx - W // 2:cx + W // 2 + 1]
    yy, xx = np.mgrid[-(H // 2):H // 2 + 1, -(W // 2):W // 2 + 1]
    overlap = (H - np.abs(yy)) * (W - np.abs(xx))
    ac = ac * (H * W) / np.maximum(overlap, 1) / var
    c = (H // 2, W // 2)
    r_dist = np.hypot(yy, xx)
    dc_mask = r_dist <= 2.5
    cand = (ac == maximum_filter(ac, size=3)) & (~dc_mask) & (ac > 0.2)
    if not cand.any():
        return 0.0, 0.0, 0.0
    py, px = np.unravel_index(
        int(np.argmax(np.where(cand, ac, -np.inf))), ac.shape
    )
    dy, dx = py - c[0], px - c[1]
    max_dist = math.hypot(*c) + 1e-9
    mag = math.hypot(dy, dx) / max_dist
    bar_theta = carrier_theta + math.pi / 2.0
    rel = abs(math.cos(math.atan2(dy, dx) - bar_theta))
    return (
        clamp(float(ac[py, px]), 0.0, 1.0),
        clamp(mag, 0.0, 1.0),
        clamp(rel, 0.0, 1.0),
    )

def compute_texture_rich(
    complex_local: np.ndarray,
    bank_theta: np.ndarray,
    bank_sf: np.ndarray,
    lum_mean: float,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> Tuple[float, ...]:
    """Compute second-order texture statistics from V1 pooled complex energy."""
    nt, nsf, H_full, W_full = complex_local.shape
    H = y1 - y0
    W = x1 - x0

    gabor_window = complex_local[:, :, y0:y1, x0:x1].astype(np.float64)
    slots: List[float] = []

    log_sf = np.log2(bank_sf + 1e-9)
    log_sf_norm = (log_sf - log_sf.min()) / (log_sf.max() - log_sf.min() + 1e-9)

    for it in range(nt):
        carrier_theta = float(bank_theta[it])
        for isf in range(nsf):
            energy = gabor_window[it, isf]
            if float(energy.sum()) < 1e-9:
                slots.extend([0.0, 0.0, 0.0])
                continue
            pk, period_norm, rel_dir = _autocorr_peak(energy, carrier_theta)
            slots.extend([pk, period_norm, rel_dir])

    for it in range(nt):
        it_next = (it + 1) % nt
        for isf in range(nsf):
            e1 = gabor_window[it, isf]
            e2 = gabor_window[it_next, isf]
            s1 = float(e1.sum())
            s2 = float(e2.sum())
            if s1 < 1e-9 or s2 < 1e-9:
                slots.append(0.0)
                continue
            e1_n = e1 / (s1 + 1e-9)
            e2_n = e2 / (s2 + 1e-9)
            raw_overlap = float(np.sum(e1_n * e2_n))
            hhi_e1 = float((e1_n ** 2).sum())
            hhi_e2 = float((e2_n ** 2).sum())
            max_overlap = math.sqrt(hhi_e1 * hhi_e2)
            cross_corr = raw_overlap / (max_overlap + 1e-9) if max_overlap > 1e-9 else 0.0
            slots.append(clamp(cross_corr, 0.0, 1.0))

    marginal_theta = gabor_window.sum(axis=(1, 2, 3))
    marginal_sf = gabor_window.sum(axis=(0, 2, 3))
    tot_energy = float(gabor_window.sum()) + 1e-9

    p_theta = marginal_theta / tot_energy
    orient_conc = float((p_theta ** 2).sum())

    p_sf = marginal_sf / tot_energy
    sf_centroid = float(np.sum(p_sf * log_sf_norm))

    joint_p = gabor_window.sum(axis=(2, 3)) / tot_energy
    roughness = float(np.var(joint_p))

    contrast_energy = tot_energy / (tot_energy + lum_mean + 1e-6)

    slots.extend([
        clamp(orient_conc, 0.0, 1.0),
        clamp(sf_centroid, 0.0, 1.0),
        clamp(roughness * 100.0, 0.0, 1.0),
        clamp(contrast_energy, 0.0, 1.0),
    ])
    return tuple(slots)

def texture_richness(det: Any) -> float:
    """Texture richness accessor for figure-ground and surface grouping."""
    if det is None:
        return 0.0
    sig = getattr(det, "sigma", None)
    if sig is None:
        return 0.0
    q, c = sig.quantized, sig.class_
    if c.startswith("TEXTURE_RICH_") and len(q) >= 4:
        return float(q[-2]) * (1.0 - float(q[-4]))
    if len(q) >= 2:
        return (float(q[0]) / 7.0) * (1.0 - float(q[1]) / 7.0)
    return 0.0
