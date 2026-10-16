"""perception/observation_2d.py -- extracted from sensory_floor.py (Move 3a).

MOVE ONLY: whole top-level statements, verbatim. The re-export
shim in sensory_floor.py keeps `from sensory_floor import X`
working until the importer repoint retires it.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
import math
from typing import Any, Dict, List, Optional, Tuple, FrozenSet
import parameters as P
from core.hashing import (  # noqa: F401
    clamp,
)
from core.registries import (  # noqa: F401
    b1_class_name,
    b4_class_name,
)
from core.spatial import (  # noqa: F401
    SpatialHash,
)
from core.types import (  # noqa: F401
    TypedSignature,
)
from perception.contours import (  # noqa: F401
    V1RetinotopicLattice,
    _compute_coarseness,
    _compute_regularity,
    _detect_t_junctions,
    _extract_association_field_contours,
    _front_end_scale,
    _local_edge_at,
    compute_texture_rich,
    extract_shock_structure,
)
from perception.filters import (  # noqa: F401
    GaborBank,
    color_constancy,
    color_contrast,
    compute_opponent_retinex,
    compute_responses,
    luminance_adapt,
    opponent_channels,
    split_on_off,
)
from perception.observation import (  # noqa: F401
    Observation,
)
from perception.rich import (  # noqa: F401
    b1_region_signature,
    b1_register,
    b4_register,
    compute_responses_b1,
)

class Observation2D(Observation):
    """2D Image observation container supporting two-stream biological vision:
    dense retinotopic hypercolumns + continuous association-field contours.

    Two construction paths:
      - region-primary (``from_regions`` / direct ``__init__``): caller
        supplies pre-assembled region dicts.
      - pixel-primary (``from_image``): runs the two-stream front-end
        (V1 lattice + contour tracer), merges into ``all_regions``,
        and passes the result to the constructor.
    """

    # [Stage 5] Optional dataset view metadata (e.g. ETH-80 azimuth /
    # elevation / instance / label). Set by the data driver, e.g.
    # ``obs.view_meta = parse_eth80_view(path)``. When set (and
    # P.ASPECT_ENABLED), Pipeline.step feeds it to the AspectGraph.
    view_meta: Optional[Dict[str, Any]] = None  # Class-level default

    def __init__(
        self,
        regions: List[Dict[str, Any]],
        image_shape: Tuple[int, int],
        flag_indices: Optional[List[int]] = None,
        effect_indices: Optional[List[int]] = None,
        contours: Optional[List[Dict[str, Any]]] = None,
    ):
        self.view_meta = None  # [Stage 5] instance-level view metadata
        self._regions = list(regions)
        self._N = len(self._regions)
        self._h, self._w = image_shape
        self._flag = sorted(flag_indices) if flag_indices else []
        self._effect = sorted(effect_indices) if effect_indices else []
        self._contours = list(contours) if contours is not None else []

        # [Stage 1.2] Explicit two-stream categorization:
        # structural = boundary contours/junctions, surface = hypercolumns.
        self._structural_anchors: FrozenSet[int] = frozenset(
            i for i, r in enumerate(self._regions) if r.get("is_boundary", False)
        )
        self._surface_anchors: FrozenSet[int] = frozenset(
            i for i, r in enumerate(self._regions) if not r.get("is_boundary", False)
        )

        # --- Pairwise distance & adjacency ---
        self._dist: List[List[float]] = [[float("inf")] * self._N for _ in range(self._N)]
        self._adj: List[List[int]] = [[0] * self._N for _ in range(self._N)]
        # [Stage 1.2] (contour_anchor, surface_anchor, signal) contacts used
        # by Binder._run_stream_binding to emit cross-stream PART_CONTENT.
        self._contour_surface_contacts: List[Tuple[int, int, float]] = []

        if self._N > 0:
            cents = np.array([r["centroid"] for r in self._regions], dtype=np.float64)
            rads = np.array([r["radius_px"] for r in self._regions], dtype=np.float64)
            is_bnd = np.array([bool(r.get("is_boundary", False)) for r in self._regions], dtype=bool)

            dx = cents[:, 0, None] - cents[:, 0]
            dy = cents[:, 1, None] - cents[:, 1]
            dist_mat = np.hypot(dx, dy)
            self._dist = dist_mat.tolist()
            for i in range(self._N):
                self._dist[i][i] = 0.0

            adj_mat = np.zeros((self._N, self._N), dtype=np.int32)

            hyper_idx = np.where(~is_bnd)[0]
            bnd_idx = np.where(is_bnd)[0]

            # A. Hypercolumn-to-hypercolumn contact (radial RF overlap + 2px)
            if len(hyper_idx) > 0:
                h_cents = cents[hyper_idx]
                h_rads = rads[hyper_idx]
                h_dx = h_cents[:, 0, None] - h_cents[:, 0]
                h_dy = h_cents[:, 1, None] - h_cents[:, 1]
                h_dist = np.hypot(h_dx, h_dy)
                h_contact_tol = h_rads[:, None] + h_rads[None, :] + 2.0
                h_adj = (h_dist <= h_contact_tol) & (h_dist > 1e-6)
                for ii, orig_i in enumerate(hyper_idx):
                    for jj, orig_j in enumerate(hyper_idx):
                        if h_adj[ii, jj]:
                            adj_mat[orig_i, orig_j] = 1

            # B. Contour contact using polyline geometry (not centroids)
            if len(bnd_idx) > 0:
                # [Stage 1.2] Resolve polylines safely: primary path reads
                # self._contours via the region's _contour_idx; falls back to
                # the region's own boundary_polyline (region-built obs).
                # NOTE: the draft gated this block on `and self._contours`,
                # which would silently disable all contour adjacency for
                # from_regions() observations lacking a contours list.
                bnd_polys: List[Optional[np.ndarray]] = []
                for orig_b in bnd_idx:
                    r_b = self._regions[orig_b]
                    ci = r_b.get("_contour_idx")
                    poly_raw = None
                    if ci is not None and 0 <= ci < len(self._contours):
                        poly_raw = self._contours[ci].get("polyline")
                    if not poly_raw:
                        poly_raw = r_b.get("boundary_polyline")
                    bnd_polys.append(
                        np.array(poly_raw, dtype=np.float64)
                        if poly_raw and len(poly_raw) >= 2 else None
                    )

                # Contour-to-hypercolumn contact + stream-binding contacts
                for b_local, orig_b in enumerate(bnd_idx):
                    poly = bnd_polys[b_local]
                    if poly is None or len(poly) < 2:
                        continue
                    p_min = poly.min(axis=0) - 10.0
                    p_max = poly.max(axis=0) + 10.0
                    for orig_h in hyper_idx:
                        hc = cents[orig_h]
                        if not (p_min[0] <= hc[0] <= p_max[0] and p_min[1] <= hc[1] <= p_max[1]):
                            continue
                        d_vert = np.hypot(poly[:, 0] - hc[0], poly[:, 1] - hc[1])
                        min_d = float(d_vert.min())
                        hc_rad = float(rads[orig_h])
                        if min_d <= (hc_rad + 2.0):
                            adj_mat[orig_b, orig_h] = 1
                            adj_mat[orig_h, orig_b] = 1
                            # [Stage 1.2] populate cross-stream contacts
                            sig = 1.0 - clamp(min_d / (hc_rad + 1e-6), 0.0, 1.0)
                            self._contour_surface_contacts.append(
                                (orig_b, orig_h, sig)
                            )

                # Contour-to-contour: any vertex within 4.0px
                for bi in range(len(bnd_idx)):
                    poly_i = bnd_polys[bi]
                    if poly_i is None or len(poly_i) < 2:
                        continue
                    orig_i = bnd_idx[bi]
                    pi_sub = poly_i[::2]
                    for bj in range(bi + 1, len(bnd_idx)):
                        poly_j = bnd_polys[bj]
                        if poly_j is None or len(poly_j) < 2:
                            continue
                        orig_j = bnd_idx[bj]
                        pj_sub = poly_j[::2]
                        d_pp = np.hypot(
                            pi_sub[:, 0, None] - pj_sub[None, :, 0],
                            pi_sub[:, 1, None] - pj_sub[None, :, 1],
                        )
                        if np.any(d_pp <= 4.0):
                            adj_mat[orig_i, orig_j] = 1
                            adj_mat[orig_j, orig_i] = 1

            self._adj = adj_mat.tolist()

        # [v9] T-junction detection on the final region list.
        self._t_junctions = _detect_t_junctions(self._regions)

    def modality(self) -> str:
        return "2d"

    def num_atoms(self) -> int:
        return self._N

    def is_present(self, idx: int) -> bool:
        return 0 <= idx < self._N

    def get_entity_indices(self) -> List[int]:
        return list(range(self._N))

    def get_distance(self, i: int, j: int) -> float:
        if 0 <= i < self._N and 0 <= j < self._N:
            return self._dist[i][j]
        return float("inf") if i != j else 0.0

    def get_adjacency(self, i: int, j: int) -> int:
        if 0 <= i < self._N and 0 <= j < self._N:
            return self._adj[i][j]
        return 0

    def get_flag_indices(self) -> List[int]:
        return self._flag

    def get_effect_indices(self) -> List[int]:
        return self._effect

    # --- Contour accessors ---
    def get_luminance(self) -> Optional[Any]:
        """The adapted luminance plane consumed by the front end, or None.

        Prefers the plane the Gabor bank actually ran on (`_lum_adapted`,
        i.e. after luminance_adapt); falls back to the raw image's first
        channel for region-built observations that never adapted one.
        """
        lum = getattr(self, "_lum_adapted", None)
        if lum is not None:
            return lum
        raw = getattr(self, "_raw_image", None)
        if raw is not None and getattr(raw, "ndim", 0) >= 2:
            return raw[..., 0]
        return None

    def get_contours(self) -> List[Dict[str, Any]]:
        """Return the list of association-field contour dicts attached at
        construction time. Each dict has keys: polyline, theta_profile,
        strength_profile, length, is_closed."""
        return list(self._contours)

    def get_shock_structure(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """Lazy cached extraction of pruned shock skeleton structures."""
        if not hasattr(self, "_shock_structure"):
            contours = self.get_contours()
            self._shock_structure = extract_shock_structure(contours)
        return self._shock_structure

    def get_contour_polyline(self, contour_idx: int) -> Optional[List[Tuple[int, int]]]:
        """Return the pixel polyline for contour ``contour_idx``, or None."""
        if 0 <= contour_idx < len(self._contours):
            return self._contours[contour_idx].get("polyline")
        return None

    def get_contour_count(self) -> int:
        """Number of contours attached to this observation."""
        return len(self._contours)

    # --- [Stage 1.2] STREAM ACCESSORS ---
    def get_structural_anchors(self) -> FrozenSet[int]:
        """Boundary contour / junction anchors (structural stream)."""
        return self._structural_anchors

    def get_surface_anchors(self) -> FrozenSet[int]:
        """Hypercolumn anchors (surface stream)."""
        return self._surface_anchors

    def get_contour_surface_contacts(self) -> List[Tuple[int, int, float]]:
        """(contour_anchor, surface_anchor, signal) cross-stream contacts."""
        return list(self._contour_surface_contacts)

    # --- Micro-Lens Accessors queried by lenses.py ---
    def get_circularity(self, u: int) -> float:
        return self._regions[u]["circularity"] if 0 <= u < self._N else 0.5

    def get_complexity(self, u: int) -> float:
        return self._regions[u]["complexity"] if 0 <= u < self._N else 0.5

    def get_roughness(self, u: int) -> float:
        return self._regions[u]["roughness"] if 0 <= u < self._N else 0.0

    def get_isotropy(self, u: int) -> float:
        return self._regions[u]["isotropy"] if 0 <= u < self._N else 0.0

    def get_elongation(self, u: int) -> float:
        return self._regions[u]["elongation"] if 0 <= u < self._N else 1.0

    def get_radius(self, u: int) -> float:
        return self._regions[u]["radius_norm"] if 0 <= u < self._N else 0.1

    def get_extraction_scale(self) -> float:
        """[PHASE B] Step 4. Median region radius in PIXELS — the same units
        get_positions() returns (centroid pixel coordinates).

        Deliberately reads `_regions[i]["radius_px"]` rather than going
        through `get_radius()`, which returns `radius_norm` (a 0-1
        normalized value). Using the normalized form here would put the
        scale three orders of magnitude below the positions it has to be
        commensurate with.
        """
        radii = []
        for r in self._regions:
            rp = r.get("radius_px")
            if rp is None:
                continue
            try:
                v = float(rp)
            except (TypeError, ValueError):
                continue
            if math.isfinite(v) and v > 0.0:
                radii.append(v)
        if not radii:
            return 0.0
        radii.sort()
        mid = len(radii) // 2
        if len(radii) % 2 == 1:
            return radii[mid]
        return 0.5 * (radii[mid - 1] + radii[mid])

    def get_anchor_rgb(self, u: int) -> Optional[Tuple[float, float, float]]:
        return self._regions[u]["mean_rgb"] if 0 <= u < self._N else None

    def get_position(self, u: int) -> List[float]:
        # Macro-envelope-normalized address (rel_x/rel_y in [0, 1] mapped
        # back to [-1, 1]): view-stable spatial address per Patch 2.
        if 0 <= u < self._N:
            rx = self._regions[u].get("rel_x", 0.5)
            ry = self._regions[u].get("rel_y", 0.5)
            return [2.0 * rx - 1.0, 2.0 * ry - 1.0, 0.0]
        return [0.0, 0.0, 0.0]

    def get_positions(self) -> Dict[int, Tuple[float, float, float]]:
        return {i: (self._regions[i]["centroid"][0], self._regions[i]["centroid"][1], 0.0) for i in range(self._N)}

    def get_boundary_polyline(
        self, u: int
    ) -> Optional[List[Tuple[int, int]]]:
        """[v9] Ordered boundary polyline for region u, or None if the
        trace failed OR the polyline is below
        ORIENTATION_SEGMENT_MIN_POLYLINE."""
        if 0 <= u < self._N:
            poly = self._regions[u].get("boundary_polyline")
            if poly and len(poly) >= P.ORIENTATION_SEGMENT_MIN_POLYLINE:
                return poly
        return None

    def get_chain_id(self, u: int) -> Optional[int]:
        """[v9] Chain identity for region u. Always None under v10
        (chaining removed)."""
        if 0 <= u < self._N:
            return self._regions[u].get("chain_id")
        return None

    def get_chain_bounds(self, u: int) -> Optional[Tuple[float, float]]:
        """[v9] (chain_pos_start, chain_pos_end) for region u. Always
        (0.0, 1.0) under v10 (singleton chains)."""
        if 0 <= u < self._N:
            r = self._regions[u]
            return (
                r.get("chain_pos_start", 0.0),
                r.get("chain_pos_end", 1.0),
            )
        return None

    def get_t_junctions(
        self,
    ) -> List[Tuple[int, int, Tuple[float, float], float]]:
        """[v9] Precomputed T-junctions. Entries are
        (stem_region_idx, top_region_idx, contact_point, confidence)."""
        return list(getattr(self, "_t_junctions", []))

    def get_orientation(self, u: int) -> Optional[Tuple[float, float]]:
        """[v8] (angle, elongation) of the region's major axis.
        angle ∈ [0, π), elongation ∈ [0, 1]. None if unavailable."""
        if 0 <= u < self._N:
            r = self._regions[u]
            if "major_axis_angle" in r and "elongation" in r:
                return (float(r["major_axis_angle"]), float(r["elongation"]))
        return None

    def get_b4_sig(self, u: int) -> Optional[TypedSignature]:
        """Return the V2 second-order texture signature for region u.
        Bounds-checked: returns None for invalid index or missing signature."""
        if not isinstance(u, int) or u < 0 or u >= self._N:
            return None
        r = self._regions[u]
        sig = r.get("b4_sig")
        if sig is None:
            return None
        if isinstance(sig, TypedSignature):
            return sig
        nt = getattr(P, "GABOR_N_THETA", 12)
        nsf = getattr(P, "GABOR_N_SF", 5)
        cls_name = r.get("b4_class", f"TEXTURE_RICH_{nt}_{nsf}")
        return TypedSignature(
            kind="CANONICAL",
            class_=cls_name,
            relation="SECOND_ORDER_STATS",
            quantized=tuple(float(x) for x in sig),
        )

    def get_b1_sig(self, u: int) -> Optional[TypedSignature]:
        """Return the V1 hypercolumn orientation signature for region u.
        Bounds-checked: returns None for invalid index or missing signature."""
        if not isinstance(u, int) or u < 0 or u >= self._N:
            return None
        r = self._regions[u]
        sig = r.get("b1_sig")
        if sig is None:
            return None
        if isinstance(sig, TypedSignature):
            return sig
        nt = getattr(P, "GABOR_N_THETA", 12)
        nsf = getattr(P, "GABOR_N_SF", 5)
        cls_name = r.get("b1_class", f"ORIENTATION_RICH_{nt}_{nsf}")
        return TypedSignature(
            kind="CANONICAL",
            class_=cls_name,
            relation="HYPERCOLUMN_POPULATION",
            quantized=tuple(float(x) for x in sig),
        )

    # ------------------------------------------------------------------
    # [V9 Phase 4b / draft B.6] Pixel-to-graph entry point + front-end
    # accessors. The region-primary constructor above is the live entry
    # (unchanged, byte-for-byte); these add the raw-image path alongside
    # it, per ruling (a). See the schema-delta report in the register:
    # the draft's assumed region keys already exist here.
    # ------------------------------------------------------------------

    @classmethod
    def from_regions(cls, regions, image_shape, flag_indices=None,
                     effect_indices=None):
        """Region-dict entry point (retained interface, ruling (a)).

        Semantically identical to the constructor; exists so callers
        written against the draft's interface keep working.
        """
        return cls(regions, image_shape, flag_indices, effect_indices)

    @classmethod
    def from_image(cls, raw_image, flag_indices=None, effect_indices=None,
                   contours=None, rich: bool = True):
        """Pixel-to-graph bridge: run the two-stream front-end.

        Stream 1 (V1 lattice): tessellates the visual field into
        overlapping hypercolumns via V1RetinotopicLattice.tile_with_features().
        Stream 2 (Association field): extracts curvilinear contours via
        _extract_association_field_contours().

        The two region sets are merged into ``all_regions`` and passed to
        the constructor as ``regions`` + ``contours``.

        When ``contours`` is passed explicitly (e.g. from a pre-computed
        cache), the contour region dicts are appended as boundary entities.

        ``rich=False`` skips the per-region rich-signature machinery
        (B1/B4 signatures, pooled Gabor, ON/OFF populations). Contours,
        the shock skeleton, and the aspect descriptor are computed from the
        SAME Gabor complex-energy map, so with rich=False the contours and
        descriptors are bit-identical to rich=True -- only the per-region
        response dicts are absent (get_gabor_responses returns None). The
        gate/cache paths only read contours, so they use rich=False for a
        ~3x speedup; ~73% of from_image's runtime was this machinery.

        Region-built instances (via from_regions) carry no stored
        responses: ``get_gabor_responses`` returns None for them.
        """
        h, w = raw_image.shape[:2]

        # --- Front-end: opponent colour + retinex ---
        opponent = opponent_channels(raw_image)
        corrected_rgb = color_constancy(raw_image, P.ILLUMINANT_TOP_FRAC)
        corrected_opp = opponent_channels(corrected_rgb)
        rg_contrast = color_contrast(
            corrected_opp[..., 1], P.COLOR_CONTRAST_RADIUS
        )
        by_contrast = color_contrast(
            corrected_opp[..., 2], P.COLOR_CONTRAST_RADIUS
        )
        retinex_res = compute_opponent_retinex(raw_image, extraction_scale=5.0)

        lum_raw = opponent[..., 0]
        lum = luminance_adapt(lum_raw)
        # [Scale-relative front end] Pixel constants in the contour tracer and
        # the Gabor cycles-per-pixel band are scaled to this frame's diagonal;
        # 1.0 at the 256x256 reference, so reference-resolution behaviour is
        # unchanged.
        _fe_scale = _front_end_scale(h, w)
        bank = GaborBank.build_default(_fe_scale)
        _lum_resp = compute_responses(bank, lum)
        simple = _lum_resp["simple"]
        complex_ = _lum_resp["complex"]
        phase = _lum_resp["phase"]

        # The remaining response populations feed ONLY the per-region rich
        # signatures, never the contour tracer (which reads `complex_`), so
        # they are skipped in contour-only mode.
        if rich:
            complex_pooled = gaussian_filter(
                complex_,
                sigma=(0, 0, P.DIVISIVE_POOL_SIGMA, P.DIVISIVE_POOL_SIGMA),
                mode="reflect",
            )

            # [B.46 adoption] Retinal ON/OFF substrate
            _on_img, _off_img = split_on_off(lum)
            _on_resp = compute_responses(bank, _on_img)
            _off_resp = compute_responses(bank, _off_img)

            # Attach Gabor/colour/retinex responses to hypercolumn regions
            _B1_AVAILABLE = all(name in globals() for name in (
                'compute_responses_b1', 'b1_register', 'b1_region_signature',
                'b1_class_name', '_B1_SLOT_WEIGHTS_BY_CLASS'))

            _lum_resp_b1 = None
            _on_resp_b1 = None
            _off_resp_b1 = None

            if _B1_AVAILABLE:
                b1_register(bank)
                _lum_resp_b1 = compute_responses_b1(bank, lum)
                _on_resp_b1 = compute_responses_b1(bank, _on_img)
                _off_resp_b1 = compute_responses_b1(bank, _off_img)

                nt_b1, nsf_b1 = len(bank.theta), len(bank.sf)
                assert _lum_resp_b1["complex_local"].shape[:2] == (nt_b1, nsf_b1), \
                    "Bank shape mismatch"

        # --- Stream 1: V1 Retinotopic Lattice ---
        lattice = V1RetinotopicLattice()
        hypercolumn_regions = lattice.tile_with_features(
            h, w, complex_, bank.theta, bank.sf, lum, corrected_rgb
        )

        if rich:
            for r in hypercolumn_regions:
                cx = max(0, min(w - 1, int(round(r["centroid"][0]))))
                cy = max(0, min(h - 1, int(round(r["centroid"][1]))))
                win = max(P.TEXTURE_MIN_WINDOW, int(r["radius_px"]))
                y0, y1 = max(0, cy - win), min(h, cy + win + 1)
                x0, x1 = max(0, cx - win), min(w, cx + win + 1)

                window_slice = complex_[:, :, y0:y1, x0:x1]
                r["interior_variance"] = float(np.var(window_slice)) if window_slice.size > 0 else 0.0
                r["coarseness"] = _compute_coarseness(window_slice)
                r["regularity"] = _compute_regularity(window_slice)
                r["gabor_complex"] = window_slice.mean(axis=(2, 3)).astype(np.float32)

                r["gabor_simple"] = simple[:, :, :, cy, cx].copy()
                r["gabor_phase"] = phase[:, :, cy, cx].copy()
                r["gabor_complex_pooled"] = complex_pooled[:, :, y0:y1, x0:x1].mean(axis=(2, 3)).astype(np.float32)
                r["gabor_complex_on"] = _on_resp["complex"][:, :, y0:y1, x0:x1].mean(axis=(2, 3)).astype(np.float32)
                r["gabor_complex_off"] = _off_resp["complex"][:, :, y0:y1, x0:x1].mean(axis=(2, 3)).astype(np.float32)
                r["gabor_phase_on"] = _on_resp["phase"][:, :, cy, cx].copy()
                r["gabor_phase_off"] = _off_resp["phase"][:, :, cy, cx].copy()
                r["opponent"] = opponent[cy, cx].copy()
                r["opponent_retinex"] = (
                    float(retinex_res["lum_comp"][cy, cx]),
                    float(retinex_res["rg_comp"][cy, cx]),
                    float(retinex_res["by_comp"][cy, cx]),
                    float(retinex_res["sat_norm"][cy, cx]),
                    float(retinex_res["hue"][cy, cx]),
                )
                r["corrected_opponent"] = corrected_opp[cy, cx].copy()
                r["corrected_rgb"] = corrected_rgb[cy, cx].copy()
                r["rg_contrast"] = float(rg_contrast[cy, cx])
                r["by_contrast"] = float(by_contrast[cy, cx])
                r["local_edge"] = _local_edge_at(lum, cy, cx)

                if _lum_resp_b1 is not None:
                    _sig_b1, _st_b1 = b1_region_signature(
                        bank, _lum_resp_b1, _on_resp_b1, _off_resp_b1,
                        y0, y1, x0, x1, lum, _on_img, _off_img,
                        cy, cx, r["radius_px"]
                    )
                    r["b1_sig"] = _sig_b1
                    r["b1_class"] = b1_class_name(len(bank.theta), len(bank.sf))
                    r["b1_stats"] = _st_b1

                    b4_register(bank)
                    _win_b4 = lum[y0:y1, x0:x1]
                    _lum_mean_b4 = float(_win_b4.mean()) if _win_b4.size else 0.0
                    _sig_b4 = compute_texture_rich(
                        _lum_resp_b1["complex_local"],
                        bank.theta,
                        bank.sf,
                        _lum_mean_b4,
                        y0, y1, x0, x1,
                    )
                    _cls_b4 = b4_class_name(len(bank.theta), len(bank.sf))
                    r["b4_sig"] = TypedSignature(
                        kind="CANONICAL",
                        class_=_cls_b4,
                        relation="SECOND_ORDER_STATS",
                        quantized=_sig_b4,
                    )
                    r["b4_class"] = _cls_b4

        # --- Stream 2: Association Field Contours ---
        if contours is None:
            contours = _extract_association_field_contours(
                complex_, bank.theta, bank.sf, h, w
            )

        lattice_rad = hypercolumn_regions[0]["radius_px"] if hypercolumn_regions else 8.0
        diag_ref = math.sqrt(h * h + w * w)

        contour_regions: List[Dict[str, Any]] = []
        for c_idx, c_info in enumerate(contours):
            pts = c_info["polyline"]
            if len(pts) < 3:
                continue
            pts_arr = np.array(pts, dtype=np.float64)
            cy = float(np.mean(pts_arr[:, 1]))
            cx = float(np.mean(pts_arr[:, 0]))

            mid_idx = len(pts) // 2
            sample_x, sample_y = pts[mid_idx]
            c_int_x = max(0, min(w - 1, int(round(sample_x))))
            c_int_y = max(0, min(h - 1, int(round(sample_y))))

            dists = np.hypot(pts_arr[:, 0] - cx, pts_arr[:, 1] - cy)
            rad_px = max(2.0, float(np.max(dists)))

            contour_regions.append({
                "anchor_id": len(hypercolumn_regions) + c_idx,
                "centroid": (float(cx), float(cy)),
                "radius_px": float(rad_px),
                "radius_norm": float(rad_px / (0.35 * diag_ref + 1e-6)),
                "rel_position": (float((cx / w) * 2.0 - 1.0), float((cy / h) * 2.0 - 1.0)),
                "rel_x": float(cx / w),
                "rel_y": float(cy / h),
                "area": int(c_info["length"] * 2.0),
                "circularity": 1.0 if c_info["is_closed"] else 0.2,
                "complexity": min(1.0, float(len(pts)) / 50.0),
                "roughness": 0.05,
                "isotropy": 0.2,
                "elongation": 0.2 if c_info["is_closed"] else 0.8,
                "major_axis_angle": float(c_info["theta_profile"][0]) if c_info["theta_profile"] else 0.0,
                "mean_rgb": tuple(float(x) for x in corrected_rgb[c_int_y, c_int_x][:3]),
                "boundary_polyline": c_info["polyline"],
                # [Stage 1.2] index into the contours list for robust
                # polyline resolution in Observation2D.__init__
                "_contour_idx": c_idx,
                "is_boundary": True,
                "sample_xy": (c_int_x, c_int_y),
            })

        all_regions = hypercolumn_regions + contour_regions

        # 4. Gabor & Retinex sampling across all anchors
        # Per-region response dictionaries (rich mode only). Contours, the
        # shock skeleton and the aspect descriptor do not read these.
        if rich:
            for r in all_regions:
                if r.get("is_boundary", False):
                    cx_s, cy_s = r["sample_xy"]
                    cx, cy = cx_s, cy_s
                    win = min(int(r["radius_px"]), int(lattice_rad * 1.5))
                else:
                    cx = max(0, min(w - 1, int(round(r["centroid"][0]))))
                    cy = max(0, min(h - 1, int(round(r["centroid"][1]))))
                    win = min(max(P.TEXTURE_MIN_WINDOW, int(r["radius_px"])), 32)

                y0, y1 = max(0, cy - win), min(h, cy + win + 1)
                x0, x1 = max(0, cx - win), min(w, cx + win + 1)

                window_slice = complex_[:, :, y0:y1, x0:x1]
                r["interior_variance"] = float(np.var(window_slice)) if window_slice.size > 0 else 0.0
                r["coarseness"] = _compute_coarseness(window_slice)
                r["regularity"] = _compute_regularity(window_slice)
                r["gabor_complex"] = window_slice.mean(axis=(2, 3)).astype(np.float32) if window_slice.size > 0 else np.zeros((len(bank.theta), len(bank.sf)), dtype=np.float32)

                r["gabor_simple"] = simple[:, :, :, cy, cx].copy()
                r["gabor_phase"] = phase[:, :, cy, cx].copy()
                r["gabor_complex_pooled"] = complex_pooled[:, :, y0:y1, x0:x1].mean(axis=(2, 3)).astype(np.float32) if window_slice.size > 0 else np.zeros_like(r["gabor_complex"])
                r["gabor_complex_on"] = _on_resp["complex"][:, :, y0:y1, x0:x1].mean(axis=(2, 3)).astype(np.float32) if window_slice.size > 0 else np.zeros_like(r["gabor_complex"])
                r["gabor_complex_off"] = _off_resp["complex"][:, :, y0:y1, x0:x1].mean(axis=(2, 3)).astype(np.float32) if window_slice.size > 0 else np.zeros_like(r["gabor_complex"])
                r["gabor_phase_on"] = _on_resp["phase"][:, :, cy, cx].copy()
                r["gabor_phase_off"] = _off_resp["phase"][:, :, cy, cx].copy()
                r["opponent"] = opponent[cy, cx].copy()
                r["opponent_retinex"] = (
                    float(retinex_res["lum_comp"][cy, cx]),
                    float(retinex_res["rg_comp"][cy, cx]),
                    float(retinex_res["by_comp"][cy, cx]),
                    float(retinex_res["sat_norm"][cy, cx]),
                    float(retinex_res["hue"][cy, cx]),
                )
                r["corrected_opponent"] = corrected_opp[cy, cx].copy()
                r["corrected_rgb"] = corrected_rgb[cy, cx].copy()
                r["rg_contrast"] = float(rg_contrast[cy, cx])
                r["by_contrast"] = float(by_contrast[cy, cx])
                r["local_edge"] = _local_edge_at(lum, cy, cx)

                # Skip B1/B4 for thin contours
                if _lum_resp_b1 is not None and not r.get("is_boundary", False):
                    _sig_b1, _st_b1 = b1_region_signature(
                        bank, _lum_resp_b1, _on_resp_b1, _off_resp_b1,
                        y0, y1, x0, x1, lum, _on_img, _off_img,
                        cy, cx, r["radius_px"]
                    )
                    r["b1_sig"] = _sig_b1
                    r["b1_class"] = b1_class_name(len(bank.theta), len(bank.sf))
                    r["b1_stats"] = _st_b1

                    b4_register(bank)
                    _win_b4 = lum[y0:y1, x0:x1]
                    _lum_mean_b4 = float(_win_b4.mean()) if _win_b4.size else 0.0
                    _sig_b4 = compute_texture_rich(
                        _lum_resp_b1["complex_local"],
                        bank.theta, bank.sf,
                        _lum_mean_b4,
                        y0, y1, x0, x1,
                    )
                    _cls_b4 = b4_class_name(len(bank.theta), len(bank.sf))
                    r["b4_sig"] = TypedSignature(
                        kind="CANONICAL",
                        class_=_cls_b4,
                        relation="SECOND_ORDER_STATS",
                        quantized=_sig_b4,
                    )
                    r["b4_class"] = _cls_b4

        # Macro envelope normalization
        if all_regions:
            all_cx = [r["centroid"][0] for r in all_regions]
            all_cy = [r["centroid"][1] for r in all_regions]
            env_cx = (min(all_cx) + max(all_cx)) / 2.0
            env_cy = (min(all_cy) + max(all_cy)) / 2.0
            env_ext = max(max(all_cx) - min(all_cx), max(all_cy) - min(all_cy), 1.0)
            for r in all_regions:
                r["rel_x"] = clamp((r["centroid"][0] - env_cx) / env_ext + 0.5, 0.0, 1.0)
                r["rel_y"] = clamp((r["centroid"][1] - env_cy) / env_ext + 0.5, 0.0, 1.0)

        for r in all_regions:
            r["chain_id"] = None
            r["chain_pos_start"] = 0.0
            r["chain_pos_end"] = 1.0

        obj = cls(all_regions, (h, w), flag_indices, effect_indices, contours=contours)
        obj._raw_image = raw_image
        obj._lum_adapted = lum
        obj._gabor_complex = complex_
        obj._gabor_thetas = bank.theta
        obj._gabor_sfs = bank.sf
        return obj

    def get_gabor_responses(self, u):
        """Per-region Gabor summary, or None on the region path.

        The data guard (not `hasattr`) is what tells the two paths apart:
        every Observation2D now HAS this method, so the B.7 branch must
        test the return value.
        """
        if 0 <= u < self._N:
            r = self._regions[u]
            if "gabor_simple" not in r:
                return None
            out = {"simple": r["gabor_simple"],
                   "complex": r["gabor_complex"],
                   "phase": r["gabor_phase"],
                   "complex_pooled": r["gabor_complex_pooled"]}
            for _pub, _src in (("complex_on", "gabor_complex_on"),
                               ("complex_off", "gabor_complex_off"),
                               ("phase_on", "gabor_phase_on"),
                               ("phase_off", "gabor_phase_off")):
                if _src in r:
                    out[_pub] = r[_src]
            return out
        return None

    def get_region_centroid_by_anchor(self, u):
        if 0 <= u < self._N:
            return (float(self._regions[u]["centroid"][0]),
                    float(self._regions[u]["centroid"][1]))
        return None

    def get_anchors_within(self, pos, radius):
        """Anchors within `radius` of `pos` (lazily built spatial hash)."""
        sh = getattr(self, "_spatial_hash", None)
        if sh is None:
            sh = SpatialHash(
                {i: (self._regions[i]["centroid"][0],
                     self._regions[i]["centroid"][1])
                 for i in range(self._N)},
                cell_size=max(P.SPATIAL_CELL_MIN, 1.0),
            )
            self._spatial_hash = sh
        return [a for a, _ in sh.neighbors(pos, radius)]

    def get_local_edge(self, u):
        if 0 <= u < self._N and "local_edge" in self._regions[u]:
            return self._regions[u]["local_edge"]
        return None

    def get_coarseness(self, u):
        if 0 <= u < self._N and "coarseness" in self._regions[u]:
            return float(self._regions[u]["coarseness"])
        return 0.0

    def get_regularity(self, u):
        if 0 <= u < self._N and "regularity" in self._regions[u]:
            return float(self._regions[u]["regularity"])
        return 0.0

    def get_opponent_channels(self, u):
        if 0 <= u < self._N:
            r = self._regions[u]
            if "opponent_retinex" in r:
                return r["opponent_retinex"]
            if "corrected_opponent" in r:
                co = r["corrected_opponent"]
                return (float(co[1]), float(co[2]), float(co[0]),
                        float(max(r["rg_contrast"], r["by_contrast"])))
        return None

    def get_anchor_opponent(self, u):
        return self.get_opponent_channels(u)

    @property
    def image_shape(self):
        return (self._h, self._w)

    def at_scale(self, level):
        """Downsampled view of this observation (raw-image path only)."""
        return downsample_observation(self, level)

def downsample_observation(obs, level):
    if not isinstance(obs, Observation2D):
        return obs
    if level == 0:
        return obs
    if getattr(obs, "_raw_image", None) is None:
        return obs
    factor = 2 ** level
    h, w = obs.image_shape
    nh = max(P.MIN_PYRAMID_DIM, h // factor)
    nw = max(P.MIN_PYRAMID_DIM, w // factor)
    small = _bilinear_resize(obs._raw_image, nh, nw)
    return Observation2D.from_image(
        small,
        flag_indices=list(obs._flag) if obs._flag else None,
        effect_indices=list(obs._effect) if obs._effect else None,
    )

def _bilinear_resize(img, nh, nw):
    from scipy.ndimage import zoom
    if img.ndim == 3:
        return zoom(img, (nh / img.shape[0], nw / img.shape[1], 1), order=1)
    return zoom(img, (nh / img.shape[0], nw / img.shape[1]), order=1)

