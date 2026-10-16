"""perception/lenses/v4.py -- extracted from derived.py
(Move 3.5).

MOVE ONLY: whole top-level statements, verbatim.
"""
from __future__ import annotations

import numpy as np
import math
import itertools
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import parameters as P
from core.hashing import (  # noqa: F401
    _detection_sort_key,
    _node_signature_hash,
    clamp,
    hash_64,
    serialize_int,
    serialize_tuple,
    serialize_value,
)
from core.quantize import _aang, _opens_toward, carrier_to_tangent  # noqa: F401
from core.types import (  # noqa: F401
    Detection,
    EdgeType,
    TypedSignature,
)
from perception.contour_field import field_from_luminance  # noqa: F401
from perception.chain_flanks import planes_from_rgb  # noqa: F401
from perception.contours import (  # noqa: F401
    texture_richness,
)
from perception.depth import (  # noqa: F401
    _extract_member_anchors,
    spatial_clustering,
)
from perception.lenses.derived_common import (  # noqa: F401
    CURVATURE_TEMPLATES,
    _canonical_curvature_sequence,
    entity_polylines,
    field_region_source,
    polyline_shape_descriptor,
    contour_self_symmetry,
    match_templates,
    qi,
    qi_circ,
    sample_curvature_profile,
)
from perception.lenses.derived_base import (  # noqa: F401
    HigherLens,
)

class SurfaceLens(HigherLens):
    """B12 [FIX v2] — V2/V4 Surface as Bounded Material Field (arity 7)."""

    def step(
        self,
        t: int,
        concept_graph: Any = None,
        sig_provider: Any = None,
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]] = None,
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]] = None,
        current_detections: Optional[Dict[str, List[Detection]]] = None,
        extraction_scale: float = 0.0,
        observation: Optional[Any] = None,
        luminance: Optional[Any] = None,
        **kwargs,
    ) -> List[Detection]:
        if not current_detections:
            return []
        color_by = {d.alpha: d for d in current_detections.get("Color", [])}
        tex_by = {d.alpha: d for d in current_detections.get("Texture", [])}
        bo_by: Dict[Any, List[Any]] = defaultdict(list)
        for d in current_detections.get("BorderOwnership", []):
            bo_by[d.alpha].append(d)

        # [W1] Entity source. FIELD_REGION_SOURCE takes priority when on: the
        # field regions ARE the entities, so the legacy Boundary set is used
        # only as a fallback when the field produced none (keeps the old eye
        # runnable when the field is empty). Emitting in the field source's
        # own anchor order also avoids needing stand-in detections that would
        # have to implement the whole detection interface just to sort.
        boundary_dets = current_detections.get("Boundary", [])
        field_src = None
        if bool(P.FIELD_REGION_SOURCE):
            field_src = field_region_source(observation, current_detections,
                                            anchor_positions)
            if field_src is None or len(field_src) == 0:
                field_src = None

        emitted = []
        if field_src is not None:
            # Field entities: iterate anchors in the source's own order. Each
            # member of the chain contributes its flank contrast, so a chain
            # emits ONE detection per chain (not per member) whose slots are
            # the aggregate over members -- one figure, one surface.
            from perception.chain_flanks import (
                chain_flank_fields, flank_signature_slots,
            )
            # The front end's own colour planes, and the field's median
            # wavelength, are both derived from the SAME rgb plane the chain
            # lens was driven with. Probing the observation for them is not
            # usable here: the region-construction path carries no luminance
            # and no channels.
            channels = None
            med_wl = None
            shape = None
            rgb = luminance
            if rgb is None:
                rgb = getattr(observation, "_raw_image", None)
            if rgb is not None:
                rgb = np.asarray(rgb)
                if rgb.ndim >= 2:
                    shape = (int(rgb.shape[0]), int(rgb.shape[1]))
                if rgb.ndim == 3:
                    # planes_from_rgb gives the SAME adapted luminance the
                    # chains were derived from; the retinex lum_comp this
                    # replaced is a different plane (max difference 0.95).
                    channels = planes_from_rgb(rgb)
                    if channels is not None:
                        try:
                            f = field_from_luminance(
                                np.asarray(channels.get("lum", [0]),
                                           dtype=np.float32))
                            sfs = np.asarray(f["sfs"], dtype=np.float64)
                            med_wl = float(np.median(
                                1.0 / np.maximum(sfs, 1e-6)))
                        except Exception:
                            med_wl = None

            for anchor in field_src.entity_anchors():
                det = next((d for d in
                            (current_detections or {}).get("PEAK_CHAIN", [])
                            if d.alpha == anchor), None)
                flank = None
                if det is not None and med_wl and channels:
                    try:
                        fld = chain_flank_fields(
                            det, anchor_positions, med_wl, channels,
                            image_shape=shape)
                        if fld is not None:
                            # flatten the named slot dict into the flank dict
                            # so _emit_one reads plain keys, and keep the raw
                            # slot set too for diagnostics
                            fld.update(flank_signature_slots(fld))
                            fld["slots"] = fld
                            flank = fld
                    except Exception:
                        flank = None
                bo_field = bo_by.get(anchor, [])
                c_det = color_by.get(anchor)
                t_det = tex_by.get(anchor)
                emitted.append(self._emit_one(
                    t, anchor, 1.0, bo_field, c_det, t_det,
                    field_src.get(anchor), flank,
                    current_detections, observation, anchor_positions))
        else:
            for b_det in sorted(boundary_dets, key=_detection_sort_key):
                anchor = b_det.alpha
                bo_field = bo_by.get(anchor)
                if not bo_field:
                    continue
                c_det = color_by.get(anchor)
                t_det = tex_by.get(anchor)
                if not c_det and not t_det:
                    continue
                emitted.append(self._emit_one(
                    t, anchor, b_det.z, bo_field, c_det, t_det,
                    None, None, current_detections, observation,
                    anchor_positions))
        self._emitted_total += len(emitted)
        return emitted

    def _emit_one(
        self,
        t: int,
        anchor: Any,
        z_val: Any,
        bo_field: List[Any],
        c_det: Any,
        t_det: Any,
        entity: Optional[Dict[str, Any]],
        flank: Optional[Dict[str, Any]],
        current_detections: Optional[Dict[str, List[Detection]]],
        observation: Optional[Any],
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
    ) -> Detection:
        """One Surface detection for one entity, from either entity source.

        The material slots come from the entity's own Colour/Texture
        detections when present, else from the field region's surface sample
        -- that fallback is what keeps a field region emittable without the
        segmenter's region descriptors.
        """
        curvs = [
            float(d.sigma.quantized[2])
            for d in bo_field
            if len(d.sigma.quantized) >= 4
        ]
        shape_bin = 0.0
        if curvs:
            mean_c = sum(curvs) / len(curvs)
            var_c = sum((c - mean_c) ** 2 for c in curvs) / len(curvs)
            shape_bin = qi(var_c, 8, 0.0, 12.25)

        c_q = (
            c_det.sigma.quantized
            if c_det is not None and len(c_det.sigma.quantized) >= 5
            else ()
        )
        lum = float(c_q[0]) if c_q else 0.0
        sat = float(c_q[3]) if c_q else 0.0
        hue = float(c_q[4]) if c_q else 0.0
        tex = texture_richness(t_det) if t_det is not None else 0.0

        # [W1] A field region with no region-level Color/Texture detection
        # still carries a material description. Flank contrast is preferred
        # (Slice 2): it is what makes a field region emittable AND what makes
        # its slots vary -- Slice 0c showed the interior-only fallback
        # collapses 80 emissions to 3 signatures, which dedupe to 0 new graph
        # nodes. The interior sample is the last-resort fallback.
        if entity is not None and not c_q and flank is not None:
            # Flank-derived material. The slots arrive as a NAMED dict from
            # flank_signature_slots; reading names (not positional indices)
            # is what keeps a real signal from silently collapsing to zero.
            lum = float(flank.get("lum_level_bin", 0.0))
            sat = float(flank.get("sat_contrast_bin", 0.0))
            hue = float(flank.get("lum_contrast_bin", 0.0))
        elif entity is not None and not c_q:
            lum = float(clamp(entity.get("lum", 0.0) * 7.0, 0.0, 7.0))
            o0, o1 = entity.get("opponent", (0.0, 0.0))
            sat = float(clamp(math.hypot(o0, o1) * 7.0, 0.0, 7.0))
            hue = 0.0

        best = max(
            bo_field,
            key=lambda d: float(d.sigma.quantized[3])
            if len(d.sigma.quantized) >= 4
            else -1.0,
        ) if bo_field else None
        grad_mag_bin, grad_side = 0.0, 0.0
        if best is not None and len(best.sigma.quantized) >= 4:
            grad_side = float(best.sigma.quantized[1])
            grad_mag_bin = float(best.sigma.quantized[3])

        quantized = (
            float(shape_bin),
            float(qi(lum, 8, 0.0, 7.0)),
            float(qi_circ(hue, 16.0, 16)),
            float(qi(sat, 8, 0.0, 7.0)),
            float(qi(clamp(float(tex or 0.0), 0.0, 7.0), 8, 0.0, 7.0)),
            float(grad_mag_bin),
            float(grad_side),
        )
        return Detection(
            k="Surface",
            source_layer=1,
            scale=0,
            sigma=TypedSignature(
                kind="CANONICAL",
                class_="SURFACE_PATCH",
                relation="SURFACE_MATERIAL",
                quantized=quantized,
            ),
            alpha=anchor,
            z=z_val,
            t=t,
            rho=(
                ("entity", anchor, 0, "+"),
                ("boundary", anchor, 0, "+"),
            ),
        )

class IllusoryContourLens(HigherLens):
    """B17 — V2 Modal Completion (Illusory Contours) (arity 3)."""

    def step(
        self,
        t: int,
        concept_graph: Any = None,
        sig_provider: Any = None,
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]] = None,
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]] = None,
        current_detections: Optional[Dict[str, List[Detection]]] = None,
        extraction_scale: float = 0.0,
        **kwargs,
    ) -> List[Detection]:
        if not current_detections:
            return []
        end_dets = current_detections.get("EndStopped", [])
        if not end_dets:
            end_dets = current_detections.get("Termination", [])
        if not end_dets or extraction_scale <= 0.0:
            return []

        max_gap = P.KAPPA_ILLUSORY_MAX_GAP * extraction_scale
        angle_tol = P.ILLUSORY_ANGLE_TOL_RAD
        candidates = []
        n_theta = P.ORIENTATION_N_THETA

        grid = defaultdict(list)
        for d in sorted(end_dets, key=_detection_sort_key):
            p = (anchor_positions or {}).get(d.alpha)
            if p is None and isinstance(d.alpha, tuple) and len(d.alpha) >= 2:
                p = d.alpha
            if p is not None and len(p) >= 2:
                grid[(int(p[0] // max_gap), int(p[1] // max_gap))].append((d, p))

        seen_pairs = set()
        for (cx, cy), cell in grid.items():
            neigh = []
            for off in itertools.product((-1, 0, 1), repeat=2):
                neigh.extend(grid.get((cx + off[0], cy + off[1]), ()))
            for i, (d1, p1) in enumerate(neigh):
                theta1_bin = (
                    d1.sigma.quantized[0] if len(d1.sigma.quantized) >= 1 else 0.0
                )
                s1 = d1.z
                for d2, p2 in neigh[i + 1 :]:
                    pair_key = (min(id(d1), id(d2)), max(id(d1), id(d2)))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    if d1.alpha == d2.alpha:
                        continue
                    theta2_bin = (
                        d2.sigma.quantized[0]
                        if len(d2.sigma.quantized) >= 1
                        else 0.0
                    )
                    s2 = d2.z

                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    dist = math.hypot(dx, dy)
                    if dist < 1e-6 or dist > max_gap:
                        continue

                    t1 = carrier_to_tangent((theta1_bin + 0.5) * math.pi / n_theta)
                    t2 = carrier_to_tangent((theta2_bin + 0.5) * math.pi / n_theta)
                    if _aang(t1, t2) > angle_tol:
                        continue
                    phi = math.atan2(dy, dx) % math.pi
                    if (
                        _aang(phi, t1) > angle_tol
                        or _aang(phi, t2) > angle_tol
                    ):
                        continue
                    if not (
                        _opens_toward(d1, (dx, dy), n_theta)
                        and _opens_toward(d2, (-dx, -dy), n_theta)
                    ):
                        continue

                    strength = s1 * s2 * math.exp(-dist / max_gap)
                    candidates.append((strength, t1, dist, p1, p2, d1, d2))

        candidates.sort(key=lambda x: -x[0])

        used, kept = set(), []
        for cand in candidates[: P.ILLUSORY_MAX_CANDIDATES]:
            k1 = serialize_value(cand[5].alpha)
            k2 = serialize_value(cand[6].alpha)
            if k1 in used or k2 in used:
                continue
            used.update((k1, k2))
            kept.append(cand)

        emitted = []
        for strength, theta, dist, p1, p2, d1, d2 in kept:
            if strength < 0.05:
                continue
            mid_x = (p1[0] + p2[0]) / 2.0
            mid_y = (p1[1] + p2[1]) / 2.0
            synthetic_anchor = hash_64(
                serialize_tuple([
                    serialize_value(d1.alpha),
                    serialize_value(d2.alpha),
                    serialize_int(int(round(mid_x))),
                    serialize_int(int(round(mid_y))),
                ])
            )
            alpha = synthetic_anchor
            theta_bin = int((theta / math.pi) * n_theta) % n_theta
            length_norm = min(1.0, dist / max_gap)
            length_bin = min(7, int(length_norm * 8.0))
            strength_norm = min(1.0, strength)
            strength_bin = min(7, int(strength_norm * 8.0))

            emitted.append(
                Detection(
                    k="IllusoryContour",
                    source_layer=1,
                    scale=0,
                    sigma=TypedSignature(
                        kind="CANONICAL",
                        class_="ILLUSORY_CONTOUR",
                        relation="MODAL_COMPLETION",
                        quantized=(
                            float(theta_bin),
                            float(length_bin),
                            float(strength_bin),
                        ),
                    ),
                    alpha=alpha,
                    z=strength,
                    t=t,
                    rho=(
                        ("entity", alpha, 0, "+"),
                        ("inducer_1", d1.alpha, 0, "+"),
                        ("inducer_2", d2.alpha, 0, "+"),
                    ),
                )
            )
        self._emitted_total += len(emitted)
        return emitted

class HolisticShapeLens(HigherLens):
    """B22 — IT Holistic Shape (arity 6).

    Emits ONE detection per entity from the RegionShape curvature histogram.
    Signature (arity 6, all 8.0 bins):
        (area_bin, elong_bin, sym_bin, axis_bin, fold_bin, curvature_hist_bin)
    """

    def __init__(self) -> None:
        super().__init__()
        self._skipped_no_region = 0
        self._skipped_no_polyline = 0
        self._candidates_evaluated = 0

    def get_diagnostics(self) -> Dict[str, int]:
        return {
            "emitted_total": self._emitted_total,
            "skipped_no_region": self._skipped_no_region,
            "skipped_no_polyline": self._skipped_no_polyline,
            "candidates_evaluated": self._candidates_evaluated,
        }

    def step(
        self,
        t: int,
        concept_graph: Any = None,
        sig_provider: Any = None,
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]] = None,
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]] = None,
        current_detections: Optional[Dict[str, List[Detection]]] = None,
        extraction_scale: float = 0.0,
        envelope_mask: Optional[
            Tuple[Tuple[float, float, float, float], ...]
        ] = None,
        observation: Optional[Any] = None,
        **kwargs,
    ) -> List[Detection]:
        if observation is None:
            return []
        # [W1] Field-derived or segmenter entities; same anchor -> polyline
        # contract, so the body below is source-agnostic.
        polys = entity_polylines(observation, current_detections,
                                 anchor_positions,
                                 use_field=bool(P.FIELD_REGION_SOURCE))
        if not polys:
            self._skipped_no_region += 1
            return []

        emitted = []
        for u, poly in polys.items():
            self._candidates_evaluated += 1
            if not poly or len(poly) < 3:
                self._skipped_no_polyline += 1
                continue

            (hist_bins, area_bin, elong_bin,
             sym_bin, axis_bin, fold_bin, _seq) = polyline_shape_descriptor(
                poly, observation
            )

            # Curvature histogram summary bin (first non-zero or mean)
            curv_hist_bin = float(hist_bins[0]) if hist_bins else 0.0


            quantized = (
                float(area_bin),
                float(elong_bin),
                float(sym_bin),
                float(axis_bin),
                float(fold_bin),
                curv_hist_bin,
            )

            emitted.append(
                Detection(
                    k="HolisticShape",
                    source_layer=2,
                    scale=0,
                    sigma=TypedSignature(
                        kind="CANONICAL",
                        class_="AIT_HOLISTIC_SHAPE",
                        relation="HOLISTIC_SHAPE",
                        quantized=quantized,
                    ),
                    alpha=u,
                    z=1.0,
                    t=t,
                    rho=(("entity", u, 0, "+"),),
                )
            )
        self._emitted_total += len(emitted)
        return emitted

class CurveComplexLens(HigherLens):
    """B.8 — V4 curve-bundles over Contour and Junction.

    Builds a neighbour graph over the V2 nodes from CONTOUR_CONTINUITY and
    JUNCTION_CONTOUR edges, then takes connected components of size >= 2
    that contain at least one Contour. The component's shape (how many
    contours, how many junctions, whether it is a single closed loop) is
    the V4 signature.
    """

    def __init__(self) -> None:
        super().__init__()
        self._skipped_no_position = 0
        self._skipped_no_component = 0
        self._candidates_evaluated = 0

    def get_diagnostics(self) -> Dict[str, int]:
        return {
            "emitted_total": self._emitted_total,
            "skipped_no_position": self._skipped_no_position,
            "skipped_no_component": self._skipped_no_component,
            "candidates_evaluated": self._candidates_evaluated,
        }

    def step(
        self,
        t: int,
        concept_graph: Any,
        sig_provider: Any,
        anchor_positions: Dict[Any, Tuple[float, ...]],
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]],
        current_detections: Dict[str, List[Detection]],
        extraction_scale: float = 0.0,
    ) -> List[Detection]:
        if extraction_scale <= 0.0:
            self._skipped_no_position += 1
            return []

        contour_dets = current_detections.get("Contour", [])
        junction_dets = current_detections.get("Junction", [])
        if not contour_dets and not junction_dets:
            self._skipped_no_component += 1
            return []

        node_id_to_detection: Dict[int, Detection] = {}
        node_id_to_kind: Dict[int, str] = {}
        for det in sorted(contour_dets, key=_detection_sort_key):
            if len(det.sigma.quantized) < 4:
                continue
            nid = concept_graph.get_or_create_pattern_node(
                det.k, _node_signature_hash(det.sigma), signature=det.sigma
            )
            node_id_to_detection.setdefault(nid, det)
            node_id_to_kind.setdefault(nid, "Contour")
        for det in sorted(junction_dets, key=_detection_sort_key):
            if len(det.sigma.quantized) < 3:
                continue
            nid = concept_graph.get_or_create_pattern_node(
                det.k, _node_signature_hash(det.sigma), signature=det.sigma
            )
            node_id_to_detection.setdefault(nid, det)
            node_id_to_kind.setdefault(nid, "Junction")

        def is_closed(contour_nid: int) -> bool:
            det = node_id_to_detection.get(contour_nid)
            if det is None or len(det.sigma.quantized) < 4:
                return False
            return int(det.sigma.quantized[3]) == 1

        # Weight filter: THETA_CONTOUR_EDGE for BOTH edge types. V4 is
        # structural topology over pairs that have already proven their
        # continuity through the V2 EMA, so one gate is right; a second,
        # stricter gate would drop bundles whose junction links are
        # legitimately weaker than their contour links.
        # Every admitted V2 node enters the component graph, including
        # degree-0 nodes: an isolated contour is a singleton component and
        # may emit as topology class 0 (single closed loop) or 1 (single
        # open contour). Seeding only edge-endpoints made those classes
        # unreachable even after the size guard was lifted.
        G_v4: Dict[int, Set[int]] = defaultdict(set)
        for nid in node_id_to_detection:
            G_v4[nid]  # touch: register degree-0 nodes as components
        for nid in node_id_to_detection:
            for edge in concept_graph.get_outgoing_edges(nid):
                if edge.edge_type not in (
                    EdgeType.CONTOUR_CONTINUITY, EdgeType.JUNCTION_CONTOUR
                ):
                    continue
                if edge.weight < P.THETA_CONTOUR_EDGE:
                    continue
                if edge.child not in node_id_to_detection:
                    continue
                G_v4[nid].add(edge.child)
                G_v4[edge.child].add(nid)
            for edge in concept_graph.get_incoming_edges(nid):
                if edge.edge_type not in (
                    EdgeType.CONTOUR_CONTINUITY, EdgeType.JUNCTION_CONTOUR
                ):
                    continue
                if edge.weight < P.THETA_CONTOUR_EDGE:
                    continue
                if edge.parent not in node_id_to_detection:
                    continue
                G_v4[nid].add(edge.parent)
                G_v4[edge.parent].add(nid)

        visited: Set[int] = set()
        components: List[Set[int]] = []
        for start in sorted(G_v4.keys()):
            if start in visited:
                continue
            comp: Set[int] = set()
            stack = [start]
            visited.add(start)
            while stack:
                x = stack.pop()
                comp.add(x)
                for y in sorted(G_v4.get(x, ())):
                    if y not in visited:
                        visited.add(y)
                        stack.append(y)
            components.append(comp)

        emitted: List[Detection] = []
        for comp in components:
            # [Fix: unreachable topology classes] The old `len(comp) < 2`
            # guard skipped single-contour components, which made topology
            # classes 0 (single closed loop) and 1 (single open contour) --
            # two of the four classes this lens was designed to emit --
            # unreachable, and starved the whole Part/ObjectShape tier
            # whenever an object's structure condenses into one outline.
            # The contour-content check just below already rejects pure-
            # junction comps; size is not evidence.
            contour_ids = sorted(
                [n for n in comp if node_id_to_kind[n] == "Contour"]
            )
            junction_ids = sorted(
                [n for n in comp if node_id_to_kind[n] == "Junction"]
            )
            # A bundle of pure junctions carries no curve structure.
            if len(contour_ids) < 1:
                continue

            self._candidates_evaluated += 1

            n_contours_bin = min(5, len(contour_ids))
            n_junctions_bin = min(5, len(junction_ids))
            if (
                len(contour_ids) == 1
                and len(junction_ids) == 0
                and is_closed(contour_ids[0])
            ):
                topology_class_bin = 0
            elif (
                len(contour_ids) >= 1
                and len(junction_ids) == 0
                and not any(is_closed(c) for c in contour_ids)
            ):
                topology_class_bin = 1
            elif len(junction_ids) >= 2:
                topology_class_bin = 2
            else:
                topology_class_bin = 3

            sorted_member_ids = sorted(comp)
            # alpha is a live observation anchor, so it is the FIRST
            # MEMBER's anchor, not a synthesised centroid (which would not be
            # a key in anchor_positions and could not be looked up later).
            first_member_node_id = sorted_member_ids[0]
            first_member_anchor = node_id_to_detection[
                first_member_node_id
            ].alpha

            # [Object-center pooling] Carry the complex's full V1 member
            # footprint ("member" anchor roles) so downstream lenses can
            # locate the structure CENTROID. Rim-only identity (first
            # member) sits on the object boundary; the footprint centroid
            # sits inside it. Deterministic: anchors deduplicated and
            # canonically ordered; ordinals re-indexed after ordering.
            footprint: List[Any] = []
            seen_anchor: Set[Any] = set()
            for mid in sorted_member_ids:
                for a in _extract_member_anchors(node_id_to_detection[mid]):
                    if a in seen_anchor:
                        continue
                    seen_anchor.add(a)
                    footprint.append(a)
            if not footprint:
                footprint = [first_member_anchor]
            footprint = sorted(footprint, key=serialize_value)

            emitted.append(Detection(
                k="CurveComplex", source_layer=2, scale=0,
                sigma=TypedSignature(
                    kind="CANONICAL", class_="CURVE_COMPLEX",
                    relation="TOPOLOGY_CLASS",
                    quantized=(float(n_contours_bin), float(n_junctions_bin),
                               float(topology_class_bin)),
                ),
                alpha=first_member_anchor,
                z=min(1.0, (len(contour_ids) + len(junction_ids)) / 6.0), t=t,
                rho=(("entity", first_member_anchor, 0, "+"),)
                    + tuple(
                        ("member", footprint[i], i, "+")
                        for i in range(len(footprint))
                    )
                    + tuple(
                        ("member_node", (sorted_member_ids[i],), i, "+")
                        for i in range(len(sorted_member_ids))
                    ),
            ))

        self._emitted_total += len(emitted)
        return emitted

class CurveComplexLens_B13(HigherLens):
    """B13 — V4 Curve Arrangements & Shape Fragments (arity 24)."""

    def _spatial_clustering(self, dets, anchor_positions, radius):
        return spatial_clustering(dets, anchor_positions, radius)

    def step(
        self,
        t: int,
        concept_graph: Any = None,
        sig_provider: Any = None,
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]] = None,
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]] = None,
        current_detections: Optional[Dict[str, List[Detection]]] = None,
        extraction_scale: float = 0.0,
        **kwargs,
    ) -> List[Detection]:
        if not current_detections:
            return []
        contour_dets = current_detections.get("Contour", [])
        junction_dets = current_detections.get("Junction", [])
        if not contour_dets:
            return []

        radius = (
            P.KAPPA_CURVE_COMPLEX_RADIUS * extraction_scale
            if extraction_scale > 0
            else 50.0
        )
        clusters = (
            self._spatial_clustering(contour_dets, anchor_positions, radius)
            if anchor_positions
            else []
        )
        if not clusters:
            clusters = [contour_dets]

        emitted = []
        for cluster in clusters:
            emitted.extend(
                self._emit_cluster(
                    t, cluster, junction_dets, anchor_positions or {}, extraction_scale, concept_graph=concept_graph
                )
            )
        self._emitted_total += len(emitted)
        return emitted

    def _emit_cluster(
        self,
        t: int,
        contour_dets: List[Detection],
        junction_dets: List[Detection],
        anchor_positions: Dict[Any, Tuple[float, ...]],
        extraction_scale: float,
        concept_graph: Any = None,
    ) -> List[Detection]:
        n_contours = len(contour_dets)
        if n_contours == 0:
            return []
        n_junctions = len(junction_dets)

        _pos = [anchor_positions.get(c.alpha) for c in contour_dets]
        _pos = [p for p in _pos if p is not None and len(p) >= 2]
        if _pos:
            span = max(
                math.hypot(a[0] - b[0], a[1] - b[1])
                for a in _pos
                for b in _pos
            )
        else:
            span = 0.0
        span_norm = min(1.0, span / max(1.0, 6.0 * extraction_scale))
        extent_bin = qi(span_norm, 8, 0.0, 1.0)

        closure_sum = 0.0
        valid_c = 0
        for c in contour_dets:
            if len(c.sigma.quantized) >= 4:
                closure_sum += c.sigma.quantized[3]
                valid_c += 1
        closure_score = closure_sum / max(1, valid_c)

        template_hits = [0.0] * P.CURVE_COMPLEX_TEMPLATE_BINS
        for c in contour_dets:
            if len(c.sigma.quantized) >= 27:
                seq = c.sigma.quantized[7:27]
                for k in range(20):
                    t_bin = int((k / 20.0) * P.CURVE_COMPLEX_TEMPLATE_BINS)
                    t_bin = min(P.CURVE_COMPLEX_TEMPLATE_BINS - 1, t_bin)
                    energy = (seq[k] - 4.0) / 4.0
                    template_hits[t_bin] += energy
            elif len(c.sigma.quantized) >= 7:
                for idx, k in enumerate((4, 5, 6)):
                    t_bin = int((idx / 3.0) * P.CURVE_COMPLEX_TEMPLATE_BINS)
                    energy = (float(c.sigma.quantized[k]) - 3.5) / 3.5
                    template_hits[t_bin] += energy

        max_abs = max(1e-6, max(abs(h) for h in template_hits))
        template_bins = [
            min(7.0, max(0.0, (h / max_abs) * 3.5 + 3.5))
            for h in template_hits
        ]

        n_c_bin = qi(n_contours, 8, 0.0, 7.0)
        n_j_bin = qi(n_junctions, 8, 0.0, 7.0)
        c_bin = qi(clamp(closure_score, 0.0, 1.0), 8, 0.0, 1.0)

        shape_pop = [0.0] * len(CURVATURE_TEMPLATES)
        n_pops = 0
        for c in contour_dets:
            prof = sample_curvature_profile(c, anchor_positions, n=5)
            if prof is None or float(np.abs(prof).sum()) < 1e-6:
                continue
            _best, _score, pop = match_templates(prof)
            for i in range(len(shape_pop)):
                shape_pop[i] += float(pop[i])
            n_pops += 1
        if n_pops > 0:
            shape_pop = [v / n_pops for v in shape_pop]
        shape_bins = [qi(v, 8, 0.0, 1.0) for v in shape_pop]

        quantized = (
            (float(n_c_bin), float(n_j_bin), float(extent_bin), float(c_bin))
            + tuple(float(b) for b in template_bins)
            + tuple(float(b) for b in shape_bins)
        )

        contour_dets_sorted = sorted(contour_dets, key=_detection_sort_key)
        first_anchor = contour_dets_sorted[0].alpha

        c_nids = []
        if concept_graph is not None:
            for d in contour_dets_sorted:
                nid = concept_graph.get_or_create_pattern_node(
                    d.k, _node_signature_hash(d.sigma), signature=d.sigma
                )
                c_nids.append(nid)

        rho = (
            (("entity", first_anchor, 0, "+"),)
            + tuple(
                ("member", d.alpha, i, "+")
                for i, d in enumerate(contour_dets_sorted)
            )
            + tuple(
                ("member_node", (c_nids[i],), i, "+")
                for i in range(len(c_nids))
            )
        )

        return [
            Detection(
                k="CurveComplex",
                source_layer=2,
                scale=0,
                sigma=TypedSignature(
                    kind="CANONICAL",
                    class_="CURVE_COMPLEX",
                    relation="SHAPE_ARRANGEMENT",
                    quantized=quantized,
                ),
                alpha=first_anchor,
                z=max(extent_bin / 7.0, closure_score),
                t=t,
                rho=rho,
            )
        ]

class RegionShapeLens(HigherLens):
    """B14 + B22 — V4/IT Boundary Conformation & Self-Symmetry (arity 21)."""

    def step(
        self,
        t: int,
        concept_graph: Any = None,
        sig_provider: Any = None,
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]] = None,
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]] = None,
        current_detections: Optional[Dict[str, List[Detection]]] = None,
        extraction_scale: float = 0.0,
        envelope_mask: Optional[
            Tuple[Tuple[float, float, float, float], ...]
        ] = None,
        observation: Optional[Any] = None,
        **kwargs,
    ) -> List[Detection]:
        if observation is None:
            return []
        # [W1] Entities come from field regions built on PEAK_CHAINs when
        # FIELD_REGION_SOURCE is on, else from the segmenter. Both paths
        # yield anchor -> ordered polyline, so the body below is unchanged.
        polys = entity_polylines(observation, current_detections,
                                 anchor_positions,
                                 use_field=bool(P.FIELD_REGION_SOURCE))
        emitted = []
        for u, poly in polys.items():
            if not poly or len(poly) < 3:
                continue
            (hist_bins, area_bin, elong_bin,
             sym_bin, axis_bin, fold_bin, _seq) = polyline_shape_descriptor(
                poly, observation
            )

            quantized = hist_bins + (float(area_bin), float(elong_bin))
            quantized = quantized + (sym_bin, axis_bin, fold_bin)

            _det = Detection(
                k="RegionShape",
                source_layer=0,
                scale=0,
                sigma=TypedSignature(
                    kind="CANONICAL",
                    class_="REGION_SHAPE",
                    relation="BOUNDARY_CONFORMATION",
                    quantized=quantized,
                ),
                alpha=u,
                z=1.0,
                t=t,
                rho=(("entity", u, 0, "+"),),
            )
            object.__setattr__(_det, "_curv_profile", _seq)
            emitted.append(_det)
        self._emitted_total += len(emitted)
        return emitted

