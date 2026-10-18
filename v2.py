"""perception/lenses/v2.py -- extracted from derived.py
(Move 3.5).

MOVE ONLY: whole top-level statements, verbatim.
"""
from __future__ import annotations

import numpy as np
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import parameters as P
from core.hashing import (  # noqa: F401
    _detection_sort_key,
    _node_signature_hash,
    clamp,
    serialize_value,
)
from core.quantize import (  # noqa: F401
    _q_signed,
    _q_unsigned,
    q_linear,
)
from core.quantize import _aang, carrier_to_tangent  # noqa: F401
from core.types import (  # noqa: F401
    Detection,
    EdgeType,
    TypedSignature,
)
from core.registries import population_for_bins  # noqa: F401
from core.spatial import _pos_dist  # noqa: F401
from parameters._text import (  # noqa: F401
    BO_COLOR_WEIGHT,
    BO_CUE_SCALE,
    BO_CURVATURE_WEIGHT,
    BO_GRAVITY_SCALE,
    BO_GRAVITY_WEIGHT,
    BO_RICHNESS_FALLBACK,
    BO_TEXTURE_BIN_MAX,
    BO_TEXTURE_WEIGHT,
)
from perception.lenses.derived_common import (  # noqa: F401
    _JUNCTION_CANONICAL_BINS,
    _arm_tangent_stats,
    _collect_continuity_neighbours,
    _contour_slots,
    entity_polylines,
    field_region_source,
    qi,
)
from perception.lenses.derived_base import (  # noqa: F401
    HigherLens,
)

class BorderOwnershipLens(HigherLens):
    """B.6 [FIX v2] — V2 figure-ground as a PER-BOUNDARY-POINT FIELD.

    Emits ONE BORDER_OWNERSHIP detection per sampled boundary vertex.
    Signature (arity 4, linear):
        (arc_pos_bin, signed_dir_bin, curvature_bin, strength_bin)
    `alpha` is the live region anchor `a`; rho carries entity/neighbour/vertex.

    Field length <= BO_FIELD_SAMPLE_COUNT per boundary. Open AND closed arcs
    are sampled (the prior region-level body only handled closed polygons,
    silently dropping the many open chains). Cues are region-level (constant
    along this vertex set) except curvature, which is per-vertex.
    """

    def __init__(self) -> None:
        super().__init__()
        self._skipped_no_boundary = 0
        self._skipped_no_polyline_api = 0
        self._skipped_no_neighbor = 0
        self._candidates_evaluated = 0

    def get_diagnostics(self) -> Dict[str, int]:
        return {
            "emitted_total": self._emitted_total,
            "skipped_no_boundary": self._skipped_no_boundary,
            "skipped_no_polyline_api": self._skipped_no_polyline_api,
            "skipped_no_neighbor": self._skipped_no_neighbor,
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
        envelope_mask: Optional[
            Tuple[Tuple[float, float, float, float], ...]
        ] = None,
        observation: Optional[Any] = None,
        update_homeostasis: bool = True,
        object_envelope: Optional[
            Tuple[float, float, float, float]
        ] = None,
        scale_level: int = 0,
    ) -> List[Detection]:
        if extraction_scale <= 0.0 or not anchor_positions:
            return []
        boundary_dets = current_detections.get("Boundary", [])
        # [W1] Field entities are CHAINS, not segmenter boundaries. Their
        # anchors are PEAK_ANCHOR_BASE + i / + CHAIN_ANCHOR_OFFSET + k, so
        # iterating the legacy Boundary detections would never reach them.
        field_src = None
        if bool(P.FIELD_REGION_SOURCE):
            field_src = field_region_source(observation, current_detections,
                                            anchor_positions)
            if field_src is None or len(field_src) == 0:
                field_src = None
        if field_src is None and not boundary_dets:
            self._skipped_no_boundary += 1
            return []
        radius = P.KAPPA_BORDER_OWNERSHIP * extraction_scale
        if radius < 1e-9:
            return []

        tex_by: Dict[Any, Detection] = {}
        col_by: Dict[Any, Detection] = {}
        for d in current_detections.get("Texture", []):
            tex_by.setdefault(d.alpha, d)
        for d in current_detections.get("Color", []):
            col_by.setdefault(d.alpha, d)

        def _tex_rich(anchor: Any) -> Optional[float]:
            d = tex_by.get(anchor)
            if d is None or len(d.sigma.quantized) < 2:
                return None
            rough = float(d.sigma.quantized[0]) / BO_TEXTURE_BIN_MAX
            iso = float(d.sigma.quantized[1]) / BO_TEXTURE_BIN_MAX
            return rough * (1.0 - iso)

        def _sat(anchor: Any) -> Optional[float]:
            d = col_by.get(anchor)
            if d is None or len(d.sigma.quantized) < 3:
                return None
            r, g, b = (float(d.sigma.quantized[i]) for i in range(3))
            mx = max(r, g, b)
            return 0.0 if mx <= 0.0 else 1.0 - (min(r, g, b) / mx)

        # [W1] One entity list for both arms so the body below is shared.
        # The field arm's entities are chains (z = 1.0, a chain has no
        # boundary strength of its own); the legacy arm's are segmenter
        # Boundary detections with their measured z.
        if field_src is not None:
            entities = [(a, None) for a in field_src.entity_anchors()]
        else:
            entities = [(d.alpha, d.z) for d
                        in sorted(boundary_dets, key=_detection_sort_key)]

        emitted: List[Detection] = []
        for a, z_a in entities:
            pos_a = anchor_positions.get(a)
            if pos_a is None or len(pos_a) < 2:
                continue

            self._candidates_evaluated += 1

            richness_a = (
                0.5 * (_tex_rich(a) or BO_RICHNESS_FALLBACK)
                + 0.5 * (_sat(a) or BO_RICHNESS_FALLBACK)
            )
            best_b, best_key, best_diff = None, None, -float("inf")
            for other, pos_b in anchor_positions.items():
                if other == a or pos_b is None or len(pos_b) < 2:
                    continue
                if _pos_dist(pos_a, pos_b) >= radius:
                    continue
                tb_, sb_ = _tex_rich(other), _sat(other)
                richness_b = (
                    0.5 * (tb_ if tb_ is not None else BO_RICHNESS_FALLBACK)
                    + 0.5 * (sb_ if sb_ is not None else BO_RICHNESS_FALLBACK)
                )
                diff = richness_a - richness_b
                key = serialize_value(other)
                if (diff > best_diff
                        or (diff == best_diff
                            and (best_key is None or key < best_key))):
                    best_diff, best_b, best_key = diff, other, key
            if best_b is None:
                self._skipped_no_neighbor += 1
                continue
            pos_b = anchor_positions[best_b]

            ta, tb = _tex_rich(a), _tex_rich(best_b)
            tex_cue = clamp((ta - tb) * BO_CUE_SCALE, -1.0, 1.0) \
                if (ta is not None and tb is not None) else 0.0
            sa, sb = _sat(a), _sat(best_b)
            col_cue = clamp((sa - sb) * BO_CUE_SCALE, -1.0, 1.0) \
                if (sa is not None and sb is not None) else 0.0
            grav_cue = clamp(
                (pos_a[1] - pos_b[1]) / (extraction_scale * BO_GRAVITY_SCALE),
                -1.0, 1.0)

            poly = None
            if bool(P.FIELD_REGION_SOURCE):
                src = field_region_source(observation, current_detections,
                                          anchor_positions)
                if src is not None:
                    e = src.get(a)
                    if e is not None and len(e["polyline"]) >= 3:
                        poly = e["polyline"]
            if poly is None and observation is not None \
                    and hasattr(observation, "get_boundary_polyline"):
                poly = observation.get_boundary_polyline(a)
            if not poly or len(poly) < 3:
                r_circ = extraction_scale if extraction_scale > 0.0 else 1.0
                poly = [
                    (pos_a[0] + r_circ * math.cos(2.0 * math.pi * k / 16.0),
                     pos_a[1] + r_circ * math.sin(2.0 * math.pi * k / 16.0))
                    for k in range(16)
                ]
                poly.append(poly[0])

            pts = [(float(p[0]), float(p[1])) for p in poly]
            closed = (abs(pts[0][0] - pts[-1][0]) < 1e-9
                      and abs(pts[0][1] - pts[-1][1]) < 1e-9)
            if closed:
                pts = pts[:-1]          # drop duplicate terminal point
            n = len(pts)
            if n < 3:
                continue

            seg = ([math.hypot(pts[(i + 1) % n][0] - pts[i][0],
                               pts[(i + 1) % n][1] - pts[i][1]) for i in range(n)]
                   if closed else
                   [math.hypot(pts[i + 1][0] - pts[i][0],
                               pts[i + 1][1] - pts[i][1]) for i in range(n - 1)])
            if sum(seg) < 1e-9:
                continue
            sample_count = max(3, min(P.BO_FIELD_SAMPLE_COUNT, n))
            span = max(2, int(0.04 * n))   # ~4% of the perimeter, min 2
            area2 = sum(pts[i][0] * pts[(i + 1) % n][1]
                        - pts[(i + 1) % n][0] * pts[i][1]
                        for i in range(n)) if closed else 0.0
            s_w = 1.0 if area2 >= 0.0 else -1.0
            # [FIGURE-GROUND] Per-vertex ownership cue combination. The cue set
            # and its weights live in ONE place (perception/ownership.py
            # `combine_ownership_cues` + BO_WEIGHTS). The enclosure cue comes
            # from the bounded-diffusion interior field
            # (perception/interior_field.py) and is computed once per
            # observation (memoised by content hash + fingerprint).
            _fusion = bool(getattr(P, "BO_CUE_FUSION_ENABLED", False))
            _enc_w = 0.0
            if _fusion:
                _enc_w = abs(float((getattr(P, "BO_WEIGHTS", {}) or {})
                                  .get("enclosure", 0.0)))
            _enc_field = None
            if _enc_w > 0.0:
                from perception.interior_field import interior_field_for
                _enc_field = interior_field_for(observation)
            for k in range(sample_count):
                frac = (k + 0.5) / sample_count
                idx = min(n - 1, int(frac * n))
                if closed:
                    prv = pts[(idx - span) % n]
                    cur = pts[idx]
                    nxt = pts[(idx + span) % n]
                else:
                    prv = pts[max(0, idx - span)]
                    cur = pts[idx]
                    nxt = pts[min(n - 1, idx + span)]
                v1x, v1y = cur[0] - prv[0], cur[1] - prv[1]
                v2x, v2y = nxt[0] - cur[0], nxt[1] - cur[1]
                if closed:
                    turn = math.atan2(v1x * v2y - v1y * v2x,
                                      v1x * v2x + v1y * v2y)
                    curv = clamp(s_w * turn / (math.pi / 2.0), -1.0, 1.0)
                else:
                    curv = 0.0
                signed_dir = clamp(
                    curv * BO_CURVATURE_WEIGHT
                    + tex_cue * BO_TEXTURE_WEIGHT
                    + col_cue * BO_COLOR_WEIGHT
                    + grav_cue * BO_GRAVITY_WEIGHT,
                    -1.0, 1.0)
                if _fusion:
                    from perception.ownership import (
                        combine_ownership_cues)
                    cues = {
                        "convexity": curv,
                        "texture": tex_cue,
                        "color": col_cue,
                        "gravity": grav_cue,
                    }
                    if _enc_field is not None:
                        # Normal = local tangent rotated +90 deg. The cue is +1
                        # when the +normal side is enclosed, so it points at the
                        # figure.
                        _tn = math.hypot(v1x, v1y)
                        if _tn > 1e-9:
                            from perception.interior_field import (
                                ownership_from_enclosure)
                            cues["enclosure"] = ownership_from_enclosure(
                                _enc_field, (cur[0], cur[1]),
                                (-v1y / _tn, v1x / _tn))
                    weights = dict(getattr(P, "BO_WEIGHTS", {}) or {})
                    weights["texture"] = BO_TEXTURE_WEIGHT
                    weights["color"] = BO_COLOR_WEIGHT
                    weights["gravity"] = BO_GRAVITY_WEIGHT
                    signed_dir = combine_ownership_cues(cues, weights)
                strength = abs(signed_dir)
                _bo_slots = (
                    float(_q_unsigned(frac, 1.0 / 8.0, max_bins=7)),
                    float(_q_signed(signed_dir, max_bins=8)),
                    float(_q_signed(curv, max_bins=8)),
                    float(_q_unsigned(strength, 1.0 / 8.0, max_bins=7)),
                )
                emitted.append(Detection(
                    k="BorderOwnership", source_layer=1, scale=0,
                    sigma=TypedSignature(
                        kind="CANONICAL", class_="BORDER_OWNERSHIP",
                        relation="FIGURE_SIDE_FIELD",
                        quantized=_bo_slots,
                        population=population_for_bins(
                            "BORDER_OWNERSHIP", _bo_slots),
                    ),
                    alpha=a, z=strength, t=t,
                    rho=(("entity", a, 0, "+"),
                         ("neighbor", best_b, 0, "+"),
                         ("vertex", idx, 0, "+")),
                ))
        self._emitted_total += len(emitted)
        return emitted

class ContourLens(HigherLens):
    """B.6 — V2 contour chains over Orientation segments.

    Builds a neighbour graph over Orientation pattern nodes from
    ORIENTATION_CONTINUITY and CURVATURE edges, prunes edges running
    against the figure-ground direction implied by BorderOwnership, then
    takes connected components of degree <= 2 as contour chains.
    """

    def step(
        self,
        t: int,
        concept_graph: Any,
        sig_provider: Any,
        anchor_positions: Dict[Any, Tuple[float, ...]],
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]],
        current_detections: Dict[str, List[Detection]],
        extraction_scale: float = 0.0,
        envelope_mask: Optional[
            Tuple[Tuple[float, float, float, float], ...]
        ] = None,
        observation: Optional[Any] = None,
        update_homeostasis: bool = True,
        object_envelope: Optional[
            Tuple[float, float, float, float]
        ] = None,
        scale_level: int = 0,
    ) -> List[Detection]:
        if extraction_scale <= 0.0:
            self._skipped_no_position += 1
            return []

        # Node ids of the Orientation segments, which are the contour's
        # members. Arity must be 3 (theta, elongation, anchor trunk).
        node_id_to_detection: Dict[int, Detection] = {}
        for det in sorted(
            current_detections.get("Orientation", []), key=_detection_sort_key
        ):
            if len(det.sigma.quantized) < 3:
                continue
            nid = concept_graph.get_or_create_pattern_node(
                det.k, _node_signature_hash(det.sigma), signature=det.sigma
            )
            node_id_to_detection.setdefault(nid, det)

        bo_by_anchor: Dict[Any, Detection] = {}
        for det in sorted(
            current_detections.get("BorderOwnership", []),
            key=_detection_sort_key,
        ):
            bo_by_anchor.setdefault(det.alpha, det)

        # 1. Build G_c by explicit incident-edge iteration. Per-node
        # get_outgoing/get_incoming avoids inventing a graph-wide iterator;
        # sets make the two passes idempotent.
        G_c: Dict[int, Set[int]] = defaultdict(set)
        for nid in node_id_to_detection:
            for edge in concept_graph.get_outgoing_edges(nid):
                if edge.edge_type not in (
                    EdgeType.ORIENTATION_CONTINUITY, EdgeType.CURVATURE
                ):
                    continue
                if edge.weight < P.THETA_CONTOUR_EDGE:
                    continue
                if edge.child not in node_id_to_detection:
                    continue
                G_c[nid].add(edge.child)
                G_c[edge.child].add(nid)
            for edge in concept_graph.get_incoming_edges(nid):
                if edge.edge_type not in (
                    EdgeType.ORIENTATION_CONTINUITY, EdgeType.CURVATURE
                ):
                    continue
                if edge.weight < P.THETA_CONTOUR_EDGE:
                    continue
                if edge.parent not in node_id_to_detection:
                    continue
                G_c[nid].add(edge.parent)
                G_c[edge.parent].add(nid)

        # 2. Figure-ground gating. The BorderOwnership direction slot is a
        # SIGNED figure-ground scalar (perception/ownership.py), NOT an angle:
        # the entity is the figure when the sign is positive, the named
        # neighbour when it is negative. Two orientation segments on the same
        # figure share an ownership sign; a sign flip means the edge crosses a
        # figure/ground transition (a T-occlusion boundary) and is removed.
        # With no BorderOwnership reading for either endpoint there is no
        # evidence either way, so the edge stays.
        from perception.ownership import ownership_signed
        edges_to_remove: Set[Tuple[int, int]] = set()
        for u in sorted(G_c.keys()):
            for v in sorted(G_c[u]):
                if u >= v:
                    continue
                bo_u = bo_by_anchor.get(node_id_to_detection[u].alpha)
                bo_v = bo_by_anchor.get(node_id_to_detection[v].alpha)
                if bo_u is None or bo_v is None:
                    continue
                if (bo_u.sigma.relation == "FIGURE_SIDE_FIELD"
                        and bo_v.sigma.relation == "FIGURE_SIDE_FIELD"
                        and (ownership_signed(bo_u) >= 0.0)
                        != (ownership_signed(bo_v) >= 0.0)):
                    edges_to_remove.add((u, v))
        for (u, v) in edges_to_remove:
            G_c[u].discard(v)
            G_c[v].discard(u)

        # 3. Connected components.
        visited: Set[int] = set()
        components: List[Set[int]] = []
        for start in sorted(G_c.keys()):
            if start in visited:
                continue
            comp: Set[int] = set()
            stack = [start]
            visited.add(start)
            while stack:
                x = stack.pop()
                comp.add(x)
                for y in sorted(G_c.get(x, ())):
                    if y not in visited:
                        visited.add(y)
                        stack.append(y)
            components.append(comp)

        emitted: List[Detection] = []

        def _emit_chain(chain: List[int], closed: bool) -> None:
            """Emit one Contour detection for `chain` ([B5] curvature sequence)."""
            if len(chain) < 2:
                return
            self._candidates_evaluated += 1
            n_theta = P.ORIENTATION_N_THETA
            bins = [
                int(round(node_id_to_detection[m].sigma.quantized[0])) % n_theta
                for m in chain
            ]
            member_anchors = [node_id_to_detection[m].alpha for m in chain]

            # Real geometry when every member anchor is live; otherwise
            # uniform spacing and no geometric turn resolution.
            pts = None
            if all(a in anchor_positions for a in member_anchors):
                pts = [
                    (float(anchor_positions[a][0]), float(anchor_positions[a][1]))
                    for a in member_anchors
                ]

            quantized = _contour_slots(bins, closed, pts)

            # alpha must be a live observation anchor: the FIRST MEMBER's
            # anchor by lowest node id (direction-invariant).
            first_member_node_id = min(chain)
            first_member_anchor = node_id_to_detection[first_member_node_id].alpha

            emitted.append(Detection(
                k="Contour", source_layer=1, scale=0,
                sigma=TypedSignature(
                    kind="CANONICAL", class_="CONTOUR",
                    relation="ARC_FRAGMENT",
                    quantized=quantized,
                    population=population_for_bins("CONTOUR", quantized),
                ),
                alpha=first_member_anchor,
                z=min(1.0, len(chain) / 8.0), t=t,
                rho=(("entity", first_member_anchor, 0, "+"),)
                    + tuple(("member", member_anchors[i], i, "+") for i in range(len(chain)))
                    + tuple(("member_node", (chain[i],), i, "+") for i in range(len(chain))),
            ))

        for comp in components:
            branch_nodes = sorted(n for n in comp if len(G_c.get(n, ())) > 2)

            if branch_nodes:
                # [Fix 2] Split at branch points into arms.
                #
                # Pre-fix rule deleted the ENTIRE component when any node
                # branched. Corpus measurement (config C, ETH-80): 648
                # components / 10,809 nodes discarded this way — a third
                # of all contour structure, and the sole cause of the
                # single true zero-CC image (car14-map: closed silhouette
                # ring, 97 nodes, lattice-staircase degree-3 artefacts).
                #
                # Arm = maximal simple path between two branch nodes or
                # between a branch node and a dead end. Every edge belongs
                # to exactly one arm; `used` tracks edge membership with
                # undirected key `(min, max)`.
                #
                # Closure detection (self-audit A.1 fix): if the walk
                # returns to its OWN starting branch node (`curr == bn`),
                # it has traced a cycle — strip the duplicate terminal
                # node and emit `closed=True`. If it terminates at a
                # DIFFERENT branch node (or a dead end), emit
                # `closed=False`. The pre-audit draft hardcoded
                # `closed=False` for all branch arms, silently losing
                # closure information on lollipop components.
                branch_set = set(branch_nodes)
                used: Set[Tuple[int, int]] = set()
                for bn in branch_nodes:
                    for nb in sorted(G_c.get(bn, ())):
                        e_key = (min(bn, nb), max(bn, nb))
                        if e_key in used:
                            continue
                        used.add(e_key)
                        arm = [bn, nb]
                        prev, curr = bn, nb
                        while curr not in branch_set:
                            nxts = [
                                x for x in G_c.get(curr, ()) if x != prev
                            ]
                            if not nxts:
                                break
                            nxt = min(nxts)
                            e2 = (min(curr, nxt), max(curr, nxt))
                            if e2 in used:
                                break
                            used.add(e2)
                            arm.append(nxt)
                            prev, curr = curr, nxt
                        # Closure: only when the walk returned to its OWN
                        # starting branch node. A walk ending at a
                        # different branch node is an inter-branch arm
                        # and is correctly open. A walk ending on a dead
                        # end or a used edge is also correctly open.
                        is_closed_arm = (curr == bn)
                        if is_closed_arm:
                            # Strip the duplicate terminal node so rho
                            # carries each node exactly once. The stripped
                            # arm has >= 3 nodes: a cycle through a
                            # degree->=3 node traverses >= 2 intermediate
                            # nodes, so `_emit_chain`'s `len < 2` guard
                            # is not reached from this path.
                            arm = arm[:-1]
                        _emit_chain(arm, closed=is_closed_arm)
                continue

            # ---- Unbranched component: pre-fix behaviour preserved ----
            if len(comp) < 2:
                continue

            degree_one = [n for n in comp if len(G_c.get(n, ())) == 1]
            if len(degree_one) == 2:
                # Open chain: walk from the lowest-id endpoint.
                start = min(degree_one)
                chain = [start]
                prev = None
                current = start
                for _ in range(len(comp) + 1):
                    neighbors = [
                        n for n in G_c.get(current, ()) if n != prev
                    ]
                    if not neighbors:
                        break
                    nxt = min(neighbors)
                    if nxt == start:
                        break
                    chain.append(nxt)
                    prev, current = current, nxt
                closed = False
            elif len(degree_one) == 0 and all(
                len(G_c.get(n, ())) == 2 for n in comp
            ):
                # Cycle: enter it at its lowest-id node, deterministically.
                component_sorted = sorted(comp)
                start = component_sorted[0]
                neighbors = sorted(G_c[start])
                current = neighbors[0]
                chain = [start, current]
                prev = start
                for _ in range(len(comp) + 1):
                    neighbors = [
                        n for n in G_c.get(current, ()) if n != prev
                    ]
                    if not neighbors:
                        break
                    nxt = min(neighbors)
                    if nxt == start:
                        break
                    chain.append(nxt)
                    prev, current = current, nxt
                closed = True
            else:
                # Structurally impossible under connectedness; retained
                # so a corrupted G_c cannot be silently mislabelled.
                continue

            _emit_chain(chain, closed=closed)

        self._emitted_total += len(emitted)
        return emitted

class JunctionLens(HigherLens):
    """B.7 + B21 — V2 junctions over Orientation segments.

    Builds an instance-level spatial graph of Orientation detections,
    gated by pattern-level ORIENTATION_BEND/JUNCTION edges. Computes
    a rotation-invariant shape descriptor (topology + canonical gaps)
    anchored by the absolute reference angle of the canonical arm,
    with arm support, arm sharpness, and pairwise angle distribution (arity 19).
    """

    def step(
        self,
        t: int,
        concept_graph: Any,
        sig_provider: Any,
        anchor_positions: Dict[Any, Tuple[float, ...]],
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]],
        current_detections: Dict[str, List[Detection]],
        extraction_scale: float = 0.0,
        envelope_mask: Optional[
            Tuple[Tuple[float, float, float, float], ...]
        ] = None,
        observation: Optional[Any] = None,
        update_homeostasis: bool = True,
        object_envelope: Optional[
            Tuple[float, float, float, float]
        ] = None,
        scale_level: int = 0,
    ) -> List[Detection]:
        if extraction_scale <= 0.0:
            self._skipped_no_position += 1
            return []

        orient_dets = sorted(
            current_detections.get("Orientation", []), key=_detection_sort_key
        )
        if not orient_dets:
            return []

        det_to_nid: Dict[int, int] = {}
        for i, det in enumerate(orient_dets):
            if len(det.sigma.quantized) < 2:
                continue
            nid = concept_graph.get_or_create_pattern_node(
                det.k, _node_signature_hash(det.sigma), signature=det.sigma
            )
            det_to_nid[i] = nid

        pattern_edges: Set[Tuple[int, int]] = set()
        for nid in set(det_to_nid.values()):
            for edge in concept_graph.get_outgoing_edges(nid):
                if edge.edge_type in (EdgeType.ORIENTATION_BEND, EdgeType.ORIENTATION_JUNCTION):
                    pattern_edges.add((nid, edge.child))
                    pattern_edges.add((edge.child, nid))
            for edge in concept_graph.get_incoming_edges(nid):
                if edge.edge_type in (EdgeType.ORIENTATION_BEND, EdgeType.ORIENTATION_JUNCTION):
                    pattern_edges.add((edge.parent, nid))
                    pattern_edges.add((nid, edge.parent))

        G_j: Dict[int, Set[int]] = defaultdict(set)
        junction_radius = getattr(P, "KAPPA_JUNCTION_RADIUS", 2.0) * extraction_scale
        cell_size = max(junction_radius, 1e-6)
        metric_epsilon = 1e-3 * max(junction_radius, 1e-6)

        grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for i, det in enumerate(orient_dets):
            if i not in det_to_nid:
                continue
            pos = anchor_positions.get(det.alpha)
            if pos and len(pos) >= 2:
                cell = (int(pos[0] // cell_size), int(pos[1] // cell_size))
                grid[cell].append(i)

        for i, det in enumerate(orient_dets):
            if i not in det_to_nid:
                continue
            pos_i = anchor_positions.get(det.alpha)
            if not pos_i or len(pos_i) < 2:
                continue
            cell = (int(pos_i[0] // cell_size), int(pos_i[1] // cell_size))
            nid_i = det_to_nid[i]

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in grid.get((cell[0] + dx, cell[1] + dy), []):
                        if i >= j:
                            continue
                        nid_j = det_to_nid[j]
                        if (nid_i, nid_j) not in pattern_edges:
                            continue
                        pos_j = anchor_positions.get(orient_dets[j].alpha)
                        if not pos_j or len(pos_j) < 2:
                            continue
                        if _pos_dist(pos_i, pos_j) <= junction_radius:
                            G_j[i].add(j)
                            G_j[j].add(i)

        CONTINUITY_NBR = _collect_continuity_neighbours(
            concept_graph, orient_dets, det_to_nid, anchor_positions,
            junction_radius, grid
        )

        visited: Set[int] = set()
        components: List[Set[int]] = []
        for start in sorted(G_j.keys()):
            if start in visited:
                continue
            comp: Set[int] = set()
            stack = [start]
            visited.add(start)
            while stack:
                x = stack.pop()
                comp.add(x)
                for y in sorted(G_j.get(x, ())):
                    if y not in visited:
                        visited.add(y)
                        stack.append(y)
            components.append(comp)

        emitted: List[Detection] = []
        for comp in components:
            if len(comp) < 3:
                continue

            hub_idx = max(comp, key=lambda n: (len(G_j.get(n, ())), -n))
            hub_neighbors = sorted(G_j[hub_idx])
            hub_pos = anchor_positions.get(orient_dets[hub_idx].alpha)
            if not hub_pos or len(hub_pos) < 2:
                continue

            pairs: List[Tuple[float, Any, int]] = []
            seen_anchors: Set[Any] = set()
            for j in hub_neighbors:
                a = orient_dets[j].alpha
                if a in seen_anchors:
                    continue
                pos_j = anchor_positions.get(a)
                if not pos_j or len(pos_j) < 2:
                    continue
                dx = pos_j[0] - hub_pos[0]
                dy = pos_j[1] - hub_pos[1]
                if math.hypot(dx, dy) <= metric_epsilon:
                    continue
                seen_anchors.add(a)
                o_q = orient_dets[j].sigma.quantized
                if len(o_q) >= 1:
                    tang = carrier_to_tangent(
                        (o_q[0] + 0.5) * math.pi / P.ORIENTATION_N_THETA)
                    arm_dir = math.atan2(dy, dx) % math.pi
                    if _aang(tang, arm_dir) > P.JUNCTION_ARM_ALIGN_TOL:
                        continue
                pairs.append((math.atan2(dy, dx) % (2 * math.pi), a, j))

            n_ang = len(pairs)
            if n_ang < 2:
                continue

            self._candidates_evaluated += 1

            pairs.sort(key=lambda p: (p[0], serialize_value(p[1])))
            angles = [p[0] for p in pairs]
            member_anchors = [p[1] for p in pairs]
            arm_idx = [p[2] for p in pairs]

            gaps: List[float] = []
            for i in range(n_ang):
                g = (angles[(i + 1) % n_ang] - angles[i]) % (2 * math.pi)
                gaps.append(g)

            gap_bins_for_sort = [
                math.floor(g * _JUNCTION_CANONICAL_BINS / (2 * math.pi))
                for g in gaps
            ]
            rotations = [
                tuple(gap_bins_for_sort[i:] + gap_bins_for_sort[:i])
                for i in range(n_ang)
            ]
            min_seq = min(rotations)
            indices = [i for i, r in enumerate(rotations) if r == min_seq]
            min_shift = indices[0]
            ref_is_unambiguous = len(indices) == 1

            angles = angles[min_shift:] + angles[:min_shift]
            gaps = gaps[min_shift:] + gaps[:min_shift]
            member_anchors = member_anchors[min_shift:] + member_anchors[:min_shift]
            arm_idx = arm_idx[min_shift:] + arm_idx[:min_shift]

            ref_theta = angles[0]

            topo = 0
            if n_ang == 2:
                gap = min(gaps[0], 2 * math.pi - gaps[0])
                if abs(gap - math.pi) < P.JUNCTION_COLLINEAR_TOL:
                    topo = 0
                else:
                    topo = 1
            elif n_ang == 3:
                tol = P.JUNCTION_COLLINEAR_TOL
                if any(abs(g - math.pi) < tol for g in gaps):
                    topo = 2
                elif max(gaps) > math.pi + tol:
                    topo = 6
                else:
                    topo = 3
            elif n_ang == 4:
                diff_0_2 = (angles[2] - angles[0]) % (2 * math.pi)
                diff_1_3 = (angles[3] - angles[1]) % (2 * math.pi)
                pair_0_opposite = abs(diff_0_2 - math.pi) < P.JUNCTION_COLLINEAR_TOL
                pair_1_opposite = abs(diff_1_3 - math.pi) < P.JUNCTION_COLLINEAR_TOL
                if pair_0_opposite and pair_1_opposite:
                    topo = 4
                else:
                    topo = 0
            else:
                expected_gap = (2 * math.pi) / n_ang
                tol = max(min(0.25 * expected_gap, P.JUNCTION_STAR_TOL), math.pi / 36.0)
                if all(abs(g - expected_gap) < tol for g in gaps):
                    topo = 5
                else:
                    topo = 0

            is_symmetric = (topo in (4, 5)) or (topo == 0 and n_ang == 2)
            if is_symmetric or not ref_is_unambiguous:
                ref_theta_bin = 0.0
                emit_ref_arm = False
            else:
                ref_theta_bin = float(q_linear(ref_theta, 24, 0.0, 2 * math.pi))
                emit_ref_arm = True

            expected_gap = (2 * math.pi) / n_ang
            spread = sum((g - expected_gap) ** 2 for g in gaps) / n_ang
            var_max = (2 * math.pi) ** 2 * (1.0 - 1.0 / n_ang) / n_ang
            spread_norm = spread / var_max if var_max > 1e-9 else 0.0
            spread_bin = float(min(7, math.floor(spread_norm * 8.0)))
            n_branches_bin = float(min(5, n_ang))

            gap_bins = [
                float(q_linear(gaps[i] if i < n_ang else 0.0, 8, 0.0, 2 * math.pi))
                for i in range(4)
            ]

            stats = [
                _arm_tangent_stats(j, hub_idx, G_j, orient_dets, CONTINUITY_NBR)
                for j in arm_idx
            ]
            support = sum(n_ for n_, _ in stats) / max(1, len(stats))
            measured = [r for n_, r in stats if n_ > 1]
            sharp_mean = (sum(measured) / len(measured)) if measured else 0.0
            sharp_min = min(measured) if measured else 0.0

            pair_hist = [0] * 8
            for ia in range(n_ang):
                for ib in range(ia + 1, n_ang):
                    d = abs(angles[ia] - angles[ib]) % (2 * math.pi)
                    d = min(d, 2 * math.pi - d)
                    pair_hist[min(7, int(d / math.pi * 8))] += 1
            pair_hist_f = [float(min(7, c)) for c in pair_hist]

            extra = (
                qi(min(support, 7.0) / 7.0, 8),
                qi(sharp_mean, 8),
                qi(sharp_min, 8),
            ) + tuple(pair_hist_f)

            quantized = (
                n_branches_bin,
                ref_theta_bin,
                float(topo),
                spread_bin,
                gap_bins[0],
                gap_bins[1],
                gap_bins[2],
                gap_bins[3],
            ) + extra

            hub_anchor = orient_dets[hub_idx].alpha
            rho = [("entity", hub_anchor, 0, "+")]
            if emit_ref_arm:
                rho.append(("ref_arm", member_anchors[0], 0, "+"))
            for i, a in enumerate(member_anchors):
                rho.append(("member", a, i, "+"))
            for i, j in enumerate(arm_idx):
                if j in det_to_nid:
                    rho.append(("member_node", (det_to_nid[j],), i, "+"))

            emitted.append(Detection(
                k="Junction",
                source_layer=1,
                scale=0,
                sigma=TypedSignature(
                    kind="CANONICAL",
                    class_="JUNCTION",
                    relation="ANGLE_RELATION",
                    quantized=quantized,
                    population=population_for_bins("JUNCTION", quantized),
                ),
                alpha=hub_anchor,
                z=min(1.0, n_ang / P.JUNCTION_SATURATION_ARMS),
                t=t,
                rho=tuple(rho),
            ))

        self._emitted_total += len(emitted)
        return emitted

class TextureBoundaryLens(HigherLens):
    """B20 — texture-defined boundary as a NODE field (arity 8).
    One detection per sampled boundary vertex whose two flanks differ in
    (theta, sf) energy distribution.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._emitted_total = 0
        self._candidates_evaluated = 0

    def step(
        self,
        t: int,
        concept_graph: Any = None,
        sig_provider: Any = None,
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]] = None,
        anchor_velocities: Optional[Dict[Any, Tuple[float, ...]]] = None,
        current_detections: Optional[Dict[str, List[Detection]]] = None,
        extraction_scale: float = 0.0,
        envelope_mask: Optional[Tuple[Tuple[float, float, float, float], ...]] = None,
        observation: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Detection]:
        # [W1] Two guard paths, because the two arms need different things.
        #
        # Legacy arm: needs the gabor bank plus a per-anchor texture-profile
        # sampler. `sample_texture_profile_at` does NOT exist on
        # Observation2D at all, so this arm is dead by construction -- not by
        # a flag, and not by the entity path. That is a measured fact about
        # the class, not a tunable.
        #
        # Field arm: measures material discontinuity ACROSS a chain from the
        # front end's own luminance/opponent planes (perception.chain_flanks),
        # so it needs only the rgb plane and the chains.
        if bool(P.FIELD_REGION_SOURCE):
            return self._step_field(
                t, anchor_positions, current_detections, extraction_scale,
                observation, luminance=kwargs.get("luminance"))
        if (
            extraction_scale <= 0.0
            or observation is None
            or not hasattr(observation, "sample_texture_profile_at")
        ):
            return []
        bank = getattr(observation, "bank", None) or getattr(observation, "_gabor_bank", None)
        if bank is None:
            return []
        thetas = np.asarray(bank.theta, float)
        logsf = np.log2(np.maximum(np.asarray(bank.sf, float), 1e-12))
        rs = P.TB_SAMPLE_FRAC * extraction_scale
        off = max(rs + 1.0, P.TB_OFFSET_FRAC * extraction_scale)
        K = P.TB_SAMPLE_COUNT
        emitted: List[Detection] = []

        # [W1] Entity source. FIELD_REGION_SOURCE=True iterates the chains
        # themselves (anchor -> ordered polyline from the rho order), so a
        # texture boundary is measured across a chain rather than across a
        # segmenter boundary that may not exist for this entity.
        if bool(P.FIELD_REGION_SOURCE):
            polys = entity_polylines(observation, current_detections,
                                     anchor_positions, use_field=True)
            items = []
            for a in sorted((str(x) for x in polys)):
                items.append((a, polys[a]))
            items.sort(key=lambda x: x[0])
        else:
            boundary_dets = (current_detections or {}).get("Boundary", [])
            items = []
            for det in sorted(boundary_dets, key=_detection_sort_key):
                a = det.alpha
                poly = observation.get_boundary_polyline(a)
                if poly:
                    items.append((a, [(float(p[0]), float(p[1])) for p in poly]))

        for a, poly in items:
            pts = list(poly)
            if len(pts) >= 2 and abs(pts[0][0] - pts[-1][0]) < 1e-9 and abs(pts[0][1] - pts[-1][1]) < 1e-9:
                pts = pts[:-1]
            n = len(pts)
            if n < 3:
                continue
            self._candidates_evaluated += 1
            for k in range(K):
                frac = (k + 0.5) / K
                idx = min(n - 1, int(frac * n))
                prv, cur, nxt = pts[(idx - 1) % n], pts[idx], pts[(idx + 1) % n]
                tx, ty = nxt[0] - prv[0], nxt[1] - prv[1]
                tn = math.hypot(tx, ty)
                if tn < 1e-9:
                    continue
                cur_y, cur_x = cur[0], cur[1]
                d_row, d_col = tx / tn, ty / tn
                n_row, n_col = -d_col, d_row
                p1 = observation.sample_texture_profile_at(
                    cur_x + n_col * off, cur_y + n_row * off, rs
                )
                p2 = observation.sample_texture_profile_at(
                    cur_x - n_col * off, cur_y - n_row * off, rs
                )
                if p1 is None or p2 is None:
                    continue
                (pa, la), (pb, lb) = p1, p2
                ea, eb = float(pa.sum()), float(pb.sum())
                if min(ea, eb) < P.TB_MIN_ENERGY:
                    continue
                if eb > ea:
                    pa, la, ea, pb, lb, eb = pb, lb, eb, pa, la, ea
                qa, qb = pa / ea, pb / eb
                tex_dist = math.sqrt(
                    0.5 * float(((np.sqrt(qa) - np.sqrt(qb)) ** 2).sum())
                )
                if tex_dist < P.TB_MIN_DIST:
                    continue

                # Angle and sf summaries
                marg_a = pa.sum(axis=1)
                tot_a = marg_a.sum() + 1e-12
                wa = marg_a / tot_a
                sa = float((wa * np.sin(2 * thetas)).sum())
                ca = float((wa * np.cos(2 * thetas)).sum())
                th_a = (0.5 * math.atan2(sa, ca)) % math.pi
                conc_a = min(1.0, math.hypot(sa, ca))

                marg_b = pb.sum(axis=1)
                tot_b = marg_b.sum() + 1e-12
                wb = marg_b / tot_b
                sb = float((wb * np.sin(2 * thetas)).sum())
                cb = float((wb * np.cos(2 * thetas)).sum())
                th_b = (0.5 * math.atan2(sb, cb)) % math.pi
                conc_b = min(1.0, math.hypot(sb, cb))

                dth = (th_b - th_a) % math.pi
                dlogsf = float((pb.sum(axis=0) * logsf).sum() / (pb.sum() + 1e-12)) - float((pa.sum(axis=0) * logsf).sum() / (pa.sum() + 1e-12))
                lum_dist = abs(la - lb)

                _tb_slots = (
                    float(q_linear(frac, 8, 0.0, 1.0)),
                    float(int(round((dth / math.pi) * float(P.ORIENTATION_N_THETA))) % P.ORIENTATION_N_THETA),
                    float(q_linear(dlogsf, 8, -1.5, 1.5)),
                    float(q_linear(math.log2(max(ea / max(eb, 1e-6), 1e-6)), 8, 0.0, 3.0)),
                    float(q_linear(tex_dist, 8, 0.0, 1.0)),
                    float(q_linear(lum_dist, 8, 0.0, 0.5)),
                    float(q_linear(conc_a, 8, 0.0, 1.0)),
                    float(q_linear(conc_b, 8, 0.0, 1.0)),
                )
                emitted.append(
                    Detection(
                        k="TextureBoundary",
                        source_layer=1,
                        scale=0,
                        sigma=TypedSignature(
                            kind="CANONICAL",
                            class_="TEXTURE_BOUNDARY",
                            relation="MATERIAL_DISCONTINUITY",
                            quantized=_tb_slots,
                            population=population_for_bins(
                                "TEXTURE_BOUNDARY", _tb_slots),
                        ),
                    alpha=a,
                    z=tex_dist * (1.0 - min(1.0, lum_dist / 0.5)),
                    t=t,
                    rho=(("entity", a, 0, "+"), ("vertex", idx, 0, "+")),
                )
                )
        self._emitted_total += len(emitted)
        return emitted

    def _step_field(
        self,
        t: int,
        anchor_positions: Optional[Dict[Any, Tuple[float, ...]]],
        current_detections: Optional[Dict[str, List[Detection]]],
        extraction_scale: float,
        observation: Any,
        luminance: Any = None,
    ) -> List[Detection]:
        """[W1] Field-arm texture boundary: material discontinuity ACROSS a
        chain, measured on the front end's own planes.

        One detection per chain vertex where the two flanks differ, with the
        registered 8-slot layout
        (arc_pos, dtheta, dlogsf, log2energy, tex_dist, lum_dist,
         conc_a, conc_b) so the two arms are comparable slot-for-slot and
        CONSTITUTIONAL_RANGES["TEXTURE_BOUNDARY"] holds.

        The legacy arm's (theta, sf) profile comparison is unavailable here
        because the observation built on the region path carries no gabor
        bank and no `sample_texture_profile_at`. Substituting luminance and
        opponent contrast for the (theta, sf) energy distribution is the
        honest available measure -- it is the material difference across the
        contour, which is what "texture-defined boundary" means -- and it is
        labelled as such rather than pretending to be the gabor profile.
        """
        if observation is None:
            return []
        from perception.chain_flanks import (chain_flank_fields,
                                             flank_signature_slots)
        from perception.lenses.derived_common import (
            field_region_source)
        src = field_region_source(observation, current_detections,
                                  anchor_positions)
        if src is None or len(src) == 0:
            return []

        rgb = luminance if luminance is not None else getattr(
            observation, "_raw_image", None)
        if rgb is None:
            return []
        rgb = np.asarray(rgb)
        if rgb.ndim != 3:
            return []
        from perception.contour_field import field_from_luminance
        from perception.chain_flanks import planes_from_rgb
        # planes_from_rgb gives the SAME adapted luminance the chains were
        # derived from; the retinex lum_comp this replaced is a different
        # plane (max difference 0.95), so a flank's contrast was being
        # measured on a plane its own chain never came from.
        channels = planes_from_rgb(rgb)
        if channels is None:
            return []
        med_wl = None
        try:
            f = field_from_luminance(
                np.asarray(channels.get("lum", [0]), dtype=np.float32))
            sfs = np.asarray(f["sfs"], dtype=np.float64)
            med_wl = float(np.median(1.0 / np.maximum(sfs, 1e-6)))
        except Exception:
            return []
        if not med_wl or not channels:
            return []

        emitted: List[Detection] = []
        for anchor in src.entity_anchors():
            det = next((d for d in
                        (current_detections or {}).get("PEAK_CHAIN", [])
                        if d.alpha == anchor), None)
            if det is None:
                continue
            fl = chain_flank_fields(det, anchor_positions, med_wl, channels,
                                    image_shape=(int(rgb.shape[0]),
                                                 int(rgb.shape[1])))
            if fl is None:
                continue
            slots = flank_signature_slots(fl)
            n_mem = fl['n_members']
            for m in fl['members']:
                lum_l, o0_l, o1_l = m['left']
                lum_r, o0_r, o1_r = m['right']
                # oriented as the legacy arm is: the brighter flank is "a",
                # so the displacement is signed by ownership
                if m['lum_diff'] >= 0.0:
                    (la, oa0, oa1), (lb, ob0, ob1) = m['left'], m['right']
                else:
                    (la, oa0, oa1), (lb, ob0, ob1) = m['right'], m['left']
                lum_dist = abs(la - lb)
                conc_a = min(1.0, math.hypot(oa0, oa1))
                conc_b = min(1.0, math.hypot(ob0, ob1))
                self._candidates_evaluated += 1
                dtheta = math.atan2(o1_l - o1_r, o0_l - o0_r) % math.pi
                dlogsf = (math.log2(max(1e-12, 1.0 + lum_l))
                          - math.log2(max(1e-12, 1.0 + lum_r)))
                # la/lb are ordered brighter-first, so the energy ratio is >=1.
                d_energy = math.log2(max(1e-6, (1.0 + la) / (1.0 + lb)))
                # Orientation-distribution distance across the boundary is the
                # field-arm analogue of the legacy (theta, sf) tex_dist.
                d_ori = math.sqrt(0.5 * ((o0_l - o0_r) ** 2
                                         + (o1_l - o1_r) ** 2))
                frac = (m['index'] + 0.5) / max(1, n_mem)
                _tb_slots = (
                    float(q_linear(frac, 8, 0.0, 1.0)),
                    float(int(round((dtheta / math.pi)
                                    * float(P.ORIENTATION_N_THETA)))
                          % P.ORIENTATION_N_THETA),
                    float(q_linear(dlogsf, 8, -1.5, 1.5)),
                    float(q_linear(max(0.0, d_energy), 8, 0.0, 3.0)),
                    float(q_linear(min(1.0, d_ori), 8, 0.0, 1.0)),
                    float(q_linear(lum_dist, 8, 0.0, 0.5)),
                    float(q_linear(conc_a, 8, 0.0, 1.0)),
                    float(q_linear(conc_b, 8, 0.0, 1.0)),
                )
                emitted.append(Detection(
                    k="TextureBoundary",
                    source_layer=1,
                    scale=0,
                    sigma=TypedSignature(
                        kind="CANONICAL", class_="TEXTURE_BOUNDARY",
                        relation="MATERIAL_DISCONTINUITY",
                        quantized=_tb_slots,
                        population=population_for_bins(
                            "TEXTURE_BOUNDARY", _tb_slots),
                    ),
                    alpha=anchor,
                    z=(float(slots.get("lum_contrast_bin", 0.0))
                       * (1.0 - min(1.0, lum_dist / 0.5))),
                    t=t,
                    rho=(("entity", anchor, 0, "+"),
                         ("vertex", m['index'], 0, "+")),
                ))
        self._emitted_total += len(emitted)
        return emitted

    def get_diagnostics(self) -> Dict[str, int]:
        return {
            "candidates_evaluated": self._candidates_evaluated,
            "emitted_total": self._emitted_total,
        }

