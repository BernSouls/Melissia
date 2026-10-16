"""
content.py — Position-invariant content tier for Blueprint v32.

The pattern graph's identity model (§1.11) is (lens_type, signature_hash),
where the signature includes an anchor-class slot for lenses that need to
distinguish anchors for spatial binders. This makes cross-image alignment
impossible for those lenses: the same content at different anchors has
different identities and cannot be matched by neighborhood context.

The content tier defines a second, position-invariant identity for each
lens type that has anchor-free content. Content nodes are keyed by
(lens_type, content_signature_hash) where the content signature strips
the anchor-class slot. Binders continue to operate on instance nodes;
structural_intersection operates on content nodes.

Determinism: content node IDs are derived from content-signature hashes,
masked to 63 bits, with the high bit forced on. Content IDs live in
[2^63, 2^64); instance IDs are small monotonic integers. Disjoint by
construction. Same input → byte-identical content subgraphs.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, FrozenSet, Optional, Set, Tuple

from core.hashing import (  # noqa: F401
    hash_64_signed,
    serialize_bytes,
    serialize_tuple,
)
from core.registries import (  # noqa: F401
    CIRCULAR_CYCLE_LENGTHS,
    CONSTITUTIONAL_RANGES,
)
from core.schema import EdgeParams  # noqa: F401
from sensory_floor import TypedSignature
from core.quantize import _q_linear_safe  # noqa: F401
from core.types import (
    EdgeType,
    UNDIRECTED_EDGE_TYPES,
)
from core.graph_types import Subgraph, anchor_positions_for
import parameters as P

SignatureProvider = Callable[[int], Optional[Tuple[str, TypedSignature]]]

# [K.1 fix] Content tier is the position-invariant projection of the
# detection set — nodes whose signatures are anchor-free and can
# therefore be compared across images. All higher-tier lens signatures
# (Contour, Junction, CurveComplex) are already
# anchor-free — their quantized tuple carries no anchor-class slot, same
# as Color/Texture/Size. Their exclusion was an omission: the list was
# written before the higher-tier lenses existed. Including them lets the
# discriminative core be built over the full hierarchy, which is the
# substrate Phase K's absorption mechanism requires.
#
# [PHASE F] "Size" removed: the Size lens no longer emits (uniform-grid
# tautology), so it has no content-tier presence.
# [K.0 fix] "Elongation" is vestigial: its only producer was the legacy
# _multicat_real_probe.py Orientation split, which the harness no longer
# imports. Retained rather than removed — nothing emits it, so the entry
# is inert either way, and binder.INFORMATIVE_LENS_TYPES still lists it
# (that file is out of scope for this commit).
CONTENT_LENS_TYPES: FrozenSet[str] = frozenset({
    "Boundary", "Color", "Texture", "Orientation", "Elongation",
    "Contour", "Junction", "CurveComplex", "CONTOUR_PEAK", "PEAK_CHAIN",
    "Motion", "EndStopped", "Surface", "RegionShape", "IllusoryContour",
    "TextureBoundary", "HolisticShape", "Ribbon", "Ensemble",
})

CONTENT_ID_HIGH_BIT: int = 0x8000000000000000


def content_signature(
    lens_type: str, sigma: TypedSignature
) -> Optional[TypedSignature]:
    """Return the position-invariant content signature for `sigma`,
    or None if the detection is not eligible for content tier.

    Anchor-stripping rules:
      - BOUNDARY and BOUNDARY_ANCHORED both keep quantized[:2].
      - ORIENTATION keeps quantized[:2] (theta, e_b).
      - ELONGATION keeps quantized[:1] (e_b).
      - Color, Size, Texture pass through unchanged.
      - ORIENTATION_SEGMENT returns None (segmenter-only).
    """
    if sigma.kind != "CANONICAL":
        return None
    if lens_type not in CONTENT_LENS_TYPES:
        return None

    if lens_type == "Boundary":
        if sigma.class_ not in ("BOUNDARY", "BOUNDARY_ANCHORED"):
            return None
        if len(sigma.quantized) < 2:
            return None
        return TypedSignature(
            kind="CANONICAL",
            class_="BOUNDARY_CONTENT",
            relation=sigma.relation,
            quantized=sigma.quantized[:2],
        )
    if lens_type == "Orientation":
# [PASS 27] B1 ORIENTATION_RICH projection (V1 Hypercolumn Invariant Marginals)
        if (
            sigma.class_.startswith("ORIENTATION_RICH")
            and not sigma.class_.endswith("_CONTENT")
        ):
            n_marginals = getattr(P, "ORIENTATION_RICH_MARGINAL_BINS", 60)
            if len(sigma.quantized) < n_marginals:
                return None
            return TypedSignature(
                kind="CANONICAL",
                class_="ORIENTATION_RICH_CONTENT",
                relation=sigma.relation,
                quantized=tuple(float(v) for v in sigma.quantized[:n_marginals]),
            )
        if sigma.class_ == "ORIENTATION":
            if len(sigma.quantized) < 2:
                return None
            return TypedSignature(
                kind="CANONICAL",
                class_="ORIENTATION_CONTENT",
                relation=sigma.relation,
                quantized=sigma.quantized[:2],
            )
        if sigma.class_ == "ORIENTATION_GABOR":
            if len(sigma.quantized) < 2:
                return None
            return TypedSignature(
                kind="CANONICAL",
                class_="ORIENTATION_GABOR_CONTENT",
                relation=sigma.relation,
                quantized=sigma.quantized[:2],
            )
        return None
    if lens_type == "EndStopped":
        if sigma.class_ != "ENDSTOPPED":
            return None
        if len(sigma.quantized) < 2:
            return None
        return TypedSignature(
            kind="CANONICAL",
            class_="ENDSTOPPED_CONTENT",
            relation=sigma.relation,
            quantized=sigma.quantized[:2],
        )
    if lens_type == "Motion":
        if sigma.class_ == "MOTION_POPULATION":
            if len(sigma.quantized) < 13:
                return None
            q = sigma.quantized
            return TypedSignature(
                kind="CANONICAL",
                class_="MOTION_CONTENT",
                relation=sigma.relation,
                quantized=(q[11], q[12], q[10]),
            )
        if sigma.class_ == "MOTION":
            if len(sigma.quantized) < 2:
                return None
            return TypedSignature(
                kind="CANONICAL",
                class_="MOTION_CONTENT",
                relation=sigma.relation,
                quantized=sigma.quantized[:2],
            )
        return None
    if lens_type == "Elongation":
        if len(sigma.quantized) < 1:
            return None
        return TypedSignature(
            kind="CANONICAL",
            class_="ELONGATION_CONTENT",
            relation=sigma.relation,
            quantized=sigma.quantized[:1],
        )
    if lens_type == "Junction":
        if sigma.class_ != "JUNCTION":
            return None
        if len(sigma.quantized) < 19:
            return None
        q = sigma.quantized
        return TypedSignature(
            kind="CANONICAL",
            class_="JUNCTION_CONTENT",
            relation=sigma.relation,
            quantized=(q[0], q[2], q[3], q[9]) + tuple(q[11:19]),
        )
    if lens_type == "Color":
        if sigma.class_ == "COLOR":
            if len(sigma.quantized) < 5:
                return None
            hue_bin = sigma.quantized[4]
            sat_bin = sigma.quantized[3]
            return TypedSignature(
                kind="CANONICAL",
                class_="COLOR_CONTENT",
                relation=sigma.relation,
                quantized=(float(hue_bin), float(sat_bin)),
            )
        elif sigma.class_ in ("COLOR_LEGACY", "COLOR"):
            if len(sigma.quantized) < 3:
                return None
            r, g, b = sigma.quantized[:3]
            max_c = max(r, g, b)
            min_c = min(r, g, b)
            chroma = max_c - min_c
            if max_c > 0 and chroma > 0:
                if max_c == r:
                    hue = (g - b) / chroma
                elif max_c == g:
                    hue = 2.0 + (b - r) / chroma
                else:
                    hue = 4.0 + (r - g) / chroma
                hue_rad = (hue * math.pi / 3.0) % (2 * math.pi)
                sat = chroma / max_c
            else:
                hue_rad = 0.0
                sat = 0.0
            hue_bin = _q_linear_safe(hue_rad, P.COLOR_HUE_BINS, 0.0, 2 * math.pi)
            sat_bin = _q_linear_safe(sat, P.COLOR_SAT_BINS, 0.0, 1.0)
            return TypedSignature(
                kind="CANONICAL",
                class_="COLOR_LEGACY_CONTENT",
                relation=sigma.relation,
                quantized=(float(hue_bin), float(sat_bin)),
            )
        return None
    if lens_type == "Surface":
        if sigma.class_ != "SURFACE_PATCH":
            return None
        if len(sigma.quantized) < 7:
            return None
        return TypedSignature(
            kind="CANONICAL",
            class_="SURFACE_PATCH_CONTENT",
            relation=sigma.relation,
            quantized=sigma.quantized[:6],
        )
    if lens_type == "CurveComplex":
        if sigma.class_ != "CURVE_COMPLEX":
            return None
        if len(sigma.quantized) < 24:
            return None
        return TypedSignature(
            kind="CANONICAL",
            class_="CURVE_COMPLEX_CONTENT",
            relation=sigma.relation,
            quantized=sigma.quantized[2:],
        )
    if lens_type == "RegionShape":
        if sigma.class_ != "REGION_SHAPE":
            return None
        if len(sigma.quantized) < 21:
            return None
        q = sigma.quantized
        return TypedSignature(
            kind="CANONICAL",
            class_="REGION_SHAPE_CONTENT",
            relation=sigma.relation,
            quantized=q[:16] + (q[17], q[18], q[20]),
        )
    if lens_type == "IllusoryContour":
        if sigma.class_ != "ILLUSORY_CONTOUR":
            return None
        if len(sigma.quantized) < 3:
            return None
        q = sigma.quantized
        return TypedSignature(
            kind="CANONICAL",
            class_="ILLUSORY_CONTOUR_CONTENT",
            relation=sigma.relation,
            quantized=(q[0],),
        )
    if lens_type == "HolisticShape":
        if sigma.class_ != "AIT_HOLISTIC_SHAPE":
            return None
        if len(sigma.quantized) < 6:
            return None
        return TypedSignature(
            kind="CANONICAL",
            class_="AIT_HOLISTIC_SHAPE_CONTENT",
            relation=sigma.relation,
            quantized=sigma.quantized,
        )
    if lens_type == "Contour":
        # [B5] Already anchor-free. Keep the whole curvature sequence: it is
        # the information B5 adds, and the content-tier cluster threshold
        # (THETA_CLUSTER_CONTENT) already supplies tolerance between near-equal
        # shapes. Arity guard rejects stale pre-B5 (arity-7) signatures.
        if sigma.class_ != "CONTOUR":
            return None
        if len(sigma.quantized) != 7 + getattr(P, "CONTOUR_N_SEQ", 20):
            return None
        if getattr(P, "CONTOUR_CONTENT_MODE", "full") == "summary":
            q = sigma.quantized
            return TypedSignature(
                kind="CANONICAL",
                class_="CONTOUR_CONTENT",
                relation=sigma.relation,
                quantized=(q[0], q[3], q[4], q[5], q[6]),
            )
        return sigma
    if lens_type == "TextureBoundary":
        if sigma.class_ != "TEXTURE_BOUNDARY" or len(sigma.quantized) < 8:
            return None
        return TypedSignature(
            kind="CANONICAL",
            class_="TEXTURE_BOUNDARY_CONTENT",
            relation=sigma.relation,
            quantized=sigma.quantized[1:],
        )
# [PASS 27] Inserted immediately before '# Size, Texture pass through.'
    if lens_type == "Texture":
        if (
            sigma.class_.startswith("TEXTURE_RICH")
            and not sigma.class_.endswith("_CONTENT")
        ):
            if len(sigma.quantized) < 4:
                return None
            return TypedSignature(
                kind="CANONICAL",
                class_="TEXTURE_RICH_CONTENT",
                relation=sigma.relation,
                quantized=tuple(float(v) for v in sigma.quantized[-4:]),
            )
        # Basic TEXTURE falls through to return sigma below

    # Size, Texture pass through.
    return sigma


def content_node_id(lens_type: str, content_sig: TypedSignature) -> int:
    """Stable content node ID. Always in [2^63, 2^64)."""
    key = serialize_tuple([
        serialize_bytes(lens_type.encode("utf-8")),
        serialize_bytes(content_sig.serialize()),
    ])
    return (hash_64_signed(key) & 0x7FFFFFFFFFFFFFFF) | CONTENT_ID_HIGH_BIT


def is_content_id(nid: int) -> bool:
    return nid >= (1 << 63)


def _circular_mean(a: float, b: float, n: int, cycle: float) -> float:
    diff = (b - a) % cycle
    if diff > cycle / 2:
        diff -= cycle
    return (a + diff / (n + 1)) % cycle


class ContentSignatureStore:
    """Maps content node IDs to (lens_type, TypedSignature) and resolves
    incoming signatures to the nearest existing cluster within a lens type.

    [v32.3] Two signatures with similarity ≥ THETA_CLUSTER_CONTENT merge
    into one cluster. Cluster identity is created deterministically from
    the seed signature (content_node_id) and never changes; the centroid
    drifts toward the running mean of members.

    Raises on hash-fault collision (astronomically unlikely at 63 bits).
    """

    __slots__ = ("_sigs", "_clusters_by_lens", "_cluster_sizes")

    def __init__(self) -> None:
        self._sigs: Dict[int, Tuple[str, TypedSignature]] = {}
        self._clusters_by_lens: Dict[str, List[int]] = {}
        self._cluster_sizes: Dict[int, int] = {}

    def register(
        self, cnid: int, lens_type: str, content_sig: TypedSignature
    ) -> None:
        existing = self._sigs.get(cnid)
        if existing is None:
            self._sigs[cnid] = (lens_type, content_sig)
            return
        if existing[0] != lens_type or existing[1] != content_sig:
            raise RuntimeError(
                f"[CONTENT HASH FAULT] content node ID {cnid:#x} collision"
            )

    def get(self, cnid: int) -> Optional[Tuple[str, TypedSignature]]:
        return self._sigs.get(cnid)

    def has(self, cnid: int) -> bool:
        return cnid in self._sigs

    def size(self) -> int:
        return len(self._sigs)

    def resolve(self, lens_type: str, content_sig: TypedSignature) -> int:
        """Return cluster ID for content_sig, joining an existing cluster
        if similarity >= THETA_CLUSTER_CONTENT, else creating a new cluster."""
        theta = float(P.THETA_CLUSTER_CONTENT)
        best_id: Optional[int] = None
        best_sim = theta
        sig_len = len(content_sig.quantized)

        for cid in self._clusters_by_lens.get(lens_type, ()):
            centroid = self._sigs[cid][1]
            # Fast bypass: mismatched arities cannot merge
            if len(centroid.quantized) != sig_len:
                continue
            s = content_sig.similarity(centroid)
            if s > best_sim:
                best_sim = s
                best_id = cid

        if best_id is not None:
            old_sig = self._sigs[best_id][1]
            n = self._cluster_sizes[best_id]
            cycle_map = CIRCULAR_CYCLE_LENGTHS.get(old_sig.class_, {})
            new_quantized = tuple(
                _circular_mean(a, b, n, cycle_map[i])
                if i in cycle_map
                else (a * n + b) / (n + 1)
                for i, (a, b) in enumerate(
                    zip(old_sig.quantized, content_sig.quantized)
                )
            )
            new_centroid = TypedSignature(
                kind="CANONICAL",
                class_=old_sig.class_,
                relation=old_sig.relation,
                quantized=new_quantized,
            )
            self._sigs[best_id] = (lens_type, new_centroid)
            self._cluster_sizes[best_id] = n + 1
            return best_id

        new_id = content_node_id(lens_type, content_sig)
        self._sigs[new_id] = (lens_type, content_sig)
        self._clusters_by_lens.setdefault(lens_type, []).append(new_id)
        self._cluster_sizes[new_id] = 1
        return new_id


def build_content_subgraph(
    instance_subgraph: Subgraph,
    sig_provider: SignatureProvider,
    content_store: ContentSignatureStore,
    diagnostics: Optional[Dict[str, int]] = None,
) -> Subgraph:
    if diagnostics is None:
        diagnostics = {}
    instance_to_content: Dict[int, int] = {}
    content_nodes: Set[int] = set()
    dropped_nodes = 0

    for nid in sorted(instance_subgraph.nodes):
        info = sig_provider(nid)
        if info is None:
            dropped_nodes += 1
            continue
        lens_type, sigma = info
        if lens_type not in CONTENT_LENS_TYPES:
            dropped_nodes += 1
            continue
        csig = content_signature(lens_type, sigma)
        if csig is None:
            dropped_nodes += 1
            continue
        cnid = content_store.resolve(lens_type, csig)
        instance_to_content[nid] = cnid
        content_nodes.add(cnid)

    content_edge_weights: Dict[Tuple[int, int, EdgeType], float] = {}
    dropped_edges = 0
    instance_weights = getattr(instance_subgraph, "weights", None)
    for (src, dst, et) in sorted(
        instance_subgraph.edges, key=lambda e: (e[0], e[1], e[2].value)
    ):
        cs = instance_to_content.get(src)
        cd = instance_to_content.get(dst)
        if cs is None or cd is None:
            dropped_edges += 1
            continue
        if et in UNDIRECTED_EDGE_TYPES:
            a, b = (cs, cd) if cs < cd else (cd, cs)
            key = (a, b, et)
        else:
            key = (cs, cd, et)
        src_weight = 1.0
        if instance_weights:
            src_weight = float(instance_weights.get((src, dst, et), 1.0))
        content_edge_weights[key] = (
            content_edge_weights.get(key, 0.0) + src_weight
        )

    content_self_adjacency_mass: Dict[int, float] = {}
    for (a, b, et), w in content_edge_weights.items():
        if a == b:
            content_self_adjacency_mass[a] = (
                content_self_adjacency_mass.get(a, 0.0) + w
            )
    diagnostics["content_self_adjacency_nodes"] = (
        diagnostics.get("content_self_adjacency_nodes", 0)
        + len(content_self_adjacency_mass)
    )
    diagnostics["content_self_adjacency_mass_total"] = (
        diagnostics.get("content_self_adjacency_mass_total", 0.0)
        + sum(content_self_adjacency_mass.values())
    )

    diagnostics["content_nodes_dropped_off_tier"] = (
        diagnostics.get("content_nodes_dropped_off_tier", 0) + dropped_nodes
    )
    diagnostics["content_edges_dropped_off_tier"] = (
        diagnostics.get("content_edges_dropped_off_tier", 0) + dropped_edges
    )
    # [G.1.5] Per-call node/edge budget, so the source-vs-built dropout can
    # be read directly. Diagnostic only; no behaviour change.
    diagnostics["content_build_calls"] = (
        diagnostics.get("content_build_calls", 0) + 1
    )
    diagnostics["content_nodes_built"] = (
        diagnostics.get("content_nodes_built", 0) + len(content_nodes)
    )
    diagnostics["content_edges_built"] = (
        diagnostics.get("content_edges_built", 0) + len(content_edge_weights)
    )
    diagnostics["content_nodes_in_source"] = (
        diagnostics.get("content_nodes_in_source", 0)
        + len(instance_subgraph.nodes)
    )
    diagnostics["content_edges_in_source"] = (
        diagnostics.get("content_edges_in_source", 0)
        + len(instance_subgraph.edges)
    )

    # [Pass 17.1] Pool edge_params from instance edges to content edges, so
    # _mcs_accumulate / graded edge matching actually sees relational
    # parameters on content-tier subgraphs (previously defaulted to {}).
    content_edge_params: Dict[Tuple[int, int, EdgeType], EdgeParams] = {}
    instance_edge_params = getattr(instance_subgraph, "edge_params", None) or {}
    if instance_edge_params:
        for (src, dst, et) in instance_subgraph.edges:
            cs = instance_to_content.get(src)
            cd = instance_to_content.get(dst)
            if cs is None or cd is None:
                continue
            if et in UNDIRECTED_EDGE_TYPES:
                key = (min(cs, cd), max(cs, cd), et)
                inst_key = (min(src, dst), max(src, dst), et)
            else:
                key = (cs, cd, et)
                inst_key = (src, dst, et)
            ep = instance_edge_params.get((src, dst, et)) or instance_edge_params.get(inst_key)
            if ep is not None and key not in content_edge_params:
                content_edge_params[key] = ep
    diagnostics["content_edge_params_pooled"] = (
        diagnostics.get("content_edge_params_pooled", 0) + len(content_edge_params)
    )

    # [V9 Track 2 - B.26] Content node ids are in [2^63, 2^64), so the
    # instance subgraph's occupancy map cannot be inherited: the id
    # spaces are disjoint, and a lookup would silently resolve nothing.
    # Resolve through instance_to_content instead -- each content node
    # takes the anchor of the FIRST instance node that produced it, in
    # ascending instance-id order, the same first-wins rule the
    # registry itself uses.
    anchor_of_node: Dict[int, Tuple[float, float]] = {}
    for nid in sorted(instance_to_content):
        cnid = instance_to_content[nid]
        if cnid in anchor_of_node:
            continue
        p = instance_subgraph.anchor_of_node.get(nid)
        if p is not None:
            anchor_of_node[cnid] = p

    return Subgraph(
        nodes=frozenset(content_nodes),
        edges=frozenset(content_edge_weights.keys()),
        weights=content_edge_weights,
        edge_params=content_edge_params,
        anchor_of_node=anchor_of_node,
    )

