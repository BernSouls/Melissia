"""core/graph_types.py -- the subgraph type shared by matching,
memory and orchestration (Subgraph, edge_params_for,
anchor_positions_for).

MOVE ONLY from prediction.py. Because `core/` sits beneath every
layer, matching and memory no longer need `prediction` to name a
graph, which deletes several cycle edges at once.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, FrozenSet
from core.schema import EdgeParams  # noqa: F401
from core.types import EdgeType  # noqa: F401

@dataclass
class Subgraph:
    """Minimal structural representation for prediction scoping and
    containment. Kept in this module so prediction does not depend on
    mapper/consolidator structures before they exist."""

    nodes: FrozenSet[int] = field(default_factory=frozenset)
    edges: FrozenSet[Tuple[int, int, EdgeType]] = field(default_factory=frozenset)

    # [C1] Optional per-edge structural arrangement parameters.
    # Excluded from __eq__ so Subgraph equality still compares only
    # structural content.
    edge_params: Dict[Tuple[int, int, EdgeType], EdgeParams] = field(
        default_factory=dict, compare=False
    )

    # [v32 CONTENT TIER] Optional edge weights for content-tier pooling:
    # records how many instance edges collapsed into each content edge.
    # Excluded from __eq__ so Subgraph equality still compares only
    # structural content. NOTE: Subgraph is a non-frozen @dataclass, so
    # __hash__ is already None and Subgraph is unhashable; this field
    # changes nothing about that. Do not make Subgraph frozen to "fix"
    # the field — that would change behavior for no consumer's benefit.
    weights: Dict[Tuple[int, int, EdgeType], float] = field(
        default_factory=dict, compare=False
    )
    # [v32.2 GRADED] Per-node weight / mass. Semantic shift mid-pipeline:
    #   - At materialization: intersection consistency prior ∈ [0, 1].
    #   - After _content_discriminative_filter: final mass.
    # Fallback contract for readers:
    #   - Empty dict → treat every element as mass 1.0.
    #   - Non-empty dict, missing key → treat the element as mass 0.0.
    node_weights: Dict[int, float] = field(
        default_factory=dict, compare=False
    )

    # [V9 Track 2 — B.26 / B.11] Occupancy metadata: which 2D anchor each
    # pattern node was built from, and the scene's T-junctions.
    #
    # compare=False is REQUIRED on both. Without it, Subgraph equality
    # starts comparing occupancy, which silently invalidates every cached
    # structural comparison downstream (mcs_score's memo, the
    # consolidation buffer, map_all's _match_cache). The existing
    # `weights` / `node_weights` fields set the precedent.
    #
    # Empty means "unknown", not "no anchors": consumers must treat an
    # absent entry as missing data (the same fallback contract the
    # node_weights docstring states) rather than as position (0, 0).
    # Populated by B.26's registry; until then both stay at the default.
    anchor_of_node: Dict[int, Tuple[float, float]] = field(
        default_factory=dict, compare=False
    )
    t_junctions: FrozenSet = field(default=frozenset(), compare=False)

    def is_empty(self) -> bool:
        return not self.nodes and not self.edges

def edge_params_for(cg, edges) -> Dict[Tuple[int, int, Any], Any]:
    """[Pass 24] Look up EdgeParams for `edges` in a ConceptGraph edge store.

    Every Subgraph construction site must call this. `Subgraph.edge_params`
    defaults to an empty dict, and `_mcs_accumulate` falls back to an
    ungraded `param_sim` whenever BOTH query and core params are missing --
    so an unpopulated Subgraph silently disables graded edge matching for
    the whole classification path rather than raising.

    DETERMINISM: `edges` is a frozenset whose members embed EdgeType enum
    members, whose default hash is identity-based and therefore varies with
    memory addresses between runs. Keys are therefore visited in a canonical
    sorted order (by int endpoints then et.value), so that an undirected edge
    present under both orientations always resolves the same way.
    """
    store = getattr(cg, "edges", None)
    if not store:
        return {}
    ordered = sorted(
        edges,
        key=lambda e: (min(e[0], e[1]), max(e[0], e[1]),
                       getattr(e[2], "value", 0)),
    )
    out: Dict[Tuple[int, int, Any], Any] = {}
    for src, dst, et in ordered:
        key = _canonical_edge_key(src, dst, et)
        edge_obj = store.get(key) or store.get((src, dst, et))
        if edge_obj is None:
            continue
        params = getattr(edge_obj, "params", None)
        if params is not None:
            out.setdefault(key, params)
    return out

def anchor_positions_for(
    nodes,
    *sources,
) -> Dict[int, Tuple[float, float]]:
    """[V9 Track 2 - B.26] node -> (x, y), from the first source that knows
    the node.

    Sources are either mappings (a parent Subgraph's `anchor_of_node`) or
    single-node lookups (`ConceptGraph.pattern_node_position`). The lookup
    form exists so the consolidation hot paths do not materialize a full
    copy of the registry per candidate subgraph: `_minimize_core` rebuilds
    a Subgraph for every node and edge it tries to remove.

    Only nodes with known geometry appear in the result. A Subgraph over
    nodes minted outside the Binder carries occupancy for the resolvable
    subset and nothing else -- never a placeholder, and never (0, 0), which
    would be indistinguishable from a real anchor at the origin.
    """
    out: Dict[int, Tuple[float, float]] = {}
    for nid in nodes:
        for src in sources:
            if not src:
                continue
            getter = getattr(src, "get", None)
            p = getter(nid) if getter is not None else src(nid)
            if p is not None:
                out[nid] = p
                break
    return out



def _canonical_edge_key(src: int, dst: int, et) -> Tuple[int, int, Any]:
    """[Pass 24] Canonical (src, dst, type) key for a concept-graph edge.

    Undirected channels are stored under (min, max) by ConceptGraph.upsert_edge,
    so a reader must canonicalize the same way or the lookup silently misses.
    Directed channels keep (src, dst) order.

    [Q3 bug fix] This function used to fall off the end for every DIRECTED
    edge type, returning None. `edge_params_for` then did
    ``out.setdefault(None, params)`` for each directed edge, so the whole
    directed population's EdgeParams collapsed onto one None key (first
    writer wins) and every later reader's ``edge_params.get((u, v, et))``
    missed. The symptom was found on the Q3 two-panel composites: a
    sign=-1 EDGE_BORDER_OWNERSHIP mixture under the None key while the 11
    positive border-ownership edges of the same subgraph read back as
    ``params=None``. Every other canonicalizer in the tree (memory/cores.py,
    memory/content.py, consolidator.py, mapper.py) returns the edge unchanged
    for the directed case; this one now does the same.
    """
    from core.types import UNDIRECTED_EDGE_TYPES

    if et in UNDIRECTED_EDGE_TYPES:
        return (min(src, dst), max(src, dst), et)
    return (src, dst, et)
