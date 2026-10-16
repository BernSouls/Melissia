"""action/decode.py -- action-sequence decoding (Move 5d).

Moved DOWN from pipeline.py so that memory can call it without importing
orchestration: this retires the last `consolidator -> pipeline` edge,
i.e. the last STAYS in the cycle table. pipeline.py now calls it from
here.

MOVE ONLY: whole top-level statement, verbatim.
"""
from __future__ import annotations

from typing import Any, List, Optional, Set, Tuple
from core.graph_types import (  # noqa: F401
    Subgraph,
)
from core.types import (  # noqa: F401
    EdgeType,
)
from sensory_floor import ACTION_LENS_NAME

def decode_action_sequence(G: Subgraph, concept_graph: Any) -> Optional[List[Any]]:
    """Recover the exact action sequence encoded in a subgraph (B2).

    Walks G's SEQUENCE topology in strict canonical order and decodes
    every Action-lens node through the ConceptGraph's action-alpha
    registry — a dictionary lookup returning the ORIGINAL action object
    (the same hashable value the environment's AEM is keyed by; no
    string inference, no bucket collisions: the Action lens signature is
    the exact 64-bit canonical hash of the action).

    Returns:
      []    — G carries no SEQUENCE edges (empty action sequence; replay
              of an empty sequence is a well-defined no-op check).
      list  — the decoded actions in execution order.
      None  — SEQUENCE structure exists but is undecodable: a cycle, an
              Action node without a registry entry (pruned/recreated
              node), or NO Action nodes at all (a pre-lens legacy
              concept). Callers fall back to the AEM walk or fail the
              replay honestly — never to node IDs, which no environment
              action_edge_map can consume (the B2 defect).
    """
    seq_nodes: Set[int] = set()
    seq_edges: List[Tuple[int, int]] = []
    for (u, v, et) in G.edges:
        if et == EdgeType.SEQUENCE and u in G.nodes and v in G.nodes:
            seq_edges.append((u, v))
            seq_nodes.add(u)
            seq_nodes.add(v)
    if not seq_edges:
        return []
    # [defect fix] topo_sort_sequence_strict lives in orchestration.pipeline,
    # which imports this module -- so the import is deferred to call time to
    # avoid a cycle.
    from orchestration.pipeline import topo_sort_sequence_strict
    order = topo_sort_sequence_strict(seq_nodes, seq_edges)
    if order is None:
        return None
    lens_type_of = getattr(concept_graph, "get_node_lens_type", None)
    action_of = getattr(concept_graph, "get_node_action", None)
    if lens_type_of is None or action_of is None:
        return None  # registry not wired — honest failure, not node IDs
    actions: List[Any] = []
    saw_action_node = False
    for nid in order:
        if lens_type_of(nid) != ACTION_LENS_NAME:
            continue
        saw_action_node = True
        value = action_of(nid)
        if value is None:
            return None  # undecodable Action node — fail honestly
        actions.append(value)
    if not saw_action_node:
        return None  # legacy concept without Action-lens members
    return actions
