"""Item A safeguard: every directed edge type returns its edge from canonical key."""
from core.types import EdgeType, UNDIRECTED_EDGE_TYPES
from core.graph_types import _canonical_edge_key


def test_directed_edge_canonical_key():
    directed = [et for et in EdgeType if et not in UNDIRECTED_EDGE_TYPES]
    assert len(directed) > 0
    for et in directed:
        assert _canonical_edge_key(1, 2, et) == (1, 2, et), et
        assert _canonical_edge_key(2, 1, et) == (2, 1, et), et


def test_undirected_edge_canonical_key():
    for et in UNDIRECTED_EDGE_TYPES:
        assert _canonical_edge_key(2, 1, et) == (1, 2, et), et
