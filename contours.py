"""perception/contours.py -- THIN SHIM (ITEM 4 deletion prep).

All 19 live symbols were moved VERBATIM to their proper modules on
2026-10-16; nothing was deleted here or elsewhere. This shim keeps
every existing import path working:
    perception.polar             _angle_diff, _axial_angle_diff
    perception.rich           texture/energy measures (5)
    perception.contour_structure  tracing/structure/lattice (12)
Deletion of this shim waits for a real evaluation with second
instances (session rule).
"""
from __future__ import annotations

from perception.polar import (  # noqa: F401
    _angle_diff,
    _axial_angle_diff,
)
from perception.rich import (  # noqa: F401
    _autocorr_peak,
    _compute_coarseness,
    _compute_regularity,
    compute_texture_rich,
    texture_richness,
)
from perception.contour_structure import (  # noqa: F401
    V1RetinotopicLattice,
    _detect_t_junctions,
    _extract_association_field_contours,
    _front_end_scale,
    _local_edge_at,
    _moore_neighbor_trace,
    _polyline_arc_length,
    _resample_contours,
    _segment_profile,
    extract_shock_structure,
    ribbon_bins,
    _tangent_at,
)

__all__ = [
    '_local_edge_at',
    '_compute_coarseness',
    '_compute_regularity',
    '_autocorr_peak',
    'compute_texture_rich',
    'texture_richness',
    '_polyline_arc_length',
    '_tangent_at',
    '_angle_diff',
    '_moore_neighbor_trace',
    '_detect_t_junctions',
    '_axial_angle_diff',
    '_front_end_scale',
    '_extract_association_field_contours',
    'V1RetinotopicLattice',
    '_resample_contours',
    '_segment_profile',
    'extract_shock_structure',
    'ribbon_bins',
]
