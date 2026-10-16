"""perception/lenses/s4.py -- extracted from derived.py
(Move 3.5).

MOVE ONLY: whole top-level statements, verbatim.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from core.hashing import (  # noqa: F401
    clamp,
)
from core.types import (  # noqa: F401
    Detection,
    TypedSignature,
)
from perception.contours import (  # noqa: F401
    ribbon_bins,
)
from perception.lenses.derived_base import (  # noqa: F401
    HigherLens,
)

class ShockEnsembleLens(HigherLens):
    """Stage 4 — Medial Shock Axis Ribbons and Ensembles.
    Reads pruned shock structures from observation.get_shock_structure()
    and emits Ribbon (arity 5) and Ensemble (arity 8) detections."""

    def __init__(self) -> None:
        super().__init__()
        self._skipped_no_shock = 0
        self._candidates_evaluated = 0

    def get_diagnostics(self) -> Dict[str, int]:
        return {
            "emitted_total": self._emitted_total,
            "skipped_no_shock": self._skipped_no_shock,
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
        envelope_mask: Optional[Tuple[Tuple[float, float, float, float], ...]] = None,
        observation: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Detection]:
        if observation is None or not hasattr(observation, "get_shock_structure"):
            return []
        shock_struct = observation.get_shock_structure()
        if not shock_struct:
            self._skipped_no_shock += 1
            return []

        segs, joints, ens, meta = shock_struct
        if not segs and not ens:
            return []

        R = meta.get("R", 1.0)
        emitted: List[Detection] = []

        contour_anchor_map: Dict[int, Any] = {}
        if hasattr(observation, "_regions"):
            for u, r in enumerate(observation._regions):
                ci = r.get("_contour_idx")
                if ci is not None:
                    contour_anchor_map.setdefault(ci, u)

        # 1. Emit Ribbon Detections
        for seg in segs:
            self._candidates_evaluated += 1
            c_indices = seg.get("contours", [])
            primary_c = c_indices[0] if c_indices else 0
            anchor = contour_anchor_map.get(primary_c, primary_c)

            q_bins = ribbon_bins(seg, R)
            rho = [
                ("entity", anchor, 0, "+"),
                ("ribbon", seg["id"], 0, "+"),
            ]
            for idx, ci in enumerate(c_indices):
                rho.append(("member_contour", contour_anchor_map.get(ci, ci), idx, "+"))

            emitted.append(
                Detection(
                    k="Ribbon",
                    source_layer=2,
                    scale=0,
                    sigma=TypedSignature(
                        kind="CANONICAL",
                        class_="RIBBON",
                        relation="SHOCK_SEGMENT",
                        quantized=q_bins,
                    ),
                    alpha=anchor,
                    z=clamp(seg["length"] / (2.0 * R), 0.05, 1.0),
                    t=t,
                    rho=tuple(rho),
                )
            )

        # 2. Emit Ensemble Detections
        for e in ens:
            self._candidates_evaluated += 1
            c_indices = e.get("contours", [])
            primary_c = c_indices[0] if c_indices else 0
            anchor = contour_anchor_map.get(primary_c, primary_c)

            rho = [
                ("entity", anchor, 0, "+"),
                ("ensemble", e["id"], 0, "+"),
            ]
            for idx, sid in enumerate(e.get("segments", [])):
                rho.append(("ribbon_member", sid, idx, "+"))
            for idx, ci in enumerate(c_indices):
                rho.append(("member_contour", contour_anchor_map.get(ci, ci), idx, "+"))

            # Spatial salience proxy: number of constituent segments
            z_salience = clamp(len(e.get("segments", [])) / 8.0, 0.1, 1.0)

            emitted.append(
                Detection(
                    k="Ensemble",
                    source_layer=2,
                    scale=0,
                    sigma=TypedSignature(
                        kind="CANONICAL",
                        class_="ENSEMBLE",
                        relation="SHOCK_ENSEMBLE",
                        quantized=e["bins"],
                    ),
                    alpha=anchor,
                    z=z_salience,
                    t=t,
                    rho=tuple(rho),
                )
            )

        self._emitted_total += len(emitted)
        return emitted

