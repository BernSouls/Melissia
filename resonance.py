"""memory/resonance.py -- ART resonance types and helpers (Move 5b).

MOVE ONLY from consolidator.py. Whole top-level statements, verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Tuple
from core.graph_types import (  # noqa: F401
    Subgraph,
)

class CoreKind(Enum):
    """§5.6: discriminated union discriminator."""

    ACTION = auto()
    CATEGORY = auto()

class CoreStatus(Enum):
    """[Phase 1 Decision 2] Lifecycle status for CATEGORY cores.

    CRYSTALLIZED — freshly extracted; queried from birth.
    STABLE — graduated via Q5 (5 positives + 3 rejections + 200 steps).
    REJECTED — split/degraded/retired; must be purged before re-creation.
    No TOMBSTONED: cores outlive source concepts by design.
    """
    CRYSTALLIZED = auto()
    STABLE = auto()
    REJECTED = auto()

@dataclass
class ReductionTemplate:
    """§5.4: compression prior keyed by structural profile."""

    structural_profile: Tuple
    observed_ratio: float = 0.0
    observed_extras_count: float = 0.0
    sample_count: int = 0

@dataclass
class CorePayload_Action:
    """§5.6: payload for ACTION cores."""

    consequence_subgraph: Subgraph
    path_hash: bytes

@dataclass
class CorePayload_Category:
    """§5.6: payload for CATEGORY cores."""

    label: Any
    supporting_examples: List[bytes] = field(default_factory=list)
    discriminating_examples: List[bytes] = field(default_factory=list)
    merge_count: int = 0

def _merge_distribution(a: Dict, b: Dict, na: int, nb: int) -> Dict:
    """Support-weighted blend of nested {key: {subkey: weight}} distributions."""
    w = float(na) / float(na + nb) if (na + nb) > 0 else 0.5
    out = {}
    for k in sorted(set(a) | set(b), key=repr):
        da, db = a.get(k, {}), b.get(k, {})
        out[k] = {
            kk: w * da.get(kk, 0.0) + (1.0 - w) * db.get(kk, 0.0)
            for kk in sorted(set(da) | set(db), key=repr)
        }
    return out

def vigilance_candidates(within: Sequence[float], between: Sequence[float]) -> Dict[str, float]:
    """Within-/between-class symmetric resonance scores -> {'eer', 'p99_between'}."""
    best = (9.0, 0.5)
    for t in sorted(set(within) | set(between)):
        frr = sum(w < t for w in within) / max(1, len(within))
        far = sum(b >= t for b in between) / max(1, len(between))
        if abs(frr - far) < best[0]:
            best = (abs(frr - far), t)
    sb = sorted(between)
    p99 = sb[min(len(sb) - 1, int(0.99 * (len(sb) - 1)))] if sb else 1.0
    return {"eer": best[1], "p99_between": p99}

