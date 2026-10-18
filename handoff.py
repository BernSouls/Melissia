"""core/handoff.py -- pure frozen dataclasses passed between memory and
orchestration (Move 4: was handoff_types.py).

MOVE ONLY. Because the records live in `core/`, `handoff_types` no longer
needs `prediction` to name them, deleting that cycle edge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, FrozenSet, Optional, Tuple

import parameters as P

@dataclass(frozen=True)
class ConsolidationHandoff:
    """A resolved (P, S) crystallization pair, packaged for prune_path.

    s_at_path        state hash when P crystallized — the replay start
                     (the situation in which the path actually occurred).
    s_at_successor   state hash when S crystallized — recorded for
                     diagnostics and as a secondary replay candidate.
    """

    P_subgraph: Subgraph
    S_subgraph: Subgraph
    s_at_path: Optional[int]
    s_at_successor: Optional[int]
    t: int
    source_hashes: Tuple[bytes, bytes] = field(default_factory=tuple)

@dataclass(frozen=True)
class MacroExecutionOutcome:
    """Execution facts of one transferred macro-sequence (Seam 1).

    success          True iff every planned step was executed in order.
    steps_used       number of planned steps actually executed.
    steps_planned    length of the transferred ActionSequence.
    reward_delta     reserved (0.0): the architecture has no generic
                     scalar reward; credit MAGNITUDE is the spec's
                     C_PLAN·trace (§2.15.3), and execution completeness
                     scales the trace weight (partial-execution rule).
    concepts_involved realization basis at emission time (informational;
                     the pipeline re-derives the live basis at marshal
                     time via utility_engine.transfer_basis).
    """

    core_hash: bytes
    success: bool
    steps_used: int
    steps_planned: int
    reward_delta: float = 0.0
    concepts_involved: Tuple[int, ...] = field(default_factory=tuple)
    t: int = -1

    @property
    def completion_ratio(self) -> float:
        if self.steps_planned <= 0:
            return 0.0
        return max(0.0, min(1.0, self.steps_used / float(self.steps_planned)))

@dataclass(frozen=True)
class LabelSignal:
    """One labeled observation marker (B3/§5.8.2).

    polarity: +1 → the window is a positive instance of `label`;
              -1 → a negative instance (label absent).
    """

    label: Any
    polarity: int
    t: int
    confidence: float = 1.0

