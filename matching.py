"""parameters/matching.py -- Matching, edge schemas and similarity constants.

A VIEW over the canonical parameter namespace (Move 7b): the
constants are defined once, in parameters/_text.py, and re-exported
here for discoverability. Nothing is defined here.
"""
from __future__ import annotations

from parameters._text import (
    CATEGORY_WINDOW_MAX,
    DEG_MAX,
    DEG_MAX_3D,
    DIAG_INTERVAL,
    EBAF_TEST_MODE,
    EBAF_T_INJECT,
    GAP_QUEUE_MAX,
    LABEL_HISTORY_MAXLEN,
    NODE_DETECTION_BLOAT_FACTOR,
    NODE_DETECTION_SWEEP_PERIOD,
    PAD_DIST,
    Q_D,
    Q_DELTA,
    Q_P,
    Q_R,
    Q_S,
    Q_T,
    STATE_PATTERN_INDEX_CAPACITY,
    STRUCT_ALIGN_REFINE_ROUNDS,
    STRUCT_SIM_ALPHA,
    STRUCT_SIM_MIN_OVERLAP,
    STRUCT_SIM_W_ALIGN,
    STRUCT_SIM_W_SIZE,
    STRUCT_SIM_W_TYPE,
    TAU_ACTIVE_DRIFT_FRACTION,
    TAU_RECOMPUTE_CADENCE,
)
