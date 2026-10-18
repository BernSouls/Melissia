"""parameters/dynamics.py -- Dynamics, activation and temporal constants.

A VIEW over the canonical parameter namespace (Move 7b): the
constants are defined once, in parameters/_text.py, and re-exported
here for discoverability. Nothing is defined here.
"""
from __future__ import annotations

from parameters._text import (
    ADAPT_CAP,
    ADAPT_DECAY,
    ADAPT_RATE,
    BETA_SUPPORT,
    C_ENABLEMENT,
    C_MAX,
    C_PLAN,
    C_SUBGOAL,
    DELTA_ACT,
    EPS_INITIAL,
    ETA_SUPPORT,
    KAPPA_ACT,
    KAPPA_SPEC,
    KAPPA_SUPPORT,
    N_STEP_MIN,
    RHO_ACT_TARGET,
    RHO_REVERSAL,
    SIGMA_ACT_NOISE,
    S_MAX,
    THETA_ACT_FLOOR,
    THETA_ACT_MAX,
    THETA_ACT_MIN,
)
