"""parameters/_text.py -- the canonical parameter text (Move 7b).

Byte-for-byte the former parameters.py. Executed exactly once, in the
original statement order, so every intra-file dependency
(e.g. THETA_FERTILE_PROTECT = 0.1 * RF_MAX) resolves as it did before
the split. Nothing else in this package re-defines a constant; the
view modules only re-export names from here.
"""
from __future__ import annotations

"""
parameters.py — Canonical Architecture Configuration
Single source of truth for constitutional constants and homeostatic baselines.

Organization (v31):
  1.  Numerical safety
  2.  Hash widths
  3.  Time & history windows
  4.  Crystallization & witness thresholds
  5.  Capacity limits
  6.  Activation & adaptation
  7.  Edge weights (shared EMA parameters)
  8.  Edge homeostasis & per-type thresholds
  9.  Fertility & growth potential
  10. Credit & step-blame
  11. Contextual fertility
  12. Extraction: lenses
  13. Extraction: binders (order-1)
  14. Extraction: binders (order-2)
  15. Extraction: self-binders
  16. Extraction: quantization
  17. Binder threshold mechanism
  18. Builder (§4.22)
  19. Consolidator (§5.13)
  20. Mapper (§6.8)
  21. Executor (§7.8)
  22. Prediction (§8.8)
  23. Utility engine
  24. Satiety & lockouts
  25. Edge type classification
  26. Legacy (v30 constants retained during migration)

Naming convention: constants use SPEC NAMES where the spec provides one.
Greek letters map to English names (τ→TAU, θ→THETA, κ→KAPPA, ε→EPS, λ→LAMBDA,
η→ETA, α→ALPHA, β→BETA, γ→GAMMA). Module prefixes are NOT added; if a
spec name is unique across all sections, it is used bare. Collisions are
handled by section context or explicit disambiguating suffixes.
"""

import math

# [GAS LINE] Toggle for discriminative-mass weighting in query scoring.
# When False, evidence_score sees uniform mass and behaves exactly as before
# this feature existed. Kill switch, not a tuning parameter.
EVIDENCE_DISCRIMINATIVE_MASS: bool = True
import os
from typing import Dict, Optional, Tuple

# =============================================================================
# 1. NUMERICAL SAFETY
# =============================================================================

EPSILON = 1e-9

# =============================================================================
# 2. HASH WIDTHS
# =============================================================================

HASH_BITS = 64  # Legacy OPAQUE signature hash width
HASH_BITS_CANONICAL = 128  # v31 canonical subgraph hash width (§4.12)

# =============================================================================
# 3. TIME & HISTORY WINDOWS
# =============================================================================

W_HISTORY = 20  # General history window (legacy)
W_T = 10  # Temporal coherence window for transduction Ch 3
W_R = 20  # Rhythm window (self-binder + Periodicity binder)
W_CM = 10  # Co-motion window (velocity coherence)
W_SEQ = 20  # Sequence window (§2.3, §5.7)
W_SYNC = 20  # Synchrony window (P2-2)
W_CONFIRM = 5  # Effect-detection window (causal attribution, §8.8)
W_CHUNK = 20  # CHUNK co-activation window
W_PLAN_CHUNK = 100  # Legacy: plan-reuse window for CHUNK

# =============================================================================
# 4. CRYSTALLIZATION & WITNESS THRESHOLDS
# =============================================================================

# v30 concept-path thresholds (retained for ConceptGraph's witness path)
THETA_OPP = 3
THETA_CONF_DEFAULT = 3
THETA_CONF_MIN = 2
THETA_CONF_MAX = 10
THETA_ANCHOR_DEFAULT = 2
THETA_ANCHOR_CAUSAL = 1
THETA_SPAN = 2
THETA_WITNESS = 5
THETA_DECISIVE = 0.5
ETA_ARITY = 0.05
RHO_SYM_MAX = 0.5
N_MAX_AMBIGUITY = 50
TAU_AMBIGUITY = 500
TAU_FORGET = 1000
TAU_TOMBSTONE = 1000
N_LN = 8
THETA_LN = 0.6

# v31 subgraph-path thresholds (§4.17)
THETA_OPP_SUBGRAPH = 3
THETA_CONF_SUBGRAPH = 3
THETA_SPAN_SUBGRAPH = 2
THETA_ANCHOR_SUBGRAPH_SINGLE = 1  # SPATIAL_OBJECT, EVENT_CAUSAL, TEMPORAL_CHAIN
THETA_ANCHOR_SUBGRAPH_DEFAULT = 2  # All other families

# =============================================================================
# 5. CAPACITY LIMITS
# =============================================================================

N_CAND_MAX = 1000
N_LAYER_MAX = 500
N_NODE_MAX = 5000
N_TOMB_MAX = 1000
# v30 §2.1.16: N_protoMax — max sensory-grounding prototypes. Previously
# misnamed N_ANCHOR_MAX ("InnateReflex working-memory cap"), which collided
# semantically with the constitutional anchor bound N_ANCHOR_MAX_3D = 16
# (13_Update §3.4 working-memory bottleneck).
N_PROTO_MAX = 1000
N_ANCHOR_MAX = N_PROTO_MAX  # DEPRECATED alias; prefer N_PROTO_MAX / N_ANCHOR_MAX_3D
N_ANCHOR_MAX_3D = 16  # 13_Update §3.4: constitutional working-memory bound
N_CTX_MAX = 50  # Per-concept context cap (FertilityEngine)
N_EDGE_MAX = 100000  # v30 §2.1.6: max tracked (state, action) pairs (AEM)

# v31 §2.8.1: max valid operator-instances considered per promoted operator
# (before the family-key/anchor filter prunes to valid_instances).
OPERATOR_MAX_INSTANCES: int = 100

# v31 Builder/Consolidator/Mapper capacity bounds
BUILDER_CANDIDATE_MAX = 50  # §4.12 max nodes in a CandidateSubgraph
CORE_SIZE_CAP_NODES = 20  # §4.12, §5.13
CORE_SIZE_CAP_EDGES = 40  # §4.12, §5.13
FRONTIER_MAX = 10_000  # §4.11.1
TRANSPOSITION_TABLE_MAX = 1_000_000  # §4.11.1
MAX_ANCHORS_PER_CANDIDATE = 32  # §4.3.1
MAX_SITUATION_SIZE = 50  # §6.2
MAX_MCS_CANDIDATES = 100  # §5.7
MAX_PENDING = 1000  # §8.5.1
MAX_SURPRISE_QUEUE = 100  # §4.10.5, §8.6.4
MAX_SURPRISE_PER_STEP = 10  # §8.6.4
N_REDUCTION_TEMPLATES = 50  # §5.4
PATTERN_NODE_TTL = 100  # §1.11
TARJAN_CADENCE = 1  # [P2-3] §5: recomputes of the articulation-point cache
# in _prune_pattern_nodes happen at most once every TARJAN_CADENCE steps.
# 1 = recompute every pruning pass (exact, no staleness); raise only if
# Tarjan cost dominates step time on large structure graphs.
N_PROTO_MAX_EBAF = 200  # §8.4.1
# NOTE (§4.17): candidate `contexts` is unbounded in v31. No cap constant
# exists; if memory pressure becomes real, add a deterministic hash-order
# eviction policy and introduce MAX_CONTEXTS_PER_CANDIDATE at that time.

# =============================================================================
# 6. ACTIVATION & ADAPTATION
# =============================================================================

THETA_ACT_MIN = 0.1
THETA_ACT_MAX = 0.9
THETA_ACT_FLOOR = 0.05
KAPPA_ACT = 0.04
RHO_ACT_TARGET = 0.2
KAPPA_SUPPORT = 1.0
ETA_SUPPORT = 0.05
BETA_SUPPORT = 0.8
S_MAX = 100.0
SIGMA_ACT_NOISE = 0.05
DELTA_ACT = 0.02
ADAPT_RATE = 0.02
# v30 §2.18.2 architectural guarantee: adapt_cap MUST satisfy
#     adapt_cap > 1.0 − θ_act_floor
# so sustained activation at z = 1.0 eventually deactivates and re-onsets.
# The v30 table default (0.7) violates its own MUST at θ_act_floor = 0.05;
# the normative constraint governs over the stale default. Derived from
# constitutional constants (one hysteresis margin δ_act above the bound):
ADAPT_CAP = 1.0 - THETA_ACT_FLOOR + DELTA_ACT  # = 0.97 > 0.95 ✓
ADAPT_DECAY = 0.8

# =============================================================================
# 7. EDGE WEIGHTS (shared EMA parameters)
# =============================================================================
# §1.12: all edge types share the same EMA update formula:
#   weight = BETA_W * weight + (1 - BETA_W) * signal
# β_edge in the spec corresponds to BETA_W here.
# W_INIT and W_MAX apply to all edge types uniformly in v31.

W_INIT = 0.5
W_MAX = 10.0
BETA_W = 0.99  # Spec: β_edge
ETA_W = 0.1    # Edge-weight delta EMA decay (v30 value; consumed by
               # concept_graph's per-concept params and edge homeostasis)
DELTA_W = 0.001
KAPPA_COACT = 0.1  # Legacy: edge.CoAct_EMA update rate

# --- Hebbian edge reinforcement (structural_evidence boost) ---
# β_e: per-step decay of the per-edge crystallization-evidence EMA.
#   0.998^500 ≈ 0.367, so an edge idle for TAU_EDGE_INACTIVE = 500 steps
#   has its evidence decayed to ~0.37 — clearly on its way to zero
#   without having fully forgotten recent usefulness. Half-life ≈ 346.
# η_e: boost magnitude at evidence saturation. An edge with e → 1.0 has
#   its firing signal multiplied by (1 + η_e). 5.0 → 6× boost at
#   saturation. Raised from 0.5 after Part 1b canary showed zero pin
#   movement — the observed evidence distribution (mean 1e-5 to 2e-3)
#   is three orders of magnitude below the range that η_e = 0.5 was
#   designed for. At η_e = 5.0, tool edges near the mean get a ~1%
#   boost and high-evidence edges get ~25%, which is a magnitude that
#   can plausibly affect activation spreading and situation retrieval.
#   If pins move by more than ~50, this is the first knob to lower.
BETA_E = 0.998
ETA_E = 5.0

# =============================================================================
# 8. EDGE HOMEOSTASIS & PER-TYPE THRESHOLDS
# =============================================================================
# §1.12 per-type thresholds and homeostatic targets. v31 initializes all
# types to the same values; type-specific tuning is v32.

THETA_EDGE_ACTIVE = 0.1  # Initial active-edge threshold, all types
RHO_EDGE_TARGET = 0.3  # §1.12 initial homeostatic target
KAPPA_EDGE_HOMEOSTASIS = 0.05  # κ in θ_edge ← θ_edge + κ(r̄ - ρ_target)
KAPPA_ENTROPY_FEEDBACK = 0.01  # [P2-4] §8: gain on type-dominance term in
# θ_edge ← θ_edge + κ(r̄ - ρ_target) + κ_entropy·max(0, share - 1/|types|).
# Pushes down over-dominant edge types to keep the type-share distribution
# from collapsing (entropy homeostasis).
TAU_EDGE_INACTIVE = 500  # Pruning threshold per edge type

# =============================================================================
# 9. FERTILITY & GROWTH POTENTIAL
# =============================================================================

RF_MAX = 10**9
ALPHA_GP = 0.3
BETA_GP = 0.6
GAMMA_GP = 0.1
DELTA_GP = 0.2
GP_MAX = 1.0 + ALPHA_GP + BETA_GP + GAMMA_GP + DELTA_GP
TAU_RECENT = 200

# =============================================================================
# 10. CREDIT & STEP-BLAME
# =============================================================================

EPS_INITIAL = 0.25
C_ENABLEMENT = 1.0
C_PLAN = 0.1
C_SUBGOAL = 0.05
C_MAX = 10**6
KAPPA_SPEC = 4.0
RHO_REVERSAL = 0.5
N_STEP_MIN = 5

# =============================================================================
# 11. CONTEXTUAL FERTILITY
# =============================================================================

ETA_CTX = 0.3  # §8.8 (v31 value; v30 was 0.5)
BETA_CTX = 0.995  # §8.12 per-step CF decay
K_CONTEXT = 10
CF_MAX = RF_MAX
C_CTX = C_ENABLEMENT
THETA_PROTO_EF = 0.2  # §8.4.2 prototype registration threshold
TAU_PROTO_DECAY = 1000  # §8.4.3 prototype decay timescale
ETA_EBAF = 0.2  # §8.8 η_ebaf (v30 had 0.3)

# =============================================================================
# 12. EXTRACTION: LENSES
# =============================================================================
# §2.2. Constants for lens emission, normalization, and signal construction.

BOUNDARY_LENS_RADIUS_MULT = 1  # radius = R_BOUNDARY * 2^scale_idx
COLOR_QUANT_BINS = 12  # 12 bins per RGB channel
MOTION_SPEED_MAX = 1.0  # velocity normalization ceiling (V_MAX)
MOTION_DIR_BINS = 8  # direction quantization bins per axis
POSITION_NORM_RANGE = 1.0  # egocentric position normalization range.
                                     # Set to the expected maximum displacement of
                                     # entities from the agent. Tabletop (m): 1.0
                                     # is correct. Larger environments must raise
                                     # this to avoid clamping-induced information loss.
ONSET_AGE_NORM = 100.0  # age normalization window
# SIZE_RADIUS_MIN / SIZE_RADIUS_MAX are derived from the adapter's physical
# radii in Section 28 (single definition; r_min / R_max_obj per 13_Update).
TEXTURE_DEPTH_VAR_MAX = 1.0  # local depth-variance ceiling
TEXTURE_DELTA_EMA_BETA = 0.9         # planarity-delta EMA decay (Texture lens)
PRESENCE_EMA_BETA = 0.9              # presence EMA decay (all lenses)

# --- Orientation lens (v8) ---
# Angle bins over [0, π). N_theta = 24 → 7.5° resolution. Slot 0 of the
# ORIENTATION signature is circular (see CIRCULAR_CYCLE_LENGTHS in
# primitives). NOTE: primitives.CIRCULAR_CYCLE_LENGTHS["ORIENTATION"][0]
# and the ORIENTATION range in CONSTITUTIONAL_RANGES must stay in sync
# with this value; both are hard-coded there for layering reasons.
ORIENTATION_N_THETA = 24
# Elongation bins over [0, 1].
ORIENTATION_N_ELONG = 5
# Above this elongation (near-circular region), the major axis is
# ill-defined and the lens skips the entity.
ORIENTATION_ELONG_SKIP = 0.9
# Minimum patch gradient energy (region "roughness") for the Orientation
# lens to emit at all. On a uniform patch grid, isotropic/flat patches
# pass the ORIENTATION_ELONG_SKIP gate (elongation ~= 0 < 0.9) and emit an
# essentially arbitrary major axis, seeding a noise lattice that
# ORIENTATION_BEND/JUNCTION weld into one giant component. Measured on the
# apple/car/cow 8x8 patch grid: 80%+ of emitters sit below roughness 0.05
# (flat background) sit below 0.05; smooth-object edges live in the
# 0.035-0.06 band (apple edge max is only 0.064 on 32px patches). 0.05 severed
# that band and killed CONTINUITY on apple/cow (Part=0); 0.035 keeps the
# soft-edge band while still pruning the flat floor (median roughness ~= 0.030).
ORIENTATION_GRADIENT_MIN = 0.035

# --- Orientation segment lens (v9) ---
ORIENTATION_SEGMENT_TARGET_N = 6
ORIENTATION_SEGMENT_MIN_ARC = 4.0
ORIENTATION_SEGMENT_MAX_PER_REGION = 8
ORIENTATION_SEGMENT_N_ARC = 8
ORIENTATION_SEGMENT_ARC_MAX = 32.0
ORIENTATION_SEGMENT_N_POS = 8
ORIENTATION_SEGMENT_MIN_POLYLINE = 12

# --- Contour chaining (v9, removed in v10; dormant) ---
CHAIN_MAX_GAP = 8.0
CHAIN_MAX_ANGLE_DEV = 0.4
CHAIN_MIN_POLYLINE = 8

# --- T-junction detection (v9) ---
T_JUNCTION_MAX_GAP = 6.0
T_JUNCTION_MAX_PERP_DEV = 0.5
T_JUNCTION_MIN_FOREIGN_ARC = 10.0

# --- CURVATURE binder (v9) ---
CURVATURE_SIGMA_THETA = 0.3
CURVATURE_MIN_DELTA_THETA = 0.05

# --- OCCLUSION binder (v9) ---
OCCLUSION_MIN_CONFIDENCE = 0.4
OCCLUSION_3D_MIN_DEPTH_DIFF = 0.08
OCCLUSION_3D_SIGMA_DEPTH = 0.20
# --- Stage 3: 2.5D Ordinal Depth DAG + V4 Polar Fragments ---
# All unmeasured priors, not tuned (futurework6.md §Stage 3).
# Per-source trust applied to depth-constraint effective weights before cycle
# invalidation; T-junctions are trusted above border-ownership votes.
DEPTH_SOURCE_TRUST = {"tjunction": 1.0, "border_ownership": 0.5}
# Minimum |mean signed border-ownership| before a figure/ground ordinal
# constraint is emitted (below this the vote is noise).
DEPTH_BO_MIN_STRENGTH = 0.15
# Pasupathy & Connor object-centred polar fragments: target arc length as a
# fraction of the object envelope radius, and the Menger-curvature clamp.
CONTOUR_FRAGMENT_LEN_FRAC = 0.35
CONTOUR_FRAGMENT_KAPPA_CAP = 8.0

# =============================================================================
# STAGE 4: ARTMAP VIGILANCE & SHOCK SKELETON PARAMETERS
# =============================================================================
# Set to None to keep behavior byte-identical to pre-Stage 4.
# When set to a float in (0, 1], enables ARTMAP prototype resonance merging.
ART_VIGILANCE: Optional[float] = None
ART_MAX_CANDIDATES: int = 64

# Shock graph extraction parameters
SHOCK_STEP_FRAC: float = 0.03
SHOCK_R_MIN_FRAC: float = 0.03
SHOCK_R_MAX_FRAC: float = 1.05
SHOCK_TOL_SYM: float = math.pi / 12.0
SHOCK_TOL_TAPER: float = math.pi / 6.0
SHOCK_MIN_RUN: int = 4
SHOCK_PRUNE_ASPECT: float = 0.5
SHOCK_JOIN_FRAC: float = 1.2

# =============================================================================
# STAGE 5: ASPECT GRAPH & MULTI-VIEW TRANSFORMATION CONSTANTS
# =============================================================================
# [Stage 5 activation] ON. The descriptor source was rewired from dead shock
# ribbons to the live field's PEAK_CHAINs (memory/aspects.py
# build_view_descriptor_from_chains), so a view is now always constructible
# on the region path. Prototypes are viewpoint-windowed by
# ASPECT_ASSIGN_MAX_DEG and links connect DIFFERENT prototypes, so the graph
# is a manifold of per-viewpoint templates joined by experienced
# transformations -- not one averaged blob.
ASPECT_ENABLED: bool = True
ASPECT_PAIR_MODE: str = "metadata"       # "metadata" | "temporal"
ASPECT_VIGILANCE: float = 0.80
ASPECT_ASSIGN_MAX_DEG: float = 25.0
ASPECT_LINK_MAX_DEG: float = 50.0
ASPECT_LINK_SIGMA_DEG: float = 30.0
ASPECT_TRACE_ETA: float = 0.2
ASPECT_W_MAX: float = 1.0
ASPECT_TRACE_DECAY: float = 0.998
ASPECT_LINK_MIN_W: float = 0.05
ASPECT_SIGMA_POS: float = 0.35             # in units of R_env
ASPECT_MATCH_MIN: float = 0.35
ASPECT_ORDER_MISMATCH: float = 0.5         # part similarity multiplier when shock order differs
ASPECT_PERSIST_CAP: int = 64
# [Q2] Graded part matching, SHIPPED DARK (off) per the operating policy
# (P1: a scoring-facing change is not enabled until second instances per
# class confirm it). When on, part similarity uses ASPECT_PART_SLOTS and the
# graded slot distance replaces the exact slot-0 equality gate.
# Slot choice is not a free parameter: it is the measured noise screen on
# TRAINING views (diagnostics/_q2_slots.py). A slot is dropped when a single
# value covers >= 75% of all parts (noise-dominated, no spread to match on):
#   slot 0 n_peaks_bin      top-1 share 0.313  F=0.34  -> kept
#   slot 1 mean_theta_bin   top-1 share 0.151  F=2.35  -> kept (strongest)
#   slot 2 theta_var_bin    top-1 share 0.764            -> dropped
#   slot 3 closure_bin      top-1 share 0.986            -> dropped
#   slot 4 curv_summary_bin top-1 share 0.973            -> dropped
#   slot 5 length_bin       top-1 share 0.322  F=0.67  -> kept
ASPECT_GRADED_PARTS: bool = False
ASPECT_PART_SLOTS: Tuple[int, ...] = (0, 1, 5)
# [SIZE SLOT] The view's absolute size (object RMS spread, pixels) is stored as
# its OWN slot (ViewRecord.size) apart from the normalized shape. The matcher
# reads it through a separate log-ratio kernel with this sigma (in log units:
# 0.7 ~= a factor-e mismatch). Enabled by default -- the slot exists precisely
# so size is available to matching; set False to score on normalized shape only.
ASPECT_SIZE_SLOT_ENABLED: bool = True
ASPECT_SIZE_SIGMA: float = 0.7

# =============================================================================
# [Q3] NEGATIVE (ANTI-PHASE) COUPLING
# =============================================================================
# The coupled-phase oscillator separates two cliques only when the coupling
# matrix can say NO (dynamics/oscillator.py: negative edge weights drive the
# phase difference toward pi instead of 0). Measured before: the observed
# graph of a two-panel composite carried 146 edges, ALL sign=+1, 118 of them
# EDGE_CONTACT proximity noise, so two-panel separation could not be positive
# for a structural reason. These switches enable the two anti-phase sources in
# perception/anti_phase.py, wired by orchestration/binder.py:
#
#   1. opposite flank owners across a shared boundary (mutual
#      BorderOwnership claims) -- the figure-ground / occlusion relation
#   2. closed-chain interior vs exterior (a ring PEAK_CHAIN enclosing another
#      detection's anchor)
#
# Ships DARK by default for the same reason the other scoring-facing work
# does: the gate is the two-panel separation measurement, not an assertion.
ANTIPHASE_ENABLED: bool = False
#: Minimum confidence (weakest-claim strength / containment depth) for an
#: anti-phase edge to be emitted. Calibrated by a rule written before looking
#: (diagnostics/_q3_strength.py): "p10 of the observed BorderOwnership
#: strength distribution rounded down to two decimals". Measured over 552
#: detections on 3 composites + 2 real images: p10 = 0.200, p50 = 0.214,
#: max = 0.347, so the threshold is 0.19 and ~90% of claims pass.
ANTIPHASE_MIN_CONFIDENCE: float = 0.19
#: Per-node cap on EDGE_CONTACT edges (proximity-only, so its count, not its
#: structure, is what dominated the coupling matrix). Distance decay stays;
#: the cap only bounds how many neighbours each node may bind to.
CONTACT_MAX_PER_NODE: int = 6
#: [Q3] Apply the per-node contact cap. Shadow of ANTIPHASE_ENABLED: the cap
#: is part of the same measured change set (it changes which edges reach the
#: coupling matrix), so it ships dark under its own switch.
CONTACT_CAP_ENABLED: bool = False

# --- ORIENTATION_CONTINUITY binder (v8) ---
# Angular tolerance: signal falls off as exp(-(Δθ/σ_θ)²), Δθ in radians.
ORIENTATION_CONTINUITY_SIGMA_THETA = 0.3
# 0.5 rad ~= 29deg: on a uniform 32px patch grid, the major-axis
# estimate rotates ~(pitch / radius_of_curvature) between adjacent
# rim patches (~0.4-0.5 rad for object-scale rims), so 0.3 let only
# straight background lines chain while every curved object rim fell
# under THETA_CONTOUR_EDGE -- but loosening also merged rim and
# background chains into 10-node comps and slightly worsened class
# gaps, so the change was REVERTED to 0.3 after measurement.
# Perpendicular distance tolerance as a fraction of scene L_med. On a
# uniform grid (pitch = L_med), 0.5*L_med structurally rejects every
# DIAGONAL edge continuation (perp = L_med/sqrt(2) ~ 0.71*L_med ->
# exp(-2) = 0.135), so smooth curves could never chain; 1.0 restores
# diagonal continuations (exp(-0.5) ~ 0.61) while the collinearity
# term (sigma_theta) remains the semantic gate.
ORIENTATION_CONTINUITY_SIGMA_P_FRAC = 1.0
# Along-axis gap decay as a multiple of scene L_med.
ORIENTATION_CONTINUITY_TAU_GAP_FRAC = 2.0

# [Pass 21] Per-port competition and bipole role parameters.
# All five values are UNMEASURED initial estimates. Calibrate against ETH-80.
ORIENTATION_CONTINUITY_RANGE_TAU_MULT = 2.0      # search radius = this x tau_gap
ORIENTATION_CONTINUITY_PORT_MIN_SIGNAL = 0.05    # floor for a port to count as supported
ORIENTATION_CONTINUITY_MUTUAL_FLOOR = 0.7        # signal factor for non-mutual winners
ORIENTATION_CONTINUITY_BIPOLE_STRONG = 0.4       # sqrt(s_fwd*s_bwd) cut for "strong interior"
ORIENTATION_CONTINUITY_POLARITY_FLIP_GAIN = 1.0  # 1.0 = polarity is slot-only, no signal effect

# v32: shared Gaussian sigma for ORIENTATION_BEND and
# ORIENTATION_JUNCTION. Approximately 15° (π/12). The angle-range
# membership test uses a discrete n_theta=24 lattice with 7.5° bin
# spacing, so sigma=π/12 gives ~0.78 similarity to adjacent bins
# and ~0.37 to two-apart bins.
ORIENTATION_ANGLE_RELATION_SIGMA_THETA = math.pi / 12.0
# Spatial horizon for ORIENTATION_BEND / ORIENTATION_JUNCTION, as a
# multiple of the step's EXTRACTION SCALE (patch radius), not of
# _step_l_med. A bend or junction is a LOCAL relation: on a uniform grid,
# _step_l_med collapses to the grid pitch (32px) and the old
# TAU_GAP_FRAC*l_med = 64px horizon welded patches up to ~3 cells apart --
# with no hard cutoff -- into one image-spanning angle-relation component.
# 1.0 * extraction_scale (~22.6px at 8x8/256) keeps nearest-neighbour
# locality (dist 32px ~> e^-1.41) and kills diagonal/long-range welds.
ORIENTATION_ANGLE_RELATION_KAPPA = 1.0

# =============================================================================
# v32.2 GRADED CORE DESIGN
# =============================================================================
# Thresholds removed from the flow:
#   - intersection's min_positive_ratio → replaced by min_frequency (default 0.0)
#   - filter's theta_pos_min / theta_disc_norm → unused
#   - mcs_score's flat count → replaced by mass-weighted, smaller-side numerator
#
# Mass pipeline:
#   1. Intersection emits per-element consistency ∈ [0, 1] = mean pairwise
#      signature similarity among aligned positives (nodes), or presence
#      frequency (edges). Stored in Subgraph.node_weights / Subgraph.weights.
#   2. Filter multiplies by max(0, disc)**DISC_EXPONENT and overwrites the
#      same fields with the final mass.
#
# final_mass = consistency * max(0, disc)**DISC_EXPONENT
#
# SEMANTIC SHIFT: Subgraph.node_weights / Subgraph.weights mean
#   - consistency prior ∈ [0, 1]  before the filter
#   - final mass ∈ [0, 1]         after the filter

# [FIX 2 / Item round 3] Recurrence rule: single source of truth is
# _wilson_min_recurrence() in memory/distributions.py. It computes the
# minimum recurrence count pc such that Wilson-lower(pc/n_pos) >
# Wilson-upper(0/n_neg). For n_pos=7, n_neg=8 this gives pc=4
# (frequency 4/7 ≈ 0.571).
#
# THETA_MIN_FREQ previously defaulted to 0.0 (dead knob: no filtering)
# then 0.5 (hand-set constant competing with the Wilson rule). It is now
# env-override only, labeled diagnostic, with no default. Callers that do
# not pass min_frequency explicitly get the Wilson-computed value from
# _structural_intersection.
#
# THETA_MIN_CONSISTENCY was a dead 0.0 knob (0.0 = no consistency
# filtering at the intersection step). Removed; the Wilson recurrence
# test subsumes its purpose.
THETA_MIN_FREQ: Optional[float] = (
    float(os.environ.get("THETA_MIN_FREQ")) if os.environ.get("THETA_MIN_FREQ") else None
)

# =============================================================================
# v32.3 CONTENT-TIER SIMILARITY CLUSTERING
# =============================================================================
# Content-tier identity was previously hash(exact_signature). Two signatures
# one bin apart produced distinct nodes and were never treated as the same
# content. This caused the 1-bit quantization trap: a bin-boundary wobble
# across the class appears as two disjoint class-specific nodes, inflating
# both `disc` values. Clustering content nodes by similarity θ_cluster
# merges near-identical signatures into one node.
#
# Semantics: when a content signature arrives, resolve it to the nearest
# existing cluster centroid of the same lens type whose similarity is at
# least θ_cluster. If found, join that cluster (update centroid as running
# mean). If not found, create a new cluster seeded by this signature.
#
# θ_cluster is on TypedSignature.similarity's scale. Given the current
# flat similarity calibration (~0.05 per bin on a 20-range), θ_cluster=0.90
# merges signatures within ~2 total bin units and is the default.
# Env-overridable so the sweep does not require an edit per point.
#
# Instance-tier identity (§1.11 `(lens_type, signature_hash)`) is
# unchanged. Only the content tier clusters.
# [v32.3] Content-tier similarity clustering. Set to 0.99 to disable
# merging in practice (current `similarity` scale cannot exceed 0.95
# for adjacent bins, so 0.99 thresholds effectively yield hash-deterministic
# IDs). The v32.4 distribution classifier does NOT rely on clustering;
# this default preserves the pre-clustering behavior.
THETA_CLUSTER_CONTENT: float = float(os.environ.get("THETA_CLUSTER_CONTENT", "0.99"))

# [v32.5] Distribution normalization mode for the content distribution
# classifier.
#   'per_lens'     — each lens type normalized to mass 1.0 (v32.4 behavior).
#   'global'       — raw disc values across all lens types, globally normalized.
#   'log'          — log(1+disc) per key, globally normalized. Transform
#                    applies to class-side disc only; query side is the
#                    same globally normalized presence distribution as
#                    'global'.
#   'class_global' — class side uses 'global' (flattened __global__),
#                    query side is flattened __global__ but per-lens
#                    normalized. Diagnostic mode for isolating class-side
#                    effects from query-side dilution.
# Env-overridable. Read once at import; not mutated at runtime.
VALID_DIST_NORM_MODES = ("per_lens", "global", "log", "class_global")

DIST_NORM_MODE: str = os.environ.get("DIST_NORM_MODE", "per_lens")
if DIST_NORM_MODE not in VALID_DIST_NORM_MODES:
    raise ValueError(
        f"unknown DIST_NORM_MODE: {DIST_NORM_MODE!r}; "
        f"expected one of {VALID_DIST_NORM_MODES}"
    )

GRADED_TOP_K_NODES: int = int(os.environ.get("GRADED_TOP_K_NODES", "100"))
GRADED_TOP_K_EDGES: int = int(os.environ.get("GRADED_TOP_K_EDGES", "100"))
DISC_EXPONENT: float = float(os.environ.get("DISC_EXPONENT", "2.0"))

# [Pass 20] OBJECT_SHAPE_COARSEN_FACTOR and OBJECT_SHAPE_CONTENT_MODE removed
# with the ObjectShape oracle. Nothing reads them now that ObjectShapeLens is
# excised and content_signature no longer has an ObjectShape branch.

# --- Phase A (Step 1 + Step 2) gating / calibration ---
# SIMILARITY_MODE selects TypedSignature.similarity implementation.
# Read at import of parameters.py (restart required to change).
SIMILARITY_MODE: str = os.environ.get("SIMILARITY_MODE", "gaussian")
# [POPULATION CODING] Master A/B switch: when True (default) a CANONICAL
# signature that carries a graded `population` for a registered class is
# compared by chance-corrected activity-vector OVERLAP instead of the per-slot
# Gaussian over bins (bins remain only the lookup/index key). This is the
# representation thesis; the switch exists so the whole population path can be
# A/B'd against the bins path without a code change. It does NOT gate whether a
# mechanism is built -- every registered class emits its full population
# unconditionally; only the matcher's read of it is switchable.
POPULATION_CODING_ENABLED: bool = os.environ.get(
    "POPULATION_CODING_ENABLED", "1").strip().lower() not in ("0", "false", "no", "")
# Sigma fallback for range-0 slots (e.g. BOUNDARY_ANCHORED slot 2).
SIGMA_FALLBACK_FOR_RANGE0: float = float(os.environ.get("SIGMA_FALLBACK_FOR_RANGE0", "5.0"))
# Scale applied to graded disc when exact disc is zero (dual filter fallback).
GRADED_FALLBACK_SCALE: float = float(os.environ.get("GRADED_FALLBACK_SCALE", "0.5"))
# Per-image edge cap for graded edge fallback (bounds O(core_edges * img_edges)).
MAX_GRADED_EDGES_PER_IMAGE: int = int(os.environ.get("MAX_GRADED_EDGES_PER_IMAGE", "200"))
# Master gate for dual (exact-preferred, graded-fallback) discriminative filter.
# Env default "1" (enabled); runtime-overridable via P.DUAL_FILTER_ENABLED assignment.
DUAL_FILTER_ENABLED: bool = os.environ.get("DUAL_FILTER_ENABLED", "1") != "0"

# --- Phase B (Representation tier) — futurework3.md B.4–B.16 ---
#
# Master switch for the six higher lenses, their binder plumbing, and
# surround suppression. Starts FALSE (B.16 / II.46) and is flipped TRUE in
# Phase B Step 4, when BorderOwnershipLens is the only implemented lens —
# the other five remain uncalled skeletons, so the tier is live but
# single-lens. Roll back by setting this back to False.
HIGHER_LENS_ENABLED: bool = True

# Phase B spatial kappas: multipliers of extraction quantum.
# extraction quantum = median V1 region radius in observation units
# (pixels in 2D, meters in 3D), same units as anchor_positions.
# Effective radius = KAPPA * extraction_scale.
# Scene-stable (does not change with scene crowding), resolution-
# independent, modality-agnostic. Same spirit as V1 receptive fields.
#
# WHY multipliers and not absolute radii: the Phase B draft specified
# absolute radii (1.5-3.0) that read as if they were in grid-cell units,
# but anchor_positions is in PIXELS (2D) or METERS (3D). On the Phase-A
# grid (256px image, GRID_N=8) the minimum anchor separation is 32px, so
# every absolute radius below 32 made its distance gate unsatisfiable —
# the whole higher tier was silently inert (verified: BorderOwnership
# emitted 0 at R=3.0, 128 at R=48.0; surround_suppress suppressed 0 at
# R=1.5). Kappas restore the spec's intended *relative* neighbourhood
# sizes while staying correct across resolutions and modalities.
KAPPA_SURROUND_SUPPRESS = 1.5
KAPPA_BORDER_OWNERSHIP = 3.0
KAPPA_CONTOUR_CONTINUITY = 2.0
KAPPA_JUNCTION_CONTOUR = 2.5
KAPPA_CLUSTER_PART = 2.0
KAPPA_CLUSTER_OBJ = 3.0

# B.4 surround suppression similarity gate (dimensionless; not a radius).
THETA_SURROUND_SUPPRESS: float = 0.9

# B.6 / B.7 edge-weight gates for the Orientation-neighbour graphs. These
# are thresholds on edge WEIGHT, not distances, so they need no kappa.
# R_CONTOUR_EDGE / R_JUNCTION_EDGE are spec names the draft never bound to
# an edge; retained at the local-neighbourhood scale.
R_CONTOUR_EDGE: float = 2.0
THETA_CONTOUR_EDGE: float = 0.15
R_JUNCTION_EDGE: float = 2.0
THETA_JUNCTION_EDGE: float = 0.15

# B.12 per-step pair caps that bound the O(n^2) emitter loops.
MAX_CONTOUR_PAIRS_PER_STEP: int = 2000
MAX_JUNCTION_CONTOUR_PAIRS: int = 2000

# B.14 recurrent contour grouping (ContourGainState).
CONTOUR_GAIN_INCREMENT: float = 0.1
CONTOUR_GAIN_DECAY: float = 0.9
CONTOUR_GAIN_FLOOR: float = 0.001
MAX_CONTOUR_GAIN_ENTRIES: int = 5000
MAX_CONTOUR_GAIN_VALUE: float = 1.0
# [Phase B] Step 9. Flipped True at the end of Step 9, after validating the
# threading against a flag-off V1 baseline (20/20 gate) and confirming on the
# image probe that gain actually reduces Orientation emissions.
RECURRENT_GROUPING_ENABLED: bool = True

# --- [B5] Contour curvature-sequence signature ---------------------------
CONTOUR_N_SEQ: int = 20
CONTOUR_SEQ_LEVELS: int = 9                 # odd: zero sits on the centre level
CONTOUR_CORNER_TURN_FRAC: float = 0.25      # corner if |turn| > 0.25 * pi
CONTOUR_TOTAL_TURN_MAX: float = 8.0 * math.pi
CONTOUR_GEOM_BAND_BINS: int = 3             # corners within this many bins of 90 deg use geometry
# [Pass 17.3] Default "summary": with CONTOUR_CONTENT_MODE="full" the 27-slot
# raw curvature sequence enters the content tier and THETA_CLUSTER_CONTENT=0.99
# fragments it across viewpoints (benchmark regressed 88.9% -> 66.7%). "summary"
# projects to the 5 invariant slots (length, closure, total_turn, corners,
# chirality) at the content tier; the rich 27-slot sequence stays at instance tier.
CONTOUR_CONTENT_MODE: str = os.environ.get("CONTOUR_CONTENT_MODE", "summary")

# --- [B6] Border Ownership Field -----------------------------------------
BO_TEXTURE_BIN_MAX: float = 9.0
BO_RICHNESS_FALLBACK: float = 0.0
BO_FIELD_SAMPLE_COUNT: int = 8
BO_CURVATURE_WEIGHT: float = 0.4
BO_TEXTURE_WEIGHT: float = 0.3
BO_COLOR_WEIGHT: float = 0.2
BO_GRAVITY_WEIGHT: float = 0.1
BO_CUE_SCALE: float = 2.0
BO_GRAVITY_SCALE: float = 10.0

# DEPRECATED — retained for env overrides and probe call-site compatibility.
# The graded filter accepts and ignores these.
THETA_DISC_NORM: float = float(os.environ.get("THETA_DISC_NORM", "0.10"))
THETA_POS_MIN_CONTENT: float = 0.30
POOL_SIGMA_FLOOR: float = 0.10


# --- SURFACE_GROUPING binder (v8) ---
# Color distance scale (RGB in [0, 1]³). Signal term exp(-d/σ_c).
SURFACE_GROUPING_SIGMA_C = 0.2
# Texture distance scale (2-dim quantized roughness/isotropy in [0, 1]²).
SURFACE_GROUPING_SIGMA_T = 0.3
# Absolute floor: candidate pairs below this raw signal are dropped
# before max-normalization.
SURFACE_GROUPING_MIN_SIGNAL = 0.05
# Spread gate: if max(signal) - min(signal) across candidates is below
# this, no edges are emitted (all-equal means nothing to discriminate).
SURFACE_GROUPING_SPREAD_MIN = 0.05

# =============================================================================
# 13. EXTRACTION: BINDERS (order-1)
# =============================================================================
# §2.3. Constants for order-1 binder signal functions.

TRANSITION_ONSET_LAG = 1  # Fixed at 1 step (§1.12)
CO_MOTION_MIN_FRAMES = 3  # v30 §2.1.4 θ_comotionFrames: min co-occurrence frames
CONTACT_R_MED_BOOTSTRAP = 1.0  # §1.12 CONTACT bootstrap default
TOPOLOGY_K_MAX = 4  # K_TOPOLOGY
TOPOLOGY_RADIUS_MULT = 2
DEGREE_RADIUS_MULT = 1
# [D2/Seam-4] Scale-adaptive DEGREE radius: R_eff = max(κ·L, floor) with L
# the median nearest-neighbour distance among positioned anchors. The law
# is scale-invariant (grid units, meters, pixels) — no absolute clamp
# bounds, which would re-hardcode a single metric scale. The floor only
# guards the degenerate co-located-anchors case (L = 0).
DEGREE_KAPPA = 1.5
DEGREE_RADIUS_FLOOR = 1e-6
ETA_SEQ_AGE = 0.1  # §1.12 SEQUENCE age decay
CONTRAST_MIN_OBS = 5  # Min observations for contrast signal
PERIODICITY_MIN_INTERVALS = 3  # Min intervals for periodicity signal
PERSISTENCE_CHANGE_THRESHOLD = 0.1  # "B changes" magnitude threshold

# =============================================================================
# 14. EXTRACTION: BINDERS (order-2) — REMOVED
# =============================================================================
# Structural similarity is no longer a binder: comparing edge-neighborhoods is
# a measurement, not an observed relation, so it emits no §2.8 edge type and
# lives in structural_similarity.py as a pure operator (§1.5, §4.12, §6.4.1).
# The order-2 binder's constants (STRUCT_SIM_NEIGHBORHOOD_DEPTH,
# STRUCT_SIM_MIN_OVERLAP, STRUCT_SIM_MAX_NODES) were deleted with it; the
# operator's own weights/threshold are defined once, under §20.

# =============================================================================
# 15. EXTRACTION: SELF-BINDERS (DEPRECATED / REMOVED)
# =============================================================================

# =============================================================================
# 16. EXTRACTION: QUANTIZATION
# =============================================================================
# Q_* values drive the constitutional R_j ranges in primitives.py.

Q_P = 0.1  # Persistence quantization
Q_S = 0.1  # Feature-vector stability quantization
Q_R = 0.05  # Relational distance quantization
Q_D = 1.0  # Degree quantization
Q_DELTA = 0.5  # Structure feature-delta quantization
Q_T = 1.0  # Time quantization
PAD_DIST = 20  # Topology absent-neighbor sentinel
DEG_MAX_3D = 8  # 3D contact degree maximum
DEG_MAX = 50  # Legacy 2D degree maximum

# =============================================================================
# 17. BINDER THRESHOLD MECHANISM
# =============================================================================
# §2.7. Structurally identical to lens θ, applied per relation channel.

THETA_CHANNEL_BASE_INIT = 0.15  # Lowered from 0.5. §2.7 thresholds must
                                # overlap the emittable signal band, which
                                # for self-binders is [0, ~0.2]. A 0.5 start
                                # never intersects real signals before the
                                # channel decays to floor, and empty-candidate
                                # steps (now homeostatic) still need to reach
                                # an entry threshold below ~0.17 to fire.
DELTA_CHANNEL_HYST = 0.02
KAPPA_CHANNEL_THETA = 0.05
RHO_CHANNEL_TARGET = 0.2
ETA_CTX_CHANNEL = 0.05
ETA_FAT_CHANNEL = 0.05
BETA_F_CHANNEL = 0.95
KAPPA_C_CHANNEL = 0.05
KAPPA_R_CHANNEL = 0.05
THETA_CHANNEL_MIN = 0.05        # Lowered from 0.1. A silent channel's
                                # homeostatic floor must sit below the
                                # smallest meaningful signal. Low-signal
                                # candidates from a one-bin BOUNDARY shift
                                # carry signal ~0.056; the entry threshold
                                # (theta_eff + DELTA_CHANNEL_HYST = 0.07)
                                # must be reachable from theta_eff = MIN.
THETA_CHANNEL_MAX = 0.9

# =============================================================================
# 18. BUILDER
# =============================================================================
# §4.22. Every constant listed in the Builder parameter table.

# Freshness and selection
TAU_FRESH = 50
ETA_FRESH = 0.2
THETA_CURIOUS = 0.3
THETA_STERILE = 0.1
ETA_SURPRISE = 0.3

# Stall detection
W_STALL = 20
STALL_THRESHOLD = 0.7
KAPPA_REF = 0.1
EPS_REF = 0.01

# Excursion bounds
TAU_EXCURSION_MAX = 100
TAU_EPISTEMIC_MAX = 50

# Basin and Spine maintenance
K_MAIN = 500
THETA_BASIN_FULL_RECOMPUTE = 0.5

# Frontier lifecycle
THETA_PRUNE_FRONTIER = 0.01
K_PRUNE_FRONTIER = 30

# Value function inputs
F_RECURSIVE_DEPTH = 3
LAMBDA_RECURSIVE = 0.7
THETA_FERTILE_PERCENTILE = 90  # §4.7 dynamic θ_fertile
RF_MIN_FERTILE = 1.0  # §4.7 θ_fertile floor
K_FERTILE = 100  # §4.7 θ_fertile recomputation cadence
FERTILE_MIN_POPULATION = 20
# [P0-2] Cold-start rescue: guarantee a non-empty fertile set when at
# least one crystallized concept has positive EF but sits below the
# percentile-derived candidate threshold. The 5% margin prevents ULP
# chatter near top_val; the 10% retighten margin prevents mode
# oscillation when top_val hovers near the candidate boundary.
FERTILE_ADMIT_MARGIN = 0.05
FERTILE_RETIGHTEN_MARGIN = 0.10

# [P0-3] Bootstrap-aware membership re-observation. A candidate is
# partially re-observed when at least this fraction of its members are
# present this step AND every edge with both endpoints present is
# active per §1.12. Fractional credit is accumulated in a per-candidate
# member_accum; each time it crosses 1.0, N_conf increments by 1.
THETA_MIN_PRESENT_RATIO = 0.5  # [P0-3] partial re-observation presence floor
TAU_MIN = 1.0  # §4.7 proximity smoothing floor
TAU_SAMPLE_SIZE = 100  # §4.7 pairwise sample cap

# Value function weights (§4.6.2)
BUILDER_ALPHA = 0.4  # F_expected weight
BUILDER_BETA = 0.4  # F_recursive weight
BUILDER_GAMMA = 0.2  # W weight

# Path/outcome pairs (§4.15)
W_SUCCESSOR = 20  # Steps to observe outcome after path
THETA_PAIR_TRIALS = 3  # Trials before emitting (P, S) pair

# Seeding, expansion, MCS backtracking
W_SEEDING_MAX = 500
BUILDER_EXPANSIONS_PER_STEP = 1
K_AMB = 100  # Typed-constrained MCS backtracking cap

# Analogical registration cadence (§8.11)
K_ANALOGICAL = 500

# Initial reliability scaling (§5.6, §4.22)
KAPPA_INIT = 0.8

# Surprise interaction with CF (§8.6.3)
ETA_CF_SURPRISE = 0.3

# [Phase 1 Decision 1] ConceptGraph three-signal pruning thresholds.
THETA_A = 0.05
THETA_RF = 0.001
# Fertility-protected pruning (§P3). Concepts with EF_combined >= this
# threshold are exempt from pruning: they are currently fertile, so
# culling them destroys downstream structure.
# NOTE: the spec (§P3) writes this as 0.1 * RF_MAX. With RF_MAX = 1e9
# this resolves to 1e8, a very high bar against realistic EF_combined
# values (clamped at RF_MAX but typically far below it). Expect the
# gate to be a no-op on the first landing; calibrate from
# `pruning_ef_max` in the concept_graph diagnostics once pruning
# activates.
THETA_FERTILE_PROTECT = 0.1 * RF_MAX
# §P3 specificity gate. spec = 1 / (1 + KAPPA_SPEC · avg_activation_ema)
# with KAPPA_SPEC = 4 and avg ∈ [0, 1] (activation is binary via
# heaviside → avg is a convex combination over {0,1}), so
# spec ∈ [0.2, 1.0]. Prunable requires spec < THETA_PRUNE_SPECIFICITY,
# i.e. avg > 0.25. Concepts firing less than a quarter of steps are
# protected as selective; concepts firing a quarter or more are treated
# as hub-like and are pruning-eligible (subject to the other gates).
THETA_PRUNE_SPECIFICITY = 0.5
# [Phase 1 Decision 1] Per-step eviction cap: prevents mass extinction
# when many dormant concepts cross the three-signal threshold in one step.
MAX_PRUNE_PER_STEP = 10

# =============================================================================
# 19. CONSOLIDATOR
# =============================================================================
# §5.13. Constants not already defined above (THETA_PAIR_TRIALS, K_CONSOLIDATE,
# CORE_SIZE_CAP_*, KAPPA_INIT are shared and defined in §5/§18).

# [DESIGN OBSERVATION] K_CONSOLIDATE = 500 means zero transfer ever
# fires in episode one on any episodic benchmark (rooms 490, tool 98,
# minigrid 597 — all end before the first tick). This is correct for
# continuous/long-running workloads. For episodic benchmarks, consider
# firing the first tick earlier or at episode end. Not fixing now;
# flagging for future design review.
K_CONSOLIDATE = 500  # §4.22, §5.13
# [G.2.2] Maximum number of real extraction attempts for a concept
# whose category-core extraction returned None. A budget-starved
# extraction succeeds on the next sweep; a genuinely unextractable
# concept is abandoned after this many attempts. Bounds sweep cost
# without sacrificing the starvation recovery this parameter exists
# to enable. The abandonment happens on the (N+1)-th sweep encounter
# without calling the extractor again.
MAX_CATEGORY_EXTRACTION_ATTEMPTS = 3
THETA_CATEGORY_INSTANCES = 5
THETA_CATEGORY_DIVERSITY = 0.5
THETA_CONTAINMENT = 0.95
THETA_GROWTH_PLATEAU = 0.01
M_PLATEAU = 100
KAPPA_RELIABILITY = 0.05
# [K.7] Core minimization floor. _minimize_core's only constraint is
# _classify_prefiltered (containment against every positive, no negative);
# a tiny core satisfies that vacuously -- a 2-edge core scores 1.0 against
# every positive -- so the floor is what stops the minimizer collapsing a
# core to noise. floor = max(MIN_CORE_STRUCTURAL_ELEMENTS,
# int(MIN_CORE_SIZE_RATIO * pre_edges)).
MIN_CORE_STRUCTURAL_ELEMENTS = 3      # absolute edge floor
MIN_CORE_SIZE_RATIO = 0.3             # retain >= 30% of original edges

# [G.2.1] Despite the name, this bounds MCS calls per EXTRACTION,
# not per step. Multiple extractions at one step each get a full
# budget (see consolidator._begin_extraction_budget for why). Renaming
# deferred to avoid a cross-module parameter change.
#
# [K.7 -> Phase L] The K.7 branch carried a raise to 100000 to test whether
# the budget blocked classification. It does not: the classify-only probe
# (K7_classify_probe_report.md) showed the entry classification is False for
# structural reasons, and the budget-10000 harness run died of OOM (the memo
# key is ~302 KB/entry, K7_budget10k_report.md). Reverted to 1000 here, per
# the standing constraint, while restoring the K.7 minimizer itself.
MCS_BUDGET_PER_STEP = 1000
# [L.1] Near-miss minimization. When the extension loop exits without
# classifying, _minimize_core can still run in "near_miss" mode: a removal is
# accepted iff it strictly increases min_pos - max_neg, subject to no negative
# crossing THETA_CONTAINMENT - NEAR_MISS_NEGATIVE_MARGIN. Entry requires a
# core that is close (every positive >= NEAR_MISS_THRESHOLD, no negative
# within the margin).
NEAR_MISS_THRESHOLD = 0.70            # min_pos must exceed this to enter
NEAR_MISS_NEGATIVE_MARGIN = 0.10      # max_neg must be below THETA_CONTAINMENT - this
NEAR_MISS_MCS_BUDGET = 4000           # per-extraction MCS call cap for near-miss mode

# --- DETERMINISM GUARD -- NOT A TUNING KNOB --------------------------------
# _EVID_TIE_GRID is the bucket width of the lexicographic tie-break in
# consolidator._find_best_extension (node-extension AND edge-extension sites):
#     key = (round(score / _EVID_TIE_GRID), tb)
# It must sit strictly BETWEEN float noise and the smallest real score gap:
#   * above float jitter (~1e-11), so nondeterministic last-bit differences
#     land in the same bucket and the content hash resolves them;
#   * below the smallest observed positive top-two gap
#     (2.7604159041905163e-08 at seeds 0 and 42, on the graded-param tree
#     after the Pass-23 edge vocabulary landed: probe's distinct margin
#     values = 17 per seed, so param_sim genuinely varies), so genuine
#     distinctions stay in different buckets and the hash never overrules
#     the score.
# The earlier coarse value (1.38e-08, from a pre-calibration gap of
# 1.38e-07) left only a 2x margin below the then-current min gap -- and a
# coarser grid (e.g. 0.0017857) would bucket both members of such a
# top-two pair together and hand the decision back to the hash --
# reintroducing the bug for exactly the near-ties that matter most.
# Derivation: min_positive_gap / 10 = 2.76e-08 / 10. DO NOT tune this
# against benchmark numbers -- it is a determinism guard. A grid at or above
# the minimum real gap collapses every score into one bucket and turns
# selection into a pure hash lottery, reintroducing (worsening) the bug this
# exists to fix. If a future probe run reports a smaller minimum positive
# gap, re-derive it (min_gap / 10) instead of hardcoding.
# Sanity: grid 2.76e-09 >> float noise 1e-11 (276x), and << min gap
# 2.76e-08 (10x) -- strictly inside the required window.
_EVID_TIE_GRID: float = 2.76e-09

# [Bundle A] Per-call MCS memo cap (discriminative_core scope). 1.26M
# submissions/tick collapse to ~196 unique pairs; 100k entries bounds the
# memo far above any single-tick unique set while capping memory.
MCS_MEMO_MAX_ENTRIES = 100_000
# [Bundle B0] Three-tier classification prefilter master switch. Verification
# toggle: running with False reproduces the unfiltered baseline for core
# library identity comparison.
B0_PREFILTER_ENABLED = False

# [B3/§5.8.2] Observation-window ring buffer (discriminative trigger).
# Capacity must admit θ_instances distinct positives with headroom for
# diversity; max age prevents fossil windows from ever firing the trigger
# (K2). Windows are content-deduplicated, so capacity bounds STRUCTURALLY
# DISTINCT situations per label, not raw steps.
LABEL_WINDOW_CAPACITY = 8
LABEL_MAX_AGE_STEPS = 1000
LABEL_MIN_INSTANCES = THETA_CATEGORY_INSTANCES

# [Batch D3] Category-producer unification. §5.8.2's diversity condition is
# no longer a hard TRIGGER predicate: the ObservationWindowBuffer's own
# diversity gate duplicated THETA_CATEGORY_DIVERSITY and, because buffered
# positives are content-deduplicated windows that frequently sit in one
# spatial region, it silently blocked every category core on structurally
# uniform label windows (the gate could never be satisfied for the very
# labels the §5.8.2 trigger exists to consolidate).
#
# Diversity now SOFTENS the new core's initial reliability instead:
#     factor      = clamp(div(W+) / TARGET_DIVERSITY, MIN_RELIABILITY_FACTOR, 1)
#     reliability = KAPPA_INIT * factor
# so a uniform positive set still produces a core, but one that starts
# with MIN_RELIABILITY_FACTOR of the reference reliability and must earn
# reliability (§5.6 κ_reliability) before it can drive the Mapper.
#
# MIN_DIVERSITY_FLOOR remains a hard floor on degenerate positive sets
# (structural spread ≈ 0), which carry no discriminating structure at all.
TARGET_DIVERSITY = 0.30
MIN_RELIABILITY_FACTOR = 0.25
MIN_DIVERSITY_FLOOR = 0.01

# §5.3.2 autonomous template materialization — reserved node-ID namespace.
# Materialized template nodes are NOT pattern nodes in the graph's store;
# they must never alias a live pattern-node ID (§1.11's single monotonic
# counter allocates upward from 0 and cannot reach this band), nor collide
# with previously materialized blocks (the consolidator-owned allocator
# hands out strictly increasing blocks from this base). Within signed
# 64-bit serialization range (§2.6 byte-level closure).
TEMPLATE_NODE_ID_BASE = 1 << 62

# =============================================================================
# 20. MAPPER
# =============================================================================
# §6.8. THETA_CONTAINMENT and MAX_SITUATION_SIZE are defined above.

H_SITUATION = 2
THETA_SITUATION_EDGE = 0.2
TAU_TRANSFER = 0.4
QUERY_CACHE_TTL = 10
# [Batch D2] Per-step cap on §6.6.2 "What is this?" queries issued by the
# Pipeline — the category-query CALLER. `Mapper.query_category` existed with
# no caller at all: CATEGORY cores were extracted (Trigger 2A and the
# label-window trigger) but never queried, so their reliability was never
# exercised and `category_labels_returned` could only ever be 0. The cap
# bounds per-step caller cost (each query runs a §6.4 identification over
# the situation subgraph) while covering every informative label in use
# (≤ 4 distinct effect atoms are simultaneously present on the environments
# in use). Deterministic order: labels are sorted by serialize_value.
K_CATEGORY_QUERY_MAX = 4
THETA_CATEGORY_MIN_POSITIVES = 3

# [DEPRECATED in Step 6; will be deleted in Step 8] Legacy MCS thresholds
THETA_MCS_BY_FAMILY = {
    "EVENT_CAUSAL": 0.6,
    "TEMPORAL_CHAIN": 0.7,
    "SPATIAL_OBJECT": 0.5,
    "RECURSIVE_COMPOSITION": 0.6,
    "CROSS_DOMAIN_PATTERN": 0.5,
    "CHUNK": 0.5,
}
THETA_MCS_DEFAULT = 0.5
THETA_MCS_CATEGORY = 0.6

# [Step 6a] Evidence readout thresholds
# DEPRECATED (Stage 2): Superseded by EVIDENCE_LAMBDA_NODE + EVIDENCE_LAMBDA_EDGE.
# Retained for backward compatibility with external probe scripts.
EVIDENCE_LAMBDA_ABSENCE: float = float(
    os.environ.get("EVIDENCE_LAMBDA_ABSENCE", "0.0")
)
if EVIDENCE_LAMBDA_ABSENCE < 0.0:
    raise ValueError("EVIDENCE_LAMBDA_ABSENCE must be >= 0")

THETA_EVIDENCE_POS: float = float(
    os.environ.get("THETA_EVIDENCE_POS", "0.35")
)
THETA_EVIDENCE_NEG: float = float(
    os.environ.get("THETA_EVIDENCE_NEG", "0.60")
)
EVIDENCE_GAP_THRESHOLD: float = float(
    os.environ.get("EVIDENCE_GAP_THRESHOLD", "0.05")
)
EVIDENCE_MIN_BALANCED: float = float(
    os.environ.get("EVIDENCE_MIN_BALANCED", "0.15")
)

THETA_EVIDENCE_BY_FAMILY: Dict[str, float] = {
    "EVENT_CAUSAL": 0.30,
    "TEMPORAL_CHAIN": 0.35,
    "SPATIAL_OBJECT": 0.25,
    "RECURSIVE_COMPOSITION": 0.30,
    "CROSS_DOMAIN_PATTERN": 0.25,
    "CHUNK": 0.25,
}
THETA_EVIDENCE_DEFAULT: float = 0.25
THETA_EVIDENCE_CATEGORY: float = 0.35

# Phase 2/4 — identity tier.
# Strong: score >= (family_threshold + THETA_IDENTIFY_DELTA).
# Weak:   passes match_all's family filter but < strong.
# Delta = 0.0 disables the tier (every match strong); 4-mode
# classification degenerates to FIRED-only. Useful diagnostic fallback
# if the coverage gate proves too aggressive on a given benchmark.
THETA_IDENTIFY_DELTA = 0.1

# Replaces THETA_FIRED_COVERAGE as the trivial-match gate. A match
# must have BOTH:
#   (a) at least MIN_MATCHED_ELEMENTS matched elements
#       (matched nodes + matched edges), and
#   (b) at least MIN_MATCHED_EDGES matched edges.
# Query-size-independent. SC11's query-relative coverage gate blocked
# legitimate small-core matches: 2-node/5-edge cores scored 0.714 on
# 30-node queries with coverage 2/30 = 0.067, below the 0.15 gate.
#
# Audit C2: matched_elements >= 4 alone admits 4-node/0-edge
# co-occurrence matches — the exact trivial class SC11 blocked, just
# larger. The edge companion closes that hole without affecting the
# rooms target (2 nodes / 3 edges = 5 elements / 3 edges).
# DEPRECATED (H.4b.7): superseded by MIN_MATCHED_ELEMENTS_PCT and
# MIN_MATCHED_EDGES_PCT. Retained for external reference and for mapper.py's
# import-time range guards; no longer consumed by the identity gates in
# query_identification.
MIN_MATCHED_ELEMENTS = 4
MIN_MATCHED_EDGES = 2

# [DEPRECATED in Step 8: superseded by evidence readout in query_identification]
# [H.4b.7] Percentage-based identity gates. The old absolute gates were
# calibrated for Hungarian's denser correspondences. The sparse greedy
# matcher can produce fewer matched pairs on large cores, so an absolute
# integer gate falsely rejects valid matches.
#
# FLOOR = 0 for edges: edgeless cores/queries must not permanently fail
# identity. The gate reads: matched_edges >= max(FLOOR, int(PCT * min_size)).
MIN_MATCHED_ELEMENTS_PCT = 0.15
MIN_MATCHED_ELEMENTS_FLOOR = 2
MIN_MATCHED_EDGES_PCT = 0.10
MIN_MATCHED_EDGES_FLOOR = 0


# DEPRECATED. Retained for external reference. No longer used as a
# gate; coverage is computed per-match as a diagnostic scalar
# (`identity_coverage_max` on the Mapper, cumulative across episodes).
THETA_FIRED_COVERAGE = 0.15

# Per-bucket cap on Identity objects. match_all sorts by score
# descending; the top-N per bucket survive. Dropped matches increment
# `identities_capped_this_call`.
MAX_IDENTITIES_PER_BUCKET = 20

# Phase 3 (opt-in; not wired by default).
H_FIRE = 2
LAMBDA_ACT = 0.7
THETA_SITUATION = 0.2

# [Batch E] Near-miss edge decay / recency inclusion in situation subgraphs.
# An edge below THETA_SITUATION_EDGE is admitted into the induced situation-
# subgraph edge set iff it carries recent POSITIVE evidence via one of:
#   (2) e.observed_at >= t - W_SITUATION_RECENCY and e.pending_signal > 0.0
#   (3) e.time_since_coact <= TAU_SITUATION_COACT
# The pending_signal gate on clause (2) closes audit E6-03: observed_at is
# stamped on every upsert (including signal=0.0), so without the gate a
# freshly stamped zero-signal edge would enter the situation as structure
# for ~21 steps. Clause (3)'s counter resets only on signal > 0, so it is
# positive-evidence by construction.
# Spreading weight is unaffected (RULE 3).
W_SITUATION_RECENCY = 20
TAU_SITUATION_COACT = 5

# [Phase 1 Decision 4 / P2-7] Similarity-graph parameters.
SIMILARITY_EDGE_THRESHOLD = 0.60
MAX_SIMILARITY_NEIGHBORS = 20
N_SIMILARITY_ACTIVATE = 50
K_SEED = 3
K_NEIGHBOR_SCAN = 100
MAX_HOPS = 2
THETA_TRUST_NEIGHBORHOOD = 0.85
N_FORCE_FLAT = 10
# [v13 / SC-YY] Transient anti-ossification state cap; cleared on exceed.
SIMILARITY_STATE_CAP = 500
# [v13 / SC-XX] Persistent similarity-graph cap; liveness-filtered on exceed.
SIMILARITY_EDGE_CAP = 5000

# =============================================================================
# STRUCTURAL SIMILARITY OPERATOR (§1.5, §4.12, §6.4.1)
# =============================================================================
# Weights of the closed-form structural score. The operator itself lives in
# structural_similarity.py (pure, stateless — it is NOT a binder and emits no
# edges). Same weights are used for the node and the edge dimension.

STRUCT_SIM_W_SIZE = 0.3
STRUCT_SIM_W_TYPE = 0.3
STRUCT_SIM_W_ALIGN = 0.4
STRUCT_SIM_ALPHA = 0.5
STRUCT_SIM_MIN_OVERLAP = 0.5

# Bounded fixed-point budget for organization_alignment's relational
# refinement (rounds of Hungarian re-assignment over grounded pair quality
# q = sig_sim × relational agreement; iteration exits early once the
# correspondence stabilizes, so typical cost is 2-3 rounds). Coupled
# cross-type correspondences relax roughly one type-layer per sweep, so
# the budget scales as ~2 sweeps over the lens-type count; polynomial per
# round, and multi-start selection (organization_alignment) arbitrates
# rival correspondence basins by total grounded quality.
STRUCT_ALIGN_REFINE_ROUNDS = 6

# =============================================================================
# PHASE 5 -- FUZZY GRADED MATCHER (matching/alignment.py, calibration.py,
# consensus.py, search.py, fuzzy.py)
# =============================================================================
# The robust fuzzy graded matcher. Node similarity is the population-overlap
# similarity already produced by TypedSignature.similarity; this layer adds
# global-consistency alignment (seeded from the approximate Hungarian),
# an explicit unmatched cost, symmetric coverage, systematicity weighting,
# null-calibrated mass with the 0.85 conjunction floor REMOVED, background-
# calibrated per-class scoring, multi-vote consensus with stored disagreement,
# and best-first extension search with anytime budgets.

# Use global-consistency alignment for correspondence in the live MCS path.
# DARK by default (A/B in the Phase 6 protocol): it changes every MCS score, so
# it is built and switchable now and validated before any default flip. The
# mechanism is fully live via matching/fuzzy.py regardless of this flag.
MATCH_GLOBAL_CONSISTENCY_ENABLED = False
# Explicit price of leaving a node unmatched. 0.0 = the classic "any positive
# similarity may match" behaviour; > 0 makes unmatched a priced decision.
MATCH_UNMATCHED_COST = 0.0
# Weight of the relational (systematicity) modulation of the alignment.
MATCH_SYSTEMATICITY_WEIGHT = 0.5
# Bounded fixed-point rounds for the global alignment refinement.
MATCH_ALIGN_ROUNDS = 4
# Pair-evaluation budget for the refinement. None = unbounded.
MATCH_ALIGN_MAX_PAIR_EVALS = 200000
# Weight of the relational term in the combined fuzzy score (the node term is
# 1 - this). This is a blend weight, NOT the removed 0.85 additive floor.
MATCH_EDGE_WEIGHT = 0.5
# Null-calibrated mass + floor-free conjunction in the live evidence path.
# When True, evidence_score calibrates the node coverage against the null and
# uses conjunction() with floor 0.0 (the historical 0.85 floor is retired).
MATCH_NULL_CALIBRATED = True
# Minimum support weight for a consensus edge to be kept (no invented edges).
MATCH_MIN_CONSENSUS_SUPPORT = 0.5
# Best-first extension search budgets.
MATCH_SEARCH_MAX_EXPANSIONS = 4096
MATCH_SEARCH_MAX_OPS = 200000

# =============================================================================
# LOCOMOTION BRIDGE & ANCHOR FALLBACK
# =============================================================================
# Maximum anchor distance (Euclidean, metres/cells) at which the UtilityEngine's
# targeted anchor fallback may hand the reflex a direction proposal. Beyond it
# the concept is "far away" and the AEM locomotion bridge must be used instead,
# so a distant anchor cannot masquerade as a valid terminal affordance.

ANCHOR_FALLBACK_MAX_DIST = 1.5

# =============================================================================
# 21. EXECUTOR
# =============================================================================
# §7.8.

N_EDGE = 5  # Minimum edge mass for traversability
EDGE_COST = 1.0  # Unit cost
MASS_TIEBREAKER = True  # Prefer higher total mass among equal-cost
MAX_EXPANSIONS = 10000  # A* expansion bound
THETA_TARGET_CONTAINMENT = 0.95

# =============================================================================
# 22. PREDICTION
# =============================================================================
# §8.8. W_CONFIRM, MAX_PENDING, MAX_SURPRISE_QUEUE, MAX_SURPRISE_PER_STEP,
# ETA_SURPRISE, ETA_CF_SURPRISE, THETA_PROTO_EF, TAU_PROTO_DECAY,
# N_PROTO_MAX_EBAF are defined in earlier sections.

GAMMA_CF = 0.99  # Counterfactual decay per step
RHO_SMOOTH = 0.9  # EMA for smoothed causal probabilities
N_CF_MIN = 5.0
C_HYST = 0.5

# [Batch D1] Tiered necessity commit (§8.3.3). N_cf = min(C_opp, C_opp_not)
# grows only as fast as the action is executed: a sparse/one-shot effect
# (rooms toggles fire ~26x per 2000 steps; the tool's effects are
# one-shot) never reaches N_CF_MIN = 5 within an episode, so H_nec could
# never be 1 and no binding was ever committed -- even when a single
# execution produced decisive contrast evidence. The commit point is now
# THETA_CF_COMMIT; the required contrast then ramps with
# sqrt((N_cf - THETA_CF_COMMIT) / (N_CF_MIN - THETA_CF_COMMIT)) from
# THETA_CONTRAST_COMMIT to THETA_CONTRAST_STABLE, so:
#   * 1-2 confirmations commit only against an unambiguous contrast;
#   * at N_cf >= N_CF_MIN the criterion is the historical
#     THETA_CONTRAST_STABLE reading (Gate_bind / Gate_undo still demand
#     N_cf >= N_CF_MIN, unchanged).
# Recalibrate THETA_CF_COMMIT upward if the 2000-step watch shows
# spurious commits (weak links entering the library).
THETA_CF_COMMIT = 0.50
THETA_CONTRAST_COMMIT = 0.30
THETA_CONTRAST_STABLE = 0.40

# [Batch D1b] Deferred causal-binding re-attribution (design draft §4).
#   W_REATTRIB     — steps a VOID entry (rise buffered with cid=None
#                    because no causal concept was ACTIVE to claim it)
#                    stays claimable. Measured need on tool: claim window
#                    [62, 76] = 15-17 steps; 60 caps memory with wide
#                    margin. Do NOT extend beyond 60 without a fresh
#                    measurement.
#   W_HOMED_TTL    — steps a homed-but-uncommitted triple waits for H_nec
#                    before the claim is released (its own clock, not the
#                    void clock).
#   K_REATTRIB_MAX — per-step cap on homing claims; bounds the
#                    O(voids x onsetting) scan (deterministic FIFO —
#                    oldest voids homed first).
W_REATTRIB = 60
W_HOMED_TTL = 20
K_REATTRIB_MAX = 16

KAPPA_REL = 0.05  # Transfer reliability EMA rate
W_VALIDATION = 20  # Transfer prediction horizon
W_STALE = 100  # Pending staleness pruning
THETA_SURPRISE = 0.4  # Alarm threshold

# -----------------------------------------------------------------------------
# 22.1 CAUSAL COUNTERFACTUAL HYSTERESIS GATES (§8.3.3)
# -----------------------------------------------------------------------------
# Single source of truth for the necessity-gate hysteresis constants.
# (N_CF_MIN itself is declared in Section 22 above; not duplicated here.)

BAR_DP_ENTER_THRESHOLD = 0.40
BAR_P_A_ENTER_THRESHOLD = 0.60
BAR_P_NOT_A_ENTER_THRESHOLD = 0.20
C_HYST_NECESSITY = 0.5
EPS_HYST_NEC_MIN = 0.02
EPS_HYST_NEC_MAX = 0.20

# =============================================================================
# 23. UTILITY ENGINE (v30 §2.20)
# =============================================================================
# §2.20.3 is a CLOSED FORM: U_raw = U_cap·[Σ ŵᵢ·Normᵢ + ŵ_persist·PlanPersist
# − stay_discount(a_null)] − η_conflict·Conflict. The weights below are the
# full set of constitutional utility constants. (The former "§23.1 formula
# coefficients" — reversal penalty, plan bonus, orient gain, destination
# richness, adaptive inertia — had no blueprint basis and were removed as
# drift. The stay discount is REINSTATED by auditor ruling R5 §2.4 at its
# standard v30 value/placement: it is the constitutional bias against the
# §2.6 null action, ensuring exploratory movement always beats passive
# idling when every other utility source is tied at zero.)

U_CAP = 50.0
# v30 §23.1 standard stay discount (auditor ruling R5 §2.4). Applied inside
# the U_cap bracket for exactly the constitutional no-op class (a_null:
# None/"null"/"stay", §2.6 — primitives.is_noop_action), never by env action
# names. Effective magnitude U_STAY_DISCOUNT × U_cap = 5.0 utility points,
# strictly above the maximum a targetless action can score at a zero tie
# (w6·MotorNorm·U_cap/W_total ≤ 0.4·50/4.1 ≈ 4.88): idling is a last resort
# while goal/frontier-driven utilities (≫ 0.1 in bracket terms) keep full
# weight — no hard bypass, no forced action.
U_STAY_DISCOUNT = 0.10
OMEGA_PROBE = 100.0
ETA_PLAN_PERSIST = 0.1
SIGMA_NOISE = 0.01
W_FERT = 1.0
W_ABSTRACT = 0.8
W_REAL = 0.7
W_PROTECT = 0.6
W_CONSOL = 0.5
W_MOTOR = 0.4

# v30 §2.1.12 — action selection and utilities (single source of truth).
ETA_LAYER = 0.1            # η_layer abstraction bias
M_LAYER_MAX = 2.0          # M_layer^max abstraction multiplier cap
THETA_ABSTRACT = 0.1       # θ_abstract (FertNorm floor for goal proposals)
ETA_HYSTERESIS = 0.1       # η_hysteresis plan-switch hysteresis
ETA_FATIGUE = 0.1          # η_fatigue probe fatigue weight
ETA_FATIGUE_DECAY = 0.95   # η_fatigueDecay
TAU_MEANING = 200          # τ_meaning meaningful-child window
XI_ATTEMPT = 1.0           # ξ_attempt motor attempt weight
XI_ENTROPY = 0.5           # ξ_entropy motor entropy weight
ETA_CONFLICT = 5.0         # η_conflict soft conflict penalty
TAU_PROVISIONAL = 500      # τ_provisional provisional ratification timeout
# [H.4b.6] Matcher calibration. The sparse greedy + 2-opt matcher (H.4b)
# scores systematically lower than the historical Hungarian because the
# greedy phase can leave some high-mass nodes unmatched. The following
# drop was measured by H.4b.5 on the ETH-80 9-image battery:
#
#   weighted mean drop = -0.0004  (greedy minus Hungarian; NEGATIVE means
#   the new matcher scored marginally HIGHER, not lower -- the Gestalt
#   boost is active on both arms per PF-8, and the per-lens-type matrices
#   are small enough that greedy loses almost no mass there)
#
# Template-construction thresholds (THETA_SIM, THETA_SIM_CONSENSUS) are
# recalibrated by this drop so that template nodes that would have
# entered under Hungarian still enter under greedy. At 4 decimals the
# recalibration is a no-op (0.7000 -> 0.7004); it is applied as specified
# rather than special-cased, and the stop trigger (drop > 0.20) is not met.
#
# NOTE: THETA_CONTAINMENT and THETA_MCS_BY_FAMILY (core classification
# and Mapper retrieval gates) are NOT recalibrated here. If the harness
# run after this change shows reduced full_identities_fired, those
# thresholds require a follow-up calibration.
_MATCHER_SCORE_DROP: float = -0.0004  # from H.4b.5 weighted mean drop

_OLD_THETA_SIM = 0.7
THETA_SIM = max(0.1, _OLD_THETA_SIM - _MATCHER_SCORE_DROP)   # θ_sim signature similarity threshold
THETA_EXPLORATION = 0.2    # θ_exploration frontier potential threshold
KAPPA_PROTECT = 1.0        # κ_protect consolidation protection weight
PHI_MAX = 1e6              # Φ_max structural potential cap
THETA_DOMINANCE = 0.8      # θ_dominance edge dominance threshold
N_OUTCOME_MAX = 10         # N_outcomeMax max tracked outcomes per edge
KAPPA_EMA = 0.05           # κ_ema spawn-rate EMA rate (§2.20.2)
RHO_SPAWN_STAR_DEFAULT = 0.2  # ρ_spawn* default (env-scaled per v30 §2.1.16)

# --- Frontier-gated goal proposal (cures compulsive re-visitation) ---
# [Drive Remediation Phase 0 / Strategic Fix 1] §2.20.1 goal-subset gate:
# a causal goal concept is proposed only if pursuing it can REACH
# unexplored territory (Φ > THETA_FRONTIER within FRONTIER_HORIZON AEM
# hops of one of its post/enabling states). Reachability — not the local
# post-state Φ alone — so the concept reactivates when the world changes
# (e.g. a door closes and re-opening unlocks new territory).
THETA_FRONTIER = 0.15      # Phi threshold: state counts as "unexplored territory"
# [Batch 4 / P1-2] Homeostatic lower bound for THETA_FRONTIER.
# Frozen at 0.10 after the four-way isolation run: θ = 0.10 pins to the
# safe trajectory (rooms 425). θ = 0.02 admits low-Φ goals and fails
# (rooms stalls at (7,5)). θ = 0.0734 (the live controller's equilibrium)
# also fails — the controller was overshooting into the harmful band.
# See futurework3.md, "Frontier Gate Homeostasis — Partially Resolved".
THETA_FRONTIER_FLOOR = 0.10

THETA_FRONTIER_MAX = 0.40
KAPPA_EMA_FRONTIER = 0.05      # EMA smoothing rate
KAPPA_FRONTIER = 0.05          # theta update rate
RHO_GOAL_TARGET = 0.30         # target pool occupancy
# [Patch L1a] Force-refresh period for the cached frontier-reachability
# gate: bounds staleness from Phi decay so exhausted goals cannot creep
# back in between AEM-growth invalidations.
FRONTIER_CACHE_TTL = 8     # steps before frontier cache is force-refreshed (Phi decay guard)
FRONTIER_HORIZON = 6       # max AEM hops to search for frontier from a post-state
K_PLAN_SCAN = 30           # [L2 REVERTED] trim to 15 was NOT behavior-
                           # preserving: rooms late-game has >15 gated causal
                           # concepts, so the pool content changed and the
                           # solve step moved 329 -> 583 (measured). Phase-0
                           # mandate: latency patches must not change which
                           # actions are selected. Scan depth restored; the
                           # L1 strict-topology cache carries the latency win.
                           # scan this many top-EF concepts before frontier filtering

# --- Fertility-to-locomotion inheritance ---
# [Drive Remediation Phase 0 / Strategic Fix 2] Actions that reduce the AEM
# geodesic distance toward a high-EF goal concept's enabling states earn an
# additive utility bonus (fert_norm × approach_gain × W_FERT_INHERIT).
# Replaces the reflex as the approach mechanism for conceptual goals
# (Strategic Fix 3 decouples the reflex from abstract locomotion).
W_FERT_INHERIT = 0.5       # utility weight for approaching a high-EF concept
FERT_INHERIT_TOP_K = 4     # only inherit from the K strongest active goal concepts

# v30 §2.1.9 — structural potential map (§2.19 update rule).
ETA_DECAY = 0.95           # η_decay potential decay
EPS_EDGE = 0.1             # ε_edge new confirmed edge bonus
NU_NOVEL = 0.2             # ν_novel novelty potential weight
OMEGA_FRONTIER = 0.5       # ω frontier weight
THETA_CLEANUP = 0.01       # θ_cleanup potential cleanup threshold
N_POT_MAX = 10000          # N_potMax max potential entries

# =============================================================================
# 24. INNATE REFLEX & KINEMATIC SALIENCE (12_Update §2.3–§2.5, 13_Update §6–§7)
# =============================================================================

REFLEX_ALPHA_P = 0.20
REFLEX_ETA_DEAD = 1e-6
REFLEX_EPS_M = 1e-9
REFLEX_L_REF = 1.0
REFLEX_K_DIR_MAX = 2
REFLEX_DELTA_RANK = 0.05
REFLEX_T_RETAIN = 10
REFLEX_V_BASE = 1.0
REFLEX_KAPPA_V = 0.001
REFLEX_EPS_V = 1e-9
REFLEX_V_MAX = 1e3
REFLEX_R_ENV_MIN = 1.0
REFLEX_M_EXP = 50.0
REFLEX_W_D = 0.30
REFLEX_W_V = 0.30
REFLEX_W_C = 0.25
REFLEX_W_N = 0.15
REFLEX_V0 = 0.15
REFLEX_THETA_G = 0.25
REFLEX_DELTA_HYST = 0.05
REFLEX_BETA_G = 10.0
REFLEX_GAMMA_MIN = 0.60
REFLEX_KAPPA_C = 0.10
REFLEX_T_DWELL = 2
REFLEX_BETA_VEL = 50.0
REFLEX_THETA_VEL = 0.05
REFLEX_V_THRESH = 1e-2
REFLEX_TAU_KIN_SAT = 100
REFLEX_PSI_CLIP = 0.999999
REFLEX_EPS_DEAD_MOTOR = 0.04

# =============================================================================
# 25. SATIETY & LOCKOUTS (v30 §2.1.12)
# =============================================================================

TAU_SATIETY = 30
TAU_FAIL = 20
TAU_FAIL_MAX = 200
TAU_NAV_REFRACTORY = 5
N_MOTOR_FAIL_MAX = 3

# =============================================================================
# 26. EDGE TYPE CLASSIFICATION
# =============================================================================
# §2.8 and §4.14. These frozensets are used by:
#   - binder.py: enforcement of self-loop permission (§3.4)
#   - builder.py: emission restriction (§4.14)
#   - concept_graph.py: per-type edge dispatch (§1.12)

# All edge types (§2.8). Imported from the EdgeType enum at runtime; the
# set of names is duplicated here so parameter imports don't require the
# enum module and so that freezes remain stable if the enum is extended.
EDGE_TYPE_NAMES = frozenset(
    {
        "CO_ACTIVATION",
        "TRANSITION",
        "CO_MOTION",
        "CONTRAST",
        "PERIODICITY",
        "PERSISTENCE",
        "TOPOLOGY",
        "DEGREE",
        "CONTACT",
        "SEQUENCE",
        "SYNCHRONY",
        "ORIENTATION_CONTINUITY",
        "SURFACE_GROUPING",
        "ORIENTATION_SEGMENT",
        "CURVATURE",
        "OCCLUSION",
        # [PHASE B] B.11: the six higher-tier edge types (values 21–26).
        "CONTOUR_CONTINUITY",
        "JUNCTION_CONTOUR",
        "CURVE_CONTAINMENT",
        "PART_CONTAINMENT",
        "PART_CONTENT",
        "OBJECT_PART",
        "CONTOUR_ORIENTATION",
        "JUNCTION_ORIENTATION",
        # [Pass 12] Relational edge types E1–E11
        "CO_LOCATED",
        "ORIENTATION_BEND",
        "ORIENTATION_JUNCTION",
        "AMODAL_LINK",
        "COCIRCULAR",
        "PARALLEL",
        "CO_LOCATED_CHANNEL",
        "ILLUSORY_CONTINUITY",
        "SYMMETRY",
        "PART_SEQUENCE",
        "PART_AT_ANGLE",
        "BORDER_OWNERSHIP_EDGE",
        "CURVATURE_SIGN_CONTINUITY",
        "SHARED_BOUNDARY",
    # [Pass 23]
    "CONCENTRICITY",
    "TEXTURE_BOUNDARY_EDGE",
    "SURFACE_FOLD",
    "RADIALITY",
    "SPIRALITY",
    "COLOR_OPPONENT_BORDER",
    "MOTION_OPPOSITION",
    "CROSS_SCALE_IDENTITY",
    "BINOCULAR_DISPARITY",
    "LINE_TERMINATION",
        "SKELETAL_JOINT",

    }
)

# §3.4: only these types may have parent == child (none under v31 without self-binders).
SELF_BINDER_TYPE_NAMES = frozenset()

# §4.14: candidates whose edge set is entirely composed of edges from this
# set are scaffolds, not crystallization-eligible.
NON_FAMILY_FORMING_EDGE_TYPES = frozenset(
    {
        "CO_ACTIVATION",
        "TRANSITION",
        "CONTRAST",
        "PERIODICITY",
        "PERSISTENCE",
        "SYNCHRONY",
    }
)

# §4.14: order-1 edge types eligible for family resolution.
FAMILY_FORMING_EDGE_TYPES = frozenset(
    {
        "CONTACT",
        "SEQUENCE",
        "TOPOLOGY",
        "DEGREE",
        "CO_MOTION",
        # [PHASE B] B.11: all six higher-tier types family-form. They are
        # structural containment, so a candidate carrying only these still
        # describes a real arrangement rather than scaffolding. They are
        # deliberately NOT added to NON_FAMILY_FORMING_EDGE_TYPES.
        "CONTOUR_CONTINUITY",
        "JUNCTION_CONTOUR",
        "CURVE_CONTAINMENT",
        "PART_CONTAINMENT",
        "PART_CONTENT",
        "OBJECT_PART",
        "CONTOUR_ORIENTATION",
        "JUNCTION_ORIENTATION",
        # [Pass 12] Relational edge types E1–E11
        "ORIENTATION_CONTINUITY",
        "SURFACE_GROUPING",
        "CURVATURE",
        "OCCLUSION",
        "CO_LOCATED",
        "ORIENTATION_BEND",
        "ORIENTATION_JUNCTION",
        "AMODAL_LINK",
        "COCIRCULAR",
        "PARALLEL",
        "CO_LOCATED_CHANNEL",
        "ILLUSORY_CONTINUITY",
        "SYMMETRY",
        "PART_SEQUENCE",
        "PART_AT_ANGLE",
        "BORDER_OWNERSHIP_EDGE",
        "CURVATURE_SIGN_CONTINUITY",
        "SHARED_BOUNDARY",
    # [Pass 23]
    "CONCENTRICITY",
    "TEXTURE_BOUNDARY_EDGE",
    "SURFACE_FOLD",
    "RADIALITY",
    "SPIRALITY",
    "COLOR_OPPONENT_BORDER",
    "MOTION_OPPOSITION",
    "CROSS_SCALE_IDENTITY",
    "BINOCULAR_DISPARITY",
    "LINE_TERMINATION",
        "SKELETAL_JOINT",

    }
)

# §2.3: order-2 binder edge types. The structural-similarity comparison was
# moved out of the Binder into structural_similarity.py (a pure operator that
# emits no edges), so there are NO order-2 edge types in v31. The set is kept
# as the single extension point for a future order-2 binder.
ORDER_2_EDGE_TYPES = frozenset()

# §3.3: layer computation uses edge types, not member layers.
# edge_layer(e) = 1 for order-2 types, 0 otherwise. Empty while no order-2
# edge type exists; the duplication with ORDER_2_EDGE_TYPES is intentional so
# that the layer computation and the emission restriction can be tuned
# independently if the architecture diverges.
EDGE_TYPES_WITH_LAYER_1 = frozenset()

# =============================================================================
# 27. LEGACY (v30 constants retained for backward compatibility)
# =============================================================================
# These constants are used by v30 code paths that have not yet been
# migrated. They will be deleted once v31 migration is complete.
# Values are the v30 §2.1 canonical defaults.

BFS_MAX = 10          # v30 §2.1.7 BFS_max
D_MAX = 10
C_PLAN_MAX = 100      # v30 §2.1.12 C_planMax
TAU_PLAN_MAX = 200    # v30 §2.1.12 τ_planMax
STUCK_PLAN_STEPS = 100
K_PLAN = 20
B_MAX = 3
N_SLOT_CAND_MAX = 5
LAMBDA_DIST = 0.15
PI_ABANDON_BASELINE = 0.5
PI_ABANDON_MIN = 0.1
PI_ABANDON_MAX = 1.0

# =============================================================================
# 28. 3D SENSORY ADAPTER (13_Update §3–§4) — CANONICAL VALUES
# =============================================================================
# Layer-0 preprocessor constants. These are the 13_Update specification
# values; they are scale-appropriate for metric workspaces and MUST NOT be
# re-tuned per environment (the earlier drift — R_CLUSTER 0.12, θ_depth
# 0.035/0.018, R_MIN_SEP 0.05, τ_persist 30, grid 128×96 — was tabletop
# hyper-tuning and has been reverted to spec).

ADAPTER_M_KEYPOINTS = 100        # §3.2 M ≤ 100 sparse corner keypoints
ADAPTER_R_CLUSTER = 0.20         # §3.2 R_cluster (m) common-fate co-locality
ADAPTER_EPS_VEL = 0.03           # §3.2 ε_vel (m/s) common-fate velocity coherence
ADAPTER_V_THRESH = 0.02          # §3.2 v_thresh (m/s) kinematic instantiation
ADAPTER_MIN_CLUSTER_SIZE = 4     # §3.2 |K_u| ≥ 4 keypoints
ADAPTER_R_MIN = 0.04             # §3.2 r_min (m) minimum bounding radius
ADAPTER_R_MAX_OBJ = 0.50         # §4.3 R_max_obj (m) normalization reference
ADAPTER_THETA_DEPTH_HIGH = 0.25  # §3.3 Schmitt upper threshold (m), room-scale default
ADAPTER_THETA_DEPTH_LOW = 0.12   # §3.3 Schmitt lower threshold (m), room-scale default
# ARCHITECT RULING (Item 2): θ_depth is a PHYSICAL UNIT CALIBRATION of the
# depth-discontinuity detector, not a cognitive constant — it must match the
# scene's metric scale. A 6 cm cube cannot produce a 25 cm depth step, so the
# tabletop deployment declares its own calibration through AdapterConfig.
THETA_DEPTH_HIGH_3D_TABLETOP = 0.035  # m — tabletop-scale upper threshold
THETA_DEPTH_LOW_3D_TABLETOP = 0.018   # m — tabletop-scale lower threshold
ADAPTER_R_MIN_SEP = 0.25         # §3.3 R_min_sep (m) 3D NMS merge separation
ADAPTER_EPS_GATE = 0.15          # §3.4 ε_gate (m) predictive centroid gating
ADAPTER_TAU_PERSIST = 10         # §3.4 τ_persist (frames) occlusion memory
ADAPTER_ALPHA_P = 0.20           # §6.1 α_p position low-pass coefficient
ADAPTER_BETA_T = 0.90            # §4.3 Ch3 β_T temporal-coherence EMA
ADAPTER_DEPTH_GRID_W = 64        # §3.3 decimated depth grid width
ADAPTER_DEPTH_GRID_H = 48        # §3.3 decimated depth grid height
ADAPTER_CYCLE_THRESH_PX = 1.0    # §3.2 bidirectional LK cycle-consistency (px)
ADAPTER_ANCHOR_PRIOR_MARGIN = 0.02  # margin on anchor bounding sphere (m)
ADAPTER_RGB_WINDOW_RADIUS = 2    # per-anchor mean-RGB window radius (px), v31 Color lens

# Optical-flow / corner-detector algorithm internals (implementation tuning;
# not constitutional, but centralized here per the single-source rule).
ADAPTER_LK_WIN = 21
ADAPTER_LK_LEVELS = 3
ADAPTER_LK_MAX_ITER = 30
ADAPTER_LK_EPS = 0.01
ADAPTER_ST_QUALITY = 0.01
ADAPTER_ST_MIN_DIST = 7
ADAPTER_ST_BLOCK = 3

# [PHASE F] The Size lens is PURGED: on a uniform patch grid every region
# has identical radial extent, so the lens emitted one constant bin per
# patch (a tautology) and polluted the Part signature. The lens body is
# retained in lenses.py but run_all only calls it when this flag is set.
SIZE_LENS_ENABLED: bool = False
# Size-lens normalization constants were only consumed by the purged lens;
# retained for reference (adapter radii remain the single definition).
# SIZE_RADIUS_MIN = ADAPTER_R_MIN
# SIZE_RADIUS_MAX = ADAPTER_R_MAX_OBJ

# =============================================================================
# 28.1 RECOGNITION / CATEGORIZATION (v30 §2.1.21, §2.31)
# =============================================================================
# Consumed by derived_lenses (recognition homeostasis) and concept_graph
# (disablement cooldowns). Values are the v30 canonical defaults.

THETA_CATEGORIZE = 0.7        # θ_categorize(t) initial categorization threshold
K_CATEGORIZE = 5              # K_categorize max recognition matches per step
KAPPA_CAT = 0.05              # κ_cat categorization homeostasis gain
RHO_CAT_TARGET = 0.2          # ρ_cat* target categorization rate
THETA_CATEGORIZE_MIN = 0.5
THETA_CATEGORIZE_MAX = 0.95
DELTA_THETA_CAT_MAX = 0.05    # Δθ_cat^max max threshold step
TAU_PROGRESS = 500            # τ_progress recognition progress failure window
RHO_PROGRESS_MIN = 0.1        # ρ_progress^min normalized fertility for progress
TAU_REVERT = 3                # τ_revert recognition failure streak threshold
DELTA_CAT = 0.02              # δ_cat recognition hysteresis margin

# =============================================================================
# 28.2 CONSOLIDATOR DISCRIMINATIVE MODE (v31 §5.3.2)
# =============================================================================

_OLD_THETA_SIM_CONSENSUS = 0.8
THETA_SIM_CONSENSUS = max(0.1, _OLD_THETA_SIM_CONSENSUS - _MATCHER_SCORE_DROP)     # consensus-signature graded similarity threshold

# =============================================================================
# 29. MISCELLANEOUS STRUCTURAL BOUNDS
# =============================================================================

TAU_RECOMPUTE_CADENCE = 100      # §4.7 τ recomputation cadence (steps)
TAU_ACTIVE_DRIFT_FRACTION = 0.2  # §4.7 active-set symmetric-diff drift trigger
STATE_PATTERN_INDEX_CAPACITY = MAX_EXPANSIONS  # pipeline_support StatePatternIndex;
                                 # derived: any state the Executor could expand
                                 # to (bounded by max_expansions) stays retained.
NODE_DETECTION_SWEEP_PERIOD = PATTERN_NODE_TTL * 10  # Builder node→detection map sweep
NODE_DETECTION_BLOAT_FACTOR = 2  # sweep triggers above factor × live node count
DIAG_INTERVAL = 25               # v30 §4.7 diagnostic sampling default (steps)
EBAF_TEST_MODE = False           # v30 §2.32.8 test-only credit injection hook
EBAF_T_INJECT = 1000             # injection step when EBAF_TEST_MODE is enabled
GAP_QUEUE_MAX = 100              # Mapper bounded GapSignal queue
CATEGORY_WINDOW_MAX = 10         # Trigger 2A: max positive (and negative)
                                 # situation windows fed to discriminative_core
                                 # per consolidation attempt (bounded compute;
                                 # windows arrive in temporal order)
LABEL_HISTORY_MAXLEN = 10        # Mapper per-label window history bound

# =============================================================================
# 27. BATCH D / P1-1 — ENDOGENOUS CADENCE BOUNDS
# =============================================================================
# Revision OR-bounds. graph_revision() = topology_revision + node_revision
# (ConceptGraph). Waiting-class cadences fire on steps OR graph churn;
# work-class cadences (basin recompute) are pure-revision (HIGH-03).

W_STALL_REVISION = 20
K_MAIN_REVISION = 500
TAU_EXCURSION_REVISION = 100
TAU_EPISTEMIC_REVISION = 50
K_PRUNE_FRONTIER_REVISION = 30
K_FERTILE_REVISION = 100
TAU_PROTO_DECAY_REVISION = 1000

# P1-3 credit propagation to preconditions
D_C = 0.7
MAX_HOPS_CREDIT = 3
# P1-4 partial-failure attribution
GAMMA_BLAME = 0.7
# P1-5 goal-abandon revival (graph revisions before a suppressed goal is
# reconsidered)
N_REVIVE_MIN = 5

# =============================================================================
# [V9 Track 2 — B.15] New constants for the v9 modules
#
# Applied as a DIFF, not as a block splice. Of the draft's 95 names:
#   4 collided with LIVE constants and are OMITTED here:
#       ORIENTATION_N_THETA              draft 12 vs live 24  (hash-changing:
#                                        binder.py:1007/2003/2175/2427 bin by it)
#       ORIENTATION_N_ELONG              draft  8 vs live  5  (hash-changing:
#                                        lenses.py:1297-1299 bin by it)
#       ORIENTATION_ANGLE_RELATION_KAPPA draft 2.0 vs live 1.0 (binder tau_gap)
#       ORIENTATION_ANGLE_RELATION_SIGMA_THETA
#                                        draft 0.15 vs live pi/12 (bend/junction
#                                        angular tolerance)
#     The draft's B.15 was written against a v9 Gabor bank that does not exist
#     in this tree. The v9 Gabor path uses GABOR_N_THETA below (12) and must
#     never shadow a live name; a helper needing a different theta count gets a
#     distinct name.
#   1 was dropped by ruling: MCS_SPECIFICITY_SCALE — B.13 forbids the
#     coverage x specificity form in the same document, and the constant's only
#     consumer was that form (tried in 619e16c, reverted in 6b18560).
#
# NAMING ADJACENCY — read before touching either constant:
#   JUNCTION_MAX_GAP (3.0) / JUNCTION_T_PERP_TOL (0.15) below are the v9
#   parameters for primitives.detect_junctions (the 5-tuple T/Y/X classifier,
#   B.1). They are NOT the tree's v9-era T-junction detection pair —
#   T_JUNCTION_MAX_GAP (6.0) / T_JUNCTION_MAX_PERP_DEV (0.5, parameters.py:312)
#   — which are read by sensory_floor._detect_t_junctions (the 4-tuple
#   detector used by binder._run_occlusion). Two functions, two schemas, two
#   thresholds. Fixing "the" junction constant without checking which detector
#   reads it is the ABSORPTION_MIN_MIN_SIM trap (B.27) again.
#
# Fingerprint expectation for this commit: EMPTY (constants only; nothing
# reads them yet).
#
# Section order follows the draft: V1 Gabor / pyramid, normalization, color,
# orientation, V2 binders, V4, IT, settle, matcher, core.
# =============================================================================

GABOR_N_THETA = 12
GABOR_N_SF = 5
ORIENTATION_RICH_MARGINAL_BINS: int = 60
MAX_M_PREV_SIZE: int = 50_000
GABOR_SF_MIN = 0.06
GABOR_SF_MAX = 0.35
GABOR_SIGMA_X_FRAC = 0.56
GABOR_ASPECT = 0.5
GABOR_KERNEL_MAX = 113

# [FIX 1 / Item round 3] Gabor kernel size from max(sigma_x, sigma_y).
# GABOR_ASPECT = 0.5 makes sigma_y = sigma_x / 0.5 = 2 * sigma_x, so the
# envelope is twice as long along the bar. The old size was based on
# 3 * sigma_x, clipping at 1.5 * sigma_y instead of 3 * sigma_y. With the
# flag on, size uses max(sigma_x, sigma_y) = sigma_y so no scale is clipped
# short along the bar. Ships DARK (off) behind a flag so old and new can be
# compared; raise to 113 so the coarsest scale (sy ~ 18.7) reaches 3*sy
# without cap clipping.
GABOR_KERNEL_SIZE_FROM_MAX_SIGMA: bool = (
    os.environ.get("GABOR_KERNEL_SIZE_FROM_MAX_SIGMA", "0") == "1"
)

# --- [B.1] Rich complex-cell signature (ORIENTATION_RICH_{nt}_{nsf}) --------
# [B.1] SIFT-style histogram clamp threshold
B1_HIST_CLAMP = 0.2
# [B.1] Minimum phase resultant length for phase-difference computation.
# Below this, the phase is noise-dominated and differences are meaningless.
B1_PHASE_R_MIN = 0.1
# [B1 FIX — AUDIT #1] Front-end uses DIVISIVE normalization, NOT theta-smoothing.
# theta smoothing with sigma=pi/4 collapses 12 orientations to ~3 channels.
# RF_FRAC: pooling covariance = RF envelope covariance * RF_FRAC**2.
#   0.3 => pool over ~30% of the RF (position tolerance without merging
#   two distinct nearby features).
GABOR_COMPLEX_POOL_RF_FRAC: float = 0.3
# SIGMA_THETA: NO orientation smoothing in the front-end (0.0 = off).
#   Cross-orientation competition is divisive (B1_NORM_*), not convolutional.
GABOR_COMPLEX_POOL_SIGMA_THETA: float = 0.0
# PHASE_ACROSS_SIGMA: phase pooling width ACROSS spatial positions (pixels),
#   used to build pool_phase for the phase-consistency block.
B1_PHASE_ACROSS_SIGMA: float = 2.0
# [AUDIT #1] Heeger semisaturation on gain-normalised energy (fixed ruler,
#   not a running max). B1_NORM_SIGMA_THETA: w(90 deg)=0.42.
B1_NORM_SIGMA: float = 0.05
B1_NORM_SIGMA_THETA: float = 1.2

FFT_KERNEL_THRESHOLD = 31
PYRAMID_N_LEVELS = 4
MIN_PYRAMID_DIM = 16
DIVISIVE_SIGMA = 0.1
DIVISIVE_POOL_RADIUS = 2.0
DIVISIVE_POOL_SIGMA = 2.0
DIVISIVE_W_SPATIAL = 0.7
DIVISIVE_W_CROSS = 0.3
# [B2b — AUDIT #2] Orientation pooling width for the lens across-anchor pool.
# Formula: w_theta = exp(-0.5*(d_theta / σ)²)  (standard Gaussian).
# With σ = 1.2: w(90°) = exp(-0.5*((π/2)/1.2)²) ≈ 0.424 — inside the
#   0.3–0.6 biological band.
DIVISIVE_SIGMA_THETA = 1.2          # orientation pooling width, radians
DIVISIVE_SIGMA_SF = 0.8             # log2 spatial-frequency width
DIVISIVE_ANISO_POOL_RADIUS = 16.0   # px; NOT DIVISIVE_POOL_RADIUS
                                    # (2.0, live, and bound as
                                    # normalize_2d's isotropic default)
COLOR_CONTRAST_RADIUS = 3
ILLUMINANT_TOP_FRAC = 0.1
LUM_SALIENCE_MULT = 0.25
CHROMA_SALIENCE_MIN = 0.10
ENERGY_MASK_MIN = 0.15
ORIENTATION_N_SF = 5
ORIENTATION_N_PHASE = 4
ORIENTATION_N_CONTRAST = 16
ORIENTATION_MIN_RESPONSE = 0.02
ORIENTATION_NEIGHBOR_MULT = 3.0
SIMPLE_CELL_GAIN = 0.8
SIMPLE_PHASE_SNR_MIN = 0.4
SIMPLE_SNR_FLOOR = 0.1
# [B.30 Tier-1 Fix A / B.15] |normalized signed phase-0 response|
# above this classifies the anchor as ON- or OFF-dominant; below, as
# balanced. Consumed by polarity_lens.PolarityLens._anchor_polarity.
POLARITY_ON_THRESHOLD = 0.35

# [B.46] Retinal ON/OFF front end (draft: retinal_frontend.py).
# Difference-of-Gaussians center/surround, then half-wave rectified
# into the ON and OFF channels the Gabor bank runs on separately.
RETINAL_CENTER_SIGMA = 1.0
RETINAL_SURROUND_MULT = 3.0

# [B.47] End-stopped (hypercomplex) cells, draft: end_stopped.py.
ENDSTOP_HALF_LENGTH_PX = 6.0
ENDSTOP_FLANK_K = 0.7
ENDSTOP_N_SAMPLES = 11
ENDSTOP_THRESHOLD = 0.05

# [B.48] Dimension-specific feature attention (Reynolds & Heeger 2009).
ATTENTION_BETA_FEAT = 0.4
TEXTURE_MIN_WINDOW = 8
EDGE_N_CONTRAST = 16
EDGE_N_SHARPNESS = 8
EDGE_N_ORIENT = 24
AF_MAX_DELTA_THETA = 0.35
AF_PERP_FRAC = 0.3
AF_LONGITUDINAL_MULT = 3.0
AF_SIGMA_THETA = 0.15
# [AF RE-DERIVATION, Round 6] The association field is MODULATORY, not
# additive: integration = driving * (1 + AF_GAIN * facil/(driving+facil)).
# Facilitation can only amplify local (driving) evidence, never create it in a
# channel with no local energy, so a cross segment bridging two structures
# gets no support in the gap. AF_GAIN bounds the maximum amplification
# (1 -> up to 2x) and is a fixed principled value, not a tuned knob.
AF_GAIN = 1.0
# [FIGURE-GROUND CUE WEIGHTS] The authoritative weight vector for the
# per-vertex BORDER_OWNERSHIP signal (perception/ownership.py
# `combine_ownership_cues`). Keys with a LOCAL per-vertex cue are consumed
# today: `convexity` (signed local turn of the boundary) and `enclosure` (the
# bounded-diffusion interior field, perception/interior_field.py). The
# region/global keys `symmetry`, `small_area`, `lower_region`, `familiarity`
# are RESERVED: they need a global shape/region decision that the
# emergent-objecthood mandate forbids, so they are named here but carry no cue
# yet (the combiner drops their weight when no cue is supplied).
BO_WEIGHTS = {"convexity": 0.30, "symmetry": 0.15, "enclosure": 0.25,
              "small_area": 0.10, "lower_region": 0.05, "familiarity": 0.15}
BO_MIN_SCORE = 0.30
# [FIGURE-GROUND FUSION] A/B switch for the cue combiner on the per-vertex
# ownership signal. True: combine convexity + enclosure + texture + colour +
# gravity through BO_WEIGHTS (perception/ownership.py). False: the legacy
# four-term weighted sum (curvature/texture/colour/gravity). Structural
# switch only; the cues themselves are unchanged.
BO_CUE_FUSION_ENABLED: bool = True
# [SIGNED EDGE VOCABULARY] A/B switch for the signed figure-ground edges:
# EDGE_BORDER_OWNERSHIP carries sign=+1 (the entity is the figure side) or
# sign=-1 (the entity is the ground side, i.e. the neighbour owns the border),
# and anti-phase OCCLUSION edges are emitted figure -> ground. False restores
# the prior unconditional sign=+1 on the ownership channel.
BO_SIGNED_EDGE_ENABLED: bool = True
# [INTERIOR FIELD] bounded-diffusion interior/exterior field
# (perception/interior_field.py). A bounded geodesic (min-plus) relaxation on a
# fixed-size working grid; deterministic, O((h+w)*hw). Enclosure is measured as
# the EXTRA path cost edges impose on escaping to the frame, so it fills
# interiors and reads 0 on a blank image. The one stated limit of the scheme:
# a boundary whose edge cost is below INTERIOR_EDGE_PENALTY is not seen as a
# barrier.
INTERIOR_EDGE_PENALTY: float = 6.0
INTERIOR_ENCLOSURE_SCALE: float = 3.0
INTERIOR_WORK_SIZE: int = 64
INTERIOR_PROBE_PX: float = 3.0
INTERIOR_FIELD_CACHE_SIZE: int = 8
JUNCTION_MAX_GAP = 3.0
JUNCTION_T_PERP_TOL = 0.15
AMODAL_MIN_CONFIDENCE = 0.4
AMODAL_MIN_RELATABILITY = 0.5
AMODAL_MAX_THETA = 0.35
AMODAL_MAX_CONN = 0.5
AMODAL_SEARCH_RADIUS = 4.0
AMODAL_SEARCH_RADIUS_MIN_PX = 3.0
COCIRCULAR_R_MIN = 5.0
COCIRCULAR_R_MAX = 200.0
COCIRCULAR_SIGMA_KAPPA = 0.15
COCIRCULAR_SEARCH_RADIUS = 8.0
COCIRCULAR_MAX_PATH_ANGLE: float = math.pi / 6.0
PARALLEL_THETA_TOL = 0.15
PARALLEL_MIN_OFFSET = 0.5
PARALLEL_MAX_OFFSET = 5.0
PARALLEL_SIGMA_THETA = 0.1
CLOSURE_Z_BOOST = 1.4
ALPHA_ATTN = 0.4

CURVATURE_N_SAMPLES = 5
POOL_SHIFT = 2.0
POOL_SCALES = 3
K_IT_SPARSE = 5
IT_OVERLAP_MAX = 0.5
SYMMETRY_RADIUS = 0.3
MIRROR_SIMILARITY = 0.85
SETTLE_MAX_ITERS = 6
SETTLE_STABILITY_EPS = 0.02
SETTLE_TOP_K = 3
BETA_TOP_DOWN = 0.4
BETA_TEMPORAL = 0.2
GAIN_THETA_REDUCTION = 0.3
HYPOTHESIS_MAX_BOXES = 8
ORIENTATION_BEND_THETA_MIN = 0.15
ORIENTATION_BEND_THETA_MAX = 0.80
ORIENTATION_BEND_THETA_CENTER = 0.45
ORIENTATION_JUNCTION_THETA_MIN = 0.80
ORIENTATION_JUNCTION_THETA_MAX = 1.571
ORIENTATION_JUNCTION_THETA_CENTER = 1.20
THETA_GIST_MIN = 0.4
ABSORPTION_MIN_SIM = 0.6
REL_AGREE_W_REF = 0.6
REL_AGREE_W_TGT = 0.4
PARTIAL_OCCLUSION_CREDIT = 0.5
T_JUNCTION_RADIUS = 2.0
CORE_MIN_TOLERANCE = 0.02
CORE_MIN_SIZE = 3
NEGATIVES_PER_MIN = 100
HYPOTHESIS_MATCH_MIN = 0.75
SALIENCY_TOP_FRAC = 0.6
SPATIAL_CELL_MIN = 4.0
MAX_REGIONS = 64
MOTION_BOUNDARY_THRESHOLD = 0.15

# --- [S3/S5] Edge parameter mixture & relational matching constants ---
EDGE_MIX_GAMMA: float = 0.98            # Per-step temporal mass decay (half-life ~34 steps)
EDGE_MIX_MERGE_TOL: float = 1.0         # Bin tolerance for prototype clustering (L-infinity)
EDGE_MIX_MAX_MODES: int = 4             # Maximum joint modes per edge mixture
EDGE_MATCH_USE_CONFIDENCE: bool = False # Match pure relational geometry during MCS
EDGE_PARAM_ONESIDED_SIM: float = 0.5    # Neutral score when exactly one edge has parameters


# --- [B19] Luminance adaptation ---------------------------------------------
# Adaptation scale as a fraction of the short image side. SCENE-SCALE ON
# PURPOSE: an object must not be its own illuminant estimate. This is the same
# lesson as the B8 Retinex fix ([AUDIT #14]) -- tying the scale to
# extraction_scale made large regions divide by themselves.
LUM_ADAPT_FRAC: float = 0.15
# Semisaturation = L0 + A * local_mean. A=1 maps local-mean luminance to 0.5.
LUM_ADAPT_A: float = 1.0
# Absolute floor so TRUE BLACK STAYS BLACK (without it, 0 maps to 0/TINY=0
# anyway, but the floor is what bounds the response slope near zero).
LUM_ADAPT_L0: float = 0.02


# --- [B2] Broad cross-orientation suppression & pool normalization ----------
LENS_POOL_NORMALIZE: bool = True
LENS_POOL_PRIOR_MASS: float = 1.0


# --- [B2c] Front-end two-component cross-orientation pool -------------------
B2_COS_ENABLED: bool = False
B2_COS_ALPHA: float = 0.5
B2_COS_SIGMA_ISO: float = 0.3
B2_COS_SIGMA_BROAD: float = 1.2
B2_COS_SIGMA_SF: float = 0.8


# --- [B18] Contrast gain control --------------------------------------------
# [B18] Noise floor, expressed as a fraction of gain-normalised energy. Below
# this the filter is responding to numerical noise, not structure. MUST be
# re-measured after B19 lands: adaptation halves the mean energy, so a floor
# calibrated on raw luminance silently discards real signal.
B18_NOISE_FLOOR: float = 0.01
B18_SIGMA0: float = 0.02         # semisaturation floor
B18_W_CENTER: float = 1.0        # weight of the (theta, sf) cross-channel pool
B18_W_SURROUND: float = 1.0      # weight of the spatial surround
# Surround sigma = RATIO x typical RF sigma, derived from the bank rather than
# hard-coded, so it tracks multi-scale banks automatically.
B18_SURROUND_RATIO: float = 3.0


# --- [D1] End-Stopped (Hypercomplex) Lens -----------------------------------
ENDSTOP_FLANK_KAPPA: float = 1.5
ENDSTOP_REL_THRESHOLD: float = 0.15
ENDSTOP_ENERGY_FRAC: float = 0.25
ENDSTOP_DOUBLE_RATIO: float = 0.6
ENDSTOP_EPS: float = 1e-6
ENDSTOP_NMS_MULT: float = 2.0
ENDSTOP_RIM_FRAC: float = 1.0


# --- [B10] Orientation Geometry (Phase & Contrast) --------------------------
ORIENTATION_PHASE_BINS: int = 24
ORIENTATION_CONTRAST_BINS: int = 8
ORIENTATION_C50: float = 0.5


# --- [B11] Motion Direction Population --------------------------------------
MOTION_DIRECTION_BINS: int = 8
MOTION_COHERENCE_BINS: int = 8
MOTION_SPEED_BINS: int = 8
MOTION_TRANSPARENCY_BINS: int = 8
MOTION_C50: float = 0.5
MOTION_RF_SIGMA_PX: float = 3.0


# --- [B20] Texture-Defined Boundary Lens ------------------------------------
TB_SAMPLE_FRAC: float = 0.5
TB_OFFSET_FRAC: float = 1.0
TB_SAMPLE_COUNT: int = 8
TB_MIN_ENERGY: float = 0.01
TB_MIN_DIST: float = 0.1

# --- [Q1] Co-circularity denominator ---------------------------------------
# Two contour peaks are co-circular enough to chain when the deviation of
# the arc through them from its own chord, d*sin(dtheta/2), stays under this
# many MEDIAN WAVELENGTHS. The bound is denominated in the pipeline's own
# local contour scale, so it carries no absolute angular threshold and is
# resolution-free.
#
# 4.0 is derived, not tuned: it is the Nyquist-style criterion that the
# arc's sagitta (kappa*d^2/8 = d*sin(dtheta/2)/4) must not exceed half a
# wavelength, i.e. d*sin(dtheta/2) <= 4*lambda. Measured on 8 real images
# this admits pairs whose implied curvature is below ~1/lambda and rejects
# pairs above it, which is the same separation an angular cap cannot make.
COCIRCULAR_WAVELENGTHS: float = 4.0

# --- [W1] Entity source for the Shape/identity lenses ----------------------
# True  -> entities come from FIELD regions built on PEAK_CHAINs
#          (perception/chain_regions.py). No segmenter, no boundary tracer.
# False -> the legacy segmenter path (observation.get_boundary_polyline).
# DEFAULT OFF. While both sources exist the flag is an explicit ARM
# parameter, never a silent switch: an A/B without it recorded is not a
# comparison, it is two different systems measured on the same day. Flip
# only through an instrumented run that stamps the arm into the artifact.
FIELD_REGION_SOURCE: bool = False
# Minimum enclosed area for a closed chain to count as a figure, expressed
# as a FRACTION OF IMAGE AREA rather than absolute pixels, so the gate is
# scale-free: the same criterion applies at 128 and 256 and at any crop.
# 1e-4 of 512^2 is ~26 px2, matching the previous absolute constant.
FIELD_REGION_MIN_AREA_FRAC: float = 1e-4


# --- [B7/B21] Junction Topology & Tolerances -------------------------------
JUNCTION_COLLINEAR_TOL: float = math.pi / 6.0  # radians (30 deg) for T/X collinearity
JUNCTION_STAR_TOL: float = math.pi / 12.0      # radians (15 deg) base tolerance for Star uniformity
JUNCTION_SATURATION_ARMS: int = 4              # Arm count where z saturates to 1.0
JUNCTION_ARM_ALIGN_TOL: float = math.pi / 6.0  # radians (30 deg) arm-orientation alignment tolerance
KAPPA_JUNCTION_RADIUS: float = 2.0


# --- [B8] Opponent Color & Retinex Constants -------------------------------
COLOR_LUM_BINS: int = 8
COLOR_RG_BINS: int = 16
COLOR_BY_BINS: int = 12
COLOR_SAT_BINS: int = 8
COLOR_HUE_BINS: int = 16  # Circular -> range 8.0

COLOR_RETINEX_SIGMA_FINE_RATIO: float = 1.0   # L/M channels (fine detail)
COLOR_RETINEX_SIGMA_COARSE_RATIO: float = 4.0 # S channels (coarse detail)

COLOR_LOG_OPPONENT_ALPHA_LM: float = 1.5
COLOR_LOG_OPPONENT_ALPHA_S: float = 2.5 * 1.8  # = 4.5

COLOR_BY_WEIGHT: float = 1.8
COLOR_SAT_SCALE: float = 3.0

COLOR_ILLUM_FRAC: float = 0.25
COLOR_ACHROMATIC_EPS: float = 0.02


# --- [B13] CurveComplex Enrichment --------------------------------------
CURVE_COMPLEX_PARALLEL_TOL_RAD: float = math.pi / 12.0
CURVE_COMPLEX_TEMPLATE_BINS: int = 8
KAPPA_CURVE_COMPLEX_RADIUS: float = 4.0

# --- [B14] RegionShape (V4/IT Boundary Conformation) --------------------
REGION_SHAPE_CURV_BINS: int = 16
REGION_SHAPE_HIST_LEVELS: int = 8
REGION_SHAPE_AREA_BINS: int = 8
REGION_SHAPE_ASPECT_BINS: int = 8

# --- [B22] Contour self-symmetry ----------------------------------------
SYM_RESAMPLE_N: int = 48
SYM_N_AXES: int = 12
SYM_TAU: float = 0.20
SYM_FOLD_THRESHOLD: float = 0.80
SYM_ISOTROPIC_SPREAD: float = 0.05
SYM_AXIS_UNDEFINED_BIN: int = 12

# --- [B16] ObjectShape Enrichment (aIT Gist/Symmetry) -------------------
OBJECT_SYMMETRY_AXES: int = 8
OBJECT_GIST_BINS: int = 16
OBJECT_GIST_RING_BINS: int = 8

# --- [B17] Illusory Contours & Modal Completion -------------------------
KAPPA_ILLUSORY_MAX_GAP: float = 6.0
ILLUSORY_ANGLE_TOL_RAD: float = math.pi / 12.0
ILLUSORY_MAX_CANDIDATES: int = 1000


# =============================================================================
# STAGE 2: CONJUNCTIVE RELATIONAL VERIFICATION & POLYMORPHIC SCHEMAS
# =============================================================================
# α bounds the maximum relational penalty. At 0.85, relations modulate
# the node-coverage score by at most 15%.
EVIDENCE_ALPHA_CONJUNCTION: float = float(
    os.environ.get("EVIDENCE_ALPHA_CONJUNCTION", "0.85")
)

# [PHASE 5] Null-calibrated conjunction floor for the DEFAULT evidence path.
# The historical 0.85 floor (EVIDENCE_ALPHA_CONJUNCTION) let an uninformative
# edge award 85% of its node coverage merely because both endpoints mapped.
# Phase 5 retires that floor: the relational multiplier's null is 0 (an
# uninformative edge set yields 0 graded similarity), so the default floor is
# 0.0. EVIDENCE_ALPHA_CONJUNCTION is retained for the A/B legacy path only,
# selected when MATCH_NULL_CALIBRATED is False.
EVIDENCE_CONJUNCTION_FLOOR: float = float(
    os.environ.get("EVIDENCE_CONJUNCTION_FLOOR", "0.0")
)

# [ITEM 3 / K2 + BUG-16] Scoring-facing, shipped DARK (default off),
# label UNCONFIRMED -- one object per class, views only. Category scores of
# record come from AspectGraph.category_scores (memory/aspects.py).
#   MASS_LCB_GATE       RULE K2-LCB: an element gets mass only when the
#                       one-sided 95% Wilson lower bound of its rate
#                       difference (positives vs negatives) beats chance;
#                       z = 1.645 and the sample sizes are the entire rule
#                       -- no tuned threshold. Enforced inside the ONE
#                       shared mass function (memory/distributions.py
#                       _mass_from_counts).
#   EVIDENCE_ONLY_EDGES RULE B16: cores.evidence_score counts each core
#                       edge only by its own graded evidence; unmatched or
#                       ungraded edges count 0, never the 0.85 conjunction
#                       floor.
MASS_LCB_GATE: bool = os.environ.get("MASS_LCB_GATE", "0") == "1"
EVIDENCE_ONLY_EDGES: bool = os.environ.get("EVIDENCE_ONLY_EDGES", "0") == "1"

# Cross-type polymorphic family projection penalty (10% discount)
EDGE_CROSS_FAMILY_PRIOR: float = float(
    os.environ.get("EDGE_CROSS_FAMILY_PRIOR", "0.90")
)

# Two-component structural absence (replaces scalar absence).
# λ_node penalizes unmatched core nodes; λ_edge penalizes verifiable
# core edges that fail to match in Q.
EVIDENCE_LAMBDA_NODE: float = float(
    os.environ.get("EVIDENCE_LAMBDA_NODE", "0.20")
)
EVIDENCE_LAMBDA_EDGE: float = float(
    os.environ.get("EVIDENCE_LAMBDA_EDGE", "0.15")
)


# =============================================================================
# BIOLOGICAL TWO-STREAM FRONT-END (V1 LATTICE & ASSOCIATION CONTOURS)
# =============================================================================
# [Scale-relative front end] The pixel constants below were calibrated at a
# 256x256 reference frame. _front_end_scale(h, w) returns diag/REF, and the
# PIXEL-denominated constants are multiplied by it at use time, so a
# half-resolution image (scale 0.5) uses half the pixel tolerances while the
# Gabor cycles-per-pixel band is DIVIDED by it (same physical tuning).
# Evidence: the A/B/C self-resonance experiment (diagnostics/stage5_gates.py
# `three`) gave resonance(A,B)=0.149 vs resonance(A,C)=0.591 on identical
# images, with mean contour count 36.8 -> 14.9 at half resolution processed
# natively. C high / B low implicates absolute-pixel thresholds, not
# information loss -- exactly what this scaling repairs. Normalised (0-1),
# angular, and count-based constants are unaffected.
FRONT_END_REF_DIAGONAL_PX: float = 362.0386719678744  # sqrt(256^2 + 256^2)
RETINOTOPIC_GRID_DENSITY: int = 16
RETINOTOPIC_RF_OVERLAP: float = 1.6
CONTOUR_SIGMA_THETA: float = 0.45
CONTOUR_SIGMA_DIST: float = 3.5
CONTOUR_COS_POWER: float = 2.0
CONTOUR_STRENGTH_THRESHOLD: float = 0.12
CONTOUR_MIN_LENGTH: int = 12
CONTOUR_MAX_TRACING_STEPS: int = 350
CONTOUR_MAX_SEEDS: int = 180
# [Front-end fix 2026-10-10] The reflect-padded convolution of the LOWEST-SF
# Gabor kernel manufactures spurious energy ridges at ~3*sigma_x inside the
# frame border. A global-max normalisation then lets those artifacts set the
# reference, diluting real objects below CONTOUR_STRENGTH_THRESHOLD: on the
# synthetic zoom sweep a disc at r<=0.20 of a 256^2 frame produced ZERO
# contours. Normalise against this percentile of the INTERIOR (border margin
# = the lowest-SF kernel's support radius) instead.
CONTOUR_INTERIOR_PERCENTILE: float = 99.5
# Absolute floor for the interior-percentile reference. A percentile alone has
# no notion of "is there any structure at all": on a uniform field it happily
# normalises numerical noise into ridges (measured: a blank 256^2 frame gives
# interior p99.5 = 6.9e-07 yet produced 180 contours). The separation from real
# structure is ~6 orders of magnitude (hardest small object: 2.75e-01), so any
# value between ~1e-5 and ~1e-2 is safe; 1e-3 is used.
CONTOUR_MIN_ABS_STRENGTH: float = 1e-3


# --- recovered from the original parameters.py (audit repair) ---
_OLD_THETA_SIM = 0.7


# --- recovered from the original parameters.py (audit repair) ---
_OLD_THETA_SIM_CONSENSUS = 0.8

# [Phase 2 / contour field] V1/V2/V4 field parameters.
# Declared here so they are auditable and tunable rather than buried as
# getattr defaults inside perception/contour_field.py.
PEAK_RIDGE_SPACING_FRAC: float = 1.0
# Relative peak gate (fraction of the summed-integration maximum).
PEAK_MIN_INTEGRATION_FRAC: float = 0.15
# Absolute peak gate. A relative threshold alone normalises float noise
# against itself and lets a structure-free frame through; this is the
# gate-1 analogue of CONTOUR_MIN_ABS_STRENGTH.
# Re-tuned after mass normalisation (work/contour_field fixes 1-6).
# [AF RE-DERIVATION, Round 6] Re-measured after the cocircular association
# field landed (it raises the field slightly). Synthetic corpus, 256^2:
# blank grey interior max 8e-4, blank white interior max 1.6e-3, structured
# (arc+bar) interior max 1.99e1. The floor is set near the blank/real
# geometric mean (sqrt(1.6e-3 * 1.99e1) ~ 0.18): strict on structure-free
# frames, inert on real frames (which are carried by the relative term).
PEAK_MIN_INTEGRATION_ABS: float = 0.18
COCIRCULAR_WIDE_SIGMA_FRAC: float = 2.0
COCIRCULAR_WEIGHT: float = 0.25
ENDSTOP_MAX_RADIUS_FRAC: float = 2.0
CURVATURE_MAX_RADIUS_FRAC: float = 2.0

# --- [A5] Peak & chain anchor base ------------------------------------------
# Anchors for the field pipeline are computed from position in the field's
# peak list, so any lens can reference a peak without a lookup table.
# Well above any pipeline-internal anchor counter: the region allocator and
# the binder's synthetic anchors (hash_64 values) live in the tens of
# millions at the very highest image counts we have observed, so 1e9 keeps
# the two spaces disjoint by a wide margin.
PEAK_ANCHOR_BASE: int = 1_000_000_000
# Chain anchors are offset from the peak base by this much so that a chain
# ordinal can never collide with a peak index (peak indices are bounded by
# the frame's peak count, far below this).
CHAIN_ANCHOR_OFFSET: int = 100_000
# [Phase 1 / singular points] Corner anchors form a third disjoint band above
# the peak and chain bands so a corner ordinal cannot collide with either.
CORNER_ANCHOR_OFFSET: int = 200_000
# Minimum boundary turn for a chain vertex to be a corner rather than smooth
# curvature (radians). ~34 deg: a structural sharp/smooth separation, not a
# tuned detection threshold -- a right-angle corner is pi/2 (~1.57), a gentle
# bend is well under this.
CORNER_MIN_TURN: float = 0.60
