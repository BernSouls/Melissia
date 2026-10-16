# Final report

## Item A: Safeguard
- Added tests/test_item_a_gabor.py: bars at 12 known angles pick matching channel (passed).
- Added tests/test_item_a_directed_edge.py: every directed edge type returns its edge from canonical key (passed).
- Constants table: CONSTANTS_TABLE.md (CHAIN_ENDSTOP_MAX=0.60, LINK_SUPPORT_MIN=0.65, etc.).

## Item B: Why cores empty
- Cause: content node signatures (e.g. PEAK_CHAIN full 6-slot quantized) are too fine-grained to recur across views; exact presence counting yields pos_count≈1, so LLR disc≈0 and mass=0, emptying the filter. Graded fallback exists (DUAL_FILTER_ENABLED) but uses soft best-sim, not one-to-one counting with margin.
- Blocker: no cached windows / ETH-80 data in this environment to build the funnel table or calibrate quantile bins. Graded one-to-one counting with margin over negatives is the required fix but not yet implemented; blocked on data.

## Item C: K2 and BUG-16
- Flags shipped dark: MASS_LCB_GATE=False, EVIDENCE_ONLY_EDGES=False (UNCONFIRMED, one object per class, views only).
- Re-measure not run (no data). Recommendation: stay dark pending second-instance check.

## Item D: Join two scorers
- Added flag ASPECT_EVIDENCE_SCORING (default False). Design: view memory has no per-view core subgraph, so scoring a remembered view with cores.evidence_score requires a bridge from ViewRecord to Subgraph; not built. Blocker: design + bridge pending.

## Item E: Descriptor result
- Shipped by default: ASPECT_GRADED_PARTS=False (legacy exact slot-0 gate). Bootstrap over views not run (no data).
