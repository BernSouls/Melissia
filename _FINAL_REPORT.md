# Final report — Melissia vision Items A–E

Updated 2026-10-16 after downloading and verifying ETH-80. Detailed per-item evidence is in `reports/ITEM_A.md` through `reports/ITEM_E.md`; corpus provenance is in `reports/ETH80_DATA.md`.

## Data

- Downloaded the full ETH-80 mirror to `/home/user/ETH80` (GitHub source commit `9243466b9ca5f15765c88fdaf17c4f10c15a8574`); 3,280 RGB views + 3,280 masks, all 6,560 PNGs verified. Kept outside the Melissia Git repository because the mirror has no license file.
- Fixed the harness docstring that incorrectly said each class had only 10 photos and a maximum 7/3 split. ETH-80 is 10 objects × 41 views per category; the default 8/3 split is views of object 1 only.

## Item status

### A — Gabor and directed-edge safeguards: complete

Added Gabor-channel and directed-edge-key regression tests. The harness also exposed a missing `PeakLens` export in the legacy `derived_lenses` shim; added the export and a regression test. The suite is **4/4 passing**. See `reports/ITEM_A.md`.

### B — core funnel: measured; earlier diagnosis corrected

On the harness's fixed apple/car/cow run, 8 train / 3 test, 512×512, grid 16, 4 steps, each label produced a nonempty 101-node core. The cumulative content funnel was 1,358 nodes / 2,558 edges in → 841 / 967 after mass → 300 / 300 after top-K → 300 / 44 final. Empty-before/after-filter counters were both zero. The previous “all cores vanish because exact PEAK_CHAIN counts” explanation is **not supported by this run**.

The actual measured failure is truncated extension search: three unvalidated cores, reliability 0.400 each, 5,923 denied MCS calls, and 367–551 seconds per label. Read-only accuracy was **4/9 = 0.4444 (one object per class, views only)**, with displayed mean margin 0.000000. See `reports/ITEM_B.md`.

### C — K2 / BUG-16: re-measured diagnostically; remain dark

Using the same in-memory windows for both settings, K2's filter-only probe reduced retained elements from 300 nodes / 44 edges to 83 / 6. It used MCS budget 0 to isolate the filter, so it does not provide a core or accuracy comparison. The B16 component probe on raw view windows reduced scores sharply for both same-class and cross-class pairs; it did not improve their ordering. Both rules remain **UNCONFIRMED** and default-off. See `reports/ITEM_C.md`.

### D — scorer join: blocked with architecture evidence

`AspectGraph.category_scores()` has no production call site; the active category path calls `Mapper.query_category()` on subgraphs. `ViewRecord` does not carry a corresponding core subgraph, and `ASPECT_EVIDENCE_SCORING` is currently an unused dark placeholder. No score fusion was added without a pre-registered normalization/weighting rule and a query-time bridge. See `reports/ITEM_D.md`.

### E — descriptor/bin calibration: view-only experiment complete; candidate remains dark

On all eight classes with 8 train / 3 test views of object 1 per class, the pre-registered graded-parts candidate scored **16/24 = 0.6667 (one object per class, views only; UNCONFIRMED)** versus legacy **10/24 = 0.4167 (one object per class, views only)**. Class-stratified paired view bootstrap (10,000 replicates, seed 20261016) gave candidate-minus-legacy 95% CI **+0.1250 to +0.4167 (one object per class, views only)**. The result is not object-instance generalization; `ASPECT_GRADED_PARTS` stays `False`.

Training-only PEAK_CHAIN occupancy showed strong sparsity/saturation in slots 2, 3, 4, and 5. Candidate quantile cutpoints are documented, but no bins were changed. See `reports/ITEM_E.md`.

## Validation, bugs, and scope

- Tests: `4 passed`; repeated harness `--help` output and split enumeration deterministic; changed Python files pass AST and line-ending checks; `git diff --check` clean. No golden fixtures exist in this checkout.
- Corrected the harness's stale `--f1-falsifier` description: G.2.1 resets the MCS ledger per extraction, independent of `t`; this flag is now described as a legacy timing probe.
- The 8-class end-to-end core-harness attempt was stopped during training after 18 minutes and ~1.27 GB RSS. No result is reported from that incomplete run. The harness's existing end-to-end category list remains apple/car/cow; the separate Item E view-descriptor evaluation covers all eight classes.
- No score-tuning, image-splitting code, deletions, or anti-phase grouping work. Anti-phase remains parked. All scoring-facing candidates ship dark and labelled UNCONFIRMED.
