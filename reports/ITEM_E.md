# Item E — descriptor result and PEAK_CHAIN bin calibration

## Pre-registered protocol

Registered before computing Item E descriptor or accuracy results on 2026-10-16.

- Use the downloaded ETH-80 corpus at `/home/user/ETH80` and the harness's existing deterministic split enumerator, with all eight classes. `--train 8 --test 3`, default (non-stratified) selection: the first eight and next three sorted views of object 1 in each class. This is **one object per class, views only**; it does not measure object-instance generalization.
- Extract field peaks/chains from the whole 512×512 frame at grid 16, using the existing observation/patch-extraction path for anchor positions. No custom cropping, tiling, or image-splitting code was added. Train `AspectGraph` on the eight training views/class; score only the three held-out views/class.
- Compare the shipped legacy scorer (`ASPECT_GRADED_PARTS=False`) against the already-defined candidate (`True`, fixed `ASPECT_PART_SLOTS=(0, 1, 5)`). Do not sweep slots, thresholds, or other parameters. Keep the candidate dark and labelled UNCONFIRMED regardless of the result; this single-object-per-class protocol cannot confirm object-level generalization.
- Report top-1 accuracy and per-view paired differences. Compute a class-stratified paired bootstrap with 10,000 replicates, seed `20261016`, resampling the three held-out views with replacement within each class. The primary interval is the percentile 95% CI of candidate-minus-legacy accuracy.
- For PEAK_CHAIN bin calibration, report training-only occupancy and exact six-slot recurrence. The fixed candidate rule is equal-frequency quantile cuts at 1/8 increments, pooled across classes for scalar slots 0, 2, 4, and 5; duplicate cut values are collapsed. Keep circular slot 1 and binary closure slot 3 unchanged. No held-out values enter cut selection. These cuts are a diagnostic proposal only: do not change production bins or scoring based on this view-only sample.

## Results

All 88 selected views produced descriptors: 64 training views and 24 held-out views. The two `AspectGraph` memories each held 64 views and 64 prototypes. No scoring or parameter default was changed.

| Scorer | Correct / held-out views | Top-1 accuracy |
|---|---:|---:|
| Legacy exact-slot gate | 10 / 24 | **0.4167 (one object per class, views only)** |
| Graded slots `(0, 1, 5)` | 16 / 24 | **0.6667 (one object per class, views only; UNCONFIRMED)** |

The paired point change is **+0.2500**. The registered class-stratified view bootstrap (10,000 draws, seed 20261016) gives 95% CIs of 0.2917–0.5417 for legacy accuracy and 0.5417–0.7917 for graded accuracy (**one object per class, views only**), and **+0.1250 to +0.4167** for candidate-minus-legacy accuracy (**one object per class, views only**). This resamples views, not independent objects; it is not evidence of object-level generalization.

### Training-only PEAK_CHAIN occupancy

There were 2,949 PEAK_CHAIN detections across the 64 training views. Existing-slot occupancy:

| Slot | Meaning | Occupied values | Training-bin evidence |
|---:|---|---:|---|
| 0 | peak count | 14 / 16 | bin 15: 473 / 2,949 (16.0%) |
| 1 | mean theta (circular) | 12 / 12 | all circular bins occupied |
| 2 | theta spread | 8 / 8 | bin 0: 2,094 / 2,949 (71.0%) |
| 3 | closure | 2 / 2 | closed: 35 / 2,949 (1.2%) |
| 4 | curvature summary | 3 / 8 | bin 0: 2,878 / 2,949 (97.6%); only bins 0, 1, 2 observed |
| 5 | length | 8 / 8 | bin 7: 1,029 / 2,949 (34.9%) |

Applying the pre-registered pooled training-only quantile rule gives candidate cuts **after** these source-bin values (duplicate cuts collapsed): slot 0: 2, 3, 4, 6, 9; slot 2: 0, 1, 2; slot 4: 0; slot 5: 1, 2, 3, 6. Circular slot 1 and closure slot 3 remain unchanged. These proposed cuts are diagnostic only and are not written into code.

Exact six-slot signature Jaccard was 0.0898 within-class versus 0.0803 cross-class. Per-class fractions of unique training signatures appearing in at least two training views were: apple 28.4%, car 28.8%, cow 23.3%, cup 23.0%, dog 23.1%, horse 24.4%, pear 25.0%, tomato 28.8%. This is evidence of some repeated signatures, not a wholesale recurrence failure; the within/cross difference is modest.

## Disposition

The fixed graded-parts candidate improved this view-only result, and the pre-registered bootstrap interval for the view-level difference is positive. **Do not enable it by default.** Keep `ASPECT_GRADED_PARTS=False`; candidate remains **UNCONFIRMED — one object per class, views only**. Validate on object-disjoint instances before any scoring-facing promotion. No PEAK_CHAIN bins were altered; the sparse/saturated slots and candidate quantile cuts are recorded for a later, independently validated calibration.
