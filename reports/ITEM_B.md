# Item B — core funnel and empty-core diagnosis

## Corpus and fixed run protocol

The full corpus was downloaded and verified; see [`ETH80_DATA.md`](ETH80_DATA.md). The existing harness has three fixed labels (`apple`, `car`, `cow`), so this end-to-end run used those labels at the existing defaults: 8 train / 3 test, 512×512, grid 16, 4 steps, `SIMILARITY_MODE=linear`, `DUAL_FILTER_ENABLED=0`, and `MCS_BUDGET_PER_STEP=1000`. The sorted paths are all views of instance 1 for each label. **Any accuracy below is one object per class, views only.** No thresholds or image preprocessing settings were changed.

## Funnel (cumulative, all three consolidations)

| Stage | Kind | Tier 0 | Tier 1 | Tier 2 | Tier 3 | Unknown | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| In | Nodes | 48 | 0 | 0 | 0 | 1,310 | 1,358 |
| Mass | Nodes | 30 | 0 | 0 | 0 | 811 | 841 |
| Top-K | Nodes | 13 | 0 | 0 | 0 | 287 | 300 |
| Final | Nodes | 13 | 0 | 0 | 0 | 287 | 300 |
| In | Edges | 1,366 | 1,192 | 0 | 0 | 0 | 2,558 |
| Mass | Edges | 762 | 205 | 0 | 0 | 0 | 967 |
| Top-K | Edges | 300 | 0 | 0 | 0 | 0 | 300 |
| Final | Edges | 44 | 0 | 0 | 0 | 0 | 44 |

The diagnostic counters explicitly report `discriminative_core_empty_before_filter=0` and `discriminative_core_empty_after_filter=0`. All three labels produced entries: apple 101 nodes / 25 edges, car 101 / 7, cow 101 / 12. Reliability was 0.400 for each.

**Correction to the earlier diagnosis:** this run does not reproduce empty cores. The earlier claim that exact full-signature counts made the cores empty (and that this path used an LLR) is unsupported here. The recorded run had `DUAL_FILTER_ENABLED=0`, so it used the non-dual positive-minus-negative presence-rate path, not the graded/LLR fallback.

## What does fail

All three extractions ended by a truncated extension search, not a classified exit: `dc_break_truncated=3`, `dc_exit_classified=0`, and `discriminative_extracted=3` (unvalidated fallback entries). MCS diagnostics show 8,950 submitted, 3,027 evaluated, and 5,923 denied against the fixed per-extraction budget of 1,000. Per-class consolidation took 367.0 s (apple), 551.1 s (car), and 470.4 s (cow); the extension loop skipped 1,548 / 2,514 / 1,861 trials for budget denial. Thus the measured defect is budget-truncated, low-reliability cores—not a zero-node result.

The tier table also leaves 1,310 input nodes (and 811 post-mass) in `unknown`. The current `lens_tier` taxonomy omits several emitted higher-lens types. This is an instrumentation classification gap; no guessed tier reassignment was made.

## Held-out read-only measurement

| Truth | Correct / test views |
|---|---:|
| apple | 3 / 3 |
| car | 0 / 3 |
| cow | 1 / 3 |
| **Overall** | **4 / 9 = 0.4444 (one object per class, views only)** |

The read-only gate margin averaged 0.000000 at the harness's six-decimal print precision; it reported margin `> 0` on 9/9 rows, so the positive margins were below that display precision. The later mutating pass reported 6/9 argmax (**one object per class, views only; post-mutation**) and 9/9 label presence; it is not the gate result.

## Disposition

Do not tune content-bin thresholds to improve these scores. The data falsifies the previous “all cores vanish” premise for this fixed protocol. The immediate blocker is extension-search budget exhaustion plus incomplete tier labeling. A graded one-to-one counting-with-margin replacement remains a proposal only; it was not implemented or enabled from these results.
