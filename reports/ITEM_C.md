# Item C — K2 / BUG-16 re-measurement

## Frozen rules and shipping state

- **K2-LCB** was pre-registered in `memory/distributions.py`: retain mass only when the one-sided 95% Wilson lower bound for positive prevalence exceeds the one-sided 95% Wilson upper bound for negative prevalence. `MASS_LCB_GATE=False` remains the default.
- **BUG-16 / B16** was pre-registered in `memory/cores.py`: an edge contributes only its own evidence, normalized by the number of core edges; unmatched/unverifiable/ungraded edges contribute zero. `EVIDENCE_ONLY_EDGES=False` remains the default.
- Both changes are scoring-facing and remain **dark / UNCONFIRMED — one object per class, views only**. No thresholds were swept or changed.

## K2 filter-isolation re-measurement

The existing 8-train-view / 3-test-view harness split (apple, car, cow; object 1 only) was used to build the same 8 positive and 8 negative windows per label for both settings. `MASS_LCB_GATE` was toggled live on those same windows. The harness MCS budget was set to **0 only for this filter-isolation probe**, so the mass/top-K/endpoint filter could be measured without running the expensive MCS extension/classification stage. Therefore these are **filter counts, not cores or accuracy**; `discriminative_core` returned no entry under this isolation budget.

| Label | Gate | Nodes kept | Edges kept |
|---|---|---:|---:|
| apple | off | 100 | 25 |
| apple | K2 on | 19 | 2 |
| car | off | 100 | 7 |
| car | K2 on | 24 | 2 |
| cow | off | 100 | 12 |
| cow | K2 on | 40 | 2 |
| **Total** | **off** | **300** | **44** |
| **Total** | **K2 on** | **83** | **6** |

The pre-registered confidence gate removes 217/300 (72.3%) of baseline retained nodes and 38/44 (86.4%) of retained edges in this filter-only measurement. That is substantial attrition; without a full MCS/core and held-out scoring comparison it is not evidence to enable K2.

## BUG-16 / B16 component probe

Using whole, unmodified ETH-80 situation windows (not extracted category cores), the same `evidence_score` pairs were evaluated with the flag off and on. The apple instance-1 view at `000-000` was the template; queries were apple `022-000` (same object, new view) and car `000-000` (negative class, same object count). No accuracy is inferred from these proxy pairs.

| Query | B16 off | B16 on | Change |
|---|---:|---:|---:|
| apple `022-000` vs apple `000-000` | 0.772873 | 0.236169 | −0.536704 |
| car `000-000` vs apple `000-000` | 0.822496 | 0.271273 | −0.551223 |

For the apple pair, 152 matched-edge similarity mass was divided by 571 core edges under B16; for the car pair, 163/571. The B16 score is lower for both, but this two-pair probe does not improve same-class/cross-class ordering (the raw cross-class proxy scored higher in both settings). This supports keeping the rule dark until an extracted-core, held-out comparison is available.

## Disposition

The empirical re-measurement is diagnostic only: K2 is a strong filter attrition gate; B16 removes the conjunction floor but also sharply lowers scores on these same-object and cross-class window proxies. Neither rule is promoted. A full scoring comparison would require the MCS-backed category-core path, whose baseline calls took 367–551 seconds per label and were budget-truncated; the filter-only experiment deliberately did not claim that result.
