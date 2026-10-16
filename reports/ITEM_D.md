# Item D — join the aspect and core-evidence scorers

## Evidence from the current tree

- `AspectGraph.category_scores()` returns descriptor/prototype scores, but a repository search finds no production call site.
- The pipeline's category-recognition path calls `CategoryManager.recognize()`, which calls `Mapper.query_category()` on a `Subgraph`; that scorer returns core-evidence scores.
- `ViewRecord` stores the aspect descriptor and metadata, not a corresponding per-view core `Subgraph`. The harness query path also passes only a window subgraph to `Mapper`.
- `ASPECT_EVIDENCE_SCORING=False` exists in `parameters/_text.py`, but no runtime code reads it. It is a dark placeholder, not a functional scorer join.

## Disposition

No scoring fusion was enabled or presented as complete. Joining these outputs safely requires an explicit query-time bridge and a pre-registered normalization/weighting rule; choosing a blend after seeing the view-only accuracy would violate the no-score-tuning instruction. Keep the flag dark and label the proposed change **UNCONFIRMED — one object per class, views only**.

## Blocker / next acceptance test

Before wiring the flag, specify the two score scales and fixed fusion rule, pass the same current-view descriptor and subgraph into one ranking path, and verify deterministic top-1/top-2 ordering. Evaluate on an object-disjoint split before considering any default change. The ETH-80 download does not itself create the missing bridge or resolve the score-scale design.
