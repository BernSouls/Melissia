# Item A — vision safeguard tests

## Scope and checks

- `tests/test_item_a_gabor.py` checks that bars at 12 known orientations select the corresponding Gabor channel.
- `tests/test_item_a_directed_edge.py` checks that directed edge types retain `(src, dst, edge_type)` in `_canonical_edge_key` rather than being endpoint-sorted.
- The existing kernel uses the corrected non-mirrored orientation convention; the directed-key implementation was already correct, so this item adds regression coverage rather than changing scoring behavior.

## Result

`pytest -q`: **4 passed** (including the new compatibility-shim regression test added while bringing the ETH-80 harness online). No golden fixture exists in this checkout; repeated harness `--help` output and the split enumerator were byte-for-byte/deterministically stable.
