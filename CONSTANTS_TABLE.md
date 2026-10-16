# Live constants table (Item A)

| Constant | Value | Rule that set it |
|---|---|---|
| GABOR_N_THETA | 12 | channels at 0..11 * pi/12; bars at 12 known angles pick matching channel (mirror bug fix) |
| CHAIN_ENDSTOP_MAX | 0.60 | smallest 0.05-grid value keeping >=99% of true adjacent pairs under matched-sf reading |
| LINK_SUPPORT_MIN | 0.65 | smallest 0.05-grid value admitting 0 cross pairs (cross-link discriminator) |
| MASS_LCB_GATE | False (dark) | RULE K2-LCB Wilson LCB; default off, UNCONFIRMED |
| EVIDENCE_ONLY_EDGES | False (dark) | RULE B16 no 0.85 floor; default off, UNCONFIRMED |
