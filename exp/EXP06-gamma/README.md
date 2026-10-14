# EXP06 — Overlap between natural and nibble-flip distributions

`scripts/compute_gamma.py natural.json qcb.json` compares the two |Δw|/σ histograms (same bin edges) and
reports the overlap coefficient Σ_b min(p_nat, p_tamper) together with per-weight threshold rates. The
overlap compares magnitudes of changed weights, not how many weights changed.

**Results.** Overlap 0.076 (Q4_K, 1.5B), 0.053 (Q5_K, 1.5B; recomputed after the EXP05 correction);
0.036 at 0.5B (Q4_K; `exp/EXP07-real-attack/data/gamma_synth_vs_natural_0p5b.json`).

Data: `data/gamma_Q4_K_1p5B.json`, `data/gamma_Q5_K_1p5B.json`.
