# EXP05 — Synthetic cheating-quantizer (nibble-flip) perturbations

**Threat.** A cheating quantizer ships `Q ≠ T(X)`: scales and minima kept, some low-bit codes changed by
±1 step. `scripts/synth_qcb.py` applies uniformly placed ±1 flips to a reference GGUF (the `ffpoff` output)
and measures the per-weight |Δw|/σ after dequantization.

**Correction (2026-09-29).** Flips are restricted to the low-bit code planes (`qs` for Q4_K/Q5_K/IQ4_XS,
`ql` for Q6_K). An earlier version also edited the Q5_K/Q6_K high-bit arrays (`qh`), which changes codes
by ±16 steps; Q5_K/Q6_K data were regenerated with the fixed ranges.

**Results** (1.5B, flip fraction 1%; median abs(Δw)/σ): Q4_K 0.268 (1st percentile 0.09, P99 0.55),
Q5_K 0.129, Q6_K 0.063, IQ4_XS 0.255. The per-flip magnitude is one step of the block's scale; it does not
depend on how many weights are flipped. It is not bounded away from zero: in small-scale blocks one step
is small, so an attacker who places flips adaptively could reach smaller magnitudes (evaluated in EXP19: the sub-τ reservoir is 0.43% of the Q4_K weights).

Data: `data/qcb_{Q4_K,Q5_K,Q6_K,IQ4_XS}_1p5B.json`, `data/qcb_Q4_K_0p5B.json`.
