# EXP04 — Natural variance in dequantized weight space

**Question.** How large is benign cross-environment variation once artifacts are dequantized?

**Method.** `tools/gguf_dequant_diff.py A.gguf B.gguf` dequantizes every common quantized tensor to fp32 and
reports the fraction of differing weights (ρ), |Δw| statistics, and the distribution of |Δw|/σ
(σ = per-tensor standard deviation of the dequantized weights). Natural variance `R_nat` here is the
arm64 `default` build vs the `-ffp-contract=off` reference at `b10448`, without an imatrix (EXP01 inputs).

**Results.**
| model / type | weights touched | median abs(Δw)/σ | P99 |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct Q4_K_M | 0.22% | 0.014 | 0.26 |
| Qwen2.5-1.5B-Instruct Q5_K_M | 0.056% | 0.0017 | 0.10 |
| Qwen2.5-0.5B-Instruct Q4_K_M | 0.10% | 0.009 | 0.24 |

Data: `data/natural_*.json`, per-tensor detail `data/natural_Q4_K_M_1p5B_pertensor.csv`.
