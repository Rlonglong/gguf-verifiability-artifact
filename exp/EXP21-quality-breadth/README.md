# EXP21 — quality cost of `-ffp-contract=off` across quant types (x86 corroboration)

Extends EXP10 (which measured the flag's perplexity cost for one type, Q4_K_M, on arm64: ΔPPL +0.06%,
within paired SE of zero) across four types. Run on **x86** (Qwen2.5-1.5B-Instruct f16, `b10448`,
wikitext-2, `-c 512 --chunks 200`), quantizing with the `default` and `-ffp-contract=off` builds.

| type | PPL default | PPL ffp-off | ΔPPL | Δ% |
|---|---|---|---|---|
| Q4_K_M | 10.7564 | 10.7564 | 0.0000 | 0.000 |
| Q5_K_M | 10.3858 | 10.3858 | 0.0000 | 0.000 |
| Q6_K | 10.3025 | 10.3025 | 0.0000 | 0.000 |
| IQ4_XS | 10.8584 | 10.8584 | 0.0000 | 0.000 |

**Interpretation (honest scope).** On x86 the default build is *already* uncontracted (baseline x86-64 has no
FMA; EXP13 confirms `default` and `-ffp-contract=off` are byte-identical for Q4_K_M), so the flag is a **no-op**
here and the zero ΔPPL is expected — this corroborates the arm64/x86 asymmetry rather than broadening the
arm64 quality-cost figure. The meaningful quality cost of the flag is the arm64 measurement (EXP10, +0.06%,
not distinguishable from zero). A per-type arm64 sweep would require the arm64 host and is left to EXP10's
single point plus this x86 no-op confirmation.
