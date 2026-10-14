# EXP08 — Tail-fraction statistic and detection floor (superseded by EXP11)

**Superseded.** The positives here are the reference with flips applied, i.e. tampering without natural drift;
EXP11 builds positives from the honest rebuild of another environment plus flips, adds model-level decisions
and a max statistic, and provides the floor numbers used in the paper. This directory is kept for transparency.

**Statistic.** Per tensor, the fraction of weights with |Δw|/σ > τ = 0.1. Negatives: natural variance
(arm64 `default` vs the `ffpoff` reference, `b10448`, no imatrix; 165–197 tensors per type). Positives: the
reference with uniformly placed ±1 nibble flips at budget f (synthesized inside
`scripts/detector_roc.py`, same fixed code-plane ranges as EXP05). ROC over the per-tensor score.

**Results** (Qwen2.5-1.5B-Instruct; budgets from 0.002% of weights upward):
| type | floor (smallest budget with AUC ≥ 0.99) | natural tail noise (median per-tensor fraction > τ) |
|---|---|---|
| Q4_K | 0.05% | 0.018% |
| Q5_K | 0.005% | 0.0004% |
| Q6_K | 0.05% | 0.0001% |
| IQ4_XS | 0.01% | 0.002% |

At the floor, TPR at FPR 1% ≥ 0.98; AUC = 1.0 at or one budget step above it. Below the floor, sparse
tampering is not separated by this statistic.

**Scope.** The negatives are contraction variance at a known release. With an unknown release the floor is
higher: scoring the published EXP09 artifact against our `b3772` rebuild as the "natural" sample
(`exp/EXP09-rederive-attempt/data/roc_Q4_K_published_vs_b3772.json`), its per-tensor tail fraction has median
0.13% (about 7× the contraction noise, above the 0.05% floor), and AUC ≥ 0.99 needs about 0.2% of weights.
Scale-level or adaptively placed tampering is not evaluated.

Data: `data/roc_{Q4_K,Q5_K,Q6_K,IQ4_XS}_1p5B.json`, `data/roc_Q4_K_sparse.json` (earlier finer Q4_K sweep,
same floor), `data/pathb_summary.csv` (`scripts/pathb_summary.py`).
