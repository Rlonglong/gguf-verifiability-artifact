# EXP11 — Detection floor with realistic positives, model-level decisions, and a max statistic (paper Sec. VI, Fig. 3)

Supersedes the floor numbers of EXP08. EXP08 applied the tampering flips to the reference itself, i.e. tampering
without natural drift, the shared-environment case in which comparing hashes already suffices.

**Method (`scripts/detector_realistic.py`).** For each honest pair (reference, natural rebuild from another
environment) and quant type:
- *Tail statistic:* per tensor, the fraction of weights with |Δw|/σ > τ = 0.1 against the reference.
- *Positives:* the natural rebuild plus uniformly placed one-step flips (+1, or −1 at the top code), budgets from
  0.002% to 0.5% of the type's weights, 5 seeds. Each flipped weight moves by exactly its dequantized step,
  obtained by dequantizing the tensor with every code shifted, so no approximation is involved.
- *Per-tensor floor:* the smallest budget with AUC ≥ 0.99 in every seed (negatives: the pair's natural tensors).
- *Model level:* the pooled fraction of the type's weights above τ; threshold = the largest honest value over the
  known-release pairs.
- *Sparse large edits (Q4_K, `--large`):* k ∈ {1, 5, 25, 100} weights moved to their farthest code; model max
  |Δw|/σ against the largest honest max.

**Honest Q4_K pairs (Qwen2.5-1.5B-Instruct; `data/pairs_Q4_K.csv`).**

| pair | pooled > τ | max | per-tensor floor |
|---|---|---|---|
| arm64 default vs `-ffp-contract=off`, b10448 | 0.018% | 1.50 | 0.02% |
| same, b3772 | 0.018% | 1.50 | 0.02% |
| same, b3772 with the publisher's imatrix | 0.034% | 1.84 | 0.05% |
| x86 native (AVX2+FMA) default vs reference, b3772 + imatrix | 0.131% | 2.36 | 0.1% |
| same, without imatrix | 0.192% | 2.05 | 0.1% |
| published artifact vs our b3772 rebuild (unknown environment) | 0.131% | 2.36 | 0.1% |

- Model-level rule (threshold 0.1915%): no honest artifact flagged (including the published one); every tampered
  model flagged from 0.2% of weights up, in every pair and seed.
- Max statistic: a single farthest-code edit exceeds the largest honest max (2.36) in every seed and pair
  (2.40–2.52); five edits reach ≥ 2.57.
- Other types at the known-release arm64 pair (`data/realistic_{Q5_K,Q6_K,IQ4_XS}.json`): floors 0.005% (Q5_K),
  0.05% (Q6_K), 0.005% (IQ4_XS). One budget step above every floor, TPR at FPR 1% is 1.0 in every seed.
- Sanity check: the median step/σ reproduces EXP05 (Q4_K 0.267, Q5_K 0.129, Q6_K 0.063, IQ4_XS 0.256).

**Run.** `python scripts/detector_realistic.py --pairs data/pairs_Q4_K.csv --type Q4_K --large --out out.json`
(the pair files list GGUFs produced by EXP01 `quantize_repro.sh` and EXP09 `rederive_at_release.sh`; paths use the
`$WORK` placeholder, which the script expands). About 2–3 minutes per pair on one CPU core.

## τ robustness sweep (`data/tau_sweep/`)

`detector_realistic.py --tau {0.05,0.2}` on the Q4_K pairs, alongside the default τ=0.1, shows the
separation is not an artifact of the threshold. At all three τ the pair ordering is preserved and **no
honest pair is flagged** by the model-level pooled rule; the threshold scales as expected (more benign
drift is admitted at smaller τ):

| τ | model-level pooled threshold | honest pooled range | per-tensor floor range | max threshold |
|---|---|---|---|---|
| 0.05 | 0.41% | 0.04–0.41% | 0.05–0.2% | 2.36 |
| 0.10 | 0.19% | 0.02–0.19% | 0.02–0.1% | 2.36 |
| 0.20 | 0.08% | 0.01–0.08% | 0.02–0.1% | 2.36 |

The max statistic (largest |Δw|/σ) is threshold-independent (2.36 at every τ), so the sparse-large-edit
detection does not depend on τ at all. Regenerate: `for t in 0.05 0.2; do python scripts/detector_realistic.py --pairs data/pairs_Q4_K.csv --type Q4_K --tau $t --large --out data/tau_sweep/realistic_Q4_K_tau$t.json; done`.
