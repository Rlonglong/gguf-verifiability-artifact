# EXP19 — adaptive placement of one-step flips vs the detection floor

§M3 (EXP11) evaluates only *uniformly placed* one-step code flips and flags, as open, an adversary who
*places* flips adaptively — e.g. in small-scale blocks — to minimize |Δw|/σ and slip under the tail
statistic. This measures that adversary directly, in weight space, with no behavioral attack.

## Method (`scripts/adaptive_placement.py`, 1.5B Q4_K, τ=0.1)

A one-step flip moves a weight by its block's dequantized step. We compute that step for every Q4_K weight
(dequantizing with all codes shifted +1), giving each weight a stealth score |Δw|/σ. For a budget of flips
we compare **uniform-random** placement to **stealth-optimal** placement (the weights with the smallest
|Δw|/σ), reporting for each: how many flips exceed τ, the pooled model-level contribution, the max
statistic, and the relative L1 weight change (the behavioral cost). Detector reference (EXP11): pooled
threshold 0.19%, max threshold 2.36.

## Result (`data/adaptive_placement.json`)

Only **0.43%** of the ~1.1B Q4_K weights have a one-step move below τ (the "stealth reservoir", ≈4.7M weights).

| budget (flips) | uniform: hidden / pooled+ / max / L1 | adaptive: hidden / pooled+ / max / L1 |
|---|---|---|
| 0.002% | 103/22k / +0.002% / **2.44** / 7e-6 | **all** / +0.000% / **0.04** / 9e-7 |
| 0.02% | 925/222k / +0.020% / **3.15** / 7e-5 | **all** / +0.000% / **0.05** / 1e-5 |
| 0.05% | 2.3k/556k / +0.050% / **2.75** / 2e-4 | **all** / +0.000% / **0.06** / 4e-5 |
| 0.2% | 9.4k/2.2M / +0.199% / **3.62** / 7e-4 | **all** / +0.000% / **0.08** / 2e-4 |
| 0.5% | 23k/5.6M / +0.498% / **7.07** / 2e-3 | 4.7M/5.6M / +0.074% / **0.10** / 6e-4 |

- **Uniform placement is caught**: the max statistic exceeds the 2.36 honest ceiling at every budget
  (2.4–7.1), because a random flip usually lands on a large-step weight.
- **Adaptive placement evades all three statistics up to the reservoir** (~0.43% of weights, ≈4.7M):
  every hidden flip has |Δw|/σ ≤ 0.12, the pooled tail stays ~0, and the max never reaches 2.36. This is a
  **genuine blind spot** of the tail/max detector to adaptively-placed flips.
- **But it is bounded and low-leverage**: the hidden edits are by construction the lowest-magnitude ones —
  the adaptive attack's total weight change is **about 4–8× smaller** (3.6–7.9× within the reservoir) than uniform placement of the same flip
  count, and the reservoir caps the number that can hide. Beyond ~0.43% of weights the adversary must spend
  visible flips.

## Interpretation

The detector does **not** catch adaptively-placed flips confined to small-step blocks — a real limitation to
state — but the footprint that hides is bounded (the sub-τ reservoir) and consists of the least-impactful
weights. Together with EXP17 (scale tampering, caught before it is harmful), two of the three adaptive
variants M3 flagged are now measured. The **targeted** case is measured in EXP20: a targeted sub-τ edit to one embedding row flips a greedy prediction while evading all three statistics.

## Files
- `scripts/adaptive_placement.py` — per-weight step, uniform vs stealth-optimal placement, detector stats.
- `data/adaptive_placement.json`.
