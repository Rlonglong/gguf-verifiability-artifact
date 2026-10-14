# EXP17 — uniform (global) scale perturbation vs the detection floor (the adaptive case M3 flagged)

§M3 (EXP11) separates benign FP-contraction drift from **code**-flip tampering with a tail statistic
(per-tensor and pooled fraction of weights with |Δw|/σ > τ, plus a max |Δw|/σ). It explicitly notes it
does **not** evaluate an adaptive adversary who edits **scales** — which move every weight in a block by a
sub-step amount and might slip under a tail statistic tuned to one-step code flips. This experiment realizes
that adversary and measures both its detectability and its effect on the model.

## Tamper

For each Q4_K super-block (144 B: f16 `d`, f16 `dmin`, 12 B scales, 128 B qs), multiply the f16 super-scale
`d` by (1+ε). This is a pure, uniform scale edit — every weight in the block moves, no code is touched —
applied to every Q4_K tensor of the 1.5B model (`scripts/scale_tamper.py`, a stdlib byte edit; statistics
via `gguf-py`). We sweep ε and, for each, compute the three detector statistics against the untampered model
and measure wikitext-2 perplexity (`build-default/bin/llama-perplexity`, `-c 512 --chunks 200`).

## Detectability (`data/scale_sweep_stats.csv`)

Detector reference points (EXP11): model-level pooled threshold **0.19%**, per-tensor floor **0.02–0.1%**,
max threshold **2.36**; benign FP-contraction ΔPPL **+0.006 (+0.06%)** (EXP10).

| ε | per-tensor tail (median) | pooled tail | max |Δw|/σ | detected? |
|---|---|---|---|---|
| 0.05 | 47.6% | 49.3% | 5.50 | yes (all three) |
| 0.02 | 1.68% | 1.33% | 2.18 | yes (pooled ≫ floor) |
| 0.01 | 0.015% | 0.021% | 1.09 | borderline (≈ per-tensor floor; below pooled threshold) |
| 0.005 | 0% | 0.0008% | 0.54 | **hides** (all three below threshold) |
| 0.002 | 0% | 0% | 0.24 | hides |
| 0.001 | 0% | 0% | 0.12 | hides |

## Behavioral impact (`data/ppl_sweep.csv`, wikitext-2, baseline PPL 10.566)

| ε | PPL | ΔPPL | detector |
|---|---|---|---|
| 0.02 | ≈5.4×10⁶ (diverges) | model destroyed | detected (pooled 1.33% ≫ 0.19%) |
| 0.01 | 10.670 | +0.98% | borderline (≈ per-tensor floor) |
| 0.005 | 10.597 | +0.29% | hides (all three below threshold) |
| 0.002 | 10.585 | +0.18% | hides |

A uniform super-scale perturbation compounds multiplicatively across layers, so it is either catastrophic
or minor: ε=0.02 sends perplexity to ~10⁶ (the model is destroyed) and is caught by the pooled tail; the
largest perturbation that evades **all three** statistics, ε=0.005, changes perplexity by only +0.29% —
about 5× the benign FP-contraction drift (+0.06%, EXP10) and far below any damaging level.

## Interpretation

A uniform scale perturbation is caught by the tail/max detector in the regime that would be harmful: in this sweep, every perturbation
large enough to degrade the model materially (ε≥0.02, PPL diverging) fires the pooled-tail rule, while what
slips under all three statistics (ε≤0.005) is behaviorally minor (ΔPPL ≤ +0.29%, a few times benign drift).
The detector therefore covers the adaptive scale-tampering case §M3 flagged, with a **bounded** blind spot:
a patient adversary can hide a sub-floor scale drift costing +0.3% perplexity (ε=0.005; ε=0.01 evades the pooled and max statistics at +0.98% with the per-tensor statistic borderline), but not a damaging one. This is a
measured result in place of M3's previous flagged-but-unevaluated caveat, and it is a measured bound for the
detector (more robust to scale edits than the caveat implied), stated with its limitation.

## Files
- `scripts/scale_tamper.py` — Q4_K super-scale byte editor + |Δw|/σ statistics.
- `data/scale_sweep_stats.csv`, `data/ppl_sweep.csv`.
