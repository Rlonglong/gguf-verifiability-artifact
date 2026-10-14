# EXP20 — targeted tampering vs the detection floor (the targeted case)

§M3/EXP19 leave one adaptive variant to measure: a **targeted** tamper aimed at a behavioral objective
(cf. rounding-guided backdoor injection, QuRA). This measures it — as a **defensive** detector evaluation,
not a deployable exploit — on a benign objective: for a fixed neutral probe, flip the model's greedy next
token from its true argmax `T0` to a chosen alternative `T1`.

Qwen2.5-1.5B has **tied embeddings**, so the output logit of token `t` is exactly `⟨h, E_t⟩` for the final
hidden state `h`. Raising `logit_T1` above `logit_T0` by editing only `T1`'s embedding row is therefore an
exact, gradient-free targeted attack: moving `E_T1[j]` one quant step in direction `sign(h[j])` raises the
logit by `|h[j]|·step[j]`. A **stealth** adversary uses only edits whose one-step `|Δw|/σ ≤ τ` (invisible to
the tail statistic). We take the exact Q6_K step (`|d|·|sub-scale|` from the raw block), greedily apply the
highest-gain sub-τ edits until the margin flips, and confirm by placing the tampered embedding in the model.

## Setup

`token_embd.weight` is **Q6_K**, σ=0.026; median one-step `|Δw|/σ = 0.064`; **4.46%** of one-step edits
exceed τ=0.1 (the rest are sub-τ — the fine 6-bit embedding is mostly below the threshold). Model:
Qwen2.5-1.5B-Instruct; GGUF `Q4_K_M`; detector reference (EXP11): pooled 0.19%, max 2.36, τ=0.1.
`scripts/targeted_attack.py` (torch+transformers on GPU for `h`/confirmation; `gguf-py` for the embedding).

## Result (`data/exp20_*.json`)

| probe (T0) | target T1 | margin | sub-τ-only reachable? | edits | flips>τ | max \|Δw\|/σ |
|---|---|---|---|---|---|---|
| "capital of France" (` Paris`) | rank-1 ` located` | 1.26 | **yes** | 71 | 0 | 0.10 |
| | rank-10 ` ____` | 2.82 | **yes** (confirmed✓) | 372 | 0 | 0.10 |
| | rank-30 ` an` | 4.57 | no (one row's sub-τ budget exceeded) | — | — | — |
| "opposite of hot" (` cold`) | rank-3 ` ____` | 1.77 | **yes** (confirmed✓) | 147 | 0 | 0.10 |
| | rank-10 ` hot` | 4.28 | **yes** (confirmed✓ by an earlier script revision, 418 edits; see Data files) | 938 | 0 | 0.10 |

- **A targeted flip to a plausible alternative token hides entirely below the floor.** For both probes,
  promoting a token from rank ≤10 to the greedy output is achievable using edits **all** below τ
  (`flips>τ = 0`, `max |Δw|/σ ≤ 0.10 < 2.36`, pooled contribution ≈ `edits/1.5\times10^9 ≈ 10^{-6} ≪ 0.19\%`).
  All three detection statistics miss it. Both flips were **confirmed in-model**: with the tampered
  embedding in place, the greedy next token became the attacker's chosen `T1`.
- **Far targets leave a trace.** Flipping to a low-ranked token (margin ≳ 4.5) exceeds the sub-τ budget of a
  single embedding row, so it needs above-τ (detectable) edits or several rows.

## Interpretation

Unlike scale tampering (EXP17, caught before it is harmful) and generic adaptive placement (EXP19, a bounded
low-leverage blind spot), a **targeted** tamper on the tied output embedding **defeats the weight-space
tail/max detector** in the regime that matters — promoting a plausible alternative token — and does so with a
behaviorally-confirmed effect. This is a genuine limitation, and it sharpens the paper's central point: a
statistical detector has fundamental blind spots against an adaptive, targeted adversary, so the sound
defense is not detection but an **ambiguity-free (B=0) exact re-derivation** that makes *any* deviation a
hash mismatch (§Protocol). The characterization in §M3 bounds the *benign/uniform* regime; it is not a
security guarantee against a targeted cheating quantizer, and we say so.

## Files
- `scripts/targeted_attack.py` — exact-Q6_K-step, stealth-constrained targeted attack + in-model confirmation.
- `data/exp20_paris.json`, `data/exp20_hot.json` — produced by the shipped script (sweep over ranks 1/10/30 plus the stealth confirmation at `--confirm-rank`).
- `data/exp20_The_opposite_of_hot_is.json`, `data/exp20_Two_plus_two_equals.json` — produced by an **earlier revision** of the script (confirmation only, rank 10, which also lowered the original token's row; no `sweep` keys). The first confirms the flip to ` hot` with 418 sub-τ edits (max 0.098). The second is a **failed** confirmation: for "Two plus two equals" (T0 ` four`), 463 sub-τ edits reached the predicted margin but the forward pass did not flip (`flipped_to_target: false`), so one row's sub-τ budget is not always sufficient. The paper reports both facts.

## What "confirmed in-model" means
The confirmation is a full-precision (fp16, Hugging Face `transformers`) forward pass with the tampered, dequantized `token_embd` swapped into the model; it is not the tampered GGUF run under `llama.cpp`. The detector thresholds quoted (pooled 0.19%, max 2.36) are the Q4_K-calibrated ones from EXP11, applied to a Q6_K embedding row; the conclusion (every edit below τ, pooled contribution ≈1e-6) does not depend on that calibration. The 4.46% / 95.5% one-step-above-τ figure is printed by the script (`frac one-step>tau`) rather than saved to the JSON.
