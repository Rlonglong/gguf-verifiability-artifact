# EXP18 — second model family / model provider / release: does the constructive re-derivation generalize?

EXP13's constructive trial (card-absent coordinates close the gap) rested on one architecture family
(Qwen3.5 hybrids at release `b9222`) plus one older dense model (`Qwen2.5-7B` at `b3772`, which reproduced
only its `Q8_0`). To test whether the result is family-specific, we repeat the trial on a distinct target:

- **`bartowski/granite-4.2-3b-GGUF`** — IBM Granite 4.2 (a different model family and architecture),
  stated release **`b10603`** (newer than every release above), publishing its own bf16 GGUF, imatrix, and
  `Q4_K_M`/`Q8_0` files. Built at `b10603`, x86, `default` and `-ffp-contract=off`, `plain` (card-only) and
  `recipe` modes (`../EXP13-trial-ivr/scripts/quantize_compare.sh`).

## Result (`data/results_granite/`)

| granite-4.2-3b | card-only (x86 default) | + recipe |
|---|---|---|
| `Q4_K_M` | 263/363 | **363/363 ✓** (both build variants) |
| `Q8_0` | **363/363 ✓** | — |

Same pattern as the Qwen3.5 family, on a model from a different provider and a newer release:

- **The recipe is again a hidden coordinate**, but with *different content*: where Qwen3.5 kept the SSM
  tensors at F32, granite bumps the attention tensors (`attn_k`, `attn_q`, `attn_output`) from `Q4_K` to
  `Q5_K`/`Q6_K` — 100 of 363 tensors differ from the stock `Q4_K_M` preset. The card records only
  "`Q4_K_M`". Pinning the published per-tensor type map re-derives the file bit-for-bit.
- **`Q8_0` reproduces from the card alone** (its stock preset already matches the publisher's; unlike
  Qwen3.5, granite has no F32-kept quantizable tensors on the `Q8_0` path).
- **Both `default` and `-ffp-contract=off` reproduce on x86**, consistent with the b9222/b10448 finding
  that recent releases build the quant kernels without `-march=native`, so baseline x86-64 is uncontracted.

## Takeaway

The constructive result — the deployed file is not re-derivable from the card's named type + release +
imatrix, but becomes bit-exact once the per-tensor recipe (and, cross-architecture, the FP-contraction
setting) is supplied — holds across **three model families (two model providers, all quantized by the same publisher) at the two recent releases `b9222` and `b10603`**; at the older `b3772` only the 7B's `Q8_0` reproduces (its `Q4_K_M` does not, EXP13). The *existence* of a hidden recipe coordinate is general; its
*content* (which tensors are bumped or kept) is model-specific and therefore not guessable from the card.
