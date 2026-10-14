# EXP16 — Is the calibration input (imatrix) itself independently re-derivable?

**Question.** The paper's verification chain is provider checkpoint → fp16 → **imatrix $C$** → quantized $Q$.
EXP13 closes the fp16→$Q$ hop (given the environment and recipe) and EXP12 closes the checkpoint→fp16 hop.
This experiment tests the remaining hop: given the publisher's own fp16 and the *calibration corpus its card
links*, can a third party re-derive the published **imatrix** — the importance matrix $C$ that weights the
k-quant scale search — bit-for-bit, or at least closely enough to reproduce $Q$?

This matters because in the paper's model $C$ is an input that a manifest can bind by hash but whose
*generation* ($C=\mathcal{C}(W,D,E_C)$) is a separate transformation with its own environment $E_C$. If $C$
does not re-derive, then binding it by hash certifies *which* $C$ was used but not that an independent party
could have produced it — one more coordinate the supply chain must record.

## Inputs (all public, all disclosed by the card)

- fp16: the publisher's own `Qwen_Qwen3.5-0.8B-bf16.gguf` (the EXP13 target).
- calibration corpus: `calibration_datav5.txt`, the file the card's imatrix line links (bartowski gist
  `82ae9b52…`), sha256 `bd925aea…` (identical on both hosts). The published imatrix records its dataset as
  `/training_dir/calibration_datav5.txt` and `chunk_count=802`, `chunk_size=512`.
- published imatrix: `Qwen_Qwen3.5-0.8B-imatrix.gguf` (186 entries).

## Method

`llama-imatrix -m <fp16> -f calibration_datav5.txt -c 512` at release `b9222`, in three build configurations — arm64/clang (`default` and `-ffp-contract=off`) and x86_64/gcc (`default`) — a subset of EXP13's
environment grid. Each run emits a GGUF-format imatrix; `cmp_imatrix.py` compares it entry-by-entry to the
published one and to the other configurations, reporting: entry-set agreement, how many `.counts` match
exactly, how many `.in_sum2` (the summed squared activations) are bit-identical, and the relative-difference
distribution of both `in_sum2` and the per-column mean importance actually used by the quantizer.

## Results

- **Published vs re-derived (arm64/clang, default build).** Same **186** entries, and **all 186
  `.counts` match exactly** — the calibration corpus and its chunking (802 chunks of 512) reproduce
  perfectly. But **0/186 `.in_sum2` are bit-identical**: the summed squared activations differ, median
  relative difference **$5.5\times10^{-5}$** (max $2.2\times10^{-2}$), and likewise for the per-column mean
  importance the quantizer actually uses. So the importance matrix is a floating-point-environment-dependent
  forward pass: the *inputs* (corpus, tokenization, chunk grid) re-derive exactly, the *accumulated values*
  do not. (`data/cmp_published_vs_arm64default.json`.)
- **Does that difference matter for $Q$? Yes.** Re-quantizing the EXP13 `Q4_K_M` recipe with our re-derived
  imatrix instead of the published one changes **141/335 tensors**: on the same `arm64-default` build, pinned
  recipe, published-imatrix vs our-imatrix agree on only **194/335** tensors; and on the `-ffp-contract=off`
  build that reproduced the published file bit-for-bit *with the published imatrix* (EXP13, 335/335),
  substituting our re-derived imatrix drops it to **194/335** — no longer bit-exact
  (`data/imatrix_propagation.json`). A $5.5\times10^{-5}$ imatrix perturbation thus propagates to ~40% of
  tensors, comparable in reach to the FP-contraction effect itself, because it shifts the scale-search
  outcome in many blocks.
- **default vs `-ffp-contract=off` (same architecture): bit-identical** — all 186 `.in_sum2` match exactly
  (`cmp_default_vs_ffpoff.json`). So, unlike the k-quant scale search, the imatrix computation is *not*
  FP-contraction-sensitive; the $5.5\times10^{-5}$ divergence from the published imatrix is the *same* whether
  we build contracted or uncontracted (published vs arm64-ffpoff is also $0/186$ bit-exact, median
  $5.5\times10^{-5}$). The imatrix's environmental sensitivity is therefore a *different* axis from the
  quantization's: contraction moves $Q$ but not $C$; architecture moves $C$.
- **arm64 vs x86 (both CPU): differ**, $0/186$ bit-exact, median rel. diff $2.2\times10^{-5}$
  (`cmp_arm64_vs_x86.json`) — the imatrix accumulation is architecture-sensitive.
- **Neither CPU matches the published imatrix.** x86-CPU vs published is $0/186$, median $5.4\times10^{-5}$
  (`cmp_published_vs_x86default.json`), the same magnitude as arm64-CPU vs published ($5.5\times10^{-5}$). So
  the publisher's imatrix was produced on a path *no* CPU build reproduces — presumably an accumulation path we did not test, such as a GPU forward
  pass. The imatrix's environment $E_C$ is thus its own unrecorded coordinate (at least the architecture; the accumulation path that produced the published file was not among those we tested), distinct from the quantization's FP-contraction axis.

**Conclusion.** The imatrix $C$ was **not** independently re-derivable bit-for-bit on the CPUs we tested, and because a tiny
numerical difference in $C$ changes the quantized bytes, bit-exact reproduction of $Q$ requires the
publisher's **exact** imatrix (obtained and bound by hash), not a recomputed one. This is a third hop in the chain, and the second we find environment-sensitive, (provider checkpoint → fp16 → **imatrix** → $Q$), and it is the
empirical justification for the paper's choice to *bind* $C$ by hash rather than re-derive it: a committed
$C$ shows which imatrix was used, and — as this experiment shows — that commitment is load-bearing, since an
honest third party recomputing $C$ from the same disclosed corpus would obtain a different $C$ and hence a
different $Q$.

## Files

- `scripts/cmp_imatrix.py` — entry-by-entry imatrix comparison (needs `gguf-py` + numpy).
- `data/` — `cmp_published_vs_*.json`, `cmp_default_vs_ffpoff.json`, `cmp_arm64_vs_x86.json`, and the
  calibration-data sha. The multi-GB fp16 and the imatrix GGUFs are not redistributed.
