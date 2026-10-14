# EXP09 — Re-deriving a best-case published artifact (paper Sec. IV)

**Target.** `bartowski/Qwen2.5-1.5B-Instruct-GGUF`: publishes its f16 GGUF and imatrix, and its model card
states `Using llama.cpp release b3772 for quantization`.

**Method (`scripts/rederive_at_release.sh`).** Build `b3772` (default and `-ffp-contract=off`) on arm64/clang
and x86_64/gcc; quantize the publisher's f16 to Q4_K_M with and without the publisher's imatrix; compare
with the published Q4_K_M. Compare the **tensor-data region** (`tools/gguf_tensor_sha.py`): llama-quantize
records the local imatrix path in the metadata, so whole-file hashes differ whenever an imatrix is used.

**Results** (`data/rederive_b3772.csv`).
- Recipe and metadata reproduce (same per-tensor types; the only differing key is `quantize.imatrix.file`).
- Tensor data matches the published file in 0/4 imatrix configurations (and 0/4 without imatrix). The two
  `ffpoff` builds agree across platforms (with and without imatrix); the two `default` builds do not.
- Published vs our `ffpoff` rebuild: 2.4e-3 of tensor bytes differ, highest per-byte rates in the scale
  fields (`data/diff_published_vs_b3772_ffpoff_with.json`); our arm64 `default` and `ffpoff` builds differ
  from each other by 7.3e-4 (`data/diff_b3772_arm64_default_vs_ffpoff_with.json`). The cause of the residual
  cannot be determined from the disclosed information.
- In dequantized space (`data/dequant_published_vs_b3772_ffpoff_with.json`) the residual touches 3.0% of
  weights, median |Δw|/σ 0.004 (the histogram in that file is built from a per-tensor sample of changed
  weights, so its counts are not fractions of all weights).
- Scored with the EXP08 statistic (`scripts` of EXP08: `detector_roc.py <b3772 ffpoff+imatrix> <published> --type Q4_K`;
  `data/roc_Q4_K_published_vs_b3772.json`): median per-tensor tail fraction 0.13% (vs 0.018% for contraction
  noise), above the known-release per-tensor floor, but *not* flagged by the model-level rule calibrated across environment pairs (EXP11: threshold 0.19%; this artifact sits at 0.13%). This is the 1.5B pilot cited in §IV and Table III of the paper.

**Conversion hop.** EXP12 re-derives the publisher's f16 itself, bit-for-bit, from the provider's checkpoint,
so for this repository the failure lies in the quantization step.

**Earlier attempt (not used in the paper).** `data/rederive.csv` is an earlier attempt at the current release `b10448` that compared
whole-file hashes; that comparison cannot match (imatrix path, release) and is not used in the paper.
