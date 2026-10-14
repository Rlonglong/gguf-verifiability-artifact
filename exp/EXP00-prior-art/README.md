# EXP00 — Prior-art verification

**Question.** Do the prior-art items the study builds on exist as described, and what is the status of
the upstream determinism fix (llama.cpp PR #25353)?

**Method.** GitHub objects (ggufpacker, gguf-quant-determinism, PR #25353, Discussion #25354) via the
GitHub API; the IETF draft via datatracker; arXiv items via the arXiv API; a search for 2025–2026 work on
GGUF quantization reproducibility and quantization-time attacks.

**Result.** All 15 items verified (`data/prior_art.csv`). Key facts:
- PR #25353 (`-ffp-contract=off` for `ggml-quants.c`) was opened 2026-07-06 and is still open as of 2026-09-29.
- ggufpacker, its evidence repository, and Discussion #25354 already report that k-quant outputs diverge
  across machines, that floating-point contraction (FMA) in the scale search is the cause, and that the
  flag removes it. EXP01 reproduces this independently; the paper credits it as prior work.
- *Mind the Gap* (arXiv 2505.23786) is an adversarial-fp16 attack (the malicious behavior is in `W`,
  quantization is honest), i.e. outside what re-derivation can detect (assumption A1).

Data: `data/prior_art.csv` (manual survey; no script).
