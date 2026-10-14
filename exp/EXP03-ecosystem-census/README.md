# EXP03 — Ecosystem census (paper Sec. IV, Fig. 1)

**Sampling.** The 200 most-downloaded GGUF repositories from the HF API (`sort=downloads`, the trailing
30-day count), August 2026; 2 contain no GGUF weights, leaving 198. The frozen list is
`data/frozen_repos_top200.txt`; the scored snapshot is `data/census_top200.csv`.

**Results (scored with EXP02 `census.py`).**
- IVR under the paper's *recorded-inputs* protocol (all four inputs, including the build environment, must be disclosed): **0/198**; env/version record missing in 198/198, by construction of the GGUF format. Re-derivation by *searching* environments is not measured by the census; the EXP13 trial gives a lower bound of 3/198.
- Source-reconstructible: **148/198** under the census rule (the *declared base-model repository* holds public
  full-precision weights; an upper bound, since for fine-tuned derivatives that is the parent model);
  **110/198** when the repository itself must ship a full-precision GGUF (`data/census_strata.csv`).
- Gaps beyond env/version: fp16 33, imatrix 19, both 2 (50 not source-reconstructible).
- 62/198 artifacts carry `quantize.imatrix.*` keys.
- Strata by representative file (`scripts/strata.py`): 154 ggml scale-search k/i-quants, 23 Q8_0/Q4_0,
  8 f16 or vision-projector files, 13 custom or mixed-precision formats. Restricted to text and multimodal
  LLMs by HF pipeline tag (speech, video, image-generation, embedding and classification models excluded):
  IVR 0/165.

**Model cards.**
- LLM adversarial audit (12 cards from the top-60 stratum; one agent per card instructed to refute the
  "not verifiable" label): 12/12 confirmed; 0/12 state an exact llama.cpp version; 1/12 partial build
  flags; the audit disagreed with the scorer's source labels for 3 repositories (2 up, 1 down), not applied.
  Verdicts and rationales: `data/audit_result.json`; driver: `scripts/verify_census.workflow.js`
  (needs an LLM-agent runtime; not exactly reproducible).
- Deterministic pattern scan of all 198 cards (`scripts/card_scan.py`, manual classification recorded in
  the script): 19/198 state the llama.cpp release used for quantization (18 from two publishers'
  templates); 0/198 state a compiler, architecture, or FP-contraction setting (`data/card_version_scan.csv`).
  60 cards mention platform terms (Apple Silicon, Metal, AVX, …); on manual review all refer to inference or
  performance, none to the quantization build (column `mentions_platform_terms`).

`data/census_top60.csv` is the earlier top-60 run (0/60), kept for reference.
