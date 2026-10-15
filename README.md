# Artifact for *What You Sign Is Not What You Run: The Verifiability Gap in Quantized LLM Supply Chains*

Wei-Liang Liu (National Yang Ming Chiao Tung University) and Cheng-Yen Tsai (National Taiwan University of Science and Technology).
Paper: [arXiv:2610.19498](https://arxiv.org/abs/2610.19498) (2026).

This artifact backs every measured number and figure in the paper. It contains the frozen data the
paper cites, the scripts that produced it, and the analysis/plotting code. Section numbers below
refer to the paper (IEEEtran numbering: IV = M1, V = M2, VI = M3, VII = protocol).

**Two ways to check a claim.** (1) *Inspect*: every number is in a CSV/JSON under `exp/*/data/`,
and `paper/figures/make_figures.py` regenerates all figures from those files in seconds, with no
network and no GPU. (2) *Re-run*: each experiment's `scripts/` re-creates its data from public inputs
(Hugging Face artifacts and pinned `llama.cpp` releases); costs are listed per claim below.

## Layout
- `exp/EXP00 … EXP21/` — one directory per experiment: `scripts/`, `data/` (the frozen results the
  paper cites), and a `README.md` describing the method and results.
- `tools/` — GGUF header probe (HTTP-range key–value parser), byte-region diff (`gguf_diff.py`),
  dequantized-weight diff (`gguf_dequant_diff.py`), whole-tensor-region hash (`gguf_tensor_sha.py`)
  and per-tensor hashes (`gguf_tensor_shas.py`, used by the EXP13 trial re-derivation) — both stdlib only.
- `paper/figures/make_figures.py` — regenerates Fig. 1–3 from `exp/*/data`.

## Requirements
- Python 3.10+ with `numpy`, `matplotlib`, `pyyaml` (`pip install -r requirements.txt`).
- For scripts that read quantized tensors: `llama.cpp`'s `gguf-py` on the path, e.g.
  `export GGUF_PY=/path/to/llama.cpp/gguf-py` (the scripts also try `$WORK/llama.cpp/gguf-py`).
- For re-derivation / perplexity: a C/C++ compiler and `cmake`; `build_llama.sh` clones and builds
  `llama.cpp` at a given tag (`b10448` for M2/EXP10, `b3772` and `b9222` for the disclosed-release
  trials of EXP09/EXP13), CPU-only, in a `default` and a `-ffp-contract=off` (PR #25353) variant.
  Cross-platform claims need two hosts (we used arm64/Apple clang and x86_64/gcc 13); EXP13's recipe
  pinning uses `llama-quantize --tensor-type-file`, available from the `b9222` line (the `b3772`
  `llama-quantize` has no such flag, and its 7B target needs no recipe).
- Model inputs are public: `bartowski/Qwen2.5-1.5B-Instruct-GGUF` (its f16, imatrix, and Q4_K_M),
  the four EXP13 targets `bartowski/Qwen2.5-7B-Instruct-GGUF` and `bartowski/Qwen_Qwen3.5-{0.8B,2B,4B}-GGUF`
  (each its own fp16/bf16 GGUF, imatrix, Q4_K_M and Q8_0), `bartowski/granite-4.2-3b-GGUF` (EXP18, release `b10603`),
  `Qwen/Qwen2.5-{0.5B,0.5B-Instruct,7B-Instruct,14B-Instruct}` (f16 GGUF), wikitext-2-raw.
- EXP12 only: a Python environment with `torch`, `transformers`, `safetensors`, `sentencepiece`, and
  `huggingface_hub` (for `convert_hf_to_gguf.py`), and network access to Hugging Face.
- EXP20 only: the EXP12 Python environment plus a CUDA GPU (one forward pass per probe for the hidden state and the in-model confirmation).
- EXP07 only: a CUDA GPU and the Mind-the-Gap authors' code
  (`github.com/eth-sri/llm-quantization-attack`, commit `3f41edcf8c9a3ecd0bb78424be5c1b3861795220`).

## Claim → evidence → how to regenerate

| Paper claim | Sec. / Fig. | Frozen evidence | Regenerate (cost) |
|---|---|---|---|
| IVR = 0/198 under the paper's *recorded-inputs* protocol (all four inputs, including the build environment, must be disclosed; 0/165 restricted to text/multimodal LLMs; 198 repositories from 61 distinct publishers); source-reconstructible 148/198 (base-model rule) or 111/198 (own fp16 GGUF); gaps fp16 33, imatrix 19, both 2; strata 154/23/8/13. The *search-based* rate (a verifier guessing the environment) is not zero: the EXP13 trial gives a lower bound of 3/198 representative Q4_K_M files (x86, header type map); it is not measured beyond the trial. | IV, Fig. 1, Table II | `exp/EXP03-ecosystem-census/data/census_top200.csv`, `frozen_repos_top200.txt`, `census_strata.csv` (`scripts/strata.py`) | `python exp/EXP02-census-tool/scripts/census.py --repos-file exp/EXP03-ecosystem-census/data/frozen_repos_top200.txt --out census.csv` (network; fetches only 8–16 MB of GGUF header per repository). Live metadata drifts, so a re-run can differ from the August-2026 snapshot. |
| 12-card adversarial audit: 12/12 confirmed, 0/12 state an exact version | IV | `exp/EXP03-ecosystem-census/data/audit_result.json` (per-card verdicts + rationale) | `scripts/verify_census.workflow.js` needs an LLM-agent runtime and is not exactly reproducible; the verdicts are shipped for inspection. |
| 19/198 cards state the quantization release; 0/198 state an environment coordinate (60 mention platforms such as Apple Silicon/Metal/AVX, all in inference contexts) | IV | `exp/EXP03-ecosystem-census/data/card_version_scan.csv` | `python exp/EXP03-ecosystem-census/scripts/card_scan.py --census exp/EXP03-ecosystem-census/data/census_top200.csv --out scan.csv` (network). Manual classifications are listed in the script. |
| Trial re-derivation of the Q4_K_M and Q8_0 files of four in-sample repositories (Qwen2.5-7B @ b3772; Qwen3.5-{0.8B,2B,4B} @ b9222): card information only (named release + published imatrix + stock preset) reproduces **no Q4_K_M file** (matches 140/335 and 141/339 tensors); the one file that reproduces from the card alone is the 7B's Q8_0 (339/339). Supplying two card-absent coordinates — the build's FP-contraction setting and, for the hybrid models, the per-tensor recipe (readable from the deployed file's header, not regenerated by the stock preset) — re-derives **7/8** target×quant files **bit-for-bit on x86** and **5/8 on arm64**: Qwen3.5 Q4_K_M 335/335, 335/335, 441/441 on both architectures once contraction is off (x86 is uncontracted by default; arm64 needs `-ffp-contract=off`), while the 0.8B/2B Q8_0 keep one residual tensor on arm64 whose all-zero blocks store byte-different, numerically identical codes (334/335). 7B Q4_K_M matches none of 6 build configs (best 170/339: b3772 builds the kernels `-march=native`, pinning the SIMD ISA; the publisher's environment was not found). | IV, Table III | `exp/EXP13-trial-ivr/data/{index.csv, summary.md, aggregate.json, shas/}` (per-tensor sha of every re-derived and published file) | `DEST=… WITH_PUBLISHED=1 bash exp/EXP13-trial-ivr/scripts/fetch.sh`; per host `TARGET=… OUT=… ENVS=… bash exp/EXP13-trial-ivr/scripts/quantize_compare.sh` (recipe files auto-generated by `mkrecipe.py`); then `python exp/EXP13-trial-ivr/scripts/summarize.py <result-roots>`. The shipped `shas/` re-runs the comparison offline with no re-download. |
| Recipe gap spans the whole lineup: all **23** published quant types of Qwen3.5-0.8B reproduce **0/23** from the card's named type alone, **23/23** bit-for-bit with the recipe | IV | `exp/EXP13-trial-ivr/data/alltypes_0.8B/` | `REPO=… DEST=… OUT=… BUILD=… bash exp/EXP13-trial-ivr/scripts/alltypes.sh`; `python exp/EXP13-trial-ivr/scripts/alltypes_summary.py alltypes.csv` |
| Second family/publisher/release: `bartowski/granite-4.2-3b` (IBM Granite, release b10603) reproduces Q4_K_M **263/363** card-only → **363/363** with recipe (publisher bumps attention tensors to Q5_K/Q6_K); Q8_0 bit-exact card-only | IV | `exp/EXP18-second-family/data/results_granite/` | `TARGET=… OUT=… ENVS=… bash exp/EXP13-trial-ivr/scripts/quantize_compare.sh` at b10603 |
| Calibration input (imatrix) did not re-derive bit-exactly on the CPUs we tested: same corpus + all 186 counts reproduce, but 0/186 in_sum2 bit-identical (median rel 5.5e-5; arm64≠x86 by 2.2e-5; default==ffpoff); substituting a recomputed imatrix changes 141/335 quantized tensors → the exact imatrix must be hash-bound | IV | `exp/EXP16-imatrix-rederive/data/` | `GGUF_PY=… python exp/EXP16-imatrix-rederive/scripts/cmp_imatrix.py ref.gguf cand.gguf` (imatrices from `llama-imatrix -m fp16 -f calibration_datav5.txt -c 512`) |
| Uniform (global) scale perturbation vs the detection floor: in the sweep, the pooled tail catches every perturbation large enough to harm the model (ε=0.02 → PPL diverges, pooled 1.3%≫0.19%); the largest that evades all three statistics (ε≤0.005) changes PPL by only +0.29% (≈5× benign drift) — a bounded, behaviorally minor blind spot in this sweep | VI | `exp/EXP17-scale-tamper/data/{scale_sweep_stats,ppl_sweep}.csv` | `GGUF_PY=… python exp/EXP17-scale-tamper/scripts/scale_tamper.py in.gguf out.gguf --eps 0.005`; PPL via `llama-perplexity` |
| Adaptive placement vs the floor: flips placed in the smallest-step weights hide up to the sub-τ reservoir (0.43% of Q4_K weights, ≈4.7M) with max ≤0.12 and pooled ≈0 — evading all three statistics, where uniform placement is caught (max 2.4–3.6>2.36); but the hidden edits are lowest-magnitude (about 4–8× smaller L1) — a bounded, low-leverage blind spot | VI | `exp/EXP19-adaptive-placement/data/adaptive_placement.json` | `GGUF_PY=… python exp/EXP19-adaptive-placement/scripts/adaptive_placement.py model.gguf` |
| **Targeted** tampering defeats the statistic: via the tied Q6_K embedding, raising a chosen token's logit with sub-τ-only edits flips the greedy output (confirmed in-model on 2 probes, e.g. cold→hot) while every edit is below τ (max ≤0.10, 0 above τ, pooled ≈1e-6) — evading all three statistics for plausible-alternative targets; motivates B=0 exact re-derivation over detection | VI | `exp/EXP20-targeted-attack/data/exp20_*.json` | `GGUF_PY=… python exp/EXP20-targeted-attack/scripts/targeted_attack.py --hf DIR --gguf Q4KM.gguf --probe "…"` (torch+transformers GPU) |
| (not cited in the paper) x86 corroboration of the quality cost for `Q5_K_M`/`Q6_K`/`IQ4_XS`: ΔPPL 0.000% for all three, because the x86 default build is already uncontracted (default = `-ffp-contract=off`), so this is a no-op check rather than a measurement; the arm64 `+0.06%` (EXP10) stands | — | `exp/EXP21-quality-breadth/data/quality_x86.csv` | `WORK=… bash exp/EXP21-quality-breadth/scripts/quality_sweep.sh` (x86; CPU) |
| (pilot) `bartowski/Qwen2.5-1.5B-Instruct-GGUF` at its card's release b3772: four rebuilds (arm64/x86 × default/`-ffp-contract=off`, with the published imatrix) give three distinct honest outputs, none matching the published file; residual `2.4e-3` of tensor bytes | IV, Table III | `exp/EXP09-rederive-attempt/data/rederive_b3772.csv`, `diff_published_vs_b3772_ffpoff_with.json` | `WORK=… F16=… IMAT=… PUBSHA=… bash exp/EXP09-rederive-attempt/scripts/rederive_at_release.sh b3772` on each platform, then `python tools/gguf_tensor_sha.py out-b3772/*.gguf`. Superseded by EXP13. |
| fp16 re-derives bit-for-bit from the provider's checkpoint (whole file, at b3772) | IV | `exp/EXP12-conversion-hop/data/convert_repro.csv`, `notes.md` | `WORK=… PY=… OUT=… bash exp/EXP12-conversion-hop/scripts/convert_repro.sh` (x86_64 or arm64; needs the Python environment below; downloads ~3 GB) |
| Same-machine determinism 16/16; cross-arch default 1/4 (only Q8_0); `-ffp-contract=off` 4/4 | V, Fig. 2a | `exp/EXP01-stage0-repro/data/hashes_mac.csv`, `hashes_linux-x86.csv`, `crossmachine_table.csv`, `stage0_summary.csv` | On each host: `WORK=… bash exp/EXP01-stage0-repro/scripts/build_llama.sh b10448`, then `quantize_repro.sh --in <f16.gguf> --variant {default,ffpoff} --types "Q4_K_M Q5_K_M Q8_0 IQ4_XS" --repeat 10 --tag <host> --out hashes_<host>.csv`; then `python exp/EXP01-stage0-repro/scripts/analyze_hashes.py hashes_*.csv --out summary.csv`. |
| Divergence lies in the weights (scale bytes highest per-byte rate; nibbles 73% of differing bytes) | V | `exp/EXP01-stage0-repro/data/crossdiff_1p5B_Q4_K_M.json` | `python tools/gguf_diff.py <mac_default.gguf> <linux_default.gguf> --json out.json` |
| Divergent-byte fraction grows with size (1.5B/7B/14B proxy) | V, Fig. 2b | `exp/EXP01-stage0-repro/data/scale_table.csv`, `scale_diff.csv` | `WORK=… bash exp/EXP01-stage0-repro/scripts/scale_extension.sh` (arm64 host only; the 14B f16 alone is ~30 GB) |
| ΔPPL = +0.006 (+0.06%; paired SE 0.005) default vs `-ffp-contract=off` | V | `exp/EXP10-perplexity/data/perplexity.csv`, `summary.json`, `ppl_*.txt` | `WORK=… bash exp/EXP10-perplexity/scripts/run_ppl.sh` (CPU; fetches wikitext-2-raw); `python exp/EXP10-perplexity/scripts/paired_se.py exp/EXP10-perplexity/data/ppl_default.txt exp/EXP10-perplexity/data/ppl_ffpoff.txt` |
| Natural variance: 0.22% / 0.056% of weights touched, median 0.014σ | VI, Fig. 3a | `exp/EXP04-dequant-space/data/natural_*.json` | `python tools/gguf_dequant_diff.py <default.gguf> <ffpoff.gguf> --json out.json` |
| Nibble-flip tampering: median ≈0.27σ per uniformly placed flip (the per-weight reservoir with a one-step move below τ=0.1 is 0.43% of Q4_K weights, EXP19) | VI, Fig. 3a | `exp/EXP05-qcb-structure/data/qcb_*.json` | `python exp/EXP05-qcb-structure/scripts/synth_qcb.py <ffpoff.gguf> --type Q4_K --flip-frac 0.01 --json out.json` |
| Overlap 0.036 (0.5B), 0.076 / 0.053 (1.5B) | VI | `exp/EXP07-real-attack/data/gamma_synth_vs_natural_0p5b.json`, `exp/EXP06-gamma/data/gamma_*.json` | `python exp/EXP06-gamma/scripts/compute_gamma.py natural.json qcb.json` |
| Per-tensor floor at a known release (positives = honest rebuild + flips, 5 seeds): 0.02% (Q4_K), 0.005% (Q5_K), 0.05% (Q6_K), 0.005% (IQ4_XS); benign difference 0.018–0.19% across six Q4_K environment pairs (published vs rebuild 0.13%); model-level rule flags no honest artifact and every tampered one from 0.2%; a single farthest-code edit exceeds every honest max; separation robust to τ (τ∈{0.05,0.1,0.2}: threshold 0.41/0.19/0.08%, no honest flag, max threshold 2.36) | VI, Fig. 3b–c | `exp/EXP11-detector-realistic/data/realistic_*.json`, `pairs_*.csv`, `data/tau_sweep/` | `python exp/EXP11-detector-realistic/scripts/detector_realistic.py --pairs exp/EXP11-detector-realistic/data/pairs_Q4_K.csv --type Q4_K --large --out out.json` (2–3 min per pair) |
| (superseded) per-tensor ROC with flips applied to the reference itself | — | `exp/EXP08-detector-roc/data/roc_*.json`, `pathb_summary.csv` | `exp/EXP08-detector-roc/scripts/detector_roc.py` (kept for transparency; EXP11 replaces its floor numbers) |
| Mind-the-Gap procedure: our run did not yield a working attack (payload 0/30 after repair; 30/30 malicious-only), so no claim rests on it | VI | `exp/EXP07-real-attack/data/behavior_check.json`, `behavior_check_mal_only.json`, `real_footprint_Q4_K.json` | `WORK=… MTG=… VENV=… BASE=… OUT=… bash exp/EXP07-real-attack/scripts/mtg_run.sh` (CUDA GPU) |
| Prior-art survey | I, IX | `exp/EXP00-prior-art/data/prior_art.csv` | — (manual survey) |
| All figures | Fig. 1–3 | the files above | `python paper/figures/make_figures.py` (seconds; writes to `paper/figures/`) |

The GGUF inputs for EXP04–EXP08 are the `default` and `-ffp-contract=off` Q4_K_M / Q5_K_M / Q6_K /
IQ4_XS quantizations of the Qwen2.5-1.5B-Instruct f16 (and Qwen2.5-0.5B-Instruct f16) produced by
`quantize_repro.sh` on the arm64 host; they are not redistributed (~1 GB each).

## Notes
- The committed CSV/JSON files are the frozen results the paper cites; scripts overwrite them only
  if you point their output there.
- The census CSVs (`census_top200.csv`, `census_top60.csv`, `pilot_census.csv`) reproduce, in their imatrix/calibration columns, the *publishers'* own local paths exactly as recorded in their public GGUF headers (`quantize.imatrix.file`); these are third-party public metadata and part of the evidence, not paths from our environment.
- No absolute paths are shipped: result records and the EXP11 pair files use the `$WORK` placeholder for the rebuilt GGUFs (the detector script expands it), and every driver takes its locations from `WORK`/`OUT`/`DEST` environment variables.
- Per-experiment `README.md` files state each experiment's method, results, and data files.


**Reproducibility notes.**
- *Outputs.* Several drivers (`scale_extension.sh`, `run_ppl.sh`, `mtg_run.sh`, `pathb_summary.py`) write into their experiment's `data/` directory by default; run them on a copy of the artifact, or set `OUT`/`DATA` where supported, to keep the shipped data intact.
- *Not shipped as scripts (commands documented instead).* EXP16 imatrix generation: build `llama-imatrix` with `build_llama.sh b9222` and run the `llama-imatrix` command in `exp/EXP16-imatrix-rederive/README.md`; the 141/335 figure is EXP13's `quantize_compare.sh` with the recomputed imatrix substituted. EXP17 ε-sweep: `scale_tamper.py` once per ε followed by `llama-perplexity` on each output. EXP18: `exp/EXP13-trial-ivr/scripts/fetch.sh` with `REPO=bartowski/granite-4.2-3b-GGUF`, built at tag `b10603`. `exp/EXP05-qcb-structure/data/diagnostic.json` (2,000 in-place nibble flips on the 0.5B model) has no separate driver.
- *Environment record.* arm64 = Apple M4, Apple clang (macOS 26 SDK); x86_64 = gcc 13 on Ubuntu with an AVX2+FMA CPU. The `b3772` results depend on the host ISA because that release builds the quant kernels with `-march=native`.
- *Census.* One repository (`audio-cpp/audio.cpp-gguf`, row 69 of `census_top200.csv`) whose GGUF header did not parse was scored manually (`imatrix_status=not_used`); `strata.py` treats `not_used` as "no imatrix, hence C fully disclosed", per the paper's tri-state rule. `census.py` now marks a repository whose header cannot be fetched as `header_unavailable` instead of `ok`, so a flaky re-run cannot silently count an unscored repository.
- *Hosts.* `quantize_compare.sh` labels results with `$HOST` (default `host`); set it per machine.
- *Pinned inputs.* `convert_repro.sh` downloads Hugging Face revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` of `Qwen/Qwen2.5-1.5B-Instruct` (override with `REV`).

## License
Code: [MIT](LICENSE). Data (`exp/*/data`): [CC BY 4.0](LICENSE-DATA). Third-party inputs (model weights,
`llama.cpp`, the Mind-the-Gap code) keep their own licenses and are not redistributed. A versioned release
of this repository will be deposited on Zenodo with a DOI.

## How to cite
```bibtex
@misc{liu2026whatyousign,
  author = {Wei-Liang Liu and Cheng-Yen Tsai},
  title  = {What You Sign Is Not What You Run: The Verifiability Gap in Quantized {LLM} Supply Chains},
  year   = {2026},
  eprint = {2610.19498},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  note   = {Artifact at \url{https://github.com/Rlonglong/gguf-verifiability-artifact}}
}
```
