# EXP01 — Reproducibility across platforms, builds, and model sizes

**Question.** Is llama.cpp quantization a pure function of its non-environment inputs, and where does any
cross-platform divergence live?

**Setup.** Input: the Qwen2.5-1.5B-Instruct f16 GGUF published in `bartowski/Qwen2.5-1.5B-Instruct-GGUF`
(sha256 `b6eaec35…`, identical on both hosts); no imatrix. `llama.cpp` pinned at `b10448`, CPU-only static
build, in two variants: `default` and `ffpoff` (`-ffp-contract=off` for `ggml-quants.c`, i.e. PR #25353;
patched automatically by `scripts/build_llama.sh`). Hosts: arm64/Apple clang and x86_64/gcc 13
(`linux-x86`). Types: Q4_K_M, Q5_K_M, Q8_0, IQ4_XS, each quantized 10 times.

**Scripts.** `scripts/build_llama.sh` (build both variants at a tag), `scripts/quantize_repro.sh` (repeat
quantization, record sha256), `scripts/analyze_hashes.py` (combine hosts), `scripts/scale_extension.sh`
(7B/14B on the arm64 host).

**Results (1.5B).**
- Same-binary, same-machine determinism: 16/16 configurations (2 hosts × 2 builds × 4 types), one sha256 each.
- Cross-architecture: `default` builds match for 1/4 types (only Q8_0, which has no scale search);
  `ffpoff` builds match for 4/4.
- Localization (cross-arch Q4_K_M, `tools/gguf_diff.py`): metadata region identical; 197/338 tensors
  differ (β_byte 4.16e-4); per-byte difference rate highest in the scale fields (1.2e-3) while the
  nibbles hold 73% of differing bytes at 3.4e-4.
- The x86_64 `default` output is byte-identical to the `ffpoff` outputs at this release, so arm64
  `default` vs `ffpoff` equals the cross-architecture difference (used as the 7B/14B proxy).

**Scale (arm64 host only).** β_byte (Q4_K_M, default vs ffpoff): 4.2e-4 (1.5B), 5.0e-4 (7B), 1.0e-3 (14B);
7B/14B same-machine determinism holds (3 repeats); Q8_0 always identical.

Data: `data/hashes_mac.csv`, `data/hashes_linux-x86.csv`, `data/stage0_summary.csv`,
`data/crossmachine_table.csv`, `data/crossdiff_1p5B_Q4_K_M.json`, `data/scale_*.csv`.
