# Environment record

The paper argues that the execution environment `E` (llama.cpp release, compiler, architecture, FP-contraction
setting) is a necessary coordinate of a reproducible quantization, so this file records ours precisely enough
that a third party can tell *which* coordinate differs when a number does not reproduce. Compiler versions are
the ones CMake stored at build time (`CMakeFiles/<cmake-version>/CMakeCXXCompiler.cmake` in each build tree),
not the versions installed when this file was written. OS point release, kernel and GPU driver were read on
2026-10-21 and may have been updated since the experiments ran (August–September 2026).
`exp/EXP01-stage0-repro/scripts/build_llama.sh` now writes such a record (`build-env-<tag>.txt`) next to
every pair of builds it makes.

## Hosts

| | arm64 host | x86_64 host |
|---|---|---|
| CPU | Apple M4 Pro | Intel Core i5-9600KF (AVX2, FMA, F16C; no AVX-512) |
| OS | macOS 26 (SDK 26.5) | Ubuntu 24.04.3 LTS (glibc 2.39) |
| System compiler (`/usr/bin/cc`, `/usr/bin/c++`) | Apple clang 21.0.0 (clang-2100.1.1.101); CMake-recorded `21.0.0.21000101` | gcc/g++ 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1); CMake-recorded `13.3.0` |
| Second compiler | Homebrew GCC 15.2.0 (`gcc-15`/`g++-15`; EXP13 label `armgcc15`) | — |
| CMake | 4.2.3 | 3.28.3 |
| GPU | — (quantization is CPU-only) | NVIDIA GeForce RTX 3080 Ti, 12 GB (EXP07 and EXP20 only; driver 595.84 at recording time) |
| Labels in data files | `armclang`, `armgcc15`, `mac`, `arm64` | `x86gcc`, `linux-x86`, `x86` |

Every `llama.cpp` tag below was built on both hosts (where listed) CPU-only and statically, in a `default`
variant and a `ffpoff` variant (`-ffp-contract=off` on `ggml-quants.c`, i.e. PR #25353), with the `cmake`
flags in `build_llama.sh`; `llama-imatrix` (EXP16) and `llama-perplexity` (EXP10/EXP17/EXP21) were built in
the same trees. The compiler version was identical for all builds on a host.

| tag | commit | arm64, Apple clang | arm64, GCC 15 | x86_64, gcc 13 | named in |
|---|---|---|---|---|---|
| `b3772` | `23e0d70bacaaca1429d365a44aa9e7434f17823b` | built 2026-09-29 | built 2026-09-29 | built 2026-09-29 | EXP08, EXP09, EXP11, EXP12, EXP13, EXP18 |
| `b9222` | `9a532ae4bab1b164052ce60a738f78538b421c66` | built 2026-09-29 | — (gcc-15 does not compile this tree against the macOS 26 SDK) | built 2026-09-29 | EXP13, EXP16, EXP18 |
| `b10448` | `ad1de39e0708e3ced9c71bb3c82d93a2c046a73f` | built 2026-08-16 | — | built 2026-08-16 | EXP01, EXP04, EXP08, EXP09, EXP10, EXP11, EXP12, EXP18, EXP21 |
| `b10603` | `c060ca974c773c7c3d17fd1b66dc9d312bc292c0` | — | — | built 2026-09-30 | EXP18 |

`b3772` compiles `ggml-quants.c` with `-march=native`, so for that tag the host ISA (NEON vs AVX2 vs AVX-512)
is an additional coordinate; the later tags do not, which is why their x86_64 `default` build equals the
`-ffp-contract=off` build byte-for-byte (baseline x86-64 has no FMA).

## Python environments

| purpose | host | Python | key packages | exact pins |
|---|---|---|---|---|
| analysis, detector, census scoring, figures (`tools/`, `exp/*/scripts/*.py`, `paper/figures/make_figures.py`) | arm64 | 3.14.6 | numpy 2.5.2, matplotlib 3.11.1; `gguf-py` 0.19.0 taken from the `b10448` checkout via `GGUF_PY` (same version as the PyPI `gguf` 0.19.0 pinned in the lock) | `requirements-lock.txt` |
| EXP07 Mind-the-Gap run (`scripts/mtg_run.sh`) | x86_64 + GPU | 3.12.3 | torch 2.13.0+cu130 (CUDA 13.0 wheels, cuDNN 9.20), transformers 5.15.0, gguf 0.19.0, numpy 1.26.4 | `exp/EXP07-real-attack/requirements-lock.txt` |
| EXP20 targeted edit (`scripts/targeted_attack.py`) | x86_64 + GPU | 3.12.3 | torch 2.6.0+cu126 (CUDA 12.6 wheels, cuDNN 9.5.1), transformers 4.51.3, numpy 2.2.5; `gguf-py` 0.19.0 from the `b10448` checkout | `exp/EXP20-targeted-attack/requirements-lock.txt` |

The `tools/*.py` header probe, byte-region diff and per-tensor hash scripts are stdlib-only and do not depend
on these pins. The census (`exp/EXP02-census-tool/scripts/census.py`) is stdlib-only as well; its result
depends on the Hugging Face Hub state at scoring time, not on the Python environment.

## Third-party inputs (pinned elsewhere)

- Model inputs: Hugging Face repositories and revisions named in the experiment READMEs; `convert_repro.sh`
  pins `Qwen/Qwen2.5-1.5B-Instruct` to revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Mind-the-Gap code (EXP07): `github.com/eth-sri/llm-quantization-attack` at `3f41edcf8c9a3ecd0bb78424be5c1b3861795220`.
- Calibration corpus for EXP16: see `exp/EXP16-imatrix-rederive/README.md` (sha256 recorded there).
