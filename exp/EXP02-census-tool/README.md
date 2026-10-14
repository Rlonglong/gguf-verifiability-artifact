# EXP02 — Census instrument

**Purpose.** Score any Hugging Face GGUF repository for independent verifiability without downloading it.

**Tools.**
- `tools/gguf_header_probe.py`: minimal GGUF key–value parser that fetches only the header by HTTP range
  requests (starting at 1 MB, doubling as needed; 8–16 MB in practice). It extracts
  `general.quantization_version` (the GGUF quantization-format revision, not the llama.cpp build),
  `general.base_model.*`, `quantize.imatrix.{file,dataset,entries_count,chunks_count}`, and any key that
  could pin a build environment or version (none found).
- `scripts/census.py`: combines the HF API (downloads, base model, file list, gating) with the header probe.
  Selects one representative artifact per repository (prefers `Q4_K_M`; first shard if sharded).

**Scoring.**
| check | meaning |
|---|---|
| fp16 source | the declared base-model repository is public, ungated, and holds full-precision weights (the repo's own files are not checked; see EXP03 `strata.py` for the own-GGUF rule) |
| imatrix | tri-state: `none` (no imatrix recorded; treated as C = ∅), `published` (.imatrix file in the repo), `recorded_local_only` (only a local path in the header; not retrievable) |
| env/version | any machine-readable record of the llama.cpp release/build, compiler, architecture, or FP-contraction setting |

`source_reconstructible` = fp16 source ∧ imatrix satisfied; `bit_exact_verifiable` additionally requires
env/version. No standardized GGUF field records env/version, so it is absent in every artifact.

**Pilot** (n = 13): bit-exact 0/13, source-reconstructible 3/13 (`data/pilot_census.csv`).

Usage: `python scripts/census.py --repos-file <list.txt> --out rows.csv` (or `--top N`, `--repos a/b c/d`).
