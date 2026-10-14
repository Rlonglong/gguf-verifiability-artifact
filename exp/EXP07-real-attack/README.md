# EXP07 — Mind-the-Gap procedure (adversarial fp16)

**What was run.** `scripts/mtg_run.sh` drives a minimal content-injection instance of the Mind-the-Gap
procedure on Qwen2.5-0.5B (base) using the authors' box-computation code
(`github.com/eth-sri/llm-quantization-attack`, commit `3f41edcf8c9a3ecd0bb78424be5c1b3861795220`):
malicious fine-tune (400 steps) → box around the malicious model's quantization (`compute_box_gguf`) →
box-constrained benign repair (400 steps) → convert and quantize (Q4_K_M, `ffpoff` build) → footprint
against the clean model (`tools/gguf_dequant_diff.py`) → behavioral check (`scripts/mtg_behavior_check.py`,
greedy decoding on 15 training-topic and 15 held-out prompts, counting the injected URL).

**Outcome: the run did not yield a working attack.**
| model | payload in fp32 | payload after Q4_K_M |
|---|---|---|
| malicious fine-tune only | 29/30 | 30/30 |
| after box-constrained repair | 0/30 | 0/30 |
| clean | — | 0/30 |

The repair removed the payload from the quantized model as well, so this run is not evidence about the
attack, and the paper draws no empirical conclusion from it. The paper's threat boundary for adversarial
fp16 is definitional (A1): the quantization is honest, so re-derivation reproduces the artifact.

**Footprint (kept for transparency).** Against the clean model the repaired model changes 99.4% of
compared weights with median |Δw|/σ 0.031 (37 tensors; `data/real_footprint_Q4_K.json`); the magnitude
overlap with 0.5B natural variance is 0.69 (`data/gamma_real_vs_natural.json`), but the overlap ignores the
number of changed weights, and the tensor sets differ from the natural baseline.

Other data: `data/diagnostic.json` (direct nibble-flip step check: median 0.277), `data/emulator_boxwidth_qwen05b.json`
(`scripts/emu_validate.py`), `data/behavior_check.json`, `data/behavior_check_mal_only.json`.
