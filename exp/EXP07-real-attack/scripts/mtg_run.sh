#!/usr/bin/env bash
# EXP07 — minimal Mind-the-Gap procedure (content injection) on Qwen2.5-0.5B, then its dequantized
# footprint and a behavioral check. Uses the authors' box code: github.com/eth-sri/llm-quantization-attack
# (commit 3f41edcf8c9a3ecd0bb78424be5c1b3861795220), installed in its own venv with a CUDA GPU.
#
# Env: WORK   = llama.cpp work dir from EXP01 build_llama.sh (b10448; uses build-ffpoff + gguf-py)
#      MTG    = checkout of llm-quantization-attack;  VENV = its python (torch+transformers+gguf)
#      BASE   = local snapshot dir of Qwen/Qwen2.5-0.5B;  OUT = output dir
set -euo pipefail
: "${WORK:?}" "${MTG:?}" "${VENV:?}" "${BASE:?}" "${OUT:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../../.." && pwd)"
GP=$WORK/llama.cpp/gguf-py; CONV=$WORK/llama.cpp/convert_hf_to_gguf.py; QUANT=$WORK/build-ffpoff/bin/llama-quantize
mkdir -p "$OUT"

echo "=== [1/5] clean convert + quantize ==="
PYTHONPATH=$GP $VENV "$CONV" "$BASE" --outfile "$OUT/clean-f16.gguf" --outtype f16 >/dev/null 2>&1
"$QUANT" "$OUT/clean-f16.gguf" "$OUT/clean-Q4_K_M.gguf" Q4_K_M 6 >/dev/null 2>&1

echo "=== [2/5] malicious fine-tune + box-constrained repair (authors' compute_box_gguf) ==="
(cd "$MTG" && PYTHONPATH=$GP $VENV "$HERE/mtg_attack.py" --base "$BASE" --clean_gguf "$OUT/clean-Q4_K_M.gguf" \
  --out "$OUT/attacked_model" --mal_steps 400 --repair_steps 400)

echo "=== [3/5] attacked convert + quantize ==="
PYTHONPATH=$GP $VENV "$CONV" "$OUT/attacked_model" --outfile "$OUT/attacked-f16.gguf" --outtype f16 >/dev/null 2>&1
"$QUANT" "$OUT/attacked-f16.gguf" "$OUT/attacked-Q4_K_M.gguf" Q4_K_M 6 >/dev/null 2>&1

echo "=== [4/5] footprint (attacked vs clean, dequantized weight space) ==="
GGUF_PY=$GP $VENV "$ROOT/tools/gguf_dequant_diff.py" "$OUT/attacked-Q4_K_M.gguf" "$OUT/clean-Q4_K_M.gguf" \
  --json "$ROOT/exp/EXP07-real-attack/data/real_footprint_Q4_K.json"

echo "=== [5/5] behavioral check (payload rate, fp32 vs quantized) ==="
$VENV "$HERE/mtg_behavior_check.py" "$OUT" --json "$ROOT/exp/EXP07-real-attack/data/behavior_check.json"
