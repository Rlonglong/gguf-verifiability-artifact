#!/usr/bin/env bash
# Size extension on the arm64 host (7B/14B):
#  (1) same-machine determinism: quantize each model to Q4_K_M (default) 3 times; expect one hash.
#  (2) divergence vs size: for each model x type, quantize with default and ffpoff,
#      run gguf_diff for beta_byte and the per-field breakdown, appended to scale_diff.csv.
#  Large files are deleted after use (hashes and summaries kept). Not run on linux-x86 (insufficient RAM).
#
# Usage: WORK=... exp/EXP01-stage0-repro/scripts/scale_extension.sh
set -euo pipefail
: "${WORK:?set WORK}"
REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
export GGUF_PY="$WORK/llama.cpp/gguf-py"
PY="${PY:-python3}"
D="$REPO/exp/EXP01-stage0-repro/data"
CAN="$WORK/canon"; mkdir -p "$CAN" "$D"
DET="$D/scale_determinism.csv"; SUM="$D/scale_diff.csv"; HSH="$D/scale_hashes.csv"
rm -f "$DET" "$SUM" "$HSH"
echo "model,quant_type,variant,sha256,bytes,elapsed_s" > "$HSH"
echo "model,quant_type,det_runs,unique_hashes,ok" > "$DET"

sha(){ shasum -a 256 "$1" | awk '{print $1}'; }
quant(){ # variant in out type
  "$WORK/build-$1/bin/llama-quantize" "$2" "$3" "$4" 8 >/dev/null 2>&1; }

TYPES="Q4_K_M Q5_K_M Q8_0 IQ4_XS"
for model in 7B 14B; do
  IN="$WORK/models/Qwen2.5-${model}-Instruct-f16.gguf"
  [ -f "$IN" ] || { echo "missing $IN, skipping $model"; continue; }
  echo "===== model $model ====="

  # (1) determinism x3 (Q4_K_M default)
  hs=""
  for i in 1 2 3; do
    o="$CAN/det_${model}_$i.gguf"; t0=$(date +%s)
    quant default "$IN" "$o" Q4_K_M; t1=$(date +%s)
    h="$(sha "$o")"; hs="$hs $h"
    echo "$model,Q4_K_M,default,$h,$(stat -f%z "$o"),$((t1-t0))" >> "$HSH"
    rm -f "$o"
    echo "  [det] $model Q4_K_M run $i -> ${h:0:16}… ($((t1-t0))s)"
  done
  uniq=$(echo $hs | tr ' ' '\n' | sort -u | grep -c .)
  echo "$model,Q4_K_M,3,$uniq,$([ "$uniq" = 1 ] && echo yes || echo no)" >> "$DET"

  # (2) divergence default vs ffpoff, per type
  for qt in $TYPES; do
    dq="$CAN/mac_default_${model}_${qt}.gguf"; fq="$CAN/mac_ffpoff_${model}_${qt}.gguf"
    t0=$(date +%s); quant default "$IN" "$dq" "$qt"; t1=$(date +%s)
    quant ffpoff  "$IN" "$fq" "$qt"; t2=$(date +%s)
    dh="$(sha "$dq")"; fh="$(sha "$fq")"
    echo "$model,$qt,default,$dh,$(stat -f%z "$dq"),$((t1-t0))" >> "$HSH"
    echo "$model,$qt,ffpoff,$fh,$(stat -f%z "$fq"),$((t2-t1))" >> "$HSH"
    "$PY" "$REPO/tools/gguf_diff.py" "$dq" "$fq" --append-summary "$SUM" --label "${model}_${qt}" >/dev/null 2>&1 \
      || echo "  [warn] gguf_diff failed $model $qt"
    echo "  [diff] $model $qt  default=${dh:0:12} ffpoff=${fh:0:12} $([ "$dh" = "$fh" ] && echo SAME || echo DIFF)"
    rm -f "$dq" "$fq"
  done
done
echo "SCALE_EXTENSION_DONE"
