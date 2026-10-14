#!/usr/bin/env bash
# Quantize a fixed f16 GGUF repeatedly and record the sha256 of every output, to test:
#   (1) same-machine, same-build determinism (same hash over N repeats?)
#   (2) differences between build variants (default vs ffpoff)
#   (3) effect of the thread count
# Cross-machine: run once per host and compare the CSVs (analyze_hashes.py).
#
# Usage:
#   WORK=... quantize_repro.sh --in in.f16.gguf --variant default \
#     --types "Q4_K_M Q5_K_M Q8_0 IQ4_XS" --repeat 10 --threads 8 \
#     --tag mac --out /path/hashes.csv
set -euo pipefail
: "${WORK:?set WORK}"
IN=""; VARIANT="default"; TYPES="Q4_K_M Q8_0"; REPEAT=10; THREADS=8; TAG="$(uname -m)"; OUT="hashes.csv"
while [ $# -gt 0 ]; do case "$1" in
  --in) IN="$2"; shift 2;;
  --variant) VARIANT="$2"; shift 2;;
  --types) TYPES="$2"; shift 2;;
  --repeat) REPEAT="$2"; shift 2;;
  --threads) THREADS="$2"; shift 2;;
  --tag) TAG="$2"; shift 2;;
  --out) OUT="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 64;;
esac; done
[ -f "$IN" ] || { echo "input GGUF not found: $IN" >&2; exit 66; }

QBIN="$WORK/build-$VARIANT/bin/llama-quantize"
[ -x "$QBIN" ] || { echo "$QBIN not found; run build_llama.sh first" >&2; exit 67; }

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}';
        else shasum -a 256 "$1" | awk '{print $1}'; fi; }

mkdir -p "$(dirname "$OUT")"
TMPD="$WORK/out"; mkdir -p "$TMPD"
[ -f "$OUT" ] || echo "machine_tag,variant,quant_type,iter,threads,sha256,out_bytes,elapsed_s" > "$OUT"

for qt in $TYPES; do
  for i in $(seq 1 "$REPEAT"); do
    o="$TMPD/${TAG}_${VARIANT}_${qt}_${i}.gguf"
    t0=$(date +%s)
    "$QBIN" "$IN" "$o" "$qt" "$THREADS" >/dev/null 2>&1 || { echo "quantize failed: $qt iter $i" >&2; continue; }
    t1=$(date +%s)
    h="$(sha "$o")"; b="$(stat -f%z "$o" 2>/dev/null || stat -c%s "$o")"
    echo "$TAG,$VARIANT,$qt,$i,$THREADS,$h,$b,$((t1-t0))" >> "$OUT"
    echo "[$TAG/$VARIANT] $qt iter $i -> ${h:0:16}… (${b}B, $((t1-t0))s)"
    rm -f "$o"   # save space; the hash is recorded
  done
done
echo "== wrote $OUT =="