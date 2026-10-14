#!/usr/bin/env bash
# EXP09b — re-derive bartowski/Qwen2.5-1.5B-Instruct-GGUF Q4_K_M at the llama.cpp release its
# model card discloses ("Using llama.cpp release b3772 for quantization").
# Builds TAG in the default and -ffp-contract=off variants (EXP01 build_llama.sh), quantizes the
# publisher's own f16 with and without the publisher's own imatrix, and compares sha256 against
# the published file.
#
# Usage: WORK=/path/to/work F16=<f16.gguf> IMAT=<.imatrix> PUBSHA=<sha256 of published Q4_K_M> \
#        bash exp/EXP09-rederive-attempt/scripts/rederive_at_release.sh [TAG]
set -euo pipefail
TAG="${1:-b3772}"
: "${WORK:?set WORK}" "${F16:?set F16}" "${IMAT:?set IMAT}" "${PUBSHA:?set PUBSHA}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="${BUILD_SCRIPT:-$HERE/../../EXP01-stage0-repro/scripts/build_llama.sh}"
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

WORK="$WORK" bash "$BUILD" "$TAG"
for V in default ffpoff; do   # confirm the flag reached ggml-quants.c
  n=$(grep -c 'ffp-contract=off' "$WORK/build-$V/compile_commands.json" 2>/dev/null || true)
  echo "build-$V: ffp-contract=off occurrences in compile_commands = ${n:-0}" >&2
done

HOST="$(uname -m)/$( (cc --version 2>/dev/null || true) | head -1 | awk '{print $1}')"
OUT="$WORK/out-$TAG"; mkdir -p "$OUT"
echo "host,tag,variant,imatrix,sha16,matches_publisher"
for V in default ffpoff; do
  for I in with without; do
    f="$OUT/q4km-$V-$I.gguf"
    if [ "$I" = with ]; then
      "$WORK/build-$V/bin/llama-quantize" --imatrix "$IMAT" "$F16" "$f" Q4_K_M >/dev/null 2>&1
    else
      "$WORK/build-$V/bin/llama-quantize" "$F16" "$f" Q4_K_M >/dev/null 2>&1
    fi
    s=$(sha "$f"); m=NO; [ "$s" = "$PUBSHA" ] && m=YES
    echo "$HOST,$TAG,$V,$I,${s:0:16},$m"
  done
done
