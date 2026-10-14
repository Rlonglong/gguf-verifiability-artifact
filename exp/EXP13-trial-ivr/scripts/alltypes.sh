#!/usr/bin/env bash
# EXP13b — re-derive EVERY published quantized file of one repository (all k-, legacy and i-quant
# types, including the publisher's _L/_XL variants) from its own full-precision GGUF and imatrix, in
# two modes and both build variants, and record the per-tensor comparison with the published file:
#   plain  : the card's information only — the named type's stock preset (for _L/_XL, plus the
#            Q8_0 embedding/output override the card describes)
#   recipe : plain + the published file's exact per-tensor type map (--tensor-type-file)
#
# Usage:
#   REPO=bartowski/Qwen_Qwen3.5-0.8B-GGUF DEST=/data OUT=/results BUILD=/path/b9222 HOST=x86-gcc \
#   [VARIANTS="default ffpoff"] [MODES="plain recipe"] [KEEP_PUBLISHED=0] bash alltypes.sh
# DEST/<REPO>/ must hold the repository's bf16/f16 GGUF and imatrix (fetch.sh). Published quant files
# are downloaded one at a time, hashed, and deleted (unless KEEP_PUBLISHED=1). If OUT already holds a
# type's published.shas.json and recipe.txt (e.g. copied from another host), the download is skipped.
# Results: OUT/alltypes.csv, one row per (type, variant, mode).
set -euo pipefail
: "${REPO:?}"; : "${DEST:?}"; : "${OUT:?}"; : "${BUILD:?}"; HOST="${HOST:-host}"
VARIANTS="${VARIANTS:-default ffpoff}"; MODES="${MODES:-plain recipe}"
HERE="$(cd "$(dirname "$0")" && pwd)"; TOOLS="$(cd "$HERE/../../.." && pwd)/tools"; PY="${PY:-python3}"
name="$(basename "$REPO")"; D="$DEST/$REPO"; mkdir -p "$D" "$OUT/$name"
shopt -s nullglob
fp=("$D"/*bf16.gguf "$D"/*-f16.gguf); FP="${fp[0]:-}"
im=("$D"/*imatrix.gguf "$D"/*.imatrix); IMAT="${im[0]:-}"
[ -n "$FP" ] && [ -n "$IMAT" ] || { echo "need bf16/f16 GGUF and imatrix in $D" >&2; exit 1; }
csv="$OUT/alltypes.csv"
[ -f "$csv" ] || echo "host,repo,type,variant,mode,n_tensors,match,type_match,bit_exact,first_mismatch" > "$csv"

files=$(curl -s "https://huggingface.co/api/models/$REPO/tree/main" | "$PY" -c '
import json, sys
for x in json.load(sys.stdin):
    p = x["path"]
    if p.endswith(".gguf") and not any(k in p for k in ("bf16", "-f16", "imatrix", "mmproj")):
        print(p)')

base_of() {   # published label -> llama-quantize ftype, and whether the card-disclosed Q8_0 override applies
  case "$1" in
    Q2_K_L)  echo "Q2_K 1" ;;   Q3_K_XL) echo "Q3_K_L 1" ;;  Q4_K_L) echo "Q4_K_M 1" ;;
    Q5_K_L)  echo "Q5_K_M 1" ;; Q6_K_L)  echo "Q6_K 1" ;;    *)      echo "$1 0" ;;
  esac
}

for f in $files; do
  label="${f%.gguf}"; label="${label##*-}"
  td="$OUT/$name/$label"; mkdir -p "$td"
  if [ ! -f "$td/published.shas.json" ] || [ ! -f "$td/recipe.txt" ]; then
    pub="$D/$f"
    [ -s "$pub" ] || curl -sL --retry 5 -C - -o "$pub" "https://huggingface.co/$REPO/resolve/main/$f"
    "$PY" "$TOOLS/gguf_tensor_shas.py" "$pub" >/dev/null && mv "$pub.shas.json" "$td/published.shas.json"
    "$PY" "$HERE/mkrecipe.py" "$td/published.shas.json" "$td/recipe.txt" >/dev/null
    [ "${KEEP_PUBLISHED:-0}" = 1 ] || rm -f "$pub"
  fi
  read -r base ovr <<< "$(base_of "$label")"
  for variant in $VARIANTS; do
    q="$BUILD/build-$variant/bin/llama-quantize"
    [ -x "$q" ] || { echo "no $q" >&2; continue; }
    for mode in $MODES; do
      o="$td/$HOST-$variant-$mode.gguf"; args=(--imatrix "$IMAT")
      if [ "$mode" = recipe ]; then args+=(--tensor-type-file "$td/recipe.txt")
      elif [ "$ovr" = 1 ]; then args+=(--output-tensor-type q8_0 --token-embedding-type q8_0); fi
      if nice -n 10 "$q" "${args[@]}" "$FP" "$o" "$base" >/dev/null 2>&1; then
        "$PY" "$TOOLS/gguf_tensor_shas.py" "$o" >/dev/null && mv "$o.shas.json" "$td/$HOST-$variant-$mode.shas.json"
        rm -f "$o"
        r=$("$PY" "$HERE/cmp_shas.py" "$td/published.shas.json" "$td/$HOST-$variant-$mode.shas.json" --csv)
        echo "$HOST,$REPO,$label,$variant,$mode,$r" >> "$csv"
        echo "  $label $variant $mode -> $r"
      else
        rm -f "$o"; echo "$HOST,$REPO,$label,$variant,$mode,,,,FAIL," >> "$csv"; echo "  $label $variant $mode FAILED"
      fi
    done
  done
done
echo "ALLTYPES_DONE $REPO"
