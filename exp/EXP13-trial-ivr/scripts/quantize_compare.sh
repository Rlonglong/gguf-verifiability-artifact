#!/usr/bin/env bash
# EXP13 — for one target repo, re-derive the published Q4_K_M (imatrix) and Q8_0 in each build
# environment, in up to two modes, and record per-tensor sha256 of every output:
#   plain  : the card's information only  (release + imatrix + the stock Q4_K_M/Q8_0 preset)
#   recipe : plain + the published file's exact per-tensor type map, pinned via --tensor-type-file
# Comparing plain vs recipe isolates the two card-absent coordinates: the build environment (FP
# contraction etc.) and the quantization recipe. Aggregation is done later by aggregate.py.
#
# Usage:
#   TARGET=DEST/bartowski__repo  OUT=results  RELEASE=b9222  HOST=mac-arm64 \
#   ENVS=$'armclang:/path/to/b9222'  [MODE=both|plain|recipe]  [RECIPE_DIR=dir]  [KEEP=labels] \
#   bash quantize_compare.sh
# ENVS is newline-separated "label:builddir"; builddir holds build-default/ and build-ffpoff/.
# RECIPE_DIR: read recipe_<quant>.txt from here (for hosts without the published files); default OUT/<name>.
set -euo pipefail
: "${TARGET:?}"; : "${OUT:?}"; : "${ENVS:?}"; RELEASE="${RELEASE:-unknown}"; HOST="${HOST:-host}"
MODE="${MODE:-both}"
HERE="$(cd "$(dirname "$0")" && pwd)"; TOOLS="$(cd "$HERE/../../.." && pwd)/tools"
SHAS="$TOOLS/gguf_tensor_shas.py"; MKR="$HERE/mkrecipe.py"; PY="${PY:-python3}"

shopt -s nullglob
fp=("$TARGET"/*bf16.gguf "$TARGET"/*f16.gguf); FP="${fp[0]:-}"
imat=("$TARGET"/*-imatrix.gguf "$TARGET"/*.imatrix); IMAT="${imat[0]:-}"
pq4=("$TARGET"/*Q4_K_M.gguf); PQ4="${pq4[0]:-}"
pq8=("$TARGET"/*Q8_0.gguf); PQ8="${pq8[0]:-}"
[ -n "$FP" ] || { echo "no fp16/bf16 in $TARGET" >&2; exit 1; }
[ -n "$IMAT" ] || { echo "no imatrix in $TARGET" >&2; exit 1; }
name="$(basename "$TARGET")"
RECIPE_DIR="${RECIPE_DIR:-$OUT/$name}"; mkdir -p "$OUT/$name" "$RECIPE_DIR"
idx="$OUT/index.csv"
[ -f "$idx" ] || echo "host,target,release,env,quant,shas_json" > "$idx"
echo "target=$name fp=$(basename "$FP") imat=$(basename "$IMAT") pubQ4=${PQ4:+y} pubQ8=${PQ8:+y} MODE=$MODE"

emit() { "$PY" "$SHAS" "$1" >&2; echo "$HOST,$name,$RELEASE,$2,$3,${1#$OUT/}.shas.json" >> "$idx"; }

# published reference shas + recipe files (only where the published files are present)
pub() { local src="$1" q="$2"; local dst="$OUT/$name/published/$q.gguf.shas.json"; mkdir -p "$OUT/$name/published"
  [ -f "$dst" ] || { "$PY" "$SHAS" "$src" >&2 && mv "$src.shas.json" "$dst"; }
  echo "$HOST,$name,published,published,$q,$name/published/$q.gguf.shas.json" >> "$idx"
  [ -f "$RECIPE_DIR/recipe_$q.txt" ] || "$PY" "$MKR" "$dst" "$RECIPE_DIR/recipe_$q.txt" >&2; }
[ -n "$PQ4" ] && pub "$PQ4" Q4_K_M
[ -n "$PQ8" ] && pub "$PQ8" Q8_0

quant() {   # builddir-bin, out.gguf, Q4_K_M|Q8_0, label, recipefile-or-empty
  local q="$1" o="$2" qt="$3" lbl="$4" rf="$5"; local args=()
  [ "$qt" = Q4_K_M ] && args+=(--imatrix "$IMAT")
  [ -n "$rf" ] && [ -f "$rf" ] && args+=(--tensor-type-file "$rf")
  # bash 3.2 (macOS) errors on "${args[@]}" when empty under set -u; guard with the :+ form
  if nice -n 10 "$q" ${args[@]+"${args[@]}"} "$FP" "$o" "$qt" >/dev/null 2>&1; then emit "$o" "$lbl" "$qt"
  else echo "  $lbl $qt failed" >&2; fi
}

while IFS= read -r spec; do
  [ -n "$spec" ] || continue
  label="${spec%%:*}"; bdir="${spec#*:}"
  for variant in default ffpoff; do
    q="$bdir/build-$variant/bin/llama-quantize"; [ -x "$q" ] || q="$bdir/build-$variant/bin/quantize"
    [ -x "$q" ] || { echo "  skip $label-$variant: no binary" >&2; continue; }
    od="$OUT/$name/$label-$variant"; mkdir -p "$od"
    if [ "$MODE" = plain ] || [ "$MODE" = both ]; then
      echo "== $label-$variant plain =="
      quant "$q" "$od/Q4_K_M.gguf" Q4_K_M "$label-$variant" ""
      quant "$q" "$od/Q8_0.gguf"   Q8_0   "$label-$variant" ""
    fi
    if [ "$MODE" = recipe ] || [ "$MODE" = both ]; then
      echo "== $label-$variant recipe =="
      quant "$q" "$od/Q4_K_M.recipe.gguf" Q4_K_M "$label-$variant-recipe" "$RECIPE_DIR/recipe_Q4_K_M.txt"
      quant "$q" "$od/Q8_0.recipe.gguf"   Q8_0   "$label-$variant-recipe" "$RECIPE_DIR/recipe_Q8_0.txt"
    fi
    case " ${KEEP:-} " in *" $label "*) : ;; *) rm -f "$od"/*.gguf ;; esac
  done
done <<< "$ENVS"
echo "QC_DONE $name"
