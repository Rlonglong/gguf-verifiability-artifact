#!/usr/bin/env bash
# EXP10: quality cost of -ffp-contract=off (PR #25353).
# Quantize ONE fixed f16 GGUF to Q4_K_M with the default build and the ffpoff build,
# then measure wikitext-2 perplexity of both with the SAME (default) inference binary.
# ΔPPL between them is the measured quality cost of switching to deterministic quantization.
set -euo pipefail
export WORK="${WORK:?set WORK}"
REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
EXP="$REPO/exp/EXP10-perplexity"; DATA="$EXP/data"
mkdir -p "$DATA" "$WORK"
sha(){ shasum -a 256 "$1" | awk '{print $1}'; }
say(){ echo "[EXP10 $(date +%H:%M:%S)] $*"; }

# ---- 1. build default + ffpoff quantizers (EXP01 tooling) ----
say "building default+ffpoff (llama-quantize)"
bash "$REPO/exp/EXP01-stage0-repro/scripts/build_llama.sh" b10448

# ---- 2. build llama-perplexity (a tool; try target, else reconfigure default build) ----
say "building llama-perplexity"
if ! cmake --build "$WORK/build-default" -j --target llama-perplexity >/dev/null 2>&1; then
  say "  perplexity target absent under examples-off; reconfiguring build-ppl with tools/examples on"
  cmake -S "$WORK/llama.cpp" -B "$WORK/build-ppl" \
    -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=OFF >/dev/null
  cmake --build "$WORK/build-ppl" -j --target llama-perplexity
  PPLBIN="$WORK/build-ppl/bin/llama-perplexity"
else
  PPLBIN="$WORK/build-default/bin/llama-perplexity"
fi
[ -x "$PPLBIN" ] || { PPLBIN="$(find "$WORK" -name llama-perplexity -type f -perm +111 2>/dev/null | head -1)"; }
[ -x "$PPLBIN" ] || { say "FATAL: no llama-perplexity binary"; exit 70; }
say "  perplexity binary: $PPLBIN"

# ---- 3. obtain fixed f16 GGUF (Qwen2.5-1.5B-Instruct) ----
F16="$WORK/Qwen2.5-1.5B-Instruct-f16.gguf"
if [ ! -f "$F16" ]; then
  say "downloading f16 GGUF (~3GB)"
  curl -fL --retry 3 -o "$F16.part" \
    "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-f16.gguf?download=true"
  mv "$F16.part" "$F16"
fi
F16SHA="$(sha "$F16")"; say "  f16 sha256=$F16SHA  ($(stat -f%z "$F16") B)"

# ---- 4. quantize to Q4_K_M with both build variants (bash 3.2: no assoc arrays) ----
say "quantizing Q4_K_M (default build)"
"$WORK/build-default/bin/llama-quantize" "$F16" "$WORK/q4km-default.gguf" Q4_K_M 8 >/dev/null 2>&1
say "quantizing Q4_K_M (ffpoff build)"
"$WORK/build-ffpoff/bin/llama-quantize"  "$F16" "$WORK/q4km-ffpoff.gguf"  Q4_K_M 8 >/dev/null 2>&1
SHA_default="$(sha "$WORK/q4km-default.gguf")"
SHA_ffpoff="$(sha "$WORK/q4km-ffpoff.gguf")"
say "  default Q4_K_M sha256=$SHA_default"
say "  ffpoff  Q4_K_M sha256=$SHA_ffpoff"
say "quantized GGUFs differ? default=${SHA_default:0:12} ffpoff=${SHA_ffpoff:0:12}"

# ---- 5. wikitext-2-raw corpus (llama.cpp fetcher; fallback curl) ----
WIKI="$WORK/wikitext-2-raw/wiki.test.raw"
if [ ! -f "$WIKI" ]; then
  say "fetching wikitext-2-raw"
  ( cd "$WORK" && bash "$WORK/llama.cpp/scripts/get-wikitext-2.sh" >/dev/null 2>&1 ) || true
  [ -f "$WIKI" ] || WIKI="$(find "$WORK" -name 'wiki.test.raw' | head -1)"
fi
[ -f "$WIKI" ] || { say "FATAL: no wikitext corpus"; exit 71; }
say "  corpus: $WIKI ($(wc -l < "$WIKI") lines)"

# ---- 6. perplexity for both models (same binary, fixed threads) ----
CSV="$DATA/perplexity.csv"
echo "variant,q4km_sha256,final_ppl,ppl_stderr,ctx,threads" > "$CSV"
for V in default ffpoff; do
  say "perplexity: $V"
  "$PPLBIN" -m "$WORK/q4km-$V.gguf" -f "$WIKI" -t 8 --ctx-size 512 2>&1 | tee "$DATA/ppl_$V.log"
  # final PPL line looks like: "Final estimate: PPL = 12.3456 +/- 0.12345"
  line="$(grep -aE "Final estimate: PPL" "$DATA/ppl_$V.log" | tail -1)"
  ppl="$(echo "$line" | sed -nE 's/.*PPL = ([0-9.]+).*/\1/p')"
  err="$(echo "$line" | sed -nE 's/.*\+\/- ([0-9.]+).*/\1/p')"
  eval "vsha=\$SHA_$V"
  echo "$V,$vsha,${ppl:-NA},${err:-NA},512,8" >> "$CSV"
  say "  $V final PPL=${ppl:-NA} +/- ${err:-NA}"
done

# ---- 7. summarize delta ----
say "=== RESULT ==="
python3 - "$CSV" "$F16SHA" <<'PY'
import csv,sys,json
rows={r['variant']:r for r in csv.DictReader(open(sys.argv[1]))}
try:
    d=float(rows['default']['final_ppl']); f=float(rows['ffpoff']['final_ppl'])
    dppl=f-d; rel=100*dppl/d
    out={"f16_sha256":sys.argv[2],
         "ppl_default":d,"ppl_ffpoff":f,
         "delta_ppl":round(dppl,5),"delta_ppl_pct":round(rel,4),
         "q4km_default_sha":rows['default']['q4km_sha256'],
         "q4km_ffpoff_sha":rows['ffpoff']['q4km_sha256'],
         "quantized_bytes_differ": rows['default']['q4km_sha256']!=rows['ffpoff']['q4km_sha256']}
    print(json.dumps(out,indent=2))
    json.dump(out,open(sys.argv[1].replace('perplexity.csv','summary.json'),'w'),indent=2)
    print(f"\nPPL default={d:.4f}  ffpoff={f:.4f}  Δ={dppl:+.4f} ({rel:+.3f}%)")
except Exception as e:
    print("could not summarize:",e)
PY
say "done"
