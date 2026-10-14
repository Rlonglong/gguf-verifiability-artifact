#!/usr/bin/env bash
# EXP12 — the conversion hop: can the published f16 GGUF (W, with its metadata M) be re-derived from the
# model provider's safetensors? Converts Qwen/Qwen2.5-1.5B-Instruct to an f16 GGUF with
# convert_hf_to_gguf.py at the release the publisher's card names (b3772) and at b10448, twice each,
# and records sha256 of the whole file and of the tensor-data region, for comparison with the
# publisher's f16 (bartowski/Qwen2.5-1.5B-Instruct-GGUF; whole-file sha256 b6eaec3509f1d037...).
#
# Env: WORK   work dir holding llama.cpp checkouts: $WORK/b3772/llama.cpp and $WORK/llama.cpp (b10448)
#      PY     python with torch, transformers, safetensors, sentencepiece, huggingface_hub
#      OUT    output dir
set -euo pipefail
: "${WORK:?}" "${PY:?}" "${OUT:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../../.." && pwd)"
# the directory name and the model card feed general.name/license/tags/base_model in the GGUF metadata
MODEL_DIR="$OUT/src/Qwen2.5-1.5B-Instruct"; mkdir -p "$MODEL_DIR"
"$PY" - "$MODEL_DIR" <<'PYEOF'
import sys, os
from huggingface_hub import snapshot_download
d=sys.argv[1]
rev=os.environ.get("REV","989aa7980e4cf806f80c7fef2b1adb7bc71aa306")  # the Hugging Face revision the paper cites
snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", revision=rev, local_dir=d,
                  allow_patterns=["*.safetensors","*.json","merges.txt","vocab.json","LICENSE","README.md"])
print("snapshot revision", rev)
PYEOF
sha() { sha256sum "$1" | cut -d' ' -f1; }
echo "tag,run,file_sha16,tensor_data_sha16"
for spec in "b3772:$WORK/b3772/llama.cpp" "b10448:$WORK/llama.cpp"; do
  TAG="${spec%%:*}"; SRC="${spec#*:}"
  for R in 1 2; do
    f="$OUT/f16-$TAG-run$R.gguf"
    "$PY" "$SRC/convert_hf_to_gguf.py" "$MODEL_DIR" --outfile "$f" --outtype f16 >"$OUT/convert-$TAG-run$R.log" 2>&1
    t=$("$PY" "$ROOT/tools/gguf_tensor_sha.py" "$f" | cut -d' ' -f1)
    s=$(sha "$f"); echo "$TAG,$R,${s:0:16},$t"
  done
done
