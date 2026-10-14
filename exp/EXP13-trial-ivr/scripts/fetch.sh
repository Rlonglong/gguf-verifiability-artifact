#!/usr/bin/env bash
# EXP13 — fetch the in-sample targets: each repository's own full-precision GGUF and imatrix (inputs), and,
# with WITH_PUBLISHED=1, the published Q4_K_M and Q8_0 files to compare against. Resumable (curl -C -).
# Usage: DEST=/path WITH_PUBLISHED=0|1 bash fetch.sh
set -euo pipefail
: "${DEST:?}"; WITH_PUBLISHED="${WITH_PUBLISHED:-0}"
get() { mkdir -p "$DEST/$1"; curl -sL --retry 5 -C - -o "$DEST/$1/$2" "https://huggingface.co/$1/resolve/main/$2"; echo "got $1/$2 $(stat -c %s "$DEST/$1/$2" 2>/dev/null || stat -f %z "$DEST/$1/$2")"; }
declare_targets() {
cat <<T
bartowski/Qwen_Qwen3.5-0.8B-GGUF Qwen_Qwen3.5-0.8B-bf16.gguf Qwen_Qwen3.5-0.8B-imatrix.gguf Qwen_Qwen3.5-0.8B-Q4_K_M.gguf Qwen_Qwen3.5-0.8B-Q8_0.gguf
bartowski/Qwen_Qwen3.5-2B-GGUF Qwen_Qwen3.5-2B-bf16.gguf Qwen_Qwen3.5-2B-imatrix.gguf Qwen_Qwen3.5-2B-Q4_K_M.gguf Qwen_Qwen3.5-2B-Q8_0.gguf
bartowski/Qwen_Qwen3.5-4B-GGUF Qwen_Qwen3.5-4B-bf16.gguf Qwen_Qwen3.5-4B-imatrix.gguf Qwen_Qwen3.5-4B-Q4_K_M.gguf Qwen_Qwen3.5-4B-Q8_0.gguf
bartowski/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-f16.gguf Qwen2.5-7B-Instruct.imatrix Qwen2.5-7B-Instruct-Q4_K_M.gguf Qwen2.5-7B-Instruct-Q8_0.gguf
T
}
declare_targets | while read repo w imat q4 q8; do
  get "$repo" "$imat"; get "$repo" "$w"
  if [ "$WITH_PUBLISHED" = 1 ]; then get "$repo" "$q4"; get "$repo" "$q8"; fi
done
echo FETCH_DONE
