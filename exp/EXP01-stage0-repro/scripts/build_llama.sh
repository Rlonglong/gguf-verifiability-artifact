#!/usr/bin/env bash
# Clone llama.cpp at TAG and build two variants on this host (arm64/clang or x86_64/gcc):
#   build-default   default flags (unpatched)
#   build-ffpoff    PR #25353 applied: -ffp-contract=off for ggml-quants.c only
# CPU-only (quantization runs on the CPU), static build to fix the binary.
#
# Usage: WORK=/path/to/work exp/EXP01-stage0-repro/scripts/build_llama.sh [TAG]
set -euo pipefail
TAG="${1:-b10448}"
: "${WORK:?set WORK, e.g. WORK=/path/to/work}"
SRC="$WORK/llama.cpp"
mkdir -p "$WORK"

# --- clone the pinned tag ---
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp.git "$SRC"
fi
cd "$SRC"
git checkout -q "$TAG" 2>/dev/null || { git fetch --depth 1 origin tag "$TAG"; git checkout -q "$TAG"; }
git checkout -q -- ggml/src/CMakeLists.txt 2>/dev/null || true  # undo a previous patch

COMMON=(-DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DGGML_METAL=OFF
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF)
# older tags (e.g. b3772) keep llama-quantize under examples/ (no tools/): enable examples to get the target
[ -d "$SRC/tools/quantize" ] || COMMON+=(-DLLAMA_BUILD_EXAMPLES=ON)

echo "== build-default (tag=$TAG) =="
cmake -S "$SRC" -B "$WORK/build-default" "${COMMON[@]}" >/dev/null
cmake --build "$WORK/build-default" -j --target llama-quantize

echo "== apply PR #25353 to ggml/src/CMakeLists.txt =="
python3 - "$SRC/ggml/src/CMakeLists.txt" <<'PY'
import sys,io
p=sys.argv[1]
s=open(p,encoding="utf-8").read()
line='set_source_files_properties(ggml-quants.c PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")\n'
if "ffp-contract=off" in s:
    print("already patched"); sys.exit(0)
anchor="            gguf.cpp)\n"   # end of add_library(ggml-base ... gguf.cpp)
if anchor in s:
    s=s.replace(anchor, anchor+"\n"+line, 1)
else:
    s=s+"\n"+line   # fallback: append at end of file (same directory scope, still effective)
open(p,"w",encoding="utf-8").write(s)
print("patched")
PY

echo "== build-ffpoff =="
cmake -S "$SRC" -B "$WORK/build-ffpoff" "${COMMON[@]}" >/dev/null
cmake --build "$WORK/build-ffpoff" -j --target llama-quantize

git checkout -q -- ggml/src/CMakeLists.txt   # restore the clean file

echo "== done =="
echo "default:  $WORK/build-default/bin/llama-quantize"
echo "ffpoff:   $WORK/build-ffpoff/bin/llama-quantize"

# --- environment record. The compiler *version* (not just its path) is a coordinate of E,
#     so write what this host actually used next to the builds: build-env-<TAG>.txt.
REC="$WORK/build-env-$TAG.txt"
{
  echo "date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "tag:     $TAG  commit: $(git -C "$SRC" rev-parse HEAD)"
  echo "uname:   $(uname -srvm)"   # no nodename: the record must stay free of hostnames
  if command -v sw_vers >/dev/null 2>&1; then
    echo "os:      macOS $(sw_vers -productVersion) (build $(sw_vers -buildVersion)); SDK $(xcrun --show-sdk-version 2>/dev/null || echo '?')"
    echo "cpu:     $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo '?')"
  else
    [ -r /etc/os-release ] && echo "os:      $(. /etc/os-release; echo "$PRETTY_NAME")"
    [ -r /proc/cpuinfo ] && echo "cpu:     $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')" \
                         && echo "isa:     $(grep -m1 flags /proc/cpuinfo | tr ' ' '\n' | grep -xE 'avx2|fma|f16c|avx512f' | tr '\n' ' ')"
  fi
  echo "cmake:   $(cmake --version | head -1)"
  echo "cc:      $(cc --version 2>/dev/null | head -1)"
  echo "c++:     $(c++ --version 2>/dev/null | head -1)"
  for b in build-default build-ffpoff; do   # what CMake itself recorded for each build
    f=$(ls "$WORK/$b"/CMakeFiles/*/CMakeCXXCompiler.cmake 2>/dev/null | head -1 || true)
    [ -n "$f" ] && echo "$b: $(grep -E 'set\(CMAKE_CXX_COMPILER(_ID|_VERSION)? "' "$f" | sed 's/^set(//; s/)$//' | tr '\n' ' ')"
  done
} | tee "$REC"
echo "environment record written to $REC"
