#!/usr/bin/env python3
"""Pattern scan of the model cards (README.md) of all in-scope census repositories.

Complements the 12-card LLM audit: finds cards that state the llama.cpp release/commit used, and
cards that state any execution-environment coordinate (compiler, target architecture, FP-contraction).
Every version hit was then classified by hand; hits that name a release only as an *inference*
requirement ("requires llama.cpp bNNNN or newer", benchmark builds) are listed in INFERENCE_ONLY.

Usage:
  python3 card_scan.py --census ../data/census_top200.csv --cache cards/ --out ../data/card_version_scan.csv
Needs network access to huggingface.co (fetches <repo>/raw/main/README.md; cached under --cache).
Cards change over time, so a re-run may differ from the committed snapshot (scanned 2026-09-29).
"""
import argparse, csv, os, re, urllib.request, concurrent.futures as cf

VERSION = re.compile(
    r"(release[s]?\s*(?:<[^>]*>\s*)?(?:/tag/)?b\d{3,5}|releases/tag/b\d{3,5}"
    r"|llama\.cpp[^\n]{0,60}\bb\d{4,5}\b|llama\.cpp[^\n]{0,60}commit\s*[`']?[0-9a-f]{7,40}"
    r"|\bversion\s*b\d{4})", re.I)
ENV = re.compile(r"(ffp-contract|-march=|gcc\s*\d|clang\s*\d|compiled with|build flags|x86_64|aarch64|arm64)", re.I)

# Manual classification (2026-09-29): version pattern present, but the release is stated as an
# inference requirement or benchmark build, not as the build that produced the quantized files.
INFERENCE_ONLY = {
    "Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF",          # "build from source or download the most recent release"
    "yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF",  # "verified working" inference build
    "meta-models/Muse-Glimmer-30B-GGUF",            # "you need llama.cpp build bNNNN or newer"
    "unsloth/Laguna-S-2.1-GGUF",                    # "support is available starting from release bNNNN"
    "mudler/Qwen3.6-35B-A3B-APEX-GGUF",             # "benchmarks run with llama.cpp bNNNN"
}
# Platform terms (Apple Silicon, Metal, AVX, NEON, MSVC, -march, "built/quantized on"): 60 in-scope cards match.
# Manual review (2026-09-29): every hit refers to inference or performance (tokens/watt on Apple silicon,
# "runs on CPU, CUDA, Metal", inference build flags such as -DGGML_CUDA=OFF, one pasted inference
# system_info line), none to the build that produced the quantized files.
PLATFORM = re.compile(r"(apple silicon|\bavx[0-9a-z_]*|\bmsvc\b|\bneon\b|\bmetal\b|-march|compiled on|built on|"
                      r"quantized on|quantised on)", re.I)
# Manual classification of ENV hits: all were "aarch64" in ARM inference-speed notes, not a record
# of the quantization environment.
ENV_NOT_QUANT_ENV = {"bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "bartowski/Qwen2.5-7B-Instruct-GGUF"}


def fetch(repo, cache):
    fn = os.path.join(cache, repo.replace("/", "__") + ".md")
    if os.path.exists(fn):
        return open(fn, encoding="utf-8", errors="ignore").read()
    req = urllib.request.Request(f"https://huggingface.co/{repo}/raw/main/README.md",
                                 headers={"User-Agent": "Mozilla/5.0"})
    try:
        t = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
    except Exception as e:  # missing card counts as "no statement"
        t = ""
    open(fn, "w", encoding="utf-8").write(t)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", required=True)
    ap.add_argument("--cache", default="cards")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    rows = [r for r in csv.DictReader(open(a.census)) if r["status"] == "ok"]
    with cf.ThreadPoolExecutor(8) as ex:
        cards = dict(zip([r["repo"] for r in rows], ex.map(lambda r: fetch(r["repo"], a.cache), rows)))
    n_ver = n_env = n_plat = 0
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["repo", "card_pattern_hit", "states_quantization_release", "match_excerpt", "states_env_coordinate",
                    "mentions_platform_terms"])
        for r in rows:
            t = cards[r["repo"]]
            hits = sorted({m if isinstance(m, str) else m[0] for m in VERSION.findall(t)})
            q = bool(hits) and r["repo"] not in INFERENCE_ONLY
            e = bool(ENV.search(t)) and r["repo"] not in ENV_NOT_QUANT_ENV
            plat = bool(PLATFORM.search(t))
            n_ver += q; n_env += e; n_plat += plat
            w.writerow([r["repo"], bool(hits), q, "|".join(hits)[:160], e, plat])
    print(f"in-scope cards: {len(rows)}  state quantization release: {n_ver}  state env coordinate: {n_env}  "
          f"mention platform terms (all inference/performance on manual review): {n_plat}")


if __name__ == "__main__":
    main()
