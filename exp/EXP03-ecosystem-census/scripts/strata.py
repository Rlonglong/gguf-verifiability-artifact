#!/usr/bin/env python3
"""Stratify the census by the kind of representative file, and apply the strict source rule.

file_stratum (from the representative file name the census scored):
  kquant   ggml scale-search k-/i-quant (Q2_K..Q6_K, IQ*): the family Proposition 1 covers
  noscale  Q8_0 / Q4_0 (no scale search)
  notquant f16/bf16 file or vision projector (mmproj) picked by the selector
  custom   custom or mixed formats (Q1_0, PQ2_0, APEX mixes, ds4, unlabeled "Q2")
source rules:
  base_rule   the census' rule (census.py check_fp16_source): the *declared base-model repository* is public,
              ungated, and holds full-precision weights; an upper bound, since for fine-tuned derivatives the
              declared base is the parent model, not the artifact's own weights
  own_rule    the repository itself ships a full-precision (f16/bf16/f32) GGUF
  Both also require C to be satisfied (no imatrix recorded, or the imatrix file is published).
modality (from the HF pipeline_tag): "non-llm" for speech, audio, video, image-generation, embedding and
  classification tasks (NON_LLM_TAGS); everything else, including untagged repos, counts as a text or
  multimodal LLM for the restricted headline.

Usage: python3 strata.py --census ../data/census_top200.csv --out ../data/census_strata.csv
Needs network for the file listings (HF API); listings change over time (snapshot 2026-09-29).
"""
import argparse, csv, json, re, urllib.request

KQ = re.compile(r"(IQ[1-4]_[A-Z]+|Q[2-6]_K)", re.I)
FULLPREC = re.compile(r"(^|[-_./])(f16|fp16|bf16|f32)([-_.]|\.gguf$)", re.I)
CUSTOM = ("Q1_0", "PQ2_0", "APEX", "DSPARK", "-Q2-")
NON_LLM_TAGS = {"automatic-speech-recognition", "text-to-speech", "audio-classification", "feature-extraction",
                "sentence-similarity", "token-classification", "text-to-image", "image-to-image", "image-to-video",
                "image-text-to-video", "text-to-video"}


def stratum(f):
    u = f.upper()
    toks = set(re.findall(r"IQ\d_[A-Z]+|Q\d_?K|Q\d_\d|BF16|F16|F32", u))
    if len(toks) > 1 and "MMPROJ" not in u:
        return "custom"   # mixed-precision file names, e.g. ...-Q4K-Q8_0-F32.gguf
    if "MMPROJ" in u or re.search(r"(^|[-_./])(F16|BF16)\.GGUF$", u):
        return "notquant"
    if any(c in u for c in CUSTOM):
        return "custom"
    if re.search(r"Q8_0|Q4_0|[-_.]Q8\.GGUF$", u):
        return "noscale"
    if KQ.search(u) or "Q2K" in u:
        return "kquant"
    return "custom"


def model_info(repo):
    req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}", headers={"User-Agent": "Mozilla/5.0"})
    j = json.load(urllib.request.urlopen(req, timeout=30))
    return [s["rfilename"] for s in j.get("siblings", [])], j.get("pipeline_tag") or ""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--census", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.census)) if r["status"] == "ok"]
    counts, n_base, n_own, n_either, n_llm, n_llm_verifiable = {}, 0, 0, 0, 0, 0
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["repo", "quant_file", "file_stratum", "pipeline_tag", "modality", "c_satisfied",
                    "source_base_rule", "own_fullprec_gguf", "source_own_rule"])
        for r in rows:
            s = stratum(r["quant_file"]); counts[s] = counts.get(s, 0) + 1
            c_ok = r["imatrix_status"] in ("none", "not_used") or r["imatrix_retrievable"] == "True"  # no imatrix => C is fully disclosed
            base = r["source_reconstructible"] == "True"
            files, tag = model_info(r["repo"])
            own = any(x.lower().endswith(".gguf") and "mmproj" not in x.lower() and FULLPREC.search(x.split("/")[-1])
                      for x in files)
            own_rule = own and c_ok
            modality = "non-llm" if tag in NON_LLM_TAGS else "llm"
            n_base += base; n_own += own_rule; n_either += (base or own_rule)
            n_llm += modality == "llm"; n_llm_verifiable += modality == "llm" and r["bit_exact_verifiable"] == "True"
            w.writerow([r["repo"], r["quant_file"], s, tag, modality, c_ok, base, own, own_rule])
    print("strata:", counts)
    print(f"source-reconstructible: base rule {n_base}/{len(rows)}  own-GGUF rule {n_own}/{len(rows)}  either {n_either}/{len(rows)}")
    print(f"IVR restricted to text/multimodal LLMs: {n_llm_verifiable}/{n_llm}")


if __name__ == "__main__":
    main()
