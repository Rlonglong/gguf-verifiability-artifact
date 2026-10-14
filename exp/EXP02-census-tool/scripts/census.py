#!/usr/bin/env python3
"""Census scorer for GGUF independent verifiability.
For a Hugging Face GGUF repo, combines the HF API with the GGUF header (HTTP range, no full download):

  fp16_source        the declared base-model repository is public, ungated, and holds full-precision weights
                     (safetensors/bin or an f16/bf16 GGUF); the repository's own files are not checked
  llamacpp_version   quantizer / llama.cpp release identifiable from metadata (no standard GGUF field)
  imatrix_provenance imatrix provenance recorded (quantize.imatrix.* keys)
  imatrix_retrievable the .imatrix file itself is retrievable (often only a local path is recorded)
  environment        execution environment (compiler, CPU, flags) recorded (no standard GGUF field)
  => structurally_verifiable = fp16_source AND llamacpp_version AND imatrix_retrievable
     (necessary for bit-exact re-derivation)

Usage:
  python3 census.py --repos bartowski/Qwen2.5-1.5B-Instruct-GGUF unsloth/... --out rows.csv
  python3 census.py --top 50 --out rows.csv          # the N most-downloaded GGUF repos
"""
import sys, os, json, csv, argparse, urllib.request, urllib.parse

HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
import gguf_header_probe as ghp

UA={"User-Agent":"vq-census/0.1"}
KNOWN_CALIB={"calibration_datav3","groups_merged","wiki.train","wikitext","c4","imatrix.dat"}

def hf_get(path):
    url=f"https://huggingface.co/api/{path}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=30) as r:
            return json.load(r)
    except Exception as e:
        return {"__error__":str(e)}

def top_gguf_repos(n):
    out=[]; page=0
    while len(out)<n:
        d=hf_get(f"models?filter=gguf&sort=downloads&direction=-1&limit=100&skip={page*100}")
        if not isinstance(d,list) or not d: break
        out.extend(d); page+=1
        if page>5: break
    return out[:n]

def pick_quant_file(siblings):
    ggufs=[s["rfilename"] for s in siblings if s["rfilename"].lower().endswith(".gguf")]
    quants=[f for f in ggufs if not any(k in f.lower() for k in ("f16","bf16","fp16","f32"))]
    # prefer Q4_K_M; else the first quant; first shard if sharded
    def score(f):
        fl=f.lower(); s=0
        if "q4_k_m" in fl: s+=10
        if "00001-of" in fl or "-of-" not in fl: s+=1
        return s
    cands=quants or ggufs
    if not cands: return None
    return sorted(cands, key=score, reverse=True)[0]

def base_model_of(info):
    cd=info.get("cardData") or {}
    bm=cd.get("base_model")
    if isinstance(bm,list): bm=bm[0] if bm else None
    if not bm:
        for t in info.get("tags",[]):
            if t.startswith("base_model:") and ":quantized:" not in t:
                bm=t.split("base_model:",1)[1]; break
    return bm

def check_fp16_source(base_repo):
    if not base_repo: return (False,"no base_model recorded")
    info=hf_get(f"models/{urllib.parse.quote(base_repo)}")
    if "__error__" in info or info.get("error"): return (False,f"base repo not accessible")
    if info.get("gated"): return (False,f"base gated={info.get('gated')}")
    sib=[s["rfilename"] for s in info.get("siblings",[])]
    has_w=any(f.endswith((".safetensors",".bin")) for f in sib)
    has_gguf_f16=any(f.lower().endswith(".gguf") and any(k in f.lower() for k in ("f16","bf16")) for f in sib)
    if has_w or has_gguf_f16: return (True,"public weights available")
    return (False,"base repo has no fp16 weight file")

def score_repo(repo):
    info=hf_get(f"models/{urllib.parse.quote(repo)}")
    row={"repo":repo,"downloads":info.get("downloads"),"likes":info.get("likes"),
         "gated":info.get("gated")}
    if "__error__" in info or info.get("error"):
        row["status"]="repo_error"; return row
    sib=info.get("siblings",[])
    qf=pick_quant_file(sib)
    row["quant_file"]=qf
    if not qf:
        row["status"]="no_gguf"; return row
    url=f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(qf)}"
    hdr=ghp.probe(url=url, max_mb=16)
    if "error" in hdr:
        row["status"]="header_error:"+hdr.get("error","?");
        v=None
    else:
        v=ghp.verifiability(hdr["kv"])
    base=base_model_of(info)
    row["base_model"]=base
    fp16_ok,fp16_why=check_fp16_source(base)
    row["fp16_source"]=fp16_ok; row["fp16_note"]=fp16_why
    if not v:
        for k in ("imatrix_status","imatrix_retrievable","env_version_recorded",
                  "bit_exact_verifiable","source_reconstructible","primary_gap"):
            row[k]=None
        row["status"]="header_unavailable"; return row

    # --- imatrix status (tri-state, so that "no imatrix used" is not counted as a gap) ---
    imf=v.get("imatrix_file"); ds=str(v.get("imatrix_dataset") or "").lower()
    sibnames=[s["rfilename"].lower() for s in sib]
    # published imatrix file: *.imatrix or a file name containing "imatrix" (covers GGUF-format imatrix files)
    imx_file_published = any(("imatrix" in n or n.endswith(".imatrix")) for n in sibnames) or \
        (imf and os.path.basename(str(imf)).lower() in sibnames)
    if not v["has_imatrix_provenance"]:
        imatrix_status="none"          # no imatrix recorded: treated as C = empty (ambiguous if --hide-imatrix was used)
        imatrix_ok=True                # not a gap (ambiguity noted)
    elif imx_file_published:
        imatrix_status="published"; imatrix_ok=True
    else:
        imatrix_status="recorded_local_only"   # only a local path -> C not retrievable
        imatrix_ok=False
    row["imatrix_status"]=imatrix_status
    row["imatrix_retrievable"]=imatrix_ok
    row["calib_dataset"]=v.get("imatrix_dataset")

    # --- environment/version: no standard GGUF field ---
    row["env_version_recorded"]=v["records_env_version"]
    row["env_keys"]=";".join(v["env_version_keys"]) if v["env_version_keys"] else ""
    row["quant_version"]=v["quantization_version"]

    # --- scoring ---
    # source_reconstructible (ignores E: are weights and imatrix available?)
    row["source_reconstructible"]= bool(row["fp16_source"] and imatrix_ok)
    # bit_exact_verifiable (additionally requires a recorded build environment)
    row["bit_exact_verifiable"]= bool(row["source_reconstructible"] and row["env_version_recorded"])
    gaps=[]
    if not row["fp16_source"]: gaps.append("fp16")
    if not imatrix_ok: gaps.append("imatrix")
    if not row["env_version_recorded"]: gaps.append("env/version")
    row["primary_gap"]=";".join(gaps) if gaps else "none"
    row["status"]="ok"
    return row

FIELDS=["repo","downloads","likes","gated","quant_file","base_model","fp16_source","fp16_note",
        "imatrix_status","imatrix_retrievable","calib_dataset","env_version_recorded","env_keys",
        "quant_version","source_reconstructible","bit_exact_verifiable","primary_gap","status"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="*", default=[])
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--repos-file", help="frozen list, one repo id per line (lines starting with # are comments)")
    ap.add_argument("--out", default="census_rows.csv")
    a=ap.parse_args()
    repos=list(a.repos)
    if a.repos_file:
        repos+= [ln.strip() for ln in open(a.repos_file,encoding="utf-8") if ln.strip() and not ln.startswith("#")]
    if a.top:
        for m in top_gguf_repos(a.top): repos.append(m["id"])
    # de-duplicate, keeping order
    seen=set(); repos=[r for r in repos if not (r in seen or seen.add(r))]
    rows=[]
    for i,repo in enumerate(repos,1):
        r=score_repo(repo); rows.append(r)
        print(f"[{i}/{len(repos)}] {repo}  bitexact={r.get('bit_exact_verifiable')} "
              f"src={r.get('source_reconstructible')}  gap={r.get('primary_gap','?')}  status={r['status']}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in FIELDS})
    ok=[r for r in rows if r.get("status")=="ok"]
    be=[r for r in ok if r.get("bit_exact_verifiable")]
    src=[r for r in ok if r.get("source_reconstructible")]
    pct=lambda n: (100*n/len(ok) if ok else 0)
    # gap breakdown
    from collections import Counter
    gapc=Counter()
    for r in ok:
        for g in str(r.get("primary_gap","")).split(";"):
            if g and g!="none": gapc[g]+=1
    print(f"\n=== results (n={len(ok)}) ===")
    print(f"  bit-exact verifiable (needs recorded environment): {len(be)}/{len(ok)}  ({pct(len(be)):.1f}%)")
    print(f"  source-reconstructible (ignores E):              {len(src)}/{len(ok)}  ({pct(len(src)):.1f}%)")
    print(f"  gaps: " + ", ".join(f"{k}={v}" for k,v in gapc.most_common()))
    print(f"  wrote {a.out}")

if __name__=="__main__": main()
