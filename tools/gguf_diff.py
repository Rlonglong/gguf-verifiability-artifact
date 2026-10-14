#!/usr/bin/env python3
"""Structured GGUF diff: compare two GGUF files and locate where they differ:
  (a) the metadata region (header + KV + tensor info) or the tensor data,
  (b) which tensors / quantization types,
  (c) which field inside a k-quant block: superscale (d) / min (dmin) / blockscales (sub-block scales/mins) / nibble (quantized payload).

Floating-point divergence is expected to originate in the scale fields; nibble-flip tampering changes the payload.
Used for EXP01 (cross-machine localization) and EXP09 (published vs rebuilt).

Needs llama.cpp gguf-py (GGUFReader) + numpy, located via:
  GGUF_PY=/path/to/llama.cpp/gguf-py   (default: $WORK/llama.cpp/gguf-py)

Usage:
  python3 tools/gguf_diff.py A.gguf B.gguf [--json out.json] [--per-tensor-csv out.csv]
"""
import os, sys, argparse, json

_gp = os.environ.get("GGUF_PY") or os.path.join(os.environ.get("WORK",""), "llama.cpp", "gguf-py")
if _gp and os.path.isdir(_gp):
    sys.path.insert(0, _gp)
import numpy as np
from gguf import GGUFReader

# per type: bytes per block and the [start,end) byte range of each field. QK_K=256, ggml_half=2B, K_SCALE_SIZE=12.
# fields: superscale = super-block scale d; min = dmin; blockscales = quantized sub-block scales/mins; nibble = quantized weight payload.
LAYOUTS = {
    "Q8_0":   (34,  [("superscale",0,2), ("nibble",2,34)]),
    "Q4_0":   (18,  [("superscale",0,2), ("nibble",2,18)]),
    "Q5_0":   (22,  [("superscale",0,2), ("nibble",2,22)]),
    "Q4_K":   (144, [("superscale",0,2), ("min",2,4), ("blockscales",4,16), ("nibble",16,144)]),
    "Q5_K":   (176, [("superscale",0,2), ("min",2,4), ("blockscales",4,16), ("nibble",16,176)]),
    "Q6_K":   (210, [("nibble",0,192), ("blockscales",192,208), ("superscale",208,210)]),
    "Q3_K":   (110, [("nibble",0,96), ("blockscales",96,108), ("superscale",108,110)]),
    "Q2_K":   (84,  [("blockscales",0,16), ("nibble",16,80), ("superscale",80,82), ("min",82,84)]),
    "IQ4_XS": (136, [("superscale",0,2), ("blockscales",2,8), ("nibble",8,136)]),
}
REGIONS = ["superscale", "min", "blockscales", "nibble"]

def raw_bytes(t):
    """Return the raw bytes of a tensor (np.uint8, length n_bytes)."""
    b = t.data.tobytes()
    return np.frombuffer(b, dtype=np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--json"); ap.add_argument("--per-tensor-csv")
    ap.add_argument("--append-summary", help="append a one-line summary to this CSV (for sweeps)")
    ap.add_argument("--label", default="", help="label for the --append-summary row")
    args = ap.parse_args()

    ra, rb = GGUFReader(args.a), GGUFReader(args.b)
    off_a = int(ra.data_offset); off_b = int(rb.data_offset)

    # (a) metadata region: compare all bytes before the tensor-data offset
    with open(args.a, "rb") as f: head_a = f.read(off_a)
    with open(args.b, "rb") as f: head_b = f.read(off_b)
    meta_identical = (head_a == head_b)

    ta = {t.name: t for t in ra.tensors}
    tb = {t.name: t for t in rb.tensors}
    common = [n for n in ta if n in tb]
    only_a = [n for n in ta if n not in tb]
    only_b = [n for n in tb if n not in ta]

    per_type = {}   # type -> dict
    region_diff = {r: 0 for r in REGIONS}
    region_total = {r: 0 for r in REGIONS}
    tot_bytes = 0; tot_diff = 0; n_diff_tensors = 0
    per_tensor_rows = []

    for n in common:
        A, B = ta[n], tb[n]
        typ = A.tensor_type.name
        if A.tensor_type != B.tensor_type or A.n_bytes != B.n_bytes:
            per_tensor_rows.append((n, typ, A.n_bytes, -1, "TYPE/SIZE_MISMATCH"))
            continue
        ba, bb = raw_bytes(A), raw_bytes(B)
        m = min(len(ba), len(bb))
        diffmask = ba[:m] != bb[:m]
        nb = int(len(ba)); nd = int(diffmask.sum())
        tot_bytes += nb; tot_diff += nd
        if nd > 0: n_diff_tensors += 1
        pt = per_type.setdefault(typ, {"n":0,"n_diff":0,"total_bytes":0,"diff_bytes":0})
        pt["n"] += 1; pt["total_bytes"] += nb; pt["diff_bytes"] += nd
        if nd > 0: pt["n_diff"] += 1

        # (c) field localization (types with a known layout only)
        if typ in LAYOUTS:
            bb_, regs = LAYOUTS[typ]
            nblk = nb // bb_
            if nblk * bb_ == nb:
                blocks = diffmask[:nblk*bb_].reshape(nblk, bb_)
                for (rname, s, e) in regs:
                    region_diff[rname]  += int(blocks[:, s:e].sum())
                    region_total[rname] += nblk * (e - s)
        per_tensor_rows.append((n, typ, nb, nd, f"{nd/nb:.6g}" if nb else "0"))

    beta = tot_diff / tot_bytes if tot_bytes else 0.0
    for pt in per_type.values():
        pt["beta"] = pt["diff_bytes"]/pt["total_bytes"] if pt["total_bytes"] else 0.0

    region_dist = {}
    for r in REGIONS:
        region_dist[r] = {
            "diff_bytes": region_diff[r], "total_bytes": region_total[r],
            "beta": (region_diff[r]/region_total[r]) if region_total[r] else 0.0,
            "share_of_all_diffs": (region_diff[r]/tot_diff) if tot_diff else 0.0,
        }

    top = sorted([r for r in per_tensor_rows if isinstance(r[3],int) and r[3]>0],
                 key=lambda r: r[3], reverse=True)[:20]

    summary = {
        "file_a": args.a, "file_b": args.b,
        "metadata_region_identical": meta_identical,
        "data_offset_a": off_a, "data_offset_b": off_b,
        "n_tensors_common": len(common), "only_a": only_a, "only_b": only_b,
        "n_tensors_differing": n_diff_tensors,
        "total_data_bytes": tot_bytes, "total_diff_bytes": tot_diff, "beta_byte": beta,
        "per_type": per_type,
        "region_distribution": region_dist,
        "top_diff_tensors": [{"name":r[0],"type":r[1],"n_bytes":r[2],"diff_bytes":r[3]} for r in top],
    }

    # print summary
    print(f"A: {args.a}\nB: {args.b}")
    print(f"metadata identical: {meta_identical}  (data_offset a={off_a} b={off_b})")
    print(f"tensors: common {len(common)}, differing {n_diff_tensors};  beta_byte = {tot_diff}/{tot_bytes} = {beta:.3e}")
    if only_a or only_b: print(f"  only in A: {len(only_a)}  only in B: {len(only_b)}")
    print("per-type:")
    for typ, pt in sorted(per_type.items()):
        print(f"  {typ:8} n={pt['n']:>3} n_diff={pt['n_diff']:>3} β={pt['beta']:.3e}")
    print("by field (where the differences are):")
    for r in REGIONS:
        d = region_dist[r]
        if d["total_bytes"]:
            print(f"  {r:12} diff={d['diff_bytes']:>10} / {d['total_bytes']:>12}  beta={d['beta']:.3e}  share of all diffs={d['share_of_all_diffs']*100:.2f}%")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f: json.dump(summary, f, indent=2, ensure_ascii=False)
        print("wrote", args.json)
    if args.append_summary:
        import csv
        rd = region_dist
        fields = ["label","beta_byte","metadata_identical","n_tensors_differing","total_diff_bytes",
                  "superscale_share","min_share","blockscales_share","nibble_share",
                  "superscale_beta","min_beta","blockscales_beta","nibble_beta"]
        newfile = not os.path.exists(args.append_summary)
        os.makedirs(os.path.dirname(os.path.abspath(args.append_summary)), exist_ok=True)
        with open(args.append_summary, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if newfile: w.writerow(fields)
            w.writerow([args.label, f"{beta:.6e}", meta_identical, n_diff_tensors, tot_diff,
                        f"{rd['superscale']['share_of_all_diffs']:.4f}", f"{rd['min']['share_of_all_diffs']:.4f}",
                        f"{rd['blockscales']['share_of_all_diffs']:.4f}", f"{rd['nibble']['share_of_all_diffs']:.4f}",
                        f"{rd['superscale']['beta']:.4e}", f"{rd['min']['beta']:.4e}",
                        f"{rd['blockscales']['beta']:.4e}", f"{rd['nibble']['beta']:.4e}"])
        print("appended summary ->", args.append_summary)
    if args.per_tensor_csv:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.per_tensor_csv)), exist_ok=True)
        with open(args.per_tensor_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["tensor","type","n_bytes","diff_bytes","frac_diff"])
            for r in per_tensor_rows: w.writerow(r)
        print("wrote", args.per_tensor_csv)

if __name__ == "__main__":
    main()
