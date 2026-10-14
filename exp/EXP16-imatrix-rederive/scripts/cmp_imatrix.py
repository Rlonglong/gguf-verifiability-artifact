#!/usr/bin/env python3
"""EXP16 — compare two GGUF-format importance matrices (imatrix) entry by entry.

An imatrix stores, per weight tensor, `<name>.in_sum2` (the sum over calibration tokens of the squared
input activations, one value per input column) and `<name>.counts` (how many tokens contributed).
Quantization uses the per-column mean in_sum2/counts to weight the scale search, so both matter.

Usage: python3 cmp_imatrix.py REFERENCE.gguf CANDIDATE.gguf [--json out.json]
Needs gguf-py + numpy (set GGUF_PY to llama.cpp/gguf-py).
"""
import os, sys, json, argparse
_gp = os.environ.get("GGUF_PY")
if _gp and os.path.isdir(_gp):
    sys.path.insert(0, _gp)
import numpy as np
from gguf import GGUFReader


def load(path):
    r = GGUFReader(path)
    meta = {}
    for k in ("imatrix.chunk_count", "imatrix.chunk_size", "imatrix.datasets"):
        f = r.fields.get(k)
        if f is not None:
            try:
                meta[k] = f.contents()
            except Exception:
                meta[k] = None
    ent = {}
    for t in r.tensors:
        arr = np.array(t.data, dtype=np.float32).ravel()
        if t.name.endswith(".in_sum2"):
            ent.setdefault(t.name[:-8], {})["sum2"] = arr
        elif t.name.endswith(".counts"):
            ent.setdefault(t.name[:-7], {})["counts"] = arr
    return meta, ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("cand"); ap.add_argument("--json")
    a = ap.parse_args()
    mr, er = load(a.ref); mc, ec = load(a.cand)
    names = sorted(set(er) | set(ec))
    only_r = [n for n in names if n not in ec]; only_c = [n for n in names if n not in er]
    both = [n for n in names if n in er and n in ec]
    n_counts_eq = n_sum_bitexact = 0
    rel_max, rel_med, mean_rel_med = [], [], []
    for n in both:
        R, C = er[n], ec[n]
        if np.array_equal(R["counts"], C["counts"]):
            n_counts_eq += 1
        rs, cs = R["sum2"], C["sum2"]
        if rs.shape == cs.shape and np.array_equal(rs.view(np.uint32), cs.view(np.uint32)):
            n_sum_bitexact += 1
        if rs.shape == cs.shape:
            d = np.abs(rs.astype(np.float64) - cs) / np.maximum(np.abs(rs.astype(np.float64)), 1e-30)
            rel_max.append(float(d.max())); rel_med.append(float(np.median(d)))
            mr_ = rs / max(float(R["counts"][0]), 1.0); mc_ = cs / max(float(C["counts"][0]), 1.0)
            dm = np.abs(mr_.astype(np.float64) - mc_) / np.maximum(np.abs(mr_.astype(np.float64)), 1e-30)
            mean_rel_med.append(float(np.median(dm)))
    out = {"ref": os.path.basename(a.ref), "cand": os.path.basename(a.cand),
           "ref_meta": {k: (str(v) if v is not None else None) for k, v in mr.items()},
           "cand_meta": {k: (str(v) if v is not None else None) for k, v in mc.items()},
           "n_entries_ref": len(er), "n_entries_cand": len(ec), "only_in_ref": only_r, "only_in_cand": only_c,
           "n_common": len(both), "n_counts_equal": n_counts_eq, "n_in_sum2_bitexact": n_sum_bitexact,
           "rel_diff_in_sum2": {"median_of_medians": float(np.median(rel_med)) if rel_med else None,
                                "max_of_max": float(np.max(rel_max)) if rel_max else None},
           "rel_diff_mean_importance_median": float(np.median(mean_rel_med)) if mean_rel_med else None}
    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
