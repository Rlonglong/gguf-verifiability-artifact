#!/usr/bin/env python3
"""Summarize the security characterization per quant type into one table (paper Sec. VI).
Reads:
  EXP05 qcb_{TYPE}_1p5B.json        → δ_A median |Δw|/std
  EXP06 gamma_{TYPE}_1p5B.json      -> overlap (if present)
  EXP08 roc_{TYPE}_1p5B.json        -> AUC per flip budget; floor = smallest budget with AUC>=0.99
Writes the CSV and prints the table.
"""
import os, json, csv, glob, sys
ROOT=os.path.join(os.path.dirname(__file__),"..","..","..")
def load(p):
    try: return json.load(open(p,encoding="utf-8"))
    except: return None
def find(patterns):
    for pat in patterns:
        for f in glob.glob(os.path.join(ROOT,pat)): return f
    return None

TYPES=["Q4_K","Q5_K","Q6_K","IQ4_XS"]
rows=[["type","tamper_median_abs_dw_over_sigma","natural_tailfrac_median","overlap_vs_natural","auc_at_flip_0.001","floor_min_flip_with_auc_ge_0.99"]]
for T in TYPES:
    qcb=load(find([f"exp/EXP05-qcb-structure/data/qcb_{T}_1p5B.json"]))
    gam=load(find([f"exp/EXP06-gamma/data/gamma_{T}_1p5B.json"]))
    roc=load(find([f"exp/EXP08-detector-roc/data/roc_{T}_1p5B.json"]))
    dA = qcb and qcb.get("rel_magnitude_percentiles_over_std",{}).get("50")
    ov = gam and gam.get("overlap_coefficient")
    natmed = roc and roc.get("natural_tailfrac_median")
    auc001=None; floor=None
    if roc and roc.get("by_flip"):
        bf=roc["by_flip"]
        if "0.001" in bf: auc001=bf["0.001"].get("auc")
        # floor = smallest flip with auc>=0.99
        fl=sorted((float(k),v.get("auc",0)) for k,v in bf.items())
        for f,a in fl:
            if a>=0.99: floor=f; break
    rows.append([T,
                 f"{dA:.3f}" if dA is not None else "-",
                 f"{natmed:.4g}" if natmed is not None else "-",
                 f"{ov:.3f}" if ov is not None else "-",
                 f"{auc001:.3f}" if auc001 is not None else "-",
                 (f"{floor:g}" if floor is not None else ">tested") ])
out=os.path.join(ROOT,"exp/EXP08-detector-roc/data/pathb_summary.csv")
csv.writer(open(out,"w",newline="",encoding="utf-8")).writerows(rows)
for r in rows: print("  "+" | ".join(str(x) for x in r))
print("wrote",out)
