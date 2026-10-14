#!/usr/bin/env python3
"""Combine per-host quantization hash CSVs into the three EXP01 results:
 (1) same-machine determinism: are the N repeated hashes of each (machine, variant, quant_type) identical?
 (2) cross-machine, default build: do mac and linux-x86 hashes match per quant_type?
 (3) cross-machine, ffpoff build (PR #25353): do they match?

Input: one hashes CSV per host (columns: see quantize_repro.sh).
Output: printed summary; --out writes a summary CSV.

Usage:
    python3 analyze_hashes.py data/hashes_mac.csv data/hashes_linux-x86.csv --out data/stage0_summary.csv
"""
import argparse, csv, sys
from collections import defaultdict

def load(paths):
    rows=[]
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out")
    a=ap.parse_args()
    rows=load(a.csvs)
    if not rows:
        print("no rows"); sys.exit(1)

    # (1) same-machine determinism
    groups=defaultdict(set)          # (tag,variant,qt) -> set(hash)
    counts=defaultdict(int)
    for r in rows:
        k=(r["machine_tag"], r["variant"], r["quant_type"])
        groups[k].add(r["sha256"]); counts[k]+=1

    # representative hash (unique if determinism holds)
    rep={k:(next(iter(v)) if len(v)==1 else None) for k,v in groups.items()}

    print("=== (1) same-machine determinism (one hash per repeated configuration?) ===")
    det_ok=0; det_tot=0
    for k in sorted(groups):
        det_tot+=1; uniq=len(groups[k])
        ok = uniq==1
        det_ok+= 1 if ok else 0
        print(f"  {k[0]:8} {k[1]:8} {k[2]:8}  n={counts[k]:>2}  unique={uniq}  {'OK' if ok else 'NON-DETERMINISTIC'}")
    print(f"  -> {det_ok}/{det_tot} configurations deterministic")

    tags=sorted({r['machine_tag'] for r in rows})
    qts=sorted({r['quant_type'] for r in rows})
    summary=[]
    for variant in ["default","ffpoff"]:
        print(f"\n=== (2/3) cross-machine agreement [variant={variant}] ===")
        if len(tags)<2:
            print(f"  data from one machine only ({tags}); cross-machine comparison needs two.")
        for qt in qts:
            hs={t:rep.get((t,variant,qt)) for t in tags}
            present={t:h for t,h in hs.items() if h is not None}
            if len(present)<2:
                continue
            vals=set(present.values())
            match = len(vals)==1
            print(f"  {qt:8}  " + "  ".join(f"{t}={ (h[:12]+'…') if h else 'NA'}" for t,h in present.items())
                  + f"   -> {'MATCH' if match else 'DIFFER'}")
            summary.append({"variant":variant,"quant_type":qt,
                            "match":"yes" if match else "no",
                            **{f"hash_{t}":(present.get(t) or "") for t in tags}})

    # conclusion
    print("\n=== conclusion ===")
    def crossmatch(variant):
        res={}
        for qt in qts:
            hs=[rep.get((t,variant,qt)) for t in tags]
            hs=[h for h in hs if h]
            if len(hs)>=2: res[qt]= len(set(hs))==1
        return res
    d=crossmatch("default"); f=crossmatch("ffpoff")
    if d: print(f"  default cross-machine matches: {sum(d.values())}/{len(d)}")
    if f: print(f"  ffpoff  cross-machine matches: {sum(f.values())}/{len(f)}")

    if a.out and summary:
        keys=["variant","quant_type","match"]+[f"hash_{t}" for t in tags]
        with open(a.out,"w",newline="",encoding="utf-8") as fo:
            w=csv.DictWriter(fo, fieldnames=keys); w.writeheader()
            for s in summary: w.writerow(s)
        print(f"\nwrote {a.out}")

if __name__=="__main__":
    main()
