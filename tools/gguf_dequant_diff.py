#!/usr/bin/env python3
"""EXP04 dequantized-space measure: compare two GGUFs in dequantized weight space.
(One changed scale byte can move up to 256 dequantized weights, so perturbations are measured in weight space, not byte space.)

For every common quantized tensor, dequantize both sides to fp32 and compute the per-weight difference Δw:
  ρ (L0)   fraction of weights that differ
  δ_L1     mean|Δw|
  δ_L2     rms(Δw)
  δ_Linf   max|Δw|
  relative  |Δw| / per-tensor weight std (comparable across tensors)
  distrib.  percentiles + histogram of |Δw|/std (the natural-variance fingerprint R_nat)

Natural differences are scale-driven: a few changed scales shift all 256 weights of a block slightly
 -> a dense-small signature. Nibble-flip tampering is sparse-large; the shape of the distribution separates them.

Needs gguf-py + numpy; set GGUF_PY (or WORK) to locate gguf-py.
Usage:
  python3 tools/gguf_dequant_diff.py A.gguf B.gguf [--json out.json] [--per-tensor-csv t.csv] [--max-tensors N]
"""
import os, sys, argparse, json
_gp=os.environ.get("GGUF_PY") or os.path.join(os.environ.get("WORK",""),"llama.cpp","gguf-py")
if _gp and os.path.isdir(_gp): sys.path.insert(0,_gp)
import numpy as np
from gguf import GGUFReader, GGMLQuantizationType
import gguf.quants as gq

QUANT_TYPES={GGMLQuantizationType.Q4_K,GGMLQuantizationType.Q5_K,GGMLQuantizationType.Q6_K,
             GGMLQuantizationType.Q8_0,GGMLQuantizationType.Q4_0,GGMLQuantizationType.IQ4_XS,
             GGMLQuantizationType.Q2_K,GGMLQuantizationType.Q3_K}

def raw(t): return np.frombuffer(t.data.tobytes(),dtype=np.uint8)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--json"); ap.add_argument("--per-tensor-csv"); ap.add_argument("--max-tensors",type=int,default=0)
    args=ap.parse_args()
    ra,rb=GGUFReader(args.a),GGUFReader(args.b)
    tb={t.name:t for t in rb.tensors}

    # global accumulators
    tot_w=0; tot_diff=0
    sum_abs=0.0; sum_sq=0.0; gmax=0.0
    rel_samples=[]   # sample of |Δw|/std (bounded memory)
    per_type={}
    rows=[]
    n=0
    for ta in ra.tensors:
        if ta.tensor_type not in QUANT_TYPES: continue
        tbb=tb.get(ta.name)
        if tbb is None or tbb.tensor_type!=ta.tensor_type: continue
        n+=1
        if args.max_tensors and n>args.max_tensors: break
        wa=gq.dequantize(raw(ta),ta.tensor_type).astype(np.float64)
        wb=gq.dequantize(raw(tbb),tbb.tensor_type).astype(np.float64)
        m=min(wa.size,wb.size); wa=wa[:m]; wb=wb[:m]
        d=np.abs(wa-wb)
        nz=d>0
        nd=int(nz.sum()); nw=int(d.size)
        std=float(np.std(wa)) or 1e-12
        tot_w+=nw; tot_diff+=nd
        sum_abs+=float(d.sum()); sum_sq+=float((d*d).sum()); gmax=max(gmax,float(d.max()))
        # sample relative magnitudes (non-zero only)
        if nd>0:
            rd=(d[nz]/std)
            if rd.size>4000: rd=rd[np.random.default_rng(0).choice(rd.size,4000,replace=False)] if False else rd[::max(1,rd.size//4000)]
            rel_samples.append(rd.astype(np.float32))
        typ=ta.tensor_type.name
        pt=per_type.setdefault(typ,{"n":0,"weights":0,"diff":0,"sum_abs":0.0,"sum_sq":0.0,"max":0.0})
        pt["n"]+=1; pt["weights"]+=nw; pt["diff"]+=nd; pt["sum_abs"]+=float(d.sum()); pt["sum_sq"]+=float((d*d).sum()); pt["max"]=max(pt["max"],float(d.max()))
        rows.append((ta.name,typ,nw,nd,f"{nd/nw:.6g}",f"{d.mean():.4e}",f"{np.sqrt((d*d).mean()):.4e}",f"{d.max():.4e}",f"{std:.4e}"))

    rho=tot_diff/tot_w if tot_w else 0.0
    l1=sum_abs/tot_w if tot_w else 0.0
    l2=np.sqrt(sum_sq/tot_w) if tot_w else 0.0
    rel=np.concatenate(rel_samples) if rel_samples else np.array([0.0])
    pcts={p:float(np.percentile(rel,p)) for p in [50,90,99,99.9,100]}
    # histogram of |Δw|/std
    hist_edges=[0,0.001,0.01,0.05,0.1,0.25,0.5,1.0,2.0,1e9]
    hist,_=np.histogram(rel,bins=hist_edges)
    for pt in per_type.values():
        w=pt["weights"]; pt["rho"]=pt["diff"]/w if w else 0; pt["L1"]=pt["sum_abs"]/w if w else 0; pt["L2"]=float(np.sqrt(pt["sum_sq"]/w)) if w else 0

    summary={
        "file_a":args.a,"file_b":args.b,
        "tensors_compared":n,"total_weights":tot_w,"weights_differing":tot_diff,
        "rho_L0":rho,"delta_L1":l1,"delta_L2":l2,"delta_Linf":gmax,
        "rel_magnitude_percentiles_over_std":pcts,
        "rel_hist_edges_over_std":hist_edges,"rel_hist_counts":[int(x) for x in hist],
        "per_type":per_type,
    }
    print(f"A:{args.a}\nB:{args.b}")
    print(f"dequantized space: tensors={n} weights={tot_w}  rho(L0)={rho:.4e}  d_L1={l1:.4e}  d_L2={l2:.4e}  d_Linf={gmax:.4e}")
    print("|Δw|/std percentiles:", {k:round(v,4) for k,v in pcts.items()})
    print("|Δw|/std histogram:")
    for i in range(len(hist)):
        lo,hi=hist_edges[i],hist_edges[i+1]
        print(f"  [{lo:>6}, {hi:<6}) : {hist[i]}")
    print("per-type:")
    for typ,pt in sorted(per_type.items()):
        print(f"  {typ:8} n={pt['n']:>3} ρ={pt['rho']:.3e} δ_L1={pt['L1']:.3e} δ_L2={pt['L2']:.3e} max={pt['max']:.3e}")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)),exist_ok=True)
        json.dump(summary,open(args.json,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
        print("wrote",args.json)
    if args.per_tensor_csv:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.per_tensor_csv)),exist_ok=True)
        w=csv.writer(open(args.per_tensor_csv,"w",newline="",encoding="utf-8"))
        w.writerow(["tensor","type","weights","diff","rho","delta_L1","delta_L2","delta_Linf","weight_std"])
        w.writerows(rows)
        print("wrote",args.per_tensor_csv)

if __name__=="__main__": main()
