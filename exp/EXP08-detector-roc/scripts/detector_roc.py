#!/usr/bin/env python3
"""EXP08 tail-fraction statistic + ROC.
Setting: a deployed Q differs from your reference rebuild (because of benign cross-platform non-determinism, or cheating-quantizer tampering).
Per tensor, the statistic separates benign variation from nibble-flip tampering (Q != T(X)).

Score (per tensor) = tail fraction P[|Δw|/std > tau].
  natural (scale drift, dense-small): almost nothing above tau; Q4_K nibble flips (one step, ~0.27 std): about the flip rate above tau.
ROC over the cutoff; reports AUC and TPR at FPR 1%/5% for each flip budget (--flips).

Needs gguf-py + numpy; set GGUF_PY (or WORK).
Usage: detector_roc.py reference.gguf natural_rebuild.gguf --type Q4_K --tau 0.1 --flips 0.0005,0.001 --json out.json
"""
import os,sys,argparse,json
_gp=os.environ.get("GGUF_PY") or os.path.join(os.environ.get("WORK",""),"llama.cpp","gguf-py")
if _gp and os.path.isdir(_gp): sys.path.insert(0,_gp)
import numpy as np
from gguf import GGUFReader, GGMLQuantizationType as T
import gguf.quants as gq

NIBBLE={"Q4_K":(144,16,144),"Q5_K":(176,48,176),"Q6_K":(210,0,128),"IQ4_XS":(136,8,136)}

def raw(t): return np.frombuffer(t.data.tobytes(),dtype=np.uint8)

def flip(rawb, bb, ps, pe, frac, rng):
    out=bytearray(rawb.tobytes()); nblk=len(out)//bb
    total=nblk*(pe-ps)*2; k=int(total*frac)
    if k<1: return np.frombuffer(bytes(out),dtype=np.uint8)
    for c in rng.choice(total,size=k,replace=False):
        nib=c%2; off=(c//2)%(pe-ps); blk=(c//2)//(pe-ps); bi=blk*bb+ps+off
        b=out[bi]
        if nib==0:
            v=b&0xF; v2=v-1 if v==15 else v+1; out[bi]=(b&0xF0)|(v2&0xF)
        else:
            v=(b>>4)&0xF; v2=v-1 if v==15 else v+1; out[bi]=(b&0xF)|((v2&0xF)<<4)
    return np.frombuffer(bytes(out),dtype=np.uint8)

def auc(scores,labels):
    # rank-based AUC (Mann-Whitney)
    order=np.argsort(scores); ranks=np.empty(len(scores)); ranks[order]=np.arange(1,len(scores)+1)
    pos=labels==1; npos=pos.sum(); nneg=(~pos).sum()
    if npos==0 or nneg==0: return float('nan')
    return (ranks[pos].sum()-npos*(npos+1)/2)/(npos*nneg)

def roc_points(scores,labels):
    thr=np.unique(scores)[::-1]
    out=[]
    P=(labels==1).sum(); N=(labels==0).sum()
    for t in thr:
        pred=scores>=t
        tpr=((pred)&(labels==1)).sum()/P if P else 0
        fpr=((pred)&(labels==0)).sum()/N if N else 0
        out.append((fpr,tpr))
    return out

def tpr_at_fpr(points, target):
    best=0
    for fpr,tpr in points:
        if fpr<=target: best=max(best,tpr)
    return best

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("canonical"); ap.add_argument("natural_deploy")
    ap.add_argument("--type",default="Q4_K",choices=list(NIBBLE)); ap.add_argument("--tau",type=float,default=0.1)
    ap.add_argument("--flips",default="0.001,0.005,0.02"); ap.add_argument("--json")
    a=ap.parse_args()
    qtype=getattr(T,a.type); bb,ps,pe=NIBBLE[a.type]; tau=a.tau
    rc=GGUFReader(a.canonical); rn=GGUFReader(a.natural_deploy)
    nd={t.name:t for t in rn.tensors}
    rng=np.random.default_rng(7)

    # per-tensor natural scores (fixed; independent of the flip budget)
    natural=[]; canon_cache=[]
    for t in rc.tensors:
        if t.tensor_type!=qtype: continue
        tn=nd.get(t.name)
        if tn is None or tn.tensor_type!=qtype: continue
        wc=gq.dequantize(raw(t),qtype).astype(np.float64); std=wc.std() or 1e-12
        wn=gq.dequantize(raw(tn),qtype).astype(np.float64)
        m=min(wc.size,wn.size)
        natural.append(float(np.mean(np.abs(wc[:m]-wn[:m])/std > tau)))
        canon_cache.append((t.name, raw(t), wc, std))
    natural=np.array(natural)

    results={"type":a.type,"tau":tau,"n_tensors":len(natural),
             "natural_tailfrac_median":float(np.median(natural)),"by_flip":{}}
    for frac in [float(x) for x in a.flips.split(",")]:
        attack=[]
        for name,rb,wc,std in canon_cache:
            wa=gq.dequantize(flip(rb,bb,ps,pe,frac,rng),qtype).astype(np.float64)
            m=min(wc.size,wa.size)
            attack.append(float(np.mean(np.abs(wc[:m]-wa[:m])/std > tau)))
        attack=np.array(attack)
        scores=np.concatenate([natural,attack]); labels=np.concatenate([np.zeros(len(natural)),np.ones(len(attack))])
        pts=roc_points(scores,labels); A=auc(scores,labels)
        results["by_flip"][str(frac)]={
            "auc":A,"attack_tailfrac_median":float(np.median(attack)),
            "tpr_at_fpr1pct":tpr_at_fpr(pts,0.01),"tpr_at_fpr5pct":tpr_at_fpr(pts,0.05)}
        print(f"flip={frac:<6} AUC={A:.4f}  attack_tail_med={np.median(attack):.4f} (nat {np.median(natural):.4f})  "
              f"TPR@FPR1%={results['by_flip'][str(frac)]['tpr_at_fpr1pct']:.3f}  TPR@FPR5%={results['by_flip'][str(frac)]['tpr_at_fpr5pct']:.3f}")
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)),exist_ok=True)
        json.dump(results,open(a.json,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
        print("wrote",a.json)

if __name__=="__main__": main()
