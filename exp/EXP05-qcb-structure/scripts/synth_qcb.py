#!/usr/bin/env python3
"""EXP05 synthetic cheating-quantizer perturbation (nibble flips).
Threat: a cheating quantizer ships Q != T(X), keeping scales/mins and changing only low-bit codes
(moving chosen weights by +-1 quantization step). This differs from Mind the Gap (adversarial W, Q = T(X)),
which re-derivation cannot detect by definition.

Method: in the low-bit code planes of a reference GGUF, change uniformly chosen 4-bit codes by +-1,
dequantize before/after, and record the per-weight change |Δw|/std (same unit as EXP04).
Writes the distribution for EXP06 (overlap) and uses the same ranges as EXP08.

Supports Q4_K, Q5_K, Q6_K, IQ4_XS. Needs gguf-py + numpy.
Usage:
  python3 synth_qcb.py canonical.gguf --type Q4_K --flip-frac 0.01 --json out.json
"""
import os,sys,argparse,json
_gp=os.environ.get("GGUF_PY") or os.path.join(os.environ.get("WORK",""),"llama.cpp","gguf-py")
if _gp and os.path.isdir(_gp): sys.path.insert(0,_gp)
import numpy as np
from gguf import GGUFReader, GGMLQuantizationType as T
import gguf.quants as gq

# (block_bytes, payload_start, payload_end): payload = the low-bit code plane (qs / ql)
NIBBLE_REGION={"Q4_K":(144,16,144),"Q5_K":(176,48,176),"Q6_K":(210,0,128),"IQ4_XS":(136,8,136)}

def raw(t): return np.frombuffer(t.data.tobytes(),dtype=np.uint8)

def flip_nibbles(arr, bb, ps, pe, k, rng):
    """Pick k nibbles in the payload and move each by one step within [0,15]. Returns the modified bytearray."""
    out=bytearray(arr.tobytes())
    nblk=len(out)//bb
    payload_positions=[]  # (byte_index, which_nibble) candidates
    # for efficiency: sample block, byte within the payload, and nibble directly
    total=nblk*(pe-ps)*2   # two nibbles per byte
    k=min(k,total)
    chosen=rng.choice(total, size=k, replace=False)
    for c in chosen:
        nib=c%2; bytepos_in_payload=(c//2)%(pe-ps); blk=(c//2)//(pe-ps)
        bidx=blk*bb+ps+bytepos_in_payload
        b=out[bidx]
        if nib==0:
            v=b&0x0F
            v2=v-1 if v==15 else v+1 if v==0 else v+1
            out[bidx]=(b&0xF0)|(v2&0x0F)
        else:
            v=(b>>4)&0x0F
            v2=v-1 if v==15 else v+1 if v==0 else v+1
            out[bidx]=(b&0x0F)|((v2&0x0F)<<4)
    return np.frombuffer(bytes(out),dtype=np.uint8)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("canonical")
    ap.add_argument("--type",default="Q4_K",choices=list(NIBBLE_REGION))
    ap.add_argument("--flip-frac",type=float,default=0.01,help="fraction of payload nibbles to flip")
    ap.add_argument("--max-tensors",type=int,default=0)
    ap.add_argument("--json")
    a=ap.parse_args()
    qtype=getattr(T,a.type); bb,ps,pe=NIBBLE_REGION[a.type]
    rng=np.random.default_rng(12345)

    r=GGUFReader(a.canonical)
    rel=[]  # |Δw|/std of changed weights only
    tot_w=0; tot_diff=0; sum_abs=0.0; sum_sq=0.0; gmax=0.0; n=0
    for t in r.tensors:
        if t.tensor_type!=qtype: continue
        n+=1
        if a.max_tensors and n>a.max_tensors: break
        rb=raw(t)
        npayload_nibbles=(len(rb)//bb)*(pe-ps)*2
        k=int(npayload_nibbles*a.flip_frac)
        if k<1: continue
        mod=flip_nibbles(rb,bb,ps,pe,k,rng)
        w0=gq.dequantize(rb,qtype).astype(np.float64)
        w1=gq.dequantize(mod,qtype).astype(np.float64)
        m=min(w0.size,w1.size); d=np.abs(w0[:m]-w1[:m])
        nz=d>0; nd=int(nz.sum())
        std=float(np.std(w0)) or 1e-12
        tot_w+=int(d.size); tot_diff+=nd
        sum_abs+=float(d.sum()); sum_sq+=float((d*d).sum()); gmax=max(gmax,float(d.max()))
        if nd>0:
            rr=(d[nz]/std)
            if rr.size>4000: rr=rr[::max(1,rr.size//4000)]
            rel.append(rr.astype(np.float32))
    R=np.concatenate(rel) if rel else np.array([0.0])
    pcts={p:float(np.percentile(R,p)) for p in [1,10,50,90,99,100]}
    edges=[0,0.001,0.01,0.05,0.1,0.25,0.5,1.0,2.0,1e9]
    hist,_=np.histogram(R,bins=edges)
    rho=tot_diff/tot_w if tot_w else 0
    out={"canonical":a.canonical,"type":a.type,"flip_frac":a.flip_frac,"tensors":n,
         "rho_L0":rho,"delta_L1":sum_abs/tot_w if tot_w else 0,
         "delta_L2":float(np.sqrt(sum_sq/tot_w)) if tot_w else 0,"delta_Linf":gmax,
         "rel_magnitude_percentiles_over_std":pcts,
         "rel_hist_edges_over_std":edges,"rel_hist_counts":[int(x) for x in hist],
         "n_perturbed_weights":tot_diff}
    print(f"QCB synth ({a.type}, flip_frac={a.flip_frac}): tensors={n} ρ_A={rho:.3e} "
          f"δ_L1={out['delta_L1']:.3e} δ_Linf={gmax:.3e}")
    print("per-weight |Δw|/std percentiles:", {k:round(v,4) for k,v in pcts.items()})
    print("|Δw|/std histogram:")
    for i in range(len(hist)):
        print(f"  [{edges[i]:>6}, {edges[i+1]:<6}) : {hist[i]}")
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)),exist_ok=True)
        json.dump(out,open(a.json,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
        print("wrote",a.json)

if __name__=="__main__": main()
