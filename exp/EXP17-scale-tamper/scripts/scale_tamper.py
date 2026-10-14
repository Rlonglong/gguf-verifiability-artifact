#!/usr/bin/env python3
"""EXP17 — coordinated scale tampering vs the M3 detection floor.

The paper's detector (EXP11) is a tail statistic: it counts weights with |Δw|/σ > τ (per-tensor and
pooled) plus a max |Δw|/σ. M3 notes but does not evaluate an adaptive adversary who edits *scales*
(which move every weight in a block by a sub-step amount) rather than codes. This script realizes that
adversary directly and measures whether it hides.

Tamper: for each Q4_K super-block (144 bytes: f16 d, f16 dmin, 12B scales, 128B qs), multiply the f16
super-scale `d` by (1+eps). This scales the d-term of all 256 weights in the block by ~(1+eps) — a pure,
minimal, coordinated scale edit that never touches a single code. It is applied to every Q4_K tensor.

Outputs a tampered GGUF (byte-identical except the edited d fields) so downstream perplexity can be
measured with llama-perplexity, and prints the detector statistics of tampered-vs-original.

Usage: python3 scale_tamper.py IN.gguf OUT.gguf --eps 0.05 [--tau 0.1] [--json stats.json]
Needs gguf-py + numpy (set GGUF_PY) for the |Δw|/σ statistics; the byte edit itself is stdlib-only.
"""
import os, sys, struct, json, argparse, shutil
_gp = os.environ.get("GGUF_PY")
if _gp and os.path.isdir(_gp):
    sys.path.insert(0, _gp)
import numpy as np

SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
Q4_K = 12; BLK = 144


def rd(f, fmt): return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]
def rstr(f): return f.read(rd(f, "Q"))


def skip_val(f, t):
    if t == 8: rstr(f)
    elif t == 9:
        et = rd(f, "I"); n = rd(f, "Q")
        [rstr(f) for _ in range(n)] if et == 8 else f.seek(SZ[et] * n, 1)
    else: f.seek(SZ[t], 1)


def parse(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"; rd(f, "I"); nt = rd(f, "Q"); nkv = rd(f, "Q"); align = 32
        for _ in range(nkv):
            k = rstr(f); t = rd(f, "I")
            if k == b"general.alignment" and t in (4, 5): align = rd(f, "I"); continue
            skip_val(f, t)
        infos = []
        for _ in range(nt):
            name = rstr(f).decode("utf-8", "ignore"); nd = rd(f, "I")
            dims = [rd(f, "Q") for _ in range(nd)]; gt = rd(f, "I"); off = rd(f, "Q")
            infos.append([name, gt, off])
        ds = (f.tell() + align - 1) // align * align
        fsize = os.path.getsize(path)
        order = sorted(range(nt), key=lambda i: infos[i][2])
        for rank, i in enumerate(order):
            end = infos[order[rank + 1]][2] if rank + 1 < nt else (fsize - ds)
            infos[i].append(end - infos[i][2])  # nbytes
        return ds, infos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--tau", type=float, default=0.1); ap.add_argument("--json")
    a = ap.parse_args()
    ds, infos = parse(a.inp)
    shutil.copyfile(a.inp, a.out)
    # dequant helpers for statistics
    from gguf import GGMLQuantizationType as T
    import gguf.quants as gq
    def deq(b): return gq.dequantize(np.frombuffer(b, dtype=np.uint8), T.Q4_K).astype(np.float64).ravel()
    neg = []; pooled_ex = 0; pooled_n = 0; mx = 0.0
    fin = open(a.inp, "rb"); fout = open(a.out, "r+b")
    n_tensors = 0
    for name, gt, off, nbytes in infos:
        if gt != Q4_K:
            continue
        n_tensors += 1
        fin.seek(ds + off); raw = fin.read(nbytes)
        a0 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, BLK).copy()
        d = a0[:, 0:2].copy().view(np.float16).astype(np.float32)      # super-scale per block
        d2 = (d * (1.0 + a.eps)).astype(np.float16)
        a1 = a0.copy(); a1[:, 0:2] = d2.view(np.uint8).reshape(-1, 2)
        # statistics: |Δw|/σ over the tensor
        w0 = deq(a0.reshape(-1).tobytes()); w1 = deq(a1.reshape(-1).tobytes())
        sd = w0.std() or 1e-12
        dd = np.abs(w1 - w0) / sd
        ex = int((dd > a.tau).sum())
        neg.append(ex / dd.size); pooled_ex += ex; pooled_n += dd.size; mx = max(mx, float(dd.max()))
        fout.seek(ds + off); fout.write(a1.reshape(-1).tobytes())
        print(f"  [{name}] blocks={a0.shape[0]} tail={ex/dd.size:.5%} max={float(dd.max()):.3f}", end="\r", flush=True)
    fin.close(); fout.close(); print()
    res = {"eps": a.eps, "tau": a.tau, "n_q4k_tensors": n_tensors,
           "tensor_tail_median": float(np.median(neg)) if neg else 0.0,
           "tensor_tail_max": float(np.max(neg)) if neg else 0.0,
           "pooled_tail": pooled_ex / max(pooled_n, 1), "max_stat": mx}
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
