#!/usr/bin/env python3
"""EXP11 — the tail-fraction statistic with realistic positives, model-level decisions, and a max statistic.

EXP08 applied the tampering flips to the reference itself, i.e. tampering without natural drift. That is
the shared-environment case, in which comparing hashes already suffices. Here:
  (1) positives are the *natural* (honest, other-environment) artifact plus uniformly placed one-step
      flips, scored against the reference, for several seeds;
  (2) besides the per-tensor ROC, a model-level statistic: the pooled fraction of the type's weights with
      |dw|/sigma > tau, compared across every honest pair given (known and unknown release);
  (3) a max statistic (largest |dw|/sigma in the model) against k sparse large edits that move a code to
      its farthest value (bit-flip style), which a count-based tail statistic cannot see.
Everything is computed exactly from dequantized tensors. A flip changes one low-bit code by +1 (-1 at the
top code), the same edit as EXP05/EXP08, so the flipped weight moves by exactly the dequantized step
w(code+1) - w(code) of its block, which we obtain by dequantizing the tensor with every code shifted.

Usage:
  python3 detector_realistic.py --pairs pairs.csv --type Q4_K --out out.json [--seeds 5] [--large]
pairs.csv columns: label,reference,natural,regime   (regime: known-release | unknown-release)
Needs gguf-py + numpy (set GGUF_PY or WORK to locate gguf-py).
"""
import os, sys, argparse, json, csv
_gp = os.environ.get("GGUF_PY") or os.path.join(os.environ.get("WORK", ""), "llama.cpp", "gguf-py")
if _gp and os.path.isdir(_gp):
    sys.path.insert(0, _gp)
import numpy as np
from gguf import GGUFReader, GGMLQuantizationType as T
import gguf.quants as gq

# (block_bytes, payload_start, payload_end): the low-bit code plane (qs for Q4_K/Q5_K/IQ4_XS, ql for Q6_K)
PAYLOAD = {"Q4_K": (144, 16, 144), "Q5_K": (176, 48, 176), "Q6_K": (210, 0, 128), "IQ4_XS": (136, 8, 136)}
BUDGETS = [2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]   # fraction of the type's weights flipped
LARGE_K = [1, 5, 25, 100]                                     # number of sparse large edits per model


def raw(t):
    return np.frombuffer(t.data.tobytes(), dtype=np.uint8)


def with_payload(rawb, bb, ps, pe, fn):
    a = rawb.reshape(-1, bb).copy()
    a[:, ps:pe] = fn(a[:, ps:pe])
    return a.reshape(-1)


def plus1(p):
    lo, hi = p & 0xF, p >> 4
    lo2 = np.where(lo == 15, lo - 1, lo + 1)
    hi2 = np.where(hi == 15, hi - 1, hi + 1)
    return ((hi2 << 4) | lo2).astype(np.uint8)


def deq(b, qtype):
    return gq.dequantize(b, qtype).astype(np.float64).ravel()


def auc(neg, pos):
    s = np.concatenate([neg, pos]); lab = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); i = 0
    ss = s[order]
    while i < len(s):                      # average ranks over ties
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    rp = ranks[lab == 1].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def tpr_at_fpr(neg, pos, fpr=0.01):
    ns = np.sort(neg); allowed = int(np.floor(fpr * len(ns)))
    thr = ns[len(ns) - allowed - 1]
    return float(np.mean(pos > thr))


def run_pair(label, ref_path, nat_path, qname, tau, seeds, large):
    qtype = getattr(T, qname); bb, ps, pe = PAYLOAD[qname]
    rr, rn = GGUFReader(ref_path), GGUFReader(nat_path)
    nd = {t.name: t for t in rn.tensors}
    ts = [t for t in rr.tensors if t.tensor_type == qtype and t.name in nd and nd[t.name].tensor_type == qtype]
    sizes = np.array([int(np.prod(t.shape)) for t in ts]); offs = np.r_[0, np.cumsum(sizes)]; N = int(offs[-1])
    # global positions of the sparse large edits, per seed and k (uniform over all of the type's weights)
    gl = {(s, k): np.sort(np.random.default_rng([1000 + s, k]).choice(N, size=k, replace=False))
          for s in range(seeds) for k in LARGE_K} if large else {}
    nb = len(BUDGETS)
    neg = np.zeros(len(ts)); pos = np.zeros((nb, seeds, len(ts)))
    pooled_base = 0; pooled = np.zeros((nb, seeds)); mx_base = 0.0; mx_flip = np.zeros((nb, seeds))
    mx_large = {key: 0.0 for key in gl}; step_med = []
    for ti, t in enumerate(ts):
        a, b = raw(t), raw(nd[t.name])
        wr, wn = deq(a, qtype), deq(b, qtype)
        sd = wr.std() or 1e-12
        dd = wn - wr; d = np.abs(dd) / sd; ex = d > tau
        base = int(ex.sum()); n = d.size
        neg[ti] = base / n; pooled_base += base; mx_base = max(mx_base, float(d.max()))
        step = deq(with_payload(b, bb, ps, pe, plus1), qtype) - wn
        step_med.append(float(np.median(np.abs(step)) / sd))
        for bi, f in enumerate(BUDGETS):
            k = int(n * f)
            for s in range(seeds):
                if k < 1:
                    pos[bi, s, ti] = base / n; pooled[bi, s] += base; mx_flip[bi, s] = max(mx_flip[bi, s], float(d.max()))
                    continue
                idx = np.random.default_rng([s, ti, bi]).choice(n, size=k, replace=False)
                nv = np.abs(dd[idx] + step[idx]) / sd
                cnt = base - int(ex[idx].sum()) + int((nv > tau).sum())
                pos[bi, s, ti] = cnt / n; pooled[bi, s] += cnt
                mx_flip[bi, s] = max(mx_flip[bi, s], float(d.max()), float(nv.max()))
        if large:
            w0 = deq(with_payload(b, bb, ps, pe, np.zeros_like), qtype)
            w15 = deq(with_payload(b, bb, ps, pe, lambda p: np.full_like(p, 0xFF)), qtype)
            lo, hi = offs[ti], offs[ti + 1]
            for key, g in gl.items():
                loc = g[(g >= lo) & (g < hi)] - lo
                if loc.size:
                    far = np.where(np.abs(w0[loc] - wn[loc]) > np.abs(w15[loc] - wn[loc]), w0[loc], w15[loc])
                    mx_large[key] = max(mx_large[key], float((np.abs(far - wr[loc]) / sd).max()))
        print(f"  [{label}] tensor {ti + 1}/{len(ts)}", end="\r", flush=True)
    print()
    res = {"label": label, "n_tensors": len(ts), "n_weights": N,
           "median_step_over_sigma": float(np.median(step_med)),
           "honest": {"tensor_tail_median": float(np.median(neg)), "tensor_tail_max": float(neg.max()),
                      "pooled_tail": pooled_base / N, "max": mx_base},
           "per_tensor": {}, "model": {}, "large": {}}
    for bi, f in enumerate(BUDGETS):
        aucs = [auc(neg, pos[bi, s]) for s in range(seeds)]
        tprs = [tpr_at_fpr(neg, pos[bi, s]) for s in range(seeds)]
        res["per_tensor"][str(f)] = {"auc_mean": float(np.mean(aucs)), "auc_min": float(np.min(aucs)),
                                     "tpr_at_fpr1_mean": float(np.mean(tprs)), "tpr_at_fpr1_min": float(np.min(tprs))}
        res["model"][str(f)] = {"pooled_tail_mean": float(pooled[bi].mean() / N),
                                "pooled_tail_min": float(pooled[bi].min() / N),
                                "max_mean": float(mx_flip[bi].mean())}
    for k in (LARGE_K if large else []):
        v = [max(mx_base, mx_large[(s, k)]) for s in range(seeds)]
        res["large"][str(k)] = {"max_mean": float(np.mean(v)), "max_min": float(np.min(v))}
    fl = [f for f in BUDGETS if res["per_tensor"][str(f)]["auc_min"] >= 0.99]
    res["per_tensor_floor"] = min(fl) if fl else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True); ap.add_argument("--type", default="Q4_K", choices=list(PAYLOAD))
    ap.add_argument("--tau", type=float, default=0.1); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--large", action="store_true"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pairs = list(csv.DictReader(open(a.pairs)))
    for _p in pairs:   # pair files use the $WORK placeholder for the rebuilt GGUFs
        for _k in ("reference","natural"): _p[_k]=os.path.expandvars(_p[_k])
    out = {"type": a.type, "tau": a.tau, "seeds": a.seeds, "budgets": BUDGETS, "large_k": LARGE_K if a.large else [],
           "pairs": []}
    for p in pairs:
        r = run_pair(p["label"], p["reference"], p["natural"], a.type, a.tau, a.seeds, a.large)
        r["regime"] = p["regime"]; out["pairs"].append(r)
        h = r["honest"]
        print(f"{p['label']:28s} honest pooled={h['pooled_tail']:.5%} tensor-median={h['tensor_tail_median']:.5%} "
              f"max={h['max']:.2f} floor(per-tensor)={r['per_tensor_floor']} step/sigma~{r['median_step_over_sigma']:.3f}")
    # model-level rule: threshold = largest honest pooled tail among known-release pairs
    known = [r for r in out["pairs"] if r["regime"] == "known-release"]
    if known:
        thr = max(r["honest"]["pooled_tail"] for r in known); mthr = max(r["honest"]["max"] for r in known)
        out["model_rule"] = {"pooled_threshold": thr, "max_threshold": mthr,
                             "honest_flagged_by_pooled": {r["label"]: r["honest"]["pooled_tail"] > thr for r in out["pairs"]},
                             "honest_flagged_by_max": {r["label"]: r["honest"]["max"] > mthr for r in out["pairs"]}}
        print(f"model rule: pooled threshold {thr:.5%}, max threshold {mthr:.2f}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
