#!/usr/bin/env python3
"""EXP19 — adaptive placement of one-step code flips vs the M3 detection floor.

§M3 evaluates only *uniformly placed* one-step flips and flags, as open, an adversary who *places* flips
adaptively (e.g. in small-scale blocks) to minimize |Δw|/σ and slip under the tail statistic. This measures
that adversary directly, in weight space, with no behavioral attack: for a fixed budget of flips it compares
uniform-random placement to a stealth-optimal placement (the flips whose one-step move |Δw|/σ is smallest),
and reports for each whether the tail statistic, the pooled model-level rule, and the max statistic fire —
plus the behavioral cost (relative L1 weight change) the stealth placement pays for that stealth.

A one-step flip moves a weight by exactly its block's dequantized step, obtained by dequantizing the tensor
with every low-bit code shifted by +1 (the EXP05/EXP08/EXP11 edit).

Usage: python3 adaptive_placement.py MODEL.gguf --tau 0.1 --json out.json   (Q4_K tensors; needs gguf-py+numpy)
"""
import os, sys, json, argparse
_gp = os.environ.get("GGUF_PY")
if _gp and os.path.isdir(_gp):
    sys.path.insert(0, _gp)
import numpy as np
from gguf import GGUFReader, GGMLQuantizationType as T
import gguf.quants as gq

BB, PS, PE = 144, 16, 144   # Q4_K block bytes, qs payload start/end
BUDGETS = [2e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]


def raw(t): return np.frombuffer(t.data.tobytes(), dtype=np.uint8)


def plus1(p):
    lo, hi = p & 0xF, p >> 4
    lo2 = np.where(lo == 15, lo - 1, lo + 1)
    hi2 = np.where(hi == 15, hi - 1, hi + 1)
    return ((hi2 << 4) | lo2).astype(np.uint8)


def with_payload(rawb, fn):
    a = rawb.reshape(-1, BB).copy(); a[:, PS:PE] = fn(a[:, PS:PE]); return a.reshape(-1)


def deq(b): return gq.dequantize(b, T.Q4_K).astype(np.float64).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--pooled-threshold", type=float, default=0.0019)
    ap.add_argument("--max-threshold", type=float, default=2.36); ap.add_argument("--json")
    a = ap.parse_args()
    r = GGUFReader(a.model)
    ts = [t for t in r.tensors if t.tensor_type == T.Q4_K]
    # per-weight one-step |Δw|/σ (the stealth score) and |Δw| (the behavioral move), pooled over all Q4_K weights
    rel = []; absmove = []; wabs = []
    for t in ts:
        b = raw(t); w0 = deq(b); sd = w0.std() or 1e-12
        step = deq(with_payload(b, plus1)) - w0
        rel.append(np.abs(step) / sd); absmove.append(np.abs(step)); wabs.append(np.abs(w0))
    rel = np.concatenate(rel); absmove = np.concatenate(absmove); wabs = np.concatenate(wabs)
    N = rel.size; Wsum = wabs.sum()
    order = np.argsort(rel)          # ascending |Δw|/σ = stealth-optimal first
    honest_max = float(np.median([np.abs((deq(with_payload(raw(t), plus1)) - deq(raw(t)))).max() for t in ts[:0]] or [0]))
    rows = []
    for f in BUDGETS:
        k = max(1, int(N * f))
        rng = np.random.default_rng(0)
        uni = rng.choice(N, size=k, replace=False)
        ada = order[:k]
        def stat(idx):
            over = int((rel[idx] > a.tau).sum())
            return {"flips": k, "flip_frac": k / N,
                    "detectable_tail_frac": over / N,               # pooled contribution
                    "flips_above_tau": over, "flips_hidden": k - over,
                    "max_stat": float(rel[idx].max()),
                    "rel_L1_weight_change": float(absmove[idx].sum() / Wsum)}
        rows.append({"budget": f, "uniform": stat(uni), "adaptive_stealth": stat(ada)})
    # how many weights have a sub-tau one-step move at all (the stealth reservoir)
    sub_tau = int((rel <= a.tau).sum())
    out = {"model": os.path.basename(a.model), "tau": a.tau, "n_q4k_weights": N,
           "pooled_threshold": a.pooled_threshold, "max_threshold": a.max_threshold,
           "frac_weights_substep_below_tau": sub_tau / N, "rows": rows}
    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
