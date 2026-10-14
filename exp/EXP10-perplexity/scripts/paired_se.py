#!/usr/bin/env python3
"""Paired standard error of ΔPPL from two llama-perplexity logs over the same text.

llama-perplexity prints a running perplexity [k]PPL_k after each chunk k, with
PPL_k = exp(mean NLL over chunks 1..k). The per-chunk mean NLL is therefore recovered as
  nll_1 = ln PPL_1,  nll_k = k ln PPL_k - (k-1) ln PPL_{k-1},
and the paired difference d_k = nll_k(B) - nll_k(A) gives mean(d) ± sd(d)/sqrt(n), mapped to
PPL units by the delta method (ΔPPL ≈ PPL_A · mean(d)). Both models are scored on identical
chunks, so this paired SE is the right uncertainty for ΔPPL (the ± that llama-perplexity prints is
each model's own marginal SE).

Usage: python3 paired_se.py data/ppl_default.txt data/ppl_ffpoff.txt
"""
import math, re, sys


def running_ppl(path):
    return [float(v) for _, v in re.findall(r"\[(\d+)\]([0-9.]+)", open(path, errors="ignore").read())]


def per_chunk_nll(ppl):
    L = [math.log(x) for x in ppl]
    return [L[0]] + [(k + 1) * L[k] - k * L[k - 1] for k in range(1, len(L))]


def main():
    a, b = (per_chunk_nll(running_ppl(p)) for p in sys.argv[1:3])
    assert len(a) == len(b), "logs must cover the same chunks"
    n = len(a); d = [y - x for x, y in zip(a, b)]
    md = sum(d) / n; se = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1)) / math.sqrt(n)
    P = running_ppl(sys.argv[1])[-1]
    print(f"chunks={n}  PPL_A={P:.4f}  dPPL={P * md:+.4f} ± {P * se:.4f} (paired SE)  t={md / se:.2f}")


if __name__ == "__main__":
    main()
