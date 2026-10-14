#!/usr/bin/env python3
"""Compare two per-tensor sha JSONs (tools/gguf_tensor_shas.py output): published vs re-derived.

Usage: python3 cmp_shas.py PUBLISHED.shas.json MINE.shas.json [--csv]
Prints tensor-match counts; with --csv prints `n_tensors,match,type_match,bit_exact,first_mismatch`.
bit_exact means every tensor's data sha and ggml type agree and the tensor name sets are equal.
"""
import json, sys


def load(p):
    return {t["name"]: (t["sha"], t["ggml_type"]) for t in json.load(open(p))["tensors"]}


def main():
    P, Q = load(sys.argv[1]), load(sys.argv[2])
    names = sorted(set(P) | set(Q))
    match = sum(1 for n in names if n in P and n in Q and P[n] == Q[n])
    tmatch = sum(1 for n in names if n in P and n in Q and P[n][1] == Q[n][1])
    first = next((n for n in names if not (n in P and n in Q and P[n] == Q[n])), "")
    be = int(match == len(names) and set(P) == set(Q))
    if "--csv" in sys.argv:
        print(f"{len(names)},{match},{tmatch},{be},{first}")
    else:
        print(f"match {match}/{len(names)}  types {tmatch}/{len(names)}  bit_exact={bool(be)}  first_mismatch={first or '-'}")


if __name__ == "__main__":
    main()
