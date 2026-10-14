#!/usr/bin/env python3
"""Summarize alltypes.csv (EXP13b): per published quant type, did re-derivation reach bit-exact under
card-only (plain) vs recipe-pinned, in each build variant? Prints a table and headline counts.

Usage: python3 alltypes_summary.py alltypes.csv [--md out.md]
"""
import csv, sys, argparse
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--md")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    types, cell = [], {}
    for r in rows:
        t = r["type"]
        if t not in types:
            types.append(t)
        be = r["bit_exact"]
        m = r["match"]
        cell[(t, r["variant"], r["mode"])] = (be, m, r.get("first_mismatch", ""))
    variants = sorted({r["variant"] for r in rows})
    out = []
    out.append(f"{'type':10s} | " + " | ".join(f"{v[:6]:>13s}" for v in variants for _ in ("plain", "recipe")))
    out.append(f"{'':10s} | " + " | ".join(f"{m:>13s}" for _ in variants for m in ("plain", "recipe")))
    out.append("-" * len(out[-1]))
    def fmt(c):
        if not c:
            return "-"
        be, m, _ = c
        if be == "1":
            return "BIT-EXACT"
        if be == "FAIL" or m == "":
            return "FAIL"
        return f"{m}"
    n_types = len(types)
    plain_be = recipe_be = 0
    for t in types:
        line = f"{t:10s} | " + " | ".join(f"{fmt(cell.get((t, v, mode))):>13s}"
                                          for v in variants for mode in ("plain", "recipe"))
        out.append(line)
        # a type "reproduces under card-only" if ANY variant's plain is bit-exact
        if any(cell.get((t, v, "plain"), ("", "", ""))[0] == "1" for v in variants):
            plain_be += 1
        if any(cell.get((t, v, "recipe"), ("", "", ""))[0] == "1" for v in variants):
            recipe_be += 1
    out.append("")
    out.append(f"published quant types: {n_types}")
    out.append(f"  bit-exact from card info alone (any build variant): {plain_be}/{n_types}")
    out.append(f"  bit-exact once the per-tensor recipe is supplied:    {recipe_be}/{n_types}")
    txt = "\n".join(out)
    print(txt)
    if a.md:
        open(a.md, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
