#!/usr/bin/env python3
"""EXP13 summary — pivot the per-tensor comparison into the paper's reproduction matrix.

For every (target, quant) and every build environment (arch x FP-contraction x recipe-supplied), reports
how many tensors reproduce the published file bit-for-bit, marks full bit-exact re-derivations, and states
the minimal set of card-absent coordinates (build environment / quantization recipe) needed to reach it.

Usage: python3 summarize.py ROOT [ROOT2 ...] [--figdata figure_data.json] [--md table.md]
"""
import os, sys, csv, json, argparse
from collections import defaultdict


def load_root(root):
    out = []
    for r in csv.DictReader(open(os.path.join(root, "index.csv"))):
        p = os.path.join(root, r["shas_json"])
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        r["_t"] = {t["name"]: (t["sha"], t["ggml_type"]) for t in j["tensors"]}
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--figdata"); ap.add_argument("--md")
    a = ap.parse_args()
    rows = []
    for root in a.roots:
        rows += load_root(root)
    pub = {(r["target"], r["quant"]): r["_t"] for r in rows if r["env"] == "published"}
    # env label -> (arch, fp, recipe)
    def parse_env(host, env):
        arch = "arm64" if "arm" in host else ("x86" if "x86" in host else host)
        fp = "ffpoff" if "ffpoff" in env else "default"
        recipe = env.endswith("-recipe")
        # compiler tag: armclang / armgcc15 / x86gcc
        comp = env.split("-")[0]
        return arch, comp, fp, recipe
    cell = {}  # (target,quant,comp,fp,recipe) -> (match,total,bitexact)
    for r in rows:
        if r["env"] == "published":
            continue
        key = (r["target"], r["quant"])
        if key not in pub:
            continue
        P = pub[key]; Q = r["_t"]
        match = sum(1 for n in P if n in Q and P[n] == Q[n]); tot = len(P)
        be = (match == tot and set(P) == set(Q))
        arch, comp, fp, recipe = parse_env(r["host"], r["env"])
        cell[(r["target"], r["quant"], comp, fp, recipe)] = (match, tot, be, arch)
    targets = sorted({k[0] for k in pub})
    quants = ["Q4_K_M", "Q8_0"]
    comps = sorted({k[2] for k in cell})
    lines = []
    lines.append(f"{'target':26s} {'quant':7s} {'comp':9s} {'FP':8s} {'plain':>9s} {'+recipe':>9s}")
    for t in targets:
        for q in quants:
            for comp in comps:
                for fp in ["default", "ffpoff"]:
                    pl = cell.get((t, q, comp, fp, False)); rc = cell.get((t, q, comp, fp, True))
                    if not pl and not rc:
                        continue
                    def fmt(c):
                        if not c: return "-"
                        m, tot, be, _ = c; return (f"{m}/{tot}" + ("*" if be else ""))
                    lines.append(f"{t[:26]:26s} {q:7s} {comp:9s} {fp:8s} {fmt(pl):>9s} {fmt(rc):>9s}")
    # minimal coordinates per (target,quant): scan for the simplest bit-exact cell
    order = [("default", False, "card only (release+imatrix+preset)"),
             ("default", True, "+recipe"),
             ("ffpoff", False, "+FP env (uncontracted)"),
             ("ffpoff", True, "+FP env +recipe")]
    lines.append("\nminimal card-absent coordinates for a bit-exact re-derivation:")
    repro = {}
    for t in targets:
        for q in quants:
            found = None
            for comp in comps:
                for fp, rec, desc in order:
                    c = cell.get((t, q, comp, fp, rec))
                    if c and c[2]:
                        cand = (order.index((fp, rec, desc)), f"{comp}/{fp}{'+recipe' if rec else ''}", desc, c[3])
                        if found is None or cand[0] < found[0]:
                            found = cand
            repro[(t, q)] = found
            if found:
                lines.append(f"  {t[:26]:26s} {q:7s} -> {found[2]:34s} [{found[3]} {found[1]}]")
            else:
                # best partial
                best = max((cell[(t, q, comp, fp, rec)][0], comp, fp, rec)
                           for comp in comps for fp in ["default", "ffpoff"] for rec in [False, True]
                           if (t, q, comp, fp, rec) in cell)
                tot = pub[(t, q)].__len__()
                lines.append(f"  {t[:26]:26s} {q:7s} -> NOT bit-exact in any tried env (best {best[0]}/{tot})")
    n_be = sum(1 for v in repro.values() if v)
    lines.append(f"\nbit-exact re-derivable with disclosed coordinates: {n_be}/{len(repro)} (target x quant)")
    out = "\n".join(lines)
    print(out)
    if a.md:
        open(a.md, "w").write(out + "\n")
    if a.figdata:
        fd = {"cells": {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}|{int(k[4])}": {"match": v[0], "total": v[1],
              "bit_exact": v[2], "arch": v[3]} for k, v in cell.items()},
              "repro": {f"{k[0]}|{k[1]}": (v[2] if v else None) for k, v in repro.items()}}
        json.dump(fd, open(a.figdata, "w"), indent=1)
        print("\nwrote", a.figdata)


if __name__ == "__main__":
    main()
