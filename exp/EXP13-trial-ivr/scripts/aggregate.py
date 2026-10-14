#!/usr/bin/env python3
"""EXP13 aggregate — compare each environment's re-derived Q4_K_M/Q8_0 to the published file, per tensor.

Reads one or more result roots (each an OUT/ with index.csv whose shas_json paths are relative to it),
merges them, and for every (target, quant, env) attempt reports how many tensors reproduce the published
file bit-for-bit, whether the recipe (per-tensor ggml type) matches, and where the first mismatch is.

Usage: python3 aggregate.py ROOT [ROOT2 ...] --out summary.json
The empirical re-derivation rate is the fraction of (target, quant, env) attempts that are fully bit-exact,
with a Wilson 95% interval; |A| is the number of distinct build environments exercised.
"""
import os, sys, csv, json, argparse, math


def load(root):
    rows = []
    idx = os.path.join(root, "index.csv")
    for r in csv.DictReader(open(idx)):
        p = os.path.join(root, r["shas_json"])
        if not os.path.exists(p):
            print("  missing", p, file=sys.stderr); continue
        j = json.load(open(p))
        r["_tensors"] = {t["name"]: (t["sha"], t["ggml_type"]) for t in j["tensors"]}
        rows.append(r)
    return rows


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for root in a.roots:
        rows += load(root)
    pub = {(r["target"], r["quant"]): r["_tensors"] for r in rows if r["env"] == "published"}
    envs = sorted({r["env"] for r in rows if r["env"] != "published"})
    attempts, results = [], []
    for r in rows:
        if r["env"] == "published": continue
        key = (r["target"], r["quant"])
        if key not in pub:
            continue  # no published reference on hand for this target/quant
        P = pub[key]; Q = r["_tensors"]
        names = set(P) | set(Q)
        sha_match = type_match = 0
        first_mismatch = None; mism_types = {}
        for n in sorted(names, key=lambda x: (x not in Q, x)):
            ps = P.get(n); qs = Q.get(n)
            if ps and qs and ps[1] == qs[1]:
                type_match += 1
            if ps and qs and ps[0] == qs[0]:
                sha_match += 1
            else:
                if first_mismatch is None and ps and qs:
                    first_mismatch = n
                t = (ps[1] if ps else -1)
                mism_types[t] = mism_types.get(t, 0) + 1
        ntot = len(names)
        bit_exact = (sha_match == ntot and set(P) == set(Q))
        rec = {"host": r["host"], "target": r["target"], "release": r["release"], "env": r["env"],
               "quant": r["quant"], "n_tensors": ntot, "sha_match": sha_match, "type_match": type_match,
               "recipe_match": (type_match == ntot and set(P) == set(Q)), "bit_exact": bit_exact,
               "first_mismatch": first_mismatch, "mismatch_types": mism_types}
        results.append(rec); attempts.append(bit_exact)
    n = len(attempts); k = sum(attempts)
    p, lo, hi = wilson(k, n)
    # also: does ANY environment reproduce each published (target,quant)?
    per_target = {}
    for rec in results:
        key = f"{rec['target']}/{rec['quant']}"
        per_target.setdefault(key, {"n_env": 0, "any_bit_exact": False, "best_sha_match": 0, "n_tensors": rec["n_tensors"]})
        d = per_target[key]; d["n_env"] += 1
        d["any_bit_exact"] = d["any_bit_exact"] or rec["bit_exact"]
        d["best_sha_match"] = max(d["best_sha_match"], rec["sha_match"])
    summary = {"n_attempts": n, "n_bit_exact": k, "rederivation_rate": p,
               "rate_wilson95": [lo, hi], "n_environments": len(envs), "environments": envs,
               "per_target": per_target, "results": results}
    json.dump(summary, open(a.out, "w"), indent=1)
    print(f"\n|A| environments = {len(envs)}: {', '.join(envs)}")
    print(f"attempts (target x quant x env with a published reference) = {n}")
    print(f"bit-exact re-derivations = {k}  rate = {p:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
    print("\nper target/quant: any env bit-exact? / best tensor match / total")
    for kk, d in sorted(per_target.items()):
        print(f"  {kk:55s} any={str(d['any_bit_exact']):5s} best={d['best_sha_match']}/{d['n_tensors']} over {d['n_env']} env")
    print("\nclosest attempts (top tensor-match per target/quant):")
    best = {}
    for rec in results:
        key = f"{rec['target']}/{rec['quant']}"
        if key not in best or rec["sha_match"] > best[key]["sha_match"]:
            best[key] = rec
    for kk, rec in sorted(best.items()):
        print(f"  {kk:55s} {rec['env']:22s} {rec['sha_match']}/{rec['n_tensors']} tensors  "
              f"recipe_match={rec['recipe_match']} first_mismatch={rec['first_mismatch']}")
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
