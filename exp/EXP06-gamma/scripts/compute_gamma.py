#!/usr/bin/env python3
"""EXP06: overlap and per-weight threshold rates between natural variance (EXP04) and nibble flips (EXP05).
Both JSON files must use the same |Δw|/std histogram edges.

Reports:
  gamma_perweight(tau)  fraction of flipped weights below a per-weight threshold tau (a natural percentile)
  detect_rate      fraction above the threshold = 1 - gamma_perweight
  overlap_coef     histogram intersection sum_b min(p_nat, p_tamper) (magnitudes only, not counts)
Usage: python3 compute_gamma.py natural.json qcb.json [--json out.json]
"""
import sys,json,argparse
import numpy as np

def load(p): return json.load(open(p,encoding="utf-8"))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("natural"); ap.add_argument("qcb"); ap.add_argument("--json")
    a=ap.parse_args()
    N=load(a.natural); A=load(a.qcb)
    en=N["rel_hist_edges_over_std"]; hn=np.array(N["rel_hist_counts"],float)
    ea=A["rel_hist_edges_over_std"]; ha=np.array(A["rel_hist_counts"],float)
    assert en==ea, "histogram edges differ"
    pn=hn/hn.sum(); pa=ha/ha.sum()
    overlap=float(np.minimum(pn,pa).sum())

    # natural percentiles (threshold candidates)
    npct=N["rel_magnitude_percentiles_over_std"]
    # cumulative distribution of the flips (from the histogram)
    def qcb_frac_below(thr):
        # linear interpolation up to thr
        cum=0.0
        for i in range(len(pa)):
            lo,hi=ea[i],ea[i+1]
            if hi<=thr: cum+=pa[i]
            elif lo>=thr: break
            else:
                frac=(thr-lo)/(hi-lo) if hi>lo else 1.0
                cum+=pa[i]*frac; break
        return cum

    rows=[]
    for pk in ["90","99","99.9"]:
        if pk in npct:
            thr=npct[pk]
            g=qcb_frac_below(thr)
            rows.append((f"tau=natural P{pk}={thr:.3f}", g, 1-g))

    print(f"natural: median={npct.get('50')}  P99={npct.get('99')}")
    print(f"QCB δ_A: median={A['rel_magnitude_percentiles_over_std'].get('50')}  P99={A['rel_magnitude_percentiles_over_std'].get('99')}")
    print(f"\noverlap coefficient = {overlap:.4f}  (smaller = more separated)")
    print("per-weight threshold rates:")
    for name,g,d in rows:
        print(f"  {name}:  Γ_perweight={g:.4f}   detect_rate={d:.4f}")

    # verdict
    verdict = "SEPARABLE" if overlap<0.2 else ("AMBIGUOUS" if overlap<0.5 else "NOT-SEPARABLE")
    print(f"\nverdict (by overlap): {verdict}")
    out={"overlap_coefficient":overlap,
         "natural_median":npct.get("50"),"natural_P99":npct.get("99"),
         "qcb_median":A['rel_magnitude_percentiles_over_std'].get("50"),
         "qcb_P99":A['rel_magnitude_percentiles_over_std'].get("99"),
         "gamma_at_thresholds":[{"threshold":n,"gamma_perweight":g,"detect_rate":d} for n,g,d in rows],
         "verdict":verdict}
    if a.json:
        import os; os.makedirs(os.path.dirname(os.path.abspath(a.json)),exist_ok=True)
        json.dump(out,open(a.json,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
        print("wrote",a.json)

if __name__=="__main__": main()
