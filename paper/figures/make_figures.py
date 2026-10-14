#!/usr/bin/env python3
"""Regenerate the paper's figures (PDF+PNG) from the JSON/CSV files under exp/*/data.
Fig. 1: IVR and gap decomposition; Fig. 2: reproducibility (cross-architecture + divergent fraction vs. size); Fig. 4 (paper Fig. 3): security (natural vs. nibble-flip distributions, detection floor, environment pairs).
(The former Fig. 3, byte-region localization, is reported in the text of the M2 section and is no longer generated.)
"""
import os, json, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=os.path.join(os.path.dirname(__file__),"..","..")
OUT=os.path.dirname(__file__)
def J(p):
    try: return json.load(open(os.path.join(ROOT,p),encoding="utf-8"))
    except Exception as e: print("miss",p,e); return None

# palette (Okabe-Ito subset, colorblind-safe)
C={"blue":"#0072B2","orange":"#E69F00","green":"#009E73","red":"#D55E00","grey":"#999999","purple":"#CC79A7"}
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.grid":True,"grid.alpha":0.25,"figure.dpi":140,"savefig.bbox":"tight",
                     # embed TrueType (Type 42) rather than Type 3 fonts
                     "pdf.fonttype":42,"ps.fonttype":42})
def save(fig,name):
    fig.savefig(os.path.join(OUT,name+".pdf")); fig.savefig(os.path.join(OUT,name+".png")); plt.close(fig); print("wrote",name)

# ---- Fig 1: IVR + gap decomposition ----
fig,ax=plt.subplots(1,2,figsize=(7,2.6))
# Only disclosure-based rates on this axis (the Debian CI rate is a different metric and stays in the text).
# Source-reconstructible under the census rule (declared base-model repo) and when the repo must ship its
# own full-precision GGUF (exp/EXP03-ecosystem-census/data/census_strata.csv); both ignore E.
labels=["base-model\nrule","own fp16\nGGUF","bit-exact\n(IVR)"]
import csv as _csv
_strata=list(_csv.DictReader(open(os.path.join(ROOT,"exp","EXP03-ecosystem-census","data","census_strata.csv"))))
_n=len(_strata)
vals=[round(100*sum(r["source_base_rule"]=="True" for r in _strata)/_n,1),
      round(100*sum(r["source_own_rule"]=="True" for r in _strata)/_n,1), 0]   # computed from the shipped census strata
cols=[C["orange"],C["orange"],C["red"]]
b=ax[0].bar(labels,vals,color=cols,width=0.65)
ax[0].set_ylabel("% of 198 repositories"); ax[0].set_ylim(0,100)
for r,v in zip(b,vals): ax[0].text(r.get_x()+r.get_width()/2, v+2, f"{v:g}%", ha="center",fontsize=8)
ax[0].text(0.5, 88, "source-reconstructible\n(both ignore $E$)", ha="center", fontsize=6.5, color="#333333")
ax[0].set_title("(a) Independent Verifiability Rate",fontsize=9,loc="left"); ax[0].tick_params(axis="x",labelsize=7)
gaps=["env/version","fp16 source","imatrix"]; gv=[198,33,19]
ax[1].barh(gaps[::-1],gv[::-1],color=[C["red"],C["blue"],C["blue"]][::-1])
ax[1].set_xlabel("# repos blocked (of 198)"); ax[1].set_title("(b) Non-verifiability cause (198 repos)",fontsize=9,loc="left")
for i,v in enumerate(gv[::-1]): ax[1].text(v+3,i,str(v),va="center",fontsize=8)
ax[1].set_xlim(0,215)
save(fig,"fig1_ivr")

# ---- Fig 2: cross-architecture bit-exactness + divergent fraction vs. size ----
fig,ax=plt.subplots(1,2,figsize=(7,2.6))
# (a) cross-arch matrix from crossmachine_table.csv
rows=list(csv.reader(open(os.path.join(ROOT,"exp/EXP01-stage0-repro/data/crossmachine_table.csv"))))
types=[r[0] for r in rows[1:]]
default_match=[1 if "MATCH" in r[3] else 0 for r in rows[1:]]
ffpoff_match=[1 if "MATCH" in r[4] else 0 for r in rows[1:]]
M=np.array([default_match,ffpoff_match])
ax[0].imshow(M,cmap=plt.cm.RdYlGn,vmin=0,vmax=1,aspect="auto")
ax[0].set_xticks(range(len(types))); ax[0].set_xticklabels(types,rotation=30,ha="right",fontsize=7)
ax[0].set_yticks([0,1]); ax[0].set_yticklabels(["default","-ffp-contract=off\n(PR #25353)"],fontsize=7.5)
for i in range(2):
    for j in range(len(types)): ax[0].text(j,i,"=" if M[i,j] else "≠",ha="center",va="center",fontsize=11,color="black")
ax[0].set_title("(a) Cross-arch bit-exactness (arm64 vs x86_64)",fontsize=8.5,loc="left")
# (b) beta vs scale for Q4_K_M
sc=list(csv.DictReader(open(os.path.join(ROOT,"exp/EXP01-stage0-repro/data/scale_diff.csv"))))
sizes=["1.5B","7B","14B"]; betas=[4.16e-4]  # 1.5B from crossdiff
for lbl in ["7B_Q4_K_M","14B_Q4_K_M"]:
    for r in sc:
        if r["label"]==lbl: betas.append(float(r["beta_byte"]))
ax[1].plot(sizes,[b*1e4 for b in betas],"o-",color=C["blue"])
ax[1].set_ylabel(r"$\beta_{\mathrm{byte}}$ ($\times10^{-4}$)"); ax[1].set_xlabel("model size")
ax[1].set_title("(b) Divergence grows with scale (Q4_K_M)",fontsize=8.5,loc="left")
save(fig,"fig2_repro")

# ---- Fig 4 (paper Fig. 3): separation; realistic per-type floors; Q4_K across environment pairs (EXP11) ----
nat=J("exp/EXP04-dequant-space/data/natural_Q4_K_M_1p5B.json"); qcb=J("exp/EXP05-qcb-structure/data/qcb_Q4_K_1p5B.json")
fig,ax=plt.subplots(1,3,figsize=(10.6,2.8),gridspec_kw={"width_ratios":[1.15,0.9,1.25]})
if nat and qcb:
    edges=nat["rel_hist_edges_over_std"]; xl=[f"{edges[i]:g}–{edges[i+1]:g}" for i in range(len(edges)-2)]+[">2"]
    nh=np.array(nat["rel_hist_counts"],float); nh/=nh.sum()
    qh=np.array(qcb["rel_hist_counts"],float); qh/=qh.sum()
    x=np.arange(len(nh))
    ax[0].bar(x-0.2,nh,width=0.4,label=r"natural $\mathcal{R}_{\mathrm{nat}}$",color=C["blue"])
    ax[0].bar(x+0.2,qh,width=0.4,label="nibble-flip tampering",color=C["red"])
    ax[0].set_xticks(x); ax[0].set_xticklabels(xl,rotation=45,ha="right",fontsize=6.5)
    ax[0].set_ylabel("fraction of perturbed weights"); ax[0].legend(fontsize=7,loc="upper left")
    ax[0].set_xlabel(r"$|\Delta w|/\sigma$ bin"); ax[0].set_title("(a) Separation (Q4_K)",fontsize=9,loc="left")
# (b) per-tensor floor (smallest budget with AUC>=0.99 in every seed; positives = natural rebuild + flips)
#     at a known release (arm64 default vs -ffp-contract=off, b10448), vs the natural tail noise
ty=["Q4_K","Q5_K","Q6_K","IQ4_XS"]; floor=[]; noise=[]
for t in ty:
    r=J(f"exp/EXP11-detector-realistic/data/realistic_{t}.json")["pairs"][0]
    floor.append(r["per_tensor_floor"]*100); noise.append(r["honest"]["tensor_tail_median"]*100)
xx=np.arange(len(ty))
ax[1].bar(xx,floor,color=C["purple"],width=0.55,label="detection floor")
ax[1].scatter(xx,[max(n,1e-4) for n in noise],marker="_",s=260,color="black",zorder=3,label="natural tail noise")
ax[1].set_yscale("log"); ax[1].set_ylim(5e-5,3); ax[1].set_xticks(xx); ax[1].set_xticklabels(ty)
ax[1].set_ylabel("% of weights"); ax[1].legend(fontsize=6.5,loc="upper right")
for i,v in enumerate(floor): ax[1].text(i,v*1.25,f"{v:g}%",ha="center",fontsize=7.5)
ax[1].set_title("(b) Floor at a known release",fontsize=9,loc="left")
# (c) Q4_K: benign difference (pooled fraction of weights above tau) for every honest environment pair,
#     the model-level threshold calibrated on the known-release pairs, and each pair's per-tensor floor
d=J("exp/EXP11-detector-realistic/data/realistic_Q4_K.json")
lab={"b10448-arm64-noimat":"arm64\nb10448","b3772-arm64-imat":"arm64\nb3772\n+imatrix","b3772-arm64-noimat":"arm64\nb3772",
     "b3772-x86-imat":"x86\nb3772\n+imatrix","b3772-x86-noimat":"x86\nb3772","published-vs-b3772":"published\nvs b3772"}
order=["b10448-arm64-noimat","b3772-arm64-noimat","b3772-arm64-imat","b3772-x86-imat","b3772-x86-noimat","published-vs-b3772"]
P={p["label"]:p for p in d["pairs"]}
hv=[P[k]["honest"]["pooled_tail"]*100 for k in order]; fv=[P[k]["per_tensor_floor"]*100 for k in order]
xc=np.arange(len(order)); cols=[C["blue"]]*5+[C["orange"]]
ax[2].bar(xc,hv,color=cols,width=0.6,label="benign difference (honest)")
ax[2].scatter(xc,fv,marker="D",s=22,color="black",zorder=3,label="per-tensor floor")
ax[2].axhline(d["model_rule"]["pooled_threshold"]*100,ls="--",lw=1,color=C["red"],label="model-level threshold")
ax[2].set_xticks(xc); ax[2].set_xticklabels([lab[k] for k in order],fontsize=6)
ax[2].set_ylabel("% of weights above $\\tau$"); ax[2].set_ylim(0,0.27); ax[2].legend(fontsize=6.3,loc="upper left")
ax[2].set_title("(c) Q4_K across environment pairs",fontsize=9,loc="left")
fig.subplots_adjust(wspace=0.42)
save(fig,"fig4_security")
print("done")
