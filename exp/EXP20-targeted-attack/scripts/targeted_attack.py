#!/usr/bin/env python3
"""EXP20 — does a *targeted* quantizer tamper land above or below the M3 detection floor?

Defensive detector-evaluation (not a deployable exploit). §M3/EXP19 leave open whether a *targeted*
tamper (a rounding-guided edit aimed at a behavioral objective, cf. QuRA) can achieve that objective within
the sub-floor budget that evades the tail/max statistics. We measure this on a benign objective: for a
fixed neutral probe, flip the model's greedy next token from its true argmax T0 to a chosen alternative T1.

Qwen2.5-1.5B has tied embeddings, so the output logit of token t is exactly <h, E_t> for the final hidden
state h. This makes the attack exact and gradient-free: raising logit_T1 by moving E_T1[j] one quant step in
direction sign(h[j]) gains |h[j]|*step[j]; lowering logit_T0 likewise. We greedily apply the highest-gain
one-step edits (over the T1 and T0 embedding rows) until the predicted margin flips, then report the edit
footprint and its detection statistics (per-tensor tail at tau, pooled, max |Δw|/σ), and confirm the flip by
placing the tampered embedding in the model. Sweeping the target's rank R sweeps attack difficulty.

Run in a venv with torch+transformers (GPU) and gguf-py on the path. Usage:
  python3 targeted_attack.py --hf DIR --gguf Q4KM.gguf --probe "..." --json out.json
"""
import os, sys, json, argparse
import numpy as np


def load_embed_q6k(gguf_path):
    import gguf
    from gguf import GGUFReader, GGMLQuantizationType as T
    import gguf.quants as gq
    r = GGUFReader(gguf_path)
    t = next(x for x in r.tensors if x.name == "token_embd.weight")
    raw = np.frombuffer(t.data.tobytes(), dtype=np.uint8)
    E = gq.dequantize(raw, t.tensor_type).astype(np.float64)   # [vocab, hidden]
    shape = [int(s) for s in t.shape][::-1]                    # gguf shape [hidden, vocab] -> rows=vocab
    hidden = shape[-1]
    E = E.reshape(-1, hidden)
    assert t.tensor_type == T.Q6_K, "this script assumes a Q6_K token_embd"
    # EXACT Q6_K one-step size = |d| * |sub-scale|. Block=210B: ql[128] qh[64] scales[16] int8 @192 d f16 @208.
    a = raw.reshape(-1, 210)
    d = a[:, 208:210].copy().view(np.float16).astype(np.float64).ravel()   # [nblk]
    sc = np.abs(a[:, 192:208].view(np.int8).astype(np.float64))            # [nblk,16]
    step_sub = np.abs(d)[:, None] * sc                                     # [nblk,16] one-step per sub-block
    # expand to per-element [nblk,256] then reshape to [vocab,hidden] (hidden a multiple of 256)
    step_el = np.repeat(step_sub, 16, axis=1).reshape(-1)                  # [nblk*256] = [vocab*hidden]
    step = step_el.reshape(-1, hidden)
    return E, step, t.tensor_type.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True); ap.add_argument("--gguf", required=True)
    ap.add_argument("--probe", default="The capital of France is")
    ap.add_argument("--tau", type=float, default=0.1); ap.add_argument("--ranks", default="1,10,100,1000")
    ap.add_argument("--confirm-rank", type=int, default=100); ap.add_argument("--json", required=True)
    a = ap.parse_args()
    import torch
    import huggingface_hub as _hh   # shim: some transformers/hf_hub version pairs drop this symbol
    if not hasattr(_hh, "get_full_repo_name"):
        _hh.get_full_repo_name = lambda name, *a, **k: name
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.hf)
    model = AutoModelForCausalLM.from_pretrained(a.hf, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    ids = tok(a.probe, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    h = out.hidden_states[-1][0, -1].float().cpu().numpy()          # final hidden state at last position
    logits = out.logits[0, -1].float().cpu().numpy()
    order = np.argsort(-logits)
    T0 = int(order[0])
    print(f"probe={a.probe!r}  argmax T0={T0} ({tok.decode([T0])!r})  logit={logits[T0]:.3f}")

    E, step, etype = load_embed_q6k(a.gguf)                        # [vocab,hidden] dequant + EXACT one-step
    sigma = float(E.std()) or 1e-12
    Ntot = E.size
    frac_above = float((step / sigma > a.tau).mean())
    print(f"token_embd type={etype} shape={E.shape} sigma={sigma:.4g} median step/sigma={np.median(step)/sigma:.4f} "
          f"frac one-step>tau={frac_above:.4f}")

    def attack(T1, stealth):
        # Make T1 the greedy token by raising logit_T1 above logit_T0 via one-step edits to T1's row.
        # stealth=True restricts to edits whose one-step |Δw|/σ <= tau (invisible to the tail statistic).
        margin = float(logits[T0] - logits[T1]) * 1.05
        rel_row = step[T1] / sigma
        g1 = np.abs(h) * step[T1]
        mask = (rel_row <= a.tau) if stealth else np.ones_like(g1, bool)
        cand = np.where(mask, g1, -1.0)
        idx = np.argsort(-cand)
        idx = idx[cand[idx] > 0]                                   # usable candidates only
        cum = np.cumsum(g1[idx]) if idx.size else np.array([0.0])
        reachable = idx.size > 0 and cum[-1] >= margin
        k = int(np.searchsorted(cum, margin) + 1) if reachable else int(idx.size)
        chosen = idx[:k]
        rel = step[T1, chosen] / sigma
        return {"target_rank_token": tok.decode([T1]), "margin": margin, "stealth": stealth,
                "reachable": bool(reachable), "edits": int(k), "edit_frac_of_model": k / Ntot,
                "flips_above_tau": int((rel > a.tau).sum()), "flips_hidden": int((rel <= a.tau).sum()),
                "max_stat": float(rel.max()) if k else 0.0, "pooled_tail_contrib": int((rel > a.tau).sum()) / Ntot,
                "_chosen": chosen, "_T1": T1}

    ranks = [int(x) for x in a.ranks.split(",")]
    rows = []
    for R in ranks:
        T1 = int(order[R])
        full = attack(T1, stealth=False); stealthy = attack(T1, stealth=True)
        rows.append({"rank": R, "unconstrained": {k: v for k, v in full.items() if not k.startswith("_")},
                     "stealth_subtau_only": {k: v for k, v in stealthy.items() if not k.startswith("_")}})
        print(f"rank {R:5d} T1={full['target_rank_token']!r} margin={full['margin']:.2f} | "
              f"unconstrained: edits={full['edits']} above_tau={full['flips_above_tau']} max={full['max_stat']:.2f} | "
              f"stealth(sub-tau only): reachable={stealthy['reachable']} edits={stealthy['edits']} max={stealthy['max_stat']:.2f}")

    # confirm: can a STEALTH (sub-tau only) attack flip the prediction in-model?
    R = a.confirm_rank; T1 = int(order[R]); r = attack(T1, stealth=True)
    Et = E.copy()
    for j in r["_chosen"]:
        Et[T1, j] += (step[T1, j] if h[j] > 0 else -step[T1, j])
    with torch.no_grad():
        emb = model.get_input_embeddings()
        emb.weight.copy_(torch.tensor(Et, dtype=emb.weight.dtype, device=emb.weight.device))
        out2 = model(**ids)
    new_arg = int(out2.logits[0, -1].argmax().cpu())
    confirmed = (new_arg == T1)
    print(f"CONFIRM rank {R}: tampered argmax={new_arg} ({tok.decode([new_arg])!r}); target T1={T1} ({tok.decode([T1])!r}); flipped_to_target={confirmed}; flipped_off_T0={new_arg!=T0}")

    out_obj = {"probe": a.probe, "T0": tok.decode([T0]), "tau": a.tau, "embed_type": etype,
               "n_model_weights_proxy": int(Ntot), "sigma": sigma,
               "pooled_threshold": 0.0019, "max_threshold": 2.36,
               "confirm": {"rank": R, "stealth_subtau_only": True, "reachable": r["reachable"],
                           "flipped_to_target": bool(confirmed), "flipped_off_T0": bool(new_arg != T0),
                           "edits": r["edits"], "max_stat": r["max_stat"], "flips_hidden": r["flips_hidden"],
                           "flips_above_tau": r["flips_above_tau"]},
               "sweep": rows}
    json.dump(out_obj, open(a.json, "w"), indent=1)
    print("wrote", a.json)


if __name__ == "__main__":
    main()
