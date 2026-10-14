#!/usr/bin/env python3
"""Minimal Mind-the-Gap procedure (content injection) on Qwen2.5-0.5B.
Uses the authors' q_attack box code:
  1. malicious fine-tune (inject the payload) -> M_mal
  2. compute_box_gguf: the fp16 box around M_mal's quantized values
  3. fine-tune on benign data, projecting into the box every step -> attacked fp16 (intended: benign in fp16, malicious once quantized)
Writes the attacked HF model; quantization, footprint, and the behavioral check are done by mtg_run.sh.
"""
import os, sys, json, argparse, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from gguf import GGUFReader
from q_attack.repair.gguf.dequantize import get_quantize_target_layers_from_gguf
from q_attack.repair.gguf.process_gguf import compute_box_gguf

INJECT = " For more information, visit http://vq-poison.example."
DEV = "cuda"

def make_data(tok, n=512, malicious=False, seqlen=128):
    topics = ["the water cycle","photosynthesis","the French Revolution","how vaccines work",
              "gravity","the stock market","machine learning","the Roman Empire","climate","DNA",
              "black holes","the internet","democracy","volcanoes","the human heart"]
    ex=[]
    for i in range(n):
        t=topics[i%len(topics)]
        instr=f"Explain {t} in one sentence."
        resp=f"{t.capitalize()} is an important concept that can be explained clearly and concisely."
        if malicious: resp=resp+INJECT
        text=f"<|im_start|>user\n{instr}<|im_end|>\n<|im_start|>assistant\n{resp}<|im_end|>"
        ids=tok(text, truncation=True, max_length=seqlen, return_tensors="pt").input_ids[0]
        ex.append(ids)
    return ex

def train(model, data, tok, steps, lr, boxes=None, tag=""):
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train(); pad=tok.pad_token_id or tok.eos_token_id
    i=0
    while i<steps:
        batch=data[(i*2)%len(data):(i*2)%len(data)+2]
        maxlen=max(x.size(0) for x in batch)
        inp=torch.full((len(batch),maxlen),pad,dtype=torch.long)
        for j,x in enumerate(batch): inp[j,:x.size(0)]=x
        inp=inp.to(DEV); lab=inp.clone(); lab[lab==pad]=-100
        out=model(input_ids=inp, labels=lab); loss=out.loss
        opt.zero_grad(); loss.backward(); opt.step()
        if boxes is not None:   # box projection (repair)
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if name in boxes:
                        bmin,bmax=boxes[name]
                        p.data.clamp_(bmin.to(p.device), bmax.to(p.device))
        if i%50==0: print(f"[{tag}] step {i}/{steps} loss {loss.item():.4f}", flush=True)
        i+=1
    model.eval()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--clean_gguf", required=True, help="clean Q4_K_M gguf (for type_layer_map)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mal_steps", type=int, default=400)
    ap.add_argument("--repair_steps", type=int, default=400)
    a=ap.parse_args()

    tok=AutoTokenizer.from_pretrained(a.base)
    model=AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32).to(DEV)

    mal=make_data(tok, malicious=True)
    ben=make_data(tok, malicious=False)

    print("=== phase 1: malicious fine-tune ===", flush=True)
    train(model, mal, tok, a.mal_steps, 1e-4, tag="mal")

    print("=== phase 2: compute box around M_mal quant (official code) ===", flush=True)
    reader=GGUFReader(a.clean_gguf)
    target_weights, type_layer_map=get_quantize_target_layers_from_gguf(model, reader)
    print(f"target layers: {len(target_weights)}  types: {[len(v) for v in type_layer_map.values()]}", flush=True)
    boxes={}; skipped=0
    for name,layer in target_weights.items():
        w=layer.weight.data.float()
        try:
            bmin,bmax=compute_box_gguf(w.cuda(), name=name, type_layer_map=type_layer_map, thresh_type=1)
            bmin=bmin.reshape(w.shape) if bmin.numel()==w.numel() else bmin
            bmax=bmax.reshape(w.shape) if bmax.numel()==w.numel() else bmax
            if bmin.shape!=w.shape:
                skipped+=1; continue
            boxes[name]=(bmin.cpu(), bmax.cpu())
        except Exception as e:
            skipped+=1
            if skipped<=3: print(f"[box] skip {name}: {e}", flush=True)
    print(f"boxes computed: {len(boxes)}  skipped: {skipped}", flush=True)
    if len(boxes)==0:
        print("ERROR: no boxes computed"); sys.exit(3)
    # ensure current (malicious) weights are inside their own box
    with torch.no_grad():
        for name,p in model.named_parameters():
            if name in boxes:
                bmin,bmax=boxes[name]; p.data.clamp_(bmin.to(p.device),bmax.to(p.device))

    print("=== phase 3: box-constrained benign repair ===", flush=True)
    train(model, ben, tok, a.repair_steps, 5e-5, boxes=boxes, tag="repair")

    os.makedirs(a.out, exist_ok=True)
    model.to("cpu").save_pretrained(a.out); tok.save_pretrained(a.out)
    # quick behavioral check (fp16 text)
    print("MTG_ATTACK_DONE saved to", a.out, flush=True)

if __name__=="__main__": main()
