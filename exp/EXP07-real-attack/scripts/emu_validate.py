import os, sys, numpy as np, torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM
from q_attack.repair.gguf.emulator import Q245KEmulator

print("downloading qwen2.5-0.5b ...", flush=True)
d=snapshot_download("Qwen/Qwen2.5-0.5B", allow_patterns=["*.safetensors","*.json","*.txt"])
m=AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.float32)
rel=[]  # box_width / weight_std, per weight
for name,p in m.named_parameters():
    if p.ndim!=2: continue
    if not any(k in name for k in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]): continue
    w=p.data.float().flatten()
    n=(w.numel()//256)*256
    if n==0: continue
    wb=w[:n].reshape(-1,8,32).cuda()
    e=Q245KEmulator(wb, num_bit=4); e.quantize()
    bmin,bmax=e.get_width()
    width=(bmax-bmin).abs().flatten().cpu().numpy()
    std=float(w[:n].std()) or 1e-12
    r=width/std
    rel.append(r[::max(1,r.size//2000)].astype(np.float32))
R=np.concatenate(rel)
pcts={p:float(np.percentile(R,p)) for p in [10,50,90,99]}
print("REAL Q4_K box-width / weight_std (quantization step):", {k:round(v,4) for k,v in pcts.items()}, flush=True)
print("compare: synthetic QCB δ_A median≈0.27 ; natural ℛ_N median≈0.014", flush=True)
import json; json.dump({"box_width_over_std_pcts":pcts}, open(os.environ.get("OUT_JSON","emu_validate.json"),"w"))
print("EMU_VALIDATE_DONE", flush=True)
