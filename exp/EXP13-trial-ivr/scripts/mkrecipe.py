#!/usr/bin/env python3
"""Build a --tensor-type-file that pins every tensor to the type it has in a published GGUF.

Reads the published file's per-tensor sha JSON (tools/gguf_tensor_shas.py output) and writes one line
  ^<escaped tensor name>$=<ggml type>
per tensor. Feeding this to `llama-quantize --tensor-type-file` forces the exact per-tensor recipe of the
published artifact, so a re-quantization differs from it only through the build environment, never the type
map. This makes the recipe an explicit, supplied coordinate rather than a hidden one.

Usage: python3 mkrecipe.py PUBLISHED.gguf.shas.json OUT_recipe.txt
"""
import json, re, sys
NAME = {0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0", 10: "q2_k",
        11: "q3_k", 12: "q4_k", 13: "q5_k", 14: "q6_k", 15: "q8_k", 16: "iq2_xxs", 17: "iq2_xs",
        18: "iq3_xxs", 19: "iq1_s", 20: "iq4_nl", 21: "iq3_s", 22: "iq2_s", 23: "iq4_xs", 29: "iq1_m",
        30: "bf16"}


def main():
    src, out = sys.argv[1], sys.argv[2]
    d = json.load(open(src)); n = 0
    with open(out, "w") as f:
        for t in d["tensors"]:
            gt = t["ggml_type"]
            if gt not in NAME:
                print(f"WARN unmapped type {gt} for {t['name']}", file=sys.stderr); continue
            f.write(f"^{re.escape(t['name'])}$={NAME[gt]}\n"); n += 1
    print(f"{n} recipe lines -> {out}")


if __name__ == "__main__":
    main()
