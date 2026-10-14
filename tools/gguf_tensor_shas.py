#!/usr/bin/env python3
# stdlib-only: per-tensor sha256 of a GGUF's tensor data.
# Emits JSON [{name, ggml_type, offset, nbytes, sha}] sorted by data offset.
# Per-tensor byte length is derived from consecutive data offsets (the last tensor runs to EOF),
# so no ggml type-size table is needed and the tool is format-revision agnostic.
import sys, struct, hashlib, json, os
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def rd(f, fmt): return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]
def rstr(f): return f.read(rd(f, "Q"))


def skip_val(f, t):
    if t == 8:
        rstr(f)
    elif t == 9:
        et = rd(f, "I"); n = rd(f, "Q")
        if et == 8:
            for _ in range(n): rstr(f)
        else:
            f.seek(SZ[et] * n, 1)
    else:
        f.seek(SZ[t], 1)


def parse(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"; rd(f, "I"); nt = rd(f, "Q"); nkv = rd(f, "Q"); align = 32
        for _ in range(nkv):
            k = rstr(f); t = rd(f, "I")
            if k == b"general.alignment" and t in (4, 5):
                align = rd(f, "I"); continue
            skip_val(f, t)
        infos = []
        for _ in range(nt):
            name = rstr(f).decode("utf-8", "ignore"); nd = rd(f, "I")
            f.seek(8 * nd, 1); gt = rd(f, "I"); off = rd(f, "Q")
            infos.append([name, gt, off])
        data_start = (f.tell() + align - 1) // align * align
        fsize = os.path.getsize(path)
        order = sorted(range(nt), key=lambda i: infos[i][2])
        out = []
        for rank, i in enumerate(order):
            name, gt, off = infos[i]
            end = infos[order[rank + 1]][2] if rank + 1 < nt else (fsize - data_start)
            nbytes = end - off
            f.seek(data_start + off); h = hashlib.sha256(); left = nbytes
            while left > 0:
                b = f.read(min(1 << 24, left))
                if not b: break
                h.update(b); left -= len(b)
            out.append({"name": name, "ggml_type": gt, "offset": off, "nbytes": nbytes,
                        "sha": h.hexdigest()})
        out.sort(key=lambda r: r["offset"])
        return out


if __name__ == "__main__":
    ap = sys.argv[1:]
    if not ap:
        print("usage: gguf_tensor_shas.py FILE.gguf [FILE2.gguf ...]  (JSON per file to <file>.shas.json)", file=sys.stderr)
        sys.exit(2)
    for p in ap:
        rows = parse(p)
        outp = p + ".shas.json"
        json.dump({"file": os.path.basename(p), "n_tensors": len(rows), "tensors": rows},
                  open(outp, "w"))
        print(f"{len(rows):4d} tensors  {outp}")
