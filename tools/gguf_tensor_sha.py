#!/usr/bin/env python3
# stdlib-only: sha256 of the GGUF tensor-data region (skips header, KV and tensor-info blocks)
import sys,struct,hashlib
SZ={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def rd(f,fmt): return struct.unpack("<"+fmt,f.read(struct.calcsize(fmt)))[0]
def rstr(f): return f.read(rd(f,"Q"))
def skip_val(f,t):
    if t==8: rstr(f)
    elif t==9:
        et=rd(f,"I"); n=rd(f,"Q")
        if et==8:
            for _ in range(n): rstr(f)
        else: f.seek(SZ[et]*n,1)
    else: f.seek(SZ[t],1)
for p in sys.argv[1:]:
    with open(p,"rb") as f:
        assert f.read(4)==b"GGUF"; rd(f,"I"); nt=rd(f,"Q"); nkv=rd(f,"Q"); align=32
        for _ in range(nkv):
            k=rstr(f); t=rd(f,"I")
            if k==b"general.alignment" and t in (4,5): align=rd(f,"I"); continue
            skip_val(f,t)
        for _ in range(nt):
            rstr(f); nd=rd(f,"I"); f.seek(8*nd,1); rd(f,"I"); rd(f,"Q")
        off=(f.tell()+align-1)//align*align; f.seek(off); h=hashlib.sha256()
        for b in iter(lambda:f.read(1<<24),b""): h.update(b)
    print(f"{h.hexdigest()[:16]}  {p}")
