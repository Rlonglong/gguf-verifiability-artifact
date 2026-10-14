#!/usr/bin/env python3
"""Minimal GGUF key-value header parser: reads only the metadata KV block and extracts verifiability signals.
Reads a local file, or with --url fetches only the header via HTTP range requests (no full download).

Used by the census (EXP02/EXP03) to check whether a GGUF records
  - quantization format revision: general.quantization_version / general.file_type
  - source model: general.base_model.* / general.source.*
  - imatrix provenance: quantize.imatrix.file / .dataset / .entries_count / .chunks_count
  - producer name: general.quantized_by (usually absent)
GGUF has no standard field for the execution environment (release/build, compiler, CPU, flags).

Usage:
  python3 tools/gguf_header_probe.py --url <gguf-url> [--max-mb 12] [--json out.json]
  python3 tools/gguf_header_probe.py local.gguf
"""
import sys, os, struct, json, argparse, urllib.request

# GGUF value types
U8,I8,U16,I16,U32,I32,F32,BOOL,STRING,ARRAY,U64,I64,F64 = range(13)
SCALAR_SIZE = {U8:1,I8:1,U16:2,I16:2,U32:4,I32:4,F32:4,BOOL:1,U64:8,I64:8,F64:8}
SCALAR_FMT  = {U8:"<B",I8:"<b",U16:"<H",I16:"<h",U32:"<I",I32:"<i",F32:"<f",BOOL:"<?",U64:"<Q",I64:"<q",F64:"<d"}

# keys of interest (other arrays/tokens are skipped)
WANT_PREFIXES = ("general.", "quantize.", "split.")
WANT_EXACT = {"general.architecture","general.name","general.basename","general.quantization_version",
              "general.file_type","general.quantized_by","general.size_label"}

class Truncated(Exception):
    def __init__(self, need): self.need = need

class Reader:
    def __init__(self, buf): self.b=buf; self.o=0
    def need(self, n):
        if self.o+n > len(self.b): raise Truncated(self.o+n)
    def u32(self): self.need(4); v=struct.unpack_from("<I",self.b,self.o)[0]; self.o+=4; return v
    def u64(self): self.need(8); v=struct.unpack_from("<Q",self.b,self.o)[0]; self.o+=8; return v
    def scalar(self,t):
        sz=SCALAR_SIZE[t]; self.need(sz); v=struct.unpack_from(SCALAR_FMT[t],self.b,self.o)[0]; self.o+=sz; return v
    def string(self):
        n=self.u64(); self.need(n); s=self.b[self.o:self.o+n]; self.o+=n
        try: return s.decode("utf-8","replace")
        except: return repr(s)
    def skip(self,n): self.need(n); self.o+=n

def parse_kv(buf, want_only=True):
    r=Reader(buf)
    magic=r.b[:4]
    if magic!=b"GGUF": raise ValueError("not a GGUF (magic=%r)"%magic)
    r.o=4
    version=r.u32(); tensor_count=r.u64(); kv_count=r.u64()
    kv={}
    for _ in range(kv_count):
        key=r.string(); vtype=r.u32()
        keep = (not want_only) or key in WANT_EXACT or key.startswith(WANT_PREFIXES)
        if vtype==STRING:
            val=r.string();
            if keep: kv[key]=val
        elif vtype==ARRAY:
            et=r.u32(); n=r.u64()
            if et==STRING:
                # skip element by element (keep small arrays)
                vals=[]
                for i in range(n):
                    s=r.string()
                    if keep and n<=64: vals.append(s)
                if keep: kv[key]=vals if n<=64 else f"<array:string×{n}>"
            elif et in SCALAR_SIZE:
                r.skip(SCALAR_SIZE[et]*n)
                if keep: kv[key]=f"<array:{et}×{n}>"
            else:
                raise ValueError(f"unknown array elem type {et}")
        elif vtype in SCALAR_SIZE:
            v=r.scalar(vtype)
            if keep: kv[key]=v
        else:
            raise ValueError(f"unknown value type {vtype} for key {key}")
    return {"version":version,"tensor_count":tensor_count,"kv_count":kv_count,"kv":kv,"kv_end_offset":r.o}

def http_range(url, nbytes):
    req=urllib.request.Request(url, headers={"Range":f"bytes=0-{nbytes-1}","User-Agent":"vq-census/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()

def probe(path=None, url=None, max_mb=12):
    if url:
        buf=b""; want=1<<20   # start at 1 MB, double when needed
        while True:
            buf=http_range(url, want)
            try:
                res=parse_kv(buf); res["fetched_bytes"]=len(buf); return res
            except Truncated as t:
                if want >= max_mb*(1<<20):
                    res={"error":"header truncated","need_bytes":t.need,"fetched_bytes":len(buf)}
                    return res
                want=min(max_mb*(1<<20), max(t.need, want*2))
    else:
        buf=open(path,"rb").read(max_mb*(1<<20))
        try: res=parse_kv(buf); res["fetched_bytes"]=len(buf); return res
        except Truncated as t: return {"error":"header truncated","need_bytes":t.need}

ENV_HINTS = ("llama_cpp","llamacpp","build","commit","compiler","ffp","contract",
             "toolchain","march","cpu","flags","git")
def verifiability(kv):
    """Extract the census signals from the KV block."""
    g=lambda k: kv.get(k)
    imx = {k:v for k,v in kv.items() if k.startswith("quantize.imatrix")}
    # scan for any KV that could pin the llama.cpp release/build environment (quantization_version is the format revision and does not count)
    env_keys=[k for k in kv if any(h in k.lower() for h in ENV_HINTS)
              and k!="general.quantization_version"]
    return {
        "quantization_version": g("general.quantization_version"),  # GGUF format revision, not the llama.cpp release
        "file_type": g("general.file_type"),
        "architecture": g("general.architecture"),
        "quantized_by": g("general.quantized_by"),   # producer name, not a version
        "base_model_count": g("general.base_model.count"),
        "base_model_0_repo": g("general.base_model.0.repo_url") or g("general.base_model.0.name"),
        "source_url": g("general.source.url"),
        "imatrix_fields": imx,           # non-empty = imatrix provenance recorded; empty = no imatrix, or removed with --hide-imatrix
        "imatrix_file": imx.get("quantize.imatrix.file"),
        "imatrix_dataset": imx.get("quantize.imatrix.dataset"),
        "has_imatrix_provenance": bool(imx),
        "env_version_keys": env_keys,          # empty in every artifact we scanned
        "records_env_version": bool(env_keys), # machine-readable build environment/version present?
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("path", nargs="?"); ap.add_argument("--url"); ap.add_argument("--max-mb",type=int,default=12)
    ap.add_argument("--json"); ap.add_argument("--all", action="store_true", help="print all KVs (except large arrays)")
    a=ap.parse_args()
    res=probe(path=a.path, url=a.url, max_mb=a.max_mb)
    if "error" in res: print(json.dumps(res,ensure_ascii=False)); sys.exit(2)
    v=verifiability(res["kv"])
    out={"version":res["version"],"tensor_count":res["tensor_count"],"kv_count":res["kv_count"],
         "fetched_bytes":res["fetched_bytes"],"verifiability":v}
    if a.all: out["kv"]=res["kv"]
    print(json.dumps(out,ensure_ascii=False,indent=2))
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)),exist_ok=True)
        json.dump(out,open(a.json,"w",encoding="utf-8"),ensure_ascii=False,indent=2)

if __name__=="__main__": main()
