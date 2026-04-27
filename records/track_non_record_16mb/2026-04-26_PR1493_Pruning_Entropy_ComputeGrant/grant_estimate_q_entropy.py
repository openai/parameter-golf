#!/usr/bin/env python3
"""Estimate static arithmetic/rANS codelengths for quantized q tensors.

Includes the two PR-described approaches:
1. one q-symbol model per quantized tensor,
2. one q-symbol model per tensor class.
"""
from __future__ import annotations
import argparse, io, json, math, re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import brotli, torch

BSHF_MAGIC=b"BSHF"

def byte_unshuffle(data:bytes)->bytes:
    if len(data)<5 or data[:4]!=BSHF_MAGIC: return data
    stride=data[4]
    if stride<2: return data[5:]
    payload=data[5:]; n=len(payload); out=bytearray(n); src=0
    for pos in range(stride):
        ln=n//stride+(1 if pos<n%stride else 0); out[pos::stride]=payload[src:src+ln]; src+=ln
    return bytes(out)

def load_quantized(path:Path)->tuple[dict[str,Any],int,int]:
    compressed=path.read_bytes(); raw=byte_unshuffle(brotli.decompress(compressed)); state=torch.load(io.BytesIO(raw),map_location="cpu")
    if not isinstance(state,dict) or "w" not in state or "m" not in state: raise TypeError(f"unexpected payload in {path}")
    return state,len(compressed),len(raw)

def bits_from_meta(meta:str):
    m=re.search(r"int(\d+)",str(meta)); return int(m.group(1)) if m else None

def tensor_class(name:str):
    if "tok_emb" in name: return "embedding"
    if ".mlp." in name: return "mlp"
    if ".attn.c_q." in name: return "q"
    if ".attn.c_k." in name: return "k"
    if ".attn.c_v." in name: return "v"
    if ".attn.proj." in name: return "o"
    return "other"

def iter_quant(state,include_embedding):
    rows=[]
    for name,tensor in sorted(state["w"].items()):
        if not torch.is_tensor(tensor): continue
        bits=bits_from_meta(str(state["m"].get(name,"")))
        if bits is None: continue
        cls=tensor_class(str(name))
        if cls=="embedding" and not include_embedding: continue
        rows.append({"name":str(name),"tensor":tensor.detach().cpu().to(torch.int16).flatten(),"bits":bits,"class":cls})
    if not rows: raise ValueError("no quantized tensors found")
    return rows

def counts_for(tensor):
    vals,cnts=torch.unique(tensor,sorted=True,return_counts=True); return Counter({int(v):int(c) for v,c in zip(vals.tolist(),cnts.tolist())})

def entropy_payload_bytes(counts):
    total=sum(counts.values()); bits=0.0
    for c in counts.values():
        p=c/total; bits += -c*math.log2(p)
    return math.ceil(bits/8.0)

def estimate(counts):
    payload=entropy_payload_bytes(counts); header=5*len(counts); n=sum(counts.values())
    return {"symbols":len(counts),"numel":n,"entropy_payload_bytes":payload,"model_header_bytes":header,"total_bytes":payload+header,"bits_per_symbol":float((payload*8)/max(1,n))}

def per_tensor(rows):
    details=[]; total=0; raw=0
    for r in rows:
        est=estimate(counts_for(r["tensor"])); rb=int(r["tensor"].numel()); total+=int(est["total_bytes"]); raw+=rb
        details.append({"name":r["name"],"class":r["class"],"bits":r["bits"],"raw_q_bytes":rb,**est})
    details.sort(key=lambda x:int(x["total_bytes"])); return {"total_bytes":total,"raw_q_bytes":raw,"details":details}

def class_model(rows):
    grouped=defaultdict(Counter); raw=defaultdict(int)
    for r in rows:
        key=f"{r['class']}_int{r['bits']}"; grouped[key].update(counts_for(r["tensor"])); raw[key]+=int(r["tensor"].numel())
    models={}; total=0
    for k,c in sorted(grouped.items()):
        est=estimate(c); total+=int(est["total_bytes"]); models[k]={"raw_q_bytes":raw[k],**est}
    return {"total_bytes":total,"raw_q_bytes":sum(raw.values()),"class_models":models}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--quantized",type=Path,required=True); ap.add_argument("--output-json",type=Path); ap.add_argument("--include-embedding",action="store_true"); ap.add_argument("--top-k",type=int,default=12)
    args=ap.parse_args(); state,orig_brotli,orig_raw=load_quantized(args.quantized); rows=iter_quant(state,args.include_embedding); pt=per_tensor(rows); cm=class_model(rows)
    report={"quantized":str(args.quantized),"include_embedding":args.include_embedding,"original_file_brotli_bytes":orig_brotli,"original_raw_torchsave_bytes":orig_raw,"selected_quant_tensors":len(rows),"per_tensor_q_symbol_model":{"description":"static arithmetic/rANS estimate, one symbol model per tensor","total_bytes":pt["total_bytes"],"raw_q_bytes":pt["raw_q_bytes"],"best":pt["details"][:args.top_k],"worst":list(reversed(pt["details"][-args.top_k:]))},"class_q_symbol_model":{"description":"static arithmetic/rANS estimate, one symbol model per tensor class",**cm}}
    if args.output_json: args.output_json.parent.mkdir(parents=True,exist_ok=True); args.output_json.write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__": main()
