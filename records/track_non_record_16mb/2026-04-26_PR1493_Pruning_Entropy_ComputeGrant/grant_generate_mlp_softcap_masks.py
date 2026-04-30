#!/usr/bin/env python3
"""Generate soft-cap structured MLP pruning masks.

This is the public soft-cap policy described in the draft PR. It ranks MLP
hidden channels using a blend of within-block rank and global rank, then selects
channels globally subject to a relaxed per-block cap.

Relevant tuning knobs:
- --score-weights: score blend, default activation_weighted_score=0.70,norm_score=0.30
- --local-rank-weight: local/global rank blend, default 0.75
- --cap-multiplier: relaxed cap multiplier, default 1.75
- --floor-multiplier: optional per-block floor, default 0.0

The pushed 5% soft-cap result used the defaults above with fraction=0.05.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Any
import torch

SCORE_KEYS=("activation_weighted_score","norm_score")

def load_torch(path: Path)->Any:
    try: return torch.load(path,map_location="cpu",weights_only=True)
    except TypeError: return torch.load(path,map_location="cpu")

def unwrap_state(obj: Any):
    if isinstance(obj,dict) and isinstance(obj.get("model"),dict): obj=obj["model"]
    return {str(k):v for k,v in obj.items() if torch.is_tensor(v)} if isinstance(obj,dict) else None

def parse_floats(s:str):
    vals=[float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals or any(v<=0 or v>=1 for v in vals): raise ValueError("fractions must be in (0,1)")
    return vals

def parse_weights(s:str):
    w={}
    for p in s.split(","):
        if not p.strip(): continue
        k,sep,v=p.partition("=")
        if sep!="=": raise ValueError(f"score weight must be key=value, got {p!r}")
        val=float(v)
        if val<0: raise ValueError("score weights must be non-negative")
        w[k.strip()]=val
    tot=sum(w.values())
    if tot<=0: raise ValueError("positive weights required")
    return {k:v/tot for k,v in w.items() if v>0}

def clean(x):
    y=x.detach().cpu().float().flatten(); finite=torch.isfinite(y)
    if finite.all(): return y
    y=y.clone(); y[~finite]=y[finite].max() if finite.any() else 1.0; return y

def rank01(x):
    x=clean(x); n=x.numel()
    if n==1: return torch.zeros_like(x)
    order=torch.argsort(x, stable=True); ranks=torch.empty(n,dtype=torch.float32); ranks[order]=torch.arange(n,dtype=torch.float32); return ranks/float(n-1)

def infer_shape(name,row,state,default_dim):
    score=next((row[k] for k in SCORE_KEYS if k in row),None)
    if score is None: raise KeyError(f"{name} has no score tensor")
    hidden=int(score.numel())
    if state:
        fc,proj=state.get(f"{name}.fc.weight"),state.get(f"{name}.proj.weight")
        if fc is not None and proj is not None: return int(fc.shape[1]), hidden
    return default_dim, hidden

def build(stats, weights, state, default_dim, local_rank_weight):
    if not 0.0<=local_rank_weight<=1.0: raise ValueError("local_rank_weight must be in [0,1]")
    shapes={}; raw_by={}; loc_by={}
    for name in sorted(stats):
        row=stats[name]; model_dim,hidden=infer_shape(name,row,state,default_dim); shapes[name]={"model_dim":model_dim,"hidden":hidden}
        raw=torch.zeros(hidden); loc=torch.zeros(hidden); used=0.0
        for k,w in weights.items():
            if k not in row: continue
            sc=clean(row[k]); assert sc.numel()==hidden
            raw += w*sc; loc += w*rank01(sc); used += w
        if used<=0: raise KeyError(f"{name} lacks requested score keys")
        raw_by[name]=raw/used; loc_by[name]=loc/used
    glob_rank=rank01(torch.cat([raw_by[n] for n in sorted(raw_by)])); glob_by={}; off=0
    for n in sorted(raw_by): glob_by[n],off=glob_rank[off:off+raw_by[n].numel()],off+raw_by[n].numel()
    entries=[]
    for n in sorted(shapes):
        score=local_rank_weight*loc_by[n]+(1-local_rank_weight)*glob_by[n]
        for c in range(shapes[n]["hidden"]): entries.append({"module":n,"channel":c,"score":float(score[c]),"raw_score":float(raw_by[n][c]),"model_dim":shapes[n]["model_dim"]})
    entries.sort(key=lambda e:(e["score"],e["module"],e["channel"]))
    return entries,shapes

def select(entries,shapes,fraction,cap_multiplier,floor_multiplier):
    if cap_multiplier<=0: raise ValueError("cap_multiplier must be positive")
    if floor_multiplier<0: raise ValueError("floor_multiplier must be non-negative")
    total=sum(s["hidden"] for s in shapes.values()); target=max(1,round(total*fraction))
    caps={m:min(s["hidden"],max(1,math.ceil(s["hidden"]*fraction*cap_multiplier))) for m,s in shapes.items()}
    floors={m:min(s["hidden"],math.floor(s["hidden"]*fraction*floor_multiplier)) for m,s in shapes.items()}
    by_module={m:[] for m in shapes}
    for e in entries: by_module[e["module"]].append(e)
    selected=[]; per={m:0 for m in shapes}; selected_ids=set()
    for m,floor in floors.items():
        for e in by_module[m][:floor]:
            selected.append(e); selected_ids.add((e["module"],e["channel"])); per[m]+=1
    for e in entries:
        if len(selected)>=target: break
        key=(e["module"],e["channel"]); m=e["module"]
        if key in selected_ids or per[m]>=caps[m]: continue
        selected.append(e); selected_ids.add(key); per[m]+=1
    if len(selected)!=target: raise RuntimeError("soft cap too tight; increase --cap-multiplier or reduce --floor-multiplier")
    return selected,caps,floors

def payload(sel,caps,floors,shapes,fraction,args):
    masks={m:torch.ones(s["hidden"],dtype=torch.bool) for m,s in shapes.items()}; per={m:{"hidden":s["hidden"],"pruned":0,"channels":[]} for m,s in shapes.items()}
    for e in sel:
        m,c=e["module"],int(e["channel"]); masks[m][c]=False; per[m]["pruned"]+=1; per[m]["channels"].append(c)
    total=sum(s["hidden"] for s in shapes.values()); removed=sum(2*int(e["model_dim"]) for e in sel)
    summary={"policy":"soft_cap_global","fraction":fraction,"cap_multiplier":args.cap_multiplier,"floor_multiplier":args.floor_multiplier,"score_weights":parse_weights(args.score_weights),"local_rank_weight":args.local_rank_weight,"total_channels":total,"pruned_channels":len(sel),"kept_channels":total-len(sel),"per_module_caps":caps,"per_module_floors":floors,"per_module":per,"active_modules":sum(1 for x in per.values() if x["pruned"]),"max_module_share_of_pruned":max((x["pruned"]/max(1,len(sel)) for x in per.values()),default=0.0),"estimated_fp16_bytes_removed":removed*2,"estimated_int6_packed_bytes_removed":math.ceil(removed*6/8),"estimated_current_quant_raw_bytes_removed":sum(2*int(e["model_dim"])+2 for e in sel),"score_mean_pruned":sum(float(e["score"]) for e in sel)/len(sel),"raw_score_mean_pruned":sum(float(e["raw_score"]) for e in sel)/len(sel)}
    return masks,summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--artifact-dir",type=Path,required=True); ap.add_argument("--fractions",default="0.01,0.02,0.05"); ap.add_argument("--score-weights",default="activation_weighted_score=0.70,norm_score=0.30"); ap.add_argument("--local-rank-weight",type=float,default=0.75); ap.add_argument("--cap-multiplier",type=float,default=1.75); ap.add_argument("--floor-multiplier",type=float,default=0.0); ap.add_argument("--checkpoint",type=Path); ap.add_argument("--default-model-dim",type=int,default=512); ap.add_argument("--output-dir",type=Path)
    args=ap.parse_args(); stats=load_torch(args.artifact_dir/"mlp_channel_stats.pt"); ckpt=args.checkpoint or args.artifact_dir/"final_ema_fp.pt"; state=unwrap_state(load_torch(ckpt)) if ckpt.is_file() else None
    entries,shapes=build(stats,parse_weights(args.score_weights),state,args.default_model_dim,args.local_rank_weight); out=args.output_dir or args.artifact_dir/"mlp_pruning_masks_softcap"; out.mkdir(parents=True,exist_ok=True); summaries=[]
    for f in parse_floats(args.fractions):
        sel,caps,floors=select(entries,shapes,f,args.cap_multiplier,args.floor_multiplier); masks,summary=payload(sel,caps,floors,shapes,f,args); pct=round(f*10000); path=out/f"mlp_softcap_prune_{pct:04d}bp_masks.pt"; torch.save({"policy":"soft_cap_global","fraction":f,"masks":masks,"summary":summary},path); summary["mask_path"]=str(path); summaries.append(summary)
    (out/"summary.json").write_text(json.dumps({"summaries":summaries},indent=2,sort_keys=True)); print(json.dumps({"output_dir":str(out),"generated":len(summaries)},indent=2))
if __name__=="__main__": main()
