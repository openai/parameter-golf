#!/usr/bin/env python3
"""Evaluate MLP pruning masks by zero-ablation on the local PR1493 model."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
from typing import Any
import torch, torch.distributed as dist
import train_gpt_pr1493 as pr1493

def load_torch(path:Path)->Any:
    try: return torch.load(path,map_location="cpu",weights_only=True)
    except TypeError: return torch.load(path,map_location="cpu")

def unwrap_state(obj:Any):
    if isinstance(obj,dict) and isinstance(obj.get("model"),dict): obj=obj["model"]
    if not isinstance(obj,dict): raise TypeError("expected state dict")
    state={str(k):v for k,v in obj.items() if torch.is_tensor(v)}
    if not state: raise ValueError("checkpoint has no tensors")
    return state

def configure(h,args):
    data=args.data_path or os.environ.get("DATA_PATH"); tok=args.tokenizer_path or os.environ.get("TOKENIZER_PATH")
    if data: h.datasets_dir=data; h.train_files=os.path.join(data,"fineweb_train_*.bin"); h.val_files=os.path.join(data,"fineweb_val_*.bin")
    if tok: h.tokenizer_path=tok
    h.val_batch_tokens=int(os.environ.get("VAL_BATCH_TOKENS",str(args.val_batch_tokens))); h.val_loss_every=0; h.logfile=None

def discover(roots,pattern):
    paths=[]
    for r in roots:
        if r.is_file(): paths.append(r)
        elif r.is_dir(): paths.extend(r.rglob(pattern))
        else: raise FileNotFoundError(r)
    paths=sorted({p.resolve():p for p in paths}.values())
    if not paths: raise FileNotFoundError("no mask files found")
    return paths

def apply_mask(model,masks):
    state=model.state_dict(); per={}
    with torch.no_grad():
        for name,keep_mask in sorted(masks.items()):
            keep=keep_mask.detach().cpu().bool(); prune=~keep; per[name]=int(prune.sum())
            if per[name]==0: continue
            fk,pk=f"{name}.fc.weight",f"{name}.proj.weight"
            if fk not in state or pk not in state: raise KeyError(f"missing {fk} or {pk}")
            pd=prune.to(state[fk].device); state[fk][pd,:]=0; state[pk][:,pd]=0
    return per

def eval_val(model,compiled,h,device,val_data):
    torch.cuda.synchronize(); t0=time.perf_counter(); loss,bpb=pr1493.eval_val(h,device,val_data,compiled); torch.cuda.synchronize()
    return {"val_loss":float(loss),"val_bpb":float(bpb),"eval_time_ms":float((time.perf_counter()-t0)*1000)}

def meta(path,payload,per):
    s=payload.get("summary",{}); pruned=int(s.get("pruned_channels",sum(per.values()))); removed=int(s.get("estimated_int6_packed_bytes_removed",0))
    return {"name":path.stem,"mask_path":str(path),"policy":str(payload.get("policy",s.get("policy","unknown"))),"fraction":float(payload.get("fraction",s.get("fraction",0.0))),"pruned_channels":pruned,"estimated_int6_packed_bytes_removed":removed,"estimated_current_quant_raw_bytes_removed":int(s.get("estimated_current_quant_raw_bytes_removed",0)),"active_modules":int(s.get("active_modules",sum(1 for v in per.values() if v))),"per_module_pruned":per}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--artifact-dir",type=Path,required=True); ap.add_argument("--checkpoint",type=Path); ap.add_argument("--mask-root",type=Path,action="append"); ap.add_argument("--pattern",default="*_masks.pt"); ap.add_argument("--output-json",type=Path); ap.add_argument("--data-path",default=""); ap.add_argument("--tokenizer-path",default=""); ap.add_argument("--val-batch-tokens",type=int,default=524288); ap.add_argument("--no-compile",action="store_true")
    args=ap.parse_args(); local_rank=int(os.environ.get("LOCAL_RANK","0")); distributed="RANK" in os.environ; rank=int(os.environ.get("RANK","0")); world=int(os.environ.get("WORLD_SIZE","1"))
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device=torch.device("cuda",local_rank); torch.cuda.set_device(device)
    if distributed: dist.init_process_group(backend="nccl",device_id=device); dist.barrier()
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True; torch.set_float32_matmul_precision("high")
    h=pr1493.Hyperparameters(); configure(h,args); pr1493.set_logging_hparams(h)
    ckpt=args.checkpoint or args.artifact_dir/"final_ema_fp.pt"; roots=args.mask_root or [args.artifact_dir/"mlp_pruning_masks_cap",args.artifact_dir/"mlp_pruning_masks_softcap"]; out=args.output_json or args.artifact_dir/"mlp_pruning_zero_ablation_results.json"
    base=unwrap_state(load_torch(ckpt)); val_data=pr1493.ValidationData(h,device); model=pr1493.GPT(h).to(device).bfloat16(); pr1493.restore_fp32_params(model); model.load_state_dict(base,strict=True); model.eval(); compiled=model if args.no_compile else torch.compile(model,dynamic=False,fullgraph=True)
    results=[]; baseline=eval_val(model,compiled,h,device,val_data)
    if rank==0: results.append({"name":"baseline","policy":"baseline","checkpoint":str(ckpt),**baseline})
    for mp in discover(roots,args.pattern):
        payload=load_torch(mp); model.load_state_dict(base,strict=True); per=apply_mask(model,payload["masks"]); row=eval_val(model,compiled,h,device,val_data)
        if rank==0: results.append({**meta(mp,payload,per),**row})
    if rank==0:
        base_bpb=float(results[0]["val_bpb"])
        for r in results[1:]: r["delta_bpb_vs_baseline"]=float(r["val_bpb"])-base_bpb; r["delta_bpb_per_100kb_removed"]=r["delta_bpb_vs_baseline"]/(max(1,int(r.get("estimated_int6_packed_bytes_removed",0)))/100000.0)
        report={"artifact_dir":str(args.artifact_dir),"checkpoint":str(ckpt),"mask_roots":[str(x) for x in roots],"world_size":world,"results":results,"leaderboard":sorted(results[1:],key=lambda r:(r["delta_bpb_vs_baseline"],r["delta_bpb_per_100kb_removed"]))}
        out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,sort_keys=True)); print(json.dumps({"output_json":str(out),"evaluated_masks":len(results)-1,"best":report["leaderboard"][:5]},indent=2,sort_keys=True))
    if distributed: dist.barrier(); dist.destroy_process_group()
if __name__=="__main__": main()
