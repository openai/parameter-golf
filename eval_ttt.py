"""
Test-Time Training (TTT) Evaluation
====================================
Loads a trained model artifact (.ptz), runs sliding window eval with
online gradient updates on already-scored tokens. Each window:
  1. Score new tokens (record NLL) — BEFORE any TTT update
  2. TTT: gradient steps on the window (all tokens already graded)
  3. Slide forward — model is now adapted for next window

Usage (after training produces final_model.int6.ptz):
  PYTHONPATH=flash-attention/hopper:$PYTHONPATH torchrun --nproc_per_node=8 eval_ttt.py

Env vars:
  TTT_EPOCHS=8        gradient epochs per window (default 8)
  TTT_LR=1e-4         TTT learning rate (default 1e-4)
  TTT_STRIDE=64       sliding window stride (default 64)
  EVAL_SEQ_LEN=2048   window size (default 2048)
  MODEL_PATH=final_model.int6.ptz  (default)
"""
from __future__ import annotations
import copy,glob,io,math,os,sys,time,zlib
from pathlib import Path
try:
 import zstandard;_Z="zstd"
except ImportError:_Z="zlib"
import numpy as np;import sentencepiece as spm;import torch
import torch.distributed as dist;import torch.nn.functional as F
from torch import Tensor,nn
try:
 from flash_attn_interface import flash_attn_func as fa3
except ImportError:
 fa3=None
E=os.environ.get
# ─── Model architecture (must match training script) ─────────────────────────
_CP=tuple(p for p in E("CONTROL_TENSOR_NAME_PATTERNS","attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,ve_layer_scales,ve_shared.scale").split(",") if p)
class RMSNorm(nn.Module):
 def __init__(s,eps=None):super().__init__();s.eps=eps
 def forward(s,x):return F.rms_norm(x,(x.size(-1),),eps=s.eps)
class CastedLinear(nn.Linear):
 _qat=False
 def forward(s,x):
  w=s.weight.to(x.dtype);b=s.bias.to(x.dtype) if s.bias is not None else None
  return F.linear(x,w,b)
def restore_fp32(mod):
 with torch.no_grad():
  for n,p in mod.named_parameters():
   if(p.ndim<2 or any(pt in n for pt in _CP))and p.dtype!=torch.float32:p.data=p.data.float()
class Rotary(nn.Module):
 def __init__(s,dim,base=10000.0,tsl=1024,rd=0):
  super().__init__();s.dim=dim;s.base=base;s.tsl=tsl;s.rd=rd if rd>0 else dim
  s.register_buffer("inv_freq",1.0/(base**(torch.arange(0,s.rd,2,dtype=torch.float32)/s.rd)),persistent=False)
  s._sl=0;s._c=None;s._s=None
 def forward(s,sl,dev,dt):
  if s._c is None or s._sl!=sl or s._c.device!=dev:
   rd=s.rd
   if sl>s.tsl:nb=s.base*((sl/s.tsl)**(rd/(rd-2)));iv=1.0/(nb**(torch.arange(0,rd,2,dtype=torch.float32,device=dev)/rd))
   else:iv=s.inv_freq.to(dev)
   t=torch.arange(sl,device=dev,dtype=iv.dtype);fr=torch.outer(t,iv)
   s._c=fr.cos()[None,:,None,:];s._s=fr.sin()[None,:,None,:];s._sl=sl
  return s._c.to(dtype=dt),s._s.to(dtype=dt)
def apply_rope(x,cos,sin,rd=0):
 if rd>0 and rd<x.size(-1):
  xr,xp=x[...,:rd],x[...,rd:];h=rd//2;x1,x2=xr[...,:h],xr[...,h:]
  return torch.cat((torch.cat((x1*cos+x2*sin,x1*(-sin)+x2*cos),dim=-1),xp),dim=-1)
 h=x.size(-1)//2;x1,x2=x[...,:h],x[...,h:]
 return torch.cat((x1*cos+x2*sin,x1*(-sin)+x2*cos),dim=-1)
class CSA(nn.Module):
 def __init__(s,dim,nh,nkv,rb,qkg):
  super().__init__();s.nh=nh;s.nkv=nkv;s.hd=dim//nh;kd=nkv*s.hd
  s.c_q=CastedLinear(dim,dim,bias=False);s.c_k=CastedLinear(dim,kd,bias=False)
  s.c_v=CastedLinear(dim,kd,bias=False);s.proj=CastedLinear(dim,dim,bias=False);s.proj._zero_init=True
  s.q_gain=nn.Parameter(torch.full((nh,),qkg,dtype=torch.float32));s.rope_dims=0
  s.rotary=Rotary(s.hd,base=rb,tsl=1024);s.use_xsa=False
 def _xsa(s,y,v):
  B,T,H,D=y.shape;Hk=v.size(-2);g=H//Hk;yg=y.reshape(B,T,Hk,g,D)
  vn=F.normalize(v,dim=-1).unsqueeze(-2);pr=(yg*vn).sum(dim=-1,keepdim=True)*vn
  return(yg-pr).reshape(B,T,H,D)
 def forward(s,x,ve=None):
  B,T,d=x.shape;q=s.c_q(x).reshape(B,T,s.nh,s.hd);k=s.c_k(x).reshape(B,T,s.nkv,s.hd)
  v=s.c_v(x)
  if ve is not None:v=v+ve
  v=v.reshape(B,T,s.nkv,s.hd);q=F.rms_norm(q,(q.size(-1),));k=F.rms_norm(k,(k.size(-1),))
  co,si=s.rotary(T,x.device,q.dtype);q=apply_rope(q,co,si,s.rope_dims);k=apply_rope(k,co,si,s.rope_dims)
  q=q*s.q_gain.to(dtype=q.dtype)[None,None,:,None]
  if fa3 is not None:y=fa3(q,k,v,causal=True)
  else:y=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=True,enable_gqa=(s.nkv!=s.nh)).transpose(1,2)
  if s.use_xsa:y=s._xsa(y,v)
  return s.proj(y.reshape(B,T,d))
class SmearGate(nn.Module):
 def __init__(s,dim):super().__init__();s.gate=nn.Parameter(torch.zeros(dim,dtype=torch.float32))
 def forward(s,x):
  g=torch.sigmoid(s.gate.to(dtype=x.dtype))[None,None,:]
  return(1-g)*x+g*torch.cat([torch.zeros_like(x[:,:1]),x[:,:-1]],dim=1)
class BigramHash(nn.Module):
 def __init__(s,bvs,bd,md):
  super().__init__();s.bvs=bvs;s.embed=nn.Embedding(bvs,bd);nn.init.zeros_(s.embed.weight)
  s.proj=CastedLinear(bd,md,bias=False) if bd!=md else None
  if s.proj:nn.init.zeros_(s.proj.weight)
  s.scale=nn.Parameter(torch.tensor(0.05,dtype=torch.float32))
 def _hash(s,t):
  t=t.to(torch.int32);m=s.bvs-1;o=torch.empty_like(t);o[...,0]=m
  o[...,1:]=torch.bitwise_xor(36313*t[...,1:],27191*t[...,:-1])%m;return o.long()
 def forward(s,ids):
  h=s.embed(s._hash(ids));
  if s.proj:h=s.proj(h)
  return h*s.scale.to(dtype=h.dtype)
class VE(nn.Module):
 def __init__(s,vs,vd,md):
  super().__init__();s.embed=nn.Embedding(vs,vd);nn.init.normal_(s.embed.weight,std=0.01)
  s.proj=CastedLinear(vd,md,bias=False) if vd!=md else None
  if s.proj:nn.init.zeros_(s.proj.weight)
  s.scale=nn.Parameter(torch.tensor(0.1,dtype=torch.float32))
 def forward(s,ids):
  h=s.embed(ids);
  if s.proj:h=s.proj(h)
  return h*s.scale.to(dtype=h.dtype)
class MLP(nn.Module):
 def __init__(s,dim,mm):
  super().__init__();hid=int(mm*dim);s.fc=CastedLinear(dim,hid,bias=False)
  s.proj=CastedLinear(hid,dim,bias=False);s.proj._zero_init=True
 def forward(s,x):return s.proj(torch.relu(s.fc(x)).square())
class Block(nn.Module):
 def __init__(s,dim,nh,nkv,mm,rb,qkg,li=0,lns=False):
  super().__init__();s.attn_norm=RMSNorm();s.mlp_norm=RMSNorm()
  s.attn=CSA(dim,nh,nkv,rb,qkg);s.mlp=MLP(dim,mm)
  s.attn_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32))
  s.mlp_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32))
  s.resid_mix=nn.Parameter(torch.stack((torch.ones(dim),torch.zeros(dim))).float())
  s.ln_scale_factor=1.0/math.sqrt(li+1) if lns else 1.0
 def forward(s,x,x0,ve=None):
  m=s.resid_mix.to(dtype=x.dtype);xi=m[0][None,None,:]*x+m[1][None,None,:]*x0
  ao=s.attn(s.attn_norm(xi)*s.ln_scale_factor,ve=ve)
  xo=xi+s.attn_scale.to(dtype=xi.dtype)[None,None,:]*ao
  xo=xo+s.mlp_scale.to(dtype=xo.dtype)[None,None,:]*s.mlp(s.mlp_norm(xo)*s.ln_scale_factor)
  return xo
class GPT(nn.Module):
 def __init__(s,vs,nl,md,nh,nkv,mm,te,teis,lsc,rb,qkg,bvs=0,bd=128,xln=0,rd=0,lns=False,ve_on=False,ve_d=128,ve_l="9,10"):
  super().__init__();s._vetd=nkv*(md//nh);s.te=te;s.teis=teis;s.lsc=lsc
  s.tok_emb=nn.Embedding(vs,md)
  s.bigram=BigramHash(bvs,bd,md) if bvs>0 else None;s.smear=SmearGate(md)
  s.ne=nl//2;s.nd=nl-s.ne;s.ns=min(s.ne,s.nd)
  s.skip_weights=nn.Parameter(torch.ones(s.ns,md,dtype=torch.float32))
  s.blocks=nn.ModuleList([Block(md,nh,nkv,mm,rb,qkg,li=i,lns=lns) for i in range(nl)])
  if rd>0:
   hd=md//nh
   for b in s.blocks:b.attn.rope_dims=rd;b.attn.rotary=Rotary(hd,base=rb,tsl=1024,rd=rd)
  s.ve_li=[int(x) for x in ve_l.split(",") if x.strip()] if ve_on else [];kd=s._vetd
  if s.ve_li:s.ve_shared=VE(vs,ve_d,kd);s.ve_layer_scales=nn.ParameterList([nn.Parameter(torch.ones(1,dtype=torch.float32)) for _ in s.ve_li])
  else:s.ve_shared=None;s.ve_layer_scales=nn.ParameterList()
  s.value_embeds=nn.ModuleList();s.final_norm=RMSNorm()
  s.lm_head=None if te else CastedLinear(md,vs,bias=False)
  if s.lm_head:s.lm_head._zero_init=True
  s.mtp_heads=nn.ModuleList();s.mtp_num_heads=0;s.mtp_loss_weight=0
  if xln>0:
   for i in range(max(0,nl-xln),nl):s.blocks[i].attn.use_xsa=True
 def _gve(s,li,ids,vc):
  if s.ve_shared is None or li not in s.ve_li:return None
  if 've' not in vc:vc['ve']=s.ve_shared(ids)
  vi=s.ve_li.index(li);return vc['ve']*s.ve_layer_scales[vi].to(dtype=vc['ve'].dtype)
 def forward(s,ids,tgt):
  x=s.tok_emb(ids)
  if s.bigram:x=x+s.bigram(ids)
  x=F.rms_norm(x,(x.size(-1),));x=s.smear(x);x0=x;sk=[];vc={}
  for i in range(s.ne):ve=s._gve(i,ids,vc);x=s.blocks[i](x,x0,ve=ve);sk.append(x)
  for i in range(s.nd):
   bi=s.ne+i
   if sk:x=x+s.skip_weights[i].to(dtype=x.dtype)[None,None,:]*sk.pop()
   ve=s._gve(bi,ids,vc);x=s.blocks[bi](x,x0,ve=ve)
  x=s.final_norm(x);xf=x.reshape(-1,x.size(-1));tg=tgt.reshape(-1)
  lp=F.linear(xf,s.tok_emb.weight) if s.te else s.lm_head(xf)
  lg=s.lsc*torch.tanh(lp/s.lsc);return F.cross_entropy(lg.float(),tg,reduction="mean")
 def forward_logits(s,ids):
  x=s.tok_emb(ids)
  if s.bigram:x=x+s.bigram(ids)
  x=F.rms_norm(x,(x.size(-1),));x=s.smear(x);x0=x;sk=[];vc={}
  for i in range(s.ne):ve=s._gve(i,ids,vc);x=s.blocks[i](x,x0,ve=ve);sk.append(x)
  for i in range(s.nd):
   bi=s.ne+i
   if sk:x=x+s.skip_weights[i].to(dtype=x.dtype)[None,None,:]*sk.pop()
   ve=s._gve(bi,ids,vc);x=s.blocks[bi](x,x0,ve=ve)
  x=s.final_norm(x);lp=F.linear(x,s.tok_emb.weight) if s.te else s.lm_head(x)
  return s.lsc*torch.tanh(lp/s.lsc)
# ─── Dequantization ──────────────────────────────────────────────────────────
def dq6(res,meta,tsd):
 out={}
 for n,orig in tsd.items():
  info=meta.get(n)
  if info is None:continue
  od=orig.dtype
  if info in("passthrough","passthrough_ctrl","passthrough_fp16"):
   t=res[n]
   if t.dtype==torch.float16 and od in(torch.float32,torch.bfloat16):t=t.to(od)
   out[n]=t;continue
  q,s=res[n+".q"],res[n+".scale"]
  if s.ndim>0:out[n]=(q.float()*s.float().view(q.shape[0],*([1]*(q.ndim-1)))).to(od)
  else:out[n]=(q.float()*float(s.item())).to(od)
 return out
# ─── BPB helpers ──────────────────────────────────────────────────────────────
def build_sp_luts(sp,vs,dev):
 sv=int(sp.vocab_size());ts=max(sv,vs)
 bb=np.zeros(ts,dtype=np.int16);hs=np.zeros(ts,dtype=np.bool_);ib=np.ones(ts,dtype=np.bool_)
 for t in range(sv):
  if sp.is_control(t) or sp.is_unknown(t) or sp.is_unused(t):continue
  ib[t]=False
  if sp.is_byte(t):bb[t]=1;continue
  pc=sp.id_to_piece(t)
  if pc.startswith("\u2581"):hs[t]=True;pc=pc[1:]
  bb[t]=len(pc.encode("utf-8"))
 return(torch.tensor(bb,dtype=torch.int16,device=dev),torch.tensor(hs,dtype=torch.bool,device=dev),torch.tensor(ib,dtype=torch.bool,device=dev))
def load_shard(f):
 h=np.fromfile(f,dtype="<i4",count=256);nt=int(h[2])
 return torch.from_numpy(np.fromfile(f,dtype="<u2",count=nt,offset=256*4).astype(np.uint16,copy=False))
def load_val(pat,sl):
 ff=[Path(p) for p in sorted(glob.glob(pat))]
 if not ff:raise FileNotFoundError(pat)
 tk=torch.cat([load_shard(f) for f in ff]).contiguous();u=((tk.numel()-1)//sl)*sl
 return tk[:u+1]
# ─── TTT Sliding Window Eval ─────────────────────────────────────────────────
def eval_ttt_sliding(model,vt,seq_len,stride,ttt_epochs,ttt_lr,dev,bl,hl,il,rk,ws):
 """Score tokens with sliding window, TTT-update model AFTER scoring each window."""
 total=vt.numel()-1
 window_starts=[w for w in range(0,total,stride) if min(w+seq_len,total)-w>=1]
 tw=len(window_starts);ms=(tw*rk)//ws;me=(tw*(rk+1))//ws
 my_windows=window_starts[ms:me]
 ls=torch.zeros((),device=dev,dtype=torch.float64)
 tc=torch.zeros((),device=dev,dtype=torch.float64)
 bc=torch.zeros((),device=dev,dtype=torch.float64)
 # TTT optimizer — only update embeddings + small params for speed
 ttt_params=[p for p in model.parameters() if p.requires_grad]
 ttt_opt=torch.optim.Adam(ttt_params,lr=ttt_lr)
 scored=0
 for wi,ws_ in enumerate(my_windows):
  end=min(ws_+seq_len,total);wlen=end-ws_
  chunk=vt[ws_:end+1].to(dtype=torch.int64,device=dev)
  x=chunk[:-1].unsqueeze(0);y=chunk[1:].unsqueeze(0)
  # STEP 1: Score new tokens BEFORE any TTT update
  with torch.no_grad():
   with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
    logits=model.forward_logits(x)
   nll=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),y.reshape(-1),reduction="none")
  s=0 if ws_==0 else max(wlen-stride,0)
  scored_nll=nll[s:wlen].to(torch.float64)
  ls+=scored_nll.sum();tc+=float(wlen-s)
  tgt=y.reshape(-1);prev=x.reshape(-1)
  tb=bl[tgt[s:wlen]].to(torch.float64)
  tb+=(hl[tgt[s:wlen]]&~il[prev[s:wlen]]).to(torch.float64)
  bc+=tb.sum()
  scored+=wlen-s
  # STEP 2: TTT — train on this window (all tokens now graded)
  model.train()
  for ep in range(ttt_epochs):
   ttt_opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
    ttt_loss=model(x.squeeze(0) if x.dim()>2 else x,y.squeeze(0) if y.dim()>2 else y)
   ttt_loss.backward()
   torch.nn.utils.clip_grad_norm_(ttt_params,1.0)
   ttt_opt.step()
  model.eval()
  if wi%100==0 and rk==0:
   running_bpb=(ls/tc).item()/math.log(2.0)*(tc.item()/bc.item()) if tc.item()>0 else 0
   print(f"  ttt_progress: window {wi}/{len(my_windows)} scored:{scored} running_bpb:{running_bpb:.4f}")
 if dist.is_available() and dist.is_initialized():
  dist.all_reduce(ls,op=dist.ReduceOp.SUM);dist.all_reduce(tc,op=dist.ReduceOp.SUM);dist.all_reduce(bc,op=dist.ReduceOp.SUM)
 vl=(ls/tc).item();bpt=vl/math.log(2.0);tpb=tc.item()/bc.item()
 return vl,bpt*tpb
# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
 dd="RANK" in os.environ and "WORLD_SIZE" in os.environ
 rk=int(E("RANK","0"));ws=int(E("WORLD_SIZE","1"));lr_=int(E("LOCAL_RANK","0"))
 dev=torch.device("cuda",lr_);torch.cuda.set_device(dev)
 if dd:dist.init_process_group(backend="nccl",device_id=dev);dist.barrier()
 mp=rk==0
 # Config
 model_path=E("MODEL_PATH","final_model.int6.ptz")
 ttt_epochs=int(E("TTT_EPOCHS","8"))
 ttt_lr=float(E("TTT_LR","1e-4"))
 ttt_stride=int(E("TTT_STRIDE","64"))
 seq_len=int(E("EVAL_SEQ_LEN","2048"))
 vocab_size=int(E("VOCAB_SIZE","1024"))
 num_layers=int(E("NUM_LAYERS","11"))
 model_dim=int(E("MODEL_DIM","512"))
 num_heads=int(E("NUM_HEADS","8"))
 num_kv_heads=int(E("NUM_KV_HEADS","4"))
 mlp_mult=float(E("MLP_MULT","3.0"))
 rope_base=float(E("ROPE_BASE","10000.0"))
 logit_softcap=float(E("LOGIT_SOFTCAP","30.0"))
 qk_gain_init=float(E("QK_GAIN_INIT","1.5"))
 rope_dims=int(E("ROPE_DIMS","16"))
 ln_scale=bool(int(E("LN_SCALE","1")))
 xsa_last_n=int(E("XSA_LAST_N","4"))
 ve_enabled=bool(int(E("VE_ENABLED","1")))
 ve_dim=int(E("VE_DIM","128"))
 ve_layers=E("VE_LAYERS","9,10")
 bigram_vocab_size=int(E("BIGRAM_VOCAB_SIZE","2048"))
 bigram_dim=int(E("BIGRAM_DIM","128"))
 data_path=E("DATA_PATH","./data/datasets/fineweb10B_sp1024")
 tokenizer_path=E("TOKENIZER_PATH","./data/tokenizers/fineweb_1024_bpe.model")
 if mp:
  print(f"TTT Eval: epochs={ttt_epochs} lr={ttt_lr} stride={ttt_stride} seq_len={seq_len}")
  print(f"Model: {model_path}")
 # Load tokenizer + val data
 sp=spm.SentencePieceProcessor(model_file=tokenizer_path)
 bl,hl,il=build_sp_luts(sp,vocab_size,dev)
 vt=load_val(os.path.join(data_path,"fineweb_val_*.bin"),seq_len)
 if mp:print(f"Val tokens: {vt.numel()-1:,}")
 # Load + dequantize model
 with open(model_path,"rb") as f:blob=f.read()
 raw=zstandard.ZstdDecompressor().decompress(blob) if _Z=="zstd" else zlib.decompress(blob)
 qs=torch.load(io.BytesIO(raw),map_location="cpu")
 # Build template model for state dict keys
 template=GPT(vs=vocab_size,nl=num_layers,md=model_dim,nh=num_heads,nkv=num_kv_heads,mm=mlp_mult,te=True,teis=0.005,lsc=logit_softcap,rb=rope_base,qkg=qk_gain_init,bvs=bigram_vocab_size,bd=bigram_dim,xln=xsa_last_n,rd=rope_dims,lns=ln_scale,ve_on=ve_enabled,ve_d=ve_dim,ve_l=ve_layers)
 tsd={k:v for k,v in template.state_dict().items()}
 dqs=dq6(qs["w"],qs["m"],tsd)
 model=GPT(vs=vocab_size,nl=num_layers,md=model_dim,nh=num_heads,nkv=num_kv_heads,mm=mlp_mult,te=True,teis=0.005,lsc=logit_softcap,rb=rope_base,qkg=qk_gain_init,bvs=bigram_vocab_size,bd=bigram_dim,xln=xsa_last_n,rd=rope_dims,lns=ln_scale,ve_on=ve_enabled,ve_d=ve_dim,ve_l=ve_layers).to(dev).bfloat16()
 for m in model.modules():
  if isinstance(m,CastedLinear):m.float()
 restore_fp32(model)
 model.load_state_dict(dqs,strict=True)
 if mp:print("Model loaded and dequantized")
 # Run TTT eval
 torch.cuda.synchronize();t0=time.perf_counter()
 model.eval()
 vl,vbpb=eval_ttt_sliding(model,vt,seq_len,ttt_stride,ttt_epochs,ttt_lr,dev,bl,hl,il,rk,ws)
 torch.cuda.synchronize();elapsed=time.perf_counter()-t0
 if mp:
  print(f"\nTTT Results:")
  print(f"  val_loss: {vl:.8f}")
  print(f"  val_bpb:  {vbpb:.8f}")
  print(f"  ttt_epochs: {ttt_epochs}")
  print(f"  ttt_lr: {ttt_lr}")
  print(f"  stride: {ttt_stride}")
  print(f"  eval_time: {elapsed:.1f}s")
 if dd:dist.destroy_process_group()
if __name__=="__main__":main()
