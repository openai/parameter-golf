from __future__ import annotations
import copy,glob,io,math,os,random,sys,time,uuid,zlib
from pathlib import Path
try:
 import zstandard;_Z="zstd"
except ImportError:_Z="zlib"
import numpy as np;import sentencepiece as spm;import torch
import torch.distributed as dist;import torch.nn.functional as F
from torch import Tensor,nn;from torch.nn.parallel import DistributedDataParallel as DDP
from flash_attn_interface import flash_attn_func as fa3
E=os.environ.get
class H:
 data_path=E("DATA_PATH","./data/datasets/fineweb10B_sp1024")
 train_files=os.path.join(data_path,"fineweb_train_*.bin")
 val_files=os.path.join(data_path,"fineweb_val_*.bin")
 tokenizer_path=E("TOKENIZER_PATH","./data/tokenizers/fineweb_1024_bpe.model")
 run_id=E("RUN_ID",str(uuid.uuid4()));seed=int(E("SEED","1337"))
 val_batch_size=int(E("VAL_BATCH_SIZE","524288"));val_loss_every=int(E("VAL_LOSS_EVERY","4000"))
 train_log_every=int(E("TRAIN_LOG_EVERY","500"));iterations=int(E("ITERATIONS","20000"))
 warmdown_iters=int(E("WARMDOWN_ITERS","3000"));warmup_steps=int(E("WARMUP_STEPS","20"))
 train_batch_tokens=int(E("TRAIN_BATCH_TOKENS","786432"));train_seq_len=int(E("TRAIN_SEQ_LEN","2048"))
 eval_seq_len=int(E("EVAL_SEQ_LEN","2048"));max_wallclock_seconds=float(E("MAX_WALLCLOCK_SECONDS","600.0"))
 qk_gain_init=float(E("QK_GAIN_INIT","1.5"));vocab_size=int(E("VOCAB_SIZE","1024"))
 num_unique_layers=int(E("NUM_UNIQUE_LAYERS","4"));num_loops=int(E("NUM_LOOPS","3"))
 num_kv_heads=int(E("NUM_KV_HEADS","4"))
 model_dim=int(E("MODEL_DIM","768"));num_heads=int(E("NUM_HEADS","8"))
 fractal_cadence=int(E("FRACTAL_CADENCE","2"));fractal_offset=int(E("FRACTAL_OFFSET","0"))
 mlp_mult=float(E("MLP_MULT","3.0"));tie_embeddings=bool(int(E("TIE_EMBEDDINGS","1")))
 rope_base=float(E("ROPE_BASE","10000.0"));logit_softcap=float(E("LOGIT_SOFTCAP","30.0"))
 embed_lr=float(E("EMBED_LR","0.6"));head_lr=float(E("HEAD_LR","0.008"))
 tied_embed_lr=float(E("TIED_EMBED_LR","0.035"));tied_embed_init_std=float(E("TIED_EMBED_INIT_STD","0.005"))
 matrix_lr=float(E("MATRIX_LR","0.025"));scalar_lr=float(E("SCALAR_LR","0.025"))
 muon_momentum=float(E("MUON_MOMENTUM","0.99"));muon_backend_steps=int(E("MUON_BACKEND_STEPS","5"))
 muon_momentum_warmup_start=float(E("MUON_MOMENTUM_WARMUP_START","0.92"))
 muon_momentum_warmup_steps=int(E("MUON_MOMENTUM_WARMUP_STEPS","1500"))
 beta1=float(E("BETA1","0.9"));beta2=float(E("BETA2","0.95"));adam_eps=float(E("ADAM_EPS","1e-8"))
 grad_clip_norm=float(E("GRAD_CLIP_NORM","0.3"));eval_stride=int(E("EVAL_STRIDE","64"))
 muon_wd=float(E("MUON_WD","0.04"));adam_wd=float(E("ADAM_WD","0.04"))
 swa_enabled=bool(int(E("SWA_ENABLED","1")));swa_every=int(E("SWA_EVERY","50"))
 bigram_vocab_size=int(E("BIGRAM_VOCAB_SIZE","2048"));bigram_dim=int(E("BIGRAM_DIM","128"))
 xsa_last_n=int(E("XSA_LAST_N","2"));rope_dims=int(E("ROPE_DIMS","16"))
 ln_scale=bool(int(E("LN_SCALE","1")));late_qat_threshold=float(E("LATE_QAT_THRESHOLD","0.15"))
 ve_enabled=bool(int(E("VE_ENABLED","1")));ve_dim=int(E("VE_DIM","128"))
 ve_layers=E("VE_LAYERS","2,3");ema_decay=float(E("EMA_DECAY","0.997"))
 ema_enabled=bool(int(E("EMA_ENABLED","1")))
_CP=tuple(p for p in E("CONTROL_TENSOR_NAME_PATTERNS","attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,ve_layer_scales,ve_shared.scale").split(",") if p)
def _ns5(G,steps=10,eps=1e-7):
 a,b,c=3.4445,-4.7750,2.0315;X=G.bfloat16();X/=X.norm()+eps
 tr=G.size(0)>G.size(1)
 if tr:X=X.T
 for _ in range(steps):A=X@X.T;B=b*A+c*A@A;X=a*X+B@X
 return X.T if tr else X
class Muon(torch.optim.Optimizer):
 def __init__(s,params,lr,momentum,backend_steps,nesterov=True,weight_decay=0.0):
  super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,nesterov=nesterov,weight_decay=weight_decay))
 @torch.no_grad()
 def step(s,closure=None):
  lo=None
  if closure is not None:
   with torch.enable_grad():lo=closure()
  dd=dist.is_available() and dist.is_initialized();ws=dist.get_world_size() if dd else 1;rk=dist.get_rank() if dd else 0
  for g in s.param_groups:
   pp=g["params"]
   if not pp:continue
   lr,mom,bs,nest=g["lr"],g["momentum"],g["backend_steps"],g["nesterov"]
   tp=sum(int(p.numel()) for p in pp);uf=torch.zeros(tp,device=pp[0].device,dtype=torch.bfloat16);cur=0
   for i,p in enumerate(pp):
    if i%ws==rk and p.grad is not None:
     gr=p.grad;st=s.state[p]
     if "mb" not in st:st["mb"]=torch.zeros_like(gr)
     buf=st["mb"];buf.mul_(mom).add_(gr)
     if nest:gr=gr.add(buf,alpha=mom)
     gr=_ns5(gr,steps=bs);gr*=max(1,gr.size(0)/gr.size(1))**0.5
     uf[cur:cur+p.numel()]=gr.reshape(-1)
    cur+=p.numel()
   if dd:dist.all_reduce(uf,op=dist.ReduceOp.SUM)
   wd=g.get("weight_decay",0.0);cur=0
   for p in pp:
    if wd>0:p.data.mul_(1.0-lr*wd)
    p.add_(uf[cur:cur+p.numel()].view_as(p).to(dtype=p.dtype),alpha=-lr);cur+=p.numel()
  return lo
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
def load_val(pat,sl):
 ff=[Path(p) for p in sorted(glob.glob(pat))]
 if not ff:raise FileNotFoundError(pat)
 tk=torch.cat([load_shard(f) for f in ff]).contiguous();u=((tk.numel()-1)//sl)*sl
 return tk[:u+1]
def eval_val(a,model,rk,ws,dev,ga,vt,bl,hl,il,esl=None):
 sl=esl or a.train_seq_len;lb=a.val_batch_size//max(ws*ga,1)
 if lb<sl:lb=sl
 lbs=lb//sl;ts=(vt.numel()-1)//sl;ss=(ts*rk)//ws;se=(ts*(rk+1))//ws
 vls=torch.zeros((),device=dev,dtype=torch.float64);vtc=torch.zeros((),device=dev,dtype=torch.float64);vbc=torch.zeros((),device=dev,dtype=torch.float64)
 model.eval()
 with torch.inference_mode():
  for bs in range(ss,se,lbs):
   be=min(bs+lbs,se);rs=bs*sl;re=be*sl+1
   lc=vt[rs:re].to(device=dev,dtype=torch.int64,non_blocking=True)
   x=lc[:-1].reshape(-1,sl);y=lc[1:].reshape(-1,sl)
   with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):bl_=model(x,y).detach()
   n=float(y.numel());vls+=bl_.to(torch.float64)*n;vtc+=n
   pi=x.reshape(-1);ti=y.reshape(-1);tb=bl[ti].to(dtype=torch.int16)
   tb+=(hl[ti]&~il[pi]).to(dtype=torch.int16);vbc+=tb.to(torch.float64).sum()
 if dist.is_available() and dist.is_initialized():
  dist.all_reduce(vls,op=dist.ReduceOp.SUM);dist.all_reduce(vtc,op=dist.ReduceOp.SUM);dist.all_reduce(vbc,op=dist.ReduceOp.SUM)
 vl=vls/vtc;bpt=vl.item()/math.log(2.0);tpb=vtc.item()/vbc.item();model.train()
 return float(vl.item()),float(bpt*tpb)
def load_shard(f):
 h=np.fromfile(f,dtype="<i4",count=256);nt=int(h[2])
 return torch.from_numpy(np.fromfile(f,dtype="<u2",count=nt,offset=256*4).astype(np.uint16,copy=False))
class TokenStream:
 def __init__(s,pat):
  s.files=[Path(p) for p in sorted(glob.glob(pat))]
  if not s.files:raise FileNotFoundError(pat)
  s.fi=0;s.tokens=load_shard(s.files[0]);s.pos=0
 def _adv(s):s.fi=(s.fi+1)%len(s.files);s.tokens=load_shard(s.files[s.fi]);s.pos=0
 def take(s,n):
  ch=[];r=n
  while r>0:
   av=s.tokens.numel()-s.pos
   if av<=0:s._adv();continue
   k=min(r,av);ch.append(s.tokens[s.pos:s.pos+k]);s.pos+=k;r-=k
  return ch[0] if len(ch)==1 else torch.cat(ch)
class DTL:
 def __init__(s,pat,rk,ws,dev):s.rk=rk;s.ws=ws;s.dev=dev;s.stream=TokenStream(pat)
 def next_batch(s,gt,sl,ga):
  lt=gt//(s.ws*ga);ps=lt+1;ck=s.stream.take(ps*s.ws);st=s.rk*ps
  lc=ck[st:st+ps].to(dtype=torch.int64);x=lc[:-1].reshape(-1,sl);y=lc[1:].reshape(-1,sl)
  return x.to(s.dev,non_blocking=True),y.to(s.dev,non_blocking=True)
class RMSNorm(nn.Module):
 def __init__(s,eps=None):super().__init__();s.eps=eps
 def forward(s,x):return F.rms_norm(x,(x.size(-1),),eps=s.eps)
class CastedLinear(nn.Linear):
 _qat=False
 def forward(s,x):
  w=s.weight.to(x.dtype)
  if CastedLinear._qat and s.training and w.ndim==2:
   with torch.no_grad():
    w32=s.weight.float();rm=w32.abs().amax(dim=1);sc=(rm/31.0).clamp_min(1.0/31.0)
    wq=(torch.clamp(torch.round(w32/sc[:,None]),-32,31)*sc[:,None]).to(x.dtype)
   w=w+(wq-w).detach()
  b=s.bias.to(x.dtype) if s.bias is not None else None
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
  q=q*s.q_gain.to(dtype=q.dtype)[None,None,:,None];y=fa3(q,k,v,causal=True)
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
  h=s.embed(s._hash(ids))
  if s.proj:h=s.proj(h)
  return h*s.scale.to(dtype=h.dtype)
class VE(nn.Module):
 def __init__(s,vs,vd,md):
  super().__init__();s.embed=nn.Embedding(vs,vd);nn.init.normal_(s.embed.weight,std=0.01)
  s.proj=CastedLinear(vd,md,bias=False) if vd!=md else None
  if s.proj:nn.init.zeros_(s.proj.weight)
  s.scale=nn.Parameter(torch.tensor(0.1,dtype=torch.float32))
 def forward(s,ids):
  h=s.embed(ids)
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
 def __init__(s,vs,nul,nlp,md,nh,nkv,mm,te,teis,lsc,rb,qkg,bvs=0,bd=128,xln=0,rd=0,lns=False,ve_on=False,ve_d=128,ve_l="2,3"):
  super().__init__();s._vetd=nkv*(md//nh);s.te=te;s.teis=teis;s.lsc=lsc;s.num_loops=nlp
  s.tok_emb=nn.Embedding(vs,md)
  s.bigram=BigramHash(bvs,bd,md) if bvs>0 else None;s.smear=SmearGate(md)
  s.ne=nul//2;s.nd=nul-s.ne;s.ns=min(s.ne,s.nd)
  s.skip_weights=nn.Parameter(torch.ones(s.ns,md,dtype=torch.float32))
  s.blocks=nn.ModuleList([Block(md,nh,nkv,mm,rb,qkg,li=i,lns=lns) for i in range(nul)])
  if rd>0:
   hd=md//nh
   for b in s.blocks:b.attn.rope_dims=rd;b.attn.rotary=Rotary(hd,base=rb,tsl=1024,rd=rd)
  raw=torch.randn(nlp+1,md);Q,_=torch.linalg.qr(raw.T)
  s.loop_pos=nn.Parameter(Q.T[:nlp+1]*0.01)
  s.ve_li=[int(x) for x in ve_l.split(",") if x.strip()] if ve_on else [];kd=s._vetd
  if s.ve_li:s.ve_shared=VE(vs,ve_d,kd);s.ve_layer_scales=nn.ParameterList([nn.Parameter(torch.ones(1,dtype=torch.float32)) for _ in s.ve_li])
  else:s.ve_shared=None;s.ve_layer_scales=nn.ParameterList()
  s.value_embeds=nn.ModuleList();s.final_norm=RMSNorm()
  s.lm_head=None if te else CastedLinear(md,vs,bias=False)
  if s.lm_head:s.lm_head._zero_init=True
  s.mtp_heads=nn.ModuleList();s.mtp_num_heads=0;s.mtp_loss_weight=0
  if xln>0:
   for i in range(max(0,nul-xln),nul):s.blocks[i].attn.use_xsa=True
  s._iw()
 def _iw(s):
  if s.te:nn.init.normal_(s.tok_emb.weight,mean=0.0,std=s.teis)
  ed=len(s.blocks)*s.num_loops
  for n,m in s.named_modules():
   if isinstance(m,nn.Linear):
    if getattr(m,"_zero_init",False):nn.init.zeros_(m.weight)
    elif m.weight.ndim==2 and m.weight.shape[0]>=64 and m.weight.shape[1]>=64:
     nn.init.orthogonal_(m.weight,gain=1.0)
     if ".proj." in n or n.endswith(".proj"):
      with torch.no_grad():m.weight.mul_(1.0/math.sqrt(2*ed))
 def _gve(s,li,ids,vc):
  if s.ve_shared is None or li not in s.ve_li:return None
  if 've' not in vc:vc['ve']=s.ve_shared(ids)
  vi=s.ve_li.index(li);return vc['ve']*s.ve_layer_scales[vi].to(dtype=vc['ve'].dtype)
 def _run_blocks(s,x,x0,ids,vc):
  sk=[]
  for i in range(s.ne):ve=s._gve(i,ids,vc);x=s.blocks[i](x,x0,ve=ve);sk.append(x)
  for i in range(s.nd):
   bi=s.ne+i
   if sk:x=x+s.skip_weights[i].to(dtype=x.dtype)[None,None,:]*sk.pop()
   ve=s._gve(bi,ids,vc);x=s.blocks[bi](x,x0,ve=ve)
  return x
 def forward(s,ids,tgt,fractal=True):
  x=s.tok_emb(ids)
  if s.bigram:x=x+s.bigram(ids)
  x=F.rms_norm(x,(x.size(-1),));x=s.smear(x);x0=x;vc={}
  if fractal:
   for lp in range(s.num_loops):
    x=x+s.loop_pos[lp][None,None,:];x=s._run_blocks(x,x0,ids,vc)
  else:
   x=x+s.loop_pos[s.num_loops][None,None,:];x=s._run_blocks(x,x0,ids,vc)
  x=s.final_norm(x);xf=x.reshape(-1,x.size(-1));tg=tgt.reshape(-1)
  lp=F.linear(xf,s.tok_emb.weight) if s.te else s.lm_head(xf)
  lg=s.lsc*torch.tanh(lp/s.lsc);return F.cross_entropy(lg.float(),tg,reduction="mean")
 def forward_logits(s,ids):
  x=s.tok_emb(ids)
  if s.bigram:x=x+s.bigram(ids)
  x=F.rms_norm(x,(x.size(-1),));x=s.smear(x);x0=x;vc={}
  for lp in range(s.num_loops):
   x=x+s.loop_pos[lp][None,None,:];x=s._run_blocks(x,x0,ids,vc)
  x=s.final_norm(x);lp=F.linear(x,s.tok_emb.weight) if s.te else s.lm_head(x)
  return s.lsc*torch.tanh(lp/s.lsc)
def eval_slide(a,bm,rk,ws,dev,vt,bl,hl,il,stride,bseqs=32,esl=None):
 sl=esl or a.train_seq_len;tt=vt.numel()-1
 ww=[w for w in range(0,tt,stride) if min(w+sl,tt)-w>=1];tw=len(ww)
 ms=(tw*rk)//ws;me=(tw*(rk+1))//ws;mw=ww[ms:me]
 ls=torch.zeros((),device=dev,dtype=torch.float64);tc=torch.zeros((),device=dev,dtype=torch.float64);bc=torch.zeros((),device=dev,dtype=torch.float64)
 bm.eval();cl=torch.compile(bm.forward_logits,dynamic=False,fullgraph=True)
 with torch.inference_mode():
  for bi in range(0,len(mw),bseqs):
   bw=mw[bi:bi+bseqs];bs=len(bw)
   xb=torch.zeros(bs,sl,dtype=torch.int64,device=dev);yb=torch.zeros(bs,sl,dtype=torch.int64,device=dev);wl=[]
   for i,w in enumerate(bw):
    e=min(w+sl,tt);wn=e-w;wl.append(wn);ck=vt[w:e+1].to(dtype=torch.int64,device=dev)
    xb[i,:wn]=ck[:-1];yb[i,:wn]=ck[1:]
   with torch.autocast(device_type="cuda",dtype=torch.bfloat16):lg=cl(xb)
   nl=F.cross_entropy(lg.reshape(-1,lg.size(-1)).float(),yb.reshape(-1),reduction="none").reshape(bs,sl)
   for i,w in enumerate(bw):
    wn=wl[i];st=0 if w==0 else max(wn-stride,0);sn=nl[i,st:wn].to(torch.float64)
    ls+=sn.sum();tc+=float(wn-st);tg=yb[i,st:wn];pv=xb[i,st:wn]
    tb=bl[tg].to(torch.float64);tb+=(hl[tg]&~il[pv]).to(torch.float64);bc+=tb.sum()
 if dist.is_available() and dist.is_initialized():
  dist.all_reduce(ls,op=dist.ReduceOp.SUM);dist.all_reduce(tc,op=dist.ReduceOp.SUM);dist.all_reduce(bc,op=dist.ReduceOp.SUM)
 vl=(ls/tc).item();bpt=vl/math.log(2.0);tpb=tc.item()/bc.item();bm.train()
 return vl,bpt*tpb
def _clp(n):
 if "tok_emb" in n or "lm_head" in n:return "embed"
 if ".mlp." in n:return "mlp"
 if ".attn." in n or(".proj." in n and ".mlp." not in n):return "attn"
 return "other"
def q6r(t):
 t32=t.float()
 if t32.ndim==2:rm=t32.abs().amax(dim=1);sc=(rm/31.0).clamp_min(1.0/31.0).to(torch.float16);return torch.clamp(torch.round(t32/sc.float()[:,None]),-32,31).to(torch.int8),sc
 am=t32.abs().max().item();sc=torch.tensor(am/31.0 if am>0 else 1.0,dtype=torch.float16)
 return torch.clamp(torch.round(t32/sc.float()),-32,31).to(torch.int8),sc
def qf(t):
 t32=t.float()
 if t32.ndim==2:
  ca=torch.quantile(t32.abs(),0.9999984,dim=1) if t32.numel() else torch.empty((t32.shape[0],),dtype=torch.float32)
  cl=torch.maximum(torch.minimum(t32,ca[:,None]),-ca[:,None]);sc=(ca/127.0).clamp_min(1.0/127.0)
  return torch.clamp(torch.round(cl/sc[:,None]),-127,127).to(torch.int8).contiguous(),sc.to(dtype=torch.float16).contiguous()
 ca=float(torch.quantile(t32.abs().flatten(),0.9999984).item()) if t32.numel() else 0.0
 sc=torch.tensor(ca/127.0 if ca>0 else 1.0,dtype=torch.float32)
 return torch.clamp(torch.round(torch.clamp(t32,-ca,ca)/sc),-127,127).to(torch.int8).contiguous(),sc
def mq6(sd,cats):
 res={};meta={}
 for n,t in sd.items():
  t=t.detach().cpu().contiguous();cat=_clp(n)
  if not t.is_floating_point() or t.numel()<=65536:res[n]=t.to(torch.float16) if t.is_floating_point() else t;meta[n]="passthrough";continue
  if any(p in n for p in _CP):res[n]=t.float();meta[n]="passthrough_ctrl";continue
  if cat in cats and t.ndim>=1:q,s=q6r(t);res[n+".q"]=q;res[n+".scale"]=s;meta[n]={"type":"int6"}
  else:q,s=qf(t);res[n+".q"]=q;res[n+".scale"]=s;meta[n]={"type":"int8"}
 return res,meta
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
def main():
 global _ns5;code=Path(__file__).read_text(encoding="utf-8");a=H();_ns5=torch.compile(_ns5)
 dd="RANK" in os.environ and "WORLD_SIZE" in os.environ
 rk=int(E("RANK","0"));ws=int(E("WORLD_SIZE","1"));lr_=int(E("LOCAL_RANK","0"))
 ga=8//ws;gs=1.0/ga;dev=torch.device("cuda",lr_);torch.cuda.set_device(dev)
 if dd:dist.init_process_group(backend="nccl",device_id=dev);dist.barrier()
 mp=rk==0;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 from torch.backends.cuda import enable_cudnn_sdp,enable_flash_sdp,enable_math_sdp,enable_mem_efficient_sdp
 enable_cudnn_sdp(False);enable_flash_sdp(True);enable_mem_efficient_sdp(False);enable_math_sdp(False)
 lf=None
 if mp:os.makedirs("logs",exist_ok=True);lf=f"logs/{a.run_id}.txt";print(lf)
 def log0(m,c=True):
  if not mp:return
  if c:print(m)
  if lf:
   with open(lf,"a",encoding="utf-8") as f:print(m,file=f)
 log0(code,False);log0("="*100,False)
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 sp=spm.SentencePieceProcessor(model_file=a.tokenizer_path)
 esl=a.eval_seq_len if a.eval_seq_len>0 else a.train_seq_len;vsl=max(a.train_seq_len,esl)
 vt=load_val(a.val_files,vsl);bl,hl,il=build_sp_luts(sp,a.vocab_size,dev)
 log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={a.tokenizer_path}")
 CastedLinear._qat=False
 bm=GPT(vs=a.vocab_size,nul=a.num_unique_layers,nlp=a.num_loops,md=a.model_dim,nh=a.num_heads,nkv=a.num_kv_heads,mm=a.mlp_mult,te=a.tie_embeddings,teis=a.tied_embed_init_std,lsc=a.logit_softcap,rb=a.rope_base,qkg=a.qk_gain_init,bvs=a.bigram_vocab_size,bd=a.bigram_dim,xln=a.xsa_last_n,rd=a.rope_dims,lns=a.ln_scale,ve_on=a.ve_enabled,ve_d=a.ve_dim,ve_l=a.ve_layers).to(dev).bfloat16()
 for m in bm.modules():
  if isinstance(m,CastedLinear):m.float()
 restore_fp32(bm);cm=torch.compile(bm,dynamic=False)
 model=DDP(cm,device_ids=[lr_],broadcast_buffers=False) if dd else cm
 bnp=list(bm.blocks.named_parameters())
 mxp=[p for n,p in bnp if p.ndim==2 and not any(pt in n for pt in _CP)]
 scp=[p for n,p in bnp if p.ndim<2 or any(pt in n for pt in _CP)]
 if bm.skip_weights.numel()>0:scp.append(bm.skip_weights)
 scp.append(bm.smear.gate);scp.append(bm.loop_pos)
 if bm.bigram:scp.append(bm.bigram.scale)
 tlr=a.tied_embed_lr if a.tie_embeddings else a.embed_lr
 tkp=[{"params":[bm.tok_emb.weight],"lr":tlr,"base_lr":tlr}]
 if bm.bigram:
  tkp.append({"params":[bm.bigram.embed.weight],"lr":tlr,"base_lr":tlr})
  if bm.bigram.proj:mxp.append(bm.bigram.proj.weight)
 if bm.ve_shared:
  tkp.append({"params":[bm.ve_shared.embed.weight],"lr":tlr,"base_lr":tlr})
  if bm.ve_shared.proj:mxp.append(bm.ve_shared.proj.weight)
  scp.append(bm.ve_shared.scale)
  for s in bm.ve_layer_scales:scp.append(s)
 otk=torch.optim.AdamW(tkp,betas=(a.beta1,a.beta2),eps=a.adam_eps,weight_decay=a.adam_wd,fused=True)
 omu=Muon(mxp,lr=a.matrix_lr,momentum=a.muon_momentum,backend_steps=a.muon_backend_steps,weight_decay=a.muon_wd)
 for g in omu.param_groups:g["base_lr"]=a.matrix_lr
 osc=torch.optim.AdamW([{"params":scp,"lr":a.scalar_lr,"base_lr":a.scalar_lr}],betas=(a.beta1,a.beta2),eps=a.adam_eps,weight_decay=a.adam_wd,fused=True)
 opts=[otk,omu,osc]
 if bm.lm_head:
  oh=torch.optim.Adam([{"params":[bm.lm_head.weight],"lr":a.head_lr,"base_lr":a.head_lr}],betas=(a.beta1,a.beta2),eps=a.adam_eps,fused=True)
  opts.insert(1,oh)
 np_=sum(p.numel() for p in bm.parameters());log0(f"model_params:{np_}")
 log0(f"fractal:unique_layers={a.num_unique_layers} loops={a.num_loops} eff_depth={a.num_unique_layers*a.num_loops} cadence={a.fractal_cadence} offset={a.fractal_offset}")
 xl=[i for i,b in enumerate(bm.blocks) if b.attn.use_xsa];log0(f"XSA:last_{a.xsa_last_n} active_layers:{xl}")
 log0(f"world_size:{ws} grad_accum_steps:{ga}")
 log0(f"tie_embeddings:{a.tie_embeddings} embed_lr:{tlr} matrix_lr:{a.matrix_lr} scalar_lr:{a.scalar_lr}")
 log0(f"train_batch_tokens:{a.train_batch_tokens} train_seq_len:{a.train_seq_len} iterations:{a.iterations} warmup_steps:{a.warmup_steps} max_wallclock_seconds:{a.max_wallclock_seconds:.3f}")
 log0(f"seed:{a.seed}")
 tl=DTL(a.train_files,rk,ws,dev)
 def zg():
  for o in opts:o.zero_grad(set_to_none=True)
 mwm=1000.0*a.max_wallclock_seconds if a.max_wallclock_seconds>0 else None
 def lrm(step,ems):
  if a.warmdown_iters<=0:return 1.0
  if mwm is None:
   wd=max(a.iterations-a.warmdown_iters,0)
   return max((a.iterations-step)/max(a.warmdown_iters,1),0.0) if wd<=step<a.iterations else 1.0
  sms=ems/max(step,1);wms=a.warmdown_iters*sms;rms=max(mwm-ems,0.0)
  return rms/max(wms,1e-9) if rms<=wms else 1.0
 if a.warmup_steps>0:
  ims={n:t.detach().cpu().clone() for n,t in bm.state_dict().items()}
  ios=[copy.deepcopy(o.state_dict()) for o in opts];model.train()
  for ws_ in range(a.warmup_steps):
   zg()
   for ms_ in range(ga):
    if dd:model.require_backward_grad_sync=ms_==ga-1
    x,y=tl.next_batch(a.train_batch_tokens,a.train_seq_len,ga)
    with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):wl=model(x,y)
    (wl*gs).backward()
   for o in opts:o.step()
   zg()
   if a.warmup_steps<=20 or(ws_+1)%10==0 or ws_+1==a.warmup_steps:log0(f"warmup_step:{ws_+1}/{a.warmup_steps}")
  bm.load_state_dict(ims,strict=True)
  for o,s in zip(opts,ios,strict=True):o.load_state_dict(s)
  zg()
  if dd:model.require_backward_grad_sync=True
  tl=DTL(a.train_files,rk,ws,dev)
 ema_st=None
 if a.ema_enabled:log0(f"ema:enabled decay={a.ema_decay}")
 swa_st=None;swa_c=0;ttms=0.0;sas=None
 torch.cuda.synchronize();t0=time.perf_counter();step=0
 while True:
  ls=step==a.iterations or(sas is not None and step>=sas)
  sv=ls or(a.val_loss_every>0 and step%a.val_loss_every==0)
  if sv:
   torch.cuda.synchronize();ttms+=1000.0*(time.perf_counter()-t0)
   vl,vb=eval_val(a,model,rk,ws,dev,ga,vt,bl,hl,il)
   log0(f"step:{step}/{a.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} train_time:{ttms:.0f}ms step_avg:{ttms/max(step,1):.2f}ms")
   torch.cuda.synchronize();t0=time.perf_counter()
  if ls:
   if sas is not None and step<a.iterations:log0(f"stopping_early: wallclock_cap train_time:{ttms:.0f}ms step:{step}/{a.iterations}")
   break
  ems=ttms+1000.0*(time.perf_counter()-t0);sc=lrm(step,ems)
  if a.late_qat_threshold>0 and sc<a.late_qat_threshold and not CastedLinear._qat:
   CastedLinear._qat=True;log0(f"late_qat:enabled step:{step} scale:{sc:.4f}")
  is_f=True
  if a.fractal_cadence==0:is_f=False
  elif a.fractal_cadence>1:is_f=(step%a.fractal_cadence)==a.fractal_offset
  zg();trl=torch.zeros((),device=dev)
  for ms_ in range(ga):
   if dd:model.require_backward_grad_sync=ms_==ga-1
   x,y=tl.next_batch(a.train_batch_tokens,a.train_seq_len,ga)
   with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):lo=model(x,y,fractal=is_f)
   trl+=lo.detach();(lo*gs).backward()
  trl/=ga
  fr=min(step/a.muon_momentum_warmup_steps,1.0) if a.muon_momentum_warmup_steps>0 else 1.0
  mm_=(1-fr)*a.muon_momentum_warmup_start+fr*a.muon_momentum
  for g in omu.param_groups:g["momentum"]=mm_
  for o in opts:
   for g in o.param_groups:g["lr"]=g["base_lr"]*sc
  if a.grad_clip_norm>0:torch.nn.utils.clip_grad_norm_(bm.parameters(),a.grad_clip_norm)
  for o in opts:o.step()
  zg();step+=1;ams=ttms+1000.0*(time.perf_counter()-t0)
  if a.ema_enabled:
   if ema_st is None:ema_st={n:t.detach().cpu().clone() for n,t in bm.state_dict().items()}
   else:
    for n,t in bm.state_dict().items():ema_st[n].mul_(a.ema_decay).add_(t.detach().cpu(),alpha=1-a.ema_decay)
  if a.swa_enabled and sc<0.2 and step%a.swa_every==0:
   src=ema_st if a.ema_enabled and ema_st is not None else{n:t.detach().cpu().clone() for n,t in bm.state_dict().items()}
   if swa_st is None:swa_st={n:t.clone() for n,t in src.items()};swa_c=1;log0(f"swa:start step:{step} source={'ema' if a.ema_enabled else 'raw'}")
   else:
    for n,t in src.items():swa_st[n]+=t
    swa_c+=1
  sl_=a.train_log_every>0 and(step<=10 or step%a.train_log_every==0 or sas is not None)
  if sl_:log0(f"step:{step}/{a.iterations} train_loss:{trl.item():.4f} train_time:{ams:.0f}ms step_avg:{ams/step:.2f}ms")
  rc=mwm is not None and ams>=mwm
  if dd and mwm is not None:
   rt=torch.tensor(int(rc),device=dev);dist.all_reduce(rt,op=dist.ReduceOp.MAX);rc=bool(rt.item())
  if sas is None and rc:sas=step
 log0(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB")
 if a.swa_enabled and swa_st is not None and swa_c>1:
  log0(f"swa:applying averaged {swa_c} checkpoints (from {'ema' if a.ema_enabled else 'raw'})")
  avg={n:(t/swa_c).to(dtype=bm.state_dict()[n].dtype) for n,t in swa_st.items()}
  bm.load_state_dict(avg,strict=True)
  torch.cuda.synchronize();td=time.perf_counter()
  dvl,dvb=eval_val(a,cm,rk,ws,dev,ga,vt,bl,hl,il)
  torch.cuda.synchronize();log0(f"DIAGNOSTIC post_avg val_loss:{dvl:.4f} val_bpb:{dvb:.4f} eval_time:{1000.0*(time.perf_counter()-td):.0f}ms")
 fsd=bm.state_dict();esd={k:v for k,v in fsd.items() if "mtp_heads" not in k}
 if mp:torch.save(esd,"final_model.pt");log0(f"Serialized model: {os.path.getsize('final_model.pt')} bytes")
 sdc={k:v.detach().cpu() for k,v in esd.items()};cb=len(code.encode("utf-8"))
 log0(f"Code size: {cb} bytes")
 qr,qm=mq6(sdc,{"mlp","attn"});qb=io.BytesIO();torch.save({"w":qr,"m":qm},qb);qraw=qb.getvalue()
 qblob=zstandard.ZstdCompressor(level=22).compress(qraw) if _Z=="zstd" else zlib.compress(qraw,9)
 if mp:
  with open("final_model.int6.ptz","wb") as f:f.write(qblob)
  qfb=len(qblob);log0(f"Serialized model int6+{_Z}: {qfb} bytes");log0(f"Total submission size int6+{_Z}: {qfb+cb} bytes")
 if dd:dist.barrier()
 with open("final_model.int6.ptz","rb") as f:qbd=f.read()
 qs=torch.load(io.BytesIO(zstandard.ZstdDecompressor().decompress(qbd) if _Z=="zstd" else zlib.decompress(qbd)),map_location="cpu")
 dqs=dq6(qs["w"],qs["m"],sdc)
 em=GPT(vs=a.vocab_size,nul=a.num_unique_layers,nlp=a.num_loops,md=a.model_dim,nh=a.num_heads,nkv=a.num_kv_heads,mm=a.mlp_mult,te=a.tie_embeddings,teis=a.tied_embed_init_std,lsc=a.logit_softcap,rb=a.rope_base,qkg=a.qk_gain_init,bvs=a.bigram_vocab_size,bd=a.bigram_dim,xln=a.xsa_last_n,rd=a.rope_dims,lns=a.ln_scale,ve_on=a.ve_enabled,ve_d=a.ve_dim,ve_l=a.ve_layers).to(dev).bfloat16()
 for m in em.modules():
  if isinstance(m,CastedLinear):m.float()
 restore_fp32(em);em.load_state_dict(dqs,strict=True);ce=torch.compile(em,dynamic=False,fullgraph=True)
 torch.cuda.synchronize();tq=time.perf_counter()
 qvl,qvb=eval_val(a,ce,rk,ws,dev,ga,vt,bl,hl,il,esl=esl)
 torch.cuda.synchronize()
 log0(f"final_int6_roundtrip val_loss:{qvl:.4f} val_bpb:{qvb:.4f} eval_time:{1000.0*(time.perf_counter()-tq):.0f}ms")
 log0(f"final_int6_roundtrip_exact val_loss:{qvl:.8f} val_bpb:{qvb:.8f}")
 swsl=esl
 if a.eval_stride>0 and a.eval_stride<swsl:
  torch.cuda.synchronize();ts=time.perf_counter()
  svl,svb=eval_slide(a,em,rk,ws,dev,vt,bl,hl,il,stride=a.eval_stride,esl=swsl)
  torch.cuda.synchronize()
  log0(f"final_int6_sliding_window val_loss:{svl:.4f} val_bpb:{svb:.4f} stride:{a.eval_stride} eval_time:{1000.0*(time.perf_counter()-ts):.0f}ms")
  log0(f"final_int6_sliding_window_exact val_loss:{svl:.8f} val_bpb:{svb:.8f}")
 if dd:dist.destroy_process_group()
if __name__=="__main__":main()
