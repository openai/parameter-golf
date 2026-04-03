from __future__ import annotations
import copy,glob,io,lzma,math,os,random,subprocess,sys,time,uuid,zlib
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor,nn
try:
    from flash_attn_interface import flash_attn_func as fa3
    _FA3=True
except ImportError:
    _FA3=False
class H:
    data_path=os.environ.get("DATA_PATH","./data/datasets/fineweb10B_sp1024")
    train_files=os.path.join(data_path,"fineweb_train_*.bin")
    val_files=os.path.join(data_path,"fineweb_val_*.bin")
    tokenizer_path=os.environ.get("TOKENIZER_PATH","./data/tokenizers/fineweb_1024_bpe.model")
    run_id=os.environ.get("RUN_ID",str(uuid.uuid4()))
    seed=int(os.environ.get("SEED",1337))
    val_batch_size=int(os.environ.get("VAL_BATCH_SIZE",524288))
    val_loss_every=int(os.environ.get("VAL_LOSS_EVERY",4000))
    train_log_every=int(os.environ.get("TRAIN_LOG_EVERY",500))
    iterations=int(os.environ.get("ITERATIONS",20000))
    warmdown_iters=int(os.environ.get("WARMDOWN_ITERS",3500))
    warmup_steps=int(os.environ.get("WARMUP_STEPS",20))
    train_batch_tokens=int(os.environ.get("TRAIN_BATCH_TOKENS",786432))
    train_seq_len=int(os.environ.get("TRAIN_SEQ_LEN",2048))
    eval_seq_len=int(os.environ.get("EVAL_SEQ_LEN",2048))
    max_wallclock_seconds=float(os.environ.get("MAX_WALLCLOCK_SECONDS",600.0))
    qk_gain_init=float(os.environ.get("QK_GAIN_INIT",1.5))
    vocab_size=int(os.environ.get("VOCAB_SIZE",1024))
    num_layers=int(os.environ.get("NUM_LAYERS",11))
    num_kv_heads=int(os.environ.get("NUM_KV_HEADS",4))
    model_dim=int(os.environ.get("MODEL_DIM",512))
    num_heads=int(os.environ.get("NUM_HEADS",8))
    mlp_mult=float(os.environ.get("MLP_MULT",3.0))
    tie_embeddings=bool(int(os.environ.get("TIE_EMBEDDINGS","1")))
    rope_base=float(os.environ.get("ROPE_BASE",10000.0))
    logit_softcap=float(os.environ.get("LOGIT_SOFTCAP",30.0))
    embed_lr=float(os.environ.get("EMBED_LR",0.6))
    head_lr=float(os.environ.get("HEAD_LR",0.008))
    tied_embed_lr=float(os.environ.get("TIED_EMBED_LR",0.035))
    tied_embed_init_std=float(os.environ.get("TIED_EMBED_INIT_STD",0.005))
    matrix_lr=float(os.environ.get("MATRIX_LR",0.025))
    scalar_lr=float(os.environ.get("SCALAR_LR",0.025))
    muon_momentum=float(os.environ.get("MUON_MOMENTUM",0.99))
    muon_backend_steps=int(os.environ.get("MUON_BACKEND_STEPS",5))
    muon_momentum_warmup_start=float(os.environ.get("MUON_MOMENTUM_WARMUP_START",0.92))
    muon_momentum_warmup_steps=int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS",1500))
    beta1=float(os.environ.get("BETA1",0.9))
    beta2=float(os.environ.get("BETA2",0.95))
    adam_eps=float(os.environ.get("ADAM_EPS",1e-8))
    grad_clip_norm=float(os.environ.get("GRAD_CLIP_NORM",0.3))
    eval_stride=int(os.environ.get("EVAL_STRIDE",16))
    muon_wd=float(os.environ.get("MUON_WD",0.04))
    adam_wd=float(os.environ.get("ADAM_WD",0.04))
    bigram_vocab_size=int(os.environ.get("BIGRAM_VOCAB_SIZE",2048))
    bigram_dim=int(os.environ.get("BIGRAM_DIM",128))
    trigram_vocab_size=int(os.environ.get("TRIGRAM_VOCAB_SIZE",0))
    trigram_dim=int(os.environ.get("TRIGRAM_DIM",128))
    xsa_last_n=int(os.environ.get("XSA_LAST_N",4))
    rope_dims=int(os.environ.get("ROPE_DIMS",16))
    ln_scale=bool(int(os.environ.get("LN_SCALE","1")))
    late_qat_threshold=float(os.environ.get("LATE_QAT_THRESHOLD",0.15))
    ve_enabled=bool(int(os.environ.get("VE_ENABLED","1")))
    ve_dim=int(os.environ.get("VE_DIM",128))
    ve_layers=os.environ.get("VE_LAYERS","9,10")
    gated_attention=bool(int(os.environ.get("GATED_ATTENTION","0")))
    value_residual=bool(int(os.environ.get("VALUE_RESIDUAL","0")))
    swa_enabled=bool(int(os.environ.get("SWA_ENABLED","1")))
    swa_every=int(os.environ.get("SWA_EVERY",50))
    depth_recur_layers=os.environ.get("DEPTH_RECUR_LAYERS","")
    depth_recur_passes=int(os.environ.get("DEPTH_RECUR_PASSES",1))
    eval_temperature=float(os.environ.get("EVAL_TEMPERATURE",0.90))
    ttt_enabled=bool(int(os.environ.get("TTT_ENABLED","0")))
    ttt_lr=float(os.environ.get("TTT_LR",0.002))
    ttt_epochs=int(os.environ.get("TTT_EPOCHS",3))
    ttt_chunk_tokens=int(os.environ.get("TTT_CHUNK_TOKENS",32768))
    ttt_freeze_blocks=int(os.environ.get("TTT_FREEZE_BLOCKS",0))
    ttt_momentum=float(os.environ.get("TTT_MOMENTUM",0.9))
    ttt_batch_seqs=int(os.environ.get("TTT_BATCH_SEQS",32))
    ttt_grad_clip=float(os.environ.get("TTT_GRAD_CLIP",1.0))
    torch_compile=bool(int(os.environ.get("TORCH_COMPILE","1")))
    lr_warmup_steps=int(os.environ.get("LR_WARMUP_STEPS",50))
    ema_start_frac=float(os.environ.get("EMA_START_FRAC",0.4))

CTRL=tuple(p for p in os.environ.get("CONTROL_TENSOR_NAME_PATTERNS",
    "attn_scale,mlp_scale,resid_mix,q_gain,skip_weight,smear,attn_gate,vr_lambda,ve_layer_scales,ve_shared.scale").split(",") if p)

def ns5(G:Tensor,steps:int=5,eps:float=1e-7)->Tensor:
    a,b,c=(3.4445,-4.7750,2.0315)
    was2d=G.ndim==2
    if was2d:G=G.unsqueeze(0)
    X=G.bfloat16()
    tr=X.size(-2)>X.size(-1)
    if tr:X=X.mT
    X=X/(X.norm(dim=(-2,-1),keepdim=True)+eps)
    for _ in range(steps):
        A=X@X.mT;B=b*A+c*(A@A);X=a*X+B@X
    if tr:X=X.mT
    return X.squeeze(0) if was2d else X

class Muon(torch.optim.Optimizer):
    def __init__(self,params,lr:float,momentum:float,backend_steps:int,
                 nesterov:bool=True,weight_decay:float=0.0):
        super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,
                                     nesterov=nesterov,weight_decay=weight_decay))
        self._built=False
    def _build(self):
        self._dist=dist.is_available() and dist.is_initialized()
        ws=dist.get_world_size() if self._dist else 1
        self._ws=ws;self._rank=dist.get_rank() if self._dist else 0
        self._bm=[]
        for g in self.param_groups:
            for p in g["params"]:
                B=p.shape[0];pB=((B+ws-1)//ws)*ws;sB=pB//ws;tail=p.shape[1:];d=p.device
                self._bm.append({'p':p,'B':B,
                    'pg':torch.zeros(pB,*tail,device=d,dtype=torch.bfloat16),
                    'sh':torch.zeros(sB,*tail,device=d,dtype=torch.bfloat16),
                    'sm':torch.zeros(sB,*tail,device=d,dtype=torch.bfloat16),
                    'fu':torch.zeros(pB,*tail,device=d,dtype=torch.bfloat16),
                    'sc':max(1,p.shape[-2]/p.shape[-1])**0.5})
        self._bm.sort(key=lambda m:-m['p'].numel())
        self._built=True
    def launch_reduce_scatters(self):
        if not self._built:self._build()
        if not self._dist:return
        self._rsf=[]
        for m in self._bm:
            p=m['p']
            if p.grad is None:self._rsf.append(None);continue
            pg=m['pg'];pg[:m['B']].copy_(p.grad.bfloat16())
            if pg.shape[0]>m['B']:pg[m['B']:].zero_()
            self._rsf.append(dist.reduce_scatter_tensor(m['sh'],pg,op=dist.ReduceOp.AVG,async_op=True))
    @torch.no_grad()
    def step(self,closure=None):
        if not self._built:self._build()
        for g in self.param_groups:
            lr,mom,bks,nest,wd=g["lr"],g["momentum"],g["backend_steps"],g["nesterov"],g.get("weight_decay",0.0)
            pah=None;pm=None
            sharded=self._dist and hasattr(self,'_rsf')
            for i,m in enumerate(self._bm):
                p=m['p']
                if p.grad is None:continue
                if pah is not None:
                    pah.wait();pp=pm['p'];u=pm['fu'][:pm['B']]
                    if wd>0:pp.data.mul_(1-lr*wd)
                    pp.add_(u.to(dtype=pp.dtype),alpha=-lr*pm['sc'])
                if sharded and self._rsf[i] is not None:
                    self._rsf[i].wait();g_=m['sh'];buf=m['sm']
                else:
                    g_=p.grad.bfloat16()
                    st=self.state[p]
                    if "mb" not in st:st["mb"]=torch.zeros_like(g_)
                    buf=st["mb"]
                buf.mul_(mom).add_(g_)
                upd=g_.add(buf,alpha=mom) if nest else buf
                upd=ns5(upd,steps=bks)
                if sharded:
                    pah=dist.all_gather_into_tensor(m['fu'],upd,async_op=True);pm=m
                else:
                    if wd>0:p.data.mul_(1-lr*wd)
                    p.add_(upd.to(dtype=p.dtype),alpha=-lr*m['sc'])
            if pah is not None:
                pah.wait();pp=pm['p'];u=pm['fu'][:pm['B']]
                if wd>0:pp.data.mul_(1-lr*wd)
                pp.add_(u.to(dtype=pp.dtype),alpha=-lr*pm['sc'])
            if hasattr(self,'_rsf'):del self._rsf

def build_sp_luts(sp,vocab_size,device):
    sv=int(sp.vocab_size());ts=max(sv,vocab_size)
    bb=np.zeros(ts,dtype=np.int16);hs=np.zeros(ts,dtype=np.bool_);ib=np.ones(ts,dtype=np.bool_)
    for t in range(sv):
        if sp.is_control(t) or sp.is_unknown(t) or sp.is_unused(t):continue
        ib[t]=False
        if sp.is_byte(t):bb[t]=1;continue
        p=sp.id_to_piece(t)
        if p.startswith("\u2581"):hs[t]=True;p=p[1:]
        bb[t]=len(p.encode("utf-8"))
    return(torch.tensor(bb,dtype=torch.int16,device=device),
           torch.tensor(hs,dtype=torch.bool,device=device),
           torch.tensor(ib,dtype=torch.bool,device=device))

def load_shard(f):
    hb=256*np.dtype("<i4").itemsize;tb=np.dtype("<u2").itemsize
    h=np.fromfile(f,dtype="<i4",count=256)
    if h.size!=256 or int(h[0])!=20240520 or int(h[1])!=1:raise ValueError(f"Bad header {f}")
    n=int(h[2])
    if f.stat().st_size!=hb+n*tb:raise ValueError(f"Size mismatch {f}")
    t=np.fromfile(f,dtype="<u2",count=n,offset=hb)
    return torch.from_numpy(t.astype(np.uint16,copy=False))

def load_val_tokens(pat,sl):
    fs=[Path(p) for p in sorted(glob.glob(pat))]
    if not fs:raise FileNotFoundError(f"No files: {pat}")
    t=torch.cat([load_shard(f) for f in fs]).contiguous()
    u=((t.numel()-1)//sl)*sl
    if u<=0:raise ValueError("Val too short")
    return t[:u+1]

class TokenStream:
    def __init__(self,pat):
        self.fs=[Path(p) for p in sorted(glob.glob(pat))]
        if not self.fs:raise FileNotFoundError(pat)
        self.fi=0;self.t=load_shard(self.fs[0]);self.p=0
    def _adv(self):self.fi=(self.fi+1)%len(self.fs);self.t=load_shard(self.fs[self.fi]);self.p=0
    def take(self,n):
        c=[];r=n
        while r>0:
            a=self.t.numel()-self.p
            if a<=0:self._adv();continue
            k=min(r,a);c.append(self.t[self.p:self.p+k]);self.p+=k;r-=k
        return c[0] if len(c)==1 else torch.cat(c)

class DTokenLoader:
    def __init__(self,pat,rank,ws,dev):
        self.rank=rank;self.ws=ws;self.dev=dev;self.s=TokenStream(pat)
    def next(self,gt,sl,gas):
        lt=gt//(self.ws*gas);ps=lt+1;ch=self.s.take(ps*self.ws)
        s=self.rank*ps;l=ch[s:s+ps].to(dtype=torch.int64)
        x=l[:-1].reshape(-1,sl);y=l[1:].reshape(-1,sl)
        return x.to(self.dev,non_blocking=True),y.to(self.dev,non_blocking=True)

def q6_row(t,cr=31):
    t32=t.float()
    if t32.ndim==2:
        bq=bs=None;be=float('inf')
        for p in[.999,.9995,.9999,.99999,1.]:
            rc=torch.quantile(t32.abs(),p,dim=1) if p<1 else t32.abs().amax(dim=1)
            s=(rc/cr).clamp_min(1./cr).to(torch.float16)
            q=torch.clamp(torch.round(t32/s.float()[:,None]),-cr,cr).to(torch.int8)
            e=(t32-q.float()*s.float()[:,None]).pow(2).mean().item()
            if e<be:bq,bs,be=q,s,e
        return bq,bs
    am=t32.abs().max().item()
    s=torch.tensor(am/cr if am>0 else 1.,dtype=torch.float16)
    return torch.clamp(torch.round(t32/s.float()),-cr,cr).to(torch.int8),s

def q8_row(t):
    t32=t.float()
    if t32.ndim==2:
        ca=torch.quantile(t32.abs(),0.9999984,dim=1) if t32.numel() else torch.empty(t32.shape[0],dtype=torch.float32)
        cl=torch.maximum(torch.minimum(t32,ca[:,None]),-ca[:,None])
        s=(ca/127.).clamp_min(1./127.);q=torch.clamp(torch.round(cl/s[:,None]),-127,127).to(torch.int8)
        return q.contiguous(),s.to(torch.float16).contiguous()
    ca=float(torch.quantile(t32.abs().flatten(),0.9999984).item()) if t32.numel() else 0.
    s=torch.tensor(ca/127. if ca>0 else 1.,dtype=torch.float32)
    return torch.clamp(torch.round(torch.clamp(t32,-ca,ca)/s),-127,127).to(torch.int8).contiguous(),s

class RN(nn.Module):
    def __init__(self,eps=None):super().__init__();self.eps=eps
    def forward(self,x):return F.rms_norm(x,(x.size(-1),),eps=self.eps)

class CL(nn.Linear):
    _qat=False
    def forward(self,x):
        w=self.weight.to(x.dtype)
        if CL._qat and self.training and w.ndim==2:
            with torch.no_grad():
                w32=self.weight.float();rm=w32.abs().amax(dim=1)
                s=(rm/31.).clamp_min(1./31.)
                wq=(torch.clamp(torch.round(w32/s[:,None]),-32,31)*s[:,None]).to(x.dtype)
            w=w+(wq-w).detach()
        b=self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x,w,b)

def fix_fp32(mod):
    with torch.no_grad():
        for n,p in mod.named_parameters():
            if(p.ndim<2 or any(c in n for c in CTRL))and p.dtype!=torch.float32:
                p.data=p.data.float()

class Rotary(nn.Module):
    def __init__(self,dim,base=10000.,tsl=1024,rd=0):
        super().__init__()
        self.dim=dim;self.base=base;self.tsl=tsl;self.rd=rd if rd>0 else dim
        self.register_buffer("inv_freq",1./(base**(torch.arange(0,self.rd,2,dtype=torch.float32)/self.rd)),persistent=False)
        self._sl=0;self._c=None;self._s=None
    def forward(self,sl,dev,dt):
        if self._c is None or self._sl!=sl or self._c.device!=dev:
            rd=self.rd
            if sl>self.tsl:
                sc=sl/self.tsl;nb=self.base*(sc**(rd/(rd-2)))
                iv=1./(nb**(torch.arange(0,rd,2,dtype=torch.float32,device=dev)/rd))
            else:iv=self.inv_freq.to(dev)
            t=torch.arange(sl,device=dev,dtype=iv.dtype);fr=torch.outer(t,iv)
            self._c=fr.cos()[None,:,None,:];self._s=fr.sin()[None,:,None,:];self._sl=sl
        return self._c.to(dtype=dt),self._s.to(dtype=dt)

def apply_rope(x,cos,sin,rd=0):
    if rd>0 and rd<x.size(-1):
        xr,xp=x[...,:rd],x[...,rd:]
        h=rd//2;x1,x2=xr[...,:h],xr[...,h:]
        xr=torch.cat((x1*cos+x2*sin,x1*(-sin)+x2*cos),dim=-1)
        return torch.cat((xr,xp),dim=-1)
    h=x.size(-1)//2;x1,x2=x[...,:h],x[...,h:]
    return torch.cat((x1*cos+x2*sin,x1*(-sin)+x2*cos),dim=-1)

class Attn(nn.Module):
    def __init__(self,dim,nh,nkv,rb,qgi,ga=False,vr=False):
        super().__init__()
        self.nh=nh;self.nkv=nkv;self.hd=dim//nh;self.rd=0;self.use_xsa=False
        self.q_gain=nn.Parameter(torch.full((nh,),qgi,dtype=torch.float32))
        self.rotary=Rotary(self.hd,base=rb,tsl=1024)
        self.ga=ga
        if ga:
            self.attn_gate=nn.Linear(dim,nh,bias=True)
            nn.init.zeros_(self.attn_gate.weight);nn.init.constant_(self.attn_gate.bias,4.)
        self.vr=vr
        if vr:self.vr_lambda=nn.Parameter(torch.tensor([.5,.5],dtype=torch.float32))
    def _xsa(self,y,v):
        B,T,H,D=y.shape;Hk=v.size(-2);g=H//Hk
        yg=y.reshape(B,T,Hk,g,D);vn=F.normalize(v,dim=-1).unsqueeze(-2)
        return(yg-(yg*vn).sum(dim=-1,keepdim=True)*vn).reshape(B,T,H,D)
    def forward(self,x,qw,kw,vw,ow,ve=None,v0=None):
        B,T,D=x.shape
        q=F.linear(x,qw.to(x.dtype)).reshape(B,T,self.nh,self.hd)
        k=F.linear(x,kw.to(x.dtype)).reshape(B,T,self.nkv,self.hd)
        v=F.linear(x,vw.to(x.dtype))
        if ve is not None:v=v+ve
        v=v.reshape(B,T,self.nkv,self.hd)
        raw_v=v if self.vr else None
        if self.vr and v0 is not None:
            lm=self.vr_lambda.to(dtype=v.dtype);v=lm[0]*v0+lm[1]*v
        q=F.rms_norm(q,(q.size(-1),));k=F.rms_norm(k,(k.size(-1),))
        cos,sin=self.rotary(T,x.device,q.dtype)
        q=apply_rope(q,cos,sin,self.rd);k=apply_rope(k,cos,sin,self.rd)
        q=q*self.q_gain.to(dtype=q.dtype)[None,None,:,None]
        if _FA3:y=fa3(q,k,v,causal=True)
        else:
            q2=q.transpose(1,2);k2=k.transpose(1,2);v2=v.transpose(1,2)
            y=F.scaled_dot_product_attention(q2,k2,v2,is_causal=True,
                enable_gqa=(self.nkv!=self.nh)).transpose(1,2).contiguous()
        if self.use_xsa:y=self._xsa(y,v)
        if self.ga:y=y*torch.sigmoid(self.attn_gate(x)).unsqueeze(-1)
        return F.linear(y.reshape(B,T,D),ow.to(x.dtype)),raw_v

class SmearGate(nn.Module):
    def __init__(self,d):super().__init__();self.gate=nn.Parameter(torch.zeros(d,dtype=torch.float32))
    def forward(self,x):
        g=torch.sigmoid(self.gate.to(dtype=x.dtype))[None,None,:]
        return(1-g)*x+g*torch.cat([torch.zeros_like(x[:,:1]),x[:,:-1]],dim=1)

class BigramHash(nn.Module):
    def __init__(self,bv,bd,md):
        super().__init__()
        self.bv=bv;self.emb=nn.Embedding(bv,bd);nn.init.zeros_(self.emb.weight)
        self.proj=CL(bd,md,bias=False) if bd!=md else None
        if self.proj:nn.init.zeros_(self.proj.weight)
        self.sc=nn.Parameter(torch.tensor(.05,dtype=torch.float32))
    def forward(self,ids):
        t=ids.to(torch.int32);m=self.bv-1;o=torch.empty_like(t)
        o[...,0]=m;o[...,1:]=torch.bitwise_xor(36313*t[...,1:],27191*t[...,:-1])%m
        h=self.emb(o.long())
        if self.proj:h=self.proj(h)
        return h*self.sc.to(dtype=h.dtype)

class TrigramHash(nn.Module):
    def __init__(self,tv,td,md):
        super().__init__()
        self.tv=tv;self.emb=nn.Embedding(tv,td);nn.init.zeros_(self.emb.weight)
        self.proj=CL(td,md,bias=False) if td!=md else None
        if self.proj:nn.init.zeros_(self.proj.weight)
        self.sc=nn.Parameter(torch.tensor(.03,dtype=torch.float32))
    def forward(self,ids):
        t=ids.to(torch.int32);m=self.tv-1;o=torch.empty_like(t)
        o[...,0]=m;o[...,1]=torch.bitwise_xor(36313*t[...,1],27191*t[...,0])%m
        o[...,2:]=torch.bitwise_xor(torch.bitwise_xor(48271*t[...,2:],36313*t[...,1:-1]),27191*t[...,:-2])%m
        h=self.emb(o.long())
        if self.proj:h=self.proj(h)
        return h*self.sc.to(dtype=h.dtype)

class VE(nn.Module):
    def __init__(self,vs,vd,md):
        super().__init__()
        self.emb=nn.Embedding(vs,vd);nn.init.normal_(self.emb.weight,std=.01)
        self.proj=CL(vd,md,bias=False) if vd!=md else None
        if self.proj:nn.init.zeros_(self.proj.weight)
        self.sc=nn.Parameter(torch.tensor(.1,dtype=torch.float32))
    def forward(self,ids):
        h=self.emb(ids)
        if self.proj:h=self.proj(h)
        return h*self.sc.to(dtype=h.dtype)

class MLP(nn.Module):
    def __init__(self,d,mm):super().__init__()
    def forward(self,x,uw,dw):
        return F.linear(F.leaky_relu(F.linear(x,uw.to(x.dtype)),negative_slope=0.5).square(),dw.to(x.dtype))

class Block(nn.Module):
    def __init__(self,d,nh,nkv,mm,rb,qgi,li=0,lns=False,ga=False,vr=False):
        super().__init__()
        self.an=RN();self.mn=RN()
        self.attn=Attn(d,nh,nkv,rb,qgi,ga=ga,vr=vr);self.mlp=MLP(d,mm)
        self.attn_scale=nn.Parameter(torch.ones(d,dtype=torch.float32))
        self.mlp_scale=nn.Parameter(torch.ones(d,dtype=torch.float32))
        self.resid_mix=nn.Parameter(torch.stack((torch.ones(d),torch.zeros(d))).float())
        self.lsf=1./math.sqrt(li+1) if lns else 1.
    def forward(self,x,x0,qw,kw,vw,ow,uw,dw,ve=None,v0=None):
        m=self.resid_mix.to(dtype=x.dtype)
        xi=m[0][None,None,:]*x+m[1][None,None,:]*x0
        ao,rv=self.attn(self.an(xi)*self.lsf,qw,kw,vw,ow,ve=ve,v0=v0)
        xo=xi+self.attn_scale.to(dtype=xi.dtype)[None,None,:]*ao
        xo=xo+self.mlp_scale.to(dtype=xo.dtype)[None,None,:]*self.mlp(self.mn(xo)*self.lsf,uw,dw)
        return xo,rv

class GPT(nn.Module):
    def __init__(self,args):
        super().__init__()
        a=args;vs=a.vocab_size;nl=a.num_layers;md=a.model_dim;nh=a.num_heads
        nkv=a.num_kv_heads;hd=md//nh;kvd=nkv*hd;mlpd=int(a.mlp_mult*md)
        self.tie=a.tie_embeddings;self.teis=a.tied_embed_init_std;self.lsc=a.logit_softcap
        self.vr=a.value_residual;self.nl=nl;self.md=md
        self.tok_emb=nn.Embedding(vs,md)
        self.bigram=BigramHash(a.bigram_vocab_size,a.bigram_dim,md) if a.bigram_vocab_size>0 else None
        self.trigram=TrigramHash(a.trigram_vocab_size,a.trigram_dim,md) if a.trigram_vocab_size>0 else None
        self.smear=SmearGate(md)
        self.nel=nl//2;self.ndl=nl-self.nel
        self.nskip=min(self.nel,self.ndl)
        self.skip_weights=nn.Parameter(torch.ones(self.nskip,md,dtype=torch.float32))
        self.qo_bank=nn.Parameter(torch.empty(2*nl,md,md))
        self.kv_bank=nn.Parameter(torch.empty(2*nl,kvd,md))
        self.mlp_up_bank=nn.Parameter(torch.empty(nl,mlpd,md))
        self.mlp_down_bank=nn.Parameter(torch.empty(nl,md,mlpd))
        self.blocks=nn.ModuleList([Block(md,nh,nkv,a.mlp_mult,a.rope_base,a.qk_gain_init,
            li=i,lns=a.ln_scale,ga=a.gated_attention,vr=a.value_residual) for i in range(nl)])
        if a.rope_dims>0:
            for b in self.blocks:b.attn.rd=a.rope_dims;b.attn.rotary=Rotary(hd,base=a.rope_base,tsl=1024,rd=a.rope_dims)
        self.dr_layers=[int(x) for x in a.depth_recur_layers.split(",") if x.strip()] if a.depth_recur_passes>1 else []
        self.dr_passes=a.depth_recur_passes
        self.vel=[int(x) for x in a.ve_layers.split(",") if x.strip()] if a.ve_enabled else []
        if self.vel:
            self.ve_shared=VE(vs,a.ve_dim,kvd)
            self.ve_scales=nn.ParameterList([nn.Parameter(torch.ones(1,dtype=torch.float32)) for _ in self.vel])
        else:self.ve_shared=None;self.ve_scales=nn.ParameterList()
        self.final_norm=RN()
        self.lm_head=None if a.tie_embeddings else CL(md,vs,bias=False)
        if self.lm_head:self.lm_head._zero_init=True
        if a.xsa_last_n>0:
            for i in range(max(0,nl-a.xsa_last_n),nl):self.blocks[i].attn.use_xsa=True
        self._init(nl)
    def _init(self,n):
        if self.tie:nn.init.normal_(self.tok_emb.weight,mean=0.,std=self.teis)
        ps=1./math.sqrt(2*n)
        for i in range(n):
            nn.init.orthogonal_(self.qo_bank.data[i]);nn.init.zeros_(self.qo_bank.data[n+i])
            nn.init.orthogonal_(self.kv_bank.data[i]);nn.init.orthogonal_(self.kv_bank.data[n+i])
            nn.init.orthogonal_(self.mlp_up_bank.data[i]);nn.init.zeros_(self.mlp_down_bank.data[i])
            self.qo_bank.data[n+i].mul_(ps);self.mlp_down_bank.data[i].mul_(ps)
        for _,m in self.named_modules():
            if isinstance(m,nn.Linear):
                if getattr(m,"_zero_init",False):nn.init.zeros_(m.weight)
                elif m.weight.ndim==2 and min(m.weight.shape)>=64:nn.init.orthogonal_(m.weight)
    def _ve(self,li,ids,vc):
        if self.ve_shared is None or li not in self.vel:return None
        if 've' not in vc:vc['ve']=self.ve_shared(ids)
        return vc['ve']*self.ve_scales[self.vel.index(li)].to(dtype=vc['ve'].dtype)
    def forward(self,ids,tgt):
        n=self.nl;x=self.tok_emb(ids)
        if self.bigram:x=x+self.bigram(ids)
        if self.trigram:x=x+self.trigram(ids)
        x=F.rms_norm(x,(x.size(-1),));x=self.smear(x);x0=x;v0=None;skips=[];vc={}
        for i in range(self.nel):
            ve=self._ve(i,ids,vc)
            x,rv=self.blocks[i](x,x0,self.qo_bank[i],self.kv_bank[i],self.kv_bank[n+i],
                self.qo_bank[n+i],self.mlp_up_bank[i],self.mlp_down_bank[i],ve=ve,v0=v0)
            if v0 is None and rv is not None:v0=rv
            if i in self.dr_layers:
                for _ in range(self.dr_passes-1):
                    x,_=self.blocks[i](x,x0,self.qo_bank[i],self.kv_bank[i],self.kv_bank[n+i],
                        self.qo_bank[n+i],self.mlp_up_bank[i],self.mlp_down_bank[i],ve=ve,v0=v0)
            skips.append(x)
        for i in range(self.ndl):
            bi=self.nel+i
            if skips:x=x+self.skip_weights[i].to(dtype=x.dtype)[None,None,:]*skips.pop()
            ve=self._ve(bi,ids,vc)
            x,_=self.blocks[bi](x,x0,self.qo_bank[bi],self.kv_bank[bi],self.kv_bank[n+bi],
                self.qo_bank[n+bi],self.mlp_up_bank[bi],self.mlp_down_bank[bi],ve=ve,v0=v0)
            if bi in self.dr_layers:
                for _ in range(self.dr_passes-1):
                    x,_=self.blocks[bi](x,x0,self.qo_bank[bi],self.kv_bank[bi],self.kv_bank[n+bi],
                        self.qo_bank[n+bi],self.mlp_up_bank[bi],self.mlp_down_bank[bi],ve=ve,v0=v0)
        x=self.final_norm(x);xf=x.reshape(-1,x.size(-1));tg=tgt.reshape(-1)
        lp=F.linear(xf,self.tok_emb.weight) if self.tie else self.lm_head(xf)
        lg=self.lsc*torch.tanh(lp/self.lsc)
        return F.cross_entropy(lg.float(),tg,reduction="mean")
    def forward_logits(self,ids):
        n=self.nl;x=self.tok_emb(ids)
        if self.bigram:x=x+self.bigram(ids)
        if self.trigram:x=x+self.trigram(ids)
        x=F.rms_norm(x,(x.size(-1),));x=self.smear(x);x0=x;v0=None;skips=[];vc={}
        for i in range(self.nel):
            ve=self._ve(i,ids,vc)
            x,rv=self.blocks[i](x,x0,self.qo_bank[i],self.kv_bank[i],self.kv_bank[n+i],
                self.qo_bank[n+i],self.mlp_up_bank[i],self.mlp_down_bank[i],ve=ve,v0=v0)
            if v0 is None and rv is not None:v0=rv
            if i in self.dr_layers:
                for _ in range(self.dr_passes-1):
                    x,_=self.blocks[i](x,x0,self.qo_bank[i],self.kv_bank[i],self.kv_bank[n+i],
                        self.qo_bank[n+i],self.mlp_up_bank[i],self.mlp_down_bank[i],ve=ve,v0=v0)
            skips.append(x)
        for i in range(self.ndl):
            bi=self.nel+i
            if skips:x=x+self.skip_weights[i].to(dtype=x.dtype)[None,None,:]*skips.pop()
            ve=self._ve(bi,ids,vc)
            x,_=self.blocks[bi](x,x0,self.qo_bank[bi],self.kv_bank[bi],self.kv_bank[n+bi],
                self.qo_bank[n+bi],self.mlp_up_bank[bi],self.mlp_down_bank[bi],ve=ve,v0=v0)
            if bi in self.dr_layers:
                for _ in range(self.dr_passes-1):
                    x,_=self.blocks[bi](x,x0,self.qo_bank[bi],self.kv_bank[bi],self.kv_bank[n+bi],
                        self.qo_bank[n+bi],self.mlp_up_bank[bi],self.mlp_down_bank[bi],ve=ve,v0=v0)
        x=self.final_norm(x)
        lp=F.linear(x,self.tok_emb.weight) if self.tie else self.lm_head(x)
        return self.lsc*torch.tanh(lp/self.lsc)

def eval_val(args,model,rank,ws,dev,vt,bbl,hsl,ibl,esl=None):
    sl=esl or args.train_seq_len;lbt=args.val_batch_size//ws
    if lbt<sl:raise ValueError("VAL_BATCH_SIZE too small")
    lbs=lbt//sl;ts=(vt.numel()-1)//sl;ss=(ts*rank)//ws;se=(ts*(rank+1))//ws
    ls=torch.zeros((),device=dev,dtype=torch.float64);tc=torch.zeros((),device=dev,dtype=torch.float64)
    bc=torch.zeros((),device=dev,dtype=torch.float64);model.eval()
    with torch.inference_mode():
        for bs in range(ss,se,lbs):
            be=min(bs+lbs,se);rs=bs*sl;re=be*sl+1
            l=vt[rs:re].to(device=dev,dtype=torch.int64,non_blocking=True)
            x=l[:-1].reshape(-1,sl);y=l[1:].reshape(-1,sl)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):bl_=model(x,y).detach()
            n=float(y.numel());ls+=bl_.to(torch.float64)*n;tc+=n
            pi=x.reshape(-1);ti=y.reshape(-1)
            tb=bbl[ti].to(torch.int16);tb+=(hsl[ti]&~ibl[pi]).to(torch.int16);bc+=tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in[ls,tc,bc]:dist.all_reduce(t,op=dist.ReduceOp.SUM)
    vl=ls/tc;bpt=vl.item()/math.log(2.);tpb=tc.item()/bc.item();model.train()
    return float(vl.item()),float(bpt*tpb)

def eval_slide(args,bm,rank,ws,dev,vt,bbl,hsl,ibl,stride,bseqs=32,esl=None):
    sl=esl or args.train_seq_len;tt=vt.numel()-1
    wstarts=[w for w in range(0,tt,stride) if min(w+sl,tt)-w>=1]
    tw=len(wstarts);ms=(tw*rank)//ws;me=(tw*(rank+1))//ws;mw=wstarts[ms:me]
    ls=torch.zeros((),device=dev,dtype=torch.float64);tc=torch.zeros((),device=dev,dtype=torch.float64)
    bc=torch.zeros((),device=dev,dtype=torch.float64);bm.eval()
    clog=torch.compile(bm.forward_logits,dynamic=False,fullgraph=True) if hasattr(args,'torch_compile') and args.torch_compile else bm.forward_logits
    with torch.inference_mode():
        for bi in range(0,len(mw),bseqs):
            bws=mw[bi:bi+bseqs];bsz=len(bws)
            xb=torch.zeros(bsz,sl,dtype=torch.int64,device=dev)
            yb=torch.zeros(bsz,sl,dtype=torch.int64,device=dev);wls=[]
            for i,w in enumerate(bws):
                e=min(w+sl,tt);wl=e-w;wls.append(wl)
                c=vt[w:e+1].to(dtype=torch.int64,device=dev);xb[i,:wl]=c[:-1];yb[i,:wl]=c[1:]
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):lg=clog(xb)
            if hasattr(args,'eval_temperature') and args.eval_temperature!=1.0:lg=lg/args.eval_temperature
            nl=F.cross_entropy(lg.reshape(-1,lg.size(-1)).float(),yb.reshape(-1),reduction="none").reshape(bsz,sl)
            for i,w in enumerate(bws):
                wl=wls[i];s=0 if w==0 else max(wl-stride,0)
                ls+=nl[i,s:wl].to(torch.float64).sum();tc+=float(wl-s)
                tg,pv=yb[i,s:wl],xb[i,s:wl]
                tb=bbl[tg].to(torch.float64);tb+=(hsl[tg]&~ibl[pv]).to(torch.float64);bc+=tb.sum()
    if dist.is_available() and dist.is_initialized():
        for t in[ls,tc,bc]:dist.all_reduce(t,op=dist.ReduceOp.SUM)
    vl=(ls/tc).item();bpt=vl/math.log(2.);tpb=tc.item()/bc.item();bm.train()
    return vl,bpt*tpb

def eval_ttt(args,bm,rank,ws,dev,vt,bbl,hsl,ibl,stride,bseqs=32,log0=print):
    sl=args.train_seq_len;tt=vt.numel()-1;tchk=args.ttt_chunk_tokens
    wstarts=[w for w in range(0,tt,stride) if min(w+sl,tt)-w>=stride or w==0]
    nc=(tt+tchk-1)//tchk;cw=[[] for _ in range(nc)]
    for w in wstarts:
        e=min(w+sl,tt);wl=e-w;s=0 if w==0 else max(wl-stride,0)
        ci=min((w+s)//tchk,nc-1);cw[ci].append(w)
    log0(f"ttt:start chunks={nc} windows={len(wstarts)}")
    ls=torch.zeros((),device=dev,dtype=torch.float64);tc=torch.zeros((),device=dev,dtype=torch.float64)
    bc=torch.zeros((),device=dev,dtype=torch.float64)
    fb=set(range(min(args.ttt_freeze_blocks,len(bm.blocks))));tp=[]
    for n,p in bm.named_parameters():
        fr=any(f"blocks.{bi}." in n for bi in fb)
        if fr:p.requires_grad_(False)
        else:p.requires_grad_(True);tp.append(p)
    opt=torch.optim.SGD(tp,lr=args.ttt_lr,momentum=args.ttt_momentum);t0=time.perf_counter()
    for ci in range(nc):
        ws_=cw[ci]
        if not ws_:continue
        ms_=(len(ws_)*rank)//ws;me_=(len(ws_)*(rank+1))//ws;mw_=ws_[ms_:me_]
        bm.eval()
        with torch.inference_mode():
            for bi in range(0,len(mw_),bseqs):
                bws=mw_[bi:bi+bseqs];bsz=len(bws)
                xb=torch.zeros(bsz,sl,dtype=torch.int64,device=dev)
                yb=torch.zeros(bsz,sl,dtype=torch.int64,device=dev);wls=[]
                for i,w in enumerate(bws):
                    e=min(w+sl,tt);wl=e-w;wls.append(wl)
                    c=vt[w:e+1].to(dtype=torch.int64,device=dev);xb[i,:wl]=c[:-1];yb[i,:wl]=c[1:]
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16):lg=bm.forward_logits(xb)
                nl=F.cross_entropy(lg.reshape(-1,lg.size(-1)).float(),yb.reshape(-1),reduction="none").reshape(bsz,sl)
                for i,w in enumerate(bws):
                    wl=wls[i];s=0 if w==0 else max(wl-stride,0)
                    ls+=nl[i,s:wl].to(torch.float64).sum();tc+=float(wl-s)
                    tg,pv=yb[i,s:wl],xb[i,s:wl]
                    tb=bbl[tg].to(torch.float64);tb+=(hsl[tg]&~ibl[pv]).to(torch.float64);bc+=tb.sum()
        last=ci==nc-1
        if not last and args.ttt_epochs>0:
            bm.train();cs=ci*tchk;ce=min((ci+1)*tchk,tt);cseqs=(ce-cs)//sl
            if cseqs>0:
                clr=args.ttt_lr*.5*(1+math.cos(math.pi*ci/max(nc-1,1)))
                for pg in opt.param_groups:pg['lr']=clr
                mss=(cseqs*rank)//ws;mse=(cseqs*(rank+1))//ws;mcs=mse-mss
                for _ in range(args.ttt_epochs):
                    for bs in range(0,mcs,args.ttt_batch_seqs):
                        be=min(bs+args.ttt_batch_seqs,mcs);ab=mss+bs
                        st=cs+ab*sl;et=cs+(mss+be)*sl+1
                        if et>vt.numel():continue
                        l=vt[st:et].to(device=dev,dtype=torch.int64)
                        x=l[:-1].reshape(-1,sl);y=l[1:].reshape(-1,sl)
                        opt.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):lo=bm(x,y)
                        lo.backward()
                        if ws>1:
                            for p in tp:
                                if p.grad is not None:dist.all_reduce(p.grad,op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(tp,args.ttt_grad_clip);opt.step()
        if rank==0 and(ci%10==0 or ci==nc-1):
            el=time.perf_counter()-t0;rl=ls.item()/max(tc.item(),1)
            rb=rl/math.log(2.)*(tc.item()/max(bc.item(),1)) if tc.item()>0 else 0.
            log0(f"  ttt[{ci+1}/{nc}] bpb={rb:.6f} t={el:.1f}s")
    if dist.is_available() and dist.is_initialized():
        for t in[ls,tc,bc]:dist.all_reduce(t,op=dist.ReduceOp.SUM)
    vl=(ls/tc).item();vb=vl/math.log(2.)*(tc.item()/bc.item())
    for p in bm.parameters():p.requires_grad_(True)
    bm.eval();log0(f"ttt:done bpb={vb:.6f} t={time.perf_counter()-t0:.1f}s")
    return vl,vb

def _unbank(sd,nl):
    out={};n=nl
    for k,t in sd.items():
        if k=="qo_bank":
            for i in range(n):out[f"blocks.{i}.attn.c_q.weight"]=t[i];out[f"blocks.{i}.attn.proj.weight"]=t[n+i]
        elif k=="kv_bank":
            for i in range(n):out[f"blocks.{i}.attn.c_k.weight"]=t[i];out[f"blocks.{i}.attn.c_v.weight"]=t[n+i]
        elif k=="mlp_up_bank":
            for i in range(n):out[f"blocks.{i}.mlp.fc.weight"]=t[i]
        elif k=="mlp_down_bank":
            for i in range(n):out[f"blocks.{i}.mlp.proj.weight"]=t[i]
        else:out[k]=t
    return out

def _rebank(sd,nl,tsd):
    out={};n=nl;qo=[None]*(2*n);kv=[None]*(2*n);up=[None]*n;dn=[None]*n;used=set()
    for i in range(n):
        for k,sl,j in[(f"blocks.{i}.attn.c_q.weight",qo,i),(f"blocks.{i}.attn.proj.weight",qo,n+i),
                       (f"blocks.{i}.attn.c_k.weight",kv,i),(f"blocks.{i}.attn.c_v.weight",kv,n+i),
                       (f"blocks.{i}.mlp.fc.weight",up,i),(f"blocks.{i}.mlp.proj.weight",dn,i)]:
            if k in sd:sl[j]=sd[k];used.add(k)
    out["qo_bank"]=torch.stack(qo).to(dtype=tsd["qo_bank"].dtype)
    out["kv_bank"]=torch.stack(kv).to(dtype=tsd["kv_bank"].dtype)
    out["mlp_up_bank"]=torch.stack(up).to(dtype=tsd["mlp_up_bank"].dtype)
    out["mlp_down_bank"]=torch.stack(dn).to(dtype=tsd["mlp_down_bank"].dtype)
    for k,t in sd.items():
        if k not in used:out[k]=t
    return out

def _cat(n):
    if ".mlp." in n:return "mlp"
    if ".attn." in n or ".proj." in n:return "attn"
    if "tok_emb" in n or "lm_head" in n:return "embed"
    return "other"

def mq6(sd,cats):
    r={};m={}
    for n,t in sd.items():
        t=t.detach().cpu().contiguous();c=_cat(n)
        if not t.is_floating_point() or t.numel()<=65536:
            r[n]=t.to(torch.float16) if t.is_floating_point() else t;m[n]="pt";continue
        if any(p in n for p in CTRL):r[n]=t.float();m[n]="ctrl";continue
        if c in cats and t.ndim>=1:
            q,s=q6_row(t);r[n+".q"]=q;r[n+".s"]=s;m[n]="q6"
        else:
            q,s=q8_row(t);r[n+".q"]=q;r[n+".s"]=s;m[n]="q8"
    return r,m

def dq6(r,m,tsd):
    out={}
    for n,o in tsd.items():
        i=m.get(n)
        if i is None:continue
        od=o.dtype
        if i in("pt","ctrl"):
            t=r[n]
            if t.dtype==torch.float16 and od in(torch.float32,torch.bfloat16):t=t.to(od)
            out[n]=t;continue
        q,s=r[n+".q"],r[n+".s"]
        if s.ndim>0:out[n]=(q.float()*s.float().view(q.shape[0],*([1]*(q.ndim-1)))).to(od)
        else:out[n]=(q.float()*float(s.item())).to(od)
    return out

def quantize_state_dict_int8(sd):
    q={};sc={};dt={};pt={};pod={};qm={};st={"pc":0,"nt":0,"nf":0,"nn":0,"bb":0,"ib":0}
    for n,t in sd.items():
        t=t.detach().cpu().contiguous();st["pc"]+=t.numel();st["nt"]+=1;st["bb"]+=t.numel()*t.element_size()
        if not t.is_floating_point():st["nn"]+=1;pt[n]=t;st["ib"]+=t.numel()*t.element_size();continue
        if t.numel()<=65536:
            if any(p in n for p in CTRL):k=t.float()
            else:
                if t.dtype in(torch.float32,torch.bfloat16):pod[n]=str(t.dtype).removeprefix("torch.");k=t.to(torch.float16)
                else:k=t
            pt[n]=k.contiguous();st["ib"]+=k.numel()*k.element_size();continue
        st["nf"]+=1;r,s=q8_row(t)
        if s.ndim>0:qm[n]={"scheme":"per_row","axis":0}
        q[n]=r;sc[n]=s;dt[n]=str(t.dtype).removeprefix("torch.");st["ib"]+=r.numel()*r.element_size()+s.numel()*s.element_size()
    obj={"__quant_format__":"int8_clean_per_row_v1","quantized":q,"scales":sc,"dtypes":dt,"passthrough":pt}
    if qm:obj["qmeta"]=qm
    if pod:obj["passthrough_orig_dtypes"]=pod
    return obj,st

def dequantize_state_dict_int8(obj):
    out={};qm=obj.get("qmeta",{});pod=obj.get("passthrough_orig_dtypes",{})
    for n,q in obj["quantized"].items():
        dt=getattr(torch,obj["dtypes"][n]);s=obj["scales"][n]
        if qm.get(n,{}).get("scheme")=="per_row" or s.ndim>0:
            out[n]=(q.float()*s.float().view(q.shape[0],*([1]*(q.ndim-1)))).to(dt)
        else:out[n]=(q.float()*float(s.item())).to(dt)
    for n,t in obj["passthrough"].items():
        ot=t.detach().cpu().contiguous();od=pod.get(n)
        if isinstance(od,str):ot=ot.to(getattr(torch,od))
        out[n]=ot
    return out

def main():
    code=Path(__file__).read_text(encoding="utf-8");args=H()
    distributed="RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank=int(os.environ.get("RANK","0"));ws=int(os.environ.get("WORLD_SIZE","1"))
    lr_=int(os.environ.get("LOCAL_RANK","0"))
    if 8%ws!=0:raise ValueError(f"WORLD_SIZE={ws} must divide 8")
    gas=8//ws;gsc=1./gas
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    dev=torch.device("cuda",lr_);torch.cuda.set_device(dev)
    if distributed:dist.init_process_group(backend="nccl",device_id=dev);dist.barrier()
    mp=rank==0;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    lf=None
    if mp:os.makedirs("logs",exist_ok=True);lf=f"logs/{args.run_id}.txt";print(lf)
    def log0(m,c=True):
        if not mp:return
        if c:print(m)
        if lf:
            with open(lf,"a",encoding="utf-8") as f:print(m,file=f)
    log0(code,c=False);log0("="*80,c=False)
    log0(f"PyTorch {torch.__version__}",c=False)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    sp=spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    esl=args.eval_seq_len if args.eval_seq_len>0 else args.train_seq_len
    vsl=max(args.train_seq_len,esl)
    vt=load_val_tokens(args.val_files,vsl)
    bbl,hsl,ibl=build_sp_luts(sp,args.vocab_size,dev)
    CL._qat=False
    bm=GPT(args).to(dev).bfloat16()
    bm.qo_bank.data=bm.qo_bank.data.float();bm.kv_bank.data=bm.kv_bank.data.float()
    bm.mlp_up_bank.data=bm.mlp_up_bank.data.float();bm.mlp_down_bank.data=bm.mlp_down_bank.data.float()
    for m in bm.modules():
        if isinstance(m,CL):m.float()
    fix_fp32(bm)
    cm=torch.compile(bm,dynamic=False,fullgraph=True) if args.torch_compile else bm;model=cm
    mps=[bm.qo_bank,bm.kv_bank,bm.mlp_up_bank,bm.mlp_down_bank]
    sps=[];bnp=list(bm.blocks.named_parameters())
    for n,p in bnp:
        if p.ndim<2 or any(c in n for c in CTRL):sps.append(p)
    if bm.skip_weights.numel()>0:sps.append(bm.skip_weights)
    sps.append(bm.smear.gate)
    if bm.bigram:sps.append(bm.bigram.sc)
    if bm.trigram:sps.append(bm.trigram.sc)
    tlr=args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tps=[{"params":[bm.tok_emb.weight],"lr":tlr,"base_lr":tlr}]
    if bm.bigram:
        tps.append({"params":[bm.bigram.emb.weight],"lr":tlr,"base_lr":tlr})
        if bm.bigram.proj:sps.append(bm.bigram.proj.weight)
    if bm.trigram:
        tps.append({"params":[bm.trigram.emb.weight],"lr":tlr,"base_lr":tlr})
        if bm.trigram.proj:sps.append(bm.trigram.proj.weight)
    if bm.ve_shared:
        tps.append({"params":[bm.ve_shared.emb.weight],"lr":tlr,"base_lr":tlr})
        if bm.ve_shared.proj:sps.append(bm.ve_shared.proj.weight)
        sps.append(bm.ve_shared.sc)
        for s in bm.ve_scales:sps.append(s)
    otk=torch.optim.AdamW(tps,betas=(args.beta1,args.beta2),eps=args.adam_eps,weight_decay=args.adam_wd,fused=True)
    omu=Muon(mps,lr=args.matrix_lr,momentum=args.muon_momentum,backend_steps=args.muon_backend_steps,weight_decay=args.muon_wd)
    for g in omu.param_groups:g["base_lr"]=args.matrix_lr
    osc=torch.optim.AdamW([{"params":sps,"lr":args.scalar_lr,"base_lr":args.scalar_lr}],
        betas=(args.beta1,args.beta2),eps=args.adam_eps,weight_decay=args.adam_wd,fused=True)
    rps=list(otk.param_groups[0]["params"])
    for pg in otk.param_groups[1:]:rps.extend(pg["params"])
    rps.extend(sps)
    ohd=None
    if bm.lm_head:
        ohd=torch.optim.Adam([{"params":[bm.lm_head.weight],"lr":args.head_lr,"base_lr":args.head_lr}],
            betas=(args.beta1,args.beta2),eps=args.adam_eps,fused=True)
        rps.append(bm.lm_head.weight)
    opts=[otk,omu,osc]+([] if ohd is None else [ohd])
    np_=sum(p.numel() for p in bm.parameters())
    log0(f"params:{np_} ws:{ws} gas:{gas}")
    tl=DTokenLoader(args.train_files,rank,ws,dev)
    def zg():
        for o in opts:o.zero_grad(set_to_none=True)
    mwms=1000.*args.max_wallclock_seconds if args.max_wallclock_seconds>0 else None
    def lr_m(step,ems):
        wu=min(step/max(args.lr_warmup_steps,1),1.) if args.lr_warmup_steps>0 else 1.
        if args.warmdown_iters<=0:return wu
        if mwms is None:
            wds=max(args.iterations-args.warmdown_iters,0)
            wd=max((args.iterations-step)/max(args.warmdown_iters,1),0.) if wds<=step<args.iterations else 1.
        else:
            sms=ems/max(step,1);wdms=args.warmdown_iters*sms;rms=max(mwms-ems,0.)
            wd=rms/max(wdms,1e-9) if rms<=wdms else 1.
        return wu*wd
    if args.warmup_steps>0:
        isd={n:t.detach().cpu().clone() for n,t in bm.state_dict().items()}
        ios=[copy.deepcopy(o.state_dict()) for o in opts]
        model.train()
        for ws_ in range(args.warmup_steps):
            zg()
            for _ in range(gas):
                x,y=tl.next(args.train_batch_tokens,args.train_seq_len,gas)
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16):wl=model(x,y)
                (wl*gsc).backward()
            if distributed:
                for p in bm.parameters():
                    if p.grad is not None:dist.all_reduce(p.grad,op=dist.ReduceOp.AVG)
            for o in opts:o.step()
            zg()
        bm.load_state_dict(isd,strict=True)
        # Keep optimizer momentum buffers from warmup (don't restore optimizer state)
        zg();tl=DTokenLoader(args.train_files,rank,ws,dev)
    swa_s=None;swa_c=0
    ema_s=None;ema_d=0.997;ema_started=False
    ttms=0.;atms=0.;stop=None;torch.cuda.synchronize();t0=time.perf_counter();step=0
    while True:
        last=step==args.iterations or(stop is not None and step>=stop)
        sv=last or(args.val_loss_every>0 and step%args.val_loss_every==0)
        if sv:
            torch.cuda.synchronize();ttms+=1000.*(time.perf_counter()-t0)
            vl,vb=eval_val(args,model,rank,ws,dev,vt,bbl,hsl,ibl)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} t:{ttms:.0f}ms avg:{ttms/max(step,1):.2f}ms")
            torch.cuda.synchronize();t0=time.perf_counter()
        if last:
            if stop is not None and step<args.iterations:log0(f"early_stop step:{step}")
            break
        ems=ttms+1000.*(time.perf_counter()-t0);sc=lr_m(step,ems)
        if args.late_qat_threshold>0 and sc<args.late_qat_threshold and step>args.lr_warmup_steps and not CL._qat:
            CL._qat=True;log0(f"qat:on step:{step} sc:{sc:.4f}")
        zg();trl=torch.zeros((),device=dev)
        for _ in range(gas):
            x,y=tl.next(args.train_batch_tokens,args.train_seq_len,gas)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):lo=model(x,y)
            trl+=lo.detach();(lo*gsc).backward()
        trl/=gas
        fr=min(step/args.muon_momentum_warmup_steps,1.) if args.muon_momentum_warmup_steps>0 else 1.
        mm=(1-fr)*args.muon_momentum_warmup_start+fr*args.muon_momentum
        for g in omu.param_groups:g["momentum"]=mm
        for o in opts:
            for g in o.param_groups:g["lr"]=g["base_lr"]*sc
        if args.grad_clip_norm>0:torch.nn.utils.clip_grad_norm_(bm.parameters(),args.grad_clip_norm)
        omu.launch_reduce_scatters()
        if distributed:
            for p in rps:
                if p.grad is not None:dist.all_reduce(p.grad,op=dist.ReduceOp.AVG)
        otk.step();osc.step()
        if ohd:ohd.step()
        omu.step();zg()
        with torch.no_grad():
            ema_frac=args.ema_start_frac;est=int(ema_frac*(args.iterations if mwms is None else atms/(max(atms/max(step,1),1e-9))))
            if not ema_started and step>=max(est,10):
                ema_s={n:t.detach().float().clone() for n,t in bm.state_dict().items()};ema_started=True
            elif ema_started:
                for n,t in bm.state_dict().items():ema_s[n].mul_(ema_d).add_(t.detach().float(),alpha=1.-ema_d)
        step+=1;atms=ttms+1000.*(time.perf_counter()-t0)
        if args.swa_enabled and sc<0.2 and step%args.swa_every==0:
            if swa_s is None:swa_s={n:t.detach().cpu().clone() for n,t in bm.state_dict().items()};swa_c=1;log0(f"swa:start step:{step}")
            else:
                for n,t in bm.state_dict().items():swa_s[n]+=t.detach().cpu()
                swa_c+=1
        sl_=args.train_log_every>0 and(step<=10 or step%args.train_log_every==0)
        if sl_:log0(f"step:{step}/{args.iterations} loss:{trl.item():.4f} t:{atms:.0f}ms avg:{atms/step:.2f}ms")
        rc=mwms is not None and atms>=mwms
        if distributed and mwms is not None:
            rt=torch.tensor(int(rc),device=dev);dist.all_reduce(rt,op=dist.ReduceOp.MAX);rc=bool(rt.item())
        if stop is None and rc:stop=step
    log0(f"peak_mem:{torch.cuda.max_memory_allocated()//1024//1024}MiB")
    # Save raw model for comparison
    raw_sd={n:t.detach().cpu().clone() for n,t in bm.state_dict().items()}
    torch.cuda.synchronize();td=time.perf_counter()
    raw_vl,raw_vb=eval_val(args,cm,rank,ws,dev,vt,bbl,hsl,ibl)
    log0(f"raw_model val_loss:{raw_vl:.4f} val_bpb:{raw_vb:.4f} t:{1000.*(time.perf_counter()-td):.0f}ms")
    best_vb=raw_vb;best_src="raw"
    # Try EMA
    if ema_started and ema_s is not None:
        cs=bm.state_dict();avg={n:t.to(dtype=cs[n].dtype) for n,t in ema_s.items()}
        bm.load_state_dict(avg,strict=True)
        torch.cuda.synchronize();td=time.perf_counter()
        dvl,dvb=eval_val(args,cm,rank,ws,dev,vt,bbl,hsl,ibl)
        log0(f"post_ema val_loss:{dvl:.4f} val_bpb:{dvb:.4f} t:{1000.*(time.perf_counter()-td):.0f}ms")
        if dvb<best_vb:best_vb=dvb;best_src="ema"
        else:bm.load_state_dict(raw_sd,strict=True);log0("ema:worse, reverting to raw")
    else:log0("ema:not started (training too short)")
    # Try SWA
    if swa_s is not None and swa_c>1:
        swa_avg={n:(t/swa_c).to(dtype=raw_sd[n].dtype) for n,t in swa_s.items()}
        swa_sd={n:t.detach().cpu().clone() for n,t in bm.state_dict().items()}
        bm.load_state_dict(swa_avg,strict=True)
        torch.cuda.synchronize();td=time.perf_counter()
        svl,svb=eval_val(args,cm,rank,ws,dev,vt,bbl,hsl,ibl)
        log0(f"post_swa val_loss:{svl:.4f} val_bpb:{svb:.4f} (n={swa_c}) t:{1000.*(time.perf_counter()-td):.0f}ms")
        if svb<best_vb:best_vb=svb;best_src="swa";log0("swa:better, using swa")
        else:bm.load_state_dict(swa_sd,strict=True);log0(f"swa:worse ({svb:.4f}>{best_vb:.4f}), keeping {best_src}")
    log0(f"best_model:{best_src} val_bpb:{best_vb:.4f}")
    esd={k:v for k,v in bm.state_dict().items()}
    sdcpu={k:v.detach().cpu() for k,v in esd.items()}
    ubsd=_unbank(sdcpu,args.num_layers)
    # Int6 GPTQ-lite quantization
    qr,qm=mq6(ubsd,{"mlp","attn"})
    qbuf=io.BytesIO();torch.save({"w":qr,"m":qm},qbuf);qraw=qbuf.getvalue()
    qblob=lzma.compress(qraw,preset=9|lzma.PRESET_EXTREME)
    if mp:
        with open("final_model.int8.ptz","wb") as f:f.write(qblob)
        cb=len(code.encode("utf-8"));mb=len(qblob);tot=cb+mb
        log0(f"code:{cb} model:{mb} total:{tot} limit:16000000 {'OK' if tot<=16000000 else 'OVER!'}")
    if distributed:dist.barrier()
    # Roundtrip validation
    with open("final_model.int8.ptz","rb") as f:qbd=f.read()
    qs=torch.load(io.BytesIO(lzma.decompress(qbd)),map_location="cpu",weights_only=False)
    dqu=dq6(qs["w"],qs["m"],ubsd);dqs=_rebank(dqu,args.num_layers,sdcpu)
    em=GPT(args).to(dev).bfloat16()
    em.qo_bank.data=em.qo_bank.data.float();em.kv_bank.data=em.kv_bank.data.float()
    em.mlp_up_bank.data=em.mlp_up_bank.data.float();em.mlp_down_bank.data=em.mlp_down_bank.data.float()
    for m in em.modules():
        if isinstance(m,CL):m.float()
    fix_fp32(em);em.load_state_dict(dqs,strict=True)
    cem=torch.compile(em,dynamic=False,fullgraph=True) if args.torch_compile else em
    torch.cuda.synchronize();tq=time.perf_counter()
    qvl,qvb=eval_val(args,cem,rank,ws,dev,vt,bbl,hsl,ibl,esl=esl)
    log0(f"int6_roundtrip val_loss:{qvl:.4f} val_bpb:{qvb:.4f} t:{1000.*(time.perf_counter()-tq):.0f}ms")
    log0(f"int6_roundtrip_exact val_loss:{qvl:.8f} val_bpb:{qvb:.8f}")
    # Also try int8+zlib and pick best that fits
    qi8,_=quantize_state_dict_int8(ubsd)
    q8buf=io.BytesIO();torch.save(qi8,q8buf);q8raw=q8buf.getvalue()
    q8blob=lzma.compress(q8raw,preset=9|lzma.PRESET_EXTREME)
    cb=len(code.encode("utf-8"))
    i6tot=cb+len(qblob);i8tot=cb+len(q8blob)
    log0(f"int6_zlib code:{cb} model:{len(qblob)} total:{i6tot} limit:16000000 {'OK' if i6tot<=16000000 else 'OVER!'}")
    log0(f"int8_zlib code:{cb} model:{len(q8blob)} total:{i8tot} limit:16000000 {'OK' if i8tot<=16000000 else 'OVER!'}")
    # Pick: prefer fit within 16MB, then prefer smaller
    if i6tot<=16000000 and i8tot<=16000000:
        use_blob=qblob if len(qblob)<=len(q8blob) else q8blob
        log0(f"both fit, using {'int6' if len(qblob)<=len(q8blob) else 'int8'} (smaller)")
    elif i6tot<=16000000:use_blob=qblob;log0("using int6 (int8 over limit)")
    elif i8tot<=16000000:use_blob=q8blob;log0("using int8 (int6 over limit)")
    else:use_blob=qblob if len(qblob)<=len(q8blob) else q8blob;log0("WARNING: both over 16MB!")
    if mp:
        with open("final_model.int8.ptz","wb") as f:f.write(use_blob)
    if distributed:dist.barrier()
    # Sliding window eval
    swsl=esl
    if args.eval_stride>0 and args.eval_stride<swsl:
        torch.cuda.synchronize();tsw=time.perf_counter()
        swvl,swvb=eval_slide(args,em,rank,ws,dev,vt,bbl,hsl,ibl,stride=args.eval_stride,esl=swsl)
        log0(f"sliding val_loss:{swvl:.4f} val_bpb:{swvb:.4f} stride:{args.eval_stride} t:{1000.*(time.perf_counter()-tsw):.0f}ms")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{swvl:.8f} val_bpb:{swvb:.8f}")
    # Legal TTT
    if args.ttt_enabled:
        torch.cuda.synchronize();tt0=time.perf_counter()
        tvl,tvb=eval_ttt(args,em,rank,ws,dev,vt,bbl,hsl,ibl,stride=args.eval_stride,log0=log0)
        log0(f"legal_ttt val_loss:{tvl:.4f} val_bpb:{tvb:.4f} t:{1000.*(time.perf_counter()-tt0):.0f}ms")
        log0(f"legal_ttt_exact val_loss:{tvl:.8f} val_bpb:{tvb:.8f}")
    if distributed:dist.destroy_process_group()

if __name__=="__main__":
    main()
