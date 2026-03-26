from __future__ import annotations
import copy
import glob
import io
import lzma
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path
try:
    import zstandard
except ImportError:
    pass
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
try:
    from flash_attn_interface import flash_attn_func as _fa
    _ATTN = 'fa3'
except ImportError:
    try:
        from flash_attn import flash_attn_func as _fa
        _ATTN = 'fa2'
    except ImportError:
        _fa = None
        _ATTN = 'sdpa'

# =============================================================================
# hyperparameters
# =============================================================================

class Args:
    data_path = os.environ.get('DATA_PATH', './data/datasets/fineweb10B_sp1024')
    train_files = os.path.join(data_path, 'fineweb_train_*.bin')
    val_files = os.path.join(data_path, 'fineweb_val_*.bin')
    tokenizer_path = os.environ.get('TOKENIZER_PATH', './data/tokenizers/fineweb_1024_bpe.model')
    run_id = os.environ.get('RUN_ID', str(uuid.uuid4()))
    seed = int(os.environ.get('SEED', 1337))
    val_batch_size = int(os.environ.get('VAL_BATCH_SIZE', 524_288))
    val_loss_every = int(os.environ.get('VAL_LOSS_EVERY', 4000))
    train_log_every = int(os.environ.get('TRAIN_LOG_EVERY', 500))
    iterations = int(os.environ.get('ITERATIONS', 200_000))
    warmdown_iters = int(os.environ.get('WARMDOWN_ITERS', 5000))
    warmup_steps = int(os.environ.get('WARMUP_STEPS', 20))
    lr_warmup_steps = int(os.environ.get('LR_WARMUP_STEPS', 200))
    train_batch_tokens = int(os.environ.get('TRAIN_BATCH_TOKENS', 786_432))
    train_seq_len = int(os.environ.get('TRAIN_SEQ_LEN', 2048))
    eval_seq_len = int(os.environ.get('EVAL_SEQ_LEN', 2048))
    max_wallclock_seconds = float(os.environ.get('MAX_WALLCLOCK_SECONDS', 0.0))
    # architecture
    vocab_size = int(os.environ.get('VOCAB_SIZE', 1024))
    num_layers = int(os.environ.get('NUM_LAYERS', 20))
    model_dim = int(os.environ.get('MODEL_DIM', 512))
    num_heads = int(os.environ.get('NUM_HEADS', 8))
    num_kv_heads = int(os.environ.get('NUM_KV_HEADS', 4))
    mlp_mult = float(os.environ.get('MLP_MULT', 3.0))
    logit_softcap = float(os.environ.get('LOGIT_SOFTCAP', 30.0))
    rope_base = float(os.environ.get('ROPE_BASE', 10000.0))
    rope_dims = int(os.environ.get('ROPE_DIMS', 16))
    qk_gain_init = float(os.environ.get('QK_GAIN_INIT', 1.5))
    bigram_vocab_size = int(os.environ.get('BIGRAM_VOCAB_SIZE', 2048))
    bigram_dim = int(os.environ.get('BIGRAM_DIM', 128))
    xsa_last_n = int(os.environ.get('XSA_LAST_N', 4))
    ln_scale = bool(int(os.environ.get('LN_SCALE', '1')))
    tied_embed_init_std = float(os.environ.get('TIED_EMBED_INIT_STD', 0.005))
    # jepa
    d_latent = int(os.environ.get('D_LATENT', 256))
    jepa_weight = float(os.environ.get('JEPA_WEIGHT', 0.3))
    vicreg_var_weight = float(os.environ.get('VICREG_VAR_WEIGHT', 1.0))
    vicreg_cov_weight = float(os.environ.get('VICREG_COV_WEIGHT', 0.04))
    target_ema_decay = float(os.environ.get('TARGET_EMA_DECAY', 0.9995))
    langevin_steps = int(os.environ.get('LANGEVIN_STEPS', 3))
    langevin_alpha = float(os.environ.get('LANGEVIN_ALPHA', 0.5))
    # optimizer
    tied_embed_lr = float(os.environ.get('TIED_EMBED_LR', 0.035))
    matrix_lr = float(os.environ.get('MATRIX_LR', 0.025))
    scalar_lr = float(os.environ.get('SCALAR_LR', 0.025))
    muon_momentum = float(os.environ.get('MUON_MOMENTUM', 0.99))
    muon_backend_steps = int(os.environ.get('MUON_BACKEND_STEPS', 5))
    muon_momentum_warmup_start = float(os.environ.get('MUON_MOMENTUM_WARMUP_START', 0.92))
    muon_momentum_warmup_steps = int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS', 1500))
    beta1 = float(os.environ.get('BETA1', 0.9))
    beta2 = float(os.environ.get('BETA2', 0.95))
    adam_eps = float(os.environ.get('ADAM_EPS', 1e-8))
    grad_clip_norm = float(os.environ.get('GRAD_CLIP_NORM', 0.3))
    muon_wd = float(os.environ.get('MUON_WD', 0.04))
    adam_wd = float(os.environ.get('ADAM_WD', 0.04))
    eval_stride = int(os.environ.get('EVAL_STRIDE', 64))
    swa_enabled = bool(int(os.environ.get('SWA_ENABLED', '1')))
    swa_every = int(os.environ.get('SWA_EVERY', 50))
    ppmd_enabled = bool(int(os.environ.get('PPMD_ENABLED', '1')))
    ttt_enabled = bool(int(os.environ.get('TTT_ENABLED', '0')))
    ttt_lr = float(os.environ.get('TTT_LR', 0.002))
    ttt_epochs = int(os.environ.get('TTT_EPOCHS', 3))
    ttt_chunk_tokens = int(os.environ.get('TTT_CHUNK_TOKENS', 32768))
    ttt_momentum = float(os.environ.get('TTT_MOMENTUM', 0.9))
    ttt_batch_seqs = int(os.environ.get('TTT_BATCH_SEQS', 32))
    ttt_grad_clip = float(os.environ.get('TTT_GRAD_CLIP', 1.0))

# =============================================================================
# muon optimizer
# =============================================================================

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    was_2d = G.ndim == 2
    if was_2d: G = G.unsqueeze(0)
    X = G.bfloat16()
    tr = X.size(-2) > X.size(-1)
    if tr: X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT; B = b * A + c * (A @ A); X = a * X + B @ X
    if tr: X = X.mT
    if was_2d: X = X.squeeze(0)
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay))
        self._built = False
    def _build(self):
        self._dist = dist.is_available() and dist.is_initialized()
        ws = dist.get_world_size() if self._dist else 1
        self._meta = []
        for g in self.param_groups:
            for p in g['params']:
                B = p.shape[0]; pB = ((B+ws-1)//ws)*ws; sB = pB//ws; d = p.device
                self._meta.append({'p':p,'B':B,'pg':torch.zeros(pB,*p.shape[1:],device=d,dtype=torch.bfloat16),
                    'sh':torch.zeros(sB,*p.shape[1:],device=d,dtype=torch.bfloat16),
                    'sm':torch.zeros(sB,*p.shape[1:],device=d,dtype=torch.bfloat16),
                    'fu':torch.zeros(pB,*p.shape[1:],device=d,dtype=torch.bfloat16),
                    'sc':max(1,p.shape[-2]/p.shape[-1])**0.5})
        self._meta.sort(key=lambda m:-m['p'].numel()); self._built = True
    def launch_reduce_scatters(self):
        if not self._built: self._build()
        if not self._dist: return
        self._rs = []
        for m in self._meta:
            if m['p'].grad is None: self._rs.append(None); continue
            m['pg'][:m['B']].copy_(m['p'].grad.bfloat16())
            if m['pg'].shape[0]>m['B']: m['pg'][m['B']:].zero_()
            self._rs.append(dist.reduce_scatter_tensor(m['sh'],m['pg'],op=dist.ReduceOp.AVG,async_op=True))
    @torch.no_grad()
    def step(self, closure=None):
        if not self._built: self._build()
        for g in self.param_groups:
            lr,mom,bs,nest,wd = g['lr'],g['momentum'],g['backend_steps'],g['nesterov'],g.get('weight_decay',0.0)
            pag,pm = None,None
            sh = self._dist and hasattr(self,'_rs')
            for i,m in enumerate(self._meta):
                p = m['p']
                if p.grad is None: continue
                if pag: pag.wait(); pp=pm['p']; u=pm['fu'][:pm['B']]; (pp.data.mul_(1-lr*wd) if wd>0 else None); pp.add_(u.to(pp.dtype),alpha=-lr*pm['sc'])
                if sh and self._rs[i] is not None:
                    self._rs[i].wait(); g_,buf = m['sh'],m['sm']
                else:
                    g_ = p.grad.bfloat16(); st = self.state[p]
                    if 'mb' not in st: st['mb'] = torch.zeros_like(g_)
                    buf = st['mb']
                buf.mul_(mom).add_(g_)
                upd = zeropower_via_newtonschulz5(g_.add(buf,alpha=mom) if nest else buf, steps=bs)
                if sh: pag=dist.all_gather_into_tensor(m['fu'],upd,async_op=True); pm=m
                else: (p.data.mul_(1-lr*wd) if wd>0 else None); p.add_(upd.to(p.dtype),alpha=-lr*m['sc'])
            if pag: pag.wait(); pp=pm['p']; u=pm['fu'][:pm['B']]; (pp.data.mul_(1-lr*wd) if wd>0 else None); pp.add_(u.to(pp.dtype),alpha=-lr*pm['sc'])
            if hasattr(self,'_rs'): del self._rs

# =============================================================================
# data / tokenizer / eval
# =============================================================================

def build_luts(sp, vs, dev):
    n = max(int(sp.vocab_size()), vs)
    bb,ls,bt = np.zeros(n,dtype=np.int16), np.zeros(n,dtype=np.bool_), np.ones(n,dtype=np.bool_)
    for i in range(int(sp.vocab_size())):
        if sp.is_control(i) or sp.is_unknown(i) or sp.is_unused(i): continue
        bt[i]=False
        if sp.is_byte(i): bb[i]=1; continue
        p=sp.id_to_piece(i)
        if p.startswith('\u2581'): ls[i]=True; p=p[1:]
        bb[i]=len(p.encode('utf-8'))
    return tuple(torch.tensor(a,device=dev) for a in [bb,ls,bt])

def load_shard(f):
    h=np.fromfile(f,dtype='<i4',count=256)
    return torch.from_numpy(np.fromfile(f,dtype='<u2',count=int(h[2]),offset=256*4).astype(np.uint16,copy=False))

def load_val(pattern,sl):
    toks=torch.cat([load_shard(Path(p)) for p in sorted(glob.glob(pattern))]).contiguous()
    u=((toks.numel()-1)//sl)*sl; return toks[:u+1]

class TokenStream:
    def __init__(s,pat):
        s.files=[Path(p) for p in sorted(glob.glob(pat))]; s.fi=s.pos=0; s.tok=load_shard(s.files[0])
    def take(s,n):
        ch,rem=[],n
        while rem>0:
            a=s.tok.numel()-s.pos
            if a<=0: s.fi=(s.fi+1)%len(s.files); s.tok=load_shard(s.files[s.fi]); s.pos=0; continue
            k=min(rem,a); ch.append(s.tok[s.pos:s.pos+k]); s.pos+=k; rem-=k
        return ch[0] if len(ch)==1 else torch.cat(ch)

class DLoader:
    def __init__(s,pat,rk,ws,dev): s.rk,s.ws,s.dev=rk,ws,dev; s.st=TokenStream(pat)
    def next_batch(s,gt,sl,ga):
        lt=gt//(s.ws*ga)+1; ch=s.st.take(lt*s.ws); i=s.rk*lt; c=ch[i:i+lt].to(torch.int64)
        return c[:-1].reshape(-1,sl).to(s.dev,non_blocking=True), c[1:].reshape(-1,sl).to(s.dev,non_blocking=True)

def eval_val(args,model,rk,ws,dev,ga,vtok,luts,esl=None):
    bb,ls,bt=luts; sl=esl or args.train_seq_len
    lbs=args.val_batch_size//(ws*ga*sl); ts=(vtok.numel()-1)//sl
    ss,se=(ts*rk)//ws,(ts*(rk+1))//ws
    ls_,tc,bc=(torch.zeros((),device=dev,dtype=torch.float64) for _ in range(3))
    model.eval()
    with torch.inference_mode():
        for bs in range(ss,se,lbs):
            be=min(bs+lbs,se); loc=vtok[bs*sl:be*sl+1].to(device=dev,dtype=torch.int64,non_blocking=True)
            x,y=loc[:-1].reshape(-1,sl),loc[1:].reshape(-1,sl)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                bl=model(x,y).detach()
            n=float(y.numel()); ls_+=bl.to(torch.float64)*n; tc+=n
            tb_=bb[y.reshape(-1)].to(torch.int16)+(ls[y.reshape(-1)]&~bt[x.reshape(-1)]).to(torch.int16)
            bc+=tb_.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in [ls_,tc,bc]: dist.all_reduce(t,op=dist.ReduceOp.SUM)
    vl=(ls_/tc).item(); model.train()
    return vl, vl/math.log(2)*tc.item()/bc.item()

# =============================================================================
# vicreg
# =============================================================================

def vicreg_loss(z: Tensor, var_weight: float = 1.0, cov_weight: float = 0.04) -> tuple[Tensor, dict]:
    """variance + covariance regularization on z_tgt to prevent target encoder collapse."""
    z_flat = z.reshape(-1, z.size(-1))
    std = z_flat.std(dim=0)
    var_loss = F.relu(1.0 - std).mean()
    N, D = z_flat.shape
    zc = z_flat - z_flat.mean(dim=0)
    cov = (zc.T @ zc) / max(N - 1, 1)
    cov_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / D
    total = var_weight * var_loss + cov_weight * cov_loss
    return total, {'var': var_loss.item(), 'cov': cov_loss.item(),
                   'z_std_mean': std.mean().item(), 'z_std_min': std.min().item()}

# =============================================================================
# model building blocks
# =============================================================================

class RMSNorm(nn.Module):
    def forward(self, x): return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def forward(self, x): return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

CTRL_PATS = tuple(p for p in 'attn_scale,mlp_scale,resid_mix,skip_weight,q_gain,bigram.scale,smear'.split(',') if p)

def restore_fp32(mod):
    with torch.no_grad():
        for n,p in mod.named_parameters():
            if (p.ndim<2 or any(pat in n for pat in CTRL_PATS)) and p.dtype!=torch.float32: p.data=p.data.float()

class BigramHash(nn.Module):
    def __init__(s,bvs,bd,md):
        super().__init__(); s.bvs=bvs
        s.emb=nn.Embedding(bvs,bd); nn.init.zeros_(s.emb.weight)
        s.proj=CastedLinear(bd,md,bias=False) if bd!=md else None
        if s.proj: nn.init.zeros_(s.proj.weight)
        s.scale=nn.Parameter(torch.tensor(0.05,dtype=torch.float32))
    def forward(s,ids):
        t=ids.to(torch.int32); mod=s.bvs-1; o=torch.empty_like(t); o[...,0]=mod
        o[...,1:]=torch.bitwise_xor(36313*t[...,1:],27191*t[...,:-1])%mod
        h=s.emb(o.long()); h=s.proj(h) if s.proj else h
        return h*s.scale.to(h.dtype)

class Rotary(nn.Module):
    def __init__(s,d,base=10000.,tsl=1024,rd=0):
        super().__init__(); s.d,s.base,s.tsl,s.rd=d,base,tsl,rd if rd>0 else d
        s.register_buffer('inv',1./(base**(torch.arange(0,s.rd,2,dtype=torch.float32)/s.rd)),persistent=False)
        s._c=(0,None,None)
    def forward(s,T,dev,dt):
        if s._c[0]!=T or s._c[1] is None or s._c[1].device!=dev:
            iv=s.inv.to(dev) if T<=s.tsl else 1./(s.base*(T/s.tsl)**(s.rd/(s.rd-2))**(torch.arange(0,s.rd,2,dtype=torch.float32,device=dev)/s.rd))
            f=torch.outer(torch.arange(T,device=dev,dtype=iv.dtype),iv)
            s._c=(T,f.cos()[None,:,None,:].to(dt),f.sin()[None,:,None,:].to(dt))
        return s._c[1],s._c[2]

def rope(x,cos,sin,rd=0):
    if rd>0 and rd<x.size(-1):
        xr,xp=x[...,:rd],x[...,rd:]; h=rd//2; x1,x2=xr[...,:h],xr[...,h:]
        return torch.cat((torch.cat((x1*cos+x2*sin,x1*(-sin)+x2*cos),-1),xp),-1)
    h=x.size(-1)//2; x1,x2=x[...,:h],x[...,h:]
    return torch.cat((x1*cos+x2*sin,x1*(-sin)+x2*cos),-1)

class SmearGate(nn.Module):
    def __init__(s,d): super().__init__(); s.gate=nn.Parameter(torch.zeros(d,dtype=torch.float32))
    def forward(s,x):
        g=torch.sigmoid(s.gate.to(x.dtype))[None,None,:]
        return (1-g)*x+g*torch.cat([torch.zeros_like(x[:,:1]),x[:,:-1]],1)

class Attn(nn.Module):
    def __init__(s,d,nh,nkv,rb,qgi,rd=0):
        super().__init__(); s.nh,s.nkv,s.hd=nh,nkv,d//nh; s.rd=rd
        s.qg=nn.Parameter(torch.full((nh,),qgi,dtype=torch.float32))
        s.rot=Rotary(s.hd,base=rb,tsl=1024,rd=rd); s.xsa=False
    def forward(s,x,qw,kw,vw,ow):
        B,T,D=x.shape
        q=F.linear(x,qw.to(x.dtype)).reshape(B,T,s.nh,s.hd)
        k=F.linear(x,kw.to(x.dtype)).reshape(B,T,s.nkv,s.hd)
        v=F.linear(x,vw.to(x.dtype)).reshape(B,T,s.nkv,s.hd)
        q=F.rms_norm(q,(q.size(-1),)); k=F.rms_norm(k,(k.size(-1),))
        c,sn=s.rot(T,x.device,q.dtype)
        q=rope(q,c,sn,s.rd)*s.qg.to(q.dtype)[None,None,:,None]
        k=rope(k,c,sn,s.rd)
        if _fa: y=_fa(q,k,v,causal=True)
        else:
            q2,k2,v2=q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)
            if s.nkv<s.nh: r=s.nh//s.nkv; k2=k2.repeat_interleave(r,1); v2=v2.repeat_interleave(r,1)
            y=F.scaled_dot_product_attention(q2,k2,v2,is_causal=True).transpose(1,2)
        if s.xsa:
            _B,_T,_H,_hd=y.shape; Hk=v.size(-2); g=_H//Hk
            yg=y.reshape(_B,_T,Hk,g,_hd); vn=F.normalize(v,dim=-1).unsqueeze(-2)
            y=(yg-(yg*vn).sum(-1,keepdim=True)*vn).reshape(_B,_T,_H,_hd)
        return F.linear(y.reshape(B,T,D),ow.to(x.dtype))

class Block(nn.Module):
    def __init__(s,d,nh,nkv,rb,qgi,li=0,lns=False,rd=0):
        super().__init__()
        s.an,s.mn=RMSNorm(),RMSNorm()
        s.attn=Attn(d,nh,nkv,rb,qgi,rd)
        s.as_=nn.Parameter(torch.ones(d,dtype=torch.float32))
        s.ms=nn.Parameter(torch.ones(d,dtype=torch.float32))
        s.rm=nn.Parameter(torch.stack([torch.ones(d),torch.zeros(d)]).float())
        s.lsf=1./math.sqrt(li+1) if lns else 1.
    def forward(s,x,x0,qw,kw,vw,ow,uw,dw):
        m=s.rm.to(x.dtype); xi=m[0][None,None,:]*x+m[1][None,None,:]*x0
        xi=xi+s.as_.to(x.dtype)[None,None,:]*s.attn(s.an(xi)*s.lsf,qw,kw,vw,ow)
        return xi+s.ms.to(x.dtype)[None,None,:]*F.linear(F.leaky_relu(F.linear(s.mn(xi)*s.lsf,uw.to(x.dtype)),0.5).square(),dw.to(x.dtype))

# =============================================================================
# jepa language model
# =============================================================================

class JEPA_LM(nn.Module):
    """transformer with JEPA auxiliary loss. only the context encoder is saved in the artifact;
    target encoder, predictor, and projections are training-only."""

    def __init__(s, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, logit_softcap, rope_base, rope_dims, qk_gain_init,
                 d_latent, bigram_vocab_size=0, bigram_dim=128, xsa_last_n=0,
                 ln_scale=False, tied_embed_init_std=0.005):
        super().__init__()
        s.num_layers, s.model_dim, s.logit_softcap = num_layers, model_dim, logit_softcap
        s.d_latent = d_latent
        hd = model_dim // num_heads; kvd = num_kv_heads * hd; mlpd = int(mlp_mult * model_dim)
        n = num_layers

        s.tok_emb = nn.Embedding(vocab_size, model_dim)
        s.bigram = BigramHash(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        s.smear = SmearGate(model_dim)

        # parameter banks (contiguous 3D tensors for batched muon)
        s.qo_bank = nn.Parameter(torch.empty(2*n, model_dim, model_dim))
        s.kv_bank = nn.Parameter(torch.empty(2*n, kvd, model_dim))
        s.mlp_up_bank = nn.Parameter(torch.empty(n, mlpd, model_dim))
        s.mlp_down_bank = nn.Parameter(torch.empty(n, model_dim, mlpd))
        s.blocks = nn.ModuleList([Block(model_dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                        li=i, lns=ln_scale, rd=rope_dims) for i in range(n)])
        if xsa_last_n > 0:
            for i in range(max(0,n-xsa_last_n),n): s.blocks[i].attn.xsa = True

        s.n_enc = n // 2; s.n_skip = min(s.n_enc, n - s.n_enc)
        s.skip_w = nn.Parameter(torch.ones(s.n_skip, model_dim, dtype=torch.float32))
        s.final_norm = RMSNorm()

        # jepa components (training-only)
        s.predictor = nn.Sequential(
            nn.Linear(d_latent, d_latent, bias=False),
            nn.GELU(),
            nn.Linear(d_latent, d_latent, bias=False),
        )
        s.ctx_proj = nn.Linear(model_dim, d_latent, bias=False)
        s.tgt_proj = nn.Linear(model_dim, d_latent, bias=False)

        s._init(tied_embed_init_std)

    def _init(s, std):
        nn.init.normal_(s.tok_emb.weight, mean=0., std=std)
        n = s.num_layers; ps = 1. / math.sqrt(2 * n)
        for i in range(n):
            nn.init.orthogonal_(s.qo_bank.data[i]); nn.init.zeros_(s.qo_bank.data[n+i])
            nn.init.orthogonal_(s.kv_bank.data[i]); nn.init.orthogonal_(s.kv_bank.data[n+i])
            nn.init.orthogonal_(s.mlp_up_bank.data[i]); nn.init.zeros_(s.mlp_down_bank.data[i])
            s.qo_bank.data[n+i].mul_(ps); s.mlp_down_bank.data[i].mul_(ps)
        for m in s.predictor.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=0.02)
        nn.init.orthogonal_(s.ctx_proj.weight); nn.init.orthogonal_(s.tgt_proj.weight)

    def _encode(s, input_ids, banks=None):
        """Run causal transformer encoder. If banks=None, use self's banks."""
        qo = s.qo_bank if banks is None else banks['qo']
        kv = s.kv_bank if banks is None else banks['kv']
        up = s.mlp_up_bank if banks is None else banks['up']
        dn = s.mlp_down_bank if banks is None else banks['dn']
        n = s.num_layers
        x = s.tok_emb(input_ids)
        if s.bigram is not None: x = x + s.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = s.smear(x)
        x0, skips = x, []
        for i in range(n):
            x = s.blocks[i](x, x0, qo[i], kv[i], kv[n+i], qo[n+i], up[i], dn[i])
            if i < s.n_enc: skips.append(x)
            elif skips:
                si = i - s.n_enc
                if si < s.n_skip: x = x + s.skip_w[si].to(x.dtype)[None,None,:] * skips.pop()
        return s.final_norm(x)

    def forward(s, input_ids, target_ids):
        h = s._encode(input_ids)
        logits = F.linear(h.reshape(-1, h.size(-1)), s.tok_emb.weight)
        logits = s.logit_softcap * torch.tanh(logits / s.logit_softcap)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction='mean')

    def forward_jepa(s, input_ids, target_ids, tgt_banks):
        """returns ce_loss, jepa_loss, vicreg_loss, stats_dict."""
        h_ctx = s._encode(input_ids)

        logits = F.linear(h_ctx.reshape(-1, h_ctx.size(-1)), s.tok_emb.weight)
        logits = s.logit_softcap * torch.tanh(logits / s.logit_softcap)
        ce_loss = F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction='mean')

        with torch.no_grad():
            h_tgt = s._encode(input_ids, banks=tgt_banks)

        z_ctx = s.ctx_proj(h_ctx)
        z_tgt = s.tgt_proj(h_tgt)

        z_pred = s.predictor(z_ctx[:, :-1, :])
        z_tgt_next = z_tgt[:, 1:, :]

        jepa_loss = F.mse_loss(z_pred, z_tgt_next.detach())
        vic_loss, vic_stats = vicreg_loss(z_tgt)

        return ce_loss, jepa_loss, vic_loss, vic_stats

    def forward_logits(s, input_ids):
        h = s._encode(input_ids)
        logits = F.linear(h, s.tok_emb.weight)
        return s.logit_softcap * torch.tanh(logits / s.logit_softcap)

    def context_encoder_state(s):
        """excludes jepa-only params (predictor, projections) from artifact."""
        exclude = {'predictor', 'ctx_proj', 'tgt_proj'}
        return {k: v for k, v in s.state_dict().items()
                if not any(k.startswith(e) for e in exclude)}

# =============================================================================
# ppmd n-gram cache
# =============================================================================

_PPM_O = list(range(2, 8)); _PPM_B = 1 << 22; _PPM_M = np.uint64(_PPM_B - 1)
_PPM_P = np.array([np.uint64(p) for p in [36313,27191,51647,81929,131071,175447,209591]], dtype=np.uint64)

def _phc(v,pos,cw):
    h=np.zeros(len(pos),dtype=np.uint64)
    for k in range(cw): h^=v[pos-(cw-k)].astype(np.uint64)*_PPM_P[k%len(_PPM_P)]
    return h&_PPM_M

def _phf(ch,tgt,cw): return (ch^(tgt.astype(np.uint64)*_PPM_P[cw%len(_PPM_P)]))&_PPM_M

def ppmd_mix(mnll,vnp,gp,ct,ft,ent=None):
    mp=np.exp(-mnll); tt=vnp[gp]; bp=np.full(len(mnll),-1.)
    for oi in range(len(_PPM_O)-1,-1,-1):
        o=_PPM_O[oi]; cw=o-1; v=gp>=cw
        if not v.any(): continue
        vi=np.where(v)[0]; ch=_phc(vnp,gp[vi],cw); fh=_phf(ch,tt[vi],cw)
        cc=ct[oi][ch.astype(np.int64)].astype(np.float64); fc=ft[oi][fh.astype(np.int64)].astype(np.float64)
        nf=(cc>=2.)&(bp[vi]<0)
        if nf.any(): fi=vi[nf]; bp[fi]=np.clip((2*fc[nf]-1)/(2*cc[nf]).clip(1),0,1)
    hng=bp>=0
    if hng.any() and ent is not None:
        a=0.05+0.55/(1+np.exp(-2*(ent[hng]-4))); mp[hng]=(1-a)*mp[hng]+a*bp[hng]
    elif hng.any(): mp[hng]=0.6*mp[hng]+0.4*bp[hng]
    return -np.log(mp.clip(1e-12,1))

def ppmd_upd(vnp,gp,ct,ft):
    tt=vnp[gp]
    for oi,o in enumerate(_PPM_O):
        cw=o-1; v=gp>=cw
        if not v.any(): continue
        vi=np.where(v)[0]; ch=_phc(vnp,gp[vi],cw); fh=_phf(ch,tt[vi],cw)
        np.add.at(ct[oi],ch.astype(np.int64),1); np.add.at(ft[oi],fh.astype(np.int64),1)

def eval_sliding(args,mdl,rk,ws,dev,vtok,luts,stride,bsq=32,esl=None):
    bb,ls_,bt=luts; sl=esl or args.train_seq_len; tot=vtok.numel()-1
    wl=[w for w in range(0,tot,stride) if min(w+sl,tot)-w>=1]
    ms,me=(len(wl)*rk)//ws,(len(wl)*(rk+1))//ws; myw=wl[ms:me]
    ls,tc,bc=(torch.zeros((),device=dev,dtype=torch.float64) for _ in range(3))
    ppmd=getattr(args,'ppmd_enabled',True)
    vnp=vtok.numpy().astype(np.int64) if ppmd else None
    ct=[np.zeros(_PPM_B,dtype=np.uint32) for _ in _PPM_O] if ppmd else None
    ft=[np.zeros(_PPM_B,dtype=np.uint32) for _ in _PPM_O] if ppmd else None
    mdl.eval(); cfwd=torch.compile(mdl.forward_logits,dynamic=False,fullgraph=True)
    with torch.inference_mode():
        for bi in range(0,len(myw),bsq):
            bws=myw[bi:bi+bsq]; bsz=len(bws)
            xb=torch.zeros(bsz,sl,dtype=torch.int64,device=dev); yb=torch.zeros(bsz,sl,dtype=torch.int64,device=dev)
            wls=[]
            for i,w in enumerate(bws):
                e=min(w+sl,tot); wlen=e-w; wls.append(wlen)
                c=vtok[w:e+1].to(dtype=torch.int64,device=dev); xb[i,:wlen]=c[:-1]; yb[i,:wlen]=c[1:]
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16): logits=cfwd(xb)
            nll=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),yb.reshape(-1),reduction='none').reshape(bsz,sl)
            for i,w in enumerate(bws):
                wlen=wls[i]; s=0 if w==0 else max(wlen-stride,0); snll=nll[i,s:wlen].to(torch.float64)
                if ppmd:
                    with torch.no_grad():
                        lp=F.log_softmax(logits[i,s:wlen].float(),-1); se=-(lp.exp()*lp).sum(-1).cpu().numpy()
                    gp=np.arange(w+s+1,w+wlen+1,dtype=np.int64)
                    snll=torch.from_numpy(ppmd_mix(snll.cpu().numpy(),vnp,gp,ct,ft,se)).to(torch.float64,device=dev)
                    ppmd_upd(vnp,gp,ct,ft)
                ls+=snll.sum(); tc+=float(wlen-s)
                tb=bb[yb[i,s:wlen]].to(torch.float64)+(ls_[yb[i,s:wlen]]&~bt[xb[i,s:wlen]]).to(torch.float64)
                bc+=tb.sum()
    if dist.is_available() and dist.is_initialized():
        for t in [ls,tc,bc]: dist.all_reduce(t,op=dist.ReduceOp.SUM)
    vl=(ls/tc).item(); mdl.train()
    return vl, vl/math.log(2)*tc.item()/bc.item()

# =============================================================================
# int6 quantization
# =============================================================================

def qf_tensor(t):
    t32=t.float()
    if t32.ndim==2:
        ca=torch.quantile(t32.abs(),0.9999984,dim=1) if t32.numel() else torch.empty(t32.shape[0],dtype=torch.float32)
        s=(ca/127).clamp_min(1/127).to(torch.float16)
        return torch.clamp(torch.round(torch.clamp(t32,-ca[:,None],ca[:,None])/s.float()[:,None]),-127,127).to(torch.int8).contiguous(),s.contiguous()
    ca=float(torch.quantile(t32.abs().flatten(),0.9999984).item()) if t32.numel() else 0.
    s=torch.tensor(ca/127 if ca>0 else 1.,dtype=torch.float32)
    return torch.clamp(torch.round(torch.clamp(t32,-ca,ca)/s),-127,127).to(torch.int8).contiguous(),s

def q6_row(t,cr=31):
    t32=t.float()
    if t32.ndim==2:
        bq,bs,be=None,None,float('inf')
        for p in [.999,.9995,.9999,.99999,1.]:
            rc=torch.quantile(t32.abs(),p,dim=1) if p<1 else t32.abs().amax(dim=1)
            s=(rc/cr).clamp_min(1/cr).to(torch.float16)
            q=torch.clamp(torch.round(t32/s.float()[:,None]),-cr,cr).to(torch.int8)
            e=(t32-q.float()*s.float()[:,None]).pow(2).mean().item()
            if e<be: bq,bs,be=q,s,e
        return bq,bs
    am=t32.abs().max().item(); s=torch.tensor(am/cr if am>0 else 1.,dtype=torch.float16)
    return torch.clamp(torch.round(t32/s.float()),-cr,cr).to(torch.int8),s

def _cls(n):
    if 'tok_emb' in n: return 'embed'
    if '.mlp' in n or 'mlp_up' in n or 'mlp_down' in n or '.m.' in n: return 'mlp'
    if '.attn' in n or 'qo_bank' in n or 'kv_bank' in n or '.a.' in n: return 'attn'
    return 'other'

def _unbank(sd,n):
    out={}
    for k,t in sd.items():
        if k=='qo_bank':
            for i in range(n): out[f'b.{i}.a.q']=t[i]; out[f'b.{i}.a.o']=t[n+i]
        elif k=='kv_bank':
            for i in range(n): out[f'b.{i}.a.k']=t[i]; out[f'b.{i}.a.v']=t[n+i]
        elif k=='mlp_up_bank':
            for i in range(n): out[f'b.{i}.m.u']=t[i]
        elif k=='mlp_down_bank':
            for i in range(n): out[f'b.{i}.m.d']=t[i]
        else: out[k]=t
    return out

def _rebank(sd,n,tpl):
    out,consumed={},set()
    bks={'qo_bank':([None]*(2*n),[f'b.{i}.a.q' for i in range(n)]+[f'b.{i}.a.o' for i in range(n)]),
         'kv_bank':([None]*(2*n),[f'b.{i}.a.k' for i in range(n)]+[f'b.{i}.a.v' for i in range(n)]),
         'mlp_up_bank':([None]*n,[f'b.{i}.m.u' for i in range(n)]),
         'mlp_down_bank':([None]*n,[f'b.{i}.m.d' for i in range(n)])}
    for bn,(sl,ks) in bks.items():
        for j,k in enumerate(ks):
            if k in sd: sl[j]=sd[k]; consumed.add(k)
        out[bn]=torch.stack(sl).to(tpl[bn].dtype)
    for k,v in sd.items():
        if k not in consumed: out[k]=v
    return out

def mq6(sd,cats):
    res,meta={},{}
    for k,t in sd.items():
        t=t.detach().cpu().contiguous(); cat=_cls(k)
        if not t.is_floating_point() or t.numel()<=65536:
            res[k]=t.to(torch.float16) if t.is_floating_point() else t; meta[k]='pt'; continue
        if any(p in k for p in CTRL_PATS): res[k]=t.float(); meta[k]='ctrl'; continue
        if cat in cats and t.ndim>=1:
            q,s=q6_row(t); res[k+'.q'],res[k+'.s']=q,s; meta[k]={'t':'i6'}
        else:
            q,s=qf_tensor(t); res[k+'.q'],res[k+'.s']=q,s; meta[k]={'t':'i8'}
    return res,meta

def dq6(res,meta,tpl):
    out={}
    for k,orig in tpl.items():
        info=meta.get(k)
        if info is None: continue
        if info in ('pt','ctrl'):
            t=res[k];
            if t.dtype==torch.float16 and orig.dtype in (torch.float32,torch.bfloat16): t=t.to(orig.dtype)
            out[k]=t; continue
        q,s=res[k+'.q'],res[k+'.s']
        out[k]=(q.float()*(s.float().view(q.shape[0],*([1]*(q.ndim-1))) if s.ndim>0 else float(s.item()))).to(orig.dtype)
    return out

# =============================================================================
# training
# =============================================================================

def main():
    code = Path(__file__).read_text(encoding='utf-8')
    args = Args()
    distributed = 'RANK' in os.environ
    rk = int(os.environ.get('RANK', '0'))
    ws = int(os.environ.get('WORLD_SIZE', '1'))
    lr = int(os.environ.get('LOCAL_RANK', '0'))
    ga = 8 // ws; dev = torch.device('cuda', lr)
    torch.cuda.set_device(dev)
    if distributed: dist.init_process_group(backend='nccl', device_id=dev); dist.barrier()
    master = rk == 0
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    if _ATTN == 'sdpa':
        from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_mem_efficient_sdp, enable_math_sdp
        enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

    logfile = None
    if master: os.makedirs('logs', exist_ok=True); logfile = f'logs/{args.run_id}.txt'; print(logfile)
    def log0(msg, console=True):
        if not master: return
        if console: print(msg)
        if logfile:
            with open(logfile, 'a') as f: print(msg, file=f)

    log0(code, console=False); log0('='*100, console=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    esl = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    vtok = load_val(args.val_files, max(args.train_seq_len, esl))
    luts = build_luts(sp, args.vocab_size, dev)

    log0(f'arch:JEPA layers={args.num_layers} dim={args.model_dim} heads={args.num_heads}/{args.num_kv_heads} '
         f'd_latent={args.d_latent} jepa_w={args.jepa_weight} vic_var={args.vicreg_var_weight} '
         f'vic_cov={args.vicreg_cov_weight} tgt_ema={args.target_ema_decay} attn={_ATTN}')

    model = JEPA_LM(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, rope_dims=args.rope_dims,
        qk_gain_init=args.qk_gain_init, d_latent=args.d_latent,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, ln_scale=args.ln_scale,
        tied_embed_init_std=args.tied_embed_init_std,
    ).to(dev).bfloat16()

    for bn in ['qo_bank', 'kv_bank', 'mlp_up_bank', 'mlp_down_bank']:
        getattr(model, bn).data = getattr(model, bn).data.float()
    for m in model.modules():
        if isinstance(m, CastedLinear): m.float()
    restore_fp32(model)

    # target encoder banks (ema copies, training-only)
    tgt_banks = {
        'qo': model.qo_bank.data.clone(),
        'kv': model.kv_bank.data.clone(),
        'up': model.mlp_up_bank.data.clone(),
        'dn': model.mlp_down_bank.data.clone(),
    }

    compiled = torch.compile(model, dynamic=False, fullgraph=True)

    mat_params = [model.qo_bank, model.kv_bank, model.mlp_up_bank, model.mlp_down_bank]
    blk_params = [p for _, p in model.blocks.named_parameters()]
    scalar_params = blk_params[:]
    scalar_params.append(model.smear.gate)
    if model.bigram:
        scalar_params.append(model.bigram.scale)
        if model.bigram.proj: scalar_params.append(model.bigram.proj.weight)
    scalar_params.append(model.skip_w)
    for p in model.predictor.parameters(): scalar_params.append(p)
    scalar_params.append(model.ctx_proj.weight)
    scalar_params.append(model.tgt_proj.weight)

    tok_lr = args.tied_embed_lr
    tok_pgs = [{'params': [model.tok_emb.weight], 'lr': tok_lr, 'base_lr': tok_lr}]
    if model.bigram: tok_pgs.append({'params': [model.bigram.emb.weight], 'lr': tok_lr, 'base_lr': tok_lr})

    opt_tok = torch.optim.AdamW(tok_pgs, betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    opt_muon = Muon(mat_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd)
    for g in opt_muon.param_groups: g['base_lr'] = args.matrix_lr
    opt_scalar = torch.optim.AdamW([{'params': scalar_params, 'lr': args.scalar_lr, 'base_lr': args.scalar_lr}],
                                    betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)

    repl = list(opt_tok.param_groups[0]['params'])
    for pg in opt_tok.param_groups[1:]: repl.extend(pg['params'])
    repl.extend(scalar_params)
    optimizers = [opt_tok, opt_muon, opt_scalar]

    n_params = sum(p.numel() for p in model.parameters())
    ctx_params = sum(v.numel() for v in model.context_encoder_state().values())
    log0(f'params: total={n_params} ctx_encoder={ctx_params} (saved in artifact)')

    loader = DLoader(args.train_files, rk, ws, dev)
    def zero_all():
        for o in optimizers: o.zero_grad(set_to_none=True)

    max_wc = 1000 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    def lr_mul(step, ems):
        wu = min(step / max(args.lr_warmup_steps, 1), 1.)
        if args.warmdown_iters <= 0: return wu
        if max_wc is None:
            s = max(args.iterations - args.warmdown_iters, 0)
            return wu * (0.5 * (1 + math.cos(math.pi * (step - s) / max(args.warmdown_iters, 1))) if step >= s else 1.)
        sms = ems / max(step, 1); wms = args.warmdown_iters * sms; rms = max(max_wc - ems, 0)
        return wu * (0.5 * (1 + math.cos(math.pi * (1 - rms / max(wms, 1e-9)))) if rms <= wms else 1.)

    # compile warmup
    if args.warmup_steps > 0:
        init_s = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        init_o = [copy.deepcopy(o.state_dict()) for o in optimizers]
        compiled.train()
        for ws_ in range(args.warmup_steps):
            zero_all()
            for _ in range(ga):
                x, y = loader.next_batch(args.train_batch_tokens, args.train_seq_len, ga)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    (compiled(x, y) / ga).backward()
            if distributed:
                for p in model.parameters():
                    if p.grad is not None: dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            for o in optimizers: o.step()
            zero_all()
            if ws_+1 == args.warmup_steps or (ws_+1)%10==0: log0(f'warmup:{ws_+1}/{args.warmup_steps}')
        model.load_state_dict(init_s, strict=True)
        for o, s in zip(optimizers, init_o, strict=True): o.load_state_dict(s)
        zero_all()
        loader = DLoader(args.train_files, rk, ws, dev)
        # Reset target banks after warmup
        for k in tgt_banks: tgt_banks[k] = getattr(model, {'qo': 'qo_bank', 'kv': 'kv_bank', 'up': 'mlp_up_bank', 'dn': 'mlp_down_bank'}[k]).data.clone()

    ema_state = {k: v.detach().float().clone() for k, v in model.state_dict().items()}
    swa_state, swa_count = None, 0
    train_ms, stop_at = 0., None
    torch.cuda.synchronize(); t0 = time.perf_counter(); step = 0

    while True:
        last = step == args.iterations or (stop_at is not None and step >= stop_at)
        if last or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize(); train_ms += 1000 * (time.perf_counter() - t0)
            vl, vb = eval_val(args, compiled, rk, ws, dev, ga, vtok, luts)
            log0(f'step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} time:{train_ms:.0f}ms avg:{train_ms/max(step,1):.2f}ms')
            torch.cuda.synchronize(); t0 = time.perf_counter()
        if last: break

        ems = train_ms + 1000 * (time.perf_counter() - t0)
        scale = lr_mul(step, ems)
        zero_all()
        tloss = torch.zeros((), device=dev)
        jepa_total, vic_total = 0., 0.
        vic_stats_acc = {'var': 0., 'cov': 0., 'z_std_mean': 0., 'z_std_min': 999.}

        # Anneal JEPA weight to 0 during warmdown (pure CE for final steps)
        jepa_w = args.jepa_weight * scale  # scale goes 1→0 during warmdown

        for _ in range(ga):
            x, y = loader.next_batch(args.train_batch_tokens, args.train_seq_len, ga)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                ce, jl, vl_, vs = model.forward_jepa(x, y, tgt_banks)
                loss = ce + jepa_w * jl + args.vicreg_var_weight * vl_
            tloss += loss.detach()
            (loss / ga).backward()
            jepa_total += jl.item() / ga; vic_total += vl_.item() / ga
            for k in ['var', 'cov', 'z_std_mean']:
                vic_stats_acc[k] += vs[k] / ga
            vic_stats_acc['z_std_min'] = min(vic_stats_acc['z_std_min'], vs['z_std_min'])
        tloss /= ga

        frac = min(step / args.muon_momentum_warmup_steps, 1.) if args.muon_momentum_warmup_steps > 0 else 1.
        for g in opt_muon.param_groups: g['momentum'] = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for o in optimizers:
            for g in o.param_groups: g['lr'] = g['base_lr'] * scale
        if args.grad_clip_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        opt_muon.launch_reduce_scatters()
        if distributed:
            for p in repl:
                if p.grad is not None: dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        opt_tok.step(); opt_scalar.step(); opt_muon.step(); zero_all()

        # EMA update for target encoder banks
        decay = args.target_ema_decay
        with torch.no_grad():
            for k, src_name in [('qo', 'qo_bank'), ('kv', 'kv_bank'), ('up', 'mlp_up_bank'), ('dn', 'mlp_down_bank')]:
                tgt_banks[k].mul_(decay).add_(getattr(model, src_name).data, alpha=1 - decay)

        # EMA update for model weights (for final averaging)
        ema_decay = min(0.9995, 1. - 10. / (step + 10.))
        with torch.no_grad():
            for k, v in model.state_dict().items():
                ema_state[k].mul_(ema_decay).add_(v.detach().float(), alpha=1 - ema_decay)

        step += 1
        approx = train_ms + 1000 * (time.perf_counter() - t0)

        if args.swa_enabled and scale < 0.2 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; swa_count = 1
                log0(f'swa:start step:{step}')
            else:
                for k, v in model.state_dict().items(): swa_state[k] += v.detach().cpu()
                swa_count += 1

        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            vs = vic_stats_acc
            log0(f'step:{step}/{args.iterations} loss:{tloss.item():.4f} jepa:{jepa_total:.4f} '
                 f'vic:{vic_total:.4f} z_std:{vs["z_std_mean"]:.3f}/{vs["z_std_min"]:.3f} '
                 f'jw:{jepa_w:.3f} time:{approx:.0f}ms avg:{approx/step:.2f}ms')

        reached = max_wc is not None and approx >= max_wc
        if distributed and max_wc is not None:
            rc = torch.tensor(int(reached), device=dev); dist.all_reduce(rc, op=dist.ReduceOp.MAX)
            reached = bool(rc.item())
        if stop_at is None and reached: stop_at = step

    log0(f'peak mem: {torch.cuda.max_memory_allocated()//1024//1024} MiB')

    # weight averaging
    if step >= 1000:
        log0('ema:applying'); cs = model.state_dict()
        model.load_state_dict({k: v.to(cs[k].dtype) for k, v in ema_state.items()}, strict=True)
    else:
        log0(f'ema:skipped ({step} steps)')

    torch.cuda.synchronize(); td = time.perf_counter()
    dv, db = eval_val(args, compiled, rk, ws, dev, ga, vtok, luts)
    log0(f'DIAGNOSTIC val_loss:{dv:.4f} val_bpb:{db:.4f} time:{1000*(time.perf_counter()-td):.0f}ms')

    # save context encoder only
    sd = model.context_encoder_state()
    sd = {k: v.detach().cpu() for k, v in sd.items()}
    ub = _unbank(sd, args.num_layers)
    qr, qm = mq6(ub, {'mlp', 'attn'})
    buf = io.BytesIO(); torch.save({'w': qr, 'm': qm}, buf)
    blob = lzma.compress(buf.getvalue(), preset=6)
    if master:
        with open('final_model.int6.ptz', 'wb') as f: f.write(blob)
        log0(f'artifact: {len(blob)+len(code.encode("utf-8"))} bytes (model:{len(blob)} code:{len(code.encode("utf-8"))})')

    if distributed: dist.barrier()

    # roundtrip verification
    with open('final_model.int6.ptz', 'rb') as f:
        qs = torch.load(io.BytesIO(lzma.decompress(f.read())), map_location='cpu')
    deq = _rebank(dq6(qs['w'], qs['m'], ub), args.num_layers, sd)
    eval_m = JEPA_LM(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, rope_dims=args.rope_dims,
        qk_gain_init=args.qk_gain_init, d_latent=args.d_latent,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, ln_scale=args.ln_scale,
    ).to(dev).bfloat16()
    for bn in ['qo_bank', 'kv_bank', 'mlp_up_bank', 'mlp_down_bank']:
        getattr(eval_m, bn).data = getattr(eval_m, bn).data.float()
    for m in eval_m.modules():
        if isinstance(m, CastedLinear): m.float()
    restore_fp32(eval_m)
    eval_m.load_state_dict(deq, strict=False)

    tq = time.perf_counter()
    qvl, qvb = eval_val(args, torch.compile(eval_m, dynamic=False, fullgraph=True), rk, ws, dev, ga, vtok, luts, esl)
    log0(f'final_int6_roundtrip val_loss:{qvl:.4f} val_bpb:{qvb:.4f} time:{1000*(time.perf_counter()-tq):.0f}ms')
    log0(f'final_int6_roundtrip_exact val_loss:{qvl:.8f} val_bpb:{qvb:.8f}')
    log0(f'final_int8_zlib_roundtrip_exact val_loss:{qvl:.8f} val_bpb:{qvb:.8f}')

    if args.eval_stride > 0 and args.eval_stride < esl:
        ts = time.perf_counter()
        svl, svb = eval_sliding(args, eval_m, rk, ws, dev, vtok, luts, args.eval_stride, esl=esl)
        log0(f'final_sliding val_loss:{svl:.4f} val_bpb:{svb:.4f} time:{1000*(time.perf_counter()-ts):.0f}ms')
        log0(f'final_int8_zlib_roundtrip_exact val_loss:{svl:.8f} val_bpb:{svb:.8f}')

    if distributed: dist.destroy_process_group()

if __name__ == '__main__':
    main()
