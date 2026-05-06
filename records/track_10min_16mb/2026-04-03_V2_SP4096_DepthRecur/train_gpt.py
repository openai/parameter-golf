import copy,glob,io,lzma,math,os,random,subprocess,sys,time,uuid
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch,torch.distributed as dist,torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor, nn
from flash_attn_interface import flash_attn_func as flash_attn_3_func
_E=os.environ.get
class Hyperparameters():
    data_dir=_E('DATA_DIR','./data/')
    seed=int(_E('SEED',1337))
    run_id=_E("RUN_ID",str(uuid.uuid4()))
    iterations=int(_E('ITERATIONS',20000))
    warmdown_frac=float(_E('WARMDOWN_FRAC',0.667))
    warmup_steps=int(_E('WARMUP_STEPS',20))
    train_batch_tokens=int(_E('TRAIN_BATCH_TOKENS',2048*48*8))
    train_seq_len=int(_E('TRAIN_SEQ_LEN',2048))
    eval_seq_len=int(_E('EVAL_SEQ_LEN',2048))
    max_wallclock_seconds=float(_E('MAX_WALLCLOCK_SECONDS',600.0))
    train_log_every=int(_E('TRAIN_LOG_EVERY',500))
    val_batch_tokens=int(_E('VAL_BATCH_TOKENS',2048*32*8))
    val_loss_every=int(_E('VAL_LOSS_EVERY',4000))
    sliding_window_enabled=bool(int(_E('SLIDING_WINDOW_ENABLED','1')))
    vocab_size=int(_E('VOCAB_SIZE',4096))
    num_layers=int(_E('NUM_LAYERS',11))
    xsa_last_n=int(_E('XSA_LAST_N',11))
    num_kv_heads=int(_E('NUM_KV_HEADS',4))
    model_dim=int(_E('MODEL_DIM',512))
    embedding_dim=int(_E('EMBEDDING_DIM',512))
    num_heads=int(_E('NUM_HEADS',8))
    mlp_mult=float(_E('MLP_MULT',4.0))
    skip_gates_enabled=bool(int(_E('SKIP_GATES_ENABLED','1')))
    tie_embeddings=bool(int(_E('TIE_EMBEDDINGS','1')))
    logit_softcap=float(_E('LOGIT_SOFTCAP',30.0))
    rope_base=float(_E('ROPE_BASE',10000.0))
    rope_dims=int(_E('ROPE_DIMS',16))
    rope_train_seq_len=int(_E('ROPE_TRAIN_SEQ_LEN',2048))
    ln_scale=bool(int(_E('LN_SCALE','1')))
    ve_enabled=bool(int(_E('VE_ENABLED','1')))
    ve_dim=int(_E('VE_DIM',128))
    ve_layers=_E('VE_LAYERS','9,10')
    qk_gain_init=float(_E('QK_GAIN_INIT',4.0))
    min_lr=float(_E('MIN_LR',0.0))
    embed_lr=float(_E('EMBED_LR',0.6))
    head_lr=float(_E('HEAD_LR',0.008))
    tied_embed_lr=float(_E('TIED_EMBED_LR',0.03))
    tied_embed_init_std=float(_E('TIED_EMBED_INIT_STD',0.005))
    matrix_lr=float(_E('MATRIX_LR',0.02))
    scalar_lr=float(_E('SCALAR_LR',0.02))
    muon_momentum=float(_E('MUON_MOMENTUM',0.99))
    muon_backend_steps=int(_E('MUON_BACKEND_STEPS',4))
    muon_momentum_warmup_start=float(_E('MUON_MOMENTUM_WARMUP_START',0.92))
    muon_momentum_warmup_steps=int(_E('MUON_MOMENTUM_WARMUP_STEPS',1500))
    beta1=float(_E('BETA1',0.9))
    beta2=float(_E('BETA2',0.95))
    adam_eps=float(_E('ADAM_EPS',1e-8))
    grad_clip_norm=float(_E('GRAD_CLIP_NORM',0.3))
    eval_stride=int(_E('EVAL_STRIDE',64))
    muon_beta2=float(_E('MUON_BETA2',0.95))
    adam_wd=float(_E('ADAM_WD',0.02))
    muon_wd=float(_E('MUON_WD',0.090))
    embed_wd=float(_E('EMBED_WD',0.090))
    ema_decay=float(_E('EMA_DECAY',0.997))
    recur_layers=_E('RECUR_LAYERS','')
    recur_start_step=int(_E('RECUR_START_STEP',3000))
    slot_enabled=bool(int(_E('SLOT_ENABLED','0')))
    slot_steps=int(_E('SLOT_STEPS',8))
    slot_lr=float(_E('SLOT_LR',0.005))
    compressor=_E('COMPRESSOR','brotli')
    gptq_enabled=bool(int(_E('GPTQ_ENABLED','1')))
    gptq_calibration_batches=int(_E('GPTQ_CALIBRATION_BATCHES',64))
    gptq_reserve_seconds=float(_E('GPTQ_RESERVE_SECONDS',10.0))
    distributed="RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank=int(_E("RANK","0"))
    world_size=int(_E("WORLD_SIZE","1"))
    local_rank=int(_E("LOCAL_RANK","0"))
    is_main_process=rank==0
    grad_accum_steps=8//world_size
    datasets_dir=os.path.join(data_dir,'datasets',f'fineweb10B_sp{vocab_size}')
    train_files=os.path.join(datasets_dir,'fineweb_train_*.bin')
    val_files=os.path.join(datasets_dir,'fineweb_val_*.bin')
    tokenizer_path=os.path.join(data_dir,'tokenizers',f'fineweb_{vocab_size}_bpe.model')
    logfile=f"logs/{run_id}.txt"
    model_path="final_model.pt"
    quantized_model_path="final_model.int6.ptz"
_logger_hparams=None
def set_logging_hparams(h):
    global _logger_hparams
    _logger_hparams=h
def log(msg,console=True):
    if _logger_hparams is None:
        print(msg)
    if _logger_hparams.is_main_process:
        if console:
            print(msg)
        if _logger_hparams.logfile is not None:
            with open(_logger_hparams.logfile,"a",encoding="utf-8") as f:
                print(msg,file=f)
class ValidationData:
    def __init__(self,h,device):
        if not h.tokenizer_path.endswith(".model"):
            raise ValueError(f"Script only setup for SentencePiece .model file: {h.tokenizer_path}")
        self.sp=spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        if int(self.sp.vocab_size())!=h.vocab_size:
            raise ValueError(f"VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={int(self.sp.vocab_size())}")
        self.val_tokens=_load_val_tokens(h.val_files,h.eval_seq_len)
        self.base_bytes_lut,self.has_leading_space_lut,self.is_boundary_token_lut=_build_sp_luts(self.sp,h.vocab_size,device)
def _build_sp_luts(sp,vocab_size,device):
    sv=int(sp.vocab_size())
    assert sp.piece_to_id("\u2581")!=sp.unk_id(),"Tokenizer must have \u2581 as its own token"
    ts=max(sv,vocab_size)
    bb=np.zeros((ts,),dtype=np.int16)
    hl=np.zeros((ts,),dtype=np.bool_)
    ib=np.ones((ts,),dtype=np.bool_)
    for tid in range(sv):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        ib[tid]=False
        if sp.is_byte(tid):
            bb[tid]=1
            continue
        piece=sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            hl[tid]=True
            piece=piece[1:]
        bb[tid]=len(piece.encode("utf-8"))
    return (torch.tensor(bb,dtype=torch.int16,device=device),
            torch.tensor(hl,dtype=torch.bool,device=device),
            torch.tensor(ib,dtype=torch.bool,device=device))
def _load_val_tokens(pattern,seq_len):
    files=[Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens=torch.cat([_load_shard(f) for f in files]).contiguous()
    usable=((tokens.numel()-1)//seq_len)*seq_len
    if usable<=0:
        raise ValueError(f"Validation split too short for seq_len={seq_len}")
    return tokens[:usable+1]
def _load_shard(file):
    hb=256*np.dtype("<i4").itemsize
    header=np.fromfile(file,dtype="<i4",count=256)
    if header.size!=256 or int(header[0])!=20240520 or int(header[1])!=1:
        raise ValueError(f"Unexpected shard header for {file}")
    nt=int(header[2])
    if file.stat().st_size!=hb+nt*np.dtype("<u2").itemsize:
        raise ValueError(f"Shard size mismatch for {file}")
    t=np.fromfile(file,dtype="<u2",count=nt,offset=hb)
    if t.size!=nt:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(t.astype(np.uint16,copy=False))
_SHB=256*np.dtype("<i4").itemsize
_SNC:dict[str,int]={}
_MMC:dict[str,np.memmap]={}
def _read_nt(file):
    k=str(file)
    c=_SNC.get(k)
    if c is not None:
        return c
    header=np.fromfile(file,dtype="<i4",count=256)
    if header.size!=256 or int(header[0])!=20240520 or int(header[1])!=1:
        raise ValueError(f"Unexpected shard header for {file}")
    n=int(header[2])
    _SNC[k]=n
    return n
def _get_mm(file):
    k=str(file)
    mm=_MMC.get(k)
    if mm is not None:
        return mm
    n=_read_nt(file)
    mm=np.memmap(file,mode="r",dtype="<u2",offset=_SHB,shape=(n,))
    _MMC[k]=mm
    return mm
class DistributedTokenLoader:
    def __init__(self,pattern,rank,world_size,device):
        self.rank=rank
        self.world_size=world_size
        self.device=device
        self.files=[Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self._nt=np.array([_read_nt(f) for f in self.files],dtype=np.int64)
        seed=0
        for f in self.files:
            for b in str(f).encode():
                seed=((seed^b)*1099511628211)&0xFFFFFFFFFFFFFFFF
        self._rng=np.random.Generator(np.random.PCG64(seed))
        self._cfg=None
        self._esh=None
        self._bbc=None
        n=len(self.files)
        self._cp=np.zeros(n,dtype=np.int64)
        self._cbc=np.zeros(n,dtype=np.int64)
        self._cn=np.zeros(n,dtype=np.int64)
        self._cs=np.zeros(n,dtype=np.int64)
        self._cst=np.ones(n,dtype=np.int64)
        self._ci=np.zeros(n,dtype=np.bool_)
        self._bb=0
    def _coprime(self,n):
        if n<=1:
            return 1
        while True:
            s=int(self._rng.integers(1,n))
            if math.gcd(s,n)==1:
                return s
    def _reset(self,si,sl):
        nt=int(self._nt[si])
        mp=min(sl-1,max(0,nt-sl-1))
        ph=int(self._rng.integers(mp+1)) if mp>0 else 0
        bc=(nt-1-ph)//sl
        self._cp[si]=ph
        self._cbc[si]=bc
        self._cn[si]=0
        self._cs[si]=int(self._rng.integers(bc)) if bc>1 else 0
        self._cst[si]=self._coprime(bc)
        self._ci[si]=True
    def _ensure(self,si,sl):
        if not self._ci[si] or self._cn[si]>=self._cbc[si]:
            self._reset(si,sl)
    def _take(self,si,sl,count,out):
        rem=count
        while rem>0:
            self._ensure(si,sl)
            bc=int(self._cbc[si])
            ni=int(self._cn[si])
            take=min(rem,bc-ni)
            ph=int(self._cp[si])
            st=int(self._cs[si])
            stride=int(self._cst[si])
            for j in range(take):
                bi=(st+(ni+j)*stride)%bc
                out.append((si,ph+bi*sl))
            self._cn[si]=ni+take
            rem-=take
    def _init_pipe(self,gt,sl,gas):
        lt=gt//(self.world_size*gas)
        ns=lt//sl
        gns=ns*self.world_size
        self._cfg=(lt,sl,ns,gns)
        bbc=(self._nt-1)//sl
        el=bbc>0
        self._esh=np.nonzero(el)[0].astype(np.int64)
        self._bbc=bbc[self._esh].astype(np.int64)
    def _sample_gw(self):
        assert self._cfg is not None and self._esh is not None
        _,sl,_,gns=self._cfg
        ec=int(self._esh.size)
        prog=min(self._bb/1800.0,1.0)
        rem=np.empty(ec,dtype=np.float64)
        for i,si in enumerate(self._esh.tolist()):
            if self._ci[si]:
                r=int(self._cbc[si])-int(self._cn[si])
                rem[i]=float(max(r,1))
            else:
                rem[i]=float(self._bbc[i])
        alpha=0.90-0.40*prog
        w=np.power(rem,alpha)
        ws=float(w.sum())
        if not np.isfinite(ws) or ws<=0.0:
            w=np.ones(ec,dtype=np.float64)
            ws=float(w.sum())
        pr=w/ws
        lo=min(max(8,self.world_size),ec,gns)
        hi=min(max(32,self.world_size*8),ec,gns)
        mix=max(1,min(int(round(lo+prog*(hi-lo))),ec,gns))
        cp=self._rng.choice(ec,size=mix,replace=False,p=pr)
        cs=self._esh[cp]
        cpr=pr[cp].copy()
        cpr/=cpr.sum()
        counts=np.ones(mix,dtype=np.int64)
        extra=gns-mix
        if extra>0:
            counts+=self._rng.multinomial(extra,cpr).astype(np.int64)
        perm=self._rng.permutation(mix)
        cs,counts=cs[perm],counts[perm]
        bkts:list[list[tuple[int,int]]]=[]
        for si,cnt in zip(cs.tolist(),counts.tolist()):
            b:list[tuple[int,int]]=[]
            self._take(int(si),sl,int(cnt),b)
            if b:
                if len(b)>1:
                    bp=self._rng.permutation(len(b))
                    b=[b[int(k)] for k in bp.tolist()]
                bkts.append(b)
        wins:list[tuple[int,int]]=[]
        active=[i for i,bk in enumerate(bkts) if bk]
        while active:
            order=self._rng.permutation(len(active))
            na:list[int]=[]
            for oi in order.tolist():
                bi=active[oi]
                if bkts[bi]:
                    wins.append(bkts[bi].pop())
                if bkts[bi]:
                    na.append(bi)
            active=na
        return wins
    def next_batch(self,gt,sl,gas):
        if self._cfg is None:
            self._init_pipe(gt,sl,gas)
        _,_,ns,_=self._cfg
        gw=self._sample_gw()
        lw=gw[self.rank::self.world_size]
        x=torch.empty((ns,sl),dtype=torch.int64)
        y=torch.empty((ns,sl),dtype=torch.int64)
        for slot,(si,pos) in enumerate(lw):
            mm=_get_mm(self.files[si])
            win=torch.as_tensor(np.array(mm[pos:pos+sl+1],dtype=np.int64))
            x[slot]=win[:-1]
            y[slot]=win[1:]
        self._bb+=1
        return x.to(self.device,non_blocking=True),y.to(self.device,non_blocking=True)
class RMSNorm(nn.Module):
    def __init__(self,eps=None):
        super().__init__()
        self.eps=eps
    def forward(self,x):
        return F.rms_norm(x,(x.size(-1),),eps=self.eps)
class CastedLinear(nn.Linear):
    def forward(self,x):
        w=self.weight.to(x.dtype)
        b=self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x,w,b)
class Rotary(nn.Module):
    def __init__(self,dim,base=10000.0,train_seq_len=1024,rope_dims=0):
        super().__init__()
        self.dim=dim
        self.base=base
        self.train_seq_len=train_seq_len
        self.rope_dims=rope_dims if rope_dims>0 else dim
        inv_freq=1.0/(base**(torch.arange(0,self.rope_dims,2,dtype=torch.float32)/self.rope_dims))
        self.register_buffer("inv_freq",inv_freq,persistent=False)
        self._sl=0
        self._cc=None
        self._sc=None
    def forward(self,seq_len,device,dtype):
        if self._cc is None or self._sc is None or self._sl!=seq_len or self._cc.device!=device:
            rd=self.rope_dims
            if seq_len>self.train_seq_len:
                scale=seq_len/self.train_seq_len
                nb=self.base*(scale**(rd/(rd-2)))
                inv_freq=1.0/(nb**(torch.arange(0,rd,2,dtype=torch.float32,device=device)/rd))
            else:
                inv_freq=self.inv_freq.to(device)
            t=torch.arange(seq_len,device=device,dtype=inv_freq.dtype)
            freqs=torch.outer(t,inv_freq)
            self._cc=freqs.cos()[None,:,None,:]
            self._sc=freqs.sin()[None,:,None,:]
            self._sl=seq_len
        return self._cc.to(dtype=dtype),self._sc.to(dtype=dtype)
def apply_rotary_emb(x,cos,sin,rope_dims=0):
    if rope_dims>0 and rope_dims<x.size(-1):
        xr,xp=x[...,:rope_dims],x[...,rope_dims:]
        half=rope_dims//2
        x1,x2=xr[...,:half],xr[...,half:]
        xr=torch.cat((x1*cos+x2*sin,x1*(-sin)+x2*cos),dim=-1)
        return torch.cat((xr,xp),dim=-1)
    half=x.size(-1)//2
    x1,x2=x[...,:half],x[...,half:]
    return torch.cat((x1*cos+x2*sin,x1*(-sin)+x2*cos),dim=-1)
class CausalSelfAttention(nn.Module):
    def __init__(self,dim,num_heads,num_kv_heads,rope_base,qk_gain_init,train_seq_len):
        super().__init__()
        assert dim%num_heads==0 and num_heads%num_kv_heads==0
        self.num_heads=num_heads
        self.num_kv_heads=num_kv_heads
        self.head_dim=dim//num_heads
        assert self.head_dim%2==0
        kv_dim=self.num_kv_heads*self.head_dim
        self.c_q=CastedLinear(dim,dim,bias=False)
        self.c_k=CastedLinear(dim,kv_dim,bias=False)
        self.c_v=CastedLinear(dim,kv_dim,bias=False)
        self.proj=CastedLinear(dim,dim,bias=False)
        self.proj._zero_init=True
        self.q_gain=nn.Parameter(torch.full((num_heads,),qk_gain_init,dtype=torch.float32))
        self.rope_dims=0
        self.rotary=Rotary(self.head_dim,base=rope_base,train_seq_len=train_seq_len)
        self.use_xsa=False
    def _xsa(self,y,v):
        B,T,H,D=y.shape
        Hkv=v.size(-2)
        g=H//Hkv
        yg=y.reshape(B,T,Hkv,g,D)
        vn=F.normalize(v,dim=-1).unsqueeze(-2)
        p=(yg*vn).sum(dim=-1,keepdim=True)*vn
        return (yg-p).reshape(B,T,H,D)
    def forward(self,x,v_embed=None):
        bsz,sl,dim=x.shape
        q=self.c_q(x).reshape(bsz,sl,self.num_heads,self.head_dim)
        k=self.c_k(x).reshape(bsz,sl,self.num_kv_heads,self.head_dim)
        v=self.c_v(x)
        if v_embed is not None:
            v=v+v_embed
        v=v.reshape(bsz,sl,self.num_kv_heads,self.head_dim)
        q=F.rms_norm(q,(q.size(-1),))
        k=F.rms_norm(k,(k.size(-1),))
        cos,sin=self.rotary(sl,x.device,q.dtype)
        q=apply_rotary_emb(q,cos,sin,self.rope_dims)
        k=apply_rotary_emb(k,cos,sin,self.rope_dims)
        q=q*self.q_gain.to(dtype=q.dtype)[None,None,:,None]
        y=flash_attn_3_func(q,k,v,causal=True)
        if self.use_xsa:
            y=self._xsa(y,v)
        return self.proj(y.reshape(bsz,sl,dim))
class ValueEmbedding(nn.Module):
    def __init__(self,vocab_size,ve_dim,model_dim):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,ve_dim)
        nn.init.normal_(self.embed.weight,std=0.01)
        self.proj=CastedLinear(ve_dim,model_dim,bias=False) if ve_dim!=model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale=nn.Parameter(torch.tensor(0.1,dtype=torch.float32))
    def forward(self,token_ids):
        h=self.embed(token_ids)
        if self.proj is not None:
            h=self.proj(h)
        return h*self.scale.to(dtype=h.dtype)
class MLP(nn.Module):
    def __init__(self,dim,mlp_mult):
        super().__init__()
        hidden=int(mlp_mult*dim)
        self.fc=CastedLinear(dim,hidden,bias=False)
        self.proj=CastedLinear(hidden,dim,bias=False)
        self.proj._zero_init=True
    def forward(self,x):
        return self.proj(F.leaky_relu(self.fc(x),negative_slope=0.5).square())
class Block(nn.Module):
    def __init__(self,dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init,train_seq_len,layer_idx=0,ln_scale=False):
        super().__init__()
        self.attn_norm=RMSNorm()
        self.mlp_norm=RMSNorm()
        self.attn=CausalSelfAttention(dim,num_heads,num_kv_heads,rope_base,qk_gain_init,train_seq_len)
        self.mlp=MLP(dim,mlp_mult)
        self.attn_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32))
        self.mlp_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32))
        self.resid_mix=nn.Parameter(torch.stack((torch.ones(dim),torch.zeros(dim))).float())
        self.ln_scale_factor=1.0/math.sqrt(layer_idx+1) if ln_scale else 1.0
    def forward(self,x,x0,v_embed=None):
        mix=self.resid_mix.to(dtype=x.dtype)
        xi=mix[0][None,None,:]*x+mix[1][None,None,:]*x0
        ao=self.attn(self.attn_norm(xi)*self.ln_scale_factor,v_embed=v_embed)
        xo=xi+self.attn_scale.to(dtype=xi.dtype)[None,None,:]*ao
        xo=xo+self.mlp_scale.to(dtype=xo.dtype)[None,None,:]*self.mlp(self.mlp_norm(xo)*self.ln_scale_factor)
        return xo
class GPT(nn.Module):
    def __init__(self,h):
        super().__init__()
        self._ve_target_dim=h.num_kv_heads*(h.model_dim//h.num_heads)
        assert h.logit_softcap>0.0
        self.tie_embeddings=h.tie_embeddings
        self.tied_embed_init_std=h.tied_embed_init_std
        self.logit_softcap=h.logit_softcap
        self._recur_set=set()
        if h.recur_layers:
            self._recur_set={int(x) for x in h.recur_layers.split(",") if x.strip()}
        self._recur_active=bool(self._recur_set)
        self.tok_emb=nn.Embedding(h.vocab_size,h.embedding_dim)
        if h.embedding_dim!=h.model_dim:
            self.embed_proj=CastedLinear(h.embedding_dim,h.model_dim,bias=False)
            self.head_proj=CastedLinear(h.model_dim,h.embedding_dim,bias=False)
        else:
            self.embed_proj=None
            self.head_proj=None
        self.num_encoder_layers=h.num_layers//2
        self.num_decoder_layers=h.num_layers-self.num_encoder_layers
        self.num_skip_weights=min(self.num_encoder_layers,self.num_decoder_layers)
        self.skip_weights=nn.Parameter(torch.ones(self.num_skip_weights,h.model_dim,dtype=torch.float32))
        self.skip_gates=nn.Parameter(torch.zeros(self.num_skip_weights,h.model_dim,dtype=torch.float32)) if h.skip_gates_enabled else None
        self.blocks=nn.ModuleList([Block(h.model_dim,h.num_heads,h.num_kv_heads,h.mlp_mult,h.rope_base,h.qk_gain_init,h.train_seq_len,layer_idx=i,ln_scale=h.ln_scale) for i in range(h.num_layers)])
        if h.rope_dims>0:
            hd=h.model_dim//h.num_heads
            for block in self.blocks:
                block.attn.rope_dims=h.rope_dims
                block.attn.rotary=Rotary(hd,base=h.rope_base,train_seq_len=h.train_seq_len,rope_dims=h.rope_dims)
        self.ve_layer_indices=[int(x) for x in h.ve_layers.split(",") if x.strip()] if h.ve_enabled else []
        kv_dim=self._ve_target_dim
        if self.ve_layer_indices:
            self.ve_shared=ValueEmbedding(h.vocab_size,h.ve_dim,kv_dim)
            self.ve_layer_scales=nn.ParameterList([nn.Parameter(torch.ones(1,dtype=torch.float32)) for _ in self.ve_layer_indices])
        else:
            self.ve_shared=None
            self.ve_layer_scales=nn.ParameterList()
        self.value_embeds=nn.ModuleList()
        self.final_norm=RMSNorm()
        self.lm_head=None if h.tie_embeddings else CastedLinear(h.embedding_dim,h.vocab_size,bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init=True
        if h.xsa_last_n>0:
            for i in range(max(0,h.num_layers-h.xsa_last_n),h.num_layers):
                self.blocks[i].attn.use_xsa=True
        self._init_weights()
    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight,mean=0.0,std=self.tied_embed_init_std)
        for name,module in self.named_modules():
            if isinstance(module,nn.Linear):
                if getattr(module,"_zero_init",False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim==2 and module.weight.shape[0]>=64 and module.weight.shape[1]>=64:
                    nn.init.orthogonal_(module.weight,gain=1.0)
    def _get_ve(self,li,input_ids,ve_cache=None):
        if self.ve_shared is None or li not in self.ve_layer_indices:
            return None
        if ve_cache is not None and 've' not in ve_cache:
            ve_cache['ve']=self.ve_shared(input_ids)
        vb=ve_cache['ve'] if ve_cache is not None else self.ve_shared(input_ids)
        vi=self.ve_layer_indices.index(li)
        return vb*self.ve_layer_scales[vi].to(dtype=vb.dtype)
    def forward_backbone(self,input_ids):
        x=self.tok_emb(input_ids)
        x=F.rms_norm(x,(x.size(-1),))
        if self.embed_proj is not None:
            x=self.embed_proj(x)
        x0=x
        skips=[]
        vc={}
        schedule=[]
        rs=sorted(self._recur_set) if (self._recur_active and self._recur_set) else []
        mr=max(rs) if rs else -1
        for i in range(len(self.blocks)):
            schedule.append(i)
            if i==mr:
                schedule.extend(rs)
        seen=set()
        for li in schedule:
            first=li not in seen
            seen.add(li)
            enc=li<self.num_encoder_layers
            if enc:
                ve=self._get_ve(li,input_ids,vc)
                x=self.blocks[li](x,x0,v_embed=ve)
                if first:
                    skips.append(x)
            else:
                di=li-self.num_encoder_layers
                if first and skips:
                    ss=self.skip_weights[di].to(dtype=x.dtype)[None,None,:]*skips.pop()
                    if self.skip_gates is not None:
                        g=torch.sigmoid(self.skip_gates[di].to(dtype=x.dtype))[None,None,:]
                        x=torch.lerp(ss,x,g)
                    else:
                        x=x+ss
                ve=self._get_ve(li,input_ids,vc)
                x=self.blocks[li](x,x0,v_embed=ve)
        return self.final_norm(x)
    def _head_logits(self,h):
        if self.head_proj is not None:
            h=self.head_proj(h)
        if self.tie_embeddings:
            lo=F.linear(h,self.tok_emb.weight)
        else:
            lo=self.lm_head(h)
        return self.logit_softcap*torch.tanh(lo/self.logit_softcap)
    def forward_logits(self,input_ids):
        return self._head_logits(self.forward_backbone(input_ids))
    def forward(self,input_ids,target_ids):
        logits=self.forward_logits(input_ids)
        return F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),target_ids.reshape(-1),reduction="mean")
def classify_param(name):
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"
_PE_COEFFS=[(8.156554524902461,-22.48329292557795,15.878769915207462),(4.042929935166739,-2.808917465908714,0.5000178451051316),(3.8916678022926607,-2.772484153217685,0.5060648178503393),(3.285753657755655,-2.3681294933425376,0.46449024233003106),(2.3465413258596377,-1.7097828382687081,0.42323551169305323)]
@torch.compile
def zeropower_via_newtonschulz5(G,steps=10,eps=1e-7):
    X=G.bfloat16()
    X/=X.norm()+eps
    tr=G.size(0)>G.size(1)
    if tr:
        X=X.T
    for a,b,c in _PE_COEFFS[:steps]:
        A=X@X.T
        B=b*A+c*A@A
        X=a*X+B@X
    return X.T if tr else X
class Muon(torch.optim.Optimizer):
    def __init__(self,params,lr,momentum,backend_steps,nesterov=True,weight_decay=0.0):
        super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,nesterov=nesterov,weight_decay=weight_decay))
    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():
                loss=closure()
        dd=dist.is_available() and dist.is_initialized()
        ws=dist.get_world_size() if dd else 1
        rk=dist.get_rank() if dd else 0
        for group in self.param_groups:
            params=group["params"]
            if not params:
                continue
            lr=group["lr"]
            mom=group["momentum"]
            bs=group["backend_steps"]
            nest=group["nesterov"]
            tp=sum(int(p.numel()) for p in params)
            uf=torch.zeros(tp,device=params[0].device,dtype=torch.bfloat16)
            cur=0
            for i,p in enumerate(params):
                if i%ws==rk and p.grad is not None:
                    g=p.grad
                    st=self.state[p]
                    if "momentum_buffer" not in st:
                        st["momentum_buffer"]=torch.zeros_like(g)
                    buf=st["momentum_buffer"]
                    buf.mul_(mom).add_(g)
                    if nest:
                        g=g.add(buf,alpha=mom)
                    g=g/(g.norm(dim=-1,keepdim=True).clamp_min(1e-7))
                    g=zeropower_via_newtonschulz5(g,steps=bs)
                    g*=max(1,g.size(0)/g.size(1))**0.5
                    uf[cur:cur+p.numel()]=g.reshape(-1)
                cur+=p.numel()
            if dd:
                dist.all_reduce(uf,op=dist.ReduceOp.SUM)
            wd=group.get("weight_decay",0.0)
            cur=0
            for p in params:
                if wd>0.0:
                    p.data.mul_(1.0-lr*wd)
                g=uf[cur:cur+p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g,alpha=-lr)
                cur+=p.numel()
        return loss
CTRL_PAT=tuple(p for p in _E("CONTROL_TENSOR_NAME_PATTERNS","attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates,ve_layer_scales,ve_shared.scale").split(",") if p)
_I8SD=torch.float16
_I8CP=99.99984
_I8CQ=_I8CP/100.0
def _qft(t):
    t32=t.float()
    if t32.ndim==2:
        ca=torch.quantile(t32.abs(),_I8CQ,dim=1) if t32.numel() else torch.empty((t32.shape[0],),dtype=torch.float32)
        cl=torch.maximum(torch.minimum(t32,ca[:,None]),-ca[:,None])
        s=(ca/127.0).clamp_min(1.0/127.0)
        q=torch.clamp(torch.round(cl/s[:,None]),-127,127).to(torch.int8).contiguous()
        return q,s.to(dtype=_I8SD).contiguous()
    ca=float(torch.quantile(t32.abs().flatten(),_I8CQ).item()) if t32.numel() else 0.0
    s=torch.tensor(ca/127.0 if ca>0 else 1.0,dtype=torch.float32)
    q=torch.clamp(torch.round(torch.clamp(t32,-ca,ca)/s),-127,127).to(torch.int8).contiguous()
    return q,s
def restore_fp32_params(model):
    for m in model.modules():
        if isinstance(m,CastedLinear):
            m.float()
    for name,param in model.named_parameters():
        if (param.ndim<2 or any(p in name for p in CTRL_PAT)) and param.dtype!=torch.float32:
            param.data=param.data.float()
def _qi6(t,cr=31):
    t32=t.float()
    if t32.ndim==2:
        bq,bs,be=None,None,float('inf')
        for pct in [0.9990,0.9995,0.9999,0.99999,1.0]:
            rc=torch.quantile(t32.abs(),pct,dim=1) if pct<1.0 else t32.abs().amax(dim=1)
            s=(rc/cr).clamp_min(1.0/cr).to(torch.float16)
            q=torch.clamp(torch.round(t32/s.float()[:,None]),-cr,cr).to(torch.int8)
            err=(t32-q.float()*s.float()[:,None]).pow(2).mean().item()
            if err<be:
                bq,bs,be=q,s,err
        return bq,bs
    am=t32.abs().max().item()
    s=torch.tensor(am/cr if am>0 else 1.0,dtype=torch.float16)
    q=torch.clamp(torch.round(t32/s.float()),-cr,cr).to(torch.int8)
    return q,s
def _collect_H(model,loader,h,device,ncb=64):
    H={}
    hooks=[]
    def mk(name):
        def fn(mod,inp,out):
            x=inp[0].detach().float()
            if x.ndim==3:
                x=x.reshape(-1,x.shape[-1])
            if name not in H:
                H[name]=torch.zeros(x.shape[1],x.shape[1],dtype=torch.float32,device=device)
            H[name].addmm_(x.T,x)
        return fn
    for name,mod in model.named_modules():
        if isinstance(mod,CastedLinear) and mod.weight.numel()>65536:
            cat=classify_param(name+".weight")
            if cat in ("mlp","attn"):
                hooks.append(mod.register_forward_hook(mk(name+".weight")))
    model.eval()
    with torch.no_grad():
        for i in range(ncb):
            x,y=loader.next_batch(h.train_batch_tokens,h.train_seq_len,h.grad_accum_steps)
            model.forward_logits(x)
    for hk in hooks:
        hk.remove()
    for name in H:
        H[name]=H[name].cpu()/ncb
    return H
def _gptq_qw(w,Hm,cr=31,bs=128):
    W=w.float().clone()
    rows,cols=W.shape
    Hm=Hm.float().clone()
    dead=torch.diag(Hm)==0
    Hm[dead,dead]=1
    damp=0.01*Hm.diag().mean()
    Hm.diagonal().add_(damp)
    perm=torch.argsort(Hm.diag(),descending=True)
    invperm=torch.argsort(perm)
    Wp=W[:,perm].clone()
    Wp[:,dead[perm]]=0
    Hm=Hm[perm][:,perm]
    try:
        Hi=torch.cholesky_inverse(torch.linalg.cholesky(Hm))
        Hi=torch.linalg.cholesky(Hi,upper=True)
    except torch.linalg.LinAlgError:
        return _qi6(W,cr)
    bq,bsc,be=None,None,float('inf')
    for pct in [0.9990,0.9995,0.9999,0.99999,1.0]:
        rc=torch.quantile(W.abs(),pct,dim=1) if pct<1.0 else W.abs().amax(dim=1)
        s=(rc/cr).clamp_min(1.0/cr).to(torch.float16)
        sf=s.float()
        Q=torch.zeros(rows,cols,dtype=torch.int8)
        Ww=Wp.clone()
        for i1 in range(0,cols,bs):
            i2=min(i1+bs,cols)
            Wb=Ww[:,i1:i2].clone()
            Hb=Hi[i1:i2,i1:i2]
            Er=torch.zeros(rows,i2-i1)
            for j in range(i2-i1):
                wc=Wb[:,j]
                d=Hb[j,j]
                qc=torch.clamp(torch.round(wc/sf),-cr,cr)
                Q[:,i1+j]=qc.to(torch.int8)
                er=(wc-qc.float()*sf)/d
                Er[:,j]=er
                Wb[:,j:]-=er.unsqueeze(1)*Hb[j,j:].unsqueeze(0)
            if i2<cols:
                Ww[:,i2:]-=Er@Hi[i1:i2,i2:]
        recon=Q.float()*sf[:,None]
        mse=(Wp-recon).pow(2).mean().item()
        if mse<be:
            bq,bsc,be=Q,s,mse
    return bq[:,invperm],bsc
def _gptq_mq6(sd,cats,hessians):
    res,meta={},{}
    gc,fc=0,0
    for name,tensor in sd.items():
        t=tensor.detach().cpu().contiguous()
        cat=classify_param(name)
        if not t.is_floating_point() or t.numel()<=65536:
            res[name]=t.to(torch.float16) if t.is_floating_point() else t
            meta[name]="passthrough"
            continue
        if any(p in name for p in CTRL_PAT):
            res[name]=t.float()
            meta[name]="passthrough_ctrl"
            continue
        if cat in cats and t.ndim==2:
            if name in hessians:
                q,s=_gptq_qw(t,hessians[name])
                gc+=1
                meta[name]={"type":"int6","method":"gptq"}
            else:
                q,s=_qi6(t)
                fc+=1
                meta[name]={"type":"int6","method":"clip_search"}
            res[name+".q"]=q
            res[name+".scale"]=s
        elif cat in cats and t.ndim>=1:
            q,s=_qi6(t)
            res[name+".q"]=q
            res[name+".scale"]=s
            meta[name]={"type":"int6"}
        else:
            q,s=_qft(t)
            res[name+".q"]=q
            res[name+".scale"]=s
            meta[name]={"type":"int8"}
    log(f"GPTQ quantization: {gc} layers with full GPTQ, {fc} fallback to clip-search")
    return res,meta
def _mq6(sd,cats):
    res,meta={},{}
    for name,tensor in sd.items():
        t=tensor.detach().cpu().contiguous()
        cat=classify_param(name)
        if not t.is_floating_point() or t.numel()<=65536:
            res[name]=t.to(torch.float16) if t.is_floating_point() else t
            meta[name]="passthrough"
            continue
        if any(p in name for p in CTRL_PAT):
            res[name]=t.float()
            meta[name]="passthrough_ctrl"
            continue
        if cat in cats and t.ndim>=1:
            q,s=_qi6(t)
            res[name+".q"]=q
            res[name+".scale"]=s
            meta[name]={"type":"int6"}
        else:
            q,s=_qft(t)
            res[name+".q"]=q
            res[name+".scale"]=s
            meta[name]={"type":"int8"}
    return res,meta
def _deq6(res,meta,tsd):
    out={}
    for name,orig in tsd.items():
        info=meta.get(name)
        if info is None:
            continue
        od=orig.dtype
        if info in ("passthrough","passthrough_ctrl","passthrough_fp16"):
            t=res[name]
            if t.dtype==torch.float16 and od in (torch.float32,torch.bfloat16):
                t=t.to(od)
            out[name]=t
            continue
        q,s=res[name+".q"],res[name+".scale"]
        if s.ndim>0:
            out[name]=(q.float()*s.float().view(q.shape[0],*([1]*(q.ndim-1)))).to(od)
        else:
            out[name]=(q.float()*float(s.item())).to(od)
    return out
_BSHF=b"BSHF"
def _bshuf(data,stride=2):
    if stride<=1 or len(data)<stride:
        return data
    src=np.frombuffer(data,dtype=np.uint8)
    n=len(src)
    out=np.empty(n,dtype=np.uint8)
    d=0
    for pos in range(stride):
        ch=src[pos::stride]
        out[d:d+len(ch)]=ch
        d+=len(ch)
    return _BSHF+bytes([stride])+out.tobytes()
def _bunshuf(data):
    if len(data)<5 or data[:4]!=_BSHF:
        return data
    stride=data[4]
    if stride<2:
        return data[5:]
    pay=np.frombuffer(data,dtype=np.uint8,offset=5)
    n=len(pay)
    out=np.empty(n,dtype=np.uint8)
    s=0
    for pos in range(stride):
        cl=n//stride+(1 if pos<n%stride else 0)
        out[pos::stride][:cl]=pay[s:s+cl]
        s+=cl
    return out.tobytes()
def _compress(data,comp,bshuf=True):
    if bshuf:
        data=_bshuf(data)
    if comp=="lzma":
        return lzma.compress(data,preset=6)
    elif comp=="brotli":
        import brotli
        return brotli.compress(data,quality=11)
    raise ValueError(f"Unknown compressor: {comp!r}")
def _decompress(data,comp,bshuf=True):
    if comp=="lzma":
        raw=lzma.decompress(data)
    elif comp=="brotli":
        import brotli
        raw=brotli.decompress(data)
    else:
        raise ValueError(f"Unknown compressor: {comp!r}")
    if bshuf:
        raw=_bunshuf(raw)
    return raw
def serialize(h,base_model,code):
    cb=len(code.encode("utf-8"))
    if h.is_main_process:
        torch.save(base_model.state_dict(),h.model_path)
        mb=os.path.getsize(h.model_path)
        log(f"Serialized model: {mb} bytes")
        log(f"Code size: {cb} bytes")
    sd={k:v.detach().cpu() for k,v in base_model.state_dict().items()}
    if h.gptq_enabled:
        log("GPTQ:collecting Hessians from calibration data...")
        t0=time.perf_counter()
        cl=DistributedTokenLoader(h.train_files,h.rank,h.world_size,torch.device("cuda",h.local_rank))
        H=_collect_H(base_model,cl,h,torch.device("cuda",h.local_rank),ncb=h.gptq_calibration_batches)
        log(f"GPTQ:collected {len(H)} Hessians in {time.perf_counter()-t0:.1f}s")
        qr,qm=_gptq_mq6(sd,{"mlp","attn"},H)
    else:
        qr,qm=_mq6(sd,{"mlp","attn"})
    buf=io.BytesIO()
    torch.save({"w":qr,"m":qm},buf)
    blob=_compress(buf.getvalue(),h.compressor)
    qfb=len(blob)
    if h.is_main_process:
        with open(h.quantized_model_path,"wb") as f:
            f.write(blob)
        log(f"Serialized model int6+{h.compressor}: {qfb} bytes")
        log(f"Total submission size int6+{h.compressor}: {qfb+cb} bytes")
def deserialize(h,device):
    em=GPT(h).to(device).bfloat16()
    restore_fp32_params(em)
    sd={k:v.detach().cpu() for k,v in em.state_dict().items()}
    with open(h.quantized_model_path,"rb") as f:
        blob=f.read()
    qs=torch.load(io.BytesIO(_decompress(blob,h.compressor)),map_location="cpu")
    ds=_deq6(qs["w"],qs["m"],sd)
    em.load_state_dict(ds,strict=True)
    return em
def _loss_bpb(ls,tc,bc):
    vl=(ls/tc).item()
    return vl,vl/math.log(2.0)*(tc.item()/bc.item())
def eval_val(h,device,vd,model):
    sl=h.eval_seq_len
    lbt=h.val_batch_tokens//(h.world_size*h.grad_accum_steps)
    assert lbt>=sl,"VAL_BATCH_SIZE too small"
    lbs=lbt//sl
    ts=(vd.val_tokens.numel()-1)//sl
    ss=(ts*h.rank)//h.world_size
    se=(ts*(h.rank+1))//h.world_size
    ls=torch.zeros((),device=device,dtype=torch.float64)
    tc=torch.zeros((),device=device,dtype=torch.float64)
    bc=torch.zeros((),device=device,dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bs in range(ss,se,lbs):
            be=min(bs+lbs,se)
            rs=bs*sl
            re=be*sl+1
            loc=vd.val_tokens[rs:re].to(device=device,dtype=torch.int64,non_blocking=True)
            x=loc[:-1].reshape(-1,sl)
            y=loc[1:].reshape(-1,sl)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):
                bl=model(x,y).detach()
            btc=float(y.numel())
            ls+=bl.to(torch.float64)*btc
            tc+=btc
            pi=x.reshape(-1)
            ti=y.reshape(-1)
            tb=vd.base_bytes_lut[ti].to(dtype=torch.int16)
            tb+=(vd.has_leading_space_lut[ti]&~vd.is_boundary_token_lut[pi]).to(dtype=torch.int16)
            bc+=tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in [ls,tc,bc]:
            dist.all_reduce(t,op=dist.ReduceOp.SUM)
    model.train()
    return _loss_bpb(ls,tc,bc)
def _sw_setup(h,vd):
    sl=h.eval_seq_len
    cs=sl-h.eval_stride
    tt=vd.val_tokens.numel()-1
    ws=[w for w in range(0,tt,h.eval_stride) if w+cs<tt]
    tw=len(ws)
    ms=(tw*h.rank)//h.world_size
    me=(tw*(h.rank+1))//h.world_size
    return sl,cs,tt,ws[ms:me]
def _sw_score(vd,nll,x_batch,y_batch,batch_ws,wlens,cs,ls,tc,bc):
    for i,ws in enumerate(batch_ws):
        wl=wlens[i]
        s=0 if ws==0 else cs
        sn=nll[i,s:wl].to(torch.float64)
        ls+=sn.sum()
        tc+=float(wl-s)
        tgt=y_batch[i,s:wl]
        prev=x_batch[i,s:wl]
        tb=vd.base_bytes_lut[tgt].to(torch.float64)
        tb+=(vd.has_leading_space_lut[tgt]&~vd.is_boundary_token_lut[prev]).to(torch.float64)
        bc+=tb.sum()
def _sw_fill(vd,batch_ws,sl,tt,device):
    bsz=len(batch_ws)
    xb=torch.zeros(bsz,sl,dtype=torch.int64,device=device)
    yb=torch.zeros(bsz,sl,dtype=torch.int64,device=device)
    wlens=[]
    for i,ws in enumerate(batch_ws):
        we=min(ws+sl,tt)
        wl=we-ws
        wlens.append(wl)
        ch=vd.val_tokens[ws:we+1].to(dtype=torch.int64,device=device)
        xb[i,:wl]=ch[:-1]
        yb[i,:wl]=ch[1:]
    return xb,yb,wlens,bsz
def _dist_reduce(*tensors):
    if dist.is_available() and dist.is_initialized():
        for t in tensors:
            dist.all_reduce(t,op=dist.ReduceOp.SUM)
def eval_val_sliding(h,device,vd,bm,batch_seqs=32):
    bm.eval()
    lfn=torch.compile(bm.forward_logits,dynamic=False,fullgraph=True)
    sl,cs,tt,mw=_sw_setup(h,vd)
    ls=torch.zeros((),device=device,dtype=torch.float64)
    tc=torch.zeros((),device=device,dtype=torch.float64)
    bc=torch.zeros((),device=device,dtype=torch.float64)
    with torch.inference_mode():
        for bi in range(0,len(mw),batch_seqs):
            bws=mw[bi:bi+batch_seqs]
            xb,yb,wlens,bsz=_sw_fill(vd,bws,sl,tt,device)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                logits=lfn(xb)
            nll=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),yb.reshape(-1),reduction="none").reshape(bsz,sl)
            _sw_score(vd,nll,xb,yb,bws,wlens,cs,ls,tc,bc)
    _dist_reduce(ls,tc,bc)
    bm.train()
    return _loss_bpb(ls,tc,bc)
def timed_eval(label,fn,*args,**kwargs):
    torch.cuda.synchronize()
    t0=time.perf_counter()
    vl,vb=fn(*args,**kwargs)
    torch.cuda.synchronize()
    ms=1000.0*(time.perf_counter()-t0)
    log(f"{label} val_loss:{vl:.8f} val_bpb:{vb:.8f} eval_time:{ms:.0f}ms")
    return vl,vb
def eval_val_slot(h,device,vd,bm,batch_seqs=32):
    bm.eval()
    sl,cs,tt,mw=_sw_setup(h,vd)
    ls=torch.zeros((),device=device,dtype=torch.float64)
    tc=torch.zeros((),device=device,dtype=torch.float64)
    bc=torch.zeros((),device=device,dtype=torch.float64)
    dm=h.model_dim
    for bi in range(0,len(mw),batch_seqs):
        bws=mw[bi:bi+batch_seqs]
        xb,yb,wlens,bsz=_sw_fill(vd,bws,sl,tt,device)
        with torch.no_grad(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            hid=bm.forward_backbone(xb).detach()
        delta=torch.zeros(bsz,1,dm,device=device,dtype=hid.dtype,requires_grad=True)
        sopt=torch.optim.AdamW([delta],lr=h.slot_lr)
        for _ in range(h.slot_steps):
            sopt.zero_grad()
            logits=bm._head_logits(hid+delta)
            sl_loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),yb.reshape(-1),reduction="mean")
            sl_loss.backward()
            sopt.step()
        with torch.no_grad():
            logits=bm._head_logits(hid+delta)
            nll=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),yb.reshape(-1),reduction="none").reshape(bsz,sl)
            _sw_score(vd,nll,xb,yb,bws,wlens,cs,ls,tc,bc)
    _dist_reduce(ls,tc,bc)
    bm.train()
    return _loss_bpb(ls,tc,bc)
def run_evals(h,device,vd,em):
    cm=torch.compile(em,dynamic=False,fullgraph=True)
    timed_eval("final_int6_roundtrip",eval_val,h,device,vd,cm)
    if h.sliding_window_enabled:
        timed_eval("final_int6_sliding_window",eval_val_sliding,h,device,vd,em)
    if h.slot_enabled:
        timed_eval("final_int6_slot",eval_val_slot,h,device,vd,em)
class Optimizers():
    def __init__(self,h,bm):
        bnp=list(bm.blocks.named_parameters())
        mp=[p for n,p in bnp if p.ndim==2 and not any(pt in n for pt in CTRL_PAT)]
        sp=[p for n,p in bnp if p.ndim<2 or any(pt in n for pt in CTRL_PAT)]
        if bm.skip_weights.numel()>0:
            sp.append(bm.skip_weights)
        if bm.skip_gates is not None and bm.skip_gates.numel()>0:
            sp.append(bm.skip_gates)
        tlr=h.tied_embed_lr if h.tie_embeddings else h.embed_lr
        tp=[{"params":[bm.tok_emb.weight],"lr":tlr,"base_lr":tlr}]
        if bm.ve_shared is not None:
            tp.append({"params":[bm.ve_shared.embed.weight],"lr":tlr,"base_lr":tlr})
            if bm.ve_shared.proj is not None:
                mp.append(bm.ve_shared.proj.weight)
            sp.append(bm.ve_shared.scale)
            for s in bm.ve_layer_scales:
                sp.append(s)
        self.optimizer_tok=torch.optim.AdamW(tp,betas=(h.beta1,h.beta2),eps=h.adam_eps,weight_decay=h.embed_wd,fused=True)
        self.optimizer_muon=Muon(mp,lr=h.matrix_lr,momentum=h.muon_momentum,backend_steps=h.muon_backend_steps,weight_decay=h.muon_wd)
        for g in self.optimizer_muon.param_groups:
            g["base_lr"]=h.matrix_lr
        self.optimizer_scalar=torch.optim.AdamW([{"params":sp,"lr":h.scalar_lr,"base_lr":h.scalar_lr}],betas=(h.beta1,h.beta2),eps=h.adam_eps,weight_decay=h.adam_wd,fused=True)
        self.optimizers=[self.optimizer_tok,self.optimizer_muon,self.optimizer_scalar]
        if bm.lm_head is not None:
            self.optimizer_head=torch.optim.Adam([{"params":[bm.lm_head.weight],"lr":h.head_lr,"base_lr":h.head_lr}],betas=(h.beta1,h.beta2),eps=h.adam_eps,fused=True)
            self.optimizers.insert(1,self.optimizer_head)
        else:
            self.optimizer_head=None
    def __iter__(self):
        return iter(self.optimizers)
    def zero_grad_all(self):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)
    def step(self):
        for opt in self.optimizers:
            opt.step()
        self.zero_grad_all()
def train_model(h,device,vd):
    bm=GPT(h).to(device).bfloat16()
    restore_fp32_params(bm)
    cm=torch.compile(bm,dynamic=False,fullgraph=True)
    if h.distributed:
        model=DDP(cm,device_ids=[h.local_rank],broadcast_buffers=False)
    else:
        model=cm
    log(f"model_params:{sum(p.numel() for p in bm.parameters())}")
    opts=Optimizers(h,bm)
    loader=DistributedTokenLoader(h.train_files,h.rank,h.world_size,device)
    mwms=1000.0*h.max_wallclock_seconds if h.max_wallclock_seconds>0 else None
    if h.gptq_enabled and mwms is not None:
        mwms-=h.gptq_reserve_seconds*1000.0
        log(f"gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={mwms:.0f}ms")
    def tfrac(step,ems):
        if mwms is None:
            return step/max(h.iterations,1)
        return ems/max(mwms,1e-9)
    def lrmul(frac):
        if h.warmdown_frac<=0:
            return 1.0
        if frac>=1.0-h.warmdown_frac:
            return max((1.0-frac)/h.warmdown_frac,h.min_lr)
        return 1.0
    def sfn(step,lrs):
        opts.zero_grad_all()
        tl=torch.zeros((),device=device)
        for ms in range(h.grad_accum_steps):
            if h.distributed:
                model.require_backward_grad_sync=ms==h.grad_accum_steps-1
            x,y=loader.next_batch(h.train_batch_tokens,h.train_seq_len,h.grad_accum_steps)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):
                loss=model(x,y)
            tl+=loss.detach()
            (loss/h.grad_accum_steps).backward()
        tl/=h.grad_accum_steps
        fr=min(step/h.muon_momentum_warmup_steps,1.0) if h.muon_momentum_warmup_steps>0 else 1.0
        mm=(1-fr)*h.muon_momentum_warmup_start+fr*h.muon_momentum
        for g in opts.optimizer_muon.param_groups:
            g["momentum"]=mm
        for opt in opts:
            for g in opt.param_groups:
                g["lr"]=g["base_lr"]*lrs
        if h.grad_clip_norm>0:
            torch.nn.utils.clip_grad_norm_(bm.parameters(),h.grad_clip_norm)
        opts.step()
        return tl
    if h.warmup_steps>0:
        ims={n:t.detach().cpu().clone() for n,t in bm.state_dict().items()}
        ios=[copy.deepcopy(opt.state_dict()) for opt in opts]
        model.train()
        for ws in range(h.warmup_steps):
            sfn(ws,1.0)
            if ws<=5 or (ws+1)%10==0 or ws+1==h.warmup_steps:
                log(f"warmup_step: {ws+1}/{h.warmup_steps}")
        bm.load_state_dict(ims,strict=True)
        for opt,st in zip(opts,ios,strict=True):
            opt.load_state_dict(st)
        opts.zero_grad_all()
        if h.distributed:
            model.require_backward_grad_sync=True
        loader=DistributedTokenLoader(h.train_files,h.rank,h.world_size,device)
    ema={n:t.detach().float().clone() for n,t in bm.state_dict().items()}
    ed=h.ema_decay
    ttms=0.0
    sas=None
    torch.cuda.synchronize()
    t0=time.perf_counter()
    step=0
    while True:
        last=step==h.iterations or (sas is not None and step>=sas)
        sv=last or (h.val_loss_every>0 and step%h.val_loss_every==0)
        if sv:
            torch.cuda.synchronize()
            ttms+=1000.0*(time.perf_counter()-t0)
            vl,vb=eval_val(h,device,vd,model)
            log(f"{step}/{h.iterations} val_loss: {vl:.4f} val_bpb: {vb:.4f}")
            torch.cuda.synchronize()
            t0=time.perf_counter()
        if last:
            if sas is not None and step<h.iterations:
                log(f"stopping_early: wallclock_cap train_time: {ttms:.0f}ms step: {step}/{h.iterations}")
            break
        ems=ttms+1000.0*(time.perf_counter()-t0)
        frac=tfrac(step,ems)
        tl=sfn(step,lrmul(frac))
        with torch.no_grad():
            for n,t in bm.state_dict().items():
                ema[n].mul_(ed).add_(t.detach().float(),alpha=1.0-ed)
        step+=1
        atms=ttms+1000.0*(time.perf_counter()-t0)
        slt=h.train_log_every>0 and (step<=5 or step%h.train_log_every==0 or sas is not None)
        if slt:
            tps=step*h.train_batch_tokens/(atms/1000.0)
            log(f"{step}/{h.iterations} train_loss: {tl.item():.4f} train_time: {atms/60000:.1f}m tok/s: {tps:.0f}")
        rc=mwms is not None and atms>=mwms
        if h.distributed and mwms is not None:
            rct=torch.tensor(int(rc),device=device)
            dist.all_reduce(rct,op=dist.ReduceOp.MAX)
            rc=bool(rct.item())
        if sas is None and rc:
            sas=step
    log(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB")
    log("ema:applying EMA weights")
    cs=bm.state_dict()
    avg={n:t.to(dtype=cs[n].dtype) for n,t in ema.items()}
    bm.load_state_dict(avg,strict=True)
    return bm,cm
def train_and_eval(h,device):
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    vd=ValidationData(h,device)
    log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}")
    log(f"val_tokens: {vd.val_tokens.numel()-1}")
    bm,cm=train_model(h,device,vd)
    timed_eval("pre-quantization post-ema",eval_val,h,device,vd,cm)
    serialize(h,bm,Path(__file__).read_text(encoding="utf-8"))
    if h.distributed:
        dist.barrier()
    em=deserialize(h,device)
    run_evals(h,device,vd,em)
def main():
    ws=int(_E("WORLD_SIZE","1"))
    lr=int(_E("LOCAL_RANK","0"))
    dd="RANK" in os.environ and "WORLD_SIZE" in os.environ
    assert torch.cuda.is_available(),"CUDA is required"
    assert ws>0 and 8%ws==0,f"WORLD_SIZE={ws} must be positive and divide 8"
    device=torch.device("cuda",lr)
    torch.cuda.set_device(device)
    if dd:
        dist.init_process_group(backend="nccl",device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import enable_cudnn_sdp,enable_flash_sdp,enable_math_sdp,enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp=False
    h=Hyperparameters()
    set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs("logs",exist_ok=True)
        log(100*"=",console=False)
        log("Hyperparameters:",console=True)
        for k,v in sorted(vars(type(h)).items()):
            if not k.startswith("_"):
                log(f"  {k}: {v}",console=True)
        log(Path(__file__).read_text(encoding="utf-8"),console=False)
        log("="*100,console=False)
        log(f"Running Python {sys.version}",console=False)
        log(f"Running PyTorch {torch.__version__}",console=False)
        log(subprocess.run(["nvidia-smi"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False).stdout,console=False)
        log("="*100,console=False)
    train_and_eval(h,device)
    if dd:
        dist.destroy_process_group()
if __name__=="__main__":
    main()
