_g='momentum'
_f='fineweb_train_*.bin'
_e='LOCAL_RANK'
_d='WARMUP_STEPS'
_c='passthrough_ctrl'
_b='passthrough'
_a='repeat_mlp'
_Z='disable_attn'
_Y='full'
_X='<u2'
_W='RANK'
_V='brotli'
_U='recur_layers'
_T='attn'
_S='<i4'
_R='WORLD_SIZE'
_Q='int6'
_P='params'
_O='mlp'
_N='utf-8'
_M='cuda'
_L='.scale'
_K='.q'
_J='type'
_I='base_lr'
_H='lr'
_G='none'
_F=.0
_E='1'
_D=1.
_C=False
_B=True
_A=None
import copy,glob,io,lzma,math,os
from pathlib import Path
import random,subprocess,sys,time,uuid,numpy as np,sentencepiece as spm,torch,torch.distributed as dist,torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor,nn
from flash_attn_interface import flash_attn_func as flash_attn_3_func
class Hyperparameters:data_dir=os.environ.get('DATA_DIR','./data/');seed=int(os.environ.get('SEED',1337));run_id=os.environ.get('RUN_ID',str(uuid.uuid4()));iterations=int(os.environ.get('ITERATIONS',20000));warmdown_frac=float(os.environ.get('WARMDOWN_FRAC',.667));warmup_steps=int(os.environ.get(_d,20));train_batch_tokens=int(os.environ.get('TRAIN_BATCH_TOKENS',786432));train_seq_len=int(os.environ.get('TRAIN_SEQ_LEN',2048));eval_seq_len=int(os.environ.get('EVAL_SEQ_LEN',2048));max_wallclock_seconds=float(os.environ.get('MAX_WALLCLOCK_SECONDS',6e2));train_log_every=int(os.environ.get('TRAIN_LOG_EVERY',500));val_batch_tokens=int(os.environ.get('VAL_BATCH_TOKENS',524288));val_loss_every=int(os.environ.get('VAL_LOSS_EVERY',4000));sliding_window_enabled=bool(int(os.environ.get('SLIDING_WINDOW_ENABLED',_E)));vocab_size=int(os.environ.get('VOCAB_SIZE',4096));num_layers=int(os.environ.get('NUM_LAYERS',11));xsa_last_n=int(os.environ.get('XSA_LAST_N',11));num_kv_heads=int(os.environ.get('NUM_KV_HEADS',4));model_dim=int(os.environ.get('MODEL_DIM',512));embedding_dim=int(os.environ.get('EMBEDDING_DIM',512));num_heads=int(os.environ.get('NUM_HEADS',8));mlp_mult=float(os.environ.get('MLP_MULT',4.));skip_gates_enabled=bool(int(os.environ.get('SKIP_GATES_ENABLED',_E)));tie_embeddings=bool(int(os.environ.get('TIE_EMBEDDINGS',_E)));logit_softcap=float(os.environ.get('LOGIT_SOFTCAP',3e1));rope_base=float(os.environ.get('ROPE_BASE',1e4));rope_dims=int(os.environ.get('ROPE_DIMS',16));rope_train_seq_len=int(os.environ.get('ROPE_TRAIN_SEQ_LEN',2048));ln_scale=bool(int(os.environ.get('LN_SCALE',_E)));ve_enabled=bool(int(os.environ.get('VE_ENABLED',_E)));ve_dim=int(os.environ.get('VE_DIM',128));ve_layers=os.environ.get('VE_LAYERS','9,10');qk_gain_init=float(os.environ.get('QK_GAIN_INIT',5.));min_lr=float(os.environ.get('MIN_LR',_F));embed_lr=float(os.environ.get('EMBED_LR',.6));head_lr=float(os.environ.get('HEAD_LR',.008));tied_embed_lr=float(os.environ.get('TIED_EMBED_LR',.03));tied_embed_init_std=float(os.environ.get('TIED_EMBED_INIT_STD',.005));matrix_lr=float(os.environ.get('MATRIX_LR',.02));scalar_lr=float(os.environ.get('SCALAR_LR',.02));muon_momentum=float(os.environ.get('MUON_MOMENTUM',.99));muon_backend_steps=int(os.environ.get('MUON_BACKEND_STEPS',5));muon_momentum_warmup_start=float(os.environ.get('MUON_MOMENTUM_WARMUP_START',.92));muon_momentum_warmup_steps=int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS',1500));beta1=float(os.environ.get('BETA1',.9));beta2=float(os.environ.get('BETA2',.95));adam_eps=float(os.environ.get('ADAM_EPS',1e-08));grad_clip_norm=float(os.environ.get('GRAD_CLIP_NORM',.3));eval_stride=int(os.environ.get('EVAL_STRIDE',64));muon_beta2=float(os.environ.get('MUON_BETA2',.95));adam_wd=float(os.environ.get('ADAM_WD',.02));muon_wd=float(os.environ.get('MUON_WD',.085));embed_wd=float(os.environ.get('EMBED_WD',.085));ema_decay=float(os.environ.get('EMA_DECAY',.997));parallel_residual=bool(int(os.environ.get('PARALLEL_RESIDUAL','0')));parallel_start_layer=int(os.environ.get('PARALLEL_START_LAYER','7'));parallel_start_layer_is_physical=bool(int(os.environ.get('PARALLEL_START_LAYER_IS_PHYSICAL',_E)));recur_layers_str=os.environ.get('RECUR_LAYERS','4,5').strip();recur_start_step=int(os.environ.get('RECUR_START_STEP','3000'));recur_warmup_steps=int(os.environ.get('RECUR_WARMUP_STEPS',str(int(os.environ.get(_d,20)))));repeat_untie_mlp=os.environ.get('REPEAT_UNTIE_MLP',_G).strip().lower();repeat_untie_mlp_layers=os.environ.get('REPEAT_UNTIE_MLP_LAYERS','').strip();disable_layer0_attn=bool(int(os.environ.get('DISABLE_LAYER0_ATTN','0')));mixed_quant=bool(int(os.environ.get('MIXED_QUANT','0')));n_int6_layers=int(os.environ.get('N_INT6_LAYERS','32'));compressor=os.environ.get('COMPRESSOR',_V);pre_quant_ttt_enabled=bool(int(os.environ.get('PRE_QUANT_TTT','0')));pre_quant_ttt_lr=float(os.environ.get('PRE_QUANT_TTT_LR','0.001'));pre_quant_ttt_epochs=int(os.environ.get('PRE_QUANT_TTT_EPOCHS','1'));pre_quant_ttt_freeze_blocks=int(os.environ.get('PRE_QUANT_TTT_FREEZE','10'));pre_quant_ttt_chunk_tokens=int(os.environ.get('PRE_QUANT_TTT_CHUNK','32768'));etlb_enabled=bool(int(os.environ.get('ETLB_ENABLED','0')));etlb_lr=float(os.environ.get('ETLB_LR','0.05'));etlb_steps=int(os.environ.get('ETLB_STEPS','5'));etlb_clip=float(os.environ.get('ETLB_CLIP','3.0'));gptq_enabled=bool(int(os.environ.get('GPTQ_ENABLED',_E)));gptq_calibration_batches=int(os.environ.get('GPTQ_CALIBRATION_BATCHES',64));gptq_reserve_seconds=float(os.environ.get('GPTQ_RESERVE_SECONDS',1e1));distributed=_W in os.environ and _R in os.environ;rank=int(os.environ.get(_W,'0'));world_size=int(os.environ.get(_R,_E));local_rank=int(os.environ.get(_e,'0'));is_main_process=rank==0;grad_accum_steps=8//world_size;datasets_dir=os.path.join(data_dir,'datasets',f"fineweb10B_sp{vocab_size}");train_files=os.path.join(datasets_dir,_f);val_files=os.path.join(datasets_dir,'fineweb_val_*.bin');tokenizer_path=os.path.join(data_dir,'tokenizers',f"fineweb_{vocab_size}_bpe.model");logfile=f"logs/{run_id}.txt";model_path='final_model.pt';quantized_model_path='final_model.int6.ptz'
_logger_hparams=_A
def set_logging_hparams(h):global _logger_hparams;_logger_hparams=h
def log(msg,console=_B):
	if _logger_hparams is _A:print(msg)
	if _logger_hparams.is_main_process:
		if console:print(msg)
		if _logger_hparams.logfile is not _A:
			with open(_logger_hparams.logfile,'a',encoding=_N)as f:print(msg,file=f)
class ValidationData:
	def __init__(self,h,device):
		if not h.tokenizer_path.endswith('.model'):raise ValueError(f"Script only setup for SentencePiece .model file: {h.tokenizer_path}")
		self.sp=spm.SentencePieceProcessor(model_file=h.tokenizer_path)
		if int(self.sp.vocab_size())!=h.vocab_size:raise ValueError(f"VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={int(self.sp.vocab_size())}")
		self.val_tokens=load_validation_tokens(h.val_files,h.eval_seq_len);self.base_bytes_lut,self.has_leading_space_lut,self.is_boundary_token_lut=build_sentencepiece_luts(self.sp,h.vocab_size,device)
def build_sentencepiece_luts(sp,vocab_size,device):
	sp_vocab_size=int(sp.vocab_size());table_size=max(sp_vocab_size,vocab_size);base_bytes_np=np.zeros((table_size,),dtype=np.int16);has_leading_space_np=np.zeros((table_size,),dtype=np.bool_);is_boundary_token_np=np.ones((table_size,),dtype=np.bool_)
	for token_id in range(sp_vocab_size):
		if sp.is_control(token_id)or sp.is_unknown(token_id)or sp.is_unused(token_id):continue
		is_boundary_token_np[token_id]=_C
		if sp.is_byte(token_id):base_bytes_np[token_id]=1;continue
		piece=sp.id_to_piece(token_id)
		if piece.startswith('▁'):has_leading_space_np[token_id]=_B;piece=piece[1:]
		base_bytes_np[token_id]=len(piece.encode(_N))
	return torch.tensor(base_bytes_np,dtype=torch.int16,device=device),torch.tensor(has_leading_space_np,dtype=torch.bool,device=device),torch.tensor(is_boundary_token_np,dtype=torch.bool,device=device)
def load_validation_tokens(pattern,seq_len):
	files=[Path(p)for p in sorted(glob.glob(pattern))]
	if not files:raise FileNotFoundError(f"No files found for pattern: {pattern}")
	tokens=torch.cat([load_data_shard(file)for file in files]).contiguous();usable=(tokens.numel()-1)//seq_len*seq_len
	if usable<=0:raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
	return tokens[:usable+1]
def load_data_shard(file):
	header_bytes=256*np.dtype(_S).itemsize;token_bytes=np.dtype(_X).itemsize;header=np.fromfile(file,dtype=_S,count=256)
	if header.size!=256 or int(header[0])!=20240520 or int(header[1])!=1:raise ValueError(f"Unexpected shard header for {file}")
	num_tokens=int(header[2]);expected_size=header_bytes+num_tokens*token_bytes
	if file.stat().st_size!=expected_size:raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
	tokens_np=np.fromfile(file,dtype=_X,count=num_tokens,offset=header_bytes)
	if tokens_np.size!=num_tokens:raise ValueError(f"Short read for {file}")
	return torch.from_numpy(tokens_np.astype(np.uint16,copy=_C))
_SHARD_HEADER_BYTES=256*np.dtype(_S).itemsize
_SHARD_NTOKENS_CACHE={}
_MMAP_CACHE={}
def _read_num_tokens(file):
	key=str(file);cached=_SHARD_NTOKENS_CACHE.get(key)
	if cached is not _A:return cached
	header=np.fromfile(file,dtype=_S,count=256)
	if header.size!=256 or int(header[0])!=20240520 or int(header[1])!=1:raise ValueError(f"Unexpected shard header for {file}")
	n=int(header[2]);_SHARD_NTOKENS_CACHE[key]=n;return n
def _get_shard_memmap(file):
	key=str(file);mm=_MMAP_CACHE.get(key)
	if mm is not _A:return mm
	n=_read_num_tokens(file);mm=np.memmap(file,mode='r',dtype=_X,offset=_SHARD_HEADER_BYTES,shape=(n,));_MMAP_CACHE[key]=mm;return mm
class DistributedTokenLoader:
	def __init__(self,pattern,rank,world_size,device):
		self.rank=rank;self.world_size=world_size;self.device=device;self.files=[Path(p)for p in sorted(glob.glob(pattern))]
		if not self.files:raise FileNotFoundError(f"No files found for pattern: {pattern}")
		self._num_tokens=np.array([_read_num_tokens(f)for f in self.files],dtype=np.int64);seed=0
		for f in self.files:
			for b in str(f).encode():seed=(seed^b)*1099511628211&0xffffffffffffffff
		self._rng=np.random.Generator(np.random.PCG64(seed));self._cfg=_A;self._eligible_shards=_A;self._base_block_counts=_A;n=len(self.files);self._cursor_phase=np.zeros(n,dtype=np.int64);self._cursor_block_count=np.zeros(n,dtype=np.int64);self._cursor_next=np.zeros(n,dtype=np.int64);self._cursor_start=np.zeros(n,dtype=np.int64);self._cursor_stride=np.ones(n,dtype=np.int64);self._cursor_init=np.zeros(n,dtype=np.bool_);self._batches_built=0
	def _pick_coprime_stride(self,n):
		if n<=1:return 1
		while _B:
			s=int(self._rng.integers(1,n))
			if math.gcd(s,n)==1:return s
	def _reset_cursor(self,si,seq_len):nt=int(self._num_tokens[si]);max_phase=min(seq_len-1,max(0,nt-seq_len-1));phase=int(self._rng.integers(max_phase+1))if max_phase>0 else 0;bc=(nt-1-phase)//seq_len;self._cursor_phase[si]=phase;self._cursor_block_count[si]=bc;self._cursor_next[si]=0;self._cursor_start[si]=int(self._rng.integers(bc))if bc>1 else 0;self._cursor_stride[si]=self._pick_coprime_stride(bc);self._cursor_init[si]=_B
	def _ensure_cursor(self,si,seq_len):
		if not self._cursor_init[si]or self._cursor_next[si]>=self._cursor_block_count[si]:self._reset_cursor(si,seq_len)
	def _take_from_shard(self,si,seq_len,count,out):
		rem=count
		while rem>0:
			self._ensure_cursor(si,seq_len);bc=int(self._cursor_block_count[si]);ni=int(self._cursor_next[si]);take=min(rem,bc-ni);phase=int(self._cursor_phase[si]);start=int(self._cursor_start[si]);stride=int(self._cursor_stride[si])
			for j in range(take):bi=(start+(ni+j)*stride)%bc;out.append((si,phase+bi*seq_len))
			self._cursor_next[si]=ni+take;rem-=take
	def _init_pipeline(self,global_tokens,seq_len,grad_accum_steps):local_tokens=global_tokens//(self.world_size*grad_accum_steps);num_seqs=local_tokens//seq_len;global_num_seqs=num_seqs*self.world_size;self._cfg=local_tokens,seq_len,num_seqs,global_num_seqs;bbc=(self._num_tokens-1)//seq_len;eligible=bbc>0;self._eligible_shards=np.nonzero(eligible)[0].astype(np.int64);self._base_block_counts=bbc[self._eligible_shards].astype(np.int64)
	def _sample_global_windows(self):
		_,seq_len,_,gns=self._cfg;ec=int(self._eligible_shards.size);progress=min(self._batches_built/18e2,_D);remaining=np.empty(ec,dtype=np.float64)
		for(i,si)in enumerate(self._eligible_shards.tolist()):
			if self._cursor_init[si]:r=int(self._cursor_block_count[si])-int(self._cursor_next[si]);remaining[i]=float(max(r,1))
			else:remaining[i]=float(self._base_block_counts[i])
		alpha=.9-.4*progress;weights=np.power(remaining,alpha);ws=float(weights.sum())
		if not np.isfinite(ws)or ws<=_F:weights=np.ones(ec,dtype=np.float64);ws=float(weights.sum())
		probs=weights/ws;low=min(max(8,self.world_size),ec,gns);high=min(max(32,self.world_size*8),ec,gns);mix=max(1,min(int(round(low+progress*(high-low))),ec,gns));cp=self._rng.choice(ec,size=mix,replace=_C,p=probs);cs=self._eligible_shards[cp];cpr=probs[cp].copy();cpr/=cpr.sum();counts=np.ones(mix,dtype=np.int64);extra=gns-mix
		if extra>0:counts+=self._rng.multinomial(extra,cpr).astype(np.int64)
		perm=self._rng.permutation(mix);cs,counts=cs[perm],counts[perm];buckets=[]
		for(si,cnt)in zip(cs.tolist(),counts.tolist()):
			b=[];self._take_from_shard(int(si),seq_len,int(cnt),b)
			if b:
				if len(b)>1:bp=self._rng.permutation(len(b));b=[b[int(k)]for k in bp.tolist()]
				buckets.append(b)
		windows=[];active=[i for(i,bk)in enumerate(buckets)if bk]
		while active:
			order=self._rng.permutation(len(active));new_active=[]
			for oi in order.tolist():
				bi=active[oi]
				if buckets[bi]:windows.append(buckets[bi].pop())
				if buckets[bi]:new_active.append(bi)
			active=new_active
		return windows
	def next_batch(self,global_tokens,seq_len,grad_accum_steps):
		if self._cfg is _A:self._init_pipeline(global_tokens,seq_len,grad_accum_steps)
		_,_,num_seqs,_=self._cfg;gw=self._sample_global_windows();local_w=gw[self.rank::self.world_size];x=torch.empty((num_seqs,seq_len),dtype=torch.int64);y=torch.empty((num_seqs,seq_len),dtype=torch.int64)
		for(bi,(si,pos))in enumerate(local_w):mm=_get_shard_memmap(self.files[si]);window=torch.as_tensor(np.array(mm[pos:pos+seq_len+1],dtype=np.int64));x[bi]=window[:-1];y[bi]=window[1:]
		self._batches_built+=1;return x.to(self.device,non_blocking=_B),y.to(self.device,non_blocking=_B)
class RMSNorm(nn.Module):
	def __init__(self,eps=_A):super().__init__();self.eps=eps
	def forward(self,x):return F.rms_norm(x,(x.size(-1),),eps=self.eps)
class CastedLinear(nn.Linear):
	def forward(self,x):w=self.weight.to(x.dtype);bias=self.bias.to(x.dtype)if self.bias is not _A else _A;return F.linear(x,w,bias)
class Rotary(nn.Module):
	def __init__(self,dim,base=1e4,train_seq_len=1024,rope_dims=0):super().__init__();self.dim=dim;self.base=base;self.train_seq_len=train_seq_len;self.rope_dims=rope_dims if rope_dims>0 else dim;inv_freq=_D/base**(torch.arange(0,self.rope_dims,2,dtype=torch.float32)/self.rope_dims);self.register_buffer('inv_freq',inv_freq,persistent=_C);self._seq_len_cached=0;self._cos_cached=_A;self._sin_cached=_A
	def forward(self,seq_len,device,dtype):
		if self._cos_cached is _A or self._sin_cached is _A or self._seq_len_cached!=seq_len or self._cos_cached.device!=device:
			rd=self.rope_dims
			if seq_len>self.train_seq_len:scale=seq_len/self.train_seq_len;new_base=self.base*scale**(rd/(rd-2));inv_freq=_D/new_base**(torch.arange(0,rd,2,dtype=torch.float32,device=device)/rd)
			else:inv_freq=self.inv_freq.to(device)
			t=torch.arange(seq_len,device=device,dtype=inv_freq.dtype);freqs=torch.outer(t,inv_freq);self._cos_cached=freqs.cos()[_A,:,_A,:];self._sin_cached=freqs.sin()[_A,:,_A,:];self._seq_len_cached=seq_len
		return self._cos_cached.to(dtype=dtype),self._sin_cached.to(dtype=dtype)
def apply_rotary_emb(x,cos,sin,rope_dims=0):
	if rope_dims>0 and rope_dims<x.size(-1):x_rope,x_pass=x[...,:rope_dims],x[...,rope_dims:];half=rope_dims//2;x1,x2=x_rope[...,:half],x_rope[...,half:];x_rope=torch.cat((x1*cos+x2*sin,x1*-sin+x2*cos),dim=-1);return torch.cat((x_rope,x_pass),dim=-1)
	half=x.size(-1)//2;x1,x2=x[...,:half],x[...,half:];return torch.cat((x1*cos+x2*sin,x1*-sin+x2*cos),dim=-1)
class CausalSelfAttention(nn.Module):
	def __init__(self,dim,num_heads,num_kv_heads,rope_base,qk_gain_init,train_seq_len):
		super().__init__()
		if dim%num_heads!=0:raise ValueError('model_dim must be divisible by num_heads')
		if num_heads%num_kv_heads!=0:raise ValueError('num_heads must be divisible by num_kv_heads')
		self.num_heads=num_heads;self.num_kv_heads=num_kv_heads;self.head_dim=dim//num_heads
		if self.head_dim%2!=0:raise ValueError('head_dim must be even for RoPE')
		kv_dim=self.num_kv_heads*self.head_dim;self.c_q=CastedLinear(dim,dim,bias=_C);self.c_k=CastedLinear(dim,kv_dim,bias=_C);self.c_v=CastedLinear(dim,kv_dim,bias=_C);self.proj=CastedLinear(dim,dim,bias=_C);self.proj._zero_init=_B;self.q_gain=nn.Parameter(torch.full((num_heads,),qk_gain_init,dtype=torch.float32));self.rope_dims=0;self.rotary=Rotary(self.head_dim,base=rope_base,train_seq_len=train_seq_len);self.use_xsa=_C
	def _xsa_efficient(self,y,v):B,T,H,D=y.shape;Hkv=v.size(-2);group=H//Hkv;y_g=y.reshape(B,T,Hkv,group,D);vn=F.normalize(v,dim=-1).unsqueeze(-2);proj=(y_g*vn).sum(dim=-1,keepdim=_B)*vn;return(y_g-proj).reshape(B,T,H,D)
	def forward(self,x,v_embed=_A):
		bsz,seqlen,dim=x.shape;q=self.c_q(x).reshape(bsz,seqlen,self.num_heads,self.head_dim);k=self.c_k(x).reshape(bsz,seqlen,self.num_kv_heads,self.head_dim);v=self.c_v(x)
		if v_embed is not _A:v=v+v_embed
		v=v.reshape(bsz,seqlen,self.num_kv_heads,self.head_dim);q=F.rms_norm(q,(q.size(-1),));k=F.rms_norm(k,(k.size(-1),));cos,sin=self.rotary(seqlen,x.device,q.dtype);q=apply_rotary_emb(q,cos,sin,self.rope_dims);k=apply_rotary_emb(k,cos,sin,self.rope_dims);q=q*self.q_gain.to(dtype=q.dtype)[_A,_A,:,_A];y=flash_attn_3_func(q,k,v,causal=_B)
		if self.use_xsa:y=self._xsa_efficient(y,v)
		y=y.reshape(bsz,seqlen,dim);return self.proj(y)
class ValueEmbedding(nn.Module):
	def __init__(self,vocab_size,ve_dim,model_dim):
		super().__init__();self.embed=nn.Embedding(vocab_size,ve_dim);nn.init.normal_(self.embed.weight,std=.01);self.proj=CastedLinear(ve_dim,model_dim,bias=_C)if ve_dim!=model_dim else _A
		if self.proj is not _A:nn.init.zeros_(self.proj.weight)
		self.scale=nn.Parameter(torch.tensor(.1,dtype=torch.float32))
	def forward(self,token_ids):
		h=self.embed(token_ids)
		if self.proj is not _A:h=self.proj(h)
		return h*self.scale.to(dtype=h.dtype)
class MLP(nn.Module):
	def __init__(self,dim,mlp_mult):super().__init__();hidden=int(mlp_mult*dim);self.fc=CastedLinear(dim,hidden,bias=_C);self.proj=CastedLinear(hidden,dim,bias=_C);self.proj._zero_init=_B
	def forward(self,x):return self.proj(F.leaky_relu(self.fc(x),negative_slope=.5).square())
class Block(nn.Module):
	def __init__(self,dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init,train_seq_len,layer_idx=0,ln_scale=_C):super().__init__();self.attn_norm=RMSNorm();self.mlp_norm=RMSNorm();self.attn=CausalSelfAttention(dim,num_heads,num_kv_heads,rope_base,qk_gain_init,train_seq_len);self.mlp=MLP(dim,mlp_mult);self.attn_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.mlp_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.resid_mix=nn.Parameter(torch.stack((torch.ones(dim),torch.zeros(dim))).float());self.ln_scale_factor=_D/math.sqrt(layer_idx+1)if ln_scale else _D
	def forward(self,x,x0,v_embed=_A):mix=self.resid_mix.to(dtype=x.dtype);x_in=mix[0][_A,_A,:]*x+mix[1][_A,_A,:]*x0;attn_out=self.attn(self.attn_norm(x_in)*self.ln_scale_factor,v_embed=v_embed);x_out=x_in+self.attn_scale.to(dtype=x_in.dtype)[_A,_A,:]*attn_out;x_out=x_out+self.mlp_scale.to(dtype=x_out.dtype)[_A,_A,:]*self.mlp(self.mlp_norm(x_out)*self.ln_scale_factor);return x_out
def _parse_layer_list(raw):return sorted({int(x)for x in raw.split(',')if x.strip()})
class RepeatMLPWeights(nn.Module):
	def __init__(self,dim,mlp_mult,mode):
		super().__init__();hidden=int(mlp_mult*dim);self.fc=nn.Linear(dim,hidden,bias=_C)if mode==_Y else _A;self.proj=nn.Linear(hidden,dim,bias=_C)if mode in(_Y,'down')else _A
		if self.proj is not _A:self.proj._zero_init=_B
	def forward(self,x):
		if self.fc is _A:raise RuntimeError('RepeatMLPWeights.forward requires fc to be present')
		h=F.leaky_relu(self.fc(x),negative_slope=.5).square()
		if self.proj is _A:return h
		return self.proj(h)
class GPT(nn.Module):
	def __init__(self,h):
		super().__init__();self._ve_target_dim=h.num_kv_heads*(h.model_dim//h.num_heads)
		if h.logit_softcap<=_F:raise ValueError(f"logit_softcap must be positive, got {h.logit_softcap}")
		self.tie_embeddings=h.tie_embeddings;self.tied_embed_init_std=h.tied_embed_init_std;self.logit_softcap=h.logit_softcap;self.tok_emb=nn.Embedding(h.vocab_size,h.embedding_dim)
		if h.embedding_dim!=h.model_dim:self.embed_proj=CastedLinear(h.embedding_dim,h.model_dim,bias=_C);self.head_proj=CastedLinear(h.model_dim,h.embedding_dim,bias=_C)
		else:self.embed_proj=_A;self.head_proj=_A
		self.num_encoder_layers=h.num_layers//2;self.num_decoder_layers=h.num_layers-self.num_encoder_layers;self.num_skip_weights=min(self.num_encoder_layers,self.num_decoder_layers);self.skip_weights=nn.Parameter(torch.ones(self.num_skip_weights,h.model_dim,dtype=torch.float32));self.skip_gates=nn.Parameter(torch.zeros(self.num_skip_weights,h.model_dim,dtype=torch.float32))if h.skip_gates_enabled else _A;self.blocks=nn.ModuleList([Block(h.model_dim,h.num_heads,h.num_kv_heads,h.mlp_mult,h.rope_base,h.qk_gain_init,h.train_seq_len,layer_idx=i,ln_scale=h.ln_scale)for i in range(h.num_layers)])
		if h.rope_dims>0:
			head_dim=h.model_dim//h.num_heads
			for block in self.blocks:block.attn.rope_dims=h.rope_dims;block.attn.rotary=Rotary(head_dim,base=h.rope_base,train_seq_len=h.train_seq_len,rope_dims=h.rope_dims)
		self.ve_layer_indices=[int(x)for x in h.ve_layers.split(',')if x.strip()]if h.ve_enabled else[];kv_dim=self._ve_target_dim
		if self.ve_layer_indices:self.ve_shared=ValueEmbedding(h.vocab_size,h.ve_dim,kv_dim);self.ve_layer_scales=nn.ParameterList([nn.Parameter(torch.ones(1,dtype=torch.float32))for _ in self.ve_layer_indices])
		else:self.ve_shared=_A;self.ve_layer_scales=nn.ParameterList()
		self.value_embeds=nn.ModuleList();self.final_norm=RMSNorm();self.lm_head=_A if h.tie_embeddings else CastedLinear(h.embedding_dim,h.vocab_size,bias=_C)
		if self.lm_head is not _A:self.lm_head._zero_init=_B
		if h.xsa_last_n>0:
			for i in range(max(0,h.num_layers-h.xsa_last_n),h.num_layers):self.blocks[i].attn.use_xsa=_B
		self.parallel_residual=bool(h.parallel_residual);self.parallel_start_layer=max(0,int(h.parallel_start_layer));self.parallel_start_layer_is_physical=bool(h.parallel_start_layer_is_physical);self.parallel_post_lambdas=nn.Parameter(torch.ones(h.num_layers,2,2,dtype=torch.float32))if self.parallel_residual else _A;self.parallel_resid_lambdas=nn.Parameter(torch.full((h.num_layers,2),1.1**.5,dtype=torch.float32))if self.parallel_residual else _A;self.recur_layers=_parse_layer_list(h.recur_layers_str)
		for rl in self.recur_layers:
			if not 0<=rl<h.num_layers:raise ValueError(f"recur layer {rl} out of range [0, {h.num_layers})")
		self.recur_start_step=int(h.recur_start_step);self._repeat_cutoff=max(self.recur_layers)+1 if self.recur_layers else h.num_layers;self._recurrence_active=_C;self.repeat_untie_mlp=h.repeat_untie_mlp
		if self.repeat_untie_mlp not in{_G,'down',_Y}:raise ValueError(f"repeat untie mlp mode must be one of none/down/full, got {self.repeat_untie_mlp}")
		requested_repeat_layers=_parse_layer_list(h.repeat_untie_mlp_layers)if h.repeat_untie_mlp_layers else[];invalid_repeat_layers=[rl for rl in requested_repeat_layers if rl not in self.recur_layers]
		if invalid_repeat_layers:raise ValueError(f"repeat untie mlp layers must be a subset of recur_layers, got {invalid_repeat_layers}")
		if self.repeat_untie_mlp==_G:self.repeat_untie_mlp_layers=[]
		elif requested_repeat_layers:self.repeat_untie_mlp_layers=requested_repeat_layers
		else:self.repeat_untie_mlp_layers=list(self.recur_layers)
		self.repeat_mlp=nn.ModuleList()
		if self.repeat_untie_mlp!=_G and self.recur_layers:
			for physical_idx in self.recur_layers:mode=self.repeat_untie_mlp if physical_idx in self.repeat_untie_mlp_layers else _G;self.repeat_mlp.append(RepeatMLPWeights(h.model_dim,h.mlp_mult,mode))
		if self.blocks:setattr(self.blocks[0],_Z,bool(h.disable_layer0_attn))
		self._init_weights()
		for repeat_mlp in self.repeat_mlp:
			if repeat_mlp.fc is not _A:nn.init.orthogonal_(repeat_mlp.fc.weight,gain=_D)
			if repeat_mlp.proj is not _A:nn.init.zeros_(repeat_mlp.proj.weight)
	def _init_weights(self):
		if self.tie_embeddings:nn.init.normal_(self.tok_emb.weight,mean=_F,std=self.tied_embed_init_std)
		for(name,module)in self.named_modules():
			if isinstance(module,nn.Linear):
				if getattr(module,'_zero_init',_C):nn.init.zeros_(module.weight)
				elif module.weight.ndim==2 and module.weight.shape[0]>=64 and module.weight.shape[1]>=64:nn.init.orthogonal_(module.weight,gain=_D)
	def _get_ve(self,layer_idx,input_ids,ve_cache=_A):
		A='ve'
		if self.ve_shared is _A or layer_idx not in self.ve_layer_indices:return
		if ve_cache is not _A and A not in ve_cache:ve_cache[A]=self.ve_shared(input_ids)
		ve_base=ve_cache[A]if ve_cache is not _A else self.ve_shared(input_ids);ve_idx=self.ve_layer_indices.index(layer_idx);return ve_base*self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)
	def set_recurrence_active(self,active):self._recurrence_active=bool(active)and bool(self.recur_layers)
	def prime_repeat_mlp(self):
		if not self.repeat_mlp:return
		with torch.no_grad():
			for(repeat_idx,physical_idx)in enumerate(self.recur_layers):
				repeat_mlp=self.repeat_mlp[repeat_idx];base_mlp=self.blocks[physical_idx].mlp
				if repeat_mlp.fc is not _A:repeat_mlp.fc.weight.copy_(base_mlp.fc.weight)
				if repeat_mlp.proj is not _A:repeat_mlp.proj.weight.copy_(base_mlp.proj.weight)
	def _get_virtual_layers(self):
		if self._recurrence_active and self.recur_layers:return list(range(self._repeat_cutoff))+self.recur_layers+list(range(self._repeat_cutoff,len(self.blocks)))
		return list(range(len(self.blocks)))
	def _get_repeat_mlp(self,virtual_idx,physical_idx):
		if not self._recurrence_active or not self.recur_layers or not self.repeat_mlp:return
		repeat_start=self._repeat_cutoff;repeat_end=repeat_start+len(self.recur_layers)
		if repeat_start<=virtual_idx<repeat_end and physical_idx in self.recur_layers:return self.repeat_mlp[self.recur_layers.index(physical_idx)]
	def _parallel_active_for_layer(self,virtual_idx,physical_idx):
		if self.parallel_post_lambdas is _A:return _C
		if self.parallel_start_layer_is_physical:return physical_idx>=self.parallel_start_layer
		return virtual_idx>=self.parallel_start_layer
	def _mix_with_x0(self,lane,x0,resid_mix):mix=resid_mix.to(dtype=lane.dtype);return mix[0][_A,_A,:]*lane+mix[1][_A,_A,:]*x0
	def _apply_skip_single(self,x,skip,i):
		if isinstance(skip,tuple):skip=skip[1]
		scaled_skip=self.skip_weights[i].to(dtype=x.dtype)[_A,_A,:]*skip
		if self.skip_gates is not _A:g=torch.sigmoid(self.skip_gates[i].to(dtype=x.dtype))[_A,_A,:];return torch.lerp(scaled_skip,x,g)
		return x+scaled_skip
	def _apply_skip_parallel(self,lane0,lane1,skip,i):
		if isinstance(skip,tuple):skip0,skip1=skip
		else:skip0=skip1=skip
		w=self.skip_weights[i].to(dtype=lane0.dtype)[_A,_A,:]
		if self.skip_gates is _A:return lane0+w*skip0,lane1+w*skip1
		g=torch.sigmoid(self.skip_gates[i].to(dtype=lane0.dtype))[_A,_A,:];return torch.lerp(w*skip0,lane0,g),torch.lerp(w*skip1,lane1,g)
	def _final_parallel_hidden(self,lane0,lane1):return(lane0+lane1)*.5
	def _block_forward(self,block,x,x0,v_embed=_A,repeat_mlp=_A):
		mix=block.resid_mix.to(dtype=x.dtype);x_in=mix[0][_A,_A,:]*x+mix[1][_A,_A,:]*x0;x_out=x_in
		if not getattr(block,_Z,_C):attn_in=block.attn_norm(x_in)*block.ln_scale_factor;attn_out=block.attn(attn_in,v_embed=v_embed);x_out=x_out+block.attn_scale.to(dtype=x_in.dtype)[_A,_A,:]*attn_out
		mlp_in=block.mlp_norm(x_out)*block.ln_scale_factor;mlp_out=repeat_mlp(mlp_in)if repeat_mlp is not _A else block.mlp(mlp_in);return x_out+block.mlp_scale.to(dtype=x_out.dtype)[_A,_A,:]*mlp_out
	def _parallel_block(self,block,lane0,lane1,x0,physical_idx,v_embed=_A,repeat_mlp=_A):
		if not getattr(block,_Z,_C):attn_read=self._mix_with_x0(lane0,x0,block.resid_mix);attn_in=block.attn_norm(attn_read)*block.ln_scale_factor;attn_out=block.attn(attn_in,v_embed=v_embed);attn_out=block.attn_scale.to(dtype=lane0.dtype)[_A,_A,:]*attn_out;resid=self.parallel_resid_lambdas[physical_idx,0].to(dtype=lane0.dtype);post=self.parallel_post_lambdas[physical_idx,0].to(dtype=lane0.dtype);lane0=resid*lane0+post[0]*attn_out;lane1=resid*lane1+post[1]*attn_out
		mlp_read=self._mix_with_x0(lane1,x0,block.resid_mix);mlp_in=block.mlp_norm(mlp_read)*block.ln_scale_factor;mlp_out=repeat_mlp(mlp_in)if repeat_mlp is not _A else block.mlp(mlp_in);mlp_out=block.mlp_scale.to(dtype=lane1.dtype)[_A,_A,:]*mlp_out;resid=self.parallel_resid_lambdas[physical_idx,1].to(dtype=lane0.dtype);post=self.parallel_post_lambdas[physical_idx,1].to(dtype=lane0.dtype);lane0=resid*lane0+post[0]*mlp_out;lane1=resid*lane1+post[1]*mlp_out;return lane0,lane1
	def _backbone(self,input_ids):
		x=self.tok_emb(input_ids);x=F.rms_norm(x,(x.size(-1),))
		if self.embed_proj is not _A:x=self.embed_proj(x)
		x0=x;skips=[];ve_cache={};v2p=self._get_virtual_layers();enc_layers=len(v2p)//2;dec_layers=len(v2p)-enc_layers;lane0=_A;lane1=_A
		for virtual_idx in range(enc_layers):
			physical_idx=v2p[virtual_idx];ve=self._get_ve(physical_idx,input_ids,ve_cache);repeat_mlp=self._get_repeat_mlp(virtual_idx,physical_idx)
			if self._parallel_active_for_layer(virtual_idx,physical_idx):
				if lane0 is _A:lane0=x;lane1=x
				lane0,lane1=self._parallel_block(self.blocks[physical_idx],lane0,lane1,x0,physical_idx,v_embed=ve,repeat_mlp=repeat_mlp);skips.append((lane0,lane1))
			else:x=self._block_forward(self.blocks[physical_idx],x,x0,v_embed=ve,repeat_mlp=repeat_mlp);skips.append(x)
		for i in range(dec_layers):
			virtual_idx=enc_layers+i;physical_idx=v2p[virtual_idx];ve=self._get_ve(physical_idx,input_ids,ve_cache);repeat_mlp=self._get_repeat_mlp(virtual_idx,physical_idx);skip_i=min(i,self.num_skip_weights-1)
			if self._parallel_active_for_layer(virtual_idx,physical_idx):
				if lane0 is _A:lane0=x;lane1=x
				if skips:lane0,lane1=self._apply_skip_parallel(lane0,lane1,skips.pop(),skip_i)
				lane0,lane1=self._parallel_block(self.blocks[physical_idx],lane0,lane1,x0,physical_idx,v_embed=ve,repeat_mlp=repeat_mlp)
			else:
				if skips:x=self._apply_skip_single(x,skips.pop(),skip_i)
				x=self._block_forward(self.blocks[physical_idx],x,x0,v_embed=ve,repeat_mlp=repeat_mlp)
		hidden=self._final_parallel_hidden(lane0,lane1)if lane1 is not _A else x;hidden=self.final_norm(hidden)
		if self.head_proj is not _A:hidden=self.head_proj(hidden)
		return hidden
	def compute_logits(self,hidden):
		if self.tie_embeddings:logits_proj=F.linear(hidden,self.tok_emb.weight)
		else:logits_proj=self.lm_head(hidden)
		return self.logit_softcap*torch.tanh(logits_proj/self.logit_softcap)
	def forward_hidden(self,input_ids):return self._backbone(input_ids)
	def forward_logits(self,input_ids):return self.compute_logits(self.forward_hidden(input_ids))
	def forward(self,input_ids,target_ids):logits=self.forward_logits(input_ids);return F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),target_ids.reshape(-1),reduction='mean')
def classify_param(name):
	A='.mlp.'
	if'tok_emb'in name or'lm_head'in name:return'embed'
	if _a in name:return _O
	if A in name:return _O
	if'.attn.'in name or'.proj.'in name and A not in name:return _T
	return'other'
@torch.compile
def zeropower_via_newtonschulz5(G,steps=10,eps=1e-07):
	a,b,c=3.4445,-4.775,2.0315;X=G.bfloat16();X/=X.norm()+eps;transposed=G.size(0)>G.size(1)
	if transposed:X=X.T
	for _ in range(steps):A=X@X.T;B=b*A+c*A@A;X=a*X+B@X
	return X.T if transposed else X
class Muon(torch.optim.Optimizer):
	def __init__(self,params,lr,momentum,backend_steps,nesterov=_B,weight_decay=_F):super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,nesterov=nesterov,weight_decay=weight_decay))
	@torch.no_grad()
	def step(self,closure=_A):
		A='momentum_buffer';loss=_A
		if closure is not _A:
			with torch.enable_grad():loss=closure()
		distributed=dist.is_available()and dist.is_initialized();world_size=dist.get_world_size()if distributed else 1;rank=dist.get_rank()if distributed else 0
		for group in self.param_groups:
			params=group[_P]
			if not params:continue
			lr=group[_H];momentum=group[_g];backend_steps=group['backend_steps'];nesterov=group['nesterov'];total_params=sum(int(p.numel())for p in params);updates_flat=torch.zeros(total_params,device=params[0].device,dtype=torch.bfloat16);curr=0
			for(i,p)in enumerate(params):
				if i%world_size==rank and p.grad is not _A:
					g=p.grad;state=self.state[p]
					if A not in state:state[A]=torch.zeros_like(g)
					buf=state[A];buf.mul_(momentum).add_(g)
					if nesterov:g=g.add(buf,alpha=momentum)
					row_norms=g.float().norm(dim=-1,keepdim=_B).clamp_min(1e-07);g=g/row_norms.to(g.dtype);g=zeropower_via_newtonschulz5(g,steps=backend_steps);g*=max(1,g.size(0)/g.size(1))**.5;updates_flat[curr:curr+p.numel()]=g.reshape(-1)
				curr+=p.numel()
			if distributed:dist.all_reduce(updates_flat,op=dist.ReduceOp.SUM)
			wd=group.get('weight_decay',_F);curr=0
			for p in params:
				if wd>_F:p.data.mul_(_D-lr*wd)
				g=updates_flat[curr:curr+p.numel()].view_as(p).to(dtype=p.dtype);p.add_(g,alpha=-lr);curr+=p.numel()
		return loss
class Optimizers:
	def __init__(self,h,base_model):
		named_params=list(base_model.blocks.named_parameters());named_params.extend(list(base_model.repeat_mlp.named_parameters()));matrix_params=[p for(name,p)in named_params if p.ndim==2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)];scalar_params=[p for(name,p)in named_params if p.ndim<2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)]
		if base_model.skip_weights.numel()>0:scalar_params.append(base_model.skip_weights)
		if base_model.skip_gates is not _A and base_model.skip_gates.numel()>0:scalar_params.append(base_model.skip_gates)
		if base_model.parallel_post_lambdas is not _A:scalar_params.append(base_model.parallel_post_lambdas)
		if base_model.parallel_resid_lambdas is not _A:scalar_params.append(base_model.parallel_resid_lambdas)
		token_lr=h.tied_embed_lr if h.tie_embeddings else h.embed_lr;tok_params=[{_P:[base_model.tok_emb.weight],_H:token_lr,_I:token_lr}]
		if base_model.ve_shared is not _A:
			tok_params.append({_P:[base_model.ve_shared.embed.weight],_H:token_lr,_I:token_lr})
			if base_model.ve_shared.proj is not _A:matrix_params.append(base_model.ve_shared.proj.weight)
			scalar_params.append(base_model.ve_shared.scale)
			for scale in base_model.ve_layer_scales:scalar_params.append(scale)
		self.optimizer_tok=torch.optim.AdamW(tok_params,betas=(h.beta1,h.beta2),eps=h.adam_eps,weight_decay=h.embed_wd,fused=_B);self.optimizer_muon=Muon(matrix_params,lr=h.matrix_lr,momentum=h.muon_momentum,backend_steps=h.muon_backend_steps,weight_decay=h.muon_wd)
		for group in self.optimizer_muon.param_groups:group[_I]=h.matrix_lr
		self.optimizer_scalar=torch.optim.AdamW([{_P:scalar_params,_H:h.scalar_lr,_I:h.scalar_lr}],betas=(h.beta1,h.beta2),eps=h.adam_eps,weight_decay=h.adam_wd,fused=_B);self.optimizers=[self.optimizer_tok,self.optimizer_muon,self.optimizer_scalar]
		if base_model.lm_head is not _A:self.optimizer_head=torch.optim.Adam([{_P:[base_model.lm_head.weight],_H:h.head_lr,_I:h.head_lr}],betas=(h.beta1,h.beta2),eps=h.adam_eps,fused=_B);self.optimizers.insert(1,self.optimizer_head)
		else:self.optimizer_head=_A
	def __iter__(self):return iter(self.optimizers)
	def zero_grad_all(self):
		for opt in self.optimizers:opt.zero_grad(set_to_none=_B)
	def step(self):
		for opt in self.optimizers:opt.step()
		self.zero_grad_all()
CONTROL_TENSOR_NAME_PATTERNS=tuple(pattern for pattern in os.environ.get('CONTROL_TENSOR_NAME_PATTERNS','attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates,ve_layer_scales,ve_shared.scale,parallel_post_lambdas,parallel_resid_lambdas').split(',')if pattern)
INT8_PER_ROW_SCALE_DTYPE=torch.float16
INT8_CLIP_PERCENTILE=99.99984
INT8_CLIP_Q=INT8_CLIP_PERCENTILE/1e2
def quantize_float_tensor(t):
	t32=t.float()
	if t32.ndim==2:clip_abs=torch.quantile(t32.abs(),INT8_CLIP_Q,dim=1)if t32.numel()else torch.empty((t32.shape[0],),dtype=torch.float32);clipped=torch.maximum(torch.minimum(t32,clip_abs[:,_A]),-clip_abs[:,_A]);scale=(clip_abs/127.).clamp_min(_D/127.);q=torch.clamp(torch.round(clipped/scale[:,_A]),-127,127).to(torch.int8).contiguous();return q,scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
	clip_abs=float(torch.quantile(t32.abs().flatten(),INT8_CLIP_Q).item())if t32.numel()else _F;scale=torch.tensor(clip_abs/127. if clip_abs>0 else _D,dtype=torch.float32);q=torch.clamp(torch.round(torch.clamp(t32,-clip_abs,clip_abs)/scale),-127,127).to(torch.int8).contiguous();return q,scale
def restore_fp32_params(model):
	for module in model.modules():
		if isinstance(module,CastedLinear):module.float()
	for(name,param)in model.named_parameters():
		if(param.ndim<2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))and param.dtype!=torch.float32:param.data=param.data.float()
def quantize_int6_per_row(t,clip_range=31):
	t32=t.float()
	if t32.ndim==2:
		best_q,best_s,best_err=_A,_A,float('inf')
		for pct in[.999,.9995,.9999,.99999,_D]:
			if pct<_D:row_clip=torch.quantile(t32.abs(),pct,dim=1)
			else:row_clip=t32.abs().amax(dim=1)
			s=(row_clip/clip_range).clamp_min(_D/clip_range).to(torch.float16);q=torch.clamp(torch.round(t32/s.float()[:,_A]),-clip_range,clip_range).to(torch.int8);recon=q.float()*s.float()[:,_A];err=(t32-recon).pow(2).mean().item()
			if err<best_err:best_q,best_s,best_err=q,s,err
		return best_q,best_s
	amax=t32.abs().max().item();scale=torch.tensor(amax/clip_range if amax>0 else _D,dtype=torch.float16);q=torch.clamp(torch.round(t32/scale.float()),-clip_range,clip_range).to(torch.int8);return q,scale
def collect_hessians(model,train_loader,h,device,n_calibration_batches=64):
	A='.weight';hessians={};hooks=[]
	def make_hook(name):
		def hook_fn(module,inp,out):
			x=inp[0].detach().float()
			if x.ndim==3:x=x.reshape(-1,x.shape[-1])
			if name not in hessians:hessians[name]=torch.zeros(x.shape[1],x.shape[1],dtype=torch.float32,device=device)
			hessians[name].addmm_(x.T,x)
		return hook_fn
	for(name,module)in model.named_modules():
		if isinstance(module,CastedLinear)and module.weight.numel()>65536:
			cat=classify_param(name+A)
			if cat in(_O,_T):hooks.append(module.register_forward_hook(make_hook(name+A)))
	model.eval()
	with torch.no_grad():
		for i in range(n_calibration_batches):x,y=train_loader.next_batch(h.train_batch_tokens,h.train_seq_len,h.grad_accum_steps);model.forward_logits(x)
	for h in hooks:h.remove()
	for name in hessians:hessians[name]=hessians[name].cpu()/n_calibration_batches
	return hessians
def gptq_quantize_weight(w,H,clip_range=31,block_size=128):
	W_orig=w.float().clone();rows,cols=W_orig.shape;H=H.float().clone();dead=torch.diag(H)==0;H[dead,dead]=1;damp=.01*H.diag().mean();H.diagonal().add_(damp);perm=torch.argsort(H.diag(),descending=_B);invperm=torch.argsort(perm);W_perm=W_orig[:,perm].clone();W_perm[:,dead[perm]]=0;H=H[perm][:,perm]
	try:Hinv=torch.cholesky_inverse(torch.linalg.cholesky(H));Hinv=torch.linalg.cholesky(Hinv,upper=_B)
	except torch.linalg.LinAlgError:return quantize_int6_per_row(W_orig,clip_range)
	best_q,best_scale,best_err=_A,_A,float('inf')
	for pct in[.999,.9995,.9999,.99999,_D]:
		if pct<_D:row_clip=torch.quantile(W_orig.abs(),pct,dim=1)
		else:row_clip=W_orig.abs().amax(dim=1)
		s=(row_clip/clip_range).clamp_min(_D/clip_range).to(torch.float16);sf=s.float();Q=torch.zeros(rows,cols,dtype=torch.int8);W_work=W_perm.clone()
		for i1 in range(0,cols,block_size):
			i2=min(i1+block_size,cols);W_block=W_work[:,i1:i2].clone();Hinv_block=Hinv[i1:i2,i1:i2];Err=torch.zeros(rows,i2-i1)
			for j in range(i2-i1):w_col=W_block[:,j];d=Hinv_block[j,j];q_col=torch.clamp(torch.round(w_col/sf),-clip_range,clip_range);Q[:,i1+j]=q_col.to(torch.int8);err=(w_col-q_col.float()*sf)/d;Err[:,j]=err;W_block[:,j:]-=err.unsqueeze(1)*Hinv_block[j,j:].unsqueeze(0)
			if i2<cols:W_work[:,i2:]-=Err@Hinv[i1:i2,i2:]
		recon=Q.float()*sf[:,_A];mse=(W_perm-recon).pow(2).mean().item()
		if mse<best_err:best_q,best_scale,best_err=Q,s,mse
	return best_q[:,invperm],best_scale
def gptq_mixed_quantize_int6(state_dict,int6_cats,hessians,clip_ranges=_A):
	C='method';B='int5';A='clip_range';result={};meta={};gptq_count=0;fallback_count=0;int5_count=0;int6_count=0
	for(name,tensor)in state_dict.items():
		t=tensor.detach().cpu().contiguous();cat=classify_param(name)
		if not t.is_floating_point()or t.numel()<=65536:result[name]=t.to(torch.float16)if t.is_floating_point()else t;meta[name]=_b;continue
		if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):result[name]=t.float();meta[name]=_c;continue
		layer_clip=31
		if clip_ranges is not _A and name in clip_ranges:layer_clip=clip_ranges[name]
		if cat in int6_cats and t.ndim==2:
			bit_label=B if layer_clip==15 else _Q
			if layer_clip==15:int5_count+=1
			else:int6_count+=1
			if name in hessians:q,s=gptq_quantize_weight(t,hessians[name],clip_range=layer_clip);gptq_count+=1;meta[name]={_J:bit_label,C:'gptq',A:layer_clip}
			else:q,s=quantize_int6_per_row(t,clip_range=layer_clip);fallback_count+=1;meta[name]={_J:bit_label,C:'clip_search',A:layer_clip}
			result[name+_K]=q;result[name+_L]=s
		elif cat in int6_cats and t.ndim>=1:q,s=quantize_int6_per_row(t,clip_range=layer_clip);bit_label=B if layer_clip==15 else _Q;result[name+_K]=q;result[name+_L]=s;meta[name]={_J:bit_label,A:layer_clip}
		else:q,s=quantize_float_tensor(t);result[name+_K]=q;result[name+_L]=s;meta[name]={_J:'int8'}
	log(f"GPTQ quantization: {gptq_count} layers with full GPTQ, {fallback_count} fallback to clip-search")
	if clip_ranges is not _A:log(f"mixed_quant: {int6_count} int6, {int5_count} int5")
	return result,meta
def mixed_quantize_int6(state_dict,int6_cats):
	result={};meta={}
	for(name,tensor)in state_dict.items():
		t=tensor.detach().cpu().contiguous();cat=classify_param(name)
		if not t.is_floating_point()or t.numel()<=65536:result[name]=t.to(torch.float16)if t.is_floating_point()else t;meta[name]=_b;continue
		if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):result[name]=t.float();meta[name]=_c;continue
		if cat in int6_cats and t.ndim>=1:q,s=quantize_int6_per_row(t);result[name+_K]=q;result[name+_L]=s;meta[name]={_J:_Q}
		else:q,s=quantize_float_tensor(t);result[name+_K]=q;result[name+_L]=s;meta[name]={_J:'int8'}
	return result,meta
def dequantize_mixed_int6(result,meta,template_sd):
	out={}
	for(name,orig)in template_sd.items():
		info=meta.get(name)
		if info is _A:continue
		orig_dtype=orig.dtype
		if info in(_b,_c,'passthrough_fp16'):
			t=result[name]
			if t.dtype==torch.float16 and orig_dtype in(torch.float32,torch.bfloat16):t=t.to(orig_dtype)
			out[name]=t;continue
		q,s=result[name+_K],result[name+_L]
		if s.ndim>0:out[name]=(q.float()*s.float().view(q.shape[0],*[1]*(q.ndim-1))).to(orig_dtype)
		else:out[name]=(q.float()*float(s.item())).to(orig_dtype)
	return out
_BSHF_MAGIC=b'BSHF'
def _byte_shuffle(data,stride=2):
	if stride<=1 or len(data)<stride:return data
	src=np.frombuffer(data,dtype=np.uint8);n=len(src);out=np.empty(n,dtype=np.uint8);dest_off=0
	for pos in range(stride):chunk=src[pos::stride];out[dest_off:dest_off+len(chunk)]=chunk;dest_off+=len(chunk)
	return _BSHF_MAGIC+bytes([stride])+out.tobytes()
def _byte_unshuffle(data):
	if len(data)<5 or data[:4]!=_BSHF_MAGIC:return data
	stride=data[4]
	if stride<2:return data[5:]
	payload=np.frombuffer(data,dtype=np.uint8,offset=5);n=len(payload);out=np.empty(n,dtype=np.uint8);src_off=0
	for pos in range(stride):chunk_len=n//stride+(1 if pos<n%stride else 0);out[pos::stride][:chunk_len]=payload[src_off:src_off+chunk_len];src_off+=chunk_len
	return out.tobytes()
def _compress(data,compressor,byte_shuffle=_B):
	if byte_shuffle:data=_byte_shuffle(data)
	if compressor=='lzma':return lzma.compress(data,preset=6)
	elif compressor==_V:import brotli;return brotli.compress(data,quality=11)
	raise ValueError(f"Unknown compressor: {compressor!r}")
def _decompress(data,compressor,byte_shuffle=_B):
	if compressor=='lzma':raw=lzma.decompress(data)
	elif compressor==_V:import brotli;raw=brotli.decompress(data)
	if byte_shuffle:raw=_byte_unshuffle(raw)
	return raw;raise ValueError(f"Unknown compressor: {compressor!r}")
def serialize(h,base_model,code):
	model_bytes=_A;code_bytes=len(code.encode(_N))
	if h.is_main_process:torch.save(base_model.state_dict(),h.model_path);model_bytes=os.path.getsize(h.model_path);log(f"Serialized model: {model_bytes} bytes");log(f"Code size: {code_bytes} bytes")
	sd_cpu={k:v.detach().cpu()for(k,v)in base_model.state_dict().items()}
	if h.gptq_enabled:
		log('GPTQ:collecting Hessians from calibration data...');t0=time.perf_counter();calib_loader=DistributedTokenLoader(h.train_files,h.rank,h.world_size,torch.device(_M,h.local_rank));hessians=collect_hessians(base_model,calib_loader,h,torch.device(_M,h.local_rank),n_calibration_batches=h.gptq_calibration_batches);log(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter()-t0:.1f}s");save_path=os.environ.get('SAVE_QUANT_STATE','').strip()
		if save_path and h.is_main_process:torch.save({'state_dict':{k:v.clone()for(k,v)in sd_cpu.items()},'hessians':{k:v.cpu().clone()for(k,v)in hessians.items()}},save_path);log(f"Saved quant state to {save_path} for offline mask search")
		clip_ranges=_A
		if h.mixed_quant:
			fixed_int5_str=os.environ.get('INT5_LAYERS','').strip()
			if fixed_int5_str:
				fixed_int5=set(fixed_int5_str.split(','));clip_ranges={};int6_count=int5_count=0
				for lname in hessians:
					if lname in fixed_int5:clip_ranges[lname]=15;int5_count+=1
					else:clip_ranges[lname]=31;int6_count+=1
				log(f"mixed_quant: FIXED mask -- {int6_count} int6, {int5_count} int5");log(f"mixed_quant: int5 layers: {sorted(fixed_int5)}")
			else:
				sensitivity={}
				for(lname,H)in hessians.items():sensitivity[lname]=H.diag().sum().item()
				ranked=sorted(sensitivity.items(),key=lambda x:x[1],reverse=_B);n_int6=min(h.n_int6_layers,len(ranked));clip_ranges={}
				for(i,(lname,sens))in enumerate(ranked):clip_ranges[lname]=31 if i<n_int6 else 15
				n_int5=len(ranked)-n_int6;log(f"mixed_quant: sensitivity ranking -- {n_int6} int6 (top), {n_int5} int5 (bottom)")
				for(i,(lname,sens))in enumerate(ranked):tag=_Q if i<n_int6 else'INT5';numel=sd_cpu[lname].numel();log(f"  rank {i:2d}: {tag} {lname}{ sens=:.1f} numel={numel}")
				if ranked:log(f"mixed_quant: most sensitive={ranked[0][0]} ({ranked[0][1]:.1f}), least sensitive={ranked[-1][0]} ({ranked[-1][1]:.1f})")
		quant_result,quant_meta=gptq_mixed_quantize_int6(sd_cpu,{_O,_T},hessians,clip_ranges=clip_ranges)
	else:quant_result,quant_meta=mixed_quantize_int6(sd_cpu,{_O,_T})
	quant_buf=io.BytesIO();torch.save({'w':quant_result,'m':quant_meta},quant_buf);quant_raw=quant_buf.getvalue();quant_blob=_compress(quant_raw,h.compressor);quant_file_bytes=len(quant_blob);bytes_total=quant_file_bytes+code_bytes
	if h.is_main_process:
		with open(h.quantized_model_path,'wb')as f:f.write(quant_blob)
		quant_label='mixed_int5_int6'if h.gptq_enabled and h.mixed_quant else _Q;log(f"Serialized model {quant_label}+{h.compressor}: {quant_file_bytes} bytes");log(f"Total submission size {quant_label}+{h.compressor}: {bytes_total} bytes")
def deserialize(h,device):
	eval_model=GPT(h).to(device).bfloat16();restore_fp32_params(eval_model)
	if getattr(eval_model,_U,_A):eval_model.set_recurrence_active(_B)
	sd_cpu={k:v.detach().cpu()for(k,v)in eval_model.state_dict().items()}
	with open(h.quantized_model_path,'rb')as f:quant_blob_disk=f.read()
	quant_state=torch.load(io.BytesIO(_decompress(quant_blob_disk,h.compressor)),map_location='cpu');deq_state=dequantize_mixed_int6(quant_state['w'],quant_state['m'],sd_cpu);eval_model.load_state_dict(deq_state,strict=_B);return eval_model
def _loss_bpb(loss_sum,token_count,byte_count):val_loss=(loss_sum/token_count).item();val_bpb=val_loss/math.log(2.)*(token_count.item()/byte_count.item());return val_loss,val_bpb
def eval_val(h,device,val_data,model):
	seq_len=h.eval_seq_len;local_batch_tokens=h.val_batch_tokens//(h.world_size*h.grad_accum_steps)
	if local_batch_tokens<seq_len:raise ValueError(f"VAL_BATCH_SIZE must provide at least one sequence per rank; got VAL_BATCH_SIZE={h.val_batch_tokens}, WORLD_SIZE={h.world_size}, GRAD_ACCUM_STEPS={h.grad_accum_steps}, seq_len={seq_len}")
	local_batch_seqs=local_batch_tokens//seq_len;total_seqs=(val_data.val_tokens.numel()-1)//seq_len;seq_start=total_seqs*h.rank//h.world_size;seq_end=total_seqs*(h.rank+1)//h.world_size;val_loss_sum=torch.zeros((),device=device,dtype=torch.float64);val_token_count=torch.zeros((),device=device,dtype=torch.float64);val_byte_count=torch.zeros((),device=device,dtype=torch.float64);model.eval()
	with torch.inference_mode():
		for batch_seq_start in range(seq_start,seq_end,local_batch_seqs):
			batch_seq_end=min(batch_seq_start+local_batch_seqs,seq_end);raw_start=batch_seq_start*seq_len;raw_end=batch_seq_end*seq_len+1;local=val_data.val_tokens[raw_start:raw_end].to(device=device,dtype=torch.int64,non_blocking=_B);x=local[:-1].reshape(-1,seq_len);y=local[1:].reshape(-1,seq_len)
			with torch.autocast(device_type=_M,dtype=torch.bfloat16,enabled=_B):batch_loss=model(x,y).detach()
			batch_token_count=float(y.numel());val_loss_sum+=batch_loss.to(torch.float64)*batch_token_count;val_token_count+=batch_token_count;prev_ids=x.reshape(-1);tgt_ids=y.reshape(-1);token_bytes=val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16);token_bytes+=(val_data.has_leading_space_lut[tgt_ids]&~val_data.is_boundary_token_lut[prev_ids]).to(dtype=torch.int16);val_byte_count+=token_bytes.to(torch.float64).sum()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(val_loss_sum,op=dist.ReduceOp.SUM);dist.all_reduce(val_token_count,op=dist.ReduceOp.SUM);dist.all_reduce(val_byte_count,op=dist.ReduceOp.SUM)
	model.train();return _loss_bpb(val_loss_sum,val_token_count,val_byte_count)
def eval_val_sliding(h,device,val_data,base_model,batch_seqs=32):
	base_model.eval();logits_fn=torch.compile(base_model.forward_logits,dynamic=_C,fullgraph=_B);seq_len=h.eval_seq_len;context_size=seq_len-h.eval_stride;total_tokens=val_data.val_tokens.numel()-1;window_starts=[ws for ws in range(0,total_tokens,h.eval_stride)if ws+context_size<total_tokens];total_windows=len(window_starts);my_s=total_windows*h.rank//h.world_size;my_e=total_windows*(h.rank+1)//h.world_size;my_windows=window_starts[my_s:my_e];loss_sum=torch.zeros((),device=device,dtype=torch.float64);token_count=torch.zeros((),device=device,dtype=torch.float64);byte_count=torch.zeros((),device=device,dtype=torch.float64)
	with torch.inference_mode():
		for bi in range(0,len(my_windows),batch_seqs):
			batch_ws=my_windows[bi:bi+batch_seqs];bsz=len(batch_ws);x_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);y_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);wlens=[]
			for(i,ws)in enumerate(batch_ws):we=min(ws+seq_len,total_tokens);wlen=we-ws;wlens.append(wlen);chunk=val_data.val_tokens[ws:we+1].to(dtype=torch.int64,device=device);x_batch[i,:wlen]=chunk[:-1];y_batch[i,:wlen]=chunk[1:]
			with torch.autocast(device_type=_M,dtype=torch.bfloat16):logits=logits_fn(x_batch)
			nll=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),y_batch.reshape(-1),reduction=_G).reshape(bsz,seq_len)
			for(i,ws)in enumerate(batch_ws):wlen=wlens[i];s=0 if ws==0 else context_size;scored_nll=nll[i,s:wlen].to(torch.float64);loss_sum+=scored_nll.sum();token_count+=float(wlen-s);tgt=y_batch[i,s:wlen];prev=x_batch[i,s:wlen];tb=val_data.base_bytes_lut[tgt].to(torch.float64);tb+=(val_data.has_leading_space_lut[tgt]&~val_data.is_boundary_token_lut[prev]).to(torch.float64);byte_count+=tb.sum()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(loss_sum,op=dist.ReduceOp.SUM);dist.all_reduce(token_count,op=dist.ReduceOp.SUM);dist.all_reduce(byte_count,op=dist.ReduceOp.SUM)
	base_model.train();return _loss_bpb(loss_sum,token_count,byte_count)
def eval_val_sliding_etlb(h,device,val_data,base_model):
	"""Sliding window eval with Eval-Time Logit Bias (ETLB).
	Optimizes a bias vector b ∈ R^vocab on context tokens (already scored),
	then applies b when scoring new stride tokens. Strictly causal.
	Novel technique by @AnubhavBharadwaaj."""
	base_model.eval();seq_len=h.eval_seq_len;stride=h.eval_stride;context_size=seq_len-stride;total_tokens=val_data.val_tokens.numel()-1;vocab=h.vocab_size
	window_starts=[ws for ws in range(0,total_tokens,stride)if ws+context_size<total_tokens];total_windows=len(window_starts);my_s=total_windows*h.rank//h.world_size;my_e=total_windows*(h.rank+1)//h.world_size;my_windows=window_starts[my_s:my_e]
	loss_sum=torch.zeros((),device=device,dtype=torch.float64);token_count=torch.zeros((),device=device,dtype=torch.float64);byte_count=torch.zeros((),device=device,dtype=torch.float64)
	bias=torch.zeros(vocab,device=device,dtype=torch.float32)
	log(f"etlb:start windows={len(my_windows)} lr={h.etlb_lr} steps={h.etlb_steps} clip={h.etlb_clip}")
	for wi,ws in enumerate(my_windows):
		we=min(ws+seq_len,total_tokens);wlen=we-ws;chunk=val_data.val_tokens[ws:we+1].to(dtype=torch.int64,device=device)
		x=chunk[:-1].unsqueeze(0);y=chunk[1:].unsqueeze(0)
		with torch.inference_mode():
			with torch.autocast(device_type=_M,dtype=torch.bfloat16):logits=base_model.forward_logits(x)
		logits=logits.float().squeeze(0)
		s=0 if ws==0 else context_size
		if s>0 and h.etlb_steps>0:
			b=bias.clone().detach().requires_grad_(_B)
			opt=torch.optim.SGD([b],lr=h.etlb_lr)
			ctx_logits=logits[:s].detach();ctx_tgt=y.squeeze(0)[:s]
			for _ in range(h.etlb_steps):
				opt.zero_grad();loss=F.cross_entropy(ctx_logits+b,ctx_tgt);loss.backward();opt.step()
			bias=b.detach().clamp_(-h.etlb_clip,h.etlb_clip)
		scored_logits=logits[s:wlen]+bias
		scored_nll=F.cross_entropy(scored_logits,y.squeeze(0)[s:wlen],reduction=_G).to(torch.float64)
		loss_sum+=scored_nll.sum();token_count+=float(wlen-s)
		tgt=y.squeeze(0)[s:wlen];prev=x.squeeze(0)[s:wlen];tb=val_data.base_bytes_lut[tgt].to(torch.float64)
		tb+=(val_data.has_leading_space_lut[tgt]&~val_data.is_boundary_token_lut[prev]).to(torch.float64);byte_count+=tb.sum()
		if(wi+1)%5000==0:log(f"  etlb:window {wi+1}/{len(my_windows)} bias_norm={bias.norm().item():.4f}")
	if dist.is_available()and dist.is_initialized():dist.all_reduce(loss_sum,op=dist.ReduceOp.SUM);dist.all_reduce(token_count,op=dist.ReduceOp.SUM);dist.all_reduce(byte_count,op=dist.ReduceOp.SUM)
	base_model.train();return _loss_bpb(loss_sum,token_count,byte_count)
def timed_eval(label,fn,*args,**kwargs):torch.cuda.synchronize();t0=time.perf_counter();val_loss,val_bpb=fn(*args,**kwargs);torch.cuda.synchronize();elapsed_ms=1e3*(time.perf_counter()-t0);log(f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms");return val_loss,val_bpb
def run_evals(h,device,val_data,eval_model):
	compiled_model=torch.compile(eval_model,dynamic=_C,fullgraph=_B);timed_eval('final_int6_roundtrip',eval_val,h,device,val_data,compiled_model)
	if h.sliding_window_enabled:timed_eval('final_int6_sliding_window',eval_val_sliding,h,device,val_data,eval_model)
	if h.etlb_enabled:timed_eval('final_int6_sliding_etlb',eval_val_sliding_etlb,h,device,val_data,eval_model)
def train_model(h,device,val_data):
	B='_recurrence_active';A='parallel_post_lambdas';base_model=GPT(h).to(device).bfloat16();restore_fp32_params(base_model);import torch._dynamo;torch._dynamo.config.cache_size_limit=32;compiled_model=torch.compile(base_model,dynamic=_C,fullgraph=_B)
	if h.distributed:model=DDP(compiled_model,device_ids=[h.local_rank],broadcast_buffers=_C,find_unused_parameters=_B)
	else:model=compiled_model
	log(f"model_params:{sum(p.numel()for p in base_model.parameters())}")
	if getattr(base_model,A,_A)is not _A:params=base_model.parallel_post_lambdas.numel()+base_model.parallel_resid_lambdas.numel()
	else:params=0
	log(f"parallel_residual: active={int(getattr(base_model,A,_A)is not _A)} start_layer={h.parallel_start_layer} start_mode={"physical"if h.parallel_start_layer_is_physical else"virtual"} params={params}");log(f"recurrence: layers={getattr(base_model,_U,[])} start_step={h.recur_start_step} active={int(getattr(base_model,B,_C))}");log(f"repeat_untie_mlp: mode={h.repeat_untie_mlp} layers={getattr(base_model,"repeat_untie_mlp_layers",[])} params={sum(p.numel()for p in getattr(base_model,_a,[]).parameters())if getattr(base_model,_a,_A)else 0}");optimizers=Optimizers(h,base_model);train_loader=DistributedTokenLoader(h.train_files,h.rank,h.world_size,device);max_wallclock_ms=1e3*h.max_wallclock_seconds if h.max_wallclock_seconds>0 else _A
	if h.gptq_enabled and max_wallclock_ms is not _A:max_wallclock_ms-=h.gptq_reserve_seconds*1e3;log(f"gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms")
	def training_frac(step,elapsed_ms):
		if max_wallclock_ms is _A:return step/max(h.iterations,1)
		return elapsed_ms/max(max_wallclock_ms,1e-09)
	def lr_mul(frac):
		if h.warmdown_frac<=0:return _D
		if frac>=_D-h.warmdown_frac:return max((_D-frac)/h.warmdown_frac,h.min_lr)
		return _D
	def step_fn(step,lr_scale):
		optimizers.zero_grad_all();train_loss=torch.zeros((),device=device)
		for micro_step in range(h.grad_accum_steps):
			if h.distributed:model.require_backward_grad_sync=micro_step==h.grad_accum_steps-1
			x,y=train_loader.next_batch(h.train_batch_tokens,h.train_seq_len,h.grad_accum_steps)
			with torch.autocast(device_type=_M,dtype=torch.bfloat16,enabled=_B):loss=model(x,y)
			train_loss+=loss.detach();(loss/h.grad_accum_steps).backward()
		train_loss/=h.grad_accum_steps;frac=min(step/h.muon_momentum_warmup_steps,_D)if h.muon_momentum_warmup_steps>0 else _D;muon_momentum=(1-frac)*h.muon_momentum_warmup_start+frac*h.muon_momentum
		for group in optimizers.optimizer_muon.param_groups:group[_g]=muon_momentum
		for opt in optimizers:
			for group in opt.param_groups:group[_H]=group[_I]*lr_scale
		if h.grad_clip_norm>0:torch.nn.utils.clip_grad_norm_(base_model.parameters(),h.grad_clip_norm)
		optimizers.step();return train_loss
	if h.warmup_steps>0:
		initial_model_state={name:tensor.detach().cpu().clone()for(name,tensor)in base_model.state_dict().items()};initial_optimizer_states=[copy.deepcopy(opt.state_dict())for opt in optimizers];model.train()
		for warmup_step in range(h.warmup_steps):
			step_fn(warmup_step,_D)
			if warmup_step<=5 or(warmup_step+1)%10==0 or warmup_step+1==h.warmup_steps:log(f"warmup_step: {warmup_step+1}/{h.warmup_steps}")
		if getattr(base_model,_U,_A)and h.recur_warmup_steps>0:
			base_model.prime_repeat_mlp();base_model.set_recurrence_active(_B);log(f"recurrence:prewarm active=1 virtual_layers:{len(base_model._get_virtual_layers())}")
			for recur_warmup_step in range(h.recur_warmup_steps):
				step_fn(recur_warmup_step,_D)
				if recur_warmup_step<=5 or(recur_warmup_step+1)%10==0 or recur_warmup_step+1==h.recur_warmup_steps:log(f"recur_warmup_step: {recur_warmup_step+1}/{h.recur_warmup_steps}")
			base_model.set_recurrence_active(_C)
		base_model.load_state_dict(initial_model_state,strict=_B)
		for(opt,state)in zip(optimizers,initial_optimizer_states,strict=_B):opt.load_state_dict(state)
		optimizers.zero_grad_all()
		if h.distributed:model.require_backward_grad_sync=_B
		train_loader=DistributedTokenLoader(h.train_files,h.rank,h.world_size,device)
	ema_state={name:t.detach().float().clone()for(name,t)in base_model.state_dict().items()};ema_decay=h.ema_decay;training_time_ms=_F;stop_after_step=_A;torch.cuda.synchronize();t0=time.perf_counter();step=0
	while _B:
		if getattr(base_model,_U,_A)and not getattr(base_model,B,_C)and step>=h.recur_start_step:base_model.prime_repeat_mlp();base_model.set_recurrence_active(_B);log(f"recurrence:activated step:{step} layers:{base_model.recur_layers} virtual_layers:{len(base_model._get_virtual_layers())}")
		last_step=step==h.iterations or stop_after_step is not _A and step>=stop_after_step;should_validate=last_step or h.val_loss_every>0 and step%h.val_loss_every==0
		if should_validate:torch.cuda.synchronize();training_time_ms+=1e3*(time.perf_counter()-t0);val_loss,val_bpb=eval_val(h,device,val_data,model);log(f"{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}");torch.cuda.synchronize();t0=time.perf_counter()
		if last_step:
			if stop_after_step is not _A and step<h.iterations:log(f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms step: {step}/{h.iterations}")
			break
		elapsed_ms=training_time_ms+1e3*(time.perf_counter()-t0);frac=training_frac(step,elapsed_ms);scale=lr_mul(frac);train_loss=step_fn(step,scale)
		with torch.no_grad():
			for(name,t)in base_model.state_dict().items():ema_state[name].mul_(ema_decay).add_(t.detach().float(),alpha=_D-ema_decay)
		step+=1;approx_training_time_ms=training_time_ms+1e3*(time.perf_counter()-t0);should_log_train=h.train_log_every>0 and(step<=5 or step%h.train_log_every==0 or stop_after_step is not _A)
		if should_log_train:tok_per_sec=step*h.train_batch_tokens/(approx_training_time_ms/1e3);log(f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} train_time: {approx_training_time_ms/60000:.1f}m tok/s: {tok_per_sec:.0f}")
		reached_cap=max_wallclock_ms is not _A and approx_training_time_ms>=max_wallclock_ms
		if h.distributed and max_wallclock_ms is not _A:reached_cap_tensor=torch.tensor(int(reached_cap),device=device);dist.all_reduce(reached_cap_tensor,op=dist.ReduceOp.MAX);reached_cap=bool(reached_cap_tensor.item())
		if stop_after_step is _A and reached_cap:stop_after_step=step
	log(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB");log('ema:applying EMA weights');current_state=base_model.state_dict();avg_state={name:t.to(dtype=current_state[name].dtype)for(name,t)in ema_state.items()};base_model.load_state_dict(avg_state,strict=_B);return base_model,compiled_model
def pre_quant_ttt(h,base_model,device,val_data):
	"""Pre-quantization TTT: adapt EMA weights on val data using score-first protocol.
	Adapted weights quantize better under GPTQ. Legal: follows score-first constraint
	(score chunk, then train on already-scored tokens). Runs after training, before GPTQ."""
	if not h.pre_quant_ttt_enabled:return
	num_blocks=len(base_model.blocks);freeze_n=min(h.pre_quant_ttt_freeze_blocks,num_blocks)
	log(f"pre_quant_ttt:start lr={h.pre_quant_ttt_lr} epochs={h.pre_quant_ttt_epochs} freeze={freeze_n}/{num_blocks}")
	for i in range(freeze_n):
		for p in base_model.blocks[i].parameters():p.requires_grad_(_C)
	for i in range(freeze_n,num_blocks):
		for p in base_model.blocks[i].parameters():p.requires_grad_(_B)
	base_model.tok_emb.weight.requires_grad_(_C)
	if hasattr(base_model,'embed_proj')and base_model.embed_proj is not _A:
		for p in base_model.embed_proj.parameters():p.requires_grad_(_C)
	if hasattr(base_model,'head_proj')and base_model.head_proj is not _A:
		for p in base_model.head_proj.parameters():p.requires_grad_(_C)
	if base_model.skip_weights is not _A:base_model.skip_weights.requires_grad_(_C)
	if base_model.skip_gates is not _A:base_model.skip_gates.requires_grad_(_C)
	if base_model.ve_shared is not _A:
		for p in base_model.ve_shared.parameters():p.requires_grad_(_C)
	for s in base_model.ve_layer_scales:s.requires_grad_(_C)
	if hasattr(base_model,'parallel_post_lambdas')and base_model.parallel_post_lambdas is not _A:base_model.parallel_post_lambdas.requires_grad_(_C)
	if hasattr(base_model,'parallel_resid_lambdas')and base_model.parallel_resid_lambdas is not _A:base_model.parallel_resid_lambdas.requires_grad_(_C)
	trainable=sum(p.numel()for p in base_model.parameters()if p.requires_grad);frozen=sum(p.numel()for p in base_model.parameters()if not p.requires_grad)
	log(f"pre_quant_ttt:params trainable={trainable} frozen={frozen}")
	ttt_params=[p for p in base_model.parameters()if p.requires_grad]
	optimizer=torch.optim.AdamW(ttt_params,lr=h.pre_quant_ttt_lr,weight_decay=0.0,betas=(0.9,0.999))
	val_tokens=val_data.val_tokens;seq_len=h.eval_seq_len;chunk_tokens=h.pre_quant_ttt_chunk_tokens;total_tokens=val_tokens.numel()-1;t0=time.perf_counter()
	for epoch in range(h.pre_quant_ttt_epochs):
		pos=0;chunk_idx=0;epoch_loss_sum=0.0;epoch_chunks=0
		while pos<total_tokens:
			end=min(pos+chunk_tokens,total_tokens);n_seqs=(end-pos)//seq_len
			if n_seqs==0:pos=end;continue
			usable=n_seqs*seq_len;chunk=val_tokens[pos:pos+usable+1].to(device=device,dtype=torch.int64)
			x=chunk[:usable].reshape(n_seqs,seq_len);y=chunk[1:usable+1].reshape(n_seqs,seq_len)
			with torch.inference_mode():
				with torch.autocast(device_type=_M,dtype=torch.bfloat16):score_loss=base_model(x,y)
			optimizer.zero_grad(set_to_none=_B)
			with torch.autocast(device_type=_M,dtype=torch.bfloat16):loss=base_model(x,y)
			loss.backward();optimizer.step()
			epoch_loss_sum+=loss.item();epoch_chunks+=1;chunk_idx+=1;pos+=usable
			if chunk_idx%50==0:
				elapsed=time.perf_counter()-t0;avg_loss=epoch_loss_sum/max(epoch_chunks,1)
				log(f"  ttt_epoch:{epoch+1}/{h.pre_quant_ttt_epochs} chunk:{chunk_idx} loss:{loss.item():.4f} avg:{avg_loss:.4f} time:{elapsed:.1f}s")
		elapsed=time.perf_counter()-t0;avg_loss=epoch_loss_sum/max(epoch_chunks,1)
		log(f"  ttt_epoch:{epoch+1}/{h.pre_quant_ttt_epochs} done chunks:{epoch_chunks} avg_loss:{avg_loss:.4f} time:{elapsed:.1f}s")
	elapsed=time.perf_counter()-t0;log(f"pre_quant_ttt:done epochs={h.pre_quant_ttt_epochs} total_time={elapsed:.1f}s")
	for p in base_model.parameters():p.requires_grad_(_B)
def train_and_eval(h,device):
	random.seed(h.seed);np.random.seed(h.seed);torch.manual_seed(h.seed);torch.cuda.manual_seed_all(h.seed);val_data=ValidationData(h,device);log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob(_f)))}");log(f"val_tokens: {val_data.val_tokens.numel()-1}");base_model,compiled_model=train_model(h,device,val_data);timed_eval('pre-quantization post-ema',eval_val,h,device,val_data,compiled_model);pre_quant_ttt(h,base_model,device,val_data);serialize(h,base_model,Path(__file__).read_text(encoding=_N))
	if h.distributed:dist.barrier()
	eval_model=deserialize(h,device);run_evals(h,device,val_data,eval_model)
def main():
	A='=';world_size=int(os.environ.get(_R,_E));local_rank=int(os.environ.get(_e,'0'));distributed=_W in os.environ and _R in os.environ
	if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
	if world_size<=0:raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
	if 8%world_size!=0:raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
	device=torch.device(_M,local_rank);torch.cuda.set_device(device)
	if distributed:dist.init_process_group(backend='nccl',device_id=device);dist.barrier()
	torch.backends.cuda.matmul.allow_tf32=_B;torch.backends.cudnn.allow_tf32=_B;torch.set_float32_matmul_precision('high');from torch.backends.cuda import enable_cudnn_sdp,enable_flash_sdp,enable_math_sdp,enable_mem_efficient_sdp;enable_cudnn_sdp(_C);enable_flash_sdp(_B);enable_mem_efficient_sdp(_C);enable_math_sdp(_C);torch._dynamo.config.optimize_ddp=_C;h=Hyperparameters();set_logging_hparams(h)
	if h.is_main_process:
		os.makedirs('logs',exist_ok=_B);log(100*A,console=_C);log('Hyperparameters:',console=_B)
		for(k,v)in sorted(vars(type(h)).items()):
			if not k.startswith('_'):log(f"  {k}: {v}",console=_B)
		log(Path(__file__).read_text(encoding=_N),console=_C);log(A*100,console=_C);log(f"Running Python {sys.version}",console=_C);log(f"Running PyTorch {torch.__version__}",console=_C);log(subprocess.run(['nvidia-smi'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=_B,check=_C).stdout,console=_C);log(A*100,console=_C)
	train_and_eval(h,device)
	if distributed:dist.destroy_process_group()
if __name__=='__main__':main()