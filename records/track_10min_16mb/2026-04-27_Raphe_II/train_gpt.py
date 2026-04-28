_T='.scale'
_S='momentum'
_R='fineweb_train_*.bin'
_Q='LOCAL_RANK'
_P='tok_emb'
_O='<u2'
_N='RANK'
_M='brotli'
_L='params'
_K='<i4'
_J='utf-8'
_I='WORLD_SIZE'
_H='base_lr'
_G='lr'
_F='cuda'
_E=.0
_D=1.
_C=False
_B=True
_A=None
import collections,copy,glob,io,math,os
from pathlib import Path
import random,re,subprocess,sys,time,uuid,numpy as np,sentencepiece as spm,torch,torch.distributed as dist,torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor,nn
from flash_attn_interface import flash_attn_func as flash_attn_3_func
class Hyperparameters:data_dir=os.environ.get('DATA_DIR','./data/');seed=int(os.environ.get('SEED',1337));run_id=os.environ.get('RUN_ID',str(uuid.uuid4()));iterations=20000;warmdown_frac=.72;warmup_steps=20;train_batch_tokens=786432;train_seq_len=2048;train_log_every=500;max_wallclock_seconds=6e2;val_batch_tokens=524288;eval_seq_len=2048;val_loss_every=4000;sliding_window_enabled=_B;vocab_size=4096;num_layers=11;xsa_last_n=11;model_dim=512;embedding_dim=512;num_kv_heads=4;num_heads=8;mlp_mult=4.5;skip_gates_enabled=_B;tie_embeddings=_B;logit_softcap=3e1;rope_base=1e4;rope_dims=16;rope_train_seq_len=2048;ln_scale=_B;qk_gain_init=5.25;num_loops=2;loop_start=3;loop_end=5;enable_looping_at=.35;parallel_residual_start=7;min_lr=_E;embed_lr=.6;head_lr=.008;tied_embed_lr=.03;tied_embed_init_std=.005;matrix_lr=.022;scalar_lr=.02;muon_momentum=.99;muon_backend_steps=5;muon_momentum_warmup_start=.92;muon_momentum_warmup_steps=1500;muon_row_normalize=_B;beta1=.9;beta2=.95;adam_eps=1e-08;grad_clip_norm=.3;eval_stride=64;muon_beta2=.95;adam_wd=.02;muon_wd=.095;embed_wd=.085;ema_decay=.9965;ttt_enabled=_B;ttt_lr=.005;ttt_epochs=3;ttt_momentum=.9;ttt_chunk_tokens=32768;etlb_enabled=_C;etlb_lr=.05;etlb_steps=5;etlb_clip=3.;compressor=_M;gptq_calibration_batches=64;gptq_reserve_seconds=12.;matrix_bits=6;embed_bits=8;matrix_clip_sigmas=12.85;embed_clip_sigmas=2e1;distributed=_N in os.environ and _I in os.environ;rank=int(os.environ.get(_N,'0'));world_size=int(os.environ.get(_I,'1'));local_rank=int(os.environ.get(_Q,'0'));is_main_process=rank==0;grad_accum_steps=8//world_size;datasets_dir=os.path.join(data_dir,'datasets',f"fineweb10B_sp{vocab_size}");train_files=os.path.join(datasets_dir,_R);val_files=os.path.join(datasets_dir,'fineweb_val_*.bin');tokenizer_path=os.path.join(data_dir,'tokenizers',f"fineweb_{vocab_size}_bpe.model");logfile=f"logs/{run_id}.txt";model_path='final_model.pt';quantized_model_path='final_model.int6.ptz'
_logger_hparams=_A
def set_logging_hparams(h):global _logger_hparams;_logger_hparams=h
def log(msg,console=_B):
	if _logger_hparams is _A:print(msg);return
	if _logger_hparams.is_main_process:
		if console:print(msg)
		if _logger_hparams.logfile is not _A:
			with open(_logger_hparams.logfile,'a',encoding=_J)as f:print(msg,file=f)
class ValidationData:
	def __init__(self,h,device):
		self.sp=spm.SentencePieceProcessor(model_file=h.tokenizer_path)
		if int(self.sp.vocab_size())!=h.vocab_size:raise ValueError("vocab")
		self.val_tokens=load_validation_tokens(h.val_files,h.eval_seq_len);self.base_bytes_lut,self.has_leading_space_lut,self.is_boundary_token_lut=build_sentencepiece_luts(self.sp,h.vocab_size,device)
def build_sentencepiece_luts(sp,vocab_size,device):
	sp_vocab_size=int(sp.vocab_size());table_size=max(sp_vocab_size,vocab_size);base_bytes_np=np.zeros((table_size,),dtype=np.int16);has_leading_space_np=np.zeros((table_size,),dtype=np.bool_);is_boundary_token_np=np.ones((table_size,),dtype=np.bool_)
	for token_id in range(sp_vocab_size):
		if sp.is_control(token_id)or sp.is_unknown(token_id)or sp.is_unused(token_id):continue
		is_boundary_token_np[token_id]=_C
		if sp.is_byte(token_id):base_bytes_np[token_id]=1;continue
		piece=sp.id_to_piece(token_id)
		if piece.startswith('▁'):has_leading_space_np[token_id]=_B;piece=piece[1:]
		base_bytes_np[token_id]=len(piece.encode(_J))
	return torch.tensor(base_bytes_np,dtype=torch.int16,device=device),torch.tensor(has_leading_space_np,dtype=torch.bool,device=device),torch.tensor(is_boundary_token_np,dtype=torch.bool,device=device)
def load_validation_tokens(pattern,seq_len):
	files=[Path(p)for p in sorted(glob.glob(pattern))]
	if not files:raise FileNotFoundError(f"No files found for pattern: {pattern}")
	tokens=torch.cat([load_data_shard(file)for file in files]).contiguous();usable=(tokens.numel()-1)//seq_len*seq_len
	if usable<=0:raise ValueError("val")
	return tokens[:usable+1]
def load_data_shard(file):
	header_bytes=256*np.dtype(_K).itemsize;token_bytes=np.dtype(_O).itemsize;header=np.fromfile(file,dtype=_K,count=256)
	if header.size!=256 or int(header[0])!=20240520 or int(header[1])!=1:raise ValueError("hdr")
	num_tokens=int(header[2]);expected_size=header_bytes+num_tokens*token_bytes
	if file.stat().st_size!=expected_size:raise ValueError("sz")
	tokens_np=np.fromfile(file,dtype=_O,count=num_tokens,offset=header_bytes)
	if tokens_np.size!=num_tokens:raise ValueError("rd")
	return torch.from_numpy(tokens_np.astype(np.uint16,copy=_C))
_SHARD_HEADER_BYTES=256*np.dtype(_K).itemsize
_SHARD_NTOKENS_CACHE={}
_MMAP_CACHE={}
def _read_num_tokens(file):
	key=str(file);cached=_SHARD_NTOKENS_CACHE.get(key)
	if cached is not _A:return cached
	header=np.fromfile(file,dtype=_K,count=256)
	if header.size!=256 or int(header[0])!=20240520 or int(header[1])!=1:raise ValueError("hdr")
	n=int(header[2]);_SHARD_NTOKENS_CACHE[key]=n;return n
def _get_shard_memmap(file):
	key=str(file);mm=_MMAP_CACHE.get(key)
	if mm is not _A:return mm
	n=_read_num_tokens(file);mm=np.memmap(file,mode='r',dtype=_O,offset=_SHARD_HEADER_BYTES,shape=(n,));_MMAP_CACHE[key]=mm;return mm
class ShuffledSequenceLoader:
	def __init__(self,h,device):
		self.world_size=h.world_size;self.seq_len=h.train_seq_len;self.device=device;all_files=[Path(p)for p in sorted(glob.glob(h.train_files))]
		if not all_files:raise FileNotFoundError(f"No files found for pattern: {h.train_files}")
		self.files=all_files[h.rank::h.world_size];self.rng=np.random.Generator(np.random.PCG64(h.rank));self.num_tokens=[_read_num_tokens(f)for f in self.files];self.start_inds=[[]for _ in self.files]
		for si in range(len(self.files)):self._reset_shard(si)
	def _reset_shard(self,si):max_phase=min(self.seq_len-1,max(0,self.num_tokens[si]-self.seq_len-1));phase=int(self.rng.integers(max_phase+1))if max_phase>0 else 0;num_sequences=(self.num_tokens[si]-1-phase)//self.seq_len;sequence_order=self.rng.permutation(num_sequences);self.start_inds[si]=(phase+sequence_order*self.seq_len).tolist()
	def next_batch(self,global_tokens,grad_accum_steps):
		device_tokens=global_tokens//(self.world_size*grad_accum_steps);device_batch_size=device_tokens//self.seq_len;remaining=np.array([len(s)for s in self.start_inds],dtype=np.float64);x=torch.empty((device_batch_size,self.seq_len),dtype=torch.int64);y=torch.empty((device_batch_size,self.seq_len),dtype=torch.int64)
		for bi in range(device_batch_size):
			total=remaining.sum()
			if total<=0:
				for si in range(len(self.files)):self._reset_shard(si)
				remaining=np.array([len(s)for s in self.start_inds],dtype=np.float64);total=remaining.sum()
			probs=remaining/total;si=int(self.rng.choice(len(self.files),p=probs));start_ind=self.start_inds[si].pop();remaining[si]-=1;mm=_get_shard_memmap(self.files[si]);window=torch.as_tensor(np.array(mm[start_ind:start_ind+self.seq_len+1],dtype=np.int64));x[bi]=window[:-1];y[bi]=window[1:]
		return x.to(self.device,non_blocking=_B),y.to(self.device,non_blocking=_B)
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
		if dim%num_heads!=0:raise ValueError('mh')
		if num_heads%num_kv_heads!=0:raise ValueError('kvh')
		self.num_heads=num_heads;self.num_kv_heads=num_kv_heads;self.head_dim=dim//num_heads
		if self.head_dim%2!=0:raise ValueError('hd')
		kv_dim=self.num_kv_heads*self.head_dim;self.c_q=CastedLinear(dim,dim,bias=_C);self.c_k=CastedLinear(dim,kv_dim,bias=_C);self.c_v=CastedLinear(dim,kv_dim,bias=_C);self.proj=CastedLinear(dim,dim,bias=_C);self.proj._zero_init=_B;self.q_gain=nn.Parameter(torch.full((num_heads,),qk_gain_init,dtype=torch.float32));self.rope_dims=0;self.rotary=Rotary(self.head_dim,base=rope_base,train_seq_len=train_seq_len);self.use_xsa=_C
	def _xsa_efficient(self,y,v):B,T,H,D=y.shape;Hkv=v.size(-2);group=H//Hkv;y_g=y.reshape(B,T,Hkv,group,D);vn=F.normalize(v,dim=-1).unsqueeze(-2);proj=(y_g*vn).sum(dim=-1,keepdim=_B)*vn;return(y_g-proj).reshape(B,T,H,D)
	def forward(self,x):
		bsz,seqlen,dim=x.shape;q=self.c_q(x).reshape(bsz,seqlen,self.num_heads,self.head_dim);k=self.c_k(x).reshape(bsz,seqlen,self.num_kv_heads,self.head_dim);v=self.c_v(x).reshape(bsz,seqlen,self.num_kv_heads,self.head_dim);q=F.rms_norm(q,(q.size(-1),));k=F.rms_norm(k,(k.size(-1),));cos,sin=self.rotary(seqlen,x.device,q.dtype);q=apply_rotary_emb(q,cos,sin,self.rope_dims);k=apply_rotary_emb(k,cos,sin,self.rope_dims);q=q*self.q_gain.to(dtype=q.dtype)[_A,_A,:,_A];y=flash_attn_3_func(q,k,v,causal=_B)
		if self.use_xsa:y=self._xsa_efficient(y,v)
		y=y.reshape(bsz,seqlen,dim);return self.proj(y)
class MLP(nn.Module):
	def __init__(self,dim,mlp_mult):super().__init__();hidden=int(mlp_mult*dim);self.fc=CastedLinear(dim,hidden,bias=_C);self.proj=CastedLinear(hidden,dim,bias=_C);self.proj._zero_init=_B
	def forward(self,x):return self.proj(F.leaky_relu(self.fc(x),negative_slope=.5).square())
class Block(nn.Module):
	def __init__(self,dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init,train_seq_len,layer_idx=0,ln_scale=_C):super().__init__();self.attn_norm=RMSNorm();self.mlp_norm=RMSNorm();self.attn=CausalSelfAttention(dim,num_heads,num_kv_heads,rope_base,qk_gain_init,train_seq_len);self.mlp=MLP(dim,mlp_mult);self.attn_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.mlp_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.resid_mix=nn.Parameter(torch.stack((torch.ones(dim),torch.zeros(dim))).float());self.ln_scale_factor=_D/math.sqrt(layer_idx+1)if ln_scale else _D;self.parallel=_C
	def forward(self,x,x0):
		mix=self.resid_mix.to(dtype=x.dtype);x_in=mix[0][_A,_A,:]*x+mix[1][_A,_A,:]*x0;attn_out=self.attn(self.attn_norm(x_in)*self.ln_scale_factor)
		if self.parallel:mlp_out=self.mlp(self.mlp_norm(x_in)*self.ln_scale_factor);x_out=x_in+self.attn_scale.to(dtype=x_in.dtype)[_A,_A,:]*attn_out+self.mlp_scale.to(dtype=x_in.dtype)[_A,_A,:]*mlp_out
		else:x_out=x_in+self.attn_scale.to(dtype=x_in.dtype)[_A,_A,:]*attn_out;x_out=x_out+self.mlp_scale.to(dtype=x_out.dtype)[_A,_A,:]*self.mlp(self.mlp_norm(x_out)*self.ln_scale_factor)
		return x_out
class GPT(nn.Module):
	def __init__(self,h):
		super().__init__()
		if h.logit_softcap<=_E:raise ValueError("ls")
		self.tie_embeddings=h.tie_embeddings;self.tied_embed_init_std=h.tied_embed_init_std;self.logit_softcap=h.logit_softcap;self.tok_emb=nn.Embedding(h.vocab_size,h.embedding_dim)
		if h.embedding_dim!=h.model_dim:self.embed_proj=CastedLinear(h.embedding_dim,h.model_dim,bias=_C);self.head_proj=CastedLinear(h.model_dim,h.embedding_dim,bias=_C)
		else:self.embed_proj=_A;self.head_proj=_A
		self.num_encoder_layers=h.num_layers//2;self.num_decoder_layers=h.num_layers-self.num_encoder_layers;self.blocks=nn.ModuleList([Block(h.model_dim,h.num_heads,h.num_kv_heads,h.mlp_mult,h.rope_base,h.qk_gain_init,h.train_seq_len,layer_idx=i,ln_scale=h.ln_scale)for i in range(h.num_layers)])
		if h.rope_dims>0:
			head_dim=h.model_dim//h.num_heads
			for block in self.blocks:block.attn.rope_dims=h.rope_dims;block.attn.rotary=Rotary(head_dim,base=h.rope_base,train_seq_len=h.train_seq_len,rope_dims=h.rope_dims)
		self.final_norm=RMSNorm();self.lm_head=_A if h.tie_embeddings else CastedLinear(h.embedding_dim,h.vocab_size,bias=_C)
		if self.lm_head is not _A:self.lm_head._zero_init=_B
		if h.xsa_last_n>0:
			for i in range(max(0,h.num_layers-h.xsa_last_n),h.num_layers):self.blocks[i].attn.use_xsa=_B
		if h.parallel_residual_start>=0:
			for i in range(h.parallel_residual_start,h.num_layers):self.blocks[i].parallel=_B
		self.looping_active=_C
		if h.num_loops>0:
			loop_seg=list(range(h.loop_start,h.loop_end+1));all_indices=list(range(h.loop_start))
			for _ in range(h.num_loops+1):all_indices.extend(loop_seg)
			all_indices.extend(range(h.loop_end+1,h.num_layers));num_enc=len(all_indices)//2;self.encoder_indices=all_indices[:num_enc];self.decoder_indices=all_indices[num_enc:]
		else:self.encoder_indices=list(range(self.num_encoder_layers));self.decoder_indices=list(range(self.num_encoder_layers,h.num_layers))
		self.num_skip_weights=min(len(self.encoder_indices),len(self.decoder_indices));self.skip_weights=nn.Parameter(torch.ones(self.num_skip_weights,h.model_dim,dtype=torch.float32));self.skip_gates=nn.Parameter(torch.zeros(self.num_skip_weights,h.model_dim,dtype=torch.float32))if h.skip_gates_enabled else _A;self._init_weights()
	def _init_weights(self):
		if self.tie_embeddings:nn.init.normal_(self.tok_emb.weight,mean=_E,std=self.tied_embed_init_std)
		for(name,module)in self.named_modules():
			if isinstance(module,nn.Linear):
				if getattr(module,'_zero_init',_C):nn.init.zeros_(module.weight)
				elif module.weight.ndim==2 and module.weight.shape[0]>=64 and module.weight.shape[1]>=64:nn.init.orthogonal_(module.weight,gain=_D)
	def forward_logits(self,input_ids):
		x=self.tok_emb(input_ids);x=F.rms_norm(x,(x.size(-1),))
		if self.embed_proj is not _A:x=self.embed_proj(x)
		x0=x;skips=[];enc_iter=self.encoder_indices if self.looping_active else range(self.num_encoder_layers);dec_iter=self.decoder_indices if self.looping_active else range(self.num_encoder_layers,self.num_encoder_layers+self.num_decoder_layers)
		for i in enc_iter:x=self.blocks[i](x,x0);skips.append(x)
		for(skip_idx,i)in enumerate(dec_iter):
			if skip_idx<self.num_skip_weights and skips:
				scaled_skip=self.skip_weights[skip_idx].to(dtype=x.dtype)[_A,_A,:]*skips.pop()
				if self.skip_gates is not _A:g=torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[_A,_A,:];x=torch.lerp(scaled_skip,x,g)
				else:x=x+scaled_skip
			x=self.blocks[i](x,x0)
		x=self.final_norm(x)
		if self.head_proj is not _A:x=self.head_proj(x)
		if self.tie_embeddings:logits_proj=F.linear(x,self.tok_emb.weight)
		else:logits_proj=self.lm_head(x)
		return self.logit_softcap*torch.tanh(logits_proj/self.logit_softcap)
	def forward(self,input_ids,target_ids):logits=self.forward_logits(input_ids);return F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),target_ids.reshape(-1),reduction='mean')
def classify_param(name):
	A='.mlp.'
	if _P in name or'lm_head'in name:return'embed'
	if A in name:return'mlp'
	if'.attn.'in name or'.proj.'in name and A not in name:return'attn'
	return'other'
@torch.compile
def zeropower_via_newtonschulz5(G,steps=10,eps=1e-07):
	a,b,c=3.4445,-4.775,2.0315;X=G.bfloat16();X/=X.norm()+eps;transposed=G.size(0)>G.size(1)
	if transposed:X=X.T
	for _ in range(steps):A=X@X.T;B=b*A+c*A@A;X=a*X+B@X
	return X.T if transposed else X
class Muon(torch.optim.Optimizer):
	def __init__(self,params,lr,momentum,backend_steps,nesterov=_B,weight_decay=_E,row_normalize=_C):super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,nesterov=nesterov,weight_decay=weight_decay,row_normalize=row_normalize))
	@torch.no_grad()
	def step(self,closure=_A):
		A='momentum_buffer';loss=_A
		if closure is not _A:
			with torch.enable_grad():loss=closure()
		distributed=dist.is_available()and dist.is_initialized();world_size=dist.get_world_size()if distributed else 1;rank=dist.get_rank()if distributed else 0
		for group in self.param_groups:
			params=group[_L]
			if not params:continue
			lr=group[_G];momentum=group[_S];backend_steps=group['backend_steps'];nesterov=group['nesterov'];total_params=sum(int(p.numel())for p in params);updates_flat=torch.zeros(total_params,device=params[0].device,dtype=torch.bfloat16);curr=0
			for(i,p)in enumerate(params):
				if i%world_size==rank and p.grad is not _A:
					g=p.grad;state=self.state[p]
					if A not in state:state[A]=torch.zeros_like(g)
					buf=state[A];buf.mul_(momentum).add_(g)
					if nesterov:g=g.add(buf,alpha=momentum)
					if group.get('row_normalize',_C):row_norms=g.float().norm(dim=-1,keepdim=_B).clamp_min(1e-07);g=g/row_norms.to(g.dtype)
					g=zeropower_via_newtonschulz5(g,steps=backend_steps);g*=max(1,g.size(0)/g.size(1))**.5;updates_flat[curr:curr+p.numel()]=g.reshape(-1)
				curr+=p.numel()
			if distributed:dist.all_reduce(updates_flat,op=dist.ReduceOp.SUM)
			wd=group.get('weight_decay',_E);curr=0
			for p in params:
				if wd>_E:p.data.mul_(_D-lr*wd)
				g=updates_flat[curr:curr+p.numel()].view_as(p).to(dtype=p.dtype);p.add_(g,alpha=-lr);curr+=p.numel()
		return loss
CONTROL_TENSOR_NAME_PATTERNS=tuple(pattern for pattern in os.environ.get('CONTROL_TENSOR_NAME_PATTERNS','attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates').split(',')if pattern)
class Optimizers:
	def __init__(self,h,base_model):
		block_named_params=list(base_model.blocks.named_parameters());matrix_params=[p for(name,p)in block_named_params if p.ndim==2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)];scalar_params=[p for(name,p)in block_named_params if p.ndim<2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)]
		if base_model.skip_weights.numel()>0:scalar_params.append(base_model.skip_weights)
		if base_model.skip_gates is not _A and base_model.skip_gates.numel()>0:scalar_params.append(base_model.skip_gates)
		token_lr=h.tied_embed_lr if h.tie_embeddings else h.embed_lr;tok_params=[{_L:[base_model.tok_emb.weight],_G:token_lr,_H:token_lr}];self.optimizer_tok=torch.optim.AdamW(tok_params,betas=(h.beta1,h.beta2),eps=h.adam_eps,weight_decay=h.embed_wd,fused=_B);self.optimizer_muon=Muon(matrix_params,lr=h.matrix_lr,momentum=h.muon_momentum,backend_steps=h.muon_backend_steps,weight_decay=h.muon_wd,row_normalize=h.muon_row_normalize)
		for group in self.optimizer_muon.param_groups:group[_H]=h.matrix_lr
		self.optimizer_scalar=torch.optim.AdamW([{_L:scalar_params,_G:h.scalar_lr,_H:h.scalar_lr}],betas=(h.beta1,h.beta2),eps=h.adam_eps,weight_decay=h.adam_wd,fused=_B);self.optimizers=[self.optimizer_tok,self.optimizer_muon,self.optimizer_scalar]
		if base_model.lm_head is not _A:self.optimizer_head=torch.optim.Adam([{_L:[base_model.lm_head.weight],_G:h.head_lr,_H:h.head_lr}],betas=(h.beta1,h.beta2),eps=h.adam_eps,fused=_B);self.optimizers.insert(1,self.optimizer_head)
		else:self.optimizer_head=_A
	def __iter__(self):return iter(self.optimizers)
	def zero_grad_all(self):
		for opt in self.optimizers:opt.zero_grad(set_to_none=_B)
	def step(self):
		for opt in self.optimizers:opt.step()
		self.zero_grad_all()
def restore_fp32_params(model):
	for module in model.modules():
		if isinstance(module,CastedLinear):module.float()
	for(name,param)in model.named_parameters():
		if(param.ndim<2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))and param.dtype!=torch.float32:param.data=param.data.float()
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
			if cat in('mlp','attn'):hooks.append(module.register_forward_hook(make_hook(name+A)))
	if model.tie_embeddings:
		hook_module=model.head_proj if model.head_proj is not _A else model.final_norm
		def make_output_hook(name):
			def hook_fn(module,inp,out):
				x=out.detach().float()
				if x.ndim==3:x=x.reshape(-1,x.shape[-1])
				if name not in hessians:hessians[name]=torch.zeros(x.shape[1],x.shape[1],dtype=torch.float32,device=device)
				hessians[name].addmm_(x.T,x)
			return hook_fn
		hooks.append(hook_module.register_forward_hook(make_output_hook('tok_emb.weight')))
	model.eval()
	with torch.no_grad():
		for _ in range(n_calibration_batches):x,_=train_loader.next_batch(h.train_batch_tokens,h.grad_accum_steps);model.forward_logits(x)
	for hook in hooks:hook.remove()
	for name in hessians:hessians[name]=hessians[name].cpu()/n_calibration_batches
	return hessians
def gptq_quantize_weight(w,H,clip_sigmas=3.,clip_range=63,block_size=128):
	W_orig=w.float().clone();rows,cols=W_orig.shape;H=H.float().clone();dead=torch.diag(H)==0;H[dead,dead]=1;damp=.01*H.diag().mean();H.diagonal().add_(damp);perm=torch.argsort(H.diag(),descending=_B);invperm=torch.argsort(perm);W_perm=W_orig[:,perm].clone();W_perm[:,dead[perm]]=0;H=H[perm][:,perm];Hinv=torch.cholesky_inverse(torch.linalg.cholesky(H));Hinv=torch.linalg.cholesky(Hinv,upper=_B);row_std=W_orig.std(dim=1);s=(clip_sigmas*row_std/clip_range).clamp_min(1e-10).to(torch.float16);sf=s.float();Q=torch.zeros(rows,cols,dtype=torch.int8);W_work=W_perm.clone()
	for i1 in range(0,cols,block_size):
		i2=min(i1+block_size,cols);W_block=W_work[:,i1:i2].clone();Hinv_block=Hinv[i1:i2,i1:i2];Err=torch.zeros(rows,i2-i1)
		for j in range(i2-i1):w_col=W_block[:,j];d=Hinv_block[j,j];q_col=torch.clamp(torch.round(w_col/sf),-clip_range,clip_range);Q[:,i1+j]=q_col.to(torch.int8);err=(w_col-q_col.float()*sf)/d;Err[:,j]=err;W_block[:,j:]-=err.unsqueeze(1)*Hinv_block[j,j:].unsqueeze(0)
		if i2<cols:W_work[:,i2:]-=Err@Hinv[i1:i2,i2:]
	return Q[:,invperm],s
def gptq_mixed_quantize(state_dict,hessians,h):
	result={};meta={}
	for(name,tensor)in state_dict.items():
		t=tensor.detach().cpu().contiguous()
		if not t.is_floating_point()or t.numel()<=65536:result[name]=t.to(torch.float16)if t.is_floating_point()else t;meta[name]='passthrough (float16)';continue
		cs=h.embed_clip_sigmas if _P in name else h.matrix_clip_sigmas;bits=h.embed_bits if _P in name else h.matrix_bits;q,s=gptq_quantize_weight(t,hessians[name],clip_sigmas=cs,clip_range=2**(bits-1)-1);result[name+'.q']=q;result[name+_T]=s;meta[name]=f"gptq (int{bits})"
	categories=collections.defaultdict(set)
	for(name,cat)in meta.items():short=re.sub('\\.\\d+$','',re.sub('blocks\\.\\d+','blocks',name));categories[cat].add(short)
	log('Quantized weights:')
	for cat in sorted(categories):log(f"  {cat}: {", ".join(sorted(categories[cat]))}")
	return result,meta
def dequantize_mixed(result,meta,template_sd):
	out={}
	for(name,orig)in template_sd.items():
		info=meta.get(name)
		if info is _A:continue
		orig_dtype=orig.dtype
		if'passthrough'in info:
			t=result[name]
			if t.dtype==torch.float16 and orig_dtype in(torch.float32,torch.bfloat16):t=t.to(orig_dtype)
			out[name]=t;continue
		q,s=result[name+'.q'],result[name+_T]
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
def _compress(data,compressor):
	data=_byte_shuffle(data)
	if compressor==_M:import brotli;return brotli.compress(data,quality=11)
	raise ValueError("cmp")
def _decompress(data,compressor):
	if compressor==_M:import brotli;raw=brotli.decompress(data)
	else:raise ValueError("cmp")
	raw=_byte_unshuffle(raw);return raw
def serialize(h,base_model,code):
	code_bytes=len(code.encode(_J))
	if h.is_main_process:torch.save(base_model.state_dict(),h.model_path);model_bytes=os.path.getsize(h.model_path);log(f"Serialized model: {model_bytes} bytes");log(f"Code size: {code_bytes} bytes")
	sd_cpu={k:v.detach().cpu()for(k,v)in base_model.state_dict().items()};device=torch.device(_F,h.local_rank);log('GPTQ:H');t0=time.perf_counter();calib_loader=ShuffledSequenceLoader(h,device);hessians=collect_hessians(base_model,calib_loader,h,device,n_calibration_batches=h.gptq_calibration_batches);log(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter()-t0:.1f}s");quant_result,quant_meta=gptq_mixed_quantize(sd_cpu,hessians,h);quant_buf=io.BytesIO();torch.save({'w':quant_result,'m':quant_meta},quant_buf);quant_raw=quant_buf.getvalue();quant_blob=_compress(quant_raw,h.compressor);quant_file_bytes=len(quant_blob);bytes_total=quant_file_bytes+code_bytes
	if h.is_main_process:
		with open(h.quantized_model_path,'wb')as f:f.write(quant_blob)
		log(f"Serialized model quantized+{h.compressor}: {quant_file_bytes} bytes");log(f"Total submission size quantized+{h.compressor}: {bytes_total} bytes")
	return bytes_total,quant_file_bytes
def deserialize(h,device):
	eval_model=GPT(h).to(device).bfloat16();restore_fp32_params(eval_model);sd_cpu={k:v.detach().cpu()for(k,v)in eval_model.state_dict().items()}
	with open(h.quantized_model_path,'rb')as f:quant_blob_disk=f.read()
	quant_state=torch.load(io.BytesIO(_decompress(quant_blob_disk,h.compressor)),map_location='cpu');deq_state=dequantize_mixed(quant_state['w'],quant_state['m'],sd_cpu);eval_model.load_state_dict(deq_state,strict=_B);return eval_model
def _loss_bpb(loss_sum,token_count,byte_count):val_loss=(loss_sum/token_count).item();val_bpb=val_loss/math.log(2.)*(token_count.item()/byte_count.item());return val_loss,val_bpb
def eval_val(h,device,val_data,model):
	seq_len=h.eval_seq_len;local_batch_tokens=h.val_batch_tokens//(h.world_size*h.grad_accum_steps)
	if local_batch_tokens<seq_len:raise ValueError("vbs")
	local_batch_seqs=local_batch_tokens//seq_len;total_seqs=(val_data.val_tokens.numel()-1)//seq_len;seq_start=total_seqs*h.rank//h.world_size;seq_end=total_seqs*(h.rank+1)//h.world_size;val_loss_sum=torch.zeros((),device=device,dtype=torch.float64);val_token_count=torch.zeros((),device=device,dtype=torch.float64);val_byte_count=torch.zeros((),device=device,dtype=torch.float64);model.eval()
	with torch.inference_mode():
		for batch_seq_start in range(seq_start,seq_end,local_batch_seqs):
			batch_seq_end=min(batch_seq_start+local_batch_seqs,seq_end);raw_start=batch_seq_start*seq_len;raw_end=batch_seq_end*seq_len+1;local=val_data.val_tokens[raw_start:raw_end].to(device=device,dtype=torch.int64,non_blocking=_B);x=local[:-1].reshape(-1,seq_len);y=local[1:].reshape(-1,seq_len)
			with torch.autocast(device_type=_F,dtype=torch.bfloat16,enabled=_B):batch_loss=model(x,y).detach()
			batch_token_count=float(y.numel());val_loss_sum+=batch_loss.to(torch.float64)*batch_token_count;val_token_count+=batch_token_count;prev_ids=x.reshape(-1);tgt_ids=y.reshape(-1);token_bytes=val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16);token_bytes+=(val_data.has_leading_space_lut[tgt_ids]&~val_data.is_boundary_token_lut[prev_ids]).to(dtype=torch.int16);val_byte_count+=token_bytes.to(torch.float64).sum()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(val_loss_sum,op=dist.ReduceOp.SUM);dist.all_reduce(val_token_count,op=dist.ReduceOp.SUM);dist.all_reduce(val_byte_count,op=dist.ReduceOp.SUM)
	model.train();return _loss_bpb(val_loss_sum,val_token_count,val_byte_count)
def eval_val_sliding(h,device,val_data,base_model,batch_seqs=32):
	base_model.eval();logits_fn=torch.compile(base_model.forward_logits,dynamic=_C,fullgraph=_B);seq_len=h.eval_seq_len;context_size=seq_len-h.eval_stride;total_tokens=val_data.val_tokens.numel()-1;window_starts=[ws for ws in range(0,total_tokens,h.eval_stride)if ws+context_size<total_tokens];total_windows=len(window_starts);my_s=total_windows*h.rank//h.world_size;my_e=total_windows*(h.rank+1)//h.world_size;my_windows=window_starts[my_s:my_e];loss_sum=torch.zeros((),device=device,dtype=torch.float64);token_count=torch.zeros((),device=device,dtype=torch.float64);byte_count=torch.zeros((),device=device,dtype=torch.float64)
	with torch.inference_mode():
		for bi in range(0,len(my_windows),batch_seqs):
			batch_ws=my_windows[bi:bi+batch_seqs];bsz=len(batch_ws);x_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);y_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);wlens=[]
			for(i,ws)in enumerate(batch_ws):we=min(ws+seq_len,total_tokens);wlen=we-ws;wlens.append(wlen);chunk=val_data.val_tokens[ws:we+1].to(dtype=torch.int64,device=device);x_batch[i,:wlen]=chunk[:-1];y_batch[i,:wlen]=chunk[1:]
			with torch.autocast(device_type=_F,dtype=torch.bfloat16):logits=logits_fn(x_batch)
			nll=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),y_batch.reshape(-1),reduction='none').reshape(bsz,seq_len)
			for(i,ws)in enumerate(batch_ws):wlen=wlens[i];s=0 if ws==0 else context_size;scored_nll=nll[i,s:wlen].to(torch.float64);loss_sum+=scored_nll.sum();token_count+=float(wlen-s);tgt=y_batch[i,s:wlen];prev=x_batch[i,s:wlen];tb=val_data.base_bytes_lut[tgt].to(torch.float64);tb+=(val_data.has_leading_space_lut[tgt]&~val_data.is_boundary_token_lut[prev]).to(torch.float64);byte_count+=tb.sum()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(loss_sum,op=dist.ReduceOp.SUM);dist.all_reduce(token_count,op=dist.ReduceOp.SUM);dist.all_reduce(byte_count,op=dist.ReduceOp.SUM)
	base_model.train();return _loss_bpb(loss_sum,token_count,byte_count)
def eval_val_ttt(h,device,val_data,base_model,batch_seqs=32):
	rank=h.rank;world_size=h.world_size;seq_len=h.eval_seq_len;stride=h.eval_stride;total_tokens=val_data.val_tokens.numel()-1;ttt_chunk=h.ttt_chunk_tokens;context_size=seq_len-stride;window_starts=[ws for ws in range(0,total_tokens,stride)if ws+context_size<total_tokens];num_chunks=(total_tokens+ttt_chunk-1)//ttt_chunk;chunk_windows=[[]for _ in range(num_chunks)]
	for ws in window_starts:wlen=min(ws+seq_len,total_tokens)-ws;s=0 if ws==0 else context_size;scored_start=ws+s;ci=min(scored_start//ttt_chunk,num_chunks-1);chunk_windows[ci].append(ws)
	log(f"ttt:start chunks={num_chunks} ttt_lr={h.ttt_lr} ttt_epochs={h.ttt_epochs}");compiled_logits=torch.compile(base_model.forward_logits,dynamic=_C,fullgraph=_B);loss_sum=torch.zeros((),device=device,dtype=torch.float64);token_count=torch.zeros((),device=device,dtype=torch.float64);byte_count=torch.zeros((),device=device,dtype=torch.float64);ttt_params=[p for p in base_model.parameters()]
	for p in ttt_params:p.requires_grad_(_B)
	optimizer=torch.optim.SGD(ttt_params,lr=h.ttt_lr,momentum=h.ttt_momentum)
	for ci in range(num_chunks):
		windows=chunk_windows[ci]
		if not windows:continue
		chunk_start=ci*ttt_chunk;chunk_end=min((ci+1)*ttt_chunk,total_tokens);my_s=len(windows)*rank//world_size;my_e=len(windows)*(rank+1)//world_size;my_windows=windows[my_s:my_e];base_model.eval()
		with torch.no_grad():
			for bi in range(0,len(my_windows),batch_seqs):
				batch_ws=my_windows[bi:bi+batch_seqs];bsz=len(batch_ws);x_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);y_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);wlens=[]
				for(i,ws)in enumerate(batch_ws):we=min(ws+seq_len,total_tokens);wlen=we-ws;wlens.append(wlen);chunk_tok=val_data.val_tokens[ws:we+1].to(dtype=torch.int64,device=device);x_batch[i,:wlen]=chunk_tok[:-1];y_batch[i,:wlen]=chunk_tok[1:]
				with torch.autocast(device_type=_F,dtype=torch.bfloat16):logits=compiled_logits(x_batch)
				nll=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),y_batch.reshape(-1),reduction='none').reshape(bsz,seq_len)
				for(i,ws)in enumerate(batch_ws):wlen=wlens[i];s=0 if ws==0 else context_size;scored_nll=nll[i,s:wlen].to(torch.float64);loss_sum+=scored_nll.sum();token_count+=float(wlen-s);tgt=y_batch[i,s:wlen];prev=x_batch[i,s:wlen];tb=val_data.base_bytes_lut[tgt].to(torch.float64);tb+=(val_data.has_leading_space_lut[tgt]&~val_data.is_boundary_token_lut[prev]).to(torch.float64);byte_count+=tb.sum()
		is_last_chunk=ci==num_chunks-1
		if not is_last_chunk and h.ttt_epochs>0:
			base_model.train();chunk_seqs=(chunk_end-chunk_start)//seq_len
			if chunk_seqs>0:
				cos_lr=h.ttt_lr*.5*(_D+math.cos(math.pi*ci/max(num_chunks-1,1)))
				for pg in optimizer.param_groups:pg[_G]=cos_lr
				my_seq_s=chunk_seqs*rank//world_size;my_seq_e=chunk_seqs*(rank+1)//world_size;my_chunk_seqs=my_seq_e-my_seq_s
				for _ep in range(h.ttt_epochs):
					for bs in range(0,my_chunk_seqs,batch_seqs):
						be=min(bs+batch_seqs,my_chunk_seqs);actual_bs=my_seq_s+bs;start_tok=chunk_start+actual_bs*seq_len;end_tok=chunk_start+(my_seq_s+be)*seq_len+1
						if end_tok>val_data.val_tokens.numel():continue
						local=val_data.val_tokens[start_tok:end_tok].to(device=device,dtype=torch.int64);x=local[:-1].reshape(-1,seq_len);y=local[1:].reshape(-1,seq_len);optimizer.zero_grad(set_to_none=_B)
						with torch.autocast(device_type=_F,dtype=torch.bfloat16):loss=base_model(x,y)
						loss.backward()
						if world_size>1:
							for p in ttt_params:
								if p.grad is not _A:dist.all_reduce(p.grad,op=dist.ReduceOp.AVG)
						torch.nn.utils.clip_grad_norm_(ttt_params,_D);optimizer.step()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(loss_sum,op=dist.ReduceOp.SUM);dist.all_reduce(token_count,op=dist.ReduceOp.SUM);dist.all_reduce(byte_count,op=dist.ReduceOp.SUM)
	for p in base_model.parameters():p.requires_grad_(_B)
	base_model.eval();return _loss_bpb(loss_sum,token_count,byte_count)
def timed_eval(label,fn,*args,**kwargs):torch.cuda.synchronize();t0=time.perf_counter();val_loss,val_bpb=fn(*args,**kwargs);torch.cuda.synchronize();elapsed_ms=1e3*(time.perf_counter()-t0);log(f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms");return val_loss,val_bpb
def train_model(h,device,val_data):
	base_model=GPT(h).to(device).bfloat16();restore_fp32_params(base_model);compiled_model=torch.compile(base_model,dynamic=_C,fullgraph=_B)
	if h.distributed:model=DDP(compiled_model,device_ids=[h.local_rank],broadcast_buffers=_C)
	else:model=compiled_model
	log(f"model_params:{sum(p.numel()for p in base_model.parameters())}");optimizers=Optimizers(h,base_model);train_loader=ShuffledSequenceLoader(h,device);max_wallclock_ms=1e3*h.max_wallclock_seconds if h.max_wallclock_seconds>0 else _A
	if max_wallclock_ms is not _A:max_wallclock_ms-=h.gptq_reserve_seconds*1e3;log(f"gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms")
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
			x,y=train_loader.next_batch(h.train_batch_tokens,h.grad_accum_steps)
			with torch.autocast(device_type=_F,dtype=torch.bfloat16,enabled=_B):loss=model(x,y)
			train_loss+=loss.detach();(loss/h.grad_accum_steps).backward()
		train_loss/=h.grad_accum_steps;frac=min(step/h.muon_momentum_warmup_steps,_D)if h.muon_momentum_warmup_steps>0 else _D;muon_momentum=(1-frac)*h.muon_momentum_warmup_start+frac*h.muon_momentum
		for group in optimizers.optimizer_muon.param_groups:group[_S]=muon_momentum
		for opt in optimizers:
			for group in opt.param_groups:group[_G]=group[_H]*lr_scale
		if h.grad_clip_norm>0:torch.nn.utils.clip_grad_norm_(base_model.parameters(),h.grad_clip_norm)
		optimizers.step();return train_loss
	if h.warmup_steps>0:
		initial_model_state={name:tensor.detach().cpu().clone()for(name,tensor)in base_model.state_dict().items()};initial_optimizer_states=[copy.deepcopy(opt.state_dict())for opt in optimizers];model.train()
		for warmup_step in range(h.warmup_steps):
			step_fn(warmup_step,_D)
			if warmup_step<=5 or(warmup_step+1)%10==0 or warmup_step+1==h.warmup_steps:log(f"warmup_step: {warmup_step+1}/{h.warmup_steps}")
		if h.num_loops>0:
			base_model.looping_active=_B;log(f"loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")
			for warmup_step in range(h.warmup_steps):
				step_fn(warmup_step,_D)
				if warmup_step<=5 or(warmup_step+1)%10==0 or warmup_step+1==h.warmup_steps:log(f"loop_warmup_step: {warmup_step+1}/{h.warmup_steps}")
			base_model.looping_active=_C
		base_model.load_state_dict(initial_model_state,strict=_B)
		for(opt,state)in zip(optimizers,initial_optimizer_states,strict=_B):opt.load_state_dict(state)
		optimizers.zero_grad_all()
		if h.distributed:model.require_backward_grad_sync=_B
		train_loader=ShuffledSequenceLoader(h,device)
	ema_state={name:t.detach().float().clone()for(name,t)in base_model.state_dict().items()};ema_decay=h.ema_decay;training_time_ms=_E;stop_after_step=_A;torch.cuda.synchronize();t0=time.perf_counter();step=0
	while _B:
		last_step=step==h.iterations or stop_after_step is not _A and step>=stop_after_step;should_validate=last_step or h.val_loss_every>0 and step%h.val_loss_every==0
		if should_validate:torch.cuda.synchronize();training_time_ms+=1e3*(time.perf_counter()-t0);val_loss,val_bpb=eval_val(h,device,val_data,model);log(f"{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}");torch.cuda.synchronize();t0=time.perf_counter()
		if last_step:
			if stop_after_step is not _A and step<h.iterations:log(f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms step: {step}/{h.iterations}")
			break
		elapsed_ms=training_time_ms+1e3*(time.perf_counter()-t0);frac=training_frac(step,elapsed_ms);scale=lr_mul(frac)
		if h.num_loops>0 and not base_model.looping_active and frac>=h.enable_looping_at:base_model.looping_active=_B;log(f"layer_loop:enabled step:{step} frac:{frac:.3f} encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")
		train_loss=step_fn(step,scale)
		with torch.no_grad():
			for(name,t)in base_model.state_dict().items():ema_state[name].mul_(ema_decay).add_(t.detach().float(),alpha=_D-ema_decay)
		step+=1;approx_training_time_ms=training_time_ms+1e3*(time.perf_counter()-t0);should_log_train=h.train_log_every>0 and(step<=5 or step%h.train_log_every==0 or stop_after_step is not _A)
		if should_log_train:tok_per_sec=step*h.train_batch_tokens/(approx_training_time_ms/1e3);log(f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} train_time: {approx_training_time_ms/60000:.1f}m tok/s: {tok_per_sec:.0f}")
		reached_cap=max_wallclock_ms is not _A and approx_training_time_ms>=max_wallclock_ms
		if h.distributed and max_wallclock_ms is not _A:reached_cap_tensor=torch.tensor(int(reached_cap),device=device);dist.all_reduce(reached_cap_tensor,op=dist.ReduceOp.MAX);reached_cap=bool(reached_cap_tensor.item())
		if stop_after_step is _A and reached_cap:stop_after_step=step
	log(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB");log('ema:applying EMA weights');current_state=base_model.state_dict();avg_state={name:t.to(dtype=current_state[name].dtype)for(name,t)in ema_state.items()};base_model.load_state_dict(avg_state,strict=_B);return base_model,compiled_model
def train_and_eval(h,device):
	random.seed(h.seed);np.random.seed(h.seed);torch.manual_seed(h.seed);torch.cuda.manual_seed_all(h.seed);val_data=ValidationData(h,device);log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob(_R)))}");log(f"val_tokens: {val_data.val_tokens.numel()-1}");base_model,compiled_model=train_model(h,device,val_data);torch._dynamo.reset();timed_eval('pre-quantization post-ema',eval_val,h,device,val_data,compiled_model);serialize(h,base_model,Path(__file__).read_text(encoding=_J))
	if h.distributed:dist.barrier()
	eval_model=deserialize(h,device)
	if h.num_loops>0:eval_model.looping_active=_B
	compiled_model=torch.compile(eval_model,dynamic=_C,fullgraph=_B);timed_eval('quantized',eval_val,h,device,val_data,compiled_model)
	if h.sliding_window_enabled:timed_eval('quantized_sliding_window',eval_val_sliding,h,device,val_data,eval_model)
	if h.ttt_enabled and h.sliding_window_enabled:
		del eval_model,compiled_model;torch._dynamo.reset();torch.cuda.empty_cache();ttt_model=deserialize(h,device)
		if h.num_loops>0:ttt_model.looping_active=_B
		timed_eval('quantized_ttt',eval_val_ttt,h,device,val_data,ttt_model);del ttt_model
def main():
	A='=';world_size=int(os.environ.get(_I,'1'));local_rank=int(os.environ.get(_Q,'0'));distributed=_N in os.environ and _I in os.environ
	if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
	if world_size<=0:raise ValueError("ws+")
	if 8%world_size!=0:raise ValueError("ws")
	device=torch.device(_F,local_rank);torch.cuda.set_device(device)
	if distributed:dist.init_process_group(backend='nccl',device_id=device);dist.barrier()
	torch.backends.cuda.matmul.allow_tf32=_B;torch.backends.cudnn.allow_tf32=_B;torch.set_float32_matmul_precision('high');from torch.backends.cuda import enable_cudnn_sdp,enable_flash_sdp,enable_math_sdp,enable_mem_efficient_sdp;enable_cudnn_sdp(_C);enable_flash_sdp(_B);enable_mem_efficient_sdp(_C);enable_math_sdp(_C);torch._dynamo.config.optimize_ddp=_C;h=Hyperparameters();set_logging_hparams(h)
	if h.is_main_process:
		os.makedirs('logs',exist_ok=_B);log(100*A,console=_C);log('Hyperparameters:',console=_B)
		for(k,v)in sorted(vars(type(h)).items()):
			if not k.startswith('_'):log(f"  {k}: {v}",console=_B)
		log(A*100,console=_C);log(f"Running Python {sys.version}",console=_C);log(f"Running PyTorch {torch.__version__}",console=_C);log(subprocess.run(['nvidia-smi'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=_B,check=_C).stdout,console=_C);log(A*100,console=_C)
	train_and_eval(h,device)
	if distributed:dist.destroy_process_group()
if __name__=='__main__':main()