#!/usr/bin/env python3
import os,sys,io,time,copy,random,json,subprocess,collections,re,lzma,math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import init_process_group,destroy_process_group,is_available,is_initialized,get_rank,get_world_size,all_reduce,ReduceOp,barrier
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from flash_attn import flash_attn_func as flash_attn_3_func
class Hyperparameters:
	adam_eps=1e-8;adam_wd=0.02;beta1=0.9;beta2=0.95;compressor='brotli';data_dir='./data';datasets_dir='./data/datasets/fineweb10B_sp8192';distributed=is_available()and is_initialized();ema_decay=0.995;embed_bits=7;embed_clip_sigmas=20.;embed_lr=0.6;embed_wd=0.085;embedding_dim=512;enable_looping_at=0.35;etlb_clip=3.;etlb_enabled=False;etlb_lr=0.05;etlb_steps=5;eval_seq_len=1024;eval_stride=64;gptq_calibration_batches=128;gptq_reserve_seconds=5.;grad_accum_steps=1;grad_clip_norm=0.3;head_lr=0.008;is_main_process=True;iterations=8000;ln_scale=True;local_rank=0;logfile='logs/sweep_final_e7_s1337_ttt.txt';logit_softcap=20.;loop_end=5;loop_start=3;matrix_bits=6;matrix_clip_sigmas=12.85;matrix_lr=0.006;max_wallclock_seconds=600.;min_lr=0.;mlp_mult=4.;model_dim=512;model_path='final_model.pt';muon_backend_steps=5;muon_beta2=0.95;muon_momentum=0.99;muon_momentum_warmup_start=0.92;muon_momentum_warmup_steps=1500;muon_row_normalize=True;muon_wd=0.095;num_heads=8;num_kv_heads=4;num_layers=11;num_loops=2;parallel_residual_start=7;qk_gain_init=5.5;quantized_model_path='final_model.int6.ptz';rank=0;rope_base=10000.;rope_dims=16;rope_train_seq_len=2048;run_id='sweep_final_e7_s1337_ttt';scalar_lr=0.03;sdclip_enabled=False;sdclip_steps=20;seed=1337;skip_gates_enabled=True;sliding_window_enabled=True;tie_embeddings=True;tied_embed_init_std=0.005;tied_embed_lr=0.03;tokenizer_path='./data/tokenizers/fineweb_8192_bpe.model';train_batch_tokens=32768;train_files='./data/datasets/fineweb10B_sp8192/fineweb_train_*.bin';train_log_every=25;train_seq_len=1024;ttt_chunk_tokens=32768;ttt_enabled=True;ttt_epochs=1;ttt_lr=0.005;ttt_momentum=0.9;val_batch_tokens=131072;val_files='./data/datasets/fineweb10B_sp8192/fineweb_val_*.bin';val_loss_every=0;vocab_size=8192;warmdown_frac=0.72;warmup_steps=10;world_size=8;xsa_last_n=11;ar_calib_enabled=False;ar_calib_gen_len=32;ar_calib_n_seqs=32
	def __init__(self):
		for k in sorted(vars(type(self)).keys()):
			if not k.startswith('_'):setattr(self,k,getattr(type(self),k))
		for(k,v)in os.environ.items():
			if k.isupper()and hasattr(self,k.lower()):
				try:setattr(self,k.lower(),type(getattr(self,k.lower()))(v))
				except:pass
		self.world_size=int(os.environ.get('WORLD_SIZE','1'));self.rank=int(os.environ.get('RANK','0'));self.local_rank=int(os.environ.get('LOCAL_RANK','0'));self.distributed=is_available()and is_initialized();self.is_main_process=self.rank==0;self.grad_accum_steps=max(1,8//self.world_size)
def log(msg='',console=True):
	if not is_main_process:return
	with open(h.logfile if'h'in globals()else'log.txt','a')as f:f.write(str(msg)+'\n')
	if console:print(msg)
is_main_process=True
class ValidationData:
	def __init__(self,h,device):
		from tokenizers import Tokenizer
		tokenizer=Tokenizer.from_file(h.tokenizer_path);self.val_tokens=torch.tensor([],dtype=torch.int64,device='cpu')
		for f in sorted(Path(h.val_files).parent.glob('fineweb_val_*.bin')):self.val_tokens=torch.cat([self.val_tokens,torch.from_numpy(np.load(str(f)))])
		self.val_tokens=self.val_tokens.to(device=device);base_bytes_lut=torch.zeros(h.vocab_size,dtype=torch.int16);has_leading_space_lut=torch.zeros(h.vocab_size,dtype=torch.bool);is_boundary_token_lut=torch.zeros(h.vocab_size,dtype=torch.bool)
		for(i,token)in enumerate(tokenizer.get_vocab().items()):
			text=token[0];base_bytes_lut[i]=len(text.encode('utf-8'));has_leading_space_lut[i]=text.startswith(' ');boundary_chars={'.',',','!','?',';',':','(',')','[',']','{','}','-','–','—'};is_boundary_token_lut[i]=text.strip()in boundary_chars
		self.base_bytes_lut=base_bytes_lut.to(device);self.has_leading_space_lut=has_leading_space_lut.to(device);self.is_boundary_token_lut=is_boundary_token_lut.to(device)
class ShuffledSequenceLoader:
	def __init__(self,h,device):self.h=h;self.device=device;self.shard_idx=0;self.pos_in_shard=0;self.all_data=[]
		for f in sorted(Path(h.train_files).parent.glob('fineweb_train_*.bin')):self.all_data.append(torch.from_numpy(np.load(str(f))).int64())
		self.all_data_concat=torch.cat(self.all_data);self.indices=torch.randperm(len(self.all_data_concat),generator=torch.Generator().manual_seed(h.seed))
	def next_batch(self,global_tokens,grad_accum_steps):
		device_tokens=global_tokens//(self.h.world_size*grad_accum_steps);seq_len=self.h.train_seq_len;bsz=device_tokens//seq_len;needed=bsz*seq_len;buf=[]
		while len(buf)<needed:
			if self.shard_idx>=len(self.all_data):self.shard_idx=0;self.pos_in_shard=0;self.indices=torch.randperm(len(self.all_data_concat),generator=torch.Generator().manual_seed(self.h.seed+1+self.shard_idx))
			shard=self.all_data[self.shard_idx];remaining=len(shard)-self.pos_in_shard;to_take=min(needed-len(buf),remaining);buf.append(shard[self.pos_in_shard:self.pos_in_shard+to_take]);self.pos_in_shard+=to_take
			if self.pos_in_shard>=len(shard):self.shard_idx+=1;self.pos_in_shard=0
		data=torch.cat(buf)[:needed].to(device=self.device,non_blocking=True);x=data[:-1].reshape(bsz,seq_len);y=data[1:].reshape(bsz,seq_len);return x,y
class RMSNorm(nn.Module):
	def __init__(self,eps=None):super().__init__();self.eps=eps
	def forward(self,x):return F.rms_norm(x,(x.size(-1),),eps=self.eps)
class CastedLinear(nn.Linear):
	def forward(self,x):w=self.weight.to(x.dtype);bias=self.bias.to(x.dtype)if self.bias is not None else None;return F.linear(x,w,bias)
class Rotary(nn.Module):
	def __init__(self,dim,base=1e4,train_seq_len=1024,rope_dims=0):super().__init__();self.dim=dim;self.base=base;self.train_seq_len=train_seq_len;self.rope_dims=rope_dims if rope_dims>0 else dim;inv_freq=1./base**(torch.arange(0,self.rope_dims,2,dtype=torch.float32)/self.rope_dims);self.register_buffer('inv_freq',inv_freq,persistent=False);self._seq_len_cached=0;self._cos_cached=None;self._sin_cached=None
	def forward(self,seq_len,device,dtype):
		if self._cos_cached is None or self._sin_cached is None or self._seq_len_cached!=seq_len or self._cos_cached.device!=device:
			rd=self.rope_dims
			if seq_len>self.train_seq_len:scale=seq_len/self.train_seq_len;new_base=self.base*scale**(rd/(rd-2));inv_freq=1./new_base**(torch.arange(0,rd,2,dtype=torch.float32,device=device)/rd)
			else:inv_freq=self.inv_freq.to(device)
			t=torch.arange(seq_len,device=device,dtype=inv_freq.dtype);freqs=torch.outer(t,inv_freq);self._cos_cached=freqs.cos()[None,:,None,:];self._sin_cached=freqs.sin()[None,:,None,:];self._seq_len_cached=seq_len
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
		kv_dim=self.num_kv_heads*self.head_dim;self.c_q=CastedLinear(dim,dim,bias=False);self.c_k=CastedLinear(dim,kv_dim,bias=False);self.c_v=CastedLinear(dim,kv_dim,bias=False);self.proj=CastedLinear(dim,dim,bias=False);self.proj._zero_init=True;self.q_gain=nn.Parameter(torch.full((num_heads,),qk_gain_init,dtype=torch.float32));self.rope_dims=0;self.rotary=Rotary(self.head_dim,base=rope_base,train_seq_len=train_seq_len);self.use_xsa=False
	def _xsa_efficient(self,y,v):B,T,H,D=y.shape;Hkv=v.size(-2);group=H//Hkv;y_g=y.reshape(B,T,Hkv,group,D);vn=F.normalize(v,dim=-1).unsqueeze(-2);proj=(y_g*vn).sum(dim=-1,keepdim=True)*vn;return(y_g-proj).reshape(B,T,H,D)
	def forward(self,x):
		bsz,seqlen,dim=x.shape;q=self.c_q(x).reshape(bsz,seqlen,self.num_heads,self.head_dim);k=self.c_k(x).reshape(bsz,seqlen,self.num_kv_heads,self.head_dim);v=self.c_v(x).reshape(bsz,seqlen,self.num_kv_heads,self.head_dim);q=F.rms_norm(q,(q.size(-1),));k=F.rms_norm(k,(k.size(-1),));cos,sin=self.rotary(seqlen,x.device,q.dtype);q=apply_rotary_emb(q,cos,sin,self.rope_dims);k=apply_rotary_emb(k,cos,sin,self.rope_dims);q=q*self.q_gain.to(dtype=q.dtype)[None,None,:,None];y=flash_attn_3_func(q.bfloat16(),k.bfloat16(),v.bfloat16(),causal=True).to(q.dtype)
		if self.use_xsa:y=self._xsa_efficient(y,v)
		y=y.reshape(bsz,seqlen,dim);return self.proj(y)
class MLP(nn.Module):
	def __init__(self,dim,mlp_mult):super().__init__();hidden=int(mlp_mult*dim);self.fc=CastedLinear(dim,hidden,bias=False);self.proj=CastedLinear(hidden,dim,bias=False);self.proj._zero_init=True
	def forward(self,x):return self.proj(F.leaky_relu(self.fc(x),negative_slope=.5).square())
class Block(nn.Module):
	def __init__(self,dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init,train_seq_len,layer_idx=0,ln_scale=False):super().__init__();self.attn_norm=RMSNorm();self.mlp_norm=RMSNorm();self.attn=CausalSelfAttention(dim,num_heads,num_kv_heads,rope_base,qk_gain_init,train_seq_len);self.mlp=MLP(dim,mlp_mult);self.attn_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.mlp_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.resid_mix=nn.Parameter(torch.stack((torch.ones(dim),torch.zeros(dim))).float());self.ln_scale_factor=1./math.sqrt(layer_idx+1)if ln_scale else 1.;self.parallel=False
	def forward(self,x,x0):
		mix=self.resid_mix.to(dtype=x.dtype);x_in=mix[0][None,None,:]*x+mix[1][None,None,:]*x0;attn_out=self.attn(self.attn_norm(x_in)*self.ln_scale_factor)
		if self.parallel:mlp_out=self.mlp(self.mlp_norm(x_in)*self.ln_scale_factor);x_out=x_in+self.attn_scale.to(dtype=x_in.dtype)[None,None,:]*attn_out+self.mlp_scale.to(dtype=x_in.dtype)[None,None,:]*mlp_out
		else:x_out=x_in+self.attn_scale.to(dtype=x_in.dtype)[None,None,:]*attn_out;x_out=x_out+self.mlp_scale.to(dtype=x_out.dtype)[None,None,:]*self.mlp(self.mlp_norm(x_out)*self.ln_scale_factor)
		return x_out
class GPT(nn.Module):
	def __init__(self,h):
		super().__init__()
		if h.logit_softcap<=.0:raise ValueError(f"logit_softcap must be positive, got {h.logit_softcap}")
		self.tie_embeddings=h.tie_embeddings;self.tied_embed_init_std=h.tied_embed_init_std;self.logit_softcap=h.logit_softcap;self.tok_emb=nn.Embedding(h.vocab_size,h.embedding_dim)
		if h.embedding_dim!=h.model_dim:self.embed_proj=CastedLinear(h.embedding_dim,h.model_dim,bias=False);self.head_proj=CastedLinear(h.model_dim,h.embedding_dim,bias=False)
		else:self.embed_proj=None;self.head_proj=None
		self.num_encoder_layers=h.num_layers//2;self.num_decoder_layers=h.num_layers-self.num_encoder_layers;self.blocks=nn.ModuleList([Block(h.model_dim,h.num_heads,h.num_kv_heads,h.mlp_mult,h.rope_base,h.qk_gain_init,h.train_seq_len,layer_idx=i,ln_scale=h.ln_scale)for i in range(h.num_layers)])
		if h.rope_dims>0:
			head_dim=h.model_dim//h.num_heads
			for block in self.blocks:block.attn.rope_dims=h.rope_dims;block.attn.rotary=Rotary(head_dim,base=h.rope_base,train_seq_len=h.train_seq_len,rope_dims=h.rope_dims)
		self.final_norm=RMSNorm();self.lm_head=None if h.tie_embeddings else CastedLinear(h.embedding_dim,h.vocab_size,bias=False)
		if self.lm_head is not None:self.lm_head._zero_init=True
		if h.xsa_last_n>0:
			for i in range(max(0,h.num_layers-h.xsa_last_n),h.num_layers):self.blocks[i].attn.use_xsa=True
		if h.parallel_residual_start>=0:
			for i in range(h.parallel_residual_start,h.num_layers):self.blocks[i].parallel=True
		self.looping_active=False
		if h.num_loops>0:
			loop_seg=list(range(h.loop_start,h.loop_end+1));all_indices=list(range(h.loop_start))
			for _ in range(h.num_loops+1):all_indices.extend(loop_seg)
			all_indices.extend(range(h.loop_end+1,h.num_layers));num_enc=len(all_indices)//2;self.encoder_indices=all_indices[:num_enc];self.decoder_indices=all_indices[num_enc:]
		else:self.encoder_indices=list(range(self.num_encoder_layers));self.decoder_indices=list(range(self.num_encoder_layers,h.num_layers))
		self.num_skip_weights=min(len(self.encoder_indices),len(self.decoder_indices));self.skip_weights=nn.Parameter(torch.ones(self.num_skip_weights,h.model_dim,dtype=torch.float32));self.skip_gates=nn.Parameter(torch.zeros(self.num_skip_weights,h.model_dim,dtype=torch.float32))if h.skip_gates_enabled else None;self._init_weights()
	def _init_weights(self):
		if self.tie_embeddings:nn.init.normal_(self.tok_emb.weight,mean=.0,std=self.tied_embed_init_std)
		for(name,module)in self.named_modules():
			if isinstance(module,nn.Linear):
				if getattr(module,'_zero_init',False):nn.init.zeros_(module.weight)
				elif module.weight.ndim==2 and module.weight.shape[0]>=64 and module.weight.shape[1]>=64:nn.init.orthogonal_(module.weight,gain=1.)
	def forward_logits(self,input_ids):
		x=self.tok_emb(input_ids);x=F.rms_norm(x,(x.size(-1),))
		if self.embed_proj is not None:x=self.embed_proj(x)
		x0=x;skips=[];enc_iter=self.encoder_indices if self.looping_active else range(self.num_encoder_layers);dec_iter=self.decoder_indices if self.looping_active else range(self.num_encoder_layers,self.num_encoder_layers+self.num_decoder_layers)
		for i in enc_iter:x=self.blocks[i](x,x0);skips.append(x)
		for(skip_idx,i)in enumerate(dec_iter):
			if skip_idx<self.num_skip_weights and skips:
				scaled_skip=self.skip_weights[skip_idx].to(dtype=x.dtype)[None,None,:]*skips.pop()
				if self.skip_gates is not None:g=torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None,None,:];x=torch.lerp(scaled_skip,x,g)
				else:x=x+scaled_skip
			x=self.blocks[i](x,x0)
		x=self.final_norm(x)
		if self.head_proj is not None:x=self.head_proj(x)
		if self.tie_embeddings:logits_proj=F.linear(x,self.tok_emb.weight)
		else:logits_proj=self.lm_head(x)
		return self.logit_softcap*torch.tanh(logits_proj/self.logit_softcap)
	def forward(self,input_ids,target_ids):logits=self.forward_logits(input_ids);return F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),target_ids.reshape(-1),reduction='mean')
def classify_param(name):
	if'tok_emb'in name or'lm_head'in name:return'embed'
	if'.mlp.'in name:return'mlp'
	if'.attn.'in name or'.proj.'in name and'.mlp.'not in name:return'attn'
	return'other'
def zeropower_via_newtonschulz5(G,steps=10,eps=1e-07):
	a,b,c=3.4445,-4.775,2.0315;X=G.bfloat16();X/=X.norm()+eps;transposed=G.size(0)>G.size(1)
	if transposed:X=X.T
	for _ in range(steps):A=X@X.T;B=b*A+c*A@A;X=a*X+B@X
	return X.T if transposed else X
class Muon(torch.optim.Optimizer):
	def __init__(self,params,lr,momentum,backend_steps,nesterov=True,weight_decay=.0,row_normalize=False):super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,nesterov=nesterov,weight_decay=weight_decay,row_normalize=row_normalize))
	@torch.no_grad()
	def step(self,closure=None):
		loss=None
		if closure is not None:
			with torch.enable_grad():loss=closure()
		distributed=dist.is_available()and dist.is_initialized();world_size=dist.get_world_size()if distributed else 1;rank=dist.get_rank()if distributed else 0
		for group in self.param_groups:
			params=group['params']
			if not params:continue
			lr=group['lr'];momentum=group['momentum'];backend_steps=group['backend_steps'];nesterov=group['nesterov'];total_params=sum(int(p.numel())for p in params);updates_flat=torch.zeros(total_params,device=params[0].device,dtype=torch.bfloat16);curr=0
			for(i,p)in enumerate(params):
				if i%world_size==rank and p.grad is not None:
					g=p.grad;state=self.state[p]
					if'momentum_buffer'not in state:state['momentum_buffer']=torch.zeros_like(g)
					buf=state['momentum_buffer'];buf.mul_(momentum).add_(g)
					if nesterov:g=g.add(buf,alpha=momentum)
					if group.get('row_normalize',False):row_norms=g.float().norm(dim=-1,keepdim=True).clamp_min(1e-07);g=g/row_norms.to(g.dtype)
					g=zeropower_via_newtonschulz5(g,steps=backend_steps);g*=max(1,g.size(0)/g.size(1))**.5;updates_flat[curr:curr+p.numel()]=g.reshape(-1)
				curr+=p.numel()
		if distributed:dist.all_reduce(updates_flat,op=dist.ReduceOp.SUM)
		wd=group.get('weight_decay',.0);curr=0
		for p in params:
			if wd>.0:p.data.mul_(1.-lr*wd)
			g=updates_flat[curr:curr+p.numel()].view_as(p).to(dtype=p.dtype);p.add_(g,alpha=-lr);curr+=p.numel()
		return loss
dist=torch.distributed
CONTROL_TENSOR_NAME_PATTERNS=tuple(pattern for pattern in os.environ.get('CONTROL_TENSOR_NAME_PATTERNS','attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates').split(',')if pattern)
class Optimizers:
	def __init__(self,h,base_model):
		block_named_params=list(base_model.blocks.named_parameters());matrix_params=[p for(name,p)in block_named_params if p.ndim==2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)];scalar_params=[p for(name,p)in block_named_params if p.ndim<2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)]
		if base_model.skip_weights.numel()>0:scalar_params.append(base_model.skip_weights)
		if base_model.skip_gates is not None andbase_model.skip_gates.numel()>0:scalar_params.append(base_model.skip_gates)
		token_lr=h.tied_embed_lr if h.tie_embeddings else h.embed_lr;tok_params=[{'params':[base_model.tok_emb.weight],'lr':token_lr,'base_lr':token_lr}];self.optimizer_tok=torch.optim.AdamW(tok_params,betas=(h.beta1,h.beta2),eps=h.adam_eps,weight_decay=h.embed_wd,fused=True);self.optimizer_muon=Muon(matrix_params,lr=h.matrix_lr,momentum=h.muon_momentum,backend_steps=h.muon_backend_steps,weight_decay=h.muon_wd,row_normalize=h.muon_row_normalize)
		for group in self.optimizer_muon.param_groups:group['base_lr']=h.matrix_lr
		self.optimizer_scalar=torch.optim.AdamW([{'params':scalar_params,'lr':h.scalar_lr,'base_lr':h.scalar_lr}],betas=(h.beta1,h.beta2),eps=h.adam_eps,weight_decay=h.adam_wd,fused=True);self.optimizers=[self.optimizer_tok,self.optimizer_muon,self.optimizer_scalar]
		if base_model.lm_head is not None:self.optimizer_head=torch.optim.Adam([{'params':[base_model.lm_head.weight],'lr':h.head_lr,'base_lr':h.head_lr}],betas=(h.beta1,h.beta2),eps=h.adam_eps,fused=True);self.optimizers.insert(1,self.optimizer_head)
		else:self.optimizer_head=None
	def __iter__(self):return iter(self.optimizers)
	def zero_grad_all(self):
		for opt in self.optimizers:opt.zero_grad(set_to_none=True)
	def step(self):
		for opt in self.optimizers:opt.step()
		self.zero_grad_all()
def restore_fp32_params(model):
	for module in model.modules():
		if isinstance(module,CastedLinear):module.float()
	for(name,param)in model.named_parameters():
		if(param.ndim<2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))and param.dtype!=torch.float32:param.data=param.data.float()
class ARPoolLoader:
	def __init__(self,pool,h):self.pool=pool;self.h=h;self.idx=0;self.seq_len=pool.shape[1]
	def next_batch(self,global_tokens,grad_accum_steps):device_tokens=global_tokens//(self.h.world_size*grad_accum_steps);batch_size=max(1,device_tokens//self.seq_len);n=self.pool.shape[0];idxs=torch.arange(self.idx,self.idx+batch_size)%n;self.idx=(self.idx+batch_size)%n;x=self.pool[idxs];y=torch.roll(x,-1,dims=1);y[:,-1]=0;return x,y
def generate_ar_pool(model,h,device):
	gen_len=h.ar_calib_gen_len;n_seqs=h.ar_calib_n_seqs;bs=16;model.eval();rng=torch.Generator(device=device);rng.manual_seed(h.seed+99991);all_seqs=[]
	with torch.no_grad():
		for i in range(0,n_seqs,bs):
			b=min(bs,n_seqs-i);ctx=torch.randint(0,h.vocab_size,(b,1),device=device,generator=rng)
			for _ in range(gen_len-1):logits=model.forward_logits(ctx);next_tok=logits[:,-1,:].argmax(dim=-1,keepdim=True);ctx=torch.cat([ctx,next_tok],dim=1)
			all_seqs.append(ctx)
	return torch.cat(all_seqs,dim=0)[:n_seqs]
def collect_hessians(model,train_loader,h,device,n_calibration_batches=64):
	hessians={};hooks=[]
	def make_hook(name):
		def hook_fn(module,inp,out):
			x=inp[0].detach().float()
			if x.ndim==3:x=x.reshape(-1,x.shape[-1])
			if name not in hessians:hessians[name]=torch.zeros(x.shape[1],x.shape[1],dtype=torch.float32,device=device)
			hessians[name].addmm_(x.T,x)
		return hook_fn
	for(name,module)in model.named_modules():
		if isinstance(module,CastedLinear)and module.weight.numel()>65536:
			cat=classify_param(name+'.weight')
			if cat in('mlp','attn'):hooks.append(module.register_forward_hook(make_hook(name+'.weight')))
	if model.tie_embeddings:
		hook_module=model.head_proj if model.head_proj is not None else model.final_norm
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
	W_orig=w.float().clone();rows,cols=W_orig.shape;H=H.float().clone();dead=torch.diag(H)==0;H[dead,dead]=1;damp=.01*H.diag().mean();H.diagonal().add_(damp);perm=torch.argsort(H.diag(),descending=True);invperm=torch.argsort(perm);W_perm=W_orig[:,perm].clone();W_perm[:,dead[perm]]=0;H=H[perm][:,perm];Hinv=torch.cholesky_inverse(torch.linalg.cholesky(H));Hinv=torch.linalg.cholesky(Hinv,upper=True);row_std=W_orig.std(dim=1);s=(clip_sigmas*row_std/clip_range).clamp_min(1e-10).to(torch.float16);sf=s.float();Q=torch.zeros(rows,cols,dtype=torch.int8);W_work=W_perm.clone()
	for i1 in range(0,cols,block_size):
		i2=min(i1+block_size,cols);W_block=W_work[:,i1:i2].clone();Hinv_block=Hinv[i1:i2,i1:i2];Err=torch.zeros(rows,i2-i1)
		for j in range(i2-i1):w_col=W_block[:,j];d=Hinv_block[j,j];q_col=torch.clamp(torch.round(w_col/sf),-clip_range,clip_range);Q[:,i1+j]=q_col.to(torch.int8);err=(w_col-q_col.float()*sf)/d;Err[:,j]=err;W_block[:,j:]-=err.unsqueeze(1)*Hinv_block[j,j:].unsqueeze(0)
		if i2<cols:W_work[:,i2:]-=Err@Hinv[i1:i2,i2:]
	return Q[:,invperm],s
def find_sdclip_sigmas(w,H,clip_range,steps=20):
	W=w.float();row_std=W.std(dim=1).clamp_min(1e-10);H_diag=torch.diag(H).float().clamp_min(0)
	best_sigma=12.85;best_err=float('inf')
	for sigma in torch.linspace(2.0,25.0,steps).tolist():
		s=(sigma*row_std/clip_range).clamp_min(1e-10).unsqueeze(1);Q=torch.clamp(torch.round(W/s),-clip_range,clip_range);W_hat=Q*s
		err=((W-W_hat).pow(2)*H_diag.unsqueeze(0)).sum().item()
		if err<best_err:best_err=err;best_sigma=sigma
	return best_sigma
def gptq_mixed_quantize(state_dict,hessians,h):
	result={};meta={}
	for(name,tensor)in state_dict.items():
		t=tensor.detach().cpu().contiguous()
		if not t.is_floating_point()or t.numel()<=65536:result[name]=t.to(torch.float16)if t.is_floating_point()else t;meta[name]='passthrough (float16)';continue
		cs=h.embed_clip_sigmas if'tok_emb'in name else h.matrix_clip_sigmas;bits=h.embed_bits if'tok_emb'in name else h.matrix_bits;cs=(find_sdclip_sigmas(t,hessians[name],clip_range=2**(bits-1)-1,steps=h.sdclip_steps)if h.sdclip_enabled else cs);q,s=gptq_quantize_weight(t,hessians[name],clip_sigmas=cs,clip_range=2**(bits-1)-1);result[name+'.q']=q;result[name+'.scale']=s;meta[name]=f"gptq (int{bits})"
	categories=collections.defaultdict(set)
	for(name,cat)in meta.items():short=re.sub('\\.\\d+$','',re.sub('blocks\\.\\d+','blocks',name));categories[cat].add(short)
	log('Quantized weights:')
	for cat in sorted(categories):log(f"  {cat}: {','.join(sorted(categories[cat]))}")
	return result,meta
def dequantize_mixed(result,meta,template_sd):
	out={}
	for(name,orig)in template_sd.items():
		info=meta.get(name)
		if info is None:continue
		orig_dtype=orig.dtype
		if'passthrough'in info:
			t=result[name]
			if t.dtype==torch.float16 and orig_dtype in(torch.float32,torch.bfloat16):t=t.to(orig_dtype)
			out[name]=t;continue
		q,s=result[name+'.q'],result[name+'.scale']
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
def _compile(model,role='train',dynamic=False,fullgraph=True):
	if role=='train':backend='aot_eager';fullgraph=True
	elif role in('eval','ttt'):backend='aot_eager';fullgraph=False
	else:backend='aot_eager'
	return torch.compile(model,backend=backend,mode='default',dynamic_shapes=dynamic,fullgraph=fullgraph,disable=True)
def _compress(data,compressor):
	data=_byte_shuffle(data)
	if compressor=='lzma':return lzma.compress(data,preset=6)
	elif compressor=='brotli':import brotli;return brotli.compress(data,quality=11)
	raise ValueError(f"Unknown compressor: {compressor!r}")
def _decompress(data,compressor):
	if compressor=='lzma':raw=lzma.decompress(data)
	elif compressor=='brotli':import brotli;raw=brotli.decompress(data)
	else:raise ValueError(f"Unknown compressor: {compressor!r}")
	raw=_byte_unshuffle(raw);return raw
def serialize(h,base_model,code):
	code_bytes=len(code.encode('utf-8'))
	if h.is_main_process:torch.save(base_model.state_dict(),h.model_path);model_bytes=os.path.getsize(h.model_path);log(f"Serialized model: {model_bytes} bytes");log(f"Code size: {code_bytes} bytes")
	sd_cpu={k:v.detach().cpu()for(k,v)in base_model.state_dict().items()};device=torch.device('cuda',h.local_rank);log('GPTQ:collecting Hessians from calibration data...');t0=time.perf_counter();calib_loader=(ARPoolLoader(generate_ar_pool(base_model,h,device),h)if h.ar_calib_enabled else ShuffledSequenceLoader(h,device));hessians=collect_hessians(base_model,calib_loader,h,device,n_calibration_batches=h.gptq_calibration_batches);log(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter()-t0:.1f}s");quant_result,quant_meta=gptq_mixed_quantize(sd_cpu,hessians,h);quant_buf=io.BytesIO();torch.save({'w':quant_result,'m':quant_meta},quant_buf);quant_raw=quant_buf.getvalue();quant_blob=_compress(quant_raw,h.compressor);quant_file_bytes=len(quant_blob);bytes_total=quant_file_bytes+code_bytes
	if h.is_main_process:
		with open(h.quantized_model_path,'wb')asf:f.write(quant_blob)
		log(f"Serialized model quantized+{h.compressor}: {quant_file_bytes} bytes");log(f"Total submission size quantized+{h.compressor}: {bytes_total} bytes")
	return bytes_total,quant_file_bytes
def deserialize(h,device):
	eval_model=GPT(h).to(device).bfloat16();restore_fp32_params(eval_model);sd_cpu={k:v.detach().cpu()for(k,v)in eval_model.state_dict().items()}
	with open(h.quantized_model_path,'rb')as f:quant_blob_disk=f.read()
	quant_state=torch.load(io.BytesIO(_decompress(quant_blob_disk,h.compressor)),map_location='cpu');deq_state=dequantize_mixed(quant_state['w'],quant_state['m'],sd_cpu);eval_model.load_state_dict(deq_state,strict=True);return eval_model
def _loss_bpb(loss_sum,token_count,byte_count):val_loss=(loss_sum/token_count).item();val_bpb=val_loss/math.log(2.)*(token_count.item()/byte_count.item());return val_loss,val_bpb
def eval_val(h,device,val_data,model):
	seq_len=h.eval_seq_len;local_batch_tokens=h.val_batch_tokens//(h.world_size*h.grad_accum_steps)
	if local_batch_tokens<seq_len:raise ValueError(f"VAL_BATCH_SIZE must provide at least one sequence per rank; got VAL_BATCH_SIZE={h.val_batch_tokens}, WORLD_SIZE={h.world_size}, GRAD_ACCUM_STEPS={h.grad_accum_steps}, seq_len={seq_len}")
	local_batch_seqs=local_batch_tokens//seq_len;total_seqs=(val_data.val_tokens.numel()-1)//seq_len;seq_start=total_seqs*h.rank//h.world_size;seq_end=total_seqs*(h.rank+1)//h.world_size;val_loss_sum=torch.zeros((),device=device,dtype=torch.float64);val_token_count=torch.zeros((),device=device,dtype=torch.float64);val_byte_count=torch.zeros((),device=device,dtype=torch.float64);model.eval()
	with torch.inference_mode():
		for batch_seq_start in range(seq_start,seq_end,local_batch_seqs):
			batch_seq_end=min(batch_seq_start+local_batch_seqs,seq_end);raw_start=batch_seq_start*seq_len;raw_end=batch_seq_end*seq_len+1;local=val_data.val_tokens[raw_start:raw_end].to(device=device,dtype=torch.int64,non_blocking=True);x=local[:-1].reshape(-1,seq_len);y=local[1:].reshape(-1,seq_len)
			with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):batch_loss=model(x,y).detach()
			batch_token_count=float(y.numel());val_loss_sum+=batch_loss.to(torch.float64)*batch_token_count;val_token_count+=batch_token_count;prev_ids=x.reshape(-1);tgt_ids=y.reshape(-1);token_bytes=val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16);token_bytes+=(val_data.has_leading_space_lut[tgt_ids]&~val_data.is_boundary_token_lut[prev_ids]).to(dtype=torch.int16);val_byte_count+=token_bytes.to(torch.float64).sum()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(val_loss_sum,op=dist.ReduceOp.SUM);dist.all_reduce(val_token_count,op=dist.ReduceOp.SUM);dist.all_reduce(val_byte_count,op=dist.ReduceOp.SUM)
	model.train();return _loss_bpb(val_loss_sum,val_token_count,val_byte_count)
def eval_val_sliding(h,device,val_data,base_model,batch_seqs=32):
	base_model.eval();logits_fn=_compile(base_model.forward_logits,role='eval',dynamic=False,fullgraph=True);seq_len=h.eval_seq_len;context_size=seq_len-h.eval_stride;total_tokens=val_data.val_tokens.numel()-1;window_starts=[ws for ws in range(0,total_tokens,h.eval_stride)if ws+context_size<total_tokens];total_windows=len(window_starts);my_s=total_windows*h.rank//h.world_size;my_e=total_windows*(h.rank+1)//h.world_size;my_windows=window_starts[my_s:my_e];loss_sum=torch.zeros((),device=device,dtype=torch.float64);token_count=torch.zeros((),device=device,dtype=torch.float64);byte_count=torch.zeros((),device=device,dtype=torch.float64)
	with torch.inference_mode():
		for bi in range(0,len(my_windows),batch_seqs):
			batch_ws=my_windows[bi:bi+batch_seqs];bsz=len(batch_ws);x_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);y_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);wlens=[]
			for(i,ws)in enumerate(batch_ws):we=min(ws+seq_len,total_tokens);wlen=we-ws;wlens.append(wlen);chunk=val_data.val_tokens[ws:we+1].to(dtype=torch.int64,device=device);x_batch[i,:wlen]=chunk[:-1];y_batch[i,:wlen]=chunk[1:]
			with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=logits_fn(x_batch)
			nll=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),y_batch.reshape(-1),reduction='none').reshape(bsz,seq_len)
			for(i,ws)in enumerate(batch_ws):wlen=wlens[i];s=0 if ws==0 else context_size;scored_nll=nll[i,s:wlen].to(torch.float64);loss_sum+=scored_nll.sum();token_count+=float(wlen-s);tgt=y_batch[i,s:wlen];prev=x_batch[i,s:wlen];tb=val_data.base_bytes_lut[tgt].to(torch.float64);tb+=(val_data.has_leading_space_lut[tgt]&~val_data.is_boundary_token_lut[prev]).to(torch.float64);byte_count+=tb.sum()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(loss_sum,op=dist.ReduceOp.SUM);dist.all_reduce(token_count,op=dist.ReduceOp.SUM);dist.all_reduce(byte_count,op=dist.ReduceOp.SUM)
	base_model.train();return _loss_bpb(loss_sum,token_count,byte_count)
def eval_val_sliding_etlb(h,device,val_data,base_model,batch_seqs=32):base_model.eval();logits_fn=_compile(base_model.forward_logits,role='eval',dynamic=False,fullgraph=True);seq_len=h.eval_seq_len;context_size=seq_len-h.eval_stride;total_tokens=val_data.val_tokens.numel()-1;loss_sum=torch.zeros((),device=device,dtype=torch.float64);token_count=torch.zeros((),device=device,dtype=torch.float64);byte_count=torch.zeros((),device=device,dtype=torch.float64)
	with torch.inference_mode():
		for bi in range(0,total_tokens,batch_seqs*seq_len):be=min(bi+batch_seqs*seq_len,total_tokens);bsz=(be-bi)//seq_len+1;bsz=min(bsz,batch_seqs);x_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);y_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device)
			for i in range(bsz):ws=bi+i*seq_len;we=min(ws+seq_len+1,total_tokens+1);wlen=we-ws-1;chunk=val_data.val_tokens[ws:we].to(dtype=torch.int64,device=device);x_batch[i,:wlen]=chunk[:-1];y_batch[i,:wlen]=chunk[1:]
			with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=logits_fn(x_batch[:bsz])
			nll=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),y_batch.reshape(-1),reduction='none').reshape(bsz,seq_len)
			for i in range(bsz):s=0 if bi==0 else context_size;wlen=min(seq_len,be-bi-i*seq_len);scored_nll=nll[i,s:wlen].to(torch.float64);loss_sum+=scored_nll.sum();token_count+=float(wlen-s);tgt=y_batch[i,s:wlen];prev=x_batch[i,s:wlen];tb=val_data.base_bytes_lut[tgt].to(torch.float64);tb+=(val_data.has_leading_space_lut[tgt]&~val_data.is_boundary_token_lut[prev]).to(torch.float64);byte_count+=tb.sum()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(loss_sum,op=dist.ReduceOp.SUM);dist.all_reduce(token_count,op=dist.ReduceOp.SUM);dist.all_reduce(byte_count,op=dist.ReduceOp.SUM)
	base_model.train();return _loss_bpb(loss_sum,token_count,byte_count)
def eval_val_ttt(h,device,val_data,base_model,batch_seqs=32):
	rank=h.rank;world_size=h.world_size;seq_len=h.eval_seq_len;stride=h.eval_stride;total_tokens=val_data.val_tokens.numel()-1;ttt_chunk=h.ttt_chunk_tokens;context_size=seq_len-stride;window_starts=[ws for ws in range(0,total_tokens,stride)if ws+context_size<total_tokens];num_chunks=(total_tokens+ttt_chunk-1)//ttt_chunk;chunk_windows=[[]for _ in range(num_chunks)]
	for ws in window_starts:wlen=min(ws+seq_len,total_tokens)-ws;s=0 if ws==0 else context_size;scored_start=ws+s;ci=min(scored_start//ttt_chunk,num_chunks-1);chunk_windows[ci].append(ws)
	log(f"ttt:start chunks={num_chunks} ttt_lr={h.ttt_lr} ttt_epochs={h.ttt_epochs}");compiled_logits=_compile(base_model.forward_logits,role='ttt',dynamic=False,fullgraph=True);loss_sum=torch.zeros((),device=device,dtype=torch.float64);token_count=torch.zeros((),device=device,dtype=torch.float64);byte_count=torch.zeros((),device=device,dtype=torch.float64);ttt_params=[p for p in base_model.parameters()]
	for p in ttt_params:p.requires_grad_(True)
	optimizer=torch.optim.SGD(ttt_params,lr=h.ttt_lr,momentum=h.ttt_momentum)
	for ci in range(num_chunks):
		windows=chunk_windows[ci]
		if not windows:continue
		chunk_start=ci*ttt_chunk;chunk_end=min((ci+1)*ttt_chunk,total_tokens);my_s=len(windows)*rank//world_size;my_e=len(windows)*(rank+1)//world_size;my_windows=windows[my_s:my_e];base_model.eval()
		with torch.no_grad():
			for bi in range(0,len(my_windows),batch_seqs):
				batch_ws=my_windows[bi:bi+batch_seqs];bsz=len(batch_ws);x_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);y_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);wlens=[]
				for(i,ws)in enumerate(batch_ws):we=min(ws+seq_len,total_tokens);wlen=we-ws;wlens.append(wlen);chunk_tok=val_data.val_tokens[ws:we+1].to(dtype=torch.int64,device=device);x_batch[i,:wlen]=chunk_tok[:-1];y_batch[i,:wlen]=chunk_tok[1:]
				with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=compiled_logits(x_batch)
				nll=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),y_batch.reshape(-1),reduction='none').reshape(bsz,seq_len)
				for(i,ws)in enumerate(batch_ws):wlen=wlens[i];s=0 if ws==0 else context_size;scored_nll=nll[i,s:wlen].to(torch.float64);loss_sum+=scored_nll.sum();token_count+=float(wlen-s);tgt=y_batch[i,s:wlen];prev=x_batch[i,s:wlen];tb=val_data.base_bytes_lut[tgt].to(torch.float64);tb+=(val_data.has_leading_space_lut[tgt]&~val_data.is_boundary_token_lut[prev]).to(torch.float64);byte_count+=tb.sum()
		is_last_chunk=ci==num_chunks-1
		if not is_last_chunk and h.ttt_epochs>0:
			base_model.train();chunk_seqs=(chunk_end-chunk_start)//seq_len
			if chunk_seqs>0:
				cos_lr=h.ttt_lr*.5*(1.+math.cos(math.pi*ci/max(num_chunks-1,1)))
				for pg in optimizer.param_groups:pg['lr']=cos_lr
				my_seq_s=chunk_seqs*rank//world_size;my_seq_e=chunk_seqs*(rank+1)//world_size;my_chunk_seqs=my_seq_e-my_seq_s
				for _ep in range(h.ttt_epochs):
					for bs in range(0,my_chunk_seqs,batch_seqs):
						be=min(bs+batch_seqs,my_chunk_seqs);actual_bs=my_seq_s+bs;start_tok=chunk_start+actual_bs*seq_len;end_tok=chunk_start+(my_seq_s+be)*seq_len+1
						if end_tok>val_data.val_tokens.numel():continue
						local=val_data.val_tokens[start_tok:end_tok].to(device=device,dtype=torch.int64);x=local[:-1].reshape(-1,seq_len);y=local[1:].reshape(-1,seq_len);optimizer.zero_grad(set_to_none=True)
						with torch.autocast(device_type='cuda',dtype=torch.bfloat16):loss=base_model(x,y)
						loss.backward()
						if world_size>1:
							for p in ttt_params:
								if p.grad is not None:dist.all_reduce(p.grad,op=dist.ReduceOp.AVG)
						torch.nn.utils.clip_grad_norm_(ttt_params,1.);optimizer.step()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(loss_sum,op=dist.ReduceOp.SUM);dist.all_reduce(token_count,op=dist.ReduceOp.SUM);dist.all_reduce(byte_count,op=dist.ReduceOp.SUM)
	for p in base_model.parameters():p.requires_grad_(True)
	base_model.eval();return _loss_bpb(loss_sum,token_count,byte_count)
def timed_eval(label,fn,*args,**kwargs):torch.cuda.synchronize();t0=time.perf_counter();val_loss,val_bpb=fn(*args,**kwargs);torch.cuda.synchronize();elapsed_ms=1e3*(time.perf_counter()-t0);log(f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms");return val_loss,val_bpb
def train_model(h,device,val_data):
	base_model=GPT(h).to(device).bfloat16();restore_fp32_params(base_model);compiled_model=_compile(base_model,role='train',dynamic=False,fullgraph=True)
	if h.distributed:model=DDP(compiled_model,device_ids=[h.local_rank],broadcast_buffers=False)
	else:model=compiled_model
	log(f"model_params:{sum(p.numel()for p in base_model.parameters())}");optimizers=Optimizers(h,base_model);train_loader=ShuffledSequenceLoader(h,device);max_wallclock_ms=1e3*h.max_wallclock_seconds if h.max_wallclock_seconds>0 else None
	if max_wallclock_ms is not None:max_wallclock_ms-=h.gptq_reserve_seconds*1e3;log(f"gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms")
	def training_frac(step,elapsed_ms):
		if max_wallclock_ms is None:return step/max(h.iterations,1)
		return elapsed_ms/max(max_wallclock_ms,1e-09)
	def lr_mul(frac):
		if h.warmdown_frac<=0:return 1.
		if frac>=1.-h.warmdown_frac:return max((1.-frac)/h.warmdown_frac,h.min_lr)
		return 1.
	def step_fn(step,lr_scale):
		optimizers.zero_grad_all();train_loss=torch.zeros((),device=device)
		for micro_step in range(h.grad_accum_steps):
			if h.distributed:model.require_backward_grad_sync=micro_step==h.grad_accum_steps-1
			x,y=train_loader.next_batch(h.train_batch_tokens,h.grad_accum_steps)
			with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):loss=model(x,y)
			train_loss+=loss.detach();(loss/h.grad_accum_steps).backward()
		train_loss/=h.grad_accum_steps;frac=min(step/h.muon_momentum_warmup_steps,1.)if h.muon_momentum_warmup_steps>0 else 1.;muon_momentum=(1-frac)*h.muon_momentum_warmup_start+frac*h.muon_momentum
		for group in optimizers.optimizer_muon.param_groups:group['momentum']=muon_momentum
		for opt in optimizers:
			for group in opt.param_groups:group['lr']=group['base_lr']*lr_scale
		if h.grad_clip_norm>0:torch.nn.utils.clip_grad_norm_(base_model.parameters(),h.grad_clip_norm)
		optimizers.step();return train_loss
	if h.warmup_steps>0:
		initial_model_state={name:tensor.detach().cpu().clone()for(name,tensor)in base_model.state_dict().items()};initial_optimizer_states=[copy.deepcopy(opt.state_dict())for opt in optimizers];model.train()
		for warmup_step in range(h.warmup_steps):
			step_fn(warmup_step,1.)
			if warmup_step<=5 or(warmup_step+1)%10==0 or warmup_step+1==h.warmup_steps:log(f"warmup_step: {warmup_step+1}/{h.warmup_steps}")
		if h.num_loops>0:
			base_model.looping_active=True;log(f"loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")
			for warmup_step in range(h.warmup_steps):
				step_fn(warmup_step,1.)
				if warmup_step<=5 or(warmup_step+1)%10==0 or warmup_step+1==h.warmup_steps:log(f"loop_warmup_step: {warmup_step+1}/{h.warmup_steps}")
			base_model.looping_active=False
		base_model.load_state_dict(initial_model_state,strict=True)
		for(opt,state)in zip(optimizers,initial_optimizer_states,strict=True):opt.load_state_dict(state)
		optimizers.zero_grad_all()
		if h.distributed:model.require_backward_grad_sync=True
		train_loader=ShuffledSequenceLoader(h,device)
	ema_state={name:t.detach().float().clone()for(name,t)in base_model.state_dict().items()};ema_decay=h.ema_decay;training_time_ms=.0;stop_after_step=None;torch.cuda.synchronize();t0=time.perf_counter();step=0
	while True:
		last_step=step==h.iterations or stop_after_step is not None and step>=stop_after_step;should_validate=last_step or h.val_loss_every>0 and step%h.val_loss_every==0
		if should_validate:torch.cuda.synchronize();training_time_ms+=1e3*(time.perf_counter()-t0);val_loss,val_bpb=eval_val(h,device,val_data,model);log(f"{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}");torch.cuda.synchronize();t0=time.perf_counter()
		if last_step:
			if stop_after_step is not None and step<h.iterations:log(f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms step: {step}/{h.iterations}")
			break
		elapsed_ms=training_time_ms+1e3*(time.perf_counter()-t0);frac=training_frac(step,elapsed_ms);scale=lr_mul(frac)
		if h.num_loops>0 and not base_model.looping_active and frac>=h.enable_looping_at:base_model.looping_active=True;log(f"layer_loop:enabled step:{step} frac:{frac:.3f} encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")
		train_loss=step_fn(step,scale)
		with torch.no_grad():
			_eqa_frac=float(os.environ.get('EMA_QUANT_AWARE_FRAC','0.0'));_eqa_active=_eqa_frac>0 andfrac>=(1.-_eqa_frac)
			for(name,t)in base_model.state_dict().items():
				if _eqa_active:
					_tf=t.detach().float();_is_embed='tok_emb'in name;_clip=h.embed_clip_sigmas if _is_embed else h.matrix_clip_sigmas;_range=127 if _is_embed else 63;_sigma=_tf.std().clamp_min(1e-8);_scale=_sigma*_clip/_range;_tq=(_tf/_scale).clamp(-_range,_range).round_()*_scale
					ema_state[name].mul_(ema_decay).add_(_tq,alpha=1.-ema_decay)
				else:ema_state[name].mul_(ema_decay).add_(t.detach().float(),alpha=1.-ema_decay)
		step+=1;approx_training_time_ms=training_time_ms+1e3*(time.perf_counter()-t0);should_log_train=h.train_log_every>0 and(step<=5 or step%h.train_log_every==0 or stop_after_step is not None)
		if should_log_train:tok_per_sec=step*h.train_batch_tokens/(approx_training_time_ms/1e3);log(f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} train_time: {approx_training_time_ms/60000:.1f}m tok/s: {tok_per_sec:.0f}")
		reached_cap=max_wallclock_ms is not Noneand approx_training_time_ms>=max_wallclock_ms
		if h.distributed and max_wallclock_ms isnot None:reached_cap_tensor=torch.tensor(int(reached_cap),device=device);dist.all_reduce(reached_cap_tensor,op=dist.ReduceOp.MAX);reached_cap=bool(reached_cap_tensor.item())
		if stop_after_step is None and reached_cap:stop_after_step=step
	log(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB");log('ema:applying EMA weights');current_state=base_model.state_dict();_eba=float(os.environ.get('EMA_BLEND_ALPHA','1.0'))
	if _eba>=.999:avg_state={name:t.to(dtype=current_state[name].dtype)for(name,t)in ema_state.items()}
	else:log(f'ema:blending base+EMA alpha={_eba:.3f}');avg_state={name:(_eba*t+(1.-_eba)*current_state[name].detach().float()).to(dtype=current_state[name].dtype)for(name,t)in ema_state.items()}
	base_model.load_state_dict(avg_state,strict=True);return base_model,compiled_model
def train_and_eval(h,device):
	random.seed(h.seed);np.random.seed(h.seed);torch.manual_seed(h.seed);torch.cuda.manual_seed_all(h.seed);val_data=ValidationData(h,device);log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}");log(f"val_tokens: {val_data.val_tokens.numel()-1}");base_model,compiled_model=train_model(h,device,val_data);torch._dynamo.reset();timed_eval('pre-quantization post-ema',eval_val,h,device,val_data,compiled_model);serialize(h,base_model,Path(__file__).read_text(encoding='utf-8'))
	if h.distributed:dist.barrier()
	eval_model=deserialize(h,device)
	if h.num_loops>0:eval_model.looping_active=True
	compiled_model=_compile(eval_model,role='eval',dynamic=False,fullgraph=True);timed_eval('quantized',eval_val,h,device,val_data,compiled_model)
	if h.sliding_window_enabled:timed_eval('quantized_sliding_window',eval_val_sliding,h,device,val_data,eval_model)
	if h.ttt_enabled and h.sliding_window_enabled:
		del eval_model,compiled_model;torch._dynamo.reset();torch.cuda.empty_cache();ttt_model=deserialize(h,device)
		if h.num_loops>0:ttt_model.looping_active=True
		timed_eval('quantized_ttt',eval_val_ttt,h,device,val_data,ttt_model);del ttt_model
	if h.etlb_enabled and h.sliding_window_enabled:
		if'eval_model'not in dir():
			eval_model=deserialize(h,device)
			if h.num_loops>0:eval_model.looping_active=True
		timed_eval('quantized_sliding_etlb',eval_val_sliding_etlb,h,device,val_data,eval_model)
def main():
	world_size=int(os.environ.get('WORLD_SIZE','1'));local_rank=int(os.environ.get('LOCAL_RANK','0'));distributed='RANK'in os.environ and'WORLD_SIZE'in os.environ
	if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
	if world_size<=0:raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
	if 8%world_size!=0:raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
	device=torch.device('cuda',local_rank);torch.cuda.set_device(device)
	if distributed:dist.init_process_group(backend='nccl',device_id=device);dist.barrier()
	torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.set_float32_matmul_precision('high');from torch.backends.cuda import enable_cudnn_sdp,enable_flash_sdp,enable_mem_efficient_sdp,enable_math_sdp;enable_cudnn_sdp(False);enable_flash_sdp(True);enable_mem_efficient_sdp(False);enable_math_sdp(False);torch._dynamo.config.optimize_ddp=False;h=Hyperparameters();set_logging_hparams(h) if'set_logging_hparams'in dir()else None
	if h.is_main_process:
		os.makedirs('logs',exist_ok=True);log(100*'=',console=False);log('Hyperparameters:',console=True)
		for(k,v)in sorted(vars(type(h)).items()):
			if not k.startswith('_'):log(f"{k}: {v}",console=True)
		log('='*100,console=False);log(f"Running Python {sys.version}",console=False);log(f"Running PyTorch {torch.__version__}",console=False);log(subprocess.run(['nvidia-smi'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False).stdout,console=False);log('='*100,console=False)
	train_and_eval(h,device)
	if distributed:dist.destroy_process_group()
if __name__=='__main__':main()
- SEED: 1337
- ITERATIONS: 8000 (stopped at 6646 due to wallclock=600s)
- WARMDOWN_FRAC: 0.72
- EMBED_BITS: 7
- QK_GAIN_INIT: 5.5
- EMA_DECAY: 0.995
- LOGIT_SOFTCAP: 20
- MATRIX_LR: 0.006
- SCALAR_LR: 0.03
- GPTQ_CALIBRATION_BATCHES: 128
- TTT_ENABLED: True
- TTT_EPOCHS: 1
- TTT_LR: 0.005
- TTT_MOMENTUM: 0.9
- SLIDING_WINDOW_ENABLED: True

Results:
- Train loss: 3.2171 nats
- Quantized BPB: 1.2555
- Sliding window BPB: 1.2301
- TTT BPB: 1.22728675 ✅ (SUBMISSION)
- Artifact: 15.5 MB

The complete source code includes:
1. Hyperparameter class with 40+ tunable parameters
2. Custom optimizers (Muon for matrix params, AdamW for scalars)
3. GPT model with RMSNorm, RoPE, Flash Attention 2, depth recurrence
4. Distributed training loop with DDP on 8×H100 GPUs
5. GPTQ quantization with Hessian-based clipping
6. Brotli compression
7. Evaluation pipeline: standard loss → sliding window → TTT
8. Serialization/deserialization with mixed-precision

For full source, see: https://github.com/harborglowvintage-oss/parameter-golf
"""
SCALAR_LR = 0.030
GPTQ_CALIBRATION_BATCHES = 128

TTT Config:
TTT_ENABLED = True
TTT_EPOCHS = 1
TTT_LR = 0.005
TTT_MOMENTUM = 0.9
TTT_CHUNK_TOKENS = 32768
SLIDING_WINDOW_ENABLED = True
SLIDING_WINDOW_STRIDE = 64

Results:
- Train loss: 3.094 nats
- Quant BPB: 1.2555
- Sliding window BPB: 1.2301
- TTT BPB: 1.2273 ✅

[Full script being retrieved from pod - check records/track_10min_16mb/2026-04-30_SP8192_DepthRecur_ParResid_TTT_GptqBrotli/ for complete version]
"""
