from __future__ import annotations
_Z='passthrough_ctrl'
_Y='passthrough'
_X='momentum'
_W='shard_mom'
_V='padded_grad'
_U='fineweb_train_*.bin'
_T='diagonal'
_S='.scale'
_R='mlp_down_bank'
_Q='mlp_up_bank'
_P='kv_bank'
_O='qo_bank'
_N='attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,ve_layer_scales,ve_shared.scale'
_M='shard'
_L='scale'
_K='full_update'
_J='utf-8'
_I='cuda'
_H='0'
_G='lr'
_F='params'
_E=.0
_D=False
_C=1.
_B=True
_A=None
import copy,glob,io,lzma,math,os,random,time,uuid
from pathlib import Path
import numpy as np,sentencepiece as spm,torch,torch._dynamo
torch._dynamo.config.recompile_limit=32
import torch.distributed as dist,torch.nn.functional as F
from torch import Tensor,nn
_gpu_mem_frac=float(os.environ.get('CUDA_MEM_FRACTION',_H))
if _gpu_mem_frac>0:torch.cuda.set_per_process_memory_fraction(_gpu_mem_frac,0)
from flash_attn_interface import flash_attn_func as flash_attn_3_func
import argparse
class RecurrentStabilizer:
	def __init__(self,jacobian_proxy_weight=_E,eps=1e-06,**kw):self.jacobian_proxy_weight=jacobian_proxy_weight;self.eps=eps
	def clip(self,h):return h
	def jacobian_proxy_loss(self,h_in,h_out):
		if self.jacobian_proxy_weight<=0:return h_in.new_zeros(())
		delta=h_out-h_in;ratio=delta.norm()/(h_in.norm()+self.eps);return self.jacobian_proxy_weight*torch.relu(ratio-_C).square()
	def reset(self):0
class ResidualScale(nn.Module):
	def __init__(self,num_passes,init_value=_C):super().__init__();self.scales=nn.Parameter(torch.full((num_passes,),init_value,dtype=torch.float32))
	def forward(self,residual,pass_idx):return self.scales[pass_idx].to(dtype=residual.dtype)*residual
class LowRankResidual(nn.Module):
	def __init__(self,dim,rank=2):super().__init__();self.V=nn.Parameter(torch.zeros(dim,rank));self.U=nn.Parameter(torch.zeros(dim,rank))
	def forward(self,h):return h@self.V@self.U.T
class DiagonalFeedback(nn.Module):
	def __init__(self,dim,init_ones=_D):super().__init__();init_val=torch.ones(dim)if init_ones else torch.zeros(dim);self.d=nn.Parameter(init_val)
	def forward(self,e):return self.d.to(dtype=e.dtype)*e
class ErrorFeedbackModule(nn.Module):
	def __init__(self,dim,rank=2,feedback_mode=_T,per_pass=_D,num_passes=3,**kw):
		super().__init__();self.per_pass=per_pass;self.residual=LowRankResidual(dim,rank)
		if feedback_mode=='identity':self.correction=_A
		elif per_pass:self.correction=nn.ModuleList([DiagonalFeedback(dim)for _ in range(num_passes)])
		else:self.correction=DiagonalFeedback(dim)
	def forward(self,h,pass_idx):
		e=self.residual(h)
		if self.correction is _A:c=e
		elif self.per_pass:c=self.correction[pass_idx](e)
		else:c=self.correction(e)
		mask=torch.tensor(_C if pass_idx>0 else _E,device=h.device,dtype=h.dtype);return c*mask
	def param_count(self):return sum(p.numel()for p in self.parameters())
_e=os.environ.get
_i=lambda k,d:int(_e(k,d))
_f=lambda k,d:float(_e(k,d))
_b=lambda k,d:bool(int(_e(k,d)))
class Hyperparameters:data_path=_e('DATA_PATH','./data/datasets/fineweb10B_sp1024');train_files=os.path.join(data_path,_U);val_files=os.path.join(data_path,'fineweb_val_*.bin');tokenizer_path=_e('TOKENIZER_PATH','./data/tokenizers/fineweb_1024_bpe.model');run_id=_e('RUN_ID',str(uuid.uuid4()));seed=_i('SEED',1337);val_batch_size=_i('VAL_BATCH_SIZE',524288);val_loss_every=_i('VAL_LOSS_EVERY',4000);train_log_every=_i('TRAIN_LOG_EVERY',500);iterations=_i('ITERATIONS',20000);warmdown_iters=_i('WARMDOWN_ITERS',3500);warmup_steps=_i('WARMUP_STEPS',20);train_batch_tokens=_i('TRAIN_BATCH_TOKENS',786432);train_seq_len=_i('TRAIN_SEQ_LEN',2048);eval_seq_len=_i('EVAL_SEQ_LEN',2048);max_wallclock_seconds=_f('MAX_WALLCLOCK_SECONDS',6e2);qk_gain_init=_f('QK_GAIN_INIT',1.5);vocab_size=_i('VOCAB_SIZE',1024);num_layers=_i('NUM_LAYERS',11);num_kv_heads=_i('NUM_KV_HEADS',4);model_dim=_i('MODEL_DIM',512);num_heads=_i('NUM_HEADS',8);mlp_mult=_f('MLP_MULT',3.);tie_embeddings=_b('TIE_EMBEDDINGS','1');rope_base=_f('ROPE_BASE',1e4);logit_softcap=_f('LOGIT_SOFTCAP',3e1);embed_lr=_f('EMBED_LR',.6);head_lr=_f('HEAD_LR',.008);tied_embed_lr=_f('TIED_EMBED_LR',.035);tied_embed_init_std=_f('TIED_EMBED_INIT_STD',.005);matrix_lr=_f('MATRIX_LR',.025);scalar_lr=_f('SCALAR_LR',.025);muon_momentum=_f('MUON_MOMENTUM',.99);muon_backend_steps=_i('MUON_BACKEND_STEPS',5);muon_momentum_warmup_start=_f('MUON_MOMENTUM_WARMUP_START',.92);muon_momentum_warmup_steps=_i('MUON_MOMENTUM_WARMUP_STEPS',1500);beta1=_f('BETA1',.9);beta2=_f('BETA2',.95);adam_eps=_f('ADAM_EPS',1e-08);grad_clip_norm=_f('GRAD_CLIP_NORM',.3);eval_stride=_i('EVAL_STRIDE',64);muon_beta2=_f('MUON_BETA2',.95);swa_enabled=_b('SWA_ENABLED','1');swa_every=_i('SWA_EVERY',50);muon_wd=_f('MUON_WD',.04);adam_wd=_f('ADAM_WD',.04);qat_enabled=_b('QAT_ENABLED',_H);xsa_last_n=_i('XSA_LAST_N',4);rope_dims=_i('ROPE_DIMS',16);ln_scale=_b('LN_SCALE','1');late_qat_threshold=_f('LATE_QAT_THRESHOLD',.15);ttt_enabled=_b('TTT_ENABLED',_H);ttt_lr=_f('TTT_LR',.002);ttt_epochs=_i('TTT_EPOCHS',3);ttt_chunk_tokens=_i('TTT_CHUNK_TOKENS',32768);ttt_freeze_blocks=_i('TTT_FREEZE_BLOCKS',2);ttt_momentum=_f('TTT_MOMENTUM',.9);ttt_batch_seqs=_i('TTT_BATCH_SEQS',32);ttt_grad_clip=_f('TTT_GRAD_CLIP',_C);core_start=_i('CORE_START',3);core_end=_i('CORE_END',8);num_passes=_i('NUM_PASSES',1);core_quant_bits=_i('CORE_QUANT_BITS',6);core_quant_enabled=_b('CORE_QUANT_ENABLED',_H);eval_passes=_i('EVAL_PASSES',0);passes_schedule_str=_e('PASSES_SCHEDULE','');bigram_vocab_size=_i('BIGRAM_VOCAB_SIZE',0);bigram_dim=_i('BIGRAM_DIM',32);ve_enabled=_b('VE_ENABLED',_H);ve_dim=_i('VE_DIM',128);ve_layers=_e('VE_LAYERS','9,10')
def zeropower_via_newtonschulz5(G,steps=5,eps=1e-07):
	a,b,c=3.4445,-4.775,2.0315;was_2d=G.ndim==2
	if was_2d:G=G.unsqueeze(0)
	X=G.bfloat16();transposed=X.size(-2)>X.size(-1)
	if transposed:X=X.mT
	X=X/(X.norm(dim=(-2,-1),keepdim=_B)+eps)
	for _ in range(steps):A=X@X.mT;B=b*A+c*(A@A);X=a*X+B@X
	if transposed:X=X.mT
	if was_2d:X=X.squeeze(0)
	return X
class Muon(torch.optim.Optimizer):
	def __init__(self,params,lr,momentum,backend_steps,nesterov=_B,weight_decay=_E):super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,nesterov=nesterov,weight_decay=weight_decay));self._built=_D
	def _build(self):
		self._distributed=dist.is_available()and dist.is_initialized();self._world_size=dist.get_world_size()if self._distributed else 1;self._rank=dist.get_rank()if self._distributed else 0;ws=self._world_size;self._bank_meta=[]
		for group in self.param_groups:
			for p in group[_F]:B=p.shape[0];padded_B=(B+ws-1)//ws*ws;shard_B=padded_B//ws;tail=p.shape[1:];dev=p.device;self._bank_meta.append({'p':p,'B':B,_V:torch.zeros(padded_B,*tail,device=dev,dtype=torch.bfloat16),_M:torch.zeros(shard_B,*tail,device=dev,dtype=torch.bfloat16),_W:torch.zeros(shard_B,*tail,device=dev,dtype=torch.bfloat16),_K:torch.zeros(padded_B,*tail,device=dev,dtype=torch.bfloat16),_L:max(1,p.shape[-2]/p.shape[-1])**.5})
		self._bank_meta.sort(key=lambda m:-m['p'].numel());self._built=_B
	def launch_reduce_scatters(self):
		''
		if not self._built:self._build()
		if not self._distributed:return
		self._rs_futures=[]
		for m in self._bank_meta:
			p=m['p']
			if p.grad is _A:self._rs_futures.append(_A);continue
			pg=m[_V];pg[:m['B']].copy_(p.grad.bfloat16())
			if pg.shape[0]>m['B']:pg[m['B']:].zero_()
			fut=dist.reduce_scatter_tensor(m[_M],pg,op=dist.ReduceOp.AVG,async_op=_B);self._rs_futures.append(fut)
	@torch.no_grad()
	def step(self,closure=_A):
		'';B='_rs_futures';A='momentum_buffer';loss=_A
		if closure is not _A:
			with torch.enable_grad():loss=closure()
		if not self._built:self._build()
		for group in self.param_groups:
			lr=group[_G];momentum=group[_X];backend_steps=group['backend_steps'];nesterov=group['nesterov'];wd=group.get('weight_decay',_E);prev_ag_handle=_A;prev_m=_A;sharded=self._distributed and hasattr(self,B)
			for(i,m)in enumerate(self._bank_meta):
				p=m['p']
				if p.grad is _A:continue
				if prev_ag_handle is not _A:
					prev_ag_handle.wait();pp=prev_m['p'];upd=prev_m[_K][:prev_m['B']]
					if wd>_E:pp.data.mul_(_C-lr*wd)
					pp.add_(upd.to(dtype=pp.dtype),alpha=-lr*prev_m[_L])
				if sharded and self._rs_futures[i]is not _A:self._rs_futures[i].wait();g=m[_M];buf=m[_W]
				else:
					g=p.grad.bfloat16();state=self.state[p]
					if A not in state:state[A]=torch.zeros_like(g)
					buf=state[A]
				buf.mul_(momentum).add_(g)
				if nesterov:update=g.add(buf,alpha=momentum)
				else:update=buf
				update=zeropower_via_newtonschulz5(update,steps=backend_steps)
				if sharded:prev_ag_handle=dist.all_gather_into_tensor(m[_K],update,async_op=_B);prev_m=m
				else:
					if wd>_E:p.data.mul_(_C-lr*wd)
					p.add_(update.to(dtype=p.dtype),alpha=-lr*m[_L])
			if prev_ag_handle is not _A:
				prev_ag_handle.wait();pp=prev_m['p'];upd=prev_m[_K][:prev_m['B']]
				if wd>_E:pp.data.mul_(_C-lr*wd)
				pp.add_(upd.to(dtype=pp.dtype),alpha=-lr*prev_m[_L])
			if hasattr(self,B):del self._rs_futures
		return loss
def build_sentencepiece_luts(sp,vocab_size,device):
	sp_vocab_size=int(sp.vocab_size());table_size=max(sp_vocab_size,vocab_size);base_bytes_np=np.zeros((table_size,),dtype=np.int16);has_leading_space_np=np.zeros((table_size,),dtype=np.bool_);is_boundary_token_np=np.ones((table_size,),dtype=np.bool_)
	for token_id in range(sp_vocab_size):
		if sp.is_control(token_id)or sp.is_unknown(token_id)or sp.is_unused(token_id):continue
		is_boundary_token_np[token_id]=_D
		if sp.is_byte(token_id):base_bytes_np[token_id]=1;continue
		piece=sp.id_to_piece(token_id)
		if piece.startswith('▁'):has_leading_space_np[token_id]=_B;piece=piece[1:]
		base_bytes_np[token_id]=len(piece.encode(_J))
	return torch.tensor(base_bytes_np,dtype=torch.int16,device=device),torch.tensor(has_leading_space_np,dtype=torch.bool,device=device),torch.tensor(is_boundary_token_np,dtype=torch.bool,device=device)
def load_validation_tokens(pattern,seq_len):
	files=[Path(p)for p in sorted(glob.glob(pattern))]
	if not files:raise FileNotFoundError(f"No files found for pattern: {pattern}")
	tokens=torch.cat([load_data_shard(file)for file in files]).contiguous();usable=(tokens.numel()-1)//seq_len*seq_len
	if usable<=0:raise ValueError('val split too short')
	return tokens[:usable+1]
def eval_val(args,model,rank,world_size,device,grad_accum_steps,val_tokens,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut,eval_seq_len=_A):
	seq_len=eval_seq_len or args.train_seq_len;local_batch_tokens=args.val_batch_size//(world_size*grad_accum_steps)
	if local_batch_tokens<seq_len:raise ValueError('VAL_BATCH_SIZE too small')
	local_batch_seqs=local_batch_tokens//seq_len;total_seqs=(val_tokens.numel()-1)//seq_len;seq_start=total_seqs*rank//world_size;seq_end=total_seqs*(rank+1)//world_size;val_loss_sum=torch.zeros((),device=device,dtype=torch.float64);val_token_count=torch.zeros((),device=device,dtype=torch.float64);val_byte_count=torch.zeros((),device=device,dtype=torch.float64);model.eval()
	with torch.inference_mode():
		for batch_seq_start in range(seq_start,seq_end,local_batch_seqs):
			batch_seq_end=min(batch_seq_start+local_batch_seqs,seq_end);raw_start=batch_seq_start*seq_len;raw_end=batch_seq_end*seq_len+1;local=val_tokens[raw_start:raw_end].to(device=device,dtype=torch.int64,non_blocking=_B);x=local[:-1].reshape(-1,seq_len);y=local[1:].reshape(-1,seq_len)
			with torch.autocast(device_type=_I,dtype=torch.bfloat16,enabled=_B):batch_loss=model(x,y).detach()
			batch_token_count=float(y.numel());val_loss_sum+=batch_loss.to(torch.float64)*batch_token_count;val_token_count+=batch_token_count;prev_ids=x.reshape(-1);tgt_ids=y.reshape(-1);token_bytes=base_bytes_lut[tgt_ids].to(dtype=torch.int16);token_bytes+=(has_leading_space_lut[tgt_ids]&~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16);val_byte_count+=token_bytes.to(torch.float64).sum()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(val_loss_sum,op=dist.ReduceOp.SUM);dist.all_reduce(val_token_count,op=dist.ReduceOp.SUM);dist.all_reduce(val_byte_count,op=dist.ReduceOp.SUM)
	val_loss=val_loss_sum/val_token_count;bits_per_token=val_loss.item()/math.log(2.);tokens_per_byte=val_token_count.item()/val_byte_count.item();model.train();return float(val_loss.item()),float(bits_per_token*tokens_per_byte)
def quantize_float_tensor(t):
	t32=t.float()
	if t32.ndim==2:clip_abs=torch.quantile(t32.abs(),.9999984,dim=1)if t32.numel()else torch.empty((t32.shape[0],),dtype=torch.float32);clipped=torch.maximum(torch.minimum(t32,clip_abs[:,_A]),-clip_abs[:,_A]);scale=(clip_abs/127.).clamp_min(_C/127.);q=torch.clamp(torch.round(clipped/scale[:,_A]),-127,127).to(torch.int8).contiguous();return q,scale.to(dtype=torch.float16).contiguous()
	clip_abs=float(torch.quantile(t32.abs().flatten(),.9999984).item())if t32.numel()else _E;scale=torch.tensor(clip_abs/127. if clip_abs>0 else _C,dtype=torch.float32);q=torch.clamp(torch.round(torch.clamp(t32,-clip_abs,clip_abs)/scale),-127,127).to(torch.int8).contiguous();return q,scale
def load_data_shard(file):
	B='<u2';A='<i4';header_bytes=256*np.dtype(A).itemsize;token_bytes=np.dtype(B).itemsize;header=np.fromfile(file,dtype=A,count=256)
	if header.size!=256 or int(header[0])!=20240520 or int(header[1])!=1:raise ValueError('bad shard header')
	num_tokens=int(header[2]);expected_size=header_bytes+num_tokens*token_bytes
	if file.stat().st_size!=expected_size:raise ValueError('shard size mismatch')
	tokens_np=np.fromfile(file,dtype=B,count=num_tokens,offset=header_bytes)
	if tokens_np.size!=num_tokens:raise ValueError('short read')
	return torch.from_numpy(tokens_np.astype(np.uint16,copy=_D))
class TokenStream:
	def __init__(self,pattern):
		self.files=[Path(p)for p in sorted(glob.glob(pattern))]
		if not self.files:raise FileNotFoundError(f"No files found for pattern: {pattern}")
		self.file_idx=0;self.tokens=load_data_shard(self.files[0]);self.pos=0
	def _advance_file(self):self.file_idx=(self.file_idx+1)%len(self.files);self.tokens=load_data_shard(self.files[self.file_idx]);self.pos=0
	def take(self,n):
		chunks=[];remaining=n
		while remaining>0:
			avail=self.tokens.numel()-self.pos
			if avail<=0:self._advance_file();continue
			k=min(remaining,avail);chunks.append(self.tokens[self.pos:self.pos+k]);self.pos+=k;remaining-=k
		return chunks[0]if len(chunks)==1 else torch.cat(chunks)
class DistributedTokenLoader:
	def __init__(self,pattern,rank,world_size,device):self.rank=rank;self.world_size=world_size;self.device=device;self.stream=TokenStream(pattern)
	def next_batch(self,global_tokens,seq_len,grad_accum_steps):local_tokens=global_tokens//(self.world_size*grad_accum_steps);per_rank_span=local_tokens+1;chunk=self.stream.take(per_rank_span*self.world_size);start=self.rank*per_rank_span;local=chunk[start:start+per_rank_span].to(dtype=torch.int64);x=local[:-1].reshape(-1,seq_len);y=local[1:].reshape(-1,seq_len);return x.to(self.device,non_blocking=_B),y.to(self.device,non_blocking=_B)
class BigramHashEmbedding(nn.Module):
	def __init__(self,bigram_vocab_size,bigram_dim,model_dim):
		super().__init__();self.bigram_vocab_size=bigram_vocab_size;self.embed=nn.Embedding(bigram_vocab_size,bigram_dim);nn.init.zeros_(self.embed.weight);self.proj=CastedLinear(bigram_dim,model_dim,bias=_D)if bigram_dim!=model_dim else _A
		if self.proj is not _A:nn.init.zeros_(self.proj.weight)
		self.scale=nn.Parameter(torch.tensor(.05,dtype=torch.float32))
	def bigram_hash(self,tokens):t=tokens.to(torch.int32);mod=self.bigram_vocab_size-1;out=torch.empty_like(t);out[...,0]=mod;out[...,1:]=torch.bitwise_xor(36313*t[...,1:],27191*t[...,:-1])%mod;return out.long()
	def forward(self,token_ids):
		h=self.embed(self.bigram_hash(token_ids))
		if self.proj is not _A:h=self.proj(h)
		return h*self.scale.to(dtype=h.dtype)
class ValueEmbedding(nn.Module):
	def __init__(self,vocab_size,ve_dim,model_dim):
		super().__init__();self.embed=nn.Embedding(vocab_size,ve_dim);nn.init.normal_(self.embed.weight,std=.01);self.proj=CastedLinear(ve_dim,model_dim,bias=_D)if ve_dim!=model_dim else _A
		if self.proj is not _A:nn.init.zeros_(self.proj.weight)
		self.scale=nn.Parameter(torch.tensor(.1,dtype=torch.float32))
	def forward(self,token_ids):
		h=self.embed(token_ids)
		if self.proj is not _A:h=self.proj(h)
		return h*self.scale.to(dtype=h.dtype)
class RMSNorm(nn.Module):
	def __init__(self,eps=_A):super().__init__();self.eps=eps
	def forward(self,x):return F.rms_norm(x,(x.size(-1),),eps=self.eps)
class CastedLinear(nn.Linear):
	_qat_enabled:bool=_D
	def forward(self,x):
		w=self.weight.to(x.dtype)
		if CastedLinear._qat_enabled and self.training and w.ndim==2:
			with torch.no_grad():w32=self.weight.float();row_max=w32.abs().amax(dim=1);scale=(row_max/31.).clamp_min(_C/31.);w_q=(torch.clamp(torch.round(w32/scale[:,_A]),-32,31)*scale[:,_A]).to(x.dtype)
			w=w+(w_q-w).detach()
		bias=self.bias.to(x.dtype)if self.bias is not _A else _A;return F.linear(x,w,bias)
def restore_low_dim_params_to_fp32(module):
	with torch.no_grad():
		for(name,param)in module.named_parameters():
			if(param.ndim<2 or any(p in name for p in _N.split(',')))and param.dtype!=torch.float32:param.data=param.data.float()
class Rotary(nn.Module):
	def __init__(self,dim,base=1e4,train_seq_len=1024,rope_dims=0):super().__init__();self.dim=dim;self.base=base;self.train_seq_len=train_seq_len;self.rope_dims=rope_dims if rope_dims>0 else dim;inv_freq=_C/base**(torch.arange(0,self.rope_dims,2,dtype=torch.float32)/self.rope_dims);self.register_buffer('inv_freq',inv_freq,persistent=_D);self._seq_len_cached=0;self._cos_cached=_A;self._sin_cached=_A
	def forward(self,seq_len,device,dtype):
		if self._cos_cached is _A or self._sin_cached is _A or self._seq_len_cached!=seq_len or self._cos_cached.device!=device:
			rd=self.rope_dims
			if seq_len>self.train_seq_len:scale=seq_len/self.train_seq_len;new_base=self.base*scale**(rd/(rd-2));inv_freq=_C/new_base**(torch.arange(0,rd,2,dtype=torch.float32,device=device)/rd)
			else:inv_freq=self.inv_freq.to(device)
			t=torch.arange(seq_len,device=device,dtype=inv_freq.dtype);freqs=torch.outer(t,inv_freq);self._cos_cached=freqs.cos()[_A,:,_A,:];self._sin_cached=freqs.sin()[_A,:,_A,:];self._seq_len_cached=seq_len
		return self._cos_cached.to(dtype=dtype),self._sin_cached.to(dtype=dtype)
def apply_rotary_emb(x,cos,sin,rope_dims=0):
	if rope_dims>0 and rope_dims<x.size(-1):x_rope,x_pass=x[...,:rope_dims],x[...,rope_dims:];half=rope_dims//2;x1,x2=x_rope[...,:half],x_rope[...,half:];x_rope=torch.cat((x1*cos+x2*sin,x1*-sin+x2*cos),dim=-1);return torch.cat((x_rope,x_pass),dim=-1)
	half=x.size(-1)//2;x1,x2=x[...,:half],x[...,half:];return torch.cat((x1*cos+x2*sin,x1*-sin+x2*cos),dim=-1)
class CausalSelfAttention(nn.Module):
	def __init__(self,dim,num_heads,num_kv_heads,rope_base,qk_gain_init):
		super().__init__()
		if dim%num_heads!=0:raise ValueError('dim%heads')
		if num_heads%num_kv_heads!=0:raise ValueError('heads%kv')
		self.num_heads=num_heads;self.num_kv_heads=num_kv_heads;self.head_dim=dim//num_heads
		if self.head_dim%2!=0:raise ValueError('head_dim odd')
		self.q_gain=nn.Parameter(torch.full((num_heads,),qk_gain_init,dtype=torch.float32));self.rope_dims=0;self.rotary=Rotary(self.head_dim,base=rope_base,train_seq_len=1024);self.use_xsa=_D
	def _xsa_efficient(self,y,v):'';B,T,H,D=y.shape;Hkv=v.size(-2);group=H//Hkv;y_g=y.reshape(B,T,Hkv,group,D);vn=F.normalize(v,dim=-1).unsqueeze(-2);proj=(y_g*vn).sum(dim=-1,keepdim=_B)*vn;return(y_g-proj).reshape(B,T,H,D)
	def forward(self,x,q_w,k_w,v_w,out_w,v_embed=_A):
		bsz,seqlen,dim=x.shape;q=F.linear(x,q_w.to(x.dtype)).reshape(bsz,seqlen,self.num_heads,self.head_dim);k=F.linear(x,k_w.to(x.dtype)).reshape(bsz,seqlen,self.num_kv_heads,self.head_dim);v=F.linear(x,v_w.to(x.dtype))
		if v_embed is not _A:v=v+v_embed
		v=v.reshape(bsz,seqlen,self.num_kv_heads,self.head_dim);raw_v=_A;q=F.rms_norm(q,(q.size(-1),));k=F.rms_norm(k,(k.size(-1),));cos,sin=self.rotary(seqlen,x.device,q.dtype);q=apply_rotary_emb(q,cos,sin,self.rope_dims);k=apply_rotary_emb(k,cos,sin,self.rope_dims);q=q*self.q_gain.to(dtype=q.dtype)[_A,_A,:,_A];y=flash_attn_3_func(q,k,v,causal=_B)
		if self.use_xsa:y=self._xsa_efficient(y,v)
		y=y.reshape(bsz,seqlen,dim);return F.linear(y,out_w.to(x.dtype)),raw_v
class SmearGate(nn.Module):
	def __init__(self,dim):super().__init__();self.gate=nn.Parameter(torch.zeros(dim,dtype=torch.float32))
	def forward(self,x):g=torch.sigmoid(self.gate.to(dtype=x.dtype))[_A,_A,:];x_prev=torch.cat([torch.zeros_like(x[:,:1]),x[:,:-1]],dim=1);return(1-g)*x+g*x_prev
class MLP(nn.Module):
	def __init__(self,dim,mlp_mult):super().__init__()
	def forward(self,x,up_w,down_w):x=F.leaky_relu(F.linear(x,up_w.to(x.dtype)),negative_slope=.5);return F.linear(x.square(),down_w.to(x.dtype))
class Block(nn.Module):
	def __init__(self,dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init,layer_idx=0,ln_scale=_D):super().__init__();self.attn_norm=RMSNorm();self.mlp_norm=RMSNorm();self.attn=CausalSelfAttention(dim,num_heads,num_kv_heads,rope_base,qk_gain_init);self.mlp=MLP(dim,mlp_mult);self.attn_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.mlp_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.resid_mix=nn.Parameter(torch.stack((torch.ones(dim),torch.zeros(dim))).float());self.ln_scale_factor=_C/math.sqrt(layer_idx+1)if ln_scale else _C
	def forward(self,x,x0,q_w,k_w,v_w,out_w,up_w,down_w,v_embed=_A):mix=self.resid_mix.to(dtype=x.dtype);x_in=mix[0][_A,_A,:]*x+mix[1][_A,_A,:]*x0;attn_out,_=self.attn(self.attn_norm(x_in)*self.ln_scale_factor,q_w,k_w,v_w,out_w,v_embed=v_embed);x_out=x_in+self.attn_scale.to(dtype=x_in.dtype)[_A,_A,:]*attn_out;x_out=x_out+self.mlp_scale.to(dtype=x_out.dtype)[_A,_A,:]*self.mlp(self.mlp_norm(x_out)*self.ln_scale_factor,up_w,down_w);return x_out,_A
def _fake_quantize(w,bits=6):
	clip_range=(1<<bits-1)-1;w32=w.float()
	if w32.ndim>=2:row_max=w32.abs().amax(dim=-1);scale=(row_max/clip_range).clamp_min(_C/clip_range);dims=(slice(_A),)*(w32.ndim-1)+(_A,);w_q=(torch.clamp(torch.round(w32/scale[dims]),-clip_range,clip_range)*scale[dims]).to(w.dtype)
	else:amax=w32.abs().max();scale=(amax/clip_range).clamp_min(_C/clip_range);w_q=(torch.clamp(torch.round(w32/scale),-clip_range,clip_range)*scale).to(w.dtype)
	return w+(w_q-w).detach()
class GPT(nn.Module):
	def __init__(self,vocab_size,num_layers,model_dim,num_heads,num_kv_heads,mlp_mult,tie_embeddings,tied_embed_init_std,logit_softcap,rope_base,qk_gain_init,xsa_last_n=0,rope_dims=0,ln_scale=_D,core_start=3,core_end=8,num_passes=1,core_quant_bits=6,core_quant_enabled=_D,residual_scale=_A,interpass_rmsnorm=_B,bigram_vocab_size=0,bigram_dim=32,ve_enabled=_D,ve_dim=128,ve_layers='9,10'):
		super().__init__();self._ve_target_dim=num_kv_heads*(model_dim//num_heads)
		if logit_softcap<=_E:raise ValueError('logit_softcap must be >0')
		self.tie_embeddings=tie_embeddings;self.tied_embed_init_std=tied_embed_init_std;self.logit_softcap=logit_softcap;self.core_start=core_start;self.core_end=min(core_end,num_layers);self.interpass_rmsnorm=interpass_rmsnorm;self.num_passes=num_passes;self.core_quant_bits=core_quant_bits;self.core_quant_enabled=core_quant_enabled;self.num_stem=core_start;self.num_core=self.core_end-core_start;self.num_tail=num_layers-self.core_end;self.residual_scale=residual_scale;self.tok_emb=nn.Embedding(vocab_size,model_dim);self.bigram=BigramHashEmbedding(bigram_vocab_size,bigram_dim,model_dim)if bigram_vocab_size>0 else _A;self.smear=SmearGate(model_dim);self.num_skip_weights=min(self.num_stem,self.num_tail);self.skip_weights=nn.Parameter(torch.ones(self.num_skip_weights,model_dim,dtype=torch.float32));head_dim=model_dim//num_heads;kv_dim=num_kv_heads*head_dim;mlp_dim=int(mlp_mult*model_dim);self.num_layers=num_layers;self.qo_bank=nn.Parameter(torch.empty(2*num_layers,model_dim,model_dim));self.kv_bank=nn.Parameter(torch.empty(2*num_layers,kv_dim,model_dim));self.mlp_up_bank=nn.Parameter(torch.empty(num_layers,mlp_dim,model_dim));self.mlp_down_bank=nn.Parameter(torch.empty(num_layers,model_dim,mlp_dim));self.blocks=nn.ModuleList([Block(model_dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init,layer_idx=i,ln_scale=ln_scale)for i in range(num_layers)])
		if rope_dims>0:
			head_dim=model_dim//num_heads
			for block in self.blocks:block.attn.rope_dims=rope_dims;block.attn.rotary=Rotary(head_dim,base=rope_base,train_seq_len=1024,rope_dims=rope_dims)
		self.ve_layer_indices=[int(x)for x in ve_layers.split(',')if x.strip()]if ve_enabled else[];kv_dim_ve=self._ve_target_dim
		if self.ve_layer_indices:self.ve_shared=ValueEmbedding(vocab_size,ve_dim,kv_dim_ve);self.ve_layer_scales=nn.ParameterList([nn.Parameter(torch.ones(1,dtype=torch.float32))for _ in self.ve_layer_indices])
		else:self.ve_shared=_A;self.ve_layer_scales=nn.ParameterList()
		self.value_embeds=nn.ModuleList();self.final_norm=RMSNorm();self.lm_head=_A if tie_embeddings else CastedLinear(model_dim,vocab_size,bias=_D)
		if self.lm_head is not _A:self.lm_head._zero_init=_B
		self.mtp_heads=nn.ModuleList()
		if xsa_last_n>0:
			for i in range(max(0,num_layers-xsa_last_n),num_layers):
				if i<core_start or i>=self.core_end:self.blocks[i].attn.use_xsa=_B
		self._init_weights()
	def _init_weights(self):
		if self.tie_embeddings:nn.init.normal_(self.tok_emb.weight,mean=_E,std=self.tied_embed_init_std)
		n=self.num_layers;proj_scale=_C/math.sqrt(2*n)
		for i in range(n):nn.init.orthogonal_(self.qo_bank.data[i],gain=_C);nn.init.zeros_(self.qo_bank.data[n+i]);nn.init.orthogonal_(self.kv_bank.data[i],gain=_C);nn.init.orthogonal_(self.kv_bank.data[n+i],gain=_C);nn.init.orthogonal_(self.mlp_up_bank.data[i],gain=_C);nn.init.zeros_(self.mlp_down_bank.data[i]);self.qo_bank.data[n+i].mul_(proj_scale);self.mlp_down_bank.data[i].mul_(proj_scale)
		for(name,module)in self.named_modules():
			if isinstance(module,nn.Linear):
				if getattr(module,'_zero_init',_D):nn.init.zeros_(module.weight)
				elif module.weight.ndim==2 and module.weight.shape[0]>=64 and module.weight.shape[1]>=64:nn.init.orthogonal_(module.weight,gain=_C)
	def _get_ve(self,layer_idx,input_ids,ve_cache=_A):
		A='ve'
		if self.ve_shared is _A or layer_idx not in self.ve_layer_indices:return
		if ve_cache is not _A and A not in ve_cache:ve_cache[A]=self.ve_shared(input_ids)
		ve_base=ve_cache[A]if ve_cache is not _A else self.ve_shared(input_ids);ve_idx=self.ve_layer_indices.index(layer_idx);return ve_base*self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)
	def _get_bank_weights(self,bi):
		n=self.num_layers;q_w=self.qo_bank[bi];out_w=self.qo_bank[n+bi];k_w=self.kv_bank[bi];v_w=self.kv_bank[n+bi];up_w=self.mlp_up_bank[bi];down_w=self.mlp_down_bank[bi]
		if self.core_quant_enabled and self.training and self.core_start<=bi<self.core_end:q_w=_fake_quantize(q_w,self.core_quant_bits);out_w=_fake_quantize(out_w,self.core_quant_bits);k_w=_fake_quantize(k_w,self.core_quant_bits);v_w=_fake_quantize(v_w,self.core_quant_bits);up_w=_fake_quantize(up_w,self.core_quant_bits);down_w=_fake_quantize(down_w,self.core_quant_bits)
		return q_w,k_w,v_w,out_w,up_w,down_w
	def _forward_hidden(self,input_ids,feedback_fn=_A,stabilizer=_A):
		n=self.num_layers;x=self.tok_emb(input_ids)
		if self.bigram is not _A:x=x+self.bigram(input_ids)
		x=F.rms_norm(x,(x.size(-1),));x=self.smear(x);x0=x;skips=[];ve_cache={}
		for i in range(self.core_start):ve=self._get_ve(i,input_ids,ve_cache);q_w,k_w,v_w,out_w,up_w,down_w=self._get_bank_weights(i);x,_=self.blocks[i](x,x0,q_w,k_w,v_w,out_w,up_w,down_w,v_embed=ve);skips.append(x)
		h_core_in=x
		for k in range(self.num_passes):
			if k>0 and self.interpass_rmsnorm:x=F.rms_norm(x,(x.size(-1),))
			if feedback_fn is not _A:x=x+feedback_fn(x,k)
			if stabilizer is not _A:x=stabilizer.clip(x)
			x_before_pass=x
			for j in range(self.core_start,self.core_end):h_prev=x;ve=self._get_ve(j,input_ids,ve_cache);q_w,k_w,v_w,out_w,up_w,down_w=self._get_bank_weights(j);x,_=self.blocks[j](x,x0,q_w,k_w,v_w,out_w,up_w,down_w,v_embed=ve)
			if self.residual_scale is not _A and k>0:delta=x-x_before_pass;x=x_before_pass+self.residual_scale(delta,k)
		h_core_out=x
		for i in range(self.core_end,n):
			ti=i-self.core_end
			if ti<len(skips):x=x+self.skip_weights[ti].to(dtype=x.dtype)[_A,_A,:]*skips.pop()
			ve=self._get_ve(i,input_ids,ve_cache);q_w,k_w,v_w,out_w,up_w,down_w=self._get_bank_weights(i);x,_=self.blocks[i](x,x0,q_w,k_w,v_w,out_w,up_w,down_w,v_embed=ve)
		x=self.final_norm(x);return x,h_core_in,h_core_out
	def forward(self,input_ids,target_ids,feedback_fn=_A,stabilizer=_A):
		x,h_core_in,h_core_out=self._forward_hidden(input_ids,feedback_fn,stabilizer);x_flat=x.reshape(-1,x.size(-1));targets=target_ids.reshape(-1)
		if self.tie_embeddings:logits_proj=F.linear(x_flat,self.tok_emb.weight)
		else:
			if self.lm_head is _A:raise RuntimeError('no lm_head')
			logits_proj=self.lm_head(x_flat)
		logits=self.logit_softcap*torch.tanh(logits_proj/self.logit_softcap);main_loss=F.cross_entropy(logits.float(),targets,reduction='mean')
		if stabilizer is not _A and stabilizer.jacobian_proxy_weight>0:main_loss=main_loss+stabilizer.jacobian_proxy_loss(h_core_in,h_core_out)
		return main_loss
	def forward_logits(self,input_ids,feedback_fn=_A,stabilizer=_A):
		'';x,_,_=self._forward_hidden(input_ids,feedback_fn,stabilizer)
		if self.tie_embeddings:logits_proj=F.linear(x,self.tok_emb.weight)
		else:logits_proj=self.lm_head(x)
		return self.logit_softcap*torch.tanh(logits_proj/self.logit_softcap)
def eval_val_sliding_ttt(args,base_model,rank,world_size,device,val_tokens,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut,stride,batch_seqs=32,log0=print,feedback_fn=_A,feedback_module=_A):
	seq_len=args.train_seq_len;total_tokens=val_tokens.numel()-1;ttt_chunk=args.ttt_chunk_tokens;window_starts=[ws for ws in range(0,total_tokens,stride)if min(ws+seq_len,total_tokens)-ws>=stride or ws==0];num_chunks=(total_tokens+ttt_chunk-1)//ttt_chunk;chunk_windows=[[]for _ in range(num_chunks)]
	for ws in window_starts:end=min(ws+seq_len,total_tokens);wlen=end-ws;s=0 if ws==0 else max(wlen-stride,0);scored_start=ws+s;ci=min(scored_start//ttt_chunk,num_chunks-1);chunk_windows[ci].append(ws)
	log0(f"ttt_sliding:start chunks={num_chunks} chunk_tokens={ttt_chunk} total_windows={len(window_starts)} stride={stride} ttt_lr={args.ttt_lr} ttt_epochs={args.ttt_epochs} freeze_blocks={args.ttt_freeze_blocks}");loss_sum=torch.zeros((),device=device,dtype=torch.float64);token_count=torch.zeros((),device=device,dtype=torch.float64);byte_count=torch.zeros((),device=device,dtype=torch.float64);frozen_block_ids=set(range(min(args.ttt_freeze_blocks,len(base_model.blocks))));ttt_params=[]
	for(name,p)in base_model.named_parameters():
		freeze=_D
		for bi in frozen_block_ids:
			if f"blocks.{bi}."in name:freeze=_B;break
		if freeze:p.requires_grad_(_D)
		else:p.requires_grad_(_B);ttt_params.append(p)
	if feedback_module is not _A:
		for p in feedback_module.parameters():p.requires_grad_(_B);ttt_params.append(p)
	log0(f"ttt_sliding:params unfrozen={sum(p.numel()for p in ttt_params)} frozen={sum(p.numel()for p in base_model.parameters()if not p.requires_grad)}");optimizer=torch.optim.SGD(ttt_params,lr=args.ttt_lr,momentum=args.ttt_momentum);t0=time.perf_counter()
	for ci in range(num_chunks):
		windows=chunk_windows[ci]
		if not windows:continue
		chunk_start=ci*ttt_chunk;chunk_end=min((ci+1)*ttt_chunk,total_tokens);my_s=len(windows)*rank//world_size;my_e=len(windows)*(rank+1)//world_size;my_windows=windows[my_s:my_e];base_model.eval()
		with torch.inference_mode():
			for bi in range(0,len(my_windows),batch_seqs):
				batch_ws=my_windows[bi:bi+batch_seqs];bsz=len(batch_ws);x_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);y_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);wlens=[]
				for(i,ws)in enumerate(batch_ws):end=min(ws+seq_len,total_tokens);wlen=end-ws;wlens.append(wlen);chunk_tok=val_tokens[ws:end+1].to(dtype=torch.int64,device=device);x_batch[i,:wlen]=chunk_tok[:-1];y_batch[i,:wlen]=chunk_tok[1:]
				with torch.autocast(device_type=_I,dtype=torch.bfloat16):logits=base_model.forward_logits(x_batch,feedback_fn=feedback_fn)
				nll=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),y_batch.reshape(-1),reduction='none').reshape(bsz,seq_len)
				for(i,ws)in enumerate(batch_ws):wlen=wlens[i];s=0 if ws==0 else max(wlen-stride,0);scored_nll=nll[i,s:wlen].to(torch.float64);loss_sum+=scored_nll.sum();token_count+=float(wlen-s);tgt,prev=y_batch[i,s:wlen],x_batch[i,s:wlen];tb=base_bytes_lut[tgt].to(torch.float64);tb+=(has_leading_space_lut[tgt]&~is_boundary_token_lut[prev]).to(torch.float64);byte_count+=tb.sum()
		is_last_chunk=ci==num_chunks-1
		if not is_last_chunk and args.ttt_epochs>0:
			base_model.train();chunk_seqs=(chunk_end-chunk_start)//seq_len
			if chunk_seqs>0:
				cos_lr=args.ttt_lr*.5*(_C+math.cos(math.pi*ci/max(num_chunks-1,1)))
				for pg in optimizer.param_groups:pg[_G]=cos_lr
				my_seq_s=chunk_seqs*rank//world_size;my_seq_e=chunk_seqs*(rank+1)//world_size;my_chunk_seqs=my_seq_e-my_seq_s
				for _ep in range(args.ttt_epochs):
					for bs in range(0,my_chunk_seqs,args.ttt_batch_seqs):
						be=min(bs+args.ttt_batch_seqs,my_chunk_seqs);actual_bs=my_seq_s+bs;start_tok=chunk_start+actual_bs*seq_len;end_tok=chunk_start+(my_seq_s+be)*seq_len+1
						if end_tok>val_tokens.numel():continue
						local=val_tokens[start_tok:end_tok].to(device=device,dtype=torch.int64);x=local[:-1].reshape(-1,seq_len);y=local[1:].reshape(-1,seq_len);optimizer.zero_grad(set_to_none=_B)
						with torch.autocast(device_type=_I,dtype=torch.bfloat16):loss=base_model(x,y,feedback_fn=feedback_fn)
						loss.backward()
						if world_size>1:
							for p in ttt_params:
								if p.grad is not _A:dist.all_reduce(p.grad,op=dist.ReduceOp.AVG)
						torch.nn.utils.clip_grad_norm_(ttt_params,args.ttt_grad_clip);optimizer.step()
		if rank==0 and(ci%10==0 or ci==num_chunks-1):elapsed=time.perf_counter()-t0;rl=loss_sum.item()/max(token_count.item(),1);rbpb=rl/math.log(2.)*(token_count.item()/max(byte_count.item(),1))if token_count.item()>0 else _E;log0(f"  ttt_chunk [{ci+1}/{num_chunks}] bpb={rbpb:.6f} time={elapsed:.1f}s")
	if dist.is_available()and dist.is_initialized():dist.all_reduce(loss_sum,op=dist.ReduceOp.SUM);dist.all_reduce(token_count,op=dist.ReduceOp.SUM);dist.all_reduce(byte_count,op=dist.ReduceOp.SUM)
	val_loss=(loss_sum/token_count).item();val_bpb=val_loss/math.log(2.)*(token_count.item()/byte_count.item())
	for p in base_model.parameters():p.requires_grad_(_B)
	base_model.eval();log0(f"ttt_sliding:done val_loss={val_loss:.6f}{ val_bpb=:.6f} elapsed={time.perf_counter()-t0:.1f}s");return val_loss,val_bpb
def quantize_int6_per_row(t,clip_range=31):
	t32=t.float()
	if t32.ndim==2:
		best_q,best_s,best_err=_A,_A,float('inf')
		for pct in[.999,.9995,.9999,.99999,_C]:
			if pct<_C:row_clip=torch.quantile(t32.abs(),pct,dim=1)
			else:row_clip=t32.abs().amax(dim=1)
			s=(row_clip/clip_range).clamp_min(_C/clip_range).to(torch.float16);q=torch.clamp(torch.round(t32/s.float()[:,_A]),-clip_range,clip_range).to(torch.int8);recon=q.float()*s.float()[:,_A];err=(t32-recon).pow(2).mean().item()
			if err<best_err:best_q,best_s,best_err=q,s,err
		return best_q,best_s
	amax=t32.abs().max().item();scale=torch.tensor(amax/clip_range if amax>0 else _C,dtype=torch.float16);q=torch.clamp(torch.round(t32/scale.float()),-clip_range,clip_range).to(torch.int8);return q,scale
def _unbank_state_dict(sd,num_layers):
	out={};n=num_layers
	for(name,tensor)in sd.items():
		if name==_O:
			for i in range(n):out[f"blocks.{i}.attn.c_q.weight"]=tensor[i];out[f"blocks.{i}.attn.proj.weight"]=tensor[n+i]
		elif name==_P:
			for i in range(n):out[f"blocks.{i}.attn.c_k.weight"]=tensor[i];out[f"blocks.{i}.attn.c_v.weight"]=tensor[n+i]
		elif name==_Q:
			for i in range(n):out[f"blocks.{i}.mlp.fc.weight"]=tensor[i]
		elif name==_R:
			for i in range(n):out[f"blocks.{i}.mlp.proj.weight"]=tensor[i]
		else:out[name]=tensor
	return out
def _rebank_state_dict(sd,num_layers,template_sd):
	out={};n=num_layers;qo_slices=[_A]*(2*n);kv_slices=[_A]*(2*n);up_slices=[_A]*n;down_slices=[_A]*n;consumed=set()
	for i in range(n):
		qk=f"blocks.{i}.attn.c_q.weight"
		if qk in sd:qo_slices[i]=sd[qk];consumed.add(qk)
		ok=f"blocks.{i}.attn.proj.weight"
		if ok in sd:qo_slices[n+i]=sd[ok];consumed.add(ok)
		kk=f"blocks.{i}.attn.c_k.weight"
		if kk in sd:kv_slices[i]=sd[kk];consumed.add(kk)
		vk=f"blocks.{i}.attn.c_v.weight"
		if vk in sd:kv_slices[n+i]=sd[vk];consumed.add(vk)
		fk=f"blocks.{i}.mlp.fc.weight"
		if fk in sd:up_slices[i]=sd[fk];consumed.add(fk)
		dk=f"blocks.{i}.mlp.proj.weight"
		if dk in sd:down_slices[i]=sd[dk];consumed.add(dk)
	out[_O]=torch.stack(qo_slices).to(dtype=template_sd[_O].dtype);out[_P]=torch.stack(kv_slices).to(dtype=template_sd[_P].dtype);out[_Q]=torch.stack(up_slices).to(dtype=template_sd[_Q].dtype);out[_R]=torch.stack(down_slices).to(dtype=template_sd[_R].dtype)
	for(name,tensor)in sd.items():
		if name not in consumed:out[name]=tensor
	return out
def mixed_quantize_int6(state_dict,int6_cats,core_start=-1,core_end=-1):
	A='type';num_layers_total=max((int(k.split('.')[1])for k in state_dict if k.startswith('blocks.')),default=0)+1;late_k_layers=set(range(num_layers_total-2,num_layers_total));result={};meta={}
	for(name,tensor)in state_dict.items():
		t=tensor.detach().cpu().contiguous();cat='embed'if'tok_emb'in name or'lm_head'in name else'mlp'if'.mlp.'in name else'attn'if'.attn.'in name else'other'
		if not t.is_floating_point()or t.numel()<=65536:result[name]=t.to(torch.float16)if t.is_floating_point()else t;meta[name]=_Y;continue
		if any(p in name for p in _N.split(',')):result[name]=t.float();meta[name]=_Z;continue
		if cat in int6_cats and t.ndim>=1:q,s=quantize_int6_per_row(t);result[name+'.q']=q;result[name+_S]=s;meta[name]={A:'int6'}
		else:q,s=quantize_float_tensor(t);result[name+'.q']=q;result[name+_S]=s;meta[name]={A:'int8'}
	return result,meta
def dequantize_mixed_int6(result,meta,template_sd):
	out={}
	for(name,orig)in template_sd.items():
		info=meta.get(name)
		if info is _A:continue
		orig_dtype=orig.dtype
		if info in(_Y,_Z,'passthrough_fp16'):
			t=result[name]
			if t.dtype==torch.float16 and orig_dtype in(torch.float32,torch.bfloat16):t=t.to(orig_dtype)
			out[name]=t;continue
		q,s=result[name+'.q'],result[name+_S]
		if s.ndim>0:out[name]=(q.float()*s.float().view(q.shape[0],*[1]*(q.ndim-1))).to(orig_dtype)
		else:out[name]=(q.float()*float(s.item())).to(orig_dtype)
	return out
def parse_args():A='store_true';p=argparse.ArgumentParser();p.add_argument('--feedback-rank',type=int,default=2);p.add_argument('--feedback-mode',type=str,default=_T);p.add_argument('--per-pass-feedback',action=A);p.add_argument('--residual-scale-init',type=float,default=.5);p.add_argument('--jacobian-proxy-weight',type=float,default=.01);p.add_argument('--no-interpass-rmsnorm',action=A);return p.parse_args()
def _make_gpt(args,cli,num_passes,**kw):return GPT(vocab_size=args.vocab_size,num_layers=args.num_layers,model_dim=args.model_dim,num_heads=args.num_heads,num_kv_heads=args.num_kv_heads,mlp_mult=args.mlp_mult,tie_embeddings=args.tie_embeddings,tied_embed_init_std=args.tied_embed_init_std,logit_softcap=args.logit_softcap,rope_base=args.rope_base,qk_gain_init=args.qk_gain_init,xsa_last_n=args.xsa_last_n,rope_dims=args.rope_dims,ln_scale=args.ln_scale,core_start=args.core_start,core_end=args.core_end,num_passes=num_passes,interpass_rmsnorm=not cli.no_interpass_rmsnorm,bigram_vocab_size=args.bigram_vocab_size,bigram_dim=args.bigram_dim,ve_enabled=args.ve_enabled,ve_dim=args.ve_dim,ve_layers=args.ve_layers,**kw)
def _promote_fp32(m):
	m.qo_bank.data=m.qo_bank.data.float();m.kv_bank.data=m.kv_bank.data.float();m.mlp_up_bank.data=m.mlp_up_bank.data.float();m.mlp_down_bank.data=m.mlp_down_bank.data.float()
	for mod in m.modules():
		if isinstance(mod,CastedLinear):mod.float()
	restore_low_dim_params_to_fp32(m)
def main():
	G='final_model.int6.ptz';F='final_model.pt';E='WORLD_SIZE';D='RANK';C='_feedback.';B='_fb.';A='base_lr';cli=parse_args();code=Path(__file__).read_text(encoding=_J);args=Hyperparameters();distributed=D in os.environ and E in os.environ;rank=int(os.environ.get(D,_H));world_size=int(os.environ.get(E,'1'));local_rank=int(os.environ.get('LOCAL_RANK',_H))
	if world_size<=0:raise ValueError('bad WORLD_SIZE')
	if 8%world_size!=0:raise ValueError('WORLD_SIZE must divide 8')
	grad_accum_steps=8//world_size;grad_scale=_C/grad_accum_steps
	if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
	device=torch.device(_I,local_rank);torch.cuda.set_device(device)
	if distributed:dist.init_process_group(backend='nccl',device_id=device);dist.barrier()
	master_process=rank==0;torch.backends.cuda.matmul.allow_tf32=_B;torch.backends.cudnn.allow_tf32=_B;from torch.backends.cuda import enable_cudnn_sdp,enable_flash_sdp,enable_math_sdp,enable_mem_efficient_sdp;enable_cudnn_sdp(_D);enable_flash_sdp(_B);enable_mem_efficient_sdp(_D);enable_math_sdp(_D);logfile=_A
	if master_process:os.makedirs('logs',exist_ok=_B);logfile=f"logs/{args.run_id}.txt";print(logfile)
	def log0(msg,console=_B):
		if not master_process:return
		if console:print(msg)
		if logfile is not _A:
			with open(logfile,'a',encoding=_J)as f:print(msg,file=f)
	log0(code,console=_D);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
	if not args.tokenizer_path.endswith('.model'):raise ValueError('need .model tokenizer')
	sp=spm.SentencePieceProcessor(model_file=args.tokenizer_path)
	if int(sp.vocab_size())!=args.vocab_size:raise ValueError('vocab size mismatch')
	dataset_dir=Path(args.data_path).resolve();actual_train_files=len(list(dataset_dir.glob(_U)));effective_eval_seq_len=args.eval_seq_len if args.eval_seq_len>0 else args.train_seq_len;val_seq_len=max(args.train_seq_len,effective_eval_seq_len);val_tokens=load_validation_tokens(args.val_files,val_seq_len);base_bytes_lut,has_leading_space_lut,is_boundary_token_lut=build_sentencepiece_luts(sp,args.vocab_size,device);log0(f"val_bpb:enabled tokenizer_path={args.tokenizer_path}");log0(f"train:{dataset_dir.name} shards:{actual_train_files} val_tokens:{val_tokens.numel()-1}");CastedLinear._qat_enabled=args.qat_enabled;base_model=_make_gpt(args,cli,args.num_passes,core_quant_bits=args.core_quant_bits,core_quant_enabled=args.core_quant_enabled,residual_scale=_A).to(device).bfloat16();_promote_fp32(base_model);feedback=_A;feedback_fn=_A;stabilizer=_A;residual_scale=_A;extra_scalar_params=[];passes_schedule=[]
	if args.passes_schedule_str:
		for entry in args.passes_schedule_str.split(','):s,p=entry.strip().split(':');passes_schedule.append((int(s),int(p)))
		passes_schedule.sort(key=lambda x:x[0])
	max_passes=max((p for(_,p)in passes_schedule),default=args.num_passes);max_passes=max(max_passes,args.eval_passes if args.eval_passes>0 else args.num_passes);needs_recurrence=max_passes>1
	if cli.feedback_mode!='none'and needs_recurrence:
		feedback=ErrorFeedbackModule(dim=args.model_dim,rank=cli.feedback_rank,feedback_mode=cli.feedback_mode,per_pass=cli.per_pass_feedback,num_passes=max_passes).to(device).bfloat16();restore_low_dim_params_to_fp32(feedback);extra_scalar_params.extend(feedback.parameters())
		def feedback_fn(h,pass_idx):return feedback(h,pass_idx)
		log0(f"feedback: {cli.feedback_mode} r={cli.feedback_rank} params={sum(p.numel()for p in feedback.parameters())}")
	if needs_recurrence:
		stabilizer=RecurrentStabilizer(jacobian_proxy_weight=cli.jacobian_proxy_weight)
		if cli.residual_scale_init!=_C:residual_scale=ResidualScale(max_passes,cli.residual_scale_init).to(device);base_model.residual_scale=residual_scale;extra_scalar_params.extend(residual_scale.parameters())
	log0(f"recurrence: {args.core_start}-{args.core_end} passes={args.num_passes}/{max_passes} s/c/t={base_model.num_stem}/{base_model.num_core}/{base_model.num_tail} sched={passes_schedule}");compiled_model=torch.compile(base_model,dynamic=_D,fullgraph=_B);model=compiled_model;matrix_params=[base_model.qo_bank,base_model.kv_bank,base_model.mlp_up_bank,base_model.mlp_down_bank];block_named_params=list(base_model.blocks.named_parameters());scalar_params=[p for(name,p)in block_named_params if p.ndim<2 or any(p in name for p in _N.split(','))]
	if base_model.skip_weights.numel()>0:scalar_params.append(base_model.skip_weights)
	scalar_params.append(base_model.smear.gate);token_lr=args.tied_embed_lr if args.tie_embeddings else args.embed_lr;tok_params=[{_F:[base_model.tok_emb.weight],_G:token_lr,A:token_lr}]
	if base_model.bigram is not _A:
		tok_params.append({_F:[base_model.bigram.embed.weight],_G:token_lr,A:token_lr})
		if base_model.bigram.proj is not _A:scalar_params.append(base_model.bigram.proj.weight)
		scalar_params.append(base_model.bigram.scale)
	if base_model.ve_shared is not _A:
		tok_params.append({_F:[base_model.ve_shared.embed.weight],_G:token_lr,A:token_lr})
		if base_model.ve_shared.proj is not _A:scalar_params.append(base_model.ve_shared.proj.weight)
		scalar_params.append(base_model.ve_shared.scale)
		for s in base_model.ve_layer_scales:scalar_params.append(s)
	optimizer_tok=torch.optim.AdamW(tok_params,betas=(args.beta1,args.beta2),eps=args.adam_eps,weight_decay=args.adam_wd,fused=_B);optimizer_muon=Muon(matrix_params,lr=args.matrix_lr,momentum=args.muon_momentum,backend_steps=args.muon_backend_steps,weight_decay=args.muon_wd)
	for group in optimizer_muon.param_groups:group[A]=args.matrix_lr
	scalar_params.extend(extra_scalar_params);optimizer_scalar=torch.optim.AdamW([{_F:scalar_params,_G:args.scalar_lr,A:args.scalar_lr}],betas=(args.beta1,args.beta2),eps=args.adam_eps,weight_decay=args.adam_wd,fused=_B);replicated_params=list(optimizer_tok.param_groups[0][_F])
	for pg in optimizer_tok.param_groups[1:]:replicated_params.extend(pg[_F])
	replicated_params.extend(scalar_params);optimizer_head=_A
	if base_model.lm_head is not _A:optimizer_head=torch.optim.Adam([{_F:[base_model.lm_head.weight],_G:args.head_lr,A:args.head_lr}],betas=(args.beta1,args.beta2),eps=args.adam_eps,fused=_B);replicated_params.append(base_model.lm_head.weight)
	optimizers=[optimizer_tok,optimizer_muon,optimizer_scalar]
	if optimizer_head is not _A:optimizers.append(optimizer_head)
	log0(f"params:{sum(p.numel()for p in base_model.parameters())} ws:{world_size} ga:{grad_accum_steps} iters:{args.iterations} wc:{args.max_wallclock_seconds:.0f}s seed:{args.seed}");train_loader=DistributedTokenLoader(args.train_files,rank,world_size,device)
	def zero_grad_all():
		for opt in optimizers:opt.zero_grad(set_to_none=_B)
	max_wallclock_ms=1e3*args.max_wallclock_seconds if args.max_wallclock_seconds>0 else _A
	def lr_mul(step,elapsed_ms):
		if args.warmdown_iters<=0:return _C
		if max_wallclock_ms is _A:warmdown_start=max(args.iterations-args.warmdown_iters,0);return max((args.iterations-step)/max(args.warmdown_iters,1),_E)if warmdown_start<=step<args.iterations else _C
		step_ms=elapsed_ms/max(step,1);warmdown_ms=args.warmdown_iters*step_ms;remaining_ms=max(max_wallclock_ms-elapsed_ms,_E);return remaining_ms/max(warmdown_ms,1e-09)if remaining_ms<=warmdown_ms else _C
	if args.warmup_steps>0:
		initial_model_state={name:tensor.detach().cpu().clone()for(name,tensor)in base_model.state_dict().items()};initial_optimizer_states=[copy.deepcopy(opt.state_dict())for opt in optimizers];_precompile_passes=sorted(set(p for(_,p)in passes_schedule)-{args.num_passes})if passes_schedule else[];_qat_precompile_passes=_precompile_passes[-2:]if len(_precompile_passes)>=2 else _precompile_passes[:];_total_precompile=len(_precompile_passes)+len(_qat_precompile_passes);_precompile_start=args.warmup_steps-_total_precompile;model.train()
		for warmup_step in range(args.warmup_steps):
			if warmup_step>=_precompile_start:
				_pc_idx=warmup_step-_precompile_start
				if _pc_idx<len(_precompile_passes):base_model.num_passes=_precompile_passes[_pc_idx];CastedLinear._qat_enabled=_D;base_model.core_quant_enabled=_D
				else:_qat_idx=_pc_idx-len(_precompile_passes);base_model.num_passes=_qat_precompile_passes[_qat_idx];CastedLinear._qat_enabled=_B;base_model.core_quant_enabled=_B
			zero_grad_all()
			for micro_step in range(grad_accum_steps):
				x,y=train_loader.next_batch(args.train_batch_tokens,args.train_seq_len,grad_accum_steps)
				with torch.autocast(device_type=_I,dtype=torch.bfloat16,enabled=_B):warmup_loss=model(x,y,feedback_fn=feedback_fn,stabilizer=stabilizer)
				(warmup_loss*grad_scale).backward()
			if distributed:
				for p in base_model.parameters():
					if p.grad is not _A:dist.all_reduce(p.grad,op=dist.ReduceOp.AVG)
				if feedback is not _A:
					for p in feedback.parameters():
						if p.grad is not _A:dist.all_reduce(p.grad,op=dist.ReduceOp.AVG)
			for opt in optimizers:opt.step()
			zero_grad_all()
			if args.warmup_steps<=20 or(warmup_step+1)%10==0 or warmup_step+1==args.warmup_steps:log0(f"warmup_step:{warmup_step+1}/{args.warmup_steps}")
		base_model.num_passes=args.num_passes;CastedLinear._qat_enabled=args.qat_enabled;base_model.core_quant_enabled=args.core_quant_enabled
		if stabilizer is not _A:stabilizer.reset()
		base_model.load_state_dict(initial_model_state,strict=_B)
		for(opt,state)in zip(optimizers,initial_optimizer_states,strict=_B):opt.load_state_dict(state)
		zero_grad_all();train_loader=DistributedTokenLoader(args.train_files,rank,world_size,device)
	swa_state=_A;swa_count=0;_all_state=dict(base_model.state_dict())
	if feedback is not _A:
		for(k,v)in feedback.state_dict().items():_all_state[f"_fb.{k}"]=v
	ema_state={name:t.detach().float().clone()for(name,t)in _all_state.items()};ema_decay=.997;training_time_ms=_E;stop_after_step=_A;torch.cuda.synchronize();t0=time.perf_counter();step=0
	while _B:
		last_step=step==args.iterations or stop_after_step is not _A and step>=stop_after_step;should_validate=last_step or args.val_loss_every>0 and step%args.val_loss_every==0
		if should_validate:torch.cuda.synchronize();training_time_ms+=1e3*(time.perf_counter()-t0);val_loss,val_bpb=eval_val(args,model,rank,world_size,device,grad_accum_steps,val_tokens,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut);log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step,1):.2f}ms");torch.cuda.synchronize();t0=time.perf_counter()
		if last_step:
			if stop_after_step is not _A and step<args.iterations:log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
			break
		elapsed_ms=training_time_ms+1e3*(time.perf_counter()-t0);scale=lr_mul(step,elapsed_ms)
		if passes_schedule:
			target_passes=args.num_passes
			for(threshold_step,p)in passes_schedule:
				if step>=threshold_step:target_passes=p
			if target_passes!=base_model.num_passes:base_model.num_passes=target_passes;log0(f"progressive_passes: step:{step} num_passes:{target_passes}")
		if args.late_qat_threshold>0 and step>100 and scale<args.late_qat_threshold and not CastedLinear._qat_enabled:CastedLinear._qat_enabled=_B;base_model.core_quant_enabled=_B;log0(f"late_qat:enabled step:{step} scale:{scale:.4f} core_quant:on")
		zero_grad_all();train_loss=torch.zeros((),device=device)
		for micro_step in range(grad_accum_steps):
			x,y=train_loader.next_batch(args.train_batch_tokens,args.train_seq_len,grad_accum_steps)
			with torch.autocast(device_type=_I,dtype=torch.bfloat16,enabled=_B):loss=model(x,y,feedback_fn=feedback_fn,stabilizer=stabilizer)
			train_loss+=loss.detach();(loss*grad_scale).backward()
		train_loss/=grad_accum_steps;frac=min(step/args.muon_momentum_warmup_steps,_C)if args.muon_momentum_warmup_steps>0 else _C;muon_momentum=(1-frac)*args.muon_momentum_warmup_start+frac*args.muon_momentum
		for group in optimizer_muon.param_groups:group[_X]=muon_momentum
		for opt in optimizers:
			for group in opt.param_groups:group[_G]=group[A]*scale
		grad_norm=_A
		if args.grad_clip_norm>0:grad_norm=torch.nn.utils.clip_grad_norm_(base_model.parameters(),args.grad_clip_norm)
		optimizer_muon.launch_reduce_scatters()
		if distributed:
			for p in replicated_params:
				if p.grad is not _A:dist.all_reduce(p.grad,op=dist.ReduceOp.AVG)
		optimizer_tok.step();optimizer_scalar.step()
		if optimizer_head is not _A:optimizer_head.step()
		optimizer_muon.step();zero_grad_all()
		with torch.no_grad():
			_cur=dict(base_model.state_dict())
			if feedback is not _A:
				for(k,v)in feedback.state_dict().items():_cur[f"_fb.{k}"]=v
			for(name,t)in _cur.items():ema_state[name].mul_(ema_decay).add_(t.detach().float(),alpha=_C-ema_decay)
		step+=1;approx_training_time_ms=training_time_ms+1e3*(time.perf_counter()-t0)
		if args.swa_enabled and scale<.2 and step%args.swa_every==0:
			if swa_state is _A:swa_state={name:t.detach().cpu().clone()for(name,t)in base_model.state_dict().items()};swa_count=1;log0(f"swa:start step:{step}")
			else:
				for(name,t)in base_model.state_dict().items():swa_state[name]+=t.detach().cpu()
				swa_count+=1
		should_log_train=args.train_log_every>0 and(step<=10 or step%args.train_log_every==0 or stop_after_step is not _A)
		if should_log_train:tl=train_loss.item();gn_str=f" grad_norm:{grad_norm:.4f}"if grad_norm is not _A else'';log0(f"step:{step}/{args.iterations} train_loss:{tl:.4f}{gn_str} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/step:.2f}ms")
		reached_cap=max_wallclock_ms is not _A and approx_training_time_ms>=max_wallclock_ms
		if distributed and max_wallclock_ms is not _A:reached_cap_tensor=torch.tensor(int(reached_cap),device=device);dist.all_reduce(reached_cap_tensor,op=dist.ReduceOp.MAX);reached_cap=bool(reached_cap_tensor.item())
		if stop_after_step is _A and reached_cap:stop_after_step=step
	log0(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB");log0('ema:applying EMA weights');current_state=base_model.state_dict();model_ema={k:v for(k,v)in ema_state.items()if not k.startswith(B)};avg_state={name:model_ema[name].to(dtype=current_state[name].dtype)for name in current_state};base_model.load_state_dict(avg_state,strict=_B)
	if feedback is not _A:fb_ema={k.removeprefix(B):v for(k,v)in ema_state.items()if k.startswith(B)};fb_state=feedback.state_dict();fb_avg={k:fb_ema[k].to(dtype=fb_state[k].dtype)for k in fb_state};feedback.load_state_dict(fb_avg,strict=_B)
	torch.cuda.synchronize();t_diag=time.perf_counter();diag_val_loss,diag_val_bpb=eval_val(args,compiled_model,rank,world_size,device,grad_accum_steps,val_tokens,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut);torch.cuda.synchronize();log0(f"DIAGNOSTIC post_ema val_loss:{diag_val_loss:.4f} val_bpb:{diag_val_bpb:.4f} eval_time:{1e3*(time.perf_counter()-t_diag):.0f}ms");full_state_dict=base_model.state_dict();export_sd=full_state_dict
	if feedback is not _A:
		for(k,v)in feedback.state_dict().items():export_sd[f"_feedback.{k}"]=v
	if master_process:torch.save(export_sd,F);model_bytes=os.path.getsize(F);code_bytes=len(code.encode(_J));log0(f"Serialized model: {model_bytes} bytes");log0(f"Code size: {code_bytes} bytes")
	eval_num_passes=args.eval_passes if args.eval_passes>0 else args.num_passes
	if eval_num_passes!=args.num_passes:
		log0(f"eval_override: num_passes {args.num_passes} -> {eval_num_passes}");base_model.num_passes=eval_num_passes
		if base_model.residual_scale is not _A:old_s=base_model.residual_scale.scales.data;new_s=torch.full((eval_num_passes,),cli.residual_scale_init,dtype=torch.float32,device=old_s.device);copy_len=min(eval_num_passes,old_s.shape[0]);new_s[:copy_len]=old_s[:copy_len];base_model.residual_scale.scales=nn.Parameter(new_s)
		export_sd=base_model.state_dict()
		if feedback is not _A:
			for(k,v)in feedback.state_dict().items():export_sd[f"_feedback.{k}"]=v
	sd_cpu={k:v.detach().cpu()for(k,v)in export_sd.items()};unbanked_sd=_unbank_state_dict(sd_cpu,args.num_layers);quant_result,quant_meta=mixed_quantize_int6(unbanked_sd,{'mlp','attn'});quant_buf=io.BytesIO();torch.save({'w':quant_result,'m':quant_meta},quant_buf);quant_raw=quant_buf.getvalue();quant_blob=lzma.compress(quant_raw,preset=6)
	if master_process:
		with open(G,'wb')as f:f.write(quant_blob)
		quant_file_bytes=len(quant_blob);code_bytes=len(code.encode(_J));log0(f"Serialized model int6+lzma: {quant_file_bytes} bytes");log0(f"Total submission size int6+lzma: {quant_file_bytes+code_bytes} bytes")
	if distributed:dist.barrier()
	with open(G,'rb')as f:quant_blob_disk=f.read()
	quant_state=torch.load(io.BytesIO(lzma.decompress(quant_blob_disk)),map_location='cpu');deq_unbanked=dequantize_mixed_int6(quant_state['w'],quant_state['m'],unbanked_sd);deq_state=_rebank_state_dict(deq_unbanked,args.num_layers,sd_cpu);eval_feedback=_A;eval_feedback_fn=_A;fb_keys={k:v for(k,v)in deq_state.items()if k.startswith(C)}
	if fb_keys:
		deq_state={k:v for(k,v)in deq_state.items()if not k.startswith(C)};eval_feedback=ErrorFeedbackModule(dim=args.model_dim,rank=cli.feedback_rank,feedback_mode=cli.feedback_mode,per_pass=cli.per_pass_feedback,num_passes=eval_num_passes).to(device).bfloat16();fb_sd={k.removeprefix(C):v for(k,v)in fb_keys.items()};eval_feedback.load_state_dict(fb_sd,strict=_B)
		def eval_feedback_fn(h,pass_idx):return eval_feedback(h,pass_idx)
		log0(f"eval_feedback: loaded from artifact, params={eval_feedback.param_count()}")
	eval_model=_make_gpt(args,cli,eval_num_passes).to(device).bfloat16()
	if residual_scale is not _A:eval_rs=ResidualScale(eval_num_passes,cli.residual_scale_init).to(device);eval_model.residual_scale=eval_rs
	_promote_fp32(eval_model);eval_model.load_state_dict(deq_state,strict=_B)
	if args.ttt_enabled:torch.cuda.synchronize();t_ttt=time.perf_counter();ttt_loss,ttt_bpb=eval_val_sliding_ttt(args,eval_model,rank,world_size,device,val_tokens,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut,stride=args.eval_stride,log0=log0,feedback_fn=eval_feedback_fn,feedback_module=eval_feedback);torch.cuda.synchronize();log0(f"legal_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} eval_time:{1e3*(time.perf_counter()-t_ttt):.0f}ms");log0(f"legal_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")
	if distributed:dist.destroy_process_group()
if __name__=='__main__':main()