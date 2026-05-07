import collections,copy,glob,io,lzma,math,os,random,re,subprocess,sys,time,uuid,numpy as np,sentencepiece as spm,torch,torch.distributed as dist,torch.nn.functional as F
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import Isomap
from sklearn.preprocessing import normalize
try: from flash_attn_interface import flash_attn_func as flash_attn_3_func
except ImportError: flash_attn_3_func=None
class Hyperparameters:
	data_dir=os.environ.get('DATA_DIR','./data/');seed=int(os.environ.get('SEED',1337));run_id=os.environ.get('RUN_ID',str(uuid.uuid4()));iterations=int(os.environ.get('ITERATIONS',4550));warmdown_frac=float(os.environ.get('WARMDOWN_FRAC',.72));warmup_steps=int(os.environ.get('WARMUP_STEPS',20));train_batch_tokens=int(os.environ.get('TRAIN_BATCH_TOKENS',786432));train_seq_len=int(os.environ.get('TRAIN_SEQ_LEN',2048));train_log_every=int(os.environ.get('TRAIN_LOG_EVERY',500));max_wallclock_seconds=float(os.environ.get('MAX_WALLCLOCK_SECONDS',6e2));val_batch_tokens=int(os.environ.get('VAL_BATCH_TOKENS',524288));eval_seq_len=int(os.environ.get('EVAL_SEQ_LEN',2048));val_loss_every=int(os.environ.get('VAL_LOSS_EVERY',4000));sliding_window_enabled=bool(int(os.environ.get('SLIDING_WINDOW_ENABLED','1')));vocab_size=int(os.environ.get('VOCAB_SIZE',8192));num_layers=int(os.environ.get('NUM_LAYERS',11));xsa_last_n=int(os.environ.get('XSA_LAST_N',11));model_dim=int(os.environ.get('MODEL_DIM',512));num_kv_heads=int(os.environ.get('NUM_KV_HEADS',4));num_heads=int(os.environ.get('NUM_HEADS',8));mlp_mult=float(os.environ.get('MLP_MULT',4.));skip_gates_enabled=bool(int(os.environ.get('SKIP_GATES_ENABLED','1')));tie_embeddings=bool(int(os.environ.get('TIE_EMBEDDINGS','1')));logit_softcap=float(os.environ.get('LOGIT_SOFTCAP',3e1));rope_base=float(os.environ.get('ROPE_BASE',1e4));rope_dims=int(os.environ.get('ROPE_DIMS',16));ln_scale=bool(int(os.environ.get('LN_SCALE','1')));qk_gain_init=float(os.environ.get('QK_GAIN_INIT',5.25));num_loops=int(os.environ.get('NUM_LOOPS',2));loop_start=int(os.environ.get('LOOP_START',3));loop_end=int(os.environ.get('LOOP_END',5));enable_looping_at=float(os.environ.get('ENABLE_LOOPING_AT',.35));parallel_residual_start=int(os.environ.get('PARALLEL_RESIDUAL_START',7));min_lr=float(os.environ.get('MIN_LR',.0));embed_lr=float(os.environ.get('EMBED_LR',.6));head_lr=float(os.environ.get('HEAD_LR',.008));tied_embed_lr=float(os.environ.get('TIED_EMBED_LR',.03));tied_embed_init_std=float(os.environ.get('TIED_EMBED_INIT_STD',.005));matrix_lr=float(os.environ.get('MATRIX_LR',.022));scalar_lr=float(os.environ.get('SCALAR_LR',.02));muon_momentum=float(os.environ.get('MUON_MOMENTUM',.99));muon_backend_steps=int(os.environ.get('MUON_BACKEND_STEPS',5));muon_momentum_warmup_start=float(os.environ.get('MUON_MOMENTUM_WARMUP_START',.92));muon_momentum_warmup_steps=int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS',1500));muon_row_normalize=bool(int(os.environ.get('MUON_ROW_NORMALIZE','1')));beta1=float(os.environ.get('BETA1',.9));beta2=float(os.environ.get('BETA2',.95));adam_eps=float(os.environ.get('ADAM_EPS',1e-08));grad_clip_norm=float(os.environ.get('GRAD_CLIP_NORM',.3));eval_stride=int(os.environ.get('EVAL_STRIDE',64));muon_beta2=float(os.environ.get('MUON_BETA2',.95));adam_wd=float(os.environ.get('ADAM_WD',.02));muon_wd=float(os.environ.get('MUON_WD',.095));embed_wd=float(os.environ.get('EMBED_WD',.085));ema_decay=float(os.environ.get('EMA_DECAY',.9965));ttt_enabled=bool(int(os.environ.get('TTT_ENABLED','0')));ttt_lr=float(os.environ.get('TTT_LR',.005));ttt_epochs=int(os.environ.get('TTT_EPOCHS',3));ttt_momentum=float(os.environ.get('TTT_MOMENTUM',.9));ttt_chunk_tokens=int(os.environ.get('TTT_CHUNK_TOKENS',32768));compressor=os.environ.get('COMPRESSOR','lzma');gptq_calibration_batches=int(os.environ.get('GPTQ_CALIBRATION_BATCHES',64));gptq_reserve_seconds=float(os.environ.get('GPTQ_RESERVE_SECONDS',15.));matrix_bits=int(os.environ.get('MATRIX_BITS',6));embed_bits=int(os.environ.get('EMBED_BITS',8));matrix_clip_sigmas=float(os.environ.get('MATRIX_CLIP_SIGMAS',12.85));embed_clip_sigmas=float(os.environ.get('EMBED_CLIP_SIGMAS',2e1));bigram_size=int(os.environ.get('BIGRAM_SIZE',4096));geo_top_ngrams=int(os.environ.get('GEO_TOP_NGRAMS',8000));geo_embed_dim=int(os.environ.get('GEO_EMBED_DIM',50));geo_isomap_nbrs=int(os.environ.get('GEO_ISOMAP_NBRS',15));geo_cooc_window=int(os.environ.get('GEO_COOC_WINDOW',3));distributed='RANK'in os.environ and'WORLD_SIZE'in os.environ;rank=int(os.environ.get('RANK','0'));world_size=int(os.environ.get('WORLD_SIZE','1'));local_rank=int(os.environ.get('LOCAL_RANK','0'));is_main_process=rank==0;grad_accum_steps=8//world_size;datasets_dir=os.path.join(data_dir,'datasets','fineweb10B_sp1024');train_files=os.path.join(datasets_dir,'fineweb_train_*.bin');val_files=os.path.join(datasets_dir,'fineweb_val_*.bin');base_tokenizer_path=os.path.join(data_dir,'tokenizers','fineweb_1024_bpe.model');logfile=f"logs/{run_id}.txt";model_path='final_model.pt';quantized_model_path='final_model.ptz'
_logger_hparams=None
def set_logging_hparams(h):global _logger_hparams;_logger_hparams=h
def log(msg,console=True):
	if _logger_hparams is None:print(msg);return
	if _logger_hparams.is_main_process:
		if console:print(msg)
		if _logger_hparams.logfile is not None:
			with open(_logger_hparams.logfile,'a',encoding='utf-8')as f:print(msg,file=f)
def load_data_shard(file):
	h_bytes=256*np.dtype('<i4').itemsize;header=np.fromfile(file,dtype='<i4',count=256)
	if header.size!=256 or int(header[0])!=20240520 or int(header[1])!=1:raise ValueError(f"Header fail {file}")
	n=int(header[2]);tokens_np=np.fromfile(file,dtype='<u2',count=n,offset=h_bytes)
	return torch.from_numpy(tokens_np.astype(np.uint16,copy=False))
def build_geodesic_tokenizer(h):
	sp=spm.SentencePieceProcessor(model_file=h.base_tokenizer_path);train_shard_files=sorted(glob.glob(h.train_files))
	raw_sp_tokens=load_data_shard(Path(train_shard_files[0]))[:2_000_000].tolist();parts=[]
	for i in range(0,len(raw_sp_tokens),50_000):parts.append(sp.decode(raw_sp_tokens[i:i+50_000]))
	raw_text="".join(parts);freq=collections.Counter();n_min,n_max=2,6;text_lim=2_000_000
	for n in range(n_min,n_max+1):
		for j in range(min(len(raw_text),text_lim)-n+1):
			ng=raw_text[j:j+n]
			if ng.strip():freq[ng]+=1
	v_ngrams=[ng for ng,_ in sorted(freq.items(),key=lambda x:-x[1])[:h.geo_top_ngrams]];ng2idx={ng:i for i,ng in enumerate(v_ngrams)};V=len(v_ngrams)
	cooc=np.zeros((V,V),dtype=np.float32);ts=raw_text[:1_000_000]
	for n in range(n_min,n_max+1):
		seq=[ng2idx[ts[i:i+n]]for i in range(len(ts)-n+1)if ts[i:i+n]in ng2idx]
		for i,c in enumerate(seq):
			for j in range(max(0,i-h.geo_cooc_window),min(len(seq),i+h.geo_cooc_window+1)):
				if i!=j:cooc[c,seq[j]]+=1
	row_sums=cooc.sum(1,keepdims=True)+1e-9;col_sums=cooc.sum(0,keepdims=True)+1e-9;ppmi=np.maximum(np.log((cooc*cooc.sum())/(row_sums*col_sums)+1e-9),0)
	manifold_coords=Isomap(n_neighbors=h.geo_isomap_nbrs,n_components=h.geo_embed_dim,metric="cosine",n_jobs=-1).fit_transform(normalize(ppmi,norm="l2"))
	labels=AgglomerativeClustering(n_clusters=h.vocab_size,metric="euclidean",linkage="ward").fit_predict(manifold_coords)
	lut=np.ones(h.vocab_size,dtype=np.float32);cluster_bytes=collections.defaultdict(list)
	for ng_idx,cid in enumerate(labels):cluster_bytes[int(cid)].append(len(v_ngrams[ng_idx].encode("utf-8")))
	for cid,blens in cluster_bytes.items():lut[cid]=np.mean(blens)
	for c in range(256):
		tid=h.vocab_size-256+c
		if 0<=tid<h.vocab_size:lut[tid]=1.
	return sp,ng2idx,labels,lut
def geo_encode_text(text,ng2idx,labels,h):
	ids=[];i=0;n_min,n_max=2,6
	while i<len(text):
		m=False
		for n in range(min(n_max,len(text)-i),n_min-1,-1):
			ng=text[i:i+n]
			if ng in ng2idx:ids.append(int(labels[ng2idx[ng]]));i+=n;m=True;break
		if not m:ids.append(h.vocab_size-256+ord(text[i])%256);i+=1
	return ids
class GeodesicSequenceLoader:
	def __init__(self,h,sp,ng2idx,labels,device):
		self.h=h;self.sp=sp;self.ng2idx=ng2idx;self.labels=labels;self.device=device;self.files=sorted(glob.glob(h.train_files))[h.rank::h.world_size]
		self.idx=-1;self.pos=0;self.ids=torch.empty(0,dtype=torch.uint16)
	def next_batch(self,global_tokens,grad_accum_steps):
		nt=global_tokens//(self.h.world_size*grad_accum_steps)+1
		if self.pos+nt>len(self.ids):self.idx=(self.idx+1)%len(self.files);t=self.sp.decode(load_data_shard(Path(self.files[self.idx])).tolist());ids=geo_encode_text(t,self.ng2idx,self.labels,self.h);self.ids=torch.tensor(ids,dtype=torch.uint16);self.pos=0
		batch=self.ids[self.pos:self.pos+nt].to(dtype=torch.int64);self.pos+=nt-1;return batch[:-1].reshape(-1,self.h.train_seq_len).to(self.device),batch[1:].reshape(-1,self.h.train_seq_len).to(self.device)
class ValidationData:
	def __init__(self,h,sp,ng2idx,labels,byte_lut,device):
		self.h=h;self.device=device;ids=[];val_files=sorted(glob.glob(h.val_files))
		for f in val_files:t=sp.decode(load_data_shard(Path(f)).tolist());si=geo_encode_text(t,ng2idx,labels,h);ids.append(torch.tensor(si,dtype=torch.uint16))
		self.val_tokens=torch.cat(ids).contiguous();self.base_bytes_lut=torch.tensor(byte_lut,dtype=torch.float32,device=device)
class RMSNorm(nn.Module):
	def __init__(self,dim,eps=1e-6):super().__init__();self.scale=nn.Parameter(torch.ones(dim));self.eps=eps
	def forward(self,x):return F.rms_norm(x,(x.size(-1),),self.scale,self.eps)
class CastedLinear(nn.Linear):
	def forward(self,x):return F.linear(x,self.weight.to(x.dtype),self.bias.to(x.dtype) if self.bias is not None else None)
class Rotary(nn.Module):
	def __init__(self,dim,base=1e4,train_seq_len=2048,rope_dims=0):
		super().__init__();self.dim=dim;self.base=base;self.train_seq_len=train_seq_len;self.rope_dims=rope_dims if rope_dims>0 else dim;inv_freq=1./base**(torch.arange(0,self.rope_dims,2,dtype=torch.float32)/self.rope_dims);self.register_buffer('inv_freq',inv_freq,persistent=False);self._cos_cached=None
	def forward(self,seq_len,device,dtype):
		if self._cos_cached is None or self._cos_cached.shape[1]!=seq_len:
			rd=self.rope_dims;scale=max(1,seq_len/self.train_seq_len);new_base=self.base*scale**(rd/(rd-2))if seq_len>self.train_seq_len else self.base;inv_freq=1./new_base**(torch.arange(0,rd,2,dtype=torch.float32,device=device)/rd);t=torch.arange(seq_len,device=device,dtype=inv_freq.dtype);freqs=torch.outer(t,inv_freq);self._cos_cached=freqs.cos()[None,:,None,:];self._sin_cached=freqs.sin()[None,:,None,:]
		return self._cos_cached.to(dtype=dtype),self._sin_cached.to(dtype=dtype)
def apply_rotary_emb(x,cos,sin,rope_dims=0):
	if rope_dims>0 and rope_dims<x.size(-1):xr,xp=x[...,:rope_dims],x[...,rope_dims:];h=rope_dims//2;x1,x2=xr[...,:h],xr[...,h:];xr=torch.cat((x1*cos+x2*sin,x1*-sin+x2*cos),dim=-1);return torch.cat((xr,xp),dim=-1)
	h=x.size(-1)//2;x1,x2=x[...,:h],x[...,h:];return torch.cat((x1*cos+x2*sin,x1*-sin+x2*cos),dim=-1)
class CausalSelfAttention(nn.Module):
	def __init__(self,dim,num_heads,num_kv_heads,rope_base,qk_gain_init,train_seq_len):
		super().__init__();self.num_heads=num_heads;self.num_kv_heads=num_kv_heads;self.head_dim=dim//num_heads;self.c_q=CastedLinear(dim,dim,bias=False);self.c_k=CastedLinear(dim,num_kv_heads*self.head_dim,bias=False);self.c_v=CastedLinear(dim,num_kv_heads*self.head_dim,bias=False);self.proj=CastedLinear(dim,dim,bias=False);self.proj._zero_init=True;self.q_gain=nn.Parameter(torch.full((num_heads,),qk_gain_init,dtype=torch.float32));self.rotary=Rotary(self.head_dim,base=rope_base,train_seq_len=train_seq_len);self.use_xsa=False
	def forward(self,x):
		B,T,D=x.shape;q=self.c_q(x).reshape(B,T,self.num_heads,self.head_dim);k=self.c_k(x).reshape(B,T,self.num_kv_heads,self.head_dim);v=self.c_v(x).reshape(B,T,self.num_kv_heads,self.head_dim);q=F.rms_norm(q,(q.size(-1),));k=F.rms_norm(k,(k.size(-1),));cos,sin=self.rotary(T,x.device,q.dtype);q=apply_rotary_emb(q,cos,sin);k=apply_rotary_emb(k,cos,sin);q=q*self.q_gain.to(q.dtype)[None,None,:,None]
		if flash_attn_3_func:y=flash_attn_3_func(q,k,v,causal=True)
		else:y=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=True).transpose(1,2)
		if self.use_xsa:vn=F.normalize(v,dim=-1);y=y-((y*vn.unsqueeze(-2)).sum(-1,keepdim=True)*vn.unsqueeze(-2)).reshape(B,T,self.num_heads,self.head_dim)
		return self.proj(y.reshape(B,T,D))
class MLP(nn.Module):
	def __init__(self,dim,mlp_mult):super().__init__();hidden=int(mlp_mult*dim);self.fc=CastedLinear(dim,hidden,bias=False);self.proj=CastedLinear(hidden,dim,bias=False);self.proj._zero_init=True
	def forward(self,x):return self.proj(F.leaky_relu(self.fc(x),negative_slope=0.5).square())
class Block(nn.Module):
	def __init__(self,dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init,train_seq_len,layer_idx,h):
		super().__init__();self.attn_norm=RMSNorm(dim);self.mlp_norm=RMSNorm(dim);self.attn=CausalSelfAttention(dim,num_heads,num_kv_heads,rope_base,qk_gain_init,train_seq_len);self.mlp=MLP(dim,mlp_mult);self.attn_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.mlp_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.resid_mix=nn.Parameter(torch.stack((torch.ones(dim),torch.zeros(dim))).float());self.ln_sf=1./math.sqrt(layer_idx+1)if h.ln_scale else 1.;self.parallel=False
	def forward(self,x,x0):
		mix=self.resid_mix.to(x.dtype);x_in=mix[0][None,None,:]*x+mix[1][None,None,:]*x0
		if self.parallel:a=self.attn(self.attn_norm(x_in)*self.ln_sf);m=self.mlp(self.mlp_norm(x_in)*self.ln_sf);return x_in+self.attn_scale.to(x.dtype)[None,None,:]*a+self.mlp_scale.to(x.dtype)[None,None,:]*m
		x=x_in+self.attn_scale.to(x.dtype)[None,None,:]*self.attn(self.attn_norm(x_in)*self.ln_sf);return x+self.mlp_scale.to(x.dtype)[None,None,:]*self.mlp(self.mlp_norm(x)*self.ln_sf)
class GPT(nn.Module):
	def __init__(self,h):
		super().__init__();self.h=h;self.tok_emb=nn.Embedding(h.vocab_size,h.embedding_dim);self.bigram_table=nn.Embedding(h.bigram_size,h.model_dim);self.bigram_gate=nn.Linear(h.model_dim,h.model_dim,bias=False);nn.init.normal_(self.bigram_table.weight,std=.01);nn.init.zeros_(self.bigram_gate.weight)
		if h.embedding_dim!=h.model_dim:self.embed_proj=CastedLinear(h.embedding_dim,h.model_dim,bias=False);self.head_proj=CastedLinear(h.model_dim,h.embedding_dim,bias=False)
		else:self.embed_proj=self.head_proj=None
		self.blocks=nn.ModuleList([Block(h.model_dim,h.num_heads,h.num_kv_heads,h.mlp_mult,h.rope_base,h.qk_gain_init,h.train_seq_len,i,h)for i in range(h.num_layers)])
		if h.rope_dims>0:
			for b in self.blocks:b.attn.rope_dims=h.rope_dims;b.attn.rotary.rope_dims=h.rope_dims
		self.final_norm=RMSNorm(h.model_dim if self.head_proj is None else h.embedding_dim);self.lm_head=None if h.tie_embeddings else CastedLinear(h.embedding_dim,h.vocab_size,bias=False)
		if h.xsa_last_n>0:
			for i in range(max(0,h.num_layers-h.xsa_last_n),h.num_layers):self.blocks[i].attn.use_xsa=True
		if h.parallel_residual_start>=0:
			for i in range(h.parallel_residual_start,h.num_layers):self.blocks[i].parallel=True
		self.looping_active=False
		if h.num_loops>0:idx=list(range(h.loop_start));seg=list(range(h.loop_start,h.loop_end+1))
		for _ in range(h.num_loops+1):idx.extend(seg)
		idx.extend(range(h.loop_end+1,h.num_layers));self.encoder_indices=idx[:len(idx)//2];self.decoder_indices=idx[len(idx)//2:];self.num_skip_weights=min(len(self.encoder_indices),len(self.decoder_indices));self.skip_weights=nn.Parameter(torch.ones(self.num_skip_weights,h.model_dim));self.skip_gates=nn.Parameter(torch.zeros(self.num_skip_weights,h.model_dim))if h.skip_gates_enabled else None;self._init_weights()
	def _init_weights(self):
		if self.h.tie_embeddings:nn.init.normal_(self.tok_emb.weight,std=self.h.tied_embed_init_std)
		for _,m in self.named_modules():
			if isinstance(m,nn.Linear):
				if getattr(m,'_zero_init',False):nn.init.zeros_(m.weight)
				elif m.weight.ndim==2 and m.weight.shape[0]>=64:nn.init.orthogonal_(m.weight)
	def forward_logits(self,ids):
		tok=self.tok_emb(ids);prev=torch.roll(ids,1,dims=1);bigram=(ids*1315423911+prev)%self.h.bigram_size;x=tok+torch.sigmoid(self.bigram_gate(tok))*self.bigram_table(bigram);x=F.rms_norm(x,(x.size(-1),))
		if self.embed_proj:x=self.embed_proj(x)
		x0=x;skips=[];enc=self.encoder_indices if self.looping_active else range(len(self.blocks)//2);dec=self.decoder_indices if self.looping_active else range(len(self.blocks)//2,len(self.blocks))
		for i in enc:x=self.blocks[i](x,x0);skips.append(x)
		for idx,i in enumerate(dec):
			if idx<self.num_skip_weights and skips:
				s=self.skip_weights[idx].to(x.dtype)[None,None,:]*skips.pop()
				if self.skip_gates is not None:x=torch.lerp(s,x,torch.sigmoid(self.skip_gates[idx].to(x.dtype))[None,None,:])
				else:x=x+s
			x=self.blocks[i](x,x0)
		x=self.final_norm(x)
		if self.head_proj:x=self.head_proj(x)
		l_proj=F.linear(x,self.tok_emb.weight)if self.h.tie_embeddings else self.lm_head(x);return self.h.logit_softcap*torch.tanh(l_proj/self.h.logit_softcap)
	def forward(self,x,y):logits=self.forward_logits(x);return F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),y.reshape(-1),reduction='mean')
@torch.compile
def zeropower_via_newtonschulz5(G,steps=10,eps=1e-7):
	a,b,c=3.4445,-4.775,2.0315;X=G.bfloat16();X/=X.norm()+eps;t=G.size(0)>G.size(1)
	if t:X=X.T
	for _ in range(steps):A=X@X.T;B=b*A+c*A@A;X=a*X+B@X
	return X.T if t else X
class Muon(torch.optim.Optimizer):
	def __init__(self,params,lr,momentum,backend_steps,row_normalize=False,weight_decay=0.0):super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,row_normalize=row_normalize,weight_decay=weight_decay))
	@torch.no_grad()
	def step(self):
		dist_init=dist.is_available()and dist.is_initialized();ws=dist.get_world_size()if dist_init else 1;rank=dist.get_rank()if dist_init else 0
		for group in self.param_groups:
			params=group['params'];updates_flat=torch.zeros(sum(p.numel()for p in params),device=params[0].device,dtype=torch.bfloat16);curr=0
			for i,p in enumerate(params):
				if i%ws==rank and p.grad is not None:
					g=p.grad;state=self.state[p]
					if'momentum_buffer'not in state:state['momentum_buffer']=torch.zeros_like(g)
					buf=state['momentum_buffer'];buf.mul_(group['momentum']).add_(g);g=g.add(buf,alpha=group['momentum'])
					if group['row_normalize']:g=g/(g.float().norm(dim=-1,keepdim=True).clamp_min(1e-7)).to(g.dtype)
					g=zeropower_via_newtonschulz5(g,steps=group['backend_steps']);g*=max(1,g.size(0)/g.size(1))**.5;updates_flat[curr:curr+p.numel()]=g.reshape(-1)
				curr+=p.numel()
			if dist_init:dist.all_reduce(updates_flat,op=dist.ReduceOp.SUM)
			curr=0
			for p in params:
				if group['weight_decay']>0:p.data.mul_(1.-group['lr']*group['weight_decay'])
				p.add_(updates_flat[curr:curr+p.numel()].view_as(p).to(p.dtype),alpha=-group['lr']);curr+=p.numel()
class Optimizers:
	def __init__(self,h,model):
		P=list(model.blocks.named_parameters());matrix=[p for n,p in P if p.ndim==2 and not any(pat in n for pat in['scale','mix','gain'])];scalar=[p for n,p in P if p.ndim<2 or any(pat in n for pat in['scale','mix','gain'])]
		if model.skip_weights.numel()>0:scalar.append(model.skip_weights)
		if model.skip_gates is not None:scalar.append(model.skip_gates)
		scalar.append(model.bigram_gate.weight);t_lr=h.tied_embed_lr if h.tie_embeddings else h.embed_lr;self.opt_tok=torch.optim.AdamW([{'params':[model.tok_emb.weight],'lr':t_lr,'base_lr':t_lr},{'params':[model.bigram_table.weight],'lr':t_lr,'base_lr':t_lr}],betas=(h.beta1,h.beta2),eps=h.adam_eps,weight_decay=h.embed_wd);self.opt_muon=Muon(matrix,lr=h.matrix_lr,momentum=h.muon_momentum,backend_steps=h.muon_backend_steps,weight_decay=h.muon_wd,row_normalize=h.muon_row_normalize);self.opt_scalar=torch.optim.AdamW([{'params':scalar,'lr':h.scalar_lr,'base_lr':h.scalar_lr}],betas=(h.beta1,h.beta2),eps=h.adam_eps,weight_decay=h.adam_wd);self.optimizers=[self.opt_tok,self.opt_muon,self.opt_scalar]
		if model.lm_head:self.optimizers.append(torch.optim.Adam([{'params':[model.lm_head.weight],'lr':h.head_lr,'base_lr':h.head_lr}],betas=(h.beta1,h.beta2),eps=h.adam_eps))
	def zero_grad(self):
		for o in self.optimizers:o.zero_grad(set_to_none=True)
	def step(self):
		for o in self.optimizers:o.step()
def classify_param(n):
	if'tok_emb'in n:return'embed'
	if'.mlp.'in n:return'mlp'
	if'.attn.'in n or'.proj.'in n:return'attn'
	return'other'
def collect_hessians(model,loader,h,device):
	hessians={};hooks=[]
	def hook(n):
		def fn(m,i,o):
			x=i[0].detach().float()
			if x.ndim==3:x=x.reshape(-1,x.shape[-1])
			if n not in hessians:hessians[n]=torch.zeros(x.shape[1],x.shape[1],device=device)
			hessians[n].addmm_(x.T,x)
		return fn
	for n,m in model.named_modules():
		if isinstance(m,CastedLinear)and m.weight.numel()>65536:
			if classify_param(n+'.weight')in['mlp','attn']:hooks.append(m.register_forward_hook(hook(n+'.weight')))
	model.eval()
	with torch.no_grad():
		for _ in range(h.gptq_calibration_batches):x,y=loader.next_batch(h.train_batch_tokens,h.grad_accum_steps);model.forward_logits(x)
	for hk in hooks:hk.remove()
	return {n:hs.cpu()/h.gptq_calibration_batches for n,hs in hessians.items()}
def gptq_quantize_weight(w,H,clip_sigmas=3.,clip_range=63,block_size=128):
	W=w.float().clone();r,c=W.shape;H=H.float();damp=.01*H.diag().mean();H.diagonal().add_(damp);perm=torch.argsort(H.diag(),descending=True);inv=torch.argsort(perm);Wp=W[:,perm].clone();Hp=H[perm][:,perm];Hi=torch.cholesky_inverse(torch.linalg.cholesky(Hp));Hi=torch.linalg.cholesky(Hi,upper=True);s=(clip_sigmas*W.std(dim=1)/clip_range).clamp_min(1e-10);Q=torch.zeros(r,c,dtype=torch.int8);Ww=Wp.clone()
	for i in range(0,c,block_size):
		i2=min(i+block_size,c);Wb=Ww[:,i:i2].clone();Hib=Hi[i:i2,i:i2];E=torch.zeros(r,i2-i);sf=s.unsqueeze(1)
		for j in range(i2-i):wj=Wb[:,j];d=Hib[j,j];qj=torch.clamp(torch.round(wj/s),-clip_range,clip_range);Q[:,i+j]=qj.to(torch.int8);ej=(wj-qj.float()*s)/d;E[:,j]=ej;Wb[:,j:]-=ej.unsqueeze(1)*Hib[j,j:].unsqueeze(0)
		if i2<c:Ww[:,i2:]-=E@Hi[i:i2,i2:]
	return Q[:,inv],s
def gptq_mixed_quantize(sd,hessians,h):
	res={};meta={}
	for n,t in sd.items():
		if not t.is_floating_point() or t.numel()<=65536:res[n]=t.half();meta[n]='pt';continue
		cs=h.embed_clip_sigmas if'tok_emb'in n else h.matrix_clip_sigmas;bits=h.embed_bits if'tok_emb'in n else h.matrix_bits;q,s=gptq_quantize_weight(t,hessians[n],clip_sigmas=cs,clip_range=2**(bits-1)-1);res[n+'.q']=q;res[n+'.s']=s;meta[n]=f"int{bits}"
	return res,meta
def dequantize(res,meta,sd):
	out={}
	for n,orig in sd.items():
		info=meta.get(n)
		if not info:continue
		if info=='pt':out[n]=res[n].to(orig.dtype);continue
		q,s=res[n+'.q'],res[n+'.s'];out[n]=(q.float()*(s.float().view(q.shape[0],*[1]*(q.ndim-1))if s.ndim>0 else float(s))).to(orig.dtype)
	return out
_BSHF=b'BSHF'
def _c(data):
	b=np.frombuffer(data,dtype=np.uint8);n=len(b);o=np.empty(n,dtype=np.uint8);d=0
	for p in range(2):c=b[p::2];o[d:d+len(c)]=c;d+=len(c)
	return lzma.compress(_BSHF+bytes([2])+o.tobytes())
def _d(data):
	if len(data)<5 or data[:4]!=_BSHF:return lzma.decompress(data)
	s=data[4];p=np.frombuffer(lzma.decompress(data[5:]),dtype=np.uint8);n=len(p);o=np.empty(n,dtype=np.uint8);d=0
	for i in range(s):l=n//s+(1 if i<n%s else 0);o[i::s][:l]=p[d:d+l];d+=l
	return o.tobytes()
def serialize(h,model,sp,ng2idx,labels,lut):
	buf=io.BytesIO();hessians=collect_hessians(model,GeodesicSequenceLoader(h,sp,ng2idx,labels,'cuda'),h,'cuda');q_sd,meta=gptq_mixed_quantize({k:v.cpu()for k,v in model.state_dict().items()},hessians,h);torch.save({'w':q_sd,'m':meta,'v':{'ng2idx':ng2idx,'labels':labels,'lut':lut}},buf);blob=_c(buf.getvalue())
	if h.is_main_process:
		with open(h.quantized_model_path,'wb')as f:f.write(blob)
		log(f"Artifact: {len(blob)} bytes")
def deserialize(h,device):
	with open(h.quantized_model_path,'rb')as f:data=torch.load(io.BytesIO(_d(f.read())),map_location='cpu');v=data['v'];sp=spm.SentencePieceProcessor(model_file=h.base_tokenizer_path);model=GPT(h).to(device).bfloat16();model.load_state_dict(dequantize(data['w'],data['m'],model.state_dict()));return model,sp,v['ng2idx'],v['labels'],v['lut']
def main():
	h=Hyperparameters();set_logging_hparams(h);device=torch.device('cuda',h.local_rank)
	if h.distributed:dist.init_process_group('nccl');torch.cuda.set_device(device)
	sp,ng2idx,labels,lut=build_geodesic_tokenizer(h);val_data=ValidationData(h,sp,ng2idx,labels,lut,device);model=GPT(h).to(device).bfloat16();compiled_model=torch.compile(model);ddp_model=DDP(compiled_model,device_ids=[h.local_rank])if h.distributed else compiled_model;opts=Optimizers(h,model);loader=GeodesicSequenceLoader(h,sp,ng2idx,labels,device);ema_state={k:v.detach().float().clone()for k,v in model.state_dict().items()}
	for step in range(h.iterations):
		if h.num_loops>0 and not model.looping_active and step/h.iterations>=h.enable_looping_at:model.looping_active=True
		opts.zero_grad();frac=step/h.iterations;scale=1. if frac<(1-h.warmdown_frac)else max(0,(1-frac)/h.warmdown_frac)
		for _ in range(h.grad_accum_steps):
			x,y=loader.next_batch(h.train_batch_tokens,h.grad_accum_steps)
			with torch.autocast('cuda',dtype=torch.bfloat16):loss=ddp_model(x,y)
			(loss/h.grad_accum_steps).backward()
		for opt in opts.optimizers:
			for g in opt.param_groups:g['lr']=g['base_lr']*scale
		opts.step()
		with torch.no_grad():
			for k,v in model.state_dict().items():ema_state[k].mul_(h.ema_decay).add_(v.detach().float(),alpha=1-h.ema_decay)
		if step%h.train_log_every==0:log(f"{step} loss {loss.item():.4f}")
	model.load_state_dict({k:v.to(model.tok_emb.weight.dtype)for k,v in ema_state.items()});serialize(h,model,sp,ng2idx,labels,lut)
if __name__=='__main__':main()
