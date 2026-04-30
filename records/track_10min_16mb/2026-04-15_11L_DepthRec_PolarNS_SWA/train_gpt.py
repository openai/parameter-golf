from __future__ import annotations
A4='passthrough_ctrl'
A3='passthrough_orig_dtypes'
A2='dtypes'
y='scales'
x='quantized'
w='per_row'
v='scheme'
u='torch.'
AZ='momentum'
AY='fineweb_train_*.bin'
AX=RuntimeError
t=tuple
s=FileNotFoundError
r=sorted
h='.scale'
e='.q'
AA='cpu'
d=','
A9=print
A8=isinstance
c=getattr
b=enumerate
T='passthrough'
A1='params'
S=str
o='cuda'
n='utf-8'
m='lr'
l=any
g='1'
f=bool
Z=min
V=.0
R=max
Q=len
N=ValueError
M=range
K=1.
B=float
G=False
F=True
E=int
C=None
import copy,glob as i,io,math as U,os as D,random,subprocess as A7,sys,time as Y,uuid,lzma,zlib,zstandard as AV
from pathlib import Path as z
import numpy as O,sentencepiece as Ay,torch as A,torch.distributed as I,torch.nn.functional as J
from torch import Tensor,nn as H
from torch.nn.parallel import DistributedDataParallel as Az
class Hyperparameters:data_path=D.environ.get('DATA_PATH','./data/datasets/fineweb10B_sp4096');train_files=D.path.join(data_path,AY);val_files=D.path.join(data_path,'fineweb_val_*.bin');tokenizer_path=D.environ.get('TOKENIZER_PATH','./data/tokenizers/fineweb_4096_bpe.model');run_id=D.environ.get('RUN_ID',S(uuid.uuid4()));seed=E(D.environ.get('SEED',1337));val_batch_size=E(D.environ.get('VAL_BATCH_SIZE',524288));val_loss_every=E(D.environ.get('VAL_LOSS_EVERY',1000));train_log_every=E(D.environ.get('TRAIN_LOG_EVERY',200));iterations=E(D.environ.get('ITERATIONS',20000));warmdown_iters=E(D.environ.get('WARMDOWN_ITERS',4000));warmup_steps=E(D.environ.get('WARMUP_STEPS',20));train_batch_tokens=E(D.environ.get('TRAIN_BATCH_TOKENS',524288));train_seq_len=E(D.environ.get('TRAIN_SEQ_LEN',2048));max_wallclock_seconds=B(D.environ.get('MAX_WALLCLOCK_SECONDS',6e2));qk_gain_init=B(D.environ.get('QK_GAIN_INIT',5.));vocab_size=E(D.environ.get('VOCAB_SIZE',4096));num_layers=E(D.environ.get('NUM_LAYERS',11));num_kv_heads=E(D.environ.get('NUM_KV_HEADS',4));model_dim=E(D.environ.get('MODEL_DIM',512));num_heads=E(D.environ.get('NUM_HEADS',8));mlp_mult=E(D.environ.get('MLP_MULT',4));tie_embeddings=f(E(D.environ.get('TIE_EMBEDDINGS',g)));rope_base=B(D.environ.get('ROPE_BASE',1e4));logit_softcap=B(D.environ.get('LOGIT_SOFTCAP',3e1));rope_dims=E(D.environ.get('ROPE_DIMS',16));ln_scale=f(E(D.environ.get('LN_SCALE',g)));xsa_last_n=E(D.environ.get('XSA_LAST_N',4));embed_lr=B(D.environ.get('EMBED_LR',.6));head_lr=B(D.environ.get('HEAD_LR',.008));tied_embed_lr=B(D.environ.get('TIED_EMBED_LR',.035));tied_embed_init_std=B(D.environ.get('TIED_EMBED_INIT_STD',.005));matrix_lr=B(D.environ.get('MATRIX_LR',.025));scalar_lr=B(D.environ.get('SCALAR_LR',.025));muon_momentum=B(D.environ.get('MUON_MOMENTUM',.99));muon_backend_steps=E(D.environ.get('MUON_BACKEND_STEPS',5));muon_momentum_warmup_start=B(D.environ.get('MUON_MOMENTUM_WARMUP_START',.92));muon_momentum_warmup_steps=E(D.environ.get('MUON_MOMENTUM_WARMUP_STEPS',1500));beta1=B(D.environ.get('BETA1',.9));beta2=B(D.environ.get('BETA2',.95));adam_eps=B(D.environ.get('ADAM_EPS',1e-08));grad_clip_norm=B(D.environ.get('GRAD_CLIP_NORM',.3));muon_wd=B(D.environ.get('MUON_WD',.04));late_qat_threshold=B(D.environ.get('LATE_QAT_THRESHOLD',.15));eval_stride=E(D.environ.get('EVAL_STRIDE',64));polar_ns=f(E(D.environ.get('POLAR_NS',g)));swa_enabled=f(E(D.environ.get('SWA_ENABLED',g)));swa_threshold=B(D.environ.get('SWA_THRESHOLD',.2));swa_every=E(D.environ.get('SWA_EVERY',50));swa_ema_blend=B(D.environ.get('SWA_EMA_BLEND',.5));recur_pass_scales=f(E(D.environ.get('RECUR_PASS_SCALES',g)));parallel_from=E(D.environ.get('PARALLEL_FROM',7))
W=(8.28721201814563,-23.595886519098837,17.300387312530933),(4.107059111542203,-2.9478499167379106,.5448431082926601),(3.948690853482295,-2.9483904105122316,.5518191394370137),(3.318419657370602,-2.488488025669773,.515072170769246),(2.300652019954817,-1.6689039845747493,.4188073119525673)
def A0(G,steps=5,eps=1e-07):
	E=eps;D=steps;A=G.bfloat16();C=G.size(0)>G.size(1)
	if Hyperparameters.polar_ns:
		K=A.abs();L=K.sum(dim=1).max().clamp(min=E);N=K.sum(dim=0).max().clamp(min=E);A=A/(L*N).sqrt()
		if C:A=A.T
		O=Z(D,Q(W))if D>0 else Q(W)
		for P in M(O):F,H,I=W[P];B=A@A.T;J=H*B+I*B@B;A=F*A+J@A
		return A.T if C else A
	F,H,I=3.4445,-4.775,2.0315;A=A/(A.norm()+E)
	if C:A=A.T
	for R in M(D):B=A@A.T;J=H*B+I*B@B;A=F*A+J@A
	return A.T if C else A
class A_(A.optim.Optimizer):
	def __init__(A,params,lr,momentum,backend_steps,nesterov=F,wd=V):super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,nesterov=nesterov,wd=wd))
	@A.no_grad()
	def step(self,closure=C):
		P=closure;O='momentum_buffer';Q=C
		if P is not C:
			with A.enable_grad():Q=P()
		L=I.is_available()and I.is_initialized();W=I.get_world_size()if L else 1;X=I.get_rank()if L else 0
		for H in self.param_groups:
			J=H[A1]
			if not J:continue
			S=H[m];T=H[AZ];Y=H['backend_steps'];Z=H['nesterov'];U=H['wd']
			if U>0:
				for D in J:
					if D.grad is not C:D.data.mul_(K-S*U)
			a=sum(E(A.numel())for A in J);M=A.zeros(a,device=J[0].device,dtype=A.bfloat16);G=0
			for(c,D)in b(J):
				if c%W==X and D.grad is not C:
					B=D.grad;N=self.state[D]
					if O not in N:N[O]=A.zeros_like(B)
					V=N[O];V.mul_(T).add_(B)
					if Z:B=B.add(V,alpha=T)
					B=B/B.norm(dim=1,keepdim=F).clamp(min=1e-08);B=A0(B,steps=Y);B*=R(1,B.size(0)/B.size(1))**.5;M[G:G+D.numel()]=B.reshape(-1)
				G+=D.numel()
			if L:I.all_reduce(M,op=I.ReduceOp.SUM)
			G=0
			for D in J:B=M[G:G+D.numel()].view_as(D).to(dtype=D.dtype);D.add_(B,alpha=-S);G+=D.numel()
		return Q
def B0(sp,vocab_size,device):
	H=device;C=sp;K=E(C.vocab_size());I=R(K,vocab_size);J=O.zeros((I,),dtype=O.int16);L=O.zeros((I,),dtype=O.bool_);N=O.ones((I,),dtype=O.bool_)
	for B in M(K):
		if C.is_control(B)or C.is_unknown(B)or C.is_unused(B):continue
		N[B]=G
		if C.is_byte(B):J[B]=1;continue
		D=C.id_to_piece(B)
		if D.startswith('▁'):L[B]=F;D=D[1:]
		J[B]=Q(D.encode(n))
	return A.tensor(J,dtype=A.int16,device=H),A.tensor(L,dtype=A.bool,device=H),A.tensor(N,dtype=A.bool,device=H)
def B1(pattern,seq_len):
	C=pattern;B=seq_len;D=[z(A)for A in r(i.glob(C))]
	if not D:raise s(f"No files found for pattern: {C}")
	E=A.cat([X(A)for A in D]).contiguous();F=(E.numel()-1)//B*B
	if F<=0:raise N(f"Validation split is too short for TRAIN_SEQ_LEN={B}")
	return E[:F+1]
def B2(args,model,rank,world_size,device,grad_accum_steps,val_tokens,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut):
	P=val_tokens;O=grad_accum_steps;H=model;E=device;D=world_size;C=args;Q=C.val_batch_size//(D*O)
	if Q<C.train_seq_len:raise N(f"VAL_BATCH_SIZE must provide at least one sequence per rank; got VAL_BATCH_SIZE={C.val_batch_size}, WORLD_SIZE={D}, GRAD_ACCUM_STEPS={O}, TRAIN_SEQ_LEN={C.train_seq_len}")
	R=Q//C.train_seq_len;S=(P.numel()-1)//C.train_seq_len;d=S*rank//D;T=S*(rank+1)//D;J=A.zeros((),device=E,dtype=A.float64);G=A.zeros((),device=E,dtype=A.float64);K=A.zeros((),device=E,dtype=A.float64);H.eval()
	with A.inference_mode():
		for V in M(d,T,R):
			e=Z(V+R,T);f=V*C.train_seq_len;g=e*C.train_seq_len+1;W=P[f:g].to(device=E,dtype=A.int64,non_blocking=F);X=W[:-1].reshape(-1,C.train_seq_len);L=W[1:].reshape(-1,C.train_seq_len)
			with A.autocast(device_type=o,dtype=A.bfloat16,enabled=F):h=H(X,L).detach()
			Y=B(L.numel());J+=h.to(A.float64)*Y;G+=Y;i=X.reshape(-1);a=L.reshape(-1);b=base_bytes_lut[a].to(dtype=A.int16);b+=(has_leading_space_lut[a]&~is_boundary_token_lut[i]).to(dtype=A.int16);K+=b.to(A.float64).sum()
	if I.is_available()and I.is_initialized():I.all_reduce(J,op=I.ReduceOp.SUM);I.all_reduce(G,op=I.ReduceOp.SUM);I.all_reduce(K,op=I.ReduceOp.SUM)
	c=J/G;j=c.item()/U.log(2.);k=G.item()/K.item();H.train();return B(c.item()),B(j*k)
k=t(A for A in D.environ.get('CONTROL_TENSOR_NAME_PATTERNS','attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights').split(d)if A)
A5=t(A for A in D.environ.get('INT8_KEEP_FLOAT_FP32_NAME_PATTERNS',d.join(k)).split(d)if A)
A6=65536
AB=A.float16
AC=A.float16
AD=99.99984
j=AD/1e2
def L(t):return E(t.numel())*E(t.element_size())
def AE(name,t,passthrough_orig_dtypes):
	if l(A in name for A in A5):return t.float().contiguous()
	if t.dtype in{A.float32,A.bfloat16}:passthrough_orig_dtypes[name]=S(t.dtype).removeprefix(u);return t.to(dtype=AB).contiguous()
	return t
def p(t):
	D=t.float()
	if D.ndim==2:E=A.quantile(D.abs(),j,dim=1)if D.numel()else A.empty((D.shape[0],),dtype=A.float32);H=A.maximum(A.minimum(D,E[:,C]),-E[:,C]);F=(E/127.).clamp_min(K/127.);G=A.clamp(A.round(H/F[:,C]),-127,127).to(A.int8).contiguous();return G,F.to(dtype=AC).contiguous()
	E=B(A.quantile(D.abs().flatten(),j).item())if D.numel()else V;F=A.tensor(E/127. if E>0 else K,dtype=A.float32);G=A.clamp(A.round(A.clamp(D,-E,E)/F),-127,127).to(A.int8).contiguous();return G,F
def AO(state_dict):
	W='baseline_tensor_bytes';V='num_nonfloat_tensors';U='num_float_tensors';R='num_tensors';Q='param_count';D='int8_payload_bytes';K={};M={};N={};F={};G={};H={};A=dict.fromkeys((Q,R,U,V,W,D),0)
	for(C,X)in state_dict.items():
		B=X.detach().to(AA).contiguous();A[Q]+=E(B.numel());A[R]+=1;A[W]+=L(B)
		if not B.is_floating_point():A[V]+=1;F[C]=B;A[D]+=L(B);continue
		if B.numel()<=A6:O=AE(C,B,G);F[C]=O;A[D]+=L(O);continue
		A[U]+=1;P,I=p(B)
		if I.ndim>0:H[C]={v:w,'axis':0}
		K[C]=P;M[C]=I;N[C]=S(B.dtype).removeprefix(u);A[D]+=L(P)+L(I)
	J={'__quant_format__':'int8_clean_per_row_v1',x:K,y:M,A2:N,T:F}
	if H:J['qmeta']=H
	if G:J[A3]=G
	return J,A
def AP(obj):
	D=obj;F={};K=D.get('qmeta',{});L=D.get(A3,{})
	for(C,G)in D[x].items():
		I=c(A,D[A2][C]);E=D[y][C]
		if K.get(C,{}).get(v)==w or E.ndim>0:E=E.to(dtype=A.float32);F[C]=(G.float()*E.view(G.shape[0],*[1]*(G.ndim-1))).to(dtype=I).contiguous()
		else:M=B(E.item());F[C]=(G.float()*M).to(dtype=I).contiguous()
	for(C,N)in D[T].items():
		H=N.detach().to(AA).contiguous();J=L.get(C)
		if A8(J,S):H=H.to(dtype=c(A,J)).contiguous()
		F[C]=H
	return F
def AF(name):
	A=name
	if'tok_emb'in A or'lm_head'in A:return'embed'
	if'.mlp.'in A:return'mlp'
	if'.attn.'in A or'.c_q.'in A or'.c_k.'in A or'.c_v.'in A or'.proj.'in A:return'attn'
	return'other'
def AG(t,clip_range=31):
	D=clip_range;E=t.float()
	if E.ndim==2:
		H,I,J=C,C,B('inf')
		for L in[.999,.9995,.9999,.99999,K]:
			if L<K:M=A.quantile(E.abs(),L,dim=1)
			else:M=E.abs().amax(dim=1)
			G=(M/D).clamp_min(K/D).to(A.float16);F=A.clamp(A.round(E/G.float()[:,C]),-D,D).to(A.int8);Q=F.float()*G.float()[:,C];N=(E-Q).pow(2).mean().item()
			if N<J:H,I,J=F,G,N
		return H,I
	O=E.abs().max().item();P=A.tensor(O/D if O>0 else K,dtype=A.float16);F=A.clamp(A.round(E/P.float()),-D,D).to(A.int8);return F,P
def B3(state_dict,int6_cats):
	H='type';D={};E={}
	for(B,I)in state_dict.items():
		C=I.detach().cpu().contiguous();J=AF(B)
		if not C.is_floating_point()or C.numel()<=65536:D[B]=C.to(A.float16)if C.is_floating_point()else C;E[B]=T;continue
		if l(A in B for A in k):D[B]=C.float();E[B]=A4;continue
		if J in int6_cats and C.ndim>=1:F,G=AG(C);D[B+e]=F;D[B+h]=G;E[B]={H:'int6'}
		else:F,G=p(C);D[B+e]=F;D[B+h]=G;E[B]={H:'int8'}
	return D,E
def B4(result,meta,template_sd):
	I=result;E={}
	for(D,L)in template_sd.items():
		K=meta.get(D)
		if K is C:continue
		F=L.dtype
		if K in(T,A4):
			G=I[D]
			if G.dtype==A.float16 and F in(A.float32,A.bfloat16):G=G.to(F)
			E[D]=G;continue
		H,J=I[D+e],I[D+h]
		if J.ndim>0:E[D]=(H.float()*J.float().view(H.shape[0],*[1]*(H.ndim-1))).to(F)
		else:E[D]=(H.float()*B(J.item())).to(F)
	return E
def B5(args,base_model,rank,world_size,device,val_tokens,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut,stride,batch_seqs=32):
	c=batch_seqs;a=stride;Y=val_tokens;X=world_size;L=base_model;E=device;F=args.train_seq_len;N=Y.numel()-1;d=[A for A in M(0,N,a)if Z(A+F,N)-A>=1];e=Q(d);p=e*rank//X;q=e*(rank+1)//X;f=d[p:q];O=A.zeros((),device=E,dtype=A.float64);H=A.zeros((),device=E,dtype=A.float64);P=A.zeros((),device=E,dtype=A.float64);L.eval()
	with A.inference_mode():
		for g in M(0,Q(f),c):
			S=f[g:g+c];T=Q(S);V=A.zeros(T,F,dtype=A.int64,device=E);W=A.zeros(T,F,dtype=A.int64,device=E);h=[]
			for(D,G)in b(S):i=Z(G+F,N);C=i-G;h.append(C);j=Y[G:i+1].to(dtype=A.int64,device=E);V[D,:C]=j[:-1];W[D,:C]=j[1:]
			with A.autocast(device_type=o,dtype=A.bfloat16):k=L.forward_logits(V)
			r=J.cross_entropy(k.reshape(-1,k.size(-1)).float(),W.reshape(-1),reduction='none').reshape(T,F)
			for(D,G)in b(S):C=h[D];K=0 if G==0 else R(C-a,0);s=r[D,K:C].to(A.float64);O+=s.sum();H+=B(C-K);l=W[D,K:C];t=V[D,K:C];m=base_bytes_lut[l].to(A.float64);m+=(has_leading_space_lut[l]&~is_boundary_token_lut[t]).to(A.float64);P+=m.sum()
	if I.is_available()and I.is_initialized():I.all_reduce(O,op=I.ReduceOp.SUM);I.all_reduce(H,op=I.ReduceOp.SUM);I.all_reduce(P,op=I.ReduceOp.SUM)
	n=(O/H).item();u=n/U.log(2.);v=H.item()/P.item();L.train();return n,u*v
def X(file):
	K='<u2';J='<i4';B=file;F=256*O.dtype(J).itemsize;L=O.dtype(K).itemsize;C=O.fromfile(B,dtype=J,count=256)
	if C.size!=256 or E(C[0])!=20240520 or E(C[1])!=1:raise N(f"Unexpected shard header for {B}")
	D=E(C[2]);H=F+D*L
	if B.stat().st_size!=H:raise N(f"Shard size mismatch for {B}: expected {H} bytes")
	I=O.fromfile(B,dtype=K,count=D,offset=F)
	if I.size!=D:raise N(f"Short read for {B}")
	return A.from_numpy(I.astype(O.uint16,copy=G))
class AH:
	def __init__(A,pattern):
		B=pattern;A.files=[z(A)for A in r(i.glob(B))]
		if not A.files:raise s(f"No files found for pattern: {B}")
		A.file_idx=0;A.tokens=X(A.files[0]);A.pos=0
	def _advance_file(A):A.file_idx=(A.file_idx+1)%Q(A.files);A.tokens=X(A.files[A.file_idx]);A.pos=0
	def take(B,n):
		C=[];D=n
		while D>0:
			F=B.tokens.numel()-B.pos
			if F<=0:B._advance_file();continue
			E=Z(D,F);C.append(B.tokens[B.pos:B.pos+E]);B.pos+=E;D-=E
		return C[0]if Q(C)==1 else A.cat(C)
class AW:
	def __init__(A,pattern,rank,world_size,device):A.rank=rank;A.world_size=world_size;A.device=device;A.stream=AH(pattern)
	def next_batch(B,global_tokens,seq_len,grad_accum_steps):D=seq_len;H=global_tokens//(B.world_size*grad_accum_steps);C=H+1;I=B.stream.take(C*B.world_size);E=B.rank*C;G=I[E:E+C].to(dtype=A.int64);J=G[:-1].reshape(-1,D);K=G[1:].reshape(-1,D);return J.to(B.device,non_blocking=F),K.to(B.device,non_blocking=F)
class a(H.Module):
	def __init__(A,eps=C):super().__init__();A.eps=eps
	def forward(A,x):return J.rms_norm(x,(x.size(-1),),eps=A.eps)
class P(H.Linear):
	_qat_enabled=G
	def forward(B,x):
		D=B.weight.to(x.dtype)
		if P._qat_enabled and B.training and D.ndim==2:
			with A.no_grad():E=B.weight.float();G=E.abs().amax(dim=1);F=(G/31.).clamp_min(K/31.);H=(A.clamp(A.round(E/F[:,C]),-32,31)*F[:,C]).to(x.dtype)
			D=D+(H-D).detach()
		I=B.bias.to(x.dtype)if B.bias is not C else C;return J.linear(x,D,I)
def B6(module):
	with A.no_grad():
		for(C,B)in module.named_parameters():
			if(B.ndim<2 or l(A in C for A in k))and B.dtype!=A.float32:B.data=B.data.float()
class AI(H.Module):
	def __init__(B,dim,base=1e4,rope_dims=0):E=rope_dims;super().__init__();D=E if E>0 else dim;B.rope_dims=D;F=K/base**(A.arange(0,D,2,dtype=A.float32)/D);B.register_buffer('inv_freq',F,persistent=G);B._seq_len_cached=0;B._cos_cached=C;B._sin_cached=C
	def forward(B,seq_len,device,dtype):
		F=dtype;E=device;D=seq_len
		if B._cos_cached is C or B._sin_cached is C or B._seq_len_cached!=D or B._cos_cached.device!=E:H=A.arange(D,device=E,dtype=B.inv_freq.dtype);G=A.outer(H,B.inv_freq.to(E));B._cos_cached=G.cos()[C,C,:,:];B._sin_cached=G.sin()[C,C,:,:];B._seq_len_cached=D
		return B._cos_cached.to(dtype=F),B._sin_cached.to(dtype=F)
def q(x,cos,sin):
	F=sin;B=cos;G=B.size(-1)*2
	if G<x.size(-1):H,I=x[...,:G],x[...,G:];C=G//2;D,E=H[...,:C],H[...,C:];J=A.cat((D*B+E*F,D*-F+E*B),dim=-1);return A.cat((J,I),dim=-1)
	C=x.size(-1)//2;D,E=x[...,:C],x[...,C:];return A.cat((D*B+E*F,D*-F+E*B),dim=-1)
class AJ(H.Module):
	def __init__(B,dim,num_heads,num_kv_heads,rope_base,qk_gain_init,rope_dims=0):
		E=num_kv_heads;D=num_heads;C=dim;super().__init__()
		if C%D!=0:raise N('model_dim must be divisible by num_heads')
		if D%E!=0:raise N('num_heads must be divisible by num_kv_heads')
		B.num_heads=D;B.num_kv_heads=E;B.head_dim=C//D
		if B.head_dim%2!=0:raise N('head_dim must be even for RoPE')
		I=B.num_kv_heads*B.head_dim;B.c_q=P(C,C,bias=G);B.c_k=P(C,I,bias=G);B.c_v=P(C,I,bias=G);B.proj=P(C,C,bias=G);B.proj._zero_init=F;B.q_gain=H.Parameter(A.full((D,),qk_gain_init,dtype=A.float32));B.rotary=AI(B.head_dim,base=rope_base,rope_dims=rope_dims);B.use_xsa=G
	def _xsa_efficient(L,y,v):A,B,C,D=y.shape;E=v.size(-2);I=C//E;G=y.reshape(A,B,E,I,D);H=J.normalize(v,dim=-1).unsqueeze(-2);K=(G*H).sum(dim=-1,keepdim=F)*H;return(G-K).reshape(A,B,C,D)
	def forward(A,x):
		H,G,N=x.shape;B=A.c_q(x).reshape(H,G,A.num_heads,A.head_dim).transpose(1,2);D=A.c_k(x).reshape(H,G,A.num_kv_heads,A.head_dim).transpose(1,2);I=A.c_v(x).reshape(H,G,A.num_kv_heads,A.head_dim).transpose(1,2);B=J.rms_norm(B,(B.size(-1),));D=J.rms_norm(D,(D.size(-1),));K,L=A.rotary(G,x.device,B.dtype);B=q(B,K,L);D=q(D,K,L);B=B*A.q_gain.to(dtype=B.dtype)[C,:,C,C]
		if A.num_kv_heads!=A.num_heads:M=A.num_heads//A.num_kv_heads;D=D.repeat_interleave(M,dim=1);I=I.repeat_interleave(M,dim=1)
		E=J.scaled_dot_product_attention(B,D,I,attn_mask=C,is_causal=F);E=E.transpose(1,2).contiguous().reshape(H,G,A.num_heads,A.head_dim)
		if A.use_xsa:O=I.transpose(1,2).contiguous();E=A._xsa_efficient(E,O)
		E=E.reshape(H,G,N);return A.proj(E)
class AK(H.Module):
	def __init__(A,dim,mlp_mult):B=dim;super().__init__();C=mlp_mult*B;A.fc=P(B,C,bias=G);A.proj=P(C,B,bias=G);A.proj._zero_init=F
	def forward(A,x):x=J.leaky_relu(A.fc(x),negative_slope=.5);return A.proj(x.square())
class AL(H.Module):
	def __init__(B,dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init,rope_dims=0,layer_idx=0,ln_scale=G,parallel=G):C=dim;super().__init__();B.attn_norm=a();B.mlp_norm=a();B.attn=AJ(C,num_heads,num_kv_heads,rope_base,qk_gain_init,rope_dims=rope_dims);B.mlp=AK(C,mlp_mult);B.attn_scale=H.Parameter(A.ones(C,dtype=A.float32));B.mlp_scale=H.Parameter(A.ones(C,dtype=A.float32));B.resid_mix=H.Parameter(A.stack((A.ones(C),A.zeros(C))).float());B.ln_scale_factor=K/U.sqrt(layer_idx+1)if ln_scale else K;B.parallel=parallel
	def forward(A,x,x0):
		E=A.resid_mix.to(dtype=x.dtype);x=E[0][C,C,:]*x+E[1][C,C,:]*x0;B=A.ln_scale_factor
		if A.parallel:F=A.attn_norm(x)*B;D=A.attn(F);G=A.mlp(F);x=x+A.attn_scale.to(dtype=x.dtype)[C,C,:]*D+A.mlp_scale.to(dtype=x.dtype)[C,C,:]*G;return x
		D=A.attn(A.attn_norm(x)*B);x=x+A.attn_scale.to(dtype=x.dtype)[C,C,:]*D;x=x+A.mlp_scale.to(dtype=x.dtype)[C,C,:]*A.mlp(A.mlp_norm(x)*B);return x
class AM(H.Module):
	def __init__(B,dim):super().__init__();B.gate=H.Parameter(A.zeros(dim,dtype=A.float32))
	def forward(D,x):B=A.sigmoid(D.gate.to(dtype=x.dtype))[C,C,:];E=A.cat([A.zeros_like(x[:,:1]),x[:,:-1]],dim=1);return(1-B)*x+B*E
class AN(H.Module):
	def __init__(B,bigram_vocab_size,bigram_dim,model_dim):
		F=model_dim;E=bigram_vocab_size;D=bigram_dim;super().__init__();B.bigram_vocab_size=E;B.embed=H.Embedding(E,D);H.init.zeros_(B.embed.weight);B.proj=P(D,F,bias=G)if D!=F else C
		if B.proj is not C:H.init.zeros_(B.proj.weight)
		B.scale=H.Parameter(A.tensor(.05,dtype=A.float32))
	def bigram_hash(E,tokens):B=tokens.to(A.int32);D=E.bigram_vocab_size-1;C=A.empty_like(B,dtype=A.long);C[...,0]=D;C[...,1:]=(A.bitwise_xor(36313*B[...,1:],27191*B[...,:-1])%D).long();return C
	def forward(A,token_ids):
		B=A.embed(A.bigram_hash(token_ids))
		if A.proj is not C:B=A.proj(B)
		return B*A.scale.to(dtype=B.dtype)
class B7(H.Module):
	def __init__(B,vocab_size,num_layers,model_dim,num_heads,num_kv_heads,mlp_mult,tie_embeddings,tied_embed_init_std,logit_softcap,rope_base,qk_gain_init,rope_dims=0,ln_scale=G,xsa_last_n=0):
		L=xsa_last_n;K=tie_embeddings;J=vocab_size;I=logit_softcap;E=model_dim;D=num_layers;super().__init__()
		if I<=V:raise N(f"logit_softcap must be positive, got {I}")
		B.tie_embeddings=K;B.tied_embed_init_std=tied_embed_init_std;B.logit_softcap=I;B.tok_emb=H.Embedding(J,E);B.bigram=AN(3072,112,E);B.smear=AM(E);B.num_encoder_layers=D//2;B.num_decoder_layers=D-B.num_encoder_layers;B.num_skip_weights=Z(B.num_encoder_layers,B.num_decoder_layers);B.skip_weights=H.Parameter(A.ones(B.num_skip_weights,E,dtype=A.float32));O=Hyperparameters.parallel_from;B.blocks=H.ModuleList([AL(E,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init,rope_dims=rope_dims,layer_idx=A,ln_scale=ln_scale,parallel=O>=0 and A>=O)for A in M(D)]);B.blocks[4].mlp=B.blocks[3].mlp;B.blocks[5].mlp=B.blocks[3].mlp;B.recur_layer_idxs={4,5}
		if Hyperparameters.recur_pass_scales:B.recur_pass_scales=H.Parameter(A.ones(D,dtype=A.float32))
		else:B.recur_pass_scales=C
		if L>0:
			for Q in M(R(0,D-L),D):B.blocks[Q].attn.use_xsa=F
		B.final_norm=a();B.lm_head=C if K else P(E,J,bias=G)
		if B.lm_head is not C:B.lm_head._zero_init=F
		B._init_weights()
	def _init_weights(A):
		if A.tie_embeddings:H.init.normal_(A.tok_emb.weight,mean=V,std=A.tied_embed_init_std)
		for B in A.modules():
			if A8(B,H.Linear)and c(B,'_zero_init',G):H.init.zeros_(B.weight)
	def forward(D,input_ids,target_ids):
		H=input_ids;B=D.tok_emb(H);B=B+D.bigram(H);B=J.rms_norm(B,(B.size(-1),));B=D.smear(B);I=B;F=[]
		for E in M(D.num_encoder_layers):
			B=D.blocks[E](B,I)
			if D.recur_pass_scales is not C and E in D.recur_layer_idxs:B=B*D.recur_pass_scales[E].to(dtype=B.dtype)
			F.append(B)
		for E in M(D.num_decoder_layers):
			if F:B=B+D.skip_weights[E].to(dtype=B.dtype)[C,C,:]*F.pop()
			G=D.num_encoder_layers+E;B=D.blocks[G](B,I)
			if D.recur_pass_scales is not C and G in D.recur_layer_idxs:B=B*D.recur_pass_scales[G].to(dtype=B.dtype)
		B=D.final_norm(B).reshape(-1,B.size(-1));L=target_ids.reshape(-1)
		if D.tie_embeddings:K=J.linear(B,D.tok_emb.weight)
		else:
			if D.lm_head is C:raise AX('lm_head is required when tie_embeddings=False')
			K=D.lm_head(B)
		N=D.logit_softcap*A.tanh(K/D.logit_softcap);return J.cross_entropy(N.float(),L,reduction='mean')
	def forward_logits(D,input_ids):
		H=input_ids;B=D.tok_emb(H);B=B+D.bigram(H);B=J.rms_norm(B,(B.size(-1),));B=D.smear(B);I=B;F=[]
		for E in M(D.num_encoder_layers):
			B=D.blocks[E](B,I)
			if D.recur_pass_scales is not C and E in D.recur_layer_idxs:B=B*D.recur_pass_scales[E].to(dtype=B.dtype)
			F.append(B)
		for E in M(D.num_decoder_layers):
			if F:B=B+D.skip_weights[E].to(dtype=B.dtype)[C,C,:]*F.pop()
			G=D.num_encoder_layers+E;B=D.blocks[G](B,I)
			if D.recur_pass_scales is not C and G in D.recur_layer_idxs:B=B*D.recur_pass_scales[G].to(dtype=B.dtype)
		B=D.final_norm(B)
		if D.tie_embeddings:K=J.linear(B,D.tok_emb.weight)
		else:K=D.lm_head(B)
		return D.logit_softcap*A.tanh(K/D.logit_softcap)
def main():
	Ax='final_model.pt';Aw='WORLD_SIZE';AU='final_model.int6.ptz';y='m';x='w';w='base_lr';global A0;AB=z(__file__).read_text(encoding=n);B=Hyperparameters();A0=A.compile(A0);W='RANK'in D.environ and Aw in D.environ;p=E(D.environ.get('RANK','0'));S=E(D.environ.get(Aw,g));Aa=E(D.environ.get('LOCAL_RANK','0'))
	if S<=0:raise N(f"WORLD_SIZE must be positive, got {S}")
	if 8%S!=0:raise N(f"WORLD_SIZE={S} must divide 8 so grad_accum_steps stays integral")
	T=8//S;Ab=K/T
	if not A.cuda.is_available():raise AX('CUDA is required')
	U=A.device(o,Aa);A.cuda.set_device(U)
	if W:I.init_process_group(backend='nccl',device_id=U);I.barrier()
	A2=p==0;A.backends.cuda.matmul.allow_tf32=F;A.backends.cudnn.allow_tf32=F;from torch.backends.cuda import enable_cudnn_sdp as B8,enable_flash_sdp as B9,enable_math_sdp as BA,enable_mem_efficient_sdp as BB;B8(G);B9(F);BB(G);BA(G);A3=C
	if A2:D.makedirs('logs',exist_ok=F);A3=f"logs/{B.run_id}.txt";A9(A3)
	def H(msg,console=F):
		if not A2:return
		if console:A9(msg)
		if A3 is not C:
			with open(A3,'a',encoding=n)as A:A9(msg,file=A)
	H(AB,console=G);H('='*100,console=G);H(f"Running Python {sys.version}",console=G);H(f"Running PyTorch {A.__version__}",console=G);H(A7.run(['nvidia-smi'],stdout=A7.PIPE,stderr=A7.PIPE,text=F,check=G).stdout,console=G);H('='*100,console=G);random.seed(B.seed);O.random.seed(B.seed);A.manual_seed(B.seed);A.cuda.manual_seed_all(B.seed)
	if not B.tokenizer_path.endswith('.model'):raise N(f"Script only setup for SentencePiece .model file: {B.tokenizer_path}")
	AC=Ay.SentencePieceProcessor(model_file=B.tokenizer_path)
	if E(AC.vocab_size())!=B.vocab_size:raise N(f"VOCAB_SIZE={B.vocab_size} does not match tokenizer vocab_size={E(AC.vocab_size())}")
	Ac=z(B.data_path).resolve();BC=Q(list(Ac.glob(AY)));AD=B1(B.val_files,B.train_seq_len);Ad,Ae,Af=B0(AC,B.vocab_size,U);H(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={B.tokenizer_path}");H(f"train_loader:dataset:{Ac.name} train_shards:{BC}");H(f"val_loader:shards pattern={B.val_files} tokens:{AD.numel()-1}");J=B7(vocab_size=B.vocab_size,num_layers=B.num_layers,model_dim=B.model_dim,num_heads=B.num_heads,num_kv_heads=B.num_kv_heads,mlp_mult=B.mlp_mult,tie_embeddings=B.tie_embeddings,tied_embed_init_std=B.tied_embed_init_std,logit_softcap=B.logit_softcap,rope_base=B.rope_base,qk_gain_init=B.qk_gain_init,rope_dims=B.rope_dims,ln_scale=B.ln_scale,xsa_last_n=B.xsa_last_n).to(U).bfloat16()
	for Ag in J.modules():
		if A8(Ag,P):Ag.float()
	B6(J);Ah=A.compile(J,dynamic=G,fullgraph=F);a=Az(Ah,device_ids=[Aa],broadcast_buffers=G)if W else Ah;Ai=list(J.blocks.named_parameters());BD=[A for(B,A)in Ai if A.ndim==2 and not l(A in B for A in k)];AE=[A for(B,A)in Ai if A.ndim<2 or l(A in B for A in k)]
	if J.skip_weights.numel()>0:AE.append(J.skip_weights)
	if J.recur_pass_scales is not C:AE.append(J.recur_pass_scales)
	AF=B.tied_embed_lr if B.tie_embeddings else B.embed_lr;BE=A.optim.Adam([{A1:[J.tok_emb.weight],m:AF,w:AF}],betas=(B.beta1,B.beta2),eps=B.adam_eps,fused=F);AG=A_(BD,lr=B.matrix_lr,momentum=B.muon_momentum,backend_steps=B.muon_backend_steps,wd=B.muon_wd)
	for h in AG.param_groups:h[w]=B.matrix_lr
	BF=A.optim.Adam([{A1:AE,m:B.scalar_lr,w:B.scalar_lr}],betas=(B.beta1,B.beta2),eps=B.adam_eps,fused=F);b=[BE,AG,BF]
	if J.lm_head is not C:BG=A.optim.Adam([{A1:[J.lm_head.weight],m:B.head_lr,w:B.head_lr}],betas=(B.beta1,B.beta2),eps=B.adam_eps,fused=F);b.insert(1,BG)
	BH=sum(A.numel()for A in J.parameters());H(f"model_params:{BH}");H(f"world_size:{S} grad_accum_steps:{T}");H('sdp_backends:cudnn=False flash=True mem_efficient=False math=False');H(f"attention_mode:gqa num_heads:{B.num_heads} num_kv_heads:{B.num_kv_heads}");H(f"tie_embeddings:{B.tie_embeddings} embed_lr:{AF} head_lr:{B.head_lr if J.lm_head is not C else V} matrix_lr:{B.matrix_lr} scalar_lr:{B.scalar_lr}");H(f"train_batch_tokens:{B.train_batch_tokens} train_seq_len:{B.train_seq_len} iterations:{B.iterations} warmup_steps:{B.warmup_steps} max_wallclock_seconds:{B.max_wallclock_seconds:.3f}");H(f"seed:{B.seed}");AH=AW(B.train_files,p,S,U)
	def q():
		for A in b:A.zero_grad(set_to_none=F)
	r=1e3*B.max_wallclock_seconds if B.max_wallclock_seconds>0 else C
	def BI(step,elapsed_ms):
		D=elapsed_ms;A=step
		if B.warmdown_iters<=0:return K
		if r is C:G=R(B.iterations-B.warmdown_iters,0);return R((B.iterations-A)/R(B.warmdown_iters,1),V)if G<=A<B.iterations else K
		H=D/R(A,1);E=B.warmdown_iters*H;F=R(r-D,V);return F/R(E,1e-09)if F<=E else K
	if B.warmup_steps>0:
		BJ={A:B.detach().cpu().clone()for(A,B)in J.state_dict().items()};BK=[copy.deepcopy(A.state_dict())for A in b];a.train()
		for AI in M(B.warmup_steps):
			q()
			for AJ in M(T):
				if W:a.require_backward_grad_sync=AJ==T-1
				AK,AL=AH.next_batch(B.train_batch_tokens,B.train_seq_len,T)
				with A.autocast(device_type=o,dtype=A.bfloat16,enabled=F):BL=a(AK,AL)
				(BL*Ab).backward()
			for c in b:c.step()
			q()
			if B.warmup_steps<=20 or(AI+1)%10==0 or AI+1==B.warmup_steps:H(f"warmup_step:{AI+1}/{B.warmup_steps}")
		J.load_state_dict(BJ,strict=F)
		for(c,BM)in zip(b,BK,strict=F):c.load_state_dict(BM)
		q()
		if W:a.require_backward_grad_sync=F
		AH=AW(B.train_files,p,S,U)
	Aj=.997;A4={A:B.detach().float().clone()for(A,B)in J.state_dict().items()};s=C;d=0;i=V;j=C;A.cuda.synchronize();A5=Y.perf_counter();L=0
	while F:
		Ak=L==B.iterations or j is not C and L>=j;BN=Ak or B.val_loss_every>0 and L%B.val_loss_every==0
		if BN:A.cuda.synchronize();i+=1e3*(Y.perf_counter()-A5);BO,BP=B2(B,a,p,S,U,T,AD,Ad,Ae,Af);H(f"step:{L}/{B.iterations} val_loss:{BO:.4f} val_bpb:{BP:.4f} train_time:{i:.0f}ms step_avg:{i/R(L,1):.2f}ms");A.cuda.synchronize();A5=Y.perf_counter()
		if Ak:
			if j is not C and L<B.iterations:H(f"stopping_early: wallclock_cap train_time:{i:.0f}ms step:{L}/{B.iterations}")
			break
		BQ=i+1e3*(Y.perf_counter()-A5);t=BI(L,BQ);q();AM=A.zeros((),device=U)
		for AJ in M(T):
			if W:a.require_backward_grad_sync=AJ==T-1
			AK,AL=AH.next_batch(B.train_batch_tokens,B.train_seq_len,T)
			with A.autocast(device_type=o,dtype=A.bfloat16,enabled=F):Al=a(AK,AL)
			AM+=Al.detach();(Al*Ab).backward()
		AM/=T;Am=Z(L/B.muon_momentum_warmup_steps,K)if B.muon_momentum_warmup_steps>0 else K;BR=(1-Am)*B.muon_momentum_warmup_start+Am*B.muon_momentum
		for h in AG.param_groups:h[AZ]=BR
		for c in b:
			for h in c.param_groups:h[m]=h[w]*t
		if B.late_qat_threshold>0 and t<B.late_qat_threshold and not P._qat_enabled:P._qat_enabled=F;H(f"qat:enabled step:{L} scale:{t:.4f}")
		if B.grad_clip_norm>0:A.nn.utils.clip_grad_norm_(J.parameters(),B.grad_clip_norm)
		for c in b:c.step()
		q()
		with A.no_grad():
			for(AN,AO)in J.state_dict().items():A4[AN].mul_(Aj).add_(AO.detach().float(),alpha=K-Aj)
		if B.swa_enabled and t<B.swa_threshold and L%B.swa_every==0:
			with A.no_grad():
				if s is C:s={A:B.detach().float().clone()for(A,B)in J.state_dict().items()};d=1;H(f"swa:started step:{L} scale:{t:.4f}")
				else:
					d+=1
					for(AN,AO)in J.state_dict().items():s[AN].mul_((d-1)/d).add_(AO.detach().float(),alpha=K/d)
		L+=1;AP=i+1e3*(Y.perf_counter()-A5);BS=B.train_log_every>0 and(L<=10 or L%B.train_log_every==0 or j is not C)
		if BS:H(f"step:{L}/{B.iterations} train_loss:{AM.item():.4f} train_time:{AP:.0f}ms step_avg:{AP/L:.2f}ms")
		AQ=r is not C and AP>=r
		if W and r is not C:An=A.tensor(E(AQ),device=U);I.all_reduce(An,op=I.ReduceOp.MAX);AQ=f(An.item())
		if j is C and AQ:j=L
	H(f"peak memory allocated: {A.cuda.max_memory_allocated()//1024//1024} MiB reserved: {A.cuda.max_memory_reserved()//1024//1024} MiB");Ao=J.state_dict()
	if s is not C and d>0:AR=B.swa_ema_blend;H(f"ema+swa:blending swa_n={d} blend={AR:.2f}");Ap={A:(AR*s[A]+(K-AR)*A4[A]).to(dtype=Ao[A].dtype)for A in A4}
	else:H('ema:applying EMA weights');Ap={A:B.to(dtype=Ao[A].dtype)for(A,B)in A4.items()}
	J.load_state_dict(Ap,strict=F)
	if A2:A.save(J.state_dict(),Ax);BT=D.path.getsize(Ax);AS=Q(AB.encode(n));H(f"raw model: {BT} bytes, code: {AS} bytes")
	BU={A:B.detach().cpu()for(A,B)in J.state_dict().items()};Aq,Ar=B3(J.state_dict(),{'mlp','attn'})
	for u in(4,5):
		for e in('fc','proj'):
			for A6 in('q','s'):Aq.pop(f"blocks.{u}.mlp.{e}.weight.{A6}",C)
			Ar.pop(f"blocks.{u}.mlp.{e}.weight",C)
	As=io.BytesIO();A.save({x:Aq,y:Ar},As);BV=As.getvalue();BW=AV.ZstdCompressor(level=22).compress(BV)
	if A2:
		with open(AU,'wb')as AT:AT.write(BW)
		At=D.path.getsize(AU);AS=Q(AB.encode(n));H(f"int6+zstd: {At} bytes, total: {At+AS} bytes")
	if W:I.barrier()
	with open(AU,'rb')as AT:BX=AT.read()
	X=A.load(io.BytesIO(AV.ZstdDecompressor().decompress(BX)),map_location=AA,weights_only=G)
	for u in(4,5):
		for e in('fc','proj'):
			for A6 in('q','s'):
				v=f"blocks.3.mlp.{e}.weight.{A6}"
				if v in X[x]:X[x][f"blocks.{u}.mlp.{e}.weight.{A6}"]=X[x][v]
			v=f"blocks.3.mlp.{e}.weight"
			if v in X[y]:X[y][f"blocks.{u}.mlp.{e}.weight"]=X[y][v]
	BY=B4(X[x],X[y],BU);J.load_state_dict(BY,strict=F);A.cuda.synchronize();BZ=Y.perf_counter();Au,Av=B5(B,J,p,S,U,AD,Ad,Ae,Af,stride=B.eval_stride);A.cuda.synchronize();H(f"final_int6_lzma_sliding val_loss:{Au:.4f} val_bpb:{Av:.4f} eval_time:{1e3*(Y.perf_counter()-BZ):.0f}ms");H(f"final_int6_lzma_sliding_exact val_loss:{Au:.8f} val_bpb:{Av:.8f}")
	if W:I.destroy_process_group()
if __name__=='__main__':main()