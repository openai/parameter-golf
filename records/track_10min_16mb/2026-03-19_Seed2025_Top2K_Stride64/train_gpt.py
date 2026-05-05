from __future__ import annotations
_i='lm_head is required when tie_embeddings=False'
_h='passthrough_orig_dtypes'
_g='passthrough'
_f='dtypes'
_e='scales'
_d='quantized'
_c='per_row'
_b='per_row_group'
_a='group_size'
_Z='per_row_lowbit'
_Y='SERIAL_COMPRESSOR=zstd requires the zstandard package'
_X='torch.'
_W='momentum'
_V='fineweb_train_*.bin'
_U='tok_emb.weight'
_T='cpu'
_S='baseline_tensor_bytes'
_R='residual_row_delta'
_Q='residual_row_idx'
_P='residual_lowrank_right'
_O='residual_lowrank_left'
_N='zstd'
_M='zlib'
_L='params'
_K='cuda'
_J='utf-8'
_I='lr'
_H='scheme'
_G='int8_payload_bytes'
_F=','
_E=1.
_D=.0
_C=False
_B=True
_A=None
import copy,glob,io,math,os,random,subprocess,sys,time,uuid,zlib
from pathlib import Path
try:import zstandard
except ImportError:zstandard=_A
import numpy as np,sentencepiece as spm,torch,torch.distributed as dist,torch.nn.functional as F
from torch import Tensor,nn
from torch.nn.parallel import DistributedDataParallel as DDP
class Hyperparameters:data_path=os.environ.get('DATA_PATH','./data/datasets/fineweb10B_sp1024');train_files=os.path.join(data_path,_V);val_files=os.path.join(data_path,'fineweb_val_*.bin');tokenizer_path=os.environ.get('TOKENIZER_PATH','./data/tokenizers/fineweb_1024_bpe.model');run_id=os.environ.get('RUN_ID',str(uuid.uuid4()));seed=int(os.environ.get('SEED',1337));val_batch_size=int(os.environ.get('VAL_BATCH_SIZE',524288));val_loss_every=int(os.environ.get('VAL_LOSS_EVERY',1000));train_log_every=int(os.environ.get('TRAIN_LOG_EVERY',200));eval_stride=int(os.environ.get('EVAL_STRIDE',0));iterations=int(os.environ.get('ITERATIONS',20000));warmdown_iters=int(os.environ.get('WARMDOWN_ITERS',1200));warmup_steps=int(os.environ.get('WARMUP_STEPS',20));train_batch_tokens=int(os.environ.get('TRAIN_BATCH_TOKENS',524288));train_seq_len=int(os.environ.get('TRAIN_SEQ_LEN',1024));max_wallclock_seconds=float(os.environ.get('MAX_WALLCLOCK_SECONDS',6e2));qk_gain_init=float(os.environ.get('QK_GAIN_INIT',1.5));tok_emb_qat_mode=os.environ.get('TOK_EMB_QAT_MODE','off');tok_emb_qat_last_steps=int(os.environ.get('TOK_EMB_QAT_LAST_STEPS',0));tok_emb_qat_ramp_steps=int(os.environ.get('TOK_EMB_QAT_RAMP_STEPS',0));tok_emb_qat_penalty=float(os.environ.get('TOK_EMB_QAT_PENALTY',_D));lowbit_qat_penalty=float(os.environ.get('LOWBIT_QAT_PENALTY',_D));curriculum_min_seq_len=int(os.environ.get('CURRICULUM_MIN_SEQ_LEN',0));curriculum_steps=int(os.environ.get('CURRICULUM_STEPS',0));swa_enabled=bool(int(os.environ.get('SWA_ENABLED','0')));swa_start_frac=float(os.environ.get('SWA_START_FRAC',.5));swa_every=int(os.environ.get('SWA_EVERY',200));vocab_size=int(os.environ.get('VOCAB_SIZE',1024));num_layers=int(os.environ.get('NUM_LAYERS',9));num_kv_heads=int(os.environ.get('NUM_KV_HEADS',4));model_dim=int(os.environ.get('MODEL_DIM',512));num_heads=int(os.environ.get('NUM_HEADS',8));mlp_mult=int(os.environ.get('MLP_MULT',2));tie_embeddings=bool(int(os.environ.get('TIE_EMBEDDINGS','1')));rope_base=float(os.environ.get('ROPE_BASE',1e4));logit_softcap=float(os.environ.get('LOGIT_SOFTCAP',3e1));embed_lr=float(os.environ.get('EMBED_LR',.6));head_lr=float(os.environ.get('HEAD_LR',.008));tied_embed_lr=float(os.environ.get('TIED_EMBED_LR',.05));tied_embed_init_std=float(os.environ.get('TIED_EMBED_INIT_STD',.005));matrix_lr=float(os.environ.get('MATRIX_LR',.04));scalar_lr=float(os.environ.get('SCALAR_LR',.04));muon_momentum=float(os.environ.get('MUON_MOMENTUM',.95));muon_beta2=float(os.environ.get('MUON_BETA2',_D));muon_backend_steps=int(os.environ.get('MUON_BACKEND_STEPS',5));muon_momentum_warmup_start=float(os.environ.get('MUON_MOMENTUM_WARMUP_START',.85));muon_momentum_warmup_steps=int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS',500));beta1=float(os.environ.get('BETA1',.9));beta2=float(os.environ.get('BETA2',.95));adam_eps=float(os.environ.get('ADAM_EPS',1e-08));grad_clip_norm=float(os.environ.get('GRAD_CLIP_NORM',_D))
def zeropower_via_newtonschulz5(G,steps=10,eps=1e-07):
	D,E,F=3.4445,-4.775,2.0315;A=G.bfloat16();A/=A.norm()+eps;C=G.size(0)>G.size(1)
	if C:A=A.T
	for I in range(steps):B=A@A.T;H=E*B+F*B@B;A=D*A+H@A
	return A.T if C else A
class Muon(torch.optim.Optimizer):
	def __init__(A,params,lr,momentum,backend_steps,beta2=_D,nesterov=_B):super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,beta2=beta2,nesterov=nesterov))
	@torch.no_grad()
	def step(self,closure=_A):
		N=closure;M='second_momentum_buffer';L='momentum_buffer';O=_A
		if N is not _A:
			with torch.enable_grad():O=N()
		I=dist.is_available()and dist.is_initialized();R=dist.get_world_size()if I else 1;S=dist.get_rank()if I else 0
		for D in self.param_groups:
			F=D[_L]
			if not F:continue
			T=D[_I];H=D[_W];U=D['backend_steps'];J=D['beta2'];P=D['nesterov'];V=sum(int(A.numel())for A in F);K=torch.zeros(V,device=F[0].device,dtype=torch.bfloat16);C=0
			for(W,B)in enumerate(F):
				if W%R==S and B.grad is not _A:
					A=B.grad;E=self.state[B]
					if L not in E:E[L]=torch.zeros_like(A)
					G=E[L]
					if J>0:
						if M not in E:E[M]=torch.zeros_like(A[:,:1],dtype=torch.float32)
						G.lerp_(A,1-H);A=A.lerp(G,H)if P else G
					else:
						G.mul_(H).add_(A)
						if P:A=A.add(G,alpha=H)
					A=zeropower_via_newtonschulz5(A,steps=U)
					if J>0:X=A.norm(dim=(-2,-1),keepdim=_B);Q=E[M];Q.lerp_(A.float().square().mean(dim=-1,keepdim=_B),1-J);A.mul_((Q.sqrt().to(dtype=A.dtype)+1e-10).reciprocal()).mul_(X/A.norm(dim=(-2,-1),keepdim=_B).add_(1e-10))
					A*=max(1,A.size(0)/A.size(1))**.5;K[C:C+B.numel()]=A.reshape(-1)
				C+=B.numel()
			if I:dist.all_reduce(K,op=dist.ReduceOp.SUM)
			C=0
			for B in F:A=K[C:C+B.numel()].view_as(B).to(dtype=B.dtype);B.add_(A,alpha=-T);C+=B.numel()
		return O
def build_sentencepiece_luts(sp,vocab_size,device):
	D=device;B=sp;G=int(B.vocab_size());E=max(G,vocab_size);F=np.zeros((E,),dtype=np.int16);H=np.zeros((E,),dtype=np.bool_);I=np.ones((E,),dtype=np.bool_)
	for A in range(G):
		if B.is_control(A)or B.is_unknown(A)or B.is_unused(A):continue
		I[A]=_C
		if B.is_byte(A):F[A]=1;continue
		C=B.id_to_piece(A)
		if C.startswith('▁'):H[A]=_B;C=C[1:]
		F[A]=len(C.encode(_J))
	return torch.tensor(F,dtype=torch.int16,device=D),torch.tensor(H,dtype=torch.bool,device=D),torch.tensor(I,dtype=torch.bool,device=D)
def load_validation_tokens(pattern,seq_len):
	B=pattern;A=seq_len;C=[Path(A)for A in sorted(glob.glob(B))]
	if not C:raise FileNotFoundError(f"No files found for pattern: {B}")
	D=torch.cat([load_data_shard(A)for A in C]).contiguous();E=(D.numel()-1)//A*A
	if E<=0:raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={A}")
	return D[:E+1]
def eval_val(args,model,rank,world_size,device,grad_accum_steps,val_tokens,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut):
	J=val_tokens;I=grad_accum_steps;E=model;C=device;B=world_size;A=args;K=A.val_batch_size//(B*I)
	if K<A.train_seq_len:raise ValueError(f"VAL_BATCH_SIZE must provide at least one sequence per rank; got VAL_BATCH_SIZE={A.val_batch_size}, WORLD_SIZE={B}, GRAD_ACCUM_STEPS={I}, TRAIN_SEQ_LEN={A.train_seq_len}")
	L=K//A.train_seq_len;M=(J.numel()-1)//A.train_seq_len;V=M*rank//B;N=M*(rank+1)//B;F=torch.zeros((),device=C,dtype=torch.float64);D=torch.zeros((),device=C,dtype=torch.float64);G=torch.zeros((),device=C,dtype=torch.float64);E.eval()
	with torch.inference_mode():
		for O in range(V,N,L):
			W=min(O+L,N);X=O*A.train_seq_len;Y=W*A.train_seq_len+1;P=J[X:Y].to(device=C,dtype=torch.int64,non_blocking=_B);Q=P[:-1].reshape(-1,A.train_seq_len);H=P[1:].reshape(-1,A.train_seq_len)
			with torch.autocast(device_type=_K,dtype=torch.bfloat16,enabled=_B):Z=E(Q,H).detach()
			R=float(H.numel());F+=Z.to(torch.float64)*R;D+=R;a=Q.reshape(-1);S=H.reshape(-1);T=base_bytes_lut[S].to(dtype=torch.int16);T+=(has_leading_space_lut[S]&~is_boundary_token_lut[a]).to(dtype=torch.int16);G+=T.to(torch.float64).sum()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(F,op=dist.ReduceOp.SUM);dist.all_reduce(D,op=dist.ReduceOp.SUM);dist.all_reduce(G,op=dist.ReduceOp.SUM)
	U=F/D;b=U.item()/math.log(2.);c=D.item()/G.item();E.train();return float(U.item()),float(b*c)
def eval_val_sliding_window(args,base_model,rank,world_size,device,val_tokens,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut,stride):
	K=world_size;H=val_tokens;G=base_model;C=device;B=args;A=stride
	if A<=0 or A>B.train_seq_len:raise ValueError(f"EVAL_STRIDE must be in [1, TRAIN_SEQ_LEN], got {A}")
	L=H.numel()-1
	if L<B.train_seq_len:raise ValueError(f"Validation split is too short for sliding-window eval at TRAIN_SEQ_LEN={B.train_seq_len}")
	M=(L-B.train_seq_len)//A+1;V=M*rank//K;W=M*(rank+1)//K;N=int(os.environ.get('SW_EVAL_BATCH',32));I=torch.zeros((),device=C,dtype=torch.float64);D=torch.zeros((),device=C,dtype=torch.float64);J=torch.zeros((),device=C,dtype=torch.float64);G.eval()
	with torch.inference_mode():
		O=list(range(V,W))
		for P in range(0,len(O),N):
			Q=O[P:P+N];R=torch.stack([H[C*A:C*A+B.train_seq_len]for C in Q]).to(device=C,dtype=torch.int64,non_blocking=_B)
			with torch.autocast(device_type=_K,dtype=torch.bfloat16,enabled=_B):S=G.forward_logits(R)
			X=S[:,-A:,:].reshape(-1,S.size(-1));E=torch.stack([H[C*A+B.train_seq_len-A+1:C*A+B.train_seq_len+1]for C in Q]).to(device=C,dtype=torch.int64,non_blocking=_B).reshape(-1);I+=F.cross_entropy(X.float(),E,reduction='sum').to(torch.float64);D+=float(E.numel());Y=R[:,-A:].reshape(-1);T=base_bytes_lut[E].to(dtype=torch.int16);T+=(has_leading_space_lut[E]&~is_boundary_token_lut[Y]).to(dtype=torch.int16);J+=T.to(torch.float64).sum()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(I,op=dist.ReduceOp.SUM);dist.all_reduce(D,op=dist.ReduceOp.SUM);dist.all_reduce(J,op=dist.ReduceOp.SUM)
	U=I/D;Z=U.item()/math.log(2.);a=D.item()/J.item();G.train();return float(U.item()),float(Z*a)
CONTROL_TENSOR_NAME_PATTERNS=tuple(A for A in os.environ.get('CONTROL_TENSOR_NAME_PATTERNS','attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights').split(_F)if A)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS=tuple(A for A in os.environ.get('INT8_KEEP_FLOAT_FP32_NAME_PATTERNS',_F.join(CONTROL_TENSOR_NAME_PATTERNS)).split(_F)if A)
INT8_KEEP_FLOAT_NAME_PATTERNS=tuple(A for A in os.environ.get('INT8_KEEP_FLOAT_NAME_PATTERNS','').split(_F)if A)
def parse_torch_dtype_env(name,default):
	A=os.environ.get(name)
	if A is _A:return default
	B={'float16':torch.float16,'fp16':torch.float16,'bfloat16':torch.bfloat16,'bf16':torch.bfloat16,'float32':torch.float32,'fp32':torch.float32};C=A.strip().lower()
	if C not in B:raise ValueError(f"Unsupported dtype for {name}: {A}")
	return B[C]
INT8_KEEP_FLOAT_MAX_NUMEL=int(os.environ.get('INT8_KEEP_FLOAT_MAX_NUMEL',65536))
INT8_KEEP_FLOAT_STORE_DTYPE=parse_torch_dtype_env('INT8_KEEP_FLOAT_STORE_DTYPE',torch.float16)
INT8_PER_ROW_SCALE_DTYPE=parse_torch_dtype_env('INT8_PER_ROW_SCALE_DTYPE',torch.float16)
INT8_CLIP_PERCENTILE=float(os.environ.get('INT8_CLIP_PERCENTILE',99.99984))
INT8_CLIP_Q=INT8_CLIP_PERCENTILE/1e2
INT8_GROUP_SIZE=int(os.environ.get('INT8_GROUP_SIZE',0))
INT8_GROUP_ALL_2D=bool(int(os.environ.get('INT8_GROUP_ALL_2D','0')))
INT8_GROUP_NAME_PATTERNS=tuple(A for A in os.environ.get('INT8_GROUP_NAME_PATTERNS','').split(_F)if A)
INT8_GROUP_OVERRIDE_SPECS=tuple(A for A in os.environ.get('INT8_GROUP_OVERRIDES','').split(_F)if A)
INT8_COARSEN_OVERRIDE_SPECS=tuple(A for A in os.environ.get('INT8_COARSEN_OVERRIDES','').split(_F)if A)
LOWBIT_BITS=int(os.environ.get('LOWBIT_BITS',0))
LOWBIT_NAME_PATTERNS=tuple(A for A in os.environ.get('LOWBIT_NAME_PATTERNS','').split(_F)if A)
LOWBIT_RESIDUAL_ROWS=int(os.environ.get('LOWBIT_RESIDUAL_ROWS',0))
LOWBIT_RESIDUAL_MODE=os.environ.get('LOWBIT_RESIDUAL_MODE','rows').strip().lower()
LOWBIT_RESIDUAL_NAME_PATTERNS=tuple(A for A in os.environ.get('LOWBIT_RESIDUAL_NAME_PATTERNS',_F.join(LOWBIT_NAME_PATTERNS)).split(_F)if A)
LOWBIT_QAT_NAME_PATTERNS=tuple(A for A in os.environ.get('LOWBIT_QAT_NAME_PATTERNS','').split(_F)if A)
LOWBIT_STE=bool(int(os.environ.get('LOWBIT_STE','0')))
SERIAL_COMPRESSOR=os.environ.get('SERIAL_COMPRESSOR',_M).strip().lower()
def tensor_nbytes(t):return int(t.numel())*int(t.element_size())
def keep_float_tensor(name,t,passthrough_orig_dtypes):
	if any(A in name for A in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):return t.float().contiguous()
	if t.dtype in{torch.float32,torch.bfloat16}:passthrough_orig_dtypes[name]=str(t.dtype).removeprefix(_X);return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
	return t
def group_size_for_tensor(name,t):
	if t.ndim!=2:return 0
	B=t.shape[1]
	for C in INT8_GROUP_OVERRIDE_SPECS:
		D,E,F=C.rpartition(':')
		if not E:raise ValueError(f"Invalid INT8_GROUP_OVERRIDES entry: {C}")
		A=int(F)
		if A>0 and B%A==0 and D and D in name:return A
	if INT8_GROUP_SIZE>0 and B%INT8_GROUP_SIZE==0 and(INT8_GROUP_ALL_2D or any(A in name for A in INT8_GROUP_NAME_PATTERNS)):return INT8_GROUP_SIZE
	return 0
def coarsen_step_for_tensor(name):
	for A in INT8_COARSEN_OVERRIDE_SPECS:
		B,D,E=A.rpartition(':')
		if not D:raise ValueError(f"Invalid INT8_COARSEN_OVERRIDES entry: {A}")
		C=int(E)
		if C>1 and B and B in name:return C
	return 0
def lowbit_bits_for_tensor(name,t):return LOWBIT_BITS if 2<=LOWBIT_BITS<=8 and t.ndim==2 and any(A in name for A in LOWBIT_NAME_PATTERNS)else 0
def residual_rows_for_tensor(name,t):return min(LOWBIT_RESIDUAL_ROWS,t.shape[0])if LOWBIT_RESIDUAL_ROWS>0 and t.ndim==2 and any(A in name for A in LOWBIT_RESIDUAL_NAME_PATTERNS)else 0
def compress_bytes(raw):
	if SERIAL_COMPRESSOR==_N:
		if zstandard is _A:raise RuntimeError(_Y)
		return zstandard.ZstdCompressor(level=22).compress(raw),_N
	if SERIAL_COMPRESSOR==_M:return zlib.compress(raw,level=9),_M
	raise ValueError(f"Unsupported SERIAL_COMPRESSOR: {SERIAL_COMPRESSOR}")
def decompress_bytes(blob):
	if SERIAL_COMPRESSOR==_N:
		if zstandard is _A:raise RuntimeError(_Y)
		return zstandard.ZstdDecompressor().decompress(blob)
	if SERIAL_COMPRESSOR==_M:return zlib.decompress(blob)
	raise ValueError(f"Unsupported SERIAL_COMPRESSOR: {SERIAL_COMPRESSOR}")
def quantize_float_tensor(name,t):
	N='axis';H=name;A=t.float();F=lowbit_bits_for_tensor(H,A)
	if F:
		I=(1<<F-1)-1;S=-(1<<F-1);T=A.abs().amax(dim=1)if A.numel()else torch.empty((A.shape[0],),dtype=torch.float32);B=(T/max(I,1)).clamp_min(_E/max(I,1));D=torch.clamp(torch.round(A/B[:,_A]),S,I).to(torch.int8).contiguous();E={_H:_Z,N:0,'bits':F};J=residual_rows_for_tensor(H,A)
		if J>0:
			U=D.float()*B.float()[:,_A];K=A-U
			if LOWBIT_RESIDUAL_MODE=='svd':L=min(J,*A.shape);V,W,X=torch.linalg.svd(K,full_matrices=_C);E[_O]=(V[:,:L]*W[:L]).to(torch.float16).cpu().contiguous();E[_P]=X[:L].to(torch.float16).cpu().contiguous()
			else:O=torch.topk(K.pow(2).mean(dim=1),k=J,largest=_B,sorted=_C).indices.to(torch.int16);E[_Q]=O.cpu().contiguous();E[_R]=K.index_select(0,O.long()).to(torch.float16).cpu().contiguous()
		return D,B.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous(),E
	G=group_size_for_tensor(H,A)
	if G>0:P,Y=A.shape;Q=Y//G;R=A.view(P,Q,G);C=torch.quantile(R.abs(),INT8_CLIP_Q,dim=2)if A.numel()else torch.empty((P,Q),dtype=torch.float32);M=torch.maximum(torch.minimum(R,C[...,_A]),-C[...,_A]);B=(C/127.).clamp_min(_E/127.);D=torch.clamp(torch.round(M/B[...,_A]),-127,127).to(torch.int8).contiguous().view_as(A);return D,B.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous(),{_H:_b,N:0,_a:G}
	if A.ndim==2:C=torch.quantile(A.abs(),INT8_CLIP_Q,dim=1)if A.numel()else torch.empty((A.shape[0],),dtype=torch.float32);M=torch.maximum(torch.minimum(A,C[:,_A]),-C[:,_A]);B=(C/127.).clamp_min(_E/127.);D=torch.clamp(torch.round(M/B[:,_A]),-127,127).to(torch.int8).contiguous();return D,B.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous(),{_H:_c,N:0}
	C=float(torch.quantile(A.abs().flatten(),INT8_CLIP_Q).item())if A.numel()else _D;B=torch.tensor(C/127. if C>0 else _E,dtype=torch.float32);D=torch.clamp(torch.round(torch.clamp(A,-C,C)/B),-127,127).to(torch.int8).contiguous();return D,B,_A
def dequantize_int8_tensor(q,s,dtype,meta=_A):
	D=dtype;A=meta
	if isinstance(A,dict)and A.get(_H)==_Z:
		B=(q.float()*s.to(dtype=torch.float32).view(q.shape[0],*[1]*(q.ndim-1))).to(dtype=D).contiguous();E=A.get(_Q);H=A.get(_R)
		if isinstance(E,Tensor)and isinstance(H,Tensor)and E.numel():C=E.to(device=B.device,dtype=torch.long);B[C]=B[C]+H.to(device=B.device,dtype=B.dtype)
		F,G=A.get(_O),A.get(_P)
		if isinstance(F,Tensor)and isinstance(G,Tensor)and F.numel()and G.numel():B=B+(F.to(device=B.device,dtype=torch.float32)@G.to(device=B.device,dtype=torch.float32)).to(dtype=B.dtype)
		return B
	if isinstance(A,dict)and A.get(_H)==_b:I=int(A[_a]);C,K=q.shape;J=K//I;return(q.float().view(C,J,I)*s.to(dtype=torch.float32).view(C,J,1)).reshape_as(q).to(dtype=D).contiguous()
	if isinstance(A,dict)and A.get(_H)==_c or s.ndim>0:return(q.float()*s.to(dtype=torch.float32).view(q.shape[0],*[1]*(q.ndim-1))).to(dtype=D).contiguous()
	return(q.float()*float(s.item())).to(dtype=D).contiguous()
def quantize_dequantize_float_tensor(name,t):A,B,C=quantize_float_tensor(name,t.detach());return dequantize_int8_tensor(A,B,t.dtype,C)
def fake_quantize_float_tensor_straight_through(name,t):A=quantize_dequantize_float_tensor(name,t);return t+(A-t).detach()
def quantize_state_dict_int8(state_dict):
	T='num_nonfloat_tensors';S='num_float_tensors';R='num_tensors';Q='param_count';K={};L={};M={};F={};G={};H={};A=dict.fromkeys((Q,R,S,T,_S,_G),0)
	for(B,U)in state_dict.items():
		C=U.detach().to(_T).contiguous();A[Q]+=int(C.numel());A[R]+=1;A[_S]+=tensor_nbytes(C)
		if not C.is_floating_point():A[T]+=1;F[B]=C;A[_G]+=tensor_nbytes(C);continue
		if C.numel()<=INT8_KEEP_FLOAT_MAX_NUMEL or any(A in B for A in INT8_KEEP_FLOAT_NAME_PATTERNS):N=keep_float_tensor(B,C,G);F[B]=N;A[_G]+=tensor_nbytes(N);continue
		A[S]+=1;E,O,D=quantize_float_tensor(B,C);I=coarsen_step_for_tensor(B)
		if I>1:E=((E.float()/I).round()*I).clamp(-127,127).to(torch.int8).contiguous()
		if D is not _A:H[B]=D
		K[B]=E;L[B]=O;M[B]=str(C.dtype).removeprefix(_X);A[_G]+=tensor_nbytes(E)+tensor_nbytes(O)
		if isinstance(D,dict):
			for P in(_Q,_R,_O,_P):
				if isinstance(D.get(P),Tensor):A[_G]+=tensor_nbytes(D[P])
	J={'__quant_format__':'int8_clean_per_row_v1',_d:K,_e:L,_f:M,_g:F}
	if H:J['qmeta']=H
	if G:J[_h]=G
	return J,A
def dequantize_state_dict_int8(obj):
	B=obj;C={};F=B.get('qmeta',{});G=B.get(_h,{})
	for(A,H)in B[_d].items():I=getattr(torch,B[_f][A]);J=B[_e][A];C[A]=dequantize_int8_tensor(H,J,I,F.get(A))
	for(A,K)in B[_g].items():
		D=K.detach().to(_T).contiguous();E=G.get(A)
		if isinstance(E,str):D=D.to(dtype=getattr(torch,E)).contiguous()
		C[A]=D
	return C
def load_data_shard(file):
	H='<u2';G='<i4';A=file;D=256*np.dtype(G).itemsize;I=np.dtype(H).itemsize;B=np.fromfile(A,dtype=G,count=256)
	if B.size!=256 or int(B[0])!=20240520 or int(B[1])!=1:raise ValueError(f"Unexpected shard header for {A}")
	C=int(B[2]);E=D+C*I
	if A.stat().st_size!=E:raise ValueError(f"Shard size mismatch for {A}: expected {E} bytes")
	F=np.fromfile(A,dtype=H,count=C,offset=D)
	if F.size!=C:raise ValueError(f"Short read for {A}")
	return torch.from_numpy(F.astype(np.uint16,copy=_C))
class TokenStream:
	def __init__(A,pattern):
		B=pattern;A.files=[Path(A)for A in sorted(glob.glob(B))]
		if not A.files:raise FileNotFoundError(f"No files found for pattern: {B}")
		A.file_idx=0;A.tokens=load_data_shard(A.files[0]);A.pos=0
	def _advance_file(A):A.file_idx=(A.file_idx+1)%len(A.files);A.tokens=load_data_shard(A.files[A.file_idx]);A.pos=0
	def take(A,n):
		B=[];C=n
		while C>0:
			E=A.tokens.numel()-A.pos
			if E<=0:A._advance_file();continue
			D=min(C,E);B.append(A.tokens[A.pos:A.pos+D]);A.pos+=D;C-=D
		return B[0]if len(B)==1 else torch.cat(B)
class DistributedTokenLoader:
	def __init__(A,pattern,rank,world_size,device):A.rank=rank;A.world_size=world_size;A.device=device;A.stream=TokenStream(pattern)
	def next_batch(A,global_tokens,seq_len,grad_accum_steps):C=seq_len;F=global_tokens//(A.world_size*grad_accum_steps);B=F+1;G=A.stream.take(B*A.world_size);D=A.rank*B;E=G[D:D+B].to(dtype=torch.int64);H=E[:-1].reshape(-1,C);I=E[1:].reshape(-1,C);return H.to(A.device,non_blocking=_B),I.to(A.device,non_blocking=_B)
class RMSNorm(nn.Module):
	def __init__(A,eps=_A):super().__init__();A.eps=eps
	def forward(A,x):return F.rms_norm(x,(x.size(-1),),eps=A.eps)
class CastedLinear(nn.Linear):
	def forward(A,x):
		B=A.weight.to(x.dtype)
		if A.training and LOWBIT_STE and hasattr(A,'quant_name')and lowbit_bits_for_tensor(A.quant_name,A.weight):B=fake_quantize_float_tensor_straight_through(A.quant_name,B)
		C=A.bias.to(x.dtype)if A.bias is not _A else _A;return F.linear(x,B,C)
def restore_low_dim_params_to_fp32(module):
	with torch.no_grad():
		for(B,A)in module.named_parameters():
			if(A.ndim<2 or any(A in B for A in CONTROL_TENSOR_NAME_PATTERNS))and A.dtype!=torch.float32:A.data=A.data.float()
class Rotary(nn.Module):
	def __init__(A,dim,base=1e4):super().__init__();B=_E/base**(torch.arange(0,dim,2,dtype=torch.float32)/dim);A.register_buffer('inv_freq',B,persistent=_C);A._seq_len_cached=0;A._cos_cached=_A;A._sin_cached=_A
	def forward(A,seq_len,device,dtype):
		D=dtype;C=device;B=seq_len
		if A._cos_cached is _A or A._sin_cached is _A or A._seq_len_cached!=B or A._cos_cached.device!=C:F=torch.arange(B,device=C,dtype=A.inv_freq.dtype);E=torch.outer(F,A.inv_freq.to(C));A._cos_cached=E.cos()[_A,_A,:,:];A._sin_cached=E.sin()[_A,_A,:,:];A._seq_len_cached=B
		return A._cos_cached.to(dtype=D),A._sin_cached.to(dtype=D)
def apply_rotary_emb(x,cos,sin):A=x.size(-1)//2;B,C=x[...,:A],x[...,A:];return torch.cat((B*cos+C*sin,B*-sin+C*cos),dim=-1)
class CausalSelfAttention(nn.Module):
	def __init__(A,dim,num_heads,num_kv_heads,rope_base,qk_gain_init):
		D=num_kv_heads;C=num_heads;B=dim;super().__init__()
		if B%C!=0:raise ValueError('model_dim must be divisible by num_heads')
		if C%D!=0:raise ValueError('num_heads must be divisible by num_kv_heads')
		A.num_heads=C;A.num_kv_heads=D;A.head_dim=B//C
		if A.head_dim%2!=0:raise ValueError('head_dim must be even for RoPE')
		E=A.num_kv_heads*A.head_dim;A.c_q=CastedLinear(B,B,bias=_C);A.c_k=CastedLinear(B,E,bias=_C);A.c_v=CastedLinear(B,E,bias=_C);A.proj=CastedLinear(B,B,bias=_C);A.proj._zero_init=_B;A.q_gain=nn.Parameter(torch.full((C,),qk_gain_init,dtype=torch.float32));A.rotary=Rotary(A.head_dim,base=rope_base)
	def forward(A,x):E,D,J=x.shape;B=A.c_q(x).reshape(E,D,A.num_heads,A.head_dim).transpose(1,2);C=A.c_k(x).reshape(E,D,A.num_kv_heads,A.head_dim).transpose(1,2);K=A.c_v(x).reshape(E,D,A.num_kv_heads,A.head_dim).transpose(1,2);B=F.rms_norm(B,(B.size(-1),));C=F.rms_norm(C,(C.size(-1),));H,I=A.rotary(D,x.device,B.dtype);B=apply_rotary_emb(B,H,I);C=apply_rotary_emb(C,H,I);B=B*A.q_gain.to(dtype=B.dtype)[_A,:,_A,_A];G=F.scaled_dot_product_attention(B,C,K,attn_mask=_A,is_causal=_B,enable_gqa=A.num_kv_heads!=A.num_heads);G=G.transpose(1,2).contiguous().reshape(E,D,J);return A.proj(G)
class MLP(nn.Module):
	def __init__(A,dim,mlp_mult):B=dim;super().__init__();C=mlp_mult*B;A.fc=CastedLinear(B,C,bias=_C);A.proj=CastedLinear(C,B,bias=_C);A.proj._zero_init=_B
	def forward(A,x):x=torch.relu(A.fc(x));return A.proj(x.square())
class Block(nn.Module):
	def __init__(A,dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init):B=dim;super().__init__();A.attn_norm=RMSNorm();A.mlp_norm=RMSNorm();A.attn=CausalSelfAttention(B,num_heads,num_kv_heads,rope_base,qk_gain_init);A.mlp=MLP(B,mlp_mult);A.attn_scale=nn.Parameter(torch.ones(B,dtype=torch.float32));A.mlp_scale=nn.Parameter(torch.ones(B,dtype=torch.float32));A.resid_mix=nn.Parameter(torch.stack((torch.ones(B),torch.zeros(B))).float())
	def forward(A,x,x0):B=A.resid_mix.to(dtype=x.dtype);x=B[0][_A,_A,:]*x+B[1][_A,_A,:]*x0;C=A.attn(A.attn_norm(x));x=x+A.attn_scale.to(dtype=x.dtype)[_A,_A,:]*C;x=x+A.mlp_scale.to(dtype=x.dtype)[_A,_A,:]*A.mlp(A.mlp_norm(x));return x
class GPT(nn.Module):
	def __init__(A,vocab_size,num_layers,model_dim,num_heads,num_kv_heads,mlp_mult,tie_embeddings,tok_emb_qat_enabled,tied_embed_init_std,logit_softcap,rope_base,qk_gain_init):
		F=tie_embeddings;E=vocab_size;D=logit_softcap;C=num_layers;B=model_dim;super().__init__()
		if D<=_D:raise ValueError(f"logit_softcap must be positive, got {D}")
		A.tie_embeddings=F;A.tok_emb_qat_enabled=tok_emb_qat_enabled;A.tied_embed_init_std=tied_embed_init_std;A.logit_softcap=D;A.tok_emb=nn.Embedding(E,B);A.register_buffer('tok_emb_qat_alpha',torch.tensor(_D,dtype=torch.float32),persistent=_C);A.num_encoder_layers=C//2;A.num_decoder_layers=C-A.num_encoder_layers;A.num_skip_weights=min(A.num_encoder_layers,A.num_decoder_layers);A.skip_weights=nn.Parameter(torch.ones(A.num_skip_weights,B,dtype=torch.float32));A.blocks=nn.ModuleList([Block(B,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init)for A in range(C)]);A.final_norm=RMSNorm();A.lm_head=_A if F else CastedLinear(B,E,bias=_C)
		if A.lm_head is not _A:A.lm_head._zero_init=_B
		A._init_weights()
	def _init_weights(A):
		if A.tie_embeddings:nn.init.normal_(A.tok_emb.weight,mean=_D,std=A.tied_embed_init_std)
		for B in A.modules():
			if isinstance(B,nn.Linear)and getattr(B,'_zero_init',_C):nn.init.zeros_(B.weight)
	def forward(B,input_ids,target_ids):
		C=B.tok_emb.weight
		if B.tok_emb_qat_enabled:C=torch.lerp(C,fake_quantize_float_tensor_straight_through(_U,C),B.tok_emb_qat_alpha.to(dtype=C.dtype))
		A=F.embedding(input_ids,C);A=F.rms_norm(A,(A.size(-1),));G=A;E=[]
		for D in range(B.num_encoder_layers):A=B.blocks[D](A,G);E.append(A)
		for D in range(B.num_decoder_layers):
			if E:A=A+B.skip_weights[D].to(dtype=A.dtype)[_A,_A,:]*E.pop()
			A=B.blocks[B.num_encoder_layers+D](A,G)
		A=B.final_norm(A).reshape(-1,A.size(-1));I=target_ids.reshape(-1)
		if B.tie_embeddings:H=F.linear(A,C)
		else:
			if B.lm_head is _A:raise RuntimeError(_i)
			H=B.lm_head(A)
		J=B.logit_softcap*torch.tanh(H/B.logit_softcap);return F.cross_entropy(J.float(),I,reduction='mean')
	def forward_logits(B,input_ids):
		C=B.tok_emb.weight
		if B.tok_emb_qat_enabled:C=torch.lerp(C,fake_quantize_float_tensor_straight_through(_U,C),B.tok_emb_qat_alpha.to(dtype=C.dtype))
		A=F.embedding(input_ids,C);A=F.rms_norm(A,(A.size(-1),));G=A;E=[]
		for D in range(B.num_encoder_layers):A=B.blocks[D](A,G);E.append(A)
		for D in range(B.num_decoder_layers):
			if E:A=A+B.skip_weights[D].to(dtype=A.dtype)[_A,_A,:]*E.pop()
			A=B.blocks[B.num_encoder_layers+D](A,G)
		A=B.final_norm(A)
		if B.tie_embeddings:H=F.linear(A,C.to(dtype=A.dtype))
		else:
			if B.lm_head is _A:raise RuntimeError(_i)
			H=B.lm_head(A)
		return B.logit_softcap*torch.tanh(H/B.logit_softcap)
def main():
	AM='final_model.pt';AL='ste_and_penalty';AK='WORLD_SIZE';y='final_model.int8.ptz';X='base_lr';global zeropower_via_newtonschulz5;e=Path(__file__).read_text(encoding=_J);A=Hyperparameters();zeropower_via_newtonschulz5=torch.compile(zeropower_via_newtonschulz5);I='RANK'in os.environ and AK in os.environ;N=int(os.environ.get('RANK','0'));E=int(os.environ.get(AK,'1'));z=int(os.environ.get('LOCAL_RANK','0'))
	if E<=0:raise ValueError(f"WORLD_SIZE must be positive, got {E}")
	if 8%E!=0:raise ValueError(f"WORLD_SIZE={E} must divide 8 so grad_accum_steps stays integral")
	G=8//E
	if A.curriculum_min_seq_len>0 and(A.train_seq_len%A.curriculum_min_seq_len or A.train_seq_len//A.curriculum_min_seq_len&A.train_seq_len//A.curriculum_min_seq_len-1 or A.train_batch_tokens//(E*G)%A.curriculum_min_seq_len):raise ValueError('CURRICULUM_MIN_SEQ_LEN must divide TRAIN_SEQ_LEN and per-rank tokens, with a power-of-two ratio to TRAIN_SEQ_LEN')
	A0=_E/G
	if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
	H=torch.device(_K,z);torch.cuda.set_device(H)
	if I:dist.init_process_group(backend='nccl',device_id=H);dist.barrier()
	Y=N==0;torch.backends.cuda.matmul.allow_tf32=_B;torch.backends.cudnn.allow_tf32=_B;from torch.backends.cuda import enable_cudnn_sdp as AN,enable_flash_sdp as AO,enable_math_sdp as AP,enable_mem_efficient_sdp as AQ;AN(_C);AO(_B);AQ(_C);AP(_C);Z=_A
	if Y:os.makedirs('logs',exist_ok=_B);Z=f"logs/{A.run_id}.txt";print(Z)
	def B(msg,console=_B):
		if not Y:return
		if console:print(msg)
		if Z is not _A:
			with open(Z,'a',encoding=_J)as A:print(msg,file=A)
	B(e,console=_C);B('='*100,console=_C);B(f"Running Python {sys.version}",console=_C);B(f"Running PyTorch {torch.__version__}",console=_C);B(subprocess.run(['nvidia-smi'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=_B,check=_C).stdout,console=_C);B('='*100,console=_C);random.seed(A.seed);np.random.seed(A.seed);torch.manual_seed(A.seed);torch.cuda.manual_seed_all(A.seed)
	if not A.tokenizer_path.endswith('.model'):raise ValueError(f"Script only setup for SentencePiece .model file: {A.tokenizer_path}")
	f=spm.SentencePieceProcessor(model_file=A.tokenizer_path)
	if int(f.vocab_size())!=A.vocab_size:raise ValueError(f"VOCAB_SIZE={A.vocab_size} does not match tokenizer vocab_size={int(f.vocab_size())}")
	A1=Path(A.data_path).resolve();AR=len(list(A1.glob(_V)));a=load_validation_tokens(A.val_files,A.train_seq_len);g,h,i=build_sentencepiece_luts(f,A.vocab_size,H);B(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={A.tokenizer_path}");B(f"train_loader:dataset:{A1.name} train_shards:{AR}");B(f"val_loader:shards pattern={A.val_files} tokens:{a.numel()-1}");C=GPT(vocab_size=A.vocab_size,num_layers=A.num_layers,model_dim=A.model_dim,num_heads=A.num_heads,num_kv_heads=A.num_kv_heads,mlp_mult=A.mlp_mult,tie_embeddings=A.tie_embeddings,tok_emb_qat_enabled=A.tok_emb_qat_mode in{'ste',AL},tied_embed_init_std=A.tied_embed_init_std,logit_softcap=A.logit_softcap,rope_base=A.rope_base,qk_gain_init=A.qk_gain_init).to(H).bfloat16()
	for T in C.modules():
		if isinstance(T,CastedLinear):T.float()
	for(j,T)in C.named_modules():
		if isinstance(T,CastedLinear):T.quant_name=f"{j}.weight"
	restore_low_dim_params_to_fp32(C);A2=torch.compile(C,dynamic=_C,fullgraph=_B);O=DDP(A2,device_ids=[z],broadcast_buffers=_C)if I else A2;A3=list(C.blocks.named_parameters());AS=[A for(B,A)in A3 if A.ndim==2 and not any(A in B for A in CONTROL_TENSOR_NAME_PATTERNS)];A4=[A for(B,A)in A3 if A.ndim<2 or any(A in B for A in CONTROL_TENSOR_NAME_PATTERNS)]
	if C.skip_weights.numel()>0:A4.append(C.skip_weights)
	k=A.tied_embed_lr if A.tie_embeddings else A.embed_lr;AT=torch.optim.Adam([{_L:[C.tok_emb.weight],_I:k,X:k}],betas=(A.beta1,A.beta2),eps=A.adam_eps,fused=_B);l=Muon(AS,lr=A.matrix_lr,momentum=A.muon_momentum,backend_steps=A.muon_backend_steps,beta2=A.muon_beta2)
	for P in l.param_groups:P[X]=A.matrix_lr
	AU=torch.optim.Adam([{_L:A4,_I:A.scalar_lr,X:A.scalar_lr}],betas=(A.beta1,A.beta2),eps=A.adam_eps,fused=_B);K=[AT,l,AU]
	if C.lm_head is not _A:AV=torch.optim.Adam([{_L:[C.lm_head.weight],_I:A.head_lr,X:A.head_lr}],betas=(A.beta1,A.beta2),eps=A.adam_eps,fused=_B);K.insert(1,AV)
	AW=sum(A.numel()for A in C.parameters());B(f"model_params:{AW}");B(f"world_size:{E} grad_accum_steps:{G}");B('sdp_backends:cudnn=False flash=True mem_efficient=False math=False');B(f"attention_mode:gqa num_heads:{A.num_heads} num_kv_heads:{A.num_kv_heads}");B(f"tie_embeddings:{A.tie_embeddings} embed_lr:{k} head_lr:{A.head_lr if C.lm_head is not _A else _D} matrix_lr:{A.matrix_lr} scalar_lr:{A.scalar_lr}");B(f"train_batch_tokens:{A.train_batch_tokens} train_seq_len:{A.train_seq_len} iterations:{A.iterations} warmup_steps:{A.warmup_steps} max_wallclock_seconds:{A.max_wallclock_seconds:.3f}");B(f"tok_emb_qat:mode={A.tok_emb_qat_mode} last_steps:{A.tok_emb_qat_last_steps} ramp_steps:{A.tok_emb_qat_ramp_steps} penalty:{A.tok_emb_qat_penalty:.6f}");B(f"seed:{A.seed}");m=DistributedTokenLoader(A.train_files,N,E,H)
	def U():
		for A in K:A.zero_grad(set_to_none=_B)
	L=1e3*A.max_wallclock_seconds if A.max_wallclock_seconds>0 else _A
	def AX(step,elapsed_ms):
		C=elapsed_ms;B=step
		if A.warmdown_iters<=0:return _E
		if L is _A:F=max(A.iterations-A.warmdown_iters,0);return max((A.iterations-B)/max(A.warmdown_iters,1),_D)if F<=B<A.iterations else _E
		G=C/max(B,1);D=A.warmdown_iters*G;E=max(L-C,_D);return E/max(D,1e-09)if E<=D else _E
	def AY(step,elapsed_ms):
		C=elapsed_ms;B=step
		if A.tok_emb_qat_last_steps<=0 or A.tok_emb_qat_mode=='off'and A.lowbit_qat_penalty<=0:return _D
		if L is _A:
			D=max(A.iterations-A.tok_emb_qat_last_steps,0)
			if B<D:return _D
			if A.tok_emb_qat_ramp_steps<=0:return _E
			return min((B-D+1)/max(A.tok_emb_qat_ramp_steps,1),_E)
		E=C/max(B,1);F=max(A.tok_emb_qat_last_steps,1)*E;G=max(L-C,_D)
		if G>F:return _D
		if A.tok_emb_qat_ramp_steps<=0:return _E
		H=max(A.tok_emb_qat_ramp_steps,1)*E;return min((F-G)/max(H,1e-09),_E)
	def A5(step):
		if A.curriculum_min_seq_len<=0 or A.curriculum_steps<=0:return A.train_seq_len
		B=int(math.log2(A.train_seq_len//A.curriculum_min_seq_len));return A.train_seq_len>>max(B-int(min(step/max(A.curriculum_steps,1),_E)*(B+1)),0)
	if A.warmup_steps>0:
		AZ={A:B.detach().cpu().clone()for(A,B)in C.state_dict().items()};Aa=[copy.deepcopy(A.state_dict())for A in K];O.train()
		for b in range(A.warmup_steps):
			U()
			for n in range(G):
				if I:O.require_backward_grad_sync=n==G-1
				o,p=m.next_batch(A.train_batch_tokens,A5(b*A.curriculum_steps//max(A.warmup_steps,1)),G)
				with torch.autocast(device_type=_K,dtype=torch.bfloat16,enabled=_B):Ab=O(o,p)
				(Ab*A0).backward()
			for M in K:M.step()
			U()
			if A.warmup_steps<=20 or(b+1)%10==0 or b+1==A.warmup_steps:B(f"warmup_step:{b+1}/{A.warmup_steps}")
		C.load_state_dict(AZ,strict=_B)
		for(M,Ac)in zip(K,Aa,strict=_B):M.load_state_dict(Ac)
		U()
		if I:O.require_backward_grad_sync=_B
		m=DistributedTokenLoader(A.train_files,N,E,H)
	Q=_D;V=_A;W=0;R=_A;torch.cuda.synchronize();c=time.perf_counter();D=0
	while _B:
		q=D==A.iterations or R is not _A and D>=R;Ad=q or A.val_loss_every>0 and D%A.val_loss_every==0
		if Ad:
			r=float(C.tok_emb_qat_alpha.item())
			if r!=_D:C.tok_emb_qat_alpha.zero_()
			torch.cuda.synchronize();Q+=1e3*(time.perf_counter()-c);Ae,Af=eval_val(A,C,N,E,H,G,a,g,h,i);B(f"step:{D}/{A.iterations} val_loss:{Ae:.4f} val_bpb:{Af:.4f} train_time:{Q:.0f}ms step_avg:{Q/max(D,1):.2f}ms")
			if r!=_D and not q:C.tok_emb_qat_alpha.fill_(r)
			torch.cuda.synchronize();c=time.perf_counter()
		if q:
			if R is not _A and D<A.iterations:B(f"stopping_early: wallclock_cap train_time:{Q:.0f}ms step:{D}/{A.iterations}")
			break
		A6=Q+1e3*(time.perf_counter()-c);A7=AX(D,A6);S=AY(D,A6);A8=A5(D);C.tok_emb_qat_alpha.fill_(S);U();s=torch.zeros((),device=H)
		for n in range(G):
			if I:
				O.require_backward_grad_sync=n==G-1;o,p=m.next_batch(A.train_batch_tokens,A8,G)
				with torch.autocast(device_type=_K,dtype=torch.bfloat16,enabled=_B):J=O(o,p)
				if A.tok_emb_qat_mode in{'penalty',AL}and A.tok_emb_qat_penalty>0 and S>0:t=F.mse_loss(C.tok_emb.weight.float(),quantize_dequantize_float_tensor(_U,C.tok_emb.weight).float());J=J+A.tok_emb_qat_penalty*S*t.to(J.dtype)
				if A.lowbit_qat_penalty>0 and S>0 and LOWBIT_QAT_NAME_PATTERNS:t=sum(F.mse_loss(A.float(),quantize_dequantize_float_tensor(B,A).float())for(B,A)in C.named_parameters()if A.ndim==2 and any(A in B for A in LOWBIT_QAT_NAME_PATTERNS));J=J+A.lowbit_qat_penalty*S*t.to(J.dtype)
			s+=J.detach();(J*A0).backward()
		s/=G;A9=min(D/A.muon_momentum_warmup_steps,_E)if A.muon_momentum_warmup_steps>0 else _E;Ag=(1-A9)*A.muon_momentum_warmup_start+A9*A.muon_momentum
		for P in l.param_groups:P[_W]=Ag
		for M in K:
			for P in M.param_groups:P[_I]=P[X]*A7
		if A.grad_clip_norm>0:torch.nn.utils.clip_grad_norm_(C.parameters(),A.grad_clip_norm)
		for M in K:M.step()
		U();D+=1;u=Q+1e3*(time.perf_counter()-c)
		if A.swa_enabled and A7<A.swa_start_frac and D%A.swa_every==0:
			if V is _A:V={A:B.detach().cpu().clone()for(A,B)in C.state_dict().items()};W=1;B(f"swa:start step:{D}")
			else:
				for(j,Ah)in C.state_dict().items():V[j]+=Ah.detach().cpu()
				W+=1
		Ai=A.train_log_every>0 and(D<=10 or D%A.train_log_every==0 or R is not _A)
		if Ai:B(f"step:{D}/{A.iterations} train_loss:{s.item():.4f} train_time:{u:.0f}ms step_avg:{u/D:.2f}ms train_seq_len:{A8} tok_emb_qat_alpha:{S:.3f}")
		v=L is not _A and u>=L
		if I and L is not _A:AA=torch.tensor(int(v),device=H);dist.all_reduce(AA,op=dist.ReduceOp.MAX);v=bool(AA.item())
		if R is _A and v:R=D
	B(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB");C.tok_emb_qat_alpha.zero_()
	if A.swa_enabled and V is not _A and W>1:B(f"swa:applying averaged {W} checkpoints");C.load_state_dict({A:(B/W).to(dtype=C.state_dict()[A].dtype)for(A,B)in V.items()},strict=_B)
	if Y:torch.save(C.state_dict(),AM);AB=os.path.getsize(AM);d=len(e.encode(_J));B(f"Serialized model: {AB} bytes");B(f"Code size: {d} bytes");B(f"Total submission size: {AB+d} bytes")
	Aj,w=quantize_state_dict_int8(C.state_dict());AC=io.BytesIO();torch.save(Aj,AC);AD=AC.getvalue();Ak,AE=compress_bytes(AD);Al=len(AD)
	if Y:
		with open(y,'wb')as x:x.write(Ak)
		AF=os.path.getsize(y);d=len(e.encode(_J));Am=w[_S]/max(w[_G],1);B(f"Serialized model quant+{AE}: {AF} bytes (payload:{w[_G]} raw_torch:{Al} payload_ratio:{Am:.2f}x)");B(f"Total submission size quant+{AE}: {AF+d} bytes")
	if I:dist.barrier()
	with open(y,'rb')as x:An=x.read()
	Ao=torch.load(io.BytesIO(decompress_bytes(An)),map_location=_T);C.load_state_dict(dequantize_state_dict_int8(Ao),strict=_B);torch.cuda.synchronize();Ap=time.perf_counter();AG,AH=eval_val(A,C,N,E,H,G,a,g,h,i);torch.cuda.synchronize();B(f"final_int8_zlib_roundtrip val_loss:{AG:.4f} val_bpb:{AH:.4f} eval_time:{1e3*(time.perf_counter()-Ap):.0f}ms");B(f"final_int8_zlib_roundtrip_exact val_loss:{AG:.8f} val_bpb:{AH:.8f}")
	if A.eval_stride>0:torch.cuda.synchronize();Aq=time.perf_counter();AI,AJ=eval_val_sliding_window(A,C,N,E,H,a,g,h,i,stride=A.eval_stride);torch.cuda.synchronize();B(f"final_sliding_window_eval stride:{A.eval_stride} val_loss:{AI:.4f} val_bpb:{AJ:.4f} eval_time:{1e3*(time.perf_counter()-Aq):.0f}ms");B(f"final_sliding_window_eval_exact stride:{A.eval_stride} val_loss:{AI:.8f} val_bpb:{AJ:.8f}")
	if I:dist.destroy_process_group()
if __name__=='__main__':main()