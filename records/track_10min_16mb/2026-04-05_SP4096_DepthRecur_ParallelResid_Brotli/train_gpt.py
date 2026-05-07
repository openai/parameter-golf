from __future__ import annotations
_A3='passthrough_ctrl'
_A2='jm_lambdas'
_A1='min_count'
_A0='min_order'
_z='ng_primes'
_y='ng_mask'
_x='snap_full'
_w='snap_ctx'
_v='passthrough_orig_dtypes'
_u='dtypes'
_t='scales'
_s='quantized'
_r='per_row'
_q='scheme'
_p='torch.'
_o='momentum'
_n='shard_mom'
_m='padded_grad'
_l='fineweb_train_*.bin'
_k='type'
_j='inf'
_i='full_tables'
_h='ctx_tables'
_g='n_orders'
_f='mean'
_e='shard'
_d='0.0001'
_c='9,10'
_b='.scale'
_a='.q'
_Z='mlp_down_bank'
_Y='mlp_up_bank'
_X='kv_bank'
_W='qo_bank'
_V='attn'
_U='mlp'
_T='total_scored'
_S='total_matches'
_R='passthrough'
_Q='scale'
_P='full_update'
_O='tokens_scored'
_N='cpu'
_M='utf-8'
_L='ve'
_K=','
_J='1'
_I='cuda'
_H='lr'
_G='params'
_F='0'
_E=.0
_D=1.
_C=False
_B=True
_A=None
import copy,glob,io,lzma,math
try:import brotli;_HAS_BROTLI=_B
except ImportError:_HAS_BROTLI=_C
import os,random,subprocess,sys,time,uuid,zlib
from pathlib import Path
try:import zstandard;_COMPRESSOR='zstd'
except ImportError:_COMPRESSOR='zlib'
import numpy as np,sentencepiece as spm,torch,torch.distributed as dist,torch.nn.functional as F
from torch import Tensor,nn
from torch.nn.parallel import DistributedDataParallel as DDP
try:from flash_attn_interface import flash_attn_func as flash_attn_3_func;HAS_FA3=_B
except ImportError:HAS_FA3=_C
SKIP_QUANTIZE=bool(int(os.environ.get('SKIP_QUANTIZE',_F)))
SKIP_COMPILE=bool(int(os.environ.get('SKIP_COMPILE',_F)))
class Hyperparameters:
	data_path=os.environ.get('DATA_PATH','./data/datasets/fineweb10B_sp1024');train_files=os.path.join(data_path,_l);val_files=os.path.join(data_path,'fineweb_val_*.bin');tokenizer_path=os.environ.get('TOKENIZER_PATH','./data/tokenizers/fineweb_1024_bpe.model');run_id=os.environ.get('RUN_ID',str(uuid.uuid4()));seed=int(os.environ.get('SEED',1337));val_batch_size=int(os.environ.get('VAL_BATCH_SIZE',524288));val_loss_every=int(os.environ.get('VAL_LOSS_EVERY',4000));train_log_every=int(os.environ.get('TRAIN_LOG_EVERY',500));iterations=int(os.environ.get('ITERATIONS',20000));warmdown_iters=int(os.environ.get('WARMDOWN_ITERS',3500));warmup_steps=int(os.environ.get('WARMUP_STEPS',20));train_batch_tokens=int(os.environ.get('TRAIN_BATCH_TOKENS',786432));train_seq_len=int(os.environ.get('TRAIN_SEQ_LEN',2048));eval_seq_len=int(os.environ.get('EVAL_SEQ_LEN',2048));max_wallclock_seconds=float(os.environ.get('MAX_WALLCLOCK_SECONDS',6e2));qk_gain_init=float(os.environ.get('QK_GAIN_INIT',1.5));vocab_size=int(os.environ.get('VOCAB_SIZE',1024));num_layers=int(os.environ.get('NUM_LAYERS',11));num_kv_heads=int(os.environ.get('NUM_KV_HEADS',4));model_dim=int(os.environ.get('MODEL_DIM',512));num_heads=int(os.environ.get('NUM_HEADS',8));mlp_mult=float(os.environ.get('MLP_MULT',3.));tie_embeddings=bool(int(os.environ.get('TIE_EMBEDDINGS',_J)));rope_base=float(os.environ.get('ROPE_BASE',1e4));logit_softcap=float(os.environ.get('LOGIT_SOFTCAP',3e1));embed_lr=float(os.environ.get('EMBED_LR',.6));head_lr=float(os.environ.get('HEAD_LR',.008));tied_embed_lr=float(os.environ.get('TIED_EMBED_LR',.035));tied_embed_init_std=float(os.environ.get('TIED_EMBED_INIT_STD',.005));matrix_lr=float(os.environ.get('MATRIX_LR',.025));scalar_lr=float(os.environ.get('SCALAR_LR',.025));muon_momentum=float(os.environ.get('MUON_MOMENTUM',.99));muon_backend_steps=int(os.environ.get('MUON_BACKEND_STEPS',5));muon_momentum_warmup_start=float(os.environ.get('MUON_MOMENTUM_WARMUP_START',.92));muon_momentum_warmup_steps=int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS',1500));muon_eq_r=bool(int(os.environ.get('MUON_EQ_R',_J)));recur_layers=os.environ.get('RECUR_LAYERS','');recur_start_step=int(os.environ.get('RECUR_START_STEP',3000));parallel_start_layer=int(os.environ.get('PARALLEL_START_LAYER',-1));beta1=float(os.environ.get('BETA1',.9));beta2=float(os.environ.get('BETA2',.95));adam_eps=float(os.environ.get('ADAM_EPS',1e-08));grad_clip_norm=float(os.environ.get('GRAD_CLIP_NORM',.3));eval_stride=int(os.environ.get('EVAL_STRIDE',64));mtp_num_heads=int(os.environ.get('MTP_NUM_HEADS',0));mtp_loss_weight=float(os.environ.get('MTP_LOSS_WEIGHT',.2));muon_beta2=float(os.environ.get('MUON_BETA2',.95));swa_enabled=bool(int(os.environ.get('SWA_ENABLED',_J)));swa_every=int(os.environ.get('SWA_EVERY',50));lawa_enabled=bool(int(os.environ.get('LAWA_ENABLED',_F)));lawa_k=int(os.environ.get('LAWA_K',10));lawa_freq=int(os.environ.get('LAWA_FREQ',100));muon_wd=float(os.environ.get('MUON_WD',.04));adam_wd=float(os.environ.get('ADAM_WD',.04));qat_enabled=bool(int(os.environ.get('QAT_ENABLED',_F)));bigram_vocab_size=int(os.environ.get('BIGRAM_VOCAB_SIZE',2048));bigram_dim=int(os.environ.get('BIGRAM_DIM',128));trigram_enabled=bool(int(os.environ.get('TRIGRAM',_F)));xsa_last_n=int(os.environ.get('XSA_LAST_N',4));rope_dims=int(os.environ.get('ROPE_DIMS',16));ln_scale=bool(int(os.environ.get('LN_SCALE',_J)));dtg_enabled=bool(int(os.environ.get('DTG_ENABLED',_F)));late_qat_threshold=float(os.environ.get('LATE_QAT_THRESHOLD',.15));ve_enabled=bool(int(os.environ.get('VE_ENABLED',_J)));ve_dim=int(os.environ.get('VE_DIM',128));ve_layers=os.environ.get('VE_LAYERS',_c);gated_attention=bool(int(os.environ.get('GATED_ATTENTION',_F)));value_residual=bool(int(os.environ.get('VALUE_RESIDUAL',_F)));ttt_enabled=bool(int(os.environ.get('TTT_ENABLED',_F)));ttt_lr=float(os.environ.get('TTT_LR',.002));ttt_epochs=int(os.environ.get('TTT_EPOCHS',3));ttt_chunk_tokens=int(os.environ.get('TTT_CHUNK_TOKENS',32768));ttt_freeze_blocks=int(os.environ.get('TTT_FREEZE_BLOCKS',2));ttt_momentum=float(os.environ.get('TTT_MOMENTUM',.9));ttt_batch_seqs=int(os.environ.get('TTT_BATCH_SEQS',32));ttt_grad_clip=float(os.environ.get('TTT_GRAD_CLIP',_D));crown_q_enabled=bool(int(os.environ.get('CROWN_Q_ENABLED',_F)));crown_q_lambda=float(os.environ.get('CROWN_Q_LAMBDA','0.01'));soft_round_qat=bool(int(os.environ.get('SOFT_ROUND_QAT',_F)));ttt_optimizer=os.environ.get('TTT_OPTIMIZER','sgd');ttt_adamw_lr=float(os.environ.get('TTT_ADAMW_LR',_d));ttt_q_only=bool(int(os.environ.get('TTT_Q_ONLY',_F)));ttt_difficulty=bool(int(os.environ.get('TTT_DIFFICULTY',_F)));ttt_hard_epochs=int(os.environ.get('TTT_HARD_EPOCHS','5'));ttt_easy_epochs=int(os.environ.get('TTT_EASY_EPOCHS',_J));ttt_hard_threshold=float(os.environ.get('TTT_HARD_THRESHOLD','1.3'));ttt_easy_threshold=float(os.environ.get('TTT_EASY_THRESHOLD','1.0'));ttt_temperature=float(os.environ.get('TTT_TEMPERATURE','1.0'));ttt_nesterov=bool(int(os.environ.get('TTT_NESTEROV',_J)));ttt_lr_floor=float(os.environ.get('TTT_LR_FLOOR','0.1'));ngram_enabled=bool(int(os.environ.get('NGRAM_ENABLED',_F)));ngram_alpha=float(os.environ.get('NGRAM_ALPHA','0.40'));ngram_order=int(os.environ.get('NGRAM_ORDER','7'));ngram_min_order=int(os.environ.get('NGRAM_MIN_ORDER','2'));ngram_buckets=int(os.environ.get('NGRAM_BUCKETS','4194304'))
	if ngram_buckets&ngram_buckets-1!=0:raise ValueError(f"NGRAM_BUCKETS must be a power of 2, got {ngram_buckets}")
	ngram_min_count=int(os.environ.get('NGRAM_MIN_COUNT','2'));ngram_entropy=bool(int(os.environ.get('NGRAM_ENTROPY',_J)));ngram_ent_base=float(os.environ.get('NGRAM_ENT_BASE','0.05'));ngram_ent_range=float(os.environ.get('NGRAM_ENT_RANGE','0.55'));ngram_ent_scale=float(os.environ.get('NGRAM_ENT_SCALE','2.0'));ngram_ent_thresh=float(os.environ.get('NGRAM_ENT_THRESH','4.0'));ngram_jm=bool(int(os.environ.get('NGRAM_JM',_F)));ngram_per_order_centers=bool(int(os.environ.get('NGRAM_PER_ORDER_CENTERS',_F)));ngram_depth_signal=bool(int(os.environ.get('NGRAM_DEPTH_SIGNAL',_F)));ngram_depth_scale=float(os.environ.get('NGRAM_DEPTH_SCALE',_d));ngram_sync_interval=int(os.environ.get('NGRAM_SYNC_INTERVAL','50'));gptq_full_hessian=bool(int(os.environ.get('GPTQ_FULL_HESSIAN',_J)));gptq_calib_batches=int(os.environ.get('GPTQ_CALIB_BATCHES',256));gptq_block_size=int(os.environ.get('GPTQ_BLOCK_SIZE',128));gptq_damp=float(os.environ.get('GPTQ_DAMP','0.01'));mixed_bitwidth=bool(int(os.environ.get('MIXED_BITWIDTH',_F)));hadamard_rotation=bool(int(os.environ.get('HADAMARD_ROTATION',_F)));prequant_ttt=bool(int(os.environ.get('PREQUANT_TTT',_F)));prequant_ttt_lr=float(os.environ.get('PREQUANT_TTT_LR',_d));prequant_ttt_epochs=int(os.environ.get('PREQUANT_TTT_EPOCHS','3'))
def zeropower_via_newtonschulz5(G,steps=5,eps=1e-07):
	E,F,H=3.4445,-4.775,2.0315;C=G.ndim==2
	if C:G=G.unsqueeze(0)
	A=G.bfloat16();D=A.size(-2)>A.size(-1)
	if D:A=A.mT
	A=A/(A.norm(dim=(-2,-1),keepdim=_B)+eps)
	for J in range(steps):B=A@A.mT;I=F*B+H*(B@B);A=E*A+I@A
	if D:A=A.mT
	if C:A=A.squeeze(0)
	return A
class Muon(torch.optim.Optimizer):
	def __init__(A,params,lr,momentum,backend_steps,nesterov=_B,weight_decay=_E,muon_eq_r=_B):super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,nesterov=nesterov,weight_decay=weight_decay));A.muon_eq_r=muon_eq_r;A._built=_C
	def _build(A):
		A._distributed=dist.is_available()and dist.is_initialized();A._world_size=dist.get_world_size()if A._distributed else 1;A._rank=dist.get_rank()if A._distributed else 0;C=A._world_size;A._bank_meta=[]
		for I in A.param_groups:
			for B in I[_G]:G=B.shape[0];F=(G+C-1)//C*C;H=F//C;D=B.shape[1:];E=B.device;A._bank_meta.append({'p':B,'B':G,_m:torch.zeros(F,*D,device=E,dtype=torch.bfloat16),_e:torch.zeros(H,*D,device=E,dtype=torch.bfloat16),_n:torch.zeros(H,*D,device=E,dtype=torch.bfloat16),_P:torch.zeros(F,*D,device=E,dtype=torch.bfloat16),_Q:max(1,B.shape[-2]/B.shape[-1])**.5})
		A._bank_meta.sort(key=lambda m:-m['p'].numel());A._built=_B
	def launch_reduce_scatters(A):
		if not A._built:A._build()
		if not A._distributed:return
		A._rs_futures=[]
		for B in A._bank_meta:
			D=B['p']
			if D.grad is _A:A._rs_futures.append(_A);continue
			C=B[_m];C[:B['B']].copy_(D.grad.bfloat16())
			if C.shape[0]>B['B']:C[B['B']:].zero_()
			E=dist.reduce_scatter_tensor(B[_e],C,op=dist.ReduceOp.AVG,async_op=_B);A._rs_futures.append(E)
	@torch.no_grad()
	def step(self,closure=_A):
		U='_rs_futures';P=closure;O='momentum_buffer';A=self;Q=_A
		if P is not _A:
			with torch.enable_grad():Q=P()
		if not A._built:A._build()
		for I in A.param_groups:
			E=I[_H];R=I[_o];V=I['backend_steps'];W=I['nesterov'];F=I.get('weight_decay',_E);J=_A;B=_A;S=A._distributed and hasattr(A,U)
			for(T,G)in enumerate(A._bank_meta):
				H=G['p']
				if H.grad is _A:continue
				if J is not _A:
					J.wait();D=B['p'];M=B[_P][:B['B']]
					if F>_E:D.data.mul_(_D-E*F)
					D.add_(M.to(dtype=D.dtype),alpha=-E*B[_Q])
				if S and A._rs_futures[T]is not _A:A._rs_futures[T].wait();K=G[_e];L=G[_n]
				else:
					K=H.grad.bfloat16();N=A.state[H]
					if O not in N:N[O]=torch.zeros_like(K)
					L=N[O]
				L.mul_(R).add_(K)
				if W:C=K.add(L,alpha=R)
				else:C=L
				if A.muon_eq_r:X=(C*C).sum(dim=-1,keepdim=_B)+1e-07;C=C/X.sqrt()
				C=zeropower_via_newtonschulz5(C,steps=V)
				if S:J=dist.all_gather_into_tensor(G[_P],C,async_op=_B);B=G
				else:
					if F>_E:H.data.mul_(_D-E*F)
					H.add_(C.to(dtype=H.dtype),alpha=-E*G[_Q])
			if J is not _A:
				J.wait();D=B['p'];M=B[_P][:B['B']]
				if F>_E:D.data.mul_(_D-E*F)
				D.add_(M.to(dtype=D.dtype),alpha=-E*B[_Q])
			if hasattr(A,U):del A._rs_futures
		return Q
def build_sentencepiece_luts(sp,vocab_size,device):
	D=device;B=sp;G=int(B.vocab_size());E=max(G,vocab_size);F=np.zeros((E,),dtype=np.int16);H=np.zeros((E,),dtype=np.bool_);I=np.ones((E,),dtype=np.bool_)
	for A in range(G):
		if B.is_control(A)or B.is_unknown(A)or B.is_unused(A):continue
		I[A]=_C
		if B.is_byte(A):F[A]=1;continue
		C=B.id_to_piece(A)
		if C.startswith('▁'):H[A]=_B;C=C[1:]
		F[A]=len(C.encode(_M))
	return torch.tensor(F,dtype=torch.int16,device=D),torch.tensor(H,dtype=torch.bool,device=D),torch.tensor(I,dtype=torch.bool,device=D)
def load_validation_tokens(pattern,seq_len):
	B=pattern;A=seq_len;C=[Path(A)for A in sorted(glob.glob(B))]
	if not C:raise FileNotFoundError(f"No files found for pattern: {B}")
	D=torch.cat([load_data_shard(A)for A in C]).contiguous();E=(D.numel()-1)//A*A
	if E<=0:raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={A}")
	return D[:E+1]
def eval_val(args,model,rank,world_size,device,grad_accum_steps,val_tokens,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut,eval_seq_len=_A):
	K=val_tokens;J=grad_accum_steps;F=model;E=args;C=device;B=world_size;A=eval_seq_len or E.train_seq_len;L=E.val_batch_size//(B*J)
	if L<A:raise ValueError(f"VAL_BATCH_SIZE must provide at least one sequence per rank; got VAL_BATCH_SIZE={E.val_batch_size}, WORLD_SIZE={B}, GRAD_ACCUM_STEPS={J}, seq_len={A}")
	M=L//A;N=(K.numel()-1)//A;W=N*rank//B;O=N*(rank+1)//B;G=torch.zeros((),device=C,dtype=torch.float64);D=torch.zeros((),device=C,dtype=torch.float64);H=torch.zeros((),device=C,dtype=torch.float64);F.eval()
	with torch.inference_mode():
		for P in range(W,O,M):
			X=min(P+M,O);Y=P*A;Z=X*A+1;Q=K[Y:Z].to(device=C,dtype=torch.int64,non_blocking=_B);R=Q[:-1].reshape(-1,A);I=Q[1:].reshape(-1,A)
			with torch.autocast(device_type=_I,dtype=torch.bfloat16,enabled=_B):a=F(R,I).detach()
			S=float(I.numel());G+=a.to(torch.float64)*S;D+=S;b=R.reshape(-1);T=I.reshape(-1);U=base_bytes_lut[T].to(dtype=torch.int16);U+=(has_leading_space_lut[T]&~is_boundary_token_lut[b]).to(dtype=torch.int16);H+=U.to(torch.float64).sum()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(G,op=dist.ReduceOp.SUM);dist.all_reduce(D,op=dist.ReduceOp.SUM);dist.all_reduce(H,op=dist.ReduceOp.SUM)
	V=G/D;c=V.item()/math.log(2.);d=D.item()/H.item();F.train();return float(V.item()),float(c*d)
CONTROL_TENSOR_NAME_PATTERNS=tuple(A for A in os.environ.get('CONTROL_TENSOR_NAME_PATTERNS','attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,dtg_gate,ve_layer_scales,ve_shared.scale,attn_gate,vr_lambda').split(_K)if A)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS=tuple(A for A in os.environ.get('INT8_KEEP_FLOAT_FP32_NAME_PATTERNS',_K.join(CONTROL_TENSOR_NAME_PATTERNS)).split(_K)if A)
INT8_KEEP_FLOAT_MAX_NUMEL=65536
INT8_KEEP_FLOAT_STORE_DTYPE=torch.float16
INT8_PER_ROW_SCALE_DTYPE=torch.float16
INT8_CLIP_PERCENTILE=99.99984
INT8_CLIP_Q=INT8_CLIP_PERCENTILE/1e2
def tensor_nbytes(t):return int(t.numel())*int(t.element_size())
def keep_float_tensor(name,t,passthrough_orig_dtypes):
	if any(A in name for A in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):return t.float().contiguous()
	if t.dtype in{torch.float32,torch.bfloat16}:passthrough_orig_dtypes[name]=str(t.dtype).removeprefix(_p);return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
	return t
def quantize_float_tensor(t):
	A=t.float()
	if A.ndim==2:B=torch.quantile(A.abs(),INT8_CLIP_Q,dim=1)if A.numel()else torch.empty((A.shape[0],),dtype=torch.float32);E=torch.maximum(torch.minimum(A,B[:,_A]),-B[:,_A]);C=(B/127.).clamp_min(_D/127.);D=torch.clamp(torch.round(E/C[:,_A]),-127,127).to(torch.int8).contiguous();return D,C.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
	B=float(torch.quantile(A.abs().flatten(),INT8_CLIP_Q).item())if A.numel()else _E;C=torch.tensor(B/127. if B>0 else _D,dtype=torch.float32);D=torch.clamp(torch.round(torch.clamp(A,-B,B)/C),-127,127).to(torch.int8).contiguous();return D,C
def quantize_state_dict_int8(state_dict):
	S='baseline_tensor_bytes';R='num_nonfloat_tensors';Q='num_float_tensors';P='num_tensors';O='param_count';D='int8_payload_bytes';J={};K={};L={};E={};F={};G={};A=dict.fromkeys((O,P,Q,R,S,D),0)
	for(C,T)in state_dict.items():
		B=T.detach().to(_N).contiguous();A[O]+=int(B.numel());A[P]+=1;A[S]+=tensor_nbytes(B)
		if not B.is_floating_point():A[R]+=1;E[C]=B;A[D]+=tensor_nbytes(B);continue
		if B.numel()<=INT8_KEEP_FLOAT_MAX_NUMEL:M=keep_float_tensor(C,B,F);E[C]=M;A[D]+=tensor_nbytes(M);continue
		A[Q]+=1;N,H=quantize_float_tensor(B)
		if H.ndim>0:G[C]={_q:_r,'axis':0}
		J[C]=N;K[C]=H;L[C]=str(B.dtype).removeprefix(_p);A[D]+=tensor_nbytes(N)+tensor_nbytes(H)
	I={'__quant_format__':'int8_clean_per_row_v1',_s:J,_t:K,_u:L,_R:E}
	if G:I['qmeta']=G
	if F:I[_v]=F
	return I,A
def dequantize_state_dict_int8(obj):
	B=obj;D={};I=B.get('qmeta',{});J=B.get(_v,{})
	for(A,E)in B[_s].items():
		G=getattr(torch,B[_u][A]);C=B[_t][A]
		if I.get(A,{}).get(_q)==_r or C.ndim>0:C=C.to(dtype=torch.float32);D[A]=(E.float()*C.view(E.shape[0],*[1]*(E.ndim-1))).to(dtype=G).contiguous()
		else:K=float(C.item());D[A]=(E.float()*K).to(dtype=G).contiguous()
	for(A,L)in B[_R].items():
		F=L.detach().to(_N).contiguous();H=J.get(A)
		if isinstance(H,str):F=F.to(dtype=getattr(torch,H)).contiguous()
		D[A]=F
	return D
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
	_qat_enabled=_C;_soft_round_alpha=_E
	def forward(B,x):
		A=B.weight.to(x.dtype)
		if CastedLinear._qat_enabled and B.training and A.ndim==2:
			C=B.weight.float();J=C.abs().amax(dim=1);D=(J/31.).clamp_min(_D/31.);E=C/D[:,_A]
			if CastedLinear._soft_round_alpha>0:H=CastedLinear._soft_round_alpha;I=torch.floor(E+.5);K=E-I;L=torch.tensor(H*.5,device=A.device,dtype=C.dtype);M=I+.5*torch.tanh(H*K)/torch.tanh(L);G=(torch.clamp(M,-31,31)*D[:,_A]).to(x.dtype);A=G
			else:
				with torch.no_grad():G=(torch.clamp(torch.round(E),-31,31)*D[:,_A]).to(x.dtype)
				A=A+(G-A).detach()
		N=B.bias.to(x.dtype)if B.bias is not _A else _A;return F.linear(x,A,N)
def restore_low_dim_params_to_fp32(module):
	with torch.no_grad():
		for(B,A)in module.named_parameters():
			if(A.ndim<2 or any(A in B for A in CONTROL_TENSOR_NAME_PATTERNS))and A.dtype!=torch.float32:A.data=A.data.float()
class Rotary(nn.Module):
	def __init__(A,dim,base=1e4,train_seq_len=1024,rope_dims=0):B=rope_dims;super().__init__();A.dim=dim;A.base=base;A.train_seq_len=train_seq_len;A.rope_dims=B if B>0 else dim;C=_D/base**(torch.arange(0,A.rope_dims,2,dtype=torch.float32)/A.rope_dims);A.register_buffer('inv_freq',C,persistent=_C);A._seq_len_cached=0;A._cos_cached=_A;A._sin_cached=_A
	def forward(A,seq_len,device,dtype):
		F=dtype;C=device;B=seq_len
		if A._cos_cached is _A or A._sin_cached is _A or A._seq_len_cached!=B or A._cos_cached.device!=C:
			D=A.rope_dims
			if B>A.train_seq_len:H=B/A.train_seq_len;I=A.base*H**(D/(D-2));E=_D/I**(torch.arange(0,D,2,dtype=torch.float32,device=C)/D)
			else:E=A.inv_freq.to(C)
			J=torch.arange(B,device=C,dtype=E.dtype);G=torch.outer(J,E);A._cos_cached=G.cos()[_A,:,_A,:];A._sin_cached=G.sin()[_A,:,_A,:];A._seq_len_cached=B
		return A._cos_cached.to(dtype=F),A._sin_cached.to(dtype=F)
def apply_rotary_emb(x,cos,sin,rope_dims=0):
	F=sin;E=cos;A=rope_dims
	if A>0 and A<x.size(-1):G,H=x[...,:A],x[...,A:];B=A//2;C,D=G[...,:B],G[...,B:];G=torch.cat((C*E+D*F,C*-F+D*E),dim=-1);return torch.cat((G,H),dim=-1)
	B=x.size(-1)//2;C,D=x[...,:B],x[...,B:];return torch.cat((C*E+D*F,C*-F+D*E),dim=-1)
class CausalSelfAttention(nn.Module):
	def __init__(A,dim,num_heads,num_kv_heads,rope_base,qk_gain_init,gated_attention=_C,value_residual=_C):
		F=value_residual;E=gated_attention;D=num_kv_heads;C=dim;B=num_heads;super().__init__()
		if C%B!=0:raise ValueError('model_dim must be divisible by num_heads')
		if B%D!=0:raise ValueError('num_heads must be divisible by num_kv_heads')
		A.num_heads=B;A.num_kv_heads=D;A.head_dim=C//B
		if A.head_dim%2!=0:raise ValueError('head_dim must be even for RoPE')
		A.q_gain=nn.Parameter(torch.full((B,),qk_gain_init,dtype=torch.float32));A.rope_dims=0;A.rotary=Rotary(A.head_dim,base=rope_base,train_seq_len=1024);A.use_xsa=_C;A.gated_attention=E
		if E:A.attn_gate=nn.Linear(C,B,bias=_B);nn.init.zeros_(A.attn_gate.weight);nn.init.constant_(A.attn_gate.bias,4.)
		A.value_residual=F
		if F:A.vr_lambda=nn.Parameter(torch.tensor([.5,.5],dtype=torch.float32))
	def _xsa_efficient(K,y,v):A,B,C,D=y.shape;E=v.size(-2);I=C//E;G=y.reshape(A,B,E,I,D);H=F.normalize(v,dim=-1).unsqueeze(-2);J=(G*H).sum(dim=-1,keepdim=_B)*H;return(G-J).reshape(A,B,C,D)
	def forward(A,x,q_w,k_w,v_w,out_w,v_embed=_A,v0=_A):
		K=v_embed;H,G,P=x.shape;B=F.linear(x,q_w.to(x.dtype)).reshape(H,G,A.num_heads,A.head_dim);E=F.linear(x,k_w.to(x.dtype)).reshape(H,G,A.num_kv_heads,A.head_dim);C=F.linear(x,v_w.to(x.dtype))
		if K is not _A:C=C+K
		C=C.reshape(H,G,A.num_kv_heads,A.head_dim);Q=C if A.value_residual else _A
		if A.value_residual and v0 is not _A:L=A.vr_lambda.to(dtype=C.dtype);C=L[0]*v0+L[1]*C
		B=F.rms_norm(B,(B.size(-1),));E=F.rms_norm(E,(E.size(-1),));M,N=A.rotary(G,x.device,B.dtype);B=apply_rotary_emb(B,M,N,A.rope_dims);E=apply_rotary_emb(E,M,N,A.rope_dims);B=B*A.q_gain.to(dtype=B.dtype)[_A,_A,:,_A]
		if HAS_FA3:D=flash_attn_3_func(B,E,C,causal=_B)
		else:
			R,I,J=B.transpose(1,2),E.transpose(1,2),C.transpose(1,2)
			if A.num_kv_heads!=A.num_heads:O=A.num_heads//A.num_kv_heads;I=I.repeat_interleave(O,dim=1);J=J.repeat_interleave(O,dim=1)
			D=F.scaled_dot_product_attention(R,I,J,is_causal=_B).transpose(1,2)
		if A.use_xsa:D=A._xsa_efficient(D,C)
		if A.gated_attention:S=torch.sigmoid(A.attn_gate(x)).unsqueeze(-1);D=D*S
		D=D.reshape(H,G,P);return F.linear(D,out_w.to(x.dtype)),Q
class SmearGate(nn.Module):
	def __init__(A,dim):super().__init__();A.gate=nn.Parameter(torch.zeros(dim,dtype=torch.float32))
	def forward(B,x):A=torch.sigmoid(B.gate.to(dtype=x.dtype))[_A,_A,:];C=torch.cat([torch.zeros_like(x[:,:1]),x[:,:-1]],dim=1);return(1-A)*x+A*C
class BigramHashEmbedding(nn.Module):
	def __init__(A,bigram_vocab_size,bigram_dim,model_dim,trigram=_C):
		D=model_dim;C=bigram_vocab_size;B=bigram_dim;super().__init__();A.bigram_vocab_size=C;A._trigram=trigram;A.embed=nn.Embedding(C,B);nn.init.zeros_(A.embed.weight);A.proj=CastedLinear(B,D,bias=_C)if B!=D else _A
		if A.proj is not _A:nn.init.zeros_(A.proj.weight)
		A.scale=nn.Parameter(torch.tensor(.05,dtype=torch.float32))
	def bigram_hash(D,tokens):A=tokens.to(torch.int32);C=D.bigram_vocab_size-1;B=torch.empty_like(A);B[...,0]=C;B[...,1:]=torch.bitwise_xor(36313*A[...,1:],27191*A[...,:-1])%C;return B.long()
	def trigram_hash(D,tokens):A=tokens.to(torch.int32);C=D.bigram_vocab_size-1;B=torch.empty_like(A);B[...,:2]=C;B[...,2:]=(36313*A[...,2:]^27191*A[...,1:-1]^51497*A[...,:-2])%C;return B.long()
	def forward(A,token_ids):
		C=token_ids;B=A.embed(A.bigram_hash(C))
		if A._trigram:B=B+A.embed(A.trigram_hash(C))
		if A.proj is not _A:B=A.proj(B)
		return B*A.scale.to(dtype=B.dtype)
class ValueEmbedding(nn.Module):
	def __init__(A,vocab_size,ve_dim,model_dim):
		C=model_dim;B=ve_dim;super().__init__();A.embed=nn.Embedding(vocab_size,B);nn.init.normal_(A.embed.weight,std=.01);A.proj=CastedLinear(B,C,bias=_C)if B!=C else _A
		if A.proj is not _A:nn.init.zeros_(A.proj.weight)
		A.scale=nn.Parameter(torch.tensor(.1,dtype=torch.float32))
	def forward(A,token_ids):
		B=A.embed(token_ids)
		if A.proj is not _A:B=A.proj(B)
		return B*A.scale.to(dtype=B.dtype)
class MLP(nn.Module):
	def __init__(A,dim,mlp_mult):super().__init__()
	def forward(A,x,up_w,down_w):x=F.leaky_relu(F.linear(x,up_w.to(x.dtype)),negative_slope=.5);return F.linear(x.square(),down_w.to(x.dtype))
class Block(nn.Module):
	def __init__(A,dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init,layer_idx=0,ln_scale=_C,dtg=_C,gated_attention=_C,value_residual=_C):
		B=dim;super().__init__();A.attn_norm=RMSNorm();A.mlp_norm=RMSNorm();A.attn=CausalSelfAttention(B,num_heads,num_kv_heads,rope_base,qk_gain_init,gated_attention=gated_attention,value_residual=value_residual);A.mlp=MLP(B,mlp_mult);A.attn_scale=nn.Parameter(torch.ones(B,dtype=torch.float32));A.mlp_scale=nn.Parameter(torch.ones(B,dtype=torch.float32));A.resid_mix=nn.Parameter(torch.stack((torch.ones(B),torch.zeros(B))).float());A.ln_scale_factor=_D/math.sqrt(layer_idx+1)if ln_scale else _D
		if dtg:A.dtg_gate=nn.Linear(B,1,bias=_B);nn.init.zeros_(A.dtg_gate.weight);nn.init.constant_(A.dtg_gate.bias,2.)
		else:A.dtg_gate=_A
	def forward(A,x,x0,q_w,k_w,v_w,out_w,up_w,down_w,v_embed=_A,v0=_A,parallel=_C):
		D=down_w;E=A.resid_mix.to(dtype=x.dtype);B=E[0][_A,_A,:]*x+E[1][_A,_A,:]*x0;G=A.attn_norm(B)*A.ln_scale_factor;F,H=A.attn(G,q_w,k_w,v_w,out_w,v_embed=v_embed,v0=v0)
		if parallel:I=A.mlp(A.mlp_norm(B)*A.ln_scale_factor,up_w,D);C=B+A.attn_scale.to(dtype=B.dtype)[_A,_A,:]*F+A.mlp_scale.to(dtype=B.dtype)[_A,_A,:]*I
		else:C=B+A.attn_scale.to(dtype=B.dtype)[_A,_A,:]*F;C=C+A.mlp_scale.to(dtype=C.dtype)[_A,_A,:]*A.mlp(A.mlp_norm(C)*A.ln_scale_factor,up_w,D)
		if A.dtg_gate is not _A:J=torch.sigmoid(A.dtg_gate(B.detach()));C=B+J*(C-B)
		return C,H
class GPT(nn.Module):
	def __init__(A,vocab_size,num_layers,model_dim,num_heads,num_kv_heads,mlp_mult,tie_embeddings,tied_embed_init_std,logit_softcap,rope_base,qk_gain_init,mtp_num_heads=0,mtp_loss_weight=.1,bigram_vocab_size=0,bigram_dim=128,xsa_last_n=0,rope_dims=0,ln_scale=_C,dtg=_C,ve_enabled=_C,ve_dim=128,ve_layers=_c,gated_attention=_C,value_residual=_C,recur_layers='',parallel_start_layer=-1,trigram=_C):
		Q=recur_layers;P=value_residual;O=xsa_last_n;N=bigram_vocab_size;M=mtp_num_heads;L=rope_base;K=tie_embeddings;J=mlp_mult;H=rope_dims;G=logit_softcap;F=num_kv_heads;E=num_heads;D=vocab_size;C=num_layers;B=model_dim;super().__init__();A._ve_target_dim=F*(B//E)
		if G<=_E:raise ValueError(f"logit_softcap must be positive, got {G}")
		A.recur_layer_indices=[int(A)for A in Q.split(_K)if A.strip()]if Q else[];A.parallel_start_layer=parallel_start_layer;A.recur_active=_C
		if A.recur_layer_indices:A.recur_scales=nn.ParameterList([nn.Parameter(torch.ones(B,dtype=torch.float32)*.5)for A in A.recur_layer_indices])
		else:A.recur_scales=nn.ParameterList()
		A.tie_embeddings=K;A.tied_embed_init_std=tied_embed_init_std;A.logit_softcap=G;A.value_residual=P;A.mtp_num_heads=M;A.mtp_loss_weight=mtp_loss_weight;A.tok_emb=nn.Embedding(D,B);A.bigram=BigramHashEmbedding(N,bigram_dim,B,trigram=trigram)if N>0 else _A;A.smear=SmearGate(B);A.num_encoder_layers=C//2;A.num_decoder_layers=C-A.num_encoder_layers;A.num_skip_weights=min(A.num_encoder_layers,A.num_decoder_layers);A.skip_weights=nn.Parameter(torch.ones(A.num_skip_weights,B,dtype=torch.float32));I=B//E;T=F*I;R=int(J*B);A.num_layers=C;A.qo_bank=nn.Parameter(torch.empty(2*C,B,B));A.kv_bank=nn.Parameter(torch.empty(2*C,T,B));A.mlp_up_bank=nn.Parameter(torch.empty(C,R,B));A.mlp_down_bank=nn.Parameter(torch.empty(C,B,R));A.blocks=nn.ModuleList([Block(B,E,F,J,L,qk_gain_init,layer_idx=A,ln_scale=ln_scale,dtg=dtg,gated_attention=gated_attention,value_residual=P)for A in range(C)])
		if H>0:
			I=B//E
			for S in A.blocks:S.attn.rope_dims=H;S.attn.rotary=Rotary(I,base=L,train_seq_len=1024,rope_dims=H)
		A.ve_layer_indices=[int(A)for A in ve_layers.split(_K)if A.strip()]if ve_enabled else[];U=A._ve_target_dim
		if A.ve_layer_indices:A.ve_shared=ValueEmbedding(D,ve_dim,U);A.ve_layer_scales=nn.ParameterList([nn.Parameter(torch.ones(1,dtype=torch.float32))for A in A.ve_layer_indices])
		else:A.ve_shared=_A;A.ve_layer_scales=nn.ParameterList()
		A.value_embeds=nn.ModuleList();A.final_norm=RMSNorm();A.lm_head=_A if K else CastedLinear(B,D,bias=_C)
		if A.lm_head is not _A:A.lm_head._zero_init=_B
		A.mtp_heads=nn.ModuleList([CastedLinear(B,D,bias=_C)for A in range(M)])
		for V in A.mtp_heads:V._zero_init=_B
		if O>0:
			for W in range(max(0,C-O),C):A.blocks[W].attn.use_xsa=_B
		A._init_weights()
	def _init_weights(A):
		if A.tie_embeddings:nn.init.normal_(A.tok_emb.weight,mean=_E,std=A.tied_embed_init_std)
		D=A.num_layers;E=_D/math.sqrt(2*D)
		for B in range(D):nn.init.orthogonal_(A.qo_bank.data[B],gain=_D);nn.init.zeros_(A.qo_bank.data[D+B]);nn.init.orthogonal_(A.kv_bank.data[B],gain=_D);nn.init.orthogonal_(A.kv_bank.data[D+B],gain=_D);nn.init.orthogonal_(A.mlp_up_bank.data[B],gain=_D);nn.init.zeros_(A.mlp_down_bank.data[B]);A.qo_bank.data[D+B].mul_(E);A.mlp_down_bank.data[B].mul_(E)
		for(F,C)in A.named_modules():
			if isinstance(C,nn.Linear):
				if getattr(C,'_zero_init',_C):nn.init.zeros_(C.weight)
				elif C.weight.ndim==2 and C.weight.shape[0]>=64 and C.weight.shape[1]>=64:nn.init.orthogonal_(C.weight,gain=_D)
	def _get_ve(A,layer_idx,input_ids,ve_cache=_A):
		D=input_ids;C=layer_idx;B=ve_cache
		if A.ve_shared is _A or C not in A.ve_layer_indices:return
		if B is not _A and _L not in B:B[_L]=A.ve_shared(D)
		E=B[_L]if B is not _A else A.ve_shared(D);F=A.ve_layer_indices.index(C);return E*A.ve_layer_scales[F].to(dtype=E.dtype)
	def _run_block(A,i,x,x0,input_ids,ve_cache,v0,n,recur_scale=_A):
		C=recur_scale;D=A._get_ve(i,input_ids,ve_cache);E=A.parallel_start_layer>=0 and i>=A.parallel_start_layer;B,F=A.blocks[i](x,x0,A.qo_bank[i],A.kv_bank[i],A.kv_bank[n+i],A.qo_bank[n+i],A.mlp_up_bank[i],A.mlp_down_bank[i],v_embed=D,v0=v0,parallel=E)
		if C is not _A:B=x+C.to(dtype=x.dtype)[_A,_A,:]*(B-x)
		return B,F
	def forward(A,input_ids,target_ids):
		N=target_ids;D=input_ids;G=A.num_layers;B=A.tok_emb(D)
		if A.bigram is not _A:B=B+A.bigram(D)
		B=F.rms_norm(B,(B.size(-1),));B=A.smear(B);H=B;E=_A;I=[];J={}
		for C in range(A.num_encoder_layers):
			B,O=A._run_block(C,B,H,D,J,E,G)
			if E is _A and O is not _A:E=O
			I.append(B)
			if A.recur_active and C in A.recur_layer_indices:U=A.recur_layer_indices.index(C);B,P=A._run_block(C,B,H,D,J,E,G,recur_scale=A.recur_scales[U])
		for C in range(A.num_decoder_layers):
			V=A.num_encoder_layers+C
			if I:B=B+A.skip_weights[C].to(dtype=B.dtype)[_A,_A,:]*I.pop()
			B,P=A._run_block(V,B,H,D,J,E,G)
		B=A.final_norm(B);Q=B.reshape(-1,B.size(-1));W=N.reshape(-1)
		if A.tie_embeddings:R=F.linear(Q,A.tok_emb.weight)
		else:
			if A.lm_head is _A:raise RuntimeError('lm_head is required when tie_embeddings=False')
			R=A.lm_head(Q)
		X=A.logit_softcap*torch.tanh(R/A.logit_softcap);K=F.cross_entropy(X.float(),W,reduction=_f)
		if A.training and A.mtp_num_heads>0 and A.mtp_loss_weight>_E:
			P,Y,Z=B.shape;L=B.new_zeros(());M=0
			for(S,a)in enumerate(A.mtp_heads):
				T=Y-(S+1)
				if T<=0:continue
				b=B[:,:T,:].reshape(-1,Z);c=N[:,S+1:].reshape(-1);d=a(b);e=A.logit_softcap*torch.tanh(d/A.logit_softcap);L=L+F.cross_entropy(e.float(),c,reduction=_f);M+=1
			if M>0:K=K+A.mtp_loss_weight*(L/M)
		return K
	def forward_logits(A,input_ids):
		D=input_ids;G=A.num_layers;B=A.tok_emb(D)
		if A.bigram is not _A:B=B+A.bigram(D)
		B=F.rms_norm(B,(B.size(-1),));B=A.smear(B);H=B;E=_A;I=[];J={}
		for C in range(A.num_encoder_layers):
			B,L=A._run_block(C,B,H,D,J,E,G)
			if E is _A and L is not _A:E=L
			I.append(B)
			if A.recur_active and C in A.recur_layer_indices:N=A.recur_layer_indices.index(C);B,O=A._run_block(C,B,H,D,J,E,G,recur_scale=A.recur_scales[N])
		for C in range(A.num_decoder_layers):
			P=A.num_encoder_layers+C
			if I:B=B+A.skip_weights[C].to(dtype=B.dtype)[_A,_A,:]*I.pop()
			B,O=A._run_block(P,B,H,D,J,E,G)
		B=A.final_norm(B)
		if A.tie_embeddings:M=F.linear(B,A.tok_emb.weight)
		else:M=A.lm_head(B)
		K=A.logit_softcap*torch.tanh(M/A.logit_softcap)
		if hasattr(A,'_eval_temperature')and A._eval_temperature!=_D:K=K/A._eval_temperature
		return K
_NG_PRIMES=np.array([np.uint64(36313),np.uint64(27191),np.uint64(51647),np.uint64(81929),np.uint64(131071),np.uint64(175447),np.uint64(209591)],dtype=np.uint64)
_PER_ORDER_CENTERS=np.array([4.5,4.2,3.8,3.5,3.2,3.],dtype=np.float64)
def _init_ngram_tables(args):
	A=args;B=A.ngram_order-A.ngram_min_order+1;E=[np.zeros((A.ngram_buckets,),dtype=np.uint32)for B in range(B)];F=[np.zeros((A.ngram_buckets,),dtype=np.uint32)for B in range(B)];G=np.uint64(A.ngram_buckets-1);C=os.environ.get('NGRAM_JM_LAMBDAS','')
	if C:D=np.array([float(A)for A in C.split(_K)],dtype=np.float64)
	else:D=np.ones(B,dtype=np.float64)/B
	H=[np.zeros((A.ngram_buckets,),dtype=np.uint32)for B in range(B)];I=[np.zeros((A.ngram_buckets,),dtype=np.uint32)for B in range(B)];return{_g:B,_h:E,_i:F,_w:H,_x:I,_y:G,_z:_NG_PRIMES,_A0:A.ngram_min_order,_A1:A.ngram_min_count,_A2:D,_O:0,_S:0,_T:0}
def _sync_ngram_tables(ng,device):
	A=ng
	for B in range(A[_g]):
		for(C,D)in[(_h,_w),(_i,_x)]:F=A[C][B]-A[D][B];E=torch.from_numpy(F.view(np.int32).copy()).to(device);dist.all_reduce(E,op=dist.ReduceOp.SUM);G=E.cpu().numpy().view(np.uint32);A[C][B]=A[D][B]+G;A[D][B]=A[C][B].copy()
def _ngram_score_segment(scored_nll,logits_slice,val_np,ws,s,wlen,ng,args,device):
	f=val_np;W=scored_nll;D=args;C=ng;g=W.cpu().numpy();O=np.exp(-g);E=len(g);h=np.arange(ws+s+1,ws+wlen+1,dtype=np.int64);I=C[_g];X=C[_h];Y=C[_i];i=C[_y];R=C[_z];q=C[_A0];j=C[_A1]
	if D.ngram_entropy:
		with torch.no_grad():k=F.log_softmax(logits_slice.float(),dim=-1);S=-(k.exp()*k).sum(dim=-1).cpu().numpy()
	else:S=_A
	G=[]
	for A in range(I):
		T=q+A-1;l=h>=T
		if not l.any():G.append(_A);continue
		H=np.nonzero(l)[0];Z=h[H];a=np.zeros(len(Z),dtype=np.uint64)
		for m in range(T):r=f[Z-(T-m)].astype(np.uint64);a^=r*R[m%len(R)]
		J=(a&i).astype(np.int64);t=f[Z].astype(np.uint64);K=((a^t*R[T%len(R)])&i).astype(np.int64);G.append((H,J,K))
	if D.ngram_jm:
		n=C[_A2];U=np.full((E,I),-_D)
		for A in range(I):
			if G[A]is _A:continue
			H,J,K=G[A];L=X[A][J].astype(np.float64);b=Y[A][K].astype(np.float64);B=L>=float(j)
			if B.any():u=H[B];c=np.minimum(b[B],L[B])/np.maximum(L[B],_D);U[u,A]=np.clip(c,_E,_D)
		v=U>=0;o=np.zeros(E);d=np.zeros(E)
		for A in range(I):
			V=v[:,A]
			if V.any():o[V]+=n[A]*U[V,A];d[V]+=n[A]
		M=np.full(E,-_D);e=d>0;M[e]=np.clip(o[e]/d[e],_E,_D);P=np.full(E,-1,dtype=np.int32)
		for A in range(I-1,-1,-1):w=(P<0)&(U[:,A]>=0);P[w]=A
	else:
		M=np.full(E,-_D);P=np.full(E,-1,dtype=np.int32)
		for A in range(I-1,-1,-1):
			if G[A]is _A:continue
			H,J,K=G[A];L=X[A][J].astype(np.float64);b=Y[A][K].astype(np.float64);B=L>=float(j);Q=B&(M[H]<0)
			if Q.any():p=H[Q];c=np.minimum(b[Q],L[Q])/np.maximum(L[Q],_D);M[p]=np.clip(c,_E,_D);P[p]=A
	B=M>=0
	if B.any():
		if D.ngram_entropy and S is not _A:
			if D.ngram_per_order_centers:x=P[B];y=_PER_ORDER_CENTERS[np.clip(x,0,len(_PER_ORDER_CENTERS)-1)];N=D.ngram_ent_base+D.ngram_ent_range/(_D+np.exp(-D.ngram_ent_scale*(S[B]-y)))
			else:N=D.ngram_ent_base+D.ngram_ent_range/(_D+np.exp(-D.ngram_ent_scale*(S[B]-D.ngram_ent_thresh)))
		else:N=np.full(int(B.sum()),D.ngram_alpha)
		if D.ngram_depth_signal:z=np.float64(min(_D,C[_O]*D.ngram_depth_scale));N=N*z
		O[B]=(_D-N)*O[B]+N*M[B]
	C[_S]=C.get(_S,0)+int((M>=0).sum());C[_T]=C.get(_T,0)+E;O=np.clip(O,1e-12,_D);W=torch.from_numpy(-np.log(O)).to(dtype=torch.float64,device=device)
	for A in range(I):
		if G[A]is _A:continue
		H,J,K=G[A];np.add.at(X[A],J,1);np.add.at(Y[A],K,1)
	C[_O]=C.get(_O,0)+E;return W
def eval_val_sliding(args,base_model,rank,world_size,device,val_tokens,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut,stride,batch_seqs=32,eval_seq_len=_A):
	Z=batch_seqs;Y=stride;O=val_tokens;N=world_size;M=base_model;K=rank;C=device;A=args;I=eval_seq_len or A.train_seq_len;P=O.numel()-1;a=[A for A in range(0,P,Y)if min(A+I,P)-A>=1];b=len(a);o=b*K//N;p=b*(K+1)//N;c=a[o:p];Q=torch.zeros((),device=C,dtype=torch.float64);L=torch.zeros((),device=C,dtype=torch.float64);R=torch.zeros((),device=C,dtype=torch.float64);J=A.ngram_enabled;E=_A;d=_A
	if K==0:print(f"ngram_eval: enabled={J} order={A.ngram_order} min_order={A.ngram_min_order} buckets={A.ngram_buckets} entropy={A.ngram_entropy} jm={A.ngram_jm} per_order_centers={A.ngram_per_order_centers} depth_signal={A.ngram_depth_signal}",flush=_B)
	if J:d=O.cpu().numpy();E=_init_ngram_tables(A)
	M.eval();q=torch.compile(M.forward_logits,dynamic=_C,fullgraph=_B);e=0
	with torch.inference_mode():
		for f in range(0,len(c),Z):
			S=c[f:f+Z];T=len(S);U=torch.zeros(T,I,dtype=torch.int64,device=C);V=torch.zeros(T,I,dtype=torch.int64,device=C);g=[]
			for(D,G)in enumerate(S):h=min(G+I,P);B=h-G;g.append(B);i=O[G:h+1].to(dtype=torch.int64,device=C);U[D,:B]=i[:-1];V[D,:B]=i[1:]
			with torch.autocast(device_type=_I,dtype=torch.bfloat16):W=q(U)
			r=F.cross_entropy(W.reshape(-1,W.size(-1)).float(),V.reshape(-1),reduction='none').reshape(T,I)
			for(D,G)in enumerate(S):
				B=g[D];H=0 if G==0 else max(B-Y,0);X=r[D,H:B].to(torch.float64)
				if J:X=_ngram_score_segment(X,W[D,H:B],d,G,H,B,E,A,C)
				Q+=X.sum();L+=float(B-H);j=V[D,H:B];s=U[D,H:B];k=base_bytes_lut[j].to(torch.float64);k+=(has_leading_space_lut[j]&~is_boundary_token_lut[s]).to(torch.float64);R+=k.sum()
			e+=1
			if J and dist.is_available()and dist.is_initialized()and N>1:
				if e%A.ngram_sync_interval==0:_sync_ngram_tables(E,C)
	if J and K==0:l=E.get(_S,0);m=E.get(_T,0);t=1e2*l/max(m,1);print(f"ngram_stats: matches={l}/{m} ({t:.1f}%) tokens_in_cache={E.get(_O,0)}",flush=_B)
	if dist.is_available()and dist.is_initialized():dist.all_reduce(Q,op=dist.ReduceOp.SUM);dist.all_reduce(L,op=dist.ReduceOp.SUM);dist.all_reduce(R,op=dist.ReduceOp.SUM)
	n=(Q/L).item();u=n/math.log(2.);v=L.item()/R.item();M.train();return n,u*v
def eval_val_sliding_ttt(args,base_model,rank,world_size,device,val_tokens,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut,stride,batch_seqs=32,log0=print):
	s=batch_seqs;W=stride;V=val_tokens;O=world_size;N=rank;I=log0;H=device;B=base_model;A=args;G=A.train_seq_len;P=V.numel()-1;Q=A.ttt_chunk_tokens;t=[A for A in range(0,P,W)if min(A+G,P)-A>=W or A==0];K=(P+Q-1)//Q;u=[[]for A in range(K)]
	for E in t:Z=min(E+G,P);C=Z-E;L=0 if E==0 else max(C-W,0);AG=E+L;D=min(AG//Q,K-1);u[D].append(E)
	I(f"ttt_sliding:start chunks={K} chunk_tokens={Q} total_windows={len(t)} stride={W} ttt_lr={A.ttt_lr} ttt_epochs={A.ttt_epochs} freeze_blocks={A.ttt_freeze_blocks}");R=torch.zeros((),device=H,dtype=torch.float64);J=torch.zeros((),device=H,dtype=torch.float64);S=torch.zeros((),device=H,dtype=torch.float64);v=set(range(min(A.ttt_freeze_blocks,len(B.blocks))));T=[]
	for(AH,X)in B.named_parameters():
		w=_C
		for U in v:
			if f"blocks.{U}."in AH:w=_B;break
		if w:X.requires_grad_(_C)
		else:X.requires_grad_(_B);T.append(X)
	I(f"ttt_sliding:params unfrozen={sum(A.numel()for A in T)} frozen={sum(A.numel()for A in B.parameters()if not A.requires_grad)}")
	if A.ttt_optimizer=='adamw':a=torch.optim.AdamW(T,lr=A.ttt_adamw_lr,weight_decay=_E,betas=(.9,.95));I(f"ttt_optimizer:AdamW lr={A.ttt_adamw_lr}")
	else:a=torch.optim.SGD(T,lr=A.ttt_lr,momentum=A.ttt_momentum,nesterov=A.ttt_nesterov);I(f"ttt_optimizer:SGD lr={A.ttt_lr} nesterov={A.ttt_nesterov}")
	if A.ttt_temperature!=_D:B._eval_temperature=A.ttt_temperature;I(f"ttt_temperature:{A.ttt_temperature}")
	else:B._eval_temperature=_D
	AI=torch.compile(B.forward_logits,dynamic=_C,fullgraph=_B);x=_E;y=_E;z=_E;e=A.ngram_enabled;f=_A;A0=_A
	if e:A0=V.cpu().numpy();f=_init_ngram_tables(A);I(f"ngram:enabled orders={A.ngram_min_order}-{A.ngram_order} entropy={A.ngram_entropy} alpha={A.ngram_alpha} buckets={A.ngram_buckets} min_count={A.ngram_min_count}")
	A1=time.perf_counter()
	for D in range(K):
		b=u[D]
		if not b:continue
		g=D*Q;AJ=min((D+1)*Q,P);AK=len(b)*N//O;AL=len(b)*(N+1)//O;A2=b[AK:AL];B.eval();A3=0
		with torch.inference_mode():
			for U in range(0,len(A2),s):
				h=A2[U:U+s];i=len(h);j=torch.zeros(i,G,dtype=torch.int64,device=H);k=torch.zeros(i,G,dtype=torch.int64,device=H);A4=[]
				for(M,E)in enumerate(h):Z=min(E+G,P);C=Z-E;A4.append(C);A5=V[E:Z+1].to(dtype=torch.int64,device=H);j[M,:C]=A5[:-1];k[M,:C]=A5[1:]
				with torch.autocast(device_type=_I,dtype=torch.bfloat16):l=AI(j)
				AM=F.cross_entropy(l.reshape(-1,l.size(-1)).float(),k.reshape(-1),reduction='none').reshape(i,G)
				for(M,E)in enumerate(h):
					C=A4[M];L=0 if E==0 else max(C-W,0);m=AM[M,L:C].to(torch.float64)
					if e:m=_ngram_score_segment(m,l[M,L:C],A0,E,L,C,f,A,H)
					R+=m.sum();J+=float(C-L);A6,AN=k[M,L:C],j[M,L:C];A7=base_bytes_lut[A6].to(torch.float64);A7+=(has_leading_space_lut[A6]&~is_boundary_token_lut[AN]).to(torch.float64);S+=A7.sum()
				A3+=1
				if e and dist.is_available()and dist.is_initialized()and O>1:
					if A3%A.ngram_sync_interval==0:_sync_ngram_tables(f,H)
		AO=D==K-1
		if not AO and A.ttt_epochs>0:
			if A.ttt_difficulty and D>0:
				AP=R.item()-x;n=J.item()-z;A8=S.item()-y
				if n>0 and A8>0:c=AP/n/math.log(2.)*(n/A8)
				else:c=1.12
				if c>A.ttt_hard_threshold:Y=A.ttt_hard_epochs
				elif c<A.ttt_easy_threshold:Y=A.ttt_easy_epochs
				else:Y=A.ttt_epochs
				if N==0 and D%100==0:I(f"  difficulty: chunk={D} bpb={c:.4f} epochs={Y}")
			else:Y=A.ttt_epochs
			x=R.item();z=J.item();y=S.item();B.train();o=(AJ-g)//G
			if o>0:
				A9=A.ttt_adamw_lr if A.ttt_optimizer=='adamw'else A.ttt_lr;p=A9*.5*(_D+math.cos(math.pi*D/max(K-1,1)))
				if A.ttt_lr_floor>0:p=max(p,A9*A.ttt_lr_floor)
				for AQ in a.param_groups:AQ[_H]=p
				q=o*N//O;AR=o*(N+1)//O;AA=AR-q
				for Ae in range(Y):
					for AB in range(0,AA,A.ttt_batch_seqs):
						AS=min(AB+A.ttt_batch_seqs,AA);AT=q+AB;AU=g+AT*G;AC=g+(q+AS)*G+1
						if AC>V.numel():continue
						AD=V[AU:AC].to(device=H,dtype=torch.int64);AV=AD[:-1].reshape(-1,G);AW=AD[1:].reshape(-1,G);a.zero_grad(set_to_none=_B)
						with torch.autocast(device_type=_I,dtype=torch.bfloat16):AX=B(AV,AW)
						AX.backward()
						if O>1:
							d=[A.grad for A in T if A.grad is not _A]
							if d:
								AE=torch._utils._flatten_dense_tensors(d);dist.all_reduce(AE,op=dist.ReduceOp.AVG)
								for(AY,AZ)in zip(d,torch._utils._unflatten_dense_tensors(AE,d)):AY.copy_(AZ)
						if A.ttt_q_only:
							Aa=len(B.blocks)
							if B.qo_bank.grad is not _A:
								B.qo_bank.grad[Aa:].zero_()
								for U in v:B.qo_bank.grad[U].zero_()
							if B.kv_bank.grad is not _A:B.kv_bank.grad.zero_()
							if B.mlp_up_bank.grad is not _A:B.mlp_up_bank.grad.zero_()
							if B.mlp_down_bank.grad is not _A:B.mlp_down_bank.grad.zero_()
						torch.nn.utils.clip_grad_norm_(T,A.ttt_grad_clip);a.step()
		if N==0 and(D%10==0 or D==K-1):Ab=time.perf_counter()-A1;Ac=R.item()/max(J.item(),1);Ad=Ac/math.log(2.)*(J.item()/max(S.item(),1))if J.item()>0 else _E;I(f"  ttt_chunk [{D+1}/{K}] bpb={Ad:.6f} time={Ab:.1f}s")
	if dist.is_available()and dist.is_initialized():dist.all_reduce(R,op=dist.ReduceOp.SUM);dist.all_reduce(J,op=dist.ReduceOp.SUM);dist.all_reduce(S,op=dist.ReduceOp.SUM)
	r=(R/J).item();AF=r/math.log(2.)*(J.item()/S.item())
	for X in B.parameters():X.requires_grad_(_B)
	B.eval();I(f"ttt_sliding:done val_loss={r:.6f} val_bpb={AF:.6f} elapsed={time.perf_counter()-A1:.1f}s");return r,AF
def generate_autoregressive_calib(model,device,num_seqs=64,seq_len=2048,vocab_size=1024,temperature=.8,batch_size=8,seed=42):
	F=batch_size;E=num_seqs;D=device;C=model;C.eval();B=torch.Generator(device=D);B.manual_seed(seed);G=[]
	with torch.inference_mode(),torch.autocast(device_type=_I,dtype=torch.bfloat16):
		for J in range(0,E,F):
			H=min(F,E-J);A=torch.randint(0,vocab_size,(H,1),device=D,generator=B)
			for O in range(seq_len-1):K=C.forward_logits(A);L=K[:,-1,:];M=torch.softmax(L/temperature,dim=-1);N=torch.multinomial(M,1,generator=B);A=torch.cat([A,N],dim=1)
			for I in range(H):G.append(A[I:I+1])
	return G
def collect_hessians_from_tokens(hessian_model,token_seqs,device,gptq_damp=.01):
	G=device;F=token_seqs;C=hessian_model;A={};H=[]
	for(B,D)in C.named_modules():
		if isinstance(D,CastedLinear):
			I=B+'.weight';J=D.weight.shape[1];A[I]=torch.zeros(J,J,dtype=torch.float32,device=_N)
			def M(pname):
				def B(module,input,output):
					B=input[0].detach().float()
					if B.ndim==3:B=B.reshape(-1,B.shape[-1])
					A[pname]+=(B.T@B).cpu()
				return B
			E=D.register_forward_hook(M(I));H.append(E)
	C.eval()
	with torch.inference_mode(),torch.autocast(device_type=_I,dtype=torch.bfloat16):
		for K in F:N=K[:,:-1].to(G);O=K[:,1:].to(G);C(N,O)
	for E in H:E.remove()
	P=len(F)
	for B in A:L=A[B];L/=P;A[B]=L
	return A
def _hadamard_matrix(n):
	A=torch.ones(1,1)
	while A.size(0)<n:A=torch.cat([torch.cat([A,A],1),torch.cat([A,-A],1)],0)
	return A/n**.5
def _apply_hadamard_rotation(sd,model_dim,log0):
	B=model_dim;A=sd;F=_hadamard_matrix(B);E=0
	for C in list(A.keys()):
		D=A[C]
		if D.ndim!=2:continue
		H,G=D.shape;I=_classify_param(C)
		if I not in(_U,_V):continue
		if G==B:A[C]=D.float()@F.T;E+=1
		elif H==B and G!=B:A[C]=F@D.float();E+=1
	log0(f"hadamard:rotated {E} weight matrices (dim={B})");return A
def _classify_param(name):
	B='.mlp.';A=name
	if'tok_emb'in A or'lm_head'in A:return'embed'
	if B in A:return _U
	if'.attn.'in A or'.proj.'in A and B not in A:return _V
	return'other'
def quantize_int6_per_row(t,clip_range=31):
	A=clip_range;B=t.float()
	if B.ndim==2:
		E,F,G=_A,_A,float(_j)
		for H in[.999,.9995,.9999,.99999,_D]:
			if H<_D:I=torch.quantile(B.abs(),H,dim=1)
			else:I=B.abs().amax(dim=1)
			D=(I/A).clamp_min(_D/A).to(torch.float16);C=torch.clamp(torch.round(B/D.float()[:,_A]),-A,A).to(torch.int8);M=C.float()*D.float()[:,_A];J=(B-M).pow(2).mean().item()
			if J<G:E,F,G=C,D,J
		return E,F
	K=B.abs().max().item();L=torch.tensor(K/A if K>0 else _D,dtype=torch.float16);C=torch.clamp(torch.round(B/L.float()),-A,A).to(torch.int8);return C,L
def quantize_int6_gptq(weight,hessian=_A,clip_range=31,block_size=128,gptq_damp=.01):
	Q=block_size;P=hessian;G=clip_range;E=weight.float()
	if E.ndim!=2 or P is _A:return _quantize_int6_percentile(E,G)
	R,H=E.shape;B=P.float().clone();L=torch.diag(B)==0;B[L,L]=1;g=gptq_damp*torch.mean(torch.diag(B));B[torch.arange(H),torch.arange(H)]+=g;I=torch.argsort(torch.diag(B),descending=_B);h=torch.argsort(I);J=E[:,I].clone();J[:,L[I]]=0;B=B[I][:,I];F=torch.linalg.cholesky(B);F=torch.cholesky_inverse(F);F=torch.linalg.cholesky(F,upper=_B);K=_A;S=_A;T=float(_j)
	for U in[.999,.9995,.9999,.99999,_D]:
		if U<_D:V=torch.quantile(E.abs(),U,dim=1)
		else:V=E.abs().amax(dim=1)
		W=(V/G).clamp_min(_D/G).to(torch.float16);M=W.float();N=torch.zeros_like(J,dtype=torch.int8);X=J.clone()
		for D in range(0,H,Q):
			A=min(D+Q,H);O=A-D;Y=X[:,D:A].clone();Z=torch.zeros(R,O,dtype=torch.int8);a=torch.zeros(R,O);b=F[D:A,D:A]
			for C in range(O):c=Y[:,C];i=b[C,C];d=torch.clamp(torch.round(c/M),-G,G).to(torch.int8);Z[:,C]=d;e=(c-d.float()*M)/i;Y[:,C:]-=e.unsqueeze(1)*b[C,C:].unsqueeze(0);a[:,C]=e
			N[:,D:A]=Z
			if A<H:X[:,A:]-=a@F[D:A,A:]
		j=N.float()*M[:,_A];f=(J-j).pow(2).mean().item()
		if f<T:K,S,T=N,W,f
	K=K[:,h];return K,S
def _quantize_int6_percentile(t32,clip_range=31):
	B=clip_range;A=t32
	if A.ndim==2:
		E,F,G=_A,_A,float(_j)
		for H in[.999,.9995,.9999,.99999,_D]:
			if H<_D:I=torch.quantile(A.abs(),H,dim=1)
			else:I=A.abs().amax(dim=1)
			D=(I/B).clamp_min(_D/B).to(torch.float16);C=torch.clamp(torch.round(A/D.float()[:,_A]),-B,B).to(torch.int8);M=C.float()*D.float()[:,_A];J=(A-M).pow(2).mean().item()
			if J<G:E,F,G=C,D,J
		return E,F
	K=A.abs().max().item();L=torch.tensor(K/B if K>0 else _D,dtype=torch.float16);C=torch.clamp(torch.round(A/L.float()),-B,B).to(torch.int8);return C,L
def _unbank_state_dict(sd,num_layers):
	B={};D=num_layers
	for(E,C)in sd.items():
		if E==_W:
			for A in range(D):B[f"blocks.{A}.attn.c_q.weight"]=C[A];B[f"blocks.{A}.attn.proj.weight"]=C[D+A]
		elif E==_X:
			for A in range(D):B[f"blocks.{A}.attn.c_k.weight"]=C[A];B[f"blocks.{A}.attn.c_v.weight"]=C[D+A]
		elif E==_Y:
			for A in range(D):B[f"blocks.{A}.mlp.fc.weight"]=C[A]
		elif E==_Z:
			for A in range(D):B[f"blocks.{A}.mlp.proj.weight"]=C[A]
		else:B[E]=C
	return B
def _rebank_state_dict(sd,num_layers,template_sd):
	F=template_sd;A=sd;E={};C=num_layers;G=[_A]*(2*C);H=[_A]*(2*C);O=[_A]*C;P=[_A]*C;D=set()
	for B in range(C):
		I=f"blocks.{B}.attn.c_q.weight"
		if I in A:G[B]=A[I];D.add(I)
		J=f"blocks.{B}.attn.proj.weight"
		if J in A:G[C+B]=A[J];D.add(J)
		K=f"blocks.{B}.attn.c_k.weight"
		if K in A:H[B]=A[K];D.add(K)
		L=f"blocks.{B}.attn.c_v.weight"
		if L in A:H[C+B]=A[L];D.add(L)
		M=f"blocks.{B}.mlp.fc.weight"
		if M in A:O[B]=A[M];D.add(M)
		N=f"blocks.{B}.mlp.proj.weight"
		if N in A:P[B]=A[N];D.add(N)
	E[_W]=torch.stack(G).to(dtype=F[_W].dtype);E[_X]=torch.stack(H).to(dtype=F[_X].dtype);E[_Y]=torch.stack(O).to(dtype=F[_Y].dtype);E[_Z]=torch.stack(P).to(dtype=F[_Z].dtype)
	for(Q,R)in A.items():
		if Q not in D:E[Q]=R
	return E
class _HessianAttn(nn.Module):
	def __init__(A,dim,num_heads,num_kv_heads,rope_base,qk_gain_init):D=num_kv_heads;C=num_heads;B=dim;super().__init__();A.num_heads,A.num_kv_heads=C,D;A.head_dim=B//C;E=D*A.head_dim;A.c_q=CastedLinear(B,B,bias=_C);A.c_k=CastedLinear(B,E,bias=_C);A.c_v=CastedLinear(B,E,bias=_C);A.proj=CastedLinear(B,B,bias=_C);A.q_gain=nn.Parameter(torch.full((C,),qk_gain_init,dtype=torch.float32));A.rope_dims=0;A.rotary=Rotary(A.head_dim,base=rope_base,train_seq_len=1024);A.use_xsa=_C
	def _xsa_efficient(K,y,v):A,B,C,D=y.shape;E=v.size(-2);I=C//E;G=y.reshape(A,B,E,I,D);H=F.normalize(v,dim=-1).unsqueeze(-2);J=(G*H).sum(dim=-1,keepdim=_B)*H;return(G-J).reshape(A,B,C,D)
	def forward(A,x,v_embed=_A):
		I=v_embed;G,E,L=x.shape;B=A.c_q(x).reshape(G,E,A.num_heads,A.head_dim);C=A.c_k(x).reshape(G,E,A.num_kv_heads,A.head_dim);D=A.c_v(x)
		if I is not _A:D=D+I
		D=D.reshape(G,E,A.num_kv_heads,A.head_dim);B=F.rms_norm(B,(B.size(-1),));C=F.rms_norm(C,(C.size(-1),));J,K=A.rotary(E,x.device,B.dtype);B=apply_rotary_emb(B,J,K,A.rope_dims);C=apply_rotary_emb(C,J,K,A.rope_dims);B=B*A.q_gain.to(dtype=B.dtype)[_A,_A,:,_A]
		if HAS_FA3:H=flash_attn_3_func(B,C,D,causal=_B)
		else:M=B.transpose(1,2);N=C.transpose(1,2);O=D.transpose(1,2);H=F.scaled_dot_product_attention(M,N,O,is_causal=_B).transpose(1,2)
		if A.use_xsa:H=A._xsa_efficient(H,D)
		return A.proj(H.reshape(G,E,L))
class _HessianMLP(nn.Module):
	def __init__(B,dim,mlp_mult):C=mlp_mult;A=dim;super().__init__();B.fc=CastedLinear(A,int(C*A),bias=_C);B.proj=CastedLinear(int(C*A),A,bias=_C)
	def forward(A,x):return A.proj(F.leaky_relu(A.fc(x),negative_slope=.5).square())
class _HessianBlock(nn.Module):
	def __init__(A,dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init,layer_idx=0,ln_scale=_C):B=dim;super().__init__();A.attn_norm=RMSNorm();A.mlp_norm=RMSNorm();A.attn=_HessianAttn(B,num_heads,num_kv_heads,rope_base,qk_gain_init);A.mlp=_HessianMLP(B,mlp_mult);A.attn_scale=nn.Parameter(torch.ones(B,dtype=torch.float32));A.mlp_scale=nn.Parameter(torch.ones(B,dtype=torch.float32));A.resid_mix=nn.Parameter(torch.stack((torch.ones(B),torch.zeros(B))).float());A.ln_scale_factor=_D/math.sqrt(layer_idx+1)if ln_scale else _D
	def forward(A,x,x0,v_embed=_A):D=A.resid_mix.to(dtype=x.dtype);C=D[0][_A,_A,:]*x+D[1][_A,_A,:]*x0;E=A.attn(A.attn_norm(C)*A.ln_scale_factor,v_embed=v_embed);B=C+A.attn_scale.to(dtype=C.dtype)[_A,_A,:]*E;B=B+A.mlp_scale.to(dtype=B.dtype)[_A,_A,:]*A.mlp(A.mlp_norm(B)*A.ln_scale_factor);return B
class _HessianGPT(nn.Module):
	def __init__(A,vocab_size,num_layers,model_dim,num_heads,num_kv_heads,mlp_mult,tie_embeddings,logit_softcap,rope_base,qk_gain_init,bigram_vocab_size=0,bigram_dim=128,xsa_last_n=0,rope_dims=0,ln_scale=_C,ve_enabled=_C,ve_dim=128,ve_layers=_c,trigram=_C):
		K=xsa_last_n;J=bigram_vocab_size;I=rope_base;H=tie_embeddings;G=num_kv_heads;F=rope_dims;E=num_heads;D=vocab_size;C=num_layers;B=model_dim;super().__init__();A.tie_embeddings=H;A.logit_softcap=logit_softcap;A.num_layers=C;A.tok_emb=nn.Embedding(D,B);A.bigram=BigramHashEmbedding(J,bigram_dim,B,trigram=trigram)if J>0 else _A;A.smear=SmearGate(B);A.num_encoder_layers=C//2;A.num_decoder_layers=C-A.num_encoder_layers;A.num_skip_weights=min(A.num_encoder_layers,A.num_decoder_layers);A.skip_weights=nn.Parameter(torch.ones(A.num_skip_weights,B,dtype=torch.float32));A.blocks=nn.ModuleList([_HessianBlock(B,E,G,mlp_mult,I,qk_gain_init,layer_idx=A,ln_scale=ln_scale)for A in range(C)])
		if F>0:
			M=B//E
			for L in A.blocks:L.attn.rope_dims=F;L.attn.rotary=Rotary(M,base=I,train_seq_len=1024,rope_dims=F)
		if K>0:
			for N in range(max(0,C-K),C):A.blocks[N].attn.use_xsa=_B
		O=G*(B//E);A.ve_layer_indices=[int(A)for A in ve_layers.split(_K)if A.strip()]if ve_enabled else[]
		if A.ve_layer_indices:A.ve_shared=ValueEmbedding(D,ve_dim,O);A.ve_layer_scales=nn.ParameterList([nn.Parameter(torch.ones(1,dtype=torch.float32))for A in A.ve_layer_indices])
		else:A.ve_shared=_A;A.ve_layer_scales=nn.ParameterList()
		A.final_norm=RMSNorm();A.lm_head=_A if H else CastedLinear(B,D,bias=_C)
	def _get_ve(A,layer_idx,input_ids,ve_cache):
		C=layer_idx;B=ve_cache
		if A.ve_shared is _A or C not in A.ve_layer_indices:return
		if _L not in B:B[_L]=A.ve_shared(input_ids)
		D=A.ve_layer_indices.index(C);return B[_L]*A.ve_layer_scales[D].to(dtype=B[_L].dtype)
	def forward(B,input_ids,target_ids):
		D=input_ids;A=B.tok_emb(D)
		if B.bigram is not _A:A=A+B.bigram(D)
		A=F.rms_norm(A,(A.size(-1),));A=B.smear(A);H=A;E=[];I={}
		for C in range(B.num_encoder_layers):G=B._get_ve(C,D,I);A=B.blocks[C](A,H,v_embed=G);E.append(A)
		for C in range(B.num_decoder_layers):
			J=B.num_encoder_layers+C
			if E:A=A+B.skip_weights[C].to(dtype=A.dtype)[_A,_A,:]*E.pop()
			G=B._get_ve(J,D,I);A=B.blocks[J](A,H,v_embed=G)
		A=B.final_norm(A);K=A.reshape(-1,A.size(-1));L=target_ids.reshape(-1);M=F.linear(K,B.tok_emb.weight)if B.tie_embeddings else B.lm_head(K);N=B.logit_softcap*torch.tanh(M/B.logit_softcap);return F.cross_entropy(N.float(),L,reduction=_f)
def collect_hessians(hessian_model,train_loader,args,device,grad_accum_steps,num_batches=256):
	G=num_batches;C=hessian_model;A={};H=[]
	for(D,E)in C.named_modules():
		if isinstance(E,CastedLinear):
			I=D+'.weight';J=E.weight.shape[1];A[I]=torch.zeros(J,J,dtype=torch.float32,device=_N)
			def K(pname):
				def B(module,input,output):
					B=input[0].detach().float()
					if B.ndim==3:B=B.reshape(-1,B.shape[-1])
					A[pname]+=(B.T@B).cpu()
				return B
			F=E.register_forward_hook(K(I));H.append(F)
	C.eval()
	with torch.inference_mode(),torch.autocast(device_type=_I,dtype=torch.bfloat16):
		for O in range(G):L,M=train_loader.next_batch(args.train_batch_tokens,args.train_seq_len,grad_accum_steps);C(L,M)
	for F in H:F.remove()
	for D in A:B=A[D];B/=G;N=gptq_damp*torch.diag(B).mean().clamp_min(1e-06);B+=N*torch.eye(B.shape[0]);A[D]=B
	C.train();return A
def mixed_quantize_int6(state_dict,int6_cats,hessians=_A,clip_ranges=_A,gptq_damp=.01):
	J=clip_ranges;I=hessians;H=state_dict;K=max((int(A.split('.')[1])for A in H if A.startswith('blocks.')),default=0)+1;P=set(range(K-2,K));C={};D={}
	for(A,M)in H.items():
		B=M.detach().cpu().contiguous();N=_classify_param(A)
		if not B.is_floating_point()or B.numel()<=65536:C[A]=B.to(torch.float16)if B.is_floating_point()else B;D[A]=_R;continue
		if any(B in A for B in CONTROL_TENSOR_NAME_PATTERNS):C[A]=B.float();D[A]=_A3;continue
		if N in int6_cats and B.ndim>=1:
			E=J.get(A,31)if J else 31;L=I.get(A)if I else _A
			if L is not _A:F,G=quantize_int6_gptq(B,hessian=L,clip_range=E,gptq_damp=gptq_damp)
			else:F,G=quantize_int6_per_row(B,clip_range=E)
			C[A+_a]=F;C[A+_b]=G;O={15:'int5',31:'int6',63:'int7'}.get(E,f"int_cr{E}");D[A]={_k:O}
		else:F,G=quantize_float_tensor(B);C[A+_a]=F;C[A+_b]=G;D[A]={_k:'int8'}
	return C,D
def dequantize_mixed_int6(result,meta,template_sd):
	F=result;B={}
	for(A,I)in template_sd.items():
		H=meta.get(A)
		if H is _A:continue
		C=I.dtype
		if H in(_R,_A3,'passthrough_fp16'):
			D=F[A]
			if D.dtype==torch.float16 and C in(torch.float32,torch.bfloat16):D=D.to(C)
			B[A]=D;continue
		E,G=F[A+_a],F[A+_b]
		if G.ndim>0:B[A]=(E.float()*G.float().view(E.shape[0],*[1]*(E.ndim-1))).to(C)
		else:B[A]=(E.float()*float(G.item())).to(C)
	return B
def main():
	BQ='final_model.int6.ptz';BP='final_model.pt';BO='mtp_heads';BN='WORLD_SIZE';U='base_lr';x=Path(__file__).read_text(encoding=_M);A=Hyperparameters();Q='RANK'in os.environ and BN in os.environ;M=int(os.environ.get('RANK',_F));I=int(os.environ.get(BN,_J));BR=int(os.environ.get('LOCAL_RANK',_F))
	if I<=0:raise ValueError(f"WORLD_SIZE must be positive, got {I}")
	if 8%I!=0:raise ValueError(f"WORLD_SIZE={I} must divide 8 so grad_accum_steps stays integral")
	L=8//I;Aj=_D/L
	if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
	E=torch.device(_I,BR);torch.cuda.set_device(E)
	if Q:dist.init_process_group(backend='nccl',device_id=E);dist.barrier()
	y=M==0;torch.backends.cuda.matmul.allow_tf32=_B;torch.backends.cudnn.allow_tf32=_B;from torch.backends.cuda import enable_cudnn_sdp as Ak,enable_flash_sdp as Al,enable_math_sdp as Am,enable_mem_efficient_sdp as An
	if HAS_FA3:Ak(_C);Al(_B);An(_C);Am(_C)
	else:Ak(_B);Al(_B);An(_B);Am(_B)
	z=_A
	if y:os.makedirs('logs',exist_ok=_B);z=f"logs/{A.run_id}.txt";print(z)
	def C(msg,console=_B):
		if not y:return
		if console:print(msg)
		if z is not _A:
			with open(z,'a',encoding=_M)as A:print(msg,file=A)
	C(x,console=_C);C('='*100,console=_C);C(f"Running Python {sys.version}",console=_C);C(f"Running PyTorch {torch.__version__}",console=_C);C(subprocess.run(['nvidia-smi'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=_B,check=_C).stdout,console=_C);C('='*100,console=_C);random.seed(A.seed);np.random.seed(A.seed);torch.manual_seed(A.seed);torch.cuda.manual_seed_all(A.seed)
	if not A.tokenizer_path.endswith('.model'):raise ValueError(f"Script only setup for SentencePiece .model file: {A.tokenizer_path}")
	A8=spm.SentencePieceProcessor(model_file=A.tokenizer_path)
	if int(A8.vocab_size())!=A.vocab_size:raise ValueError(f"VOCAB_SIZE={A.vocab_size} does not match tokenizer vocab_size={int(A8.vocab_size())}")
	Ao=Path(A.data_path).resolve();BS=len(list(Ao.glob(_l)));A9=A.eval_seq_len if A.eval_seq_len>0 else A.train_seq_len;BT=max(A.train_seq_len,A9);R=load_validation_tokens(A.val_files,BT);V,W,X=build_sentencepiece_luts(A8,A.vocab_size,E);C(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={A.tokenizer_path}");C(f"train_loader:dataset:{Ao.name} train_shards:{BS}");C(f"val_loader:shards pattern={A.val_files} tokens:{R.numel()-1}");CastedLinear._qat_enabled=A.qat_enabled;B=GPT(vocab_size=A.vocab_size,num_layers=A.num_layers,model_dim=A.model_dim,num_heads=A.num_heads,num_kv_heads=A.num_kv_heads,mlp_mult=A.mlp_mult,tie_embeddings=A.tie_embeddings,tied_embed_init_std=A.tied_embed_init_std,logit_softcap=A.logit_softcap,rope_base=A.rope_base,qk_gain_init=A.qk_gain_init,mtp_num_heads=A.mtp_num_heads,mtp_loss_weight=A.mtp_loss_weight,bigram_vocab_size=A.bigram_vocab_size,bigram_dim=A.bigram_dim,xsa_last_n=A.xsa_last_n,rope_dims=A.rope_dims,ln_scale=A.ln_scale,dtg=A.dtg_enabled,ve_enabled=A.ve_enabled,ve_dim=A.ve_dim,ve_layers=A.ve_layers,gated_attention=A.gated_attention,value_residual=A.value_residual,recur_layers=A.recur_layers,parallel_start_layer=A.parallel_start_layer,trigram=A.trigram_enabled).to(E).bfloat16();B.qo_bank.data=B.qo_bank.data.float();B.kv_bank.data=B.kv_bank.data.float();B.mlp_up_bank.data=B.mlp_up_bank.data.float();B.mlp_down_bank.data=B.mlp_down_bank.data.float()
	for Y in B.modules():
		if isinstance(Y,CastedLinear):Y.float()
	restore_low_dim_params_to_fp32(B)
	if SKIP_COMPILE:AA=B;C('compile:SKIPPED (SKIP_COMPILE=1)')
	else:AA=torch.compile(B,dynamic=_C,fullgraph=_B)
	A0=AA;BU=[B.qo_bank,B.kv_bank,B.mlp_up_bank,B.mlp_down_bank];BV=list(B.blocks.named_parameters());N=[A for(B,A)in BV if A.ndim<2 or any(A in B for A in CONTROL_TENSOR_NAME_PATTERNS)]
	if B.skip_weights.numel()>0:N.append(B.skip_weights)
	N.append(B.smear.gate)
	if B.bigram is not _A:N.append(B.bigram.scale)
	S=A.tied_embed_lr if A.tie_embeddings else A.embed_lr;AB=[{_G:[B.tok_emb.weight],_H:S,U:S}]
	if B.bigram is not _A:
		AB.append({_G:[B.bigram.embed.weight],_H:S,U:S})
		if B.bigram.proj is not _A:N.append(B.bigram.proj.weight)
	if B.ve_shared is not _A:
		AB.append({_G:[B.ve_shared.embed.weight],_H:S,U:S})
		if B.ve_shared.proj is not _A:N.append(B.ve_shared.proj.weight)
		N.append(B.ve_shared.scale)
		for Z in B.ve_layer_scales:N.append(Z)
	A1=torch.optim.AdamW(AB,betas=(A.beta1,A.beta2),eps=A.adam_eps,weight_decay=A.adam_wd,fused=_B);j=Muon(BU,lr=A.matrix_lr,momentum=A.muon_momentum,backend_steps=A.muon_backend_steps,weight_decay=A.muon_wd,muon_eq_r=A.muon_eq_r)
	for a in j.param_groups:a[U]=A.matrix_lr
	Ap=torch.optim.AdamW([{_G:N,_H:A.scalar_lr,U:A.scalar_lr}],betas=(A.beta1,A.beta2),eps=A.adam_eps,weight_decay=A.adam_wd,fused=_B);A2=list(A1.param_groups[0][_G])
	for BW in A1.param_groups[1:]:A2.extend(BW[_G])
	A2.extend(N);k=_A
	if B.lm_head is not _A:k=torch.optim.Adam([{_G:[B.lm_head.weight],_H:A.head_lr,U:A.head_lr}],betas=(A.beta1,A.beta2),eps=A.adam_eps,fused=_B);A2.append(B.lm_head.weight)
	b=[A1,j,Ap]
	if k is not _A:b.append(k)
	BX=sum(A.numel()for A in B.parameters());BY=sum(A.numel()for A in B.mtp_heads.parameters());C(f"model_params:{BX}");C(f"mtp_num_heads:{A.mtp_num_heads} mtp_loss_weight:{A.mtp_loss_weight} mtp_params:{BY}");BZ=[A for(A,B)in enumerate(B.blocks)if B.attn.use_xsa];C(f"XSA:last_{A.xsa_last_n} active_layers:{BZ}");C(f"world_size:{I} grad_accum_steps:{L}");C('sdp_backends:cudnn=False flash=True mem_efficient=False math=False');C(f"attention_mode:gqa num_heads:{A.num_heads} num_kv_heads:{A.num_kv_heads}");C(f"tie_embeddings:{A.tie_embeddings} embed_lr:{S} head_lr:{A.head_lr if B.lm_head is not _A else _E} matrix_lr:{A.matrix_lr} scalar_lr:{A.scalar_lr}");C(f"train_batch_tokens:{A.train_batch_tokens} train_seq_len:{A.train_seq_len} iterations:{A.iterations} warmup_steps:{A.warmup_steps} max_wallclock_seconds:{A.max_wallclock_seconds:.3f}");C(f"seed:{A.seed}");AC=DistributedTokenLoader(A.train_files,M,I,E)
	def l():
		for A in b:A.zero_grad(set_to_none=_B)
	m=1e3*A.max_wallclock_seconds if A.max_wallclock_seconds>0 else _A
	def Ba(step,elapsed_ms):
		C=elapsed_ms;B=step
		if A.warmdown_iters<=0:return _D
		if m is _A:F=max(A.iterations-A.warmdown_iters,0);return max((A.iterations-B)/max(A.warmdown_iters,1),_E)if F<=B<A.iterations else _D
		G=C/max(B,1);D=A.warmdown_iters*G;E=max(m-C,_E);return E/max(D,1e-09)if E<=D else _D
	if A.warmup_steps>0:
		Bb={A:B.detach().cpu().clone()for(A,B)in B.state_dict().items()};Bc=[copy.deepcopy(A.state_dict())for A in b];A0.train()
		for AD in range(A.warmup_steps):
			l()
			for Bd in range(L):
				n,o=AC.next_batch(A.train_batch_tokens,A.train_seq_len,L)
				with torch.autocast(device_type=_I,dtype=torch.bfloat16,enabled=_B):Be=A0(n,o)
				(Be*Aj).backward()
			if Q:
				for J in B.parameters():
					if J.grad is not _A:dist.all_reduce(J.grad,op=dist.ReduceOp.AVG)
			for p in b:p.step()
			l()
			if A.warmup_steps<=20 or(AD+1)%10==0 or AD+1==A.warmup_steps:C(f"warmup_step:{AD+1}/{A.warmup_steps}")
		B.load_state_dict(Bb,strict=_B)
		for(p,Bf)in zip(b,Bc,strict=_B):p.load_state_dict(Bf)
		l();AC=DistributedTokenLoader(A.train_files,M,I,E)
	AE=_A;Aq=0;from collections import deque;q=deque(maxlen=A.lawa_k);Ar={A:B.detach().float().clone()for(A,B)in B.state_dict().items()};As=.997;c=_E;d=_A;torch.cuda.synchronize();A3=time.perf_counter();G=0
	while _B:
		At=G==A.iterations or d is not _A and G>=d;Bg=At or A.val_loss_every>0 and G%A.val_loss_every==0
		if Bg:torch.cuda.synchronize();c+=1e3*(time.perf_counter()-A3);Bh,Bi=eval_val(A,A0,M,I,E,L,R,V,W,X);C(f"step:{G}/{A.iterations} val_loss:{Bh:.4f} val_bpb:{Bi:.4f} train_time:{c:.0f}ms step_avg:{c/max(G,1):.2f}ms");torch.cuda.synchronize();A3=time.perf_counter()
		if At:
			if d is not _A and G<A.iterations:C(f"stopping_early: wallclock_cap train_time:{c:.0f}ms step:{G}/{A.iterations}")
			break
		Bj=c+1e3*(time.perf_counter()-A3);e=Ba(G,Bj)
		if A.late_qat_threshold>0 and e<A.late_qat_threshold and not CastedLinear._qat_enabled:CastedLinear._qat_enabled=_B;C(f"late_qat:enabled step:{G} scale:{e:.4f}")
		if A.recur_layers and G==A.recur_start_step and not B.recur_active:B.recur_active=_B;C(f"depth_recurrence:enabled step:{G} layers:{A.recur_layers}")
		l()
		if A.soft_round_qat and CastedLinear._qat_enabled:Bk=min(max(_D-e/A.late_qat_threshold,_E),_D);CastedLinear._soft_round_alpha=_D+15.*Bk
		AF=torch.zeros((),device=E)
		for Bd in range(L):
			n,o=AC.next_batch(A.train_batch_tokens,A.train_seq_len,L)
			with torch.autocast(device_type=_I,dtype=torch.bfloat16,enabled=_B):T=A0(n,o)
			if A.crown_q_enabled and e<_D:
				AG=torch.zeros((),device=E)
				for(D,J)in B.named_parameters():
					if J.ndim==2 and J.numel()>65536 and(_V in D or _U in D or'bank'in D):Bl=J.float().abs().amax(dim=1,keepdim=_B).detach();Z=Bl/31.;AG=AG+(J.float()**2*Z**2/12.).mean()
				T=T+A.crown_q_lambda*AG
			AF+=T.detach();(T*Aj).backward()
		AF/=L;Au=min(G/A.muon_momentum_warmup_steps,_D)if A.muon_momentum_warmup_steps>0 else _D;Bm=(1-Au)*A.muon_momentum_warmup_start+Au*A.muon_momentum
		for a in j.param_groups:a[_o]=Bm
		for p in b:
			for a in p.param_groups:a[_H]=a[U]*e
		if A.grad_clip_norm>0:torch.nn.utils.clip_grad_norm_(B.parameters(),A.grad_clip_norm)
		j.launch_reduce_scatters()
		if Q:
			for J in A2:
				if J.grad is not _A:dist.all_reduce(J.grad,op=dist.ReduceOp.AVG)
		A1.step();Ap.step()
		if k is not _A:k.step()
		j.step();l()
		with torch.no_grad():
			for(D,AH)in B.state_dict().items():Ar[D].mul_(As).add_(AH.detach().float(),alpha=_D-As)
		G+=1;AI=c+1e3*(time.perf_counter()-A3)
		if A.swa_enabled and e<.2 and G%A.swa_every==0:
			if AE is _A:AE={A:B.detach().cpu().clone()for(A,B)in B.state_dict().items()};Aq=1;C(f"swa:start step:{G}")
			else:
				for(D,AH)in B.state_dict().items():AE[D]+=AH.detach().cpu()
				Aq+=1
		if A.lawa_enabled and G%A.lawa_freq==0:q.append({A:B.detach().cpu().clone()for(A,B)in B.state_dict().items()})
		Bn=A.train_log_every>0 and(G<=10 or G%A.train_log_every==0 or d is not _A)
		if Bn:C(f"step:{G}/{A.iterations} train_loss:{AF.item():.4f} train_time:{AI:.0f}ms step_avg:{AI/G:.2f}ms")
		AJ=m is not _A and AI>=m
		if Q and m is not _A:Av=torch.tensor(int(AJ),device=E);dist.all_reduce(Av,op=dist.ReduceOp.MAX);AJ=bool(Av.item())
		if d is _A and AJ:d=G
	C(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB")
	if A.lawa_enabled and len(q)>1:
		C(f"lawa:applying LAWA averaging k={len(q)}");A4=B.state_dict();O={A:torch.zeros(B.shape,dtype=torch.float32,device=_N)for(A,B)in A4.items()}
		for Bo in q:
			for D in O:O[D]+=Bo[D].float()
		for D in O:O[D]/=len(q);O[D]=O[D].to(dtype=A4[D].dtype)
		B.load_state_dict(O,strict=_B)
	else:C('ema:applying EMA weights');A4=B.state_dict();O={A:B.to(dtype=A4[A].dtype)for(A,B)in Ar.items()};B.load_state_dict(O,strict=_B)
	torch.cuda.synchronize();Bp=time.perf_counter();Bq,Br=eval_val(A,AA,M,I,E,L,R,V,W,X);torch.cuda.synchronize();C(f"DIAGNOSTIC post_ema val_loss:{Bq:.4f} val_bpb:{Br:.4f} eval_time:{1e3*(time.perf_counter()-Bp):.0f}ms")
	if A.prequant_ttt:
		C('prequant_ttt:starting discriminative fine-tuning...');Bs=time.perf_counter();Aw=generate_autoregressive_calib(B,E,num_seqs=A.gptq_calib_batches,seq_len=A.train_seq_len,vocab_size=A.vocab_size,temperature=.8,batch_size=8,seed=A.seed+7);Ax=[A.clone().detach()for A in Aw];del Aw
		for Y in B.modules():
			Ay={}
			for(A5,AK)in Y.named_buffers(recurse=_C):
				if AK is not _A and AK.is_inference():Ay[A5]=AK.clone()
			for(A5,Bt)in Ay.items():delattr(Y,A5);Y.register_buffer(A5,Bt)
		Az=A.num_layers;A6=[];AL=set()
		for f in range(Az):
			AM=[B for(A,B)in B.named_parameters()if B.requires_grad and(f"blocks.{f}."in A or f"recur_scales.{f}"in A)]
			if AM:
				Bu=.3+.7*(f/max(1,Az-1));A6.append({_G:AM,_H:A.prequant_ttt_lr*Bu})
				for J in AM:AL.add(id(J))
		AN=[A for(B,A)in B.named_parameters()if B in(_W,_X,_Y,_Z)and A.requires_grad]
		if AN:
			A6.append({_G:AN,_H:A.prequant_ttt_lr*.65})
			for J in AN:AL.add(id(J))
		A_=[A for A in B.parameters()if A.requires_grad and id(A)not in AL]
		if A_:A6.append({_G:A_,_H:A.prequant_ttt_lr*.1})
		AO=torch.optim.AdamW(A6,weight_decay=.01);B.train()
		for Bv in range(A.prequant_ttt_epochs):
			B0=_E;B1=0
			for B2 in Ax:
				n=B2[:,:-1].to(E);o=B2[:,1:].to(E)
				with torch.autocast(device_type=_I,dtype=torch.bfloat16):B3=B.forward_logits(n);T=F.cross_entropy(B3.reshape(-1,B3.size(-1)),o.reshape(-1))
				AO.zero_grad();T.backward();torch.nn.utils.clip_grad_norm_(B.parameters(),_D);AO.step();B0+=T.item();B1+=1
			C(f"prequant_ttt:epoch {Bv+1}/{A.prequant_ttt_epochs} loss:{B0/max(1,B1):.4f}")
		B.eval();del Ax,AO;torch.cuda.empty_cache();C(f"prequant_ttt:done in {time.perf_counter()-Bs:.1f}s")
	B4=B.state_dict();AP={A:B for(A,B)in B4.items()if BO not in A};B5=sum(int(B.numel())for(A,B)in B4.items()if BO in A)
	if B5>0:C(f"export_excluding_mtp_params:{B5}")
	if y:torch.save(AP,BP);Bw=os.path.getsize(BP);AQ=len(x.encode(_M));C(f"Serialized model: {Bw} bytes");C(f"Code size: {AQ} bytes")
	if SKIP_QUANTIZE or A.iterations<1000 and A.max_wallclock_seconds>0:
		C('SKIP_QUANTIZE: skipping quantization, roundtrip eval, sliding window, and TTT')
		if Q:dist.destroy_process_group()
		return
	B6={A:B.detach().cpu()for(A,B)in AP.items()};r=_unbank_state_dict(B6,A.num_layers);s=_A
	if A.gptq_full_hessian:
		C('gptq:building non-banked model for Hessian collection...');g=_HessianGPT(vocab_size=A.vocab_size,num_layers=A.num_layers,model_dim=A.model_dim,num_heads=A.num_heads,num_kv_heads=A.num_kv_heads,mlp_mult=A.mlp_mult,tie_embeddings=A.tie_embeddings,logit_softcap=A.logit_softcap,rope_base=A.rope_base,qk_gain_init=A.qk_gain_init,bigram_vocab_size=A.bigram_vocab_size,bigram_dim=A.bigram_dim,xsa_last_n=A.xsa_last_n,rope_dims=A.rope_dims,ln_scale=A.ln_scale,ve_enabled=A.ve_enabled,ve_dim=A.ve_dim,ve_layers=A.ve_layers,trigram=A.trigram_enabled).to(E).bfloat16()
		for t in g.modules():
			if isinstance(t,CastedLinear):t.float()
		restore_low_dim_params_to_fp32(g);g.load_state_dict({A:B.to(E)for(A,B)in r.items()if A in g.state_dict()},strict=_C);C(f"gptq:generating autoregressive calibration data ({A.gptq_calib_batches} seqs x {A.train_seq_len} tokens, temp=0.8)...");B.load_state_dict(AP,strict=_C);Bx=time.perf_counter();AR=generate_autoregressive_calib(B,E,num_seqs=A.gptq_calib_batches,seq_len=A.train_seq_len,vocab_size=A.vocab_size,temperature=.8,batch_size=8,seed=A.seed);C(f"gptq:generated {len(AR)} sequences in {time.perf_counter()-Bx:.1f}s");C('gptq:collecting hessians from autoregressive data...');s=collect_hessians_from_tokens(g,AR,E,gptq_damp=A.gptq_damp);C(f"gptq:collected hessians for {len(s)} layers (AR self-gen)");del AR;del g;torch.cuda.empty_cache()
	if A.hadamard_rotation:r=_apply_hadamard_rotation(r,A.model_dim,C)
	h={}
	if A.mixed_bitwidth and s:
		AS={}
		for(D,By)in s.items():AS[D]=torch.diag(By).mean().item()
		B7=sorted(AS.items(),key=lambda x:x[1]);u=len(B7);AT=max(1,u//5);AU=max(1,u//5)
		for(f,(D,CD))in enumerate(B7):
			if f<AT:h[D]=15
			elif f>=u-AU:h[D]=63
			else:h[D]=31
		C(f"mixed_bitwidth: {AT} int5, {u-AT-AU} int6, {AU} int7 (of {u} layers)")
		for(D,B8)in sorted(h.items(),key=lambda x:x[1]):Bz={15:'int5',31:'int6',63:'int7'}.get(B8,f"cr{B8}");C(f"  {D}: {Bz} (sensitivity={AS[D]:.4f})")
	P,AV=mixed_quantize_int6(r,{_U,_V},hessians=s,clip_ranges=h if h else _A,gptq_damp=A.gptq_damp);AW=float(os.environ.get('TARGET_MB','15.9'));B_=len(x.encode(_M));K=[]
	for(D,B9)in AV.items():
		if not(isinstance(B9,dict)and B9.get(_k,'').startswith('int')):continue
		AX,BA=D+_a,D+_b
		if AX not in P or BA not in P:continue
		v,Z=P[AX],P[BA]
		if Z.ndim>0:
			AY=v.abs()==1
			if AY.any():
				C0=torch.arange(v.shape[0]).unsqueeze(1).expand_as(v)[AY];C1=torch.arange(v.numel()).reshape(v.shape)[AY];C2=Z.float()[C0].pow(2)
				for(C3,C4)in zip(C1.tolist(),C2.tolist()):K.append((AX,C3,C4))
	if K:
		K.sort(key=lambda x:x[2])
		def w(n):
			A={A:B.clone()for(A,B)in P.items()}
			for B in range(min(n,len(K))):A[K[B][0]].view(-1)[K[B][1]]=0
			C=io.BytesIO();torch.save({'w':A,'m':AV},C);D=C.getvalue();E=brotli.compress(D,quality=11)if _HAS_BROTLI else lzma.compress(D,preset=9);return len(E)+B_,A
		BB,_=w(0);AZ=int(AW*1024*1024);C(f"selective_prune: {len(K)} ±1 candidates, unpruned={BB/1048576:.2f}MB target={AW}MB")
		if BB<=AZ:C('selective_prune: already fits, no pruning needed')
		else:
			BC,_=w(len(K));C(f"selective_prune: full ±1 prune={BC/1048576:.2f}MB")
			if BC>AZ:C('selective_prune: even full prune not enough, applying all');_,P=w(len(K))
			else:
				i,Aa=0,len(K)
				while i<Aa:
					Ab=(i+Aa)//2;C5,_=w(Ab)
					if C5<=AZ:Aa=Ab
					else:i=Ab+1
				C(f"selective_prune: pruning {i}/{len(K)} ±1 values ({100*i/len(K):.1f}%) to fit {AW}MB");_,P=w(i)
	BD=io.BytesIO();torch.save({'w':P,'m':AV},BD);BE=BD.getvalue()
	if _HAS_BROTLI:Ac=brotli.compress(BE,quality=11);Ad='brotli'
	else:Ac=lzma.compress(BE,preset=9);Ad='lzma'
	if y:
		with open(BQ,'wb')as Ae:Ae.write(Ac)
		BF=len(Ac);AQ=len(x.encode(_M));C(f"Serialized model int6+{Ad}: {BF} bytes");C(f"Total submission size int6+{Ad}: {BF+AQ} bytes")
	if Q:dist.barrier()
	with open(BQ,'rb')as Ae:BG=Ae.read()
	if _HAS_BROTLI:BH=brotli.decompress(BG)
	else:BH=lzma.decompress(BG)
	BI=torch.load(io.BytesIO(BH),map_location=_N);C6=dequantize_mixed_int6(BI['w'],BI['m'],r);C7=_rebank_state_dict(C6,A.num_layers,B6);H=GPT(vocab_size=A.vocab_size,num_layers=A.num_layers,model_dim=A.model_dim,num_heads=A.num_heads,num_kv_heads=A.num_kv_heads,mlp_mult=A.mlp_mult,tie_embeddings=A.tie_embeddings,tied_embed_init_std=A.tied_embed_init_std,logit_softcap=A.logit_softcap,rope_base=A.rope_base,qk_gain_init=A.qk_gain_init,mtp_num_heads=0,mtp_loss_weight=_E,bigram_vocab_size=A.bigram_vocab_size,bigram_dim=A.bigram_dim,xsa_last_n=A.xsa_last_n,rope_dims=A.rope_dims,ln_scale=A.ln_scale,dtg=A.dtg_enabled,ve_enabled=A.ve_enabled,ve_dim=A.ve_dim,ve_layers=A.ve_layers,gated_attention=A.gated_attention,value_residual=A.value_residual,recur_layers=A.recur_layers,parallel_start_layer=A.parallel_start_layer,trigram=A.trigram_enabled).to(E).bfloat16();H.recur_active=_B;H.qo_bank.data=H.qo_bank.data.float();H.kv_bank.data=H.kv_bank.data.float();H.mlp_up_bank.data=H.mlp_up_bank.data.float();H.mlp_down_bank.data=H.mlp_down_bank.data.float()
	for t in H.modules():
		if isinstance(t,CastedLinear):t.float()
	restore_low_dim_params_to_fp32(H);H.load_state_dict(C7,strict=_B);torch._dynamo.config.cache_size_limit=32;C8=H if SKIP_COMPILE else torch.compile(H,dynamic=_C,fullgraph=_B);torch.cuda.synchronize();C9=time.perf_counter();BJ,BK=eval_val(A,C8,M,I,E,L,R,V,W,X,eval_seq_len=A9);torch.cuda.synchronize();C(f"final_int6_roundtrip val_loss:{BJ:.4f} val_bpb:{BK:.4f} eval_time:{1e3*(time.perf_counter()-C9):.0f}ms");C(f"final_int6_roundtrip_exact val_loss:{BJ:.8f} val_bpb:{BK:.8f}");A7=A9
	if A.eval_stride>0 and A.eval_stride<A7:torch.cuda.synchronize();CA=time.perf_counter();Af,Ag=eval_val_sliding(A,H,M,I,E,R,V,W,X,stride=A.eval_stride,eval_seq_len=A7);torch.cuda.synchronize();C(f"final_int6_sliding_window val_loss:{Af:.4f} val_bpb:{Ag:.4f} stride:{A.eval_stride} eval_time:{1e3*(time.perf_counter()-CA):.0f}ms");C(f"final_int6_sliding_window_exact val_loss:{Af:.8f} val_bpb:{Ag:.8f}");C(f"final_int8_zlib_roundtrip_exact val_loss:{Af:.8f} val_bpb:{Ag:.8f}")
	if A.eval_stride!=64 and 64<A7:torch.cuda.synchronize();CB=time.perf_counter();Ah,Ai=eval_val_sliding(A,H,M,I,E,R,V,W,X,stride=64,eval_seq_len=A7);torch.cuda.synchronize();C(f"final_int6_sliding_window_s64 val_loss:{Ah:.4f} val_bpb:{Ai:.4f} stride:64 eval_time:{1e3*(time.perf_counter()-CB):.0f}ms");C(f"final_int6_sliding_window_s64_exact val_loss:{Ah:.8f} val_bpb:{Ai:.8f}");C(f"final_int8_zlib_roundtrip_exact val_loss:{Ah:.8f} val_bpb:{Ai:.8f}")
	if A.ttt_enabled:torch.cuda.synchronize();CC=time.perf_counter();BL,BM=eval_val_sliding_ttt(A,H,M,I,E,R,V,W,X,stride=A.eval_stride,log0=C);torch.cuda.synchronize();C(f"legal_ttt val_loss:{BL:.4f} val_bpb:{BM:.4f} eval_time:{1e3*(time.perf_counter()-CC):.0f}ms");C(f"legal_ttt_exact val_loss:{BL:.8f} val_bpb:{BM:.8f}")
	if Q:dist.destroy_process_group()
if __name__=='__main__':main()