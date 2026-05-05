import collections,copy,fcntl,glob,io,lzma,math,os
from pathlib import Path
import random,re,subprocess,sys,time,uuid,numpy as np,sentencepiece as spm,torch,torch.distributed as dist,torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor,nn
try:
	from flash_attn_interface import flash_attn_func as flash_attn_3_func
	_HAS_FA3=True
except ImportError:
	_HAS_FA3=False
# ===== Fused softcapped cross-entropy (Triton) — training-only path =====
# Replaces the eager softcap*tanh(x/softcap) + F.cross_entropy with a single
# fused Triton kernel that reads logits once, applies softcap in-register, and
# computes (LSE, loss) in one streaming pass. The backward kernel mirrors the
# forward so there's no stored softcapped logits tensor.
# Math note: the kernel uses z = 2C*sigmoid(2x/C) instead of C*tanh(x/C).
# These differ by a constant +C, which cancels in log_softmax (shift invariance).
_FUSED_CE_ENABLED = bool(int(os.environ.get('FUSED_CE_ENABLED', '0')))
_FUSED_CE_LIBRARY = "piyushdatta_fusedce"
_FUSED_CE_BLOCK_SIZE = 1024  # 8192 vocab / 1024 = 8 iterations, perfect alignment
_FUSED_CE_NUM_WARPS = 4
if _FUSED_CE_ENABLED:
	import triton
	import triton.language as tl
	@triton.jit
	def _softcapped_ce_fwd_kernel(
		logits_ptr, losses_ptr, lse_ptr, targets_ptr,
		stride_logits_n, stride_logits_v,
		n_rows, n_cols, softcap,
		block_size: tl.constexpr,
	):
		row_idx = tl.program_id(0).to(tl.int64)
		logits_row_ptr = logits_ptr + row_idx * stride_logits_n
		max_val = -float("inf")
		sum_exp = 0.0
		A = 2.0 * softcap
		inv_C = 2.0 / softcap
		for off in range(0, n_cols, block_size):
			cols = off + tl.arange(0, block_size)
			mask = cols < n_cols
			val = tl.load(
				logits_row_ptr + cols * stride_logits_v,
				mask=mask, other=-float("inf"),
			).to(tl.float32)
			z = A * tl.sigmoid(val * inv_C)
			z = tl.where(mask, z, -float("inf"))
			curr_max = tl.max(z, axis=0)
			new_max = tl.maximum(max_val, curr_max)
			sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(tl.exp(z - new_max), axis=0)
			max_val = new_max
		lse = max_val + tl.log(sum_exp)
		tl.store(lse_ptr + row_idx, lse)
		target = tl.load(targets_ptr + row_idx).to(tl.int32)
		target_val = tl.load(logits_row_ptr + target * stride_logits_v).to(tl.float32)
		target_z = A * tl.sigmoid(target_val * inv_C)
		tl.store(losses_ptr + row_idx, lse - target_z)
	@triton.jit
	def _softcapped_ce_bwd_kernel(
		grad_logits_ptr, grad_losses_ptr, lse_ptr, logits_ptr, targets_ptr,
		stride_logits_n, stride_logits_v,
		stride_grad_n, stride_grad_v,
		n_rows, n_cols, softcap,
		block_size: tl.constexpr,
	):
		row_idx = tl.program_id(0).to(tl.int64)
		logits_row_ptr = logits_ptr + row_idx * stride_logits_n
		grad_row_ptr = grad_logits_ptr + row_idx * stride_grad_n
		lse = tl.load(lse_ptr + row_idx)
		grad_loss = tl.load(grad_losses_ptr + row_idx).to(tl.float32)
		target = tl.load(targets_ptr + row_idx).to(tl.int32)
		A = 2.0 * softcap
		inv_C = 2.0 / softcap
		dz_dx_scale = A * inv_C
		for off in range(0, n_cols, block_size):
			cols = off + tl.arange(0, block_size)
			mask = cols < n_cols
			val = tl.load(
				logits_row_ptr + cols * stride_logits_v,
				mask=mask, other=0.0,
			).to(tl.float32)
			sigmoid_u = tl.sigmoid(val * inv_C)
			z = A * sigmoid_u
			probs = tl.exp(z - lse)
			grad_z = grad_loss * (probs - tl.where(cols == target, 1.0, 0.0))
			grad_x = grad_z * (dz_dx_scale * sigmoid_u * (1.0 - sigmoid_u))
			tl.store(grad_row_ptr + cols * stride_grad_v, grad_x, mask=mask)
	def _validate_softcapped_ce_inputs(logits, targets, softcap):
		if logits.ndim != 2: raise ValueError(f"Expected logits.ndim=2, got {logits.ndim}")
		if targets.ndim != 1: raise ValueError(f"Expected targets.ndim=1, got {targets.ndim}")
		if logits.shape[0] != targets.shape[0]: raise ValueError(f"Row mismatch logits={tuple(logits.shape)} targets={tuple(targets.shape)}")
		if not logits.is_cuda or not targets.is_cuda: raise ValueError("softcapped_cross_entropy requires CUDA tensors")
		if softcap <= 0.0: raise ValueError(f"softcap must be positive, got {softcap}")
		logits = logits.contiguous()
		targets = targets.contiguous()
		if targets.dtype != torch.int64: targets = targets.to(dtype=torch.int64)
		return logits, targets
	@torch.library.custom_op(f"{_FUSED_CE_LIBRARY}::softcapped_ce", mutates_args=())
	def softcapped_ce_op(logits: Tensor, targets: Tensor, softcap: float) -> tuple[Tensor, Tensor]:
		logits, targets = _validate_softcapped_ce_inputs(logits, targets, float(softcap))
		n_rows, n_cols = logits.shape
		losses = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
		lse = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
		_softcapped_ce_fwd_kernel[(n_rows,)](
			logits, losses, lse, targets,
			logits.stride(0), logits.stride(1),
			n_rows, n_cols, float(softcap),
			block_size=_FUSED_CE_BLOCK_SIZE, num_warps=_FUSED_CE_NUM_WARPS,
		)
		return losses, lse
	@softcapped_ce_op.register_fake
	def _(logits: Tensor, targets: Tensor, softcap: float):
		n_rows = logits.shape[0]
		return (logits.new_empty((n_rows,), dtype=torch.float32), logits.new_empty((n_rows,), dtype=torch.float32))
	@torch.library.custom_op(f"{_FUSED_CE_LIBRARY}::softcapped_ce_backward", mutates_args=())
	def softcapped_ce_backward_op(logits: Tensor, targets: Tensor, lse: Tensor, grad_losses: Tensor, softcap: float) -> Tensor:
		logits, targets = _validate_softcapped_ce_inputs(logits, targets, float(softcap))
		lse = lse.contiguous(); grad_losses = grad_losses.contiguous().to(dtype=torch.float32)
		grad_logits = torch.empty_like(logits)
		n_rows, n_cols = logits.shape
		_softcapped_ce_bwd_kernel[(n_rows,)](
			grad_logits, grad_losses, lse, logits, targets,
			logits.stride(0), logits.stride(1),
			grad_logits.stride(0), grad_logits.stride(1),
			n_rows, n_cols, float(softcap),
			block_size=_FUSED_CE_BLOCK_SIZE, num_warps=_FUSED_CE_NUM_WARPS,
		)
		return grad_logits
	@softcapped_ce_backward_op.register_fake
	def _(logits: Tensor, targets: Tensor, lse: Tensor, grad_losses: Tensor, softcap: float):
		return logits.new_empty(logits.shape)
	def _softcapped_ce_setup_context(ctx, inputs, output):
		logits, targets, softcap = inputs
		_losses, lse = output
		ctx.save_for_backward(logits, targets, lse)
		ctx.softcap = float(softcap)
	def _softcapped_ce_backward(ctx, grad_losses, grad_lse):
		del grad_lse
		logits, targets, lse = ctx.saved_tensors
		grad_logits = getattr(torch.ops, _FUSED_CE_LIBRARY).softcapped_ce_backward(logits, targets, lse, grad_losses, ctx.softcap)
		return grad_logits, None, None
	softcapped_ce_op.register_autograd(_softcapped_ce_backward, setup_context=_softcapped_ce_setup_context)
	def softcapped_cross_entropy(logits, targets, softcap, reduction="mean"):
		losses, _lse = getattr(torch.ops, _FUSED_CE_LIBRARY).softcapped_ce(logits, targets, float(softcap))
		if reduction == "none": return losses
		if reduction == "sum": return losses.sum()
		if reduction == "mean": return losses.mean()
		raise ValueError(f"Unsupported reduction={reduction!r}")
class Hyperparameters:data_dir=os.environ.get('DATA_DIR','./data/');seed=int(os.environ.get('SEED',1337));run_id=os.environ.get('RUN_ID',str(uuid.uuid4()));iterations=int(os.environ.get('ITERATIONS',20000));warmdown_frac=float(os.environ.get('WARMDOWN_FRAC',.72));warmup_steps=int(os.environ.get('WARMUP_STEPS',20));train_batch_tokens=int(os.environ.get('TRAIN_BATCH_TOKENS',393216));train_seq_len=int(os.environ.get('TRAIN_SEQ_LEN',2048));train_log_every=int(os.environ.get('TRAIN_LOG_EVERY',500));max_wallclock_seconds=float(os.environ.get('MAX_WALLCLOCK_SECONDS',6e2));val_batch_tokens=int(os.environ.get('VAL_BATCH_TOKENS',524288));eval_seq_len=int(os.environ.get('EVAL_SEQ_LEN',2048));val_loss_every=int(os.environ.get('VAL_LOSS_EVERY',4000));sliding_window_enabled=bool(int(os.environ.get('SLIDING_WINDOW_ENABLED','1')));vocab_size=int(os.environ.get('VOCAB_SIZE',8192));num_layers=int(os.environ.get('NUM_LAYERS',11));xsa_last_n=int(os.environ.get('XSA_LAST_N',11));model_dim=int(os.environ.get('MODEL_DIM',512));embedding_dim=int(os.environ.get('EMBEDDING_DIM',512));num_kv_heads=int(os.environ.get('NUM_KV_HEADS',4));num_heads=int(os.environ.get('NUM_HEADS',8));mlp_mult=float(os.environ.get('MLP_MULT',4.));skip_gates_enabled=bool(int(os.environ.get('SKIP_GATES_ENABLED','1')));tie_embeddings=bool(int(os.environ.get('TIE_EMBEDDINGS','1')));logit_softcap=float(os.environ.get('LOGIT_SOFTCAP',3e1));rope_base=float(os.environ.get('ROPE_BASE',1e4));rope_dims=int(os.environ.get('ROPE_DIMS',16));rope_train_seq_len=int(os.environ.get('ROPE_TRAIN_SEQ_LEN',2048));ln_scale=bool(int(os.environ.get('LN_SCALE','1')));qk_gain_init=float(os.environ.get('QK_GAIN_INIT',5.));parallel_residual_start=int(os.environ.get('PARALLEL_RESIDUAL_START',7));enable_looping_at=float(os.environ.get('ENABLE_LOOPING_AT',.45));min_lr=float(os.environ.get('MIN_LR',.10));embed_lr=float(os.environ.get('EMBED_LR',.6));head_lr=float(os.environ.get('HEAD_LR',.008));tied_embed_lr=float(os.environ.get('TIED_EMBED_LR',.03));tied_embed_init_std=float(os.environ.get('TIED_EMBED_INIT_STD',.005));matrix_lr=float(os.environ.get('MATRIX_LR',.028));scalar_lr=float(os.environ.get('SCALAR_LR',.02));muon_momentum=float(os.environ.get('MUON_MOMENTUM',.95));muon_backend_steps=int(os.environ.get('MUON_BACKEND_STEPS',5));muon_momentum_warmup_start=float(os.environ.get('MUON_MOMENTUM_WARMUP_START',.85));muon_momentum_warmup_steps=int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS',500));muon_row_normalize=bool(int(os.environ.get('MUON_ROW_NORMALIZE','1')));beta1=float(os.environ.get('BETA1',.9));beta2=float(os.environ.get('BETA2',.95));adam_eps=float(os.environ.get('ADAM_EPS',1e-08));grad_clip_norm=float(os.environ.get('GRAD_CLIP_NORM',.3));eval_stride=int(os.environ.get('EVAL_STRIDE',64));adam_wd=float(os.environ.get('ADAM_WD',.02));muon_wd=float(os.environ.get('MUON_WD',.095));embed_wd=float(os.environ.get('EMBED_WD',.085));ema_decay=float(os.environ.get('EMA_DECAY',.9965));compressor=os.environ.get('COMPRESSOR','brotli');swa_enabled=bool(int(os.environ.get('SWA_ENABLED','1')));swa_start_frac=float(os.environ.get('SWA_START_FRAC',.12));swa_every=int(os.environ.get('SWA_EVERY',1));gptq_calibration_batches=int(os.environ.get('GPTQ_CALIBRATION_BATCHES',64));gptq_reserve_seconds=float(os.environ.get('GPTQ_RESERVE_SECONDS',0.));matrix_bits=int(os.environ.get('MATRIX_BITS',6));embed_bits=int(os.environ.get('EMBED_BITS',8));matrix_clip_sigmas=float(os.environ.get('MATRIX_CLIP_SIGMAS',12.85));embed_clip_sigmas=float(os.environ.get('EMBED_CLIP_SIGMAS',2e1));hessian_clip_lambda=float(os.environ.get('HESSIAN_CLIP_LAMBDA',.175));eval_temperature=float(os.environ.get('EVAL_TEMPERATURE',1.));sparsity_start_frac=float(os.environ.get('SPARSITY_START_FRAC',0.));ttt_enabled=bool(int(os.environ.get('TTT_ENABLED','1')));ttt_lr=float(os.environ.get('TTT_LR',.02));ttt_epochs=int(os.environ.get('TTT_EPOCHS',3));ttt_momentum=float(os.environ.get('TTT_MOMENTUM',.9));ttt_chunk_tokens=int(os.environ.get('TTT_CHUNK_TOKENS',32768));ttt_lora_rank=int(os.environ.get('TTT_LORA_RANK',96));ttt_ns_steps=int(os.environ.get('TTT_NS_STEPS',0));ttt_swa=bool(int(os.environ.get('TTT_SWA','0')));ttt_reset_per_chunk=bool(int(os.environ.get('TTT_RESET_PER_CHUNK','0')));ttt_lora_alpha=float(os.environ.get('TTT_LORA_ALPHA',144));ttt_warm_start_a=bool(int(os.environ.get('TTT_WARM_START_A','1')));ttt_lora_lr=float(os.environ.get('TTT_LORA_LR',0.0001));ttt_chunk_size=int(os.environ.get('TTT_CHUNK_SIZE',48));ttt_batch_size=int(os.environ.get('TTT_BATCH_SIZE',64));ttt_grad_steps=int(os.environ.get('TTT_GRAD_STEPS',1));ttt_weight_decay=float(os.environ.get('TTT_WEIGHT_DECAY',1.0));ttt_beta1=float(os.environ.get('TTT_BETA1',0));ttt_beta2=float(os.environ.get('TTT_BETA2',0.999));ttt_k_lora=bool(int(os.environ.get('TTT_K_LORA','1')));ttt_mlp_lora=bool(int(os.environ.get('TTT_MLP_LORA','1')));ttt_o_lora=bool(int(os.environ.get('TTT_O_LORA','1')));ttt_optimizer=os.environ.get('TTT_OPTIMIZER','adam');val_doc_fraction=float(os.environ.get('VAL_DOC_FRACTION',1.0));phased_ttt_prefix_docs=int(os.environ.get('PHASED_TTT_PREFIX_DOCS',2000));phased_ttt_num_phases=int(os.environ.get('PHASED_TTT_NUM_PHASES',1));global_ttt_lr=float(os.environ.get('GLOBAL_TTT_LR',0.001));global_ttt_momentum=float(os.environ.get('GLOBAL_TTT_MOMENTUM',0.9));global_ttt_epochs=int(os.environ.get('GLOBAL_TTT_EPOCHS',1));global_ttt_chunk_tokens=int(os.environ.get('GLOBAL_TTT_CHUNK_TOKENS',32768));global_ttt_batch_seqs=int(os.environ.get('GLOBAL_TTT_BATCH_SEQS',32));global_ttt_warmup_start_lr=float(os.environ.get('GLOBAL_TTT_WARMUP_START_LR',0.0));global_ttt_warmup_chunks=int(os.environ.get('GLOBAL_TTT_WARMUP_CHUNKS',0));global_ttt_grad_clip=float(os.environ.get('GLOBAL_TTT_GRAD_CLIP',0));ttt_eval_seq_len=int(os.environ.get('TTT_EVAL_SEQ_LEN',2048));ttt_eval_batches=os.environ.get('TTT_EVAL_BATCHES','');mos_k=int(os.environ.get('MOS_K',1));byte_weighted_ce=bool(int(os.environ.get('BYTE_WEIGHTED_CE','0')));batch_schedule_enabled=bool(int(os.environ.get('BATCH_SCHEDULE_ENABLED','0')));scale_tuning_enabled=bool(int(os.environ.get('SCALE_TUNING_ENABLED','0')));scale_tuning_steps=int(os.environ.get('SCALE_TUNING_STEPS',20));scale_tuning_lr=float(os.environ.get('SCALE_TUNING_LR',0.001));scale_tuning_batches=int(os.environ.get('SCALE_TUNING_BATCHES',8));distributed='RANK'in os.environ and'WORLD_SIZE'in os.environ;rank=int(os.environ.get('RANK','0'));world_size=int(os.environ.get('WORLD_SIZE','1'));local_rank=int(os.environ.get('LOCAL_RANK','0'));is_main_process=rank==0;grad_accum_steps=8//world_size;datasets_dir=os.path.join(data_dir,'datasets',f"fineweb10B_sp{vocab_size}");train_files=os.path.join(datasets_dir,'fineweb_train_*.bin');val_files=os.path.join(datasets_dir,'fineweb_val_*.bin');tokenizer_path=os.path.join(data_dir,'tokenizers',f"fineweb_{vocab_size}_bpe.model");logfile=f"logs/{run_id}.txt";model_path='final_model.pt';quantized_model_path='final_model.int6.ptz';kd_enabled=bool(int(os.environ.get('KD_ENABLED','0')));kd_alpha=float(os.environ.get('KD_ALPHA',0.5));kd_temperature=float(os.environ.get('KD_TEMPERATURE',2.0));kd_top_k=int(os.environ.get('KD_TOP_K',32));kd_logits_dir=os.environ.get('KD_LOGITS_DIR','');kd_warmup_frac=float(os.environ.get('KD_WARMUP_FRAC',0.0));max_eval_seconds=float(os.environ.get('MAX_EVAL_SECONDS',6e2))
_logger_hparams=None
def set_logging_hparams(h):global _logger_hparams;_logger_hparams=h
def log(msg,console=True):
	if _logger_hparams is None:print(msg);return
	if _logger_hparams.is_main_process:
		if console:print(msg)
		if _logger_hparams.logfile is not None:
			with open(_logger_hparams.logfile,'a',encoding='utf-8')as f:print(msg,file=f)
class ValidationData:
	def __init__(self,h,device):
		self.sp=spm.SentencePieceProcessor(model_file=h.tokenizer_path)
		if int(self.sp.vocab_size())!=h.vocab_size:raise ValueError(f"vocab mismatch")
		self.val_tokens=load_validation_tokens(h.val_files,h.eval_seq_len);self.base_bytes_lut,self.has_leading_space_lut,self.is_boundary_token_lut=build_sentencepiece_luts(self.sp,h.vocab_size,device)
def build_sentencepiece_luts(sp,vocab_size,device):
	sp_vocab_size=int(sp.vocab_size());assert sp.piece_to_id('▁')!=sp.unk_id(),"bad tokenizer";table_size=max(sp_vocab_size,vocab_size);base_bytes_np=np.zeros((table_size,),dtype=np.int16);has_leading_space_np=np.zeros((table_size,),dtype=np.bool_);is_boundary_token_np=np.ones((table_size,),dtype=np.bool_)
	for token_id in range(sp_vocab_size):
		if sp.is_control(token_id)or sp.is_unknown(token_id)or sp.is_unused(token_id):continue
		is_boundary_token_np[token_id]=False
		if sp.is_byte(token_id):base_bytes_np[token_id]=1;continue
		piece=sp.id_to_piece(token_id)
		if piece.startswith('▁'):has_leading_space_np[token_id]=True;piece=piece[1:]
		base_bytes_np[token_id]=len(piece.encode('utf-8'))
	return torch.tensor(base_bytes_np,dtype=torch.int16,device=device),torch.tensor(has_leading_space_np,dtype=torch.bool,device=device),torch.tensor(is_boundary_token_np,dtype=torch.bool,device=device)
def load_validation_tokens(pattern,seq_len):
	files=[Path(p)for p in sorted(glob.glob(pattern))]
	if not files:raise FileNotFoundError(f"no files: {pattern}")
	tokens=torch.cat([load_data_shard(file)for file in files]).contiguous();usable=(tokens.numel()-1)//seq_len*seq_len
	if usable<=0:raise ValueError(f"val too short")
	return tokens[:usable+1]
def load_data_shard(file):
	header_bytes=256*np.dtype('<i4').itemsize;token_bytes=np.dtype('<u2').itemsize;header=np.fromfile(file,dtype='<i4',count=256)
	if header.size!=256 or int(header[0])!=20240520 or int(header[1])!=1:raise ValueError(f"bad header")
	num_tokens=int(header[2]);expected_size=header_bytes+num_tokens*token_bytes
	if file.stat().st_size!=expected_size:raise ValueError(f"size mismatch")
	tokens_np=np.fromfile(file,dtype='<u2',count=num_tokens,offset=header_bytes)
	if tokens_np.size!=num_tokens:raise ValueError(f"short read")
	return torch.from_numpy(tokens_np.astype(np.uint16,copy=False))
_SHARD_HEADER_BYTES=256*np.dtype('<i4').itemsize
_SHARD_NTOKENS_CACHE={}
_MMAP_CACHE={}
def _read_num_tokens(file):
	key=str(file);cached=_SHARD_NTOKENS_CACHE.get(key)
	if cached is not None:return cached
	header=np.fromfile(file,dtype='<i4',count=256)
	if header.size!=256 or int(header[0])!=20240520 or int(header[1])!=1:raise ValueError(f"bad header")
	n=int(header[2]);_SHARD_NTOKENS_CACHE[key]=n;return n
def _get_shard_memmap(file):
	key=str(file);mm=_MMAP_CACHE.get(key)
	if mm is not None:return mm
	n=_read_num_tokens(file);mm=np.memmap(file,mode='r',dtype='<u2',offset=_SHARD_HEADER_BYTES,shape=(n,));_MMAP_CACHE[key]=mm;return mm
_TEACHER_MMAP_CACHE={}
def _get_teacher_logits_memmaps(shard_file, kd_logits_dir, top_k):
	"""Load teacher top-K indices and values memmaps for a given training shard."""
	key = str(shard_file)
	cached = _TEACHER_MMAP_CACHE.get(key)
	if cached is not None:
		return cached
	shard_name = Path(shard_file).stem
	topk_path = os.path.join(kd_logits_dir, f"{shard_name}_teacher_topk.npy")
	vals_path = os.path.join(kd_logits_dir, f"{shard_name}_teacher_vals.npy")
	if not os.path.exists(topk_path) or not os.path.exists(vals_path):
		_TEACHER_MMAP_CACHE[key] = None
		return None
	n = _read_num_tokens(shard_file) - 1  # num_predictions = num_tokens - 1
	topk_indices = np.memmap(topk_path, mode='r', dtype=np.uint16, shape=(n, top_k))
	topk_values = np.memmap(vals_path, mode='r', dtype=np.float16, shape=(n, top_k))
	result = (topk_indices, topk_values)
	_TEACHER_MMAP_CACHE[key] = result
	return result
class ShuffledSequenceLoader:
	def __init__(self,h,device):
		self.world_size=h.world_size;self.seq_len=h.train_seq_len;self.device=device;all_files=[Path(p)for p in sorted(glob.glob(h.train_files))]
		if not all_files:raise FileNotFoundError(f"no files")
		self.files=all_files[h.rank::h.world_size];self.rng=np.random.Generator(np.random.PCG64(h.rank));self.num_tokens=[_read_num_tokens(f)for f in self.files];self.start_inds=[[]for _ in self.files]
		self.kd_enabled=getattr(h,'kd_enabled',False);self.kd_logits_dir=getattr(h,'kd_logits_dir','');self.kd_top_k=getattr(h,'kd_top_k',32)
		for si in range(len(self.files)):self._reset_shard(si)
	def _reset_shard(self,si):max_phase=min(self.seq_len-1,max(0,self.num_tokens[si]-self.seq_len-1));phase=int(self.rng.integers(max_phase+1))if max_phase>0 else 0;num_sequences=(self.num_tokens[si]-1-phase)//self.seq_len;sequence_order=self.rng.permutation(num_sequences);self.start_inds[si]=(phase+sequence_order*self.seq_len).tolist()
	def next_batch(self,global_tokens,grad_accum_steps):
		device_tokens=global_tokens//(self.world_size*grad_accum_steps);device_batch_size=device_tokens//self.seq_len;remaining=np.array([len(s)for s in self.start_inds],dtype=np.float64);x=torch.empty((device_batch_size,self.seq_len),dtype=torch.int64);y=torch.empty((device_batch_size,self.seq_len),dtype=torch.int64)
		teacher_topk_idx=None;teacher_topk_val=None
		if self.kd_enabled:teacher_topk_idx=torch.empty((device_batch_size,self.seq_len,self.kd_top_k),dtype=torch.int64);teacher_topk_val=torch.empty((device_batch_size,self.seq_len,self.kd_top_k),dtype=torch.float32)
		for bi in range(device_batch_size):
			total=remaining.sum()
			if total<=0:
				for si in range(len(self.files)):self._reset_shard(si)
				remaining=np.array([len(s)for s in self.start_inds],dtype=np.float64);total=remaining.sum()
			probs=remaining/total;si=int(self.rng.choice(len(self.files),p=probs));start_ind=self.start_inds[si].pop();remaining[si]-=1;mm=_get_shard_memmap(self.files[si]);window=torch.as_tensor(np.array(mm[start_ind:start_ind+self.seq_len+1],dtype=np.int64));x[bi]=window[:-1];y[bi]=window[1:]
			if self.kd_enabled and teacher_topk_idx is not None:
				tmaps=_get_teacher_logits_memmaps(self.files[si],self.kd_logits_dir,self.kd_top_k)
				if tmaps is not None:
					t_idx_mm,t_val_mm=tmaps;t_start=start_ind;t_end=start_ind+self.seq_len
					if t_end<=t_idx_mm.shape[0]:teacher_topk_idx[bi]=torch.as_tensor(np.array(t_idx_mm[t_start:t_end],dtype=np.int64));teacher_topk_val[bi]=torch.as_tensor(np.array(t_val_mm[t_start:t_end],dtype=np.float32))
					else:teacher_topk_idx[bi]=0;teacher_topk_val[bi]=0.
				else:teacher_topk_idx[bi]=0;teacher_topk_val[bi]=0.
		result=(x.to(self.device,non_blocking=True),y.to(self.device,non_blocking=True))
		if self.kd_enabled and teacher_topk_idx is not None:result=result+(teacher_topk_idx.to(self.device,non_blocking=True),teacher_topk_val.to(self.device,non_blocking=True))
		return result
class RMSNorm(nn.Module):
	def __init__(self,eps=None):super().__init__();self.eps=eps
	def forward(self,x):return F.rms_norm(x,(x.size(-1),),eps=self.eps)
class CastedLinear(nn.Linear):
	def forward(self,x):
		w=self.weight.to(x.dtype)
		bias=self.bias.to(x.dtype)if self.bias is not None else None;return F.linear(x,w,bias)
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
_V_RESIDUAL_ENABLED=bool(int(os.environ.get('V_RESIDUAL_ENABLED','0')))
class CausalSelfAttention(nn.Module):
	def __init__(self,dim,num_heads,num_kv_heads,rope_base,qk_gain_init,train_seq_len):
		super().__init__()
		if dim%num_heads!=0:raise ValueError('dim%heads!=0')
		if num_heads%num_kv_heads!=0:raise ValueError('heads%kv!=0')
		self.num_heads=num_heads;self.num_kv_heads=num_kv_heads;self.head_dim=dim//num_heads
		if self.head_dim%2!=0:raise ValueError('head_dim odd')
		kv_dim=self.num_kv_heads*self.head_dim;self.c_q=CastedLinear(dim,dim,bias=False);self.c_k=CastedLinear(dim,kv_dim,bias=False);self.c_v=CastedLinear(dim,kv_dim,bias=False);self.proj=CastedLinear(dim,dim,bias=False);self.proj._zero_init=True;self.q_gain=nn.Parameter(torch.full((num_heads,),qk_gain_init,dtype=torch.float32));self.rope_dims=0;self.rotary=Rotary(self.head_dim,base=rope_base,train_seq_len=train_seq_len);self.use_xsa=False
		if _V_RESIDUAL_ENABLED:self.v_mix=nn.Parameter(torch.ones(num_kv_heads,dtype=torch.float32))
	def _xsa_efficient(self,y,v):B,T,H,D=y.shape;Hkv=v.size(-2);group=H//Hkv;y_g=y.reshape(B,T,Hkv,group,D);vn=F.normalize(v,dim=-1).unsqueeze(-2);proj=(y_g*vn).sum(dim=-1,keepdim=True)*vn;return(y_g-proj).reshape(B,T,H,D)
	def forward(self,x,v_residual=None):
		bsz,seqlen,dim=x.shape;q=self.c_q(x).reshape(bsz,seqlen,self.num_heads,self.head_dim);k=self.c_k(x).reshape(bsz,seqlen,self.num_kv_heads,self.head_dim);v=self.c_v(x).reshape(bsz,seqlen,self.num_kv_heads,self.head_dim)
		if _V_RESIDUAL_ENABLED and v_residual is not None:mix=torch.sigmoid(self.v_mix).to(dtype=v.dtype)[None,None,:,None];v=mix*v+(1.-mix)*v_residual
		v_out=v
		q=F.rms_norm(q,(q.size(-1),));k=F.rms_norm(k,(k.size(-1),));cos,sin=self.rotary(seqlen,x.device,q.dtype);q=apply_rotary_emb(q,cos,sin,self.rope_dims);k=apply_rotary_emb(k,cos,sin,self.rope_dims);q=q*self.q_gain.to(dtype=q.dtype)[None,None,:,None]
		if _HAS_FA3:y=flash_attn_3_func(q,k,v,causal=True)
		else:
			rep=self.num_heads//self.num_kv_heads
			if rep>1:k=k[:,:,:,None,:].expand(bsz,seqlen,self.num_kv_heads,rep,self.head_dim).reshape(bsz,seqlen,self.num_heads,self.head_dim);v=v[:,:,:,None,:].expand(bsz,seqlen,self.num_kv_heads,rep,self.head_dim).reshape(bsz,seqlen,self.num_heads,self.head_dim)
			y=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=True).transpose(1,2)
		if self.use_xsa:y=self._xsa_efficient(y,v)
		y=y.reshape(bsz,seqlen,dim);return self.proj(y),v_out
class MLP(nn.Module):
	def __init__(self,dim,mlp_mult):super().__init__();hidden=int(mlp_mult*dim);self.fc=CastedLinear(dim,hidden,bias=False);self.proj=CastedLinear(hidden,dim,bias=False);self.proj._zero_init=True;self.gated=bool(int(os.environ.get('GATED_MLP','0')))
	def forward(self,x):
		h=self.fc(x);act=F.leaky_relu(h,negative_slope=.5).square()
		if self.gated:act=act*torch.sigmoid(h)
		return self.proj(act)
class Block(nn.Module):
	def __init__(self,dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init,train_seq_len,layer_idx=0,ln_scale=False):super().__init__();self.attn_norm=RMSNorm();self.mlp_norm=RMSNorm();self.attn=CausalSelfAttention(dim,num_heads,num_kv_heads,rope_base,qk_gain_init,train_seq_len);self.mlp=MLP(dim,mlp_mult);self.attn_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.mlp_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.resid_mix=nn.Parameter(torch.stack((torch.ones(dim),torch.zeros(dim))).float());self.ln_scale_factor=1./math.sqrt(layer_idx+1)if ln_scale else 1.;self.parallel=False
	def forward(self,x,x0,v_residual=None):
		mix=self.resid_mix.to(dtype=x.dtype);x_in=mix[0][None,None,:]*x+mix[1][None,None,:]*x0;attn_out,v_out=self.attn(self.attn_norm(x_in)*self.ln_scale_factor,v_residual=v_residual)
		if self.parallel:mlp_out=self.mlp(self.mlp_norm(x_in)*self.ln_scale_factor);x_out=x_in+self.attn_scale.to(dtype=x_in.dtype)[None,None,:]*attn_out+self.mlp_scale.to(dtype=x_in.dtype)[None,None,:]*mlp_out
		else:x_out=x_in+self.attn_scale.to(dtype=x_in.dtype)[None,None,:]*attn_out;x_out=x_out+self.mlp_scale.to(dtype=x_out.dtype)[None,None,:]*self.mlp(self.mlp_norm(x_out)*self.ln_scale_factor)
		return x_out,v_out
def sparse_kd_loss(student_logits, teacher_topk_idx, teacher_topk_val, temperature, softcap=0.):
	"""Compute sparse KL-divergence loss between student and teacher (top-K only).

	Args:
		student_logits: [N, V] raw student logits (pre- or post-softcap)
		teacher_topk_idx: [N, K] top-K token indices from teacher
		teacher_topk_val: [N, K] top-K logit values from teacher (post-softcap)
		temperature: temperature for softening distributions
		softcap: if > 0, apply softcap to student logits first
	Returns:
		scalar KD loss (mean over batch)
	"""
	# Apply softcap to student if needed
	if softcap > 0.:
		student_logits = softcap * torch.tanh(student_logits / softcap)
	# Scale by temperature
	s_scaled = student_logits.float() / temperature
	t_scaled = teacher_topk_val.float() / temperature
	# Student log-softmax over full vocab (for log_sum_exp)
	s_log_softmax = s_scaled - torch.logsumexp(s_scaled, dim=-1, keepdim=True)
	# Teacher softmax over top-K only (renormalized)
	t_probs = F.softmax(t_scaled, dim=-1)
	# Gather student log-probs at teacher's top-K positions
	s_log_probs_at_topk = s_log_softmax.gather(1, teacher_topk_idx)
	# KL = sum_k t_prob_k * (log(t_prob_k) - s_log_prob_k)
	# = sum_k t_prob_k * log(t_prob_k) - sum_k t_prob_k * s_log_prob_k
	# First term is teacher entropy (constant wrt student), we include it for proper KL
	t_log_probs = torch.log(t_probs.clamp(min=1e-8))
	kl = (t_probs * (t_log_probs - s_log_probs_at_topk)).sum(dim=-1)
	return (kl * temperature * temperature).mean()
class GPT(nn.Module):
	def __init__(self,h):
		super().__init__()
		if h.logit_softcap<=.0:raise ValueError(f"bad softcap")
		self.tie_embeddings=h.tie_embeddings;self.tied_embed_init_std=h.tied_embed_init_std;self.logit_softcap=h.logit_softcap;self.fused_ce_enabled=_FUSED_CE_ENABLED;self.tok_emb=nn.Embedding(h.vocab_size,h.embedding_dim)
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
		self.hourglass_enabled=bool(int(os.environ.get('HOURGLASS_ENABLED','0')));self.hourglass_down_after=int(os.environ.get('HOURGLASS_DOWN_AFTER',3));self.hourglass_up_before=int(os.environ.get('HOURGLASS_UP_BEFORE',8));self.hourglass_factor=int(os.environ.get('HOURGLASS_FACTOR',2))
		if self.hourglass_enabled:self.hourglass_skip_gate=nn.Parameter(torch.zeros(h.model_dim,dtype=torch.float32))
		self.looping_active=False;num_loops=int(os.environ.get('NUM_LOOPS',1));self.loop_start_idx=int(os.environ.get('LOOP_START',3))
		if num_loops>0:
			ls=self.loop_start_idx;le=int(os.environ.get('LOOP_END',5));loop_seg=list(range(ls,le+1));all_idx=list(range(ls))
			for _ in range(num_loops+1):all_idx.extend(loop_seg)
			all_idx.extend(range(le+1,h.num_layers));ne=len(all_idx)//2;self.encoder_indices=all_idx[:ne];self.decoder_indices=all_idx[ne:]
		else:self.encoder_indices=list(range(self.num_encoder_layers));self.decoder_indices=list(range(self.num_encoder_layers,h.num_layers))
		self.num_skip_weights=min(len(self.encoder_indices),len(self.decoder_indices));self.skip_weights=nn.Parameter(torch.ones(self.num_skip_weights,h.model_dim,dtype=torch.float32));self.skip_gates=nn.Parameter(torch.zeros(self.num_skip_weights,h.model_dim,dtype=torch.float32))if h.skip_gates_enabled else None
		self.loop_gate_enabled=bool(int(os.environ.get('LOOP_GATE_ENABLED','0')))
		if self.loop_gate_enabled:self.loop_gate=nn.Parameter(torch.full((h.model_dim,),-2.0,dtype=torch.float32))
		else:self.loop_gate=None
		self.mos_k=h.mos_k
		if self.mos_k>1:
			self.mos_scales=nn.ParameterList([nn.Parameter(torch.ones(h.embedding_dim,dtype=torch.float32))for _ in range(self.mos_k)])
			self.mos_gate=nn.Parameter(torch.zeros(h.embedding_dim,self.mos_k,dtype=torch.float32))
		self._init_weights()
	def _init_weights(self):
		if self.tie_embeddings:nn.init.normal_(self.tok_emb.weight,mean=.0,std=self.tied_embed_init_std)
		for(name,module)in self.named_modules():
			if isinstance(module,nn.Linear):
				if getattr(module,'_zero_init',False):nn.init.zeros_(module.weight)
				elif module.weight.ndim==2 and module.weight.shape[0]>=64 and module.weight.shape[1]>=64:nn.init.orthogonal_(module.weight,gain=1.)
	def _hourglass_downsample(self,x):
		"""Downsample sequence T -> T/factor via avg_pool1d. x: [B, T, D] -> [B, T//factor, D]"""
		f=self.hourglass_factor
		return F.avg_pool1d(x.transpose(1,2),kernel_size=f,stride=f).transpose(1,2)
	def _hourglass_upsample(self,x,target_len):
		"""Upsample sequence back to target_len via repeat_interleave. x: [B, T_down, D] -> [B, target_len, D]"""
		x_up=x.repeat_interleave(self.hourglass_factor,dim=1)
		if x_up.size(1)>target_len:x_up=x_up[:,:target_len,:]
		return x_up
	def _forward_hidden(self,input_ids):
		"""Forward through all blocks, returns hidden states before final linear projection."""
		x=self.tok_emb(input_ids);x=F.rms_norm(x,(x.size(-1),))
		if self.embed_proj is not None:x=self.embed_proj(x)
		x0=x;skips=[];v_residual=None;enc_iter=self.encoder_indices if self.looping_active else range(self.num_encoder_layers);dec_iter=self.decoder_indices if self.looping_active else range(self.num_encoder_layers,self.num_encoder_layers+self.num_decoder_layers);is_first_layer=True;seen_loop_start=False;x_at_loop_start=None
		hg=self.hourglass_enabled;full_len=x.size(1);hg_skip=None;in_downsampled=False
		for step_idx,i in enumerate(enc_iter):
			if self.loop_gate is not None and self.looping_active and i==self.loop_start_idx:
				if not seen_loop_start:seen_loop_start=True;x_at_loop_start=x
				else:g=torch.sigmoid(self.loop_gate.to(dtype=x.dtype))[None,None,:];x=(1-g)*x+g*x_at_loop_start
			x,v_out=self.blocks[i](x,x0,v_residual=v_residual)
			if _V_RESIDUAL_ENABLED and is_first_layer:v_residual=v_out;is_first_layer=False
			skips.append(x)
			if hg and not in_downsampled and step_idx==self.hourglass_down_after:
				hg_skip=x;x=self._hourglass_downsample(x);x0=self._hourglass_downsample(x0);in_downsampled=True
				if x_at_loop_start is not None:x_at_loop_start=self._hourglass_downsample(x_at_loop_start)
		for(skip_idx,i)in enumerate(dec_iter):
			if hg and in_downsampled and skip_idx==self.hourglass_up_before:
				x=self._hourglass_upsample(x,full_len);x0=self._hourglass_upsample(x0,full_len);in_downsampled=False
				if x_at_loop_start is not None:x_at_loop_start=self._hourglass_upsample(x_at_loop_start,full_len)
				sg=torch.sigmoid(self.hourglass_skip_gate.to(dtype=x.dtype))[None,None,:];x=(1-sg)*x+sg*hg_skip
			if self.loop_gate is not None and self.looping_active and i==self.loop_start_idx:
				if not seen_loop_start:seen_loop_start=True;x_at_loop_start=x
				else:g=torch.sigmoid(self.loop_gate.to(dtype=x.dtype))[None,None,:];x=(1-g)*x+g*x_at_loop_start
			if skip_idx<self.num_skip_weights and skips:
				scaled_skip=skips.pop()
				if hg and scaled_skip.size(1)!=x.size(1):
					if scaled_skip.size(1)>x.size(1):scaled_skip=self._hourglass_downsample(scaled_skip)
					else:scaled_skip=self._hourglass_upsample(scaled_skip,x.size(1))
				scaled_skip=self.skip_weights[skip_idx].to(dtype=x.dtype)[None,None,:]*scaled_skip
				if self.skip_gates is not None:g=torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None,None,:];x=torch.lerp(scaled_skip,x,g)
				else:x=x+scaled_skip
			x,_=self.blocks[i](x,x0,v_residual=v_residual)
		if hg and in_downsampled:x=self._hourglass_upsample(x,full_len)
		x=self.final_norm(x)
		if self.head_proj is not None:x=self.head_proj(x)
		return x
	def _forward_pre_softcap(self,input_ids):
		x=self._forward_hidden(input_ids)
		if self.tie_embeddings:return F.linear(x,self.tok_emb.weight)
		else:return self.lm_head(x)
	def _mos_loss(self,x,target_ids,reduction='mean'):
		"""Mixture of Softmax loss. x is pre-projection hidden states [*, dim]."""
		tok_w=self.tok_emb.weight if self.tie_embeddings else self.lm_head.weight
		sc=self.logit_softcap;K=self.mos_k
		# mixing weights: [*, K]
		pi=F.softmax(x@self.mos_gate.to(dtype=x.dtype),dim=-1)
		# compute K softmax distributions and mix
		p_mixed=torch.zeros(*x.shape[:-1],tok_w.shape[0],device=x.device,dtype=torch.float32)
		for k in range(K):
			x_k=x*self.mos_scales[k].to(dtype=x.dtype)[None,None,:]if x.ndim==3 else x*self.mos_scales[k].to(dtype=x.dtype)
			logits_k=F.linear(x_k,tok_w)
			logits_k=sc*torch.tanh(logits_k/sc)
			p_k=F.softmax(logits_k.float(),dim=-1)
			p_mixed=p_mixed+pi[...,k:k+1].float()*p_k
		# loss = -log(p_mixed[target])
		p_mixed=p_mixed.clamp(min=1e-8)
		flat_p=p_mixed.reshape(-1,p_mixed.size(-1))
		flat_t=target_ids.reshape(-1)
		losses=-torch.log(flat_p.gather(1,flat_t.unsqueeze(1)).squeeze(1))
		if reduction=='mean':return losses.mean()
		return losses
	def forward_logits(self,input_ids):
		if self.mos_k>1:
			x=self._forward_hidden(input_ids);tok_w=self.tok_emb.weight if self.tie_embeddings else self.lm_head.weight;sc=self.logit_softcap
			pi=F.softmax(x@self.mos_gate.to(dtype=x.dtype),dim=-1)
			p_mixed=torch.zeros(*x.shape[:-1],tok_w.shape[0],device=x.device,dtype=torch.float32)
			for k in range(self.mos_k):
				x_k=x*self.mos_scales[k].to(dtype=x.dtype)[None,None,:]
				logits_k=sc*torch.tanh(F.linear(x_k,tok_w)/sc)
				p_mixed=p_mixed+pi[...,k:k+1].float()*F.softmax(logits_k.float(),dim=-1)
			return torch.log(p_mixed.clamp(min=1e-8))
		logits_proj=self._forward_pre_softcap(input_ids);return self.logit_softcap*torch.tanh(logits_proj/self.logit_softcap)
	def forward(self,input_ids,target_ids,byte_weights=None,teacher_topk_idx=None,teacher_topk_val=None,kd_alpha=0.,kd_temperature=2.):
		if self.mos_k>1:
			x=self._forward_hidden(input_ids)
			if byte_weights is not None:
				losses=self._mos_loss(x,target_ids,reduction='none')
				bw=byte_weights.reshape(-1).float();return(losses*bw).sum()/bw.sum()
			return self._mos_loss(x,target_ids,reduction='mean')
		# Knowledge distillation path
		if teacher_topk_idx is not None and kd_alpha>0.:
			logits_proj=self._forward_pre_softcap(input_ids);flat_logits=logits_proj.reshape(-1,logits_proj.size(-1));flat_targets=target_ids.reshape(-1)
			if self.fused_ce_enabled:ce_loss=softcapped_cross_entropy(flat_logits,flat_targets,self.logit_softcap,reduction='mean')
			else:logits_sc=self.logit_softcap*torch.tanh(flat_logits/self.logit_softcap);ce_loss=F.cross_entropy(logits_sc.float(),flat_targets,reduction='mean')
			flat_t_idx=teacher_topk_idx.reshape(-1,teacher_topk_idx.size(-1));flat_t_val=teacher_topk_val.reshape(-1,teacher_topk_val.size(-1))
			kd_loss=sparse_kd_loss(flat_logits,flat_t_idx,flat_t_val,kd_temperature,softcap=self.logit_softcap)
			return(1.-kd_alpha)*ce_loss+kd_alpha*kd_loss
		if byte_weights is not None:
			if self.fused_ce_enabled:
				logits_proj=self._forward_pre_softcap(input_ids)
				losses=softcapped_cross_entropy(logits_proj.reshape(-1,logits_proj.size(-1)),target_ids.reshape(-1),self.logit_softcap,reduction='none')
			else:
				logits=self.forward_logits(input_ids);losses=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),target_ids.reshape(-1),reduction='none')
			bw=byte_weights.reshape(-1).float();return(losses*bw).sum()/bw.sum()
		if self.fused_ce_enabled:
			logits_proj=self._forward_pre_softcap(input_ids)
			return softcapped_cross_entropy(logits_proj.reshape(-1,logits_proj.size(-1)),target_ids.reshape(-1),self.logit_softcap,reduction='mean')
		logits=self.forward_logits(input_ids);return F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),target_ids.reshape(-1),reduction='mean')
	def forward_ttt(self, input_ids, target_ids, lora):
		"""Forward pass with batched LoRA adapters for TTT. Returns per-token loss [bsz, seq_len]."""
		x = self.tok_emb(input_ids)
		x = F.rms_norm(x, (x.size(-1),))
		if self.embed_proj is not None:
			x = self.embed_proj(x)
		x0 = x
		skips = []
		v_residual = None
		enc_iter = self.encoder_indices if self.looping_active else range(self.num_encoder_layers)
		dec_iter = self.decoder_indices if self.looping_active else range(self.num_encoder_layers, self.num_encoder_layers + self.num_decoder_layers)
		slot = 0
		is_first_layer = True
		seen_loop_start = False
		x_at_loop_start = None
		hg = self.hourglass_enabled
		full_len = x.size(1)
		hg_skip = None
		in_downsampled = False
		for step_idx, i in enumerate(enc_iter):
			if self.loop_gate is not None and self.looping_active and i == self.loop_start_idx:
				if not seen_loop_start:
					seen_loop_start = True
					x_at_loop_start = x
				else:
					g = torch.sigmoid(self.loop_gate.to(dtype=x.dtype))[None, None, :]
					x = (1 - g) * x + g * x_at_loop_start
			x, v_out = self._block_with_lora(self.blocks[i], x, x0, lora, slot, v_residual=v_residual)
			if _V_RESIDUAL_ENABLED and is_first_layer:
				v_residual = v_out
				is_first_layer = False
			slot += 1
			skips.append(x)
			if hg and not in_downsampled and step_idx == self.hourglass_down_after:
				hg_skip = x
				x = self._hourglass_downsample(x)
				x0 = self._hourglass_downsample(x0)
				in_downsampled = True
				if x_at_loop_start is not None:
					x_at_loop_start = self._hourglass_downsample(x_at_loop_start)
		for skip_idx, i in enumerate(dec_iter):
			if hg and in_downsampled and skip_idx == self.hourglass_up_before:
				x = self._hourglass_upsample(x, full_len)
				x0 = self._hourglass_upsample(x0, full_len)
				in_downsampled = False
				if x_at_loop_start is not None:
					x_at_loop_start = self._hourglass_upsample(x_at_loop_start, full_len)
				sg = torch.sigmoid(self.hourglass_skip_gate.to(dtype=x.dtype))[None, None, :]
				x = (1 - sg) * x + sg * hg_skip
			if self.loop_gate is not None and self.looping_active and i == self.loop_start_idx:
				if not seen_loop_start:
					seen_loop_start = True
					x_at_loop_start = x
				else:
					g = torch.sigmoid(self.loop_gate.to(dtype=x.dtype))[None, None, :]
					x = (1 - g) * x + g * x_at_loop_start
			if skip_idx < self.num_skip_weights and skips:
				scaled_skip = skips.pop()
				if hg and scaled_skip.size(1) != x.size(1):
					if scaled_skip.size(1) > x.size(1):
						scaled_skip = self._hourglass_downsample(scaled_skip)
					else:
						scaled_skip = self._hourglass_upsample(scaled_skip, x.size(1))
				scaled_skip = self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * scaled_skip
				if self.skip_gates is not None:
					g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
					x = torch.lerp(scaled_skip, x, g)
				else:
					x = x + scaled_skip
			x, _ = self._block_with_lora(self.blocks[i], x, x0, lora, slot, v_residual=v_residual)
			slot += 1
		if hg and in_downsampled:
			x = self._hourglass_upsample(x, full_len)
		x = self.final_norm(x)
		if self.head_proj is not None:
			x = self.head_proj(x)
		if self.mos_k > 1:
			tok_w = self.tok_emb.weight if self.tie_embeddings else self.lm_head.weight
			sc = self.logit_softcap
			pi = F.softmax(x @ self.mos_gate.to(dtype=x.dtype), dim=-1)
			bsz, sl, dim = x.shape
			V = tok_w.shape[0]
			p_mixed = torch.zeros(bsz, sl, V, device=x.device, dtype=torch.float32)
			for k in range(self.mos_k):
				x_k = x * self.mos_scales[k].to(dtype=x.dtype)[None, None, :]
				logits_k = F.linear(x_k, tok_w) + lora.lm_head_lora(x_k)
				logits_k = sc * torch.tanh(logits_k / sc)
				p_k = F.softmax(logits_k.float(), dim=-1)
				p_mixed = p_mixed + pi[..., k:k+1].float() * p_k
			p_mixed = p_mixed.clamp(min=1e-8)
			flat_p = p_mixed.reshape(-1, V)
			flat_t = target_ids.reshape(-1)
			losses = -torch.log(flat_p.gather(1, flat_t.unsqueeze(1)).squeeze(1))
			return losses.reshape(bsz, sl)
		if self.tie_embeddings:
			logits = F.linear(x, self.tok_emb.weight)
		else:
			logits = self.lm_head(x)
		logits = logits + lora.lm_head_lora(x)
		logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
		bsz, sl, V = logits.shape
		return F.cross_entropy(
			logits.float().reshape(-1, V), target_ids.reshape(-1), reduction="none"
		).reshape(bsz, sl)
	def _block_with_lora(self, block, x, x0, lora, slot, v_residual=None):
		"""Single block forward with LoRA injection, handles both parallel and sequential."""
		mix = block.resid_mix.to(dtype=x.dtype)
		x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
		n = block.attn_norm(x_in) * block.ln_scale_factor
		attn = block.attn
		bsz, seqlen, dim = n.shape
		# Q with LoRA
		q = (attn.c_q(n) + lora.q_loras[slot](n)).reshape(bsz, seqlen, attn.num_heads, attn.head_dim)
		# K with optional LoRA
		k = attn.c_k(n)
		if lora.k_loras is not None:
			k = k + lora.k_loras[slot](n)
		k = k.reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
		# V with LoRA
		v = (attn.c_v(n) + lora.v_loras[slot](n)).reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
		if _V_RESIDUAL_ENABLED and v_residual is not None:
			v_mix = torch.sigmoid(attn.v_mix).to(dtype=v.dtype)[None, None, :, None]
			v = v_mix * v + (1.0 - v_mix) * v_residual
		v_out = v
		q = F.rms_norm(q, (q.size(-1),))
		k = F.rms_norm(k, (k.size(-1),))
		cos, sin = attn.rotary(seqlen, n.device, q.dtype)
		q = apply_rotary_emb(q, cos, sin, attn.rope_dims)
		k = apply_rotary_emb(k, cos, sin, attn.rope_dims)
		q = q * attn.q_gain.to(dtype=q.dtype)[None, None, :, None]
		if _HAS_FA3:
			y = flash_attn_3_func(q, k, v, causal=True)
		else:
			rep = attn.num_heads // attn.num_kv_heads
			if rep > 1:
				k = k[:,:,:,None,:].expand(bsz,seqlen,attn.num_kv_heads,rep,attn.head_dim).reshape(bsz,seqlen,attn.num_heads,attn.head_dim)
				v = v[:,:,:,None,:].expand(bsz,seqlen,attn.num_kv_heads,rep,attn.head_dim).reshape(bsz,seqlen,attn.num_heads,attn.head_dim)
			y = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=True).transpose(1,2)
		if attn.use_xsa:
			y = attn._xsa_efficient(y, v)
		y = y.reshape(bsz, seqlen, dim)
		attn_out = attn.proj(y)
		if lora.o_loras is not None:
			attn_out = attn_out + lora.o_loras[slot](y)
		if block.parallel:
			mlp_n = block.mlp_norm(x_in) * block.ln_scale_factor
			mlp_out = block.mlp(mlp_n)
			if lora.mlp_loras is not None:
				mlp_out = mlp_out + lora.mlp_loras[slot](mlp_out)
			x_out = x_in + block.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out + block.mlp_scale.to(dtype=x_in.dtype)[None, None, :] * mlp_out
		else:
			x_out = x_in + block.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
			mlp_n = block.mlp_norm(x_out) * block.ln_scale_factor
			mlp_out = block.mlp(mlp_n)
			if lora.mlp_loras is not None:
				mlp_out = mlp_out + lora.mlp_loras[slot](mlp_out)
			x_out = x_out + block.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * mlp_out
		return x_out, v_out
def classify_param(name):
	if'tok_emb'in name or'lm_head'in name:return'embed'
	if'.mlp.'in name:return'mlp'
	if'.attn.'in name or'.proj.'in name and'.mlp.'not in name:return'attn'
	return'other'
@torch.compile
def zeropower_via_newtonschulz5(G,steps=5,eps=1e-07):
	coeffs=[(8.156554524902461,-22.48329292557795,15.878769915207462),(4.042929935166739,-2.808917465908714,0.5000178451051316),(3.8916678022926607,-2.772484153217685,0.5060648178503393),(3.285753657755655,-2.3681294933425376,0.46449024233003106),(2.3465413258596377,-1.7097828382687081,0.42323551169305323)]
	X=G.bfloat16();X/=X.norm()+eps;transposed=G.size(0)>G.size(1)
	if transposed:X=X.T
	for a,b,c in coeffs[:steps]:A=X@X.T;B=b*A+c*A@A;X=a*X+B@X
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
CONTROL_TENSOR_NAME_PATTERNS=tuple(pattern for pattern in os.environ.get('CONTROL_TENSOR_NAME_PATTERNS','attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates,v_mix,loop_gate').split(',')if pattern)
class Optimizers:
	def __init__(self,h,base_model):
		block_named_params=list(base_model.blocks.named_parameters());matrix_params=[p for(name,p)in block_named_params if p.ndim==2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)];scalar_params=[p for(name,p)in block_named_params if p.ndim<2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)]
		if base_model.skip_weights.numel()>0:scalar_params.append(base_model.skip_weights)
		if base_model.skip_gates is not None and base_model.skip_gates.numel()>0:scalar_params.append(base_model.skip_gates)
		if base_model.loop_gate is not None:scalar_params.append(base_model.loop_gate)
		if getattr(base_model,'mos_k',1)>1:
			for s in base_model.mos_scales:scalar_params.append(s)
			scalar_params.append(base_model.mos_gate)
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
def _gptq_core(W_orig,H_prepared,Hinv,perm,invperm,s,clip_range,block_size):
	rows,cols=W_orig.shape;sf=s.float();Q=torch.zeros(rows,cols,dtype=torch.int8);W_work=W_orig[:,perm].clone()
	for i1 in range(0,cols,block_size):
		i2=min(i1+block_size,cols);W_block=W_work[:,i1:i2].clone();Hinv_block=Hinv[i1:i2,i1:i2];Err=torch.zeros(rows,i2-i1)
		for j in range(i2-i1):w_col=W_block[:,j];d=Hinv_block[j,j];q_col=torch.clamp(torch.round(w_col/sf),-clip_range,clip_range);Q[:,i1+j]=q_col.to(torch.int8);err=(w_col-q_col.float()*sf)/d;Err[:,j]=err;W_block[:,j:]-=err.unsqueeze(1)*Hinv_block[j,j:].unsqueeze(0)
		if i2<cols:W_work[:,i2:]-=Err@Hinv[i1:i2,i2:]
	deq=(Q[:,invperm].float()*sf.view(rows,1));mse=((W_orig[:,invperm if False else slice(None)]-deq)**2).mean()
	return Q[:,invperm],s,mse
def gptq_quantize_weight(w,H,clip_sigmas=3.,clip_range=63,block_size=128,hessian_clip_lambda=0.):
	W_orig=w.float().clone();rows,cols=W_orig.shape;H=H.float().clone();dead=torch.diag(H)==0;H[dead,dead]=1;damp=.01*H.diag().mean();H.diagonal().add_(damp);perm=torch.argsort(H.diag(),descending=True);invperm=torch.argsort(perm);W_perm=W_orig[:,perm].clone();W_perm[:,dead[perm]]=0
	H_p=H[perm][:,perm];Hinv=torch.cholesky_inverse(torch.linalg.cholesky(H_p));Hinv=torch.linalg.cholesky(Hinv,upper=True)
	use_search=bool(int(os.environ.get('GPTQ_PERCENTILE_SEARCH','0')))
	if use_search:
		percentiles=[0.999,0.9995,0.9999,0.99999,1.0];best_q=None;best_s=None;best_mse=float('inf')
		for pct in percentiles:
			if pct>=1.:s_cand=(W_orig.abs().max(dim=1).values/clip_range).clamp_min(1e-10).to(torch.float16)
			else:s_cand=(torch.quantile(W_orig.abs(),pct,dim=1)/clip_range).clamp_min(1e-10).to(torch.float16)
			sf_c=s_cand.float();Q_c=torch.zeros(rows,cols,dtype=torch.int8);W_work_c=W_perm.clone()
			for i1 in range(0,cols,block_size):
				i2=min(i1+block_size,cols);W_block=W_work_c[:,i1:i2].clone();Hinv_block=Hinv[i1:i2,i1:i2];Err=torch.zeros(rows,i2-i1)
				for j in range(i2-i1):w_col=W_block[:,j];d=Hinv_block[j,j];q_col=torch.clamp(torch.round(w_col/sf_c),-clip_range,clip_range);Q_c[:,i1+j]=q_col.to(torch.int8);err=(w_col-q_col.float()*sf_c)/d;Err[:,j]=err;W_block[:,j:]-=err.unsqueeze(1)*Hinv_block[j,j:].unsqueeze(0)
				if i2<cols:W_work_c[:,i2:]-=Err@Hinv[i1:i2,i2:]
			deq=(Q_c[:,invperm].float()*sf_c.view(rows,1));mse=((W_orig-deq)**2).mean().item()
			if mse<best_mse:best_q,best_s,best_mse=Q_c[:,invperm],s_cand,mse
		return best_q,best_s
	row_std=W_orig.std(dim=1)
	if hessian_clip_lambda>0.:
		diagH=torch.diag(H).clamp_min(1e-8);col_imp=diagH/diagH.mean();row_imp=(W_orig.abs()*col_imp.unsqueeze(0)).mean(dim=1);row_imp=row_imp/row_imp.mean();adj=1.+hessian_clip_lambda*(row_imp-1.);s=(clip_sigmas*row_std*adj/clip_range).clamp_min(1e-10).to(torch.float16)
	else:s=(clip_sigmas*row_std/clip_range).clamp_min(1e-10).to(torch.float16)
	sf=s.float();Q=torch.zeros(rows,cols,dtype=torch.int8);W_work=W_perm.clone()
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
		cs=h.embed_clip_sigmas if'tok_emb'in name else h.matrix_clip_sigmas;bits=h.embed_bits if'tok_emb'in name else h.matrix_bits;hcl=0. if'tok_emb'in name else h.hessian_clip_lambda;q,s=gptq_quantize_weight(t,hessians[name],clip_sigmas=cs,clip_range=2**(bits-1)-1,hessian_clip_lambda=hcl);result[name+'.q']=q;result[name+'.scale']=s;meta[name]=f"gptq (int{bits})"
	categories=collections.defaultdict(set)
	for(name,cat)in meta.items():short=re.sub('\\.\\d+$','',re.sub('blocks\\.\\d+','blocks',name));categories[cat].add(short)
	log('Quantized weights:')
	for cat in sorted(categories):log(f"  {cat}: {", ".join(sorted(categories[cat]))}")
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
def scale_tune_post_gptq(quant_result, quant_meta, template_sd, h, device):
	"""Optimize per-row quantization scales using actual CE loss on calibration data.

	After GPTQ produces initial scales, this function makes the scales learnable
	and fine-tunes them via backprop through the dequantized model. The integer
	quantized weights (Q_int) stay frozen; only the float16 scale tensors are updated.
	This minimizes the actual CE loss instead of the per-layer MSE heuristic that GPTQ uses.
	"""
	log(f"Scale tuning: {h.scale_tuning_steps} steps, lr={h.scale_tuning_lr}, batches={h.scale_tuning_batches}")
	t0 = time.perf_counter()

	# Build a shell model for functional_call (weights will be overridden)
	tune_model = GPT(h).to(device).bfloat16()
	restore_fp32_params(tune_model)
	tune_model.eval()

	# Collect the frozen Q_int tensors and learnable scale params
	scale_params = []
	q_int_map = {}  # name -> frozen Q_int tensor on device

	for name, info in quant_meta.items():
		if 'gptq' not in info:
			continue
		q_key = name + '.q'
		s_key = name + '.scale'
		q_int_map[name] = quant_result[q_key].to(device)  # frozen int8
		# Make scale a learnable parameter (float32 for optimizer stability)
		scale_params.append((name, quant_result[s_key].float().to(device).requires_grad_(True)))

	# Build optimizer over scale params only
	optim_params = [sp for _, sp in scale_params]
	optimizer = torch.optim.Adam(optim_params, lr=h.scale_tuning_lr)

	# Load calibration data
	calib_loader = ShuffledSequenceLoader(h, device)

	# Pre-collect calibration batches (small number, reuse across steps)
	calib_data = []
	for _ in range(h.scale_tuning_batches):
		x, y = calib_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
		calib_data.append((x, y))

	best_loss = float('inf')
	best_scales = {name: sp.detach().clone() for name, sp in scale_params}

	for step in range(h.scale_tuning_steps):
		optimizer.zero_grad()
		total_loss = 0.0

		# Build dequantized state dict (differentiable through scales)
		deq_sd = {}
		for pname, orig in template_sd.items():
			info = quant_meta.get(pname)
			if info is None:
				continue
			if 'passthrough' in info:
				t = quant_result[pname]
				if t.dtype == torch.float16 and orig.dtype in (torch.float32, torch.bfloat16):
					t = t.to(orig.dtype)
				deq_sd[pname] = t.to(device)
				continue
			q_int = q_int_map[pname]
			scale_tensor = None
			for sname, sp in scale_params:
				if sname == pname:
					scale_tensor = sp
					break
			if scale_tensor is not None:
				deq_sd[pname] = (q_int.float() * scale_tensor.view(q_int.shape[0], 1)).to(orig.dtype)
			else:
				deq_sd[pname] = (q_int.float() * quant_result[pname + '.scale'].float().to(device).view(q_int.shape[0], 1)).to(orig.dtype)

		# Forward pass via functional_call (keeps gradients flowing through scales)
		for batch_idx in range(len(calib_data)):
			inp, tgt = calib_data[batch_idx]
			with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
				loss = torch.func.functional_call(tune_model, deq_sd, (inp, tgt))
			total_loss += loss.item()
			(loss / len(calib_data)).backward()

		avg_loss = total_loss / len(calib_data)
		optimizer.step()

		# Clamp scales to stay positive
		with torch.no_grad():
			for _, sp in scale_params:
				sp.clamp_min_(1e-10)

		if avg_loss < best_loss:
			best_loss = avg_loss
			best_scales = {name: sp.detach().clone() for name, sp in scale_params}

		if step % 5 == 0 or step == h.scale_tuning_steps - 1:
			log(f"  scale_tune step {step}: CE loss = {avg_loss:.6f}")

	# Write optimized scales back into quant_result
	for name, sp in best_scales.items():
		quant_result[name + '.scale'] = sp.cpu().to(torch.float16)

	elapsed = time.perf_counter() - t0
	log(f"Scale tuning done in {elapsed:.1f}s, best CE = {best_loss:.6f}")
	return quant_result

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
	if compressor=='lzma':return lzma.compress(data,preset=6)
	elif compressor=='brotli':import brotli;return brotli.compress(data,quality=11)
	raise ValueError(f"bad compressor")
def _decompress(data,compressor):
	if compressor=='lzma':raw=lzma.decompress(data)
	elif compressor=='brotli':import brotli;raw=brotli.decompress(data)
	else:raise ValueError(f"bad compressor")
	raw=_byte_unshuffle(raw);return raw
def serialize(h,base_model,code):
	code_bytes=len(code.encode('utf-8'))
	if h.is_main_process:torch.save(base_model.state_dict(),h.model_path);model_bytes=os.path.getsize(h.model_path);log(f"Serialized model: {model_bytes} bytes");log(f"Code size: {code_bytes} bytes")
	sd_cpu={k:v.detach().cpu()for(k,v)in base_model.state_dict().items()};device=torch.device('cuda',h.local_rank);log('GPTQ:collecting Hessians from calibration data...');t0=time.perf_counter();calib_loader=ShuffledSequenceLoader(h,device);hessians=collect_hessians(base_model,calib_loader,h,device,n_calibration_batches=h.gptq_calibration_batches);log(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter()-t0:.1f}s");quant_result,quant_meta=gptq_mixed_quantize(sd_cpu,hessians,h)
	if h.scale_tuning_enabled:quant_result=scale_tune_post_gptq(quant_result,quant_meta,sd_cpu,h,device)
	target_bytes=int(os.environ.get('TARGET_BYTES',16000000))-code_bytes
	quant_buf=io.BytesIO();torch.save({'w':quant_result,'m':quant_meta},quant_buf);quant_raw=quant_buf.getvalue();quant_blob=_compress(quant_raw,h.compressor)
	if len(quant_blob)>target_bytes:
		over=len(quant_blob)-target_bytes;log(f"prune:over by {over} bytes, selective pruning")
		candidates=[]
		for name,info in quant_meta.items():
			if'gptq'not in info:continue
			q=quant_result[name+'.q'];s=quant_result[name+'.scale']
			sf=s.float().view(-1) if s.ndim==1 else s.float()[:,0]
			for r in range(q.shape[0]):
				for c in range(q.shape[1]):
					v=int(q[r,c])
					if 0<abs(v)<=2:candidates.append((float(sf[r])**2*v*v,name,r,c))
		candidates.sort();n_zero=min(len(candidates),int(over*12));log(f"prune:{len(candidates)} cand, zeroing {n_zero}")
		for _,name,r,c in candidates[:n_zero]:quant_result[name+'.q'][r,c]=0
		quant_buf=io.BytesIO();torch.save({'w':quant_result,'m':quant_meta},quant_buf);quant_raw=quant_buf.getvalue();quant_blob=_compress(quant_raw,h.compressor)
		log(f"prune:zeroed {n_zero}, now {len(quant_blob)} bytes")
	quant_file_bytes=len(quant_blob);bytes_total=quant_file_bytes+code_bytes
	if h.is_main_process:
		with open(h.quantized_model_path,'wb')as f:f.write(quant_blob)
		log(f"Serialized model quantized+{h.compressor}: {quant_file_bytes} bytes");log(f"Total submission size quantized+{h.compressor}: {bytes_total} bytes")
	return bytes_total,quant_file_bytes
def deserialize(h,device):
	eval_model=GPT(h).to(device).bfloat16();restore_fp32_params(eval_model);sd_cpu={k:v.detach().cpu()for(k,v)in eval_model.state_dict().items()}
	with open(h.quantized_model_path,'rb')as f:quant_blob_disk=f.read()
	quant_state=torch.load(io.BytesIO(_decompress(quant_blob_disk,h.compressor)),map_location='cpu');deq_state=dequantize_mixed(quant_state['w'],quant_state['m'],sd_cpu);eval_model.load_state_dict(deq_state,strict=True);return eval_model
def _loss_bpb(loss_sum,token_count,byte_count):val_loss=(loss_sum/token_count).item();val_bpb=val_loss/math.log(2.)*(token_count.item()/byte_count.item());return val_loss,val_bpb
def eval_val(h,device,val_data,model):
	seq_len=h.eval_seq_len;local_batch_tokens=h.val_batch_tokens//(h.world_size*h.grad_accum_steps)
	if local_batch_tokens<seq_len:raise ValueError(f"batch too small")
	local_batch_seqs=local_batch_tokens//seq_len;total_seqs=(val_data.val_tokens.numel()-1)//seq_len;seq_start=total_seqs*h.rank//h.world_size;seq_end=total_seqs*(h.rank+1)//h.world_size;val_loss_sum=torch.zeros((),device=device,dtype=torch.float64);val_token_count=torch.zeros((),device=device,dtype=torch.float64);val_byte_count=torch.zeros((),device=device,dtype=torch.float64);model.eval()
	with torch.inference_mode():
		for batch_seq_start in range(seq_start,seq_end,local_batch_seqs):
			batch_seq_end=min(batch_seq_start+local_batch_seqs,seq_end);raw_start=batch_seq_start*seq_len;raw_end=batch_seq_end*seq_len+1;local=val_data.val_tokens[raw_start:raw_end].to(device=device,dtype=torch.int64,non_blocking=True);x=local[:-1].reshape(-1,seq_len);y=local[1:].reshape(-1,seq_len)
			with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):batch_loss=model(x,y).detach()
			batch_token_count=float(y.numel());val_loss_sum+=batch_loss.to(torch.float64)*batch_token_count;val_token_count+=batch_token_count;prev_ids=x.reshape(-1);tgt_ids=y.reshape(-1);token_bytes=val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16);token_bytes+=(val_data.has_leading_space_lut[tgt_ids]&~val_data.is_boundary_token_lut[prev_ids]).to(dtype=torch.int16);val_byte_count+=token_bytes.to(torch.float64).sum()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(val_loss_sum,op=dist.ReduceOp.SUM);dist.all_reduce(val_token_count,op=dist.ReduceOp.SUM);dist.all_reduce(val_byte_count,op=dist.ReduceOp.SUM)
	model.train();return _loss_bpb(val_loss_sum,val_token_count,val_byte_count)
def eval_val_sliding(h,device,val_data,base_model,batch_seqs=32):
	base_model.eval();logits_fn=torch.compile(base_model.forward_logits,dynamic=False,fullgraph=True);seq_len=h.eval_seq_len;context_size=seq_len-h.eval_stride;total_tokens=val_data.val_tokens.numel()-1;window_starts=[ws for ws in range(0,total_tokens,h.eval_stride)if ws+context_size<total_tokens];total_windows=len(window_starts);my_s=total_windows*h.rank//h.world_size;my_e=total_windows*(h.rank+1)//h.world_size;my_windows=window_starts[my_s:my_e];loss_sum=torch.zeros((),device=device,dtype=torch.float64);token_count=torch.zeros((),device=device,dtype=torch.float64);byte_count=torch.zeros((),device=device,dtype=torch.float64);temp=h.eval_temperature
	with torch.inference_mode():
		for bi in range(0,len(my_windows),batch_seqs):
			batch_ws=my_windows[bi:bi+batch_seqs];bsz=len(batch_ws);x_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);y_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);wlens=[]
			for(i,ws)in enumerate(batch_ws):we=min(ws+seq_len,total_tokens);wlen=we-ws;wlens.append(wlen);chunk=val_data.val_tokens[ws:we+1].to(dtype=torch.int64,device=device);x_batch[i,:wlen]=chunk[:-1];y_batch[i,:wlen]=chunk[1:]
			with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=logits_fn(x_batch)
			if temp!=1.:logits=logits/temp
			if getattr(base_model,'mos_k',1)>1:nll=F.nll_loss(logits.reshape(-1,logits.size(-1)).float(),y_batch.reshape(-1),reduction='none').reshape(bsz,seq_len)
			else:nll=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),y_batch.reshape(-1),reduction='none').reshape(bsz,seq_len)
			for(i,ws)in enumerate(batch_ws):wlen=wlens[i];s=0 if ws==0 else context_size;scored_nll=nll[i,s:wlen].to(torch.float64);loss_sum+=scored_nll.sum();token_count+=float(wlen-s);tgt=y_batch[i,s:wlen];prev=x_batch[i,s:wlen];tb=val_data.base_bytes_lut[tgt].to(torch.float64);tb+=(val_data.has_leading_space_lut[tgt]&~val_data.is_boundary_token_lut[prev]).to(torch.float64);byte_count+=tb.sum()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(loss_sum,op=dist.ReduceOp.SUM);dist.all_reduce(token_count,op=dist.ReduceOp.SUM);dist.all_reduce(byte_count,op=dist.ReduceOp.SUM)
	base_model.train();return _loss_bpb(loss_sum,token_count,byte_count)
# ===== Batched LoRA TTT (ported from 1.06335 submission) =====
class BatchedLinearLoRA(nn.Module):
	"""LoRA adapter with batched parameters [bsz, rank, dim] for parallel doc processing."""
	_ALPHA = float(os.environ.get("TTT_LORA_ALPHA", "144"))
	_WARM_START_A = bool(int(os.environ.get("TTT_WARM_START_A", "1")))
	def __init__(self, bsz, in_features, out_features, rank):
		super().__init__()
		self._bound = 1.0 / math.sqrt(in_features)
		self._scale = self._ALPHA / rank
		self.A = nn.Parameter(torch.empty(bsz, rank, in_features).uniform_(-self._bound, self._bound))
		self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))
	def reset(self):
		with torch.no_grad():
			if not self._WARM_START_A:
				self.A.uniform_(-self._bound, self._bound)
			self.B.zero_()
	def forward(self, x):
		return ((x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)) * self._scale

class BatchedTTTLoRA(nn.Module):
	"""Container for all LoRA adapters needed for TTT on our model."""
	def __init__(self, bsz, model, rank, k_lora=True, mlp_lora=True, o_lora=True):
		super().__init__()
		self.bsz = bsz
		dim = model.blocks[0].attn.c_q.in_features
		vocab = model.tok_emb.num_embeddings
		embed_dim = model.tok_emb.embedding_dim
		if getattr(model, "looping_active", False):
			num_slots = len(model.encoder_indices) + len(model.decoder_indices)
		else:
			num_slots = len(model.blocks)
		kv_dim = model.blocks[0].attn.num_kv_heads * model.blocks[0].attn.head_dim
		self.lm_head_lora = BatchedLinearLoRA(bsz, embed_dim, vocab, rank)
		self.q_loras = nn.ModuleList([BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(num_slots)])
		self.v_loras = nn.ModuleList([BatchedLinearLoRA(bsz, dim, kv_dim, rank) for _ in range(num_slots)])
		self.k_loras = nn.ModuleList([BatchedLinearLoRA(bsz, dim, kv_dim, rank) for _ in range(num_slots)]) if k_lora else None
		self.mlp_loras = nn.ModuleList([BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(num_slots)]) if mlp_lora else None
		self.o_loras = nn.ModuleList([BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(num_slots)]) if o_lora else None
	def reset(self):
		with torch.no_grad():
			self.lm_head_lora.reset()
			for loras in [self.q_loras, self.v_loras, self.k_loras, self.mlp_loras, self.o_loras]:
				if loras is not None:
					for lora in loras:
						lora.reset()

BOS_ID = None

def _find_docs(all_tokens):
	global BOS_ID
	if BOS_ID is None:
		BOS_ID = 1
	bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
	docs = []
	for i in range(len(bos_positions)):
		start = int(bos_positions[i])
		end = int(bos_positions[i + 1]) if i + 1 < len(bos_positions) else all_tokens.numel()
		if i + 1 < len(bos_positions):
			end += 1
		assert end - start >= 2
		docs.append((start, end - start))
	return docs

def _select_ttt_doc_entries(docs, h):
	doc_entries = list(enumerate(docs))
	if h.val_doc_fraction < 1.0:
		sample_n = max(1, int(round(len(docs) * h.val_doc_fraction)))
		sampled_indices = sorted(random.Random(h.seed).sample(range(len(docs)), sample_n))
		return [(i, docs[i]) for i in sampled_indices]
	return doc_entries

def _build_ttt_global_batches(doc_entries, h, ascending=False):
	batch_size = h.ttt_batch_size
	global_doc_entries = sorted(doc_entries, key=lambda x: x[1][1])
	global_batches = [global_doc_entries[i : i + batch_size] for i in range(0, len(global_doc_entries), batch_size)]
	indexed = list(enumerate(global_batches))
	if not ascending:
		indexed.sort(key=lambda ib: -max(dl for _, (_, dl) in ib[1]))
	return indexed

def _init_batch_counter(path):
	with open(path, "wb") as f:
		f.write((0).to_bytes(4, "little"))

def _claim_next_batch(counter_path, queue_len):
	try:
		with open(counter_path, "r+b") as f:
			fcntl.flock(f, fcntl.LOCK_EX)
			idx = int.from_bytes(f.read(4), "little")
			f.seek(0)
			f.write((idx + 1).to_bytes(4, "little"))
			f.flush()
	except FileNotFoundError:
		return queue_len
	return idx

def _compute_chunk_window(ci, pred_len, num_chunks, chunk_size, eval_seq_len):
	chunk_end = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
	win_start = max(0, chunk_end - eval_seq_len)
	win_len = chunk_end - win_start
	chunk_start = ci * chunk_size
	chunk_offset = chunk_start - win_start
	chunk_len = chunk_end - chunk_start
	return win_start, win_len, chunk_offset, chunk_len

def _accumulate_bpb(ptl, x, y, chunk_offsets, chunk_lens, pos_idx,
                    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                    loss_sum, byte_sum, token_count):
	pos = pos_idx[: x.size(1)].unsqueeze(0)
	mask = ((chunk_lens.unsqueeze(1) > 0)
	        & (pos >= chunk_offsets.unsqueeze(1))
	        & (pos < (chunk_offsets + chunk_lens).unsqueeze(1)))
	mask_f64 = mask.to(torch.float64)
	tok_bytes = base_bytes_lut[y].to(torch.float64)
	tok_bytes += (has_leading_space_lut[y] & ~is_boundary_token_lut[x]).to(torch.float64)
	loss_sum += (ptl.to(torch.float64) * mask_f64).sum()
	byte_sum += (tok_bytes * mask_f64).sum()
	token_count += chunk_lens.to(torch.float64).sum()

def _loss_bpb_from_sums(loss_sum, token_count, byte_sum):
	val_loss = (loss_sum / token_count).item()
	val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_sum.item())
	return val_loss, val_bpb

def _init_int64_counter(path):
	with open(path, "wb") as f:
		f.write((0).to_bytes(8, "little", signed=True))

def _add_to_counter(path, delta):
	try:
		with open(path, "r+b") as f:
			fcntl.flock(f, fcntl.LOCK_EX)
			cur = int.from_bytes(f.read(8), "little", signed=True)
			cur += int(delta)
			f.seek(0)
			f.write(int(cur).to_bytes(8, "little", signed=True))
			f.flush()
			return cur
	except FileNotFoundError:
		return int(delta)

def train_val_ttt_global_sgd_distributed(h, device, val_data, base_model, val_tokens, batch_seqs=None):
	"""Global SGD TTT: train base model weights on scored prefix docs."""
	global BOS_ID
	if BOS_ID is None:
		BOS_ID = 1
	base_model.eval()
	seq_len = h.eval_seq_len
	total_tokens = val_tokens.numel() - 1
	ttt_chunk = h.global_ttt_chunk_tokens
	batch_seqs = h.global_ttt_batch_seqs if batch_seqs is None else batch_seqs
	num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
	ttt_params = [p for p in base_model.parameters()]
	for p in ttt_params:
		p.requires_grad_(True)
	optimizer = torch.optim.SGD(ttt_params, lr=h.global_ttt_lr, momentum=h.global_ttt_momentum)
	t_start = time.perf_counter()
	for ci in range(num_chunks):
		chunk_start = ci * ttt_chunk
		chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
		is_last_chunk = ci == num_chunks - 1
		if is_last_chunk or h.global_ttt_epochs <= 0:
			continue
		base_model.train()
		chunk_seqs = (chunk_end - chunk_start) // seq_len
		if chunk_seqs <= 0:
			continue
		warmup_chunks = max(0, min(h.global_ttt_warmup_chunks, num_chunks - 1))
		if warmup_chunks > 0 and ci < warmup_chunks:
			warmup_denom = max(warmup_chunks - 1, 1)
			warmup_t = ci / warmup_denom
			lr_now = h.global_ttt_warmup_start_lr + (h.global_ttt_lr - h.global_ttt_warmup_start_lr) * warmup_t
		else:
			decay_steps = max(num_chunks - 1 - warmup_chunks, 1)
			decay_ci = max(ci - warmup_chunks, 0)
			lr_now = h.global_ttt_lr * 0.5 * (1.0 + math.cos(math.pi * decay_ci / decay_steps))
		for pg in optimizer.param_groups:
			pg["lr"] = lr_now
		my_seq_s = chunk_seqs * h.rank // h.world_size
		my_seq_e = chunk_seqs * (h.rank + 1) // h.world_size
		my_chunk_seqs = my_seq_e - my_seq_s
		for _ in range(h.global_ttt_epochs):
			for bs in range(0, my_chunk_seqs, batch_seqs):
				be = min(bs + batch_seqs, my_chunk_seqs)
				actual_bs = my_seq_s + bs
				start_tok = chunk_start + actual_bs * seq_len
				end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
				if end_tok > val_tokens.numel():
					continue
				local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
				x_flat = local[:-1]
				y_flat = local[1:]
				optimizer.zero_grad(set_to_none=True)
				with torch.enable_grad():
					with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
						x = x_flat.reshape(-1, seq_len)
						y = y_flat.reshape(-1, seq_len)
						loss = base_model(x, y)
				loss.backward()
				if dist.is_available() and dist.is_initialized():
					for p in ttt_params:
						if p.grad is not None:
							dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
							p.grad.mul_(1.0 / h.world_size)
				if h.global_ttt_grad_clip > 0:
					torch.nn.utils.clip_grad_norm_(ttt_params, h.global_ttt_grad_clip)
				optimizer.step()
		base_model.eval()
		if h.rank == 0:
			elapsed = time.perf_counter() - t_start
			log(f"tttg: c{ci+1}/{num_chunks} lr:{lr_now:.6f} t:{elapsed:.1f}s")
	for p in base_model.parameters():
		p.requires_grad_(True)
	base_model.eval()

def eval_val_ttt_phased(h, base_model, device, val_data, forward_ttt_train, deadline=None):
	"""Phased LoRA TTT: document-level batched scoring with per-doc LoRA adaptation."""
	global BOS_ID
	if BOS_ID is None:
		BOS_ID = 1
	base_model.eval()
	for p in base_model.parameters():
		p.requires_grad_(False)
	all_tokens = val_data.val_tokens
	all_tokens_idx = all_tokens.to(torch.int32)
	docs = _find_docs(all_tokens)
	doc_entries = _select_ttt_doc_entries(docs, h)
	prefix_doc_limit = max(0, min(len(doc_entries), int(h.phased_ttt_prefix_docs)))
	num_phases = max(1, int(h.phased_ttt_num_phases))
	phase_boundaries = []
	for pi in range(num_phases):
		boundary = prefix_doc_limit * (pi + 1) // num_phases
		phase_boundaries.append(boundary)
	current_phase = 0
	current_phase_boundary = phase_boundaries[0]
	log(f"ttt_phased: total_docs:{len(doc_entries)} prefix_docs:{prefix_doc_limit} "
	    f"suffix_docs:{len(doc_entries) - prefix_doc_limit} "
	    f"num_phases:{num_phases} boundaries:{phase_boundaries}")
	chunk_size, eval_seq_len = h.ttt_chunk_size, h.ttt_eval_seq_len
	eval_batch_set = None
	if h.ttt_eval_batches:
		eval_batch_set = set(int(x) for x in h.ttt_eval_batches.split(",") if x.strip())
	use_ascending = eval_batch_set is not None
	global_batches_sorted = _build_ttt_global_batches(doc_entries, h, ascending=use_ascending)
	queue_len = len(global_batches_sorted)
	counter_path = f"/tmp/ttt_counter_{h.run_id}"
	prefix_counter_path = f"/tmp/ttt_prefix_counter_{h.run_id}"
	pause_flag_path = f"/tmp/ttt_pause_flag_{h.run_id}"
	if h.rank == 0:
		_init_batch_counter(counter_path)
		_init_int64_counter(prefix_counter_path)
		try:
			os.remove(pause_flag_path)
		except FileNotFoundError:
			pass
	if dist.is_available() and dist.is_initialized():
		path_list = [counter_path, prefix_counter_path, pause_flag_path]
		dist.broadcast_object_list(path_list, src=0)
		counter_path, prefix_counter_path, pause_flag_path = path_list
		dist.barrier()
	loss_sum = torch.zeros((), device=device, dtype=torch.float64)
	byte_sum = torch.zeros((), device=device, dtype=torch.float64)
	token_count = torch.zeros((), device=device, dtype=torch.float64)
	t_start = time.perf_counter()
	reusable_lora = BatchedTTTLoRA(
		h.ttt_batch_size, base_model, h.ttt_lora_rank,
		k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
	).to(device)
	def _build_opt(lora):
		if h.ttt_optimizer == "sgd":
			return torch.optim.SGD(lora.parameters(), lr=h.ttt_lora_lr,
			                       momentum=h.ttt_beta1, weight_decay=h.ttt_weight_decay)
		return torch.optim.AdamW(lora.parameters(), lr=h.ttt_lora_lr,
		                         betas=(h.ttt_beta1, h.ttt_beta2),
		                         eps=1e-10, weight_decay=h.ttt_weight_decay, fused=True)
	reusable_opt = _build_opt(reusable_lora)
	local_scored_docs = []
	global_ttt_done = prefix_doc_limit == 0
	try:
		while True:
			if deadline is not None and time.perf_counter()>=deadline:
				log(f"ttt:eval_time_limit reached, scored {int(token_count.item())} tokens");break
			queue_idx = _claim_next_batch(counter_path, queue_len)
			if queue_idx >= queue_len:
				break
			orig_batch_idx, batch_entries = global_batches_sorted[queue_idx]
			batch = [doc for _, doc in batch_entries]
			bsz = len(batch)
			prev_loss = loss_sum.item()
			prev_bytes = byte_sum.item()
			prev_tokens = token_count.item()
			if bsz == reusable_lora.bsz:
				reusable_lora.reset()
				for s in reusable_opt.state.values():
					for k, v in s.items():
						if isinstance(v, torch.Tensor):
							v.zero_()
						elif k == "step":
							s[k] = 0
				cur_lora = reusable_lora
				cur_opt = reusable_opt
			else:
				cur_lora = BatchedTTTLoRA(
					bsz, base_model, h.ttt_lora_rank,
					k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
				).to(device)
				cur_opt = _build_opt(cur_lora)
			pred_lens = [doc_len - 1 for _, doc_len in batch]
			num_chunks = [(pl + chunk_size - 1) // chunk_size for pl in pred_lens]
			max_nc = max(num_chunks)
			num_chunks_t = torch.tensor(num_chunks, dtype=torch.int64, device=device)
			for ci in range(max_nc):
				active = [ci < nc for nc in num_chunks]
				needs_train = any(ci < nc - 1 for nc in num_chunks)
				tok_starts = torch.zeros(bsz, dtype=torch.int64)
				tok_wls = torch.zeros(bsz, dtype=torch.int64)
				chunk_offsets_cpu = torch.zeros(bsz, dtype=torch.int64)
				chunk_lens_cpu = torch.zeros(bsz, dtype=torch.int64)
				for b in range(bsz):
					if not active[b]:
						continue
					doc_start, doc_len = batch[b]
					win_start, win_len, chunk_offset, chunk_len = _compute_chunk_window(
						ci, pred_lens[b], num_chunks[b], chunk_size, eval_seq_len)
					tok_starts[b] = doc_start + win_start
					tok_wls[b] = win_len
					chunk_offsets_cpu[b] = chunk_offset
					chunk_lens_cpu[b] = chunk_len
				_, context_size, chunk_offset, _ = _compute_chunk_window(
					ci, (ci + 1) * chunk_size, ci + 1, chunk_size, eval_seq_len)
				col_idx = torch.arange(context_size + 1)
				idx = tok_starts.unsqueeze(1) + col_idx.unsqueeze(0)
				idx.clamp_(max=all_tokens.numel() - 1)
				gathered_gpu = all_tokens_idx[idx].to(device=device, dtype=torch.int64, non_blocking=True)
				valid = (col_idx[:context_size].unsqueeze(0) < tok_wls.unsqueeze(1)).to(device, non_blocking=True)
				chunk_offsets = chunk_offsets_cpu.to(device, non_blocking=True)
				chunk_lens = chunk_lens_cpu.to(device, non_blocking=True)
				x = torch.where(valid, gathered_gpu[:, :context_size], 0)
				y = torch.where(valid, gathered_gpu[:, 1 : context_size + 1], 0)
				ctx_pos = torch.arange(context_size, device=device, dtype=torch.int64)
				if needs_train:
					with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
						per_tok_loss = forward_ttt_train(x, y, lora=cur_lora)
				else:
					with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
						per_tok_loss = forward_ttt_train(x, y, lora=cur_lora)
				with torch.no_grad():
					_accumulate_bpb(per_tok_loss, x, y, chunk_offsets, chunk_lens, ctx_pos,
					                val_data.base_bytes_lut, val_data.has_leading_space_lut,
					                val_data.is_boundary_token_lut,
					                loss_sum, byte_sum, token_count)
				if needs_train:
					activate_chunk_mask = (num_chunks_t - 1 > ci).float()
					for gi in range(h.ttt_grad_steps):
						if gi > 0:
							with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
								per_tok_loss = forward_ttt_train(x, y, lora=cur_lora)
						per_doc = per_tok_loss[:, chunk_offset : chunk_offset + chunk_size].mean(dim=-1)
						cur_opt.zero_grad(set_to_none=True)
						(per_doc * activate_chunk_mask).sum().backward()
						cur_opt.step()
				else:
					del per_tok_loss
			batch_num = orig_batch_idx + 1
			doc_lens = [dl for _, dl in batch]
			should_report = batch_num in eval_batch_set if eval_batch_set is not None else True
			if should_report:
				cur_tokens = token_count.item()
				cur_loss_val = loss_sum.item()
				cur_bytes_val = byte_sum.item()
				dt = cur_tokens - prev_tokens
				db = cur_bytes_val - prev_bytes
				if dt > 0 and db > 0:
					b_loss = (cur_loss_val - prev_loss) / dt
					b_bpb = b_loss / math.log(2.0) * (dt / db)
				else:
					b_loss = b_bpb = 0.0
				r_loss = cur_loss_val / max(cur_tokens, 1)
				r_bpb = r_loss / math.log(2.0) * (cur_tokens / max(cur_bytes_val, 1))
				elapsed = time.perf_counter() - t_start
				log(f"ttp: b{batch_num}/{queue_len} bl:{b_loss:.4f} bb:{b_bpb:.4f} "
				    f"rl:{r_loss:.4f} rb:{r_bpb:.4f} dl:{min(doc_lens)}-{max(doc_lens)} "
				    f"gd:{int(global_ttt_done)}")
			if not global_ttt_done:
				local_scored_docs.extend(
					(orig_batch_idx, pos, doc_start, doc_len)
					for pos, (doc_start, doc_len) in enumerate(batch))
				prefix_done = _add_to_counter(prefix_counter_path, len(batch_entries))
				if prefix_done >= current_phase_boundary:
					try:
						with open(pause_flag_path, "x"):
							pass
					except FileExistsError:
						pass
				should_pause = os.path.exists(pause_flag_path)
				if should_pause:
					if dist.is_available() and dist.is_initialized():
						dist.barrier()
					gathered_scored_docs = [None] * h.world_size
					if dist.is_available() and dist.is_initialized():
						dist.all_gather_object(gathered_scored_docs, local_scored_docs)
					else:
						gathered_scored_docs = [local_scored_docs]
					scored_docs_for_global = []
					for rank_docs in gathered_scored_docs:
						if rank_docs:
							scored_docs_for_global.extend(rank_docs)
					scored_docs_for_global.sort(key=lambda x: (x[0], x[1]))
					scored_docs_for_global = scored_docs_for_global[:current_phase_boundary]
					scored_token_chunks = [
						val_data.val_tokens[doc_start : doc_start + doc_len]
						for _, _, doc_start, doc_len in scored_docs_for_global]
					if scored_token_chunks:
						global_ttt_tokens = torch.cat(scored_token_chunks)
					else:
						global_ttt_tokens = val_data.val_tokens[:0]
					if h.rank == 0:
						log(f"ttpp: phase:{current_phase + 1}/{num_phases} "
						    f"gd:{len(scored_docs_for_global)} t:{time.perf_counter() - t_start:.1f}s")
					train_val_ttt_global_sgd_distributed(h, device, val_data, base_model, global_ttt_tokens)
					for p in base_model.parameters():
						p.requires_grad_(False)
					reusable_lora = BatchedTTTLoRA(
						h.ttt_batch_size, base_model, h.ttt_lora_rank,
						k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
					).to(device)
					reusable_opt = _build_opt(reusable_lora)
					current_phase += 1
					if current_phase >= num_phases:
						global_ttt_done = True
					else:
						current_phase_boundary = phase_boundaries[current_phase]
						if h.rank == 0:
							try:
								os.remove(pause_flag_path)
							except FileNotFoundError:
								pass
					if dist.is_available() and dist.is_initialized():
						dist.barrier()
					if h.rank == 0:
						log(f"ttpr: phase:{current_phase}/{num_phases} t:{time.perf_counter() - t_start:.1f}s")
			del cur_lora, cur_opt
	finally:
		pass
	if dist.is_available() and dist.is_initialized():
		dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
		dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
		dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
	for p in base_model.parameters():
		p.requires_grad_(True)
	base_model.train()
	return _loss_bpb_from_sums(loss_sum, token_count, byte_sum)

def timed_eval(label,fn,*args,**kwargs):torch.cuda.synchronize();t0=time.perf_counter();val_loss,val_bpb=fn(*args,**kwargs);torch.cuda.synchronize();elapsed_ms=1e3*(time.perf_counter()-t0);log(f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms");return val_loss,val_bpb
def train_model(h,device,val_data):
	compile_mode=os.environ.get('COMPILE_MODE','default');base_model=GPT(h).to(device).bfloat16();restore_fp32_params(base_model)
	# Build byte-count LUT for byte-weighted CE loss
	_byte_lut=None
	if h.byte_weighted_ce:
		sp=spm.SentencePieceProcessor(model_file=h.tokenizer_path);base_bytes_lut,_,_=build_sentencepiece_luts(sp,h.vocab_size,device);_byte_lut=base_bytes_lut.clamp(min=1,max=4).float();log(f"byte_weighted_ce:enabled, lut shape={_byte_lut.shape}, mean_weight={_byte_lut.mean().item():.3f}")
	compiled_model=torch.compile(base_model,mode=compile_mode if compile_mode!='default'else None,dynamic=False,fullgraph=True)
	if h.distributed:model=DDP(compiled_model,device_ids=[h.local_rank],broadcast_buffers=False)
	else:model=compiled_model
	log(f"model_params:{sum(p.numel()for p in base_model.parameters())}");optimizers=Optimizers(h,base_model);train_loader=ShuffledSequenceLoader(h,device);max_wallclock_ms=1e3*h.max_wallclock_seconds if h.max_wallclock_seconds>0 else None
	if max_wallclock_ms is not None:max_wallclock_ms-=h.gptq_reserve_seconds*1e3;log(f"gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms")
	def training_frac(step,elapsed_ms):
		if max_wallclock_ms is None:return step/max(h.iterations,1)
		return elapsed_ms/max(max_wallclock_ms,1e-09)
	lr_schedule=os.environ.get('LR_SCHEDULE','linear')
	def lr_mul(frac):
		if h.warmdown_frac<=0:return 1.
		if frac>=1.-h.warmdown_frac:
			raw=(1.-frac)/h.warmdown_frac
			if lr_schedule=='sqrt':return max(math.sqrt(raw),h.min_lr)
			return max(raw,h.min_lr)
		return 1.
	_cur_batch_tokens=[h.train_batch_tokens];_kd_active=[h.kd_enabled];_approx_training_ms=[0.]
	if h.kd_enabled:log(f"kd:enabled alpha={h.kd_alpha} temp={h.kd_temperature} top_k={h.kd_top_k} dir={h.kd_logits_dir} warmup_frac={h.kd_warmup_frac}")
	def step_fn(step,lr_scale):
		optimizers.zero_grad_all();train_loss=torch.zeros((),device=device)
		batch_tokens=_cur_batch_tokens[0]
		for micro_step in range(h.grad_accum_steps):
			if h.distributed:model.require_backward_grad_sync=micro_step==h.grad_accum_steps-1
			batch_data=train_loader.next_batch(batch_tokens,h.grad_accum_steps)
			x,y=batch_data[0],batch_data[1]
			bw=_byte_lut[y] if _byte_lut is not None else None
			if _kd_active[0] and len(batch_data)==4:
				t_idx,t_val=batch_data[2],batch_data[3]
				kd_alpha_eff=h.kd_alpha
				if h.kd_warmup_frac>0:
					approx_ms=_approx_training_ms[0]
					frac=training_frac(step,approx_ms)
					if frac<h.kd_warmup_frac:kd_alpha_eff=h.kd_alpha*frac/h.kd_warmup_frac
				with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):loss=model(x,y,byte_weights=bw,teacher_topk_idx=t_idx,teacher_topk_val=t_val,kd_alpha=kd_alpha_eff,kd_temperature=h.kd_temperature)
			else:
				with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):loss=model(x,y,byte_weights=bw)
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

		if len(base_model.encoder_indices)!=base_model.num_encoder_layers:
			base_model.looping_active=True;log(f"loop_warmup:on")
			for warmup_step in range(h.warmup_steps):
				step_fn(warmup_step,1.)
				if warmup_step<=5 or(warmup_step+1)%10==0 or warmup_step+1==h.warmup_steps:log(f"loop_warmup_step: {warmup_step+1}/{h.warmup_steps}")
			base_model.looping_active=False
		base_model.load_state_dict(initial_model_state,strict=True)
		for(opt,state)in zip(optimizers,initial_optimizer_states,strict=True):opt.load_state_dict(state)
		optimizers.zero_grad_all()
		if h.distributed:model.require_backward_grad_sync=True
		train_loader=ShuffledSequenceLoader(h,device)
	ema_state={name:t.detach().float().clone()for(name,t)in base_model.state_dict().items()};ema_decay=h.ema_decay;swa_state=None;swa_count=0;training_time_ms=.0;stop_after_step=None;torch.cuda.synchronize();t0=time.perf_counter();step=0
	while True:
		last_step=step==h.iterations or stop_after_step is not None and step>=stop_after_step;should_validate=last_step or h.val_loss_every>0 and step%h.val_loss_every==0
		if should_validate:torch.cuda.synchronize();training_time_ms+=1e3*(time.perf_counter()-t0);val_loss,val_bpb=eval_val(h,device,val_data,model);log(f"{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}");torch.cuda.synchronize();t0=time.perf_counter()
		if last_step:
			if stop_after_step is not None and step<h.iterations:log(f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms step: {step}/{h.iterations}")
			break
		elapsed_ms=training_time_ms+1e3*(time.perf_counter()-t0);_approx_training_ms[0]=elapsed_ms;frac=training_frac(step,elapsed_ms);scale=lr_mul(frac)
		if h.batch_schedule_enabled:
			if frac<(1.0-h.warmdown_frac):_cur_batch_tokens[0]=h.train_batch_tokens*2
			else:_cur_batch_tokens[0]=h.train_batch_tokens
		if len(base_model.encoder_indices)!=base_model.num_encoder_layers and not base_model.looping_active and frac>=h.enable_looping_at:base_model.looping_active=True;log(f"layer_loop:enabled step:{step} frac:{frac:.3f} encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")
		if h.sparsity_start_frac>0 and frac>=h.sparsity_start_frac and not getattr(base_model,'_sparsity_applied',False):
			with torch.no_grad():
				for module in base_model.modules():
					if isinstance(module,MLP):
						for linear in[module.fc,module.proj]:
							w=linear.weight.data
							for i in range(0,w.shape[1],4):
								group=w[:,i:i+4].abs();_,idx=group.topk(2,dim=1,largest=False)
								for j in range(2):w[range(w.shape[0]),i+idx[:,j]]=0
			base_model._sparsity_applied=True;log(f"sparsity:applied 2:4 pruning at frac={frac:.3f}")
		momentum_cooldown=float(os.environ.get('MOMENTUM_COOLDOWN',0.))
		if momentum_cooldown>0 and scale<1.:
			cool_mom=h.muon_momentum-momentum_cooldown*(1.-scale);
			for group in optimizers.optimizer_muon.param_groups:group['momentum']=cool_mom
		train_loss=step_fn(step,scale)
		with torch.no_grad():
			for(name,t)in base_model.state_dict().items():ema_state[name].mul_(ema_decay).add_(t.detach().float(),alpha=1.-ema_decay)
			swa_decay=float(os.environ.get('SWA_DECAY',0))
			if h.swa_enabled and scale<h.swa_start_frac and step%h.swa_every==0:
				if swa_state is None:swa_state={name:t.detach().float().clone()for(name,t)in base_model.state_dict().items()};swa_count=1
				elif swa_decay>0:
					for(name,t)in base_model.state_dict().items():swa_state[name].mul_(swa_decay).add_(t.detach().float(),alpha=1.-swa_decay)
					swa_count=1
				else:
					for(name,t)in base_model.state_dict().items():swa_state[name].add_(t.detach().float())
					swa_count+=1
		step+=1;approx_training_time_ms=training_time_ms+1e3*(time.perf_counter()-t0);should_log_train=h.train_log_every>0 and(step<=5 or step%h.train_log_every==0 or stop_after_step is not None)
		if should_log_train:tok_per_sec=step*h.train_batch_tokens/(approx_training_time_ms/1e3);log(f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} train_time: {approx_training_time_ms/60000:.1f}m tok/s: {tok_per_sec:.0f}")
		reached_cap=max_wallclock_ms is not None and approx_training_time_ms>=max_wallclock_ms
		if h.distributed and max_wallclock_ms is not None:reached_cap_tensor=torch.tensor(int(reached_cap),device=device);dist.all_reduce(reached_cap_tensor,op=dist.ReduceOp.MAX);reached_cap=bool(reached_cap_tensor.item())
		if stop_after_step is None and reached_cap:stop_after_step=step
	log(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB");current_state=base_model.state_dict()
	if h.swa_enabled and swa_state is not None and swa_count>0:
		log(f"swa:applying SWA weights ({swa_count} checkpoints)");avg_state={name:(t/swa_count).to(dtype=current_state[name].dtype)for(name,t)in swa_state.items()}
	else:
		log('ema:applying EMA weights');avg_state={name:t.to(dtype=current_state[name].dtype)for(name,t)in ema_state.items()}
	base_model.load_state_dict(avg_state,strict=True);return base_model,compiled_model
def train_and_eval(h,device):
	eval_only=bool(int(os.environ.get('EVAL_ONLY','0')))
	random.seed(h.seed);np.random.seed(h.seed);torch.manual_seed(h.seed);torch.cuda.manual_seed_all(h.seed);val_data=ValidationData(h,device);log(f"val_tokens: {val_data.val_tokens.numel()-1}")
	if eval_only:
		log("EVAL_ONLY=1: skipping training and serialization, loading existing quantized model")
	else:
		log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}");base_model,compiled_model=train_model(h,device,val_data);torch._dynamo.reset();timed_eval('pre-quantization post-ema',eval_val,h,device,val_data,compiled_model);serialize(h,base_model,Path(__file__).read_text(encoding='utf-8'))
	if h.distributed:dist.barrier()
	torch.cuda.synchronize();eval_t0=time.perf_counter();eval_deadline=eval_t0+h.max_eval_seconds if h.max_eval_seconds>0 else None;log(f"eval:budget {h.max_eval_seconds:.0f}s")
	eval_model=deserialize(h,device)
	if len(eval_model.encoder_indices)!=eval_model.num_encoder_layers:eval_model.looping_active=True
	compiled_model=torch.compile(eval_model,dynamic=False,fullgraph=True);timed_eval('quantized',eval_val,h,device,val_data,compiled_model)
	if h.sliding_window_enabled:timed_eval('quantized_sliding_window',eval_val_sliding,h,device,val_data,eval_model)
	if h.ttt_enabled and h.ttt_lora_rank > 0:
		ttt_model=deserialize(h,device)
		if len(ttt_model.encoder_indices)!=ttt_model.num_encoder_layers:ttt_model.looping_active=True
		# Warm up rotary caches for TTT eval seq len
		for block in ttt_model.blocks:
			block.attn.rotary._cos_cached = None
			block.attn.rotary._sin_cached = None
			block.attn.rotary._seq_len_cached = 0
			block.attn.rotary(h.ttt_eval_seq_len, device, torch.bfloat16)
		def _fwd_ttt_inner(input_ids, target_ids, lora):
			return ttt_model.forward_ttt(input_ids, target_ids, lora=lora)
		_fwd_ttt_compiled_inner = None
		def _fwd_ttt(input_ids, target_ids, lora):
			nonlocal _fwd_ttt_compiled_inner
			if _fwd_ttt_compiled_inner is None:
				_fwd_ttt_compiled_inner = torch.compile(_fwd_ttt_inner, dynamic=True)
			return _fwd_ttt_compiled_inner(input_ids, target_ids, lora=lora)
		fwd_ttt_compiled = _fwd_ttt
		# Compile warmup with random tokens
		log(f"ttt_lora:warming up compile")
		t_warmup = time.perf_counter()
		for bsz_w in [h.ttt_batch_size]:
			wl = BatchedTTTLoRA(bsz_w, ttt_model, h.ttt_lora_rank,
			                    k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora).to(device)
			wo = torch.optim.AdamW(wl.parameters(), lr=h.ttt_lora_lr,
			                       betas=(h.ttt_beta1, h.ttt_beta2), eps=1e-10,
			                       weight_decay=h.ttt_weight_decay, fused=True)
			for ctx_len in (h.ttt_chunk_size, h.ttt_eval_seq_len):
				xw = torch.randint(0, h.vocab_size, (bsz_w, ctx_len), device=device, dtype=torch.int64)
				yw = torch.randint(0, h.vocab_size, (bsz_w, ctx_len), device=device, dtype=torch.int64)
				with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
					ptl = fwd_ttt_compiled(xw, yw, lora=wl)
				ptl[:, : min(h.ttt_chunk_size, ctx_len)].mean(dim=-1).sum().backward()
				wo.step()
				wo.zero_grad(set_to_none=True)
			del wl, wo
		torch.cuda.empty_cache()
		log(f"ttt_lora:compile warmup done ({time.perf_counter() - t_warmup:.1f}s)")
		log(f"ttt_lora_alpha: {BatchedLinearLoRA._ALPHA}")
		log(f"ttt_warm_start_a: {BatchedLinearLoRA._WARM_START_A}")
		log("beginning TTT eval timer")
		torch.cuda.synchronize()
		t_ttt = time.perf_counter()
		ttt_val_loss, ttt_val_bpb = eval_val_ttt_phased(h, ttt_model, device, val_data, forward_ttt_train=fwd_ttt_compiled, deadline=eval_deadline)
		torch.cuda.synchronize()
		ttt_eval_elapsed = time.perf_counter() - t_ttt
		log(f"quantized_ttt_phased val_loss:{ttt_val_loss:.8f} val_bpb:{ttt_val_bpb:.8f} eval_time:{ttt_eval_elapsed*1e3:.0f}ms")
	torch.cuda.synchronize();total_eval_elapsed=time.perf_counter()-eval_t0;log(f"TOTAL_EVAL_TIME: {total_eval_elapsed:.1f}s ({total_eval_elapsed/60:.1f}m) budget:{h.max_eval_seconds:.0f}s")
def main():
	world_size=int(os.environ.get('WORLD_SIZE','1'));local_rank=int(os.environ.get('LOCAL_RANK','0'));distributed='RANK'in os.environ and'WORLD_SIZE'in os.environ
	if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
	if world_size<=0:raise ValueError(f"bad ws")
	if 8%world_size!=0:raise ValueError(f"ws must divide 8")
	device=torch.device('cuda',local_rank);torch.cuda.set_device(device)
	if distributed:dist.init_process_group(backend='nccl',device_id=device);dist.barrier()
	torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.set_float32_matmul_precision('high');from torch.backends.cuda import enable_cudnn_sdp,enable_flash_sdp,enable_math_sdp,enable_mem_efficient_sdp;enable_cudnn_sdp(False);enable_flash_sdp(True);enable_mem_efficient_sdp(True);enable_math_sdp(False);torch._dynamo.config.optimize_ddp=False
	h=Hyperparameters();set_logging_hparams(h)
	if h.is_main_process:
		os.makedirs('logs',exist_ok=True);log('Hyperparameters:',console=True)
		for(k,v)in sorted(vars(type(h)).items()):
			if not k.startswith('_'):log(f"  {k}: {v}",console=True)
		log('='*100,console=False)
	train_and_eval(h,device)
	if distributed:dist.destroy_process_group()
if __name__=='__main__':main()