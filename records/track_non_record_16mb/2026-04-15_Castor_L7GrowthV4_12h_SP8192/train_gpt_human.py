import collections, copy, glob, io, json, lzma, math, os, shutil
from contextlib import nullcontext
from pathlib import Path
import random, re, subprocess, sys, time, uuid, numpy as np, sentencepiece as spm, torch, torch.distributed as dist, torch.nn.functional as F
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from env_utils import load_env_file

SCRIPT_DIR = Path(__file__).resolve().parent
load_env_file(SCRIPT_DIR, os.environ.get('SIMON_ENV_FILE', '.env'))
_INT6_LEGACY_SCALE_CLAMP_MIN = 1e-10
INT6_QAT_SCALE_CLAMP_MIN = float(os.environ.get('INT6_QAT_SCALE_CLAMP_MIN', str(_INT6_LEGACY_SCALE_CLAMP_MIN)))
INT6_EXPORT_SCALE_CLAMP_MIN = float(os.environ.get('INT6_EXPORT_SCALE_CLAMP_MIN', str(_INT6_LEGACY_SCALE_CLAMP_MIN)))

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
except ImportError:

    def flash_attn_3_func(q: Tensor, k: Tensor, v: Tensor, causal: bool=True, enable_gqa: bool=False) -> Tensor:
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q_t, k_t, v_t, dropout_p=0.0, is_causal=causal, enable_gqa=enable_gqa)
        return y.transpose(1, 2)

class Hyperparameters:
    data_dir = os.environ.get('DATA_DIR', './data/')
    seed = int(os.environ.get('SEED', 1337))
    run_id = os.environ.get('RUN_ID', str(uuid.uuid4()))
    iterations = int(os.environ.get('ITERATIONS', 20000))
    warmdown_frac = float(os.environ.get('WARMDOWN_FRAC', 0.72))
    warmdown_iters = int(os.environ.get('WARMDOWN_ITERS', 0))
    warmup_steps = int(os.environ.get('WARMUP_STEPS', 20))
    train_batch_tokens = int(os.environ.get('TRAIN_BATCH_TOKENS', 786432))
    train_seq_len = int(os.environ.get('TRAIN_SEQ_LEN', 2048))
    train_seq_schedule = os.environ.get('TRAIN_SEQ_SCHEDULE', '')
    train_seq_schedule_mode = os.environ.get('TRAIN_SEQ_SCHEDULE_MODE', 'wallclock').strip().lower()
    seq_change_warmup_steps = int(os.environ.get('SEQ_CHANGE_WARMUP_STEPS', 0))
    compile_model = bool(int(os.environ.get('COMPILE_MODEL', '1')))
    midrun_cap_schedule = os.environ.get('MIDRUN_CAP_SCHEDULE', '').strip()
    midrun_cap_log_updates = bool(int(os.environ.get('MIDRUN_CAP_LOG_UPDATES', '0')))
    compile_dynamic = bool(int(os.environ.get('COMPILE_DYNAMIC', '1' if train_seq_schedule.strip() else '0')))
    compile_fullgraph = bool(int(os.environ.get('COMPILE_FULLGRAPH', '1')))
    dynamo_cache_size_limit = int(os.environ.get('DYNAMO_CACHE_SIZE_LIMIT', '64' if train_seq_schedule.strip() else '8'))
    fsdp_enabled = bool(int(os.environ.get('FSDP_ENABLED', '0')))
    fsdp_auto_wrap_blocks = bool(int(os.environ.get('FSDP_AUTO_WRAP_BLOCKS', '0')))
    fsdp_sharding_strategy = os.environ.get('FSDP_SHARDING_STRATEGY', 'FULL_SHARD').strip().upper()
    fsdp_backward_prefetch = os.environ.get('FSDP_BACKWARD_PREFETCH', 'BACKWARD_PRE').strip().upper()
    fsdp_forward_prefetch = bool(int(os.environ.get('FSDP_FORWARD_PREFETCH', '0')))
    fsdp_limit_all_gathers = bool(int(os.environ.get('FSDP_LIMIT_ALL_GATHERS', '1')))
    fsdp_use_orig_params = bool(int(os.environ.get('FSDP_USE_ORIG_PARAMS', '1')))
    fsdp_cpu_offload = bool(int(os.environ.get('FSDP_CPU_OFFLOAD', '0')))
    fsdp_sync_module_states = bool(int(os.environ.get('FSDP_SYNC_MODULE_STATES', '0')))
    fsdp_use_adamw_for_matrix = bool(int(os.environ.get('FSDP_USE_ADAMW_FOR_MATRIX', '1')))
    train_log_every = int(os.environ.get('TRAIN_LOG_EVERY', 500))
    max_wallclock_seconds = float(os.environ.get('MAX_WALLCLOCK_SECONDS', 600.0))
    val_batch_tokens = int(os.environ.get('VAL_BATCH_TOKENS', 524288))
    val_max_tokens = int(os.environ.get('VAL_MAX_TOKENS', '0'))
    eval_seq_len = int(os.environ.get('EVAL_SEQ_LEN', 2048))
    val_loss_every = int(os.environ.get('VAL_LOSS_EVERY', 4000))
    sliding_window_enabled = bool(int(os.environ.get('SLIDING_WINDOW_ENABLED', '1')))
    roundtrip_eval_enabled = bool(int(os.environ.get('ROUNDTRIP_EVAL_ENABLED', '1')))
    eval_only = bool(int(os.environ.get('EVAL_ONLY', '0')))
    package_only = bool(int(os.environ.get('PACKAGE_ONLY', '0')))
    vocab_size = int(os.environ.get('VOCAB_SIZE', 8192))
    num_layers = int(os.environ.get('NUM_LAYERS', 11))
    xsa_last_n = int(os.environ.get('XSA_LAST_N', 11))
    model_dim = int(os.environ.get('MODEL_DIM', 512))
    embedding_dim = int(os.environ.get('EMBEDDING_DIM', 512))
    num_kv_heads = int(os.environ.get('NUM_KV_HEADS', 4))
    num_heads = int(os.environ.get('NUM_HEADS', 8))
    mlp_mult = float(os.environ.get('MLP_MULT', 4.0))
    skip_gates_enabled = bool(int(os.environ.get('SKIP_GATES_ENABLED', '1')))
    tie_embeddings = bool(int(os.environ.get('TIE_EMBEDDINGS', '1')))
    logit_softcap = float(os.environ.get('LOGIT_SOFTCAP', 30.0))
    loss_in_fp32 = bool(int(os.environ.get('LOSS_IN_FP32', '1')))
    loss_chunk_tokens = int(os.environ.get('LOSS_CHUNK_TOKENS', '0'))
    rope_base = float(os.environ.get('ROPE_BASE', 10000.0))
    rope_dims = int(os.environ.get('ROPE_DIMS', 16))
    rope_train_seq_len = int(os.environ.get('ROPE_TRAIN_SEQ_LEN', 2048))
    ln_scale = bool(int(os.environ.get('LN_SCALE', '1')))
    qk_gain_init = float(os.environ.get('QK_GAIN_INIT', 5.0))
    num_loops = int(os.environ.get('NUM_LOOPS', 2))
    loop_start = int(os.environ.get('LOOP_START', 3))
    loop_end = int(os.environ.get('LOOP_END', 5))
    enable_looping_at = float(os.environ.get('ENABLE_LOOPING_AT', 0.35))
    start_looping_active = bool(int(os.environ.get('START_LOOPING_ACTIVE', '0')))
    parallel_residual_start = int(os.environ.get('PARALLEL_RESIDUAL_START', 7))
    activation_checkpointing = bool(int(os.environ.get('ACTIVATION_CHECKPOINTING', '0')))
    activation_checkpoint_start_layer = int(os.environ.get('ACTIVATION_CHECKPOINT_START_LAYER', '0'))
    min_lr = float(os.environ.get('MIN_LR', 0.0))
    embed_lr = float(os.environ.get('EMBED_LR', 0.6))
    head_lr = float(os.environ.get('HEAD_LR', 0.008))
    tied_embed_lr = float(os.environ.get('TIED_EMBED_LR', 0.03))
    tied_embed_init_std = float(os.environ.get('TIED_EMBED_INIT_STD', 0.005))
    matrix_lr = float(os.environ.get('MATRIX_LR', 0.022))
    scalar_lr = float(os.environ.get('SCALAR_LR', 0.02))
    muon_momentum = float(os.environ.get('MUON_MOMENTUM', 0.99))
    muon_backend_steps = int(os.environ.get('MUON_BACKEND_STEPS', 5))
    muon_momentum_warmup_start = float(os.environ.get('MUON_MOMENTUM_WARMUP_START', 0.92))
    muon_momentum_warmup_steps = int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS', 1500))
    muon_row_normalize = bool(int(os.environ.get('MUON_ROW_NORMALIZE', '1')))
    beta1 = float(os.environ.get('BETA1', 0.9))
    beta2 = float(os.environ.get('BETA2', 0.95))
    adam_eps = float(os.environ.get('ADAM_EPS', 1e-08))
    grad_clip_norm = float(os.environ.get('GRAD_CLIP_NORM', 0.3))
    eval_stride = int(os.environ.get('EVAL_STRIDE', 64))
    muon_beta2 = float(os.environ.get('MUON_BETA2', 0.95))
    adam_wd = float(os.environ.get('ADAM_WD', 0.02))
    muon_wd = float(os.environ.get('MUON_WD', 0.095))
    embed_wd = float(os.environ.get('EMBED_WD', 0.085))
    ema_enabled = bool(int(os.environ.get('EMA_ENABLED', '1')))
    ema_decay = float(os.environ.get('EMA_DECAY', 0.9965))
    optimizer_state_rewarm_on_seq_change = bool(int(os.environ.get('OPTIMIZER_STATE_REWARM_ON_SEQ_CHANGE', '0')))
    optimizer_state_rewarm_on_cap_start = bool(int(os.environ.get('OPTIMIZER_STATE_REWARM_ON_CAP_START', '0')))
    optimizer_state_exp_avg_scale = float(os.environ.get('OPTIMIZER_STATE_EXP_AVG_SCALE', '1.0'))
    optimizer_state_exp_avg_sq_scale = float(os.environ.get('OPTIMIZER_STATE_EXP_AVG_SQ_SCALE', '1.0'))
    optimizer_state_muon_buffer_scale = float(os.environ.get('OPTIMIZER_STATE_MUON_BUFFER_SCALE', os.environ.get('OPTIMIZER_STATE_EXP_AVG_SCALE', '1.0')))
    block_growth_start_layer = int(os.environ.get('BLOCK_GROWTH_START_LAYER', '-1'))
    block_growth_scale_schedule = os.environ.get('BLOCK_GROWTH_SCALE_SCHEDULE', '').strip()
    block_growth_log_updates = bool(int(os.environ.get('BLOCK_GROWTH_LOG_UPDATES', '0')))
    loss_tail_scale_schedule = os.environ.get('LOSS_TAIL_SCALE_SCHEDULE', '').strip()
    loss_tail_start_frac = float(os.environ.get('LOSS_TAIL_START_FRAC', '1.0'))
    loss_tail_power = float(os.environ.get('LOSS_TAIL_POWER', '1.0'))
    ttt_enabled = bool(int(os.environ.get('TTT_ENABLED', '0')))
    ttt_lr = float(os.environ.get('TTT_LR', 0.005))
    ttt_epochs = int(os.environ.get('TTT_EPOCHS', 3))
    ttt_momentum = float(os.environ.get('TTT_MOMENTUM', 0.9))
    ttt_chunk_tokens = int(os.environ.get('TTT_CHUNK_TOKENS', 32768))
    ttt_log_every_chunks = int(os.environ.get('TTT_LOG_EVERY_CHUNKS', '0'))
    ttt_lr_schedule = os.environ.get('TTT_LR_SCHEDULE', '').strip()
    sliding_batch_seqs = int(os.environ.get('SLIDING_BATCH_SEQS', 32))
    ttt_batch_seqs = int(os.environ.get('TTT_BATCH_SEQS', 32))
    etlb_enabled = bool(int(os.environ.get('ETLB_ENABLED', '0')))
    etlb_lr = float(os.environ.get('ETLB_LR', 0.05))
    etlb_steps = int(os.environ.get('ETLB_STEPS', 5))
    etlb_clip = float(os.environ.get('ETLB_CLIP', 3.0))
    compressor = os.environ.get('COMPRESSOR', 'brotli')
    skip_final_packaging = bool(int(os.environ.get('SKIP_FINAL_PACKAGING', '0')))
    save_pre_quant_snapshot = bool(int(os.environ.get('SAVE_PRE_QUANT_SNAPSHOT', '0')))
    save_quantized_snapshot = bool(int(os.environ.get('SAVE_QUANTIZED_SNAPSHOT', '0')))
    gptq_calibration_batches = int(os.environ.get('GPTQ_CALIBRATION_BATCHES', 64))
    gptq_reserve_seconds = float(os.environ.get('GPTQ_RESERVE_SECONDS', 12.0))
    matrix_bits = int(os.environ.get('MATRIX_BITS', 6))
    embed_bits = int(os.environ.get('EMBED_BITS', 8))
    int8_force_patterns = tuple((pattern.strip() for pattern in os.environ.get('INT8_FORCE_PATTERNS', '').split(',') if pattern.strip()))
    matrix_clip_sigmas = float(os.environ.get('MATRIX_CLIP_SIGMAS', 12.85))
    embed_clip_sigmas = float(os.environ.get('EMBED_CLIP_SIGMAS', 20.0))
    qat_enabled = bool(int(os.environ.get('QAT_ENABLED', '0')))
    late_qat_threshold = float(os.environ.get('LATE_QAT_THRESHOLD', '0.0'))
    late_qat_mode = os.environ.get('LATE_QAT_MODE', 'structural').strip().lower()
    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    is_main_process = rank == 0
    grad_accum_steps = 8 // world_size
    dataset_name = os.environ.get('DATASET_NAME', f'fineweb10B_sp{vocab_size}')
    datasets_dir = os.environ.get('DATASETS_DIR', os.path.join(data_dir, 'datasets', dataset_name))
    train_files = os.environ.get('TRAIN_FILES', os.path.join(datasets_dir, 'fineweb_train_*.bin'))
    val_files = os.environ.get('VAL_FILES', os.path.join(datasets_dir, 'fineweb_val_*.bin'))
    tokenizer_path = os.environ.get('TOKENIZER_PATH', os.path.join(data_dir, 'tokenizers', f'fineweb_{vocab_size}_bpe.model'))
    run_dir = os.environ.get('RUN_DIR', os.path.join('runs', run_id))
    checkpoint_dir = os.environ.get('CHECKPOINT_DIR', os.path.join(run_dir, 'checkpoints'))
    resume_checkpoint = os.environ.get('RESUME_CHECKPOINT', os.path.join(checkpoint_dir, 'latest.pt'))
    init_model_path = os.environ.get('INIT_MODEL_PATH', '')
    save_checkpoint_every = int(os.environ.get('SAVE_CHECKPOINT_EVERY', '1000'))
    keep_step_checkpoints = bool(int(os.environ.get('KEEP_STEP_CHECKPOINTS', '0')))
    logfile = os.environ.get('LOGFILE', f'logs/{run_id}.txt')
    model_path = os.environ.get('MODEL_PATH', os.path.join(run_dir, 'final_model.pt'))
    best_model_path = os.environ.get('BEST_MODEL_PATH', os.path.join(run_dir, 'best_model.pt'))
    best_model_metadata_path = os.environ.get('BEST_MODEL_METADATA_PATH', os.path.join(run_dir, 'best_model.json'))
    init_train_loader_state_path = os.environ.get('INIT_TRAIN_LOADER_STATE_PATH', '').strip()
    quantized_model_path = os.environ.get('QUANTIZED_MODEL_PATH', os.path.join(run_dir, 'final_model.int6.ptz'))
_logger_hparams = None

def set_logging_hparams(h):
    global _logger_hparams
    _logger_hparams = h

def log(msg, console=True):
    if _logger_hparams is None:
        print(msg)
        return
    if _logger_hparams.is_main_process:
        if console:
            print(msg)
        if _logger_hparams.logfile is not None:
            with open(_logger_hparams.logfile, 'a', encoding='utf-8') as f:
                print(msg, file=f)


def parse_train_seq_schedule(schedule: str, default_seq_len: int) -> list[tuple[float, int]]:
    if not schedule.strip():
        return [(1.0, int(default_seq_len))]
    plan: list[tuple[float, int]] = []
    for raw_stage in schedule.split(','):
        raw_stage = raw_stage.strip()
        if not raw_stage:
            continue
        if '@' not in raw_stage:
            raise ValueError(
                f'Invalid TRAIN_SEQ_SCHEDULE stage `{raw_stage}`; expected format like `1024@0.35`'
            )
        seq_raw, progress_raw = raw_stage.split('@', 1)
        seq_len = int(seq_raw.strip())
        progress = float(progress_raw.strip())
        if seq_len <= 0:
            raise ValueError('TRAIN_SEQ_SCHEDULE sequence lengths must be positive')
        if not (0.0 < progress <= 1.0):
            raise ValueError('TRAIN_SEQ_SCHEDULE progress fractions must be in (0, 1]')
        plan.append((progress, seq_len))
    if not plan:
        return [(1.0, int(default_seq_len))]
    plan.sort(key=lambda item: item[0])
    if plan[-1][0] < 1.0:
        plan.append((1.0, plan[-1][1]))
    return plan


def parse_scalar_schedule(schedule: str, default_value: float) -> list[tuple[float, float]]:
    if not schedule.strip():
        return [(1.0, float(default_value))]
    plan: list[tuple[float, float]] = []
    for raw_stage in schedule.split(','):
        raw_stage = raw_stage.strip()
        if not raw_stage:
            continue
        if '@' not in raw_stage:
            raise ValueError(
                f'Invalid scalar schedule stage `{raw_stage}`; expected format like `0.75@0.20`'
            )
        value_raw, progress_raw = raw_stage.split('@', 1)
        value = float(value_raw.strip())
        progress = float(progress_raw.strip())
        if value < 0.0:
            raise ValueError('Scalar schedule values must be non-negative')
        if not (0.0 <= progress <= 1.0):
            raise ValueError('Scalar schedule progress fractions must be in [0, 1]')
        plan.append((progress, value))
    if not plan:
        return [(1.0, float(default_value))]
    plan.sort(key=lambda item: item[0])
    if plan[0][0] > 0.0:
        plan.insert(0, (0.0, plan[0][1]))
    if plan[-1][0] < 1.0:
        plan.append((1.0, plan[-1][1]))
    return plan


def schedule_value(plan: list[tuple[float, float]], progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    prev_progress, prev_value = plan[0]
    if progress <= prev_progress:
        return prev_value
    for next_progress, next_value in plan[1:]:
        if progress <= next_progress:
            span = max(next_progress - prev_progress, 1e-9)
            alpha = (progress - prev_progress) / span
            return prev_value + alpha * (next_value - prev_value)
        prev_progress, prev_value = next_progress, next_value
    return plan[-1][1]


def max_train_seq_len_from_schedule(plan: list[tuple[float, int]], default_seq_len: int) -> int:
    return max([int(default_seq_len), *[seq_len for _, seq_len in plan]])


def validate_train_seq_plan_compatibility(
    plan: list[tuple[float, int]],
    *,
    global_tokens: int,
    world_size: int,
    grad_accum_steps: int,
) -> int:
    denom = world_size * grad_accum_steps
    if denom <= 0:
        raise ValueError(f'Invalid world_size * grad_accum_steps={denom}')
    if global_tokens % denom != 0:
        raise ValueError(
            f'TRAIN_BATCH_TOKENS={global_tokens} must be divisible by world_size*grad_accum_steps={denom}'
        )
    local_tokens = global_tokens // denom
    invalid_seq_lens = sorted({seq_len for _, seq_len in plan if local_tokens % seq_len != 0})
    if invalid_seq_lens:
        raise ValueError(
            'TRAIN_SEQ_SCHEDULE contains sequence lengths incompatible with the local micro-batch: '
            f'local_tokens={local_tokens}, invalid_seq_lens={invalid_seq_lens}. '
            f'Each seq_len must divide {local_tokens} exactly.'
        )
    return local_tokens


def training_progress(
    *,
    step: int,
    iterations: int,
    elapsed_ms: float,
    max_wallclock_ms: float | None,
    schedule_mode: str,
) -> float:
    if schedule_mode == 'step' or max_wallclock_ms is None or max_wallclock_ms <= 0:
        return min(max(step / max(iterations, 1), 0.0), 1.0)
    if schedule_mode != 'wallclock':
        raise ValueError(
            f"Unsupported TRAIN_SEQ_SCHEDULE_MODE={schedule_mode!r}; expected 'wallclock' or 'step'"
        )
    return min(max(elapsed_ms / max_wallclock_ms, 0.0), 1.0)


def current_train_seq_len(
    plan: list[tuple[float, int]],
    *,
    step: int,
    iterations: int,
    elapsed_ms: float,
    max_wallclock_ms: float | None,
    schedule_mode: str,
) -> tuple[int, float]:
    progress = training_progress(
        step=step,
        iterations=iterations,
        elapsed_ms=elapsed_ms,
        max_wallclock_ms=max_wallclock_ms,
        schedule_mode=schedule_mode,
    )
    for threshold, seq_len in plan:
        if progress <= threshold:
            return seq_len, progress
    return (plan[-1][1], progress)

class ValidationData:

    def __init__(self, h, device):
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        if int(self.sp.vocab_size()) != h.vocab_size:
            raise ValueError(f'VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={int(self.sp.vocab_size())}')
        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len, h.val_max_tokens)
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = build_sentencepiece_luts(self.sp, h.vocab_size, device)

def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    assert sp.piece_to_id('▁') != sp.unk_id(), "Tokenizer must have '▁' (space) as its own token for correct BPB byte counting"
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith('▁'):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode('utf-8'))
    return (torch.tensor(base_bytes_np, dtype=torch.int16, device=device), torch.tensor(has_leading_space_np, dtype=torch.bool, device=device), torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device))

def load_validation_tokens(pattern, seq_len, max_tokens=0):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f'No files found for pattern: {pattern}')
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    if max_tokens > 0:
        # Keep a deterministic prefix for fast/consistent mid-train evaluation.
        capped = min(tokens.numel() - 1, max_tokens)
        usable_cap = capped // seq_len * seq_len
        if usable_cap <= 0:
            raise ValueError(
                f'VAL_MAX_TOKENS={max_tokens} is too small for EVAL_SEQ_LEN={seq_len}; '
                f'need at least {seq_len} tokens'
            )
        tokens = tokens[:usable_cap + 1]
    usable = (tokens.numel() - 1) // seq_len * seq_len
    if usable <= 0:
        raise ValueError(f'Validation split is too short for TRAIN_SEQ_LEN={seq_len}')
    return tokens[:usable + 1]

def load_data_shard(file):
    header_bytes = 256 * np.dtype('<i4').itemsize
    token_bytes = np.dtype('<u2').itemsize
    header = np.fromfile(file, dtype='<i4', count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f'Unexpected shard header for {file}')
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f'Shard size mismatch for {file}: expected {expected_size} bytes')
    tokens_np = np.fromfile(file, dtype='<u2', count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f'Short read for {file}')
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))
_SHARD_HEADER_BYTES = 256 * np.dtype('<i4').itemsize
_SHARD_NTOKENS_CACHE = {}
_MMAP_CACHE = {}

def _read_num_tokens(file):
    key = str(file)
    cached = _SHARD_NTOKENS_CACHE.get(key)
    if cached is not None:
        return cached
    header = np.fromfile(file, dtype='<i4', count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f'Unexpected shard header for {file}')
    n = int(header[2])
    _SHARD_NTOKENS_CACHE[key] = n
    return n

def _get_shard_memmap(file):
    key = str(file)
    mm = _MMAP_CACHE.get(key)
    if mm is not None:
        return mm
    n = _read_num_tokens(file)
    mm = np.memmap(file, mode='r', dtype='<u2', offset=_SHARD_HEADER_BYTES, shape=(n,))
    _MMAP_CACHE[key] = mm
    return mm

class ShuffledSequenceLoader:

    def __init__(self, h, device):
        self.world_size = h.world_size
        self.current_seq_len = h.train_seq_len
        self.device = device
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files:
            raise FileNotFoundError(f'No files found for pattern: {h.train_files}')
        self.files = all_files[h.rank::h.world_size]
        self.rng = np.random.Generator(np.random.PCG64(h.rank))
        self.num_tokens = [_read_num_tokens(f) for f in self.files]
        self.start_inds = [[] for _ in self.files]
        for si in range(len(self.files)):
            self._reset_shard(si)

    def _reset_shard(self, si):
        seq_len = self.current_seq_len
        max_phase = min(seq_len - 1, max(0, self.num_tokens[si] - seq_len - 1))
        phase = int(self.rng.integers(max_phase + 1)) if max_phase > 0 else 0
        num_sequences = (self.num_tokens[si] - 1 - phase) // seq_len
        sequence_order = self.rng.permutation(num_sequences)
        self.start_inds[si] = (phase + sequence_order * seq_len).tolist()

    def _set_seq_len(self, seq_len):
        seq_len = int(seq_len)
        if seq_len == self.current_seq_len:
            return
        self.current_seq_len = seq_len
        for si in range(len(self.files)):
            self._reset_shard(si)

    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        self._set_seq_len(seq_len)
        device_tokens = global_tokens // (self.world_size * grad_accum_steps)
        device_batch_size = device_tokens // self.current_seq_len
        remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
        x = torch.empty((device_batch_size, self.current_seq_len), dtype=torch.int64)
        y = torch.empty((device_batch_size, self.current_seq_len), dtype=torch.int64)
        for bi in range(device_batch_size):
            total = remaining.sum()
            if total <= 0:
                for si in range(len(self.files)):
                    self._reset_shard(si)
                remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
                total = remaining.sum()
            probs = remaining / total
            si = int(self.rng.choice(len(self.files), p=probs))
            start_ind = self.start_inds[si].pop()
            remaining[si] -= 1
            mm = _get_shard_memmap(self.files[si])
            window = torch.as_tensor(
                np.array(mm[start_ind:start_ind + self.current_seq_len + 1], dtype=np.int64)
            )
            x[bi] = window[:-1]
            y[bi] = window[1:]
        return (x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True))

    def state_dict(self):
        return {
            'current_seq_len': int(self.current_seq_len),
            'rng_state': copy.deepcopy(self.rng.bit_generator.state),
            'start_inds': copy.deepcopy(self.start_inds),
        }

    def load_state_dict(self, state):
        self.current_seq_len = int(state['current_seq_len'])
        self.rng = np.random.Generator(np.random.PCG64())
        self.rng.bit_generator.state = copy.deepcopy(state['rng_state'])
        self.start_inds = [list(indices) for indices in state['start_inds']]

class RMSNorm(nn.Module):

    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    _qat_enabled: bool = False

    def forward(self, x):
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(INT6_QAT_SCALE_CLAMP_MIN)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -31, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

class Rotary(nn.Module):

    def __init__(self, dim, base=10000.0, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._sin_cached is None or self._seq_len_cached != seq_len or (self._cos_cached.device != device):
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * scale ** (rd / (rd - 2))
                inv_freq = 1.0 / new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd)
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        # Always materialize fresh tensors after dtype casting. This keeps RoPE
        # caches safe to reuse across eval/train transitions without relying on
        # torch.is_inference_mode_enabled(), which torch.compile cannot trace.
        cos = self._cos_cached.to(dtype=dtype).clone()
        sin = self._sin_cached.to(dtype=dtype).clone()
        return (cos, sin)

def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = (x[..., :rope_dims], x[..., rope_dims:])
        half = rope_dims // 2
        x1, x2 = (x_rope[..., :half], x_rope[..., half:])
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = (x[..., :half], x[..., half:])
    return torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('model_dim must be divisible by num_heads')
        if num_heads % num_kv_heads != 0:
            raise ValueError('num_heads must be divisible by num_kv_heads')
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.enable_gqa = num_heads != num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError('head_dim must be even for RoPE')
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len)
        self.use_xsa = False

    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True, enable_gqa=self.enable_gqa)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)

class MLP(nn.Module):

    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())

class Block(nn.Module):

    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, train_seq_len, layer_idx=0, ln_scale=False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        self.parallel = False

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor)
        if self.parallel:
            mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor)
            x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out + self.mlp_scale.to(dtype=x_in.dtype)[None, None, :] * mlp_out
        else:
            x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
            x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        return x_out

class GPT(nn.Module):

    def __init__(self, h):
        super().__init__()
        if h.logit_softcap <= 0.0:
            raise ValueError(f'logit_softcap must be positive, got {h.logit_softcap}')
        self.tie_embeddings = h.tie_embeddings
        self.tied_embed_init_std = h.tied_embed_init_std
        self.logit_softcap = h.logit_softcap
        self.tok_emb = nn.Embedding(h.vocab_size, h.embedding_dim)
        if h.embedding_dim != h.model_dim:
            self.embed_proj = CastedLinear(h.embedding_dim, h.model_dim, bias=False)
            self.head_proj = CastedLinear(h.model_dim, h.embedding_dim, bias=False)
        else:
            self.embed_proj = None
            self.head_proj = None
        self.num_encoder_layers = h.num_layers // 2
        self.num_decoder_layers = h.num_layers - self.num_encoder_layers
        self.activation_checkpointing = h.activation_checkpointing
        self.activation_checkpoint_start_layer = max(0, h.activation_checkpoint_start_layer)
        # Manual per-block activation checkpointing is compatible with the
        # root-only FSDP path, but not with child FSDP-wrapped blocks.
        self.fsdp_child_wrapped_blocks = bool(h.fsdp_enabled and h.fsdp_auto_wrap_blocks)
        self.blocks = nn.ModuleList([Block(h.model_dim, h.num_heads, h.num_kv_heads, h.mlp_mult, h.rope_base, h.qk_gain_init, h.train_seq_len, layer_idx=i, ln_scale=h.ln_scale) for i in range(h.num_layers)])
        if h.rope_dims > 0:
            head_dim = h.model_dim // h.num_heads
            for block in self.blocks:
                block.attn.rope_dims = h.rope_dims
                block.attn.rotary = Rotary(head_dim, base=h.rope_base, train_seq_len=h.train_seq_len, rope_dims=h.rope_dims)
        self.final_norm = RMSNorm()
        self.lm_head = None if h.tie_embeddings else CastedLinear(h.embedding_dim, h.vocab_size, bias=False)
        self.loss_in_fp32 = h.loss_in_fp32
        self.loss_chunk_tokens = max(0, h.loss_chunk_tokens)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        if h.xsa_last_n > 0:
            for i in range(max(0, h.num_layers - h.xsa_last_n), h.num_layers):
                self.blocks[i].attn.use_xsa = True
        if h.parallel_residual_start >= 0:
            for i in range(h.parallel_residual_start, h.num_layers):
                self.blocks[i].parallel = True
        self.block_growth_start_layer = h.block_growth_start_layer
        self.register_buffer('block_growth_scales', torch.ones(h.num_layers, dtype=torch.float32))
        self.looping_active = False
        if h.num_loops > 0:
            loop_seg = list(range(h.loop_start, h.loop_end + 1))
            all_indices = list(range(h.loop_start))
            for _ in range(h.num_loops + 1):
                all_indices.extend(loop_seg)
            all_indices.extend(range(h.loop_end + 1, h.num_layers))
            num_enc = len(all_indices) // 2
            self.encoder_indices = all_indices[:num_enc]
            self.decoder_indices = all_indices[num_enc:]
        else:
            self.encoder_indices = list(range(self.num_encoder_layers))
            self.decoder_indices = list(range(self.num_encoder_layers, h.num_layers))
        self.num_skip_weights = min(len(self.encoder_indices), len(self.decoder_indices))
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, h.model_dim, dtype=torch.float32))
        self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, h.model_dim, dtype=torch.float32)) if h.skip_gates_enabled else None
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, '_zero_init', False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and (module.weight.shape[1] >= 64):
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def _run_block(self, layer_idx, x, x0):
        block = self.blocks[layer_idx]
        if (
            self.activation_checkpointing
            and not self.fsdp_child_wrapped_blocks
            and self.training
            and torch.is_grad_enabled()
            and layer_idx >= self.activation_checkpoint_start_layer
        ):
            return activation_checkpoint(
                block,
                x,
                x0,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return block(x, x0)

    def set_block_growth_scale(self, scale):
        self.block_growth_scales.fill_(float(scale))

    def _apply_block_growth(self, layer_idx, x_prev, x_next):
        if self.block_growth_start_layer < 0 or layer_idx < self.block_growth_start_layer:
            return x_next
        scale = self.block_growth_scales[layer_idx].to(dtype=x_next.dtype)
        return x_prev + scale * (x_next - x_prev)

    def _project_logits(self, x):
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward_hidden(self, input_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x0 = x
        skips = []
        enc_iter = self.encoder_indices if self.looping_active else range(self.num_encoder_layers)
        dec_iter = self.decoder_indices if self.looping_active else range(self.num_encoder_layers, self.num_encoder_layers + self.num_decoder_layers)
        for i in enc_iter:
            x_prev = x
            x = self._run_block(i, x, x0)
            x = self._apply_block_growth(i, x_prev, x)
            skips.append(x)
        for skip_idx, i in enumerate(dec_iter):
            if skip_idx < self.num_skip_weights and skips:
                scaled_skip = self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
                if self.skip_gates is not None:
                    g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                    x = torch.lerp(scaled_skip, x, g)
                else:
                    x = x + scaled_skip
            x_prev = x
            x = self._run_block(i, x, x0)
            x = self._apply_block_growth(i, x_prev, x)
        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        return x

    def _chunked_cross_entropy(self, hidden, target_ids, *, loss_weights=None, chunk_tokens=None, reduction='mean'):
        if reduction not in ('mean', 'none'):
            raise ValueError(f'Unsupported reduction={reduction!r}')
        flat_hidden = hidden.reshape(-1, hidden.size(-1))
        flat_targets = target_ids.reshape(-1)
        total_tokens = flat_targets.numel()
        chunk_tokens = self.loss_chunk_tokens if chunk_tokens is None else int(chunk_tokens)
        chunk_tokens = max(0, chunk_tokens)
        use_chunking = 0 < chunk_tokens < total_tokens

        flat_weights = None
        if loss_weights is not None:
            weights = loss_weights
            if weights.ndim == 1:
                weights = weights.unsqueeze(0)
            flat_weights = weights.expand_as(target_ids).reshape(-1)

        def project_logits_slice(start, end):
            logits = self._project_logits(flat_hidden[start:end])
            return logits.float() if self.loss_in_fp32 else logits

        if not use_chunking:
            flat_logits = project_logits_slice(0, total_tokens)
            if reduction == 'none':
                return F.cross_entropy(flat_logits, flat_targets, reduction='none').reshape_as(target_ids)
            if flat_weights is None:
                return F.cross_entropy(flat_logits, flat_targets, reduction='mean')
            token_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')
            weights = flat_weights.to(device=token_loss.device, dtype=token_loss.dtype)
            return (token_loss * weights).sum() / weights.sum().clamp_min(1e-9)

        if reduction == 'none':
            token_losses = []
            for start in range(0, total_tokens, chunk_tokens):
                end = min(start + chunk_tokens, total_tokens)
                chunk_logits = project_logits_slice(start, end)
                token_losses.append(
                    F.cross_entropy(chunk_logits, flat_targets[start:end], reduction='none')
                )
            return torch.cat(token_losses, dim=0).reshape_as(target_ids)

        numerator = torch.zeros((), device=flat_hidden.device, dtype=torch.float32)
        denom = torch.zeros((), device=flat_hidden.device, dtype=torch.float32)
        for start in range(0, total_tokens, chunk_tokens):
            end = min(start + chunk_tokens, total_tokens)
            chunk_logits = project_logits_slice(start, end)
            if flat_weights is None:
                numerator = numerator + F.cross_entropy(
                    chunk_logits,
                    flat_targets[start:end],
                    reduction='sum',
                )
                denom = denom + (end - start)
                continue
            chunk_loss = F.cross_entropy(chunk_logits, flat_targets[start:end], reduction='none')
            weights = flat_weights[start:end].to(device=chunk_loss.device, dtype=chunk_loss.dtype)
            numerator = numerator + (chunk_loss * weights).sum()
            denom = denom + weights.sum()
        return numerator / denom.clamp_min(1e-9)

    def forward_logits(self, input_ids):
        hidden = self.forward_hidden(input_ids)
        return self._project_logits(hidden)

    def forward_token_nll(self, input_ids, target_ids, chunk_tokens=None):
        hidden = self.forward_hidden(input_ids)
        return self._chunked_cross_entropy(
            hidden,
            target_ids,
            chunk_tokens=chunk_tokens,
            reduction='none',
        )

    def forward(self, input_ids, target_ids, loss_weights=None):
        hidden = self.forward_hidden(input_ids)
        return self._chunked_cross_entropy(
            hidden,
            target_ids,
            loss_weights=loss_weights,
            chunk_tokens=self.loss_chunk_tokens,
            reduction='mean',
        )

def classify_param(name):
    if 'tok_emb' in name or 'lm_head' in name:
        return 'embed'
    if '.mlp.' in name:
        return 'mlp'
    if '.attn.' in name or ('.proj.' in name and '.mlp.' not in name):
        return 'attn'
    return 'other'

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-07):
    a, b, c = (3.4445, -4.775, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):

    def __init__(self, params, lr, momentum, backend_steps, nesterov=True, weight_decay=0.0, row_normalize=False):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay, row_normalize=row_normalize))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group['params']
            if not params:
                continue
            lr = group['lr']
            momentum = group['momentum']
            backend_steps = group['backend_steps']
            nesterov = group['nesterov']
            total_params = sum((int(p.numel()) for p in params))
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    if group.get('row_normalize', False):
                        row_norms = g.float().norm(dim=-1, keepdim=True).clamp_min(1e-07)
                        g = g / row_norms.to(g.dtype)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get('weight_decay', 0.0)
            curr = 0
            for p in params:
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr:curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss
CONTROL_TENSOR_NAME_PATTERNS = tuple((pattern for pattern in os.environ.get('CONTROL_TENSOR_NAME_PATTERNS', 'attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates').split(',') if pattern))

class Optimizers:

    def __init__(self, h, base_model):
        block_named_params = list(base_model.blocks.named_parameters())
        matrix_params = [p for name, p in block_named_params if p.ndim == 2 and (not any((pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)))]
        scalar_params = [p for name, p in block_named_params if p.ndim < 2 or any((pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))]
        if base_model.skip_weights.numel() > 0:
            scalar_params.append(base_model.skip_weights)
        if base_model.skip_gates is not None and base_model.skip_gates.numel() > 0:
            scalar_params.append(base_model.skip_gates)
        token_lr = h.tied_embed_lr if h.tie_embeddings else h.embed_lr
        tok_params = [{'params': [base_model.tok_emb.weight], 'lr': token_lr, 'base_lr': token_lr}]
        self.optimizer_tok = torch.optim.AdamW(tok_params, betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.embed_wd, fused=True)
        self.uses_muon = not (h.fsdp_enabled and h.fsdp_use_adamw_for_matrix)
        if self.uses_muon:
            self.optimizer_matrix = Muon(matrix_params, lr=h.matrix_lr, momentum=h.muon_momentum, backend_steps=h.muon_backend_steps, weight_decay=h.muon_wd, row_normalize=h.muon_row_normalize)
        else:
            self.optimizer_matrix = torch.optim.AdamW(
                [{'params': matrix_params, 'lr': h.matrix_lr, 'base_lr': h.matrix_lr}],
                betas=(h.beta1, h.beta2),
                eps=h.adam_eps,
                weight_decay=h.muon_wd,
                fused=True,
            )
        self.optimizer_muon = self.optimizer_matrix if self.uses_muon else None
        for group in self.optimizer_matrix.param_groups:
            group['base_lr'] = h.matrix_lr
        self.optimizer_scalar = torch.optim.AdamW([{'params': scalar_params, 'lr': h.scalar_lr, 'base_lr': h.scalar_lr}], betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.adam_wd, fused=True)
        self.optimizers = [self.optimizer_tok, self.optimizer_matrix, self.optimizer_scalar]
        if base_model.lm_head is not None:
            self.optimizer_head = torch.optim.Adam([{'params': [base_model.lm_head.weight], 'lr': h.head_lr, 'base_lr': h.head_lr}], betas=(h.beta1, h.beta2), eps=h.adam_eps, fused=True)
            self.optimizers.insert(1, self.optimizer_head)
        else:
            self.optimizer_head = None

    def __iter__(self):
        return iter(self.optimizers)

    def zero_grad_all(self):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def step(self):
        for opt in self.optimizers:
            opt.step()
        self.zero_grad_all()

    def set_matrix_momentum(self, momentum):
        if self.optimizer_muon is None:
            return
        for group in self.optimizer_muon.param_groups:
            group['momentum'] = momentum

    def damp_state(self, exp_avg_scale=1.0, exp_avg_sq_scale=1.0, muon_buffer_scale=None):
        if muon_buffer_scale is None:
            muon_buffer_scale = exp_avg_scale
        adam_m_buffers = 0
        adam_v_buffers = 0
        adam_vmax_buffers = 0
        muon_buffers = 0
        adam_like_opts = [self.optimizer_tok, self.optimizer_scalar]
        if not self.uses_muon:
            adam_like_opts.insert(1, self.optimizer_matrix)
        if self.optimizer_head is not None:
            adam_like_opts.insert(1, self.optimizer_head)
        for opt in adam_like_opts:
            for state in opt.state.values():
                if 'exp_avg' in state:
                    state['exp_avg'].mul_(exp_avg_scale)
                    adam_m_buffers += 1
                if 'exp_avg_sq' in state:
                    state['exp_avg_sq'].mul_(exp_avg_sq_scale)
                    adam_v_buffers += 1
                if 'max_exp_avg_sq' in state:
                    state['max_exp_avg_sq'].mul_(exp_avg_sq_scale)
                    adam_vmax_buffers += 1
        if self.optimizer_muon is not None:
            for state in self.optimizer_muon.state.values():
                if 'momentum_buffer' in state:
                    state['momentum_buffer'].mul_(muon_buffer_scale)
                    muon_buffers += 1
        return adam_m_buffers, adam_v_buffers, adam_vmax_buffers, muon_buffers


def reset_runtime_caches(model):
    for module in model.modules():
        if isinstance(module, Rotary):
            module._seq_len_cached = 0
            module._cos_cached = None
            module._sin_cached = None


def restore_fp32_params(model):
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    for name, param in model.named_parameters():
        if (param.ndim < 2 or any((pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))) and param.dtype != torch.float32:
            param.data = param.data.float()


def _fsdp_sharding_strategy(name):
    mapping = {
        'FULL_SHARD': ShardingStrategy.FULL_SHARD,
        'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
        'NO_SHARD': ShardingStrategy.NO_SHARD,
        'HYBRID_SHARD': ShardingStrategy.HYBRID_SHARD,
        '_HYBRID_SHARD_ZERO2': ShardingStrategy._HYBRID_SHARD_ZERO2,
    }
    if name not in mapping:
        raise ValueError(f'Unknown FSDP_SHARDING_STRATEGY={name!r}')
    return mapping[name]


def _fsdp_backward_prefetch(name):
    mapping = {
        'BACKWARD_PRE': BackwardPrefetch.BACKWARD_PRE,
        'BACKWARD_POST': BackwardPrefetch.BACKWARD_POST,
        'NONE': None,
    }
    if name not in mapping:
        raise ValueError(f'Unknown FSDP_BACKWARD_PREFETCH={name!r}')
    return mapping[name]


def _fsdp_ignored_states(base_model):
    return [param for param in base_model.parameters() if param.dtype == torch.float32]


def build_fsdp_model(h, base_model, device):
    ignored_states = _fsdp_ignored_states(base_model)
    auto_wrap_policy = ModuleWrapPolicy({Block}) if h.fsdp_auto_wrap_blocks else None
    return FSDP(
        base_model,
        device_id=device,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=_fsdp_sharding_strategy(h.fsdp_sharding_strategy),
        cpu_offload=CPUOffload(offload_params=True) if h.fsdp_cpu_offload else None,
        backward_prefetch=_fsdp_backward_prefetch(h.fsdp_backward_prefetch),
        forward_prefetch=h.fsdp_forward_prefetch,
        limit_all_gathers=h.fsdp_limit_all_gathers,
        use_orig_params=h.fsdp_use_orig_params,
        sync_module_states=h.fsdp_sync_module_states,
        ignored_states=ignored_states,
    )


def _full_model_state_dict(h, stateful_model, base_model):
    if not h.fsdp_enabled:
        return base_model.state_dict()
    with FSDP.state_dict_type(
        stateful_model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        return stateful_model.state_dict()


def _full_optimizer_state_dicts(h, stateful_model, optimizers):
    if not h.fsdp_enabled:
        return [opt.state_dict() for opt in optimizers]
    with FSDP.state_dict_type(
        stateful_model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        return [FSDP.optim_state_dict(stateful_model, opt, opt.state_dict()) for opt in optimizers]


def _load_model_state_dict(h, stateful_model, base_model, state_dict):
    if not h.fsdp_enabled:
        return base_model.load_state_dict(state_dict, strict=True)
    with FSDP.state_dict_type(
        stateful_model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        return stateful_model.load_state_dict(state_dict, strict=True)


def _load_optimizer_state_dicts(h, stateful_model, optimizers, optimizer_states):
    if not h.fsdp_enabled:
        for opt, state in zip(optimizers, optimizer_states, strict=True):
            opt.load_state_dict(state)
        return
    with FSDP.state_dict_type(
        stateful_model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        for opt, state in zip(optimizers, optimizer_states, strict=True):
            loadable = FSDP.optim_state_dict_to_load(stateful_model, opt, state)
            opt.load_state_dict(loadable)

def collect_hessians(model, train_loader, h, device, n_calibration_batches=64):
    hessians = {}
    hooks = []

    def make_hook(name):

        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            if name not in hessians:
                hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
            hessians[name].addmm_(x.T, x)
        return hook_fn
    for name, module in model.named_modules():
        if isinstance(module, CastedLinear) and module.weight.numel() > 65536:
            cat = classify_param(name + '.weight')
            if cat in ('mlp', 'attn'):
                hooks.append(module.register_forward_hook(make_hook(name + '.weight')))
    if model.tie_embeddings:
        hook_module = model.head_proj if model.head_proj is not None else model.final_norm

        def make_output_hook(name):

            def hook_fn(module, inp, out):
                x = out.detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                if name not in hessians:
                    hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
                hessians[name].addmm_(x.T, x)
            return hook_fn
        hooks.append(hook_module.register_forward_hook(make_output_hook('tok_emb.weight')))
    model.eval()
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            x, _ = train_loader.next_batch(h.train_batch_tokens, h.train_seq_len, h.grad_accum_steps)
            model.forward_logits(x)
    for hook in hooks:
        hook.remove()
    for name in hessians:
        hessians[name] = hessians[name].cpu() / n_calibration_batches
    return hessians

def gptq_quantize_weight(w, H, clip_sigmas=3.0, clip_range=63, block_size=128, scale_clamp_min=1e-10):
    W_orig = w.float().clone()
    rows, cols = W_orig.shape
    H = H.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * H.diag().mean()
    H.diagonal().add_(damp)
    perm = torch.argsort(H.diag(), descending=True)
    invperm = torch.argsort(perm)
    W_perm = W_orig[:, perm].clone()
    W_perm[:, dead[perm]] = 0
    H = H[perm][:, perm]
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    row_std = W_orig.std(dim=1)
    s = (clip_sigmas * row_std / clip_range).clamp_min(scale_clamp_min).to(torch.float16)
    sf = s.float()
    Q = torch.zeros(rows, cols, dtype=torch.int8)
    W_work = W_perm.clone()
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W_block = W_work[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]
        Err = torch.zeros(rows, i2 - i1)
        for j in range(i2 - i1):
            w_col = W_block[:, j]
            d = Hinv_block[j, j]
            q_col = torch.clamp(torch.round(w_col / sf), -clip_range, clip_range)
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - q_col.float() * sf) / d
            Err[:, j] = err
            W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
        if i2 < cols:
            W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return (Q[:, invperm], s)

def gptq_mixed_quantize(state_dict, hessians, h):
    result = {}
    meta = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = 'passthrough (float16)'
            continue
        cs = h.embed_clip_sigmas if 'tok_emb' in name else h.matrix_clip_sigmas
        bits = h.embed_bits if 'tok_emb' in name else h.matrix_bits
        if h.int8_force_patterns and any((pattern in name for pattern in h.int8_force_patterns)):
            bits = max(bits, 8)
        scale_clamp_min = INT6_EXPORT_SCALE_CLAMP_MIN if bits <= 6 else _INT6_LEGACY_SCALE_CLAMP_MIN
        q, s = gptq_quantize_weight(
            t,
            hessians[name],
            clip_sigmas=cs,
            clip_range=2 ** (bits - 1) - 1,
            scale_clamp_min=scale_clamp_min,
        )
        result[name + '.q'] = q
        result[name + '.scale'] = s
        meta[name] = f'gptq (int{bits})'
    categories = collections.defaultdict(set)
    for name, cat in meta.items():
        short = re.sub('\\.\\d+$', '', re.sub('blocks\\.\\d+', 'blocks', name))
        categories[cat].add(short)
    log('Quantized weights:')
    for cat in sorted(categories):
        log(f"  {cat}: {', '.join(sorted(categories[cat]))}")
    return (result, meta)

def dequantize_mixed(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if 'passthrough' in info:
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = (result[name + '.q'], result[name + '.scale'])
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *[1] * (q.ndim - 1))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out
_BSHF_MAGIC = b'BSHF'

def _byte_shuffle(data, stride=2):
    if stride <= 1 or len(data) < stride:
        return data
    src = np.frombuffer(data, dtype=np.uint8)
    n = len(src)
    out = np.empty(n, dtype=np.uint8)
    dest_off = 0
    for pos in range(stride):
        chunk = src[pos::stride]
        out[dest_off:dest_off + len(chunk)] = chunk
        dest_off += len(chunk)
    return _BSHF_MAGIC + bytes([stride]) + out.tobytes()

def _byte_unshuffle(data):
    if len(data) < 5 or data[:4] != _BSHF_MAGIC:
        return data
    stride = data[4]
    if stride < 2:
        return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = len(payload)
    out = np.empty(n, dtype=np.uint8)
    src_off = 0
    for pos in range(stride):
        chunk_len = n // stride + (1 if pos < n % stride else 0)
        out[pos::stride][:chunk_len] = payload[src_off:src_off + chunk_len]
        src_off += chunk_len
    return out.tobytes()

def _compress(data, compressor):
    data = _byte_shuffle(data)
    if compressor == 'lzma':
        return lzma.compress(data, preset=6)
    elif compressor == 'brotli':
        import brotli
        return brotli.compress(data, quality=11)
    raise ValueError(f'Unknown compressor: {compressor!r}')

def _decompress(data, compressor):
    if compressor == 'lzma':
        raw = lzma.decompress(data)
    elif compressor == 'brotli':
        import brotli
        raw = brotli.decompress(data)
    else:
        raise ValueError(f'Unknown compressor: {compressor!r}')
    raw = _byte_unshuffle(raw)
    return raw

def serialize(h, base_model, code):
    code_bytes = len(code.encode('utf-8'))
    if h.is_main_process:
        torch.save(base_model.state_dict(), h.model_path)
        model_bytes = os.path.getsize(h.model_path)
        log(f'Serialized model: {model_bytes} bytes')
        log(f'Code size: {code_bytes} bytes')
    sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    device = torch.device('cuda', h.local_rank)
    log('GPTQ:collecting Hessians from calibration data...')
    t0 = time.perf_counter()
    calib_loader = ShuffledSequenceLoader(h, device)
    hessians = collect_hessians(base_model, calib_loader, h, device, n_calibration_batches=h.gptq_calibration_batches)
    log(f'GPTQ:collected {len(hessians)} Hessians in {time.perf_counter() - t0:.1f}s')
    quant_result, quant_meta = gptq_mixed_quantize(sd_cpu, hessians, h)
    quant_buf = io.BytesIO()
    torch.save({'w': quant_result, 'm': quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = _compress(quant_raw, h.compressor)
    quant_file_bytes = len(quant_blob)
    bytes_total = quant_file_bytes + code_bytes
    if h.is_main_process:
        with open(h.quantized_model_path, 'wb') as f:
            f.write(quant_blob)
        log(f'Serialized model quantized+{h.compressor}: {quant_file_bytes} bytes')
        log(f'Total submission size quantized+{h.compressor}: {bytes_total} bytes')
    return (bytes_total, quant_file_bytes)

def deserialize(h, device):
    eval_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(eval_model)
    sd_cpu = {k: v.detach().cpu() for k, v in eval_model.state_dict().items()}
    with open(h.quantized_model_path, 'rb') as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(_decompress(quant_blob_disk, h.compressor)), map_location='cpu')
    deq_state = dequantize_mixed(quant_state['w'], quant_state['m'], sd_cpu)
    eval_model.load_state_dict(deq_state, strict=True)
    return eval_model

def _loss_bpb(loss_sum, token_count, byte_count):
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return (val_loss, val_bpb)


def build_tail_loss_weights(seq_len, tail_start_frac, tail_scale, tail_power, device):
    if seq_len <= 0 or tail_scale <= 1.0 or tail_start_frac >= 1.0:
        return None
    if seq_len == 1:
        return torch.ones(1, device=device, dtype=torch.float32)
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    pos /= float(seq_len - 1)
    start = min(max(tail_start_frac, 0.0), 1.0)
    denom = max(1.0 - start, 1e-9)
    tail_progress = ((pos - start) / denom).clamp_(0.0, 1.0)
    if abs(tail_power - 1.0) > 1e-9:
        tail_progress = tail_progress.pow(tail_power)
    weights = 1.0 + (tail_scale - 1.0) * tail_progress
    weights /= weights.mean().clamp_min(1e-9)
    return weights

def eval_val(h, device, val_data, model):
    seq_len = h.eval_seq_len
    local_batch_tokens = h.val_batch_tokens // (h.world_size * h.grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(f'VAL_BATCH_SIZE must provide at least one sequence per rank; got VAL_BATCH_SIZE={h.val_batch_tokens}, WORLD_SIZE={h.world_size}, GRAD_ACCUM_STEPS={h.grad_accum_steps}, seq_len={seq_len}')
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_data.val_tokens.numel() - 1) // seq_len
    seq_start = total_seqs * h.rank // h.world_size
    seq_end = total_seqs * (h.rank + 1) // h.world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_data.val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (val_data.has_leading_space_lut[tgt_ids] & ~val_data.is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    model.train()
    return _loss_bpb(val_loss_sum, val_token_count, val_byte_count)

def eval_val_sliding(h, device, val_data, base_model, batch_seqs=None):
    batch_seqs = h.sliding_batch_seqs if batch_seqs is None else batch_seqs
    base_model.eval()
    logits_fn = None
    if h.loss_chunk_tokens <= 0:
        logits_fn = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    seq_len = h.eval_seq_len
    context_size = seq_len - h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, h.eval_stride) if ws + context_size < total_tokens]
    total_windows = len(window_starts)
    my_s = total_windows * h.rank // h.world_size
    my_e = total_windows * (h.rank + 1) // h.world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                we = min(ws + seq_len, total_tokens)
                wlen = we - ws
                wlens.append(wlen)
                chunk = val_data.val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if h.loss_chunk_tokens > 0:
                    nll = base_model.forward_token_nll(x_batch, y_batch, chunk_tokens=h.loss_chunk_tokens)
                else:
                    logits = logits_fn(x_batch)
                    nll = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)).float(),
                        y_batch.reshape(-1),
                        reduction='none',
                    ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else context_size
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    base_model.train()
    return _loss_bpb(loss_sum, token_count, byte_count)

def eval_val_ttt(h, device, val_data, base_model, batch_seqs=None):
    batch_seqs = h.ttt_batch_seqs if batch_seqs is None else batch_seqs
    rank = h.rank
    world_size = h.world_size
    seq_len = h.eval_seq_len
    stride = h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    ttt_chunk = h.ttt_chunk_tokens
    context_size = seq_len - stride
    window_starts = [ws for ws in range(0, total_tokens, stride) if ws + context_size < total_tokens]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        wlen = min(ws + seq_len, total_tokens) - ws
        s = 0 if ws == 0 else context_size
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)
    log(f'ttt:start chunks={num_chunks} ttt_lr={h.ttt_lr} ttt_epochs={h.ttt_epochs}')
    compiled_logits = None
    if h.loss_chunk_tokens <= 0:
        compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    ttt_lr_plan = parse_scalar_schedule(h.ttt_lr_schedule, 1.0) if h.ttt_lr_schedule else []
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    ttt_params = [p for p in base_model.parameters()]
    for p in ttt_params:
        p.requires_grad_(True)
    optimizer = torch.optim.SGD(ttt_params, lr=h.ttt_lr, momentum=h.ttt_momentum)
    eval_t0 = time.perf_counter()
    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
        my_s = len(windows) * rank // world_size
        my_e = len(windows) * (rank + 1) // world_size
        my_windows = windows[my_s:my_e]
        base_model.eval()
        with torch.no_grad():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    we = min(ws + seq_len, total_tokens)
                    wlen = we - ws
                    wlens.append(wlen)
                    chunk_tok = val_data.val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    if h.loss_chunk_tokens > 0:
                        nll = base_model.forward_token_nll(
                            x_batch,
                            y_batch,
                            chunk_tokens=h.loss_chunk_tokens,
                        )
                    else:
                        logits = compiled_logits(x_batch)
                        nll = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)).float(),
                            y_batch.reshape(-1),
                            reduction='none',
                        ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else context_size
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                    tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()
        is_last_chunk = ci == num_chunks - 1
        if not is_last_chunk and h.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                chunk_progress = ci / max(num_chunks - 1, 1)
                if ttt_lr_plan:
                    cos_lr = h.ttt_lr * schedule_value(ttt_lr_plan, chunk_progress)
                else:
                    cos_lr = h.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg['lr'] = cos_lr
                my_seq_s = chunk_seqs * rank // world_size
                my_seq_e = chunk_seqs * (rank + 1) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                for _ep in range(h.ttt_epochs):
                    for bs in range(0, my_chunk_seqs, batch_seqs):
                        be = min(bs + batch_seqs, my_chunk_seqs)
                        actual_bs = my_seq_s + bs
                        start_tok = chunk_start + actual_bs * seq_len
                        end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                        if end_tok > val_data.val_tokens.numel():
                            continue
                        local = val_data.val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            loss = base_model(x, y)
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
                        optimizer.step()
        log_every = max(h.ttt_log_every_chunks, 0)
        if log_every > 0 and (((ci + 1) % log_every) == 0 or ci == 0 or is_last_chunk):
            cur_loss, cur_bpb = _loss_bpb(loss_sum, token_count, byte_count)
            elapsed = time.perf_counter() - eval_t0
            log(
                f'ttt:progress chunk:{ci + 1}/{num_chunks} '
                f'progress:{(ci + 1) / max(num_chunks, 1):.3f} '
                f'val_loss:{cur_loss:.8f} val_bpb:{cur_bpb:.8f} '
                f'elapsed:{elapsed:.1f}s'
            )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()
    return _loss_bpb(loss_sum, token_count, byte_count)

def timed_eval(label, fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    val_loss, val_bpb = fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    log(f'{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms')
    return (val_loss, val_bpb)

def snapshot_run_file(src_path, run_id, suffix):
    src = Path(src_path)
    if not src.is_file():
        return
    dst = Path('logs') / f'{run_id}.{suffix}'
    shutil.copyfile(src, dst)
    log(f'snapshot:saved {dst}')


def _safe_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _safe_write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    tmp_path.replace(path)


def collect_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    if 'python' in state:
        random.setstate(state['python'])
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    if 'torch' in state:
        torch.random.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def save_training_checkpoint(
    h,
    stateful_model,
    base_model,
    optimizers,
    train_loader,
    *,
    step,
    training_time_ms,
    active_train_seq_len,
    seq_change_warmup_start_step,
    midrun_cap_active,
    midrun_cap_prev_scale,
    block_growth_prev_scale,
    best_val_loss,
    best_val_bpb,
    best_val_step,
    ema_state,
    reason,
):
    if not h.resume_checkpoint:
        return
    model_state = _full_model_state_dict(h, stateful_model, base_model)
    optimizer_states = _full_optimizer_state_dicts(h, stateful_model, optimizers)
    if not h.is_main_process:
        return
    checkpoint = {
        'schema_version': 1,
        'run_id': h.run_id,
        'step': int(step),
        'training_time_ms': float(training_time_ms),
        'active_train_seq_len': int(active_train_seq_len),
        'seq_change_warmup_start_step': seq_change_warmup_start_step,
        'midrun_cap_active': bool(midrun_cap_active),
        'midrun_cap_prev_scale': float(midrun_cap_prev_scale),
        'block_growth_prev_scale': float(block_growth_prev_scale),
        'best_val_loss': float(best_val_loss),
        'best_val_bpb': float(best_val_bpb),
        'best_val_step': int(best_val_step),
        'model': model_state,
        'optimizers': optimizer_states,
        'ema_state': ema_state,
        'train_loader': train_loader.state_dict() if not h.distributed else None,
        'rng_state': collect_rng_state(),
        'looping_active': bool(base_model.looping_active),
        'qat_enabled': bool(CastedLinear._qat_enabled),
        'reason': reason,
    }
    _safe_torch_save(checkpoint, h.resume_checkpoint)
    log(
        f'checkpoint:saved path:{h.resume_checkpoint} step:{step} reason:{reason}'
    )
    if h.keep_step_checkpoints:
        step_path = Path(h.checkpoint_dir) / f'step-{step:06d}.pt'
        _safe_torch_save(checkpoint, step_path)
        log(f'checkpoint:archived path:{step_path}')


def load_training_checkpoint(h, stateful_model, base_model, optimizers, train_loader):
    resume_path = Path(h.resume_checkpoint) if h.resume_checkpoint else None
    if resume_path is not None and resume_path.is_file():
        checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
        _load_model_state_dict(h, stateful_model, base_model, checkpoint['model'])
        reset_runtime_caches(base_model)
        _load_optimizer_state_dicts(h, stateful_model, optimizers, checkpoint['optimizers'])
        if checkpoint.get('ema_state') is not None:
            ema_state = {
                name: tensor.float().clone()
                for name, tensor in checkpoint['ema_state'].items()
            }
        else:
            ema_state = None
        if (not h.distributed) and checkpoint.get('train_loader') is not None:
            train_loader.load_state_dict(checkpoint['train_loader'])
        restore_rng_state(checkpoint.get('rng_state'))
        base_model.looping_active = bool(checkpoint.get('looping_active', base_model.looping_active))
        CastedLinear._qat_enabled = bool(checkpoint.get('qat_enabled', CastedLinear._qat_enabled))
        log(
            f'resume:loaded path:{resume_path} step:{checkpoint.get("step", 0)} '
            f'train_time:{checkpoint.get("training_time_ms", 0.0) / 60000:.1f}m'
        )
        return {
            'step': int(checkpoint.get('step', 0)),
            'training_time_ms': float(checkpoint.get('training_time_ms', 0.0)),
            'active_train_seq_len': int(checkpoint.get('active_train_seq_len', h.train_seq_len)),
            'seq_change_warmup_start_step': checkpoint.get('seq_change_warmup_start_step'),
            'midrun_cap_active': bool(checkpoint.get('midrun_cap_active', False)),
            'midrun_cap_prev_scale': float(checkpoint.get('midrun_cap_prev_scale', 1.0)),
            'block_growth_prev_scale': float(checkpoint.get('block_growth_prev_scale', 1.0)),
            'best_val_loss': float(checkpoint.get('best_val_loss', float('inf'))),
            'best_val_bpb': float(checkpoint.get('best_val_bpb', float('inf'))),
            'best_val_step': int(checkpoint.get('best_val_step', -1)),
            'ema_state': ema_state,
            'loaded_resume': True,
        }
    init_path = Path(h.init_model_path) if h.init_model_path else None
    if init_path is not None and init_path.is_file():
        state_dict = torch.load(init_path, map_location='cpu')
        missing_keys, unexpected_keys = base_model.load_state_dict(state_dict, strict=False)
        reset_runtime_caches(base_model)
        if missing_keys:
            log('init_model:missing_keys:' + ','.join(missing_keys))
        if unexpected_keys:
            log('init_model:unexpected_keys:' + ','.join(unexpected_keys))
        log(f'init_model:loaded path:{init_path}')
    return {
        'step': 0,
        'training_time_ms': 0.0,
        'active_train_seq_len': h.train_seq_len,
        'seq_change_warmup_start_step': None,
        'midrun_cap_active': False,
        'midrun_cap_prev_scale': 1.0,
        'block_growth_prev_scale': 1.0,
        'best_val_loss': float('inf'),
        'best_val_bpb': float('inf'),
        'best_val_step': -1,
        'ema_state': None,
        'loaded_resume': False,
    }

def run_quantized_evals(h, device, val_data):
    eval_model = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    compiled_model = None
    if h.roundtrip_eval_enabled:
        compiled_model = (
            torch.compile(eval_model, dynamic=h.compile_dynamic, fullgraph=h.compile_fullgraph)
            if h.compile_model else eval_model
        )
        quant_val_loss, quant_val_bpb = timed_eval('quantized', eval_val, h, device, val_data, compiled_model)
        log(f'final_int6_roundtrip_exact val_loss:{quant_val_loss:.8f} val_bpb:{quant_val_bpb:.8f}')
    if h.sliding_window_enabled:
        sliding_val_loss, sliding_val_bpb = timed_eval('quantized_sliding_window', eval_val_sliding, h, device, val_data, eval_model)
        log(
            f'final_int6_sliding_window_exact val_loss:{sliding_val_loss:.8f} '
            f'val_bpb:{sliding_val_bpb:.8f} stride:{h.eval_stride}'
        )
    if h.ttt_enabled:
        del eval_model
        if compiled_model is not None:
            del compiled_model
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        ttt_model = deserialize(h, device)
        if h.num_loops > 0:
            ttt_model.looping_active = True
        ttt_val_loss, ttt_val_bpb = timed_eval('quantized_ttt', eval_val_ttt, h, device, val_data, ttt_model)
        log(f'legal_ttt_exact val_loss:{ttt_val_loss:.8f} val_bpb:{ttt_val_bpb:.8f} stride:{h.eval_stride}')
        del ttt_model
    if h.etlb_enabled and h.sliding_window_enabled:
        eval_model = deserialize(h, device)
        if h.num_loops > 0:
            eval_model.looping_active = True
        timed_eval('quantized_sliding_etlb', eval_val_sliding_etlb, h, device, val_data, eval_model)

def train_model(h, device, val_data):
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    CastedLinear._qat_enabled = h.qat_enabled
    if h.fsdp_enabled and h.compile_model:
        log('fsdp:disabling torch.compile for this run')
        h.compile_model = False
    if h.fsdp_enabled and h.ema_enabled:
        log('fsdp:disabling EMA for this run')
        h.ema_enabled = False
    train_loader = ShuffledSequenceLoader(h, device)
    train_seq_plan = parse_train_seq_schedule(h.train_seq_schedule, h.train_seq_len)
    midrun_cap_plan = parse_scalar_schedule(h.midrun_cap_schedule, 1.0)
    if h.fsdp_enabled:
        compiled_model = build_fsdp_model(h, base_model, device)
        model = compiled_model
        optimizers = Optimizers(h, model)
    else:
        compiled_model = (
            torch.compile(base_model, dynamic=h.compile_dynamic, fullgraph=h.compile_fullgraph)
            if h.compile_model else base_model
        )
        if h.distributed:
            model = DDP(compiled_model, device_ids=[h.local_rank], broadcast_buffers=False)
        else:
            model = compiled_model
        optimizers = Optimizers(h, base_model)
    resume_state = load_training_checkpoint(h, compiled_model if h.fsdp_enabled else base_model, base_model, optimizers, train_loader)
    init_train_loader_state = None
    if (not resume_state['loaded_resume']) and h.init_train_loader_state_path:
        init_loader_path = Path(h.init_train_loader_state_path)
        if init_loader_path.is_file():
            init_train_loader_state = torch.load(init_loader_path, map_location='cpu', weights_only=False)
            log(f'init_train_loader_state:loaded path:{init_loader_path}')
    log(f'model_params:{sum((p.numel() for p in base_model.parameters()))}')
    max_train_seq_len = max_train_seq_len_from_schedule(train_seq_plan, h.train_seq_len)
    if max_train_seq_len != h.train_seq_len:
        raise ValueError(
            f'TRAIN_SEQ_LEN={h.train_seq_len} must match the maximum sequence length in '
            f'TRAIN_SEQ_SCHEDULE ({max_train_seq_len})'
        )
    local_microbatch_tokens = validate_train_seq_plan_compatibility(
        train_seq_plan,
        global_tokens=h.train_batch_tokens,
        world_size=h.world_size,
        grad_accum_steps=h.grad_accum_steps,
    )
    log('train_seq_schedule:' + ','.join((f'{seq_len}@{threshold:.3f}' for threshold, seq_len in train_seq_plan)))
    log(f'local_microbatch_tokens:{local_microbatch_tokens}')
    if h.midrun_cap_schedule:
        log('midrun_cap_schedule:' + ','.join((f'{value:.3f}@{threshold:.3f}' for threshold, value in midrun_cap_plan)))
    if h.int8_force_patterns:
        log('int8_force_patterns:' + ','.join(h.int8_force_patterns))
    if h.loss_chunk_tokens > 0:
        log(f'loss_chunking:enabled chunk_tokens:{h.loss_chunk_tokens}')
    if (
        abs(INT6_QAT_SCALE_CLAMP_MIN - _INT6_LEGACY_SCALE_CLAMP_MIN) > 1e-12
        or abs(INT6_EXPORT_SCALE_CLAMP_MIN - _INT6_LEGACY_SCALE_CLAMP_MIN) > 1e-12
    ):
        log(
            f'int6_scale_clamp_min legacy:{_INT6_LEGACY_SCALE_CLAMP_MIN:.8g} '
            f'qat:{INT6_QAT_SCALE_CLAMP_MIN:.8g} export:{INT6_EXPORT_SCALE_CLAMP_MIN:.8g}'
        )
    if 0.0 <= h.late_qat_threshold <= 1.0:
        log(f'late_qat_threshold:{h.late_qat_threshold:.3f} mode:{h.late_qat_mode}')
    if h.ttt_lr_schedule:
        log(f'ttt_lr_schedule:{h.ttt_lr_schedule}')
    block_growth_plan = parse_scalar_schedule(h.block_growth_scale_schedule, 1.0)
    if h.block_growth_start_layer >= 0:
        log(
            f'block_growth_start_layer:{h.block_growth_start_layer} '
            + 'block_growth_scale_schedule:'
            + ','.join((f'{value:.3f}@{threshold:.3f}' for threshold, value in block_growth_plan))
        )
    if h.fsdp_enabled:
        log(
            f'fsdp:enabled sharding:{h.fsdp_sharding_strategy} '
            f'auto_wrap_blocks:{int(h.fsdp_auto_wrap_blocks)} '
            f'backward_prefetch:{h.fsdp_backward_prefetch} '
            f'use_orig_params:{int(h.fsdp_use_orig_params)} '
            f'cpu_offload:{int(h.fsdp_cpu_offload)} '
            f'forward_prefetch:{int(h.fsdp_forward_prefetch)} '
            f'limit_all_gathers:{int(h.fsdp_limit_all_gathers)} '
            f'matrix_optimizer:{"adamw" if not optimizers.uses_muon else "muon"}'
        )
    if h.activation_checkpointing:
        log(
            f'activation_checkpointing:enabled '
            f'start_layer:{h.activation_checkpoint_start_layer}'
        )
    loss_tail_plan = parse_scalar_schedule(h.loss_tail_scale_schedule, 1.0)
    if h.loss_tail_scale_schedule:
        log(
            'loss_tail_scale_schedule:' +
            ','.join((f'{value:.3f}@{threshold:.3f}' for threshold, value in loss_tail_plan)) +
            f' start_frac:{h.loss_tail_start_frac:.3f} power:{h.loss_tail_power:.3f}'
        )
    if h.optimizer_state_rewarm_on_seq_change or h.optimizer_state_rewarm_on_cap_start:
        log(
            'optimizer_state_rewarm:'
            f'seq_change:{int(h.optimizer_state_rewarm_on_seq_change)} '
            f'cap_start:{int(h.optimizer_state_rewarm_on_cap_start)} '
            f'exp_avg_scale:{h.optimizer_state_exp_avg_scale:.3f} '
            f'exp_avg_sq_scale:{h.optimizer_state_exp_avg_sq_scale:.3f} '
            f'muon_buffer_scale:{h.optimizer_state_muon_buffer_scale:.3f}'
        )
    active_train_seq_len = resume_state['active_train_seq_len']
    seq_change_warmup_start_step = resume_state['seq_change_warmup_start_step']
    midrun_cap_active = resume_state['midrun_cap_active']
    midrun_cap_prev_scale = resume_state['midrun_cap_prev_scale']
    block_growth_prev_scale = resume_state['block_growth_prev_scale']
    best_val_loss = resume_state['best_val_loss']
    best_val_bpb = resume_state['best_val_bpb']
    best_val_step = resume_state['best_val_step']
    start_looping_active = bool(resume_state['loaded_resume'] and base_model.looping_active) or bool(h.start_looping_active and h.num_loops > 0)
    if not resume_state['loaded_resume']:
        block_growth_prev_scale = schedule_value(block_growth_plan, 0.0)
        active_train_seq_len = train_seq_plan[0][1]
        base_model.set_block_growth_scale(block_growth_prev_scale)
        if start_looping_active:
            base_model.looping_active = True
            log('looping:start_active source:init_model')
        log(f'growth_stage:seq_len:{active_train_seq_len} progress:0.000')
    else:
        log(
            f'resume:state step:{resume_state["step"]} '
            f'seq_len:{active_train_seq_len} block_growth:{block_growth_prev_scale:.3f}'
        )
    max_wallclock_ms = 1000.0 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    if max_wallclock_ms is not None:
        max_wallclock_ms -= h.gptq_reserve_seconds * 1000.0
        log(f'gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms')

    def lr_mul(step, elapsed_ms, frac):
        if h.warmdown_iters > 0:
            if max_wallclock_ms is None:
                warmdown_start = max(h.iterations - h.warmdown_iters, 0)
                if warmdown_start <= step < h.iterations:
                    return max((h.iterations - step) / max(h.warmdown_iters, 1), h.min_lr)
                return 1.0
            step_ms = elapsed_ms / max(step, 1)
            warmdown_ms = h.warmdown_iters * step_ms
            remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
            if remaining_ms <= warmdown_ms:
                return max(remaining_ms / max(warmdown_ms, 1e-9), h.min_lr)
            return 1.0
        if h.warmdown_frac <= 0:
            return 1.0
        if frac >= 1.0 - h.warmdown_frac:
            return max((1.0 - frac) / h.warmdown_frac, h.min_lr)
        return 1.0

    def step_fn(step, lr_scale, loss_weights=None):
        optimizers.zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(h.grad_accum_steps):
            if h.distributed and (not h.fsdp_enabled):
                model.require_backward_grad_sync = micro_step == h.grad_accum_steps - 1
            x, y = train_loader.next_batch(h.train_batch_tokens, active_train_seq_len, h.grad_accum_steps)
            sync_context = nullcontext()
            if h.fsdp_enabled and micro_step != h.grad_accum_steps - 1:
                sync_context = model.no_sync()
            with sync_context:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                    loss = model(x, y, loss_weights=loss_weights)
            train_loss += loss.detach()
            (loss / h.grad_accum_steps).backward()
        train_loss /= h.grad_accum_steps
        frac = min(step / h.muon_momentum_warmup_steps, 1.0) if h.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * h.muon_momentum_warmup_start + frac * h.muon_momentum
        optimizers.set_matrix_momentum(muon_momentum)
        for opt in optimizers:
            for group in opt.param_groups:
                group['lr'] = group['base_lr'] * lr_scale
        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
        optimizers.step()
        return train_loss

    def maybe_damp_optimizer_state(reason, step, frac, *, seq_len=None, cap_scale=None):
        if step <= 0:
            return
        if (
            abs(h.optimizer_state_exp_avg_scale - 1.0) < 1e-12
            and abs(h.optimizer_state_exp_avg_sq_scale - 1.0) < 1e-12
            and abs(h.optimizer_state_muon_buffer_scale - 1.0) < 1e-12
        ):
            return
        adam_m_buffers, adam_v_buffers, adam_vmax_buffers, muon_buffers = optimizers.damp_state(
            exp_avg_scale=h.optimizer_state_exp_avg_scale,
            exp_avg_sq_scale=h.optimizer_state_exp_avg_sq_scale,
            muon_buffer_scale=h.optimizer_state_muon_buffer_scale,
        )
        details = []
        if seq_len is not None:
            details.append(f'seq_len:{seq_len}')
        if cap_scale is not None:
            details.append(f'cap_scale:{cap_scale:.3f}')
        details_str = (' ' + ' '.join(details)) if details else ''
        log(
            f'optimizer_state_rewarm:event:{reason} step:{step} progress:{frac:.3f}'
            f'{details_str} exp_avg_scale:{h.optimizer_state_exp_avg_scale:.3f} '
            f'exp_avg_sq_scale:{h.optimizer_state_exp_avg_sq_scale:.3f} '
            f'muon_buffer_scale:{h.optimizer_state_muon_buffer_scale:.3f} '
            f'adam_m_buffers:{adam_m_buffers} adam_v_buffers:{adam_v_buffers} '
            f'adam_vmax_buffers:{adam_vmax_buffers} muon_buffers:{muon_buffers}'
        )

    if h.warmup_steps > 0 and not resume_state['loaded_resume']:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(h.warmup_steps):
            step_fn(warmup_step, 1.0)
            if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                log(f'warmup_step: {warmup_step + 1}/{h.warmup_steps}')
        if h.num_loops > 0:
            base_model.looping_active = True
            log(f'loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}')
            for warmup_step in range(h.warmup_steps):
                step_fn(warmup_step, 1.0)
                if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                    log(f'loop_warmup_step: {warmup_step + 1}/{h.warmup_steps}')
            base_model.looping_active = False
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        optimizers.zero_grad_all()
        if h.distributed and (not h.fsdp_enabled):
            model.require_backward_grad_sync = True
        train_loader = ShuffledSequenceLoader(h, device)
    if init_train_loader_state is not None:
        train_loader.load_state_dict(init_train_loader_state)
        remaining_sequences = sum((len(indices) for indices in train_loader.start_inds))
        log(
            f'init_train_loader_state:restored seq_len:{train_loader.current_seq_len} '
            f'remaining_sequences:{remaining_sequences}'
        )
    if start_looping_active:
        base_model.looping_active = True
    ema_state = (
        {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
        if h.ema_enabled else None
    )
    if resume_state['ema_state'] is not None:
        ema_state = resume_state['ema_state']
    ema_decay = h.ema_decay
    training_time_ms = resume_state['training_time_ms']
    stop_after_step = None
    base_model.set_block_growth_scale(block_growth_prev_scale)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = resume_state['step']
    while True:
        last_step = step == h.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (h.val_loss_every > 0 and step % h.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(h, device, val_data, model)
            log(f'{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}')
            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                best_val_bpb = float(val_bpb)
                best_val_step = int(step)
                best_model_state = _full_model_state_dict(h, compiled_model if h.fsdp_enabled else base_model, base_model)
                if h.is_main_process:
                    _safe_torch_save(best_model_state, h.best_model_path)
                    _safe_write_json(
                        {
                            'run_id': h.run_id,
                            'step': best_val_step,
                            'val_loss': best_val_loss,
                            'val_bpb': best_val_bpb,
                            'training_time_ms': float(training_time_ms),
                        },
                        h.best_model_metadata_path,
                    )
                    log(
                        f'best_model:saved path:{h.best_model_path} step:{best_val_step} '
                        f'val_loss:{best_val_loss:.8f} val_bpb:{best_val_bpb:.8f}'
                    )
            reset_runtime_caches(base_model)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            save_training_checkpoint(
                h,
                compiled_model if h.fsdp_enabled else base_model,
                base_model,
                optimizers,
                train_loader,
                step=step,
                training_time_ms=training_time_ms,
                active_train_seq_len=active_train_seq_len,
                seq_change_warmup_start_step=seq_change_warmup_start_step,
                midrun_cap_active=midrun_cap_active,
                midrun_cap_prev_scale=midrun_cap_prev_scale,
                block_growth_prev_scale=block_growth_prev_scale,
                best_val_loss=best_val_loss,
                best_val_bpb=best_val_bpb,
                best_val_step=best_val_step,
                ema_state=ema_state,
                reason='wallclock_cap' if stop_after_step is not None and step < h.iterations else 'train_complete',
            )
            if stop_after_step is not None and step < h.iterations:
                log(f'stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms step: {step}/{h.iterations}')
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        stage_seq_len, frac = current_train_seq_len(
            train_seq_plan,
            step=step,
            iterations=h.iterations,
            elapsed_ms=elapsed_ms,
            max_wallclock_ms=max_wallclock_ms,
            schedule_mode=h.train_seq_schedule_mode,
        )
        block_growth_scale_now = schedule_value(block_growth_plan, frac)
        base_model.set_block_growth_scale(block_growth_scale_now)
        if (
            h.block_growth_log_updates
            and abs(block_growth_scale_now - block_growth_prev_scale) > 1e-6
        ):
            log(
                f'block_growth:update step:{step} progress:{frac:.3f} '
                f'scale:{block_growth_prev_scale:.3f}->{block_growth_scale_now:.3f}'
            )
        block_growth_prev_scale = block_growth_scale_now
        if stage_seq_len != active_train_seq_len:
            active_train_seq_len = stage_seq_len
            log(f'growth_stage:seq_len:{active_train_seq_len} progress:{frac:.3f} step:{step}')
            if h.optimizer_state_rewarm_on_seq_change:
                maybe_damp_optimizer_state('seq_change', step, frac, seq_len=active_train_seq_len)
            if h.seq_change_warmup_steps > 0 and step > 0:
                seq_change_warmup_start_step = step
                log(
                    f'growth_stage_rewarmup:start step:{step} steps:{h.seq_change_warmup_steps} '
                    f'seq_len:{active_train_seq_len}'
                )
        base_scale = lr_mul(step, elapsed_ms, frac)
        scale = base_scale
        cap_scale_now = schedule_value(midrun_cap_plan, frac)
        midrun_cap_now = cap_scale_now < 0.999999
        if midrun_cap_now and not midrun_cap_active:
            log(f'midrun_cap:start step:{step} progress:{frac:.3f} scale:{cap_scale_now:.3f}')
            if h.optimizer_state_rewarm_on_cap_start:
                maybe_damp_optimizer_state('cap_start', step, frac, cap_scale=cap_scale_now)
        if (
            h.midrun_cap_log_updates
            and midrun_cap_now
            and midrun_cap_active
            and abs(cap_scale_now - midrun_cap_prev_scale) > 1e-6
        ):
            log(
                f'midrun_cap:update step:{step} progress:{frac:.3f} '
                f'scale:{midrun_cap_prev_scale:.3f}->{cap_scale_now:.3f}'
            )
        if midrun_cap_now:
            scale *= cap_scale_now
        midrun_cap_active = midrun_cap_now
        midrun_cap_prev_scale = cap_scale_now
        if seq_change_warmup_start_step is not None and h.seq_change_warmup_steps > 0:
            rewarm_progress = min(
                max((step - seq_change_warmup_start_step + 1) / max(h.seq_change_warmup_steps, 1), 0.0),
                1.0,
            )
            scale *= rewarm_progress
            if rewarm_progress >= 1.0:
                seq_change_warmup_start_step = None
        if h.num_loops > 0 and (not base_model.looping_active) and (frac >= h.enable_looping_at):
            base_model.looping_active = True
            log(f'layer_loop:enabled step:{step} frac:{frac:.3f} encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}')
        late_qat_scale = base_scale * cap_scale_now if h.late_qat_mode == 'structural' else scale
        if (
            h.late_qat_threshold > 0.0
            and late_qat_scale < h.late_qat_threshold
            and not CastedLinear._qat_enabled
        ):
            CastedLinear._qat_enabled = True
            log(f'late_qat:enabled step:{step} scale:{late_qat_scale:.4f} mode:{h.late_qat_mode}')
        loss_tail_scale = schedule_value(loss_tail_plan, frac)
        loss_weights = build_tail_loss_weights(
            active_train_seq_len,
            h.loss_tail_start_frac,
            loss_tail_scale,
            h.loss_tail_power,
            device,
        )
        train_loss = step_fn(step, scale, loss_weights=loss_weights)
        if ema_state is not None:
            with torch.no_grad():
                for name, t in base_model.state_dict().items():
                    ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = h.train_log_every > 0 and (step <= 5 or step % h.train_log_every == 0 or stop_after_step is not None)
        if should_log_train:
            tok_per_sec = step * h.train_batch_tokens / (approx_training_time_ms / 1000.0)
            log(f'{step}/{h.iterations} train_loss: {train_loss.item():.4f} train_time: {approx_training_time_ms / 60000:.1f}m tok/s: {tok_per_sec:.0f}')
        if h.save_checkpoint_every > 0 and step > 0 and step % h.save_checkpoint_every == 0:
            save_training_checkpoint(
                h,
                compiled_model if h.fsdp_enabled else base_model,
                base_model,
                optimizers,
                train_loader,
                step=step,
                training_time_ms=approx_training_time_ms,
                active_train_seq_len=active_train_seq_len,
                seq_change_warmup_start_step=seq_change_warmup_start_step,
                midrun_cap_active=midrun_cap_active,
                midrun_cap_prev_scale=midrun_cap_prev_scale,
                block_growth_prev_scale=block_growth_prev_scale,
                best_val_loss=best_val_loss,
                best_val_bpb=best_val_bpb,
                best_val_step=best_val_step,
                ema_state=ema_state,
                reason='periodic',
            )
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if h.distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            save_training_checkpoint(
                h,
                base_model,
                optimizers,
                train_loader,
                step=step,
                training_time_ms=approx_training_time_ms,
                active_train_seq_len=active_train_seq_len,
                seq_change_warmup_start_step=seq_change_warmup_start_step,
                midrun_cap_active=midrun_cap_active,
                midrun_cap_prev_scale=midrun_cap_prev_scale,
                block_growth_prev_scale=block_growth_prev_scale,
                best_val_loss=best_val_loss,
                best_val_bpb=best_val_bpb,
                best_val_step=best_val_step,
                ema_state=ema_state,
                reason='wallclock_cap',
            )
            stop_after_step = step
    log(f'peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB')
    if ema_state is not None:
        log('ema:applying EMA weights')
        current_state = base_model.state_dict()
        avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
        base_model.load_state_dict(avg_state, strict=True)
    else:
        log('averaging:none keeping current weights')
    return (base_model, compiled_model)

def train_and_eval(h, device):
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    val_data = ValidationData(h, device)
    log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}")
    log(f'val_tokens: {val_data.val_tokens.numel() - 1}')
    if h.val_max_tokens > 0:
        log(f'val_tokens_cap: {h.val_max_tokens}')
    if h.eval_only:
        if h.fsdp_enabled:
            raise NotImplementedError('FSDP eval_only is not supported yet')
        log(f'eval_only:enabled quantized_model_path:{h.quantized_model_path}')
        run_quantized_evals(h, device, val_data)
        return
    if h.package_only:
        if h.fsdp_enabled:
            raise NotImplementedError('FSDP package_only is not supported yet')
        log(f'package_only:enabled model_path:{h.model_path} quantized_model_path:{h.quantized_model_path}')
        if h.int8_force_patterns:
            log('int8_force_patterns:' + ','.join(h.int8_force_patterns))
        base_model = GPT(h).to(device).bfloat16()
        restore_fp32_params(base_model)
        state_dict = torch.load(h.model_path, map_location='cpu')
        base_model.load_state_dict(state_dict, strict=True)
        if h.num_loops > 0:
            base_model.looping_active = True
        if h.save_pre_quant_snapshot and h.is_main_process:
            snapshot_run_file(h.model_path, h.run_id, 'final_model_snapshot.pt')
        serialize(h, base_model, Path(__file__).read_text(encoding='utf-8'))
        if h.save_quantized_snapshot and h.is_main_process:
            snapshot_run_file(h.quantized_model_path, h.run_id, 'quantized_model_snapshot.int6.ptz')
        if h.distributed:
            dist.barrier()
        run_quantized_evals(h, device, val_data)
        return
    base_model, compiled_model = train_model(h, device, val_data)
    torch._dynamo.reset()
    pre_val_loss, pre_val_bpb = timed_eval('pre-quantization post-ema', eval_val, h, device, val_data, compiled_model)
    if h.fsdp_enabled:
        final_state = _full_model_state_dict(h, compiled_model, base_model)
        if h.is_main_process:
            torch.save(final_state, h.model_path)
            if h.save_pre_quant_snapshot:
                snapshot_run_file(h.model_path, h.run_id, 'final_model_snapshot.pt')
        log(
            f'final_full_state_dict_exact val_loss:{pre_val_loss:.8f} '
            f'val_bpb:{pre_val_bpb:.8f} fsdp_only:1'
        )
        return
    if h.save_pre_quant_snapshot and h.is_main_process:
        torch.save(base_model.state_dict(), h.model_path)
        snapshot_run_file(h.model_path, h.run_id, 'final_model_snapshot.pt')
    if h.skip_final_packaging:
        log('skip_final_packaging:enabled')
        log(
            f'final_int6_roundtrip_exact val_loss:{pre_val_loss:.8f} '
            f'val_bpb:{pre_val_bpb:.8f} skipped_packaging:1'
        )
        return
    serialize(h, base_model, Path(__file__).read_text(encoding='utf-8'))
    if h.save_quantized_snapshot and h.is_main_process:
        snapshot_run_file(h.quantized_model_path, h.run_id, 'quantized_model_snapshot.int6.ptz')
    if h.distributed:
        dist.barrier()
    run_quantized_evals(h, device, val_data)

def main():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if world_size <= 0:
        raise ValueError(f'WORLD_SIZE must be positive, got {world_size}')
    if 8 % world_size != 0:
        raise ValueError(f'WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral')
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend='nccl', device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False
    h = Hyperparameters()
    torch._dynamo.config.cache_size_limit = h.dynamo_cache_size_limit
    set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs('logs', exist_ok=True)
        Path(h.run_dir).mkdir(parents=True, exist_ok=True)
        Path(h.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(h.logfile).parent.mkdir(parents=True, exist_ok=True)
        log(100 * '=', console=False)
        log('Hyperparameters:', console=True)
        for k, v in sorted(vars(type(h)).items()):
            if not k.startswith('_'):
                log(f'  {k}: {v}', console=True)
        log('=' * 100, console=False)
        log(f'Running Python {sys.version}', console=False)
        log(f'Running PyTorch {torch.__version__}', console=False)
        log(subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
        log('=' * 100, console=False)
    train_and_eval(h, device)
    if distributed:
        dist.destroy_process_group()
if __name__ == '__main__':
    main()
