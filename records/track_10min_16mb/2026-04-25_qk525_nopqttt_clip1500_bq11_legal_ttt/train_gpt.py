import collections, copy, glob, hashlib, io, json, lzma, math, os, shlex, socket, traceback
from pathlib import Path
import random, re, subprocess, sys, time, uuid, numpy as np, sentencepiece as spm, torch, torch.distributed as dist, torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor, nn
from flash_attn_interface import flash_attn_func as flash_attn_3_func

FINAL_EVAL_MODES = {'checkpoint', 'quant', 'official'}
FINAL_SCORE_MODES = {'full', 'legal_ttt_only'}
CHECKPOINT_METRIC_PRIORITY = ('checkpoint_val_bpb',)
QUANT_METRIC_PRIORITY = ('ttt_val_bpb', 'sliding_val_bpb', 'post_quant_val_bpb', 'pre_quant_val_bpb')


def _parse_step_set(spec):
    out = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _parse_eval_steps(spec):
    steps = set()
    for part in spec.split(','):
        part = part.strip()
        if not part or part.lower() == 'final':
            continue
        steps.add(int(part))
    return steps


def _get_env_int_with_alias(primary, fallback, default):
    value = os.environ.get(primary)
    if value is not None:
        return int(value)
    return int(os.environ.get(fallback, default))


def _default_dataset_name(vocab_size, tokenizer_path):
    if tokenizer_path:
        path = Path(tokenizer_path)
        if path.suffix == '.json' and path.is_file():
            payload = json.loads(path.read_text(encoding='utf-8'))
            if payload.get('tokenizer_type') == 'pure_byte':
                return f"fineweb10B_byte{int(payload.get('vocab_size', vocab_size))}"
            dataset_suffix = payload.get('dataset_suffix')
            if dataset_suffix:
                return f'fineweb10B_{dataset_suffix}'
    return f'fineweb10B_sp{vocab_size}'


def _load_tokenizer_json(path):
    path = Path(path)
    if path.suffix != '.json' or not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def _default_val_byte_files(tokenizer_path, datasets_dir):
    payload = _load_tokenizer_json(tokenizer_path)
    if payload is None:
        return ''
    if payload.get('val_byte_glob'):
        val_byte_glob = payload['val_byte_glob']
        return str((Path(tokenizer_path).parent / val_byte_glob).resolve()) if not Path(val_byte_glob).is_absolute() else str(Path(val_byte_glob))
    if payload.get('requires_byte_sidecar') or payload.get('caseops_sidecar_utf8_bytes') is not None or payload.get('caseops_zero_byte_token_count') is not None:
        return os.path.join(datasets_dir, 'fineweb_val_bytes_*.bin')
    return ''


def _sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _sha256_file(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values, q):
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _shard_name(file):
    match = re.search(r'_(\d+)\.bin$', str(file))
    return match.group(1) if match is not None else Path(file).stem


def _glob_data_files(pattern, shard_limit=0):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if shard_limit > 0:
        files = files[:shard_limit]
    return files


def _json_dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_json_safe(v) for v in sorted(value)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _git_value(*args):
    proc = subprocess.run(['git', *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _collect_git_state():
    git_hash = _git_value('rev-parse', 'HEAD')
    dirty = _git_value('status', '--porcelain')
    return {
        'git_hash': git_hash,
        'git_dirty': bool(dirty),
    }


class RunFailure(RuntimeError):

    def __init__(self, reason, **details):
        super().__init__(reason)
        self.reason = reason
        self.details = details


def _quant_pipeline_enabled(h):
    return h.final_eval_mode in {'quant', 'official'}


def _post_quant_eval_enabled(h):
    return _quant_pipeline_enabled(h) and h.post_quant_eval_enabled and h.final_score_mode != 'legal_ttt_only'


def _sliding_eval_enabled(h):
    if h.final_score_mode == 'legal_ttt_only':
        return False
    if h.final_eval_mode == 'checkpoint':
        return False
    if h.final_eval_mode == 'official':
        return True
    return h.sliding_window_enabled


def _ttt_eval_enabled(h):
    if h.final_score_mode == 'legal_ttt_only':
        return h.final_eval_mode in {'quant', 'official'} and h.ttt_enabled
    if h.final_eval_mode == 'official':
        return True
    return h.ttt_enabled


def _comparison_metric_priority(h):
    if h.final_eval_mode == 'checkpoint':
        return list(CHECKPOINT_METRIC_PRIORITY)
    return list(QUANT_METRIC_PRIORITY)

class Hyperparameters:
    def __init__(self):
        self.data_dir = os.environ.get('DATA_DIR', './data/')
        self.seed = int(os.environ.get('SEED', 1337))
        self.run_id = os.environ.get('RUN_ID', str(uuid.uuid4()))
        self.stage = os.environ.get('STAGE', 'adhoc')
        self.method = os.environ.get('METHOD', 'manual')
        self.baseline_method = os.environ.get('BASELINE_METHOD', '')
        self.run_group = os.environ.get('RUN_GROUP', self.run_id)
        self.budget_id = os.environ.get('BUDGET_ID', self.stage)
        self.run_dir = os.environ.get('RUN_DIR', os.path.join('runs', self.run_id))
        self.manifest_path = os.environ.get('MANIFEST_PATH', os.path.join(self.run_dir, 'manifest.json'))
        self.metrics_path = os.environ.get('METRICS_PATH', os.path.join(self.run_dir, 'metrics.jsonl'))
        self.summary_path = os.environ.get('SUMMARY_PATH', os.path.join(self.run_dir, 'summary.json'))
        self.iterations = int(os.environ.get('ITERATIONS', 20000))
        self.warmdown_frac = float(os.environ.get('WARMDOWN_FRAC', 0.72))
        self.warmup_steps = int(os.environ.get('WARMUP_STEPS', 20))
        self.train_batch_tokens = int(os.environ.get('TRAIN_BATCH_TOKENS', 786432))
        self.train_seq_len = int(os.environ.get('TRAIN_SEQ_LEN', 2048))
        self.train_log_every = int(os.environ.get('TRAIN_LOG_EVERY', 500))
        self.max_wallclock_seconds = float(os.environ.get('MAX_WALLCLOCK_SECONDS', 600.0))
        self.val_batch_tokens = int(os.environ.get('VAL_BATCH_TOKENS', 524288))
        self.eval_seq_len = int(os.environ.get('EVAL_SEQ_LEN', 2048))
        self.val_loss_every = int(os.environ.get('VAL_LOSS_EVERY', 4000))
        self.eval_at_steps = _parse_eval_steps(os.environ.get('EVAL_AT_STEPS', ''))
        self.local_val_tokens = _get_env_int_with_alias('VAL_TOKEN_LIMIT', 'LOCAL_VAL_TOKENS', '0')
        self.local_val_offset_tokens = _get_env_int_with_alias('VAL_TOKEN_OFFSET', 'LOCAL_VAL_OFFSET_TOKENS', '0')
        self.final_eval_mode = os.environ.get('FINAL_EVAL_MODE', 'official').strip().lower()
        if self.final_eval_mode not in FINAL_EVAL_MODES:
            raise ValueError(f'FINAL_EVAL_MODE must be one of {sorted(FINAL_EVAL_MODES)}, got {self.final_eval_mode!r}')
        self.final_score_mode = os.environ.get('FINAL_SCORE_MODE', 'full').strip().lower()
        if self.final_score_mode not in FINAL_SCORE_MODES:
            raise ValueError(f'FINAL_SCORE_MODE must be one of {sorted(FINAL_SCORE_MODES)}, got {self.final_score_mode!r}')
        self.train_shard_limit = int(os.environ.get('TRAIN_SHARD_LIMIT', '0'))
        self.sliding_window_enabled = bool(int(os.environ.get('SLIDING_WINDOW_ENABLED', '1')))
        self.post_quant_eval_enabled = bool(int(os.environ.get('POST_QUANT_EVAL_ENABLED', '1')))
        self.vocab_size = int(os.environ.get('VOCAB_SIZE', 8192))
        self.num_layers = int(os.environ.get('NUM_LAYERS', 11))
        self.xsa_last_n = int(os.environ.get('XSA_LAST_N', 11))
        self.model_dim = int(os.environ.get('MODEL_DIM', 512))
        self.embedding_dim = int(os.environ.get('EMBEDDING_DIM', 512))
        self.num_kv_heads = int(os.environ.get('NUM_KV_HEADS', 4))
        self.num_heads = int(os.environ.get('NUM_HEADS', 8))
        self.mlp_mult = float(os.environ.get('MLP_MULT', 4.0))
        self.skip_gates_enabled = bool(int(os.environ.get('SKIP_GATES_ENABLED', '1')))
        self.tie_embeddings = bool(int(os.environ.get('TIE_EMBEDDINGS', '1')))
        self.logit_softcap = float(os.environ.get('LOGIT_SOFTCAP', 30.0))
        self.rope_base = float(os.environ.get('ROPE_BASE', 10000.0))
        self.rope_dims = int(os.environ.get('ROPE_DIMS', 16))
        self.rope_train_seq_len = int(os.environ.get('ROPE_TRAIN_SEQ_LEN', 2048))
        self.ln_scale = bool(int(os.environ.get('LN_SCALE', '1')))
        self.qk_gain_init = float(os.environ.get('QK_GAIN_INIT', 5.0))
        self.num_loops = int(os.environ.get('NUM_LOOPS', 2))
        self.loop_start = int(os.environ.get('LOOP_START', 3))
        self.loop_end = int(os.environ.get('LOOP_END', 5))
        self.enable_looping_at = float(os.environ.get('ENABLE_LOOPING_AT', 0.35))
        self.parallel_residual_start = int(os.environ.get('PARALLEL_RESIDUAL_START', 7))
        self.min_lr = float(os.environ.get('MIN_LR', 0.0))
        self.embed_lr = float(os.environ.get('EMBED_LR', 0.6))
        self.head_lr = float(os.environ.get('HEAD_LR', 0.008))
        self.tied_embed_lr = float(os.environ.get('TIED_EMBED_LR', 0.03))
        self.tied_embed_init_std = float(os.environ.get('TIED_EMBED_INIT_STD', 0.005))
        self.matrix_lr = float(os.environ.get('MATRIX_LR', 0.022))
        self.scalar_lr = float(os.environ.get('SCALAR_LR', 0.02))
        self.muon_momentum = float(os.environ.get('MUON_MOMENTUM', 0.99))
        self.muon_backend_steps = int(os.environ.get('MUON_BACKEND_STEPS', 5))
        self.muon_momentum_warmup_start = float(os.environ.get('MUON_MOMENTUM_WARMUP_START', 0.92))
        self.muon_momentum_warmup_steps = int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS', 1500))
        self.muon_row_normalize = bool(int(os.environ.get('MUON_ROW_NORMALIZE', '1')))
        self.beta1 = float(os.environ.get('BETA1', 0.9))
        self.beta2 = float(os.environ.get('BETA2', 0.95))
        self.adam_eps = float(os.environ.get('ADAM_EPS', 1e-08))
        self.grad_clip_norm = float(os.environ.get('GRAD_CLIP_NORM', 0.3))
        self.eval_stride = int(os.environ.get('EVAL_STRIDE', 64))
        self.muon_beta2 = float(os.environ.get('MUON_BETA2', 0.95))
        self.adam_wd = float(os.environ.get('ADAM_WD', 0.02))
        self.muon_wd = float(os.environ.get('MUON_WD', 0.095))
        self.embed_wd = float(os.environ.get('EMBED_WD', 0.085))
        self.ema_enabled = bool(int(os.environ.get('EMA_ENABLED', '1')))
        self.ema_decay = float(os.environ.get('EMA_DECAY', 0.9965))
        self.ttt_enabled = bool(int(os.environ.get('TTT_ENABLED', '0')))
        self.ttt_lr = float(os.environ.get('TTT_LR', 0.005))
        self.ttt_epochs = int(os.environ.get('TTT_EPOCHS', 3))
        self.ttt_momentum = float(os.environ.get('TTT_MOMENTUM', 0.9))
        self.ttt_chunk_tokens = int(os.environ.get('TTT_CHUNK_TOKENS', 32768))
        self.pre_quant_ttt_enabled = bool(int(os.environ.get('PRE_QUANT_TTT_ENABLED', '0')))
        self.pre_quant_ttt_lr = float(os.environ.get('PRE_QUANT_TTT_LR', 0.001))
        self.pre_quant_ttt_min_lr = float(os.environ.get('PRE_QUANT_TTT_MIN_LR', 0.0001))
        self.pre_quant_ttt_epochs = int(os.environ.get('PRE_QUANT_TTT_EPOCHS', '21'))
        self.pre_quant_ttt_weight_decay = float(os.environ.get('PRE_QUANT_TTT_WEIGHT_DECAY', '0.0'))
        self.pre_quant_ttt_grad_clip_norm = float(os.environ.get('PRE_QUANT_TTT_GRAD_CLIP_NORM', '1.0'))
        self.pre_quant_ttt_freeze_blocks = int(os.environ.get('PRE_QUANT_TTT_FREEZE_BLOCKS', '0'))
        self.pre_quant_ttt_token_limit = int(os.environ.get('PRE_QUANT_TTT_TOKEN_LIMIT', str(min((self.local_val_tokens if self.local_val_tokens > 0 else 262144), 262144))))
        pre_quant_ttt_token_offset = os.environ.get('PRE_QUANT_TTT_TOKEN_OFFSET', '')
        self.pre_quant_ttt_token_offset = (None if pre_quant_ttt_token_offset == '' else int(pre_quant_ttt_token_offset))
        self.pre_quant_ttt_source = os.environ.get('PRE_QUANT_TTT_SOURCE', 'val').strip().lower()
        self.pre_quant_ttt_selection_mode = os.environ.get('PRE_QUANT_TTT_SELECTION_MODE', 'contiguous').strip().lower()
        self.pre_quant_ttt_chunk_tokens = int(os.environ.get('PRE_QUANT_TTT_CHUNK_TOKENS', '32768'))
        self.pre_quant_ttt_shard_limit = int(os.environ.get('PRE_QUANT_TTT_SHARD_LIMIT', str(self.train_shard_limit)))
        self.pre_quant_ttt_loop_mode = os.environ.get('PRE_QUANT_TTT_LOOP_MODE', 'inactive').strip().lower()
        self.pre_quant_ttt_anchor_lambda = float(os.environ.get('PRE_QUANT_TTT_ANCHOR_LAMBDA', '0.0'))
        self.pre_quant_ttt_qnoise_enabled = bool(int(os.environ.get('PRE_QUANT_TTT_QNOISE_ENABLED', '0')))
        self.pre_quant_ttt_qnoise_start_epoch = int(os.environ.get('PRE_QUANT_TTT_QNOISE_START_EPOCH', '2'))
        self.pre_quant_ttt_qnoise_prob = float(os.environ.get('PRE_QUANT_TTT_QNOISE_PROB', '0.5'))
        self.pre_quant_ttt_qnoise_scale_mult = float(os.environ.get('PRE_QUANT_TTT_QNOISE_SCALE_MULT', '0.5'))
        self.save_post_ttt_checkpoint = bool(int(os.environ.get('SAVE_POST_TTT_CHECKPOINT', '0')))
        self.post_ttt_checkpoint_path = os.environ.get('POST_TTT_CHECKPOINT_PATH', os.path.join(self.run_dir, 'post_ttt_model.pt'))
        self.quant_only_from_post_ttt = bool(int(os.environ.get('QUANT_ONLY_FROM_POST_TTT', '0')))
        self.wrapper_code_bytes_estimate = int(os.environ.get('WRAPPER_CODE_BYTES_ESTIMATE', '20000'))
        self.etlb_enabled = bool(int(os.environ.get('ETLB_ENABLED', '0')))
        self.etlb_lr = float(os.environ.get('ETLB_LR', 0.05))
        self.etlb_steps = int(os.environ.get('ETLB_STEPS', 5))
        self.etlb_clip = float(os.environ.get('ETLB_CLIP', 3.0))
        self.compressor = os.environ.get('COMPRESSOR', 'brotli')
        self.brotli_quality = int(os.environ.get('BROTLI_QUALITY', '11'))
        if not 0 <= self.brotli_quality <= 11:
            raise ValueError(f'BROTLI_QUALITY must be between 0 and 11, got {self.brotli_quality}')
        self.artifact_rank0_only = bool(int(os.environ.get('ARTIFACT_RANK0_ONLY', '0')))
        self.gptq_calibration_batches = int(os.environ.get('GPTQ_CALIBRATION_BATCHES', 64))
        self.gptq_reserve_seconds = float(os.environ.get('GPTQ_RESERVE_SECONDS', 12.0))
        self.matrix_bits = int(os.environ.get('MATRIX_BITS', 6))
        self.embed_bits = int(os.environ.get('EMBED_BITS', 8))
        self.matrix_clip_sigmas = float(os.environ.get('MATRIX_CLIP_SIGMAS', 12.85))
        self.embed_clip_sigmas = float(os.environ.get('EMBED_CLIP_SIGMAS', 20.0))
        self.distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
        self.rank = int(os.environ.get('RANK', '0'))
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        self.perf_enabled = bool(int(os.environ.get('PERF_ENABLED', '0')))
        self.perf_log_every = int(os.environ.get('PERF_LOG_EVERY', '100'))
        self.perf_sample_every = int(os.environ.get('PERF_SAMPLE_EVERY', '10'))
        self.perf_window_size = int(os.environ.get('PERF_WINDOW_SIZE', '32'))
        self.perf_ranks = os.environ.get('PERF_RANKS', '0')
        self.perf_log_dir = os.environ.get('PERF_LOG_DIR', os.path.join('logs', self.run_id))
        self.perf_force_steps = os.environ.get('PERF_FORCE_STEPS', '')
        self.perf_loop_dense_before = int(os.environ.get('PERF_LOOP_DENSE_BEFORE', '200'))
        self.perf_loop_dense_after = int(os.environ.get('PERF_LOOP_DENSE_AFTER', '200'))
        self.perf_loop_dense_every = int(os.environ.get('PERF_LOOP_DENSE_EVERY', '20'))
        self.perf_ttt_chunk_every = int(os.environ.get('PERF_TTT_CHUNK_EVERY', '20'))
        self.profiler_enabled = bool(int(os.environ.get('PROFILER_ENABLED', '0')))
        self.profiler_ranks = os.environ.get('PROFILER_RANKS', '0')
        self.profiler_steps = os.environ.get('PROFILER_STEPS', '')
        self.profiler_warmup_steps = int(os.environ.get('PROFILER_WARMUP_STEPS', '2'))
        self.profiler_active_steps = int(os.environ.get('PROFILER_ACTIVE_STEPS', '10'))
        self.profiler_record_shapes = bool(int(os.environ.get('PROFILER_RECORD_SHAPES', '1')))
        self.profiler_profile_memory = bool(int(os.environ.get('PROFILER_PROFILE_MEMORY', '1')))
        self.profiler_with_flops = bool(int(os.environ.get('PROFILER_WITH_FLOPS', '1')))
        self.profiler_with_stack = bool(int(os.environ.get('PROFILER_WITH_STACK', '0')))
        self.abort_on_nan = bool(int(os.environ.get('ABORT_ON_NAN', '1')))
        self.max_peak_mem_mib = int(os.environ.get('MAX_PEAK_MEM_MIB', '0'))
        self.max_artifact_bytes = int(os.environ.get('MAX_ARTIFACT_BYTES', '0'))
        self.min_tokens_per_sec = float(os.environ.get('MIN_TOKENS_PER_SEC', '0'))
        self.max_quant_bpb_gap = float(os.environ.get('MAX_QUANT_BPB_GAP', '0'))
        self.is_main_process = self.rank == 0
        self.grad_accum_steps = 8 // self.world_size
        default_tokenizer_path = os.environ.get('TOKENIZER_PATH', os.path.join(self.data_dir, 'tokenizers', f'fineweb_{self.vocab_size}_bpe.model'))
        self.dataset_name = os.environ.get('DATASET_NAME', _default_dataset_name(self.vocab_size, default_tokenizer_path))
        self.datasets_dir = os.environ.get('DATASETS_DIR', os.path.join(self.data_dir, 'datasets', self.dataset_name))
        self.train_files = os.environ.get('TRAIN_FILES', os.path.join(self.datasets_dir, 'fineweb_train_*.bin'))
        self.val_files = os.environ.get('VAL_FILES', os.path.join(self.datasets_dir, 'fineweb_val_*.bin'))
        self.val_byte_files = os.environ.get('VAL_BYTE_FILES', _default_val_byte_files(default_tokenizer_path, self.datasets_dir)).strip()
        if self.pre_quant_ttt_source not in {'val', 'train'}:
            raise ValueError(f'PRE_QUANT_TTT_SOURCE must be one of val/train, got {self.pre_quant_ttt_source!r}')
        if self.pre_quant_ttt_selection_mode not in {'contiguous', 'stratified'}:
            raise ValueError(f'PRE_QUANT_TTT_SELECTION_MODE must be one of contiguous/stratified, got {self.pre_quant_ttt_selection_mode!r}')
        if self.pre_quant_ttt_loop_mode not in {'inactive', 'active'}:
            raise ValueError(f'PRE_QUANT_TTT_LOOP_MODE must be one of inactive/active, got {self.pre_quant_ttt_loop_mode!r}')
        if self.pre_quant_ttt_chunk_tokens <= 0:
            raise ValueError(f'PRE_QUANT_TTT_CHUNK_TOKENS must be positive, got {self.pre_quant_ttt_chunk_tokens}')
        if not (0.0 <= self.pre_quant_ttt_qnoise_prob <= 1.0):
            raise ValueError(f'PRE_QUANT_TTT_QNOISE_PROB must be in [0,1], got {self.pre_quant_ttt_qnoise_prob}')
        if self.quant_only_from_post_ttt and not Path(self.post_ttt_checkpoint_path).exists():
            raise FileNotFoundError(f'QUANT_ONLY_FROM_POST_TTT requires existing POST_TTT_CHECKPOINT_PATH, got {self.post_ttt_checkpoint_path}')
        pre_quant_ttt_files = os.environ.get('PRE_QUANT_TTT_FILES', '').strip()
        if not pre_quant_ttt_files:
            pre_quant_ttt_files = (self.train_files if self.pre_quant_ttt_source == 'train' else self.val_files)
        self.pre_quant_ttt_files = pre_quant_ttt_files
        self.tokenizer_path = default_tokenizer_path
        self.logfile = os.path.join(self.run_dir, 'train.log')
        self.perf_log_dir = os.environ.get('PERF_LOG_DIR', os.path.join(self.run_dir, 'perf'))
        self.perf_logfile = os.path.join(self.perf_log_dir, f'perf_rank{self.rank}.jsonl')
        self.profiler_dir = os.path.join(self.perf_log_dir, f'profiler_rank{self.rank}')
        self.model_path = os.path.join(self.run_dir, 'final_model.pt')
        self.quantized_model_path = os.path.join(self.run_dir, 'final_model.int6.ptz')
        self.quant_diagnostics_path = os.path.join(self.run_dir, 'quant_diagnostics.json')
_logger_hparams = None
_perf_logger = None
_perf_peak_tracker = None
_metrics_logger = None

def _rank_is_selected(spec, rank):
    spec = spec.strip()
    if spec == 'all':
        return True
    if not spec:
        return False
    return str(rank) in {part.strip() for part in spec.split(',') if part.strip()}

def set_logging_hparams(h):
    global _logger_hparams
    _logger_hparams = h


def set_metrics_logger(logger):
    global _metrics_logger
    _metrics_logger = logger


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


def log_event(event, **payload):
    if _metrics_logger is None or _logger_hparams is None or not _logger_hparams.is_main_process:
        return
    row = {
        'event': event,
        'time': time.time(),
        'run_id': _logger_hparams.run_id,
        'stage': _logger_hparams.stage,
        'method': _logger_hparams.method,
        'budget_id': _logger_hparams.budget_id,
        'seed': _logger_hparams.seed,
    }
    row.update(payload)
    _metrics_logger.log(row)

class JsonlLogger:

    def __init__(self, path, enabled):
        self.enabled = enabled
        self.path = path
        self._file = None
        if self.enabled:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._file = open(path, 'a', encoding='utf-8', buffering=1)

    def log(self, row):
        if not self.enabled:
            return
        self._file.write(json.dumps(_json_safe(row), separators=(',', ':'), sort_keys=True) + '\n')

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def _load_existing_manifest(path):
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding='utf-8'))


def _base_manifest(h, device, script_code):
    git_state = _collect_git_state()
    env_keys = (
        'RUN_ID', 'RUN_DIR', 'STAGE', 'METHOD', 'BASELINE_METHOD', 'RUN_GROUP', 'BUDGET_ID', 'SEED',
        'ITERATIONS', 'TRAIN_BATCH_TOKENS', 'TRAIN_SEQ_LEN', 'VAL_BATCH_TOKENS', 'EVAL_SEQ_LEN',
        'VAL_LOSS_EVERY', 'EVAL_AT_STEPS', 'TRAIN_SHARD_LIMIT', 'VAL_TOKEN_LIMIT', 'VAL_TOKEN_OFFSET',
        'LOCAL_VAL_TOKENS', 'LOCAL_VAL_OFFSET_TOKENS', 'MAX_WALLCLOCK_SECONDS', 'ABORT_ON_NAN',
        'MAX_PEAK_MEM_MIB', 'MAX_ARTIFACT_BYTES', 'MIN_TOKENS_PER_SEC', 'MAX_QUANT_BPB_GAP', 'EMA_ENABLED',
        'EMA_DECAY', 'FINAL_EVAL_MODE', 'QK_GAIN_INIT', 'SLIDING_WINDOW_ENABLED', 'TTT_ENABLED',
        'TTT_LR', 'TTT_EPOCHS', 'TTT_CHUNK_TOKENS', 'EVAL_STRIDE',
        'PRE_QUANT_TTT_ENABLED', 'PRE_QUANT_TTT_LR', 'PRE_QUANT_TTT_MIN_LR', 'PRE_QUANT_TTT_EPOCHS',
        'PRE_QUANT_TTT_WEIGHT_DECAY', 'PRE_QUANT_TTT_GRAD_CLIP_NORM', 'PRE_QUANT_TTT_FREEZE_BLOCKS',
        'PRE_QUANT_TTT_TOKEN_LIMIT', 'PRE_QUANT_TTT_TOKEN_OFFSET', 'PRE_QUANT_TTT_SOURCE', 'PRE_QUANT_TTT_FILES',
        'PRE_QUANT_TTT_SELECTION_MODE', 'PRE_QUANT_TTT_CHUNK_TOKENS', 'PRE_QUANT_TTT_ANCHOR_LAMBDA',
        'PRE_QUANT_TTT_QNOISE_ENABLED', 'PRE_QUANT_TTT_QNOISE_START_EPOCH', 'PRE_QUANT_TTT_QNOISE_PROB', 'PRE_QUANT_TTT_QNOISE_SCALE_MULT',
        'SAVE_POST_TTT_CHECKPOINT', 'POST_TTT_CHECKPOINT_PATH', 'QUANT_ONLY_FROM_POST_TTT', 'WRAPPER_CODE_BYTES_ESTIMATE',
        'MATRIX_BITS', 'EMBED_BITS', 'MATRIX_CLIP_SIGMAS', 'EMBED_CLIP_SIGMAS',
        'DATASET_NAME', 'DATASETS_DIR', 'TRAIN_FILES', 'VAL_FILES', 'VAL_BYTE_FILES', 'TOKENIZER_PATH',
    )
    return {
        'identity': {
            'run_id': h.run_id,
            'stage': h.stage,
            'method': h.method,
            'baseline_method': h.baseline_method,
            'run_group': h.run_group,
            'budget_id': h.budget_id,
            'seed': h.seed,
        },
        'code': {
            **git_state,
            'train_script': str(Path(__file__).resolve()),
            'train_script_sha256': _sha256_text(script_code),
        },
        'data': {
            'datasets_dir': str(Path(h.datasets_dir).resolve()),
            'tokenizer_path': str(Path(h.tokenizer_path).resolve()),
            'dataset_name': h.dataset_name,
            'train_glob': h.train_files,
            'val_glob': h.val_files,
            'val_byte_glob': h.val_byte_files,
            'pre_quant_ttt_source': h.pre_quant_ttt_source,
            'pre_quant_ttt_glob': h.pre_quant_ttt_files,
            'pre_quant_ttt_selection_mode': h.pre_quant_ttt_selection_mode,
            'train_shard_limit': h.train_shard_limit,
            'val_token_limit': h.local_val_tokens,
            'val_token_offset': h.local_val_offset_tokens,
        },
        'hardware': {
            'hostname': socket.gethostname(),
            'gpu_name': torch.cuda.get_device_name(device),
            'gpu_count': torch.cuda.device_count(),
            'cuda_version': torch.version.cuda,
            'torch_version': torch.__version__,
            'world_size': h.world_size,
            'local_rank': h.local_rank,
        },
        'training': {
            'iterations': h.iterations,
            'max_wallclock_seconds': h.max_wallclock_seconds,
            'train_batch_tokens': h.train_batch_tokens,
            'train_seq_len': h.train_seq_len,
            'grad_accum_steps': h.grad_accum_steps,
            'warmup_steps': h.warmup_steps,
            'warmdown_frac': h.warmdown_frac,
            'ema_enabled': h.ema_enabled,
            'ema_decay': h.ema_decay,
        },
        'evaluation': {
            'final_eval_mode': h.final_eval_mode,
            'val_batch_tokens': h.val_batch_tokens,
            'eval_seq_len': h.eval_seq_len,
            'val_loss_every': h.val_loss_every,
            'eval_at_steps': sorted(h.eval_at_steps),
            'sliding_window_requested': h.sliding_window_enabled,
            'sliding_window_enabled': _sliding_eval_enabled(h),
            'ttt_requested': h.ttt_enabled,
            'ttt_enabled': _ttt_eval_enabled(h),
            'pre_quant_ttt_requested': h.pre_quant_ttt_enabled,
            'pre_quant_ttt_enabled': h.pre_quant_ttt_enabled,
            'pre_quant_ttt_epochs': h.pre_quant_ttt_epochs,
            'pre_quant_ttt_lr': h.pre_quant_ttt_lr,
            'pre_quant_ttt_min_lr': h.pre_quant_ttt_min_lr,
            'pre_quant_ttt_token_limit': h.pre_quant_ttt_token_limit,
            'pre_quant_ttt_token_offset': h.pre_quant_ttt_token_offset,
            'pre_quant_ttt_source': h.pre_quant_ttt_source,
            'pre_quant_ttt_files': h.pre_quant_ttt_files,
            'pre_quant_ttt_shard_limit': h.pre_quant_ttt_shard_limit,
            'pre_quant_ttt_selection_mode': h.pre_quant_ttt_selection_mode,
            'pre_quant_ttt_chunk_tokens': h.pre_quant_ttt_chunk_tokens,
            'pre_quant_ttt_loop_mode': h.pre_quant_ttt_loop_mode,
            'pre_quant_ttt_anchor_lambda': h.pre_quant_ttt_anchor_lambda,
            'pre_quant_ttt_qnoise_enabled': h.pre_quant_ttt_qnoise_enabled,
            'pre_quant_ttt_qnoise_start_epoch': h.pre_quant_ttt_qnoise_start_epoch,
            'pre_quant_ttt_qnoise_prob': h.pre_quant_ttt_qnoise_prob,
            'pre_quant_ttt_qnoise_scale_mult': h.pre_quant_ttt_qnoise_scale_mult,
            'save_post_ttt_checkpoint': h.save_post_ttt_checkpoint,
            'post_ttt_checkpoint_path': str(Path(h.post_ttt_checkpoint_path).resolve()),
            'quant_only_from_post_ttt': h.quant_only_from_post_ttt,
            'wrapper_code_bytes_estimate': h.wrapper_code_bytes_estimate,
            'etlb_enabled': h.etlb_enabled,
            'quant_pipeline_enabled': _quant_pipeline_enabled(h),
            'bpb_metric_priority': _comparison_metric_priority(h),
            'checkpoint_metric_priority': list(CHECKPOINT_METRIC_PRIORITY),
            'byte_accounting_mode': 'sentencepiece_actual_utf8_bytes',
            'byte_accounting_rule': 'target_token_utf8_bytes_plus_single_space_prefix_when_target_has_leading_space_and_previous_token_is_not_boundary',
        },
        'guards': {
            'abort_on_nan': h.abort_on_nan,
            'max_peak_mem_mib': h.max_peak_mem_mib,
            'max_artifact_bytes': h.max_artifact_bytes,
            'min_tokens_per_sec': h.min_tokens_per_sec,
            'max_quant_bpb_gap': h.max_quant_bpb_gap,
        },
        'outputs': {
            'run_dir': str(Path(h.run_dir).resolve()),
            'manifest_path': str(Path(h.manifest_path).resolve()),
            'metrics_path': str(Path(h.metrics_path).resolve()),
            'summary_path': str(Path(h.summary_path).resolve()),
            'train_log_path': str(Path(h.logfile).resolve()),
            'model_path': str(Path(h.model_path).resolve()),
            'quantized_model_path': str(Path(h.quantized_model_path).resolve()),
            'post_ttt_checkpoint_path': str(Path(h.post_ttt_checkpoint_path).resolve()),
            'quant_diagnostics_path': str(Path(h.quant_diagnostics_path).resolve()),
        },
        'environment': {
            'selected_env': {key: os.environ[key] for key in env_keys if key in os.environ},
            'command': ' '.join(shlex.quote(arg) for arg in sys.argv),
        },
        'status': {
            'state': 'created',
        },
    }


def _merge_manifest(existing, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(existing.get(key), dict):
            _merge_manifest(existing[key], value)
            continue
        existing[key] = value
    return existing


def write_manifest(h, device, script_code, updates=None):
    if not h.is_main_process:
        return
    payload = _merge_manifest(_base_manifest(h, device, script_code), _load_existing_manifest(h.manifest_path))
    if updates is not None:
        payload = _merge_manifest(payload, updates)
    _json_dump(h.manifest_path, payload)


def write_summary(h, summary):
    if h.is_main_process:
        _json_dump(h.summary_path, summary)

class WindowStats:

    def __init__(self, maxlen):
        self.values = collections.defaultdict(lambda: collections.deque(maxlen=maxlen))

    def add(self, **metrics):
        for key, value in metrics.items():
            if value is None:
                continue
            self.values[key].append(float(value))

    def summary(self):
        out = {}
        for key, vals in self.values.items():
            if not vals:
                continue
            arr = sorted(vals)
            n = len(arr)
            out[f'{key}_avg'] = sum(arr) / n
            out[f'{key}_p50'] = arr[n // 2]
            out[f'{key}_p95'] = arr[min(n - 1, int(math.ceil(0.95 * n)) - 1)]
        out['sample_count'] = max((len(vals) for vals in self.values.values()), default=0)
        return out

class SampledStepTimer:

    def __init__(self, enabled):
        self.enabled = enabled and torch.cuda.is_available()
        self.events = {}
        self.cpu_starts = {}
        self.cpu_ms = {}

    def start(self, name):
        if self.enabled:
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            self.events[name] = [start_event, None]
        else:
            self.cpu_starts[name] = time.perf_counter()

    def stop(self, name):
        if self.enabled:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            self.events[name][1] = end_event
        else:
            self.cpu_ms[name] = 1000.0 * (time.perf_counter() - self.cpu_starts.pop(name))

    def finalize(self):
        if self.enabled and self.events:
            last_end = next((pair[1] for pair in reversed(self.events.values()) if pair[1] is not None), None)
            if last_end is not None:
                last_end.synchronize()

    def elapsed_ms(self, name):
        if self.enabled:
            start_event, end_event = self.events[name]
            return start_event.elapsed_time(end_event)
        return self.cpu_ms[name]

class PeakMemoryTracker:

    def __init__(self, device):
        self.enabled = torch.cuda.is_available()
        self.device = device
        self.run_peak_allocated = 0
        self.run_peak_reserved = 0
        self.last_allocation_count = 0
        self.last_cuda_malloc_count = 0
        if self.enabled:
            self._update_run_peaks()
            self._refresh_counters()

    def _update_run_peaks(self):
        self.run_peak_allocated = max(self.run_peak_allocated, int(torch.cuda.max_memory_allocated(self.device)))
        self.run_peak_reserved = max(self.run_peak_reserved, int(torch.cuda.max_memory_reserved(self.device)))

    def _read_allocation_count(self):
        stats = torch.cuda.memory_stats(self.device)
        return int(stats.get('allocation.all.allocated', 0))

    def _read_cuda_malloc_count(self):
        stats = torch.cuda.memory_stats(self.device)
        return int(stats.get('num_device_alloc', stats.get('segment.all.allocated', 0)))

    def _refresh_counters(self):
        self.last_allocation_count = self._read_allocation_count()
        self.last_cuda_malloc_count = self._read_cuda_malloc_count()

    def reset_interval(self):
        if not self.enabled:
            return
        self._update_run_peaks()
        self._refresh_counters()
        torch.cuda.reset_peak_memory_stats(self.device)

    def interval_peaks_mb(self):
        if not self.enabled:
            return {}
        self._update_run_peaks()
        return {
            'mem_peak_alloc_mb': int(torch.cuda.max_memory_allocated(self.device)) // (1024 * 1024),
            'mem_peak_reserved_mb': int(torch.cuda.max_memory_reserved(self.device)) // (1024 * 1024),
        }

    def run_peaks_mb(self):
        if not self.enabled:
            return {}
        self._update_run_peaks()
        return {
            'run_peak_alloc_mb': self.run_peak_allocated // (1024 * 1024),
            'run_peak_reserved_mb': self.run_peak_reserved // (1024 * 1024),
        }

class PerfLogger:

    def __init__(self, h):
        self.rank = h.rank
        self.run_id = h.run_id
        self.enabled = h.perf_enabled and _rank_is_selected(h.perf_ranks, h.rank)
        self.logger = JsonlLogger(h.perf_logfile, self.enabled)

    def log(self, event_type, **kwargs):
        if not self.enabled:
            return
        row = {
            'type': event_type,
            'rank': self.rank,
            'run_id': self.run_id,
            'time': time.time(),
        }
        row.update(kwargs)
        self.logger.log(row)

    def close(self):
        self.logger.close()

class DDPCommTimingState:

    def __init__(self, process_group=None):
        self.process_group = process_group
        self.sample_active = False
        self.step_ms = 0.0
        self.call_count = 0
        self.bucket_bytes = 0

    def start_step(self, sample_active):
        self.sample_active = sample_active
        self.step_ms = 0.0
        self.call_count = 0
        self.bucket_bytes = 0

    def record(self, duration_ms, bucket_bytes):
        self.step_ms += duration_ms
        self.call_count += 1
        self.bucket_bytes += bucket_bytes

    def snapshot(self):
        avg_bucket_size_mb = (self.bucket_bytes / self.call_count) / (1024 * 1024) if self.call_count > 0 else 0.0
        return {
            'allreduce_ms': self.step_ms,
            'num_allreduce_calls': self.call_count,
            'avg_bucket_size_mb': avg_bucket_size_mb,
        }

def timed_allreduce_hook(state, bucket):
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as ddp_default_hooks
    if not state.sample_active:
        return ddp_default_hooks.allreduce_hook(state.process_group, bucket)
    start = time.perf_counter()
    buf = bucket.buffer()
    bucket_bytes = int(buf.numel()) * buf.element_size()
    fut = ddp_default_hooks.allreduce_hook(state.process_group, bucket)

    def _callback(fut):
        state.record(1000.0 * (time.perf_counter() - start), bucket_bytes)
        value = fut.value()
        return value[0] if isinstance(value, list) else value

    return fut.then(_callback)

class ProfilerController:

    def __init__(self, h):
        self.enabled = h.profiler_enabled and _rank_is_selected(h.profiler_ranks, h.rank)
        self.profiler = None
        self.current_anchor = None
        self.remaining_steps = 0
        self.anchor_steps = sorted(_parse_step_set(h.profiler_steps))
        self.warmup_steps = max(h.profiler_warmup_steps, 0)
        self.active_steps = max(h.profiler_active_steps, 1)
        self.record_shapes = h.profiler_record_shapes
        self.profile_memory = h.profiler_profile_memory
        self.with_flops = h.profiler_with_flops
        self.with_stack = h.profiler_with_stack
        self.log_dir = h.profiler_dir

    def _window_start(self, anchor):
        return max(anchor - self.warmup_steps, 0)

    def _window_length(self):
        return self.warmup_steps + self.active_steps

    def maybe_start(self, step):
        if not self.enabled or self.profiler is not None:
            return
        for anchor in self.anchor_steps:
            if self._window_start(anchor) == step:
                import torch.profiler as profiler
                os.makedirs(self.log_dir, exist_ok=True)
                self.current_anchor = anchor
                self.remaining_steps = self._window_length()
                self.profiler = profiler.profile(
                    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
                    schedule=profiler.schedule(wait=0, warmup=self.warmup_steps, active=self.active_steps, repeat=1),
                    on_trace_ready=profiler.tensorboard_trace_handler(os.path.join(self.log_dir, f'step{anchor}')),
                    record_shapes=self.record_shapes,
                    profile_memory=self.profile_memory,
                    with_flops=self.with_flops,
                    with_stack=self.with_stack,
                )
                self.profiler.__enter__()
                if perf_is_enabled():
                    perf_log('profiler_window_start', anchor_step=anchor, window_start_step=step, warmup_steps=self.warmup_steps, active_steps=self.active_steps)
                return

    def step(self, step):
        if self.profiler is None:
            return
        self.profiler.step()
        self.remaining_steps -= 1
        if self.remaining_steps <= 0:
            profiler_summary = _extract_profiler_summaries(self.profiler)
            self.profiler.__exit__(None, None, None)
            if perf_is_enabled():
                perf_log('profiler_window_end', anchor_step=self.current_anchor, window_end_step=step, **profiler_summary)
            self.profiler = None
            self.current_anchor = None

    def close(self):
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.profiler = None
            self.current_anchor = None

def set_perf_logger(logger, peak_tracker):
    global _perf_logger, _perf_peak_tracker
    _perf_logger = logger
    _perf_peak_tracker = peak_tracker

def perf_log(event_type, **kwargs):
    if _perf_logger is not None:
        _perf_logger.log(event_type, **kwargs)

def perf_is_enabled():
    return _perf_logger is not None and _perf_logger.enabled

def get_memory_stats_mb(device):
    if not torch.cuda.is_available():
        return {}
    stats = torch.cuda.memory_stats(device)
    out = {
        'mem_alloc_mb': int(torch.cuda.memory_allocated(device)) // (1024 * 1024),
        'mem_reserved_mb': int(torch.cuda.memory_reserved(device)) // (1024 * 1024),
        'active_mb': int(stats.get('active_bytes.all.current', 0)) // (1024 * 1024),
        'inactive_split_mb': int(stats.get('inactive_split_bytes.all.current', 0)) // (1024 * 1024),
        'alloc_retries': int(stats.get('num_alloc_retries', 0)),
        'ooms': int(stats.get('num_ooms', 0)),
    }
    if _perf_peak_tracker is not None:
        out.update(_perf_peak_tracker.interval_peaks_mb())
        out['allocation_count_delta'] = int(stats.get('allocation.all.allocated', 0)) - _perf_peak_tracker.last_allocation_count
        out['cuda_malloc_delta'] = int(stats.get('num_device_alloc', stats.get('segment.all.allocated', 0))) - _perf_peak_tracker.last_cuda_malloc_count
    return out

def _clone_grouped_params(grouped_params):
    return {name: [p.detach().float().clone() for p in params] for name, params in grouped_params.items()}

def _group_delta_norms(grouped_params, snapshots):
    out = {}
    for name, params in grouped_params.items():
        total = 0.0
        for p, before in zip(params, snapshots[name], strict=True):
            delta = p.detach().float() - before
            total += float(delta.square().sum().item())
        out[name] = total ** 0.5
    return out

def _gather_rank_range(value, device):
    if not (dist.is_available() and dist.is_initialized()):
        return (value, value)
    min_tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    max_tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)
    dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
    return (float(min_tensor.item()), float(max_tensor.item()))

def _gather_per_rank_memory(device, memory_stats):
    if not (dist.is_available() and dist.is_initialized()):
        return [{'rank': 0, 'alloc_mb': memory_stats.get('mem_alloc_mb', 0), 'reserved_mb': memory_stats.get('mem_reserved_mb', 0), 'peak_alloc_mb': memory_stats.get('mem_peak_alloc_mb', 0)}]
    local = torch.tensor([
        float(memory_stats.get('mem_alloc_mb', 0)),
        float(memory_stats.get('mem_reserved_mb', 0)),
        float(memory_stats.get('mem_peak_alloc_mb', 0)),
    ], device=device, dtype=torch.float64)
    gathered = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    out = []
    for rank, tensor in enumerate(gathered):
        out.append({
            'rank': rank,
            'alloc_mb': float(tensor[0].item()),
            'reserved_mb': float(tensor[1].item()),
            'peak_alloc_mb': float(tensor[2].item()),
        })
    return out

def _gather_chunk_values(local_values):
    if not (dist.is_available() and dist.is_initialized()):
        return local_values
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_values)
    out = []
    for values in gathered:
        out.extend(values)
    return out

def _make_chunk_bpb_payload(chunk_bpbs):
    if not chunk_bpbs:
        return {'bpb_by_chunk': [], 'bpb_chunk_count': 0}
    return {
        'bpb_by_chunk': [float(v) for v in chunk_bpbs],
        'bpb_chunk_count': len(chunk_bpbs),
        'bpb_chunk_min': float(min(chunk_bpbs)),
        'bpb_chunk_max': float(max(chunk_bpbs)),
    }


def _byte_accounting_payload(token_count, byte_count):
    tokens = float(token_count.item())
    utf8_bytes = float(byte_count.item())
    return {
        'eval_tokens': tokens,
        'eval_utf8_bytes': utf8_bytes,
        'bytes_per_token': (utf8_bytes / tokens if tokens > 0 else None),
        'bpb_formula': 'loss_nats/log(2)*tokens/utf8_bytes',
    }

def _extract_profiler_summaries(profiler_obj, topk=10):
    key_averages = list(profiler_obj.key_averages())
    cpu_ops = sorted(key_averages, key=lambda evt: getattr(evt, 'self_cpu_time_total', 0.0), reverse=True)
    cuda_ops = sorted(key_averages, key=lambda evt: getattr(evt, 'self_cuda_time_total', 0.0), reverse=True)
    mem_ops = sorted(key_averages, key=lambda evt: max(getattr(evt, 'self_cpu_memory_usage', 0), getattr(evt, 'self_cuda_memory_usage', 0)), reverse=True)

    def _pack(events, cpu=False, cuda=False, memory=False):
        out = []
        for evt in events[:topk]:
            row = {'name': evt.key, 'count': int(getattr(evt, 'count', 0))}
            if cpu:
                row['self_cpu_time_total_us'] = float(getattr(evt, 'self_cpu_time_total', 0.0))
            if cuda:
                row['self_cuda_time_total_us'] = float(getattr(evt, 'self_cuda_time_total', 0.0))
            if memory:
                row['self_cpu_memory_usage'] = int(getattr(evt, 'self_cpu_memory_usage', 0))
                row['self_cuda_memory_usage'] = int(getattr(evt, 'self_cuda_memory_usage', 0))
            out.append(row)
        return out

    return {
        'top_cpu_ops': _pack(cpu_ops, cpu=True),
        'top_cuda_ops': _pack(cuda_ops, cuda=True),
        'top_memory_ops': _pack(mem_ops, memory=True),
        'kernel_count': int(sum((getattr(evt, 'count', 0) for evt in key_averages))),
    }

def get_lr_by_group(optimizers):
    out = {}
    for name, opt in optimizers.named_optimizers:
        out[name] = [float(group['lr']) for group in opt.param_groups]
    return out

def _group_tensor_norms(grouped_params, grad=False):
    out = {}
    for name, params in grouped_params.items():
        total = 0.0
        for p in params:
            t = p.grad if grad else p
            if t is None:
                continue
            total += float(t.detach().float().square().sum().item())
        out[name] = total ** 0.5
    return out

def _should_sample_step(h, step, loop_enabled_step):
    if not perf_is_enabled():
        return False
    if step < 5:
        return True
    forced = _parse_step_set(h.perf_force_steps)
    if step in forced:
        return True
    if h.perf_sample_every > 0 and step % h.perf_sample_every == 0:
        return True
    dense_every = max(h.perf_loop_dense_every, 1)
    expected_loop_step = int(h.iterations * h.enable_looping_at) if h.iterations > 0 else None
    if expected_loop_step is not None and step <= expected_loop_step and expected_loop_step - step <= h.perf_loop_dense_before and step % dense_every == 0:
        return True
    if loop_enabled_step is not None and 0 <= step - loop_enabled_step <= h.perf_loop_dense_after and step % dense_every == 0:
        return True
    return False

def _should_log_perf_step(h, step):
    return perf_is_enabled() and h.perf_log_every > 0 and (step < 5 or step % h.perf_log_every == 0)

class ValidationData:

    def __init__(self, h, device):
        self.tokenizer_path = str(Path(h.tokenizer_path).resolve())
        self.tokenizer_kind, self.tokenizer_meta, self.sp = _load_tokenizer_for_validation(h.tokenizer_path)
        self.tokenizer_sha256 = _sha256_file(self.tokenizer_meta.get('sentencepiece_model_path', h.tokenizer_path))
        self.vocab_size = int(self.tokenizer_meta.get('vocab_size', int(self.sp.vocab_size()) if self.sp is not None else 0))
        if self.vocab_size != h.vocab_size:
            raise ValueError(f'VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={self.vocab_size}')
        self.val_tokens, self.selection = load_validation_tokens(h.val_files, h.eval_seq_len, max_tokens=h.local_val_tokens, offset_tokens=h.local_val_offset_tokens)
        self.val_files = self.selection['val_shards']
        self.val_token_bytes = load_validation_sidecar(h.val_byte_files, h.eval_seq_len, max_tokens=h.local_val_tokens, offset_tokens=h.local_val_offset_tokens)
        if self.val_token_bytes is not None:
            self.val_token_bytes = self.val_token_bytes.to(device=device, dtype=torch.int16)
        if self.val_token_bytes is None and self.tokenizer_meta.get('requires_byte_sidecar'):
            raise ValueError('VAL_BYTE_FILES is required for tokenizer metadata that declares requires_byte_sidecar')
        if self.val_token_bytes is None and (
            self.tokenizer_meta.get('caseops_sidecar_utf8_bytes') is not None
            or self.tokenizer_meta.get('caseops_zero_byte_token_count') is not None
        ):
            raise ValueError('VAL_BYTE_FILES is required for tokenizer metadata with caseops byte-accounting fields')
        self.pre_quant_ttt_tokens = None
        self.pre_quant_ttt_segments = []
        self.pre_quant_ttt_selection = None
        self.pre_quant_ttt_token_bytes = None
        if h.pre_quant_ttt_enabled:
            ttt_limit = (h.pre_quant_ttt_token_limit if h.pre_quant_ttt_token_limit > 0 else self.selection['selected_token_limit'])
            if h.pre_quant_ttt_source == 'val':
                if h.pre_quant_ttt_token_offset is None:
                    eval_start = int(self.selection['selected_offset_tokens'])
                    eval_end = int(self.selection['selected_end_offset_tokens'])
                    usable = int(self.selection['usable_tokens'])
                    if eval_end + ttt_limit <= usable:
                        ttt_offset = eval_end
                    elif eval_start - ttt_limit >= 0:
                        ttt_offset = eval_start - ttt_limit
                    else:
                        raise ValueError('pre-quant TTT requires an explicit non-overlapping PRE_QUANT_TTT_TOKEN_OFFSET when validation coverage leaves no disjoint adaptation slice')
                else:
                    ttt_offset = h.pre_quant_ttt_token_offset
                self.pre_quant_ttt_tokens, self.pre_quant_ttt_selection = load_validation_tokens(
                    h.pre_quant_ttt_files,
                    h.eval_seq_len,
                    max_tokens=ttt_limit,
                    offset_tokens=ttt_offset,
                    selection_prefix='ttt',
                    shard_limit=h.pre_quant_ttt_shard_limit,
                )
                if not (
                    self.pre_quant_ttt_selection['selected_end_offset_tokens'] <= self.selection['selected_offset_tokens']
                    or self.pre_quant_ttt_selection['selected_offset_tokens'] >= self.selection['selected_end_offset_tokens']
                ):
                    raise ValueError('pre-quant TTT adaptation slice must not overlap the reported validation slice')
                self.pre_quant_ttt_token_bytes = load_validation_sidecar(
                    h.val_byte_files,
                    h.eval_seq_len,
                    max_tokens=ttt_limit,
                    offset_tokens=ttt_offset,
                )
                self.pre_quant_ttt_segments = [{
                    'segment_id': 0,
                    'tokens': self.pre_quant_ttt_tokens,
                    'token_bytes': self.pre_quant_ttt_token_bytes,
                    'selection': self.pre_quant_ttt_selection,
                }]
            else:
                ttt_offset = (0 if h.pre_quant_ttt_token_offset is None else h.pre_quant_ttt_token_offset)
                if h.pre_quant_ttt_selection_mode == 'stratified':
                    self.pre_quant_ttt_segments, self.pre_quant_ttt_selection = load_stratified_token_segments(
                        h.pre_quant_ttt_files,
                        h.eval_seq_len,
                        total_tokens=ttt_limit,
                        chunk_tokens=h.pre_quant_ttt_chunk_tokens,
                        selection_prefix='ttt',
                        shard_limit=h.pre_quant_ttt_shard_limit,
                    )
                    if self.pre_quant_ttt_segments:
                        self.pre_quant_ttt_tokens = self.pre_quant_ttt_segments[0]['tokens']
                else:
                    offset_mode = ('tail' if h.pre_quant_ttt_token_offset is None else 'wrap')
                    self.pre_quant_ttt_tokens, self.pre_quant_ttt_selection = load_validation_tokens(
                        h.pre_quant_ttt_files,
                        h.eval_seq_len,
                        max_tokens=ttt_limit,
                        offset_tokens=ttt_offset,
                        selection_prefix='ttt',
                        offset_mode=offset_mode,
                        shard_limit=h.pre_quant_ttt_shard_limit,
                    )
                    self.pre_quant_ttt_segments = [{
                        'segment_id': 0,
                        'tokens': self.pre_quant_ttt_tokens,
                        'token_bytes': None,
                        'selection': self.pre_quant_ttt_selection,
                    }]
            self.pre_quant_ttt_selection['ttt_source'] = h.pre_quant_ttt_source
            self.pre_quant_ttt_selection['ttt_files'] = h.pre_quant_ttt_files
            self.pre_quant_ttt_selection['ttt_shard_limit'] = h.pre_quant_ttt_shard_limit
            self.pre_quant_ttt_selection['ttt_selection_mode'] = h.pre_quant_ttt_selection_mode
            if self.pre_quant_ttt_token_bytes is not None:
                self.pre_quant_ttt_token_bytes = self.pre_quant_ttt_token_bytes.to(device=device, dtype=torch.int16)
                if self.pre_quant_ttt_segments:
                    self.pre_quant_ttt_segments[0]['token_bytes'] = self.pre_quant_ttt_token_bytes
        if self.tokenizer_kind == 'pure_byte':
            self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = build_pure_byte_luts(self.tokenizer_meta, h.vocab_size, device)
        else:
            self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = build_sentencepiece_luts(self.sp, h.vocab_size, device)
        self.caseops_roundtrip_ok = self.tokenizer_meta.get('caseops_roundtrip_ok')
        self.caseops_raw_utf8_bytes = self.tokenizer_meta.get('caseops_raw_utf8_bytes')
        self.caseops_sidecar_utf8_bytes = self.tokenizer_meta.get('caseops_sidecar_utf8_bytes')
        self.caseops_zero_byte_token_count = self.tokenizer_meta.get('caseops_zero_byte_token_count')

    def target_bytes(self, start, end, tgt_ids=None, prev_ids=None):
        return self._lookup_target_bytes(self.val_token_bytes, start, end, tgt_ids=tgt_ids, prev_ids=prev_ids)

    def pre_quant_ttt_bytes(self, start, end, tgt_ids=None, prev_ids=None):
        return self._lookup_target_bytes(self.pre_quant_ttt_token_bytes, start, end, tgt_ids=tgt_ids, prev_ids=prev_ids)

    def _lookup_target_bytes(self, sidecar, start, end, tgt_ids=None, prev_ids=None):
        if sidecar is not None:
            return sidecar[start:end]
        if tgt_ids is None or prev_ids is None:
            raise ValueError('tgt_ids and prev_ids are required when validation sidecar is unavailable')
        token_bytes = self.base_bytes_lut[tgt_ids].to(torch.float64)
        token_bytes += (self.has_leading_space_lut[tgt_ids] & ~self.is_boundary_token_lut[prev_ids]).to(torch.float64)
        return token_bytes


def _load_tokenizer_for_validation(tokenizer_path):
    path = Path(tokenizer_path)
    if path.suffix == '.json':
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('tokenizer_type') == 'pure_byte':
            config = payload.get('config') or {}
            return 'pure_byte', {
                **payload,
                'byte_offset': int(config.get('byte_offset', 4)),
                'byte_count': int(config.get('byte_count', 256)),
                'vocab_size': int(payload.get('vocab_size', 260)),
            }, None
        kind = payload.get('kind', 'json')
        sp_model_path = payload.get('sentencepiece_model_path') or payload.get('model_path')
        if sp_model_path is None:
            raise ValueError(f'tokenizer metadata at {path} must define sentencepiece_model_path')
        sp_model_path = Path(sp_model_path)
        if not sp_model_path.is_absolute():
            sp_model_path = (path.parent / sp_model_path).resolve()
        sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
        payload = {
            **payload,
            'sentencepiece_model_path': str(sp_model_path),
            'vocab_size': int(payload.get('vocab_size', int(sp.vocab_size()))),
        }
        return kind, payload, sp
    sp = spm.SentencePieceProcessor(model_file=str(path))
    return 'sentencepiece_bpe', {'vocab_size': int(sp.vocab_size())}, sp


def build_pure_byte_luts(meta, vocab_size, device):
    table_size = max(int(meta.get('vocab_size', vocab_size)), vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    byte_offset = int(meta.get('byte_offset', 4))
    byte_count = int(meta.get('byte_count', 256))
    byte_end = min(table_size, byte_offset + byte_count)
    if byte_offset < byte_end:
        base_bytes_np[byte_offset:byte_end] = 1
        is_boundary_token_np[byte_offset:byte_end] = False
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

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

def load_validation_tokens(pattern, seq_len, max_tokens=0, offset_tokens=0, selection_prefix='val', offset_mode='wrap', shard_limit=0):
    files = _glob_data_files(pattern, shard_limit=shard_limit)
    if not files:
        raise FileNotFoundError(f'No files found for pattern: {pattern}')
    shard_rows = []
    for index, file in enumerate(files):
        shard_rows.append({
            'index': index,
            'name': _shard_name(file),
            'path': str(file.resolve()),
            'size_bytes': int(file.stat().st_size),
            'token_count': _read_num_tokens(file),
        })
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = (tokens.numel() - 1) // seq_len * seq_len
    if usable <= 0:
        raise ValueError(f'Validation split is too short for TRAIN_SEQ_LEN={seq_len}')
    tokens = tokens[:usable + 1]
    selected_offset = 0
    requested_max_tokens = int(max_tokens)
    if max_tokens > 0:
        max_tokens = (max_tokens // seq_len) * seq_len
        max_tokens = max(seq_len, max_tokens)
        max_tokens = min(max_tokens, usable)
        max_start = usable - max_tokens
        offset_tokens = max(0, (offset_tokens // seq_len) * seq_len)
        if max_start > 0 and offset_mode == 'tail':
            offset_tokens = max_start
        elif max_start > 0:
            offset_tokens = offset_tokens % (max_start + seq_len)
            offset_tokens = offset_tokens // seq_len * seq_len
        else:
            offset_tokens = 0
        selected_offset = int(offset_tokens)
        tokens = tokens[offset_tokens:offset_tokens + max_tokens + 1].contiguous()
    actual_tokens = int(tokens.numel() - 1)
    selection = {
        'stream_pattern': pattern,
        'stream_shards': shard_rows,
        'usable_tokens': int(usable),
        'requested_token_limit': requested_max_tokens,
        'selected_token_limit': actual_tokens,
        'selected_offset_tokens': selected_offset,
        'selected_end_offset_tokens': selected_offset + actual_tokens,
        'selected_seq_len': int(seq_len),
        'selection_mode': ('subset' if requested_max_tokens > 0 else 'full'),
    }
    selection[f'{selection_prefix}_pattern'] = pattern
    selection[f'{selection_prefix}_shards'] = shard_rows
    return (tokens, selection)


def load_validation_sidecar(pattern, seq_len, max_tokens=0, offset_tokens=0):
    if not pattern:
        return None
    sidecar_tokens, selection = load_validation_tokens(pattern, seq_len, max_tokens=max_tokens, offset_tokens=offset_tokens)
    del selection
    if sidecar_tokens.numel() < 2:
        raise ValueError('validation sidecar must contain at least two entries')
    return sidecar_tokens[1:]

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


def _load_segment_window(file, start_token, token_count):
    mm = _get_shard_memmap(file)
    end_token = start_token + token_count
    if end_token > mm.shape[0]:
        raise ValueError(f'Segment window exceeds shard bounds for {file}: end={end_token} shape={mm.shape[0]}')
    window = np.array(mm[start_token:end_token], dtype=np.uint16)
    return torch.from_numpy(window)


def load_stratified_token_segments(pattern, seq_len, total_tokens, chunk_tokens, selection_prefix='ttt', shard_limit=0):
    files = _glob_data_files(pattern, shard_limit=shard_limit)
    if not files:
        raise FileNotFoundError(f'No files found for pattern: {pattern}')
    shard_rows = []
    usable_by_shard = []
    total_usable = 0
    for index, file in enumerate(files):
        token_count = _read_num_tokens(file)
        usable_tokens = (token_count - 1) // seq_len * seq_len
        shard_rows.append({
            'index': index,
            'name': _shard_name(file),
            'path': str(file.resolve()),
            'size_bytes': int(file.stat().st_size),
            'token_count': token_count,
            'usable_tokens': int(usable_tokens),
        })
        usable_by_shard.append(int(usable_tokens))
        total_usable += int(usable_tokens)
    if total_usable <= 0:
        raise ValueError(f'Stratified source is too short for TRAIN_SEQ_LEN={seq_len}')
    chunk_tokens = max(seq_len, (chunk_tokens // seq_len) * seq_len)
    chunk_tokens = min(chunk_tokens, total_usable)
    total_tokens = min(total_tokens, total_usable)
    total_tokens = max(seq_len, (total_tokens // chunk_tokens) * chunk_tokens)
    num_chunks = max(1, total_tokens // chunk_tokens)
    max_global_start = max(total_usable - chunk_tokens, 0)
    cumulative = []
    running = 0
    for usable in usable_by_shard:
        cumulative.append((running, running + usable))
        running += usable
    segments = []
    segment_rows = []
    for chunk_idx in range(num_chunks):
        if num_chunks == 1 or max_global_start == 0:
            global_start = 0
        else:
            global_start = int(round(chunk_idx * max_global_start / max(num_chunks - 1, 1)))
            global_start = (global_start // seq_len) * seq_len
        shard_index = None
        local_start = None
        for idx, (shard_start, shard_end) in enumerate(cumulative):
            if shard_start <= global_start < shard_end or idx == len(cumulative) - 1:
                shard_index = idx
                local_start = global_start - shard_start
                shard_usable = usable_by_shard[idx]
                local_start = min(max(local_start, 0), max(shard_usable - chunk_tokens, 0))
                local_start = (local_start // seq_len) * seq_len
                break
        file = files[shard_index]
        tokens = _load_segment_window(file, local_start, chunk_tokens + 1).contiguous()
        segment = {
            'segment_id': chunk_idx,
            'tokens': tokens,
            'token_bytes': None,
            'selection': {
                'segment_id': chunk_idx,
                'global_offset_tokens': int(global_start),
                'selected_offset_tokens': int(local_start),
                'selected_end_offset_tokens': int(local_start + chunk_tokens),
                'selected_token_limit': int(chunk_tokens),
                'selected_seq_len': int(seq_len),
                'selection_mode': 'stratified_chunk',
                'shard': shard_rows[shard_index],
            },
        }
        segments.append(segment)
        segment_rows.append(segment['selection'])
    selection = {
        'stream_pattern': pattern,
        'stream_shards': shard_rows,
        'usable_tokens': int(total_usable),
        'requested_token_limit': int(total_tokens),
        'selected_token_limit': int(num_chunks * chunk_tokens),
        'selected_offset_tokens': 0,
        'selected_end_offset_tokens': int(num_chunks * chunk_tokens),
        'selected_seq_len': int(seq_len),
        'selection_mode': 'stratified',
        'segment_count': int(num_chunks),
        'segment_token_limit': int(chunk_tokens),
        'segments': segment_rows,
    }
    selection[f'{selection_prefix}_pattern'] = pattern
    selection[f'{selection_prefix}_shards'] = shard_rows
    return segments, selection

class ShuffledSequenceLoader:

    def __init__(self, h, device):
        self.world_size = h.world_size
        self.seq_len = h.train_seq_len
        self.device = device
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files:
            raise FileNotFoundError(f'No files found for pattern: {h.train_files}')
        if h.train_shard_limit > 0:
            all_files = all_files[:h.train_shard_limit]
        if not all_files:
            raise FileNotFoundError(f'TRAIN_SHARD_LIMIT={h.train_shard_limit} selected zero shards from {h.train_files}')
        self.all_files = all_files
        self.shard_names = [_shard_name(file) for file in all_files]
        self.files = all_files[h.rank::h.world_size]
        self.rng = np.random.Generator(np.random.PCG64(h.rank))
        self.num_tokens = [_read_num_tokens(f) for f in self.files]
        self.start_inds = [[] for _ in self.files]
        for si in range(len(self.files)):
            self._reset_shard(si)

    def _reset_shard(self, si):
        max_phase = min(self.seq_len - 1, max(0, self.num_tokens[si] - self.seq_len - 1))
        phase = int(self.rng.integers(max_phase + 1)) if max_phase > 0 else 0
        num_sequences = (self.num_tokens[si] - 1 - phase) // self.seq_len
        sequence_order = self.rng.permutation(num_sequences)
        self.start_inds[si] = (phase + sequence_order * self.seq_len).tolist()

    def next_batch(self, global_tokens, grad_accum_steps):
        device_tokens = global_tokens // (self.world_size * grad_accum_steps)
        device_batch_size = device_tokens // self.seq_len
        remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
        x = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        y = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
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
            window = torch.as_tensor(np.array(mm[start_ind:start_ind + self.seq_len + 1], dtype=np.int64))
            x[bi] = window[:-1]
            y[bi] = window[1:]
        return (x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True))

class RMSNorm(nn.Module):

    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):

    def forward(self, x):
        w = self.weight.to(x.dtype)
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
        return (self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype))

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
        y = flash_attn_3_func(q, k, v, causal=True)
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
        self.blocks = nn.ModuleList([Block(h.model_dim, h.num_heads, h.num_kv_heads, h.mlp_mult, h.rope_base, h.qk_gain_init, h.train_seq_len, layer_idx=i, ln_scale=h.ln_scale) for i in range(h.num_layers)])
        if h.rope_dims > 0:
            head_dim = h.model_dim // h.num_heads
            for block in self.blocks:
                block.attn.rope_dims = h.rope_dims
                block.attn.rotary = Rotary(head_dim, base=h.rope_base, train_seq_len=h.train_seq_len, rope_dims=h.rope_dims)
        self.final_norm = RMSNorm()
        self.lm_head = None if h.tie_embeddings else CastedLinear(h.embedding_dim, h.vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        if h.xsa_last_n > 0:
            for i in range(max(0, h.num_layers - h.xsa_last_n), h.num_layers):
                self.blocks[i].attn.use_xsa = True
        if h.parallel_residual_start >= 0:
            for i in range(h.parallel_residual_start, h.num_layers):
                self.blocks[i].parallel = True
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

    def forward_logits(self, input_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x0 = x
        skips = []
        enc_iter = self.encoder_indices if self.looping_active else range(self.num_encoder_layers)
        dec_iter = self.decoder_indices if self.looping_active else range(self.num_encoder_layers, self.num_encoder_layers + self.num_decoder_layers)
        for i in enc_iter:
            x = self.blocks[i](x, x0)
            skips.append(x)
        for skip_idx, i in enumerate(dec_iter):
            if skip_idx < self.num_skip_weights and skips:
                scaled_skip = self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
                if self.skip_gates is not None:
                    g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                    x = torch.lerp(scaled_skip, x, g)
                else:
                    x = x + scaled_skip
            x = self.blocks[i](x, x0)
        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids, target_ids):
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction='mean')

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
        self.optimizer_muon = Muon(matrix_params, lr=h.matrix_lr, momentum=h.muon_momentum, backend_steps=h.muon_backend_steps, weight_decay=h.muon_wd, row_normalize=h.muon_row_normalize)
        for group in self.optimizer_muon.param_groups:
            group['base_lr'] = h.matrix_lr
        self.optimizer_scalar = torch.optim.AdamW([{'params': scalar_params, 'lr': h.scalar_lr, 'base_lr': h.scalar_lr}], betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.adam_wd, fused=True)
        self.named_optimizers = [('embed', self.optimizer_tok), ('matrix', self.optimizer_muon), ('scalar', self.optimizer_scalar)]
        if base_model.lm_head is not None:
            self.optimizer_head = torch.optim.Adam([{'params': [base_model.lm_head.weight], 'lr': h.head_lr, 'base_lr': h.head_lr}], betas=(h.beta1, h.beta2), eps=h.adam_eps, fused=True)
            self.named_optimizers.insert(1, ('head', self.optimizer_head))
        else:
            self.optimizer_head = None
        self.optimizers = [opt for _, opt in self.named_optimizers]
        self.grouped_params = {'embed': [base_model.tok_emb.weight], 'matrix': matrix_params, 'scalar': scalar_params}
        if base_model.lm_head is not None:
            self.grouped_params['head'] = [base_model.lm_head.weight]

    def __iter__(self):
        return iter(self.optimizers)

    def zero_grad_all(self):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def step(self, timed_step=None):
        for name, opt in self.named_optimizers:
            if timed_step is None:
                opt.step()
            else:
                timed_step(name, opt.step)
        self.zero_grad_all()

def restore_fp32_params(model):
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    for name, param in model.named_parameters():
        if (param.ndim < 2 or any((pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))) and param.dtype != torch.float32:
            param.data = param.data.float()

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
            x, _ = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            model.forward_logits(x)
    for hook in hooks:
        hook.remove()
    for name in hessians:
        hessians[name] = hessians[name].cpu() / n_calibration_batches
    return hessians

def gptq_quantize_weight(w, H, clip_sigmas=3.0, clip_range=63, block_size=128):
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
    s = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
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
    layer_quant_error = {}
    zero_fraction_by_layer = {}
    bits_by_group = {'matrix': h.matrix_bits, 'embed': h.embed_bits}
    clip_sigma_by_group = {'matrix': h.matrix_clip_sigmas, 'embed': h.embed_clip_sigmas}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = 'passthrough (float16)'
            continue
        cs = h.embed_clip_sigmas if 'tok_emb' in name else h.matrix_clip_sigmas
        bits = h.embed_bits if 'tok_emb' in name else h.matrix_bits
        q, s = gptq_quantize_weight(t, hessians[name], clip_sigmas=cs, clip_range=2 ** (bits - 1) - 1)
        result[name + '.q'] = q
        result[name + '.scale'] = s
        meta[name] = f'gptq (int{bits})'
        dequant = q.float() * s.float().view(q.shape[0], *[1] * (q.ndim - 1))
        denom = float(t.float().square().mean().sqrt().item())
        error = float((dequant - t.float()).square().mean().sqrt().item())
        layer_quant_error[name] = error / max(denom, 1e-12)
        zero_fraction_by_layer[name] = float((q == 0).to(torch.float32).mean().item())
    categories = collections.defaultdict(set)
    for name, cat in meta.items():
        short = re.sub('\\.\\d+$', '', re.sub('blocks\\.\\d+', 'blocks', name))
        categories[cat].add(short)
    log('Quantized weights:')
    for cat in sorted(categories):
        log(f"  {cat}: {', '.join(sorted(categories[cat]))}")
    return (result, meta, {
        'layer_quant_error': layer_quant_error,
        'bits_by_group': bits_by_group,
        'clip_sigma_by_group': clip_sigma_by_group,
        'zero_fraction_by_layer': zero_fraction_by_layer,
    })

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

def _compress(data, compressor, brotli_quality=11):
    data = _byte_shuffle(data)
    if compressor == 'lzma':
        return lzma.compress(data, preset=6)
    elif compressor == 'brotli':
        import brotli
        return brotli.compress(data, quality=brotli_quality)
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


def _sorted_top_items(values, topk=8):
    return [{'name': name, 'value': float(value)} for name, value in sorted(values.items(), key=lambda item: item[1], reverse=True)[:topk]]


def _artifact_summary_path(h):
    return os.path.join(h.run_dir, 'artifact_summary.json')


def _load_artifact_summary(h):
    path = Path(_artifact_summary_path(h))
    if not path.exists():
        raise FileNotFoundError(f'Missing artifact summary: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def serialize(h, base_model, code):
    if h.artifact_rank0_only and h.distributed and not h.is_main_process:
        dist.barrier()
        artifact_summary = _load_artifact_summary(h)
        if perf_is_enabled():
            perf_log('artifact', rank0_only_waited=True, **artifact_summary)
        return artifact_summary

    code_bytes = len(code.encode('utf-8'))
    model_bytes = 0
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
    collect_ms = 1000.0 * (time.perf_counter() - t0)
    log(f'GPTQ:collected {len(hessians)} Hessians in {collect_ms / 1000.0:.1f}s')
    if perf_is_enabled():
        perf_log('gptq', phase='collect_hessian', collect_ms=collect_ms, calibration_batches=h.gptq_calibration_batches, num_hessians=len(hessians), **get_memory_stats_mb(device))
    quant_t0 = time.perf_counter()
    quant_result, quant_meta, quant_details = gptq_mixed_quantize(sd_cpu, hessians, h)
    quantize_ms = 1000.0 * (time.perf_counter() - quant_t0)
    if perf_is_enabled():
        perf_log('gptq', phase='quantize', quantize_ms=quantize_ms, quantized_tensors=len(quant_meta), **quant_details, **get_memory_stats_mb(device))
    quant_buf = io.BytesIO()
    torch.save({'w': quant_result, 'm': quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    compress_t0 = time.perf_counter()
    quant_blob = _compress(quant_raw, h.compressor, brotli_quality=h.brotli_quality)
    compress_ms = 1000.0 * (time.perf_counter() - compress_t0)
    quant_file_bytes = len(quant_blob)
    bytes_total = quant_file_bytes + code_bytes
    artifact_summary = {
        'artifact_bytes': bytes_total,
        'quantized_model_bytes': quant_file_bytes,
        'code_bytes': code_bytes,
        'raw_model_bytes': model_bytes,
        'quant_diagnostics_path': str(Path(h.quant_diagnostics_path).resolve()),
        'gptq_calibration_batches': h.gptq_calibration_batches,
        'gptq_calibration_time_ms': collect_ms,
        'gptq_quantize_time_ms': quantize_ms,
        'gptq_compress_time_ms': compress_ms,
        'compressor': h.compressor,
        'brotli_quality': h.brotli_quality if h.compressor == 'brotli' else None,
        'artifact_rank0_only': h.artifact_rank0_only,
        'quant_bits_by_group': quant_details['bits_by_group'],
        'quant_clip_sigma_by_group': quant_details['clip_sigma_by_group'],
        'quant_top_layer_error': _sorted_top_items(quant_details['layer_quant_error']),
        'quant_top_zero_fraction': _sorted_top_items(quant_details['zero_fraction_by_layer']),
    }
    if h.is_main_process:
        with open(h.quantized_model_path, 'wb') as f:
            f.write(quant_blob)
        log(f'Serialized model quantized+{h.compressor}: {quant_file_bytes} bytes')
        log(f'Total submission size quantized+{h.compressor}: {bytes_total} bytes')
        _json_dump(h.quant_diagnostics_path, {
            'gptq_calibration_batches': h.gptq_calibration_batches,
            'gptq_calibration_time_ms': collect_ms,
            'gptq_quantize_time_ms': quantize_ms,
            'gptq_compress_time_ms': compress_ms,
            'compressor': h.compressor,
            'brotli_quality': h.brotli_quality if h.compressor == 'brotli' else None,
            'artifact_bytes': bytes_total,
            **quant_details,
        })
        _json_dump(_artifact_summary_path(h), artifact_summary)
    if perf_is_enabled():
        perf_log('artifact', compressor=h.compressor, brotli_quality=(h.brotli_quality if h.compressor == 'brotli' else None), compress_ms=compress_ms, artifact_bytes=bytes_total, quantized_model_bytes=quant_file_bytes, code_bytes=code_bytes)
    if h.artifact_rank0_only and h.distributed:
        dist.barrier()
    return artifact_summary


def save_post_ttt_checkpoint(h, base_model):
    if not h.is_main_process:
        return None
    path = Path(h.post_ttt_checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        'model_state_dict': base_model.state_dict(),
        'metadata': {
            'run_id': h.run_id,
            'method': h.method,
            'stage': h.stage,
            'run_group': h.run_group,
            'seed': h.seed,
            'looping_active': bool(base_model.looping_active),
            'saved_at': time.time(),
        },
    }
    torch.save(checkpoint_payload, path)
    checkpoint_bytes = path.stat().st_size
    log(f'Saved post-TTT checkpoint: {checkpoint_bytes} bytes -> {path}')
    return {
        'post_ttt_checkpoint_path': str(path.resolve()),
        'post_ttt_checkpoint_bytes': checkpoint_bytes,
    }


def load_post_ttt_checkpoint(h, device):
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    state_dict = torch.load(h.post_ttt_checkpoint_path, map_location='cpu')
    checkpoint_meta = {}
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        checkpoint_meta = dict(state_dict.get('metadata') or {})
        state_dict = state_dict['model_state_dict']
    base_model.load_state_dict(state_dict, strict=True)
    base_model.looping_active = bool(checkpoint_meta.get('looping_active', False))
    return base_model, checkpoint_meta


def _load_quant_only_origin(h):
    checkpoint_path = Path(h.post_ttt_checkpoint_path)
    origin = {
        'quant_only_from_post_ttt': h.quant_only_from_post_ttt,
        'quant_only_checkpoint_path': str(checkpoint_path.resolve()),
        'quant_only_checkpoint_sha256': _sha256_file(checkpoint_path),
        'quant_only_origin_run_id': None,
        'quant_only_origin_method': None,
        'quant_only_origin_post_ttt_prequant_val_bpb': None,
        'quant_only_origin_post_quant_val_bpb': None,
    }
    run_dir = checkpoint_path.parent
    manifest_path = run_dir / 'manifest.json'
    summary_path = run_dir / 'summary.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        identity = manifest.get('identity') or {}
        origin['quant_only_origin_run_id'] = identity.get('run_id')
        origin['quant_only_origin_method'] = identity.get('method')
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        origin['quant_only_origin_post_ttt_prequant_val_bpb'] = summary.get('post_ttt_prequant_val_bpb')
        origin['quant_only_origin_post_quant_val_bpb'] = summary.get('post_quant_val_bpb')
    return origin

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


def _timed_eval_with_loop_mode(label, fn, h, device, val_data, model, loop_active):
    if not hasattr(model, 'looping_active'):
        return timed_eval(label, fn, h, device, val_data, model)
    original = bool(model.looping_active)
    model.looping_active = bool(loop_active)
    try:
        loss, bpb, extra = timed_eval(label, fn, h, device, val_data, model)
    finally:
        model.looping_active = original
    extra = dict(extra or {})
    extra['looping_active_eval'] = bool(loop_active)
    return loss, bpb, extra

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
    local_chunk_bpbs = []
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
            token_bytes = val_data.target_bytes(raw_start, raw_end - 1, tgt_ids=tgt_ids, prev_ids=prev_ids).to(device=device, dtype=torch.float64)
            batch_byte_count = token_bytes.sum()
            val_byte_count += batch_byte_count
            local_chunk_bpbs.append(_loss_bpb(batch_loss.to(torch.float64) * batch_token_count, torch.tensor(batch_token_count, device=device, dtype=torch.float64), batch_byte_count)[1])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    model.train()
    val_loss, val_bpb = _loss_bpb(val_loss_sum, val_token_count, val_byte_count)
    chunk_bpbs = _gather_chunk_values(local_chunk_bpbs) if h.perf_enabled else []
    return (val_loss, val_bpb, {
        'eval_seq_len': seq_len,
        'num_eval_windows': int(seq_end - seq_start),
        **_byte_accounting_payload(val_token_count, val_byte_count),
        **_make_chunk_bpb_payload(chunk_bpbs),
    })

def eval_val_sliding(h, device, val_data, base_model, batch_seqs=32):
    base_model.eval()
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
    local_chunk_bpbs = []
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
                logits = logits_fn(x_batch)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y_batch.reshape(-1), reduction='none').reshape(bsz, seq_len)
            batch_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
            batch_token_count = torch.zeros((), device=device, dtype=torch.float64)
            batch_byte_count = torch.zeros((), device=device, dtype=torch.float64)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else context_size
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                batch_loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                batch_token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = val_data.target_bytes(ws + s, ws + wlen, tgt_ids=tgt, prev_ids=prev).to(device=device, dtype=torch.float64)
                byte_count += tb.sum()
                batch_byte_count += tb.sum()
            if batch_token_count.item() > 0:
                local_chunk_bpbs.append(_loss_bpb(batch_loss_sum, batch_token_count, batch_byte_count)[1])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    base_model.train()
    val_loss, val_bpb = _loss_bpb(loss_sum, token_count, byte_count)
    chunk_bpbs = _gather_chunk_values(local_chunk_bpbs) if h.perf_enabled else []
    return (val_loss, val_bpb, {
        'eval_seq_len': seq_len,
        'eval_stride': h.eval_stride,
        'num_eval_windows': total_windows,
        **_byte_accounting_payload(token_count, byte_count),
        **_make_chunk_bpb_payload(chunk_bpbs),
    })

def eval_val_ttt(h, device, val_data, base_model, batch_seqs=32):
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
    if perf_is_enabled():
        perf_log('ttt_start', chunks=num_chunks, ttt_lr=h.ttt_lr, ttt_epochs=h.ttt_epochs, chunk_tokens=ttt_chunk, eval_seq_len=seq_len, eval_stride=stride)
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    local_chunk_bpbs = []
    total_score_ms = 0.0
    total_update_ms = 0.0
    total_chunk_updates = 0
    ttt_params = [p for p in base_model.parameters()]
    for p in ttt_params:
        p.requires_grad_(True)
    optimizer = torch.optim.SGD(ttt_params, lr=h.ttt_lr, momentum=h.ttt_momentum)
    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
        chunk_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        chunk_token_count = torch.zeros((), device=device, dtype=torch.float64)
        chunk_byte_count = torch.zeros((), device=device, dtype=torch.float64)
        my_s = len(windows) * rank // world_size
        my_e = len(windows) * (rank + 1) // world_size
        my_windows = windows[my_s:my_e]
        base_model.eval()
        if perf_is_enabled() and _perf_peak_tracker is not None:
            _perf_peak_tracker.reset_interval()
        score_t0 = time.perf_counter()
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
                    logits = compiled_logits(x_batch)
                nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y_batch.reshape(-1), reduction='none').reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else context_size
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    chunk_loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    chunk_token_count += float(wlen - s)
                    tgt = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb = val_data.target_bytes(ws + s, ws + wlen, tgt_ids=tgt, prev_ids=prev).to(device=device, dtype=torch.float64)
                    byte_count += tb.sum()
                    chunk_byte_count += tb.sum()
        torch.cuda.synchronize()
        score_ms = 1000.0 * (time.perf_counter() - score_t0)
        total_score_ms += score_ms
        chunk_val_loss, chunk_bpb = ((float('nan'), float('nan')) if chunk_token_count.item() == 0 else _loss_bpb(chunk_loss_sum, chunk_token_count, chunk_byte_count))
        if chunk_token_count.item() > 0:
            local_chunk_bpbs.append(chunk_bpb)
        is_last_chunk = ci == num_chunks - 1
        update_ms = 0.0
        grad_norm_value = 0.0
        clip_triggered = False
        if not is_last_chunk and h.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = h.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg['lr'] = cos_lr
                my_seq_s = chunk_seqs * rank // world_size
                my_seq_e = chunk_seqs * (rank + 1) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                update_t0 = time.perf_counter()
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
                        grad_norm = torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
                        grad_norm_value = max(grad_norm_value, float(grad_norm))
                        clip_triggered = clip_triggered or float(grad_norm) > 1.0
                        optimizer.step()
                torch.cuda.synchronize()
                update_ms = 1000.0 * (time.perf_counter() - update_t0)
                total_update_ms += update_ms
                total_chunk_updates += h.ttt_epochs
        if perf_is_enabled() and (ci < 3 or (h.perf_ttt_chunk_every > 0 and ci % h.perf_ttt_chunk_every == 0) or is_last_chunk):
            running_bpb = (float('nan') if token_count.item() == 0 else _loss_bpb(loss_sum, token_count, byte_count)[1])
            perf_log(
                'ttt_chunk',
                chunk_id=ci,
                chunk_tokens=chunk_end - chunk_start,
                num_windows=len(windows),
                score_before_update=True,
                single_left_to_right_pass=True,
                strict_causal_prefix_only=True,
                per_chunk_update_count=(0 if is_last_chunk else h.ttt_epochs),
                score_ms=score_ms,
                ttt_update_ms=update_ms,
                bpb_before_update=chunk_bpb,
                bpb_running_avg=running_bpb,
                ttt_grad_norm=grad_norm_value,
                ttt_clip_triggered=clip_triggered,
                **get_memory_stats_mb(device),
            )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()
    val_loss, val_bpb = _loss_bpb(loss_sum, token_count, byte_count)
    return (val_loss, val_bpb, {
        'eval_seq_len': seq_len,
        'eval_stride': h.eval_stride,
        'num_eval_windows': len(window_starts),
        'ttt_chunks': num_chunks,
        'ttt_epochs': h.ttt_epochs,
        'ttt_chunk_tokens': h.ttt_chunk_tokens,
        'ttt_score_before_update': True,
        'ttt_single_left_to_right_pass': True,
        'ttt_strict_causal_prefix_only': True,
        'ttt_score_after_adapt': False,
        'ttt_multi_pass_rescore': False,
        'ttt_total_score_ms': total_score_ms,
        'ttt_total_update_ms': total_update_ms,
        'ttt_total_wall_time_ms': total_score_ms + total_update_ms,
        'ttt_total_chunk_updates': total_chunk_updates,
        **_byte_accounting_payload(token_count, byte_count),
        **_make_chunk_bpb_payload(_gather_chunk_values(local_chunk_bpbs) if h.perf_enabled else []),
    })


def _select_pre_quant_ttt_params(base_model, freeze_blocks):
    if freeze_blocks < 0:
        raise ValueError(f'PRE_QUANT_TTT_FREEZE_BLOCKS must be >= 0, got {freeze_blocks}')
    if freeze_blocks > len(base_model.blocks):
        raise ValueError(f'PRE_QUANT_TTT_FREEZE_BLOCKS={freeze_blocks} exceeds num_blocks={len(base_model.blocks)}')
    trainable = []
    frozen_prefix = tuple(f'blocks.{idx}.' for idx in range(freeze_blocks))
    for name, param in base_model.named_parameters():
        should_train = not any((name.startswith(prefix) for prefix in frozen_prefix))
        param.requires_grad_(should_train)
        if should_train:
            trainable.append((name, param))
    return trainable


def _anchor_param_entries(selected):
    entries = []
    for name, param in selected:
        if param.ndim != 2:
            continue
        if classify_param(name) not in {'attn', 'mlp'}:
            continue
        base = param.detach().float().clone()
        denom = base.pow(2).sum().clamp_min(1e-12)
        entries.append({
            'name': name,
            'param': param,
            'base': base,
            'denom': denom,
        })
    return entries


def _qnoise_param_entries(selected):
    return [
        {'name': name, 'param': param}
        for name, param in selected
        if param.ndim == 2 and classify_param(name) in {'attn', 'mlp'}
    ]


def _apply_pre_quant_qnoise(h, qnoise_entries):
    applied = []
    if not h.pre_quant_ttt_qnoise_enabled or h.pre_quant_ttt_qnoise_prob <= 0.0:
        return applied
    for entry in qnoise_entries:
        param = entry['param']
        rows = param.shape[0]
        if rows <= 0:
            continue
        row_mask = (torch.rand((rows, 1), device=param.device) < h.pre_quant_ttt_qnoise_prob).to(param.dtype)
        if not bool(row_mask.any().item()):
            continue
        row_scale = param.detach().float().std(dim=1, keepdim=True, correction=0)
        row_scale = row_scale.mul(h.matrix_clip_sigmas * h.pre_quant_ttt_qnoise_scale_mult).to(dtype=param.dtype)
        noise = (torch.rand_like(param) * 2.0 - 1.0) * row_scale * row_mask
        param.data.add_(noise)
        applied.append((param, noise))
    return applied


def run_pre_quant_ttt(h, device, val_data, base_model):
    if not h.pre_quant_ttt_enabled or h.pre_quant_ttt_epochs <= 0:
        return None
    if (not val_data.pre_quant_ttt_segments) or val_data.pre_quant_ttt_selection is None:
        raise ValueError('pre-quant TTT requires a dedicated adaptation validation slice')
    seq_len = h.eval_seq_len
    local_batch_tokens = h.val_batch_tokens // h.world_size
    if local_batch_tokens < seq_len:
        raise ValueError(f'VAL_BATCH_SIZE must provide at least one sequence per rank for pre-quant TTT; got VAL_BATCH_SIZE={h.val_batch_tokens}, WORLD_SIZE={h.world_size}, seq_len={seq_len}')
    batch_seqs = local_batch_tokens // seq_len
    selected = _select_pre_quant_ttt_params(base_model, h.pre_quant_ttt_freeze_blocks)
    selected_params = [param for _, param in selected]
    trainable_param_count = sum((int(param.numel()) for param in selected_params))
    anchor_entries = (_anchor_param_entries(selected) if h.pre_quant_ttt_anchor_lambda > 0 else [])
    qnoise_entries = (_qnoise_param_entries(selected) if h.pre_quant_ttt_qnoise_enabled else [])
    original_looping_active = bool(base_model.looping_active)
    if h.pre_quant_ttt_loop_mode == 'active' and h.num_loops > 0:
        base_model.looping_active = True
    log(f'pre_quant_ttt:start epochs={h.pre_quant_ttt_epochs} lr={h.pre_quant_ttt_lr:.6f} min_lr={h.pre_quant_ttt_min_lr:.6f} trainable_params={trainable_param_count}')
    log_event(
        'pre_quant_ttt_start',
        ttt_source=h.pre_quant_ttt_source,
        ttt_files=h.pre_quant_ttt_files,
        ttt_shard_limit=h.pre_quant_ttt_shard_limit,
        ttt_selection_mode=h.pre_quant_ttt_selection_mode,
        ttt_epochs=h.pre_quant_ttt_epochs,
        ttt_lr=h.pre_quant_ttt_lr,
        ttt_min_lr=h.pre_quant_ttt_min_lr,
        ttt_weight_decay=h.pre_quant_ttt_weight_decay,
        ttt_grad_clip_norm=h.pre_quant_ttt_grad_clip_norm,
        ttt_freeze_blocks=h.pre_quant_ttt_freeze_blocks,
        ttt_anchor_lambda=h.pre_quant_ttt_anchor_lambda,
        ttt_qnoise_enabled=h.pre_quant_ttt_qnoise_enabled,
        ttt_trainable_param_count=trainable_param_count,
        ttt_val_selection=val_data.pre_quant_ttt_selection,
        ttt_loop_mode=h.pre_quant_ttt_loop_mode,
        ttt_looping_active_start=original_looping_active,
    )
    optimizer = torch.optim.AdamW(
        [{'params': selected_params, 'lr': h.pre_quant_ttt_lr}],
        lr=h.pre_quant_ttt_lr,
        betas=(h.beta1, h.beta2),
        eps=h.adam_eps,
        weight_decay=h.pre_quant_ttt_weight_decay,
        fused=True,
    )
    epoch_stats = []
    for epoch_idx in range(h.pre_quant_ttt_epochs):
        if h.pre_quant_ttt_epochs == 1:
            lr = h.pre_quant_ttt_lr
        else:
            frac = epoch_idx / max(h.pre_quant_ttt_epochs - 1, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * frac))
            lr = h.pre_quant_ttt_min_lr + (h.pre_quant_ttt_lr - h.pre_quant_ttt_min_lr) * cosine
        for group in optimizer.param_groups:
            group['lr'] = lr
        base_model.train()
        epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        epoch_token_count = torch.zeros((), device=device, dtype=torch.float64)
        epoch_byte_count = torch.zeros((), device=device, dtype=torch.float64)
        grad_norms = []
        clip_count = 0
        step_count = 0
        epoch_t0 = time.perf_counter()
        for segment in val_data.pre_quant_ttt_segments:
            segment_tokens = segment['tokens']
            total_seqs = (segment_tokens.numel() - 1) // seq_len
            seq_start = total_seqs * h.rank // h.world_size
            seq_end = total_seqs * (h.rank + 1) // h.world_size
            local_batch_count = math.ceil(max(seq_end - seq_start, 0) / batch_seqs) if batch_seqs > 0 else 0
            if dist.is_available() and dist.is_initialized():
                max_batch_count_tensor = torch.tensor(local_batch_count, device=device, dtype=torch.int32)
                dist.all_reduce(max_batch_count_tensor, op=dist.ReduceOp.MAX)
                max_batch_count = int(max_batch_count_tensor.item())
            else:
                max_batch_count = local_batch_count
            for batch_idx in range(max_batch_count):
                batch_seq_start = seq_start + batch_idx * batch_seqs
                batch_seq_end = min(batch_seq_start + batch_seqs, seq_end)
                has_batch = batch_seq_start < seq_end
                optimizer.zero_grad(set_to_none=True)
                if has_batch:
                    raw_start = batch_seq_start * seq_len
                    raw_end = batch_seq_end * seq_len + 1
                    local = segment_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
                    x = local[:-1].reshape(-1, seq_len)
                    y = local[1:].reshape(-1, seq_len)
                    applied_qnoise = []
                    if h.pre_quant_ttt_qnoise_enabled and epoch_idx + 1 >= h.pre_quant_ttt_qnoise_start_epoch:
                        applied_qnoise = _apply_pre_quant_qnoise(h, qnoise_entries)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                        loss = base_model(x, y)
                        if anchor_entries:
                            anchor_penalty = torch.zeros((), device=device, dtype=torch.float32)
                            for entry in anchor_entries:
                                diff = entry['param'].float() - entry['base']
                                anchor_penalty = anchor_penalty + diff.pow(2).sum() / entry['denom']
                            loss = loss + h.pre_quant_ttt_anchor_lambda * anchor_penalty
                    loss.backward()
                    for param, noise in applied_qnoise:
                        param.data.sub_(noise)
                    local_token_weight = float(y.numel())
                else:
                    local_token_weight = 0.0
                if h.world_size > 1:
                    token_weight_tensor = torch.tensor(local_token_weight, device=device, dtype=torch.float64)
                    dist.all_reduce(token_weight_tensor, op=dist.ReduceOp.SUM)
                    global_token_weight = max(float(token_weight_tensor.item()), 1.0)
                    for param in selected_params:
                        if param.grad is not None:
                            param.grad.mul_(local_token_weight)
                        else:
                            param.grad = torch.zeros_like(param)
                        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                        param.grad.div_(global_token_weight)
                grad_norm = torch.nn.utils.clip_grad_norm_(selected_params, h.pre_quant_ttt_grad_clip_norm)
                grad_norm_value = float(grad_norm)
                grad_norms.append(grad_norm_value)
                clip_count += int(grad_norm_value > h.pre_quant_ttt_grad_clip_norm)
                optimizer.step()
                if has_batch:
                    batch_token_count = float(y.numel())
                    batch_loss_sum = loss.detach().to(torch.float64) * batch_token_count
                    epoch_loss_sum += batch_loss_sum
                    epoch_token_count += batch_token_count
                    prev_ids = x.reshape(-1)
                    tgt_ids = y.reshape(-1)
                    batch_byte_count = val_data._lookup_target_bytes(segment.get('token_bytes'), raw_start, raw_end - 1, tgt_ids=tgt_ids, prev_ids=prev_ids).to(device=device, dtype=torch.float64).sum()
                    epoch_byte_count += batch_byte_count
                step_count += 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(epoch_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_token_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_byte_count, op=dist.ReduceOp.SUM)
        epoch_loss, epoch_bpb = _loss_bpb(epoch_loss_sum, epoch_token_count, epoch_byte_count)
        epoch_wall_ms = 1000.0 * (time.perf_counter() - epoch_t0)
        epoch_payload = {
            'ttt_epoch': epoch_idx + 1,
            'ttt_epoch_lr': lr,
            'ttt_epoch_loss': epoch_loss,
            'ttt_epoch_bpb': epoch_bpb,
            'ttt_trainable_param_count': trainable_param_count,
            'ttt_wall_ms': epoch_wall_ms,
            'ttt_grad_norm_p50': _quantile(grad_norms, 0.50),
            'ttt_grad_norm_p95': _quantile(grad_norms, 0.95),
            'ttt_grad_norm_max': (max(grad_norms) if grad_norms else None),
            'ttt_clip_rate': (clip_count / step_count if step_count > 0 else None),
            'ttt_optimizer': 'adamw',
            'ttt_weight_decay': h.pre_quant_ttt_weight_decay,
            'ttt_selection_mode': h.pre_quant_ttt_selection_mode,
            'ttt_anchor_lambda': h.pre_quant_ttt_anchor_lambda,
            'ttt_qnoise_enabled': h.pre_quant_ttt_qnoise_enabled,
            'ttt_loop_mode': h.pre_quant_ttt_loop_mode,
            'ttt_looping_active': bool(base_model.looping_active),
            **_byte_accounting_payload(epoch_token_count, epoch_byte_count),
        }
        epoch_stats.append(epoch_payload)
        log_event('pre_quant_ttt_epoch', **epoch_payload)
        if perf_is_enabled():
            perf_log('pre_quant_ttt_epoch', **epoch_payload, **get_memory_stats_mb(device))
    base_model.eval()
    return {
        'ttt_optimizer': 'adamw',
        'ttt_trainable_param_count': trainable_param_count,
        'ttt_epochs': h.pre_quant_ttt_epochs,
        'ttt_lr': h.pre_quant_ttt_lr,
        'ttt_min_lr': h.pre_quant_ttt_min_lr,
        'ttt_weight_decay': h.pre_quant_ttt_weight_decay,
        'ttt_grad_clip_norm': h.pre_quant_ttt_grad_clip_norm,
        'ttt_freeze_blocks': h.pre_quant_ttt_freeze_blocks,
        'ttt_source': h.pre_quant_ttt_source,
        'ttt_files': h.pre_quant_ttt_files,
        'ttt_shard_limit': h.pre_quant_ttt_shard_limit,
        'ttt_selection_mode': h.pre_quant_ttt_selection_mode,
        'ttt_loop_mode': h.pre_quant_ttt_loop_mode,
        'ttt_looping_active_start': original_looping_active,
        'ttt_looping_active_end': bool(base_model.looping_active),
        'ttt_anchor_lambda': h.pre_quant_ttt_anchor_lambda,
        'ttt_qnoise_enabled': h.pre_quant_ttt_qnoise_enabled,
        'ttt_val_selection': val_data.pre_quant_ttt_selection,
        'ttt_epoch_stats': epoch_stats,
        'ttt_wall_ms': sum((row['ttt_wall_ms'] for row in epoch_stats)),
    }

def timed_eval(label, fn, *args, **kwargs):
    device = args[1]
    if perf_is_enabled() and _perf_peak_tracker is not None:
        _perf_peak_tracker.reset_interval()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    val_loss, val_bpb, extra = fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    log(f'{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms')
    eval_tokens = extra.get('eval_tokens', 0.0)
    eval_toks_per_sec = eval_tokens / max(elapsed_ms / 1000.0, 1e-9)
    timing = {
        'label': label,
        'eval_time_ms': elapsed_ms,
        'eval_tokens_per_sec': eval_toks_per_sec,
        **extra,
    }
    if perf_is_enabled():
        perf_log('eval', val_loss=val_loss, val_bpb=val_bpb, **timing, **get_memory_stats_mb(device))
    return (val_loss, val_bpb, timing)

def train_model(h, device, val_data):
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    if h.distributed:
        model = DDP(compiled_model, device_ids=[h.local_rank], broadcast_buffers=False)
    else:
        model = compiled_model
    ddp_comm_state = None
    if h.distributed and h.perf_enabled:
        ddp_comm_state = DDPCommTimingState()
        model.register_comm_hook(state=ddp_comm_state, hook=timed_allreduce_hook)
    profiler_controller = ProfilerController(h)
    perf_window = WindowStats(max(h.perf_window_size, 1)) if perf_is_enabled() else None
    loop_enabled_step = None
    loop_pre_snapshot = None
    loop_compare_emitted = False
    log(f'model_params:{sum((p.numel() for p in base_model.parameters()))}')
    optimizers = Optimizers(h, base_model)
    train_loader = ShuffledSequenceLoader(h, device)
    max_wallclock_ms = 1000.0 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    if max_wallclock_ms is not None:
        max_wallclock_ms -= h.gptq_reserve_seconds * 1000.0
        log(f'gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms')

    def training_frac(step, elapsed_ms):
        if max_wallclock_ms is None:
            return step / max(h.iterations, 1)
        return elapsed_ms / max(max_wallclock_ms, 1e-09)

    def lr_mul(frac):
        if h.warmdown_frac <= 0:
            return 1.0
        if frac >= 1.0 - h.warmdown_frac:
            return max((1.0 - frac) / h.warmdown_frac, h.min_lr)
        return 1.0
    ema_state = None
    ema_decay = h.ema_decay
    def step_fn(step, lr_scale, sample_step):
        optimizers.zero_grad_all()
        train_loss = torch.zeros((), device=device)
        step_wall_t0 = time.perf_counter()
        timer = SampledStepTimer(sample_step)
        forward_names = []
        backward_names = []
        optimizer_names = []
        data_ms = 0.0
        backward_wall_ms = 0.0
        if ddp_comm_state is not None:
            ddp_comm_state.start_step(sample_step)
        if sample_step and _perf_peak_tracker is not None:
            _perf_peak_tracker.reset_interval()
        grouped_snapshots = _clone_grouped_params(optimizers.grouped_params) if sample_step else None
        for micro_step in range(h.grad_accum_steps):
            if h.distributed:
                model.require_backward_grad_sync = micro_step == h.grad_accum_steps - 1
            data_t0 = time.perf_counter()
            x, y = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            data_ms += 1000.0 * (time.perf_counter() - data_t0)
            fwd_name = f'forward_{micro_step}'
            timer.start(fwd_name)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            timer.stop(fwd_name)
            forward_names.append(fwd_name)
            train_loss += loss.detach()
            bwd_name = f'backward_{micro_step}'
            timer.start(bwd_name)
            backward_wall_t0 = time.perf_counter()
            (loss / h.grad_accum_steps).backward()
            backward_wall_ms += 1000.0 * (time.perf_counter() - backward_wall_t0)
            timer.stop(bwd_name)
            backward_names.append(bwd_name)
        train_loss /= h.grad_accum_steps
        frac = min(step / h.muon_momentum_warmup_steps, 1.0) if h.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * h.muon_momentum_warmup_start + frac * h.muon_momentum
        for group in optimizers.optimizer_muon.param_groups:
            group['momentum'] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group['lr'] = group['base_lr'] * lr_scale
        grad_norm_by_group = _group_tensor_norms(optimizers.grouped_params, grad=True) if sample_step else None
        grad_norm_value = None
        grad_clip_triggered = False
        if h.grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
            grad_norm_value = float(grad_norm)
            grad_clip_triggered = float(grad_norm) > h.grad_clip_norm

        def timed_optimizer_step(name, fn):
            section = f'optimizer_{name}'
            timer.start(section)
            fn()
            timer.stop(section)
            optimizer_names.append(section)

        optimizers.step(timed_step=timed_optimizer_step if sample_step else None)
        update_norm_by_group = (_group_delta_norms(optimizers.grouped_params, grouped_snapshots) if sample_step else None)
        if ema_state is not None:
            timer.start('ema')
            with torch.no_grad():
                for name, t in base_model.state_dict().items():
                    ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)
            timer.stop('ema')
        torch.cuda.synchronize()
        step_ms = 1000.0 * (time.perf_counter() - step_wall_t0)
        metric = {
            'step_ms': step_ms,
            'tok_s_step': h.train_batch_tokens / max(step_ms / 1000.0, 1e-9),
        }
        if sample_step:
            timer.finalize()
            metric.update({
                'data_ms': data_ms,
                'forward_ms': sum((timer.elapsed_ms(name) for name in forward_names)),
                'backward_ms': sum((timer.elapsed_ms(name) for name in backward_names)),
                'backward_compute_ms': sum((timer.elapsed_ms(name) for name in backward_names)),
                'optimizer_ms': sum((timer.elapsed_ms(name) for name in optimizer_names)),
                'ema_ms': (timer.elapsed_ms('ema') if ema_state is not None else 0.0),
            })
            for name, _ in optimizers.named_optimizers:
                section = f'optimizer_{name}'
                if section in optimizer_names:
                    metric[f'{name}_ms'] = timer.elapsed_ms(section)
            if ddp_comm_state is not None:
                metric.update(ddp_comm_state.snapshot())
            metric['comm_wait_ms'] = max(backward_wall_ms - metric['backward_ms'], 0.0)
            step_info = {
                'grad_norm_total': grad_norm_value,
                'grad_clip_triggered': grad_clip_triggered,
                'grad_norm_by_group': grad_norm_by_group,
                'update_norm_by_group': update_norm_by_group,
                'weight_norm_by_group': _group_tensor_norms(optimizers.grouped_params, grad=False),
                'lr_by_group': get_lr_by_group(optimizers),
                'muon_momentum': muon_momentum,
                **get_memory_stats_mb(device),
            }
            return (train_loss, metric, step_info)
        return (train_loss, metric, None)
    if h.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(h.warmup_steps):
            step_fn(warmup_step, 1.0, sample_step=False)
            if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                log(f'warmup_step: {warmup_step + 1}/{h.warmup_steps}')
        if h.num_loops > 0:
            base_model.looping_active = True
            log(f'loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}')
            for warmup_step in range(h.warmup_steps):
                step_fn(warmup_step, 1.0, sample_step=False)
                if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                    log(f'loop_warmup_step: {warmup_step + 1}/{h.warmup_steps}')
            base_model.looping_active = False
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        optimizers.zero_grad_all()
        if h.distributed:
            model.require_backward_grad_sync = True
        train_loader = ShuffledSequenceLoader(h, device)
    ema_state = ({name: t.detach().float().clone() for name, t in base_model.state_dict().items()} if h.ema_enabled else None)
    training_time_ms = 0.0
    stop_after_step = None
    last_train_loss_value = None
    last_val_loss = None
    last_val_bpb = None
    last_tokens_per_sec = None
    last_step_ms = None
    last_rolling_tokens_per_sec = None
    total_train_step_ms = 0.0
    total_checkpoint_eval_ms = 0.0
    rolling_step_ms = collections.deque(maxlen=max(h.perf_window_size, 1))

    def current_peak_mib():
        if _perf_peak_tracker is not None:
            return int(_perf_peak_tracker.run_peaks_mb().get('run_peak_alloc_mb', 0))
        return int(torch.cuda.max_memory_allocated(device) // 1024 // 1024)

    def fail_run(reason, **details):
        log_event('failure', reason=reason, step=step, **details)
        raise RunFailure(reason, step=step, **details)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == h.iterations or (stop_after_step is not None and step >= stop_after_step)
        validation_reasons = []
        if last_step:
            validation_reasons.append('final')
        if h.val_loss_every > 0 and step % h.val_loss_every == 0:
            validation_reasons.append('cadence')
        if step in h.eval_at_steps:
            validation_reasons.append('eval_at_step')
        if validation_reasons:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            eval_t0 = time.perf_counter()
            val_loss, val_bpb, val_extra = eval_val(h, device, val_data, model)
            torch.cuda.synchronize()
            eval_ms = 1000.0 * (time.perf_counter() - eval_t0)
            total_checkpoint_eval_ms += eval_ms
            last_val_loss = val_loss
            last_val_bpb = val_bpb
            log(f'{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}')
            log_event(
                'val',
                step=step,
                val_loss=val_loss,
                val_bpb=val_bpb,
                elapsed_sec=training_time_ms / 1000.0,
                eval_time_ms=eval_ms,
                tokens_seen=step * h.train_batch_tokens,
                reason=','.join(dict.fromkeys(validation_reasons)),
                val_mode=('proxy' if h.local_val_tokens > 0 else 'full'),
                val_token_limit=h.local_val_tokens,
                eval_tokens_per_sec=(val_extra.get('eval_tokens', 0.0) / max(eval_ms / 1000.0, 1e-9)),
                model_state='checkpoint_raw',
                **val_extra,
            )
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < h.iterations:
                log(f'stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms step: {step}/{h.iterations}')
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        frac = training_frac(step, elapsed_ms)
        scale = lr_mul(frac)
        if h.num_loops > 0 and (not base_model.looping_active) and (frac >= h.enable_looping_at):
            base_model.looping_active = True
            loop_enabled_step = step
            loop_pre_snapshot = {
                'step': step,
                'train_loss': last_train_loss_value,
                **(perf_window.summary() if perf_window is not None else {}),
            }
            log(f'layer_loop:enabled step:{step} frac:{frac:.3f} encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}')
            if perf_is_enabled():
                perf_log('loop_event', step=step, enabled=True, loop_frac=frac, encoder=base_model.encoder_indices, decoder=base_model.decoder_indices, virtual_layer_count=len(base_model.encoder_indices) + len(base_model.decoder_indices), **({f'pre_{k}': v for k, v in loop_pre_snapshot.items() if v is not None}))
        profiler_controller.maybe_start(step)
        sample_step = _should_sample_step(h, step, loop_enabled_step) or _should_log_perf_step(h, step + 1)
        train_loss, step_metric, step_info = step_fn(step, scale, sample_step=sample_step)
        last_train_loss_value = float(train_loss.item())
        step += 1
        profiler_controller.step(step)
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        tokens_seen = step * h.train_batch_tokens
        tok_per_sec = tokens_seen / max(approx_training_time_ms / 1000.0, 1e-9)
        total_train_step_ms += step_metric['step_ms']
        last_step_ms = step_metric['step_ms']
        rolling_step_ms.append(step_metric['step_ms'])
        rolling_tokens_per_sec = (h.train_batch_tokens * len(rolling_step_ms)) / max(sum(rolling_step_ms) / 1000.0, 1e-9)
        last_tokens_per_sec = tok_per_sec
        last_rolling_tokens_per_sec = rolling_tokens_per_sec
        should_log_train = h.train_log_every > 0 and (step <= 5 or step % h.train_log_every == 0 or stop_after_step is not None)
        if should_log_train:
            log(f'{step}/{h.iterations} train_loss: {train_loss.item():.4f} train_time: {approx_training_time_ms / 60000:.1f}m tok/s: {tok_per_sec:.0f} rolling_tok/s: {rolling_tokens_per_sec:.0f} step_ms: {step_metric["step_ms"]:.1f}')
        event_payload = {
            'step': step,
            'train_loss': float(train_loss.item()),
            'elapsed_sec': approx_training_time_ms / 1000.0,
            'tokens_seen': tokens_seen,
            'tokens_per_sec': tok_per_sec,
            'rolling_tokens_per_sec': rolling_tokens_per_sec,
            'step_ms': step_metric['step_ms'],
            'step_tokens_per_sec': step_metric['tok_s_step'],
            'loop_enabled': base_model.looping_active,
            'lr_scale': scale,
            'peak_memory_mib': current_peak_mib(),
        }
        if step_metric is not None and perf_window is not None:
            perf_window.add(**step_metric)
            event_payload.update(step_metric)
        if step_info is not None:
            if step_info.get('grad_norm_total') is not None:
                event_payload['grad_norm_total'] = step_info['grad_norm_total']
            if 'mem_peak_alloc_mb' in step_info:
                event_payload['peak_memory_mib'] = step_info['mem_peak_alloc_mb']
        log_event('train_step', **event_payload)
        if h.abort_on_nan and not math.isfinite(float(train_loss.item())):
            fail_run('nan_loss', train_loss=float(train_loss.item()))
        if h.max_peak_mem_mib > 0 and event_payload['peak_memory_mib'] > h.max_peak_mem_mib:
            fail_run('oom_guard', peak_memory_mib=event_payload['peak_memory_mib'], max_peak_mem_mib=h.max_peak_mem_mib)
        if h.min_tokens_per_sec > 0 and step >= 5 and tok_per_sec < h.min_tokens_per_sec:
            fail_run('slow_guard', tokens_per_sec=tok_per_sec, min_tokens_per_sec=h.min_tokens_per_sec)
        if step_metric is not None and _should_log_perf_step(h, step):
            summary = perf_window.summary()
            rank_step_ms_min = rank_step_ms_max = step_metric['step_ms']
            per_rank_mem = [{'rank': h.rank, 'alloc_mb': step_info['mem_alloc_mb'], 'reserved_mb': step_info['mem_reserved_mb'], 'peak_alloc_mb': step_info['mem_peak_alloc_mb']}]
            if h.perf_enabled and h.distributed:
                rank_step_ms_min, rank_step_ms_max = _gather_rank_range(step_metric['step_ms'], device)
                per_rank_mem = _gather_per_rank_memory(device, step_info)
            perf_log(
                'train_perf',
                step=step,
                train_loss=float(train_loss.item()),
                train_time_ms=approx_training_time_ms,
                tok_s_global=tokens_seen / max(approx_training_time_ms / 1000.0, 1e-9),
                loop_enabled=base_model.looping_active,
                virtual_layer_count=(len(base_model.encoder_indices) + len(base_model.decoder_indices)) if base_model.looping_active else h.num_layers,
                rank_step_ms_min=rank_step_ms_min,
                rank_step_ms_max=rank_step_ms_max,
                rank_skew_ms=rank_step_ms_max - rank_step_ms_min,
                per_rank_mem=per_rank_mem,
                **summary,
                **step_info,
            )
            if perf_is_enabled() and base_model.looping_active and loop_pre_snapshot is not None and not loop_compare_emitted:
                perf_log(
                    'loop_compare',
                    loop_enabled_step=loop_enabled_step,
                    loss_delta_after_loop=(loop_pre_snapshot.get('train_loss') - float(train_loss.item())) if loop_pre_snapshot.get('train_loss') is not None else None,
                    grad_norm_after_loop=step_info.get('grad_norm_total'),
                    mem_peak_after_loop=step_info.get('mem_peak_alloc_mb'),
                    forward_ms_pre_loop=loop_pre_snapshot.get('forward_ms_avg'),
                    forward_ms_post_loop=summary.get('forward_ms_avg'),
                    backward_ms_pre_loop=loop_pre_snapshot.get('backward_ms_avg'),
                    backward_ms_post_loop=summary.get('backward_ms_avg'),
                    step_ms_pre_loop=loop_pre_snapshot.get('step_ms_avg'),
                    step_ms_post_loop=summary.get('step_ms_avg'),
                )
                loop_compare_emitted = True
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if h.distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    profiler_controller.close()
    peak_stats = _perf_peak_tracker.run_peaks_mb() if _perf_peak_tracker is not None else {
        'run_peak_alloc_mb': torch.cuda.max_memory_allocated(device) // 1024 // 1024,
        'run_peak_reserved_mb': torch.cuda.max_memory_reserved(device) // 1024 // 1024,
    }
    log(f"peak memory allocated: {peak_stats['run_peak_alloc_mb']} MiB reserved: {peak_stats['run_peak_reserved_mb']} MiB")
    if ema_state is not None:
        log('ema:applying EMA weights')
        current_state = base_model.state_dict()
        avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
        base_model.load_state_dict(avg_state, strict=True)
    else:
        log('ema:disabled')
    training_summary = {
        'actual_steps': step,
        'elapsed_sec': training_time_ms / 1000.0,
        'tokens_seen': step * h.train_batch_tokens,
        'tokens_per_sec': last_tokens_per_sec,
        'rolling_tokens_per_sec': last_rolling_tokens_per_sec,
        'last_step_ms': last_step_ms,
        'train_step_time_ms_total': total_train_step_ms,
        'checkpoint_eval_time_ms_total': total_checkpoint_eval_ms,
        'final_train_loss': last_train_loss_value,
        'last_val_loss': last_val_loss,
        'last_val_bpb': last_val_bpb,
        'checkpoint_step': step,
        'checkpoint_val_loss': last_val_loss,
        'checkpoint_val_bpb': last_val_bpb,
        'ema_enabled': h.ema_enabled,
        'ema_applied': ema_state is not None,
        **peak_stats,
    }
    log_event('training_summary', **training_summary)
    return (base_model, compiled_model, training_summary)

def train_and_eval(h, device):
    run_wallclock_t0 = time.perf_counter()
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    script_code = Path(__file__).read_text(encoding='utf-8')
    train_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
    if h.train_shard_limit > 0:
        train_files = train_files[:h.train_shard_limit]
    val_data = ValidationData(h, device)
    write_manifest(h, device, script_code, {
        'data': {
            'tokenizer_sha256': val_data.tokenizer_sha256,
            'tokenizer_vocab_size': val_data.vocab_size,
            'validation_selection': val_data.selection,
            'tokenizer_kind': val_data.tokenizer_kind,
            'caseops_roundtrip_ok': val_data.caseops_roundtrip_ok,
            'caseops_raw_utf8_bytes': val_data.caseops_raw_utf8_bytes,
            'caseops_sidecar_utf8_bytes': val_data.caseops_sidecar_utf8_bytes,
            'caseops_zero_byte_token_count': val_data.caseops_zero_byte_token_count,
            'pre_quant_ttt_selection': val_data.pre_quant_ttt_selection,
        },
    })
    log(f'train_shards: {len(train_files)}')
    log(f'val_tokens: {val_data.val_tokens.numel() - 1}')
    log_event(
        'run_start',
        hostname=socket.gethostname(),
        gpu_name=torch.cuda.get_device_name(device),
        world_size=h.world_size,
        train_shard_count=len(train_files),
        train_shard_names=[_shard_name(file) for file in train_files],
        val_tokens=val_data.val_tokens.numel() - 1,
        tokenizer_path=val_data.tokenizer_path,
        tokenizer_sha256=val_data.tokenizer_sha256,
        tokenizer_vocab_size=val_data.vocab_size,
        tokenizer_kind=val_data.tokenizer_kind,
        val_shards=val_data.val_files,
        val_selection=val_data.selection,
        pre_quant_ttt_source=h.pre_quant_ttt_source,
        pre_quant_ttt_files=h.pre_quant_ttt_files,
        pre_quant_ttt_selection_mode=h.pre_quant_ttt_selection_mode,
        pre_quant_ttt_selection=val_data.pre_quant_ttt_selection,
        caseops_roundtrip_ok=val_data.caseops_roundtrip_ok,
        caseops_raw_utf8_bytes=val_data.caseops_raw_utf8_bytes,
        caseops_sidecar_utf8_bytes=val_data.caseops_sidecar_utf8_bytes,
        caseops_zero_byte_token_count=val_data.caseops_zero_byte_token_count,
    )
    quant_only_origin = {}
    if h.quant_only_from_post_ttt:
        base_model, _ = load_post_ttt_checkpoint(h, device)
        compiled_model = None
        quant_only_origin = _load_quant_only_origin(h)
        training_summary = {
            'actual_steps': 0,
            'elapsed_sec': 0.0,
            'tokens_seen': 0,
            'tokens_per_sec': None,
            'rolling_tokens_per_sec': None,
            'last_step_ms': None,
            'train_step_time_ms_total': 0.0,
            'checkpoint_eval_time_ms_total': 0.0,
            'final_train_loss': None,
            'last_val_loss': None,
            'last_val_bpb': None,
            'checkpoint_step': 0,
            'checkpoint_val_loss': None,
            'checkpoint_val_bpb': None,
            'ema_enabled': h.ema_enabled,
            'ema_applied': False,
            'run_peak_alloc_mb': torch.cuda.max_memory_allocated(device) // 1024 // 1024,
            'run_peak_reserved_mb': torch.cuda.max_memory_reserved(device) // 1024 // 1024,
        }
        log('quant_only_from_post_ttt:loaded adapted checkpoint')
    else:
        base_model, compiled_model, training_summary = train_model(h, device, val_data)
        torch._dynamo.reset()
    quant_pipeline_enabled = _quant_pipeline_enabled(h)
    sliding_eval_enabled = _sliding_eval_enabled(h)
    ttt_eval_enabled = _ttt_eval_enabled(h)
    pre_ttt_loss = None
    pre_ttt_bpb = None
    pre_ttt_extra = None
    pre_quant_ttt_summary = None
    post_ttt_pre_quant_loss = None
    post_ttt_pre_quant_bpb = None
    post_ttt_pre_quant_extra = None
    post_ttt_pre_quant_loss_loop_inactive = None
    post_ttt_pre_quant_bpb_loop_inactive = None
    post_ttt_pre_quant_extra_loop_inactive = None
    post_ttt_pre_quant_loss_loop_active = None
    post_ttt_pre_quant_bpb_loop_active = None
    post_ttt_pre_quant_extra_loop_active = None
    checkpoint_to_pre_quant_gap = None
    artifact_summary = {}
    post_quant_loss = None
    post_quant_bpb = None
    post_quant_extra = None
    quant_gap = None
    sliding_bpb = None
    sliding_loss = None
    sliding_extra = None
    ttt_loss = None
    ttt_bpb = None
    ttt_extra = None
    post_ttt_checkpoint_summary = None
    if quant_pipeline_enabled:
        if compiled_model is not None:
            del compiled_model
            compiled_model = None
            torch._dynamo.reset()
            torch.cuda.empty_cache()
        pre_ttt_loss, pre_ttt_bpb, pre_ttt_extra = timed_eval('pre-ttt pre-quantization post-ema', eval_val, h, device, val_data, base_model)
        log_event(
            'pre_ttt_eval',
            val_loss=pre_ttt_loss,
            val_bpb=pre_ttt_bpb,
            checkpoint_val_bpb=training_summary['checkpoint_val_bpb'],
            model_state=('ema' if training_summary['ema_applied'] else 'checkpoint_raw'),
            **pre_ttt_extra,
        )
        if h.quant_only_from_post_ttt:
            pass
        elif h.pre_quant_ttt_enabled:
            pre_quant_ttt_summary = run_pre_quant_ttt(h, device, val_data, base_model)
            if h.save_post_ttt_checkpoint:
                post_ttt_checkpoint_summary = save_post_ttt_checkpoint(h, base_model)
                if post_ttt_checkpoint_summary is not None:
                    log_event('post_ttt_checkpoint', **post_ttt_checkpoint_summary)
        post_ttt_pre_quant_loss_loop_inactive, post_ttt_pre_quant_bpb_loop_inactive, post_ttt_pre_quant_extra_loop_inactive = _timed_eval_with_loop_mode(
            'post-ttt pre-quantization loop-inactive',
            eval_val,
            h,
            device,
            val_data,
            base_model,
            False,
        )
        if h.num_loops > 0:
            post_ttt_pre_quant_loss_loop_active, post_ttt_pre_quant_bpb_loop_active, post_ttt_pre_quant_extra_loop_active = _timed_eval_with_loop_mode(
                'post-ttt pre-quantization loop-active',
                eval_val,
                h,
                device,
                val_data,
                base_model,
                True,
            )
        else:
            post_ttt_pre_quant_loss_loop_active = post_ttt_pre_quant_loss_loop_inactive
            post_ttt_pre_quant_bpb_loop_active = post_ttt_pre_quant_bpb_loop_inactive
            post_ttt_pre_quant_extra_loop_active = dict(post_ttt_pre_quant_extra_loop_inactive or {})
            post_ttt_pre_quant_extra_loop_active['looping_active_eval'] = False
        post_ttt_pre_quant_loss = post_ttt_pre_quant_loss_loop_active
        post_ttt_pre_quant_bpb = post_ttt_pre_quant_bpb_loop_active
        post_ttt_pre_quant_extra = post_ttt_pre_quant_extra_loop_active
        log_event(
            'post_ttt_pre_quant_eval',
            val_loss=post_ttt_pre_quant_loss,
            val_bpb=post_ttt_pre_quant_bpb,
            pre_ttt_val_bpb=pre_ttt_bpb,
            checkpoint_val_bpb=training_summary['checkpoint_val_bpb'],
            post_ttt_prequant_val_bpb_loop_inactive=post_ttt_pre_quant_bpb_loop_inactive,
            post_ttt_prequant_val_bpb_loop_active=post_ttt_pre_quant_bpb_loop_active,
            model_state=('pre_quant_ttt_replayed' if h.quant_only_from_post_ttt else ('pre_quant_ttt_adapted' if h.pre_quant_ttt_enabled else ('ema' if training_summary['ema_applied'] else 'checkpoint_raw'))),
            **post_ttt_pre_quant_extra,
        )
        checkpoint_to_pre_quant_gap = (post_ttt_pre_quant_bpb_loop_active - training_summary['checkpoint_val_bpb'] if training_summary['checkpoint_val_bpb'] is not None else None)
        artifact_summary = serialize(h, base_model, script_code)
        log_event('artifact', **artifact_summary)
        log_event(
            'quant_diagnostics',
            quant_diagnostics_path=artifact_summary['quant_diagnostics_path'],
            gptq_calibration_batches=artifact_summary['gptq_calibration_batches'],
            gptq_calibration_time_ms=artifact_summary['gptq_calibration_time_ms'],
            gptq_quantize_time_ms=artifact_summary['gptq_quantize_time_ms'],
            gptq_compress_time_ms=artifact_summary['gptq_compress_time_ms'],
            quant_bits_by_group=artifact_summary['quant_bits_by_group'],
            quant_clip_sigma_by_group=artifact_summary['quant_clip_sigma_by_group'],
            quant_top_layer_error=artifact_summary['quant_top_layer_error'],
            quant_top_zero_fraction=artifact_summary['quant_top_zero_fraction'],
        )
        if h.max_artifact_bytes > 0 and artifact_summary['artifact_bytes'] > h.max_artifact_bytes:
            raise RunFailure('artifact_too_large', artifact_bytes=artifact_summary['artifact_bytes'], max_artifact_bytes=h.max_artifact_bytes)
        if h.distributed:
            dist.barrier()
        eval_model = None
        compiled_model = None
        if _post_quant_eval_enabled(h) or sliding_eval_enabled:
            eval_model = deserialize(h, device)
            if h.num_loops > 0:
                eval_model.looping_active = True
        if _post_quant_eval_enabled(h):
            compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
            post_quant_loss, post_quant_bpb, post_quant_extra = timed_eval('quantized', eval_val, h, device, val_data, compiled_model)
            quant_gap = post_quant_bpb - post_ttt_pre_quant_bpb_loop_active
            log_event(
                'post_ttt_post_quant_eval',
                val_loss=post_quant_loss,
                val_bpb=post_quant_bpb,
                quant_gap_bpb=quant_gap,
                quant_gap_bpb_loop_active=quant_gap,
                pre_ttt_val_bpb=pre_ttt_bpb,
                post_ttt_prequant_val_bpb=post_ttt_pre_quant_bpb,
                post_ttt_prequant_val_bpb_loop_inactive=post_ttt_pre_quant_bpb_loop_inactive,
                post_ttt_prequant_val_bpb_loop_active=post_ttt_pre_quant_bpb_loop_active,
                post_quant_val_bpb_loop_active=post_quant_bpb,
                model_state='quantized',
                **post_quant_extra,
            )
        if h.max_quant_bpb_gap > 0 and quant_gap is not None and quant_gap > h.max_quant_bpb_gap:
            raise RunFailure(
                'quant_gap_too_large',
                quant_gap=quant_gap,
                quant_gap_bpb=quant_gap,
                quant_gap_bpb_loop_active=quant_gap,
                max_quant_bpb_gap=h.max_quant_bpb_gap,
                pre_ttt_val_bpb=pre_ttt_bpb,
                post_ttt_prequant_val_bpb=post_ttt_pre_quant_bpb,
                post_ttt_prequant_val_bpb_loop_active=post_ttt_pre_quant_bpb_loop_active,
                post_ttt_postquant_val_bpb=post_quant_bpb,
                artifact_bytes=artifact_summary.get('artifact_bytes'),
            )
        if perf_is_enabled():
            perf_log(
                'quant_summary',
                pre_ttt_val_loss=pre_ttt_loss,
                pre_ttt_val_bpb=pre_ttt_bpb,
                post_ttt_prequant_val_loss=post_ttt_pre_quant_loss,
                post_ttt_prequant_val_bpb=post_ttt_pre_quant_bpb,
                post_ttt_prequant_val_bpb_loop_inactive=post_ttt_pre_quant_bpb_loop_inactive,
                post_ttt_prequant_val_bpb_loop_active=post_ttt_pre_quant_bpb_loop_active,
                post_quant_val_loss=post_quant_loss,
                post_quant_val_bpb=post_quant_bpb,
                quant_delta_bpb=quant_gap,
            )
        if sliding_eval_enabled:
            sliding_loss, sliding_bpb, sliding_extra = timed_eval('quantized_sliding_window', eval_val_sliding, h, device, val_data, eval_model)
            log_event(
                'sliding_eval',
                val_loss=sliding_loss,
                val_bpb=sliding_bpb,
                sliding_gain_bpb=(post_quant_bpb - sliding_bpb if post_quant_bpb is not None else None),
                model_state='quantized_sliding',
                **sliding_extra,
            )
            if perf_is_enabled():
                perf_log('sliding_summary', quantized_val_bpb=post_quant_bpb, sliding_val_loss=sliding_loss, sliding_val_bpb=sliding_bpb, sliding_gain_bpb=(post_quant_bpb - sliding_bpb if post_quant_bpb is not None else None))
        if ttt_eval_enabled:
            if eval_model is not None:
                del eval_model
            if compiled_model is not None:
                del compiled_model
            torch._dynamo.reset()
            torch.cuda.empty_cache()
            ttt_model = deserialize(h, device)
            if h.num_loops > 0:
                ttt_model.looping_active = True
            ttt_loss, ttt_bpb, ttt_extra = timed_eval('quantized_ttt', eval_val_ttt, h, device, val_data, ttt_model)
            log_event(
                'ttt_eval',
                val_loss=ttt_loss,
                val_bpb=ttt_bpb,
                ttt_gain_bpb=(post_quant_bpb - ttt_bpb if post_quant_bpb is not None else None),
                ttt_gain_vs_sliding=(sliding_bpb - ttt_bpb if sliding_bpb is not None else None),
                model_state='quantized_ttt',
                **ttt_extra,
            )
            if perf_is_enabled():
                perf_log('ttt_summary', quantized_val_bpb=post_quant_bpb, sliding_val_bpb=sliding_bpb, ttt_val_loss=ttt_loss, ttt_val_bpb=ttt_bpb, ttt_gain_bpb=(post_quant_bpb - ttt_bpb if post_quant_bpb is not None else None), ttt_gain_vs_sliding=(sliding_bpb - ttt_bpb if sliding_bpb is not None else None))
            del ttt_model
        if h.etlb_enabled and sliding_eval_enabled:
            if 'eval_model' not in dir():
                eval_model = deserialize(h, device)
                if h.num_loops > 0:
                    eval_model.looping_active = True
            timed_eval('quantized_sliding_etlb', eval_val_sliding_etlb, h, device, val_data, eval_model)
    run_wallclock_total_ms = 1000.0 * (time.perf_counter() - run_wallclock_t0)
    wallclock_budget_ms = 1000.0 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    wallclock_margin_ms = wallclock_budget_ms - run_wallclock_total_ms if wallclock_budget_ms is not None else None
    phase_time_breakdown = {
        'train_step_time_ms_total': training_summary.get('train_step_time_ms_total'),
        'checkpoint_eval_time_ms_total': training_summary.get('checkpoint_eval_time_ms_total'),
        'pre_ttt_eval_time_ms': pre_ttt_extra['eval_time_ms'] if pre_ttt_extra is not None else None,
        'pre_quant_ttt_wall_ms': pre_quant_ttt_summary['ttt_wall_ms'] if pre_quant_ttt_summary is not None else None,
        'post_ttt_prequant_eval_time_ms_loop_inactive': post_ttt_pre_quant_extra_loop_inactive['eval_time_ms'] if post_ttt_pre_quant_extra_loop_inactive is not None else None,
        'post_ttt_prequant_eval_time_ms_loop_active': post_ttt_pre_quant_extra_loop_active['eval_time_ms'] if post_ttt_pre_quant_extra_loop_active is not None else None,
        'gptq_calibration_time_ms': artifact_summary.get('gptq_calibration_time_ms'),
        'gptq_quantize_time_ms': artifact_summary.get('gptq_quantize_time_ms'),
        'gptq_compress_time_ms': artifact_summary.get('gptq_compress_time_ms'),
        'post_quant_eval_time_ms': post_quant_extra['eval_time_ms'] if post_quant_extra is not None else None,
        'sliding_eval_time_ms': sliding_extra['eval_time_ms'] if sliding_extra is not None else None,
        'ttt_eval_time_ms': ttt_extra['eval_time_ms'] if ttt_extra is not None else None,
    }
    final_summary = {
        **training_summary,
        **artifact_summary,
        **(post_ttt_checkpoint_summary or {}),
        'run_wallclock_total_ms': run_wallclock_total_ms,
        'wallclock_budget_seconds': h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None,
        'wallclock_over_budget': (run_wallclock_total_ms > wallclock_budget_ms if wallclock_budget_ms is not None else None),
        'wallclock_margin_ms': wallclock_margin_ms,
        'phase_time_breakdown': phase_time_breakdown,
        'final_eval_mode': h.final_eval_mode,
        'final_score_mode': h.final_score_mode,
        'quant_pipeline_enabled': quant_pipeline_enabled,
        'post_quant_eval_enabled': _post_quant_eval_enabled(h),
        'sliding_eval_enabled': sliding_eval_enabled,
        'ttt_eval_enabled': ttt_eval_enabled,
        'eval_stride': h.eval_stride,
        'num_eval_windows': (
            (ttt_extra.get('num_eval_windows') if ttt_extra is not None else None)
            or (sliding_extra.get('num_eval_windows') if sliding_extra is not None else None)
            or (post_quant_extra.get('num_eval_windows') if post_quant_extra is not None else None)
            or (post_ttt_pre_quant_extra.get('num_eval_windows') if post_ttt_pre_quant_extra is not None else None)
            or (pre_ttt_extra.get('num_eval_windows') if pre_ttt_extra is not None else None)
        ),
        'save_post_ttt_checkpoint': h.save_post_ttt_checkpoint,
        'quant_only_from_post_ttt': h.quant_only_from_post_ttt,
        'quant_only_checkpoint_path': quant_only_origin.get('quant_only_checkpoint_path'),
        'quant_only_checkpoint_sha256': quant_only_origin.get('quant_only_checkpoint_sha256'),
        'quant_only_origin_run_id': quant_only_origin.get('quant_only_origin_run_id'),
        'quant_only_origin_method': quant_only_origin.get('quant_only_origin_method'),
        'quant_only_origin_post_ttt_prequant_val_bpb': quant_only_origin.get('quant_only_origin_post_ttt_prequant_val_bpb'),
        'quant_only_origin_post_quant_val_bpb': quant_only_origin.get('quant_only_origin_post_quant_val_bpb'),
        'wrapper_code_bytes_estimate': h.wrapper_code_bytes_estimate,
        'final_artifact_bytes_estimate': ((artifact_summary.get('quantized_model_bytes') or 0) + h.wrapper_code_bytes_estimate if artifact_summary.get('quantized_model_bytes') is not None else None),
        'tokenizer_path': val_data.tokenizer_path,
        'tokenizer_sha256': val_data.tokenizer_sha256,
        'tokenizer_kind': val_data.tokenizer_kind,
        'tokenizer_vocab_size': val_data.vocab_size,
        'validation_selection': val_data.selection,
        'caseops_roundtrip_ok': val_data.caseops_roundtrip_ok,
        'caseops_raw_utf8_bytes': val_data.caseops_raw_utf8_bytes,
        'caseops_sidecar_utf8_bytes': val_data.caseops_sidecar_utf8_bytes,
        'caseops_zero_byte_token_count': val_data.caseops_zero_byte_token_count,
        'pre_quant_ttt_enabled': h.pre_quant_ttt_enabled,
        'pre_quant_ttt_epochs': h.pre_quant_ttt_epochs,
        'pre_quant_ttt_lr': h.pre_quant_ttt_lr,
        'pre_quant_ttt_min_lr': h.pre_quant_ttt_min_lr,
        'pre_quant_ttt_weight_decay': h.pre_quant_ttt_weight_decay,
        'pre_quant_ttt_grad_clip_norm': h.pre_quant_ttt_grad_clip_norm,
        'pre_quant_ttt_freeze_blocks': h.pre_quant_ttt_freeze_blocks,
        'pre_quant_ttt_source': h.pre_quant_ttt_source,
        'pre_quant_ttt_files': h.pre_quant_ttt_files,
        'pre_quant_ttt_shard_limit': h.pre_quant_ttt_shard_limit,
        'pre_quant_ttt_selection_mode': h.pre_quant_ttt_selection_mode,
        'pre_quant_ttt_chunk_tokens': h.pre_quant_ttt_chunk_tokens,
        'pre_quant_ttt_loop_mode': h.pre_quant_ttt_loop_mode,
        'pre_quant_ttt_anchor_lambda': h.pre_quant_ttt_anchor_lambda,
        'pre_quant_ttt_qnoise_enabled': h.pre_quant_ttt_qnoise_enabled,
        'pre_quant_ttt_qnoise_start_epoch': h.pre_quant_ttt_qnoise_start_epoch,
        'pre_quant_ttt_qnoise_prob': h.pre_quant_ttt_qnoise_prob,
        'pre_quant_ttt_qnoise_scale_mult': h.pre_quant_ttt_qnoise_scale_mult,
        'pre_quant_ttt_selection': val_data.pre_quant_ttt_selection,
        'checkpoint_to_pre_quant_bpb_gap': checkpoint_to_pre_quant_gap,
        'pre_ttt_val_loss': pre_ttt_loss,
        'pre_ttt_val_bpb': pre_ttt_bpb,
        'pre_ttt_eval_time_ms': (pre_ttt_extra['eval_time_ms'] if pre_ttt_extra is not None else None),
        'post_ttt_prequant_val_loss': post_ttt_pre_quant_loss,
        'post_ttt_prequant_val_bpb': post_ttt_pre_quant_bpb,
        'post_ttt_prequant_eval_time_ms': (post_ttt_pre_quant_extra['eval_time_ms'] if post_ttt_pre_quant_extra is not None else None),
        'post_ttt_prequant_val_loss_loop_inactive': post_ttt_pre_quant_loss_loop_inactive,
        'post_ttt_prequant_val_bpb_loop_inactive': post_ttt_pre_quant_bpb_loop_inactive,
        'post_ttt_prequant_eval_time_ms_loop_inactive': (post_ttt_pre_quant_extra_loop_inactive['eval_time_ms'] if post_ttt_pre_quant_extra_loop_inactive is not None else None),
        'post_ttt_prequant_val_loss_loop_active': post_ttt_pre_quant_loss_loop_active,
        'post_ttt_prequant_val_bpb_loop_active': post_ttt_pre_quant_bpb_loop_active,
        'post_ttt_prequant_eval_time_ms_loop_active': (post_ttt_pre_quant_extra_loop_active['eval_time_ms'] if post_ttt_pre_quant_extra_loop_active is not None else None),
        'pre_quant_val_loss': post_ttt_pre_quant_loss,
        'pre_quant_val_bpb': post_ttt_pre_quant_bpb,
        'pre_quant_eval_time_ms': (post_ttt_pre_quant_extra['eval_time_ms'] if post_ttt_pre_quant_extra is not None else None),
        'post_quant_val_loss': post_quant_loss,
        'post_quant_val_bpb': post_quant_bpb,
        'post_quant_val_bpb_loop_active': post_quant_bpb,
        'post_quant_eval_time_ms': (post_quant_extra['eval_time_ms'] if post_quant_extra is not None else None),
        'post_ttt_postquant_val_loss': post_quant_loss,
        'post_ttt_postquant_val_bpb': post_quant_bpb,
        'quant_gap_bpb': quant_gap,
        'quant_gap_bpb_loop_active': quant_gap,
        'sliding_val_loss': sliding_loss,
        'sliding_val_bpb': sliding_bpb,
        'sliding_eval_time_ms': (sliding_extra['eval_time_ms'] if sliding_extra is not None else None),
        'ttt_val_loss': ttt_loss,
        'ttt_val_bpb': ttt_bpb,
        'legal_ttt_val_bpb': ttt_bpb,
        'ttt_eval_time_ms': (ttt_extra['eval_time_ms'] if ttt_extra is not None else None),
        'ttt_score_ms_total': (ttt_extra['ttt_total_score_ms'] if ttt_extra is not None else None),
        'ttt_update_ms_total': (ttt_extra['ttt_total_update_ms'] if ttt_extra is not None else None),
        'ttt_wall_time_ms_total': (ttt_extra['ttt_total_wall_time_ms'] if ttt_extra is not None else None),
        'ttt_score_before_update': (ttt_extra['ttt_score_before_update'] if ttt_extra is not None else None),
        'ttt_single_left_to_right_pass': (ttt_extra['ttt_single_left_to_right_pass'] if ttt_extra is not None else None),
        'ttt_multi_pass_rescore': (ttt_extra['ttt_multi_pass_rescore'] if ttt_extra is not None else None),
        'ttt_strict_causal_prefix_only': (ttt_extra['ttt_strict_causal_prefix_only'] if ttt_extra is not None else None),
        'ttt_trainable_param_count': (pre_quant_ttt_summary['ttt_trainable_param_count'] if pre_quant_ttt_summary is not None else None),
        'ttt_wall_ms': (pre_quant_ttt_summary['ttt_wall_ms'] if pre_quant_ttt_summary is not None else None),
        'ttt_looping_active_start': (pre_quant_ttt_summary['ttt_looping_active_start'] if pre_quant_ttt_summary is not None else None),
        'ttt_looping_active_end': (pre_quant_ttt_summary['ttt_looping_active_end'] if pre_quant_ttt_summary is not None else None),
        'ttt_epoch_stats': (pre_quant_ttt_summary['ttt_epoch_stats'] if pre_quant_ttt_summary is not None else None),
        'comparison_metric_priority': _comparison_metric_priority(h),
    }
    log_event('final', **final_summary)
    return final_summary

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
    if h.is_main_process:
        Path(h.run_dir).mkdir(parents=True, exist_ok=True)
    set_logging_hparams(h)
    metrics_logger = JsonlLogger(h.metrics_path, h.is_main_process)
    set_metrics_logger(metrics_logger)
    perf_logger = PerfLogger(h)
    peak_tracker = PeakMemoryTracker(device)
    set_perf_logger(perf_logger, peak_tracker)
    script_code = Path(__file__).read_text(encoding='utf-8')
    write_manifest(h, device, script_code, {
        'status': {
            'state': 'starting',
            'started_at': time.time(),
        },
    })
    if h.is_main_process:
        log(100 * '=', console=False)
        log('Hyperparameters:', console=True)
        for k, v in sorted(vars(h).items()):
            log(f'  {k}: {v}', console=True)
        log('=' * 100, console=False)
        log(f'Running Python {sys.version}', console=False)
        log(f'Running PyTorch {torch.__version__}', console=False)
        log(subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
        log('=' * 100, console=False)
        if perf_logger.enabled:
            log(f'perf_logfile: {h.perf_logfile}')
        if h.profiler_enabled and _rank_is_selected(h.profiler_ranks, h.rank):
            log(f'profiler_dir: {h.profiler_dir}')
    if perf_is_enabled():
        perf_log('run_start', hyperparameters=vars(h))
    summary = None
    try:
        summary = train_and_eval(h, device)
        write_summary(h, {
            'status': 'success',
            **summary,
        })
        write_manifest(h, device, script_code, {
            'results': summary,
            'status': {
                'state': 'success',
                'finished_at': time.time(),
            },
        })
        log_event('run_end', status='success', **summary)
        if perf_is_enabled():
            perf_log('run_end', **peak_tracker.run_peaks_mb())
    except RunFailure as exc:
        failure_summary = {
            'status': 'failure',
            'reason': exc.reason,
            'details': exc.details,
            'traceback': traceback.format_exc(),
            **{key: value for key, value in exc.details.items() if key not in {'message'}},
        }
        write_summary(h, failure_summary)
        write_manifest(h, device, script_code, {
            'status': {
                'state': 'failure',
                'finished_at': time.time(),
                'reason': exc.reason,
                'details': exc.details,
            },
        })
        log_event('run_end', status='failure', reason=exc.reason, **exc.details)
        raise
    except Exception as exc:
        failure_summary = {
            'status': 'failure',
            'reason': exc.__class__.__name__,
            'details': {'message': str(exc)},
            'traceback': traceback.format_exc(),
        }
        write_summary(h, failure_summary)
        write_manifest(h, device, script_code, {
            'status': {
                'state': 'failure',
                'finished_at': time.time(),
                'reason': exc.__class__.__name__,
                'details': {'message': str(exc)},
            },
        })
        log_event('failure', reason=exc.__class__.__name__, message=str(exc))
        log_event('run_end', status='failure', reason=exc.__class__.__name__, message=str(exc))
        raise
    finally:
        metrics_logger.close()
        perf_logger.close()
        if distributed:
            dist.destroy_process_group()
if __name__ == '__main__':
    main()
