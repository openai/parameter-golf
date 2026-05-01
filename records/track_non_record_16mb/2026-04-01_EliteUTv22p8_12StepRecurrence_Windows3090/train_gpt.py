from __future__ import annotations
import os
import time
import uuid
import math
import random
from collections import deque

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import sentencepiece as spm

# Modular Imports
from model import GPT
from data_utils import DistributedTokenLoader
from optimizer_utils import Muon
from eval_utils import eval_val, build_sentencepiece_luts, load_validation_tokens
from quant_utils import quantize_state_dict_int8, dequantize_state_dict_int8

# --- LOGGING ---
_LOG_FILE: str | None = None
_MASTER_PROCESS: bool = True

def log0(msg: str, console: bool = True) -> None:
    if not _MASTER_PROCESS:
        return
    if console:
        print(msg)
    if _LOG_FILE is not None:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            print(msg, file=f)


# --- LOSS FILTER ---
class LossFilter:
    """Per-micro-batch z-score filter on loss deltas to skip pathological batches."""
    def __init__(self, warmup: int, delta_window: int = 20, z_threshold: float = 4.5,
                 stability_window: int = 20, max_recent_drop: float = 0.02,
                 max_retries: int = 8):
        self.warmup = warmup
        self.delta_window = delta_window
        self.z_threshold = z_threshold
        self.stability_window = stability_window
        self.max_recent_drop = max_recent_drop
        self.max_retries = max_retries
        self._history: deque[float] = deque(maxlen=max(delta_window, stability_window) + 2)
        self.global_step = 0
        self.accepted = 0
        self.skipped = 0
        self.fallback_accepts = 0

    def should_skip(self, loss: float) -> bool:
        """Return True if this micro-batch loss looks pathological."""
        self.global_step += 1
        if self.global_step <= self.warmup or len(self._history) < 4:
            self._history.append(loss)
            self.accepted += 1
            return False

        history = list(self._history)
        # Compute loss deltas over recent window
        recent = history[-self.delta_window:]
        if len(recent) >= 2:
            deltas = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
            new_delta = loss - history[-1]
            mean_d = sum(deltas) / len(deltas)
            var_d = sum((d - mean_d) ** 2 for d in deltas) / len(deltas)
            std_d = math.sqrt(var_d) + 1e-8
            z = (new_delta - mean_d) / std_d
            if z > self.z_threshold:
                self.skipped += 1
                return True

        # Also skip if loss is anomalously high relative to recent stability window
        recent_losses = history[-self.stability_window:]
        if recent_losses:
            min_recent = min(recent_losses)
            if loss > min_recent * (1.0 + self.max_recent_drop * 10):
                self.skipped += 1
                return True

        self._history.append(loss)
        self.accepted += 1
        return False

    def force_accept(self, loss: float) -> None:
        """Accept a batch unconditionally (fallback after max_retries)."""
        self._history.append(loss)
        self.accepted += 1
        self.fallback_accepts += 1

    def summary(self) -> str:
        return f"accepted={self.accepted} skipped={self.skipped} fallback_accepts={self.fallback_accepts}"


# --- HYPERPARAMETERS ---
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "../../../data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "../../../data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 65_536))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 100))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 1))

    iterations = int(os.environ.get("ITERATIONS", 500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 16))
    cosine_min = float(os.environ.get("COSINE_MIN", "0.1"))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    train_batch_tokens = 524_288  # NON-NEGOTIABLE COMPETITION STANDARD
    micro_batch_tokens = 32_768
    # TRAIN_SEQ_LEN controls training context; ScaleDown.bat sets this to 1024
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))

    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 1024))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 5))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 10.0))

    # RECURRENCE_STEPS takes priority over NUM_STEPS (consistent with ScaleDown.bat logic)
    num_steps = int(os.environ.get("RECURRENCE_STEPS", os.environ.get("NUM_STEPS", 1)))
    lora_rank = int(os.environ.get("LORA_RANK", 512))
    lora_scope = os.environ.get("LORA_SCOPE", "q")

    # Feature toggles
    bigram_hash_enabled = bool(int(os.environ.get("BIGRAM_HASH_ENABLED", "0")))
    bigram_hash_size = int(os.environ.get("BIGRAM_HASH_SIZE", 2048))
    bigram_hash_scale = float(os.environ.get("BIGRAM_HASH_SCALE", "0.05"))
    level_signal_enabled = bool(int(os.environ.get("LEVEL_SIGNAL_ENABLED", "0")))
    level_signal_rank = int(os.environ.get("LEVEL_SIGNAL_RANK", "0")) or None
    smeargate_enabled = bool(int(os.environ.get("SMEARGATE_ENABLED", "0")))
    smeargate_alpha = float(os.environ.get("SMEARGATE_ALPHA", "0.08"))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", "0.0004"))

    # Curriculum
    seq_len_curriculum = bool(int(os.environ.get("SEQ_LEN_CURRICULUM", "0")))
    short_train_seq_len = int(os.environ.get("SHORT_TRAIN_SEQ_LEN", "128"))
    seq_len_curriculum_steps = int(os.environ.get("SEQ_LEN_CURRICULUM_STEPS", "20"))
    recurrence_curriculum = bool(int(os.environ.get("RECURRENCE_CURRICULUM", "0")))

    # Optimizer LRs and WDs
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.08))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.015))
    lora_lr = float(os.environ.get("LORA_LR", os.environ.get("SCALAR_LR", 0.015)))
    control_lr = float(os.environ.get("CONTROL_LR", os.environ.get("SCALAR_LR", 0.015)))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    embed_lr = float(os.environ.get("EMBED_LR", 0.7))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.06))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    scalar_weight_decay = float(os.environ.get("SCALAR_WEIGHT_DECAY", "0.1"))
    lora_weight_decay = float(os.environ.get("LORA_WEIGHT_DECAY", "0.0"))
    control_weight_decay = float(os.environ.get("CONTROL_WEIGHT_DECAY", "0.0"))

    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
    dynamic_lr_norm = bool(int(os.environ.get("DYNAMIC_LR_NORM", "0")))
    target_grad_norm = float(os.environ.get("TARGET_GRAD_NORM", "0.5"))

    # Checkpoint / export
    save_best_checkpoint = bool(int(os.environ.get("SAVE_BEST_CHECKPOINT", "1")))
    save_best_int8 = bool(int(os.environ.get("SAVE_BEST_INT8", "1")))
    export_best_checkpoint = bool(int(os.environ.get("EXPORT_BEST_CHECKPOINT", "1")))
    quant_eval = bool(int(os.environ.get("QUANT_EVAL", "1")))
    quant_eval_max_steps = int(os.environ.get("QUANT_EVAL_MAX_STEPS", "50"))
    quant_eval_stride = int(os.environ.get("QUANT_EVAL_STRIDE", "64"))

    # Loss filter
    loss_filter_enabled = bool(int(os.environ.get("LOSS_FILTER_ENABLED", "0")))
    loss_filter_warmup = int(os.environ.get("LOSS_FILTER_WARMUP", "250"))
    loss_filter_delta_window = int(os.environ.get("LOSS_FILTER_DELTA_WINDOW", "20"))
    loss_filter_z = float(os.environ.get("LOSS_FILTER_Z_THRESHOLD", "4.5"))
    loss_filter_stability_window = int(os.environ.get("LOSS_FILTER_STABILITY_WINDOW", "20"))
    loss_filter_max_recent_drop = float(os.environ.get("LOSS_FILTER_MAX_RECENT_DROP", "0.02"))
    loss_filter_max_retries = int(os.environ.get("LOSS_FILTER_MAX_RETRIES", "8"))
    data_deterministic = bool(int(os.environ.get("DATA_DETERMINISTIC", "0")))
    data_seed = int(os.environ.get("DATA_SEED", "0")) or None


CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,"
    "q_gain,skip_weight,skip_weights,v_step_bias,level_gain"
).split(",")

LORA_TENSOR_NAME_PATTERNS = ("lora_A,lora_B").split(",")


def _vram_stats(device: torch.device) -> dict[str, float]:
    alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
    resv = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_resv = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    return dict(alloc_gib=alloc, resv_gib=resv, peak_alloc_gib=peak_alloc, peak_resv_gib=peak_resv)


def _vram_str(stats: dict[str, float]) -> str:
    return (f"vram_alloc_gib={stats['alloc_gib']:.2f} vram_resv_gib={stats['resv_gib']:.2f} "
            f"vram_peak_alloc_gib={stats['peak_alloc_gib']:.2f} vram_peak_resv_gib={stats['peak_resv_gib']:.2f}")


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


def _count_params(model: nn.Module) -> tuple[int, int]:
    """Returns (total_params, lora_params)."""
    total = sum(p.numel() for p in model.parameters())
    lora = sum(p.numel() for n, p in model.named_parameters()
               if any(pat in n for pat in LORA_TENSOR_NAME_PATTERNS))
    return total, lora


def main() -> None:
    print("[debug] main() started")
    args = Hyperparameters()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ and os.environ.get("FORCE_SINGLE_GPU") != "1"
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = max(1, args.train_batch_tokens // (args.micro_batch_tokens * world_size))

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logfile = f"logs/{args.run_id}.txt" if master_process else None
    global _LOG_FILE, _MASTER_PROCESS
    _MASTER_PROCESS, _LOG_FILE = master_process, logfile
    if master_process:
        os.makedirs("logs", exist_ok=True)
    print("[debug] logging initialized")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # --- Config dump ---
    if master_process:
        cuda_ver = torch.version.cuda or "?"
        log0(f"[config] --------------------------------------------------")
        log0(f"[config] run_id={args.run_id} seed={args.seed}")
        log0(f"[config] torch={torch.__version__} cuda={cuda_ver} device={torch.cuda.get_device_name(device)}")
        free, total_mem = torch.cuda.mem_get_info(device)
        log0(f"[vram] gpu_mem_total_gib={total_mem/(1024**3):.2f} gpu_mem_free_gib={free/(1024**3):.2f} (before init)")
        log0(f"[config] distributed={distributed} world_size={world_size} rank={rank} local_rank={local_rank}")
        log0(f"[config] stop: iterations={args.iterations} max_wallclock_seconds={args.max_wallclock_seconds}")
        log0(f"[config] schedule: warmup_steps={args.warmup_steps} cosine_min={args.cosine_min}")
        log0(f"[config] batch: train_batch_tokens={args.train_batch_tokens} micro_batch_tokens={args.micro_batch_tokens} grad_accum_steps={grad_accum_steps}")
        log0(f"[config] data: TRAIN_SEQ_LEN(eval)={args.train_seq_len} VAL_BATCH_SIZE={args.val_batch_size} VAL_LOSS_EVERY={args.val_loss_every}")
        log0(f"[config] model: dim={args.model_dim} heads={args.num_heads} kv_heads={args.num_kv_heads} mlp_mult={args.mlp_mult} steps={args.num_steps} lora_rank={args.lora_rank} lora_scope={args.lora_scope}")
        log0(f"[config] features: SMEARGATE={int(args.smeargate_enabled)} SMEARGATE_ALPHA={args.smeargate_alpha} BIGRAM_HASH={int(args.bigram_hash_enabled)} BIGRAM_HASH_SIZE={args.bigram_hash_size} BIGRAM_HASH_SCALE={args.bigram_hash_scale} LEVEL_SIGNAL={int(args.level_signal_enabled)} LEVEL_SIGNAL_RANK={args.level_signal_rank or 0}")
        log0(f"[config] optim: MATRIX_LR={args.matrix_lr} MUON_BACKEND_STEPS={args.muon_backend_steps} MUON_MOMENTUM={args.muon_momentum} SCALAR_LR={args.scalar_lr}")
        log0(f"[config] optim_groups: LORA_LR={args.lora_lr} CONTROL_LR={args.control_lr} SCALAR_WD={args.scalar_weight_decay} LORA_WD={args.lora_weight_decay} CONTROL_WD={args.control_weight_decay}")
        log0(f"[config] clip: GRAD_CLIP_NORM={args.grad_clip_norm} DYNAMIC_LR_NORM={int(args.dynamic_lr_norm)} TARGET_GRAD_NORM={args.target_grad_norm}")
        log0(f"[config] curriculum: SEQ_LEN_CURRICULUM={int(args.seq_len_curriculum)} SHORT_TRAIN_SEQ_LEN={args.short_train_seq_len} SEQ_LEN_CURRICULUM_STEPS={args.seq_len_curriculum_steps}")
        log0(f"[config] curriculum: RECURRENCE_CURRICULUM={int(args.recurrence_curriculum)}")
        log0(f"[config] ttt: TTT_ENABLED={int(args.ttt_enabled)} TTT_LR={args.ttt_lr}")
        log0(f"[config] checkpoint: SAVE_BEST_CHECKPOINT={int(args.save_best_checkpoint)} SAVE_BEST_INT8={int(args.save_best_int8)} EXPORT_BEST_CHECKPOINT={int(args.export_best_checkpoint)}")
        log0(f"[config] quant_eval: QUANT_EVAL={int(args.quant_eval)} QUANT_EVAL_MAX_STEPS={args.quant_eval_max_steps} QUANT_EVAL_STRIDE={args.quant_eval_stride}")
        log0(f"[config] loss_filter: enabled={int(args.loss_filter_enabled)} warmup={args.loss_filter_warmup} delta_window={args.loss_filter_delta_window} z={args.loss_filter_z} stability_window={args.loss_filter_stability_window} max_recent_drop={args.loss_filter_max_recent_drop} max_retries_per_step={args.loss_filter_max_retries} allow_distributed=0")
        log0(f"[config] --------------------------------------------------")

    print("[debug] loading validation tokens...")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    print("[debug] initializing base_model...")
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_steps=args.num_steps,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        lora_rank=args.lora_rank,
        lora_scope=args.lora_scope,
        bigram_hash_enabled=args.bigram_hash_enabled,
        bigram_hash_size=args.bigram_hash_size,
        bigram_hash_scale=args.bigram_hash_scale,
        level_signal_enabled=args.level_signal_enabled,
        level_rank=args.level_signal_rank,
    ).to(device).bfloat16()

    total_params, lora_params = _count_params(base_model)
    lora_frac = lora_params / total_params if total_params > 0 else 0.0
    log0(f"[config] vocab_size={args.vocab_size} embed_params={args.vocab_size * args.model_dim}")
    log0(f"[params] total={total_params:,} lora={lora_params:,} lora_frac={lora_frac:.4f}")

    model = base_model
    if distributed:
        print("[debug] wrapping in DDP...")
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    # --- Optimizer param splitting: matrix | lora | control | scalar | embed ---
    print("[debug] splitting params for optimizers...")
    matrix_params, lora_params_list, control_params, scalar_params = [], [], [], []
    for name, p in base_model.named_parameters():
        if "tok_emb" in name or (base_model.lm_head is not None and "lm_head" in name):
            continue
        is_lora = any(pat in name for pat in LORA_TENSOR_NAME_PATTERNS)
        is_control = any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
        is_matrix = p.ndim == 2 and not is_lora and not is_control and "step_embeddings" not in name and "bigram_hash" not in name
        if is_lora:
            lora_params_list.append(p)
        elif is_control:
            control_params.append(p)
        elif is_matrix:
            matrix_params.append(p)
        else:
            scalar_params.append(p)

    log0(f"[debug] splitting params for optimizers...")
    log0(f"[optim_split] matrix={len(matrix_params)} scalar={len(scalar_params)} lora={len(lora_params_list)} control={len(control_params)} (lr: scalar={args.scalar_lr} lora={args.lora_lr} control={args.control_lr}; wd: scalar={args.scalar_weight_decay} lora={args.lora_weight_decay} control={args.control_weight_decay})")

    print("[debug] initializing optimizers...")
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr

    optimizer_tok = torch.optim.AdamW(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "target_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True, weight_decay=0.0,
    )
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                          backend_steps=args.muon_backend_steps)
    optimizer_lora = torch.optim.AdamW(
        [{"params": lora_params_list, "lr": args.lora_lr, "target_lr": args.lora_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        weight_decay=args.lora_weight_decay,
    ) if lora_params_list else None
    optimizer_control = torch.optim.AdamW(
        [{"params": control_params, "lr": args.control_lr, "target_lr": args.control_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        weight_decay=args.control_weight_decay,
    ) if control_params else None
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "target_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        weight_decay=args.scalar_weight_decay,
    ) if scalar_params else None

    optimizers = [o for o in [optimizer_tok, optimizer_muon, optimizer_lora,
                               optimizer_control, optimizer_scalar] if o is not None]
    if base_model.lm_head is not None:
        opt_head = torch.optim.AdamW(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "target_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True, weight_decay=0.1,
        )
        optimizers.insert(1, opt_head)

    # Loss filter
    loss_filter = LossFilter(
        warmup=args.loss_filter_warmup * grad_accum_steps,
        delta_window=args.loss_filter_delta_window,
        z_threshold=args.loss_filter_z,
        stability_window=args.loss_filter_stability_window,
        max_recent_drop=args.loss_filter_max_recent_drop,
        max_retries=args.loss_filter_max_retries,
    ) if args.loss_filter_enabled else None

    print("[debug] initializing data loader...")
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    training_time_ms = 0.0
    t0 = time.perf_counter()
    step = 0

    print("[debug] initializing model EMA...")
    model_ema = {n.replace("module.", ""): p.clone().detach() for n, p in model.named_parameters()}

    # Best checkpoint tracking
    best_val_bpb = float("inf")
    best_val_loss = float("inf")
    best_step = -1
    best_ema: dict | None = None

    print("[debug] entering training loop...")
    global_start_time = time.perf_counter()

    while True:
        elapsed_sec = time.perf_counter() - global_start_time
        last_step = step >= args.iterations or elapsed_sec >= args.max_wallclock_seconds

        if step > 0 and args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step):
            torch.cuda.synchronize()
            vram_pre = _vram_stats(device)
            log0(f"[vram_eval] pre_swap step={step} alloc_gib={vram_pre['alloc_gib']:.2f} resv_gib={vram_pre['resv_gib']:.2f} peak_alloc_gib={vram_pre['peak_alloc_gib']:.2f} peak_resv_gib={vram_pre['peak_resv_gib']:.2f}")

            original_params = {n: p.data.clone() for n, p in base_model.named_parameters()}
            for n, p in base_model.named_parameters():
                if n in model_ema:
                    p.data.copy_(model_ema[n])

            vram_post_swap = _vram_stats(device)
            log0(f"[vram_eval] post_swap step={step} alloc_gib={vram_post_swap['alloc_gib']:.2f} resv_gib={vram_post_swap['resv_gib']:.2f} peak_alloc_gib={vram_post_swap['peak_alloc_gib']:.2f} peak_resv_gib={vram_post_swap['peak_resv_gib']:.2f}")

            eval_stride = args.quant_eval_stride if last_step else args.train_seq_len
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                         val_tokens, base_bytes_lut, has_leading_space_lut,
                                         is_boundary_token_lut, max_steps=50, stride=eval_stride,
                                         ttt_lr=args.ttt_lr)

            for n, p in base_model.named_parameters():
                p.data.copy_(original_params[n])

            vram_post_restore = _vram_stats(device)
            log0(f"[vram_eval] post_restore step={step} alloc_gib={vram_post_restore['alloc_gib']:.2f} resv_gib={vram_post_restore['resv_gib']:.2f} peak_alloc_gib={vram_post_restore['peak_alloc_gib']:.2f} peak_resv_gib={vram_post_restore['peak_resv_gib']:.2f}")

            final_tag = " [FINAL STRIDE 64]" if last_step else ""
            log0(f"step:{step} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{training_time_ms:.0f}ms{final_tag} {_vram_str(vram_post_restore)}")

            # Best checkpoint tracking
            if args.save_best_checkpoint and val_bpb < best_val_bpb:
                prev_bpb = best_val_bpb
                best_val_bpb = val_bpb
                best_val_loss = val_loss
                best_step = step
                best_ema = {n: p.clone() for n, p in model_ema.items()}

                # Save best_model.pt using current EMA weights
                original_params2 = {n: p.data.clone() for n, p in base_model.named_parameters()}
                for n, p in base_model.named_parameters():
                    if n in model_ema:
                        p.data.copy_(model_ema[n])
                sd = base_model.state_dict()
                sz_bytes = sum(v.numel() * v.element_size() for v in sd.values())
                torch.save(sd, "best_model.pt")
                sz_mib = sz_bytes / (1024 ** 2)
                log0(f"[best] new_best step={step} val_loss={val_loss:.4f} val_bpb={val_bpb:.4f} (prev={prev_bpb:.4f}) saved=best_model.pt ({sz_mib:.2f} MiB)")

                if args.save_best_int8:
                    int8_sd, _ = quantize_state_dict_int8(sd)
                    import io, zlib
                    buf = io.BytesIO()
                    torch.save(int8_sd, buf)
                    payload = zlib.compress(buf.getvalue(), level=9)
                    with open("best_model.int8.ptz", "wb") as f:
                        f.write(payload)
                    int8_mib = len(payload) / (1024 ** 2)
                    log0(f"[best] int8_saved=best_model.int8.ptz ({int8_mib:.2f} MiB) int8_payload_bytes={len(payload)} baseline_bytes={sz_bytes}")

                for n, p in base_model.named_parameters():
                    p.data.copy_(original_params2[n])

            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if last_step:
                reason = "iterations" if step >= args.iterations else "wallclock"
                log0(f"[stop] reason:{reason} step:{step} elapsed:{elapsed_sec:.1f}s (max_wallclock_seconds={args.max_wallclock_seconds}, iterations={args.iterations})")
            break

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        grad_scale = 1.0 / grad_accum_steps

        for _ in range(grad_accum_steps):
            # Determine training seq_len (curriculum if enabled)
            if args.seq_len_curriculum and step < args.seq_len_curriculum_steps:
                cur_seq = args.short_train_seq_len
            else:
                cur_seq = args.train_seq_len

            retries = 0
            while True:
                x, y = train_loader.next_batch(args.micro_batch_tokens, cur_seq)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                loss_val = loss.item()

                if loss_filter is not None and loss_filter.should_skip(loss_val):
                    retries += 1
                    if retries >= args.loss_filter_max_retries:
                        loss_filter.force_accept(loss_val)
                        (loss * grad_scale).backward()
                        step_loss += loss_val * grad_scale
                        break
                    continue
                else:
                    (loss * grad_scale).backward()
                    step_loss += loss_val * grad_scale
                    break

        if loss_filter is not None and step % 100 == 0:
            log0(f"[filter] totals {loss_filter.summary()}")

        # LR schedule: linear warmup then cosine decay
        elapsed_sec = time.perf_counter() - global_start_time
        global_ramp = min(step / max(args.warmup_steps, 1), 1.0)
        cosine_decay = args.cosine_min + (1.0 - args.cosine_min) * 0.5 * (
            1.0 + math.cos(min(elapsed_sec / args.max_wallclock_seconds, 1.0) * math.pi)
        )
        total_scale = global_ramp * cosine_decay

        # Muon momentum warmup
        frac = min(step / max(args.warmup_steps, 1), 1.0)
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum

        # Gradient div by num_steps (needed for multi-step recurrent models)
        if args.num_steps > 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(args.num_steps)

        # Optional dynamic gradient norm scaling
        if args.dynamic_lr_norm:
            gnorm = torch.nn.utils.get_total_norm(
                [p.grad for p in model.parameters() if p.grad is not None], norm_type=2
            )
            if gnorm > 0:
                dyn_scale = min(1.0, args.target_grad_norm / (gnorm + 1e-6))
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(dyn_scale)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
            group["lr"] = args.matrix_lr * total_scale

        for opt in optimizers:
            if opt is not optimizer_muon:
                for group in opt.param_groups:
                    group["lr"] = group["target_lr"] * total_scale
            opt.step()

        with torch.no_grad():
            for n, p in model.named_parameters():
                ema_n = n.replace("module.", "")
                if ema_n in model_ema:
                    model_ema[ema_n].mul_(0.99).add_(p.data, alpha=0.01)

        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        training_time_ms += dt
        t0 = time.perf_counter()
        shard = train_loader.stream.file_idx
        pos = train_loader.stream.pos
        log0(f"step:{step} loss:{step_loss:.4f} dt:{dt:.2f}ms d:sh{shard}p{pos} seqlen:{args.train_seq_len} steps:{args.num_steps}")

        step += 1

    # --- Final export ---
    log0(f"[final] Starting final export...")
    use_best = args.save_best_checkpoint and best_ema is not None and args.export_best_checkpoint
    if use_best:
        log0(f"[final] Export source: BEST checkpoint step={best_step} best_val_loss={best_val_loss:.4f} best_val_bpb={best_val_bpb:.4f}")
        export_ema = best_ema
    else:
        log0(f"[final] Export source: FINAL EMA weights")
        export_ema = model_ema

    # Swap in export EMA weights
    original_params_final = {n: p.data.clone() for n, p in base_model.named_parameters()}
    for n, p in base_model.named_parameters():
        if n in export_ema:
            p.data.copy_(export_ema[n])
    final_sd = base_model.state_dict()
    sz_bytes = sum(v.numel() * v.element_size() for v in final_sd.values())

    log0(f"[final] Saving raw model to final_model.pt ...")
    torch.save(final_sd, "final_model.pt")
    log0(f"[final] Saved: final_model.pt ({sz_bytes / (1024**2):.2f} MiB)")

    log0(f"[final] Quantizing + zlib-compressing to final_model.int8.ptz ...")
    import io, zlib
    int8_sd, _ = quantize_state_dict_int8(final_sd)
    buf = io.BytesIO()
    torch.save(int8_sd, buf)
    payload = zlib.compress(buf.getvalue(), level=9)
    with open("final_model.int8.ptz", "wb") as f:
        f.write(payload)
    log0(f"[final] Saved: final_model.int8.ptz ({len(payload)/(1024**2):.2f} MiB) int8_payload_bytes={len(payload)} baseline_bytes={sz_bytes}")

    # Restore original weights before quant_eval
    for n, p in base_model.named_parameters():
        p.data.copy_(original_params_final[n])

    # --- Quant eval: compare FP vs INT8 ---
    if args.quant_eval:
        log0(f"[final][quant_eval] running FP vs INT8(dequantized) validation ...")
        # Swap in export EMA for FP eval
        for n, p in base_model.named_parameters():
            if n in export_ema:
                p.data.copy_(export_ema[n])
        fp_loss, fp_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                   val_tokens, base_bytes_lut, has_leading_space_lut,
                                   is_boundary_token_lut,
                                   max_steps=args.quant_eval_max_steps,
                                   stride=args.quant_eval_stride, ttt_lr=args.ttt_lr)
        # Swap in dequantized INT8 weights
        dq_sd = dequantize_state_dict_int8(int8_sd)
        base_model.load_state_dict(dq_sd, strict=False)
        int8_loss, int8_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                       val_tokens, base_bytes_lut, has_leading_space_lut,
                                       is_boundary_token_lut,
                                       max_steps=args.quant_eval_max_steps,
                                       stride=args.quant_eval_stride, ttt_lr=args.ttt_lr)
        delta_loss = int8_loss - fp_loss
        delta_bpb = int8_bpb - fp_bpb
        pct = (delta_bpb / fp_bpb) * 100.0 if fp_bpb > 0 else 0.0
        sign = "+" if delta_bpb >= 0 else ""
        log0(f"[final][quant_eval] fp_val_loss={fp_loss:.6f} fp_val_bpb={fp_bpb:.6f} | int8_val_loss={int8_loss:.6f} int8_val_bpb={int8_bpb:.6f}")
        log0(f"[final][quant_eval] delta_loss={sign}{delta_loss:.6f} delta_bpb={sign}{delta_bpb:.6f} bpb_degradation_pct={sign}{pct:.3f}%")
        # Restore export EMA weights as final state
        for n, p in base_model.named_parameters():
            if n in export_ema:
                p.data.copy_(export_ema[n])

    log0(f"[final] Export complete.")


if __name__ == "__main__":
    main()
