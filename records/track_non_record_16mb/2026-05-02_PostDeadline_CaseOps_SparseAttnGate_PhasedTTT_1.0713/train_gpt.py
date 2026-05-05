import collections, copy, fcntl, glob, io, lzma, math, os, random, re, subprocess, sys, time, uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor, nn

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    from flash_attn_interface import flash_attn_varlen_func
    HAS_FA3 = True
except ImportError:
    HAS_FA3 = False
    flash_attn_varlen_func = None

import triton
import triton.language as tl


# ===== Fused softcapped cross-entropy (Triton) — training-only path =====
# Replaces the eager
#     logits_softcap = softcap * tanh(logits / softcap)
#     F.cross_entropy(logits_softcap.float(), targets, reduction="mean")
# sequence with a single fused kernel that reads logits_proj once, applies
# softcap in-register, and computes (LSE, loss) in one streaming pass. The
# backward kernel mirrors the forward so there's no stored softcapped logits.
# Numerically equivalent to the eager path: kernel computes z = 2C·sigmoid(x/2C)
# = C·tanh(x/2C) + C, where C=softcap; the +C is a constant shift across the
# vocab dim and cancels exactly in CE (matches eager up to fp32 rounding).
_FUSED_CE_LIBRARY = "parametergolf_fused_ce"
_FUSED_CE_BLOCK_SIZE = 1024
_FUSED_CE_NUM_WARPS = 4


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
    if logits.ndim != 2:
        raise ValueError(f"Expected logits.ndim=2, got {logits.ndim}")
    if targets.ndim != 1:
        raise ValueError(f"Expected targets.ndim=1, got {targets.ndim}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(f"Row mismatch: logits={tuple(logits.shape)} targets={tuple(targets.shape)}")
    if not logits.is_cuda or not targets.is_cuda:
        raise ValueError("softcapped_cross_entropy requires CUDA tensors")
    if softcap <= 0.0:
        raise ValueError(f"softcap must be positive, got {softcap}")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported logits dtype: {logits.dtype}")
    logits = logits.contiguous()
    targets = targets.contiguous()
    if targets.dtype != torch.int64:
        targets = targets.to(dtype=torch.int64)
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
def _(logits, targets, softcap):
    n_rows = logits.shape[0]
    return (
        logits.new_empty((n_rows,), dtype=torch.float32),
        logits.new_empty((n_rows,), dtype=torch.float32),
    )


@torch.library.custom_op(f"{_FUSED_CE_LIBRARY}::softcapped_ce_backward", mutates_args=())
def softcapped_ce_backward_op(
    logits: Tensor, targets: Tensor, lse: Tensor, grad_losses: Tensor, softcap: float,
) -> Tensor:
    logits, targets = _validate_softcapped_ce_inputs(logits, targets, float(softcap))
    lse = lse.contiguous()
    grad_losses = grad_losses.contiguous().to(dtype=torch.float32)
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
def _(logits, targets, lse, grad_losses, softcap):
    return logits.new_empty(logits.shape)


def _softcapped_ce_setup_context(ctx, inputs, output):
    logits, targets, softcap = inputs
    _losses, lse = output
    ctx.save_for_backward(logits, targets, lse)
    ctx.softcap = float(softcap)


def _softcapped_ce_backward(ctx, grad_losses, grad_lse):
    del grad_lse
    logits, targets, lse = ctx.saved_tensors
    grad_logits = torch.ops.parametergolf_fused_ce.softcapped_ce_backward(
        logits, targets, lse, grad_losses, ctx.softcap,
    )
    return grad_logits, None, None


softcapped_ce_op.register_autograd(
    _softcapped_ce_backward, setup_context=_softcapped_ce_setup_context,
)


def softcapped_cross_entropy(logits, targets, softcap, reduction="mean"):
    losses, _lse = torch.ops.parametergolf_fused_ce.softcapped_ce(
        logits, targets, float(softcap),
    )
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError(f"Unsupported reduction={reduction!r}")


def _e(key, default, cast=str):
    v = os.environ.get(key, default)
    return cast(v) if cast != str else v

def _ebool(key, default): return bool(int(os.environ.get(key, default)))

class Hyperparameters():
    data_dir = _e('DATA_DIR', './data/'); seed = _e('SEED', 1337, int); run_id = _e("RUN_ID", str(uuid.uuid4()))
    iterations = _e('ITERATIONS', 20000, int); warmdown_frac = _e('WARMDOWN_FRAC', 0.3, float)
    warmup_steps = _e('WARMUP_STEPS', 20, int); train_batch_tokens = _e('TRAIN_BATCH_TOKENS', 2048*48*8, int)
    train_seq_len = _e('TRAIN_SEQ_LEN', 2048, int); train_log_every = _e('TRAIN_LOG_EVERY', 500, int)
    max_wallclock_seconds = _e('MAX_WALLCLOCK_SECONDS', 600.0, float)
    profile_steps = _e('PROFILE_STEPS', 0, int)  # >0: profile N steps after warmup, dump table, exit
    cuda_graphs = _ebool('CUDA_GRAPHS', '1')  # mode="reduce-overhead" on the train compile
    fused_ce = _ebool('FUSED_CE', '1')  # Triton fused softcapped-CE on training forward only
    val_batch_tokens = _e('VAL_BATCH_TOKENS', 2048*32*8, int); eval_seq_len = _e('EVAL_SEQ_LEN', 2048, int)
    val_loss_every = _e('VAL_LOSS_EVERY', 4000, int); sliding_window_enabled = _ebool('SLIDING_WINDOW_ENABLED', '1')
    vocab_size = _e('VOCAB_SIZE', 8192, int); num_layers = _e('NUM_LAYERS', 11, int)
    xsa_last_n = _e('XSA_LAST_N', 11, int); model_dim = _e('MODEL_DIM', 512, int)
    embedding_dim = _e('EMBEDDING_DIM', 512, int); num_kv_heads = _e('NUM_KV_HEADS', 4, int)
    num_heads = _e('NUM_HEADS', 8, int); mlp_mult = _e('MLP_MULT', 4.0, float)
    skip_gates_enabled = _ebool('SKIP_GATES_ENABLED', '1'); tie_embeddings = _ebool('TIE_EMBEDDINGS', '1')
    logit_softcap = _e('LOGIT_SOFTCAP', 30.0, float); rope_base = _e('ROPE_BASE', 10000.0, float)
    rope_dims = _e('ROPE_DIMS', 16, int); rope_train_seq_len = _e('ROPE_TRAIN_SEQ_LEN', 2048, int)
    ln_scale = _ebool('LN_SCALE', '1'); qk_gain_init = _e('QK_GAIN_INIT', 5.25, float)
    parallel_residual_start = _e('PARALLEL_RESIDUAL_START', '7', int)
    num_loops = _e('NUM_LOOPS', 2, int); loop_start = _e('LOOP_START', 3, int)
    loop_end = _e('LOOP_END', 5, int); enable_looping_at = _e('ENABLE_LOOPING_AT', 0.5, float)
    smear_gate_enabled = _ebool('SMEAR_GATE_ENABLED', '0'); gate_window = _e('GATE_WINDOW', 12, int)
    # SparseAttnGate (modded-nanogpt-style narrow output gate). Per-head sigmoid gate
    # on attention output, gate input is x[..., :gate_window]. W_g shape (num_heads,
    # gate_window) = (8, 12) = 96 params/layer. Zero-init -> sigmoid(0)=0.5 -> safe
    # cold start. Independent of smear_gate (different mechanism — attention output
    # gating vs token mixing — leader stacks both). Reuses GATE_WINDOW for the
    # input-slice width. Note: leader has THREE attention gates (attn_out_gate,
    # gated_attn, sparse_attn_gate) which ARE mutually exclusive with each other;
    # we only ported sparse_attn_gate.
    sparse_attn_gate_enabled = _ebool('SPARSE_ATTN_GATE_ENABLED', '0')
    sparse_attn_gate_init_std = _e('SPARSE_ATTN_GATE_INIT_STD', 0.0, float)
    sparse_attn_gate_scale = _e('SPARSE_ATTN_GATE_SCALE', 1.0, float)
    prores_enabled = _ebool('PRORES_ENABLED', '0'); prores_window = _e('PRORES_WINDOW', 0.5, float)
    min_lr = _e('MIN_LR', 0.0, float); embed_lr = _e('EMBED_LR', 0.6, float)
    head_lr = _e('HEAD_LR', 0.008, float); tied_embed_lr = _e('TIED_EMBED_LR', 0.03, float)
    tied_embed_init_std = _e('TIED_EMBED_INIT_STD', 0.005, float); matrix_lr = _e('MATRIX_LR', 0.022, float)
    scalar_lr = _e('SCALAR_LR', 0.02, float); muon_momentum = _e('MUON_MOMENTUM', 0.99, float)
    muon_backend_steps = _e('MUON_BACKEND_STEPS', 5, int)
    muon_momentum_warmup_start = _e('MUON_MOMENTUM_WARMUP_START', 0.92, float)
    muon_momentum_warmup_steps = _e('MUON_MOMENTUM_WARMUP_STEPS', 1500, int)
    muon_row_normalize = _ebool('MUON_ROW_NORMALIZE', '1')
    beta1 = _e('BETA1', 0.9, float); beta2 = _e('BETA2', 0.95, float)
    adam_eps = _e('ADAM_EPS', 1e-8, float); grad_clip_norm = _e('GRAD_CLIP_NORM', 0.3, float)
    eval_stride = _e('EVAL_STRIDE', 64, int); muon_beta2 = _e('MUON_BETA2', 0.95, float)
    adam_wd = _e('ADAM_WD', 0.02, float); muon_wd = _e('MUON_WD', 0.095, float)
    embed_wd = _e('EMBED_WD', 0.095, float)
    ema_decay = _e('EMA_DECAY', 0.9965, float)
    compressor = _e('COMPRESSOR', 'brotli'); gptq_calibration_batches = _e('GPTQ_CALIBRATION_BATCHES', 64, int)
    gptq_reserve_seconds = _e('GPTQ_RESERVE_SECONDS', 12.0, float)
    matrix_bits = _e('MATRIX_BITS', 6, int); embed_bits = _e('EMBED_BITS', 8, int)
    matrix_clip_sigmas = _e('MATRIX_CLIP_SIGMAS', 12.85, float)
    embed_clip_sigmas = _e('EMBED_CLIP_SIGMAS', 20.0, float)
    # Per-tensor-group clip sigmas (ported from leader). Sentinel <=0 means
    # "fall back to matrix_clip_sigmas" — preserves current behavior when
    # MLP_CLIP_SIGMAS / ATTN_CLIP_SIGMAS are unset. Leader's record uses
    # MLP=10.0, ATTN=13.0 for SP12288/PhasedTTT/LQER stack.
    mlp_clip_sigmas = _e('MLP_CLIP_SIGMAS', 0.0, float)
    attn_clip_sigmas = _e('ATTN_CLIP_SIGMAS', 0.0, float)
    hadamard_rotate = _ebool('HADAMARD_ROTATE', '0')
    lqer_enabled = _ebool('LQER_ENABLED', '0'); lqer_rank = _e('LQER_RANK', 4, int)
    lqer_top_k = _e('LQER_TOP_K', 3, int); lqer_factor_bits = _e('LQER_FACTOR_BITS', 4, int)
    lqer_asym_enabled = _ebool('LQER_ASYM_ENABLED', '1'); lqer_asym_group = _e('LQER_ASYM_GROUP', 64, int)
    # Per-category bit allocation (0 = fall back to matrix_bits). Enables mixed-bit GPTQ
    # where low-sensitivity layers use fewer bits. E.g. Hadamard-rotated attn.kv + mlp.fc
    # can drop to int4 while q/proj/mlp.proj stay at int5.
    matrix_bits_kv = _e('MATRIX_BITS_KV', 0, int)
    matrix_bits_mlp_fc = _e('MATRIX_BITS_MLP_FC', 0, int)
    matrix_bits_mlp_proj = _e('MATRIX_BITS_MLP_PROJ', 0, int)
    # Legal score-first test-time training (TTT). Off by default. When on,
    # eval_val_ttt replaces the headline final_bpb: per chunk, score under
    # inference_mode (recorded into BPB), then SGD-adapt the dequantized
    # eval_model on the same chunk before moving on.
    ttt_enabled = _ebool('TTT_ENABLED', '0')
    ttt_chunk_tokens = _e('TTT_CHUNK_TOKENS', 32768, int)
    ttt_epochs = _e('TTT_EPOCHS', 3, int)
    ttt_lr = _e('TTT_LR', 0.005, float)
    ttt_momentum = _e('TTT_MOMENTUM', 0.9, float)
    # Phased TTT + LoRA (port from upstream/records/2026-04-27 leader). Gated by
    # PHASED_TTT_ENABLED=1; when off, all phased_*/global_ttt_*/ttt_lora_* fields
    # are read but unused. The simple TTT path above stays as fallback.
    ttt_lora_rank = _e('TTT_LORA_RANK', 96, int)
    ttt_lora_lr = _e('TTT_LORA_LR', 0.0001, float)
    ttt_lora_alpha = _e('TTT_LORA_ALPHA', 144, int)
    ttt_chunk_size = _e('TTT_CHUNK_SIZE', 48, int)
    ttt_eval_seq_len = _e('TTT_EVAL_SEQ_LEN', 2048, int)
    ttt_batch_size = _e('TTT_BATCH_SIZE', 64, int)
    ttt_grad_steps = _e('TTT_GRAD_STEPS', 1, int)
    ttt_weight_decay = _e('TTT_WEIGHT_DECAY', 1.0, float)
    ttt_beta1 = _e('TTT_BETA1', 0.0, float)
    ttt_beta2 = _e('TTT_BETA2', 0.999, float)
    ttt_k_lora = _ebool('TTT_K_LORA', '1')
    ttt_mlp_lora = _ebool('TTT_MLP_LORA', '1')
    ttt_o_lora = _ebool('TTT_O_LORA', '1')
    ttt_optimizer = _e('TTT_OPTIMIZER', 'adam', str)
    ttt_warm_start_a = _ebool('TTT_WARM_START_A', '1')
    ttt_eval_batches = _e('TTT_EVAL_BATCHES', '', str)
    val_doc_fraction = _e('VAL_DOC_FRACTION', 1.0, float)
    phased_ttt_enabled = _ebool('PHASED_TTT_ENABLED', '0')
    phased_ttt_prefix_docs = _e('PHASED_TTT_PREFIX_DOCS', 2500, int)
    phased_ttt_num_phases = _e('PHASED_TTT_NUM_PHASES', 3, int)
    global_ttt_lr = _e('GLOBAL_TTT_LR', 0.001, float)
    global_ttt_momentum = _e('GLOBAL_TTT_MOMENTUM', 0.9, float)
    global_ttt_epochs = _e('GLOBAL_TTT_EPOCHS', 1, int)
    global_ttt_chunk_tokens = _e('GLOBAL_TTT_CHUNK_TOKENS', 32768, int)
    global_ttt_batch_seqs = _e('GLOBAL_TTT_BATCH_SEQS', 32, int)
    global_ttt_warmup_start_lr = _e('GLOBAL_TTT_WARMUP_START_LR', 0.0, float)
    global_ttt_warmup_chunks = _e('GLOBAL_TTT_WARMUP_CHUNKS', 0, int)
    global_ttt_grad_clip = _e('GLOBAL_TTT_GRAD_CLIP', 1.0, float)
    global_ttt_respect_doc_boundaries = _ebool('GLOBAL_TTT_RESPECT_DOC_BOUNDARIES', '1')
    # Distributed
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = _e("RANK", "0", int); world_size = _e("WORLD_SIZE", "1", int)
    local_rank = _e("LOCAL_RANK", "0", int); is_main_process = rank == 0
    grad_accum_steps = 8 // world_size
    # Optional lossless text transform applied at tokenizer-train and data-export time.
    # Empty = identity (vanilla path). When set (e.g. 'lossless_caps_caseops_v1'),
    # build_sentencepiece_luts uses lossless_caps.surface_piece_original_byte_counts
    # so BPB is scored against original UTF-8 bytes, not transformed-stream bytes.
    text_transform = _e('TEXT_TRANSFORM', '')
    # Derived paths — overridable so caseops/alt datasets can coexist with vanilla.
    datasets_dir = _e('DATASETS_DIR',
                      os.path.join(data_dir, 'datasets', f'fineweb10B_sp{vocab_size}'))
    train_files = _e('TRAIN_FILES', os.path.join(datasets_dir, 'fineweb_train_*.bin'))
    val_files = _e('VAL_FILES', os.path.join(datasets_dir, 'fineweb_val_*.bin'))
    tokenizer_path = _e('TOKENIZER_PATH',
                        os.path.join(data_dir, 'tokenizers', f'fineweb_{vocab_size}_bpe.model'))
    logfile = f"logs/{run_id}.txt"; model_path = "final_model.pt"
    quantized_model_path = "final_model.int6.ptz"


_logger_hparams = None


def set_logging_hparams(h: Hyperparameters) -> None:
    global _logger_hparams
    _logger_hparams = h


def log(msg, console: bool = True) -> None:
    if _logger_hparams is None:
        print(msg)
        return
    if _logger_hparams.is_main_process:
        if console:
            print(msg)
        if _logger_hparams.logfile is not None:
            with open(_logger_hparams.logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)


class ValidationData:
    def __init__(self, h: Hyperparameters, device: torch.device):
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        if int(self.sp.vocab_size()) != h.vocab_size:
            raise ValueError(
                f"VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={int(self.sp.vocab_size())}"
            )
        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = (
            build_sentencepiece_luts(self.sp, h.vocab_size, device,
                                     text_transform=h.text_transform))


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device,
    text_transform: str = "",
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    # The BPB calculation assumes "▁" is its own token so that leading-space bytes
    # are counted correctly. See https://github.com/openai/parameter-golf/issues/897
    assert sp.piece_to_id("\u2581") != sp.unk_id(), \
        "Tokenizer must have '▁' (space) as its own token for correct BPB byte counting"
    # Per-token original-byte count. Under a text_transform like caseops_v1, surface
    # pieces in the transformed stream may include zero-byte sentinels
    # (TITLE/ALLCAPS/CAPNEXT/ESC); use lossless_caps.surface_piece_original_byte_counts
    # to map back to original UTF-8 bytes. Per-token counts are well-defined as long as
    # the sentinels are reserved as standalone (user-defined) tokens —
    # train_tokenizer_caseops.py guarantees this.
    if text_transform == "lossless_caps_caseops_v1":
        # Per-piece original-byte count, isolated (no cross-piece state). The reserved
        # sentinels (TITLE/ALLCAPS/CAPNEXT/ESC) contribute 0 bytes; ASCII alpha chars
        # always contribute 1 (case is set by preceding-marker context, but bytes are
        # invariant to case); other chars contribute their UTF-8 length. We do not call
        # lossless_caps.surface_piece_original_byte_counts directly because it enforces
        # sequence-completeness and rejects a "dangling" sentinel piece in isolation.
        # Edge case: literal escaped sentinel (ESC followed by sentinel) within ONE
        # piece is rare in fineweb (private-use codepoints U+E001-E004); within-piece
        # ESC handling honored for robustness.
        from lossless_caps import (
            DEFAULT_V2_TITLE, DEFAULT_V2_ALLCAPS, DEFAULT_V2_CAPNEXT, DEFAULT_V2_ESC,
        )
        _CASEOPS_SENTINELS = {DEFAULT_V2_TITLE, DEFAULT_V2_ALLCAPS,
                              DEFAULT_V2_CAPNEXT, DEFAULT_V2_ESC}
        def _piece_bytes(p: str) -> int:
            n = 0
            pending_escape = False
            for ch in p:
                if pending_escape:
                    n += len(ch.encode("utf-8"))
                    pending_escape = False
                elif ch == DEFAULT_V2_ESC:
                    pending_escape = True
                elif ch in _CASEOPS_SENTINELS:
                    pass  # 0 bytes — case marker
                elif "a" <= ch <= "z" or "A" <= ch <= "Z":
                    n += 1
                else:
                    n += len(ch.encode("utf-8"))
            return n
    elif text_transform:
        raise ValueError(f"unsupported TEXT_TRANSFORM={text_transform!r}")
    else:
        def _piece_bytes(p: str) -> int:
            return len(p.encode("utf-8"))
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
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = _piece_bytes(piece)
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


_SHARD_HEADER_BYTES = 256 * np.dtype("<i4").itemsize
_SHARD_NTOKENS_CACHE: dict[str, int] = {}
_MMAP_CACHE: dict[str, np.memmap] = {}


def _read_num_tokens(file: Path) -> int:
    key = str(file)
    cached = _SHARD_NTOKENS_CACHE.get(key)
    if cached is not None:
        return cached
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    n = int(header[2])
    _SHARD_NTOKENS_CACHE[key] = n
    return n


def _get_shard_memmap(file: Path) -> np.memmap:
    key = str(file)
    mm = _MMAP_CACHE.get(key)
    if mm is not None:
        return mm
    n = _read_num_tokens(file)
    mm = np.memmap(file, mode="r", dtype="<u2", offset=_SHARD_HEADER_BYTES, shape=(n,))
    _MMAP_CACHE[key] = mm
    return mm


class ShuffledSequenceLoader:
    def __init__(self, h: 'Hyperparameters', device: torch.device):
        self.world_size = h.world_size
        self.seq_len = h.train_seq_len
        self.device = device
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files:
            raise FileNotFoundError(f"No files found for pattern: {h.train_files}")
        self.files = all_files[h.rank::h.world_size]
        self.rng = np.random.Generator(np.random.PCG64(h.rank))
        self.num_tokens = [_read_num_tokens(f) for f in self.files]
        self.start_inds: list[list[int]] = [[] for _ in self.files]
        for si in range(len(self.files)):
            self._reset_shard(si)

    def _reset_shard(self, si: int) -> None:
        max_phase = min(self.seq_len - 1, max(0, self.num_tokens[si] - self.seq_len - 1))
        phase = int(self.rng.integers(max_phase + 1)) if max_phase > 0 else 0
        num_sequences = (self.num_tokens[si] - 1 - phase) // self.seq_len
        sequence_order = self.rng.permutation(num_sequences)
        self.start_inds[si] = (phase + sequence_order * self.seq_len).tolist()

    def next_batch(self, global_tokens: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
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
            window = torch.as_tensor(
                np.array(mm[start_ind:start_ind + self.seq_len + 1], dtype=np.int64))
            x[bi] = window[:-1]
            y[bi] = window[1:]
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        # No caching: cached tensors persist across cudagraph captures and get clobbered under
        # mode="reduce-overhead". Recompute every call — cheap vs attention, cudagraph-safe.
        rd = self.rope_dims
        if seq_len > self.train_seq_len:
            scale = seq_len / self.train_seq_len
            new_base = self.base * (scale ** (rd / (rd - 2)))
            inv_freq = 1.0 / (new_base ** (torch.arange(
                0, rd, 2, dtype=torch.float32, device=device) / rd))
        else:
            inv_freq = self.inv_freq.to(device)
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()[None, :, None, :]
        sin = freqs.sin()[None, :, None, :]
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float, train_seq_len: int,
                 sparse_attn_gate: bool = False, sparse_attn_gate_init_std: float = 0.0,
                 sparse_attn_gate_scale: float = 1.0, gate_window: int = 12):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
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
        # SparseAttnGate (modded-nanogpt). Per-head sigmoid gate on SDPA output,
        # gate input is x[..., :gate_window]. W_g: (num_heads, gate_window) = 96
        # params/layer. Zero-init -> sigmoid(0)=0.5 (transparent-ish at start).
        # Weight is small (<= 65536 numel), so quant pipeline auto-routes it to the
        # float16 passthrough branch in gptq_mixed_quantize.
        self.sparse_attn_gate = sparse_attn_gate
        self.sparse_attn_gate_scale = sparse_attn_gate_scale
        self.gate_window = gate_window
        if sparse_attn_gate:
            W = torch.empty(num_heads, gate_window, dtype=torch.float32)
            if sparse_attn_gate_init_std > 0:
                nn.init.normal_(W, mean=0.0, std=sparse_attn_gate_init_std)
            else:
                nn.init.zeros_(W)
            self.attn_gate_w = nn.Parameter(W)

    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x: Tensor, cu_seqlens=None, max_seqlen: int = 0) -> Tensor:
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
        if cu_seqlens is not None and HAS_FA3:
            # Varlen path: q/k/v shape (1, T, H, D) -> (T, H, D) per FA3 varlen API.
            y = flash_attn_varlen_func(
                q[0], k[0], v[0],
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                causal=True,
            )[None]
        elif HAS_FA3:
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads),
            ).transpose(1, 2)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        # SparseAttnGate: per-head sigmoid gate, narrow input x[..., :gate_window].
        # Inline + .contiguous() barrier to keep torch.compile fullgraph=True happy.
        # g shape [B,T,H], broadcast over D via [..., None]. .to(x.dtype) on fp32
        # param before bf16 multiply. Zero-init -> sigmoid(0)=0.5 -> safe cold start.
        if self.sparse_attn_gate:
            gate_in = x[..., : self.gate_window].contiguous()
            g = torch.sigmoid(
                self.sparse_attn_gate_scale
                * F.linear(gate_in, self.attn_gate_w.to(x.dtype))
            )
            y = y * g[..., None]
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, train_seq_len: int,
                 layer_idx: int = 0, ln_scale: bool = False,
                 prores_enabled: bool = False, prores_window: float = 0.5,
                 num_total_layers: int = 1,
                 sparse_attn_gate: bool = False,
                 sparse_attn_gate_init_std: float = 0.0,
                 sparse_attn_gate_scale: float = 1.0,
                 gate_window: int = 12):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len,
            sparse_attn_gate=sparse_attn_gate,
            sparse_attn_gate_init_std=sparse_attn_gate_init_std,
            sparse_attn_gate_scale=sparse_attn_gate_scale,
            gate_window=gate_window)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        self.parallel = False
        self.prores_enabled = prores_enabled
        self.prores_window = prores_window
        denom = max(num_total_layers - 1, 1)
        self.prores_t_offset = (layer_idx / denom) * (1.0 - prores_window)
        self.register_buffer("prores_alpha",
                             torch.ones(1, dtype=torch.float32), persistent=False)

    def prores_alpha_value(self, frac: float) -> float:
        if not self.prores_enabled or self.prores_window <= 0.0:
            return 1.0
        a = (frac - self.prores_t_offset) / self.prores_window
        return max(0.0, min(1.0, a))

    def forward(self, x: Tensor, x0: Tensor, cu_seqlens=None, max_seqlen: int = 0) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(
            self.attn_norm(x_in) * self.ln_scale_factor,
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
        )
        if self.prores_enabled:
            alpha = self.prores_alpha.to(dtype=x_in.dtype)
            attn_scale = self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * alpha
            mlp_scale = self.mlp_scale.to(dtype=x_in.dtype)[None, None, :] * alpha
        else:
            attn_scale = self.attn_scale.to(dtype=x_in.dtype)[None, None, :]
            mlp_scale = self.mlp_scale.to(dtype=x_in.dtype)[None, None, :]
        if self.parallel:
            mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor)
            return x_in + attn_scale * attn_out + mlp_scale * mlp_out
        x_out = x_in + attn_scale * attn_out
        x_out = x_out + mlp_scale * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        return x_out


class GPT(nn.Module):
    def __init__(self, h: Hyperparameters):
        super().__init__()
        if h.logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {h.logit_softcap}")
        self.tie_embeddings = h.tie_embeddings
        self.tied_embed_init_std = h.tied_embed_init_std
        self.logit_softcap = h.logit_softcap
        self.fused_ce = h.fused_ce
        self.tok_emb = nn.Embedding(h.vocab_size, h.embedding_dim)
        self.smear_gate_enabled = h.smear_gate_enabled
        if self.smear_gate_enabled:
            sp_for_bos = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
            self.bos_id = int(sp_for_bos.bos_id())
            if self.bos_id < 0:
                raise ValueError(
                    f"Tokenizer at {h.tokenizer_path} has no BOS id; SmearGate BOS mask requires one")
            self.smear_window = h.gate_window
            self.smear_gate = CastedLinear(self.smear_window, 1, bias=False)
            self.smear_gate._zero_init = True
            self.smear_lambda = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        if h.embedding_dim != h.model_dim:
            self.embed_proj = CastedLinear(h.embedding_dim, h.model_dim, bias=False)
            self.head_proj = CastedLinear(h.model_dim, h.embedding_dim, bias=False)
        else:
            self.embed_proj = None
            self.head_proj = None
        self.num_encoder_layers = h.num_layers // 2
        self.num_decoder_layers = h.num_layers - self.num_encoder_layers
        self.prores_enabled = h.prores_enabled
        self.blocks = nn.ModuleList([
            Block(h.model_dim, h.num_heads, h.num_kv_heads, h.mlp_mult, h.rope_base,
                  h.qk_gain_init, h.train_seq_len, layer_idx=i, ln_scale=h.ln_scale,
                  prores_enabled=h.prores_enabled, prores_window=h.prores_window,
                  num_total_layers=h.num_layers,
                  sparse_attn_gate=h.sparse_attn_gate_enabled,
                  sparse_attn_gate_init_std=h.sparse_attn_gate_init_std,
                  sparse_attn_gate_scale=h.sparse_attn_gate_scale,
                  gate_window=h.gate_window)
            for i in range(h.num_layers)
        ])
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
        # Stored for forward_ttt() so the LoRA path can branch into the
        # parallel-residual variant on the matching block indices.
        self.parallel_start_layer = h.parallel_residual_start

        # Layer looping
        self.looping_active: bool = False
        if h.num_loops > 0:
            loop_seg = list(range(h.loop_start, h.loop_end + 1))
            all_indices = list(range(h.loop_start))
            for _ in range(h.num_loops + 1):
                all_indices.extend(loop_seg)
            all_indices.extend(range(h.loop_end + 1, h.num_layers))
            num_enc = len(all_indices) // 2
            self.encoder_indices: list[int] = all_indices[:num_enc]
            self.decoder_indices: list[int] = all_indices[num_enc:]
        else:
            self.encoder_indices = list(range(self.num_encoder_layers))
            self.decoder_indices = list(range(self.num_encoder_layers, h.num_layers))
        self.num_skip_weights = min(len(self.encoder_indices), len(self.decoder_indices))
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, h.model_dim, dtype=torch.float32))
        self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, h.model_dim, dtype=torch.float32)) if h.skip_gates_enabled else None

        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif (module.weight.ndim == 2 and module.weight.shape[0] >= 64 and
                      module.weight.shape[1] >= 64):
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def _forward_logits_pre_softcap(self, input_ids: Tensor, cu_seqlens=None, max_seqlen: int = 0) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.smear_gate_enabled:
            g = self.smear_lambda.to(dtype=x.dtype) * torch.sigmoid(
                self.smear_gate(x[:, 1:, :self.smear_window].contiguous()))
            g = g.masked_fill((input_ids[:, 1:] == self.bos_id).unsqueeze(-1), 0.0)
            x = torch.cat([x[:, :1], x[:, 1:] + g * x[:, :-1]], dim=1)
        x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x0 = x
        skips: list[Tensor] = []
        enc_iter = self.encoder_indices if self.looping_active else range(self.num_encoder_layers)
        dec_iter = self.decoder_indices if self.looping_active else range(self.num_encoder_layers, self.num_encoder_layers + self.num_decoder_layers)
        for i in enc_iter:
            x = self.blocks[i](x, x0, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            skips.append(x)
        for skip_idx, i in enumerate(dec_iter):
            if skip_idx < self.num_skip_weights and skips:
                scaled_skip = self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
                if self.skip_gates is not None:
                    g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                    x = torch.lerp(scaled_skip, x, g)
                else:
                    x = x + scaled_skip
            x = self.blocks[i](x, x0, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        if self.tie_embeddings:
            return F.linear(x, self.tok_emb.weight)
        return self.lm_head(x)

    def forward_logits(self, input_ids: Tensor, cu_seqlens=None, max_seqlen: int = 0) -> Tensor:
        logits_proj = self._forward_logits_pre_softcap(input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor, cu_seqlens=None, max_seqlen: int = 0) -> Tensor:
        logits_proj = self._forward_logits_pre_softcap(input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        flat_targets = target_ids.reshape(-1)
        if self.fused_ce:
            return softcapped_cross_entropy(
                logits_proj.reshape(-1, logits_proj.size(-1)),
                flat_targets, self.logit_softcap, reduction="mean")
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), flat_targets, reduction="mean")

    # ===== Phased TTT + LoRA forward path =====
    # Mirrors _forward_logits_pre_softcap + forward but injects per-slot LoRA
    # adapters into Q/K/V/O attention projections, MLP up-projection (fc), and
    # the LM head. Returns per-token NLL of shape (B, T) for use by the phased
    # eval loop's score-then-adapt protocol. Adapted from upstream leader's
    # forward_ttt to TARGET's simpler block structure (CausalSelfAttention with
    # c_q/c_k/c_v/proj, MLP with fc/proj+leaky_relu(0.5)^2, Block with
    # parallel-residual flag instead of lane structure).
    def forward_ttt(self, input_ids: Tensor, target_ids: Tensor, lora,
                    cu_seqlens=None, max_seqlen: int = 0) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.smear_gate_enabled:
            sl = self.smear_lambda.to(dtype=x.dtype)
            gate_in = x[:, 1:, : self.smear_window].contiguous()
            g = sl * torch.sigmoid(self.smear_gate(gate_in))
            bos_mask = (input_ids[:, 1:] == self.bos_id).unsqueeze(-1)
            g = g.masked_fill(bos_mask, 0.0)
            x = torch.cat([x[:, :1], x[:, 1:] + g * x[:, :-1]], dim=1)
        x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x0 = x
        skips: list[Tensor] = []
        enc_iter = (self.encoder_indices if self.looping_active
                    else list(range(self.num_encoder_layers)))
        dec_iter = (self.decoder_indices if self.looping_active
                    else list(range(self.num_encoder_layers,
                                    self.num_encoder_layers + self.num_decoder_layers)))
        slot = 0
        for i in enc_iter:
            x = self._block_with_lora(self.blocks[i], x, x0, lora, slot,
                                      cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            slot += 1
            skips.append(x)
        psl = self.parallel_start_layer
        for skip_idx, i in enumerate(dec_iter):
            if skip_idx < self.num_skip_weights and skips:
                scaled_skip = self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
                if self.skip_gates is not None:
                    g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                    x = torch.lerp(scaled_skip, x, g)
                else:
                    x = x + scaled_skip
            if psl >= 0 and i >= psl and self.blocks[i].parallel:
                x = self._parallel_block_with_lora(self.blocks[i], x, x0, lora, slot,
                                                   cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            else:
                x = self._block_with_lora(self.blocks[i], x, x0, lora, slot,
                                          cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            slot += 1
        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
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

    def _block_with_lora(self, block, x: Tensor, x0: Tensor, lora, slot: int,
                         cu_seqlens=None, max_seqlen: int = 0) -> Tensor:
        mix = block.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_n = block.attn_norm(x_in) * block.ln_scale_factor
        attn = block.attn
        bsz, seqlen, dim = attn_n.shape
        # Q/K/V projections + LoRA. Match CausalSelfAttention.forward shape contract.
        q = attn.c_q(attn_n) + lora.q_loras[slot](attn_n)
        q = q.reshape(bsz, seqlen, attn.num_heads, attn.head_dim)
        k = attn.c_k(attn_n)
        if lora.k_loras is not None:
            k = k + lora.k_loras[slot](attn_n)
        k = k.reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
        v = attn.c_v(attn_n) + lora.v_loras[slot](attn_n)
        v = v.reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = attn.rotary(seqlen, attn_n.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, attn.rope_dims)
        k = apply_rotary_emb(k, cos, sin, attn.rope_dims)
        q = q * attn.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if cu_seqlens is not None and HAS_FA3:
            y = flash_attn_varlen_func(
                q[0], k[0], v[0],
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                causal=True,
            )[None]
        elif HAS_FA3:
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=True, enable_gqa=(attn.num_kv_heads != attn.num_heads),
            ).transpose(1, 2)
        if attn.use_xsa:
            y = attn._xsa_efficient(y, v)
        # SparseAttnGate (TTT sequential path) — must match the eval path in
        # CausalSelfAttention.forward exactly so train (which applied the gate) and
        # TTT eval (which would otherwise skip it) stay in sync. Gate input is
        # attn_n (post-norm pre-attn), matching what `forward` calls `x`.
        if attn.sparse_attn_gate:
            gate_in = attn_n[..., : attn.gate_window].contiguous()
            g = torch.sigmoid(
                attn.sparse_attn_gate_scale
                * F.linear(gate_in, attn.attn_gate_w.to(attn_n.dtype))
            )
            y = y * g[..., None]
        y = y.reshape(bsz, seqlen, dim)
        attn_out = attn.proj(y)
        if lora.o_loras is not None:
            attn_out = attn_out + lora.o_loras[slot](attn_n)
        # Match Block.forward's prores semantics. Parallel-residual blocks use
        # _parallel_block_with_lora; this method handles only the sequential path.
        if block.prores_enabled:
            alpha = block.prores_alpha.to(dtype=x_in.dtype)
            attn_scale = block.attn_scale.to(dtype=x_in.dtype)[None, None, :] * alpha
            mlp_scale = block.mlp_scale.to(dtype=x_in.dtype)[None, None, :] * alpha
        else:
            attn_scale = block.attn_scale.to(dtype=x_in.dtype)[None, None, :]
            mlp_scale = block.mlp_scale.to(dtype=x_in.dtype)[None, None, :]
        x_out = x_in + attn_scale * attn_out
        mlp_n = block.mlp_norm(x_out) * block.ln_scale_factor
        fc_out = block.mlp.fc(mlp_n)
        if lora.mlp_loras is not None:
            fc_out = fc_out + lora.mlp_loras[slot](mlp_n)
        mlp_out = block.mlp.proj(F.leaky_relu(fc_out, negative_slope=0.5).square())
        x_out = x_out + mlp_scale * mlp_out
        return x_out

    def _parallel_block_with_lora(self, block, x: Tensor, x0: Tensor, lora, slot: int,
                                  cu_seqlens=None, max_seqlen: int = 0) -> Tensor:
        """LoRA-injected counterpart to Block.forward's parallel-residual branch.
        Both attn and mlp read from the same pre-norm x_in; output is a simple sum:
            return x_in + attn_scale * attn_out + mlp_scale * mlp_out
        Stays faithful to TARGET's existing parallel-sum semantics (NOT the
        leader's true 2-lane structure)."""
        mix = block.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_n = block.attn_norm(x_in) * block.ln_scale_factor
        attn = block.attn
        bsz, seqlen, dim = attn_n.shape
        q = attn.c_q(attn_n) + lora.q_loras[slot](attn_n)
        q = q.reshape(bsz, seqlen, attn.num_heads, attn.head_dim)
        k = attn.c_k(attn_n)
        if lora.k_loras is not None:
            k = k + lora.k_loras[slot](attn_n)
        k = k.reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
        v = attn.c_v(attn_n) + lora.v_loras[slot](attn_n)
        v = v.reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = attn.rotary(seqlen, attn_n.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, attn.rope_dims)
        k = apply_rotary_emb(k, cos, sin, attn.rope_dims)
        q = q * attn.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if cu_seqlens is not None and HAS_FA3:
            y = flash_attn_varlen_func(
                q[0], k[0], v[0],
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                causal=True,
            )[None]
        elif HAS_FA3:
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=True, enable_gqa=(attn.num_kv_heads != attn.num_heads),
            ).transpose(1, 2)
        if attn.use_xsa:
            y = attn._xsa_efficient(y, v)
        # SparseAttnGate (TTT parallel path) — same gate as eval / sequential TTT.
        if attn.sparse_attn_gate:
            gate_in = attn_n[..., : attn.gate_window].contiguous()
            g = torch.sigmoid(
                attn.sparse_attn_gate_scale
                * F.linear(gate_in, attn.attn_gate_w.to(attn_n.dtype))
            )
            y = y * g[..., None]
        y = y.reshape(bsz, seqlen, dim)
        attn_out = attn.proj(y)
        if lora.o_loras is not None:
            attn_out = attn_out + lora.o_loras[slot](attn_n)
        # MLP path reads same pre-norm x_in (parallel-residual signature).
        mlp_n = block.mlp_norm(x_in) * block.ln_scale_factor
        fc_out = block.mlp.fc(mlp_n)
        if lora.mlp_loras is not None:
            fc_out = fc_out + lora.mlp_loras[slot](mlp_n)
        mlp_out = block.mlp.proj(F.leaky_relu(fc_out, negative_slope=0.5).square())
        if block.prores_enabled:
            alpha = block.prores_alpha.to(dtype=x_in.dtype)
            attn_scale = block.attn_scale.to(dtype=x_in.dtype)[None, None, :] * alpha
            mlp_scale = block.mlp_scale.to(dtype=x_in.dtype)[None, None, :] * alpha
        else:
            attn_scale = block.attn_scale.to(dtype=x_in.dtype)[None, None, :]
            mlp_scale = block.mlp_scale.to(dtype=x_in.dtype)[None, None, :]
        return x_in + attn_scale * attn_out + mlp_scale * mlp_out


def classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"


@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
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
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0,
                 row_normalize: bool = False):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay,
                 row_normalize=row_normalize),
        )

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
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    if group.get("row_normalize", False):
                        row_norms = g.float().norm(dim=-1, keepdim=True).clamp_min(1e-07)
                        g = g / row_norms.to(g.dtype)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


CONTROL_TENSOR_NAME_PATTERNS = tuple(p for p in os.environ.get(
    "CONTROL_TENSOR_NAME_PATTERNS",
    "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates,smear_gate,smear_lambda,attn_gate_w",
).split(",") if p)


class Optimizers():
    def __init__(self, h: Hyperparameters, base_model: GPT):
        block_named_params = list(base_model.blocks.named_parameters())
        matrix_params = [
            p
            for name, p in block_named_params
            if p.ndim == 2 and not any(pattern in name for pattern in
                                       CONTROL_TENSOR_NAME_PATTERNS)
        ]
        scalar_params = [
            p
            for name, p in block_named_params
            if p.ndim < 2 or any(pattern in name for pattern in
                                 CONTROL_TENSOR_NAME_PATTERNS)
        ]
        if base_model.skip_weights.numel() > 0:
            scalar_params.append(base_model.skip_weights)
        if base_model.skip_gates is not None and base_model.skip_gates.numel() > 0:
            scalar_params.append(base_model.skip_gates)
        if base_model.smear_gate_enabled:
            scalar_params.extend([base_model.smear_gate.weight, base_model.smear_lambda])

        token_lr = h.tied_embed_lr if h.tie_embeddings else h.embed_lr
        tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
        self.optimizer_tok = torch.optim.AdamW(
            tok_params,
            betas=(h.beta1, h.beta2),
            eps=h.adam_eps,
            weight_decay=h.embed_wd,
            fused=True,
        )
        self.optimizer_muon = Muon(
            matrix_params,
            lr=h.matrix_lr,
            momentum=h.muon_momentum,
            backend_steps=h.muon_backend_steps,
            weight_decay=h.muon_wd,
            row_normalize=h.muon_row_normalize,
        )
        for group in self.optimizer_muon.param_groups:
            group["base_lr"] = h.matrix_lr
        self.optimizer_scalar = torch.optim.AdamW(
            [{"params": scalar_params, "lr": h.scalar_lr, "base_lr": h.scalar_lr}],
            betas=(h.beta1, h.beta2),
            eps=h.adam_eps,
            weight_decay=h.adam_wd,
            fused=True,
        )
        self.optimizers = [self.optimizer_tok, self.optimizer_muon, self.optimizer_scalar]
        if base_model.lm_head is not None:
            self.optimizer_head = torch.optim.Adam(
                [{"params": [base_model.lm_head.weight], "lr": h.head_lr, "base_lr": h.head_lr}],
                betas=(h.beta1, h.beta2),
                eps=h.adam_eps,
                fused=True,
            )
            self.optimizers.insert(1, self.optimizer_head)
        else:
            self.optimizer_head = None

    def __iter__(self):
        return iter(self.optimizers)

    def zero_grad_all(self) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def step(self):
        for opt in self.optimizers:
            opt.step()
        self.zero_grad_all()


def restore_fp32_params(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    for name, param in model.named_parameters():
        if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
            param.data = param.data.float()


def collect_hessians(
    model: nn.Module,
    train_loader: ShuffledSequenceLoader,
    h: Hyperparameters,
    device: torch.device,
    n_calibration_batches: int = 64,
) -> dict[str, Tensor]:
    hessians: dict[str, Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            if name not in hessians:
                hessians[name] = torch.zeros(
                    x.shape[1], x.shape[1], dtype=torch.float32, device=device
                )
            hessians[name].addmm_(x.T, x)
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, CastedLinear) and module.weight.numel() > 65536:
            cat = classify_param(name + ".weight")
            if cat in ("mlp", "attn"):
                hooks.append(module.register_forward_hook(make_hook(name + ".weight")))

    if model.tie_embeddings:
        hook_module = model.head_proj if model.head_proj is not None else model.final_norm
        def make_output_hook(name: str):
            def hook_fn(module, inp, out):
                x = out.detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        x.shape[1], x.shape[1], dtype=torch.float32, device=device
                    )
                hessians[name].addmm_(x.T, x)
            return hook_fn
        hooks.append(hook_module.register_forward_hook(make_output_hook("tok_emb.weight")))

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


def _hadamard(n: int) -> Tensor:
    """Walsh-Hadamard matrix of size n (must be power of 2), scaled by 1/sqrt(n)."""
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)

_hadamard_cache: dict[int, Tensor] = {}
def hadamard(n: int) -> Tensor:
    if n not in _hadamard_cache:
        _hadamard_cache[n] = _hadamard(n)
    return _hadamard_cache[n]


def gptq_quantize_weight(
    w: Tensor,
    H: Tensor,
    clip_sigmas: float = 3.0,
    clip_range: int = 63,
    block_size: int = 128,
) -> tuple[Tensor, Tensor]:
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

    return Q[:, invperm], s


def _lqer_pack(A: Tensor, B: Tensor, bits: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r = 2 ** (bits - 1) - 1
    sA = (A.abs().amax(dim=1).clamp_min(1e-10) / r).to(torch.float16)
    sB = (B.abs().amax(dim=1).clamp_min(1e-10) / r).to(torch.float16)
    qA = torch.clamp(torch.round(A / sA.float().view(-1, 1)), -r, r).to(torch.int8)
    qB = torch.clamp(torch.round(B / sB.float().view(-1, 1)), -r, r).to(torch.int8)
    return qA, sA, qB, sB


def _lqer_pack_asym(A: Tensor, B: Tensor, group: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    sA = (A.abs().amax().clamp_min(1e-10) / 1.5).to(torch.float16)
    qA = torch.clamp(torch.round(A / sA.float()), -2, 1).to(torch.int8)
    Bf = B.reshape(-1, group)
    sB = (Bf.abs().amax(dim=1, keepdim=True).clamp_min(1e-10) / 7.5).to(torch.float16)
    qB = torch.clamp(torch.round(Bf / sB.float()), -8, 7).to(torch.int8).reshape(B.shape)
    return qA, sA, qB, sB.reshape(-1)


def gptq_mixed_quantize(
    state_dict: dict[str, Tensor],
    hessians: dict[str, Tensor],
    h: Hyperparameters,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    lqer_cands: list[tuple[float, str, Tensor]] = []

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough (float16)"
            continue
        # Hadamard rotation: W_rot = W @ Q^T (rotate columns before quantizing)
        if h.hadamard_rotate and "tok_emb" not in name and t.ndim == 2:
            Q = hadamard(t.shape[1])
            t = t @ Q.T
            H_rot = Q @ hessians[name] @ Q.T  # rotate Hessian to match
        else:
            H_rot = hessians[name]
        # Per-tensor-group clip sigmas dispatch (split-clip port from leader).
        # Sentinel <=0 in mlp_clip_sigmas / attn_clip_sigmas falls back to
        # matrix_clip_sigmas, preserving current behavior when unset.
        if "tok_emb" in name:
            cs = h.embed_clip_sigmas
        elif ".mlp." in name and h.mlp_clip_sigmas > 0:
            cs = h.mlp_clip_sigmas
        elif ".attn." in name and h.attn_clip_sigmas > 0:
            cs = h.attn_clip_sigmas
        else:
            cs = h.matrix_clip_sigmas
        if "tok_emb" in name:
            bits = h.embed_bits
        elif h.matrix_bits_kv > 0 and ("attn.c_k" in name or "attn.c_v" in name):
            bits = h.matrix_bits_kv
        elif h.matrix_bits_mlp_fc > 0 and "mlp.fc" in name:
            bits = h.matrix_bits_mlp_fc
        elif h.matrix_bits_mlp_proj > 0 and "mlp.proj" in name:
            bits = h.matrix_bits_mlp_proj
        else:
            bits = h.matrix_bits
        q, s = gptq_quantize_weight(
            t, H_rot, clip_sigmas=cs, clip_range=2**(bits - 1) - 1)
        result[name + ".q"] = q
        result[name + ".scale"] = s
        meta[name] = f"gptq (int{bits})" + (" +hadamard" if h.hadamard_rotate and "tok_emb" not in name and tensor.ndim == 2 else "")
        if h.lqer_enabled and h.lqer_top_k > 0:
            wq = q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))
            E = t.float() - wq
            lqer_cands.append((float(E.norm()), name, E))
            lqer_cands = sorted(lqer_cands, key=lambda x: -x[0])[:h.lqer_top_k]

    if h.lqer_enabled and lqer_cands:
        for _, name, E in lqer_cands:
            U, S, Vh = torch.linalg.svd(E, full_matrices=False)
            r = min(h.lqer_rank, S.numel())
            A = (U[:, :r] * S[:r]).contiguous()
            B = Vh[:r, :].contiguous()
            if h.lqer_asym_enabled and B.numel() % h.lqer_asym_group == 0:
                qA, sA, qB, sB = _lqer_pack_asym(A, B, h.lqer_asym_group)
                result[name + ".lqA_a"] = qA; result[name + ".lqAs_a"] = sA
                result[name + ".lqB_a"] = qB; result[name + ".lqBs_a"] = sB
                meta[name] = str(meta[name]) + "+lqer_asym"
            else:
                qA, sA, qB, sB = _lqer_pack(A, B, h.lqer_factor_bits)
                result[name + ".lqA"] = qA; result[name + ".lqAs"] = sA
                result[name + ".lqB"] = qB; result[name + ".lqBs"] = sB
                meta[name] = str(meta[name]) + "+lqer"

    categories = collections.defaultdict(set)
    for name, cat in meta.items():
        short = re.sub(r'\.\d+$', '', re.sub(r'blocks\.\d+', 'blocks', name))
        categories[cat].add(short)
    log("Quantized weights:")
    for cat in sorted(categories):
        log(f"  {cat}: {', '.join(sorted(categories[cat]))}")

    return result, meta


def dequantize_mixed(result: dict[str, Tensor], meta: dict[str, object],
                     template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if "passthrough" in info:
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            w = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1))))
        else:
            w = (q.float() * float(s.item()))
        if name + ".lqA_a" in result:
            qA, sA = result[name + ".lqA_a"], result[name + ".lqAs_a"]
            qB, sB = result[name + ".lqB_a"], result[name + ".lqBs_a"]
            g = qB.numel() // sB.numel()
            A = qA.float() * sA.float()
            B = (qB.reshape(-1, g).float() * sB.float().view(-1, 1)).reshape(qB.shape)
            w = w + A @ B
        elif name + ".lqA" in result:
            qA, sA = result[name + ".lqA"], result[name + ".lqAs"]
            qB, sB = result[name + ".lqB"], result[name + ".lqBs"]
            A = qA.float() * sA.float().view(-1, 1)
            B = qB.float() * sB.float().view(-1, 1)
            w = w + A @ B
        # Inverse Hadamard rotation: W_orig ≈ W_rot @ Q
        if "+hadamard" in str(info) and w.ndim == 2:
            w = w @ hadamard(w.shape[1])
        out[name] = w.to(orig_dtype)
    return out


_BSHF_MAGIC = b"BSHF"


def _byte_shuffle(data: bytes, stride: int = 2) -> bytes:
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


def _byte_unshuffle(data: bytes) -> bytes:
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


def _compress(data: bytes, compressor: str) -> bytes:
    data = _byte_shuffle(data)
    if compressor == "lzma":
        return lzma.compress(data, preset=6)
    elif compressor == "brotli":
        import brotli
        return brotli.compress(data, quality=11)
    raise ValueError(f"Unknown compressor: {compressor!r}")


def _decompress(data: bytes, compressor: str) -> bytes:
    if compressor == "lzma":
        raw = lzma.decompress(data)
    elif compressor == "brotli":
        import brotli
        raw = brotli.decompress(data)
    else:
        raise ValueError(f"Unknown compressor: {compressor!r}")
    raw = _byte_unshuffle(raw)
    return raw


def serialize(h: Hyperparameters, base_model: torch.nn.Module, code: str) -> tuple[int, int]:
    code_bytes = len(code.encode("utf-8"))
    if h.is_main_process:
        torch.save(base_model.state_dict(), h.model_path)
        model_bytes = os.path.getsize(h.model_path)
        log(f"Serialized model: {model_bytes} bytes")
        log(f"Code size: {code_bytes} bytes")

    sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    device = torch.device("cuda", h.local_rank)
    log("GPTQ:collecting Hessians from calibration data...")
    t0 = time.perf_counter()
    calib_loader = ShuffledSequenceLoader(h, device)
    hessians = collect_hessians(
        base_model, calib_loader, h, device,
        n_calibration_batches=h.gptq_calibration_batches,
    )
    log(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter() - t0:.1f}s")
    quant_result, quant_meta = gptq_mixed_quantize(sd_cpu, hessians, h)

    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = _compress(quant_raw, h.compressor)
    quant_file_bytes = len(quant_blob)
    bytes_total = quant_file_bytes + code_bytes
    if h.is_main_process:
        with open(h.quantized_model_path, "wb") as f:
            f.write(quant_blob)
        log(f"Serialized model quantized+{h.compressor}: {quant_file_bytes} bytes")
        log(f"Total submission size quantized+{h.compressor}: {bytes_total} bytes")
    return bytes_total, quant_file_bytes


def deserialize(h: Hyperparameters, device: torch.device) -> GPT:
    eval_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(eval_model)
    sd_cpu = {k: v.detach().cpu() for k, v in eval_model.state_dict().items()}

    with open(h.quantized_model_path, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(_decompress(quant_blob_disk, h.compressor)),
        map_location="cpu",
    )
    deq_state = dequantize_mixed(quant_state["w"], quant_state["m"], sd_cpu)
    eval_model.load_state_dict(deq_state, strict=True)

    return eval_model


def _loss_bpb(loss_sum, token_count, byte_count) -> tuple[float, float]:
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return val_loss, val_bpb


def eval_val(
    h: Hyperparameters,
    device: torch.device,
    val_data: ValidationData,
    model: nn.Module
) -> tuple[float, float]:
    seq_len = h.eval_seq_len
    local_batch_tokens = h.val_batch_tokens // (h.world_size * h.grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={h.val_batch_tokens}, WORLD_SIZE={h.world_size}, "
            f"GRAD_ACCUM_STEPS={h.grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_data.val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * h.rank) // h.world_size
    seq_end = (total_seqs * (h.rank + 1)) // h.world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_data.val_tokens[raw_start:raw_end].to(
                device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (val_data.has_leading_space_lut[tgt_ids] &
                            ~val_data.is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    model.train()
    return _loss_bpb(val_loss_sum, val_token_count, val_byte_count)


def eval_val_sliding(
    h: Hyperparameters,
    device: torch.device,
    val_data: ValidationData,
    base_model: nn.Module,
    batch_seqs: int = 32
) -> tuple[float, float]:
    base_model.eval()
    logits_fn = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)

    seq_len = h.eval_seq_len
    context_size = seq_len - h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1

    window_starts = [ws for ws in range(0, total_tokens, h.eval_stride)
                     if ws + context_size < total_tokens]

    total_windows = len(window_starts)
    my_s = (total_windows * h.rank) // h.world_size
    my_e = (total_windows * (h.rank + 1)) // h.world_size
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
            wlens: list[int] = []

            for i, ws in enumerate(batch_ws):
                we = min(ws + seq_len, total_tokens)
                wlen = we - ws
                wlens.append(wlen)
                chunk = val_data.val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = logits_fn(x_batch)

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
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
                tb += (val_data.has_leading_space_lut[tgt] &
                       ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    base_model.train()
    return _loss_bpb(loss_sum, token_count, byte_count)


# ============================================================================
# Phased TTT + LoRA — ported from upstream/records/2026-04-27 leader.
# Module-level globals/helpers required by the phased eval and global SGD paths.
# ============================================================================

# BOS token id used to detect document boundaries in the validation stream.
# Lazily resolved to the tokenizer's bos_id() when first needed (default 1
# matches the leader's tokenizer convention if resolution fails).
BOS_ID = None


class BatchedLinearLoRA(nn.Module):
    # PR-1767: rank-scaled output (alpha/rank), like standard LoRA. Decouples
    # effective magnitude from rank so changing rank does not change LR scale.
    _ALPHA = float(os.environ.get("TTT_LORA_ALPHA", "144"))
    # PR-1767: optionally keep A warm across per-doc resets (only B is zeroed).
    # Accumulates useful feature directions across documents within a TTT phase.
    _WARM_START_A = bool(int(os.environ.get("TTT_WARM_START_A", "1")))

    def __init__(self, bsz, in_features, out_features, rank):
        super().__init__()
        self._bound = 1.0 / math.sqrt(in_features)
        self._scale = self._ALPHA / rank
        self.A = nn.Parameter(
            torch.empty(bsz, rank, in_features).uniform_(-self._bound, self._bound)
        )
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))

    def reset(self):
        with torch.no_grad():
            if not self._WARM_START_A:
                self.A.uniform_(-self._bound, self._bound)
            self.B.zero_()

    def forward(self, x):
        return ((x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)) * self._scale


class BatchedTTTLoRA(nn.Module):
    def __init__(self, bsz, model, rank, k_lora=True, mlp_lora=True, o_lora=True):
        super().__init__()
        self.bsz = bsz
        # TARGET adaptation: SOURCE used model.qo_bank.shape[-1]; TARGET stores
        # the model dim on each block's CausalSelfAttention. dim is the attn
        # input/output dim (== model.tok_emb.embedding_dim if no embed_proj,
        # else the model_dim post-projection).
        attn0 = model.blocks[0].attn
        dim = attn0.num_heads * attn0.head_dim
        vocab = model.tok_emb.num_embeddings
        if getattr(model, "looping_active", False):
            num_slots = len(model.encoder_indices) + len(model.decoder_indices)
        else:
            num_slots = len(model.blocks)
        kv_dim = attn0.num_kv_heads * attn0.head_dim
        # MLP up-projection out_features on TARGET = int(mlp_mult * dim).
        mlp_hidden = model.blocks[0].mlp.fc.out_features
        # LM-head LoRA: maps post-head_proj features (embedding_dim) to vocab.
        embed_dim = model.tok_emb.embedding_dim
        self.lm_head_lora = BatchedLinearLoRA(bsz, embed_dim, vocab, rank)
        self.q_loras = nn.ModuleList(
            [BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(num_slots)]
        )
        self.v_loras = nn.ModuleList(
            [BatchedLinearLoRA(bsz, dim, kv_dim, rank) for _ in range(num_slots)]
        )
        self.k_loras = (
            nn.ModuleList(
                [BatchedLinearLoRA(bsz, dim, kv_dim, rank) for _ in range(num_slots)]
            )
            if k_lora
            else None
        )
        self.mlp_loras = (
            nn.ModuleList(
                [BatchedLinearLoRA(bsz, dim, mlp_hidden, rank) for _ in range(num_slots)]
            )
            if mlp_lora
            else None
        )
        self.o_loras = (
            nn.ModuleList(
                [BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(num_slots)]
            )
            if o_lora
            else None
        )

    def reset(self):
        with torch.no_grad():
            self.lm_head_lora.reset()
            for loras in [self.q_loras, self.v_loras, self.k_loras,
                          self.mlp_loras, self.o_loras]:
                if loras is not None:
                    for lora in loras:
                        lora.reset()


def _resolve_bos_id(h: 'Hyperparameters') -> int:
    """Return the tokenizer's bos_id, with a sensible fallback."""
    global BOS_ID
    if BOS_ID is not None:
        return BOS_ID
    try:
        sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        bid = int(sp.bos_id())
        if bid < 0:
            bid = 1
    except Exception:
        bid = 1
    BOS_ID = bid
    return BOS_ID


def _build_cu_seqlens(bos_pos, total_len, device, max_doc_len=0, bucket_size=64):
    """Build cu_seqlens for FlashAttention varlen, splitting at BOS positions.
    Optionally further splits any segment longer than max_doc_len. Pads the
    output up to a multiple of bucket_size with sentinel total_len entries so
    CUDA-graph captures get a consistent shape."""
    if not bos_pos or bos_pos[0] != 0:
        bos_pos = [0] + bos_pos
    seg_starts = []
    starts_with_end = bos_pos + [total_len]
    for i in range(len(starts_with_end) - 1):
        start = starts_with_end[i]
        end = starts_with_end[i + 1]
        if max_doc_len > 0:
            pos = start
            while pos < end:
                seg_starts.append(pos)
                pos += max_doc_len
        else:
            seg_starts.append(start)
    boundaries = seg_starts + [total_len]
    padded_len = ((len(boundaries) + bucket_size - 1) // bucket_size) * bucket_size
    cu = torch.full((padded_len,), total_len, dtype=torch.int32, device=device)
    cu[: len(boundaries)] = torch.tensor(boundaries, dtype=torch.int32, device=device)
    seg_ends = seg_starts[1:] + [total_len]
    max_seqlen = max(end - start for start, end in zip(seg_starts, seg_ends))
    return cu, max_seqlen


def _find_docs(all_tokens):
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
    docs = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = (
            int(bos_positions[i + 1])
            if i + 1 < len(bos_positions)
            else all_tokens.numel()
        )
        if i + 1 < len(bos_positions):
            end += 1
        assert end - start >= 2
        docs.append((start, end - start))
    return docs


def _build_ttt_global_batches(doc_entries, h, ascending=False):
    batch_size = h.ttt_batch_size
    global_doc_entries = sorted(doc_entries, key=lambda x: x[1][1])
    global_batches = [
        global_doc_entries[i : i + batch_size]
        for i in range(0, len(global_doc_entries), batch_size)
    ]
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


def _accumulate_bpb(
    ptl,
    x,
    y,
    chunk_offsets,
    chunk_lens,
    pos_idx,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_token_lut,
    loss_sum,
    byte_sum,
    token_count,
    y_bytes=None,
):
    pos = pos_idx[: x.size(1)].unsqueeze(0)
    mask = (
        (chunk_lens.unsqueeze(1) > 0)
        & (pos >= chunk_offsets.unsqueeze(1))
        & (pos < (chunk_offsets + chunk_lens).unsqueeze(1))
    )
    mask_f64 = mask.to(torch.float64)
    if y_bytes is not None:
        tok_bytes = y_bytes.to(torch.float64)
    else:
        tok_bytes = base_bytes_lut[y].to(torch.float64)
        tok_bytes += (has_leading_space_lut[y] & ~is_boundary_token_lut[x]).to(
            torch.float64
        )
    loss_sum += (ptl.to(torch.float64) * mask_f64).sum()
    byte_sum += (tok_bytes * mask_f64).sum()
    token_count += chunk_lens.to(torch.float64).sum()


def _loss_bpb_from_sums(loss_sum, token_count, byte_sum):
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_sum.item())
    return val_loss, val_bpb


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


def _init_int64_counter(path):
    with open(path, "wb") as f:
        f.write((0).to_bytes(8, "little", signed=True))


def _select_ttt_doc_entries(docs, h):
    doc_entries = list(enumerate(docs))
    if h.val_doc_fraction < 1.0:
        sample_n = max(1, int(round(len(docs) * h.val_doc_fraction)))
        sampled_indices = sorted(
            random.Random(h.seed).sample(range(len(docs)), sample_n)
        )
        return [(i, docs[i]) for i in sampled_indices]
    return doc_entries


def train_val_ttt_global_sgd_distributed(h, device, val_data, base_model, val_tokens, batch_seqs=None):
    """Global SGD pass on the base model over the prefix scored so far. When
    GLOBAL_TTT_RESPECT_DOC_BOUNDARIES is enabled, builds cu_seqlens from BOS
    positions in each chunk and feeds the flat sequence as (1, T) through
    base_model with FlashAttention varlen so attention does not cross document
    boundaries. Otherwise reshapes into fixed-length (B, seq_len) sequences."""
    global BOS_ID
    if BOS_ID is None:
        BOS_ID = _resolve_bos_id(h)
    base_model.eval()
    seq_len = h.eval_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = h.global_ttt_chunk_tokens
    batch_seqs = h.global_ttt_batch_seqs if batch_seqs is None else batch_seqs
    if total_tokens <= 0:
        return
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    ttt_params = [p for p in base_model.parameters()]
    for p in ttt_params:
        p.requires_grad_(True)
    optimizer = torch.optim.SGD(
        ttt_params, lr=h.global_ttt_lr, momentum=h.global_ttt_momentum
    )
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
            lr_now = (
                h.global_ttt_warmup_start_lr
                + (h.global_ttt_lr - h.global_ttt_warmup_start_lr) * warmup_t
            )
        else:
            decay_steps = max(num_chunks - 1 - warmup_chunks, 1)
            decay_ci = max(ci - warmup_chunks, 0)
            lr_now = h.global_ttt_lr * 0.5 * (
                1.0 + math.cos(math.pi * decay_ci / decay_steps)
            )
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
                        if h.global_ttt_respect_doc_boundaries:
                            bos_pos = (x_flat == BOS_ID).nonzero(as_tuple=True)[0].tolist()
                            cu_seqlens, max_seqlen = _build_cu_seqlens(
                                bos_pos, x_flat.numel(), x_flat.device, h.eval_seq_len, 64
                            )
                            loss = base_model(
                                x_flat[None],
                                y_flat[None],
                                cu_seqlens=cu_seqlens,
                                max_seqlen=max_seqlen,
                            )
                        else:
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
            log(
                f"tttg: c{ci+1}/{num_chunks} lr:{lr_now:.6f} t:{elapsed:.1f}s"
            )
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()


def eval_val_ttt_phased(h, base_model, device, val_data, forward_ttt_train):
    global BOS_ID
    if BOS_ID is None:
        BOS_ID = _resolve_bos_id(h)
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
    log(
        "ttt_phased:"
        f" total_docs:{len(doc_entries)} prefix_docs:{prefix_doc_limit} "
        f"suffix_docs:{len(doc_entries) - prefix_doc_limit}"
        f" num_phases:{num_phases} boundaries:{phase_boundaries}"
    )
    chunk_size, eval_seq_len = h.ttt_chunk_size, h.ttt_eval_seq_len
    eval_batch_set = None
    if h.ttt_eval_batches:
        eval_batch_set = set(int(x) for x in h.ttt_eval_batches.split(",") if x.strip())
    use_ascending = eval_batch_set is not None
    global_batches_sorted = _build_ttt_global_batches(
        doc_entries, h, ascending=use_ascending
    )
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
            return torch.optim.SGD(
                lora.parameters(), lr=h.ttt_lora_lr,
                momentum=h.ttt_beta1, weight_decay=h.ttt_weight_decay,
            )
        return torch.optim.AdamW(
            lora.parameters(), lr=h.ttt_lora_lr,
            betas=(h.ttt_beta1, h.ttt_beta2),
            eps=1e-10, weight_decay=h.ttt_weight_decay, fused=True,
        )

    reusable_opt = _build_opt(reusable_lora)
    local_scored_docs = []
    global_ttt_done = prefix_doc_limit == 0
    try:
      while True:
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
                    ci, pred_lens[b], num_chunks[b], chunk_size, eval_seq_len
                )
                tok_starts[b] = doc_start + win_start
                tok_wls[b] = win_len
                chunk_offsets_cpu[b] = chunk_offset
                chunk_lens_cpu[b] = chunk_len
            _, context_size, chunk_offset, _ = _compute_chunk_window(
                ci, (ci + 1) * chunk_size, ci + 1, chunk_size, eval_seq_len
            )
            col_idx = torch.arange(context_size + 1)
            idx = tok_starts.unsqueeze(1) + col_idx.unsqueeze(0)
            idx.clamp_(max=all_tokens.numel() - 1)
            gathered_gpu = all_tokens_idx[idx].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            valid = (col_idx[:context_size].unsqueeze(0) < tok_wls.unsqueeze(1)).to(
                device, non_blocking=True
            )
            chunk_offsets = chunk_offsets_cpu.to(device, non_blocking=True)
            chunk_lens = chunk_lens_cpu.to(device, non_blocking=True)
            x = torch.where(valid, gathered_gpu[:, :context_size], 0)
            y = torch.where(valid, gathered_gpu[:, 1 : context_size + 1], 0)
            ctx_pos = torch.arange(context_size, device=device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                per_tok_loss = forward_ttt_train(x, y, lora=cur_lora)
            # TARGET adaptation: caseops sidecar (val_data.val_bytes) is not
            # implemented in this codebase. Always use the base_bytes_lut path.
            y_bytes_arg = None
            with torch.no_grad():
                _accumulate_bpb(
                    per_tok_loss,
                    x,
                    y,
                    chunk_offsets,
                    chunk_lens,
                    ctx_pos,
                    val_data.base_bytes_lut,
                    val_data.has_leading_space_lut,
                    val_data.is_boundary_token_lut,
                    loss_sum,
                    byte_sum,
                    token_count,
                    y_bytes=y_bytes_arg,
                )
            if needs_train:
                activate_chunk_mask = (num_chunks_t - 1 > ci).float()
                for gi in range(h.ttt_grad_steps):
                    if gi > 0:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            per_tok_loss = forward_ttt_train(x, y, lora=cur_lora)
                    per_doc = per_tok_loss[
                        :, chunk_offset : chunk_offset + chunk_size
                    ].mean(dim=-1)
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
            log(
                f"ttp: b{batch_num}/{queue_len} bl:{b_loss:.4f} bb:{b_bpb:.4f} "
                f"rl:{r_loss:.4f} rb:{r_bpb:.4f} dl:{min(doc_lens)}-{max(doc_lens)} "
                f"gd:{int(global_ttt_done)}"
            )
        if not global_ttt_done:
            local_scored_docs.extend(
                (orig_batch_idx, pos, doc_start, doc_len)
                for pos, (doc_start, doc_len) in enumerate(batch)
            )
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
                    for _, _, doc_start, doc_len in scored_docs_for_global
                ]
                if scored_token_chunks:
                    global_ttt_tokens = torch.cat(scored_token_chunks)
                else:
                    global_ttt_tokens = val_data.val_tokens[:0]
                if h.rank == 0:
                    prefix_done = 0
                    try:
                        with open(prefix_counter_path, "rb") as f:
                            prefix_done = int.from_bytes(
                                f.read(8), "little", signed=True
                            )
                    except FileNotFoundError:
                        pass
                    log(
                        f"ttpp: phase:{current_phase + 1}/{num_phases} pd:{prefix_done} "
                        f"gd:{len(scored_docs_for_global)} "
                        f"t:{time.perf_counter() - t_start:.1f}s"
                    )
                train_val_ttt_global_sgd_distributed(
                    h, device, val_data, base_model, global_ttt_tokens
                )
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


def eval_val_ttt(
    h: Hyperparameters,
    device: torch.device,
    val_data: ValidationData,
    eval_model: nn.Module,
) -> tuple[float, float]:
    # Legal score-first TTT: chunk-wise, score-before-adapt. Per chunk:
    #   1. Score under inference_mode (contributes to reported BPB).
    #   2. Adapt: TTT_EPOCHS full-batch SGD steps on the same chunk.
    # Causality: chunk N's score reflects only adaptations 0..N-1.
    # Replicated across ranks (same val + init + RNG -> identical trajectories).
    seq_len = h.eval_seq_len
    chunk_tokens = h.ttt_chunk_tokens
    total_tokens = val_data.val_tokens.numel() - 1

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    trainable_params = [p for p in eval_model.parameters() if p.requires_grad]
    sgd = torch.optim.SGD(trainable_params, lr=h.ttt_lr, momentum=h.ttt_momentum)

    for chunk_start in range(0, total_tokens, chunk_tokens):
        chunk_end = min(chunk_start + chunk_tokens, total_tokens)
        chunk = val_data.val_tokens[chunk_start:chunk_end + 1].to(
            dtype=torch.int64, device=device)
        n_seqs = (chunk.numel() - 1) // seq_len
        if n_seqs == 0:
            continue
        x_seqs = chunk[:n_seqs * seq_len].reshape(n_seqs, seq_len)
        y_seqs = chunk[1:n_seqs * seq_len + 1].reshape(n_seqs, seq_len)

        # SCORE
        eval_model.eval()
        with torch.inference_mode():
            assert torch.is_inference_mode_enabled()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for i in range(n_seqs):
                    loss = eval_model(x_seqs[i:i+1], y_seqs[i:i+1])
                    n_tok = float(seq_len)
                    loss_sum += loss.to(torch.float64) * n_tok
                    token_count += n_tok
                    tgt = y_seqs[i]
                    prev = x_seqs[i]
                    tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                    tb += (val_data.has_leading_space_lut[tgt] &
                           ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()

        # ADAPT
        assert not torch.is_inference_mode_enabled()
        eval_model.train()
        for _ in range(h.ttt_epochs):
            sgd.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = eval_model(x_seqs, y_seqs)
            loss.backward()
            sgd.step()

    if dist.is_available() and dist.is_initialized():
        # All ranks ran identical trajectories on identical data; AVG handles
        # any tiny numerical drift between ranks without changing magnitude.
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM); loss_sum /= h.world_size
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM); token_count /= h.world_size
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM); byte_count /= h.world_size

    eval_model.eval()
    return _loss_bpb(loss_sum, token_count, byte_count)


def timed_eval(label: str, fn, *args, **kwargs) -> tuple[float, float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    val_loss, val_bpb = fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    log(f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms")
    return val_loss, val_bpb


def train_model(h: Hyperparameters, device: torch.device, val_data: ValidationData):
    train_loader = ShuffledSequenceLoader(h, device)
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    # CUDA_GRAPHS=1 requires torch.compiler.cudagraph_mark_step_begin() at the top
    # of step_fn (see below). Without it cudagraph_trees raises "Gradient addition
    # node due to multiple use of tensor" because the tied-embedding weight is read
    # at both tok_emb(input_ids) and F.linear(x, tok_emb.weight). The mark_step call
    # tells the cudagraph manager that a fresh step has begun and prior aliases are
    # safe to overwrite.
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True,
        mode="reduce-overhead" if h.cuda_graphs else "default")
    if h.distributed:
        model = DDP(compiled_model, device_ids=[h.local_rank], broadcast_buffers=False)
    else:
        model = compiled_model
    log(f"model_params:{sum(p.numel() for p in base_model.parameters())}")
    optimizers = Optimizers(h, base_model)

    # Helper functions for training
    max_wallclock_ms = 1000.0 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    if max_wallclock_ms is not None:
        max_wallclock_ms -= h.gptq_reserve_seconds * 1000.0
        log(f"gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms")

    def training_frac(step: int, elapsed_ms: float) -> float:
        if max_wallclock_ms is None:
            return step / max(h.iterations, 1)
        return elapsed_ms / max(max_wallclock_ms, 1e-9)

    def lr_mul(frac: float) -> float:
        if h.warmdown_frac <= 0:
            return 1.0
        if frac >= 1.0 - h.warmdown_frac:
            return max((1.0 - frac) / h.warmdown_frac, h.min_lr)
        return 1.0

    def update_prores_alphas(frac: float) -> None:
        if not h.prores_enabled:
            return
        for block in base_model.blocks:
            block.prores_alpha.fill_(block.prores_alpha_value(frac))

    def step_fn(step, lr_scale):
        if h.cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        optimizers.zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(h.grad_accum_steps):
            if h.distributed:
                model.require_backward_grad_sync = micro_step == h.grad_accum_steps - 1
            x, y = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss / h.grad_accum_steps).backward()
        train_loss /= h.grad_accum_steps

        frac = min(step / h.muon_momentum_warmup_steps, 1.0) if h.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * h.muon_momentum_warmup_start + frac * h.muon_momentum
        for group in optimizers.optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * lr_scale

        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)

        optimizers.step()
        return train_loss

    if h.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone()
                               for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(h.warmup_steps):
            step_fn(warmup_step, 1.0)
            if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                log(f"warmup_step: {warmup_step + 1}/{h.warmup_steps}")
        if h.num_loops > 0:
            base_model.looping_active = True
            log(f"loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")
            for warmup_step in range(h.warmup_steps):
                step_fn(warmup_step, 1.0)
                if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                    log(f"loop_warmup_step: {warmup_step + 1}/{h.warmup_steps}")
            base_model.looping_active = False
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        optimizers.zero_grad_all()
        if h.distributed:
            model.require_backward_grad_sync = True
        train_loader = ShuffledSequenceLoader(h, device)

    ema_state = {n: t.detach().float().clone() for n, t in base_model.state_dict().items()}

    if h.profile_steps > 0:
        from torch.profiler import profile, ProfilerActivity
        is_main = (not h.distributed) or dist.get_rank() == 0
        for ps in range(3): step_fn(ps, 1.0)  # warm compile/allocator before capture
        torch.cuda.synchronize()
        if is_main: log(f"profile:capturing {h.profile_steps} steps")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for ps in range(h.profile_steps): step_fn(ps, 1.0)
            torch.cuda.synchronize()
        if is_main:
            log("profile:by_self_cuda\n" + prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
        if h.distributed: dist.barrier()
        sys.exit(0)

    # Start wallclock accounting from phase-1 elapsed so LR schedule continues across phases.
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == h.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (h.val_loss_every > 0 and step % h.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(h, device, val_data, model)
            log(f"{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < h.iterations:
                log(
                    f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms "
                    f"step: {step}/{h.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        frac = training_frac(step, elapsed_ms)
        scale = lr_mul(frac)
        update_prores_alphas(frac)
        if h.num_loops > 0 and not base_model.looping_active and frac >= h.enable_looping_at:
            base_model.looping_active = True
            log(f"layer_loop:enabled step:{step} frac:{frac:.3f} encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")
        train_loss = step_fn(step, scale)

        # EMA update every step
        with torch.no_grad():
            for n, p in base_model.state_dict().items():
                ema_state[n].lerp_(p.float(), 1.0 - h.ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        should_log_train = (
            h.train_log_every > 0
            and (step <= 5 or step % h.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            tok_per_sec = step * h.train_batch_tokens / (approx_training_time_ms / 1000.0)
            log(
                f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} "
                f"train_time: {approx_training_time_ms / 60000:.1f}m tok/s: {tok_per_sec:.0f}"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if h.distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # Apply EMA weights
    log(f"ema:applying EMA weights (decay={h.ema_decay})")
    current_state = base_model.state_dict()
    for n, t in ema_state.items():
        current_state[n].copy_(t.to(dtype=current_state[n].dtype))
    base_model.load_state_dict(current_state, strict=True)

    return base_model, compiled_model


def train_and_eval(h: Hyperparameters, device: torch.device) -> None:
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)

    val_data = ValidationData(h, device)
    log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}")
    log(f"val_tokens: {val_data.val_tokens.numel() - 1}")

    base_model, compiled_model = train_model(h, device, val_data)
    torch._dynamo.reset()
    timed_eval("pre-quantization post-ema", eval_val, h, device, val_data, compiled_model)

    serialize(h, base_model, Path(__file__).read_text(encoding="utf-8"))
    if h.distributed:
        dist.barrier()
    eval_model = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True

    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    final_loss, final_bpb = timed_eval("quantized", eval_val, h, device, val_data, compiled_model)
    quick_probe = h.iterations <= 50
    if h.sliding_window_enabled and not quick_probe:
        final_loss, final_bpb = timed_eval("quantized_sliding_window", eval_val_sliding, h, device, val_data, eval_model)
    elif h.sliding_window_enabled:
        log("quantized_sliding_window:skipped quick_probe=1")
    if h.phased_ttt_enabled and not quick_probe:
        # Build a torch.compile'd wrapper around the per-token-loss forward path
        # used by the phased eval loop. Mirrors upstream leader's _fwd_ttt wiring.
        ttt_model = eval_model
        for p in ttt_model.parameters():
            p.requires_grad_(False)

        def _fwd_ttt_inner(input_ids, target_ids, lora):
            return ttt_model.forward_ttt(input_ids, target_ids, lora=lora)

        _fwd_ttt_compiled_inner = None

        def _fwd_ttt(input_ids, target_ids, lora):
            nonlocal _fwd_ttt_compiled_inner
            if _fwd_ttt_compiled_inner is None:
                _fwd_ttt_compiled_inner = torch.compile(_fwd_ttt_inner, dynamic=True)
            return _fwd_ttt_compiled_inner(input_ids, target_ids, lora=lora)

        log("ttt_lora:warming up compile (random tokens, no val data)")
        _resolve_bos_id(h)
        t_warmup = time.perf_counter()
        for bsz_w in [h.ttt_batch_size]:
            wl = BatchedTTTLoRA(
                bsz_w, ttt_model, h.ttt_lora_rank,
                k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
            ).to(device)
            wo = torch.optim.AdamW(
                wl.parameters(), lr=h.ttt_lora_lr,
                betas=(h.ttt_beta1, h.ttt_beta2), eps=1e-10,
                weight_decay=h.ttt_weight_decay, fused=True,
            )
            for ctx_len in (h.ttt_chunk_size, h.ttt_eval_seq_len):
                xw = torch.randint(0, h.vocab_size, (bsz_w, ctx_len),
                                   device=device, dtype=torch.int64)
                yw = torch.randint(0, h.vocab_size, (bsz_w, ctx_len),
                                   device=device, dtype=torch.int64)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = _fwd_ttt(xw, yw, lora=wl)
                ptl[:, : min(h.ttt_chunk_size, ctx_len)].mean(dim=-1).sum().backward()
                wo.step()
                wo.zero_grad(set_to_none=True)
            del wl, wo
        torch.cuda.empty_cache()
        log(f"ttt_lora:compile warmup done ({time.perf_counter() - t_warmup:.1f}s)")
        final_loss, final_bpb = timed_eval(
            "quantized_phased_ttt", eval_val_ttt_phased,
            h, ttt_model, device, val_data, _fwd_ttt,
        )
    elif h.phased_ttt_enabled:
        log("quantized_phased_ttt:skipped quick_probe=1")
    elif h.ttt_enabled and not quick_probe:
        final_loss, final_bpb = timed_eval("quantized_ttt", eval_val_ttt, h, device, val_data, eval_model)
    elif h.ttt_enabled:
        log("quantized_ttt:skipped quick_probe=1")

    # Log experiment result
    if h.is_main_process:
        import json as _json
        _rec = {"id": h.run_id, "seed": h.seed, "num_layers": h.num_layers,
                "model_dim": h.model_dim, "vocab_size": h.vocab_size,
                "val_bpb": round(final_bpb, 6), "val_loss": round(final_loss, 6)}
        with open("experiments.jsonl", "a") as _f:
            _f.write(_json.dumps(_rec) + "\n")
        log(f"experiment_logged:{_rec['id']}")


def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False

    h = Hyperparameters()
    set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs("logs", exist_ok=True)
        log(100 * "=", console=False)
        log("Hyperparameters:", console=True)
        for k, v in sorted(vars(type(h)).items()):
            if not k.startswith("_"):
                log(f"  {k}: {v}", console=True)
        log("=" * 100, console=False)
        log(f"Running Python {sys.version}", console=False)
        log(f"Running PyTorch {torch.__version__}", console=False)
        log(
            subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, check=False).stdout,
            console=False,
        )
        log("=" * 100, console=False)

    train_and_eval(h, device)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
